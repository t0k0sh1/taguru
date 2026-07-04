//! Disk-backed context registry: the server-side lifecycle around the
//! library's `Context`. Disk is the source of truth — every context is
//! one image file (`{name}.ctx`, the bytes of `Context::to_bytes`) plus
//! a sidecar `{name}.meta.json` holding the routing description, the
//! cache policy flag, and a stats snapshot. The sidecar stays outside
//! the image on purpose: the image format remains a pure dump of the
//! network, and server metadata can evolve without bumping it.
//!
//! Memory is a cache over that truth, managed at whole-context
//! granularity — access locality is per 文脈 (a session works one
//! context for many queries), and a whole image loads in low
//! milliseconds. Contexts are registered cold at boot and loaded on
//! first touch; when the resident estimate of unpinned hot contexts
//! exceeds the cache budget, the least recently used are flushed and
//! dropped. Pinned contexts (glossaries and other always-hot 文脈)
//! load at boot, never count against the budget, and are never evicted.
//!
//! Durability: writes mark a context dirty; dirty contexts are
//! persisted by the periodic flusher, on eviction, and on graceful
//! shutdown. A crash can therefore lose at most the writes since the
//! last flush — the accepted window until an operation WAL is needed.
//! Creation and metadata changes persist immediately.
//!
//! Locking contract: the registry lock guards only the name → entry map
//! and is held just long enough to look up, insert, or remove; every
//! context sits behind its own entry lock. A caller clones the entry's
//! `Arc` and releases the registry immediately, so a slow operation on
//! one context never blocks the others — and a panic poisons only the
//! context it happened in.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use associative_rag::context::Context;
use serde::{Deserialize, Serialize};

use crate::embedding::{EmbeddingProvider, VectorStore, fnv1a, similarity};

/// Server-side metadata for one context: the prose half of the routing
/// directory plus the cache policy flag.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextMeta {
    /// What this 文脈 covers, written by whoever creates the context
    /// (typically the ingesting LLM). Routing quality depends on it, so
    /// the directory serves it next to stats that cannot go stale.
    pub description: String,
    /// Pinned contexts stay resident regardless of cache pressure — for
    /// small, always-hot contexts like glossaries.
    pub pinned: bool,
    /// Per-context fuzzy-entry floor for resolve; `None` means the
    /// library default (0.3). Lower admits more distant near-miss
    /// spellings, higher keeps entry strict. Re-applied to the context
    /// on every load, since the image itself carries no config.
    pub dice_floor: Option<f64>,
    /// Per-context floor for the semantic entry tier (cosine over
    /// glosses); `None` means the calibrated default (0.35). Same
    /// tuning story as `dice_floor`: config lives in the sidecar, never
    /// in the image.
    pub semantic_floor: Option<f32>,
}

/// Mechanically derived "what is this context about" numbers. Served by
/// the routing directory so an LLM can pick a context without anything
/// being loaded or scanned per request; unlike the prose description,
/// these are recomputed from the network itself and cannot drift.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextStats {
    pub associations: usize,
    pub concepts: usize,
    pub labels: usize,
    pub sources: usize,
    pub footprint_bytes: usize,
    /// Most connected concepts with their degree, most connected first.
    pub top_concepts: Vec<(String, usize)>,
    /// The first labels of the relation vocabulary (capped; the full
    /// list is at `GET /contexts/{name}/labels`).
    pub label_sample: Vec<String>,
}

impl ContextStats {
    const TOP_CONCEPTS: usize = 10;
    const LABEL_SAMPLE: usize = 50;

    fn of(context: &Context) -> Self {
        Self {
            associations: context.association_count(),
            concepts: context.concept_count(),
            labels: context.label_count(),
            sources: context.source_count(),
            footprint_bytes: context.footprint(),
            top_concepts: context
                .top_concepts(Self::TOP_CONCEPTS)
                .into_iter()
                .map(|(name, degree)| (name.to_string(), degree))
                .collect(),
            label_sample: context
                .labels()
                .into_iter()
                .take(Self::LABEL_SAMPLE)
                .map(String::from)
                .collect(),
        }
    }
}

/// What `{name}.meta.json` holds: the meta inline plus the stats
/// snapshot as of the last save, so a directory listing can describe a
/// cold context without touching its image.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
struct MetaFile {
    #[serde(flatten)]
    meta: ContextMeta,
    stats: ContextStats,
}

/// One row of `GET /contexts` — the routing directory an LLM client
/// reads to decide which context to search, skills-style: a name, the
/// prose description, and the mechanical stats that keep it honest.
/// Stats are live for loaded contexts and the last saved snapshot for
/// cold ones.
#[derive(Debug, Clone, Serialize)]
pub struct DirectoryEntry {
    pub name: String,
    pub description: String,
    pub pinned: bool,
    pub loaded: bool,
    /// Per-context fuzzy-entry floor; null means the default (0.3).
    pub dice_floor: Option<f64>,
    /// Per-context semantic floor; null means the default (0.35).
    pub semantic_floor: Option<f32>,
    pub stats: ContextStats,
}

/// Whether a context's network is resident. Cold entries keep only
/// their metadata and stats snapshot in memory.
enum Slot {
    Hot(Box<Context>),
    Cold,
}

pub struct Entry {
    inner: RwLock<EntryInner>,
    /// Set on every write, cleared when the image is persisted. Only
    /// ever changed while `inner` is write-locked; the atomic just lets
    /// the flusher skip clean entries without locking them.
    dirty: AtomicBool,
    /// Logical timestamp of the last operation, for LRU eviction.
    last_touch: AtomicU64,
    /// The vector sidecar, held after first use so the semantic
    /// fallback never re-reads a many-megabyte file per query. Replaced
    /// by refresh, cleared by eviction, and counted against the cache
    /// budget. Lock order: `inner` before `vectors`, never the reverse.
    vectors: Mutex<Option<Arc<VectorStore>>>,
}

impl Entry {
    fn new(meta: ContextMeta, stats: ContextStats, slot: Slot) -> Self {
        Self {
            inner: RwLock::new(EntryInner { meta, stats, slot }),
            dirty: AtomicBool::new(false),
            last_touch: AtomicU64::new(0),
            vectors: Mutex::new(None),
        }
    }
}

struct EntryInner {
    meta: ContextMeta,
    stats: ContextStats,
    slot: Slot,
}

pub enum CreateError {
    AlreadyExists,
    Io(io::Error),
}

/// Why an operation on a named context could not run.
pub enum AccessError {
    NotFound,
    /// The context exists but its image could not be loaded from disk.
    Load(String),
}

/// Shared server state: the data directory, the cache budget, and the
/// context registry.
#[derive(Clone)]
pub struct AppState(Arc<StateInner>);

struct StateInner {
    data_dir: PathBuf,
    /// Resident-bytes budget for unpinned hot contexts, enforced after
    /// every operation by evicting least-recently-used contexts. The
    /// most recently used context is never evicted, so one context
    /// larger than the whole budget still works — it just stays alone.
    cache_bytes: usize,
    registry: RwLock<HashMap<String, Arc<Entry>>>,
    /// Logical clock behind `Entry::last_touch`.
    clock: AtomicU64,
    /// The optional semantic entry tier; `None` keeps resolve purely
    /// lexical.
    embedder: Option<Arc<dyn EmbeddingProvider>>,
    /// Process-lifetime cache of cue embeddings — an LLM client repeats
    /// query wording, and every hit saves a provider round trip on the
    /// fallback path. Valid for the whole process because the provider
    /// (and so the model) is fixed at boot.
    cue_cache: Mutex<CueCache>,
}

/// A FIFO-bounded map of cue → embedding. FIFO rather than LRU keeps it
/// trivial; at the cap it holds ~12 MB of vectors, and evicting a warm
/// cue merely costs one re-embed.
#[derive(Default)]
struct CueCache {
    vectors: HashMap<String, Arc<Vec<f32>>>,
    order: VecDeque<String>,
}

impl CueCache {
    const CAP: usize = 1024;

    fn get(&self, cue: &str) -> Option<Arc<Vec<f32>>> {
        self.vectors.get(cue).cloned()
    }

    fn insert(&mut self, cue: String, vector: Arc<Vec<f32>>) {
        if self.vectors.contains_key(&cue) {
            return;
        }
        if self.vectors.len() >= Self::CAP
            && let Some(oldest) = self.order.pop_front()
        {
            self.vectors.remove(&oldest);
        }
        self.order.push_back(cue.clone());
        self.vectors.insert(cue, vector);
    }
}

impl AppState {
    /// Opens (creating if needed) the data directory and registers every
    /// context image found in it — cold, described by their sidecar
    /// snapshots. Pinned contexts are loaded eagerly; a pinned image
    /// that fails to load is left cold with a warning rather than
    /// taking the server down.
    pub fn boot(
        data_dir: PathBuf,
        cache_bytes: usize,
        embedder: Option<Arc<dyn EmbeddingProvider>>,
    ) -> io::Result<Self> {
        fs::create_dir_all(&data_dir)?;
        let mut registry = HashMap::new();
        for dir_entry in fs::read_dir(&data_dir)? {
            let path = dir_entry?.path();
            if path.extension().and_then(|e| e.to_str()) != Some("ctx") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Some(name) = name_from_stem(stem) else {
                eprintln!("skipping {}: file name does not decode", path.display());
                continue;
            };
            let MetaFile { meta, stats } = read_meta_file(&data_dir, stem);
            registry.insert(name, Arc::new(Entry::new(meta, stats, Slot::Cold)));
        }

        let state = Self(Arc::new(StateInner {
            data_dir,
            cache_bytes,
            registry: RwLock::new(registry),
            clock: AtomicU64::new(0),
            embedder,
            cue_cache: Mutex::new(CueCache::default()),
        }));
        for (name, entry) in state.snapshot() {
            let mut inner = entry.inner.write().unwrap();
            if inner.meta.pinned
                && let Err(error) = ensure_hot(&state.0.data_dir, &name, &mut inner)
            {
                eprintln!("pinned context '{name}' not preloaded: {error}");
            }
        }
        Ok(state)
    }

    pub fn context_count(&self) -> usize {
        self.0.registry.read().unwrap().len()
    }

    /// Whether the semantic entry tier has a provider at all.
    pub fn embeddings_configured(&self) -> bool {
        self.0.embedder.is_some()
    }

    /// Registers an empty context and persists it immediately, so its
    /// existence (and description) survives a crash from the moment the
    /// create call returns. A persistence failure fails the create.
    pub fn create(&self, name: &str, meta: ContextMeta) -> Result<(), CreateError> {
        let mut registry = self.0.registry.write().unwrap();
        if registry.contains_key(name) {
            return Err(CreateError::AlreadyExists);
        }
        let mut context = Context::default();
        context.set_dice_floor(meta.dice_floor);
        let stats = ContextStats::of(&context);
        save_files(&self.0.data_dir, name, &meta, &stats, &context).map_err(CreateError::Io)?;
        registry.insert(
            name.to_string(),
            Arc::new(Entry::new(meta, stats, Slot::Hot(Box::new(context)))),
        );
        Ok(())
    }

    /// Removes a context from the registry and deletes its files. Waits
    /// for any in-flight operation on the entry (its lock is taken after
    /// removal), so a concurrent flush cannot recreate the files. Any
    /// unflushed writes are discarded — deletion destroys the context.
    pub fn delete(&self, name: &str) -> Option<io::Result<()>> {
        let entry = self.0.registry.write().unwrap().remove(name)?;
        let _in_flight = entry.inner.write().unwrap();
        let stem = file_stem(name);
        let mut outcome = Ok(());
        for file in [
            format!("{stem}.ctx"),
            format!("{stem}.meta.json"),
            format!("{stem}.sources.json"),
            format!("{stem}.vectors.bin"),
        ] {
            if let Err(error) = fs::remove_file(self.0.data_dir.join(file))
                && error.kind() != io::ErrorKind::NotFound
            {
                outcome = Err(error);
            }
        }
        Some(outcome)
    }

    /// Registers original text passages behind source ids, merge-upsert,
    /// persisted immediately. This is the server-side "storage of
    /// record" convenience the library deliberately does not have: the
    /// graph indexes knowledge and attributions carry opaque source ids;
    /// this store lets a client dereference those ids back to original
    /// wording — find with the graph, answer from the text. Passages are
    /// optional per source; nothing requires one to exist.
    pub fn store_passages(
        &self,
        name: &str,
        passages: BTreeMap<String, String>,
    ) -> Option<io::Result<usize>> {
        let entry = self.lookup(name)?;
        // The entry lock only serializes read-modify-write on the
        // passages file; the context itself is never loaded for this.
        let _guard = entry.inner.write().unwrap();
        let path = sources_path(&self.0.data_dir, &file_stem(name));
        let mut stored = read_passages(&path);
        let added = passages.len();
        stored.extend(passages);
        let outcome = serde_json::to_vec_pretty(&stored)
            .map_err(io::Error::from)
            .and_then(|bytes| write_atomic(&path, &bytes))
            .map(|()| added);
        Some(outcome)
    }

    /// Dereferences source ids (as found on attributions) back to their
    /// registered passages, reporting the ids that have none.
    pub fn lookup_passages(
        &self,
        name: &str,
        sources: &[String],
    ) -> Option<(BTreeMap<String, String>, Vec<String>)> {
        let entry = self.lookup(name)?;
        let _guard = entry.inner.read().unwrap();
        let stored = read_passages(&sources_path(&self.0.data_dir, &file_stem(name)));
        let mut passages = BTreeMap::new();
        let mut missing = Vec::new();
        for source in sources {
            match stored.get(source) {
                Some(text) => {
                    passages.insert(source.clone(), text.clone());
                }
                None => missing.push(source.clone()),
            }
        }
        Some((passages, missing))
    }

    /// The source ids that currently have a registered passage.
    pub fn passage_sources(&self, name: &str) -> Option<Vec<String>> {
        let entry = self.lookup(name)?;
        let _guard = entry.inner.read().unwrap();
        let stored = read_passages(&sources_path(&self.0.data_dir, &file_stem(name)));
        Some(stored.into_keys().collect())
    }

    /// Name pairs whose GLOSSES sit close in embedding space — the
    /// synonym-fork candidates (創業年 vs 設立年) that no spelling
    /// comparison can see. Works off the stored vector sidecar alone
    /// (no provider round trip), so it runs even when the provider is
    /// gone, and is skipped with a note when no vectors exist or a
    /// namespace is too large for the O(N²) sweep. Returns
    /// (concept_pairs, label_pairs, skipped_note).
    #[allow(clippy::type_complexity)]
    pub fn semantic_twins(
        &self,
        name: &str,
        cosine_floor: f32,
    ) -> Option<(
        Vec<(String, String, f32)>,
        Vec<(String, String, f32)>,
        Option<String>,
    )> {
        /// Past this many names a namespace's pairwise sweep is skipped.
        const SWEEP_CAP: usize = 2000;
        /// At most this many pairs per namespace come back.
        const PAIR_CAP: usize = 100;

        let entry = self.lookup(name)?;
        let floor = cosine_floor.clamp(0.0, 1.0);
        let store = self.entry_vectors(&entry, &file_stem(name));
        if store.concepts.is_empty() && store.labels.is_empty() {
            return Some((
                Vec::new(),
                Vec::new(),
                Some(
                    "ベクトル未生成のため意味的検出はスキップ (POST embeddings/refresh を実行)"
                        .to_string(),
                ),
            ));
        }

        let mut skipped = None;
        let sweep = |table: &HashMap<String, (u64, Vec<f32>)>,
                     skipped: &mut Option<String>|
         -> Vec<(String, String, f32)> {
            if table.len() > SWEEP_CAP {
                *skipped = Some(format!(
                    "語彙が {} 名を超えるためこの名前空間の意味的検出はスキップ",
                    SWEEP_CAP
                ));
                return Vec::new();
            }
            let entries: Vec<(&String, &Vec<f32>)> = {
                let mut entries: Vec<_> = table.iter().map(|(name, (_, v))| (name, v)).collect();
                entries.sort_by_key(|(name, _)| name.as_str());
                entries
            };
            let mut pairs = Vec::new();
            for (i, (name_a, vector_a)) in entries.iter().enumerate() {
                for (name_b, vector_b) in &entries[i + 1..] {
                    let score = similarity(vector_a, vector_b);
                    if score >= floor {
                        pairs.push(((*name_a).clone(), (*name_b).clone(), score));
                    }
                }
            }
            pairs.sort_by(|x, y| {
                y.2.total_cmp(&x.2)
                    .then_with(|| (&x.0, &x.1).cmp(&(&y.0, &y.1)))
            });
            pairs.truncate(PAIR_CAP);
            pairs
        };
        let mut concepts = sweep(&store.concepts, &mut skipped);
        let mut labels = sweep(&store.labels, &mut skipped);
        // Related is not duplicate: concepts joined by an edge and labels
        // co-used on one subject resemble each other BECAUSE they are
        // related (glosses quote shared facts), and would bury the real
        // fork candidates in noise. Filtering needs the graph, so the
        // context loads if cold — acceptable for an explicit audit.
        match self.read_context(name, |context| {
            concepts.retain(|(a, b, _)| !context.adjacent(a, b));
            labels.retain(|(a, b, _)| !context.labels_share_subject(a, b));
        }) {
            Ok(()) => {}
            Err(AccessError::NotFound) => return None,
            Err(AccessError::Load(message)) => {
                // Vectors were readable but the graph was not: serve the
                // unfiltered pairs and say why they are noisier.
                skipped = Some(format!(
                    "関連ペアの除外はスキップ (グラフ未ロード: {message})"
                ));
            }
        }
        Some((concepts, labels, skipped))
    }

    /// Withdraws one source from a context — its graph contributions and
    /// its registered passage — the per-document differential-sync move:
    /// retract the old version of a changed document, then re-ingest the
    /// new one, instead of rebuilding the whole context. Returns how
    /// many associations were touched and whether a passage was removed.
    pub fn retract_source(&self, name: &str, source: &str) -> Result<(usize, bool), AccessError> {
        let touched =
            self.write_context(name, |context| context.retract_source(source).unwrap_or(0))?;

        let Some(entry) = self.lookup(name) else {
            // Raced with a delete; there is nothing left to clean up.
            return Ok((touched, false));
        };
        let _guard = entry.inner.write().unwrap();
        let path = sources_path(&self.0.data_dir, &file_stem(name));
        let mut stored = read_passages(&path);
        let mut passage_removed = false;
        if stored.remove(source).is_some() {
            match serde_json::to_vec_pretty(&stored)
                .map_err(io::Error::from)
                .and_then(|bytes| write_atomic(&path, &bytes))
            {
                Ok(()) => passage_removed = true,
                Err(error) => {
                    eprintln!("passage for '{source}' not removed from disk: {error}");
                }
            }
        }
        Ok((touched, passage_removed))
    }

    /// Full-text search over the registered passages — the second lane
    /// beside the graph, for knowledge that does not decompose into
    /// triples (procedures, conditions, discourse). BM25 over the same
    /// normalization as the entry index with mixed terms (ASCII words,
    /// character bigrams elsewhere); scores are recomputed per query,
    /// which is fine at sidecar scale.
    pub fn search_passages(
        &self,
        name: &str,
        query: &str,
        limit: usize,
    ) -> Option<Vec<(String, f32, String)>> {
        const K1: f32 = 1.2;
        const B: f32 = 0.75;

        let entry = self.lookup(name)?;
        let stored = {
            let _guard = entry.inner.read().unwrap();
            read_passages(&sources_path(&self.0.data_dir, &file_stem(name)))
        };
        let query_grams: Vec<u64> = {
            let normalized = associative_rag::context::normalize_entry(query);
            let mut seen = std::collections::HashSet::new();
            text_terms(&normalized)
                .into_iter()
                .filter(|gram| seen.insert(*gram))
                .collect()
        };
        if stored.is_empty() || query_grams.is_empty() {
            return Some(Vec::new());
        }

        // Term statistics per passage, one normalization pass each.
        let passages: Vec<(&String, &String, HashMap<u64, f32>, f32)> = stored
            .iter()
            .map(|(source, text)| {
                let normalized = associative_rag::context::normalize_entry(text);
                let mut frequencies: HashMap<u64, f32> = HashMap::new();
                let mut length = 0f32;
                for gram in text_terms(&normalized) {
                    *frequencies.entry(gram).or_insert(0.0) += 1.0;
                    length += 1.0;
                }
                (source, text, frequencies, length)
            })
            .collect();
        let total = passages.len() as f32;
        let average_length =
            (passages.iter().map(|(.., length)| length).sum::<f32>() / total).max(1.0);

        let mut scored: Vec<(String, f32, String)> = passages
            .iter()
            .map(|(source, text, frequencies, length)| {
                let mut score = 0f32;
                for gram in &query_grams {
                    let Some(&frequency) = frequencies.get(gram) else {
                        continue;
                    };
                    let document_frequency = passages
                        .iter()
                        .filter(|(_, _, other, _)| other.contains_key(gram))
                        .count() as f32;
                    let idf = (1.0
                        + (total - document_frequency + 0.5) / (document_frequency + 0.5))
                        .ln();
                    score += idf * (frequency * (K1 + 1.0))
                        / (frequency + K1 * (1.0 - B + B * length / average_length));
                }
                ((*source).clone(), score, (*text).clone())
            })
            .filter(|&(_, score, _)| score > 0.0)
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(limit);
        Some(scored)
    }

    /// Embeds the GLOSS of every canonical concept and label — the name
    /// plus its heaviest facts — and persists the vector sidecar. Bare
    /// names carry too little signal for sentence-trained embedding
    /// models; the graph supplies the context itself. Each vector
    /// remembers the hash of the gloss it was computed from, so a
    /// refresh re-embeds exactly the names that are new or whose graph
    /// context changed. Explicit rather than automatic — an agent or
    /// operator calls this after ingesting, so embedding spend stays
    /// intentional. Returns (newly embedded, total vectors), or `None`
    /// for an unknown context.
    pub fn refresh_embeddings(&self, name: &str) -> Option<Result<(usize, usize), String>> {
        /// How many facts a concept gloss carries.
        const GLOSS_FACTS: usize = 4;
        /// How many example triples a label gloss carries.
        const GLOSS_EXAMPLES: usize = 3;

        let Some(embedder) = self.0.embedder.clone() else {
            return Some(Err(
                "no embedding provider is configured (set ARAG_EMBED_URL and ARAG_EMBED_MODEL)"
                    .to_string(),
            ));
        };
        let glosses = match self.read_context(name, |context| {
            let concepts: Vec<(String, String)> = context
                .concept_names()
                .into_iter()
                .map(|name| {
                    let gloss = context
                        .concept_gloss(name, GLOSS_FACTS)
                        .unwrap_or_else(|| name.to_string());
                    (name.to_string(), gloss)
                })
                .collect();
            let labels: Vec<(String, String)> = context
                .labels()
                .into_iter()
                .map(|name| {
                    let gloss = context
                        .label_gloss(name, GLOSS_EXAMPLES)
                        .unwrap_or_else(|| name.to_string());
                    (name.to_string(), gloss)
                })
                .collect();
            (concepts, labels)
        }) {
            Ok(glosses) => glosses,
            Err(AccessError::NotFound) => return None,
            Err(AccessError::Load(message)) => return Some(Err(message)),
        };
        let (concepts, labels) = glosses;
        let entry = self.lookup(name)?;
        let path = vectors_path(&self.0.data_dir, &file_stem(name));

        // Diff and embed without holding the entry lock — provider round
        // trips can take seconds. Concurrent refreshes at worst re-embed
        // the same names; the merge below stays correct.
        let existing = VectorStore::load(&path);
        let fresh_model = existing.model != embedder.model();
        let stale = |table: &HashMap<String, (u64, Vec<f32>)>,
                     entries: &[(String, String)]|
         -> Vec<(String, String, u64)> {
            entries
                .iter()
                .filter_map(|(name, gloss)| {
                    let hash = fnv1a(gloss);
                    let outdated =
                        fresh_model || table.get(name).is_none_or(|&(stored, _)| stored != hash);
                    outdated.then(|| (name.clone(), gloss.clone(), hash))
                })
                .collect()
        };
        let stale_concepts = stale(&existing.concepts, &concepts);
        let stale_labels = stale(&existing.labels, &labels);
        let mut embedded_concepts = Vec::new();
        let mut embedded_labels = Vec::new();
        for (stale, embedded) in [
            (&stale_concepts, &mut embedded_concepts),
            (&stale_labels, &mut embedded_labels),
        ] {
            for chunk in stale.chunks(128) {
                let texts: Vec<&str> = chunk.iter().map(|(_, gloss, _)| gloss.as_str()).collect();
                match embedder.embed(&texts) {
                    Ok(vectors) => embedded.extend(
                        chunk
                            .iter()
                            .zip(vectors)
                            .map(|((name, _, hash), vector)| (name.clone(), (*hash, vector))),
                    ),
                    Err(error) => return Some(Err(error)),
                }
            }
        }
        let newly_embedded = embedded_concepts.len() + embedded_labels.len();

        // Merge under the entry lock so concurrent refreshes serialize
        // on the read-modify-write of the sidecar.
        let _guard = entry.inner.write().unwrap();
        let mut store = VectorStore::load(&path);
        if store.model != embedder.model() {
            store = VectorStore {
                model: embedder.model().to_string(),
                ..Default::default()
            };
        }
        store.concepts.extend(embedded_concepts);
        store.labels.extend(embedded_labels);
        let total = store.concepts.len() + store.labels.len();
        if newly_embedded > 0
            && let Err(error) = store.save(&path)
        {
            return Some(Err(format!("vector store not persisted: {error}")));
        }
        // Publish the fresh store so queries never re-read the sidecar.
        *entry.vectors.lock().unwrap() = Some(Arc::new(store));
        Some(Ok((newly_embedded, total)))
    }

    /// The semantic fallback behind resolve: nearest stored names by
    /// cosine over the vector sidecar. Meant to run only after the
    /// lexical tiers found nothing; scores are cosine similarities — a
    /// different scale from lexical scores, which the API marks by tier.
    /// Empty when no provider is configured, no refresh has run, or the
    /// sidecar belongs to another model.
    pub fn semantic_resolve(
        &self,
        name: &str,
        cue: &str,
        labels: bool,
        floor_override: Option<f32>,
    ) -> Option<Result<Vec<(String, f32)>, String>> {
        // Calibrated against text-embedding-3-large with GLOSSED names
        // (name + graph context): true matches land at ~0.44–0.58 —
        // jargon paraphrases included (醸造責任者×杜氏 0.53, 質問形
        // 「酒造りの責任者は誰」0.58, アップル×りんご 0.45) — while the
        // noise band drops to ~0.17 (自動車×杜氏グロス 0.09,
        // 自動車×りんごグロス 0.17), far better separated than bare
        // names ever were. 0.35 admits the weakest true matches with
        // ~2× margin over noise.
        const SEMANTIC_FLOOR: f32 = 0.35;
        const SEMANTIC_LIMIT: usize = 5;

        let Some(embedder) = self.0.embedder.clone() else {
            return Some(Ok(Vec::new()));
        };
        let entry = self.lookup(name)?;
        // One-call override beats the context setting beats the default.
        let context_floor = entry.inner.read().unwrap().meta.semantic_floor;
        let floor = floor_override
            .or(context_floor)
            .unwrap_or(SEMANTIC_FLOOR)
            .clamp(0.0, 1.0);
        let store = self.entry_vectors(&entry, &file_stem(name));
        if store.model != embedder.model() {
            return Some(Ok(Vec::new()));
        }
        let table = if labels {
            &store.labels
        } else {
            &store.concepts
        };
        if table.is_empty() {
            return Some(Ok(Vec::new()));
        }
        // Cue vectors come from the process cache when the wording has
        // been seen before; no lock is held across the provider call.
        let cached = self.0.cue_cache.lock().unwrap().get(cue);
        let cue_vector = match cached {
            Some(vector) => vector,
            None => {
                let vector = match embedder.embed(&[cue]) {
                    Ok(mut vectors) => Arc::new(vectors.pop().unwrap_or_default()),
                    Err(error) => return Some(Err(error)),
                };
                self.0
                    .cue_cache
                    .lock()
                    .unwrap()
                    .insert(cue.to_string(), Arc::clone(&vector));
                vector
            }
        };
        let mut scored: Vec<(String, f32)> = table
            .iter()
            .map(|(name, (_, vector))| (name.clone(), similarity(&cue_vector, vector)))
            .filter(|&(_, score)| score >= floor)
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(SEMANTIC_LIMIT);
        Some(Ok(scored))
    }

    /// Updates the description and/or pin flag, persisting the sidecar
    /// immediately. Pinning loads the context now (pinned means
    /// resident); unpinning subjects it to the cache budget again.
    pub fn update_meta(
        &self,
        name: &str,
        description: Option<String>,
        pinned: Option<bool>,
        dice_floor: Option<f64>,
        semantic_floor: Option<f32>,
    ) -> Option<io::Result<ContextMeta>> {
        let entry = self.lookup(name)?;
        let outcome = {
            let mut guard = entry.inner.write().unwrap();
            let inner = &mut *guard;
            if let Some(description) = description {
                inner.meta.description = description;
            }
            if let Some(pinned) = pinned {
                inner.meta.pinned = pinned;
            }
            if let Some(floor) = dice_floor {
                inner.meta.dice_floor = Some(floor.clamp(0.0, 1.0));
                // A loaded context picks the new floor up immediately;
                // a cold one gets it on its next load.
                if let Slot::Hot(context) = &mut inner.slot {
                    context.set_dice_floor(inner.meta.dice_floor);
                }
            }
            if let Some(floor) = semantic_floor {
                // Read at query time from the meta; nothing to push into
                // the loaded context.
                inner.meta.semantic_floor = Some(floor.clamp(0.0, 1.0));
            }
            if inner.meta.pinned
                && let Err(error) = ensure_hot(&self.0.data_dir, name, inner)
            {
                return Some(Err(io::Error::other(error)));
            }
            let inner = &*inner;
            write_meta(
                &self.0.data_dir,
                &file_stem(name),
                &inner.meta,
                &inner.stats,
            )
            .map(|()| inner.meta.clone())
        };
        self.enforce_budget(name);
        Some(outcome)
    }

    /// The routing directory: every context's name, description, policy,
    /// residency, and stats, in name order.
    pub fn directory(&self) -> Vec<DirectoryEntry> {
        let mut directory: Vec<DirectoryEntry> = self
            .snapshot()
            .into_iter()
            .map(|(name, entry)| {
                let inner = entry.inner.read().unwrap();
                let (loaded, stats) = match &inner.slot {
                    Slot::Hot(context) => (true, ContextStats::of(context)),
                    Slot::Cold => (false, inner.stats.clone()),
                };
                DirectoryEntry {
                    name,
                    description: inner.meta.description.clone(),
                    pinned: inner.meta.pinned,
                    loaded,
                    dice_floor: inner.meta.dice_floor,
                    semantic_floor: inner.meta.semantic_floor,
                    stats,
                }
            })
            .collect();
        directory.sort_by(|a, b| a.name.cmp(&b.name));
        directory
    }

    /// Runs a read-only operation on one context, loading it first if
    /// cold.
    pub fn read_context<T>(
        &self,
        name: &str,
        operate: impl FnOnce(&Context) -> T,
    ) -> Result<T, AccessError> {
        self.with_hot(name, false, |context| operate(context))
    }

    /// Runs a mutating operation on one context, loading it first if
    /// cold, and marks it dirty. The write becomes durable at the next
    /// flush — periodic, on eviction, or on shutdown.
    pub fn write_context<T>(
        &self,
        name: &str,
        operate: impl FnOnce(&mut Context) -> T,
    ) -> Result<T, AccessError> {
        self.with_hot(name, true, operate)
    }

    /// Persists every dirty context and returns the names it flushed —
    /// the periodic flusher feeds those into the auto embedding refresh
    /// when that is enabled. Called once more on graceful shutdown; a
    /// failed save is retried on the next tick (the entry stays dirty).
    pub fn flush_dirty(&self) -> Vec<String> {
        let mut flushed = Vec::new();
        for (name, entry) in self.snapshot() {
            if !entry.dirty.load(Ordering::Relaxed) {
                continue;
            }
            let mut guard = entry.inner.write().unwrap();
            let inner = &mut *guard;
            let Slot::Hot(context) = &inner.slot else {
                continue;
            };
            let stats = ContextStats::of(context);
            match save_files(&self.0.data_dir, &name, &inner.meta, &stats, context) {
                Ok(()) => {
                    inner.stats = stats;
                    entry.dirty.store(false, Ordering::Relaxed);
                    flushed.push(name);
                }
                Err(error) => {
                    eprintln!("flush of context '{name}' failed (will retry): {error}");
                }
            }
        }
        flushed
    }

    fn lookup(&self, name: &str) -> Option<Arc<Entry>> {
        self.0.registry.read().unwrap().get(name).cloned()
    }

    fn snapshot(&self) -> Vec<(String, Arc<Entry>)> {
        self.0
            .registry
            .read()
            .unwrap()
            .iter()
            .map(|(name, entry)| (name.clone(), Arc::clone(entry)))
            .collect()
    }

    fn with_hot<T>(
        &self,
        name: &str,
        mark_dirty: bool,
        operate: impl FnOnce(&mut Context) -> T,
    ) -> Result<T, AccessError> {
        let entry = self.lookup(name).ok_or(AccessError::NotFound)?;
        let result = {
            let mut inner = entry.inner.write().unwrap();
            ensure_hot(&self.0.data_dir, name, &mut inner).map_err(AccessError::Load)?;
            let Slot::Hot(context) = &mut inner.slot else {
                unreachable!("ensure_hot leaves the slot hot");
            };
            let result = operate(context);
            if mark_dirty {
                entry.dirty.store(true, Ordering::Relaxed);
            }
            result
        };
        entry.last_touch.store(
            self.0.clock.fetch_add(1, Ordering::Relaxed) + 1,
            Ordering::Relaxed,
        );
        self.enforce_budget(name);
        Ok(result)
    }

    /// Evicts least-recently-used, unpinned, hot contexts until their
    /// resident estimate fits the budget. `except` (the context just
    /// used) is never evicted, so a single oversized context cannot
    /// thrash. Dirty contexts are persisted before eviction; if that
    /// save fails they stay resident rather than losing writes.
    /// The entry's vector store, loaded from its sidecar on first use
    /// and held until refresh replaces it or eviction clears it.
    fn entry_vectors(&self, entry: &Entry, stem: &str) -> Arc<VectorStore> {
        let mut cached = entry.vectors.lock().unwrap();
        match &*cached {
            Some(store) => Arc::clone(store),
            None => {
                let store = Arc::new(VectorStore::load(&vectors_path(&self.0.data_dir, stem)));
                *cached = Some(Arc::clone(&store));
                store
            }
        }
    }

    fn enforce_budget(&self, except: &str) {
        let mut candidates: Vec<(u64, usize, String, Arc<Entry>)> = Vec::new();
        let mut total = 0usize;
        for (name, entry) in self.snapshot() {
            let inner = entry.inner.read().unwrap();
            if inner.meta.pinned {
                continue;
            }
            let resident = match &inner.slot {
                Slot::Hot(context) => context.footprint(),
                Slot::Cold => 0,
            };
            drop(inner);
            // Cached vector stores count too — a cold entry can hold a
            // large one after a semantic query.
            let vectors = entry
                .vectors
                .lock()
                .unwrap()
                .as_ref()
                .map(|store| store.footprint())
                .unwrap_or(0);
            let bytes = resident + vectors;
            if bytes == 0 {
                continue;
            }
            total += bytes;
            candidates.push((entry.last_touch.load(Ordering::Relaxed), bytes, name, entry));
        }
        if total <= self.0.cache_bytes {
            return;
        }

        candidates.sort_unstable_by_key(|&(touch, ..)| touch);
        for (_, bytes, name, entry) in candidates {
            if total <= self.0.cache_bytes {
                break;
            }
            if name == except {
                continue;
            }
            let mut guard = entry.inner.write().unwrap();
            let inner = &mut *guard;
            // Re-check under the write lock; the entry may have changed
            // between the snapshot and now.
            if inner.meta.pinned {
                continue;
            }
            if let Slot::Hot(context) = &inner.slot {
                if entry.dirty.load(Ordering::Relaxed) {
                    let stats = ContextStats::of(context);
                    if let Err(error) =
                        save_files(&self.0.data_dir, &name, &inner.meta, &stats, context)
                    {
                        eprintln!("context '{name}' stays resident, eviction save failed: {error}");
                        continue;
                    }
                    inner.stats = stats;
                    entry.dirty.store(false, Ordering::Relaxed);
                } else {
                    inner.stats = ContextStats::of(context);
                }
                inner.slot = Slot::Cold;
            }
            *entry.vectors.lock().unwrap() = None;
            total = total.saturating_sub(bytes);
        }
    }
}

/// Loads the image behind a cold slot; hot slots pass through. On
/// success the slot is hot and the stats are fresh.
fn ensure_hot(data_dir: &Path, name: &str, inner: &mut EntryInner) -> Result<(), String> {
    if matches!(inner.slot, Slot::Hot(_)) {
        return Ok(());
    }
    let path = image_path(data_dir, &file_stem(name));
    let bytes = fs::read(&path).map_err(|e| format!("context '{name}' image unreadable: {e}"))?;
    let mut context =
        Context::from_bytes(&bytes).map_err(|e| format!("context '{name}' image corrupt: {e}"))?;
    // The image carries knowledge only; tuning config lives in the
    // sidecar and is re-applied on every load.
    context.set_dice_floor(inner.meta.dice_floor);
    inner.stats = ContextStats::of(&context);
    inner.slot = Slot::Hot(Box::new(context));
    Ok(())
}

fn image_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.ctx"))
}

fn meta_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.meta.json"))
}

fn sources_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.sources.json"))
}

fn vectors_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.vectors.bin"))
}

/// The "terms" of the passage search, as u64 keys. ASCII-alphanumeric
/// runs count as whole words; everything else contributes adjacent
/// character pairs within its run (a run of one contributes the lone
/// character). Space-delimited languages need word terms — character
/// pairs occur in every English document alike, which flattens IDF to
/// nothing — while undelimited Japanese needs the bigrams. Runs break
/// at spaces and punctuation, and a script switch breaks the run too,
/// so terms never straddle "第10篇"-style boundaries.
fn text_terms(text: &str) -> Vec<u64> {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x1_0000_01b3;
    let mut terms = Vec::new();
    let mut word = FNV_OFFSET; // running FNV-1a over the current ASCII word
    let mut in_word = false;
    let mut run: Option<char> = None; // previous char of the current non-ASCII run
    let mut run_len = 0usize;
    let flush_run = |terms: &mut Vec<u64>, run: &mut Option<char>, run_len: &mut usize| {
        if let (Some(last), 1) = (*run, *run_len) {
            terms.push(last as u64); // below the pair space: pairs always have bits 32+
        }
        *run = None;
        *run_len = 0;
    };
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            flush_run(&mut terms, &mut run, &mut run_len);
            word ^= ch as u64;
            word = word.wrapping_mul(FNV_PRIME);
            in_word = true;
        } else {
            if in_word {
                terms.push(word | 1 << 63); // disjoint from pair keys (chars < 2^21)
                word = FNV_OFFSET;
                in_word = false;
            }
            if ch.is_alphanumeric() {
                if let Some(prev) = run {
                    terms.push(((prev as u64) << 32) | ch as u64);
                }
                run = Some(ch);
                run_len += 1;
            } else {
                flush_run(&mut terms, &mut run, &mut run_len);
            }
        }
    }
    if in_word {
        terms.push(word | 1 << 63);
    }
    flush_run(&mut terms, &mut run, &mut run_len);
    terms
}

/// Reads a passages file, treating any problem as "no passages" — a
/// corrupt sidecar must not block the graph or new registrations.
fn read_passages(path: &Path) -> BTreeMap<String, String> {
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|error| {
            eprintln!("ignoring corrupt passages at {}: {error}", path.display());
            BTreeMap::new()
        }),
        Err(_) => BTreeMap::new(),
    }
}

fn save_files(
    dir: &Path,
    name: &str,
    meta: &ContextMeta,
    stats: &ContextStats,
    context: &Context,
) -> io::Result<()> {
    let stem = file_stem(name);
    write_atomic(&image_path(dir, &stem), &context.to_bytes())?;
    write_meta(dir, &stem, meta, stats)
}

fn write_meta(dir: &Path, stem: &str, meta: &ContextMeta, stats: &ContextStats) -> io::Result<()> {
    let file = MetaFile {
        meta: meta.clone(),
        stats: stats.clone(),
    };
    write_atomic(&meta_path(dir, stem), &serde_json::to_vec_pretty(&file)?)
}

/// Reads the sidecar, falling back to defaults on any problem — a
/// missing or corrupt sidecar must not make the image unreachable.
fn read_meta_file(dir: &Path, stem: &str) -> MetaFile {
    match fs::read(meta_path(dir, stem)) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|error| {
            eprintln!("ignoring corrupt sidecar for '{stem}': {error}");
            MetaFile::default()
        }),
        Err(_) => MetaFile::default(),
    }
}

/// Writes via a temporary file and rename, so a crash mid-write leaves
/// the previous version intact instead of a torn file.
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)
}

/// Encodes a context name as a file stem: bytes outside [A-Za-z0-9_-]
/// become %XX. Context names arrive from URL paths and may contain path
/// separators or dots; encoding them keeps every name inside the data
/// directory (no traversal) and reversible.
fn file_stem(name: &str) -> String {
    let mut stem = String::new();
    for byte in name.bytes() {
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' => stem.push(byte as char),
            _ => stem.push_str(&format!("%{byte:02X}")),
        }
    }
    stem
}

/// Decodes [`file_stem`]'s encoding back into a context name.
fn name_from_stem(stem: &str) -> Option<String> {
    let mut bytes = Vec::with_capacity(stem.len());
    let mut cursor = stem.bytes();
    while let Some(byte) = cursor.next() {
        if byte == b'%' {
            let high = cursor.next()?;
            let low = cursor.next()?;
            let hex = [high, low];
            let hex = std::str::from_utf8(&hex).ok()?;
            bytes.push(u8::from_str_radix(hex, 16).ok()?);
        } else {
            bytes.push(byte);
        }
    }
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("arag-registry-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn loaded_map(state: &AppState) -> HashMap<String, bool> {
        state
            .directory()
            .into_iter()
            .map(|entry| (entry.name, entry.loaded))
            .collect()
    }

    #[test]
    fn file_stem_roundtrips_any_name() {
        for name in [
            "sake",
            "用語集",
            "a/b\\c..d",
            "MiXed-123_ok",
            "%weird%",
            "空白 と 記号!?",
        ] {
            let stem = file_stem(name);
            assert!(
                stem.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'%'),
                "stem '{stem}' carries raw special bytes"
            );
            assert_eq!(name_from_stem(&stem).as_deref(), Some(name));
        }
    }

    #[test]
    fn name_from_stem_rejects_torn_encodings() {
        assert_eq!(name_from_stem("%"), None);
        assert_eq!(name_from_stem("%4"), None);
        assert_eq!(name_from_stem("%zz"), None);
        // Undecodable UTF-8 is refused rather than lossily replaced.
        assert_eq!(name_from_stem("%FF%FE"), None);
    }

    #[test]
    fn budget_evicts_lru_and_reloads_transparently() {
        let dir = scratch_dir("evict");
        // A budget of one byte: at most the just-used context stays hot.
        let state = AppState::boot(dir.clone(), 1, None).unwrap();
        state
            .create("a", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .create("b", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        state
            .write_context("a", |context| {
                context.associate("私", "好き", "りんご", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();
        // Touching b evicts a (least recently used, and b is protected
        // as the context just used) — flushing a's dirty write first.
        state
            .read_context("b", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        let loaded = loaded_map(&state);
        assert!(!loaded["a"], "a must be evicted");
        assert!(loaded["b"], "the just-used context must stay");

        // The evicted write must have survived the disk roundtrip.
        let recalled = state
            .read_context("a", |context| context.recall("私").len())
            .map_err(|_| "reload")
            .unwrap();
        assert_eq!(recalled, 1);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn pinned_contexts_are_never_evicted_and_preload_on_boot() {
        let dir = scratch_dir("pin");
        {
            let state = AppState::boot(dir.clone(), 1, None).unwrap();
            let pinned = ContextMeta {
                description: "glossary".into(),
                pinned: true,
                ..ContextMeta::default()
            };
            state
                .create("glossary", pinned)
                .map_err(|_| "create")
                .unwrap();
            state
                .create("other", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .write_context("glossary", |context| {
                    context.associate("用語", "意味", "定義", 1.0).unwrap();
                })
                .map_err(|_| "write")
                .unwrap();

            // Churning through the other context must not push the
            // pinned one out.
            state
                .read_context("other", |context| context.association_count())
                .map_err(|_| "read")
                .unwrap();
            assert!(loaded_map(&state)["glossary"]);
            state.flush_dirty();
        }

        // A fresh boot preloads pinned contexts and leaves the rest cold.
        let state = AppState::boot(dir.clone(), 1, None).unwrap();
        let loaded = loaded_map(&state);
        assert!(loaded["glossary"], "pinned must preload");
        assert!(!loaded["other"], "unpinned must boot cold");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dirty_contexts_survive_flush_and_cold_boot() {
        let dir = scratch_dir("flush");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .write_context("sake", |context| {
                    context
                        .associate("青嶺酒造", "代表銘柄", "青嶺", 1.0)
                        .unwrap();
                })
                .map_err(|_| "write")
                .unwrap();
            state.flush_dirty();
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        // Cold entries serve directory stats from the sidecar snapshot.
        let directory = state.directory();
        let sake = directory.iter().find(|e| e.name == "sake").unwrap();
        assert!(!sake.loaded);
        assert_eq!(sake.stats.associations, 1);

        let recalled = state
            .read_context("sake", |context| context.recall("青嶺").len())
            .map_err(|_| "reload")
            .unwrap();
        assert_eq!(recalled, 1);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn passages_store_lookup_and_survive_restart() {
        let dir = scratch_dir("passages");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            let mut passages = BTreeMap::new();
            passages.insert(
                "第1段落".to_string(),
                "青嶺酒造は、雲居県霧沢町にある日本酒の蔵元である。".to_string(),
            );
            assert_eq!(state.store_passages("sake", passages).unwrap().unwrap(), 1);
        }

        // A fresh boot serves the registered passage; unknown sources
        // come back as missing rather than erroring.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let (passages, missing) = state
            .lookup_passages("sake", &["第1段落".to_string(), "第9段落".to_string()])
            .unwrap();
        assert!(passages["第1段落"].starts_with("青嶺酒造は"));
        assert_eq!(missing, vec!["第9段落".to_string()]);
        assert_eq!(state.passage_sources("sake").unwrap(), vec!["第1段落"]);
        assert!(state.lookup_passages("nope", &[]).is_none());

        // Deleting the context removes its passages file with it.
        state.delete("sake").unwrap().unwrap();
        assert!(!sources_path(&dir, &file_stem("sake")).exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dice_floor_persists_in_the_sidecar_and_reapplies_on_load() {
        let dir = scratch_dir("floor");
        // One shared informative bigram of 4+3: Dice ≈ 0.286 — misses
        // the 0.3 default, lands once the context is tuned to 0.25.
        let fuzzy_cue = "青嶺の純米";
        let lands = |state: &AppState| {
            state
                .read_context("sake", |context| {
                    context
                        .resolve(fuzzy_cue)
                        .iter()
                        .any(|hit| hit.name == "青嶺酒造")
                })
                .map_err(|_| "read")
                .unwrap()
        };
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .write_context("sake", |context| {
                    context.associate("青嶺酒造", "分類", "酒蔵", 1.0).unwrap();
                })
                .map_err(|_| "write")
                .unwrap();

            assert!(!lands(&state), "default floor must reject the cue");

            // Tuning applies to the loaded context immediately.
            state
                .update_meta("sake", None, None, Some(0.25), None)
                .unwrap()
                .unwrap();
            assert!(lands(&state), "tuned floor must admit the cue");
            // The flusher learns which contexts it persisted — that list
            // feeds the auto embedding refresh.
            assert_eq!(state.flush_dirty(), vec!["sake".to_string()]);
            assert!(state.flush_dirty().is_empty());
        }

        // A cold boot re-applies the floor from the sidecar — the image
        // itself carries no config.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(lands(&state), "floor must survive the restart");
        assert_eq!(state.directory()[0].dice_floor, Some(0.25));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn passage_search_ranks_the_answering_text_first() {
        let dir = scratch_dir("bm25");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "第2段落".to_string(),
            "原料米には主に山田錦を使い、精米歩合は50パーセントまで磨く。".to_string(),
        );
        passages.insert(
            "第3段落".to_string(),
            "杜氏の高瀬は南部杜氏の出身で、経験は30年を超える。".to_string(),
        );
        passages.insert(
            "第5段落".to_string(),
            "蔵開きの祭りでは、雲居山の伏流水で仕込んだ新酒がふるまわれる。".to_string(),
        );
        state.store_passages("sake", passages).unwrap().unwrap();

        // The procedural question never became a triple; the text lane
        // must still hand back the passage that answers it, first.
        let hits = state
            .search_passages("sake", "精米歩合はどこまで磨く?", 3)
            .unwrap();
        assert_eq!(hits[0].0, "第2段落");
        assert!(hits[0].1 > 0.0);

        // No shared bigrams at all → nothing, not noise.
        assert!(
            state
                .search_passages("sake", "unrelated english words", 3)
                .unwrap()
                .is_empty()
        );
        assert!(state.search_passages("nope", "x", 3).is_none());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn passage_search_discriminates_english_by_words() {
        let dir = scratch_dir("bm25-english");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("papers", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // English prose shares nearly every character pair across
        // documents; only word terms can tell these two apart. The
        // real-world case: a famous quote had to be found in the essay
        // that contains it, not in whichever essay mentions the topic.
        let mut passages = BTreeMap::new();
        passages.insert(
            "第51篇".to_string(),
            "The great security against a gradual concentration of the several powers \
             in the same department consists in giving to those who administer each \
             department the necessary constitutional means and personal motives to \
             resist encroachments of the others. Ambition must be made to counteract \
             ambition."
                .to_string(),
        );
        passages.insert(
            "第70篇".to_string(),
            "Energy in the executive is a leading character in the definition of good \
             government. It is essential to the protection of the community against \
             foreign attacks and to the security of liberty against the enterprises \
             and assaults of ambition, of faction, and of anarchy."
                .to_string(),
        );
        state.store_passages("papers", passages).unwrap().unwrap();

        let hits = state
            .search_passages("papers", "ambition must be made to counteract ambition", 2)
            .unwrap();
        assert_eq!(hits[0].0, "第51篇");
        assert!(
            hits.len() < 2 || hits[0].1 > hits[1].1,
            "the containing passage must win decisively, not by tie-break"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// Deterministic provider: a text starting with a mapped key gets
    /// that key's unit vector (glosses start with their name), anything
    /// else lands on an axis orthogonal to all of them. Counts provider
    /// round trips so cache behavior is observable.
    struct MockEmbeddings {
        keys: Vec<(String, Vec<f32>)>,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl MockEmbeddings {
        fn fruity(calls: &Arc<std::sync::atomic::AtomicUsize>) -> Self {
            Self {
                keys: vec![
                    ("りんご".to_string(), vec![1.0, 0.0, 0.0]),
                    ("アップル".to_string(), vec![0.96, 0.28, 0.0]),
                    ("みかん".to_string(), vec![0.28, 0.96, 0.0]),
                ],
                calls: Arc::clone(calls),
            }
        }
    }

    impl EmbeddingProvider for MockEmbeddings {
        fn model(&self) -> &str {
            "mock"
        }

        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(texts
                .iter()
                .map(|text| {
                    self.keys
                        .iter()
                        .find(|(key, _)| text.starts_with(key.as_str()))
                        .map(|(_, vector)| vector.clone())
                        .unwrap_or_else(|| vec![0.0, 0.0, 1.0])
                })
                .collect())
        }
    }

    #[test]
    fn semantic_fallback_lands_paraphrases_after_refresh() {
        let dir = scratch_dir("embed");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let embedder = Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .write_context("fruit", |context| {
                context.associate("りんご", "分類", "果物", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();

        // アップル shares no normalized characters with りんご: every
        // lexical tier misses, and before a refresh so does semantics.
        let lexical = state
            .read_context("fruit", |context| context.resolve("アップル"))
            .map_err(|_| "read")
            .unwrap();
        assert!(lexical.is_empty());
        assert!(
            state
                .semantic_resolve("fruit", "アップル", false, None)
                .unwrap()
                .unwrap()
                .is_empty()
        );

        // Refresh embeds every canonical name's gloss once; a second run
        // is a no-op.
        let (embedded, total) = state.refresh_embeddings("fruit").unwrap().unwrap();
        assert_eq!(embedded, 3); // りんご, 果物 + label 分類
        assert_eq!(total, 3);
        assert_eq!(state.refresh_embeddings("fruit").unwrap().unwrap().0, 0);

        // Now the paraphrase lands on the stored spelling by cosine, and
        // unrelated names stay under the floor.
        let hits = state
            .semantic_resolve("fruit", "アップル", false, None)
            .unwrap()
            .unwrap();
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].0, "りんご");
        assert!(hits[0].1 > 0.9);

        // A new fact changes りんご's gloss: the next refresh re-embeds
        // exactly what changed — りんご plus the new 青森 and 産地 —
        // while 果物 and 分類, whose glosses are untouched, are not
        // re-sent to the provider.
        state
            .write_context("fruit", |context| {
                context.associate("りんご", "産地", "青森", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();
        let (embedded, total) = state.refresh_embeddings("fruit").unwrap().unwrap();
        assert_eq!(embedded, 3);
        assert_eq!(total, 5);

        assert!(state.semantic_resolve("nope", "x", false, None).is_none());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn semantic_path_caches_cue_vectors_and_the_sidecar() {
        let dir = scratch_dir("semcache");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let embedder = Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), 1, embedder).unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .write_context("fruit", |context| {
                context.associate("りんご", "分類", "果物", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();
        state.refresh_embeddings("fruit").unwrap().unwrap();
        // One batch per namespace: concepts, then labels.
        assert_eq!(calls.load(Ordering::Relaxed), 2);

        // First query embeds the cue; repeating the wording does not.
        let first = state
            .semantic_resolve("fruit", "アップル", false, None)
            .unwrap()
            .unwrap();
        assert_eq!(first[0].0, "りんご");
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        state
            .semantic_resolve("fruit", "アップル", false, None)
            .unwrap()
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 3, "cue must come from cache");

        // The sidecar is held in memory after first use: even with the
        // file gone, the same query keeps answering.
        fs::remove_file(vectors_path(&dir, &file_stem("fruit"))).unwrap();
        let held = state
            .semantic_resolve("fruit", "アップル", false, None)
            .unwrap()
            .unwrap();
        assert_eq!(held[0].0, "りんご");

        // Eviction clears the cached store (budget is one byte, and the
        // vector cache counts): after touching another context, the
        // deleted sidecar means no vectors — proving the memory copy
        // was dropped rather than leaked.
        state
            .create("other", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .read_context("other", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert!(
            state
                .semantic_resolve("fruit", "アップル", false, None)
                .unwrap()
                .unwrap()
                .is_empty()
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn semantic_twins_surface_synonym_forks_from_stored_vectors() {
        let dir = scratch_dir("twins");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        // Two label glosses embed close together: a synonym fork that
        // no spelling comparison could see.
        let embedder = MockEmbeddings {
            keys: vec![
                ("創業年".to_string(), vec![1.0, 0.0, 0.0]),
                ("設立年".to_string(), vec![0.95, 0.31, 0.0]),
            ],
            calls: Arc::clone(&calls),
        };
        let embedder = Some(Arc::new(embedder) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .write_context("sake", |context| {
                context
                    .associate("青嶺酒造", "創業年", "1907年", 1.0)
                    .unwrap();
                context
                    .associate("別の蔵", "設立年", "1950年", 1.0)
                    .unwrap();
            })
            .map_err(|_| "write")
            .unwrap();

        // Before any vectors exist the semantic half is skipped, loudly.
        let (concepts, labels, note) = state.semantic_twins("sake", 0.6).unwrap();
        assert!(concepts.is_empty() && labels.is_empty());
        assert!(note.is_some());

        state.refresh_embeddings("sake").unwrap().unwrap();
        let (concepts, labels, note) = state.semantic_twins("sake", 0.6).unwrap();
        assert!(note.is_none());
        // Directly connected concepts (青嶺酒造 —創業年→ 1907年) are
        // related, not duplicates, and must be filtered out however
        // similar their vectors are.
        let pairs_up = |a: &str, b: &str, x: &str, y: &str| a.contains(x) && b.contains(y);
        assert!(
            concepts.iter().all(|(a, b, _)| !pairs_up(a, b, "青嶺酒造", "1907年")
                && !pairs_up(a, b, "1907年", "青嶺酒造")),
            "{concepts:?}"
        );
        assert_eq!(labels.len(), 1, "{labels:?}");
        assert_eq!(
            (labels[0].0.as_str(), labels[0].1.as_str()),
            ("創業年", "設立年")
        );
        assert!(labels[0].2 > 0.9);

        // No provider round trip happens for the sweep itself: the two
        // audits above added no embed calls beyond the refresh batches
        // (2 namespaces) — stored vectors are compared directly.
        assert_eq!(calls.load(Ordering::Relaxed), 2);

        assert!(state.semantic_twins("nope", 0.6).is_none());
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn semantic_floor_is_tunable_per_context_and_per_call() {
        let dir = scratch_dir("semfloor");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let embedder = Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .write_context("fruit", |context| {
                context.associate("りんご", "分類", "果物", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();
        state.refresh_embeddings("fruit").unwrap().unwrap();

        // みかん×りんご sits at cosine 0.28 — under the 0.35 default.
        let miss = |floor: Option<f32>| {
            state
                .semantic_resolve("fruit", "みかん", false, floor)
                .unwrap()
                .unwrap()
        };
        assert!(miss(None).is_empty());
        // A one-call override admits it without changing the context ...
        assert_eq!(miss(Some(0.2))[0].0, "りんご");
        assert!(miss(None).is_empty());
        // ... and the context setting changes the default, persisting
        // in the sidecar across a reboot.
        state
            .update_meta("fruit", None, None, None, Some(0.2))
            .unwrap()
            .unwrap();
        assert_eq!(miss(None)[0].0, "りんご");
        assert_eq!(state.directory()[0].semantic_floor, Some(0.2));

        let _ = fs::remove_dir_all(dir);
    }
}
