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

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use associative_rag::context::Context;
use serde::{Deserialize, Serialize};

use crate::embedding::{EmbeddingProvider, VectorStore, similarity};

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
}

impl Entry {
    fn new(meta: ContextMeta, stats: ContextStats, slot: Slot) -> Self {
        Self {
            inner: RwLock::new(EntryInner { meta, stats, slot }),
            dirty: AtomicBool::new(false),
            last_touch: AtomicU64::new(0),
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

    /// Embeds every canonical concept and label name that has no vector
    /// yet (all of them after a model change) and persists the vector
    /// sidecar. Explicit rather than automatic — an agent or operator
    /// calls this after ingesting, so embedding spend stays intentional.
    /// Returns (newly embedded, total vectors), or `None` for an unknown
    /// context.
    pub fn refresh_embeddings(&self, name: &str) -> Option<Result<(usize, usize), String>> {
        let Some(embedder) = self.0.embedder.clone() else {
            return Some(Err(
                "no embedding provider is configured (set ARAG_EMBED_URL and ARAG_EMBED_MODEL)"
                    .to_string(),
            ));
        };
        let names = match self.read_context(name, |context| {
            let concepts: Vec<String> = context
                .concept_names()
                .into_iter()
                .map(String::from)
                .collect();
            let labels: Vec<String> = context.labels().into_iter().map(String::from).collect();
            (concepts, labels)
        }) {
            Ok(names) => names,
            Err(AccessError::NotFound) => return None,
            Err(AccessError::Load(message)) => return Some(Err(message)),
        };
        let (concepts, labels) = names;
        let entry = self.lookup(name)?;
        let path = vectors_path(&self.0.data_dir, &file_stem(name));

        // Diff and embed without holding the entry lock — provider round
        // trips can take seconds. Concurrent refreshes at worst re-embed
        // the same names; the merge below stays correct.
        let existing = VectorStore::load(&path);
        let fresh_model = existing.model != embedder.model();
        let missing = |table: &HashMap<String, Vec<f32>>, names: &[String]| -> Vec<String> {
            names
                .iter()
                .filter(|name| fresh_model || !table.contains_key(*name))
                .cloned()
                .collect()
        };
        let missing_concepts = missing(&existing.concepts, &concepts);
        let missing_labels = missing(&existing.labels, &labels);
        let mut embedded_concepts = Vec::new();
        let mut embedded_labels = Vec::new();
        for (missing, embedded) in [
            (&missing_concepts, &mut embedded_concepts),
            (&missing_labels, &mut embedded_labels),
        ] {
            for chunk in missing.chunks(128) {
                let texts: Vec<&str> = chunk.iter().map(String::as_str).collect();
                match embedder.embed(&texts) {
                    Ok(vectors) => embedded.extend(chunk.iter().cloned().zip(vectors)),
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
    ) -> Option<Result<Vec<(String, f32)>, String>> {
        const SEMANTIC_FLOOR: f32 = 0.5;
        const SEMANTIC_LIMIT: usize = 5;

        let Some(embedder) = self.0.embedder.clone() else {
            return Some(Ok(Vec::new()));
        };
        self.lookup(name)?;
        let store = VectorStore::load(&vectors_path(&self.0.data_dir, &file_stem(name)));
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
        let cue_vector = match embedder.embed(&[cue]) {
            Ok(mut vectors) => vectors.pop().unwrap_or_default(),
            Err(error) => return Some(Err(error)),
        };
        let mut scored: Vec<(String, f32)> = table
            .iter()
            .map(|(name, vector)| (name.clone(), similarity(&cue_vector, vector)))
            .filter(|&(_, score)| score >= SEMANTIC_FLOOR)
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

    /// Persists every dirty context. Called by the periodic flusher and
    /// once more on graceful shutdown; a failed save is retried on the
    /// next tick (the entry stays dirty).
    pub fn flush_dirty(&self) {
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
                }
                Err(error) => {
                    eprintln!("flush of context '{name}' failed (will retry): {error}");
                }
            }
        }
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
    fn enforce_budget(&self, except: &str) {
        let mut hot: Vec<(u64, usize, String, Arc<Entry>)> = Vec::new();
        let mut total = 0usize;
        for (name, entry) in self.snapshot() {
            let inner = entry.inner.read().unwrap();
            if inner.meta.pinned {
                continue;
            }
            if let Slot::Hot(context) = &inner.slot {
                let bytes = context.footprint();
                total += bytes;
                drop(inner);
                hot.push((entry.last_touch.load(Ordering::Relaxed), bytes, name, entry));
            }
        }
        if total <= self.0.cache_bytes {
            return;
        }

        hot.sort_unstable_by_key(|&(touch, ..)| touch);
        for (_, bytes, name, entry) in hot {
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
            let Slot::Hot(context) = &inner.slot else {
                continue;
            };
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
                .update_meta("sake", None, None, Some(0.25))
                .unwrap()
                .unwrap();
            assert!(lands(&state), "tuned floor must admit the cue");
            state.flush_dirty();
        }

        // A cold boot re-applies the floor from the sidecar — the image
        // itself carries no config.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(lands(&state), "floor must survive the restart");
        assert_eq!(state.directory()[0].dice_floor, Some(0.25));

        let _ = fs::remove_dir_all(dir);
    }

    /// Deterministic provider: mapped names get fixed unit vectors,
    /// everything else lands on an axis orthogonal to all of them.
    struct MockEmbeddings(HashMap<String, Vec<f32>>);

    impl EmbeddingProvider for MockEmbeddings {
        fn model(&self) -> &str {
            "mock"
        }

        fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
            Ok(texts
                .iter()
                .map(|text| {
                    self.0
                        .get(*text)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0, 0.0, 1.0])
                })
                .collect())
        }
    }

    #[test]
    fn semantic_fallback_lands_paraphrases_after_refresh() {
        let dir = scratch_dir("embed");
        let mut vectors = HashMap::new();
        vectors.insert("りんご".to_string(), vec![1.0, 0.0, 0.0]);
        vectors.insert("アップル".to_string(), vec![0.96, 0.28, 0.0]);
        let embedder = Some(Arc::new(MockEmbeddings(vectors)) as Arc<dyn EmbeddingProvider>);
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
                .semantic_resolve("fruit", "アップル", false)
                .unwrap()
                .unwrap()
                .is_empty()
        );

        // Refresh embeds every canonical name once; a second run is a
        // no-op.
        let (embedded, total) = state.refresh_embeddings("fruit").unwrap().unwrap();
        assert_eq!(embedded, 3); // りんご, 果物 + label 分類
        assert_eq!(total, 3);
        assert_eq!(state.refresh_embeddings("fruit").unwrap().unwrap().0, 0);

        // Now the paraphrase lands on the stored spelling by cosine, and
        // unrelated names stay under the floor.
        let hits = state
            .semantic_resolve("fruit", "アップル", false)
            .unwrap()
            .unwrap();
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].0, "りんご");
        assert!(hits[0].1 > 0.9);

        assert!(state.semantic_resolve("nope", "x", false).is_none());

        let _ = fs::remove_dir_all(dir);
    }
}
