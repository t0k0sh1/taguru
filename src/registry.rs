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
//! Durability: every acknowledged graph write is staged in the
//! context's write-ahead log (fsynced, before it touches memory), so
//! a crash loses nothing — loading replays whatever the log holds
//! above the image's watermark. The periodic flusher, eviction, and
//! graceful shutdown still persist the image; the flush interval is
//! now just image-freshness cadence, not a loss window. Disabling the
//! WAL (`TAGURU_WAL=0`) restores the old posture: a crash loses at
//! most the writes since the last flush. Creation and metadata
//! changes persist immediately either way.
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
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use serde::{Deserialize, Serialize};
use taguru::context::{AliasError, Context};

use crate::embedding::{
    EmbedPurpose, EmbeddingProvider, VectorStore, VectorTable, fnv1a, similarity,
};
use crate::metrics::{GaugeSnapshot, Metrics};
use crate::wal::{self, WalOp};

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

/// Cumulative usage counters for one context — the "is this context
/// earning its keep" numbers the directory serves. `reads` counts the
/// retrieval operations (resolve, describe, query, activate, recall,
/// explore, passage search/lookup), `empty_reads` the subset that
/// matched nothing, `writes` the data mutations. The two failure modes
/// of a memory read differently: a context nobody reads was never
/// CHOSEN (description/routing problem), while a high empty share
/// means it gets chosen but cannot ANSWER (coverage problem).
///
/// Advisory data, deliberately outside the WAL guarantee: counters
/// live in memory and reach the sidecar when the context flushes for
/// other reasons, plus one sweep at graceful shutdown — a crash loses
/// the increments since then, and reads never cause disk writes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextUsage {
    pub reads: u64,
    pub empty_reads: u64,
    pub writes: u64,
    /// Unix seconds of the most recent read / write; 0 = never.
    pub last_read_epoch: u64,
    pub last_write_epoch: u64,
}

/// Lock-free mirror of [`ContextUsage`]: bumped on the request path
/// with relaxed atomics — counting a read must never queue behind a
/// writer holding the entry lock — and snapshotted for the directory
/// and the sidecar.
#[derive(Default)]
struct UsageCounters {
    reads: AtomicU64,
    empty_reads: AtomicU64,
    writes: AtomicU64,
    last_read_epoch: AtomicU64,
    last_write_epoch: AtomicU64,
}

impl UsageCounters {
    fn seeded(usage: &ContextUsage) -> Self {
        Self {
            reads: AtomicU64::new(usage.reads),
            empty_reads: AtomicU64::new(usage.empty_reads),
            writes: AtomicU64::new(usage.writes),
            last_read_epoch: AtomicU64::new(usage.last_read_epoch),
            last_write_epoch: AtomicU64::new(usage.last_write_epoch),
        }
    }

    fn snapshot(&self) -> ContextUsage {
        ContextUsage {
            reads: self.reads.load(Ordering::Relaxed),
            empty_reads: self.empty_reads.load(Ordering::Relaxed),
            writes: self.writes.load(Ordering::Relaxed),
            last_read_epoch: self.last_read_epoch.load(Ordering::Relaxed),
            last_write_epoch: self.last_write_epoch.load(Ordering::Relaxed),
        }
    }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_secs())
        .unwrap_or(0)
}

/// What `{name}.meta.json` holds: the meta inline plus the stats
/// snapshot as of the last save, so a directory listing can describe a
/// cold context without touching its image. `usage` rides along under
/// `#[serde(default)]`, so sidecars from before it existed load with
/// zeroed counters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
struct MetaFile {
    #[serde(flatten)]
    meta: ContextMeta,
    stats: ContextStats,
    usage: ContextUsage,
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
    pub usage: ContextUsage,
}

/// Whether a context's network is resident. Cold entries keep only
/// their metadata and stats snapshot in memory. Deleted is the
/// tombstone [`AppState::delete`] leaves for anyone who cloned the
/// entry's `Arc` out of the registry before the removal — the
/// flusher's and evictor's snapshots, a looked-up handle racing the
/// delete. Whoever takes the entry lock next must treat the context
/// as gone: a stale flush that still saw `Hot` here used to recreate
/// the files of a deleted context, resurrecting it on the next boot.
enum Slot {
    Hot(Box<Context>),
    Cold,
    Deleted,
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
    /// Usage counters (see [`ContextUsage`]). `usage_dirty` marks
    /// increments the sidecar has not seen yet, so the shutdown sweep
    /// skips the contexts nobody touched.
    usage: UsageCounters,
    usage_dirty: AtomicBool,
}

impl Entry {
    fn new(
        meta: ContextMeta,
        stats: ContextStats,
        slot: Slot,
        wal_bytes: u64,
        usage: ContextUsage,
    ) -> Self {
        Self {
            inner: RwLock::new(EntryInner {
                meta,
                stats,
                slot,
                wal_seq: 1,
                wal_bytes,
                counted_bytes: 0,
            }),
            dirty: AtomicBool::new(false),
            last_touch: AtomicU64::new(0),
            vectors: Mutex::new(None),
            usage: UsageCounters::seeded(&usage),
            usage_dirty: AtomicBool::new(false),
        }
    }

    /// The entry's write lock, or `None` if a delete beat the caller to
    /// it — a handle that predates the removal must not touch the files
    /// the delete just removed, let alone recreate them. Every
    /// post-lookup lock acquisition goes through here so no path can
    /// forget the tombstone.
    #[allow(clippy::readonly_write_lock)] // some callers lock purely for exclusion
    fn lock_unless_deleted(&self) -> Option<std::sync::RwLockWriteGuard<'_, EntryInner>> {
        let guard = self.inner.write().unwrap();
        (!matches!(guard.slot, Slot::Deleted)).then_some(guard)
    }

    /// Bytes the cached vector sidecar holds resident — zero when none
    /// is loaded. The cache budget and the gauges count it the same way.
    fn vectors_footprint(&self) -> usize {
        self.vectors
            .lock()
            .unwrap()
            .as_ref()
            .map(|store| store.footprint())
            .unwrap_or(0)
    }
}

struct EntryInner {
    meta: ContextMeta,
    stats: ContextStats,
    slot: Slot,
    /// The next WAL sequence number this context hands out. Sequences
    /// start at 1 — watermark 0 means "nothing logged is reflected".
    /// Plain u64, not atomic: every touch happens under this entry's
    /// write lock (append and flush both hold it). Meaningful while
    /// hot; a cold load recomputes it from the replay's tail.
    wal_seq: u64,
    /// Size of this context's log on disk — the growth signal behind
    /// the `taguru_wal_bytes` gauge and the `TAGURU_WAL_MAX_BYTES`
    /// backstop. Advanced on append, re-stat'ed on load, zeroed on
    /// truncation; a log only shrinks after a successful image save,
    /// so sustained growth here means flushes are failing.
    wal_bytes: u64,
    /// What this entry currently contributes to the global resident
    /// estimate (graph footprint; 0 while cold, deleted, or pinned —
    /// the budget covers unpinned residents only). Kept absolute and
    /// recounted under this lock, so the global sum cannot drift from
    /// double-applied deltas.
    counted_bytes: usize,
}

#[derive(Debug)]
pub enum CreateError {
    AlreadyExists,
    Io(io::Error),
}

/// Why an operation on a named context could not run.
#[derive(Debug)]
pub enum AccessError {
    NotFound,
    /// The context exists but its image could not be loaded from disk.
    Load(String),
    /// The write-ahead log could not durably record the operation;
    /// NOTHING was applied — the client must never hold a 200 the
    /// disk cannot replay.
    Unpersisted(String),
}

/// One requested association — the wire shape of the associations
/// endpoint and the WAL payload, one struct for both so they cannot
/// drift apart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssocOp {
    pub subject: String,
    pub label: String,
    pub object: String,
    pub weight: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// A batch write that got partway before the library rejected an
/// item: how many ops applied, the rejection's message (the exact
/// text each op family always reported), and whether it was a
/// capacity error (507) rather than a conflict (409).
#[derive(Debug)]
pub struct PartialWrite {
    pub applied: usize,
    pub message: String,
    pub full: bool,
}

/// Default per-context WAL ceiling (`TAGURU_WAL_MAX_BYTES`): a healthy
/// server truncates the log every flush interval, so a log this large
/// means the image has been failing to save for a long time — refuse
/// new writes rather than grow without bound.
pub const DEFAULT_WAL_MAX_BYTES: usize = 256 * 1024 * 1024;

/// The boot knobs `taguru serve` and `taguru import` read identically —
/// one reading, so the two entrances cannot drift. The values stay
/// visible (rather than being swallowed by [`AppState::boot_with`])
/// because serve's "server ready" line reports them.
pub struct BootConfig {
    pub data_dir: PathBuf,
    pub cache_bytes: usize,
    pub wal_enabled: bool,
    pub wal_max_bytes: usize,
    pub semantic_floor: Option<f32>,
}

impl BootConfig {
    pub fn from_env() -> Self {
        Self {
            data_dir: PathBuf::from(
                std::env::var("TAGURU_DATA_DIR").unwrap_or_else(|_| "data".into()),
            ),
            cache_bytes: crate::env_number("TAGURU_CACHE_BYTES", 512 * 1024 * 1024),
            // The WAL closes the flush-interval loss window; opting out
            // (TAGURU_WAL=0) restores the old posture for benchmarks or
            // explicit risk acceptance.
            wal_enabled: std::env::var("TAGURU_WAL")
                .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
                .unwrap_or(true),
            // Backstop for a persistently failing flush: past this,
            // writes are refused rather than growing the log without
            // bound (0 = no cap).
            wal_max_bytes: crate::env_number("TAGURU_WAL_MAX_BYTES", DEFAULT_WAL_MAX_BYTES),
            // The right semantic floor is a property of the embedding
            // model (cosine bands differ per model), so its
            // recalibration lives beside TAGURU_EMBED_MODEL rather
            // than on every context.
            semantic_floor: crate::env_floor("TAGURU_SEMANTIC_FLOOR"),
        }
    }

    /// [`AppState::boot_with`], parameterized by this configuration.
    pub fn boot(&self, embedder: Option<Arc<dyn EmbeddingProvider>>) -> io::Result<AppState> {
        AppState::boot_with(
            self.data_dir.clone(),
            self.cache_bytes,
            embedder,
            self.wal_enabled,
            self.wal_max_bytes,
            self.semantic_floor,
        )
    }
}

/// Shared server state: the data directory, the cache budget, and the
/// context registry.
#[derive(Clone)]
pub struct AppState(Arc<StateInner>);

/// Floor for the semantic entry tier when neither the call, the
/// context, nor the server (`TAGURU_SEMANTIC_FLOOR`) sets one.
/// Calibrated against text-embedding-3-large with GLOSSED names
/// (name + graph context): true matches land at ~0.44–0.58 — jargon
/// paraphrases included (醸造責任者×杜氏 0.53, 質問形「酒造りの責任者は誰」
/// 0.58, アップル×りんご 0.45) — while the noise band drops to ~0.17
/// (自動車×杜氏グロス 0.09, 自動車×りんごグロス 0.17), far better
/// separated than bare names ever were. 0.35 admits the weakest true
/// matches with ~2× margin over noise.
///
/// The right floor is a property of the EMBEDDING MODEL, not of any
/// context: amazon.titan-embed-text-v2 (512d), for one, puts Japanese
/// true matches at ~0.2–0.3 over a ~0.15 noise band, so 0.35 silently
/// discards its correct answers — that deployment wants
/// `TAGURU_SEMANTIC_FLOOR≈0.2` next to its `TAGURU_EMBED_MODEL`.
const DEFAULT_SEMANTIC_FLOOR: f32 = 0.35;

struct StateInner {
    data_dir: PathBuf,
    /// The advisory exclusive lock that makes this process the data
    /// directory's single writer — held (never read) for the whole
    /// life of the state, released by the OS when the last clone
    /// drops or the process dies. See [`lock_data_dir`].
    _dir_lock: fs::File,
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
    /// Fallback semantic floor when neither the call nor the context
    /// sets one — the server default ([`DEFAULT_SEMANTIC_FLOOR`] unless
    /// `TAGURU_SEMANTIC_FLOOR` recalibrates it for the configured
    /// embedding model).
    default_semantic_floor: f32,
    /// Process-lifetime cache of cue embeddings — an LLM client repeats
    /// query wording, and every hit saves a provider round trip on the
    /// fallback path. Valid for the whole process because the provider
    /// (and so the model) is fixed at boot.
    cue_cache: Mutex<CueCache>,
    /// The shared observability registry; every `AppState` clone —
    /// handlers, middleware, the flusher task — increments the same
    /// counters.
    metrics: Metrics,
    /// Whether acknowledged graph writes are staged in the per-context
    /// WAL before they apply. Off restores the pre-WAL posture:
    /// durability bounded by the flush interval. Replay always runs
    /// regardless — a log left behind by an earlier WAL-enabled run
    /// must never be ignored.
    wal_enabled: bool,
    /// Per-context ceiling on the log (`TAGURU_WAL_MAX_BYTES`, 0 =
    /// unlimited). The log only truncates after a successful image
    /// save, so a persistently failing flush would otherwise grow it
    /// without bound; past the cap new writes are refused
    /// ([`AccessError::Unpersisted`]) instead.
    wal_max_bytes: usize,
    /// Running estimate of unpinned resident graph bytes — the cheap
    /// gate in front of the budget sweep. Adjusted by absolute
    /// per-entry recounts (see `EntryInner::counted_bytes`); the
    /// periodic full sweep reconciles it against measured reality
    /// (folding in vector stores, which are not tracked between
    /// sweeps). Signed so a transient over-subtraction cannot wrap.
    resident_estimate: AtomicI64,
    /// Operation counter behind the every-64th forced sweep — the
    /// bound on how stale the estimate can get.
    budget_ops: AtomicU64,
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
    /// [`AppState::boot_with`] with the WAL on and the default log cap
    /// — the deployment defaults, and what the tests boot with (so the
    /// whole existing suite exercises the WAL-enabled paths).
    #[cfg(test)]
    pub fn boot(
        data_dir: PathBuf,
        cache_bytes: usize,
        embedder: Option<Arc<dyn EmbeddingProvider>>,
    ) -> io::Result<Self> {
        Self::boot_with(
            data_dir,
            cache_bytes,
            embedder,
            true,
            DEFAULT_WAL_MAX_BYTES,
            None,
        )
    }

    /// Opens (creating if needed) the data directory and registers every
    /// context image found in it — cold, described by their sidecar
    /// snapshots. Pinned contexts are loaded eagerly; a pinned image
    /// that fails to load is left cold with a warning rather than
    /// taking the server down. `wal_enabled: false` restores the
    /// flush-interval durability window (`TAGURU_WAL=0`);
    /// `wal_max_bytes` is the per-context log ceiling (0 = unlimited);
    /// `default_semantic_floor` recalibrates the semantic entry floor
    /// for the configured embedding model (`TAGURU_SEMANTIC_FLOOR`,
    /// `None` = the text-embedding-3-large calibration).
    pub fn boot_with(
        data_dir: PathBuf,
        cache_bytes: usize,
        embedder: Option<Arc<dyn EmbeddingProvider>>,
        wal_enabled: bool,
        wal_max_bytes: usize,
        default_semantic_floor: Option<f32>,
    ) -> io::Result<Self> {
        fs::create_dir_all(&data_dir)?;
        // Before reading anything: two live registries over one
        // directory (a second serve, or an import against a running
        // server) would each cache and flush independently — last
        // writer wins, silently.
        let dir_lock = lock_data_dir(&data_dir)?;
        let registry = scan_data_dir(&data_dir)?;

        let state = Self(Arc::new(StateInner {
            data_dir,
            _dir_lock: dir_lock,
            cache_bytes,
            registry: RwLock::new(registry),
            clock: AtomicU64::new(0),
            embedder,
            default_semantic_floor: default_semantic_floor
                .unwrap_or(DEFAULT_SEMANTIC_FLOOR)
                .clamp(0.0, 1.0),
            cue_cache: Mutex::new(CueCache::default()),
            metrics: Metrics::default(),
            wal_enabled,
            wal_max_bytes,
            resident_estimate: AtomicI64::new(0),
            budget_ops: AtomicU64::new(0),
        }));
        state.preload_pinned();
        Ok(state)
    }

    /// Loads every pinned context now — serial and chatty on purpose:
    /// a boot that spends seconds loading pinned contexts should say
    /// what it is loading, not sit silent until "server ready".
    fn preload_pinned(&self) {
        for (name, entry) in self.snapshot() {
            let mut inner = entry.inner.write().unwrap();
            if !inner.meta.pinned {
                continue;
            }
            let preload_started = std::time::Instant::now();
            match ensure_hot(&self.0.data_dir, &name, &mut inner, &self.0.metrics) {
                Ok(()) => tracing::info!(
                    context = %name,
                    ms = preload_started.elapsed().as_millis() as u64,
                    "preloaded pinned context"
                ),
                Err(error) => tracing::warn!("pinned context '{name}' not preloaded: {error}"),
            }
        }
    }

    /// The shared observability registry — the HTTP middleware records
    /// into it, GET /metrics renders it.
    pub fn metrics(&self) -> &Metrics {
        &self.0.metrics
    }

    /// Counts one successful retrieval twice over: the aggregate
    /// searches family (by operation) and the context's own usage row.
    pub fn note_search(&self, op: crate::metrics::SearchOp, name: &str, empty: bool) {
        self.0.metrics.record_search(op, empty);
        self.note_read(name, empty);
    }

    /// Bumps a context's read counters — relaxed atomics only, so a
    /// read is counted without ever waiting on the entry lock. Unknown
    /// names (a delete racing the response) are silently skipped.
    pub fn note_read(&self, name: &str, empty: bool) {
        let Some(entry) = self.lookup(name) else {
            return;
        };
        entry.usage.reads.fetch_add(1, Ordering::Relaxed);
        if empty {
            entry.usage.empty_reads.fetch_add(1, Ordering::Relaxed);
        }
        entry
            .usage
            .last_read_epoch
            .store(unix_now(), Ordering::Relaxed);
        entry.usage_dirty.store(true, Ordering::Relaxed);
    }

    /// Bumps a context's write counter, same contract as
    /// [`AppState::note_read`].
    pub fn note_write(&self, name: &str) {
        let Some(entry) = self.lookup(name) else {
            return;
        };
        entry.usage.writes.fetch_add(1, Ordering::Relaxed);
        entry
            .usage
            .last_write_epoch
            .store(unix_now(), Ordering::Relaxed);
        entry.usage_dirty.store(true, Ordering::Relaxed);
    }

    /// Persists every usage snapshot the sidecars have not seen — the
    /// graceful-shutdown sweep behind the crash-loss contract on
    /// [`ContextUsage`]. Purely-read contexts never flush, so without
    /// this their counters would evaporate on every restart. Runs
    /// after the final [`AppState::flush_dirty`], so the stats written
    /// beside the counters are current.
    pub fn persist_usage(&self) {
        for (name, entry) in self.snapshot() {
            if !entry.usage_dirty.swap(false, Ordering::Relaxed) {
                continue;
            }
            let Some(guard) = entry.lock_unless_deleted() else {
                continue;
            };
            let outcome = write_meta(
                &self.0.data_dir,
                &file_stem(&name),
                &guard.meta,
                &guard.stats,
                &entry.usage.snapshot(),
            );
            if let Err(error) = outcome {
                entry.usage_dirty.store(true, Ordering::Relaxed);
                tracing::warn!("usage counters for '{name}' not persisted: {error}");
            }
        }
    }

    /// Point-in-time gauges for a scrape, computed from the registry
    /// so they cannot drift: how many contexts exist, how many are
    /// resident, and the resident-bytes estimate (loaded graphs plus
    /// cached vector stores — the same accounting the cache budget
    /// uses).
    pub fn gauge_snapshot(&self) -> GaugeSnapshot {
        let snapshot = self.snapshot();
        let contexts_registered = snapshot.len() as u64;
        let mut contexts_resident = 0u64;
        let mut resident_bytes = 0u64;
        let mut wal_bytes = 0u64;
        for (_, entry) in snapshot {
            let inner = entry.inner.read().unwrap();
            if let Slot::Hot(context) = &inner.slot {
                contexts_resident += 1;
                resident_bytes += context.footprint() as u64;
            }
            wal_bytes += inner.wal_bytes;
            drop(inner);
            resident_bytes += entry.vectors_footprint() as u64;
        }
        GaugeSnapshot {
            contexts_registered,
            contexts_resident,
            resident_bytes,
            wal_bytes,
        }
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
        // A name can be reused after a delete, and a delete that failed
        // partway (the name is unregistered first) or a half-restored
        // backup leaves the old generation's files behind. Nothing may
        // bleed into the new context — a stale WAL would even replay
        // the old generation's acknowledged writes into the fresh image
        // on its next cold load. Clear the slate before writing the
        // image: a crash in between leaves no image, so nothing
        // registers and the next attempt clears again; durability of
        // the unlinks rides on save_files' parent-directory fsync just
        // below. A leftover that cannot be removed fails the create —
        // registering on top of it would hand out a haunted context.
        let stem = file_stem(name);
        for stale in [
            wal_path(&self.0.data_dir, &stem),
            sources_path(&self.0.data_dir, &stem),
            vectors_path(&self.0.data_dir, &stem),
        ] {
            if let Err(error) = fs::remove_file(&stale)
                && error.kind() != io::ErrorKind::NotFound
            {
                return Err(CreateError::Io(error));
            }
        }
        let mut context = Context::default();
        context.set_dice_floor(meta.dice_floor);
        let stats = ContextStats::of(&context);
        let usage = ContextUsage::default();
        save_files(&self.0.data_dir, name, &meta, &stats, &usage, &context)
            .map_err(CreateError::Io)?;
        registry.insert(
            name.to_string(),
            Arc::new(Entry::new(
                meta,
                stats,
                Slot::Hot(Box::new(context)),
                0,
                usage,
            )),
        );
        Ok(())
    }

    /// Removes a context from the registry and deletes its files. The
    /// entry's lock is taken after the removal — waiting out any
    /// in-flight operation — and the slot becomes a tombstone under
    /// it: a flusher, evictor, or writer whose handle predates the
    /// removal finds [`Slot::Deleted`] when it finally locks, and
    /// backs off instead of recreating the files. Any unflushed writes
    /// are discarded — deletion destroys the context.
    pub fn delete(&self, name: &str) -> Option<io::Result<()>> {
        let entry = self.0.registry.write().unwrap().remove(name)?;
        let mut in_flight = entry.inner.write().unwrap();
        in_flight.slot = Slot::Deleted;
        self.recount_entry(&mut in_flight);
        entry.dirty.store(false, Ordering::Relaxed);
        // Lock order: `inner` before `vectors`, as documented on Entry.
        *entry.vectors.lock().unwrap() = None;
        let stem = file_stem(name);
        let mut outcome = Ok(());
        for file in [
            format!("{stem}.ctx"),
            format!("{stem}.meta.json"),
            format!("{stem}.sources.json"),
            format!("{stem}.vectors.bin"),
            format!("{stem}.wal.jsonl"),
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
        let _guard = entry.lock_unless_deleted()?;
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
        let sweep = |table: &VectorTable,
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
            Err(AccessError::Load(message)) | Err(AccessError::Unpersisted(message)) => {
                // Vectors were readable but the graph was not: serve the
                // unfiltered pairs and say why they are noisier. (A
                // read never yields Unpersisted; the arm is for the
                // type, not a path.)
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
        let op = WalOp::RetractSource {
            source: source.to_string(),
        };
        let touched = self.logged_write(name, std::slice::from_ref(&op), |context| {
            context.retract_source(source).unwrap_or(0)
        })?;

        let Some(entry) = self.lookup(name) else {
            // Raced with a delete; there is nothing left to clean up.
            return Ok((touched, false));
        };
        let Some(_guard) = entry.lock_unless_deleted() else {
            // Same race, one step later: the delete beat us to the lock.
            return Ok((touched, false));
        };
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
                    tracing::warn!("passage for '{source}' not removed from disk: {error}");
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
            let mut seen = std::collections::HashSet::new();
            passage_terms(query)
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
                let mut frequencies: HashMap<u64, f32> = HashMap::new();
                let mut length = 0f32;
                for gram in passage_terms(text) {
                    *frequencies.entry(gram).or_insert(0.0) += 1.0;
                    length += 1.0;
                }
                (source, text, frequencies, length)
            })
            .collect();
        let total = passages.len() as f32;
        let average_length =
            (passages.iter().map(|(.., length)| length).sum::<f32>() / total).max(1.0);
        // Document frequencies once per query term, not per (term ×
        // passage) pair — IDF only reads how many passages carry the term.
        let document_frequencies: HashMap<u64, f32> = query_grams
            .iter()
            .map(|&gram| {
                let carriers = passages
                    .iter()
                    .filter(|(_, _, frequencies, _)| frequencies.contains_key(&gram))
                    .count() as f32;
                (gram, carriers)
            })
            .collect();

        let mut scored: Vec<(String, f32, String)> = passages
            .iter()
            .map(|(source, text, frequencies, length)| {
                let mut score = 0f32;
                for gram in &query_grams {
                    let Some(&frequency) = frequencies.get(gram) else {
                        continue;
                    };
                    let document_frequency = document_frequencies[gram];
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
                "no embedding provider is configured (set TAGURU_EMBED_URL and TAGURU_EMBED_MODEL)"
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
            Err(AccessError::Load(message)) | Err(AccessError::Unpersisted(message)) => {
                return Some(Err(message));
            }
        };
        let (concepts, labels) = glosses;
        let entry = self.lookup(name)?;
        let path = vectors_path(&self.0.data_dir, &file_stem(name));

        // Diff and embed without holding the entry lock — provider round
        // trips can take seconds. Concurrent refreshes at worst re-embed
        // the same names; the merge below stays correct.
        let existing = VectorStore::load(&path);
        let fresh_model = existing.model != embedder.model();
        let embedded_concepts =
            match self.embed_stale(&*embedder, &existing.concepts, &concepts, fresh_model) {
                Ok(embedded) => embedded,
                Err(error) => return Some(Err(error)),
            };
        let embedded_labels =
            match self.embed_stale(&*embedder, &existing.labels, &labels, fresh_model) {
                Ok(embedded) => embedded,
                Err(error) => return Some(Err(error)),
            };
        let newly_embedded = embedded_concepts.len() + embedded_labels.len();

        // Merge under the entry lock so concurrent refreshes serialize
        // on the read-modify-write of the sidecar. A `None` means a
        // delete won the lock first: don't recreate the sidecar.
        let _guard = entry.lock_unless_deleted()?;
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

    /// Diffs one gloss table against its stored vectors and embeds what
    /// is new or changed, 128 glosses per provider call. Each vector
    /// remembers the hash of the gloss it came from; `fresh_model`
    /// marks everything stale.
    fn embed_stale(
        &self,
        embedder: &dyn EmbeddingProvider,
        stored: &VectorTable,
        entries: &[(String, String)],
        fresh_model: bool,
    ) -> Result<VectorTable, String> {
        let stale: Vec<(String, String, u64)> = entries
            .iter()
            .filter_map(|(name, gloss)| {
                let hash = fnv1a(gloss);
                let outdated =
                    fresh_model || stored.get(name).is_none_or(|&(hashed, _)| hashed != hash);
                outdated.then(|| (name.clone(), gloss.clone(), hash))
            })
            .collect();
        let mut embedded = VectorTable::new();
        for chunk in stale.chunks(128) {
            let texts: Vec<&str> = chunk.iter().map(|(_, gloss, _)| gloss.as_str()).collect();
            match embedder.embed(&texts, EmbedPurpose::Index) {
                Ok(vectors) => {
                    self.0.metrics.record_embed_refresh(true);
                    embedded.extend(
                        chunk
                            .iter()
                            .zip(vectors)
                            .map(|((name, _, hash), vector)| (name.clone(), (*hash, vector))),
                    );
                }
                Err(error) => {
                    self.0.metrics.record_embed_refresh(false);
                    return Err(error);
                }
            }
        }
        Ok(embedded)
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
        const SEMANTIC_LIMIT: usize = 5;

        let Some(embedder) = self.0.embedder.clone() else {
            return Some(Ok(Vec::new()));
        };
        let entry = self.lookup(name)?;
        // One-call override beats the context setting beats the server
        // default (see [`DEFAULT_SEMANTIC_FLOOR`] for the calibration).
        let context_floor = entry.inner.read().unwrap().meta.semantic_floor;
        let floor = floor_override
            .or(context_floor)
            .unwrap_or(self.0.default_semantic_floor)
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
                let vector = match embedder.embed(&[cue], EmbedPurpose::Query) {
                    Ok(mut vectors) => {
                        self.0.metrics.record_embed_resolve(true);
                        Arc::new(vectors.pop().unwrap_or_default())
                    }
                    Err(error) => {
                        self.0.metrics.record_embed_resolve(false);
                        return Some(Err(error));
                    }
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
            // A `None` means a delete won the lock first: don't
            // recreate the sidecar it just removed.
            let mut guard = entry.lock_unless_deleted()?;
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
                && let Err(error) = ensure_hot(&self.0.data_dir, name, inner, &self.0.metrics)
            {
                return Some(Err(io::Error::other(error)));
            }
            // A pin toggle moves the entry into or out of the budget's
            // world; the estimate must follow.
            self.recount_entry(inner);
            let inner = &*inner;
            write_meta(
                &self.0.data_dir,
                &file_stem(name),
                &inner.meta,
                &inner.stats,
                &entry.usage.snapshot(),
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
            .filter_map(|(name, entry)| describe_entry(name, &entry))
            .collect();
        directory.sort_by(|a, b| a.name.cmp(&b.name));
        directory
    }

    /// One directory row by name, or `None` for an unknown context.
    pub fn directory_entry(&self, name: &str) -> Option<DirectoryEntry> {
        let entry = self.lookup(name)?;
        describe_entry(name.to_string(), &entry)
    }

    /// Runs a read-only operation on one context, loading it first if
    /// cold. A hot context is served under the SHARED lock, so
    /// concurrent reads of one context run in parallel — a long explore
    /// no longer makes every recall on the same context queue behind
    /// it. Only a cold load (and every write) takes the exclusive path.
    pub fn read_context<T>(
        &self,
        name: &str,
        operate: impl FnOnce(&Context) -> T,
    ) -> Result<T, AccessError> {
        let entry = self.lookup(name).ok_or(AccessError::NotFound)?;
        // Fast path: already resident, shared lock, no exclusivity.
        {
            let inner = entry.inner.read().unwrap();
            match &inner.slot {
                Slot::Hot(context) => {
                    self.0.metrics.record_cache_hit();
                    let result = operate(context);
                    drop(inner);
                    self.touch(&entry);
                    self.enforce_budget(name);
                    return Ok(result);
                }
                Slot::Deleted => return Err(AccessError::NotFound),
                Slot::Cold => {}
            }
        }
        // Slow path: load under the exclusive lock. ensure_hot
        // re-checks the slot, so losing the load race to a concurrent
        // reader is fine — its load counts as ours.
        let result = {
            let mut inner = entry.lock_unless_deleted().ok_or(AccessError::NotFound)?;
            ensure_hot(&self.0.data_dir, name, &mut inner, &self.0.metrics)
                .map_err(AccessError::Load)?;
            self.recount_entry(&mut inner);
            let Slot::Hot(context) = &inner.slot else {
                unreachable!("ensure_hot leaves the slot hot");
            };
            operate(context)
        };
        self.touch(&entry);
        self.enforce_budget(name);
        Ok(result)
    }

    /// Runs a mutating operation on one context, loading it first if
    /// cold, and marks it dirty — the raw primitive under the tests.
    /// HTTP-reachable mutations go through [`AppState::logged_write`]
    /// instead so the WAL sees them; a test mutation was never
    /// acknowledged to any client, so it needs no log coverage and its
    /// durability is the flush it triggers.
    #[cfg(test)]
    pub fn write_context<T>(
        &self,
        name: &str,
        operate: impl FnOnce(&mut Context) -> T,
    ) -> Result<T, AccessError> {
        let entry = self.lookup(name).ok_or(AccessError::NotFound)?;
        let result = {
            let mut inner = entry.lock_unless_deleted().ok_or(AccessError::NotFound)?;
            ensure_hot(&self.0.data_dir, name, &mut inner, &self.0.metrics)
                .map_err(AccessError::Load)?;
            let Slot::Hot(context) = &mut inner.slot else {
                unreachable!("ensure_hot leaves the slot hot");
            };
            let result = operate(context);
            entry.dirty.store(true, Ordering::Relaxed);
            self.recount_entry(&mut inner);
            result
        };
        self.touch(&entry);
        self.enforce_budget(name);
        Ok(result)
    }

    /// Persists every dirty context and returns the names it flushed —
    /// the periodic flusher feeds those into the auto embedding refresh
    /// when that is enabled. Called once more on graceful shutdown; a
    /// failed save is retried on the next tick (the entry stays dirty).
    pub fn flush_dirty(&self) -> Vec<String> {
        let mut flushed = Vec::new();
        for (name, entry) in self.snapshot() {
            if self.flush_entry(&name, &entry) {
                flushed.push(name);
            }
        }
        flushed
    }

    /// One entry's flush. Split out of [`AppState::flush_dirty`] so the
    /// delete-race regression test can drive the exact window a real
    /// flusher hits — snapshot taken before a delete, entry lock taken
    /// after it.
    ///
    /// The image's disk work runs with the entry UNLOCKED: serialize a
    /// consistent snapshot under the lock, stage it (write + fsync,
    /// the megabytes half) without the lock, then re-take the lock to
    /// publish. Readers and writers of the context proceed while the
    /// bytes land; before this, every flush stalled them for the whole
    /// write.
    fn flush_entry(&self, name: &str, entry: &Entry) -> bool {
        // Claim the flag: one flusher per entry at a time, so
        // concurrent flush_dirty calls (a tick against the shutdown
        // flush) can never stage the same image twice.
        if !entry.dirty.swap(false, Ordering::Relaxed) {
            return false;
        }
        let (bytes, meta, stats, watermark) = {
            let mut guard = entry.inner.write().unwrap();
            let inner = &mut *guard;
            let watermark = inner.wal_seq - 1;
            // Cold has nothing to write; Deleted must write nothing —
            // that snapshot predates a delete and the files are gone
            // for good.
            let Slot::Hot(context) = &mut inner.slot else {
                return false;
            };
            // The image about to be written reflects everything logged
            // so far: bake that in as the watermark, and those WAL
            // records are replay-inert even if truncation below never
            // happens (crash, unwritable file — doesn't matter).
            context.set_applied_seq(watermark);
            let stats = ContextStats::of(context);
            (context.to_bytes(), inner.meta.clone(), stats, watermark)
        };

        let stem = file_stem(name);
        let image = image_path(&self.0.data_dir, &stem);
        let staged = match stage_bytes(&image, &bytes) {
            Ok(staged) => staged,
            Err(error) => {
                tracing::warn!("flush of context '{name}' failed (will retry): {error}");
                entry.dirty.store(true, Ordering::Relaxed);
                self.0.metrics.record_flush(false);
                return false;
            }
        };

        // Publication goes back under the lock: files for a name are
        // only ever created while holding its entry lock — the
        // tombstone invariant — so a delete that won the race while we
        // were staging must find us backing off, not recreating.
        let Some(mut guard) = entry.lock_unless_deleted() else {
            let _ = fs::remove_file(&staged);
            return false;
        };
        let inner = &mut *guard;
        // Claim the usage flag before snapshotting: an increment racing
        // this write either lands in the snapshot or re-marks the flag —
        // never both lost. (Advisory counters; a failed write re-marks.)
        entry.usage_dirty.store(false, Ordering::Relaxed);
        let outcome = commit_staged(&staged, &image).and_then(|()| {
            write_meta(
                &self.0.data_dir,
                &stem,
                &meta,
                &stats,
                &entry.usage.snapshot(),
            )
        });
        match outcome {
            Ok(()) => {
                inner.stats = stats;
                self.0.metrics.record_flush(true);
                // Truncation is only sound if the image just published
                // covers the whole log: a write that landed while the
                // bytes were in flight sits PAST our watermark, and its
                // records must survive until the next flush bakes them
                // into an image of their own.
                if inner.wal_seq - 1 == watermark {
                    self.truncate_wal(name, inner);
                } else {
                    entry.dirty.store(true, Ordering::Relaxed);
                }
                true
            }
            Err(error) => {
                tracing::warn!("flush of context '{name}' failed (will retry): {error}");
                let _ = fs::remove_file(&staged);
                entry.dirty.store(true, Ordering::Relaxed);
                entry.usage_dirty.store(true, Ordering::Relaxed);
                self.0.metrics.record_flush(false);
                false
            }
        }
    }

    /// Truncates a context's log once an image covering everything in
    /// it has published. Failure is harmless — the image's watermark
    /// already makes the logged records replay-inert — so it warns and
    /// moves on.
    fn truncate_wal(&self, name: &str, inner: &mut EntryInner) {
        match wal::reset(&wal_path(&self.0.data_dir, &file_stem(name))) {
            Ok(()) => inner.wal_bytes = 0,
            Err(error) => {
                tracing::warn!("WAL for '{name}' not truncated (harmless): {error}");
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

    /// Stamps the LRU clock on an entry after an operation.
    fn touch(&self, entry: &Entry) {
        entry.last_touch.store(
            self.0.clock.fetch_add(1, Ordering::Relaxed) + 1,
            Ordering::Relaxed,
        );
    }

    /// Recounts one entry's contribution to the resident estimate —
    /// absolute, not a delta, so repeated calls under the entry lock
    /// can never double-apply. Called wherever residency, size, or
    /// pinnedness can change: loads, writes, pin toggles, eviction,
    /// delete. Pinned entries count as zero — the budget covers
    /// unpinned residents only, and the gate must agree with the
    /// sweep on that or pinned-heavy deployments would sweep forever.
    fn recount_entry(&self, inner: &mut EntryInner) {
        let now = match (&inner.slot, inner.meta.pinned) {
            (Slot::Hot(context), false) => context.footprint(),
            _ => 0,
        };
        let before = inner.counted_bytes;
        inner.counted_bytes = now;
        self.0
            .resident_estimate
            .fetch_add(now as i64 - before as i64, Ordering::Relaxed);
    }

    /// The write path of the HTTP mutators: stage the whole batch in
    /// the context's WAL — one fsync, group commit at exactly the
    /// granularity the API already locks at — and only then run
    /// `operate` to apply it. An append that cannot be made durable
    /// refuses the write outright ([`AccessError::Unpersisted`],
    /// nothing applied): the client must never hold an acknowledgment
    /// the disk cannot replay. With the WAL disabled the staging step
    /// is skipped and durability falls back to the flush interval.
    fn logged_write<T>(
        &self,
        name: &str,
        ops: &[WalOp],
        operate: impl FnOnce(&mut Context) -> T,
    ) -> Result<T, AccessError> {
        let entry = self.lookup(name).ok_or(AccessError::NotFound)?;
        let result = {
            // Same tombstone rule as with_hot: a delete that beat us to
            // this lock owns the name — appending here would recreate
            // the WAL file it just removed.
            let mut inner = entry.lock_unless_deleted().ok_or(AccessError::NotFound)?;
            ensure_hot(&self.0.data_dir, name, &mut inner, &self.0.metrics)
                .map_err(AccessError::Load)?;
            if self.0.wal_enabled {
                // Backstop against unbounded growth: the log only
                // truncates after a successful image save, so a
                // persistently failing flush would grow it forever.
                // Past the cap, refuse writes — loudly — instead.
                if self.0.wal_max_bytes > 0 && inner.wal_bytes >= self.0.wal_max_bytes as u64 {
                    tracing::warn!(
                        context = %name,
                        wal_bytes = inner.wal_bytes,
                        cap = self.0.wal_max_bytes,
                        "WAL over its cap with the image failing to flush; write refused"
                    );
                    return Err(AccessError::Unpersisted(format!(
                        "the write-ahead log is at {} bytes (cap {}): the image has been \
                         failing to flush — check disk space and the server log",
                        inner.wal_bytes, self.0.wal_max_bytes
                    )));
                }
                let path = wal_path(&self.0.data_dir, &file_stem(name));
                match wal::append_batch(&path, inner.wal_seq, ops) {
                    Ok(appended) => {
                        self.0.metrics.record_wal_append(true);
                        inner.wal_bytes += appended;
                        inner.wal_seq += ops.len() as u64;
                    }
                    Err(error) => {
                        // The client sees the refusal; the operator
                        // must too — the core durability promise just
                        // failed to engage.
                        self.0.metrics.record_wal_append(false);
                        tracing::warn!(context = %name, %error, "WAL append failed; write refused");
                        return Err(AccessError::Unpersisted(error.to_string()));
                    }
                }
            }
            let Slot::Hot(context) = &mut inner.slot else {
                unreachable!("ensure_hot leaves the slot hot");
            };
            let result = operate(context);
            entry.dirty.store(true, Ordering::Relaxed);
            self.recount_entry(&mut inner);
            result
        };
        self.touch(&entry);
        self.enforce_budget(name);
        Ok(result)
    }

    /// Applies one document's extracted facts, staging them in the WAL
    /// first. `Ok(Err(PartialWrite))` reproduces the associations
    /// endpoint's historic partial semantics: items before the failing
    /// one are applied, each all-or-nothing in the library.
    pub fn add_associations(
        &self,
        name: &str,
        ops: Vec<AssocOp>,
    ) -> Result<Result<usize, PartialWrite>, AccessError> {
        let wal_ops: Vec<WalOp> = ops.into_iter().map(WalOp::Associate).collect();
        self.logged_write(name, &wal_ops, |context| apply_in_order(context, &wal_ops))
    }

    /// Registers alias batches (concepts then labels, in map order),
    /// staged in the WAL first — the same partial semantics as
    /// associations, with the conflict/capacity distinction preserved
    /// in [`PartialWrite::full`].
    pub fn add_aliases(
        &self,
        name: &str,
        concepts: &BTreeMap<String, String>,
        labels: &BTreeMap<String, String>,
    ) -> Result<Result<usize, PartialWrite>, AccessError> {
        let mut wal_ops = Vec::with_capacity(concepts.len() + labels.len());
        for (alias, canonical) in concepts {
            wal_ops.push(WalOp::AliasConcept {
                alias: alias.clone(),
                canonical: canonical.clone(),
            });
        }
        for (alias, canonical) in labels {
            wal_ops.push(WalOp::AliasLabel {
                alias: alias.clone(),
                canonical: canonical.clone(),
            });
        }
        self.logged_write(name, &wal_ops, |context| apply_in_order(context, &wal_ops))
    }

    /// Withdraws alias registrations (concept spellings then label
    /// spellings, in the order given), staged in the WAL first — the
    /// same partial semantics as every batch write. `Ok(Ok(n))`
    /// counts spellings withdrawn; canonical names and unknown
    /// spellings are refused as conflicts, never applied silently.
    pub fn remove_aliases(
        &self,
        name: &str,
        concepts: &[String],
        labels: &[String],
    ) -> Result<Result<usize, PartialWrite>, AccessError> {
        let mut wal_ops = Vec::with_capacity(concepts.len() + labels.len());
        for alias in concepts {
            wal_ops.push(WalOp::UnaliasConcept {
                alias: alias.clone(),
            });
        }
        for alias in labels {
            wal_ops.push(WalOp::UnaliasLabel {
                alias: alias.clone(),
            });
        }
        self.logged_write(name, &wal_ops, |context| apply_in_order(context, &wal_ops))
    }

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

    /// Evicts least-recently-used, unpinned, hot contexts until their
    /// resident estimate fits the budget. `except` (the context just
    /// used) is never evicted, so a single oversized context cannot
    /// thrash. Dirty contexts are persisted before eviction; if that
    /// save fails they stay resident rather than losing writes.
    fn enforce_budget(&self, except: &str) {
        // Cheap gate in front of the O(contexts) sweep below: one
        // atomic load of the graph estimate, which per-entry recounts
        // keep exact. Vector-store residency is NOT tracked between
        // sweeps, so every 64th operation forces a real sweep anyway —
        // its reconciling store below bounds any staleness or drift at
        // 64 operations. Before this gate the full sweep (snapshot,
        // two lock acquisitions per context) ran on every request.
        let ops = self.0.budget_ops.fetch_add(1, Ordering::Relaxed);
        let budget = i64::try_from(self.0.cache_bytes).unwrap_or(i64::MAX);
        if !ops.is_multiple_of(64) && self.0.resident_estimate.load(Ordering::Relaxed) <= budget {
            return;
        }

        let mut candidates: Vec<(u64, usize, String, Arc<Entry>)> = Vec::new();
        let mut total = 0usize;
        for (name, entry) in self.snapshot() {
            let inner = entry.inner.read().unwrap();
            if inner.meta.pinned {
                continue;
            }
            let resident = match &inner.slot {
                Slot::Hot(context) => context.footprint(),
                // A deleted entry holds nothing: the tombstone dropped
                // the graph and the delete cleared the vectors.
                Slot::Cold | Slot::Deleted => 0,
            };
            drop(inner);
            // Cached vector stores count too — a cold entry can hold a
            // large one after a semantic query.
            let bytes = resident + entry.vectors_footprint();
            if bytes == 0 {
                continue;
            }
            total += bytes;
            candidates.push((entry.last_touch.load(Ordering::Relaxed), bytes, name, entry));
        }
        // Reconcile the gate with measured reality — vectors included,
        // and any drift folded away.
        self.0
            .resident_estimate
            .store(total as i64, Ordering::Relaxed);
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
            if self.evict_entry(&name, &entry) {
                total = total.saturating_sub(bytes);
            }
        }
        self.0
            .resident_estimate
            .store(total as i64, Ordering::Relaxed);
    }

    /// One entry's eviction, everything under its write lock: persist
    /// if dirty, drop the graph, clear the cached vectors. `false`
    /// means nothing was freed — the entry got pinned since the
    /// caller's sweep, or its save failed (it stays resident rather
    /// than losing writes).
    fn evict_entry(&self, name: &str, entry: &Entry) -> bool {
        let mut guard = entry.inner.write().unwrap();
        let inner = &mut *guard;
        // Re-check under the write lock; the entry may have changed
        // between the snapshot and now.
        if inner.meta.pinned {
            return false;
        }
        let watermark = inner.wal_seq - 1;
        if let Slot::Hot(context) = &mut inner.slot {
            if entry.dirty.load(Ordering::Relaxed) {
                context.set_applied_seq(watermark);
                let stats = ContextStats::of(context);
                if let Err(error) = save_files(
                    &self.0.data_dir,
                    name,
                    &inner.meta,
                    &stats,
                    &entry.usage.snapshot(),
                    context,
                ) {
                    tracing::warn!(
                        "context '{name}' stays resident, eviction save failed: {error}"
                    );
                    self.0.metrics.record_eviction(false);
                    return false;
                }
                inner.stats = stats;
                entry.dirty.store(false, Ordering::Relaxed);
                self.truncate_wal(name, inner);
            } else {
                inner.stats = ContextStats::of(context);
            }
            inner.slot = Slot::Cold;
            // Local zero only: the caller's absolute store settles the
            // global, so a recount's delta would double-count.
            inner.counted_bytes = 0;
            self.0.metrics.record_eviction(true);
        }
        *entry.vectors.lock().unwrap() = None;
        true
    }
}

/// One directory row, or `None` when the entry was deleted between the
/// caller's snapshot/lookup and this lock.
fn describe_entry(name: String, entry: &Entry) -> Option<DirectoryEntry> {
    let inner = entry.inner.read().unwrap();
    let (loaded, stats) = match &inner.slot {
        Slot::Hot(context) => (true, ContextStats::of(context)),
        Slot::Cold => (false, inner.stats.clone()),
        Slot::Deleted => return None,
    };
    Some(DirectoryEntry {
        name,
        description: inner.meta.description.clone(),
        pinned: inner.meta.pinned,
        loaded,
        dice_floor: inner.meta.dice_floor,
        semantic_floor: inner.meta.semantic_floor,
        stats,
        usage: entry.usage.snapshot(),
    })
}

/// One boot-time pass over the data directory: crash leftovers of
/// staged writes are deleted (never published, and nothing may linger
/// as unbounded disk litter), and every context image found is
/// registered cold, described by its sidecar snapshot.
fn scan_data_dir(data_dir: &Path) -> io::Result<HashMap<String, Arc<Entry>>> {
    let mut registry = HashMap::new();
    for dir_entry in fs::read_dir(data_dir)? {
        let path = dir_entry?.path();
        let extension = path.extension().and_then(|e| e.to_str());
        if extension.is_some_and(|e| e.starts_with("tmp")) {
            let _ = fs::remove_file(&path);
            continue;
        }
        if extension != Some("ctx") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(name) = name_from_stem(stem) else {
            tracing::warn!("skipping {}: file name does not decode", path.display());
            continue;
        };
        let MetaFile { meta, stats, usage } = read_meta_file(data_dir, stem);
        // The gauge must see leftover logs from the first scrape, not
        // only after each context's first touch.
        let wal_bytes = fs::metadata(wal_path(data_dir, stem))
            .map(|meta| meta.len())
            .unwrap_or(0);
        registry.insert(
            name,
            Arc::new(Entry::new(meta, stats, Slot::Cold, wal_bytes, usage)),
        );
    }
    Ok(registry)
}

/// Loads the image behind a cold slot and replays whatever the WAL
/// holds above the image's watermark; hot slots pass through. On
/// success the slot is hot, the stats are fresh, and `wal_seq`
/// continues from the replay's tail. Every call lands in the cache
/// metrics: hot is a hit, a Cold→Hot attempt is a load.
fn ensure_hot(
    data_dir: &Path,
    name: &str,
    inner: &mut EntryInner,
    metrics: &Metrics,
) -> Result<(), String> {
    if matches!(inner.slot, Slot::Hot(_)) {
        metrics.record_cache_hit();
        return Ok(());
    }
    // Callers check the tombstone under this same lock before calling;
    // seeing one here means a caller forgot.
    if matches!(inner.slot, Slot::Deleted) {
        return Err(format!("context '{name}' is deleted"));
    }
    let stem = file_stem(name);
    let loaded = fs::read(image_path(data_dir, &stem))
        .map_err(|e| format!("context '{name}' image unreadable: {e}"))
        .and_then(|bytes| {
            Context::from_bytes(&bytes).map_err(|e| format!("context '{name}' image corrupt: {e}"))
        })
        .and_then(|mut context| {
            // Replay runs whether or not the WAL is currently enabled:
            // a log left behind by an earlier run holds acknowledged
            // writes and must never be ignored. A corrupt log is the
            // image-corrupt severity, not a shrug — it holds writes
            // that exist nowhere else.
            let (ops, top) = wal::replay(&wal_path(data_dir, &stem), context.applied_seq())
                .map_err(|e| format!("context '{name}' WAL unreadable: {e}"))?;
            for op in ops {
                replay_op(&mut context, &op);
            }
            Ok((context, top))
        });
    let (mut context, top) = match loaded {
        Ok(loaded) => loaded,
        Err(error) => {
            metrics.record_cache_load(false);
            return Err(error);
        }
    };
    metrics.record_cache_load(true);
    // The image carries knowledge only; tuning config lives in the
    // sidecar and is re-applied on every load.
    context.set_dice_floor(inner.meta.dice_floor);
    inner.stats = ContextStats::of(&context);
    inner.slot = Slot::Hot(Box::new(context));
    inner.wal_seq = top + 1;
    // Re-stat rather than trust the registration-time size: appends
    // and truncations may have happened while this entry sat cold.
    inner.wal_bytes = fs::metadata(wal_path(data_dir, &stem))
        .map(|meta| meta.len())
        .unwrap_or(0);
    Ok(())
}

/// Applies ops front to back, stopping at the first rejection — the
/// batch endpoints' historic partial semantics: everything before the
/// failing item stays applied.
fn apply_in_order(context: &mut Context, ops: &[WalOp]) -> Result<usize, PartialWrite> {
    let mut applied = 0usize;
    for op in ops {
        if let Err((message, full)) = apply_op(context, op) {
            return Err(PartialWrite {
                applied,
                message,
                full,
            });
        }
        applied += 1;
    }
    Ok(applied)
}

/// Re-applies one replayed op. A deterministic library rejection here
/// is the same rejection the original write already reported to its
/// client — replay reruns the op on the exact state the original saw
/// — so it is logged, never fatal.
fn replay_op(context: &mut Context, op: &WalOp) {
    if let Err((message, _)) = apply_op(context, op) {
        tracing::warn!("WAL replay skipped an op (same rejection as the original): {message}");
    }
}

/// Applies one op to the graph; `Err` carries the human message each
/// op family has always reported through the API, plus whether it was
/// a capacity error.
fn apply_op(context: &mut Context, op: &WalOp) -> Result<(), (String, bool)> {
    match op {
        WalOp::Associate(op) => {
            let result = match &op.source {
                Some(source) => context.associate_from(
                    op.subject.as_str(),
                    op.label.as_str(),
                    op.object.as_str(),
                    op.weight,
                    source.as_str(),
                ),
                None => context.associate(
                    op.subject.as_str(),
                    op.label.as_str(),
                    op.object.as_str(),
                    op.weight,
                ),
            };
            result.map_err(|full| (full.to_string(), true))
        }
        WalOp::AliasConcept { alias, canonical } => context
            .add_concept_alias(alias.as_str(), canonical)
            .map_err(|error| {
                (
                    format!("concept alias '{alias}' → '{canonical}': {error}"),
                    matches!(error, AliasError::Full(_)),
                )
            }),
        WalOp::AliasLabel { alias, canonical } => context
            .add_label_alias(alias.as_str(), canonical)
            .map_err(|error| {
                (
                    format!("label alias '{alias}' → '{canonical}': {error}"),
                    matches!(error, AliasError::Full(_)),
                )
            }),
        // A withdrawal of an absent spelling is a client mistake on
        // the live path (409, like every conflict), and on replay the
        // usual logged skip. Never a capacity error: removal frees.
        WalOp::UnaliasConcept { alias } => match context.remove_concept_alias(alias) {
            Some(_) => Ok(()),
            None => Err((
                format!(
                    "'{alias}' is not a concept alias (canonical names cannot be \
                     removed this way)"
                ),
                false,
            )),
        },
        WalOp::UnaliasLabel { alias } => match context.remove_label_alias(alias) {
            Some(_) => Ok(()),
            None => Err((
                format!(
                    "'{alias}' is not a label alias (canonical names cannot be \
                     removed this way)"
                ),
                false,
            )),
        },
        WalOp::RetractSource { source } => {
            context.retract_source(source);
            Ok(())
        }
    }
}

pub(crate) fn image_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.ctx"))
}

pub(crate) fn meta_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.meta.json"))
}

pub(crate) fn sources_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.sources.json"))
}

pub(crate) fn vectors_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.vectors.bin"))
}

pub(crate) fn wal_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.wal.jsonl"))
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x1_0000_01b3;

/// The terms of one passage or query: [`text_terms`] over the
/// normalized text, plus a word term per piece of every camelCase run
/// in the RAW text — normalization lowercases away exactly the
/// boundaries that let `state` reach `AppState`, so the split must
/// read the original. One function serves both sides of the search,
/// so they cannot disagree about what a term is.
fn passage_terms(raw: &str) -> Vec<u64> {
    let mut terms = text_terms(&taguru::context::normalize_entry(raw));
    let mut run: Vec<char> = Vec::new();
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() {
            run.push(ch);
        } else {
            camel_pieces(&run, &mut terms);
            run.clear();
        }
    }
    camel_pieces(&run, &mut terms);
    terms
}

/// Appends one lowercased word term per piece of an ASCII run that
/// splits at case boundaries: `aB` → `a|B`, digits stick to their
/// piece (`U64Max` → `u64|max`), and an acronym ends before its last
/// capital (`HTTPServer` → `http|server`). A run with no boundary
/// appends nothing — its whole-word term is already in the stream.
/// Pieces hash exactly like [`text_terms`] words, so a piece matches
/// wherever the same word occurs standalone.
fn camel_pieces(run: &[char], terms: &mut Vec<u64>) {
    let mut starts = vec![0];
    for at in 1..run.len() {
        if !run[at].is_ascii_uppercase() {
            continue;
        }
        let after_lower = run[at - 1].is_ascii_lowercase() || run[at - 1].is_ascii_digit();
        let ends_acronym = run[at - 1].is_ascii_uppercase()
            && run.get(at + 1).is_some_and(|ch| ch.is_ascii_lowercase());
        if after_lower || ends_acronym {
            starts.push(at);
        }
    }
    if starts.len() < 2 {
        return;
    }
    starts.push(run.len());
    for window in starts.windows(2) {
        let mut word = FNV_OFFSET;
        for ch in &run[window[0]..window[1]] {
            word ^= ch.to_ascii_lowercase() as u64;
            word = word.wrapping_mul(FNV_PRIME);
        }
        terms.push(word | 1 << 63);
    }
}

/// The word/bigram layer under [`passage_terms`], as u64 keys.
/// ASCII-alphanumeric runs count as whole words; everything else
/// contributes adjacent character pairs within its run (a run of one
/// contributes the lone character). Space-delimited languages need
/// word terms — character pairs occur in every English document alike,
/// which flattens IDF to nothing — while undelimited Japanese needs
/// the bigrams. Runs break at spaces and punctuation, and a script
/// switch breaks the run too, so terms never straddle "第10篇"-style
/// boundaries.
fn text_terms(text: &str) -> Vec<u64> {
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
            tracing::warn!("ignoring corrupt passages at {}: {error}", path.display());
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
    usage: &ContextUsage,
    context: &Context,
) -> io::Result<()> {
    let stem = file_stem(name);
    write_atomic(&image_path(dir, &stem), &context.to_bytes())?;
    write_meta(dir, &stem, meta, stats, usage)
}

fn write_meta(
    dir: &Path,
    stem: &str,
    meta: &ContextMeta,
    stats: &ContextStats,
    usage: &ContextUsage,
) -> io::Result<()> {
    let file = MetaFile {
        meta: meta.clone(),
        stats: stats.clone(),
        usage: usage.clone(),
    };
    write_atomic(&meta_path(dir, stem), &serde_json::to_vec_pretty(&file)?)
}

/// Reads the sidecar, falling back to defaults on any problem — a
/// missing or corrupt sidecar must not make the image unreachable.
fn read_meta_file(dir: &Path, stem: &str) -> MetaFile {
    match fs::read(meta_path(dir, stem)) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|error| {
            tracing::warn!("ignoring corrupt sidecar for '{stem}': {error}");
            MetaFile::default()
        }),
        Err(_) => MetaFile::default(),
    }
}

/// Writes via a temporary file, fsync, and rename — a crash mid-write
/// leaves the previous version intact, and power loss after return
/// cannot tear or lose the new one. The rename itself is an entry in
/// the parent directory's own data, so the parent is fsynced too;
/// without that a crash can forget the rename even though the file
/// contents reached disk.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let staged = stage_bytes(path, bytes)?;
    commit_staged(&staged, path)
}

/// A unique per-process staging name beside `path`. Concurrent
/// stagers (the flusher against an eviction, a shutdown flush against
/// a tick) must never write the same temporary file — with a fixed
/// name, one truncates the other mid-write and a torn image gets
/// renamed into place. Leftovers from a crash are swept at boot.
fn staging_path(path: &Path) -> PathBuf {
    static STAGING_NONCE: AtomicU64 = AtomicU64::new(0);
    let nonce = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
    path.with_extension(format!("tmp{nonce}"))
}

/// The heavy half of [`write_atomic`]: writes and fsyncs `bytes` under
/// a staging name beside `path`. Safe to run without any lock — the
/// file is invisible until [`commit_staged`] publishes it.
fn stage_bytes(path: &Path, bytes: &[u8]) -> io::Result<PathBuf> {
    use std::io::Write;

    let staged = staging_path(path);
    let mut file = fs::File::create(&staged)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(staged)
}

/// The cheap half of [`write_atomic`]: atomically publishes a staged
/// file at its final path — rename plus parent-directory fsync.
fn commit_staged(staged: &Path, path: &Path) -> io::Result<()> {
    fs::rename(staged, path)?;
    fsync_parent_dir(path)
}

/// Takes the advisory exclusive lock (`.taguru.lock`) that admits one
/// registry per data directory at a time: `taguru serve` and `taguru
/// import` both boot through here, so whichever is second gets a
/// refusal naming the conflict instead of a silent last-flush-wins
/// overwrite. The lock lives on the open descriptor, not the file —
/// a crash releases it with the process, and the empty lock file
/// left behind means nothing. Advisory: it binds taguru processes,
/// not arbitrary tools, and network filesystems honor it unreliably.
fn lock_data_dir(dir: &Path) -> io::Result<fs::File> {
    let file = fs::File::create(dir.join(".taguru.lock"))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(fs::TryLockError::WouldBlock) => Err(io::Error::other(format!(
            "data directory {} is held by another taguru process \
             (a running serve, or an import) — stop that one first",
            dir.display()
        ))),
        Err(fs::TryLockError::Error(error)) => Err(error),
    }
}

/// Persists a rename or file creation by syncing the directory that
/// holds the entry. Unix-only; elsewhere the rename stays atomic
/// against a crash mid-write, just not durable against power loss —
/// unix is what this server targets.
pub(crate) fn fsync_parent_dir(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        fs::File::open(parent)?.sync_all()?;
    }
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
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
pub(crate) fn name_from_stem(stem: &str) -> Option<String> {
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
        let dir =
            std::env::temp_dir().join(format!("taguru-registry-{tag}-{}", std::process::id()));
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

    /// The rendered /metrics body — the public read surface every
    /// counter assertion goes through.
    fn rendered(state: &AppState) -> String {
        state.metrics().render_prometheus(&state.gauge_snapshot())
    }

    #[test]
    fn meta_sidecar_without_usage_field_loads_with_zeroed_counters() {
        // A sidecar written before usage counters existed.
        let json = br#"{"description":"d","pinned":false,"stats":{"associations":3}}"#;
        let file: MetaFile = serde_json::from_slice(json).unwrap();
        assert_eq!(file.stats.associations, 3);
        assert_eq!(file.usage.reads, 0);
        assert_eq!(file.usage.last_read_epoch, 0);
    }

    #[test]
    fn usage_notes_accumulate_and_survive_a_reboot_via_the_shutdown_sweep() {
        let dir = scratch_dir("usage-sweep");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state.note_read("sake", false);
            state.note_read("sake", true);
            state.note_write("sake");
            let usage = state.directory_entry("sake").unwrap().usage;
            assert_eq!((usage.reads, usage.empty_reads, usage.writes), (2, 1, 1));
            assert!(usage.last_read_epoch > 0);
            assert!(usage.last_write_epoch > 0);
            // Nothing marked the graph dirty since create, so no flush
            // will run: the sweep alone must put the counters on disk.
            state.persist_usage();
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let usage = state.directory_entry("sake").unwrap().usage;
        assert_eq!((usage.reads, usage.empty_reads, usage.writes), (2, 1, 1));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_data_directory_admits_one_registry_at_a_time() {
        let dir = scratch_dir("dir-lock");
        let holder = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        // flock-style locks are per descriptor, so a second registry in
        // the SAME process is refused exactly as a second process would
        // be — which is what lets one test prove the contract.
        let error = AppState::boot(dir.clone(), usize::MAX, None)
            .map(|_| ())
            .unwrap_err();
        assert!(
            error.to_string().contains("another taguru process"),
            "{error}"
        );
        // The lock dies with its holder; the directory is reusable.
        drop(holder);
        let _reborn = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn ensure_hot_records_hits_and_loads() {
        let dir = scratch_dir("m-cache");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state.flush_dirty();
        }

        // A fresh boot leaves the context cold: the first read loads
        // from disk, the second is a pure hit.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert!(rendered(&state).contains("taguru_cache_loads_total{outcome=\"ok\"} 1"));
        state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert!(rendered(&state).contains("taguru_cache_hits_total 1"));

        // Gauges come from the registry itself.
        assert!(rendered(&state).contains("taguru_contexts_registered 1"));
        assert!(rendered(&state).contains("taguru_contexts_resident 1"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn flush_and_eviction_record_their_outcomes() {
        let dir = scratch_dir("m-flush");
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
        // Touching b evicts a under the one-byte budget.
        state
            .read_context("b", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        let body = rendered(&state);
        assert!(
            body.contains("taguru_cache_evictions_total{outcome=\"ok\"} 1"),
            "{body}"
        );

        state
            .write_context("b", |context| {
                context.associate("用語", "意味", "定義", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();
        state.flush_dirty();
        assert!(
            rendered(&state).contains("taguru_flush_total{outcome=\"ok\"} 1"),
            "a flushed dirty context must count"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn embedding_calls_record_success_and_failure() {
        /// Same model name as the mock, so stored vectors stay usable,
        /// but every provider round trip fails.
        struct FailingEmbeddings;
        impl EmbeddingProvider for FailingEmbeddings {
            fn model(&self) -> &str {
                "mock"
            }
            fn embed(
                &self,
                _texts: &[&str],
                _purpose: EmbedPurpose,
            ) -> Result<Vec<Vec<f32>>, String> {
                Err("provider down".to_string())
            }
        }

        let dir = scratch_dir("m-embed");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        {
            let embedder =
                Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
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
            // One batch per namespace: two successful provider calls.
            assert!(rendered(&state).contains(
                "taguru_embedding_requests_total{operation=\"refresh\",outcome=\"ok\"} 2"
            ));
            state.flush_dirty();
        }

        // Same data, failing provider: the resolve-path cue embedding
        // fails and is counted as such.
        let embedder = Some(Arc::new(FailingEmbeddings) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        assert!(
            state
                .semantic_resolve("fruit", "アップル", false, None)
                .unwrap()
                .is_err()
        );
        let body = rendered(&state);
        assert!(
            body.contains(
                "taguru_embedding_requests_total{operation=\"resolve\",outcome=\"failed\"} 1"
            ),
            "{body}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    fn assoc_op(
        subject: &str,
        label: &str,
        object: &str,
        weight: f64,
        source: Option<&str>,
    ) -> AssocOp {
        AssocOp {
            subject: subject.to_string(),
            label: label.to_string(),
            object: object.to_string(),
            weight,
            source: source.map(String::from),
        }
    }

    #[test]
    fn unflushed_writes_survive_a_process_restart_via_the_wal() {
        let dir = scratch_dir("wal-restart");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op(
                        "青嶺酒造",
                        "創業年",
                        "1907年",
                        1.0,
                        Some("第1段落"),
                    )],
                )
                .unwrap()
                .unwrap();
            // NO flush_dirty: dropping the state here is the crash.
            // The 5-second window would have eaten this write; the WAL
            // must not.
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let recalled = state
            .read_context("sake", |context| context.recall("青嶺酒造"))
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(recalled.len(), 1, "the acknowledged write must survive");
        assert_eq!(recalled[0].object, "1907年");
        assert_eq!(
            recalled[0].attributions.len(),
            1,
            "attributions ride the WAL too"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_flush_whose_snapshot_predates_a_delete_does_not_resurrect_the_files() {
        let dir = scratch_dir("delete-vs-flush");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("victim", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "victim",
                vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
            )
            .unwrap()
            .unwrap();

        // The flusher's world an instant before the delete: entry Arcs
        // cloned out of the registry, the victim still dirty.
        let stale = state.snapshot();
        state.delete("victim").unwrap().unwrap();

        // Even if a stale handle re-marks the entry dirty (delete does
        // clear the flag, but that is an optimization), the tombstone
        // is what must hold.
        for (_, entry) in &stale {
            entry.dirty.store(true, Ordering::Relaxed);
        }
        // The flusher arrives late and works through its stale snapshot.
        for (name, entry) in &stale {
            assert!(
                !state.flush_entry(name, entry),
                "a deleted context must not flush"
            );
        }

        let stem = file_stem("victim");
        for suffix in [
            "ctx",
            "meta.json",
            "sources.json",
            "vectors.bin",
            "wal.jsonl",
        ] {
            let path = dir.join(format!("{stem}.{suffix}"));
            assert!(
                !path.exists(),
                "{} came back after the delete",
                path.display()
            );
        }
        // The resurrection a user would see: a reboot re-registering
        // it. A real reboot means the old process is gone — and the
        // directory lock with it.
        drop(state);
        let reborn = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert_eq!(
            reborn.context_count(),
            0,
            "the deleted context re-registered on boot"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn create_does_not_inherit_files_left_by_an_earlier_generation() {
        let dir = scratch_dir("create-clean-slate");
        fs::create_dir_all(&dir).unwrap();
        let stem = file_stem("sake");
        // Litter an earlier generation can leave when its delete fails
        // partway (the name is unregistered first) or when files are
        // restored by hand: an acknowledged-write log, passages,
        // vectors — but no image, so nothing registers at boot.
        wal::append_batch(
            &wal_path(&dir, &stem),
            1,
            &[WalOp::Associate(assoc_op(
                "幽霊",
                "正体",
                "枯れ尾花",
                1.0,
                None,
            ))],
        )
        .unwrap();
        fs::write(sources_path(&dir, &stem), br#"{"ghost":"old passage"}"#).unwrap();
        fs::write(vectors_path(&dir, &stem), b"stale").unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert_eq!(state.context_count(), 0, "no image, nothing registers");
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        assert!(
            !sources_path(&dir, &stem).exists(),
            "stale passages survived the create"
        );
        assert!(
            !vectors_path(&dir, &stem).exists(),
            "stale vectors survived the create"
        );
        drop(state);

        // The reboot is where inheritance would bite: a cold load
        // replays whatever the WAL holds above the fresh image's
        // watermark 0.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let recalled = state
            .read_context("sake", |context| context.recall("幽霊"))
            .map_err(|_| "read")
            .unwrap();
        assert!(
            recalled.is_empty(),
            "the old generation's WAL replayed into the new context"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn wal_growth_is_visible_and_the_cap_refuses_further_writes() {
        let dir = scratch_dir("wal-gauge");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            assert_eq!(state.gauge_snapshot().wal_bytes, 0);
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
                )
                .unwrap()
                .unwrap();
            let grown = state.gauge_snapshot().wal_bytes;
            assert!(grown > 0, "an append must show in the gauge");
            assert!(rendered(&state).contains("taguru_wal_appends_total{outcome=\"ok\"} 1"));
            assert!(rendered(&state).contains(&format!("taguru_wal_bytes {grown}")));
            state.flush_dirty();
            assert_eq!(
                state.gauge_snapshot().wal_bytes,
                0,
                "truncation must zero the gauge"
            );

            // Leave an unflushed write behind for the reboot check.
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("青嶺酒造", "杜氏", "高瀬", 1.0, None)],
                )
                .unwrap()
                .unwrap();
        }

        // Registration alone — no touch — must already see the
        // leftover log, or the first scrapes after a reboot lie.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(
            state.gauge_snapshot().wal_bytes > 0,
            "boot must stat leftover logs"
        );

        // A tiny cap: the first write passes (the log is empty), the
        // second is refused — the backstop for a flush that never
        // succeeds again.
        let capped_dir = scratch_dir("wal-capped");
        let state =
            AppState::boot_with(capped_dir.clone(), usize::MAX, None, true, 1, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations("sake", vec![assoc_op("a", "l", "b", 1.0, None)])
            .unwrap()
            .unwrap();
        let refused = state.add_associations("sake", vec![assoc_op("c", "l", "d", 1.0, None)]);
        assert!(
            matches!(refused, Err(AccessError::Unpersisted(_))),
            "over the cap the write must be refused: {refused:?}"
        );

        let _ = fs::remove_dir_all(dir);
        let _ = fs::remove_dir_all(capped_dir);
    }

    #[test]
    #[cfg(unix)]
    fn health_follows_the_flusher_down_and_back_up() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("health");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "sake",
                vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
            )
            .unwrap()
            .unwrap();
        state.flush_dirty();
        assert!(state.metrics().flush_is_healthy());
        let stamped = state.metrics().last_flush_success_epoch();
        assert!(stamped > 0, "a successful flush must stamp the gauge");
        assert!(rendered(&state).contains(&format!(
            "taguru_last_flush_success_timestamp_seconds {stamped}"
        )));

        // The disk goes bad: the next flush fails, health turns with it.
        state
            .add_associations(
                "sake",
                vec![assoc_op("青嶺酒造", "杜氏", "高瀬", 1.0, None)],
            )
            .unwrap()
            .unwrap();
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
        assert!(state.flush_dirty().is_empty());
        assert!(!state.metrics().flush_is_healthy());

        // The disk recovers: the next tick heals the signal.
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(state.flush_dirty(), vec!["sake".to_string()]);
        assert!(state.metrics().flush_is_healthy());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn writes_racing_the_flusher_survive_a_restart() {
        use std::thread;

        let dir = scratch_dir("flush-race");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            let writer = {
                let state = state.clone();
                thread::spawn(move || {
                    for index in 0..100 {
                        state
                            .add_associations(
                                "sake",
                                vec![assoc_op(&format!("s{index}"), "l", "o", 1.0, None)],
                            )
                            .unwrap()
                            .unwrap();
                    }
                })
            };
            // Flush continuously while the writes land, so staging and
            // publication interleave with appends every which way. The
            // dangerous outcome is a truncation eating a record the
            // published image does not contain.
            while !writer.is_finished() {
                state.flush_dirty();
            }
            writer.join().unwrap();
            // No final flush: the drop is the crash, and some tail of
            // writes lives only in the WAL.
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let count = state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(count, 100, "every acknowledged write must survive");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn the_budget_gate_tracks_loads_writes_pins_and_deletes() {
        let dir = scratch_dir("estimate");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "sake",
                vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
            )
            .unwrap()
            .unwrap();
        let after_write = state.0.resident_estimate.load(Ordering::Relaxed);
        assert!(after_write > 0, "a resident written context must count");

        // Pinning moves it out of the budget's world…
        state
            .update_meta("sake", None, Some(true), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(state.0.resident_estimate.load(Ordering::Relaxed), 0);
        // …and unpinning brings it back.
        state
            .update_meta("sake", None, Some(false), None, None)
            .unwrap()
            .unwrap();
        assert!(state.0.resident_estimate.load(Ordering::Relaxed) >= after_write);

        state.delete("sake").unwrap().unwrap();
        assert_eq!(state.0.resident_estimate.load(Ordering::Relaxed), 0);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn concurrent_reads_of_one_hot_context_do_not_serialize() {
        use std::sync::atomic::AtomicUsize;
        use std::thread;
        use std::time::Duration;

        let dir = scratch_dir("read-parallel");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "sake",
                vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
            )
            .unwrap()
            .unwrap();

        let in_read = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut readers = Vec::new();
        for _ in 0..2 {
            let state = state.clone();
            let in_read = Arc::clone(&in_read);
            let peak = Arc::clone(&peak);
            readers.push(thread::spawn(move || {
                state
                    .read_context("sake", |context| {
                        let now = in_read.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(now, Ordering::SeqCst);
                        // Long enough that the two readers MUST overlap
                        // unless one lock is excluding the other.
                        thread::sleep(Duration::from_millis(150));
                        in_read.fetch_sub(1, Ordering::SeqCst);
                        context.association_count()
                    })
                    .map_err(|_| "read")
                    .unwrap();
            }));
        }
        for reader in readers {
            reader.join().unwrap();
        }
        assert_eq!(
            peak.load(Ordering::SeqCst),
            2,
            "two readers must be inside one hot context at the same time"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn every_latecomer_behind_a_delete_finds_the_tombstone() {
        let dir = scratch_dir("delete-tombstone");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("victim", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let (_, stale) = state
            .snapshot()
            .into_iter()
            .find(|(name, _)| name == "victim")
            .unwrap();
        state.delete("victim").unwrap().unwrap();

        // The gate every post-lookup lock acquisition goes through:
        // a handle that predates the removal must be turned away.
        assert!(
            stale.lock_unless_deleted().is_none(),
            "the tombstone must refuse a stale handle"
        );
        // And the public write path answers NotFound rather than
        // recreating the WAL file the delete just removed.
        assert!(matches!(
            state.add_associations(
                "victim",
                vec![assoc_op("幽霊", "は", "残らない", 1.0, None)],
            ),
            Err(AccessError::NotFound)
        ));
        assert!(!wal_path(&dir, &file_stem("victim")).exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_flusher_stalled_on_another_context_cannot_resurrect_a_delete() {
        use std::thread;
        use std::time::Duration;

        let dir = scratch_dir("delete-mid-flush");
        for round in 0..12 {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            for name in ["decoy", "victim"] {
                state
                    .create(name, ContextMeta::default())
                    .map_err(|_| "create")
                    .unwrap();
                state
                    .add_associations(
                        name,
                        vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
                    )
                    .unwrap()
                    .unwrap();
            }

            // A periodic flusher mid-run: it snapshots BOTH contexts,
            // then stalls on the decoy's lock — in the rounds where the
            // decoy comes first in its iteration order — while the
            // delete below runs to completion. The stalled half then
            // reaches the victim through its stale handle. Whatever the
            // iteration order, the end state must be identical: the
            // victim stays deleted.
            let decoy = state.lookup("decoy").unwrap();
            let hold = decoy.inner.write().unwrap();
            let flusher = {
                let state = state.clone();
                thread::spawn(move || {
                    state.flush_dirty();
                })
            };
            thread::sleep(Duration::from_millis(20)); // flusher snapshots, then parks on the decoy
            state.delete("victim").unwrap().unwrap();
            drop(hold);
            flusher.join().unwrap();

            let stem = file_stem("victim");
            for suffix in [
                "ctx",
                "meta.json",
                "sources.json",
                "vectors.bin",
                "wal.jsonl",
            ] {
                let path = dir.join(format!("{stem}.{suffix}"));
                assert!(
                    !path.exists(),
                    "round {round}: {} survived the delete",
                    path.display()
                );
            }
            // A reboot's view — with the old generation (and its
            // directory lock) gone first, as in any real reboot.
            drop(state);
            let reborn = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            assert_eq!(
                reborn.context_count(),
                1,
                "round {round}: only the decoy may remain"
            );
            let _ = fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn alias_removal_survives_a_restart_via_the_wal() {
        let dir = scratch_dir("unalias-wal");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("青嶺酒造", "創業年", "1907年", 1.0, None)],
                )
                .unwrap()
                .unwrap();
            state
                .add_aliases(
                    "sake",
                    &BTreeMap::from([("Aomine".to_string(), "青嶺酒造".to_string())]),
                    &BTreeMap::new(),
                )
                .unwrap()
                .unwrap();
            // Bake the alias into the image; the removal that follows
            // lives only in the WAL when the process dies.
            state.flush_dirty();
            assert_eq!(
                state
                    .remove_aliases("sake", &["Aomine".to_string()], &[])
                    .unwrap()
                    .unwrap(),
                1
            );
        }
        let reborn = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let aliases = reborn
            .read_context("sake", |context| context.concept_aliases().len())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(aliases, 0, "the un-flushed removal must replay");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn replay_does_not_double_apply_records_already_baked_into_the_image() {
        let dir = scratch_dir("wal-noreplay");
        let wal_file = wal_path(&dir, &file_stem("sake"));
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
                )
                .unwrap()
                .unwrap();
            let logged = fs::read(&wal_file).unwrap();
            assert!(!logged.is_empty());

            // The flush bakes the watermark into the image and
            // truncates the log ...
            assert_eq!(state.flush_dirty(), vec!["sake".to_string()]);
            assert_eq!(fs::metadata(&wal_file).unwrap().len(), 0);
            // ... so putting the pre-truncation bytes back simulates a
            // crash between the image rename and the truncate (or a
            // truncate that simply never ran).
            fs::write(&wal_file, logged).unwrap();
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let weight = state
            .read_context("sake", |context| {
                context.query(Some("青嶺酒造"), Some("代表銘柄"), Some("青嶺"))[0].weight
            })
            .map_err(|_| "read")
            .unwrap();
        // associate accumulates: a wrongly replayed record would make
        // this 2.0 — the silent corruption the watermark exists to
        // prevent.
        assert_eq!(weight, 1.0);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn aliases_and_retractions_ride_the_wal_across_a_restart() {
        let dir = scratch_dir("wal-ops");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .add_associations(
                    "sake",
                    vec![
                        assoc_op("青嶺酒造", "仕込み水", "伏流水", 1.0, Some("第2段落")),
                        assoc_op("青嶺酒造", "仕込み水", "伏流水", 1.0, Some("第5段落")),
                    ],
                )
                .unwrap()
                .unwrap();
            state
                .add_aliases(
                    "sake",
                    &BTreeMap::from([("Aomine".to_string(), "青嶺酒造".to_string())]),
                    &BTreeMap::new(),
                )
                .unwrap()
                .unwrap();
            let (touched, _) = state.retract_source("sake", "第5段落").unwrap();
            assert_eq!(touched, 1);
            // No flush — every one of those op kinds must replay.
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let matches = state
            .read_context("sake", |context| {
                context.query(Some("Aomine"), Some("仕込み水"), None)
            })
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(matches.len(), 1, "the alias entry point must replay");
        assert_eq!(matches[0].weight, 1.0, "the retraction must replay");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_wal_that_cannot_be_written_refuses_the_write() {
        let dir = scratch_dir("wal-refuse");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // A directory sitting where the log file belongs makes the
        // append fail deterministically.
        fs::create_dir_all(wal_path(&dir, &file_stem("sake"))).unwrap();

        let outcome = state.add_associations(
            "sake",
            vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
        );
        assert!(matches!(outcome, Err(AccessError::Unpersisted(_))));
        // Refused cleanly: nothing reached the graph.
        let count = state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(count, 0);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disabling_the_wal_restores_the_flush_window() {
        let dir = scratch_dir("wal-off");
        {
            let state = AppState::boot_with(
                dir.clone(),
                usize::MAX,
                None,
                false,
                DEFAULT_WAL_MAX_BYTES,
                None,
            )
            .unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
                )
                .unwrap()
                .unwrap();
            assert!(
                !wal_path(&dir, &file_stem("sake")).exists(),
                "no log may be written when disabled"
            );
            // No flush: with the WAL off, this write is the accepted
            // crash casualty — exactly the pre-WAL posture.
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let count = state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(count, 0);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn write_atomic_replaces_content_and_leaves_no_staging_file() {
        let dir = scratch_dir("atomic");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("file.bin");

        write_atomic(&path, b"first").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"first");
        write_atomic(&path, b"second").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"second");
        // A successful write consumes its (uniquely named) staging file.
        let leftovers = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.starts_with("tmp"))
            })
            .count();
        assert_eq!(leftovers, 0, "staging files must not survive a commit");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn text_terms_mix_ascii_words_with_bigram_runs() {
        let pair = |a: char, b: char| ((a as u64) << 32) | b as u64;

        // An ASCII run is one whole-word term; extra separators change
        // nothing, and different words hash apart.
        assert_eq!(text_terms("word").len(), 1);
        assert_eq!(text_terms("ambition must"), text_terms("ambition  must"));
        assert_ne!(text_terms("word"), text_terms("words"));

        // Undelimited text contributes adjacent pairs per run; a run of
        // one contributes the lone character.
        assert_eq!(
            text_terms("仕込み"),
            vec![pair('仕', '込'), pair('込', 'み')]
        );
        assert_eq!(text_terms("水"), vec!['水' as u64]);

        // Punctuation breaks the run: no pair straddles the comma.
        assert_eq!(text_terms("水、源"), vec!['水' as u64, '源' as u64]);

        // A script switch breaks the run too: 第10篇 is 第 + word "10"
        // + 篇, never a pair that straddles the digits.
        let mixed = text_terms("第10篇");
        assert_eq!(mixed.len(), 3);
        assert!(mixed.contains(&('第' as u64)));
        assert!(mixed.contains(&('篇' as u64)));
        assert!(mixed.iter().any(|term| term & (1 << 63) != 0));
    }

    #[test]
    fn camel_pieces_join_the_passage_term_stream() {
        let word = |text: &str| text_terms(text)[0];

        // A camelCase run carries its whole-word term AND its pieces,
        // hashed exactly like standalone words.
        let terms = passage_terms("AppState");
        assert!(terms.contains(&word("appstate")));
        assert!(terms.contains(&word("app")));
        assert!(terms.contains(&word("state")));

        // An acronym ends before its last capital.
        let terms = passage_terms("HTTPServer");
        assert!(terms.contains(&word("http")));
        assert!(terms.contains(&word("server")));

        // Digits stick to their piece; the boundary is the case change.
        let terms = passage_terms("U64Max");
        assert!(terms.contains(&word("u64")));
        assert!(terms.contains(&word("max")));

        // Runs with no case boundary add nothing: snake_case already
        // splits at the underscore, ALLCAPS and lowercase stay whole.
        assert_eq!(passage_terms("flush_dirty"), text_terms("flush_dirty"));
        assert_eq!(passage_terms("WAL replay"), text_terms("wal replay"));
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

    #[test]
    fn a_camel_case_piece_finds_the_passage() {
        let dir = scratch_dir("camel");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("code", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // "AppState" and "PathBuf" occur only as camelCase tokens here —
        // none of their pieces appears as a standalone word.
        let mut passages = BTreeMap::new();
        passages.insert(
            "src/registry.rs:AppState".to_string(),
            "impl AppState { pub fn boot_with(dir: PathBuf) -> Self { todo!() } }".to_string(),
        );
        state.store_passages("code", passages).unwrap().unwrap();

        for query in ["state", "State", "app", "path"] {
            let hits = state.search_passages("code", query, 3).unwrap();
            assert_eq!(
                hits.first().map(|hit| hit.0.as_str()),
                Some("src/registry.rs:AppState"),
                "a piece of a camelCase identifier must reach its passage (query {query:?})"
            );
        }

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

        fn embed(&self, texts: &[&str], _purpose: EmbedPurpose) -> Result<Vec<Vec<f32>>, String> {
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

    /// The two provider call sites declare opposite purposes: gloss
    /// refresh embeds as `Index`, live cue resolution as `Query` — the
    /// distinction an asymmetric-model proxy keys `input_type` on.
    #[test]
    fn refresh_embeds_as_index_and_cue_resolution_as_query() {
        struct RecordingEmbeddings(Arc<Mutex<Vec<EmbedPurpose>>>);
        impl EmbeddingProvider for RecordingEmbeddings {
            fn model(&self) -> &str {
                "recorder"
            }
            fn embed(
                &self,
                texts: &[&str],
                purpose: EmbedPurpose,
            ) -> Result<Vec<Vec<f32>>, String> {
                self.0.lock().unwrap().push(purpose);
                Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
            }
        }

        let dir = scratch_dir("purpose");
        let purposes = Arc::new(Mutex::new(Vec::new()));
        let embedder = Some(
            Arc::new(RecordingEmbeddings(Arc::clone(&purposes))) as Arc<dyn EmbeddingProvider>
        );
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("p", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .write_context("p", |context| {
                context.associate("a", "l", "b", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();
        state.refresh_embeddings("p").unwrap().unwrap();
        state
            .semantic_resolve("p", "cue", false, None)
            .unwrap()
            .unwrap();

        let seen = purposes.lock().unwrap().clone();
        let (cue_call, refresh_calls) = seen.split_last().unwrap();
        assert!(!refresh_calls.is_empty());
        assert!(refresh_calls.iter().all(|p| *p == EmbedPurpose::Index));
        assert_eq!(*cue_call, EmbedPurpose::Query);

        let _ = fs::remove_dir_all(dir);
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
            concepts
                .iter()
                .all(|(a, b, _)| !pairs_up(a, b, "青嶺酒造", "1907年")
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

    /// TAGURU_SEMANTIC_FLOOR reaches boot as a server-wide default that
    /// sits UNDER the per-context setting and the per-call override —
    /// it recalibrates the floor for the configured embedding model
    /// without touching any context.
    #[test]
    fn semantic_floor_server_default_recalibrates_under_context_and_call() {
        let dir = scratch_dir("semfloor-srv");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let embedder = Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            embedder,
            true,
            DEFAULT_WAL_MAX_BYTES,
            Some(0.2),
        )
        .unwrap();
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

        let hits = |floor: Option<f32>| {
            state
                .semantic_resolve("fruit", "みかん", false, floor)
                .unwrap()
                .unwrap()
        };
        // みかん×りんご = cosine 0.28: lost under the built-in 0.35,
        // admitted by the recalibrated server default.
        assert_eq!(hits(None)[0].0, "りんご");
        // The context setting still beats the server default ...
        state
            .update_meta("fruit", None, None, None, Some(0.9))
            .unwrap()
            .unwrap();
        assert!(hits(None).is_empty());
        // ... and the one-call override still beats them both.
        assert_eq!(hits(Some(0.1))[0].0, "りんご");

        let _ = fs::remove_dir_all(dir);
    }
}
