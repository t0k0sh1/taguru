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
//! context it happened in. Groups (tiny, always-resident records) sit
//! behind one separate lock; the only operations that need both take
//! `groups` BEFORE `registry` — never the other way around.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use serde::{Deserialize, Serialize};
use taguru::context::{AliasError, Context, LabelUsage};

use crate::embedding::{
    EmbedPurpose, EmbeddingProvider, PassageKey, PassageVectorStore, VectorStore, VectorTable,
    fnv1a, similarity,
};
use crate::groups::{self, GroupRecord};
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
    /// Same `{label, count}` shape as `describe`'s `as_subject`/`as_object`.
    #[serde(deserialize_with = "deserialize_top_concepts")]
    pub top_concepts: Vec<LabelUsage>,
    /// The first labels of the relation vocabulary (capped; the full
    /// list is at `GET /contexts/{name}/labels`).
    pub label_sample: Vec<String>,
}

/// Accepts both the current `{label, count}` object shape and the
/// pre-#36 `[name, count]` tuple shape, so a `.meta.json` sidecar
/// written before the shape change still loads its saved description,
/// policy, and usage counters instead of a corrupt-sidecar fallback to
/// `MetaFile::default()`.
fn deserialize_top_concepts<'de, D>(deserializer: D) -> Result<Vec<LabelUsage>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Entry {
        Object(LabelUsage),
        Tuple(String, usize),
    }
    Vec::<Entry>::deserialize(deserializer).map(|entries| {
        entries
            .into_iter()
            .map(|entry| match entry {
                Entry::Object(usage) => usage,
                Entry::Tuple(label, count) => LabelUsage { label, count },
            })
            .collect()
    })
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
                .map(|(name, degree)| LabelUsage {
                    label: name.to_string(),
                    count: degree,
                })
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
    /// Set on every write; cleared by a flush the moment it CLAIMS the
    /// entry — under `inner`, before the image is staged, not after it
    /// commits. Only ever changed while `inner` is write-locked; the
    /// atomic just lets the flusher skip clean entries without locking
    /// them. A write that lands while a flush stages its bytes re-sets
    /// the flag under the same lock, so it flushes into an image of its
    /// own next tick — the reason clearing early loses nothing, WAL on
    /// or off. The window this opens — `dirty` clear but the bytes not
    /// yet on disk — is covered by `flushing`.
    dirty: AtomicBool,
    /// A flush is staging this entry right now: set at claim (under
    /// `inner`, alongside clearing `dirty`) and released when the flush
    /// ends. Two jobs. It dedups flushers — a concurrent `flush_dirty`
    /// (a tick against the shutdown flush) skips an entry already in
    /// flight instead of staging the same image twice. And it fences a
    /// racing eviction: because the claim clears `dirty`, an evict that
    /// locks mid-stage would read "clean" and drop the entry, losing a
    /// write that (with the WAL off) lives nowhere else — so eviction
    /// saves on `dirty || flushing`, and this flag is what stays true
    /// across the staging window `dirty` no longer marks.
    flushing: AtomicBool,
    /// Logical timestamp of the last operation, for LRU eviction.
    last_touch: AtomicU64,
    /// The vector sidecar, held after first use so the semantic
    /// fallback never re-reads a many-megabyte file per query. Replaced
    /// by refresh, cleared by eviction, and counted against the cache
    /// budget. Lock order: `inner` before `vectors`, never the reverse.
    vectors: Mutex<Option<Arc<VectorStore>>>,
    /// Serializes whole gloss-embedding refreshes — the same reason as
    /// `passage_refresh` below: a diff computed outside the entry lock
    /// (provider round trips can take seconds) races on which of two
    /// overlapping refreshes' provider calls lands first, not on which
    /// one read the newer gloss. Held for the whole refresh, not just
    /// the merge, so a slower refresh of an older gloss can never
    /// finish after — and clobber — a faster refresh of a newer one.
    vectors_refresh: Mutex<()>,
    /// The passage store, resident after first use (see
    /// [`crate::passages::PassageStore`]) and counted against the cache
    /// budget like `vectors`. Lock order: `inner` (the tombstone fence,
    /// held SHARED for a passage operation's whole run) → `passages` →
    /// `vectors`; the fence means files for a name are only touched
    /// while a delete cannot be planting its tombstone.
    passages: Mutex<Option<Arc<crate::passages::PassageStore>>>,
    /// The resident BM25 paragraph index — derived from `passages`,
    /// rebuilt from a store snapshot whenever missing or tombstone-
    /// heavy, dropped on eviction. Lock rule: acquire this only AFTER
    /// every passage-store lock is released (holding `bm25` while
    /// READING the store is fine and is how a build works; the reverse
    /// nesting would deadlock against it).
    bm25: RwLock<Option<crate::bm25::Bm25Index>>,
    /// Set when the resident index diverges from its `{stem}.bm25.bin`
    /// sidecar (a build, a repair, an in-place update); the flush tick
    /// persists and clears it, eviction saves best-effort. The sidecar
    /// spares the next residency a full re-tokenization — it is still
    /// derived data, so a failed save only warns.
    bm25_dirty: AtomicBool,
    /// The paragraph vector sidecar, resident after first use — the
    /// vector lane's mirror of `vectors`, in its own slot so resolve's
    /// small hot gloss store never shares a fate with this big one.
    passage_vectors: Mutex<Option<Arc<PassageVectorStore>>>,
    /// Set when a passage store/retract lands, cleared when a passage
    /// embedding refresh claims it — the auto-refresh ticker's signal
    /// (passage writes do not mark the GRAPH dirty, so the flush list
    /// alone would miss passage-only activity).
    passages_embed_dirty: AtomicBool,
    /// Serializes whole passage-embedding refreshes: the ticker and
    /// the HTTP endpoint can fire concurrently, each computes offline
    /// for seconds, and an unserialized OLDER run would publish (and
    /// persist) its sidecar over a newer one's on the way out. The
    /// loser of this lock re-diffs against the winner's hashes and
    /// no-ops.
    passage_refresh: Mutex<()>,
    /// Usage counters (see [`ContextUsage`]). `usage_dirty` marks
    /// increments the sidecar has not seen yet, so the shutdown sweep
    /// skips the contexts nobody touched.
    usage: UsageCounters,
    usage_dirty: AtomicBool,
    /// The passage store's last failed load, while it is being
    /// remembered — [`EntryInner::load_failure`]'s counterpart for the
    /// passage side, which caches its store in `passages` rather than
    /// the slot. Only ever locked while `passages` is held (or alone
    /// by the test aging helper), so it adds no lock-order edge.
    passages_load_failure: Mutex<Option<(std::time::Instant, String)>>,
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
                load_failure: None,
                image_generation: 0,
            }),
            dirty: AtomicBool::new(false),
            flushing: AtomicBool::new(false),
            last_touch: AtomicU64::new(0),
            vectors: Mutex::new(None),
            vectors_refresh: Mutex::new(()),
            passages: Mutex::new(None),
            bm25: RwLock::new(None),
            bm25_dirty: AtomicBool::new(false),
            passage_vectors: Mutex::new(None),
            passages_embed_dirty: AtomicBool::new(false),
            passage_refresh: Mutex::new(()),
            usage: UsageCounters::seeded(&usage),
            usage_dirty: AtomicBool::new(false),
            passages_load_failure: Mutex::new(None),
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

    /// The read half of the tombstone fence: passage operations hold
    /// this SHARED guard for their whole run — concurrent with graph
    /// reads and with each other, but correctly serialized against
    /// [`AppState::delete`], whose exclusive lock plants the tombstone.
    /// Whichever side locks first wins cleanly: a fence taken first
    /// makes the delete wait; a tombstone planted first turns the
    /// operation into a no-op instead of a file resurrection.
    fn read_unless_deleted(&self) -> Option<std::sync::RwLockReadGuard<'_, EntryInner>> {
        let guard = self.inner.read().unwrap();
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

    /// Bytes the resident passage store holds — zero while cold.
    fn passages_footprint(&self) -> usize {
        self.passages
            .lock()
            .unwrap()
            .as_ref()
            .map(|store| store.footprint())
            .unwrap_or(0)
    }

    /// Bytes the resident BM25 index holds — zero while cold.
    fn bm25_footprint(&self) -> usize {
        self.bm25
            .read()
            .unwrap()
            .as_ref()
            .map(|index| index.footprint())
            .unwrap_or(0)
    }

    /// Bytes the resident paragraph vectors hold — zero while cold.
    fn passage_vectors_footprint(&self) -> usize {
        self.passage_vectors
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
    /// The last failed image load, while it is being remembered: when
    /// it failed and the refusal it produced. While fresh
    /// ([`LOAD_FAILURE_RETRY`]), `ensure_hot` answers the cached
    /// refusal without touching the disk — a permanently corrupt
    /// context must not cost a full read + parse per request under
    /// client retries. Cleared by the next successful load; never
    /// persisted.
    load_failure: Option<(std::time::Instant, String)>,
    /// Bumped whenever `slot` is replaced by a NEW `Context` object
    /// while staying `Hot` — currently only `compact_context`. Plain
    /// u64 like `wal_seq`: every touch happens under this entry's write
    /// lock. `flush_entry` stages its bytes with the entry unlocked and
    /// re-checks `slot` on republish; that recheck sees Hot in both an
    /// untouched entry and a freshly-compacted one, so it cannot alone
    /// tell "unchanged since I read it" apart from "replaced by a
    /// compaction while I staged" — this generation is what makes the
    /// two distinguishable.
    image_generation: u64,
}

#[derive(Debug)]
pub enum CreateError {
    AlreadyExists,
    /// The name is not usable as a context — currently only the empty
    /// string, which would `file_stem` to `""` and land as a bare
    /// `.ctx` file that `scan_data_dir` (keying on the `ctx` extension,
    /// which a leading-dot name has none of) never rediscovers: a
    /// context that vanishes on the next restart.
    InvalidName,
    Io(io::Error),
}

#[derive(Debug)]
pub enum CreateGroupError {
    AlreadyExists,
    /// Same trap as [`CreateError::InvalidName`]: an empty name would
    /// persist as a bare `.group` file the boot scan (keying on the
    /// extension) never rediscovers.
    InvalidName,
    /// A listed member is not a registered context — carried by name so
    /// the client hears WHICH one. Strict on purpose: an add must never
    /// mint a dangling reference.
    NoSuchContext(String),
    /// A listed child is not a registered group — the same strictness,
    /// one namespace over. A create can never trip the cycle check
    /// through this gate: a child naming the group being created is
    /// not registered yet, so it refuses here first.
    NoSuchGroup(String),
    /// The children exist but the shape is not allowed: a cycle, or a
    /// chain of more than [`groups::MAX_GROUP_DEPTH`] groups.
    Nesting(groups::NestingViolation),
    Io(io::Error),
}

#[derive(Debug)]
pub enum UpdateGroupError {
    NotFound,
    /// Same strictness as [`CreateGroupError::NoSuchContext`], for
    /// `add_contexts`. Removals are exempt — removing a name that is
    /// not a member is an idempotent no-op, never an error.
    NoSuchContext(String),
    /// [`CreateGroupError::NoSuchGroup`]'s twin, for `add_groups`.
    NoSuchGroup(String),
    /// [`CreateGroupError::Nesting`]'s twin — and here the cycle arm is
    /// reachable: the group being updated IS registered, so adding it
    /// (or an ancestor) as its own child passes the existence gate and
    /// lands in the validator's lap.
    Nesting(groups::NestingViolation),
    Io(io::Error),
}

/// What one compaction accomplished — the before/after footprint and
/// the dead weight shed, for the CLI report and the endpoint response.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct CompactOutcome {
    pub bytes_before: usize,
    pub bytes_after: usize,
    pub dead_edges: usize,
    pub aliases_dropped: usize,
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
    /// Locates the fact within `source` (e.g. a paragraph index).
    /// Meaningless without a source, so it is only ever honored when
    /// `source` is also present — see [`apply_op`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paragraph: Option<u32>,
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

/// Default ceiling for a context's PASSAGE log
/// (`TAGURU_PASSAGES_WAL_MAX_BYTES`). Larger than the graph's: the
/// ratio-triggered compaction legitimately lets the log grow to about
/// the snapshot's own size before compacting, so this is sized as a
/// backstop for a compaction that is failing outright, not as a bound
/// any healthy context ever nears (the refusal additionally requires
/// the log to have outgrown 2× the last snapshot — see
/// `PassageStore::store`).
pub const DEFAULT_PASSAGES_WAL_MAX_BYTES: usize = 1024 * 1024 * 1024;

/// The boot knobs `taguru serve` and `taguru import` read identically —
/// one reading, so the two entrances cannot drift. The values stay
/// visible (rather than being swallowed by [`AppState::boot_with`])
/// because serve's "server ready" line reports them.
pub struct BootConfig {
    pub data_dir: PathBuf,
    pub cache_bytes: usize,
    pub wal_enabled: bool,
    pub wal_max_bytes: usize,
    pub passages_wal_max_bytes: usize,
    pub embed_passages: bool,
    pub passage_vector_limit: usize,
    pub semantic_floor: Option<f32>,
}

/// Default ceiling on how many paragraphs per context get a vector
/// (`TAGURU_PASSAGE_VECTOR_LIMIT`). The vector lane's footprint is
/// paragraphs × dimensions × 4 bytes — 20 000 rows of a 1536-dim model
/// is ~120 MiB — and past the limit the lexical lane still serves
/// every paragraph; only the semantic lane goes partial.
pub const DEFAULT_PASSAGE_VECTOR_LIMIT: usize = 20_000;

/// One fused passage-search hit with its per-lane evidence: 1-based
/// rank and raw score in each lane that surfaced it. BM25 scores and
/// cosine similarities live on different scales — which is exactly why
/// the fusion above them is rank-based.
#[derive(Debug)]
pub struct PassageSearchHit {
    pub source: String,
    pub index: u32,
    pub score: f32,
    pub text: String,
    pub bm25: Option<(usize, f32)>,
    pub vector: Option<(usize, f32)>,
}

/// The outcome of resolving one `(source, paragraph index)` citation:
/// found, or which half of the lookup missed. Kept distinct from the
/// outer `Option<io::Result<_>>` (context-absent / I/O failure), which
/// stays reserved for the store itself being unreachable.
#[derive(Debug)]
pub(crate) enum CitationLookup {
    Found(String, Option<String>),
    UnknownSource,
    IndexOutOfRange,
}

/// Accumulator behind the fusion: each lane keeps the (rank, score,
/// paragraph hash) it actually scored, so staleness settles per lane
/// against the store's current text.
#[derive(Default)]
struct FusedHit {
    bm25: Option<(usize, f32, u64)>,
    vector: Option<(usize, f32, u64)>,
}

/// What one passage embedding refresh accomplished: rows newly
/// embedded, rows now in the sidecar, and rows the per-context limit
/// cut off. Text rows and doc2query question rows count alike in all
/// three — the limit itself is row-denominated, so `skipped_over_limit`
/// is exactly how far it fell short, not a paragraph count.
#[derive(Debug)]
pub struct PassageRefreshOutcome {
    pub embedded: usize,
    pub total: usize,
    pub skipped_over_limit: usize,
}

/// The behavioral knobs [`AppState::boot_with`] takes as one struct:
/// the list only grows with the server, and a test that wants ONE knob
/// should not have to name seven.
pub struct BootOptions {
    pub wal_enabled: bool,
    pub wal_max_bytes: usize,
    pub passages_wal_max_bytes: usize,
    /// Whether paragraphs get embedded at all (`TAGURU_EMBED_PASSAGES`,
    /// default off): a corpus is orders of magnitude more text than its
    /// glosses, so the spend is opt-in even where gloss embedding is on.
    pub embed_passages: bool,
    pub passage_vector_limit: usize,
    pub default_semantic_floor: Option<f32>,
}

impl Default for BootOptions {
    fn default() -> Self {
        Self {
            wal_enabled: true,
            wal_max_bytes: DEFAULT_WAL_MAX_BYTES,
            passages_wal_max_bytes: DEFAULT_PASSAGES_WAL_MAX_BYTES,
            embed_passages: false,
            passage_vector_limit: DEFAULT_PASSAGE_VECTOR_LIMIT,
            default_semantic_floor: None,
        }
    }
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
            // The passage log's own backstop; it only engages when
            // compaction is demonstrably stuck (see PassageStore).
            passages_wal_max_bytes: crate::env_number(
                "TAGURU_PASSAGES_WAL_MAX_BYTES",
                DEFAULT_PASSAGES_WAL_MAX_BYTES,
            ),
            // Paragraph embedding is opt-in on top of the provider
            // being configured — a corpus is orders of magnitude more
            // text than its glosses.
            embed_passages: std::env::var("TAGURU_EMBED_PASSAGES")
                .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            passage_vector_limit: crate::env_number(
                "TAGURU_PASSAGE_VECTOR_LIMIT",
                DEFAULT_PASSAGE_VECTOR_LIMIT,
            ),
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
            BootOptions {
                wal_enabled: self.wal_enabled,
                wal_max_bytes: self.wal_max_bytes,
                passages_wal_max_bytes: self.passages_wal_max_bytes,
                embed_passages: self.embed_passages,
                passage_vector_limit: self.passage_vector_limit,
                default_semantic_floor: self.semantic_floor,
            },
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
    /// Groups: bundles of context names and child-group names (a
    /// shallow DAG, at most [`groups::MAX_GROUP_DEPTH`] groups tall and
    /// never cyclic), each persisted as one `{stem}.group` file. Small
    /// enough to stay resident in full, so one lock over the whole map
    /// suffices — and it is held across the record's own fsync on
    /// writes: a group write blocks only other group writes, never any
    /// context operation. BTreeMap keeps the directory listing in name
    /// order for free.
    groups: RwLock<BTreeMap<String, GroupRecord>>,
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
    /// Same backstop for each context's passage log
    /// (`TAGURU_PASSAGES_WAL_MAX_BYTES`, 0 = unlimited), handed to the
    /// store on load; it refuses stores only when compaction is
    /// demonstrably stuck.
    passages_wal_max_bytes: usize,
    /// Whether paragraphs get embedded (`TAGURU_EMBED_PASSAGES`) and
    /// how many per context at most (`TAGURU_PASSAGE_VECTOR_LIMIT`).
    embed_passages: bool,
    passage_vector_limit: usize,
    /// Names whose delete is still removing files. A delete takes the
    /// name out of the registry FIRST and only then (unlocked) unlinks
    /// the file family — without this set, a create() in that window
    /// would lay down a new generation for the tail of the delete's
    /// unlink loop to destroy. Entered in the same critical section
    /// that removes the name, left when the files are gone.
    pending_deletes: Mutex<std::collections::HashSet<String>>,
    /// Names whose create is still writing files — the create-side twin
    /// of `pending_deletes`. A create reserves the name here FIRST and
    /// only then (unlocked) clears leftovers and fsyncs the fresh file
    /// family; without this set the registry lock would have to stay
    /// held across that disk work, stalling every operation on every
    /// context behind one create's fsyncs. Entered under the registry
    /// guard, left in the critical section that registers the entry.
    pending_creates: Mutex<std::collections::HashSet<String>>,
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
        Self::boot_with(data_dir, cache_bytes, embedder, BootOptions::default())
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
        options: BootOptions,
    ) -> io::Result<Self> {
        fs::create_dir_all(&data_dir)?;
        // Before reading anything: two live registries over one
        // directory (a second serve, or an import against a running
        // server) would each cache and flush independently — last
        // writer wins, silently.
        let dir_lock = lock_data_dir(&data_dir)?;
        let registry = scan_data_dir(&data_dir)?;
        // Groups scan after contexts (the context scan also sweeps
        // staging leftovers), then reconcile unconditionally: whatever
        // put a dangling member, a dangling child, or an illegal
        // nesting into a group file — a crash between a deletion and
        // the sweep's rewrite, a sweep that could not persist, a
        // hand-edited directory — boot drops it and writes the fix
        // back, so "a group names only live contexts and live groups,
        // acyclically, within the depth cap" holds from the first
        // request on, without exception.
        let mut groups = groups::scan_groups(&data_dir)?;
        reconcile_groups(&data_dir, &registry, &mut groups);

        let state = Self(Arc::new(StateInner {
            data_dir,
            _dir_lock: dir_lock,
            cache_bytes,
            registry: RwLock::new(registry),
            groups: RwLock::new(groups),
            clock: AtomicU64::new(0),
            embedder,
            default_semantic_floor: options
                .default_semantic_floor
                .unwrap_or(DEFAULT_SEMANTIC_FLOOR)
                .clamp(0.0, 1.0),
            cue_cache: Mutex::new(CueCache::default()),
            metrics: Metrics::default(),
            wal_enabled: options.wal_enabled,
            wal_max_bytes: options.wal_max_bytes,
            passages_wal_max_bytes: options.passages_wal_max_bytes,
            embed_passages: options.embed_passages,
            passage_vector_limit: options.passage_vector_limit,
            pending_deletes: Mutex::new(std::collections::HashSet::new()),
            pending_creates: Mutex::new(std::collections::HashSet::new()),
            resident_estimate: AtomicI64::new(0),
            budget_ops: AtomicU64::new(0),
        }));
        state.preload_pinned();
        Ok(state)
    }

    /// Loads every pinned context now — in parallel, because this runs
    /// before the listener binds and its wall-clock IS the downtime a
    /// single-writer deploy pays (stop-then-start; see the README's
    /// rollout note), and chatty on purpose: a boot that spends
    /// seconds loading should say what it is loading, not sit silent
    /// until "server ready". Entries have independent locks, so the
    /// workers never contend with each other.
    fn preload_pinned(&self) {
        let pinned: Vec<(String, Arc<Entry>)> = self
            .snapshot()
            .into_iter()
            .filter(|(_, entry)| entry.inner.read().unwrap().meta.pinned)
            .collect();
        if pinned.is_empty() {
            return;
        }
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(pinned.len());
        let queue = Mutex::new(pinned.into_iter());
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    loop {
                        let Some((name, entry)) = queue.lock().unwrap().next() else {
                            break;
                        };
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
                            Err(error) => {
                                tracing::warn!("pinned context '{name}' not preloaded: {error}");
                            }
                        }
                    }
                });
            }
        });
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
        let mut passages_wal_bytes = 0u64;
        for (name, entry) in snapshot {
            let inner = entry.inner.read().unwrap();
            if let Slot::Hot(context) = &inner.slot {
                contexts_resident += 1;
                resident_bytes += context.footprint() as u64;
            }
            wal_bytes += inner.wal_bytes;
            drop(inner);
            resident_bytes += entry.vectors_footprint() as u64;
            resident_bytes += entry.passages_footprint() as u64;
            resident_bytes += entry.bm25_footprint() as u64;
            resident_bytes += entry.passage_vectors_footprint() as u64;
            // A resident store knows its pending log; a cold one gets a
            // stat — the gauge must not go blind just because a context
            // was evicted.
            passages_wal_bytes += {
                let resident = entry
                    .passages
                    .lock()
                    .unwrap()
                    .as_ref()
                    .map(|store| store.pending_log_bytes());
                resident.unwrap_or_else(|| {
                    fs::metadata(passages_wal_path(&self.0.data_dir, &file_stem(&name)))
                        .map(|meta| meta.len())
                        .unwrap_or(0)
                })
            };
        }
        GaugeSnapshot {
            contexts_registered,
            groups_registered: self.0.groups.read().unwrap().len() as u64,
            contexts_resident,
            resident_bytes,
            wal_bytes,
            passages_wal_bytes,
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
    ///
    /// The registry lock is NOT held across the disk work (up to seven
    /// unlinks plus save_files' fsyncs — seconds on slow storage,
    /// behind which every operation on every context would otherwise
    /// stall). The name is reserved in `pending_creates` under the
    /// registry guard, the files are written unlocked, and the entry
    /// lands in a second critical section — the create twin of
    /// delete's `pending_deletes` choreography.
    pub fn create(&self, name: &str, meta: ContextMeta) -> Result<(), CreateError> {
        // An empty name has no file stem — it would persist as a bare
        // `.ctx` and disappear from the registry on the next restart.
        // Refuse it at the lowest boundary, so no entrance (import,
        // direct call) can conjure a self-erasing context.
        if name.is_empty() {
            return Err(CreateError::InvalidName);
        }
        {
            let registry = self.0.registry.read().unwrap();
            // A name mid-delete is still taken: its delete has left the
            // registry but is still unlinking files, and a create landing
            // now would have its fresh generation destroyed by the tail of
            // that loop. A name mid-create is equally taken. The client
            // sees the same refusal as for a live name and simply retries
            // after the other call's response.
            if registry.contains_key(name) || self.0.pending_deletes.lock().unwrap().contains(name)
            {
                return Err(CreateError::AlreadyExists);
            }
            // Reserving while still under the registry guard closes the
            // gap against a finishing create: reservations leave only
            // inside the registry's WRITE section below, so "not in the
            // map, not reserved" cannot be observed between a sibling's
            // insert and its unreserve.
            if !self
                .0
                .pending_creates
                .lock()
                .unwrap()
                .insert(name.to_string())
            {
                return Err(CreateError::AlreadyExists);
            }
        }
        let created = self.create_files(name, &meta);
        // Success or failure, the reservation leaves in the same
        // critical section that (on success) makes the entry visible.
        let mut registry = self.0.registry.write().unwrap();
        let outcome = created.map(|(stats, usage, context)| {
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
        });
        self.0.pending_creates.lock().unwrap().remove(name);
        outcome
    }

    /// The disk half of [`AppState::create`], run WITHOUT the registry
    /// lock — the `pending_creates` reservation is what keeps the name
    /// taken meanwhile.
    ///
    /// A name can be reused after a delete, and a delete that failed
    /// partway (the name is unregistered first) or a half-restored
    /// backup leaves the old generation's files behind. Nothing may
    /// bleed into the new context — a stale WAL would even replay
    /// the old generation's acknowledged writes into the fresh image
    /// on its next cold load. Clear the slate — the OLD IMAGE INCLUDED —
    /// before writing the new one: `save_files` lands the image last, so
    /// removing the old image up front means a crash anywhere before the
    /// new image commits leaves NO image at all. Nothing registers (the
    /// scan keys on `.ctx`), the next attempt clears again, and the old
    /// generation's data can never resurface under the new create's
    /// metadata. Durability of the unlinks rides on save_files'
    /// parent-directory fsync just below. A leftover that cannot be
    /// removed fails the create — registering on top of it would hand out
    /// a haunted context.
    fn create_files(
        &self,
        name: &str,
        meta: &ContextMeta,
    ) -> Result<(ContextStats, ContextUsage, Context), CreateError> {
        let stem = file_stem(name);
        for stale in [
            image_path(&self.0.data_dir, &stem),
            wal_path(&self.0.data_dir, &stem),
            sources_path(&self.0.data_dir, &stem),
            passages_path(&self.0.data_dir, &stem),
            passages_wal_path(&self.0.data_dir, &stem),
            pvectors_path(&self.0.data_dir, &stem),
            bm25_path(&self.0.data_dir, &stem),
            vectors_path(&self.0.data_dir, &stem),
            // A leftover marker from an earlier delete that could not
            // finish MUST go before this new generation of files
            // lands — otherwise the next boot's resume-sweep sees the
            // marker and deletes the context we are creating right now.
            deleted_marker_path(&self.0.data_dir, &stem),
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
        save_files(&self.0.data_dir, name, meta, &stats, &usage, &context)
            .map_err(CreateError::Io)?;
        Ok((stats, usage, context))
    }

    /// Removes a context from the registry and deletes its files. The
    /// entry's lock is taken after the removal — waiting out any
    /// in-flight operation — and the slot becomes a tombstone under
    /// it: a flusher, evictor, or writer whose handle predates the
    /// removal finds [`Slot::Deleted`] when it finally locks, and
    /// backs off instead of recreating the files. Any unflushed writes
    /// are discarded — deletion destroys the context.
    ///
    /// The name enters `pending_deletes` in the same critical section
    /// that unregisters it and leaves only after the unlink loop: to a
    /// concurrent create() the name stays taken for the delete's whole
    /// run, so no new generation of files can appear under the tail of
    /// this one's removals.
    pub fn delete(&self, name: &str) -> Option<io::Result<()>> {
        let entry = {
            let mut registry = self.0.registry.write().unwrap();
            let entry = registry.remove(name)?;
            self.0
                .pending_deletes
                .lock()
                .unwrap()
                .insert(name.to_string());
            entry
        };
        let mut in_flight = entry.inner.write().unwrap();
        in_flight.slot = Slot::Deleted;
        self.recount_entry(&mut in_flight);
        entry.dirty.store(false, Ordering::Relaxed);
        // Lock order: `inner` before `passages` before `vectors`, as
        // documented on Entry.
        *entry.passages.lock().unwrap() = None;
        *entry.bm25.write().unwrap() = None;
        *entry.passage_vectors.lock().unwrap() = None;
        *entry.vectors.lock().unwrap() = None;
        let stem = file_stem(name);
        // The durable half of the acknowledgment: while this marker
        // exists, boot resumes the unlinks — so a partial failure here
        // (a held handle, a flaky mount) can leak bytes only until the
        // next start, and a surviving `.ctx` can never resurrect a
        // context the API reported gone. Written before the first
        // unlink; removed only after the last one succeeds.
        let marker = deleted_marker_path(&self.0.data_dir, &stem);
        if let Err(error) = write_atomic(&marker, b"") {
            tracing::warn!(context = %name, %error, "deletion marker not persisted; a partial delete would not resume at boot");
        }
        // Membership must not outlive the member: drop the name from
        // every group now, before the unlink loop's disk time. Best
        // effort — the delete's own durability rides on the marker
        // alone, and a sweep that could not persist is healed by the
        // next boot's reconciliation.
        self.sweep_context_from_groups(name);
        let mut outcome = Ok(());
        for file in context_files(&stem) {
            if let Err(error) = fs::remove_file(self.0.data_dir.join(file))
                && error.kind() != io::ErrorKind::NotFound
            {
                outcome = Err(error);
            }
        }
        if outcome.is_ok() {
            let _ = fs::remove_file(&marker);
        }
        self.0.pending_deletes.lock().unwrap().remove(name);
        Some(outcome)
    }

    /// Registers a group and persists it immediately — the create twin
    /// for groups, without the `pending_creates` choreography: the one
    /// fsync happens under the groups lock, which blocks only other
    /// group writes (see the field's doc), so nothing here needs the
    /// reservation dance.
    ///
    /// Member validation happens under both locks (`groups` before
    /// `registry` — the documented order): `contains_key` already
    /// answers false for a name mid-delete, because delete() removes
    /// the name and reserves it in `pending_deletes` inside one
    /// critical section. Child groups are judged against the same map
    /// the write lock already holds.
    pub fn create_group(
        &self,
        name: &str,
        description: String,
        contexts: BTreeSet<String>,
        children: BTreeSet<String>,
    ) -> Result<(), CreateGroupError> {
        if name.is_empty() {
            return Err(CreateGroupError::InvalidName);
        }
        let mut groups = self.0.groups.write().unwrap();
        if groups.contains_key(name) {
            return Err(CreateGroupError::AlreadyExists);
        }
        {
            let registry = self.0.registry.read().unwrap();
            if let Some(missing) =
                first_missing(&contexts, |context| registry.contains_key(context))
            {
                return Err(CreateGroupError::NoSuchContext(missing.clone()));
            }
        }
        if let Some(missing) = first_missing(&children, |child| groups.contains_key(child)) {
            return Err(CreateGroupError::NoSuchGroup(missing.clone()));
        }
        let record = GroupRecord {
            description,
            contexts,
            groups: children,
        };
        // The nesting validator wants the prospective map, so insert
        // first and unwind on refusal — nothing escapes the write lock
        // half-done.
        groups.insert(name.to_string(), record);
        if let Err(violation) = groups::validate_nesting(&groups) {
            groups.remove(name);
            return Err(CreateGroupError::Nesting(violation));
        }
        if let Err(error) = groups::write_group(&self.0.data_dir, &file_stem(name), &groups[name]) {
            groups.remove(name);
            return Err(CreateGroupError::Io(error));
        }
        Ok(())
    }

    /// Applies a delta to one group — context and child removals
    /// first, then additions (a name in both ends up a member), then
    /// the description — validates the nesting that results, and
    /// persists. Nothing applies unless everything does: a refused
    /// nesting or a failed persist restores the previous record, the
    /// update twin of [`rollback_meta`].
    pub fn update_group(
        &self,
        name: &str,
        description: Option<String>,
        add_contexts: BTreeSet<String>,
        remove_contexts: BTreeSet<String>,
        add_groups: BTreeSet<String>,
        remove_groups: BTreeSet<String>,
    ) -> Result<GroupRecord, UpdateGroupError> {
        let mut groups = self.0.groups.write().unwrap();
        if !groups.contains_key(name) {
            return Err(UpdateGroupError::NotFound);
        }
        if !add_contexts.is_empty() {
            let registry = self.0.registry.read().unwrap();
            if let Some(missing) =
                first_missing(&add_contexts, |context| registry.contains_key(context))
            {
                return Err(UpdateGroupError::NoSuchContext(missing.clone()));
            }
        }
        // `name` itself IS registered, so a self-add passes this gate —
        // and lands in the validator's lap as the smallest cycle.
        if let Some(missing) = first_missing(&add_groups, |child| groups.contains_key(child)) {
            return Err(UpdateGroupError::NoSuchGroup(missing.clone()));
        }
        let record = groups.get_mut(name).unwrap();
        let previous = record.clone();
        for context in &remove_contexts {
            record.contexts.remove(context);
        }
        record.contexts.extend(add_contexts);
        for group in &remove_groups {
            record.groups.remove(group);
        }
        record.groups.extend(add_groups);
        if let Some(description) = description {
            record.description = description;
        }
        if let Err(violation) = groups::validate_nesting(&groups) {
            *groups.get_mut(name).unwrap() = previous;
            return Err(UpdateGroupError::Nesting(violation));
        }
        if let Err(error) = groups::write_group(&self.0.data_dir, &file_stem(name), &groups[name]) {
            *groups.get_mut(name).unwrap() = previous;
            return Err(UpdateGroupError::Io(error));
        }
        Ok(groups[name].clone())
    }

    /// Removes a group — the bundling only, never the member contexts
    /// nor the child groups. `None` for an unknown name, mirroring
    /// [`AppState::delete`]. Parents naming the group are swept inside
    /// the same critical section, so no reader ever observes a
    /// dangling child.
    ///
    /// One file, so no deletion marker: the memory drop and the unlink
    /// are it. The weaker guarantee is deliberate and priced in — if
    /// the unlink fails, the surviving file re-registers the group at
    /// the next boot, and the error message says so.
    pub fn delete_group(&self, name: &str) -> Option<io::Result<()>> {
        let mut groups = self.0.groups.write().unwrap();
        groups.remove(name)?;
        // Nesting must not outlive the child — the same sweep a
        // deleted context gets, on the child field, under the write
        // lock already held (best effort past the removal; boot
        // reconciliation heals a sweep that could not persist).
        sweep_membership(&self.0.data_dir, &mut groups, name, |record| {
            &mut record.groups
        });
        Some(groups::remove_group_file(
            &self.0.data_dir,
            &file_stem(name),
        ))
    }

    /// One group's record by name, or `None` for an unknown group.
    pub fn group(&self, name: &str) -> Option<GroupRecord> {
        self.0.groups.read().unwrap().get(name).cloned()
    }

    /// Union of every context reachable from the named groups — direct
    /// members plus everything nested children bundle, transitively.
    /// The scoped write gate judges a group by what it ADDRESSES, so
    /// this is its view; unknown names contribute nothing.
    pub fn group_context_closures<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
    ) -> BTreeSet<String> {
        groups::context_closure(&self.0.groups.read().unwrap(), names)
    }

    /// [`group_context_closures`] with existence semantics: the first
    /// name that is not a registered group comes back as the error
    /// instead of contributing nothing. The cross-context searches
    /// resolve their `groups` targets here — a caller who NAMES a group
    /// deserves a `no_group` refusal, not a silently empty search —
    /// checked and walked under one lock acquisition so a concurrent
    /// group delete cannot slip between the two.
    pub fn resolve_groups(&self, names: &[String]) -> Result<BTreeSet<String>, String> {
        let groups = self.0.groups.read().unwrap();
        if let Some(missing) = first_missing(names, |name| groups.contains_key(name)) {
            return Err(missing.clone());
        }
        Ok(groups::context_closure(
            &groups,
            names.iter().map(String::as_str),
        ))
    }

    /// One name-ordered page of groups plus the cursor-independent
    /// total. Scope filtering is the API layer's business, as with
    /// [`AppState::directory`] — but unlike the context directory,
    /// which clones only `Arc` handles and can hand over the whole
    /// map, a group's record IS its data, so the page is cut here
    /// under the read lock and only the survivors are cloned.
    pub fn group_page(
        &self,
        after: Option<&str>,
        limit: usize,
    ) -> (usize, Vec<(String, GroupRecord)>) {
        use std::ops::Bound;

        let groups = self.0.groups.read().unwrap();
        let start = match after {
            Some(after) => Bound::Excluded(after),
            None => Bound::Unbounded,
        };
        let page = groups
            .range::<str, _>((start, Bound::Unbounded))
            .take(limit)
            .map(|(name, record)| (name.clone(), record.clone()))
            .collect();
        (groups.len(), page)
    }

    /// Drops a deleted context out of every group, persisting each
    /// touched record. Called from [`AppState::delete`] with the
    /// deletion marker already durable; best effort past that point —
    /// a rewrite that fails leaves memory correct and the file stale,
    /// which the next boot's reconciliation heals.
    fn sweep_context_from_groups(&self, context_name: &str) {
        let mut groups = self.0.groups.write().unwrap();
        sweep_membership(&self.0.data_dir, &mut groups, context_name, |record| {
            &mut record.contexts
        });
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
        passages: BTreeMap<String, crate::passages::PassageSubmission>,
    ) -> Option<io::Result<crate::passages::StoreOutcome>> {
        let entry = self.lookup(name)?;
        let fence = entry.read_unless_deleted()?;
        let outcome = match self.entry_passages(&entry, &file_stem(name)) {
            Ok(store) => {
                let sources: Vec<String> = passages.keys().cloned().collect();
                let stored = store.store(passages);
                if stored.is_ok() {
                    // Every store lock is released again; fold the new
                    // paragraphs into the resident index.
                    self.refresh_bm25(&entry, &store, &sources);
                    entry.passages_embed_dirty.store(true, Ordering::Relaxed);
                }
                stored
            }
            Err(error) => Err(error),
        };
        drop(fence);
        // Passage text is resident now; give the budget a chance to
        // evict something (possibly this context's own cold graph).
        self.enforce_budget(name);
        Some(outcome)
    }

    /// Dereferences source ids (as found on attributions) back to their
    /// registered passages, reporting the ids that have none.
    #[allow(clippy::type_complexity)]
    pub fn lookup_passages(
        &self,
        name: &str,
        sources: &[String],
    ) -> Option<io::Result<(BTreeMap<String, String>, Vec<String>)>> {
        let entry = self.lookup(name)?;
        let _fence = entry.read_unless_deleted()?;
        let store = match self.entry_passages(&entry, &file_stem(name)) {
            Ok(store) => store,
            Err(error) => return Some(Err(error)),
        };
        let mut passages = BTreeMap::new();
        let mut missing = Vec::new();
        for source in sources {
            match store.get(source) {
                Some(record) => {
                    passages.insert(source.clone(), record.text.to_string());
                }
                None => missing.push(source.clone()),
            }
        }
        Some(Ok((passages, missing)))
    }

    /// Resolves one `(source, paragraph index)` pair to its verbatim
    /// excerpt — the located counterpart of `lookup_passages`'
    /// whole-document dereference. Reuses `PassageRecord::paragraph`,
    /// the same slice `search_passages` goes through for its hits, so
    /// the two can never disagree about what a paragraph's text is.
    /// The section label comes from the same resident record via
    /// `section_for`, `None` when the index falls outside every
    /// section the source's import stored.
    pub fn citation(
        &self,
        name: &str,
        source: &str,
        index: u32,
    ) -> Option<io::Result<CitationLookup>> {
        let entry = self.lookup(name)?;
        let _fence = entry.read_unless_deleted()?;
        let store = match self.entry_passages(&entry, &file_stem(name)) {
            Ok(store) => store,
            Err(error) => return Some(Err(error)),
        };
        let Some(record) = store.get(source) else {
            return Some(Ok(CitationLookup::UnknownSource));
        };
        let Some((_, text)) = record.paragraph(index as usize) else {
            return Some(Ok(CitationLookup::IndexOutOfRange));
        };
        let section = record.section_for(index as usize).map(str::to_string);
        Some(Ok(CitationLookup::Found(text.to_string(), section)))
    }

    /// Resolves `(source, paragraph)` locators — as found on
    /// attributions — to the section label governing each, batching
    /// every pair an association-bearing response needs into one
    /// passage-store load rather than one per attribution. Best-effort:
    /// an unknown context, a deleted entry, or a passage-store load
    /// failure all resolve to an empty map rather than an error.
    /// Association reads (recall, query, explore, activate,
    /// unreachable_from) are graph reads first; a section label is
    /// enrichment on top, not a hard dependency the way `citation`'s
    /// text lookup is. A pair with no covering marker is simply absent
    /// from the map — the same null-means-nothing contract
    /// `Attribution::paragraph` already makes, never a fabricated
    /// label. An empty `locators` skips the passage-store load
    /// entirely, so a graph-only response (no attribution carries a
    /// paragraph) never touches passages.
    pub fn resolve_sections(
        &self,
        name: &str,
        locators: impl Iterator<Item = (String, u32)>,
    ) -> HashMap<(String, u32), String> {
        let mut locators = locators.peekable();
        if locators.peek().is_none() {
            return HashMap::new();
        }
        let Some(entry) = self.lookup(name) else {
            return HashMap::new();
        };
        let Some(_fence) = entry.read_unless_deleted() else {
            return HashMap::new();
        };
        let store = match self.entry_passages(&entry, &file_stem(name)) {
            Ok(store) => store,
            Err(error) => {
                tracing::warn!(
                    context = %name,
                    %error,
                    "section resolution: passage store load failed; continuing without section labels"
                );
                return HashMap::new();
            }
        };
        locators
            .filter_map(|(source, paragraph)| {
                let record = store.get(&source)?;
                let section = record.section_for(paragraph as usize)?;
                Some(((source, paragraph), section.to_string()))
            })
            .collect()
    }

    /// The source ids that currently have a registered passage.
    pub fn passage_sources(&self, name: &str) -> Option<io::Result<Vec<String>>> {
        let entry = self.lookup(name)?;
        let _fence = entry.read_unless_deleted()?;
        Some(
            self.entry_passages(&entry, &file_stem(name))
                .map(|store| store.source_ids()),
        )
    }

    /// The resident passage store, loaded on first access (the same
    /// lazy shape as [`AppState::entry_vectors`]). Callers hold the
    /// entry's read fence, so the load can never race the delete that
    /// removes the files. A load failure is an error, not an empty
    /// store: the snapshot and log hold acknowledged passages.
    fn entry_passages(
        &self,
        entry: &Entry,
        stem: &str,
    ) -> io::Result<Arc<crate::passages::PassageStore>> {
        let mut slot = entry.passages.lock().unwrap();
        if let Some(store) = slot.as_ref() {
            return Ok(Arc::clone(store));
        }
        // A store whose last load failed is quarantined exactly like a
        // graph image (`ensure_hot`): answer the remembered refusal
        // while it is fresh instead of re-reading a broken snapshot on
        // every passage request.
        {
            let failure = entry.passages_load_failure.lock().unwrap();
            if let Some((failed_at, refusal)) = &*failure
                && failed_at.elapsed() < LOAD_FAILURE_RETRY
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{refusal} (quarantined after the failed load; the disk is \
                         retried at most every {}s)",
                        LOAD_FAILURE_RETRY.as_secs()
                    ),
                ));
            }
        }
        // The live server heals a torn tail on load; the torn indicator
        // is for read-only inspection, so drop it here.
        let (store, _torn) = crate::passages::PassageStore::load(
            passages_path(&self.0.data_dir, stem),
            &sources_path(&self.0.data_dir, stem),
            passages_wal_path(&self.0.data_dir, stem),
            self.0.passages_wal_max_bytes,
            true,
        )
        .inspect_err(|error| {
            *entry.passages_load_failure.lock().unwrap() =
                Some((std::time::Instant::now(), error.to_string()));
        })?;
        *entry.passages_load_failure.lock().unwrap() = None;
        let store = Arc::new(store);
        *slot = Some(Arc::clone(&store));
        Ok(store)
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
        let touched = self.logged_write(
            name,
            std::slice::from_ref(&op),
            |context| context.retract_source(source).unwrap_or(0),
            // The single RetractSource op never fails to apply.
            |_| 1,
        )?;

        let Some(entry) = self.lookup(name) else {
            // Raced with a delete; there is nothing left to clean up.
            return Ok((touched, false));
        };
        let Some(_fence) = entry.read_unless_deleted() else {
            // Same race, one step later: the delete beat us to the lock.
            return Ok((touched, false));
        };
        // The graph retraction above already succeeded; a passage-side
        // failure must not turn it into an error, only into an honest
        // `passage_removed: false`.
        let passage_removed = match self.entry_passages(&entry, &file_stem(name)) {
            Ok(store) => match store.retract(source) {
                Ok(removed) => {
                    if removed {
                        self.refresh_bm25(
                            &entry,
                            &store,
                            std::slice::from_ref(&source.to_string()),
                        );
                        entry.passages_embed_dirty.store(true, Ordering::Relaxed);
                    }
                    removed
                }
                Err(error) => {
                    tracing::warn!("passage for '{source}' not removed from disk: {error}");
                    false
                }
            },
            Err(error) => {
                tracing::warn!("passages for '{name}' unavailable during retract: {error}");
                false
            }
        };
        Ok((touched, passage_removed))
    }

    /// Full-text search over the registered passages — the second lane
    /// beside the graph, for knowledge that does not decompose into
    /// triples (procedures, conditions, discourse). Scored per
    /// PARAGRAPH: the ranking unit is what an answer actually cites,
    /// and a long document no longer buries its best paragraph inside
    /// its own length normalization.
    ///
    /// Two ranking lanes, fused: the lexical one runs on the resident
    /// [`crate::bm25::Bm25Index`] (built once per residency, updated in
    /// place by store/retract — a query never re-tokenizes the corpus),
    /// and, where paragraph embedding is on, a semantic one sweeps the
    /// paragraph vectors with the query embedded as
    /// [`EmbedPurpose::Query`]. Fusion is reciprocal rank (k = 60):
    /// rank-based, so the two lanes' incomparable score scales never
    /// need reconciling. Each hit reports its per-lane rank and raw
    /// score — the server presents evidence, the reading LLM judges,
    /// same as resolve's tiers. With the vector lane off or its
    /// provider failing, results degrade to pure BM25 (raw score,
    /// evidence intact) — a broken decoration never breaks the answer.
    ///
    /// Texts resolve through the store afterwards, hash-checked: a
    /// paragraph that changed between a lane's view and now is dropped
    /// rather than served with a stale score against fresh text.
    pub fn search_passages(
        &self,
        name: &str,
        query: &str,
        limit: usize,
    ) -> Option<io::Result<Vec<PassageSearchHit>>> {
        const RRF_K: f32 = 60.0;

        let entry = self.lookup(name)?;
        if limit == 0 {
            return entry.read_unless_deleted().map(|_| Ok(Vec::new()));
        }
        let query_grams: Vec<u64> = {
            let mut seen = std::collections::HashSet::new();
            passage_terms(query)
                .into_iter()
                .filter(|gram| seen.insert(*gram))
                .collect()
        };
        if query_grams.is_empty() {
            return entry.read_unless_deleted().map(|_| Ok(Vec::new()));
        }
        // Both lanes over-fetch: fusion can promote a hit neither lane
        // put in its own top `limit`, and the staleness checks below
        // may drop stragglers.
        let pool = limit.saturating_mul(4).max(50);

        // The semantic lane's query embedding runs BEFORE any lock: a
        // provider round trip must never extend the fence below.
        let cue = if self.passage_embedding_enabled() {
            let embedder = self.0.embedder.clone().expect("enabled implies a provider");
            match self.cue_vector(&*embedder, query) {
                Ok(vector) => Some(vector),
                Err(error) => {
                    // Degrade, loudly: the lexical lane still answers.
                    tracing::warn!(
                        context = %name,
                        error,
                        "passage query embedding failed; serving the lexical lane alone"
                    );
                    None
                }
            }
        } else {
            None
        };

        // Everything below holds the read fence: eviction and deletion
        // are excluded for the whole search, so `store` IS the resident
        // store throughout — an index built from a handle that predates
        // an eviction would silently hide writes that landed in the
        // freshly reloaded store until the next rebuild.
        let fence = entry.read_unless_deleted()?;
        let store = match self.entry_passages(&entry, &file_stem(name)) {
            Ok(store) => store,
            Err(error) => return Some(Err(error)),
        };

        // Lexical lane: ensure a resident index — build on the
        // residency's first search, rebuild when tombstones have piled
        // up. Double-checked so concurrent first searches build once.
        let stale = {
            let guard = entry.bm25.read().unwrap();
            match &*guard {
                None => true,
                Some(index) => index.needs_reclaim(),
            }
        };
        if stale {
            let mut guard = entry.bm25.write().unwrap();
            let rebuild = match &*guard {
                None => true,
                Some(index) => index.needs_reclaim(),
            };
            if rebuild {
                let records = store.snapshot();
                let built_at = std::time::Instant::now();
                let index = if guard.take().is_some() {
                    // Tombstone reclamation: rebuild fresh from the store.
                    entry.bm25_dirty.store(true, Ordering::Relaxed);
                    crate::bm25::Bm25Index::build(&records)
                } else if let Some(mut loaded) =
                    crate::bm25::Bm25Index::load(&bm25_path(&self.0.data_dir, &file_stem(name)))
                {
                    // A sidecar spares the re-tokenization, but its save
                    // cadence is the flush tick — repair whatever drifted
                    // (per source, both directions) instead of trusting
                    // or rebuilding wholesale.
                    let mut disk = loaded.source_digests();
                    let mut drifted = 0usize;
                    for (source, record) in &records {
                        if disk.remove(source) != Some(crate::bm25::record_digest(record)) {
                            loaded.upsert_source(source, record);
                            drifted += 1;
                        }
                    }
                    drifted += disk.len();
                    for source in disk.keys() {
                        loaded.remove_source(source);
                    }
                    if drifted > 0 {
                        entry.bm25_dirty.store(true, Ordering::Relaxed);
                    }
                    loaded
                } else {
                    entry.bm25_dirty.store(true, Ordering::Relaxed);
                    crate::bm25::Bm25Index::build(&records)
                };
                *guard = Some(index);
                tracing::info!(
                    context = %name,
                    sources = records.len(),
                    ms = built_at.elapsed().as_millis() as u64,
                    "BM25 index ready",
                );
            }
        }
        let lexical = {
            let guard = entry.bm25.read().unwrap();
            let index = guard.as_ref().expect("index was just built");
            index.search(&query_grams, pool)
        };

        // Semantic lane: sweep the paragraph vectors with the
        // pre-embedded query, then drop candidates below the same floor
        // semantic_resolve applies to its own cosine matches — context
        // setting beats the server default (`fence` is already this
        // entry's read lock, taken above; search_passages has no
        // one-call override to slot in ahead of it).
        let semantic: Vec<(String, u32, u64, f32)> = match &cue {
            Some(cue) => {
                let vectors = self.entry_passage_vectors(&entry, &file_stem(name));
                let model_matches = self
                    .0
                    .embedder
                    .as_ref()
                    .is_some_and(|embedder| vectors.model == embedder.model());
                if vectors.is_empty() || !model_matches {
                    Vec::new()
                } else {
                    let floor = fence
                        .meta
                        .semantic_floor
                        .unwrap_or(self.0.default_semantic_floor)
                        .clamp(0.0, 1.0);
                    vectors
                        .top_matches(cue, pool)
                        .into_iter()
                        .filter(|&(_, score)| score >= floor)
                        .map(|(key, score)| (key.source.clone(), key.index, key.hash, score))
                        .collect()
                }
            }
            None => Vec::new(),
        };

        // Fuse by rank, then validate EACH LANE against the store's
        // current paragraph: every lane scored the text it saw, and
        // vectors routinely lag the text between refreshes — a stale
        // lane must neither smuggle its outdated score onto fresh text
        // nor veto the other lane's fresh match, so each loses exactly
        // its own evidence (and its fusion term). The top-level score
        // stays the raw BM25 number when no semantic lane ran, so a
        // lexical-only deployment keeps its historical score semantics.
        let fused = !semantic.is_empty();
        let mut accumulated: HashMap<(String, u32), FusedHit> = HashMap::new();
        for (rank, (source, index, hash, score)) in lexical.into_iter().enumerate() {
            accumulated.entry((source, index)).or_default().bm25 = Some((rank + 1, score, hash));
        }
        for (rank, (source, index, hash, score)) in semantic.into_iter().enumerate() {
            // A paragraph can hit this lane several times (its own text
            // row plus its doc2query question rows); ranks ascend, so
            // the first arrival is its best showing and later ones must
            // not overwrite it.
            let slot = accumulated.entry((source, index)).or_default();
            if slot.vector.is_none() {
                slot.vector = Some((rank + 1, score, hash));
            }
        }

        let rrf =
            |lane: &Option<(usize, f32)>| lane.map_or(0.0, |(rank, _)| 1.0 / (RRF_K + rank as f32));
        let mut hits: Vec<PassageSearchHit> = Vec::new();
        for ((source, index), lanes) in accumulated {
            let Some(record) = store.get(&source) else {
                continue;
            };
            let Some((span, text)) = record.paragraph(index as usize) else {
                continue;
            };
            let bm25 = lanes
                .bm25
                .filter(|&(.., hash)| hash == span.hash)
                .map(|(rank, score, _)| (rank, score));
            let vector = lanes
                .vector
                .filter(|&(.., hash)| hash == span.hash)
                .map(|(rank, score, _)| (rank, score));
            if bm25.is_none() && vector.is_none() {
                continue;
            }
            let score = if fused {
                rrf(&bm25) + rrf(&vector)
            } else {
                bm25.map(|(_, score)| score).unwrap_or(0.0)
            };
            hits.push(PassageSearchHit {
                source,
                index,
                score,
                text: text.to_string(),
                bm25,
                vector,
            });
        }
        hits.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.source.cmp(&b.source))
                .then_with(|| a.index.cmp(&b.index))
        });
        hits.truncate(limit);
        Some(Ok(hits))
    }

    /// One provider round trip, timed into the embed-latency histogram
    /// whatever the outcome — the ok/failed counters cannot tell a
    /// slow provider from a down one; the histogram can.
    fn timed_embed(
        &self,
        embedder: &dyn EmbeddingProvider,
        texts: &[&str],
        purpose: EmbedPurpose,
    ) -> Result<Vec<Vec<f32>>, String> {
        let started = std::time::Instant::now();
        let outcome = embedder.embed(texts, purpose);
        self.0.metrics.record_embed_latency(started.elapsed());
        outcome
    }

    /// The query side of every embedding lookup: process cache first,
    /// provider (as [`EmbedPurpose::Query`]) on a miss. No lock is held
    /// across the provider call.
    fn cue_vector(
        &self,
        embedder: &dyn EmbeddingProvider,
        cue: &str,
    ) -> Result<Arc<Vec<f32>>, String> {
        if let Some(vector) = self.0.cue_cache.lock().unwrap().get(cue) {
            return Ok(vector);
        }
        match self.timed_embed(embedder, &[cue], EmbedPurpose::Query) {
            Ok(mut vectors) => {
                self.0.metrics.record_embed_resolve(true);
                let vector = Arc::new(vectors.pop().unwrap_or_default());
                self.0
                    .cue_cache
                    .lock()
                    .unwrap()
                    .insert(cue.to_string(), Arc::clone(&vector));
                Ok(vector)
            }
            Err(error) => {
                self.0.metrics.record_embed_resolve(false);
                Err(error)
            }
        }
    }

    /// The paragraph vector sidecar, loaded on first use and held until
    /// refresh replaces it or eviction clears it.
    fn entry_passage_vectors(&self, entry: &Entry, stem: &str) -> Arc<PassageVectorStore> {
        let mut cached = entry.passage_vectors.lock().unwrap();
        match &*cached {
            Some(store) => Arc::clone(store),
            None => {
                let store = Arc::new(PassageVectorStore::load(&pvectors_path(
                    &self.0.data_dir,
                    stem,
                )));
                *cached = Some(Arc::clone(&store));
                store
            }
        }
    }

    /// Folds freshly stored/retracted sources into the resident BM25
    /// index, if one exists (a cold index just rebuilds on its next
    /// search). Called with every passage-store lock RELEASED — the
    /// documented order for `Entry::bm25`.
    fn refresh_bm25(
        &self,
        entry: &Entry,
        store: &crate::passages::PassageStore,
        sources: &[String],
    ) {
        let mut guard = entry.bm25.write().unwrap();
        let Some(index) = guard.as_mut() else {
            return;
        };
        for source in sources {
            match store.get(source) {
                Some(record) => index.upsert_source(source, &record),
                None => index.remove_source(source),
            }
        }
        if !sources.is_empty() {
            entry.bm25_dirty.store(true, Ordering::Relaxed);
        }
    }

    /// Persists a dirty resident index. Derived data: a failed save
    /// re-marks and warns — the next tick retries, and the worst case
    /// is one re-tokenization on some future load. The fence keeps a
    /// racing delete from finding its file recreated.
    fn flush_bm25(&self, name: &str, entry: &Entry) {
        if !entry.bm25_dirty.swap(false, Ordering::Relaxed) {
            return;
        }
        let Some(_fence) = entry.read_unless_deleted() else {
            return;
        };
        let bytes = {
            let guard = entry.bm25.read().unwrap();
            match &*guard {
                Some(index) => index.to_bytes(),
                // Dropped since (eviction): its own save path ran.
                None => return,
            }
        };
        if let Err(error) = write_atomic(&bm25_path(&self.0.data_dir, &file_stem(name)), &bytes) {
            entry.bm25_dirty.store(true, Ordering::Relaxed);
            tracing::warn!("BM25 index for '{name}' not persisted (will retry): {error}");
        }
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
        let Some(embedder) = self.0.embedder.clone() else {
            return Some(Err(
                "no embedding provider is configured (set TAGURU_EMBED_URL and TAGURU_EMBED_MODEL)"
                    .to_string(),
            ));
        };
        let entry = self.lookup(name)?;
        // One refresh per context at a time (see Entry::vectors_refresh
        // for why); held across the gloss read too, not just the embed
        // and merge, so no overlapping refresh can be mid-flight against
        // a gloss state this one hasn't seen yet.
        let _serial = entry.vectors_refresh.lock().unwrap();
        let glosses = match self.read_context(name, |context| {
            let concepts: Vec<(String, String)> = context
                .concept_names()
                .into_iter()
                .map(|name| {
                    let gloss = context
                        .concept_gloss(name, Context::GLOSS_FACTS)
                        .unwrap_or_else(|| name.to_string());
                    (name.to_string(), gloss)
                })
                .collect();
            let labels: Vec<(String, String)> = context
                .labels()
                .into_iter()
                .map(|name| {
                    let gloss = context
                        .label_gloss(name, Context::GLOSS_EXAMPLES)
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
        let path = vectors_path(&self.0.data_dir, &file_stem(name));

        // Diff and embed while still holding `_serial`, not the entry's
        // data lock — provider round trips can take seconds and must
        // not block graph reads/writes. `_serial` (not this) is what
        // keeps two overlapping refreshes from racing: only one can be
        // here at a time, so the diff below always runs against
        // whatever the previous refresh (if any) already published.
        let existing = VectorStore::load(&path);
        let mut fresh_model = existing.model != embedder.model();
        let mut embedded_concepts =
            match self.embed_stale(&*embedder, &existing.concepts, &concepts, fresh_model) {
                Ok(embedded) => embedded,
                Err(error) => return Some(Err(error)),
            };
        let mut embedded_labels =
            match self.embed_stale(&*embedder, &existing.labels, &labels, fresh_model) {
                Ok(embedded) => embedded,
                Err(error) => return Some(Err(error)),
            };
        // The model NAME is the staleness discriminator, but a provider
        // can change output width behind a stable name (a backend swap
        // behind the same proxy or gateway). Old-width rows carried next
        // to new-width ones would feed `similarity` mismatched
        // dimensions — no error, no score — so a width disagreement
        // stales the whole table, exactly as if the model were renamed.
        let width = |table: &VectorTable| table.values().map(|(_, vector)| vector.len()).next();
        let carried_width = width(&existing.concepts).or_else(|| width(&existing.labels));
        let mut fresh_width = width(&embedded_concepts).or_else(|| width(&embedded_labels));
        // Unchanged hashes embed nothing, which would leave the width
        // change of exactly this scenario — backend swap, no gloss
        // edits — undetectable forever. One probe embedding per no-op
        // refresh keeps that from hiding.
        if !fresh_model
            && carried_width.is_some()
            && fresh_width.is_none()
            && let Some((_, gloss)) = concepts.first().or_else(|| labels.first())
        {
            match self.timed_embed(embedder.as_ref(), &[gloss.as_str()], EmbedPurpose::Index) {
                Ok(vectors) => {
                    self.0.metrics.record_embed_refresh(true);
                    fresh_width = vectors.first().map(Vec::len);
                }
                Err(error) => {
                    self.0.metrics.record_embed_refresh(false);
                    return Some(Err(error));
                }
            }
        }
        if !fresh_model
            && let (Some(carried), Some(fresh)) = (carried_width, fresh_width)
            && carried != fresh
        {
            tracing::warn!(
                context = name,
                model = embedder.model(),
                carried,
                fresh,
                "embedding width changed under an unchanged model name; re-embedding every gloss"
            );
            fresh_model = true;
            embedded_concepts =
                match self.embed_stale(&*embedder, &existing.concepts, &concepts, true) {
                    Ok(embedded) => embedded,
                    Err(error) => return Some(Err(error)),
                };
            embedded_labels = match self.embed_stale(&*embedder, &existing.labels, &labels, true) {
                Ok(embedded) => embedded,
                Err(error) => return Some(Err(error)),
            };
        }
        let newly_embedded = embedded_concepts.len() + embedded_labels.len();

        // Publish under the entry lock (a delete that may have won it
        // must not see its sidecar recreated) — `_serial` above, held
        // since before the gloss read, is what makes this
        // read-modify-write race-free, not this lock by itself.
        let _guard = entry.lock_unless_deleted()?;
        let mut store = VectorStore::load(&path);
        // `fresh_model` also covers the width change above: rows for
        // names that have since left the graph must not linger at the
        // old width either.
        if fresh_model || store.model != embedder.model() {
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
            match self.timed_embed(embedder, &texts, EmbedPurpose::Index) {
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

    /// Whether the vector lane over paragraphs is on at all: a provider
    /// is configured AND the operator opted the corpus in
    /// (`TAGURU_EMBED_PASSAGES`).
    pub fn passage_embedding_enabled(&self) -> bool {
        self.0.embed_passages && self.0.embedder.is_some()
    }

    /// Contexts whose passages changed since their last embedding
    /// refresh — the auto-refresh ticker's work list. Claiming is the
    /// caller's job via [`AppState::refresh_passage_embeddings`].
    pub fn passage_embed_dirty_names(&self) -> Vec<String> {
        self.snapshot()
            .into_iter()
            .filter(|(_, entry)| entry.passages_embed_dirty.load(Ordering::Relaxed))
            .map(|(name, _)| name)
            .collect()
    }

    /// Embeds every stored paragraph (`EmbedPurpose::Index`) into the
    /// `{stem}.pvectors.bin` sidecar: the vector lane's index side.
    /// Diff-driven like the gloss refresh — a paragraph whose FNV-1a
    /// hash already has a row under the current model is carried
    /// forward, a vanished paragraph's row is dropped (retraction
    /// pruning falls out of the rebuild), and only the rest go to the
    /// provider, 128 per call. The sidecar is written AT MOST ONCE per
    /// refresh: writing per batch would multiply a large store's bytes
    /// across the whole backfill. A provider failure partway persists
    /// what did land and reports the error — the next refresh continues
    /// from there instead of re-buying the same vectors.
    pub fn refresh_passage_embeddings(
        &self,
        name: &str,
    ) -> Option<Result<PassageRefreshOutcome, String>> {
        let Some(embedder) = self.0.embedder.clone() else {
            return Some(Err(
                "no embedding provider is configured (set TAGURU_EMBED_URL and TAGURU_EMBED_MODEL)"
                    .to_string(),
            ));
        };
        if !self.0.embed_passages {
            return Some(Err(
                "passage embedding is disabled (set TAGURU_EMBED_PASSAGES=1)".to_string(),
            ));
        }
        let entry = self.lookup(name)?;
        // One refresh per context at a time (see Entry::passage_refresh
        // for why); the diff below makes the loser's pass a no-op.
        let _serial = entry.passage_refresh.lock().unwrap();
        // Claim the dirty flag up front: work that lands mid-refresh
        // re-marks it, so the ticker returns — never lost, never
        // double-claimed.
        entry.passages_embed_dirty.store(false, Ordering::Relaxed);
        let store = {
            let _fence = entry.read_unless_deleted()?;
            match self.entry_passages(&entry, &file_stem(name)) {
                Ok(store) => store,
                Err(error) => {
                    // The claim above must not eat the work: a store
                    // that cannot load now still needs its refresh once
                    // it can.
                    entry.passages_embed_dirty.store(true, Ordering::Relaxed);
                    return Some(Err(error.to_string()));
                }
            }
        };
        let records = store.snapshot();
        let path = pvectors_path(&self.0.data_dir, &file_stem(name));
        let existing = PassageVectorStore::load(&path);
        let fresh_model = existing.model != embedder.model();
        let carried: HashMap<(&str, u32, u64, Option<u64>), &[f32]> = if fresh_model {
            HashMap::new()
        } else {
            existing
                .iter()
                .map(|(key, row)| {
                    (
                        (key.source.as_str(), key.index, key.hash, key.question_hash),
                        row,
                    )
                })
                .collect()
        };

        // Deterministic walk — snapshot() is sorted by source, spans by
        // position, questions by paragraph — so the same rows win the
        // limit run after run. Each paragraph offers its own text row
        // and then one row per stored question, every one keyed to the
        // PARAGRAPH (hash included) with the question's own hash as the
        // discriminator.
        let mut fresh = PassageVectorStore::new(embedder.model());
        let mut to_embed: Vec<(PassageKey, String)> = Vec::new();
        let mut skipped_over_limit = 0usize;
        for (source, record) in &records {
            for (span, text) in record.paragraph_texts() {
                let question_rows = record
                    .questions
                    .iter()
                    .filter(|&&(paragraph, _)| paragraph == span.index)
                    .map(|(_, question)| (Some(fnv1a(question)), question.as_str()));
                for (question_hash, row_text) in std::iter::once((None, text)).chain(question_rows)
                {
                    // Stored before the write surfaces refused empty
                    // question text, an empty row would be sent to the
                    // provider verbatim — and providers refuse
                    // zero-length input, failing that row's whole
                    // chunk and abandoning the pass at the same spot
                    // on every retry. Empty text retrieves nothing
                    // anyway: skip it.
                    if row_text.is_empty() {
                        continue;
                    }
                    if fresh.len() + to_embed.len() >= self.0.passage_vector_limit {
                        skipped_over_limit += 1;
                        continue;
                    }
                    let key = PassageKey {
                        source: source.clone(),
                        index: span.index,
                        hash: span.hash,
                        question_hash,
                    };
                    match carried.get(&(source.as_str(), span.index, span.hash, question_hash)) {
                        Some(row) => fresh.push(key, row.to_vec()),
                        None => to_embed.push((key, row_text.to_string())),
                    }
                }
            }
        }

        let mut embedded = 0usize;
        let mut failure: Option<String> = None;
        for chunk in to_embed.chunks(128) {
            let texts: Vec<&str> = chunk.iter().map(|(_, text)| text.as_str()).collect();
            match self.timed_embed(embedder.as_ref(), &texts, EmbedPurpose::Index) {
                Ok(vectors) => {
                    self.0.metrics.record_embed_refresh(true);
                    for ((key, _), vector) in chunk.iter().zip(vectors) {
                        fresh.push(key.clone(), vector);
                        embedded += 1;
                    }
                }
                Err(error) => {
                    self.0.metrics.record_embed_refresh(false);
                    failure = Some(error);
                    break;
                }
            }
        }

        // Publish under the entry lock (a delete that won it must not
        // see its files recreated), and only when something changed —
        // an all-carried refresh is a no-op, not a rewrite.
        let changed =
            embedded > 0 || fresh.len() != existing.len() || (fresh_model && !fresh.is_empty());
        let _guard = entry.lock_unless_deleted()?;
        if changed && let Err(error) = fresh.save(&path) {
            entry.passages_embed_dirty.store(true, Ordering::Relaxed);
            return Some(Err(format!("passage vectors not persisted: {error}")));
        }
        let total_rows = fresh.len();
        *entry.passage_vectors.lock().unwrap() = Some(Arc::new(fresh));
        match failure {
            Some(error) => {
                // What landed is durable; the rest stays claimed as work.
                entry.passages_embed_dirty.store(true, Ordering::Relaxed);
                Some(Err(error))
            }
            None => Some(Ok(PassageRefreshOutcome {
                embedded,
                total: total_rows,
                skipped_over_limit,
            })),
        }
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
        let cue_vector = match self.cue_vector(&*embedder, cue) {
            Ok(vector) => vector,
            Err(error) => return Some(Err(error)),
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
            // Saved so a load or persist failure below can restore the
            // pre-call state — without it, memory would hold fields
            // that never reached the sidecar, and a later, unrelated
            // successful update would persist them as a side effect.
            let previous = inner.meta.clone();
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
                rollback_meta(inner, previous);
                self.recount_entry(inner);
                return Some(Err(io::Error::other(error)));
            }
            // A pin toggle moves the entry into or out of the budget's
            // world; the estimate must follow.
            self.recount_entry(inner);
            let result = write_meta(
                &self.0.data_dir,
                &file_stem(name),
                &inner.meta,
                &inner.stats,
                &entry.usage.snapshot(),
            )
            .map(|()| inner.meta.clone());
            if result.is_err() {
                rollback_meta(inner, previous);
                self.recount_entry(inner);
            }
            result
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

    /// Whether a context exists, by registry membership. The
    /// cross-context search entrances vet their whole target list up
    /// front, so a mistyped name refuses before any context is
    /// searched; a context deleted between this check and its read is
    /// still caught by the read itself.
    pub fn context_exists(&self, name: &str) -> bool {
        self.lookup(name).is_some()
    }

    /// Runs a read-only operation on one context, loading it first if
    /// cold. A hot context is served under the SHARED lock, so
    /// concurrent reads of one context run in parallel — a long explore
    /// no longer makes every recall on the same context queue behind
    /// it. Only a cold load (and every write) takes the exclusive path;
    /// the cold load is real disk IO plus full-image validation, so it
    /// steps off the async runtime (see [`offload`]) — a post-restart
    /// burst of reads against distinct cold contexts must not consume
    /// the worker pool on synchronous loads.
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
        let result = offload(|| {
            let mut inner = entry.lock_unless_deleted().ok_or(AccessError::NotFound)?;
            ensure_hot(&self.0.data_dir, name, &mut inner, &self.0.metrics)
                .map_err(AccessError::Load)?;
            self.recount_entry(&mut inner);
            let Slot::Hot(context) = &inner.slot else {
                unreachable!("ensure_hot leaves the slot hot");
            };
            Ok(operate(context))
        })?;
        self.touch(&entry);
        self.enforce_budget(name);
        Ok(result)
    }

    /// Materializes everything one context's export stream renders
    /// from — graph, aliases, meta, passages — under a single fence.
    /// The graph half is read under `inner` (shared when hot,
    /// exclusive across a cold load), which every graph write also
    /// takes exclusively, so the associations and the passage
    /// snapshot cannot shear against a retraction or a batch apply.
    /// A concurrent passage store (which runs under the SHARED fence)
    /// can still land between the two — the passage text may be
    /// newer than the graph, never torn within one source.
    ///
    /// Cost note, like `compact_context`: the whole graph is walked and
    /// materialized into owned strings while the (shared) fence is held,
    /// so on a large context writers to THAT context wait out the
    /// materialization. It is a per-context stall, off the async runtime
    /// (`block_in_place` at the HTTP layer); a streaming, lock-light
    /// export is future work, not a v1 promise.
    pub fn export_context(&self, name: &str) -> Result<crate::export::ExportSnapshot, AccessError> {
        let entry = self.lookup(name).ok_or(AccessError::NotFound)?;
        let stem = file_stem(name);
        // Fast path: already resident, shared lock (mirrors read_context).
        {
            let inner = entry.inner.read().unwrap();
            match &inner.slot {
                Slot::Hot(context) => {
                    self.0.metrics.record_cache_hit();
                    let snapshot = self.export_snapshot(&entry, &stem, &inner.meta, context);
                    drop(inner);
                    self.touch(&entry);
                    self.enforce_budget(name);
                    return snapshot;
                }
                Slot::Deleted => return Err(AccessError::NotFound),
                Slot::Cold => {}
            }
        }
        // Slow path: load under the exclusive lock, as read_context does.
        // The `?` skips touch/enforce_budget on a load failure, matching
        // read_context and compact_context — a repeatedly-failing export
        // must not keep bumping a broken entry's LRU recency.
        let snapshot = offload(|| {
            let mut inner = entry.lock_unless_deleted().ok_or(AccessError::NotFound)?;
            ensure_hot(&self.0.data_dir, name, &mut inner, &self.0.metrics)
                .map_err(AccessError::Load)?;
            self.recount_entry(&mut inner);
            let Slot::Hot(context) = &inner.slot else {
                unreachable!("ensure_hot leaves the slot hot");
            };
            self.export_snapshot(&entry, &stem, &inner.meta, context)
        })?;
        self.touch(&entry);
        self.enforce_budget(name);
        Ok(snapshot)
    }

    /// The materialization inside [`AppState::export_context`]'s fence.
    /// Lock order: the caller holds `inner`; `entry_passages` takes
    /// `passages` — the documented `inner` → `passages` order.
    fn export_snapshot(
        &self,
        entry: &Entry,
        stem: &str,
        meta: &ContextMeta,
        context: &Context,
    ) -> Result<crate::export::ExportSnapshot, AccessError> {
        let passages = self
            .entry_passages(entry, stem)
            .map_err(|error| AccessError::Load(format!("passage store: {error}")))?
            .snapshot();
        let owned = |pairs: Vec<(&str, &str)>| -> Vec<(String, String)> {
            pairs
                .into_iter()
                .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
                .collect()
        };
        Ok(crate::export::ExportSnapshot {
            meta: meta.clone(),
            associations: context.query_any(&[], &[], &[]),
            concept_aliases: owned(context.concept_aliases()),
            label_aliases: owned(context.label_aliases()),
            passages,
        })
    }

    /// Rebuilds one context's image without its dead weight — the
    /// append-only storage's accumulated retracted edges, unlinked
    /// attributions, and arena slack (see [`Context::compacted`]) —
    /// then persists the fresh image immediately. Runs under the
    /// context's exclusive lock for the rebuild: requests to THIS
    /// context wait; every other context is untouched. Crash-safe by
    /// construction: the fresh context carries the old WAL watermark,
    /// so a crash before the flush lands simply boots the old image
    /// and replays the same log — compaction lost, nothing corrupted.
    pub fn compact_context(&self, name: &str) -> Result<CompactOutcome, AccessError> {
        let entry = self.lookup(name).ok_or(AccessError::NotFound)?;
        let outcome = offload(|| {
            let mut inner = entry.lock_unless_deleted().ok_or(AccessError::NotFound)?;
            ensure_hot(&self.0.data_dir, name, &mut inner, &self.0.metrics)
                .map_err(AccessError::Load)?;
            let Slot::Hot(context) = &inner.slot else {
                unreachable!("ensure_hot leaves the slot hot");
            };
            let bytes_before = context.footprint();
            let (mut fresh, stats) = context
                .compacted()
                .map_err(|full| AccessError::Load(format!("compaction refused: {full}")))?;
            // Config the image never carries, re-applied exactly as a
            // load would; the watermark keeps WAL replay monotonic.
            fresh.set_applied_seq(context.applied_seq());
            fresh.set_dice_floor(inner.meta.dice_floor);
            let bytes_after = fresh.footprint();
            inner.slot = Slot::Hot(Box::new(fresh));
            inner.image_generation += 1;
            let Slot::Hot(context) = &inner.slot else {
                unreachable!("just installed");
            };
            inner.stats = ContextStats::of(context);
            entry.dirty.store(true, Ordering::Relaxed);
            self.recount_entry(&mut inner);
            Ok(CompactOutcome {
                bytes_before,
                bytes_after,
                dead_edges: stats.dead_edges,
                aliases_dropped: stats.aliases_dropped,
            })
        })?;
        // Persist the shrunken image now (flush_entry takes its own
        // locks); a failure leaves the entry dirty for the next tick,
        // which is the flusher's ordinary retry story.
        self.flush_entry(name, &entry);
        self.touch(&entry);
        self.enforce_budget(name);
        Ok(outcome)
    }

    /// Test-only: rewinds any remembered load failure (graph image and
    /// passage store both) so the quarantine window can elapse without
    /// the test sleeping through it.
    #[cfg(test)]
    pub fn age_load_failures(&self, name: &str, by: std::time::Duration) {
        let entry = self.lookup(name).expect("the context must be registered");
        if let Some((failed_at, _)) = &mut entry.inner.write().unwrap().load_failure {
            *failed_at = failed_at
                .checked_sub(by)
                .expect("test ages within the Instant range");
        }
        if let Some((failed_at, _)) = &mut *entry.passages_load_failure.lock().unwrap() {
            *failed_at = failed_at
                .checked_sub(by)
                .expect("test ages within the Instant range");
        }
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
            self.flush_bm25(&name, &entry);
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
        // Skip a clean entry without locking it (the flusher's one
        // sanctioned lock-free read of `dirty`).
        if !entry.dirty.load(Ordering::Relaxed) {
            return false;
        }
        let (bytes, meta, stats, watermark, generation) = {
            let mut guard = entry.inner.write().unwrap();
            let inner = &mut *guard;
            // Claim the flush UNDER the lock. `flushing` gates concurrent
            // flushers (a tick against the shutdown flush) so the same
            // image is never staged twice, and — set before `dirty` is
            // cleared, both under this lock — it tells a racing eviction
            // "a flush of this entry is in flight, its bytes are not on
            // disk yet, persist me before dropping me." Clearing `dirty`
            // here (not out of the lock, as before) is what makes that
            // hand-off atomic: an evict that locks next sees `flushing`
            // set, never a bare "clean". A write that lands while we stage
            // re-sets `dirty`, so nothing is lost even with the WAL off,
            // where `wal_seq` does not move to flag the race.
            if entry.flushing.swap(true, Ordering::Relaxed) {
                return false;
            }
            entry.dirty.store(false, Ordering::Relaxed);
            let watermark = inner.wal_seq - 1;
            // Cold has nothing to write; Deleted must write nothing —
            // that snapshot predates a delete and the files are gone
            // for good.
            let Slot::Hot(context) = &mut inner.slot else {
                entry.flushing.store(false, Ordering::Relaxed);
                return false;
            };
            // The image about to be written reflects everything logged
            // so far: bake that in as the watermark, and those WAL
            // records are replay-inert even if truncation below never
            // happens (crash, unwritable file — doesn't matter).
            context.set_applied_seq(watermark);
            let stats = ContextStats::of(context);
            (
                context.to_bytes(),
                inner.meta.clone(),
                stats,
                watermark,
                inner.image_generation,
            )
        };

        let stem = file_stem(name);
        let image = image_path(&self.0.data_dir, &stem);
        let staged = match stage_bytes(&image, &bytes, false) {
            Ok(staged) => staged,
            Err(error) => {
                tracing::warn!("flush of context '{name}' failed (will retry): {error}");
                entry.dirty.store(true, Ordering::Relaxed);
                entry.flushing.store(false, Ordering::Relaxed);
                self.0.metrics.record_flush(name, false);
                return false;
            }
        };

        // Publication goes back under the lock: files for a name are
        // only ever created while holding its entry lock — the
        // tombstone invariant — so a delete that won the race while we
        // were staging must find us backing off, not recreating.
        let Some(mut guard) = entry.lock_unless_deleted() else {
            let _ = fs::remove_file(&staged);
            entry.flushing.store(false, Ordering::Relaxed);
            return false;
        };
        let inner = &mut *guard;
        // An eviction cooled us to Cold while we staged. Seeing `flushing`
        // it persisted this entry itself, so our staged snapshot is at
        // best a duplicate of what it wrote and, if a write beat the
        // evict, a step behind — publishing it would regress the image.
        // Drop the staged bytes and leave `dirty` as the evict (and any
        // racing write) left it. A compaction that swapped `slot` for a
        // fresh `Context` while we staged is the same story with the
        // variant unchanged — Hot in, Hot out — so `image_generation` is
        // what catches it: publishing our snapshot now would overwrite
        // the compacted image with the pre-compaction one it just
        // replaced, and stamp the entry's stats back to match. Compaction
        // already left `dirty` set, so backing off here costs nothing —
        // the next tick flushes the current image instead.
        if !matches!(inner.slot, Slot::Hot(_)) || inner.image_generation != generation {
            let _ = fs::remove_file(&staged);
            entry.flushing.store(false, Ordering::Relaxed);
            return false;
        }
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
        let published = match outcome {
            Ok(()) => {
                inner.stats = stats;
                self.0.metrics.record_flush(name, true);
                // `dirty` stays as claimed: clear (nothing raced) means the
                // image is current; a racing write re-set it and will
                // re-flush. Truncation is sound only when the image covers
                // the whole log — a write that landed mid-stage sits past
                // our watermark and its records must survive.
                if inner.wal_seq - 1 == watermark {
                    self.truncate_wal(name, inner);
                }
                true
            }
            Err(error) => {
                tracing::warn!("flush of context '{name}' failed (will retry): {error}");
                let _ = fs::remove_file(&staged);
                entry.dirty.store(true, Ordering::Relaxed);
                entry.usage_dirty.store(true, Ordering::Relaxed);
                self.0.metrics.record_flush(name, false);
                false
            }
        };
        entry.flushing.store(false, Ordering::Relaxed);
        published
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
    /// `operate` may apply fewer than `ops.len()` — `apply_in_order`
    /// stops at the first rejection, but the WAL above was already
    /// appended in full (durability can't wait on a result it doesn't
    /// have yet). Left alone, the untried tail would sit on disk
    /// looking exactly like an applied record: `ensure_hot`'s replay
    /// (`replay_op`) continues past a rejection where the live path
    /// stopped at the first one, so that tail would be tried
    /// independently — and could succeed — next time this context
    /// goes cold. `applied` reports how many ops actually landed so
    /// the excess can be trimmed back out before this returns.
    fn logged_write<T>(
        &self,
        name: &str,
        ops: &[WalOp],
        operate: impl FnOnce(&mut Context) -> T,
        applied: impl FnOnce(&T) -> usize,
    ) -> Result<T, AccessError> {
        let entry = self.lookup(name).ok_or(AccessError::NotFound)?;
        let mut wal_behind = false;
        let result = {
            // Same tombstone rule as with_hot: a delete that beat us to
            // this lock owns the name — appending here would recreate
            // the WAL file it just removed.
            let mut inner = entry.lock_unless_deleted().ok_or(AccessError::NotFound)?;
            ensure_hot(&self.0.data_dir, name, &mut inner, &self.0.metrics)
                .map_err(AccessError::Load)?;
            // Count the promotion NOW, before the WAL-cap and append-failure
            // early returns below. A Cold→Hot load just added this context's
            // footprint to resident memory; a refusal that returns without
            // counting it leaves the resident estimate short, so the budget
            // sweep never reclaims those bytes. recount_entry is absolute, so
            // the post-`operate` recount below just refreshes this.
            self.recount_entry(&mut inner);
            let first_seq = inner.wal_seq;
            let mut staged = None;
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
                let len_before = inner.wal_bytes;
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
                        // A failed append may still have leaked complete
                        // bytes (write landed, sync then failed, rollback
                        // failed too). Memory is untouched — `operate` never
                        // ran — so mark the entry dirty: the next flush
                        // stages this pre-write image at watermark
                        // `wal_seq - 1` and, since `wal_seq` did not move,
                        // truncates the log, carrying off the leaked tail
                        // before a replay can apply it. (`replay` de-dupes
                        // by seq as the second line of defense.)
                        entry.dirty.store(true, Ordering::Relaxed);
                        return Err(AccessError::Unpersisted(error.to_string()));
                    }
                }
                staged = Some((path, len_before));
            }
            let Slot::Hot(context) = &mut inner.slot else {
                unreachable!("ensure_hot leaves the slot hot");
            };
            let result = operate(context);
            entry.dirty.store(true, Ordering::Relaxed);
            self.recount_entry(&mut inner);

            if let Some((path, len_before)) = staged {
                let landed = applied(&result);
                if landed < ops.len() {
                    match wal::truncate_to(&path, len_before) {
                        Ok(()) if landed > 0 => {
                            match wal::append_batch(&path, first_seq, &ops[..landed]) {
                                Ok(appended) => {
                                    inner.wal_bytes = len_before + appended;
                                    inner.wal_seq = first_seq + landed as u64;
                                }
                                Err(error) => {
                                    // The applied prefix already happened
                                    // in memory and will not be undone —
                                    // and the truncate above already threw
                                    // its records off the disk, so the
                                    // caller now holds an acknowledgment
                                    // the log cannot replay. Flush the
                                    // image below (out of this lock) to
                                    // close that crash window now rather
                                    // than waiting out the flush interval.
                                    tracing::warn!(
                                        context = %name, %error,
                                        "WAL re-append after a partial apply failed; \
                                         flushing the image now to re-cover memory"
                                    );
                                    inner.wal_bytes = len_before;
                                    inner.wal_seq = first_seq;
                                    wal_behind = true;
                                }
                            }
                        }
                        Ok(()) => {
                            inner.wal_bytes = len_before;
                            inner.wal_seq = first_seq;
                        }
                        Err(error) => {
                            // The untried tail is still on disk looking
                            // exactly like applied records, and replay
                            // does not stop where the live apply did —
                            // left until the next flush tick, a crash
                            // would apply ops the caller was just told
                            // failed. Same medicine as the re-append
                            // failure above: flush the image now (out
                            // of this lock). Its watermark covers the
                            // whole batch's seqs, so the tail becomes
                            // replay-inert even if the log keeps
                            // refusing to shrink. Bookkeeping stays at
                            // the full-batch values: that is what the
                            // file still holds if the truncate never
                            // landed, and a successful flush resets
                            // both anyway.
                            tracing::warn!(
                                context = %name, %error,
                                "WAL truncate after a partial apply failed; \
                                 flushing the image now to retire the untried tail"
                            );
                            wal_behind = true;
                        }
                    }
                }
            }

            result
        };
        // The one state `logged_write` must never return in: ops the
        // caller is being told succeeded, present in memory only — the
        // trimmed log no longer holds them and a crash before the next
        // flush would silently lose them. An immediate image flush
        // restores the "acknowledged means replayable" contract; if it
        // fails too, the entry stays dirty, the flusher keeps retrying,
        // and /health reports the failing flush until one lands.
        if wal_behind {
            self.flush_entry(name, &entry);
        }
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
        let ops = self.clamp_out_of_range_paragraphs(name, ops);
        let wal_ops: Vec<WalOp> = ops.into_iter().map(WalOp::Associate).collect();
        self.logged_write(
            name,
            &wal_ops,
            |context| apply_in_order(context, &wal_ops),
            applied_count,
        )
    }

    /// Drops a paragraph locator that falls outside its source's
    /// stored passage, the same silent-drop posture `StoreOutcome`
    /// already applies to out-of-range questions and sections. This
    /// is the general-purpose backstop: callers that hand the batch's
    /// passage text to the ingest pipeline get a cheaper, unconditional
    /// clamp there, but a bare HTTP call or a later `add_associations`
    /// against an already-stored source has no such text in hand, so
    /// this checks the resident passage store instead.
    ///
    /// Best-effort like [`AppState::resolve_sections`]: an unknown
    /// context, a deleted entry, a source with no stored passage, or a
    /// store load failure all leave `paragraph` as given rather than
    /// fail the write — an unresolved locator is still meaningful
    /// (just without a section label), so this only removes locators
    /// it can positively prove are out of range.
    fn clamp_out_of_range_paragraphs(&self, name: &str, mut ops: Vec<AssocOp>) -> Vec<AssocOp> {
        if !ops.iter().any(|op| op.paragraph.is_some()) {
            return ops;
        }
        let Some(entry) = self.lookup(name) else {
            return ops;
        };
        let Some(_fence) = entry.read_unless_deleted() else {
            return ops;
        };
        let Ok(store) = self.entry_passages(&entry, &file_stem(name)) else {
            return ops;
        };
        for op in &mut ops {
            let Some(paragraph) = op.paragraph else {
                continue;
            };
            let Some(source) = op.source.as_deref() else {
                continue;
            };
            let Some(record) = store.get(source) else {
                continue;
            };
            if paragraph as usize >= record.paragraphs.len() {
                op.paragraph = None;
            }
        }
        ops
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
        self.logged_write(
            name,
            &wal_ops,
            |context| apply_in_order(context, &wal_ops),
            applied_count,
        )
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
        self.logged_write(
            name,
            &wal_ops,
            |context| apply_in_order(context, &wal_ops),
            applied_count,
        )
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
            // Cached vector stores, resident passages, the BM25 index,
            // and paragraph vectors count too — a cold entry can hold
            // plenty of each.
            let bytes = resident
                + entry.vectors_footprint()
                + entry.passages_footprint()
                + entry.bm25_footprint()
                + entry.passage_vectors_footprint();
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
    /// caller's sweep, its save failed (it stays resident rather than
    /// losing writes), or a concurrent eviction already cleared it. That
    /// last case matters: two budget sweeps snapshot the directory under
    /// a shared lock and can carry the same candidate, so the loser must
    /// report `false` or the caller subtracts its bytes from the
    /// residency estimate a second time.
    fn evict_entry(&self, name: &str, entry: &Entry) -> bool {
        let mut guard = entry.inner.write().unwrap();
        let inner = &mut *guard;
        // Re-check under the write lock; the entry may have changed
        // between the snapshot and now.
        if inner.meta.pinned {
            return false;
        }
        // Tracks whether THIS call actually released something — a graph,
        // a passage store, an index, or a vector cache. A call that finds
        // everything already gone (a rival sweep won the race) returns
        // `false` so its bytes are not double-counted.
        let mut freed = false;
        let watermark = inner.wal_seq - 1;
        if let Slot::Hot(context) = &mut inner.slot {
            // Persist before dropping if the image is stale (`dirty`) OR a
            // flush is mid-stage (`flushing`). A flush claims its entry by
            // clearing `dirty` under this same lock, then stages the image
            // with the lock dropped; in that window `dirty` reads clean
            // though the write is NOT yet on disk. Saving on `flushing`
            // too means we never drop a hot context whose only un-imaged
            // write lives in a flush that has not committed — the write
            // that, with the WAL off, exists nowhere else.
            if entry.dirty.load(Ordering::Relaxed) || entry.flushing.load(Ordering::Relaxed) {
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
            freed = true;
        }
        // Dropping the passage store loses nothing (its log is fsynced
        // per batch); a best-effort compaction first just spares the
        // next load a replay. Failure changes neither.
        {
            let mut passages = entry.passages.lock().unwrap();
            if let Some(store) = passages.take() {
                freed = true;
                if store.pending_log_bytes() > 0
                    && let Err(error) = store.compact()
                {
                    tracing::warn!("passages for '{name}' evicted uncompacted: {error}");
                }
            }
        }
        // Same best-effort posture for a dirty index: saving it spares
        // the next residency a re-tokenization, and the entry lock held
        // above keeps a racing delete away from the file.
        {
            let mut bm25 = entry.bm25.write().unwrap();
            if let Some(index) = bm25.take() {
                freed = true;
                if entry.bm25_dirty.swap(false, Ordering::Relaxed)
                    && let Err(error) = write_atomic(
                        &bm25_path(&self.0.data_dir, &file_stem(name)),
                        &index.to_bytes(),
                    )
                {
                    tracing::warn!("BM25 index for '{name}' evicted unpersisted: {error}");
                }
            }
        }
        if entry.passage_vectors.lock().unwrap().take().is_some() {
            freed = true;
        }
        if entry.vectors.lock().unwrap().take().is_some() {
            freed = true;
        }
        freed
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
    // Unfinished deletions first: a `.deleted` marker means delete()
    // acknowledged the removal but could not unlink the whole family —
    // without this sweep, a surviving `.ctx` would RESURRECT a context
    // the API already reported gone (and a surviving sidecar would
    // leak forever). Resuming the deletion here makes the marker the
    // durable half of the operation: acknowledged deletes stay deleted
    // across any crash or IO failure, eventually.
    for dir_entry in fs::read_dir(data_dir)? {
        let path = dir_entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("deleted")
            && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
        {
            tracing::warn!(stem, "resuming an unfinished context deletion");
            let stem = stem.to_string();
            for file in context_files(&stem) {
                let target = data_dir.join(file);
                if let Err(error) = fs::remove_file(&target)
                    && error.kind() != io::ErrorKind::NotFound
                {
                    tracing::warn!(path = %target.display(), %error, "unfinished deletion: file still held");
                }
            }
            // The marker goes last: it only leaves once the family did.
            if fs::remove_file(&path).is_err() {
                tracing::warn!(path = %path.display(), "unfinished deletion: marker still held");
            }
        }
    }
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
        let Some((stem, name)) = scanned_stem_and_name(&path) else {
            continue;
        };
        let stem = stem.as_str();
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

/// The scan-side decode shared by the context and group sweeps: a
/// discovered file's stem and the entity name it encodes, or `None`
/// (logged) when the name does not decode — one function, so the two
/// scans cannot drift on what "undecodable" means.
pub(crate) fn scanned_stem_and_name(path: &Path) -> Option<(String, String)> {
    let stem = path.file_stem().and_then(|s| s.to_str())?;
    match name_from_stem(stem) {
        Some(name) => Some((stem.to_string(), name)),
        None => {
            tracing::warn!("skipping {}: file name does not decode", path.display());
            None
        }
    }
}

/// The strict-membership gate shared by the group writes: the first
/// requested name the given namespace does not have, if any. Strict on
/// purpose — an add must never mint a dangling reference — and one
/// function for both namespaces (member contexts, child groups): the
/// caller supplies the existence test.
fn first_missing<'a>(
    names: impl IntoIterator<Item = &'a String>,
    exists: impl Fn(&str) -> bool,
) -> Option<&'a String> {
    names.into_iter().find(|name| !exists(name))
}

/// Shared body of the two membership sweeps — a deleted context out of
/// every group's members, a deleted group out of every parent's
/// children: removes `stale` from the chosen set field of every record
/// and persists each touched one. Best effort by design — a rewrite
/// that fails leaves memory correct and the file stale, which the next
/// boot's reconciliation heals. Lock-free on purpose: every caller
/// already holds the groups write lock.
fn sweep_membership(
    data_dir: &Path,
    groups: &mut BTreeMap<String, GroupRecord>,
    stale: &str,
    field: impl Fn(&mut GroupRecord) -> &mut BTreeSet<String>,
) {
    for (group_name, record) in groups.iter_mut() {
        if !field(record).remove(stale) {
            continue;
        }
        if let Err(error) = groups::write_group(data_dir, &file_stem(group_name), record) {
            tracing::warn!(
                group = %group_name,
                removed = %stale,
                %error,
                "group membership sweep not persisted; the next boot's reconciliation drops it"
            );
        }
    }
}

/// Boot-time counterpart of the delete-path sweeps: drops every group
/// member that is not a registered context, every child that is not a
/// scanned group, and every nesting edge that would close a cycle or
/// stack more than [`groups::MAX_GROUP_DEPTH`] groups (hand-edits
/// only — nothing running can persist such a shape). Each fix is
/// written back to disk immediately — disk is the source of truth, and
/// a fix that only lived in memory would leave the file lying to
/// `taguru inspect` and to file-level backups until the next unrelated
/// write. Runs unconditionally: the causes it heals (a crash between a
/// deletion and the sweep's rewrite, a sweep that could not persist, a
/// hand-edited data directory) leave no marker behind, and the whole
/// collection is small enough that checking it all costs nothing.
fn reconcile_groups(
    data_dir: &Path,
    registry: &HashMap<String, Arc<Entry>>,
    groups: &mut BTreeMap<String, GroupRecord>,
) {
    let scanned = groups.clone();
    for record in groups.values_mut() {
        record
            .contexts
            .retain(|context| registry.contains_key(context));
        record.groups.retain(|child| scanned.contains_key(child));
    }
    // The dangling references are gone; what remains can still be the
    // wrong SHAPE — the repair drops exactly the edges the validator
    // refuses, deterministically.
    groups::repair_nesting(groups);
    for (name, record) in groups.iter() {
        let before = &scanned[name];
        if before == record {
            continue;
        }
        match groups::write_group(data_dir, &file_stem(name), record) {
            Ok(()) => {
                tracing::info!(
                    group = %name,
                    dropped_contexts = before.contexts.len() - record.contexts.len(),
                    dropped_children = before.groups.len() - record.groups.len(),
                    "dropped dangling or ill-nested group reference(s) at boot"
                );
            }
            Err(error) => {
                tracing::warn!(
                    group = %name,
                    %error,
                    "boot reconciliation not persisted; memory is correct, the file heals on the next successful group write"
                );
            }
        }
    }
}

/// Restores `inner.meta` to `previous` after a load or persist failure
/// partway through `update_meta`. Also un-applies the floor from any
/// already-loaded context, matching the one place `update_meta` pushes
/// a field straight into the hot context instead of just the sidecar.
fn rollback_meta(inner: &mut EntryInner, previous: ContextMeta) {
    if let Slot::Hot(context) = &mut inner.slot {
        context.set_dice_floor(previous.dice_floor);
    }
    inner.meta = previous;
}

/// How long a failed load's refusal is answered from memory before the
/// disk is tried again. Long enough that a client retry storm against
/// one broken context cannot grind the disk; short enough that
/// restoring the files heals the context without a restart.
const LOAD_FAILURE_RETRY: std::time::Duration = std::time::Duration::from_secs(30);

/// Loads the image behind a cold slot and replays whatever the WAL
/// holds above the image's watermark; hot slots pass through. On
/// success the slot is hot, the stats are fresh, and `wal_seq`
/// continues from the replay's tail. Every call lands in the cache
/// metrics: hot is a hit, a Cold→Hot attempt is a load — but a
/// quarantined refusal (below) is neither: no disk was touched.
///
/// A failed load is remembered: for the next [`LOAD_FAILURE_RETRY`]
/// this answers the same refusal without re-reading anything, so a
/// permanently corrupt context costs one read per interval instead of
/// one per request. The heal paths stay what they were — restore the
/// files and the next retry loads, or DELETE the context.
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
    if let Some((failed_at, refusal)) = &inner.load_failure
        && failed_at.elapsed() < LOAD_FAILURE_RETRY
    {
        return Err(format!(
            "{refusal} (quarantined after the failed load; the disk is retried at \
             most every {}s — restore the file family or DELETE the context)",
            LOAD_FAILURE_RETRY.as_secs()
        ));
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
            inner.load_failure = Some((std::time::Instant::now(), error.clone()));
            return Err(error);
        }
    };
    metrics.record_cache_load(true);
    inner.load_failure = None;
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

/// How many ops an `apply_in_order` result actually landed — the full
/// count on success, the prefix on a partial write. Feeds
/// `logged_write`'s WAL trim: it never inspects `T` itself, only how
/// far the batch got.
fn applied_count(result: &Result<usize, PartialWrite>) -> usize {
    match result {
        Ok(applied) => *applied,
        Err(partial) => partial.applied,
    }
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
                    op.paragraph,
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

pub(crate) fn passages_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.passages.bin"))
}

pub(crate) fn passages_wal_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.passages.wal.jsonl"))
}

pub(crate) fn pvectors_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.pvectors.bin"))
}

pub(crate) fn bm25_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.bm25.bin"))
}

pub(crate) fn vectors_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.vectors.bin"))
}

pub(crate) fn wal_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.wal.jsonl"))
}

/// The durable-deletion marker: while it exists, boot resumes the
/// unlinks (see `delete`/`scan_data_dir`). One builder so the writer,
/// the boot sweep, and the create-time cleanup can never disagree
/// about its name — a stale marker beside a freshly recreated context
/// would otherwise make the next boot delete the new context.
pub(crate) fn deleted_marker_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.deleted"))
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x1_0000_01b3;

/// The terms of one passage or query: [`text_terms`] over the
/// normalized text, plus a word term per piece of every camelCase run.
/// The split reads an NFKC-folded but NOT lowercased view of the input:
/// lowercasing would erase the very case boundaries that let `state`
/// reach `AppState`, while the width fold keeps a full-width `Ａ` — which
/// the normalized whole-word term already folds to ASCII — in the same
/// run as its ASCII neighbors instead of breaking it (so `ＡpplePie`
/// yields the `apple` piece, matching a plain `apple` cue). One function
/// serves both sides of the search, so they cannot disagree about what a
/// term is.
pub(crate) fn passage_terms(raw: &str) -> Vec<u64> {
    use unicode_normalization::UnicodeNormalization;
    let mut terms = text_terms(&taguru::context::normalize_entry(raw));
    let mut run: Vec<char> = Vec::new();
    for ch in raw.nfkc() {
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

fn save_files(
    dir: &Path,
    name: &str,
    meta: &ContextMeta,
    stats: &ContextStats,
    usage: &ContextUsage,
    context: &Context,
) -> io::Result<()> {
    let stem = file_stem(name);
    // The image is what `scan_data_dir` keys a context's existence on, so
    // it lands LAST: each `write_atomic` fully commits (fsync + rename +
    // parent-dir fsync) before returning, so by the time the `.ctx` is
    // durably in the directory its `.meta.json` companion already is too.
    // A crash between the two therefore leaves at worst an orphan sidecar
    // with no image — invisible to the scan and overwritten by the next
    // same-name create — never a durable image with a defaulted sidecar,
    // which would resurrect a context `create` told the client had failed.
    // (Image-then-meta would do exactly that; see `create`'s doc.)
    write_meta(dir, &stem, meta, stats, usage)?;
    write_atomic(&image_path(dir, &stem), &context.to_bytes())
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
    write_atomic_with(path, bytes, false)
}

/// [`write_atomic`] for secret-bearing files (the OAuth grant store):
/// the staged file drops to owner-only permissions BEFORE any content
/// lands in it, and the rename carries the mode to the final name —
/// no moment exists where another local account could read the bytes.
/// Non-Unix platforms have no mode bits and get the plain behavior.
pub(crate) fn write_atomic_private(path: &Path, bytes: &[u8]) -> io::Result<()> {
    write_atomic_with(path, bytes, true)
}

fn write_atomic_with(path: &Path, bytes: &[u8], private: bool) -> io::Result<()> {
    let staged = stage_bytes(path, bytes, private)?;
    let result = commit_staged(&staged, path);
    if result.is_err() {
        // A failed rename leaves `staged` sitting under its temporary
        // name — clean it up rather than leave it for unbounded disk
        // litter until the next boot's sweep. (A failed parent fsync
        // after a successful rename finds nothing here: the file is
        // already `path`, and removing the stale `staged` name is a
        // harmless no-op.)
        let _ = fs::remove_file(&staged);
    }
    result
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
fn stage_bytes(path: &Path, bytes: &[u8], private: bool) -> io::Result<PathBuf> {
    use std::io::Write;

    #[cfg(not(unix))]
    let _ = private;
    let staged = staging_path(path);
    // A private file must be BORN owner-only: create-then-chmod leaves a
    // window (the default-umask create, ~0644, before the chmod) in
    // which another local account can open() the staging file and keep
    // reading it — the secret bytes land in that fd afterwards. `mode`
    // on the open() sets the creation mode atomically, so no readable
    // moment ever exists. `create_new` also refuses to reuse a file an
    // attacker pre-created, closing the mirror-image swap. The staging
    // name is per-process unique (`staging_path`), so create_new never
    // collides with our own concurrent stagers.
    let open = |staged: &Path| -> io::Result<fs::File> {
        #[cfg(unix)]
        if private {
            use std::os::unix::fs::OpenOptionsExt;
            return fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(staged);
        }
        fs::File::create(staged)
    };
    let write = open(&staged).and_then(|mut file| {
        file.write_all(bytes)?;
        file.sync_all()
    });
    match write {
        Ok(()) => Ok(staged),
        Err(error) => {
            // The file (if it even got created) never held valid
            // content and was never handed to a caller — remove it
            // rather than leave a partial write behind under its
            // temporary name.
            let _ = fs::remove_file(&staged);
            Err(error)
        }
    }
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

/// Runs blocking work — a cold load's disk read plus full-image
/// validation — off the async runtime when called from one:
/// `block_in_place` tells the multi-thread runtime this worker will
/// stall, so queued tasks migrate to other workers instead of waiting
/// behind synchronous IO. The CLI entrances (import, export) and plain
/// `#[test]`s run with no runtime, and a current_thread test runtime
/// cannot block-in-place — both fall through to running the work
/// inline. Nested calls are safe: tokio treats an inner
/// `block_in_place` on an already-blocking thread as a no-op (the
/// api layer wraps writes and passage search in one already).
fn offload<T>(work: impl FnOnce() -> T) -> T {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(work)
        }
        _ => work(),
    }
}

/// One context's whole file family, by stem — the delete loop and the
/// boot-time deletion sweep must never disagree about what "the whole
/// family" means, so both read this one list.
fn context_files(stem: &str) -> [String; 9] {
    [
        format!("{stem}.ctx"),
        format!("{stem}.meta.json"),
        format!("{stem}.sources.json"),
        format!("{stem}.passages.bin"),
        format!("{stem}.passages.wal.jsonl"),
        format!("{stem}.pvectors.bin"),
        format!("{stem}.bm25.bin"),
        format!("{stem}.vectors.bin"),
        format!("{stem}.wal.jsonl"),
    ]
}

/// Encodes a context name as a file stem: bytes outside [A-Za-z0-9_-]
/// become %XX. Context names arrive from URL paths and may contain path
/// separators or dots; encoding them keeps every name inside the data
/// directory (no traversal) and reversible.
pub(crate) fn file_stem(name: &str) -> String {
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
    fn meta_sidecar_with_tuple_top_concepts_loads_the_pre_36_shape() {
        // A sidecar written before #36 switched top_concepts from
        // [name, count] tuples to {label, count} objects.
        let json = br#"{"description":"d","pinned":false,"stats":{"associations":3,
            "top_concepts":[["sake brewery",6],["brewing",5]]}}"#;
        let file: MetaFile = serde_json::from_slice(json).unwrap();
        assert_eq!(file.meta.description, "d");
        assert_eq!(
            file.stats.top_concepts,
            vec![
                LabelUsage {
                    label: "sake brewery".to_string(),
                    count: 6
                },
                LabelUsage {
                    label: "brewing".to_string(),
                    count: 5
                },
            ]
        );
    }

    #[test]
    fn meta_sidecar_with_object_top_concepts_loads_the_current_shape() {
        let json = br#"{"description":"d","pinned":false,"stats":{"associations":3,
            "top_concepts":[{"label":"sake brewery","count":6}]}}"#;
        let file: MetaFile = serde_json::from_slice(json).unwrap();
        assert_eq!(
            file.stats.top_concepts,
            vec![LabelUsage {
                label: "sake brewery".to_string(),
                count: 6
            }]
        );
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

    /// An empty context name is refused at the registry boundary — the
    /// last guard against a bare `.ctx` file that `scan_data_dir` (which
    /// keys on the file stem) would never rediscover, silently orphaning
    /// every write to it. Parse and API refuse it earlier; this locks
    /// the floor beneath them.
    #[test]
    fn an_empty_context_name_is_refused_by_create() {
        let dir = scratch_dir("empty-name");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(matches!(
            state.create("", ContextMeta::default()),
            Err(CreateError::InvalidName)
        ));
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

    /// A `.deleted` marker is the durable half of a delete: boot
    /// resumes the unlinks it finds one for, so an acknowledged
    /// deletion can never resurrect — however the unlink loop failed.
    #[test]
    fn an_unfinished_deletion_is_resumed_at_boot() {
        let dir = scratch_dir("deleted-sweep");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("a.md"))],
                )
                .unwrap()
                .unwrap();
            state.flush_dirty();
        }
        assert!(dir.join("sake.ctx").exists());
        // The crash-shaped state: delete() wrote its marker, then the
        // process died before (or while) the unlinks ran.
        fs::write(dir.join("sake.deleted"), b"").unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(
            state.directory_entry("sake").is_none(),
            "an acknowledged deletion must not resurrect"
        );
        assert!(!dir.join("sake.ctx").exists(), "the family must be gone");
        assert!(!dir.join("sake.wal.jsonl").exists());
        assert!(
            !dir.join("sake.deleted").exists(),
            "the marker leaves once the family did"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// The dangerous interleaving: a delete leaves a marker behind
    /// (partial failure), the SAME running server recreates the
    /// context, and a later restart must NOT let the stale marker
    /// destroy the freshly created files. create() clears the marker.
    #[test]
    fn recreating_a_context_clears_a_stale_deletion_marker() {
        let dir = scratch_dir("deleted-recreate");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state.delete("sake");
            // Simulate the failure mode delete() cannot fully guard: its
            // unlink loop errored before removing the marker, so the
            // marker survives on disk while the name is free again.
            fs::write(deleted_marker_path(&dir, "sake"), b"").unwrap();
            // The same server recreates the context; create() must clear
            // that stale marker so the next boot does not resume it.
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "recreate")
                .unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("a.md"))],
                )
                .unwrap()
                .unwrap();
            state.flush_dirty();
        }
        assert!(
            !dir.join("sake.deleted").exists(),
            "recreate must clear the stale marker"
        );
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(
            state.directory_entry("sake").is_some(),
            "the recreated context must survive the restart"
        );
        let count = state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(count, 1, "its data must be intact");
        let _ = fs::remove_dir_all(&dir);
    }

    /// Compaction sheds the dead weight retraction leaves behind,
    /// preserves every live fact, keeps the WAL watermark monotonic —
    /// a write after the compact replays correctly across a hard crash
    /// — and the shrunken image is what a restart boots.
    #[test]
    fn compaction_shrinks_the_image_and_stays_crash_safe() {
        let dir = scratch_dir("compact");
        let live_facts = |state: &AppState| -> Vec<(String, String, String, u64)> {
            let mut facts = state
                .read_context("sake", |context| {
                    context
                        .query_any(&[], &[], &[])
                        .into_iter()
                        .filter(|association| association.count > 0)
                        .map(|association| {
                            (
                                association.subject,
                                association.label,
                                association.object,
                                association.count,
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .map_err(|_| "read")
                .unwrap();
            facts.sort();
            facts
        };
        let before;
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
                        assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md")),
                        assoc_op("蔵", "銘柄", "青嶺", 1.0, Some("keep.md")),
                        assoc_op("蔵", "廃止銘柄", "旧銘", 1.0, Some("gone.md")),
                    ],
                )
                .unwrap()
                .unwrap();
            state.retract_source("sake", "gone.md").unwrap();
            before = live_facts(&state);

            let outcome = state.compact_context("sake").unwrap();
            assert!(
                outcome.bytes_after < outcome.bytes_before,
                "{outcome:?} must shrink"
            );
            assert_eq!(outcome.dead_edges, 1, "{outcome:?}");
            assert_eq!(live_facts(&state), before, "live content must survive");

            // A write AFTER the compact must replay across a crash —
            // the fresh image carries the old watermark, so the WAL
            // sequence keeps counting from where it was.
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("蔵", "創業年", "1907年", 1.0, Some("keep.md"))],
                )
                .unwrap()
                .unwrap();
            // Drop WITHOUT flushing: the crash.
        }
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let after = live_facts(&state);
        assert_eq!(after.len(), before.len() + 1, "{after:?}");
        assert!(
            after
                .iter()
                .any(|(_, label, object, _)| label == "創業年" && object == "1907年"),
            "the post-compact write must replay: {after:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A context whose load failed answers the remembered refusal
    /// without touching the disk until the retry window elapses — a
    /// permanently corrupt context must not cost a read + full parse
    /// per request under client retries — and heals by itself on the
    /// first retry after the files are restored.
    #[test]
    fn a_failed_load_is_quarantined_until_the_retry_window_elapses() {
        let dir = scratch_dir("quarantine");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("a.md"))],
                )
                .unwrap()
                .unwrap();
            state.flush_dirty();
        }
        let image = dir.join("sake.ctx");
        let healthy = fs::read(&image).unwrap();
        let mut corrupt = healthy.clone();
        corrupt[8] = 0xFF; // the version field — refused by from_bytes
        fs::write(&image, &corrupt).unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let AccessError::Load(first) = state.read_context("sake", |_| ()).unwrap_err() else {
            panic!("a corrupt image must refuse the read");
        };
        assert!(first.contains("corrupt"), "{first}");

        // Repair the file. Within the window the CACHED refusal still
        // answers — proof the disk is not consulted per request.
        fs::write(&image, &healthy).unwrap();
        let AccessError::Load(second) = state.read_context("sake", |_| ()).unwrap_err() else {
            panic!("the quarantine must still refuse");
        };
        assert!(second.contains("quarantined"), "{second}");
        assert!(
            rendered(&state).contains("taguru_cache_loads_total{outcome=\"failed\"} 1"),
            "a quarantined refusal is not a second load attempt"
        );

        // Past the window, the retry sees the repaired image and heals.
        state.age_load_failures("sake", LOAD_FAILURE_RETRY);
        let count = state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(count, 1);
        let _ = fs::remove_dir_all(&dir);
    }

    /// The passage store gets the same quarantine as the graph image:
    /// a broken snapshot or log answers its remembered refusal instead
    /// of being re-read on every passage request.
    #[test]
    fn a_failed_passage_load_is_quarantined_like_the_image() {
        let dir = scratch_dir("passage-quarantine");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .store_passages(
                    "sake",
                    BTreeMap::from([(
                        "a.md".to_string(),
                        crate::passages::PassageSubmission::plain("本文。"),
                    )]),
                )
                .unwrap()
                .unwrap();
            state.flush_dirty();
        }
        let log = dir.join("sake.passages.wal.jsonl");
        let healthy = fs::read(&log).unwrap();
        let mut corrupt = healthy.clone();
        corrupt.splice(0..0, *b"not json\n"); // a corrupt INTERIOR line
        fs::write(&log, &corrupt).unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let sources = ["a.md".to_string()];
        let first = state
            .lookup_passages("sake", &sources)
            .expect("registered")
            .unwrap_err();
        assert!(!first.to_string().contains("quarantined"), "{first}");

        fs::write(&log, &healthy).unwrap();
        let second = state
            .lookup_passages("sake", &sources)
            .expect("registered")
            .unwrap_err();
        assert!(second.to_string().contains("quarantined"), "{second}");

        state.age_load_failures("sake", LOAD_FAILURE_RETRY);
        let (passages, missing) = state
            .lookup_passages("sake", &sources)
            .expect("registered")
            .unwrap();
        assert!(missing.is_empty());
        assert_eq!(passages["a.md"], "本文。");
        let _ = fs::remove_dir_all(&dir);
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
            paragraph: None,
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
    fn a_failed_reappend_after_a_partial_apply_still_persists_acknowledged_ops() {
        // A partial apply trims the WAL back and re-appends just the
        // applied prefix. The trim durably removed the whole batch, so
        // if that re-append then fails, the ops the caller is being
        // told succeeded exist in memory only — logged_write must close
        // that crash window itself (an immediate image flush), not
        // leave it open until the next flush tick.
        let dir = scratch_dir("wal-reappend-fault");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            // Two concepts for the aliases to target, and an alias the
            // batch's second op will re-point — the conflict that makes
            // the apply partial.
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("青嶺酒造", "所在地", "京都酒造", 1.0, None)],
                )
                .unwrap()
                .unwrap();
            state
                .add_aliases(
                    "sake",
                    &BTreeMap::from([("kyo".to_string(), "京都酒造".to_string())]),
                    &BTreeMap::new(),
                )
                .unwrap()
                .unwrap();

            // BTreeMap order: "aomine" applies first, "kyo" then
            // conflicts (re-pointing an existing alias). The batch
            // append itself succeeds; the fault fires on the re-append
            // of the applied prefix, right after the trim.
            wal::fail_appends_after(1);
            let partial = state
                .add_aliases(
                    "sake",
                    &BTreeMap::from([
                        ("aomine".to_string(), "青嶺酒造".to_string()),
                        ("kyo".to_string(), "青嶺酒造".to_string()),
                    ]),
                    &BTreeMap::new(),
                )
                .unwrap()
                .unwrap_err();
            assert_eq!(
                partial.applied, 1,
                "the first alias landed before the conflict"
            );
            // NO flush_dirty: dropping the state here is the crash.
        }

        let reborn = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let aliases = reborn
            .read_context("sake", |context| {
                context
                    .concept_aliases()
                    .into_iter()
                    .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
                    .collect::<Vec<_>>()
            })
            .map_err(|_| "read")
            .unwrap();
        assert!(
            aliases
                .iter()
                .any(|(alias, canonical)| alias == "aomine" && canonical == "青嶺酒造"),
            "the acknowledged alias must survive the crash: {aliases:?}"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_failed_wal_trim_after_a_partial_apply_keeps_the_untried_tail_from_replaying() {
        // A partial apply trims the untried tail back out of the WAL.
        // If that trim itself fails, the tail sits on disk looking
        // exactly like applied records — and replay does not stop at
        // the rejection the live apply stopped at, so a crash would
        // apply ops the caller was just told failed. logged_write must
        // close that window itself: an immediate image flush whose
        // watermark retires the whole batch's seqs.
        let dir = scratch_dir("wal-trim-fault");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("青嶺酒造", "所在地", "京都酒造", 1.0, None)],
                )
                .unwrap()
                .unwrap();
            state
                .add_aliases(
                    "sake",
                    &BTreeMap::from([("kyo".to_string(), "京都酒造".to_string())]),
                    &BTreeMap::new(),
                )
                .unwrap()
                .unwrap();

            // BTreeMap order: "aomine" applies, "kyo" conflicts
            // (re-pointing an existing alias) and stops the batch,
            // "mine" is never tried. The batch append succeeds; the
            // fault fires on the trim that should carry "kyo" and
            // "mine" back off the disk.
            wal::fail_truncates_after(0);
            let partial = state
                .add_aliases(
                    "sake",
                    &BTreeMap::from([
                        ("aomine".to_string(), "青嶺酒造".to_string()),
                        ("kyo".to_string(), "青嶺酒造".to_string()),
                        ("mine".to_string(), "青嶺酒造".to_string()),
                    ]),
                    &BTreeMap::new(),
                )
                .unwrap()
                .unwrap_err();
            assert_eq!(
                partial.applied, 1,
                "only the first alias landed before the conflict"
            );
            // NO flush_dirty: dropping the state here is the crash.
        }

        let reborn = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let aliases = reborn
            .read_context("sake", |context| {
                context
                    .concept_aliases()
                    .into_iter()
                    .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
                    .collect::<Vec<_>>()
            })
            .map_err(|_| "read")
            .unwrap();
        assert!(
            aliases
                .iter()
                .any(|(alias, canonical)| alias == "aomine" && canonical == "青嶺酒造"),
            "the acknowledged alias must survive the crash: {aliases:?}"
        );
        assert!(
            aliases
                .iter()
                .any(|(alias, canonical)| alias == "kyo" && canonical == "京都酒造"),
            "the conflicting alias must keep its original target: {aliases:?}"
        );
        assert!(
            !aliases.iter().any(|(alias, _)| alias == "mine"),
            "an op the caller was told failed must not replay into existence: {aliases:?}"
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
        wal::append_batch(
            &passages_wal_path(&dir, &stem),
            1,
            &[crate::passages::PassageOp::Store {
                source: "ghost".to_string(),
                text: "前世代の本文".to_string(),
                questions: Vec::new(),
                sections: Vec::new(),
            }],
        )
        .unwrap();

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
            !passages_wal_path(&dir, &stem).exists(),
            "the old generation's passage log survived the create"
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
        assert!(
            state.passage_sources("sake").unwrap().unwrap().is_empty(),
            "the old generation's passage log replayed into the new context"
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
        let state = AppState::boot_with(
            capped_dir.clone(),
            usize::MAX,
            None,
            BootOptions {
                wal_max_bytes: 1,
                ..BootOptions::default()
            },
        )
        .unwrap();
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
    #[cfg(unix)]
    fn a_failed_persist_does_not_leave_the_failed_change_in_memory() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("meta-rollback");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        // A clean update lands on disk.
        let meta = state
            .update_meta("sake", Some("A".to_string()), None, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(meta.description, "A");

        // The disk goes bad: this update must be refused...
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
        let failed = state
            .update_meta("sake", Some("B".to_string()), None, None, None)
            .unwrap();
        assert!(failed.is_err(), "a persist failure must surface as Err");
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();

        // ...and must not have left "B" sitting in memory — a later,
        // unrelated successful update must still see and persist "A",
        // not silently resurrect the failed change.
        let meta = state
            .update_meta("sake", None, Some(true), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(
            meta.description, "A",
            "the failed update to \"B\" must not have survived in memory"
        );

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
        let (weight, count) = state
            .read_context("sake", |context| {
                let assoc = &context.query(Some("青嶺酒造"), Some("代表銘柄"), Some("青嶺"))[0];
                (assoc.weight, assoc.count)
            })
            .map_err(|_| "read")
            .unwrap();
        // A wrongly replayed record would double both sum and count,
        // leaving their ratio — weight — unchanged at 1.0; count is what
        // actually catches the double-apply the watermark exists to
        // prevent.
        assert_eq!(weight, 1.0);
        assert_eq!(count, 1);

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

    /// `apply_in_order` stops at the first rejection, but durability
    /// appends the whole batch to the WAL before `operate` even runs.
    /// Without trimming, the untried tail sits on disk indistinguishable
    /// from an applied record, and `ensure_hot`'s replay (`replay_op`,
    /// which continues past a rejection instead of stopping) would try
    /// it independently on the next cold load — applying an op the
    /// client was told never ran.
    #[test]
    fn replay_does_not_apply_ops_a_partial_batch_never_tried() {
        let dir = scratch_dir("wal-partial-tail");
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
            state
                .add_aliases(
                    "sake",
                    &BTreeMap::from([
                        ("Aomine".to_string(), "青嶺酒造".to_string()),
                        ("AomineShuzo".to_string(), "青嶺酒造".to_string()),
                    ]),
                    &BTreeMap::new(),
                )
                .unwrap()
                .unwrap();

            // A 3-op batch whose middle op is a guaranteed, deterministic
            // rejection (withdrawing an alias that was never registered):
            // `apply_in_order` applies "Aomine", stops at "NONEXISTENT",
            // and never even attempts "AomineShuzo".
            let outcome = state
                .remove_aliases(
                    "sake",
                    &[
                        "Aomine".to_string(),
                        "NONEXISTENT".to_string(),
                        "AomineShuzo".to_string(),
                    ],
                    &[],
                )
                .unwrap();
            let partial = outcome.unwrap_err();
            assert_eq!(partial.applied, 1, "only the leading op ran");
            assert!(
                !partial.full,
                "an absent alias is a conflict, not a capacity error"
            );
            // No flush — the WAL, not the image, must carry this state
            // into the restart below.
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let untried = state
            .read_context("sake", |context| {
                context.query(Some("AomineShuzo"), Some("代表銘柄"), None)
            })
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(
            untried.len(),
            1,
            "the untried alias must survive: the live batch never touched it, so replay must not either"
        );
        let removed = state
            .read_context("sake", |context| {
                context.query(Some("Aomine"), Some("代表銘柄"), None)
            })
            .map_err(|_| "read")
            .unwrap();
        assert!(
            removed.is_empty(),
            "the removal that did apply live must still hold after replay"
        );

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
                BootOptions {
                    wal_enabled: false,
                    ..BootOptions::default()
                },
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

    /// With the WAL off, an entry's image is its ONLY durable home. A
    /// flush must therefore not clear `dirty` before it has published that
    /// image: if it did, an eviction racing the flush would read "clean",
    /// drop the hot context WITHOUT saving, and lose the acknowledged
    /// write outright. Two contexts thrash a one-byte budget — each write
    /// evicts the other while it is still dirty — as a flusher runs flat
    /// out, so staging (unlocked) and eviction interleave every which way.
    /// Every acknowledged write must still be readable, with no restart
    /// and no log to fall back on.
    #[test]
    fn a_flush_racing_an_eviction_loses_no_write_with_the_wal_off() {
        use std::thread;

        let dir = scratch_dir("flush-evict-race");
        let state = AppState::boot_with(
            dir.clone(),
            1, // one byte: only a single context stays resident at a time
            None,
            BootOptions {
                wal_enabled: false,
                ..BootOptions::default()
            },
        )
        .unwrap();
        for name in ["sake", "wine"] {
            state
                .create(name, ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
        }

        const WRITES: usize = 200;
        let writers: Vec<_> = ["sake", "wine"]
            .into_iter()
            .map(|name| {
                let state = state.clone();
                thread::spawn(move || {
                    for index in 0..WRITES {
                        state
                            .add_associations(
                                name,
                                vec![assoc_op(&format!("s{index}"), "l", "o", 1.0, None)],
                            )
                            .unwrap()
                            .unwrap();
                    }
                })
            })
            .collect();
        while writers.iter().any(|writer| !writer.is_finished()) {
            state.flush_dirty();
        }
        for writer in writers {
            writer.join().unwrap();
        }

        // Whatever survives lives in the images the flushes and evictions
        // wrote — there is no WAL to replay. Every write must be there.
        for name in ["sake", "wine"] {
            let count = state
                .read_context(name, |context| context.association_count())
                .map_err(|_| "read")
                .unwrap();
            assert_eq!(count, WRITES, "context '{name}' lost writes to the race");
        }

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_compaction_racing_a_periodic_flush_never_loses_the_image() {
        use std::thread;

        // `compact_context` swaps `slot` for a freshly rebuilt `Context`
        // while a `flush_dirty` tick may already be mid-stage for the
        // same entry (bytes read from the OLD `Context`, disk write in
        // flight, entry unlocked). `image_generation` is what lets that
        // flush's republish recognize the swap and back off instead of
        // overwriting the fresh image with the one it just replaced.
        // Hammering writes, compactions, and flush ticks on one context
        // concurrently gives that window many chances to open; without
        // the generation check this reliably drops associations from
        // the persisted image (a stale flush's snapshot predates them).
        let dir = scratch_dir("compact-flush-race");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        const ROUNDS: usize = 200;
        let writer = {
            let state = state.clone();
            thread::spawn(move || {
                for index in 0..ROUNDS {
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
        let compactor = {
            let state = state.clone();
            thread::spawn(move || {
                for _ in 0..ROUNDS {
                    let _ = state.compact_context("sake");
                }
            })
        };
        while !writer.is_finished() || !compactor.is_finished() {
            state.flush_dirty();
        }
        writer.join().unwrap();
        compactor.join().unwrap();

        let expected = state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(expected, ROUNDS, "the race lost associations in memory");
        // One uncontested flush so the last round's write or compaction
        // is guaranteed durable before the reboot below — the polling
        // loop above only guarantees eventual, not final, convergence.
        state.flush_dirty();
        drop(state);

        let reborn = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let recovered = reborn
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(
            recovered, expected,
            "the race lost associations from the image"
        );

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

    /// The private variant owns its bytes from the first write: the
    /// mode is set on the staged file before content lands, and the
    /// rename carries it to the final name.
    #[cfg(unix)]
    #[test]
    fn write_atomic_private_lands_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("atomic-private");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("secrets.json");
        write_atomic_private(&path, b"{}").unwrap();
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "mode {mode:o}");
        // Re-persisting keeps the tightened mode.
        write_atomic_private(&path, b"{\"v\":2}").unwrap();
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "mode {mode:o}");

        let _ = fs::remove_dir_all(dir);
    }

    /// The STAGING file — not just the committed one — is owner-only
    /// the instant it exists: `stage_bytes` must leave no readable
    /// window between create and the secret write (the TOCTOU the
    /// create_new+mode fix closes). Inspect the temp file's mode before
    /// it is committed.
    #[cfg(unix)]
    #[test]
    fn private_staging_file_is_born_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("atomic-private-staging");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("secrets.json");
        let staged = stage_bytes(&path, b"secret", true).unwrap();
        let mode = fs::metadata(&staged).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "staging file mode {mode:o} — must be 0600 at birth, not chmod'd after"
        );
        // The non-private path stays world-default (no regression).
        let staged_plain = stage_bytes(&dir.join("plain.bin"), b"x", false).unwrap();
        assert!(fs::metadata(&staged_plain).is_ok());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn write_atomic_cleans_up_its_staging_file_when_the_commit_fails() {
        let dir = scratch_dir("atomic-commit-fail");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("file.bin");
        // Renaming a plain file onto a non-empty directory fails on
        // every platform this project targets — staging (a distinctly
        // named file) still succeeds, so this isolates a commit-phase
        // failure without touching stage_bytes.
        fs::create_dir(&path).unwrap();
        fs::write(path.join("occupied"), b"x").unwrap();

        let result = write_atomic(&path, b"payload");
        assert!(
            result.is_err(),
            "renaming a file onto a non-empty directory must fail"
        );

        let leftovers: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.starts_with("tmp"))
            })
            .map(|entry| entry.path())
            .collect();
        assert!(
            leftovers.is_empty(),
            "a failed commit must not leave its staging file behind: {leftovers:?}"
        );

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

        // A full-width letter folds to ASCII and stays IN the run, so its
        // camelCase pieces match the plain-ASCII words. Before the NFKC
        // fold the full-width 'Ａ' broke the run, the piece read "pple",
        // and a plain "apple" cue never matched.
        let terms = passage_terms("ＡpplePie");
        assert!(terms.contains(&word("apple")));
        assert!(terms.contains(&word("pie")));

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
            assert_eq!(
                state
                    .store_passages("sake", plain(passages))
                    .unwrap()
                    .unwrap()
                    .stored,
                1
            );
        }

        // A fresh boot serves the registered passage; unknown sources
        // come back as missing rather than erroring.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let (passages, missing) = state
            .lookup_passages("sake", &["第1段落".to_string(), "第9段落".to_string()])
            .unwrap()
            .unwrap();
        assert!(passages["第1段落"].starts_with("青嶺酒造は"));
        assert_eq!(missing, vec!["第9段落".to_string()]);
        assert_eq!(
            state.passage_sources("sake").unwrap().unwrap(),
            vec!["第1段落"]
        );
        assert!(state.lookup_passages("nope", &[]).is_none());

        // Deleting the context removes the whole passage file family:
        // the log the store just wrote, any snapshot, and a legacy
        // sources file left over from before the migration.
        fs::write(
            sources_path(&dir, &file_stem("sake")),
            br#"{"legacy":"remnant"}"#,
        )
        .unwrap();
        state.delete("sake").unwrap().unwrap();
        assert!(!sources_path(&dir, &file_stem("sake")).exists());
        assert!(!passages_path(&dir, &file_stem("sake")).exists());
        assert!(!passages_wal_path(&dir, &file_stem("sake")).exists());

        let _ = fs::remove_dir_all(dir);
    }

    /// A failed create must release its `pending_creates` reservation —
    /// otherwise one disk refusal would leave the name reading as taken
    /// until restart.
    #[test]
    fn a_failed_create_releases_the_name() {
        let dir = scratch_dir("create-release");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

        // A directory where create expects at most a stale FILE:
        // remove_file refuses it with something other than NotFound,
        // failing the clear-the-slate pass after the name is reserved.
        let obstruction = wal_path(&dir, &file_stem("sake"));
        fs::create_dir_all(&obstruction).unwrap();
        assert!(matches!(
            state.create("sake", ContextMeta::default()),
            Err(CreateError::Io(_))
        ));

        // Obstruction gone, the same name must create cleanly — the
        // failed attempt's reservation may not linger.
        fs::remove_dir_all(&obstruction).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_create_racing_a_slow_delete_is_refused_not_interleaved() {
        let dir = scratch_dir("delete-create-race");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        // Stall the delete mid-flight: it unregisters the name, then
        // must wait for this read guard before it may touch files —
        // exactly the window where a create used to interleave and
        // have its new generation unlinked from under it.
        let entry = state.lookup("sake").unwrap();
        let stall = entry.inner.read().unwrap();
        let deleter = {
            let state = state.clone();
            std::thread::spawn(move || state.delete("sake").unwrap().unwrap())
        };
        while state.lookup("sake").is_some() {
            std::thread::yield_now();
        }
        assert!(
            matches!(
                state.create("sake", ContextMeta::default()),
                Err(CreateError::AlreadyExists)
            ),
            "a mid-delete name must read as taken"
        );

        drop(stall);
        deleter.join().unwrap();
        // The delete has fully finished: the name is free again and the
        // recreate starts from a clean slate.
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "recreate")
            .unwrap();
        assert!(image_path(&dir, &file_stem("sake")).exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_passage_write_racing_a_delete_backs_off_at_the_tombstone() {
        let dir = scratch_dir("passage-delete-race");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("第1段落".to_string(), "本文。".to_string());
        state
            .store_passages("sake", plain(passages.clone()))
            .unwrap()
            .unwrap();

        // The racing writer's handle predates the delete — exactly the
        // window the read fence exists for.
        let entry = state.lookup("sake").unwrap();
        state.delete("sake").unwrap().unwrap();
        assert!(
            entry.read_unless_deleted().is_none(),
            "a handle from before the delete must see the tombstone"
        );
        assert!(
            state.store_passages("sake", plain(passages)).is_none(),
            "the name is gone; nothing may recreate it"
        );
        assert!(
            !passages_wal_path(&dir, &file_stem("sake")).exists(),
            "no passage file rose from the dead"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn eviction_drops_resident_passages_and_a_later_lookup_still_answers() {
        let dir = scratch_dir("passage-evict");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "第1段落".to_string(),
            "仕込み水は雲居山の伏流水。".to_string(),
        );
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        let entry = state.lookup("sake").unwrap();
        assert!(state.evict_entry("sake", &entry));
        assert!(
            entry.passages.lock().unwrap().is_none(),
            "eviction must drop the resident passage store"
        );
        // Durability never depended on residency: the next access
        // reloads from the log (or the snapshot the eviction wrote).
        let (found, missing) = state
            .lookup_passages("sake", &["第1段落".to_string()])
            .unwrap()
            .unwrap();
        assert!(found["第1段落"].starts_with("仕込み水"));
        assert!(missing.is_empty());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_second_eviction_of_the_same_entry_frees_nothing() {
        // Two concurrent budget sweeps can snapshot the same candidate;
        // the loser must report `false` so the caller does not subtract
        // the freed bytes from the residency estimate a second time.
        let dir = scratch_dir("double-evict");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "第1段落".to_string(),
            "仕込み水は雲居山の伏流水。".to_string(),
        );
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        let entry = state.lookup("sake").unwrap();
        // The first eviction does the work and reports it.
        assert!(state.evict_entry("sake", &entry));
        // The second finds the slot cold and every cache already cleared,
        // so it frees nothing — and must say so.
        assert!(
            !state.evict_entry("sake", &entry),
            "a repeat eviction must not claim a second freeing"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn cache_budget_accounts_for_resident_passage_text() {
        let dir = scratch_dir("passage-budget");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let before = state.gauge_snapshot().resident_bytes;
        let mut passages = BTreeMap::new();
        passages.insert("大きな段落".to_string(), "あ".repeat(300_000));
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        let after = state.gauge_snapshot().resident_bytes;
        assert!(
            after >= before + 900_000,
            "resident passage text must count against the budget \
             (before {before}, after {after})"
        );

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
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        // The procedural question never became a triple; the text lane
        // must still hand back the passage that answers it, first.
        let hits = state
            .search_passages("sake", "精米歩合はどこまで磨く?", 3)
            .unwrap()
            .unwrap();
        assert_eq!(hits[0].source, "第2段落");
        assert!(hits[0].score > 0.0);

        // No shared bigrams at all → nothing, not noise.
        assert!(
            state
                .search_passages("sake", "unrelated english words", 3)
                .unwrap()
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
        state
            .store_passages("papers", plain(passages))
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages("papers", "ambition must be made to counteract ambition", 2)
            .unwrap()
            .unwrap();
        assert_eq!(hits[0].source, "第51篇");
        assert!(
            hits.len() < 2 || hits[0].score > hits[1].score,
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
        state
            .store_passages("code", plain(passages))
            .unwrap()
            .unwrap();

        for query in ["state", "State", "app", "path"] {
            let hits = state.search_passages("code", query, 3).unwrap().unwrap();
            assert_eq!(
                hits.first().map(|hit| hit.source.as_str()),
                Some("src/registry.rs:AppState"),
                "a piece of a camelCase identifier must reach its passage (query {query:?})"
            );
        }

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn passage_search_returns_the_answering_paragraph_not_the_whole_document() {
        let dir = scratch_dir("bm25-paragraph");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // One document, three paragraphs; only the middle one answers.
        let mut passages = BTreeMap::new();
        passages.insert(
            "docs/aomine.md".to_string(),
            "青嶺酒造は雲居県霧沢町の蔵元である。\n\n原料米には山田錦を使い、精米歩合は50パーセントまで磨く。\n\n蔵開きの祭りでは新酒がふるまわれる。".to_string(),
        );
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages("sake", "精米歩合はどこまで磨く?", 3)
            .unwrap()
            .unwrap();
        assert_eq!(
            (hits[0].source.as_str(), hits[0].index),
            ("docs/aomine.md", 1),
            "the hit names the paragraph, not just the document"
        );
        assert_eq!(
            hits[0].text, "原料米には山田錦を使い、精米歩合は50パーセントまで磨く。",
            "the text is the answering paragraph alone"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_store_after_the_first_search_updates_the_index_in_place() {
        let dir = scratch_dir("bm25-incremental");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("第1章".to_string(), "青嶺酒造の創業は1907年。".to_string());
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        // First search builds the resident index.
        assert!(
            !state
                .search_passages("sake", "創業はいつ", 3)
                .unwrap()
                .unwrap()
                .is_empty()
        );

        // A later store must reach searches through the in-place
        // update — no reclaim is due, so a rebuild cannot be the
        // reason this passes.
        let mut more = BTreeMap::new();
        more.insert(
            "第2章".to_string(),
            "杜氏の高瀬は南部杜氏の出身。".to_string(),
        );
        state.store_passages("sake", plain(more)).unwrap().unwrap();
        let hits = state
            .search_passages("sake", "杜氏の出身", 3)
            .unwrap()
            .unwrap();
        assert_eq!(hits[0].source, "第2章");

        // And a retraction disappears the same way.
        state.retract_source("sake", "第2章").unwrap();
        assert!(
            state
                .search_passages("sake", "杜氏の出身", 3)
                .unwrap()
                .unwrap()
                .iter()
                .all(|hit| hit.source != "第2章"),
            "a retracted source must leave the index too"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn passage_search_survives_a_restart_without_retokenizing() {
        let dir = scratch_dir("bm25-persist");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            let mut passages = BTreeMap::new();
            passages.insert("第1章".to_string(), "青嶺酒造の創業は1907年。".to_string());
            state
                .store_passages("sake", plain(passages))
                .unwrap()
                .unwrap();
            // First search builds and marks dirty; the tick persists.
            state
                .search_passages("sake", "創業はいつ", 3)
                .unwrap()
                .unwrap();
            state.flush_dirty();
            assert!(bm25_path(&dir, &file_stem("sake")).exists());
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let hits = state
            .search_passages("sake", "創業はいつ", 3)
            .unwrap()
            .unwrap();
        assert_eq!(hits[0].source, "第1章");
        let entry = state.lookup("sake").unwrap();
        assert!(
            !entry.bm25_dirty.load(Ordering::Relaxed),
            "a clean sidecar loads as-is — nothing drifted, nothing re-tokenized"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_stale_index_sidecar_is_repaired_by_the_source_digest_mismatch() {
        let dir = scratch_dir("bm25-stale");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            let mut passages = BTreeMap::new();
            passages.insert("第1章".to_string(), "杜氏は高瀬である。".to_string());
            passages.insert("第2章".to_string(), "仕込み水は伏流水。".to_string());
            state
                .store_passages("sake", plain(passages))
                .unwrap()
                .unwrap();
            state.search_passages("sake", "杜氏", 3).unwrap().unwrap();
            state.flush_dirty(); // the sidecar now says 高瀬
        }

        // A new run edits 第1章 and searches BEFORE any flush: the
        // sidecar on disk still carries the old paragraph.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let mut edited = BTreeMap::new();
        edited.insert("第1章".to_string(), "杜氏は佐伯に交代した。".to_string());
        state
            .store_passages("sake", plain(edited))
            .unwrap()
            .unwrap();
        let hits = state
            .search_passages("sake", "杜氏は誰", 3)
            .unwrap()
            .unwrap();
        assert!(
            hits[0].text.contains("佐伯"),
            "the digest mismatch must repair 第1章 from the store, got {:?}",
            hits[0].text
        );
        let entry = state.lookup("sake").unwrap();
        assert!(
            entry.bm25_dirty.load(Ordering::Relaxed),
            "a repair leaves the sidecar stale until the next tick"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_corrupt_index_sidecar_falls_back_to_a_full_rebuild_not_an_outage() {
        let dir = scratch_dir("bm25-corrupt");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "第1章".to_string(),
            "蔵開きの祭りでは新酒がふるまわれる。".to_string(),
        );
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        fs::write(bm25_path(&dir, &file_stem("sake")), b"not an index").unwrap();

        let hits = state
            .search_passages("sake", "蔵開きの祭り", 3)
            .unwrap()
            .unwrap();
        assert_eq!(hits[0].source, "第1章");
        state.flush_dirty();
        assert!(
            crate::bm25::Bm25Index::load(&bm25_path(&dir, &file_stem("sake"))).is_some(),
            "the tick replaces the corpse with a valid sidecar"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn eviction_persists_a_dirty_index_for_the_next_residency() {
        let dir = scratch_dir("bm25-evict-save");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "第1章".to_string(),
            "麹室の湿度は五十パーセント。".to_string(),
        );
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        state
            .search_passages("sake", "麹室の湿度", 3)
            .unwrap()
            .unwrap();

        let entry = state.lookup("sake").unwrap();
        assert!(state.evict_entry("sake", &entry));
        assert!(
            bm25_path(&dir, &file_stem("sake")).exists(),
            "a dirty index rides out with the eviction"
        );
        // The next residency loads it clean instead of re-tokenizing.
        state
            .search_passages("sake", "麹室の湿度", 3)
            .unwrap()
            .unwrap();
        let entry = state.lookup("sake").unwrap();
        assert!(!entry.bm25_dirty.load(Ordering::Relaxed));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn eviction_drops_the_resident_index_and_a_later_search_rebuilds_it() {
        let dir = scratch_dir("bm25-evict");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "第1章".to_string(),
            "蔵開きの祭りでは新酒がふるまわれる。".to_string(),
        );
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        assert!(
            !state
                .search_passages("sake", "蔵開きの祭り", 3)
                .unwrap()
                .unwrap()
                .is_empty()
        );

        let entry = state.lookup("sake").unwrap();
        assert!(state.evict_entry("sake", &entry));
        assert!(
            entry.bm25.read().unwrap().is_none(),
            "eviction must drop the resident index"
        );
        assert_eq!(
            state
                .search_passages("sake", "蔵開きの祭り", 3)
                .unwrap()
                .unwrap()[0]
                .source,
            "第1章",
            "the next search rebuilds and still answers"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// Wraps a plain source→text map as submissions — the shape almost
    /// every passage test wants.
    fn plain(
        passages: BTreeMap<String, String>,
    ) -> BTreeMap<String, crate::passages::PassageSubmission> {
        passages
            .into_iter()
            .map(|(source, text)| (source, crate::passages::PassageSubmission::plain(text)))
            .collect()
    }

    fn boot_for_passage_embedding(
        dir: &Path,
        embedder: Arc<dyn EmbeddingProvider>,
        limit: usize,
    ) -> AppState {
        AppState::boot_with(
            dir.to_path_buf(),
            usize::MAX,
            Some(embedder),
            BootOptions {
                embed_passages: true,
                passage_vector_limit: limit,
                ..BootOptions::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn refresh_passage_embeddings_embeds_every_paragraph_once_then_nothing() {
        let dir = scratch_dir("pvec-first");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "doc-a".to_string(),
            "最初の段落。\n\n二番目の段落。".to_string(),
        );
        passages.insert("doc-b".to_string(), "三番目の段落。".to_string());
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        assert_eq!(
            state.passage_embed_dirty_names(),
            vec!["sake".to_string()],
            "a store marks the context for the auto ticker"
        );

        let outcome = state.refresh_passage_embeddings("sake").unwrap().unwrap();
        assert_eq!(
            (outcome.embedded, outcome.total, outcome.skipped_over_limit),
            (3, 3, 0)
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "three paragraphs fit one provider batch"
        );
        assert!(
            state.passage_embed_dirty_names().is_empty(),
            "the refresh claims the dirty flag"
        );

        // Unchanged corpus: nothing embeds, nobody talks to the provider.
        let again = state.refresh_passage_embeddings("sake").unwrap().unwrap();
        assert_eq!((again.embedded, again.total), (0, 3));
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn refresh_passage_embeddings_re_embeds_only_the_changed_paragraph() {
        let dir = scratch_dir("pvec-diff");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "doc-a".to_string(),
            "変わらない段落。\n\n古い版の段落。".to_string(),
        );
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        state.refresh_passage_embeddings("sake").unwrap().unwrap();

        let mut updated = BTreeMap::new();
        updated.insert(
            "doc-a".to_string(),
            "変わらない段落。\n\n新しい版の段落。".to_string(),
        );
        state
            .store_passages("sake", plain(updated))
            .unwrap()
            .unwrap();
        let outcome = state.refresh_passage_embeddings("sake").unwrap().unwrap();
        assert_eq!(
            outcome.embedded, 1,
            "the unchanged paragraph rides its hash, only the edit re-embeds"
        );
        assert_eq!(outcome.total, 2);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn refresh_passage_embeddings_prunes_vectors_for_a_retracted_source() {
        let dir = scratch_dir("pvec-prune");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "残る段落。".to_string());
        passages.insert("doc-b".to_string(), "消える段落。".to_string());
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        state.refresh_passage_embeddings("sake").unwrap().unwrap();

        state.retract_source("sake", "doc-b").unwrap();
        let outcome = state.refresh_passage_embeddings("sake").unwrap().unwrap();
        assert_eq!(
            (outcome.embedded, outcome.total),
            (0, 1),
            "the retracted source's row is gone without any re-embedding"
        );
        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("sake")));
        assert_eq!(sidecar.len(), 1, "the prune reached the disk too");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn refresh_passage_embeddings_skips_paragraphs_beyond_the_configured_limit() {
        let dir = scratch_dir("pvec-limit");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state = boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 2);
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "doc-a".to_string(),
            "一つ目。\n\n二つ目。\n\n三つ目。".to_string(),
        );
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        let outcome = state.refresh_passage_embeddings("sake").unwrap().unwrap();
        assert_eq!(
            (outcome.embedded, outcome.total, outcome.skipped_over_limit),
            (2, 2, 1),
            "past the limit the lexical lane still serves; only the vector lane goes partial"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn refresh_passage_embeddings_persists_partial_progress_after_a_provider_failure() {
        /// Succeeds except on exactly its `fail_on`-th call (0-based).
        struct FlakyEmbeddings {
            calls: std::sync::atomic::AtomicUsize,
            fail_on: usize,
        }
        impl EmbeddingProvider for FlakyEmbeddings {
            fn model(&self) -> &str {
                "flaky"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
            ) -> Result<Vec<Vec<f32>>, String> {
                let call = self.calls.fetch_add(1, Ordering::Relaxed);
                if call == self.fail_on {
                    return Err("provider hiccup".to_string());
                }
                Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
            }
        }

        let dir = scratch_dir("pvec-partial");
        let state = boot_for_passage_embedding(
            &dir,
            Arc::new(FlakyEmbeddings {
                calls: std::sync::atomic::AtomicUsize::new(0),
                fail_on: 1,
            }),
            20_000,
        );
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // 129 paragraphs = one full batch of 128 plus one more, so the
        // second provider call is the one that fails.
        let text = (0..129)
            .map(|i| format!("段落その{i}。"))
            .collect::<Vec<_>>()
            .join("\n\n");
        let mut passages = BTreeMap::new();
        passages.insert("doc-big".to_string(), text);
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        let error = state
            .refresh_passage_embeddings("sake")
            .unwrap()
            .unwrap_err();
        assert!(error.contains("hiccup"), "{error}");
        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("sake")));
        assert_eq!(
            sidecar.len(),
            128,
            "the batch that landed is durable despite the failure"
        );
        assert_eq!(
            state.passage_embed_dirty_names(),
            vec!["sake".to_string()],
            "unfinished work stays claimed for the ticker"
        );

        // The next refresh buys only the missing paragraph.
        let outcome = state.refresh_passage_embeddings("sake").unwrap().unwrap();
        assert_eq!((outcome.embedded, outcome.total), (1, 129));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_stored_question_row_answers_a_question_shaped_query_at_its_paragraph() {
        let dir = scratch_dir("doc2query-hit");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "doc".to_string(),
            crate::passages::PassageSubmission {
                text: "りんごは真っ赤に実った。".to_string(),
                questions: vec![(0, "アップルはどんな色?".to_string())],
                sections: Vec::new(),
            },
        );
        state.store_passages("fruit", passages).unwrap().unwrap();
        let outcome = state.refresh_passage_embeddings("fruit").unwrap().unwrap();
        assert_eq!(
            (outcome.embedded, outcome.total),
            (2, 2),
            "the paragraph row and its question row both embed"
        );

        // The query matches the QUESTION row's wording, not the text's;
        // both rows point at the same paragraph, so the lane must fold
        // them into one hit at the question row's better rank.
        let hits = state
            .search_passages("fruit", "アップル", 3)
            .unwrap()
            .unwrap();
        assert_eq!(hits.len(), 1, "one paragraph, one hit — never a dup");
        assert_eq!((hits[0].source.as_str(), hits[0].index), ("doc", 0));
        assert!(
            hits[0].text.contains("りんご"),
            "the text served is the PARAGRAPH"
        );
        let (rank, cosine) = hits[0].vector.expect("found via the question row");
        assert_eq!(rank, 1);
        assert!(cosine > 0.99, "{cosine}");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn refresh_re_embeds_only_the_question_whose_text_changed() {
        let dir = scratch_dir("doc2query-diff");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let submission = |question: &str| {
            let mut passages = BTreeMap::new();
            passages.insert(
                "doc".to_string(),
                crate::passages::PassageSubmission {
                    text: "りんごは真っ赤に実った。".to_string(),
                    questions: vec![(0, question.to_string())],
                    sections: Vec::new(),
                },
            );
            passages
        };
        state
            .store_passages("fruit", submission("アップルはどんな色?"))
            .unwrap()
            .unwrap();
        state.refresh_passage_embeddings("fruit").unwrap().unwrap();

        state
            .store_passages("fruit", submission("アップルは何色ですか?"))
            .unwrap()
            .unwrap();
        let outcome = state.refresh_passage_embeddings("fruit").unwrap().unwrap();
        assert_eq!(
            (outcome.embedded, outcome.total),
            (1, 2),
            "the unchanged paragraph row is carried; only the reworded question re-embeds"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn refresh_counts_question_rows_against_the_vector_limit() {
        let dir = scratch_dir("doc2query-limit");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state = boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 2);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "doc".to_string(),
            crate::passages::PassageSubmission {
                text: "りんごは真っ赤に実った。".to_string(),
                questions: vec![
                    (0, "アップルはどんな色?".to_string()),
                    (0, "みかんとの違いは?".to_string()),
                ],
                sections: Vec::new(),
            },
        );
        state.store_passages("fruit", passages).unwrap().unwrap();
        let outcome = state.refresh_passage_embeddings("fruit").unwrap().unwrap();
        assert_eq!(
            (outcome.embedded, outcome.total, outcome.skipped_over_limit),
            (2, 2, 1),
            "a question row spends the same budget a paragraph row does"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn hybrid_search_surfaces_a_vector_only_hit_that_shares_no_letters() {
        // The Q→A gap in miniature: the query shares no bigrams with
        // the stored paragraph, so the lexical lane alone returns
        // nothing — the vector lane is what finds it, and the response
        // says so.
        let dir = scratch_dir("hybrid-vector-only");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごは真っ赤に実った。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();
        state.refresh_passage_embeddings("fruit").unwrap().unwrap();

        let hits = state
            .search_passages("fruit", "アップル", 3)
            .unwrap()
            .unwrap();
        assert_eq!(hits[0].source, "doc-a");
        assert!(
            hits[0].bm25.is_none(),
            "no shared bigram — the lexical lane must not have seen this"
        );
        let (rank, cosine) = hits[0].vector.expect("the vector lane found it");
        assert_eq!(rank, 1);
        assert!(cosine > 0.9, "{cosine}");
        assert!(
            (hits[0].score - 1.0 / 61.0).abs() < 1e-6,
            "one lane at rank 1 fuses to 1/(60+1), got {}",
            hits[0].score
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn hybrid_search_reports_both_lanes_when_both_agree() {
        let dir = scratch_dir("hybrid-both");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごは真っ赤に実った。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();
        state.refresh_passage_embeddings("fruit").unwrap().unwrap();

        let hits = state
            .search_passages("fruit", "りんごは真っ赤", 3)
            .unwrap()
            .unwrap();
        assert_eq!(hits[0].source, "doc-a");
        let (bm25_rank, bm25_score) = hits[0].bm25.expect("shared bigrams");
        let (vector_rank, _) = hits[0].vector.expect("same fruit, same axis");
        assert_eq!((bm25_rank, vector_rank), (1, 1));
        assert!(bm25_score > 0.0);
        assert!(
            (hits[0].score - 2.0 / 61.0).abs() < 1e-6,
            "two lanes at rank 1 fuse to 2/(60+1), got {}",
            hits[0].score
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_lane_that_scored_outdated_text_loses_its_evidence_not_the_hit() {
        // Vectors refresh on their own cadence and routinely lag the
        // text. After an edit, the in-place-updated BM25 lane must
        // keep answering while the vector lane's stale evidence is
        // dropped — never attached to text it did not score, and never
        // vetoing the fresh lexical match.
        let dir = scratch_dir("hybrid-stale-lane");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごは真っ赤に実った。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();
        state.refresh_passage_embeddings("fruit").unwrap().unwrap();

        // The text changes; the vector sidecar is NOT refreshed.
        let mut edited = BTreeMap::new();
        edited.insert(
            "doc-a".to_string(),
            "りんごは青森の名産である。".to_string(),
        );
        state
            .store_passages("fruit", plain(edited))
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages("fruit", "りんご", 3)
            .unwrap()
            .unwrap();
        assert_eq!(hits[0].source, "doc-a");
        assert!(hits[0].text.contains("青森"), "the CURRENT text is served");
        assert!(
            hits[0].bm25.is_some(),
            "the lexical lane scored the fresh text and survives"
        );
        assert!(
            hits[0].vector.is_none(),
            "the vector lane scored the OLD text; its evidence must drop"
        );
        let (bm25_rank, _) = hits[0].bm25.unwrap();
        assert!(
            (hits[0].score - 1.0 / (60.0 + bm25_rank as f32)).abs() < 1e-6,
            "the fused score counts surviving lanes only, got {}",
            hits[0].score
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn hybrid_search_degrades_to_bm25_when_the_query_embedding_fails() {
        struct FlakyEmbeddings {
            calls: std::sync::atomic::AtomicUsize,
            fail_on: usize,
        }
        impl EmbeddingProvider for FlakyEmbeddings {
            fn model(&self) -> &str {
                "flaky"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
            ) -> Result<Vec<Vec<f32>>, String> {
                let call = self.calls.fetch_add(1, Ordering::Relaxed);
                if call == self.fail_on {
                    return Err("provider hiccup".to_string());
                }
                Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
            }
        }

        let dir = scratch_dir("hybrid-degrade");
        let state = boot_for_passage_embedding(
            &dir,
            Arc::new(FlakyEmbeddings {
                calls: std::sync::atomic::AtomicUsize::new(0),
                fail_on: 1, // the refresh succeeds; the QUERY embed fails
            }),
            20_000,
        );
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごは真っ赤に実った。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();
        state.refresh_passage_embeddings("fruit").unwrap().unwrap();

        let hits = state
            .search_passages("fruit", "りんご", 3)
            .unwrap()
            .unwrap();
        assert_eq!(
            hits[0].source, "doc-a",
            "a broken decoration must not break the answer"
        );
        assert!(hits[0].vector.is_none());
        let (_, bm25_score) = hits[0].bm25.unwrap();
        assert_eq!(
            hits[0].score, bm25_score,
            "with no semantic lane the score stays the raw BM25 number"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn lexical_only_deployments_keep_raw_bm25_scores() {
        let dir = scratch_dir("hybrid-lexical-only");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごは真っ赤に実った。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages("fruit", "りんご", 3)
            .unwrap()
            .unwrap();
        assert!(hits[0].vector.is_none());
        let (rank, bm25_score) = hits[0].bm25.unwrap();
        assert_eq!(rank, 1);
        assert_eq!(hits[0].score, bm25_score);
        assert!(hits[0].score > 0.1, "raw BM25, not a tiny RRF quotient");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn search_passages_vector_lane_drops_candidates_below_the_default_semantic_floor() {
        // みかん×りんご sits at cosine 0.28 — under the 0.35 default
        // floor — and the query shares no bigram with the stored text,
        // so a fusion that ignored the floor would surface a
        // near-irrelevant paragraph on the vector lane alone.
        let dir = scratch_dir("passages-floor-default");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごは真っ赤に実った。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();
        state.refresh_passage_embeddings("fruit").unwrap().unwrap();

        let hits = state
            .search_passages("fruit", "みかん", 3)
            .unwrap()
            .unwrap();
        assert!(
            hits.is_empty(),
            "cosine 0.28 sits under the default floor: {hits:?}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn search_passages_vector_lane_honors_a_lowered_context_semantic_floor() {
        // Same 0.28 cosine as the default-floor test, but the context's
        // floor is lowered under it: the candidate must now clear the
        // vector lane and contribute its RRF term, same as
        // semantic_resolve honors the context setting over the server
        // default.
        let dir = scratch_dir("passages-floor-context");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごは真っ赤に実った。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();
        state.refresh_passage_embeddings("fruit").unwrap().unwrap();
        state
            .update_meta("fruit", None, None, None, Some(0.2))
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages("fruit", "みかん", 3)
            .unwrap()
            .unwrap();
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].source, "doc-a");
        let (rank, cosine) = hits[0].vector.expect("cleared the lowered floor");
        assert_eq!(rank, 1);
        assert!((cosine - 0.28).abs() < 1e-6, "{cosine}");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn refresh_passage_embeddings_requires_the_opt_in() {
        let dir = scratch_dir("pvec-optin");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let embedder = Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let error = state
            .refresh_passage_embeddings("sake")
            .unwrap()
            .unwrap_err();
        assert!(error.contains("TAGURU_EMBED_PASSAGES"), "{error}");

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

    /// A provider that changes output width behind a stable model name
    /// (a backend swap behind the same proxy) must stale the whole
    /// carried table: gloss hashes are unchanged, so without the width
    /// check nothing re-embeds and old-width rows sit next to new-width
    /// ones — which `similarity` scores as nothing, silently.
    #[test]
    fn a_width_change_under_the_same_model_name_re_embeds_everything() {
        struct WidthEmbeddings(usize);
        impl EmbeddingProvider for WidthEmbeddings {
            fn model(&self) -> &str {
                "stable-name"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
            ) -> Result<Vec<Vec<f32>>, String> {
                Ok(texts
                    .iter()
                    .map(|_| {
                        let mut vector = vec![0.0; self.0];
                        vector[0] = 1.0;
                        vector
                    })
                    .collect())
            }
        }

        let dir = scratch_dir("width-change");
        {
            let embedder = Some(Arc::new(WidthEmbeddings(2)) as Arc<dyn EmbeddingProvider>);
            let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
            state
                .create("w", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .write_context("w", |context| {
                    context.associate("a", "l", "b", 1.0).unwrap();
                })
                .map_err(|_| "write")
                .unwrap();
            let (embedded, total) = state.refresh_embeddings("w").unwrap().unwrap();
            assert_eq!((embedded, total), (3, 3)); // a, b, and the label l
            state.flush_dirty();
        }

        // Same model name, wider vectors: every gloss must re-embed
        // (hashes alone would say "nothing to do") and the published
        // sidecar must be uniformly the new width.
        let embedder = Some(Arc::new(WidthEmbeddings(3)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        let (embedded, total) = state.refresh_embeddings("w").unwrap().unwrap();
        assert_eq!((embedded, total), (3, 3));
        let store = VectorStore::load(&vectors_path(&dir, &file_stem("w")));
        assert!(
            store
                .concepts
                .values()
                .chain(store.labels.values())
                .all(|(_, vector)| vector.len() == 3),
            "old-width rows must not survive the width change"
        );

        // A no-op refresh against the same-width provider stays a no-op
        // (the probe embeds one gloss but re-embeds nothing).
        let (embedded, total) = state.refresh_embeddings("w").unwrap().unwrap();
        assert_eq!((embedded, total), (0, 3));

        let _ = fs::remove_dir_all(dir);
    }

    /// `existing`/`embed_stale` run before the entry's data lock is
    /// ever taken — provider round trips can take seconds and must not
    /// block graph reads — so two concurrent first-time refreshes would
    /// both diff against the same empty sidecar and both call the
    /// provider. Unless `vectors_refresh` excludes them for the whole
    /// refresh (not just the merge), those two provider calls overlap;
    /// whichever refresh then merges last silently wins over the
    /// other's, with no ordering guarantee that the winner saw the
    /// newer gloss. This pins the observable down directly: the
    /// provider must never see two calls in flight at once.
    #[test]
    fn concurrent_gloss_refreshes_serialize_their_provider_calls() {
        use std::sync::atomic::AtomicUsize;
        use std::thread;
        use std::time::Duration;

        struct SlowEmbeddings {
            in_flight: Arc<AtomicUsize>,
            peak: Arc<AtomicUsize>,
        }
        impl EmbeddingProvider for SlowEmbeddings {
            fn model(&self) -> &str {
                "slow"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
            ) -> Result<Vec<Vec<f32>>, String> {
                let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                self.peak.fetch_max(now, Ordering::SeqCst);
                // Long enough that two unserialized refreshes' provider
                // calls MUST overlap unless `vectors_refresh` excludes them.
                thread::sleep(Duration::from_millis(150));
                self.in_flight.fetch_sub(1, Ordering::SeqCst);
                Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
            }
        }

        let dir = scratch_dir("refresh-serialize");
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let embedder = Some(Arc::new(SlowEmbeddings {
            in_flight: Arc::clone(&in_flight),
            peak: Arc::clone(&peak),
        }) as Arc<dyn EmbeddingProvider>);
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

        let mut refreshers = Vec::new();
        for _ in 0..2 {
            let state = state.clone();
            refreshers.push(thread::spawn(move || {
                state.refresh_embeddings("fruit").unwrap().unwrap();
            }));
        }
        for refresher in refreshers {
            refresher.join().unwrap();
        }

        assert_eq!(
            peak.load(Ordering::SeqCst),
            1,
            "two first-time refreshes both diff against an empty sidecar; without \
             vectors_refresh serializing the whole refresh, their provider calls \
             overlap and whichever merges last can clobber a fresher gloss with a \
             staler one"
        );

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
            BootOptions {
                default_semantic_floor: Some(0.2),
                ..BootOptions::default()
            },
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

    #[test]
    fn group_crud_validates_members_and_survives_a_reboot() {
        let dir = scratch_dir("groups-crud");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state.create("beer", ContextMeta::default()).unwrap();

        state
            .create_group(
                "drinks",
                "beverage knowledge".into(),
                BTreeSet::from(["sake".to_string()]),
                BTreeSet::new(),
            )
            .unwrap();
        assert!(matches!(
            state.create_group("drinks", String::new(), BTreeSet::new(), BTreeSet::new()),
            Err(CreateGroupError::AlreadyExists)
        ));
        assert!(matches!(
            state.create_group("", String::new(), BTreeSet::new(), BTreeSet::new()),
            Err(CreateGroupError::InvalidName)
        ));
        assert!(matches!(
            state.create_group(
                "ghosts",
                String::new(),
                BTreeSet::from(["nope".to_string()]),
                BTreeSet::new()
            ),
            Err(CreateGroupError::NoSuchContext(missing)) if missing == "nope"
        ));
        // The refused create left nothing behind, in memory or on disk.
        assert!(state.group("ghosts").is_none());
        assert!(!groups::group_path(&dir, &file_stem("ghosts")).exists());

        // Deltas: removals first, then adds — and an unknown add
        // refuses the whole update, membership untouched.
        assert!(matches!(
            state.update_group(
                "drinks",
                None,
                BTreeSet::from(["nope".to_string()]),
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
            ),
            Err(UpdateGroupError::NoSuchContext(missing)) if missing == "nope"
        ));
        assert_eq!(
            state.group("drinks").unwrap().contexts,
            BTreeSet::from(["sake".to_string()])
        );
        let updated = state
            .update_group(
                "drinks",
                Some("all drinks".into()),
                BTreeSet::from(["beer".to_string()]),
                BTreeSet::from(["sake".to_string()]),
                BTreeSet::new(),
                BTreeSet::new(),
            )
            .unwrap();
        assert_eq!(updated.description, "all drinks");
        assert_eq!(updated.contexts, BTreeSet::from(["beer".to_string()]));
        assert!(matches!(
            state.update_group(
                "ghosts",
                None,
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new()
            ),
            Err(UpdateGroupError::NotFound)
        ));

        // The whole collection survives a reboot from disk alone.
        drop(state);
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let survived = state.group("drinks").unwrap();
        assert_eq!(survived.description, "all drinks");
        assert_eq!(survived.contexts, BTreeSet::from(["beer".to_string()]));

        // Deletion removes the record and its file; the members live on.
        state.delete_group("drinks").unwrap().unwrap();
        assert!(state.group("drinks").is_none());
        assert!(state.delete_group("drinks").is_none());
        assert!(!groups::group_path(&dir, &file_stem("drinks")).exists());
        assert_eq!(state.group_page(None, usize::MAX), (0, Vec::new()));
        assert!(state.directory().iter().any(|entry| entry.name == "beer"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn deleting_a_context_sweeps_it_out_of_every_group() {
        let dir = scratch_dir("groups-sweep");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state.create("beer", ContextMeta::default()).unwrap();
        for group in ["drinks", "fermented"] {
            state
                .create_group(
                    group,
                    String::new(),
                    BTreeSet::from(["sake".to_string(), "beer".to_string()]),
                    BTreeSet::new(),
                )
                .unwrap();
        }

        state.delete("sake").unwrap().unwrap();

        for group in ["drinks", "fermented"] {
            assert_eq!(
                state.group(group).unwrap().contexts,
                BTreeSet::from(["beer".to_string()]),
                "'{group}' still names the deleted context"
            );
        }
        // The sweep persisted: a reboot reads the same membership.
        drop(state);
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        for group in ["drinks", "fermented"] {
            assert_eq!(
                state.group(group).unwrap().contexts,
                BTreeSet::from(["beer".to_string()])
            );
        }

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn boot_reconciliation_drops_dangling_members_and_rewrites_the_file() {
        let dir = scratch_dir("groups-reconcile");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        drop(state);

        // A dangling member, planted the way a crash between a
        // context's deletion and the sweep's rewrite would leave one.
        groups::write_group(
            &dir,
            &file_stem("drinks"),
            &GroupRecord {
                description: "d".into(),
                contexts: BTreeSet::from(["sake".to_string(), "gone".to_string()]),
                groups: BTreeSet::new(),
            },
        )
        .unwrap();

        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        assert_eq!(
            state.group("drinks").unwrap().contexts,
            BTreeSet::from(["sake".to_string()])
        );
        // Disk is the source of truth: the fix reached the file, not
        // just memory.
        let on_disk = fs::read_to_string(groups::group_path(&dir, &file_stem("drinks"))).unwrap();
        assert!(
            !on_disk.contains("gone"),
            "the dangling member survived on disk: {on_disk}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_corrupt_group_file_keeps_its_name_and_loses_its_content() {
        let dir = scratch_dir("groups-corrupt");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        drop(state);
        fs::write(
            groups::group_path(&dir, &file_stem("mangled")),
            b"{not json",
        )
        .unwrap();

        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let record = state.group("mangled").unwrap();
        assert_eq!(record, GroupRecord::default());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn groups_nest_to_the_cap_without_cycles_and_sweep_deleted_children() {
        use crate::groups::NestingViolation;

        let dir = scratch_dir("groups-nesting");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        for context in ["sake", "beer", "tea"] {
            state.create(context, ContextMeta::default()).unwrap();
        }
        let one = |name: &str| BTreeSet::from([name.to_string()]);
        state
            .create_group("leaf", String::new(), one("tea"), BTreeSet::new())
            .unwrap();
        state
            .create_group("mid", String::new(), one("beer"), one("leaf"))
            .unwrap();
        state
            .create_group("top", String::new(), one("sake"), one("mid"))
            .unwrap();

        // The closure reads through the nesting; the record stays flat.
        assert_eq!(
            state.group_context_closures(["top"]),
            ["sake", "beer", "tea"]
                .iter()
                .map(|c| c.to_string())
                .collect()
        );
        assert_eq!(state.group("top").unwrap().contexts, one("sake"));

        // A fourth storey over a full chain refuses, and nothing lands.
        assert!(matches!(
            state.create_group("over", String::new(), BTreeSet::new(), one("top")),
            Err(CreateGroupError::Nesting(NestingViolation::TooDeep(_)))
        ));
        assert!(state.group("over").is_none());
        assert!(!groups::group_path(&dir, &file_stem("over")).exists());

        // Closing the chain into a cycle refuses — the self-loop
        // included — and an unknown child answers in the group
        // namespace; the record survives every refusal unchanged.
        assert!(matches!(
            state.update_group(
                "leaf",
                None,
                BTreeSet::new(),
                BTreeSet::new(),
                one("top"),
                BTreeSet::new()
            ),
            Err(UpdateGroupError::Nesting(NestingViolation::Cycle(_)))
        ));
        assert!(matches!(
            state.update_group(
                "leaf",
                None,
                BTreeSet::new(),
                BTreeSet::new(),
                one("leaf"),
                BTreeSet::new()
            ),
            Err(UpdateGroupError::Nesting(NestingViolation::Cycle(_)))
        ));
        assert!(matches!(
            state.update_group(
                "leaf",
                None,
                BTreeSet::new(),
                BTreeSet::new(),
                one("nope"),
                BTreeSet::new()
            ),
            Err(UpdateGroupError::NoSuchGroup(missing)) if missing == "nope"
        ));
        assert_eq!(state.group("leaf").unwrap().groups, BTreeSet::new());

        // Deleting a child sweeps it out of every parent, durably.
        state.delete_group("mid").unwrap().unwrap();
        assert_eq!(state.group("top").unwrap().groups, BTreeSet::new());
        drop(state);
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        assert_eq!(state.group("top").unwrap().groups, BTreeSet::new());
        assert_eq!(state.group("leaf").unwrap().contexts, one("tea"));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn boot_reconciliation_untangles_hand_written_nesting() {
        let dir = scratch_dir("groups-reconcile-nesting");
        drop(AppState::boot(dir.clone(), 1 << 20, None).unwrap());
        let write = |name: &str, children: &[&str]| {
            groups::write_group(
                &dir,
                &file_stem(name),
                &GroupRecord {
                    description: String::new(),
                    contexts: BTreeSet::new(),
                    groups: children.iter().map(|child| child.to_string()).collect(),
                },
            )
            .unwrap();
        };
        // A two-cycle, a four-group chain, and a child that exists
        // nowhere — none of which a running server can persist.
        write("cyc-a", &["cyc-b"]);
        write("cyc-b", &["cyc-a"]);
        write("n1", &["ghost", "n2"]);
        write("n2", &["n3"]);
        write("n3", &["n4"]);
        write("n4", &[]);

        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        // The dangling child is gone, the cycle is open, the chain fits
        // the cap — deterministically: edges re-admitted in name order,
        // so (cyc-b, cyc-a) and (n3, n4) are the ones that fall.
        let children =
            |name: &str| -> BTreeSet<String> { state.group(name).unwrap().groups.clone() };
        assert_eq!(children("n1"), BTreeSet::from(["n2".to_string()]));
        assert_eq!(children("n2"), BTreeSet::from(["n3".to_string()]));
        assert_eq!(children("n3"), BTreeSet::new());
        assert_eq!(children("cyc-a"), BTreeSet::from(["cyc-b".to_string()]));
        assert_eq!(children("cyc-b"), BTreeSet::new());
        // Disk is the source of truth: the repairs reached the files.
        for (group, dropped) in [("cyc-b", "cyc-a"), ("n3", "n4"), ("n1", "ghost")] {
            let on_disk = fs::read_to_string(groups::group_path(&dir, &file_stem(group))).unwrap();
            assert!(
                !on_disk.contains(dropped),
                "'{group}' still names '{dropped}' on disk: {on_disk}"
            );
        }

        let _ = fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[test]
    fn a_failed_group_persist_rolls_the_update_back() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("groups-rollback");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state
            .create_group(
                "drinks",
                "before".into(),
                BTreeSet::from(["sake".to_string()]),
                BTreeSet::new(),
            )
            .unwrap();

        // A directory that refuses the staging write fails the persist;
        // nothing may apply in memory either.
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
        let outcome = state.update_group(
            "drinks",
            Some("after".into()),
            BTreeSet::new(),
            BTreeSet::from(["sake".to_string()]),
            BTreeSet::new(),
            BTreeSet::new(),
        );
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
        assert!(matches!(outcome, Err(UpdateGroupError::Io(_))));
        let record = state.group("drinks").unwrap();
        assert_eq!(record.description, "before");
        assert_eq!(record.contexts, BTreeSet::from(["sake".to_string()]));

        let _ = fs::remove_dir_all(dir);
    }
}
