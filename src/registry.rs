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
//! one context never blocks the others. Locks here are parking_lot, not
//! std::sync: a panic while one is held unwinds without poisoning it, so
//! neither that context nor a sibling nor the registry itself bricks for
//! the rest of the process. Safety across the panic comes from the
//! write-ahead log, not the lock — a write is durable once it's staged
//! and fsynced there, before it ever touches memory, so a panic mid-write
//! loses at most the in-memory half of an update that the next load
//! replays past anyway. Groups (tiny, always-resident records) sit
//! behind one separate lock; the only operations that need both take
//! `groups` BEFORE `registry` — never the other way around.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use taguru::context::{AliasError, CompactionError, Context, LabelUsage, dead_ratio_of};
use taguru::deadline::{Deadline, DeadlineExceeded};

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
    /// Live count of edges with `count == 0` — dead weight `compact`
    /// would shed right now. See [`Context::dead_edges`] for how this
    /// differs from the one-time `CompactionStats::dead_edges`.
    pub dead_edges: usize,
    /// Live count of attribution records unlinked from every chain but
    /// not yet reclaimed by compaction.
    pub dead_attributions: usize,
    /// Lower-bound count of arena bytes occupied by removed aliases'
    /// spellings — see [`Context::arena_slack`] for why it is a lower
    /// bound.
    pub arena_slack: usize,
    /// Live count of edges carrying weight no named source explains —
    /// see [`Context::unsourced_summary`].
    pub unsourced_edges: usize,
    /// Total unsourced weight (absolute value, summed across
    /// `unsourced_edges`) — see [`Context::unsourced_summary`].
    pub unsourced_weight: f64,
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
        let (unsourced_edges, unsourced_weight) = context.unsourced_summary();
        Self {
            associations: context.association_count(),
            concepts: context.concept_count(),
            labels: context.label_count(),
            sources: context.source_count(),
            footprint_bytes: context.footprint(),
            dead_edges: context.dead_edges(),
            dead_attributions: context.dead_attributions(),
            arena_slack: context.arena_slack(),
            unsourced_edges,
            unsourced_weight,
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

    /// Fraction of associations that are currently dead weight — the
    /// same formula [`Context::dead_ratio`] uses, so a hot context
    /// (recomputed live) and a cold one (this cached snapshot) can
    /// never disagree about it.
    pub fn dead_ratio(&self) -> f64 {
        dead_ratio_of(self.dead_edges, self.associations)
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
    /// Set when a gloss-embedding refresh's save to the sidecar fails
    /// after the provider already sold it the rows — `vectors` above
    /// caches them even though the disk does not have them yet.
    /// Cleared on the save that finally lands. The next refresh reads
    /// this, not just its own diff (which reads that same cache and so
    /// can land on "nothing new to buy"), to decide whether to retry
    /// the write even when its own pass bought nothing.
    vectors_save_pending: AtomicBool,
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
        passages_wal_bytes: u64,
        usage: ContextUsage,
    ) -> Self {
        Self {
            inner: RwLock::new(EntryInner {
                meta,
                stats,
                slot,
                wal_seq: 1,
                wal_bytes,
                passages_wal_bytes,
                counted_bytes: 0,
                load_failure: None,
                image_generation: 0,
            }),
            dirty: AtomicBool::new(false),
            flushing: AtomicBool::new(false),
            last_touch: AtomicU64::new(0),
            vectors: Mutex::new(None),
            vectors_refresh: Mutex::new(()),
            vectors_save_pending: AtomicBool::new(false),
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
    fn lock_unless_deleted(&self) -> Option<parking_lot::RwLockWriteGuard<'_, EntryInner>> {
        let guard = self.inner.write();
        (!matches!(guard.slot, Slot::Deleted)).then_some(guard)
    }

    /// The read half of the tombstone fence: passage operations hold
    /// this SHARED guard for their whole run — concurrent with graph
    /// reads and with each other, but correctly serialized against
    /// [`AppState::delete`], whose exclusive lock plants the tombstone.
    /// Whichever side locks first wins cleanly: a fence taken first
    /// makes the delete wait; a tombstone planted first turns the
    /// operation into a no-op instead of a file resurrection.
    fn read_unless_deleted(&self) -> Option<parking_lot::RwLockReadGuard<'_, EntryInner>> {
        let guard = self.inner.read();
        (!matches!(guard.slot, Slot::Deleted)).then_some(guard)
    }

    /// Bytes the cached vector sidecar holds resident — zero when none
    /// is loaded. The cache budget and the gauges count it the same way.
    fn vectors_footprint(&self) -> usize {
        self.vectors
            .lock()
            .as_ref()
            .map(|store| store.footprint())
            .unwrap_or(0)
    }

    /// Bytes the resident passage store holds — zero while cold.
    fn passages_footprint(&self) -> usize {
        self.passages
            .lock()
            .as_ref()
            .map(|store| store.footprint())
            .unwrap_or(0)
    }

    /// Bytes the resident BM25 index holds — zero while cold.
    fn bm25_footprint(&self) -> usize {
        self.bm25
            .read()
            .as_ref()
            .map(|index| index.footprint())
            .unwrap_or(0)
    }

    /// Bytes the resident paragraph vectors hold — zero while cold.
    fn passage_vectors_footprint(&self) -> usize {
        self.passage_vectors
            .lock()
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
    /// Size of this context's passage log on disk while cold — the same
    /// role as `wal_bytes` but for `PassageStore`, which only knows its
    /// own pending bytes while resident. Seeded at scan, refreshed the
    /// moment `evict_entry` drops the store back to cold; `gauge_snapshot`
    /// reads the live store when hot and this field otherwise, so it
    /// never has to re-stat the log on every scrape.
    passages_wal_bytes: u64,
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
    /// One of the two membership sets would hold more than
    /// [`groups::MAX_GROUP_MEMBERS`] names; carries which one
    /// ("member contexts" / "child groups"). Judged before existence,
    /// like the request-list caps: a count needs no lookups.
    OverCap(&'static str),
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
    /// [`CreateGroupError::OverCap`]'s twin, judged on the RESULT of
    /// the delta: removals apply first, so trading members out makes
    /// room in the same request. Checked before existence — the count
    /// needs no lookups, and a delta that cannot fit should not spend
    /// them.
    OverCap(&'static str),
    Io(io::Error),
}

#[derive(Debug)]
pub enum RenameContextError {
    NotFound,
    /// Same trap as [`CreateError::InvalidName`]: an empty destination
    /// would persist as a bare `.ctx` the boot scan never rediscovers.
    InvalidName,
    AlreadyExists,
    /// `from` or `to` is reserved by a create, delete, or another
    /// rename already in flight — retry once it settles.
    Busy,
    Io(io::Error),
}

/// [`AppState::rename_context_locked`]'s result: whether a failure left
/// `from` safely free again or stuck pending a boot resume. See that
/// function's doc for the full reasoning.
enum RenameOutcome {
    Ok,
    RolledBack(RenameContextError),
    Stuck(RenameContextError),
}

#[derive(Debug)]
pub enum RenameGroupError {
    NotFound,
    /// Same trap as [`CreateGroupError::InvalidName`], for the
    /// destination name.
    InvalidName,
    AlreadyExists,
    Io(io::Error),
}

/// Why [`AppState::restore_groups`] refused. Every arm but `Io` is a
/// validation refusal with NOTHING applied — the set is judged whole
/// before the first write. `Io` names how many records persisted
/// first; either way re-importing the same stream is exact, because a
/// restore is a replace and a replay converges.
#[derive(Debug)]
pub enum RestoreGroupsError {
    /// [`CreateGroupError::InvalidName`]'s twin — the parse layers
    /// refuse it first, but the invariant is the registry's.
    InvalidName,
    /// The same record name twice in one set: the set claims two
    /// truths for one group, and "last wins" would silently discard
    /// the other. The stream and file layers refuse duplicates with
    /// their line and path; this is the registry's own backstop.
    Duplicate(String),
    /// A member of the named group is not a registered context. As
    /// strict as [`CreateGroupError::NoSuchContext`] — batches of the
    /// same run applied first, so "import contexts and groups
    /// together" satisfies it, and anything else deserves the refusal.
    NoSuchContext { group: String, context: String },
    /// A child of the named group is neither registered nor in the
    /// restore set itself (records in one set may reference each other
    /// in any order — the set lands children-first).
    NoSuchChild { group: String, child: String },
    /// [`CreateGroupError::OverCap`]'s twin, judged per record.
    OverCap { group: String, field: &'static str },
    /// The set's records plus the standing groups would close a cycle
    /// or stack more than [`groups::MAX_GROUP_DEPTH`] groups.
    Nesting(groups::NestingViolation),
    /// A record would not persist; the `applied` records before it did
    /// (children-first order, so what landed never dangles on what did
    /// not).
    Io {
        group: String,
        applied: usize,
        error: io::Error,
    },
    /// The request budget ran out mid-write; the `applied` records
    /// before it landed durably (children-first order, so what landed
    /// never dangles). Re-importing the stream is exact — a restore is
    /// a replace and a replay converges.
    Timeout { applied: usize },
}

impl RestoreGroupsError {
    /// How many records durably landed despite the refusal — zero for
    /// every validation arm, the write count for [`Self::Io`] and
    /// [`Self::Timeout`].
    pub fn applied(&self) -> usize {
        match self {
            Self::Io { applied, .. } | Self::Timeout { applied } => *applied,
            _ => 0,
        }
    }

    /// One sentence for both entrances, [`crate::ingest::ApplyRefusal::text`]'s
    /// twin: the CLI prints it, the HTTP endpoint sends the same words
    /// under the matching status.
    pub fn text(&self) -> String {
        match self {
            Self::InvalidName => "a group name must not be empty".to_string(),
            Self::Duplicate(group) => {
                format!("group '{group}' appears twice in the restore set")
            }
            Self::NoSuchContext { group, context } => format!(
                "group '{group}' names member context '{context}', which does not exist; \
                 no group was applied (batches of the same stream apply first — import \
                 the contexts, then the groups)"
            ),
            Self::NoSuchChild { group, child } => format!(
                "group '{group}' names child group '{child}', which neither exists nor \
                 rides this stream; no group was applied"
            ),
            Self::OverCap { group, field } => format!(
                "group '{group}' would bundle more than {} {field}; no group was applied \
                 — split into nested child groups",
                groups::MAX_GROUP_MEMBERS
            ),
            Self::Nesting(groups::NestingViolation::Cycle(group)) => format!(
                "the restored nesting would close a cycle through group '{group}'; \
                 no group was applied"
            ),
            Self::Nesting(groups::NestingViolation::TooDeep(group)) => format!(
                "the restored nesting would stack more than {} groups under group \
                 '{group}'; no group was applied",
                groups::MAX_GROUP_DEPTH
            ),
            Self::Io {
                group,
                applied,
                error,
            } => format!(
                "group '{group}' could not be persisted after {applied} group record(s) \
                 landed: {error} — re-importing the stream is exact"
            ),
            Self::Timeout { applied } => format!(
                "the request budget ran out after {applied} group record(s) landed; \
                 re-importing the stream is exact"
            ),
        }
    }
}

/// What restoring one group record amounted to, for the import report:
/// the record now stands either way, and the label says what it
/// replaced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GroupRestoreOutcome {
    Created,
    Replaced,
    /// The standing record already equalled the incoming one — nothing
    /// was rewritten, which is what makes re-importing a stream a
    /// cheap no-op instead of a directory-wide fsync storm.
    Unchanged,
}

impl GroupRestoreOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Replaced => "replaced",
            Self::Unchanged => "unchanged",
        }
    }
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

/// One context's [`CompactOutcome`] inside a
/// [`AppState::run_maintenance_compaction`] sweep, named so the sweep's
/// response can say which contexts it touched.
#[derive(Debug, Clone, Serialize)]
pub struct MaintenanceCompactionEntry {
    pub name: String,
    #[serde(flatten)]
    pub outcome: CompactOutcome,
}

/// What a `POST /maintenance/compact` sweep accomplished: every context
/// it compacted, worst dead ratio first, and whether the deadline cut
/// the sweep short of the full candidate list.
#[derive(Debug, Clone, Serialize)]
pub struct MaintenanceCompactionOutcome {
    pub contexts: Vec<MaintenanceCompactionEntry>,
    pub deadline_exceeded: bool,
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
    /// The request's time budget ran out partway through a
    /// `block_in_place` section. Never produced by the CLI binaries —
    /// they pass `Deadline::unbounded()` — only by HTTP handlers.
    DeadlineExceeded,
}

/// One requested association — the wire shape of the associations
/// endpoint and the WAL payload, one struct for both so they cannot
/// drift apart.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq))]
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
    pub embed_parallel: usize,
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

/// The outcome of locating a search-explain target, in
/// [`CitationLookup`]'s shape: the non-`Explained` arms end the
/// explanation before any scoring runs, and the outer
/// `Option<io::Result<_>>` stays reserved for context-absent /
/// store-unreachable.
#[cfg_attr(test, derive(Debug))]
pub(crate) enum PassageExplainLookup {
    /// The store holds no record for the source — never stored under
    /// this name, or stored and later retracted: the store keeps no
    /// tombstone history, so the two are indistinguishable by design.
    UnknownSource,
    IndexOutOfRange {
        paragraphs: usize,
    },
    /// The query yields no searchable terms; a search of it answers
    /// the empty list before either lane runs.
    NoQueryTerms,
    Explained(Box<PassageSearchExplanation>),
}

/// Everything `explain_passage_search` established, verdict-free — the
/// registry reports what happened; the API layer names the verdict and
/// writes the summary from it.
#[cfg_attr(test, derive(Debug))]
pub(crate) struct PassageSearchExplanation {
    /// The paragraph the explanation settled on.
    pub(crate) paragraph: u32,
    /// How many paragraphs the source holds.
    pub(crate) paragraphs: usize,
    /// Whether the request named the paragraph or the best showing was
    /// chosen for it.
    pub(crate) paragraph_named: bool,
    /// The query's terms — every deduplicated key with its spelling,
    /// in first-occurrence order, exactly the grams both lanes ran.
    pub(crate) query_terms: Vec<(String, u64)>,
    /// The lexical lane's per-term evidence for the target paragraph
    /// (query-term order). `None` when the index holds no live,
    /// current-text slot for it.
    pub(crate) lexical: Option<crate::bm25::IndexEvidence>,
    /// The target paragraph's own spellings (questions included,
    /// deduplicated) — materialized only when the paragraph shares no
    /// term with the query, the verdict that needs both sides shown.
    pub(crate) paragraph_terms: Option<Vec<String>>,
    /// The vector lane's account.
    pub(crate) vector: VectorLaneReport,
    /// Whether the ranking fused both lanes (RRF) or served raw BM25.
    pub(crate) fused: bool,
    /// The full ranking's size: every candidate either lane surfaced,
    /// staleness-validated, nothing truncated.
    pub(crate) ranked: usize,
    /// The target's place in that full ranking (1-based), its score
    /// there, and its per-lane showings.
    pub(crate) rank: Option<usize>,
    pub(crate) score: Option<f32>,
    pub(crate) bm25_lane: Option<(usize, f32)>,
    pub(crate) vector_lane: Option<(usize, f32)>,
    /// The request's limit, whether the target is served at it, and
    /// the weakest score that was — the bar the target had to clear.
    pub(crate) limit: usize,
    pub(crate) served: bool,
    pub(crate) cutoff_score: Option<f32>,
    /// A limit VERIFIED to serve the target by rerunning the real
    /// serve computation, pool caps included; `None` when the target
    /// ranks nowhere (or no limit up to the ranking's size reaches it).
    pub(crate) limit_to_reach: Option<usize>,
}

/// The vector lane's side of an explanation: why it did not run, or
/// what it saw when it did.
#[cfg_attr(test, derive(Debug))]
pub(crate) enum VectorLaneReport {
    /// `TAGURU_EMBED_PASSAGES` is off, or no provider is configured —
    /// `provider_configured` says which.
    Off { provider_configured: bool },
    /// The lane should have run; the provider refused the query.
    QueryEmbeddingFailed(String),
    /// Nothing embedded yet (no sidecar rows).
    NoVectors,
    /// The sidecar's rows belong to another model; they are never
    /// served, and the next refresh discards and re-embeds them.
    ModelChanged { stored: String, current: String },
    /// The sweep ran under `floor`; `cosine` is the target's best
    /// current-text row (text or doc2query question), `None` when it
    /// has none — not yet embedded, or embedded before its last edit.
    Ran { floor: f32, cosine: Option<f32> },
}

/// Why the vector lane can or cannot sweep an entry's paragraphs —
/// search takes the `Ready` arm and silently skips the rest; explain
/// names them in its [`VectorLaneReport`] (which carries the
/// provider-configured distinction itself).
enum PassageVectorGate {
    Disabled,
    Empty,
    ModelChanged { stored: String, current: String },
    Ready(Arc<PassageVectorStore>),
}

/// How many candidates the semantic resolve tier serves — the cap
/// [`AppState::semantic_resolve`] truncates to, and the one its
/// explain twin reports when a floor-passing name still missed.
pub(crate) const SEMANTIC_RESOLVE_LIMIT: usize = 5;

/// The gloss lane's side of a resolve explanation:
/// [`VectorLaneReport`]'s shape for the resolve tiers, told apart
/// where `semantic_resolve` deliberately folds — provider off, model
/// changed, and nothing embedded all answer the same empty list there.
#[cfg_attr(test, derive(Debug))]
pub(crate) enum GlossLaneReport {
    /// No embedding provider is configured.
    Off,
    /// The gloss sidecar's vectors belong to another model; the next
    /// refresh discards and re-embeds them.
    ModelChanged { stored: String, current: String },
    /// The namespace holds no gloss vectors yet — no refresh has run
    /// (or none since this context gained its vocabulary).
    EmptyTable,
    /// The provider refused the cue.
    QueryEmbeddingFailed(String),
    /// The sweep could run: the floor in effect, the expected name's
    /// own gloss cosine (`None` when it has no vector yet — added
    /// after the last refresh), its 1-based rank among the `passing`
    /// floor-clearing candidates, and the tier's serving `cap`.
    Ran {
        floor: f32,
        cosine: Option<f32>,
        rank: Option<usize>,
        passing: usize,
        cap: usize,
    },
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
    pub embed_parallel: usize,
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
            embed_parallel: 1,
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
            wal_enabled: crate::env_bool("TAGURU_WAL", true),
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
            embed_passages: crate::env_bool("TAGURU_EMBED_PASSAGES", false),
            passage_vector_limit: crate::env_number(
                "TAGURU_PASSAGE_VECTOR_LIMIT",
                DEFAULT_PASSAGE_VECTOR_LIMIT,
            ),
            // Worker threads dispatching each 128-item embedding chunk to
            // the provider concurrently; 1 keeps the old strictly-
            // sequential behavior. Raise to match the provider's rate
            // limit, not the machine's core count.
            embed_parallel: crate::env_number("TAGURU_EMBED_PARALLEL", 1),
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
                embed_parallel: self.embed_parallel,
                default_semantic_floor: self.semantic_floor,
            },
        )
    }
}

/// Shared server state: the data directory, the cache budget, and the
/// context registry.
#[derive(Clone)]
pub struct AppState(Arc<StateInner>);

/// Holds the server open-for-maintenance for as long as it lives —
/// acquired through [`AppState::try_enter_maintenance`], released by
/// `Drop` on every exit path (normal return, an early `?`, or a panic
/// unwind) so a wedged sweep can never leave the flag stuck.
pub struct MaintenanceGuard(AppState);

impl Drop for MaintenanceGuard {
    fn drop(&mut self) {
        self.0.metrics().exit_maintenance();
    }
}

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

/// Names with a reservation in flight for `create`, `delete`, or
/// `rename_context` — held under ONE mutex so that checking all three
/// sets and reserving a name in one of them happen as a single
/// critical section. Splitting these into three independent mutexes
/// let a `create`'s check-then-insert interleave with a concurrent
/// `rename_context`'s own check-then-insert (each only ever took
/// `registry.read()`, which does not exclude the other): both could
/// observe the other's set as still empty and both proceed to reserve
/// the same name, racing their disk writes. `delete` needs no such
/// care against the other two — it holds `registry.write()` across
/// its whole check-then-reserve phase, which already excludes any
/// concurrent `create`/`rename_context` (both readers).
#[derive(Default)]
struct PendingNames {
    /// Names whose delete is still removing files. A delete takes the
    /// name out of the registry FIRST and only then (unlocked) unlinks
    /// the file family — without this set, a create() in that window
    /// would lay down a new generation for the tail of the delete's
    /// unlink loop to destroy. Entered in the same critical section
    /// that removes the name, left when the files are gone.
    deletes: HashSet<String>,
    /// Names whose create is still writing files — the create-side twin
    /// of `deletes`. A create reserves the name here FIRST and only
    /// then (unlocked) clears leftovers and fsyncs the fresh file
    /// family; without this set the registry lock would have to stay
    /// held across that disk work, stalling every operation on every
    /// context behind one create's fsyncs. Entered under the registry
    /// guard, left in the critical section that registers the entry.
    creates: HashSet<String>,
    /// Both the `from` and `to` names of an in-flight rename — reserved
    /// before the marker is written, released only once the rename's
    /// last step (the group membership rewrite) lands. `create` and
    /// `delete` both refuse a name reserved here: a create under `to`
    /// would collide with the files about to land there, and a delete
    /// of either name would race the move or strand the marker.
    renames: HashSet<String>,
}

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
    /// BTreeMap keeps the directory listing (and `directory_page`'s
    /// keyset seek) in name order for free — the same reason `groups`
    /// below does.
    registry: RwLock<BTreeMap<String, Arc<Entry>>>,
    /// Groups: bundles of context names and child-group names (a
    /// shallow DAG, at most [`groups::MAX_GROUP_DEPTH`] groups tall and
    /// never cyclic, each set at most [`groups::MAX_GROUP_MEMBERS`]
    /// names), each persisted as one `{stem}.group` file. Small enough
    /// to stay resident in full, so one lock over the whole map
    /// suffices — and it is held across the record's own fsync on
    /// writes. That fsync therefore stalls group READS too (the
    /// directory, and any search that names a group) — briefly, and
    /// only when a group write is in flight — but never a context
    /// operation: a request that names no group never touches this
    /// lock (the cross-context searches skip group resolution for an
    /// empty list). BTreeMap keeps the directory listing in name order
    /// for free.
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
    /// Worker threads dispatching each 128-item embedding chunk to the
    /// provider concurrently (`TAGURU_EMBED_PARALLEL`, default 1 = the
    /// old strictly-sequential behavior). Raise to match the provider's
    /// rate limit, not the machine's core count — ureq's calls are
    /// synchronous, so this is the only lever for provider-side
    /// concurrency.
    embed_parallel: usize,
    /// See [`PendingNames`] for why `create`, `delete`, and
    /// `rename_context` share one mutex here instead of one each.
    pending: Mutex<PendingNames>,
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

/// An LRU-bounded map of cue → embedding: an LLM client repeats query
/// wording, and recency (not insertion order) is what predicts the next
/// hit. At the cap it holds ~12 MB of vectors. Recency is tracked by a
/// counter dedicated to this cache rather than `AppState::clock`
/// (documented for a different purpose) to keep the two concerns apart.
#[derive(Default)]
struct CueCache {
    vectors: HashMap<String, (Arc<Vec<f32>>, u64)>,
    tick: u64,
}

impl CueCache {
    const CAP: usize = 1024;

    fn get(&mut self, cue: &str) -> Option<Arc<Vec<f32>>> {
        self.tick += 1;
        let tick = self.tick;
        let entry = self.vectors.get_mut(cue)?;
        entry.1 = tick;
        Some(Arc::clone(&entry.0))
    }

    fn insert(&mut self, cue: String, vector: Arc<Vec<f32>>) {
        if self.vectors.contains_key(&cue) {
            return;
        }
        if self.vectors.len() >= Self::CAP
            && let Some(oldest) = self
                .vectors
                .iter()
                .min_by_key(|(_, (_, tick))| *tick)
                .map(|(cue, _)| cue.clone())
        {
            self.vectors.remove(&oldest);
        }
        self.tick += 1;
        self.vectors.insert(cue, (vector, self.tick));
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
        let (registry, resumed_context_renames) = scan_data_dir(&data_dir)?;
        // Groups scan after contexts (the context scan also sweeps
        // staging leftovers). Both scans finish moving any in-flight
        // rename's files, and hand back the (from, to) pairs whose
        // marker survived; rewrite group membership for those FIRST —
        // before reconcile, which has no notion of a rename in flight
        // and would see `from` as a plain dangling reference (nothing
        // registered under that name any more) and drop it instead of
        // carrying it to `to`. Each rewrite persists immediately (it
        // cannot rely on reconcile's own before/after diff, which
        // would see no further change to make and skip the write), so
        // the marker is safe to remove right after.
        let (mut groups, resumed_group_renames) = groups::scan_groups(&data_dir)?;
        // Rewrite membership only once the destination's pivot has
        // landed (else there is no `to` to point at, and `from` still
        // holds the files); remove the marker only once the move is
        // complete (else a straggler still needs the next boot to
        // retry). See `ResumedRename` for why these must not be one
        // condition.
        for rename in &resumed_context_renames {
            if rename.landed {
                rename_in_membership(&data_dir, &mut groups, &rename.from, &rename.to, |record| {
                    &mut record.contexts
                });
            }
            if rename.complete {
                let _ = fs::remove_file(renaming_marker_path(&data_dir, &file_stem(&rename.from)));
            }
        }
        for rename in &resumed_group_renames {
            if rename.landed {
                rename_in_membership(&data_dir, &mut groups, &rename.from, &rename.to, |record| {
                    &mut record.groups
                });
            }
            if rename.complete {
                let _ = fs::remove_file(groups::group_renaming_marker_path(
                    &data_dir,
                    &file_stem(&rename.from),
                ));
            }
        }
        // Reconcile unconditionally: whatever put a dangling member, a
        // dangling child, or an illegal nesting into a group file — a
        // crash between a deletion and the sweep's rewrite, a sweep
        // that could not persist, a hand-edited directory — boot drops
        // it and writes the fix back, so "a group names only live
        // contexts and live groups, acyclically, within the depth cap"
        // holds from the first request on, without exception.
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
            embed_parallel: options.embed_parallel,
            pending: Mutex::new(PendingNames::default()),
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
            .filter(|(_, entry)| entry.inner.read().meta.pinned)
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
                        let Some((name, entry)) = queue.lock().next() else {
                            break;
                        };
                        let mut inner = entry.inner.write();
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

    /// Attempts to become the sole maintenance sweep; `None` means one is
    /// already running. The returned guard is what keeps the server
    /// closed to ordinary traffic — dropping it (by any means) reopens.
    pub fn try_enter_maintenance(&self) -> Option<MaintenanceGuard> {
        self.0
            .metrics
            .try_enter_maintenance()
            .then(|| MaintenanceGuard(self.clone()))
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
        let mut dead_edges_total = 0u64;
        let mut dead_attributions_total = 0u64;
        let mut arena_slack_total = 0u64;
        let mut unsourced_edges_total = 0u64;
        let mut unsourced_weight_total = 0.0f64;
        for (_name, entry) in snapshot {
            let inner = entry.inner.read();
            if let Slot::Hot(context) = &inner.slot {
                contexts_resident += 1;
                resident_bytes += context.footprint() as u64;
                dead_edges_total += context.dead_edges() as u64;
                dead_attributions_total += context.dead_attributions() as u64;
                arena_slack_total += context.arena_slack() as u64;
                let (unsourced_edges, unsourced_weight) = context.unsourced_summary();
                unsourced_edges_total += unsourced_edges as u64;
                unsourced_weight_total += unsourced_weight;
            } else {
                // Cold (or mid-delete): the graph is not in memory, so
                // this stays whatever `inner.stats` last cached at
                // eviction/compact/flush — the same staleness trade
                // `resident_bytes`'s hot-only branch already accepts,
                // just from the other side (dead weight persists on
                // disk whether or not the context is loaded).
                dead_edges_total += inner.stats.dead_edges as u64;
                dead_attributions_total += inner.stats.dead_attributions as u64;
                arena_slack_total += inner.stats.arena_slack as u64;
                unsourced_edges_total += inner.stats.unsourced_edges as u64;
                unsourced_weight_total += inner.stats.unsourced_weight;
            }
            wal_bytes += inner.wal_bytes;
            let cold_passages_wal_bytes = inner.passages_wal_bytes;
            drop(inner);
            resident_bytes += entry.vectors_footprint() as u64;
            resident_bytes += entry.passages_footprint() as u64;
            resident_bytes += entry.bm25_footprint() as u64;
            resident_bytes += entry.passage_vectors_footprint() as u64;
            // A resident store knows its pending log; a cold one uses
            // the value `evict_entry` cached on the way down — the
            // gauge must not go blind just because a context was
            // evicted, nor re-`stat` the log on every scrape.
            passages_wal_bytes += entry
                .passages
                .lock()
                .as_ref()
                .map(|store| store.pending_log_bytes())
                .unwrap_or(cold_passages_wal_bytes);
        }
        GaugeSnapshot {
            contexts_registered,
            groups_registered: self.0.groups.read().len() as u64,
            contexts_resident,
            resident_bytes,
            wal_bytes,
            passages_wal_bytes,
            dead_edges_total,
            dead_attributions_total,
            arena_slack_total,
            unsourced_edges_total,
            unsourced_weight_total,
        }
    }

    pub fn context_count(&self) -> usize {
        self.0.registry.read().len()
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
    /// stall). The name is reserved in `pending.creates` under the
    /// registry guard, the files are written unlocked, and the entry
    /// lands in a second critical section — the create twin of
    /// delete's `pending.deletes` choreography.
    pub fn create(&self, name: &str, meta: ContextMeta) -> Result<(), CreateError> {
        // An empty name has no file stem — it would persist as a bare
        // `.ctx` and disappear from the registry on the next restart.
        // Refuse it at the lowest boundary, so no entrance (import,
        // direct call) can conjure a self-erasing context.
        if name.is_empty() {
            return Err(CreateError::InvalidName);
        }
        {
            let registry = self.0.registry.read();
            // A name mid-delete is still taken: its delete has left the
            // registry but is still unlinking files, and a create landing
            // now would have its fresh generation destroyed by the tail of
            // that loop. A name mid-create is equally taken. A name that
            // is either end of an in-flight rename is taken too — `to`
            // because a create now would collide with the files the
            // rename is about to land there, `from` because the rename
            // has not yet torn its files down. The client sees the same
            // refusal as for a live name and simply retries after the
            // other call's response.
            if registry.contains_key(name) {
                return Err(CreateError::AlreadyExists);
            }
            // Checking the other two sets and reserving this one all
            // happen under the SAME lock, in one critical section — see
            // `PendingNames`'s doc for why that atomicity is what closes
            // the gap against a concurrent `rename_context` (the only
            // sibling that, like this call, holds only `registry.read()`
            // for its own check-then-reserve).
            let mut pending = self.0.pending.lock();
            if pending.deletes.contains(name)
                || pending.renames.contains(name)
                || !pending.creates.insert(name.to_string())
            {
                return Err(CreateError::AlreadyExists);
            }
        }
        let created = self.create_files(name, &meta);
        // Success or failure, the reservation leaves in the same
        // critical section that (on success) makes the entry visible.
        let mut registry = self.0.registry.write();
        let outcome = created.map(|(stats, usage, context)| {
            registry.insert(
                name.to_string(),
                Arc::new(Entry::new(
                    meta,
                    stats,
                    Slot::Hot(Box::new(context)),
                    0,
                    0,
                    usage,
                )),
            );
        });
        self.0.pending.lock().creates.remove(name);
        outcome
    }

    /// The disk half of [`AppState::create`], run WITHOUT the registry
    /// lock — the `pending.creates` reservation is what keeps the name
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
            // The same hazard for a rename that half-finished with THIS
            // name as its SOURCE: its `.renaming` marker sits at this
            // stem, and boot's resume-sweep would otherwise move the
            // generation we are about to write onto the rename's
            // destination stem, losing it silently.
            renaming_marker_path(&self.0.data_dir, &stem),
        ] {
            if let Err(error) = remove_persisted_file(&stale)
                && error.kind() != io::ErrorKind::NotFound
            {
                return Err(CreateError::Io(error));
            }
        }
        // A rename that half-finished with THIS name as its DESTINATION
        // left its marker under the SOURCE's stem — a stem we cannot
        // derive from `name`. Boot's resume-sweep would move that source
        // family onto the generation we are about to write, erasing it.
        // Scan for any marker that names us as `to` and drop it; a fresh
        // create abandons a stuck rename either way.
        for stale in rename_markers_targeting(&self.0.data_dir, name, "renaming") {
            if let Err(error) = remove_persisted_file(&stale)
                && error.kind() != io::ErrorKind::NotFound
            {
                return Err(CreateError::Io(error));
            }
        }
        // Stale import markers are part of the same earlier generation:
        // left beside the new files, boot would report the fresh
        // context as carrying a torn import it never ran.
        for stale in import_marker_paths(&self.0.data_dir, &stem) {
            if let Err(error) = remove_persisted_file(&stale)
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
    /// The name enters `pending.deletes` in the same critical section
    /// that unregisters it and leaves only after the unlink loop: to a
    /// concurrent create() the name stays taken for the delete's whole
    /// run, so no new generation of files can appear under the tail of
    /// this one's removals.
    pub fn delete(&self, name: &str) -> Option<io::Result<()>> {
        let entry = {
            let mut registry = self.0.registry.write();
            if !registry.contains_key(name) {
                return None;
            }
            // A name mid-rename is refused rather than torn down: its
            // marker durably promises a move-then-membership-rewrite,
            // and a delete winning the race here would either destroy
            // the files the rename is about to move (as `from`) or the
            // files it just landed (as `to`), leaving the marker to
            // resume a rename with nothing left to finish at boot.
            // Reported through the same `Option<io::Result<()>>` a
            // live name already uses — the caller sees a name that
            // exists but cannot be deleted right now, not "no such
            // context".
            if self.0.pending.lock().renames.contains(name) {
                return Some(Err(io::Error::other(format!(
                    "context '{name}' is mid-rename; retry after it completes"
                ))));
            }
            let entry = registry.remove(name)?;
            self.0.pending.lock().deletes.insert(name.to_string());
            entry
        };
        let mut in_flight = entry.inner.write();
        self.tombstone_locked(&mut in_flight, &entry);
        // The rest of this function is disk I/O (marker, group sweep,
        // unlinks) guarded by `pending.deletes`, not by `inner` — hold
        // it no longer than the in-memory teardown above needs.
        drop(in_flight);
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
            if let Err(error) = remove_persisted_file(self.0.data_dir.join(file))
                && error.kind() != io::ErrorKind::NotFound
            {
                outcome = Err(error);
            }
        }
        // Import markers go with the family: deletion makes any
        // half-applied batch moot, and a survivor would have boot
        // report a tear in a context that no longer exists. Same
        // failure handling as the fixed files — a miss keeps the
        // `.deleted` marker, and boot finishes the job.
        for path in import_marker_paths(&self.0.data_dir, &stem) {
            if let Err(error) = remove_persisted_file(&path)
                && error.kind() != io::ErrorKind::NotFound
            {
                outcome = Err(error);
            }
        }
        if outcome.is_ok() {
            let _ = remove_persisted_file(&marker);
        }
        self.0.pending.lock().deletes.remove(name);
        Some(outcome)
    }

    /// Renames a context: its whole file family moves under the new
    /// name and group membership follows, while the OLD name becomes a
    /// tombstone exactly as `delete` leaves one — so a flusher's or
    /// evictor's handle cloned before the rename backs off instead of
    /// recreating files a name no longer owns.
    ///
    /// Unlike `delete`, a rename must not discard unflushed writes: the
    /// entry's whole current state is drained to disk under the OLD
    /// name, under one lock, before the tombstone lands (see
    /// `drain_entry_for_rename`) — so no racing write can land in the
    /// gap between "durably saved" and "this entry stops accepting
    /// writes" and be silently lost the way `delete` allows.
    ///
    /// The marker (`renaming_marker_path`) is written and durable
    /// BEFORE anything else moves, and only removed after the group
    /// membership rewrite lands — stricter than `delete`'s best-effort
    /// marker, because a rename whose files moved but whose group
    /// membership rewrite did not would otherwise have boot's
    /// `reconcile_groups` see the old name as a dangling reference and
    /// silently drop it, rather than resuming the rewrite.
    pub fn rename_context(&self, from: &str, to: &str) -> Result<(), RenameContextError> {
        if to.is_empty() {
            return Err(RenameContextError::InvalidName);
        }
        if from == to {
            return Ok(());
        }
        let entry = {
            let registry = self.0.registry.read();
            let Some(entry) = registry.get(from) else {
                return Err(RenameContextError::NotFound);
            };
            if registry.contains_key(to) {
                return Err(RenameContextError::AlreadyExists);
            }
            // Checking all three sets and reserving both names in
            // `renames` all happen under the SAME lock, in one critical
            // section — see `PendingNames`'s doc for why that atomicity
            // is what closes the gap against a concurrent `create` (the
            // only sibling that, like this call, holds only
            // `registry.read()` for its own check-then-reserve).
            let mut pending = self.0.pending.lock();
            if pending.deletes.contains(from)
                || pending.deletes.contains(to)
                || pending.creates.contains(to)
                || pending.renames.contains(from)
                || pending.renames.contains(to)
            {
                return Err(RenameContextError::Busy);
            }
            pending.renames.insert(from.to_string());
            pending.renames.insert(to.to_string());
            Arc::clone(entry)
        };
        match self.rename_context_locked(from, to, &entry) {
            RenameOutcome::Ok => {
                let mut pending = self.0.pending.lock();
                pending.renames.remove(from);
                pending.renames.remove(to);
                Ok(())
            }
            // Rolled back before the point of no return: the registry
            // and the marker are both back to their pre-call state, so
            // both names are genuinely free again.
            RenameOutcome::RolledBack(error) => {
                let mut pending = self.0.pending.lock();
                pending.renames.remove(from);
                pending.renames.remove(to);
                Err(error)
            }
            // Failed AT or AFTER the point of no return: `from` is
            // already gone from the registry, but its files (and the
            // durable `.renaming` marker) may still be sitting there
            // half-moved. `from` MUST stay reserved — releasing it
            // would let a client's create(from) sweep away the marker
            // and the old generation's files as ordinary "stale
            // leftovers" (see create_files), destroying them beyond any
            // recovery. Only a boot resume-sweep can finish or roll
            // this back, so the reservation outlives this call; `to`
            // was never touched and is safe to free now.
            RenameOutcome::Stuck(error) => {
                tracing::error!(
                    from = %from, to = %to, ?error,
                    "context rename failed after the point of no return; the \
                     source name stays reserved until the next restart resumes \
                     it from the .renaming marker"
                );
                self.0.pending.lock().renames.remove(to);
                Err(error)
            }
        }
    }

    /// The disk-and-registry half of [`AppState::rename_context`], run
    /// with `from` and `to` both reserved in `pending.renames` — see
    /// that function's doc for why the marker is strict rather than
    /// best-effort.
    ///
    /// The return type spells out what the caller may safely release on
    /// failure: [`RenameOutcome::RolledBack`] means the attempt never
    /// passed the point of no return (the registry still lists `from`,
    /// and any marker written was cleaned up), so both names are free
    /// again. [`RenameOutcome::Stuck`] means it failed after `from` was
    /// already removed from the registry — the marker survives on disk
    /// and only a boot resume-sweep (or a successful retry) can resolve
    /// it, so `from` must stay reserved in the meantime.
    fn rename_context_locked(&self, from: &str, to: &str, entry: &Arc<Entry>) -> RenameOutcome {
        let from_stem = file_stem(from);
        let to_stem = file_stem(to);
        let marker = renaming_marker_path(&self.0.data_dir, &from_stem);
        if let Err(error) = write_rename_marker(&marker, from, to) {
            return RenameOutcome::RolledBack(RenameContextError::Io(error));
        }
        if let Err(error) = self.drain_entry_for_rename(from, entry) {
            let _ = fs::remove_file(&marker);
            return RenameOutcome::RolledBack(RenameContextError::Io(error));
        }
        self.0.registry.write().remove(from);
        // POINT OF NO RETURN: memory already reflects the rename (the
        // tombstone under `from`). Every failure from here on is
        // reported as `Stuck` — see this function's doc — so the only
        // way back is finishing the move and the membership rewrite, at
        // boot if not now.
        if let Err(error) = move_context_files(&self.0.data_dir, &from_stem, &to_stem) {
            return RenameOutcome::Stuck(RenameContextError::Io(error));
        }
        let MetaFile { meta, stats, usage } = read_meta_file(&self.0.data_dir, &to_stem);
        let pinned = meta.pinned;
        let wal_bytes = fs::metadata(wal_path(&self.0.data_dir, &to_stem))
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let passages_wal_bytes = fs::metadata(passages_wal_path(&self.0.data_dir, &to_stem))
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let new_entry = Arc::new(Entry::new(
            meta,
            stats,
            Slot::Cold,
            wal_bytes,
            passages_wal_bytes,
            usage,
        ));
        self.0
            .registry
            .write()
            .insert(to.to_string(), Arc::clone(&new_entry));
        if pinned {
            let mut inner = new_entry.inner.write();
            match ensure_hot(&self.0.data_dir, to, &mut inner, &self.0.metrics) {
                Ok(()) => self.recount_entry(&mut inner),
                Err(error) => {
                    tracing::warn!(context = %to, %error, "renamed context not preloaded; it stays cold until first use");
                }
            }
        }
        {
            let mut groups = self.0.groups.write();
            rename_in_membership(&self.0.data_dir, &mut groups, from, to, |record| {
                &mut record.contexts
            });
        }
        let _ = fs::remove_file(&marker);
        RenameOutcome::Ok
    }

    /// Writes an entry's whole current state to disk under `name` —
    /// its image (if Hot), sidecar, and stats — then tombstones the
    /// slot, all under one lock: no write racing the rename can land
    /// in the gap between "durably saved" and "this entry stops
    /// accepting writes" and be silently discarded. `delete`'s
    /// in-memory teardown discards unflushed writes on purpose; a
    /// rename must carry them to the new name instead — that is the
    /// one difference from `delete`'s teardown below.
    ///
    /// Derived indexes (passages, BM25, paragraph vectors) are cleared
    /// resident-only, exactly as `delete` clears them: their sidecars
    /// already hold their own last-saved state on disk and move with
    /// the rest of the file family, so at most a not-yet-persisted
    /// refresh is lost — a rename does not owe them the graph's
    /// durability guarantee.
    fn drain_entry_for_rename(&self, name: &str, entry: &Entry) -> io::Result<()> {
        let mut inner = entry.inner.write();
        // Read everything `save_files` and the watermark need before
        // borrowing `inner.slot` mutably below — `EntryInner` sits
        // behind a lock guard, so the borrow checker cannot see the
        // two borrows as disjoint fields the way it would on a bare
        // struct.
        let watermark = inner.wal_seq.saturating_sub(1);
        let meta = inner.meta.clone();
        let usage = entry.usage.snapshot();
        if let Slot::Hot(context) = &mut inner.slot {
            // `ensure_hot`'s replay only applies WAL entries past
            // `applied_seq`, so baking in this watermark before saving
            // the image means the log — which rides along unmodified
            // under the new name — replays as a no-op once the file
            // family moves.
            context.set_applied_seq(watermark);
            let stats = ContextStats::of(context);
            save_files(&self.0.data_dir, name, &meta, &stats, &usage, context)?;
            inner.stats = stats;
        }
        self.tombstone_locked(&mut inner, entry);
        entry.usage_dirty.store(false, Ordering::Relaxed);
        drop(inner);
        Ok(())
    }

    /// Tombstones a write-locked entry: marks the slot `Deleted`,
    /// recounts its (now zero) footprint, clears the dirty flag, and
    /// drops every derived index (passages, BM25, paragraph vectors,
    /// term vectors) resident-only — their sidecars keep the
    /// last-saved state on disk, so at most a not-yet-persisted
    /// refresh is lost. The in-memory teardown `delete` and
    /// `drain_entry_for_rename` both need before their disk halves
    /// diverge (discard the state vs. carry it to a new name).
    ///
    /// Lock order: `inner` before `passages` before `vectors`, as
    /// documented on Entry — the caller holds `inner` across this
    /// call.
    fn tombstone_locked(&self, inner: &mut EntryInner, entry: &Entry) {
        inner.slot = Slot::Deleted;
        self.recount_entry(inner);
        entry.dirty.store(false, Ordering::Relaxed);
        *entry.passages.lock() = None;
        *entry.bm25.write() = None;
        *entry.passage_vectors.lock() = None;
        *entry.vectors.lock() = None;
    }

    /// Registers a group and persists it immediately — the create twin
    /// for groups, without the `pending.creates` choreography: the one
    /// fsync happens under the groups lock, which blocks only other
    /// group writes (see the field's doc), so nothing here needs the
    /// reservation dance.
    ///
    /// Member validation happens under both locks (`groups` before
    /// `registry` — the documented order): `contains_key` already
    /// answers false for a name mid-delete, because delete() removes
    /// the name and reserves it in `pending.deletes` inside one
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
        let mut groups = self.0.groups.write();
        if groups.contains_key(name) {
            return Err(CreateGroupError::AlreadyExists);
        }
        // Unreachable through HTTP while the request-list cap equals
        // [`groups::MAX_GROUP_MEMBERS`] (a create can't say more names
        // than a group may hold), but the invariant belongs to the
        // registry, not to whoever happens to call it.
        check_member_caps(&contexts, &children).map_err(CreateGroupError::OverCap)?;
        {
            let registry = self.0.registry.read();
            if let Some(missing) =
                first_missing(&contexts, |context| registry.contains_key(context))
            {
                return Err(CreateGroupError::NoSuchContext(missing.clone()));
            }
        }
        if let Some(missing) = first_missing(&children, |child| groups.contains_key(child)) {
            return Err(CreateGroupError::NoSuchGroup(missing.clone()));
        }
        // A group rename that half-finished under THIS name — as the
        // source (its `.grouprenaming` marker sits at this stem) or the
        // destination (some other stem's marker names it) — would
        // otherwise have boot's resume-sweep move a stale group file over
        // the one we are about to write. A create is a clean start:
        // abandon any such marker. Groups clear their marker on a
        // graceful move failure, so this only bites after a crash or a
        // best-effort boot cleanup that could not remove it — cheap
        // insurance keeping the group path symmetric with create_files.
        let mut stale_markers = rename_markers_targeting(&self.0.data_dir, name, "grouprenaming");
        stale_markers.push(groups::group_renaming_marker_path(
            &self.0.data_dir,
            &file_stem(name),
        ));
        for marker in stale_markers {
            if let Err(error) = remove_persisted_file(&marker)
                && error.kind() != io::ErrorKind::NotFound
            {
                return Err(CreateGroupError::Io(error));
            }
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
        let mut groups = self.0.groups.write();
        if !groups.contains_key(name) {
            return Err(UpdateGroupError::NotFound);
        }
        // The cap is judged on the delta's RESULT — removals first,
        // exactly as they will apply — and before the existence
        // lookups: counting needs none, and refusing here lets the
        // membership tests speak in names that need not exist.
        {
            let record = &groups[name];
            for (field, current, add, remove) in [
                (
                    "member contexts",
                    &record.contexts,
                    &add_contexts,
                    &remove_contexts,
                ),
                ("child groups", &record.groups, &add_groups, &remove_groups),
            ] {
                // An empty delta cannot move a set past the cap the
                // invariant already holds it under — skip the rebuild.
                if add.is_empty() && remove.is_empty() {
                    continue;
                }
                let mut prospective: BTreeSet<&str> = current.iter().map(String::as_str).collect();
                for removed in remove {
                    prospective.remove(removed.as_str());
                }
                prospective.extend(add.iter().map(String::as_str));
                if prospective.len() > groups::MAX_GROUP_MEMBERS {
                    return Err(UpdateGroupError::OverCap(field));
                }
            }
        }
        if !add_contexts.is_empty() {
            let registry = self.0.registry.read();
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

    /// Restores a set of group records — import's create-or-replace
    /// twin of [`Self::create_group`]/[`Self::update_group`]: each
    /// record replaces its whole row (description and both member
    /// sets), and a group absent from the set is untouched, parents
    /// naming a restored group included. The SET is judged before the
    /// first write, under the groups write lock: every member context
    /// registered, every child registered or in the set itself, both
    /// caps per record, and the nesting that results — the standing
    /// map overlaid with every record — acyclic and within
    /// [`groups::MAX_GROUP_DEPTH`]. A validation refusal applies
    /// nothing.
    ///
    /// Writes land children-first (depth order over the prospective
    /// map), so a persist failure partway strands no record dangling
    /// on an absent child; memory tracks exactly what persisted. A
    /// record equal to the standing row skips its write, so
    /// re-importing a stream converges to a no-op instead of a
    /// directory-wide fsync storm.
    pub fn restore_groups(
        &self,
        records: &[(String, GroupRecord)],
        deadline: Deadline,
    ) -> Result<Vec<(String, GroupRestoreOutcome)>, RestoreGroupsError> {
        let mut groups = self.0.groups.write();
        // The prospective map — what memory becomes if every write
        // lands — is what the validators judge.
        let mut prospective = groups.clone();
        let mut incoming: BTreeSet<&str> = BTreeSet::new();
        for (name, record) in records {
            if name.is_empty() {
                return Err(RestoreGroupsError::InvalidName);
            }
            if !incoming.insert(name) {
                return Err(RestoreGroupsError::Duplicate(name.clone()));
            }
            check_member_caps(&record.contexts, &record.groups).map_err(|field| {
                RestoreGroupsError::OverCap {
                    group: name.clone(),
                    field,
                }
            })?;
            prospective.insert(name.clone(), record.clone());
        }
        {
            // Lock order: `groups` before `registry`, as documented on
            // the field.
            let registry = self.0.registry.read();
            for (name, record) in records {
                if let Some(missing) =
                    first_missing(&record.contexts, |context| registry.contains_key(context))
                {
                    return Err(RestoreGroupsError::NoSuchContext {
                        group: name.clone(),
                        context: missing.clone(),
                    });
                }
                if let Some(missing) =
                    first_missing(&record.groups, |child| prospective.contains_key(child))
                {
                    return Err(RestoreGroupsError::NoSuchChild {
                        group: name.clone(),
                        child: missing.clone(),
                    });
                }
            }
        }
        let depths = groups::nesting_depths(&prospective).map_err(RestoreGroupsError::Nesting)?;
        let outcomes: Vec<(String, GroupRestoreOutcome)> = records
            .iter()
            .map(|(name, record)| {
                let outcome = match groups.get(name) {
                    Some(standing) if standing == record => GroupRestoreOutcome::Unchanged,
                    Some(_) => GroupRestoreOutcome::Replaced,
                    None => GroupRestoreOutcome::Created,
                };
                (name.clone(), outcome)
            })
            .collect();
        let mut order: Vec<usize> = (0..records.len()).collect();
        // Every record's name sits in `prospective`, so the settled
        // map has its depth — indexing panics loudly if that invariant
        // ever breaks, where a fallback would silently misorder.
        order.sort_by_key(|&index| depths[records[index].0.as_str()]);
        let mut applied = 0usize;
        for &index in &order {
            // Bound the fsync-per-record storm to the request budget. A
            // stream of many tiny group records would otherwise pin
            // `groups.write()` through one fsync each, freezing every
            // group op long past the deadline the batch loop honors.
            // What landed stands (children-first order, so it never
            // dangles); re-POSTing the whole stream is exact.
            if deadline.expired() {
                return Err(RestoreGroupsError::Timeout { applied });
            }
            let (name, record) = &records[index];
            if outcomes[index].1 == GroupRestoreOutcome::Unchanged {
                // Already standing in the desired state — it counts as
                // landed for the Io/Timeout report below.
                applied += 1;
                continue;
            }
            if let Err(error) = groups::write_group(&self.0.data_dir, &file_stem(name), record) {
                return Err(RestoreGroupsError::Io {
                    group: name.clone(),
                    applied,
                    error,
                });
            }
            groups.insert(name.clone(), record.clone());
            applied += 1;
        }
        Ok(outcomes)
    }

    /// Every context a scoped key must hold to restore `records`: the
    /// closures of the named groups over the STANDING map (what the
    /// replace would release) unioned with their closures over the
    /// prospective one (what it would address). The import gate judges
    /// group records with this — [`Self::group_context_closures`]'
    /// twin for the restore path, where children may be names the set
    /// itself brings.
    pub fn group_restore_involves(&self, records: &[(String, GroupRecord)]) -> BTreeSet<String> {
        let groups = self.0.groups.read();
        let roots: Vec<&str> = records.iter().map(|(name, _)| name.as_str()).collect();
        let mut involved = groups::context_closure(&groups, roots.iter().copied());
        let mut prospective = groups.clone();
        for (name, record) in records {
            prospective.insert(name.clone(), record.clone());
        }
        involved.extend(groups::context_closure(&prospective, roots.iter().copied()));
        involved
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
        let mut groups = self.0.groups.write();
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

    /// Renames a group: its file moves to the new name and every OTHER
    /// group's `groups` field naming it is rewritten to match.
    /// `groups.write()` covers the whole operation, so — unlike a
    /// context rename — no separate reservation is needed: no
    /// concurrent create, update, delete, or rename can observe a
    /// half-renamed state, only wait behind this lock.
    ///
    /// The marker is written and durable BEFORE the file moves, for
    /// the same reason as [`AppState::rename_context`]'s: a crash
    /// between the file move and the membership rewrite must not have
    /// boot's `reconcile_groups` see the old name as a dangling
    /// reference and drop it, rather than resuming the rewrite.
    pub fn rename_group(&self, from: &str, to: &str) -> Result<(), RenameGroupError> {
        if to.is_empty() {
            return Err(RenameGroupError::InvalidName);
        }
        if from == to {
            return Ok(());
        }
        let mut groups = self.0.groups.write();
        if !groups.contains_key(from) {
            return Err(RenameGroupError::NotFound);
        }
        if groups.contains_key(to) {
            return Err(RenameGroupError::AlreadyExists);
        }
        let from_stem = file_stem(from);
        let to_stem = file_stem(to);
        let marker = groups::group_renaming_marker_path(&self.0.data_dir, &from_stem);
        write_rename_marker(&marker, from, to).map_err(RenameGroupError::Io)?;
        if let Err(error) = commit_staged(
            &groups::group_path(&self.0.data_dir, &from_stem),
            &groups::group_path(&self.0.data_dir, &to_stem),
        ) {
            let _ = fs::remove_file(&marker);
            return Err(RenameGroupError::Io(error));
        }
        let record = groups.remove(from).expect("checked contains_key above");
        groups.insert(to.to_string(), record);
        rename_in_membership(&self.0.data_dir, &mut groups, from, to, |record| {
            &mut record.groups
        });
        let _ = fs::remove_file(&marker);
        Ok(())
    }

    /// One group's record by name, or `None` for an unknown group.
    pub fn group(&self, name: &str) -> Option<GroupRecord> {
        self.0.groups.read().get(name).cloned()
    }

    /// Union of every context reachable from the named groups — direct
    /// members plus everything nested children bundle, transitively.
    /// The scoped write gate judges a group by what it ADDRESSES, so
    /// this is its view; unknown names contribute nothing.
    pub fn group_context_closures<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
    ) -> BTreeSet<String> {
        groups::context_closure(&self.0.groups.read(), names)
    }

    /// [`group_context_closures`] with existence semantics: the first
    /// name that is not a registered group comes back as the error
    /// instead of contributing nothing. The cross-context searches
    /// resolve their `groups` targets here — a caller who NAMES a group
    /// deserves a `no_group` refusal, not a silently empty search —
    /// checked and walked under one lock acquisition so a concurrent
    /// group delete cannot slip between the two.
    pub fn resolve_groups(&self, names: &[String]) -> Result<BTreeSet<String>, String> {
        let groups = self.0.groups.read();
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

        let groups = self.0.groups.read();
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
        let mut groups = self.0.groups.write();
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
        let mut slot = entry.passages.lock();
        if let Some(store) = slot.as_ref() {
            return Ok(Arc::clone(store));
        }
        // A store whose last load failed is quarantined exactly like a
        // graph image (`ensure_hot`): answer the remembered refusal
        // while it is fresh instead of re-reading a broken snapshot on
        // every passage request.
        {
            let failure = entry.passages_load_failure.lock();
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
        // and the unverified-record count are for read-only inspection,
        // so drop them here.
        let (store, _torn, _unchecked) = crate::passages::PassageStore::load(
            passages_path(&self.0.data_dir, stem),
            &sources_path(&self.0.data_dir, stem),
            passages_wal_path(&self.0.data_dir, stem),
            self.0.passages_wal_max_bytes,
            true,
        )
        .inspect_err(|error| {
            *entry.passages_load_failure.lock() =
                Some((std::time::Instant::now(), error.to_string()));
        })?;
        *entry.passages_load_failure.lock() = None;
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
        deadline: Deadline,
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
                if deadline.expired() {
                    *skipped = Some(
                        "意味的検出は期限切れのため途中で打ち切り (一部の結果のみ)".to_string(),
                    );
                    break;
                }
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
            // read_context never consults a deadline itself — the
            // caller checks its own budget before calling in —
            // unreachable in practice, kept for exhaustiveness.
            Err(AccessError::DeadlineExceeded) => {
                skipped = Some("関連ペアの除外はスキップ (期限切れ)".to_string());
            }
        }
        Some((concepts, labels, skipped))
    }

    /// Withdraws one association from a context outright — the surgical
    /// correction for a single fact that should never have been
    /// asserted, where [`AppState::retract_source`] would discard the
    /// whole document's contribution. Returns how many attributions
    /// were unlinked, or `None` when the triple names no live edge
    /// (nothing was changed — the caller answers honestly instead of
    /// pretending a write happened).
    pub fn retract_association(
        &self,
        name: &str,
        subject: &str,
        label: &str,
        object: &str,
    ) -> Result<Option<usize>, AccessError> {
        let op = WalOp::RetractAssociation {
            subject: subject.to_string(),
            label: label.to_string(),
            object: object.to_string(),
        };
        self.logged_write(
            name,
            std::slice::from_ref(&op),
            |context| context.retract_association(subject, label, object),
            // The single RetractAssociation op never fails to apply.
            |_| 1,
        )
    }

    /// The read-only twin of [`Self::retract_source`]'s edge count —
    /// `POST /import?dry_run=true`'s preview of what a retraction would
    /// report, without unlinking anything.
    pub fn count_source_edges(&self, name: &str, source: &str) -> Result<usize, AccessError> {
        self.read_context(name, |context| context.count_source_edges(source))
    }

    /// Withdraws one source from a context — its graph contributions and
    /// its registered passage — the per-document differential-sync move:
    /// retract the old version of a changed document, then re-ingest the
    /// new one, instead of rebuilding the whole context. Returns how
    /// many associations were touched and whether a passage was removed.
    ///
    /// Brackets [`Self::retract_source_unmarked`]'s two independently
    /// durable writes (the graph's own WAL, then the passage store's)
    /// with the same batch-open marker `apply_batch` uses: a crash
    /// between them would otherwise leave the graph durably retracted
    /// while the passage text survives on disk, undetected by boot or
    /// `taguru inspect` — the same hazard the marker already closes for
    /// a whole import batch, at the smaller two-write scale of a
    /// standalone retraction. `apply_batch` calls
    /// [`Self::retract_source_unmarked`] directly instead of this
    /// method: its own marker already brackets that call along with the
    /// store/associate/alias steps that follow it, and clearing the
    /// marker here too would reopen the batch to the exact gap it
    /// exists to close.
    pub fn retract_source(&self, name: &str, source: &str) -> Result<(usize, bool), AccessError> {
        self.open_import_marker(name, source).map_err(|error| {
            AccessError::Unpersisted(format!(
                "import marker not persisted: {error} — nothing was retracted"
            ))
        })?;
        let (touched, passage_removed, passage_removal_errored) =
            self.retract_source_unmarked(name, source)?;
        // A genuine passage-store failure must leave the marker in
        // place: clearing it here would erase the only surviving
        // witness (surfaced by boot and `taguru inspect`) that this
        // source's truth is now half-applied — the graph side already
        // retracted, the passage still sitting on disk. "Nothing was
        // there to remove" (raced with a delete, or never had a
        // passage) is not this case and still clears normally.
        if !passage_removal_errored {
            self.clear_import_marker(name, source);
        }
        Ok((touched, passage_removed))
    }

    /// The marker-less core of [`Self::retract_source`] — see there for
    /// behavior and for why only `apply_batch` should call this
    /// directly. The third element of the returned tuple is `true`
    /// only when the passage store's own removal genuinely errored
    /// (store unavailable, or its `retract` call failed) — as opposed
    /// to `false`/`false`, which also covers "there was nothing to
    /// remove." `apply_batch` ignores it: its own `store_passages` call
    /// right after overwrites whatever stale passage a failed
    /// retraction left behind, so the failure there is self-healing.
    /// [`Self::retract_source`] is the one caller that cannot heal it
    /// the same way and uses it to decide whether clearing its marker
    /// is safe.
    pub(crate) fn retract_source_unmarked(
        &self,
        name: &str,
        source: &str,
    ) -> Result<(usize, bool, bool), AccessError> {
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
            return Ok((touched, false, false));
        };
        let Some(_fence) = entry.read_unless_deleted() else {
            // Same race, one step later: the delete beat us to the lock.
            return Ok((touched, false, false));
        };
        // The graph retraction above already succeeded; a passage-side
        // failure must not turn it into an error, only into an honest
        // `passage_removed: false` — paired with a `true` third element
        // so a marker-clearing caller can still tell "nothing to
        // remove" and "removal genuinely failed" apart.
        let (passage_removed, passage_removal_errored) =
            match self.entry_passages(&entry, &file_stem(name)) {
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
                        (removed, false)
                    }
                    Err(error) => {
                        tracing::warn!("passage for '{source}' not removed from disk: {error}");
                        (false, true)
                    }
                },
                Err(error) => {
                    tracing::warn!("passages for '{name}' unavailable during retract: {error}");
                    (false, true)
                }
            };
        Ok((touched, passage_removed, passage_removal_errored))
    }

    /// Opens the batch-open marker for one source's import — see
    /// [`import_marker_path`] for what it means while it exists. Called
    /// by `apply_batch` before the batch's first mutation, and by
    /// [`Self::retract_source`] before its own two-write sequence; an
    /// error refuses the operation, because proceeding would silently
    /// reintroduce the undetectable-tear gap the marker exists to close
    /// (and a disk that cannot land a hundred-byte marker is not going
    /// to land the writes either). `write_atomic` makes it durable,
    /// directory entry included, before any tracked write can need it.
    pub fn open_import_marker(&self, context: &str, source: &str) -> io::Result<()> {
        let marker = ImportMarker {
            context: context.to_string(),
            source: source.to_string(),
        };
        let body = serde_json::to_vec(&marker).map_err(io::Error::from)?;
        write_atomic(
            &import_marker_path(&self.0.data_dir, &file_stem(context), source),
            &body,
        )
    }

    /// Removes one source's batch-open marker: the batch completed, or
    /// the operator repaired the tear by retracting the source outright
    /// (either way the source's truth is consistent again). Best
    /// effort, loudly: a marker that cannot be removed only means boot
    /// keeps reporting a tear that is no longer one, until a re-import
    /// or a hand unlink clears it.
    pub fn clear_import_marker(&self, context: &str, source: &str) {
        let path = import_marker_path(&self.0.data_dir, &file_stem(context), source);
        if let Err(error) = remove_persisted_file(&path)
            && error.kind() != io::ErrorKind::NotFound
        {
            tracing::warn!(
                context,
                source,
                %error,
                "import marker not removed; boot will keep reporting this batch as torn",
            );
        }
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
        deadline: Deadline,
    ) -> Option<io::Result<Vec<PassageSearchHit>>> {
        let entry = self.lookup(name)?;
        if limit == 0 {
            return entry.read_unless_deleted().map(|_| Ok(Vec::new()));
        }
        let query_grams = deduped_query_grams(query);
        if query_grams.is_empty() {
            return entry.read_unless_deleted().map(|_| Ok(Vec::new()));
        }
        let pool = lane_pool(limit);

        // The semantic lane's query embedding runs BEFORE any lock: a
        // provider round trip must never extend the fence below.
        let cue = match self.passage_query_cue(query, deadline) {
            Ok(cue) => cue,
            Err(error) => {
                // Degrade, loudly: the lexical lane still answers.
                tracing::warn!(
                    context = %name,
                    error,
                    "passage query embedding failed; serving the lexical lane alone"
                );
                None
            }
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

        if self
            .ensure_bm25_index(&entry, &store, name, deadline)
            .is_err()
        {
            return Some(Err(io::Error::other(DeadlineExceeded)));
        }
        let lexical = {
            let guard = entry.bm25.read();
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
            Some(cue) => match self.passage_vector_gate(&entry, &file_stem(name)) {
                PassageVectorGate::Ready(vectors) => semantic_lane_hits(
                    vectors.top_matches(cue, pool, deadline),
                    self.effective_semantic_floor(&fence.meta),
                ),
                _ => Vec::new(),
            },
            None => Vec::new(),
        };

        Some(Ok(fuse_passage_lanes(&store, lexical, semantic, limit)))
    }

    /// The whole account of one source (or one of its paragraphs)
    /// against one query — the same search re-run with nothing
    /// truncated, the target located in it, and each lane's reason
    /// when it has no evidence. `paragraph: None` settles on the
    /// source's best showing: its best-ranked paragraph, or the one
    /// sharing the most query terms when nothing ranked. `limit` is
    /// the search call being explained — the serve/cutoff boundary is
    /// recomputed exactly as `search_passages(limit)` computes it,
    /// pool caps included. Read-only, and bounded like one normal
    /// query plus one targeted scoring.
    pub fn explain_passage_search(
        &self,
        name: &str,
        query: &str,
        source: &str,
        paragraph: Option<u32>,
        limit: usize,
        deadline: Deadline,
    ) -> Option<io::Result<PassageExplainLookup>> {
        let entry = self.lookup(name)?;
        let query_terms = deduped_spelled_query_terms(query);
        let query_grams: Vec<u64> = query_terms.iter().map(|&(_, gram)| gram).collect();

        // Mirror search_passages: the embedding runs only when a search
        // would have reached it, and before the fence.
        let cue = if query_grams.is_empty() {
            Ok(None)
        } else {
            self.passage_query_cue(query, deadline)
        };

        let fence = entry.read_unless_deleted()?;
        let store = match self.entry_passages(&entry, &file_stem(name)) {
            Ok(store) => store,
            Err(error) => return Some(Err(error)),
        };

        let Some(record) = store.get(source) else {
            return Some(Ok(PassageExplainLookup::UnknownSource));
        };
        let paragraphs = record.paragraphs.len();
        if let Some(index) = paragraph
            && (index as usize) >= paragraphs
        {
            return Some(Ok(PassageExplainLookup::IndexOutOfRange { paragraphs }));
        }
        if paragraphs == 0 {
            return Some(Ok(PassageExplainLookup::IndexOutOfRange { paragraphs }));
        }
        if query_grams.is_empty() {
            return Some(Ok(PassageExplainLookup::NoQueryTerms));
        }

        if self
            .ensure_bm25_index(&entry, &store, name, deadline)
            .is_err()
        {
            return Some(Err(io::Error::other(DeadlineExceeded)));
        }
        let guard = entry.bm25.read();
        let index = guard.as_ref().expect("index was just built");

        // Sweep both lanes whole, once: a lane's pool cap is a prefix
        // of the same deterministically ordered sweep, so every pool
        // size fusion needs below is a `take` away.
        let lexical_full = index.search(&query_grams, usize::MAX);
        let floor = self.effective_semantic_floor(&fence.meta);
        let gate = match &cue {
            Ok(Some(_)) => Some(self.passage_vector_gate(&entry, &file_stem(name))),
            _ => None,
        };
        let vector_rows: Vec<(String, u32, u64, f32)> = match (&cue, &gate) {
            (Ok(Some(cue)), Some(PassageVectorGate::Ready(vectors))) => vectors
                .top_matches(cue, usize::MAX, deadline)
                .into_iter()
                .map(|(key, score)| (key.source.clone(), key.index, key.hash, score))
                .collect(),
            _ => Vec::new(),
        };
        let lexical_lane = |pool: usize| -> Vec<crate::bm25::IndexHit> {
            lexical_full.iter().take(pool).cloned().collect()
        };
        // Pool first, floor second — the order search_passages applies
        // them in (`top_matches(cue, pool)` then the filter).
        let semantic_lane = |pool: usize| -> Vec<(String, u32, u64, f32)> {
            vector_rows
                .iter()
                .take(pool)
                .filter(|&&(.., score)| score >= floor)
                .cloned()
                .collect()
        };

        let full = fuse_passage_lanes(
            &store,
            lexical_lane(usize::MAX),
            semantic_lane(usize::MAX),
            usize::MAX,
        );
        // The served list exactly as `search_passages(limit)` builds it.
        let served_hits = fuse_passage_lanes(
            &store,
            lexical_lane(lane_pool(limit)),
            semantic_lane(lane_pool(limit)),
            limit,
        );

        let chosen = paragraph
            .or_else(|| {
                full.iter()
                    .find(|hit| hit.source == source)
                    .map(|hit| hit.index)
            })
            .unwrap_or_else(|| {
                // Nothing of the source ranked at all: the best showing
                // is the paragraph sharing the most query terms (the
                // first one, when they tie — including at zero).
                let mut best = (0u32, 0usize);
                for at in 0..paragraphs as u32 {
                    let shared = index.explain(&query_grams, source, at).map_or(0, |lex| {
                        lex.terms.iter().filter(|term| term.tf > 0.0).count()
                    });
                    if shared > best.1 {
                        best = (at, shared);
                    }
                }
                best.0
            });
        let (span, text) = record
            .paragraph(chosen as usize)
            .expect("the chosen paragraph is within the record");

        // The index answered for the text it saw; evidence about an
        // edited paragraph would explain the wrong bytes, so it is
        // withheld exactly like a stale search hit.
        let lexical = index
            .explain(&query_grams, source, chosen)
            .filter(|lex| lex.hash == span.hash);
        let overlap = lexical
            .as_ref()
            .is_some_and(|lex| lex.terms.iter().any(|term| term.tf > 0.0));
        let paragraph_terms = (!overlap).then(|| {
            // Both sides of the 表記ゆれ verdict on one table: the
            // paragraph's own spellings, questions included, deduped.
            let mut seen = std::collections::HashSet::new();
            let mut terms: Vec<String> = Vec::new();
            let mut take = |raw: &str| {
                for (spelling, gram) in spelled_passage_terms(raw) {
                    if seen.insert(gram) {
                        terms.push(spelling);
                    }
                }
            };
            take(text);
            for (at, question) in &record.questions {
                if *at == chosen {
                    take(question);
                }
            }
            terms
        });

        let vector = match (cue, gate) {
            (Err(error), _) => VectorLaneReport::QueryEmbeddingFailed(error),
            (Ok(None), _) | (Ok(Some(_)), Some(PassageVectorGate::Disabled)) => {
                VectorLaneReport::Off {
                    provider_configured: self.0.embedder.is_some(),
                }
            }
            (Ok(Some(_)), Some(PassageVectorGate::Empty)) => VectorLaneReport::NoVectors,
            (Ok(Some(_)), Some(PassageVectorGate::ModelChanged { stored, current })) => {
                VectorLaneReport::ModelChanged { stored, current }
            }
            (Ok(Some(_)), Some(PassageVectorGate::Ready(_))) => {
                // The target's best cosine across its rows (text row
                // and doc2query question rows alike), current-text rows
                // only — a stale row IS "not yet re-embedded". The
                // sweep is score-descending, so the first row is the
                // best one, floor or no floor.
                let cosine = vector_rows
                    .iter()
                    .find(|&&(ref row_source, row_index, hash, _)| {
                        row_source == source && row_index == chosen && hash == span.hash
                    })
                    .map(|&(.., score)| score);
                VectorLaneReport::Ran { floor, cosine }
            }
            (Ok(Some(_)), None) => unreachable!("the gate is read whenever a cue exists"),
        };

        let is_target = |hit: &PassageSearchHit| hit.source == source && hit.index == chosen;
        let rank = full.iter().position(is_target);
        let target = rank.map(|at| &full[at]);
        let served = served_hits.iter().any(is_target);

        // The smallest VERIFIED limit that serves the target: start at
        // its full-ranking rank (and low enough lane pools), rerun the
        // real serve computation, and grow on a miss — RRF against
        // capped pools can seat late double-lane candidates above a
        // mid-pool single-lane hit, so the unbounded rank alone is a
        // floor, not an answer.
        //
        // The starting pool must cover the WORSE of the two lane
        // ranks, not the better one: a dual-lane target's rerun score
        // only matches its full-ranking score once both lanes are
        // in-pool, and a pool sized off the better rank routinely
        // truncates away the worse lane's contribution, understating
        // the target and forcing extra doublings (or exhausting the
        // retry budget below) to reach a candidate this same target
        // would have cleared on the first try.
        let limit_to_reach = if served {
            Some(limit)
        } else {
            rank.map(|at| at + 1).and_then(|first| {
                let lane_need = target
                    .and_then(|hit| match (hit.bm25, hit.vector) {
                        (Some((bm25, _)), Some((vector, _))) => Some(bm25.max(vector)),
                        (Some((lane, _)), None) | (None, Some((lane, _))) => Some(lane),
                        (None, None) => None,
                    })
                    .map_or(1, |lane| lane.div_ceil(4));
                // `lane_need` sizes the pool a RAW lane rank would need,
                // unlike `first` it is not bounded by `full.len()` —
                // under heavy staleness (many raw lane hits filtered
                // out as stale before `full` was built) it can overshoot
                // past `full.len()` on this very first candidate. Clamp
                // rather than let that overshoot read as "unreachable":
                // `full.len()` is always a legal, still-untried candidate
                // (it's `first`'s own upper bound), and skipping it would
                // report a false negative for a limit that would have
                // served the target.
                let mut candidate = first.max(lane_need).min(full.len());
                for _ in 0..8 {
                    let rerun = fuse_passage_lanes(
                        &store,
                        lexical_lane(lane_pool(candidate)),
                        semantic_lane(lane_pool(candidate)),
                        candidate,
                    );
                    if rerun.iter().any(is_target) {
                        return Some(candidate);
                    }
                    if candidate >= full.len() {
                        return None;
                    }
                    candidate = (candidate.saturating_mul(2)).min(full.len());
                }
                None
            })
        };

        Some(Ok(PassageExplainLookup::Explained(Box::new(
            PassageSearchExplanation {
                paragraph: chosen,
                paragraphs,
                paragraph_named: paragraph.is_some(),
                query_terms,
                lexical,
                paragraph_terms,
                vector,
                fused: !semantic_lane(usize::MAX).is_empty(),
                ranked: full.len(),
                rank: rank.map(|at| at + 1),
                score: target.map(|hit| hit.score),
                bm25_lane: target.and_then(|hit| hit.bm25),
                vector_lane: target.and_then(|hit| hit.vector),
                limit,
                served,
                cutoff_score: served_hits.last().map(|hit| hit.score),
                limit_to_reach,
            },
        ))))
    }

    /// A resident BM25 index for this entry: build on the residency's
    /// first search, repair a drifted sidecar per source, rebuild when
    /// tombstones have piled up. Double-checked so concurrent first
    /// searches build once. Called under the entry's read fence with
    /// every passage-store lock released — the documented order for
    /// `Entry::bm25` (holding `bm25` while READING the store is fine,
    /// and is how the build works). `Err` when `deadline` expires
    /// before a needed rebuild starts; the index is left as it was.
    fn ensure_bm25_index(
        &self,
        entry: &Entry,
        store: &crate::passages::PassageStore,
        name: &str,
        deadline: Deadline,
    ) -> Result<(), DeadlineExceeded> {
        let stale = {
            let guard = entry.bm25.read();
            match &*guard {
                None => true,
                Some(index) => index.needs_reclaim(),
            }
        };
        if stale {
            let mut guard = entry.bm25.write();
            let rebuild = match &*guard {
                None => true,
                Some(index) => index.needs_reclaim(),
            };
            if rebuild {
                if deadline.expired() {
                    return Err(DeadlineExceeded);
                }
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
        Ok(())
    }

    /// The semantic lane's query embedding, run BEFORE any lock — a
    /// provider round trip must never extend an entry fence. `Ok(None)`
    /// when the lane is off; `Err` carries the provider's refusal for
    /// the caller to log (search) or report (explain).
    fn passage_query_cue(
        &self,
        query: &str,
        deadline: Deadline,
    ) -> Result<Option<Arc<Vec<f32>>>, String> {
        if !self.passage_embedding_enabled() {
            return Ok(None);
        }
        let embedder = self.0.embedder.clone().expect("enabled implies a provider");
        self.cue_vector(&*embedder, query, deadline).map(Some)
    }

    /// Why the vector lane can or cannot sweep this entry's paragraphs.
    /// Search takes the `Ready` arm and silently skips the rest;
    /// explain names them.
    fn passage_vector_gate(&self, entry: &Entry, stem: &str) -> PassageVectorGate {
        let Some(embedder) = self.0.embedder.as_ref().filter(|_| self.0.embed_passages) else {
            return PassageVectorGate::Disabled;
        };
        let vectors = self.entry_passage_vectors(entry, stem);
        if vectors.is_empty() {
            return PassageVectorGate::Empty;
        }
        if vectors.model != embedder.model() {
            return PassageVectorGate::ModelChanged {
                stored: vectors.model.clone(),
                current: embedder.model().to_string(),
            };
        }
        PassageVectorGate::Ready(vectors)
    }

    /// The floor the semantic lane drops cosine matches below — the
    /// same one `semantic_resolve` applies to its own matches; context
    /// setting beats the server default.
    fn effective_semantic_floor(&self, meta: &ContextMeta) -> f32 {
        meta.semantic_floor
            .unwrap_or(self.0.default_semantic_floor)
            .clamp(0.0, 1.0)
    }

    /// One provider round trip, timed into the embed-latency histogram
    /// whatever the outcome — the ok/failed counters cannot tell a
    /// slow provider from a down one; the histogram can.
    fn timed_embed(
        &self,
        embedder: &dyn EmbeddingProvider,
        texts: &[&str],
        purpose: EmbedPurpose,
        deadline: Deadline,
    ) -> Result<Vec<Vec<f32>>, String> {
        let started = std::time::Instant::now();
        let outcome = embedder.embed(texts, purpose, deadline);
        self.0.metrics.record_embed_latency(started.elapsed());
        outcome
    }

    /// `timed_embed` for an [`EmbedPurpose::Index`] refresh call, plus
    /// the ok/failed refresh counters every refresh site needs recorded
    /// around it — the width probe below and each chunk dispatched by
    /// `embed_stale`/`refresh_passage_embeddings` all wrap this same
    /// pair.
    fn timed_embed_for_refresh(
        &self,
        embedder: &dyn EmbeddingProvider,
        texts: &[&str],
        deadline: Deadline,
    ) -> Result<Vec<Vec<f32>>, String> {
        match self.timed_embed(embedder, texts, EmbedPurpose::Index, deadline) {
            Ok(vectors) => {
                self.0.metrics.record_embed_refresh(true);
                Ok(vectors)
            }
            Err(error) => {
                self.0.metrics.record_embed_refresh(false);
                Err(error)
            }
        }
    }

    /// The query side of every embedding lookup: process cache first,
    /// provider (as [`EmbedPurpose::Query`]) on a miss. No lock is held
    /// across the provider call.
    fn cue_vector(
        &self,
        embedder: &dyn EmbeddingProvider,
        cue: &str,
        deadline: Deadline,
    ) -> Result<Arc<Vec<f32>>, String> {
        if let Some(vector) = self.0.cue_cache.lock().get(cue) {
            return Ok(vector);
        }
        match self.timed_embed(embedder, &[cue], EmbedPurpose::Query, deadline) {
            Ok(mut vectors) => {
                self.0.metrics.record_embed_resolve(true);
                let vector = Arc::new(vectors.pop().unwrap_or_default());
                self.0
                    .cue_cache
                    .lock()
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
        let mut cached = entry.passage_vectors.lock();
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
        let mut guard = entry.bm25.write();
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
            let guard = entry.bm25.read();
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
    pub fn refresh_embeddings(
        &self,
        name: &str,
        deadline: Deadline,
    ) -> Option<Result<(usize, usize), String>> {
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
        let _serial = entry.vectors_refresh.lock();
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
            // read_context never consults a deadline itself — the
            // caller checks its own budget before calling in —
            // unreachable in practice, kept for exhaustiveness.
            Err(AccessError::DeadlineExceeded) => {
                return Some(Err("request deadline exceeded".to_string()));
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
        //
        // Read through the memory cache, not straight off disk: a prior
        // refresh's save can fail after the provider already sold it
        // the rows (see the tail of this function), and the cache is
        // where those survive even though the sidecar does not. Empty
        // and disk agree whenever nothing failed, so this changes
        // nothing on the common path.
        let existing = self.entry_vectors(&entry, &file_stem(name));
        // Claim the save-pending flag up front: it only ever reflects a
        // prior pass's save failure (this pass owns the whole write
        // side via `_serial`, so nothing else can set it mid-flight),
        // and tells this pass to retry the write below even if its own
        // diff buys nothing new.
        let was_pending = entry.vectors_save_pending.swap(false, Ordering::Relaxed);
        let mut fresh_model = existing.model != embedder.model();
        let (mut embedded_concepts, concept_failure) = self.embed_stale(
            &*embedder,
            &existing.concepts,
            &concepts,
            fresh_model,
            deadline,
        );
        let (mut embedded_labels, label_failure) =
            self.embed_stale(&*embedder, &existing.labels, &labels, fresh_model, deadline);
        // Persist whatever either table bought even when the other fails:
        // losing already-billed vectors to a sibling's provider error is
        // the bug this mirrors from the passage refresh. A partial failure
        // does skip the width probe just below — spending more provider
        // budget on a pass that already reports Err and gets retried buys
        // nothing — but not the carried-vs-fresh reconciliation after it:
        // that one decides whether what already landed this pass is fit to
        // persist at all.
        let mut failure = concept_failure.or(label_failure);
        // The model NAME is the staleness discriminator, but a provider
        // can change output width behind a stable name (a backend swap
        // behind the same proxy or gateway). Old-width rows carried next
        // to new-width ones would feed `similarity` mismatched
        // dimensions — no error, no score — so a width disagreement
        // stales the whole table, exactly as if the model were renamed.
        // Concepts and labels are sampled and compared independently —
        // collapsing to "whichever table is non-empty, concepts first"
        // would miss a width change confined to whichever table that
        // fallback didn't happen to sample.
        let width = |table: &VectorTable| table.values().map(|(_, vector)| vector.len()).next();
        let carried_concepts_width = width(&existing.concepts);
        let carried_labels_width = width(&existing.labels);
        let mut fresh_width = width(&embedded_concepts).or_else(|| width(&embedded_labels));
        // Unchanged hashes embed nothing, which would leave the width
        // change of exactly this scenario — backend swap, no gloss
        // edits — undetectable forever. One probe embedding per no-op
        // refresh keeps that from hiding.
        if failure.is_none()
            && !fresh_model
            && (carried_concepts_width.is_some() || carried_labels_width.is_some())
            && fresh_width.is_none()
            && let Some((_, gloss)) = concepts.first().or_else(|| labels.first())
        {
            match self.timed_embed_for_refresh(embedder.as_ref(), &[gloss.as_str()], deadline) {
                Ok(vectors) => {
                    fresh_width = vectors.first().map(Vec::len);
                }
                Err(error) => {
                    failure = Some(error);
                }
            }
        }
        // Not gated on `failure.is_none()`: a sibling table's provider
        // error must not excuse persisting this pass's already-landed
        // vectors at a width that disagrees with what is carried —
        // that mismatch is decided below, then reconciled regardless of
        // what else failed.
        let width_mismatch = fresh_width.is_some_and(|fresh| {
            carried_concepts_width.is_some_and(|carried| carried != fresh)
                || carried_labels_width.is_some_and(|carried| carried != fresh)
        });
        if !fresh_model && width_mismatch {
            tracing::warn!(
                context = name,
                model = embedder.model(),
                carried_concepts = ?carried_concepts_width,
                carried_labels = ?carried_labels_width,
                fresh = fresh_width,
                "embedding width changed under an unchanged model name; re-embedding every gloss"
            );
            fresh_model = true;
            let (concepts_reembedded, concept_failure) =
                self.embed_stale(&*embedder, &existing.concepts, &concepts, true, deadline);
            embedded_concepts = concepts_reembedded;
            let (labels_reembedded, label_failure) =
                self.embed_stale(&*embedder, &existing.labels, &labels, true, deadline);
            embedded_labels = labels_reembedded;
            failure = concept_failure.or(label_failure);
        }
        let newly_embedded = embedded_concepts.len() + embedded_labels.len();

        // Publish under the entry's tombstone fence (a delete that may
        // have won it must not see its sidecar recreated) — `_serial`
        // above, held since before the gloss read, is what makes this
        // read-modify-write race-free, not this lock by itself. A SHARED
        // fence is enough: nothing below touches the entry's own data,
        // only `entry.vectors` (its own lock) and the sidecar file on
        // disk, so there is no reason to block concurrent graph reads for
        // the length of this save — the same trade `flush_bm25` makes
        // for its sidecar.
        let _fence = entry.read_unless_deleted()?;
        // Same basis `existing` was diffed against above, not a fresh
        // disk read: `_serial` has excluded every other refresh of
        // this context since before `existing` was taken, so nothing
        // could have changed the sidecar (or the cache backing it) in
        // between — re-reading here would only risk losing rows a
        // still-unpersisted prior pass bought and cached but a
        // straight disk read cannot see.
        //
        // `fresh_model` also covers the width change above: rows for
        // names that have since left the graph must not linger at the
        // old width either.
        let mut store = if fresh_model || existing.model != embedder.model() {
            VectorStore {
                model: embedder.model().to_string(),
                ..Default::default()
            }
        } else {
            (*existing).clone()
        };
        store.concepts.extend(embedded_concepts);
        store.labels.extend(embedded_labels);
        // Prune ghost rows: a name dropped by compaction leaves the live
        // gloss lists, so nothing above re-embeds or carries it, yet its
        // stored vector would linger here forever and
        // semantic_resolve/semantic_twins would keep surfacing a name the
        // graph no longer holds. A model/width wipe above already dropped
        // such rows wholesale; this covers ordinary retraction, the way
        // the passage refresh gets for free by rebuilding.
        let live_concepts: HashSet<&str> = concepts.iter().map(|(name, _)| name.as_str()).collect();
        let live_labels: HashSet<&str> = labels.iter().map(|(name, _)| name.as_str()).collect();
        let before_prune = store.concepts.len() + store.labels.len();
        store
            .concepts
            .retain(|name, _| live_concepts.contains(name.as_str()));
        store
            .labels
            .retain(|name, _| live_labels.contains(name.as_str()));
        let total = store.concepts.len() + store.labels.len();
        let pruned = before_prune - total;
        // `was_pending` covers a prior save that failed after already
        // buying rows: this pass's own diff can land on newly_embedded
        // == 0 and pruned == 0 (everything it needs is already carried
        // from `existing`, which reads the memory cache — see above)
        // while the disk image is still whatever the failed save left
        // it as. Without this, that state would never retry the write.
        if (newly_embedded > 0 || pruned > 0 || was_pending)
            && let Err(error) = store.save(&path)
        {
            entry.vectors_save_pending.store(true, Ordering::Relaxed);
            // The provider already sold `embedded_concepts`/
            // `embedded_labels`; cache the merged store anyway so the
            // next refresh's `existing` (read from this same cache,
            // not the disk — see above) does not buy them a second
            // time. Only the sidecar write failed, not this.
            *entry.vectors.lock() = Some(Arc::new(store));
            return Some(Err(format!("vector store not persisted: {error}")));
        }
        // Publish the fresh store so queries never re-read the sidecar.
        *entry.vectors.lock() = Some(Arc::new(store));
        // What landed is durable; a provider failure still returns Err so
        // the caller sees the pass was partial, and the stale rows it
        // skipped stay stale for the next refresh to retry.
        match failure {
            Some(error) => Some(Err(error)),
            None => Some(Ok((newly_embedded, total))),
        }
    }

    /// Diffs one gloss table against its stored vectors and embeds what
    /// is new or changed, 128 glosses per provider call. Each vector
    /// remembers the hash of the gloss it came from; `fresh_model`
    /// marks everything stale. Returns the vectors that landed alongside
    /// the first provider error, if any — the caller persists the former
    /// so a sibling table's failure never discards billed work, and the
    /// stale rows the error skipped stay stale for the next refresh to
    /// retry. Chunks dispatch concurrently, so a provider mid-migration
    /// can answer two chunks of the very same call with different
    /// widths; unlike `PassageVectorStore::push`, `VectorTable` has no
    /// dimension of its own to enforce, so a vector that disagrees with
    /// the width this batch already settled on is dropped here — loudly,
    /// and left stale for the next refresh — rather than merged into a
    /// table `similarity` would then silently stop matching against.
    fn embed_stale(
        &self,
        embedder: &dyn EmbeddingProvider,
        stored: &VectorTable,
        entries: &[(String, String)],
        fresh_model: bool,
        deadline: Deadline,
    ) -> (VectorTable, Option<String>) {
        let stale: Vec<(String, String, u64)> = entries
            .iter()
            .filter_map(|(name, gloss)| {
                let hash = fnv1a(gloss);
                let outdated =
                    fresh_model || stored.get(name).is_none_or(|&(hashed, _)| hashed != hash);
                outdated.then(|| (name.clone(), gloss.clone(), hash))
            })
            .collect();
        let stale_chunks: Vec<&[(String, String, u64)]> = stale.chunks(128).collect();
        let outcomes =
            dispatch_chunks_concurrently(&stale_chunks, self.0.embed_parallel, |chunk| {
                if deadline.expired() {
                    return Err(DeadlineExceeded.to_string());
                }
                let texts: Vec<&str> = chunk.iter().map(|(_, gloss, _)| gloss.as_str()).collect();
                self.timed_embed_for_refresh(embedder, &texts, deadline)
            });
        let mut embedded = VectorTable::new();
        let mut width: Option<usize> = None;
        let mut failure: Option<String> = None;
        for (chunk, outcome) in stale_chunks.iter().zip(outcomes) {
            match outcome {
                Some(Ok(vectors)) => {
                    for ((name, _, hash), vector) in chunk.iter().zip(vectors) {
                        let expected = *width.get_or_insert(vector.len());
                        if vector.len() != expected {
                            tracing::warn!(
                                name = name.as_str(),
                                expected,
                                got = vector.len(),
                                "dropping a gloss vector whose width disagrees with this \
                                 refresh's other chunks — a provider mid-migration; it stays \
                                 stale for the next refresh to retry"
                            );
                            continue;
                        }
                        embedded.insert(name.clone(), (*hash, vector));
                    }
                }
                // Keep the vectors that did land so the caller can persist
                // them; report the first error. Stale rows this failure
                // skipped stay stale in the diff for the next refresh.
                Some(Err(error)) => failure = failure.or(Some(error)),
                None => {}
            }
        }
        (embedded, failure)
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
        deadline: Deadline,
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
        let _serial = entry.passage_refresh.lock();
        // Claim the dirty flag up front: work that lands mid-refresh
        // re-marks it, so the ticker returns — never lost, never
        // double-claimed. The prior value still matters: besides a
        // fresh passage store/retract, it is also how a save that
        // failed after buying rows (see `changed` below) tells this
        // pass to retry the write even if its own diff finds nothing
        // new — otherwise a would-be-`changed: false` pass would never
        // flush the cache the failed save left behind onto disk.
        let was_dirty = entry.passages_embed_dirty.swap(false, Ordering::Relaxed);
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
        // Read through the memory cache, not straight off disk: a prior
        // refresh's save can fail after the provider already sold it
        // the rows (see the tail of this function), and the cache is
        // where those survive even though the sidecar does not. Empty
        // and disk agree whenever nothing failed, so this changes
        // nothing on the common path.
        let existing = self.entry_passage_vectors(&entry, &file_stem(name));
        let mut fresh_model = existing.model != embedder.model();
        // A provider can change output width behind a stable model name
        // (a backend swap behind the same proxy). Old-width rows carried
        // next to new-width ones would let PassageVectorStore::push drop
        // every new row this pass embeds — a stale store that also
        // over-reports what it stored — so a width disagreement stales
        // the whole table, exactly as a model rename does. Detected the
        // way the concept refresh detects it; the redo re-walks `records`
        // (still in scope) so it carries no extra memory. `dim` is
        // private, so the carried width is the first stored row's length.
        let carried_width = existing.iter().next().map(|(_, row)| row.len());
        let (fresh, embedded, skipped_over_limit, failure) = loop {
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
                    for (question_hash, row_text) in
                        std::iter::once((None, text)).chain(question_rows)
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
                        match carried.get(&(source.as_str(), span.index, span.hash, question_hash))
                        {
                            Some(row) => fresh.push(key, row.to_vec()),
                            None => to_embed.push((key, row_text.to_string())),
                        }
                    }
                }
            }

            let to_embed_chunks: Vec<&[(PassageKey, String)]> = to_embed.chunks(128).collect();
            let outcomes =
                dispatch_chunks_concurrently(&to_embed_chunks, self.0.embed_parallel, |chunk| {
                    if deadline.expired() {
                        return Err(DeadlineExceeded.to_string());
                    }
                    let texts: Vec<&str> = chunk.iter().map(|(_, text)| text.as_str()).collect();
                    self.timed_embed_for_refresh(embedder.as_ref(), &texts, deadline)
                });
            let mut embedded = 0usize;
            let mut failure: Option<String> = None;
            let mut fresh_width: Option<usize> = None;
            for (chunk, outcome) in to_embed_chunks.iter().zip(outcomes) {
                match outcome {
                    Some(Ok(vectors)) => {
                        for ((key, _), vector) in chunk.iter().zip(vectors) {
                            fresh_width.get_or_insert(vector.len());
                            // `push` silently drops a row whose width
                            // disagrees with the dimension `fresh` already
                            // settled on (the same provider-mid-migration
                            // hazard `embed_stale` guards against for
                            // glosses) — count only the rows that actually
                            // landed, or `embedded` over-reports what
                            // `total_rows` below can already prove didn't
                            // all land.
                            let before = fresh.len();
                            fresh.push(key.clone(), vector);
                            embedded += fresh.len() - before;
                        }
                    }
                    Some(Err(error)) => failure = failure.or(Some(error)),
                    None => {}
                }
            }
            // Unchanged hashes embed nothing, which would leave the width
            // change of exactly this scenario — backend swap, no passage
            // edits — undetectable. One probe embedding per no-op refresh
            // keeps it from hiding, matching the concept refresh.
            if failure.is_none()
                && !fresh_model
                && carried_width.is_some()
                && fresh_width.is_none()
                && let Some(probe) = records
                    .iter()
                    .flat_map(|(_, record)| record.paragraph_texts())
                    .map(|(_, text)| text)
                    .find(|text| !text.is_empty())
            {
                match self.timed_embed_for_refresh(embedder.as_ref(), &[probe], deadline) {
                    Ok(vectors) => fresh_width = vectors.first().map(Vec::len),
                    Err(error) => failure = Some(error),
                }
            }
            // Not gated on `failure.is_none()`: a chunk that failed must
            // not excuse persisting this pass's already-landed rows at a
            // width that disagrees with what is carried — that mismatch
            // is decided here and reconciled regardless of what else
            // failed.
            if !fresh_model
                && let (Some(carried_w), Some(fresh_w)) = (carried_width, fresh_width)
                && carried_w != fresh_w
            {
                tracing::warn!(
                    context = name,
                    model = embedder.model(),
                    carried = carried_w,
                    fresh = fresh_w,
                    "passage embedding width changed under an unchanged model name; re-embedding every passage"
                );
                fresh_model = true;
                continue;
            }
            break (fresh, embedded, skipped_over_limit, failure);
        };

        // Publish under the entry's tombstone fence (a delete that won
        // it must not see its files recreated), and only when something
        // changed — an all-carried refresh is a no-op, not a rewrite. A
        // SHARED fence, exactly like the read phase above: nothing here
        // touches the entry's own data, only `entry.passage_vectors`
        // (its own lock) and the sidecar file, so graph reads need not
        // block for the length of this save.
        // `was_dirty` covers a prior save that failed after already
        // buying rows: this pass's own diff can land on `changed:
        // false` (everything it needs is already carried from
        // `existing`, which reads the memory cache — see above) while
        // the disk image is still whatever the failed save left it
        // as. Without this, that state would never retry the write.
        let changed = embedded > 0
            || fresh.len() != existing.len()
            || (fresh_model && !fresh.is_empty())
            || was_dirty;
        let _fence = entry.read_unless_deleted()?;
        let total_rows = fresh.len();
        if changed && let Err(error) = fresh.save(&path) {
            entry.passages_embed_dirty.store(true, Ordering::Relaxed);
            // The provider already sold `embedded` of these rows; cache
            // them anyway so the next refresh's `existing` (read from
            // this same cache, not the disk — see above) does not buy
            // them a second time. Only the sidecar write failed, not
            // this.
            *entry.passage_vectors.lock() = Some(Arc::new(fresh));
            return Some(Err(format!("passage vectors not persisted: {error}")));
        }
        *entry.passage_vectors.lock() = Some(Arc::new(fresh));
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
        deadline: Deadline,
    ) -> Option<Result<Vec<(String, f32)>, String>> {
        let Some(embedder) = self.0.embedder.clone() else {
            return Some(Ok(Vec::new()));
        };
        let entry = self.lookup(name)?;
        // One-call override beats the context setting beats the server
        // default (see [`DEFAULT_SEMANTIC_FLOOR`] for the calibration).
        let context_floor = entry.inner.read().meta.semantic_floor;
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
        let cue_vector = match self.cue_vector(&*embedder, cue, deadline) {
            Ok(vector) => vector,
            Err(error) => return Some(Err(error)),
        };
        let mut scored: Vec<(String, f32)> = table
            .iter()
            .map(|(name, (_, vector))| (name.clone(), similarity(&cue_vector, vector)))
            .filter(|&(_, score)| score >= floor)
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(SEMANTIC_RESOLVE_LIMIT);
        Some(Ok(scored))
    }

    /// The gloss lane's account of one (cue, expected) pair — why
    /// [`AppState::semantic_resolve`] could not have surfaced the
    /// expected name (it folds provider-off, model-changed, and
    /// nothing-embedded into one empty answer; explain needs them
    /// apart), or exactly where it stood when the sweep could run:
    /// its own gloss cosine against the floor in effect, and its rank
    /// in the very ordering `semantic_resolve` truncates. `None` when
    /// the context does not exist.
    pub fn explain_semantic_resolve(
        &self,
        name: &str,
        cue: &str,
        expected: &str,
        labels: bool,
        floor_override: Option<f32>,
        deadline: Deadline,
    ) -> Option<GlossLaneReport> {
        let Some(embedder) = self.0.embedder.clone() else {
            return Some(GlossLaneReport::Off);
        };
        let entry = self.lookup(name)?;
        let context_floor = entry.inner.read().meta.semantic_floor;
        let floor = floor_override
            .or(context_floor)
            .unwrap_or(self.0.default_semantic_floor)
            .clamp(0.0, 1.0);
        let store = self.entry_vectors(&entry, &file_stem(name));
        // A never-refreshed sidecar is empty, whatever model string it
        // carries — report the missing refresh, not a model change.
        if store.concepts.is_empty() && store.labels.is_empty() {
            return Some(GlossLaneReport::EmptyTable);
        }
        if store.model != embedder.model() {
            return Some(GlossLaneReport::ModelChanged {
                stored: store.model.clone(),
                current: embedder.model().to_string(),
            });
        }
        let table = if labels {
            &store.labels
        } else {
            &store.concepts
        };
        if table.is_empty() {
            return Some(GlossLaneReport::EmptyTable);
        }
        let cue_vector = match self.cue_vector(&*embedder, cue, deadline) {
            Ok(vector) => vector,
            Err(error) => return Some(GlossLaneReport::QueryEmbeddingFailed(error)),
        };
        let cosine = table
            .get(expected)
            .map(|(_, vector)| similarity(&cue_vector, vector));
        // The expected name's 1-based rank in semantic_resolve's exact
        // ordering (cosine desc, name asc): candidates strictly ahead
        // of it, plus one. Counted, not sorted — one sweep.
        let mut passing = 0usize;
        let mut ahead = 0usize;
        for (candidate, (_, vector)) in table.iter() {
            let score = similarity(&cue_vector, vector);
            if score < floor {
                continue;
            }
            passing += 1;
            if let Some(cosine) = cosine
                && (score > cosine || (score == cosine && candidate.as_str() < expected))
            {
                ahead += 1;
            }
        }
        let rank = cosine.filter(|&cosine| cosine >= floor).map(|_| ahead + 1);
        Some(GlossLaneReport::Ran {
            floor,
            cosine,
            rank,
            passing,
            cap: SEMANTIC_RESOLVE_LIMIT,
        })
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
    /// residency, and stats, in name order. For large registries, prefer
    /// [`AppState::directory_page`], which seeks a page in O(log n + k)
    /// instead of describing every entry on every call.
    pub fn directory(&self) -> Vec<DirectoryEntry> {
        self.snapshot()
            .into_iter()
            .filter_map(|(name, entry)| describe_entry(name, &entry))
            .collect()
    }

    /// One name-ordered page of the routing directory plus the
    /// cursor-independent total, seeked in O(log n + k) against the
    /// `BTreeMap`-backed registry — the paged sibling of
    /// [`AppState::directory`]. Cuts the page under the registry lock
    /// (cloning only `Arc` handles, same as [`AppState::snapshot`]) and
    /// describes the survivors after dropping it: a context's
    /// `Entry::inner` lock must never be taken while the registry lock
    /// is held, the same ordering `directory`/`lookup` already keep.
    /// A page can come back shorter than `limit` if a context in it is
    /// deleted in the instant between the seek and [`describe_entry`]
    /// reading its slot — the same race `directory` already tolerates.
    /// If EVERY entry in the seek window loses that race, re-seek past
    /// it instead of reporting an empty page: callers (the SDKs' `iter`
    /// helpers among them) treat an empty page as the end of the
    /// directory, so returning one while later entries still exist
    /// would truncate the walk. Each retry's `after` strictly advances,
    /// so this terminates in at most `total` iterations.
    pub fn directory_page(
        &self,
        after: Option<&str>,
        limit: usize,
    ) -> (usize, Vec<DirectoryEntry>) {
        use std::ops::Bound;

        let mut after = after.map(str::to_string);
        loop {
            let (total, slice) = {
                let registry = self.0.registry.read();
                let start = match &after {
                    Some(after) => Bound::Excluded(after.as_str()),
                    None => Bound::Unbounded,
                };
                let slice: Vec<(String, Arc<Entry>)> = registry
                    .range::<str, _>((start, Bound::Unbounded))
                    .take(limit)
                    .map(|(name, entry)| (name.clone(), Arc::clone(entry)))
                    .collect();
                (registry.len(), slice)
            };
            let Some(last_seeked) = slice.last().map(|(name, _)| name.clone()) else {
                return (total, Vec::new());
            };
            let page: Vec<DirectoryEntry> = slice
                .into_iter()
                .filter_map(|(name, entry)| describe_entry(name, &entry))
                .collect();
            if !page.is_empty() {
                return (total, page);
            }
            after = Some(last_seeked);
        }
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
            let inner = entry.inner.read();
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
    pub fn export_context(
        &self,
        name: &str,
        deadline: Deadline,
    ) -> Result<crate::export::ExportSnapshot, AccessError> {
        let entry = self.lookup(name).ok_or(AccessError::NotFound)?;
        let stem = file_stem(name);
        // Fast path: already resident, shared lock (mirrors read_context).
        {
            let inner = entry.inner.read();
            match &inner.slot {
                Slot::Hot(context) => {
                    self.0.metrics.record_cache_hit();
                    let snapshot =
                        self.export_snapshot(&entry, &stem, &inner.meta, context, deadline);
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
            self.export_snapshot(&entry, &stem, &inner.meta, context, deadline)
        })?;
        self.touch(&entry);
        self.enforce_budget(name);
        Ok(snapshot)
    }

    /// The materialization inside [`AppState::export_context`]'s fence.
    /// Lock order: the caller holds `inner`; `entry_passages` takes
    /// `passages` — the documented `inner` → `passages` order.
    ///
    /// `deadline` is checked once, before any of it — not inside
    /// `context.query_any(&[], &[], &[])` below, which collects every
    /// association up front (its all-wildcard fast path), so a deadline
    /// that is already tight when this is called cannot shorten that
    /// initial O(edges) collection (the same limitation documented on
    /// [`Context::compacted`]).
    fn export_snapshot(
        &self,
        entry: &Entry,
        stem: &str,
        meta: &ContextMeta,
        context: &Context,
        deadline: Deadline,
    ) -> Result<crate::export::ExportSnapshot, AccessError> {
        if deadline.expired() {
            return Err(AccessError::DeadlineExceeded);
        }
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
    pub fn compact_context(
        &self,
        name: &str,
        deadline: Deadline,
    ) -> Result<CompactOutcome, AccessError> {
        let entry = self.lookup(name).ok_or(AccessError::NotFound)?;
        let outcome = offload(|| {
            let mut inner = entry.lock_unless_deleted().ok_or(AccessError::NotFound)?;
            ensure_hot(&self.0.data_dir, name, &mut inner, &self.0.metrics)
                .map_err(AccessError::Load)?;
            let Slot::Hot(context) = &inner.slot else {
                unreachable!("ensure_hot leaves the slot hot");
            };
            let bytes_before = context.footprint();
            let (mut fresh, stats) =
                context
                    .compacted(deadline)
                    .map_err(|failure| match failure {
                        CompactionError::Full(full) => {
                            AccessError::Load(format!("compaction refused: {full}"))
                        }
                        CompactionError::DeadlineExceeded => AccessError::DeadlineExceeded,
                    })?;
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

    /// Server-wide sweep: every context whose live dead ratio strictly
    /// exceeds `min_dead_ratio` is rebuilt via [`Self::compact_context`],
    /// worst ratio first, so a deadline that cuts the sweep short still
    /// recovers the most it could. Candidates are read from each entry's
    /// existing dead ratio ([`Context::dead_ratio`] while hot, the
    /// cached snapshot's while cold) — nothing is loaded or rebuilt just
    /// to be asked whether it qualifies. Sequential by design: the caller
    /// (`POST /maintenance/compact`) has already drained ordinary
    /// traffic before calling this, so there is no concurrency to hide
    /// behind parallelism, and one context at a time caps the sweep's
    /// peak memory at a single context's footprint.
    pub fn run_maintenance_compaction(
        &self,
        min_dead_ratio: f64,
        deadline: Deadline,
    ) -> MaintenanceCompactionOutcome {
        let mut candidates: Vec<(String, f64)> = self
            .snapshot()
            .into_iter()
            .filter_map(|(name, entry)| {
                let inner = entry.inner.read();
                let ratio = match &inner.slot {
                    Slot::Hot(context) => context.dead_ratio(),
                    Slot::Cold => inner.stats.dead_ratio(),
                    Slot::Deleted => return None,
                };
                // NaN (a malformed `min_dead_ratio`) makes every
                // comparison false, so the sweep simply selects nothing
                // rather than mis-selecting.
                (ratio > min_dead_ratio).then_some((name, ratio))
            })
            .collect();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut contexts = Vec::with_capacity(candidates.len());
        let mut deadline_exceeded = false;
        for (name, _) in candidates {
            if deadline.expired() {
                deadline_exceeded = true;
                break;
            }
            match self.compact_context(&name, deadline) {
                Ok(outcome) => contexts.push(MaintenanceCompactionEntry { name, outcome }),
                Err(AccessError::DeadlineExceeded) => {
                    deadline_exceeded = true;
                    break;
                }
                Err(error) => {
                    tracing::warn!(
                        context = %name,
                        ?error,
                        "maintenance sweep skipped a context"
                    );
                }
            }
        }
        MaintenanceCompactionOutcome {
            contexts,
            deadline_exceeded,
        }
    }

    /// Test-only: rewinds any remembered load failure (graph image and
    /// passage store both) so the quarantine window can elapse without
    /// the test sleeping through it.
    #[cfg(test)]
    pub fn age_load_failures(&self, name: &str, by: std::time::Duration) {
        let entry = self.lookup(name).expect("the context must be registered");
        if let Some((failed_at, _)) = &mut entry.inner.write().load_failure {
            *failed_at = failed_at
                .checked_sub(by)
                .expect("test ages within the Instant range");
        }
        if let Some((failed_at, _)) = &mut *entry.passages_load_failure.lock() {
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
            let mut guard = entry.inner.write();
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
            let _ = remove_persisted_file(&staged);
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
        // replaced, and stamp the entry's stats back to match. An
        // eviction that reloaded Hot again is the same story once more:
        // it bumps `image_generation` on its way to Cold specifically so
        // this catches that case too, the same as compaction's. Either
        // way, backing off here costs nothing — the next tick flushes
        // the current image instead.
        if !matches!(inner.slot, Slot::Hot(_)) || inner.image_generation != generation {
            let _ = remove_persisted_file(&staged);
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
                let _ = remove_persisted_file(&staged);
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
        self.0.registry.read().get(name).cloned()
    }

    fn snapshot(&self) -> Vec<(String, Arc<Entry>)> {
        self.0
            .registry
            .read()
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
            let result =
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operate(context))) {
                    Ok(result) => result,
                    Err(payload) => {
                        // The WAL append above already landed, but `operate`
                        // panicked before the in-memory apply it durably
                        // promises could finish, and before `dirty` below
                        // could cover it. Left Hot, this half-mutated
                        // Context would keep serving reads and accepting
                        // further writes forever — parking_lot doesn't
                        // poison. Forcing the slot back to Cold makes the
                        // next access rebuild Hot from the image plus a
                        // full WAL replay instead, which reapplies the very
                        // op that just panicked through the same validated
                        // path replay always uses. recount_entry reflects
                        // the entry's now-zero resident footprint right
                        // away, matching the promotion it counted above.
                        inner.slot = Slot::Cold;
                        self.recount_entry(&mut inner);
                        std::panic::resume_unwind(payload);
                    }
                };
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
        // restores the "acknowledged means replayable" contract. It can
        // come back false two ways: a real I/O failure (already logged
        // and counted by `record_flush`, which is what /health reads) or
        // a silent no-op racing a flush already in flight (claimed
        // `flushing` first) — that path never touches `record_flush`, so
        // /health stays green even though this write is, for the moment,
        // relying on the periodic flusher rather than the WAL to survive
        // a crash. `dirty` is already set (unconditionally, above)
        // either way, so the next tick still retries it — but a crash in
        // that window would replay a WAL that no longer matches memory,
        // so the gap is worth its own loud signal rather than blending
        // into an ordinary retry.
        if wal_behind && !self.flush_entry(name, &entry) {
            tracing::warn!(
                context = %name,
                "post-write recovery flush did not land immediately; the WAL for this \
                 write is no longer trustworthy and durability now depends on the next \
                 periodic flush landing before a crash"
            );
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
        deadline: Deadline,
    ) -> Result<Result<usize, PartialWrite>, AccessError> {
        if deadline.expired() {
            return Err(AccessError::DeadlineExceeded);
        }
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
        let mut cached = entry.vectors.lock();
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
            let inner = entry.inner.read();
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

    /// One entry's eviction: persist if dirty, drop the graph, clear
    /// the cached vectors. `false` means nothing was freed — the entry
    /// got pinned since the caller's sweep, its save failed (it stays
    /// resident rather than losing writes), or a concurrent eviction
    /// already cleared it. That last case matters: two budget sweeps
    /// snapshot the directory under a shared lock and can carry the
    /// same candidate, so the loser must report `false` or the caller
    /// subtracts its bytes from the residency estimate a second time.
    ///
    /// The common case persists BEFORE the entry's write lock below is
    /// taken, via [`Self::flush_bm25`] and [`Self::flush_entry`] —
    /// both already stage their disk work (serialize + fsync) with
    /// the lock released (see `flush_entry`'s doc comment) and re-take
    /// it only to publish. Calling them here, instead of this function
    /// doing its own lock-held save the way it once did unconditionally,
    /// means an eviction no longer stalls every reader and writer of
    /// the context for as long as the image takes to land — the same
    /// stall `flush_entry` was written to avoid in the first place.
    /// The lock-held save below still exists as the fallback for the
    /// rare case a rival flush is already mid-flight when this call
    /// starts: `flush_entry`'s own claim would just lose that race and
    /// no-op, and skipping the drop-to-Cold below in that case would
    /// mean the caller's eviction sweep might never make progress on a
    /// context under sustained write pressure.
    fn evict_entry(&self, name: &str, entry: &Entry) -> bool {
        self.flush_bm25(name, entry);
        self.flush_entry(name, entry);

        let mut guard = entry.inner.write();
        let inner = &mut *guard;
        // Re-check under the write lock; the entry may have changed
        // between the snapshot and now.
        if inner.meta.pinned {
            return false;
        }
        let mut freed = false;
        let watermark = inner.wal_seq - 1;
        if let Slot::Hot(context) = &mut inner.slot {
            // Still dirty/flushing after the attempt above: either a
            // rival flush was already mid-flight (its own claim swap
            // made `flush_entry` above a no-op) or `flush_entry` itself
            // lost a race and backed off. Either way the durable image
            // is stale or absent, so fall back to saving it here, under
            // the lock, same as this function always did — the rare
            // cost of a lock-held serialize+fsync only on the rare
            // path where flush and eviction land on the same entry at
            // the same instant.
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
            // Bump so a flush that staged its image before this
            // eviction can tell, once it re-locks, that the slot it
            // captured is gone — a later reload plus that flush's
            // re-publish would otherwise resurrect a stale image over
            // whatever was written after the reload.
            inner.image_generation += 1;
            // Local zero only: the caller's absolute store settles
            // the global, so a recount's delta would double-count.
            inner.counted_bytes = 0;
            self.0.metrics.record_eviction(true);
            freed = true;
        }
        drop(guard);

        // Dropping the passage store loses nothing (its log is fsynced
        // per batch); a best-effort compaction first just spares the
        // next load a replay. Failure changes neither.
        let compacted_wal_bytes = {
            let mut passages = entry.passages.lock();
            match passages.take() {
                Some(store) => {
                    if store.pending_log_bytes() > 0
                        && let Err(error) = store.compact()
                    {
                        tracing::warn!("passages for '{name}' evicted uncompacted: {error}");
                    }
                    Some(store.pending_log_bytes())
                }
                None => None,
            }
        };
        if let Some(bytes) = compacted_wal_bytes {
            freed = true;
            // Cold from here on: `gauge_snapshot` reads this cached
            // value instead of re-`stat`ing the log on every scrape.
            entry.inner.write().passages_wal_bytes = bytes;
        }
        // Same best-effort posture for a dirty index: saving it spares
        // the next residency a re-tokenization. `flush_bm25` above
        // already persisted it if it was dirty, so `bm25_dirty` is
        // normally already clear here and this is just a `take()`.
        {
            let mut bm25 = entry.bm25.write();
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
        if entry.passage_vectors.lock().take().is_some() {
            freed = true;
        }
        if entry.vectors.lock().take().is_some() {
            freed = true;
        }
        freed
    }
}

/// One directory row, or `None` when the entry was deleted between the
/// caller's snapshot/lookup and this lock.
fn describe_entry(name: String, entry: &Entry) -> Option<DirectoryEntry> {
    let inner = entry.inner.read();
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

/// Runs `f` over `items` on up to `workers` threads pulling from one
/// shared queue — the same divide-the-queue-not-the-slice shape
/// `preload_pinned` uses, generalized so a caller only supplies the
/// per-item work. Each worker collects into a local `Vec` and merges
/// into the shared result once at the end, so contention is limited to
/// the queue itself; results come back in arrival order, not input
/// order — callers that need input order carry an index through `T`/`R`
/// and sort afterward.
pub(crate) fn parallel_map<T, R>(items: Vec<T>, workers: usize, f: impl Fn(T) -> R + Sync) -> Vec<R>
where
    T: Send,
    R: Send,
{
    if items.is_empty() {
        return Vec::new();
    }
    let workers = workers.min(items.len()).max(1);
    let queue = Mutex::new(items.into_iter());
    let results: Mutex<Vec<R>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let Some(item) = queue.lock().next() else {
                        break;
                    };
                    local.push(f(item));
                }
                results.lock().extend(local);
            });
        }
    });
    results.into_inner()
}

/// Runs `f` over each of `chunks` on up to `workers` threads, claiming
/// indices in order. Unlike `parallel_map` above — arrival-order
/// results, no notion of failure — this preserves input order and
/// stops claiming new work once a chunk's failure has been recorded.
/// Every caller (`extract_chunks_concurrently` in src/extract.rs, and
/// `embed_stale` / `refresh_passage_embeddings` below) needs both: an
/// input-order-preserving result to fold correctly, and best-effort
/// early termination once a failure surfaces, so a batch that is going
/// to fail stops enlisting new work. Fold-on-failure semantics differ
/// per caller (fail the whole batch vs. keep whatever succeeded), so
/// the fold itself is left to them — this returns the raw, unfolded
/// per-index outcome.
///
/// `next` and `first_failure` are independent atomics; SeqCst on both
/// is required so a worker claiming an index past a just-recorded
/// failure actually observes it (Relaxed would silently reintroduce
/// unbounded over-dispatch past a failure). Every index at or below the
/// true minimum failing index is guaranteed a `Some` slot — a foldable
/// prefix callers can trust. Slots past it are best-effort: `None` if
/// never claimed, `Some` if a worker finished before the failure was
/// recorded. Their count is NOT bounded by `workers` — a failure slow
/// to surface lets the other workers complete arbitrarily many later
/// indices first — so callers fold on the prefix, never on a count of
/// what landed past the failure.
pub(crate) fn dispatch_chunks_concurrently<C: Sync, R: Send + Sync>(
    chunks: &[C],
    workers: usize,
    f: impl Fn(&C) -> Result<R, String> + Sync,
) -> Vec<Option<Result<R, String>>> {
    if chunks.is_empty() {
        return Vec::new();
    }
    let workers = workers.min(chunks.len()).max(1);
    let next = AtomicUsize::new(0);
    let first_failure = AtomicUsize::new(usize::MAX);
    let results: Vec<OnceLock<Result<R, String>>> =
        (0..chunks.len()).map(|_| OnceLock::new()).collect();

    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::SeqCst);
                    if index >= chunks.len() || index > first_failure.load(Ordering::SeqCst) {
                        break;
                    }
                    let outcome = f(&chunks[index]);
                    if outcome.is_err() {
                        first_failure.fetch_min(index, Ordering::SeqCst);
                    }
                    let _ = results[index].set(outcome);
                }
            });
        }
    });
    results.into_iter().map(OnceLock::into_inner).collect()
}

/// One boot-time pass over the data directory: crash leftovers of
/// staged writes are deleted (never published, and nothing may linger
/// as unbounded disk litter), and every context image found is
/// registered cold, described by its sidecar snapshot.
fn scan_data_dir(data_dir: &Path) -> io::Result<(BTreeMap<String, Arc<Entry>>, ResumedRenames)> {
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
                if let Err(error) = remove_persisted_file(&target)
                    && error.kind() != io::ErrorKind::NotFound
                {
                    tracing::warn!(path = %target.display(), %error, "unfinished deletion: file still held");
                }
            }
            // The marker goes last: it only leaves once the family did.
            if remove_persisted_file(&path).is_err() {
                tracing::warn!(path = %path.display(), "unfinished deletion: marker still held");
            }
        }
    }
    // Unfinished renames next, before the `.ctx` scan below: a
    // `.renaming` marker means `rename_context` moved (or was about to
    // move) the whole file family but crashed before the group
    // membership rewrite landed. Finishing the move here — repeatable,
    // since a missing source file just means it already moved — lets
    // the `.ctx` scan discover the context under its NEW name. The
    // marker itself survives this pass; `boot_with` removes it only
    // after also rewriting group membership, so a second crash still
    // has everything it needs to resume.
    let resumed_renames = resume_rename_markers(
        data_dir,
        "renaming",
        "context",
        |from_stem, to_stem| move_context_files(data_dir, from_stem, to_stem),
        // The pivot is `.ctx` — its arrival is what lets the `.ctx` scan
        // below register the context under `to`.
        |to_stem| data_dir.join(format!("{to_stem}.ctx")).exists(),
    )?;
    let mut candidates: Vec<(String, String)> = Vec::new();
    let mut import_markers: Vec<PathBuf> = Vec::new();
    for dir_entry in fs::read_dir(data_dir)? {
        let path = dir_entry?.path();
        let extension = path.extension().and_then(|e| e.to_str());
        if extension.is_some_and(|e| e.starts_with("tmp")) {
            let _ = remove_persisted_file(&path);
            continue;
        }
        // Import markers are judged after the scan, once it is known
        // which contexts exist — collect them on the way through.
        if extension == Some(IMPORT_MARKER_EXTENSION) {
            import_markers.push(path);
            continue;
        }
        if extension != Some("ctx") {
            continue;
        }
        let Some((stem, name)) = scanned_stem_and_name(&path) else {
            continue;
        };
        candidates.push((stem, name));
    }

    // The expensive part of a boot scan is the disk I/O per candidate
    // (sidecar read plus two `fs::metadata` calls), and each candidate
    // is independent — `parallel_map` pays for it in parallel the same
    // way `preload_pinned` does; arrival order cannot affect the result
    // since it only feeds a `BTreeMap`.
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let registry: BTreeMap<String, Arc<Entry>> =
        parallel_map(candidates, workers, |(stem, name)| {
            let stem = stem.as_str();
            let MetaFile { meta, stats, usage } = read_meta_file(data_dir, stem);
            // The gauge must see leftover logs from the first scrape,
            // not only after each context's first touch.
            let wal_bytes = fs::metadata(wal_path(data_dir, stem))
                .map(|meta| meta.len())
                .unwrap_or(0);
            let passages_wal_bytes = fs::metadata(passages_wal_path(data_dir, stem))
                .map(|meta| meta.len())
                .unwrap_or(0);
            (
                name,
                Arc::new(Entry::new(
                    meta,
                    stats,
                    Slot::Cold,
                    wal_bytes,
                    passages_wal_bytes,
                    usage,
                )),
            )
        })
        .into_iter()
        .collect();

    // Surviving import markers: each says a multi-store batch opened
    // and never finished — a crash (or an unretried refusal) between
    // retract_source, store_passages, add_associations, and
    // add_aliases. Every store is individually consistent, so this
    // marker is the ONLY thing that can say the source's truth is
    // half-applied. Report the live ones every boot until a re-import
    // or a retraction clears them; a marker whose context no longer
    // exists is moot (deletion destroys the batch's target) and is
    // removed here, completing delete()'s own best-effort sweep.
    for path in import_markers {
        let parsed = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<ImportMarker>(&bytes).ok());
        let Some(marker) = parsed else {
            tracing::warn!(
                path = %path.display(),
                "unreadable import marker — an import batch may be half-applied, \
                 but which source is unrecoverable; remove the file once investigated",
            );
            continue;
        };
        if registry.contains_key(&marker.context) {
            tracing::warn!(
                context = %marker.context,
                source = %marker.source,
                "an import batch for this source never completed — its truth may be \
                 half-applied (passages without associations, or associations without \
                 aliases); re-import the batch file or retract the source",
            );
        } else {
            let _ = remove_persisted_file(&path);
        }
    }
    Ok((registry, resumed_renames))
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

/// The per-set membership cap, judged wherever a WHOLE record is
/// stated at once (create, restore): the first over-cap set's label
/// comes back for the refusal. A delta update judges its prospective
/// result instead — see `update_group`.
fn check_member_caps(
    contexts: &BTreeSet<String>,
    children: &BTreeSet<String>,
) -> Result<(), &'static str> {
    for (field, set) in [("member contexts", contexts), ("child groups", children)] {
        if set.len() > groups::MAX_GROUP_MEMBERS {
            return Err(field);
        }
    }
    Ok(())
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

/// `sweep_membership`'s replace-not-remove twin: renames `from` to
/// `to` wherever it appears in the chosen set field, persisting each
/// touched record. Used both live (a context or group rename's last
/// step) and at boot (resuming a rename whose marker survived a
/// crash) — in both cases the caller already holds the groups write
/// lock and must call this BEFORE `reconcile_groups`, which would
/// otherwise see `from` as a dangling reference (nothing registered
/// under that name any more) and drop it instead of carrying it
/// forward.
fn rename_in_membership(
    data_dir: &Path,
    groups: &mut BTreeMap<String, GroupRecord>,
    from: &str,
    to: &str,
    field: impl Fn(&mut GroupRecord) -> &mut BTreeSet<String>,
) {
    for (group_name, record) in groups.iter_mut() {
        let set = field(record);
        if !set.remove(from) {
            continue;
        }
        set.insert(to.to_string());
        if let Err(error) = groups::write_group(data_dir, &file_stem(group_name), record) {
            tracing::warn!(
                group = %group_name,
                from = %from,
                to = %to,
                %error,
                "group membership rename not persisted; the next boot's resume retries it"
            );
        }
    }
}

/// Boot-time counterpart of the delete-path sweeps: drops every group
/// member that is not a registered context, every child that is not a
/// scanned group, every name past the [`groups::MAX_GROUP_MEMBERS`]
/// per-set cap, and every nesting edge that would close a cycle or
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
    registry: &BTreeMap<String, Arc<Entry>>,
    groups: &mut BTreeMap<String, GroupRecord>,
) {
    let scanned = groups.clone();
    for record in groups.values_mut() {
        record
            .contexts
            .retain(|context| registry.contains_key(context));
        record.groups.retain(|child| scanned.contains_key(child));
    }
    // Dangling names never count toward the cap — they were just
    // dropped — so the trim runs on what actually remains…
    groups::trim_membership(groups, groups::MAX_GROUP_MEMBERS);
    // …and what remains can still be the wrong SHAPE — the repair
    // drops exactly the edges the validator refuses, deterministically.
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
                    "dropped dangling, over-cap, or ill-nested group reference(s) at boot"
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
            replay_ops_guarded(&mut context, &ops)
                .map_err(|e| format!("context '{name}' WAL replay panicked: {e}"))?;
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

/// Runs the whole replay loop behind `catch_unwind`, turning a panic
/// (an actual bug tripped by some op's content, not a library
/// rejection — `replay_op` already turns those into a log line) into
/// the same `Err` shape a corrupt image or unreadable WAL produces.
/// Without this, a poisoned op would panic `ensure_hot` itself on
/// every subsequent access — this context can never come back Hot, so
/// every caller touching it crash-loops forever instead of hitting
/// the existing quarantine-and-retry path ([`LOAD_FAILURE_RETRY`]).
fn replay_ops_guarded(context: &mut Context, ops: &[WalOp]) -> Result<(), String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for op in ops {
            replay_op(context, op);
        }
    }))
    .map_err(|_| "an op panicked reapplying against a fresh load".to_string())
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
        WalOp::RetractAssociation {
            subject,
            label,
            object,
        } => {
            // A triple that names no live edge is a no-op on replay,
            // exactly like an unknown source above.
            context.retract_association(subject, label, object);
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

/// The durable-rename marker: while it exists, boot resumes the file
/// move AND re-applies the group membership rewrite (`contexts`
/// entries naming `from`) before `reconcile_groups` runs — without
/// that ordering, a crash between the move and the rewrite would have
/// reconcile see `from` as dangling and drop it, losing the
/// membership for good rather than carrying it to `to`. Removed only
/// once both the move and the rewrite are durable.
pub(crate) fn renaming_marker_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.renaming"))
}

/// The batch-open marker's file extension — `.deleted`'s sibling for
/// imports. Shared by the path builder, the boot sweep, and inspect,
/// so the three can never disagree about what counts as a marker.
pub(crate) const IMPORT_MARKER_EXTENSION: &str = "importing";

/// One import batch's in-flight marker: `{stem}.{fnv64(source)}.importing`,
/// written before the first of the batch's four separately-durable
/// mutations (retract_source → store_passages → add_associations →
/// add_aliases) and removed only after the last. While it exists, the
/// source's truth may be HALF-APPLIED — a crash between the steps
/// leaves passages without their associations, or associations without
/// their aliases, and every store is individually consistent, so
/// nothing else can tell. Boot and `taguru inspect` report survivors;
/// the repair is the documented one (re-import the batch file, whose
/// retract-then-apply is idempotent, or retract the source).
///
/// The source's name rides INSIDE the file (see [`ImportMarker`]); the
/// file name only needs to be unique per (context, source) and safe,
/// which the hash gives without an encoding scheme. Stems contain no
/// dots, so the `{stem}.` prefix plus the extension identifies a
/// marker's context unambiguously.
pub(crate) fn import_marker_path(dir: &Path, stem: &str, source: &str) -> PathBuf {
    dir.join(format!(
        "{stem}.{:016x}.{IMPORT_MARKER_EXTENSION}",
        fnv64(source.as_bytes())
    ))
}

/// Every import marker beside `stem`'s files — the enumeration the
/// delete and create sweeps need, since markers (unlike the fixed
/// `context_files` family) exist per in-flight source. Read failures
/// yield the empty list: both sweeps treat markers as best-effort
/// hygiene backed by boot's own pass.
pub(crate) fn import_marker_paths(dir: &Path, stem: &str) -> Vec<PathBuf> {
    let prefix = format!("{stem}.");
    let suffix = format!(".{IMPORT_MARKER_EXTENSION}");
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(&prefix) && name.ends_with(&suffix))
        })
        .collect()
}

/// What an import marker file says: which source's batch was open, in
/// which context — self-describing, so boot and inspect report the
/// human-readable pair without decoding file names.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ImportMarker {
    pub(crate) context: String,
    pub(crate) source: String,
}

/// What a rename marker file says: the source and destination names,
/// self-describing so boot can resume the move and the group rewrite
/// without any other input. Shared shape for contexts (`.renaming`)
/// and groups (`.grouprenaming`) — the two use different extensions
/// (a context and a group may share a name) but the same fields.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RenameMarker {
    pub(crate) from: String,
    pub(crate) to: String,
}

/// One rename whose marker `scan_data_dir`/`groups::scan_groups` found
/// at boot and tried to finish, handed back so `boot_with` can act
/// before `reconcile_groups` runs.
///
/// The two booleans decouple the two things a resume owes, because a
/// half-done move must not do the second without the first:
/// - `landed` — the destination's pivot file (a context's `.ctx`, a
///   group's `.group`) is now in place, so the scan registered the
///   entity under `to`. Group membership naming `from` must be
///   rewritten to `to`, or `reconcile_groups` — which has no notion of
///   a rename in flight — reads `from` as dangling and drops it.
/// - `complete` — every present file moved, so the marker has done its
///   job and may be removed. If a straggler sidecar was still held,
///   this stays false and the marker survives for the next boot to
///   retry, even though `landed` (and the membership rewrite) already
///   went through. Deleting the marker on a `landed`-but-not-`complete`
///   resume was the bug: the retry vanished, orphaning the straggler.
pub(crate) struct ResumedRename {
    pub(crate) from: String,
    pub(crate) to: String,
    pub(crate) landed: bool,
    pub(crate) complete: bool,
}

/// Every rename resumed in one boot scan (see [`ResumedRename`]).
pub(crate) type ResumedRenames = Vec<ResumedRename>;

/// Serializes and durably writes a rename marker at `path` — the
/// first step of both `rename_context_locked` and `rename_group`,
/// which must land before anything else moves (see their docs for
/// why the marker comes first and is not best-effort).
fn write_rename_marker(path: &Path, from: &str, to: &str) -> io::Result<()> {
    let body = serde_json::to_vec(&RenameMarker {
        from: from.to_string(),
        to: to.to_string(),
    })
    .expect("RenameMarker has no non-serializable field");
    write_atomic(path, &body)
}

/// Resumes every `extension` rename marker found in `dir`: reads it,
/// parses the `(from, to)` pair, moves that pair's files via
/// `move_files`, and returns every pair resumed (see [`ResumedRename`]
/// for what the two per-rename booleans mean and why the caller needs
/// both). `scan_data_dir` (`.renaming`, a nine-file context family) and
/// `groups::scan_groups` (`.grouprenaming`, one file) share this exact
/// shape and differ only in what "moving the files" means for their
/// entity — `entity` names it for the log lines (`"context"` /
/// `"group"`).
///
/// `destination_landed(to_stem)` answers "is the destination's pivot
/// file now in place?" — checked whether or not `move_files` returned
/// Ok, because a move can fail on a straggler AFTER the pivot arrived.
/// That is `landed`; `move_files` returning Ok is `complete`.
pub(crate) fn resume_rename_markers(
    dir: &Path,
    extension: &str,
    entity: &str,
    mut move_files: impl FnMut(&str, &str) -> io::Result<()>,
    destination_landed: impl Fn(&str) -> bool,
) -> io::Result<ResumedRenames> {
    let mut resumed = Vec::new();
    for dir_entry in fs::read_dir(dir)? {
        let path = dir_entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some(extension) {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else {
            tracing::warn!(path = %path.display(), entity, "unreadable rename marker; a rename may be stuck half-done");
            continue;
        };
        let Ok(marker) = serde_json::from_slice::<RenameMarker>(&bytes) else {
            tracing::warn!(path = %path.display(), entity, "rename marker does not parse; a rename may be stuck half-done");
            continue;
        };
        tracing::warn!(from = %marker.from, to = %marker.to, entity, "resuming an unfinished rename");
        let to_stem = file_stem(&marker.to);
        let complete = match move_files(&file_stem(&marker.from), &to_stem) {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!(from = %marker.from, to = %marker.to, entity, %error, "unfinished rename: file still held");
                false
            }
        };
        // Ask the disk, not the move's return value: a straggler sidecar
        // can stick (complete = false) long after the pivot moved, and
        // the membership rewrite keys on the pivot, not on completeness.
        let landed = destination_landed(&to_stem);
        resumed.push(ResumedRename {
            from: marker.from,
            to: marker.to,
            landed,
            complete,
        });
    }
    Ok(resumed)
}

/// Every rename marker of `extension` in `dir` that names `context` as
/// its DESTINATION. A marker sits at its SOURCE's stem, so a create of
/// the destination name cannot find it positionally the way it clears
/// the marker at its own stem; this scan lets the create sweep abandon a
/// half-done rename that would otherwise have boot's resume move the
/// source family over the fresh generation. Unreadable or unparseable
/// markers are skipped — boot's own sweep reports them. Shared by the
/// context (`renaming`) and group (`grouprenaming`) create paths.
fn rename_markers_targeting(dir: &Path, context: &str, extension: &str) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some(extension))
        .filter(|path| {
            fs::read(path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<RenameMarker>(&bytes).ok())
                .is_some_and(|marker| marker.to == context)
        })
        .collect()
}

/// FNV-1a over raw bytes — the same primitive the search terms build
/// on, exposed for the one non-search need (import marker file names).
fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x1_0000_01b3;

/// Each lane's over-fetch for one served `limit`: fusion can promote a
/// hit neither lane put in its own top `limit`, and the staleness
/// checks drop stragglers.
fn lane_pool(limit: usize) -> usize {
    limit.saturating_mul(4).max(50)
}

/// The deduplicated term keys of one query — first occurrence of each
/// key wins, so the stream keeps [`passage_terms`] order.
fn deduped_query_grams(query: &str) -> Vec<u64> {
    let mut seen = std::collections::HashSet::new();
    passage_terms(query)
        .into_iter()
        .filter(|gram| seen.insert(*gram))
        .collect()
}

/// [`deduped_query_grams`] with each key's spelling — the explain
/// path's view of the same stream (same walker, same dedup rule).
fn deduped_spelled_query_terms(query: &str) -> Vec<(String, u64)> {
    let mut seen = std::collections::HashSet::new();
    spelled_passage_terms(query)
        .into_iter()
        .filter(|&(_, gram)| seen.insert(gram))
        .collect()
}

/// The floor-filtered lane view of one vector sweep, in sweep order —
/// the pool cap is the caller's (`top_matches` already applied it).
fn semantic_lane_hits(
    rows: Vec<(&crate::embedding::PassageKey, f32)>,
    floor: f32,
) -> Vec<(String, u32, u64, f32)> {
    rows.into_iter()
        .filter(|&(_, score)| score >= floor)
        .map(|(key, score)| (key.source.clone(), key.index, key.hash, score))
        .collect()
}

/// Fuses the two lanes' pools into the served ranking. Fuse by rank,
/// then validate EACH LANE against the store's current paragraph:
/// every lane scored the text it saw, and vectors routinely lag the
/// text between refreshes — a stale lane must neither smuggle its
/// outdated score onto fresh text nor veto the other lane's fresh
/// match, so each loses exactly its own evidence (and its fusion
/// term). The top-level score stays the raw BM25 number when no
/// semantic lane ran, so a lexical-only deployment keeps its
/// historical score semantics.
fn fuse_passage_lanes(
    store: &crate::passages::PassageStore,
    lexical: Vec<crate::bm25::IndexHit>,
    semantic: Vec<(String, u32, u64, f32)>,
    limit: usize,
) -> Vec<PassageSearchHit> {
    const RRF_K: f32 = 60.0;
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
    hits
}

/// Where the term walkers deliver their stream. The index build wants
/// bare hashes; the search-explain path wants the spelling each hash
/// was computed from next to it. One walker feeds both through this
/// sink, so the two views cannot disagree about what a term is —
/// spelling events cost nothing when the sink ignores them.
trait TermSink {
    /// One character of the ASCII word being hashed, in hash order,
    /// case-folded exactly as the hash saw it. A `word` call closes
    /// the sequence.
    fn word_char(&mut self, ch: char);
    /// The word whose characters were just streamed.
    fn word(&mut self, hash: u64);
    /// An adjacent character pair inside a non-ASCII run.
    fn pair(&mut self, hash: u64, first: char, second: char);
    /// A non-ASCII run of exactly one character.
    fn lone(&mut self, hash: u64, ch: char);
}

/// The index build's sink: hashes only.
struct HashSink(Vec<u64>);

impl TermSink for HashSink {
    fn word_char(&mut self, _: char) {}

    fn word(&mut self, hash: u64) {
        self.0.push(hash);
    }

    fn pair(&mut self, hash: u64, _: char, _: char) {
        self.0.push(hash);
    }

    fn lone(&mut self, hash: u64, ch: char) {
        let _ = ch;
        self.0.push(hash);
    }
}

/// The explain path's sink: every hash next to its spelling.
struct SpellingSink {
    word: String,
    terms: Vec<(String, u64)>,
}

impl TermSink for SpellingSink {
    fn word_char(&mut self, ch: char) {
        self.word.push(ch);
    }

    fn word(&mut self, hash: u64) {
        self.terms.push((std::mem::take(&mut self.word), hash));
    }

    fn pair(&mut self, hash: u64, first: char, second: char) {
        let mut spelling = String::with_capacity(first.len_utf8() + second.len_utf8());
        spelling.push(first);
        spelling.push(second);
        self.terms.push((spelling, hash));
    }

    fn lone(&mut self, hash: u64, ch: char) {
        self.terms.push((ch.to_string(), hash));
    }
}

/// The terms of one passage or query: [`text_terms`] over the
/// normalized text, plus a word term per piece of every camelCase run.
/// One function serves both sides of the search, so they cannot
/// disagree about what a term is.
pub(crate) fn passage_terms(raw: &str) -> Vec<u64> {
    let mut sink = HashSink(Vec::new());
    walk_passage_terms(raw, &mut sink);
    sink.0
}

/// [`passage_terms`] with each hash next to the spelling it was
/// computed from — a whole lowercased word, a camelCase piece, the two
/// characters of an adjacent pair, or a lone character. Same walker,
/// same order; only the sink differs, so the rendered stream IS the
/// hashed stream. Recomputed per explain call — the index keeps hashes
/// only, and grows no reverse map for a diagnostic path.
pub(crate) fn spelled_passage_terms(raw: &str) -> Vec<(String, u64)> {
    let mut sink = SpellingSink {
        word: String::new(),
        terms: Vec::new(),
    };
    walk_passage_terms(raw, &mut sink);
    sink.terms
}

/// The walker under [`passage_terms`]. The camelCase split reads an
/// NFKC-folded but NOT lowercased view of the input: lowercasing would
/// erase the very case boundaries that let `state` reach `AppState`,
/// while the width fold keeps a full-width `Ａ` — which the normalized
/// whole-word term already folds to ASCII — in the same run as its
/// ASCII neighbors instead of breaking it (so `ＡpplePie` yields the
/// `apple` piece, matching a plain `apple` cue).
fn walk_passage_terms(raw: &str, sink: &mut impl TermSink) {
    use unicode_normalization::UnicodeNormalization;
    walk_text_terms(&taguru::context::normalize_entry(raw), sink);
    let mut run: Vec<char> = Vec::new();
    for ch in raw.nfkc() {
        if ch.is_ascii_alphanumeric() {
            run.push(ch);
        } else {
            camel_pieces(&run, sink);
            run.clear();
        }
    }
    camel_pieces(&run, sink);
}

/// Emits one lowercased word term per piece of an ASCII run that
/// splits at case boundaries: `aB` → `a|B`, digits stick to their
/// piece (`U64Max` → `u64|max`), and an acronym ends before its last
/// capital (`HTTPServer` → `http|server`). A run with no boundary
/// emits nothing — its whole-word term is already in the stream.
/// Pieces hash exactly like [`text_terms`] words, so a piece matches
/// wherever the same word occurs standalone.
fn camel_pieces(run: &[char], sink: &mut impl TermSink) {
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
            let ch = ch.to_ascii_lowercase();
            sink.word_char(ch);
            word ^= ch as u64;
            word = word.wrapping_mul(FNV_PRIME);
        }
        sink.word(word | 1 << 63);
    }
}

/// [`walk_text_terms`] collected as bare keys — the tokenization tests'
/// entrance; production goes through [`passage_terms`], which layers
/// the camelCase pieces on top of the same walker.
#[cfg(test)]
fn text_terms(text: &str) -> Vec<u64> {
    let mut sink = HashSink(Vec::new());
    walk_text_terms(text, &mut sink);
    sink.0
}

/// The word/bigram layer under [`passage_terms`]. ASCII-alphanumeric
/// runs count as whole words; everything else contributes adjacent
/// character pairs within its run (a run of one contributes the lone
/// character). Space-delimited languages need word terms — character
/// pairs occur in every English document alike, which flattens IDF to
/// nothing — while undelimited Japanese needs the bigrams. Runs break
/// at spaces and punctuation, and a script switch breaks the run too,
/// so terms never straddle "第10篇"-style boundaries.
fn walk_text_terms(text: &str, sink: &mut impl TermSink) {
    let mut word = FNV_OFFSET; // running FNV-1a over the current ASCII word
    let mut in_word = false;
    let mut run: Option<char> = None; // previous char of the current non-ASCII run
    let mut run_len = 0usize;
    fn flush_run(sink: &mut impl TermSink, run: &mut Option<char>, run_len: &mut usize) {
        if let (Some(last), 1) = (*run, *run_len) {
            sink.lone(last as u64, last); // below the pair space: pairs always have bits 32+
        }
        *run = None;
        *run_len = 0;
    }
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            flush_run(sink, &mut run, &mut run_len);
            sink.word_char(ch);
            word ^= ch as u64;
            word = word.wrapping_mul(FNV_PRIME);
            in_word = true;
        } else {
            if in_word {
                sink.word(word | 1 << 63); // disjoint from pair keys (chars < 2^21)
                word = FNV_OFFSET;
                in_word = false;
            }
            if ch.is_alphanumeric() {
                if let Some(prev) = run {
                    sink.pair(((prev as u64) << 32) | ch as u64, prev, ch);
                }
                run = Some(ch);
                run_len += 1;
            } else {
                flush_run(sink, &mut run, &mut run_len);
            }
        }
    }
    if in_word {
        sink.word(word | 1 << 63);
    }
    flush_run(sink, &mut run, &mut run_len);
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

/// Test-only deterministic fault injection for registry persistence.
///
/// The calling thread fails exactly one persistence operation after
/// `successes` stage, commit, unlink, WAL append, or WAL truncate
/// operations have run normally.
/// Keeping the counter thread-local makes parallel tests independent,
/// and routing every operation through shared choke points avoids a
/// flag for each call site.
#[cfg(test)]
pub(crate) fn fail_persistence_ops_after(successes: u32) {
    PERSISTENCE_FAULT.with(|cell| cell.set(Some(successes)));
}

#[cfg(test)]
thread_local! {
    static PERSISTENCE_FAULT: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

/// Clears an armed fault and reports whether it was still pending. A
/// sweep uses this after each attempt: `false` means the selected op
/// was reached and failed, while `true` means the operation had fewer
/// persistence steps and the sweep is complete.
#[cfg(test)]
pub(crate) fn clear_persistence_fault() -> bool {
    PERSISTENCE_FAULT.with(|cell| cell.take().is_some())
}

#[cfg(test)]
pub(crate) fn injected_persistence_failure(operation: &str) -> Option<io::Error> {
    PERSISTENCE_FAULT.with(|cell| match cell.get() {
        Some(0) => {
            cell.set(None);
            Some(io::Error::other(format!(
                "injected registry persistence failure during {operation}"
            )))
        }
        Some(remaining) => {
            cell.set(Some(remaining - 1));
            None
        }
        None => None,
    })
}

#[cfg(not(test))]
pub(crate) fn injected_persistence_failure(_operation: &str) -> Option<io::Error> {
    None
}

/// The unlink choke point shared by registry and group persistence.
pub(crate) fn remove_persisted_file(path: impl AsRef<Path>) -> io::Result<()> {
    if let Some(error) = injected_persistence_failure("unlink") {
        return Err(error);
    }
    fs::remove_file(path)
}

/// The rename choke point shared by atomic publication and recovery
/// paths that move corrupt bytes aside.
pub(crate) fn rename_persisted_file(
    from: impl AsRef<Path>,
    to: impl AsRef<Path>,
) -> io::Result<()> {
    if let Some(error) = injected_persistence_failure("commit") {
        return Err(error);
    }
    fs::rename(from, to)
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
        let _ = remove_persisted_file(&staged);
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
    if let Some(error) = injected_persistence_failure("stage") {
        return Err(error);
    }
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
            let _ = remove_persisted_file(&staged);
            Err(error)
        }
    }
}

/// The cheap half of [`write_atomic`]: atomically publishes a staged
/// file at its final path — rename plus parent-directory fsync. Also
/// the whole of a same-directory file move (`staged` need not be a
/// `.tmp*` name): `rename_group` and `groups::scan_groups`'s
/// rename-marker resume both use it that way. [`move_context_files`]
/// moves nine files under one parent and fsyncs once itself instead of
/// calling this per file — see there.
pub(crate) fn commit_staged(staged: &Path, path: &Path) -> io::Result<()> {
    rename_persisted_file(staged, path)?;
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
/// holds it. Unix-only; elsewhere the rename stays atomic against a
/// crash mid-write, just not durable against power loss — unix is
/// what this server targets.
pub(crate) fn fsync_dir(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    fs::File::open(dir)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// [`fsync_dir`] on `path`'s parent — the common case, a single file
/// rename or creation.
pub(crate) fn fsync_parent_dir(path: &Path) -> io::Result<()> {
    match path.parent() {
        Some(parent) => fsync_dir(parent),
        None => Ok(()),
    }
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

/// Moves one context's whole file family from `from_stem` to
/// `to_stem`, file by file, in the fixed order [`context_files`]
/// defines — a missing source is skipped (an earlier, interrupted
/// attempt already moved it; safe to retry at boot or from a fresh
/// call). `.ctx` is index 0 and the pivot the boot scan registers a
/// context by: if IT will not move, nothing else does either (the
/// family stays wholly under `from_stem`, cleanly retried), and the
/// call fails before touching a sidecar. Once the pivot has moved, a
/// sidecar that still sticks is best-effort — the rest are moved anyway
/// so the retry has fewer orphans to chase — but the first such error
/// is returned so the caller knows the move is incomplete and keeps the
/// rename marker. All nine share `data_dir` as their parent, so one
/// fsync after every rename covers the whole family durably instead of
/// paying for it (via `commit_staged`) up to nine times.
fn move_context_files(data_dir: &Path, from_stem: &str, to_stem: &str) -> io::Result<()> {
    let mut moved_any = false;
    let mut first_error: Option<io::Error> = None;
    for (position, (from_file, to_file)) in context_files(from_stem)
        .into_iter()
        .zip(context_files(to_stem))
        .enumerate()
    {
        match fs::rename(data_dir.join(from_file), data_dir.join(to_file)) {
            Ok(()) => moved_any = true,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            // The pivot: fail outright so nothing else moves.
            Err(error) if position == 0 => return Err(error),
            // A post-pivot straggler: keep going, remember the first.
            Err(error) => {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
    }
    if moved_any {
        fsync_dir(data_dir)?;
    }
    first_error.map_or(Ok(()), Err)
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
    use crate::context_proptest::{
        AliasInput, AssocInput, GeneratedGroupOp, RetractionInput, config as proptest_config,
        group_op_strategy, json_roundtrip_f64_strategy, scenario_strategy, wal_op_strategy,
    };
    use proptest::prelude::*;

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
                    Deadline::unbounded(),
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

    /// Every stage/commit/unlink position in context deletion either
    /// finishes immediately or leaves enough durable state for boot to
    /// finish it. The first index beyond the operation proves the sweep
    /// did not merely sample a few hand-picked failures.
    #[test]
    fn every_context_delete_persistence_failure_recovers_at_boot() {
        let mut exhausted = false;
        for failure in 0..64 {
            let dir = scratch_dir(&format!("delete-fault-{failure}"));
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("doc"))],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            state.flush_dirty();
            state
                .create_group(
                    "breweries",
                    String::new(),
                    BTreeSet::from(["sake".to_string()]),
                    BTreeSet::new(),
                )
                .unwrap();

            fail_persistence_ops_after(failure);
            let outcome = state.delete("sake").unwrap();
            let past_end = clear_persistence_fault();
            drop(state);

            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            assert!(
                state.directory_entry("sake").is_none(),
                "failure at persistence step {failure} resurrected the context: {outcome:?}"
            );
            assert!(
                state.group("breweries").unwrap().contexts.is_empty(),
                "boot did not reconcile group membership at step {failure}"
            );
            assert!(
                !deleted_marker_path(&dir, "sake").exists(),
                "boot did not finish the marker at step {failure}"
            );
            drop(state);
            let _ = fs::remove_dir_all(&dir);

            if past_end {
                assert!(outcome.is_ok());
                exhausted = true;
                break;
            }
        }
        assert!(exhausted, "context deletion exceeded the sweep bound");
    }

    /// Standalone `retract_source` — the only path the HTTP endpoint and
    /// the MCP tool ever call — used to bracket its two independently
    /// durable writes (the graph's WAL, then the passage store's) with
    /// nothing: a crash between them left the graph durably retracted
    /// while the passage text survived on disk, invisible to boot or
    /// `taguru inspect`. Every fault point must now leave either a
    /// completed, marker-free retraction, or a surviving marker naming
    /// the tear — never a silent gap between the two stores.
    #[test]
    fn every_standalone_retract_persistence_failure_is_detected_or_completes() {
        let mut exhausted = false;
        for failure in 0..24 {
            let dir = scratch_dir(&format!("retract-fault-{failure}"));
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("doc"))],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            let mut passages = BTreeMap::new();
            passages.insert("doc".to_string(), "杜氏は高瀬。".to_string());
            state
                .store_passages("sake", plain(passages))
                .unwrap()
                .unwrap();

            fail_persistence_ops_after(failure);
            let first = state.retract_source("sake", "doc");
            let past_end = clear_persistence_fault();
            let marker = import_marker_path(&dir, "sake", "doc");

            if past_end {
                assert!(
                    first.is_ok(),
                    "the past-end attempt must complete: {first:?}"
                );
                assert!(!marker.exists());
            } else {
                match &first {
                    // A witness must survive whenever the graph side may
                    // already be durably retracted while the passage side
                    // never ran: any refusal other than the marker write
                    // itself failing (which leaves nothing behind to
                    // witness, since nothing happened yet).
                    Err(AccessError::Unpersisted(message)) => {
                        let before_marker = message.contains("import marker");
                        assert_eq!(
                            marker.exists(),
                            !before_marker,
                            "a stopped retraction at step {failure} lost its tear witness: {first:?}"
                        );
                    }
                    // The graph write itself never swallows a failure —
                    // the only way this call can succeed while a fault
                    // fired somewhere is a passage-side failure folded
                    // into an honest `passage_removed: false`. The
                    // witness must survive exactly that swallow, or the
                    // half-applied state it names (graph retracted,
                    // passage still on disk) becomes permanently
                    // invisible to boot and `taguru inspect`.
                    Ok(_) => {
                        assert!(
                            marker.exists(),
                            "a swallowed passage failure at step {failure} still cleared \
                             the tear witness: {first:?}"
                        );
                    }
                    Err(_) => {}
                }
                // Retracting again is the documented repair, and
                // retract_source is idempotent per-source, so it is
                // exact even when the injected failure was swallowed
                // internally or only prevented marker cleanup.
                state.retract_source("sake", "doc").unwrap();
                assert!(
                    !marker.exists(),
                    "repair did not clear failure step {failure}"
                );
            }

            // A fully retracted edge stays (storage is append-only) but
            // nets to zero attributions — the same end-state
            // `retract_source_withdraws_its_contributions` checks.
            let attributions_gone = state
                .read_context("sake", |context| {
                    context.query(Some("蔵"), None, Some("高瀬"))[0]
                        .attributions
                        .is_empty()
                })
                .unwrap();
            assert!(
                attributions_gone,
                "retry at step {failure} did not retract the association"
            );
            let (found, missing) = state
                .lookup_passages("sake", &["doc".to_string()])
                .unwrap()
                .unwrap();
            assert!(
                !found.contains_key("doc") && missing == vec!["doc".to_string()],
                "retry at step {failure} did not retract the passage"
            );

            drop(state);
            let _ = fs::remove_dir_all(&dir);

            if past_end {
                exhausted = true;
                break;
            }
        }
        assert!(exhausted, "standalone retraction exceeded the sweep bound");
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
                    Deadline::unbounded(),
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

    /// The import batch-open marker: opened before a batch's first
    /// mutation, cleared only after its last — while it exists, boot
    /// and inspect can name a half-applied source nothing else sees.
    #[test]
    fn import_markers_open_clear_and_sweep_with_their_context() {
        let dir = scratch_dir("import-markers");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        state.open_import_marker("sake", "doc-1").unwrap();
        let marker = import_marker_path(&dir, "sake", "doc-1");
        assert!(marker.exists(), "open writes the marker");
        // Distinct sources get distinct files — concurrent imports of
        // one context never race on a shared marker.
        state.open_import_marker("sake", "doc-2").unwrap();
        assert_eq!(import_marker_paths(&dir, "sake").len(), 2);
        // The content names the pair, so reports never decode filenames.
        let parsed: ImportMarker = serde_json::from_slice(&fs::read(&marker).unwrap()).unwrap();
        assert_eq!(
            (parsed.context.as_str(), parsed.source.as_str()),
            ("sake", "doc-1")
        );

        state.clear_import_marker("sake", "doc-1");
        assert!(!marker.exists(), "clear removes exactly its own marker");
        assert_eq!(import_marker_paths(&dir, "sake").len(), 1);

        // Deletion takes the survivors with the family: a marker must
        // not have boot report a tear in a context that is gone.
        state.delete("sake").unwrap().unwrap();
        assert!(
            import_marker_paths(&dir, "sake").is_empty(),
            "delete sweeps markers"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// Boot's marker pass: a surviving marker whose context exists is
    /// the torn-import report (and stays on disk for the next boot to
    /// repeat, until re-import or retraction); one whose context is
    /// gone is moot and is removed — it completes delete()'s own
    /// best-effort sweep.
    #[test]
    fn boot_keeps_a_live_torn_import_marker_and_removes_a_moot_one() {
        let dir = scratch_dir("import-marker-boot");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            // The crash-shaped state: batches opened their markers and
            // the process died between the four mutations.
            state.open_import_marker("sake", "doc-1").unwrap();
            state.open_import_marker("ghost", "doc-9").unwrap();
            state.flush_dirty();
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(
            import_marker_path(&dir, "sake", "doc-1").exists(),
            "a live context's tear stays visible until the repair runs"
        );
        assert!(
            !import_marker_path(&dir, "ghost", "doc-9").exists(),
            "a marker without its context is moot; boot removes it"
        );
        drop(state);
        let _ = fs::remove_dir_all(&dir);
    }

    /// `.deleted`'s recreate rule, for import markers: a marker the
    /// delete sweep could not remove must not survive into a freshly
    /// created context of the same name — boot would report the new
    /// generation as carrying a tear it never ran.
    #[test]
    fn creating_a_context_clears_stale_import_markers() {
        let dir = scratch_dir("import-marker-recreate");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state.delete("sake").unwrap().unwrap();
        // The failure delete() cannot fully guard: its marker sweep
        // missed one (crash, held handle), so the file outlives the
        // name.
        fs::write(
            import_marker_path(&dir, "sake", "doc-1"),
            b"{\"context\":\"sake\",\"source\":\"doc-1\"}",
        )
        .unwrap();

        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "recreate")
            .unwrap();
        assert!(
            import_marker_paths(&dir, "sake").is_empty(),
            "create clears the earlier generation's markers"
        );
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
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            state.retract_source("sake", "gone.md").unwrap();
            before = live_facts(&state);

            let outcome = state
                .compact_context("sake", Deadline::unbounded())
                .unwrap();
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
                    Deadline::unbounded(),
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

    fn apply_generated_context(
        state: &AppState,
        assoc_ops: &[AssocInput],
        alias_ops: &[AliasInput],
        retractions: &[RetractionInput],
    ) {
        let ops = assoc_ops
            .iter()
            .map(|op| AssocOp {
                subject: op.subject.to_string(),
                label: op.label.to_string(),
                object: op.object.to_string(),
                weight: op.weight,
                source: op.source.map(str::to_string),
                paragraph: op.paragraph,
            })
            .collect();
        state
            .add_associations("generated", ops, Deadline::unbounded())
            .unwrap()
            .unwrap();

        for alias_op in alias_ops {
            let (concepts, labels) = match alias_op {
                AliasInput::Concept { alias, canonical } => (
                    BTreeMap::from([(alias.to_string(), canonical.to_string())]),
                    BTreeMap::new(),
                ),
                AliasInput::Label { alias, canonical } => (
                    BTreeMap::new(),
                    BTreeMap::from([(alias.to_string(), canonical.to_string())]),
                ),
            };
            let _ = state.add_aliases("generated", &concepts, &labels).unwrap();
        }

        for retraction in retractions {
            match retraction {
                RetractionInput::Source(source) => {
                    state.retract_source("generated", source).unwrap();
                }
                RetractionInput::Association {
                    subject,
                    label,
                    object,
                } => {
                    state
                        .retract_association("generated", subject, label, object)
                        .unwrap();
                }
            }
        }
    }

    proptest! {
        #![proptest_config(proptest_config())]

        /// Any image watermark splits the acknowledged operation history into
        /// an already-baked prefix and a replayed suffix. A further partial
        /// record beyond that is crash debris from an in-flight write: most
        /// cuts leave invalid or checksum-mismatched bytes, which must
        /// vanish without changing the acknowledged state — but the one cut
        /// that removes only the trailing newline is indistinguishable from
        /// an already-acknowledged record that lost just its delimiter, and
        /// replay keeps it (see `TornTail::Recovered`).
        #[test]
        fn wal_replay_from_any_acknowledged_prefix_rebuilds_the_acknowledged_state(
            candidates in prop::collection::vec(wal_op_strategy(), 1..16),
            watermark_pick in any::<prop::sample::Index>(),
            torn_op in wal_op_strategy(),
            torn_at in any::<prop::sample::Index>(),
        ) {
            let mut expected = Context::default();
            let candidates: Vec<_> = candidates.into_iter().map(WalOp::from).collect();
            let acknowledged: Vec<_> = candidates
                .into_iter()
                .filter(|op| apply_op(&mut expected, op).is_ok())
                .collect();
            let watermark = watermark_pick.index(acknowledged.len() + 1);

            let dir = scratch_dir("wal-prefix-property");
            fs::create_dir_all(&dir).unwrap();
            let path = dir.join("generated.wal.jsonl");
            wal::append_batch(&path, 1, &acknowledged).unwrap();
            let healthy_len = fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);

            let mut restored = Context::default();
            for op in &acknowledged[..watermark] {
                replay_op(&mut restored, op);
            }
            let (pending, top) = wal::replay::<WalOp>(&path, watermark as u64).unwrap();
            for op in &pending {
                replay_op(&mut restored, op);
            }
            prop_assert_eq!(restored.to_bytes(), expected.to_bytes());
            prop_assert_eq!(top, acknowledged.len() as u64);

            // Build one real checksummed record, then retain an arbitrary
            // non-empty strict prefix: every possible cut is a torn tail.
            let fragment_path = dir.join("fragment.wal.jsonl");
            let torn_op = WalOp::from(torn_op);
            wal::append_batch(
                &fragment_path,
                acknowledged.len() as u64 + 1,
                std::slice::from_ref(&torn_op),
            )
            .unwrap();
            let record = fs::read(&fragment_path).unwrap();
            let cut = torn_at.index(record.len() - 1) + 1;
            let mut bytes = fs::read(&path).unwrap_or_default();
            bytes.extend_from_slice(&record[..cut]);
            fs::write(&path, bytes).unwrap();

            let (pending, top) = wal::replay::<WalOp>(&path, watermark as u64).unwrap();
            let mut healed = Context::default();
            for op in &acknowledged[..watermark] {
                replay_op(&mut healed, op);
            }
            for op in &pending {
                replay_op(&mut healed, op);
            }

            if cut == record.len() - 1 {
                // The retained prefix is the record's complete,
                // checksum-valid bytes minus only its own trailing
                // newline — byte-for-byte what an already-acknowledged
                // record that lost just its delimiter looks like.
                // `replay` cannot tell the two apart and, by design (see
                // `TornTail::Recovered`), keeps it rather than discarding
                // it as debris.
                let mut fully_acknowledged = Context::default();
                for op in &acknowledged {
                    replay_op(&mut fully_acknowledged, op);
                }
                replay_op(&mut fully_acknowledged, &torn_op);
                prop_assert_eq!(healed.to_bytes(), fully_acknowledged.to_bytes());
                prop_assert_eq!(top, acknowledged.len() as u64 + 1);
                prop_assert_eq!(
                    fs::metadata(&path).unwrap().len(),
                    healthy_len + record.len() as u64
                );
            } else {
                // Every shorter prefix is genuinely incomplete — either
                // invalid JSON or bytes whose checksum cannot match — so
                // it is unacknowledged crash debris that must vanish.
                prop_assert_eq!(healed.to_bytes(), expected.to_bytes());
                prop_assert_eq!(top, acknowledged.len() as u64);
                prop_assert_eq!(fs::metadata(&path).unwrap().len(), healthy_len);
            }

            let _ = fs::remove_dir_all(dir);
        }

        /// The operator path installs the same canonical context as the
        /// library rebuild, flushes it immediately, and re-applies state
        /// that lives outside the graph image when it is loaded again.
        #[test]
        fn registry_compaction_flushes_and_reloads_the_canonical_image(
            (assoc_ops, alias_ops, retractions) in scenario_strategy(),
            dice_floor in prop::option::of(json_roundtrip_f64_strategy(-1.0f64..2.0)),
        ) {
            let dir = scratch_dir("compact-property");
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create(
                    "generated",
                    ContextMeta {
                        dice_floor,
                        ..ContextMeta::default()
                    },
                )
                .unwrap();
            apply_generated_context(&state, &assoc_ops, &alias_ops, &retractions);
            // Project the current WAL watermark into the source image first;
            // compaction must carry this known non-zero value forward.
            prop_assert_eq!(state.flush_dirty(), vec!["generated"]);

            let (expected_image, expected_seq, expected_floor, expected_stats) = state
                .read_context("generated", |context| {
                    let (mut canonical, stats) =
                        context.compacted(Deadline::unbounded()).unwrap();
                    canonical.set_applied_seq(context.applied_seq());
                    canonical.set_dice_floor(Some(context.dice_floor()));
                    (
                        canonical.to_bytes(),
                        context.applied_seq(),
                        context.dice_floor(),
                        stats,
                    )
                })
                .unwrap();

            let outcome = state
                .compact_context("generated", Deadline::unbounded())
                .unwrap();
            prop_assert_eq!(outcome.dead_edges, expected_stats.dead_edges);
            prop_assert_eq!(outcome.aliases_dropped, expected_stats.aliases_dropped);
            state
                .read_context("generated", |context| {
                    assert_eq!(context.to_bytes(), expected_image);
                    assert_eq!(context.applied_seq(), expected_seq);
                    assert_eq!(context.dice_floor(), expected_floor);
                })
                .unwrap();
            let disk_image = fs::read(image_path(&dir, &file_stem("generated"))).unwrap();
            prop_assert_eq!(&disk_image, &expected_image);

            let second = state
                .compact_context("generated", Deadline::unbounded())
                .unwrap();
            prop_assert_eq!(second.dead_edges, 0);
            prop_assert_eq!(second.aliases_dropped, 0);
            drop(state);

            let reloaded = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            reloaded
                .read_context("generated", |context| {
                    assert_eq!(context.to_bytes(), expected_image);
                    assert_eq!(context.applied_seq(), expected_seq);
                    assert_eq!(context.dice_floor(), expected_floor);
                })
                .unwrap();
            drop(reloaded);
            let _ = fs::remove_dir_all(dir);
        }
    }

    /// [`AppState::run_maintenance_compaction`] selects worst-ratio-first
    /// and drops anything at or under the floor — the ordering and the
    /// threshold `POST /maintenance/compact` promises callers.
    #[test]
    fn run_maintenance_compaction_orders_worst_ratio_first_and_applies_the_floor() {
        let dir = scratch_dir("maint-order");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

        state
            .create("clean", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "clean",
                vec![
                    assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md")),
                    assoc_op("蔵", "銘柄", "青嶺", 1.0, Some("keep.md")),
                    assoc_op("蔵", "産地", "灘", 1.0, Some("keep.md")),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();

        state
            .create("mild", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "mild",
                vec![
                    assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md")),
                    assoc_op("蔵", "銘柄", "青嶺", 1.0, Some("keep.md")),
                    assoc_op("蔵", "産地", "灘", 1.0, Some("keep.md")),
                    assoc_op("蔵", "廃止銘柄", "旧銘", 1.0, Some("gone.md")),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state.retract_source("mild", "gone.md").unwrap();

        state
            .create("rotten", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "rotten",
                vec![
                    assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md")),
                    assoc_op("蔵", "廃止銘柄", "旧銘", 1.0, Some("gone.md")),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state.retract_source("rotten", "gone.md").unwrap();

        let outcome = state.run_maintenance_compaction(0.0, Deadline::unbounded());
        assert!(!outcome.deadline_exceeded);
        let names: Vec<&str> = outcome
            .contexts
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        assert_eq!(names, vec!["rotten", "mild"], "{names:?}");
        assert_eq!(outcome.contexts[0].outcome.dead_edges, 1);
        assert_eq!(outcome.contexts[1].outcome.dead_edges, 1);

        let _ = fs::remove_dir_all(dir);
    }

    /// A cold context's dead ratio comes from its persisted stats
    /// snapshot, not a load — [`AppState::run_maintenance_compaction`]'s
    /// whole point is picking candidates without paying for residency.
    #[test]
    fn run_maintenance_compaction_selects_a_cold_candidate_from_its_saved_stats() {
        let dir = scratch_dir("maint-cold");
        let state = AppState::boot(dir.clone(), 1, None).unwrap();

        state
            .create("rotten", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "rotten",
                vec![
                    assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md")),
                    assoc_op("蔵", "廃止銘柄", "旧銘", 1.0, Some("gone.md")),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state.retract_source("rotten", "gone.md").unwrap();

        state
            .create("other", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // Touching "other" evicts "rotten" to cold under the one-byte budget.
        state
            .read_context("other", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert!(
            !state.directory_entry("rotten").unwrap().loaded,
            "rotten must be cold before the sweep"
        );

        let outcome = state.run_maintenance_compaction(0.0, Deadline::unbounded());
        assert_eq!(outcome.contexts.len(), 1, "{outcome:?}");
        assert_eq!(outcome.contexts[0].name, "rotten");
        assert_eq!(outcome.contexts[0].outcome.dead_edges, 1);
        assert!(!outcome.deadline_exceeded);

        let _ = fs::remove_dir_all(dir);
    }

    /// A `Slot::Deleted` entry observed inside the sweep (the tombstone
    /// left for anyone holding the entry's `Arc` from before a concurrent
    /// `delete`) is skipped, not treated as a crash or a candidate.
    #[test]
    fn run_maintenance_compaction_skips_a_deleted_entry_without_panicking() {
        let dir = scratch_dir("maint-deleted");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("ghost", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "ghost",
                vec![assoc_op("蔵", "廃止銘柄", "旧銘", 1.0, Some("gone.md"))],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state.retract_source("ghost", "gone.md").unwrap();

        // Simulates the race `delete()` can open: still a member of the
        // registry map (so the sweep's snapshot picks it up) but its
        // slot already flipped to the tombstone, as if a concurrent
        // `delete()` had reached that half of its two-step teardown.
        let entry = state.lookup("ghost").expect("just created");
        entry.inner.write().slot = Slot::Deleted;

        let outcome = state.run_maintenance_compaction(0.0, Deadline::unbounded());
        assert!(outcome.contexts.is_empty(), "{outcome:?}");
        assert!(!outcome.deadline_exceeded);

        let _ = fs::remove_dir_all(dir);
    }

    /// [`AppState::try_enter_maintenance`] is a one-shot CAS: a second
    /// call fails while the guard lives, and dropping it (the only way
    /// to release — success, a deadline, or a panic unwind all reach the
    /// same `Drop`) reopens the server for the next sweep.
    #[test]
    fn maintenance_guard_is_a_one_shot_cas_released_by_drop() {
        let dir = scratch_dir("maint-guard");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

        let guard = state.try_enter_maintenance().expect("first entry succeeds");
        assert!(state.metrics().maintenance_active());
        assert!(
            state.try_enter_maintenance().is_none(),
            "a second sweep must not overlap the first"
        );

        drop(guard);
        assert!(!state.metrics().maintenance_active());
        let _second = state
            .try_enter_maintenance()
            .expect("reopened once the guard drops");

        let _ = fs::remove_dir_all(dir);
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
                    Deadline::unbounded(),
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
        // 2, not 1: evicting "a" above already flushed it (evict_entry
        // persists via flush_entry so its own write lock is never held
        // across the serialize+fsync — see evict_entry's doc comment),
        // and this flush_dirty tick adds "b"'s.
        assert!(
            rendered(&state).contains("taguru_flush_total{outcome=\"ok\"} 2"),
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
                _deadline: Deadline,
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
            state
                .refresh_embeddings("fruit", Deadline::unbounded())
                .unwrap()
                .unwrap();
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
                .semantic_resolve("fruit", "アップル", false, None, Deadline::unbounded())
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

    /// `semantic_resolve` deliberately folds provider-off, model-changed,
    /// and nothing-embedded into one empty answer; its explain twin must
    /// hold them apart, and must place an expected name in exactly the
    /// ordering `semantic_resolve` truncates.
    #[test]
    fn explain_semantic_resolve_names_what_semantic_resolve_folds() {
        let dir = scratch_dir("sem-explain");
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

            // Before any refresh: nothing is embedded, and the report
            // says that — not "empty answer", not "model changed".
            assert!(matches!(
                state
                    .explain_semantic_resolve(
                        "fruit",
                        "アップル",
                        "りんご",
                        false,
                        None,
                        Deadline::unbounded()
                    )
                    .unwrap(),
                GlossLaneReport::EmptyTable
            ));

            state
                .refresh_embeddings("fruit", Deadline::unbounded())
                .unwrap()
                .unwrap();

            // The expected name's own cosine and its rank in the very
            // ordering semantic_resolve serves.
            let Some(GlossLaneReport::Ran {
                floor,
                cosine: Some(cosine),
                rank,
                passing,
                cap,
            }) = state.explain_semantic_resolve(
                "fruit",
                "アップル",
                "りんご",
                false,
                None,
                Deadline::unbounded(),
            )
            else {
                panic!("the sweep should have run with a cosine for りんご");
            };
            assert!((cosine - 0.96).abs() < 1e-6);
            assert_eq!(rank, Some(1));
            assert_eq!(passing, 1, "果物's cosine 0.0 sits under the floor");
            assert_eq!(cap, SEMANTIC_RESOLVE_LIMIT);
            assert!(floor > 0.0);

            // A below-floor name reports its cosine with no rank — the
            // "scored 0.0, floor 0.35" evidence — and a floor override
            // seats it, in semantic_resolve's exact order.
            let Some(GlossLaneReport::Ran {
                cosine: Some(low),
                rank: None,
                ..
            }) = state.explain_semantic_resolve(
                "fruit",
                "アップル",
                "果物",
                false,
                None,
                Deadline::unbounded(),
            )
            else {
                panic!("果物 has a vector; its cosine must be reported");
            };
            assert!(low.abs() < 1e-6);
            let Some(GlossLaneReport::Ran {
                rank: Some(rank),
                passing,
                ..
            }) = state.explain_semantic_resolve(
                "fruit",
                "アップル",
                "果物",
                false,
                Some(0.0),
                Deadline::unbounded(),
            )
            else {
                panic!("floor 0.0 must seat 果物");
            };
            assert_eq!((rank, passing), (2, 2));
            let served = state
                .semantic_resolve("fruit", "アップル", false, Some(0.0), Deadline::unbounded())
                .unwrap()
                .unwrap();
            assert_eq!(
                served[rank - 1].0,
                "果物",
                "rank must match the serve order"
            );

            // A name added after the refresh has no vector yet: the
            // sweep runs, its cosine does not exist.
            state
                .write_context("fruit", |context| {
                    context.associate("バナナ", "分類", "果物", 1.0).unwrap();
                })
                .map_err(|_| "write")
                .unwrap();
            assert!(matches!(
                state
                    .explain_semantic_resolve(
                        "fruit",
                        "アップル",
                        "バナナ",
                        false,
                        None,
                        Deadline::unbounded(),
                    )
                    .unwrap(),
                GlossLaneReport::Ran { cosine: None, .. }
            ));
            state.flush_dirty();
        }

        // Same sidecar, another model: named as the reason.
        struct OtherEmbeddings;
        impl EmbeddingProvider for OtherEmbeddings {
            fn model(&self) -> &str {
                "other-model"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0]).collect())
            }
        }
        let state = AppState::boot(
            dir.clone(),
            usize::MAX,
            Some(Arc::new(OtherEmbeddings) as Arc<dyn EmbeddingProvider>),
        )
        .unwrap();
        assert!(matches!(
            state
                .explain_semantic_resolve(
                    "fruit",
                    "アップル",
                    "りんご",
                    false,
                    None,
                    Deadline::unbounded(),
                )
                .unwrap(),
            GlossLaneReport::ModelChanged { .. }
        ));
        // A context that does not exist is the outer None — but only
        // once a provider exists to get past the Off arm.
        assert!(
            state
                .explain_semantic_resolve(
                    "nazo",
                    "アップル",
                    "りんご",
                    false,
                    None,
                    Deadline::unbounded(),
                )
                .is_none()
        );

        // No provider at all: Off before any lookup, exactly where
        // semantic_resolve answers its empty list. (Shadowing keeps the
        // previous state — and its data-dir lock — alive to scope end,
        // so release it by hand.)
        drop(state);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(matches!(
            state
                .explain_semantic_resolve(
                    "fruit",
                    "アップル",
                    "りんご",
                    false,
                    None,
                    Deadline::unbounded(),
                )
                .unwrap(),
            GlossLaneReport::Off
        ));

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
                    Deadline::unbounded(),
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
                    Deadline::unbounded(),
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
                    Deadline::unbounded(),
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
                Deadline::unbounded(),
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
                    Deadline::unbounded(),
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
                    Deadline::unbounded(),
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
            .add_associations(
                "sake",
                vec![assoc_op("a", "l", "b", 1.0, None)],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        let refused = state.add_associations(
            "sake",
            vec![assoc_op("c", "l", "d", 1.0, None)],
            Deadline::unbounded(),
        );
        assert!(
            matches!(refused, Err(AccessError::Unpersisted(_))),
            "over the cap the write must be refused: {refused:?}"
        );

        let _ = fs::remove_dir_all(dir);
        let _ = fs::remove_dir_all(capped_dir);
    }

    #[test]
    fn dead_weight_gauges_track_hot_and_cold_contexts() {
        let dir = scratch_dir("dead-weight-gauges");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        let baseline = state.gauge_snapshot();
        assert_eq!(baseline.dead_edges_total, 0);
        assert_eq!(baseline.dead_attributions_total, 0);
        assert_eq!(baseline.arena_slack_total, 0);
        assert_eq!(baseline.unsourced_edges_total, 0);
        assert_eq!(baseline.unsourced_weight_total, 0.0);

        // One sourced association, later retracted outright: one dead
        // edge, one unlinked attribution.
        state
            .add_associations(
                "sake",
                vec![assoc_op("蔵", "廃", "旧", 1.0, Some("x.md"))],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            state.retract_association("sake", "蔵", "廃", "旧").unwrap(),
            Some(1)
        );
        // One sourceless association: pure unsourced weight, nothing else.
        state
            .add_associations(
                "sake",
                vec![assoc_op("蔵", "銘柄", "青嶺", 2.5, None)],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        // One alias registered then withdrawn: its spelling's bytes
        // become arena slack.
        state
            .add_aliases(
                "sake",
                &BTreeMap::from([("Aomine".to_string(), "蔵".to_string())]),
                &BTreeMap::new(),
            )
            .unwrap()
            .unwrap();
        state
            .remove_aliases("sake", &["Aomine".to_string()], &[])
            .unwrap()
            .unwrap();

        let hot = state.gauge_snapshot();
        assert_eq!(hot.dead_edges_total, 1);
        assert_eq!(hot.dead_attributions_total, 1);
        assert_eq!(hot.arena_slack_total, "Aomine".len() as u64);
        assert_eq!(hot.unsourced_edges_total, 1);
        assert_eq!(hot.unsourced_weight_total, 2.5);
        let text = rendered(&state);
        assert!(text.contains("taguru_dead_edges 1"));
        assert!(text.contains("taguru_dead_attributions 1"));
        assert!(text.contains(&format!("taguru_arena_slack_bytes {}", "Aomine".len())));
        assert!(text.contains("taguru_unsourced_edges 1"));
        assert!(text.contains("taguru_unsourced_weight 2.5"));

        // Eviction to cold must not lose the totals — the gauge falls
        // back to the persisted `ContextStats` snapshot.
        let entry = state.lookup("sake").unwrap();
        assert!(state.evict_entry("sake", &entry));
        let cold = state.gauge_snapshot();
        assert_eq!(cold.dead_edges_total, hot.dead_edges_total);
        assert_eq!(cold.dead_attributions_total, hot.dead_attributions_total);
        assert_eq!(cold.arena_slack_total, hot.arena_slack_total);
        assert_eq!(cold.unsourced_edges_total, hot.unsourced_edges_total);
        assert_eq!(cold.unsourced_weight_total, hot.unsourced_weight_total);

        let _ = fs::remove_dir_all(dir);
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
                Deadline::unbounded(),
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
                Deadline::unbounded(),
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
                                Deadline::unbounded(),
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
                Deadline::unbounded(),
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
                Deadline::unbounded(),
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
                Deadline::unbounded(),
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
                        Deadline::unbounded(),
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
            let hold = decoy.inner.write();
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
                    Deadline::unbounded(),
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
                    Deadline::unbounded(),
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
                    Deadline::unbounded(),
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
                    Deadline::unbounded(),
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
            Deadline::unbounded(),
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
                    Deadline::unbounded(),
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
                                Deadline::unbounded(),
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

    /// With the WAL on, a flush that staged its image before an eviction
    /// cooled the entry — then a later write reloaded it Hot again before
    /// the flush re-locked — used to pass both re-validation checks (slot
    /// is Hot again; eviction never bumped `image_generation`) and
    /// publish its stale snapshot, regressing the on-disk image past
    /// whatever the eviction had already persisted and truncated the WAL
    /// for. Driven directly (`flush_entry` and `evict_entry`, the same
    /// trick as the delete-race test above) instead of hoping a thrash
    /// loop stumbles into the exact interleaving: spinning on `flushing`
    /// forces the claim (and thus the snapshot the flush later
    /// republishes) to land before the write below — `thread::spawn`
    /// scheduling latency otherwise routinely loses that race to this
    /// thread's very next line, and a big seed keeps the flush busy
    /// serializing and staging afterward, which is what gives the write,
    /// the eviction, and the reload below the room to land before it
    /// re-locks to publish.
    #[test]
    fn an_eviction_racing_a_reload_and_a_stale_flush_never_regresses_the_image() {
        use std::thread;
        use std::time::{Duration, Instant};

        let dir = scratch_dir("evict-reload-flush-race");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        // Big enough that `flush_entry`'s serialize-under-lock and
        // stage-unlocked steps together take long enough for the write,
        // the direct eviction, and the reload below to all land before
        // it re-locks — with a tiny graph the flush publishes before any
        // of them get a look in, and the race never opens.
        let seed: Vec<_> = (0..50_000)
            .map(|index| {
                assoc_op(
                    &format!("seed-subject-{index}-xxxxxxxxxxxxxxxxxxxx"),
                    "seed-label-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    &format!("seed-object-{index}-xxxxxxxxxxxxxxxxxxxx"),
                    1.0,
                    None,
                )
            })
            .collect();
        state
            .add_associations("sake", seed, Deadline::unbounded())
            .unwrap()
            .unwrap();

        let entry = state.lookup("sake").unwrap();
        let flusher = {
            let state = state.clone();
            let entry = Arc::clone(&entry);
            thread::spawn(move || state.flush_entry("sake", &entry))
        };
        // `flushing` only flips true once `flush_entry` has locked the
        // entry and entered its claim section, and only flips back at
        // the very end of the call — spinning on it here proves the
        // claim landed before we go on to write "b" below. Without this,
        // `thread::spawn` scheduling latency routinely lets that write
        // land first instead, folding "b" into the flush's own snapshot
        // and closing off the interleaving this test drives at.
        let spun_since = Instant::now();
        while !entry.flushing.load(Ordering::Relaxed) {
            assert!(
                spun_since.elapsed() < Duration::from_secs(5),
                "flusher never reached its claim section"
            );
            thread::yield_now();
        }
        // `flush_entry` claims (clears `dirty`) and serializes the seeded
        // image UNDER the entry lock before it stages unlocked, so this
        // call blocks until that claim is made — it cannot land before
        // the flush captured its (soon to be stale) snapshot.
        state
            .add_associations(
                "sake",
                vec![assoc_op("b", "l", "o", 1.0, None)],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        // Queue the reload-triggering write on the entry lock now, right
        // as the eviction below begins — behind the eviction (which,
        // called directly here, reliably wins the very next acquisition
        // over a freshly spawned thread, the same head start `flushing`'s
        // spin-wait above relies on) but ahead of the flusher's own
        // re-lock attempt, which only comes later once its slower
        // `stage_bytes` finishes. Without this, the flusher — parked on
        // the lock since partway through the eviction below — is
        // `parking_lot`'s fairness-guaranteed next owner the instant the
        // eviction releases it, and it sees Cold (this write not yet
        // applied) instead of the reloaded Hot the bug needs.
        let reloader = {
            let state = state.clone();
            thread::spawn(move || {
                state.add_associations(
                    "sake",
                    vec![assoc_op("c", "l", "o", 1.0, None)],
                    Deadline::unbounded(),
                )
            })
        };
        assert!(
            state.evict_entry("sake", &entry),
            "the write above must leave `sake` dirty and evictable"
        );
        reloader.join().unwrap().unwrap().unwrap();
        let published = flusher.join().unwrap();

        const EXPECTED: usize = 50_000 + 2; // the seed, plus "b" and "c"
        let live = state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(live, EXPECTED, "the race lost a write in memory");
        drop(state);

        let reborn = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let recovered = reborn
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(
            recovered, EXPECTED,
            "a stale flush (published: {published}) regressed the image past \
             what the eviction had already made durable"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn evict_entry_does_not_hold_the_entry_lock_across_its_serialize_and_fsync() {
        use std::thread;
        use std::time::{Duration, Instant};

        let dir = scratch_dir("evict-lock-free-io");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("big", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        // Big enough that flush_entry's stage (write + fsync) step, which
        // evict_entry now goes through instead of saving under its own
        // lock, takes long enough for the try_write below to land while
        // it's still in flight — with a tiny graph the flush publishes
        // before this thread gets a look in and the assertion proves
        // nothing either way.
        let seed: Vec<_> = (0..50_000)
            .map(|index| {
                assoc_op(
                    &format!("seed-subject-{index}-xxxxxxxxxxxxxxxxxxxx"),
                    "seed-label-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    &format!("seed-object-{index}-xxxxxxxxxxxxxxxxxxxx"),
                    1.0,
                    None,
                )
            })
            .collect();
        state
            .add_associations("big", seed, Deadline::unbounded())
            .unwrap()
            .unwrap();

        let entry = state.lookup("big").unwrap();
        let evictor = {
            let state = state.clone();
            let entry = Arc::clone(&entry);
            thread::spawn(move || state.evict_entry("big", &entry))
        };
        // flush_entry's claim section sets `flushing` AND serializes the
        // graph (context.to_bytes()) before it releases the lock, so
        // spinning on `flushing` alone would just as often land while
        // the lock is still held for that serialize step. Poll instead:
        // the lock must open up SOME time before the evictor finishes —
        // that's the stage_bytes (write + fsync) window, exactly what
        // the old evict_entry held its own write lock across instead,
        // stalling every reader and writer of "big" for as long as the
        // serialize+fsync took.
        let poll_since = Instant::now();
        let mut saw_free_lock = false;
        while !evictor.is_finished() {
            if entry.inner.try_write().is_some() {
                saw_free_lock = true;
                break;
            }
            assert!(
                poll_since.elapsed() < Duration::from_secs(5),
                "evict_entry seems to hold the entry lock for the whole eviction"
            );
            thread::yield_now();
        }
        assert!(
            saw_free_lock,
            "evict_entry must not hold the entry lock across its serialize+fsync \
             (never observed it open up before the evictor finished — if this \
             gets flaky, grow the seed so eviction takes longer)"
        );

        assert!(
            evictor.join().unwrap(),
            "the seeded write must have been evictable"
        );

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
                            Deadline::unbounded(),
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
                    let _ = state.compact_context("sake", Deadline::unbounded());
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

    /// A flush has image stage/commit, sidecar stage/commit, and a WAL
    /// truncate. Fail each in turn: publish failures stay dirty and
    /// retry, while a truncate failure is harmless because the image
    /// watermark makes the retained log replay-inert.
    #[test]
    fn every_flush_persistence_failure_retries_or_replays_cleanly() {
        let mut exhausted = false;
        for failure in 0..16 {
            let dir = scratch_dir(&format!("flush-fault-{failure}"));
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("doc"))],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();

            fail_persistence_ops_after(failure);
            let first = state.flush_dirty();
            let past_end = clear_persistence_fault();
            if !first.iter().any(|name| name == "sake") {
                assert!(
                    state.flush_dirty().iter().any(|name| name == "sake"),
                    "dirty retry did not flush after failure at step {failure}"
                );
            }
            drop(state);

            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            assert_eq!(
                state
                    .read_context("sake", |context| context.association_count())
                    .unwrap(),
                1,
                "flush failure at step {failure} lost or duplicated the write"
            );
            drop(state);
            let _ = fs::remove_dir_all(&dir);

            if past_end {
                exhausted = true;
                break;
            }
        }
        assert!(exhausted, "flush exceeded the sweep bound");
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
    fn spelled_terms_are_the_hashed_terms_with_their_spellings() {
        for text in [
            "精米歩合は50%まで磨く。",
            "AppState boot 水、源 第10篇 ＡpplePie",
            "HTTPServer U64Max flush_dirty",
        ] {
            let spelled = spelled_passage_terms(text);
            // Same walker, only the sink differs: the hash stream is
            // passage_terms verbatim.
            assert_eq!(
                spelled.iter().map(|(_, term)| *term).collect::<Vec<_>>(),
                passage_terms(text),
                "spelled and bare terms diverged ({text})"
            );
            // Every spelling is exactly what was hashed, so hashing the
            // spelling alone finds the same term again.
            for (spelling, term) in &spelled {
                assert_eq!(
                    passage_terms(spelling).first(),
                    Some(term),
                    "a spelling must hash back to its own term ({spelling:?} in {text:?})"
                );
            }
        }
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

    /// A failed create must release its `pending.creates` reservation —
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
        let stall = entry.inner.read();
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
            entry.passages.lock().is_none(),
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
            .search_passages("sake", "精米歩合はどこまで磨く?", 3, Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(hits[0].source, "第2段落");
        assert!(hits[0].score > 0.0);

        // No shared bigrams at all → nothing, not noise.
        assert!(
            state
                .search_passages("sake", "unrelated english words", 3, Deadline::unbounded())
                .unwrap()
                .unwrap()
                .is_empty()
        );
        assert!(
            state
                .search_passages("nope", "x", 3, Deadline::unbounded())
                .is_none()
        );

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
            .search_passages(
                "papers",
                "ambition must be made to counteract ambition",
                2,
                Deadline::unbounded(),
            )
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
            let hits = state
                .search_passages("code", query, 3, Deadline::unbounded())
                .unwrap()
                .unwrap();
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
            .search_passages("sake", "精米歩合はどこまで磨く?", 3, Deadline::unbounded())
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
                .search_passages("sake", "創業はいつ", 3, Deadline::unbounded())
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
            .search_passages("sake", "杜氏の出身", 3, Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(hits[0].source, "第2章");

        // And a retraction disappears the same way.
        state.retract_source("sake", "第2章").unwrap();
        assert!(
            state
                .search_passages("sake", "杜氏の出身", 3, Deadline::unbounded())
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
                .search_passages("sake", "創業はいつ", 3, Deadline::unbounded())
                .unwrap()
                .unwrap();
            state.flush_dirty();
            assert!(bm25_path(&dir, &file_stem("sake")).exists());
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let hits = state
            .search_passages("sake", "創業はいつ", 3, Deadline::unbounded())
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
            state
                .search_passages("sake", "杜氏", 3, Deadline::unbounded())
                .unwrap()
                .unwrap();
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
            .search_passages("sake", "杜氏は誰", 3, Deadline::unbounded())
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
            .search_passages("sake", "蔵開きの祭り", 3, Deadline::unbounded())
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
            .search_passages("sake", "麹室の湿度", 3, Deadline::unbounded())
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
            .search_passages("sake", "麹室の湿度", 3, Deadline::unbounded())
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
                .search_passages("sake", "蔵開きの祭り", 3, Deadline::unbounded())
                .unwrap()
                .unwrap()
                .is_empty()
        );

        let entry = state.lookup("sake").unwrap();
        assert!(state.evict_entry("sake", &entry));
        assert!(
            entry.bm25.read().is_none(),
            "eviction must drop the resident index"
        );
        assert_eq!(
            state
                .search_passages("sake", "蔵開きの祭り", 3, Deadline::unbounded())
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

        let outcome = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
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

        // Unchanged corpus: nothing re-embeds. The one provider call is
        // the width probe that guards against a silent backend swap (a
        // changed vector width behind an unchanged model name), the same
        // one-embedding-per-no-op cost the gloss refresh pays.
        let again = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!((again.embedded, again.total), (0, 3));
        assert_eq!(calls.load(Ordering::Relaxed), 2);

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
        state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let mut updated = BTreeMap::new();
        updated.insert(
            "doc-a".to_string(),
            "変わらない段落。\n\n新しい版の段落。".to_string(),
        );
        state
            .store_passages("sake", plain(updated))
            .unwrap()
            .unwrap();
        let outcome = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
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
        state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();

        state.retract_source("sake", "doc-b").unwrap();
        let outcome = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (outcome.embedded, outcome.total),
            (0, 1),
            "the retracted source's row is gone without any re-embedding"
        );
        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("sake")));
        assert_eq!(sidecar.len(), 1, "the prune reached the disk too");

        let _ = fs::remove_dir_all(dir);
    }

    /// A regression for a false-negative `limit_to_reach`: a raw semantic
    /// lane rank counts every row the vector store holds for a source,
    /// stale ones included, so it can run far ahead of `full.len()` (the
    /// count that actually survives the staleness filter). `lane_need`
    /// derives from that raw rank; before the fix it sized the starting
    /// candidate past `full.len()` and the search bailed out with "no
    /// limit reaches it" without ever trying `full.len()` itself — which,
    /// being the target's own rank in `full`, always would have.
    #[test]
    fn limit_to_reach_is_not_a_false_negative_when_stale_rows_inflate_the_raw_lane_rank() {
        struct MarkerEmbeddings;
        impl EmbeddingProvider for MarkerEmbeddings {
            fn model(&self) -> &str {
                "marker"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                Ok(texts
                    .iter()
                    .map(|text| {
                        if text.starts_with("TARGETMARKER") {
                            vec![0.5, 0.8660254]
                        } else {
                            vec![1.0, 0.0]
                        }
                    })
                    .collect())
            }
        }

        let dir = scratch_dir("limit-to-reach-stale");
        let state = boot_for_passage_embedding(&dir, Arc::new(MarkerEmbeddings), 20_000);
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        let mut passages = BTreeMap::new();
        passages.insert("fresh-doc".to_string(), "FRESHMARKER".to_string());
        passages.insert("target-doc".to_string(), "TARGETMARKER".to_string());
        for i in 0..15 {
            passages.insert(format!("decoy-{i:02}"), "DECOYMARKER".to_string());
        }
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();

        // Edit every decoy without re-embedding: their vector rows stay
        // in the store at the old hash, tied with `fresh-doc` at the top
        // of the raw cosine sweep, but staleness drops all 15 out of
        // `full` — so they inflate the target's raw rank without ever
        // occupying a `full` slot themselves.
        let mut edited = BTreeMap::new();
        for i in 0..15 {
            edited.insert(format!("decoy-{i:02}"), "DECOYMARKER-EDITED".to_string());
        }
        state
            .store_passages("sake", plain(edited))
            .unwrap()
            .unwrap();

        let explanation = state
            .explain_passage_search(
                "sake",
                "QUERYMARKER",
                "target-doc",
                None,
                1,
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        let PassageExplainLookup::Explained(explanation) = explanation else {
            panic!("expected an Explained verdict");
        };
        assert!(
            !explanation.served,
            "fresh-doc alone fills the one requested seat"
        );
        assert_eq!(
            explanation.ranked, 2,
            "only fresh-doc and target-doc survive the staleness filter"
        );
        assert_eq!(
            explanation.limit_to_reach,
            Some(2),
            "full.len() (2) is itself a valid, untried candidate — it must not \
             be skipped as unreachable just because lane_need overshoots it"
        );

        let hits = state
            .search_passages("sake", "QUERYMARKER", 2, Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert!(
            hits.iter().any(|hit| hit.source == "target-doc"),
            "limit_to_reach must actually reach it: {hits:?}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// A provider that changes output width behind a stable model name (a
    /// backend swap behind the same proxy) must stale the whole carried
    /// passage table, exactly as the gloss refresh does: paragraph hashes
    /// are unchanged, so without a width check old-width rows would pin
    /// the store's dimension and `PassageVectorStore::push` would drop
    /// every new-width row this pass embeds — silently, at a warn.
    #[test]
    fn a_passage_width_change_under_the_same_model_name_re_embeds_everything() {
        struct WidthEmbeddings(Arc<std::sync::atomic::AtomicUsize>);
        impl EmbeddingProvider for WidthEmbeddings {
            fn model(&self) -> &str {
                "stable-name"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                let width = self.0.load(Ordering::Relaxed);
                Ok(texts
                    .iter()
                    .map(|_| {
                        let mut vector = vec![0.0; width];
                        vector[0] = 1.0;
                        vector
                    })
                    .collect())
            }
        }

        let dir = scratch_dir("pvec-width");
        let width = Arc::new(std::sync::atomic::AtomicUsize::new(2));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(WidthEmbeddings(Arc::clone(&width))), 20_000);
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "最初の段落。".to_string());
        passages.insert("doc-b".to_string(), "二番目の段落。".to_string());
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        let first = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!((first.embedded, first.total), (2, 2));
        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("sake")));
        assert!(
            sidecar.iter().all(|(_, row)| row.len() == 2),
            "the first pass stored 2-dim rows"
        );

        // Probe path: same passages (every hash carried, nothing else
        // reveals the width) but wider vectors. One probe embedding must
        // catch the change and re-embed every row, or the store keeps its
        // stale 2-dim rows against a provider now speaking 3.
        width.store(3, Ordering::Relaxed);
        let widened = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (widened.embedded, widened.total),
            (2, 2),
            "an unchanged corpus still re-embeds every row on a width change"
        );
        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("sake")));
        assert_eq!(sidecar.len(), 2);
        assert!(
            sidecar.iter().all(|(_, row)| row.len() == 3),
            "every stored row is the new width — none dropped, none left at the old one"
        );

        // Embedded-rows path: a width change that rides alongside an edit
        // is caught from the freshly embedded rows directly, no probe. The
        // one unchanged paragraph must not survive at the old width.
        width.store(4, Ordering::Relaxed);
        let mut edited = BTreeMap::new();
        edited.insert("doc-a".to_string(), "改訂された段落。".to_string());
        edited.insert("doc-b".to_string(), "二番目の段落。".to_string());
        state
            .store_passages("sake", plain(edited))
            .unwrap()
            .unwrap();
        let mixed = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (mixed.embedded, mixed.total),
            (2, 2),
            "the edit's new width stales the carried row too"
        );
        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("sake")));
        assert_eq!(sidecar.len(), 2);
        assert!(
            sidecar.iter().all(|(_, row)| row.len() == 4),
            "the carried old-width row was re-embedded, not dropped"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// `PassageVectorStore::push` already drops whichever chunk lands at
    /// a width disagreeing with the dimension the store already
    /// settled on — but the merge loop counted every attempted push as
    /// `embedded` regardless, over-reporting work that `push` silently
    /// threw away. `embedded` must count only rows that actually landed.
    #[test]
    fn passage_refresh_reports_an_accurate_embedded_count_when_a_chunk_disagrees_on_width() {
        struct SplitWidthEmbeddings;
        impl EmbeddingProvider for SplitWidthEmbeddings {
            fn model(&self) -> &str {
                "split-width"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                // The full 128-paragraph chunk answers at width 2; the
                // trailing 44-paragraph chunk answers at width 3 — a
                // provider mid-migration serving two backend versions to
                // concurrent connections.
                let width = if texts.len() >= 128 { 2 } else { 3 };
                Ok(texts.iter().map(|_| vec![0.0; width]).collect())
            }
        }

        let dir = scratch_dir("pvec-split-width");
        let embedder = Arc::new(SplitWidthEmbeddings) as Arc<dyn EmbeddingProvider>;
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            Some(embedder),
            BootOptions {
                embed_passages: true,
                passage_vector_limit: 20_000,
                embed_parallel: 2,
                ..BootOptions::default()
            },
        )
        .unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // 172 paragraphs = a 128-item chunk plus a 44-item remainder.
        let text = (0..172)
            .map(|i| format!("段落その{i}。"))
            .collect::<Vec<_>>()
            .join("\n\n");
        let mut passages = BTreeMap::new();
        passages.insert("doc-big".to_string(), text);
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        let outcome = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (outcome.embedded, outcome.total),
            (128, 128),
            "the disagreeing trailing chunk is dropped, not merely undercounted"
        );
        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("sake")));
        assert!(sidecar.iter().all(|(_, row)| row.len() == 2));

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

        let outcome = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
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
                _deadline: Deadline,
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
            .refresh_passage_embeddings("sake", Deadline::unbounded())
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
        let outcome = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!((outcome.embedded, outcome.total), (1, 129));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    #[cfg(unix)]
    fn refresh_passage_embeddings_does_not_rebuy_rows_a_failed_save_already_bought() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("pvec-save-fail");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごの段落。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();

        // The disk goes bad right before the save: the provider still
        // gets paid (embed happens before the write), but the sidecar
        // write fails.
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
        let error = state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap_err();
        assert!(error.contains("not persisted"), "{error}");
        let calls_after_failure = calls.load(Ordering::Relaxed);
        assert!(
            calls_after_failure > 0,
            "the provider must have been paid before the save failed"
        );
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();

        // The disk recovers: the retry must not re-embed the row the
        // failed save already bought (a width probe still spends one
        // call on a no-op refresh, same as any other — see
        // a_width_change_under_the_same_model_name_re_embeds_everything),
        // yet it must still retry the write so the row does not stay
        // unpersisted forever.
        let outcome = state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            outcome.embedded, 0,
            "must not re-embed what the failed save already cached"
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            calls_after_failure + 1,
            "only the width probe's one call, not a re-embed of the cached row"
        );

        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("fruit")));
        assert_eq!(
            sidecar.len(),
            outcome.total,
            "the retried save must have actually landed on disk"
        );
        assert!(outcome.total > 0);

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
        let outcome = state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (outcome.embedded, outcome.total),
            (2, 2),
            "the paragraph row and its question row both embed"
        );

        // The query matches the QUESTION row's wording, not the text's;
        // both rows point at the same paragraph, so the lane must fold
        // them into one hit at the question row's better rank.
        let hits = state
            .search_passages("fruit", "アップル", 3, Deadline::unbounded())
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
        state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        state
            .store_passages("fruit", submission("アップルは何色ですか?"))
            .unwrap()
            .unwrap();
        let outcome = state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
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
        let outcome = state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
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
        state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages("fruit", "アップル", 3, Deadline::unbounded())
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
        state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages("fruit", "りんごは真っ赤", 3, Deadline::unbounded())
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
        state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

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
            .search_passages("fruit", "りんご", 3, Deadline::unbounded())
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
                _deadline: Deadline,
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
        state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages("fruit", "りんご", 3, Deadline::unbounded())
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
            .search_passages("fruit", "りんご", 3, Deadline::unbounded())
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
        state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages("fruit", "みかん", 3, Deadline::unbounded())
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
        state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        state
            .update_meta("fruit", None, None, None, Some(0.2))
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages("fruit", "みかん", 3, Deadline::unbounded())
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
            .refresh_passage_embeddings("sake", Deadline::unbounded())
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

        fn embed(
            &self,
            texts: &[&str],
            _purpose: EmbedPurpose,
            _deadline: Deadline,
        ) -> Result<Vec<Vec<f32>>, String> {
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

    /// A provider whose `embed` call blocks for 150ms while tracking how
    /// many calls are in flight at once (and the peak concurrency
    /// observed) — long enough that concurrent calls MUST overlap unless
    /// something serializes or gates them. Shared by the refresh
    /// concurrency tests below.
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
            _deadline: Deadline,
        ) -> Result<Vec<Vec<f32>>, String> {
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_millis(150));
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
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
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                self.0.lock().push(purpose);
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
        state
            .refresh_embeddings("p", Deadline::unbounded())
            .unwrap()
            .unwrap();
        state
            .semantic_resolve("p", "cue", false, None, Deadline::unbounded())
            .unwrap()
            .unwrap();

        let seen = purposes.lock().clone();
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
                _deadline: Deadline,
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
            let (embedded, total) = state
                .refresh_embeddings("w", Deadline::unbounded())
                .unwrap()
                .unwrap();
            assert_eq!((embedded, total), (3, 3)); // a, b, and the label l
            state.flush_dirty();
        }

        // Same model name, wider vectors: every gloss must re-embed
        // (hashes alone would say "nothing to do") and the published
        // sidecar must be uniformly the new width.
        let embedder = Some(Arc::new(WidthEmbeddings(3)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        let (embedded, total) = state
            .refresh_embeddings("w", Deadline::unbounded())
            .unwrap()
            .unwrap();
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
        let (embedded, total) = state
            .refresh_embeddings("w", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!((embedded, total), (0, 3));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_width_drift_confined_to_the_label_table_is_still_caught() {
        struct FixedWidthEmbeddings(usize);
        impl EmbeddingProvider for FixedWidthEmbeddings {
            fn model(&self) -> &str {
                "stable-name"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
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

        let dir = scratch_dir("label-only-width-drift");
        let path = vectors_path(&dir, &file_stem("w"));
        {
            let embedder = Some(Arc::new(FixedWidthEmbeddings(3)) as Arc<dyn EmbeddingProvider>);
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
            state
                .refresh_embeddings("w", Deadline::unbounded())
                .unwrap()
                .unwrap();
            state.flush_dirty();
        }

        // Shrink only the label table's vectors in place, keeping their
        // hash unchanged — the shape a width change confined to one
        // table (a partial backend rollout, or a prior pass that only
        // reconciled concepts) would leave on disk. `carried_width`
        // sampling concepts first — as this used to — would see
        // concepts already at width 3, call that "no drift", and never
        // look at labels at all.
        let mut store = VectorStore::load(&path);
        for (_, vector) in store.labels.values_mut() {
            vector.truncate(2);
        }
        store.save(&path).unwrap();

        // Same model name, same provider width (3): a no-op content
        // diff, so nothing re-embeds and only the probe/reconciliation
        // path can notice the label table is still stuck at width 2.
        let embedder = Some(Arc::new(FixedWidthEmbeddings(3)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        let (embedded, total) = state
            .refresh_embeddings("w", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (embedded, total),
            (3, 3),
            "a width drift confined to the label table must still force a full re-embed"
        );
        let reloaded = VectorStore::load(&path);
        assert!(
            reloaded
                .concepts
                .values()
                .chain(reloaded.labels.values())
                .all(|(_, vector)| vector.len() == 3),
            "the label table's stale width must not survive reconciliation"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    #[cfg(unix)]
    fn refresh_embeddings_does_not_rebuy_rows_a_failed_save_already_bought() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("gvec-save-fail");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let embedder = Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .write_context("fruit", |context| {
                context.associate("りんご", "l", "アップル", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();

        // The disk goes bad right before the save: the provider still
        // gets paid (embed happens before the write), but the sidecar
        // write fails.
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
        let error = state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap_err();
        assert!(error.contains("not persisted"), "{error}");
        let calls_after_failure = calls.load(Ordering::Relaxed);
        assert!(
            calls_after_failure > 0,
            "the provider must have been paid before the save failed"
        );
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();

        // The disk recovers: the retry must not re-embed the rows the
        // failed save already bought (a width probe still spends one
        // call on a no-op refresh, same as any other — see
        // a_width_change_under_the_same_model_name_re_embeds_everything),
        // yet it must still retry the write so those rows do not stay
        // unpersisted forever.
        let (embedded, total) = state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            embedded, 0,
            "must not re-embed what the failed save already cached"
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            calls_after_failure + 1,
            "only the width probe's one call, not a re-embed of the cached rows"
        );

        let store = VectorStore::load(&vectors_path(&dir, &file_stem("fruit")));
        assert_eq!(
            store.concepts.len() + store.labels.len(),
            total,
            "the retried save must have actually landed on disk"
        );
        assert!(total > 0);

        let _ = fs::remove_dir_all(dir);
    }

    /// Chunks within one `embed_stale` call dispatch concurrently, so a
    /// provider mid-migration can answer two chunks of the very same
    /// call with different widths. `VectorTable` has no dimension of its
    /// own to enforce (unlike `PassageVectorStore`), so without a guard
    /// in the merge loop the disagreeing chunk would land right next to
    /// the rest, corrupting the persisted table with no error — just a
    /// `similarity` that silently stops matching for those rows.
    #[test]
    fn embed_stale_drops_a_chunk_whose_width_disagrees_with_the_rest_of_the_batch() {
        struct SplitWidthEmbeddings;
        impl EmbeddingProvider for SplitWidthEmbeddings {
            fn model(&self) -> &str {
                "split-width"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                // The full 128-entry chunk answers at width 2; any
                // smaller chunk (the trailing remainder, or the
                // single-label call) answers at width 3 — a provider
                // mid-migration serving two backend versions to
                // concurrent connections.
                let width = if texts.len() >= 128 { 2 } else { 3 };
                Ok(texts.iter().map(|_| vec![0.0; width]).collect())
            }
        }

        let dir = scratch_dir("gloss-split-width");
        let embedder = Some(Arc::new(SplitWidthEmbeddings) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            embedder,
            BootOptions {
                embed_parallel: 2,
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
                for i in 0..129 {
                    context
                        .associate(format!("c{i}"), "属性", "値", 1.0)
                        .unwrap();
                }
            })
            .map_err(|_| "write")
            .unwrap();

        state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let store = VectorStore::load(&vectors_path(&dir, &file_stem("fruit")));
        assert_eq!(
            store.concepts.len(),
            128,
            "the 128-item chunk lands; the remainder disagreed on width and was dropped"
        );
        assert!(
            store.concepts.values().all(|(_, v)| v.len() == 2),
            "a disagreeing vector must never reach the persisted concept table"
        );
        // Flush before the reboot below: write_context's association is
        // otherwise only durable on the next periodic flush, and the
        // reboot must see it. Then release the data-directory lock so
        // the reboot can open the same directory again.
        state.flush_dirty();
        drop(state);

        // The dropped remainder is still stale; once the provider
        // stops disagreeing with itself, the next refresh picks it up.
        struct ConsistentEmbeddings;
        impl EmbeddingProvider for ConsistentEmbeddings {
            fn model(&self) -> &str {
                "split-width"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                Ok(texts.iter().map(|_| vec![0.0; 2]).collect())
            }
        }
        let embedder = Some(Arc::new(ConsistentEmbeddings) as Arc<dyn EmbeddingProvider>);
        let state =
            AppState::boot_with(dir.clone(), usize::MAX, embedder, BootOptions::default()).unwrap();
        state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        let store = VectorStore::load(&vectors_path(&dir, &file_stem("fruit")));
        assert!(
            store.concepts.len() > 128,
            "the previously dropped remainder must still be stale and get embedded now"
        );
        assert!(store.concepts.values().all(|(_, v)| v.len() == 2));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gloss_refresh_prunes_vectors_for_a_concept_dropped_by_compaction() {
        let dir = scratch_dir("gloss-prune");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let embedder = Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "sake",
                vec![
                    assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md")),
                    assoc_op("蔵", "廃止銘柄", "旧銘", 1.0, Some("gone.md")),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();

        let (_, total) = state
            .refresh_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            total, 5,
            "concepts 蔵/高瀬/旧銘 plus labels 杜氏/廃止銘柄 all embed"
        );

        // Retract the only source behind 旧銘/廃止銘柄, then compact so
        // those names actually leave the graph.
        state.retract_source("sake", "gone.md").unwrap();
        state
            .compact_context("sake", Deadline::unbounded())
            .unwrap();

        let (_, total) = state
            .refresh_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            total, 3,
            "the vanished concept 旧銘 and label 廃止銘柄 must not linger as ghost rows"
        );
        let sidecar = VectorStore::load(&vectors_path(&dir, &file_stem("sake")));
        assert!(
            !sidecar.concepts.contains_key("旧銘"),
            "the dropped concept's row reached neither memory nor disk"
        );
        assert!(
            !sidecar.labels.contains_key("廃止銘柄"),
            "the dropped label's row reached neither memory nor disk"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gloss_refresh_keeps_concept_vectors_when_the_label_table_fails() {
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
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                let call = self.calls.fetch_add(1, Ordering::Relaxed);
                if call == self.fail_on {
                    return Err("provider hiccup".to_string());
                }
                Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
            }
        }

        let dir = scratch_dir("gloss-partial");
        // Concepts embed on call 0 (success); the labels table is call 1,
        // the one that fails.
        let embedder = Some(Arc::new(FlakyEmbeddings {
            calls: std::sync::atomic::AtomicUsize::new(0),
            fail_on: 1,
        }) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "sake",
                vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md"))],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();

        let error = state
            .refresh_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap_err();
        assert!(error.contains("hiccup"), "{error}");
        let sidecar = VectorStore::load(&vectors_path(&dir, &file_stem("sake")));
        assert_eq!(
            sidecar.concepts.len(),
            2,
            "the concepts the provider already billed for stay durable despite the label failure"
        );
        assert!(
            sidecar.labels.is_empty(),
            "the failed label table wrote nothing"
        );

        // The next refresh buys only the labels the first pass missed.
        let (embedded, total) = state
            .refresh_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!((embedded, total), (1, 3));

        let _ = fs::remove_dir_all(dir);
    }

    /// A width change behind a stable model name must still force a
    /// full re-embed even when the *other* table's call fails in the
    /// same pass — that failure must not excuse persisting this pass's
    /// concepts at the new width right next to labels still at the old
    /// one.
    #[test]
    fn gloss_width_reconciliation_fires_even_when_a_sibling_table_fails() {
        /// Succeeds except on exactly its `fail_on`-th call (0-based);
        /// every successful call answers at `width`.
        struct FlakyWidthEmbeddings {
            calls: AtomicUsize,
            fail_on: usize,
            width: usize,
        }
        impl EmbeddingProvider for FlakyWidthEmbeddings {
            fn model(&self) -> &str {
                "flaky-width"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                let call = self.calls.fetch_add(1, Ordering::Relaxed);
                if call == self.fail_on {
                    return Err("provider hiccup".to_string());
                }
                Ok(texts.iter().map(|_| vec![0.0; self.width]).collect())
            }
        }

        let dir = scratch_dir("gloss-width-reconcile");
        // First boot: establish a carried width of 2.
        {
            let embedder = Some(Arc::new(FlakyWidthEmbeddings {
                calls: AtomicUsize::new(0),
                fail_on: usize::MAX,
                width: 2,
            }) as Arc<dyn EmbeddingProvider>);
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
            state
                .refresh_embeddings("w", Deadline::unbounded())
                .unwrap()
                .unwrap();
            state.flush_dirty();
        }

        // Second boot: same model name, width now 3, plus a brand-new
        // association so both tables carry genuinely stale content —
        // an unchanged-content reboot would leave nothing stale and
        // fall to the single probe call instead, never exercising two
        // independent per-table calls in the same pass. Concepts embed
        // on call 0 (succeeds, proving the width changed); labels are
        // call 1, the one that fails.
        let embedder = Some(Arc::new(FlakyWidthEmbeddings {
            calls: AtomicUsize::new(0),
            fail_on: 1,
            width: 3,
        }) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .write_context("w", |context| {
                context.associate("c", "m", "d", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();
        let (embedded, total) = state
            .refresh_embeddings("w", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (embedded, total),
            (6, 6),
            "the reconciliation retry re-embeds everything, old and new alike, and succeeds"
        );
        let store = VectorStore::load(&vectors_path(&dir, &file_stem("w")));
        assert!(
            store
                .concepts
                .values()
                .chain(store.labels.values())
                .all(|(_, v)| v.len() == 3),
            "a sibling table's transient failure must not leave a mixed-width store live"
        );

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
        use std::thread;

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
                state
                    .refresh_embeddings("fruit", Deadline::unbounded())
                    .unwrap()
                    .unwrap();
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
    fn gloss_refresh_dispatches_chunks_concurrently_when_embed_parallel_is_raised() {
        let dir = scratch_dir("gloss-parallel");
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let embedder = Some(Arc::new(SlowEmbeddings {
            in_flight: Arc::clone(&in_flight),
            peak: Arc::clone(&peak),
        }) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            embedder,
            BootOptions {
                embed_parallel: 2,
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
                // 129 stale concepts split into a 128-item chunk and a
                // 1-item chunk; with embed_parallel=2 both dispatch at
                // once.
                for i in 0..129 {
                    context
                        .associate(format!("c{i}"), "属性", "値", 1.0)
                        .unwrap();
                }
            })
            .map_err(|_| "write")
            .unwrap();

        state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        assert!(
            peak.load(Ordering::SeqCst) > 1,
            "129 stale concepts split into two chunks; with embed_parallel=2 both \
             should reach the provider at once"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn passage_refresh_dispatches_chunks_concurrently_when_embed_parallel_is_raised() {
        let dir = scratch_dir("passage-parallel");
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let embedder = Arc::new(SlowEmbeddings {
            in_flight: Arc::clone(&in_flight),
            peak: Arc::clone(&peak),
        }) as Arc<dyn EmbeddingProvider>;
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            Some(embedder),
            BootOptions {
                embed_passages: true,
                passage_vector_limit: 20_000,
                embed_parallel: 2,
                ..BootOptions::default()
            },
        )
        .unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // 129 paragraphs = one full batch of 128 plus one more; with
        // embed_parallel=2 both chunks dispatch at once.
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

        state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();

        assert!(
            peak.load(Ordering::SeqCst) > 1,
            "129 paragraphs split into two chunks; with embed_parallel=2 both \
             should reach the provider at once"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// Pins down the early-stop half of `dispatch_chunks_concurrently`'s
    /// contract on the schedule where it bites: when the failure is
    /// recorded PROMPTLY (here the failing chunk returns instantly while
    /// every success sleeps), no worker claims a new index past it once
    /// the record lands, so only the `workers` chunks already in flight
    /// at that moment can spill past the failure. A failure slow to
    /// surface would let the other workers run far ahead first — which is
    /// why callers fold on the guaranteed prefix (asserted below), never
    /// on a count of what landed past the failure.
    #[test]
    fn dispatch_chunks_concurrently_bounds_spillover_past_a_promptly_recorded_failure() {
        use std::time::Duration;

        const FAILING_INDEX: usize = 20;
        const WORKERS: usize = 4;
        let chunks: Vec<usize> = (0..50).collect();
        let calls: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));

        let outcomes = dispatch_chunks_concurrently(&chunks, WORKERS, |&index| {
            calls.lock().push(index);
            if index == FAILING_INDEX {
                return Err("boom".to_string());
            }
            // Slow enough that a chunk claimed after the failure lands
            // would have ample time to observe it before finishing and
            // going to claim another — if the gate were broken, this
            // sleep is what would let the assertions below catch it.
            std::thread::sleep(Duration::from_millis(20));
            Ok(index)
        });

        let called = calls.lock().clone();
        assert!(
            called.len() < chunks.len(),
            "the gate must stop dispatch well short of all {} chunks; saw {called:?}",
            chunks.len()
        );
        let past_failure = called
            .iter()
            .filter(|&&index| index > FAILING_INDEX)
            .count();
        assert!(
            past_failure <= WORKERS,
            "at most `workers` chunks can already be in flight when the failure \
             lands; saw {past_failure} claimed past index {FAILING_INDEX}: {called:?}"
        );
        for (index, outcome) in outcomes.iter().enumerate().take(FAILING_INDEX) {
            assert!(
                matches!(outcome, Some(Ok(value)) if *value == index),
                "every index below the true minimum failing index must succeed"
            );
        }
        assert!(matches!(&outcomes[FAILING_INDEX], Some(Err(message)) if message == "boom"));
    }

    #[test]
    fn refresh_passage_embeddings_persists_a_non_prefix_subset_when_parallel_dispatch_fails_early()
    {
        /// Fails every call belonging to chunk 0 (paragraphs 0..128);
        /// later chunks succeed. Which chunk a call belongs to is
        /// recovered from its first text's paragraph index, since the
        /// provider only ever sees texts, not chunk indices.
        struct FailFirstChunk {
            calls: Arc<Mutex<Vec<usize>>>,
        }
        impl EmbeddingProvider for FailFirstChunk {
            fn model(&self) -> &str {
                "fail-first"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                let first_index: usize = texts[0]
                    .trim_start_matches("段落その")
                    .trim_end_matches("。")
                    .parse()
                    .expect("well-formed test fixture text");
                let chunk_index = first_index / 128;
                self.calls.lock().push(chunk_index);
                if chunk_index == 0 {
                    // Delayed so the other two workers have time to
                    // claim and start their own chunks first — an
                    // immediate failure here can otherwise record
                    // `first_failure` before chunk 2's worker even
                    // calls `fetch_add`, gating a chunk that never got
                    // a chance to be "in flight."
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    return Err("boom".to_string());
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
                Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
            }
        }

        let dir = scratch_dir("pvec-non-prefix");
        let calls: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
        let embedder = Arc::new(FailFirstChunk {
            calls: Arc::clone(&calls),
        }) as Arc<dyn EmbeddingProvider>;
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            Some(embedder),
            BootOptions {
                embed_passages: true,
                passage_vector_limit: 20_000,
                embed_parallel: 3,
                ..BootOptions::default()
            },
        )
        .unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // 300 paragraphs = chunk 0 (index 0..128, fails), chunk 1
        // (128..256, succeeds), chunk 2 (256..300, succeeds) — with
        // embed_parallel=3 all three dispatch at once, so chunks 1 and 2
        // can complete and land before chunk 0's failure is even
        // recorded.
        let text = (0..300)
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
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap_err();
        assert!(error.contains("boom"), "{error}");

        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("sake")));
        assert_eq!(
            sidecar.len(),
            172,
            "chunks 1 and 2 (128 + 44 paragraphs) persist even though the \
             earlier chunk 0 failed — the surviving subset is not a prefix \
             of the original order"
        );
        assert!(
            sidecar.iter().all(|(key, _)| key.index >= 128),
            "no paragraph from the failed first chunk (index < 128) should be present"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// A width change behind a stable model name must still force a
    /// full re-embed even when a sibling chunk's call fails in the same
    /// pass. Left gated on that failure, the carried row's old
    /// dimension stays pinned on `fresh` and `PassageVectorStore::push`
    /// silently drops every new-width row the surviving chunk just
    /// bought — reported as a mere transient error, with no sign the
    /// work was thrown away.
    #[test]
    fn passage_width_reconciliation_fires_even_when_a_later_chunk_fails() {
        /// Succeeds except on exactly its `fail_on`-th call (0-based);
        /// every successful call answers at `width`.
        struct FlakyWidthEmbeddings {
            calls: AtomicUsize,
            fail_on: usize,
            width: usize,
        }
        impl EmbeddingProvider for FlakyWidthEmbeddings {
            fn model(&self) -> &str {
                "flaky-width"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                let call = self.calls.fetch_add(1, Ordering::Relaxed);
                if call == self.fail_on {
                    return Err("provider hiccup".to_string());
                }
                Ok(texts.iter().map(|_| vec![0.0; self.width]).collect())
            }
        }

        let dir = scratch_dir("pvec-width-reconcile");
        // First boot: one paragraph at width 2, establishing the
        // carried width the second boot must reconcile against.
        {
            let embedder = Arc::new(FlakyWidthEmbeddings {
                calls: AtomicUsize::new(0),
                fail_on: usize::MAX,
                width: 2,
            }) as Arc<dyn EmbeddingProvider>;
            let state = boot_for_passage_embedding(&dir, embedder, 20_000);
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            let mut seed = BTreeMap::new();
            seed.insert("doc-seed".to_string(), "最初の段落。".to_string());
            state.store_passages("sake", plain(seed)).unwrap().unwrap();
            state
                .refresh_passage_embeddings("sake", Deadline::unbounded())
                .unwrap()
                .unwrap();
        }

        // Second boot: same model name, width now 3. 129 new paragraphs
        // split into a 128-item chunk (call 0, succeeds) and a 1-item
        // remainder (call 1, fails) — doc-seed's unchanged paragraph is
        // carried forward, not re-embedded, so it never touches the
        // provider. The surviving chunk already proves the width
        // changed; that must be reconciled regardless of its sibling's
        // failure.
        let embedder = Arc::new(FlakyWidthEmbeddings {
            calls: AtomicUsize::new(0),
            fail_on: 1,
            width: 3,
        }) as Arc<dyn EmbeddingProvider>;
        let state = boot_for_passage_embedding(&dir, embedder, 20_000);
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

        let outcome = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (outcome.embedded, outcome.total),
            (130, 130),
            "the reconciliation retry re-embeds every row, carried and fresh alike"
        );
        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("sake")));
        assert!(
            sidecar.iter().all(|(_, row)| row.len() == 3),
            "a sibling chunk's transient failure must not block reconciling a width \
             disagreement the surviving chunk already proved"
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
                .semantic_resolve("fruit", "アップル", false, None, Deadline::unbounded())
                .unwrap()
                .unwrap()
                .is_empty()
        );

        // Refresh embeds every canonical name's gloss once; a second run
        // is a no-op.
        let (embedded, total) = state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(embedded, 3); // りんご, 果物 + label 分類
        assert_eq!(total, 3);
        assert_eq!(
            state
                .refresh_embeddings("fruit", Deadline::unbounded())
                .unwrap()
                .unwrap()
                .0,
            0
        );

        // Now the paraphrase lands on the stored spelling by cosine, and
        // unrelated names stay under the floor.
        let hits = state
            .semantic_resolve("fruit", "アップル", false, None, Deadline::unbounded())
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
        let (embedded, total) = state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(embedded, 3);
        assert_eq!(total, 5);

        assert!(
            state
                .semantic_resolve("nope", "x", false, None, Deadline::unbounded())
                .is_none()
        );

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
        state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        // One batch per namespace: concepts, then labels.
        assert_eq!(calls.load(Ordering::Relaxed), 2);

        // First query embeds the cue; repeating the wording does not.
        let first = state
            .semantic_resolve("fruit", "アップル", false, None, Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(first[0].0, "りんご");
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        state
            .semantic_resolve("fruit", "アップル", false, None, Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 3, "cue must come from cache");

        // The sidecar is held in memory after first use: even with the
        // file gone, the same query keeps answering.
        fs::remove_file(vectors_path(&dir, &file_stem("fruit"))).unwrap();
        let held = state
            .semantic_resolve("fruit", "アップル", false, None, Deadline::unbounded())
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
                .semantic_resolve("fruit", "アップル", false, None, Deadline::unbounded())
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
        let (concepts, labels, note) = state
            .semantic_twins("sake", 0.6, Deadline::unbounded())
            .unwrap();
        assert!(concepts.is_empty() && labels.is_empty());
        assert!(note.is_some());

        state
            .refresh_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        let (concepts, labels, note) = state
            .semantic_twins("sake", 0.6, Deadline::unbounded())
            .unwrap();
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

        assert!(
            state
                .semantic_twins("nope", 0.6, Deadline::unbounded())
                .is_none()
        );
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
        state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        // みかん×りんご sits at cosine 0.28 — under the 0.35 default.
        let miss = |floor: Option<f32>| {
            state
                .semantic_resolve("fruit", "みかん", false, floor, Deadline::unbounded())
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
        state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let hits = |floor: Option<f32>| {
            state
                .semantic_resolve("fruit", "みかん", false, floor, Deadline::unbounded())
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
    fn directory_page_seeks_by_name_with_a_cursor_independent_total() {
        let dir = scratch_dir("directory-page");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        for name in ["cherry", "apple", "banana"] {
            state.create(name, ContextMeta::default()).unwrap();
        }

        let (total, first) = state.directory_page(None, 2);
        assert_eq!(total, 3);
        let names: Vec<&str> = first.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, vec!["apple", "banana"]);

        let (total, second) = state.directory_page(Some("banana"), 2);
        assert_eq!(total, 3, "total stays constant across pages");
        let names: Vec<&str> = second.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, vec!["cherry"]);

        // Deleting a context drops it from the very next page.
        state.delete("apple").unwrap().unwrap();
        let (total, page) = state.directory_page(None, 10);
        assert_eq!(total, 2);
        let names: Vec<&str> = page.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, vec!["banana", "cherry"]);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn directory_page_skips_past_a_seek_window_tombstoned_before_describe() {
        let dir = scratch_dir("directory-page-tombstoned-window");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        for name in ["apple", "banana"] {
            state.create(name, ContextMeta::default()).unwrap();
        }

        // The race directory_page's doc comment describes: a handle
        // survives because the seek already cloned its Arc out of the
        // registry, but the slot it points at is tombstoned before
        // describe_entry reads it — the same effect AppState::delete
        // has on a handle the seek already cloned, without actually
        // removing "apple" from the BTreeMap so the seek still lands
        // on it.
        {
            let registry = state.0.registry.read();
            let entry = registry.get("apple").unwrap();
            entry.inner.write().slot = Slot::Deleted;
        }

        // limit 1 makes the whole seek window ("apple" alone) lose the
        // race — a false end of directory would stop right here and
        // never see "banana".
        let (total, page) = state.directory_page(None, 1);
        assert_eq!(total, 2);
        let names: Vec<&str> = page.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, vec!["banana"]);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn every_group_write_persistence_failure_rolls_back_and_retries() {
        let mut exhausted = false;
        for failure in 0..8 {
            let dir = scratch_dir(&format!("group-write-fault-{failure}"));
            let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
            state
                .create_group(
                    "drinks",
                    "old".to_string(),
                    BTreeSet::new(),
                    BTreeSet::new(),
                )
                .unwrap();

            fail_persistence_ops_after(failure);
            let first = state.update_group(
                "drinks",
                Some("new".to_string()),
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
            );
            let past_end = clear_persistence_fault();
            if past_end {
                assert!(first.is_ok());
            } else {
                assert!(matches!(first, Err(UpdateGroupError::Io(_))));
                assert_eq!(state.group("drinks").unwrap().description, "old");
                state
                    .update_group(
                        "drinks",
                        Some("new".to_string()),
                        BTreeSet::new(),
                        BTreeSet::new(),
                        BTreeSet::new(),
                        BTreeSet::new(),
                    )
                    .unwrap();
            }
            drop(state);

            let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
            assert_eq!(state.group("drinks").unwrap().description, "new");
            drop(state);
            let _ = fs::remove_dir_all(&dir);
            if past_end {
                exhausted = true;
                break;
            }
        }
        assert!(exhausted, "group write exceeded the sweep bound");
    }

    #[test]
    fn every_group_delete_persistence_failure_reconciles_at_boot() {
        let mut exhausted = false;
        for failure in 0..8 {
            let dir = scratch_dir(&format!("group-delete-fault-{failure}"));
            let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
            state
                .create_group("leaf", String::new(), BTreeSet::new(), BTreeSet::new())
                .unwrap();
            state
                .create_group(
                    "parent",
                    String::new(),
                    BTreeSet::new(),
                    BTreeSet::from(["leaf".to_string()]),
                )
                .unwrap();

            fail_persistence_ops_after(failure);
            let _ = state.delete_group("leaf").unwrap();
            let past_end = clear_persistence_fault();
            drop(state);

            let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
            assert!(
                !state.group("parent").unwrap().groups.contains("leaf"),
                "failure at step {failure} left a dangling child after boot"
            );
            drop(state);
            let _ = fs::remove_dir_all(&dir);
            if past_end {
                exhausted = true;
                break;
            }
        }
        assert!(exhausted, "group deletion exceeded the sweep bound");
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
    fn a_corrupt_group_file_keeps_its_name_and_sets_the_bytes_aside() {
        let dir = scratch_dir("groups-corrupt");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        drop(state);
        let live = groups::group_path(&dir, &file_stem("mangled"));
        fs::write(&live, b"{not json").unwrap();

        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let record = state.group("mangled").unwrap();
        assert_eq!(record, GroupRecord::default());
        // The mangled bytes moved aside for hand recovery, and a fresh
        // empty record took their place — a later write to this group
        // overwrites nothing that was never loaded.
        assert_eq!(
            fs::read(live.with_extension("group.corrupt")).unwrap(),
            b"{not json",
            "the original bytes must survive, set aside"
        );
        let on_disk: GroupRecord = serde_json::from_slice(&fs::read(&live).unwrap()).unwrap();
        assert_eq!(on_disk, GroupRecord::default());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_crash_mid_repair_never_drops_the_corrupt_group_file_itself() {
        let mut exhausted = false;
        for failure in 0..6 {
            let dir = scratch_dir(&format!("groups-corrupt-fault-{failure}"));
            let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
            drop(state);
            let live = groups::group_path(&dir, &file_stem("mangled"));
            fs::write(&live, b"{not json").unwrap();

            fail_persistence_ops_after(failure);
            let result = AppState::boot(dir.clone(), 1 << 20, None);
            let past_end = clear_persistence_fault();

            if past_end {
                let state = result.unwrap();
                assert_eq!(state.group("mangled").unwrap(), GroupRecord::default());
                drop(state);
            } else {
                // Whatever the write reached — still the mangled bytes,
                // or already the fresh empty record — `path` must
                // resolve to SOMETHING. The write-then-set-aside order
                // guarantees it; the old set-aside-then-write order
                // could lose this file entirely between the two steps.
                assert!(
                    live.exists(),
                    "failure after {failure} successes must not drop the group file"
                );
                let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
                assert_eq!(state.group("mangled").unwrap(), GroupRecord::default());
                drop(state);
            }

            let _ = fs::remove_dir_all(&dir);
            if past_end {
                exhausted = true;
                break;
            }
        }
        assert!(exhausted, "group repair exceeded the sweep bound");
    }

    #[test]
    fn an_unreadable_group_file_refuses_the_boot() {
        let dir = scratch_dir("groups-unreadable");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        drop(state);
        // A directory wearing the extension: fs::read fails on it, on
        // every platform, the way a permission hiccup would — and boot
        // must refuse rather than register 'locked' over an empty
        // record a later write would persist.
        let imposter = dir.join("locked.group");
        fs::create_dir(&imposter).unwrap();
        let message = match AppState::boot(dir.clone(), 1 << 20, None) {
            Ok(_) => panic!("boot must refuse while a group file is unreadable"),
            Err(error) => error.to_string(),
        };
        assert!(message.contains("locked"), "names the group: {message}");
        // Clearing the obstacle heals without further ceremony.
        fs::remove_dir(&imposter).unwrap();
        assert!(AppState::boot(dir.clone(), 1 << 20, None).is_ok());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn group_membership_is_capped_per_set_and_judged_before_existence() {
        let dir = scratch_dir("groups-cap");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        state.create("real", ContextMeta::default()).unwrap();
        // One name past the cap refuses — before existence, so the
        // names need not exist to hear it — and nothing lands.
        let over: BTreeSet<String> = (0..=groups::MAX_GROUP_MEMBERS)
            .map(|i| format!("c{i:04}"))
            .collect();
        assert!(matches!(
            state.create_group("g", String::new(), over.clone(), BTreeSet::new()),
            Err(CreateGroupError::OverCap("member contexts"))
        ));
        assert!(matches!(
            state.create_group("g", String::new(), BTreeSet::new(), over.clone()),
            Err(CreateGroupError::OverCap("child groups"))
        ));
        assert!(state.group("g").is_none());

        // A delta is judged on its RESULT: 1 member + cap-many adds is
        // one too many, but trading the member out in the same request
        // makes room — the cap passes and the existence gate speaks
        // next, proving the judgement order.
        let one = BTreeSet::from(["real".to_string()]);
        state
            .create_group("g", String::new(), one.clone(), BTreeSet::new())
            .unwrap();
        let cap_many: BTreeSet<String> = (0..groups::MAX_GROUP_MEMBERS)
            .map(|i| format!("c{i:04}"))
            .collect();
        assert!(matches!(
            state.update_group(
                "g",
                None,
                cap_many.clone(),
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new()
            ),
            Err(UpdateGroupError::OverCap("member contexts"))
        ));
        assert!(matches!(
            state.update_group(
                "g",
                None,
                cap_many.clone(),
                one.clone(),
                BTreeSet::new(),
                BTreeSet::new()
            ),
            Err(UpdateGroupError::NoSuchContext(missing)) if missing == "c0000"
        ));
        // Child groups ride the same cap on their own set.
        assert!(matches!(
            state.update_group(
                "g",
                None,
                BTreeSet::new(),
                BTreeSet::new(),
                over,
                BTreeSet::new()
            ),
            Err(UpdateGroupError::OverCap("child groups"))
        ));
        assert_eq!(state.group("g").unwrap().contexts, one);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn restore_groups_replaces_whole_records_and_reports_what_stood() {
        let dir = scratch_dir("groups-restore");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state.create("bunko", ContextMeta::default()).unwrap();
        state
            .create_group(
                "kura",
                "old".to_string(),
                BTreeSet::from(["sake".to_string(), "bunko".to_string()]),
                BTreeSet::new(),
            )
            .unwrap();
        // A standing parent naming the group must survive its replace:
        // a restore rewrites the record, never the references to it.
        state
            .create_group(
                "parent",
                String::new(),
                BTreeSet::new(),
                BTreeSet::from(["kura".to_string()]),
            )
            .unwrap();

        let record = |contexts: &[&str], children: &[&str]| GroupRecord {
            description: "new".to_string(),
            contexts: contexts.iter().map(|c| c.to_string()).collect(),
            groups: children.iter().map(|g| g.to_string()).collect(),
        };
        // The set references its own newcomer — parent listed first,
        // child later: order inside the set must not matter.
        let records = vec![
            ("kura".to_string(), record(&["sake"], &["kid"])),
            ("kid".to_string(), record(&["bunko"], &[])),
        ];
        let outcomes = state
            .restore_groups(&records, Deadline::unbounded())
            .unwrap();
        assert_eq!(outcomes[0].1.as_str(), "replaced");
        assert_eq!(outcomes[1].1.as_str(), "created");
        // The replace is the WHOLE record — bunko dropped, description
        // replaced — and parent still names kura.
        assert_eq!(state.group("kura").unwrap(), records[0].1);
        assert_eq!(
            state.group("parent").unwrap().groups,
            BTreeSet::from(["kura".to_string()])
        );
        // Disk agrees.
        let on_disk: GroupRecord =
            serde_json::from_slice(&fs::read(groups::group_path(&dir, &file_stem("kid"))).unwrap())
                .unwrap();
        assert_eq!(on_disk, records[1].1);

        // Restoring the same set again converges to no-ops.
        let again = state
            .restore_groups(&records, Deadline::unbounded())
            .unwrap();
        assert!(
            again
                .iter()
                .all(|(_, outcome)| outcome.as_str() == "unchanged"),
            "{again:?}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn restore_groups_reports_how_many_landed_before_an_io_failure() {
        use crate::groups::group_path;

        let dir = scratch_dir("groups-restore-io-failure");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let record = |contexts: &[&str], children: &[&str]| GroupRecord {
            description: String::new(),
            contexts: contexts.iter().map(|c| c.to_string()).collect(),
            groups: children.iter().map(|g| g.to_string()).collect(),
        };

        // "a" and "b" are independent (no nesting), so equal depth keeps
        // them in list order — "a" lands first. Occupying "b"'s target
        // path with a non-empty directory fails its rename on every
        // platform this project targets (see
        // `write_atomic_cleans_up_its_staging_file_when_the_commit_fails`),
        // isolating an Io failure strictly after "a" already landed.
        let blocked = group_path(&dir, &file_stem("b"));
        fs::create_dir(&blocked).unwrap();
        fs::write(blocked.join("occupied"), b"x").unwrap();

        let error = state
            .restore_groups(
                &[
                    ("a".to_string(), record(&[], &[])),
                    ("b".to_string(), record(&[], &[])),
                ],
                Deadline::unbounded(),
            )
            .unwrap_err();
        assert!(matches!(
            &error,
            RestoreGroupsError::Io { group, .. } if group == "b"
        ));
        assert_eq!(error.applied(), 1, "\"a\" already landed when \"b\" failed");
        assert!(
            state.group("a").is_some(),
            "\"a\"'s successful write must not be rolled back"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn restore_groups_counts_an_unchanged_record_as_already_landed() {
        use crate::groups::group_path;

        let dir = scratch_dir("groups-restore-unchanged-then-io-failure");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let record = |contexts: &[&str], children: &[&str]| GroupRecord {
            description: String::new(),
            contexts: contexts.iter().map(|c| c.to_string()).collect(),
            groups: children.iter().map(|g| g.to_string()).collect(),
        };

        // Land "a" first so the call below finds it standing exactly as
        // given and takes the `Unchanged` branch, which counts toward
        // `applied` without writing anything.
        state
            .restore_groups(
                &[("a".to_string(), record(&[], &[]))],
                Deadline::unbounded(),
            )
            .unwrap();

        // "b" is new, so it takes the `write_group` branch. Occupying its
        // target path with a non-empty directory fails its rename on
        // every platform this project targets (see
        // `write_atomic_cleans_up_its_staging_file_when_the_commit_fails`),
        // isolating an Io failure strictly after "a" is judged unchanged.
        let blocked = group_path(&dir, &file_stem("b"));
        fs::create_dir(&blocked).unwrap();
        fs::write(blocked.join("occupied"), b"x").unwrap();

        let error = state
            .restore_groups(
                &[
                    ("a".to_string(), record(&[], &[])),
                    ("b".to_string(), record(&[], &[])),
                ],
                Deadline::unbounded(),
            )
            .unwrap_err();
        assert!(matches!(
            &error,
            RestoreGroupsError::Io { group, .. } if group == "b"
        ));
        assert_eq!(
            error.applied(),
            1,
            "\"a\" must count as landed via the unchanged branch"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn restore_groups_stops_at_an_expired_deadline_without_writing() {
        use std::time::Duration;

        let dir = scratch_dir("groups-restore-timeout");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        let record = GroupRecord {
            description: String::new(),
            contexts: BTreeSet::new(),
            groups: BTreeSet::new(),
        };

        // A budget already spent must bound the fsync-per-record loop
        // before its first write, the way the batch loop bounds itself —
        // a stream of many tiny group records must not pin the write
        // lock past the deadline every other group op honors.
        let deadline = Deadline::after(Duration::ZERO);
        std::thread::sleep(Duration::from_millis(1));
        let error = state
            .restore_groups(&[("a".to_string(), record)], deadline)
            .unwrap_err();
        assert!(matches!(&error, RestoreGroupsError::Timeout { applied: 0 }));
        assert_eq!(error.applied(), 0);

        // Nothing landed — not in memory, not on disk.
        assert!(
            state.group("a").is_none(),
            "no record may land past the budget"
        );
        assert!(
            !groups::group_path(&dir, &file_stem("a")).exists(),
            "no group file may be written past the budget"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn restore_groups_judges_the_whole_set_before_writing_anything() {
        use crate::groups::NestingViolation;

        let dir = scratch_dir("groups-restore-refuse");
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        let record = |contexts: &[&str], children: &[&str]| GroupRecord {
            description: String::new(),
            contexts: contexts.iter().map(|c| c.to_string()).collect(),
            groups: children.iter().map(|g| g.to_string()).collect(),
        };

        // A dangling member refuses the set — the valid record beside
        // it included.
        let refusal = state
            .restore_groups(
                &[
                    ("fine".to_string(), record(&["sake"], &[])),
                    ("broken".to_string(), record(&["ghost"], &[])),
                ],
                Deadline::unbounded(),
            )
            .unwrap_err();
        assert!(matches!(
            &refusal,
            RestoreGroupsError::NoSuchContext { group, context }
                if group == "broken" && context == "ghost"
        ));
        assert_eq!(refusal.applied(), 0);
        assert!(
            state.group("fine").is_none(),
            "nothing applies on a refusal"
        );

        // A child neither standing nor in the set.
        assert!(matches!(
            state
                .restore_groups(&[("a".to_string(), record(&[], &["nope"]))], Deadline::unbounded())
                .unwrap_err(),
            RestoreGroupsError::NoSuchChild { group, child } if group == "a" && child == "nope"
        ));

        // A cycle the set closes with itself.
        assert!(matches!(
            state
                .restore_groups(
                    &[
                        ("a".to_string(), record(&[], &["b"])),
                        ("b".to_string(), record(&[], &["a"])),
                    ],
                    Deadline::unbounded(),
                )
                .unwrap_err(),
            RestoreGroupsError::Nesting(NestingViolation::Cycle(_))
        ));

        // Depth counts the standing groups too: records stacking two
        // more storeys under an existing 2-chain overflow the cap.
        state
            .create_group("mid", String::new(), BTreeSet::new(), BTreeSet::new())
            .unwrap();
        state
            .create_group(
                "top",
                String::new(),
                BTreeSet::new(),
                BTreeSet::from(["mid".to_string()]),
            )
            .unwrap();
        assert!(matches!(
            state
                .restore_groups(
                    &[
                        ("mid".to_string(), record(&[], &["deep"])),
                        ("deep".to_string(), record(&[], &["deeper"])),
                        ("deeper".to_string(), record(&[], &[])),
                    ],
                    Deadline::unbounded(),
                )
                .unwrap_err(),
            RestoreGroupsError::Nesting(NestingViolation::TooDeep(_))
        ));
        assert!(state.group("deep").is_none());

        // One name twice is two truths for one group.
        assert!(matches!(
            state
                .restore_groups(
                    &[
                        ("dup".to_string(), record(&[], &[])),
                        ("dup".to_string(), record(&["sake"], &[])),
                    ],
                    Deadline::unbounded(),
                )
                .unwrap_err(),
            RestoreGroupsError::Duplicate(name) if name == "dup"
        ));

        // The cap judges each record's sets.
        let over: BTreeSet<String> = (0..=groups::MAX_GROUP_MEMBERS)
            .map(|i| format!("c{i:04}"))
            .collect();
        assert!(matches!(
            state
                .restore_groups(
                    &[(
                        "wide".to_string(),
                        GroupRecord {
                            description: String::new(),
                            contexts: over,
                            groups: BTreeSet::new(),
                        },
                    )],
                    Deadline::unbounded(),
                )
                .unwrap_err(),
            RestoreGroupsError::OverCap {
                field: "member contexts",
                ..
            }
        ));

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

    fn assert_live_group_invariants(state: &AppState) {
        let contexts: BTreeSet<String> = state
            .directory()
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        let groups = state.0.groups.read();
        assert_eq!(groups::validate_nesting(&groups), Ok(()));
        for record in groups.values() {
            assert!(record.contexts.len() <= groups::MAX_GROUP_MEMBERS);
            assert!(record.groups.len() <= groups::MAX_GROUP_MEMBERS);
            assert!(record.contexts.iter().all(|name| contexts.contains(name)));
            assert!(record.groups.iter().all(|name| groups.contains_key(name)));
        }
    }

    proptest! {
        #![proptest_config(proptest_config())]

        #[test]
        fn arbitrary_group_operations_continuously_preserve_store_invariants(
            operations in prop::collection::vec(group_op_strategy(), 1..48),
        ) {
            let dir = scratch_dir("group-properties");
            let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();

            for operation in operations {
                match operation {
                    GeneratedGroupOp::CreateContext(name) => {
                        let _ = state.create(name, ContextMeta::default());
                    }
                    GeneratedGroupOp::DeleteContext(name) => {
                        let _ = state.delete(name);
                    }
                    GeneratedGroupOp::CreateGroup { name, contexts, groups } => {
                        let _ = state.create_group(
                            name,
                            String::new(),
                            contexts.into_iter().map(str::to_string).collect(),
                            groups.into_iter().map(str::to_string).collect(),
                        );
                    }
                    GeneratedGroupOp::UpdateGroup {
                        name,
                        add_contexts,
                        remove_contexts,
                        add_groups,
                        remove_groups,
                    } => {
                        let _ = state.update_group(
                            name,
                            None,
                            add_contexts.into_iter().map(str::to_string).collect(),
                            remove_contexts.into_iter().map(str::to_string).collect(),
                            add_groups.into_iter().map(str::to_string).collect(),
                            remove_groups.into_iter().map(str::to_string).collect(),
                        );
                    }
                    GeneratedGroupOp::DeleteGroup(name) => {
                        let _ = state.delete_group(name);
                    }
                }
                assert_live_group_invariants(&state);
            }

            drop(state);
            let _ = fs::remove_dir_all(dir);
        }
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

    /// parking_lot locks don't poison: a panic while holding one just
    /// unwinds and releases it, so neither the context that panicked, nor
    /// a sibling, nor the registry's own listing bricks for the rest of
    /// the process.
    #[test]
    fn a_panic_mid_write_bricks_only_that_context_not_a_sibling_or_the_registry() {
        let dir = scratch_dir("panic-mid-write");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state.create("cherry", ContextMeta::default()).unwrap();

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.write_context("sake", |_context| {
                panic!("simulated failure mid-write");
            })
        }));
        assert!(
            panicked.is_err(),
            "the panic must propagate, not be swallowed"
        );

        state
            .write_context("sake", |_context| {})
            .expect("the context that panicked stays usable — the lock never poisoned");
        state
            .write_context("cherry", |_context| {})
            .expect("a sibling context is unaffected by the panic");
        assert_eq!(
            state.directory().len(),
            2,
            "the registry's own listing survives the panic too"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// Unlike `write_context` above, `logged_write` appends the whole
    /// batch to the WAL *before* running `operate` — so a panic partway
    /// through the batch leaves the tail durably logged but (absent
    /// this recovery) never applied in memory and never marked `dirty`
    /// either, since the slot stayed Hot holding whatever `operate`
    /// half-finished. A closure that panics only *after* fully applying
    /// every op would not catch this: the in-memory state would already
    /// be complete and correct, panic or not. The real risk is the
    /// batch's untried tail — durably logged, silently missing from
    /// memory. Forcing the slot back to Cold on panic must make the
    /// very next access — no restart needed — rebuild from the image
    /// plus a WAL replay that reapplies that untried tail too.
    #[test]
    fn a_panic_inside_logged_write_forces_a_cold_reload_that_replays_the_wal() {
        let dir = scratch_dir("logged-write-panic-recovery");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();

        let ops = vec![
            WalOp::Associate(assoc_op(
                "青嶺酒造",
                "創業年",
                "1907年",
                1.0,
                Some("第1段落"),
            )),
            WalOp::Associate(assoc_op("青嶺酒造", "所在地", "灘", 1.0, Some("第2段落"))),
        ];

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.logged_write(
                "sake",
                &ops,
                |context| {
                    // Only the first op of the batch applies before the
                    // simulated bug fires; the second is durably in the
                    // WAL (appended above, before `operate` ran) but
                    // never reaches memory here.
                    apply_op(context, &ops[0]).unwrap();
                    panic!("simulated failure partway through the batch");
                },
                applied_count,
            )
        }));
        assert!(
            panicked.is_err(),
            "the panic must propagate, not be swallowed"
        );

        let recalled = state
            .read_context("sake", |context| context.recall("青嶺酒造"))
            .expect("the context that panicked stays usable — the lock never poisoned");
        assert_eq!(
            recalled.len(),
            2,
            "both WAL-logged ops must survive the panic, in this same process, with \
             no restart — including the one `operate` never got to apply"
        );
        let objects: BTreeSet<&str> = recalled.iter().map(|fact| fact.object.as_str()).collect();
        assert_eq!(objects, BTreeSet::from(["1907年", "灘"]));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rename_context_moves_the_family_and_rewrites_group_membership() {
        let dir = scratch_dir("rename-context-happy");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create(
                "sake",
                ContextMeta {
                    pinned: true,
                    ..ContextMeta::default()
                },
            )
            .unwrap();
        state
            .add_associations(
                "sake",
                vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("a.md"))],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state
            .create_group(
                "drinks",
                String::new(),
                BTreeSet::from(["sake".to_string()]),
                BTreeSet::new(),
            )
            .unwrap();

        state.rename_context("sake", "shochu").unwrap();

        assert!(
            state.directory_entry("sake").is_none(),
            "the old name must be gone"
        );
        let entry = state
            .directory_entry("shochu")
            .expect("the new name must answer");
        assert!(entry.pinned, "pinned carries over");
        assert!(
            entry.loaded,
            "a pinned context reloads hot under its new name"
        );
        assert!(!dir.join("sake.ctx").exists());
        assert!(dir.join("shochu.ctx").exists());
        assert_eq!(
            state.group("drinks").unwrap().contexts,
            BTreeSet::from(["shochu".to_string()]),
            "group membership follows the rename, not a stale name"
        );
        assert!(!renaming_marker_path(&dir, &file_stem("sake")).exists());
        let count = state
            .read_context("shochu", |context| context.association_count())
            .unwrap();
        assert_eq!(count, 1, "data must survive the move");

        // Persisted, not just in memory.
        drop(state);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.directory_entry("sake").is_none());
        assert!(state.directory_entry("shochu").is_some());
        assert_eq!(
            state.group("drinks").unwrap().contexts,
            BTreeSet::from(["shochu".to_string()])
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// Regression for the critical data-loss bug in
    /// `rename_context_locked`: once the registry has forgotten `from`
    /// (the point of no return), a failed pivot move must keep `from`
    /// reserved in `pending.renames` rather than release it. Before the
    /// fix, `rename_context` unconditionally cleared the reservation on
    /// any failure, so a client's natural reaction to seeing `from`
    /// vanish — `create(from)` — sailed through `create_files`'s
    /// stale-file sweep and deleted both the untouched old generation's
    /// files AND the `.renaming` marker that boot needs to resume the
    /// move, erasing the data beyond any recovery.
    #[test]
    fn a_rename_stuck_past_the_point_of_no_return_refuses_a_recreate_and_survives_reboot() {
        let dir = scratch_dir("rename-stuck-recreate");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state
            .add_associations(
                "sake",
                vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("a.md"))],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();

        // Block the pivot: `fs::rename` onto an existing directory
        // fails with ENOTDIR/EISDIR, deterministically breaking
        // `move_context_files`'s first (pivot) move without touching
        // permissions — which would also break the marker write that
        // must succeed first.
        let blocker = dir.join(format!("{}.ctx", file_stem("shochu")));
        fs::create_dir(&blocker).unwrap();

        let error = state.rename_context("sake", "shochu").unwrap_err();
        assert!(
            matches!(error, RenameContextError::Io(_)),
            "the pivot move must fail: {error:?}"
        );

        assert!(
            state.directory_entry("sake").is_none(),
            "memory already forgot the source name past the point of no return"
        );
        assert!(
            state.directory_entry("shochu").is_none(),
            "the destination never landed"
        );
        assert!(
            renaming_marker_path(&dir, &file_stem("sake")).exists(),
            "the marker must survive so boot can resume the move"
        );
        assert!(
            dir.join("sake.ctx").exists(),
            "the old generation's files must stay put, untouched"
        );

        // The dangerous part: a client that saw `sake` disappear (404)
        // and naturally retries with create() must be refused, not
        // handed a fresh empty context in place of the old data.
        assert!(
            matches!(
                state.create("sake", ContextMeta::default()),
                Err(CreateError::AlreadyExists)
            ),
            "a stuck rename must keep blocking create(), or create_files' \
             stale-file sweep would delete the marker and the old data"
        );
        assert!(
            renaming_marker_path(&dir, &file_stem("sake")).exists(),
            "the refused create must not have touched the marker"
        );
        assert!(
            dir.join("sake.ctx").exists(),
            "the refused create must not have touched the old data"
        );

        // Clear the obstruction and let boot's resume-sweep finish what
        // the live call could not.
        fs::remove_dir(&blocker).unwrap();
        drop(state);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.directory_entry("sake").is_none());
        let count = state
            .read_context("shochu", |context| context.association_count())
            .unwrap();
        assert_eq!(count, 1, "the resumed move must carry the old data over");
        assert!(!renaming_marker_path(&dir, &file_stem("sake")).exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rename_group_moves_the_file_and_rewrites_parent_membership() {
        let dir = scratch_dir("rename-group-happy");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state
            .create_group(
                "liquor",
                "d".into(),
                BTreeSet::from(["sake".to_string()]),
                BTreeSet::new(),
            )
            .unwrap();
        state
            .create_group(
                "drinks",
                String::new(),
                BTreeSet::new(),
                BTreeSet::from(["liquor".to_string()]),
            )
            .unwrap();

        state.rename_group("liquor", "spirits").unwrap();

        assert!(state.group("liquor").is_none());
        let spirits = state.group("spirits").unwrap();
        assert_eq!(spirits.description, "d");
        assert_eq!(spirits.contexts, BTreeSet::from(["sake".to_string()]));
        assert_eq!(
            state.group("drinks").unwrap().groups,
            BTreeSet::from(["spirits".to_string()]),
            "the parent's child reference follows the rename"
        );
        assert!(!groups::group_path(&dir, &file_stem("liquor")).exists());
        assert!(groups::group_path(&dir, &file_stem("spirits")).exists());
        assert!(!groups::group_renaming_marker_path(&dir, &file_stem("liquor")).exists());

        // Persisted, not just in memory.
        drop(state);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.group("liquor").is_none());
        assert_eq!(
            state.group("drinks").unwrap().groups,
            BTreeSet::from(["spirits".to_string()])
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// The half-done-move contract `boot_with` leans on. `landed` and
    /// `complete` must move independently: a failed move is never
    /// complete (so the marker stays for the next boot to retry), and
    /// membership may only be rewritten once the destination pivot has
    /// landed. Deleting the marker on a failed move was the bug — the
    /// retry vanished and the group association was lost with no way
    /// back.
    #[test]
    fn a_failed_resume_keeps_the_marker_and_defers_membership() {
        let dir = scratch_dir("resume-failure");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            renaming_marker_path(&dir, &file_stem("sake")),
            serde_json::to_vec(&RenameMarker {
                from: "sake".to_string(),
                to: "shochu".to_string(),
            })
            .unwrap(),
        )
        .unwrap();

        // Move fails and no pivot appears at the destination: neither
        // bit set — boot_with rewrites no membership and keeps the
        // marker. resume_rename_markers itself never removes a marker.
        let resumed = resume_rename_markers(
            &dir,
            "renaming",
            "context",
            |_, _| Err(io::Error::other("file still held")),
            |_| false,
        )
        .unwrap();
        assert_eq!(resumed.len(), 1);
        assert_eq!(resumed[0].from, "sake");
        assert_eq!(resumed[0].to, "shochu");
        assert!(!resumed[0].complete, "a failed move is not complete");
        assert!(!resumed[0].landed, "no pivot at the destination");
        assert!(
            renaming_marker_path(&dir, &file_stem("sake")).exists(),
            "the marker must survive a failed resume so the next boot retries"
        );

        // Pivot landed but a straggler stuck: landed (rewrite
        // membership) yet not complete (keep the marker to finish).
        let resumed = resume_rename_markers(
            &dir,
            "renaming",
            "context",
            |_, _| Err(io::Error::other("sidecar still held")),
            |_| true,
        )
        .unwrap();
        assert!(resumed[0].landed, "the pivot is at the destination");
        assert!(!resumed[0].complete, "a stuck straggler is not complete");

        // Everything moved: both bits set — rewrite membership, drop marker.
        let resumed =
            resume_rename_markers(&dir, "renaming", "context", |_, _| Ok(()), |_| true).unwrap();
        assert!(resumed[0].landed && resumed[0].complete);

        let _ = fs::remove_dir_all(dir);
    }

    /// The crash-shaped state: `rename_context` wrote its marker but
    /// died before the file move and the group rewrite landed. Boot
    /// must finish both, and in the right order — rewrite group
    /// membership before `reconcile_groups` runs — or reconcile sees
    /// "sake" as a plain dangling reference (nothing registered under
    /// that name any more) and drops it instead of carrying it to
    /// "shochu". This is the regression `boot_with`'s ordering exists
    /// to prevent.
    #[test]
    fn an_unfinished_context_rename_is_resumed_at_boot_before_group_reconciliation() {
        let dir = scratch_dir("rename-context-crash");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            state
                .create_group(
                    "drinks",
                    String::new(),
                    BTreeSet::from(["sake".to_string()]),
                    BTreeSet::new(),
                )
                .unwrap();
        }
        // No manual file move: `scan_data_dir` performs it itself once
        // it sees the marker, exactly as it would resuming a real crash.
        fs::write(
            renaming_marker_path(&dir, &file_stem("sake")),
            serde_json::to_vec(&RenameMarker {
                from: "sake".to_string(),
                to: "shochu".to_string(),
            })
            .unwrap(),
        )
        .unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.directory_entry("sake").is_none());
        assert!(state.directory_entry("shochu").is_some());
        assert!(!dir.join("sake.ctx").exists());
        assert!(dir.join("shochu.ctx").exists());
        assert_eq!(
            state.group("drinks").unwrap().contexts,
            BTreeSet::from(["shochu".to_string()]),
            "the membership must be REWRITTEN to the new name, not pruned as dangling"
        );
        assert!(!renaming_marker_path(&dir, &file_stem("sake")).exists());

        let _ = fs::remove_dir_all(dir);
    }

    /// The group-rename twin of
    /// `an_unfinished_context_rename_is_resumed_at_boot_before_group_reconciliation`:
    /// a surviving `.grouprenaming` marker must resume the file move
    /// AND rewrite the PARENT's `groups` set to the new child name
    /// before `reconcile_groups` runs, or the parent loses the child
    /// as a dangling reference instead of following it to its new name.
    #[test]
    fn an_unfinished_group_rename_is_resumed_at_boot_before_group_reconciliation() {
        let dir = scratch_dir("rename-group-crash");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            state
                .create_group(
                    "liquor",
                    String::new(),
                    BTreeSet::from(["sake".to_string()]),
                    BTreeSet::new(),
                )
                .unwrap();
            state
                .create_group(
                    "drinks",
                    String::new(),
                    BTreeSet::new(),
                    BTreeSet::from(["liquor".to_string()]),
                )
                .unwrap();
        }
        fs::write(
            groups::group_renaming_marker_path(&dir, &file_stem("liquor")),
            serde_json::to_vec(&RenameMarker {
                from: "liquor".to_string(),
                to: "spirits".to_string(),
            })
            .unwrap(),
        )
        .unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.group("liquor").is_none());
        let spirits = state
            .group("spirits")
            .expect("the renamed group must exist");
        assert_eq!(spirits.contexts, BTreeSet::from(["sake".to_string()]));
        assert_eq!(
            state.group("drinks").unwrap().groups,
            BTreeSet::from(["spirits".to_string()]),
            "the parent must be REWRITTEN to the new child name, not pruned as dangling"
        );
        assert!(!groups::group_path(&dir, &file_stem("liquor")).exists());
        assert!(groups::group_path(&dir, &file_stem("spirits")).exists());
        assert!(!groups::group_renaming_marker_path(&dir, &file_stem("liquor")).exists());

        let _ = fs::remove_dir_all(dir);
    }

    /// A rename that half-finished with `sake` as its SOURCE leaves a
    /// `.renaming` marker at sake's stem and frees the name to be created
    /// again on the same live server. The create must strip that marker,
    /// or the next boot's resume-sweep moves the fresh generation onto
    /// the rename's destination and `sake` silently becomes `shochu`.
    #[test]
    fn creating_a_context_abandons_a_rename_marker_at_its_own_stem() {
        let dir = scratch_dir("create-ctx-clears-source-marker");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            fs::write(
                renaming_marker_path(&dir, &file_stem("sake")),
                serde_json::to_vec(&RenameMarker {
                    from: "sake".to_string(),
                    to: "shochu".to_string(),
                })
                .unwrap(),
            )
            .unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            assert!(
                !renaming_marker_path(&dir, &file_stem("sake")).exists(),
                "create must clear a rename marker sitting at its own stem"
            );
        }
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(
            state.directory_entry("sake").is_some(),
            "the freshly created context must survive, not be swept to the rename's destination"
        );
        assert!(state.directory_entry("shochu").is_none());
        assert!(dir.join("sake.ctx").exists());
        assert!(!dir.join("shochu.ctx").exists());

        let _ = fs::remove_dir_all(dir);
    }

    /// A rename that half-finished with `sake` as its DESTINATION leaves
    /// its marker under the SOURCE's stem (`beer`) — a stem the create of
    /// `sake` cannot derive from its own name. Creating `sake` must scan
    /// for markers naming it as `to` and drop them, or the next boot's
    /// resume-sweep renames the stale `beer` family onto the fresh `sake`
    /// (fs::rename overwrites), clobbering it and erasing `beer`.
    #[test]
    fn creating_a_context_abandons_a_rename_marker_naming_it_as_destination() {
        let dir = scratch_dir("create-ctx-clears-destination-marker");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("beer", ContextMeta::default()).unwrap();
            fs::write(
                renaming_marker_path(&dir, &file_stem("beer")),
                serde_json::to_vec(&RenameMarker {
                    from: "beer".to_string(),
                    to: "sake".to_string(),
                })
                .unwrap(),
            )
            .unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            assert!(
                !renaming_marker_path(&dir, &file_stem("beer")).exists(),
                "create must clear a rename marker that names it as the destination"
            );
        }
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(
            state.directory_entry("beer").is_some(),
            "the abandoned rename must leave the untouched source context intact"
        );
        assert!(
            state.directory_entry("sake").is_some(),
            "the freshly created destination context must survive, not be overwritten by the source"
        );
        assert!(dir.join("beer.ctx").exists());
        assert!(dir.join("sake.ctx").exists());

        let _ = fs::remove_dir_all(dir);
    }

    /// The group twin of
    /// `creating_a_context_abandons_a_rename_marker_at_its_own_stem`: a
    /// `.grouprenaming` marker at the created group's own stem must be
    /// abandoned so boot does not resume-move the fresh group onto the
    /// rename's destination.
    #[test]
    fn creating_a_group_abandons_a_rename_marker_at_its_own_stem() {
        let dir = scratch_dir("create-group-clears-source-marker");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            fs::write(
                groups::group_renaming_marker_path(&dir, &file_stem("liquor")),
                serde_json::to_vec(&RenameMarker {
                    from: "liquor".to_string(),
                    to: "spirits".to_string(),
                })
                .unwrap(),
            )
            .unwrap();
            state
                .create_group("liquor", String::new(), BTreeSet::new(), BTreeSet::new())
                .unwrap();
            assert!(
                !groups::group_renaming_marker_path(&dir, &file_stem("liquor")).exists(),
                "create_group must clear a rename marker sitting at its own stem"
            );
        }
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.group("liquor").is_some());
        assert!(state.group("spirits").is_none());
        assert!(groups::group_path(&dir, &file_stem("liquor")).exists());
        assert!(!groups::group_path(&dir, &file_stem("spirits")).exists());

        let _ = fs::remove_dir_all(dir);
    }

    /// The group twin of
    /// `creating_a_context_abandons_a_rename_marker_naming_it_as_destination`:
    /// creating `spirits` must drop a `.grouprenaming` marker that names
    /// it as `to` (parked at `liquor`'s stem), or boot resume-moves the
    /// stale `liquor` group file over the fresh `spirits` and drops
    /// `liquor`.
    #[test]
    fn creating_a_group_abandons_a_rename_marker_naming_it_as_destination() {
        let dir = scratch_dir("create-group-clears-destination-marker");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create_group("liquor", String::new(), BTreeSet::new(), BTreeSet::new())
                .unwrap();
            fs::write(
                groups::group_renaming_marker_path(&dir, &file_stem("liquor")),
                serde_json::to_vec(&RenameMarker {
                    from: "liquor".to_string(),
                    to: "spirits".to_string(),
                })
                .unwrap(),
            )
            .unwrap();
            state
                .create_group("spirits", String::new(), BTreeSet::new(), BTreeSet::new())
                .unwrap();
            assert!(
                !groups::group_renaming_marker_path(&dir, &file_stem("liquor")).exists(),
                "create_group must clear a rename marker that names it as the destination"
            );
        }
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(
            state.group("liquor").is_some(),
            "the abandoned rename must leave the untouched source group intact"
        );
        assert!(
            state.group("spirits").is_some(),
            "the freshly created destination group must survive, not be overwritten by the source"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rename_context_error_cases() {
        let dir = scratch_dir("rename-context-errors");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state.create("beer", ContextMeta::default()).unwrap();

        assert!(matches!(
            state.rename_context("missing", "whatever"),
            Err(RenameContextError::NotFound)
        ));
        assert!(matches!(
            state.rename_context("sake", "beer"),
            Err(RenameContextError::AlreadyExists)
        ));
        assert!(matches!(
            state.rename_context("sake", ""),
            Err(RenameContextError::InvalidName)
        ));
        assert!(
            state.rename_context("sake", "sake").is_ok(),
            "renaming a name to itself is a no-op, not an error"
        );
        assert!(state.directory_entry("sake").is_some());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rename_group_error_cases() {
        let dir = scratch_dir("rename-group-errors");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create_group("drinks", String::new(), BTreeSet::new(), BTreeSet::new())
            .unwrap();
        state
            .create_group("food", String::new(), BTreeSet::new(), BTreeSet::new())
            .unwrap();

        assert!(matches!(
            state.rename_group("missing", "whatever"),
            Err(RenameGroupError::NotFound)
        ));
        assert!(matches!(
            state.rename_group("drinks", "food"),
            Err(RenameGroupError::AlreadyExists)
        ));
        assert!(matches!(
            state.rename_group("drinks", ""),
            Err(RenameGroupError::InvalidName)
        ));
        assert!(state.rename_group("drinks", "drinks").is_ok());
        assert!(state.group("drinks").is_some());

        let _ = fs::remove_dir_all(dir);
    }

    /// Same fence a create races against a slow delete in
    /// `a_create_racing_a_slow_delete_is_refused_not_interleaved`: a
    /// rename reserves both its names in `pending.renames` before it
    /// may touch any file, so a create for either name must be refused
    /// until the rename settles, never interleaved with it.
    #[test]
    fn a_create_racing_a_pending_context_rename_is_refused_for_both_names() {
        let dir = scratch_dir("rename-create-race");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();

        let entry = state.lookup("sake").unwrap();
        let stall = entry.inner.read();
        let renamer = {
            let state = state.clone();
            std::thread::spawn(move || state.rename_context("sake", "shochu").unwrap())
        };
        while !state.0.pending.lock().renames.contains("sake") {
            std::thread::yield_now();
        }
        assert!(
            matches!(
                state.create("sake", ContextMeta::default()),
                Err(CreateError::AlreadyExists)
            ),
            "the source name is reserved until the rename settles"
        );
        assert!(
            matches!(
                state.create("shochu", ContextMeta::default()),
                Err(CreateError::AlreadyExists)
            ),
            "the destination name is reserved too, before any file lands there"
        );

        drop(stall);
        renamer.join().unwrap();
        assert!(state.directory_entry("shochu").is_some());
        assert!(!state.0.pending.lock().renames.contains("sake"));
        assert!(!state.0.pending.lock().renames.contains("shochu"));

        let _ = fs::remove_dir_all(dir);
    }

    /// The rename twin of
    /// `a_passage_write_racing_a_delete_backs_off_at_the_tombstone`: a
    /// handle taken before the rename must see the tombstone after,
    /// not the old generation's live state, and no write may recreate
    /// the old name from under it.
    #[test]
    fn a_write_racing_a_rename_backs_off_at_the_tombstone() {
        let dir = scratch_dir("rename-write-race");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();

        let entry = state.lookup("sake").unwrap();
        state.rename_context("sake", "shochu").unwrap();
        assert!(
            entry.read_unless_deleted().is_none(),
            "a handle from before the rename must see the tombstone"
        );
        assert!(
            matches!(
                state.add_associations(
                    "sake",
                    vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("a.md"))],
                    Deadline::unbounded(),
                ),
                Err(AccessError::NotFound)
            ),
            "the old name is gone; nothing may recreate it via a write"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn cue_cache_get_promotes_recency_so_eviction_spares_the_touched_entry() {
        let mut cache = CueCache::default();
        for i in 0..CueCache::CAP {
            cache.insert(format!("cue{i}"), Arc::new(vec![i as f32]));
        }
        // Touching cue0 makes it the most recently used entry, even
        // though it was the first one inserted.
        assert!(cache.get("cue0").is_some());
        // The cache is at capacity, so this eviction must reach for the
        // least recently used entry — cue1, never touched again after
        // its insert — not the oldest insertion, which is cue0.
        cache.insert("fresh-cue".to_string(), Arc::new(vec![-1.0]));
        assert!(
            cache.get("cue0").is_some(),
            "a touched entry must survive the next eviction"
        );
        assert!(
            cache.get("cue1").is_none(),
            "the least recently used entry must be the one evicted"
        );
        assert!(cache.get("fresh-cue").is_some());
    }

    #[test]
    fn cue_cache_insert_does_not_overwrite_an_existing_key() {
        let mut cache = CueCache::default();
        cache.insert("cue".to_string(), Arc::new(vec![1.0]));
        cache.insert("cue".to_string(), Arc::new(vec![2.0]));
        assert_eq!(*cache.get("cue").unwrap(), vec![1.0]);
    }

    #[test]
    fn scan_data_dir_discovers_every_context_and_sorts_them_by_name() {
        let dir = scratch_dir("scan-parallel");
        let names = ["delta", "alpha", "charlie", "bravo", "echo", "foxtrot"];
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            for name in names {
                state
                    .create(name, ContextMeta::default())
                    .map_err(|_| "create")
                    .unwrap();
            }
        }

        // A fresh boot re-runs `scan_data_dir`'s worker-pool scan; the
        // registry it returns must still hold every context, keyed and
        // ordered by name regardless of which worker raced to finish
        // its disk reads first.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let found: Vec<String> = state.directory().into_iter().map(|e| e.name).collect();
        let mut expected: Vec<String> = names.iter().map(|s| s.to_string()).collect();
        expected.sort();
        assert_eq!(found, expected);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn passages_wal_bytes_stays_correct_after_eviction() {
        let dir = scratch_dir("passages-wal-gauge");
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
        assert!(
            state.gauge_snapshot().passages_wal_bytes > 0,
            "a freshly written passage log must show up as pending"
        );

        let entry = state.lookup("sake").unwrap();
        assert!(state.evict_entry("sake", &entry));
        // `evict_entry` compacts the store on the way down and caches
        // the resulting (now-zero) pending-log size on `EntryInner`;
        // the gauge must read that cached value rather than going
        // blind — or re-`stat`ing the log — once the context is cold.
        assert_eq!(
            state.gauge_snapshot().passages_wal_bytes,
            0,
            "eviction compacts the log, and the gauge must reflect that while cold"
        );

        let _ = fs::remove_dir_all(dir);
    }
}
