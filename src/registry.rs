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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use taguru::context::{AliasError, CompactionError, Context, LabelUsage, dead_ratio_of};
use taguru::deadline::Deadline;

use crate::embedding::{EmbedPurpose, EmbeddingProvider, PassageVectorStore, VectorStore};
use crate::groups::{self, GroupRecord};
use crate::metrics::{ContextGaugeRow, GaugeSnapshot, Metrics, PerContextMetrics};
#[cfg(test)]
use crate::storage::{clear_persistence_fault, fail_persistence_ops_after, write_atomic_private};
use crate::storage::{
    commit_staged, fsync_dir, lock_data_dir, offload, remove_persisted_file, stage_bytes,
    write_atomic,
};
use crate::wal::{self, WalOp};

mod associations;
mod boot;
mod concurrency;
mod context_io;
#[cfg(test)]
mod core_tests;
mod embeddings;
mod engine;
mod gauges;
mod group_ops;
mod lifecycle;
mod passages;
mod paths;
mod replication;
mod retrieval_cache;
mod search;
mod semantic_cache;
mod terms;
#[cfg(test)]
mod test_support;

pub use passages::PassagesWriteError;

pub(crate) use concurrency::{Semaphore, dispatch_chunks_concurrently, parallel_map};
pub(crate) use paths::{
    IMPORT_MARKER_EXTENSION, ImportMarker, ResumedRenames, bm25_path, deleted_marker_path,
    image_path, import_marker_path, import_marker_paths, meta_path, passages_path,
    passages_wal_path, pvectors_path, renaming_marker_path, resume_rename_markers, sources_path,
    vectors_path, wal_path,
};
use paths::{rename_markers_targeting, write_rename_marker};
pub(crate) use retrieval_cache::{CachedRetrieval, RetrievalKey};
pub(crate) use semantic_cache::SemanticFill;
pub(crate) use terms::{passage_terms, spelled_passage_terms};

/// Server-side metadata for one context: the prose half of the routing
/// directory plus the cache policy flag. `PartialEq` backs
/// [`AppState::update_meta`]'s changed-check: only a PATCH that
/// actually changed something bumps the config revision.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
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

/// Change counters for one context — the "has anything changed since I
/// last looked" token a retrieval cache keys on. Three counters rather
/// than one because the lanes invalidate independently: passage-search
/// results do not change on graph writes and graph reads do not change
/// on passage writes, so a consumer watches the counters it depends on
/// and compares for EQUALITY (never order across restarts).
///
/// - `graph`: applied graph write ops (associations, aliases,
///   retractions). Not `wal_seq`: that freezes with the WAL disabled
///   and rolls back on partial-batch bookkeeping, while this counts
///   what actually applied.
/// - `passages`: the passage log watermark (stores and retractions).
/// - `config`: server-side configuration that changes how requests are
///   answered — metadata updates (description, pin, floors) and
///   embedding refreshes that published something.
///
/// Guarantees: within one process every read is live and strictly
/// monotonic across changes; across a clean shutdown the persisted
/// values are exact. Across a crash a cold context can briefly serve a
/// lagging value until its first load catches the graph counter up to
/// the WAL replay — the same posture as the cold stats snapshot — and
/// a cache that outlives the process must treat a server restart (and
/// a delete-recreate of the same name) as invalidation: values can
/// repeat with different content there.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextRevision {
    pub graph: u64,
    pub passages: u64,
    pub config: u64,
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
    /// The revision counters as of this save — what a cold entry (and
    /// a replica's tailed refresh) seeds from. Defaulted for sidecars
    /// from before the field existed: those report zeros until their
    /// first load or flush catches them up.
    revision: ContextRevision,
}

/// One row of `GET /contexts` — the routing directory an LLM client
/// reads to decide which context to search, skills-style: a name, the
/// prose description, and the mechanical stats that keep it honest.
/// Stats are live for loaded contexts and the last saved snapshot for
/// cold ones.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Change counters a retrieval cache keys on — see
    /// [`ContextRevision`] for what each counts and what equality
    /// guarantees. `serde(default)` so a router merging rows from an
    /// older shard reads zeros instead of refusing the row.
    #[serde(default)]
    pub revision: ContextRevision,
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
    /// Serializes the storage-quota admission with the passage store
    /// it admits (issue #136). Passage stores run under the SHARED
    /// tombstone fence — concurrent with each other by design — so
    /// without this, two stores could read the same pre-write usage,
    /// both pass the quota gate, and only then serialize at the
    /// store's own writer mutex, already past the gate. The graph
    /// gate has no such gap: `logged_write` checks under the entry's
    /// EXCLUSIVE lock. Taken after the fence (a new lock-order edge
    /// with no cycle: nothing takes the fence after this), released
    /// once the append settles — the derived-index refresh needs no
    /// admission.
    passages_admission: Mutex<()>,
    /// [`ContextRevision::passages`]'s live value: the passage log
    /// watermark. An atomic, not an `EntryInner` field, because the
    /// passage mutators run under the SHARED tombstone fence (a read
    /// lock on `inner`) and cannot write through it. Advanced with
    /// `fetch_max` — store batches finish out of order, and a stale
    /// watermark read must never overwrite a newer one.
    passage_revision: AtomicU64,
    /// Flush-time snapshot of this context's non-WAL on-disk bytes —
    /// written by [`AppState::refresh_disk_usage`], read by
    /// `gauge_snapshot` (so a scrape never stats the data directory)
    /// and by the storage-quota gates (so a growth write never does
    /// either). Zeros until the first sweep — boot runs one when the
    /// per-context gauges are on or any storage ceiling is declared;
    /// with neither reader, they stay zeros, unread.
    disk: Mutex<ContextDiskUsage>,
}

/// On-disk bytes for the file families a scrape cannot afford to stat
/// (issue #137), grouped the way `taguru_context_disk_bytes` labels
/// them. The WAL lanes are deliberately absent: `EntryInner::wal_bytes`
/// and `passages_wal_bytes` already track those live, and a second
/// bookkeeper for the same number would drift from the first.
#[derive(Debug, Clone, Copy, Default)]
struct ContextDiskUsage {
    image_bytes: u64,
    passages_bytes: u64,
    /// Meta + sources + gloss vectors + passage vectors + BM25, summed.
    sidecar_bytes: u64,
}

impl Entry {
    fn new(
        meta: ContextMeta,
        stats: ContextStats,
        slot: Slot,
        wal_bytes: u64,
        passages_wal_bytes: u64,
        usage: ContextUsage,
        revision: ContextRevision,
    ) -> Self {
        Self {
            inner: RwLock::new(EntryInner {
                meta,
                stats,
                slot,
                wal_seq: 1,
                graph_revision: revision.graph,
                config_revision: revision.config,
                cache_identity: next_cache_identity(),
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
            passages_admission: Mutex::new(()),
            passage_revision: AtomicU64::new(revision.passages),
            disk: Mutex::new(ContextDiskUsage::default()),
        }
    }

    /// The three revision counters as one [`ContextRevision`] — the
    /// shape the directory serves and the sidecar persists. Takes the
    /// caller's `inner` guard (read or write) so every snapshot is
    /// consistent with whatever that caller is about to publish.
    fn revision_snapshot(&self, inner: &EntryInner) -> ContextRevision {
        ContextRevision {
            graph: inner.graph_revision,
            passages: self.passage_revision.load(Ordering::Relaxed),
            config: inner.config_revision,
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
    /// [`ContextRevision::graph`]'s live value: applied graph write
    /// ops, bumped in `logged_write` by the count that actually landed.
    /// Deliberately NOT `wal_seq` — that field freezes with the WAL
    /// disabled and rolls back on partial-batch log bookkeeping while
    /// the applied ops stand. Seeded from the sidecar at scan, floored
    /// with the replay top on load so a lagging sidecar catches up.
    graph_revision: u64,
    /// [`ContextRevision::config`]'s live value: metadata updates and
    /// embedding-refresh publications. Under the entry lock like its
    /// siblings so `update_meta` can bump-and-persist atomically (and
    /// roll both back together when the sidecar write fails).
    config_revision: u64,
    /// This incarnation's process-unique nonce — the retrieval cache's
    /// answer to delete-and-recreate, which restarts the revision
    /// counters at zero and could otherwise collide a recreated
    /// context's key with a cached entry of the old incarnation.
    /// Minted at construction ([`next_cache_identity`]) and re-minted
    /// by `replica_refresh`: a tailed refresh deliberately `max`es the
    /// revision counters (they must never walk backward), which pins
    /// them still across an upstream delete+recreate — the one path
    /// where content switches lineage under an unmoved revision, and
    /// exactly what a fresh nonce makes unreachable. Never persisted:
    /// the cache it guards dies with the process.
    cache_identity: u64,
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

/// Mints [`EntryInner::cache_identity`] values. A process-global
/// static rather than a field on `AppState` because entries are also
/// constructed before any state exists (the boot scan) — and the nonce
/// only has to be unique within this process, the retrieval cache's
/// whole lifetime.
fn next_cache_identity() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
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
    /// The context is at or over its declared storage ceiling
    /// (`TAGURU_CONTEXT_QUOTAS`), so a growth write was refused —
    /// 507 `storage_full`, the same client contract as the library's
    /// own capacity cap. Deliberately distinct from [`Self::Unpersisted`]:
    /// that 500 means the SERVER is failing to persist (an operator
    /// problem); this means the TENANT's allotment is spent (retract,
    /// compact, or raise the quota).
    QuotaExceeded(String),
}

/// The one refusal message every storage-quota gate serves (the write
/// path's two gates and the import loop's pre-check), naming the ways
/// down so a refused client is never stranded at the ceiling.
pub(crate) fn storage_quota_message(name: &str, used: u64, ceiling: u64) -> String {
    format!(
        "context '{name}' is at its storage quota ({used} of {ceiling} bytes): \
         retract or compact to shrink it, or raise its TAGURU_CONTEXT_QUOTAS entry"
    )
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

/// Default trigger for ratio-triggered auto-compaction (issue #135):
/// compact once a context's dead ratio strictly exceeds this — dead
/// weight outgrowing live content, the graph-side restatement of the
/// passages store's own `COMPACT_RATIO = 1` (pending log outgrows the
/// snapshot). Edges, not bytes, because the graph's dead weight has an
/// exact edge measure ([`Context::dead_ratio`]) while its byte measure
/// is a lower bound only (arena slack), and the maintenance API
/// already drew the operational line on the same scale.
pub const DEFAULT_AUTO_COMPACT_RATIO: f64 = 0.5;

/// One context's declared ceilings (issue #136), parsed from the
/// `TAGURU_CONTEXT_QUOTAS` JSON env. Both fields optional — a
/// declaration may cap disk, cache share, or both — but never neither
/// (the parser refuses an empty quota). `deny_unknown_fields` because a
/// typo'd field name must fail loudly, not silently mean "unlimited".
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextQuota {
    /// On-disk ceiling across the context's whole file family — the
    /// same sum `taguru_context_disk_bytes` serves. At or over it,
    /// growth writes are refused with 507 `storage_full`; shrink paths
    /// (retract, unalias, compact, delete) stay open — they are how a
    /// tenant gets back under.
    #[serde(default)]
    pub storage_bytes: Option<u64>,
    /// Maximum resident share within the global `TAGURU_CACHE_BYTES`:
    /// not a reservation — slack stays usable by anyone — but under
    /// pressure a context past this is evicted before any compliant
    /// one, so the eviction damage one saturating context can inflict
    /// on the rest is bounded by its ceiling. Pinning wins over this:
    /// a pinned context never enters the sweep at all.
    #[serde(default)]
    pub cache_bytes: Option<u64>,
}

/// Parses `TAGURU_CONTEXT_QUOTAS` — one JSON object mapping context
/// names to [`ContextQuota`]s, the same declarative-policy shape as
/// `TAGURU_KEY_SCOPES`. And the same failure posture: a deployment that
/// DECLARED quotas must not run without them, so any parse or
/// validation error refuses boot (the caller exits) instead of the
/// env module's usual warn-and-default. Naming a context that does not
/// exist yet is fine — contexts are created at runtime, and the
/// declaration simply waits for the name.
pub fn parse_context_quotas(json: Option<&str>) -> Result<HashMap<String, ContextQuota>, String> {
    let Some(json) = json else {
        return Ok(HashMap::new());
    };
    if json.trim().is_empty() {
        return Err("TAGURU_CONTEXT_QUOTAS is set but empty".to_string());
    }
    let quotas: HashMap<String, ContextQuota> = serde_json::from_str(json).map_err(|error| {
        format!(
            "TAGURU_CONTEXT_QUOTAS is not the documented JSON shape \
             ({{\"name\": {{\"storage_bytes\": …, \"cache_bytes\": …}}}}): {error}"
        )
    })?;
    for (name, quota) in &quotas {
        if quota.storage_bytes.is_none() && quota.cache_bytes.is_none() {
            return Err(format!(
                "TAGURU_CONTEXT_QUOTAS declares no ceiling for '{name}' — \
                 give it storage_bytes, cache_bytes, or drop the entry"
            ));
        }
        if quota.storage_bytes == Some(0) || quota.cache_bytes == Some(0) {
            return Err(format!(
                "TAGURU_CONTEXT_QUOTAS gives '{name}' a zero ceiling, which would refuse \
                 every write (storage) or every residency (cache) — scope keys to \
                 read-only via TAGURU_KEY_SCOPES if that is the intent"
            ));
        }
    }
    Ok(quotas)
}

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
    pub per_context_metrics: PerContextMetrics,
    pub auto_compact: Option<f64>,
    /// Per-context ceilings (issue #136). Empty from
    /// [`BootConfig::from_env`]: the declaration is policy for the
    /// HTTP surface, so only `serve` parses `TAGURU_CONTEXT_QUOTAS`
    /// (refusing boot on a broken one) and assigns it here — the
    /// offline commands run as the operator, outside the policy.
    pub context_quotas: HashMap<String, ContextQuota>,
}

/// Default ceiling on how many rows per context get a vector
/// (`TAGURU_PASSAGE_VECTOR_LIMIT`; a row is a paragraph text or one of
/// its doc2query questions). The vector lane's footprint is
/// rows × dimensions × 4 bytes — 20 000 rows of a 1536-dim model
/// is ~120 MiB — and past the limit the lexical lane still serves
/// every paragraph; only the semantic lane goes partial.
pub const DEFAULT_PASSAGE_VECTOR_LIMIT: usize = 20_000;

// The ANN path must stay reachable in default configuration: a store
// capped below the activation threshold would make the whole
// [`crate::embedding::PassageAnnIndex`] dead code, as it silently was
// when the threshold sat at 50k over a 20k default limit. Bumping
// either constant past the other is a decision, not an accident — make
// it fail the build.
const _: () = assert!(crate::embedding::PASSAGE_ANN_THRESHOLD <= DEFAULT_PASSAGE_VECTOR_LIMIT);

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

/// One passage search's whole account: the served hits plus what each
/// lane did for this call — the response-level half of the evidence
/// posture ([`PassageSearchHit`] carries the per-hit half). The handler
/// serializes `lanes` into the response's `plan`, so a caller can tell
/// "the semantic lane found nothing" from "the semantic lane never ran"
/// without a separate explain call. `filter` is the source filter's
/// account (#167), present exactly when the call carried one.
#[derive(Debug)]
pub struct PassageSearch {
    pub hits: Vec<PassageSearchHit>,
    pub lanes: PassageSearchLanes,
    pub filter: Option<SourceFilterReport>,
}

/// What one search's source filter (#167) selected: how many sources
/// were eligible to answer, out of how many the context stores — the
/// numbers the response plan reports so an empty page under a narrow
/// filter is diagnosable without a second call.
#[derive(Debug, Clone, Copy)]
pub struct SourceFilterReport {
    pub eligible: usize,
    pub total: usize,
}

/// What the two lanes did for one search call, including the two early
/// returns that answer before either lane starts — an empty result must
/// still say WHY it is empty.
#[derive(Debug)]
pub(crate) enum PassageSearchLanes {
    /// The query yields no searchable terms; the search answered the
    /// empty list before either lane ran.
    NoQueryTerms,
    /// `limit` was 0; nothing could be served, so neither lane ran.
    ZeroLimit,
    /// The lexical lane ran; `vector` is the semantic lane's account.
    Ran { vector: VectorLaneStatus },
}

impl PassageSearchLanes {
    /// The one lane state that recovers with no revision bump (the
    /// provider comes back; no key moves) — the handlers' fill gate:
    /// a transiently degraded result must not be pinned into the
    /// retrieval caches, where it would keep serving BM25-only answers
    /// (and canonicalize paraphrases onto them) until an unrelated
    /// write. The stable skip states stay cacheable — a vector publish
    /// or config change DOES move the key.
    pub(crate) fn embedding_failed(&self) -> bool {
        matches!(
            self,
            Self::Ran {
                vector: VectorLaneStatus::QueryEmbeddingFailed(_)
            }
        )
    }
}

/// The semantic lane's response-level account: [`VectorLaneReport`]'s
/// taxonomy minus the explain-only per-target cosine. Two types on
/// purpose — explain scores one named target and needs the cosine;
/// search reports the lane as a whole and must not pretend to have
/// one. The api layer maps both through one set of wire reason
/// strings, so the taxonomies cannot drift apart in prose.
#[derive(Debug)]
pub(crate) enum VectorLaneStatus {
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
    /// The sidecar's rows have a different dimension than the provider
    /// now answers (a dimensions setting changed behind a stable model
    /// name, #133). Scored anyway they would all be `similarity`'s
    /// silent 0.0 — so they are never served, and the next refresh
    /// discards and re-embeds them, exactly like a model change.
    WidthChanged { stored: usize, current: usize },
    /// The sweep ran, dropping matches below `floor` — the effective
    /// value after the override → context setting → server default
    /// chain, the one threshold a caller cannot reconstruct alone.
    Ran { floor: f32 },
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
pub(super) struct FusedHit {
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
    /// The source exists and is validly addressed, but the request's
    /// source filter (#167) excludes it — the search being explained
    /// never considered it, so no lane evidence could apply.
    FilteredOut,
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
    /// The sidecar's rows have a different dimension than the provider
    /// now answers — never served, re-embedded by the next refresh
    /// (see [`VectorLaneStatus::WidthChanged`]).
    WidthChanged { stored: usize, current: usize },
    /// The sweep ran under `floor`; `cosine` is the target's best
    /// current-text row (text or doc2query question), `None` when it
    /// has none — not yet embedded, or embedded before its last edit.
    Ran { floor: f32, cosine: Option<f32> },
}

/// Why the vector lane can or cannot sweep an entry's paragraphs —
/// search takes the `Ready` arm and silently skips the rest; explain
/// names them in its [`VectorLaneReport`] (which carries the
/// provider-configured distinction itself).
pub(super) enum PassageVectorGate {
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
    /// The gloss sidecar's vectors have a different dimension than the
    /// provider now answers — every cosine would be `similarity`'s
    /// silent 0.0, so the sweep is refused and named instead (#133);
    /// the next refresh discards and re-embeds them.
    WidthChanged { stored: usize, current: usize },
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

/// What `GET /contexts/{name}/embeddings` serves: the provider this
/// server is configured to call beside the (model, width) identity
/// each vector sidecar actually carries. The facts a calibration
/// report ties its floor to (#131) — and the state an operator reads
/// after a model switch without provoking a search (#133 refuses
/// mismatched vectors at serve time; this names the standing state).
#[derive(Debug, Serialize)]
pub struct EmbeddingsStatus {
    /// The configured embedding model — `None` when embeddings are off.
    pub provider_model: Option<String>,
    /// The gloss sidecar; absent until a refresh has embedded a gloss.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub glosses: Option<GlossSidecarStatus>,
    /// The passage sidecar; absent until a refresh has embedded a row.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub passages: Option<PassageSidecarStatus>,
}

/// The gloss vector sidecar's identity and size.
#[derive(Debug, Serialize)]
pub struct GlossSidecarStatus {
    pub model: String,
    /// The one width every stored row shares (the uniformity invariant
    /// `VectorStore::width` documents).
    pub width: usize,
    pub concepts: usize,
    pub labels: usize,
}

/// The passage vector sidecar's identity and size.
#[derive(Debug, Serialize)]
pub struct PassageSidecarStatus {
    pub model: String,
    pub width: usize,
    pub rows: usize,
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
    /// Whether (and how much of) the per-context gauge families the
    /// scrape carries — see [`PerContextMetrics`]. Off skips the
    /// flush-time disk sweeps entirely, not just the render.
    pub per_context_metrics: PerContextMetrics,
    /// Ratio-triggered auto-compaction (issue #135): `Some(ratio)`
    /// makes each flusher tick compact the worst context whose dead
    /// ratio strictly exceeds `ratio` (at most one per tick, behind
    /// the heavy-ops permit its caller takes); `None` restores the
    /// manual-only posture. Defaults on at
    /// [`DEFAULT_AUTO_COMPACT_RATIO`], matching the passages store's
    /// own unasked self-compaction.
    pub auto_compact: Option<f64>,
    /// Per-context storage/cache ceilings (issue #136), keyed by
    /// context name. Empty means no context is capped — the default,
    /// and the only shape the offline commands ever pass.
    pub context_quotas: HashMap<String, ContextQuota>,
    /// Present when replication is on: the shipper's progress map,
    /// consulted (never waited on) before the graph lane's
    /// housekeeping WAL reset so that shipped stream stays gapless —
    /// see [`crate::ship::ShipProgress::allows_reset`], and the
    /// `StateInner` field's doc for why the passage lane's reset
    /// deliberately does not consult it.
    pub(crate) ship_progress: Option<Arc<crate::ship::ShipProgress>>,
    /// Present when this boot hydrates lazily from the bucket (issue
    /// #128): boot registers the manifest's contexts alongside the
    /// scanned ones, and every load materializes its family through
    /// [`crate::hydrate::Hydrator::ensure_context`] first.
    pub(crate) hydrator: Option<Arc<crate::hydrate::Hydrator>>,
    /// Present under `serve --replica` (issue #129): reads only, the
    /// tailer owns the disk — see the `StateInner` field.
    pub(crate) replica: Option<Arc<crate::replica::ReplicaInfo>>,
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
            per_context_metrics: PerContextMetrics::Off,
            auto_compact: Some(DEFAULT_AUTO_COMPACT_RATIO),
            context_quotas: HashMap::new(),
            ship_progress: None,
            hydrator: None,
            replica: None,
        }
    }
}

impl BootConfig {
    pub fn from_env() -> Self {
        Self {
            data_dir: PathBuf::from(
                std::env::var("TAGURU_DATA_DIR").unwrap_or_else(|_| "data".into()),
            ),
            cache_bytes: crate::env::env_number("TAGURU_CACHE_BYTES", 512 * 1024 * 1024),
            // The WAL closes the flush-interval loss window; opting out
            // (TAGURU_WAL=0) restores the old posture for benchmarks or
            // explicit risk acceptance.
            wal_enabled: crate::env::env_bool("TAGURU_WAL", true),
            // Backstop for a persistently failing flush: past this,
            // writes are refused rather than growing the log without
            // bound (0 = no cap).
            wal_max_bytes: crate::env::env_number("TAGURU_WAL_MAX_BYTES", DEFAULT_WAL_MAX_BYTES),
            // The passage log's own backstop; it only engages when
            // compaction is demonstrably stuck (see PassageStore).
            passages_wal_max_bytes: crate::env::env_number(
                "TAGURU_PASSAGES_WAL_MAX_BYTES",
                DEFAULT_PASSAGES_WAL_MAX_BYTES,
            ),
            // Paragraph embedding is opt-in on top of the provider
            // being configured — a corpus is orders of magnitude more
            // text than its glosses.
            embed_passages: crate::env::env_bool("TAGURU_EMBED_PASSAGES", false),
            passage_vector_limit: crate::env::env_number(
                "TAGURU_PASSAGE_VECTOR_LIMIT",
                DEFAULT_PASSAGE_VECTOR_LIMIT,
            ),
            // Worker threads dispatching each 128-item embedding chunk to
            // the provider concurrently; 1 keeps the old strictly-
            // sequential behavior. Raise to match the provider's rate
            // limit, not the machine's core count.
            embed_parallel: crate::env::env_number("TAGURU_EMBED_PARALLEL", 1),
            // The right semantic floor is a property of the embedding
            // model (cosine bands differ per model), so its
            // recalibration lives beside TAGURU_EMBED_MODEL rather
            // than on every context.
            semantic_floor: crate::env::env_floor("TAGURU_SEMANTIC_FLOOR"),
            per_context_metrics: crate::env::env_per_context_metrics("TAGURU_METRICS_PER_CONTEXT"),
            // Default on, like the passages store's own self-compaction;
            // TAGURU_AUTO_COMPACT=0 restores the manual-only posture.
            auto_compact: crate::env::env_auto_compact(
                "TAGURU_AUTO_COMPACT",
                "TAGURU_AUTO_COMPACT_RATIO",
            ),
            // Serve assigns the parsed TAGURU_CONTEXT_QUOTAS after this;
            // the offline commands keep it empty (see the field's doc).
            context_quotas: HashMap::new(),
        }
    }

    /// [`AppState::boot_with`], parameterized by this configuration.
    /// `ship_progress` and `hydrator` are present only under `serve`
    /// with replication on — the offline commands (import, export,
    /// compact) pass `None` and their writes reach the bucket through
    /// the next serve's baseline sync.
    pub fn boot(
        &self,
        embedder: Option<Arc<dyn EmbeddingProvider>>,
        ship_progress: Option<Arc<crate::ship::ShipProgress>>,
        hydrator: Option<Arc<crate::hydrate::Hydrator>>,
        replica: Option<Arc<crate::replica::ReplicaInfo>>,
    ) -> io::Result<AppState> {
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
                per_context_metrics: self.per_context_metrics,
                auto_compact: self.auto_compact,
                context_quotas: self.context_quotas.clone(),
                ship_progress,
                hydrator,
                replica,
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
    /// The provider's circuit breaker, when it carries one (the HTTP
    /// provider does; test mocks do not): [`AppState::timed_embed`]'s
    /// pre-flight gate, and the source of the
    /// `taguru_embedding_breaker_*` series on /metrics. A clone of the
    /// breaker the provider itself consults — one shared state, two
    /// readers.
    embed_breaker: Option<crate::embedding::EmbedBreaker>,
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
    /// The exact-match retrieval cache (issue #150): an identical
    /// request against an identical corpus state answers from the
    /// stored response. Unlike `cue_cache`, validity is
    /// corpus-dependent, so every key carries the targets' revision
    /// lanes and identity — see the module doc.
    retrieval_cache: Mutex<retrieval_cache::RetrievalCache>,
    /// The semantic retrieval tier (issue #153): equivalence claims
    /// that rewrite a paraphrased passage query onto its canonical's
    /// exact-cache key. Holds no payloads and needs no invalidation —
    /// see the module doc. Off unless
    /// `TAGURU_SEMANTIC_CACHE_THRESHOLD` is set.
    semantic_cache: Mutex<semantic_cache::SemanticCache>,
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
    /// concurrency. Sizes both the outer per-context worker pool (the
    /// flush tick's `parallel_map`) and the inner per-chunk one
    /// (`dispatch_chunks_concurrently`); `embed_provider_slots` below is
    /// what keeps those two from multiplying past it.
    embed_parallel: usize,
    /// The actual global ceiling on concurrent provider calls from a
    /// refresh — sized to `embed_parallel`, acquired around every
    /// `timed_embed_for_refresh` call. Refresh dispatch is nested (the
    /// flush tick's per-context pool fans out into each context's own
    /// per-chunk pool), so without this a busy tick could reach
    /// `embed_parallel²` concurrent calls; this permit is what makes
    /// `embed_parallel` the true process-wide bound the field above
    /// documents. Query-time embedding (`cue_vector`) does not draw
    /// from this pool — a single ad hoc call per search request was
    /// never part of the multiplication this bounds, and gating it here
    /// would queue interactive searches behind bulk refresh work.
    embed_provider_slots: Semaphore,
    /// Whether (and how much of) the per-context gauge families the
    /// scrape carries (`TAGURU_METRICS_PER_CONTEXT`, issue #137). Read
    /// twice: [`AppState::refresh_disk_usage`] skips its stat sweep on
    /// `Off` — unless a declared storage quota needs it — and
    /// `gauge_snapshot` collects rows (and applies `Top`'s cut) for
    /// anything else.
    per_context_metrics: PerContextMetrics,
    /// The auto-compaction trigger (`TAGURU_AUTO_COMPACT`, issue
    /// #135), read only by [`AppState::auto_compact_candidate`] —
    /// which the flusher tick calls; a replica never consults it
    /// because a replica runs no flusher at all.
    auto_compact: Option<f64>,
    /// Declared per-context ceilings (`TAGURU_CONTEXT_QUOTAS`, issue
    /// #136). Storage ceilings gate the three growth entrances
    /// ([`AppState::logged_write`], [`AppState::store_passages`], the
    /// import loop's per-batch pre-check) and widen
    /// [`AppState::refresh_disk_usage`]'s sweep condition; cache
    /// ceilings reorder [`AppState::enforce_budget`]'s eviction. Both
    /// surface on the per-context gauge rows. Empty on every offline
    /// command and on a replica's storage side (writes are refused
    /// there before any gate) — though a replica's eviction ordering
    /// honors cache ceilings like anyone else's.
    context_quotas: HashMap<String, ContextQuota>,
    /// Present when replication is on. Consulted (one mutexed map
    /// read, never a wait) before the GRAPH lane's post-flush WAL
    /// reset, so the shipper reads that log's tail before it
    /// disappears. Deliberately NOT consulted by the passage lane's
    /// post-compaction reset: compaction is self-triggered from
    /// inside the write path by a ratio of `log_bytes`, so deferring
    /// its reset would leave `log_bytes` high and re-fire a full
    /// snapshot rewrite on every subsequent store — putting a slow
    /// bucket back on the write path through the side door. The
    /// shipper's series-restart path (snapshot first, then a fresh
    /// series) keeps the passage lane restore-correct without any
    /// hook; the cost is only a briefly wider RPO window for records
    /// compacted away un-shipped, at compaction's low cadence. See
    /// [`crate::ship::ShipProgress`].
    ship_progress: Option<Arc<crate::ship::ShipProgress>>,
    /// Present when this boot hydrates lazily from the bucket (issue
    /// #128). Every family materializes through it exactly once,
    /// before its first load — `ensure_hot` and `entry_passages` for
    /// reads and writes, `delete`/`rename_context` (which also veto
    /// re-materialization) for the file-level operations.
    hydrator: Option<Arc<crate::hydrate::Hydrator>>,
    /// Present under `serve --replica` (issue #129): this process
    /// serves reads only, the tailer owns the data directory's
    /// contents, and every path that would persist locally-derived
    /// state (usage counters, a rebuilt BM25 sidecar, an eviction's
    /// passage compaction) stays memory-only — a replica's disk is a
    /// verified cache of the bucket manifest, and local writes would
    /// only diverge it into refetch churn. The value carries what the
    /// write-refusal names: the writer's URL and the bucket's fence
    /// holder.
    replica: Option<Arc<crate::replica::ReplicaInfo>>,
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
        // Passages can be a context's very first touch (a passage
        // search needs no graph load), so the family hydrates here
        // too — idempotent with `ensure_hot`'s hook, and quarantined
        // the same way on failure.
        if let Some(hydrator) = &self.0.hydrator
            && let Err(error) = hydrator.ensure_context(stem)
        {
            let refusal = format!("passage family hydration failed: {error}");
            *entry.passages_load_failure.lock() =
                Some((std::time::Instant::now(), refusal.clone()));
            return Err(io::Error::new(io::ErrorKind::InvalidData, refusal));
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
        // The load just replayed the log, so this is the exact
        // watermark — it catches a sidecar-seeded value up after a
        // crash, and every search loads the store before computing, so
        // a cache fill never keys on the stale seed.
        entry
            .passage_revision
            .fetch_max(store.watermark(), Ordering::Relaxed);
        let store = Arc::new(store);
        *slot = Some(Arc::clone(&store));
        Ok(store)
    }

    /// One provider round trip, timed into the embed-latency histogram
    /// whatever the outcome — the ok/failed counters cannot tell a
    /// slow provider from a down one; the histogram can. An open
    /// circuit breaker refuses BEFORE the call and outside the timing:
    /// a short-circuit is not a provider round trip, and a burst of
    /// ~0s refusal samples would make a dead provider read as a fast
    /// one on the histogram.
    fn timed_embed(
        &self,
        embedder: &dyn EmbeddingProvider,
        texts: &[&str],
        purpose: EmbedPurpose,
        deadline: Deadline,
    ) -> Result<Vec<Vec<f32>>, String> {
        if let Some(breaker) = &self.0.embed_breaker
            && let Some(refusal) = breaker.refusal()
        {
            return Err(refusal);
        }
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
        // Held across the call, not just the dispatch: this is the one
        // choke point every refresh chunk funnels through regardless of
        // which context or which dispatch layer it came from, so it is
        // where the outer and inner worker pools' ceilings actually
        // become one global ceiling instead of two that multiply.
        let _permit = self.0.embed_provider_slots.acquire();
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

    /// Advances a context's config revision after an embedding refresh
    /// published vectors that differ from what was served before — and
    /// persists the sidecar so the bump survives a restart. Called
    /// AFTER the publish (never under the refresh's shared fence — this
    /// takes the entry's write lock), so a reader observing the new
    /// value always sees the new content; a failed sidecar write only
    /// lags the durable copy (healed by the next flush), never the
    /// served one. A tombstoned entry skips both halves: the delete
    /// owns the name and its files.
    fn bump_config_revision(&self, name: &str, entry: &Entry) {
        let Some(mut guard) = entry.lock_unless_deleted() else {
            return;
        };
        let inner = &mut *guard;
        inner.config_revision += 1;
        // A replica's refresh state is its own ephemeral view; a local
        // sidecar write would diverge the cache the tailer owns.
        if self.is_replica() {
            return;
        }
        if let Err(error) = write_meta(
            &self.0.data_dir,
            &file_stem(name),
            &inner.meta,
            &inner.stats,
            &entry.usage.snapshot(),
            entry.revision_snapshot(inner),
        ) {
            tracing::warn!(
                "config revision for '{name}' not persisted (lags until the next flush): {error}"
            );
        }
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
            ensure_hot(
                &self.0.data_dir,
                name,
                &mut inner,
                &self.0.metrics,
                self.0.hydrator.as_deref(),
            )
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
            ensure_hot(
                &self.0.data_dir,
                name,
                &mut inner,
                &self.0.metrics,
                self.0.hydrator.as_deref(),
            )
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
        revision: entry.revision_snapshot(&inner),
    })
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
    hydrator: Option<&crate::hydrate::Hydrator>,
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
    // A lazy bucket boot materializes the family before its first
    // load; hydrated families return instantly. A family that cannot
    // be materialized cannot be loaded: same failure, same quarantine
    // (and the same retry cadence once the bucket recovers).
    if let Some(hydrator) = hydrator
        && let Err(error) = hydrator.ensure_context(&stem)
    {
        let error = format!("context '{name}' hydration failed: {error}");
        metrics.record_cache_load(false);
        inner.load_failure = Some((std::time::Instant::now(), error.clone()));
        return Err(error);
    }
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
    // Floor, don't overwrite: after a crash the sidecar-seeded value
    // lags the WAL's tail and the replay top is the truth — but with
    // the WAL disabled (or a WAL-off history) the sidecar value is the
    // one that kept counting and the stale top must not regress it.
    inner.graph_revision = inner.graph_revision.max(top);
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

fn save_files(
    dir: &Path,
    name: &str,
    meta: &ContextMeta,
    stats: &ContextStats,
    usage: &ContextUsage,
    revision: ContextRevision,
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
    write_meta(dir, &stem, meta, stats, usage, revision)?;
    write_atomic(&image_path(dir, &stem), &context.to_bytes())
}

fn write_meta(
    dir: &Path,
    stem: &str,
    meta: &ContextMeta,
    stats: &ContextStats,
    usage: &ContextUsage,
    revision: ContextRevision,
) -> io::Result<()> {
    let file = MetaFile {
        meta: meta.clone(),
        stats: stats.clone(),
        usage: usage.clone(),
        revision,
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

/// One context's whole file family, by stem — the delete loop and the
/// boot-time deletion sweep must never disagree about what "the whole
/// family" means, so both read this one list. Built from the same nine
/// path builders every other caller uses, so a file kind added there
/// cannot silently miss this list.
pub(crate) fn context_files(stem: &str) -> [String; 9] {
    let unrooted = Path::new("");
    [
        image_path(unrooted, stem),
        meta_path(unrooted, stem),
        sources_path(unrooted, stem),
        passages_path(unrooted, stem),
        passages_wal_path(unrooted, stem),
        pvectors_path(unrooted, stem),
        bm25_path(unrooted, stem),
        vectors_path(unrooted, stem),
        wal_path(unrooted, stem),
    ]
    .map(|path| path.to_string_lossy().into_owned())
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
