//! Hand-rolled observability: RED metrics per route plus domain
//! counters (cache, flush, embedding), rendered as Prometheus text on
//! demand. Hand-rolled on purpose — the fixed catalog below needs a
//! few atomics and one render function, not a metrics facade crate;
//! the same reasoning that keeps BM25 and the vector store in-tree.
//!
//! Hot-path cost is one atomic increment per event. Histograms store
//! per-bin counts and defer the cumulative `le` semantics to render
//! (scrape) time.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{MatchedPath, Request, State};
use axum::http::Method;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::registry::AppState;

/// (upper bound in ms, its Prometheus `le` label). Fixed strings keep
/// the rendered form stable — `le="1"`, never a float-formatting
/// surprise like `le="1.0"`.
const LATENCY_BUCKETS: [(u64, &str); 8] = [
    (1, "0.001"),
    (5, "0.005"),
    (10, "0.01"),
    (50, "0.05"),
    (100, "0.1"),
    (500, "0.5"),
    (1000, "1"),
    (5000, "5"),
];

/// One latency distribution. `counts[i]` is the exclusive bin ending
/// at `LATENCY_BUCKETS[i]` — NOT cumulative; the `_bucket{le=…}`
/// prefix sums are computed at render time. `count` doubles as the
/// `+Inf` bucket and the `_count` line (the exposition format defines
/// them to be equal), so it also counts observations past the largest
/// finite bound.
///
/// Guarded by one mutex rather than three independent atomics: three
/// separate atomic loads (buckets, then sum, then count) can each land
/// at a different instant, and a render that catches an in-flight
/// `observe()` between its bucket increment and its count increment
/// sees a finite bucket that already includes the new observation but
/// a `+Inf`/`_count` that does not yet — an invalid histogram, since
/// `+Inf` must never be less than a finite bucket. Locking the whole
/// read (and the whole write) makes every render see one consistent
/// instant instead.
#[derive(Default)]
struct Histogram {
    state: Mutex<HistogramState>,
}

#[derive(Default, Clone, Copy)]
struct HistogramState {
    counts: [u64; LATENCY_BUCKETS.len()],
    sum_micros: u64,
    count: u64,
}

/// A [`Histogram`] read at one consistent instant: cumulative
/// `_bucket{le=…}` values (ascending, one per finite bound), the
/// running sum, and the total count all agree with each other.
struct HistogramSnapshot {
    cumulative: [u64; LATENCY_BUCKETS.len()],
    sum_micros: u64,
    count: u64,
}

impl Histogram {
    fn observe(&self, elapsed: Duration) {
        // Bucket at microsecond precision: `as_millis` truncates, so a
        // 1.9 ms observation would land in the `le="0.001"` bucket —
        // every fractional latency slid one bucket optimistic, and the
        // low buckets, where this server's common case lives, are
        // exactly where that skews `histogram_quantile` the most.
        let micros = elapsed.as_micros();
        let mut state = self.state.lock().unwrap();
        if let Some(bin) = LATENCY_BUCKETS
            .iter()
            .position(|&(bound, _)| micros <= u128::from(bound) * 1000)
        {
            state.counts[bin] += 1;
        }
        state.sum_micros += elapsed.as_micros() as u64;
        state.count += 1;
    }

    fn snapshot(&self) -> HistogramSnapshot {
        let state = self.state.lock().unwrap();
        let mut running = 0u64;
        let mut cumulative = [0u64; LATENCY_BUCKETS.len()];
        for (slot, count) in cumulative.iter_mut().zip(&state.counts) {
            running += count;
            *slot = running;
        }
        HistogramSnapshot {
            cumulative,
            sum_micros: state.sum_micros,
            count: state.count,
        }
    }
}

/// Per-(method, route template) statistics.
#[derive(Default)]
struct RouteStat {
    /// Status → count. Bounded per route: a route emits a handful of
    /// distinct statuses over its life.
    by_status: RwLock<HashMap<u16, AtomicU64>>,
    latency: Histogram,
}

/// The whole registry: shared through `AppState`, so the HTTP
/// middleware, the handlers, and the spawned flusher all reach the
/// same instance.
#[derive(Default)]
pub struct Metrics {
    /// (method, route template) → stats, interned on first sight. The
    /// lazy `RwLock<HashMap<…, Arc<…>>>` mirrors the registry's own
    /// entry map idiom. Route templates keep cardinality bounded —
    /// raw paths would mint one series per context name.
    http: RwLock<HashMap<(String, String), Arc<RouteStat>>>,
    cache_hits: AtomicU64,
    cache_loads_ok: AtomicU64,
    cache_loads_failed: AtomicU64,
    evictions_ok: AtomicU64,
    evictions_failed: AtomicU64,
    flush_ok: AtomicU64,
    flush_failed: AtomicU64,
    wal_appends_ok: AtomicU64,
    wal_appends_failed: AtomicU64,
    embed_refresh_ok: AtomicU64,
    embed_refresh_failed: AtomicU64,
    embed_resolve_ok: AtomicU64,
    embed_resolve_failed: AtomicU64,
    gloss_width_rebuilds: AtomicU64,
    passage_width_rebuilds: AtomicU64,
    errors_load: AtomicU64,
    errors_wal_refused: AtomicU64,
    errors_io: AtomicU64,
    errors_panic: AtomicU64,
    /// `[op][outcome]`, outcome 0 = hit, 1 = empty.
    searches: [[AtomicU64; 2]; SearchOp::ALL.len()],
    /// `[op][outcome]`, outcome 0 = hit, 1 = miss — the exact-match
    /// retrieval cache's pulse. Disjoint from `searches` on purpose:
    /// that family counts served retrievals (cache hits included, so
    /// dashboards stay continuous); this one says how they were
    /// computed. Nothing lands here while the cache is disabled.
    retrieval_cache: [[AtomicU64; 2]; RetrievalCacheOp::ALL.len()],
    /// `[outcome]` per [`SemanticCacheOutcome`] — the semantic tier's
    /// pulse, disjoint from `retrieval_cache` the same way that family
    /// is disjoint from `searches`: a probe's rewritten exact lookup
    /// is counted HERE (hit/stale), never there. Nothing lands here
    /// while the tier is disabled.
    semantic_cache: [AtomicU64; SemanticCacheOutcome::ALL.len()],
    resolve_tiers: [AtomicU64; ResolveTier::ALL.len()],
    /// Passage-search hits by which lane(s) surfaced them — the pulse
    /// of what the vector lane actually adds. Fixed three labels.
    passage_hits_bm25_only: AtomicU64,
    passage_hits_vector_only: AtomicU64,
    passage_hits_both_lanes: AtomicU64,
    /// Set while ANY context's most recent flush attempt is unhealed;
    /// cleared once every failing context has flushed clean again. A
    /// lock-free mirror of `flush_failing.is_empty()` so /health reads it
    /// without taking the lock. Drives /health: the flusher retries every
    /// tick, so this is a self-healing signal, never a latched one.
    flush_degraded: AtomicBool,
    /// The contexts whose latest flush failed, by name. Tracked as a set,
    /// not a single bit, so one context's success cannot mask another's
    /// failure (last-write-wins would), and a lone transient failure among
    /// many healthy contexts does not flip the whole server to 503. A
    /// context leaves the set when its next flush succeeds.
    flush_failing: Mutex<HashSet<String>>,
    /// Unix seconds of the last successful image flush (0 = none since
    /// boot). `time() - this` on a dashboard says how stale images are
    /// without knowing the flush interval.
    last_flush_success_epoch: AtomicU64,
    /// Embedding-provider round-trip latency (retries included) — the
    /// ok/failed counters say THAT the provider misbehaves; this says
    /// how slowly. Calls past the top finite bucket (5s) still land in
    /// `+Inf`/`_count`, so a provider crawling toward its timeout is
    /// visible as a growing tail.
    embed_latency: Histogram,
    /// Requests currently inside the stack (probes exempt) — the load
    /// signal behind the in-flight ceiling, and a gauge on /metrics
    /// either way.
    inflight: AtomicUsize,
    /// Requests refused at the ceiling with a 503 — sustained growth
    /// means the server is saturated, not slow.
    requests_shed: AtomicU64,
    /// Set for the duration of a `POST /maintenance/compact` sweep: closes
    /// `/health` (503 `maintenance`, distinct from an actual fault),
    /// `enforce_concurrency` (early 503 instead of admitting new work),
    /// and `spawn_flusher` (skips its tick rather than racing the sweep).
    maintenance: AtomicBool,
    /// Set when the periodic flusher's most recent tick panicked instead
    /// of completing — a bug, not a disk fault. `/health`'s flush signal
    /// is exactly that loop's own outcome (see `health` below); without
    /// this, `spawn_flusher` catching the panic to keep the loop alive
    /// would otherwise look identical to any other quiet tick, and the
    /// probe would report healthy right through a flusher that stopped
    /// making progress on every subsequent tick too. Cleared by the next
    /// tick that completes without panicking — self-healing, same shape
    /// as `flush_degraded`.
    flusher_panicked: AtomicBool,
    /// Ratio-triggered auto-compaction (issue #135): how many the
    /// flusher ran, by outcome; how many image bytes the successful
    /// ones shed; and when one last succeeded (unix seconds, 0 = none
    /// since boot — the same "how stale" convention as
    /// `last_flush_success_epoch`). Manual compactions are absent by
    /// design: they answer their caller directly, while these count
    /// the loop nobody watches.
    auto_compact_ok: AtomicU64,
    auto_compact_failed: AtomicU64,
    auto_compact_reclaimed_bytes: AtomicU64,
    auto_compact_last_success_epoch: AtomicU64,
    /// Growth writes refused at a declared storage ceiling
    /// (`TAGURU_CONTEXT_QUOTAS`, issue #136) — every gate counts here:
    /// the graph write path, the passage store path, and the import
    /// loop's per-batch pre-check. Deliberately not an `errors_*`
    /// counter: a refusal at the ceiling is the policy working, and a
    /// tenant hammering a full context should read as its own signal,
    /// not as server trouble.
    storage_quota_refusals: AtomicU64,
    /// Keyring hot reloads (issue #134): applied swaps (unchanged
    /// no-ops included — the reload RAN) and refusals that kept the
    /// previous table armed. The refusal counter is the alertable
    /// half: a rotation the operator believes they performed but that
    /// never took effect is invisible in request traffic — the old
    /// keys keep working — so stderr's error line needs a
    /// dashboard-visible twin.
    keyring_reloads: AtomicU64,
    keyring_reload_refusals: AtomicU64,
    /// Replication ("WAL shipping") counters. Uploads and errors are
    /// plain counters; `replication_fenced` LATCHES — unlike
    /// `flush_degraded` there is no retry loop behind it, because a
    /// fenced shipper stops for good by design, and the latch is the
    /// dashboard-visible half of that fail-stop.
    replication_uploads: AtomicU64,
    replication_errors: AtomicU64,
    replication_fenced: AtomicBool,
    /// Unix seconds of the last cycle that shipped everything it found
    /// (0 = none since boot) — `time() - this` on a dashboard bounds
    /// the DR restore's data loss window, the number this feature
    /// exists to shrink.
    replication_last_success_epoch: AtomicU64,
    /// (context, lane) → how far the local log is beyond the shipped
    /// one, refreshed by the shipper each cycle. BTreeMap so the
    /// rendered series come out sorted — render must stay
    /// deterministic.
    ///
    /// The ONE deliberate exception to this file's "no context names
    /// in labels" rule (see [`GaugeSnapshot`]): a restore's loss
    /// window is per lane by nature — an aggregate would hide exactly
    /// the one stuck context an operator needs named. Cardinality
    /// stays bounded the way the route map's does: one series per
    /// live lane, populated only while replication is on, and dropped
    /// (`forget_replication_lane`) when the context's family leaves
    /// the disk. Values are escaped at render (`escape_label`) since
    /// names are client-minted text.
    replication_lag: Mutex<BTreeMap<(String, &'static str), ReplicationLag>>,
    /// Replica-side telemetry (issue #129), populated only under
    /// `serve --replica` — the whole family renders only then, so a
    /// writer's scrape stays exactly what it was. The lag map mirrors
    /// `replication_lag`'s shape (and its deliberate context-name
    /// labels): per lane, the seq this replica has applied vs the
    /// newest the bucket ships, and since when the two diverge — the
    /// promotion-time RPO, on display.
    replica_mode: AtomicBool,
    replica_generation: AtomicU64,
    replica_manifest_epoch: AtomicU64,
    replica_last_poll_epoch: AtomicU64,
    replica_poll_errors: AtomicU64,
    replica_lag: Mutex<BTreeMap<(String, &'static str), ReplicaLag>>,
}

/// One log lane's shipping lag as the dashboard sees it: records not
/// yet in the bucket, and how long the oldest of them has waited.
#[derive(Clone, Copy, Default)]
struct ReplicationLag {
    behind_records: u64,
    age_secs: u64,
}

/// One log lane's tail lag as a replica sees it: the record seq its
/// local materialization carries vs the newest the manifest ships,
/// and (unix seconds, 0 = caught up) since when it has been behind.
#[derive(Clone, Copy, Default)]
struct ReplicaLag {
    applied_seq: u64,
    shipped_seq: u64,
    behind_since_epoch: u64,
}

/// Why a request answered 500. The status code alone cannot separate
/// these — and they demand different responses from an operator: `load`
/// is a corrupt or unreadable image (restore from backup), `wal_refused`
/// is the durability path failing writes (check the disk NOW), `io` is
/// a sidecar or image file operation failing outside the WAL path, and
/// `panic` is a handler unwinding on a bug — not a disk problem, so it
/// warrants a bug report instead of an operator remedy.
#[derive(Clone, Copy)]
pub enum ErrorKind {
    Load,
    WalRefused,
    Io,
    Panic,
}

/// The retrieval operations whose hit/empty split is tracked — the
/// aggregate "is the memory answering" pulse. A fixed set on purpose:
/// ops are the labels, so the family's cardinality is sealed here.
#[derive(Clone, Copy)]
pub enum SearchOp {
    Resolve,
    ResolveLabel,
    Recall,
    Query,
    Activate,
    SearchPassages,
    Explore,
}

impl SearchOp {
    const ALL: [SearchOp; 7] = [
        SearchOp::Resolve,
        SearchOp::ResolveLabel,
        SearchOp::Recall,
        SearchOp::Query,
        SearchOp::Activate,
        SearchOp::SearchPassages,
        SearchOp::Explore,
    ];

    fn as_str(self) -> &'static str {
        match self {
            SearchOp::Resolve => "resolve",
            SearchOp::ResolveLabel => "resolve_label",
            SearchOp::Recall => "recall",
            SearchOp::Query => "query",
            SearchOp::Activate => "activate",
            SearchOp::SearchPassages => "search_passages",
            SearchOp::Explore => "explore",
        }
    }
}

/// The retrieval surfaces the exact-match cache fronts — the label
/// vocabulary of `taguru_retrieval_cache_total`, and the cache key's
/// op discriminant (each op reads a different pair of revision lanes,
/// so the same request text under two ops must never collide). Cross
/// variants fold into their base op: the resolved target list already
/// distinguishes them in the key, and a per-variant label would split
/// the hit-rate signal without adding meaning.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RetrievalCacheOp {
    Recall,
    Query,
    SearchPassages,
}

impl RetrievalCacheOp {
    const ALL: [RetrievalCacheOp; 3] = [
        RetrievalCacheOp::Recall,
        RetrievalCacheOp::Query,
        RetrievalCacheOp::SearchPassages,
    ];

    fn as_str(self) -> &'static str {
        match self {
            RetrievalCacheOp::Recall => "recall",
            RetrievalCacheOp::Query => "query",
            RetrievalCacheOp::SearchPassages => "search_passages",
        }
    }

    /// The `searches` family op a cache hit replays its `note_search`
    /// under — kept here so the two vocabularies cannot drift.
    pub(crate) fn search_op(self) -> SearchOp {
        match self {
            RetrievalCacheOp::Recall => SearchOp::Recall,
            RetrievalCacheOp::Query => SearchOp::Query,
            RetrievalCacheOp::SearchPassages => SearchOp::SearchPassages,
        }
    }
}

/// How one semantic-cache probe ended — the label vocabulary of
/// `taguru_semantic_cache_total`. No op label: the tier fronts passage
/// search only. `guarded` and `stale` are first-class rather than
/// folded into `miss` because they are the tuning signals the
/// threshold's operational follow-up reads: `guarded` counts
/// paraphrase-close pairs the text guard split, `stale` counts claims
/// that held while the corpus moved on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SemanticCacheOutcome {
    /// A claim held and its rewritten exact key was live: served.
    Hit,
    /// A claim held but the rewritten key missed (a write bumped a
    /// lane, or the exact tier evicted the canonical): computed fresh.
    Stale,
    /// Cosine cleared the threshold but the negation/numeric/entity
    /// guard refused every candidate: computed fresh.
    Guarded,
    /// No candidate cleared the threshold: computed fresh.
    Miss,
}

impl SemanticCacheOutcome {
    const ALL: [SemanticCacheOutcome; 4] = [
        SemanticCacheOutcome::Hit,
        SemanticCacheOutcome::Stale,
        SemanticCacheOutcome::Guarded,
        SemanticCacheOutcome::Miss,
    ];

    fn as_str(self) -> &'static str {
        match self {
            SemanticCacheOutcome::Hit => "hit",
            SemanticCacheOutcome::Stale => "stale",
            SemanticCacheOutcome::Guarded => "guarded",
            SemanticCacheOutcome::Miss => "miss",
        }
    }
}

/// Which tier ultimately answered a resolve (or resolve_label) —
/// classified from the served payload, so every serve path lands in
/// exactly one bucket. The drift signal lives here: a rising
/// `semantic` share means the cues clients send are pulling away from
/// the stored vocabulary; a rising `miss` share means coverage gaps.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ResolveTier {
    /// A confident string match answered alone.
    Lexical,
    /// Embedding candidates were part of the answer.
    Semantic,
    /// Only sub-confidence string fragments survived (the semantic
    /// tier ran but contributed nothing, failed, or is not configured).
    WeakLexical,
    /// Nothing at all.
    Miss,
}

impl ResolveTier {
    const ALL: [ResolveTier; 4] = [
        ResolveTier::Lexical,
        ResolveTier::Semantic,
        ResolveTier::WeakLexical,
        ResolveTier::Miss,
    ];

    /// The stable name shared by the metric label and the search event
    /// log, so the two vocabularies can never drift apart.
    pub fn as_str(self) -> &'static str {
        match self {
            ResolveTier::Lexical => "lexical",
            ResolveTier::Semantic => "semantic",
            ResolveTier::WeakLexical => "weak_lexical",
            ResolveTier::Miss => "miss",
        }
    }
}

/// How much per-context detail the scrape carries
/// (`TAGURU_METRICS_PER_CONTEXT`, issue #137). Off by default on
/// purpose: per-context labels × many contexts is exactly the
/// cardinality blow-up the route-template rule at the top of this
/// file exists to prevent, so an operator opts in — and can bound a
/// large fleet's series count with `Top`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PerContextMetrics {
    /// No per-context families on the scrape (the default).
    #[default]
    Off,
    /// Every context gets its rows.
    All,
    /// Only the N largest contexts by total on-disk bytes get rows —
    /// membership shifts as sizes shift, which Prometheus handles as
    /// series going stale, not as an error.
    Top(usize),
}

/// One context's row behind the `taguru_context_*` families — collected
/// by `gauge_snapshot` only while [`PerContextMetrics`] asks for it.
/// Disk sizes come from flush-time bookkeeping, everything else from
/// registry state already in memory: a scrape never walks the data
/// directory (see `AppState::refresh_disk_usage`).
#[derive(Debug, Clone)]
pub struct ContextGaugeRow {
    pub name: String,
    pub pinned: bool,
    /// Graph plus cached gloss/passage/BM25/vector stores — the same
    /// per-entry accounting `taguru_resident_bytes` sums fleet-wide.
    pub resident_bytes: u64,
    pub disk_image_bytes: u64,
    pub disk_wal_bytes: u64,
    pub disk_passages_bytes: u64,
    pub disk_passages_wal_bytes: u64,
    /// Meta + sources + gloss vectors + passage vectors + BM25, summed.
    pub disk_sidecar_bytes: u64,
    /// Declared ceilings (`TAGURU_CONTEXT_QUOTAS`, issue #136), when
    /// this context has them — `None` renders no series at all, so an
    /// uncapped fleet's scrape is byte-identical to before quotas
    /// existed.
    pub quota_storage_bytes: Option<u64>,
    pub quota_cache_bytes: Option<u64>,
    pub concepts: u64,
    pub associations: u64,
    pub labels: u64,
    pub sources: u64,
}

impl ContextGaugeRow {
    /// Total on-disk bytes across every family — what a restore would
    /// move, and the ranking key for [`PerContextMetrics::Top`].
    pub fn disk_total_bytes(&self) -> u64 {
        self.disk_image_bytes
            + self.disk_wal_bytes
            + self.disk_passages_bytes
            + self.disk_passages_wal_bytes
            + self.disk_sidecar_bytes
    }
}

/// Point-in-time gauges, computed from the registry at scrape time
/// rather than maintained incrementally — they cannot drift.
pub struct GaugeSnapshot {
    pub contexts_registered: u64,
    pub groups_registered: u64,
    pub contexts_resident: u64,
    pub resident_bytes: u64,
    /// Total bytes across every context's write-ahead log. A healthy
    /// server truncates each log every flush interval; sustained
    /// growth here means images are failing to save.
    pub wal_bytes: u64,
    /// Total bytes across every context's PASSAGE log. This one grows
    /// legitimately up to about each context's snapshot size before its
    /// ratio-triggered compaction; growth far past the snapshots means
    /// compactions are failing.
    pub passages_wal_bytes: u64,
    /// Sum, across every context, of edges with `count == 0` — dead
    /// weight `compact` would shed right now. Deliberately NOT broken
    /// down per context here: unlike route templates, a context name is
    /// unbounded, user-chosen data, and this metrics surface only ever
    /// mints fixed-cardinality series (see `http`'s route-template
    /// comment). Per-context detail lives at `GET /contexts` and
    /// `taguru inspect` — or, opted into and bounded via
    /// `TAGURU_METRICS_PER_CONTEXT` (#137), in the `taguru_context_*`
    /// families below.
    pub dead_edges_total: u64,
    /// Sum, across every context, of attribution records unlinked from
    /// every chain but not yet reclaimed by compaction.
    pub dead_attributions_total: u64,
    /// Sum, across every context, of the lower-bound arena bytes behind
    /// removed aliases.
    pub arena_slack_total: u64,
    /// Sum, across every context, of edges carrying weight no named
    /// source explains — see [`taguru::context::Context::unsourced_summary`].
    pub unsourced_edges_total: u64,
    /// Sum, across every context, of unsourced weight (absolute value).
    pub unsourced_weight_total: f64,
    /// The embedding provider's circuit breaker, present exactly when a
    /// provider with one is configured — the family is absent from a
    /// lexical-only server's scrape, like the replica family off a
    /// writer.
    pub embed_breaker: Option<crate::embedding::BreakerSnapshot>,
    /// Entries resident in the exact-match retrieval cache, and the
    /// bytes they hold — read from the cache at scrape time like every
    /// other gauge here, so they cannot drift.
    pub retrieval_cache_entries: u64,
    pub retrieval_cache_bytes: u64,
    /// Equivalence claims resident in the semantic cache (slots, not
    /// bytes — payloads live in the exact tier).
    pub semantic_cache_entries: u64,
    /// Per-context rows, empty unless `TAGURU_METRICS_PER_CONTEXT`
    /// asked for them — the one other sanctioned exception (after the
    /// replication lag maps) to this file's no-context-labels rule,
    /// and gated the same way the replica family is: an opted-out
    /// scrape stays byte-identical to what it was.
    pub per_context: Vec<ContextGaugeRow>,
}

impl Metrics {
    /// Counts one request in, refusing past `limit` (0 = no ceiling,
    /// count only). Compare-and-swap so two racing admissions cannot
    /// both squeeze under the ceiling.
    pub(crate) fn admit_inflight(&self, limit: usize) -> bool {
        let mut current = self.inflight.load(Ordering::Relaxed);
        loop {
            if limit != 0 && current >= limit {
                return false;
            }
            match self.inflight.compare_exchange_weak(
                current,
                current + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    pub(crate) fn release_inflight(&self) {
        self.inflight.fetch_sub(1, Ordering::Relaxed);
    }

    /// Requests currently inside the stack, this call included if it was
    /// admitted through `admit_inflight`. A maintenance sweep polls this
    /// down to 1 (itself) to know every other request has drained.
    pub(crate) fn inflight_count(&self) -> usize {
        self.inflight.load(Ordering::Relaxed)
    }

    /// Attempts to become the sole maintenance sweep; `false` means one is
    /// already running, so the caller should answer 409 rather than queue
    /// behind it.
    pub(crate) fn try_enter_maintenance(&self) -> bool {
        self.maintenance
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
    }

    /// Whether a maintenance sweep currently holds the server closed to
    /// ordinary traffic.
    pub fn maintenance_active(&self) -> bool {
        self.maintenance.load(Ordering::Relaxed)
    }

    /// Reopens the server after a maintenance sweep. Idempotent, so it is
    /// safe to call unconditionally from a `Drop` guard on every exit path
    /// — success, deadline, or panic unwind.
    pub(crate) fn exit_maintenance(&self) {
        self.maintenance.store(false, Ordering::Release);
    }

    pub(crate) fn record_embed_latency(&self, elapsed: Duration) {
        self.embed_latency.observe(elapsed);
    }

    pub(crate) fn record_shed(&self) {
        self.requests_shed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_http(&self, method: &str, route: &str, status: u16, elapsed: Duration) {
        let stat = self.route_stat(method, route);
        stat.latency.observe(elapsed);
        if let Some(counter) = stat.by_status.read().unwrap().get(&status) {
            counter.fetch_add(1, Ordering::Relaxed);
            return;
        }
        stat.by_status
            .write()
            .unwrap()
            .entry(status)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    fn route_stat(&self, method: &str, route: &str) -> Arc<RouteStat> {
        if let Some(stat) = self
            .http
            .read()
            .unwrap()
            .get(&(method.to_string(), route.to_string()))
        {
            return Arc::clone(stat);
        }
        Arc::clone(
            self.http
                .write()
                .unwrap()
                .entry((method.to_string(), route.to_string()))
                .or_default(),
        )
    }

    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cache_load(&self, ok: bool) {
        let counter = if ok {
            &self.cache_loads_ok
        } else {
            &self.cache_loads_failed
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_eviction(&self, ok: bool) {
        let counter = if ok {
            &self.evictions_ok
        } else {
            &self.evictions_failed
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_flush(&self, name: &str, ok: bool) {
        let counter = if ok {
            &self.flush_ok
        } else {
            &self.flush_failed
        };
        counter.fetch_add(1, Ordering::Relaxed);
        // Track WHICH contexts are failing, not just the last outcome:
        // a single global bit let context B's success erase context A's
        // still-unhealed failure (and one transient failure flip the whole
        // server to 503). Health stays degraded while the set is non-empty.
        {
            let mut failing = self.flush_failing.lock().unwrap();
            if ok {
                failing.remove(name);
            } else {
                failing.insert(name.to_string());
            }
            self.flush_degraded
                .store(!failing.is_empty(), Ordering::Relaxed);
        }
        if ok {
            let epoch = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|since| since.as_secs())
                .unwrap_or(0);
            self.last_flush_success_epoch
                .store(epoch, Ordering::Relaxed);
        }
    }

    /// Whether every context's most recent image flush succeeded (true
    /// when none has run yet — an idle server is a healthy server), AND
    /// the flusher loop itself is still making it to a tick's end rather
    /// than panicking out from under `/health`'s only signal.
    pub fn flush_is_healthy(&self) -> bool {
        !self.flush_degraded.load(Ordering::Relaxed)
            && !self.flusher_panicked.load(Ordering::Relaxed)
    }

    /// Unix seconds of the last successful flush; 0 when none yet.
    pub fn last_flush_success_epoch(&self) -> u64 {
        self.last_flush_success_epoch.load(Ordering::Relaxed)
    }

    /// Record whether the flusher's most recent tick completed without
    /// panicking. Called from `spawn_flusher`'s `catch_unwind` boundary
    /// once per tick, success or not — so a later clean tick clears the
    /// flag the same way a later clean flush clears `flush_degraded`.
    pub fn record_flusher_tick(&self, ok: bool) {
        self.flusher_panicked.store(!ok, Ordering::Relaxed);
    }

    /// Whether the flusher's most recent tick panicked. Distinct from
    /// `flush_is_healthy` (which folds this in) so `health()` can pick a
    /// reason message that names the actual fault.
    pub fn flusher_panicked(&self) -> bool {
        self.flusher_panicked.load(Ordering::Relaxed)
    }

    /// Record one ratio-triggered auto-compaction (issue #135):
    /// `Some(bytes)` — the resident bytes the rebuild shed — counts a
    /// success and stamps the last-success clock; `None` counts a
    /// failure. Failures never latch anything: the candidate's ratio
    /// still exceeds the trigger, so the next tick retries it, and the
    /// counter pair is what makes a stuck retry loop visible.
    pub fn record_auto_compaction(&self, reclaimed_bytes: Option<u64>) {
        match reclaimed_bytes {
            Some(bytes) => {
                self.auto_compact_ok.fetch_add(1, Ordering::Relaxed);
                self.auto_compact_reclaimed_bytes
                    .fetch_add(bytes, Ordering::Relaxed);
                let epoch = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|since| since.as_secs())
                    .unwrap_or(0);
                self.auto_compact_last_success_epoch
                    .store(epoch, Ordering::Relaxed);
            }
            None => {
                self.auto_compact_failed.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Count one growth write refused at a declared storage ceiling
    /// (issue #136) — called by the gate that refuses, never by the
    /// error mapping, so a refusal is counted exactly once wherever
    /// it surfaces.
    pub fn record_storage_quota_refusal(&self) {
        self.storage_quota_refusals.fetch_add(1, Ordering::Relaxed);
    }

    /// Count one keyring reload attempt (issue #134) by whether a
    /// table (possibly identical) was armed or the previous one kept.
    pub fn record_keyring_reload(&self, applied: bool) {
        let counter = if applied {
            &self.keyring_reloads
        } else {
            &self.keyring_reload_refusals
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_wal_append(&self, ok: bool) {
        let counter = if ok {
            &self.wal_appends_ok
        } else {
            &self.wal_appends_failed
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_embed_refresh(&self, ok: bool) {
        let counter = if ok {
            &self.embed_refresh_ok
        } else {
            &self.embed_refresh_failed
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_embed_resolve(&self, ok: bool) {
        let counter = if ok {
            &self.embed_resolve_ok
        } else {
            &self.embed_resolve_failed
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// One width-triggered sidecar wipe: the provider changed vector
    /// width behind an unchanged model name, and the refresh discarded
    /// and re-embedded the whole gloss store. Counted beside the warn
    /// line so the rebuild (and its provider spend) is graphable, not
    /// just greppable.
    pub fn record_gloss_width_rebuild(&self) {
        self.gloss_width_rebuilds.fetch_add(1, Ordering::Relaxed);
    }

    /// The passage-store twin of [`Self::record_gloss_width_rebuild`].
    pub fn record_passage_width_rebuild(&self) {
        self.passage_width_rebuilds.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self, kind: ErrorKind) {
        let counter = match kind {
            ErrorKind::Load => &self.errors_load,
            ErrorKind::WalRefused => &self.errors_wal_refused,
            ErrorKind::Io => &self.errors_io,
            ErrorKind::Panic => &self.errors_panic,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// One successful retrieval, split by whether it matched anything.
    /// Error responses never land here — a 500 is not an empty search.
    pub fn record_search(&self, op: SearchOp, empty: bool) {
        self.searches[op as usize][usize::from(empty)].fetch_add(1, Ordering::Relaxed);
    }

    /// One retrieval-cache consultation. Only consultations count: a
    /// disabled cache, or a request the cache cannot key (a target
    /// vanishing mid-request), records nothing rather than a fake miss.
    pub fn record_retrieval_cache(&self, op: RetrievalCacheOp, hit: bool) {
        self.retrieval_cache[op as usize][usize::from(!hit)].fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_semantic_cache(&self, outcome: SemanticCacheOutcome) {
        self.semantic_cache[outcome as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A cache hit's replay of [`Metrics::record_passage_hit`]: the
    /// fresh path records lane contributions per served hit, so a hit
    /// serving the same response must land the same counts — in bulk,
    /// since the cached entry stores tallies, not hits.
    pub fn record_passage_hit_counts(&self, bm25_only: u64, both_lanes: u64, vector_only: u64) {
        self.passage_hits_bm25_only
            .fetch_add(bm25_only, Ordering::Relaxed);
        self.passage_hits_both_lanes
            .fetch_add(both_lanes, Ordering::Relaxed);
        self.passage_hits_vector_only
            .fetch_add(vector_only, Ordering::Relaxed);
    }

    pub fn record_resolve_tier(&self, tier: ResolveTier) {
        self.resolve_tiers[tier as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// One served passage-search hit, by which lane(s) put it there. A
    /// hit carries at least one lane by construction; anything else is
    /// counted nowhere rather than inventing a fourth label.
    pub fn record_passage_hit(&self, bm25: bool, vector: bool) {
        match (bm25, vector) {
            (true, true) => &self.passage_hits_both_lanes,
            (true, false) => &self.passage_hits_bm25_only,
            (false, true) => &self.passage_hits_vector_only,
            (false, false) => return,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_replication_upload(&self) {
        self.replication_uploads.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_replication_error(&self) {
        self.replication_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// One cycle that shipped everything it found. Deliberately NOT
    /// part of `/health`: a degraded bucket must page whoever watches
    /// the dashboards, never convince an orchestrator to restart a
    /// server whose local durability is fine.
    pub fn record_replication_success(&self) {
        let epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|since| since.as_secs())
            .unwrap_or(0);
        self.replication_last_success_epoch
            .store(epoch, Ordering::Relaxed);
    }

    /// Latches the fenced flag — the metric half of the shipper's
    /// fail-stop (the audit line is the other half). Never cleared:
    /// only a restart re-contests the bucket.
    pub fn record_replication_fenced(&self) {
        self.replication_fenced.store(true, Ordering::Relaxed);
    }

    /// Refreshes one lane's lag series, keyed (context, lane).
    pub fn note_replication_lane(
        &self,
        context: &str,
        lane: &'static str,
        behind_records: u64,
        age_secs: u64,
    ) {
        self.replication_lag.lock().unwrap().insert(
            (context.to_string(), lane),
            ReplicationLag {
                behind_records,
                age_secs,
            },
        );
    }

    /// Drops a deleted context's lane series so the scrape does not
    /// carry ghost labels forever.
    pub fn forget_replication_lane(&self, context: &str, lane: &'static str) {
        self.replication_lag
            .lock()
            .unwrap()
            .remove(&(context.to_string(), lane));
    }

    fn unix_now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|since| since.as_secs())
            .unwrap_or(0)
    }

    /// Flips this process's scrape into replica shape — set once at
    /// boot under `serve --replica`, never cleared (a role is a boot
    /// decision, not a runtime one).
    pub fn set_replica_mode(&self) {
        self.replica_mode.store(true, Ordering::Relaxed);
    }

    /// One tailer poll finished; errors count, successes stamp the
    /// freshness gauge. Like the shipper's counters, deliberately NOT
    /// part of `/health`: an unreachable bucket must page whoever
    /// watches dashboards, not convince an orchestrator to restart a
    /// replica that is still serving its watermark fine.
    pub fn record_replica_poll(&self, ok: bool) {
        if ok {
            self.replica_last_poll_epoch
                .store(Self::unix_now(), Ordering::Relaxed);
        } else {
            self.replica_poll_errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The generation this replica currently follows (its hydration
    /// target after the latest retarget).
    pub fn note_replica_generation(&self, generation: u64) {
        self.replica_generation.store(generation, Ordering::Relaxed);
    }

    /// The newest complete manifest's store-clock `last_modified`, as
    /// seen by the latest poll — `time() - this` plus the poll
    /// interval bounds this replica's staleness.
    pub fn note_replica_manifest(&self, modified: SystemTime) {
        let epoch = modified
            .duration_since(UNIX_EPOCH)
            .map(|since| since.as_secs())
            .unwrap_or(0);
        self.replica_manifest_epoch.store(epoch, Ordering::Relaxed);
    }

    /// One lane fully applied: applied == shipped, gap closed.
    pub fn note_replica_lane(
        &self,
        context: &str,
        lane: &'static str,
        applied_seq: u64,
        shipped_seq: u64,
    ) {
        let mut lag = self.replica_lag.lock().unwrap();
        let entry = lag.entry((context.to_string(), lane)).or_default();
        entry.applied_seq = applied_seq;
        entry.shipped_seq = shipped_seq;
        entry.behind_since_epoch = if applied_seq >= shipped_seq {
            0
        } else if entry.behind_since_epoch == 0 {
            Self::unix_now()
        } else {
            entry.behind_since_epoch
        };
    }

    /// The shipped side alone — for a lane whose family could not be
    /// applied this poll: the applied seq stays where it was (or at 0
    /// for a lane never applied), and the age starts counting.
    pub fn note_replica_shipped(&self, context: &str, lane: &'static str, shipped_seq: u64) {
        let mut lag = self.replica_lag.lock().unwrap();
        let entry = lag.entry((context.to_string(), lane)).or_default();
        entry.shipped_seq = shipped_seq;
        if entry.applied_seq < shipped_seq && entry.behind_since_epoch == 0 {
            entry.behind_since_epoch = Self::unix_now();
        }
    }

    /// Drops a vanished context's replica lag rows (both lanes).
    pub fn forget_replica_context(&self, context: &str) {
        let mut lag = self.replica_lag.lock().unwrap();
        lag.remove(&(context.to_string(), "graph"));
        lag.remove(&(context.to_string(), "passages"));
    }

    /// Clears every replica lag row — the tailer's move on a
    /// generation switch. Applied seqs are meaningful only within one
    /// lineage: a successor that started from an older watermark (a
    /// promotion that lost the deposed writer's tail) ships LOWER
    /// seqs, and a predecessor's applied value surviving beside them
    /// would read as caught-up on a lane that has not applied the new
    /// lineage at all. The rows rebuild from the switch's own apply
    /// results, so a family that fails to land shows its gap from
    /// zero instead of a stale success.
    pub fn reset_replica_lanes(&self) {
        self.replica_lag.lock().unwrap().clear();
    }

    /// The full Prometheus text-exposition body. Deterministic: the
    /// dynamic keys (routes, statuses) are sorted before emission, so
    /// identical state renders byte-identical output.
    pub fn render_prometheus(&self, gauges: &GaugeSnapshot) -> String {
        let mut out = String::new();

        let http = self.http.read().unwrap();
        let mut routes: Vec<(&(String, String), &Arc<RouteStat>)> = http.iter().collect();
        routes.sort_by(|a, b| a.0.cmp(b.0));

        push_header(
            &mut out,
            "taguru_http_requests_total",
            "counter",
            "Total HTTP requests by method, route template, and status code.",
        );
        for ((method, route), stat) in &routes {
            let statuses = stat.by_status.read().unwrap();
            let mut by_status: Vec<(u16, u64)> = statuses
                .iter()
                .map(|(&status, count)| (status, count.load(Ordering::Relaxed)))
                .collect();
            by_status.sort_unstable_by_key(|&(status, _)| status);
            for (status, count) in by_status {
                out.push_str(&format!(
                    "taguru_http_requests_total{{method=\"{method}\",route=\"{route}\",status=\"{status}\"}} {count}\n"
                ));
            }
        }

        push_header(
            &mut out,
            "taguru_http_request_duration_seconds",
            "histogram",
            "HTTP request latency by method and route template.",
        );
        for ((method, route), stat) in &routes {
            let snapshot = stat.latency.snapshot();
            push_histogram(
                &mut out,
                "taguru_http_request_duration_seconds",
                &format!("method=\"{method}\",route=\"{route}\""),
                &snapshot,
            );
        }
        drop(http);

        // Fixed-cardinality families always emit every label
        // combination, zeros included, so dashboards never see an
        // absent series.
        push_value(
            &mut out,
            "taguru_cache_hits_total",
            "counter",
            "Context cache hits (already resident, no disk load).",
            self.cache_hits.load(Ordering::Relaxed),
        );
        push_outcomes(
            &mut out,
            "taguru_cache_loads_total",
            "Context loads from disk into the resident cache, by outcome.",
            &self.cache_loads_ok,
            &self.cache_loads_failed,
        );
        push_outcomes(
            &mut out,
            "taguru_cache_evictions_total",
            "Contexts evicted from the resident cache, by outcome (a failed eviction save keeps the context resident).",
            &self.evictions_ok,
            &self.evictions_failed,
        );
        push_outcomes(
            &mut out,
            "taguru_flush_total",
            "Dirty-context persistence attempts, by outcome.",
            &self.flush_ok,
            &self.flush_failed,
        );
        push_outcomes(
            &mut out,
            "taguru_wal_appends_total",
            "Write-ahead log append batches, by outcome (a failed append refuses the write).",
            &self.wal_appends_ok,
            &self.wal_appends_failed,
        );
        push_header(
            &mut out,
            "taguru_embedding_requests_total",
            "counter",
            "Embedding provider round trips, by operation and outcome.",
        );
        for (operation, ok, failed) in [
            (
                "refresh",
                &self.embed_refresh_ok,
                &self.embed_refresh_failed,
            ),
            (
                "resolve",
                &self.embed_resolve_ok,
                &self.embed_resolve_failed,
            ),
        ] {
            out.push_str(&format!(
                "taguru_embedding_requests_total{{operation=\"{operation}\",outcome=\"ok\"}} {}\n",
                ok.load(Ordering::Relaxed)
            ));
            out.push_str(&format!(
                "taguru_embedding_requests_total{{operation=\"{operation}\",outcome=\"failed\"}} {}\n",
                failed.load(Ordering::Relaxed)
            ));
        }
        push_header(
            &mut out,
            "taguru_embedding_width_rebuilds_total",
            "counter",
            "Sidecar wipes forced by the provider changing vector width \
             behind an unchanged model name (each re-embeds the whole \
             store), by store.",
        );
        for (store, counter) in [
            ("gloss", &self.gloss_width_rebuilds),
            ("passages", &self.passage_width_rebuilds),
        ] {
            out.push_str(&format!(
                "taguru_embedding_width_rebuilds_total{{store=\"{store}\"}} {}\n",
                counter.load(Ordering::Relaxed)
            ));
        }
        push_header(
            &mut out,
            "taguru_embedding_duration_seconds",
            "histogram",
            "Embedding provider round-trip latency, retries included \
             (calls past the top bucket land in +Inf). Circuit-breaker \
             short-circuits never touch the provider and are excluded.",
        );
        let snapshot = self.embed_latency.snapshot();
        push_histogram(&mut out, "taguru_embedding_duration_seconds", "", &snapshot);
        if let Some(breaker) = &gauges.embed_breaker {
            push_header(
                &mut out,
                "taguru_embedding_breaker_state",
                "gauge",
                "Embedding provider circuit breaker: 0 = closed, 1 = open \
                 (calls refused until the cooldown lets a probe through), \
                 2 = half-open (probing).",
            );
            out.push_str(&format!(
                "taguru_embedding_breaker_state {}\n",
                breaker.state
            ));
            push_header(
                &mut out,
                "taguru_embedding_breaker_consecutive_failures",
                "gauge",
                "Consecutive provider-attempt failures counted toward the \
                 breaker's opening threshold (pinned at the threshold while \
                 open or half-open).",
            );
            out.push_str(&format!(
                "taguru_embedding_breaker_consecutive_failures {}\n",
                breaker.consecutive_failures
            ));
            push_header(
                &mut out,
                "taguru_embedding_breaker_opened_total",
                "counter",
                "Times the breaker opened (threshold of consecutive attempt \
                 failures reached, or a half-open probe failed).",
            );
            out.push_str(&format!(
                "taguru_embedding_breaker_opened_total {}\n",
                breaker.opened_total
            ));
            push_header(
                &mut out,
                "taguru_embedding_breaker_short_circuits_total",
                "counter",
                "Embedding calls refused without touching the provider \
                 because the breaker was open (or a probe was already in \
                 flight).",
            );
            out.push_str(&format!(
                "taguru_embedding_breaker_short_circuits_total {}\n",
                breaker.short_circuits_total
            ));
        }

        push_header(
            &mut out,
            "taguru_searches_total",
            "counter",
            "Successful retrieval requests by operation and outcome \
             (empty = the operation ran but matched nothing).",
        );
        for op in SearchOp::ALL {
            for (outcome, slot) in [("hit", 0), ("empty", 1)] {
                out.push_str(&format!(
                    "taguru_searches_total{{op=\"{}\",outcome=\"{outcome}\"}} {}\n",
                    op.as_str(),
                    self.searches[op as usize][slot].load(Ordering::Relaxed)
                ));
            }
        }
        push_header(
            &mut out,
            "taguru_retrieval_cache_total",
            "counter",
            "Exact-match retrieval cache consultations by operation and outcome \
             (a hit serves the cached response; a miss computes and fills). \
             Requests served while the cache is disabled land in neither.",
        );
        for op in RetrievalCacheOp::ALL {
            for (outcome, slot) in [("hit", 0), ("miss", 1)] {
                out.push_str(&format!(
                    "taguru_retrieval_cache_total{{op=\"{}\",outcome=\"{outcome}\"}} {}\n",
                    op.as_str(),
                    self.retrieval_cache[op as usize][slot].load(Ordering::Relaxed)
                ));
            }
        }
        push_header(
            &mut out,
            "taguru_semantic_cache_total",
            "counter",
            "Semantic (similarity) cache probes for passage search, by outcome: \
             hit = an equivalent earlier query's cached result served; stale = \
             equivalence held but the corpus moved on; guarded = cosine cleared \
             the threshold but the negation/numeric/entity guard refused; miss = \
             no candidate cleared the threshold. Probes run only on an \
             exact-cache miss with TAGURU_SEMANTIC_CACHE_THRESHOLD set.",
        );
        for outcome in SemanticCacheOutcome::ALL {
            out.push_str(&format!(
                "taguru_semantic_cache_total{{outcome=\"{}\"}} {}\n",
                outcome.as_str(),
                self.semantic_cache[outcome as usize].load(Ordering::Relaxed)
            ));
        }
        push_header(
            &mut out,
            "taguru_resolves_total",
            "counter",
            "Resolve and resolve_label requests by the tier that answered: \
             lexical = confident string match, semantic = embedding candidates \
             included, weak_lexical = only sub-confidence string fragments, \
             miss = nothing. A rising semantic share means cues are drifting \
             from the stored vocabulary.",
        );
        for tier in ResolveTier::ALL {
            out.push_str(&format!(
                "taguru_resolves_total{{tier=\"{}\"}} {}\n",
                tier.as_str(),
                self.resolve_tiers[tier as usize].load(Ordering::Relaxed)
            ));
        }

        push_header(
            &mut out,
            "taguru_passage_lane_contributions_total",
            "counter",
            "Served passage-search hits by which lane surfaced them: vector_only \
             is what the semantic lane adds beyond BM25; a persistent zero there \
             means the embedding spend buys nothing this corpus needed.",
        );
        for (lane, counter) in [
            ("bm25_only", &self.passage_hits_bm25_only),
            ("both_lanes", &self.passage_hits_both_lanes),
            ("vector_only", &self.passage_hits_vector_only),
        ] {
            out.push_str(&format!(
                "taguru_passage_lane_contributions_total{{lane=\"{lane}\"}} {}\n",
                counter.load(Ordering::Relaxed)
            ));
        }

        push_header(
            &mut out,
            "taguru_errors_total",
            "counter",
            "Requests answered 500, by cause: load = image/WAL unreadable on load, \
             wal_refused = the WAL could not durably record a write (nothing applied), \
             io = a file operation failed outside the WAL path, panic = a handler \
             unwound on a bug.",
        );
        for (kind, counter) in [
            ("io", &self.errors_io),
            ("load", &self.errors_load),
            ("panic", &self.errors_panic),
            ("wal_refused", &self.errors_wal_refused),
        ] {
            out.push_str(&format!(
                "taguru_errors_total{{kind=\"{kind}\"}} {}\n",
                counter.load(Ordering::Relaxed)
            ));
        }

        push_value(
            &mut out,
            "taguru_contexts_registered",
            "gauge",
            "Contexts known to the registry.",
            gauges.contexts_registered,
        );
        push_value(
            &mut out,
            "taguru_groups_registered",
            "gauge",
            "Groups known to the registry.",
            gauges.groups_registered,
        );
        push_value(
            &mut out,
            "taguru_contexts_resident",
            "gauge",
            "Contexts currently resident in memory.",
            gauges.contexts_resident,
        );
        push_value(
            &mut out,
            "taguru_resident_bytes",
            "gauge",
            "Modeled resident estimate of loaded contexts and cached vector stores (graph and vector footprints, NOT process RSS).",
            gauges.resident_bytes,
        );
        push_value(
            &mut out,
            "taguru_retrieval_cache_entries",
            "gauge",
            "Entries resident in the exact-match retrieval cache.",
            gauges.retrieval_cache_entries,
        );
        push_value(
            &mut out,
            "taguru_retrieval_cache_bytes",
            "gauge",
            "Bytes the exact-match retrieval cache holds (serialized responses; bounded by TAGURU_RETRIEVAL_CACHE_BYTES).",
            gauges.retrieval_cache_bytes,
        );
        push_value(
            &mut out,
            "taguru_semantic_cache_entries",
            "gauge",
            "Equivalence claims resident in the semantic cache (slots; payloads live in the exact-match cache).",
            gauges.semantic_cache_entries,
        );
        push_value(
            &mut out,
            "taguru_wal_bytes",
            "gauge",
            "Total bytes across all write-ahead logs; sustained growth means image flushes are failing.",
            gauges.wal_bytes,
        );
        push_value(
            &mut out,
            "taguru_passages_wal_bytes",
            "gauge",
            "Total bytes across all passage logs; these legitimately grow to about each context's snapshot size, so alert on growth far past that.",
            gauges.passages_wal_bytes,
        );
        push_value(
            &mut out,
            "taguru_dead_edges",
            "gauge",
            "Live count of edges with count == 0 across all contexts — dead weight compaction would shed right now.",
            gauges.dead_edges_total,
        );
        push_value(
            &mut out,
            "taguru_dead_attributions",
            "gauge",
            "Live count of attribution records unlinked from every chain but not yet reclaimed, across all contexts.",
            gauges.dead_attributions_total,
        );
        push_value(
            &mut out,
            "taguru_arena_slack_bytes",
            "gauge",
            "Lower-bound arena bytes behind removed aliases across all contexts — bytes compaction would not carry forward.",
            gauges.arena_slack_total,
        );
        push_value(
            &mut out,
            "taguru_unsourced_edges",
            "gauge",
            "Live count of edges carrying weight no named source explains, across all contexts.",
            gauges.unsourced_edges_total,
        );
        push_value(
            &mut out,
            "taguru_unsourced_weight",
            "gauge",
            "Total unsourced weight (absolute value), summed across all contexts.",
            gauges.unsourced_weight_total,
        );
        if !gauges.per_context.is_empty() {
            push_header(
                &mut out,
                "taguru_context_disk_bytes",
                "gauge",
                "On-disk bytes per context and file family: image, graph WAL, \
                 passages snapshot, passages WAL, and the sidecars (meta, \
                 sources, gloss vectors, passage vectors, BM25) summed. The \
                 WAL lanes are live bookkeeping; the rest refresh at each \
                 flush tick and POST /flush, so they lag up to one flush \
                 interval — a scrape never stats the data directory. Present \
                 only with TAGURU_METRICS_PER_CONTEXT.",
            );
            for row in &gauges.per_context {
                let context = escape_label(&row.name);
                for (file, value) in [
                    ("image", row.disk_image_bytes),
                    ("passages", row.disk_passages_bytes),
                    ("passages_wal", row.disk_passages_wal_bytes),
                    ("sidecars", row.disk_sidecar_bytes),
                    ("wal", row.disk_wal_bytes),
                ] {
                    out.push_str(&format!(
                        "taguru_context_disk_bytes{{context=\"{context}\",file=\"{file}\"}} {value}\n"
                    ));
                }
            }
            push_header(
                &mut out,
                "taguru_context_resident_bytes",
                "gauge",
                "Modeled resident bytes per context (graph plus cached \
                 gloss/passage/BM25/vector stores; 0 while cold) — the same \
                 per-entry accounting taguru_resident_bytes sums fleet-wide.",
            );
            for row in &gauges.per_context {
                out.push_str(&format!(
                    "taguru_context_resident_bytes{{context=\"{}\"}} {}\n",
                    escape_label(&row.name),
                    row.resident_bytes
                ));
            }
            push_header(
                &mut out,
                "taguru_context_pinned",
                "gauge",
                "1 when the context is pinned resident (exempt from cache \
                 eviction).",
            );
            for row in &gauges.per_context {
                out.push_str(&format!(
                    "taguru_context_pinned{{context=\"{}\"}} {}\n",
                    escape_label(&row.name),
                    u64::from(row.pinned)
                ));
            }
            type CountPick = fn(&ContextGaugeRow) -> u64;
            let count_families: [(&str, &str, CountPick); 4] = [
                (
                    "taguru_context_concepts",
                    "Concepts stored per context (live for hot contexts, \
                     last-saved snapshot for cold).",
                    |row| row.concepts,
                ),
                (
                    "taguru_context_associations",
                    "Associations stored per context (live for hot contexts, \
                     last-saved snapshot for cold).",
                    |row| row.associations,
                ),
                (
                    "taguru_context_labels",
                    "Distinct relation labels per context (live for hot \
                     contexts, last-saved snapshot for cold).",
                    |row| row.labels,
                ),
                (
                    "taguru_context_sources",
                    "Sources contributing to the graph per context (live for \
                     hot contexts, last-saved snapshot for cold).",
                    |row| row.sources,
                ),
            ];
            for (name, help, pick) in count_families {
                push_header(&mut out, name, "gauge", help);
                for row in &gauges.per_context {
                    out.push_str(&format!(
                        "{name}{{context=\"{}\"}} {}\n",
                        escape_label(&row.name),
                        pick(row)
                    ));
                }
            }
            // Declared ceilings only (issue #136) — an uncapped fleet
            // renders no series and no header. Usage lives in the
            // sibling families above, so quota-vs-usage is one
            // division away on the dashboard.
            if gauges
                .per_context
                .iter()
                .any(|row| row.quota_storage_bytes.is_some() || row.quota_cache_bytes.is_some())
            {
                push_header(
                    &mut out,
                    "taguru_context_quota_bytes",
                    "gauge",
                    "Declared per-context ceilings (TAGURU_CONTEXT_QUOTAS): \
                     resource=\"storage\" caps the on-disk family (the sum of \
                     taguru_context_disk_bytes), resource=\"cache\" caps the \
                     resident share (taguru_context_resident_bytes) under \
                     eviction pressure.",
                );
                for row in &gauges.per_context {
                    let context = escape_label(&row.name);
                    for (resource, ceiling) in [
                        ("cache", row.quota_cache_bytes),
                        ("storage", row.quota_storage_bytes),
                    ] {
                        if let Some(ceiling) = ceiling {
                            out.push_str(&format!(
                                "taguru_context_quota_bytes{{context=\"{context}\",resource=\"{resource}\"}} {ceiling}\n"
                            ));
                        }
                    }
                }
            }
        }
        push_value(
            &mut out,
            "taguru_last_flush_success_timestamp_seconds",
            "gauge",
            "Unix time of the last successful image flush (0 = none since boot); alert on time() minus this.",
            self.last_flush_success_epoch.load(Ordering::Relaxed),
        );
        push_outcomes(
            &mut out,
            "taguru_auto_compactions_total",
            "Ratio-triggered auto-compactions run by the flusher tick (TAGURU_AUTO_COMPACT), by outcome; a failed candidate is retried while its dead ratio holds.",
            &self.auto_compact_ok,
            &self.auto_compact_failed,
        );
        push_value(
            &mut out,
            "taguru_auto_compact_reclaimed_bytes_total",
            "counter",
            "Resident bytes shed by successful auto-compactions (bytes before minus after, summed).",
            self.auto_compact_reclaimed_bytes.load(Ordering::Relaxed),
        );
        push_value(
            &mut out,
            "taguru_auto_compact_last_success_timestamp_seconds",
            "gauge",
            "Unix time of the last successful auto-compaction (0 = none since boot).",
            self.auto_compact_last_success_epoch.load(Ordering::Relaxed),
        );
        push_value(
            &mut out,
            "taguru_storage_quota_refusals_total",
            "counter",
            "Growth writes refused at a declared per-context storage ceiling (TAGURU_CONTEXT_QUOTAS) — graph writes, passage stores, and the import loop's per-batch pre-check all count here.",
            self.storage_quota_refusals.load(Ordering::Relaxed),
        );
        push_value(
            &mut out,
            "taguru_keyring_reloads_total",
            "counter",
            "Keyring hot reloads that armed a table (SIGHUP or config-file change; no-change reloads included).",
            self.keyring_reloads.load(Ordering::Relaxed),
        );
        push_value(
            &mut out,
            "taguru_keyring_reload_refusals_total",
            "counter",
            "Keyring reloads refused fail-closed (unreadable or malformed sources, or a reload that would disable authentication); the previous table stayed armed.",
            self.keyring_reload_refusals.load(Ordering::Relaxed),
        );
        push_value(
            &mut out,
            "taguru_replication_uploads_total",
            "counter",
            "Objects uploaded to the replication bucket (files, WAL segments, markers).",
            self.replication_uploads.load(Ordering::Relaxed),
        );
        push_value(
            &mut out,
            "taguru_replication_errors_total",
            "counter",
            "Failed replication operations (uploads, deletes, fence claims); the shipper retries.",
            self.replication_errors.load(Ordering::Relaxed),
        );
        push_value(
            &mut out,
            "taguru_replication_fenced",
            "gauge",
            "1 once a newer writer claimed the bucket and shipping fail-stopped (latched; restart to contest).",
            u64::from(self.replication_fenced.load(Ordering::Relaxed)),
        );
        push_value(
            &mut out,
            "taguru_replication_last_success_timestamp_seconds",
            "gauge",
            "Unix time of the last cycle that shipped everything it found (0 = none since boot); time() minus this bounds the DR restore's loss window.",
            self.replication_last_success_epoch.load(Ordering::Relaxed),
        );
        {
            let lag = self.replication_lag.lock().unwrap();
            push_header(
                &mut out,
                "taguru_replication_lag_records",
                "gauge",
                "Acknowledged log records not yet in the bucket, per context and lane.",
            );
            for ((context, lane), entry) in lag.iter() {
                out.push_str(&format!(
                    "taguru_replication_lag_records{{context=\"{}\",lane=\"{lane}\"}} {}\n",
                    escape_label(context),
                    entry.behind_records
                ));
            }
            push_header(
                &mut out,
                "taguru_replication_lag_seconds",
                "gauge",
                "Age of the oldest unshipped record, per context and lane (0 = caught up).",
            );
            for ((context, lane), entry) in lag.iter() {
                out.push_str(&format!(
                    "taguru_replication_lag_seconds{{context=\"{}\",lane=\"{lane}\"}} {}\n",
                    escape_label(context),
                    entry.age_secs
                ));
            }
        }
        if self.replica_mode.load(Ordering::Relaxed) {
            push_value(
                &mut out,
                "taguru_replica",
                "gauge",
                "1 when this process serves as a read replica (writes refused, the bucket tailed).",
                1,
            );
            push_value(
                &mut out,
                "taguru_replica_generation",
                "gauge",
                "The replication generation this replica follows (0 = none complete in the bucket yet).",
                self.replica_generation.load(Ordering::Relaxed),
            );
            push_value(
                &mut out,
                "taguru_replica_manifest_timestamp_seconds",
                "gauge",
                "Store-clock mtime of the newest complete manifest at the last poll; time() minus this, plus the poll interval, bounds staleness.",
                self.replica_manifest_epoch.load(Ordering::Relaxed),
            );
            push_value(
                &mut out,
                "taguru_replica_last_poll_success_timestamp_seconds",
                "gauge",
                "Unix time of the last successful tailer poll (0 = none since boot); a growing gap means the bucket is unreachable and staleness is unbounded.",
                self.replica_last_poll_epoch.load(Ordering::Relaxed),
            );
            push_value(
                &mut out,
                "taguru_replica_poll_errors_total",
                "counter",
                "Failed tailer polls (bucket errors, un-appliable families); the tailer retries.",
                self.replica_poll_errors.load(Ordering::Relaxed),
            );
            let lag = self.replica_lag.lock().unwrap();
            push_header(
                &mut out,
                "taguru_replica_applied_seq",
                "gauge",
                "Highest record seq this replica has applied, per context and lane.",
            );
            for ((context, lane), entry) in lag.iter() {
                out.push_str(&format!(
                    "taguru_replica_applied_seq{{context=\"{}\",lane=\"{lane}\"}} {}\n",
                    escape_label(context),
                    entry.applied_seq
                ));
            }
            push_header(
                &mut out,
                "taguru_replica_shipped_seq",
                "gauge",
                "Newest record seq the bucket ships, per context and lane; minus applied_seq = the promotion-time RPO in records.",
            );
            for ((context, lane), entry) in lag.iter() {
                out.push_str(&format!(
                    "taguru_replica_shipped_seq{{context=\"{}\",lane=\"{lane}\"}} {}\n",
                    escape_label(context),
                    entry.shipped_seq
                ));
            }
            push_header(
                &mut out,
                "taguru_replica_behind_seconds",
                "gauge",
                "How long the lane has been behind the shipped stream (0 = caught up).",
            );
            let now = Self::unix_now();
            for ((context, lane), entry) in lag.iter() {
                let behind = match entry.behind_since_epoch {
                    0 => 0,
                    since => now.saturating_sub(since),
                };
                out.push_str(&format!(
                    "taguru_replica_behind_seconds{{context=\"{}\",lane=\"{lane}\"}} {behind}\n",
                    escape_label(context),
                ));
            }
        }
        push_value(
            &mut out,
            "taguru_inflight_requests",
            "gauge",
            "Requests currently being served (probe endpoints exempt).",
            self.inflight.load(Ordering::Relaxed),
        );
        push_value(
            &mut out,
            "taguru_requests_shed_total",
            "counter",
            "Requests refused with a 503 at the in-flight ceiling (TAGURU_MAX_CONCURRENT_REQUESTS).",
            self.requests_shed.load(Ordering::Relaxed),
        );
        push_header(
            &mut out,
            "taguru_build_info",
            "gauge",
            "Build metadata as labels; the value is always 1.",
        );
        out.push_str(&format!(
            "taguru_build_info{{version=\"{}\"}} 1\n",
            env!("CARGO_PKG_VERSION")
        ));

        out
    }
}

fn push_header(out: &mut String, name: &str, kind: &str, help: &str) {
    out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {kind}\n"));
}

/// Prometheus label-value escaping. Context names are user text — a
/// name carrying a quote or a backslash must not be able to break the
/// exposition line it rides in (label values may hold anything else,
/// UTF-8 included).
fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// One gauge or counter family with a single, label-free value.
fn push_value(out: &mut String, name: &str, kind: &str, help: &str, value: impl std::fmt::Display) {
    push_header(out, name, kind, help);
    out.push_str(&format!("{name} {value}\n"));
}

/// A latency histogram family — bucket counts, the +Inf bucket, sum,
/// and count, the shape both `_duration_seconds` histograms share.
/// `labels` is the label body (no braces) every line in the family
/// carries ahead of `le`, or "" for the label-free embedding histogram.
fn push_histogram(out: &mut String, name: &str, labels: &str, snapshot: &HistogramSnapshot) {
    let le_prefix = if labels.is_empty() {
        String::new()
    } else {
        format!("{labels},")
    };
    let block = if labels.is_empty() {
        String::new()
    } else {
        format!("{{{labels}}}")
    };
    for ((_, le), value) in LATENCY_BUCKETS.iter().zip(snapshot.cumulative) {
        out.push_str(&format!(
            "{name}_bucket{{{le_prefix}le=\"{le}\"}} {value}\n"
        ));
    }
    let count = snapshot.count;
    out.push_str(&format!(
        "{name}_bucket{{{le_prefix}le=\"+Inf\"}} {count}\n"
    ));
    out.push_str(&format!(
        "{name}_sum{block} {}\n",
        snapshot.sum_micros as f64 / 1_000_000.0
    ));
    out.push_str(&format!("{name}_count{block} {count}\n"));
}

/// One counter family with the ok/failed outcome label, zeros included.
fn push_outcomes(out: &mut String, name: &str, help: &str, ok: &AtomicU64, failed: &AtomicU64) {
    push_header(out, name, "counter", help);
    out.push_str(&format!(
        "{name}{{outcome=\"ok\"}} {}\n",
        ok.load(Ordering::Relaxed)
    ));
    out.push_str(&format!(
        "{name}{{outcome=\"failed\"}} {}\n",
        failed.load(Ordering::Relaxed)
    ));
}

/// Access-log + RED-metrics middleware, one pass per request.
/// `MatchedPath` comes as `Option` deliberately: the required form
/// rejects before the fallback runs and would hijack 404 handling.
/// Unmatched requests all land in one `<unmatched>` series so a path
/// scanner cannot mint unbounded label values — and the method is
/// folded to a fixed allowlist ([`normalized_method`]) for the same
/// reason: an extension-method token is just as attacker-chosen as a
/// path, and this middleware runs ahead of auth.
///
/// With span export configured this is also where the request span is
/// born — parented from the inbound trace context, named per HTTP
/// semconv, its trace id stamped into the access log so a log line
/// finds its trace and vice versa. Without it, the disabled branch
/// leaves the response path and the log shape exactly as before.
pub async fn track_http(
    State(state): State<AppState>,
    matched: Option<MatchedPath>,
    request: Request,
    next: Next,
) -> Response {
    let method = normalized_method(request.method());
    let route = matched
        .as_ref()
        .map(|matched| matched.as_str().to_string())
        .unwrap_or_else(|| "<unmatched>".to_string());
    // The object the request addressed, when the route names one. The
    // route TEMPLATE keeps metric cardinality bounded, but a log line
    // is no series: without the real name here, "which contexts did
    // this key delete" has no answer after the fact. "-" mirrors the
    // key convention below. The names are identifiers, not memory
    // content — the registry's own warnings already print them.
    // Extracted by hand: `RawPathParams` rejects param-less routes, so
    // it cannot ride the signature as an extractor the way MatchedPath
    // (which supports optional extraction) does.
    let (mut parts, body) = request.into_parts();
    let name = crate::api::path_param(&mut parts, "name")
        .await
        .unwrap_or_else(|| "-".to_string());
    // The name lands in the column matching its kind — on the group
    // routes `{name}` is a GROUP — so a log query over `context=`
    // never silently matches group names (the audit lines and the
    // /metrics gauges keep the same split).
    let (context, group) = if route.starts_with("/groups") {
        ("-".to_string(), name)
    } else {
        (name, "-".to_string())
    };
    let request = Request::from_parts(parts, body);
    let started = Instant::now();

    let (response, trace_id) = if crate::trace::enabled() {
        traced_request(method, &route, request, next).await
    } else {
        (next.run(request).await, None)
    };

    let elapsed = started.elapsed();
    let status = response.status().as_u16();
    state.metrics().record_http(method, &route, status, elapsed);
    // Which credential made the request — stamped on the response by
    // the auth layer. "-" = unauthenticated (exempt path, auth off, or
    // a rejection).
    let key = response
        .extensions()
        .get::<crate::auth::AuthKey>()
        .map_or("-", |key| key.0.as_ref())
        .to_string();
    match trace_id {
        Some(trace_id) => tracing::info!(
            method = %method,
            route = %route,
            context = %context,
            group = %group,
            status,
            key = %key,
            latency_ms = elapsed.as_secs_f64() * 1000.0,
            trace_id = %trace_id,
            "http",
        ),
        None => tracing::info!(
            method = %method,
            route = %route,
            context = %context,
            group = %group,
            status,
            key = %key,
            latency_ms = elapsed.as_secs_f64() * 1000.0,
            "http",
        ),
    }
    response
}

/// Runs the request inside an OTel server span. Span name and
/// attributes follow HTTP semconv (`{method} {route}`, method only
/// when unmatched); a 5xx marks the span as an error, a 4xx does not —
/// for a server, a client's mistake is a normal outcome.
/// Folds a request method to a fixed set of `&'static str` labels. RFC
/// 9110 leaves the method an open token, so an unauthenticated client
/// can send an unbounded stream of distinct extension methods; keyed
/// straight into the metrics map (or a span name) each would mint a new
/// series. Anything outside the standard set collapses to `<other>`,
/// mirroring how the route collapses to `<unmatched>`.
fn normalized_method(method: &Method) -> &'static str {
    match *method {
        Method::GET => "GET",
        Method::POST => "POST",
        Method::PUT => "PUT",
        Method::DELETE => "DELETE",
        Method::PATCH => "PATCH",
        Method::HEAD => "HEAD",
        Method::OPTIONS => "OPTIONS",
        Method::TRACE => "TRACE",
        Method::CONNECT => "CONNECT",
        _ => "<other>",
    }
}

async fn traced_request(
    method: &str,
    route: &str,
    request: Request,
    next: Next,
) -> (Response, Option<String>) {
    use tracing::Instrument as _;
    use tracing_opentelemetry::OpenTelemetrySpanExt as _;

    let span = tracing::info_span!(
        "request",
        otel.name = %if route == "<unmatched>" {
            method.to_string()
        } else {
            format!("{method} {route}")
        },
        otel.kind = "server",
        http.request.method = %method,
        http.route = %route,
        url.path = %request.uri().path(),
        http.response.status_code = tracing::field::Empty,
        otel.status_code = tracing::field::Empty,
    );
    // Only fails without an export layer, and we only run when one is
    // installed.
    let _ = span.set_parent(crate::trace::extract_parent(request.headers()));
    let trace_id = {
        use opentelemetry::trace::TraceContextExt as _;
        span.context().span().span_context().trace_id().to_string()
    };

    let response = next.run(request).instrument(span.clone()).await;

    // i64 keeps the attribute an OTLP int — a bare u16 records as text.
    span.record(
        "http.response.status_code",
        i64::from(response.status().as_u16()),
    );
    if response.status().is_server_error() {
        span.record("otel.status_code", "ERROR");
    }
    (response, Some(trace_id))
}

/// GET /live: pure liveness — 200 for as long as the process answers
/// at all, deliberately unconditional. A failing flush is a DISK
/// problem (that is `/health`'s signal): restarting the process fixes
/// no disk and costs a full pinned preload, so an orchestrator's
/// liveness probe belongs here, its readiness probe on `/health`.
/// Wiring both probes at `/health` turns every transient disk stall
/// into a restart loop.
pub async fn live() -> &'static str {
    "ok"
}

/// GET /health: 200 "ok" while the write path is healthy, 503 in the
/// ApiError shape when the most recent image flush failed — or the
/// flusher tick that would have flushed it panicked instead of running.
/// The check is the flusher's own outcome, so an orchestrator's probe
/// turns red within one flush interval of the disk going bad — and
/// green again one interval after it recovers. (An idle server with
/// nothing dirty reports its last known state.) The readiness signal:
/// stop routing traffic while the disk is bad, resume when it heals —
/// liveness lives at `/live`.
pub async fn health(State(state): State<AppState>) -> Response {
    if state.metrics().maintenance_active() {
        return crate::api::error(
            crate::api::ErrorCode::Maintenance,
            "a maintenance compaction sweep is running — this is an intentional \
             pause, not a fault"
                .to_string(),
            Instant::now(),
        );
    }
    if state.metrics().flush_is_healthy() {
        return "ok".into_response();
    }
    let reason = if state.metrics().flusher_panicked() {
        "the flusher task panicked on its last tick — this is a bug, not a disk \
         fault; check the server log"
            .to_string()
    } else {
        match state.metrics().last_flush_success_epoch() {
            0 => "the last image flush failed, and none has succeeded since boot — \
                  check disk space and the server log"
                .to_string(),
            epoch => format!(
                "the last image flush failed; the last success was at unix {epoch} — \
                 check disk space and the server log"
            ),
        }
    };
    crate::api::error(crate::api::ErrorCode::Unhealthy, reason, Instant::now())
}

/// GET /metrics: the whole registry in Prometheus text format.
pub async fn render(State(state): State<AppState>) -> impl IntoResponse {
    let body = state.metrics().render_prometheus(&state.gauge_snapshot());
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_gauges() -> GaugeSnapshot {
        GaugeSnapshot {
            contexts_registered: 0,
            groups_registered: 0,
            contexts_resident: 0,
            resident_bytes: 0,
            wal_bytes: 0,
            passages_wal_bytes: 0,
            dead_edges_total: 0,
            dead_attributions_total: 0,
            arena_slack_total: 0,
            unsourced_edges_total: 0,
            unsourced_weight_total: 0.0,
            embed_breaker: None,
            retrieval_cache_entries: 0,
            retrieval_cache_bytes: 0,
            semantic_cache_entries: 0,
            per_context: Vec::new(),
        }
    }

    /// The per-context families gate on the snapshot carrying rows —
    /// an opted-out scrape stays byte-free of them, like the breaker
    /// and replica families — and render every row's series with the
    /// context name escaped, since names are client-minted text.
    #[test]
    fn the_per_context_families_gate_on_the_snapshot_and_escape_names() {
        let metrics = Metrics::default();
        let without = metrics.render_prometheus(&empty_gauges());
        assert!(!without.contains("taguru_context_"), "{without}");

        let mut gauges = empty_gauges();
        gauges.per_context.push(ContextGaugeRow {
            name: "日本\"酒\\".to_string(),
            pinned: true,
            resident_bytes: 1024,
            disk_image_bytes: 2048,
            disk_wal_bytes: 64,
            disk_passages_bytes: 512,
            disk_passages_wal_bytes: 32,
            disk_sidecar_bytes: 256,
            quota_storage_bytes: Some(4096),
            quota_cache_bytes: Some(1536),
            concepts: 7,
            associations: 9,
            labels: 3,
            sources: 2,
        });
        let with = metrics.render_prometheus(&gauges);
        let escaped = "日本\\\"酒\\\\";
        for line in [
            format!("taguru_context_quota_bytes{{context=\"{escaped}\",resource=\"cache\"}} 1536"),
            format!(
                "taguru_context_quota_bytes{{context=\"{escaped}\",resource=\"storage\"}} 4096"
            ),
            format!("taguru_context_disk_bytes{{context=\"{escaped}\",file=\"image\"}} 2048"),
            format!("taguru_context_disk_bytes{{context=\"{escaped}\",file=\"passages\"}} 512"),
            format!("taguru_context_disk_bytes{{context=\"{escaped}\",file=\"passages_wal\"}} 32"),
            format!("taguru_context_disk_bytes{{context=\"{escaped}\",file=\"sidecars\"}} 256"),
            format!("taguru_context_disk_bytes{{context=\"{escaped}\",file=\"wal\"}} 64"),
            format!("taguru_context_resident_bytes{{context=\"{escaped}\"}} 1024"),
            format!("taguru_context_pinned{{context=\"{escaped}\"}} 1"),
            format!("taguru_context_concepts{{context=\"{escaped}\"}} 7"),
            format!("taguru_context_associations{{context=\"{escaped}\"}} 9"),
            format!("taguru_context_labels{{context=\"{escaped}\"}} 3"),
            format!("taguru_context_sources{{context=\"{escaped}\"}} 2"),
        ] {
            assert!(with.contains(&line), "missing {line} in: {with}");
        }
    }

    /// The breaker family gates on the snapshot carrying one — a
    /// lexical-only server's scrape stays free of it — and renders the
    /// snapshot's values verbatim when it does.
    #[test]
    fn the_breaker_family_gates_on_the_snapshot() {
        let metrics = Metrics::default();
        let without = metrics.render_prometheus(&empty_gauges());
        assert!(!without.contains("taguru_embedding_breaker"), "{without}");

        let mut gauges = empty_gauges();
        gauges.embed_breaker = Some(crate::embedding::BreakerSnapshot {
            state: 2,
            consecutive_failures: 3,
            opened_total: 1,
            short_circuits_total: 4,
        });
        let with = metrics.render_prometheus(&gauges);
        assert!(with.contains("taguru_embedding_breaker_state 2"), "{with}");
        assert!(
            with.contains("taguru_embedding_breaker_consecutive_failures 3"),
            "{with}"
        );
        assert!(
            with.contains("taguru_embedding_breaker_opened_total 1"),
            "{with}"
        );
        assert!(
            with.contains("taguru_embedding_breaker_short_circuits_total 4"),
            "{with}"
        );
    }

    /// The replica family renders only in replica mode, and the lag
    /// arithmetic transitions the way the tailer drives it: behind
    /// when shipped outruns applied (age counting from the first poll
    /// that saw the gap), caught up the moment they meet.
    #[test]
    fn the_replica_family_gates_on_role_and_tracks_the_gap() {
        let metrics = Metrics::default();
        metrics.note_replica_lane("sake", "graph", 2, 2);
        let writer_scrape = metrics.render_prometheus(&empty_gauges());
        assert!(
            !writer_scrape.contains("taguru_replica ")
                && !writer_scrape.contains("taguru_replica_applied_seq"),
            "a writer's scrape carries no replica series (the replicaTION \
             family is a different prefix): {writer_scrape}"
        );

        metrics.set_replica_mode();
        metrics.note_replica_generation(3);
        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(rendered.contains("taguru_replica 1"), "{rendered}");
        assert!(
            rendered.contains("taguru_replica_generation 3"),
            "{rendered}"
        );
        assert!(
            rendered.contains("taguru_replica_applied_seq{context=\"sake\",lane=\"graph\"} 2"),
            "{rendered}"
        );
        assert!(
            rendered.contains("taguru_replica_behind_seconds{context=\"sake\",lane=\"graph\"} 0"),
            "{rendered}"
        );

        // The shipped side outruns the applied one: the gap shows, and
        // its age starts counting.
        metrics.note_replica_shipped("sake", "graph", 5);
        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(
            rendered.contains("taguru_replica_shipped_seq{context=\"sake\",lane=\"graph\"} 5"),
            "{rendered}"
        );
        assert!(
            rendered.contains("taguru_replica_applied_seq{context=\"sake\",lane=\"graph\"} 2"),
            "{rendered}"
        );

        // Catching up zeroes the age.
        metrics.note_replica_lane("sake", "graph", 5, 5);
        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(
            rendered.contains("taguru_replica_behind_seconds{context=\"sake\",lane=\"graph\"} 0"),
            "{rendered}"
        );

        // A vanished context's rows leave the scrape.
        metrics.note_replica_lane("sake", "passages", 1, 1);
        metrics.forget_replica_context("sake");
        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(
            !rendered.contains("context=\"sake\""),
            "ghost labels must not linger: {rendered}"
        );

        // A generation switch clears the whole family: applied seqs
        // are per-lineage, and a successor that started from an older
        // watermark ships LOWER seqs — a surviving predecessor value
        // would fake a caught-up lane.
        metrics.note_replica_lane("sake", "graph", 9, 9);
        metrics.reset_replica_lanes();
        metrics.note_replica_shipped("sake", "graph", 4);
        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(
            rendered.contains("taguru_replica_applied_seq{context=\"sake\",lane=\"graph\"} 0"),
            "the successor's gap must show from zero, not the predecessor's applied: {rendered}"
        );
    }

    /// The in-flight counter: a ceiling refuses at capacity, zero means
    /// count-only, release always returns the slot — and both series
    /// render on /metrics.
    #[test]
    fn the_inflight_counter_admits_releases_and_renders() {
        let metrics = Metrics::default();
        assert!(metrics.admit_inflight(2));
        assert!(metrics.admit_inflight(2));
        assert!(!metrics.admit_inflight(2), "the ceiling holds");
        metrics.record_shed();
        assert_eq!(metrics.inflight_count(), 2);
        metrics.release_inflight();
        assert_eq!(metrics.inflight_count(), 1);
        assert!(metrics.admit_inflight(2), "a release frees a slot");
        // 0 = no ceiling; the gauge still counts.
        assert!(metrics.admit_inflight(0));

        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(
            rendered.contains("taguru_inflight_requests 3"),
            "{rendered}"
        );
        assert!(
            rendered.contains("taguru_requests_shed_total 1"),
            "{rendered}"
        );
    }

    #[test]
    fn embed_latency_renders_as_a_histogram() {
        let metrics = Metrics::default();
        metrics.record_embed_latency(Duration::from_millis(3));
        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(
            rendered.contains("taguru_embedding_duration_seconds_count 1"),
            "{rendered}"
        );
        assert!(
            rendered.contains("taguru_embedding_duration_seconds_bucket{le=\"0.005\"} 1"),
            "{rendered}"
        );
    }

    #[test]
    fn histogram_bucket_boundaries_are_cumulative() {
        let histogram = Histogram::default();
        histogram.observe(Duration::from_millis(0)); // le 1
        histogram.observe(Duration::from_millis(3)); // le 5
        histogram.observe(Duration::from_millis(3)); // le 5
        histogram.observe(Duration::from_millis(400)); // le 500

        let cumulative = histogram.snapshot().cumulative;
        assert_eq!(cumulative, [1, 3, 3, 3, 3, 4, 4, 4]);
        let mut previous = 0;
        for value in cumulative {
            assert!(value >= previous, "buckets must never decrease");
            previous = value;
        }
    }

    /// A [`Histogram`] backed by three independent atomics (the shape
    /// this used to have) can render `+Inf` as less than a finite
    /// bucket: a reader catching `observe()` between its bucket
    /// increment and its count increment sees the new observation in
    /// `cumulative` but not yet in `count`. The single mutex behind
    /// `snapshot()` rules that out — every reader sees one consistent
    /// instant — so this holds under concurrent writers and readers,
    /// not just the single-threaded shape the other tests exercise.
    #[test]
    fn concurrent_observe_and_snapshot_never_report_inf_below_a_finite_bucket() {
        let histogram = Arc::new(Histogram::default());
        let writers: Vec<_> = (0..4)
            .map(|_| {
                let histogram = Arc::clone(&histogram);
                std::thread::spawn(move || {
                    for i in 0..2_000u64 {
                        histogram.observe(Duration::from_micros(i % 6_000));
                    }
                })
            })
            .collect();

        let reader = {
            let histogram = Arc::clone(&histogram);
            std::thread::spawn(move || {
                for _ in 0..2_000 {
                    let snapshot = histogram.snapshot();
                    let max_finite = *snapshot.cumulative.last().unwrap();
                    assert!(
                        snapshot.count >= max_finite,
                        "+Inf ({}) must never be less than a finite bucket ({})",
                        snapshot.count,
                        max_finite
                    );
                }
            })
        };

        for writer in writers {
            writer.join().unwrap();
        }
        reader.join().unwrap();
    }

    #[test]
    fn fractional_milliseconds_bucket_by_true_latency_not_truncation() {
        let histogram = Histogram::default();
        // 1.9 ms is OVER the 1 ms bound: `as_millis` truncation would
        // file it under le="0.001" and skew every low quantile fast.
        histogram.observe(Duration::from_micros(1_900)); // le 5
        histogram.observe(Duration::from_micros(1_000)); // le 1, exactly
        histogram.observe(Duration::from_micros(5_100)); // le 10

        assert_eq!(histogram.snapshot().cumulative, [1, 2, 3, 3, 3, 3, 3, 3]);
    }

    #[test]
    fn histogram_plus_inf_equals_count_even_past_the_largest_bound() {
        let metrics = Metrics::default();
        metrics.record_http("GET", "/x", 200, Duration::from_millis(2));
        // Past the 5000ms top bound: lands in no finite bucket, but
        // still counts toward +Inf and _count.
        metrics.record_http("GET", "/x", 200, Duration::from_secs(60));

        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(rendered.contains(
            "taguru_http_request_duration_seconds_bucket{method=\"GET\",route=\"/x\",le=\"5\"} 1"
        ));
        assert!(rendered.contains(
            "taguru_http_request_duration_seconds_bucket{method=\"GET\",route=\"/x\",le=\"+Inf\"} 2"
        ));
        assert!(
            rendered.contains(
                "taguru_http_request_duration_seconds_count{method=\"GET\",route=\"/x\"} 2"
            )
        );
    }

    #[test]
    fn histogram_sum_renders_seconds_not_millis() {
        let metrics = Metrics::default();
        metrics.record_http("GET", "/x", 200, Duration::from_millis(250));
        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(
            rendered.contains(
                "taguru_http_request_duration_seconds_sum{method=\"GET\",route=\"/x\"} 0.25\n"
            ),
            "{rendered}"
        );
    }

    #[test]
    fn nonstandard_methods_fold_to_a_single_label() {
        // Standard methods keep their identity...
        assert_eq!(normalized_method(&Method::GET), "GET");
        assert_eq!(normalized_method(&Method::DELETE), "DELETE");
        // ...but an extension-method token — which a client can mint
        // without bound, ahead of auth — collapses to one series rather
        // than growing the metrics map per distinct value.
        let weird = Method::from_bytes(b"M0001").unwrap();
        assert_eq!(normalized_method(&weird), "<other>");
        let also = Method::from_bytes(b"FROBNICATE").unwrap();
        assert_eq!(normalized_method(&also), "<other>");
    }

    #[test]
    fn render_is_deterministic_with_sorted_dynamic_keys() {
        let metrics = Metrics::default();
        // Insertion order deliberately unsorted.
        metrics.record_http("POST", "/b", 200, Duration::from_millis(1));
        metrics.record_http("GET", "/a", 404, Duration::from_millis(1));
        metrics.record_http("GET", "/a", 200, Duration::from_millis(1));

        let first = metrics.render_prometheus(&empty_gauges());
        let second = metrics.render_prometheus(&empty_gauges());
        assert_eq!(first, second, "identical state must render identically");

        let get_a = first.find("method=\"GET\",route=\"/a\",status=").unwrap();
        let post_b = first.find("method=\"POST\",route=\"/b\",status=").unwrap();
        assert!(get_a < post_b, "routes must render in sorted order");
        let status_200 = first.find("route=\"/a\",status=\"200\"").unwrap();
        let status_404 = first.find("route=\"/a\",status=\"404\"").unwrap();
        assert!(status_200 < status_404, "statuses must render sorted");
    }

    #[test]
    fn error_kinds_render_individually_with_zeros_for_the_untouched_ones() {
        let metrics = Metrics::default();
        metrics.record_error(ErrorKind::Load);
        metrics.record_error(ErrorKind::Load);
        metrics.record_error(ErrorKind::WalRefused);
        metrics.record_error(ErrorKind::Panic);

        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(rendered.contains("taguru_errors_total{kind=\"load\"} 2"));
        assert!(rendered.contains("taguru_errors_total{kind=\"wal_refused\"} 1"));
        assert!(rendered.contains("taguru_errors_total{kind=\"panic\"} 1"));
        // The untouched kind still renders, so dashboards never see an
        // absent series.
        assert!(rendered.contains("taguru_errors_total{kind=\"io\"} 0"));
    }

    #[test]
    fn search_outcomes_render_per_op_including_untouched_zeros() {
        let metrics = Metrics::default();
        metrics.record_search(SearchOp::Recall, false);
        metrics.record_search(SearchOp::Recall, false);
        metrics.record_search(SearchOp::Recall, true);
        metrics.record_search(SearchOp::SearchPassages, true);

        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(rendered.contains("taguru_searches_total{op=\"recall\",outcome=\"hit\"} 2"));
        assert!(rendered.contains("taguru_searches_total{op=\"recall\",outcome=\"empty\"} 1"));
        assert!(
            rendered.contains("taguru_searches_total{op=\"search_passages\",outcome=\"empty\"} 1")
        );
        // Untouched ops still render both outcomes.
        assert!(rendered.contains("taguru_searches_total{op=\"explore\",outcome=\"hit\"} 0"));
        assert!(rendered.contains("taguru_searches_total{op=\"resolve\",outcome=\"empty\"} 0"));
    }

    #[test]
    fn resolve_tiers_render_all_four_buckets() {
        let metrics = Metrics::default();
        metrics.record_resolve_tier(ResolveTier::Lexical);
        metrics.record_resolve_tier(ResolveTier::Semantic);
        metrics.record_resolve_tier(ResolveTier::Semantic);
        metrics.record_resolve_tier(ResolveTier::Miss);

        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(rendered.contains("taguru_resolves_total{tier=\"lexical\"} 1"));
        assert!(rendered.contains("taguru_resolves_total{tier=\"semantic\"} 2"));
        assert!(rendered.contains("taguru_resolves_total{tier=\"weak_lexical\"} 0"));
        assert!(rendered.contains("taguru_resolves_total{tier=\"miss\"} 1"));
    }

    #[test]
    fn passage_lane_contributions_expose_all_three_labels_from_zero() {
        let metrics = Metrics::default();
        metrics.record_passage_hit(true, false);
        metrics.record_passage_hit(true, true);
        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(rendered.contains("taguru_passage_lane_contributions_total{lane=\"bm25_only\"} 1"));
        assert!(
            rendered.contains("taguru_passage_lane_contributions_total{lane=\"both_lanes\"} 1")
        );
        assert!(
            rendered.contains("taguru_passage_lane_contributions_total{lane=\"vector_only\"} 0"),
            "the label a dashboard alerts on must exist at zero"
        );
    }

    /// One context's successful flush must NOT mask another's still-failing
    /// one: health tracks the SET of failing contexts, not the last
    /// outcome. The old single global bit reported healthy after B here.
    #[test]
    fn flush_health_tracks_each_context_not_just_the_last() {
        let metrics = Metrics::default();
        assert!(metrics.flush_is_healthy(), "an idle server is healthy");

        metrics.record_flush("a", false);
        assert!(!metrics.flush_is_healthy(), "A's failure degrades health");

        metrics.record_flush("b", true);
        assert!(
            !metrics.flush_is_healthy(),
            "B's success must not mask A's unhealed failure"
        );

        // A repeated failure for the same context is one entry, not a
        // tally: the set heals fully on A's first success.
        metrics.record_flush("a", false);
        metrics.record_flush("a", true);
        assert!(
            metrics.flush_is_healthy(),
            "health returns once every failing context has flushed clean"
        );
    }

    /// Auto-compaction's three series: outcomes split, reclaimed bytes
    /// accumulate on success only, and the last-success clock moves on
    /// success only — a failing retry loop must not look recent.
    #[test]
    fn auto_compaction_counters_split_outcomes_and_stamp_success_only() {
        let metrics = Metrics::default();
        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(
            rendered.contains("taguru_auto_compactions_total{outcome=\"ok\"} 0"),
            "the series a dashboard alerts on must exist at zero"
        );
        assert!(rendered.contains("taguru_auto_compact_last_success_timestamp_seconds 0"));

        metrics.record_auto_compaction(None);
        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(rendered.contains("taguru_auto_compactions_total{outcome=\"failed\"} 1"));
        assert!(
            rendered.contains("taguru_auto_compact_last_success_timestamp_seconds 0"),
            "a failure must not stamp the success clock"
        );

        metrics.record_auto_compaction(Some(2048));
        metrics.record_auto_compaction(Some(1000));
        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(rendered.contains("taguru_auto_compactions_total{outcome=\"ok\"} 2"));
        assert!(rendered.contains("taguru_auto_compact_reclaimed_bytes_total 3048"));
        assert!(!rendered.contains("taguru_auto_compact_last_success_timestamp_seconds 0"));
    }

    /// A panicking flusher tick must degrade health just like a failed
    /// flush — `spawn_flusher` catches the panic to keep ticking, but
    /// without this a flusher stuck panicking on every tick would look
    /// identical to a healthy idle server forever, since no flush ever
    /// runs to report a failure.
    #[test]
    fn flusher_panic_degrades_health_and_a_clean_tick_heals_it() {
        let metrics = Metrics::default();
        assert!(metrics.flush_is_healthy(), "an idle server is healthy");
        assert!(!metrics.flusher_panicked());

        metrics.record_flusher_tick(false);
        assert!(
            !metrics.flush_is_healthy(),
            "a panicked tick degrades health"
        );
        assert!(metrics.flusher_panicked());

        metrics.record_flusher_tick(true);
        assert!(
            metrics.flush_is_healthy(),
            "the next clean tick heals it, same as a flush retry"
        );
        assert!(!metrics.flusher_panicked());
    }

    /// One sweep at a time, and `exit_maintenance` is safe to call more
    /// than once — the guard's `Drop` must not panic if it somehow ran
    /// twice.
    #[test]
    fn maintenance_is_a_one_shot_cas_until_exit_reopens_it() {
        let metrics = Metrics::default();
        assert!(!metrics.maintenance_active());
        assert!(metrics.try_enter_maintenance(), "first claim succeeds");
        assert!(metrics.maintenance_active());
        assert!(
            !metrics.try_enter_maintenance(),
            "a second claim is refused while one is running"
        );

        metrics.exit_maintenance();
        assert!(!metrics.maintenance_active());
        metrics.exit_maintenance(); // idempotent
        assert!(!metrics.maintenance_active());

        assert!(metrics.try_enter_maintenance(), "reopened after exit");
    }

    /// The readiness probe treats a maintenance sweep as a deliberate
    /// pause, not a fault: its own 503 code, and back to "ok" the
    /// instant the guard drops.
    #[tokio::test]
    async fn health_reports_maintenance_distinctly_from_a_flush_fault() {
        let dir = std::env::temp_dir().join(format!(
            "taguru-metrics-health-maintenance-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let state = crate::registry::AppState::boot(dir.clone(), usize::MAX, None).unwrap();

        assert_eq!(health(State(state.clone())).await.status().as_u16(), 200);

        let guard = state.try_enter_maintenance().expect("first claim succeeds");
        let response = health(State(state.clone())).await;
        assert_eq!(response.status().as_u16(), 503);
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["code"], "maintenance");

        drop(guard);
        assert_eq!(health(State(state.clone())).await.status().as_u16(), 200);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The flusher-panic reason must read differently from an ordinary
    /// flush failure: one is a disk problem worth checking disk space
    /// over, the other is a bug in the server itself. An operator
    /// paged with the disk-space message would go check the wrong
    /// thing.
    #[tokio::test]
    async fn health_reports_the_flusher_panic_reason_distinctly_from_a_flush_fault() {
        let dir = std::env::temp_dir().join(format!(
            "taguru-metrics-health-flusher-panic-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let state = crate::registry::AppState::boot(dir.clone(), usize::MAX, None).unwrap();

        assert_eq!(health(State(state.clone())).await.status().as_u16(), 200);

        state.metrics().record_flusher_tick(false);
        let response = health(State(state.clone())).await;
        assert_eq!(response.status().as_u16(), 503);
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["code"], "unhealthy");
        assert!(
            body["error"].as_str().unwrap().contains("panicked"),
            "{body}"
        );

        // The next clean tick heals it — self-healing, same as a flush
        // retry, not a latch an operator has to clear by hand.
        state.metrics().record_flusher_tick(true);
        assert_eq!(health(State(state.clone())).await.status().as_u16(), 200);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn render_carries_help_and_type_for_every_metric_name() {
        let metrics = Metrics::default();
        metrics.record_http("GET", "/a", 200, Duration::from_millis(1));
        metrics.record_flush("a", true);
        let rendered = metrics.render_prometheus(&GaugeSnapshot {
            contexts_registered: 2,
            groups_registered: 1,
            contexts_resident: 1,
            resident_bytes: 640,
            wal_bytes: 0,
            passages_wal_bytes: 0,
            dead_edges_total: 0,
            dead_attributions_total: 0,
            arena_slack_total: 0,
            unsourced_edges_total: 0,
            unsourced_weight_total: 0.0,
            embed_breaker: Some(crate::embedding::BreakerSnapshot {
                state: 1,
                consecutive_failures: 3,
                opened_total: 2,
                short_circuits_total: 7,
            }),
            retrieval_cache_entries: 3,
            retrieval_cache_bytes: 4096,
            semantic_cache_entries: 5,
            // One row so the per-context families render — their
            // HELP/TYPE discipline is checked here like everyone
            // else's.
            per_context: vec![ContextGaugeRow {
                name: "sake".to_string(),
                pinned: false,
                resident_bytes: 640,
                disk_image_bytes: 100,
                disk_wal_bytes: 10,
                disk_passages_bytes: 20,
                disk_passages_wal_bytes: 5,
                disk_sidecar_bytes: 30,
                // Declared so the quota family renders — its HELP/TYPE
                // discipline rides this same check.
                quota_storage_bytes: Some(1000),
                quota_cache_bytes: Some(2000),
                concepts: 4,
                associations: 6,
                labels: 2,
                sources: 1,
            }],
        });

        // Every sample line's metric name must have been introduced by
        // a HELP/TYPE pair (bucket/sum/count roll up to their family).
        let mut declared: Vec<&str> = Vec::new();
        for line in rendered.lines() {
            if let Some(rest) = line.strip_prefix("# TYPE ") {
                declared.push(rest.split(' ').next().unwrap());
            } else if !line.starts_with('#') && !line.is_empty() {
                let name = line.split(['{', ' ']).next().unwrap();
                let family = name
                    .strip_suffix("_bucket")
                    .or_else(|| name.strip_suffix("_sum"))
                    .or_else(|| name.strip_suffix("_count"))
                    .filter(|family| declared.contains(family))
                    .unwrap_or(name);
                assert!(declared.contains(&family), "undeclared metric {name}");
            }
        }
        // And the zero-valued fixed families are present at all.
        assert!(rendered.contains(
            "taguru_embedding_requests_total{operation=\"resolve\",outcome=\"failed\"} 0"
        ));
        assert!(rendered.contains("taguru_keyring_reloads_total 0"));
        assert!(rendered.contains("taguru_keyring_reload_refusals_total 0"));
        assert!(rendered.contains("taguru_contexts_resident 1"));
        assert!(rendered.contains(&format!(
            "taguru_build_info{{version=\"{}\"}} 1",
            env!("CARGO_PKG_VERSION")
        )));
    }
}
