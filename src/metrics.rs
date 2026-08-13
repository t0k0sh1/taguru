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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{MatchedPath, Request, State};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
// parking_lot, not std::sync: a panic while one of these is held must
// not poison the lock — `track_http` wraps every request, so a
// poisoned metrics lock would turn one contained panic into a
// process-wide failure on every later scrape and request.
use parking_lot::{Mutex, RwLock};

use crate::registry::AppState;

#[path = "metrics/histogram.rs"]
mod histogram;
#[path = "metrics/http.rs"]
mod http;
#[path = "metrics/prometheus.rs"]
mod prometheus;
#[path = "metrics/record.rs"]
mod record;
#[path = "metrics/taxonomy.rs"]
mod taxonomy;

use histogram::{Histogram, HistogramSnapshot, LATENCY_BUCKETS, RouteStat};
pub use http::{health, live, render, track_http, version};
pub use taxonomy::{
    ContextGaugeRow, ErrorKind, GaugeSnapshot, PerContextMetrics, ResolveTier, RetrievalCacheOp,
    SearchOp, SemanticCacheOutcome,
};
use taxonomy::{ReplicaLag, ReplicationLag};
pub(crate) use taxonomy::{RerankOutcomeKind, SchemaOutcome};

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
    /// A `flush_entry` attempt that backed off — a rival flush already
    /// mid-flight, a racing delete, or a slot/generation that moved
    /// out from under it (issue #562 item 9) — every one a legitimate
    /// no-op, never a `record_flush(false)`-worthy failure, but
    /// invisible to `/health` and the periodic flusher's own tick
    /// otherwise: nothing else here says "this entry's flush did not
    /// land this round."
    flush_skipped: AtomicU64,
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
    /// `[outcome]` per [`SchemaOutcome`] — counted only at the write
    /// entrances a schema actually gates, never at a dry-run or the
    /// audit/validate diagnostics (see [`SchemaOutcome`]'s own doc).
    /// Stays entirely at zero for a server with no context ever
    /// carrying an installed schema document.
    schema_checks: [AtomicU64; SchemaOutcome::ALL.len()],
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
    /// Reranker round-trip latency (retries included), and how each
    /// `rerank::drive` call ended — the #307 twin of `embed_latency`/
    /// `searches` above. Both stay entirely at zero on a server with no
    /// reranker configured, or where no caller ever names `rerank`.
    rerank_latency: Histogram,
    rerank_outcomes: [AtomicU64; RerankOutcomeKind::ALL.len()],
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
    /// A per-context disk-usage `fs::metadata` call (issue #562 item
    /// 4) failing for a reason other than the file simply not existing
    /// yet — a permission error, EIO, or a rename racing the stat.
    /// `refresh_disk_usage` keeps the entry's last-known snapshot
    /// rather than silently substituting zero for the failed lane
    /// (which would understate a storage quota's `used` and let growth
    /// through it should have refused), so this counter is the only
    /// signal that a context's disk gauges and quota accounting are
    /// running on stale data.
    disk_stat_failures: AtomicU64,
    /// `embed_provider_slots` (the process-wide cap on concurrent
    /// embedding-provider round trips, issue #563 item 4) acquires
    /// that had to queue behind a full semaphore — a rising rate says
    /// the provider is the bottleneck, not disk or lock contention.
    /// `_timeouts` is its alertable half: an acquire that queued past
    /// its request deadline and gave up, which surfaces as a refresh
    /// failure the operator otherwise has no way to distinguish from a
    /// provider error.
    embed_slot_waits: AtomicU64,
    embed_slot_timeouts: AtomicU64,
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
    /// Subset of `replication_errors` the store classified as
    /// likely-permanent (bad credentials, unsupported operation,
    /// unrecognized config key — see `ship::error::store_error`): the
    /// shipper still retries, but this will not self-heal on its own
    /// the way the rest of `replication_errors` usually does, so it
    /// gets its own alertable series instead of hiding inside the
    /// generic count.
    replication_permanent_errors: AtomicU64,
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
            rerank_breaker: None,
            retrieval_cache_entries: 0,
            retrieval_cache_bytes: 0,
            semantic_cache_entries: 0,
            embed_slot_waiters: 0,
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
            schema_violations: 5,
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
            format!("taguru_context_schema_violations_total{{context=\"{escaped}\"}} 5"),
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

    /// The #307 reranker breaker family gates on its own snapshot,
    /// independently of the embedding breaker's — a server with a
    /// reranker configured but no embedder (or vice versa) renders
    /// exactly one of the two families, never both or neither.
    #[test]
    fn the_rerank_breaker_family_gates_on_the_snapshot() {
        let metrics = Metrics::default();
        let without = metrics.render_prometheus(&empty_gauges());
        assert!(!without.contains("taguru_rerank_breaker"), "{without}");

        let mut gauges = empty_gauges();
        gauges.rerank_breaker = Some(crate::breaker::BreakerSnapshot {
            state: 2,
            consecutive_failures: 3,
            opened_total: 1,
            short_circuits_total: 4,
        });
        let with = metrics.render_prometheus(&gauges);
        assert!(with.contains("taguru_rerank_breaker_state 2"), "{with}");
        assert!(
            with.contains("taguru_rerank_breaker_consecutive_failures 3"),
            "{with}"
        );
        assert!(
            with.contains("taguru_rerank_breaker_opened_total 1"),
            "{with}"
        );
        assert!(
            with.contains("taguru_rerank_breaker_short_circuits_total 4"),
            "{with}"
        );
    }

    /// `taguru_rerank_outcomes_total`/`_duration_seconds` render at zero
    /// on a server that never ran a rerank — the same "the family
    /// exists but every label is zero" discipline `searches`/
    /// `semantic_cache` use, distinct from the breaker gauges above
    /// (which are entirely ABSENT without a configured provider).
    #[test]
    fn rerank_outcomes_render_from_zero_and_bucket_by_token() {
        let metrics = Metrics::default();
        metrics.record_rerank("ok", Duration::from_millis(5));
        metrics.record_rerank("provider_error", Duration::from_millis(1));
        metrics.record_rerank("provider_error", Duration::from_millis(1));
        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(rendered.contains("taguru_rerank_outcomes_total{outcome=\"ok\"} 1"));
        assert!(rendered.contains("taguru_rerank_outcomes_total{outcome=\"provider_error\"} 2"));
        assert!(rendered.contains("taguru_rerank_outcomes_total{outcome=\"timeout\"} 0"));
        assert!(rendered.contains("taguru_rerank_duration_seconds_count 3"));
    }

    /// The reranker's own #307 vocabulary must not silently drift from
    /// this metrics-side enum — `from_token` falls back to
    /// `ProviderError` for anything unrecognized, but every one of
    /// `rerank.rs`'s actual constants must round-trip through it as
    /// itself.
    #[test]
    fn every_rerank_reason_token_maps_to_its_own_outcome_kind() {
        // The real REASON_* constants, not string literals re-typed
        // here — `RerankOutcomeKind::from_token`'s own doc comment
        // promises this test catches drift from `rerank.rs`'s
        // vocabulary; a hardcoded literal would still pass after
        // `rerank.rs` renamed a token out from under it, since nothing
        // would fail to compile or diverge. "ok" has no named constant
        // in `rerank.rs` (it is not a failure reason), so it stays a
        // literal.
        use crate::api::evidence::rerank::{
            REASON_CIRCUIT_OPEN, REASON_EMPTY_POOL, REASON_INVALID_PERMUTATION,
            REASON_MODEL_MISMATCH, REASON_NOT_CONFIGURED, REASON_PROVIDER_ERROR, REASON_TIMEOUT,
        };
        let pairs = [
            ("ok", RerankOutcomeKind::Ok),
            (REASON_NOT_CONFIGURED, RerankOutcomeKind::NotConfigured),
            (REASON_MODEL_MISMATCH, RerankOutcomeKind::ModelMismatch),
            (REASON_EMPTY_POOL, RerankOutcomeKind::EmptyPool),
            (
                REASON_INVALID_PERMUTATION,
                RerankOutcomeKind::InvalidPermutation,
            ),
            (REASON_CIRCUIT_OPEN, RerankOutcomeKind::CircuitOpen),
            (REASON_TIMEOUT, RerankOutcomeKind::Timeout),
            (REASON_PROVIDER_ERROR, RerankOutcomeKind::ProviderError),
        ];
        for (token, expected) in pairs {
            assert_eq!(RerankOutcomeKind::from_token(token), expected, "{token}");
        }
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

    /// A shipped-only report that no longer outruns the applied seq
    /// clears the age — the same three-way move `note_replica_lane`
    /// makes, so a lineage whose shipped seqs regress can never leave
    /// a stale behind-since stamp on a lane that is in fact caught up.
    #[test]
    fn a_shipped_report_at_or_below_applied_clears_the_behind_age() {
        let metrics = Metrics::default();
        let key = ("sake".to_string(), "graph");
        metrics.note_replica_lane("sake", "graph", 5, 5);
        metrics.note_replica_shipped("sake", "graph", 7);
        let stamped = metrics.replica_lag.lock()[&key].behind_since_epoch;
        assert_ne!(stamped, 0, "a real gap starts the age");
        metrics.note_replica_shipped("sake", "graph", 5);
        let cleared = metrics.replica_lag.lock()[&key].behind_since_epoch;
        assert_eq!(cleared, 0, "no gap, no age");
        // Strictly below, not just equal — the caught-up arm is `>=`,
        // and a regressed shipped seq (a successor lineage's lower
        // watermark) must clear the age the same way.
        metrics.note_replica_shipped("sake", "graph", 7);
        metrics.note_replica_shipped("sake", "graph", 4);
        let cleared_below = metrics.replica_lag.lock()[&key].behind_since_epoch;
        assert_eq!(cleared_below, 0, "a lower shipped seq also clears the age");
    }

    /// #616 item 1: a likely-permanent shipping error (bad
    /// credentials, an unsupported operation, an unrecognized config
    /// key) gets its own alertable series on top of the generic
    /// replication-errors count.
    #[test]
    fn a_permanent_replication_error_renders_its_own_counter() {
        let metrics = Metrics::default();
        let before = metrics.render_prometheus(&empty_gauges());
        assert!(
            before.contains("taguru_replication_permanent_errors_total 0"),
            "{before}"
        );
        metrics.record_replication_permanent_error();
        metrics.record_replication_permanent_error();
        let after = metrics.render_prometheus(&empty_gauges());
        assert!(
            after.contains("taguru_replication_permanent_errors_total 2"),
            "{after}"
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

    // `normalized_method` itself now lives in `src/trace.rs` (moved
    // alongside `traced_request`, ADR 0008 §5) — its own test moved
    // with it.

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
    fn schema_checks_render_from_zero_for_every_outcome() {
        let metrics = Metrics::default();
        metrics.record_schema_check(SchemaOutcome::Ok);
        metrics.record_schema_check(SchemaOutcome::Ok);
        metrics.record_schema_check(SchemaOutcome::Refused);

        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(rendered.contains("taguru_schema_checks_total{outcome=\"ok\"} 2"));
        // Untouched outcome still renders at zero — a server that has
        // never seen a `warn`-mode write still exposes the label.
        assert!(rendered.contains("taguru_schema_checks_total{outcome=\"warned\"} 0"));
        assert!(rendered.contains("taguru_schema_checks_total{outcome=\"refused\"} 1"));
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

    /// The healthy body names this build's own version (ADR 0002 §10)
    /// so a remote CLI can read it off the one request it already
    /// sends, instead of the bare `"ok"` text an older server answers.
    #[tokio::test]
    async fn health_names_its_own_version_when_healthy() {
        let dir = std::env::temp_dir().join(format!(
            "taguru-metrics-health-version-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let state = crate::registry::AppState::boot(dir.clone(), usize::MAX, None).unwrap();

        let response = health(State(state.clone())).await;
        assert_eq!(response.status().as_u16(), 200);
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["status"], "ok");
        assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `GET /version` (ADR 0005 §6): bare JSON, every dimension named,
    /// `image_formats` the full `1..=IMAGE_VERSION` range (unlike
    /// `batch_formats`/`communities_formats`, which are single-element
    /// since they're checked for equality, not range acceptance).
    #[tokio::test]
    async fn version_names_every_contract_dimension() {
        let response = version().await;
        assert_eq!(response.status().as_u16(), 200);
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["server"], env!("CARGO_PKG_VERSION"));
        assert_eq!(body["http_contract"]["current"], 1);
        assert_eq!(body["http_contract"]["supported"], serde_json::json!([1]));
        assert_eq!(body["mcp_contract"]["current"], 1);
        assert!(
            body["mcp_protocol"]["supported"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("2025-06-18"))
        );
        assert_eq!(body["batch_formats"], serde_json::json!([1]));
        assert_eq!(
            body["image_formats"],
            serde_json::json!((1..=u64::from(taguru::context::IMAGE_VERSION)).collect::<Vec<_>>())
        );
        assert_eq!(body["communities_formats"], serde_json::json!([1]));
        assert_eq!(body["schema_formats"], serde_json::json!([1]));
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

        // Unlike `/health`, `/version` (ADR 0005 §6) answers 200 even
        // while the write path is degraded — it has to, to serve as
        // the base a compatibility check runs from.
        assert_eq!(version().await.status().as_u16(), 200);

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
        metrics.record_rerank("ok", Duration::from_millis(2));
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
            rerank_breaker: Some(crate::breaker::BreakerSnapshot {
                state: 0,
                consecutive_failures: 0,
                opened_total: 0,
                short_circuits_total: 0,
            }),
            retrieval_cache_entries: 3,
            retrieval_cache_bytes: 4096,
            semantic_cache_entries: 5,
            embed_slot_waiters: 2,
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
                schema_violations: 0,
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
        assert!(rendered.contains("taguru_rerank_outcomes_total{outcome=\"ok\"} 1"));
        assert!(rendered.contains("taguru_rerank_outcomes_total{outcome=\"not_configured\"} 0"));
        assert!(rendered.contains("taguru_rerank_breaker_state 0"));
        assert!(rendered.contains(&format!(
            "taguru_build_info{{version=\"{}\"}} 1",
            env!("CARGO_PKG_VERSION")
        )));
    }

    /// Pins `SearchOp::as_str()` against the string literals
    /// `src/mcp/retrieve.rs` and `src/api/evidence/assemble.rs` copy
    /// for `taguru.op` (dual-included into the stdio bridge, which
    /// has no `metrics` module — see the doc comment on `as_str`
    /// above). A renamed variant here with no matching literal update
    /// there would otherwise drift silently past compilation.
    #[test]
    fn taguru_op_literals_match_search_op() {
        assert_eq!(SearchOp::Resolve.as_str(), "resolve");
        assert_eq!(SearchOp::Query.as_str(), "query");
        assert_eq!(SearchOp::Activate.as_str(), "activate");
        assert_eq!(SearchOp::SearchPassages.as_str(), "search_passages");
    }
}
