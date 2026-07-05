//! Hand-rolled observability: RED metrics per route plus domain
//! counters (cache, flush, embedding), rendered as Prometheus text on
//! demand. Hand-rolled on purpose — the fixed catalog below needs a
//! few atomics and one render function, not a metrics facade crate;
//! the same reasoning that keeps BM25 and the vector store in-tree.
//!
//! Hot-path cost is one atomic increment per event. Histograms store
//! per-bin counts and defer the cumulative `le` semantics to render
//! (scrape) time.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{MatchedPath, Request, State};
use axum::http::StatusCode;
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
/// prefix sums are computed at render time so recording stays a single
/// increment. `count` doubles as the `+Inf` bucket and the `_count`
/// line (the exposition format defines them to be equal), so it also
/// counts observations past the largest finite bound.
#[derive(Default)]
struct Histogram {
    counts: [AtomicU64; LATENCY_BUCKETS.len()],
    sum_micros: AtomicU64,
    count: AtomicU64,
}

impl Histogram {
    fn observe(&self, elapsed: Duration) {
        let millis = elapsed.as_millis() as u64;
        if let Some(bin) = LATENCY_BUCKETS
            .iter()
            .position(|&(bound, _)| millis <= bound)
        {
            self.counts[bin].fetch_add(1, Ordering::Relaxed);
        }
        self.sum_micros
            .fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Cumulative `_bucket{le=…}` values, ascending, one per finite bound.
    fn cumulative(&self) -> [u64; LATENCY_BUCKETS.len()] {
        let mut running = 0u64;
        let mut out = [0u64; LATENCY_BUCKETS.len()];
        for (slot, count) in out.iter_mut().zip(&self.counts) {
            running += count.load(Ordering::Relaxed);
            *slot = running;
        }
        out
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
    errors_load: AtomicU64,
    errors_wal_refused: AtomicU64,
    errors_io: AtomicU64,
    /// `[op][outcome]`, outcome 0 = hit, 1 = empty.
    searches: [[AtomicU64; 2]; SearchOp::ALL.len()],
    resolve_tiers: [AtomicU64; ResolveTier::ALL.len()],
    /// Set while the most recent flush attempt failed; cleared by the
    /// next success. Drives /health: the flusher retries every tick,
    /// so this is a self-healing signal, never a latched one.
    flush_degraded: AtomicBool,
    /// Unix seconds of the last successful image flush (0 = none since
    /// boot). `time() - this` on a dashboard says how stale images are
    /// without knowing the flush interval.
    last_flush_success_epoch: AtomicU64,
}

/// Why a request answered 500. The status code alone cannot separate
/// these — and they demand different responses from an operator: `load`
/// is a corrupt or unreadable image (restore from backup), `wal_refused`
/// is the durability path failing writes (check the disk NOW), `io` is
/// a sidecar or image file operation failing outside the WAL path.
#[derive(Clone, Copy)]
pub enum ErrorKind {
    Load,
    WalRefused,
    Io,
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

/// Point-in-time gauges, computed from the registry at scrape time
/// rather than maintained incrementally — they cannot drift.
pub struct GaugeSnapshot {
    pub contexts_registered: u64,
    pub contexts_resident: u64,
    pub resident_bytes: u64,
    /// Total bytes across every context's write-ahead log. A healthy
    /// server truncates each log every flush interval; sustained
    /// growth here means images are failing to save.
    pub wal_bytes: u64,
}

impl Metrics {
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

    pub fn record_flush(&self, ok: bool) {
        let counter = if ok {
            &self.flush_ok
        } else {
            &self.flush_failed
        };
        counter.fetch_add(1, Ordering::Relaxed);
        self.flush_degraded.store(!ok, Ordering::Relaxed);
        if ok {
            let epoch = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|since| since.as_secs())
                .unwrap_or(0);
            self.last_flush_success_epoch
                .store(epoch, Ordering::Relaxed);
        }
    }

    /// Whether the most recent image-flush attempt succeeded (true
    /// when none has run yet — an idle server is a healthy server).
    pub fn flush_is_healthy(&self) -> bool {
        !self.flush_degraded.load(Ordering::Relaxed)
    }

    /// Unix seconds of the last successful flush; 0 when none yet.
    pub fn last_flush_success_epoch(&self) -> u64 {
        self.last_flush_success_epoch.load(Ordering::Relaxed)
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

    pub fn record_error(&self, kind: ErrorKind) {
        let counter = match kind {
            ErrorKind::Load => &self.errors_load,
            ErrorKind::WalRefused => &self.errors_wal_refused,
            ErrorKind::Io => &self.errors_io,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// One successful retrieval, split by whether it matched anything.
    /// Error responses never land here — a 500 is not an empty search.
    pub fn record_search(&self, op: SearchOp, empty: bool) {
        self.searches[op as usize][usize::from(empty)].fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_resolve_tier(&self, tier: ResolveTier) {
        self.resolve_tiers[tier as usize].fetch_add(1, Ordering::Relaxed);
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
            let cumulative = stat.latency.cumulative();
            for ((_, le), value) in LATENCY_BUCKETS.iter().zip(cumulative) {
                out.push_str(&format!(
                    "taguru_http_request_duration_seconds_bucket{{method=\"{method}\",route=\"{route}\",le=\"{le}\"}} {value}\n"
                ));
            }
            let count = stat.latency.count.load(Ordering::Relaxed);
            let sum = stat.latency.sum_micros.load(Ordering::Relaxed) as f64 / 1e6;
            out.push_str(&format!(
                "taguru_http_request_duration_seconds_bucket{{method=\"{method}\",route=\"{route}\",le=\"+Inf\"}} {count}\n"
            ));
            out.push_str(&format!(
                "taguru_http_request_duration_seconds_sum{{method=\"{method}\",route=\"{route}\"}} {sum}\n"
            ));
            out.push_str(&format!(
                "taguru_http_request_duration_seconds_count{{method=\"{method}\",route=\"{route}\"}} {count}\n"
            ));
        }
        drop(http);

        // Fixed-cardinality families always emit every label
        // combination, zeros included, so dashboards never see an
        // absent series.
        push_header(
            &mut out,
            "taguru_cache_hits_total",
            "counter",
            "Context cache hits (already resident, no disk load).",
        );
        out.push_str(&format!(
            "taguru_cache_hits_total {}\n",
            self.cache_hits.load(Ordering::Relaxed)
        ));
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
            "taguru_errors_total",
            "counter",
            "Requests answered 500, by cause: load = image/WAL unreadable on load, \
             wal_refused = the WAL could not durably record a write (nothing applied), \
             io = a file operation failed outside the WAL path.",
        );
        for (kind, counter) in [
            ("io", &self.errors_io),
            ("load", &self.errors_load),
            ("wal_refused", &self.errors_wal_refused),
        ] {
            out.push_str(&format!(
                "taguru_errors_total{{kind=\"{kind}\"}} {}\n",
                counter.load(Ordering::Relaxed)
            ));
        }

        push_header(
            &mut out,
            "taguru_contexts_registered",
            "gauge",
            "Contexts known to the registry.",
        );
        out.push_str(&format!(
            "taguru_contexts_registered {}\n",
            gauges.contexts_registered
        ));
        push_header(
            &mut out,
            "taguru_contexts_resident",
            "gauge",
            "Contexts currently resident in memory.",
        );
        out.push_str(&format!(
            "taguru_contexts_resident {}\n",
            gauges.contexts_resident
        ));
        push_header(
            &mut out,
            "taguru_resident_bytes",
            "gauge",
            "Modeled resident estimate of loaded contexts and cached vector stores (graph and vector footprints, NOT process RSS).",
        );
        out.push_str(&format!(
            "taguru_resident_bytes {}\n",
            gauges.resident_bytes
        ));
        push_header(
            &mut out,
            "taguru_wal_bytes",
            "gauge",
            "Total bytes across all write-ahead logs; sustained growth means image flushes are failing.",
        );
        out.push_str(&format!("taguru_wal_bytes {}\n", gauges.wal_bytes));
        push_header(
            &mut out,
            "taguru_last_flush_success_timestamp_seconds",
            "gauge",
            "Unix time of the last successful image flush (0 = none since boot); alert on time() minus this.",
        );
        out.push_str(&format!(
            "taguru_last_flush_success_timestamp_seconds {}\n",
            self.last_flush_success_epoch.load(Ordering::Relaxed)
        ));
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
/// scanner cannot mint unbounded label values.
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
    let method = request.method().as_str().to_string();
    let route = matched
        .as_ref()
        .map(|matched| matched.as_str().to_string())
        .unwrap_or_else(|| "<unmatched>".to_string());
    let started = Instant::now();

    let (response, trace_id) = if crate::trace::enabled() {
        traced_request(&method, &route, request, next).await
    } else {
        (next.run(request).await, None)
    };

    let elapsed = started.elapsed();
    let status = response.status().as_u16();
    state
        .metrics()
        .record_http(&method, &route, status, elapsed);
    match trace_id {
        Some(trace_id) => tracing::info!(
            method = %method,
            route = %route,
            status,
            latency_ms = elapsed.as_secs_f64() * 1000.0,
            trace_id = %trace_id,
            "http",
        ),
        None => tracing::info!(
            method = %method,
            route = %route,
            status,
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

/// GET /health: 200 "ok" while the write path is healthy, 503 in the
/// ApiError shape when the most recent image flush failed. The check
/// is the flusher's own outcome, so an orchestrator's probe turns red
/// within one flush interval of the disk going bad — and green again
/// one interval after it recovers. (An idle server with nothing dirty
/// reports its last known state.)
pub async fn health(State(state): State<AppState>) -> Response {
    if state.metrics().flush_is_healthy() {
        return "ok".into_response();
    }
    let reason = match state.metrics().last_flush_success_epoch() {
        0 => "the last image flush failed, and none has succeeded since boot — \
              check disk space and the server log"
            .to_string(),
        epoch => format!(
            "the last image flush failed; the last success was at unix {epoch} — \
             check disk space and the server log"
        ),
    };
    crate::api::error(StatusCode::SERVICE_UNAVAILABLE, reason, Instant::now())
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
            contexts_resident: 0,
            resident_bytes: 0,
            wal_bytes: 0,
        }
    }

    #[test]
    fn histogram_bucket_boundaries_are_cumulative() {
        let histogram = Histogram::default();
        histogram.observe(Duration::from_millis(0)); // le 1
        histogram.observe(Duration::from_millis(3)); // le 5
        histogram.observe(Duration::from_millis(3)); // le 5
        histogram.observe(Duration::from_millis(400)); // le 500

        let cumulative = histogram.cumulative();
        assert_eq!(cumulative, [1, 3, 3, 3, 3, 4, 4, 4]);
        let mut previous = 0;
        for value in cumulative {
            assert!(value >= previous, "buckets must never decrease");
            previous = value;
        }
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

        let rendered = metrics.render_prometheus(&empty_gauges());
        assert!(rendered.contains("taguru_errors_total{kind=\"load\"} 2"));
        assert!(rendered.contains("taguru_errors_total{kind=\"wal_refused\"} 1"));
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
    fn render_carries_help_and_type_for_every_metric_name() {
        let metrics = Metrics::default();
        metrics.record_http("GET", "/a", 200, Duration::from_millis(1));
        metrics.record_flush(true);
        let rendered = metrics.render_prometheus(&GaugeSnapshot {
            contexts_registered: 2,
            contexts_resident: 1,
            resident_bytes: 640,
            wal_bytes: 0,
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
        assert!(rendered.contains("taguru_contexts_resident 1"));
        assert!(rendered.contains(&format!(
            "taguru_build_info{{version=\"{}\"}} 1",
            env!("CARGO_PKG_VERSION")
        )));
    }
}
