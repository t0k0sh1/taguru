//! Per-request wall-clock ceiling and the per-key request budget.
//! Body size and per-parameter cost caps live where they apply (the
//! body-limit layer in main, the clamps in api); this module owns the
//! cross-cutting limits no handler can enforce for itself: total time,
//! and total request rate per credential.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderValue, header};
use axum::middleware::Next;
use axum::response::Response;
use taguru::deadline::Deadline;
use tokio::sync::Semaphore;

use crate::api;
use crate::auth;

/// The caller's source IP as a bucket key, or `None` when the request
/// carries no peer address. IP only — the ephemeral source port changes
/// per connection, so keying on it would give every connection its own
/// bucket and defeat the limit. Production always has a peer (the server
/// is built with `into_make_service_with_connect_info`); some tests do
/// not, and fall back to a shared bucket at the call sites.
///
/// Behind a reverse proxy every request arrives from the proxy's own
/// address, so all callers would share one bucket here — run the
/// throttles at the proxy, or put taguru on the connection directly,
/// when per-caller fairness matters. taguru deliberately does NOT read
/// `X-Forwarded-For`: it is client-settable, and honoring it unverified
/// would let anyone forge a fresh key per request and slip every
/// per-IP throttle entirely.
pub(crate) fn peer_ip(request: &Request) -> Option<Arc<str>> {
    request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|info| Arc::from(info.0.ip().to_string().as_str()))
}

/// Races the rest of the stack against the configured budget; a loss
/// is a 408 in the ApiError shape — retryable and client-actionable
/// (narrow the query), unlike a 503's "server unhealthy".
///
/// Caveat, documented for operators: the embedding-backed endpoints
/// run their provider round trip inside `block_in_place`, which a
/// future race cannot preempt — those requests only see the deadline
/// after the blocking call returns. With TAGURU_EMBED_URL configured,
/// set the budget above the provider's own 60s ceiling.
///
/// Stamps a [`Deadline`] onto the request before handing it to `next`,
/// so a `block_in_place` section downstream — invisible to this race
/// once entered — can still check the same budget on its own and give
/// up early instead of running to completion regardless.
pub async fn enforce_timeout(
    State(budget): State<Duration>,
    mut request: Request,
    next: Next,
) -> Response {
    let started_at = Instant::now();
    request.extensions_mut().insert(Deadline::after(budget));
    match tokio::time::timeout(budget, next.run(request)).await {
        Ok(response) => response,
        Err(_) => api::error(
            api::ErrorCode::Timeout,
            format!(
                "request exceeded the {}s budget; narrow the query \
                 (TAGURU_REQUEST_TIMEOUT_SECS tunes this)",
                budget.as_secs()
            ),
            started_at,
        ),
    }
}

/// The in-flight ceiling (`TAGURU_MAX_CONCURRENT_REQUESTS`): past it a
/// request is SHED — an immediate 503 with `Retry-After` — instead of
/// joining a queue that hides the overload until everything times out.
/// This is the last-resort valve, sitting outside auth on purpose: at
/// saturation the refusal must be the cheapest response the server can
/// make. `/health` and `/metrics` stay exempt, exactly as they are for
/// auth and the rate gate — probes and scrapes must see the overload,
/// not join it. The counter doubles as the `taguru_inflight_requests`
/// gauge, ticking whether or not a ceiling is set.
/// Shapes an immediate shed response: the `ApiError` body plus the
/// `Retry-After: 1` header every shed reason below shares.
fn shed(code: api::ErrorCode, message: impl Into<String>) -> Response {
    let mut response = api::error(code, message, Instant::now());
    response
        .headers_mut()
        .insert(header::RETRY_AFTER, HeaderValue::from(1u32));
    response
}

/// A shared, non-queuing permit pool for the whole-context CPU/disk
/// sweeps exposed by the API. Unlike the global in-flight ceiling this
/// gate is applied only to `audit_vocabulary`, `compact_context`, and
/// `audit_drift` (which runs the same pairwise scan as
/// `audit_vocabulary` when `include_twins` is set); ordinary requests
/// retain the rest of the worker pool during a burst.
#[derive(Clone)]
pub struct HeavyOpsLimiter {
    limit: usize,
    permits: Arc<Semaphore>,
}

impl HeavyOpsLimiter {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            permits: Arc::new(Semaphore::new(limit)),
        }
    }
}

/// Immediately admits a heavy operation or sheds it. Waiting here would
/// merely hide saturation until the request deadline expired, so this uses
/// `try_acquire_owned` and returns the same retryable 503 shape as the
/// global in-flight ceiling. A limit of zero disables this targeted gate.
pub async fn enforce_heavy_ops(
    State(limiter): State<HeavyOpsLimiter>,
    request: Request,
    next: Next,
) -> Response {
    if limiter.limit == 0 {
        return next.run(request).await;
    }
    let Ok(_permit) = Arc::clone(&limiter.permits).try_acquire_owned() else {
        return shed(
            api::ErrorCode::Overloaded,
            format!(
                "server is at its heavy-operation ceiling ({} concurrent calls) — retry \
                 shortly (TAGURU_MAX_CONCURRENT_HEAVY_OPS tunes this)",
                limiter.limit
            ),
        );
    };
    next.run(request).await
}

pub async fn enforce_concurrency(
    State((limit, state)): State<(usize, crate::registry::AppState)>,
    request: Request,
    next: Next,
) -> Response {
    if auth::PROBE_EXEMPT.contains(&request.uri().path()) {
        return next.run(request).await;
    }
    // Best-effort: sheds new work a little sooner than making it wait
    // out a `try_enter_maintenance` refusal inside the handler, which
    // remains the one real guarantee against two sweeps overlapping.
    // Not `record_shed()` — an intentional pause is not the same signal
    // as saturation.
    if state.metrics().maintenance_active() {
        return shed(
            api::ErrorCode::Maintenance,
            "a maintenance compaction sweep is running — retry shortly",
        );
    }
    if !state.metrics().admit_inflight(limit) {
        state.metrics().record_shed();
        return shed(
            api::ErrorCode::Overloaded,
            format!(
                "server is at its in-flight ceiling ({limit} requests) — retry \
                 shortly (TAGURU_MAX_CONCURRENT_REQUESTS tunes this)"
            ),
        );
    }
    // Drop-guard, not a manual decrement: the count must come back
    // down even when a handler panics mid-request.
    struct Release(crate::registry::AppState);
    impl Drop for Release {
        fn drop(&mut self) {
            self.0.metrics().release_inflight();
        }
    }
    let _release = Release(state.clone());
    next.run(request).await
}

/// Token buckets (`TAGURU_RATE_LIMIT_PER_MIN`): each key gets a minute's
/// allowance as capacity, refilled continuously, so a client may burst
/// its whole budget and then settle to the sustained rate. 0 disables
/// the gate. Also backs the failed-auth throttle
/// (`TAGURU_AUTH_FAIL_LIMIT_PER_MIN`) with the same mechanics.
///
/// Keys are either configured key names or caller source IPs (see
/// `peer_ip`) — the IP case makes the map caller-driven, not
/// configuration-bounded, so `admit` reclaims fully-refilled buckets
/// (an idle bucket is indistinguishable from a fresh one, so dropping
/// it changes no decision) to keep it from retaining an entry per IP
/// ever seen.
pub struct RateLimiter {
    per_minute: u32,
    state: Mutex<Buckets>,
}

struct Buckets {
    map: HashMap<Arc<str>, Bucket>,
    last_pruned: Instant,
}

struct Bucket {
    tokens: f64,
    refreshed: Instant,
}

/// Prune only once the map is big enough that a per-IP leak could
/// matter, and at most this often, so the O(n) sweep never runs per
/// request under a high-cardinality flood.
const PRUNE_THRESHOLD: usize = 1024;
const PRUNE_INTERVAL: Duration = Duration::from_secs(60);

impl RateLimiter {
    pub fn new(per_minute: u32) -> Self {
        Self {
            per_minute,
            state: Mutex::new(Buckets {
                map: HashMap::new(),
                last_pruned: Instant::now(),
            }),
        }
    }

    pub fn is_disabled(&self) -> bool {
        self.per_minute == 0
    }

    /// Admits or refuses one request for `key` at `now`; refusal names
    /// the seconds until a token exists again — the Retry-After value.
    /// `now` is a parameter so tests refill by arithmetic, not sleep.
    pub(crate) fn admit(&self, key: &Arc<str>, now: Instant) -> Result<(), u64> {
        let capacity = f64::from(self.per_minute);
        let per_second = capacity / 60.0;
        let mut state = self.state.lock().unwrap();
        // Reclaim idle buckets before this insert can grow the map
        // further: a bucket that would have refilled to capacity by now
        // holds no state a fresh one wouldn't, so dropping it is free.
        // Guarded by size and interval so the sweep stays amortized.
        if state.map.len() >= PRUNE_THRESHOLD
            && now.duration_since(state.last_pruned) >= PRUNE_INTERVAL
        {
            state.map.retain(|_, bucket| {
                let refilled =
                    bucket.tokens + now.duration_since(bucket.refreshed).as_secs_f64() * per_second;
                refilled < capacity
            });
            state.last_pruned = now;
        }
        let bucket = state.map.entry(Arc::clone(key)).or_insert(Bucket {
            tokens: capacity,
            refreshed: now,
        });
        let elapsed = now.duration_since(bucket.refreshed).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * per_second).min(capacity);
        // Two same-key requests can take the mutex in an order inverted
        // from their `Instant::now()` captures. The straggler's elapsed
        // saturates to zero above — correct — but writing its earlier
        // timestamp back would hand the NEXT request the same interval
        // again, so the clock only ever advances.
        bucket.refreshed = bucket.refreshed.max(now);
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            Ok(())
        } else {
            Err(((1.0 - bucket.tokens) / per_second).ceil() as u64)
        }
    }
}

/// Sits INSIDE auth: budget is spent by an authenticated key (a wrong
/// token costs nothing), and a 429 exits back through auth and the
/// access log, which then names the key. Auth's exempt paths stay
/// exempt here for the same reason they are exempt there.
pub async fn enforce_rate_limit(
    State(limiter): State<Arc<RateLimiter>>,
    request: Request,
    next: Next,
) -> Response {
    if limiter.is_disabled() || auth::PROBE_EXEMPT.contains(&request.uri().path()) {
        return next.run(request).await;
    }
    let started_at = Instant::now();
    // Auth (outside) stamped WHO onto the request. Without a key —
    // auth off (dev mode), or an auth-exempt OAuth endpoint — bucket by
    // source IP so one noisy peer cannot drain a single shared "anon"
    // allowance for everyone else. The "peer:" prefix keeps an IP from
    // colliding with a configured key that happens to share its text.
    // (No peer address, only in tests without ConnectInfo, falls back
    // to the shared bucket; production always carries one.)
    let key = match request.extensions().get::<auth::AuthKey>() {
        Some(key) => Arc::clone(&key.0),
        None => peer_ip(&request)
            .map(|ip| Arc::<str>::from(format!("peer:{ip}")))
            .unwrap_or_else(|| Arc::from("anon")),
    };
    match limiter.admit(&key, started_at) {
        Ok(()) => next.run(request).await,
        Err(retry_after) => {
            let mut response = api::error(
                api::ErrorCode::RateLimited,
                format!(
                    "key '{key}' is over its request budget ({} per minute) — \
                     retry in {retry_after}s",
                    limiter.per_minute
                ),
                started_at,
            );
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(retry_after));
            response
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use axum::routing::{get, post};
    use tower::util::ServiceExt;

    fn app(budget: Duration) -> Router {
        Router::new()
            .route("/fast", get(|| async { "done" }))
            .route(
                "/slow",
                get(|| async {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    "done"
                }),
            )
            .layer(axum::middleware::from_fn_with_state(
                budget,
                enforce_timeout,
            ))
    }

    async fn status_of(app: Router, path: &str) -> (u16, serde_json::Value) {
        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri(path)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status().as_u16();
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, body)
    }

    #[tokio::test]
    async fn a_fast_handler_completes_within_the_budget() {
        let (status, _) = status_of(app(Duration::from_secs(5)), "/fast").await;
        assert_eq!(status, 200);
    }

    #[tokio::test]
    async fn a_slow_handler_is_cut_off_with_a_408_in_the_api_error_shape() {
        let (status, body) = status_of(app(Duration::from_millis(20)), "/slow").await;
        assert_eq!(status, 408);
        assert_eq!(body["status"], "error");
        assert!(body["error"].as_str().unwrap().contains("budget"), "{body}");
    }

    /// At the ceiling a new request is shed — 503, Retry-After, the
    /// ApiError shape — while probes stay exempt, and the permit
    /// returns the moment the in-flight request finishes.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_inflight_ceiling_sheds_and_recovers() {
        let dir = std::env::temp_dir().join(format!("taguru-limits-shed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = crate::registry::AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let app = {
            let entered = Arc::clone(&entered);
            let release = Arc::clone(&release);
            Router::new()
                .route(
                    "/slow",
                    get(move || {
                        let entered = Arc::clone(&entered);
                        let release = Arc::clone(&release);
                        async move {
                            entered.notify_one();
                            release.notified().await;
                            "done"
                        }
                    }),
                )
                .route("/fast", get(|| async { "done" }))
                .route("/health", get(|| async { "ok" }))
                .layer(axum::middleware::from_fn_with_state(
                    (1usize, state.clone()),
                    enforce_concurrency,
                ))
        };
        let request = |path: &str| {
            HttpRequest::builder()
                .uri(path)
                .body(Body::empty())
                .unwrap()
        };

        let holder = tokio::spawn({
            let app = app.clone();
            let held = request("/slow");
            async move { app.oneshot(held).await.unwrap() }
        });
        entered.notified().await; // the permit is now held

        let shed = app.clone().oneshot(request("/fast")).await.unwrap();
        assert_eq!(shed.status(), 503);
        assert!(shed.headers().get(header::RETRY_AFTER).is_some());
        let bytes = axum::body::to_bytes(shed.into_body(), 4096).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["status"], "error");
        assert!(
            body["error"].as_str().unwrap().contains("ceiling"),
            "{body}"
        );

        // Probes must see the overload, not join it.
        let probe = app.clone().oneshot(request("/health")).await.unwrap();
        assert_eq!(probe.status(), 200);

        release.notify_one();
        assert_eq!(holder.await.unwrap().status(), 200);
        // The drop-guard returned the permit: capacity is back.
        let after = app.clone().oneshot(request("/fast")).await.unwrap();
        assert_eq!(after.status(), 200);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The targeted semaphore is shared by both expensive routes, refuses
    /// instead of queueing, and returns its permit when the holder ends.
    /// Routes outside the layered sub-router remain available throughout.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_heavy_operation_ceiling_sheds_only_heavy_routes_and_recovers() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let heavy = {
            let entered = Arc::clone(&entered);
            let release = Arc::clone(&release);
            Router::new()
                .route(
                    "/audit",
                    post(move || {
                        let entered = Arc::clone(&entered);
                        let release = Arc::clone(&release);
                        async move {
                            entered.notify_one();
                            release.notified().await;
                            "done"
                        }
                    }),
                )
                .route("/compact", post(|| async { "done" }))
                .route_layer(axum::middleware::from_fn_with_state(
                    HeavyOpsLimiter::new(1),
                    enforce_heavy_ops,
                ))
        };
        let app = Router::new()
            .route("/ordinary", get(|| async { "done" }))
            .merge(heavy);
        let request = |path: &str, method: &str| {
            HttpRequest::builder()
                .method(method)
                .uri(path)
                .body(Body::empty())
                .unwrap()
        };

        let holder = tokio::spawn({
            let app = app.clone();
            async move { app.oneshot(request("/audit", "POST")).await.unwrap() }
        });
        entered.notified().await;

        let shed = app
            .clone()
            .oneshot(request("/compact", "POST"))
            .await
            .unwrap();
        assert_eq!(shed.status(), 503);
        assert_eq!(shed.headers().get(header::RETRY_AFTER).unwrap(), "1");
        let bytes = axum::body::to_bytes(shed.into_body(), 4096).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["code"], "overloaded");
        assert!(
            body["error"].as_str().unwrap().contains("heavy-operation"),
            "{body}"
        );

        let ordinary = app
            .clone()
            .oneshot(request("/ordinary", "GET"))
            .await
            .unwrap();
        assert_eq!(ordinary.status(), 200);

        release.notify_one();
        assert_eq!(holder.await.unwrap().status(), 200);
        let after = app.oneshot(request("/compact", "POST")).await.unwrap();
        assert_eq!(after.status(), 200);
    }

    #[tokio::test]
    async fn a_zero_heavy_operation_limit_disables_the_gate() {
        let app = Router::new()
            .route("/heavy", post(|| async { "done" }))
            .layer(axum::middleware::from_fn_with_state(
                HeavyOpsLimiter::new(0),
                enforce_heavy_ops,
            ));
        let first = app
            .clone()
            .oneshot(HttpRequest::post("/heavy").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let second = app
            .oneshot(HttpRequest::post("/heavy").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(first.status(), 200);
        assert_eq!(second.status(), 200);
    }

    /// A maintenance sweep sheds new work with its own code (not
    /// `Overloaded`, not the shed counter) and reopens the moment the
    /// guard drops — independent of any in-flight ceiling.
    #[tokio::test]
    async fn maintenance_mode_sheds_new_work_without_touching_the_shed_counter() {
        let dir =
            std::env::temp_dir().join(format!("taguru-limits-maintenance-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = crate::registry::AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let app = Router::new()
            .route("/fast", get(|| async { "done" }))
            .route("/health", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn_with_state(
                (usize::MAX, state.clone()),
                enforce_concurrency,
            ));
        let request = |path: &str| {
            HttpRequest::builder()
                .uri(path)
                .body(Body::empty())
                .unwrap()
        };

        let guard = state.try_enter_maintenance().expect("first sweep admits");
        let shed = app.clone().oneshot(request("/fast")).await.unwrap();
        assert_eq!(shed.status(), 503);
        assert!(shed.headers().get(header::RETRY_AFTER).is_some());
        let bytes = axum::body::to_bytes(shed.into_body(), 4096).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["code"], "maintenance");

        // Probes still get through.
        let probe = app.clone().oneshot(request("/health")).await.unwrap();
        assert_eq!(probe.status(), 200);

        // Distinct from overload: the shed counter stays at 0.
        let rendered = state.metrics().render_prometheus(&state.gauge_snapshot());
        assert!(
            rendered.contains("taguru_requests_shed_total 0"),
            "{rendered}"
        );

        drop(guard);
        let after = app.clone().oneshot(request("/fast")).await.unwrap();
        assert_eq!(after.status(), 200);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The bucket allows a full-capacity burst, refills continuously,
    /// and names a sane wait when empty — all by clock arithmetic.
    #[test]
    fn the_bucket_bursts_refills_and_names_the_wait() {
        let limiter = RateLimiter::new(60); // one token per second
        let key: Arc<str> = Arc::from("k");
        let start = Instant::now();
        for _ in 0..60 {
            assert!(limiter.admit(&key, start).is_ok());
        }
        assert_eq!(limiter.admit(&key, start), Err(1));
        // Half a second later the refill is still short of one token...
        assert!(
            limiter
                .admit(&key, start + Duration::from_millis(500))
                .is_err()
        );
        // ...a further second buys exactly one.
        assert!(
            limiter
                .admit(&key, start + Duration::from_millis(1500))
                .is_ok()
        );
    }

    #[test]
    fn a_lock_race_cannot_rewind_the_refill_clock() {
        let limiter = RateLimiter::new(60); // one token per second
        let key: Arc<str> = Arc::from("k");
        let start = Instant::now();
        for _ in 0..60 {
            assert!(limiter.admit(&key, start).is_ok());
        }
        assert!(limiter.admit(&key, start).is_err(), "drained");

        // Two same-key requests race the mutex in an order inverted
        // from their Instant::now() captures: the later timestamp
        // lands first, the straggler's saturates to zero elapsed.
        assert!(limiter.admit(&key, start + Duration::from_secs(2)).is_ok());
        assert!(limiter.admit(&key, start + Duration::from_secs(1)).is_ok());
        // Those two seconds refilled two tokens and both are spent. If
        // the straggler had rewound the clock, this third admit at the
        // same instant would be handed the same second again.
        assert!(limiter.admit(&key, start + Duration::from_secs(2)).is_err());
    }

    #[test]
    fn keys_spend_separate_buckets() {
        let limiter = RateLimiter::new(1);
        let now = Instant::now();
        let hot: Arc<str> = Arc::from("hot");
        let calm: Arc<str> = Arc::from("calm");
        assert!(limiter.admit(&hot, now).is_ok());
        assert!(limiter.admit(&hot, now).is_err());
        assert!(limiter.admit(&calm, now).is_ok());
    }

    /// End to end through auth: the 429 wears the ApiError shape and a
    /// Retry-After header, and spends nothing on refused credentials.
    #[tokio::test]
    async fn the_rate_limited_answer_is_a_429_with_retry_after() {
        let keyring = Arc::new(auth::Keyring::parse(Some("tok".to_string()), None).unwrap());
        let gate = Arc::new(auth::Gate {
            keyring,
            oauth: None,
            fail_limiter: Arc::new(RateLimiter::new(0)),
        });
        let limiter = Arc::new(RateLimiter::new(2));
        let app = Router::new()
            .route("/contexts", get(|| async { "hi" }))
            .layer(axum::middleware::from_fn_with_state(
                limiter,
                enforce_rate_limit,
            ))
            .layer(axum::middleware::from_fn_with_state(
                gate,
                auth::require_bearer,
            ));
        let send = |authorization: &'static str| {
            let app = app.clone();
            async move {
                app.oneshot(
                    HttpRequest::builder()
                        .uri("/contexts")
                        .header("Authorization", authorization)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
            }
        };

        // A wrong token is refused OUTSIDE the limiter: it cannot
        // drain a key's budget.
        assert_eq!(send("Bearer wrong").await.status(), 401);
        for _ in 0..2 {
            assert_eq!(send("Bearer tok").await.status(), 200);
        }
        let refused = send("Bearer tok").await;
        assert_eq!(refused.status(), 429);
        assert!(refused.headers().get(header::RETRY_AFTER).is_some());
        let bytes = axum::body::to_bytes(refused.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["status"], "error");
        assert!(body["error"].as_str().unwrap().contains("budget"), "{body}");
    }
}
