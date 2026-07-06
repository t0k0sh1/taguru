//! Per-request wall-clock ceiling and the per-key request budget.
//! Body size and per-parameter cost caps live where they apply (the
//! body-limit layer in main, the clamps in api); this module owns the
//! cross-cutting limits no handler can enforce for itself: total time,
//! and total request rate per credential.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::Response;

use crate::api;
use crate::auth;

/// Races the rest of the stack against the configured budget; a loss
/// is a 408 in the ApiError shape — retryable and client-actionable
/// (narrow the query), unlike a 503's "server unhealthy".
///
/// Caveat, documented for operators: the embedding-backed endpoints
/// run their provider round trip inside `block_in_place`, which a
/// future race cannot preempt — those requests only see the deadline
/// after the blocking call returns. With TAGURU_EMBED_URL configured,
/// set the budget above the provider's own 60s ceiling.
pub async fn enforce_timeout(
    State(budget): State<Duration>,
    request: Request,
    next: Next,
) -> Response {
    let started_at = Instant::now();
    match tokio::time::timeout(budget, next.run(request)).await {
        Ok(response) => response,
        Err(_) => api::error(
            StatusCode::REQUEST_TIMEOUT,
            format!(
                "request exceeded the {}s budget; narrow the query \
                 (TAGURU_REQUEST_TIMEOUT_SECS tunes this)",
                budget.as_secs()
            ),
            started_at,
        ),
    }
}

/// Per-key token buckets (`TAGURU_RATE_LIMIT_PER_MIN`): each credential
/// gets a minute's allowance as capacity, refilled continuously, so a
/// client may burst its whole budget and then settle to the sustained
/// rate. 0 disables the gate. The map is keyed by configured key names
/// (plus "anon" when auth is off) — bounded by configuration, never by
/// callers, so it cannot be grown from outside.
pub struct RateLimiter {
    per_minute: u32,
    buckets: Mutex<HashMap<Arc<str>, Bucket>>,
}

struct Bucket {
    tokens: f64,
    refreshed: Instant,
}

impl RateLimiter {
    pub fn new(per_minute: u32) -> Self {
        Self {
            per_minute,
            buckets: Mutex::new(HashMap::new()),
        }
    }

    pub fn is_disabled(&self) -> bool {
        self.per_minute == 0
    }

    /// Admits or refuses one request for `key` at `now`; refusal names
    /// the seconds until a token exists again — the Retry-After value.
    /// `now` is a parameter so tests refill by arithmetic, not sleep.
    fn admit(&self, key: &Arc<str>, now: Instant) -> Result<(), u64> {
        let capacity = f64::from(self.per_minute);
        let per_second = capacity / 60.0;
        let mut buckets = self.buckets.lock().unwrap();
        let bucket = buckets.entry(Arc::clone(key)).or_insert(Bucket {
            tokens: capacity,
            refreshed: now,
        });
        let elapsed = now.duration_since(bucket.refreshed).as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * per_second).min(capacity);
        bucket.refreshed = now;
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
    // Auth (outside) stamped WHO onto the request; without auth
    // configured, every caller shares the anonymous bucket.
    let key = request
        .extensions()
        .get::<auth::AuthKey>()
        .map(|key| Arc::clone(&key.0))
        .unwrap_or_else(|| Arc::from("anon"));
    match limiter.admit(&key, Instant::now()) {
        Ok(()) => next.run(request).await,
        Err(retry_after) => {
            let mut response = api::error(
                StatusCode::TOO_MANY_REQUESTS,
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
    use axum::routing::get;
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
