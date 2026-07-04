//! Per-request wall-clock ceiling. Body size and per-parameter cost
//! caps live where they apply (the body-limit layer in main, the
//! clamps in api); this module owns the one cross-cutting limit no
//! handler can enforce for itself: total time.

use std::time::{Duration, Instant};

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;

use crate::api;

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
}
