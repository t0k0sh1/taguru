//! Bearer-token gate for the whole API. One shared secret
//! (`TAGURU_API_TOKEN`) — per-caller keys and tenancy come later; this
//! is the minimum for exposing the server beyond localhost, behind a
//! TLS-terminating proxy.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::Next;
use axum::response::Response;
use subtle::ConstantTimeEq;

use crate::api;

/// Paths that answer without credentials. `/health` is the liveness
/// probe — orchestrators must reach it unconfigured. `/metrics`
/// carries only aggregates and route templates (no context names, no
/// content), and exempting it keeps scrape configs trivial.
const EXEMPT: [&str; 2] = ["/health", "/metrics"];

/// Refuses any non-exempt request whose `Authorization: Bearer` token
/// does not match the configured secret. `None` (no `TAGURU_API_TOKEN`)
/// disables the gate entirely — development mode, warned about loudly
/// at startup. The comparison is constant-time (`subtle::ct_eq`), so
/// response timing cannot leak how much of a guess matched.
pub async fn require_bearer(
    State(expected): State<Arc<Option<String>>>,
    request: Request,
    next: Next,
) -> Response {
    let Some(expected) = expected.as_deref() else {
        return next.run(request).await;
    };
    if EXEMPT.contains(&request.uri().path()) {
        return next.run(request).await;
    }

    let started_at = Instant::now();
    let presented = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    let authorized =
        presented.is_some_and(|token| token.as_bytes().ct_eq(expected.as_bytes()).into());
    if authorized {
        return next.run(request).await;
    }

    let mut response = api::error(
        StatusCode::UNAUTHORIZED,
        "missing or invalid bearer token (send Authorization: Bearer <token>)",
        started_at,
    );
    response
        .headers_mut()
        .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::body::Body;
    use axum::http::Request as HttpRequest;
    use axum::routing::get;
    use tower::util::ServiceExt;

    fn app(token: Option<&str>) -> Router {
        let expected = Arc::new(token.map(String::from));
        Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/metrics", get(|| async { "counts" }))
            .route("/contexts", get(|| async { "secret" }))
            .layer(axum::middleware::from_fn_with_state(
                expected,
                require_bearer,
            ))
    }

    async fn status_of(app: Router, path: &str, authorization: Option<&str>) -> u16 {
        let mut request = HttpRequest::builder().uri(path);
        if let Some(authorization) = authorization {
            request = request.header("Authorization", authorization);
        }
        let response = app
            .oneshot(request.body(Body::empty()).unwrap())
            .await
            .unwrap();
        response.status().as_u16()
    }

    #[tokio::test]
    async fn the_correct_token_passes_through() {
        assert_eq!(
            status_of(app(Some("s3cret")), "/contexts", Some("Bearer s3cret")).await,
            200
        );
    }

    #[tokio::test]
    async fn missing_and_wrong_tokens_are_rejected_with_the_api_error_shape() {
        for authorization in [None, Some("Bearer wrong"), Some("Basic s3cret")] {
            let response = app(Some("s3cret"))
                .oneshot(
                    authorization
                        .iter()
                        .fold(HttpRequest::builder().uri("/contexts"), |request, value| {
                            request.header("Authorization", *value)
                        })
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
            assert_eq!(
                response.headers()[header::WWW_AUTHENTICATE],
                HeaderValue::from_static("Bearer")
            );
            let bytes = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["status"], "error");
            assert!(
                body["error"].is_string() && body["time"].is_number(),
                "{body}"
            );
        }
    }

    #[tokio::test]
    async fn auth_is_disabled_when_no_token_is_configured() {
        assert_eq!(status_of(app(None), "/contexts", None).await, 200);
    }

    #[tokio::test]
    async fn health_and_metrics_are_exempt_while_everything_else_is_gated() {
        assert_eq!(status_of(app(Some("s3cret")), "/health", None).await, 200);
        assert_eq!(status_of(app(Some("s3cret")), "/metrics", None).await, 200);
        assert_eq!(status_of(app(Some("s3cret")), "/contexts", None).await, 401);
    }
}
