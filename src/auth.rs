//! Bearer-token gate for the whole API. Credentials are NAMED keys —
//! the one-token spelling `TAGURU_API_TOKEN` (key name "default")
//! and/or a `TAGURU_API_TOKENS` keyring ("ci:tokA,laptop:tokB") — so
//! the access log can say WHICH caller, one leaked key dies without
//! rotating the others, and rotation itself is an overlap (add the
//! new key, move callers, drop the old). Tenancy still comes later:
//! every key opens every context.

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
/// content), and exempting it keeps scrape configs trivial. The rate
/// limiter honors the same list: probes and scrapes must not starve
/// behind a chatty client.
pub(crate) const EXEMPT: [&str; 2] = ["/health", "/metrics"];

/// The authenticated key's name, attached to the RESPONSE: the access
/// log middleware sits outside this one, so response extensions are
/// the channel that reaches it.
#[derive(Clone)]
pub struct AuthKey(pub Arc<str>);

/// The configured credentials. Empty = auth disabled (development
/// mode, warned about loudly at boot).
pub struct Keyring {
    keys: Vec<(Arc<str>, String)>,
}

impl Keyring {
    /// Builds the ring from the two variables. Malformed input is a
    /// boot REFUSAL, not a skipped entry — a credential that silently
    /// failed to arm reads as an auth hole at the worst possible
    /// moment. Error texts name entries by position, never by content:
    /// a malformed entry is likely a token pasted without its name.
    pub fn parse(single: Option<String>, named: Option<String>) -> Result<Self, String> {
        let mut keys: Vec<(Arc<str>, String)> = Vec::new();
        if let Some(token) = single {
            if token.is_empty() {
                return Err("TAGURU_API_TOKEN is set but empty".to_string());
            }
            keys.push((Arc::from("default"), token));
        }
        if let Some(ring) = named {
            if ring.trim().is_empty() {
                return Err("TAGURU_API_TOKENS is set but holds no entries".to_string());
            }
            for (position, entry) in ring.split(',').enumerate() {
                let entry = entry.trim();
                if entry.is_empty() {
                    continue; // tolerate a trailing comma
                }
                let Some((name, token)) = entry.split_once(':') else {
                    return Err(format!(
                        "TAGURU_API_TOKENS entry #{} is not name:token",
                        position + 1
                    ));
                };
                let name = name.trim();
                if name.is_empty() || token.is_empty() {
                    return Err(format!(
                        "TAGURU_API_TOKENS entry #{} has an empty name or token",
                        position + 1
                    ));
                }
                if keys.iter().any(|(existing, _)| existing.as_ref() == name) {
                    return Err(format!("key name '{name}' is configured twice"));
                }
                keys.push((Arc::from(name), token.to_string()));
            }
            if keys.is_empty() {
                return Err("TAGURU_API_TOKENS is set but holds no entries".to_string());
            }
        }
        Ok(Self { keys })
    }

    pub fn is_disabled(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Constant-shape scan: EVERY key is compared even after a match,
    /// so response timing says nothing about which key (if any) a
    /// guess grazed.
    fn authenticate(&self, presented: &str) -> Option<Arc<str>> {
        let mut matched = None;
        for (name, token) in &self.keys {
            if bool::from(presented.as_bytes().ct_eq(token.as_bytes())) {
                matched = Some(Arc::clone(name));
            }
        }
        matched
    }
}

/// Refuses any non-exempt request whose `Authorization: Bearer` token
/// matches no configured key. An empty keyring disables the gate
/// entirely — development mode. On success the key's NAME rides the
/// response out to the access log.
pub async fn require_bearer(
    State(keyring): State<Arc<Keyring>>,
    request: Request,
    next: Next,
) -> Response {
    if keyring.is_disabled() {
        return next.run(request).await;
    }
    if EXEMPT.contains(&request.uri().path()) {
        return next.run(request).await;
    }

    let started_at = Instant::now();
    let presented = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    if let Some(key) = presented.and_then(|token| keyring.authenticate(token)) {
        let mut request = request;
        // Inner layers (the rate limiter) key their work on WHO; the
        // response copy below feeds the access log on the way out.
        request.extensions_mut().insert(AuthKey(Arc::clone(&key)));
        let mut response = next.run(request).await;
        response.extensions_mut().insert(AuthKey(key));
        return response;
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

    fn ring(single: Option<&str>, named: Option<&str>) -> Arc<Keyring> {
        Arc::new(Keyring::parse(single.map(String::from), named.map(String::from)).unwrap())
    }

    fn app(keyring: Arc<Keyring>) -> Router {
        Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/metrics", get(|| async { "counts" }))
            .route("/contexts", get(|| async { "secret" }))
            .layer(axum::middleware::from_fn_with_state(
                keyring,
                require_bearer,
            ))
    }

    async fn respond(app: Router, path: &str, authorization: Option<&str>) -> Response {
        let mut request = HttpRequest::builder().uri(path);
        if let Some(authorization) = authorization {
            request = request.header("Authorization", authorization);
        }
        app.oneshot(request.body(Body::empty()).unwrap())
            .await
            .unwrap()
    }

    async fn status_of(app: Router, path: &str, authorization: Option<&str>) -> u16 {
        respond(app, path, authorization).await.status().as_u16()
    }

    #[tokio::test]
    async fn the_correct_token_passes_through() {
        assert_eq!(
            status_of(
                app(ring(Some("s3cret"), None)),
                "/contexts",
                Some("Bearer s3cret")
            )
            .await,
            200
        );
    }

    /// Every named key opens the gate, and the response carries WHICH —
    /// the fact the access log prints.
    #[tokio::test]
    async fn named_keys_authenticate_and_name_themselves() {
        let keyring = ring(Some("legacy"), Some("ci:tok-a, laptop:tok-b"));
        for (token, expected) in [("tok-a", "ci"), ("tok-b", "laptop"), ("legacy", "default")] {
            let response = respond(
                app(Arc::clone(&keyring)),
                "/contexts",
                Some(&format!("Bearer {token}")),
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            let key = response.extensions().get::<AuthKey>().expect("key name");
            assert_eq!(key.0.as_ref(), expected);
        }
        assert_eq!(
            status_of(app(keyring), "/contexts", Some("Bearer tok-c")).await,
            401
        );
    }

    /// Misconfigured credentials refuse to boot instead of silently
    /// arming a partial keyring — and the errors never echo a token.
    #[test]
    fn malformed_keyrings_are_refused_without_echoing_secrets() {
        let cases = [
            (Some("".to_string()), None),
            (None, Some("".to_string())),
            (None, Some("sekrit-pasted-alone".to_string())),
            (None, Some("ci:".to_string())),
            (None, Some(":sekrit-nameless".to_string())),
            (None, Some("ci:sekrit-a,ci:sekrit-b".to_string())),
            (
                Some("sekrit-x".to_string()),
                Some("default:sekrit-y".to_string()),
            ),
        ];
        for (single, named) in cases {
            let error = Keyring::parse(single.clone(), named.clone())
                .err()
                .unwrap_or_else(|| panic!("{single:?}/{named:?} must be refused"));
            assert!(
                !error.contains("sekrit"),
                "error text must not echo secrets: {error}"
            );
        }
        // A trailing comma is tolerable sloppiness, not a hole.
        assert_eq!(
            Keyring::parse(None, Some("ci:tok-a,".to_string()))
                .unwrap()
                .key_count(),
            1
        );
    }

    #[tokio::test]
    async fn missing_and_wrong_tokens_are_rejected_with_the_api_error_shape() {
        for authorization in [None, Some("Bearer wrong"), Some("Basic s3cret")] {
            let response =
                respond(app(ring(Some("s3cret"), None)), "/contexts", authorization).await;
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
        let keyring = Arc::new(Keyring::parse(None, None).unwrap());
        assert!(keyring.is_disabled());
        assert_eq!(status_of(app(keyring), "/contexts", None).await, 200);
    }

    #[tokio::test]
    async fn health_and_metrics_are_exempt_while_everything_else_is_gated() {
        let keyring = ring(Some("s3cret"), None);
        assert_eq!(
            status_of(app(Arc::clone(&keyring)), "/health", None).await,
            200
        );
        assert_eq!(
            status_of(app(Arc::clone(&keyring)), "/metrics", None).await,
            200
        );
        assert_eq!(status_of(app(keyring), "/contexts", None).await, 401);
    }
}
