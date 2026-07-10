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
pub(crate) const PROBE_EXEMPT: [&str; 2] = ["/health", "/metrics"];

/// What answers without a bearer token. With OAuth enabled, its
/// discovery and grant endpoints join the probes — they exist to
/// CREATE credentials — but unlike the probes they stay rate-limited.
fn is_auth_exempt(path: &str, oauth_enabled: bool) -> bool {
    PROBE_EXEMPT.contains(&path)
        || (oauth_enabled
            && (path.starts_with("/oauth/") || path.starts_with("/.well-known/oauth-")))
}

/// Strips the `Bearer` auth-scheme, RFC 7235 §2.1's case-insensitively
/// (`bearer`, `BEARER`, `Bearer` alike) — only the scheme, never the
/// token that follows it.
fn strip_bearer_prefix(value: &str) -> Option<&str> {
    let prefix_len = "Bearer ".len();
    let (scheme, rest) = value.split_at_checked(prefix_len)?;
    scheme.eq_ignore_ascii_case("Bearer ").then_some(rest)
}

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
    /// guess grazed. `pub(crate)`: the OAuth consent step validates
    /// the delegated key through the same scan.
    pub(crate) fn authenticate(&self, presented: &str) -> Option<Arc<str>> {
        let mut matched = None;
        for (name, token) in &self.keys {
            if bool::from(presented.as_bytes().ct_eq(token.as_bytes())) {
                matched = Some(Arc::clone(name));
            }
        }
        matched
    }
}

/// Everything the bearer gate consults: the static keyring, and the
/// OAuth subsystem when the operator enabled it.
pub struct Gate {
    pub keyring: Arc<Keyring>,
    pub oauth: Option<Arc<crate::oauth::Oauth>>,
}

/// Refuses any non-exempt request whose `Authorization: Bearer` token
/// matches no configured credential. An empty keyring disables the
/// gate entirely — development mode. Static keys open everything;
/// OAuth access tokens open `/mcp` only — the delegation is scoped to
/// the MCP surface, never the raw API. On success the credential's
/// NAME rides the response out to the access log.
pub async fn require_bearer(
    State(gate): State<Arc<Gate>>,
    request: Request,
    next: Next,
) -> Response {
    if gate.keyring.is_disabled() {
        return next.run(request).await;
    }
    if is_auth_exempt(request.uri().path(), gate.oauth.is_some()) {
        return next.run(request).await;
    }

    let started_at = Instant::now();
    // RFC 7230 §3.2.2: Authorization is single-valued. Two copies are a
    // malformed request — and the shape of a smuggling attempt, where a
    // front proxy authenticates on one copy while this server reads the
    // other. Refuse outright rather than silently honor the first and
    // trust the proxy agreed on which one that was.
    let mut authorizations = request.headers().get_all(header::AUTHORIZATION).iter();
    let presented = match (authorizations.next(), authorizations.next()) {
        (Some(value), None) => value.to_str().ok().and_then(strip_bearer_prefix),
        _ => None,
    };
    let key = presented.and_then(|token| {
        gate.keyring
            .authenticate(token)
            .or_else(|| match (&gate.oauth, request.uri().path()) {
                (Some(oauth), "/mcp") => oauth.authenticate(token, crate::oauth::now_secs()),
                _ => None,
            })
    });
    if let Some(key) = key {
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
    // With OAuth on, the 401 names the discovery document — the hint
    // MCP clients key their authorization flow on (RFC 9728).
    let challenge = match &gate.oauth {
        Some(oauth) => HeaderValue::from_str(&format!(
            "Bearer resource_metadata=\"{}\"",
            oauth.resource_metadata_url()
        ))
        .unwrap_or_else(|_| HeaderValue::from_static("Bearer")),
        None => HeaderValue::from_static("Bearer"),
    };
    response
        .headers_mut()
        .insert(header::WWW_AUTHENTICATE, challenge);
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
        gate_app(keyring, None)
    }

    fn gate_app(keyring: Arc<Keyring>, oauth: Option<Arc<crate::oauth::Oauth>>) -> Router {
        let gate = Arc::new(Gate { keyring, oauth });
        Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/metrics", get(|| async { "counts" }))
            .route("/contexts", get(|| async { "secret" }))
            .route("/mcp", get(|| async { "tools" }))
            .layer(axum::middleware::from_fn_with_state(gate, require_bearer))
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

    /// RFC 7235 §2.1: auth-scheme is case-insensitive — `bearer`,
    /// `BEARER`, and `Bearer` must all introduce the same credentials.
    #[tokio::test]
    async fn the_bearer_scheme_is_case_insensitive() {
        let keyring = ring(Some("s3cret"), None);
        for scheme in ["Bearer", "bearer", "BEARER", "BeArEr"] {
            assert_eq!(
                status_of(
                    app(Arc::clone(&keyring)),
                    "/contexts",
                    Some(&format!("{scheme} s3cret"))
                )
                .await,
                200,
                "scheme {scheme} must authenticate"
            );
        }
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

    /// OAuth access tokens are a scoped delegation: they open /mcp,
    /// nothing else — and anonymous 401s advertise the discovery
    /// document OAuth clients key their flow on.
    #[tokio::test]
    async fn oauth_tokens_open_mcp_only_and_401s_advertise_discovery() {
        let dir = std::env::temp_dir().join(format!("taguru-authoauth-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let oauth = Arc::new(crate::oauth::Oauth::open("https://memory.example", &dir));
        let client = oauth
            .register_client("claude", vec!["https://claude.ai/cb".to_string()])
            .unwrap();
        let now = crate::oauth::now_secs();
        let verifier = "0123456789012345678901234567890123456789012345";
        let code = oauth.issue_code(
            &client,
            "https://claude.ai/cb",
            &crate::oauth::s256_challenge(verifier),
            "laptop",
            now,
        );
        let grant = oauth
            .exchange_code(
                &client.client_id,
                &code,
                verifier,
                "https://claude.ai/cb",
                now,
            )
            .unwrap();

        let keyring = ring(Some("s3cret"), None);
        let bearer = format!("Bearer {}", grant.access_token);
        let app = || gate_app(Arc::clone(&keyring), Some(Arc::clone(&oauth)));
        assert_eq!(status_of(app(), "/mcp", Some(&bearer)).await, 200);
        assert_eq!(status_of(app(), "/contexts", Some(&bearer)).await, 401);
        // The static key keeps opening everything.
        assert_eq!(
            status_of(app(), "/contexts", Some("Bearer s3cret")).await,
            200
        );
        // Anonymous callers learn where to start the OAuth dance.
        let refused = respond(app(), "/mcp", None).await;
        let challenge = refused.headers()[header::WWW_AUTHENTICATE]
            .to_str()
            .unwrap();
        assert!(
            challenge.contains(
                "resource_metadata=\"https://memory.example/.well-known/oauth-protected-resource\""
            ),
            "{challenge}"
        );

        let _ = std::fs::remove_dir_all(dir);
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

    /// RFC 7230 §3.2.2: a repeated Authorization header is malformed. Rather
    /// than pick a winner — which a proxy and the gate might disagree on — the
    /// gate fails closed, so smuggling a second credential past the first
    /// never authenticates.
    #[tokio::test]
    async fn duplicate_authorization_headers_are_refused() {
        let request = HttpRequest::builder()
            .uri("/contexts")
            .header("Authorization", "Bearer wrong")
            .header("Authorization", "Bearer s3cret")
            .body(Body::empty())
            .unwrap();
        let response = app(ring(Some("s3cret"), None))
            .oneshot(request)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
