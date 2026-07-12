//! Bearer-token gate for the whole API. Credentials are NAMED keys —
//! the one-token spelling `TAGURU_API_TOKEN` (key name "default")
//! and/or a `TAGURU_API_TOKENS` keyring ("ci:tokA,laptop:tokB") — so
//! the access log can say WHICH caller, one leaked key dies without
//! rotating the others, and rotation itself is an overlap (add the
//! new key, move callers, drop the old).
//!
//! Authorization rides on top: `TAGURU_KEY_SCOPES` grants each key a
//! ROLE (read ⊂ write ⊂ admin) and optionally a context list, and
//! [`enforce_authorization`] holds every request — the in-process MCP
//! dispatch included — to that grant. A key the variable does not
//! name keeps the historical full grant (admin over every context),
//! so existing deployments change nothing by upgrading.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{MatchedPath, Request, State};
use axum::http::{HeaderValue, Method, header};
use axum::middleware::Next;
use axum::response::Response;
use serde::Deserialize;
use subtle::ConstantTimeEq;

use crate::api;

/// Paths that answer without credentials. `/live` (liveness) and
/// `/health` (readiness — 503 while the write path is degraded) are
/// the orchestrator probes and must be reachable unconfigured.
/// `/metrics` carries only aggregates and route templates (no context
/// names, no content), and exempting it keeps scrape configs trivial.
/// The rate limiter and the in-flight ceiling honor the same list:
/// probes and scrapes must not starve behind a chatty client.
pub(crate) const PROBE_EXEMPT: [&str; 3] = ["/health", "/live", "/metrics"];

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

/// What a key may do, ordered by inclusion: `Admin` ⊃ `Write` ⊃
/// `Read`. Read is the retrieval loop; Write adds the ingest loop
/// (create contexts, assert, store passages, heal aliases, retract
/// and re-sync its sources, refresh embeddings); Admin adds the
/// operator verbs (delete contexts, bulk import, flush).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Role {
    Read,
    Write,
    Admin,
}

impl Role {
    fn as_str(self) -> &'static str {
        match self {
            Role::Read => "read",
            Role::Write => "write",
            Role::Admin => "admin",
        }
    }
}

/// One key's grant: its role, and the contexts it may touch (`None` =
/// every context). The default — and the grant of any key
/// `TAGURU_KEY_SCOPES` does not name — is exactly what every key
/// could do before scopes existed: admin, everywhere.
#[derive(Clone, Debug)]
pub struct KeyScope {
    pub role: Role,
    pub contexts: Option<Arc<HashSet<String>>>,
}

impl Default for KeyScope {
    fn default() -> Self {
        Self {
            role: Role::Admin,
            contexts: None,
        }
    }
}

impl KeyScope {
    pub fn allows_context(&self, name: &str) -> bool {
        self.contexts
            .as_ref()
            .is_none_or(|allowed| allowed.contains(name))
    }
}

/// One entry of the `TAGURU_KEY_SCOPES` JSON: `"read"` as shorthand,
/// or `{"role": "write", "contexts": ["sake"]}` in full.
#[derive(Deserialize)]
#[serde(untagged)]
enum ScopeSpec {
    Role(String),
    Full {
        role: String,
        #[serde(default)]
        contexts: Option<Vec<String>>,
    },
}

/// The configured credentials. Empty = auth disabled (development
/// mode, warned about loudly at boot).
pub struct Keyring {
    keys: Vec<(Arc<str>, String)>,
    scopes: HashMap<String, KeyScope>,
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
                // '@' is reserved: OAuth delegations act as "key@client",
                // and `scope_of` strips at the last '@' to inherit the
                // underlying key's grant. A raw key whose name contains
                // '@' would collide with that fallback — silently
                // inheriting an unrelated scoped key's grant instead of
                // the documented default. Refuse it at boot, keyring-style.
                if name.contains('@') {
                    return Err(format!(
                        "key name '{name}' must not contain '@' (reserved for OAuth \
                         delegation, 'key@client')"
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
        Ok(Self {
            keys,
            scopes: HashMap::new(),
        })
    }

    /// Applies `TAGURU_KEY_SCOPES` — a JSON object mapping key names to
    /// grants: `{"ci": "read", "bot": {"role": "write", "contexts":
    /// ["sake"]}}`. Refusals are boot refusals, keyring-style: a scope
    /// naming no configured key is a typo that would silently guard
    /// nobody, and an empty contexts list would grant nothing at all —
    /// omitting the field is how "every context" is said.
    pub fn apply_scopes(&mut self, json: Option<&str>) -> Result<(), String> {
        let Some(json) = json else {
            return Ok(());
        };
        if json.trim().is_empty() {
            return Err("TAGURU_KEY_SCOPES is set but empty".to_string());
        }
        let raw: HashMap<String, ScopeSpec> = serde_json::from_str(json).map_err(|error| {
            format!(
                "TAGURU_KEY_SCOPES is not the documented JSON shape \
                 ({{\"name\": \"role\" | {{\"role\": …, \"contexts\": […]}}}}): {error}"
            )
        })?;
        for (name, spec) in raw {
            if !self.keys.iter().any(|(key, _)| key.as_ref() == name) {
                return Err(format!(
                    "TAGURU_KEY_SCOPES names '{name}', which is no configured key"
                ));
            }
            let (role, contexts) = match spec {
                ScopeSpec::Role(role) => (role, None),
                ScopeSpec::Full { role, contexts } => (role, contexts),
            };
            let role = match role.as_str() {
                "read" => Role::Read,
                "write" => Role::Write,
                "admin" => Role::Admin,
                other => {
                    return Err(format!(
                        "TAGURU_KEY_SCOPES key '{name}': unknown role '{other}' \
                         (read, write, or admin)"
                    ));
                }
            };
            let contexts = match contexts {
                None => None,
                Some(list) if list.is_empty() => {
                    return Err(format!(
                        "TAGURU_KEY_SCOPES key '{name}': an empty contexts list grants \
                         nothing — omit the field to grant every context"
                    ));
                }
                Some(list) => Some(Arc::new(list.into_iter().collect())),
            };
            self.scopes.insert(name, KeyScope { role, contexts });
        }
        Ok(())
    }

    /// The grant behind a key name. OAuth delegations act as
    /// "key@client", so the lookup falls back to the name before the
    /// last '@' — the delegation can never out-rank the key it wraps.
    pub fn scope_of(&self, key_name: &str) -> KeyScope {
        if let Some(scope) = self.scopes.get(key_name) {
            return scope.clone();
        }
        if let Some((base, _)) = key_name.rsplit_once('@')
            && let Some(scope) = self.scopes.get(base)
        {
            return scope.clone();
        }
        KeyScope::default()
    }

    pub fn scoped_key_count(&self) -> usize {
        self.scopes.len()
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

/// Everything the bearer gate consults: the static keyring, the OAuth
/// subsystem when the operator enabled it, and a per-source-IP throttle
/// on FAILED attempts (`TAGURU_AUTH_FAIL_LIMIT_PER_MIN`) so the gate
/// cannot be brute-forced — or used to burn CPU on the constant-time
/// scan — for free.
pub struct Gate {
    pub keyring: Arc<Keyring>,
    pub oauth: Option<Arc<crate::oauth::Oauth>>,
    pub fail_limiter: Arc<crate::limits::RateLimiter>,
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

    // No credential matched. Throttle repeated failures per source IP:
    // successful auth returned above without touching this limiter, so a
    // legitimate caller presenting the right token is never throttled —
    // only a brute-forcer (or a client burning the constant-time scan)
    // spends here, and once over budget gets a 429 instead of a 401.
    if !gate.fail_limiter.is_disabled() {
        let peer = crate::limits::peer_ip(&request).unwrap_or_else(|| Arc::from("peer:unknown"));
        if let Err(retry_after) = gate.fail_limiter.admit(&peer, started_at) {
            let mut response = api::error(
                api::ErrorCode::RateLimited,
                format!("too many failed authentication attempts — retry in {retry_after}s"),
                started_at,
            );
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(retry_after));
            return response;
        }
    }

    let mut response = api::error(
        api::ErrorCode::Unauthorized,
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

/// The least role a (method, route template) demands. Fail closed: a
/// route this table does not classify demands Admin, so an endpoint
/// added without a classification locks down for scoped keys instead
/// of leaking open. `/mcp` itself needs only Read — every tool call
/// dispatches onto a real route in-process and is judged there.
pub(crate) fn required_role(method: &Method, route: &str) -> Role {
    match (method, route) {
        // The retrieval loop, the directory, and the manual.
        (&Method::GET, "/contexts")
        | (&Method::GET, "/groups")
        | (&Method::GET, "/groups/{name}")
        | (&Method::GET, "/contexts/{name}")
        | (&Method::GET, "/contexts/{name}/labels")
        | (&Method::GET, "/contexts/{name}/aliases")
        | (&Method::GET, "/contexts/{name}/sources")
        | (&Method::GET, "/contexts/{name}/export")
        | (&Method::GET, "/protocol")
        | (&Method::POST, "/recall")
        | (&Method::POST, "/query")
        | (&Method::POST, "/sources/search")
        | (&Method::POST, "/contexts/{name}/recall")
        | (&Method::POST, "/contexts/{name}/query")
        | (&Method::POST, "/contexts/{name}/describe")
        | (&Method::POST, "/contexts/{name}/explore")
        | (&Method::POST, "/contexts/{name}/activate")
        | (&Method::POST, "/contexts/{name}/resolve")
        | (&Method::POST, "/contexts/{name}/resolve_label")
        | (&Method::POST, "/contexts/{name}/sources/lookup")
        | (&Method::POST, "/contexts/{name}/sources/search")
        | (&Method::POST, "/contexts/{name}/citations")
        | (&Method::POST, "/contexts/{name}/unreachable_from")
        | (&Method::POST, "/contexts/{name}/vocabulary/audit")
        | (&Method::POST, "/mcp") => Role::Read,
        // The ingest loop — everything the documented agent discipline
        // drives, context creation and per-source re-sync included.
        (&Method::PUT, "/contexts/{name}")
        | (&Method::PATCH, "/contexts/{name}")
        | (&Method::PUT, "/groups/{name}")
        | (&Method::PATCH, "/groups/{name}")
        | (&Method::POST, "/contexts/{name}/associations")
        | (&Method::POST, "/contexts/{name}/aliases")
        | (&Method::DELETE, "/contexts/{name}/aliases")
        | (&Method::POST, "/contexts/{name}/sources")
        | (&Method::POST, "/contexts/{name}/sources/retract")
        | (&Method::POST, "/contexts/{name}/embeddings/refresh") => Role::Write,
        // Operator verbs — and everything unclassified.
        _ => Role::Admin,
    }
}

/// Holds a request to its key's grant. Sits INSIDE the bearer gate on
/// the HTTP surface (it needs WHO), and directly on the in-process
/// `/mcp` dispatch router — `remote_mcp` stamps the outer request's
/// key onto every dispatched tool call, so the two surfaces cannot
/// drift. No key at all (auth off, or an exempt path) means no
/// restriction, exactly as before scopes existed. The resolved
/// [`KeyScope`] rides the request extensions for the handlers that
/// FILTER rather than refuse (`GET /contexts`, the group listings)
/// and for those judging context names that live in the body or the
/// stored record (`/import`, the group writes, the cross-context
/// searches).
pub async fn enforce_authorization(
    State(keyring): State<Arc<Keyring>>,
    matched: Option<MatchedPath>,
    request: Request,
    next: Next,
) -> Response {
    let Some(key) = request.extensions().get::<AuthKey>().cloned() else {
        return next.run(request).await;
    };
    let scope = keyring.scope_of(&key.0);
    let started_at = Instant::now();
    let route = matched
        .as_ref()
        .map(|matched| matched.as_str().to_string())
        .unwrap_or_else(|| "<unmatched>".to_string());
    let required = required_role(request.method(), &route);
    if scope.role < required {
        return api::error(
            api::ErrorCode::Forbidden,
            format!(
                "key '{}' has role '{}', but {} {} needs '{}'",
                key.0,
                scope.role.as_str(),
                request.method(),
                route,
                required.as_str()
            ),
            started_at,
        );
    }
    let (mut parts, body) = request.into_parts();
    // `{name}` is a CONTEXT name on every route but `/groups/{name}`,
    // where it names the group itself: a group's member contexts live
    // in the body and the stored record, out of this middleware's
    // reach, so the group handlers judge them (`api::scope_refusal`).
    // The exclusion names its one route exactly — deny-by-default
    // safe: a future route whose `{name}` is not a context and is not
    // listed here mis-answers 403 for scoped keys, never leaks open
    // (a prefix test would silently swallow future `/groups/...`
    // sub-routes instead of forcing that decision).
    if scope.contexts.is_some() && route != "/groups/{name}" {
        use axum::extract::FromRequestParts as _;
        let context = axum::extract::RawPathParams::from_request_parts(&mut parts, &())
            .await
            .ok()
            .and_then(|params| {
                params
                    .iter()
                    .find(|(name, _)| *name == "name")
                    .map(|(_, value)| value.to_string())
            });
        if let Some(context) = context
            && !scope.allows_context(&context)
        {
            return api::error(
                api::ErrorCode::Forbidden,
                format!("key '{}' has no grant on context '{context}'", key.0),
                started_at,
            );
        }
    }
    let mut request = Request::from_parts(parts, body);
    request.extensions_mut().insert(scope);
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;

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
        // Failed-auth throttle off by default here: these tests assert
        // the 401/200 verdicts, not the throttle. The throttle has its
        // own test with a ConnectInfo layer and a live budget.
        let gate = Arc::new(Gate {
            keyring,
            oauth,
            fail_limiter: Arc::new(crate::limits::RateLimiter::new(0)),
        });
        Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/live", get(|| async { "ok" }))
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
        // '@' in a key name collides with the OAuth-delegation scope
        // fallback, so it is refused at boot rather than silently
        // inheriting an unrelated key's grant.
        let error = Keyring::parse(None, Some("bot@internal:sekrit-z".to_string()))
            .err()
            .expect("an '@' key name must be refused");
        assert!(error.contains('@') && !error.contains("sekrit"), "{error}");
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
        for probe in ["/health", "/live", "/metrics"] {
            assert_eq!(
                status_of(app(Arc::clone(&keyring)), probe, None).await,
                200,
                "{probe} must answer unconfigured"
            );
        }
        assert_eq!(status_of(app(keyring), "/contexts", None).await, 401);
    }

    /// Repeated failures from one source IP trip a 429 with a
    /// Retry-After, but a valid token from that very IP still passes:
    /// the throttle bites brute force, never a caller who finally
    /// presents the right credential.
    #[tokio::test]
    async fn failed_auth_is_throttled_per_source_ip() {
        use axum::extract::connect_info::MockConnectInfo;
        use std::net::SocketAddr;

        let gate = Arc::new(Gate {
            keyring: ring(Some("s3cret"), None),
            oauth: None,
            fail_limiter: Arc::new(crate::limits::RateLimiter::new(3)),
        });
        let app = Router::new()
            .route("/contexts", get(|| async { "secret" }))
            .layer(axum::middleware::from_fn_with_state(gate, require_bearer))
            // Outermost, so every request carries a peer address before
            // the gate reads one — all from the same mocked IP, so they
            // share a bucket and the budget trips.
            .layer(MockConnectInfo(SocketAddr::from(([127, 0, 0, 1], 40000))));
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

        // The 3/min budget spends on failures...
        for _ in 0..3 {
            assert_eq!(
                send("Bearer wrong").await.status(),
                StatusCode::UNAUTHORIZED
            );
        }
        // ...and the next failure is throttled, not merely refused.
        let throttled = send("Bearer wrong").await;
        assert_eq!(throttled.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(throttled.headers().get(header::RETRY_AFTER).is_some());

        // The right token still opens the gate from the same IP: success
        // never reaches the throttle.
        assert_eq!(send("Bearer s3cret").await.status(), StatusCode::OK);
    }

    /// Scope grants parse strictly: a scope naming no configured key, a
    /// role typo, an empty contexts list, or non-JSON all refuse the
    /// boot instead of arming a partial authorization table.
    #[test]
    fn scope_grants_parse_strictly_and_resolve_with_the_oauth_fallback() {
        let mut keyring =
            Keyring::parse(None, Some("boss:tok-a,reader:tok-b,bot:tok-c".to_string())).unwrap();
        keyring
            .apply_scopes(Some(
                r#"{"reader": "read", "bot": {"role": "write", "contexts": ["sake"]}}"#,
            ))
            .unwrap();
        assert_eq!(keyring.scoped_key_count(), 2);
        // Unnamed keys keep the historical full grant.
        assert_eq!(keyring.scope_of("boss").role, Role::Admin);
        assert!(keyring.scope_of("boss").allows_context("anything"));
        assert_eq!(keyring.scope_of("reader").role, Role::Read);
        // OAuth delegations ("key@client") inherit the key's grant.
        let delegated = keyring.scope_of("bot@claude-abc123");
        assert_eq!(delegated.role, Role::Write);
        assert!(delegated.allows_context("sake"));
        assert!(!delegated.allows_context("bunko"));

        for (scopes, complaint) in [
            (r#"{"ghost": "read"}"#, "no configured key"),
            (r#"{"reader": "supreme"}"#, "unknown role"),
            (r#"{"reader": {"role": "read", "contexts": []}}"#, "empty"),
            ("not json", "documented JSON shape"),
            ("", "empty"),
        ] {
            let mut keyring = Keyring::parse(None, Some("reader:tok-b".to_string())).unwrap();
            let error = keyring.apply_scopes(Some(scopes)).unwrap_err();
            assert!(error.contains(complaint), "{scopes} → {error}");
        }
    }

    /// The role table fails closed: an endpoint nobody classified
    /// demands admin, so a scoped key locks out of new surface until
    /// someone decides otherwise.
    #[test]
    fn unclassified_routes_demand_admin() {
        assert_eq!(
            required_role(&Method::POST, "/contexts/{name}/future_thing"),
            Role::Admin
        );
        assert_eq!(
            required_role(&Method::GET, "/contexts/{name}/export"),
            Role::Read
        );
        assert_eq!(
            required_role(&Method::POST, "/contexts/{name}/sources/retract"),
            Role::Write
        );
        assert_eq!(required_role(&Method::POST, "/import"), Role::Admin);
        assert_eq!(
            required_role(&Method::DELETE, "/contexts/{name}"),
            Role::Admin
        );
    }

    /// The authorization layer end to end: role refusals, context
    /// grants, and the untouched full-grant default, all in the
    /// ApiError shape with a 403.
    #[tokio::test]
    async fn scoped_keys_are_held_to_role_and_context() {
        let mut keyring =
            Keyring::parse(None, Some("boss:tok-a,reader:tok-b,bot:tok-c".to_string())).unwrap();
        keyring
            .apply_scopes(Some(
                r#"{"reader": "read", "bot": {"role": "write", "contexts": ["sake"]}}"#,
            ))
            .unwrap();
        let keyring = Arc::new(keyring);
        let app = || {
            let gate = Arc::new(Gate {
                keyring: Arc::clone(&keyring),
                oauth: None,
                fail_limiter: Arc::new(crate::limits::RateLimiter::new(0)),
            });
            Router::new()
                .route("/contexts", get(|| async { "rows" }))
                .route(
                    "/contexts/{name}/recall",
                    axum::routing::post(|| async { "hits" }),
                )
                .route(
                    "/contexts/{name}/associations",
                    axum::routing::post(|| async { "landed" }),
                )
                .route(
                    "/contexts/{name}",
                    axum::routing::delete(|| async { "gone" }),
                )
                // Authorization innermost, the bearer gate outside it —
                // the same nesting main.rs builds.
                .layer(axum::middleware::from_fn_with_state(
                    Arc::clone(&keyring),
                    enforce_authorization,
                ))
                .layer(axum::middleware::from_fn_with_state(gate, require_bearer))
        };
        let send = |method: &'static str, path: &'static str, token: &'static str| {
            let app = app();
            async move {
                app.oneshot(
                    HttpRequest::builder()
                        .method(method)
                        .uri(path)
                        .header("Authorization", format!("Bearer {token}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap()
            }
        };

        // The read key runs the retrieval loop and nothing else.
        assert_eq!(
            send("POST", "/contexts/sake/recall", "tok-b")
                .await
                .status(),
            200
        );
        let refused = send("POST", "/contexts/sake/associations", "tok-b").await;
        assert_eq!(refused.status(), StatusCode::FORBIDDEN);
        let bytes = axum::body::to_bytes(refused.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["status"], "error");
        assert!(
            body["error"].as_str().unwrap().contains("needs 'write'"),
            "{body}"
        );

        // The context-scoped write key writes inside its grant only,
        // and never reaches the admin verbs.
        assert_eq!(
            send("POST", "/contexts/sake/associations", "tok-c")
                .await
                .status(),
            200
        );
        let outside = send("POST", "/contexts/bunko/associations", "tok-c").await;
        assert_eq!(outside.status(), StatusCode::FORBIDDEN);
        let bytes = axum::body::to_bytes(outside.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains("no grant on context 'bunko'"),
            "{body}"
        );
        assert_eq!(
            send("DELETE", "/contexts/sake", "tok-c").await.status(),
            StatusCode::FORBIDDEN
        );

        // The unscoped key keeps the historical full grant.
        assert_eq!(
            send("DELETE", "/contexts/sake", "tok-a").await.status(),
            200
        );
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
