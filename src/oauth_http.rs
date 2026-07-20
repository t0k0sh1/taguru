//! The HTTP face of the embedded OAuth server: the two well-known
//! documents, dynamic registration, the consent page, and the token
//! endpoint. All of it is exempt from bearer auth (these endpoints
//! exist to CREATE credentials) and rate-limited under the anonymous
//! bucket like any unauthenticated caller.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{ConnectInfo, Form, Query, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::auth::SharedKeyring;
use crate::oauth::{Oauth, escape_html, now_secs};

#[derive(Clone)]
pub struct OauthState {
    pub oauth: Arc<Oauth>,
    /// Loaded per consent POST, so a delegation minted after a hot
    /// reload validates the key against the table as rotated.
    pub keyring: SharedKeyring,
    /// Shared with the bearer gate. The consent POST runs the same
    /// constant-time keyring scan (`authenticate`) as the gate but
    /// unauthenticated, so a wrong key here has to spend the same
    /// per-source-IP failure budget — otherwise the page is a free
    /// brute-force (and CPU-burn) oracle against the keyring whenever the
    /// optional per-key rate limit is off, which is its default.
    pub fail_limiter: Arc<crate::limits::RateLimiter>,
}

pub fn router(state: OauthState) -> Router {
    Router::new()
        .route(
            "/.well-known/oauth-protected-resource",
            get(protected_resource),
        )
        // RFC 9728 puts the metadata for a path-carrying resource at a
        // path-inserted location; clients probe both spellings.
        .route(
            "/.well-known/oauth-protected-resource/mcp",
            get(protected_resource),
        )
        .route(
            "/.well-known/oauth-authorization-server",
            get(authorization_server),
        )
        .route("/oauth/register", post(register))
        .route("/oauth/authorize", get(consent_page).post(approve))
        .route("/oauth/token", post(token))
        .with_state(state)
}

async fn protected_resource(State(state): State<OauthState>) -> Response {
    Json(state.oauth.protected_resource_metadata()).into_response()
}

async fn authorization_server(State(state): State<OauthState>) -> Response {
    Json(state.oauth.authorization_server_metadata()).into_response()
}

/// RFC 7591 dynamic registration. Public clients only; errors use the
/// registration error shape, with the reason spelled out.
async fn register(State(state): State<OauthState>, Json(body): Json<Value>) -> Response {
    let client_name = body
        .get("client_name")
        .and_then(Value::as_str)
        .unwrap_or("client");
    let redirect_uris: Vec<String> = body
        .get("redirect_uris")
        .and_then(Value::as_array)
        .map(|uris| {
            uris.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    // `register_client` serializes the whole client store and fsyncs it
    // (via `persist`), so keep that off the async worker like every other
    // write path does.
    match tokio::task::block_in_place(|| state.oauth.register_client(client_name, redirect_uris)) {
        Ok(client) => (
            StatusCode::CREATED,
            Json(json!({
                "client_id": client.client_id,
                "client_name": client.client_name,
                "redirect_uris": client.redirect_uris,
                "token_endpoint_auth_method": "none",
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
            })),
        )
            .into_response(),
        Err(reason) => (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "invalid_client_metadata",
                "error_description": reason,
            })),
        )
            .into_response(),
    }
}

#[derive(Deserialize)]
struct AuthorizeParams {
    #[serde(default)]
    response_type: String,
    #[serde(default)]
    client_id: String,
    #[serde(default)]
    redirect_uri: String,
    #[serde(default)]
    state: String,
    #[serde(default)]
    code_challenge: String,
    #[serde(default)]
    code_challenge_method: String,
    #[serde(default)]
    resource: String,
    /// Present only on the consent POST: the operator's API key.
    #[serde(default)]
    key: String,
}

/// The caller's source address, when the listener attached one.
/// `into_make_service_with_connect_info` (how this server serves) always
/// does; a bare test `oneshot` may not. Infallible on purpose — a
/// missing address degrades the consent throttle to one shared bucket
/// instead of turning the POST into a 500, exactly as `peer_ip` does for
/// the bearer gate.
struct PeerAddr(Option<SocketAddr>);

impl<S: Send + Sync> axum::extract::FromRequestParts<S> for PeerAddr {
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        Ok(PeerAddr(
            parts
                .extensions
                .get::<ConnectInfo<SocketAddr>>()
                .map(|connect| connect.0),
        ))
    }
}

/// Validates the (client, redirect) pair — the two fields that decide
/// whether redirecting an error is safe at all. `Err` is a 400 page
/// (boxed: a Response outweighs the happy path), never a redirect to
/// an unverified target.
fn checked_redirect(
    state: &OauthState,
    params: &AuthorizeParams,
) -> Result<crate::oauth::Client, Box<Response>> {
    let refuse = |reason: &str| {
        Box::new(hardened_html(
            StatusCode::BAD_REQUEST,
            format!("<h1>Cannot continue</h1><p>{}</p>", escape_html(reason)),
        ))
    };
    let Some(client) = state.oauth.client(&params.client_id) else {
        return Err(refuse("unknown client_id — register first (RFC 7591)"));
    };
    if !client
        .redirect_uris
        .iter()
        .any(|uri| uri == &params.redirect_uri)
    {
        return Err(refuse("redirect_uri is not registered for this client"));
    }
    Ok(client)
}

/// Parameter problems AFTER the redirect target is verified go back to
/// the client as OAuth error redirects, per spec.
fn error_redirect(params: &AuthorizeParams, error: &str) -> Response {
    let mut target = format!(
        "{}{}error={error}",
        params.redirect_uri,
        query_separator(&params.redirect_uri)
    );
    if !params.state.is_empty() {
        target.push_str(&format!("&state={}", encode_query_value(&params.state)));
    }
    Redirect::to(&target).into_response()
}

/// `?` to open the query, or `&` to extend one the redirect URI already
/// carries. Registration only checks the scheme, so a registered
/// `https://app/cb?tenant=x` is legal; RFC 6749 §3.1.2 requires that
/// existing query to be kept and our parameters appended to it. An
/// unconditional `?` would yield `...?tenant=x?code=...`, folding the
/// code and state into `tenant`'s value and breaking the callback.
fn query_separator(redirect_uri: &str) -> char {
    if redirect_uri.contains('?') { '&' } else { '?' }
}

/// `state` is opaque, attacker-controlled data that gets spliced
/// straight into a redirect's query string (here and in `approve`
/// below) — percent-encode it so a value like `evil&code=injected`
/// round-trips as one opaque field instead of opening a second query
/// parameter.
fn encode_query_value(value: &str) -> String {
    url::form_urlencoded::byte_serialize(value.as_bytes()).collect()
}

/// Anything wrong with the code/PKCE/resource parameters, or nothing.
fn params_error(state: &OauthState, params: &AuthorizeParams) -> Option<&'static str> {
    // A well-formed S256 challenge is the base64url of a SHA-256 digest:
    // always exactly 43 unpadded characters. Pinning the length (rather
    // than merely non-empty) rejects a malformed challenge here, at
    // consent, with a clear `invalid_request` instead of letting it slip
    // through to a puzzling `invalid_grant` at the token endpoint.
    if params.response_type != "code"
        || params.code_challenge.len() != 43
        || params.code_challenge_method != "S256"
    {
        return Some("invalid_request");
    }
    if !params.resource.is_empty() && params.resource != state.oauth.resource() {
        return Some("invalid_target");
    }
    None
}

/// The consent page: names the client and the delegation, and asks for
/// an existing API key — possession of a key IS the login here.
async fn consent_page(
    State(state): State<OauthState>,
    Query(params): Query<AuthorizeParams>,
) -> Response {
    let client = match checked_redirect(&state, &params) {
        Ok(client) => client,
        Err(refusal) => return *refusal,
    };
    if let Some(error) = params_error(&state, &params) {
        return error_redirect(&params, error);
    }
    consent_form(&client.client_name, &params, None)
}

/// The consent decision: a valid key issues the code and sends the
/// browser back; an invalid one re-asks without leaking which part
/// was wrong.
async fn approve(
    State(state): State<OauthState>,
    PeerAddr(peer_addr): PeerAddr,
    Form(params): Form<AuthorizeParams>,
) -> Response {
    let client = match checked_redirect(&state, &params) {
        Ok(client) => client,
        Err(refusal) => return *refusal,
    };
    if let Some(error) = params_error(&state, &params) {
        return error_redirect(&params, error);
    }
    let Some(key) = state.keyring.load().authenticate(&params.key) else {
        // Same throttle the bearer gate applies to a wrong token: this
        // path runs the identical constant-time keyring scan, so a wrong
        // key spends the per-source-IP failure budget and, once over it,
        // gets a 429 instead of another guess. A correct key returns
        // below without ever touching the limiter, so a delegating
        // operator is never throttled — only a brute-forcer is.
        if !state.fail_limiter.is_disabled() {
            let peer = peer_addr
                .map(|addr| Arc::<str>::from(addr.ip().to_string().as_str()))
                .unwrap_or_else(|| Arc::from("peer:unknown"));
            if let Err(retry_after) = state.fail_limiter.admit(&peer, Instant::now()) {
                return too_many_attempts(retry_after);
            }
        }
        return consent_form(
            &client.client_name,
            &params,
            Some("that key was not accepted"),
        );
    };
    let code = state.oauth.issue_code(
        &client,
        &params.redirect_uri,
        &params.code_challenge,
        &key,
        now_secs(),
    );
    let mut target = format!(
        "{}{}code={code}",
        params.redirect_uri,
        query_separator(&params.redirect_uri)
    );
    if !params.state.is_empty() {
        target.push_str(&format!("&state={}", encode_query_value(&params.state)));
    }
    Redirect::to(&target).into_response()
}

/// Browser hardening for every credential-collecting or
/// OAuth-parameter-echoing HTML page the server renders: no framing
/// (a password form is the classic clickjacking target), no scripts
/// or external loads (these pages carry only inline style), forms post
/// to this origin alone, and the URL — which holds the OAuth state —
/// never rides a Referer out. Applied to the consent form AND to the
/// "cannot continue" refusal page, since both are reachable
/// unauthenticated at the same public /oauth/authorize entry point.
fn hardened_html(status: StatusCode, body: String) -> Response {
    let mut response = (status, Html(body)).into_response();
    let headers = response.headers_mut();
    headers.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(
            "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; \
             frame-ancestors 'none'; base-uri 'none'",
        ),
    );
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    response
}

/// The consent POST's answer once the per-source-IP failure budget is
/// spent: a hardened 429 carrying the same `Retry-After` the bearer gate
/// sends on the identical throttle. Deliberately says nothing about which
/// key or how many tries remain — it is the same page for every caller.
fn too_many_attempts(retry_after: u64) -> Response {
    let mut response = hardened_html(
        StatusCode::TOO_MANY_REQUESTS,
        "<!doctype html><meta charset=\"utf-8\">\
         <title>Taguru — too many attempts</title>\
         <p>Too many failed attempts from your address. Try again shortly.</p>"
            .to_string(),
    );
    response
        .headers_mut()
        .insert(header::RETRY_AFTER, HeaderValue::from(retry_after));
    response
}

fn consent_form(client_name: &str, params: &AuthorizeParams, error: Option<&str>) -> Response {
    let hidden = |name: &str, value: &str| {
        format!(
            "<input type=\"hidden\" name=\"{name}\" value=\"{}\">",
            escape_html(value)
        )
    };
    let notice = error
        .map(|error| format!("<p class=\"err\">{}</p>", escape_html(error)))
        .unwrap_or_default();
    let page = format!(
        "<!doctype html><meta charset=\"utf-8\">\
         <title>Taguru — authorize</title>\
         <style>body{{font:16px/1.6 system-ui;max-width:34em;margin:4em auto;padding:0 1em}}\
         .err{{color:#b00020}}input[type=password]{{width:100%;font-size:1em;padding:.4em}}\
         code{{word-break:break-all}}\
         .detail{{background:#f4f4f4;padding:.6em .8em;border-radius:.3em}}\
         button{{font-size:1em;padding:.4em 1.2em;margin-top:.8em}}</style>\
         <h1>Authorize “{client}”?</h1>\
         <p><strong>{client}</strong> asks to use this Taguru server's memory \
         through MCP (<code>/mcp</code> only). Approving delegates one of your \
         API keys: the connection acts as that key, under the name \
         <code>&lt;key&gt;@{client}</code> in the access log.</p>\
         <p class=\"detail\">Approving sends a one-time authorization code to this \
         address — approve only if you recognize it:<br><code>{ruri_text}</code><br>\
         Client ID: <code>{cid_text}</code></p>\
         {notice}\
         <form method=\"post\" action=\"/oauth/authorize\">\
         {rt}{cid}{ruri}{st}{cc}{ccm}{res}\
         <label>Paste an API key to delegate:<br>\
         <input type=\"password\" name=\"key\" autofocus autocomplete=\"off\"></label><br>\
         <button type=\"submit\">Approve</button>\
         </form>",
        client = escape_html(client_name),
        ruri_text = escape_html(&params.redirect_uri),
        cid_text = escape_html(&params.client_id),
        notice = notice,
        rt = hidden("response_type", &params.response_type),
        cid = hidden("client_id", &params.client_id),
        ruri = hidden("redirect_uri", &params.redirect_uri),
        st = hidden("state", &params.state),
        cc = hidden("code_challenge", &params.code_challenge),
        ccm = hidden("code_challenge_method", &params.code_challenge_method),
        res = hidden("resource", &params.resource),
    );
    hardened_html(StatusCode::OK, page)
}

#[derive(Deserialize)]
struct TokenParams {
    #[serde(default)]
    grant_type: String,
    #[serde(default)]
    client_id: String,
    #[serde(default)]
    code: String,
    #[serde(default)]
    code_verifier: String,
    #[serde(default)]
    redirect_uri: String,
    #[serde(default)]
    refresh_token: String,
}

/// RFC 6749 token endpoint, form-encoded in, JSON out; refusals use
/// the standard error codes with a 400.
async fn token(State(state): State<OauthState>, Form(params): Form<TokenParams>) -> Response {
    let now = now_secs();
    // Both grant paths mint a token, which serializes and fsyncs the
    // OAuth store (via `persist`); wrap each in `block_in_place` so the
    // synchronous write never stalls the async worker.
    let outcome = match params.grant_type.as_str() {
        "authorization_code" => tokio::task::block_in_place(|| {
            state.oauth.exchange_code(
                &params.client_id,
                &params.code,
                &params.code_verifier,
                &params.redirect_uri,
                now,
            )
        }),
        "refresh_token" => tokio::task::block_in_place(|| {
            state
                .oauth
                .exchange_refresh(&params.client_id, &params.refresh_token, now)
        }),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "unsupported_grant_type" })),
            )
                .into_response();
        }
    };
    match outcome {
        Ok(grant) => Json(json!({
            "access_token": grant.access_token,
            "token_type": "Bearer",
            "expires_in": grant.expires_in,
            "refresh_token": grant.refresh_token,
        }))
        .into_response(),
        Err(error) => (StatusCode::BAD_REQUEST, Json(json!({ "error": error.0 }))).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::header::LOCATION;

    fn params_with(redirect_uri: &str, state: &str) -> AuthorizeParams {
        serde_json::from_value(json!({
            "redirect_uri": redirect_uri,
            "state": state,
        }))
        .unwrap()
    }

    fn location_of(response: Response) -> String {
        response
            .headers()
            .get(LOCATION)
            .expect("a redirect carries a Location header")
            .to_str()
            .unwrap()
            .to_string()
    }

    #[test]
    fn query_separator_opens_or_extends_the_redirect_query() {
        assert_eq!(query_separator("https://app/cb"), '?');
        assert_eq!(query_separator("https://app/cb?tenant=x"), '&');
    }

    /// Every OAuth HTML page — the credential-collecting consent form
    /// AND the "cannot continue" refusal (both reachable unauthenticated
    /// at /oauth/authorize) — must refuse framing (clickjacking on a
    /// password form), lock sources to its own inline style, and keep
    /// the OAuth state out of Referer headers.
    #[test]
    fn every_oauth_html_page_carries_browser_hardening_headers() {
        let params = params_with("https://app/cb", "s");
        let assert_hardened = |response: &Response, label: &str| {
            let headers = response.headers();
            assert_eq!(headers[header::X_FRAME_OPTIONS], "DENY", "{label}");
            let csp = headers[header::CONTENT_SECURITY_POLICY].to_str().unwrap();
            assert!(csp.contains("default-src 'none'"), "{label}: {csp}");
            assert!(csp.contains("frame-ancestors 'none'"), "{label}: {csp}");
            assert!(csp.contains("form-action 'self'"), "{label}: {csp}");
            assert_eq!(headers[header::REFERRER_POLICY], "no-referrer", "{label}");
        };
        assert_hardened(&consent_form("claude", &params, None), "consent form");
        // The refusal page goes through the same helper via checked_redirect;
        // exercise the helper directly to prove the shared guarantee.
        assert_hardened(
            &hardened_html(
                StatusCode::BAD_REQUEST,
                "<h1>Cannot continue</h1>".to_string(),
            ),
            "refusal page",
        );
    }

    /// A redirect_uri that already carries a query gets the OAuth
    /// parameters `&`-joined, not a second `?` that would fold `error`
    /// and `state` into the existing parameter's value (RFC 6749 §3.1.2).
    #[test]
    fn error_redirect_extends_an_existing_query_with_ampersand() {
        let params = params_with("https://app/cb?tenant=x", "s p");
        let location = location_of(error_redirect(&params, "invalid_request"));
        assert_eq!(
            location,
            "https://app/cb?tenant=x&error=invalid_request&state=s+p"
        );
    }

    /// A bare redirect_uri still opens its query with `?`.
    #[test]
    fn error_redirect_opens_a_bare_query_with_question_mark() {
        let params = params_with("https://app/cb", "");
        let location = location_of(error_redirect(&params, "invalid_request"));
        assert_eq!(location, "https://app/cb?error=invalid_request");
    }

    /// The consent page renders the redirect target VISIBLY (a `<code>`
    /// block in the `.detail` panel), not just tucked in a hidden field:
    /// a delegating operator can catch a spoofed redirect_uri before
    /// handing over a key. The value is HTML-escaped on the way in.
    #[tokio::test]
    async fn the_consent_form_shows_the_redirect_target_where_a_human_sees_it() {
        let params = params_with("https://claude.ai/cb?x=1&y=2", "");
        let response = consent_form("claude", &params, None);
        let bytes = axum::body::to_bytes(response.into_body(), 65536)
            .await
            .unwrap();
        let page = std::str::from_utf8(&bytes).unwrap();
        assert!(page.contains("class=\"detail\""), "{page}");
        // Rendered in a visible <code> (escaped), distinct from the
        // hidden input's value="..." carrying the same string.
        assert!(
            page.contains("<code>https://claude.ai/cb?x=1&amp;y=2</code>"),
            "{page}"
        );
    }

    /// The consent POST runs the keyring's constant-time scan
    /// unauthenticated, so a wrong key spends the same per-source-IP
    /// failure budget the bearer gate uses: the first miss re-asks (200),
    /// but once the budget is gone the page answers 429 + Retry-After
    /// instead of grading another guess. A correct key never touches the
    /// limiter, so a delegating operator is not locked out even after the
    /// budget is spent. Without this the form is a free brute-force /
    /// CPU-burn oracle against the keyring whenever the optional per-key
    /// rate limit is off (its default).
    #[tokio::test]
    async fn a_wrong_consent_key_spends_the_failed_auth_budget() {
        let dir = std::env::temp_dir().join(format!(
            "taguru-oauth-consent-throttle-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let oauth = crate::oauth::Oauth::open("https://memory.example", &dir);
        let client = oauth
            .register_client("claude", vec!["https://claude.ai/cb".to_string()])
            .unwrap();
        let state = OauthState {
            oauth: Arc::new(oauth),
            keyring: SharedKeyring::new(
                crate::auth::Keyring::parse(Some("tg_correct".to_string()), None).unwrap(),
            ),
            // One failed attempt per minute: the second wrong key in the
            // same minute is already over budget.
            fail_limiter: Arc::new(crate::limits::RateLimiter::new(1)),
        };
        let authorize = |key: &str| -> AuthorizeParams {
            serde_json::from_value(json!({
                "response_type": "code",
                "client_id": client.client_id.clone(),
                "redirect_uri": "https://claude.ai/cb",
                // Any 43-char string clears the S256-shape check; the key
                // is what this test turns away.
                "code_challenge": "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM",
                "code_challenge_method": "S256",
                "key": key,
            }))
            .unwrap()
        };

        // First wrong key: rejected but under budget, so the form comes
        // back (200) for an honest retry.
        let first = approve(
            State(state.clone()),
            PeerAddr(None),
            Form(authorize("nope-1")),
        )
        .await;
        assert_eq!(first.status(), StatusCode::OK);

        // Second wrong key in the same minute: over the 1/min budget →
        // 429 with Retry-After, exactly like the bearer gate.
        let second = approve(
            State(state.clone()),
            PeerAddr(None),
            Form(authorize("nope-2")),
        )
        .await;
        assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(second.headers().contains_key(header::RETRY_AFTER));

        // The correct key is never throttled: it returns before the
        // limiter is consulted, so even now — failure budget spent — it
        // still issues the code and redirects (303), never a 429.
        let ok = approve(
            State(state.clone()),
            PeerAddr(None),
            Form(authorize("tg_correct")),
        )
        .await;
        assert_eq!(ok.status(), StatusCode::SEE_OTHER);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
