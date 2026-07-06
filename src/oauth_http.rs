//! The HTTP face of the embedded OAuth server: the two well-known
//! documents, dynamic registration, the consent page, and the token
//! endpoint. All of it is exempt from bearer auth (these endpoints
//! exist to CREATE credentials) and rate-limited under the anonymous
//! bucket like any unauthenticated caller.

use std::sync::Arc;

use axum::extract::{Form, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::auth::Keyring;
use crate::oauth::{Oauth, escape_html, now_secs};

#[derive(Clone)]
pub struct OauthState {
    pub oauth: Arc<Oauth>,
    pub keyring: Arc<Keyring>,
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
    match state.oauth.register_client(client_name, redirect_uris) {
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

/// Validates the (client, redirect) pair — the two fields that decide
/// whether redirecting an error is safe at all. `Err` is a 400 page
/// (boxed: a Response outweighs the happy path), never a redirect to
/// an unverified target.
fn checked_redirect(
    state: &OauthState,
    params: &AuthorizeParams,
) -> Result<crate::oauth::Client, Box<Response>> {
    let refuse = |reason: &str| {
        Box::new(
            (
                StatusCode::BAD_REQUEST,
                Html(format!(
                    "<h1>Cannot continue</h1><p>{}</p>",
                    escape_html(reason)
                )),
            )
                .into_response(),
        )
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
    let mut target = format!("{}?error={error}", params.redirect_uri);
    if !params.state.is_empty() {
        target.push_str(&format!("&state={}", params.state));
    }
    Redirect::to(&target).into_response()
}

/// Anything wrong with the code/PKCE/resource parameters, or nothing.
fn params_error(state: &OauthState, params: &AuthorizeParams) -> Option<&'static str> {
    if params.response_type != "code"
        || params.code_challenge.is_empty()
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
async fn approve(State(state): State<OauthState>, Form(params): Form<AuthorizeParams>) -> Response {
    let client = match checked_redirect(&state, &params) {
        Ok(client) => client,
        Err(refusal) => return *refusal,
    };
    if let Some(error) = params_error(&state, &params) {
        return error_redirect(&params, error);
    }
    let Some(key) = state.keyring.authenticate(&params.key) else {
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
    let mut target = format!("{}?code={code}", params.redirect_uri);
    if !params.state.is_empty() {
        target.push_str(&format!("&state={}", params.state));
    }
    Redirect::to(&target).into_response()
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
         button{{font-size:1em;padding:.4em 1.2em;margin-top:.8em}}</style>\
         <h1>Authorize “{client}”?</h1>\
         <p><strong>{client}</strong> asks to use this Taguru server's memory \
         through MCP (<code>/mcp</code> only). Approving delegates one of your \
         API keys: the connection acts as that key, under the name \
         <code>&lt;key&gt;@{client}</code> in the access log.</p>\
         {notice}\
         <form method=\"post\" action=\"/oauth/authorize\">\
         {rt}{cid}{ruri}{st}{cc}{ccm}{res}\
         <label>Paste an API key to delegate:<br>\
         <input type=\"password\" name=\"key\" autofocus autocomplete=\"off\"></label><br>\
         <button type=\"submit\">Approve</button>\
         </form>",
        client = escape_html(client_name),
        notice = notice,
        rt = hidden("response_type", &params.response_type),
        cid = hidden("client_id", &params.client_id),
        ruri = hidden("redirect_uri", &params.redirect_uri),
        st = hidden("state", &params.state),
        cc = hidden("code_challenge", &params.code_challenge),
        ccm = hidden("code_challenge_method", &params.code_challenge_method),
        res = hidden("resource", &params.resource),
    );
    Html(page).into_response()
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
    let outcome = match params.grant_type.as_str() {
        "authorization_code" => state.oauth.exchange_code(
            &params.client_id,
            &params.code,
            &params.code_verifier,
            &params.redirect_uri,
            now,
        ),
        "refresh_token" => {
            state
                .oauth
                .exchange_refresh(&params.client_id, &params.refresh_token, now)
        }
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
