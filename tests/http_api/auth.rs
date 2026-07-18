//! Bearer-token and OAuth-delegation coverage for every route.

use serde_json::json;

use crate::support::*;

/// Named keys (TAGURU_API_TOKENS) authenticate alongside the classic
/// single token, so one client's key can rotate or die alone.
#[test]
fn named_api_tokens_authenticate_alongside_the_default() {
    let server = Server::start_with_env(
        "keyring",
        &[
            ("TAGURU_API_TOKEN", "legacy"),
            ("TAGURU_API_TOKENS", "ci:tok-a,laptop:tok-b"),
        ],
    );

    let (status, _) = server.call("GET", "/contexts", None);
    assert_eq!(status, 401);
    for token in ["tok-a", "tok-b", "legacy"] {
        let (status, _) = server.call_with_token("GET", "/contexts", None, Some(token));
        assert_eq!(status, 200, "token {token} must authenticate");
    }
    let (status, _) = server.call_with_token("GET", "/contexts", None, Some("tok-c"));
    assert_eq!(status, 401);
}

/// The full OAuth dance a claude.ai custom connector performs:
/// discovery → dynamic registration → consent (the operator delegates
/// an existing key) → PKCE code exchange → tokens that open /mcp and
/// nothing else → refresh rotation. PKCE material is the RFC 7636
/// appendix B vector, so no hashing happens in the test.
#[test]
fn oauth_flow_delegates_a_key_to_a_remote_mcp_client() {
    const VERIFIER: &str = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
    const CHALLENGE: &str = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM";
    const CALLBACK: &str = "https%3A%2F%2Fclaude.ai%2Fapi%2Fmcp%2Fauth_callback";

    let server = Server::start_with_env(
        "oauth",
        &[
            ("TAGURU_API_TOKENS", "laptop:tok-op"),
            ("TAGURU_PUBLIC_URL", "http://taguru.test"),
        ],
    );
    let no_redirect = no_redirect_agent();

    // An anonymous /mcp names the discovery document (RFC 9728)...
    let response = no_redirect
        .post(&format!("{}/mcp", server.base))
        .header("Content-Type", "application/json")
        .send("{}")
        .expect("anonymous /mcp must answer");
    assert_eq!(response.status(), 401, "anonymous /mcp must be a 401");
    let www_authenticate = header_text(&response, "www-authenticate");
    assert!(
        !www_authenticate.is_empty(),
        "401 must carry WWW-Authenticate"
    );
    assert!(
        www_authenticate.contains(
            "resource_metadata=\"http://taguru.test/.well-known/oauth-protected-resource\""
        ),
        "{www_authenticate}"
    );

    // ...and both metadata spellings answer without credentials.
    let (status, resource_metadata) =
        server.call("GET", "/.well-known/oauth-protected-resource", None);
    assert_eq!(status, 200);
    assert_eq!(resource_metadata["resource"], "http://taguru.test/mcp");
    let (_, path_inserted) = server.call("GET", "/.well-known/oauth-protected-resource/mcp", None);
    assert_eq!(path_inserted["resource"], "http://taguru.test/mcp");
    let (_, authorization_server) =
        server.call("GET", "/.well-known/oauth-authorization-server", None);
    assert_eq!(authorization_server["issuer"], "http://taguru.test");
    assert_eq!(
        authorization_server["code_challenge_methods_supported"],
        json!(["S256"])
    );

    // Dynamic registration (RFC 7591).
    let (status, registered) = server.call(
        "POST",
        "/oauth/register",
        Some(json!({
            "client_name": "claude",
            "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
        })),
    );
    assert_eq!(status, 201);
    let client_id = registered["client_id"].as_str().unwrap().to_string();

    // The consent page names the client; a wrong key re-asks (200, no
    // redirect, no code).
    let authorize_query = format!(
        "response_type=code&client_id={client_id}&redirect_uri={CALLBACK}\
         &state=st4te&code_challenge={CHALLENGE}&code_challenge_method=S256"
    );
    let (status, consent) =
        server.call("GET", &format!("/oauth/authorize?{authorize_query}"), None);
    assert_eq!(status, 200);
    assert!(consent.as_str().unwrap().contains("claude"));
    let (status, retry) = server.call_raw(
        "POST",
        "/oauth/authorize",
        Some(&format!("{authorize_query}&key=wrong")),
        Some("application/x-www-form-urlencoded"),
    );
    assert_eq!(status, 200);
    assert!(retry.as_str().unwrap().contains("not accepted"));

    // The real key approves: back to the callback with code and state.
    let approved = no_redirect
        .post(&format!("{}/oauth/authorize", server.base))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .send(format!("{authorize_query}&key=tok-op"))
        .expect("approval must answer with a redirect");
    assert_eq!(approved.status(), 303);
    let location = header_text(&approved, "location");
    assert!(
        location.starts_with("https://claude.ai/api/mcp/auth_callback?code="),
        "{location}"
    );
    assert!(location.contains("state=st4te"), "{location}");
    let code = location
        .split("code=")
        .nth(1)
        .unwrap()
        .split('&')
        .next()
        .unwrap()
        .to_string();

    // PKCE exchange mints the Bearer pair.
    let (status, minted) = server.call_raw(
        "POST",
        "/oauth/token",
        Some(&format!(
            "grant_type=authorization_code&client_id={client_id}&code={code}\
             &code_verifier={VERIFIER}&redirect_uri={CALLBACK}"
        )),
        Some("application/x-www-form-urlencoded"),
    );
    assert_eq!(status, 200, "{minted}");
    assert_eq!(minted["token_type"], "Bearer");
    let access = minted["access_token"].as_str().unwrap().to_string();
    let refresh = minted["refresh_token"].as_str().unwrap().to_string();

    // The delegation opens /mcp...
    let ping = json!({"jsonrpc": "2.0", "id": 1, "method": "ping"});
    let (status, _) = server.call_with_token("POST", "/mcp", Some(ping.clone()), Some(&access));
    assert_eq!(status, 200);
    // ...and nothing else — the raw API stays key-only.
    let (status, _) = server.call_with_token("GET", "/contexts", None, Some(&access));
    assert_eq!(status, 401);
    let (status, _) = server.call_with_token("GET", "/contexts", None, Some("tok-op"));
    assert_eq!(status, 200);

    // The code is single-use.
    let (status, replayed) = server.call_raw(
        "POST",
        "/oauth/token",
        Some(&format!(
            "grant_type=authorization_code&client_id={client_id}&code={code}\
             &code_verifier={VERIFIER}&redirect_uri={CALLBACK}"
        )),
        Some("application/x-www-form-urlencoded"),
    );
    assert_eq!(
        (status, replayed["error"].as_str()),
        (400, Some("invalid_grant"))
    );

    // Refresh rotates: the new pair works, the presented token is dead.
    let (status, rotated) = server.call_raw(
        "POST",
        "/oauth/token",
        Some(&format!(
            "grant_type=refresh_token&client_id={client_id}&refresh_token={refresh}"
        )),
        Some("application/x-www-form-urlencoded"),
    );
    assert_eq!(status, 200);
    let fresh_access = rotated["access_token"].as_str().unwrap().to_string();
    let (status, _) = server.call_with_token("POST", "/mcp", Some(ping), Some(&fresh_access));
    assert_eq!(status, 200);
    let (status, dead) = server.call_raw(
        "POST",
        "/oauth/token",
        Some(&format!(
            "grant_type=refresh_token&client_id={client_id}&refresh_token={refresh}"
        )),
        Some("application/x-www-form-urlencoded"),
    );
    assert_eq!(
        (status, dead["error"].as_str()),
        (400, Some("invalid_grant"))
    );
}

/// `state` is opaque client data (RFC 6749 §4.1.2) that must round-trip
/// verbatim — but it is attacker-controlled, and the authorize endpoint
/// splices it straight into the redirect's query string. A `state` of
/// `evil&code=injected` must come back percent-encoded as one opaque
/// value, never as a raw `&code=injected` that opens a second `code`
/// parameter ahead of the real one.
#[test]
fn oauth_authorize_percent_encodes_state_in_redirects() {
    const CHALLENGE: &str = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM";
    const CALLBACK: &str = "https%3A%2F%2Fclaude.ai%2Fapi%2Fmcp%2Fauth_callback";
    // "evil&code=injected", pre-escaped so the test's own request line
    // carries it as a single `state` field rather than splitting it.
    const INJECTED_STATE: &str = "evil%26code%3Dinjected";

    let server = Server::start_with_env(
        "oauth-state-escape",
        &[
            ("TAGURU_API_TOKENS", "laptop:tok-op"),
            ("TAGURU_PUBLIC_URL", "http://taguru.test"),
        ],
    );
    let no_redirect = no_redirect_agent();

    let (_, registered) = server.call(
        "POST",
        "/oauth/register",
        Some(json!({
            "client_name": "claude",
            "redirect_uris": ["https://claude.ai/api/mcp/auth_callback"],
        })),
    );
    let client_id = registered["client_id"].as_str().unwrap().to_string();

    let authorize_query = format!(
        "response_type=code&client_id={client_id}&redirect_uri={CALLBACK}\
         &state={INJECTED_STATE}&code_challenge={CHALLENGE}&code_challenge_method=S256"
    );

    // The success redirect (consent approved)...
    let approved = no_redirect
        .post(&format!("{}/oauth/authorize", server.base))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .send(format!("{authorize_query}&key=tok-op"))
        .expect("approval must answer with a redirect");
    assert_eq!(approved.status(), 303);
    let location = header_text(&approved, "location");
    assert_eq!(
        location.matches("code=").count(),
        1,
        "state must not inject a second query parameter: {location}"
    );
    assert!(
        location.contains("state=evil%26code%3Dinjected"),
        "state must round-trip percent-encoded, not splice raw query syntax: {location}"
    );

    // ...and the error redirect (a bad code_challenge_method) both take
    // the same splicing path in oauth_http.rs — consent_page and approve
    // both run params_error before rendering anything, so GET redirects
    // just like POST. Use no_redirect here too: server.call's default
    // agent follows redirects, which would leak this request to the
    // real external host named in redirect_uri (claude.ai).
    let bad_query = format!(
        "response_type=code&client_id={client_id}&redirect_uri={CALLBACK}\
         &state={INJECTED_STATE}&code_challenge={CHALLENGE}&code_challenge_method=plain"
    );
    let refused_get = no_redirect
        .get(&format!("{}/oauth/authorize?{bad_query}", server.base))
        .call()
        .expect("a bad code_challenge_method must still redirect with ?error=");
    assert_eq!(refused_get.status(), 303);
    let get_error_location = header_text(&refused_get, "location");
    assert_eq!(
        get_error_location.matches("error=").count(),
        1,
        "state must not inject a second query parameter: {get_error_location}"
    );
    assert!(
        get_error_location.contains("state=evil%26code%3Dinjected"),
        "state must round-trip percent-encoded, not splice raw query syntax: {get_error_location}"
    );
    let refused = no_redirect
        .post(&format!("{}/oauth/authorize", server.base))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .send(format!("{bad_query}&key=tok-op"))
        .expect("a bad code_challenge_method must still redirect with ?error=");
    assert_eq!(refused.status(), 303);
    let error_location = header_text(&refused, "location");
    assert_eq!(
        error_location.matches("error=").count(),
        1,
        "state must not inject a second query parameter: {error_location}"
    );
    assert!(
        error_location.contains("state=evil%26code%3Dinjected"),
        "state must round-trip percent-encoded, not splice raw query syntax: {error_location}"
    );
}

/// HTTP-layer refusals the grant-store's own tests (oauth.rs) cannot
/// reach, since they never go through `oauth_http.rs` at all: bad
/// registration metadata, an authorize call whose (client, redirect)
/// pair is not verified — a 400 page, never a redirect, since the
/// target itself is unproven — and, once that pair IS verified, each
/// individual parameter mistake landing back on the client as its own
/// distinct `?error=...` redirect. Plus the token endpoint's fallback
/// arm for a grant_type it does not speak.
#[test]
fn oauth_http_layer_refuses_bad_client_redirect_and_grant_parameters() {
    let server = Server::start_with_env(
        "oauth-errors",
        &[
            ("TAGURU_API_TOKENS", "op:tok-op"),
            ("TAGURU_PUBLIC_URL", "http://taguru.test"),
        ],
    );

    // register(): bad metadata refuses before anything is stored.
    let (status, empty_redirects) = server.call(
        "POST",
        "/oauth/register",
        Some(json!({"client_name": "c", "redirect_uris": []})),
    );
    assert_eq!(status, 400);
    assert_eq!(empty_redirects["error"], "invalid_client_metadata");

    let (status, insecure_redirect) = server.call(
        "POST",
        "/oauth/register",
        Some(json!({"client_name": "c", "redirect_uris": ["http://evil.example/cb"]})),
    );
    assert_eq!(status, 400);
    assert_eq!(insecure_redirect["error"], "invalid_client_metadata");
    assert!(
        insecure_redirect["error_description"]
            .as_str()
            .unwrap()
            .contains("https or loopback"),
        "{insecure_redirect}"
    );

    // A real client, to exercise the checks that only apply once
    // registration itself is not in question.
    let (status, registered) = server.call(
        "POST",
        "/oauth/register",
        Some(json!({
            "client_name": "claude",
            "redirect_uris": ["https://claude.ai/cb"],
        })),
    );
    assert_eq!(status, 201);
    let client_id = registered["client_id"].as_str().unwrap().to_string();
    const REDIRECT: &str = "https%3A%2F%2Fclaude.ai%2Fcb";

    // checked_redirect refusals: a 400 page, never a redirect.
    let (status, page) = server.call(
        "GET",
        &format!(
            "/oauth/authorize?response_type=code&client_id=nonesuch&redirect_uri={REDIRECT}\
             &code_challenge=abc&code_challenge_method=S256"
        ),
        None,
    );
    assert_eq!(status, 400);
    assert!(
        page.as_str().unwrap().contains("unknown client_id"),
        "{page}"
    );

    let (status, page) = server.call(
        "GET",
        &format!(
            "/oauth/authorize?response_type=code&client_id={client_id}&code_challenge=abc\
             &code_challenge_method=S256&redirect_uri=https%3A%2F%2Fevil.example%2Fcb"
        ),
        None,
    );
    assert_eq!(status, 400);
    assert!(
        page.as_str()
            .unwrap()
            .contains("redirect_uri is not registered"),
        "{page}"
    );

    // Past that check, each individual parameter mistake comes back as
    // its own OAuth error redirect to the (now verified) client.
    let no_redirect = no_redirect_agent();
    let authorize = |query: &str| -> (u16, String) {
        match no_redirect
            .get(&format!("{}/oauth/authorize?{query}", server.base))
            .call()
        {
            Ok(response) => (
                response.status().as_u16(),
                header_text(&response, "location"),
            ),
            Err(error) => panic!("GET /oauth/authorize failed: {error}"),
        }
    };

    // response_type must be "code".
    let (status, location) = authorize(&format!(
        "response_type=token&client_id={client_id}&redirect_uri={REDIRECT}&state=s1\
         &code_challenge=abc&code_challenge_method=S256"
    ));
    assert_eq!(status, 303);
    assert!(
        location.starts_with("https://claude.ai/cb?error=invalid_request"),
        "{location}"
    );
    assert!(location.contains("state=s1"), "{location}");

    // code_challenge must be present.
    let (status, location) = authorize(&format!(
        "response_type=code&client_id={client_id}&redirect_uri={REDIRECT}&state=s2\
         &code_challenge_method=S256"
    ));
    assert_eq!(status, 303);
    assert!(
        location.starts_with("https://claude.ai/cb?error=invalid_request"),
        "{location}"
    );

    // code_challenge_method must be S256 — "plain" is not offered.
    let (status, location) = authorize(&format!(
        "response_type=code&client_id={client_id}&redirect_uri={REDIRECT}&state=s3\
         &code_challenge=abc&code_challenge_method=plain"
    ));
    assert_eq!(status, 303);
    assert!(
        location.starts_with("https://claude.ai/cb?error=invalid_request"),
        "{location}"
    );

    // A present-but-malformed challenge is refused just like an absent
    // one: an S256 challenge is always exactly 43 base64url chars, so a
    // short "abc" can match no verifier and is rejected here at consent.
    let (status, location) = authorize(&format!(
        "response_type=code&client_id={client_id}&redirect_uri={REDIRECT}&state=s3b\
         &code_challenge=abc&code_challenge_method=S256"
    ));
    assert_eq!(status, 303);
    assert!(
        location.starts_with("https://claude.ai/cb?error=invalid_request"),
        "{location}"
    );

    // A resource that does not name this server's own /mcp is a
    // distinct failure: invalid_target, not invalid_request — so this
    // case needs a well-formed challenge (a real S256 challenge is
    // always 43 chars) to clear the PKCE check and reach the resource
    // check at all.
    let challenge = "a".repeat(43);
    let (status, location) = authorize(&format!(
        "response_type=code&client_id={client_id}&redirect_uri={REDIRECT}&state=s4\
         &code_challenge={challenge}&code_challenge_method=S256&resource=https%3A%2F%2Fother.example%2Fmcp"
    ));
    assert_eq!(status, 303);
    assert!(
        location.starts_with("https://claude.ai/cb?error=invalid_target"),
        "{location}"
    );
    assert!(location.contains("state=s4"), "{location}");

    // token(): a grant_type this server does not speak.
    let (status, unsupported) = server.call_raw(
        "POST",
        "/oauth/token",
        Some(&format!("grant_type=password&client_id={client_id}")),
        Some("application/x-www-form-urlencoded"),
    );
    assert_eq!(status, 400);
    assert_eq!(unsupported["error"], "unsupported_grant_type");
}

/// The request budget is spent per key: one key exhausts alone while
/// the other keeps working, and the probes stay exempt.
#[test]
fn rate_limit_is_per_key_and_spares_probes() {
    let server = Server::start_with_env(
        "ratelimit",
        &[
            ("TAGURU_API_TOKENS", "hot:tok-hot,calm:tok-calm"),
            ("TAGURU_RATE_LIMIT_PER_MIN", "2"),
        ],
    );

    for _ in 0..2 {
        let (status, _) = server.call_with_token("GET", "/contexts", None, Some("tok-hot"));
        assert_eq!(status, 200);
    }
    let (status, refused) = server.call_with_token("GET", "/contexts", None, Some("tok-hot"));
    assert_eq!(status, 429);
    assert_eq!(refused["status"], "error");
    assert!(
        refused["error"].as_str().unwrap().contains("budget"),
        "{refused}"
    );

    let (status, _) = server.call_with_token("GET", "/contexts", None, Some("tok-calm"));
    assert_eq!(status, 200, "an untouched key must keep its own budget");
    let (status, _) = server.call("GET", "/health", None);
    assert_eq!(status, 200, "probes must not starve behind a hot key");
}

#[test]
fn bearer_token_gates_every_route_except_health_and_metrics() {
    let server = Server::start_with_env("auth", &[("TAGURU_API_TOKEN", "s3cret")]);

    // Liveness and the scrape answer with zero credentials.
    assert_eq!(server.call("GET", "/health", None).0, 200);
    assert_eq!(server.call("GET", "/metrics", None).0, 200);

    // Everything else refuses a missing or wrong token with the API's
    // own error shape, and accepts the right one.
    let (status, body) = server.call("GET", "/contexts", None);
    assert_eq!(status, 401);
    assert_eq!(body["status"], json!("error"));
    let (status, _) = server.call_with_token("GET", "/contexts", None, Some("wrong"));
    assert_eq!(status, 401);
    let (status, _) = server.call_with_token("GET", "/contexts", None, Some("s3cret"));
    assert_eq!(status, 200);

    // Writes are gated the same way.
    let (status, _) = server.call("PUT", "/contexts/sake", Some(json!({})));
    assert_eq!(status, 401);
    let (status, _) =
        server.call_with_token("PUT", "/contexts/sake", Some(json!({})), Some("s3cret"));
    assert_eq!(status, 200);
}
