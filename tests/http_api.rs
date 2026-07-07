//! HTTP integration tests: the real binary, spawned on a free port with
//! a scratch data directory, driven through the same retrieval loop the
//! protocol documents. Everything here was once verified by hand with
//! curl; this pins it so handler wiring and response shapes cannot
//! regress silently.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

use serde_json::{Value, json};

/// One running server on its own port and data directory, killed and
/// cleaned up on drop.
struct Server {
    child: Child,
    base: String,
    data_dir: PathBuf,
}

impl Server {
    fn start(tag: &str) -> Self {
        Self::start_with_env(tag, &[])
    }

    fn start_with_env(tag: &str, extra_env: &[(&str, &str)]) -> Self {
        let data_dir =
            std::env::temp_dir().join(format!("taguru-http-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data_dir);
        Self::spawn(tag, data_dir, extra_env)
    }

    fn start_on(tag: &str, data_dir: PathBuf) -> Self {
        Self::spawn(tag, data_dir, &[])
    }

    fn spawn(tag: &str, data_dir: PathBuf, extra_env: &[(&str, &str)]) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
        command
            .env("TAGURU_ADDR", "127.0.0.1:0")
            .env("TAGURU_DATA_DIR", &data_dir)
            .env("TAGURU_FLUSH_SECS", "1")
            .env_remove("TAGURU_EMBED_URL") // lexical-only, hermetic
            .env_remove("TAGURU_EMBED_MODEL")
            .env_remove("TAGURU_EMBED_AUTO")
            .env_remove("TAGURU_EMBED_PASSAGES")
            .env_remove("TAGURU_PASSAGE_VECTOR_LIMIT")
            .env_remove("TAGURU_PASSAGES_WAL_MAX_BYTES")
            .env_remove("TAGURU_SEMANTIC_FLOOR")
            .env_remove("TAGURU_API_TOKEN") // unauthenticated unless a test opts in
            .env_remove("TAGURU_API_TOKENS")
            .env_remove("TAGURU_RATE_LIMIT_PER_MIN")
            .env_remove("TAGURU_PUBLIC_URL") // no OAuth unless a test opts in
            // No tracing unless a test opts in — a developer shell
            // with a live OTel setup must not flip every test here
            // into export mode.
            .env_remove("OTEL_EXPORTER_OTLP_ENDPOINT")
            .env_remove("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            .env_remove("OTEL_EXPORTER_OTLP_PROTOCOL")
            // Content-free logs unless a test opts in, whatever the
            // developer shell says.
            .env_remove("TAGURU_LOG_SEARCHES")
            // And no config file from the developer shell either.
            .env_remove("TAGURU_CONFIG");
        for (key, value) in extra_env {
            command.env(key, value);
        }
        let mut child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("server binary must spawn");

        // The server prints its resolved address; read until it appears.
        let stdout = child.stdout.take().expect("stdout must be piped");
        let mut lines = BufReader::new(stdout).lines();
        let base = loop {
            let line = lines
                .next()
                .unwrap_or_else(|| panic!("server '{tag}' exited before listening"))
                .expect("server stdout must be readable");
            if let Some(addr) = line.strip_prefix("listening on ") {
                break format!("http://{addr}");
            }
        };
        // Keep draining stdout so the server never blocks on a full pipe.
        std::thread::spawn(move || for _ in lines {});

        Self {
            child,
            base,
            data_dir,
        }
    }

    /// One request; returns (status, parsed body). Non-JSON bodies come
    /// back as JSON strings.
    fn call(&self, method: &str, path: &str, body: Option<Value>) -> (u16, Value) {
        self.call_with_token(method, path, body, None)
    }

    /// [`Server::call`] with an explicit bearer token attached.
    fn call_with_token(
        &self,
        method: &str,
        path: &str,
        body: Option<Value>,
        token: Option<&str>,
    ) -> (u16, Value) {
        let mut request = ureq::AgentBuilder::new()
            .build()
            .request(method, &format!("{}{path}", self.base));
        if let Some(token) = token {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
        let response = match body {
            Some(body) => request
                .set("Content-Type", "application/json")
                .send_string(&body.to_string()),
            None => request.call(),
        };
        finish(response, method, path)
    }

    /// A raw request: the body goes out as-is, with a Content-Type only
    /// when one is given — for the header-omission cases the JSON
    /// helpers cannot express.
    fn call_raw(
        &self,
        method: &str,
        path: &str,
        body: Option<&str>,
        content_type: Option<&str>,
    ) -> (u16, Value) {
        let mut request = ureq::AgentBuilder::new()
            .build()
            .request(method, &format!("{}{path}", self.base));
        if let Some(content_type) = content_type {
            request = request.set("Content-Type", content_type);
        }
        let response = match body {
            Some(body) => request.send_string(body),
            None => request.call(),
        };
        finish(response, method, path)
    }

    fn ok(&self, method: &str, path: &str, body: Option<Value>) -> Value {
        let (status, parsed) = self.call(method, path, body);
        assert_eq!(status, 200, "{method} {path} -> {parsed}");
        parsed["result"].clone()
    }

    /// Graceful stop (SIGTERM), waiting for the shutdown flush.
    fn stop_gracefully(self) -> PathBuf {
        self.stop_with("-TERM")
    }

    /// Hard kill (SIGKILL): no shutdown flush, no cleanup — whatever
    /// durability the server claims must come from the disk alone.
    fn stop_hard(self) -> PathBuf {
        self.stop_with("-KILL")
    }

    fn stop_with(mut self, signal: &str) -> PathBuf {
        let pid = self.child.id().to_string();
        Command::new("kill")
            .args([signal, &pid])
            .status()
            .expect("kill must run");
        let _ = self.child.wait();
        let data_dir = self.data_dir.clone();
        // Drop must not re-kill or delete the directory we hand back.
        std::mem::forget(self);
        data_dir
    }
}

/// Shared response tail: status plus parsed JSON body (or the raw
/// text when it is not JSON).
fn finish(response: Result<ureq::Response, ureq::Error>, method: &str, path: &str) -> (u16, Value) {
    let (status, text) = match response {
        Ok(response) => {
            let status = response.status();
            (status, response.into_string().unwrap_or_default())
        }
        Err(ureq::Error::Status(status, response)) => {
            (status, response.into_string().unwrap_or_default())
        }
        Err(error) => panic!("request {method} {path} failed: {error}"),
    };
    let parsed = serde_json::from_str(&text).unwrap_or(Value::String(text));
    (status, parsed)
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.data_dir);
    }
}

/// GET /protocol carries a live-configuration trailer once the semantic
/// tier is configured: an agent must learn from the manual itself that
/// `refresh_embeddings` is worth calling here (or already automatic) —
/// the static text alone leaves a configured tier dark.
#[test]
fn protocol_reports_the_semantic_tier_when_configured() {
    // /protocol never calls the provider, so a dead endpoint serves.
    let embed_env = [
        ("TAGURU_EMBED_URL", "http://127.0.0.1:9/v1/embeddings"),
        ("TAGURU_EMBED_MODEL", "proto-test-model"),
    ];
    let server = Server::start_with_env("proto-embed", &embed_env);
    let (status, protocol) = server.call("GET", "/protocol", None);
    assert_eq!(status, 200);
    let text = protocol.as_str().unwrap();
    assert!(text.contains("## This server"));
    assert!(text.contains("`proto-test-model`"));
    assert!(text.contains("calling `refresh_embeddings`"));
    assert!(!text.contains("auto-refreshes"));

    let mut auto_env = embed_env.to_vec();
    auto_env.push(("TAGURU_EMBED_AUTO", "1"));
    let server = Server::start_with_env("proto-auto", &auto_env);
    let (_, protocol) = server.call("GET", "/protocol", None);
    assert!(protocol.as_str().unwrap().contains("auto-refreshes"));
}

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
    let no_redirect = ureq::AgentBuilder::new().redirects(0).build();

    // An anonymous /mcp names the discovery document (RFC 9728)...
    let www_authenticate = match no_redirect
        .post(&format!("{}/mcp", server.base))
        .set("Content-Type", "application/json")
        .send_string("{}")
    {
        Err(ureq::Error::Status(401, response)) => response
            .header("www-authenticate")
            .expect("401 must carry WWW-Authenticate")
            .to_string(),
        other => panic!("anonymous /mcp must be a 401, got {other:?}"),
    };
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
        .set("Content-Type", "application/x-www-form-urlencoded")
        .send_string(&format!("{authorize_query}&key=tok-op"))
        .expect("approval must answer with a redirect");
    assert_eq!(approved.status(), 303);
    let location = approved.header("location").unwrap().to_string();
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
    let no_redirect = ureq::AgentBuilder::new().redirects(0).build();

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
        .set("Content-Type", "application/x-www-form-urlencoded")
        .send_string(&format!("{authorize_query}&key=tok-op"))
        .expect("approval must answer with a redirect");
    assert_eq!(approved.status(), 303);
    let location = approved.header("location").unwrap().to_string();
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
    // the same splicing path in oauth_http.rs.
    let bad_query = format!(
        "response_type=code&client_id={client_id}&redirect_uri={CALLBACK}\
         &state={INJECTED_STATE}&code_challenge={CHALLENGE}&code_challenge_method=plain"
    );
    let (status, _) = server.call("GET", &format!("/oauth/authorize?{bad_query}"), None);
    assert_eq!(status, 200); // consent page — error surfaces on POST
    let refused = no_redirect
        .post(&format!("{}/oauth/authorize", server.base))
        .set("Content-Type", "application/x-www-form-urlencoded")
        .send_string(&format!("{bad_query}&key=tok-op"))
        .expect("a bad code_challenge_method must still redirect with ?error=");
    assert_eq!(refused.status(), 303);
    let error_location = refused.header("location").unwrap().to_string();
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
    let no_redirect = ureq::AgentBuilder::new().redirects(0).build();
    let authorize = |query: &str| -> (u16, String) {
        match no_redirect
            .get(&format!("{}/oauth/authorize?{query}", server.base))
            .call()
        {
            Ok(response) => (
                response.status(),
                response.header("location").unwrap_or_default().to_string(),
            ),
            Err(ureq::Error::Status(status, response)) => (
                status,
                response.header("location").unwrap_or_default().to_string(),
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

    // A resource that does not name this server's own /mcp is a
    // distinct failure: invalid_target, not invalid_request.
    let (status, location) = authorize(&format!(
        "response_type=code&client_id={client_id}&redirect_uri={REDIRECT}&state=s4\
         &code_challenge=abc&code_challenge_method=S256&resource=https%3A%2F%2Fother.example%2Fmcp"
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

/// POST /mcp speaks the MCP Streamable HTTP transport (stateless
/// profile): the same tools as the stdio bridge, over the same routes.
#[test]
fn mcp_over_http_serves_initialize_tools_and_calls() {
    let server = Server::start("mcp");

    let (status, reply) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18"}})),
    );
    assert_eq!(status, 200);
    assert_eq!(reply["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(
        reply["result"]["serverInfo"]["version"],
        env!("CARGO_PKG_VERSION")
    );
    let instructions = reply["result"]["instructions"].as_str().unwrap();
    assert!(instructions.contains("# Taguru"));

    let (_, tools) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})),
    );
    let list = tools["result"]["tools"].as_array().unwrap();
    assert!(
        list.iter().any(|tool| tool["name"] == "recall"),
        "tools/list must advertise the bridge's tool set"
    );

    // A notification: heard (202), nothing to answer with.
    let (status, _) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "method": "notifications/initialized"})),
    );
    assert_eq!(status, 202);

    // A real tool round trip: create a context, then see it listed.
    let (_, created) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "create_context",
                               "arguments": {"name": "remote", "description": "over /mcp"}}})),
    );
    assert!(created["result"].get("isError").is_none(), "{created}");
    let (_, listed) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                    "params": {"name": "list_contexts", "arguments": {}}})),
    );
    let text = listed["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("\"remote\""), "{text}");

    // A failing tool travels as isError CONTENT (the agent reads the
    // server's explanation), not as a JSON-RPC protocol error.
    let (status, failed) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                    "params": {"name": "describe",
                               "arguments": {"context": "nope", "concept": "x"}}})),
    );
    assert_eq!(status, 200);
    assert_eq!(failed["result"]["isError"], true);
    let error_text = failed["result"]["content"][0]["text"].as_str().unwrap();
    assert!(error_text.contains("HTTP 404"), "{error_text}");

    // Unknown JSON-RPC methods ARE protocol errors...
    let (_, unknown) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 6, "method": "resources/list"})),
    );
    assert_eq!(unknown["error"]["code"], -32601);
    // ...broken JSON answers -32700 in JSON-RPC dress...
    let (status, parse) = server.call_raw("POST", "/mcp", Some("{nope"), Some("application/json"));
    assert_eq!(status, 400);
    assert_eq!(parse["error"]["code"], -32700);
    // ...and the wrong verb answers like any other route.
    let (status, _) = server.call("GET", "/mcp", None);
    assert_eq!(status, 405);
}

/// /mcp sits behind the bearer token like every route — and a tool
/// dispatched through it is NOT re-authenticated inside; the /mcp
/// entry is the auth point.
#[test]
fn mcp_endpoint_honors_bearer_auth_end_to_end() {
    let server = Server::start_with_env("mcp-auth", &[("TAGURU_API_TOKEN", "mcp-secret")]);

    let init = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}});
    let (status, _) = server.call("POST", "/mcp", Some(init.clone()));
    assert_eq!(status, 401);
    let (status, _) = server.call_with_token("POST", "/mcp", Some(init), Some("mcp-secret"));
    assert_eq!(status, 200);

    // The dispatched inner request carries no Authorization header; if
    // dispatch re-entered the middleware, this would come back as an
    // isError HTTP 401 content block.
    let call = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                      "params": {"name": "create_context",
                                 "arguments": {"name": "armed", "description": "token-protected"}}});
    let (status, reply) = server.call_with_token("POST", "/mcp", Some(call), Some("mcp-secret"));
    assert_eq!(status, 200);
    assert!(reply["result"].get("isError").is_none(), "{reply}");
}

/// Two ways a POSTed body can fail to be one JSON-RPC request before
/// `classify` ever runs a method through: a batch array (the framing
/// the 2025-06 spec dropped) and an object with no "method" at all.
/// Both are -32600, distinguished only by message text, so the
/// assertions must read the text — not just the shared code.
#[test]
fn mcp_rejects_batches_and_undecodable_messages() {
    let server = Server::start("mcp-malformed");

    let (status, batch) = server.call(
        "POST",
        "/mcp",
        Some(json!([
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},
        ])),
    );
    assert_eq!(status, 400);
    assert_eq!(batch["error"]["code"], -32600);
    assert!(
        batch["error"]["message"]
            .as_str()
            .unwrap()
            .contains("batch messages are not part of MCP"),
        "{batch}"
    );

    let (status, undecodable) =
        server.call("POST", "/mcp", Some(json!({"jsonrpc": "2.0", "id": 1})));
    assert_eq!(status, 400);
    assert_eq!(undecodable["error"]["code"], -32600);
    assert!(
        undecodable["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not a JSON-RPC message (no method)"),
        "{undecodable}"
    );
}

#[test]
fn full_retrieval_loop_over_http() {
    let server = Server::start("loop");

    // Health, playbook, empty directory.
    let (status, health) = server.call("GET", "/health", None);
    assert_eq!((status, health), (200, Value::String("ok".into())));
    let (status, protocol) = server.call("GET", "/protocol", None);
    assert_eq!(status, 200);
    assert!(protocol.as_str().unwrap().contains("# Taguru"));
    // Lexical-only server: no live-configuration trailer to act on.
    assert!(!protocol.as_str().unwrap().contains("## This server"));
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["total"], json!(0));
    assert_eq!(directory["contexts"], json!([]));

    // Create; duplicates conflict; unknown contexts 404.
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "酒蔵の知識", "dice_floor": 0.3})),
    );
    let (status, _) = server.call("PUT", "/contexts/sake", Some(json!({})));
    assert_eq!(status, 409);
    let (status, _) = server.call("POST", "/contexts/nope/recall", Some(json!({"cue": "x"})));
    assert_eq!(status, 404);

    // Ingest a batch plus its passage.
    let applied = server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0, "source": "第1段落"},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 2.0, "source": "第2段落"},
            {"subject": "青嶺酒造", "label": "仕込み水", "object": "雲居山の伏流水", "weight": 1.0, "source": "第2段落"},
            {"subject": "青嶺酒造", "label": "仕込み水", "object": "雲居山の伏流水", "weight": 1.0, "source": "第5段落"},
            {"subject": "高瀬", "label": "出身", "object": "南部杜氏", "weight": 1.0, "source": "第3段落"},
            {"subject": "青嶺酒造", "label": "行う", "object": "大量生産", "weight": -1.0, "source": "第2段落"},
        ])),
    );
    assert_eq!(applied, json!(6));
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {
            "第2段落": "青嶺酒造は、仕込み水に雲居山の伏流水を使う。杜氏は高瀬である。",
        }})),
    );

    // recall/query pages carry totals; query takes OR-sets per position.
    let page = server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "青嶺酒造", "limit": 3})),
    );
    assert_eq!(page["total"], json!(4));
    assert_eq!(page["matches"].as_array().unwrap().len(), 3);
    // Truncation keeps the strongest |weight| first.
    assert_eq!(page["matches"][0]["label"], json!("杜氏"));
    let narrowed = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": ["代表銘柄", "杜氏"]})),
    );
    assert_eq!(narrowed["total"], json!(2));

    // describe outlines without materializing; corroboration shows in
    // attributions through query.
    let outline = server.ok(
        "POST",
        "/contexts/sake/describe",
        Some(json!({"concept": "青嶺酒造"})),
    );
    assert_eq!(outline["as_subject"][0]["label"], json!("代表銘柄")); // count ties -> label insertion order
    let water = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "仕込み水"})),
    );
    assert_eq!(water["matches"][0]["weight"], json!(2.0));
    assert_eq!(
        water["matches"][0]["attributions"]
            .as_array()
            .unwrap()
            .len(),
        2
    );

    // resolve tiers: exact is lexical; a typo lands through the fuzzy
    // tier; the per-call floor tightens it away.
    let exact = server.ok(
        "POST",
        "/contexts/sake/resolve",
        Some(json!({"cue": "青嶺酒造"})),
    );
    assert_eq!(exact[0]["tier"], json!("lexical"));
    assert_eq!(exact[0]["score"], json!(1.0));
    let typo = server.ok(
        "POST",
        "/contexts/sake/resolve",
        Some(json!({"cue": "青嶺酒蔵"})),
    );
    assert_eq!(typo[0]["name"], json!("青嶺酒造"));
    let strict = server.ok(
        "POST",
        "/contexts/sake/resolve",
        Some(json!({"cue": "青嶺酒蔵", "dice_floor": 0.9})),
    );
    assert!(
        !strict
            .as_array()
            .unwrap()
            .iter()
            .any(|hit| hit["name"] == json!("青嶺酒造"))
    );

    // Walks carry paths; strengths rank magnitude (the negative fact
    // outranks weight-1 facts).
    let ranked = server.ok(
        "POST",
        "/contexts/sake/activate",
        Some(json!({"origins": ["青嶺酒造"], "limit": 3})),
    );
    assert_eq!(ranked[0]["association"]["label"], json!("杜氏"));
    assert_eq!(ranked[0]["path"], json!(["青嶺酒造"]));
    let walked = server.ok(
        "POST",
        "/contexts/sake/explore",
        Some(json!({"origins": ["青嶺酒造"], "max_depth": 2})),
    );
    assert!(
        walked["matches"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["distance"] == json!(2) && r["path"] == json!(["青嶺酒造", "高瀬"]))
    );

    // Aliases resolve at entry, answer with canonical spellings, and
    // refuse to shadow existing spellings.
    server.ok(
        "POST",
        "/contexts/sake/aliases",
        Some(
            json!({"concepts": {"Aomine Brewery": "青嶺酒造"}, "labels": {"蔵元の責任者": "杜氏"}}),
        ),
    );
    let via_alias = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "Aomine Brewery", "label": "蔵元の責任者"})),
    );
    assert_eq!(via_alias["matches"][0]["subject"], json!("青嶺酒造"));
    assert_eq!(via_alias["matches"][0]["object"], json!("高瀬"));
    let (status, _) = server.call(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"concepts": {"青嶺": "青嶺酒造"}})),
    );
    assert_eq!(status, 409, "shadowing an existing concept must conflict");

    // Coverage audit, passage lookup and search, retraction.
    let orphans = server.ok(
        "POST",
        "/contexts/sake/unreachable_from",
        Some(json!({"origins": ["青嶺酒造"]})),
    );
    assert_eq!(orphans, json!({"total": 0, "matches": []}));
    let passages = server.ok(
        "POST",
        "/contexts/sake/sources/lookup",
        Some(json!({"sources": ["第2段落", "第9段落"]})),
    );
    assert!(
        passages["passages"]["第2段落"]
            .as_str()
            .unwrap()
            .contains("伏流水")
    );
    assert_eq!(passages["missing"], json!(["第9段落"]));
    let hits = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "仕込み水はどこの水?"})),
    );
    assert_eq!(hits[0]["source"], json!("第2段落"));
    let retracted = server.ok(
        "POST",
        "/contexts/sake/sources/retract",
        Some(json!({"source": "第5段落"})),
    );
    assert_eq!(retracted["associations_touched"], json!(1));
    let water = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "仕込み水"})),
    );
    assert_eq!(water["matches"][0]["weight"], json!(1.0));

    // Metadata edits show up in the directory; embeddings without a
    // provider are refused as unimplemented.
    server.ok(
        "PATCH",
        "/contexts/sake",
        Some(json!({"pinned": true, "semantic_floor": 0.2})),
    );
    let listed = server.ok("GET", "/contexts", None)["contexts"].clone();
    assert_eq!(listed[0]["pinned"], json!(true));
    assert_eq!(listed[0]["semantic_floor"], json!(0.2));
    assert_eq!(listed[0]["stats"]["associations"], json!(5));
    // The single-context row says the same thing without the listing.
    let single = server.ok("GET", "/contexts/sake", None);
    assert_eq!(single["name"], json!("sake"));
    assert_eq!(single["stats"]["associations"], json!(5));
    let (status, _) = server.call("POST", "/contexts/sake/embeddings/refresh", None);
    assert_eq!(status, 501);

    // Deletion removes the context and its files.
    server.ok("DELETE", "/contexts/sake", None);
    assert_eq!(server.ok("GET", "/contexts", None)["total"], json!(0));
    let (status, _) = server.call("GET", "/contexts/sake", None);
    assert_eq!(status, 404);
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

#[test]
fn graph_writes_survive_a_hard_kill() {
    // A one-hour flush interval: the periodic flusher provably cannot
    // have persisted anything before the SIGKILL lands — whatever
    // comes back after the restart came through the WAL.
    let server = Server::start_with_env("hardkill", &[("TAGURU_FLUSH_SECS", "3600")]);
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "source": "第1段落"},
        ])),
    );

    let data_dir = server.stop_hard();
    let server = Server::start_on("hardkill2", data_dir);
    let recalled = server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "青嶺酒造"})),
    );
    assert_eq!(recalled["matches"][0]["object"], json!("1907年"));
}

#[test]
fn an_association_batch_over_the_cap_is_rejected_before_any_write() {
    let server = Server::start("batchcap");
    server.ok("PUT", "/contexts/sake", Some(json!({})));

    let batch: Vec<Value> = (0..10_001)
        .map(|i| json!({"subject": format!("s{i}"), "label": "l", "object": "o", "weight": 1.0}))
        .collect();
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(Value::Array(batch)),
    );
    assert_eq!(status, 400, "{body}");

    // The guard ran before the write lock: nothing was applied.
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["contexts"][0]["stats"]["associations"], json!(0));
}

#[test]
fn an_insane_weight_is_rejected_before_any_write() {
    let server = Server::start("weightcap");
    server.ok("PUT", "/contexts/sake", Some(json!({})));

    // Finite but absurd: two of these would saturate an edge to
    // +Infinity, and a later retract would mint Inf − Inf = NaN — a
    // fact nothing can read or reset again.
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "a", "label": "l", "object": "b", "weight": 1.0},
            {"subject": "青嶺酒造", "label": "生産量", "object": "無限", "weight": 1.0e300},
        ])),
    );
    assert_eq!(status, 400, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("associations[1].weight"),
        "the message must point at the offending item: {body}"
    );
    // Refused whole, before the write lock: not even the sane first
    // item landed.
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["contexts"][0]["stats"]["associations"], json!(0));

    // The documented boundary stays usable, negation included.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "a", "label": "l", "object": "b", "weight": 1.0e6},
            {"subject": "a", "label": "l2", "object": "b", "weight": -1.0e6},
        ])),
    );
}

#[test]
fn the_directory_pages_by_name_and_serves_single_contexts() {
    let server = Server::start("dirpage");
    for name in ["apple", "banana", "cherry"] {
        server.ok(
            "PUT",
            &format!("/contexts/{name}"),
            Some(json!({"description": name})),
        );
    }

    let page = server.ok("GET", "/contexts?limit=2", None);
    assert_eq!(page["total"], json!(3), "total names the full count");
    let names: Vec<&str> = page["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|context| context["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["apple", "banana"], "name order, first page");

    let page = server.ok("GET", "/contexts?limit=2&after=banana", None);
    assert_eq!(page["total"], json!(3));
    let names: Vec<&str> = page["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|context| context["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["cherry"], "keyset picks up after the cursor");

    let single = server.ok("GET", "/contexts/banana", None);
    assert_eq!(single["name"], json!("banana"));
    assert_eq!(single["description"], json!("banana"));
    let (status, body) = server.call("GET", "/contexts/nope", None);
    assert_eq!(status, 404);
    assert_eq!(body["status"], json!("error"));
}

#[test]
fn a_present_body_is_parsed_whatever_the_content_type_says() {
    let server = Server::start("rawbody");

    // requests.put(url, data=json.dumps(...)) territory: a JSON body
    // with no JSON Content-Type. The description must land — this
    // used to silently drop the body and create with every field
    // defaulted, under a 200.
    let (status, body) = server.call_raw(
        "PUT",
        "/contexts/sake",
        Some(r#"{"description":"青嶺酒造の記憶","pinned":true}"#),
        None,
    );
    assert_eq!(status, 200, "{body}");
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(
        directory["contexts"][0]["description"],
        json!("青嶺酒造の記憶")
    );
    assert_eq!(directory["contexts"][0]["pinned"], json!(true));

    // A present body that is not JSON is an error, never defaults.
    let (status, body) =
        server.call_raw("PUT", "/contexts/beer", Some("definitely not json"), None);
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["status"], json!("error"));

    // An absent body still means defaults — the documented shape.
    let (status, body) = server.call_raw("PUT", "/contexts/beer", None, None);
    assert_eq!(status, 200, "{body}");

    // The other optional-body endpoint follows the same contract.
    let (status, body) = server.call_raw(
        "POST",
        "/contexts/sake/vocabulary/audit",
        Some("also not json"),
        None,
    );
    assert_eq!(status, 400, "{body}");
    let (status, body) = server.call_raw("POST", "/contexts/sake/vocabulary/audit", None, None);
    assert_eq!(status, 200, "{body}");
}

#[test]
fn off_axis_errors_speak_the_api_error_shape_too() {
    let server = Server::start("errshape");
    server.ok("PUT", "/contexts/sake", Some(json!({})));

    // Unknown path → 404 in the error shape.
    let (status, body) = server.call("GET", "/contextz", None);
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
    assert!(body["time"].is_number(), "{body}");

    // Known path, wrong method → 405 in the error shape.
    let (status, body) = server.call("DELETE", "/contexts/sake/recall", None);
    assert_eq!(status, 405, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");

    // Malformed JSON on a JSON-required endpoint → 400 in shape.
    let (status, body) = server.call_raw(
        "POST",
        "/contexts/sake/recall",
        Some("{not json"),
        Some("application/json"),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");

    // Wrong media type → 415 in shape.
    let (status, body) = server.call_raw(
        "POST",
        "/contexts/sake/recall",
        Some("cue=x"),
        Some("application/x-www-form-urlencoded"),
    );
    assert_eq!(status, 415, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");

    // Well-formed JSON of the wrong type → 422 in shape.
    let (status, body) = server.call_raw(
        "POST",
        "/contexts/sake/recall",
        Some(r#"{"cue": 42}"#),
        Some("application/json"),
    );
    assert_eq!(status, 422, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
}

#[test]
fn oversized_names_are_rejected_at_every_write_boundary() {
    let server = Server::start("namecap");
    let long = "字".repeat(400); // 1200 bytes, over the 1024-byte name cap

    // A context name becomes a file stem (percent-encoded ×3): 64
    // bytes is the cap.
    let (status, body) = server.call(
        "PUT",
        &format!("/contexts/{}", "n".repeat(65)),
        Some(json!({})),
    );
    assert_eq!(status, 400, "{body}");

    // The description rides in every directory listing.
    let (status, body) = server.call(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "d".repeat(5000)})),
    );
    assert_eq!(status, 400, "{body}");

    server.ok("PUT", "/contexts/sake", Some(json!({})));
    let (status, body) = server.call(
        "PATCH",
        "/contexts/sake",
        Some(json!({"description": "d".repeat(5000)})),
    );
    assert_eq!(status, 400, "{body}");

    // Graph names: the top-concepts snapshot carries them into every
    // GET /contexts response, far outside the cache budget.
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": long, "label": "l", "object": "o", "weight": 1.0}])),
    );
    assert_eq!(status, 400, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("associations[0].subject"),
        "the message must point at the offending field: {body}"
    );

    // Aliases and passage source ids persist names too.
    let mut concepts = serde_json::Map::new();
    concepts.insert(long.clone(), json!("x"));
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"concepts": concepts, "labels": {}})),
    );
    assert_eq!(status, 400, "{body}");
    let mut passages = serde_json::Map::new();
    passages.insert(long.clone(), json!("原文"));
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": passages})),
    );
    assert_eq!(status, 400, "{body}");

    // Nothing landed anywhere.
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["contexts"][0]["stats"]["associations"], json!(0));

    // The boundary itself stays usable.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{
            "subject": "s".repeat(1024), "label": "l", "object": "o", "weight": 1.0
        }])),
    );
}

#[test]
fn empty_names_are_rejected_at_the_write_boundary() {
    let server = Server::start("emptyname");
    server.ok("PUT", "/contexts/sake", Some(json!({})));

    // An empty subject/label/object is not a degenerate name, it is no
    // name — each must be refused on its own, naming the offending
    // field.
    for (field, triple) in [
        (
            "subject",
            json!({"subject": "", "label": "l", "object": "o", "weight": 1.0}),
        ),
        (
            "label",
            json!({"subject": "s", "label": "", "object": "o", "weight": 1.0}),
        ),
        (
            "object",
            json!({"subject": "s", "label": "l", "object": "", "weight": 1.0}),
        ),
    ] {
        let (status, body) =
            server.call("POST", "/contexts/sake/associations", Some(json!([triple])));
        assert_eq!(status, 400, "{field}: {body}");
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains(&format!("associations[0].{field}")),
            "the message must point at the offending field: {body}"
        );
    }

    // An omitted source is the ordinary unsourced-association case,
    // not a missing name — it must NOT be swept up by the same check.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": "s", "label": "l", "object": "o", "weight": 1.0}])),
    );

    // Only the one, deliberately unsourced association landed.
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["contexts"][0]["stats"]["associations"], json!(1));
}

#[test]
fn unreachable_from_pages_like_recall_and_query() {
    let server = Server::start("orphanpage");
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0},
            // Three islands no walk from the origin can reach.
            {"subject": "x1", "label": "l", "object": "y1", "weight": 1.0},
            {"subject": "x2", "label": "l", "object": "y2", "weight": 1.0},
            {"subject": "x3", "label": "l", "object": "y3", "weight": 1.0},
        ])),
    );

    let audit = server.ok(
        "POST",
        "/contexts/sake/unreachable_from",
        Some(json!({"origins": ["青嶺酒造"], "limit": 2})),
    );
    assert_eq!(audit["total"], json!(3));
    assert_eq!(audit["matches"].as_array().unwrap().len(), 2);
}

#[test]
fn explore_without_max_depth_stops_at_the_server_ceiling() {
    let server = Server::start("depthcap");
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    // A 15-hop chain: c0 → c1 → … → c15.
    let chain: Vec<Value> = (0..15)
        .map(|i| {
            json!({"subject": format!("c{i}"), "label": "next", "object": format!("c{}", i + 1), "weight": 1.0})
        })
        .collect();
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(Value::Array(chain)),
    );

    let walked = server.ok(
        "POST",
        "/contexts/sake/explore",
        Some(json!({"origins": ["c0"]})),
    );
    let deepest = walked["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["distance"].as_u64().unwrap())
        .max()
        .unwrap();
    assert_eq!(deepest, 10, "omitted max_depth must stop at the ceiling");
}

#[test]
fn explore_pages_and_keeps_the_closest_past_the_limit() {
    let server = Server::start("explorepage");
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    // A hub with four direct neighbours; one leads a hop further to a
    // heavy edge.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "hub", "label": "l", "object": "n1", "weight": 1.0},
            {"subject": "hub", "label": "l", "object": "n2", "weight": 1.0},
            {"subject": "hub", "label": "l", "object": "n3", "weight": 1.0},
            {"subject": "hub", "label": "l", "object": "n4", "weight": 1.0},
            {"subject": "n1", "label": "l", "object": "far", "weight": 9.0},
        ])),
    );

    let walked = server.ok(
        "POST",
        "/contexts/sake/explore",
        Some(json!({"origins": ["hub"], "limit": 3})),
    );
    assert_eq!(walked["total"], json!(5));
    let matches = walked["matches"].as_array().unwrap();
    assert_eq!(matches.len(), 3);
    // The cut keeps the closest structure, not the heaviest weight:
    // the distance-2 edge (weight 9.0) is what falls off.
    assert!(
        matches.iter().all(|r| r["distance"] == json!(1)),
        "{walked}"
    );
}

#[test]
fn post_flush_persists_dirty_contexts_on_demand() {
    // The periodic flusher is effectively off: the endpoint is the
    // only thing that can move the image.
    let server = Server::start_with_env("forceflush", &[("TAGURU_FLUSH_SECS", "3600")]);
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": "a", "label": "l", "object": "b", "weight": 1.0}])),
    );

    let flushed = server.ok("POST", "/flush", None);
    assert_eq!(flushed, json!(["sake"]), "the dirty context must flush");
    // Nothing left dirty: an immediate second call is a no-op.
    assert_eq!(server.ok("POST", "/flush", None), json!([]));
}

#[test]
fn the_wal_cap_env_refuses_writes_rather_than_growing_forever() {
    // Flushes effectively never run, so the log can only grow; a
    // 1-byte cap trips on the second write.
    let server = Server::start_with_env(
        "walcap",
        &[("TAGURU_WAL_MAX_BYTES", "1"), ("TAGURU_FLUSH_SECS", "3600")],
    );
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": "a", "label": "l", "object": "b", "weight": 1.0}])),
    );
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": "c", "label": "l", "object": "d", "weight": 1.0}])),
    );
    assert_eq!(status, 500, "{body}");
    assert!(
        body["error"].as_str().unwrap().contains("write-ahead log"),
        "{body}"
    );
}

#[test]
fn a_bind_failure_exits_with_a_diagnosis_not_a_panic() {
    // Occupy a port, then ask the server to bind it.
    let holder = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = holder.local_addr().unwrap().to_string();
    let data_dir =
        std::env::temp_dir().join(format!("taguru-http-bindfail-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);

    let output = Command::new(env!("CARGO_BIN_EXE_taguru"))
        .env("TAGURU_ADDR", &addr)
        .env("TAGURU_DATA_DIR", &data_dir)
        .env_remove("TAGURU_EMBED_URL")
        .env_remove("TAGURU_API_TOKEN")
        .output()
        .expect("server binary must spawn");

    assert!(!output.status.success(), "a failed bind must exit nonzero");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("cannot bind"), "{stderr}");
    assert!(
        !stderr.contains("panicked"),
        "an operator mistake must not read as a crash: {stderr}"
    );
    let _ = std::fs::remove_dir_all(&data_dir);
}

#[test]
#[cfg(unix)]
fn health_reports_503_while_flushes_fail_and_recovers_after() {
    use std::os::unix::fs::PermissionsExt;
    use std::time::{Duration, Instant};

    let server = Server::start("health503");
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    // The first write creates the WAL file while the directory is
    // still writable; afterwards appends only need the existing file.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": "a", "label": "l", "object": "b", "weight": 1.0}])),
    );

    std::fs::set_permissions(&server.data_dir, std::fs::Permissions::from_mode(0o555)).unwrap();
    // Keep the context dirty each round so a tick between our calls
    // cannot leave the flusher idle-and-green.
    let deadline = Instant::now() + Duration::from_secs(10);
    let degraded = loop {
        let _ = server.call(
            "POST",
            "/contexts/sake/associations",
            Some(json!([{"subject": "a", "label": "l", "object": "b", "weight": 0.001}])),
        );
        let (status, body) = server.call("GET", "/health", None);
        if status == 503 {
            break body;
        }
        assert!(Instant::now() < deadline, "health never degraded");
        std::thread::sleep(Duration::from_millis(200));
    };
    assert_eq!(degraded["status"], json!("error"), "{degraded}");

    std::fs::set_permissions(&server.data_dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let _ = server.call(
            "POST",
            "/contexts/sake/associations",
            Some(json!([{"subject": "a", "label": "l", "object": "b", "weight": 0.001}])),
        );
        let (status, _) = server.call("GET", "/health", None);
        if status == 200 {
            break;
        }
        assert!(Instant::now() < deadline, "health never recovered");
        std::thread::sleep(Duration::from_millis(200));
    }
}

#[test]
fn a_body_over_the_configured_limit_is_rejected_with_413() {
    let server = Server::start_with_env("bodycap", &[("TAGURU_MAX_BODY_BYTES", "16")]);
    let (status, _) = server.call(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "この説明は16バイトよりずっと長い"})),
    );
    assert_eq!(status, 413);
}

#[test]
fn a_custom_request_timeout_does_not_disturb_fast_requests() {
    // The deadline actually firing is unit-tested in limits.rs; this
    // pins the wiring — a tight budget must not break normal traffic.
    let server = Server::start_with_env("timeout", &[("TAGURU_REQUEST_TIMEOUT_SECS", "1")]);
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    assert_eq!(server.call("GET", "/contexts", None).0, 200);
}

#[test]
fn metrics_expose_prometheus_text_reflecting_traffic() {
    let server = Server::start("metrics");

    // Two health probes, then two recalls against DIFFERENT context
    // names on the same route template (both 404 — routing happened,
    // which is all the label needs).
    server.call("GET", "/health", None);
    server.call("GET", "/health", None);
    server.call("POST", "/contexts/nope1/recall", Some(json!({"cue": "x"})));
    server.call("POST", "/contexts/nope2/recall", Some(json!({"cue": "x"})));
    // And one path that matches no route at all.
    server.call("GET", "/definitely/not/a/route", None);

    let (status, body) = server.call("GET", "/metrics", None);
    assert_eq!(status, 200);
    let text = body.as_str().expect("metrics body is text, not JSON");

    // Counted traffic, keyed by route template.
    assert!(
        text.contains(
            "taguru_http_requests_total{method=\"GET\",route=\"/health\",status=\"200\"} 2"
        ),
        "{text}"
    );
    assert!(
        text.contains(
            "taguru_http_requests_total{method=\"POST\",route=\"/contexts/{name}/recall\",status=\"404\"} 2"
        ),
        "two context names must fold into ONE templated series: {text}"
    );
    // The raw paths never become label values; unmatched requests all
    // share one bucket.
    assert!(!text.contains("nope1"), "raw path leaked into labels");
    assert!(!text.contains("/definitely/not/a/route"));
    assert!(text.contains("route=\"<unmatched>\""));

    // Histogram, domain counters, and gauges are all present.
    assert!(text.contains("taguru_http_request_duration_seconds_bucket"));
    assert!(text.contains("taguru_flush_total{outcome=\"ok\"}"));
    assert!(text.contains("taguru_contexts_registered 0"));
}

#[test]
fn lookalike_candidates_carry_the_evidence_to_tell_them_apart() {
    let server = Server::start("lookalikes");
    server.ok(
        "PUT",
        "/contexts/looks",
        Some(json!({"description": "字面の近い別物たち"})),
    );
    server.ok(
        "POST",
        "/contexts/looks/associations",
        Some(json!([
            {"subject": "東京都", "label": "分類", "object": "日本の首都", "weight": 1.0},
            {"subject": "京都", "label": "所在", "object": "関西", "weight": 1.0},
            {"subject": "青嶺株式会社", "label": "業種", "object": "電機メーカー", "weight": 1.0},
            {"subject": "possible", "label": "means", "object": "can_be_done", "weight": 1.0},
            {"subject": "impossible", "label": "means", "object": "can_be_done", "weight": -1.0},
        ])),
    );

    // 東京都/京都: the containment lookalike scores a strong 0.67, and
    // the response says both how it matched and what it actually is —
    // enough to reject it without a second round trip.
    let kyoto = server.ok(
        "POST",
        "/contexts/looks/resolve",
        Some(json!({"cue": "京都"})),
    );
    assert_eq!(kyoto[0]["name"], json!("京都"));
    assert_eq!(kyoto[0]["kind"], json!("exact"));
    assert!(
        kyoto[0]["gloss"].as_str().unwrap().contains("関西"),
        "{kyoto}"
    );
    assert_eq!(kyoto[1]["name"], json!("東京都"));
    assert_eq!(kyoto[1]["kind"], json!("containment"));
    assert!(
        kyoto[1]["gloss"].as_str().unwrap().contains("日本の首都"),
        "{kyoto}"
    );

    // 前株/後株: the cue names a company that is NOT registered; the
    // stored lookalike surfaces through the fuzzy tier, and its gloss
    // (wrong line of business) is what lets the caller reject it.
    let maekabu = server.ok(
        "POST",
        "/contexts/looks/resolve",
        Some(json!({"cue": "株式会社青嶺"})),
    );
    assert_eq!(maekabu[0]["name"], json!("青嶺株式会社"));
    assert_eq!(maekabu[0]["kind"], json!("fuzzy"));
    assert!(
        maekabu[0]["gloss"]
            .as_str()
            .unwrap()
            .contains("電機メーカー"),
        "{maekabu}"
    );

    // possible/impossible: containment scores 0.8 for the antonym; the
    // negative fact renders as a denial in its gloss.
    let possible = server.ok(
        "POST",
        "/contexts/looks/resolve",
        Some(json!({"cue": "possible"})),
    );
    assert_eq!(possible[0]["name"], json!("possible"));
    assert_eq!(possible[0]["kind"], json!("exact"));
    assert_eq!(possible[1]["name"], json!("impossible"));
    assert_eq!(possible[1]["kind"], json!("containment"));
    assert_eq!(possible[1]["score"], json!(8.0 / 10.0));
    assert!(
        possible[1]["gloss"]
            .as_str()
            .unwrap()
            .contains("can_be_doneではない"),
        "{possible}"
    );

    // Labels resolve with the same evidence; the gloss shows example
    // triples so a writer can pick the right relation before minting.
    let label = server.ok(
        "POST",
        "/contexts/looks/resolve_label",
        Some(json!({"cue": "means"})),
    );
    assert_eq!(label[0]["kind"], json!("exact"));
    assert!(
        label[0]["gloss"].as_str().unwrap().contains("means"),
        "{label}"
    );
}

#[test]
fn search_outcomes_and_resolve_tiers_land_in_the_metrics_text() {
    let server = Server::start("searchmetrics");
    server.ok("PUT", "/contexts/sm", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sm/associations",
        Some(json!([{
            "subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺",
            "weight": 1.0, "source": "p1"
        }])),
    );

    // One hit and one empty recall; one confident resolve and one miss
    // (no embedding provider in the harness, so nothing rescues it).
    server.ok(
        "POST",
        "/contexts/sm/recall",
        Some(json!({"cue": "青嶺酒造"})),
    );
    server.ok("POST", "/contexts/sm/recall", Some(json!({"cue": "qqqq"})));
    server.ok(
        "POST",
        "/contexts/sm/resolve",
        Some(json!({"cue": "青嶺酒造"})),
    );
    server.ok("POST", "/contexts/sm/resolve", Some(json!({"cue": "qqqq"})));

    let (status, body) = server.call("GET", "/metrics", None);
    assert_eq!(status, 200);
    let text = body.as_str().expect("metrics body is text, not JSON");

    assert!(
        text.contains("taguru_searches_total{op=\"recall\",outcome=\"hit\"} 1"),
        "{text}"
    );
    assert!(
        text.contains("taguru_searches_total{op=\"recall\",outcome=\"empty\"} 1"),
        "{text}"
    );
    assert!(
        text.contains("taguru_searches_total{op=\"resolve\",outcome=\"hit\"} 1"),
        "{text}"
    );
    assert!(
        text.contains("taguru_resolves_total{tier=\"lexical\"} 1"),
        "{text}"
    );
    assert!(
        text.contains("taguru_resolves_total{tier=\"miss\"} 1"),
        "{text}"
    );
}

#[test]
fn usage_counters_track_reads_writes_and_empties_per_context() {
    let server = Server::start("usage");
    server.ok("PUT", "/contexts/used", Some(json!({"description": "d"})));
    server.ok("PUT", "/contexts/idle", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/used/associations",
        Some(json!([{
            "subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺",
            "weight": 1.0, "source": "p1"
        }])),
    );
    server.ok(
        "POST",
        "/contexts/used/recall",
        Some(json!({"cue": "青嶺酒造"})),
    );
    server.ok(
        "POST",
        "/contexts/used/recall",
        Some(json!({"cue": "qqqq"})),
    );
    server.ok(
        "POST",
        "/contexts/used/query",
        Some(json!({"subject": "青嶺酒造"})),
    );

    let used = server.ok("GET", "/contexts/used", None);
    assert_eq!(used["usage"]["writes"], json!(1), "{used}");
    assert_eq!(used["usage"]["reads"], json!(3), "{used}");
    assert_eq!(used["usage"]["empty_reads"], json!(1), "{used}");
    assert!(used["usage"]["last_read_epoch"].as_u64().unwrap() > 0);
    assert!(used["usage"]["last_write_epoch"].as_u64().unwrap() > 0);

    // The untouched context shows exactly that — the "never chosen"
    // signal the directory exists to expose.
    let idle = server.ok("GET", "/contexts/idle", None);
    assert_eq!(idle["usage"]["reads"], json!(0), "{idle}");
    assert_eq!(idle["usage"]["writes"], json!(0), "{idle}");
    assert_eq!(idle["usage"]["last_read_epoch"], json!(0), "{idle}");
}

/// An empty associations or aliases batch applies nothing (`applied ==
/// 0`), so it must not bump the write counter — the same rule the
/// partial-write arm already applies via `partial.applied > 0`.
#[test]
fn empty_association_and_alias_batches_do_not_bump_the_write_counter() {
    let server = Server::start("empty-batch-writes");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    let applied = server.ok("POST", "/contexts/sake/associations", Some(json!([])));
    assert_eq!(applied, json!(0));

    let applied = server.ok(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"concepts": {}, "labels": {}})),
    );
    assert_eq!(applied, json!(0));

    let entry = server.ok("GET", "/contexts/sake", None);
    assert_eq!(
        entry["usage"]["writes"],
        json!(0),
        "empty batches must not count as writes: {entry}"
    );

    // A non-empty batch still counts — proving the counter isn't just
    // stuck at zero regardless of what reaches it.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{
            "subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺",
            "weight": 1.0, "source": "p1"
        }])),
    );
    let entry = server.ok("GET", "/contexts/sake", None);
    assert_eq!(entry["usage"]["writes"], json!(1), "{entry}");
}

#[test]
fn usage_counters_survive_a_graceful_restart_even_for_read_only_sessions() {
    let server = Server::start("usagerestart");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{
            "subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺",
            "weight": 1.0, "source": "p1"
        }])),
    );
    server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "青嶺酒造"})),
    );
    let data_dir = server.stop_gracefully();

    // Second boot performs READS ONLY: nothing dirties the graph, so
    // no image flush ever writes the sidecar — the shutdown sweep is
    // the only thing standing between these counters and oblivion.
    let server = Server::start_on("usagerestart", data_dir);
    server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "青嶺酒造"})),
    );
    let data_dir = server.stop_gracefully();

    let server = Server::start_on("usagerestart", data_dir);
    let entry = server.ok("GET", "/contexts/sake", None);
    assert_eq!(entry["usage"]["reads"], json!(2), "{entry}");
    assert_eq!(entry["usage"]["writes"], json!(1), "{entry}");
}

#[test]
fn log_output_is_structured_when_json_format_is_requested() {
    let data_dir = std::env::temp_dir().join(format!("taguru-jsonlog-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
    let mut child = Command::new(env!("CARGO_BIN_EXE_taguru"))
        .env("TAGURU_ADDR", "127.0.0.1:0")
        .env("TAGURU_DATA_DIR", &data_dir)
        .env("TAGURU_LOG_FORMAT", "json")
        .env_remove("TAGURU_EMBED_URL")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("server binary must spawn");

    // The first stderr line is already a log record (boot logging runs
    // before the listener binds); it must be one JSON object with the
    // standard fields, not pretty-format text.
    let stderr = child.stderr.take().expect("stderr must be piped");
    let line = BufReader::new(stderr)
        .lines()
        .next()
        .expect("a log line must appear")
        .expect("server stderr must be readable");
    let parsed: Value =
        serde_json::from_str(&line).unwrap_or_else(|_| panic!("stderr is not JSON: {line}"));
    assert!(parsed["level"].is_string(), "{parsed}");

    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&data_dir);
}

#[test]
fn data_survives_a_graceful_restart() {
    let server = Server::start("restart");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "再起動テスト"})),
    );
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "source": "第1段落"},
        ])),
    );

    // SIGTERM triggers the shutdown flush; the same data directory must
    // come back with the knowledge intact.
    let data_dir = server.stop_gracefully();
    let server = Server::start_on("restart2", data_dir);
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["contexts"][0]["name"], json!("sake"));
    assert_eq!(
        directory["contexts"][0]["description"],
        json!("再起動テスト")
    );
    let recalled = server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "青嶺酒造"})),
    );
    assert_eq!(recalled["matches"][0]["object"], json!("1907年"));
}

// ---------------------------------------------------------------------
// Distributed tracing: the OTLP pipeline end to end, against a fake
// collector — real binary, real exporter, real wire format (OTLP/JSON,
// the same payload schema as protobuf, minus a protobuf decoder in the
// test). Everything below opts in via OTEL_EXPORTER_OTLP_ENDPOINT; the
// rest of this file keeps proving the disabled mode.

/// A single-purpose OTLP/HTTP sink: accepts POSTs, stores every body,
/// answers 200. Runs until the test process exits.
struct FakeCollector {
    endpoint: String,
    bodies: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl FakeCollector {
    fn start() -> Self {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("collector must bind");
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let bodies = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = bodies.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                // Read headers, then exactly Content-Length body bytes.
                let mut buffer = Vec::new();
                let mut chunk = [0u8; 4096];
                let header_end = loop {
                    match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => break None,
                        Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                    }
                    if let Some(at) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                        break Some(at + 4);
                    }
                };
                let Some(header_end) = header_end else {
                    continue;
                };
                let headers = String::from_utf8_lossy(&buffer[..header_end]).to_string();
                let length: usize = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse().ok())?
                    })
                    .unwrap_or(0);
                while buffer.len() < header_end + length {
                    match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                    }
                }
                let body = String::from_utf8_lossy(&buffer[header_end..]).to_string();
                sink.lock().unwrap().push(body);
                let _ = stream.write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                      Content-Length: 2\r\nConnection: close\r\n\r\n{}",
                );
            }
        });
        Self { endpoint, bodies }
    }

    /// Every span object exported so far, flattened across batches.
    fn spans(&self) -> Vec<Value> {
        let mut spans = Vec::new();
        for body in self.bodies.lock().unwrap().iter() {
            let Ok(parsed) = serde_json::from_str::<Value>(body) else {
                continue;
            };
            for resource_spans in parsed["resourceSpans"].as_array().into_iter().flatten() {
                for scope_spans in resource_spans["scopeSpans"]
                    .as_array()
                    .into_iter()
                    .flatten()
                {
                    for span in scope_spans["spans"].as_array().into_iter().flatten() {
                        let mut span = span.clone();
                        span["resource"] = resource_spans["resource"].clone();
                        spans.push(span);
                    }
                }
            }
        }
        spans
    }
}

/// One attribute value out of the OTLP attribute list shape
/// `[{"key": ..., "value": {"stringValue": ...}}]`.
fn attribute<'a>(span: &'a Value, key: &str) -> Option<&'a Value> {
    span["attributes"]
        .as_array()?
        .iter()
        .find(|attribute| attribute["key"] == key)
        .map(|attribute| &attribute["value"])
}

#[test]
fn a_request_span_reaches_the_collector_carrying_the_inbound_trace_identity() {
    let collector = FakeCollector::start();
    let server = Server::start_with_env(
        "otlp",
        &[
            ("OTEL_EXPORTER_OTLP_ENDPOINT", collector.endpoint.as_str()),
            ("OTEL_EXPORTER_OTLP_PROTOCOL", "http/json"),
            ("OTEL_BSP_SCHEDULE_DELAY", "100"),
        ],
    );

    // The upstream (a mesh, another service) already started a trace.
    let response = ureq::AgentBuilder::new()
        .build()
        .request("GET", &format!("{}/health", server.base))
        .set(
            "traceparent",
            "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        )
        .call()
        .expect("health must answer");
    assert_eq!(response.status(), 200);

    // Graceful shutdown flushes the batch exporter before exit, so by
    // the time the process is gone the span has been delivered.
    let _ = server.stop_gracefully();

    let spans = collector.spans();
    let span = spans
        .iter()
        .find(|span| span["name"] == "GET /health")
        .unwrap_or_else(|| panic!("no GET /health span among {spans:?}"));

    // Same trace, parented under the caller's span: this is the whole
    // point of accepting inbound context.
    assert_eq!(span["traceId"], "0af7651916cd43dd8448eb211c80319c");
    assert_eq!(span["parentSpanId"], "b7ad6b7169203331");
    assert_eq!(
        attribute(span, "http.route").map(|value| value["stringValue"].clone()),
        Some(json!("/health"))
    );
    assert_eq!(
        attribute(span, "http.response.status_code").cloned(),
        Some(json!({"intValue": "200"}))
    );

    // The resource names the service — the default when
    // OTEL_SERVICE_NAME is unset.
    let service = span["resource"]["attributes"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|attribute| attribute["key"] == "service.name")
        .map(|attribute| attribute["value"]["stringValue"].clone());
    assert_eq!(service, Some(json!("taguru")));
}

#[test]
fn the_access_log_carries_the_trace_id_when_export_is_configured() {
    let data_dir = std::env::temp_dir().join(format!("taguru-tracelog-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
    // The endpoint only needs to be configured, not alive: spans are
    // created (and the log correlated) regardless of delivery.
    let mut child = Command::new(env!("CARGO_BIN_EXE_taguru"))
        .env("TAGURU_ADDR", "127.0.0.1:0")
        .env("TAGURU_DATA_DIR", &data_dir)
        .env("TAGURU_LOG_FORMAT", "json")
        .env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
        .env_remove("TAGURU_EMBED_URL")
        .env_remove("TAGURU_API_TOKEN")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("server binary must spawn");

    let stdout = child.stdout.take().expect("stdout must be piped");
    let mut stdout_lines = BufReader::new(stdout).lines();
    let base = loop {
        let line = stdout_lines
            .next()
            .expect("server must print its address")
            .expect("server stdout must be readable");
        if let Some(addr) = line.strip_prefix("listening on ") {
            break format!("http://{addr}");
        }
    };

    let response = ureq::AgentBuilder::new()
        .build()
        .request("GET", &format!("{base}/health"))
        .set(
            "traceparent",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        )
        .call()
        .expect("health must answer");
    assert_eq!(response.status(), 200);

    // The access-log line for that request must carry the same trace
    // id the caller minted — the log↔trace join key.
    let stderr = child.stderr.take().expect("stderr must be piped");
    let mut found = None;
    for line in BufReader::new(stderr).lines().take(200) {
        let Ok(line) = line else { break };
        let Ok(parsed) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if parsed["fields"]["message"] == "http" && parsed["fields"]["route"] == "/health" {
            found = Some(parsed);
            break;
        }
    }
    let record = found.expect("an access-log line for /health must appear");
    assert_eq!(
        record["fields"]["trace_id"],
        json!("4bf92f3577b34da6a3ce929d0e0e4736"),
        "{record}"
    );

    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&data_dir);
}

/// One request against a manually spawned server — the JSON-log
/// sessions below run outside the `Server` harness so they can own
/// stderr. Returns the status; bodies are irrelevant to log tests.
fn raw_call(base: &str, method: &str, path: &str, body: Option<Value>) -> u16 {
    let request = ureq::AgentBuilder::new()
        .build()
        .request(method, &format!("{base}{path}"));
    let response = match body {
        Some(body) => request
            .set("Content-Type", "application/json")
            .send_string(&body.to_string()),
        None => request.call(),
    };
    match response {
        Ok(response) => response.status(),
        Err(ureq::Error::Status(status, _)) => status,
        Err(error) => panic!("{method} {path} failed: {error}"),
    }
}

/// Spawns the binary with JSON logs on a piped stderr, runs `drive`,
/// stops the server gracefully, and returns every stderr line that
/// parsed as JSON. The child has exited before the scan, so an absent
/// line is a real absence, not a race.
fn json_log_session(tag: &str, extra_env: &[(&str, &str)], drive: impl FnOnce(&str)) -> Vec<Value> {
    let data_dir = std::env::temp_dir().join(format!("taguru-log-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    command
        .env("TAGURU_ADDR", "127.0.0.1:0")
        .env("TAGURU_DATA_DIR", &data_dir)
        .env("TAGURU_LOG_FORMAT", "json")
        .env_remove("TAGURU_EMBED_URL")
        .env_remove("TAGURU_API_TOKEN")
        .env_remove("OTEL_EXPORTER_OTLP_ENDPOINT")
        .env_remove("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        .env_remove("OTEL_EXPORTER_OTLP_PROTOCOL")
        .env_remove("TAGURU_LOG_SEARCHES")
        .env_remove("TAGURU_CONFIG");
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("server binary must spawn");

    let stdout = child.stdout.take().expect("stdout must be piped");
    let mut stdout_lines = BufReader::new(stdout).lines();
    let base = loop {
        let line = stdout_lines
            .next()
            .expect("server must print its address")
            .expect("server stdout must be readable");
        if let Some(addr) = line.strip_prefix("listening on ") {
            break format!("http://{addr}");
        }
    };
    // Drain stderr concurrently: a session can log more than one pipe
    // buffer holds, and a full pipe blocks the server's workers.
    let stderr = child.stderr.take().expect("stderr must be piped");
    let reader = std::thread::spawn(move || {
        BufReader::new(stderr)
            .lines()
            .map_while(Result::ok)
            .filter_map(|line| serde_json::from_str::<Value>(&line).ok())
            .collect::<Vec<Value>>()
    });

    drive(&base);

    Command::new("kill")
        .args(["-TERM", &child.id().to_string()])
        .status()
        .expect("kill must run");
    let _ = child.wait();
    let lines = reader.join().expect("stderr reader must finish");
    let _ = std::fs::remove_dir_all(&data_dir);
    lines
}

#[test]
fn search_events_carry_cue_and_hits_when_opted_in() {
    let lines = json_log_session("searchlog-on", &[("TAGURU_LOG_SEARCHES", "1")], |base| {
        assert_eq!(
            raw_call(
                base,
                "PUT",
                "/contexts/s",
                Some(json!({"description": "d"}))
            ),
            200
        );
        assert_eq!(
            raw_call(
                base,
                "POST",
                "/contexts/s/associations",
                Some(json!([{
                    "subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺",
                    "weight": 1.0, "source": "p1"
                }]))
            ),
            200
        );
        assert_eq!(
            raw_call(
                base,
                "POST",
                "/contexts/s/recall",
                Some(json!({"cue": "青嶺酒造"}))
            ),
            200
        );
        assert_eq!(
            raw_call(
                base,
                "POST",
                "/contexts/s/resolve",
                Some(json!({"cue": "qqqq"}))
            ),
            200
        );
    });

    let searches: Vec<&Value> = lines
        .iter()
        .filter(|line| line["target"] == "taguru::search")
        .collect();
    let recall = searches
        .iter()
        .find(|line| line["fields"]["op"] == "recall")
        .expect("a recall event must be logged");
    assert_eq!(recall["fields"]["context"], json!("s"), "{recall}");
    assert_eq!(recall["fields"]["cue"], json!("青嶺酒造"), "{recall}");
    assert_eq!(recall["fields"]["hits"], json!(1), "{recall}");
    let resolve = searches
        .iter()
        .find(|line| line["fields"]["op"] == "resolve")
        .expect("a resolve event must be logged");
    assert_eq!(resolve["fields"]["cue"], json!("qqqq"), "{resolve}");
    assert_eq!(resolve["fields"]["hits"], json!(0), "{resolve}");
    assert_eq!(resolve["fields"]["tier"], json!("miss"), "{resolve}");
}

#[test]
fn search_events_stay_absent_without_the_opt_in() {
    let lines = json_log_session("searchlog-off", &[], |base| {
        assert_eq!(
            raw_call(
                base,
                "PUT",
                "/contexts/s",
                Some(json!({"description": "d"}))
            ),
            200
        );
        assert_eq!(
            raw_call(
                base,
                "POST",
                "/contexts/s/recall",
                Some(json!({"cue": "秘匿の合い言葉"}))
            ),
            200
        );
    });

    // The stream is alive — access-log lines prove the scan saw real
    // output — yet carries no search events, and so no cue content.
    assert!(
        lines.iter().any(|line| line["fields"]["message"] == "http"),
        "expected access-log lines in the scanned stderr"
    );
    assert!(
        lines.iter().all(|line| line["target"] != "taguru::search"),
        "a search event leaked without the opt-in"
    );
    assert!(
        !lines
            .iter()
            .any(|line| line.to_string().contains("秘匿の合い言葉")),
        "cue content leaked into the default log stream"
    );
}

/// Runs `taguru import` against `data_dir`, hermetic the same way the
/// server spawns are: nothing from the developer shell reaches it.
fn run_import(data_dir: &std::path::Path, args: &[&str]) -> (i32, String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_taguru"))
        .arg("import")
        .env("TAGURU_DATA_DIR", data_dir)
        .env_remove("TAGURU_EMBED_URL")
        .env_remove("TAGURU_EMBED_MODEL")
        .env_remove("TAGURU_EMBED_AUTO")
        .env_remove("TAGURU_SEMANTIC_FLOOR")
        .env_remove("TAGURU_WAL")
        .env_remove("TAGURU_WAL_MAX_BYTES")
        .env_remove("TAGURU_CACHE_BYTES")
        .env_remove("TAGURU_CONFIG")
        .args(args)
        .output()
        .expect("import must run");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// A scratch directory for batch files, separate from any data dir.
fn batch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("taguru-batches-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("batch dir must be creatable");
    dir
}

#[test]
fn an_offline_import_lands_facts_passage_and_aliases_the_server_serves() {
    let batches = batch_dir("import-serve");
    let file = batches.join("guide.jsonl");
    std::fs::write(
        &file,
        r#"{"taguru_batch": 1, "context": "sake", "source": "doc-guide", "create": {"description": "酒蔵の記憶"}}
{"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 2.0}
{"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0}
{"alias": "Aomine", "canonical": "青嶺酒造", "kind": "concept"}
{"passage": "青嶺酒造の杜氏は高瀬。1907年創業。"}
"#,
    )
    .unwrap();

    let data_dir =
        std::env::temp_dir().join(format!("taguru-http-import-serve-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
    let (code, stdout, stderr) = run_import(&data_dir, &[file.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("+2 association(s)"), "{stdout}");
    assert!(stdout.contains("(created)"), "{stdout}");

    // The server boots on the imported directory and serves it all:
    // the facts, the alias entry point, and the original passage.
    let server = Server::start_on("import-serve", data_dir);
    let brewer = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "Aomine", "label": "杜氏"})),
    );
    assert_eq!(brewer["matches"][0]["subject"], json!("青嶺酒造"));
    assert_eq!(brewer["matches"][0]["object"], json!("高瀬"));
    assert_eq!(brewer["matches"][0]["weight"], json!(2.0));
    let passages = server.ok(
        "POST",
        "/contexts/sake/sources/lookup",
        Some(json!({"sources": ["doc-guide"]})),
    );
    assert_eq!(
        passages["passages"]["doc-guide"],
        json!("青嶺酒造の杜氏は高瀬。1907年創業。")
    );
    let _ = std::fs::remove_dir_all(&batches);
}

/// `apply_batch`'s question bookkeeping (src/ingest.rs) had no coverage
/// of its own — only the direct `POST .../sources` path (see
/// `store_passages_accepts_questions_and_reports_the_bookkeeping`) was
/// exercised. This drives the same stored/dropped counters through a
/// batch file and confirms a stored question actually rides the
/// passage into the search index under the paragraph it names.
#[test]
fn an_offline_import_carries_questions_through_to_the_search_index() {
    let batches = batch_dir("import-questions");
    let file = batches.join("guide.jsonl");
    std::fs::write(
        &file,
        r#"{"taguru_batch": 1, "context": "sake", "source": "doc-guide", "create": {"description": "酒蔵の記憶"}}
{"passage": "青嶺酒造は1907年に創業した。\n\n杜氏は高瀬。"}
{"question": "杜氏は誰?", "paragraph": 1}
{"question": "存在しない段落への質問?", "paragraph": 9}
"#,
    )
    .unwrap();

    let data_dir = std::env::temp_dir().join(format!(
        "taguru-http-import-questions-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&data_dir);
    let (code, stdout, stderr) = run_import(&data_dir, &[file.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("+1 question(s) (1 dropped: no such paragraph)"),
        "{stdout}"
    );

    // The stored question rides the passage into the search index: a
    // query using the QUESTION's wording, not the paragraph's own,
    // still finds the paragraph it names.
    let server = Server::start_on("import-questions", data_dir);
    let hits = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "杜氏は誰?", "limit": 3})),
    );
    let hit = &hits[0];
    assert_eq!(hit["source"], "doc-guide");
    assert_eq!(hit["index"], 1, "the hit names the answering PARAGRAPH");
    let _ = std::fs::remove_dir_all(&batches);
}

#[test]
fn reimporting_a_source_replaces_it_instead_of_doubling() {
    let batches = batch_dir("import-idem");
    let file = batches.join("facts.jsonl");
    let header = r#"{"taguru_batch": 1, "context": "sake", "source": "doc-1", "create": {"description": "d"}}"#;
    std::fs::write(
        &file,
        format!(
            "{header}\n{}\n",
            r#"{"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 2.0}"#
        ),
    )
    .unwrap();

    let data_dir =
        std::env::temp_dir().join(format!("taguru-http-import-idem-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
    // Twice: the weight must not accumulate across identical imports.
    for _ in 0..2 {
        let (code, stdout, stderr) = run_import(&data_dir, &[file.to_str().unwrap()]);
        assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    }
    let server = Server::start_on("import-idem", data_dir);
    let edge = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "杜氏"})),
    );
    assert_eq!(edge["matches"][0]["weight"], json!(2.0));
    assert_eq!(
        edge["matches"][0]["attributions"].as_array().unwrap().len(),
        1
    );
    let data_dir = server.stop_gracefully();

    // A revised file for the same source: its truth replaces, never
    // stacks onto, the old one.
    std::fs::write(
        &file,
        format!(
            "{header}\n{}\n",
            r#"{"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 5.0}"#
        ),
    )
    .unwrap();
    let (code, _, stderr) = run_import(&data_dir, &[file.to_str().unwrap()]);
    assert_eq!(code, 0, "{stderr}");
    let server = Server::start_on("import-idem-2", data_dir);
    let edge = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "杜氏"})),
    );
    assert_eq!(edge["matches"][0]["weight"], json!(5.0));
    let _ = std::fs::remove_dir_all(&batches);
}

#[test]
fn import_refuses_a_data_directory_a_live_server_holds() {
    let server = Server::start("import-locked");
    let batches = batch_dir("import-locked");
    let file = batches.join("late.jsonl");
    std::fs::write(
        &file,
        "{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"s\", \"create\": {}}\n",
    )
    .unwrap();
    let (code, _, stderr) = run_import(&server.data_dir, &[file.to_str().unwrap()]);
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("another taguru process"), "{stderr}");
    let _ = std::fs::remove_dir_all(&batches);
}

#[test]
fn a_malformed_file_refuses_the_whole_import_before_any_write() {
    let batches = batch_dir("import-refuse");
    let good = batches.join("good.jsonl");
    std::fs::write(
        &good,
        "{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"s\", \"create\": {}}\n",
    )
    .unwrap();
    let bad = batches.join("bad.jsonl");
    std::fs::write(
        &bad,
        "{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"t\"}\n\n{\"foo\": 1}\n",
    )
    .unwrap();

    let data_dir =
        std::env::temp_dir().join(format!("taguru-http-import-refuse-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
    let (code, _, stderr) = run_import(&data_dir, &[good.to_str().unwrap(), bad.to_str().unwrap()]);
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("line 3"), "{stderr}");
    assert!(stderr.contains("nothing was applied"), "{stderr}");
    // Refused during validation: the good file was NOT applied either —
    // the data directory was never even created.
    assert!(!data_dir.exists(), "validation must not touch the disk");

    // The same holds for a clean --dry-run.
    let (code, stdout, stderr) = run_import(&data_dir, &["--dry-run", good.to_str().unwrap()]);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("dry run"), "{stdout}");
    assert!(!data_dir.exists(), "a dry run must not touch the disk");
    let _ = std::fs::remove_dir_all(&batches);
}

/// POST /import with a raw JSONL body (the batch is not a JSON
/// document, so the JSON helpers cannot carry it) and an optional
/// bearer token.
fn post_import(server: &Server, body: &str, token: Option<&str>) -> (u16, Value) {
    let mut request = ureq::AgentBuilder::new()
        .build()
        .request("POST", &format!("{}/import", server.base));
    if let Some(token) = token {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    finish(request.send_string(body), "POST", "/import")
}

#[test]
fn the_import_endpoint_applies_batches_to_a_live_server() {
    let server = Server::start_with_env("http-import", &[("TAGURU_API_TOKEN", "opskey")]);
    let batch = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-live\", \
                 \"create\": {\"description\": \"d\"}}\n\
                 {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 2.0}\n\
                 {\"passage\": \"蔵の杜氏は高瀬。\"}\n";

    // The endpoint sits behind bearer auth like any other write.
    let (status, _) = post_import(&server, batch, None);
    assert_eq!(status, 401);

    let (status, first) = post_import(&server, batch, Some("opskey"));
    assert_eq!(status, 200, "{first}");
    assert_eq!(first["result"]["created"], json!(true));
    assert_eq!(first["result"]["associations"], json!(1));
    assert_eq!(first["result"]["passage_stored"], json!(true));
    assert_eq!(first["result"]["retracted"], json!(0));

    // Same batch again: the source is replaced, not doubled — the
    // no-downtime spelling of the CLI's idempotency.
    let (status, second) = post_import(&server, batch, Some("opskey"));
    assert_eq!(status, 200, "{second}");
    assert_eq!(second["result"]["retracted"], json!(1));
    let (status, edge) = server.call_with_token(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "杜氏"})),
        Some("opskey"),
    );
    assert_eq!(status, 200);
    assert_eq!(edge["result"]["matches"][0]["weight"], json!(2.0));
}

#[test]
fn the_import_endpoint_refuses_with_the_cli_wording_and_api_statuses() {
    let server = Server::start("http-import-refuse");

    // Malformed op line: 400, named by line number.
    let (status, refusal) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"s\"}\n\n{\"foo\": 1}\n",
        None,
    );
    assert_eq!(status, 400);
    assert!(
        refusal["error"].as_str().unwrap().contains("line 3"),
        "{refusal}"
    );

    // Absent context, no create block: 404.
    let (status, refusal) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"ghost\", \"source\": \"s\"}\n",
        None,
    );
    assert_eq!(status, 404);
    assert!(
        refusal["error"].as_str().unwrap().contains("create block"),
        "{refusal}"
    );

    // Re-pointing an existing alias is the API's usual conflict: 409.
    let (status, _) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"s1\", \"create\": {}}\n\
         {\"subject\": \"X\", \"label\": \"l\", \"object\": \"Z\", \"weight\": 1.0}\n\
         {\"alias\": \"A\", \"canonical\": \"X\", \"kind\": \"concept\"}\n",
        None,
    );
    assert_eq!(status, 200);
    let (status, refusal) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"s2\"}\n\
         {\"subject\": \"Y\", \"label\": \"l\", \"object\": \"Z\", \"weight\": 1.0}\n\
         {\"alias\": \"A\", \"canonical\": \"Y\", \"kind\": \"concept\"}\n",
        None,
    );
    assert_eq!(status, 409, "{refusal}");
    assert!(
        refusal["error"].as_str().unwrap().contains("applied"),
        "{refusal}"
    );
}

#[test]
fn aliases_withdraw_and_the_spelling_is_reusable() {
    let server = Server::start("alias-remove");
    server.ok("PUT", "/contexts/c", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/c/associations",
        Some(json!([
            {"subject": "X", "label": "l", "object": "Z", "weight": 1.0},
            {"subject": "Y", "label": "l", "object": "Z", "weight": 1.0},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/c/aliases",
        Some(json!({"concepts": {"A": "X"}})),
    );

    let removed = server.ok(
        "DELETE",
        "/contexts/c/aliases",
        Some(json!({"concepts": ["A"]})),
    );
    assert_eq!(removed, json!(1));
    let listing = server.ok("GET", "/contexts/c/aliases", None);
    assert_eq!(listing["concepts"], json!({}));

    // The spelling is free to point elsewhere — the un-wedging move.
    server.ok(
        "POST",
        "/contexts/c/aliases",
        Some(json!({"concepts": {"A": "Y"}})),
    );
    let via = server.ok("POST", "/contexts/c/query", Some(json!({"subject": "A"})));
    assert_eq!(via["matches"][0]["subject"], json!("Y"));

    // Refusals: absent spellings and canonical names are conflicts,
    // and an empty withdrawal is malformed rather than a silent no-op.
    let (status, body) = server.call(
        "DELETE",
        "/contexts/c/aliases",
        Some(json!({"concepts": ["ghost"]})),
    );
    assert_eq!(status, 409, "{body}");
    let (status, _) = server.call(
        "DELETE",
        "/contexts/c/aliases",
        Some(json!({"concepts": ["X"]})),
    );
    assert_eq!(status, 409);
    let (status, _) = server.call("DELETE", "/contexts/c/aliases", Some(json!({})));
    assert_eq!(status, 400);
}

#[test]
fn an_import_alias_conflict_heals_with_a_withdrawal_then_reimport() {
    let server = Server::start("alias-heal");
    let (status, _) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"s1\", \"create\": {}}\n\
         {\"subject\": \"X\", \"label\": \"l\", \"object\": \"Z\", \"weight\": 1.0}\n\
         {\"alias\": \"A\", \"canonical\": \"X\", \"kind\": \"concept\"}\n",
        None,
    );
    assert_eq!(status, 200);
    let revised = "{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"s2\"}\n\
         {\"subject\": \"Y\", \"label\": \"l\", \"object\": \"Z\", \"weight\": 1.0}\n\
         {\"alias\": \"A\", \"canonical\": \"Y\", \"kind\": \"concept\"}\n";
    let (status, _) = post_import(&server, revised, None);
    assert_eq!(status, 409);

    // The heal the import docs prescribe: withdraw the old
    // registration deliberately, then re-import — retract-then-apply
    // makes the second attempt exact.
    server.ok(
        "DELETE",
        "/contexts/c/aliases",
        Some(json!({"concepts": ["A"]})),
    );
    let (status, body) = post_import(&server, revised, None);
    assert_eq!(status, 200, "{body}");
    let listing = server.ok("GET", "/contexts/c/aliases", None);
    assert_eq!(listing["concepts"]["A"], json!("Y"));
}

#[test]
fn importing_into_an_absent_context_needs_a_create_block() {
    let batches = batch_dir("import-nocreate");
    let file = batches.join("orphan.jsonl");
    std::fs::write(
        &file,
        "{\"taguru_batch\": 1, \"context\": \"ghost\", \"source\": \"s\"}\n",
    )
    .unwrap();
    let data_dir = std::env::temp_dir().join(format!(
        "taguru-http-import-nocreate-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&data_dir);
    let (code, _, stderr) = run_import(&data_dir, &[file.to_str().unwrap()]);
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("no create block"), "{stderr}");
    let _ = std::fs::remove_dir_all(&batches);
    let _ = std::fs::remove_dir_all(&data_dir);
}

/// A one-shot OpenAI-compatible chat stub: answers the canned
/// assistant texts in order, one connection per request, then hands
/// back every captured request (headers + body) through the join.
fn stub_chat_server(replies: Vec<String>) -> (String, std::thread::JoinHandle<Vec<String>>) {
    use std::io::{Read, Write};

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        let mut captured = Vec::new();
        for reply in replies {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buffer = Vec::new();
            let mut chunk = [0u8; 8192];
            let header_end = loop {
                let read = stream.read(&mut chunk).unwrap();
                if read == 0 {
                    break buffer.len();
                }
                buffer.extend_from_slice(&chunk[..read]);
                if let Some(position) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
                    break position + 4;
                }
            };
            let headers = String::from_utf8_lossy(&buffer[..header_end]).to_string();
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())?
                })
                .unwrap_or(0);
            while buffer.len() < header_end + content_length {
                let read = stream.read(&mut chunk).unwrap();
                if read == 0 {
                    break;
                }
                buffer.extend_from_slice(&chunk[..read]);
            }
            captured.push(format!(
                "{headers}\n{}",
                String::from_utf8_lossy(&buffer[header_end..])
            ));
            let payload = json!({
                "choices": [{"message": {"role": "assistant", "content": reply}}]
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                payload.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
        }
        captured
    });
    (url, handle)
}

/// Runs `taguru extract`, hermetic like the other spawns: only the
/// given TAGURU_EXTRACT_* values reach it.
fn run_extract(
    out_dir: &std::path::Path,
    env: &[(&str, &str)],
    args: &[&str],
) -> (i32, String, String) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    command
        .arg("extract")
        .env_remove("TAGURU_EXTRACT_URL")
        .env_remove("TAGURU_EXTRACT_MODEL")
        .env_remove("TAGURU_EXTRACT_API_KEY")
        .env_remove("TAGURU_EXTRACT_TIMEOUT_SECS")
        .env_remove("TAGURU_CONFIG");
    for (key, value) in env {
        command.env(key, value);
    }
    let output = command
        .args(["--out", out_dir.to_str().unwrap()])
        .args(args)
        .output()
        .expect("extract must run");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn extraction_turns_documents_into_batches_import_applies_and_the_server_serves() {
    let docs = batch_dir("extract-docs");
    let aomine = docs.join("aomine.md");
    let takase = docs.join("takase.md");
    std::fs::write(
        &aomine,
        "青嶺酒造は1907年に創業した。\n\n杜氏は高瀬。大量生産は行わない。",
    )
    .unwrap();
    std::fs::write(&takase, "高瀬は青嶺酒造の杜氏。").unwrap();
    let aomine_src = aomine.to_str().unwrap();
    let takase_src = takase.to_str().unwrap();

    // Dry run: no provider configured, nothing called, nothing written.
    let out = batch_dir("extract-out");
    let (code, stdout, stderr) = run_extract(
        &out,
        &[],
        &["--dry-run", "--context", "sake", aomine_src, takase_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(stdout.matches("would extract").count(), 2, "{stdout}");

    // The real run. aomine answers fenced (the extractor must strip
    // markdown fences) and carries one duplicate triple, one alias
    // whose canonical exists nowhere, and one null-valued item — real
    // models emit all three. takase answers garbage first — one
    // corrective turn — then a valid object with weight omitted.
    let aomine_reply = json!({
        "associations": [
            {"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0},
            {"subject": "青嶺酒造", "label": "行う", "object": "大量生産", "weight": -1.0},
            {"subject": "青嶺酒造", "label": "所在地", "object": null},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0}
        ],
        "aliases": [
            {"alias": "Aomine", "canonical": "青嶺酒造", "kind": "concept"},
            {"alias": "幽霊", "canonical": "存在しない", "kind": "concept"}
        ]
    })
    .to_string();
    let takase_reply =
        json!({"associations": [{"subject": "高瀬", "label": "所属", "object": "青嶺酒造"}]})
            .to_string();
    let (url, requests) = stub_chat_server(vec![
        format!("```json\n{aomine_reply}\n```"),
        "Sure! Here are the facts I found.".to_string(),
        takase_reply.clone(),
    ]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ("TAGURU_EXTRACT_API_KEY", "sekrit"),
    ];
    let (code, stdout, stderr) = run_extract(
        &out,
        &provider,
        &[
            "--context",
            "sake",
            "--description",
            "酒蔵の記憶",
            aomine_src,
            takase_src,
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("3 association(s)"), "{stdout}");
    assert!(stdout.contains("1 duplicate(s) folded"), "{stdout}");
    assert!(stdout.contains("2 item(s) dropped"), "{stdout}");
    assert!(stdout.contains("2 written"), "{stdout}");

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 3);
    assert!(requests[0].contains("Bearer sekrit"), "{}", requests[0]);
    assert!(
        requests[0].contains("青嶺酒造は1907年に創業した。"),
        "{}",
        requests[0]
    );
    // The second document's prompt carries the first document's labels…
    assert!(
        requests[1].contains("創業年"),
        "vocabulary did not accumulate: {}",
        requests[1]
    );
    // …and the corrective turn asks again after the garbage answer.
    assert!(
        requests[2].contains("only the JSON object"),
        "{}",
        requests[2]
    );

    // Import applies what extract wrote; the server serves the facts,
    // the alias entry, the negative weight, and the original passage.
    let data_dir = std::env::temp_dir().join(format!("taguru-http-extract-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
    let (code, stdout, stderr) = run_import(&data_dir, &[out.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let server = Server::start_on("extract-serve", data_dir);
    let brewer = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "Aomine", "label": "杜氏"})),
    );
    assert_eq!(brewer["matches"][0]["object"], json!("高瀬"));
    let negated = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "行う"})),
    );
    assert_eq!(negated["matches"][0]["weight"], json!(-1.0));
    let membership = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "高瀬", "label": "所属"})),
    );
    assert_eq!(membership["matches"][0]["weight"], json!(1.0));
    let passages = server.ok(
        "POST",
        "/contexts/sake/sources/lookup",
        Some(json!({"sources": [aomine_src]})),
    );
    assert_eq!(
        passages["passages"][aomine_src],
        json!("青嶺酒造は1907年に創業した。\n\n杜氏は高瀬。大量生産は行わない。")
    );
    drop(server);

    // Unchanged documents skip without a single model call: the
    // endpoint here refuses every connection, so an attempt would fail
    // loudly instead of passing.
    let dead = [
        ("TAGURU_EXTRACT_URL", "http://127.0.0.1:9"),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) =
        run_extract(&out, &dead, &["--context", "sake", aomine_src, takase_src]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(stdout.matches("unchanged, skipped").count(), 2, "{stdout}");

    // --force re-extracts both.
    let (url, requests) = stub_chat_server(vec![aomine_reply.clone(), takase_reply.clone()]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) = run_extract(
        &out,
        &provider,
        &["--force", "--context", "sake", aomine_src, takase_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("2 written"), "{stdout}");
    assert_eq!(requests.join().unwrap().len(), 2);

    // A re-pointed --context re-extracts too — a skip would leave
    // files whose headers still send everything to 'sake'.
    let (url, requests) = stub_chat_server(vec![aomine_reply, takase_reply]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) = run_extract(
        &out,
        &provider,
        &["--context", "vats", aomine_src, takase_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(!stdout.contains("unchanged, skipped"), "{stdout}");
    assert!(stdout.contains("2 written"), "{stdout}");
    assert_eq!(requests.join().unwrap().len(), 2);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn the_extract_timeout_knob_bounds_a_stalled_provider() {
    let docs = batch_dir("extract-stall-docs");
    let doc = docs.join("slow.md");
    std::fs::write(&doc, "content").unwrap();

    // A provider that accepts and never answers — the local-model
    // failure mode (a thinking model grinding away) as seen from the
    // client. Both attempts' connections are held open, unanswered.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let mut held = Vec::new();
        for _ in 0..2 {
            if let Ok((stream, _)) = listener.accept() {
                held.push(stream);
            }
        }
        std::thread::sleep(std::time::Duration::from_secs(15));
    });

    let out = batch_dir("extract-stall-out");
    let started = std::time::Instant::now();
    let (code, _, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_TIMEOUT_SECS", "1"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("timed out"), "{stderr}");
    // Two 1-second attempts plus the retry pause — nowhere near the
    // 300-second default this knob overrides.
    assert!(
        started.elapsed() < std::time::Duration::from_secs(30),
        "took {:?}",
        started.elapsed()
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// The hybrid-search wire contract on a lexical-only server: hits are
/// paragraph-granular, every hit carries its lane evidence, the
/// top-level score stays the raw BM25 number (no semantic lane ran),
/// and the vector key is absent rather than null.
#[test]
fn passage_search_serves_paragraph_hits_with_lane_evidence() {
    let server = Server::start("passage-lanes");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "蔵の知識"})),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {
            "docs/aomine.md": "青嶺酒造は雲居県霧沢町の蔵元である。\n\n\
                原料米には山田錦を使い、精米歩合は50パーセントまで磨く。"
        }})),
    );

    let hits = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "精米歩合はどこまで磨く?", "limit": 3})),
    );
    let hit = &hits[0];
    assert_eq!(hit["source"], "docs/aomine.md");
    assert_eq!(hit["index"], 1, "the hit names the answering PARAGRAPH");
    assert!(hit["text"].as_str().unwrap().starts_with("原料米"), "{hit}");
    assert_eq!(hit["lanes"]["bm25"]["rank"], 1, "{hit}");
    assert_eq!(
        hit["score"], hit["lanes"]["bm25"]["score"],
        "lexical-only deployments keep raw BM25 score semantics"
    );
    assert!(
        hit["lanes"].get("vector").is_none(),
        "no provider, no vector key: {hit}"
    );

    // A zero limit asks for nothing and gets nothing.
    let none = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "精米", "limit": 0})),
    );
    assert_eq!(none.as_array().unwrap().len(), 0);
}

/// doc2query over HTTP: questions ride the store request per source,
/// out-of-range ones are dropped with their count reported (never
/// failing the passage), and questions for a source the request does
/// not carry are refused outright.
#[test]
fn store_passages_accepts_questions_and_reports_the_bookkeeping() {
    let server = Server::start("passage-questions");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "蔵の知識"})),
    );
    let result = server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {"doc": "一つ目。\n\n二つ目。"},
            "questions": {"doc": [
                {"paragraph": 1, "question": "二つ目は何?"},
                {"paragraph": 9, "question": "存在しない段落への質問?"}
            ]}
        })),
    );
    assert_eq!(result["stored"], 1, "{result}");
    assert_eq!(result["questions_stored"], 1, "{result}");
    assert_eq!(result["questions_dropped"], 1, "{result}");

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {},
            "questions": {"ghost": [{"paragraph": 0, "question": "誰の質問?"}]}
        })),
    );
    assert_eq!(status, 400, "{body}");
}
