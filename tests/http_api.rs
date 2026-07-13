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

    /// One MCP `tools/call` round trip: builds the JSON-RPC envelope,
    /// asserts HTTP 200, and hands back the JSON-RPC `result` — whose
    /// `content`/`isError` the caller judges.
    fn call_tool(&self, id: u64, name: &str, arguments: Value) -> Value {
        let (status, answer) = self.call(
            "POST",
            "/mcp",
            Some(json!({"jsonrpc": "2.0", "id": id, "method": "tools/call",
                        "params": {"name": name, "arguments": arguments}})),
        );
        assert_eq!(status, 200, "{name} -> {answer}");
        answer["result"].clone()
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
    let get_error_location = refused_get.header("location").unwrap().to_string();
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

    // A tool payload between axum's 2 MiB extractor default and the
    // operator's body cap (8 MiB here) must go through: the outer /mcp
    // request already paid the configured cap, and the in-process
    // dispatch is deliberately uncapped rather than silently re-capped
    // at the extractor default.
    let big = "a".repeat(3 * 1024 * 1024);
    let (status, stored) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 60, "method": "tools/call",
                    "params": {"name": "store_passages",
                               "arguments": {"context": "remote",
                                             "passages": {"big.md": big}}}})),
    );
    assert_eq!(status, 200);
    assert!(
        stored["result"].get("isError").is_none(),
        "a 3 MiB tool call must clear the 2 MiB extractor default: {}",
        stored["result"]["content"][0]["text"]
    );

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

/// cite_passage over tools/call: the MCP-layer counterpart of the
/// citation HTTP tests above, proving the manifest + dispatch wiring
/// (not just the HTTP handler) carries a request end to end.
#[test]
fn cite_passage_tool_executes_end_to_end_through_mcp() {
    let server = Server::start("mcp-citation");
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

    // Acceptance criterion 1: the tool is advertised in the manifest with
    // a schema matching #5's request shape, not just reachable by name.
    let (_, tools) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})),
    );
    let manifest = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tool| tool["name"] == "cite_passage")
        .expect("tools/list must advertise cite_passage");
    assert_eq!(
        manifest["inputSchema"]["required"],
        json!(["context", "source"])
    );
    assert!(manifest["inputSchema"]["properties"]["source"].is_object());
    assert!(manifest["inputSchema"]["properties"]["paragraph"].is_object());
    // Deprecated alias for pre-#35 callers: still advertised, so the
    // schema and `paragraph`/`index` `anyOf` requirement agree that
    // either name satisfies the call.
    assert!(manifest["inputSchema"]["properties"]["index"].is_object());
    assert_eq!(
        manifest["inputSchema"]["anyOf"],
        json!([{ "required": ["paragraph"] }, { "required": ["index"] }])
    );

    let (status, reply) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "cite_passage",
                               "arguments": {"context": "sake", "source": "docs/aomine.md", "paragraph": 1}}})),
    );
    assert_eq!(status, 200);
    assert!(reply["result"].get("isError").is_none(), "{reply}");
    let text = reply["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("\"source\":\"docs/aomine.md\""), "{text}");
    assert!(text.contains("\"section\":null"), "{text}");

    // Same failure convention as the `describe` case above: the tool
    // call still succeeds as JSON-RPC, but the result carries isError.
    let (status, failed) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "cite_passage",
                               "arguments": {"context": "sake", "source": "docs/ghost.md", "paragraph": 0}}})),
    );
    assert_eq!(status, 200);
    assert_eq!(failed["result"]["isError"], true);
    let error_text = failed["result"]["content"][0]["text"].as_str().unwrap();
    assert!(error_text.contains("docs/ghost.md"), "{error_text}");

    // Acceptance criterion 2: a pre-#35 caller still on `index` gets a
    // citation back through the full MCP path, not a schema rejection.
    let (status, via_index) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                    "params": {"name": "cite_passage",
                               "arguments": {"context": "sake", "source": "docs/aomine.md", "index": 1}}})),
    );
    assert_eq!(status, 200);
    assert!(via_index["result"].get("isError").is_none(), "{via_index}");
    let text = via_index["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("\"source\":\"docs/aomine.md\""), "{text}");
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
    let sources = server.ok("GET", "/contexts/sake/sources", None);
    assert_eq!(sources, json!({"total": 1, "sources": ["第2段落"]}));

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
    // Two sources each asserting 1.0 average to 1.0 — corroboration is
    // visible via count and the two attributions below, not via weight
    // alone.
    assert_eq!(water["matches"][0]["weight"], json!(1.0));
    assert_eq!(water["matches"][0]["count"], json!(2));
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
    assert_eq!(ranked["matches"][0]["association"]["label"], json!("杜氏"));
    assert_eq!(ranked["matches"][0]["path"], json!(["青嶺酒造"]));
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
    // top_concepts uses the same {label, count} object shape as describe's
    // as_subject/as_object, not a positional [name, count] tuple.
    assert_eq!(
        listed[0]["stats"]["top_concepts"][0],
        json!({"label": "青嶺酒造", "count": 4})
    );
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
    assert_eq!(body["code"], json!("unknown_path"), "{body}");
    assert!(body["time"].is_number(), "{body}");

    // Known path, wrong method → 405 in the error shape.
    let (status, body) = server.call("DELETE", "/contexts/sake/recall", None);
    assert_eq!(status, 405, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
    assert_eq!(body["code"], json!("method_not_allowed"), "{body}");

    // Malformed JSON on a JSON-required endpoint → 400 in shape.
    let (status, body) = server.call_raw(
        "POST",
        "/contexts/sake/recall",
        Some("{not json"),
        Some("application/json"),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
    assert_eq!(body["code"], json!("malformed_request"), "{body}");

    // Wrong media type → 415 in shape.
    let (status, body) = server.call_raw(
        "POST",
        "/contexts/sake/recall",
        Some("cue=x"),
        Some("application/x-www-form-urlencoded"),
    );
    assert_eq!(status, 415, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
    assert_eq!(body["code"], json!("malformed_request"), "{body}");

    // Well-formed JSON of the wrong type → 422 in shape.
    let (status, body) = server.call_raw(
        "POST",
        "/contexts/sake/recall",
        Some(r#"{"cue": 42}"#),
        Some("application/json"),
    );
    assert_eq!(status, 422, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
    assert_eq!(body["code"], json!("malformed_request"), "{body}");
}

/// Every JSON error carries the machine-readable `code` the protocol
/// documents — the stable branch key for clients that must not parse
/// message wording. One axis per assertion, domain refusals included.
#[test]
fn errors_carry_the_documented_machine_readable_code() {
    let server = Server::start_with_env(
        "errcodes",
        &[
            ("TAGURU_API_TOKEN", "codekey"),
            // Roomy enough for the over_limit body below (an origins
            // list is ~8 KiB), tight enough to trip on the 413 probe.
            ("TAGURU_MAX_BODY_BYTES", "16384"),
        ],
    );
    let code_of = |status: u16, body: &Value| -> (u16, String) {
        (
            status,
            body["code"].as_str().unwrap_or("<missing>").to_string(),
        )
    };

    // Missing bearer token → unauthorized.
    let (status, body) = server.call("GET", "/contexts", None);
    assert_eq!(
        code_of(status, &body),
        (401, "unauthorized".into()),
        "{body}"
    );

    let key = Some("codekey");
    let (status, body) = server.call_with_token("PUT", "/contexts/sake", Some(json!({})), key);
    assert_eq!(status, 200, "{body}");

    // PUT on an existing context → already_exists.
    let (status, body) = server.call_with_token("PUT", "/contexts/sake", Some(json!({})), key);
    assert_eq!(
        code_of(status, &body),
        (409, "already_exists".into()),
        "{body}"
    );

    // Unknown context → no_context.
    let (status, body) = server.call_with_token(
        "POST",
        "/contexts/ghost/recall",
        Some(json!({"cue": "x"})),
        key,
    );
    assert_eq!(code_of(status, &body), (404, "no_context".into()), "{body}");

    // A refused value (a weight the graph must never accumulate) →
    // invalid_argument.
    let (status, body) = server.call_with_token(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": "s", "label": "l", "object": "o", "weight": 1e300}])),
        key,
    );
    assert_eq!(
        code_of(status, &body),
        (400, "invalid_argument".into()),
        "{body}"
    );

    // A list-shaped input past its cap → over_limit (split and resend).
    let origins: Vec<String> = (0..1001).map(|index| format!("o{index}")).collect();
    let (status, body) = server.call_with_token(
        "POST",
        "/contexts/sake/activate",
        Some(json!({"origins": origins})),
        key,
    );
    assert_eq!(code_of(status, &body), (400, "over_limit".into()), "{body}");

    // No embedding provider configured → embeddings_unconfigured.
    let (status, body) =
        server.call_with_token("POST", "/contexts/sake/embeddings/refresh", None, key);
    assert_eq!(
        code_of(status, &body),
        (501, "embeddings_unconfigured".into()),
        "{body}"
    );

    // Unknown source on the citation endpoint → no_source; a stored
    // source with an out-of-range locator → no_paragraph.
    let (status, body) = server.call_with_token(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"doc": "本文。"}})),
        key,
    );
    assert_eq!(status, 200, "{body}");
    let (status, body) = server.call_with_token(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "ghost", "paragraph": 0})),
        key,
    );
    assert_eq!(code_of(status, &body), (404, "no_source".into()), "{body}");
    let (status, body) = server.call_with_token(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "doc", "paragraph": 9})),
        key,
    );
    assert_eq!(
        code_of(status, &body),
        (404, "no_paragraph".into()),
        "{body}"
    );

    // The body cap now answers in the SAME JSON shape as every other
    // axis (it used to be axum's plain text) — payload_too_large.
    let (status, body) = server.call_with_token(
        "PUT",
        "/contexts/big",
        Some(json!({"description": "こ".repeat(8000)})),
        key,
    );
    assert_eq!(
        code_of(status, &body),
        (413, "payload_too_large".into()),
        "{body}"
    );
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

    // An empty alias spelling is worse than unaddressable:
    // `str::contains("")` is always true, so once interned it would
    // containment-match every future cue as a phantom hit. Both roles,
    // both namespaces.
    for (name, request) in [
        (
            "empty concept alias",
            json!({"concepts": {"": "x"}, "labels": {}}),
        ),
        (
            "empty concept canonical",
            json!({"concepts": {"a": ""}, "labels": {}}),
        ),
        (
            "empty label alias",
            json!({"concepts": {}, "labels": {"": "x"}}),
        ),
        (
            "empty label canonical",
            json!({"concepts": {}, "labels": {"l": ""}}),
        ),
    ] {
        let (status, body) = server.call("POST", "/contexts/sake/aliases", Some(request));
        assert_eq!(status, 400, "{name}: {body}");
    }

    // A source that is PRESENT is a name like any other: empty would
    // intern a real, permanent source id that unrelated callers'
    // mistakes then silently merge into (and retract together).
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(json!([{"subject": "s", "label": "l", "object": "o", "weight": 1.0, "source": ""}])),
    );
    assert_eq!(status, 400, "{body}");
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("associations[0].source"),
        "{body}"
    );

    // The passage store keys sources the same way, and question and
    // section text is embedded verbatim on refresh — providers refuse
    // zero-length input, which would stall the refresh pass for good.
    for (name, request) in [
        ("empty passage source id", json!({"passages": {"": "text"}})),
        (
            "empty question",
            json!({
                "passages": {"doc.md": "text"},
                "questions": {"doc.md": [{"paragraph": 0, "question": ""}]},
            }),
        ),
        (
            "empty section",
            json!({
                "passages": {"doc.md": "text"},
                "sections": {"doc.md": [{"paragraph": 0, "section": ""}]},
            }),
        ),
    ] {
        let (status, body) = server.call("POST", "/contexts/sake/sources", Some(request));
        assert_eq!(status, 400, "{name}: {body}");
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
    let (status, body) = server.call(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "この説明は16バイトよりずっと長い"})),
    );
    assert_eq!(status, 413);
    // The cap breach speaks the one JSON error shape like every other
    // axis (it used to be axum's plain-text rejection).
    assert_eq!(body["status"], json!("error"), "{body}");
    assert_eq!(body["code"], json!("payload_too_large"), "{body}");
}

#[test]
fn a_custom_request_timeout_does_not_disturb_fast_requests() {
    // The deadline actually firing is unit-tested in limits.rs; this
    // pins the wiring — a tight budget must not break normal traffic.
    let server = Server::start_with_env("timeout", &[("TAGURU_REQUEST_TIMEOUT_SECS", "1")]);
    server.ok("PUT", "/contexts/sake", Some(json!({})));
    assert_eq!(server.call("GET", "/contexts", None).0, 200);
}

/// The wall-clock proof for the `block_in_place` deadline work: a
/// multi-batch import large enough that landing every batch takes far
/// longer than the configured budget must still answer in roughly the
/// budget's time, not in the time the whole loop would take to drain.
#[test]
fn a_tight_timeout_cuts_a_multi_batch_import_short_instead_of_running_it_to_completion() {
    const BATCH_COUNT: usize = 8_000;
    let server = Server::start_with_env("timeout-import", &[("TAGURU_REQUEST_TIMEOUT_SECS", "1")]);
    let mut stream = String::new();
    stream.push_str(
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-0\", \
         \"create\": {\"description\": \"d\"}}\n",
    );
    stream
        .push_str("{\"subject\": \"s0\", \"label\": \"l\", \"object\": \"o0\", \"weight\": 1.0}\n");
    for i in 1..BATCH_COUNT {
        stream.push_str(&format!(
            "{{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-{i}\"}}\n"
        ));
        stream.push_str(&format!(
            "{{\"subject\": \"s{i}\", \"label\": \"l\", \"object\": \"o{i}\", \"weight\": 1.0}}\n"
        ));
    }

    let started = std::time::Instant::now();
    let (status, body) = post_import(&server, &stream, None);
    let elapsed = started.elapsed();

    assert_eq!(status, 408, "{body}");
    assert_eq!(body["code"], json!("timeout"), "{body}");
    // Each fsync-bearing batch costs roughly 10ms, so draining all
    // 8,000 would take over a minute; answering near the 1-second
    // budget instead is the point of this test.
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "took {elapsed:?} — the deadline check inside the batch loop should have cut \
         this short instead of letting every batch land first"
    );
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
fn oversized_input_lists_are_refused_before_any_work() {
    let server = Server::start("input-caps");
    server.ok("PUT", "/contexts/caps", Some(json!({"description": "d"})));

    // 1001 items trips every list-shaped read input.
    let over: Vec<String> = (0..1001).map(|i| format!("o{i}")).collect();
    for (path, body) in [
        ("/contexts/caps/explore", json!({"origins": over.clone()})),
        ("/contexts/caps/activate", json!({"origins": over.clone()})),
        (
            "/contexts/caps/unreachable_from",
            json!({"origins": over.clone()}),
        ),
        ("/contexts/caps/query", json!({"subject": over.clone()})),
        (
            "/contexts/caps/sources/lookup",
            json!({"sources": over.clone()}),
        ),
    ] {
        let (status, parsed) = server.call("POST", path, Some(body));
        assert_eq!(status, 400, "{path}: {parsed}");
        assert!(
            parsed["error"]
                .as_str()
                .unwrap()
                .contains("per-request limit"),
            "{path}: {parsed}"
        );
    }

    // The cap itself still passes — it matches the largest page
    // list_sources serves, so a paged bulk workflow fits exactly.
    let at_cap: Vec<String> = (0..1000).map(|i| format!("o{i}")).collect();
    server.ok(
        "POST",
        "/contexts/caps/explore",
        Some(json!({"origins": at_cap})),
    );

    // Alias batches are WAL writes and share the association batch cap.
    let aliases: serde_json::Map<String, Value> = (0..10_001)
        .map(|i| (format!("a{i}"), json!("青嶺酒造")))
        .collect();
    let (status, parsed) = server.call(
        "POST",
        "/contexts/caps/aliases",
        Some(json!({"concepts": aliases})),
    );
    assert_eq!(status, 400, "{parsed}");
    assert!(
        parsed["error"]
            .as_str()
            .unwrap()
            .contains("per-request limit"),
        "{parsed}"
    );

    let removals: Vec<String> = (0..10_001).map(|i| format!("a{i}")).collect();
    let (status, parsed) = server.call(
        "DELETE",
        "/contexts/caps/aliases",
        Some(json!({"concepts": removals})),
    );
    assert_eq!(status, 400, "{parsed}");
    assert!(
        parsed["error"]
            .as_str()
            .unwrap()
            .contains("per-request limit"),
        "{parsed}"
    );
}

#[test]
fn queries_with_no_pinned_position_are_refused_but_one_field_is_enough() {
    let server = Server::start("empty-query");
    server.ok(
        "PUT",
        "/contexts/empty-query",
        Some(json!({"description": "d"})),
    );

    // Single-context query: omitting subject/label/object entirely
    // would otherwise materialize and rank every edge in the context.
    let (status, parsed) = server.call("POST", "/contexts/empty-query/query", Some(json!({})));
    assert_eq!(status, 400, "{parsed}");
    assert!(
        parsed["error"]
            .as_str()
            .unwrap()
            .contains("must pin at least one value"),
        "{parsed}"
    );

    // Explicit nulls are indistinguishable from omission.
    let (status, parsed) = server.call(
        "POST",
        "/contexts/empty-query/query",
        Some(json!({"subject": null, "label": null, "object": null})),
    );
    assert_eq!(status, 400, "{parsed}");
    assert!(
        parsed["error"]
            .as_str()
            .unwrap()
            .contains("must pin at least one value"),
        "{parsed}"
    );

    // Pinning just one of the three is enough to pass, even with no
    // matches to return.
    server.ok(
        "POST",
        "/contexts/empty-query/query",
        Some(json!({"subject": "x"})),
    );

    // The cross-context route refuses the same way...
    let (status, parsed) =
        server.call("POST", "/query", Some(json!({"contexts": ["empty-query"]})));
    assert_eq!(status, 400, "{parsed}");
    assert!(
        parsed["error"]
            .as_str()
            .unwrap()
            .contains("must pin at least one value"),
        "{parsed}"
    );

    // ...nulls fold into the same refusal there too...
    let (status, parsed) = server.call(
        "POST",
        "/query",
        Some(json!({
            "contexts": ["empty-query"],
            "subject": null,
            "label": null,
            "object": null
        })),
    );
    assert_eq!(status, 400, "{parsed}");
    assert!(
        parsed["error"]
            .as_str()
            .unwrap()
            .contains("must pin at least one value"),
        "{parsed}"
    );

    // ...and one pinned field passes cross-context too.
    server.ok(
        "POST",
        "/query",
        Some(json!({"contexts": ["empty-query"], "object": "x"})),
    );

    // The refusal reaches through the MCP tool-call path as well.
    let reply = server.call_tool(1, "query", json!({"context": "empty-query"}));
    assert_eq!(reply["isError"], true, "{reply}");
    let error_text = reply["content"][0]["text"].as_str().unwrap();
    assert!(
        error_text.contains("must pin at least one value"),
        "{error_text}"
    );
}

#[test]
fn resolve_caps_its_candidate_flood_like_every_other_match_endpoint() {
    let server = Server::start("resolve-cap");
    server.ok("PUT", "/contexts/flood", Some(json!({"description": "d"})));

    // 1001 concepts all containing the cue: uncapped, resolve would
    // serve every one of them in a single response body.
    let batch: Vec<Value> = (0..1001)
        .map(|i| {
            json!({
                "subject": format!("concept{i:04}"),
                "label": "は",
                "object": "x",
                "weight": 1.0
            })
        })
        .collect();
    server.ok("POST", "/contexts/flood/associations", Some(json!(batch)));

    // The cue is more than half of every stored spelling, so entry is
    // confident-lexical — no semantic tier, hermetic here. The ceiling
    // holds even with no limit in the request.
    let served = server.ok(
        "POST",
        "/contexts/flood/resolve",
        Some(json!({"cue": "concept"})),
    );
    assert_eq!(
        served.as_array().unwrap().len(),
        1000,
        "the default is the ceiling, not the whole vocabulary"
    );

    // An explicit limit picks the page size; best-first survives.
    let five = server.ok(
        "POST",
        "/contexts/flood/resolve",
        Some(json!({"cue": "concept", "limit": 5})),
    );
    assert_eq!(five.as_array().unwrap().len(), 5, "{five}");
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
    // The registry groups unreachable_from with the association reads
    // above; the usage counters must agree. Zero orphans is the audit
    // succeeding, so it counts as a read but never as an empty one.
    server.ok(
        "POST",
        "/contexts/used/unreachable_from",
        Some(json!({"origins": ["青嶺酒造"]})),
    );

    let used = server.ok("GET", "/contexts/used", None);
    assert_eq!(used["usage"]["writes"], json!(1), "{used}");
    assert_eq!(used["usage"]["reads"], json!(4), "{used}");
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

/// TAGURU_KEY_SCOPES end to end: roles gate the verbs, context grants
/// gate the objects, the directory shows a scoped key only its world,
/// import checks its body-carried contexts, and an MCP tool call is
/// judged exactly as the route it dispatches onto.
#[test]
fn key_scopes_gate_roles_contexts_the_directory_and_mcp() {
    let server = Server::start_with_env(
        "http-scopes",
        &[
            (
                "TAGURU_API_TOKENS",
                "boss:atok,reader:rtok,scribe:wtok,potter:stok,curator:ctok",
            ),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"reader": "read", "scribe": "write", "potter": {"role": "write", "contexts": ["sake"]}, "curator": {"role": "admin", "contexts": ["sake"]}}"#,
            ),
        ],
    );
    let call = |method: &str, path: &str, body: Option<Value>, token: &str| {
        server.call_with_token(method, path, body, Some(token))
    };
    let fact = json!([{"subject": "蔵", "label": "杜氏", "object": "高瀬",
                      "weight": 1.0, "source": "a.md"}]);

    // The unscoped key keeps the historical full grant: admin, everywhere.
    assert_eq!(
        call(
            "PUT",
            "/contexts/sake",
            Some(json!({"description": "d"})),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "PUT",
            "/contexts/bunko",
            Some(json!({"description": "d"})),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "POST",
            "/contexts/sake/associations",
            Some(fact.clone()),
            "atok"
        )
        .0,
        200
    );

    // Read: the retrieval loop answers, the ingest loop refuses.
    assert_eq!(
        call(
            "POST",
            "/contexts/sake/recall",
            Some(json!({"cue": "蔵"})),
            "rtok"
        )
        .0,
        200
    );
    let (status, refusal) = call(
        "POST",
        "/contexts/sake/associations",
        Some(fact.clone()),
        "rtok",
    );
    assert_eq!(status, 403, "{refusal}");
    assert!(
        refusal["error"].as_str().unwrap().contains("needs 'write'"),
        "{refusal}"
    );
    assert_eq!(call("DELETE", "/contexts/sake", None, "rtok").0, 403);

    // Write: ingest yes, operator verbs no.
    assert_eq!(
        call(
            "POST",
            "/contexts/bunko/associations",
            Some(fact.clone()),
            "wtok"
        )
        .0,
        200
    );
    assert_eq!(call("DELETE", "/contexts/bunko", None, "wtok").0, 403);
    assert_eq!(call("POST", "/flush", None, "wtok").0, 403);

    // Flush is server-wide (it names every flushed context), so a
    // context-scoped key is refused even at admin role — the refusal
    // is the CONTEXT bypass guard, not the role check (curator IS
    // admin). The unscoped admin flushes normally.
    let (status, scoped_flush) = call("POST", "/flush", None, "ctok");
    assert_eq!(status, 403, "{scoped_flush}");
    assert!(
        scoped_flush["error"]
            .as_str()
            .unwrap()
            .contains("server-wide"),
        "{scoped_flush}"
    );
    assert_eq!(call("POST", "/flush", None, "atok").0, 200);

    // Context-scoped write: inside the grant yes, outside no — and the
    // directory shows only the granted world.
    assert_eq!(
        call(
            "POST",
            "/contexts/sake/associations",
            Some(fact.clone()),
            "stok"
        )
        .0,
        200
    );
    let (status, outside) = call("POST", "/contexts/bunko/associations", Some(fact), "stok");
    assert_eq!(status, 403);
    assert!(
        outside["error"]
            .as_str()
            .unwrap()
            .contains("no grant on context 'bunko'"),
        "{outside}"
    );
    assert_eq!(call("GET", "/contexts/bunko", None, "stok").0, 403);
    let (status, listed) = call("GET", "/contexts", None, "stok");
    assert_eq!(status, 200);
    assert_eq!(listed["result"]["total"], json!(1), "{listed}");
    assert_eq!(
        listed["result"]["contexts"][0]["name"],
        json!("sake"),
        "{listed}"
    );

    // Import carries its contexts in the body; the grant is checked
    // batch by batch before anything applies. (Import itself is an
    // admin verb, so even the granted context refuses for a writer.)
    let batch = "{\"taguru_batch\": 1, \"context\": \"bunko\", \"source\": \"s\"}\n";
    let (status, _) = post_import(&server, batch, Some("stok"));
    assert_eq!(status, 403);
    let (status, scoped_admin) = post_import(&server, batch, Some("atok"));
    assert_eq!(status, 200, "{scoped_admin}");

    // The body-carried-context refusal: curator is admin (clears the
    // role gate) but scoped to sake, so an out-of-grant bunko batch is
    // refused by the per-batch check, and an in-grant sake batch lands.
    let (status, out_of_grant) = post_import(&server, batch, Some("ctok"));
    assert_eq!(status, 403, "{out_of_grant}");
    assert!(
        out_of_grant["error"]
            .as_str()
            .unwrap()
            .contains("no grant on context 'bunko'"),
        "{out_of_grant}"
    );
    let sake_batch = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s\"}\n";
    let (status, in_grant) = post_import(&server, sake_batch, Some("ctok"));
    assert_eq!(status, 200, "{in_grant}");

    // MCP tool calls are judged as the routes they land on: the read
    // key's add_associations dispatch refuses with the same 403.
    let (status, reply) = server.call_with_token(
        "POST",
        "/mcp",
        Some(json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "add_associations", "arguments": {
                "context": "sake",
                "associations": [{"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0}],
            }},
        })),
        Some("rtok"),
    );
    assert_eq!(status, 200, "{reply}");
    assert_eq!(reply["result"]["isError"], json!(true), "{reply}");
    assert!(
        reply["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("403"),
        "{reply}"
    );
    // ...and a permitted tool call still works through the same key.
    let (status, reply) = server.call_with_token(
        "POST",
        "/mcp",
        Some(json!({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "recall", "arguments": {"context": "sake", "cue": "蔵"}},
        })),
        Some("rtok"),
    );
    assert_eq!(status, 200);
    assert_ne!(reply["result"]["isError"], json!(true), "{reply}");
    assert!(
        reply["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("matches"),
        "{reply}"
    );
}

/// Cross-context search: one request over several full context names,
/// every match tagged with the context it came from. recall/query
/// merge on |weight| (one scale across contexts), passage hits
/// interleave by per-context rank, duplicate targets dedupe, and the
/// refusals — empty list, missing name, over-cap list — land before
/// anything is searched. The MCP search tools ride the same routes
/// through their `contexts` argument.
#[test]
fn cross_context_search_merges_tagged_matches_across_named_contexts() {
    let server = Server::start("cross-search");
    for (name, fact) in [
        (
            "izakaya",
            json!([{"subject": "蔵", "label": "名物", "object": "燗酒",
                    "weight": 0.5, "source": "iz.md"}]),
        ),
        (
            "sakagura",
            json!([{"subject": "蔵", "label": "杜氏", "object": "高瀬",
                    "weight": 1.0, "source": "sk.md"}]),
        ),
        // Never named in a target list below — must never leak in.
        (
            "noise",
            json!([{"subject": "蔵", "label": "場所", "object": "港",
                    "weight": 2.0, "source": "no.md"}]),
        ),
    ] {
        server.ok("PUT", &format!("/contexts/{name}"), Some(json!({})));
        server.ok(
            "POST",
            &format!("/contexts/{name}/associations"),
            Some(fact),
        );
    }

    // recall across two of the three: both matches, each tagged.
    let recalled = server.ok(
        "POST",
        "/recall",
        Some(json!({"contexts": ["izakaya", "sakagura"], "cue": "蔵"})),
    );
    assert_eq!(recalled["total"], json!(2), "{recalled}");
    let tag_of = |matches: &Value, object: &str| -> String {
        matches
            .as_array()
            .unwrap()
            .iter()
            .find(|found| found["object"] == json!(object))
            .unwrap_or_else(|| panic!("no match with object {object}: {matches}"))["context"]
            .as_str()
            .unwrap()
            .to_string()
    };
    assert_eq!(tag_of(&recalled["matches"], "高瀬"), "sakagura");
    assert_eq!(tag_of(&recalled["matches"], "燗酒"), "izakaya");

    // Past the limit the strongest |weight| survives, across contexts.
    let cut = server.ok(
        "POST",
        "/recall",
        Some(json!({"contexts": ["izakaya", "sakagura"], "cue": "蔵", "limit": 1})),
    );
    assert_eq!(cut["total"], json!(2), "{cut}");
    assert_eq!(cut["matches"].as_array().unwrap().len(), 1, "{cut}");
    assert_eq!(cut["matches"][0]["context"], json!("sakagura"), "{cut}");

    // Naming a context twice is redundant, not double.
    let deduped = server.ok(
        "POST",
        "/recall",
        Some(json!({"contexts": ["izakaya", "izakaya", "sakagura"], "cue": "蔵"})),
    );
    assert_eq!(deduped["total"], json!(2), "{deduped}");

    // query: position-pinned, same tagging contract.
    let queried = server.ok(
        "POST",
        "/query",
        Some(json!({"contexts": ["izakaya", "sakagura"], "label": "杜氏"})),
    );
    assert_eq!(queried["total"], json!(1), "{queried}");
    assert_eq!(queried["matches"][0]["context"], json!("sakagura"));
    assert_eq!(queried["matches"][0]["object"], json!("高瀬"));

    // The text lane: hits carry their context and interleave by
    // per-context rank — both rank-0 hits lead, in target-list order.
    server.ok(
        "POST",
        "/contexts/izakaya/sources",
        Some(json!({"passages": {"iz.md": "蔵元の燗酒は冬の名物。"}})),
    );
    server.ok(
        "POST",
        "/contexts/sakagura/sources",
        Some(json!({"passages": {"sk.md": "杜氏の高瀬は蔵元を任されている。"}})),
    );
    let hits = server.ok(
        "POST",
        "/sources/search",
        Some(json!({"contexts": ["izakaya", "sakagura"], "query": "蔵元"})),
    );
    let hits = hits.as_array().unwrap();
    assert_eq!(hits.len(), 2, "both contexts must answer: {hits:?}");
    assert_eq!(hits[0]["context"], json!("izakaya"), "{hits:?}");
    assert_eq!(hits[1]["context"], json!("sakagura"), "{hits:?}");
    assert_eq!(hits[0]["source"], json!("iz.md"), "{hits:?}");

    // Refusals, each before anything is searched.
    let (status, empty) = server.call(
        "POST",
        "/recall",
        Some(json!({"contexts": [], "cue": "蔵"})),
    );
    assert_eq!(status, 400, "{empty}");
    assert_eq!(empty["code"], json!("invalid_argument"), "{empty}");

    let (status, missing) = server.call(
        "POST",
        "/recall",
        Some(json!({"contexts": ["izakaya", "ghost"], "cue": "蔵"})),
    );
    assert_eq!(status, 404, "{missing}");
    assert_eq!(missing["code"], json!("no_context"), "{missing}");
    assert!(
        missing["error"].as_str().unwrap().contains("'ghost'"),
        "{missing}"
    );

    let flood: Vec<String> = (0..1001).map(|i| format!("c{i}")).collect();
    let (status, over) = server.call(
        "POST",
        "/query",
        Some(json!({"contexts": flood, "label": "l"})),
    );
    assert_eq!(status, 400, "{over}");
    assert_eq!(over["code"], json!("over_limit"), "{over}");

    // The MCP search tools take `contexts` as the cross-context form…
    let (status, reply) = server.call(
        "POST",
        "/mcp",
        Some(json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "recall", "arguments": {
                "contexts": ["izakaya", "sakagura"], "cue": "蔵",
            }},
        })),
    );
    assert_eq!(status, 200, "{reply}");
    assert_ne!(reply["result"]["isError"], json!(true), "{reply}");
    let text = reply["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("sakagura") && text.contains("izakaya"),
        "{text}"
    );

    // …and refuse the ambiguous both-at-once form.
    let (status, reply) = server.call(
        "POST",
        "/mcp",
        Some(json!({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "recall", "arguments": {
                "context": "izakaya", "contexts": ["sakagura"], "cue": "蔵",
            }},
        })),
    );
    assert_eq!(status, 200, "{reply}");
    assert_eq!(reply["result"]["isError"], json!(true), "{reply}");
    assert!(
        reply["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("not both"),
        "{reply}"
    );
}

/// Cross-context search by group: a `groups` name searches every
/// context it reaches — nested children included — and combines with
/// `contexts`, overlaps deduped, so a context is searched once however
/// many ways it was named. Directly named contexts lead the tie order
/// and group-resolved members follow in name order; an unknown group
/// is `no_group`, an empty resolution is an empty result, and the MCP
/// search tools take `groups` beside `contexts`.
#[test]
fn cross_context_search_resolves_groups_beside_contexts() {
    let server = Server::start("cross-groups");
    for (name, fact) in [
        (
            "izakaya",
            json!([{"subject": "蔵", "label": "名物", "object": "燗酒",
                    "weight": 0.5, "source": "iz.md"}]),
        ),
        (
            "sakagura",
            json!([{"subject": "蔵", "label": "杜氏", "object": "高瀬",
                    "weight": 1.0, "source": "sk.md"}]),
        ),
    ] {
        server.ok("PUT", &format!("/contexts/{name}"), Some(json!({})));
        server.ok(
            "POST",
            &format!("/contexts/{name}/associations"),
            Some(fact),
        );
    }
    // sakaya bundles izakaya; nomiya bundles sakagura and nests sakaya.
    server.ok(
        "PUT",
        "/groups/sakaya",
        Some(json!({"contexts": ["izakaya"]})),
    );
    server.ok(
        "PUT",
        "/groups/nomiya",
        Some(json!({"contexts": ["sakagura"], "groups": ["sakaya"]})),
    );

    // One group name reaches both contexts through the nesting.
    let recalled = server.ok(
        "POST",
        "/recall",
        Some(json!({"groups": ["nomiya"], "cue": "蔵"})),
    );
    assert_eq!(recalled["total"], json!(2), "{recalled}");

    // Naming a member directly AND through the group searches it once.
    let deduped = server.ok(
        "POST",
        "/recall",
        Some(json!({"contexts": ["izakaya"], "groups": ["nomiya", "sakaya"], "cue": "蔵"})),
    );
    assert_eq!(deduped["total"], json!(2), "{deduped}");

    // query rides the same resolution.
    let queried = server.ok(
        "POST",
        "/query",
        Some(json!({"groups": ["nomiya"], "label": "杜氏"})),
    );
    assert_eq!(queried["total"], json!(1), "{queried}");
    assert_eq!(queried["matches"][0]["context"], json!("sakagura"));

    // Passage rank ties: contexts named directly lead, group-resolved
    // members follow — sakagura outranks izakaya arriving via sakaya.
    server.ok(
        "POST",
        "/contexts/izakaya/sources",
        Some(json!({"passages": {"iz.md": "蔵元の燗酒は冬の名物。"}})),
    );
    server.ok(
        "POST",
        "/contexts/sakagura/sources",
        Some(json!({"passages": {"sk.md": "杜氏の高瀬は蔵元を任されている。"}})),
    );
    let hits = server.ok(
        "POST",
        "/sources/search",
        Some(json!({"contexts": ["sakagura"], "groups": ["sakaya"], "query": "蔵元"})),
    );
    let hits = hits.as_array().unwrap();
    assert_eq!(hits.len(), 2, "both contexts must answer: {hits:?}");
    assert_eq!(hits[0]["context"], json!("sakagura"), "{hits:?}");
    assert_eq!(hits[1]["context"], json!("izakaya"), "{hits:?}");

    // An empty group is an empty result, not an error…
    server.ok("PUT", "/groups/kara", Some(json!({})));
    let empty = server.ok(
        "POST",
        "/recall",
        Some(json!({"groups": ["kara"], "cue": "蔵"})),
    );
    assert_eq!(empty["total"], json!(0), "{empty}");
    assert_eq!(empty["matches"], json!([]), "{empty}");

    // …an unknown group refuses before anything is searched…
    let (status, ghost) = server.call(
        "POST",
        "/recall",
        Some(json!({"groups": ["maboroshi"], "cue": "蔵"})),
    );
    assert_eq!(status, 404, "{ghost}");
    assert_eq!(ghost["code"], json!("no_group"), "{ghost}");
    assert!(
        ghost["error"].as_str().unwrap().contains("'maboroshi'"),
        "{ghost}"
    );

    // …naming nothing at all is a client bug, not an empty result…
    let (status, nothing) = server.call("POST", "/recall", Some(json!({"cue": "蔵"})));
    assert_eq!(status, 400, "{nothing}");
    assert_eq!(nothing["code"], json!("invalid_argument"), "{nothing}");

    // …and the groups list shares the input-items cap.
    let flood: Vec<String> = (0..1001).map(|i| format!("g{i}")).collect();
    let (status, over) = server.call(
        "POST",
        "/recall",
        Some(json!({"groups": flood, "cue": "蔵"})),
    );
    assert_eq!(status, 400, "{over}");
    assert_eq!(over["code"], json!("over_limit"), "{over}");

    // The MCP search tools take `groups` as a cross-context form…
    let (status, reply) = server.call(
        "POST",
        "/mcp",
        Some(json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "recall", "arguments": {
                "groups": ["nomiya"], "cue": "蔵",
            }},
        })),
    );
    assert_eq!(status, 200, "{reply}");
    assert_ne!(reply["result"]["isError"], json!(true), "{reply}");
    let text = reply["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("sakagura") && text.contains("izakaya"),
        "{text}"
    );

    // …and beside `context` it is the same ambiguity as `contexts`.
    let (status, reply) = server.call(
        "POST",
        "/mcp",
        Some(json!({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "recall", "arguments": {
                "context": "izakaya", "groups": ["nomiya"], "cue": "蔵",
            }},
        })),
    );
    assert_eq!(status, 200, "{reply}");
    assert_eq!(reply["result"]["isError"], json!(true), "{reply}");
    assert!(
        reply["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("not both"),
        "{reply}"
    );
}

/// A scoped key's cross-context search is refused whole on any name
/// beyond the grant — and because the grant check runs before the
/// existence check, the 403 for a live out-of-grant name and for a
/// made-up one are indistinguishable: no existence oracle. A `groups`
/// target resolves to the grant's slice instead of refusing: a refusal
/// would name the very membership the group listings hide.
#[test]
fn cross_context_search_respects_grants_without_an_existence_oracle() {
    let server = Server::start_with_env(
        "cross-scopes",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,potter:stok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"potter": {"role": "read", "contexts": ["sake"]}}"#,
            ),
        ],
    );
    let call = |method: &str, path: &str, body: Option<Value>, token: &str| {
        server.call_with_token(method, path, body, Some(token))
    };
    for name in ["sake", "bunko"] {
        assert_eq!(
            call("PUT", &format!("/contexts/{name}"), Some(json!({})), "atok").0,
            200
        );
    }

    // Inside the grant: answers.
    let (status, inside) = call(
        "POST",
        "/recall",
        Some(json!({"contexts": ["sake"], "cue": "蔵"})),
        "stok",
    );
    assert_eq!(status, 200, "{inside}");

    // One out-of-grant name refuses the whole request…
    let (status, live) = call(
        "POST",
        "/recall",
        Some(json!({"contexts": ["sake", "bunko"], "cue": "蔵"})),
        "stok",
    );
    assert_eq!(status, 403, "{live}");
    assert_eq!(live["code"], json!("forbidden"), "{live}");
    assert!(
        live["error"]
            .as_str()
            .unwrap()
            .contains("no grant on context 'bunko'"),
        "{live}"
    );

    // …and a nonexistent out-of-grant name answers the IDENTICAL
    // refusal — never the 404 that would betray which names exist.
    let (status, ghost) = call(
        "POST",
        "/recall",
        Some(json!({"contexts": ["sake", "ghost"], "cue": "蔵"})),
        "stok",
    );
    assert_eq!(status, 403, "{ghost}");
    assert_eq!(
        ghost["error"]
            .as_str()
            .unwrap()
            .replace("'ghost'", "'bunko'"),
        live["error"].as_str().unwrap(),
        "the refusals must differ in nothing but the echoed name"
    );

    // The unscoped admin hears the truth about the same request.
    let (status, truth) = call(
        "POST",
        "/recall",
        Some(json!({"contexts": ["sake", "ghost"], "cue": "蔵"})),
        "atok",
    );
    assert_eq!(status, 404, "{truth}");
    assert_eq!(truth["code"], json!("no_context"), "{truth}");

    // The other two searches run the same gate.
    let (status, _) = call(
        "POST",
        "/query",
        Some(json!({"contexts": ["sake", "bunko"], "label": "l"})),
        "stok",
    );
    assert_eq!(status, 403);
    let (status, _) = call(
        "POST",
        "/sources/search",
        Some(json!({"contexts": ["sake", "bunko"], "query": "蔵元"})),
        "stok",
    );
    assert_eq!(status, 403);

    // A group target resolves to the grant's slice instead of refusing
    // — the same slice `GET /groups` shows a scoped key — so nothing
    // in the answer betrays the out-of-grant member.
    for (context, fact) in [
        (
            "sake",
            json!({"subject": "蔵", "label": "銘柄", "object": "白露"}),
        ),
        (
            "bunko",
            json!({"subject": "蔵", "label": "所蔵", "object": "写本"}),
        ),
    ] {
        let (status, _) = call(
            "POST",
            &format!("/contexts/{context}/associations"),
            Some(json!([{"subject": fact["subject"], "label": fact["label"],
                         "object": fact["object"], "weight": 1.0, "source": "x.md"}])),
            "atok",
        );
        assert_eq!(status, 200);
    }
    let (status, _) = call(
        "PUT",
        "/groups/zenbu",
        Some(json!({"contexts": ["sake", "bunko"]})),
        "atok",
    );
    assert_eq!(status, 200);
    let (status, sliced) = call(
        "POST",
        "/recall",
        Some(json!({"groups": ["zenbu"], "cue": "蔵"})),
        "stok",
    );
    assert_eq!(status, 200, "{sliced}");
    assert_eq!(sliced["result"]["total"], json!(1), "{sliced}");
    assert_eq!(
        sliced["result"]["matches"][0]["context"],
        json!("sake"),
        "{sliced}"
    );
    assert!(
        !sliced.to_string().contains("bunko"),
        "the slice must not name the out-of-grant member: {sliced}"
    );

    // A group with nothing in the grant answers empty, exactly as an
    // empty group would — never a refusal naming the hidden member.
    let (status, _) = call(
        "PUT",
        "/groups/soto",
        Some(json!({"contexts": ["bunko"]})),
        "atok",
    );
    assert_eq!(status, 200);
    let (status, outside) = call(
        "POST",
        "/recall",
        Some(json!({"groups": ["soto"], "cue": "蔵"})),
        "stok",
    );
    assert_eq!(status, 200, "{outside}");
    assert_eq!(outside["result"]["total"], json!(0), "{outside}");
    assert!(!outside.to_string().contains("bunko"), "{outside}");

    // Directly naming the out-of-grant context still refuses whole,
    // groups on the request or not.
    let (status, direct) = call(
        "POST",
        "/recall",
        Some(json!({"contexts": ["bunko"], "groups": ["zenbu"], "cue": "蔵"})),
        "stok",
    );
    assert_eq!(status, 403, "{direct}");
}

/// The access log names the context a request addressed, and every
/// destructive operation leaves one `taguru::audit` line saying who
/// did what to which object — the route template alone cannot answer
/// "which context did this key delete" after the fact.
#[test]
fn the_access_log_names_the_context_and_destructive_ops_leave_audit_lines() {
    let data_dir = std::env::temp_dir().join(format!("taguru-auditlog-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
    let mut child = Command::new(env!("CARGO_BIN_EXE_taguru"))
        .env("TAGURU_ADDR", "127.0.0.1:0")
        .env("TAGURU_DATA_DIR", &data_dir)
        .env("TAGURU_LOG_FORMAT", "json")
        .env("TAGURU_API_TOKEN", "opskey")
        .env_remove("TAGURU_EMBED_URL")
        .env_remove("OTEL_EXPORTER_OTLP_ENDPOINT")
        .env_remove("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
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

    let call = |method: &str, path: &str, body: Option<Value>| {
        let request = ureq::AgentBuilder::new()
            .build()
            .request(method, &format!("{base}{path}"))
            .set("Authorization", "Bearer opskey");
        let response = match body {
            Some(body) => request
                .set("Content-Type", "application/json")
                .send_string(&body.to_string()),
            None => request.call(),
        };
        response
            .map(|reply| reply.status())
            .unwrap_or_else(|error| match error {
                ureq::Error::Status(status, _) => status,
                other => panic!("{method} {path}: {other}"),
            })
    };
    assert_eq!(
        call("PUT", "/contexts/sake", Some(json!({"description": "d"}))),
        200
    );
    assert_eq!(
        call(
            "POST",
            "/contexts/sake/associations",
            Some(json!([{"subject": "蔵", "label": "杜氏", "object": "高瀬",
                         "weight": 1.0, "source": "a.md"}])),
        ),
        200
    );
    // Register then remove an alias (its own audit line).
    assert_eq!(
        call(
            "POST",
            "/contexts/sake/aliases",
            Some(json!({"concepts": {"Kura": "蔵"}})),
        ),
        200
    );
    assert_eq!(
        call(
            "DELETE",
            "/contexts/sake/aliases",
            Some(json!({"concepts": ["Kura"]})),
        ),
        200
    );
    // An import batch (retract-then-apply) and a compaction: both
    // destructive, both audited. Import is NDJSON, so send it raw
    // rather than through the JSON `call` helper.
    let import_status = ureq::AgentBuilder::new()
        .build()
        .request("POST", &format!("{base}/import"))
        .set("Authorization", "Bearer opskey")
        .send_string(
            "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"b.md\"}\n\
             {\"subject\": \"蔵\", \"label\": \"銘柄\", \"object\": \"青嶺\", \"weight\": 1.0}\n",
        )
        .map(|reply| reply.status())
        .unwrap_or_else(|error| match error {
            ureq::Error::Status(status, _) => status,
            other => panic!("import: {other}"),
        });
    assert_eq!(import_status, 200);
    assert_eq!(call("POST", "/contexts/sake/compact", None), 200);
    assert_eq!(
        call(
            "POST",
            "/contexts/sake/sources/retract",
            Some(json!({"source": "a.md"})),
        ),
        200
    );
    assert_eq!(call("DELETE", "/contexts/sake", None), 200);

    // Stop the server so stderr reaches EOF, then judge the whole log.
    let pid = child.id().to_string();
    Command::new("kill")
        .args(["-TERM", &pid])
        .status()
        .expect("kill must run");
    let _ = child.wait();
    let stderr = child.stderr.take().expect("stderr must be piped");
    let lines: Vec<Value> = BufReader::new(stderr)
        .lines()
        .map_while(Result::ok)
        .filter_map(|line| serde_json::from_str(&line).ok())
        .collect();

    let access_delete = lines
        .iter()
        .find(|record| {
            record["fields"]["message"] == json!("http")
                && record["fields"]["method"] == json!("DELETE")
                && record["fields"]["route"] == json!("/contexts/{name}")
        })
        .expect("an access-log line for the DELETE must appear");
    assert_eq!(access_delete["fields"]["context"], json!("sake"));
    assert_eq!(access_delete["fields"]["key"], json!("default"));

    let retracted = lines
        .iter()
        .find(|record| {
            record["target"] == json!("taguru::audit")
                && record["fields"]["message"] == json!("source retracted")
        })
        .expect("the retraction must leave an audit line");
    assert_eq!(retracted["fields"]["context"], json!("sake"));
    assert_eq!(retracted["fields"]["source"], json!("a.md"));
    assert_eq!(retracted["fields"]["key"], json!("default"));
    assert_eq!(retracted["fields"]["associations_touched"], json!(1));

    let deleted = lines
        .iter()
        .find(|record| {
            record["target"] == json!("taguru::audit")
                && record["fields"]["message"] == json!("context deleted")
        })
        .expect("the deletion must leave an audit line");
    assert_eq!(deleted["fields"]["context"], json!("sake"));
    assert_eq!(deleted["fields"]["files_removed"], json!(true));

    // Every destructive operation — not just delete/retract — leaves an
    // audit line naming the key and the context. A missing one here is
    // a silently-narrowed guarantee.
    let audit_line = |message: &str| {
        lines
            .iter()
            .find(|record| {
                record["target"] == json!("taguru::audit")
                    && record["fields"]["message"] == json!(message)
            })
            .unwrap_or_else(|| panic!("missing audit line: {message}"))
    };
    let aliases_removed = audit_line("aliases removed");
    assert_eq!(aliases_removed["fields"]["context"], json!("sake"));
    assert_eq!(aliases_removed["fields"]["key"], json!("default"));
    let imported = audit_line("import batch applied");
    assert_eq!(imported["fields"]["context"], json!("sake"));
    assert_eq!(imported["fields"]["source"], json!("b.md"));
    assert_eq!(imported["fields"]["key"], json!("default"));
    let compacted = audit_line("context compacted");
    assert_eq!(compacted["fields"]["context"], json!("sake"));
    assert_eq!(compacted["fields"]["key"], json!("default"));

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
    assert_eq!(hit["paragraph"], 1, "the hit names the answering PARAGRAPH");
    let _ = std::fs::remove_dir_all(&batches);
}

/// The import-time bookkeeping for storing/dropping section markers —
/// the same drop-and-count convention as questions. Resolving a stored
/// section back out through recall/query is covered separately by
/// `an_attributions_section_label_resolves_on_read_but_is_never_fabricated`.
#[test]
fn an_offline_import_carries_sections_through_and_drops_out_of_range_ones() {
    let batches = batch_dir("import-sections");
    let file = batches.join("guide.jsonl");
    std::fs::write(
        &file,
        r#"{"taguru_batch": 1, "context": "sake", "source": "doc-guide", "create": {"description": "酒蔵の記憶"}}
{"passage": "青嶺酒造は1907年に創業した。\n\n杜氏は高瀬。"}
{"paragraph": 1, "section": "杜氏"}
{"paragraph": 9, "section": "存在しない段落"}
"#,
    )
    .unwrap();

    let data_dir = std::env::temp_dir().join(format!(
        "taguru-http-import-sections-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&data_dir);
    let (code, stdout, stderr) = run_import(&data_dir, &[file.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("+1 section(s) (1 dropped: no such paragraph)"),
        "{stdout}"
    );
    let _ = std::fs::remove_dir_all(&batches);
}

/// Unlike questions/sections above, an association's paragraph is
/// incidental metadata, not its reason for existing — so an out-of-range
/// one clears just the locator and keeps the whole fact, where a
/// question or section drops entirely. But the drop is still surfaced
/// with its own count and report line, symmetric with those two: a
/// locator silently vanishing is exactly the kind of loss the report
/// exists to name.
#[test]
fn an_offline_import_drops_an_out_of_range_association_paragraph_but_keeps_the_fact() {
    let batches = batch_dir("import-assoc-paragraph");
    let file = batches.join("guide.jsonl");
    std::fs::write(
        &file,
        r#"{"taguru_batch": 1, "context": "sake", "source": "doc-guide", "create": {"description": "酒蔵の記憶"}}
{"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "paragraph": 0}
{"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0, "paragraph": 9}
{"passage": "青嶺酒造は1907年に創業した。\n\n杜氏は高瀬。"}
"#,
    )
    .unwrap();

    let data_dir = std::env::temp_dir().join(format!(
        "taguru-http-import-assoc-paragraph-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&data_dir);
    let (code, stdout, stderr) = run_import(&data_dir, &[file.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    // Both facts land — the out-of-range locator does not cost its
    // association — and the one dropped locator is named, not silent.
    assert!(stdout.contains("+2 association(s)"), "{stdout}");
    assert!(
        stdout.contains("1 association paragraph locator(s) dropped: no such paragraph"),
        "{stdout}"
    );

    let server = Server::start_on("import-assoc-paragraph", data_dir);
    let founding = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "創業年"})),
    );
    assert_eq!(
        founding["matches"][0]["attributions"][0]["paragraph"],
        json!(0),
        "an in-range paragraph must survive: {founding}"
    );
    let brewer = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "杜氏"})),
    );
    assert_eq!(
        brewer["matches"][0]["weight"],
        json!(1.0),
        "the fact itself must survive even though its locator did not: {brewer}"
    );
    assert_eq!(
        brewer["matches"][0]["attributions"][0]["paragraph"],
        json!(null),
        "an out-of-range paragraph must be cleared, not left dangling: {brewer}"
    );
    let _ = std::fs::remove_dir_all(&batches);
}

/// The same silent clamp, reached through the direct HTTP path instead
/// of an import batch: nothing between `store_passages` and
/// `associations` has the passage text in hand, so
/// `AppState::add_associations` must check the resident passage store
/// itself before honoring a paragraph locator.
#[test]
fn http_associations_drops_an_out_of_range_paragraph_against_a_stored_passage() {
    let server = Server::start("http-assoc-paragraph");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "酒蔵の記憶"})),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {"doc-guide": "青嶺酒造は1907年に創業した。\n\n杜氏は高瀬。"}
        })),
    );
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "source": "doc-guide", "paragraph": 0},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "doc-guide", "paragraph": 9},
        ])),
    );

    let founding = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "創業年"})),
    );
    assert_eq!(
        founding["matches"][0]["attributions"][0]["paragraph"],
        json!(0),
        "an in-range paragraph must survive: {founding}"
    );

    let brewer = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "杜氏"})),
    );
    assert_eq!(
        brewer["matches"][0]["weight"],
        json!(1.0),
        "the fact itself must survive even though its locator did not: {brewer}"
    );
    assert_eq!(
        brewer["matches"][0]["attributions"][0]["paragraph"],
        json!(null),
        "an out-of-range paragraph must be cleared, not left dangling: {brewer}"
    );
}

/// Issue #11: an attribution whose paragraph locator falls inside a
/// stored section resolves that section's label on read — recall,
/// query, and (through the same conversion) explore, activate, and
/// unreachable_from all carry it. A paragraph that exists but sits
/// before every marker, and an attribution with no paragraph locator
/// at all, must both report `section: null` — resolution never
/// fabricates a label. Those two null outcomes come out of the one
/// shared `attribution_out` conversion every endpoint routes through,
/// so recall and query (which exhaust them) are sufficient — repeating
/// them per endpoint would not catch a broken wiring the way a
/// resolved assertion does (null passes whether or not resolution ran
/// at all).
#[test]
fn an_attributions_section_label_resolves_on_read_but_is_never_fabricated() {
    let batches = batch_dir("import-section-resolution");
    let file = batches.join("guide.jsonl");
    std::fs::write(
        &file,
        r#"{"taguru_batch": 1, "context": "sake", "source": "doc-guide", "create": {"description": "酒蔵の記憶"}}
{"passage": "青嶺酒造は1907年に創業した。\n\n杜氏は高瀬。\n\n仕込み水は雲居山の伏流水である。"}
{"paragraph": 1, "section": "杜氏"}
{"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "paragraph": 0}
{"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0, "paragraph": 1}
{"subject": "青嶺酒造", "label": "仕込み水源", "object": "雲居山", "weight": 1.0}
{"subject": "廻船問屋", "label": "取引先", "object": "山田", "weight": 1.0, "paragraph": 1}
"#,
    )
    .unwrap();

    let data_dir = std::env::temp_dir().join(format!(
        "taguru-http-import-section-resolution-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&data_dir);
    let (code, stdout, stderr) = run_import(&data_dir, &[file.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let server = Server::start_on("import-section-resolution", data_dir);

    // Paragraph 1 sits inside the "杜氏" section (which starts at
    // paragraph 1 and runs to the passage's end): recall must resolve
    // it, matching what a manual read of the source would show.
    let recalled = server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "青嶺酒造", "limit": 10})),
    );
    let brewer = recalled["matches"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["label"] == "杜氏")
        .expect("the 杜氏 association must recall from its subject");
    assert_eq!(
        brewer["attributions"][0]["section"],
        json!("杜氏"),
        "a paragraph inside a stored section must resolve its label: {brewer}"
    );

    // Paragraph 0 exists but sits BEFORE the first section marker: no
    // section governs it, so resolution must report null rather than
    // guessing at the nearest one.
    let founding = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "創業年"})),
    );
    assert_eq!(
        founding["matches"][0]["attributions"][0]["paragraph"],
        json!(0),
        "sanity: the paragraph locator itself must still be present: {founding}"
    );
    assert_eq!(
        founding["matches"][0]["attributions"][0]["section"],
        json!(null),
        "a paragraph before every marker must not resolve to a section: {founding}"
    );

    // No paragraph locator at all: section is null with nothing to
    // resolve from.
    let water = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "仕込み水源"})),
    );
    assert_eq!(
        water["matches"][0]["attributions"][0]["paragraph"],
        json!(null),
        "sanity: this attribution carries no locator: {water}"
    );
    assert_eq!(
        water["matches"][0]["attributions"][0]["section"],
        json!(null),
        "an attribution without a paragraph must not resolve to a section: {water}"
    );

    // explore nests associations one level deeper
    // (matches[].association.attributions[]) — check the same
    // resolution reaches that shape too.
    let explored = server.ok(
        "POST",
        "/contexts/sake/explore",
        Some(json!({"origins": ["青嶺酒造"], "max_depth": 1})),
    );
    let brewer_hop = explored["matches"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["association"]["label"] == "杜氏")
        .expect("the 杜氏 association must be one hop from 青嶺酒造");
    assert_eq!(
        brewer_hop["association"]["attributions"][0]["section"],
        json!("杜氏"),
        "explore's nested association must resolve sections too: {brewer_hop}"
    );

    // activate nests associations the same one level deep
    // (matches[].association) — check the same resolution reaches that
    // shape too.
    let activated = server.ok(
        "POST",
        "/contexts/sake/activate",
        Some(json!({"origins": ["青嶺酒造"], "limit": 10})),
    );
    let brewer_activation = activated["matches"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["association"]["label"] == "杜氏")
        .expect("the 杜氏 association must activate from 青嶺酒造");
    assert_eq!(
        brewer_activation["association"]["attributions"][0]["section"],
        json!("杜氏"),
        "activate's nested association must resolve sections too: {brewer_activation}"
    );

    // unreachable_from returns associations no walk from the origin can
    // reach — 廻船問屋 is isolated from 青嶺酒造's graph, so it
    // qualifies; check its section resolves too.
    let orphaned = server.ok(
        "POST",
        "/contexts/sake/unreachable_from",
        Some(json!({"origins": ["青嶺酒造"]})),
    );
    let orphan_match = orphaned["matches"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["subject"] == "廻船問屋")
        .expect("the isolated association must be unreachable from 青嶺酒造");
    assert_eq!(
        orphan_match["attributions"][0]["section"],
        json!("杜氏"),
        "unreachable_from's associations must resolve sections too: {orphan_match}"
    );

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
fn a_partially_applied_import_still_counts_the_context_for_the_refresh_pass() {
    let batches = batch_dir("import-partial-touch");
    let setup = batches.join("setup.jsonl");
    std::fs::write(
        &setup,
        r#"{"taguru_batch": 1, "context": "sake", "source": "doc-1", "create": {"description": "d"}}
{"subject": "青嶺酒造", "label": "所在地", "object": "京都酒造", "weight": 1.0}
{"alias": "kyo", "canonical": "京都酒造", "kind": "concept"}
"#,
    )
    .unwrap();
    let data_dir = std::env::temp_dir().join(format!(
        "taguru-http-import-partial-touch-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&data_dir);
    let (code, stdout, stderr) = run_import(&data_dir, &[setup.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    // BTreeMap order: "aomine" applies, then "kyo" conflicts by
    // re-pointing an existing alias — the batch refuses AFTER its
    // association and first alias landed durably.
    let partial = batches.join("partial.jsonl");
    std::fs::write(
        &partial,
        r#"{"taguru_batch": 1, "context": "sake", "source": "doc-2"}
{"subject": "新蔵", "label": "特徴", "object": "辛口", "weight": 1.0}
{"alias": "aomine", "canonical": "青嶺酒造", "kind": "concept"}
{"alias": "kyo", "canonical": "青嶺酒造", "kind": "concept"}
"#,
    )
    .unwrap();
    let (code, stdout, stderr) = run_import(&data_dir, &[partial.to_str().unwrap()]);
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("applied 1 alias(es)"), "{stderr}");
    // The context took durable writes, so the run's summary (and the
    // embeddings-refresh pass it mirrors) must cover it — a one-shot
    // import has no later tick to pick those glosses up.
    assert!(
        stdout.contains("across 1 context(s)"),
        "a partially applied batch still touched the context: {stdout}"
    );

    // And the writes the summary counts are really there.
    let server = Server::start_on("import-partial-touch", data_dir);
    let edge = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "新蔵", "label": "特徴"})),
    );
    assert_eq!(edge["matches"][0]["object"], json!("辛口"));
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
    // A single-batch body answers the same {batches: [...]} shape a
    // stream does — one shape to parse, with one entry here.
    let outcome = &first["result"]["batches"][0];
    assert_eq!(first["result"]["batches"].as_array().map(Vec::len), Some(1));
    assert_eq!(outcome["created"], json!(true));
    assert_eq!(outcome["associations"], json!(1));
    assert_eq!(outcome["passage_stored"], json!(true));
    assert_eq!(outcome["retracted"], json!(0));

    // Same batch again: the source is replaced, not doubled — the
    // no-downtime spelling of the CLI's idempotency.
    let (status, second) = post_import(&server, batch, Some("opskey"));
    assert_eq!(status, 200, "{second}");
    assert_eq!(second["result"]["batches"][0]["retracted"], json!(1));
    let (status, edge) = server.call_with_token(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "杜氏"})),
        Some("opskey"),
    );
    assert_eq!(status, 200);
    assert_eq!(edge["result"]["matches"][0]["weight"], json!(2.0));
}

/// A live import that stops partway leaves its batch-open marker on
/// disk — the tear detection #59 adds; boot and `taguru inspect`
/// report it — and the operator-facing repair verb (retract the
/// source) clears it. The other repair, re-importing the corrected
/// batch, is covered beside `apply_batch` itself.
#[test]
fn a_torn_live_import_leaves_its_marker_until_the_source_is_retracted() {
    let server = Server::start("http-import-marker");
    // An alias whose canonical nothing interned fails AFTER the
    // retraction step — a genuinely half-applied source.
    let torn = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-torn\", \
                 \"create\": {\"description\": \"d\"}}\n\
                 {\"alias\": \"Aomine\", \"canonical\": \"存在しない\", \"kind\": \"concept\"}\n";
    let (status, body) = post_import(&server, torn, None);
    assert_ne!(status, 200, "the batch must be refused: {body}");

    let markers = |dir: &std::path::Path| -> Vec<std::path::PathBuf> {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("importing"))
            .collect()
    };
    let surviving = markers(&server.data_dir);
    assert_eq!(surviving.len(), 1, "the refused batch keeps its marker");
    let content: Value = serde_json::from_slice(&std::fs::read(&surviving[0]).unwrap()).unwrap();
    assert_eq!(content["context"], json!("sake"));
    assert_eq!(content["source"], json!("doc-torn"));

    // Retracting the source makes its truth consistently absent — the
    // marker stops describing a tear and the verb removes it.
    server.ok(
        "POST",
        "/contexts/sake/sources/retract",
        Some(json!({"source": "doc-torn"})),
    );
    assert!(
        markers(&server.data_dir).is_empty(),
        "retraction clears the marker"
    );
}

#[test]
fn the_import_endpoint_reports_section_bookkeeping() {
    let server = Server::start("http-import-sections");
    let batch = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-sections\", \
                 \"create\": {\"description\": \"d\"}}\n\
                 {\"passage\": \"蔵の杜氏は高瀬。\\n\\n創業は1907年。\"}\n\
                 {\"paragraph\": 1, \"section\": \"沿革\"}\n\
                 {\"paragraph\": 9, \"section\": \"存在しない段落\"}\n";

    let (status, result) = post_import(&server, batch, None);
    assert_eq!(status, 200, "{result}");
    assert_eq!(
        result["result"]["batches"][0]["sections_stored"],
        json!(1),
        "{result}"
    );
    assert_eq!(
        result["result"]["batches"][0]["sections_dropped"],
        json!(1),
        "{result}"
    );
}

/// The import endpoint surfaces a dropped association paragraph locator
/// in its JSON just as the CLI report does: the fact still lands (unlike
/// a dropped question or section, which vanishes whole), only the
/// out-of-range locator is cleared — and counted, never silent.
#[test]
fn the_import_endpoint_reports_dropped_association_paragraphs() {
    let server = Server::start("http-import-assoc-drop");
    let batch = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-guide\", \
                 \"create\": {\"description\": \"d\"}}\n\
                 {\"passage\": \"蔵の杜氏は高瀬。\\n\\n創業は1907年。\"}\n\
                 {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0, \"paragraph\": 9}\n";

    let (status, result) = post_import(&server, batch, None);
    assert_eq!(status, 200, "{result}");
    assert_eq!(
        result["result"]["batches"][0]["associations"],
        json!(1),
        "the fact lands: {result}"
    );
    assert_eq!(
        result["result"]["batches"][0]["association_paragraphs_dropped"],
        json!(1),
        "the out-of-range locator is cleared and counted: {result}"
    );
}

/// The backup loop, live: build a context over the API, pull it back
/// out at GET /contexts/{name}/export, delete the context, and restore
/// it by POSTing the stream to /import — facts, aliases, passages,
/// questions, and sections all round-trip, and the stream response
/// reports one outcome per batch.
#[test]
fn a_context_round_trips_through_the_export_endpoint_and_import() {
    let server = Server::start("http-export-roundtrip");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "酒蔵の知識", "dice_floor": 0.25})),
    );
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺",
             "weight": 1.0, "source": "a.md", "paragraph": 0},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬",
             "weight": 2.0, "source": "b.md"},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {"a.md": "青嶺酒造の紹介。\n\n代表銘柄は青嶺。"},
            "questions": {"a.md": [{"paragraph": 0, "question": "どこの蔵?"}]},
            "sections": {"a.md": [{"paragraph": 0, "section": "概要"}]},
        })),
    );
    server.ok(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"concepts": {"Aomine": "青嶺酒造"}})),
    );

    let (status, exported) = server.call("GET", "/contexts/sake/export", None);
    assert_eq!(status, 200, "{exported}");
    let stream = exported
        .as_str()
        .expect("the export body is JSONL, not the envelope");
    // The doc2query question rides the export stream (its round trip is
    // otherwise invisible in a lexical-only test).
    assert!(
        stream.contains("どこの蔵?"),
        "export must emit the question: {stream}"
    );
    assert_eq!(
        stream.matches("\"taguru_batch\":1").count(),
        2,
        "one batch per source: {stream}"
    );
    assert!(
        stream.contains("\"description\":\"酒蔵の知識\""),
        "{stream}"
    );

    // Exporting a context that does not exist is the ordinary 404.
    let (status, missing) = server.call("GET", "/contexts/ghost/export", None);
    assert_eq!(status, 404, "{missing}");

    server.ok("DELETE", "/contexts/sake", None);
    let (status, restored) = post_import(&server, stream, None);
    assert_eq!(status, 200, "{restored}");
    let outcomes = restored["result"]["batches"]
        .as_array()
        .expect("a stream answers one outcome per batch");
    assert_eq!(outcomes.len(), 2, "{restored}");
    assert_eq!(outcomes[0]["created"], json!(true), "{restored}");

    let facts = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造"})),
    );
    assert_eq!(facts["total"], json!(2), "{facts}");
    let citation = server.ok(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "a.md", "paragraph": 0})),
    );
    assert_eq!(citation["text"], json!("青嶺酒造の紹介。"), "{citation}");
    assert_eq!(citation["section"], json!("概要"), "{citation}");
    let aliases = server.ok("GET", "/contexts/sake/aliases", None);
    assert_eq!(
        aliases["concepts"]["Aomine"],
        json!("青嶺酒造"),
        "{aliases}"
    );
    let row = server.ok("GET", "/contexts/sake", None);
    assert_eq!(row["dice_floor"], json!(0.25), "{row}");
    // The question survived the delete+restore: re-exporting the
    // restored context still carries it.
    let (status, re_exported) = server.call("GET", "/contexts/sake/export", None);
    assert_eq!(status, 200);
    assert!(
        re_exported.as_str().unwrap().contains("どこの蔵?"),
        "the question must survive the round trip: {re_exported}"
    );

    // Restoring over the restored context is a per-source replace, not
    // a doubling — the same idempotency the CLI import promises.
    let (status, again) = post_import(&server, stream, None);
    assert_eq!(status, 200, "{again}");
    let facts = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造"})),
    );
    assert_eq!(facts["total"], json!(2), "no doubling: {facts}");
}

/// Group records ride the import stream and restore AFTER every batch
/// — one body can create the member contexts and the groups bundling
/// them, in any order — as a create-or-replace of the whole record,
/// reported under `groups: [...]` (absent when the stream carried
/// none, so the pre-group response shape is untouched).
#[test]
fn import_restores_group_records_after_the_batches() {
    let server = Server::start("http-import-groups");
    // The group records sit FIRST: apply order is batches-then-groups,
    // not stream order — and `kura` names `kid`, which only this same
    // stream brings.
    let stream = "{\"taguru_group\": 1, \"name\": \"kura\", \"description\": \"蔵まとめ\", \
                   \"contexts\": [\"sake\", \"bunko\"], \"groups\": [\"kid\"]}\n\
                  {\"taguru_group\": 1, \"name\": \"kid\", \"contexts\": [\"bunko\"]}\n\
                  {\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
                   \"create\": {\"description\": \"d\"}}\n\
                  {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n\
                  {\"taguru_batch\": 1, \"context\": \"bunko\", \"source\": \"b.md\", \
                   \"create\": {\"description\": \"d\"}}\n";
    let (status, first) = post_import(&server, stream, None);
    assert_eq!(status, 200, "{first}");
    assert_eq!(first["result"]["batches"].as_array().map(Vec::len), Some(2));
    let restored = first["result"]["groups"]
        .as_array()
        .expect("group outcomes ride the response");
    assert_eq!(restored.len(), 2, "{first}");
    assert_eq!(restored[0]["name"], json!("kura"), "{first}");
    assert_eq!(restored[0]["outcome"], json!("created"), "{first}");
    assert_eq!(restored[0]["contexts"], json!(2), "{first}");
    assert_eq!(restored[0]["groups"], json!(1), "{first}");
    let row = server.ok("GET", "/groups/kura", None);
    assert_eq!(row["contexts"], json!(["bunko", "sake"]), "{row}");
    assert_eq!(row["groups"], json!(["kid"]), "{row}");
    assert_eq!(row["description"], json!("蔵まとめ"), "{row}");

    // Re-POSTing converges: the records already stand.
    let (status, second) = post_import(&server, stream, None);
    assert_eq!(status, 200, "{second}");
    assert_eq!(
        second["result"]["groups"][0]["outcome"],
        json!("unchanged"),
        "{second}"
    );

    // A stream with no group records keeps the pre-group shape: no
    // `groups` field at all.
    let batches_only = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"c.md\"}\n";
    let (status, plain) = post_import(&server, batches_only, None);
    assert_eq!(status, 200, "{plain}");
    assert!(plain["result"].get("groups").is_none(), "{plain}");

    // A restore REPLACES the record: whatever it omits drops.
    let shrunk = "{\"taguru_group\": 1, \"name\": \"kura\", \"contexts\": [\"sake\"]}\n";
    let (status, third) = post_import(&server, shrunk, None);
    assert_eq!(status, 200, "{third}");
    assert_eq!(
        third["result"]["groups"][0]["outcome"],
        json!("replaced"),
        "{third}"
    );
    let row = server.ok("GET", "/groups/kura", None);
    assert_eq!(row["contexts"], json!(["sake"]), "{row}");
    assert_eq!(row["groups"], json!([]), "{row}");
    assert_eq!(
        row["description"],
        json!(""),
        "a replace is the whole record: {row}"
    );
}

/// A group record that would dangle or misshape refuses every group
/// record — with the batches already durable — under the API's usual
/// codes: `no_context` for a missing member, `no_group` for a missing
/// child, `invalid_argument` for a cycle.
#[test]
fn import_refuses_group_records_that_would_dangle_or_misshape() {
    let server = Server::start("http-import-group-refuse");
    let stream = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
                   \"create\": {\"description\": \"d\"}}\n\
                  {\"taguru_group\": 1, \"name\": \"kura\", \"contexts\": [\"sake\", \"ghost\"]}\n";
    let (status, refusal) = post_import(&server, stream, None);
    assert_eq!(status, 404, "{refusal}");
    assert_eq!(refusal["code"], json!("no_context"), "{refusal}");
    assert!(
        refusal["error"].as_str().unwrap().contains("ghost"),
        "{refusal}"
    );
    // The batch landed; no group did.
    let (status, _) = server.call("GET", "/contexts/sake", None);
    assert_eq!(
        status, 200,
        "the batches before the group refusal are durable"
    );
    let (status, gone) = server.call("GET", "/groups/kura", None);
    assert_eq!(status, 404, "{gone}");

    // A child that neither exists nor rides the stream: no_group.
    let stream = "{\"taguru_group\": 1, \"name\": \"kura\", \"contexts\": [\"sake\"], \
                   \"groups\": [\"nope\"]}\n";
    let (status, refusal) = post_import(&server, stream, None);
    assert_eq!(status, 404, "{refusal}");
    assert_eq!(refusal["code"], json!("no_group"), "{refusal}");

    // A cycle the incoming set closes with itself: the request's own
    // shape, 400.
    let stream = "{\"taguru_group\": 1, \"name\": \"a\", \"groups\": [\"b\"]}\n\
                  {\"taguru_group\": 1, \"name\": \"b\", \"groups\": [\"a\"]}\n";
    let (status, refusal) = post_import(&server, stream, None);
    assert_eq!(status, 400, "{refusal}");
    assert_eq!(refusal["code"], json!("invalid_argument"), "{refusal}");

    // Restating one group in one stream is a parse-stage refusal.
    let stream = "{\"taguru_group\": 1, \"name\": \"a\"}\n{\"taguru_group\": 1, \"name\": \"a\"}\n";
    let (status, refusal) = post_import(&server, stream, None);
    assert_eq!(status, 400, "{refusal}");
    assert!(
        refusal["error"]
            .as_str()
            .unwrap()
            .contains("one record owns one group's truth"),
        "{refusal}"
    );
}

/// A scoped key's group records are judged like any group write — by
/// the transitive context closure, the standing record's and the
/// prospective one's both — before anything at all applies.
#[test]
fn a_scoped_key_cannot_import_group_records_beyond_its_grant() {
    let server = Server::start_with_env(
        "http-import-group-scope",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,curator:ctok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"curator": {"role": "admin", "contexts": ["sake"]}}"#,
            ),
        ],
    );
    let call = |method: &str, path: &str, body: Option<Value>, token: &str| {
        server.call_with_token(method, path, body, Some(token))
    };
    assert_eq!(
        call(
            "PUT",
            "/contexts/sake",
            Some(json!({"description": "d"})),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "PUT",
            "/contexts/bunko",
            Some(json!({"description": "d"})),
            "atok"
        )
        .0,
        200
    );

    // Out of grant through the record's own members: the whole request
    // refuses — the in-grant batch beside it included.
    let stream = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s.md\"}\n\
                  {\"taguru_group\": 1, \"name\": \"kura\", \"contexts\": [\"sake\", \"bunko\"]}\n";
    let (status, refusal) = post_import(&server, stream, Some("ctok"));
    assert_eq!(status, 403, "{refusal}");
    assert!(
        refusal["error"].as_str().unwrap().contains("bunko"),
        "{refusal}"
    );
    let (status, _) = call("GET", "/groups/kura", None, "atok");
    assert_eq!(status, 404, "the refusal precedes every apply");
    let (_, sources) = call("GET", "/contexts/sake/sources", None, "atok");
    assert!(
        !sources.to_string().contains("s.md"),
        "the batch must not land either: {sources}"
    );

    // Inside the grant the same key restores normally.
    let stream = "{\"taguru_group\": 1, \"name\": \"kura\", \"contexts\": [\"sake\"]}\n";
    let (status, applied) = post_import(&server, stream, Some("ctok"));
    assert_eq!(status, 200, "{applied}");

    // The replace side is judged too: shrinking a standing group that
    // bundles an out-of-grant member would release that member, so the
    // scoped replace refuses.
    let wide = "{\"taguru_group\": 1, \"name\": \"wide\", \"contexts\": [\"sake\", \"bunko\"]}\n";
    let (status, seeded) = post_import(&server, wide, Some("atok"));
    assert_eq!(status, 200, "{seeded}");
    let shrink = "{\"taguru_group\": 1, \"name\": \"wide\", \"contexts\": [\"sake\"]}\n";
    let (status, refusal) = post_import(&server, shrink, Some("ctok"));
    assert_eq!(status, 403, "{refusal}");
}

/// `GET /groups/{name}/export` serves one `taguru_group` record that
/// `POST /import` restores whole — and a scoped key exports exactly
/// the slice its grant lets it read.
#[test]
fn a_group_exports_as_one_import_record() {
    let server = Server::start_with_env(
        "http-group-export",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,curator:ctok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"curator": {"role": "read", "contexts": ["sake"]}}"#,
            ),
        ],
    );
    let call = |method: &str, path: &str, body: Option<Value>, token: &str| {
        server.call_with_token(method, path, body, Some(token))
    };
    assert_eq!(
        call(
            "PUT",
            "/contexts/sake",
            Some(json!({"description": "d"})),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "PUT",
            "/contexts/bunko",
            Some(json!({"description": "d"})),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(call("PUT", "/groups/kid", Some(json!({})), "atok").0, 200);
    assert_eq!(
        call(
            "PUT",
            "/groups/kura",
            Some(
                json!({"description": "蔵まとめ", "contexts": ["sake", "bunko"],
                        "groups": ["kid"]})
            ),
            "atok"
        )
        .0,
        200
    );

    // One JSONL line is itself valid JSON, so the harness hands it
    // back parsed — assert the record's shape (the byte-exact
    // rendering, field order and omitted empties, is pinned by
    // export's own unit test).
    let (status, exported) = call("GET", "/groups/kura/export", None, "atok");
    assert_eq!(status, 200, "{exported}");
    assert_eq!(
        exported,
        json!({"taguru_group": 1, "name": "kura", "description": "蔵まとめ",
               "contexts": ["bunko", "sake"], "groups": ["kid"]})
    );

    // Deleting and re-importing the record restores the group whole.
    assert_eq!(call("DELETE", "/groups/kura", None, "atok").0, 200);
    let line = format!("{exported}\n");
    let (status, restored) = post_import(&server, &line, Some("atok"));
    assert_eq!(status, 200, "{restored}");
    assert_eq!(
        restored["result"]["groups"][0]["outcome"],
        json!("created"),
        "{restored}"
    );
    let (_, row) = call("GET", "/groups/kura", None, "atok");
    assert_eq!(row["result"]["contexts"], json!(["bunko", "sake"]), "{row}");

    // A scoped key exports its grant's slice — the row it can read IS
    // the record it takes away.
    let (status, sliced) = call("GET", "/groups/kura/export", None, "ctok");
    assert_eq!(status, 200, "{sliced}");
    assert_eq!(sliced["contexts"], json!(["sake"]), "{sliced}");

    // Unknown group: the ordinary 404.
    let (status, missing) = call("GET", "/groups/ghost/export", None, "atok");
    assert_eq!(status, 404, "{missing}");
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
    // Paragraph 0 is the founding sentence; paragraph 1 is the brewer
    // and no-mass-production sentence — the tagged values below match
    // where each fact actually sits in the source text above.
    let aomine_reply = json!({
        "associations": [
            {"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "paragraph": 0},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0, "paragraph": 1},
            {"subject": "青嶺酒造", "label": "行う", "object": "大量生産", "weight": -1.0, "paragraph": 1},
            {"subject": "青嶺酒造", "label": "所在地", "object": null},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0, "paragraph": 1}
        ],
        "aliases": [
            {"alias": "Aomine", "canonical": "青嶺酒造", "kind": "concept"},
            {"alias": "幽霊", "canonical": "存在しない", "kind": "concept"}
        ]
    })
    .to_string();
    // takase's reply omits paragraph entirely — the missing-tag path
    // must still leave the fact in place (asserted below via the
    // server responses, since a dropped fact wouldn't come back at all).
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
    // Every paragraph is numbered for the model now, questions or not
    // — the same indexes aomine_reply's associations tag themselves
    // with above.
    assert!(
        requests[0].contains("[0] 青嶺酒造は1907年に創業した。"),
        "{}",
        requests[0]
    );
    assert!(
        requests[0].contains("[1] 杜氏は高瀬。大量生産は行わない。"),
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
    assert_eq!(hit["paragraph"], 1, "the hit names the answering PARAGRAPH");
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

/// The citation endpoint's wire contract: given a known (source,
/// paragraph), it returns the exact verbatim excerpt `search_passages`
/// would show for that same paragraph — sliced through the one shared
/// `PassageRecord::paragraph` accessor, so the two can never disagree —
/// plus the source, and `section` always present as a key (never
/// omitted). This source stored no sections, so it resolves to `null`;
/// `citation_resolves_the_section_governing_its_paragraph` covers the
/// case where a section is actually stored.
#[test]
fn citation_returns_the_verbatim_paragraph_named_by_source_and_paragraph() {
    let server = Server::start("citation-hit");
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

    let citation = server.ok(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "docs/aomine.md", "paragraph": 1})),
    );
    assert_eq!(
        citation["text"], "原料米には山田錦を使い、精米歩合は50パーセントまで磨く。",
        "{citation}"
    );
    assert_eq!(citation["source"], "docs/aomine.md");
    assert!(
        citation.as_object().unwrap().contains_key("section") && citation["section"].is_null(),
        "section is present and null, never omitted: {citation}"
    );
}

/// Unknown source, an out-of-range paragraph position, and an unknown
/// context all speak the same `ApiError` shape (never a panic), each
/// with a message naming what was not found.
#[test]
fn citation_reports_clear_errors_for_unknown_source_paragraph_and_context() {
    let server = Server::start("citation-miss");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "蔵の知識"})),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"docs/aomine.md": "一段落だけ。"}})),
    );

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "docs/ghost.md", "paragraph": 0})),
    );
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
    assert!(
        body["error"].as_str().unwrap().contains("docs/ghost.md"),
        "{body}"
    );

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "docs/aomine.md", "paragraph": 9})),
    );
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
    assert!(
        body["error"].as_str().unwrap().contains("docs/aomine.md"),
        "{body}"
    );

    let (status, body) = server.call(
        "POST",
        "/contexts/ghost/citations",
        Some(json!({"source": "docs/aomine.md", "paragraph": 0})),
    );
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
    assert!(body["error"].as_str().unwrap().contains("ghost"), "{body}");
}

/// A section stored via import resolves on the citation endpoint too:
/// `AppState::citation` reads it off the very same `PassageRecord` via
/// `section_for`, the accessor `resolve_sections` uses for association
/// attributions, so both report the same label for the same paragraph.
/// The paragraph preceding the first marker still resolves to `null`.
#[test]
fn citation_resolves_the_section_governing_its_paragraph() {
    let server = Server::start("citation-section");
    let batch = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-sections\", \
                 \"create\": {\"description\": \"d\"}}\n\
                 {\"passage\": \"蔵の杜氏は高瀬。\\n\\n創業は1907年。\"}\n\
                 {\"paragraph\": 1, \"section\": \"沿革\"}\n";
    let (status, result) = post_import(&server, batch, None);
    assert_eq!(status, 200, "{result}");
    assert_eq!(
        result["result"]["batches"][0]["sections_stored"],
        json!(1),
        "{result}"
    );

    let before = server.ok(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "doc-sections", "paragraph": 0})),
    );
    assert_eq!(before["text"], "蔵の杜氏は高瀬。", "{before}");
    assert!(
        before["section"].is_null(),
        "paragraph 0 precedes the first marker: {before}"
    );

    let after = server.ok(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "doc-sections", "paragraph": 1})),
    );
    assert_eq!(after["text"], "創業は1907年。", "{after}");
    assert_eq!(after["section"], json!("沿革"), "{after}");
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

/// Section markers over HTTP: they ride the store request per source
/// exactly like questions, out-of-range ones are dropped with their
/// count reported (never failing the passage), and sections for a
/// source the request does not carry are refused outright.
#[test]
fn store_passages_accepts_sections_and_reports_the_bookkeeping() {
    let server = Server::start("passage-sections");
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
            "sections": {"doc": [
                {"paragraph": 1, "section": "沿革"},
                {"paragraph": 9, "section": "存在しない段落"}
            ]}
        })),
    );
    assert_eq!(result["stored"], 1, "{result}");
    assert_eq!(result["sections_stored"], 1, "{result}");
    assert_eq!(result["sections_dropped"], 1, "{result}");

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {},
            "sections": {"ghost": [{"paragraph": 0, "section": "幽霊"}]}
        })),
    );
    assert_eq!(status, 400, "{body}");
}

/// The live path proves the same resolution as import: a section set
/// via `POST /sources` (not the batch importer) governs its paragraph
/// on the citation endpoint too, the same way
/// `citation_resolves_the_section_governing_its_paragraph` proves it
/// for import — and the paragraph preceding the first marker still
/// resolves to `null`.
#[test]
fn a_section_stored_via_store_passages_resolves_on_citation() {
    let server = Server::start("store-passages-citation-section");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "蔵の知識"})),
    );
    let result = server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {"doc-sections": "蔵の杜氏は高瀬。\n\n創業は1907年。"},
            "sections": {"doc-sections": [{"paragraph": 1, "section": "沿革"}]}
        })),
    );
    assert_eq!(result["sections_stored"], 1, "{result}");

    let before = server.ok(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "doc-sections", "paragraph": 0})),
    );
    assert_eq!(before["text"], "蔵の杜氏は高瀬。", "{before}");
    assert!(
        before["section"].is_null(),
        "paragraph 0 precedes the first marker: {before}"
    );

    let after = server.ok(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "doc-sections", "paragraph": 1})),
    );
    assert_eq!(after["text"], "創業は1907年。", "{after}");
    assert_eq!(after["section"], json!("沿革"), "{after}");
}

/// /live is the pure liveness probe: 200 whenever the process answers,
/// unauthenticated, unconditional — /health keeps the readiness
/// (write-path) signal.
#[test]
fn live_answers_unauthenticated_even_with_auth_on() {
    let server = Server::start_with_env("http-live", &[("TAGURU_API_TOKEN", "opskey")]);
    let (status, body) = server.call("GET", "/live", None);
    assert_eq!(status, 200);
    assert_eq!(body, json!("ok"));
}

/// The three collection listings page like the directory: keyset
/// cursors, a total that tells the whole story, and — for aliases —
/// one cursor spanning both namespaces, concepts first.
#[test]
fn labels_aliases_and_sources_page_with_keyset_cursors() {
    let server = Server::start("http-keyset-pages");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "a.md"},
            {"subject": "蔵", "label": "創業年", "object": "1907年", "weight": 1.0, "source": "a.md"},
            {"subject": "蔵", "label": "銘柄", "object": "青嶺", "weight": 1.0, "source": "b.md"},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"a.md": "本文。", "b.md": "本文。", "c.md": "本文。"}})),
    );
    server.ok(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({
            "concepts": {"Aomine": "青嶺", "Kura": "蔵"},
            "labels": {"establishment": "創業年"},
        })),
    );

    // labels: sorted, paged, total constant across pages.
    let first = server.ok("GET", "/contexts/sake/labels?limit=2", None);
    assert_eq!(first["total"], json!(3), "{first}");
    assert_eq!(first["labels"].as_array().unwrap().len(), 2);
    let last = first["labels"][1].as_str().unwrap();
    let second = server.ok(
        "GET",
        &format!("/contexts/sake/labels?after={}", urlencode(last)),
        None,
    );
    assert_eq!(second["total"], json!(3));
    assert_eq!(second["labels"].as_array().unwrap().len(), 1);

    // sources: keyset by id.
    let first = server.ok("GET", "/contexts/sake/sources?limit=2", None);
    assert_eq!(first["total"], json!(3), "{first}");
    assert_eq!(first["sources"], json!(["a.md", "b.md"]), "{first}");
    let second = server.ok("GET", "/contexts/sake/sources?after=b.md", None);
    assert_eq!(second["sources"], json!(["c.md"]), "{second}");

    // aliases: one cursor across both namespaces, concepts first.
    let first = server.ok("GET", "/contexts/sake/aliases?limit=2", None);
    assert_eq!(first["total"], json!(3), "{first}");
    assert_eq!(
        first["concepts"],
        json!({"Aomine": "青嶺", "Kura": "蔵"}),
        "{first}"
    );
    assert_eq!(first["labels"], json!({}), "{first}");
    let second = server.ok("GET", "/contexts/sake/aliases?after=concept:Kura", None);
    assert_eq!(second["concepts"], json!({}), "{second}");
    assert_eq!(
        second["labels"],
        json!({"establishment": "創業年"}),
        "{second}"
    );
    // A malformed cursor is a 400, not an empty page.
    let (status, refusal) = server.call("GET", "/contexts/sake/aliases?after=bogus", None);
    assert_eq!(status, 400, "{refusal}");
}

/// Percent-encodes one query value the way ureq will not do for us.
fn urlencode(value: &str) -> String {
    value
        .bytes()
        .map(|byte| match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (byte as char).to_string()
            }
            other => format!("%{other:02X}"),
        })
        .collect()
}

/// POST /contexts/{name}/compact rewrites the image live: smaller
/// footprint, identical answers, and — being an admin verb — refused
/// for write-scoped keys by the fail-closed role table.
#[test]
fn the_compact_endpoint_shrinks_live_and_is_admin_only() {
    let server = Server::start_with_env(
        "http-compact",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,scribe:wtok"),
            ("TAGURU_KEY_SCOPES", r#"{"scribe": "write"}"#),
        ],
    );
    let admin = Some("atok");
    let (status, _) = server.call_with_token(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "d"})),
        admin,
    );
    assert_eq!(status, 200);
    server.call_with_token(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "keep.md"},
            {"subject": "蔵", "label": "廃止銘柄", "object": "旧銘", "weight": 1.0, "source": "gone.md"},
        ])),
        admin,
    );
    server.call_with_token(
        "POST",
        "/contexts/sake/sources/retract",
        Some(json!({"source": "gone.md"})),
        admin,
    );

    let (status, refused) =
        server.call_with_token("POST", "/contexts/sake/compact", None, Some("wtok"));
    assert_eq!(status, 403, "{refused}");

    let (status, outcome) = server.call_with_token("POST", "/contexts/sake/compact", None, admin);
    assert_eq!(status, 200, "{outcome}");
    let shed = &outcome["result"];
    assert_eq!(shed["dead_edges"], json!(1), "{outcome}");
    assert!(
        shed["bytes_after"].as_u64().unwrap() < shed["bytes_before"].as_u64().unwrap(),
        "{outcome}"
    );
    let (status, facts) = server.call_with_token(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵"})),
        admin,
    );
    assert_eq!(status, 200);
    assert_eq!(facts["result"]["total"], json!(1), "{facts}");
    assert_eq!(
        facts["result"]["matches"][0]["label"],
        json!("杜氏"),
        "{facts}"
    );
}

// ---------------------------------------------------------------------
// Groups: flat bundles of contexts — the addressing unit cross-context
// retrieval will build on. This iteration is bundling only: CRUD,
// delta membership, strict referential integrity, and the scope story.

/// The whole group lifecycle over HTTP: create (with and without a
/// body), the keyset-paged directory, single GET, delta PATCH, DELETE —
/// and the namespace split from contexts.
#[test]
fn groups_bundle_contexts_with_crud_paging_and_a_separate_namespace() {
    let server = Server::start("groups-crud");
    for name in ["apple", "banana", "cherry"] {
        server.ok(
            "PUT",
            &format!("/contexts/{name}"),
            Some(json!({"description": name})),
        );
    }

    server.ok(
        "PUT",
        "/groups/fruit",
        Some(json!({"description": "果物の文脈", "contexts": ["banana", "apple"]})),
    );
    // Create is PUT-once: a second landing answers already_exists.
    let (status, dup) = server.call("PUT", "/groups/fruit", None);
    assert_eq!(status, 409, "{dup}");
    assert_eq!(dup["code"], json!("already_exists"));
    // An absent body is a valid create (defaults) — an empty group is
    // how "create first, fill later" starts.
    server.ok("PUT", "/groups/empty", None);

    // The directory pages by name, `total` cursor-independent, exactly
    // like /contexts.
    let page = server.ok("GET", "/groups", None);
    assert_eq!(page["total"], json!(2), "{page}");
    let names: Vec<&str> = page["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|group| group["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["empty", "fruit"], "name order");
    assert_eq!(
        page["groups"][1]["contexts"],
        json!(["apple", "banana"]),
        "members come back sorted: {page}"
    );
    let page = server.ok("GET", "/groups?limit=1", None);
    assert_eq!(page["total"], json!(2));
    assert_eq!(page["groups"][0]["name"], json!("empty"));
    let page = server.ok("GET", "/groups?limit=1&after=empty", None);
    assert_eq!(page["groups"][0]["name"], json!("fruit"));

    let single = server.ok("GET", "/groups/fruit", None);
    assert_eq!(single["description"], json!("果物の文脈"));
    let (status, missing) = server.call("GET", "/groups/nope", None);
    assert_eq!(status, 404);
    assert_eq!(missing["code"], json!("no_group"));

    // PATCH applies deltas — removals first, then adds — and answers
    // the updated row. Removing a non-member is an idempotent no-op.
    let updated = server.ok(
        "PATCH",
        "/groups/fruit",
        Some(json!({"description": "更新後", "add_contexts": ["cherry"],
                    "remove_contexts": ["apple", "never-was-a-member"]})),
    );
    assert_eq!(updated["description"], json!("更新後"));
    assert_eq!(updated["contexts"], json!(["banana", "cherry"]));
    let (status, missing) = server.call("PATCH", "/groups/nope", Some(json!({"description": "x"})));
    assert_eq!(status, 404, "{missing}");
    assert_eq!(missing["code"], json!("no_group"));

    // Groups and contexts are separate namespaces: one name, both kinds.
    server.ok("PUT", "/groups/apple", Some(json!({"description": "同名"})));
    assert_eq!(server.call("GET", "/contexts/apple", None).0, 200);
    assert_eq!(server.call("GET", "/groups/apple", None).0, 200);

    // DELETE removes the bundling alone; the members live on.
    assert_eq!(server.ok("DELETE", "/groups/fruit", None), json!(true));
    assert_eq!(server.call("GET", "/groups/fruit", None).0, 404);
    for name in ["banana", "cherry"] {
        assert_eq!(
            server.call("GET", &format!("/contexts/{name}"), None).0,
            200
        );
    }
    let (status, gone) = server.call("DELETE", "/groups/fruit", None);
    assert_eq!(status, 404, "{gone}");
    assert_eq!(gone["code"], json!("no_group"));
}

/// Strict referential integrity: an add never dangles (no_context, and
/// NOTHING applies), and deleting a context sweeps it out of every
/// group immediately.
#[test]
fn group_membership_is_strict_and_context_deletion_sweeps() {
    let server = Server::start("groups-strict");
    // A create naming a missing context refuses whole: no group.
    let (status, refused) = server.call("PUT", "/groups/g", Some(json!({"contexts": ["ghost"]})));
    assert_eq!(status, 404, "{refused}");
    assert_eq!(refused["code"], json!("no_context"));
    assert_eq!(server.call("GET", "/groups/g", None).0, 404);

    server.ok("PUT", "/contexts/a", None);
    server.ok("PUT", "/contexts/b", None);
    server.ok("PUT", "/groups/g", Some(json!({"contexts": ["a", "b"]})));

    // An add naming a missing context refuses whole: membership as was.
    let (status, refused) = server.call(
        "PATCH",
        "/groups/g",
        Some(json!({"add_contexts": ["ghost"], "remove_contexts": ["a"]})),
    );
    assert_eq!(status, 404, "{refused}");
    assert_eq!(refused["code"], json!("no_context"));
    assert_eq!(
        server.ok("GET", "/groups/g", None)["contexts"],
        json!(["a", "b"]),
        "the refused delta must not half-apply"
    );

    // Deleting a member context drops it from the group, immediately.
    server.ok("DELETE", "/contexts/a", None);
    assert_eq!(
        server.ok("GET", "/groups/g", None)["contexts"],
        json!(["b"])
    );

    // The write-boundary caps hold for groups too.
    let long_name = "x".repeat(65);
    let (status, oversized) = server.call("PUT", &format!("/groups/{long_name}"), None);
    assert_eq!(status, 400, "{oversized}");
    assert_eq!(oversized["code"], json!("invalid_argument"));
    let (status, oversized) = server.call(
        "PUT",
        "/groups/big",
        Some(json!({"description": "x".repeat(5000)})),
    );
    assert_eq!(status, 400, "{oversized}");
    let over_cap: Vec<String> = (0..1001).map(|i| format!("c{i}")).collect();
    let (status, refused) = server.call("PUT", "/groups/big", Some(json!({"contexts": over_cap})));
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("over_limit"));
}

/// Membership is capped in TOTAL, not just per request: deltas cannot
/// grow a group past 1,000 names per set. The cap is judged on the
/// delta's result, before existence — so these ghosts never need to
/// exist to hear it — and removals apply first, making room in the
/// same request.
#[test]
fn group_membership_cannot_be_grown_past_the_cap_by_deltas() {
    let server = Server::start("groups-total-cap");
    server.ok("PUT", "/contexts/a", None);
    server.ok("PUT", "/groups/g", Some(json!({"contexts": ["a"]})));
    server.ok("PUT", "/groups/kid", None);
    server.ok("PATCH", "/groups/g", Some(json!({"add_groups": ["kid"]})));

    // 1 member + 1,000 adds = one past the cap: refused whole.
    let ghosts: Vec<String> = (0..1000).map(|i| format!("ghost{i:04}")).collect();
    let (status, refused) =
        server.call("PATCH", "/groups/g", Some(json!({"add_contexts": ghosts})));
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("over_limit"), "{refused}");
    // Child groups ride the same cap on their own set.
    let ghost_kids: Vec<String> = (0..1000).map(|i| format!("gg{i:04}")).collect();
    let (status, refused) = server.call(
        "PATCH",
        "/groups/g",
        Some(json!({"add_groups": ghost_kids})),
    );
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("over_limit"), "{refused}");

    // Removing the member in the same request makes room — the cap
    // passes and the EXISTENCE gate answers next, proving the cap is
    // judged on the result and before existence.
    let ghosts: Vec<String> = (0..1000).map(|i| format!("ghost{i:04}")).collect();
    let (status, refused) = server.call(
        "PATCH",
        "/groups/g",
        Some(json!({"add_contexts": ghosts, "remove_contexts": ["a"]})),
    );
    assert_eq!(status, 404, "{refused}");
    assert_eq!(refused["code"], json!("no_context"), "{refused}");

    // Nothing half-applied anywhere along the way.
    let row = server.ok("GET", "/groups/g", None);
    assert_eq!(row["contexts"], json!(["a"]), "{row}");
    assert_eq!(row["groups"], json!(["kid"]), "{row}");
}

/// Nesting: a group may hold child groups — at most three storeys,
/// never a cycle, children must exist — child deltas patch like
/// context deltas, and deleting a child sweeps it out of every parent.
#[test]
fn groups_nest_with_a_depth_cap_and_no_cycles() {
    let server = Server::start("groups-nesting");
    for name in ["a", "b"] {
        server.ok("PUT", &format!("/contexts/{name}"), None);
    }
    server.ok("PUT", "/groups/leaf", Some(json!({"contexts": ["a"]})));
    server.ok(
        "PUT",
        "/groups/mid",
        Some(json!({"groups": ["leaf"], "contexts": ["b"]})),
    );
    server.ok("PUT", "/groups/top", Some(json!({"groups": ["mid"]})));

    // Rows carry their children; members stay the direct ones.
    let row = server.ok("GET", "/groups/mid", None);
    assert_eq!(row["groups"], json!(["leaf"]), "{row}");
    assert_eq!(row["contexts"], json!(["b"]));
    let page = server.ok("GET", "/groups", None);
    assert_eq!(page["groups"][2]["name"], json!("top"), "{page}");
    assert_eq!(page["groups"][2]["groups"], json!(["mid"]));

    // A fourth storey refuses as a cap, a cycle (the self-loop
    // included) as a bad argument, an unknown child in the group
    // namespace's own 404 — and nothing half-applies.
    let (status, refused) = server.call("PUT", "/groups/over", Some(json!({"groups": ["top"]})));
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("over_limit"));
    assert_eq!(server.call("GET", "/groups/over", None).0, 404);
    let (status, refused) = server.call(
        "PATCH",
        "/groups/leaf",
        Some(json!({"add_groups": ["top"]})),
    );
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("invalid_argument"));
    let (status, refused) = server.call(
        "PATCH",
        "/groups/leaf",
        Some(json!({"add_groups": ["leaf"]})),
    );
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("invalid_argument"));
    let (status, refused) = server.call(
        "PATCH",
        "/groups/top",
        Some(json!({"add_groups": ["ghost"], "remove_groups": ["mid"]})),
    );
    assert_eq!(status, 404, "{refused}");
    assert_eq!(refused["code"], json!("no_group"));
    assert_eq!(
        server.ok("GET", "/groups/top", None)["groups"],
        json!(["mid"]),
        "the refused delta must not half-apply"
    );

    // Child deltas move like context deltas — and a child may sit
    // under two parents at once: the shape is a DAG, not a tree.
    let updated = server.ok(
        "PATCH",
        "/groups/top",
        Some(json!({"add_groups": ["leaf"], "remove_groups": ["mid"]})),
    );
    assert_eq!(updated["groups"], json!(["leaf"]));
    assert_eq!(
        server.ok("GET", "/groups/mid", None)["groups"],
        json!(["leaf"])
    );

    // Deleting a child sweeps it from every parent; its member
    // contexts live on untouched.
    server.ok("DELETE", "/groups/leaf", None);
    assert_eq!(server.ok("GET", "/groups/top", None)["groups"], json!([]));
    assert_eq!(server.ok("GET", "/groups/mid", None)["groups"], json!([]));
    assert_eq!(server.call("GET", "/contexts/a", None).0, 200);

    // The children list rides the same input ceiling as contexts.
    let over_cap: Vec<String> = (0..1001).map(|i| format!("g{i}")).collect();
    let (status, refused) = server.call("PUT", "/groups/wide", Some(json!({"groups": over_cap})));
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("over_limit"));
}

/// Groups persist across a restart, and boot reconciliation drops any
/// dangling member a crash could have left in a group file.
#[test]
fn groups_survive_restart_and_boot_reconciles_dangling_members() {
    let server = Server::start("groups-restart");
    server.ok("PUT", "/contexts/sake", None);
    server.ok(
        "PUT",
        "/groups/drinks",
        Some(json!({"description": "飲料", "contexts": ["sake"]})),
    );

    let data_dir = server.stop_gracefully();
    // Plant a dangling member and a dangling child the way a crash
    // between a deletion and the sweep's rewrite would: straight into
    // the file.
    std::fs::write(
        data_dir.join("drinks.group"),
        serde_json::to_vec(&json!({"description": "飲料", "contexts": ["sake", "gone"],
                                   "groups": ["nowhere"]}))
        .unwrap(),
    )
    .unwrap();

    let server = Server::start_on("groups-restart2", data_dir);
    let survived = server.ok("GET", "/groups/drinks", None);
    assert_eq!(survived["description"], json!("飲料"));
    assert_eq!(
        survived["contexts"],
        json!(["sake"]),
        "boot must reconcile the planted dangling member: {survived}"
    );
    assert_eq!(
        survived["groups"],
        json!([]),
        "boot must reconcile the planted dangling child: {survived}"
    );
    // And the fix reached the file, not just memory.
    let on_disk = std::fs::read_to_string(server.data_dir.join("drinks.group")).unwrap();
    assert!(!on_disk.contains("gone"), "{on_disk}");
    assert!(!on_disk.contains("nowhere"), "{on_disk}");
}

/// The scope story for groups: every key sees every row but only its
/// granted members; a write touching any context beyond the grant —
/// current members included — refuses whole, and out-of-scope names
/// answer the same 403 whether or not they exist (no existence oracle).
#[test]
fn key_scopes_filter_group_members_and_gate_group_writes() {
    let server = Server::start_with_env(
        "groups-scopes",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,reader:rtok,potter:stok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"reader": "read", "potter": {"role": "write", "contexts": ["sake"]}}"#,
            ),
        ],
    );
    let call = |method: &str, path: &str, body: Option<Value>, token: &str| {
        server.call_with_token(method, path, body, Some(token))
    };
    for context in ["sake", "bunko"] {
        assert_eq!(
            call("PUT", &format!("/contexts/{context}"), None, "atok").0,
            200
        );
    }
    assert_eq!(
        call(
            "PUT",
            "/groups/mixed",
            Some(json!({"contexts": ["sake", "bunko"]})),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "PUT",
            "/groups/ours",
            Some(json!({"contexts": ["sake"]})),
            "atok"
        )
        .0,
        200
    );

    // Reads: every row is visible (groups are labels, not content),
    // but a scoped key sees only its granted members.
    let (status, listed) = call("GET", "/groups", None, "stok");
    assert_eq!(status, 200);
    assert_eq!(listed["result"]["total"], json!(2), "{listed}");
    assert_eq!(
        listed["result"]["groups"][0]["name"],
        json!("mixed"),
        "{listed}"
    );
    assert_eq!(
        listed["result"]["groups"][0]["contexts"],
        json!(["sake"]),
        "bunko must be filtered from a sake-scoped view: {listed}"
    );
    let (_, single) = call("GET", "/groups/mixed", None, "stok");
    assert_eq!(single["result"]["contexts"], json!(["sake"]), "{single}");

    // Role gate: read keys read, nothing more.
    let (status, refused) = call("PUT", "/groups/new", None, "rtok");
    assert_eq!(status, 403, "{refused}");
    assert!(
        refused["error"].as_str().unwrap().contains("needs 'write'"),
        "{refused}"
    );

    // Writes judge every involved context — current members included.
    // Touching a group with an out-of-grant member refuses whole, even
    // for a description-only change.
    let (status, refused) = call(
        "PATCH",
        "/groups/mixed",
        Some(json!({"description": "mine now"})),
        "stok",
    );
    assert_eq!(status, 403, "{refused}");
    assert!(
        refused["error"]
            .as_str()
            .unwrap()
            .contains("no grant on context 'bunko'"),
        "{refused}"
    );

    // The oracle stays shut: an out-of-scope name answers the same 403
    // whether it exists (bunko) or not (ghost) — never a revealing 404.
    let (status_real, real) = call(
        "PATCH",
        "/groups/ours",
        Some(json!({"add_contexts": ["bunko"]})),
        "stok",
    );
    let (status_ghost, ghost) = call(
        "PATCH",
        "/groups/ours",
        Some(json!({"add_contexts": ["ghost"]})),
        "stok",
    );
    assert_eq!((status_real, status_ghost), (403, 403), "{real} / {ghost}");
    assert_eq!(real["code"], ghost["code"], "{real} / {ghost}");
    let (_, ours) = call("GET", "/groups/ours", None, "atok");
    assert_eq!(
        ours["result"]["contexts"],
        json!(["sake"]),
        "the refused adds must not have applied: {ours}"
    );

    // Inside the grant, a scoped writer works normally...
    assert_eq!(
        call(
            "PATCH",
            "/groups/ours",
            Some(json!({"description": "陶工の棚"})),
            "stok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "PUT",
            "/groups/mine",
            Some(json!({"contexts": ["sake"]})),
            "stok"
        )
        .0,
        200
    );
    // ...but deletion is an operator verb (admin), like contexts.
    assert_eq!(call("DELETE", "/groups/mine", None, "stok").0, 403);
    assert_eq!(call("DELETE", "/groups/mine", None, "atok").0, 200);

    // Nesting counts through: a child whose members sit beyond the
    // grant poisons every write on the parent — the child's NAME stays
    // visible (labels, not content), but its contexts are what a grant
    // is about, wherever they hang.
    assert_eq!(
        call(
            "PUT",
            "/groups/shelf",
            Some(json!({"contexts": ["bunko"]})),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "PATCH",
            "/groups/ours",
            Some(json!({"add_groups": ["shelf"]})),
            "atok"
        )
        .0,
        200
    );
    let (_, nested) = call("GET", "/groups/ours", None, "stok");
    assert_eq!(nested["result"]["groups"], json!(["shelf"]), "{nested}");
    assert_eq!(nested["result"]["contexts"], json!(["sake"]), "{nested}");
    let (status, refused) = call(
        "PATCH",
        "/groups/ours",
        Some(json!({"description": "still mine"})),
        "stok",
    );
    assert_eq!(status, 403, "{refused}");
    assert!(
        refused["error"]
            .as_str()
            .unwrap()
            .contains("no grant on context 'bunko'"),
        "{refused}"
    );
    // Naming such a child in a delta refuses the same way — a scoped
    // key cannot create a parent over contexts it has no grant on.
    let (status, refused) = call(
        "PUT",
        "/groups/annex",
        Some(json!({"groups": ["shelf"]})),
        "stok",
    );
    assert_eq!(status, 403, "{refused}");
    assert_eq!(call("GET", "/groups/annex", None, "atok").0, 404);
}

/// The four group tools ride the MCP transport like every other tool —
/// one implementation behind both the stdio bridge and POST /mcp.
#[test]
fn groups_ride_the_mcp_transport() {
    let server = Server::start("groups-mcp");
    server.ok("PUT", "/contexts/sake", None);

    let tool = |id: u64, name: &str, arguments: Value| server.call_tool(id, name, arguments);

    let created = tool(
        1,
        "create_group",
        json!({"name": "drinks", "description": "飲料", "contexts": ["sake"]}),
    );
    assert!(created.get("isError").is_none(), "{created}");
    // Nesting rides the same tools: a child at create, deltas at update.
    let nested = tool(
        2,
        "create_group",
        json!({"name": "bar", "groups": ["drinks"]}),
    );
    assert!(nested.get("isError").is_none(), "{nested}");

    let listed = tool(3, "list_groups", json!({}));
    let text = listed["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("\"drinks\""), "{text}");
    assert!(text.contains("\"sake\""), "{text}");

    let updated = tool(
        4,
        "update_group",
        json!({"name": "drinks", "remove_contexts": ["sake"]}),
    );
    assert!(updated.get("isError").is_none(), "{updated}");
    let text = updated["content"][0]["text"].as_str().unwrap();
    assert!(!text.contains("\"sake\""), "{text}");

    let deleted = tool(5, "delete_group", json!({"name": "drinks"}));
    assert!(deleted.get("isError").is_none(), "{deleted}");
    let listed = tool(6, "list_groups", json!({}));
    let text = listed["content"][0]["text"].as_str().unwrap();
    // The child's deletion also swept it out of "bar", so the name is
    // gone from the whole directory, parent row included.
    assert!(!text.contains("\"drinks\""), "{text}");
    assert!(text.contains("\"bar\""), "{text}");

    // A failing group tool travels as isError content with the API's
    // machine code visible to the agent.
    let failed = tool(7, "delete_group", json!({"name": "drinks"}));
    assert_eq!(failed["isError"], json!(true), "{failed}");
    let text = failed["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("HTTP 404"), "{text}");
}

/// The changelog's stated limitation, pinned: compaction — the live
/// endpoint and the offline CLI both — leaves `.group` files
/// byte-for-byte alone. Groups hold nothing to compact, and a rewrite
/// here would be a regression in disguise.
#[test]
fn compact_leaves_group_files_byte_for_byte() {
    let server = Server::start("compact-groups");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "keep.md"},
            {"subject": "蔵", "label": "廃止銘柄", "object": "旧銘", "weight": 1.0, "source": "gone.md"},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources/retract",
        Some(json!({"source": "gone.md"})),
    );
    server.ok(
        "PUT",
        "/groups/kura",
        Some(json!({"description": "蔵元一式", "contexts": ["sake"]})),
    );
    let group_file = server.data_dir.join("kura.group");
    let before = std::fs::read(&group_file).expect("the group file must exist");

    // The live endpoint rewrites the context image, not the group file.
    let shed = server.ok("POST", "/contexts/sake/compact", None);
    assert_eq!(shed["dead_edges"], json!(1), "{shed}");
    let after_live = std::fs::read(&group_file).expect("the group file must survive");
    assert_eq!(
        before, after_live,
        "live compact must not touch group files"
    );

    // The offline CLI sweep over the whole directory: same statement.
    let data_dir = server.stop_gracefully();
    let compacted = Command::new(env!("CARGO_BIN_EXE_taguru"))
        .args(["compact"])
        .env("TAGURU_DATA_DIR", &data_dir)
        .env_remove("TAGURU_CONFIG")
        .env_remove("TAGURU_EMBED_URL")
        .output()
        .expect("binary must run");
    assert_eq!(compacted.status.code(), Some(0), "{compacted:?}");
    let after_cli = std::fs::read(&group_file).expect("the group file must survive");
    assert_eq!(
        before, after_cli,
        "offline compact must not touch group files"
    );

    // And the untouched record still boots: the group answers as stored.
    let server = Server::start_on("compact-groups-reboot", data_dir);
    let row = server.ok("GET", "/groups/kura", None);
    assert_eq!(row["contexts"], json!(["sake"]), "{row}");
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// The changelog's other stated limitation, pinned: a DELETE whose
/// unlink fails answers 500, says the group will reappear, drops it
/// from the live directory — and the next boot really does resurface
/// it, because the file survived as the on-disk truth.
#[cfg(unix)]
#[test]
fn a_failed_group_unlink_resurfaces_the_group_at_restart() {
    use std::os::unix::fs::PermissionsExt;

    let server = Server::start("group-unlink-fail");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "PUT",
        "/groups/kura",
        Some(json!({"description": "蔵元一式", "contexts": ["sake"]})),
    );
    // Nothing dirty may remain: the flusher must not collide with the
    // read-only window below.
    server.ok("POST", "/flush", None);

    // Unlink needs write permission on the PARENT directory; freezing
    // the data directory makes exactly the unlink fail.
    let frozen = std::fs::Permissions::from_mode(0o555);
    std::fs::set_permissions(&server.data_dir, frozen).expect("chmod must apply");
    let (status, refusal) = server.call("DELETE", "/groups/kura", None);
    let restored = std::fs::Permissions::from_mode(0o755);
    std::fs::set_permissions(&server.data_dir, restored).expect("chmod must restore");
    assert_eq!(status, 500, "{refusal}");
    assert_eq!(refusal["code"], json!("internal"), "{refusal}");
    assert!(
        refusal["error"]
            .as_str()
            .unwrap()
            .contains("reappears at the next restart"),
        "{refusal}"
    );

    // The live directory dropped the record, as the error admits…
    let (status, missing) = server.call("GET", "/groups/kura", None);
    assert_eq!(status, 404, "{missing}");

    // …and the surviving file resurfaces it at the next boot.
    let data_dir = server.stop_gracefully();
    assert!(
        data_dir.join("kura.group").exists(),
        "the unlink must have failed"
    );
    let server = Server::start_on("group-unlink-reboot", data_dir);
    let row = server.ok("GET", "/groups/kura", None);
    assert_eq!(row["contexts"], json!(["sake"]), "{row}");
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// A golden-question floor for cross-context retrieval: the 青嶺
/// corpus split across a brewery context and a region context, grouped,
/// and asked questions whose needed facts straddle the split. The
/// plumbing tests above prove the merge mechanics; this pins that a
/// question spanning contexts actually comes back whole.
#[test]
fn cross_context_search_answers_questions_that_straddle_the_split() {
    let server = Server::start("cross-golden");
    server.ok(
        "PUT",
        "/contexts/brewery",
        Some(json!({"description": "青嶺酒造という蔵元の知識"})),
    );
    server.ok(
        "PUT",
        "/contexts/region",
        Some(json!({"description": "霧沢町という土地の知識"})),
    );
    // 蔵元の事実は brewery に、土地の事実は region に — どちらの
    // コンテキストも単独では下の質問に答え切れない。
    server.ok(
        "POST",
        "/contexts/brewery/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "所在地", "object": "霧沢町", "weight": 1.0, "source": "第1段落"},
            {"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "source": "第1段落"},
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0, "source": "第1段落"},
            {"subject": "青嶺酒造", "label": "開く", "object": "蔵開きの祭り", "weight": 1.0, "source": "第5段落"},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/region/associations",
        Some(json!([
            {"subject": "霧沢町", "label": "所在する県", "object": "雲居県", "weight": 1.0, "source": "地誌1"},
            {"subject": "霧沢町", "label": "力を入れる", "object": "酒蔵観光", "weight": 1.0, "source": "地誌2"},
            {"subject": "蔵開きの祭り", "label": "時期", "object": "毎年2月", "weight": 1.0, "source": "地誌2"},
        ])),
    );
    server.ok(
        "PUT",
        "/groups/kirisawa",
        Some(json!({"description": "霧沢の酒", "contexts": ["brewery", "region"]})),
    );

    // 「青嶺酒造はどの県にあるか」— 所在地 (brewery) と 所在する県
    // (region) の両方が要る。ひとつの group 検索で両方が、それぞれの
    // 出所タグ付きで返ること。
    let answer = server.ok(
        "POST",
        "/query",
        Some(json!({
            "groups": ["kirisawa"],
            "subject": ["青嶺酒造", "霧沢町"],
            "label": ["所在地", "所在する県"],
        })),
    );
    assert_eq!(answer["total"], json!(2), "{answer}");
    let matches = answer["matches"].as_array().unwrap();
    let fact = |subject: &str| {
        matches
            .iter()
            .find(|m| m["subject"] == json!(subject))
            .unwrap_or_else(|| panic!("missing fact for {subject}: {answer}"))
    };
    assert_eq!(fact("青嶺酒造")["context"], json!("brewery"), "{answer}");
    assert_eq!(fact("青嶺酒造")["object"], json!("霧沢町"), "{answer}");
    assert_eq!(fact("霧沢町")["context"], json!("region"), "{answer}");
    assert_eq!(fact("霧沢町")["object"], json!("雲居県"), "{answer}");

    // 「蔵開きの祭りについて知っていること」— recall がグループ越しに
    // 両コンテキストの事実を集める (brewery: 開く, region: 時期)。
    let recalled = server.ok(
        "POST",
        "/recall",
        Some(json!({"groups": ["kirisawa"], "cue": "蔵開きの祭り"})),
    );
    assert_eq!(recalled["total"], json!(2), "{recalled}");
    let contexts: Vec<&str> = recalled["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["context"].as_str().unwrap())
        .collect();
    assert!(
        contexts.contains(&"brewery") && contexts.contains(&"region"),
        "{recalled}"
    );
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// doc2query on a lexical-only server (no embedding provider — the
/// default deployment): the attached question's wording, absent from
/// the paragraph itself, still lands the search on that paragraph
/// through the BM25 lane. Before this rode the index, questions were
/// stored and functionally inert without `TAGURU_EMBED_PASSAGES`.
#[test]
fn doc2query_questions_land_lexically_without_embeddings() {
    let server = Server::start("doc2query-lexical");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "蔵の知識"})),
    );
    let stored = server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {
                "doc": "精米歩合は50パーセントまで磨く。\n\n仕込み水は雲居山の伏流水を使う。"
            },
            "questions": {"doc": [
                {"paragraph": 0, "question": "米はどれくらい削るのか"}
            ]}
        })),
    );
    assert_eq!(stored["questions_stored"], 1, "{stored}");

    // The query shares wording with the question only — 「削る」 never
    // appears in either paragraph.
    let hits = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "米をどれくらい削る?"})),
    );
    let hits = hits.as_array().unwrap();
    assert!(!hits.is_empty(), "the question's terms must land the hit");
    assert_eq!(hits[0]["source"], json!("doc"), "{hits:?}");
    assert_eq!(hits[0]["paragraph"], json!(0), "{hits:?}");
    assert!(
        hits[0]["lanes"]["bm25"].is_object(),
        "the evidence must be lexical: {hits:?}"
    );

    // Replacing the source with a question-less revision withdraws the
    // question's terms with it (the index's wholesale-replacement unit).
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {
                "doc": "精米歩合は50パーセントまで磨く。\n\n仕込み水は雲居山の伏流水を使う。"
            }
        })),
    );
    let hits = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "米をどれくらい削る?"})),
    );
    assert_eq!(
        hits.as_array().unwrap().len(),
        0,
        "without the question the wording matches nothing: {hits}"
    );
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// The single-association retraction end to end: a write key
/// suffices where a read key is refused, aliases resolve, the outcome
/// reports found-nothing honestly, the WAL makes the withdrawal
/// survive a hard kill, and the MCP surface reaches the same handler.
#[test]
fn one_association_retracts_over_http_and_survives_a_hard_kill() {
    let server = Server::start_with_env(
        "assoc-retract",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,scribe:wtok,reader:rtok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"scribe": "write", "reader": "read"}"#,
            ),
        ],
    );
    let write = Some("wtok");
    let (status, _) = server.call_with_token(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "d"})),
        write,
    );
    assert_eq!(status, 200);
    server.call_with_token(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0, "source": "doc1"},
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0, "source": "doc2"},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "doc1"},
        ])),
        write,
    );
    server.call_with_token(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"concepts": {"Aomine Brewery": "青嶺酒造"}, "labels": {}})),
        write,
    );

    // Reads cannot retract.
    let (status, refused) = server.call_with_token(
        "POST",
        "/contexts/sake/associations/retract",
        Some(json!({"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺"})),
        Some("rtok"),
    );
    assert_eq!(status, 403, "{refused}");

    // The write key retracts through an alias; both sources' shares go.
    let (status, outcome) = server.call_with_token(
        "POST",
        "/contexts/sake/associations/retract",
        Some(json!({"subject": "Aomine Brewery", "label": "代表銘柄", "object": "青嶺"})),
        write,
    );
    assert_eq!(status, 200, "{outcome}");
    assert_eq!(outcome["result"]["retracted"], json!(true), "{outcome}");
    assert_eq!(
        outcome["result"]["attributions_removed"],
        json!(2),
        "{outcome}"
    );

    // Found-nothing honesty: the second retraction changed nothing.
    let (status, outcome) = server.call_with_token(
        "POST",
        "/contexts/sake/associations/retract",
        Some(json!({"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺"})),
        write,
    );
    assert_eq!(status, 200, "{outcome}");
    assert_eq!(outcome["result"]["retracted"], json!(false), "{outcome}");
    assert_eq!(
        outcome["result"]["attributions_removed"],
        json!(0),
        "{outcome}"
    );

    // The edge row stays visible at weight 0 (compaction sheds it);
    // the same document's other fact is untouched; activate no longer
    // carries the dead edge.
    let (_, queried) = server.call_with_token(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造"})),
        write,
    );
    let matches = queried["result"]["matches"].as_array().unwrap();
    let by_label = |label: &str| {
        matches
            .iter()
            .find(|m| m["label"] == json!(label))
            .unwrap_or_else(|| panic!("missing {label}: {queried}"))
    };
    assert_eq!(by_label("代表銘柄")["weight"], json!(0.0), "{queried}");
    assert_eq!(by_label("代表銘柄")["count"], json!(0), "{queried}");
    assert_eq!(by_label("杜氏")["weight"], json!(1.0), "{queried}");
    let (_, activated) = server.call_with_token(
        "POST",
        "/contexts/sake/activate",
        Some(json!({"origins": ["青嶺酒造"]})),
        write,
    );
    let carried: Vec<&str> = activated["result"]["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["association"]["label"].as_str().unwrap())
        .collect();
    assert!(
        !carried.contains(&"代表銘柄") && carried.contains(&"杜氏"),
        "{activated}"
    );

    // MCP reaches the same handler under the same tool vocabulary.
    let (status, answer) = server.call_with_token(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "retract_association",
                               "arguments": {"context": "sake", "subject": "青嶺酒造",
                                              "label": "杜氏", "object": "高瀬"}}})),
        write,
    );
    assert_eq!(status, 200, "{answer}");
    let text = answer["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("\"retracted\":true"), "{text}");

    // SIGKILL: no shutdown flush — the WAL alone must carry both
    // retractions across the boot.
    let data_dir = server.stop_hard();
    let server = Server::start_on("assoc-retract-reboot", data_dir);
    let (_, queried) = server.call_with_token(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造"})),
        write,
    );
    for matched in queried["result"]["matches"].as_array().unwrap() {
        assert_eq!(matched["weight"], json!(0.0), "{queried}");
        assert_eq!(matched["count"], json!(0), "{queried}");
    }
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// The operator verbs ride MCP too: flush names what it persisted and
/// the exports hand back the same streams the HTTP routes serve — an
/// agent can tend its own backup discipline (flush → export → import)
/// without leaving the tool surface. Admin gating carries over from
/// the routes the tools map onto.
#[test]
fn flush_and_export_ride_the_mcp_transport() {
    // A one-hour flush interval keeps the periodic flusher out of the
    // race: the dirty context below stays dirty until the tool runs.
    let server = Server::start_with_env("mcp-ops", &[("TAGURU_FLUSH_SECS", "3600")]);
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0, "source": "doc"},
        ])),
    );
    server.ok(
        "PUT",
        "/groups/kura",
        Some(json!({"description": "蔵元一式", "contexts": ["sake"]})),
    );

    let tool = |id: u64, name: &str, arguments: Value| server.call_tool(id, name, arguments);

    let flushed = tool(1, "flush", json!({}));
    assert!(flushed.get("isError").is_none(), "{flushed}");
    let text = flushed["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("sake"), "{text}");

    let exported = tool(2, "export_context", json!({"context": "sake"}));
    assert!(exported.get("isError").is_none(), "{exported}");
    let text = exported["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("taguru_batch"), "{text}");
    assert!(text.contains("青嶺酒造"), "{text}");

    let group = tool(3, "export_group", json!({"name": "kura"}));
    assert!(group.get("isError").is_none(), "{group}");
    let text = group["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("taguru_group"), "{text}");
    assert!(text.contains("\"kura\""), "{text}");
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// A tool result too big to buffer whole (export_context on a context
/// with enough associations) is refused with the two uncapped escape
/// hatches named — while the raw HTTP export route it names stays
/// completely unaffected by the MCP cap.
#[test]
fn an_oversized_mcp_tool_result_is_capped_but_the_raw_export_route_is_not() {
    let server =
        Server::start_with_env("mcp-result-cap", &[("TAGURU_MCP_MAX_RESULT_BYTES", "1024")]);
    server.ok("PUT", "/contexts/big", Some(json!({"description": "d"})));
    let batch: Vec<Value> = (0..200)
        .map(|i| {
            json!({"subject": format!("s{i}"), "label": "rel", "object": format!("o{i}"),
                   "weight": 1.0, "source": "doc"})
        })
        .collect();
    server.ok(
        "POST",
        "/contexts/big/associations",
        Some(Value::Array(batch)),
    );

    let reply = server.call_tool(1, "export_context", json!({"context": "big"}));
    assert_eq!(reply["isError"], true, "{reply}");
    let text = reply["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("GET /contexts/{name}/export"), "{text}");
    assert!(text.contains("taguru export"), "{text}");

    // The raw HTTP export route this message points at is uncapped —
    // the same 200-association context exports whole over it.
    let (status, exported) = server.call("GET", "/contexts/big/export", None);
    assert_eq!(status, 200, "{exported}");
    let stream = exported.as_str().expect("export body is JSONL");
    assert!(stream.contains("\"s199\""), "{stream}");
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// The MCP operator verbs keep their routes' roles: the outer /mcp
/// gate admits any read-capable key, but the inner dispatch re-checks
/// the mapped route — flush stays admin.
#[test]
fn the_mcp_flush_tool_stays_admin_gated() {
    let server = Server::start_with_env(
        "mcp-ops-auth",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,scribe:wtok"),
            ("TAGURU_KEY_SCOPES", r#"{"scribe": "write"}"#),
        ],
    );
    let call = |token: &str| {
        let (status, answer) = server.call_with_token(
            "POST",
            "/mcp",
            Some(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": "flush", "arguments": {}}})),
            Some(token),
        );
        assert_eq!(status, 200, "{answer}");
        answer["result"].clone()
    };

    let refused = call("wtok");
    assert_eq!(refused["isError"], json!(true), "{refused}");
    let text = refused["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("HTTP 403"), "{text}");

    let allowed = call("atok");
    assert!(allowed.get("isError").is_none(), "{allowed}");
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// The search-explain decision tree (#75): one call names the first
/// verdict that applies — stored at all, sharing any term, ranked but
/// cut off, or served — with the evidence that makes the verdict
/// checkable. The 表記ゆれ case is the one the endpoint exists for:
/// the paragraph spells 酒蔵, the query spells 酒造, and only seeing
/// both term tables side by side says so.
#[test]
fn search_explain_names_the_first_verdict_that_applies() {
    let server = Server::start("search-explain");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "蔵の知識"})),
    );
    // docs/many.md: seven short twin paragraphs plus one long, diluted
    // straggler — everything shares the 霧沢 bigram, so the straggler
    // scores, but dead last, past the default limit of 5.
    let straggler = "霧沢について、この段落は他の段落よりも長く、余計な語を\
                     たくさん含んでいるため、字数あたりの一致密度が下がって\
                     順位は最下位に沈む。";
    let twins: Vec<String> = (1..=7)
        .map(|n| format!("霧沢と霧沢の里、その{n}。"))
        .collect();
    let many = format!("{}\n\n{straggler}", twins.join("\n\n"));
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {
            "docs/kura.md": "青嶺酒造は雲居県の蔵元である。\n\n\
                原料米には山田錦を使い、精米歩合は50パーセントまで磨く。",
            "docs/kuramoto.md": "その酒蔵は谷あいにある。",
            "docs/many.md": many,
        }})),
    );

    // Verdict: served. The best showing is chosen when no paragraph is
    // named, and the per-term table carries the addends that put it
    // there.
    let hits = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "精米歩合はどこまで磨く?"})),
    );
    assert_eq!(hits[0]["source"], json!("docs/kura.md"));
    let served = server.ok(
        "POST",
        "/contexts/sake/sources/search/explain",
        Some(json!({"query": "精米歩合はどこまで磨く?", "source": "docs/kura.md"})),
    );
    assert_eq!(served["verdict"], json!("served"), "{served}");
    assert_eq!(served["paragraph"], hits[0]["paragraph"]);
    assert_eq!(served["paragraph_named"], json!(false));
    assert_eq!(served["ranking"]["served"], json!(true));
    assert_eq!(served["ranking"]["rank"], json!(1));
    assert_eq!(served["ranking"]["fused"], json!(false));
    assert!(
        served["bm25"]["terms"]
            .as_array()
            .unwrap()
            .iter()
            .any(|term| term["contribution"].as_f64().unwrap() > 0.0
                && term["df"].as_u64().unwrap() >= 1),
        "{served}"
    );
    // The vector lane names why it did not run.
    assert_eq!(served["vector"]["ran"], json!(false));
    assert!(
        served["vector"]["reason"]
            .as_str()
            .unwrap()
            .contains("no embedding provider"),
        "{served}"
    );

    // Verdict: below_cutoff. The straggler ranks past the default
    // limit; the reported limit_to_reach is VERIFIED — re-searching at
    // it actually surfaces the paragraph.
    let cutoff = server.ok(
        "POST",
        "/contexts/sake/sources/search/explain",
        Some(json!({"query": "霧沢", "source": "docs/many.md", "paragraph": 7})),
    );
    assert_eq!(cutoff["verdict"], json!("below_cutoff"), "{cutoff}");
    let rank = cutoff["ranking"]["rank"].as_u64().unwrap();
    assert!(
        rank > 5,
        "the straggler must rank past the default limit: {cutoff}"
    );
    assert_eq!(cutoff["ranking"]["served"], json!(false));
    assert!(cutoff["ranking"]["cutoff_score"].as_f64().unwrap() > 0.0);
    let reach = cutoff["ranking"]["limit_to_reach"].as_u64().unwrap();
    assert!(reach >= rank, "{cutoff}");
    let wider = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "霧沢", "limit": reach})),
    );
    assert!(
        wider
            .as_array()
            .unwrap()
            .iter()
            .any(|hit| hit["source"] == json!("docs/many.md") && hit["paragraph"] == json!(7)),
        "limit_to_reach must actually reach it: {wider}"
    );
    // The straggler's own term table shows the match that was too weak.
    assert!(
        cutoff["bm25"]["terms"]
            .as_array()
            .unwrap()
            .iter()
            .any(|term| term["term"] == json!("霧沢") && term["tf"].as_f64().unwrap() > 0.0),
        "{cutoff}"
    );

    // Verdict: no_term_overlap — the 表記ゆれ case. The query spells
    // 酒造, this source spells 酒蔵; both spellings sit in the answer.
    let overlap = server.ok(
        "POST",
        "/contexts/sake/sources/search/explain",
        Some(json!({"query": "酒造", "source": "docs/kuramoto.md"})),
    );
    assert_eq!(overlap["verdict"], json!("no_term_overlap"), "{overlap}");
    assert_eq!(overlap["query_terms"], json!(["酒造"]));
    assert!(
        overlap["paragraph_terms"]
            .as_array()
            .unwrap()
            .contains(&json!("酒蔵")),
        "{overlap}"
    );
    assert!(
        overlap["summary"]
            .as_str()
            .unwrap()
            .contains("shares no term"),
        "{overlap}"
    );

    // Verdict: not_stored — and retraction lands in the same verdict,
    // because the store keeps no tombstone history to tell them apart.
    let ghost = server.ok(
        "POST",
        "/contexts/sake/sources/search/explain",
        Some(json!({"query": "霧沢", "source": "docs/ghost.md"})),
    );
    assert_eq!(ghost["verdict"], json!("not_stored"), "{ghost}");
    assert!(
        ghost["summary"].as_str().unwrap().contains("retracted"),
        "{ghost}"
    );
    server.ok(
        "POST",
        "/contexts/sake/sources/retract",
        Some(json!({"source": "docs/kuramoto.md"})),
    );
    let retracted = server.ok(
        "POST",
        "/contexts/sake/sources/search/explain",
        Some(json!({"query": "酒造", "source": "docs/kuramoto.md"})),
    );
    assert_eq!(retracted["verdict"], json!("not_stored"), "{retracted}");

    // Verdict: paragraph_out_of_range, with the range that would fit.
    let out = server.ok(
        "POST",
        "/contexts/sake/sources/search/explain",
        Some(json!({"query": "霧沢", "source": "docs/kura.md", "paragraph": 9})),
    );
    assert_eq!(out["verdict"], json!("paragraph_out_of_range"), "{out}");
    assert_eq!(out["paragraphs"], json!(2));

    // Verdict: no_query_terms — punctuation tokenizes to nothing.
    let empty = server.ok(
        "POST",
        "/contexts/sake/sources/search/explain",
        Some(json!({"query": "、。", "source": "docs/kura.md"})),
    );
    assert_eq!(empty["verdict"], json!("no_query_terms"), "{empty}");

    // Unknown context stays the outer 404, never a verdict.
    let (status, _) = server.call(
        "POST",
        "/contexts/nazo/sources/search/explain",
        Some(json!({"query": "霧沢", "source": "docs/kura.md"})),
    );
    assert_eq!(status, 404);
}

/// The resolve-explain decision tree (#75): membership first, then the
/// same tiers/floors/limits the resolve endpoint runs, with the
/// expected name located in each. The exact-tier shortcut gets its own
/// verdict — a cue that IS another stored spelling never scores
/// anything else, which no floor tweak can fix.
#[test]
fn resolve_explain_names_the_first_verdict_that_applies() {
    let server = Server::start("resolve-explain");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "蔵の知識"})),
    );
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0},
            {"subject": "青嶺酒造", "label": "原料米", "object": "山田錦", "weight": 1.0},
            {"subject": "幻の蔵元", "label": "所在", "object": "山田町", "weight": 1.0},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"concepts": {"Aomine": "青嶺酒造"}})),
    );

    // Verdict: served — and an alias expectation reports its canonical.
    let served = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": "青嶺酒造", "expected": "青嶺酒造"})),
    );
    assert_eq!(served["verdict"], json!("served"), "{served}");
    assert_eq!(served["expected_kind"], json!("exact"));
    assert_eq!(served["ranking"]["rank"], json!(1));
    assert_eq!(served["lexical"]["confident"], json!(true));
    assert_eq!(served["semantic"]["entered"], json!(false));
    let alias = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": "青嶺酒", "expected": "Aomine"})),
    );
    assert_eq!(alias["verdict"], json!("served"), "{alias}");
    assert_eq!(alias["canonical"], json!("青嶺酒造"));
    assert_eq!(alias["expected_kind"], json!("alias"));

    // Verdict: cue_resolved_exactly — the exact tier answers alone.
    let eclipsed = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": "高瀬", "expected": "青嶺酒造"})),
    );
    assert_eq!(
        eclipsed["verdict"],
        json!("cue_resolved_exactly"),
        "{eclipsed}"
    );
    assert!(
        eclipsed["summary"].as_str().unwrap().contains("高瀬"),
        "{eclipsed}"
    );

    // Verdict: below_floor — the actual Dice score against the floor
    // in effect, which is also the floor that would have shown it.
    // 青嶺の酒造り shares the 青嶺/酒造 bigrams with 青嶺酒造 without
    // containing it: a fuzzy 0.5, gated by a request floor of 0.6.
    let floored = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": "青嶺の酒造り", "expected": "青嶺酒造", "dice_floor": 0.6})),
    );
    assert_eq!(floored["verdict"], json!("below_floor"), "{floored}");
    let score = floored["lexical"]["score"].as_f64().unwrap();
    assert!((score - 0.5).abs() < 1e-9, "{floored}");
    assert_eq!(floored["lexical"]["kind"], json!("fuzzy"));
    assert_eq!(floored["lexical"]["floor"], json!(0.6));

    // The dice floor gates ONLY the fuzzy tier: a containment hit
    // sails past any floor, and explain reports the serve, not a
    // fictitious floor refusal.
    let contained = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": "青嶺酒", "expected": "青嶺酒造", "dice_floor": 0.9})),
    );
    assert_eq!(contained["verdict"], json!("served"), "{contained}");
    assert_eq!(contained["lexical"]["kind"], json!("containment"));

    // Verdict: below_cutoff — lost on limit, with a verified way back.
    let cut = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": "山田", "expected": "山田錦", "limit": 1})),
    );
    assert_eq!(cut["verdict"], json!("below_cutoff"), "{cut}");
    assert_eq!(cut["ranking"]["rank"], json!(2));
    let reach = cut["ranking"]["limit_to_reach"].as_u64().unwrap();
    let wider = server.ok(
        "POST",
        "/contexts/sake/resolve",
        Some(json!({"cue": "山田", "limit": reach})),
    );
    assert!(
        wider
            .as_array()
            .unwrap()
            .iter()
            .any(|candidate| candidate["name"] == json!("山田錦")),
        "limit_to_reach must actually reach it: {wider}"
    );

    // Verdict: semantic_not_run — no lexical relation at all, and the
    // tier that could have found it names why it never ran.
    let semantic = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": "みかん", "expected": "青嶺酒造"})),
    );
    assert_eq!(semantic["verdict"], json!("semantic_not_run"), "{semantic}");
    assert_eq!(semantic["semantic"]["entered"], json!(true));
    assert!(
        semantic["semantic"]["reason"]
            .as_str()
            .unwrap()
            .contains("no embedding provider"),
        "{semantic}"
    );

    // Verdict: not_in_vocabulary — with the nearest stored spellings,
    // so the repair (register an alias) is one step away.
    let missing = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": "青嶺", "expected": "幻の蔵"})),
    );
    assert_eq!(missing["verdict"], json!("not_in_vocabulary"), "{missing}");
    assert_eq!(missing["in_vocabulary"], json!(false));
    assert_eq!(missing["nearest"]["lexical"][0]["name"], json!("幻の蔵元"));

    // The label twin answers through its own route.
    let label = server.ok(
        "POST",
        "/contexts/sake/resolve_label/explain",
        Some(json!({"cue": "杜氏の職", "expected": "杜氏"})),
    );
    assert_eq!(label["verdict"], json!("served"), "{label}");
    assert_eq!(label["canonical"], json!("杜氏"));

    // Unknown context stays the outer 404, never a verdict.
    let (status, _) = server.call(
        "POST",
        "/contexts/nazo/resolve/explain",
        Some(json!({"cue": "青嶺", "expected": "青嶺酒造"})),
    );
    assert_eq!(status, 404);

    // The MCP mirror diagnoses the same miss in one tool call.
    let (_, reply) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                    "params": {"name": "explain_resolve",
                               "arguments": {"context": "sake", "cue": "青嶺", "expected": "幻の蔵"}}})),
    );
    assert!(reply["result"].get("isError").is_none(), "{reply}");
    let text = reply["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("not_in_vocabulary"), "{text}");
    assert!(text.contains("幻の蔵元"), "{text}");

    // And the search mirror rides the same registry.
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"docs/kura.md": "その酒蔵は谷あいにある。"}})),
    );
    let (_, reply) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 8, "method": "tools/call",
                    "params": {"name": "explain_search",
                               "arguments": {"context": "sake", "query": "酒造",
                                             "source": "docs/kura.md"}}})),
    );
    assert!(reply["result"].get("isError").is_none(), "{reply}");
    let text = reply["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("no_term_overlap"), "{text}");
    assert!(text.contains("酒蔵"), "{text}");
}
