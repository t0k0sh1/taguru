//! Shared server harness for the http_api integration-test cluster: the
//! spawned-binary wrapper ([`Server`]), its plain HTTP client helpers, and
//! the offline-import helpers ([`run_import`], [`batch_dir`], [`post_import`])
//! more than one cluster file needs.

use std::path::PathBuf;
use std::process::{Child, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

#[path = "../common/spawn.rs"]
pub(crate) mod common;

/// [`common::read_listen_line`], then spawns a thread draining whatever
/// follows so the child never blocks writing to a full stdout pipe —
/// for the long-lived servers this harness keeps running past the
/// listen line (as opposed to the short-lived spawns elsewhere in this
/// directory, which read the line and move on).
fn read_listen_line_and_drain(label: &str, stdout: ChildStdout) -> String {
    let (addr, lines) = common::read_listen_line(label, stdout);
    std::thread::spawn(move || for _ in lines {});
    addr
}

/// One running server on its own port and data directory, killed and
/// cleaned up on drop.
pub struct Server {
    child: Child,
    pub base: String,
    pub data_dir: PathBuf,
}

impl Server {
    pub fn start(tag: &str) -> Self {
        Self::start_with_env(tag, &[])
    }

    pub fn start_with_env(tag: &str, extra_env: &[(&str, &str)]) -> Self {
        let data_dir = common::scratch_dir(&format!("http-{tag}"));
        Self::spawn(tag, data_dir, extra_env)
    }

    pub fn start_on(tag: &str, data_dir: PathBuf) -> Self {
        Self::spawn(tag, data_dir, &[])
    }

    /// [`Server::start_on`] with extra environment — for reusing a
    /// directory whose meaning depends on flags (a replica's cache
    /// promoted to a writer, a takeover acknowledgment).
    pub fn start_on_with_env(tag: &str, data_dir: PathBuf, extra_env: &[(&str, &str)]) -> Self {
        Self::spawn(tag, data_dir, extra_env)
    }

    /// [`Server::start_with_env`] plus `--config FILE` and stderr
    /// captured to a file — for the keyring-reload tests, which
    /// rewrite the config at runtime and then assert on the audit
    /// lines the reload leaves behind.
    pub fn start_with_config(
        tag: &str,
        config: &std::path::Path,
        stderr_to: &std::path::Path,
        extra_env: &[(&str, &str)],
    ) -> Self {
        let data_dir = common::scratch_dir(&format!("http-{tag}"));
        let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
        command.arg("--config").arg(config);
        common::scrub_taguru_env(&mut command)
            .env("TAGURU_ADDR", "127.0.0.1:0")
            .env("TAGURU_DATA_DIR", &data_dir)
            .env("TAGURU_FLUSH_SECS", "1");
        for (key, value) in extra_env {
            command.env(key, value);
        }
        let stderr =
            std::fs::File::create(stderr_to).expect("stderr capture file must be creatable");
        let mut child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::from(stderr))
            .spawn()
            .expect("server binary must spawn");
        let stdout = child.stdout.take().expect("stdout must be piped");
        let addr = read_listen_line_and_drain(&format!("server '{tag}'"), stdout);
        Self {
            child,
            base: format!("http://{addr}"),
            data_dir,
        }
    }

    /// Sends `signal` (e.g. "-HUP") to the running server without
    /// stopping it — the keyring-reload trigger.
    #[cfg(unix)]
    pub fn signal(&self, signal: &str) {
        let pid = self.child.id().to_string();
        Command::new("kill")
            .args([signal, &pid])
            .status()
            .expect("kill must run");
    }

    /// Spawns `taguru router` over the given map contents (written to a
    /// scratch file). The returned handle's `data_dir` is that scratch
    /// directory — the router itself holds no data; the field only
    /// keeps Drop's cleanup working.
    pub fn start_router(tag: &str, map_contents: &str, extra_env: &[(&str, &str)]) -> Self {
        Self::start_router_via(tag, map_contents, extra_env, "router")
    }

    /// Same as [`start_router`](Self::start_router), but naming the
    /// subcommand explicitly — `"route"` exercises the deprecated
    /// alias's real dispatch (issue #248 item 9), not just its
    /// `--help` text.
    pub fn start_router_via(
        tag: &str,
        map_contents: &str,
        extra_env: &[(&str, &str)],
        subcommand: &str,
    ) -> Self {
        let dir = common::scratch_dir(&format!("router-{tag}"));
        std::fs::create_dir_all(&dir).expect("router scratch dir must be creatable");
        let map_path = dir.join("route-map");
        std::fs::write(&map_path, map_contents).expect("route map must be writable");

        let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
        command.arg(subcommand);
        common::scrub_taguru_env(&mut command)
            .env("TAGURU_ADDR", "127.0.0.1:0")
            .env("TAGURU_ROUTE_MAP", &map_path);
        for (key, value) in extra_env {
            command.env(key, value);
        }
        let mut child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("router binary must spawn");
        let stdout = child.stdout.take().expect("stdout must be piped");
        let addr = read_listen_line_and_drain(&format!("router '{tag}'"), stdout);
        Self {
            child,
            base: format!("http://{addr}"),
            data_dir: dir,
        }
    }

    fn spawn(tag: &str, data_dir: PathBuf, extra_env: &[(&str, &str)]) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
        common::scrub_taguru_env(&mut command)
            .env("TAGURU_ADDR", "127.0.0.1:0")
            .env("TAGURU_DATA_DIR", &data_dir)
            .env("TAGURU_FLUSH_SECS", "1");
        for (key, value) in extra_env {
            command.env(key, value);
        }
        let mut child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("server binary must spawn");

        let stdout = child.stdout.take().expect("stdout must be piped");
        let addr = read_listen_line_and_drain(&format!("server '{tag}'"), stdout);

        Self {
            child,
            base: format!("http://{addr}"),
            data_dir,
        }
    }

    /// One request; returns (status, parsed body). Non-JSON bodies come
    /// back as JSON strings.
    pub fn call(&self, method: &str, path: &str, body: Option<Value>) -> (u16, Value) {
        self.call_with_token(method, path, body, None)
    }

    /// [`Server::call`] with an explicit bearer token attached.
    pub fn call_with_token(
        &self,
        method: &str,
        path: &str,
        body: Option<Value>,
        token: Option<&str>,
    ) -> (u16, Value) {
        let mut request = ureq::http::Request::builder()
            .method(method)
            .uri(format!("{}{path}", self.base));
        if let Some(token) = token {
            request = request.header("Authorization", format!("Bearer {token}"));
        }
        let response = match body {
            Some(body) => request
                .header("Content-Type", "application/json")
                .body(body.to_string())
                .map(|request| test_agent().run(request)),
            None => request.body(()).map(|request| test_agent().run(request)),
        };
        finish(response.expect("request must assemble"), method, path)
    }

    /// A raw request: the body goes out as-is, with a Content-Type only
    /// when one is given — for the header-omission cases the JSON
    /// helpers cannot express.
    pub fn call_raw(
        &self,
        method: &str,
        path: &str,
        body: Option<&str>,
        content_type: Option<&str>,
    ) -> (u16, Value) {
        let mut request = ureq::http::Request::builder()
            .method(method)
            .uri(format!("{}{path}", self.base));
        if let Some(content_type) = content_type {
            request = request.header("Content-Type", content_type);
        }
        let response = match body {
            Some(body) => request.body(body).map(|request| test_agent().run(request)),
            None => request.body(()).map(|request| test_agent().run(request)),
        };
        finish(response.expect("request must assemble"), method, path)
    }

    pub fn ok(&self, method: &str, path: &str, body: Option<Value>) -> Value {
        let (status, parsed) = self.call(method, path, body);
        assert_eq!(status, 200, "{method} {path} -> {parsed}");
        parsed["result"].clone()
    }

    /// One MCP `tools/call` round trip: builds the JSON-RPC envelope,
    /// asserts HTTP 200, and hands back the JSON-RPC `result` — whose
    /// `content`/`isError` the caller judges.
    pub fn call_tool(&self, id: u64, name: &str, arguments: Value) -> Value {
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
    pub fn stop_gracefully(self) -> PathBuf {
        self.stop_with("-TERM")
    }

    /// Hard kill (SIGKILL): no shutdown flush, no cleanup — whatever
    /// durability the server claims must come from the disk alone.
    pub fn stop_hard(self) -> PathBuf {
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

/// The tests' HTTP client: 4xx/5xx come back as responses — the
/// helpers assert on status, they never want an error for one.
pub fn test_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .http_status_as_error(false)
        .build()
        .into()
}

/// [`test_agent`], except redirects come back unfollowed — for the
/// OAuth flows that assert on the 303 itself (and must not chase a
/// Location pointing at the real external callback host).
pub fn no_redirect_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .max_redirects(0)
        .http_status_as_error(false)
        .build()
        .into()
}

/// One header as text, "" when absent — ureq 2's `header()` accessor,
/// which half these tests were written against.
pub fn header_text(response: &ureq::http::Response<ureq::Body>, name: &str) -> String {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_string()
}

/// Shared response tail: status plus parsed JSON body (or the raw
/// text when it is not JSON).
pub fn finish(
    response: Result<ureq::http::Response<ureq::Body>, ureq::Error>,
    method: &str,
    path: &str,
) -> (u16, Value) {
    let (status, text) = match response {
        Ok(mut response) => (
            response.status().as_u16(),
            response.body_mut().read_to_string().unwrap_or_default(),
        ),
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

/// Runs `taguru import` against `data_dir`, hermetic the same way the
/// server spawns are: nothing from the developer shell reaches it.
pub fn run_import(data_dir: &std::path::Path, args: &[&str]) -> (i32, String, String) {
    let mut full_args = vec!["import"];
    full_args.extend_from_slice(args);
    run_cli(
        &full_args,
        &[("TAGURU_DATA_DIR", &data_dir.to_string_lossy())],
    )
}

/// Runs any `taguru` subcommand hermetically — nothing from the
/// developer's own shell environment reaches the child. A more general
/// version of [`run_import`] (which fixes the subcommand and
/// `TAGURU_DATA_DIR` itself) for callers that need to pass the
/// subcommand and every flag themselves — e.g. `export --url`.
pub fn run_cli(args: &[&str], extra_env: &[(&str, &str)]) -> (i32, String, String) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    common::scrub_taguru_env(&mut command)
        // import-only vars scrub_taguru_env doesn't know about.
        .env_remove("TAGURU_WAL")
        .env_remove("TAGURU_WAL_MAX_BYTES")
        .env_remove("TAGURU_CACHE_BYTES")
        .args(args);
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let output = command.output().expect("CLI must run");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// Reads one HTTP/1.1 request off `stream`: header lines up to the blank
/// line, then exactly `Content-Length` more body bytes (0 if absent).
/// Header names come back lower-cased. `None` means the connection
/// closed before a header terminator ever showed up — a probe or a
/// half-open connection, not a real request. Shared by every hand-rolled
/// fake server below (`FakeCollector`, `FakeShard`) so the same
/// read-until-`\r\n\r\n`-then-drain-the-body loop exists once.
fn read_http_request(
    stream: &mut std::net::TcpStream,
) -> Option<(std::collections::HashMap<String, String>, Vec<u8>)> {
    use std::io::Read;
    // `FakeCollector`/`FakeShard` serve one connection at a time on a
    // single accept thread — a client that opens a connection and never
    // finishes sending would otherwise block this call forever, stalling
    // every later request to the same fake server.
    let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(30)));
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return None,
            Ok(n) => buffer.extend_from_slice(&chunk[..n]),
        }
        if let Some(at) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
            break at + 4;
        }
    };
    let header_text = String::from_utf8_lossy(&buffer[..header_end]).to_string();
    let mut headers = std::collections::HashMap::new();
    for line in header_text.lines().skip(1) {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }
    let length: usize = headers
        .get("content-length")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    while buffer.len() < header_end + length {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => buffer.extend_from_slice(&chunk[..n]),
        }
    }
    let body = buffer[header_end..].to_vec();
    Some((headers, body))
}

/// Spawns a thread accepting connections on `listener`: each
/// successfully-parsed request goes through `respond`, whose returned
/// bytes are written back verbatim. The one accept loop behind
/// [`FakeCollector`] and [`FakeShard`] — the fakes differ only in what
/// they record and what they answer.
fn spawn_fake_server(
    listener: std::net::TcpListener,
    mut respond: impl FnMut(std::collections::HashMap<String, String>, Vec<u8>) -> Vec<u8>
    + Send
    + 'static,
) {
    use std::io::Write;
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let Some((headers, body)) = read_http_request(&mut stream) else {
                continue;
            };
            let response = respond(headers, body);
            let _ = stream.write_all(&response);
        }
    });
}

/// A single-purpose OTLP/HTTP sink: accepts POSTs, stores every body,
/// answers 200. Runs until the test process exits. Shared by every
/// tracing test that needs a real OTLP wire target — `FakeCollector`
/// exposes the complete exported shape (parent ids, attributes,
/// events, status, resource), which is why ADR 0008 §3 chose it over
/// `opentelemetry_sdk`'s in-memory exporter (which cannot see spans
/// from the spawned server process these tests drive anyway).
pub struct FakeCollector {
    pub endpoint: String,
    bodies: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl FakeCollector {
    pub fn start() -> Self {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("collector must bind");
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let bodies = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = bodies.clone();
        spawn_fake_server(listener, move |_headers, body| {
            sink.lock()
                .unwrap()
                .push(String::from_utf8_lossy(&body).to_string());
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
              Content-Length: 2\r\nConnection: close\r\n\r\n{}"
                .to_vec()
        });
        Self { endpoint, bodies }
    }

    /// Every span object exported so far, flattened across batches.
    pub fn spans(&self) -> Vec<Value> {
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

    /// Every raw request body received so far — for the sentinel test
    /// that must prove a nonce appears NOWHERE in the wire payload,
    /// not just outside the attributes/events [`spans`] already
    /// parses out.
    pub fn raw_bodies(&self) -> Vec<String> {
        self.bodies.lock().unwrap().clone()
    }
}

/// A span's OTLP status code — `tracing-opentelemetry` consumes the
/// `otel.status_code` tracing field entirely into this real `status`
/// object, so it never shows up via [`attribute`] the way an ordinary
/// field does. `0` (UNSET, the default when a span never touched
/// status) / `1` (OK) / `2` (ERROR) per the OTLP `trace.proto`
/// `StatusCode` enum.
pub fn status_code(span: &Value) -> i64 {
    span["status"]["code"].as_i64().unwrap_or(0)
}

/// One attribute value out of the OTLP attribute list shape
/// `[{"key": ..., "value": {"stringValue": ...}}]`.
pub fn attribute<'a>(span: &'a Value, key: &str) -> Option<&'a Value> {
    span["attributes"]
        .as_array()?
        .iter()
        .find(|attribute| attribute["key"] == key)
        .map(|attribute| &attribute["value"])
}

/// One event's `taguru.reason` attribute — `taguru.skip`/`.degrade`/
/// `.cache` events all carry the reason this way (ADR 0008 §7).
pub fn event_reason(event: &Value) -> Option<&str> {
    event["attributes"]
        .as_array()?
        .iter()
        .find(|attribute| attribute["key"] == "taguru.reason")
        .and_then(|attribute| attribute["value"]["stringValue"].as_str())
}

/// Indexes a flat [`FakeCollector::spans`] list by id and by name, so a
/// tracing test can walk parent/child relationships without
/// re-implementing the same linear scan in every assertion.
pub struct SpanTree {
    by_id: std::collections::HashMap<String, Value>,
}

impl SpanTree {
    pub fn new(spans: Vec<Value>) -> Self {
        let by_id = spans
            .into_iter()
            .filter_map(|span| {
                let id = span["spanId"].as_str()?.to_string();
                Some((id, span))
            })
            .collect();
        Self { by_id }
    }

    pub fn by_name<'a>(&'a self, name: &str) -> Vec<&'a Value> {
        self.by_id
            .values()
            .filter(|span| span["name"] == name)
            .collect()
    }

    /// The one span named `name` — panics with every span's name
    /// listed when there isn't exactly one, so a failing assertion
    /// names what WAS exported instead of just "not found".
    pub fn one<'a>(&'a self, name: &str) -> &'a Value {
        let matches = self.by_name(name);
        match matches.as_slice() {
            [span] => span,
            _ => panic!(
                "expected exactly one {name:?} span, found {}; exported names: {:?}",
                matches.len(),
                self.by_id.values().map(|s| &s["name"]).collect::<Vec<_>>()
            ),
        }
    }

    /// Every exported span whose `parentSpanId` is `parent`'s own id.
    pub fn children<'a>(&'a self, parent: &Value) -> Vec<&'a Value> {
        let parent_id = parent["spanId"].as_str().unwrap_or_default();
        self.by_id
            .values()
            .filter(|span| span["parentSpanId"] == parent_id)
            .collect()
    }
}

/// A [`FakeCollector`] twin for the router-propagation tests: answers
/// a canned JSON body and records every inbound request's headers, so
/// a test can assert on what the router actually injected/forwarded
/// rather than the collector's own OTLP shape.
pub struct FakeShard {
    pub endpoint: String,
    requests: std::sync::Arc<std::sync::Mutex<Vec<std::collections::HashMap<String, String>>>>,
}

impl FakeShard {
    pub fn start(body: Value) -> Self {
        Self::start_with_status(200, body)
    }

    /// Same as [`Self::start`], but answering every request with
    /// `status` instead of a hardcoded 200 — for tests exercising a
    /// shard that answers with an HTTP error status (a router-side
    /// `"http_error"` shard outcome, not a transport failure).
    pub fn start_with_status(status: u16, body: Value) -> Self {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("shard must bind");
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let requests = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = requests.clone();
        let body_text = body.to_string();
        spawn_fake_server(listener, move |headers, _body| {
            sink.lock().unwrap().push(headers);
            format!(
                "HTTP/1.1 {status} Status\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                body_text.len(),
                body_text
            )
            .into_bytes()
        });
        Self { endpoint, requests }
    }

    /// Every request's headers, in arrival order.
    pub fn requests(&self) -> Vec<std::collections::HashMap<String, String>> {
        self.requests.lock().unwrap().clone()
    }
}

/// A scratch directory for batch files, separate from any data dir.
pub fn batch_dir(tag: &str) -> PathBuf {
    let dir = common::scratch_dir(&format!("batches-{tag}"));
    std::fs::create_dir_all(&dir).expect("batch dir must be creatable");
    dir
}

/// POST /import with a raw JSONL body (the batch is not a JSON
/// document, so the JSON helpers cannot carry it) and an optional
/// bearer token.
fn post_import_query(
    server: &Server,
    query: &str,
    body: &str,
    token: Option<&str>,
) -> (u16, Value) {
    let mut request = test_agent().post(format!("{}/import{query}", server.base));
    if let Some(token) = token {
        request = request.header("Authorization", format!("Bearer {token}"));
    }
    finish(request.send(body), "POST", "/import")
}

pub fn post_import(server: &Server, body: &str, token: Option<&str>) -> (u16, Value) {
    post_import_query(server, "", body, token)
}

pub fn post_import_dry_run(server: &Server, body: &str, token: Option<&str>) -> (u16, Value) {
    post_import_query(server, "?dry_run=true", body, token)
}
