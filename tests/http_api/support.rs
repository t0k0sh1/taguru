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
        let data_dir =
            std::env::temp_dir().join(format!("taguru-http-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&data_dir);
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

    /// Spawns `taguru route` over the given map contents (written to a
    /// scratch file). The returned handle's `data_dir` is that scratch
    /// directory — the router itself holds no data; the field only
    /// keeps Drop's cleanup working.
    pub fn start_router(tag: &str, map_contents: &str, extra_env: &[(&str, &str)]) -> Self {
        let dir = std::env::temp_dir().join(format!("taguru-router-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("router scratch dir must be creatable");
        let map_path = dir.join("route-map");
        std::fs::write(&map_path, map_contents).expect("route map must be writable");

        let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
        command.arg("route");
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
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    common::scrub_taguru_env(&mut command)
        .arg("import")
        .env("TAGURU_DATA_DIR", data_dir)
        // import-only vars scrub_taguru_env doesn't know about.
        .env_remove("TAGURU_WAL")
        .env_remove("TAGURU_WAL_MAX_BYTES")
        .env_remove("TAGURU_CACHE_BYTES")
        .args(args);
    let output = command.output().expect("import must run");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// A scratch directory for batch files, separate from any data dir.
pub fn batch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("taguru-batches-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
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
