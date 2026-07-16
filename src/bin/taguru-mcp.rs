//! taguru-mcp: an MCP (Model Context Protocol) stdio server that bridges an
//! LLM agent to a running Taguru HTTP server (`TAGURU_URL`, default
//! http://127.0.0.1:8248).
//!
//! This is the reference client the retrieval service is designed around:
//! the agent on the other side of stdio is the extractor on the write
//! path and the composer on the read path, and this bridge hands it the
//! structural tools plus the discipline. The full playbook (ingest
//! discipline, retrieval loop) is served as the MCP `instructions`,
//! fetched live from the server's /protocol (falling back to the copy
//! embedded at build time), so the agent learns the protocol the moment
//! it connects.
//!
//! The MCP substance — tool definitions, tool → HTTP routing, JSON-RPC
//! framing — lives in the shared [`mcp`] module (the server's own
//! `POST /mcp` endpoint speaks through the same code); this file is
//! only the stdio transport around it.
//!
//! Run one writer per data directory: this bridge talks to the HTTP
//! server rather than opening the data directory itself, so any number
//! of agents can share one running server.

#[path = "../mcp.rs"]
mod mcp;

use std::collections::HashSet;
use std::io::{BufRead, Read, Write};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};
use serde_json::Value;

const FALLBACK_INSTRUCTIONS: &str = include_str!("../llm-protocol.md");

/// Ceiling on how many ids a `notifications/cancelled` can register
/// before a dispatch thread claims (and clears) its own entry — a
/// cancellation naming an id that is never actually in flight (a
/// stale id, a client bug) would otherwise sit here forever.
const MAX_TRACKED_CANCELLATIONS: usize = 10_000;

fn main() {
    let base = std::env::var("TAGURU_URL").unwrap_or_else(|_| "http://127.0.0.1:8248".to_string());
    let token = std::env::var("TAGURU_API_TOKEN").ok();
    if token.is_none() {
        // Not an error — the server may run unauthenticated — but the
        // operator who armed the server and forgot the bridge deserves
        // an immediate stderr clue, not a 401 on the first tool call.
        eprintln!("taguru-mcp: TAGURU_API_TOKEN not set; requests go out without credentials");
    }
    // The bridge must outlast the server's own request budget (default
    // 30s, above 60s with embeddings configured), or the agent sees a
    // raw transport error instead of the server's 408 in the error
    // shape. 75s clears both defaults; TAGURU_MCP_TIMEOUT_SECS adjusts
    // it alongside a raised TAGURU_REQUEST_TIMEOUT_SECS.
    let timeout_secs = resolve_timeout_secs(std::env::var("TAGURU_MCP_TIMEOUT_SECS").ok());
    // Shared across the dispatch threads spawned below, one per
    // `tools/call` — an `Arc` rather than a `Clone` derive on `Bridge`
    // since nothing about it needs to vary per thread.
    let bridge = Arc::new(Bridge {
        base: base.trim_end_matches('/').to_string(),
        token,
        agent: bridge_agent(Duration::from_secs(timeout_secs)),
    });

    // The probe runs BEFORE the stdio loop: until it settles, the
    // bridge cannot answer even `initialize`. A dead server fails it in
    // milliseconds (connection refused), but a host that swallows the
    // connection — a firewall dropping SYNs, a stalled tunnel — would
    // hold the full tool-call timeout: 75 seconds of startup silence
    // that an MCP client with a shorter handshake budget reads as a
    // hang and kills. The probe gets its own short ceiling instead;
    // past it, the bundled copy serves and the loop starts. Tool calls
    // keep the long timeout — they run after the handshake, when the
    // client is already talking to us.
    let probe = Bridge {
        base: bridge.base.clone(),
        token: bridge.token.clone(),
        agent: bridge_agent(Duration::from_secs(timeout_secs.min(5))),
    };
    let instructions = probe
        .call("GET", "/protocol", None)
        .unwrap_or_else(|error| {
            // The bundled copy keeps the agent functional, but a dead or
            // misaddressed server should not be discovered one failed tool
            // call at a time.
            eprintln!(
                "taguru-mcp: GET /protocol failed ({error}); serving the bundled copy — \
             is the server up at {}?",
                bridge.base
            );
            FALLBACK_INSTRUCTIONS.to_string()
        });

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = stdin.lock();
    // One JSON-RPC message per line. A client that never sends a newline
    // must not make the bridge buffer without bound, so cap each read; a
    // line past the cap is drained to its newline and skipped, exactly as
    // an undecodable one is. The default must clear the largest LEGAL
    // message, which is an `import` call: its `stream` argument runs to
    // MAX_IMPORT_STREAM_BYTES (32 MiB), and JSON-quoting it into a string
    // argument roughly doubles that (every newline in the NDJSON escapes
    // to `\n`), so 64 MiB is the smallest cap that never rejects a legal
    // import. TAGURU_MCP_MAX_LINE_BYTES raises it for streams whose
    // escaping runs heavier, or lowers it to tighten the memory bound;
    // 0 or unparseable falls back to the default rather than degenerating.
    let max_line_bytes: u64 = std::env::var("TAGURU_MCP_MAX_LINE_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|&bytes| bytes > 0)
        .unwrap_or(64 * 1024 * 1024);
    // `tools/call` is the one dispatch that rides a full HTTP round
    // trip (an import, a recall against a large context), so it runs
    // on its own thread — otherwise a single slow call would leave
    // this loop unable to read the next line for the whole timeout,
    // including a `ping` a client sends just to check the bridge is
    // still alive, or a cancellation for that same slow call. The
    // semaphore bounds how many of those threads actually hold a live
    // HTTP request at once, so a client that pipelines calls without
    // limit cannot fan this bridge out into unbounded connections.
    let max_concurrent_tools =
        resolve_max_concurrent_tools(std::env::var("TAGURU_MCP_MAX_CONCURRENT_TOOLS").ok());
    let concurrency = Arc::new(Semaphore::new(max_concurrent_tools));
    let cancelled: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
    // Every dispatch thread spawned below is detached, so nothing but
    // this Vec stands between `main` returning and a `tools/call` still
    // mid-flight when stdin hits EOF (the client closed its write side
    // right after its last request, which is the ordinary way this loop
    // ends) — without it, the process would tear down mid-HTTP-call and
    // the reply the client is waiting on would simply never arrive.
    // Pruned finished handles on every spawn so a long session's Vec
    // tracks only what is actually still running, not its whole history.
    let mut inflight: Vec<thread::JoinHandle<()>> = Vec::new();
    loop {
        let raw = match read_frame(&mut reader, max_line_bytes) {
            Frame::Eof => break,
            // The line may have carried an id a client now waits on; its
            // bytes are gone (drained by read_frame), so a null-id parse
            // error is all this transport can say — but silence would
            // hang that client.
            Frame::TooLong => {
                eprintln!("taguru-mcp: refusing over-long line");
                emit(
                    &stdout,
                    &mcp::error_response(
                        Value::Null,
                        -32700,
                        format!("line exceeds the {max_line_bytes}-byte frame cap"),
                    ),
                );
                continue;
            }
            Frame::Line(raw) => raw,
        };
        let Ok(text) = std::str::from_utf8(&raw) else {
            eprintln!("taguru-mcp: refusing undecodable line");
            emit(
                &stdout,
                &mcp::error_response(Value::Null, -32700, "line is not UTF-8".to_string()),
            );
            continue;
        };
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        let Ok(message) = serde_json::from_str::<Value>(text) else {
            eprintln!("taguru-mcp: refusing undecodable line");
            emit(
                &stdout,
                &mcp::error_response(Value::Null, -32700, "line is not JSON".to_string()),
            );
            continue;
        };
        if message.is_array() {
            emit(&stdout, &batch_rejected());
            continue;
        }
        if let Some(target_id) = mcp::cancelled_request_id(&message) {
            let mut cancelled = cancelled.lock();
            if cancelled.len() < MAX_TRACKED_CANCELLATIONS {
                cancelled.insert(target_id.to_string());
            }
            continue;
        }
        match mcp::classify(&message) {
            mcp::Message::Request {
                id,
                call: mcp::Call::Tool { name, arguments },
            } => {
                let bridge = Arc::clone(&bridge);
                let concurrency = Arc::clone(&concurrency);
                let cancelled = Arc::clone(&cancelled);
                let response_id = id.to_string();
                inflight.retain(|handle| !handle.is_finished());
                inflight.push(thread::spawn(move || {
                    // A cancel that landed while this call was still
                    // queued skips it entirely: no permit spent, no
                    // request sent for a reply nothing is waiting on.
                    if cancelled.lock().remove(&response_id) {
                        return;
                    }
                    let _permit = concurrency.acquire();
                    let outcome = dispatch_tool(&bridge, &name, &arguments);
                    // ureq has no mid-flight abort, so a cancel that
                    // landed once the call was already running can't
                    // stop the request — but it still stops the
                    // now-unwanted reply from reaching the client.
                    if cancelled.lock().remove(&response_id) {
                        return;
                    }
                    emit(
                        &std::io::stdout(),
                        &mcp::response(id, mcp::tool_response(outcome)),
                    );
                }));
            }
            classified => {
                if let Some(response) = dispatch(&bridge, &instructions, classified) {
                    emit(&stdout, &response);
                }
            }
        }
    }
    // Stdin is exhausted, but threads spawned before that point may
    // still be running their HTTP round trip; every accepted
    // `tools/call` gets the chance to answer (or lose a race to a late
    // cancellation) before the process actually exits.
    for handle in inflight {
        let _ = handle.join();
    }
}

/// One response line out, flushed — stdio's answer must not sit in a
/// buffer while the client waits on it.
fn emit(stdout: &std::io::Stdout, response: &Value) {
    let mut out = stdout.lock();
    let _ = writeln!(out, "{response}");
    let _ = out.flush();
}

/// One newline-delimited frame read from the stdio transport.
#[derive(Debug, PartialEq, Eq)]
enum Frame {
    /// Input is exhausted — EOF, or a read error the transport treats as
    /// end-of-stream (a broken pipe leaves nothing to answer).
    Eof,
    /// A complete line, or the final unterminated tail, whose content is
    /// at or under the cap. Any trailing newline is included; the caller
    /// trims it.
    Line(Vec<u8>),
    /// A line longer than the cap. Its remaining bytes have already been
    /// drained here to the next boundary, so the next [`read_frame`] call
    /// resumes on a whole message — but this line's own id (if any) is
    /// gone, so only a null-id error can answer it.
    TooLong,
}

/// Reads one frame, holding at most `max_line_bytes` + 1 bytes in memory.
///
/// The one probe byte past the cap is what lets a line whose content is
/// EXACTLY the cap be returned whole — with its newline, or at EOF —
/// rather than mistaken for an over-long one. Only a read that fills that
/// extra byte with no terminating newline is [`Frame::TooLong`], and its
/// remainder is drained here so the next call resumes on a message
/// boundary. An unterminated tail at or under the cap is just the last
/// line before EOF, and comes back as [`Frame::Line`].
fn read_frame(reader: &mut impl BufRead, max_line_bytes: u64) -> Frame {
    // saturating_add keeps a cap of u64::MAX from wrapping the probe to 0
    // (which read_until would treat as "take nothing" and loop forever).
    let probe = max_line_bytes.saturating_add(1);
    let mut raw = Vec::new();
    match reader.take(probe).read_until(b'\n', &mut raw) {
        Ok(0) => return Frame::Eof,
        Ok(_) => {}
        Err(_) => return Frame::Eof,
    }
    if raw.last() != Some(&b'\n') && raw.len() as u64 > max_line_bytes {
        // Over the cap with no newline in the probe window: drain the rest
        // of this line so the next read lands on a message boundary.
        loop {
            let mut sink = Vec::new();
            match reader.take(probe).read_until(b'\n', &mut sink) {
                Ok(0) => break,
                Ok(_) if sink.last() == Some(&b'\n') => break,
                Ok(_) => continue,
                Err(_) => break,
            }
        }
        return Frame::TooLong;
    }
    Frame::Line(raw)
}

/// The reply to a JSON-RPC batch: batching left the MCP spec in
/// 2025-06, so it is refused plainly rather than answered half a
/// contract. Shared by `main`'s loop (which must check this itself —
/// it classifies ahead of `dispatch` to route `tools/call` to its own
/// thread) and by `handle` below, for the tests.
fn batch_rejected() -> Value {
    mcp::error_response(
        Value::Null,
        -32600,
        "batch messages are not part of MCP; send one message per line".to_string(),
    )
}

/// Runs one `tools/call` to completion over the bridge.
///
/// `retrieve` composes a variable number of tool calls from earlier
/// ones' results, so it has no single (method, path, body) for
/// `route_tool` to hand back — `run_retrieve` issues them itself, each
/// via this same synchronous `bridge.call`.
///
/// The network-facing transport (`remote_mcp`) caps this composed
/// whole against `max_result_bytes`, since an untrusted caller could
/// aim it at a server it does not run. Here the composition is left
/// uncapped on purpose: the bridge is a LOCAL proxy between the
/// operator's own MCP client and their own server, so each dispatched
/// call rides a plain REST request rather than the
/// `TAGURU_MCP_MAX_RESULT_BYTES`-capped `/mcp` route — nothing
/// server-side bounds the response but the operator's own handler, and
/// the result crosses stdio to the operator's own process — a size
/// they asked for, not a budget an adversary can spend against a third
/// party.
fn dispatch_tool(bridge: &Bridge, name: &str, arguments: &Value) -> Result<String, String> {
    if name == "retrieve" {
        mcp::run_retrieve(arguments, |method, path, body| {
            bridge.call(method, &path, body)
        })
        .map(|value| value.to_string())
    } else {
        mcp::route_tool(name, arguments)
            .and_then(|(method, path, body)| bridge.call(method, &path, body))
    }
}

/// Dispatches one already-classified, non-batch message. Notifications
/// get no reply (correct JSON-RPC — nothing is waiting); everything
/// else that cannot be dispatched gets a JSON-RPC error, exactly like
/// the HTTP transport (`remote_mcp`) — a client that sent an id must
/// never hang on silence.
///
/// `main`'s loop never routes a `Call::Tool` here — that classifies
/// ahead of this call and hands it to its own thread instead (see the
/// loop's `match`) — but `handle` below still can, for the tests.
fn dispatch(bridge: &Bridge, instructions: &str, classified: mcp::Message) -> Option<Value> {
    let (id, call) = match classified {
        mcp::Message::Notification => return None,
        mcp::Message::Undecodable { id } => {
            return Some(mcp::error_response(
                id,
                -32600,
                "not a JSON-RPC message (no method)".to_string(),
            ));
        }
        mcp::Message::Request { id, call } => (id, call),
    };
    Some(match call {
        mcp::Call::Initialize { protocol_version } => mcp::response(
            id,
            mcp::initialize_result(protocol_version.as_deref(), instructions),
        ),
        mcp::Call::Ping => mcp::response(id, serde_json::json!({})),
        mcp::Call::ToolsList => mcp::response(id, mcp::tools_result()),
        mcp::Call::Tool { name, arguments } => mcp::response(
            id,
            mcp::tool_response(dispatch_tool(bridge, &name, &arguments)),
        ),
        mcp::Call::Unknown { method } => {
            mcp::error_response(id, -32601, format!("unknown method '{method}'"))
        }
    })
}

/// Classifies and dispatches one JSON-RPC message fully synchronously
/// — used directly by the tests below. `main`'s loop does not call
/// this: it needs to see the classified message itself first, to route
/// a `tools/call` to its own thread instead (see `dispatch`'s doc).
#[cfg(test)]
fn handle(bridge: &Bridge, instructions: &str, message: &Value) -> Option<Value> {
    if message.is_array() {
        return Some(batch_rejected());
    }
    dispatch(bridge, instructions, mcp::classify(message))
}

/// The bridge's ceiling on simultaneously in-flight `tools/call`
/// dispatches, from a raw `TAGURU_MCP_MAX_CONCURRENT_TOOLS` reading. A
/// literal 0 parses fine but would deadlock every tool call waiting on
/// a permit that never opens; that — and anything unparseable or
/// unset — falls back to the default rather than bricking the bridge.
fn resolve_max_concurrent_tools(raw: Option<String>) -> usize {
    raw.and_then(|value| value.parse::<usize>().ok())
        .filter(|&permits| permits > 0)
        .unwrap_or(8)
}

/// A counting semaphore bounding how many `tools/call` dispatches hold
/// a live HTTP request at once (`std` has no built-in one) — the
/// textbook `Mutex` + `Condvar` shape, non-poisoning `parking_lot`
/// versions so one dispatch thread panicking cannot wedge every other
/// thread's `.lock()` behind it.
struct Semaphore {
    permits: Mutex<usize>,
    freed: Condvar,
}

impl Semaphore {
    fn new(permits: usize) -> Self {
        Self {
            permits: Mutex::new(permits),
            freed: Condvar::new(),
        }
    }

    /// Blocks the CALLING thread (not the stdio loop, since every
    /// caller here is already its own spawned thread) until a permit
    /// is free, then holds it until the returned guard drops — which
    /// happens even if the dispatch thread panics, same as an ordinary
    /// return.
    fn acquire(self: &Arc<Self>) -> Permit {
        let mut permits = self.permits.lock();
        while *permits == 0 {
            self.freed.wait(&mut permits);
        }
        *permits -= 1;
        drop(permits);
        Permit(Arc::clone(self))
    }
}

struct Permit(Arc<Semaphore>);

impl Drop for Permit {
    fn drop(&mut self) {
        *self.0.permits.lock() += 1;
        self.0.freed.notify_one();
    }
}

/// The bridge's global request timeout in seconds, from a raw
/// `TAGURU_MCP_TIMEOUT_SECS` reading. A literal 0 parses fine but would
/// arm a zero-second timeout that aborts every request before it can
/// answer; that — and anything unparseable or unset — falls back to the
/// 75-second default rather than bricking the bridge.
fn resolve_timeout_secs(raw: Option<String>) -> u64 {
    raw.and_then(|value| value.parse::<u64>().ok())
        .filter(|&secs| secs > 0)
        .unwrap_or(75)
}

/// The bridge's HTTP client: 4xx/5xx come back as responses, not
/// errors, so their JSON error bodies stay readable for `call`.
fn bridge_agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .http_status_as_error(false)
        .build()
        .into()
}

struct Bridge {
    base: String,
    /// Sent as `Authorization: Bearer` on every request when set —
    /// same env var name as the server (`TAGURU_API_TOKEN`), one
    /// concept to configure on both ends of a deployment.
    token: Option<String>,
    agent: ureq::Agent,
}

impl Bridge {
    /// One HTTP round trip; the API's JSON error body becomes the Err
    /// text so the agent reads the server's own explanation.
    fn call(&self, method: &str, path: &str, body: Option<Value>) -> Result<String, String> {
        let mut request = ureq::http::Request::builder()
            .method(method)
            .uri(format!("{}{path}", self.base));
        if let Some(token) = &self.token {
            request = request.header("Authorization", format!("Bearer {token}"));
        }
        // Both arms run to completion inside the match: a bodiless GET
        // and a JSON POST are differently typed requests in ureq 3.
        let response = match body {
            // A string argument (the `import` tool's NDJSON stream) rides
            // as raw text — `Value::to_string()` would JSON-quote it,
            // escaping every newline and breaking the line-oriented parse
            // on the other end.
            Some(Value::String(text)) => request
                .header("Content-Type", "application/x-ndjson; charset=utf-8")
                .body(text)
                .map(|request| self.agent.run(request)),
            Some(body) => request
                .header("Content-Type", "application/json")
                .body(body.to_string())
                .map(|request| self.agent.run(request)),
            None => request.body(()).map(|request| self.agent.run(request)),
        };
        let mut response = response
            .map_err(|error| format!("request assembly failed: {error}"))?
            .map_err(|error| format!("server unreachable at {}: {error}", self.base))?;
        let code = response.status().as_u16();
        if code < 400 {
            response
                .body_mut()
                .read_to_string()
                .map_err(|error| format!("response unreadable: {error}"))
        } else {
            let detail = response.body_mut().read_to_string().unwrap_or_default();
            Err(format!("HTTP {code}: {detail}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No network is touched: every input below is answered (or
    /// silenced) before dispatch would reach the bridge.
    fn bridge() -> Bridge {
        Bridge {
            base: "http://127.0.0.1:9".to_string(),
            token: None,
            agent: bridge_agent(Duration::from_secs(1)),
        }
    }

    #[test]
    fn undispatchable_messages_get_an_error_reply_not_silence() {
        // A message with an id but no method: its sender waits on that
        // id, so silence hangs it forever. The HTTP transport answers
        // -32600 (mcp::classify calls it Undecodable); stdio must too.
        let reply = handle(
            &bridge(),
            "",
            &serde_json::json!({"jsonrpc": "2.0", "id": 1}),
        )
        .expect("an undecodable message must be answered");
        assert_eq!(reply["error"]["code"], -32600, "{reply}");

        // A batch — even one wrapping a single well-formed request — is
        // refused with an error on both transports, never dropped.
        let reply = handle(
            &bridge(),
            "",
            &serde_json::json!([{"jsonrpc": "2.0", "id": 1, "method": "ping"}]),
        )
        .expect("a batch must be answered");
        assert_eq!(reply["error"]["code"], -32600, "{reply}");

        // Notifications stay silent — correct JSON-RPC: nothing waits.
        assert!(
            handle(
                &bridge(),
                "",
                &serde_json::json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            )
            .is_none()
        );
    }

    #[test]
    fn a_zero_or_unparseable_timeout_falls_back_to_the_default() {
        // A positive override is honored verbatim.
        assert_eq!(resolve_timeout_secs(Some("120".to_string())), 120);
        // Unset keeps the default.
        assert_eq!(resolve_timeout_secs(None), 75);
        // A literal 0 would arm a zero-second timeout that aborts every
        // request — it must not pass through as the budget.
        assert_eq!(resolve_timeout_secs(Some("0".to_string())), 75);
        // Garbage and negatives (u64 parse fails) also fall back.
        assert_eq!(resolve_timeout_secs(Some("-5".to_string())), 75);
        assert_eq!(resolve_timeout_secs(Some("soon".to_string())), 75);
        assert_eq!(resolve_timeout_secs(Some(String::new())), 75);
    }

    #[test]
    fn a_line_exactly_at_the_cap_is_framed_whole() {
        // The one probe byte past the cap is what separates "content is
        // exactly the cap" from "content is over it": an 8-byte line under
        // an 8-byte cap must come back whole, newline included, not be
        // mistaken for an over-long frame — and the reader must be left on
        // the next message, not consumed past it.
        let mut input = std::io::Cursor::new(b"AAAAAAAA\nnext\n".to_vec());
        assert_eq!(
            read_frame(&mut input, 8),
            Frame::Line(b"AAAAAAAA\n".to_vec())
        );
        assert_eq!(read_frame(&mut input, 8), Frame::Line(b"next\n".to_vec()));
        assert_eq!(read_frame(&mut input, 8), Frame::Eof);
    }

    #[test]
    fn an_unterminated_tail_at_the_cap_is_still_a_line() {
        // The last line before EOF carries no newline; at or under the cap
        // it is a line, not an over-long frame.
        let mut input = std::io::Cursor::new(b"AAAAAAAA".to_vec());
        assert_eq!(read_frame(&mut input, 8), Frame::Line(b"AAAAAAAA".to_vec()));
        assert_eq!(read_frame(&mut input, 8), Frame::Eof);
    }

    #[test]
    fn a_line_past_the_cap_is_refused_and_the_next_survives() {
        // One byte over the cap with no newline is over-long; its bytes are
        // drained so the message after it still frames cleanly.
        let mut input = std::io::Cursor::new(b"AAAAAAAAA\nhi\n".to_vec());
        assert_eq!(read_frame(&mut input, 8), Frame::TooLong);
        assert_eq!(read_frame(&mut input, 8), Frame::Line(b"hi\n".to_vec()));
        assert_eq!(read_frame(&mut input, 8), Frame::Eof);
    }

    #[test]
    fn draining_an_over_long_line_spans_multiple_probe_windows() {
        // The overflow is longer than one probe window, so the drain loop
        // must iterate; the following message must still be recovered.
        let mut input = std::io::Cursor::new(b"AAAAAAAAAAAA\nhi\n".to_vec());
        assert_eq!(read_frame(&mut input, 4), Frame::TooLong);
        assert_eq!(read_frame(&mut input, 4), Frame::Line(b"hi\n".to_vec()));
        assert_eq!(read_frame(&mut input, 4), Frame::Eof);
    }

    #[test]
    fn empty_input_is_eof() {
        let mut input = std::io::Cursor::new(Vec::new());
        assert_eq!(read_frame(&mut input, 8), Frame::Eof);
    }
}
