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

use std::collections::HashMap;
use std::io::{BufRead, Read, Write};
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use parking_lot::Mutex;
use serde_json::Value;

const FALLBACK_INSTRUCTIONS: &str = include_str!("../llm-protocol.md");

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
                "taguru-mcp: GET /protocol failed ({}); serving the bundled copy — \
             is the server up at {}?",
                error.text, bridge.base
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
    // trip (an import, a recall against a large context), so it is
    // handed off to a worker pool rather than run inline — otherwise a
    // single slow call would leave this loop unable to read the next
    // line for the whole timeout, including a `ping` a client sends
    // just to check the bridge is still alive, or a cancellation for
    // that same slow call. The pool holds a FIXED number of threads,
    // never one per call: a client that pipelines calls without limit
    // queues them instead (a small heap-allocated job apiece), rather
    // than spawning an OS thread per call that a large enough pipeline
    // could use to exhaust the process's thread or memory budget long
    // before any concurrency limit actually kicked in.
    let max_concurrent_tools =
        resolve_max_concurrent_tools(std::env::var("TAGURU_MCP_MAX_CONCURRENT_TOOLS").ok());
    // Maps a job's own internal sequence number to whether it has been
    // cancelled. Keyed by sequence rather than by the client's response
    // id, because that id is caller-chosen and may be reused while an
    // earlier call under the same id is still queued or dispatching —
    // a bare `id -> bool` map would have exactly one slot for both
    // calls to fight over, so the newer one's insert would silently
    // erase the older one's already-recorded cancellation (or a stale
    // flag could be consumed by the wrong job). Each entry exists ONLY
    // while its own call is in flight (inserted right before its job is
    // queued, removed the moment a worker claims a cancellation or
    // finishes normally).
    let tracked_calls: Arc<Mutex<HashMap<u64, bool>>> = Arc::new(Mutex::new(HashMap::new()));
    // Resolves a client response id to the sequence number of whichever
    // job is CURRENTLY tracked under it — the most recently queued call
    // sharing that id, since that is the only one a bare id-keyed
    // `notifications/cancelled` can plausibly mean. Repointed on every
    // new job for a given id, but this never touches — let alone
    // erases — any earlier job's own entry in `tracked_calls` above.
    let active_by_id: Arc<Mutex<HashMap<String, u64>>> = Arc::new(Mutex::new(HashMap::new()));
    let mut next_seq: u64 = 0;
    let (job_tx, job_rx) = mpsc::channel::<ToolJob>();
    // `mpsc::Receiver` isn't `Sync`; the mutex is what lets every
    // worker share the one queue instead of each needing its own.
    let job_rx = Arc::new(Mutex::new(job_rx));
    // Spawned once, up front — never one thread per call. Each worker
    // loops pulling its next job off the shared queue, so the pool
    // size doubles as the dispatch concurrency ceiling; no separate
    // semaphore is needed. `job_tx` stays alive until EOF, so nothing
    // stands between `main` returning and a `tools/call` still queued
    // or mid-flight — dropping it there closes the queue and lets
    // every worker drain the rest before this loop joins them.
    let workers: Vec<thread::JoinHandle<()>> = (0..max_concurrent_tools)
        .map(|_| {
            let bridge = Arc::clone(&bridge);
            let tracked_calls = Arc::clone(&tracked_calls);
            let active_by_id = Arc::clone(&active_by_id);
            let job_rx = Arc::clone(&job_rx);
            thread::spawn(move || run_tool_worker(&bridge, &tracked_calls, &active_by_id, &job_rx))
        })
        .collect();
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
            cancel(&tracked_calls, &active_by_id, &target_id.to_string());
            continue;
        }
        match mcp::classify(&message) {
            mcp::Message::Request {
                id,
                call: mcp::Call::Tool { name, arguments },
            } => {
                let response_id = id.to_string();
                next_seq += 1;
                let seq = next_seq;
                // Registered before the job is queued, not inside a
                // worker, so a cancellation racing the handoff always
                // finds this job already tracked instead of landing in
                // the gap and being dropped.
                tracked_calls.lock().insert(seq, false);
                // Repoints this id at the new job. An earlier call
                // sharing the same id — still queued or dispatching —
                // keeps its own entry in `tracked_calls` untouched, so
                // this reuse can never erase a cancellation already
                // recorded against it.
                active_by_id.lock().insert(response_id.clone(), seq);
                // No receiver ever drops before `job_tx` does (every
                // worker holds its own clone of `job_rx` until this
                // loop drops the sender below), so this send cannot
                // fail during normal operation.
                let _ = job_tx.send(ToolJob {
                    id,
                    response_id,
                    seq,
                    name,
                    arguments,
                });
            }
            classified => {
                if let Some(response) = dispatch(&bridge, &instructions, classified) {
                    emit(&stdout, &response);
                }
            }
        }
    }
    // Stdin is exhausted, but queued or in-flight calls may still be
    // outstanding. Dropping the sender closes the queue, so every
    // worker drains what's left and returns instead of blocking on
    // `recv` forever; only then does joining let each one finish
    // answering (or lose a race to a late cancellation) before the
    // process actually exits.
    drop(job_tx);
    for worker in workers {
        let _ = worker.join();
    }
}

/// One queued `tools/call` dispatch, handed from the stdio loop to
/// whichever pool worker picks it up next.
struct ToolJob {
    id: Value,
    response_id: String,
    /// This job's own slot in `tracked_calls` — never shared with any
    /// other job, even one queued for the same `response_id`.
    seq: u64,
    name: String,
    arguments: Value,
}

/// Runs on one of the pool's fixed worker threads: pulls jobs off the
/// shared queue until it closes (the stdio loop hit EOF and dropped
/// its sender), dispatching each in turn.
fn run_tool_worker(
    bridge: &Bridge,
    tracked_calls: &Mutex<HashMap<u64, bool>>,
    active_by_id: &Mutex<HashMap<String, u64>>,
    job_rx: &Mutex<mpsc::Receiver<ToolJob>>,
) {
    loop {
        let Ok(job) = job_rx.lock().recv() else {
            return;
        };
        // A cancel that landed while this call was still queued skips
        // it entirely: no request sent for a reply nothing is waiting
        // on.
        if take_cancelled(tracked_calls, job.seq) {
            clear_active(active_by_id, &job.response_id, job.seq);
            continue;
        }
        let outcome = dispatch_tool(bridge, &job.name, &job.arguments);
        // ureq has no mid-flight abort, so a cancel that landed once
        // the call was already running can't stop the request — but
        // it still stops the now-unwanted reply from reaching the
        // client.
        if take_cancelled(tracked_calls, job.seq) {
            clear_active(active_by_id, &job.response_id, job.seq);
            continue;
        }
        // Finished clean: drop the tracking entry so a LATER message
        // that reuses this id (ids are caller-chosen, not guaranteed
        // unique across a session) starts untracked instead of
        // inheriting stale state.
        tracked_calls.lock().remove(&job.seq);
        clear_active(active_by_id, &job.response_id, job.seq);
        emit(
            &std::io::stdout(),
            &mcp::response(job.id, mcp::tool_response(outcome)),
        );
    }
}

/// Flags whichever job is CURRENTLY tracked under `target_id` as
/// cancelled — the most recently queued `tools/call` sharing that
/// client id, since that is the only one a bare id-keyed cancellation
/// can plausibly mean. An id with nothing tracked (already answered, or
/// never existed) has nothing to flag and is silently a no-op instead
/// of poisoning a future call that reuses the same id.
fn cancel(
    tracked_calls: &Mutex<HashMap<u64, bool>>,
    active_by_id: &Mutex<HashMap<String, u64>>,
    target_id: &str,
) {
    let seq = active_by_id.lock().get(target_id).copied();
    if let Some(seq) = seq
        && let Some(flag) = tracked_calls.lock().get_mut(&seq)
    {
        *flag = true;
    }
}

/// Clears `response_id`'s pointer into `tracked_calls` — but only if it
/// still points at `seq`. A later call that reused the same client id
/// has since repointed it at its own, newer entry; clearing
/// unconditionally would erase that job's bookkeeping instead of this
/// finished one's.
fn clear_active(active_by_id: &Mutex<HashMap<String, u64>>, response_id: &str, seq: u64) {
    let mut active_by_id = active_by_id.lock();
    if active_by_id.get(response_id) == Some(&seq) {
        active_by_id.remove(response_id);
    }
}

/// Clears `seq`'s tracking entry and reports whether it had been
/// cancelled — but ONLY when it had. An entry that is merely pending
/// (`false`) is left in place so a cancellation arriving later can
/// still find and flag it.
fn take_cancelled(tracked_calls: &Mutex<HashMap<u64, bool>>, seq: u64) -> bool {
    let mut tracked_calls = tracked_calls.lock();
    if tracked_calls.get(&seq) == Some(&true) {
        tracked_calls.remove(&seq);
        true
    } else {
        false
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
fn dispatch_tool(bridge: &Bridge, name: &str, arguments: &Value) -> Result<String, mcp::ToolError> {
    if name == "retrieve" {
        mcp::run_retrieve(arguments, |method, path, body| {
            bridge.call(method, &path, body).map_err(|error| error.text)
        })
        .map(|value| value.to_string())
        .map_err(mcp::ToolError::from)
    } else {
        mcp::route_tool(name, arguments)
            .map_err(mcp::ToolError::from)
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
/// ahead of this call and queues it for the tool worker pool instead
/// (see the loop's `match`) — but `handle` below still can, for the
/// tests.
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
        mcp::Message::InvalidId => {
            return Some(mcp::error_response(
                Value::Null,
                -32600,
                "id must be a string, a number, or null".to_string(),
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
/// this: it needs to see the classified message itself first, to queue
/// a `tools/call` for the tool worker pool instead (see `dispatch`'s
/// doc).
#[cfg(test)]
fn handle(bridge: &Bridge, instructions: &str, message: &Value) -> Option<Value> {
    if message.is_array() {
        return Some(batch_rejected());
    }
    dispatch(bridge, instructions, mcp::classify(message))
}

/// The bridge's ceiling on simultaneously in-flight `tools/call`
/// dispatches — the tool worker pool's size — from a raw
/// `TAGURU_MCP_MAX_CONCURRENT_TOOLS` reading. A literal 0 parses fine
/// but would leave every tool call queued behind an empty pool
/// forever; that — and anything unparseable or unset — falls back to
/// the default rather than bricking the bridge.
fn resolve_max_concurrent_tools(raw: Option<String>) -> usize {
    raw.and_then(|value| value.parse::<usize>().ok())
        .filter(|&permits| permits > 0)
        .unwrap_or(8)
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
    /// text so the agent reads the server's own explanation. When that
    /// body parses as a JSON object, it also rides again as
    /// [`mcp::ToolError::structured`] (issue #182) — additive
    /// alongside the prose, never replacing it.
    fn call(
        &self,
        method: &str,
        path: &str,
        body: Option<Value>,
    ) -> Result<String, mcp::ToolError> {
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
            .map_err(|error| format!("request assembly failed: {error}"))
            .map_err(mcp::ToolError::from)?
            .map_err(|error| format!("server unreachable at {}: {error}", self.base))
            .map_err(mcp::ToolError::from)?;
        let code = response.status().as_u16();
        if code < 400 {
            response
                .body_mut()
                .read_to_string()
                .map_err(|error| mcp::ToolError::from(format!("response unreadable: {error}")))
        } else {
            let detail = response.body_mut().read_to_string().unwrap_or_default();
            let structured = serde_json::from_str::<Value>(&detail)
                .ok()
                .filter(Value::is_object);
            Err(mcp::ToolError {
                text: format!("HTTP {code}: {detail}"),
                structured,
            })
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

        // An id of a disallowed JSON-RPC type (object/array/bool, not
        // string/number/null) is refused too, with a null id per the
        // spec's own rule for a reply whose id could not be established
        // — never echoed back as-is.
        let reply = handle(
            &bridge(),
            "",
            &serde_json::json!({"jsonrpc": "2.0", "id": [1], "method": "ping"}),
        )
        .expect("an invalid-typed id must be answered");
        assert_eq!(reply["error"]["code"], -32600, "{reply}");
        assert_eq!(reply["id"], serde_json::Value::Null, "{reply}");

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

    #[test]
    fn take_cancelled_ignores_a_seq_that_is_not_tracked() {
        // A cancellation naming a job that already answered (or never
        // existed) must not be able to poison anything: no entry means
        // no-op, not an insertion that a later call could inherit.
        let tracked_calls = Mutex::new(HashMap::new());
        assert!(!take_cancelled(&tracked_calls, 404));
        assert!(tracked_calls.lock().is_empty());
    }

    #[test]
    fn take_cancelled_leaves_a_merely_pending_entry_in_place() {
        // A call that is in flight but not yet cancelled must stay
        // tracked — a cancellation arriving later still needs to find it.
        let tracked_calls = Mutex::new(HashMap::from([(1u64, false)]));
        assert!(!take_cancelled(&tracked_calls, 1));
        assert_eq!(tracked_calls.lock().get(&1), Some(&false));
    }

    #[test]
    fn take_cancelled_claims_and_clears_a_cancelled_entry() {
        // Once cancelled, the entry is consumed — a stale duplicate
        // delivery of the same cancellation (or a later id reuse) finds
        // nothing left to act on.
        let tracked_calls = Mutex::new(HashMap::from([(1u64, true)]));
        assert!(take_cancelled(&tracked_calls, 1));
        assert!(tracked_calls.lock().is_empty());
    }

    #[test]
    fn cancel_ignores_an_id_with_no_active_job() {
        // A `notifications/cancelled` naming an id that already
        // answered (or never existed) has nothing in `active_by_id` to
        // resolve, so it must be a plain no-op.
        let tracked_calls = Mutex::new(HashMap::new());
        let active_by_id = Mutex::new(HashMap::new());
        cancel(&tracked_calls, &active_by_id, "unknown");
        assert!(tracked_calls.lock().is_empty());
    }

    #[test]
    fn cancel_flags_whichever_job_is_currently_active_for_that_id() {
        let tracked_calls = Mutex::new(HashMap::from([(1u64, false)]));
        let active_by_id = Mutex::new(HashMap::from([("1".to_string(), 1u64)]));
        cancel(&tracked_calls, &active_by_id, "1");
        assert_eq!(tracked_calls.lock().get(&1), Some(&true));
    }

    #[test]
    fn a_cancellation_for_an_in_flight_call_survives_the_client_reusing_its_id() {
        // The bug this guards against: call A (seq 1) is dispatching
        // under client id "1" when it is cancelled. Before A's worker
        // gets a chance to observe that, the client reuses id "1" for a
        // brand-new call B (seq 2) — legitimate under MCP's
        // fire-and-forget cancellation, which gives the client no
        // acknowledgment to wait for before reusing the id. A blind
        // `id -> bool` map would have one slot for both calls, so B's
        // insert would silently overwrite A's already-recorded
        // cancellation with `false`. Tracking by sequence number
        // instead must let A's cancellation survive B's insert.
        let tracked_calls = Mutex::new(HashMap::new());
        let active_by_id = Mutex::new(HashMap::new());

        // A (seq 1) is registered and dispatching under id "1".
        tracked_calls.lock().insert(1, false);
        active_by_id.lock().insert("1".to_string(), 1);

        // The client cancels id "1", meaning A — the only job tracked
        // under it so far.
        cancel(&tracked_calls, &active_by_id, "1");

        // Before A's worker checks, the client reuses id "1" for B
        // (seq 2).
        tracked_calls.lock().insert(2, false);
        active_by_id.lock().insert("1".to_string(), 2);

        // A's worker checks its OWN sequence number: the reuse must not
        // have erased A's cancellation.
        assert!(
            take_cancelled(&tracked_calls, 1),
            "A's cancellation must survive id \"1\" being reused by B before A observed it"
        );
        // B was never cancelled — its own check must come back clean.
        assert!(!take_cancelled(&tracked_calls, 2));
    }

    #[test]
    fn clear_active_leaves_a_newer_jobs_pointer_alone() {
        // A (seq 1) finishes after B (seq 2) has already reused id "1"
        // — its own cleanup must not rip out B's still-live pointer.
        let active_by_id = Mutex::new(HashMap::from([("1".to_string(), 2u64)]));
        clear_active(&active_by_id, "1", 1);
        assert_eq!(active_by_id.lock().get("1"), Some(&2));
    }

    #[test]
    fn clear_active_removes_its_own_still_current_pointer() {
        let active_by_id = Mutex::new(HashMap::from([("1".to_string(), 1u64)]));
        clear_active(&active_by_id, "1", 1);
        assert!(active_by_id.lock().is_empty());
    }

    #[test]
    fn one_worker_drains_a_backlog_far_larger_than_the_pool_itself() {
        // The pool's whole point: a `tools/call` becomes a small
        // heap-allocated job on a shared queue, not an OS thread. Proven
        // here by running a backlog of 1,000 through a SINGLE worker
        // invocation — on this test's own thread, no `thread::spawn` at
        // all — which could only drain them all if jobs are cheap queue
        // entries rather than one-thread-per-call.
        let (job_tx, job_rx) = mpsc::channel::<ToolJob>();
        let job_rx = Mutex::new(job_rx);
        let tracked_calls: Mutex<HashMap<u64, bool>> = Mutex::new(HashMap::new());
        let active_by_id: Mutex<HashMap<String, u64>> = Mutex::new(HashMap::new());

        let jobs = 1000;
        for seq in 0..jobs {
            let response_id = seq.to_string();
            tracked_calls.lock().insert(seq, false);
            active_by_id.lock().insert(response_id.clone(), seq);
            // An unknown tool name is refused by `route_tool` before it
            // would ever reach the network — this test proves the
            // queue drains, not that a real HTTP round trip completes.
            job_tx
                .send(ToolJob {
                    id: serde_json::json!(seq),
                    response_id,
                    seq,
                    name: "not-a-real-tool".to_string(),
                    arguments: serde_json::json!({}),
                })
                .unwrap();
        }
        drop(job_tx);

        run_tool_worker(&bridge(), &tracked_calls, &active_by_id, &job_rx);

        assert!(
            tracked_calls.lock().is_empty(),
            "every queued job must be dispatched (and its tracking entry cleared) by the one worker"
        );
        assert!(
            active_by_id.lock().is_empty(),
            "every queued job must clear its own still-current active_by_id pointer"
        );
    }
}
