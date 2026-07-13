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

use std::io::{BufRead, Read, Write};
use std::time::Duration;

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
    let timeout_secs = std::env::var("TAGURU_MCP_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(75);
    let bridge = Bridge {
        base: base.trim_end_matches('/').to_string(),
        token,
        agent: bridge_agent(Duration::from_secs(timeout_secs)),
    };

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
    // an undecodable one is. 16 MiB clears any real message — the server
    // caps request bodies well below it — while bounding memory.
    const MAX_LINE_BYTES: u64 = 16 * 1024 * 1024;
    let mut raw = Vec::new();
    loop {
        raw.clear();
        match (&mut reader)
            .take(MAX_LINE_BYTES)
            .read_until(b'\n', &mut raw)
        {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(_) => break,
        }
        // A full cap's worth of bytes with no terminating newline is an
        // over-long line: drain its remainder so the next read lands on a
        // message boundary, then skip it. (A shorter unterminated tail is
        // just the final line before EOF — parse it.)
        if raw.last() != Some(&b'\n') && raw.len() as u64 == MAX_LINE_BYTES {
            loop {
                let mut sink = Vec::new();
                match (&mut reader)
                    .take(MAX_LINE_BYTES)
                    .read_until(b'\n', &mut sink)
                {
                    Ok(0) => break,
                    Ok(_) if sink.last() == Some(&b'\n') => break,
                    Ok(_) => continue,
                    Err(_) => break,
                }
            }
            // The line may have carried an id a client now waits on;
            // its bytes are gone, so a null-id parse error is all this
            // transport can say — but silence would hang that client.
            eprintln!("taguru-mcp: refusing over-long line");
            emit(
                &stdout,
                &mcp::error_response(
                    Value::Null,
                    -32700,
                    format!("line exceeds the {MAX_LINE_BYTES}-byte frame cap"),
                ),
            );
            continue;
        }
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
        if let Some(response) = handle(&bridge, &instructions, &message) {
            emit(&stdout, &response);
        }
    }
}

/// One response line out, flushed — stdio's answer must not sit in a
/// buffer while the client waits on it.
fn emit(stdout: &std::io::Stdout, response: &Value) {
    let mut out = stdout.lock();
    let _ = writeln!(out, "{response}");
    let _ = out.flush();
}

/// Dispatches one JSON-RPC message. Notifications get no reply (correct
/// JSON-RPC — nothing is waiting); everything else that cannot be
/// dispatched gets a JSON-RPC error, exactly like the HTTP transport
/// (`remote_mcp`) — a client that sent an id must never hang on
/// silence.
fn handle(bridge: &Bridge, instructions: &str, message: &Value) -> Option<Value> {
    // JSON-RPC batching left the MCP spec in 2025-06; refuse it plainly
    // rather than answering half a contract.
    if message.is_array() {
        return Some(mcp::error_response(
            Value::Null,
            -32600,
            "batch messages are not part of MCP; send one message per line".to_string(),
        ));
    }
    let (id, call) = match mcp::classify(message) {
        mcp::Message::Notification => return None,
        mcp::Message::Undecodable => {
            return Some(mcp::error_response(
                Value::Null,
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
        mcp::Call::Tool { name, arguments } => {
            let outcome = mcp::route_tool(&name, &arguments)
                .and_then(|(method, path, body)| bridge.call(method, &path, body));
            mcp::response(id, mcp::tool_response(outcome))
        }
        mcp::Call::Unknown { method } => {
            mcp::error_response(id, -32601, format!("unknown method '{method}'"))
        }
    })
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
}
