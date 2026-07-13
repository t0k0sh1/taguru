//! POST /mcp: the MCP Streamable HTTP transport, stateless profile.
//!
//! Taguru's MCP surface holds no per-session state — `initialize`
//! hands out the manual, and every tool call stands alone — so this
//! endpoint answers each POSTed JSON-RPC message with plain
//! `application/json`: no SSE stream, no `Mcp-Session-Id`, nothing to
//! resume. (The spec explicitly allows this profile; clients that
//! prefer SSE fall back to JSON responses.) GET lands on the router's
//! method_not_allowed fallback like any other wrong verb.
//!
//! Tool calls are dispatched IN PROCESS onto the same routes the HTTP
//! API serves, via a `Router` handle captured before the middleware
//! stack is applied. `/mcp` itself sits behind bearer auth, the
//! timeout, the body cap, and the metrics like every route — so the
//! dispatched inner request deliberately skips re-authentication and
//! re-counting: one client request, one budget, one log line.

use std::error::Error as _;
use std::sync::Arc;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::http::{Request, StatusCode, header};
use axum::response::{IntoResponse, Response};
use http_body_util::LengthLimitError;
use serde_json::{Value, json};
use taguru::deadline::Deadline;
use tower::ServiceExt;

use crate::mcp;

/// Answers one Streamable HTTP message. `dispatch` is the router the
/// tool calls run against — free of every layer except authorization;
/// `instructions` is the manual exactly as GET /protocol serves it;
/// `key` is the OUTER request's authenticated identity, stamped onto
/// each dispatched call so a scoped key's grant holds through the MCP
/// surface exactly as over raw HTTP; `max_result_bytes` bounds how
/// much of a dispatched tool's response this transport will buffer
/// (see [`RESULT_TOO_BIG`]); `deadline` is the OUTER request's budget,
/// stamped onto the dispatched call the same way — without this, a
/// tool call would run past `enforce_timeout`'s race unchecked, since
/// the dispatched request never passes back through that layer.
pub async fn serve(
    dispatch: Router,
    instructions: Arc<String>,
    key: Option<crate::auth::AuthKey>,
    body: Bytes,
    max_result_bytes: usize,
    deadline: Deadline,
) -> Response {
    let Ok(message) = serde_json::from_slice::<Value>(&body) else {
        return rpc_over_http(
            StatusCode::BAD_REQUEST,
            mcp::error_response(Value::Null, -32700, "body is not JSON".to_string()),
        );
    };
    // JSON-RPC batching left the MCP spec in 2025-06; refuse it plainly
    // rather than answering half a contract.
    if message.is_array() {
        return rpc_over_http(
            StatusCode::BAD_REQUEST,
            mcp::error_response(
                Value::Null,
                -32600,
                "batch messages are not part of MCP; send one message per request".to_string(),
            ),
        );
    }

    let (id, call) = match mcp::classify(&message) {
        // Notifications get no body — 202 says "heard, nothing to say".
        mcp::Message::Notification => return StatusCode::ACCEPTED.into_response(),
        mcp::Message::Undecodable => {
            return rpc_over_http(
                StatusCode::BAD_REQUEST,
                mcp::error_response(
                    Value::Null,
                    -32600,
                    "not a JSON-RPC message (no method)".to_string(),
                ),
            );
        }
        mcp::Message::Request { id, call } => (id, call),
    };

    let reply = match call {
        mcp::Call::Initialize { protocol_version } => mcp::response(
            id,
            mcp::initialize_result(protocol_version.as_deref(), &instructions),
        ),
        mcp::Call::Ping => mcp::response(id, json!({})),
        mcp::Call::ToolsList => mcp::response(id, mcp::tools_result()),
        mcp::Call::Tool { name, arguments } => {
            let outcome = match mcp::route_tool(&name, &arguments) {
                Ok((method, path, body)) => {
                    call_inner(
                        dispatch,
                        method,
                        &path,
                        body,
                        key.as_ref(),
                        max_result_bytes,
                        deadline,
                    )
                    .await
                }
                Err(error) => Err(error),
            };
            mcp::response(id, mcp::tool_response(outcome))
        }
        mcp::Call::Unknown { method } => {
            mcp::error_response(id, -32601, format!("unknown method '{method}'"))
        }
    };
    rpc_over_http(StatusCode::OK, reply)
}

/// One in-process round trip against the API routes — the transport
/// twin of the stdio bridge's ureq call, down to the error text, so a
/// tool failure reads identically on both transports.
async fn call_inner(
    dispatch: Router,
    method: &str,
    path: &str,
    body: Option<Value>,
    key: Option<&crate::auth::AuthKey>,
    max_result_bytes: usize,
    deadline: Deadline,
) -> Result<String, String> {
    let builder = Request::builder().method(method).uri(path);
    let mut request = match body {
        Some(body) => builder
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string())),
        None => builder.body(Body::empty()),
    }
    // The method and body are static, but `path` carries caller-supplied
    // arguments (a context name, a subject/label/object) percent-encoded
    // into the URL. A long enough one exceeds `http::Uri`'s length limit,
    // which the builder reports here — fail this one call gracefully
    // rather than panic the task.
    .map_err(|error| format!("could not build the in-process request: {error}"))?;
    // The outer request's identity travels with the dispatched call:
    // the authorization layer on the dispatch router judges each tool
    // call by the same grant the raw API would.
    if let Some(key) = key {
        request.extensions_mut().insert(key.clone());
    }
    // Likewise the outer request's time budget: dispatched calls never
    // pass back through `enforce_timeout`, so this is the only way a
    // handler's own deadline check sees it.
    request.extensions_mut().insert(deadline);
    let response = match dispatch.oneshot(request).await {
        Ok(response) => response,
        Err(never) => match never {},
    };
    let status = response.status();
    let bytes = match axum::body::to_bytes(response.into_body(), max_result_bytes).await {
        Ok(bytes) => bytes,
        Err(error) if is_length_limit(&error) => return Err(RESULT_TOO_BIG.to_string()),
        Err(error) => return Err(format!("response unreadable: {error}")),
    };
    let text = String::from_utf8_lossy(&bytes).into_owned();
    if status.is_success() {
        Ok(text)
    } else {
        Err(format!("HTTP {}: {text}", status.as_u16()))
    }
}

/// A tool result too big to buffer whole (`export_context` on a large
/// context is the common case) — names the two uncapped escape
/// hatches instead of echoing the percent-encoded tool path (which
/// may carry a caller-supplied context or concept name) back into the
/// error text.
const RESULT_TOO_BIG: &str = "tool result exceeds the MCP response cap \
    (TAGURU_MCP_MAX_RESULT_BYTES); narrow the call (a smaller `limit` works for most tools), or \
    for a full-context export use GET /contexts/{name}/export over the raw HTTP API, or the \
    `taguru export` CLI — both are uncapped";

/// `axum::body::to_bytes` wraps a cap violation as an `axum::Error`
/// whose `source()` is always the underlying `LengthLimitError` — the
/// idiom axum's own docs use to tell "too big" apart from any other
/// body read failure.
fn is_length_limit(error: &axum::Error) -> bool {
    error
        .source()
        .is_some_and(|source| source.is::<LengthLimitError>())
}

fn rpc_over_http(status: StatusCode, reply: Value) -> Response {
    (status, axum::Json(reply)).into_response()
}

#[cfg(test)]
mod tests {
    use axum::Extension;

    use super::*;

    /// A context name long enough that its percent-encoded path exceeds
    /// `http::Uri`'s length limit must come back as a JSON-RPC tool
    /// error, never panic the task that builds the in-process request.
    #[tokio::test]
    async fn an_overlong_argument_fails_the_call_without_panicking() {
        let giant = "a".repeat(100_000);
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": { "name": "create_context", "arguments": { "name": giant } },
        });
        let response = serve(
            Router::new(),
            Arc::new(String::new()),
            None,
            Bytes::from(body.to_string()),
            usize::MAX,
            Deadline::unbounded(),
        )
        .await;
        let status = response.status().as_u16();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let reply: Value = serde_json::from_slice(&bytes).unwrap();
        // The RPC round trip itself succeeds; the tool result carries the
        // failure, marked `isError`, instead of the process aborting.
        assert_eq!(status, 200);
        assert_eq!(reply["result"]["isError"], true, "{reply}");
        assert!(
            reply["result"]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("could not build"),
            "{reply}"
        );
    }

    /// The idiom this function relies on: a capped body's error source
    /// is always `LengthLimitError`, but any other read failure (here,
    /// a plain io error standing in for a dropped connection) must not
    /// be misdiagnosed as "too big".
    #[tokio::test]
    async fn is_length_limit_tells_a_capped_body_apart_from_any_other_read_failure() {
        let over_cap = axum::body::to_bytes(Body::from(vec![0u8; 10]), 1)
            .await
            .unwrap_err();
        assert!(is_length_limit(&over_cap));

        let other_failure = axum::Error::new(std::io::Error::other("connection reset"));
        assert!(!is_length_limit(&other_failure));
    }

    /// A dispatched tool call whose response outgrows the configured
    /// cap comes back as a JSON-RPC success carrying an `isError` tool
    /// result — never a panic or a half-buffered response — and the
    /// message names both uncapped escape hatches.
    #[tokio::test]
    async fn a_tool_result_over_the_cap_is_refused_with_the_escape_hatches_named() {
        let huge = Router::new().route(
            "/contexts",
            axum::routing::get(|| async { "x".repeat(1_000_000) }),
        );
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": { "name": "list_contexts", "arguments": {} },
        });
        let response = serve(
            huge,
            Arc::new(String::new()),
            None,
            Bytes::from(body.to_string()),
            1024,
            Deadline::unbounded(),
        )
        .await;
        let status = response.status().as_u16();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let reply: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(status, 200);
        assert_eq!(reply["result"]["isError"], true, "{reply}");
        let text = reply["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("GET /contexts/{name}/export"), "{text}");
        assert!(text.contains("taguru export"), "{text}");
    }

    /// The outer request's deadline never passes back through
    /// `enforce_timeout` on the dispatched call — `call_inner` must
    /// stamp it on directly, or a downstream handler's own budget
    /// check always sees an unbounded deadline regardless of what the
    /// caller actually has left.
    #[tokio::test]
    async fn the_outer_deadline_is_stamped_onto_the_dispatched_request() {
        let echo = Router::new().route(
            "/contexts",
            axum::routing::get(|Extension(deadline): Extension<Deadline>| async move {
                deadline.expired().to_string()
            }),
        );
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": { "name": "list_contexts", "arguments": {} },
        });
        let already_expired = Deadline::after(std::time::Duration::ZERO);
        std::thread::sleep(std::time::Duration::from_millis(1));
        let response = serve(
            echo,
            Arc::new(String::new()),
            None,
            Bytes::from(body.to_string()),
            usize::MAX,
            already_expired,
        )
        .await;
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let reply: Value = serde_json::from_slice(&bytes).unwrap();
        let text = reply["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("true"), "{reply}");
    }
}
