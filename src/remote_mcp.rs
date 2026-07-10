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

use std::sync::Arc;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::http::{Request, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use tower::ServiceExt;

use crate::mcp;

/// Answers one Streamable HTTP message. `dispatch` is the layer-free
/// router the tool calls run against; `instructions` is the manual
/// exactly as GET /protocol serves it.
pub async fn serve(dispatch: Router, instructions: Arc<String>, body: Bytes) -> Response {
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
                Ok((method, path, body)) => call_inner(dispatch, method, &path, body).await,
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
) -> Result<String, String> {
    let builder = Request::builder().method(method).uri(path);
    let request = match body {
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
    let response = match dispatch.oneshot(request).await {
        Ok(response) => response,
        Err(never) => match never {},
    };
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .map_err(|error| format!("response unreadable: {error}"))?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    if status.is_success() {
        Ok(text)
    } else {
        Err(format!("HTTP {}: {text}", status.as_u16()))
    }
}

fn rpc_over_http(status: StatusCode, reply: Value) -> Response {
    (status, axum::Json(reply)).into_response()
}

#[cfg(test)]
mod tests {
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
            Bytes::from(body.to_string()),
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
}
