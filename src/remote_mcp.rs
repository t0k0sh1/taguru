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
/// surface exactly as over raw HTTP — and `scope` is the grant the
/// bearer gate resolved alongside it, stamped the same way so every
/// dispatched call is judged by the keyring snapshot that
/// authenticated the outer request (a hot reload landing mid-batch
/// must not re-grade later tool calls, let alone elevate a removed
/// key to the unscoped default); `max_result_bytes` bounds how much
/// of a dispatched tool's response this transport will buffer (see
/// [`RESULT_TOO_BIG`]); `deadline` is the OUTER request's budget,
/// stamped onto the dispatched call the same way — without this, a
/// tool call would run past `enforce_timeout`'s race unchecked, since
/// the dispatched request never passes back through that layer.
pub async fn serve(
    dispatch: Router,
    instructions: Arc<String>,
    key: Option<crate::auth::AuthKey>,
    scope: Option<crate::auth::KeyScope>,
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
        mcp::Message::Undecodable { id } => {
            return rpc_over_http(
                StatusCode::BAD_REQUEST,
                mcp::error_response(id, -32600, "not a JSON-RPC message (no method)".to_string()),
            );
        }
        mcp::Message::InvalidId => {
            return rpc_over_http(
                StatusCode::BAD_REQUEST,
                mcp::error_response(
                    Value::Null,
                    -32600,
                    "id must be a string, a number, or null".to_string(),
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
        mcp::Call::Tool { name, arguments } if name == "retrieve" => {
            // retrieve issues a variable number of dispatched calls
            // built from earlier ones' results, so it cannot be a
            // single `route_tool` mapping — `run_retrieve` composes
            // them itself, synchronously, so the bridge into this
            // async handler runs on a blocking thread: `block_in_place`
            // frees this worker for other tasks while it's parked, and
            // each of `run_retrieve`'s callbacks drives `call_inner`
            // (an async fn) to completion via `Handle::block_on`.
            //
            // Guarded by `deadline.expired()` first, same as every other
            // `block_in_place` site (api.rs, api/sources.rs): once
            // `block_in_place` is entered, `enforce_timeout`'s race can
            // no longer preempt it (see that function's doc comment), so
            // a budget already spent must be caught before paying for
            // the parked worker and the dispatched fan-out behind it,
            // not after.
            let outcome = if deadline.expired() {
                Err(DEADLINE_EXCEEDED.to_string())
            } else {
                tokio::task::block_in_place(|| {
                    let handle = tokio::runtime::Handle::current();
                    mcp::run_retrieve_bounded(
                        &arguments,
                        Some(max_result_bytes),
                        |method, path, body| {
                            handle.block_on(call_inner(
                                dispatch.clone(),
                                method,
                                &path,
                                body,
                                key.as_ref(),
                                scope.as_ref(),
                                max_result_bytes,
                                deadline,
                            ))
                        },
                    )
                })
                .map(|value| value.to_string())
                // `run_retrieve_bounded`'s own running-total check already
                // stops it from dispatching further calls once the composed
                // result is doomed to be too big — this is the backstop for
                // what that estimate cannot see: the composed JSON's own
                // structure (keys, brackets) on top of the parts it kept.
                // Each dispatched call was bounded by `max_result_bytes`, but
                // retrieve composes many into one response — resolved cues,
                // outlines, associations, activations, citations, and passage
                // hits sum well past any single part's cap. Bound the composed
                // whole the same way `call_inner` bounds each part, with the
                // same too-big guidance, so the client's cap holds end to end.
                .and_then(|text| {
                    if text.len() > max_result_bytes {
                        Err(RESULT_TOO_BIG.to_string())
                    } else {
                        Ok(text)
                    }
                })
            };
            mcp::response(id, mcp::tool_response(outcome))
        }
        mcp::Call::Tool { name, arguments } => {
            let outcome = match mcp::route_tool(&name, &arguments) {
                Ok((method, path, body)) => {
                    call_inner(
                        dispatch,
                        method,
                        &path,
                        body,
                        key.as_ref(),
                        scope.as_ref(),
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
#[allow(clippy::too_many_arguments)] // the outer request's context, spread flat
async fn call_inner(
    dispatch: Router,
    method: &str,
    path: &str,
    body: Option<Value>,
    key: Option<&crate::auth::AuthKey>,
    scope: Option<&crate::auth::KeyScope>,
    max_result_bytes: usize,
    deadline: Deadline,
) -> Result<String, String> {
    let builder = Request::builder().method(method).uri(path);
    let mut request = match body {
        // A string argument (the `import` tool's NDJSON stream) rides
        // as raw text — `Value::to_string()` would JSON-quote it,
        // escaping every newline and breaking the line-oriented parse
        // on the other end.
        Some(Value::String(text)) => builder
            .header(header::CONTENT_TYPE, "application/x-ndjson; charset=utf-8")
            .body(Body::from(text)),
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
    // call by the same grant the raw API would. The scope goes with
    // it — the grant the bearer gate resolved from the snapshot that
    // authenticated this key, so a keyring reload landing mid-request
    // cannot re-grade the dispatched calls.
    if let Some(key) = key {
        request.extensions_mut().insert(key.clone());
    }
    if let Some(scope) = scope {
        request.extensions_mut().insert(scope.clone());
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
        // A downstream error's own body (a verbose stack trace, an HTML
        // error page) can outgrow max_result_bytes just as easily as a
        // large successful result can — but RESULT_TOO_BIG's "narrow the
        // call" advice is nonsense for a request that already failed
        // downstream. Surface the real status instead of swapping it for
        // an unrelated cap message.
        Err(error) if is_length_limit(&error) && status.is_success() => {
            return Err(RESULT_TOO_BIG.to_string());
        }
        Err(error) if is_length_limit(&error) => {
            return Err(format!(
                "HTTP {}: (error body exceeds the MCP response cap, \
                 TAGURU_MCP_MAX_RESULT_BYTES)",
                status.as_u16()
            ));
        }
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

/// Mirrors `api::deadline_exceeded`'s message: the same "budget already
/// spent" condition, surfaced as a tool error here (this transport
/// never returns a raw HTTP error status for a tool call, only content
/// with `isError`) instead of that function's `Response`.
const DEADLINE_EXCEEDED: &str = "request exceeded its budget before this operation could start; \
    narrow the query or raise TAGURU_REQUEST_TIMEOUT_SECS";

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

    /// A downstream *error* response (a 500 with a huge body — a stack
    /// trace, an HTML error page) that outgrows the cap must not read
    /// like an oversized successful result: RESULT_TOO_BIG's "narrow the
    /// call" advice describes a request that could be retried smaller,
    /// which is nonsense for one that already failed downstream. The
    /// real status must survive instead of being swapped for the cap
    /// message.
    #[tokio::test]
    async fn a_downstream_error_over_the_cap_reports_its_real_status_not_result_too_big() {
        let failing = Router::new().route(
            "/contexts",
            axum::routing::get(|| async {
                (StatusCode::INTERNAL_SERVER_ERROR, "x".repeat(1_000_000))
            }),
        );
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": { "name": "list_contexts", "arguments": {} },
        });
        let response = serve(
            failing,
            Arc::new(String::new()),
            None,
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
        assert_eq!(status, 200, "the RPC envelope itself still succeeds");
        assert_eq!(reply["result"]["isError"], true, "{reply}");
        let text = reply["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("HTTP 500"), "{text}");
        assert!(
            !text.contains("narrow the call"),
            "a downstream failure must not be relabeled as an oversized result: {text}"
        );
    }

    /// A panic inside a dispatched tool call must never surface as a
    /// bare 200 with no trace of the failure: `routes()` wires this
    /// same `CatchPanicLayer::custom(panic_response)` as `dispatch`'s
    /// own innermost layer specifically so a panic here is caught
    /// before it can unwind out of `call_inner`'s `dispatch.oneshot`
    /// (see `routes()`'s doc comment in main.rs). The RPC envelope
    /// still answers 200 — this transport never surfaces a raw HTTP
    /// error status for a tool call — but `panic_response` must still
    /// have recorded `ErrorKind::Panic`, since that counter is the
    /// only signal this failure leaves behind anywhere: the tool
    /// result names just "HTTP 500", nothing that identifies it as a
    /// panic rather than an ordinary handler-returned error.
    #[tokio::test]
    async fn a_panic_inside_a_dispatched_tool_call_still_records_the_error_metric() {
        use crate::api::panic_response;
        use crate::registry::AppState;
        use tower_http::catch_panic::CatchPanicLayer;

        async fn boom() -> &'static str {
            panic!("kaboom")
        }

        let dir = std::env::temp_dir().join(format!(
            "taguru-remote-mcp-dispatched-panic-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();

        let panicking = Router::new()
            .route("/contexts", axum::routing::get(boom))
            .layer({
                let state = state.clone();
                CatchPanicLayer::custom(move |payload| panic_response(payload, &state))
            });

        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": { "name": "list_contexts", "arguments": {} },
        });
        let response = serve(
            panicking,
            Arc::new(String::new()),
            None,
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
        assert_eq!(status, 200, "the RPC envelope itself still succeeds");
        assert_eq!(reply["result"]["isError"], true, "{reply}");
        let text = reply["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("HTTP 500"), "{text}");

        let rendered = state.metrics().render_prometheus(&state.gauge_snapshot());
        assert!(
            rendered.contains("taguru_errors_total{kind=\"panic\"} 1"),
            "a panic dispatched through /mcp must still reach taguru_errors_total, \
             the only place this failure is visible outside the log line: {rendered}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// retrieve composes many dispatched calls into one response; each is
    /// individually bounded by `max_result_bytes`, but their sum is not
    /// until `run_retrieve_bounded` tracks the running total itself. A
    /// retrieve whose pieces each fit yet together overflow the cap must
    /// come back as an `isError` result — and, since the budget is
    /// checked after every dispatched call, it must be refused partway
    /// through the ten-origin fan-out rather than after all ten resolves
    /// ran and only then having the composed whole discarded.
    ///
    /// Multi-threaded flavor: the retrieve branch parks on
    /// `block_in_place`, which panics on the default current-thread test
    /// runtime.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_composed_retrieve_over_the_cap_is_refused_like_a_single_oversized_call() {
        // Each resolve answer fits under any sane per-call cap; ten of
        // them summed into `resolved` do not.
        let router = Router::new()
            .route(
                "/contexts/ctx/resolve",
                axum::routing::post(|| async {
                    axum::Json(json!({ "result": [{ "name": "anchor", "note": "x".repeat(400) }] }))
                }),
            )
            .route(
                "/contexts/ctx/activate",
                axum::routing::post(|| async {
                    axum::Json(json!({ "result": { "matches": [] } }))
                }),
            );
        let origins: Vec<String> = (0..10).map(|i| format!("c{i}")).collect();
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "retrieve",
                "arguments": {
                    "context": "ctx",
                    "origins": origins,
                    // Trim the composition to resolve + activate: this is
                    // the aggregate-cap path, not describe/cite/search.
                    "auto_pick": false,
                    "describe_first": false,
                    "fetch_citations": false,
                },
            },
        });

        // A cap the summed result overruns: refused mid-fan-out, the
        // running total (not the post-hoc backstop below) catching it.
        let response = serve(
            router.clone(),
            Arc::new(String::new()),
            None,
            None,
            Bytes::from(body.to_string()),
            1024,
            Deadline::unbounded(),
        )
        .await;
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let reply: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(reply["result"]["isError"], true, "{reply}");
        let text = reply["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("already exceeds 1024 bytes"), "{text}");
        assert!(text.contains("fewer origins"), "{text}");

        // The identical call under a generous cap composes and returns —
        // proving the cap, not the composition, is what refused it above.
        let response = serve(
            router,
            Arc::new(String::new()),
            None,
            None,
            Bytes::from(body.to_string()),
            usize::MAX,
            Deadline::unbounded(),
        )
        .await;
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let reply: Value = serde_json::from_slice(&bytes).unwrap();
        assert_ne!(reply["result"]["isError"], json!(true), "{reply}");
        let text = reply["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("resolved"), "{text}");
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

    /// The retrieve branch must check the deadline itself before
    /// entering `block_in_place` — same as every other `block_in_place`
    /// site (api.rs, api/sources.rs) — since `enforce_timeout`'s race
    /// can no longer preempt it once inside (see that function's doc
    /// comment). Proven on the default current-thread test runtime:
    /// `block_in_place` panics there, so if this guard were missing this
    /// test would panic instead of returning the tool error below.
    #[tokio::test]
    async fn retrieve_refuses_an_already_expired_deadline_before_block_in_place() {
        let already_expired = Deadline::after(std::time::Duration::ZERO);
        std::thread::sleep(std::time::Duration::from_millis(1));
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "retrieve",
                "arguments": { "context": "ctx", "origins": ["x"] },
            },
        });
        let response = serve(
            Router::new(),
            Arc::new(String::new()),
            None,
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
        assert_eq!(reply["result"]["isError"], true, "{reply}");
        let text = reply["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("exceeded its budget"), "{text}");
    }

    /// An id of a disallowed JSON-RPC type (object/array/bool) must come
    /// back as -32600 with a null id, per the spec's own rule for a
    /// reply whose id could not be established — never echoed back
    /// as-is, which would just hand the client a second malformed
    /// message.
    #[tokio::test]
    async fn an_id_of_the_wrong_type_is_refused_with_a_null_id() {
        let body = json!({
            "jsonrpc": "2.0",
            "id": { "not": "allowed" },
            "method": "ping",
        });
        let response = serve(
            Router::new(),
            Arc::new(String::new()),
            None,
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
        assert_eq!(status, 400);
        assert_eq!(reply["id"], Value::Null, "{reply}");
        assert_eq!(reply["error"]["code"], -32600, "{reply}");
    }
}
