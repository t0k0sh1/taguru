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
use axum::http::{HeaderMap, Request, StatusCode, header};
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
#[allow(clippy::too_many_arguments)] // the outer request's context, spread flat
pub async fn serve(
    dispatch: Router,
    instructions: Arc<String>,
    key: Option<crate::auth::AuthKey>,
    scope: Option<crate::auth::KeyScope>,
    headers: HeaderMap,
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

    // 2026-07-28's per-request contract, applied before any dispatch: a
    // request that declares the stateless era (via `_meta` or the
    // mirrored header) must carry consistent, supported metadata — and
    // one that declares neither runs under the legacy `initialize`
    // contract untouched, exactly the dual-era split the spec draws.
    let era = match modern_gate(&headers, &message, &id) {
        Ok(era) => era,
        Err(refusal) => return rpc_over_http(StatusCode::BAD_REQUEST, refusal),
    };

    let reply = match call {
        mcp::Call::Initialize { protocol_version } => mcp::response(
            id,
            mcp::initialize_result(protocol_version.as_deref(), &instructions),
        ),
        mcp::Call::Ping => mcp::response(id, json!({})),
        // Mandatory under 2026-07-28, and answered on the legacy era
        // too: a dual-era client's first message may be exactly this
        // probe, sent to learn which era the server speaks.
        mcp::Call::Discover => mcp::response(id, mcp::discover_result(&instructions)),
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
            let outcome: Result<String, mcp::ToolError> = if deadline.expired() {
                // Refused before `run_retrieve_bounded` is even
                // reached — but it is still one logical retrieval, and
                // a trace showing nothing at all for a budget-exhausted
                // call is exactly the one a tail sampler most needs to
                // keep (ADR 0008 §5).
                let span = mcp::root_span(mcp::Transport::RemoteMcp);
                span.record("otel.status_code", "ERROR");
                span.record("taguru.error.kind", "deadline_exceeded");
                let _span = span.entered();
                Err(DEADLINE_EXCEEDED.to_string().into())
            } else {
                tokio::task::block_in_place(|| {
                    let handle = tokio::runtime::Handle::current();
                    mcp::run_retrieve_bounded(
                        &arguments,
                        Some(max_result_bytes),
                        mcp::Transport::RemoteMcp,
                        |method, path, body| {
                            use tracing::Instrument as _;
                            handle
                                .block_on(
                                    call_inner(
                                        dispatch.clone(),
                                        method,
                                        &path,
                                        body,
                                        key.as_ref(),
                                        scope.as_ref(),
                                        max_result_bytes,
                                        deadline,
                                    )
                                    // Captured at construction, not
                                    // re-read at each poll: the phase
                                    // span this dispatch belongs to
                                    // stays unambiguous even if
                                    // `call_inner` ever grows a
                                    // `spawn` (ADR 0008 §10).
                                    .instrument(tracing::Span::current()),
                                )
                                .map_err(|error| error.text)
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
                // The transport-owned size the `taguru.retrieve` span
                // itself never records (ADR 0008 §3): the composed
                // JSON's true byte length is only known here, after
                // `to_string()` and this transport's own cap. Recorded
                // on the ambient `POST /mcp` request span (declared
                // `Empty` in `trace::traced_request`) — `taguru.retrieve`
                // has already closed by the time this runs, the same
                // reason the stdio bridge records its own copy of this
                // field on `taguru.tool_call` rather than `taguru.retrieve`.
                .inspect(|text| {
                    tracing::Span::current().record("taguru.result.bytes", text.len());
                })
                .map_err(mcp::ToolError::from)
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
                Err(error) => Err(mcp::ToolError::from(error)),
            };
            mcp::response(id, mcp::tool_response(outcome))
        }
        mcp::Call::Unknown { method } => {
            // 2026-07-28 pins the HTTP status for an unimplemented RPC
            // to 404 — the JSON-RPC body is what tells it apart from a
            // legacy server that does not host /mcp at all. The legacy
            // era keeps its 200: those clients read only the body, and
            // some treat a non-2xx envelope as transport failure.
            let status = match era {
                Era::Modern => StatusCode::NOT_FOUND,
                Era::Legacy => StatusCode::OK,
            };
            return rpc_over_http(
                status,
                mcp::error_response(id, -32601, format!("unknown method '{method}'")),
            );
        }
    };
    rpc_over_http(StatusCode::OK, reply)
}

/// Which wire contract one request runs under.
enum Era {
    /// The `initialize`-handshake revisions (2025-11-25 and earlier):
    /// no required headers, no per-request `_meta` version.
    Legacy,
    /// Revision 2026-07-28 and later: version and identity travel on
    /// every request, mirrored into headers for intermediaries.
    Modern,
}

/// Sorts one request into its [`Era`] and enforces every header rule
/// the modern era imposes: `MCP-Protocol-Version` agreeing with the
/// body's `_meta`, the version actually being implemented, `Mcp-Method`
/// agreeing with `method`, and `Mcp-Name` agreeing with `params.name`
/// on a `tools/call`. Any failure is the JSON-RPC error body to send
/// back — the spec's `HeaderMismatch` (-32020) or
/// `UnsupportedProtocolVersion` (-32022), each under HTTP 400, which
/// the caller stamps on.
///
/// A non-modern version in `MCP-Protocol-Version` alone selects
/// [`Era::Legacy`]: clients have mirrored that header since 2025-06-18,
/// so its mere presence proves nothing about the era — and it may name
/// a legacy revision this build never listed (2025-11-25 was skipped
/// entirely), which deserves the same tolerance `initialize` extends
/// to an unrecognized proposed version, not a refusal.
fn modern_gate(headers: &HeaderMap, message: &Value, id: &Value) -> Result<Era, Value> {
    // "Present but unreadable" must not collapse into "absent": a
    // malformed value is exactly what the spec's server validation
    // names for -32020, and folding it to None would let it ride the
    // legacy path unexamined.
    let header_version = match headers.get("mcp-protocol-version") {
        None => None,
        Some(value) => match value.to_str() {
            Ok(text) => Some(text),
            Err(_) => {
                return Err(header_mismatch(
                    id,
                    "MCP-Protocol-Version header is not readable as ASCII".to_string(),
                ));
            }
        },
    };
    let meta_version = mcp::request_protocol_version(message);
    let version = match (meta_version, header_version) {
        (None, Some(header)) if mcp::MODERN_PROTOCOL_VERSIONS.contains(&header) => {
            return Err(header_mismatch(
                id,
                format!(
                    "MCP-Protocol-Version is '{header}' but the body carries no \
                     _meta {}",
                    mcp::META_PROTOCOL_VERSION
                ),
            ));
        }
        // No `_meta` and no modern header: the legacy contract, whatever
        // (if anything) the header names — see the doc comment above.
        (None, _) => return Ok(Era::Legacy),
        (Some(meta), Some(header)) if meta != header => {
            return Err(header_mismatch(
                id,
                format!(
                    "MCP-Protocol-Version header '{header}' does not match the body's '{meta}'"
                ),
            ));
        }
        (Some(meta), None) => {
            return Err(header_mismatch(
                id,
                format!(
                    "the body declares protocol version '{meta}' but the required \
                     MCP-Protocol-Version header is missing"
                ),
            ));
        }
        (Some(meta), Some(_)) => meta,
    };
    if !mcp::MODERN_PROTOCOL_VERSIONS.contains(&version) {
        return Err(unsupported_version(id, version));
    }
    let body_method = message.get("method").and_then(Value::as_str).unwrap_or("");
    match headers
        .get("mcp-method")
        .and_then(|value| value.to_str().ok())
    {
        Some(method) if method == body_method => {}
        Some(method) => {
            return Err(header_mismatch(
                id,
                format!("Mcp-Method header '{method}' does not match body method '{body_method}'"),
            ));
        }
        None => {
            return Err(header_mismatch(
                id,
                "the required Mcp-Method header is missing".to_string(),
            ));
        }
    }
    if body_method == "tools/call" {
        let body_name = message
            .get("params")
            .and_then(|params| params.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("");
        match headers
            .get("mcp-name")
            .and_then(|value| value.to_str().ok())
            .map(decode_sentinel)
        {
            Some(Ok(name)) if name == body_name => {}
            Some(Ok(name)) => {
                return Err(header_mismatch(
                    id,
                    format!("Mcp-Name header '{name}' does not match body name '{body_name}'"),
                ));
            }
            Some(Err(())) => {
                return Err(header_mismatch(
                    id,
                    "Mcp-Name header carries an undecodable base64 sentinel".to_string(),
                ));
            }
            None => {
                return Err(header_mismatch(
                    id,
                    "the required Mcp-Name header is missing".to_string(),
                ));
            }
        }
    }
    Ok(Era::Modern)
}

fn header_mismatch(id: &Value, message: String) -> Value {
    mcp::error_response(id.clone(), mcp::HEADER_MISMATCH, message)
}

fn unsupported_version(id: &Value, requested: &str) -> Value {
    mcp::unsupported_version_error(id.clone(), requested)
}

/// A header value, through the spec's Base64 sentinel if it wears one:
/// `=?base64?…?=` marks a value that could not ride as plain ASCII
/// (servers MUST decode before comparing against the body — a client
/// is free to encode even a plain-ASCII name). Anything unmarked is
/// itself the value.
fn decode_sentinel(value: &str) -> Result<String, ()> {
    let Some(encoded) = value
        .strip_prefix("=?base64?")
        .and_then(|rest| rest.strip_suffix("?="))
    else {
        return Ok(value.to_string());
    };
    base64_decode(encoded).and_then(|bytes| String::from_utf8(bytes).map_err(|_| ()))
}

/// RFC 4648 §4 decoding (the standard `+`/`/` alphabet the sentinel
/// carries), padding optional — decode-only, the mirror of oauth.rs's
/// encode-only `base64url`, and just as deliberately dependency-free.
fn base64_decode(text: &str) -> Result<Vec<u8>, ()> {
    fn sextet(byte: u8) -> Result<u32, ()> {
        match byte {
            b'A'..=b'Z' => Ok(u32::from(byte - b'A')),
            b'a'..=b'z' => Ok(u32::from(byte - b'a') + 26),
            b'0'..=b'9' => Ok(u32::from(byte - b'0') + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(()),
        }
    }
    let stripped = text.trim_end_matches('=').as_bytes();
    let mut out = Vec::with_capacity(stripped.len() * 3 / 4);
    for chunk in stripped.chunks(4) {
        let mut acc: u32 = 0;
        for &byte in chunk {
            acc = (acc << 6) | sextet(byte)?;
        }
        match chunk.len() {
            4 => out.extend_from_slice(&[(acc >> 16) as u8, (acc >> 8) as u8, acc as u8]),
            3 => out.extend_from_slice(&[(acc >> 10) as u8, (acc >> 2) as u8]),
            2 => out.push((acc >> 4) as u8),
            // A lone trailing sextet encodes fewer bits than a byte —
            // no valid base64 ends this way.
            _ => return Err(()),
        }
    }
    Ok(out)
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
) -> Result<String, mcp::ToolError> {
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
    .map_err(|error| format!("{}: {error}", mcp::REQUEST_BUILD_FAILED_PREFIX))
    .map_err(mcp::ToolError::from)?;
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
            return Err(RESULT_TOO_BIG.to_string().into());
        }
        Err(error) if is_length_limit(&error) => {
            return Err(format!(
                "HTTP {}: (error body exceeds the MCP response cap, \
                 TAGURU_MCP_MAX_RESULT_BYTES)",
                status.as_u16()
            )
            .into());
        }
        Err(error) => return Err(format!("response unreadable: {error}").into()),
    };
    let text = String::from_utf8_lossy(&bytes).into_owned();
    if status.is_success() {
        Ok(text)
    } else {
        // The downstream JSON error body (issue #182's structured
        // `issues`/`integrity`/`retryable_after_correction` included,
        // when the failure is one that carries them) rides again as
        // `structuredContent`, alongside the same prose this transport
        // has always put in `content[0].text` — additive, so a client
        // that only reads `content` sees no change.
        let structured = serde_json::from_str::<Value>(&text)
            .ok()
            .filter(Value::is_object);
        Err(mcp::ToolError {
            text: format!("HTTP {}: {text}", status.as_u16()),
            structured,
        })
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
            HeaderMap::new(),
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
            HeaderMap::new(),
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
            HeaderMap::new(),
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
            HeaderMap::new(),
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
            HeaderMap::new(),
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
            HeaderMap::new(),
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
            HeaderMap::new(),
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
            HeaderMap::new(),
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

    /// One serve() round trip with explicit headers, against a router
    /// that answers `GET /contexts` — enough for a `list_contexts`
    /// tools/call or any dispatch-free method.
    async fn roundtrip(headers: HeaderMap, body: Value) -> (u16, Value) {
        let router = Router::new().route("/contexts", axum::routing::get(|| async { "[]" }));
        let response = serve(
            router,
            Arc::new("manual".to_string()),
            None,
            None,
            headers,
            Bytes::from(body.to_string()),
            usize::MAX,
            Deadline::unbounded(),
        )
        .await;
        let status = response.status().as_u16();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    /// The standard 2026-07-28 request headers, body `_meta` half in
    /// [`modern_body`].
    fn modern_headers(method: &str, name: Option<&str>) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert("mcp-protocol-version", "2026-07-28".parse().unwrap());
        headers.insert("mcp-method", method.parse().unwrap());
        if let Some(name) = name {
            headers.insert("mcp-name", name.parse().unwrap());
        }
        headers
    }

    fn modern_body(method: &str, mut params: Value) -> Value {
        params.as_object_mut().unwrap().insert(
            "_meta".to_string(),
            json!({ "io.modelcontextprotocol/protocolVersion": "2026-07-28" }),
        );
        json!({ "jsonrpc": "2.0", "id": 1, "method": method, "params": params })
    }

    /// The modern happy path end to end: matching headers and `_meta`,
    /// a `tools/call` naming its tool in `Mcp-Name`, a 200 whose result
    /// wears the 2026-07-28 envelope.
    #[tokio::test]
    async fn a_conforming_modern_tools_call_runs_and_answers_complete() {
        let (status, reply) = roundtrip(
            modern_headers("tools/call", Some("list_contexts")),
            modern_body(
                "tools/call",
                json!({ "name": "list_contexts", "arguments": {} }),
            ),
        )
        .await;
        assert_eq!(status, 200, "{reply}");
        assert_eq!(reply["result"]["resultType"], "complete", "{reply}");
        assert_ne!(reply["result"]["isError"], json!(true), "{reply}");
    }

    /// `server/discover` answers on both eras — with modern headers,
    /// and bare (the backward-compatibility probe of a client that has
    /// not yet learned which era this server speaks).
    #[tokio::test]
    async fn server_discover_answers_on_either_era() {
        let (status, reply) = roundtrip(
            HeaderMap::new(),
            json!({ "jsonrpc": "2.0", "id": 1, "method": "server/discover" }),
        )
        .await;
        assert_eq!(status, 200);
        assert_eq!(
            reply["result"]["supportedVersions"][0], "2026-07-28",
            "{reply}"
        );
        assert_eq!(reply["result"]["instructions"], "manual", "{reply}");

        let (status, reply) = roundtrip(
            modern_headers("server/discover", None),
            modern_body("server/discover", json!({})),
        )
        .await;
        assert_eq!(status, 200, "{reply}");
        assert_eq!(reply["result"]["resultType"], "complete", "{reply}");
    }

    /// Requests carrying neither modern `_meta` nor any version header
    /// — and ones carrying the header legacy clients have mirrored
    /// since 2025-06-18 — run under the initialize contract untouched.
    #[tokio::test]
    async fn legacy_requests_are_untouched_by_the_modern_gate() {
        let (status, reply) = roundtrip(
            HeaderMap::new(),
            json!({ "jsonrpc": "2.0", "id": 1, "method": "ping" }),
        )
        .await;
        assert_eq!(status, 200);
        assert!(reply["error"].is_null(), "{reply}");

        let mut legacy_header = HeaderMap::new();
        legacy_header.insert("mcp-protocol-version", "2025-06-18".parse().unwrap());
        let (status, reply) = roundtrip(
            legacy_header,
            json!({ "jsonrpc": "2.0", "id": 1, "method": "ping" }),
        )
        .await;
        assert_eq!(status, 200);
        assert!(reply["error"].is_null(), "{reply}");

        // A legacy revision this build never listed (2025-11-25 was
        // skipped entirely) still selects the legacy contract — before
        // the modern gate existed this header was ignored outright, and
        // growing a 400 here would break every client speaking it.
        let mut unlisted_header = HeaderMap::new();
        unlisted_header.insert("mcp-protocol-version", "2025-11-25".parse().unwrap());
        let (status, reply) = roundtrip(
            unlisted_header,
            json!({ "jsonrpc": "2.0", "id": 1, "method": "ping" }),
        )
        .await;
        assert_eq!(status, 200, "{reply}");
        assert!(reply["error"].is_null(), "{reply}");
    }

    /// "Present but unreadable" is not "absent": a version header that
    /// cannot even be read as ASCII is malformed per the spec's server
    /// validation — refused with -32020, never quietly ridden into the
    /// legacy path as if no header had been sent.
    #[tokio::test]
    async fn an_unreadable_version_header_is_refused_not_treated_as_absent() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "mcp-protocol-version",
            axum::http::HeaderValue::from_bytes(&[0xFF, 0xFE]).unwrap(),
        );
        let (status, reply) = roundtrip(
            headers,
            json!({ "jsonrpc": "2.0", "id": 1, "method": "ping" }),
        )
        .await;
        assert_eq!(status, 400, "{reply}");
        assert_eq!(reply["error"]["code"], -32020, "{reply}");
    }

    /// Every shape of header/body disagreement the spec's server
    /// validation names is a 400 with `HeaderMismatch` (-32020): a
    /// version header contradicting `_meta`, a declared version whose
    /// required header is missing, a missing `Mcp-Method`, and an
    /// `Mcp-Name` that does not name the called tool.
    #[tokio::test]
    async fn header_body_disagreements_are_refused_with_header_mismatch() {
        let mut contradicting = modern_headers("tools/list", None);
        contradicting.insert("mcp-protocol-version", "2025-11-25".parse().unwrap());
        let disagreements = [
            (contradicting, modern_body("tools/list", json!({}))),
            (
                HeaderMap::new(),
                modern_body("tools/list", json!({})), // _meta but no headers at all
            ),
            (
                {
                    let mut headers = modern_headers("tools/list", None);
                    headers.remove("mcp-method");
                    headers
                },
                modern_body("tools/list", json!({})),
            ),
            (
                // Present but disagreeing with the body's method: a
                // different arm from the missing-header case above.
                modern_headers("tools/call", Some("list_contexts")),
                modern_body(
                    "tools/list",
                    json!({ "name": "list_contexts", "arguments": {} }),
                ),
            ),
            (
                modern_headers("tools/call", Some("delete_context")),
                modern_body(
                    "tools/call",
                    json!({ "name": "list_contexts", "arguments": {} }),
                ),
            ),
            (
                modern_headers("tools/call", None), // Mcp-Name required for tools/call
                modern_body(
                    "tools/call",
                    json!({ "name": "list_contexts", "arguments": {} }),
                ),
            ),
        ];
        for (headers, body) in disagreements {
            let (status, reply) = roundtrip(headers, body.clone()).await;
            assert_eq!(status, 400, "{body} → {reply}");
            assert_eq!(reply["error"]["code"], -32020, "{body} → {reply}");
        }
    }

    /// A version this build does not implement, DECLARED in `_meta`,
    /// earns 400 with `UnsupportedProtocolVersionError` and the list to
    /// retry from. The header alone triggers no such refusal: without
    /// `_meta` the request is a legacy one whatever the header names
    /// (see `legacy_requests_are_untouched_by_the_modern_gate`) — only
    /// a client that actually speaks the stateless era, proven by the
    /// `_meta` field, can be asked to retry from `data.supported`.
    #[tokio::test]
    async fn an_unimplemented_version_is_refused_with_the_supported_list() {
        let mut headers = HeaderMap::new();
        headers.insert("mcp-protocol-version", "2099-01-01".parse().unwrap());
        headers.insert("mcp-method", "tools/list".parse().unwrap());
        let mut body = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {} });
        body["params"]["_meta"] =
            json!({ "io.modelcontextprotocol/protocolVersion": "2099-01-01" });
        let (status, reply) = roundtrip(headers, body).await;
        assert_eq!(status, 400);
        assert_eq!(reply["error"]["code"], -32022, "{reply}");
        assert_eq!(
            reply["error"]["data"]["supported"][0], "2026-07-28",
            "{reply}"
        );
        assert_eq!(reply["error"]["data"]["requested"], "2099-01-01", "{reply}");

        let mut header_only = HeaderMap::new();
        header_only.insert("mcp-protocol-version", "2099-01-01".parse().unwrap());
        let (status, reply) = roundtrip(
            header_only,
            json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" }),
        )
        .await;
        assert_eq!(status, 200, "{reply}");
        assert!(reply["error"].is_null(), "{reply}");
    }

    /// 2026-07-28 pins an unimplemented RPC to HTTP 404 (the JSON-RPC
    /// body tells it apart from a server with no /mcp at all); the
    /// legacy era keeps its 200 envelope for the same -32601.
    #[tokio::test]
    async fn an_unknown_method_is_404_modern_but_200_legacy() {
        let (status, reply) = roundtrip(
            modern_headers("subscriptions/listen", None),
            modern_body("subscriptions/listen", json!({})),
        )
        .await;
        assert_eq!(status, 404, "{reply}");
        assert_eq!(reply["error"]["code"], -32601, "{reply}");

        let (status, reply) = roundtrip(
            HeaderMap::new(),
            json!({ "jsonrpc": "2.0", "id": 1, "method": "subscriptions/listen" }),
        )
        .await;
        assert_eq!(status, 200, "{reply}");
        assert_eq!(reply["error"]["code"], -32601, "{reply}");
    }

    /// An `Mcp-Name` wearing the Base64 sentinel must be decoded before
    /// comparison — a client is free to encode even a plain-ASCII name.
    #[tokio::test]
    async fn an_mcp_name_in_base64_sentinel_form_is_decoded_before_comparing() {
        // "list_contexts", RFC 4648 standard alphabet with padding.
        let (status, reply) = roundtrip(
            modern_headers("tools/call", Some("=?base64?bGlzdF9jb250ZXh0cw==?=")),
            modern_body(
                "tools/call",
                json!({ "name": "list_contexts", "arguments": {} }),
            ),
        )
        .await;
        assert_eq!(status, 200, "{reply}");
        assert_ne!(reply["result"]["isError"], json!(true), "{reply}");
    }

    /// The sentinel decoder against the spec's own examples, plus the
    /// shapes it must refuse.
    #[test]
    fn decode_sentinel_matches_the_spec_examples() {
        assert_eq!(decode_sentinel("us-west1"), Ok("us-west1".to_string()));
        assert_eq!(
            decode_sentinel("=?base64?SGVsbG8sIOS4lueVjA==?="),
            Ok("Hello, 世界".to_string())
        );
        assert_eq!(
            decode_sentinel("=?base64?IHBhZGRlZCA=?="),
            Ok(" padded ".to_string())
        );
        assert_eq!(
            decode_sentinel("=?base64?bGluZTEKbGluZTI=?="),
            Ok("line1\nline2".to_string())
        );
        // A sentinel wrapping a doubly-encoded literal decodes to the
        // literal itself.
        assert_eq!(
            decode_sentinel("=?base64?PT9iYXNlNjQ/bGl0ZXJhbD89?="),
            Ok("=?base64?literal?=".to_string())
        );
        // Not base64 inside the sentinel, and a length no valid base64
        // has: refused, not passed through as if literal.
        assert_eq!(decode_sentinel("=?base64?!!!?="), Err(()));
        assert_eq!(decode_sentinel("=?base64?A?="), Err(()));
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
            HeaderMap::new(),
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
