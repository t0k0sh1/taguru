use serde_json::{Value, json};

use super::schema::tool_definitions;

/// The protocol versions this build has actually been written against,
/// oldest first. `initialize` only echoes a client-named version drawn
/// from this list; anything else falls back rather than promising
/// semantics this server does not implement — the whole point of the
/// version exchange is the two sides agreeing on one wire contract,
/// which a blind echo would skip entirely.
pub const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2024-11-05", "2025-03-26", "2025-06-18"];

/// Spoken when the client does not name a protocol version itself, or
/// names one this build does not recognize. The spec requires the
/// LATEST version the server supports here, not just any supported
/// one — so this is derived from the list above rather than pinned to
/// a literal that could drift from it as versions are added.
pub const FALLBACK_PROTOCOL_VERSION: &str =
    SUPPORTED_PROTOCOL_VERSIONS[SUPPORTED_PROTOCOL_VERSIONS.len() - 1];

/// The versions a client may declare per request in `params._meta`
/// (revision 2026-07-28's stateless replacement for `initialize`).
/// Kept apart from [`SUPPORTED_PROTOCOL_VERSIONS`] on purpose: these
/// versions have no `initialize` at all, so folding them into that
/// list would let `initialize` echo — and thereby promise — a wire
/// contract under which `initialize` does not exist.
pub const MODERN_PROTOCOL_VERSIONS: &[&str] = &["2026-07-28"];

/// The `_meta` key a modern request's protocol version travels under.
pub const META_PROTOCOL_VERSION: &str = "io.modelcontextprotocol/protocolVersion";

/// `UnsupportedProtocolVersion` (2026-07-28): the request named a
/// per-request protocol version this build does not implement. From
/// the `-32020`..`-32099` sub-range the spec reserves for itself.
pub const UNSUPPORTED_PROTOCOL_VERSION: i64 = -32022;

/// `HeaderMismatch` (2026-07-28): a required Streamable HTTP header is
/// missing, malformed, or disagrees with the request body.
#[allow(dead_code)] // consumed by the HTTP transport; stdio has no headers to mismatch
pub const HEADER_MISMATCH: i64 = -32020;

/// How long a client may cache the (compile-time static) tool list and
/// discovery result before re-asking — the `ttlMs` half of 2026-07-28's
/// `CacheableResult`. An hour: the list only changes with a redeploy.
const CACHE_TTL_MS: u64 = 3_600_000;

/// The `cacheScope` half: `/mcp` sits behind bearer auth, so nothing
/// is gained by letting shared intermediaries store its responses —
/// "private" confines caching to the client itself.
const CACHE_SCOPE: &str = "private";

/// One decoded JSON-RPC message, sorted by what it obliges us to do.
pub enum Message {
    /// Carries an id: the sender expects exactly one response.
    Request { id: Value, call: Call },
    /// A method without an id (or a null one): fire-and-forget.
    Notification,
    /// No method at all — not a JSON-RPC call we can act on. Carries
    /// whatever id was present so the sender's wait isn't answered with
    /// a null it can't correlate to its request.
    Undecodable { id: Value },
    /// `id` was present but not a string, number, or null — the only
    /// shapes JSON-RPC 2.0 allows (§4). Carries no id: the spec's own
    /// rule for a reply whose id could not be established is that the
    /// id "MUST be Null", so there is nothing valid left to echo.
    InvalidId,
    /// The `jsonrpc` member is missing or not exactly the string
    /// "2.0" — required on every request by JSON-RPC 2.0 (§4), so this
    /// is not a message under the contract either transport speaks.
    /// Carries whatever (already shape-valid) id was present so the
    /// refusal stays correlatable.
    WrongJsonRpcVersion { id: Value },
}

/// The requests both transports answer.
pub enum Call {
    Initialize { protocol_version: Option<String> },
    Ping,
    Discover,
    ToolsList,
    Tool { name: String, arguments: Value },
    Unknown { method: String },
}

/// Sorts one decoded message into [`Message`]. Never fails: an id of
/// the wrong type is [`Message::InvalidId`], a missing or wrong
/// `jsonrpc` declaration is [`Message::WrongJsonRpcVersion`], and any
/// other garbage is [`Message::Undecodable`] — both transports answer
/// all three with a JSON-RPC error, the latter two carrying whatever
/// id was found so a message missing only its method (or its version
/// declaration) is still owed a correlatable reply, not a null one.
pub fn classify(message: &Value) -> Message {
    let id = match message.get("id") {
        Some(id) if !id.is_null() => id.clone(),
        _ => Value::Null,
    };
    // JSON-RPC 2.0 allows only a string, a number, or null for id
    // (§4) — an object/array/bool id can't be echoed back through
    // `response`/`error_response` without the reply becoming just as
    // malformed as the request was. Checked ahead of `method`: a bad
    // id makes the request invalid no matter what else it carries.
    if !matches!(id, Value::Null | Value::String(_) | Value::Number(_)) {
        return Message::InvalidId;
    }
    // §4 makes `"jsonrpc": "2.0"` mandatory on every request. Checked
    // after the id gate (a bad id already has nothing valid to echo)
    // but before anything else: silently serving a message that never
    // declared the contract would promise 2.0 semantics to a sender
    // who never asked for them. This refuses notifications too — a
    // method with no id but the wrong `jsonrpc` is not a 2.0
    // notification, and answering beats dropping it without a trace.
    if message.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Message::WrongJsonRpcVersion { id };
    }
    let Some(method) = message.get("method").and_then(Value::as_str) else {
        return Message::Undecodable { id };
    };
    if id.is_null() {
        return Message::Notification;
    }
    let params = message.get("params").cloned().unwrap_or(Value::Null);
    let call = match method {
        "initialize" => Call::Initialize {
            protocol_version: params
                .get("protocolVersion")
                .and_then(Value::as_str)
                .map(str::to_string),
        },
        "ping" => Call::Ping,
        "server/discover" => Call::Discover,
        "tools/list" => Call::ToolsList,
        "tools/call" => Call::Tool {
            name: params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            arguments: params.get("arguments").cloned().unwrap_or(json!({})),
        },
        _ => Call::Unknown {
            method: method.to_string(),
        },
    };
    Message::Request { id, call }
}

/// The target id of a `notifications/cancelled` message, or `None` for
/// anything else — `classify` folds every notification into one bare
/// [`Message::Notification`], discarding `params`, so a transport that
/// wants to act on this one specific notification (the stdio bridge,
/// to stop waiting on a reply nothing wants anymore) reads the raw
/// message here instead. Gated on the message actually classifying as
/// a notification: one with the wrong `jsonrpc` declaration (or an id,
/// which makes it a request) is owed its `-32600`/`-32601` refusal
/// from the ordinary dispatch path, and a refused message must not
/// ALSO reach into the queue and cancel a tracked call on the way.
#[allow(dead_code)] // consumed by the stdio bridge; the HTTP transport has no per-connection state to cancel against
pub fn cancelled_request_id(message: &Value) -> Option<Value> {
    if !matches!(classify(message), Message::Notification) {
        return None;
    }
    if message.get("method").and_then(Value::as_str) != Some("notifications/cancelled") {
        return None;
    }
    message.get("params")?.get("requestId").cloned()
}

/// The W3C trace context an MCP client may attach as
/// `params._meta.traceparent` / `params._meta.tracestate` — stdio has
/// no HTTP headers of its own to carry it. Rendered as an
/// [`http::HeaderMap`] on purpose: it feeds `trace::extract_parent`
/// unchanged, so there is exactly one `traceparent` parser in the tree
/// and the stdio transport cannot disagree with the HTTP one about a
/// malformed header (ADR 0008 §10). Absent, wrong-typed, or malformed
/// is "no parent" — an ordinary MCP client that sends no `_meta` at
/// all is unaffected — never an error.
#[allow(dead_code)] // consumed by the stdio bridge; the HTTP transport extracts from real headers instead
pub fn meta_trace_headers(message: &Value) -> http::HeaderMap {
    let mut headers = http::HeaderMap::new();
    let Some(meta) = message.get("params").and_then(|params| params.get("_meta")) else {
        return headers;
    };
    for name in ["traceparent", "tracestate"] {
        if let Some(value) = meta.get(name).and_then(Value::as_str)
            && let Ok(value) = http::HeaderValue::from_str(value)
        {
            headers.insert(http::HeaderName::from_static(name), value);
        }
    }
    headers
}

/// The protocol version a modern (2026-07-28+) request declares in
/// `params._meta`, or `None` for a legacy message that has no such
/// field — the presence of this key is what sorts a request into the
/// stateless era at all.
pub fn request_protocol_version(message: &Value) -> Option<&str> {
    message
        .get("params")?
        .get("_meta")?
        .get(META_PROTOCOL_VERSION)?
        .as_str()
}

/// The `UnsupportedProtocolVersionError` reply owed when a request
/// declares a per-request protocol version this build does not
/// implement, or `None` when the message either names a supported one
/// or is not a modern request at all. Notifications get no reply even
/// here — nothing is waiting — so only a classified request earns the
/// error.
#[allow(dead_code)] // consumed by the stdio bridge; the HTTP transport folds this check into its header gate
pub fn modern_version_rejection(message: &Value) -> Option<Value> {
    let version = request_protocol_version(message)?;
    if MODERN_PROTOCOL_VERSIONS.contains(&version) {
        return None;
    }
    let Message::Request { id, .. } = classify(message) else {
        return None;
    };
    Some(unsupported_version_error(id, version))
}

/// The `UnsupportedProtocolVersionError` reply itself — `data.supported`
/// is where a modern client looks for the versions to retry with, so
/// both transports build it here and the list can never drift.
pub fn unsupported_version_error(id: Value, requested: &str) -> Value {
    error_response_with_data(
        id,
        UNSUPPORTED_PROTOCOL_VERSION,
        "unsupported protocol version".to_string(),
        json!({ "supported": MODERN_PROTOCOL_VERSIONS, "requested": requested }),
    )
}

/// This build's identity, exactly as 2026-07-28 wants it repeated in
/// each result's `_meta` — the stateless replacement for the one-shot
/// `serverInfo` that `initialize` used to hand out.
fn server_info_meta() -> Value {
    json!({
        "io.modelcontextprotocol/serverInfo": {
            "name": "taguru",
            "version": env!("CARGO_PKG_VERSION"),
        },
    })
}

/// Stamps the 2026-07-28 result envelope onto `result`: the required
/// `resultType` ("complete" — this server never returns an interim
/// `input_required`) and the SHOULD-level serverInfo `_meta`. Applied
/// unconditionally: to a legacy client both are unknown result fields,
/// which JSON-RPC obliges it to ignore, and one shape for both eras
/// beats a fork.
fn complete_result(mut result: Value) -> Value {
    let fields = result
        .as_object_mut()
        .expect("every MCP result this server builds is an object");
    fields.insert("resultType".to_string(), json!("complete"));
    fields.insert("_meta".to_string(), server_info_meta());
    result
}

/// The `server/discover` result — the one RPC 2026-07-28 makes
/// mandatory. Doubles as the stdio backward-compatibility probe, so it
/// answers on either era's framing.
pub fn discover_result(instructions: &str) -> Value {
    complete_result(json!({
        "supportedVersions": MODERN_PROTOCOL_VERSIONS,
        "capabilities": { "tools": {} },
        "instructions": instructions,
        "ttlMs": CACHE_TTL_MS,
        "cacheScope": CACHE_SCOPE,
    }))
}

/// The `initialize` result: capabilities plus the full protocol manual
/// as `instructions`, so the agent learns the discipline the moment it
/// connects.
pub fn initialize_result(client_protocol_version: Option<&str>, instructions: &str) -> Value {
    let protocol_version = client_protocol_version
        .filter(|version| SUPPORTED_PROTOCOL_VERSIONS.contains(version))
        .unwrap_or(FALLBACK_PROTOCOL_VERSION);
    json!({
        "protocolVersion": protocol_version,
        "capabilities": { "tools": {} },
        "serverInfo": { "name": "taguru", "version": env!("CARGO_PKG_VERSION") },
        "instructions": instructions,
    })
}

/// The `tools/list` result. `ttlMs`/`cacheScope` are 2026-07-28's
/// required `CacheableResult` fields; to older clients they are just
/// two more unknown fields alongside `resultType`.
pub fn tools_result() -> Value {
    complete_result(json!({
        "tools": tool_definitions(),
        "ttlMs": CACHE_TTL_MS,
        "cacheScope": CACHE_SCOPE,
    }))
}

/// One tool call's failure. `text` is the prose every transport has
/// always put in `content[0].text`; `structured`, when the failure's
/// underlying HTTP error body parsed as a JSON object (issue #182),
/// carries that same object again as `structuredContent` — a client
/// that predates the field, or never looks for it, reads exactly the
/// prose it always has, and `content`/`text` never lose the plain
/// explanation just because the failure also carries path-addressed
/// detail.
pub struct ToolError {
    pub text: String,
    pub structured: Option<Value>,
}

impl From<String> for ToolError {
    fn from(text: String) -> Self {
        Self {
            text,
            structured: None,
        }
    }
}

/// Wraps a tool outcome the MCP way: errors travel as content with
/// `isError`, so the AGENT reads the server's explanation — a JSON-RPC
/// error would abort the loop instead of informing it. An error whose
/// underlying HTTP body was a JSON object also carries
/// `structuredContent` (MCP 2025-06-18+) alongside the prose — additive
/// on the wire, so older clients that only read `content` see no
/// change.
pub fn tool_response(outcome: Result<String, ToolError>) -> Value {
    complete_result(match outcome {
        Ok(text) => json!({ "content": [{ "type": "text", "text": text }] }),
        Err(ToolError {
            text,
            structured: None,
        }) => json!({
            "content": [{ "type": "text", "text": text }],
            "isError": true,
        }),
        Err(ToolError {
            text,
            structured: Some(structured),
        }) => json!({
            "content": [{ "type": "text", "text": text }],
            "isError": true,
            "structuredContent": structured,
        }),
    })
}

pub fn response(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

pub fn error_response(id: Value, code: i64, message: String) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// [`error_response`] with the optional JSON-RPC `error.data` member.
fn error_response_with_data(id: Value, code: i64, message: String, data: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message, "data": data },
    })
}
