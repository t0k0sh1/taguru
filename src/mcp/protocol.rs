use serde_json::{Value, json};

use super::schema::tool_definitions;

/// The protocol versions this build has actually been written against,
/// oldest first. `initialize` only echoes a client-named version drawn
/// from this list; anything else falls back rather than promising
/// semantics this server does not implement — the whole point of the
/// version exchange is the two sides agreeing on one wire contract,
/// which a blind echo would skip entirely.
pub(super) const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2024-11-05", "2025-03-26", "2025-06-18"];

/// Spoken when the client does not name a protocol version itself, or
/// names one this build does not recognize. The spec requires the
/// LATEST version the server supports here, not just any supported
/// one — so this is derived from the list above rather than pinned to
/// a literal that could drift from it as versions are added.
pub const FALLBACK_PROTOCOL_VERSION: &str =
    SUPPORTED_PROTOCOL_VERSIONS[SUPPORTED_PROTOCOL_VERSIONS.len() - 1];

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
}

/// The requests both transports answer.
pub enum Call {
    Initialize { protocol_version: Option<String> },
    Ping,
    ToolsList,
    Tool { name: String, arguments: Value },
    Unknown { method: String },
}

/// Sorts one decoded message into [`Message`]. Never fails: an id of
/// the wrong type is [`Message::InvalidId`], and any other garbage is
/// [`Message::Undecodable`] — both transports answer both with a
/// JSON-RPC error, the latter carrying whatever id was found so a
/// message missing only its method is still owed a correlatable reply,
/// not a null one.
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
/// message here instead.
#[allow(dead_code)] // consumed by the stdio bridge; the HTTP transport has no per-connection state to cancel against
pub fn cancelled_request_id(message: &Value) -> Option<Value> {
    if message.get("method").and_then(Value::as_str) != Some("notifications/cancelled") {
        return None;
    }
    message.get("params")?.get("requestId").cloned()
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

/// The `tools/list` result.
pub fn tools_result() -> Value {
    json!({ "tools": tool_definitions() })
}

/// Wraps a tool outcome the MCP way: errors travel as content with
/// `isError`, so the AGENT reads the server's explanation — a JSON-RPC
/// error would abort the loop instead of informing it.
pub fn tool_response(outcome: Result<String, String>) -> Value {
    match outcome {
        Ok(text) => json!({ "content": [{ "type": "text", "text": text }] }),
        Err(text) => json!({
            "content": [{ "type": "text", "text": text }],
            "isError": true,
        }),
    }
}

pub fn response(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

pub fn error_response(id: Value, code: i64, message: String) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}
