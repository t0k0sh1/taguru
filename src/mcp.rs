//! The MCP surface, shared by both transports: the stdio bridge
//! (`taguru-mcp`) and the server's own `POST /mcp` endpoint. Tool
//! definitions, the tool → HTTP request mapping, and the JSON-RPC
//! framing live here exactly once; a transport differs only in how a
//! routed request is executed (a ureq round trip from the bridge, an
//! in-process `Router` call on the server) and in how replies travel
//! back (stdout lines vs HTTP responses).
//!
//! Compiled into both binaries via `#[path]` — deliberately not part
//! of the library surface, which stays [`crate::context`]-only.

use serde_json::{Value, json};

/// Spoken when the client does not name a protocol version itself.
pub const FALLBACK_PROTOCOL_VERSION: &str = "2024-11-05";

/// One decoded JSON-RPC message, sorted by what it obliges us to do.
pub enum Message {
    /// Carries an id: the sender expects exactly one response.
    Request { id: Value, call: Call },
    /// A method without an id (or a null one): fire-and-forget.
    Notification,
    /// No method at all — not a JSON-RPC call we can act on.
    Undecodable,
}

/// The requests both transports answer.
pub enum Call {
    Initialize { protocol_version: Option<String> },
    Ping,
    ToolsList,
    Tool { name: String, arguments: Value },
    Unknown { method: String },
}

/// Sorts one decoded message into [`Message`]. Never fails: garbage is
/// [`Message::Undecodable`], which the stdio transport ignores and the
/// HTTP transport answers with a JSON-RPC error.
pub fn classify(message: &Value) -> Message {
    let Some(method) = message.get("method").and_then(Value::as_str) else {
        return Message::Undecodable;
    };
    let id = match message.get("id") {
        Some(id) if !id.is_null() => id.clone(),
        _ => return Message::Notification,
    };
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

/// The `initialize` result: capabilities plus the full protocol manual
/// as `instructions`, so the agent learns the discipline the moment it
/// connects.
pub fn initialize_result(client_protocol_version: Option<&str>, instructions: &str) -> Value {
    json!({
        "protocolVersion": client_protocol_version.unwrap_or(FALLBACK_PROTOCOL_VERSION),
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

/// Percent-encodes a context name for use as one URL path segment.
fn segment(name: &str) -> String {
    name.bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                (byte as char).to_string()
            }
            _ => format!("%{byte:02X}"),
        })
        .collect()
}

fn object_schema(properties: Value, required: &[&str]) -> Value {
    json!({ "type": "object", "properties": properties, "required": required })
}

/// Pulls a required string argument or explains what is missing.
fn need<'a>(arguments: &'a Value, key: &str) -> Result<&'a str, String> {
    arguments
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing required argument '{key}'"))
}

/// Copies the listed keys into a request body, skipping absent ones.
fn pick(arguments: &Value, keys: &[&str]) -> Value {
    let mut body = serde_json::Map::new();
    for &key in keys {
        if let Some(value) = arguments.get(key)
            && !value.is_null()
        {
            body.insert(key.to_string(), value.clone());
        }
    }
    Value::Object(body)
}

pub fn tool_definitions() -> Vec<Value> {
    let context = json!({ "type": "string", "description": "Context name (from list_contexts)" });
    let tools = vec![
        (
            "list_contexts",
            "Routing directory: every context's name, description, stats (counts, top concepts, label sample), and usage counters (reads/empty_reads/writes, last-used times). Pick the search/ingest target here yourself.",
            object_schema(json!({}), &[]),
        ),
        (
            "create_context",
            "Create a context. One context = one 文脈: one spelling, one referent — different things sharing a spelling get separate contexts. The description drives routing; say concretely what the context covers.",
            object_schema(
                json!({
                    "name": { "type": "string" },
                    "description": { "type": "string" },
                    "pinned": { "type": "boolean", "description": "keep resident (always-hot contexts like glossaries)" },
                    "dice_floor": { "type": "number", "description": "fuzzy-entry floor (default 0.3)" },
                    "semantic_floor": { "type": "number", "description": "semantic-entry floor (default 0.35)" }
                }),
                &["name"],
            ),
        ),
        (
            "update_context",
            "Update description / pinned / dice_floor.",
            object_schema(
                json!({
                    "name": { "type": "string" },
                    "description": { "type": "string" },
                    "pinned": { "type": "boolean" },
                    "dice_floor": { "type": "number" },
                    "semantic_floor": { "type": "number" }
                }),
                &["name"],
            ),
        ),
        (
            "delete_context",
            "Delete a context and its files (irreversible).",
            object_schema(json!({ "name": { "type": "string" } }), &["name"]),
        ),
        (
            "add_associations",
            "Write facts as a batch (one document = one call), a source id on every element. Discipline: check spellings with resolve/resolve_label and reuse before minting; don't re-assert paraphrases within one document; negation = positive label + negative weight; make implicit membership an explicit edge; weave ordered procedures with the three edges 最初の工程/次の工程/工程 (details in get_protocol).",
            object_schema(
                json!({
                    "context": context,
                    "associations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "subject": { "type": "string" },
                                "label": { "type": "string" },
                                "object": { "type": "string" },
                                "weight": { "type": "number" },
                                "source": { "type": "string" }
                            },
                            "required": ["subject", "label", "object", "weight"]
                        }
                    }
                }),
                &["context", "associations"],
            ),
        ),
        (
            "store_passages",
            "Register the original text behind each source id. Always finish an ingest with this; answers ground in originals looked up from attributions.",
            object_schema(
                json!({
                    "context": context,
                    "passages": { "type": "object", "additionalProperties": { "type": "string" } }
                }),
                &["context", "passages"],
            ),
        ),
        (
            "lookup_passages",
            "Fetch the passages behind attribution source ids — the answer-from-originals half of retrieval.",
            object_schema(
                json!({
                    "context": context,
                    "sources": { "type": "array", "items": { "type": "string" } }
                }),
                &["context", "sources"],
            ),
        ),
        (
            "list_sources",
            "Source ids with registered passages — targets for retract_source / lookup_passages, inventory for diff sync.",
            object_schema(json!({ "context": context }), &["context"]),
        ),
        (
            "resolve",
            "Resolve free wording to stored concept names (normalized entry, absorbs typos). The retrieval entry: use the canonical names it returns as origins for explore/activate. Empty → reword, or lower dice_floor (e.g. 0.2) and retry.",
            object_schema(
                json!({
                    "context": context,
                    "cue": { "type": "string" },
                    "dice_floor": { "type": "number", "description": "one-call override of the fuzzy floor" },
                    "semantic_floor": { "type": "number", "description": "one-call override of the semantic floor" }
                }),
                &["context", "cue"],
            ),
        ),
        (
            "resolve_label",
            "resolve, for relation labels. Use before writes (check before mint) and to pick query labels.",
            object_schema(
                json!({
                    "context": context,
                    "cue": { "type": "string" },
                    "dice_floor": { "type": "number" },
                    "semantic_floor": { "type": "number" }
                }),
                &["context", "cue"],
            ),
        ),
        (
            "describe",
            "A concept's outline: which labels carry how many facts, per role. Check a hub here first, then query just the labels you need — never pull a whole profile blind.",
            object_schema(
                json!({ "context": context, "concept": { "type": "string" } }),
                &["context", "concept"],
            ),
        ),
        (
            "query",
            "Position-pinned search. subject/label/object each take a string or an array (array = match any). Outline with describe, then narrow by label.",
            object_schema(
                json!({
                    "context": context,
                    "subject": { "description": "string or array" },
                    "label": { "description": "string or array" },
                    "object": { "description": "string or array" },
                    "limit": { "type": "integer" }
                }),
                &["context"],
            ),
        ),
        (
            "recall",
            "Every association touching the cue, whatever its position. Use query when the role matters.",
            object_schema(
                json!({
                    "context": context,
                    "cue": { "type": "string" },
                    "limit": { "type": "integer" }
                }),
                &["context", "cue"],
            ),
        ),
        (
            "activate",
            "Spread activation from origins, strongest first (path shows the route). The main tool for gathering related knowledge. strength orders within one call only.",
            object_schema(
                json!({
                    "context": context,
                    "origins": { "type": "array", "items": { "type": "string" } },
                    "decay": { "type": "number", "description": "default 0.5" },
                    "limit": { "type": "integer", "description": "default 20" }
                }),
                &["context", "origins"],
            ),
        ),
        (
            "explore",
            "Exhaustive structural walk with hop distances, for unranked neighborhood views. limit defaults to 100 (max 1000); truncation keeps the nearest hops (watch total).",
            object_schema(
                json!({
                    "context": context,
                    "origins": { "type": "array", "items": { "type": "string" } },
                    "max_depth": { "type": "integer" },
                    "limit": { "type": "integer" }
                }),
                &["context", "origins"],
            ),
        ),
        (
            "list_labels",
            "The full relation vocabulary (canonical only). Read it before extracting to avoid spelling forks.",
            object_schema(json!({ "context": context }), &["context"]),
        ),
        (
            "get_aliases",
            "Export registered aliases (alias → canonical).",
            object_schema(json!({ "context": context }), &["context"]),
        ),
        (
            "add_aliases",
            "Point alternate spellings at canonical names (entry-only; results always return canonicals). The fix when live wording misses. Cannot join two existing concepts — that would be a merge, which is rebuild territory.",
            object_schema(
                json!({
                    "context": context,
                    "concepts": { "type": "object", "additionalProperties": { "type": "string" }, "description": "alias → canonical" },
                    "labels": { "type": "object", "additionalProperties": { "type": "string" } }
                }),
                &["context"],
            ),
        ),
        (
            "remove_aliases",
            "Withdraw mis-registered alias spellings (exact spellings, per namespace). The spelling stops resolving and is free to re-register; canonicals and their knowledge are untouched. Canonical names are refused — removal cannot unname a record.",
            object_schema(
                json!({
                    "context": context,
                    "concepts": { "type": "array", "items": { "type": "string" }, "description": "alias spellings to withdraw" },
                    "labels": { "type": "array", "items": { "type": "string" } }
                }),
                &["context"],
            ),
        ),
        (
            "retract_source",
            "Withdraw one source's (document's) contributions from graph and passage store. Diff sync for updated documents: retract the old version, then re-ingest the new. Concepts and edges remain; only weights come down.",
            object_schema(
                json!({ "context": context, "source": { "type": "string" } }),
                &["context", "source"],
            ),
        ),
        (
            "search_passages",
            "Full-text search over registered passages (bigram BM25). The second lane, for knowledge that never fit triples (order, conditions, discourse) — look here too when graph search comes up short.",
            object_schema(
                json!({
                    "context": context,
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "description": "default 5" }
                }),
                &["context", "query"],
            ),
        ),
        (
            "refresh_embeddings",
            "After ingesting, re-embed the glosses (name + graph context) of new or changed concepts and labels (servers with embeddings only). Makes paraphrases and question-shaped cues land through resolve's semantic fallback.",
            object_schema(json!({ "context": context }), &["context"]),
        ),
        (
            "audit_vocabulary",
            "Vocabulary health check: lexical fork candidates (青嶺酒蔵/青嶺酒造) and semantic ones (創業年/設立年; needs embeddings). Candidates, not verdicts — same referent → alias onto one canonical; different → leave. Run at ingest milestones.",
            object_schema(
                json!({
                    "context": context,
                    "dice_floor": { "type": "number", "description": "lexical floor (default 0.6)" },
                    "cosine_floor": { "type": "number", "description": "semantic floor (default 0.6)" }
                }),
                &["context"],
            ),
        ),
        (
            "audit_coverage",
            "Post-ingest audit: associations unreachable from origins (the document's main entities). Non-empty = membership edges are missing — add them before finishing.",
            object_schema(
                json!({
                    "context": context,
                    "origins": { "type": "array", "items": { "type": "string" } }
                }),
                &["context", "origins"],
            ),
        ),
        (
            "get_protocol",
            "The complete manual: ingest discipline and retrieval loop.",
            object_schema(json!({}), &[]),
        ),
    ];

    tools
        .into_iter()
        .map(|(name, description, schema)| {
            json!({ "name": name, "description": description, "inputSchema": schema })
        })
        .collect()
}

/// Maps one tool call onto (method, path, body) — pure, so the mapping
/// from advertised tools to HTTP requests is testable without a server.
pub fn route_tool(
    name: &str,
    arguments: &Value,
) -> Result<(&'static str, String, Option<Value>), String> {
    let context_path = |key: &str| -> Result<String, String> {
        Ok(format!("/contexts/{}", segment(need(arguments, key)?)))
    };
    Ok(match name {
        "get_protocol" => ("GET", "/protocol".to_string(), None),
        "list_contexts" => ("GET", "/contexts".to_string(), None),
        "create_context" => (
            "PUT",
            context_path("name")?,
            Some(pick(
                arguments,
                &["description", "pinned", "dice_floor", "semantic_floor"],
            )),
        ),
        "update_context" => (
            "PATCH",
            context_path("name")?,
            Some(pick(
                arguments,
                &["description", "pinned", "dice_floor", "semantic_floor"],
            )),
        ),
        "delete_context" => ("DELETE", context_path("name")?, None),
        "add_associations" => (
            "POST",
            format!("{}/associations", context_path("context")?),
            Some(arguments.get("associations").cloned().unwrap_or(json!([]))),
        ),
        "store_passages" => (
            "POST",
            format!("{}/sources", context_path("context")?),
            Some(pick(arguments, &["passages"])),
        ),
        "lookup_passages" => (
            "POST",
            format!("{}/sources/lookup", context_path("context")?),
            Some(pick(arguments, &["sources"])),
        ),
        "list_sources" => ("GET", format!("{}/sources", context_path("context")?), None),
        "resolve" => (
            "POST",
            format!("{}/resolve", context_path("context")?),
            Some(pick(arguments, &["cue", "dice_floor", "semantic_floor"])),
        ),
        "resolve_label" => (
            "POST",
            format!("{}/resolve_label", context_path("context")?),
            Some(pick(arguments, &["cue", "dice_floor", "semantic_floor"])),
        ),
        "describe" => (
            "POST",
            format!("{}/describe", context_path("context")?),
            Some(pick(arguments, &["concept"])),
        ),
        "query" => (
            "POST",
            format!("{}/query", context_path("context")?),
            Some(pick(arguments, &["subject", "label", "object", "limit"])),
        ),
        "recall" => (
            "POST",
            format!("{}/recall", context_path("context")?),
            Some(pick(arguments, &["cue", "limit"])),
        ),
        "activate" => (
            "POST",
            format!("{}/activate", context_path("context")?),
            Some(pick(arguments, &["origins", "decay", "limit"])),
        ),
        "explore" => (
            "POST",
            format!("{}/explore", context_path("context")?),
            Some(pick(arguments, &["origins", "max_depth", "limit"])),
        ),
        "list_labels" => ("GET", format!("{}/labels", context_path("context")?), None),
        "get_aliases" => ("GET", format!("{}/aliases", context_path("context")?), None),
        "add_aliases" => (
            "POST",
            format!("{}/aliases", context_path("context")?),
            Some(pick(arguments, &["concepts", "labels"])),
        ),
        "remove_aliases" => (
            "DELETE",
            format!("{}/aliases", context_path("context")?),
            Some(pick(arguments, &["concepts", "labels"])),
        ),
        "retract_source" => (
            "POST",
            format!("{}/sources/retract", context_path("context")?),
            Some(pick(arguments, &["source"])),
        ),
        "search_passages" => (
            "POST",
            format!("{}/sources/search", context_path("context")?),
            Some(pick(arguments, &["query", "limit"])),
        ),
        "refresh_embeddings" => (
            "POST",
            format!("{}/embeddings/refresh", context_path("context")?),
            Some(json!({})),
        ),
        "audit_vocabulary" => (
            "POST",
            format!("{}/vocabulary/audit", context_path("context")?),
            Some(pick(arguments, &["dice_floor", "cosine_floor"])),
        ),
        "audit_coverage" => (
            "POST",
            format!("{}/unreachable_from", context_path("context")?),
            Some(pick(arguments, &["origins"])),
        ),
        _ => return Err(format!("unknown tool '{name}'")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wiring invariant a new tool is most likely to break: every
    /// advertised tool definition must route to an HTTP request. An
    /// argument object carrying every required key satisfies whichever
    /// subset each tool needs.
    #[test]
    fn every_advertised_tool_routes_to_a_request() {
        let arguments = json!({
            "name": "ctx", "context": "ctx", "cue": "x", "concept": "x",
            "origins": ["x"], "associations": [], "passages": {},
            "sources": ["s"], "source": "s", "query": "q",
        });
        for tool in tool_definitions() {
            let name = tool["name"].as_str().expect("definitions carry names");
            let routed = route_tool(name, &arguments);
            assert!(routed.is_ok(), "tool '{name}' does not route: {routed:?}");
            let (method, path, _) = routed.unwrap();
            assert!(
                matches!(method, "GET" | "PUT" | "PATCH" | "POST" | "DELETE"),
                "tool '{name}' uses unknown method {method}"
            );
            assert!(path.starts_with('/'), "tool '{name}' path: {path}");
        }
    }

    #[test]
    fn unknown_tools_and_missing_arguments_are_refused() {
        assert_eq!(
            route_tool("no_such_tool", &json!({})),
            Err("unknown tool 'no_such_tool'".to_string())
        );
        // A context-scoped tool without its context argument names what
        // is missing instead of building a broken path.
        assert_eq!(
            route_tool("describe", &json!({"concept": "x"})),
            Err("missing required argument 'context'".to_string())
        );
    }

    /// Context names arrive as URL path segments; anything outside the
    /// unreserved set must be percent-encoded, byte by byte.
    #[test]
    fn context_names_are_percent_encoded_into_one_segment() {
        let (_, path, _) = route_tool("list_labels", &json!({"context": "日本 語/酒"})).unwrap();
        let segment = path
            .strip_prefix("/contexts/")
            .and_then(|rest| rest.strip_suffix("/labels"))
            .expect("path shape");
        assert!(!segment.contains('/'), "slash must be encoded: {path}");
        assert!(!segment.contains(' '), "space must be encoded: {path}");
        assert_eq!(segment, "%E6%97%A5%E6%9C%AC%20%E8%AA%9E%2F%E9%85%92");
    }

    #[test]
    fn pick_copies_only_present_non_null_keys() {
        let arguments = json!({"cue": "x", "limit": null, "extra": 7});
        assert_eq!(
            pick(&arguments, &["cue", "limit", "absent"]),
            json!({"cue": "x"})
        );
    }

    /// The framing rules both transports rely on: requests carry ids,
    /// notifications don't, and non-JSON-RPC input is neither.
    #[test]
    fn classify_separates_requests_notifications_and_garbage() {
        assert!(matches!(
            classify(&json!({"jsonrpc": "2.0", "id": 1, "method": "ping"})),
            Message::Request {
                call: Call::Ping,
                ..
            }
        ));
        assert!(matches!(
            classify(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"})),
            Message::Notification
        ));
        // A null id is a notification too, not a request.
        assert!(matches!(
            classify(&json!({"jsonrpc": "2.0", "id": null, "method": "ping"})),
            Message::Notification
        ));
        assert!(matches!(
            classify(&json!({"jsonrpc": "2.0", "id": 1})),
            Message::Undecodable
        ));
        assert!(matches!(
            classify(&json!({"jsonrpc": "2.0", "id": 1, "method": "resources/list"})),
            Message::Request {
                call: Call::Unknown { .. },
                ..
            }
        ));
    }

    #[test]
    fn initialize_result_echoes_the_client_version_or_falls_back() {
        let echoed = initialize_result(Some("2025-06-18"), "manual");
        assert_eq!(echoed["protocolVersion"], "2025-06-18");
        assert_eq!(echoed["instructions"], "manual");
        assert_eq!(echoed["serverInfo"]["version"], env!("CARGO_PKG_VERSION"));

        let fallback = initialize_result(None, "manual");
        assert_eq!(fallback["protocolVersion"], FALLBACK_PROTOCOL_VERSION);
    }

    #[test]
    fn tool_response_marks_errors_without_aborting_the_rpc() {
        let ok = tool_response(Ok("fine".into()));
        assert_eq!(ok["content"][0]["text"], "fine");
        assert!(ok.get("isError").is_none());

        let err = tool_response(Err("HTTP 404: gone".into()));
        assert_eq!(err["isError"], true);
        assert_eq!(err["content"][0]["text"], "HTTP 404: gone");
    }
}
