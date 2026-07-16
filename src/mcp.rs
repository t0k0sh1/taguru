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

use std::collections::HashSet;

use serde_json::{Value, json};

/// Spoken when the client does not name a protocol version itself, or
/// names one this build does not recognize.
pub const FALLBACK_PROTOCOL_VERSION: &str = "2024-11-05";

/// The protocol versions this build has actually been written against.
/// `initialize` only echoes a client-named version drawn from this
/// list; anything else falls back rather than promising semantics
/// this server does not implement — the whole point of the version
/// exchange is the two sides agreeing on one wire contract, which a
/// blind echo would skip entirely.
const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2024-11-05", "2025-03-26", "2025-06-18"];

/// Hard ceiling on the `import` tool's `stream` argument, checked
/// before the stream ever leaves the process — `taguru-mcp` does not
/// link `ingest.rs` (its only path to the server is HTTP), so
/// `ingest::MAX_LINE_BYTES` is unreachable here and this stands in for
/// it. It is an upper bound, NOT the effective cap: the server's own
/// request body cap (`TAGURU_MAX_BODY_BYTES`, 8 MiB default) binds
/// first at that default. The bridge POSTs the raw stream to `/import`
/// under that cap, and over the `/mcp` HTTP transport the stream is
/// JSON-quoted into the *outer* envelope (every newline escaped to
/// `\n`, close to double its raw size) which must itself fit the body
/// cap — so a stream between the body cap and this ceiling passes here
/// only to be 413'd by the server. This 32 MiB becomes the binding
/// limit solely once an operator raises `TAGURU_MAX_BODY_BYTES` above
/// it. That same doubling is why `taguru-mcp`'s per-line frame cap
/// (`TAGURU_MCP_MAX_LINE_BYTES`) defaults to ~2× this value: a line
/// under the frame cap must still be able to carry a full-size stream.
const MAX_IMPORT_STREAM_BYTES: usize = 32 * 1024 * 1024;

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
/// [`Message::Undecodable`], which both transports answer with a
/// JSON-RPC error carrying whatever id was found — a message missing
/// only its method is still owed a correlatable reply, not a null one.
pub fn classify(message: &Value) -> Message {
    let id = match message.get("id") {
        Some(id) if !id.is_null() => id.clone(),
        _ => Value::Null,
    };
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

/// Percent-encodes text into the unreserved-only RFC 3986 set — safe
/// both as one URL path segment and as a query-string value.
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

/// [`object_schema`] for the search tools, which target one context or
/// several: `context`, `contexts`, and `groups` join the given
/// properties, and an `anyOf` demands at least one (the `cite_passage`
/// precedent). `contexts` and `groups` combine — both are the
/// cross-context form — but `context` beside either is refused by
/// `route_tool`, where the message can say so; a schema can only say
/// "invalid".
fn search_target_schema(properties: Value, required: &[&str]) -> Value {
    let mut schema = object_schema(properties, required);
    schema["properties"]["context"] =
        json!({ "type": "string", "description": "Context name (from list_contexts)" });
    schema["properties"]["contexts"] = json!({
        "type": "array",
        "items": { "type": "string" },
        "description": "several contexts at once — full names; every match comes back tagged with its context. Combines with groups; don't pass context beside it."
    });
    schema["properties"]["groups"] = json!({
        "type": "array",
        "items": { "type": "string" },
        "description": "group names (from list_groups) — each resolves to every context it reaches, nested children included, deduped against contexts and each other. Combines with contexts; don't pass context beside it."
    });
    schema["anyOf"] = json!([
        { "required": ["context"] },
        { "required": ["contexts"] },
        { "required": ["groups"] },
    ]);
    schema
}

/// Layers a second `anyOf` onto a [`search_target_schema`] result via
/// `allOf`, so both constraints hold at once — assigning straight into
/// `schema["anyOf"]` would silently replace the target-selection one
/// instead of adding to it.
fn require_any_of(mut schema: Value, alternatives: Value) -> Value {
    let target_any_of = schema.as_object_mut().unwrap().remove("anyOf").unwrap();
    schema["allOf"] = json!([{ "anyOf": target_any_of }, { "anyOf": alternatives }]);
    schema
}

/// Pulls a required string argument, telling an absent one apart from a
/// present-but-wrong-typed one — folding both into "missing" sends a
/// caller who passed `{"name": 42}` hunting for an argument they did
/// supply.
fn need<'a>(arguments: &'a Value, key: &str) -> Result<&'a str, String> {
    match arguments.get(key) {
        Some(Value::String(text)) => Ok(text),
        Some(Value::Null) | None => Err(format!("missing required argument '{key}'")),
        Some(_) => Err(format!("argument '{key}' must be a string")),
    }
}

/// `need`'s missing/null-counts-as-missing rule for a required
/// argument that isn't a string — an array or object body field, like
/// `add_associations`'s `associations`. Type checking past "present"
/// stays server-side, same as it already does for every argument
/// `pick` copies through untyped.
fn need_present<'a>(arguments: &'a Value, key: &str) -> Result<&'a Value, String> {
    match arguments.get(key) {
        Some(value) if !value.is_null() => Ok(value),
        _ => Err(format!("missing required argument '{key}'")),
    }
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

/// Builds a `?a=1&b=2` query string from the listed keys, skipping
/// absent/null ones — the GET-request counterpart of `pick`, for tools
/// that carry no body. Numbers pass through their JSON text; strings
/// are percent-encoded.
fn query_string(arguments: &Value, keys: &[&str]) -> String {
    let pairs: Vec<String> = keys
        .iter()
        .filter_map(|&key| {
            let value = arguments.get(key).filter(|value| !value.is_null())?;
            let text = match value {
                Value::String(text) => segment(text),
                Value::Number(number) => number.to_string(),
                Value::Bool(boolean) => boolean.to_string(),
                _ => return None,
            };
            Some(format!("{key}={text}"))
        })
        .collect();
    if pairs.is_empty() {
        String::new()
    } else {
        format!("?{}", pairs.join("&"))
    }
}

/// Like `pick`, but a value under `alias` counts for `canonical` when
/// `canonical` itself is absent — request-side back-compat for an argument
/// renamed after clients had already adopted the old name.
fn pick_with_alias(arguments: &Value, keys: &[&str], canonical: &str, alias: &str) -> Value {
    let mut body = pick(arguments, keys);
    if let Value::Object(map) = &mut body
        && !map.contains_key(canonical)
        && let Some(value) = arguments.get(alias)
        && !value.is_null()
    {
        map.insert(canonical.to_string(), value.clone());
    }
    body
}

/// Schema property `description` policy (#51): add one when a caller
/// cannot get the fact from the property's name, its own `type`, or
/// the tool's own `description` — a non-obvious default/ceiling
/// applied on omission, an `additionalProperties` map's key → value
/// shape, a deprecated-alias relationship, or a divergence from a
/// same-named property on a sibling tool (e.g. create vs update
/// semantics). Skip it when it would only restate the type or repeat
/// what the tool description already says. The same property, same
/// meaning, on two tools gets the same text; a real behavioral
/// difference gets stated, not silently dropped.
pub fn tool_definitions() -> Vec<Value> {
    let context = json!({ "type": "string", "description": "Context name (from list_contexts)" });
    let match_after = json!({
        "type": "object",
        "description": "resume past the previous page's last match: copy {weight, subject, label, object} verbatim from it, plus context too when targeting several contexts. total stays constant across pages",
        "properties": {
            "weight": { "type": "number" },
            "subject": { "type": "string" },
            "label": { "type": "string" },
            "object": { "type": "string" },
            "context": { "type": "string", "description": "required when targeting several contexts (contexts/groups); omit for a single context" }
        },
        "required": ["weight", "subject", "label", "object"]
    });
    let tools = vec![
        (
            "list_contexts",
            "Routing directory: every context's name, description, stats (counts, top concepts, label sample), and usage counters (reads/empty_reads/writes, last-used times). Pick the search/ingest target here yourself.",
            object_schema(
                json!({
                    "limit": { "type": "integer", "minimum": 0, "description": "page size, keyset-paged by name (default/ceiling 1000)" },
                    "after": { "type": "string", "description": "only contexts whose name sorts strictly after this one" },
                    "pinned": { "type": "boolean", "description": "only contexts with this pinned state" }
                }),
                &[],
            ),
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
            "Update description / pinned / dice_floor / semantic_floor.",
            object_schema(
                json!({
                    "name": { "type": "string" },
                    "description": { "type": "string" },
                    "pinned": { "type": "boolean", "description": "omit to leave unchanged" },
                    "dice_floor": { "type": "number", "description": "omit to leave unchanged" },
                    "semantic_floor": { "type": "number", "description": "omit to leave unchanged" }
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
            "rename_context",
            "Rename a context (admin role): the whole file family moves to the new name and every group naming it is rewritten to match. Fails if the destination name is already taken.",
            object_schema(
                json!({
                    "name": { "type": "string" },
                    "to": { "type": "string", "description": "the new name" }
                }),
                &["name", "to"],
            ),
        ),
        (
            "list_groups",
            "Group directory: every group's name, description, member context names, and child group names. A group bundles contexts (many-to-many) and may nest child groups up to 3 levels (cycles refused) — organize related contexts under one name. Groups and contexts are separate namespaces.",
            object_schema(
                json!({
                    "limit": { "type": "integer", "minimum": 0, "description": "page size, keyset-paged by name (default/ceiling 1000)" },
                    "after": { "type": "string", "description": "only groups whose name sorts strictly after this one" }
                }),
                &[],
            ),
        ),
        (
            "create_group",
            "Create a group bundling contexts and, optionally, child groups (nesting: at most 3 groups tall, never cyclic; each set holds at most 1000 names — past that, split into nested child groups). Every listed context and child group must already exist; membership never dangles — deleting a context or a group drops it from every group.",
            object_schema(
                json!({
                    "name": { "type": "string" },
                    "description": { "type": "string" },
                    "contexts": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "initial member context names (from list_contexts)"
                    },
                    "groups": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "initial child group names (from list_groups)"
                    }
                }),
                &["name"],
            ),
        ),
        (
            "update_group",
            "Update a group's description and/or membership. add_contexts/remove_contexts and add_groups/remove_groups are deltas against the current members, not a replacement list; a name in both ends up a member. Added contexts and child groups must exist; removing a non-member is a no-op; nesting stays at most 3 groups tall and acyclic, and the resulting membership at most 1000 member contexts and 1000 child groups (removals apply first, so one request can trade members within the cap).",
            object_schema(
                json!({
                    "name": { "type": "string" },
                    "description": { "type": "string", "description": "omit to leave unchanged" },
                    "add_contexts": { "type": "array", "items": { "type": "string" } },
                    "remove_contexts": { "type": "array", "items": { "type": "string" } },
                    "add_groups": { "type": "array", "items": { "type": "string" } },
                    "remove_groups": { "type": "array", "items": { "type": "string" } }
                }),
                &["name"],
            ),
        ),
        (
            "delete_group",
            "Delete a group (irreversible). Only the bundling goes; the member contexts, the child groups, and their data are untouched — parents naming the group just drop the child.",
            object_schema(json!({ "name": { "type": "string" } }), &["name"]),
        ),
        (
            "rename_group",
            "Rename a group (admin role): the group's file moves to the new name and every OTHER group naming it as a child is rewritten to match. Fails if the destination name is already taken.",
            object_schema(
                json!({
                    "name": { "type": "string" },
                    "to": { "type": "string", "description": "the new name" }
                }),
                &["name", "to"],
            ),
        ),
        (
            "add_associations",
            "Write facts as a batch (one document = one call, up to 10,000 associations; split larger documents), a source id on every element; single-fact calls cost a full durable write each, so collect a document's facts first. Discipline: check spellings with resolve/resolve_label and reuse before minting; don't re-assert paraphrases within one document; negation = positive label + negative weight; make implicit membership an explicit edge; weave ordered procedures with the three edges 最初の工程/次の工程/工程 (details in get_protocol).",
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
                                "source": { "type": "string" },
                                "paragraph": { "type": "integer", "description": "zero-based paragraph position" }
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
            "Register the original text behind each source id. Always finish an ingest with this; answers ground in originals looked up from attributions. Optionally attach doc2query questions per source ({source: [{paragraph, question}]}, paragraph = 0-based blank-line-separated position in THAT text): questions a user might type whose answer is that paragraph, phrased away from its wording — they embed beside the paragraph and catch question-shaped queries the text's own vector misses. Optionally attach section markers per source ({source: [{paragraph, section}]}, same paragraph numbering): a marker names where its section starts and the section implicitly governs every paragraph after it until the next marker or the passage's end — citation and every association read label their paragraph with the section that governs it.",
            object_schema(
                json!({
                    "context": context,
                    "passages": { "type": "object", "additionalProperties": { "type": "string" }, "description": "source → text" },
                    "questions": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "paragraph": { "type": "integer" },
                                    "question": { "type": "string" }
                                },
                                "required": ["paragraph", "question"]
                            }
                        }
                    },
                    "sections": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "paragraph": { "type": "integer" },
                                    "section": { "type": "string" }
                                },
                                "required": ["paragraph", "section"]
                            }
                        }
                    }
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
            "Source ids with registered passages — targets for retract_source / lookup_passages, inventory for diff sync. Keyset-paged by id; total above the returned count means more pages.",
            object_schema(
                json!({
                    "context": context,
                    "limit": { "type": "integer", "minimum": 0, "description": "page size (default/ceiling 1000)" },
                    "after": { "type": "string", "description": "only ids sorting strictly after this one" },
                    "prefix": { "type": "string", "description": "only ids starting with this text" }
                }),
                &["context"],
            ),
        ),
        (
            "resolve",
            "Resolve free wording to stored concept names (normalized entry, absorbs typos). The retrieval entry: use the canonical names it returns as origins for explore/activate. Each candidate says how it matched (kind: exact/alias = the cue IS a stored spelling; containment/fuzzy = it merely overlaps one) and carries a gloss of its heaviest facts — read the gloss before adopting a lookalike (京都 scores 0.67 against 東京都; the glosses tell them apart). Empty → reword, or lower dice_floor (e.g. 0.2) and retry.",
            object_schema(
                json!({
                    "context": context,
                    "cue": { "type": "string" },
                    "dice_floor": { "type": "number", "description": "one-call override of the fuzzy floor" },
                    "semantic_floor": { "type": "number", "description": "one-call override of the semantic floor" },
                    "limit": { "type": "integer", "minimum": 0, "description": "max candidates (default/ceiling 1000)" }
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
                    "dice_floor": { "type": "number", "description": "one-call override of the fuzzy floor" },
                    "semantic_floor": { "type": "number", "description": "one-call override of the semantic floor" },
                    "limit": { "type": "integer", "minimum": 0, "description": "max candidates (default/ceiling 1000)" }
                }),
                &["context", "cue"],
            ),
        ),
        (
            "explain_resolve",
            "Why didn't (or did) a concept come back for a cue — one call instead of re-running resolve with varied floors and cross-referencing by hand. Name the cue AND the concept you expected; the answer is the first verdict that applies: not_in_vocabulary (nearest stored spellings attached — register an alias?), cue_resolved_exactly (the cue IS another stored spelling; the exact tier answers alone), below_floor (its actual score vs the dice_floor in effect — the floor that would have shown it), below_cutoff (passed the floor, lost on limit), semantic_not_run / semantic_below_floor (whether the fallback tier joined, and its gloss cosine vs the semantic floor when it did), or served. Pass the same overrides as the resolve call being questioned.",
            object_schema(
                json!({
                    "context": context,
                    "cue": { "type": "string" },
                    "expected": { "type": "string", "description": "the concept you expected among the candidates" },
                    "dice_floor": { "type": "number", "description": "one-call override of the fuzzy floor" },
                    "semantic_floor": { "type": "number", "description": "one-call override of the semantic floor" },
                    "limit": { "type": "integer", "minimum": 0, "description": "max candidates (default/ceiling 1000)" }
                }),
                &["context", "cue", "expected"],
            ),
        ),
        (
            "explain_resolve_label",
            "explain_resolve, for relation labels.",
            object_schema(
                json!({
                    "context": context,
                    "cue": { "type": "string" },
                    "expected": { "type": "string", "description": "the label you expected among the candidates" },
                    "dice_floor": { "type": "number", "description": "one-call override of the fuzzy floor" },
                    "semantic_floor": { "type": "number", "description": "one-call override of the semantic floor" },
                    "limit": { "type": "integer", "minimum": 0, "description": "max candidates (default/ceiling 1000)" }
                }),
                &["context", "cue", "expected"],
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
            "Position-pinned search. subject/label/object each take a string or an array (array = match any); at least one of the three must be given — leaving all three out is refused rather than matching everything. Outline with describe, then narrow by label. Targets one context (context) or several at once (contexts and/or groups) — cross-context matches carry their context, and past the limit the strongest |weight| survives (weights share one scale).",
            require_any_of(
                search_target_schema(
                    json!({
                        "subject": { "type": ["string", "array"] },
                        "label": { "type": ["string", "array"] },
                        "object": { "type": ["string", "array"] },
                        "limit": { "type": "integer", "minimum": 0, "description": "default 100, capped at 1000" },
                        "after": match_after
                    }),
                    &[],
                ),
                json!([
                    { "required": ["subject"] },
                    { "required": ["label"] },
                    { "required": ["object"] },
                ]),
            ),
        ),
        (
            "recall",
            "Every association touching the cue, whatever its position. Use query when the role matters. Targets one context (context) or several at once (contexts and/or groups) — cross-context matches carry their context, and past the limit the strongest |weight| survives (weights share one scale).",
            search_target_schema(
                json!({
                    "cue": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 0, "description": "default 100, capped at 1000" },
                    "after": match_after
                }),
                &["cue"],
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
                    "limit": { "type": "integer", "minimum": 0, "description": "default 20" }
                }),
                &["context", "origins"],
            ),
        ),
        (
            "explore",
            "Exhaustive structural walk with hop distances, for unranked neighborhood views. Truncation keeps the nearest hops (watch total).",
            object_schema(
                json!({
                    "context": context,
                    "origins": { "type": "array", "items": { "type": "string" } },
                    "max_depth": { "type": "integer", "description": "hop ceiling; default and max 10" },
                    "limit": { "type": "integer", "minimum": 0, "description": "default 100, capped at 1000" },
                    "after": {
                        "type": "object",
                        "description": "resume past the previous page's last recollection — copy every field verbatim from it. total stays constant across pages",
                        "properties": {
                            "distance": { "type": "integer" },
                            "subject": { "type": "string" },
                            "label": { "type": "string" },
                            "object": { "type": "string" }
                        },
                        "required": ["distance", "subject", "label", "object"]
                    }
                }),
                &["context", "origins"],
            ),
        ),
        (
            "list_labels",
            "The relation vocabulary (canonical only). Read it before extracting to avoid spelling forks. Keyset-paged by label; total above the returned count means more pages.",
            object_schema(
                json!({
                    "context": context,
                    "limit": { "type": "integer", "minimum": 0, "description": "page size (default/ceiling 1000)" },
                    "after": { "type": "string", "description": "only labels sorting strictly after this one" },
                    "prefix": { "type": "string", "description": "only labels starting with this text" }
                }),
                &["context"],
            ),
        ),
        (
            "get_aliases",
            "Registered aliases (alias → canonical), paged across both namespaces — concepts first, then labels. total above the returned count means more pages; continue with after = 'concept:<alias>' or 'label:<alias>' (the last entry shown).",
            object_schema(
                json!({
                    "context": context,
                    "limit": { "type": "integer", "minimum": 0, "description": "page size (default/ceiling 1000)" },
                    "after": { "type": "string", "description": "'concept:<alias>' or 'label:<alias>' — the last entry of the previous page" },
                    "prefix": { "type": "string", "description": "only aliases (in either namespace) starting with this text" }
                }),
                &["context"],
            ),
        ),
        (
            "add_aliases",
            "Point alternate spellings at canonical names (entry-only; results always return canonicals). The fix when live wording misses. Cannot join two existing concepts — that would be a merge, which is rebuild territory.",
            object_schema(
                json!({
                    "context": context,
                    "concepts": { "type": "object", "additionalProperties": { "type": "string" }, "description": "alias → canonical" },
                    "labels": { "type": "object", "additionalProperties": { "type": "string" }, "description": "alias → canonical" }
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
                    "labels": { "type": "array", "items": { "type": "string" }, "description": "alias spellings to withdraw" }
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
            "retract_association",
            "Withdraw one (subject, label, object) association outright — every source's contribution to that one edge, where retract_source would discard a whole document's. The surgical correction for a fact that should never have been asserted (an extraction error, a merge mistake). A fact that is merely CONTESTED wants a negative-weight assertion instead, which preserves the dispute as evidence. Names resolve through aliases; `retracted: false` means the triple named no live edge and nothing changed. The edge row stays visible at weight 0 until compaction; re-asserting the triple later just works.",
            object_schema(
                json!({
                    "context": context,
                    "subject": { "type": "string" },
                    "label": { "type": "string" },
                    "object": { "type": "string" }
                }),
                &["context", "subject", "label", "object"],
            ),
        ),
        (
            "search_passages",
            "Paragraph search over registered passages: a lexical lane (bigram BM25) fused with a semantic lane (paragraph embeddings) where the server has them. The text lane for knowledge that never fit triples (order, conditions, discourse) — look here too when graph search comes up short. The semantic lane works best on declarative phrasing: rephrase the information need as a plausible ANSWER sentence, not a question (query \"SSO is included in the Enterprise plan\", not \"What plan includes SSO?\") — the guess only has to be shaped like the text you hope to find. Each hit names its paragraph (source + paragraph) and reports per-lane rank/score in `lanes`; a hit only the vector lane surfaced is exactly the paraphrase case the lexical lane cannot see. Targets one context (context) or several at once (contexts and/or groups) — cross-context hits carry their context and interleave by per-context rank; scores compare within one context only.",
            search_target_schema(
                json!({
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 0, "description": "default 5" }
                }),
                &["query"],
            ),
        ),
        (
            "explain_search",
            "Why didn't (or did) a source appear in search_passages — one call instead of orchestrating search, citations, and lowered limits by hand. Name the query AND the source (optionally which paragraph) you expected; the answer is the first verdict that applies: not_stored (never ingested here, or retracted), no_term_overlap (the query's terms and the paragraph's terms side by side, as strings — the spelling-mismatch case: stored under 酒蔵, you searched 酒造 — register an alias or reword), below_cutoff (its actual rank, the score cutoff at your limit, and a verified limit that reaches it), or served (its rank — it WAS there). Evidence carries per-term tf/df/BM25 contributions and the vector lane's cosine or the reason that lane never ran. One context per call.",
            object_schema(
                json!({
                    "context": context,
                    "query": { "type": "string" },
                    "source": { "type": "string", "description": "the source you expected among the hits" },
                    "paragraph": { "type": "integer", "description": "zero-based paragraph position; omitted picks the source's best showing" },
                    "limit": { "type": "integer", "minimum": 0, "description": "the search call being explained (default 5)" }
                }),
                &["context", "query", "source"],
            ),
        ),
        (
            "cite_passage",
            "Fetch one located, verbatim excerpt from a registered source by paragraph position: the citation counterpart of lookup_passages' whole-document dereference. Returns the exact paragraph text plus source and section provenance.",
            json!({
                "type": "object",
                "properties": {
                    "context": context,
                    "source": { "type": "string" },
                    "paragraph": { "type": "integer", "description": "zero-based paragraph position" },
                    "index": { "type": "integer", "description": "deprecated alias for `paragraph`; kept for pre-#35 callers, prefer `paragraph`" }
                },
                "required": ["context", "source"],
                "anyOf": [
                    { "required": ["paragraph"] },
                    { "required": ["index"] }
                ]
            }),
        ),
        (
            "retrieve",
            "The composed retrieval loop the SDKs' Context.retrieve() runs, as one call: resolve each origin cue to an anchor (auto-picking the top candidate; every candidate, gloss included, still comes back under resolved so a bad auto-pick is visible), describe each anchor, gather associations (query when labels pins the facets, activate always), fetch a citation for every located attribution, and optionally fall back to passage search. origins must already be extracted entity names, not a question — decomposing a question and phrasing a declarative text_fallback_query are the caller's job. citations rides back as a list of {source, paragraph, citation} (paragraphs missing a stored passage are silently skipped, same as the SDKs).",
            object_schema(
                json!({
                    "context": context,
                    "origins": {
                        "type": ["string", "array"],
                        "items": { "type": "string" },
                        "description": "cue(s) to resolve into anchors"
                    },
                    "labels": {
                        "type": ["string", "array"],
                        "items": { "type": "string" },
                        "description": "relation labels to query on, alongside the always-run activate"
                    },
                    "dice_floor": { "type": "number", "description": "one-call override of the resolve fuzzy floor" },
                    "semantic_floor": { "type": "number", "description": "one-call override of the resolve semantic floor" },
                    "resolve_limit": { "type": "integer", "minimum": 0, "description": "max resolve candidates per origin (default/ceiling 1000)" },
                    "auto_pick": { "type": "boolean", "description": "adopt each origin's top resolve candidate as its anchor; false uses the cue itself verbatim (default true)" },
                    "describe_first": { "type": "boolean", "description": "describe every anchor before gathering associations (default true)" },
                    "activate_decay": { "type": "number", "description": "activate's decay (default 0.5)" },
                    "activate_limit": { "type": "integer", "minimum": 0, "description": "activate's limit (default 20)" },
                    "fetch_citations": { "type": "boolean", "description": "resolve every located attribution into a cited passage (default true)" },
                    "text_fallback_query": { "type": "string", "description": "declarative-phrasing query for a search_passages fallback pass; omitted runs no fallback" },
                    "text_fallback_only_if_empty": { "type": "boolean", "description": "only run the fallback when no associations were gathered (default true)" },
                    "search_limit": { "type": "integer", "minimum": 0, "description": "the fallback search_passages call's limit (default 5)" }
                }),
                &["context", "origins"],
            ),
        ),
        (
            "refresh_embeddings",
            "After ingesting, re-embed what changed (servers with embeddings only): the glosses (name + graph context) of new or changed concepts and labels, and — where the server opted in — the stored paragraphs. Makes paraphrases and question-shaped cues land through resolve's semantic fallback and search_passages' vector lane.",
            object_schema(json!({ "context": context }), &["context"]),
        ),
        (
            "audit_vocabulary",
            "Vocabulary health check: lexical fork candidates (青嶺酒蔵/青嶺酒造) and semantic ones (創業年/設立年; needs embeddings). Candidates, not verdicts — same referent → alias onto one canonical; different things that will keep colliding → record one ordinary 別物/distinct_from fact (one direction suffices; it lands in both glosses and warns future resolves). Run at ingest milestones.",
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
                    "origins": { "type": "array", "items": { "type": "string" } },
                    "limit": { "type": "integer", "minimum": 0, "description": "default 100, capped at 1000" },
                    "after": {
                        "type": "object",
                        "description": "resume past the previous page's last match — copy every field verbatim from it. total stays constant across pages",
                        "properties": {
                            "weight": { "type": "number" },
                            "subject": { "type": "string" },
                            "label": { "type": "string" },
                            "object": { "type": "string" }
                        },
                        "required": ["weight", "subject", "label", "object"]
                    }
                }),
                &["context", "origins"],
            ),
        ),
        (
            "audit_drift",
            "Graph-vs-archive drift audit: three read-only checks in one call. Unsourced weight — edges carrying weight no named source explains, the residue plain associate() calls or an export/import round trip leave behind — worst-first, paginated, filterable with unsourced_floor. Dead-canonical aliases — alias spellings whose canonical concept or label has zero live edges left. Optionally (include_twins) the same lexical/semantic fork candidates audit_vocabulary finds. Run periodically to catch drift audit_coverage and audit_vocabulary don't: weight nothing ingested explains, and aliases pointing at names nothing uses anymore.",
            object_schema(
                json!({
                    "context": context,
                    "unsourced_floor": { "type": "number", "description": "minimum unsourced weight (by magnitude) to include; default: any amount at all" },
                    "limit": { "type": "integer", "minimum": 0, "description": "default 100, capped at 1000" },
                    "after": {
                        "type": "object",
                        "description": "resume past the previous page's last unsourced match — copy every field verbatim from it. total stays constant across pages",
                        "properties": {
                            "weight": { "type": "number" },
                            "subject": { "type": "string" },
                            "label": { "type": "string" },
                            "object": { "type": "string" }
                        },
                        "required": ["weight", "subject", "label", "object"]
                    },
                    "include_twins": { "type": "boolean", "description": "also run the lexical/semantic fork-candidate sweep and include it as `twins` (default false — it's the same CPU-bound pairwise scan audit_vocabulary runs)" },
                    "dice_floor": { "type": "number", "description": "lexical floor (default 0.6); only used when include_twins is set" },
                    "cosine_floor": { "type": "number", "description": "semantic floor (default 0.6); only used when include_twins is set" }
                }),
                &["context"],
            ),
        ),
        (
            "flush",
            "Persist every dirty context to disk now; answers the flushed names (admin role). The backup handshake's first half: flush, then snapshot the data directory — the same discipline the operator docs describe, reachable by an agent tending its own memory.",
            object_schema(json!({}), &[]),
        ),
        (
            "export_context",
            "The whole context as an import batch stream (JSON Lines text) — one batch per source, create block first, aliases last; `taguru import` or POST /import restores it (per-source retract-then-apply, idempotent). The portable, version-independent backup of one context. The stream rides back as one text block: for very large contexts prefer GET /contexts/{name}/export over plain HTTP, or `taguru export` offline.",
            object_schema(json!({ "context": context }), &["context"]),
        ),
        (
            "export_group",
            "One group as its import-stream record (a single `taguru_group` JSON line — the group's complete truth); importing it restores the group as a whole-record replace. A context-scoped key exports exactly the slice its grant can read.",
            object_schema(
                json!({ "name": { "type": "string", "description": "Group name (from list_groups)" } }),
                &["name"],
            ),
        ),
        (
            "get_context",
            "One directory row by name — the cheap existence-and-stats check, without listing everything else.",
            object_schema(json!({ "context": context }), &["context"]),
        ),
        (
            "get_group",
            "One group's name, description, member contexts, and child groups.",
            object_schema(
                json!({ "name": { "type": "string", "description": "Group name (from list_groups)" } }),
                &["name"],
            ),
        ),
        (
            "compact",
            "Rebuild one context's on-disk image without the dead weight the append-only format accumulates (retracted edges, unlinked attributions, arena slack); answers what was shed and the resulting footprint (admin role). Content is preserved — this is maintenance, not a knowledge change.",
            object_schema(json!({ "context": context }), &["context"]),
        ),
        (
            "import",
            "Apply (or, with dry_run: true, preview) an NDJSON import stream — the same format `taguru import`/POST /import accept: a create block, associations, aliases, and passage per source, retract-then-apply and idempotent (admin role). A dry run writes nothing; its `associations`/`aliases` counts are optimistic previews, every other field exact. `taguru_group` records in the stream are not applied through this tool (their outcome is likewise not previewed) — use POST /import directly for a stream that carries any. Bounded by the server's request body cap (TAGURU_MAX_BODY_BYTES, 8 MiB by default) — and smaller over the /mcp HTTP transport, where the stream is escaped into the JSON-RPC envelope that must itself fit that cap — with a hard 32 MiB tool ceiling above it; a larger stream needs POST /import or `taguru import` directly.",
            object_schema(
                json!({
                    "stream": { "type": "string", "description": "NDJSON import stream (one taguru_batch/taguru_group/fact/alias/passage line per row)" },
                    "dry_run": { "type": "boolean", "description": "preview without writing anything (default false)" }
                }),
                &["stream"],
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
    let group_path = |key: &str| -> Result<String, String> {
        Ok(format!("/groups/{}", segment(need(arguments, key)?)))
    };
    // The search tools target one context or several: `context`
    // prefixes the per-context path; `contexts` and/or `groups`
    // (arrays, riding the body) mean the cross-context route — no
    // prefix. `context` beside either is ambiguous, and none at all
    // names no target.
    let search_base = || -> Result<String, String> {
        let given = |key: &str| arguments.get(key).is_some_and(|value| !value.is_null());
        match (given("context"), given("contexts") || given("groups")) {
            (true, true) => {
                Err("pass either 'context' or 'contexts'/'groups', not both".to_string())
            }
            (false, false) => Err(
                "missing required argument 'context' (or 'contexts'/'groups', to search several at once)"
                    .to_string(),
            ),
            (true, false) => context_path("context"),
            (false, true) => Ok(String::new()),
        }
    };
    Ok(match name {
        "get_protocol" => ("GET", "/protocol".to_string(), None),
        "flush" => ("POST", "/flush".to_string(), None),
        "export_context" => ("GET", format!("{}/export", context_path("context")?), None),
        "export_group" => ("GET", format!("{}/export", group_path("name")?), None),
        "get_context" => ("GET", context_path("context")?, None),
        "get_group" => ("GET", group_path("name")?, None),
        "compact" => (
            "POST",
            format!("{}/compact", context_path("context")?),
            None,
        ),
        "import" => {
            let stream = need(arguments, "stream")?;
            if stream.len() > MAX_IMPORT_STREAM_BYTES {
                return Err(format!(
                    "stream argument is {} bytes, over the {MAX_IMPORT_STREAM_BYTES}-byte \
                     tool limit; split the import or POST the stream to /import directly",
                    stream.len()
                ));
            }
            (
                "POST",
                format!("/import{}", query_string(arguments, &["dry_run"])),
                Some(Value::String(stream.to_string())),
            )
        }
        "list_contexts" => (
            "GET",
            format!(
                "/contexts{}",
                query_string(arguments, &["limit", "after", "pinned"])
            ),
            None,
        ),
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
        "rename_context" => {
            let path = format!("{}/rename", context_path("name")?);
            need(arguments, "to")?;
            ("POST", path, Some(pick(arguments, &["to"])))
        }
        "list_groups" => (
            "GET",
            format!("/groups{}", query_string(arguments, &["limit", "after"])),
            None,
        ),
        "create_group" => (
            "PUT",
            group_path("name")?,
            Some(pick(arguments, &["description", "contexts", "groups"])),
        ),
        "update_group" => (
            "PATCH",
            group_path("name")?,
            Some(pick(
                arguments,
                &[
                    "description",
                    "add_contexts",
                    "remove_contexts",
                    "add_groups",
                    "remove_groups",
                ],
            )),
        ),
        "delete_group" => ("DELETE", group_path("name")?, None),
        "rename_group" => {
            let path = format!("{}/rename", group_path("name")?);
            need(arguments, "to")?;
            ("POST", path, Some(pick(arguments, &["to"])))
        }
        "add_associations" => {
            // Resolve `context` first so a caller who omitted BOTH hears
            // about the primary argument, not the secondary one, in the
            // order the schema lists them.
            let path = format!("{}/associations", context_path("context")?);
            // Schema-required: an omitted (or null) argument must
            // refuse, not fall back to an empty batch — that would
            // route a caller's mistake into a silent, do-nothing 200.
            let associations = arguments
                .get("associations")
                .filter(|value| !value.is_null())
                .cloned()
                .ok_or_else(|| "missing required argument 'associations'".to_string())?;
            ("POST", path, Some(associations))
        }
        "store_passages" => {
            let path = format!("{}/sources", context_path("context")?);
            need_present(arguments, "passages")?;
            (
                "POST",
                path,
                Some(pick(arguments, &["passages", "questions", "sections"])),
            )
        }
        "lookup_passages" => {
            let path = format!("{}/sources/lookup", context_path("context")?);
            need_present(arguments, "sources")?;
            ("POST", path, Some(pick(arguments, &["sources"])))
        }
        "list_sources" => (
            "GET",
            format!(
                "{}/sources{}",
                context_path("context")?,
                query_string(arguments, &["limit", "after", "prefix"])
            ),
            None,
        ),
        "resolve" => {
            let path = format!("{}/resolve", context_path("context")?);
            need(arguments, "cue")?;
            (
                "POST",
                path,
                Some(pick(
                    arguments,
                    &["cue", "dice_floor", "semantic_floor", "limit"],
                )),
            )
        }
        "resolve_label" => {
            let path = format!("{}/resolve_label", context_path("context")?);
            need(arguments, "cue")?;
            (
                "POST",
                path,
                Some(pick(
                    arguments,
                    &["cue", "dice_floor", "semantic_floor", "limit"],
                )),
            )
        }
        "explain_resolve" => {
            let path = format!("{}/resolve/explain", context_path("context")?);
            need(arguments, "cue")?;
            need(arguments, "expected")?;
            (
                "POST",
                path,
                Some(pick(
                    arguments,
                    &["cue", "expected", "dice_floor", "semantic_floor", "limit"],
                )),
            )
        }
        "explain_resolve_label" => {
            let path = format!("{}/resolve_label/explain", context_path("context")?);
            need(arguments, "cue")?;
            need(arguments, "expected")?;
            (
                "POST",
                path,
                Some(pick(
                    arguments,
                    &["cue", "expected", "dice_floor", "semantic_floor", "limit"],
                )),
            )
        }
        "describe" => {
            let path = format!("{}/describe", context_path("context")?);
            need(arguments, "concept")?;
            ("POST", path, Some(pick(arguments, &["concept"])))
        }
        "query" => (
            "POST",
            format!("{}/query", search_base()?),
            Some(pick(
                arguments,
                &[
                    "contexts", "groups", "subject", "label", "object", "limit", "after",
                ],
            )),
        ),
        "recall" => {
            let path = format!("{}/recall", search_base()?);
            need(arguments, "cue")?;
            (
                "POST",
                path,
                Some(pick(
                    arguments,
                    &["contexts", "groups", "cue", "limit", "after"],
                )),
            )
        }
        "activate" => {
            let path = format!("{}/activate", context_path("context")?);
            need_present(arguments, "origins")?;
            (
                "POST",
                path,
                Some(pick(arguments, &["origins", "decay", "limit"])),
            )
        }
        "explore" => {
            let path = format!("{}/explore", context_path("context")?);
            need_present(arguments, "origins")?;
            (
                "POST",
                path,
                Some(pick(arguments, &["origins", "max_depth", "limit", "after"])),
            )
        }
        "list_labels" => (
            "GET",
            format!(
                "{}/labels{}",
                context_path("context")?,
                query_string(arguments, &["limit", "after", "prefix"])
            ),
            None,
        ),
        "get_aliases" => (
            "GET",
            format!(
                "{}/aliases{}",
                context_path("context")?,
                query_string(arguments, &["limit", "after", "prefix"])
            ),
            None,
        ),
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
        "retract_source" => {
            let path = format!("{}/sources/retract", context_path("context")?);
            need(arguments, "source")?;
            ("POST", path, Some(pick(arguments, &["source"])))
        }
        "retract_association" => {
            let path = format!("{}/associations/retract", context_path("context")?);
            need(arguments, "subject")?;
            need(arguments, "label")?;
            need(arguments, "object")?;
            (
                "POST",
                path,
                Some(pick(arguments, &["subject", "label", "object"])),
            )
        }
        "search_passages" => {
            let path = format!("{}/sources/search", search_base()?);
            need(arguments, "query")?;
            (
                "POST",
                path,
                Some(pick(arguments, &["contexts", "groups", "query", "limit"])),
            )
        }
        "explain_search" => {
            let path = format!("{}/sources/search/explain", context_path("context")?);
            need(arguments, "query")?;
            need(arguments, "source")?;
            (
                "POST",
                path,
                Some(pick(arguments, &["query", "source", "paragraph", "limit"])),
            )
        }
        "cite_passage" => {
            let path = format!("{}/citations", context_path("context")?);
            need(arguments, "source")?;
            let has_paragraph = arguments
                .get("paragraph")
                .is_some_and(|value| !value.is_null());
            let has_index = arguments.get("index").is_some_and(|value| !value.is_null());
            if !has_paragraph && !has_index {
                return Err(
                    "missing required argument 'paragraph' (or its deprecated alias 'index')"
                        .to_string(),
                );
            }
            (
                "POST",
                path,
                Some(pick_with_alias(
                    arguments,
                    &["source", "paragraph"],
                    "paragraph",
                    "index",
                )),
            )
        }
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
        "audit_coverage" => {
            let path = format!("{}/unreachable_from", context_path("context")?);
            need_present(arguments, "origins")?;
            (
                "POST",
                path,
                Some(pick(arguments, &["origins", "limit", "after"])),
            )
        }
        "audit_drift" => (
            "POST",
            format!("{}/drift/audit", context_path("context")?),
            Some(pick(
                arguments,
                &[
                    "unsourced_floor",
                    "limit",
                    "after",
                    "include_twins",
                    "dice_floor",
                    "cosine_floor",
                ],
            )),
        ),
        _ => return Err(format!("unknown tool '{name}'")),
    })
}

/// Extracts `(subject, label, object)` from an `AssociationOut`-shaped
/// value, for `run_retrieve`'s cross-step deduplication. `None` for
/// anything not shaped that way, which the caller treats as "keep it,
/// nothing to dedupe against" rather than dropping it.
fn triple_of(association: &Value) -> Option<(String, String, String)> {
    Some((
        association.get("subject")?.as_str()?.to_string(),
        association.get("label")?.as_str()?.to_string(),
        association.get("object")?.as_str()?.to_string(),
    ))
}

/// Ceiling on the `origins` cue list [`run_retrieve`] accepts. Each cue
/// drives its own `resolve` round trip (and, with describe_first, a
/// `describe`), so an unbounded list would amplify one composed call
/// into arbitrarily many requests — slipping past the per-request cap
/// the direct read endpoints put on list inputs, which it reaches one
/// cue at a time. Mirrors `api::MAX_INPUT_ITEMS`; restated here because
/// this module compiles into the stdio bridge too, which carries no
/// `api` module to borrow the constant from.
const MAX_ORIGIN_CUES: usize = 1000;

/// [`run_retrieve_bounded`] with no byte budget — every planned call
/// fires unconditionally. What the stdio bridge calls; `taguru-mcp.rs`'s
/// `dispatch_tool` documents why its composition stays uncapped.
#[allow(dead_code)] // consumed by the stdio bridge; the HTTP transport always calls run_retrieve_bounded instead
pub fn run_retrieve(
    arguments: &Value,
    call: impl FnMut(&'static str, String, Option<Value>) -> Result<String, String>,
) -> Result<Value, String> {
    run_retrieve_bounded(arguments, None, call)
}

/// The composed retrieval loop (`Context.retrieve()` in both SDKs),
/// reimplemented here so an MCP-only agent gets it in one call instead
/// of orchestrating five tool calls by hand. `route_tool` stays a pure
/// one-shot `(method, path, body)` mapping — this is deliberately a
/// separate function rather than another `route_tool` arm, since it
/// issues a variable number of requests built from earlier ones'
/// results. Each step still builds its request by calling `route_tool`
/// itself, so this can never drift from the single-call tools it
/// composes. `call` performs one routed request; the two transports
/// supply it (a ureq round trip for the stdio bridge, an in-process
/// dispatch for the HTTP transport, which must bridge onto its own
/// async call itself).
///
/// `budget`, when `Some`, caps the running total of every dispatched
/// call's raw response size: once one call pushes the total past it,
/// the next `call_tool` refuses before firing rather than composing
/// (and paying the round-trip cost for) a result the caller's own size
/// cap would discard anyway. The running total only ever over-counts
/// the true composed size — a step often keeps just one field of a
/// response, e.g. `"result"` — so this can cut off a little early but
/// never late; the caller's own post-hoc check on the final value
/// stays the source of truth either way.
pub fn run_retrieve_bounded(
    arguments: &Value,
    budget: Option<usize>,
    mut call: impl FnMut(&'static str, String, Option<Value>) -> Result<String, String>,
) -> Result<Value, String> {
    let mut spent: usize = 0;
    let mut call_tool = |name: &'static str, args: Value| -> Result<Value, String> {
        let (method, path, body) = route_tool(name, &args)?;
        let text = call(method, path, body)?;
        spent += text.len();
        if let Some(budget) = budget
            && spent > budget
        {
            return Err(format!(
                "retrieve's composed result already exceeds {budget} bytes after the \
                 '{name}' call; narrow it — fewer origins, a smaller resolve_limit or \
                 activate_limit, or fetch_citations: false — rather than paying for calls \
                 whose result would be discarded anyway"
            ));
        }
        serde_json::from_str::<Value>(&text)
            .map_err(|error| format!("tool '{name}' returned invalid JSON: {error}"))
    };

    let context = need(arguments, "context")?.to_string();
    let origins: Vec<String> = match arguments.get("origins") {
        Some(Value::String(text)) => vec![text.clone()],
        Some(Value::Array(items)) => {
            // Each origin cue fans out to its own `resolve` round trip (and,
            // with describe_first, a `describe`), so an unbounded list
            // amplifies one call into arbitrarily many — slipping past the
            // per-request list cap the direct read endpoints enforce, since it
            // reaches them one cue at a time. Refuse an oversized list up
            // front — before cloning every cue into a `String` — at the same
            // ceiling `overlong` applies to `origins` on those endpoints.
            if items.len() > MAX_ORIGIN_CUES {
                return Err(format!(
                    "argument 'origins' carries {} cues, past the per-request limit of {}; \
                     split the retrieval",
                    items.len(),
                    MAX_ORIGIN_CUES
                ));
            }
            items
                .iter()
                .map(|item| {
                    item.as_str().map(str::to_string).ok_or_else(|| {
                        "argument 'origins' must be a string or an array of strings".to_string()
                    })
                })
                .collect::<Result<_, _>>()?
        }
        Some(Value::Null) | None => return Err("missing required argument 'origins'".to_string()),
        Some(_) => {
            return Err("argument 'origins' must be a string or an array of strings".to_string());
        }
    };
    let auto_pick = arguments
        .get("auto_pick")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let describe_first = arguments
        .get("describe_first")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let fetch_citations = arguments
        .get("fetch_citations")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let text_fallback_only_if_empty = arguments
        .get("text_fallback_only_if_empty")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    // Step 1: resolve each origin cue, auto-picking the top candidate
    // (or falling back to the cue itself verbatim when auto_pick is
    // off) into a deduplicated anchor list.
    let mut resolved = serde_json::Map::new();
    let mut anchors: Vec<String> = Vec::new();
    for cue in &origins {
        let mut resolve_args = pick(arguments, &["dice_floor", "semantic_floor"]);
        resolve_args["context"] = json!(context);
        resolve_args["cue"] = json!(cue);
        if let Some(limit) = arguments.get("resolve_limit").filter(|v| !v.is_null()) {
            resolve_args["limit"] = limit.clone();
        }
        let candidates = call_tool("resolve", resolve_args)?
            .get("result")
            .cloned()
            .unwrap_or(Value::Array(Vec::new()));
        let picked = if auto_pick {
            candidates
                .as_array()
                .and_then(|list| list.first())
                .and_then(|first| first.get("name"))
                .and_then(Value::as_str)
                .map(str::to_string)
        } else {
            Some(cue.clone())
        };
        resolved.insert(cue.clone(), candidates);
        if let Some(picked) = picked
            && !anchors.contains(&picked)
        {
            anchors.push(picked);
        }
    }

    // Step 2: describe every anchor — skippable via describe_first: false.
    let mut outline = serde_json::Map::new();
    if describe_first {
        for anchor in &anchors {
            let described =
                call_tool("describe", json!({ "context": context, "concept": anchor }))?
                    .get("result")
                    .cloned()
                    .unwrap_or(Value::Null);
            outline.insert(anchor.clone(), described);
        }
    }

    // Step 3: gather associations — query (only when labels pins the
    // facets) then always activate, deduplicated by
    // (subject, label, object) with query's matches taking priority
    // over activate's (query runs first and wins the dedupe).
    let mut associations: Vec<Value> = Vec::new();
    let mut activations: Vec<Value> = Vec::new();
    let mut seen_triples: HashSet<(String, String, String)> = HashSet::new();
    if !anchors.is_empty() {
        if let Some(labels) = arguments.get("labels").filter(|v| !v.is_null()) {
            let matched = call_tool(
                "query",
                json!({ "context": context, "subject": anchors, "label": labels }),
            )?;
            for entry in matched
                .get("result")
                .and_then(|result| result.get("matches"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                match triple_of(entry) {
                    Some(triple) => {
                        if seen_triples.insert(triple) {
                            associations.push(entry.clone());
                        }
                    }
                    None => associations.push(entry.clone()),
                }
            }
        }
        let mut activate_args = json!({ "context": context, "origins": anchors });
        if let Some(decay) = arguments.get("activate_decay").filter(|v| !v.is_null()) {
            activate_args["decay"] = decay.clone();
        }
        if let Some(limit) = arguments.get("activate_limit").filter(|v| !v.is_null()) {
            activate_args["limit"] = limit.clone();
        }
        let page = call_tool("activate", activate_args)?;
        activations = page
            .get("result")
            .and_then(|result| result.get("matches"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for activation in &activations {
            let association = activation
                .get("association")
                .cloned()
                .unwrap_or(Value::Null);
            match triple_of(&association) {
                Some(triple) => {
                    if seen_triples.insert(triple) {
                        associations.push(association);
                    }
                }
                None => associations.push(association),
            }
        }
    }

    // Step 4: fetch a citation for every located attribution,
    // deduplicated by (source, paragraph). A locator whose passage was
    // never stored (or was retracted) is skipped rather than failing
    // the whole call — the graph fact still stands; any other failure
    // (auth, a downed server) aborts immediately.
    let mut citations: Vec<Value> = Vec::new();
    if fetch_citations {
        let mut wanted: Vec<(String, u64)> = Vec::new();
        let mut seen_keys: HashSet<(String, u64)> = HashSet::new();
        for association in &associations {
            for attribution in association
                .get("attributions")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let (Some(source), Some(paragraph)) = (
                    attribution.get("source").and_then(Value::as_str),
                    attribution.get("paragraph").and_then(Value::as_u64),
                ) else {
                    continue;
                };
                let key = (source.to_string(), paragraph);
                if seen_keys.insert(key.clone()) {
                    wanted.push(key);
                }
            }
        }
        for (source, paragraph) in wanted {
            match call_tool(
                "cite_passage",
                json!({ "context": context, "source": source, "paragraph": paragraph }),
            ) {
                Ok(response) => citations.push(json!({
                    "source": source,
                    "paragraph": paragraph,
                    "citation": response.get("result").cloned().unwrap_or(Value::Null),
                })),
                Err(message) if message.starts_with("HTTP 404") => continue,
                Err(message) => return Err(message),
            }
        }
    }

    // Step 5: text-lane fallback — only when the caller named a
    // fallback query, and (by default) only when no associations were
    // gathered.
    let mut passage_hits = Value::Array(Vec::new());
    if let Some(text_fallback_query) = arguments.get("text_fallback_query").and_then(Value::as_str)
        && (!text_fallback_only_if_empty || associations.is_empty())
    {
        let mut search_args = json!({ "context": context, "query": text_fallback_query });
        if let Some(limit) = arguments.get("search_limit").filter(|v| !v.is_null()) {
            search_args["limit"] = limit.clone();
        }
        passage_hits = call_tool("search_passages", search_args)?
            .get("result")
            .cloned()
            .unwrap_or(Value::Array(Vec::new()));
    }

    Ok(json!({
        "resolved": resolved,
        "outline": outline,
        "associations": associations,
        "activations": activations,
        "citations": citations,
        "passage_hits": passage_hits,
    }))
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
            "sources": ["s"], "source": "s", "query": "q", "paragraph": 0,
            "stream": "{}", "to": "ctx2", "expected": "x",
            "subject": "s", "label": "l", "object": "o",
        });
        for tool in tool_definitions() {
            let name = tool["name"].as_str().expect("definitions carry names");
            // retrieve is a composed multi-call tool with no single
            // (method, path, body) to map onto — run_retrieve's own
            // tests cover it.
            if name == "retrieve" {
                continue;
            }
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

    /// The HTTP layer deserializes every `limit` as `Option<usize>`, so a
    /// negative value should be refused at MCP schema validation instead
    /// of surfacing as a later deserialization failure.
    #[test]
    fn every_limit_property_has_a_minimum_of_zero() {
        for tool in tool_definitions() {
            let properties = &tool["inputSchema"]["properties"];
            if let Some(limit) = properties.get("limit") {
                assert_eq!(
                    limit["minimum"],
                    json!(0),
                    "tool '{}' limit lacks minimum: 0",
                    tool["name"]
                );
            }
        }
    }

    /// The search tools target one context or several: `contexts`
    /// and/or `groups` route to the cross-context path with the arrays
    /// in the body; `context` keeps the historical per-context route,
    /// body unchanged.
    #[test]
    fn search_tools_route_to_the_cross_context_paths_on_contexts() {
        let (method, path, body) =
            route_tool("recall", &json!({"contexts": ["a", "b"], "cue": "x"})).unwrap();
        assert_eq!((method, path.as_str()), ("POST", "/recall"));
        assert_eq!(body.unwrap(), json!({"contexts": ["a", "b"], "cue": "x"}));

        let (_, path, body) =
            route_tool("query", &json!({"contexts": ["a"], "subject": "s"})).unwrap();
        assert_eq!(path, "/query");
        assert_eq!(body.unwrap(), json!({"contexts": ["a"], "subject": "s"}));

        let (_, path, body) =
            route_tool("search_passages", &json!({"contexts": ["a"], "query": "q"})).unwrap();
        assert_eq!(path, "/sources/search");
        assert_eq!(body.unwrap(), json!({"contexts": ["a"], "query": "q"}));

        // `groups` alone reaches the same route, and beside `contexts`
        // both ride the body.
        let (_, path, body) = route_tool("recall", &json!({"groups": ["g"], "cue": "x"})).unwrap();
        assert_eq!(path, "/recall");
        assert_eq!(body.unwrap(), json!({"groups": ["g"], "cue": "x"}));
        let (_, path, body) = route_tool(
            "search_passages",
            &json!({"contexts": ["a"], "groups": ["g"], "query": "q"}),
        )
        .unwrap();
        assert_eq!(path, "/sources/search");
        assert_eq!(
            body.unwrap(),
            json!({"contexts": ["a"], "groups": ["g"], "query": "q"})
        );

        // The single-context form is untouched, and the body never
        // carries the path-bound name.
        let (_, path, body) = route_tool("recall", &json!({"context": "a", "cue": "x"})).unwrap();
        assert_eq!(path, "/contexts/a/recall");
        assert_eq!(body.unwrap(), json!({"cue": "x"}));
    }

    /// The one-context form beside the cross-context form is ambiguous
    /// and no target at all is no search — each refusal says which way
    /// to fix the call, and an explicit null counts as an omission (the
    /// `pick` rule).
    #[test]
    fn search_tools_refuse_an_ambiguous_or_absent_target() {
        let ambiguous = "pass either 'context' or 'contexts'/'groups', not both";
        assert_eq!(
            route_tool(
                "recall",
                &json!({"context": "a", "contexts": ["b"], "cue": "x"})
            ),
            Err(ambiguous.to_string())
        );
        assert_eq!(
            route_tool(
                "recall",
                &json!({"context": "a", "groups": ["g"], "cue": "x"})
            ),
            Err(ambiguous.to_string())
        );
        let missing = "missing required argument 'context' (or 'contexts'/'groups', to search several at once)";
        assert_eq!(
            route_tool("search_passages", &json!({"query": "q"})),
            Err(missing.to_string())
        );
        assert_eq!(
            route_tool(
                "recall",
                &json!({"context": null, "contexts": null, "cue": "x"})
            ),
            Err(missing.to_string())
        );
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

    /// `associations` is schema-required (unlike, say, `add_aliases`'s
    /// concepts/labels, which default to empty maps server-side). A
    /// caller that omits it made a mistake, not an empty-batch request
    /// — omission must refuse, not silently route as `[]` and come
    /// back a do-nothing 200.
    #[test]
    fn add_associations_without_the_associations_argument_is_refused() {
        assert_eq!(
            route_tool("add_associations", &json!({"context": "ctx"})),
            Err("missing required argument 'associations'".to_string())
        );
        // Explicit null is the same omission, not a value.
        assert_eq!(
            route_tool(
                "add_associations",
                &json!({"context": "ctx", "associations": null})
            ),
            Err("missing required argument 'associations'".to_string())
        );
        // A deliberate empty batch is a value, not an omission — it
        // still routes.
        assert!(
            route_tool(
                "add_associations",
                &json!({"context": "ctx", "associations": []})
            )
            .is_ok()
        );
    }

    /// A required string argument present with the wrong JSON type is a
    /// caller mistake distinct from omission — say so, rather than blame a
    /// "missing" argument that was in fact supplied.
    #[test]
    fn a_required_argument_of_the_wrong_type_names_the_type_error() {
        assert_eq!(
            route_tool("describe", &json!({"context": 7, "concept": "x"})),
            Err("argument 'context' must be a string".to_string())
        );
    }

    /// When both context and its payload are missing, the context — the
    /// path segment resolved first — is the one reported, so the caller
    /// fixes the outer error before the inner one.
    #[test]
    fn add_associations_reports_the_missing_context_before_the_missing_payload() {
        assert_eq!(
            route_tool("add_associations", &json!({})),
            Err("missing required argument 'context'".to_string())
        );
    }

    /// Beyond `add_associations` (covered above), every other tool whose
    /// schema marks a body argument required must refuse routing when
    /// that argument is omitted instead of composing a request with the
    /// key silently absent — `pick` alone would drop it without a word,
    /// pushing the caller's mistake past this layer and into a slower,
    /// vaguer failure downstream.
    #[test]
    fn schema_required_body_arguments_are_refused_when_omitted() {
        let base = json!({
            "name": "ctx", "context": "ctx", "cue": "x", "concept": "x",
            "origins": ["x"], "passages": {}, "sources": ["s"], "source": "s",
            "query": "q", "paragraph": 0, "to": "ctx2", "expected": "x",
            "subject": "s", "label": "l", "object": "o",
        });
        let cases = [
            ("rename_context", "to"),
            ("rename_group", "to"),
            ("store_passages", "passages"),
            ("lookup_passages", "sources"),
            ("resolve", "cue"),
            ("resolve_label", "cue"),
            ("explain_resolve", "cue"),
            ("explain_resolve", "expected"),
            ("explain_resolve_label", "cue"),
            ("explain_resolve_label", "expected"),
            ("describe", "concept"),
            ("recall", "cue"),
            ("activate", "origins"),
            ("explore", "origins"),
            ("retract_source", "source"),
            ("retract_association", "subject"),
            ("retract_association", "label"),
            ("retract_association", "object"),
            ("search_passages", "query"),
            ("explain_search", "query"),
            ("explain_search", "source"),
            ("cite_passage", "source"),
            ("audit_coverage", "origins"),
        ];
        for (tool, key) in cases {
            let mut arguments = base.clone();
            arguments[key] = Value::Null;
            let routed = route_tool(tool, &arguments);
            assert!(
                routed.is_err(),
                "tool '{tool}' should refuse a missing '{key}', got {routed:?}"
            );
            let err = routed.unwrap_err();
            assert!(
                err.contains(key),
                "tool '{tool}' missing '{key}' error should name it, got: {err}"
            );
        }
    }

    /// `cite_passage` accepts either `paragraph` or its deprecated alias
    /// `index` (positive cases covered above); omitting both must refuse
    /// rather than route a citation request with neither name present.
    #[test]
    fn cite_passage_without_paragraph_or_index_is_refused() {
        let routed = route_tool(
            "cite_passage",
            &json!({"context": "sake", "source": "docs/aomine.md"}),
        );
        assert_eq!(
            routed,
            Err(
                "missing required argument 'paragraph' (or its deprecated alias 'index')"
                    .to_string()
            )
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

    #[test]
    fn query_string_encodes_present_keys_and_skips_absent_or_null_ones() {
        let arguments = json!({"limit": 50, "after": null, "extra": "x"});
        assert_eq!(
            query_string(&arguments, &["limit", "after", "absent"]),
            "?limit=50"
        );
    }

    #[test]
    fn query_string_percent_encodes_string_values() {
        let arguments = json!({"after": "日本 語"});
        assert_eq!(
            query_string(&arguments, &["after"]),
            "?after=%E6%97%A5%E6%9C%AC%20%E8%AA%9E"
        );
    }

    #[test]
    fn query_string_is_empty_when_no_keys_are_present() {
        assert_eq!(query_string(&json!({}), &["limit", "after"]), "");
    }

    /// `create_context`/`update_context` advertise `pinned: boolean`,
    /// and item 6 (#62) added `pinned`/`prefix` boolean/string filters
    /// to list tools — a bool argument must not silently vanish here.
    #[test]
    fn query_string_encodes_bool_values() {
        let arguments = json!({"pinned": true});
        assert_eq!(query_string(&arguments, &["pinned"]), "?pinned=true");
        let arguments = json!({"pinned": false});
        assert_eq!(query_string(&arguments, &["pinned"]), "?pinned=false");
    }

    /// list_contexts advertises limit/after and, when the caller
    /// supplies them, routes them onto the GET request's query string
    /// — the wiring the issue tracked was missing entirely.
    #[test]
    fn list_contexts_schema_advertises_limit_and_after() {
        let list_contexts = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "list_contexts")
            .expect("list_contexts is defined");
        let properties = &list_contexts["inputSchema"]["properties"];
        assert_eq!(properties["limit"]["type"], "integer");
        assert_eq!(properties["after"]["type"], "string");
    }

    #[test]
    fn list_contexts_routes_limit_and_after_onto_the_query_string() {
        let (method, path, body) =
            route_tool("list_contexts", &json!({"limit": 50, "after": "sake"})).unwrap();
        assert_eq!(method, "GET");
        assert_eq!(path, "/contexts?limit=50&after=sake");
        assert_eq!(body, None);
    }

    #[test]
    fn list_contexts_without_arguments_has_no_query_string() {
        let (_, path, _) = route_tool("list_contexts", &json!({})).unwrap();
        assert_eq!(path, "/contexts");
    }

    /// #62 item 6: `pinned` filters the directory (population, not a
    /// cursor) — advertised in the schema and routed onto the query
    /// string like `limit`/`after`.
    #[test]
    fn list_contexts_schema_advertises_pinned() {
        let list_contexts = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "list_contexts")
            .expect("list_contexts is defined");
        let properties = &list_contexts["inputSchema"]["properties"];
        assert_eq!(properties["pinned"]["type"], "boolean");
    }

    #[test]
    fn list_contexts_routes_pinned_onto_the_query_string() {
        let (_, path, _) = route_tool("list_contexts", &json!({"pinned": true})).unwrap();
        assert_eq!(path, "/contexts?pinned=true");
    }

    /// #62 item 6: `list_sources`/`list_labels`/`get_aliases` advertise
    /// and route `prefix` the same way — narrows the population, so it
    /// belongs beside `limit`/`after` in both schema and query string.
    #[test]
    fn list_sources_schema_advertises_prefix() {
        let list_sources = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "list_sources")
            .expect("list_sources is defined");
        let properties = &list_sources["inputSchema"]["properties"];
        assert_eq!(properties["prefix"]["type"], "string");
    }

    #[test]
    fn list_sources_routes_prefix_onto_the_query_string() {
        let (_, path, _) = route_tool(
            "list_sources",
            &json!({"context": "sake", "prefix": "doc-"}),
        )
        .unwrap();
        assert_eq!(path, "/contexts/sake/sources?prefix=doc-");
    }

    #[test]
    fn list_labels_schema_advertises_prefix() {
        let list_labels = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "list_labels")
            .expect("list_labels is defined");
        let properties = &list_labels["inputSchema"]["properties"];
        assert_eq!(properties["prefix"]["type"], "string");
    }

    #[test]
    fn list_labels_routes_prefix_onto_the_query_string() {
        let (_, path, _) =
            route_tool("list_labels", &json!({"context": "sake", "prefix": "産地"})).unwrap();
        assert_eq!(path, "/contexts/sake/labels?prefix=%E7%94%A3%E5%9C%B0");
    }

    #[test]
    fn get_aliases_schema_advertises_prefix() {
        let get_aliases = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "get_aliases")
            .expect("get_aliases is defined");
        let properties = &get_aliases["inputSchema"]["properties"];
        assert_eq!(properties["prefix"]["type"], "string");
    }

    #[test]
    fn get_aliases_routes_prefix_onto_the_query_string() {
        let (_, path, _) =
            route_tool("get_aliases", &json!({"context": "sake", "prefix": "a"})).unwrap();
        assert_eq!(path, "/contexts/sake/aliases?prefix=a");
    }

    /// #39: the schema had no `limit` and `route_tool` whitelisted only
    /// `origins`, so there was no way to raise the cap through this tool
    /// at all.
    #[test]
    fn audit_coverage_schema_advertises_limit() {
        let audit_coverage = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "audit_coverage")
            .expect("audit_coverage is defined");
        let properties = &audit_coverage["inputSchema"]["properties"];
        assert_eq!(properties["limit"]["type"], "integer");
    }

    #[test]
    fn audit_coverage_routes_limit_into_the_request_body() {
        let (method, path, body) = route_tool(
            "audit_coverage",
            &json!({"context": "sake", "origins": ["x"], "limit": 500}),
        )
        .unwrap();
        assert_eq!(method, "POST");
        assert_eq!(path, "/contexts/sake/unreachable_from");
        assert_eq!(body, Some(json!({"origins": ["x"], "limit": 500})));
    }

    #[test]
    fn audit_drift_schema_advertises_limit() {
        let audit_drift = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "audit_drift")
            .expect("audit_drift is defined");
        let properties = &audit_drift["inputSchema"]["properties"];
        assert_eq!(properties["limit"]["type"], "integer");
    }

    #[test]
    fn audit_drift_routes_every_option_into_the_request_body() {
        let (method, path, body) = route_tool(
            "audit_drift",
            &json!({
                "context": "sake",
                "unsourced_floor": 0.5,
                "limit": 25,
                "include_twins": true,
                "dice_floor": 0.7,
                "cosine_floor": 0.8
            }),
        )
        .unwrap();
        assert_eq!(method, "POST");
        assert_eq!(path, "/contexts/sake/drift/audit");
        assert_eq!(
            body,
            Some(json!({
                "unsourced_floor": 0.5,
                "limit": 25,
                "include_twins": true,
                "dice_floor": 0.7,
                "cosine_floor": 0.8
            }))
        );
    }

    /// #60: query/recall/explore/audit_coverage advertise `after` for
    /// resuming a page past its predecessor's last row — the MCP-layer
    /// wiring for the keyset cursors the HTTP side already accepts.
    /// Every field a client must copy verbatim is declared `required`,
    /// so a caller builds a whole cursor or none — not a partial one
    /// the downstream Rust struct would reject anyway.
    #[test]
    fn search_and_audit_tools_advertise_after() {
        let cases: [(&str, &[&str]); 5] = [
            ("query", &["weight", "subject", "label", "object"]),
            ("recall", &["weight", "subject", "label", "object"]),
            ("explore", &["distance", "subject", "label", "object"]),
            ("audit_coverage", &["weight", "subject", "label", "object"]),
            ("audit_drift", &["weight", "subject", "label", "object"]),
        ];
        for (name, required) in cases {
            let tool = tool_definitions()
                .into_iter()
                .find(|tool| tool["name"] == name)
                .unwrap_or_else(|| panic!("{name} is defined"));
            let after = &tool["inputSchema"]["properties"]["after"];
            assert_eq!(after["type"], "object", "tool '{name}' after");
            for field in required {
                assert!(
                    after["properties"].get(*field).is_some(),
                    "tool '{name}' after.properties.{field}"
                );
            }
            let actual_required: Vec<&str> = after["required"]
                .as_array()
                .unwrap_or_else(|| panic!("tool '{name}' after.required is an array"))
                .iter()
                .map(|value| value.as_str().unwrap())
                .collect();
            assert_eq!(actual_required, required, "tool '{name}' after.required");
        }
    }

    /// `after` rides straight through to the request body, whatever
    /// shape the caller sent — single-context `MatchCursor`,
    /// cross-context `CrossMatchCursor` (an extra `context` field), or
    /// explore's own `{distance, subject, label, object}`. `pick`
    /// forwards it verbatim; the downstream Rust struct is what
    /// actually validates the shape.
    #[test]
    fn search_and_audit_tools_route_after_into_the_request_body() {
        let cursor = json!({"weight": 0.5, "subject": "a", "label": "b", "object": "c"});
        let (_, _, body) = route_tool(
            "recall",
            &json!({"context": "sake", "cue": "x", "after": cursor}),
        )
        .unwrap();
        assert_eq!(body.unwrap()["after"], cursor);

        let cross_cursor = json!({
            "weight": 0.5, "context": "sake", "subject": "a", "label": "b", "object": "c"
        });
        let (_, _, body) = route_tool(
            "query",
            &json!({"contexts": ["sake"], "subject": "s", "after": cross_cursor}),
        )
        .unwrap();
        assert_eq!(body.unwrap()["after"], cross_cursor);

        let explore_cursor = json!({"distance": 2, "subject": "a", "label": "b", "object": "c"});
        let (_, _, body) = route_tool(
            "explore",
            &json!({"context": "sake", "origins": ["a"], "after": explore_cursor}),
        )
        .unwrap();
        assert_eq!(body.unwrap()["after"], explore_cursor);

        let (_, _, body) = route_tool(
            "audit_coverage",
            &json!({"context": "sake", "origins": ["a"], "after": cursor}),
        )
        .unwrap();
        assert_eq!(body.unwrap()["after"], cursor);

        let (_, _, body) =
            route_tool("audit_drift", &json!({"context": "sake", "after": cursor})).unwrap();
        assert_eq!(body.unwrap()["after"], cursor);
    }

    #[test]
    fn pick_with_alias_falls_back_to_the_old_key_name() {
        let arguments = json!({"source": "s", "index": 3});
        assert_eq!(
            pick_with_alias(&arguments, &["source", "paragraph"], "paragraph", "index"),
            json!({"source": "s", "paragraph": 3})
        );
    }

    #[test]
    fn pick_with_alias_prefers_the_canonical_key_when_both_are_present() {
        let arguments = json!({"source": "s", "paragraph": 1, "index": 99});
        assert_eq!(
            pick_with_alias(&arguments, &["source", "paragraph"], "paragraph", "index"),
            json!({"source": "s", "paragraph": 1})
        );
    }

    #[test]
    fn cite_passage_routes_to_the_citations_endpoint() {
        let (method, path, body) = route_tool(
            "cite_passage",
            &json!({"context": "sake", "source": "docs/aomine.md", "paragraph": 1}),
        )
        .unwrap();
        assert_eq!(method, "POST");
        assert_eq!(path, "/contexts/sake/citations");
        assert_eq!(
            body,
            Some(json!({"source": "docs/aomine.md", "paragraph": 1}))
        );
    }

    /// The three explain mirrors route beside their parents: per-context
    /// POSTs, the addressing key peeled off, every override passed
    /// through — including `expected`, which no parent tool carries.
    #[test]
    fn explain_tools_route_beside_their_parents() {
        let (method, path, body) = route_tool(
            "explain_search",
            &json!({"context": "sake", "query": "酒造", "source": "docs/kura.md",
                    "paragraph": 1, "limit": 5}),
        )
        .unwrap();
        assert_eq!(method, "POST");
        assert_eq!(path, "/contexts/sake/sources/search/explain");
        assert_eq!(
            body,
            Some(json!({"query": "酒造", "source": "docs/kura.md", "paragraph": 1, "limit": 5}))
        );

        let (method, path, body) = route_tool(
            "explain_resolve",
            &json!({"context": "sake", "cue": "青嶺", "expected": "青嶺酒造", "dice_floor": 0.2}),
        )
        .unwrap();
        assert_eq!(method, "POST");
        assert_eq!(path, "/contexts/sake/resolve/explain");
        assert_eq!(
            body,
            Some(json!({"cue": "青嶺", "expected": "青嶺酒造", "dice_floor": 0.2}))
        );

        let (method, path, body) = route_tool(
            "explain_resolve_label",
            &json!({"context": "sake", "cue": "醸す", "expected": "杜氏"}),
        )
        .unwrap();
        assert_eq!(method, "POST");
        assert_eq!(path, "/contexts/sake/resolve_label/explain");
        assert_eq!(body, Some(json!({"cue": "醸す", "expected": "杜氏"})));
    }

    /// MCP clients written against the pre-#35 argument name still work:
    /// `pick` alone would silently drop `index` since it only whitelists
    /// `paragraph`, so this exercises the alias fallback end to end.
    #[test]
    fn cite_passage_accepts_the_pre_35_index_argument_name() {
        let (_, _, body) = route_tool(
            "cite_passage",
            &json!({"context": "sake", "source": "docs/aomine.md", "index": 1}),
        )
        .unwrap();
        assert_eq!(
            body,
            Some(json!({"source": "docs/aomine.md", "paragraph": 1}))
        );
    }

    /// The advertised contract matches what `route_tool` actually accepts:
    /// `index` is a documented deprecated alias, and the schema requires
    /// one of `paragraph`/`index` rather than unconditionally demanding
    /// `paragraph`.
    #[test]
    fn cite_passage_schema_advertises_index_as_a_deprecated_alias() {
        let tool = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "cite_passage")
            .expect("cite_passage is defined");
        let schema = &tool["inputSchema"];
        assert!(
            schema["properties"]["index"]["type"] == "integer",
            "schema should advertise `index` as an integer: {schema}"
        );
        assert_eq!(schema["required"], json!(["context", "source"]));
        assert_eq!(
            schema["anyOf"],
            json!([{ "required": ["paragraph"] }, { "required": ["index"] }])
        );
    }

    /// `query`'s description says subject/label/object need at least
    /// one; the schema must say so too, on top of (not instead of) the
    /// target-selection `anyOf` `search_target_schema` already adds.
    #[test]
    fn query_schema_requires_a_position_alongside_a_target() {
        let tool = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "query")
            .expect("query is defined");
        let schema = &tool["inputSchema"];
        assert!(
            schema.get("anyOf").is_none(),
            "the target-selection anyOf should move under allOf, not stay alongside it: {schema}"
        );
        assert_eq!(
            schema["allOf"],
            json!([
                {
                    "anyOf": [
                        { "required": ["context"] },
                        { "required": ["contexts"] },
                        { "required": ["groups"] },
                    ]
                },
                {
                    "anyOf": [
                        { "required": ["subject"] },
                        { "required": ["label"] },
                        { "required": ["object"] },
                    ]
                },
            ])
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
        // The id survives even though the method is what's missing — the
        // sender is still waiting on a reply it can correlate, not a null.
        assert!(matches!(
            classify(&json!({"jsonrpc": "2.0", "id": 1})),
            Message::Undecodable { id } if id == json!(1)
        ));
        assert!(matches!(
            classify(&json!({"jsonrpc": "2.0", "id": 1, "method": "resources/list"})),
            Message::Request {
                call: Call::Unknown { .. },
                ..
            }
        ));
    }

    /// The one notification method a transport needs to look inside —
    /// everything else stays opaque behind `Message::Notification`.
    #[test]
    fn cancelled_request_id_reads_only_its_own_notification() {
        assert_eq!(
            cancelled_request_id(&json!({
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": { "requestId": 7 },
            })),
            Some(json!(7))
        );
        assert_eq!(
            cancelled_request_id(&json!({
                "jsonrpc": "2.0",
                "method": "notifications/cancelled",
                "params": { "requestId": "abc" },
            })),
            Some(json!("abc"))
        );
        // A different notification, a request, and a malformed
        // cancellation (no params, no requestId) all read as None.
        assert_eq!(
            cancelled_request_id(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"})),
            None
        );
        assert_eq!(
            cancelled_request_id(&json!({"jsonrpc": "2.0", "id": 1, "method": "ping"})),
            None
        );
        assert_eq!(
            cancelled_request_id(&json!({"jsonrpc": "2.0", "method": "notifications/cancelled"})),
            None
        );
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

    /// A version this build was never written against — a future spec
    /// revision, or a client just making one up — falls back instead
    /// of being echoed back as if the two sides had agreed to it.
    #[test]
    fn initialize_result_falls_back_on_an_unrecognized_client_version() {
        let unrecognized = initialize_result(Some("2099-01-01"), "manual");
        assert_eq!(unrecognized["protocolVersion"], FALLBACK_PROTOCOL_VERSION);

        let garbage = initialize_result(Some("not-a-version"), "manual");
        assert_eq!(garbage["protocolVersion"], FALLBACK_PROTOCOL_VERSION);
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

    #[test]
    fn import_schema_advertises_stream_and_dry_run() {
        let import = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "import")
            .expect("import is defined");
        let properties = &import["inputSchema"]["properties"];
        assert_eq!(properties["stream"]["type"], "string");
        assert_eq!(properties["dry_run"]["type"], "boolean");
        assert_eq!(import["inputSchema"]["required"], json!(["stream"]));
    }

    /// The stream rides the body as a raw string, not the `pick`/JSON
    /// shape every other write tool uses — `call_inner`/`Bridge::call`
    /// special-case `Value::String` so this reaches `import_batch` as
    /// literal NDJSON text, newlines intact, not `\n`-escaped inside a
    /// quoted JSON string.
    #[test]
    fn import_routes_the_stream_as_a_raw_string_body() {
        let stream = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s\"}\n";
        let (method, path, body) = route_tool("import", &json!({"stream": stream})).unwrap();
        assert_eq!(method, "POST");
        assert_eq!(path, "/import");
        assert_eq!(body, Some(Value::String(stream.to_string())));
    }

    #[test]
    fn import_routes_dry_run_onto_the_query_string() {
        let (_, path, _) = route_tool("import", &json!({"stream": "x", "dry_run": true})).unwrap();
        assert_eq!(path, "/import?dry_run=true");

        let (_, path, _) = route_tool("import", &json!({"stream": "x"})).unwrap();
        assert_eq!(
            path, "/import",
            "dry_run absent means no query string at all"
        );
    }

    #[test]
    fn import_refuses_a_stream_over_the_byte_limit() {
        let oversized = "x".repeat(MAX_IMPORT_STREAM_BYTES + 1);
        let routed = route_tool("import", &json!({"stream": oversized}));
        assert!(
            routed.is_err(),
            "a stream past the tool's own byte cap must not route"
        );
    }

    /// A minimal HTTP-response envelope for the given `route_tool` path
    /// suffix, mirroring `ApiResponse<T>`'s `{result, status, time}`
    /// shape — `run_retrieve` decodes exactly that.
    fn envelope(result: Value) -> String {
        json!({ "result": result, "status": "ok", "time": 0.0 }).to_string()
    }

    #[test]
    fn run_retrieve_resolves_describes_activates_and_cites_in_one_call() {
        let arguments = json!({ "context": "sake", "origins": ["tokyo"] });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(
                    json!([{"name": "Tokyo", "score": 1.0, "tier": "exact"}]),
                ))
            } else if path.ends_with("/describe") {
                Ok(envelope(
                    json!({"concept": "Tokyo", "as_subject": [], "as_object": []}),
                ))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({
                    "total": 1,
                    "matches": [{
                        "strength": 1.0,
                        "path": ["Tokyo"],
                        "association": {
                            "subject": "Tokyo", "label": "capital_of", "object": "Japan",
                            "weight": 1.0, "count": 1,
                            "attributions": [
                                {"source": "doc1", "weight": 1.0, "count": 1, "paragraph": 0}
                            ]
                        }
                    }]
                })))
            } else if path.ends_with("/citations") {
                Ok(envelope(
                    json!({"text": "Tokyo is the capital.", "source": "doc1", "section": null}),
                ))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");

        assert_eq!(result["resolved"]["tokyo"][0]["name"], "Tokyo");
        assert_eq!(result["outline"]["Tokyo"]["concept"], "Tokyo");
        assert_eq!(result["associations"].as_array().unwrap().len(), 1);
        assert_eq!(result["associations"][0]["subject"], "Tokyo");
        assert_eq!(result["activations"].as_array().unwrap().len(), 1);
        assert_eq!(
            result["citations"],
            json!([{
                "source": "doc1",
                "paragraph": 0,
                "citation": {"text": "Tokyo is the capital.", "source": "doc1", "section": null},
            }])
        );
        assert_eq!(result["passage_hits"], json!([]));
    }

    /// A budget that the first citation round trip alone pushes past
    /// must stop the loop right there — the second and third citations
    /// (same association, three attributions) must never dispatch, not
    /// merely have their result discarded once the whole call finishes.
    #[test]
    fn run_retrieve_bounded_stops_dispatching_citations_once_the_budget_is_spent() {
        let resolve_body = envelope(json!([{"name": "Tokyo"}]));
        let association = json!({
            "subject": "Tokyo", "label": "capital_of", "object": "Japan",
            "weight": 1.0, "count": 1,
            "attributions": [
                {"source": "doc1", "weight": 1.0, "count": 1, "paragraph": 0},
                {"source": "doc2", "weight": 1.0, "count": 1, "paragraph": 0},
                {"source": "doc3", "weight": 1.0, "count": 1, "paragraph": 0},
            ]
        });
        let activate_body = envelope(json!({
            "total": 1,
            "matches": [{"strength": 1.0, "path": ["Tokyo"], "association": association}]
        }));
        let citation_body = envelope(json!({"text": "x", "source": "doc", "section": null}));
        // One byte short of resolve + activate + one citation: the first
        // citation's response is what tips the scale, not a fourth call.
        let budget = resolve_body.len() + activate_body.len() + citation_body.len() - 1;

        let arguments = json!({ "context": "sake", "origins": ["tokyo"], "describe_first": false });
        let mut citation_calls = 0usize;
        let result = run_retrieve_bounded(&arguments, Some(budget), |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(resolve_body.clone())
            } else if path.ends_with("/activate") {
                Ok(activate_body.clone())
            } else if path.ends_with("/citations") {
                citation_calls += 1;
                Ok(citation_body.clone())
            } else {
                panic!("unexpected call: {path}");
            }
        });

        assert!(
            matches!(&result, Err(message) if message.contains(&format!("already exceeds {budget} bytes"))
                && message.contains("cite_passage")),
            "{result:?}"
        );
        assert_eq!(
            citation_calls, 1,
            "the budget must be spent inside the first citation call, before a second ever fires"
        );
    }

    /// `run_retrieve` (what the uncapped stdio bridge calls) is just
    /// `run_retrieve_bounded` with no budget — a budget so tight even
    /// the first call would trip it must still let a `None` budget
    /// through untouched.
    #[test]
    fn run_retrieve_passes_no_budget_to_run_retrieve_bounded() {
        let arguments = json!({ "context": "sake", "origins": ["tokyo"], "describe_first": false });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{"name": "Tokyo"}])))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({"total": 0, "matches": []})))
            } else {
                panic!("unexpected call: {path}");
            }
        });
        assert!(result.is_ok(), "{result:?}");
    }

    /// query (only run when `labels` is given) and activate can surface
    /// the same edge; the triple-keyed dedupe must collapse them to one
    /// entry, keeping query's copy since it is gathered first.
    #[test]
    fn run_retrieve_dedupes_associations_across_query_and_activate() {
        let arguments = json!({
            "context": "sake", "origins": ["tokyo"], "labels": ["capital_of"],
            "describe_first": false, "fetch_citations": false
        });
        let association = json!({
            "subject": "Tokyo", "label": "capital_of", "object": "Japan",
            "weight": 1.0, "count": 1, "attributions": []
        });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{"name": "Tokyo"}])))
            } else if path.ends_with("/query") {
                Ok(envelope(json!({"total": 1, "matches": [association]})))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({
                    "total": 1,
                    "matches": [{"strength": 1.0, "path": ["Tokyo"], "association": association}]
                })))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");

        assert_eq!(
            result["associations"].as_array().unwrap().len(),
            1,
            "{result}"
        );
        assert_eq!(result["activations"].as_array().unwrap().len(), 1);
    }

    /// `triple_of`'s doc comment says a value it can't parse into
    /// `(subject, label, object)` means "keep it, nothing to dedupe
    /// against" — not "drop it". A malformed `query` match and a
    /// malformed `activate` association (both missing `label`) must
    /// both still land in the final `associations` list.
    #[test]
    fn run_retrieve_keeps_an_association_triple_of_cannot_parse() {
        let arguments = json!({
            "context": "sake", "origins": ["tokyo"], "labels": ["capital_of"],
            "describe_first": false, "fetch_citations": false
        });
        let malformed_from_query = json!({
            "subject": "Tokyo", "object": "Japan", "weight": 1.0, "count": 1, "attributions": []
        });
        let malformed_from_activate = json!({
            "subject": "Osaka", "object": "Japan", "weight": 1.0, "count": 1, "attributions": []
        });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{"name": "Tokyo"}])))
            } else if path.ends_with("/query") {
                Ok(envelope(
                    json!({"total": 1, "matches": [malformed_from_query]}),
                ))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({
                    "total": 1,
                    "matches": [{
                        "strength": 1.0, "path": ["Tokyo"], "association": malformed_from_activate
                    }]
                })))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");

        assert_eq!(
            result["associations"].as_array().unwrap().len(),
            2,
            "an association triple_of cannot parse must be kept, not dropped: {result}"
        );
    }

    /// A citation attribution pointing at a passage that was never
    /// stored (or was retracted) comes back 404 from `cite_passage` —
    /// that one locator is skipped, not the whole retrieval.
    #[test]
    fn run_retrieve_skips_a_404_citation_without_failing() {
        let arguments = json!({ "context": "sake", "origins": ["tokyo"], "describe_first": false });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{"name": "Tokyo"}])))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({
                    "total": 1,
                    "matches": [{
                        "strength": 1.0,
                        "path": ["Tokyo"],
                        "association": {
                            "subject": "Tokyo", "label": "capital_of", "object": "Japan",
                            "weight": 1.0, "count": 1,
                            "attributions": [
                                {"source": "doc1", "weight": 1.0, "count": 1, "paragraph": 0}
                            ]
                        }
                    }]
                })))
            } else if path.ends_with("/citations") {
                Err("HTTP 404: {\"error\":\"no such paragraph\"}".to_string())
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("a 404 citation must not fail the whole retrieval");

        assert_eq!(result["citations"], json!([]));
    }

    /// Any citation failure other than 404 (auth, a downed server) must
    /// abort the whole call rather than being swallowed like the 404
    /// case above.
    #[test]
    fn run_retrieve_fails_outright_on_a_non_404_citation_error() {
        let arguments = json!({ "context": "sake", "origins": ["tokyo"], "describe_first": false });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{"name": "Tokyo"}])))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({
                    "total": 1,
                    "matches": [{
                        "strength": 1.0,
                        "path": ["Tokyo"],
                        "association": {
                            "subject": "Tokyo", "label": "capital_of", "object": "Japan",
                            "weight": 1.0, "count": 1,
                            "attributions": [
                                {"source": "doc1", "weight": 1.0, "count": 1, "paragraph": 0}
                            ]
                        }
                    }]
                })))
            } else if path.ends_with("/citations") {
                Err("HTTP 500: internal error".to_string())
            } else {
                panic!("unexpected call: {path}");
            }
        });
        assert_eq!(result, Err("HTTP 500: internal error".to_string()));
    }

    /// `auto_pick: false` anchors on each cue verbatim instead of
    /// resolve's top candidate — resolve still runs (so `resolved`
    /// still reports what was found), but an empty result must not
    /// empty out the anchor list too.
    #[test]
    fn run_retrieve_with_auto_pick_off_anchors_on_the_cue_itself() {
        let arguments = json!({
            "context": "sake", "origins": ["Tokyo"], "auto_pick": false,
            "describe_first": false, "fetch_citations": false
        });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([])))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({"total": 0, "matches": []})))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");

        assert_eq!(result["resolved"]["Tokyo"], json!([]));
        assert_eq!(result["associations"], json!([]));
    }

    /// The text-lane fallback only fires when a fallback query was
    /// given AND (by default) associations came back empty — here
    /// resolve finds nothing, so anchors stay empty, activate never
    /// runs, and search_passages is the only other call.
    #[test]
    fn run_retrieve_runs_the_text_fallback_when_associations_are_empty() {
        let arguments = json!({
            "context": "sake", "origins": ["nonexistent"],
            "text_fallback_query": "some declarative fact"
        });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([])))
            } else if path.ends_with("/sources/search") {
                Ok(envelope(json!([
                    {"source": "doc1", "paragraph": 0, "score": 0.9, "text": "...", "lanes": {}}
                ])))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");

        assert_eq!(result["passage_hits"].as_array().unwrap().len(), 1);
    }

    /// `text_fallback_only_if_empty: false` runs the fallback
    /// unconditionally, even alongside associations already found.
    #[test]
    fn run_retrieve_runs_the_text_fallback_unconditionally_when_the_empty_gate_is_off() {
        let arguments = json!({
            "context": "sake", "origins": ["tokyo"], "describe_first": false,
            "fetch_citations": false, "text_fallback_query": "some declarative fact",
            "text_fallback_only_if_empty": false
        });
        let result = run_retrieve(&arguments, |_method, path, _body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{"name": "Tokyo"}])))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({
                    "total": 1,
                    "matches": [{
                        "strength": 1.0,
                        "path": ["Tokyo"],
                        "association": {
                            "subject": "Tokyo", "label": "capital_of", "object": "Japan",
                            "weight": 1.0, "count": 1, "attributions": []
                        }
                    }]
                })))
            } else if path.ends_with("/sources/search") {
                Ok(envelope(json!([
                    {"source": "doc1", "paragraph": 0, "score": 0.9, "text": "...", "lanes": {}}
                ])))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");

        assert_eq!(result["associations"].as_array().unwrap().len(), 1);
        assert_eq!(result["passage_hits"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn run_retrieve_requires_context_and_origins() {
        assert_eq!(
            run_retrieve(&json!({"origins": ["x"]}), |_, _, _| unreachable!()),
            Err("missing required argument 'context'".to_string())
        );
        assert_eq!(
            run_retrieve(&json!({"context": "sake"}), |_, _, _| unreachable!()),
            Err("missing required argument 'origins'".to_string())
        );
    }

    #[test]
    fn run_retrieve_refuses_an_origins_list_past_the_input_cap() {
        // One resolve round trip per cue makes an oversized list a fanout
        // amplifier, so it is refused before the first call fires — the
        // way the direct endpoints refuse an overlong `origins` batch.
        let origins: Vec<String> = (0..=MAX_ORIGIN_CUES).map(|i| format!("cue{i}")).collect();
        let result = run_retrieve(
            &json!({ "context": "sake", "origins": origins }),
            |_, _, _| unreachable!("no request may fire once the list is refused"),
        );
        assert!(
            matches!(&result, Err(message) if message.contains("past the per-request limit")),
            "{result:?}"
        );
    }

    #[test]
    fn run_retrieve_admits_an_origins_list_at_exactly_the_input_cap() {
        // The cap refuses lists *past* the ceiling, not at it: a list of
        // exactly MAX_ORIGIN_CUES cues clears the guard and reaches the first
        // resolve round trip. Pins the `>` boundary so a `>=` slip — which
        // would refuse the largest admissible list — cannot pass unnoticed.
        let origins: Vec<String> = (0..MAX_ORIGIN_CUES).map(|i| format!("cue{i}")).collect();
        assert_eq!(origins.len(), MAX_ORIGIN_CUES);
        let mut calls = 0usize;
        let result = run_retrieve(
            &json!({ "context": "sake", "origins": origins }),
            |_, path, _| {
                calls += 1;
                assert!(
                    path.ends_with("/resolve"),
                    "first round trip is a resolve: {path}"
                );
                Err("stop past the guard".to_string())
            },
        );
        assert_eq!(
            calls, 1,
            "the admitted list fired exactly one resolve before we bailed"
        );
        assert_eq!(result, Err("stop past the guard".to_string()));
    }

    /// `resolve_limit` rides every resolve round trip's body: a caller's
    /// candidate cap must reach the resolve endpoint, not be dropped
    /// between the composed call and the per-cue request. The non-null
    /// gate that admits it is what makes a supplied cap take effect;
    /// resolve returns nothing, so no anchor forms and it is the only call.
    #[test]
    fn run_retrieve_forwards_resolve_limit_to_each_resolve_call() {
        let arguments = json!({ "context": "sake", "origins": ["tokyo"], "resolve_limit": 7 });
        let mut saw_resolve = false;
        run_retrieve(&arguments, |_method, path, body| {
            assert!(
                path.ends_with("/resolve"),
                "resolve is the only call: {path}"
            );
            let body = body.expect("resolve carries a body");
            assert_eq!(
                body["limit"], 7,
                "resolve_limit must ride the resolve body: {body}"
            );
            saw_resolve = true;
            Ok(envelope(json!([])))
        })
        .expect("run_retrieve succeeds");
        assert!(saw_resolve, "resolve must have fired");
    }

    /// `labels` both gates and rides the query round trip: naming facets
    /// must fire a `query` whose body carries them. Were the non-null gate
    /// to invert, a named facet set would silently skip query altogether.
    #[test]
    fn run_retrieve_forwards_labels_to_the_query_round_trip() {
        let arguments = json!({
            "context": "sake", "origins": ["tokyo"], "labels": ["capital_of"],
            "describe_first": false, "fetch_citations": false
        });
        let mut saw_query = false;
        run_retrieve(&arguments, |_method, path, body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{ "name": "Tokyo" }])))
            } else if path.ends_with("/query") {
                let body = body.expect("query carries a body");
                assert_eq!(
                    body["label"],
                    json!(["capital_of"]),
                    "labels must ride the query body: {body}"
                );
                saw_query = true;
                Ok(envelope(json!({ "matches": [] })))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({ "matches": [] })))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");
        assert!(saw_query, "labels must trigger a query round trip");
    }

    /// `activate_decay` and `activate_limit` both ride the activate body:
    /// the spreading-activation knobs must reach the activate endpoint
    /// rather than being dropped between the composed call and the request.
    #[test]
    fn run_retrieve_forwards_activate_decay_and_limit_to_the_activate_call() {
        let arguments = json!({
            "context": "sake", "origins": ["tokyo"],
            "activate_decay": 0.5, "activate_limit": 9,
            "describe_first": false, "fetch_citations": false
        });
        let mut saw_activate = false;
        run_retrieve(&arguments, |_method, path, body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{ "name": "Tokyo" }])))
            } else if path.ends_with("/activate") {
                let body = body.expect("activate carries a body");
                assert_eq!(
                    body["decay"], 0.5,
                    "activate_decay must ride the activate body: {body}"
                );
                assert_eq!(
                    body["limit"], 9,
                    "activate_limit must ride the activate body: {body}"
                );
                saw_activate = true;
                Ok(envelope(json!({ "matches": [] })))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");
        assert!(saw_activate, "activate must have fired");
    }

    /// `search_limit` rides the text-fallback search body, capping the
    /// fallback page as the caller asked. resolve anchors but activate
    /// returns nothing, so associations stay empty and the fallback fires.
    #[test]
    fn run_retrieve_forwards_search_limit_to_the_text_fallback() {
        let arguments = json!({
            "context": "sake", "origins": ["tokyo"], "describe_first": false,
            "fetch_citations": false, "text_fallback_query": "some declarative fact",
            "search_limit": 4
        });
        let mut saw_search = false;
        run_retrieve(&arguments, |_method, path, body| {
            if path.ends_with("/resolve") {
                Ok(envelope(json!([{ "name": "Tokyo" }])))
            } else if path.ends_with("/activate") {
                Ok(envelope(json!({ "matches": [] })))
            } else if path.ends_with("/sources/search") {
                let body = body.expect("search carries a body");
                assert_eq!(
                    body["limit"], 4,
                    "search_limit must ride the search body: {body}"
                );
                saw_search = true;
                Ok(envelope(json!([])))
            } else {
                panic!("unexpected call: {path}");
            }
        })
        .expect("run_retrieve succeeds");
        assert!(saw_search, "the text fallback must have fired a search");
    }

    #[test]
    fn retrieve_is_advertised_with_context_and_origins_required() {
        let tool = tool_definitions()
            .into_iter()
            .find(|tool| tool["name"] == "retrieve")
            .expect("retrieve is defined");
        assert_eq!(
            tool["inputSchema"]["required"],
            json!(["context", "origins"])
        );
    }
}
