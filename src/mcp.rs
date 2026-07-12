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
/// [`Message::Undecodable`], which both transports answer with a
/// JSON-RPC error — the sender may be waiting on an id this function
/// could not even find.
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

/// [`object_schema`] for the search tools, which target one context OR
/// several: `context` and `contexts` join the given properties, and an
/// `anyOf` demands one of the two (the `cite_passage` precedent).
/// Passing both is refused by `route_tool`, where the message can say
/// so; a schema can only say "invalid".
fn dual_target_schema(properties: Value, required: &[&str]) -> Value {
    let mut schema = object_schema(properties, required);
    schema["properties"]["context"] =
        json!({ "type": "string", "description": "Context name (from list_contexts)" });
    schema["properties"]["contexts"] = json!({
        "type": "array",
        "items": { "type": "string" },
        "description": "several contexts at once — full names; every match comes back tagged with its context. Pass either context or contexts, never both."
    });
    schema["anyOf"] = json!([{ "required": ["context"] }, { "required": ["contexts"] }]);
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
    let tools = vec![
        (
            "list_contexts",
            "Routing directory: every context's name, description, stats (counts, top concepts, label sample), and usage counters (reads/empty_reads/writes, last-used times). Pick the search/ingest target here yourself.",
            object_schema(
                json!({
                    "limit": { "type": "integer", "minimum": 0, "description": "page size, keyset-paged by name (default/ceiling 1000)" },
                    "after": { "type": "string", "description": "only contexts whose name sorts strictly after this one" }
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
            "Create a group bundling contexts and, optionally, child groups (nesting: at most 3 groups tall, never cyclic). Every listed context and child group must already exist; membership never dangles — deleting a context or a group drops it from every group.",
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
            "Update a group's description and/or membership. add_contexts/remove_contexts and add_groups/remove_groups are deltas against the current members, not a replacement list; a name in both ends up a member. Added contexts and child groups must exist; removing a non-member is a no-op; nesting stays at most 3 groups tall and acyclic.",
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
                    "after": { "type": "string", "description": "only ids sorting strictly after this one" }
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
            "describe",
            "A concept's outline: which labels carry how many facts, per role. Check a hub here first, then query just the labels you need — never pull a whole profile blind.",
            object_schema(
                json!({ "context": context, "concept": { "type": "string" } }),
                &["context", "concept"],
            ),
        ),
        (
            "query",
            "Position-pinned search. subject/label/object each take a string or an array (array = match any). Outline with describe, then narrow by label. Targets one context (context) or several at once (contexts) — cross-context matches carry their context, and past the limit the strongest |weight| survives (weights share one scale).",
            dual_target_schema(
                json!({
                    "subject": { "type": ["string", "array"] },
                    "label": { "type": ["string", "array"] },
                    "object": { "type": ["string", "array"] },
                    "limit": { "type": "integer", "minimum": 0, "description": "default 100, capped at 1000" }
                }),
                &[],
            ),
        ),
        (
            "recall",
            "Every association touching the cue, whatever its position. Use query when the role matters. Targets one context (context) or several at once (contexts) — cross-context matches carry their context, and past the limit the strongest |weight| survives (weights share one scale).",
            dual_target_schema(
                json!({
                    "cue": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 0, "description": "default 100, capped at 1000" }
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
                    "limit": { "type": "integer", "minimum": 0, "description": "default 100, capped at 1000" }
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
                    "after": { "type": "string", "description": "only labels sorting strictly after this one" }
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
                    "after": { "type": "string", "description": "'concept:<alias>' or 'label:<alias>' — the last entry of the previous page" }
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
            "search_passages",
            "Paragraph search over registered passages: a lexical lane (bigram BM25) fused with a semantic lane (paragraph embeddings) where the server has them. The text lane for knowledge that never fit triples (order, conditions, discourse) — look here too when graph search comes up short. The semantic lane works best on declarative phrasing: rephrase the information need as a plausible ANSWER sentence, not a question (query \"SSO is included in the Enterprise plan\", not \"What plan includes SSO?\") — the guess only has to be shaped like the text you hope to find. Each hit names its paragraph (source + paragraph) and reports per-lane rank/score in `lanes`; a hit only the vector lane surfaced is exactly the paraphrase case the lexical lane cannot see. Targets one context (context) or several at once (contexts) — cross-context hits carry their context and interleave by per-context rank; scores compare within one context only.",
            dual_target_schema(
                json!({
                    "query": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 0, "description": "default 5" }
                }),
                &["query"],
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
                    "limit": { "type": "integer", "minimum": 0, "description": "default 100, capped at 1000" }
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
    let group_path = |key: &str| -> Result<String, String> {
        Ok(format!("/groups/{}", segment(need(arguments, key)?)))
    };
    // The search tools are dual-target: `context` prefixes the
    // per-context path, `contexts` (an array, riding the body) means
    // the cross-context route — no prefix. Exactly one of the two:
    // both is ambiguous, neither names no target at all.
    let search_base = || -> Result<String, String> {
        let one = arguments
            .get("context")
            .is_some_and(|value| !value.is_null());
        let many = arguments
            .get("contexts")
            .is_some_and(|value| !value.is_null());
        match (one, many) {
            (true, true) => Err("pass either 'context' or 'contexts', not both".to_string()),
            (false, false) => Err(
                "missing required argument 'context' (or 'contexts', to search several at once)"
                    .to_string(),
            ),
            (true, false) => context_path("context"),
            (false, true) => Ok(String::new()),
        }
    };
    Ok(match name {
        "get_protocol" => ("GET", "/protocol".to_string(), None),
        "list_contexts" => (
            "GET",
            format!("/contexts{}", query_string(arguments, &["limit", "after"])),
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
        "store_passages" => (
            "POST",
            format!("{}/sources", context_path("context")?),
            Some(pick(arguments, &["passages", "questions", "sections"])),
        ),
        "lookup_passages" => (
            "POST",
            format!("{}/sources/lookup", context_path("context")?),
            Some(pick(arguments, &["sources"])),
        ),
        "list_sources" => (
            "GET",
            format!(
                "{}/sources{}",
                context_path("context")?,
                query_string(arguments, &["limit", "after"])
            ),
            None,
        ),
        "resolve" => (
            "POST",
            format!("{}/resolve", context_path("context")?),
            Some(pick(
                arguments,
                &["cue", "dice_floor", "semantic_floor", "limit"],
            )),
        ),
        "resolve_label" => (
            "POST",
            format!("{}/resolve_label", context_path("context")?),
            Some(pick(
                arguments,
                &["cue", "dice_floor", "semantic_floor", "limit"],
            )),
        ),
        "describe" => (
            "POST",
            format!("{}/describe", context_path("context")?),
            Some(pick(arguments, &["concept"])),
        ),
        "query" => (
            "POST",
            format!("{}/query", search_base()?),
            Some(pick(
                arguments,
                &["contexts", "subject", "label", "object", "limit"],
            )),
        ),
        "recall" => (
            "POST",
            format!("{}/recall", search_base()?),
            Some(pick(arguments, &["contexts", "cue", "limit"])),
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
        "list_labels" => (
            "GET",
            format!(
                "{}/labels{}",
                context_path("context")?,
                query_string(arguments, &["limit", "after"])
            ),
            None,
        ),
        "get_aliases" => (
            "GET",
            format!(
                "{}/aliases{}",
                context_path("context")?,
                query_string(arguments, &["limit", "after"])
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
        "retract_source" => (
            "POST",
            format!("{}/sources/retract", context_path("context")?),
            Some(pick(arguments, &["source"])),
        ),
        "search_passages" => (
            "POST",
            format!("{}/sources/search", search_base()?),
            Some(pick(arguments, &["contexts", "query", "limit"])),
        ),
        "cite_passage" => (
            "POST",
            format!("{}/citations", context_path("context")?),
            Some(pick_with_alias(
                arguments,
                &["source", "paragraph"],
                "paragraph",
                "index",
            )),
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
            Some(pick(arguments, &["origins", "limit"])),
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
            "sources": ["s"], "source": "s", "query": "q", "paragraph": 0,
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

    /// The search tools are dual-target: `contexts` routes to the
    /// cross-context path with the array in the body; `context` keeps
    /// the historical per-context route, body unchanged.
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

        // The single-context form is untouched, and the body never
        // carries the path-bound name.
        let (_, path, body) = route_tool("recall", &json!({"context": "a", "cue": "x"})).unwrap();
        assert_eq!(path, "/contexts/a/recall");
        assert_eq!(body.unwrap(), json!({"cue": "x"}));
    }

    /// Both targets at once is ambiguous and neither is no target at
    /// all — each refusal says which way to fix the call, and an
    /// explicit null counts as an omission (the `pick` rule).
    #[test]
    fn search_tools_refuse_an_ambiguous_or_absent_target() {
        assert_eq!(
            route_tool(
                "recall",
                &json!({"context": "a", "contexts": ["b"], "cue": "x"})
            ),
            Err("pass either 'context' or 'contexts', not both".to_string())
        );
        let missing =
            "missing required argument 'context' (or 'contexts', to search several at once)";
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
