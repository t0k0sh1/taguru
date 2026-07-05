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
//! Run one writer per data directory: this bridge talks to the HTTP
//! server rather than opening the data directory itself, so any number
//! of agents can share one running server.

use std::io::{BufRead, Write};
use std::time::Duration;

use serde_json::{Value, json};

const FALLBACK_PROTOCOL_VERSION: &str = "2024-11-05";
const FALLBACK_INSTRUCTIONS: &str = include_str!("../../docs/llm-protocol.md");

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
        agent: ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(timeout_secs))
            .build(),
    };

    let instructions = bridge
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
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(message) = serde_json::from_str::<Value>(&line) else {
            eprintln!("taguru-mcp: ignoring undecodable line");
            continue;
        };
        if let Some(response) = handle(&bridge, &instructions, &message) {
            let mut out = stdout.lock();
            let _ = writeln!(out, "{response}");
            let _ = out.flush();
        }
    }
}

/// Dispatches one JSON-RPC message; notifications (no id) get no reply.
fn handle(bridge: &Bridge, instructions: &str, message: &Value) -> Option<Value> {
    let method = message.get("method").and_then(Value::as_str)?;
    let id = message.get("id")?.clone();
    if id.is_null() {
        return None;
    }
    let params = message.get("params").cloned().unwrap_or(Value::Null);

    let outcome = match method {
        "initialize" => {
            let version = params
                .get("protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or(FALLBACK_PROTOCOL_VERSION);
            Ok(json!({
                "protocolVersion": version,
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "taguru", "version": env!("CARGO_PKG_VERSION") },
                "instructions": instructions,
            }))
        }
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tool_definitions() })),
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
            match call_tool(bridge, name, &arguments) {
                Ok(text) => Ok(json!({ "content": [{ "type": "text", "text": text }] })),
                Err(text) => Ok(json!({
                    "content": [{ "type": "text", "text": text }],
                    "isError": true,
                })),
            }
        }
        _ => Err(json!({ "code": -32601, "message": format!("unknown method '{method}'") })),
    };

    Some(match outcome {
        Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
        Err(error) => json!({ "jsonrpc": "2.0", "id": id, "error": error }),
    })
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
        let mut request = self.agent.request(method, &format!("{}{path}", self.base));
        if let Some(token) = &self.token {
            request = request.set("Authorization", &format!("Bearer {token}"));
        }
        let response = match body {
            Some(body) => request
                .set("Content-Type", "application/json")
                .send_string(&body.to_string()),
            None => request.call(),
        };
        match response {
            Ok(response) => response
                .into_string()
                .map_err(|error| format!("response unreadable: {error}")),
            Err(ureq::Error::Status(code, response)) => {
                let detail = response.into_string().unwrap_or_default();
                Err(format!("HTTP {code}: {detail}"))
            }
            Err(error) => Err(format!("server unreachable at {}: {error}", self.base)),
        }
    }
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

fn tool_definitions() -> Vec<Value> {
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

/// Executes one tool call: maps it onto its HTTP request, then runs it.
fn call_tool(bridge: &Bridge, name: &str, arguments: &Value) -> Result<String, String> {
    let (method, path, body) = route_tool(name, arguments)?;
    bridge.call(method, &path, body)
}

/// Maps one tool call onto (method, path, body) — pure, so the mapping
/// from advertised tools to HTTP requests is testable without a server.
fn route_tool(
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
}
