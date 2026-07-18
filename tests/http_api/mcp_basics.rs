//! The MCP-over-HTTP transport itself: initialize/tools/call and citation.

use serde_json::json;

use crate::support::*;

/// POST /mcp speaks the MCP Streamable HTTP transport (stateless
/// profile): the same tools as the stdio bridge, over the same routes.
#[test]
fn mcp_over_http_serves_initialize_tools_and_calls() {
    let server = Server::start("mcp");

    let (status, reply) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18"}})),
    );
    assert_eq!(status, 200);
    assert_eq!(reply["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(
        reply["result"]["serverInfo"]["version"],
        env!("CARGO_PKG_VERSION")
    );
    let instructions = reply["result"]["instructions"].as_str().unwrap();
    assert!(instructions.contains("# Taguru"));

    let (_, tools) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})),
    );
    let list = tools["result"]["tools"].as_array().unwrap();
    assert!(
        list.iter().any(|tool| tool["name"] == "recall"),
        "tools/list must advertise the bridge's tool set"
    );

    // A notification: heard (202), nothing to answer with.
    let (status, _) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "method": "notifications/initialized"})),
    );
    assert_eq!(status, 202);

    // A real tool round trip: create a context, then see it listed.
    let (_, created) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "create_context",
                               "arguments": {"name": "remote", "description": "over /mcp"}}})),
    );
    assert!(created["result"].get("isError").is_none(), "{created}");
    let (_, listed) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                    "params": {"name": "list_contexts", "arguments": {}}})),
    );
    let text = listed["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("\"remote\""), "{text}");

    // A failing tool travels as isError CONTENT (the agent reads the
    // server's explanation), not as a JSON-RPC protocol error.
    let (status, failed) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                    "params": {"name": "describe",
                               "arguments": {"context": "nope", "concept": "x"}}})),
    );
    assert_eq!(status, 200);
    assert_eq!(failed["result"]["isError"], true);
    let error_text = failed["result"]["content"][0]["text"].as_str().unwrap();
    assert!(error_text.contains("HTTP 404"), "{error_text}");

    // A tool payload between axum's 2 MiB extractor default and the
    // operator's body cap (8 MiB here) must go through: the outer /mcp
    // request already paid the configured cap, and the in-process
    // dispatch is deliberately uncapped rather than silently re-capped
    // at the extractor default.
    let big = "a".repeat(3 * 1024 * 1024);
    let (status, stored) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 60, "method": "tools/call",
                    "params": {"name": "store_passages",
                               "arguments": {"context": "remote",
                                             "passages": {"big.md": big}}}})),
    );
    assert_eq!(status, 200);
    assert!(
        stored["result"].get("isError").is_none(),
        "a 3 MiB tool call must clear the 2 MiB extractor default: {}",
        stored["result"]["content"][0]["text"]
    );

    // Unknown JSON-RPC methods ARE protocol errors...
    let (_, unknown) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 6, "method": "resources/list"})),
    );
    assert_eq!(unknown["error"]["code"], -32601);
    // ...broken JSON answers -32700 in JSON-RPC dress...
    let (status, parse) = server.call_raw("POST", "/mcp", Some("{nope"), Some("application/json"));
    assert_eq!(status, 400);
    assert_eq!(parse["error"]["code"], -32700);
    // ...and the wrong verb answers like any other route.
    let (status, _) = server.call("GET", "/mcp", None);
    assert_eq!(status, 405);
}

/// /mcp sits behind the bearer token like every route — and a tool
/// dispatched through it is NOT re-authenticated inside; the /mcp
/// entry is the auth point.
#[test]
fn mcp_endpoint_honors_bearer_auth_end_to_end() {
    let server = Server::start_with_env("mcp-auth", &[("TAGURU_API_TOKEN", "mcp-secret")]);

    let init = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}});
    let (status, _) = server.call("POST", "/mcp", Some(init.clone()));
    assert_eq!(status, 401);
    let (status, _) = server.call_with_token("POST", "/mcp", Some(init), Some("mcp-secret"));
    assert_eq!(status, 200);

    // The dispatched inner request carries no Authorization header; if
    // dispatch re-entered the middleware, this would come back as an
    // isError HTTP 401 content block.
    let call = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                      "params": {"name": "create_context",
                                 "arguments": {"name": "armed", "description": "token-protected"}}});
    let (status, reply) = server.call_with_token("POST", "/mcp", Some(call), Some("mcp-secret"));
    assert_eq!(status, 200);
    assert!(reply["result"].get("isError").is_none(), "{reply}");
}

/// Two ways a POSTed body can fail to be one JSON-RPC request before
/// `classify` ever runs a method through: a batch array (the framing
/// the 2025-06 spec dropped) and an object with no "method" at all.
/// Both are -32600, distinguished only by message text, so the
/// assertions must read the text — not just the shared code.
#[test]
fn mcp_rejects_batches_and_undecodable_messages() {
    let server = Server::start("mcp-malformed");

    let (status, batch) = server.call(
        "POST",
        "/mcp",
        Some(json!([
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
            {"jsonrpc": "2.0", "id": 2, "method": "ping"},
        ])),
    );
    assert_eq!(status, 400);
    assert_eq!(batch["error"]["code"], -32600);
    assert!(
        batch["error"]["message"]
            .as_str()
            .unwrap()
            .contains("batch messages are not part of MCP"),
        "{batch}"
    );

    let (status, undecodable) =
        server.call("POST", "/mcp", Some(json!({"jsonrpc": "2.0", "id": 1})));
    assert_eq!(status, 400);
    assert_eq!(undecodable["error"]["code"], -32600);
    assert!(
        undecodable["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not a JSON-RPC message (no method)"),
        "{undecodable}"
    );
}

/// cite_passage over tools/call: the MCP-layer counterpart of the
/// citation HTTP tests above, proving the manifest + dispatch wiring
/// (not just the HTTP handler) carries a request end to end.
#[test]
fn cite_passage_tool_executes_end_to_end_through_mcp() {
    let server = Server::start("mcp-citation");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "蔵の知識"})),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {
            "docs/aomine.md": "青嶺酒造は雲居県霧沢町の蔵元である。\n\n\
                原料米には山田錦を使い、精米歩合は50パーセントまで磨く。"
        }})),
    );

    // Acceptance criterion 1: the tool is advertised in the manifest with
    // a schema matching #5's request shape, not just reachable by name.
    let (_, tools) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})),
    );
    let manifest = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tool| tool["name"] == "cite_passage")
        .expect("tools/list must advertise cite_passage");
    assert_eq!(
        manifest["inputSchema"]["required"],
        json!(["context", "source"])
    );
    assert!(manifest["inputSchema"]["properties"]["source"].is_object());
    assert!(manifest["inputSchema"]["properties"]["paragraph"].is_object());
    // Deprecated alias for pre-#35 callers: still advertised, so the
    // schema and `paragraph`/`index` `anyOf` requirement agree that
    // either name satisfies the call.
    assert!(manifest["inputSchema"]["properties"]["index"].is_object());
    assert_eq!(
        manifest["inputSchema"]["anyOf"],
        json!([{ "required": ["paragraph"] }, { "required": ["index"] }])
    );

    let (status, reply) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "cite_passage",
                               "arguments": {"context": "sake", "source": "docs/aomine.md", "paragraph": 1}}})),
    );
    assert_eq!(status, 200);
    assert!(reply["result"].get("isError").is_none(), "{reply}");
    let text = reply["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("\"source\":\"docs/aomine.md\""), "{text}");
    assert!(text.contains("\"section\":null"), "{text}");

    // Same failure convention as the `describe` case above: the tool
    // call still succeeds as JSON-RPC, but the result carries isError.
    let (status, failed) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "cite_passage",
                               "arguments": {"context": "sake", "source": "docs/ghost.md", "paragraph": 0}}})),
    );
    assert_eq!(status, 200);
    assert_eq!(failed["result"]["isError"], true);
    let error_text = failed["result"]["content"][0]["text"].as_str().unwrap();
    assert!(error_text.contains("docs/ghost.md"), "{error_text}");

    // Acceptance criterion 2: a pre-#35 caller still on `index` gets a
    // citation back through the full MCP path, not a schema rejection.
    let (status, via_index) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                    "params": {"name": "cite_passage",
                               "arguments": {"context": "sake", "source": "docs/aomine.md", "index": 1}}})),
    );
    assert_eq!(status, 200);
    assert!(via_index["result"].get("isError").is_none(), "{via_index}");
    let text = via_index["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("\"source\":\"docs/aomine.md\""), "{text}");
}
