//! Groups over HTTP and MCP, and the MCP transport's operator/admin/
//! result-cap surface — plus a handful of tests with no real tie to
//! either (doc2query, retraction durability, search/resolve explain)
//! that landed here when this directory was split by concern and never
//! found a better home.

use serde_json::{Value, json};

use crate::support::*;

/// The four group tools ride the MCP transport like every other tool —
/// one implementation behind both the stdio bridge and POST /mcp.
#[test]
fn groups_ride_the_mcp_transport() {
    let server = Server::start("groups-mcp");
    server.ok("PUT", "/contexts/sake", None);

    let tool = |id: u64, name: &str, arguments: Value| server.call_tool(id, name, arguments);

    let created = tool(
        1,
        "create_group",
        json!({"name": "drinks", "description": "飲料", "contexts": ["sake"]}),
    );
    assert!(created.get("isError").is_none(), "{created}");
    // Nesting rides the same tools: a child at create, deltas at update.
    let nested = tool(
        2,
        "create_group",
        json!({"name": "bar", "groups": ["drinks"]}),
    );
    assert!(nested.get("isError").is_none(), "{nested}");

    let listed = tool(3, "list_groups", json!({}));
    let text = listed["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("\"drinks\""), "{text}");
    assert!(text.contains("\"sake\""), "{text}");

    let updated = tool(
        4,
        "update_group",
        json!({"name": "drinks", "remove_contexts": ["sake"]}),
    );
    assert!(updated.get("isError").is_none(), "{updated}");
    let text = updated["content"][0]["text"].as_str().unwrap();
    assert!(!text.contains("\"sake\""), "{text}");

    let deleted = tool(5, "delete_group", json!({"name": "drinks"}));
    assert!(deleted.get("isError").is_none(), "{deleted}");
    let listed = tool(6, "list_groups", json!({}));
    let text = listed["content"][0]["text"].as_str().unwrap();
    // The child's deletion also swept it out of "bar", so the name is
    // gone from the whole directory, parent row included.
    assert!(!text.contains("\"drinks\""), "{text}");
    assert!(text.contains("\"bar\""), "{text}");

    // A failing group tool travels as isError content with the API's
    // machine code visible to the agent.
    let failed = tool(7, "delete_group", json!({"name": "drinks"}));
    assert_eq!(failed["isError"], json!(true), "{failed}");
    let text = failed["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("HTTP 404"), "{text}");
}

/// A golden-question floor for cross-context retrieval: the 青嶺
/// corpus split across a brewery context and a region context, grouped,
/// and asked questions whose needed facts straddle the split. The
/// plumbing tests above prove the merge mechanics; this pins that a
/// question spanning contexts actually comes back whole.
#[test]
fn cross_context_search_answers_questions_that_straddle_the_split() {
    let server = Server::start("cross-golden");
    server.ok(
        "PUT",
        "/contexts/brewery",
        Some(json!({"description": "青嶺酒造という蔵元の知識"})),
    );
    server.ok(
        "PUT",
        "/contexts/region",
        Some(json!({"description": "霧沢町という土地の知識"})),
    );
    // 蔵元の事実は brewery に、土地の事実は region に — どちらの
    // コンテキストも単独では下の質問に答え切れない。
    server.ok(
        "POST",
        "/contexts/brewery/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "所在地", "object": "霧沢町", "weight": 1.0, "source": "第1段落"},
            {"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "source": "第1段落"},
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0, "source": "第1段落"},
            {"subject": "青嶺酒造", "label": "開く", "object": "蔵開きの祭り", "weight": 1.0, "source": "第5段落"},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/region/associations",
        Some(json!([
            {"subject": "霧沢町", "label": "所在する県", "object": "雲居県", "weight": 1.0, "source": "地誌1"},
            {"subject": "霧沢町", "label": "力を入れる", "object": "酒蔵観光", "weight": 1.0, "source": "地誌2"},
            {"subject": "蔵開きの祭り", "label": "時期", "object": "毎年2月", "weight": 1.0, "source": "地誌2"},
        ])),
    );
    server.ok(
        "PUT",
        "/groups/kirisawa",
        Some(json!({"description": "霧沢の酒", "contexts": ["brewery", "region"]})),
    );

    // 「青嶺酒造はどの県にあるか」— 所在地 (brewery) と 所在する県
    // (region) の両方が要る。ひとつの group 検索で両方が、それぞれの
    // 出所タグ付きで返ること。
    let answer = server.ok(
        "POST",
        "/query",
        Some(json!({
            "groups": ["kirisawa"],
            "subject": ["青嶺酒造", "霧沢町"],
            "label": ["所在地", "所在する県"],
        })),
    );
    assert_eq!(answer["total"], json!(2), "{answer}");
    let matches = answer["matches"].as_array().unwrap();
    let fact = |subject: &str| {
        matches
            .iter()
            .find(|m| m["subject"] == json!(subject))
            .unwrap_or_else(|| panic!("missing fact for {subject}: {answer}"))
    };
    assert_eq!(fact("青嶺酒造")["context"], json!("brewery"), "{answer}");
    assert_eq!(fact("青嶺酒造")["object"], json!("霧沢町"), "{answer}");
    assert_eq!(fact("霧沢町")["context"], json!("region"), "{answer}");
    assert_eq!(fact("霧沢町")["object"], json!("雲居県"), "{answer}");

    // 「蔵開きの祭りについて知っていること」— recall がグループ越しに
    // 両コンテキストの事実を集める (brewery: 開く, region: 時期)。
    let recalled = server.ok(
        "POST",
        "/recall",
        Some(json!({"groups": ["kirisawa"], "cue": "蔵開きの祭り"})),
    );
    assert_eq!(recalled["total"], json!(2), "{recalled}");
    let contexts: Vec<&str> = recalled["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["context"].as_str().unwrap())
        .collect();
    assert!(
        contexts.contains(&"brewery") && contexts.contains(&"region"),
        "{recalled}"
    );
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// doc2query on a lexical-only server (no embedding provider — the
/// default deployment): the attached question's wording, absent from
/// the paragraph itself, still lands the search on that paragraph
/// through the BM25 lane. Before this rode the index, questions were
/// stored and functionally inert without `TAGURU_EMBED_PASSAGES`.
#[test]
fn doc2query_questions_land_lexically_without_embeddings() {
    let server = Server::start("doc2query-lexical");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "蔵の知識"})),
    );
    let stored = server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {
                "doc": "精米歩合は50パーセントまで磨く。\n\n仕込み水は雲居山の伏流水を使う。"
            },
            "questions": {"doc": [
                {"paragraph": 0, "question": "米はどれくらい削るのか"}
            ]}
        })),
    );
    assert_eq!(stored["questions_stored"], 1, "{stored}");

    // The query shares wording with the question only — 「削る」 never
    // appears in either paragraph.
    let hits = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "米をどれくらい削る?"})),
    );
    let hits = hits.as_array().unwrap();
    assert!(!hits.is_empty(), "the question's terms must land the hit");
    assert_eq!(hits[0]["source"], json!("doc"), "{hits:?}");
    assert_eq!(hits[0]["paragraph"], json!(0), "{hits:?}");
    assert!(
        hits[0]["lanes"]["bm25"].is_object(),
        "the evidence must be lexical: {hits:?}"
    );

    // Replacing the source with a question-less revision withdraws the
    // question's terms with it (the index's wholesale-replacement unit).
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {
                "doc": "精米歩合は50パーセントまで磨く。\n\n仕込み水は雲居山の伏流水を使う。"
            }
        })),
    );
    let hits = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "米をどれくらい削る?"})),
    );
    assert_eq!(
        hits.as_array().unwrap().len(),
        0,
        "without the question the wording matches nothing: {hits}"
    );
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// The single-association retraction end to end: a write key
/// suffices where a read key is refused, aliases resolve, the outcome
/// reports found-nothing honestly, the WAL makes the withdrawal
/// survive a hard kill, and the MCP surface reaches the same handler.
#[test]
fn one_association_retracts_over_http_and_survives_a_hard_kill() {
    let server = Server::start_with_env(
        "assoc-retract",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,scribe:wtok,reader:rtok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"scribe": "write", "reader": "read"}"#,
            ),
        ],
    );
    let write = Some("wtok");
    let (status, _) = server.call_with_token(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "d"})),
        write,
    );
    assert_eq!(status, 200);
    server.call_with_token(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0, "source": "doc1"},
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0, "source": "doc2"},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "doc1"},
        ])),
        write,
    );
    server.call_with_token(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"concepts": {"Aomine Brewery": "青嶺酒造"}, "labels": {}})),
        write,
    );

    // Reads cannot retract.
    let (status, refused) = server.call_with_token(
        "POST",
        "/contexts/sake/associations/retract",
        Some(json!({"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺"})),
        Some("rtok"),
    );
    assert_eq!(status, 403, "{refused}");

    // The write key retracts through an alias; both sources' shares go.
    let (status, outcome) = server.call_with_token(
        "POST",
        "/contexts/sake/associations/retract",
        Some(json!({"subject": "Aomine Brewery", "label": "代表銘柄", "object": "青嶺"})),
        write,
    );
    assert_eq!(status, 200, "{outcome}");
    assert_eq!(outcome["result"]["retracted"], json!(true), "{outcome}");
    assert_eq!(
        outcome["result"]["attributions_removed"],
        json!(2),
        "{outcome}"
    );

    // Found-nothing honesty: the second retraction changed nothing.
    let (status, outcome) = server.call_with_token(
        "POST",
        "/contexts/sake/associations/retract",
        Some(json!({"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺"})),
        write,
    );
    assert_eq!(status, 200, "{outcome}");
    assert_eq!(outcome["result"]["retracted"], json!(false), "{outcome}");
    assert_eq!(
        outcome["result"]["attributions_removed"],
        json!(0),
        "{outcome}"
    );

    // The edge row stays visible at weight 0 (compaction sheds it);
    // the same document's other fact is untouched; activate no longer
    // carries the dead edge.
    let (_, queried) = server.call_with_token(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造"})),
        write,
    );
    let matches = queried["result"]["matches"].as_array().unwrap();
    let by_label = |label: &str| {
        matches
            .iter()
            .find(|m| m["label"] == json!(label))
            .unwrap_or_else(|| panic!("missing {label}: {queried}"))
    };
    assert_eq!(by_label("代表銘柄")["weight"], json!(0.0), "{queried}");
    assert_eq!(by_label("代表銘柄")["count"], json!(0), "{queried}");
    assert_eq!(by_label("杜氏")["weight"], json!(1.0), "{queried}");
    let (_, activated) = server.call_with_token(
        "POST",
        "/contexts/sake/activate",
        Some(json!({"origins": ["青嶺酒造"]})),
        write,
    );
    let carried: Vec<&str> = activated["result"]["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["association"]["label"].as_str().unwrap())
        .collect();
    assert!(
        !carried.contains(&"代表銘柄") && carried.contains(&"杜氏"),
        "{activated}"
    );

    // MCP reaches the same handler under the same tool vocabulary.
    let (status, answer) = server.call_with_token(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "retract_association",
                               "arguments": {"context": "sake", "subject": "青嶺酒造",
                                              "label": "杜氏", "object": "高瀬"}}})),
        write,
    );
    assert_eq!(status, 200, "{answer}");
    let text = answer["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("\"retracted\":true"), "{text}");

    // SIGKILL: no shutdown flush — the WAL alone must carry both
    // retractions across the boot.
    let data_dir = server.stop_hard();
    let server = Server::start_on("assoc-retract-reboot", data_dir);
    let (_, queried) = server.call_with_token(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造"})),
        write,
    );
    for matched in queried["result"]["matches"].as_array().unwrap() {
        assert_eq!(matched["weight"], json!(0.0), "{queried}");
        assert_eq!(matched["count"], json!(0), "{queried}");
    }
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// The operator verbs ride MCP too: flush names what it persisted and
/// the exports hand back the same streams the HTTP routes serve — an
/// agent can tend its own backup discipline (flush → export → import)
/// without leaving the tool surface. Admin gating carries over from
/// the routes the tools map onto.
#[test]
fn flush_and_export_ride_the_mcp_transport() {
    // A one-hour flush interval keeps the periodic flusher out of the
    // race: the dirty context below stays dirty until the tool runs.
    let server = Server::start_with_env("mcp-ops", &[("TAGURU_FLUSH_SECS", "3600")]);
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0, "source": "doc"},
        ])),
    );
    server.ok(
        "PUT",
        "/groups/kura",
        Some(json!({"description": "蔵元一式", "contexts": ["sake"]})),
    );

    let tool = |id: u64, name: &str, arguments: Value| server.call_tool(id, name, arguments);

    let flushed = tool(1, "flush", json!({}));
    assert!(flushed.get("isError").is_none(), "{flushed}");
    let text = flushed["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("sake"), "{text}");

    let exported = tool(2, "export_context", json!({"context": "sake"}));
    assert!(exported.get("isError").is_none(), "{exported}");
    let text = exported["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("taguru_batch"), "{text}");
    assert!(text.contains("青嶺酒造"), "{text}");

    let group = tool(3, "export_group", json!({"name": "kura"}));
    assert!(group.get("isError").is_none(), "{group}");
    let text = group["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("taguru_group"), "{text}");
    assert!(text.contains("\"kura\""), "{text}");
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// A tool result too big to buffer whole (export_context on a context
/// with enough associations) is refused with the two uncapped escape
/// hatches named — while the raw HTTP export route it names stays
/// completely unaffected by the MCP cap.
#[test]
fn an_oversized_mcp_tool_result_is_capped_but_the_raw_export_route_is_not() {
    let server =
        Server::start_with_env("mcp-result-cap", &[("TAGURU_MCP_MAX_RESULT_BYTES", "1024")]);
    server.ok("PUT", "/contexts/big", Some(json!({"description": "d"})));
    let batch: Vec<Value> = (0..200)
        .map(|i| {
            json!({"subject": format!("s{i}"), "label": "rel", "object": format!("o{i}"),
                   "weight": 1.0, "source": "doc"})
        })
        .collect();
    server.ok(
        "POST",
        "/contexts/big/associations",
        Some(Value::Array(batch)),
    );

    let reply = server.call_tool(1, "export_context", json!({"context": "big"}));
    assert_eq!(reply["isError"], true, "{reply}");
    let text = reply["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("GET /contexts/{name}/export"), "{text}");
    assert!(text.contains("taguru export"), "{text}");

    // The raw HTTP export route this message points at is uncapped —
    // the same 200-association context exports whole over it.
    let (status, exported) = server.call("GET", "/contexts/big/export", None);
    assert_eq!(status, 200, "{exported}");
    let stream = exported.as_str().expect("export body is JSONL");
    assert!(stream.contains("\"s199\""), "{stream}");
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// The MCP operator verbs keep their routes' roles: the outer /mcp
/// gate admits any read-capable key, but the inner dispatch re-checks
/// the mapped route — flush stays admin.
#[test]
fn the_mcp_flush_tool_stays_admin_gated() {
    let server = Server::start_with_env(
        "mcp-ops-auth",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,scribe:wtok"),
            ("TAGURU_KEY_SCOPES", r#"{"scribe": "write"}"#),
        ],
    );
    let call = |token: &str| {
        let (status, answer) = server.call_with_token(
            "POST",
            "/mcp",
            Some(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": "flush", "arguments": {}}})),
            Some(token),
        );
        assert_eq!(status, 200, "{answer}");
        answer["result"].clone()
    };

    let refused = call("wtok");
    assert_eq!(refused["isError"], json!(true), "{refused}");
    let text = refused["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("HTTP 403"), "{text}");

    let allowed = call("atok");
    assert!(allowed.get("isError").is_none(), "{allowed}");
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// #62 item 3: `compact` is MCP-advertised but unlisted in
/// `required_role` — the role table's fail-closed default makes it
/// admin, same as `flush`.
#[test]
fn the_mcp_compact_tool_stays_admin_gated() {
    let server = Server::start_with_env(
        "mcp-compact-auth",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,scribe:wtok"),
            ("TAGURU_KEY_SCOPES", r#"{"scribe": "write"}"#),
        ],
    );
    server.call_with_token(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "d"})),
        Some("atok"),
    );
    let call = |token: &str| {
        let (status, answer) = server.call_with_token(
            "POST",
            "/mcp",
            Some(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": "compact", "arguments": {"context": "sake"}}})),
            Some(token),
        );
        assert_eq!(status, 200, "{answer}");
        answer["result"].clone()
    };

    let refused = call("wtok");
    assert_eq!(refused["isError"], json!(true), "{refused}");
    let text = refused["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("HTTP 403"), "{text}");

    let allowed = call("atok");
    assert!(allowed.get("isError").is_none(), "{allowed}");
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// #62 item 3: `get_context`/`get_group` round-trip through MCP to the
/// same directory rows the HTTP routes serve.
#[test]
fn the_mcp_get_context_and_get_group_tools_return_the_http_rows() {
    let server = Server::start("mcp-get-passthrough");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "酒蔵の記憶"})),
    );
    server.ok(
        "PUT",
        "/groups/breweries",
        Some(json!({"description": "g", "contexts": ["sake"]})),
    );

    let call = |name: &str, arguments: Value| {
        let (status, answer) = server.call(
            "POST",
            "/mcp",
            Some(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": name, "arguments": arguments}})),
        );
        assert_eq!(status, 200, "{answer}");
        let result = answer["result"].clone();
        assert!(result.get("isError").is_none(), "{result}");
        let text = result["content"][0]["text"].as_str().unwrap().to_string();
        // The tool passes the HTTP handler's response body through
        // verbatim, envelope and all — same as every other GET-backed
        // tool (`export_context`, `list_contexts`, ...).
        serde_json::from_str::<Value>(&text).unwrap()["result"].clone()
    };

    let context = call("get_context", json!({"context": "sake"}));
    assert_eq!(context["name"], json!("sake"));
    assert_eq!(context["description"], json!("酒蔵の記憶"));

    let group = call("get_group", json!({"name": "breweries"}));
    assert_eq!(group["name"], json!("breweries"));
    assert_eq!(group["contexts"], json!(["sake"]));
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// #62 item 1: the `import` tool carries a multi-line NDJSON stream as
/// raw text, not a JSON-quoted string — the regression `call_inner`
/// (HTTP transport) and `Bridge::call` (stdio transport) both fixed:
/// naively `Value::to_string()`-encoding a string argument escapes
/// every newline, collapsing the stream onto one line and breaking
/// the line-oriented parse. A batch that only applies correctly when
/// every line lands separately is the regression test.
#[test]
fn the_mcp_import_tool_applies_a_multi_line_stream() {
    let server = Server::start("mcp-import");
    let stream = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-mcp\", \
                 \"create\": {\"description\": \"d\"}}\n\
                 {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n\
                 {\"passage\": \"蔵の杜氏は高瀬。\"}\n";

    let (status, answer) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "import", "arguments": {"stream": stream}}})),
    );
    assert_eq!(status, 200, "{answer}");
    let result = answer["result"].clone();
    assert!(result.get("isError").is_none(), "{result}");
    let text = result["content"][0]["text"].as_str().unwrap().to_string();
    let envelope: Value = serde_json::from_str(&text).unwrap();
    let outcome = &envelope["result"]["batches"][0];
    assert_eq!(outcome["created"], json!(true));
    assert_eq!(outcome["associations"], json!(1));
    assert_eq!(outcome["passage_stored"], json!(true));

    let row = server.ok("GET", "/contexts/sake", None);
    assert_eq!(row["description"], json!("d"));

    // dry_run previews without writing: the context this batch would
    // create does not exist afterward.
    let preview_stream = "{\"taguru_batch\": 1, \"context\": \"bunko\", \"source\": \"s\", \
                          \"create\": {\"description\": \"d\"}}\n";
    let (status, preview_answer) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "import",
                               "arguments": {"stream": preview_stream, "dry_run": true}}})),
    );
    assert_eq!(status, 200, "{preview_answer}");
    let preview_text = preview_answer["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string();
    let preview_envelope: Value = serde_json::from_str(&preview_text).unwrap();
    assert_eq!(
        preview_envelope["result"]["batches"][0]["created"],
        json!(true)
    );
    let (status, _) = server.call("GET", "/contexts/bunko", None);
    assert_eq!(status, 404, "dry_run through the MCP tool must not write");
    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}

/// The search-explain decision tree (#75): one call names the first
/// verdict that applies — stored at all, sharing any term, ranked but
/// cut off, or served — with the evidence that makes the verdict
/// checkable. The 表記ゆれ case is the one the endpoint exists for:
/// the paragraph spells 酒蔵, the query spells 酒造, and only seeing
/// both term tables side by side says so.
#[test]
fn search_explain_names_the_first_verdict_that_applies() {
    let server = Server::start("search-explain");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "蔵の知識"})),
    );
    // docs/many.md: seven short twin paragraphs plus one long, diluted
    // straggler — everything shares the 霧沢 bigram, so the straggler
    // scores, but dead last, past the default limit of 5.
    let straggler = "霧沢について、この段落は他の段落よりも長く、余計な語を\
                     たくさん含んでいるため、字数あたりの一致密度が下がって\
                     順位は最下位に沈む。";
    let twins: Vec<String> = (1..=7)
        .map(|n| format!("霧沢と霧沢の里、その{n}。"))
        .collect();
    let many = format!("{}\n\n{straggler}", twins.join("\n\n"));
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {
            "docs/kura.md": "青嶺酒造は雲居県の蔵元である。\n\n\
                原料米には山田錦を使い、精米歩合は50パーセントまで磨く。",
            "docs/kuramoto.md": "その酒蔵は谷あいにある。",
            "docs/many.md": many,
        }})),
    );

    // Verdict: served. The best showing is chosen when no paragraph is
    // named, and the per-term table carries the addends that put it
    // there.
    let hits = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "精米歩合はどこまで磨く?"})),
    );
    assert_eq!(hits[0]["source"], json!("docs/kura.md"));
    let served = server.ok(
        "POST",
        "/contexts/sake/sources/search/explain",
        Some(json!({"query": "精米歩合はどこまで磨く?", "source": "docs/kura.md"})),
    );
    assert_eq!(served["verdict"], json!("served"), "{served}");
    assert_eq!(served["paragraph"], hits[0]["paragraph"]);
    assert_eq!(served["paragraph_named"], json!(false));
    assert_eq!(served["ranking"]["served"], json!(true));
    assert_eq!(served["ranking"]["rank"], json!(1));
    assert_eq!(served["ranking"]["fused"], json!(false));
    assert!(
        served["bm25"]["terms"]
            .as_array()
            .unwrap()
            .iter()
            .any(|term| term["contribution"].as_f64().unwrap() > 0.0
                && term["df"].as_u64().unwrap() >= 1),
        "{served}"
    );
    // The vector lane names why it did not run.
    assert_eq!(served["vector"]["ran"], json!(false));
    assert!(
        served["vector"]["reason"]
            .as_str()
            .unwrap()
            .contains("no embedding provider"),
        "{served}"
    );

    // Verdict: below_cutoff. The straggler ranks past the default
    // limit; the reported limit_to_reach is VERIFIED — re-searching at
    // it actually surfaces the paragraph.
    let cutoff = server.ok(
        "POST",
        "/contexts/sake/sources/search/explain",
        Some(json!({"query": "霧沢", "source": "docs/many.md", "paragraph": 7})),
    );
    assert_eq!(cutoff["verdict"], json!("below_cutoff"), "{cutoff}");
    let rank = cutoff["ranking"]["rank"].as_u64().unwrap();
    assert!(
        rank > 5,
        "the straggler must rank past the default limit: {cutoff}"
    );
    assert_eq!(cutoff["ranking"]["served"], json!(false));
    assert!(cutoff["ranking"]["cutoff_score"].as_f64().unwrap() > 0.0);
    let reach = cutoff["ranking"]["limit_to_reach"].as_u64().unwrap();
    assert!(reach >= rank, "{cutoff}");
    let wider = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "霧沢", "limit": reach})),
    );
    assert!(
        wider
            .as_array()
            .unwrap()
            .iter()
            .any(|hit| hit["source"] == json!("docs/many.md") && hit["paragraph"] == json!(7)),
        "limit_to_reach must actually reach it: {wider}"
    );
    // The straggler's own term table shows the match that was too weak.
    assert!(
        cutoff["bm25"]["terms"]
            .as_array()
            .unwrap()
            .iter()
            .any(|term| term["term"] == json!("霧沢") && term["tf"].as_f64().unwrap() > 0.0),
        "{cutoff}"
    );

    // Verdict: no_term_overlap — the 表記ゆれ case. The query spells
    // 酒造, this source spells 酒蔵; both spellings sit in the answer.
    let overlap = server.ok(
        "POST",
        "/contexts/sake/sources/search/explain",
        Some(json!({"query": "酒造", "source": "docs/kuramoto.md"})),
    );
    assert_eq!(overlap["verdict"], json!("no_term_overlap"), "{overlap}");
    assert_eq!(overlap["query_terms"], json!(["酒造"]));
    assert!(
        overlap["paragraph_terms"]
            .as_array()
            .unwrap()
            .contains(&json!("酒蔵")),
        "{overlap}"
    );
    assert!(
        overlap["summary"]
            .as_str()
            .unwrap()
            .contains("shares no term"),
        "{overlap}"
    );

    // Verdict: not_stored — and retraction lands in the same verdict,
    // because the store keeps no tombstone history to tell them apart.
    let ghost = server.ok(
        "POST",
        "/contexts/sake/sources/search/explain",
        Some(json!({"query": "霧沢", "source": "docs/ghost.md"})),
    );
    assert_eq!(ghost["verdict"], json!("not_stored"), "{ghost}");
    assert!(
        ghost["summary"].as_str().unwrap().contains("retracted"),
        "{ghost}"
    );
    server.ok(
        "POST",
        "/contexts/sake/sources/retract",
        Some(json!({"source": "docs/kuramoto.md"})),
    );
    let retracted = server.ok(
        "POST",
        "/contexts/sake/sources/search/explain",
        Some(json!({"query": "酒造", "source": "docs/kuramoto.md"})),
    );
    assert_eq!(retracted["verdict"], json!("not_stored"), "{retracted}");

    // Verdict: paragraph_out_of_range, with the range that would fit.
    let out = server.ok(
        "POST",
        "/contexts/sake/sources/search/explain",
        Some(json!({"query": "霧沢", "source": "docs/kura.md", "paragraph": 9})),
    );
    assert_eq!(out["verdict"], json!("paragraph_out_of_range"), "{out}");
    assert_eq!(out["paragraphs"], json!(2));

    // Verdict: no_query_terms — punctuation tokenizes to nothing.
    let empty = server.ok(
        "POST",
        "/contexts/sake/sources/search/explain",
        Some(json!({"query": "、。", "source": "docs/kura.md"})),
    );
    assert_eq!(empty["verdict"], json!("no_query_terms"), "{empty}");

    // Unknown context stays the outer 404, never a verdict.
    let (status, _) = server.call(
        "POST",
        "/contexts/nazo/sources/search/explain",
        Some(json!({"query": "霧沢", "source": "docs/kura.md"})),
    );
    assert_eq!(status, 404);
}

/// The resolve-explain decision tree (#75): membership first, then the
/// same tiers/floors/limits the resolve endpoint runs, with the
/// expected name located in each. The exact-tier shortcut gets its own
/// verdict — a cue that IS another stored spelling never scores
/// anything else, which no floor tweak can fix.
#[test]
fn resolve_explain_names_the_first_verdict_that_applies() {
    let server = Server::start("resolve-explain");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "蔵の知識"})),
    );
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺", "weight": 1.0},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0},
            {"subject": "青嶺酒造", "label": "原料米", "object": "山田錦", "weight": 1.0},
            {"subject": "幻の蔵元", "label": "所在", "object": "山田町", "weight": 1.0},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"concepts": {"Aomine": "青嶺酒造"}})),
    );

    // Verdict: served — and an alias expectation reports its canonical.
    let served = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": "青嶺酒造", "expected": "青嶺酒造"})),
    );
    assert_eq!(served["verdict"], json!("served"), "{served}");
    assert_eq!(served["expected_kind"], json!("exact"));
    assert_eq!(served["ranking"]["rank"], json!(1));
    assert_eq!(served["lexical"]["confident"], json!(true));
    assert_eq!(served["semantic"]["entered"], json!(false));
    let alias = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": "青嶺酒", "expected": "Aomine"})),
    );
    assert_eq!(alias["verdict"], json!("served"), "{alias}");
    assert_eq!(alias["canonical"], json!("青嶺酒造"));
    assert_eq!(alias["expected_kind"], json!("alias"));

    // Verdict: cue_resolved_exactly — the exact tier answers alone.
    let eclipsed = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": "高瀬", "expected": "青嶺酒造"})),
    );
    assert_eq!(
        eclipsed["verdict"],
        json!("cue_resolved_exactly"),
        "{eclipsed}"
    );
    assert!(
        eclipsed["summary"].as_str().unwrap().contains("高瀬"),
        "{eclipsed}"
    );

    // Verdict: below_floor — the actual Dice score against the floor
    // in effect, which is also the floor that would have shown it.
    // 青嶺の酒造り shares the 青嶺/酒造 bigrams with 青嶺酒造 without
    // containing it: a fuzzy 0.5, gated by a request floor of 0.6.
    let floored = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": "青嶺の酒造り", "expected": "青嶺酒造", "dice_floor": 0.6})),
    );
    assert_eq!(floored["verdict"], json!("below_floor"), "{floored}");
    let score = floored["lexical"]["score"].as_f64().unwrap();
    assert!((score - 0.5).abs() < 1e-9, "{floored}");
    assert_eq!(floored["lexical"]["kind"], json!("fuzzy"));
    assert_eq!(floored["lexical"]["floor"], json!(0.6));

    // The dice floor gates ONLY the fuzzy tier: a containment hit
    // sails past any floor, and explain reports the serve, not a
    // fictitious floor refusal.
    let contained = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": "青嶺酒", "expected": "青嶺酒造", "dice_floor": 0.9})),
    );
    assert_eq!(contained["verdict"], json!("served"), "{contained}");
    assert_eq!(contained["lexical"]["kind"], json!("containment"));

    // Verdict: below_cutoff — lost on limit, with a verified way back.
    let cut = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": "山田", "expected": "山田錦", "limit": 1})),
    );
    assert_eq!(cut["verdict"], json!("below_cutoff"), "{cut}");
    assert_eq!(cut["ranking"]["rank"], json!(2));
    let reach = cut["ranking"]["limit_to_reach"].as_u64().unwrap();
    let wider = server.ok(
        "POST",
        "/contexts/sake/resolve",
        Some(json!({"cue": "山田", "limit": reach})),
    );
    assert!(
        wider
            .as_array()
            .unwrap()
            .iter()
            .any(|candidate| candidate["name"] == json!("山田錦")),
        "limit_to_reach must actually reach it: {wider}"
    );

    // Verdict: semantic_not_run — no lexical relation at all, and the
    // tier that could have found it names why it never ran.
    let semantic = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": "みかん", "expected": "青嶺酒造"})),
    );
    assert_eq!(semantic["verdict"], json!("semantic_not_run"), "{semantic}");
    assert_eq!(semantic["semantic"]["entered"], json!(true));
    assert!(
        semantic["semantic"]["reason"]
            .as_str()
            .unwrap()
            .contains("no embedding provider"),
        "{semantic}"
    );

    // Verdict: not_in_vocabulary — with the nearest stored spellings,
    // so the repair (register an alias) is one step away.
    let missing = server.ok(
        "POST",
        "/contexts/sake/resolve/explain",
        Some(json!({"cue": "青嶺", "expected": "幻の蔵"})),
    );
    assert_eq!(missing["verdict"], json!("not_in_vocabulary"), "{missing}");
    assert_eq!(missing["in_vocabulary"], json!(false));
    assert_eq!(missing["nearest"]["lexical"][0]["name"], json!("幻の蔵元"));

    // The label twin answers through its own route.
    let label = server.ok(
        "POST",
        "/contexts/sake/resolve_label/explain",
        Some(json!({"cue": "杜氏の職", "expected": "杜氏"})),
    );
    assert_eq!(label["verdict"], json!("served"), "{label}");
    assert_eq!(label["canonical"], json!("杜氏"));

    // Unknown context stays the outer 404, never a verdict.
    let (status, _) = server.call(
        "POST",
        "/contexts/nazo/resolve/explain",
        Some(json!({"cue": "青嶺", "expected": "青嶺酒造"})),
    );
    assert_eq!(status, 404);

    // The MCP mirror diagnoses the same miss in one tool call.
    let (_, reply) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                    "params": {"name": "explain_resolve",
                               "arguments": {"context": "sake", "cue": "青嶺", "expected": "幻の蔵"}}})),
    );
    assert!(reply["result"].get("isError").is_none(), "{reply}");
    let text = reply["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("not_in_vocabulary"), "{text}");
    assert!(text.contains("幻の蔵元"), "{text}");

    // And the search mirror rides the same registry.
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"docs/kura.md": "その酒蔵は谷あいにある。"}})),
    );
    let (_, reply) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 8, "method": "tools/call",
                    "params": {"name": "explain_search",
                               "arguments": {"context": "sake", "query": "酒造",
                                             "source": "docs/kura.md"}}})),
    );
    assert!(reply["result"].get("isError").is_none(), "{reply}");
    let text = reply["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("no_term_overlap"), "{text}");
    assert!(text.contains("酒蔵"), "{text}");
}

/// #62 item 2: the composed `retrieve` tool runs resolve → describe →
/// activate → cite_passage (and, forced on here, the search_passages
/// fallback) end to end over the real HTTP/MCP transport. The point
/// this covers beyond `run_retrieve`'s own unit tests (mock closures,
/// no real async runtime): `remote_mcp::serve` bridges `run_retrieve`'s
/// synchronous callback onto the async `call_inner` via
/// `block_in_place` + `Handle::block_on`, and that bridge only proves
/// out under an actual multi-thread tokio runtime.
#[test]
fn the_mcp_retrieve_tool_runs_the_composed_loop_end_to_end() {
    let server = Server::start("mcp-retrieve");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "酒蔵の記憶"})),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {
            "docs/kura.md": "青嶺酒造は雲居県霧沢町の蔵元である。杜氏は高瀬である。"
        }})),
    );
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0,
             "source": "docs/kura.md", "paragraph": 0},
        ])),
    );

    let result = server.call_tool(
        1,
        "retrieve",
        json!({"context": "sake", "origins": ["青嶺酒造"]}),
    );
    assert!(result.get("isError").is_none(), "{result}");
    let text = result["content"][0]["text"].as_str().unwrap();
    let envelope: Value = serde_json::from_str(text).unwrap();

    assert_eq!(
        envelope["resolved"]["青嶺酒造"][0]["name"],
        json!("青嶺酒造")
    );
    assert_eq!(
        envelope["outline"]["青嶺酒造"]["concept"],
        json!("青嶺酒造")
    );
    let associations = envelope["associations"].as_array().unwrap();
    assert_eq!(associations.len(), 1, "{envelope}");
    assert_eq!(associations[0]["object"], json!("高瀬"));
    let citations = envelope["citations"].as_array().unwrap();
    assert_eq!(citations.len(), 1, "{envelope}");
    assert_eq!(citations[0]["source"], json!("docs/kura.md"));
    assert_eq!(citations[0]["paragraph"], json!(0));
    assert!(
        citations[0]["citation"]["text"]
            .as_str()
            .unwrap()
            .contains("青嶺酒造"),
        "{envelope}"
    );
    assert_eq!(envelope["passage_hits"], json!([]));

    // A second call exercises the text-lane fallback (step 5) over the
    // same transport, forced on regardless of what associations came
    // back (text_fallback_only_if_empty: false) so it does not hinge on
    // resolve's fuzzy-match behavior.
    let fallback = server.call_tool(
        2,
        "retrieve",
        json!({
            "context": "sake", "origins": ["青嶺酒造"],
            "text_fallback_query": "杜氏は高瀬である",
            "text_fallback_only_if_empty": false
        }),
    );
    assert!(fallback.get("isError").is_none(), "{fallback}");
    let fallback_text = fallback["content"][0]["text"].as_str().unwrap();
    let fallback_envelope: Value = serde_json::from_str(fallback_text).unwrap();
    assert!(
        !fallback_envelope["passage_hits"]
            .as_array()
            .unwrap()
            .is_empty(),
        "{fallback_envelope}"
    );

    let _ = std::fs::remove_dir_all(server.stop_gracefully());
}
