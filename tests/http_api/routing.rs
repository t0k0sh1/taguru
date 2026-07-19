//! `taguru route` (issue #130): the stateless scatter-gather router.
//! The load-bearing property is EQUIVALENCE — the router over split
//! shards must answer what one instance holding the same contexts
//! answers, for every multi-context verb, merges and cursors and
//! groups and refusals alike — so the core test drives an identical
//! corpus into both topologies through their own front doors and
//! diffs the JSON, with only latency stamps and usage timestamps
//! normalized. The second test covers what has no single-instance
//! analog: a shard dying mid-fleet (labeled partials, pass-through
//! refusals, recovery), and bearer auth passing through untouched.

use serde_json::{Value, json};

use crate::support::*;

/// Fields legitimately different between two servers answering the
/// same question: the latency stamp, the directory's usage timestamps
/// (unix seconds — two runs may straddle a tick), and residency
/// (`loaded` is scheduling truth, not data truth).
fn normalized(value: &Value) -> Value {
    fn walk(value: &mut Value) {
        match value {
            Value::Object(map) => {
                for key in ["time", "last_read_epoch", "last_write_epoch"] {
                    if map.contains_key(key) {
                        map.insert(key.to_string(), json!(0));
                    }
                }
                if map.contains_key("loaded") {
                    map.insert("loaded".to_string(), json!(false));
                }
                for (_, child) in map.iter_mut() {
                    walk(child);
                }
            }
            Value::Array(items) => items.iter_mut().for_each(walk),
            _ => {}
        }
    }
    let mut copy = value.clone();
    walk(&mut copy);
    copy
}

/// One identical corpus, driven through whatever front door `server`
/// is — the single instance directly, the sharded fleet through its
/// router — so the two sides cannot drift by construction. Exercises
/// the router's own write paths while it seeds: proxied creates and
/// passage stores, a multi-batch import whose batches alternate
/// shards, and group writes that need member projection.
fn seed(server: &Server) {
    for (name, description) in [
        ("sake", "銘柄と蔵元の知識"),
        ("breweries", "蔵元の台帳"),
        ("glossary", "酒の用語集"),
    ] {
        server.ok(
            "PUT",
            &format!("/contexts/{name}"),
            Some(json!({"description": description})),
        );
    }
    // sake and breweries live on shard A, glossary on shard B (see the
    // fleet's map) — this stream's batches run A, B, A, so the router
    // must split it into three chunks and reassemble the outcomes in
    // stream order. The weights are chosen to exercise the merge: a
    // |weight| tie across contexts (2.0), an identical triple in two
    // contexts (共通/例/概念 — the cursor's `context` field is what
    // keeps them apart), and a negative weight whose magnitude tops
    // the ranking.
    let stream = concat!(
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-a\"}\n",
        "{\"subject\": \"青嶺\", \"label\": \"銘柄である\", \"object\": \"酒\", \"weight\": 2.0}\n",
        "{\"subject\": \"辛口\", \"label\": \"特徴\", \"object\": \"酒\", \"weight\": 1.0}\n",
        "{\"subject\": \"共通\", \"label\": \"例\", \"object\": \"概念\", \"weight\": 0.5}\n",
        "{\"taguru_batch\": 1, \"context\": \"glossary\", \"source\": \"doc-b\"}\n",
        "{\"subject\": \"辛口\", \"label\": \"意味する\", \"object\": \"甘くない\", \"weight\": 2.0}\n",
        "{\"subject\": \"共通\", \"label\": \"例\", \"object\": \"概念\", \"weight\": 0.5}\n",
        "{\"taguru_batch\": 1, \"context\": \"breweries\", \"source\": \"doc-c\"}\n",
        "{\"subject\": \"青嶺酒造\", \"label\": \"造る\", \"object\": \"青嶺\", \"weight\": -2.5}\n",
        "{\"taguru_group\": 1, \"name\": \"jp\", \"description\": \"日本酒\", \"contexts\": [\"sake\", \"glossary\"]}\n",
    );
    let (status, outcome) = post_import(server, stream, None);
    assert_eq!(status, 200, "{outcome}");
    // A nested group whose direct member and child live on different
    // shards: `contexts` needs projection, `groups` broadcasts whole.
    server.ok(
        "PUT",
        "/groups/all",
        Some(json!({"description": "全部", "contexts": ["breweries"], "groups": ["jp"]})),
    );
    // Passages sharing a term across the shard split, for the
    // rank-interleaved passage merge.
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"doc-a": "麹と水で仕込む。\n\n辛口の酒は麹の使い方で決まる。"}})),
    );
    server.ok(
        "POST",
        "/contexts/glossary/sources",
        Some(json!({"passages": {"doc-b": "麹（こうじ）は蒸した米に麹菌を生やしたもの。"}})),
    );
}

/// Runs one request against both front doors and asserts the answers
/// are identical after normalization. Returns the router's parsed
/// body for follow-up assertions.
fn assert_equivalent(
    single: &Server,
    router: &Server,
    method: &str,
    path: &str,
    body: Option<Value>,
) -> Value {
    let (single_status, single_body) = single.call(method, path, body.clone());
    let (router_status, router_body) = router.call(method, path, body);
    assert_eq!(
        single_status, router_status,
        "{method} {path}: status diverged — single {single_body} vs router {router_body}"
    );
    assert_eq!(
        normalized(&single_body),
        normalized(&router_body),
        "{method} {path}: bodies diverged"
    );
    router_body
}

/// The acceptance test: a three-context corpus, once on a single
/// instance and once split across two shards behind the router, must
/// answer every multi-context verb identically — including the paged
/// resume, whose cursor is anchored on the last match itself and so
/// forwards to every shard verbatim.
#[test]
fn the_router_over_split_shards_answers_exactly_like_one_instance() {
    let single = Server::start("router-eq-single");
    let shard_a = Server::start("router-eq-shard-a");
    let shard_b = Server::start("router-eq-shard-b");
    let router = Server::start_router(
        "router-eq",
        &format!(
            "sake = {}\nbreweries = {}\nglossary = {}\n",
            shard_a.base, shard_a.base, shard_b.base
        ),
        &[],
    );

    seed(&single);
    seed(&router);

    // The seeding itself already proved the import split: now the
    // responses. Cross recall with contexts, with groups (nested),
    // and mixed.
    for body in [
        json!({"contexts": ["sake", "breweries", "glossary"], "cue": "青嶺"}),
        json!({"groups": ["all"], "cue": "辛口"}),
        json!({"contexts": ["breweries"], "groups": ["jp"], "cue": "青嶺"}),
    ] {
        let answer = assert_equivalent(&single, &router, "POST", "/recall", Some(body));
        assert!(
            answer["result"]["total"].as_u64().unwrap_or(0) > 0,
            "an equivalence over empty results proves nothing: {answer}"
        );
    }

    // Cross query, paged: 4 known matches (glossary 2.0 → sake 1.0 →
    // the 0.5 tie broken by context name), cut at 2, resumed with a
    // cursor built from the last match exactly as a client builds it.
    let page_body = json!({"groups": ["all"], "subject": ["共通", "辛口"], "limit": 2});
    let page1 = assert_equivalent(&single, &router, "POST", "/query", Some(page_body));
    let matches = page1["result"]["matches"].as_array().unwrap();
    assert_eq!(matches.len(), 2, "{page1}");
    let last = &matches[1];
    let after = json!({
        "weight": last["weight"],
        "context": last["context"],
        "subject": last["subject"],
        "label": last["label"],
        "object": last["object"],
    });
    let page2 = assert_equivalent(
        &single,
        &router,
        "POST",
        "/query",
        Some(json!({"groups": ["all"], "subject": ["共通", "辛口"], "limit": 2, "after": after})),
    );
    let matches = page2["result"]["matches"].as_array().unwrap();
    assert_eq!(matches.len(), 2, "{page2}");
    // The identical-triple pair survives paging because the cursor
    // carries `context`: both 0.5 matches arrive, each tagged.
    assert!(
        matches
            .iter()
            .all(|hit| hit["subject"] == "共通" && hit["weight"] == 0.5),
        "{page2}"
    );

    // The passage merge: per-context rank interleaving, scores
    // per-context, both shard splits and the group path.
    for body in [
        json!({"contexts": ["sake", "glossary", "breweries"], "query": "麹"}),
        json!({"groups": ["all"], "query": "麹", "limit": 2}),
    ] {
        let answer = assert_equivalent(&single, &router, "POST", "/sources/search", Some(body));
        assert!(!answer["result"].as_array().unwrap().is_empty(), "{answer}");
    }

    // The directories and the group surfaces: unions must equal the
    // single instance's own rows.
    assert_equivalent(&single, &router, "GET", "/contexts", None);
    assert_equivalent(&single, &router, "GET", "/groups", None);
    assert_equivalent(&single, &router, "GET", "/groups/jp", None);
    assert_equivalent(&single, &router, "GET", "/groups/all", None);
    // The export record: the router unions per-shard projections back
    // into the one line the single instance renders.
    let (single_status, single_export) = single.call("GET", "/groups/jp/export", None);
    let (router_status, router_export) = router.call("GET", "/groups/jp/export", None);
    assert_eq!(single_status, 200);
    assert_eq!(router_status, 200);
    assert_eq!(single_export, router_export, "the exported record diverged");

    // Group deltas through the router: remove one member (a
    // projection-touching PATCH), compare the returned row, put it
    // back, compare again.
    assert_equivalent(
        &single,
        &router,
        "PATCH",
        "/groups/jp",
        Some(json!({"remove_contexts": ["glossary"]})),
    );
    assert_equivalent(
        &single,
        &router,
        "PATCH",
        "/groups/jp",
        Some(json!({"add_contexts": ["glossary"]})),
    );

    // Refusals: naming nothing, an unknown context (first in list
    // order), an unknown group — same code, same message, same status.
    assert_equivalent(
        &single,
        &router,
        "POST",
        "/recall",
        Some(json!({"cue": "x"})),
    );
    assert_equivalent(
        &single,
        &router,
        "POST",
        "/recall",
        Some(json!({"contexts": ["sake", "missing", "alsomissing"], "cue": "x"})),
    );
    assert_equivalent(
        &single,
        &router,
        "POST",
        "/query",
        Some(json!({"groups": ["nogroup"], "subject": "x"})),
    );

    // Per-context verbs proxy byte-for-byte — a routed read and a
    // routed refusal (unknown subpath falls through to the shard's
    // own 404 shape).
    assert_equivalent(
        &single,
        &router,
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "青嶺"})),
    );
    assert_equivalent(&single, &router, "GET", "/contexts/sake/export", None);
    assert_equivalent(
        &single,
        &router,
        "POST",
        "/contexts/sake/unknown-verb",
        None,
    );

    // The MCP transport over the router: the same tool call answers
    // the same content (the tool text is the response JSON — parse it
    // and normalize the latency stamp inside).
    let single_tool = single.call_tool(
        1,
        "recall",
        json!({"contexts": ["sake", "glossary"], "cue": "辛口"}),
    );
    let router_tool = router.call_tool(
        1,
        "recall",
        json!({"contexts": ["sake", "glossary"], "cue": "辛口"}),
    );
    assert_eq!(single_tool["isError"], router_tool["isError"]);
    let parse = |tool: &Value| -> Value {
        serde_json::from_str(tool["content"][0]["text"].as_str().unwrap()).unwrap()
    };
    assert_eq!(
        normalized(&parse(&single_tool)),
        normalized(&parse(&router_tool)),
        "the MCP recall tool diverged between the router and the single instance"
    );
    // `initialize` hands out the shard's own manual — fetched through
    // the router, it must be the same text a shard serves directly.
    let initialize = json!({"jsonrpc": "2.0", "id": 9, "method": "initialize", "params": {}});
    let (_, single_init) = single.call("POST", "/mcp", Some(initialize.clone()));
    let (_, router_init) = router.call("POST", "/mcp", Some(initialize));
    assert_eq!(
        single_init["result"]["instructions"], router_init["result"]["instructions"],
        "the router's initialize must hand out the shards' manual"
    );

    // Deleting a context through the router routes to its shard and
    // the directory merge reflects it — writes are first-class, not a
    // replica-style refusal.
    assert_equivalent(&single, &router, "DELETE", "/contexts/breweries", None);
    assert_equivalent(&single, &router, "GET", "/contexts", None);
}

/// What has no single-instance analog: one shard of the fleet dies.
/// Fan-out reads degrade to labeled partials (`unreached` names the
/// shard, its direct contexts, and the transport error), routed verbs
/// and group surfaces refuse crisply with `shard_unreachable`, and the
/// fleet heals the moment the shard is back. Bearer auth passes
/// through the router untouched: the shards' keyring answers, the
/// router holds none.
#[test]
fn a_dead_shard_yields_labeled_partials_and_auth_passes_through() {
    let keyed = &[("TAGURU_API_TOKEN", "sesame")][..];
    let shard_a = Server::start_with_env("router-down-a", keyed);
    let shard_b = Server::start_with_env("router-down-b", keyed);
    let router = Server::start_router(
        "router-down",
        &format!("sake = {}\nglossary = {}\n", shard_a.base, shard_b.base),
        &[],
    );
    let token = Some("sesame");

    for (name, shard) in [("sake", &shard_a), ("glossary", &shard_b)] {
        let _ = shard;
        let (status, body) =
            router.call_with_token("PUT", &format!("/contexts/{name}"), None, token);
        assert_eq!(status, 200, "{body}");
        let (status, body) = router.call_with_token(
            "POST",
            &format!("/contexts/{name}/associations"),
            Some(json!([{"subject": "麹", "label": "関わる", "object": name,
                         "weight": 1.0, "source": "s"}])),
            token,
        );
        assert_eq!(status, 200, "{body}");
    }

    // Auth is the shards': no token → their 401 passes through the
    // router verbatim, fan-out and proxy alike.
    let (status, body) = router.call("POST", "/contexts/sake/recall", Some(json!({"cue": "麹"})));
    assert_eq!(status, 401, "{body}");
    let (status, body) = router.call(
        "POST",
        "/recall",
        Some(json!({"contexts": ["sake", "glossary"], "cue": "麹"})),
    );
    assert_eq!(status, 401, "{body}");

    // Healthy fleet: full fan-out, no unreached field at all.
    let (status, body) = router.call_with_token(
        "POST",
        "/recall",
        Some(json!({"contexts": ["sake", "glossary"], "cue": "麹"})),
        token,
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"]["total"], 2, "{body}");
    assert!(body.get("unreached").is_none(), "{body}");

    let shard_b_base = shard_b.base.clone();
    let glossary_dir = shard_b.stop_hard();

    // The fan-out degrades to a labeled partial: shard A's matches
    // arrive, the envelope names what could not be asked.
    let (status, body) = router.call_with_token(
        "POST",
        "/recall",
        Some(json!({"contexts": ["sake", "glossary"], "cue": "麹"})),
        token,
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"]["total"], 1, "{body}");
    let unreached = body["unreached"]
        .as_array()
        .expect("unreached must be labeled");
    assert_eq!(unreached.len(), 1, "{body}");
    assert_eq!(unreached[0]["contexts"], json!(["glossary"]), "{body}");
    assert!(
        unreached[0]["shard"]
            .as_str()
            .unwrap()
            .starts_with("http://"),
        "{body}"
    );

    // A routed verb aimed at the dead shard refuses crisply, naming
    // the shard and the context — retryable by design, never a hang.
    let (status, body) = router.call_with_token(
        "POST",
        "/contexts/glossary/recall",
        Some(json!({"cue": "麹"})),
        token,
    );
    assert_eq!(status, 502, "{body}");
    assert_eq!(body["code"], "shard_unreachable", "{body}");
    assert!(
        body["error"].as_str().unwrap().contains("glossary"),
        "{body}"
    );

    // Group surfaces never serve a partial union — a thinned member
    // list would look complete.
    let (status, body) = router.call_with_token("GET", "/groups", None, token);
    assert_eq!(status, 502, "{body}");
    assert_eq!(body["code"], "shard_unreachable", "{body}");

    // The directory stays useful: shard A's rows plus the label.
    let (status, body) = router.call_with_token("GET", "/contexts", None, token);
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"]["total"], 1, "{body}");
    assert_eq!(
        body["unreached"].as_array().map(Vec::len),
        Some(1),
        "{body}"
    );

    // The shard comes back on its own directory AND its own address —
    // the map names that address, so healing means returning to it.
    // The fleet heals with no router restart; the first calls may
    // still land on the router's stale pooled connections, so poll
    // briefly instead of asserting the very first answer.
    let shard_b_addr = shard_b_base.trim_start_matches("http://").to_string();
    let mut env = vec![("TAGURU_ADDR", shard_b_addr.as_str())];
    env.extend_from_slice(keyed);
    let _shard_b = Server::start_on_with_env("router-down-b2", glossary_dir, &env);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        let (status, body) = router.call_with_token(
            "POST",
            "/recall",
            Some(json!({"contexts": ["sake", "glossary"], "cue": "麹"})),
            token,
        );
        if status == 200 && body["result"]["total"] == 2 && body.get("unreached").is_none() {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the fleet never healed after the shard returned: {status} {body}"
        );
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}
