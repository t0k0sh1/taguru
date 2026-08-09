//! `taguru router` (issue #130): the stateless scatter-gather router.
//! The load-bearing property is EQUIVALENCE — the router over split
//! shards must answer what one instance holding the same contexts
//! answers, for every multi-context verb, merges and cursors and
//! groups and refusals alike — so the core test drives an identical
//! corpus into both topologies through their own front doors and
//! diffs the JSON, with only latency stamps and usage timestamps
//! normalized. The second test covers what has no single-instance
//! analog: a shard dying mid-fleet (labeled partials, pass-through
//! refusals, recovery), and bearer auth passing through untouched.

use std::time::{Duration, Instant};

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
                // A group's change token is topology-specific BY
                // DESIGN: each shard hashes the members it holds and
                // the router folds the shard tokens, so the VALUE
                // cannot equal the single instance's (it is already
                // scope-specific on one instance, too). The FIELD must
                // still exist on both sides — canonicalize the value,
                // never remove the key.
                if map.contains_key("fingerprint") {
                    map.insert("fingerprint".to_string(), json!("…"));
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
    // The passages ride the stream with a FIXED stored_at (which
    // import preserves, #167) rather than a later HTTP store: the
    // single instance and the router are seeded at different moments,
    // and a store-time stamp straddling a second boundary would make
    // the byte-for-byte export equivalence below flake. doc-a is
    // tagged, so the filtered cross search can prove the router
    // forwards the filter through its scatter-gather re-serialization;
    // both texts share 麹 for the rank-interleaved passage merge.
    let stream = concat!(
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-a\"}\n",
        "{\"passage\": \"麹と水で仕込む。\\n\\n辛口の酒は麹の使い方で決まる。\", \
          \"stored_at\": 1700000000, \"tags\": [\"仕込み\"]}\n",
        "{\"subject\": \"青嶺\", \"label\": \"銘柄である\", \"object\": \"酒\", \"weight\": 2.0}\n",
        "{\"subject\": \"辛口\", \"label\": \"特徴\", \"object\": \"酒\", \"weight\": 1.0}\n",
        "{\"subject\": \"共通\", \"label\": \"例\", \"object\": \"概念\", \"weight\": 0.5}\n",
        "{\"taguru_batch\": 1, \"context\": \"glossary\", \"source\": \"doc-b\"}\n",
        "{\"passage\": \"麹（こうじ）は蒸した米に麹菌を生やしたもの。\", \"stored_at\": 1700000001}\n",
        "{\"subject\": \"辛口\", \"label\": \"意味する\", \"object\": \"甘くない\", \"weight\": 2.0}\n",
        "{\"subject\": \"共通\", \"label\": \"例\", \"object\": \"概念\", \"weight\": 0.5}\n",
        "{\"taguru_batch\": 1, \"context\": \"breweries\", \"source\": \"doc-c\"}\n",
        "{\"subject\": \"青嶺酒造\", \"label\": \"造る\", \"object\": \"青嶺\", \"weight\": -2.5}\n",
        // A schema record (ADR 0009 §13, #384): sake lives on shard A —
        // this proves the router's OWN routing table for schema
        // records (never broadcast, unlike groups) sends it to the
        // right shard rather than reusing whatever shard a nearby
        // batch chunk happened to land on.
        "{\"taguru_schema\": 1, \"context\": \"sake\", \"mode\": \"warn\", \
          \"closed_labels\": false, \"types\": {}, \"relations\": {}}\n",
        "{\"taguru_group\": 1, \"name\": \"jp\", \"description\": \"日本酒\", \"contexts\": [\"sake\", \"glossary\"]}\n",
    );
    let (status, outcome) = post_import(server, stream, None);
    assert_eq!(status, 200, "{outcome}");
    assert_eq!(
        outcome["result"]["schemas"][0]["context"],
        json!("sake"),
        "the router's own schema-routing table must land the record and report it: {outcome}"
    );
    // A nested group whose direct member and child live on different
    // shards: `contexts` needs projection, `groups` broadcasts whole.
    server.ok(
        "PUT",
        "/groups/all",
        Some(json!({"description": "全部", "contexts": ["breweries"], "groups": ["jp"]})),
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

    // The schema record's own routing (ADR 0009 §13, #384): the
    // router sent it to shard A alone (never broadcast, unlike
    // groups) — content served back through GET must still match the
    // single instance exactly.
    assert_equivalent(&single, &router, "GET", "/contexts/sake/schema", None);

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
        assert!(
            !answer["result"]["hits"].as_array().unwrap().is_empty(),
            "{answer}"
        );
        assert!(
            !answer["result"]["plan"]["contexts"]
                .as_array()
                .unwrap()
                .is_empty(),
            "the merged plan must match the single instance's byte for byte: {answer}"
        );
    }

    // The source filter (#167) rides the router's own scatter-gather
    // re-serialization: only the tagged shard answers, and each
    // target's plan carries its eligibility counts — identically to
    // the single instance.
    let filtered = assert_equivalent(
        &single,
        &router,
        "POST",
        "/sources/search",
        Some(json!({"contexts": ["sake", "glossary"], "query": "麹の使い方", "tags": ["仕込み"]})),
    );
    let hits = filtered["result"]["hits"].as_array().unwrap();
    assert!(
        !hits.is_empty() && hits.iter().all(|hit| hit["context"] == "sake"),
        "only the tagged source's context may answer through the router: {filtered}"
    );
    assert_eq!(
        filtered["result"]["plan"]["contexts"][1]["filter"],
        json!({"eligible_sources": 0, "total_sources": 1}),
        "the untagged target reports an empty eligible set: {filtered}"
    );

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

    // The router's folded fingerprint honors the same contract as a
    // shard's own (its VALUE is topology-specific — see `normalized` —
    // but a member write on either shard must still move it). Mirror
    // the write into the single instance so the corpora stay equal for
    // the assertions below.
    let fingerprint_before = router.ok("GET", "/groups/jp", None)["fingerprint"]
        .as_str()
        .unwrap()
        .to_string();
    let member_write =
        json!([{"subject": "甘口", "label": "意味する", "object": "甘い", "weight": 1.0}]);
    router.ok(
        "POST",
        "/contexts/glossary/associations",
        Some(member_write.clone()),
    );
    single.ok(
        "POST",
        "/contexts/glossary/associations",
        Some(member_write),
    );
    let fingerprint_after = router.ok("GET", "/groups/jp", None)["fingerprint"]
        .as_str()
        .unwrap()
        .to_string();
    assert_ne!(
        fingerprint_before, fingerprint_after,
        "a member write on a shard must move the router's folded token"
    );

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
    // Two keys on every shard: an unscoped one for the test's own
    // driving, and one granted `sake` only — the scoped-import case
    // below needs it.
    let keyed = &[
        ("TAGURU_API_TOKENS", "ops:sesame,limited:hush"),
        (
            "TAGURU_KEY_SCOPES",
            r#"{"limited": {"role": "write", "contexts": ["sake"]}}"#,
        ),
    ][..];
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

    // A scoped key whose stream carries an out-of-grant GROUP record:
    // a single instance scope-checks the record's closure before
    // anything applies and answers 403 with nothing landed. The
    // router's preflight must keep that — the in-grant batch ahead of
    // the record must NOT have been applied when the refusal comes
    // back from the group projection on another shard.
    let stream = concat!(
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"scoped-doc\"}\n",
        "{\"subject\": \"密造\", \"label\": \"は\", \"object\": \"だめ\", \"weight\": 1.0}\n",
        "{\"taguru_group\": 1, \"name\": \"overreach\", \"contexts\": [\"sake\", \"glossary\"]}\n",
    );
    let (status, body) = post_import(&router, stream, Some("hush"));
    assert_eq!(status, 403, "{body}");
    assert_eq!(body["code"], "forbidden", "{body}");
    let (status, body) = router.call_with_token(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "密造"})),
        token,
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["result"]["total"], 0,
        "the refused stream's batch must not have landed: {body}"
    );

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

/// The router answers its own `/health` locally rather than proxying
/// to a shard (ADR 0002 §10) — its `version` names the router binary
/// itself, beside the existing `router`/`shards` fields.
#[test]
fn the_router_health_names_its_own_version() {
    let shard = Server::start("router-health-shard");
    let router = Server::start_router("router-health", &format!("sake = {}\n", shard.base), &[]);

    let (status, body) = router.call("GET", "/health", None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["status"], json!("ok"), "{body}");
    assert_eq!(body["router"], json!(true), "{body}");
    assert_eq!(body["shards"], json!(1), "{body}");
    assert_eq!(body["version"], json!(env!("CARGO_PKG_VERSION")), "{body}");
}

/// Issue #248 item 9: `route` is a deprecated alias for `router`, not
/// just a `--help` synonym — it must dispatch into the exact same
/// running router, answering `/health` identically.
#[test]
fn the_route_alias_dispatches_into_a_real_router_identically_to_router() {
    let shard = Server::start("route-alias-shard");
    let router = Server::start_router_via(
        "route-alias",
        &format!("sake = {}\n", shard.base),
        &[],
        "route",
    );

    let (status, body) = router.call("GET", "/health", None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["status"], json!("ok"), "{body}");
    assert_eq!(body["router"], json!(true), "{body}");
    assert_eq!(body["shards"], json!(1), "{body}");
}

/// `result.schemas` must answer in STREAM order, not shard-number
/// order — `RouteMap` numbers shards by first appearance in the map
/// file, which is independent of a stream's own record order. The
/// map here deliberately lists shard B's URL first (so it becomes
/// shard 0), then sends a stream whose FIRST `taguru_schema` record
/// names a shard-A context and whose SECOND names a shard-B one — the
/// two orders disagree, so a router that iterated by shard number
/// instead of original stream index would answer `[ctx_b, ctx_a]`
/// instead of the correct `[ctx_a, ctx_b]`.
#[test]
fn schema_outcomes_answer_in_stream_order_not_shard_number_order() {
    let shard_a = Server::start("router-schema-order-a");
    let shard_b = Server::start("router-schema-order-b");
    let router = Server::start_router(
        "router-schema-order",
        &format!("ctx_b = {}\nctx_a = {}\n", shard_b.base, shard_a.base),
        &[],
    );

    router.ok("PUT", "/contexts/ctx_a", Some(json!({})));
    router.ok("PUT", "/contexts/ctx_b", Some(json!({})));

    let stream = concat!(
        "{\"taguru_schema\": 1, \"context\": \"ctx_a\", \"mode\": \"warn\", \
         \"closed_labels\": false, \"types\": {}, \"relations\": {}}\n",
        "{\"taguru_schema\": 1, \"context\": \"ctx_b\", \"mode\": \"warn\", \
         \"closed_labels\": false, \"types\": {}, \"relations\": {}}\n",
    );
    let (status, body) = post_import(&router, stream, None);
    assert_eq!(status, 200, "{body}");
    let schemas = body["result"]["schemas"].as_array().expect("schemas array");
    assert_eq!(
        schemas
            .iter()
            .map(|s| s["context"].clone())
            .collect::<Vec<_>>(),
        vec![json!("ctx_a"), json!("ctx_b")],
        "{body}"
    );
}

/// A router rewrap (a schema record already landed on one shard, then
/// a later one refuses on another) must keep the refusal's
/// STRUCTURED detail, not collapse it into prose the failing shard's
/// own note already carried: `integrity`/`durable_batches` are
/// recomputed from the true cross-shard counts (the failing shard's
/// own view of "0 landed" would otherwise under-report), and `issues`
/// rides through unedited.
#[test]
fn a_router_rewrap_keeps_structured_refusal_detail() {
    let shard_a = Server::start("router-schema-rewrap-a");
    let shard_b = Server::start("router-schema-rewrap-b");
    let router = Server::start_router(
        "router-schema-rewrap",
        &format!("ctx_ok = {}\nghost = {}\n", shard_a.base, shard_b.base),
        &[],
    );
    router.ok("PUT", "/contexts/ctx_ok", Some(json!({})));
    // `ghost` is intentionally never created — the second schema
    // record's context does not exist on its own shard.

    let stream = concat!(
        "{\"taguru_schema\": 1, \"context\": \"ctx_ok\", \"mode\": \"warn\", \
         \"closed_labels\": false, \"types\": {}, \"relations\": {}}\n",
        "{\"taguru_schema\": 1, \"context\": \"ghost\", \"mode\": \"warn\", \
         \"closed_labels\": false, \"types\": {}, \"relations\": {}}\n",
    );
    let (status, body) = post_import(&router, stream, None);
    assert_eq!(status, 404, "{body}");
    assert_eq!(
        body["integrity"],
        json!("durable_prefix"),
        "ctx_ok's schema landed on shard_a before ghost refused on shard_b: {body}"
    );
    assert!(
        body.get("durable_batches").is_none(),
        "durable_batches names batches — this stream carried none: {body}"
    );
    assert!(
        body["issues"]
            .as_array()
            .is_some_and(|issues| !issues.is_empty()),
        "the failing shard's own issue (ghost's missing context) must ride through: {body}"
    );
    assert_eq!(body["code"], json!("no_context"), "{body}");

    // The shard_a install really did land — proof the rewrap's
    // "durable_prefix" claim is true, not just structurally present.
    let installed = router.ok("GET", "/contexts/ctx_ok/schema", None);
    assert_eq!(installed["mode"], "warn", "{installed}");
}

// ---------------------------------------------------------------------------
// Route-map hot reload (issue #515)

/// Polls until `check` passes or the budget elapses — reloads are
/// asynchronous with respect to the signal / file write that asks for
/// them (same discipline as reload.rs's keyring tests).
fn eventually(budget: Duration, what: &str, mut check: impl FnMut() -> bool) {
    let deadline = Instant::now() + budget;
    while Instant::now() < deadline {
        if check() {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    panic!("timed out waiting for {what}");
}

/// One outcome's count out of the router's
/// `taguru_router_map_reloads_total` metric; 0 when the series has
/// not appeared yet.
fn map_reload_count(router: &Server, outcome: &str) -> u64 {
    let (_, body) = router.call("GET", "/metrics", None);
    let prefix = format!("taguru_router_map_reloads_total{{outcome=\"{outcome}\"}} ");
    body.as_str()
        .unwrap_or_default()
        .lines()
        .find_map(|line| line.strip_prefix(prefix.as_str()))
        .and_then(|count| count.parse().ok())
        .unwrap_or(0)
}

/// SIGHUP swaps a rewritten map without a restart: the context's
/// verbs re-route to the shard the new map names. A broken rewrite
/// first: refused whole (counted on /metrics), the old map keeps
/// serving — the same fail-closed shape as the keyring reload. The
/// map-file watch polls every ~5s, so it may fire alongside any
/// SIGHUP these tests send after a rewrite; counter assertions are
/// therefore `>=`, never `==`.
#[cfg(unix)]
#[test]
fn sighup_swaps_the_route_map_and_a_broken_edit_keeps_the_old_map() {
    let shard_a = Server::start("map-hup-a");
    let shard_b = Server::start("map-hup-b");
    let router = Server::start_router("map-hup", &format!("moved = {}\n", shard_a.base), &[]);
    router.ok(
        "PUT",
        "/contexts/moved",
        Some(json!({"description": "hot-reload target"})),
    );

    // The broken edit: refused whole, old map still routing.
    let map_path = router.data_dir.join("route-map");
    std::fs::write(&map_path, "moved http://no-equals\n").unwrap();
    router.signal("-HUP");
    eventually(
        Duration::from_secs(10),
        "the refused reload to be counted",
        || map_reload_count(&router, "refused") >= 1,
    );
    let (status, body) = router.call("GET", "/contexts/moved", None);
    assert_eq!(
        status, 200,
        "a refused reload must keep the old map serving: {body}"
    );

    // The real edit: 'moved' now routes to shard B, which never held
    // it — the router answers B's own 404 — while shard A still holds
    // the data, addressed directly.
    std::fs::write(&map_path, format!("moved = {}\n", shard_b.base)).unwrap();
    router.signal("-HUP");
    eventually(
        Duration::from_secs(10),
        "the rewritten map to take over routing",
        || router.call("GET", "/contexts/moved", None).0 == 404,
    );
    assert_eq!(shard_a.call("GET", "/contexts/moved", None).0, 200);
    assert!(map_reload_count(&router, "applied") >= 1);
}

/// The map-file watch alone — no signal anywhere — picks up a
/// rewrite, mirroring the ConfigMap/secret-volume flow where nothing
/// can send SIGHUP into the pod.
#[test]
fn the_map_file_watch_swaps_routing_with_no_signal() {
    let shard_a = Server::start("map-watch-a");
    let shard_b = Server::start("map-watch-b");
    let router = Server::start_router("map-watch", &format!("moved = {}\n", shard_a.base), &[]);
    router.ok(
        "PUT",
        "/contexts/moved",
        Some(json!({"description": "watch target"})),
    );
    std::fs::write(
        router.data_dir.join("route-map"),
        format!("moved = {}\n", shard_b.base),
    )
    .unwrap();
    // The watch polls every ~5s; the budget gives it two cycles plus
    // scheduling slack.
    eventually(
        Duration::from_secs(15),
        "the watch to swap the map with no signal",
        || router.call("GET", "/contexts/moved", None).0 == 404,
    );
}

/// One shard-outcome count out of the router's
/// `taguru_router_shard_requests_total` metric — keyed by shard URL
/// (issue #515: a map reload renumbers indices, so the URL is the
/// only stable identity); 0 when the series has not appeared yet.
fn shard_request_count(router: &Server, shard_url: &str, outcome: &str) -> u64 {
    let (_, body) = router.call("GET", "/metrics", None);
    let prefix = format!(
        "taguru_router_shard_requests_total{{shard=\"{shard_url}\",outcome=\"{outcome}\"}} "
    );
    body.as_str()
        .unwrap_or_default()
        .lines()
        .find_map(|line| line.strip_prefix(prefix.as_str()))
        .and_then(|count| count.parse().ok())
        .unwrap_or(0)
}

/// A context answered by two shards is a mid-move stray: the map's
/// owner must win the merged directory row whichever shard answered
/// first, and the duplicate must leave the total. Both orders are
/// exercised — ctx-a's owner answers before its stray, ctx-b's after
/// — because keep-vs-replace are different arms of the same guard.
#[test]
fn merge_contexts_dedups_a_mid_move_stray_by_map_ownership() {
    let shard_a = Server::start("stray-a");
    let shard_b = Server::start("stray-b");
    let router = Server::start_router(
        "stray",
        &format!("ctx-a = {}\nctx-b = {}\n", shard_a.base, shard_b.base),
        &[],
    );
    // Each context exists on BOTH shards (a mid-move leftover), with
    // descriptions naming the copy so the winner is observable.
    for (shard, tag) in [(&shard_a, "A"), (&shard_b, "B")] {
        for name in ["ctx-a", "ctx-b"] {
            shard.ok(
                "PUT",
                &format!("/contexts/{name}"),
                Some(json!({"description": format!("{name}@{tag}")})),
            );
        }
    }
    let listing = router.ok("GET", "/contexts", None);
    assert_eq!(
        listing["total"],
        json!(2),
        "each stray must leave the total: {listing}"
    );
    let description_of = |name: &str| -> String {
        listing["contexts"]
            .as_array()
            .expect("directory rows")
            .iter()
            .find(|entry| entry["name"] == json!(name))
            .unwrap_or_else(|| panic!("{name} missing from {listing}"))["description"]
            .as_str()
            .unwrap()
            .to_string()
    };
    assert_eq!(description_of("ctx-a"), "ctx-a@A");
    assert_eq!(description_of("ctx-b"), "ctx-b@B");
}

/// The operator surface through the router: `/protocol` proxies the
/// first shard's manual, `/flush` and `/maintenance/compact`
/// broadcast and merge, and the per-shard counters key on the shard
/// URL. Thin assertions on purpose — each answer's SHAPE proves the
/// handler ran rather than defaulting to an empty 200.
#[test]
fn operator_verbs_broadcast_and_shard_metrics_key_on_the_url() {
    let shard_a = Server::start("operator-a");
    let shard_b = Server::start("operator-b");
    let router = Server::start_router(
        "operator",
        &format!("sake = {}\n* = {}\n", shard_a.base, shard_b.base),
        &[],
    );
    router.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "銘柄"})),
    );

    let (status, manual) = router.call("GET", "/protocol", None);
    assert_eq!(status, 200);
    assert!(
        manual.as_str().is_some_and(|text| text.contains("taguru")),
        "the proxied manual must be the shard's own text: {manual}"
    );

    let flushed = router.ok("POST", "/flush", None);
    assert!(
        flushed.is_array(),
        "flush merges the shard lists: {flushed}"
    );

    let swept = router.ok("POST", "/maintenance/compact", None);
    assert!(
        swept["contexts"].is_array(),
        "compact merges per-context outcomes: {swept}"
    );

    assert!(
        shard_request_count(&router, &shard_a.base, "ok") >= 1,
        "the sake shard's successes must count under its URL"
    );
}

/// Group delete broadcasts through the router and answers the
/// single-instance `result: true`; a body no JSON parser accepts goes
/// to one shard verbatim so the refusal is the shard's own extractor
/// shape, not a router-invented one.
#[test]
fn group_delete_broadcasts_and_a_malformed_body_gets_the_shards_refusal() {
    let shard_a = Server::start("gdel-a");
    let shard_b = Server::start("gdel-b");
    let router = Server::start_router(
        "gdel",
        &format!("sake = {}\n* = {}\n", shard_a.base, shard_b.base),
        &[],
    );
    router.ok("PUT", "/groups/g", Some(json!({"description": "対象"})));
    let deleted = router.ok("DELETE", "/groups/g", None);
    assert_eq!(deleted, json!(true), "{deleted}");
    assert_eq!(router.call("GET", "/groups/g", None).0, 404);

    let (status, refusal) = router.call_raw(
        "PUT",
        "/groups/probe",
        Some("not json"),
        Some("application/json"),
    );
    assert_eq!(status, 400, "{refusal}");
    assert!(
        refusal["code"].is_string(),
        "the shard's own refusal shape must pass through: {refusal}"
    );
}
