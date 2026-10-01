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

/// The ids `seed`'s stream gives its three contexts — fixed, so the
/// single instance and the sharded fleet carry the same ones.
const SAKE_ID: &str = "cef2e28b-43f0-4b6c-8201-abab0785399f";
const GLOSSARY_ID: &str = "3f5dcb46-d438-4c49-bd38-6708b01a8d0a";
const BREWERIES_ID: &str = "96ba89e7-84ee-4b65-b772-bdf9ae741e0d";
/// Canonical ids no test creates, distinct from each other.
const ABSENT_A: &str = "00000000-0000-4000-8000-0000000000a1";
const ABSENT_B: &str = "00000000-0000-4000-8000-0000000000a2";

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
                // Context ids are minted per data directory (#964), so
                // the single instance and the fleet can never agree on
                // them — canonicalize UUID-shaped ids, keep the field.
                if let Some(id) = map.get("id").and_then(Value::as_str)
                    && id.len() == 36
                    && id.as_bytes()[8] == b'-'
                {
                    map.insert("id".to_string(), json!("<uuid>"));
                }
                // An error message quoting an id-addressed path (the
                // unknown-path 404) diverges the same way: mask every
                // UUID-shaped run — 36 bytes, hyphens at the fixed
                // offsets, hex elsewhere. Ids are ASCII, so a byte
                // scan can't split a multi-byte char mid-mask.
                if let Some(error) = map.get("error").and_then(Value::as_str) {
                    let bytes = error.as_bytes();
                    let mut out = String::new();
                    let mut index = 0;
                    while index < bytes.len() {
                        let uuid_shaped = index + 36 <= bytes.len()
                            && bytes[index..index + 36]
                                .iter()
                                .enumerate()
                                .all(|(offset, byte)| match offset {
                                    8 | 13 | 18 | 23 => *byte == b'-',
                                    _ => byte.is_ascii_hexdigit(),
                                });
                        if uuid_shaped {
                            out.push_str("<uuid>");
                            index += 36;
                        } else {
                            // Copy the full char, not just one byte.
                            let rest = &error[index..];
                            let ch = rest.chars().next().expect("in-bounds index");
                            out.push(ch);
                            index += ch.len_utf8();
                        }
                    }
                    map.insert("error".to_string(), json!(out));
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
    // The contexts are created by the stream's own create blocks, under
    // fixed ids: the header names a context by id (#965), and the
    // create block's NAME is what the router's route map places on a
    // shard. The same ids on both topologies keep every response
    // comparable without canonicalizing them.
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
        "{\"type\": \"source\", \"context_id\": \"cef2e28b-43f0-4b6c-8201-abab0785399f\", \"id\": \"doc-a\", \"create\": {\"name\": \"sake\", \"description\": \"銘柄と蔵元の知識\"}}\n",
        "{\"passage\": \"麹と水で仕込む。\\n\\n辛口の酒は麹の使い方で決まる。\", \
          \"stored_at\": 1700000000, \"tags\": [\"仕込み\"]}\n",
        "{\"subject\": \"青嶺\", \"label\": \"銘柄である\", \"object\": \"酒\", \"weight\": 2.0}\n",
        "{\"subject\": \"辛口\", \"label\": \"特徴\", \"object\": \"酒\", \"weight\": 1.0}\n",
        "{\"subject\": \"共通\", \"label\": \"例\", \"object\": \"概念\", \"weight\": 0.5}\n",
        "{\"type\": \"source\", \"context_id\": \"3f5dcb46-d438-4c49-bd38-6708b01a8d0a\", \"id\": \"doc-b\", \"create\": {\"name\": \"glossary\", \"description\": \"酒の用語集\"}}\n",
        "{\"passage\": \"麹（こうじ）は蒸した米に麹菌を生やしたもの。\", \"stored_at\": 1700000001}\n",
        "{\"subject\": \"辛口\", \"label\": \"意味する\", \"object\": \"甘くない\", \"weight\": 2.0}\n",
        "{\"subject\": \"共通\", \"label\": \"例\", \"object\": \"概念\", \"weight\": 0.5}\n",
        "{\"type\": \"source\", \"context_id\": \"96ba89e7-84ee-4b65-b772-bdf9ae741e0d\", \"id\": \"doc-c\", \"create\": {\"name\": \"breweries\", \"description\": \"蔵元の台帳\"}}\n",
        "{\"subject\": \"青嶺酒造\", \"label\": \"造る\", \"object\": \"青嶺\", \"weight\": -2.5}\n",
        // A schema record (ADR 0009 §13, #384): sake lives on shard A —
        // this proves the router's OWN routing table for schema
        // records (never broadcast, unlike groups) sends it to the
        // right shard rather than reusing whatever shard a nearby
        // batch chunk happened to land on.
        "{\"type\": \"schema\", \"context_id\": \"cef2e28b-43f0-4b6c-8201-abab0785399f\", \"mode\": \"warn\", \
          \"closed_labels\": false, \"types\": {}, \"relations\": {}}\n",
        "{\"type\": \"group\", \"id\": \"jp\", \"description\": \"日本酒\", \"context_ids\": [\"cef2e28b-43f0-4b6c-8201-abab0785399f\", \"3f5dcb46-d438-4c49-bd38-6708b01a8d0a\"]}\n",
    );
    let (status, outcome) = post_import(server, stream, None);
    assert_eq!(status, 200, "{outcome}");
    assert_eq!(
        outcome["result"]["schemas"][0]["context_id"],
        json!("cef2e28b-43f0-4b6c-8201-abab0785399f"),
        "the router's own schema-routing table must land the record and report it: {outcome}"
    );
    // A nested group whose direct member and child live on different
    // shards: `contexts` needs projection, `groups` broadcasts whole.
    server.ok(
        "PUT",
        "/groups/all",
        Some(json!({"description": "全部", "context_ids": [BREWERIES_ID], "groups": ["jp"]})),
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
    assert_equivalent_at(single, router, method, path, path, body)
}

/// [`assert_equivalent`] for id-addressed paths, which spell the same
/// context differently per front door (#964: ids are minted per data
/// directory) — everything else identical.
fn assert_equivalent_at(
    single: &Server,
    router: &Server,
    method: &str,
    single_path: &str,
    router_path: &str,
    body: Option<Value>,
) -> Value {
    let path = router_path;
    let (single_status, single_body) = single.call(method, single_path, body.clone());
    let (router_status, router_body) = router.call(method, router_path, body);
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
    assert_equivalent_at(
        &single,
        &router,
        "GET",
        &format!("/contexts/{}/schema", single.cx("sake")),
        &format!("/contexts/{}/schema", router.cx_http("sake")),
        None,
    );

    // The seeding itself already proved the import split: now the
    // responses. Cross recall with contexts, with groups (nested),
    // and mixed.
    for body in [
        json!({"context_ids": [SAKE_ID, BREWERIES_ID, GLOSSARY_ID], "cue": "青嶺"}),
        json!({"groups": ["all"], "cue": "辛口"}),
        json!({"context_ids": [BREWERIES_ID], "groups": ["jp"], "cue": "青嶺"}),
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
        "context_id": last["context_id"],
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
        json!({"context_ids": [SAKE_ID, GLOSSARY_ID, BREWERIES_ID], "query": "麹"}),
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
        Some(
            json!({"context_ids": [SAKE_ID, GLOSSARY_ID], "query": "麹の使い方", "tags": ["仕込み"]}),
        ),
    );
    let hits = filtered["result"]["hits"].as_array().unwrap();
    assert!(
        !hits.is_empty() && hits.iter().all(|hit| hit["context_name"] == "sake"),
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
    // `limit=0` is a legitimate keyset query and must floor to a
    // one-row page on both sides (`clamp_page`) — an empty page reads
    // as end-of-collection to the SDK iterators.
    let floored = assert_equivalent(&single, &router, "GET", "/contexts?limit=0", None);
    assert!(
        !floored["result"]["contexts"].as_array().unwrap().is_empty(),
        "{floored}"
    );
    let floored = assert_equivalent(&single, &router, "GET", "/groups?limit=0", None);
    assert!(
        !floored["result"]["groups"].as_array().unwrap().is_empty(),
        "{floored}"
    );
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
        &format!("/contexts/{}/associations", router.cx_http("glossary")),
        Some(member_write.clone()),
    );
    single.ok(
        "POST",
        &format!("/contexts/{}/associations", single.cx("glossary")),
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
        Some(json!({"remove_context_ids": [GLOSSARY_ID]})),
    );
    assert_equivalent(
        &single,
        &router,
        "PATCH",
        "/groups/jp",
        Some(json!({"add_context_ids": [GLOSSARY_ID]})),
    );
    // Removals are an idempotent set difference on a single instance —
    // never existence-checked — so a name no shard places must not be
    // refused by the router's projection either.
    assert_equivalent(
        &single,
        &router,
        "PATCH",
        "/groups/jp",
        Some(json!({"remove_context_ids": [GHOST_ID]})),
    );
    // ADDITIONS keep the existence check — `project_body`'s refusal
    // for an unplaced member must be the shard's own nonexistent-member
    // answer, byte for byte, and leave the group unchanged.
    assert_equivalent(
        &single,
        &router,
        "PATCH",
        "/groups/jp",
        Some(json!({"add_context_ids": [GHOST_ID]})),
    );
    assert_equivalent(&single, &router, "GET", "/groups/jp", None);

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
        Some(json!({"context_ids": [SAKE_ID, ABSENT_A, ABSENT_B], "cue": "x"})),
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
    let single_sake = single.cx("sake");
    let router_sake = router.cx_http("sake");
    assert_equivalent_at(
        &single,
        &router,
        "POST",
        &format!("/contexts/{single_sake}/recall"),
        &format!("/contexts/{router_sake}/recall"),
        Some(json!({"cue": "青嶺"})),
    );
    assert_equivalent_at(
        &single,
        &router,
        "GET",
        &format!("/contexts/{single_sake}/export"),
        &format!("/contexts/{router_sake}/export"),
        None,
    );
    assert_equivalent_at(
        &single,
        &router,
        "POST",
        &format!("/contexts/{single_sake}/unknown-verb"),
        &format!("/contexts/{router_sake}/unknown-verb"),
        None,
    );

    // The MCP transport over the router: the same tool call answers
    // the same content (the tool text is the response JSON — parse it
    // and normalize the latency stamp inside).
    let single_tool = single.call_tool(
        1,
        "recall",
        json!({"context_ids": [SAKE_ID, GLOSSARY_ID], "cue": "辛口"}),
    );
    let router_tool = router.call_tool(
        1,
        "recall",
        json!({"context_ids": [SAKE_ID, GLOSSARY_ID], "cue": "辛口"}),
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
    assert_equivalent_at(
        &single,
        &router,
        "DELETE",
        &format!("/contexts/{}", single.cx("breweries")),
        &format!("/contexts/{}", router.cx_http("breweries")),
        None,
    );
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
    // Two keys on every shard: one with no grant entry for the test's
    // own driving, and one granted `sake` only — the scoped-import case
    // below needs it.
    let keyed = &[
        ("TAGURU_API_TOKENS", "ops:sesame,limited:hush"),
        (
            "TAGURU_KEY_GRANTS",
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
        let (status, body) =
            router.call_with_token("POST", "/contexts", Some(json!({"name": name})), token);
        assert_eq!(status, 200, "{body}");
        let (status, body) = router.call_with_token(
            "POST",
            &format!("/contexts/{}/associations", shard.cx(name)),
            Some(json!([{"subject": "麹", "label": "関わる", "object": name,
                         "weight": 1.0, "source": "s"}])),
            token,
        );
        assert_eq!(status, 200, "{body}");
    }
    let sake_id = shard_a.cx("sake");
    let glossary_id = shard_b.cx("glossary");

    // Auth is the shards': no token → their 401 passes through the
    // router verbatim, fan-out and proxy alike.
    let (status, body) = router.call(
        "POST",
        &format!("/contexts/{sake_id}/recall"),
        Some(json!({"cue": "麹"})),
    );
    assert_eq!(status, 401, "{body}");
    let (status, body) = router.call(
        "POST",
        "/recall",
        Some(json!({"context_ids": [sake_id, glossary_id], "cue": "麹"})),
    );
    assert_eq!(status, 401, "{body}");

    // Healthy fleet: full fan-out, no unreached field at all.
    let (status, body) = router.call_with_token(
        "POST",
        "/recall",
        Some(json!({"context_ids": [sake_id, glossary_id], "cue": "麹"})),
        token,
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"]["total"], 2, "{body}");
    assert!(body.get("unreached").is_none(), "{body}");

    // A context-scoped key whose stream carries an out-of-grant GROUP record:
    // a single instance scope-checks the record's closure before
    // anything applies and answers 403 with nothing landed. The
    // router's preflight must keep that — the in-grant batch ahead of
    // the record must NOT have been applied when the refusal comes
    // back from the group projection on another shard.
    let stream = format!(
        "{{\"type\": \"source\", \"context_id\": \"{sake_id}\", \"id\": \"scoped-doc\"}}\n\
         {{\"subject\": \"密造\", \"label\": \"は\", \"object\": \"だめ\", \"weight\": 1.0}}\n\
         {{\"type\": \"group\", \"id\": \"overreach\", \"context_ids\": [\"{sake_id}\", \"{glossary_id}\"]}}\n",
    );
    let (status, body) = post_import(&router, &stream, Some("hush"));
    assert_eq!(status, 403, "{body}");
    assert_eq!(body["code"], "forbidden", "{body}");
    let (status, body) = router.call_with_token(
        "POST",
        &format!("/contexts/{sake_id}/query"),
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
        Some(json!({"context_ids": [sake_id, glossary_id], "cue": "麹"})),
        token,
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"]["total"], 1, "{body}");
    let unreached = body["unreached"]
        .as_array()
        .expect("unreached must be labeled");
    assert_eq!(unreached.len(), 1, "{body}");
    assert_eq!(unreached[0]["contexts"], json!([glossary_id]), "{body}");
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
        &format!("/contexts/{glossary_id}/recall"),
        Some(json!({"cue": "麹"})),
        token,
    );
    assert_eq!(status, 502, "{body}");
    assert_eq!(body["code"], "shard_unreachable", "{body}");
    assert!(
        body["error"].as_str().unwrap().contains(&glossary_id),
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
            Some(json!({"context_ids": [sake_id, glossary_id], "cue": "麹"})),
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
/// shard 0), then sends a stream whose FIRST `schema` record
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

    let ctx_a = router.ok("POST", "/contexts", Some(json!({"name": "ctx_a", })))["id"].clone();
    let ctx_b = router.ok("POST", "/contexts", Some(json!({"name": "ctx_b", })))["id"].clone();

    let schema_line = |id: &Value| {
        format!(
            "{{\"type\": \"schema\", \"context_id\": {id}, \"mode\": \"warn\", \
             \"closed_labels\": false, \"types\": {{}}, \"relations\": {{}}}}\n"
        )
    };
    let stream = format!("{}{}", schema_line(&ctx_a), schema_line(&ctx_b));
    let (status, body) = post_import(&router, &stream, None);
    assert_eq!(status, 200, "{body}");
    let schemas = body["result"]["schemas"].as_array().expect("schemas array");
    assert_eq!(
        schemas
            .iter()
            .map(|s| s["context_id"].clone())
            .collect::<Vec<_>>(),
        vec![ctx_a, ctx_b],
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
    let ctx_ok = router.ok("POST", "/contexts", Some(json!({"name": "ctx_ok", })))["id"].clone();
    // The second schema record names an id no shard holds — the router
    // hands it to the first shard (the one holding `ctx_ok`), which
    // refuses it mid-apply, after the first record landed.
    let ghost = "ead6ef03-d61e-460c-933d-6d450c50a1e5";

    let stream = format!(
        "{{\"type\": \"schema\", \"context_id\": {ctx_ok}, \"mode\": \"warn\", \
         \"closed_labels\": false, \"types\": {{}}, \"relations\": {{}}}}\n\
         {{\"type\": \"schema\", \"context_id\": \"{ghost}\", \"mode\": \"warn\", \
         \"closed_labels\": false, \"types\": {{}}, \"relations\": {{}}}}\n"
    );
    let (status, body) = post_import(&router, &stream, None);
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
    let installed = router.ok(
        "GET",
        &format!("/contexts/{}/schema", shard_a.cx("ctx_ok")),
        None,
    );
    assert_eq!(installed["mode"], "warn", "{installed}");
}

/// `POST /maintenance/compact` on a fleet where NO shard can be
/// reached must refuse like `POST /flush` does, never answer an
/// empty 200 sweep report that reads as "nothing needed compacting".
#[test]
fn maintenance_refuses_when_every_shard_is_unreachable() {
    // Port 1 needs privileges to bind, so a dial is refused — a dead
    // shard with no port-reuse race.
    let router = Server::start_router("maint-all-down", "sake = http://127.0.0.1:1\n", &[]);

    let (status, body) = router.call("POST", "/maintenance/compact", None);
    assert_eq!(status, 502, "{body}");
    assert_eq!(body["code"], "shard_unreachable", "{body}");
    // The sibling verb's guard, for symmetry.
    let (status, body) = router.call("POST", "/flush", None);
    assert_eq!(status, 502, "{body}");
    assert_eq!(body["code"], "shard_unreachable", "{body}");
}

/// A group record already present on one shard and created EMPTY on
/// the rest (drift healing) replaces nothing in the union — the merged
/// outcome must say "unchanged"; a created projection that actually
/// carries members is what earns "replaced".
#[test]
fn group_import_outcome_reflects_the_union_not_any_one_shard() {
    let shard_a = Server::start("gimp-outcome-a");
    let shard_b = Server::start("gimp-outcome-b");
    let router = Server::start_router(
        "gimp-outcome",
        &format!("ctx_a = {}\nctx_b = {}\n", shard_a.base, shard_b.base),
        &[],
    );
    let ctx_a = router.create_context("ctx_a");
    let ctx_b = router.create_context("ctx_b");

    // A record NO shard holds answers "created" — every projection is
    // created, and a created projection carrying members must not slip
    // into the "replaced" arm (this pins the branch order too).
    let stream =
        &format!("{{\"type\": \"group\", \"id\": \"g0\", \"context_ids\": [\"{ctx_a}\"]}}\n");
    let (status, body) = post_import(&router, stream, None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["result"]["groups"][0]["outcome"],
        json!("created"),
        "a brand-new record is created everywhere: {body}"
    );

    // Shard A already holds g's projection; shard B has never heard of
    // it — the drifted state a re-import heals.
    shard_a.ok("PUT", "/groups/g", Some(json!({"context_ids": [ctx_a]})));

    // The same membership shard A already holds: A answers
    // "unchanged", B "created" with an empty projection — nothing in
    // the union changed.
    let stream =
        &format!("{{\"type\": \"group\", \"id\": \"g\", \"context_ids\": [\"{ctx_a}\"]}}\n");
    let (status, body) = post_import(&router, stream, None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["result"]["groups"][0]["outcome"],
        json!("unchanged"),
        "an empty created projection beside unchanged siblings replaced nothing: {body}"
    );

    // The record gains ctx_b: shard B's projection now carries a
    // member, so the union's row really changed.
    let stream = &format!(
        "{{\"type\": \"group\", \"id\": \"g\", \"context_ids\": [\"{ctx_a}\", \"{ctx_b}\"]}}\n"
    );
    let (status, body) = post_import(&router, stream, None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["result"]["groups"][0]["outcome"],
        json!("replaced"),
        "{body}"
    );

    // The created-with-members clause on its own: a FRESH fleet where
    // shard D never held g2 at all and the record brings D a member —
    // "created" carrying contexts beside an "unchanged" sibling is
    // what earns "replaced" here, with no shard answering "replaced"
    // itself.
    let shard_c = Server::start("gimp-outcome-c");
    let shard_d = Server::start("gimp-outcome-d");
    let router = Server::start_router(
        "gimp-outcome-2",
        &format!("ctx_c = {}\nctx_d = {}\n", shard_c.base, shard_d.base),
        &[],
    );
    let ctx_c = router.create_context("ctx_c");
    let ctx_d = router.create_context("ctx_d");
    shard_c.ok("PUT", "/groups/g2", Some(json!({"context_ids": [ctx_c]})));
    let stream = &format!(
        "{{\"type\": \"group\", \"id\": \"g2\", \"context_ids\": [\"{ctx_c}\", \"{ctx_d}\"]}}\n"
    );
    let (status, body) = post_import(&router, stream, None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["result"]["groups"][0]["outcome"],
        json!("replaced"),
        "a created projection CARRYING members changes the union: {body}"
    );
}

/// The group union's three-way sort of shard answers: a 404 is drift
/// healing (the other shard's projection still answers the row), but
/// any OTHER error status fails the whole union and passes through
/// verbatim — a 500 swallowed as "not found on that shard" would
/// serve a thinned union that looks complete.
#[test]
fn a_group_union_passes_through_a_shard_error_but_heals_a_404() {
    // Healing first: the group lives on shard A only (drift), shard B
    // answers its own 404 — the union is still the whole group.
    let shard_a = Server::start("gmf-heal-a");
    let shard_b = Server::start("gmf-heal-b");
    let router = Server::start_router(
        "gmf-heal",
        &format!("ctx_a = {}\nctx_b = {}\n", shard_a.base, shard_b.base),
        &[],
    );
    let ctx_a = shard_a.create_context("ctx_a");
    shard_a.ok("PUT", "/groups/g", Some(json!({"context_ids": [ctx_a]})));
    let healed = router.ok("GET", "/groups/g", None);
    assert_eq!(healed["context_ids"], json!([ctx_a]), "{healed}");

    // An erroring shard: same fleet shape, but shard B answers 500 to
    // everything — the union must refuse with the shard's own status,
    // never a 200 built from the surviving projection.
    let broken = FakeShard::start_with_status(500, json!({"code": "internal", "error": "boom"}));
    let router = Server::start_router(
        "gmf-error",
        &format!("ctx_a = {}\nctx_b = {}\n", shard_a.base, broken.endpoint),
        &[],
    );
    let (status, body) = router.call("GET", "/groups/g", None);
    assert_eq!(status, 500, "{body}");
    assert_eq!(body["code"], json!("internal"), "{body}");
}

/// The MCP manual comes from the FIRST shard that answers
/// `GET /protocol` — a dead shard ahead of a live one must not take
/// the manual down with it, and a fleet with no live shard at all
/// still initializes with the router's own local text.
#[test]
fn mcp_instructions_skip_a_dead_shard_and_fall_back_to_local_text() {
    let live = Server::start("mcp-manual-live");
    let router = Server::start_router(
        "mcp-manual",
        &format!("a = http://127.0.0.1:1\nb = {}\n", live.base),
        &[],
    );
    let initialize = json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}});
    let (status, init) = router.call("POST", "/mcp", Some(initialize.clone()));
    assert_eq!(status, 200, "{init}");
    let (_, manual) = live.call("GET", "/protocol", None);
    assert_eq!(
        init["result"]["instructions"], manual,
        "the manual must be the live shard's own text, dead shard skipped"
    );

    // No shard alive anywhere: initialize still answers, with the
    // router's local text instead of an error — and that text still
    // carries the version block, since those facts are the router
    // build's own (the same its `GET /version` answers), not a shard's.
    let all_dead = Server::start_router("mcp-manual-dead", "a = http://127.0.0.1:1\n", &[]);
    let (status, init) = all_dead.call("POST", "/mcp", Some(initialize));
    assert_eq!(status, 200, "{init}");
    let text = init["result"]["instructions"]
        .as_str()
        .expect("instructions must be text");
    assert!(!text.is_empty(), "{init}");
    // The version block is the trailer's fenced block — the LAST one in
    // the manual, after the body's own examples.
    let fenced = text
        .rsplit("```json\n")
        .next()
        .and_then(|rest| rest.split("\n```").next())
        .unwrap_or_else(|| panic!("the fallback manual carries the version block: {text}"));
    let block: serde_json::Value = serde_json::from_str(fenced).unwrap();
    let (_, version) = all_dead.call("GET", "/version", None);
    assert_eq!(
        block, version,
        "the fallback's version block is the router's own /version"
    );
    assert!(
        !text.contains("Semantic entry is ON"),
        "no shard answered, so no shard fact may be claimed: {text}"
    );
}

/// The preflight is what makes a multi-chunk refusal all-or-nothing:
/// a LATER chunk's dry-run refusal must stop the whole stream before
/// the first chunk ever ships for real — nothing lands, and the
/// refusal is the shard's own body, never a durable-prefix rewrap.
#[test]
fn a_later_chunks_preflight_refusal_leaves_the_earlier_chunk_unapplied() {
    let shard_a = Server::start("preflight-a");
    let shard_b = Server::start("preflight-b");
    let router = Server::start_router(
        "preflight",
        &format!("ctx_a = {}\nctx_b = {}\n", shard_a.base, shard_b.base),
        &[],
    );
    let ctx_a = router.ok("POST", "/contexts", Some(json!({"name": "ctx_a", })))["id"].clone();
    let ctx_b = router.ok("POST", "/contexts", Some(json!({"name": "ctx_b", })))["id"].clone();
    // ctx_b's batch names an alias whose canonical nothing interns —
    // stream-level parsing accepts it, so only the owning shard's own
    // dry run can refuse it.
    let stream = format!(
        "{{\"type\": \"source\", \"context_id\": {ctx_a}, \"id\": \"a.md\"}}\n\
         {{\"subject\": \"x\", \"label\": \"y\", \"object\": \"z\", \"weight\": 1.0}}\n\
         {{\"type\": \"source\", \"context_id\": {ctx_b}, \"id\": \"b.md\"}}\n\
         {{\"alias\": \"p\", \"canonical\": \"nowhere\", \"kind\": \"concept\"}}\n"
    );
    let (status, body) = post_import(&router, &stream, None);
    assert_eq!(status, 409, "{body}");
    assert_eq!(
        body["integrity"],
        json!("nothing_written"),
        "a preflight refusal precedes every real apply — the shard's own nothing-written \
         verdict passes through, never a durable-prefix rewrap: {body}"
    );
    let (query_status, hits) = router.call(
        "POST",
        &format!("/contexts/{}/query", shard_a.cx("ctx_a")),
        Some(json!({"subject": "x"})),
    );
    assert_eq!(query_status, 200, "{hits}");
    assert_eq!(
        hits["result"]["total"],
        json!(0),
        "chunk 1 must never have shipped for real: {hits}"
    );
}

/// A refusal with NOTHING landed anywhere must be the shard's own
/// body, unedited — no `integrity` field, no durable-prefix rewrap —
/// whether the refusing record was the stream's only schema or its
/// only group. (Both pass the dry-run preflight — schema scope checks
/// only, group validation runs against live state after the batches —
/// and refuse on the real run with zero batches/schemas ahead of them.)
#[test]
fn a_refusal_with_nothing_landed_passes_the_shards_own_body_through() {
    let shard = Server::start("nothing-landed");
    let router = Server::start_router(
        "nothing-landed-router",
        &format!("ghost = {}\n* = {}\n", shard.base, shard.base),
        &[],
    );

    let schema_only = "{\"type\": \"schema\", \"context_id\": \"ead6ef03-d61e-460c-933d-6d450c50a1e5\", \"mode\": \"warn\", \
         \"closed_labels\": false, \"types\": {}, \"relations\": {}}\n";
    let (status, body) = post_import(&router, schema_only, None);
    assert_eq!(status, 404, "{body}");
    assert_ne!(
        body["integrity"],
        json!("durable_prefix"),
        "no batch and no schema landed before this refusal: {body}"
    );

    let group_only = "{\"type\": \"group\", \"id\": \"g\", \"context_ids\": [\"ead6ef03-d61e-460c-933d-6d450c50a1e5\"]}\n";
    let (status, body) = post_import(&router, group_only, None);
    assert_eq!(status, 404, "{body}");
    assert_ne!(
        body["integrity"],
        json!("durable_prefix"),
        "no batch and no schema landed before this refusal either: {body}"
    );
}

/// The inverse: a group refusal AFTER a batch landed must rewrap with
/// the true cross-shard count — `durable_prefix`, naming the one
/// batch — never pass the failing shard's own "nothing reached me"
/// view through.
#[test]
fn a_group_refusal_after_a_landed_batch_rewraps_with_the_durable_count() {
    let shard_a = Server::start("group-rewrap-a");
    let shard_b = Server::start("group-rewrap-b");
    let router = Server::start_router(
        "group-rewrap",
        &format!("ctx_a = {}\nghost = {}\n", shard_a.base, shard_b.base),
        &[],
    );
    let ctx_a = router.ok("POST", "/contexts", Some(json!({"name": "ctx_a", })))["id"].clone();
    // The batch AND a schema land first; the group names `ghost`,
    // which is routable (so the router accepts the stream) but never
    // created, so shard B's live-state validation refuses on the real
    // run — and the rewrap must name BOTH landed counts, the
    // both-nonzero arm of its landed message.
    let stream = format!(
        "{{\"type\": \"source\", \"context_id\": {ctx_a}, \"id\": \"a.md\"}}\n\
         {{\"subject\": \"x\", \"label\": \"y\", \"object\": \"z\", \"weight\": 1.0}}\n\
         {{\"type\": \"schema\", \"context_id\": {ctx_a}, \"mode\": \"warn\", \
         \"closed_labels\": false, \"types\": {{}}, \"relations\": {{}}}}\n\
         {{\"type\": \"group\", \"id\": \"g\", \"context_ids\": [\"189ac1bb-03ba-433b-b379-89fabbe4757c\", \"ead6ef03-d61e-460c-933d-6d450c50a1e5\"]}}\n"
    );
    let (status, body) = post_import(&router, &stream, None);
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["integrity"], json!("durable_prefix"), "{body}");
    assert_eq!(body["durable_batches"], json!(1), "{body}");
    assert!(
        body["error"]
            .as_str()
            .is_some_and(|error| error.contains("1 source(s) and 1 schema record(s)")),
        "both landed counts must ride the rewrap: {body}"
    );
}

/// The batch half of the landed condition on its own: a group refusal
/// after ONE landed batch — no schema anywhere in the stream — still
/// rewraps with the durable count, and the landed message names the
/// batch alone.
#[test]
fn a_group_refusal_after_only_a_landed_batch_rewraps_with_the_durable_count() {
    let shard_a = Server::start("batch-only-rewrap-a");
    let shard_b = Server::start("batch-only-rewrap-b");
    let router = Server::start_router(
        "batch-only-rewrap",
        &format!("ctx_a = {}\nghost = {}\n", shard_a.base, shard_b.base),
        &[],
    );
    let ctx_a = router.ok("POST", "/contexts", Some(json!({"name": "ctx_a", })))["id"].clone();
    let stream = format!(
        "{{\"type\": \"source\", \"context_id\": {ctx_a}, \"id\": \"a.md\"}}\n\
         {{\"subject\": \"x\", \"label\": \"y\", \"object\": \"z\", \"weight\": 1.0}}\n\
         {{\"type\": \"group\", \"id\": \"g\", \"context_ids\": [\"189ac1bb-03ba-433b-b379-89fabbe4757c\", \"ead6ef03-d61e-460c-933d-6d450c50a1e5\"]}}\n"
    );
    let (status, body) = post_import(&router, &stream, None);
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["integrity"], json!("durable_prefix"), "{body}");
    assert_eq!(body["durable_batches"], json!(1), "{body}");
    let error = body["error"].as_str().unwrap_or_default();
    assert!(error.contains("1 source(s) landed"), "{body}");
    assert!(
        !error.contains("schema record"),
        "no schema anywhere in this stream: {body}"
    );
}

/// The schema half of the landed condition: a group refusal after a
/// SCHEMA landed (and no batch anywhere in the stream) must still
/// rewrap with the durable count, naming the schema record.
#[test]
fn a_group_refusal_after_a_landed_schema_rewraps_with_the_durable_count() {
    let shard_a = Server::start("schema-group-rewrap-a");
    let shard_b = Server::start("schema-group-rewrap-b");
    let router = Server::start_router(
        "schema-group-rewrap",
        &format!("ctx_a = {}\nghost = {}\n", shard_a.base, shard_b.base),
        &[],
    );
    let ctx_a = router.ok("POST", "/contexts", Some(json!({"name": "ctx_a", })))["id"].clone();
    let stream = format!(
        "{{\"type\": \"schema\", \"context_id\": {ctx_a}, \"mode\": \"warn\", \
         \"closed_labels\": false, \"types\": {{}}, \"relations\": {{}}}}\n\
         {{\"type\": \"group\", \"id\": \"g\", \"context_ids\": [\"189ac1bb-03ba-433b-b379-89fabbe4757c\", \"ead6ef03-d61e-460c-933d-6d450c50a1e5\"]}}\n"
    );
    let (status, body) = post_import(&router, &stream, None);
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["integrity"], json!("durable_prefix"), "{body}");
    assert!(
        body.get("durable_batches").is_none(),
        "no batch anywhere in this stream: {body}"
    );
    assert!(
        body["error"]
            .as_str()
            .is_some_and(|error| error.contains("1 schema record(s)")),
        "{body}"
    );
}

/// A shard that answers success with an envelope the router cannot
/// read still LANDED its chunk — a later refusal's rewrap must count
/// it from the chunk's own batch ranges, not from the parsed
/// outcomes.
#[test]
fn a_rewrap_counts_batches_landed_even_when_an_envelope_is_unreadable() {
    let stub = FakeShard::start(json!({"ok": true}));
    let shard = Server::start("rewrap-count-real");
    let router = Server::start_router(
        "rewrap-count",
        &format!("ghost = {}\nstubbed = {}\n", shard.base, stub.endpoint),
        &[],
    );
    // The batch creates its context on the stub shard by name. `ghost`
    // is never created: the group record naming it passes the dry-run
    // preflight (group validation runs against live state, after the
    // batches) and refuses on the real run — AFTER the stub shard's
    // batch chunk landed.
    let stream = concat!(
        "{\"type\": \"source\", \"context_id\": \"6030dd60-5e04-40cb-8aae-bd67951549b7\", \"id\": \"doc\", \"create\": {\"name\": \"stubbed\"}}\n",
        "{\"subject\": \"a\", \"label\": \"b\", \"object\": \"c\", \"weight\": 1.0}\n",
        "{\"type\": \"group\", \"id\": \"g\", \"groups\": [\"nowhere\"]}\n",
    );
    let (status, body) = post_import(&router, stream, None);
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["integrity"], json!("durable_prefix"), "{body}");
    assert_eq!(
        body["durable_batches"],
        json!(1),
        "the stub's chunk landed even though its envelope was unreadable: {body}"
    );
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
        "POST",
        "/contexts",
        Some(json!({"name": "moved", "description": "hot-reload target"})),
    );
    let moved = shard_a.cx("moved");

    // The broken edit: refused whole, old map still routing.
    let map_path = router.data_dir.join("route-map");
    std::fs::write(&map_path, "moved http://no-equals\n").unwrap();
    router.signal("-HUP");
    eventually(
        Duration::from_secs(10),
        "the refused reload to be counted",
        || map_reload_count(&router, "refused") >= 1,
    );
    let (status, body) = router.call("GET", &format!("/contexts/{moved}"), None);
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
        || router.call("GET", &format!("/contexts/{moved}"), None).0 == 404,
    );
    assert_eq!(
        shard_a.call("GET", &format!("/contexts/{moved}"), None).0,
        200
    );
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
        "POST",
        "/contexts",
        Some(json!({"name": "moved", "description": "watch target"})),
    );
    let moved = shard_a.cx("moved");
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
        || router.call("GET", &format!("/contexts/{moved}"), None).0 == 404,
    );
    // Exactly the one edit reloaded: the watch's baseline is the
    // digest of the bytes boot applied, so startup itself must never
    // count as a change and re-apply the same map.
    assert_eq!(
        map_reload_count(&router, "applied"),
        1,
        "one edit, one reload — a boot-time self-reload means the baseline digest is wrong"
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
    // A mid-move stray shares its ID across shards (a move copies the
    // context's files, id and all — #964), so both shards mint the
    // same deterministic sequence here to seed the collision; two rows
    // merely sharing a NAME are two distinct contexts and both stay.
    let deterministic = &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")][..];
    let shard_a = Server::start_with_env("stray-a", deterministic);
    let shard_b = Server::start_with_env("stray-b", deterministic);
    let router = Server::start_router(
        "stray",
        &format!("ctx-a = {}\nctx-b = {}\n", shard_a.base, shard_b.base),
        &[],
    );
    // Each context exists on BOTH shards (a mid-move leftover) under
    // the SAME id, with descriptions naming the copy so the winner is
    // observable.
    for (shard, tag) in [(&shard_a, "A"), (&shard_b, "B")] {
        for name in ["ctx-a", "ctx-b"] {
            shard.ok(
                "POST",
                "/contexts",
                Some(json!({"name": name, "description": format!("{name}@{tag}")})),
            );
        }
    }
    assert_eq!(
        shard_a.cx("ctx-a"),
        shard_b.cx("ctx-a"),
        "the seeded stray must actually collide on its id"
    );
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
        "POST",
        "/contexts",
        Some(json!({"name": "sake", "description": "銘柄"})),
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

/// One HTTP/1.1 request over a bare socket, the request-target sent
/// byte for byte — an HTTP client library resolves `..` in a path
/// before sending, which is exactly what this test must not do.
/// Returns (status, body).
fn raw_request(base: &str, method: &str, target: &str) -> (u16, Value) {
    use std::io::{Read, Write};
    let authority = base.trim_start_matches("http://");
    let mut stream = std::net::TcpStream::connect(authority).expect("router must accept");
    stream
        .set_read_timeout(Some(Duration::from_secs(10)))
        .unwrap();
    write!(
        stream,
        "{method} {target} HTTP/1.1\r\nHost: {authority}\r\nContent-Length: 0\r\n\
         Connection: close\r\n\r\n"
    )
    .unwrap();
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("router must answer");
    let text = String::from_utf8_lossy(&raw);
    let status: u16 = text
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or_else(|| panic!("no status line in {text:?}"));
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .unwrap_or("");
    let body = serde_json::from_str(body).unwrap_or_else(|_| Value::String(body.to_string()));
    (status, body)
}

/// A dot segment in a context-scoped path is refused by the router
/// itself, before any shard is consulted. The proxy forwards the path
/// verbatim and the outbound URL parse resolves dot segments, so
/// `POST /contexts/sake/../../import` would otherwise reach
/// shard-of(`sake`) as `POST /import` — past the router's own
/// per-batch routing, onto any endpoint the shard hosts. Raw and
/// percent-encoded forms alike; a plain context verb still proxies.
#[test]
fn a_dot_segment_in_a_context_path_never_reaches_a_shard() {
    let shard = FakeShard::start(json!({"status": "ok", "result": true, "time": 0.0}));
    let router = Server::start_router(
        "router-dot-segment",
        &format!("sake = {}\n", shard.endpoint),
        &[],
    );

    for target in [
        "/contexts/sake/../../import",
        "/contexts/sake/%2e%2e/%2e%2e/flush",
        "/contexts/sake/./recall",
        "/contexts/../import",
    ] {
        let (status, body) = raw_request(&router.base, "POST", target);
        assert_eq!(status, 400, "{target}: {body}");
        assert_eq!(body["code"], "invalid_argument", "{target}: {body}");
        assert!(
            body["error"].as_str().unwrap_or("").contains(target),
            "{target}: {body}"
        );
    }
    assert!(
        shard.requests().is_empty(),
        "no dot-segment request may reach the shard: {:?}",
        shard.requests()
    );

    // The same verb without a dot segment is the ordinary hop.
    let (status, body) = raw_request(&router.base, "POST", "/contexts/sake/recall");
    assert_eq!(status, 200, "{body}");
    assert_eq!(shard.requests().len(), 1, "{:?}", shard.requests());
}

/// A group write whose body the single-instance extractor would refuse
/// gets that refusal through the router — a non-object body, a member
/// that is not a string — instead of a router panic (500) or a silent
/// drop; and the per-request member cap is judged on the whole list
/// before it is split per shard, so it cannot scale with the shard
/// count. Nothing lands in either case.
#[test]
fn group_writes_refuse_unshaped_bodies_and_overlong_lists_as_one_instance_would() {
    let shard_a = Server::start("gcap-a");
    let shard_b = Server::start("gcap-b");
    let router = Server::start_router(
        "gcap",
        &format!("sake = {}\n* = {}\n", shard_a.base, shard_b.base),
        &[],
    );
    let sake = router.create_context("sake");

    // A JSON array is valid JSON and not a group request. (A single
    // instance reads it as a positional struct — serde's doing — and
    // would create the group; the router refuses instead, since a
    // probe to one shard would create it there alone.)
    let (status, refusal) = router.call("PUT", "/groups/g", Some(json!([])));
    assert_eq!(status, 400, "{refusal}");
    assert_eq!(refusal["code"], "invalid_argument", "{refusal}");
    assert_eq!(router.call("GET", "/groups/g", None).0, 404);
    for shard in [&shard_a, &shard_b] {
        assert_eq!(shard.call("GET", "/groups/g", None).0, 404);
    }

    // A non-string member is refused whole, not dropped — in the
    // shard's own extractor shape (its 422), since the body went to
    // one shard verbatim.
    router.ok("PUT", "/groups/g", Some(json!({"description": "対象"})));
    let (status, refusal) = router.call(
        "PATCH",
        "/groups/g",
        Some(json!({"add_context_ids": [sake, 42]})),
    );
    assert_eq!(status, 422, "{refusal}");
    assert_eq!(refusal["code"], "malformed_request", "{refusal}");
    assert!(
        refusal["error"]
            .as_str()
            .unwrap()
            .contains("add_context_ids[1]"),
        "{refusal}"
    );
    let entry = router.ok("GET", "/groups/g", None);
    assert_eq!(entry["context_ids"], json!([]), "{entry}");

    // 1001 members over two shards: each shard's slice would pass its
    // own cap, so the router judges the whole list.
    let members: Vec<String> = (0..1001)
        .map(|i| format!("00000000-0000-4000-8000-{i:012x}"))
        .collect();
    let (status, refusal) = router.call(
        "PATCH",
        "/groups/g",
        Some(json!({"add_context_ids": members})),
    );
    assert_eq!(status, 400, "{refusal}");
    assert_eq!(refusal["code"], "over_limit", "{refusal}");
    let entry = router.ok("GET", "/groups/g", None);
    assert_eq!(entry["context_ids"], json!([]), "{entry}");
}
