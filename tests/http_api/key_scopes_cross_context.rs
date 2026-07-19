//! Scoped-key gating and cross-context search/recall merges.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

use serde_json::{Value, json};

use crate::support::*;

/// TAGURU_KEY_SCOPES end to end: roles gate the verbs, context grants
/// gate the objects, the directory shows a scoped key only its world,
/// import checks its body-carried contexts, and an MCP tool call is
/// judged exactly as the route it dispatches onto.
#[test]
fn key_scopes_gate_roles_contexts_the_directory_and_mcp() {
    let server = Server::start_with_env(
        "http-scopes",
        &[
            (
                "TAGURU_API_TOKENS",
                "boss:atok,reader:rtok,scribe:wtok,potter:stok,curator:ctok",
            ),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"reader": "read", "scribe": "write", "potter": {"role": "write", "contexts": ["sake"]}, "curator": {"role": "admin", "contexts": ["sake"]}}"#,
            ),
        ],
    );
    let call = |method: &str, path: &str, body: Option<Value>, token: &str| {
        server.call_with_token(method, path, body, Some(token))
    };
    let fact = json!([{"subject": "蔵", "label": "杜氏", "object": "高瀬",
                      "weight": 1.0, "source": "a.md"}]);

    // The unscoped key keeps the historical full grant: admin, everywhere.
    assert_eq!(
        call(
            "PUT",
            "/contexts/sake",
            Some(json!({"description": "d"})),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "PUT",
            "/contexts/bunko",
            Some(json!({"description": "d"})),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "POST",
            "/contexts/sake/associations",
            Some(fact.clone()),
            "atok"
        )
        .0,
        200
    );
    // The unscoped admin key clears every role gate, drift/audit included.
    assert_eq!(
        call("POST", "/contexts/sake/drift/audit", None, "atok").0,
        200
    );

    // Read: the retrieval loop answers, the ingest loop refuses.
    assert_eq!(
        call(
            "POST",
            "/contexts/sake/recall",
            Some(json!({"cue": "蔵"})),
            "rtok"
        )
        .0,
        200
    );
    // drift/audit is Role::Read too: the reader key reaches it directly,
    // not just via the role hierarchy other checks here exercise.
    assert_eq!(
        call("POST", "/contexts/sake/drift/audit", None, "rtok").0,
        200
    );
    let (status, refusal) = call(
        "POST",
        "/contexts/sake/associations",
        Some(fact.clone()),
        "rtok",
    );
    assert_eq!(status, 403, "{refusal}");
    assert!(
        refusal["error"].as_str().unwrap().contains("needs 'write'"),
        "{refusal}"
    );
    assert_eq!(call("DELETE", "/contexts/sake", None, "rtok").0, 403);

    // Write: ingest yes, operator verbs no.
    assert_eq!(
        call(
            "POST",
            "/contexts/bunko/associations",
            Some(fact.clone()),
            "wtok"
        )
        .0,
        200
    );
    // The role hierarchy (Admin ⊃ Write ⊃ Read) puts drift/audit's
    // Role::Read within reach of a write key too.
    assert_eq!(
        call("POST", "/contexts/bunko/drift/audit", None, "wtok").0,
        200
    );
    assert_eq!(call("DELETE", "/contexts/bunko", None, "wtok").0, 403);
    assert_eq!(call("POST", "/flush", None, "wtok").0, 403);

    // Flush is server-wide (it names every flushed context), so a
    // context-scoped key is refused even at admin role — the refusal
    // is the CONTEXT bypass guard, not the role check (curator IS
    // admin). The unscoped admin flushes normally.
    let (status, scoped_flush) = call("POST", "/flush", None, "ctok");
    assert_eq!(status, 403, "{scoped_flush}");
    assert!(
        scoped_flush["error"]
            .as_str()
            .unwrap()
            .contains("server-wide"),
        "{scoped_flush}"
    );
    assert_eq!(call("POST", "/flush", None, "atok").0, 200);

    // Context-scoped write: inside the grant yes, outside no — and the
    // directory shows only the granted world.
    assert_eq!(
        call(
            "POST",
            "/contexts/sake/associations",
            Some(fact.clone()),
            "stok"
        )
        .0,
        200
    );
    let (status, outside) = call("POST", "/contexts/bunko/associations", Some(fact), "stok");
    assert_eq!(status, 403);
    assert!(
        outside["error"]
            .as_str()
            .unwrap()
            .contains("no grant on context 'bunko'"),
        "{outside}"
    );
    assert_eq!(call("GET", "/contexts/bunko", None, "stok").0, 403);
    // Role::Read still bows to the context grant: sake-scoped write
    // reaches bunko's associations no further than bunko's drift/audit.
    assert_eq!(
        call("POST", "/contexts/bunko/drift/audit", None, "stok").0,
        403
    );
    let (status, listed) = call("GET", "/contexts", None, "stok");
    assert_eq!(status, 200);
    assert_eq!(listed["result"]["total"], json!(1), "{listed}");
    assert_eq!(
        listed["result"]["contexts"][0]["name"],
        json!("sake"),
        "{listed}"
    );

    // Import carries its contexts in the body; the grant is checked
    // batch by batch before anything applies. (Import itself is an
    // admin verb, so even the granted context refuses for a writer.)
    let batch = "{\"taguru_batch\": 1, \"context\": \"bunko\", \"source\": \"s\"}\n";
    let (status, _) = post_import(&server, batch, Some("stok"));
    assert_eq!(status, 403);
    let (status, scoped_admin) = post_import(&server, batch, Some("atok"));
    assert_eq!(status, 200, "{scoped_admin}");

    // The body-carried-context refusal: curator is admin (clears the
    // role gate) but scoped to sake, so an out-of-grant bunko batch is
    // refused by the per-batch check, and an in-grant sake batch lands.
    let (status, out_of_grant) = post_import(&server, batch, Some("ctok"));
    assert_eq!(status, 403, "{out_of_grant}");
    assert!(
        out_of_grant["error"]
            .as_str()
            .unwrap()
            .contains("no grant on context 'bunko'"),
        "{out_of_grant}"
    );
    let sake_batch = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s\"}\n";
    let (status, in_grant) = post_import(&server, sake_batch, Some("ctok"));
    assert_eq!(status, 200, "{in_grant}");

    // MCP tool calls are judged as the routes they land on: the read
    // key's add_associations dispatch refuses with the same 403.
    let (status, reply) = server.call_with_token(
        "POST",
        "/mcp",
        Some(json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "add_associations", "arguments": {
                "context": "sake",
                "associations": [{"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0}],
            }},
        })),
        Some("rtok"),
    );
    assert_eq!(status, 200, "{reply}");
    assert_eq!(reply["result"]["isError"], json!(true), "{reply}");
    assert!(
        reply["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("403"),
        "{reply}"
    );
    // ...and a permitted tool call still works through the same key.
    let (status, reply) = server.call_with_token(
        "POST",
        "/mcp",
        Some(json!({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "recall", "arguments": {"context": "sake", "cue": "蔵"}},
        })),
        Some("rtok"),
    );
    assert_eq!(status, 200);
    assert_ne!(reply["result"]["isError"], json!(true), "{reply}");
    assert!(
        reply["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("matches"),
        "{reply}"
    );
}

/// Cross-context search: one request over several full context names,
/// every match tagged with the context it came from. recall/query
/// merge on |weight| (one scale across contexts), passage hits
/// interleave by per-context rank, duplicate targets dedupe, and the
/// refusals — empty list, missing name, over-cap list — land before
/// anything is searched. The MCP search tools ride the same routes
/// through their `contexts` argument.
#[test]
fn cross_context_search_merges_tagged_matches_across_named_contexts() {
    let server = Server::start("cross-search");
    for (name, fact) in [
        (
            "izakaya",
            json!([{"subject": "蔵", "label": "名物", "object": "燗酒",
                    "weight": 0.5, "source": "iz.md"}]),
        ),
        (
            "sakagura",
            json!([{"subject": "蔵", "label": "杜氏", "object": "高瀬",
                    "weight": 1.0, "source": "sk.md"}]),
        ),
        // Never named in a target list below — must never leak in.
        (
            "noise",
            json!([{"subject": "蔵", "label": "場所", "object": "港",
                    "weight": 2.0, "source": "no.md"}]),
        ),
    ] {
        server.ok("PUT", &format!("/contexts/{name}"), Some(json!({})));
        server.ok(
            "POST",
            &format!("/contexts/{name}/associations"),
            Some(fact),
        );
    }

    // recall across two of the three: both matches, each tagged.
    let recalled = server.ok(
        "POST",
        "/recall",
        Some(json!({"contexts": ["izakaya", "sakagura"], "cue": "蔵"})),
    );
    assert_eq!(recalled["total"], json!(2), "{recalled}");
    let tag_of = |matches: &Value, object: &str| -> String {
        matches
            .as_array()
            .unwrap()
            .iter()
            .find(|found| found["object"] == json!(object))
            .unwrap_or_else(|| panic!("no match with object {object}: {matches}"))["context"]
            .as_str()
            .unwrap()
            .to_string()
    };
    assert_eq!(tag_of(&recalled["matches"], "高瀬"), "sakagura");
    assert_eq!(tag_of(&recalled["matches"], "燗酒"), "izakaya");

    // Past the limit the strongest |weight| survives, across contexts.
    let cut = server.ok(
        "POST",
        "/recall",
        Some(json!({"contexts": ["izakaya", "sakagura"], "cue": "蔵", "limit": 1})),
    );
    assert_eq!(cut["total"], json!(2), "{cut}");
    assert_eq!(cut["matches"].as_array().unwrap().len(), 1, "{cut}");
    assert_eq!(cut["matches"][0]["context"], json!("sakagura"), "{cut}");

    // Naming a context twice is redundant, not double.
    let deduped = server.ok(
        "POST",
        "/recall",
        Some(json!({"contexts": ["izakaya", "izakaya", "sakagura"], "cue": "蔵"})),
    );
    assert_eq!(deduped["total"], json!(2), "{deduped}");

    // query: position-pinned, same tagging contract.
    let queried = server.ok(
        "POST",
        "/query",
        Some(json!({"contexts": ["izakaya", "sakagura"], "label": "杜氏"})),
    );
    assert_eq!(queried["total"], json!(1), "{queried}");
    assert_eq!(queried["matches"][0]["context"], json!("sakagura"));
    assert_eq!(queried["matches"][0]["object"], json!("高瀬"));

    // The text lane: hits carry their context and interleave by
    // per-context rank — both rank-0 hits lead, in target-list order.
    server.ok(
        "POST",
        "/contexts/izakaya/sources",
        Some(json!({"passages": {"iz.md": "蔵元の燗酒は冬の名物。"}})),
    );
    server.ok(
        "POST",
        "/contexts/sakagura/sources",
        Some(json!({"passages": {"sk.md": "杜氏の高瀬は蔵元を任されている。"}})),
    );
    let hits = server.ok(
        "POST",
        "/sources/search",
        Some(json!({"contexts": ["izakaya", "sakagura"], "query": "蔵元"})),
    );
    let hits = hits["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 2, "both contexts must answer: {hits:?}");
    assert_eq!(hits[0]["context"], json!("izakaya"), "{hits:?}");
    assert_eq!(hits[1]["context"], json!("sakagura"), "{hits:?}");
    assert_eq!(hits[0]["source"], json!("iz.md"), "{hits:?}");

    // Refusals, each before anything is searched.
    let (status, empty) = server.call(
        "POST",
        "/recall",
        Some(json!({"contexts": [], "cue": "蔵"})),
    );
    assert_eq!(status, 400, "{empty}");
    assert_eq!(empty["code"], json!("invalid_argument"), "{empty}");

    let (status, missing) = server.call(
        "POST",
        "/recall",
        Some(json!({"contexts": ["izakaya", "ghost"], "cue": "蔵"})),
    );
    assert_eq!(status, 404, "{missing}");
    assert_eq!(missing["code"], json!("no_context"), "{missing}");
    assert!(
        missing["error"].as_str().unwrap().contains("'ghost'"),
        "{missing}"
    );

    let flood: Vec<String> = (0..1001).map(|i| format!("c{i}")).collect();
    let (status, over) = server.call(
        "POST",
        "/query",
        Some(json!({"contexts": flood, "label": "l"})),
    );
    assert_eq!(status, 400, "{over}");
    assert_eq!(over["code"], json!("over_limit"), "{over}");

    // The MCP search tools take `contexts` as the cross-context form…
    let (status, reply) = server.call(
        "POST",
        "/mcp",
        Some(json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "recall", "arguments": {
                "contexts": ["izakaya", "sakagura"], "cue": "蔵",
            }},
        })),
    );
    assert_eq!(status, 200, "{reply}");
    assert_ne!(reply["result"]["isError"], json!(true), "{reply}");
    let text = reply["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("sakagura") && text.contains("izakaya"),
        "{text}"
    );

    // …and refuse the ambiguous both-at-once form.
    let (status, reply) = server.call(
        "POST",
        "/mcp",
        Some(json!({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "recall", "arguments": {
                "context": "izakaya", "contexts": ["sakagura"], "cue": "蔵",
            }},
        })),
    );
    assert_eq!(status, 200, "{reply}");
    assert_eq!(reply["result"]["isError"], json!(true), "{reply}");
    assert!(
        reply["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("not both"),
        "{reply}"
    );
}

/// Every target's fetch runs concurrently (bounded by
/// `TAGURU_CROSS_SEARCH_CONCURRENCY`, default 4) rather than in a
/// sequential loop — with four targets in play, a regression to
/// per-index mishandling (e.g. a race writing the wrong slot in the
/// gather step) would show up as a wrong total, a dropped match, or a
/// match tagged with the wrong context, which this pins down.
#[test]
fn cross_recall_merges_four_targets_gathered_concurrently() {
    let server = Server::start("cross-concurrent");
    for (name, weight) in [("c1", 1.0), ("c2", 2.0), ("c3", 3.0), ("c4", 4.0)] {
        server.ok("PUT", &format!("/contexts/{name}"), Some(json!({})));
        server.ok(
            "POST",
            &format!("/contexts/{name}/associations"),
            Some(json!([
                {"subject": "蔵", "label": "l", "object": format!("o-{name}"), "weight": weight}
            ])),
        );
    }

    let recalled = server.ok(
        "POST",
        "/recall",
        Some(json!({"contexts": ["c1", "c2", "c3", "c4"], "cue": "蔵"})),
    );
    assert_eq!(recalled["total"], json!(4), "{recalled}");
    let matches = recalled["matches"].as_array().unwrap();
    assert_eq!(matches.len(), 4);
    // Every target landed, strongest |weight| first — nothing dropped
    // or misrouted by the concurrent gather.
    let contexts: Vec<&str> = matches
        .iter()
        .map(|m| m["context"].as_str().unwrap())
        .collect();
    assert_eq!(contexts, vec!["c4", "c3", "c2", "c1"], "{recalled}");
    let objects: Vec<&str> = matches
        .iter()
        .map(|m| m["object"].as_str().unwrap())
        .collect();
    assert_eq!(objects, vec!["o-c4", "o-c3", "o-c2", "o-c1"], "{recalled}");
}

/// `(subject, label, object)` alone only identifies an edge *within*
/// one target context — two different contexts can each hold an edge
/// with the identical triple and weight, so the merged pool's cursor
/// must break the tie on `context`. This also proves the *page* order
/// is `cross_rank`'s doing, not an accident of the `contexts` list
/// order in the request (deliberately given here as `["zeta",
/// "alpha"]`, the reverse of the order the pages must come back in).
#[test]
fn cross_recall_pages_with_a_cursor_across_contexts() {
    let server = Server::start("cross-cursor");
    for name in ["zeta", "alpha"] {
        server.ok("PUT", &format!("/contexts/{name}"), Some(json!({})));
        server.ok(
            "POST",
            &format!("/contexts/{name}/associations"),
            Some(json!([
                {"subject": "蔵", "label": "銘柄", "object": "青嶺", "weight": 1.0}
            ])),
        );
    }

    let first = server.ok(
        "POST",
        "/recall",
        Some(json!({"contexts": ["zeta", "alpha"], "cue": "蔵", "limit": 1})),
    );
    assert_eq!(first["total"], json!(2), "{first}");
    let matches = first["matches"].as_array().unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(
        matches[0]["context"],
        json!("alpha"),
        "context breaks the identical-triple tie, lexicographically — \
         not the ['zeta', 'alpha'] request order: {first}"
    );

    let cursor_from = |m: &Value| {
        json!({
            "weight": m["weight"], "context": m["context"],
            "subject": m["subject"], "label": m["label"], "object": m["object"],
        })
    };
    let second = server.ok(
        "POST",
        "/recall",
        Some(json!({
            "contexts": ["zeta", "alpha"], "cue": "蔵", "limit": 1,
            "after": cursor_from(&matches[0]),
        })),
    );
    assert_eq!(second["total"], json!(2), "total stays constant: {second}");
    let matches = second["matches"].as_array().unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0]["context"], json!("zeta"), "{second}");

    let third = server.ok(
        "POST",
        "/recall",
        Some(json!({
            "contexts": ["zeta", "alpha"], "cue": "蔵", "limit": 1,
            "after": cursor_from(&matches[0]),
        })),
    );
    assert_eq!(third["matches"], json!([]), "the walk has ended: {third}");

    // A cross-context cursor's extra `context` field is REQUIRED — a
    // bare MatchCursor-shaped `after` (missing `context`) is a 422
    // (body fails to deserialize), not silently accepted with that
    // field dropped.
    let (status, refusal) = server.call(
        "POST",
        "/recall",
        Some(json!({
            "contexts": ["zeta", "alpha"], "cue": "蔵",
            "after": {"weight": 1.0, "subject": "蔵", "label": "銘柄", "object": "青嶺"},
        })),
    );
    assert_eq!(status, 422, "{refusal}");
}

/// Cross-context search by group: a `groups` name searches every
/// context it reaches — nested children included — and combines with
/// `contexts`, overlaps deduped, so a context is searched once however
/// many ways it was named. Directly named contexts lead the tie order
/// and group-resolved members follow in name order; an unknown group
/// is `no_group`, an empty resolution is an empty result, and the MCP
/// search tools take `groups` beside `contexts`.
#[test]
fn cross_context_search_resolves_groups_beside_contexts() {
    let server = Server::start("cross-groups");
    for (name, fact) in [
        (
            "izakaya",
            json!([{"subject": "蔵", "label": "名物", "object": "燗酒",
                    "weight": 0.5, "source": "iz.md"}]),
        ),
        (
            "sakagura",
            json!([{"subject": "蔵", "label": "杜氏", "object": "高瀬",
                    "weight": 1.0, "source": "sk.md"}]),
        ),
    ] {
        server.ok("PUT", &format!("/contexts/{name}"), Some(json!({})));
        server.ok(
            "POST",
            &format!("/contexts/{name}/associations"),
            Some(fact),
        );
    }
    // sakaya bundles izakaya; nomiya bundles sakagura and nests sakaya.
    server.ok(
        "PUT",
        "/groups/sakaya",
        Some(json!({"contexts": ["izakaya"]})),
    );
    server.ok(
        "PUT",
        "/groups/nomiya",
        Some(json!({"contexts": ["sakagura"], "groups": ["sakaya"]})),
    );

    // One group name reaches both contexts through the nesting.
    let recalled = server.ok(
        "POST",
        "/recall",
        Some(json!({"groups": ["nomiya"], "cue": "蔵"})),
    );
    assert_eq!(recalled["total"], json!(2), "{recalled}");

    // Naming a member directly AND through the group searches it once.
    let deduped = server.ok(
        "POST",
        "/recall",
        Some(json!({"contexts": ["izakaya"], "groups": ["nomiya", "sakaya"], "cue": "蔵"})),
    );
    assert_eq!(deduped["total"], json!(2), "{deduped}");

    // query rides the same resolution.
    let queried = server.ok(
        "POST",
        "/query",
        Some(json!({"groups": ["nomiya"], "label": "杜氏"})),
    );
    assert_eq!(queried["total"], json!(1), "{queried}");
    assert_eq!(queried["matches"][0]["context"], json!("sakagura"));

    // Passage rank ties: contexts named directly lead, group-resolved
    // members follow — sakagura outranks izakaya arriving via sakaya.
    server.ok(
        "POST",
        "/contexts/izakaya/sources",
        Some(json!({"passages": {"iz.md": "蔵元の燗酒は冬の名物。"}})),
    );
    server.ok(
        "POST",
        "/contexts/sakagura/sources",
        Some(json!({"passages": {"sk.md": "杜氏の高瀬は蔵元を任されている。"}})),
    );
    let hits = server.ok(
        "POST",
        "/sources/search",
        Some(json!({"contexts": ["sakagura"], "groups": ["sakaya"], "query": "蔵元"})),
    );
    let hits = hits["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 2, "both contexts must answer: {hits:?}");
    assert_eq!(hits[0]["context"], json!("sakagura"), "{hits:?}");
    assert_eq!(hits[1]["context"], json!("izakaya"), "{hits:?}");

    // An empty group is an empty result, not an error…
    server.ok("PUT", "/groups/kara", Some(json!({})));
    let empty = server.ok(
        "POST",
        "/recall",
        Some(json!({"groups": ["kara"], "cue": "蔵"})),
    );
    assert_eq!(empty["total"], json!(0), "{empty}");
    assert_eq!(empty["matches"], json!([]), "{empty}");

    // …an unknown group refuses before anything is searched…
    let (status, ghost) = server.call(
        "POST",
        "/recall",
        Some(json!({"groups": ["maboroshi"], "cue": "蔵"})),
    );
    assert_eq!(status, 404, "{ghost}");
    assert_eq!(ghost["code"], json!("no_group"), "{ghost}");
    assert!(
        ghost["error"].as_str().unwrap().contains("'maboroshi'"),
        "{ghost}"
    );

    // …naming nothing at all is a client bug, not an empty result…
    let (status, nothing) = server.call("POST", "/recall", Some(json!({"cue": "蔵"})));
    assert_eq!(status, 400, "{nothing}");
    assert_eq!(nothing["code"], json!("invalid_argument"), "{nothing}");

    // …and the groups list shares the input-items cap.
    let flood: Vec<String> = (0..1001).map(|i| format!("g{i}")).collect();
    let (status, over) = server.call(
        "POST",
        "/recall",
        Some(json!({"groups": flood, "cue": "蔵"})),
    );
    assert_eq!(status, 400, "{over}");
    assert_eq!(over["code"], json!("over_limit"), "{over}");

    // The MCP search tools take `groups` as a cross-context form…
    let (status, reply) = server.call(
        "POST",
        "/mcp",
        Some(json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "recall", "arguments": {
                "groups": ["nomiya"], "cue": "蔵",
            }},
        })),
    );
    assert_eq!(status, 200, "{reply}");
    assert_ne!(reply["result"]["isError"], json!(true), "{reply}");
    let text = reply["result"]["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("sakagura") && text.contains("izakaya"),
        "{text}"
    );

    // …and beside `context` it is the same ambiguity as `contexts`.
    let (status, reply) = server.call(
        "POST",
        "/mcp",
        Some(json!({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "recall", "arguments": {
                "context": "izakaya", "groups": ["nomiya"], "cue": "蔵",
            }},
        })),
    );
    assert_eq!(status, 200, "{reply}");
    assert_eq!(reply["result"]["isError"], json!(true), "{reply}");
    assert!(
        reply["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("not both"),
        "{reply}"
    );
}

/// A scoped key's cross-context search is refused whole on any name
/// beyond the grant — and because the grant check runs before the
/// existence check, the 403 for a live out-of-grant name and for a
/// made-up one are indistinguishable: no existence oracle. A `groups`
/// target resolves to the grant's slice instead of refusing: a refusal
/// would name the very membership the group listings hide.
#[test]
fn cross_context_search_respects_grants_without_an_existence_oracle() {
    let server = Server::start_with_env(
        "cross-scopes",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,potter:stok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"potter": {"role": "read", "contexts": ["sake"]}}"#,
            ),
        ],
    );
    let call = |method: &str, path: &str, body: Option<Value>, token: &str| {
        server.call_with_token(method, path, body, Some(token))
    };
    for name in ["sake", "bunko"] {
        assert_eq!(
            call("PUT", &format!("/contexts/{name}"), Some(json!({})), "atok").0,
            200
        );
    }

    // Inside the grant: answers.
    let (status, inside) = call(
        "POST",
        "/recall",
        Some(json!({"contexts": ["sake"], "cue": "蔵"})),
        "stok",
    );
    assert_eq!(status, 200, "{inside}");

    // One out-of-grant name refuses the whole request…
    let (status, live) = call(
        "POST",
        "/recall",
        Some(json!({"contexts": ["sake", "bunko"], "cue": "蔵"})),
        "stok",
    );
    assert_eq!(status, 403, "{live}");
    assert_eq!(live["code"], json!("forbidden"), "{live}");
    assert!(
        live["error"]
            .as_str()
            .unwrap()
            .contains("no grant on context 'bunko'"),
        "{live}"
    );

    // …and a nonexistent out-of-grant name answers the IDENTICAL
    // refusal — never the 404 that would betray which names exist.
    let (status, ghost) = call(
        "POST",
        "/recall",
        Some(json!({"contexts": ["sake", "ghost"], "cue": "蔵"})),
        "stok",
    );
    assert_eq!(status, 403, "{ghost}");
    assert_eq!(
        ghost["error"]
            .as_str()
            .unwrap()
            .replace("'ghost'", "'bunko'"),
        live["error"].as_str().unwrap(),
        "the refusals must differ in nothing but the echoed name"
    );

    // The unscoped admin hears the truth about the same request.
    let (status, truth) = call(
        "POST",
        "/recall",
        Some(json!({"contexts": ["sake", "ghost"], "cue": "蔵"})),
        "atok",
    );
    assert_eq!(status, 404, "{truth}");
    assert_eq!(truth["code"], json!("no_context"), "{truth}");

    // The other two searches run the same gate.
    let (status, _) = call(
        "POST",
        "/query",
        Some(json!({"contexts": ["sake", "bunko"], "label": "l"})),
        "stok",
    );
    assert_eq!(status, 403);
    let (status, _) = call(
        "POST",
        "/sources/search",
        Some(json!({"contexts": ["sake", "bunko"], "query": "蔵元"})),
        "stok",
    );
    assert_eq!(status, 403);

    // A group target resolves to the grant's slice instead of refusing
    // — the same slice `GET /groups` shows a scoped key — so nothing
    // in the answer betrays the out-of-grant member.
    for (context, fact) in [
        (
            "sake",
            json!({"subject": "蔵", "label": "銘柄", "object": "白露"}),
        ),
        (
            "bunko",
            json!({"subject": "蔵", "label": "所蔵", "object": "写本"}),
        ),
    ] {
        let (status, _) = call(
            "POST",
            &format!("/contexts/{context}/associations"),
            Some(json!([{"subject": fact["subject"], "label": fact["label"],
                         "object": fact["object"], "weight": 1.0, "source": "x.md"}])),
            "atok",
        );
        assert_eq!(status, 200);
    }
    let (status, _) = call(
        "PUT",
        "/groups/zenbu",
        Some(json!({"contexts": ["sake", "bunko"]})),
        "atok",
    );
    assert_eq!(status, 200);
    let (status, sliced) = call(
        "POST",
        "/recall",
        Some(json!({"groups": ["zenbu"], "cue": "蔵"})),
        "stok",
    );
    assert_eq!(status, 200, "{sliced}");
    assert_eq!(sliced["result"]["total"], json!(1), "{sliced}");
    assert_eq!(
        sliced["result"]["matches"][0]["context"],
        json!("sake"),
        "{sliced}"
    );
    assert!(
        !sliced.to_string().contains("bunko"),
        "the slice must not name the out-of-grant member: {sliced}"
    );

    // A group with nothing in the grant answers empty, exactly as an
    // empty group would — never a refusal naming the hidden member.
    let (status, _) = call(
        "PUT",
        "/groups/soto",
        Some(json!({"contexts": ["bunko"]})),
        "atok",
    );
    assert_eq!(status, 200);
    let (status, outside) = call(
        "POST",
        "/recall",
        Some(json!({"groups": ["soto"], "cue": "蔵"})),
        "stok",
    );
    assert_eq!(status, 200, "{outside}");
    assert_eq!(outside["result"]["total"], json!(0), "{outside}");
    assert!(!outside.to_string().contains("bunko"), "{outside}");

    // Directly naming the out-of-grant context still refuses whole,
    // groups on the request or not.
    let (status, direct) = call(
        "POST",
        "/recall",
        Some(json!({"contexts": ["bunko"], "groups": ["zenbu"], "cue": "蔵"})),
        "stok",
    );
    assert_eq!(status, 403, "{direct}");
}

/// The access log names the context a request addressed, and every
/// destructive operation leaves one `taguru::audit` line saying who
/// did what to which object — the route template alone cannot answer
/// "which context did this key delete" after the fact.
#[test]
fn the_access_log_names_the_context_and_destructive_ops_leave_audit_lines() {
    let data_dir = std::env::temp_dir().join(format!("taguru-auditlog-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    common::scrub_taguru_env(&mut command)
        .env("TAGURU_ADDR", "127.0.0.1:0")
        .env("TAGURU_DATA_DIR", &data_dir)
        .env("TAGURU_LOG_FORMAT", "json")
        .env("TAGURU_API_TOKEN", "opskey");
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("server binary must spawn");

    let stdout = child.stdout.take().expect("stdout must be piped");
    let (addr, _stdout_lines) = common::read_listen_line("server", stdout);
    let base = format!("http://{addr}");

    let call = |method: &str, path: &str, body: Option<Value>| {
        let request = ureq::http::Request::builder()
            .method(method)
            .uri(format!("{base}{path}"))
            .header("Authorization", "Bearer opskey");
        let response = match body {
            Some(body) => request
                .header("Content-Type", "application/json")
                .body(body.to_string())
                .map(|request| test_agent().run(request)),
            None => request.body(()).map(|request| test_agent().run(request)),
        };
        finish(response.expect("request must assemble"), method, path).0
    };
    assert_eq!(
        call("PUT", "/contexts/sake", Some(json!({"description": "d"}))),
        200
    );
    assert_eq!(
        call(
            "POST",
            "/contexts/sake/associations",
            Some(json!([{"subject": "蔵", "label": "杜氏", "object": "高瀬",
                         "weight": 1.0, "source": "a.md"}])),
        ),
        200
    );
    // Register then remove an alias — each leaves its own audit line.
    assert_eq!(
        call(
            "POST",
            "/contexts/sake/aliases",
            Some(json!({"concepts": {"Kura": "蔵"}})),
        ),
        200
    );
    assert_eq!(
        call(
            "DELETE",
            "/contexts/sake/aliases",
            Some(json!({"concepts": ["Kura"]})),
        ),
        200
    );
    // An import batch (retract-then-apply) and a compaction: both
    // destructive, both audited. Import is NDJSON, so send it raw
    // rather than through the JSON `call` helper.
    let import_status = test_agent()
        .post(format!("{base}/import"))
        .header("Authorization", "Bearer opskey")
        .send(
            "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"b.md\"}\n\
             {\"subject\": \"蔵\", \"label\": \"銘柄\", \"object\": \"青嶺\", \"weight\": 1.0}\n",
        )
        .unwrap_or_else(|error| panic!("import: {error}"))
        .status()
        .as_u16();
    assert_eq!(import_status, 200);
    assert_eq!(call("POST", "/contexts/sake/compact", None), 200);
    assert_eq!(
        call(
            "POST",
            "/contexts/sake/sources/retract",
            Some(json!({"source": "a.md"})),
        ),
        200
    );
    assert_eq!(call("DELETE", "/contexts/sake", None), 200);

    // Stop the server so stderr reaches EOF, then judge the whole log.
    let pid = child.id().to_string();
    Command::new("kill")
        .args(["-TERM", &pid])
        .status()
        .expect("kill must run");
    let _ = child.wait();
    let stderr = child.stderr.take().expect("stderr must be piped");
    let lines: Vec<Value> = BufReader::new(stderr)
        .lines()
        .map_while(Result::ok)
        .filter_map(|line| serde_json::from_str(&line).ok())
        .collect();

    let access_delete = lines
        .iter()
        .find(|record| {
            record["fields"]["message"] == json!("http")
                && record["fields"]["method"] == json!("DELETE")
                && record["fields"]["route"] == json!("/contexts/{name}")
        })
        .expect("an access-log line for the DELETE must appear");
    assert_eq!(access_delete["fields"]["context"], json!("sake"));
    assert_eq!(access_delete["fields"]["key"], json!("default"));

    let retracted = lines
        .iter()
        .find(|record| {
            record["target"] == json!("taguru::audit")
                && record["fields"]["message"] == json!("source retracted")
        })
        .expect("the retraction must leave an audit line");
    assert_eq!(retracted["fields"]["context"], json!("sake"));
    assert_eq!(retracted["fields"]["source"], json!("a.md"));
    assert_eq!(retracted["fields"]["key"], json!("default"));
    assert_eq!(retracted["fields"]["associations_touched"], json!(1));

    let deleted = lines
        .iter()
        .find(|record| {
            record["target"] == json!("taguru::audit")
                && record["fields"]["message"] == json!("context deleted")
        })
        .expect("the deletion must leave an audit line");
    assert_eq!(deleted["fields"]["context"], json!("sake"));
    assert_eq!(deleted["fields"]["files_removed"], json!(true));

    // Every destructive operation — not just delete/retract — leaves an
    // audit line naming the key and the context. A missing one here is
    // a silently-narrowed guarantee.
    let audit_line = |message: &str| {
        lines
            .iter()
            .find(|record| {
                record["target"] == json!("taguru::audit")
                    && record["fields"]["message"] == json!(message)
            })
            .unwrap_or_else(|| panic!("missing audit line: {message}"))
    };
    let aliases_registered = audit_line("aliases registered");
    assert_eq!(aliases_registered["fields"]["context"], json!("sake"));
    assert_eq!(aliases_registered["fields"]["key"], json!("default"));
    let aliases_removed = audit_line("aliases removed");
    assert_eq!(aliases_removed["fields"]["context"], json!("sake"));
    assert_eq!(aliases_removed["fields"]["key"], json!("default"));
    let imported = audit_line("import batch applied");
    assert_eq!(imported["fields"]["context"], json!("sake"));
    assert_eq!(imported["fields"]["source"], json!("b.md"));
    assert_eq!(imported["fields"]["key"], json!("default"));
    let compacted = audit_line("context compacted");
    assert_eq!(compacted["fields"]["context"], json!("sake"));
    assert_eq!(compacted["fields"]["key"], json!("default"));

    let _ = std::fs::remove_dir_all(&data_dir);
}
