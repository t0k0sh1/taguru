//! Validation and capping shared by every candidate-matching endpoint.

use serde_json::{Value, json};

use crate::support::*;

#[test]
fn lookalike_candidates_carry_the_evidence_to_tell_them_apart() {
    let server = Server::start("lookalikes");
    server.ok(
        "PUT",
        "/contexts/looks",
        Some(json!({"description": "字面の近い別物たち"})),
    );
    server.ok(
        "POST",
        "/contexts/looks/associations",
        Some(json!([
            {"subject": "東京都", "label": "分類", "object": "日本の首都", "weight": 1.0},
            {"subject": "京都", "label": "所在", "object": "関西", "weight": 1.0},
            {"subject": "青嶺株式会社", "label": "業種", "object": "電機メーカー", "weight": 1.0},
            {"subject": "possible", "label": "means", "object": "can_be_done", "weight": 1.0},
            {"subject": "impossible", "label": "means", "object": "can_be_done", "weight": -1.0},
        ])),
    );

    // 東京都/京都: the containment lookalike scores a strong 0.67, and
    // the response says both how it matched and what it actually is —
    // enough to reject it without a second round trip.
    let kyoto = server.ok(
        "POST",
        "/contexts/looks/resolve",
        Some(json!({"cue": "京都"})),
    );
    assert_eq!(kyoto[0]["name"], json!("京都"));
    assert_eq!(kyoto[0]["kind"], json!("exact"));
    assert!(
        kyoto[0]["gloss"].as_str().unwrap().contains("関西"),
        "{kyoto}"
    );
    assert_eq!(kyoto[1]["name"], json!("東京都"));
    assert_eq!(kyoto[1]["kind"], json!("containment"));
    assert!(
        kyoto[1]["gloss"].as_str().unwrap().contains("日本の首都"),
        "{kyoto}"
    );

    // 前株/後株: the cue names a company that is NOT registered; the
    // stored lookalike surfaces through the fuzzy tier, and its gloss
    // (wrong line of business) is what lets the caller reject it.
    let maekabu = server.ok(
        "POST",
        "/contexts/looks/resolve",
        Some(json!({"cue": "株式会社青嶺"})),
    );
    assert_eq!(maekabu[0]["name"], json!("青嶺株式会社"));
    assert_eq!(maekabu[0]["kind"], json!("fuzzy"));
    assert!(
        maekabu[0]["gloss"]
            .as_str()
            .unwrap()
            .contains("電機メーカー"),
        "{maekabu}"
    );

    // possible/impossible: containment scores 0.8 for the antonym; the
    // negative fact renders as a denial in its gloss.
    let possible = server.ok(
        "POST",
        "/contexts/looks/resolve",
        Some(json!({"cue": "possible"})),
    );
    assert_eq!(possible[0]["name"], json!("possible"));
    assert_eq!(possible[0]["kind"], json!("exact"));
    assert_eq!(possible[1]["name"], json!("impossible"));
    assert_eq!(possible[1]["kind"], json!("containment"));
    assert_eq!(possible[1]["score"], json!(8.0 / 10.0));
    assert!(
        possible[1]["gloss"]
            .as_str()
            .unwrap()
            .contains("can_be_doneではない"),
        "{possible}"
    );

    // Labels resolve with the same evidence; the gloss shows example
    // triples so a writer can pick the right relation before minting.
    let label = server.ok(
        "POST",
        "/contexts/looks/resolve_label",
        Some(json!({"cue": "means"})),
    );
    assert_eq!(label[0]["kind"], json!("exact"));
    assert!(
        label[0]["gloss"].as_str().unwrap().contains("means"),
        "{label}"
    );
}

#[test]
fn oversized_input_lists_are_refused_before_any_work() {
    let server = Server::start("input-caps");
    server.ok("PUT", "/contexts/caps", Some(json!({"description": "d"})));

    // 1001 items trips every list-shaped read input.
    let over: Vec<String> = (0..1001).map(|i| format!("o{i}")).collect();
    for (path, body) in [
        ("/contexts/caps/explore", json!({"origins": over.clone()})),
        ("/contexts/caps/activate", json!({"origins": over.clone()})),
        (
            "/contexts/caps/unreachable_from",
            json!({"origins": over.clone()}),
        ),
        ("/contexts/caps/query", json!({"subject": over.clone()})),
        (
            "/contexts/caps/sources/lookup",
            json!({"sources": over.clone()}),
        ),
    ] {
        let (status, parsed) = server.call("POST", path, Some(body));
        assert_eq!(status, 400, "{path}: {parsed}");
        assert!(
            parsed["error"]
                .as_str()
                .unwrap()
                .contains("per-request limit"),
            "{path}: {parsed}"
        );
    }

    // The cap itself still passes — it matches the largest page
    // list_sources serves, so a paged bulk workflow fits exactly.
    let at_cap: Vec<String> = (0..1000).map(|i| format!("o{i}")).collect();
    server.ok(
        "POST",
        "/contexts/caps/explore",
        Some(json!({"origins": at_cap})),
    );

    // Alias batches are WAL writes and share the association batch cap.
    let aliases: serde_json::Map<String, Value> = (0..10_001)
        .map(|i| (format!("a{i}"), json!("青嶺酒造")))
        .collect();
    let (status, parsed) = server.call(
        "POST",
        "/contexts/caps/aliases",
        Some(json!({"concepts": aliases})),
    );
    assert_eq!(status, 400, "{parsed}");
    assert_eq!(parsed["code"], json!("over_limit"), "{parsed}");
    assert!(
        parsed["error"]
            .as_str()
            .unwrap()
            .contains("per-request limit"),
        "{parsed}"
    );

    // Removal is over-limit, not malformed — the same `over_limit` code
    // its add twin returns, not the `invalid_argument` it once gave.
    let removals: Vec<String> = (0..10_001).map(|i| format!("a{i}")).collect();
    let (status, parsed) = server.call(
        "DELETE",
        "/contexts/caps/aliases",
        Some(json!({"concepts": removals})),
    );
    assert_eq!(status, 400, "{parsed}");
    assert_eq!(parsed["code"], json!("over_limit"), "{parsed}");
    assert!(
        parsed["error"]
            .as_str()
            .unwrap()
            .contains("per-request limit"),
        "{parsed}"
    );

    // A passage store tokenizes each source under the context lock, so
    // an oversized batch is refused before any of it lands — like an
    // association batch, one document's worth of sources per request.
    let passages: serde_json::Map<String, Value> =
        (0..1001).map(|i| (format!("s{i}"), json!("t"))).collect();
    let (status, parsed) = server.call(
        "POST",
        "/contexts/caps/sources",
        Some(json!({"passages": passages})),
    );
    assert_eq!(status, 400, "{parsed}");
    assert_eq!(parsed["code"], json!("over_limit"), "{parsed}");
    assert!(
        parsed["error"]
            .as_str()
            .unwrap()
            .contains("per-request limit"),
        "{parsed}"
    );
}

#[test]
fn queries_with_no_pinned_position_are_refused_but_one_field_is_enough() {
    let server = Server::start("empty-query");
    server.ok(
        "PUT",
        "/contexts/empty-query",
        Some(json!({"description": "d"})),
    );

    // Single-context query: omitting subject/label/object entirely
    // would otherwise materialize and rank every edge in the context.
    let (status, parsed) = server.call("POST", "/contexts/empty-query/query", Some(json!({})));
    assert_eq!(status, 400, "{parsed}");
    assert!(
        parsed["error"]
            .as_str()
            .unwrap()
            .contains("must pin at least one value"),
        "{parsed}"
    );

    // Explicit nulls are indistinguishable from omission.
    let (status, parsed) = server.call(
        "POST",
        "/contexts/empty-query/query",
        Some(json!({"subject": null, "label": null, "object": null})),
    );
    assert_eq!(status, 400, "{parsed}");
    assert!(
        parsed["error"]
            .as_str()
            .unwrap()
            .contains("must pin at least one value"),
        "{parsed}"
    );

    // Pinning just one of the three is enough to pass, even with no
    // matches to return.
    server.ok(
        "POST",
        "/contexts/empty-query/query",
        Some(json!({"subject": "x"})),
    );

    // The cross-context route refuses the same way...
    let (status, parsed) =
        server.call("POST", "/query", Some(json!({"contexts": ["empty-query"]})));
    assert_eq!(status, 400, "{parsed}");
    assert!(
        parsed["error"]
            .as_str()
            .unwrap()
            .contains("must pin at least one value"),
        "{parsed}"
    );

    // ...nulls fold into the same refusal there too...
    let (status, parsed) = server.call(
        "POST",
        "/query",
        Some(json!({
            "contexts": ["empty-query"],
            "subject": null,
            "label": null,
            "object": null
        })),
    );
    assert_eq!(status, 400, "{parsed}");
    assert!(
        parsed["error"]
            .as_str()
            .unwrap()
            .contains("must pin at least one value"),
        "{parsed}"
    );

    // ...and one pinned field passes cross-context too.
    server.ok(
        "POST",
        "/query",
        Some(json!({"contexts": ["empty-query"], "object": "x"})),
    );

    // The refusal reaches through the MCP tool-call path as well.
    let reply = server.call_tool(1, "query", json!({"context": "empty-query"}));
    assert_eq!(reply["isError"], true, "{reply}");
    let error_text = reply["content"][0]["text"].as_str().unwrap();
    assert!(
        error_text.contains("must pin at least one value"),
        "{error_text}"
    );
}

#[test]
fn resolve_caps_its_candidate_flood_like_every_other_match_endpoint() {
    let server = Server::start("resolve-cap");
    server.ok("PUT", "/contexts/flood", Some(json!({"description": "d"})));

    // 1001 concepts all containing the cue: uncapped, resolve would
    // serve every one of them in a single response body.
    let batch: Vec<Value> = (0..1001)
        .map(|i| {
            json!({
                "subject": format!("concept{i:04}"),
                "label": "は",
                "object": "x",
                "weight": 1.0
            })
        })
        .collect();
    server.ok("POST", "/contexts/flood/associations", Some(json!(batch)));

    // The cue is more than half of every stored spelling, so entry is
    // confident-lexical — no semantic tier, hermetic here. The ceiling
    // holds even with no limit in the request.
    let served = server.ok(
        "POST",
        "/contexts/flood/resolve",
        Some(json!({"cue": "concept"})),
    );
    assert_eq!(
        served.as_array().unwrap().len(),
        1000,
        "the default is the ceiling, not the whole vocabulary"
    );

    // An explicit limit picks the page size; best-first survives.
    let five = server.ok(
        "POST",
        "/contexts/flood/resolve",
        Some(json!({"cue": "concept", "limit": 5})),
    );
    assert_eq!(five.as_array().unwrap().len(), 5, "{five}");
}
