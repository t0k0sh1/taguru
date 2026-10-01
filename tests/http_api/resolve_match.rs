//! Validation and capping shared by every candidate-matching endpoint.

use serde_json::{Value, json};

use crate::support::*;

#[test]
fn lookalike_candidates_carry_the_evidence_to_tell_them_apart() {
    let server = Server::start("lookalikes");
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": "looks", "description": "字面の近い別物たち"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx("looks")),
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
        &format!("/contexts/{}/resolve", server.cx("looks")),
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
        &format!("/contexts/{}/resolve", server.cx("looks")),
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
        &format!("/contexts/{}/resolve", server.cx("looks")),
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
        &format!("/contexts/{}/resolve_label", server.cx("looks")),
        Some(json!({"cue": "means"})),
    );
    assert_eq!(label[0]["kind"], json!("exact"));
    assert!(
        label[0]["gloss"].as_str().unwrap().contains("means"),
        "{label}"
    );
}

/// A cue that normalizes to the empty string (here, simply the empty
/// string itself — `Context::normalize`'s NFKC+lowercase+kana-fold
/// chain never shrinks a non-empty input to nothing) must degrade to
/// an empty result list, not an error: `EntryIndex::resolve` refuses
/// an empty needle outright, and both HTTP handlers pass it straight
/// through with no validation of their own.
#[test]
fn an_empty_cue_resolves_to_no_candidates_not_an_error() {
    let server = Server::start("empty-cue");
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": "empty-cue", "description": "d"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx("empty-cue")),
        Some(json!([
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0},
        ])),
    );

    let resolved = server.ok(
        "POST",
        &format!("/contexts/{}/resolve", server.cx("empty-cue")),
        Some(json!({"cue": ""})),
    );
    assert_eq!(resolved, json!([]), "{resolved}");

    let labels = server.ok(
        "POST",
        &format!("/contexts/{}/resolve_label", server.cx("empty-cue")),
        Some(json!({"cue": ""})),
    );
    assert_eq!(labels, json!([]), "{labels}");
}

#[test]
fn oversized_input_lists_are_refused_before_any_work() {
    let server = Server::start("input-caps");
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": "caps", "description": "d"})),
    );

    // 1001 items trips every list-shaped read input.
    let over: Vec<String> = (0..1001).map(|i| format!("o{i}")).collect();
    let caps = server.cx("caps");
    for (path, body) in [
        (
            format!("/contexts/{caps}/explore"),
            json!({"origins": over.clone()}),
        ),
        (
            format!("/contexts/{caps}/activate"),
            json!({"origins": over.clone()}),
        ),
        (
            format!("/contexts/{caps}/unreachable_from"),
            json!({"origins": over.clone()}),
        ),
        (
            format!("/contexts/{caps}/query"),
            json!({"subject": over.clone()}),
        ),
        (
            format!("/contexts/{caps}/sources/lookup"),
            json!({"sources": over.clone()}),
        ),
    ] {
        let path = path.as_str();
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
        &format!("/contexts/{}/explore", server.cx("caps")),
        Some(json!({"origins": at_cap})),
    );

    // Alias batches are WAL writes and share the association batch cap.
    let aliases: serde_json::Map<String, Value> = (0..10_001)
        .map(|i| (format!("a{i}"), json!("青嶺酒造")))
        .collect();
    let (status, parsed) = server.call(
        "POST",
        &format!("/contexts/{}/aliases", server.cx("caps")),
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
        &format!("/contexts/{}/aliases", server.cx("caps")),
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
        &format!("/contexts/{}/sources", server.cx("caps")),
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
        "POST",
        "/contexts",
        Some(json!({"name": "empty-query", "description": "d"})),
    );

    // Single-context query: omitting subject/label/object entirely
    // would otherwise materialize and rank every edge in the context.
    let (status, parsed) = server.call(
        "POST",
        &format!("/contexts/{}/query", server.cx("empty-query")),
        Some(json!({})),
    );
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
        &format!("/contexts/{}/query", server.cx("empty-query")),
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
        &format!("/contexts/{}/query", server.cx("empty-query")),
        Some(json!({"subject": "x"})),
    );

    // The cross-context route refuses the same way...
    let (status, parsed) = server.call(
        "POST",
        "/query",
        Some(json!({"context_ids": [server.cx("empty-query")]})),
    );
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
            "context_ids": [server.cx("empty-query")],
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
        Some(json!({"context_ids": [server.cx("empty-query")], "object": "x"})),
    );

    // The refusal reaches through the MCP tool-call path as well.
    let reply = server.call_tool(1, "query", json!({"context": server.cx("empty-query")}));
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
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": "flood", "description": "d"})),
    );

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
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx("flood")),
        Some(json!(batch)),
    );

    // The cue is more than half of every stored spelling, so entry is
    // confident-lexical — no semantic tier, hermetic here. The ceiling
    // holds even with no limit in the request.
    let served = server.ok(
        "POST",
        &format!("/contexts/{}/resolve", server.cx("flood")),
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
        &format!("/contexts/{}/resolve", server.cx("flood")),
        Some(json!({"cue": "concept", "limit": 5})),
    );
    assert_eq!(five.as_array().unwrap().len(), 5, "{five}");
}

/// `explore` and `paths` count toward `usage.empty_reads` exactly when
/// they answer nothing — a productive walk must not read as an empty
/// one (the directory's routing signal), and vice versa.
#[test]
fn explore_and_paths_count_empty_reads_only_when_empty() {
    let server = Server::start("explore-empty-reads");
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": "walks", "description": "d"})),
    );
    let id = server.cx("walks");
    server.ok(
        "POST",
        &format!("/contexts/{id}/associations"),
        Some(json!([
            {"subject": "a", "label": "l", "object": "b", "weight": 1.0},
            // A disconnected island, so a paths call between the two
            // components can answer zero trails without a refusal.
            {"subject": "c", "label": "l", "object": "d", "weight": 1.0},
        ])),
    );
    let empty_reads = |server: &Server| {
        server.ok("GET", &format!("/contexts/{id}"), None)["usage"]["empty_reads"].clone()
    };

    server.ok(
        "POST",
        &format!("/contexts/{id}/explore"),
        Some(json!({"origins": ["a"]})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{id}/paths"),
        Some(json!({"origins": ["a"], "targets": ["b"]})),
    );
    assert_eq!(
        empty_reads(&server),
        json!(0),
        "productive walks are not empty reads"
    );

    server.ok(
        "POST",
        &format!("/contexts/{id}/explore"),
        Some(json!({"origins": ["ghost"]})),
    );
    assert_eq!(empty_reads(&server), json!(1), "an empty explore counts");
    server.ok(
        "POST",
        &format!("/contexts/{id}/paths"),
        Some(json!({"origins": ["a"], "targets": ["c"]})),
    );
    assert_eq!(
        empty_reads(&server),
        json!(2),
        "a pathless paths call counts"
    );
}
