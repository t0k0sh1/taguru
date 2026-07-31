//! `POST /contexts/{name}/evidence` (#305, ADR 0006): opt-in evidence
//! assembly over the composed resolve/query/activate/search_passages/
//! cite_passage/communities-search fan-out, driven over the real HTTP
//! and MCP transports.

use serde_json::{Value, json};

use crate::support::*;

/// One context carrying both a graph fact and the passage it was
/// extracted from, so a single call exercises graph AND passage lanes
/// at once (the mixed-lane case #305's completion criteria name).
fn seed_mixed_corpus(server: &Server, name: &str) {
    server.ok(
        "PUT",
        &format!("/contexts/{name}"),
        Some(json!({"description": "酒蔵の記憶"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{name}/sources"),
        Some(json!({"passages": {
            "docs/kura.md": "青嶺酒造は雲居県霧沢町の蔵元である。杜氏は高瀬である。"
        }})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{name}/associations"),
        Some(json!([
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0,
             "source": "docs/kura.md", "paragraph": 0},
        ])),
    );
}

/// The opt-in package: mixed graph/passage evidence, complete
/// provenance, an unconfigured reranker, and every lane's plan.
#[test]
fn assembles_a_mixed_lane_provenance_complete_package() {
    let server = Server::start("evidence-mixed");
    seed_mixed_corpus(&server, "sake");

    let package = server.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"]})),
    );

    let items = package["items"].as_array().expect("items array");
    assert!(!items.is_empty(), "{package}");
    let kinds: std::collections::BTreeSet<&str> = items
        .iter()
        .map(|item| item["kind"].as_str().unwrap())
        .collect();
    assert!(
        kinds.contains("association") && kinds.contains("passage"),
        "expected both association and passage evidence: {package}"
    );

    // I1: every citation_refs locator has a matching top-level
    // citations entry, and vice versa.
    let citations = package["citations"].as_array().expect("citations array");
    for item in items {
        for reference in item["citation_refs"].as_array().unwrap_or(&Vec::new()) {
            assert!(
                citations
                    .iter()
                    .any(|entry| entry["source"] == reference["source"]
                        && entry["paragraph"] == reference["paragraph"]),
                "orphan citation_ref {reference} in {package}"
            );
        }
    }
    for entry in citations {
        assert!(
            items.iter().any(|item| item["citation_refs"]
                .as_array()
                .unwrap_or(&Vec::new())
                .iter()
                .any(|r| r["source"] == entry["source"] && r["paragraph"] == entry["paragraph"])),
            "unreferenced citation entry {entry} in {package}"
        );
    }

    // Every declared plan lane is present, and the ones this request
    // gives no reason to skip actually ran.
    let lanes = &package["plan"]["lanes"];
    for lane in [
        "resolve",
        "query",
        "activate",
        "passages",
        "communities",
        "citations",
    ] {
        assert!(lanes[lane]["ran"].is_boolean(), "{lane}: {package}");
    }
    assert_eq!(lanes["resolve"]["ran"], json!(true), "{package}");
    assert_eq!(lanes["activate"]["ran"], json!(true), "{package}");
    assert_eq!(lanes["passages"]["ran"], json!(true), "{package}");
    // No `labels` given: query does not pin any facet.
    assert_eq!(lanes["query"]["ran"], json!(false), "{package}");
    // include_communities defaults false.
    assert_eq!(lanes["communities"]["ran"], json!(false), "{package}");

    // No reranker provider exists in this tree yet (#307).
    assert_eq!(package["plan"]["reranker"]["configured"], json!(false));
    assert_eq!(package["plan"]["reranker"]["ran"], json!(false));

    assert!(package["budget"]["limits"]["max_items"].is_u64());
    assert_eq!(package["omitted_total"], json!(0), "{package}");
}

/// ADR 0006 §11: a missing communities artifact is a degrade for this
/// endpoint (200, `plan.lanes.communities.ran = false`), never the
/// refusal `communities/search` itself gives — contrasted directly
/// against that endpoint in the same test.
#[test]
fn include_communities_without_an_artifact_degrades_instead_of_refusing() {
    let server = Server::start("evidence-no-artifact");
    seed_mixed_corpus(&server, "sake");

    let (direct_status, direct_body) = server.call(
        "POST",
        "/contexts/sake/communities/search",
        Some(json!({"query": "テーマ"})),
    );
    assert_eq!(direct_status, 404, "{direct_body}");

    let package = server.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"], "include_communities": true})),
    );
    let communities_plan = &package["plan"]["lanes"]["communities"];
    assert_eq!(communities_plan["ran"], json!(false), "{package}");
    assert!(
        communities_plan["reason"]
            .as_str()
            .unwrap()
            .contains("taguru communities"),
        "{package}"
    );
}

/// ADR 0006 §8/§9: a budget too small for even the smallest candidate
/// is ordinary input, not an error — an empty package with every
/// candidate accounted for under `omitted`/`omitted_total`/
/// `omitted_by_reason`.
#[test]
fn a_tiny_budget_yields_an_empty_package_with_every_omission_accounted() {
    let server = Server::start("evidence-tiny-budget");
    seed_mixed_corpus(&server, "sake");

    let package = server.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"], "budget": {"max_items": 0}})),
    );
    assert_eq!(package["items"], json!([]), "{package}");
    assert!(package["omitted_total"].as_u64().unwrap() > 0, "{package}");
    assert!(
        package["omitted_by_reason"]["budget_exceeded"]
            .as_u64()
            .unwrap()
            > 0,
        "{package}"
    );
}

/// ADR 0006 §9 I3: the same request against the same corpus revision
/// produces the same package every time.
#[test]
fn selection_is_deterministic_across_repeated_calls() {
    let server = Server::start("evidence-determinism");
    seed_mixed_corpus(&server, "sake");

    let request = json!({"origins": ["青嶺酒造"]});
    let first = server.ok("POST", "/contexts/sake/evidence", Some(request.clone()));
    let second = server.ok("POST", "/contexts/sake/evidence", Some(request));
    assert_eq!(first, second);
}

/// The MCP tool routes onto the same endpoint and answers the same
/// package.
#[test]
fn the_assemble_evidence_tool_routes_to_the_same_endpoint() {
    let server = Server::start("evidence-mcp");
    seed_mixed_corpus(&server, "sake");

    let http_package = server.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"]})),
    );

    let result = server.call_tool(
        1,
        "assemble_evidence",
        json!({"context": "sake", "origins": ["青嶺酒造"]}),
    );
    assert!(result.get("isError").is_none(), "{result}");
    let text = result["content"][0]["text"].as_str().unwrap();
    // Tool content is the API envelope as JSON text; the payload sits
    // under `result` (the same convention `source_metadata.rs` pins).
    let mcp_package = serde_json::from_str::<Value>(text).unwrap()["result"].clone();

    assert_eq!(
        http_package["items"].as_array().unwrap().len(),
        mcp_package["items"].as_array().unwrap().len(),
        "{mcp_package}"
    );
    assert_eq!(http_package["citations"], mcp_package["citations"]);
}

/// Malformed input is refused before any lane runs: an over-cap
/// `origins` list (`over_limit`) and a wrong-shaped `budget`
/// (`malformed_request`, from the JSON extractor itself).
#[test]
fn malformed_input_is_refused_with_the_documented_error_codes() {
    let server = Server::start("evidence-errors");
    seed_mixed_corpus(&server, "sake");

    let too_many_origins: Vec<String> = (0..1001).map(|i| format!("cue{i}")).collect();
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": too_many_origins})),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("over_limit"), "{body}");

    // A well-formed-but-wrong-shaped `budget` fails the JSON extractor
    // itself, which axum answers 422 for (the same rejection-status
    // mapping `AppJson` documents) — the body still carries
    // `malformed_request`, the stable code a client branches on.
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"], "budget": "not-an-object"})),
    );
    assert_eq!(status, 422, "{body}");
    assert_eq!(body["code"], json!("malformed_request"), "{body}");
}

/// A read-scoped key reaches the endpoint; the existing direct
/// endpoints this call composes stay unaffected by its existence.
#[test]
fn a_read_scoped_key_reaches_evidence_assembly() {
    let server = Server::start_with_env(
        "evidence-scopes",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,reader:rtok"),
            ("TAGURU_KEY_SCOPES", r#"{"reader": "read"}"#),
        ],
    );
    let call = |method: &str, path: &str, body: Option<Value>, token: &str| {
        server.call_with_token(method, path, body, Some(token))
    };
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
            "POST",
            "/contexts/sake/sources",
            Some(json!({"passages": {
                "docs/kura.md": "青嶺酒造は雲居県霧沢町の蔵元である。杜氏は高瀬である。"
            }})),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "POST",
            "/contexts/sake/associations",
            Some(json!([
                {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0,
                 "source": "docs/kura.md", "paragraph": 0},
            ])),
            "atok"
        )
        .0,
        200
    );

    let (status, body) = server.call_with_token(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"]})),
        Some("rtok"),
    );
    assert_eq!(status, 200, "{body}");

    // The composed endpoint's own existence leaves the direct lanes it
    // fans out to unaffected.
    let (status, body) = server.call_with_token(
        "POST",
        "/contexts/sake/activate",
        Some(json!({"origins": ["青嶺酒造"]})),
        Some("rtok"),
    );
    assert_eq!(status, 200, "{body}");
}
