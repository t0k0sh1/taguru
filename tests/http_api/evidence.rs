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

    // No reranker configured on this server (#307): selection is
    // fully deterministic.
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

/// ADR 0006 §5.1's `query` lane runs only when `labels` pins a facet —
/// this is the only test in this file (or `assemble.rs`'s own `mod
/// tests`) that ever sends a non-empty `labels`, so the lane's own
/// contribution (`query_span`'s association count, and the
/// `labels.is_empty()` branch itself) was otherwise never exercised.
/// The same edge the graph-only `activate` lane already finds is also
/// reachable through `query` when `labels` names its own relation —
/// `fuse`'s exact-key dedup then collapses the two lane appearances
/// into one candidate, so a labeled call's `dedup_dropped` is strictly
/// greater than an unlabeled one's: proof the `query` lane actually
/// ran and returned real evidence, not merely that the response flag
/// flipped.
#[test]
fn the_query_lane_runs_only_when_labels_pins_a_facet_and_contributes_real_evidence() {
    let server = Server::start("evidence-query-lane");
    seed_mixed_corpus(&server, "sake");

    let unlabeled = server.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"]})),
    );
    assert_eq!(
        unlabeled["plan"]["lanes"]["query"]["ran"],
        json!(false),
        "{unlabeled}"
    );

    let labeled = server.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"], "labels": ["杜氏"]})),
    );
    assert_eq!(
        labeled["plan"]["lanes"]["query"]["ran"],
        json!(true),
        "{labeled}"
    );
    assert!(
        labeled["plan"]["selection"]["dedup_dropped"]
            .as_u64()
            .unwrap()
            > unlabeled["plan"]["selection"]["dedup_dropped"]
                .as_u64()
                .unwrap(),
        "the query lane's own copy of the 杜氏 edge must fold into the \
         activate lane's copy — unlabeled: {unlabeled}, labeled: {labeled}"
    );
}

/// ADR 0006 §5.1's `dice_floor`/`resolve_limit`/`semantic_floor` all
/// forward into the same `ResolveRequest` `assemble_evidence` builds
/// per origin cue — a typo cue resolves under the default floor but is
/// refused under a tightened one, the same fuzzy-tier/floor behavior
/// `POST /contexts/{name}/resolve` itself exercises directly.
/// Confirms the field actually reaches the resolve call rather than
/// being silently dropped on the way there.
#[test]
fn dice_floor_forwards_into_the_per_origin_resolve_call() {
    let server = Server::start("evidence-dice-floor");
    seed_mixed_corpus(&server, "sake");

    let lenient = server.ok(
        "POST",
        "/contexts/sake/evidence",
        // A typo of the stored concept "青嶺酒造" — the fuzzy tier
        // resolves it under the default floor.
        Some(json!({"origins": ["青嶺酒蔵"]})),
    );
    assert!(
        lenient["items"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["kind"] == json!("association")),
        "the default floor must resolve the typo to an anchor: {lenient}"
    );

    let strict = server.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒蔵"], "dice_floor": 0.9})),
    );
    assert!(
        !strict["items"]
            .as_array()
            .unwrap()
            .iter()
            .any(|item| item["kind"] == json!("association")),
        "a tightened floor must refuse the typo, leaving no anchor for \
         query/activate to run from: {strict}"
    );
    assert_eq!(
        strict["plan"]["lanes"]["activate"]["reason"],
        json!("no anchors resolved from 'origins'"),
        "{strict}"
    );
}

/// ADR 0006 §5.1's `activate_limit` forwards into `activate_excluding`'s
/// own limit argument (`assemble.rs`: `clamp(request.activate_limit,
/// 20, MAX_MATCH_LIMIT)`) — a corpus with two facts from the same
/// anchor returns both by default but only one when capped.
#[test]
fn activate_limit_forwards_into_the_activate_call() {
    let server = Server::start("evidence-activate-limit");
    seed_mixed_corpus(&server, "sake");
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "所在地", "object": "雲居県",
             "weight": 1.0, "source": "docs/kura.md", "paragraph": 0},
        ])),
    );

    let unlimited = server.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"]})),
    );
    let unlimited_associations = unlimited["items"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["kind"] == json!("association"))
        .count();
    assert!(unlimited_associations >= 2, "{unlimited}");

    let limited = server.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"], "activate_limit": 1})),
    );
    let limited_associations = limited["items"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|item| item["kind"] == json!("association"))
        .count();
    assert_eq!(
        limited_associations, 1,
        "activate_limit: 1 must cap the activate lane's own contribution to \
         exactly one association: {limited}"
    );
}

/// ADR 0006 §9 I1: an association attribution whose `(source,
/// paragraph)` was never stored as a passage (`CitationLookup::
/// UnknownSource`) is silently dropped from that item's own
/// `citation_refs` rather than ever creating an orphan reference — the
/// item itself is still admitted, only the unresolvable locator is
/// missing.
#[test]
fn an_attribution_to_a_never_stored_source_is_dropped_not_orphaned() {
    let server = Server::start("evidence-unresolvable-citation");
    seed_mixed_corpus(&server, "sake");
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "受賞歴", "object": "金賞",
             "weight": 1.0, "source": "ghost.md", "paragraph": 0},
        ])),
    );

    let package = server.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"]})),
    );
    let item = package["items"]
        .as_array()
        .unwrap()
        .iter()
        .find(|item| {
            item["kind"] == json!("association") && item["association"]["label"] == json!("受賞歴")
        })
        .unwrap_or_else(|| panic!("expected a 受賞歴 item: {package}"));
    assert!(
        item["citation_refs"].as_array().unwrap().is_empty(),
        "an unresolvable locator must be dropped, not left as an orphan \
         reference: {item}"
    );
    assert!(
        !package["citations"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["source"] == json!("ghost.md")),
        "ghost.md must never appear in the package's citations: {package}"
    );
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

    // Likewise a wrong-shaped `rerank` (#307, ADR 0006 §11: "a
    // malformed budget/rerank object is invalid_argument (400, same as
    // any other malformed request field)" — landing as the same
    // `malformed_request` code every other struct-typed field already
    // uses, via the JSON extractor itself).
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"], "rerank": 5})),
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

// --- #307: optional reranker ------------------------------------------

/// A Cohere/Jina-`/rerank`-shaped stub: reads `{model, query,
/// documents, top_n}`, hands the parsed request body AND the raw
/// request headers (lowercased, for an `Authorization` check) to
/// `respond`, and writes back whatever status/body `respond` returns.
/// `respond` runs on a fresh thread per connection, so a stub that
/// never answers (used to model an unreachable/slow provider) blocks
/// only that one request.
fn spawn_reranker_stub(
    respond: impl Fn(&Value, &str) -> (u16, String) + Send + Sync + 'static,
) -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let respond = std::sync::Arc::new(respond);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let respond = std::sync::Arc::clone(&respond);
            std::thread::spawn(move || {
                use std::io::{Read, Write};
                let mut buffer = Vec::new();
                let mut chunk = [0u8; 4096];
                let body_start = loop {
                    match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                    }
                    if let Some(at) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                        break at + 4;
                    }
                };
                let headers = String::from_utf8_lossy(&buffer[..body_start]).to_lowercase();
                let length: usize = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse().ok())?
                    })
                    .unwrap_or(0);
                while buffer.len() < body_start + length {
                    match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                    }
                }
                let request: Value =
                    serde_json::from_slice(&buffer[body_start..body_start + length])
                        .unwrap_or(Value::Null);
                let (status, body) = respond(&request, &headers);
                let status_line = match status {
                    200 => "200 OK",
                    500 => "500 Internal Server Error",
                    other => panic!("unhandled stub status {other}"),
                };
                let response = format!(
                    "HTTP/1.1 {status_line}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
            });
        }
    });
    format!("http://{addr}")
}

fn rerank_env(url: &str, model: &str, timeout_secs: &str) -> Vec<(&'static str, String)> {
    vec![
        ("TAGURU_RERANK_URL", url.to_string()),
        ("TAGURU_RERANK_MODEL", model.to_string()),
        ("TAGURU_RERANK_TIMEOUT_SECS", timeout_secs.to_string()),
    ]
}

fn start_with_owned_env(tag: &str, env: &[(&'static str, String)]) -> Server {
    let borrowed: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
    Server::start_with_env(tag, &borrowed)
}

/// Strips `plan.reranker`, the one fragment expected to legitimately
/// differ between a reranker-configured and an unconfigured call, so
/// the rest of the package can be compared for byte-identical content.
fn without_reranker_plan(mut package: Value) -> Value {
    package["plan"]["reranker"] = Value::Null;
    package
}

/// A configured reranker that reverses the fused pool reorders
/// admission (the reranker's entire observable effect, ADR 0006 §12)
/// and reports its own model identity — never touching anything else
/// in the package.
#[test]
fn a_configured_reranker_reorders_admission_and_reports_its_model() {
    let stub = spawn_reranker_stub(|request, _headers| {
        let top_n = request["top_n"].as_u64().unwrap_or(0) as usize;
        let results: Vec<Value> = (0..top_n)
            .rev()
            .map(|index| json!({"index": index, "relevance_score": 1.0}))
            .collect();
        (200, json!({"results": results}).to_string())
    });
    let server = start_with_owned_env(
        "evidence-rerank-reorders",
        &rerank_env(&stub, "stub-reranker", "5"),
    );
    seed_mixed_corpus(&server, "sake");

    let baseline = server.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"]})),
    );
    let reranked = server.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"], "rerank": {}})),
    );

    assert_eq!(
        reranked["plan"]["reranker"],
        json!({"configured": true, "ran": true, "model": "stub-reranker"}),
        "{reranked}"
    );
    let baseline_order: Vec<&str> = baseline["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["candidate_id"].as_str().unwrap())
        .collect();
    let reranked_order: Vec<&str> = reranked["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["candidate_id"].as_str().unwrap())
        .collect();
    assert_eq!(baseline_order.len(), reranked_order.len(), "{reranked}");
    assert_ne!(
        baseline_order, reranked_order,
        "a full reversal of >=2 candidates must change admission order:\n\
         baseline: {baseline}\nreranked: {reranked}"
    );
    // Reordering alone must not change WHICH candidates were admitted
    // (both requests share the same generous default budget), only
    // their order.
    let baseline_set: std::collections::BTreeSet<&str> = baseline_order.into_iter().collect();
    let reranked_set: std::collections::BTreeSet<&str> = reranked_order.into_iter().collect();
    assert_eq!(baseline_set, reranked_set, "{baseline} vs {reranked}");
}

/// `rerank` configured but not named on a call reports `configured:
/// true, ran: false` with no `reason` — a caller did not ask, so there
/// is nothing to explain.
#[test]
fn a_configured_reranker_left_unrequested_reports_ran_false_with_no_reason() {
    let stub = spawn_reranker_stub(|_request, _headers| (200, json!({"results": []}).to_string()));
    let server = start_with_owned_env(
        "evidence-rerank-unrequested",
        &rerank_env(&stub, "stub-reranker", "5"),
    );
    seed_mixed_corpus(&server, "sake");

    let package = server.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"]})),
    );
    assert_eq!(
        package["plan"]["reranker"],
        json!({"configured": true, "ran": false}),
        "{package}"
    );
}

/// A provider that answers 500, and one that is simply unreachable,
/// both degrade to the exact same deterministic package a fully
/// unconfigured server would answer — 200 either way, never a
/// call-ending refusal (ADR 0006 §11) — with `plan.reranker.reason`
/// naming `provider_error`.
#[test]
fn a_failing_or_unreachable_reranker_degrades_to_the_unconfigured_package() {
    let unconfigured = Server::start("evidence-rerank-baseline");
    seed_mixed_corpus(&unconfigured, "sake");
    let baseline = unconfigured.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"]})),
    );

    let failing_stub = spawn_reranker_stub(|_request, _headers| (500, "{}".to_string()));
    for (tag, url) in [
        ("evidence-rerank-500", failing_stub.as_str()),
        // A closed local port: nothing ever answers (the same
        // deliberately-dead-provider trick `retrieval_core.rs` uses).
        ("evidence-rerank-unreachable", "http://127.0.0.1:9"),
    ] {
        let server = start_with_owned_env(tag, &rerank_env(url, "stub-reranker", "2"));
        seed_mixed_corpus(&server, "sake");
        let degraded = server.ok(
            "POST",
            "/contexts/sake/evidence",
            Some(json!({"origins": ["青嶺酒造"], "rerank": {}})),
        );
        assert_eq!(
            degraded["plan"]["reranker"]["configured"],
            json!(true),
            "{tag}: {degraded}"
        );
        assert_eq!(
            degraded["plan"]["reranker"]["ran"],
            json!(false),
            "{tag}: {degraded}"
        );
        assert_eq!(
            degraded["plan"]["reranker"]["reason"],
            json!("provider_error"),
            "{tag}: {degraded}"
        );
        assert_eq!(
            without_reranker_plan(degraded),
            without_reranker_plan(baseline.clone()),
            "{tag}: degrade must be byte-identical to the unconfigured package \
             outside plan.reranker itself"
        );
    }
}

/// A provider response that is not a complete permutation (here: a
/// repeated index) degrades to the same deterministic package a fully
/// unconfigured server would answer, with `reason: "invalid_permutation"`
/// — the response was well-formed JSON, just not a valid reordering
/// (ADR 0006 §12).
///
/// A `timeout` HTTP-level counterpart is deliberately not included
/// here: `HttpReranker`'s per-attempt ureq timeout is
/// `min(TAGURU_RERANK_TIMEOUT_SECS, deadline.remaining())`, so making
/// the REQUEST's own deadline (not the reranker's) the binding
/// constraint — the only way `plan.reranker.reason` becomes `"timeout"`
/// rather than `"provider_error"` — necessarily races
/// `enforce_timeout`'s own `tokio::time::timeout` in `src/limits.rs`,
/// which is keyed to the exact same deadline. That race would make an
/// HTTP-level test either flaky or a guaranteed 408 depending on
/// scheduling, not a meaningful assertion. The identical branch is
/// already covered deterministically at the unit level:
/// `api::evidence::rerank::tests::a_deadline_driven_transport_timeout_is_reported_as_timeout`.
#[test]
fn an_invalid_permutation_degrades_to_the_unconfigured_package() {
    let unconfigured = Server::start("evidence-rerank-invalid-baseline");
    seed_mixed_corpus(&unconfigured, "sake");
    let baseline = unconfigured.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"]})),
    );

    let stub = spawn_reranker_stub(|_request, _headers| {
        // A repeated index: not a complete permutation of 0..len for
        // any pool of >= 2 candidates.
        (
            200,
            json!({"results": [{"index": 0}, {"index": 0}]}).to_string(),
        )
    });
    let server = start_with_owned_env(
        "evidence-rerank-invalid",
        &rerank_env(&stub, "stub-reranker", "5"),
    );
    seed_mixed_corpus(&server, "sake");
    let degraded = server.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"], "rerank": {}})),
    );
    assert_eq!(
        degraded["plan"]["reranker"],
        json!({"configured": true, "ran": false, "reason": "invalid_permutation"}),
        "{degraded}"
    );
    assert_eq!(
        without_reranker_plan(degraded),
        without_reranker_plan(baseline),
        "an invalid permutation must degrade byte-identically to the \
         unconfigured package outside plan.reranker itself"
    );
}

/// `rerank.model` naming a model the configured provider does not
/// serve degrades without ever touching the provider.
#[test]
fn a_rerank_model_mismatch_degrades_without_calling_the_provider() {
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counted = std::sync::Arc::clone(&calls);
    let stub = spawn_reranker_stub(move |_request, _headers| {
        counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        (200, json!({"results": []}).to_string())
    });
    let server = start_with_owned_env(
        "evidence-rerank-mismatch",
        &rerank_env(&stub, "stub-reranker", "5"),
    );
    seed_mixed_corpus(&server, "sake");

    let package = server.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"], "rerank": {"model": "a-different-model"}})),
    );
    assert_eq!(
        package["plan"]["reranker"],
        json!({"configured": true, "ran": false, "reason": "model_mismatch"}),
        "{package}"
    );
    assert_eq!(
        calls.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "{package}"
    );
}

/// Candidate text — the Japanese passage/association content this
/// corpus carries — reaches the configured reranker provider (that is
/// its whole job) and NOWHERE else: never the response body outside
/// the pass-through `association`/`passage` payload fields already
/// present without reranking, never `GET /metrics`. Nor does the
/// reranker's own `TAGURU_RERANK_API_KEY` ever appear anywhere a
/// client can read (ADR 0006 §12).
#[test]
fn reranker_privacy_leaks_no_candidate_text_or_credential_into_metrics() {
    const SECRET_KEY: &str = "sekrit-rerank-key-should-never-leak";
    let received_auth = std::sync::Arc::new(std::sync::Mutex::new(None::<String>));
    let captured = std::sync::Arc::clone(&received_auth);
    let stub = spawn_reranker_stub(move |request, headers| {
        // The provider legitimately receives candidate text and the
        // credential — that IS its job — but nothing else ever should.
        let documents = request["documents"].as_array().cloned().unwrap_or_default();
        assert!(
            documents
                .iter()
                .any(|d| d.as_str().unwrap_or("").contains("高瀬")),
            "the stub itself should have received candidate text: {request}"
        );
        let auth = headers
            .lines()
            .find_map(|line| line.strip_prefix("authorization:"))
            .map(|value| value.trim().to_string());
        *captured.lock().unwrap() = auth;
        (500, "{}".to_string())
    });
    let server = start_with_owned_env(
        "evidence-rerank-privacy",
        &[
            ("TAGURU_RERANK_URL", stub.clone()),
            ("TAGURU_RERANK_MODEL", "stub-reranker".to_string()),
            ("TAGURU_RERANK_TIMEOUT_SECS", "2".to_string()),
            ("TAGURU_RERANK_API_KEY", SECRET_KEY.to_string()),
        ],
    );
    seed_mixed_corpus(&server, "sake");

    let package = server.ok(
        "POST",
        "/contexts/sake/evidence",
        Some(json!({"origins": ["青嶺酒造"], "rerank": {}})),
    );
    assert_eq!(
        package["plan"]["reranker"]["reason"],
        json!("provider_error"),
        "{package}"
    );

    // The provider itself DID receive the credential (proving the key
    // is actually sent, not silently dropped) — but only there.
    assert_eq!(
        received_auth.lock().unwrap().as_deref(),
        Some(format!("bearer {SECRET_KEY}").as_str()),
        "the stub must have received the Authorization header"
    );

    // `plan.reranker` — the ONLY reranker-shaped fragment of the
    // response — carries neither the candidate text nor the key.
    // `items[].association.object` legitimately contains "高瀬" as the
    // ordinary pass-through wire payload (ADR 0006 §10) unrelated to
    // reranking, so the check is scoped to `plan.reranker`, not the
    // whole package.
    let reranker_plan_text = package["plan"]["reranker"].to_string();
    assert!(
        !reranker_plan_text.contains(SECRET_KEY),
        "the API key must never reach plan.reranker: {reranker_plan_text}"
    );
    assert!(
        !reranker_plan_text.contains("高瀬"),
        "candidate text must never reach plan.reranker: {reranker_plan_text}"
    );

    let (metrics_status, metrics_body) = server.call("GET", "/metrics", None);
    assert_eq!(metrics_status, 200);
    let metrics_text = metrics_body.as_str().expect("metrics body is text");
    assert!(
        !metrics_text.contains(SECRET_KEY),
        "the API key must never reach /metrics"
    );
    assert!(
        !metrics_text.contains("高瀬"),
        "candidate text must never reach /metrics"
    );
    assert!(
        metrics_text.contains("taguru_rerank_outcomes_total{outcome=\"provider_error\"} 1"),
        "{metrics_text}"
    );
    assert!(
        metrics_text.contains("taguru_rerank_breaker_state"),
        "a configured provider's breaker family must render: {metrics_text}"
    );
}
