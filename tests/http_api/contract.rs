//! Golden wire-contract fixtures (#301, ADR 0005 §9): the machine-
//! readable pin of the current `http_contract: 1` / `mcp_contract: 1`
//! shapes, including #216's evidence-assembly package (#305). Every
//! fixture under `tests/fixtures/wire/{http,mcp}/` is produced and
//! verified here, against the real server binary — Python and
//! TypeScript check the same committed files structurally
//! (`sdk/python/tests/unit/test_wire_contract.py`,
//! `sdk/typescript/tests/unit/wire-contract.test.ts`), and
//! `sdk/spec/check_contract.py` diffs them across a base ref so an
//! unclassified breaking change fails CI instead of shipping quietly.
//!
//! A drift here does not by itself mean a bug — see
//! `tests/fixtures/wire/README.md` for how to classify and regenerate:
//! `TAGURU_UPDATE_WIRE_FIXTURES=1 cargo test --test http_api contract`
//! rewrites every fixture this module owns from a live server.

use std::path::PathBuf;

use serde_json::{Value, json};

use crate::support::*;

fn wire_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/wire")
}

fn should_update() -> bool {
    std::env::var_os("TAGURU_UPDATE_WIRE_FIXTURES").is_some()
}

/// Blanks fields whose real value is either build-specific (`server`,
/// `version` — this crate's own SemVer), call-specific (`time`, every
/// `ApiResponse`/`ApiError`'s elapsed-seconds field), or wall-clock
/// specific (`last_read_epoch`/`last_write_epoch`, a directory entry's
/// own usage stamps) — the same list `shapes.json`'s `volatile_fields`
/// names, so a version bump, a slow CI run, or the literal wall time a
/// fixture happened to regenerate at never reads as wire drift. Runs
/// before a fixture is written OR compared, so a committed fixture
/// always already carries the placeholder and a plain
/// `assert_eq!`/`git diff` needs no special-casing.
///
/// An MCP tool result carries the whole HTTP body a second time, as a
/// JSON string inside `content[].text` (the pass-through convention
/// ADR 0005 §2.4 documents) — every volatile field inside that string
/// needs the same treatment, so a string value that itself parses as a
/// JSON object/array is recursively normalized and re-encoded in
/// place, not left as opaque text.
fn normalize_volatile(value: &mut Value) {
    match value {
        Value::Object(map) => {
            if matches!(map.get("time"), Some(Value::Number(_))) {
                map.insert("time".to_string(), json!(0.0));
            }
            if matches!(map.get("server"), Some(Value::String(_))) {
                map.insert("server".to_string(), json!("0.0.0"));
            }
            if let Some(Value::String(text)) = map.get("version")
                && text.split('.').count() >= 2
            {
                map.insert("version".to_string(), json!("0.0.0"));
            }
            for key in ["last_read_epoch", "last_write_epoch"] {
                if matches!(map.get(key), Some(Value::Number(_))) {
                    map.insert(key.to_string(), json!(0));
                }
            }
            for child in map.values_mut() {
                normalize_volatile(child);
            }
        }
        Value::Array(items) => {
            for item in items {
                normalize_volatile(item);
            }
        }
        Value::String(text) => {
            if let Ok(mut inner) = serde_json::from_str::<Value>(text)
                && (inner.is_object() || inner.is_array())
            {
                normalize_volatile(&mut inner);
                *text = serde_json::to_string(&inner).expect("a Value always re-serializes");
            }
        }
        _ => {}
    }
}

/// Checks one fixture against `tests/fixtures/wire/{transport}/{operation}.json`,
/// or (`TAGURU_UPDATE_WIRE_FIXTURES=1`) rewrites it from `fixture`.
fn check_or_update(transport: &str, operation: &str, mut fixture: Value) {
    normalize_volatile(&mut fixture);
    let path = wire_dir().join(transport).join(format!("{operation}.json"));
    if should_update() {
        let pretty = serde_json::to_string_pretty(&fixture).expect("fixture must serialize") + "\n";
        std::fs::write(&path, pretty)
            .unwrap_or_else(|error| panic!("{path:?} must be writable: {error}"));
        return;
    }
    let committed_text = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "missing wire fixture {path:?} ({error}) — run \
             `TAGURU_UPDATE_WIRE_FIXTURES=1 cargo test --test http_api contract` \
             to create it, then read tests/fixtures/wire/README.md before committing"
        )
    });
    let committed: Value = serde_json::from_str(&committed_text)
        .unwrap_or_else(|error| panic!("{path:?} is not valid JSON: {error}"));
    assert_eq!(
        committed, fixture,
        "wire fixture drift at {path:?} — classify the change against ADR 0005 §4 \
         (tests/fixtures/wire/README.md), then regenerate with \
         TAGURU_UPDATE_WIRE_FIXTURES=1 if it's intentional"
    );
}

fn http_fixture(
    operation: &str,
    method: &str,
    route: &str,
    request: Option<Value>,
    status: u16,
    response: Value,
) {
    check_or_update(
        "http",
        operation,
        json!({
            "operation": operation,
            "contract": "http_contract",
            "method": method,
            "route": route,
            "status": status,
            "request": request,
            "response": response,
        }),
    );
}

/// [`http_fixture`] pinned to `POST /contexts/{name}/evidence` — every
/// evidence-assembly and evidence-error fixture below targets this one
/// endpoint, so only `operation`/`request`/`status`/`response` vary.
fn evidence_fixture(operation: &str, request: Value, status: u16, response: Value) {
    http_fixture(
        operation,
        "POST",
        "/contexts/{name}/evidence",
        Some(request),
        status,
        response,
    );
}

fn mcp_fixture(operation: &str, route: &str, request: Value, status: u16, response: Value) {
    check_or_update(
        "mcp",
        operation,
        json!({
            "operation": operation,
            "contract": "mcp_contract",
            "route": route,
            "status": status,
            "request": request,
            "response": response,
        }),
    );
}

// --- HTTP: probes ---

#[test]
fn version_and_health() {
    let server = Server::start("contract-probes");

    let (status, body) = server.call("GET", "/version", None);
    assert_eq!(status, 200, "{body}");
    http_fixture("version", "GET", "/version", None, status, body);

    let (status, body) = server.call("GET", "/health", None);
    assert_eq!(status, 200, "{body}");
    http_fixture("health", "GET", "/health", None, status, body);
}

// --- HTTP: graph search envelopes ---

fn seed_basic_corpus(server: &Server, name: &str) {
    server.ok(
        "PUT",
        &format!("/contexts/{name}"),
        Some(json!({"description": "wire-contract corpus"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{name}/associations"),
        Some(json!([
            {"subject": "alpha", "label": "connects_to", "object": "beta", "weight": 2.0,
             "source": "doc.md", "paragraph": 0},
        ])),
    );
    // A locator (ADR 0007 §7) on the same (source, paragraph) the
    // association above names, so `attributions[].locator` in the
    // recall/explore/activate wire fixtures carries a real value, not
    // just an always-null field the golden could never actually prove.
    server.ok(
        "POST",
        &format!("/contexts/{name}/sources"),
        Some(json!({
            "passages": {"doc.md": "alpha connects to beta."},
            "locators": {"doc.md": [{"paragraph": 0, "locator": {"kind": "page", "value": "1"}}]}
        })),
    );
}

#[test]
fn recall_match_page_and_contexts_list() {
    let server = Server::start("contract-recall");
    seed_basic_corpus(&server, "corpus-a");

    let request = json!({"cue": "alpha"});
    let (status, body) = server.call("POST", "/contexts/corpus-a/recall", Some(request.clone()));
    assert_eq!(status, 200, "{body}");
    assert!(
        !body["result"]["matches"].as_array().unwrap().is_empty(),
        "{body}"
    );
    http_fixture(
        "recall",
        "POST",
        "/contexts/{name}/recall",
        Some(request),
        status,
        body,
    );

    let (status, body) = server.call("GET", "/contexts", None);
    assert_eq!(status, 200, "{body}");
    assert!(
        !body["result"]["contexts"].as_array().unwrap().is_empty(),
        "{body}"
    );
    http_fixture("contexts_list", "GET", "/contexts", None, status, body);
}

#[test]
fn explore_and_activate_pages() {
    let server = Server::start("contract-explore");
    seed_basic_corpus(&server, "corpus-b");

    let request = json!({"origins": ["alpha"]});
    let (status, body) = server.call("POST", "/contexts/corpus-b/explore", Some(request.clone()));
    assert_eq!(status, 200, "{body}");
    http_fixture(
        "explore",
        "POST",
        "/contexts/{name}/explore",
        Some(request),
        status,
        body,
    );

    let request = json!({"origins": ["alpha"]});
    let (status, body) = server.call("POST", "/contexts/corpus-b/activate", Some(request.clone()));
    assert_eq!(status, 200, "{body}");
    assert!(
        !body["result"]["matches"].as_array().unwrap().is_empty(),
        "{body}"
    );
    http_fixture(
        "activate",
        "POST",
        "/contexts/{name}/activate",
        Some(request),
        status,
        body,
    );
}

// --- HTTP: passage and community search — PassagePage is the 0.4.0
// incident shape (ADR 0005 §2.1), CommunityPage the richest of the
// thirteen pagination envelopes. ---

#[test]
fn sources_search_passage_page() {
    let server = Server::start("contract-sources-search");
    server.ok("PUT", "/contexts/corpus-c", None);
    server.ok(
        "POST",
        "/contexts/corpus-c/sources",
        Some(json!({"passages": {
            "doc.md": "青嶺酒造は雲居県霧沢町の蔵元である。"
        }})),
    );

    let request = json!({"query": "酒造"});
    let (status, body) = server.call(
        "POST",
        "/contexts/corpus-c/sources/search",
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert!(
        !body["result"]["hits"].as_array().unwrap().is_empty(),
        "{body}"
    );
    http_fixture(
        "sources_search",
        "POST",
        "/contexts/{name}/sources/search",
        Some(request),
        status,
        body,
    );
}

/// A community artifact built by hand through the same API `taguru
/// communities` itself writes through (the pattern
/// `tests/http_api/communities.rs::search_refuses_without_an_artifact_and_verdicts_staleness_with_one`
/// already uses) — no LLM stub needed for a deterministic fixture.
#[test]
fn communities_search_community_page() {
    let server = Server::start("contract-communities-search");
    server.ok("PUT", "/contexts/corpus-d", None);
    server.ok(
        "POST",
        "/contexts/corpus-d/associations",
        Some(json!([
            {"subject": "a1", "label": "近い", "object": "a2", "weight": 2.0},
        ])),
    );
    let revision = server.ok("GET", "/contexts/corpus-d", None)["revision"].clone();
    server.ok("PUT", "/contexts/corpus-d::communities", None);
    let manifest = json!({
        "taguru_communities": 1,
        "algorithm": "louvain-cc/1",
        "source_context": "corpus-d",
        "revision": revision,
        "levels": 1,
        "communities": [
            {"id": "L0-0", "level": 0, "fingerprint": "00aa00aa00aa00aa", "concept_count": 2},
        ],
    });
    server.ok(
        "POST",
        "/contexts/corpus-d::communities/sources",
        Some(json!({"passages": {
            "community:L0-0": "この共同体のテーマは酒造りの歴史です。",
            "communities:manifest": manifest.to_string(),
        }})),
    );
    server.ok(
        "POST",
        "/contexts/corpus-d::communities/associations",
        Some(json!([
            {"subject": "community:L0-0", "label": "contains", "object": "a1", "weight": 6.0},
            {"subject": "community:L0-0", "label": "contains", "object": "a2", "weight": 4.0},
        ])),
    );

    let request = json!({"query": "酒造りの歴史"});
    let (status, body) = server.call(
        "POST",
        "/contexts/corpus-d/communities/search",
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert!(
        !body["result"]["hits"].as_array().unwrap().is_empty(),
        "{body}"
    );
    http_fixture(
        "communities_search",
        "POST",
        "/contexts/{name}/communities/search",
        Some(request),
        status,
        body,
    );
}

// --- HTTP: passage storage and batch import (#346, ADR 0007 §7) — the
// two write paths a citation `locator` can ride in on. ---

#[test]
fn store_passages_response_shape() {
    let server = Server::start("contract-store-passages");
    server.ok("PUT", "/contexts/corpus-e", None);

    let request = json!({
        "passages": {"doc.md": "導入。\n\n本編。"},
        "sections": {"doc.md": [{"paragraph": 1, "section": "本編"}]},
        "locators": {"doc.md": [{"paragraph": 1, "locator": {"kind": "page", "value": "12"}}]},
    });
    let (status, body) = server.call("POST", "/contexts/corpus-e/sources", Some(request.clone()));
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"]["locators_stored"], json!(1), "{body}");
    http_fixture(
        "store_passages",
        "POST",
        "/contexts/{name}/sources",
        Some(request),
        status,
        body,
    );
}

/// S5 (#383): `warn` mode's `issues`/`schema_violations` on
/// `POST /contexts/{name}/associations` are new, wire-visible fields
/// on `ApiResponse` — additive (`HTTP_CONTRACT` unchanged, both are
/// `skip_serializing_if`-omitted on every response with nothing to
/// say), but still a shape an SDK consumer needs pinned so it stops
/// being invisible to the cross-language contract check the moment a
/// context actually turns `warn` on.
#[test]
fn add_associations_warn_mode_response_shape() {
    let server = Server::start("contract-associations-warn");
    server.ok("PUT", "/contexts/corpus-g", None);
    server.ok(
        "PUT",
        "/contexts/corpus-g/schema",
        Some(json!({
            "schema": 1,
            "mode": "warn",
            "closed_labels": false,
            "types": {"Brewery": {"is_a": []}, "Person": {"is_a": []}},
            "relations": {"杜氏": {"domain": ["Brewery"], "range": ["Person"]}}
        })),
    );

    let request = json!([
        {"subject": "高瀬", "label": "schema:type", "object": "Person", "weight": 1.0, "source": "a.md"},
        {"subject": "高瀬", "label": "杜氏", "object": "個人A", "weight": 1.0, "source": "a.md"},
    ]);
    let (status, body) = server.call(
        "POST",
        "/contexts/corpus-g/associations",
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"], json!(2), "{body}");
    assert_eq!(body["schema_violations"], json!(1), "{body}");
    assert!(body["issues"].is_array(), "{body}");
    http_fixture(
        "add_associations_warn",
        "POST",
        "/contexts/{name}/associations",
        Some(request),
        status,
        body,
    );
}

#[test]
fn import_reports_locator_bookkeeping() {
    let server = Server::start("contract-import");
    let batch = "{\"taguru_batch\": 1, \"context\": \"corpus-f\", \"source\": \"doc.md\", \
                 \"create\": {\"description\": \"wire-contract import corpus\"}}\n\
                 {\"passage\": \"導入。\\n\\n本編。\"}\n\
                 {\"paragraph\": 1, \"locator\": {\"kind\": \"page\", \"value\": \"12\"}}\n\
                 {\"subject\": \"alpha\", \"label\": \"connects_to\", \"object\": \"beta\", \
                 \"weight\": 1.0}\n";
    let (status, body) = post_import(&server, batch, None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["result"]["batches"][0]["locators_stored"],
        json!(1),
        "{body}"
    );
    http_fixture(
        "import",
        "POST",
        "/import",
        Some(json!(batch)),
        status,
        body,
    );
}

/// The `taguru_schema` record's own wire shape (#384, ADR 0009 §13) —
/// `import.json` above never carries one, so this pins
/// `response.result.schemas[]`'s exact fields (`context`/`mode`/
/// `types`/`relations`, no outcome verb) separately.
#[test]
fn import_with_schema_reports_the_schema_outcome() {
    let server = Server::start("contract-import-schema");
    let stream = "{\"taguru_batch\": 1, \"context\": \"corpus-g\", \"source\": \"doc.md\", \
                  \"create\": {\"description\": \"wire-contract schema-carrying import\"}}\n\
                  {\"subject\": \"alpha\", \"label\": \"connects_to\", \"object\": \"beta\", \
                  \"weight\": 1.0}\n\
                  {\"taguru_schema\": 1, \"context\": \"corpus-g\", \"mode\": \"warn\", \
                  \"closed_labels\": false, \"types\": {\"Concept\": {\"is_a\": []}}, \
                  \"relations\": {\"connects_to\": {\"domain\": [\"Concept\"], \
                  \"range\": [\"Concept\"]}}}\n";
    let (status, body) = post_import(&server, stream, None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["result"]["schemas"][0]["context"], "corpus-g",
        "{body}"
    );
    http_fixture(
        "import_with_schema",
        "POST",
        "/import",
        Some(json!(stream)),
        status,
        body,
    );
}

// --- HTTP: evidence assembly (#216, #305, ADR 0006 §10) — the public
// shape #301's own issue names as the thing it must cover. ---

fn seed_evidence_corpus(server: &Server, name: &str) {
    server.ok(
        "PUT",
        &format!("/contexts/{name}"),
        Some(json!({"description": "evidence wire-contract corpus"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{name}/sources"),
        Some(json!({
            "passages": {
                "docs/kura.md": "青嶺酒造は雲居県霧沢町の蔵元である。杜氏は高瀬である。"
            },
            // A locator (ADR 0007 §7) so the citation/attribution wire
            // fixtures below carry a real, non-null value.
            "locators": {"docs/kura.md": [{"paragraph": 0, "locator": {"kind": "page", "value": "1"}}]}
        })),
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

/// Mixed graph/passage evidence, complete provenance, every lane's
/// `plan` — the baseline shape.
#[test]
fn evidence_mixed_lanes() {
    let server = Server::start("contract-evidence-mixed");
    seed_evidence_corpus(&server, "sake");

    let request = json!({"origins": ["青嶺酒造"]});
    let (status, body) = server.call("POST", "/contexts/sake/evidence", Some(request.clone()));
    assert_eq!(status, 200, "{body}");
    let items = body["result"]["items"].as_array().unwrap();
    assert!(
        items.iter().any(|item| item["kind"] == "association"),
        "{body}"
    );
    assert!(items.iter().any(|item| item["kind"] == "passage"), "{body}");
    evidence_fixture("evidence_mixed_lanes", request, status, body);
}

/// A budget too small for every candidate: some admitted, some
/// `omitted` under `budget_exceeded`, `omitted_total`/`omitted_by_reason`
/// both populated (ADR 0006 §8/§9).
#[test]
fn evidence_budget_constrained() {
    let server = Server::start("contract-evidence-budget");
    server.ok("PUT", "/contexts/budget-corpus", None);
    let associations: Vec<Value> = (0..5)
        .map(|index| {
            json!({"subject": format!("s{index}"), "label": "rel",
                   "object": format!("o{index}"), "weight": 1.0})
        })
        .collect();
    server.ok(
        "POST",
        "/contexts/budget-corpus/associations",
        Some(Value::Array(associations)),
    );

    let request = json!({
        "origins": ["s0", "s1", "s2", "s3", "s4"],
        "budget": {"max_items": 2},
    });
    let (status, body) = server.call(
        "POST",
        "/contexts/budget-corpus/evidence",
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"]["budget"]["limits"]["max_items"], json!(2));
    assert!(
        body["result"]["omitted_total"].as_u64().unwrap() > 0,
        "{body}"
    );
    assert!(
        body["result"]["omitted_by_reason"]["budget_exceeded"]
            .as_u64()
            .unwrap()
            > 0,
        "{body}"
    );
    evidence_fixture("evidence_budget_constrained", request, status, body);
}

/// Two near-identical passages: the lower-ranked one is `omitted`
/// under `duplicate_passage`, naming the survivor via `duplicate_of`
/// (ADR 0006 §9). `origins: []` doubles as coverage for the
/// `resolve`/`query`/`activate` lanes' "origins was empty" skip
/// reason, since `text_fallback_query` drives the passages lane
/// directly.
#[test]
fn evidence_duplicate_passage() {
    let server = Server::start("contract-evidence-dup");
    server.ok("PUT", "/contexts/dup-corpus", None);
    server.ok(
        "POST",
        "/contexts/dup-corpus/sources",
        Some(json!({"passages": {
            "a.md": "the quick brown fox jumps over the lazy dog",
            "b.md": "the quick brown fox jumps over the lazy dogs"
        }})),
    );

    let request = json!({
        "origins": [],
        "text_fallback_query": "quick brown fox",
    });
    let (status, body) = server.call(
        "POST",
        "/contexts/dup-corpus/evidence",
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["result"]["plan"]["lanes"]["resolve"]["ran"],
        json!(false),
        "{body}"
    );
    assert!(
        body["result"]["omitted"]
            .as_array()
            .unwrap()
            .iter()
            .any(|omission| omission["reason"] == "duplicate_passage"),
        "{body}"
    );
    evidence_fixture("evidence_duplicate_passage", request, status, body);
}

/// Two associations sharing `(subject, label)` but disagreeing on
/// `object`: a contradiction group, both items' `contradicts`
/// populated (ADR 0006 §9).
#[test]
fn evidence_contradiction_group() {
    let server = Server::start("contract-evidence-contradiction");
    server.ok("PUT", "/contexts/contradiction-corpus", None);
    server.ok(
        "POST",
        "/contexts/contradiction-corpus/sources",
        Some(json!({
            "passages": {
                "s1.md": "猫は哺乳類である。",
                "s2.md": "猫は爬虫類だと主張する文献もある。"
            },
            // A locator (ADR 0007 §7) on one side, so this fixture's
            // citations carry a real, non-null value alongside the
            // other side's null.
            "locators": {"s1.md": [{"paragraph": 0, "locator": {"kind": "page", "value": "1"}}]}
        })),
    );
    server.ok(
        "POST",
        "/contexts/contradiction-corpus/associations",
        Some(json!([
            {"subject": "猫", "label": "is_a", "object": "哺乳類", "weight": 1.0,
             "source": "s1.md", "paragraph": 0},
            {"subject": "猫", "label": "is_a", "object": "爬虫類", "weight": 1.0,
             "source": "s2.md", "paragraph": 0},
        ])),
    );

    let request = json!({"origins": ["猫"]});
    let (status, body) = server.call(
        "POST",
        "/contexts/contradiction-corpus/evidence",
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    let items = body["result"]["items"].as_array().unwrap();
    assert!(
        items.iter().any(|item| !item["contradicts"]
            .as_array()
            .unwrap_or(&Vec::new())
            .is_empty()),
        "{body}"
    );
    evidence_fixture("evidence_contradiction_group", request, status, body);
}

/// `include_communities: true` with no artifact yet — a degrade, not a
/// refusal (ADR 0006 §11) — plus a `rerank` hint that no provider acts
/// on, pinning `plan.reranker.reason`.
#[test]
fn evidence_communities_degrade_and_rerank_reason() {
    let server = Server::start("contract-evidence-communities");
    server.ok("PUT", "/contexts/comm-corpus", None);
    server.ok(
        "POST",
        "/contexts/comm-corpus/associations",
        Some(json!([{"subject": "alpha", "label": "rel", "object": "beta", "weight": 1.0}])),
    );

    let request = json!({
        "origins": ["alpha"],
        "include_communities": true,
        "rerank": {"model": "not-configured"},
    });
    let (status, body) = server.call(
        "POST",
        "/contexts/comm-corpus/evidence",
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["result"]["plan"]["lanes"]["communities"]["ran"],
        json!(false),
        "{body}"
    );
    assert!(
        body["result"]["plan"]["reranker"]["reason"].is_string(),
        "{body}"
    );
    evidence_fixture(
        "evidence_communities_degrade_and_rerank_reason",
        request,
        status,
        body,
    );
}

// --- HTTP: errors ---

#[test]
fn error_no_context() {
    let server = Server::start("contract-error-no-context");
    let request = json!({"origins": ["x"]});
    let (status, body) = server.call(
        "POST",
        "/contexts/does-not-exist/evidence",
        Some(request.clone()),
    );
    assert_eq!(status, 404, "{body}");
    evidence_fixture("error_no_context", request, status, body);
}

#[test]
fn error_over_limit() {
    let server = Server::start("contract-error-over-limit");
    server.ok("PUT", "/contexts/over-limit-corpus", None);
    let request = json!({"origins": vec!["x"; 1001]});
    let (status, body) = server.call(
        "POST",
        "/contexts/over-limit-corpus/evidence",
        Some(request.clone()),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("over_limit"), "{body}");
    evidence_fixture("error_over_limit", request, status, body);
}

#[test]
fn error_malformed_request() {
    let server = Server::start("contract-error-malformed");
    server.ok("PUT", "/contexts/malformed-corpus", None);
    let request = json!({"origins": ["x"], "budget": "not-an-object"});
    let (status, body) = server.call(
        "POST",
        "/contexts/malformed-corpus/evidence",
        Some(request.clone()),
    );
    assert_eq!(status, 422, "{body}");
    assert_eq!(body["code"], json!("malformed_request"), "{body}");
    evidence_fixture("error_malformed_request", request, status, body);
}

/// A read-scoped key restricted to `forbidden-corpus` alone (never
/// `forbidden-corpus::communities`) asking for `include_communities`.
#[test]
fn error_forbidden() {
    let server = Server::start_with_env(
        "contract-error-forbidden",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,reader:rtok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"reader": {"role": "read", "contexts": ["forbidden-corpus"]}}"#,
            ),
        ],
    );
    let (status, body) =
        server.call_with_token("PUT", "/contexts/forbidden-corpus", None, Some("atok"));
    assert_eq!(status, 200, "{body}");
    let (status, body) = server.call_with_token(
        "POST",
        "/contexts/forbidden-corpus/associations",
        Some(json!([{"subject": "a", "label": "rel", "object": "b", "weight": 1.0}])),
        Some("atok"),
    );
    assert_eq!(status, 200, "{body}");

    let request = json!({"origins": ["a"], "include_communities": true});
    let (status, body) = server.call_with_token(
        "POST",
        "/contexts/forbidden-corpus/evidence",
        Some(request.clone()),
        Some("rtok"),
    );
    assert_eq!(status, 403, "{body}");
    assert_eq!(body["code"], json!("forbidden"), "{body}");
    evidence_fixture("error_forbidden", request, status, body);
}

// --- MCP ---

#[test]
fn mcp_tools_list_assemble_evidence_schema() {
    let server = Server::start("contract-mcp-schema");
    let (status, body) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})),
    );
    assert_eq!(status, 200, "{body}");
    let tools = body["result"]["tools"].as_array().expect("tools array");
    let tool = tools
        .iter()
        .find(|tool| tool["name"] == "assemble_evidence")
        .expect("assemble_evidence tool present")
        .clone();
    assert_eq!(
        tool["inputSchema"]["required"],
        json!(["context", "origins"]),
        "{tool}"
    );
    mcp_fixture(
        "assemble_evidence_tool_schema",
        "tools/list",
        json!({}),
        status,
        json!({"tools": [tool]}),
    );
}

#[test]
fn mcp_assemble_evidence_call() {
    let server = Server::start("contract-mcp-call");
    server.ok("PUT", "/contexts/mcp-corpus", None);
    server.ok(
        "POST",
        "/contexts/mcp-corpus/associations",
        Some(json!([{"subject": "a", "label": "rel", "object": "b", "weight": 1.0}])),
    );

    let arguments = json!({"context": "mcp-corpus", "origins": ["a"]});
    let (status, body) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "assemble_evidence", "arguments": arguments}})),
    );
    assert_eq!(status, 200, "{body}");
    let result = body["result"].clone();
    assert!(result.get("isError").is_none(), "{result}");
    mcp_fixture(
        "assemble_evidence_call",
        "tools/call assemble_evidence",
        arguments,
        status,
        result,
    );
}

/// `origins` missing entirely — a tool-level error (`isError: true`),
/// never a JSON-RPC abort (ADR 0005 §2.4).
#[test]
fn mcp_assemble_evidence_missing_origins_is_a_tool_error() {
    let server = Server::start("contract-mcp-error");
    server.ok("PUT", "/contexts/mcp-error-corpus", None);

    let arguments = json!({"context": "mcp-error-corpus"});
    let (status, body) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "assemble_evidence", "arguments": arguments}})),
    );
    assert_eq!(status, 200, "{body}");
    let result = body["result"].clone();
    assert_eq!(result["isError"], json!(true), "{result}");
    mcp_fixture(
        "assemble_evidence_tool_error",
        "tools/call assemble_evidence",
        arguments,
        status,
        result,
    );
}

// --- shapes.json self-consistency ---

/// `path`'s dotted segments, `[]` meaning "every element of the array
/// at this point" — the one small path language `shapes.json`'s
/// `enums` keys use, matched against a fixture's own JSON tree.
///
/// An MCP tool result carries the whole HTTP body a second time as
/// JSON text inside `content[].text` (ADR 0005 §2.4's pass-through
/// convention), one level deeper than a plain object walk reaches —
/// when `key` isn't found directly, each `content[].text` is parsed
/// and the SAME unconsumed `path` (not `rest`) is retried against it,
/// since the parsed value takes `value`'s own place at this level.
fn collect_by_path(value: &Value, path: &[&str]) -> Vec<Value> {
    let Some((head, rest)) = path.split_first() else {
        return vec![value.clone()];
    };
    let (key, is_array) = match head.strip_suffix("[]") {
        Some(key) => (key, true),
        None => (*head, false),
    };
    let Some(next) = value.get(key) else {
        let Some(content) = value.get("content").and_then(Value::as_array) else {
            return Vec::new();
        };
        return content
            .iter()
            .filter_map(|item| item.get("text")?.as_str())
            .filter_map(|text| serde_json::from_str::<Value>(text).ok())
            .flat_map(|parsed| collect_by_path(&parsed, path))
            .collect();
    };
    if is_array {
        match next.as_array() {
            Some(items) => items
                .iter()
                .flat_map(|item| collect_by_path(item, rest))
                .collect(),
            None => Vec::new(),
        }
    } else {
        collect_by_path(next, rest)
    }
}

fn load_shapes() -> Value {
    let text = std::fs::read_to_string(wire_dir().join("shapes.json"))
        .expect("tests/fixtures/wire/shapes.json must exist");
    serde_json::from_str(&text).expect("shapes.json must be valid JSON")
}

/// Every fixture under `tests/fixtures/wire/{http,mcp}/` — both
/// transports, so the two self-consistency checks below cover the MCP
/// pass-through shape too, not just the HTTP one it inherits from.
fn wire_fixtures() -> Vec<(PathBuf, Value)> {
    let mut fixtures = Vec::new();
    for transport in ["http", "mcp"] {
        let dir = wire_dir().join(transport);
        for entry in std::fs::read_dir(&dir).unwrap_or_else(|_| panic!("{dir:?} must exist")) {
            let path = entry.expect("readable dir entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("fixture must be readable");
            let value: Value = serde_json::from_str(&text).expect("fixture must be valid JSON");
            fixtures.push((path, value));
        }
    }
    fixtures
}

/// Every string value a declared `enums` path reaches, across every
/// fixture (HTTP and MCP alike), must be one of the declared values —
/// so introducing a new `kind`/`lane`/`reason`/`ErrorCode` without
/// adding it to `shapes.json` fails locally, before `check_contract.py` ever
/// runs against a base ref.
#[test]
fn shapes_enums_cover_every_value_every_fixture_actually_emits() {
    let shapes = load_shapes();
    let enums = shapes["enums"].as_object().expect("enums object");
    for (path, fixture) in wire_fixtures() {
        for (path_expr, allowed) in enums {
            let allowed: Vec<&str> = allowed
                .as_array()
                .expect("enum value list")
                .iter()
                .map(|value| value.as_str().expect("enum value must be a string"))
                .collect();
            let segments: Vec<&str> = path_expr.split('.').collect();
            for value in collect_by_path(&fixture, &segments) {
                if let Some(text) = value.as_str() {
                    assert!(
                        allowed.contains(&text),
                        "{path_expr} in {path:?} carries {text:?}, which is not in \
                         shapes.json's enums — add it there if this is a new, intentional value"
                    );
                }
            }
        }
    }
}

/// Every field `shapes.json` marks required for a route is present in
/// every fixture whose `request` targets that route — keeps
/// `required_request_fields` honest against the fixtures it classifies.
#[test]
fn shapes_required_request_fields_are_present_in_every_matching_fixture() {
    let shapes = load_shapes();
    let required = shapes["required_request_fields"]
        .as_object()
        .expect("required_request_fields object");
    let mut routes_seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (path, fixture) in wire_fixtures() {
        let Some(route) = fixture["route"].as_str() else {
            continue;
        };
        routes_seen.insert(route.to_string());
        let Some(fields) = required.get(route).and_then(Value::as_array) else {
            continue;
        };
        let Some(request) = fixture.get("request").filter(|value| !value.is_null()) else {
            continue;
        };
        for field in fields {
            let field = field.as_str().expect("required field name");
            assert!(
                request.get(field).is_some(),
                "{path:?}: shapes.json marks '{field}' required for {route}, \
                 but this fixture's request omits it"
            );
        }
    }
    // The reverse direction: a route named in `required_request_fields`
    // with no fixture left to check it against is a stale entry (a
    // renamed or removed route) that the loop above would never catch.
    for route in required.keys() {
        assert!(
            routes_seen.contains(route.as_str()),
            "shapes.json's required_request_fields names '{route}', which no \
             fixture's route matches"
        );
    }
}
