//! Golden wire-contract fixtures (#301, ADR 0005 §9): the machine-
//! readable pin of the current `http_contract: 2` / `mcp_contract: 1`
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
            // A change-feed cursor (#422) carries a per-boot ring epoch —
            // volatile by design. Keyed to the `next` field specifically,
            // not to any string that happens to start with "cf1-": a
            // source id like "cf1-report.md" must keep recording its
            // real value. The MCP pass-through case (the whole body
            // re-encoded as JSON text) is covered by the String arm
            // below re-parsing and landing back in this arm.
            if let Some(Value::String(cursor)) = map.get_mut("next")
                && cursor.starts_with("cf1-")
            {
                *cursor = "cf1-0000000000000000-0".to_string();
            }
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

/// [`http_fixture`] pinned to `POST /contexts/{id}/evidence` — every
/// evidence-assembly and evidence-error fixture below targets this one
/// endpoint, so only `operation`/`request`/`status`/`response` vary.
fn evidence_fixture(operation: &str, request: Value, status: u16, response: Value) {
    http_fixture(
        operation,
        "POST",
        "/contexts/{id}/evidence",
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
    let server =
        Server::start_with_env("contract-probes", &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")]);

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
        "POST",
        "/contexts",
        Some(json!({"name": name, "description": "wire-contract corpus"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx(name)),
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
        &format!("/contexts/{}/sources", server.cx(name)),
        Some(json!({
            "passages": {"doc.md": "alpha connects to beta."},
            "locators": {"doc.md": [{"paragraph": 0, "locator": {"kind": "page", "value": "1"}}]}
        })),
    );
}

#[test]
fn recall_match_page_and_contexts_list() {
    let server =
        Server::start_with_env("contract-recall", &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")]);
    seed_basic_corpus(&server, "corpus-a");

    let request = json!({"cue": "alpha"});
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/recall", server.cx("corpus-a")),
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert!(
        !body["result"]["matches"].as_array().unwrap().is_empty(),
        "{body}"
    );
    http_fixture(
        "recall",
        "POST",
        "/contexts/{id}/recall",
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

    // The create's answer is the new row, id included — the value
    // every other path is built from (#964), so its shape is pinned
    // exactly like the listing's.
    let request = json!({"name": "corpus-created", "description": "wire-contract corpus"});
    let (status, body) = server.call("POST", "/contexts", Some(request.clone()));
    assert_eq!(status, 200, "{body}");
    assert!(body["result"]["id"].is_string(), "{body}");
    http_fixture(
        "contexts_create",
        "POST",
        "/contexts",
        Some(request),
        status,
        body,
    );
}

#[test]
fn explore_and_activate_pages() {
    let server = Server::start_with_env(
        "contract-explore",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    seed_basic_corpus(&server, "corpus-b");

    let request = json!({"origins": ["alpha"]});
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/explore", server.cx("corpus-b")),
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    http_fixture(
        "explore",
        "POST",
        "/contexts/{id}/explore",
        Some(request),
        status,
        body,
    );

    let request = json!({"origins": ["alpha"]});
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/activate", server.cx("corpus-b")),
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert!(
        !body["result"]["matches"].as_array().unwrap().is_empty(),
        "{body}"
    );
    http_fixture(
        "activate",
        "POST",
        "/contexts/{id}/activate",
        Some(request),
        status,
        body,
    );

    let request = json!({"origins": ["alpha"], "targets": ["beta"]});
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/paths", server.cx("corpus-b")),
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert!(
        !body["result"]["matches"].as_array().unwrap().is_empty(),
        "{body}"
    );
    http_fixture(
        "paths",
        "POST",
        "/contexts/{id}/paths",
        Some(request),
        status,
        body,
    );
}

/// The change feed's page shape (#422): tail for a cursor, write, read
/// the events past it. The cursor itself is volatile (a per-boot ring
/// epoch) and normalized by `normalize_volatile`; the query string, like
/// every GET fixture's, is not part of the recorded request.
#[test]
fn changes_feed_page() {
    let server = Server::start_with_env(
        "contract-changes",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    seed_basic_corpus(&server, "corpus-cf");

    let tail = server.ok(
        "GET",
        &format!("/contexts/{}/changes", server.cx("corpus-cf")),
        None,
    );
    let cursor = tail["next"].as_str().expect("a tail always has a cursor");

    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx("corpus-cf")),
        Some(json!([
            {"subject": "beta", "label": "connects_to", "object": "gamma", "weight": 1.0},
        ])),
    );

    let (status, body) = server.call(
        "GET",
        &format!(
            "/contexts/{}/changes?since={cursor}",
            server.cx("corpus-cf")
        ),
        None,
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["result"]["events"][0]["type"],
        json!("associations_added"),
        "{body}"
    );
    http_fixture(
        "changes",
        "GET",
        "/contexts/{id}/changes",
        None,
        status,
        body,
    );

    let (status, body) = server.call(
        "GET",
        &format!(
            "/contexts/{}/changes?since=cf1-0000000000000000-999",
            server.cx("corpus-cf")
        ),
        None,
    );
    assert_eq!(status, 410, "{body}");
    assert_eq!(body["code"], json!("stale_cursor"), "{body}");
    http_fixture(
        "changes_stale_cursor",
        "GET",
        "/contexts/{id}/changes",
        None,
        status,
        body,
    );
}

// --- HTTP: passage and community search — PassagePage is the 0.4.0
// incident shape (ADR 0005 §2.1), CommunityPage the richest of the
// thirteen pagination envelopes. ---

#[test]
fn sources_search_passage_page() {
    let server = Server::start_with_env(
        "contract-sources-search",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    server.ok("POST", "/contexts", Some(json!({"name": "corpus-c"})));
    server.ok(
        "POST",
        &format!("/contexts/{}/sources", server.cx("corpus-c")),
        Some(json!({"passages": {
            "doc.md": "青嶺酒造は雲居県霧沢町の蔵元である。"
        }})),
    );

    let request = json!({"query": "酒造"});
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/sources/search", server.cx("corpus-c")),
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
        "/contexts/{id}/sources/search",
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
    let server = Server::start_with_env(
        "contract-communities-search",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    server.ok("POST", "/contexts", Some(json!({"name": "corpus-d"})));
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx("corpus-d")),
        Some(json!([
            {"subject": "a1", "label": "近い", "object": "a2", "weight": 2.0},
        ])),
    );
    let revision =
        server.ok("GET", &format!("/contexts/{}", server.cx("corpus-d")), None)["revision"].clone();
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": "corpus-d::communities"})),
    );
    let manifest = json!({
        "type": "communities_manifest",
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
        &format!("/contexts/{}/sources", server.cx("corpus-d::communities")),
        Some(json!({"passages": {
            "community:L0-0": "この共同体のテーマは酒造りの歴史です。",
            "communities:manifest": manifest.to_string(),
        }})),
    );
    server.ok(
        "POST",
        &format!(
            "/contexts/{}/associations",
            server.cx("corpus-d::communities")
        ),
        Some(json!([
            {"subject": "community:L0-0", "label": "contains", "object": "a1", "weight": 6.0},
            {"subject": "community:L0-0", "label": "contains", "object": "a2", "weight": 4.0},
        ])),
    );

    let request = json!({"query": "酒造りの歴史"});
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/communities/search", server.cx("corpus-d")),
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
        "/contexts/{id}/communities/search",
        Some(request),
        status,
        body,
    );
}

// --- HTTP: passage storage and batch import (#346, ADR 0007 §7) — the
// two write paths a citation `locator` can ride in on. ---

#[test]
fn store_passages_response_shape() {
    let server = Server::start_with_env(
        "contract-store-passages",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    server.ok("POST", "/contexts", Some(json!({"name": "corpus-e"})));

    let request = json!({
        "passages": {"doc.md": "導入。\n\n本編。"},
        "sections": {"doc.md": [{"paragraph": 1, "section": "本編"}]},
        "locators": {"doc.md": [{"paragraph": 1, "locator": {"kind": "page", "value": "12"}}]},
    });
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/sources", server.cx("corpus-e")),
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"]["locators_stored"], json!(1), "{body}");
    http_fixture(
        "store_passages",
        "POST",
        "/contexts/{id}/sources",
        Some(request),
        status,
        body,
    );
}

/// S5 (#383): `warn` mode's `issues`/`schema_violations` on
/// `POST /contexts/{id}/associations` are new, wire-visible fields
/// on `ApiResponse` — additive (`HTTP_CONTRACT` unchanged, both are
/// `skip_serializing_if`-omitted on every response with nothing to
/// say), but still a shape an SDK consumer needs pinned so it stops
/// being invisible to the cross-language contract check the moment a
/// context actually turns `warn` on.
#[test]
fn add_associations_warn_mode_response_shape() {
    let server = Server::start_with_env(
        "contract-associations-warn",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    server.ok("POST", "/contexts", Some(json!({"name": "corpus-g"})));
    server.ok(
        "PUT",
        &format!("/contexts/{}/schema", server.cx("corpus-g")),
        Some(json!({
            "type": "schema",
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
        &format!("/contexts/{}/associations", server.cx("corpus-g")),
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"], json!(2), "{body}");
    assert_eq!(body["schema_violations"], json!(1), "{body}");
    assert!(body["issues"].is_array(), "{body}");
    http_fixture(
        "add_associations_warn",
        "POST",
        "/contexts/{id}/associations",
        Some(request),
        status,
        body,
    );
}

#[test]
fn import_reports_locator_bookkeeping() {
    let server =
        Server::start_with_env("contract-import", &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")]);
    let batch = "{\"type\": \"source\", \"context_id\": \"685ce7fe-340e-4e5e-a292-df7f89b60283\", \"id\": \"doc.md\", \
                 \"create\": {\"name\": \"corpus-f\", \"description\": \"wire-contract import corpus\"}}\n\
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

/// The `/import` refusal envelope's machine-readable resume fields
/// (issue #182): `integrity` + `durable_batches`, which an importer
/// keys on to skip the landed prefix instead of re-sending the whole
/// stream. Pinned via the deterministic mid-stream rejection — the
/// group-restore budget refusal carries the same field set (see
/// `restore_refusal_frames_a_spent_budget_as_a_resumable_timeout` in
/// src/api.rs), but its arm only fires when the deadline dies between
/// the batch loop and the group phase, a window no knob can hit
/// reliably enough for a recorded fixture.
#[test]
fn import_refusal_pins_the_durable_prefix_fields() {
    let server = Server::start_with_env(
        "contract-import-refusal",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    let stream = "{\"type\": \"source\", \"context_id\": \"2eed3710-ffe1-45c6-879f-f4efcbb46b36\", \"id\": \"doc-1\", \
                   \"create\": {\"name\": \"corpus-h\", \"description\": \"wire-contract refusal corpus\"}}\n\
                  {\"subject\": \"alpha\", \"label\": \"connects_to\", \"object\": \"beta\", \
                  \"weight\": 1.0}\n\
                  {\"type\": \"source\", \"context_id\": \"2eed3710-ffe1-45c6-879f-f4efcbb46b36\", \"id\": \"doc-2\"}\n\
                  {\"alias\": \"Alpha\", \"canonical\": \"存在しない\", \"kind\": \"concept\"}\n";
    let (status, body) = post_import(&server, stream, None);
    assert_eq!(status, 409, "{body}");
    assert_eq!(body["integrity"], json!("durable_prefix"), "{body}");
    assert_eq!(body["durable_batches"], json!(1), "{body}");
    http_fixture(
        "import_refusal_durable_prefix",
        "POST",
        "/import",
        Some(json!(stream)),
        status,
        body,
    );
}

/// The `schema` record's own wire shape (#384, ADR 0009 §13) —
/// `import.json` above never carries one, so this pins
/// `response.result.schemas[]`'s exact fields (`context`/`mode`/
/// `types`/`relations`, no outcome verb) separately.
#[test]
fn import_with_schema_reports_the_schema_outcome() {
    let server = Server::start_with_env(
        "contract-import-schema",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    let stream = "{\"type\": \"source\", \"context_id\": \"aae8fff9-33c6-451f-bbe7-41ec6bb9d2bb\", \"id\": \"doc.md\", \
                  \"create\": {\"name\": \"corpus-g\", \"description\": \"wire-contract schema-carrying import\"}}\n\
                  {\"subject\": \"alpha\", \"label\": \"connects_to\", \"object\": \"beta\", \
                  \"weight\": 1.0}\n\
                  {\"type\": \"schema\", \"context_id\": \"aae8fff9-33c6-451f-bbe7-41ec6bb9d2bb\", \"mode\": \"warn\", \
                  \"closed_labels\": false, \"types\": {\"Concept\": {\"is_a\": []}}, \
                  \"relations\": {\"connects_to\": {\"domain\": [\"Concept\"], \
                  \"range\": [\"Concept\"]}}}\n";
    let (status, body) = post_import(&server, stream, None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["result"]["schemas"][0]["context_id"], "aae8fff9-33c6-451f-bbe7-41ec6bb9d2bb",
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
        "POST",
        "/contexts",
        Some(json!({"name": name, "description": "evidence wire-contract corpus"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/sources", server.cx(name)),
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
        &format!("/contexts/{}/associations", server.cx(name)),
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
    let server = Server::start_with_env(
        "contract-evidence-mixed",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    seed_evidence_corpus(&server, "sake");

    let request = json!({"origins": ["青嶺酒造"]});
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/evidence", server.cx("sake")),
        Some(request.clone()),
    );
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
    let server = Server::start_with_env(
        "contract-evidence-budget",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    server.ok("POST", "/contexts", Some(json!({"name": "budget-corpus"})));
    let associations: Vec<Value> = (0..5)
        .map(|index| {
            json!({"subject": format!("s{index}"), "label": "rel",
                   "object": format!("o{index}"), "weight": 1.0})
        })
        .collect();
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx("budget-corpus")),
        Some(Value::Array(associations)),
    );

    let request = json!({
        "origins": ["s0", "s1", "s2", "s3", "s4"],
        "budget": {"max_items": 2},
    });
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/evidence", server.cx("budget-corpus")),
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
    let server = Server::start_with_env(
        "contract-evidence-dup",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    server.ok("POST", "/contexts", Some(json!({"name": "dup-corpus"})));
    server.ok(
        "POST",
        &format!("/contexts/{}/sources", server.cx("dup-corpus")),
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
        &format!("/contexts/{}/evidence", server.cx("dup-corpus")),
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
    let server = Server::start_with_env(
        "contract-evidence-contradiction",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": "contradiction-corpus"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/sources", server.cx("contradiction-corpus")),
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
        &format!(
            "/contexts/{}/associations",
            server.cx("contradiction-corpus")
        ),
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
        &format!("/contexts/{}/evidence", server.cx("contradiction-corpus")),
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
    let server = Server::start_with_env(
        "contract-evidence-communities",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    server.ok("POST", "/contexts", Some(json!({"name": "comm-corpus"})));
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx("comm-corpus")),
        Some(json!([{"subject": "alpha", "label": "rel", "object": "beta", "weight": 1.0}])),
    );

    let request = json!({
        "origins": ["alpha"],
        "include_communities": true,
        "rerank": {"model": "not-configured"},
    });
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/evidence", server.cx("comm-corpus")),
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
    let server = Server::start_with_env(
        "contract-error-no-context",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    let request = json!({"origins": ["x"]});
    // A well-formed id nothing answers to — an unknown context is a
    // 404 (an id-SHAPED mistake would be the 400 the extractor pins
    // elsewhere).
    let (status, body) = server.call(
        "POST",
        "/contexts/00000000-0000-4000-8000-00000000dead/evidence",
        Some(request.clone()),
    );
    assert_eq!(status, 404, "{body}");
    evidence_fixture("error_no_context", request, status, body);
}

#[test]
fn error_over_limit() {
    let server = Server::start_with_env(
        "contract-error-over-limit",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": "over-limit-corpus"})),
    );
    let request = json!({"origins": vec!["x"; 1001]});
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/evidence", server.cx("over-limit-corpus")),
        Some(request.clone()),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("over_limit"), "{body}");
    evidence_fixture("error_over_limit", request, status, body);
}

#[test]
fn error_malformed_request() {
    let server = Server::start_with_env(
        "contract-error-malformed",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": "malformed-corpus"})),
    );
    let request = json!({"origins": ["x"], "budget": "not-an-object"});
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/evidence", server.cx("malformed-corpus")),
        Some(request.clone()),
    );
    assert_eq!(status, 422, "{body}");
    assert_eq!(body["code"], json!("malformed_request"), "{body}");
    evidence_fixture("error_malformed_request", request, status, body);
}

/// A key granted only Read, restricted to `forbidden-corpus` alone
/// (never `forbidden-corpus::communities`) asking for
/// `include_communities`.
#[test]
fn error_forbidden() {
    let server = Server::start_with_env(
        "contract-error-forbidden",
        &[
            ("TAGURU_TEST_DETERMINISTIC_IDS", "1"),
            ("TAGURU_API_TOKENS", "boss:atok,reader:rtok"),
            (
                "TAGURU_KEY_GRANTS",
                r#"{"reader": {"role": "read", "contexts": ["forbidden-corpus"]}}"#,
            ),
        ],
    );
    let (status, body) = server.call_with_token(
        "POST",
        "/contexts",
        Some(json!({"name": "forbidden-corpus"})),
        Some("atok"),
    );
    assert_eq!(status, 200, "{body}");
    let (status, body) = server.call_with_token(
        "POST",
        &format!("/contexts/{}/associations", server.cx("forbidden-corpus")),
        Some(json!([{"subject": "a", "label": "rel", "object": "b", "weight": 1.0}])),
        Some("atok"),
    );
    assert_eq!(status, 200, "{body}");

    let request = json!({"origins": ["a"], "include_communities": true});
    let (status, body) = server.call_with_token(
        "POST",
        &format!("/contexts/{}/evidence", server.cx("forbidden-corpus")),
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
    let server = Server::start_with_env(
        "contract-mcp-schema",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
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
    let server = Server::start_with_env(
        "contract-mcp-call",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    server.ok("POST", "/contexts", Some(json!({"name": "mcp-corpus"})));
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx("mcp-corpus")),
        Some(json!([{"subject": "a", "label": "rel", "object": "b", "weight": 1.0}])),
    );

    let arguments = json!({"context": server.cx("mcp-corpus"), "origins": ["a"]});
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
    let server = Server::start_with_env(
        "contract-mcp-error",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": "mcp-error-corpus"})),
    );

    let arguments = json!({"context": server.cx("mcp-error-corpus")});
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

// --- HTTP: graph-path promotion (#466 S2, ADR 0018) ---

/// A fully-dated promotion corpus, one source promoted and one left
/// behind — so the pinned response shows the alias accounting and an
/// all-sections audit without a wall-clock value anywhere (the audit
/// sections come back empty on this tiny corpus; their SHAPE is what
/// the fixture pins).
#[test]
fn promote_applies_and_a_dry_run_previews() {
    let server = Server::start_with_env(
        "contract-promote",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": "scratch-w", "description": "wire-contract session notes"})),
    );
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": "corpus-p", "description": "wire-contract permanent corpus"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx("scratch-w")),
        Some(json!([
            {"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0,
             "source": "session:w:a", "paragraph": 0},
            {"subject": "蔵", "label": "銘柄", "object": "青嶺", "weight": 1.0,
             "source": "session:w:stay"},
        ])),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/sources", server.cx("scratch-w")),
        Some(json!({
            "passages": {"session:w:a": "蔵の杜氏は高瀬。"},
            "dates": {"session:w:a": 1000},
            "tags": {"session:w:a": ["酒"]}
        })),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/aliases", server.cx("scratch-w")),
        Some(json!({"concepts": {"たかせ": "高瀬", "あおみね": "青嶺"}})),
    );

    let request = json!({"into": server.cx("corpus-p"), "sources": ["session:w:a"]});
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/promote?dry_run=true", server.cx("scratch-w")),
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert!(
        body["result"].get("audit").is_none() && body["result"].get("audit_skipped").is_none(),
        "a dry run omits the audit half entirely: {body}"
    );
    http_fixture(
        "promote_dry_run",
        "POST",
        "/contexts/{id}/promote?dry_run=true",
        Some(request.clone()),
        status,
        body,
    );

    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/promote", server.cx("scratch-w")),
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"]["aliases_dropped"], json!(1), "{body}");
    assert_eq!(
        body["result"]["audit"]["detector"],
        json!("consolidation/1"),
        "{body}"
    );
    http_fixture(
        "promote",
        "POST",
        "/contexts/{id}/promote",
        Some(request),
        status,
        body,
    );
}

// --- HTTP + MCP: consolidation audit (ADR 0012) ---

/// A fully-dated seed — every effective time explicit, so the pinned
/// response carries no wall-clock value anywhere.
fn seed_consolidation_corpus(server: &Server, name: &str) {
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": name, "description": "consolidation contract corpus"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx(name)),
        Some(json!([
            {"subject": "青嶺酒造", "label": "銘柄", "object": "青嶺", "weight": 1.0, "source": "doc-a"},
            {"subject": "青嶺酒蔵", "label": "銘柄", "object": "青嶺", "weight": 1.0, "source": "doc-b"},
            {"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "doc-a"},
            {"subject": "蔵", "label": "杜氏", "object": "青山", "weight": 1.0, "source": "doc-b"},
            {"subject": "蔵", "label": "行う", "object": "大量生産", "weight": 1.0, "source": "doc-a"},
            {"subject": "蔵", "label": "行う", "object": "大量生産", "weight": -2.0, "source": "doc-b"},
            {"subject": "蔵", "label": "銘柄", "object": "初霜", "weight": 1.0, "source": "doc-a"},
            {"subject": "幽霊蔵", "label": "銘柄", "object": "幻", "weight": 1.0, "source": "doc-undated"},
        ])),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/sources", server.cx(name)),
        Some(json!({
            "passages": {"doc-a": "旧情報。", "doc-b": "新情報。"},
            "dates": {"doc-a": 1000, "doc-b": 2000}
        })),
    );
}

#[test]
fn consolidation_audit_shape() {
    let server = Server::start_with_env(
        "contract-consolidation",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    seed_consolidation_corpus(&server, "fix");
    let request = json!({"checks": ["merge", "contradiction", "staleness"]});
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/consolidation/audit", server.cx("fix")),
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["result"]["detector"],
        json!("consolidation/1"),
        "{body}"
    );
    http_fixture(
        "consolidation_audit",
        "POST",
        "/contexts/{id}/consolidation/audit",
        Some(request),
        status,
        body,
    );
}

#[test]
fn mcp_tools_list_audit_consolidation_schema() {
    let server = Server::start_with_env(
        "contract-mcp-consolidation",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    let (status, body) = server.call(
        "POST",
        "/mcp",
        Some(json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})),
    );
    assert_eq!(status, 200, "{body}");
    let tools = body["result"]["tools"].as_array().expect("tools array");
    let tool = tools
        .iter()
        .find(|tool| tool["name"] == "audit_consolidation")
        .expect("audit_consolidation tool present")
        .clone();
    assert_eq!(
        tool["inputSchema"]["required"],
        json!(["context", "checks"]),
        "{tool}"
    );
    mcp_fixture(
        "audit_consolidation_tool_schema",
        "tools/list",
        json!({}),
        status,
        json!({"tools": [tool]}),
    );
}

// --- HTTP: aliases, coverage, vocabulary/drift audits, and a plain
// associations write (issue #625 finding 2) — these modules previously
// had no wire-contract pin at all, so a breaking response-shape change
// in any of them would have shipped undetected. ---

/// The ordinary (schema-free) associations write — distinct from
/// `add_associations_warn_mode_response_shape` above, which pins the
/// `issues`/`schema_violations` fields a `warn`-mode schema adds.
/// Neither field appears here (both are `skip_serializing_if`-omitted
/// with nothing to say), so the two fixtures together pin both shapes
/// the same response envelope can take.
#[test]
fn associations_store_response_shape() {
    let server = Server::start_with_env(
        "contract-associations-store",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    server.ok("POST", "/contexts", Some(json!({"name": "corpus-l"})));

    let request = json!([
        {"subject": "alpha", "label": "connects_to", "object": "beta", "weight": 1.0, "source": "doc.md"},
    ]);
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/associations", server.cx("corpus-l")),
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"], json!(1), "{body}");
    http_fixture(
        "associations_store",
        "POST",
        "/contexts/{id}/associations",
        Some(request),
        status,
        body,
    );
}

fn seed_aliases_corpus(server: &Server, name: &str) {
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": name, "description": "wire-contract aliases corpus"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx(name)),
        Some(json!([
            {"subject": "PostgreSQL 16", "label": "採用", "object": "DB", "weight": 1.0,
             "source": "doc.md"},
        ])),
    );
}

#[test]
fn aliases_register_list_and_remove() {
    let server = Server::start_with_env(
        "contract-aliases",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    seed_aliases_corpus(&server, "corpus-h");

    let request = json!({"concepts": {"Postgres": "PostgreSQL 16"}});
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/aliases", server.cx("corpus-h")),
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"], json!(1), "{body}");
    http_fixture(
        "aliases_register",
        "POST",
        "/contexts/{id}/aliases",
        Some(request),
        status,
        body,
    );

    let (status, body) = server.call(
        "GET",
        &format!("/contexts/{}/aliases", server.cx("corpus-h")),
        None,
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"]["total"], json!(1), "{body}");
    http_fixture(
        "aliases_list",
        "GET",
        "/contexts/{id}/aliases",
        None,
        status,
        body,
    );

    let request = json!({"concepts": ["Postgres"]});
    let (status, body) = server.call(
        "DELETE",
        &format!("/contexts/{}/aliases", server.cx("corpus-h")),
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"], json!(1), "{body}");
    http_fixture(
        "aliases_remove",
        "DELETE",
        "/contexts/{id}/aliases",
        Some(request),
        status,
        body,
    );
}

#[test]
fn coverage_labels_embeddings_and_unreachable() {
    let server = Server::start_with_env(
        "contract-coverage",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": "corpus-i", "description": "wire-contract coverage corpus"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx("corpus-i")),
        Some(json!([
            {"subject": "alpha", "label": "connects_to", "object": "beta", "weight": 1.0,
             "source": "doc.md"},
        ])),
    );

    let (status, body) = server.call(
        "GET",
        &format!("/contexts/{}/labels", server.cx("corpus-i")),
        None,
    );
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"]["labels"], json!(["connects_to"]), "{body}");
    http_fixture("labels", "GET", "/contexts/{id}/labels", None, status, body);

    // No embedding provider configured in this harness — pins the
    // `provider_model: null` shape, the common no-embeddings deployment.
    let (status, body) = server.call(
        "GET",
        &format!("/contexts/{}/embeddings", server.cx("corpus-i")),
        None,
    );
    assert_eq!(status, 200, "{body}");
    http_fixture(
        "embeddings_status",
        "GET",
        "/contexts/{id}/embeddings",
        None,
        status,
        body,
    );

    // "gamma" names no concept in this corpus, so nothing is reachable
    // from it — every edge in the graph counts as unreachable.
    let request = json!({"origins": ["gamma"]});
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/unreachable_from", server.cx("corpus-i")),
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert!(
        !body["result"]["matches"].as_array().unwrap().is_empty(),
        "{body}"
    );
    http_fixture(
        "unreachable_from",
        "POST",
        "/contexts/{id}/unreachable_from",
        Some(request),
        status,
        body,
    );
}

/// Two spellings of the same brewery (青嶺酒造/青嶺酒蔵) for the lexical
/// twin detector, and an edge with no `source` (so no attribution
/// explains its weight) for the drift audit's `unsourced` section.
fn seed_vocabulary_corpus(server: &Server, name: &str) {
    server.ok(
        "POST",
        "/contexts",
        Some(json!({"name": name, "description": "wire-contract vocabulary corpus"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{}/associations", server.cx(name)),
        Some(json!([
            {"subject": "青嶺酒造", "label": "銘柄", "object": "青嶺", "weight": 1.0, "source": "doc-a"},
            {"subject": "青嶺酒蔵", "label": "銘柄", "object": "青嶺", "weight": 1.0, "source": "doc-b"},
            {"subject": "蔵", "label": "行う", "object": "醸造", "weight": 3.0},
        ])),
    );
}

#[test]
fn vocabulary_audit_shape() {
    let server = Server::start_with_env(
        "contract-vocabulary",
        &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")],
    );
    seed_vocabulary_corpus(&server, "corpus-j");

    let request = json!({});
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/vocabulary/audit", server.cx("corpus-j")),
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert!(
        !body["result"]["lexical_concepts"]
            .as_array()
            .unwrap()
            .is_empty(),
        "{body}"
    );
    http_fixture(
        "vocabulary_audit",
        "POST",
        "/contexts/{id}/vocabulary/audit",
        Some(request),
        status,
        body,
    );
}

#[test]
fn drift_audit_shape() {
    let server =
        Server::start_with_env("contract-drift", &[("TAGURU_TEST_DETERMINISTIC_IDS", "1")]);
    seed_vocabulary_corpus(&server, "corpus-k");

    let request = json!({"include_twins": true});
    let (status, body) = server.call(
        "POST",
        &format!("/contexts/{}/drift/audit", server.cx("corpus-k")),
        Some(request.clone()),
    );
    assert_eq!(status, 200, "{body}");
    assert!(
        !body["result"]["unsourced"].as_array().unwrap().is_empty(),
        "{body}"
    );
    assert!(
        !body["result"]["twins"]["lexical_concepts"]
            .as_array()
            .unwrap()
            .is_empty(),
        "{body}"
    );
    http_fixture(
        "drift_audit",
        "POST",
        "/contexts/{id}/drift/audit",
        Some(request),
        status,
        body,
    );
}
