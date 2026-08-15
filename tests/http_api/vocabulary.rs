//! `POST /contexts/{name}/vocabulary/audit`'s semantic (gloss-cosine)
//! half and `/drift/audit`'s `cosine_floor` propagation (issue #626):
//! `explore_audit.rs` and `schema_type_label.rs` exercise the lexical
//! twin detector thoroughly, but nothing in the `tests/` tree ever
//! configures an embedding provider and drives the semantic detector
//! itself — `semantic_note`'s three message families, `cosine_floor`
//! actually filtering, the sweep-cap skip, and the schema type-name
//! exclusion applied to `semantic_concepts`/`semantic_labels` were all
//! unexercised outside the static wire fixtures.

use std::io::{Read, Write};
use std::net::TcpListener;

use serde_json::{Value, json};

use crate::support::*;

/// An embeddings stub keyed by CONTENT, not by the concept name that
/// carries it — the point of a semantic (as opposed to lexical) twin
/// is that two differently-spelled concepts land close in vector
/// space because their glosses share a fact, not their spelling.
/// Every gloss `Context::concept_gloss`/`label_gloss` builds
/// (`src/context/gloss.rs::gloss_text`) renders each fact as
/// `"{subject}の{label}は{object}。"` — this stub axis-matches on a
/// RELATION LABEL or OBJECT name appearing anywhere in that text, so
/// two concepts sharing a label or object (regardless of their own
/// spelling) land on the same axis. A shared low-weight last
/// dimension (as in `calibrate.rs`'s stub) keeps distinct axes from
/// being perfectly orthogonal — cosine 0.2 apart, not 0.0 — so a
/// floor between the two bands has room to sit.
fn spawn_semantic_embeddings() -> String {
    const AXES: [&str; 3] = ["使用米", "水系", "schema:type"];
    const WIDTH: usize = AXES.len() + 1;

    fn vector(text: &str) -> Vec<f32> {
        let mut vector = vec![0.0f32; WIDTH];
        let axis = AXES.iter().position(|keyword| text.contains(keyword));
        if let Some(axis) = axis {
            vector[axis] = 1.0;
            vector[WIDTH - 1] = 0.5;
        } else {
            // Unrecognized text (including bare relation-label
            // glosses with no matching keyword) still needs a unit
            // vector — put it on its own axis-free direction so it
            // never accidentally collides with either band.
            vector[WIDTH - 1] = 1.0;
        }
        vector
    }

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            std::thread::spawn(move || {
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
                let headers = String::from_utf8_lossy(&buffer[..body_start]).to_string();
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
                    serde_json::from_slice(&buffer[body_start..body_start + length]).unwrap();
                let data: Vec<Value> = request["input"]
                    .as_array()
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .map(|input| json!({ "embedding": vector(input.as_str().unwrap_or_default()) }))
                    .collect();
                let body = json!({ "data": data }).to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
            });
        }
    });
    format!("http://{addr}/v1/embeddings")
}

/// A server with the semantic stub wired in, no corpus yet.
fn semantic_server(tag: &str) -> Server {
    Server::start_with_env(
        tag,
        &[
            ("TAGURU_EMBED_URL", spawn_semantic_embeddings().as_str()),
            ("TAGURU_EMBED_MODEL", "gloss-mock"),
        ],
    )
}

/// Two concept pairs the stub resolves to two distinct bands: 鶴の井
/// and 白鷺の里 share the "使用米" fact (cosine ~1.0, the "upper"
/// band any floor up to 1.0 should keep), 鶴の井 and 遠山川 share only
/// the low-weight component (cosine ~0.2, the "lower" band a
/// default-or-higher floor should drop). The one "使用米"/"水系" label
/// pair rides the same two bands.
fn seed_semantic_corpus(server: &Server, name: &str) {
    server.ok(
        "PUT",
        &format!("/contexts/{name}"),
        Some(json!({"description": "d"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{name}/associations"),
        Some(json!([
            {"subject": "鶴の井", "label": "使用米", "object": "五百万石",
             "weight": 1.0, "source": "a.md"},
            {"subject": "白鷺の里", "label": "使用米", "object": "五百万石",
             "weight": 1.0, "source": "a.md"},
            {"subject": "遠山川", "label": "水系", "object": "支流",
             "weight": 1.0, "source": "a.md"},
        ])),
    );
    server.ok(
        "POST",
        &format!("/contexts/{name}/embeddings/refresh"),
        None,
    );
}

fn has_pair(pairs: &Value, a: &str, b: &str) -> bool {
    pairs.as_array().unwrap().iter().any(|pair| {
        let names = [pair["a"].as_str().unwrap(), pair["b"].as_str().unwrap()];
        names.contains(&a) && names.contains(&b)
    })
}

/// Before any refresh, `semantic_note` names the reason precisely —
/// not a generic empty result — and the semantic sections are empty
/// while the lexical ones (untouched by this gap) still run.
#[test]
fn semantic_note_explains_vectors_never_generated() {
    let server = Server::start("vocab-no-vectors");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "住所", "object": "京都",
             "weight": 1.0, "source": "a.md"},
        ])),
    );

    let audit = server.ok("POST", "/contexts/sake/vocabulary/audit", None);
    assert_eq!(
        audit["semantic_note"],
        json!("ベクトル未生成のため意味的検出はスキップ (POST embeddings/refresh を実行)"),
        "{audit}"
    );
    assert_eq!(audit["semantic_concepts"], json!([]), "{audit}");
    assert_eq!(audit["semantic_labels"], json!([]), "{audit}");
}

/// `cosine_floor` actually filters: a low floor keeps both the tight
/// and the loose band, a floor above the loose band's cosine drops it
/// — the parameter is not merely accepted and ignored. Out-of-range
/// input (`-1.0`/`2.0`) is clamped into `[0.0, 1.0]`
/// (`embeddings.rs`'s `cosine_floor.clamp(0.0, 1.0)`), not refused.
#[test]
fn cosine_floor_filters_the_semantic_sweep_and_clamps_out_of_range_input() {
    let server = semantic_server("vocab-cosine-floor");
    seed_semantic_corpus(&server, "sake");

    let low = server.ok(
        "POST",
        "/contexts/sake/vocabulary/audit",
        Some(json!({"cosine_floor": 0.1})),
    );
    assert!(low["semantic_note"].is_null(), "{low}");
    assert!(
        has_pair(&low["semantic_concepts"], "鶴の井", "白鷺の里"),
        "{low}"
    );
    assert!(
        has_pair(&low["semantic_concepts"], "鶴の井", "遠山川"),
        "{low}"
    );

    // Default floor (0.6): the tight band survives, the loose one
    // (~0.2) does not.
    let default = server.ok("POST", "/contexts/sake/vocabulary/audit", None);
    assert!(
        has_pair(&default["semantic_concepts"], "鶴の井", "白鷺の里"),
        "{default}"
    );
    assert!(
        !has_pair(&default["semantic_concepts"], "鶴の井", "遠山川"),
        "{default}"
    );

    // An out-of-range floor is accepted (200), not refused as invalid
    // input — `cosine_floor.clamp(0.0, 1.0)` in `embeddings.rs`.
    // Above 1.0 clamps to exactly 1.0: even the tight band's cosine
    // (0.999999..., never exactly 1.0 in floating point) no longer
    // clears it, so the sweep comes back empty rather than erroring.
    let clamped_high = server.ok(
        "POST",
        "/contexts/sake/vocabulary/audit",
        Some(json!({"cosine_floor": 2.0})),
    );
    assert_eq!(
        clamped_high["semantic_concepts"],
        json!([]),
        "{clamped_high}"
    );

    // Below 0.0 clamps to exactly 0.0: both bands clear it, same as
    // the explicit 0.1 floor above.
    let clamped_low = server.ok(
        "POST",
        "/contexts/sake/vocabulary/audit",
        Some(json!({"cosine_floor": -1.0})),
    );
    assert!(
        has_pair(&clamped_low["semantic_concepts"], "鶴の井", "白鷺の里"),
        "{clamped_low}"
    );
    assert!(
        has_pair(&clamped_low["semantic_concepts"], "鶴の井", "遠山川"),
        "{clamped_low}"
    );
}

/// `/drift/audit`'s `include_twins`/`cosine_floor` combination reaches
/// the same semantic sweep `/vocabulary/audit` does — `DriftAuditRequest`
/// has carried `cosine_floor` since it was added, but nothing drove it.
#[test]
fn drift_audit_include_twins_propagates_cosine_floor() {
    let server = semantic_server("vocab-drift-cosine-floor");
    seed_semantic_corpus(&server, "sake");

    let strict = server.ok(
        "POST",
        "/contexts/sake/drift/audit",
        Some(json!({"include_twins": true, "cosine_floor": 0.6})),
    );
    let twins = &strict["twins"];
    assert!(!twins.is_null(), "{strict}");
    assert!(
        has_pair(&twins["semantic_concepts"], "鶴の井", "白鷺の里"),
        "{strict}"
    );
    assert!(
        !has_pair(&twins["semantic_concepts"], "鶴の井", "遠山川"),
        "{strict}"
    );

    let loose = server.ok(
        "POST",
        "/contexts/sake/drift/audit",
        Some(json!({"include_twins": true, "cosine_floor": 0.1})),
    );
    assert!(
        has_pair(&loose["twins"]["semantic_concepts"], "鶴の井", "遠山川"),
        "{loose}"
    );
}

/// ADR 0009 §6.3 exclusion 3, semantic half: `type_name_concepts_are_
/// excluded_from_the_vocabulary_twin_audit_once_a_schema_exists`
/// (`schema_type_label.rs`) pins this for `lexical_concepts` only.
/// The exclusion in `vocabulary.rs` applies to `semantic_concepts`/
/// `semantic_labels` too (`concepts.retain(...)` after the semantic
/// sweep) — unexercised until now.
#[test]
fn type_name_concepts_are_excluded_from_the_semantic_twin_audit_too() {
    let server = semantic_server("vocab-semantic-type-exclusion");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    // Two DIFFERENT concepts, each the OBJECT of its own `schema:type`
    // edge (`type_name_concepts` only ever inserts the object side,
    // `src/api/vocabulary.rs`'s `names.insert(assoc.object)`) — the
    // type names themselves, not the instances asserting them. Both
    // objects' own glosses carry the literal label "schema:type",
    // landing them on the same stub axis. Plus the ordinary pair from
    // `seed_semantic_corpus`'s shape so the exclusion's scope (types
    // only, not everything) is provable.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "純米酒01", "label": "schema:type", "object": "純米原料米型",
             "weight": 1.0, "source": "a.md"},
            {"subject": "特別本醸造02", "label": "schema:type", "object": "特別本醸造原料米型",
             "weight": 1.0, "source": "a.md"},
            {"subject": "鶴の井", "label": "使用米", "object": "五百万石",
             "weight": 1.0, "source": "a.md"},
            {"subject": "白鷺の里", "label": "使用米", "object": "五百万石",
             "weight": 1.0, "source": "a.md"},
        ])),
    );
    server.ok("POST", "/contexts/sake/embeddings/refresh", None);

    let before = server.ok(
        "POST",
        "/contexts/sake/vocabulary/audit",
        Some(json!({"cosine_floor": 0.1})),
    );
    assert!(
        has_pair(
            &before["semantic_concepts"],
            "純米原料米型",
            "特別本醸造原料米型"
        ),
        "guard 1 (no schema yet): a type-name concept is an ordinary semantic \
         twin candidate: {before}"
    );

    server.ok(
        "PUT",
        "/contexts/sake/schema",
        Some(json!({
            "schema": 1, "mode": "off", "closed_labels": false,
            "types": {}, "relations": {}
        })),
    );

    let after = server.ok(
        "POST",
        "/contexts/sake/vocabulary/audit",
        Some(json!({"cosine_floor": 0.1})),
    );
    assert!(
        !has_pair(
            &after["semantic_concepts"],
            "純米原料米型",
            "特別本醸造原料米型"
        ),
        "once a schema exists, a type name must never be proposed as a semantic \
         twin candidate either: {after}"
    );
    // Scoped to type names: the ordinary pair the schema never touched
    // must still be audited, or the assertion above would also pass on
    // a bug that emptied the whole sweep.
    assert!(
        has_pair(&after["semantic_concepts"], "鶴の井", "白鷺の里"),
        "an ordinary concept pair must still be audited: {after}"
    );
}

/// A vocabulary whose concept table exceeds the pairwise sweep's
/// `SWEEP_CAP` (2000, `embeddings.rs`) skips the semantic half loudly
/// instead of paying an unbounded O(n²) cost — `semantic_note` names
/// the reason, distinct from the "never generated" and "deadline"
/// messages.
#[test]
fn semantic_note_explains_the_sweep_cap_skip() {
    let server = semantic_server("vocab-sweep-cap");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    // A chain of 2001 concepts (v0000 next v0001, ..., v1999 next
    // v2000) — one association batch, one embeddings refresh, well
    // under MAX_ASSOCIATIONS_PER_REQUEST (10_000).
    let chain: Vec<Value> = (0..2000)
        .map(|i| {
            json!({
                "subject": format!("v{i:04}"),
                "label": "次",
                "object": format!("v{:04}", i + 1),
                "weight": 1.0,
                "source": "a.md",
            })
        })
        .collect();
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(Value::Array(chain)),
    );
    server.ok("POST", "/contexts/sake/embeddings/refresh", None);

    let audit = server.ok(
        "POST",
        "/contexts/sake/vocabulary/audit",
        Some(json!({"cosine_floor": 0.0})),
    );
    assert_eq!(
        audit["semantic_note"],
        json!("語彙が 2000 名を超えるためこの名前空間の意味的検出はスキップ"),
        "{audit}"
    );
    assert_eq!(audit["semantic_concepts"], json!([]), "{audit}");
    // The label table (one label, "次") never approaches the cap, so
    // its own sweep runs — proving the skip is concept-table-scoped,
    // not a blanket "sweep failed" shortcut.
    assert!(audit["semantic_labels"].as_array().unwrap().is_empty());
}

/// `dice_floor`/`cosine_floor` share `/vocabulary/audit`'s body with
/// every other optional field: a non-numeric value is the same
/// malformed-JSON 400 `errors.rs`'s
/// `optional_body_endpoints_reject_a_non_json_body` already pins for
/// the body as a whole, now pinned for these two fields specifically.
#[test]
fn dice_floor_and_cosine_floor_reject_the_wrong_json_type() {
    let server = Server::start("vocab-floor-wrong-type");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/vocabulary/audit",
        Some(json!({"dice_floor": "high"})),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("malformed_request"), "{body}");

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/vocabulary/audit",
        Some(json!({"cosine_floor": "high"})),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("malformed_request"), "{body}");
}
