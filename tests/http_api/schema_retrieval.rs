//! ADR 0009 §12 (#387, the read-side minimum): `describe` returns a
//! concept's types, `resolve` attaches them to its top candidates, and
//! `query`/cross `query` gain `subject_types`/`object_types` filters —
//! all three gated by §6.3's single condition (an installed schema
//! document), never by `mode`. This file pins the read side; the
//! write side (`schema_issues`, S3–S5) already has its own coverage in
//! `schema.rs`/`schema_audit.rs`/`schema_import.rs`.

use serde_json::{Value, json};

use crate::support::*;

fn schema_document(mode: &str) -> Value {
    json!({
        "schema": 1,
        "mode": mode,
        "closed_labels": false,
        "types": {
            "Organization": {"is_a": []},
            "Brewery": {"is_a": ["Organization"]},
            "Person": {"is_a": []}
        },
        "relations": {}
    })
}

fn install_schema(server: &Server, context: &str, mode: &str) {
    server.ok(
        "PUT",
        &format!("/contexts/{context}/schema"),
        Some(schema_document(mode)),
    );
}

fn assoc(subject: &str, label: &str, object: &str) -> Value {
    json!({"subject": subject, "label": label, "object": object,
           "weight": 1.0, "source": "a.md"})
}

/// Seeds one context with a brewer typed `Brewery` (which `is_a`
/// `Organization`), a plain fact between the same two concepts, and an
/// untyped third concept — the fixture every test below reuses.
fn seed(server: &Server, context: &str) {
    server.ok(
        "PUT",
        &format!("/contexts/{context}"),
        Some(json!({"description": "d"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{context}/associations"),
        Some(json!([
            assoc("青嶺酒造", "schema:type", "Brewery"),
            assoc("青嶺酒造", "杜氏", "高瀬"),
            assoc("高瀬", "schema:type", "Person"),
            assoc("霧沢町", "所在", "雲居県"),
        ])),
    );
}

/// §6.3 guard 1: `describe` reports types only once a schema document
/// is installed, and `schema:type` itself is not excluded from the
/// label tally (only activate/explore, the vocabulary block, and the
/// twin sweep are — never describe).
#[test]
fn describe_reports_types_only_once_a_schema_is_installed() {
    let server = Server::start("schema-retrieval-describe");
    seed(&server, "sake");

    let before = server.ok(
        "POST",
        "/contexts/sake/describe",
        Some(json!({"concept": "青嶺酒造"})),
    );
    assert_eq!(before["types"], json!([]), "{before}");
    assert!(
        before["as_subject"]
            .as_array()
            .unwrap()
            .iter()
            .any(|usage| usage["label"] == json!("schema:type")),
        "schema:type is never excluded from describe's own tally: {before}"
    );

    install_schema(&server, "sake", "warn");
    let after = server.ok(
        "POST",
        "/contexts/sake/describe",
        Some(json!({"concept": "青嶺酒造"})),
    );
    assert_eq!(after["types"], json!(["Brewery"]), "{after}");

    // An untyped concept answers empty, not an error.
    let untyped = server.ok(
        "POST",
        "/contexts/sake/describe",
        Some(json!({"concept": "霧沢町"})),
    );
    assert_eq!(untyped["types"], json!([]), "{untyped}");
}

/// ADR 0009 §12: `resolve` attaches `types` to its top candidates only,
/// and `resolve_label` never carries them — a relation label has no
/// type.
#[test]
fn resolve_attaches_types_to_top_candidates_but_resolve_label_never_does() {
    let server = Server::start("schema-retrieval-resolve");
    seed(&server, "sake");
    install_schema(&server, "sake", "warn");

    let resolved = server.ok(
        "POST",
        "/contexts/sake/resolve",
        Some(json!({"cue": "青嶺酒造"})),
    );
    assert_eq!(resolved[0]["name"], json!("青嶺酒造"), "{resolved}");
    assert_eq!(resolved[0]["types"], json!(["Brewery"]), "{resolved}");

    // Untyped candidate: the field is simply absent (never `[]`) —
    // `#[serde(skip_serializing_if = "Option::is_none")]`, same as
    // `gloss`.
    let untyped = server.ok(
        "POST",
        "/contexts/sake/resolve",
        Some(json!({"cue": "霧沢町"})),
    );
    assert!(
        untyped[0].get("types").is_none(),
        "an untyped candidate must not carry the field at all: {untyped}"
    );

    let labels = server.ok(
        "POST",
        "/contexts/sake/resolve_label",
        Some(json!({"cue": "杜氏"})),
    );
    assert!(
        labels[0].get("types").is_none(),
        "resolve_label candidates never carry types: {labels}"
    );
}

/// The core filter contract: `subject_types`/`object_types` narrow
/// `query`'s output, `is_a`-expanded (a `Brewery` subject satisfies an
/// `Organization` filter), `total` reflects the filtered count, and an
/// unknown type name matches nothing. Pinned by `label` rather than
/// `subject` throughout, so the two candidate facts (`青嶺酒造 杜氏 高瀬`,
/// typed subject; `霧沢町 所在 雲居県`, untyped subject) carry DIFFERENT
/// subjects — a type is a property of the concept, not the edge, so
/// pinning the subject itself would make every edge off it share one
/// verdict and hide any actual narrowing.
#[test]
fn query_filters_by_type_and_expands_through_is_a() {
    let server = Server::start("schema-retrieval-query-filter");
    seed(&server, "sake");
    install_schema(&server, "sake", "warn");
    let facts = json!(["杜氏", "所在"]);

    let unfiltered = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"label": facts})),
    );
    assert_eq!(unfiltered["total"], json!(2), "{unfiltered}");

    let direct = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"label": facts, "subject_types": "Brewery"})),
    );
    assert_eq!(direct["total"], json!(1), "{direct}");
    assert_eq!(
        direct["matches"][0]["subject"],
        json!("青嶺酒造"),
        "{direct}"
    );

    // `is_a` closure: a filter on the PARENT type still matches a
    // concept only ever asserted as the child.
    let via_ancestor = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"label": facts, "subject_types": "Organization"})),
    );
    assert_eq!(via_ancestor["total"], json!(1), "{via_ancestor}");

    // object_types narrows the other side; the untyped 霧沢町/雲居県 fact
    // never matches a non-empty object filter.
    let object_filtered = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"label": facts, "object_types": "Person"})),
    );
    assert_eq!(object_filtered["total"], json!(1), "{object_filtered}");
    assert_eq!(
        object_filtered["matches"][0]["object"],
        json!("高瀬"),
        "{object_filtered}"
    );
    let no_match = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"label": facts, "object_types": "Organization"})),
    );
    assert_eq!(
        no_match["total"],
        json!(0),
        "高瀬 is a Person, not an Organization: {no_match}"
    );

    // An unknown type name matches nothing, without erroring.
    let unknown = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"label": facts, "subject_types": "存在しない型"})),
    );
    assert_eq!(unknown["total"], json!(0), "{unknown}");
}

/// §6.3 guard 1: a schema-free context answers an empty page for any
/// non-empty type filter — never an error, never the unfiltered set.
#[test]
fn query_type_filter_on_a_schema_free_context_answers_empty() {
    let server = Server::start("schema-retrieval-query-no-schema");
    seed(&server, "sake");

    let filtered = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "subject_types": "Brewery"})),
    );
    assert_eq!(filtered["total"], json!(0), "{filtered}");

    // The position pins still work — only the type axis is gated.
    let unfiltered = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造"})),
    );
    assert_eq!(unfiltered["total"], json!(2), "{unfiltered}");
}

/// `subject_types`/`object_types` are filters, never anchors: leaving
/// subject/label/object all unset is still refused even with a type
/// filter present, exactly as an unfiltered call already is.
#[test]
fn query_type_filter_never_satisfies_the_at_least_one_position_rule() {
    let server = Server::start("schema-retrieval-query-anchor");
    seed(&server, "sake");
    install_schema(&server, "sake", "warn");

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject_types": "Brewery"})),
    );
    assert_eq!(status, 400, "{body}");
}

/// A filtered and an unfiltered call for identical positions must never
/// share a retrieval-cache entry — pinned via `/metrics`' per-op hit
/// counter the same way `retrieval_cache.rs` does.
#[test]
fn query_type_filter_does_not_share_a_cache_entry_with_the_unfiltered_call() {
    fn cache_misses(server: &Server, op: &str) -> u64 {
        let (status, body) = server.call("GET", "/metrics", None);
        assert_eq!(status, 200);
        let text = body.as_str().unwrap();
        let prefix = format!("taguru_retrieval_cache_total{{op=\"{op}\",outcome=\"miss\"}}");
        text.lines()
            .find_map(|line| {
                line.strip_prefix(&prefix)
                    .and_then(|rest| rest.trim().parse::<u64>().ok())
            })
            .unwrap_or(0)
    }

    let server = Server::start("schema-retrieval-query-cache");
    seed(&server, "sake");
    install_schema(&server, "sake", "warn");

    let query = |body: Value| server.ok("POST", "/contexts/sake/query", Some(body));
    query(json!({"subject": "青嶺酒造"}));
    query(json!({"subject": "青嶺酒造", "subject_types": "Brewery"}));
    assert_eq!(
        cache_misses(&server, "query"),
        2,
        "two distinct keys must both miss, never one hit off the other's entry"
    );
    // Repeating the filtered call now hits its own entry.
    query(json!({"subject": "青嶺酒造", "subject_types": "Brewery"}));
    assert_eq!(cache_misses(&server, "query"), 2, "the repeat must hit");
}

/// Cross `POST /query` evaluates the filter per target against THAT
/// target's own installed schema (or none) — one context typed, one
/// schema-free, in the same fan-out.
#[test]
fn cross_query_type_filter_is_evaluated_per_target_schema() {
    let server = Server::start("schema-retrieval-cross-query");
    seed(&server, "typed");
    install_schema(&server, "typed", "warn");
    seed(&server, "untyped");

    let answer = server.ok(
        "POST",
        "/query",
        Some(json!({
            "contexts": ["typed", "untyped"],
            "label": ["杜氏", "所在"],
            "subject_types": "Brewery",
        })),
    );
    // Only "typed"'s 青嶺酒造 (Brewery) edge survives: "typed"'s own
    // untyped 霧沢町 edge and every one of "untyped"'s edges (no
    // installed schema at all, §6.3 guard 1) are filtered out.
    assert_eq!(answer["total"], json!(1), "{answer}");
    assert_eq!(answer["matches"][0]["context"], json!("typed"), "{answer}");
    assert_eq!(
        answer["matches"][0]["subject"],
        json!("青嶺酒造"),
        "{answer}"
    );
}

/// The MCP `query` tool forwards `subject_types`/`object_types` to its
/// HTTP twin unchanged, exactly like every other query parameter.
#[test]
fn mcp_query_tool_forwards_type_filters() {
    let server = Server::start("schema-retrieval-mcp-query");
    seed(&server, "sake");
    install_schema(&server, "sake", "warn");

    let reply = server.call_tool(
        1,
        "query",
        json!({"context": "sake", "label": ["杜氏", "所在"], "subject_types": "Brewery"}),
    );
    assert!(reply.get("isError").is_none(), "{reply}");
    let text = reply["content"][0]["text"].as_str().unwrap();
    let body: Value = serde_json::from_str(text).unwrap();
    assert_eq!(body["result"]["total"], json!(1), "{text}");
}
