//! `POST /contexts/{name}/schema/audit` and `POST
//! /contexts/{name}/schema/validate` (#385, S7 of #218's ADR 0009 split
//! §10): the standing, read-only audit over the live graph and the
//! never-persisted dry-run of a proposed document. Both share
//! [`crate::api::schema::schema_audit`]'s judgment with `strict` itself
//! (S3, #381's `schema_issues`), so a finding here is exactly what a
//! `strict` write would refuse for the same fact — this file pins that
//! contract, not `GET`/`PUT /schema`'s own round trip (`schema.rs`).

use serde_json::json;

use crate::support::*;

fn strict_document() -> serde_json::Value {
    json!({
        "schema": 1,
        "mode": "strict",
        "closed_labels": false,
        "types": {
            "Brewery": {"is_a": ["Organization"]},
            "Organization": {"is_a": []},
            "Person": {"is_a": []}
        },
        "relations": {
            "杜氏": {"domain": ["Brewery"], "range": ["Person"]}
        }
    })
}

/// `audit`'s 404s mirror `GET /schema`'s own distinction (ADR 0009
/// §6.3): no context at all vs. a context that simply never installed a
/// schema.
#[test]
fn audit_refuses_with_no_schema_or_no_context() {
    let server = Server::start("schema-audit-404");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    let (status, body) = server.call("POST", "/contexts/sake/schema/audit", None);
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["code"], "no_schema", "{body}");

    let (status, body) = server.call("POST", "/contexts/nope/schema/audit", None);
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["code"], "no_context", "{body}");
}

/// `SchemaAuditRequest` now denies unknown fields (issue #623 finding
/// 1), matching its sibling `SchemaValidateRequest` — a typo in
/// `limit`/`after` refuses instead of silently falling back to the
/// default page. `audit_schema` parses the body itself via
/// `optional_body` (not axum's `Json` extractor, which `PUT /schema`
/// uses and which 422s), so a bad body here is `error`'s ordinary 400,
/// and fires before the context lookup.
#[test]
fn audit_denies_unknown_fields() {
    let server = Server::start("schema-audit-unknown-field");

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/schema/audit",
        Some(json!({"limit": 5, "typo": true})),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], "malformed_request", "{body}");
}

/// ADR 0009 §7.1: "pre-existing violations are visible only through the
/// explicitly-invoked, read-only audit" — the audit must judge by the
/// same domain/range predicate `strict` would, even when the document's
/// actual `mode` is `off`. This is the one behavior that could not be
/// pinned by reusing `schema_issues`' own unit tests: those all pass
/// `mode` in directly.
#[test]
fn audit_reports_domain_violations_even_in_off_mode() {
    let server = Server::start("schema-audit-off-mode");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "高瀬", "label": "schema:type", "object": "Person",
             "weight": 1.0, "source": "a.md"},
            {"subject": "高瀬", "label": "杜氏", "object": "個人A",
             "weight": 1.0, "source": "a.md"},
        ])),
    );
    let mut document = strict_document();
    document["mode"] = json!("off");
    server.ok("PUT", "/contexts/sake/schema", Some(document));

    let audit = server.ok("POST", "/contexts/sake/schema/audit", None);
    assert_eq!(audit["total"], json!(1), "{audit}");
    let violations = audit["violations"].as_array().unwrap();
    assert_eq!(violations.len(), 1, "{audit}");
    let violation = &violations[0];
    assert_eq!(
        violation["association"]["subject"],
        json!("高瀬"),
        "{audit}"
    );
    assert_eq!(violation["association"]["label"], json!("杜氏"), "{audit}");
    let issues = violation["issues"].as_array().unwrap();
    assert_eq!(issues.len(), 1, "{audit}");
    assert_eq!(issues[0]["kind"], json!("domain"), "{audit}");
    // IssuePath::Edge names only the side, never a request-body index —
    // the associated edge already travels alongside the issue.
    assert_eq!(issues[0]["path"], json!("subject"), "{audit}");
    assert!(issues[0]["actual"].as_str().unwrap().contains("Person"));
}

/// `warn` and `strict` must answer identically to `off` — the whole
/// point of §7.1's framing is that the audit's answer never depends on
/// the document's own current mode.
#[test]
fn audit_answers_the_same_regardless_of_mode() {
    let server = Server::start("schema-audit-mode-invariant");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "高瀬", "label": "schema:type", "object": "Person",
             "weight": 1.0, "source": "a.md"},
            {"subject": "高瀬", "label": "杜氏", "object": "個人A",
             "weight": 1.0, "source": "a.md"},
        ])),
    );

    let mut audits = Vec::new();
    for mode in ["off", "warn", "strict"] {
        let mut document = strict_document();
        document["mode"] = json!(mode);
        server.ok("PUT", "/contexts/sake/schema", Some(document));
        audits.push(server.ok("POST", "/contexts/sake/schema/audit", None));
    }
    // The whole response, not just `total` — every section (violations'
    // issue detail included) must be identical across modes, not merely
    // the same count.
    assert_eq!(audits[0], audits[1], "off vs warn: {audits:?}");
    assert_eq!(audits[1], audits[2], "warn vs strict: {audits:?}");
}

/// `untyped_concepts` excludes concepts that are themselves asserted
/// type names (§6.3 exclusion 3's own reasoning) and concepts already
/// declared in `types`, but still names an ordinary fact concept that
/// never received a `schema:type` assertion.
#[test]
fn audit_untyped_concepts_excludes_type_names_and_declared_types() {
    let server = Server::start("schema-audit-untyped");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "schema:type", "object": "Brewery",
             "weight": 1.0, "source": "a.md"},
            // "個人A" carries no schema:type assertion of its own.
            {"subject": "青嶺酒造", "label": "所在地", "object": "個人A",
             "weight": 1.0, "source": "a.md"},
        ])),
    );
    server.ok("PUT", "/contexts/sake/schema", Some(strict_document()));

    let audit = server.ok("POST", "/contexts/sake/schema/audit", None);
    let names: Vec<&str> = audit["untyped_concepts"]["names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(names.contains(&"個人A"), "{audit}");
    assert!(
        !names.contains(&"青嶺酒造"),
        "typed via its own schema:type assertion: {audit}"
    );
    assert!(
        !names.contains(&"Brewery"),
        "a type name itself, excluded per §6.3 exclusion 3: {audit}"
    );
    assert!(
        !names.contains(&"Organization") && !names.contains(&"Person"),
        "declared types with no live use are not \"untyped concepts\": {audit}"
    );
}

/// §6.2: a type name asserted but absent from `types` is always
/// reported, unconditional on `closed_labels`.
#[test]
fn audit_undeclared_types_reported_regardless_of_closed_labels() {
    let server = Server::start("schema-audit-undeclared-types");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "schema:type", "object": "Distillery",
             "weight": 1.0, "source": "a.md"},
        ])),
    );
    for closed_labels in [false, true] {
        let mut document = strict_document();
        document["closed_labels"] = json!(closed_labels);
        server.ok("PUT", "/contexts/sake/schema", Some(document));
        let audit = server.ok("POST", "/contexts/sake/schema/audit", None);
        assert_eq!(
            audit["undeclared_types"]["names"],
            json!(["Distillery"]),
            "closed_labels={closed_labels}: {audit}"
        );
    }
}

/// §6.4: `unknown_labels` stays empty unless `closed_labels` is set, and
/// even then never names `schema:type` itself.
#[test]
fn audit_unknown_labels_only_when_closed_labels_and_never_schema_type() {
    let server = Server::start("schema-audit-unknown-labels");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "schema:type", "object": "Brewery",
             "weight": 1.0, "source": "a.md"},
            {"subject": "青嶺酒造", "label": "所在地", "object": "広島",
             "weight": 1.0, "source": "a.md"},
        ])),
    );

    server.ok("PUT", "/contexts/sake/schema", Some(strict_document()));
    let open = server.ok("POST", "/contexts/sake/schema/audit", None);
    assert_eq!(open["unknown_labels"]["names"], json!([]), "{open}");

    let mut closed = strict_document();
    closed["closed_labels"] = json!(true);
    server.ok("PUT", "/contexts/sake/schema", Some(closed));
    let audit = server.ok("POST", "/contexts/sake/schema/audit", None);
    let names: Vec<&str> = audit["unknown_labels"]["names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["所在地"], "{audit}");
    assert!(
        !names.contains(&"schema:type"),
        "schema:type never counts as an unknown label, even under closed_labels: {audit}"
    );
    // The same fact also drives a per-edge `violations` entry — the two
    // sections answer different questions, so both fire together.
    let violations = audit["violations"].as_array().unwrap();
    assert!(
        violations
            .iter()
            .any(|v| v["association"]["label"] == json!("所在地")
                && v["issues"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|issue| issue["kind"] == json!("unknown_reference"))),
        "{audit}"
    );
}

/// `violations` pages the same way `drift/audit`'s `unsourced` does:
/// `limit`/`after` resume in the same worst-magnitude-first order, and
/// `total` stays constant across pages.
#[test]
fn audit_violations_page_like_every_other_match_list() {
    let server = Server::start("schema-audit-paging");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "高瀬", "label": "schema:type", "object": "Person",
             "weight": 1.0, "source": "a.md"},
            {"subject": "高瀬", "label": "杜氏", "object": "弟子1",
             "weight": 3.0, "source": "a.md"},
            {"subject": "高瀬", "label": "杜氏", "object": "弟子2",
             "weight": 2.0, "source": "a.md"},
            {"subject": "高瀬", "label": "杜氏", "object": "弟子3",
             "weight": 1.0, "source": "a.md"},
        ])),
    );
    server.ok("PUT", "/contexts/sake/schema", Some(strict_document()));

    let full = server.ok("POST", "/contexts/sake/schema/audit", None);
    assert_eq!(full["total"], json!(3), "{full}");
    let full_matches = full["violations"].as_array().unwrap();
    // Worst-magnitude-first, exactly like `drift/audit`'s `unsourced`.
    assert_eq!(
        full_matches
            .iter()
            .map(|v| v["association"]["object"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["弟子1", "弟子2", "弟子3"],
        "{full}"
    );

    let first = server.ok(
        "POST",
        "/contexts/sake/schema/audit",
        Some(json!({"limit": 2})),
    );
    assert_eq!(first["total"], json!(3), "{first}");
    let first_matches = first["violations"].as_array().unwrap();
    assert_eq!(
        first_matches
            .iter()
            .map(|v| v["association"]["object"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["弟子1", "弟子2"],
        "{first}"
    );
    let last = &first_matches[1]["association"];
    let cursor = json!({
        "weight": last["weight"], "subject": last["subject"],
        "label": last["label"], "object": last["object"],
    });
    let second = server.ok(
        "POST",
        "/contexts/sake/schema/audit",
        Some(json!({"limit": 2, "after": cursor})),
    );
    assert_eq!(second["total"], json!(3), "{second}");
    let second_matches = second["violations"].as_array().unwrap();
    // Resumes exactly where the first page stopped — 弟子3 alone, never
    // 弟子1/弟子2 again — and the two pages together reconstruct the
    // whole unpaginated order with no gap or duplicate.
    assert_eq!(
        second_matches
            .iter()
            .map(|v| v["association"]["object"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["弟子3"],
        "{second}"
    );
    assert_eq!(second_matches[0], full_matches[2], "{second} vs {full}");
}

/// §6.3 guard 2's migration-boundary bullet refuses `PUT /schema`
/// outright when a pre-existing alias already resolves to the reserved
/// label (`schema.rs`'s own
/// `a_label_alias_resolving_to_the_reserved_type_label_refuses_the_put`)
/// — so a RESIDENT schema can never carry this conflict, only a
/// *proposed* one can. `validate` is exactly where an operator would
/// discover it: before ever attempting the `PUT` that would otherwise
/// just 400 on the alias alone, with none of the other findings this
/// route surfaces alongside it.
#[test]
fn validate_surfaces_a_pre_existing_reserved_alias_conflict() {
    let server = Server::start("schema-validate-reserved-alias");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    // Legal today — guard 1: `schema:type` is an ordinary label until a
    // schema exists — and interns the label id the alias resolves
    // against.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "蔵", "label": "schema:type", "object": "Brewery",
             "weight": 1.0, "source": "a.md"},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"labels": {"種類": "schema:type"}})),
    );

    let audit = server.ok(
        "POST",
        "/contexts/sake/schema/validate",
        Some(json!({"document": strict_document()})),
    );
    assert_eq!(
        audit["reserved_alias_conflicts"],
        json!({"total": 1, "aliases": {"種類": "schema:type"}}),
        "{audit}"
    );

    // Confirms the scenario this section exists to warn about: the same
    // document really does refuse at `PUT` time, naming the same alias.
    let (status, body) = server.call("PUT", "/contexts/sake/schema", Some(strict_document()));
    assert_eq!(status, 400, "{body}");
    assert!(body["error"].as_str().unwrap().contains("種類"), "{body}");
}

/// `validate` judges a PROPOSED document the same way `audit` judges
/// the resident one, but never persists it — `GET /schema` still 404s
/// afterward.
#[test]
fn validate_dry_runs_without_persisting() {
    let server = Server::start("schema-validate-dry-run");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "高瀬", "label": "schema:type", "object": "Person",
             "weight": 1.0, "source": "a.md"},
            {"subject": "高瀬", "label": "杜氏", "object": "個人A",
             "weight": 1.0, "source": "a.md"},
        ])),
    );

    let audit = server.ok(
        "POST",
        "/contexts/sake/schema/validate",
        Some(json!({"document": strict_document()})),
    );
    assert_eq!(audit["total"], json!(1), "{audit}");
    let violations = audit["violations"].as_array().unwrap();
    assert_eq!(
        violations[0]["issues"][0]["kind"],
        json!("domain"),
        "{audit}"
    );

    let (status, body) = server.call("GET", "/contexts/sake/schema", None);
    assert_eq!(status, 404, "validate must never persist: {body}");
    assert_eq!(body["code"], "no_schema", "{body}");
}

/// `validate` works over a schema-free context — the primary dry-run
/// use case, ADR 0009 §7.1's pre-flight before a `strict` flip — and
/// answers identically whether the context already has an installed
/// schema or none at all.
#[test]
fn validate_works_whether_or_not_a_schema_is_already_installed() {
    let server = Server::start("schema-validate-either-way");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "高瀬", "label": "schema:type", "object": "Person",
             "weight": 1.0, "source": "a.md"},
            {"subject": "高瀬", "label": "杜氏", "object": "個人A",
             "weight": 1.0, "source": "a.md"},
        ])),
    );

    let without = server.ok(
        "POST",
        "/contexts/sake/schema/validate",
        Some(json!({"document": strict_document()})),
    );

    // Install an unrelated, non-violating schema, then re-run the same
    // proposed document — the resident schema must play no part.
    server.ok(
        "PUT",
        "/contexts/sake/schema",
        Some(json!({
            "schema": 1, "mode": "off", "closed_labels": false,
            "types": {}, "relations": {}
        })),
    );
    let with = server.ok(
        "POST",
        "/contexts/sake/schema/validate",
        Some(json!({"document": strict_document()})),
    );
    // The whole response, not just `total`/`violations` — every section
    // must agree, since the resident schema plays no part in `validate`
    // at all.
    assert_eq!(without, with, "{without} vs {with}");
}

/// A malformed proposed document (here: a relation named the reserved
/// `schema:type` label, ADR 0009 §6.3 guard 3) refuses the same way
/// `PUT /schema` itself would — 400 `invalid_argument`, never a 500 or a
/// silently-accepted document.
#[test]
fn validate_refuses_an_invalid_proposed_document() {
    let server = Server::start("schema-validate-invalid-document");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    let mut document = strict_document();
    document["relations"]["schema:type"] = json!({"domain": [], "range": []});
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/schema/validate",
        Some(json!({"document": document})),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], "invalid_argument", "{body}");
}

// --- MAX_AUDIT_NAMES (100): each name-list section truncates -----------

/// `untyped_concepts` past `MAX_AUDIT_NAMES` (100): `total` reports the
/// true count, `names` is a name-ordered (`BTreeSet`) prefix capped at
/// 100 — zero-padded objects keep lexicographic and numeric order
/// identical, so the excluded one is deterministically the highest.
#[test]
fn audit_untyped_concepts_truncates_past_max_audit_names() {
    let server = Server::start("schema-audit-untyped-cap");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    let mut ops: Vec<serde_json::Value> = vec![json!({
        "subject": "青嶺酒造", "label": "schema:type", "object": "Brewery",
        "weight": 1.0, "source": "a.md",
    })];
    ops.extend((0..101).map(|i| {
        json!({
            "subject": "青嶺酒造", "label": "所在地", "object": format!("個人{i:03}"),
            "weight": 1.0, "source": "a.md",
        })
    }));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(serde_json::Value::Array(ops)),
    );
    server.ok("PUT", "/contexts/sake/schema", Some(strict_document()));

    let audit = server.ok("POST", "/contexts/sake/schema/audit", None);
    assert_eq!(audit["untyped_concepts"]["total"], json!(101), "{audit}");
    let names: Vec<&str> = audit["untyped_concepts"]["names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(names.len(), 100, "{audit}");
    assert!(names.contains(&"個人000"), "{audit:?}");
    assert!(names.contains(&"個人099"), "{audit:?}");
    assert!(
        !names.contains(&"個人100"),
        "the 101st name (highest-ordered) is the one past the cap: {names:?}"
    );
}

/// `undeclared_types` past `MAX_AUDIT_NAMES` (100) — same truncation
/// contract, over asserted `schema:type` objects instead.
#[test]
fn audit_undeclared_types_truncates_past_max_audit_names() {
    let server = Server::start("schema-audit-undeclared-cap");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    let ops: Vec<serde_json::Value> = (0..101)
        .map(|i| {
            json!({
                "subject": format!("組織{i:03}"), "label": "schema:type", "object": format!("型{i:03}"),
                "weight": 1.0, "source": "a.md",
            })
        })
        .collect();
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(serde_json::Value::Array(ops)),
    );
    server.ok("PUT", "/contexts/sake/schema", Some(strict_document()));

    let audit = server.ok("POST", "/contexts/sake/schema/audit", None);
    assert_eq!(audit["undeclared_types"]["total"], json!(101), "{audit}");
    let names: Vec<&str> = audit["undeclared_types"]["names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(names.len(), 100, "{audit}");
    assert!(names.contains(&"型000"), "{audit:?}");
    assert!(names.contains(&"型099"), "{audit:?}");
    assert!(
        !names.contains(&"型100"),
        "the 101st name (highest-ordered) is the one past the cap: {names:?}"
    );
}

/// `unknown_labels` past `MAX_AUDIT_NAMES` (100) — only populated under
/// `closed_labels`, same truncation contract as the other two sections.
#[test]
fn audit_unknown_labels_truncates_past_max_audit_names() {
    let server = Server::start("schema-audit-unknown-labels-cap");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    let ops: Vec<serde_json::Value> = (0..101)
        .map(|i| {
            json!({
                "subject": format!("s{i:03}"), "label": format!("未知{i:03}"), "object": format!("o{i:03}"),
                "weight": 1.0, "source": "a.md",
            })
        })
        .collect();
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(serde_json::Value::Array(ops)),
    );
    let mut document = strict_document();
    document["mode"] = json!("warn");
    document["closed_labels"] = json!(true);
    server.ok("PUT", "/contexts/sake/schema", Some(document));

    let audit = server.ok("POST", "/contexts/sake/schema/audit", None);
    assert_eq!(audit["unknown_labels"]["total"], json!(101), "{audit}");
    let names: Vec<&str> = audit["unknown_labels"]["names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(names.len(), 100, "{audit}");
    assert!(names.contains(&"未知000"), "{audit:?}");
    assert!(names.contains(&"未知099"), "{audit:?}");
    assert!(
        !names.contains(&"未知100"),
        "the 101st name (highest-ordered) is the one past the cap: {names:?}"
    );
}
