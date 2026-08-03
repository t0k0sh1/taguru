//! The reserved `schema:type` label's namespace guards and read-side
//! exclusions (#381, S3 of #218's ADR 0009 split, §6.3): inert until a
//! schema document exists for the context (guard 1), an alias refusal
//! that fires in every mode including `off` once one does (guard 2's
//! `add_label_alias` bullet), and the three exclusions the label's
//! representation cost requires — traversal, the default label page,
//! and the vocabulary twin audit. `PUT /schema`'s own refusals (guard
//! 3, guard 2's migration-boundary bullet) live in `schema.rs`; the
//! domain/range judgment itself (`schema_issues`) has no write entrance
//! wired yet (S4/#382, S5/#383) and is unit-tested in
//! `src/schema/check.rs`.

use serde_json::json;

use crate::support::*;

/// The gate ADR 0009 §6.3 fixes is "an installed schema document
/// exists," never "mode != off" — so this installs in `off` throughout,
/// to pin that the exclusions and guard 2 do not quietly wait for
/// `warn`/`strict`.
fn off_document() -> serde_json::Value {
    json!({
        "schema": 1,
        "mode": "off",
        "closed_labels": false,
        "types": {},
        "relations": {}
    })
}

/// Guard 1 as a baseline, then the gate flip: before any schema exists,
/// `schema:type` is an ordinary label — it appears in the label
/// vocabulary and, because two concepts share a type object, bridges
/// them in `explore` exactly like any other shared-object edge would.
/// Once a schema installs (`off` mode included), both effects vanish:
/// the label disappears from `GET .../labels` (default page and
/// `?prefix=`) and no longer bridges the two concepts.
#[test]
fn schema_type_is_ordinary_until_a_schema_exists_then_hidden_from_labels_and_traversal() {
    let server = Server::start("schema-type-label-explore");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    // Two concepts, connected only through a `schema:type` edge to a
    // shared type object — nothing else links them.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "schema:type", "object": "Brewery",
             "weight": 1.0, "source": "a.md"},
            {"subject": "旧銘酒造", "label": "schema:type", "object": "Brewery",
             "weight": 1.0, "source": "a.md"},
        ])),
    );

    let labels = server.ok("GET", "/contexts/sake/labels", None);
    assert!(
        labels["labels"]
            .as_array()
            .unwrap()
            .iter()
            .any(|label| label == "schema:type"),
        "guard 1 (no schema yet): schema:type is an ordinary label: {labels}"
    );
    let total_before = labels["total"].as_u64().unwrap();

    let reaches_the_other_brewery = |server: &Server| {
        let explored = server.ok(
            "POST",
            "/contexts/sake/explore",
            Some(json!({"origins": ["青嶺酒造"], "max_depth": 2})),
        );
        explored["matches"]
            .as_array()
            .unwrap()
            .iter()
            .any(|recollection| recollection["association"]["subject"] == "旧銘酒造")
    };
    assert!(
        reaches_the_other_brewery(&server),
        "guard 1: schema:type bridges concepts through the shared type object like any \
         other label, before a schema exists"
    );

    // Guard 2 before install is legal today (`tests/http_api/schema.rs`
    // covers that pairing, plus the install-time migration-boundary
    // guard a pre-existing such alias then runs into) — this test's own
    // concern is the traversal/label exclusion, so it moves straight to
    // the gate flip.

    // The gate flip — installed in `off` mode, deliberately.
    server.ok("PUT", "/contexts/sake/schema", Some(off_document()));

    let labels = server.ok("GET", "/contexts/sake/labels", None);
    assert!(
        !labels["labels"]
            .as_array()
            .unwrap()
            .iter()
            .any(|label| label == "schema:type"),
        "once a schema exists, schema:type is hidden from the default page even in mode \
         off: {labels}"
    );
    assert_eq!(
        labels["total"].as_u64().unwrap(),
        total_before - 1,
        "{labels}"
    );
    let prefixed = server.ok("GET", "/contexts/sake/labels?prefix=schema:", None);
    assert_eq!(
        prefixed["labels"],
        json!([]),
        "?prefix= must not be a back door around the same exclusion: {prefixed}"
    );

    assert!(
        !reaches_the_other_brewery(&server),
        "once a schema exists, schema:type must never bridge concepts, even in mode off"
    );

    // Guard 2 again, now that a schema exists: a NEW alias resolving to
    // the reserved label refuses, in `off` mode, naming the alias and
    // instructing a rename.
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"labels": {"種類": "schema:type"}})),
    );
    assert_eq!(status, 400, "{body}");
    assert!(body["error"].as_str().unwrap().contains("種類"), "{body}");
    assert!(body["error"].as_str().unwrap().contains("rename"), "{body}");
}

/// `activate`'s ranked sibling of the same exclusion — the fan
/// normalization and the propagation walk must both drop `schema:type`
/// edges, or a heavily-typed concept's real facts would be diluted by
/// its own hidden type assertions.
///
/// Reachability alone cannot tell the two halves apart (propagation
/// stopping is enough to make `旧銘酒造` unreachable even if the fan
/// denominator still counts the hidden edge), and — per
/// [`Context::activate`]'s own contract — a fact returned *directly
/// from the origin* is `activation(origin) * decay * |sum|`,
/// independent of the origin's own fan-out, so checking a one-hop
/// fact's strength would not catch dilution either. Only a fact
/// *beyond* the origin depends on how much the origin's own total
/// divides the flow reaching it, so this checks a two-hop fact's
/// strength instead: `青嶺酒造 →(所在地)→ 広島 →(属する)→ 中国地方`.
/// `広島`'s activation is `score(所在地) / total(青嶺酒造)` — larger
/// once `schema:type` no longer inflates that total — and `中国地方`'s
/// returned strength scales with `広島`'s activation.
#[test]
fn activate_never_propagates_through_schema_type_once_a_schema_exists() {
    let server = Server::start("schema-type-label-activate");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "schema:type", "object": "Brewery",
             "weight": 1.0, "source": "a.md"},
            {"subject": "旧銘酒造", "label": "schema:type", "object": "Brewery",
             "weight": 1.0, "source": "a.md"},
            {"subject": "青嶺酒造", "label": "所在地", "object": "広島",
             "weight": 1.0, "source": "a.md"},
            {"subject": "広島", "label": "属する", "object": "中国地方",
             "weight": 1.0, "source": "a.md"},
        ])),
    );

    let activate = |server: &Server| {
        server.ok(
            "POST",
            "/contexts/sake/activate",
            Some(json!({"origins": ["青嶺酒造"]})),
        )
    };
    let reaches_the_other_brewery = |activated: &serde_json::Value| {
        activated["matches"]
            .as_array()
            .unwrap()
            .iter()
            .any(|activation| activation["association"]["subject"] == "旧銘酒造")
    };
    let two_hop_strength = |activated: &serde_json::Value| {
        activated["matches"]
            .as_array()
            .unwrap()
            .iter()
            .find(|activation| activation["association"]["object"] == "中国地方")
            .map(|activation| activation["strength"].as_f64().unwrap())
    };

    let before = activate(&server);
    assert!(
        reaches_the_other_brewery(&before),
        "before a schema exists, activate propagates through schema:type like any other \
         label"
    );
    let diluted_strength =
        two_hop_strength(&before).expect("中国地方 is reachable in two hops: {before}");

    server.ok("PUT", "/contexts/sake/schema", Some(off_document()));

    let after = activate(&server);
    assert!(
        !reaches_the_other_brewery(&after),
        "once a schema exists, activate must never propagate through schema:type"
    );
    // The fan denominator must drop the schema:type edge too, not only
    // the walk — otherwise 青嶺酒造's real fan-out still counts its own
    // now-hidden type assertion, and everything beyond it stays
    // diluted by exactly the same amount as before.
    let undiluted_strength =
        two_hop_strength(&after).expect("中国地方 is still reachable in two hops: {after}");
    assert!(
        undiluted_strength > diluted_strength,
        "the fan total must exclude schema:type too: diluted={diluted_strength} \
         undiluted={undiluted_strength}"
    );
}

/// Guard 2's refusal must fire against a fresh alias in `warn`/`strict`
/// too, not only `off` — the gate is document existence, not mode.
#[test]
fn the_reserved_alias_refusal_fires_in_warn_and_strict_too() {
    let server = Server::start("schema-type-label-modes");
    for (tag, mode) in [("warn", "warn"), ("strict", "strict")] {
        let context = format!("sake-{tag}");
        server.ok(
            "PUT",
            &format!("/contexts/{context}"),
            Some(json!({"description": "d"})),
        );
        let mut document = off_document();
        document["mode"] = json!(mode);
        server.ok(
            "PUT",
            &format!("/contexts/{context}/schema"),
            Some(document),
        );
        let (status, body) = server.call(
            "POST",
            &format!("/contexts/{context}/aliases"),
            Some(json!({"labels": {"型": "schema:type"}})),
        );
        assert_eq!(status, 400, "mode {mode}: {body}");
    }
}

/// ADR 0009 §6.3 exclusion 3: a type name asserted via `schema:type`
/// never surfaces as a lexical twin-audit candidate once a schema
/// exists — a schema-authoring question ("Organization" vs
/// "Organisation"), never a spelling-drift signal.
#[test]
fn type_name_concepts_are_excluded_from_the_vocabulary_twin_audit_once_a_schema_exists() {
    let server = Server::start("schema-type-label-twin-audit");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "山田商店", "label": "schema:type", "object": "Organization",
             "weight": 1.0, "source": "a.md"},
            {"subject": "山田商会", "label": "schema:type", "object": "Organisation",
             "weight": 1.0, "source": "a.md"},
        ])),
    );

    let has_pair = |audit: &serde_json::Value, a: &str, b: &str| {
        audit["lexical_concepts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|pair| {
                let names = [pair["a"].as_str().unwrap(), pair["b"].as_str().unwrap()];
                names.contains(&a) && names.contains(&b)
            })
    };

    let before = server.ok("POST", "/contexts/sake/vocabulary/audit", None);
    assert!(
        has_pair(&before, "Organization", "Organisation"),
        "guard 1 (no schema yet): a type-name concept is an ordinary twin candidate: \
         {before}"
    );

    server.ok("PUT", "/contexts/sake/schema", Some(off_document()));

    let after = server.ok("POST", "/contexts/sake/vocabulary/audit", None);
    assert!(
        !has_pair(&after, "Organization", "Organisation"),
        "once a schema exists, a type name must never be proposed as a spelling-drift \
         candidate: {after}"
    );
    // The exclusion is scoped to type names alone — 山田商店/山田商会
    // are ordinary subject-side concepts, near-duplicate spellings in
    // their own right, and must still survive the audit. Without this,
    // the assertion above would also pass on a bug that emptied the
    // whole audit rather than just excluding type names.
    assert!(
        has_pair(&after, "山田商店", "山田商会"),
        "an ordinary concept pair must still be audited: {after}"
    );
}
