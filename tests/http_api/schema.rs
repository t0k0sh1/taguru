//! `GET`/`PUT /contexts/{name}/schema` (#380, S2 of #218's ADR 0009
//! split): the round trip, the 404/not-installed distinction ADR 0009
//! §6.3 keeps load-bearing, `install`'s refusals surfacing as 400, the
//! `off` → `strict` transition working even over existing violations
//! (§7.1 — "strict means from now on"), and `schema_mode`/`revision`
//! echoing on the directory rows. Auth/scope classification lives in
//! `key_scopes_cross_context.rs`; retrieval-cache invalidation in
//! `retrieval_cache.rs`; replica refusal in `replication.rs`.

use serde_json::json;

use crate::support::*;

fn valid_document() -> serde_json::Value {
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

/// The core contract: no schema installed answers a dedicated 404
/// distinct from "no such context" (ADR 0009 §6.3's load-bearing
/// distinction), a `PUT` round-trips through `GET`, and `mode` is
/// exactly what was sent.
#[test]
fn schema_round_trips_and_distinguishes_not_installed_from_no_context() {
    let server = Server::start("schema-roundtrip");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    let (status, body) = server.call("GET", "/contexts/sake/schema", None);
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["code"], "no_schema", "{body}");
    assert!(
        body["error"].as_str().unwrap().contains("no schema"),
        "{body}"
    );

    let (status, body) = server.call("GET", "/contexts/nope/schema", None);
    assert_eq!(status, 404, "{body}");
    assert_eq!(
        body["code"], "no_context",
        "a missing CONTEXT must not be mistaken for a missing schema: {body}"
    );

    let installed = server.ok("PUT", "/contexts/sake/schema", Some(valid_document()));
    assert_eq!(installed, valid_document());

    let fetched = server.ok("GET", "/contexts/sake/schema", None);
    assert_eq!(fetched, valid_document());
}

/// `install`'s refusals — an unread version, an `is_a` cycle, the
/// reserved relation label — surface as 400 `invalid_argument`, not a
/// 500; `deny_unknown_fields` catches a stray field before `install`
/// even runs.
#[test]
fn install_refusals_answer_400() {
    let server = Server::start("schema-refusals");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    let mut unread_version = valid_document();
    unread_version["schema"] = json!(2);
    let (status, body) = server.call("PUT", "/contexts/sake/schema", Some(unread_version));
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], "invalid_argument", "{body}");

    let mut cycle = valid_document();
    cycle["types"] = json!({"A": {"is_a": ["B"]}, "B": {"is_a": ["A"]}});
    let (status, body) = server.call("PUT", "/contexts/sake/schema", Some(cycle));
    assert_eq!(status, 400, "{body}");
    assert!(body["error"].as_str().unwrap().contains("cycle"), "{body}");

    // ADR 0009 §6.3 guard 3: a relation definition literally named
    // `schema:type` is reserved for type assertions.
    let mut reserved = valid_document();
    reserved["relations"] = json!({"schema:type": {}});
    let (status, body) = server.call("PUT", "/contexts/sake/schema", Some(reserved));
    assert_eq!(status, 400, "{body}");
    assert!(
        body["error"].as_str().unwrap().contains("schema:type"),
        "{body}"
    );

    // deny_unknown_fields at the wire, before `install` runs at all —
    // axum's own JSON-rejection status (422, a well-formed body of the
    // wrong shape), not `install`'s 400.
    let mut unknown_field = valid_document();
    unknown_field["extra"] = json!(true);
    let (status, body) = server.call("PUT", "/contexts/sake/schema", Some(unknown_field));
    assert_eq!(status, 422, "{body}");
    assert_eq!(body["code"], "malformed_request", "{body}");

    let (status, _) = server.call("GET", "/contexts/sake/schema", None);
    assert_eq!(
        status, 404,
        "no refused PUT above may have installed anything"
    );
}

/// ADR 0009 §6.3 guard 3's migration-boundary counterpart: an
/// already-persisted `label_alias` resolving to the reserved type
/// label refuses the `PUT`, naming the alias.
#[test]
fn a_label_alias_resolving_to_the_reserved_type_label_refuses_the_put() {
    let server = Server::start("schema-reserved-alias");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    // Legal today — guard 1: `schema:type` is an ordinary label until a
    // schema exists — and interns the label id the alias resolves
    // against.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(
            json!([{"subject": "蔵", "label": "schema:type", "object": "Brewery",
                      "weight": 1.0, "source": "a.md"}]),
        ),
    );
    server.ok(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"labels": {"種別": "schema:type"}})),
    );

    let (status, body) = server.call("PUT", "/contexts/sake/schema", Some(valid_document()));
    assert_eq!(status, 400, "{body}");
    assert!(body["error"].as_str().unwrap().contains("種別"), "{body}");

    let (status, _) = server.call("GET", "/contexts/sake/schema", None);
    assert_eq!(status, 404, "a refused PUT must not install anything");
}

/// ADR 0009 §7.1: `PUT /schema` never inspects the graph — a mode
/// change to `strict` succeeds even with pre-existing edges that would
/// violate it, because `strict` means "from now on", not a retroactive
/// sweep. (Enforcement itself is #381's job; this pins the write path
/// never blocks on it.)
#[test]
fn switching_to_strict_succeeds_over_a_pre_existing_violation() {
    let server = Server::start("schema-strict-switch");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "PUT",
        "/contexts/sake/schema",
        Some(json!({
            "schema": 1, "mode": "off", "closed_labels": false,
            "types": {"Brewery": {}, "Person": {}},
            "relations": {"杜氏": {"domain": ["Brewery"], "range": ["Person"]}}
        })),
    );
    // A concept typed Person on the DOMAIN side — a future violation
    // under the relation declared above, were it enforced.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(
            json!([{"subject": "高瀬", "label": "杜氏", "object": "青嶺酒造",
                      "weight": 1.0, "source": "a.md"}]),
        ),
    );

    let mut strict = valid_document();
    strict["relations"] = json!({"杜氏": {"domain": ["Brewery"], "range": ["Person"]}});
    let (status, body) = server.call("PUT", "/contexts/sake/schema", Some(strict));
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        server.ok("GET", "/contexts/sake/schema", None)["mode"],
        "strict"
    );
}

/// `DirectoryEntry.schema_mode` (ADR 0009 §7.1): `null` before install,
/// echoing `mode` afterward, on both `GET /contexts` and
/// `GET /contexts/{name}` — and `revision.config` advances on the same
/// `PUT`, the same lane `bump_config_revision` already uses for
/// `dice_floor`.
#[test]
fn schema_mode_echoes_on_the_directory_and_the_put_bumps_config_revision() {
    let server = Server::start("schema-directory-echo");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    let single = server.ok("GET", "/contexts/sake", None);
    assert_eq!(single["schema_mode"], serde_json::Value::Null);
    let before_revision = single["revision"]["config"].as_u64().unwrap();

    let mut warn = valid_document();
    warn["mode"] = json!("warn");
    server.ok("PUT", "/contexts/sake/schema", Some(warn));

    let single = server.ok("GET", "/contexts/sake", None);
    assert_eq!(single["schema_mode"], "warn", "{single}");
    assert_eq!(
        single["revision"]["config"].as_u64().unwrap(),
        before_revision + 1
    );

    let page = server.ok("GET", "/contexts", None);
    let row = page["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["name"] == "sake")
        .unwrap();
    assert_eq!(row["schema_mode"], "warn", "{row}");
}

/// A repeated `PUT` of byte-identical content is a no-op — the same
/// idempotent-update discipline `PATCH /contexts/{name}` already keeps
/// for an unchanged value — so a retry never churns `revision.config`.
#[test]
fn a_repeated_put_of_the_same_document_does_not_bump_the_revision_again() {
    let server = Server::start("schema-idempotent-put");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok("PUT", "/contexts/sake/schema", Some(valid_document()));
    let revision = server.ok("GET", "/contexts/sake", None)["revision"]["config"]
        .as_u64()
        .unwrap();

    server.ok("PUT", "/contexts/sake/schema", Some(valid_document()));
    assert_eq!(
        server.ok("GET", "/contexts/sake", None)["revision"]["config"]
            .as_u64()
            .unwrap(),
        revision,
        "identical content must not bump the revision"
    );
}
