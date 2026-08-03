//! `GET`/`PUT /contexts/{name}/schema` (#380, S2 of #218's ADR 0009
//! split): the round trip, the 404/not-installed distinction ADR 0009
//! §6.3 keeps load-bearing, `install`'s refusals surfacing as 400, the
//! `off` → `strict` transition working even over existing violations
//! (§7.1 — "strict means from now on"), and `schema_mode`/`revision`
//! echoing on the directory rows. `strict`/`warn` enforcement on
//! `POST /contexts/{name}/associations` itself (S5, #383) lives at the
//! bottom of this file. Auth/scope classification lives in
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
    // The write side of the same distinction: a PUT against a context
    // that does not exist at all is `no_context`, not `no_schema`.
    let (status, body) = server.call("PUT", "/contexts/nope/schema", Some(valid_document()));
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["code"], "no_context", "{body}");

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

// S5 (#383): `POST /contexts/{name}/associations` enforcing the
// installed schema's `domain`/`range` constraints — `strict`/`warn`/
// `off`, all sharing the one `schema_issues` pass S3 (#381) already
// unit-tests in isolation. `valid_document()`'s relation, 杜氏
// (domain [Brewery], range [Person]), is reused throughout: a subject
// typed `Person` violates the domain side, an object typed `Brewery`
// violates the range side. Typing happens via an ordinary
// `schema:type` association in the SAME batch as the fact it types —
// there is no separate typing call.

/// The `strict` half of ADR 0009 §7.3: a domain violation refuses the
/// whole batch before any write, `integrity: "nothing_written"`, the
/// same collect-all shape `errors::add_associations_collects_every_issue_in_one_pass`
/// already pins for shape-invalid batches.
#[test]
fn strict_refuses_a_domain_violation_before_any_write() {
    let server = Server::start("schema-strict-domain");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok("PUT", "/contexts/sake/schema", Some(valid_document()));

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "高瀬", "label": "schema:type", "object": "Person", "weight": 1.0, "source": "a.md"},
            {"subject": "高瀬", "label": "杜氏", "object": "個人A", "weight": 1.0, "source": "a.md"},
        ])),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("invalid_argument"), "{body}");
    assert_eq!(body["integrity"], json!("nothing_written"), "{body}");
    assert_eq!(body["retryable_after_correction"], json!(true), "{body}");
    let issues = body["issues"].as_array().expect("issues array");
    assert_eq!(issues.len(), 1, "{body}");
    assert_eq!(
        issues[0]["path"],
        json!("associations[1].subject"),
        "{body}"
    );
    assert_eq!(issues[0]["kind"], json!("domain"), "{body}");
    assert_eq!(issues[0]["actual"], json!("typed as [Person]"), "{body}");

    // Refused whole: neither the type assertion nor the fact landed.
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["contexts"][0]["stats"]["associations"], json!(0));
}

/// The range side of the same relation: an object typed `Brewery`
/// (disjoint from `杜氏`'s declared range, `[Person]`) is named on
/// `associations[{i}].object`, `kind: "range"`.
#[test]
fn strict_refuses_a_range_violation_and_names_the_object_path() {
    let server = Server::start("schema-strict-range");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok("PUT", "/contexts/sake/schema", Some(valid_document()));

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "schema:type", "object": "Brewery", "weight": 1.0, "source": "a.md"},
            {"subject": "高瀬", "label": "schema:type", "object": "Brewery", "weight": 1.0, "source": "a.md"},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "a.md"},
        ])),
    );
    assert_eq!(status, 400, "{body}");
    let issues = body["issues"].as_array().expect("issues array");
    assert_eq!(issues.len(), 1, "{body}");
    assert_eq!(issues[0]["path"], json!("associations[2].object"), "{body}");
    assert_eq!(issues[0]["kind"], json!("range"), "{body}");
}

/// The `warn` half of §8.3: the same violating batch writes anyway,
/// and the identical `Issue` a `strict` context would have refused
/// with instead rides out in the success envelope — `issues` plus a
/// `schema_violations` count surviving `MAX_LISTED_ISSUES` truncation
/// (not exercised here; one issue is far under the cap).
#[test]
fn warn_writes_and_reports_the_same_issues_a_strict_context_would_refuse() {
    let server = Server::start("schema-warn-writes");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    let mut warn = valid_document();
    warn["mode"] = json!("warn");
    server.ok("PUT", "/contexts/sake/schema", Some(warn));

    let batch = json!([
        {"subject": "高瀬", "label": "schema:type", "object": "Person", "weight": 1.0, "source": "a.md"},
        {"subject": "高瀬", "label": "杜氏", "object": "個人A", "weight": 1.0, "source": "a.md"},
    ]);
    let (status, body) = server.call("POST", "/contexts/sake/associations", Some(batch));
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["result"],
        json!(2),
        "both ops in the batch applied: {body}"
    );
    assert_eq!(body["schema_violations"], json!(1), "{body}");
    let issues = body["issues"].as_array().expect("issues array");
    assert_eq!(issues.len(), 1, "{body}");
    assert_eq!(
        issues[0]["path"],
        json!("associations[1].subject"),
        "{body}"
    );
    assert_eq!(issues[0]["kind"], json!("domain"), "{body}");

    // The write actually went ahead, unlike `strict`'s refusal above.
    let directory = server.ok("GET", "/contexts", None);
    assert_eq!(directory["contexts"][0]["stats"]["associations"], json!(2));
}

/// `off` (the default before any `PUT /schema`, and a legal explicit
/// choice) never populates `issues` — the field is
/// `skip_serializing_if = "Vec::is_empty"`, so a response with nothing
/// to say omits it entirely rather than sending an empty array.
#[test]
fn off_mode_never_reports_issues_even_over_the_same_violation() {
    let server = Server::start("schema-off-silent");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    let mut off = valid_document();
    off["mode"] = json!("off");
    server.ok("PUT", "/contexts/sake/schema", Some(off));

    let batch = json!([
        {"subject": "高瀬", "label": "schema:type", "object": "Person", "weight": 1.0, "source": "a.md"},
        {"subject": "高瀬", "label": "杜氏", "object": "個人A", "weight": 1.0, "source": "a.md"},
    ]);
    let (status, body) = server.call("POST", "/contexts/sake/associations", Some(batch));
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"], json!(2), "{body}");
    assert!(
        !body.as_object().unwrap().contains_key("issues"),
        "off must not carry an issues key at all: {body}"
    );
}

/// ADR 0009 §7.2's ordering guarantee: `TypeEnv` is built in full
/// before any op is judged, so a fact op naming its own subject's type
/// LATER in the same batch validates exactly like the reverse order —
/// line order inside a batch cannot matter.
#[test]
fn a_batch_typing_its_own_subject_after_the_fact_op_still_validates() {
    let server = Server::start("schema-strict-order");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok("PUT", "/contexts/sake/schema", Some(valid_document()));

    let batch = json!([
        {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "a.md"},
        {"subject": "青嶺酒造", "label": "schema:type", "object": "Brewery", "weight": 1.0, "source": "a.md"},
        {"subject": "高瀬", "label": "schema:type", "object": "Person", "weight": 1.0, "source": "a.md"},
    ]);
    let (status, body) = server.call("POST", "/contexts/sake/associations", Some(batch));
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["result"], json!(3), "{body}");
    assert!(!body.as_object().unwrap().contains_key("issues"), "{body}");
}

/// ADR 0009 §6.4: `closed_labels` refuses a fact op whose label has no
/// relation entry at all, distinct from a domain/range mismatch on a
/// declared one. The `schema:type` op itself is exempt by construction
/// — it never resolves to anything `closed_labels` walks.
#[test]
fn closed_labels_refuses_an_undeclared_label() {
    let server = Server::start("schema-closed-labels");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok(
        "PUT",
        "/contexts/sake/schema",
        Some(json!({
            "schema": 1,
            "mode": "strict",
            "closed_labels": true,
            "types": {},
            "relations": {"杜氏": {}}
        })),
    );

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "高瀬", "label": "未知の関係", "object": "個人A", "weight": 1.0, "source": "a.md"},
        ])),
    );
    assert_eq!(status, 400, "{body}");
    let issues = body["issues"].as_array().expect("issues array");
    assert_eq!(issues.len(), 1, "{body}");
    assert_eq!(issues[0]["path"], json!("associations[0].label"), "{body}");
    assert_eq!(issues[0]["kind"], json!("unknown_reference"), "{body}");
}
