//! `strict`/`warn` on `POST /import` (#382, S4 of #218's ADR 0009
//! split, §7.2/§8.2/§8.3): `predicted_schema_rejection` wired into
//! `apply_batch`/`preview_batch` as the schema twin of
//! `predicted_alias_rejection`. `off`/no-schema byte-identical
//! behavior, `strict` refusing before anything mutates (400, a
//! `batches[{b}].associations[{a}]...`-addressed `Issue`,
//! `nothing_written`/`durable_prefix`), `warn` proceeding with the
//! same `Issue` values riding the success envelope's `issues`, and the
//! reserved `schema:type` label's batch-local guard (409, mode-
//! independent). The associations endpoint's own pre-write arm is
//! S5/#383; `src/schema/check.rs` unit-tests `schema_issues` itself.

use serde_json::json;

use crate::support::*;

/// One relation (`杜氏`: domain `Brewery`, range `Person`) — enough to
/// exercise a domain violation without needing the range side too.
fn document(mode: &str) -> serde_json::Value {
    json!({
        "schema": 1,
        "mode": mode,
        "closed_labels": false,
        "types": {
            "Brewery": {"is_a": []},
            "Person": {"is_a": []}
        },
        "relations": {
            "杜氏": {"domain": ["Brewery"], "range": ["Person"]}
        }
    })
}

/// A domain violation: `田中` typed `Person` (disjoint from `杜氏`'s
/// declared `domain: [Brewery]`), asserted in the same batch as the
/// fact that types it — ADR 0009 §7.2's union-before-judgment, so line
/// order inside the batch does not matter. `青嶺酒造` (the object) is
/// never typed, so only the subject side violates (§6.1: untyped never
/// violates).
fn domain_violation_batch() -> String {
    "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\"}\n\
     {\"subject\": \"田中\", \"label\": \"schema:type\", \"object\": \"Person\", \"weight\": 1.0}\n\
     {\"subject\": \"田中\", \"label\": \"杜氏\", \"object\": \"青嶺酒造\", \"weight\": 1.0}\n"
        .to_string()
}

fn seed(server: &Server, mode: &str) {
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok("PUT", "/contexts/sake/schema", Some(document(mode)));
}

fn associations_count(server: &Server) -> serde_json::Value {
    server.ok("GET", "/contexts/sake", None)["stats"]["associations"].clone()
}

/// `strict`: the domain violation refuses the whole batch before
/// anything mutates — 400 `invalid_argument`, a path-addressed
/// `domain` issue, `nothing_written`, and (unlike a bare error message)
/// the context's stats prove nothing landed.
#[test]
fn strict_domain_violation_refuses_and_writes_nothing() {
    let server = Server::start("schema-import-strict-domain");
    seed(&server, "strict");

    let (status, body) = post_import(&server, &domain_violation_batch(), None);
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], "invalid_argument", "{body}");
    assert_eq!(
        body["issues"][0]["path"],
        json!("batches[0].associations[1].subject"),
        "{body}"
    );
    assert_eq!(body["issues"][0]["kind"], json!("domain"), "{body}");
    assert_eq!(body["integrity"], json!("nothing_written"), "{body}");

    assert_eq!(
        associations_count(&server),
        json!(0),
        "a strict schema refusal must not write the type assertion or the fact either"
    );
}

/// Same violation, `?dry_run=true`: the preview refuses identically —
/// `preview_batch` and `apply_batch` share `predicted_schema_rejection`
/// so the two entrances cannot disagree.
#[test]
fn strict_domain_violation_dry_run_refuses_the_same_way() {
    let server = Server::start("schema-import-strict-dry-run");
    seed(&server, "strict");

    let (status, body) = post_import_dry_run(&server, &domain_violation_batch(), None);
    assert_eq!(status, 400, "{body}");
    assert_eq!(
        body["issues"][0]["path"],
        json!("batches[0].associations[1].subject"),
        "{body}"
    );
    assert_eq!(associations_count(&server), json!(0), "{body}");
}

/// A `strict` refusal on the second batch of a stream leaves the first
/// batch's write standing — `durable_prefix`, naming exactly how many
/// batches landed (ADR 0009 §8.2).
#[test]
fn strict_domain_violation_on_a_later_batch_reports_a_durable_prefix() {
    let server = Server::start("schema-import-strict-prefix");
    seed(&server, "strict");

    let clean_batch = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"clean.md\"}\n\
                        {\"subject\": \"alpha\", \"label\": \"connects_to\", \"object\": \"beta\", \
                        \"weight\": 1.0}\n";
    let stream = format!("{clean_batch}{}", domain_violation_batch());

    let (status, body) = post_import(&server, &stream, None);
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["integrity"], json!("durable_prefix"), "{body}");
    assert_eq!(body["durable_batches"], json!(1), "{body}");
    assert_eq!(
        body["issues"][0]["path"],
        json!("batches[1].associations[1].subject"),
        "the second batch's own index, not the first: {body}"
    );
    assert_eq!(
        associations_count(&server),
        json!(1),
        "the first (clean) batch's association must survive the second batch's refusal"
    );
}

/// `warn`: the batch is applied — `schema_violations` counts the
/// violation, and the success envelope's `issues` carries the exact
/// same `Issue` (ADR 0009 §8.3: "`Issue` values are byte-identical
/// between `warn` and `strict` — only the HTTP status differs").
#[test]
fn warn_domain_violation_applies_and_reports_the_same_issue() {
    let server = Server::start("schema-import-warn");
    seed(&server, "warn");

    let (status, body) = post_import(&server, &domain_violation_batch(), None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["result"]["batches"][0]["schema_violations"],
        json!(1),
        "{body}"
    );
    assert_eq!(
        body["issues"][0]["path"],
        json!("batches[0].associations[1].subject"),
        "{body}"
    );
    assert_eq!(body["issues"][0]["kind"], json!("domain"), "{body}");

    // Unlike `strict`, the batch actually landed: both the type
    // assertion and the fact.
    assert_eq!(associations_count(&server), json!(2), "{body}");
}

/// `warn`, `?dry_run=true`: the preview reports the same violation
/// count without writing anything.
#[test]
fn warn_domain_violation_dry_run_previews_the_count() {
    let server = Server::start("schema-import-warn-dry-run");
    seed(&server, "warn");

    let (status, body) = post_import_dry_run(&server, &domain_violation_batch(), None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["result"]["batches"][0]["schema_violations"],
        json!(1),
        "{body}"
    );
    assert_eq!(associations_count(&server), json!(0), "{body}");
}

/// `off` (and, by construction, no schema at all): the response is
/// byte-identical to today's — no `issues` field rides the envelope at
/// all, and `schema_violations` is 0, even though the same batch would
/// violate the schema under `warn`/`strict`.
#[test]
fn off_mode_never_reports_a_violation_and_omits_issues_entirely() {
    let server = Server::start("schema-import-off");
    seed(&server, "off");

    let (status, body) = post_import(&server, &domain_violation_batch(), None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["result"]["batches"][0]["schema_violations"],
        json!(0),
        "{body}"
    );
    assert!(
        body.as_object().unwrap().get("issues").is_none(),
        "`issues` must be entirely absent, not an empty array: {body}"
    );
}

/// ADR 0009 §7.2's ordering guarantee: a fact op typed later in the
/// SAME batch validates identically to one typed earlier — `TypeEnv` is
/// built in full before any op is judged.
#[test]
fn a_type_declared_after_the_fact_it_types_still_validates() {
    let server = Server::start("schema-import-order");
    seed(&server, "strict");

    let batch = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\"}\n\
                 {\"subject\": \"田中\", \"label\": \"杜氏\", \"object\": \"青嶺酒造\", \
                 \"weight\": 1.0}\n\
                 {\"subject\": \"田中\", \"label\": \"schema:type\", \"object\": \"Brewery\", \
                 \"weight\": 1.0}\n";
    let (status, body) = post_import(&server, batch, None);
    assert_eq!(status, 200, "{body}");
    assert_eq!(associations_count(&server), json!(2), "{body}");
}

/// ADR 0009 §6.3 guard 2's batch-local bullet: this batch's own
/// `labels` alias resolving to the reserved `schema:type` refuses —
/// mode-independent, so `off` refuses it exactly like `warn`/`strict`
/// would (the gate is "a schema document exists," never "mode != off").
/// Distinct status/kind from a domain violation: a namespace conflict
/// (409 `conflict`), not a refused value.
#[test]
fn a_batch_label_alias_resolving_to_the_reserved_label_refuses_in_every_mode() {
    let server = Server::start("schema-import-reserved");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    // Interns `schema:type` as an ordinary label before any schema
    // exists (guard 1) — the same precedent
    // `schema.rs::a_label_alias_resolving_to_the_reserved_type_label_refuses_the_put`
    // uses for `PUT /schema` itself.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(
            json!([{"subject": "某社", "label": "schema:type", "object": "Brewery",
                      "weight": 1.0, "source": "pre.md"}]),
        ),
    );
    server.ok("PUT", "/contexts/sake/schema", Some(document("off")));

    let batch = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\"}\n\
                 {\"alias\": \"種別\", \"canonical\": \"schema:type\", \"kind\": \"label\"}\n";
    let (status, body) = post_import(&server, batch, None);
    assert_eq!(status, 409, "{body}");
    assert_eq!(body["code"], "conflict", "{body}");
    assert_eq!(
        body["issues"][0]["path"],
        json!("batches[0].labels['種別']"),
        "{body}"
    );
    assert_eq!(body["integrity"], json!("nothing_written"), "{body}");
    assert_eq!(
        associations_count(&server),
        json!(1),
        "only the pre-seeded association — the refused batch wrote nothing: {body}"
    );
}
