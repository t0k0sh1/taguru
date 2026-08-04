//! The `taguru_schema` export/import stream record and its
//! replication parity (#384, S6 of #218's ADR 0009 split, §13):
//! `POST /import` installing a schema record after every batch and
//! before any group, its context-scope check, its response shape, and
//! the CLI `--url` round trip. `strict`/`warn` enforcement on the
//! associations a batch itself carries is #382/#383's own file
//! (`schema_import.rs`); `GET`/`PUT /contexts/{name}/schema` itself is
//! `schema.rs`; replica tailing and ship→restore fidelity are
//! `replication.rs`.

use serde_json::json;

use crate::support::*;

fn schema_line(context: &str, mode: &str) -> String {
    format!(
        "{{\"taguru_schema\": 1, \"context\": \"{context}\", \"mode\": \"{mode}\", \
         \"closed_labels\": false, \"types\": {{\"Brewery\": {{}}}}, \
         \"relations\": {{\"杜氏\": {{\"domain\": [\"Brewery\"], \"range\": []}}}}}}\n"
    )
}

/// A schema record installs after every batch, before any group — a
/// record naming a context a batch of the SAME stream just created,
/// riding alongside a group naming that context too. The response
/// carries `schemas` (context/mode/types/relations, no outcome verb —
/// `put_schema` cannot itself tell an install from a no-op), and the
/// installed document is retrievable afterward exactly as `PUT
/// /schema` would have left it.
#[test]
fn a_schema_record_installs_after_batches_before_groups_and_the_response_names_it() {
    let server = Server::start("schema-stream-install");
    let stream = format!(
        "{{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
          \"create\": {{\"description\": \"d\"}}}}\n\
         {schema_record}\
         {{\"taguru_group\": 1, \"name\": \"breweries\", \"contexts\": [\"sake\"]}}\n",
        schema_record = schema_line("sake", "warn"),
    );
    let (status, outcome) = post_import(&server, &stream, None);
    assert_eq!(status, 200, "{outcome}");
    assert_eq!(
        outcome["result"]["schemas"][0]["context"], "sake",
        "{outcome}"
    );
    assert_eq!(outcome["result"]["schemas"][0]["mode"], "warn", "{outcome}");
    assert_eq!(outcome["result"]["schemas"][0]["types"], 1, "{outcome}");
    assert_eq!(outcome["result"]["schemas"][0]["relations"], 1, "{outcome}");
    // The group record (installed AFTER the schema) still landed —
    // proves ordering never blocked it.
    assert_eq!(
        outcome["result"]["groups"][0]["name"], "breweries",
        "{outcome}"
    );

    let installed = server.ok("GET", "/contexts/sake/schema", None);
    assert_eq!(installed["mode"], "warn", "{installed}");
    assert_eq!(
        installed["types"],
        json!({"Brewery": {"is_a": []}}),
        "{installed}"
    );

    // A stream with no schema record at all keeps the response shape
    // byte-identical to before this feature — no `schemas` key.
    server.ok("PUT", "/contexts/plain", Some(json!({})));
    let (status, plain) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"plain\", \"source\": \"b.md\"}\n",
        None,
    );
    assert_eq!(status, 200, "{plain}");
    assert!(
        plain["result"]
            .as_object()
            .unwrap()
            .get("schemas")
            .is_none(),
        "{plain}"
    );
}

/// A scoped key's grant is checked against a schema record's context
/// the same way a batch's is — before anything applies. The sake
/// batch precedes the bunko schema record in the SAME stream; the
/// refusal must still land with nothing written, batch included.
#[test]
fn a_scoped_key_without_a_grant_on_the_schema_records_context_refuses_with_nothing_applied() {
    let server = Server::start_with_env(
        "schema-stream-scope",
        &[
            ("TAGURU_API_TOKENS", "admin:atok,writer:wtok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"writer": {"role": "admin", "contexts": ["sake"]}}"#,
            ),
        ],
    );
    let put = |path: &str| {
        server.call_with_token("PUT", path, Some(json!({"description": "d"})), Some("atok"))
    };
    assert_eq!(put("/contexts/sake").0, 200);
    assert_eq!(put("/contexts/bunko").0, 200);
    let stream = format!(
        "{{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\"}}\n\
         {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n\
         {schema_record}",
        schema_record = schema_line("bunko", "warn"),
    );
    let (status, refusal) = post_import(&server, &stream, Some("wtok"));
    assert_eq!(status, 403, "{refusal}");
    assert!(
        refusal["error"]
            .as_str()
            .unwrap()
            .contains("no grant on context 'bunko'"),
        "{refusal}"
    );
    assert!(
        refusal["error"]
            .as_str()
            .unwrap()
            .contains("nothing was applied"),
        "{refusal}"
    );

    // Nothing applied: sake's association from the SAME stream must
    // not have landed either, even though it appears before the
    // refused schema record.
    let (status, sake) = server.call_with_token("GET", "/contexts/sake", None, Some("atok"));
    assert_eq!(status, 200, "{sake}");
    assert_eq!(sake["result"]["stats"]["associations"], 0, "{sake}");

    // The unscoped admin key carries the same stream through cleanly.
    let (status, applied) = post_import(&server, &stream, Some("atok"));
    assert_eq!(status, 200, "{applied}");
}

/// A schema record naming a context that does not exist (and was
/// never created by an earlier batch of the same stream) refuses —
/// but everything before it in the stream is already durable, exactly
/// as a batch's own mid-stream refusal leaves its predecessors.
#[test]
fn a_schema_records_nonexistent_context_refuses_naming_it_with_earlier_batches_durable() {
    let server = Server::start("schema-stream-no-context");
    let stream = format!(
        "{{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
          \"create\": {{\"description\": \"d\"}}}}\n\
         {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n\
         {schema_record}",
        schema_record = schema_line("ghost", "warn"),
    );
    let (status, refusal) = post_import(&server, &stream, None);
    assert_eq!(status, 404, "{refusal}");
    assert!(
        refusal["error"]
            .as_str()
            .unwrap()
            .contains("context 'ghost'"),
        "{refusal}"
    );

    // The sake batch that preceded the refused schema record is
    // durable — retract-then-apply's own idempotence means re-POSTing
    // a corrected stream is exact, never double-counted.
    let sake = server.ok("GET", "/contexts/sake", None);
    assert_eq!(sake["stats"]["associations"], 1, "{sake}");
}

/// A stream carrying ONLY schema records (zero batches): the first
/// installs durably, the second (naming a nonexistent context) fails.
/// `put_schema` is atomic and independent per record, so the first
/// one's install already survives the second's refusal — `integrity`
/// must say `durable_prefix`, not `nothing_written`, even though
/// `durable_batches` (a batch count, not a schema count) stays absent.
/// Regression for a partial-application accounting gap: computing
/// `integrity` from the batch count alone would call this
/// `nothing_written` despite a schema already being durably persisted.
#[test]
fn an_earlier_schema_records_own_durability_survives_a_later_schemas_refusal() {
    let server = Server::start("schema-stream-partial-integrity");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    let stream = format!(
        "{first}{second}",
        first = schema_line("sake", "warn"),
        second = schema_line("ghost", "warn"),
    );
    let (status, refusal) = post_import(&server, &stream, None);
    assert_eq!(status, 404, "{refusal}");
    assert_eq!(refusal["integrity"], "durable_prefix", "{refusal}");
    assert!(
        refusal.get("durable_batches").is_none(),
        "durable_batches names batches, not schemas — none landed: {refusal}"
    );

    // The first schema record's install is durable — proof the
    // refusal above did not, and could not, roll it back.
    let installed = server.ok("GET", "/contexts/sake/schema", None);
    assert_eq!(installed["mode"], "warn", "{installed}");
}

/// `taguru export --url` / `taguru import --url`: a schema installed
/// on the server rides the fetched stream as a `taguru_schema` record
/// and reinstalls on the other side — the CLI round trip
/// `remote_import.rs`'s own full-stream test proves for batches/
/// groups, extended to cover a schema.
#[test]
fn cli_export_and_import_url_round_trip_a_schema_record() {
    let source = Server::start("schema-cli-source");
    source.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "酒蔵の知識"})),
    );
    source.ok(
        "PUT",
        "/contexts/sake/schema",
        Some(json!({
            "schema": 1,
            "mode": "warn",
            "closed_labels": false,
            "types": {"Brewery": {"is_a": []}},
            "relations": {"杜氏": {"domain": ["Brewery"], "range": []}},
        })),
    );

    let batches = batch_dir("schema-cli-roundtrip");
    let out = batches.join("out");
    let (code, _stdout, stderr) = run_cli(
        &[
            "export",
            "--url",
            &source.base,
            "--out",
            out.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "{stderr}");
    let stream = std::fs::read_to_string(out.join("sake.jsonl")).expect("sake.jsonl must exist");
    assert!(stream.contains("\"taguru_schema\":1"), "{stream}");
    assert!(
        stream.find("taguru_schema").unwrap() < stream.find("taguru_batch").unwrap_or(usize::MAX),
        "the schema record must ride first — {stream}"
    );

    let target = Server::start("schema-cli-target");
    target.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "酒蔵の知識"})),
    );
    let (code, stdout, stderr) = run_cli(
        &[
            "import",
            "--url",
            &target.base,
            out.join("sake.jsonl").to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let installed = target.ok("GET", "/contexts/sake/schema", None);
    assert_eq!(installed["mode"], "warn", "{installed}");
    assert_eq!(
        installed["types"],
        json!({"Brewery": {"is_a": []}}),
        "{installed}"
    );

    let _ = std::fs::remove_dir_all(&batches);
}
