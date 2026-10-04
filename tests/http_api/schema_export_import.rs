//! The `schema` export/import stream record and its
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

fn schema_line(context_id: &str, mode: &str) -> String {
    format!(
        "{{\"type\": \"schema\", \"context_id\": \"{context_id}\", \"mode\": \"{mode}\", \
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
        "{{\"type\": \"source\", \"context_id\": \"cef2e28b-43f0-4b6c-8201-abab0785399f\", \"id\": \"a.md\", \
          \"create\": {{\"name\": \"sake\", \"description\": \"d\"}}}}\n\
         {schema_record}\
         {{\"type\": \"group\", \"id\": \"breweries\", \"context_ids\": [\"cef2e28b-43f0-4b6c-8201-abab0785399f\"]}}\n",
        schema_record = schema_line("cef2e28b-43f0-4b6c-8201-abab0785399f", "warn"),
    );
    let (status, outcome) = post_import(&server, &stream, None);
    assert_eq!(status, 200, "{outcome}");
    assert_eq!(
        outcome["result"]["schemas"][0]["context_id"], "cef2e28b-43f0-4b6c-8201-abab0785399f",
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

    let installed = server.ok(
        "GET",
        &format!("/contexts/{}/schema", server.cx("sake")),
        None,
    );
    assert_eq!(installed["mode"], "warn", "{installed}");
    assert_eq!(
        installed["types"],
        json!({"Brewery": {"is_a": []}}),
        "{installed}"
    );

    // A stream with no schema record at all keeps the response shape
    // byte-identical to before this feature — no `schemas` key.
    server.create_with_id(
        "a116c9ed-46d6-4077-b4a4-3317d30fd88f",
        json!({"name": "plain", }),
    );
    let (status, plain) = post_import(
        &server,
        "{\"type\": \"source\", \"context_id\": \"a116c9ed-46d6-4077-b4a4-3317d30fd88f\", \"id\": \"b.md\"}\n",
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

/// A context-scoped key's grant is checked against a schema record's
/// context the same way a batch's is — before anything applies. The
/// sake batch precedes the bunko schema record in the SAME stream; the
/// refusal must still land with nothing written, batch included.
#[test]
fn a_context_scoped_key_without_a_grant_on_the_schema_records_context_refuses_with_nothing_applied()
{
    let server = Server::start_with_env(
        "schema-stream-scope",
        &[
            ("TAGURU_API_TOKENS", "admin:atok,writer:wtok"),
            (
                "TAGURU_KEY_GRANTS",
                r#"{"writer": {"role": "admin", "contexts": ["cef2e28b-43f0-4b6c-8201-abab0785399f"]}}"#,
            ),
        ],
    );
    let create = |id: &str, name: &str| {
        let header = json!({
            "type": "source", "context_id": id, "id": "seed:create",
            "create": {"name": name, "description": "d"},
        });
        post_import(&server, &format!("{header}\n"), Some("atok"))
    };
    assert_eq!(
        create("cef2e28b-43f0-4b6c-8201-abab0785399f", "sake").0,
        200
    );
    assert_eq!(
        create("98a4dbfd-92a8-4c44-be61-6b69e8c34c26", "bunko").0,
        200
    );
    let stream = format!(
        "{{\"type\": \"source\", \"context_id\": \"cef2e28b-43f0-4b6c-8201-abab0785399f\", \"id\": \"a.md\"}}\n\
         {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n\
         {schema_record}",
        schema_record = schema_line("98a4dbfd-92a8-4c44-be61-6b69e8c34c26", "warn"),
    );
    let (status, refusal) = post_import(&server, &stream, Some("wtok"));
    assert_eq!(status, 403, "{refusal}");
    assert!(
        refusal["error"]
            .as_str()
            .unwrap()
            .contains("no grant on context '98a4dbfd-92a8-4c44-be61-6b69e8c34c26'"),
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
    let (status, sake) = server.call_with_token(
        "GET",
        &format!("/contexts/{}", server.cx("sake")),
        None,
        Some("atok"),
    );
    assert_eq!(status, 200, "{sake}");
    assert_eq!(sake["result"]["stats"]["associations"], 0, "{sake}");

    // The admin key with no grant entry carries the same stream through
    // cleanly.
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
        "{{\"type\": \"source\", \"context_id\": \"cef2e28b-43f0-4b6c-8201-abab0785399f\", \"id\": \"a.md\", \
          \"create\": {{\"name\": \"sake\", \"description\": \"d\"}}}}\n\
         {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n\
         {schema_record}",
        schema_record = schema_line("ead6ef03-d61e-460c-933d-6d450c50a1e5", "warn"),
    );
    let (status, refusal) = post_import(&server, &stream, None);
    assert_eq!(status, 404, "{refusal}");
    assert!(
        refusal["error"]
            .as_str()
            .unwrap()
            .contains("context 'ead6ef03-d61e-460c-933d-6d450c50a1e5'"),
        "{refusal}"
    );

    // The sake batch that preceded the refused schema record is
    // durable — retract-then-apply's own idempotence means re-POSTing
    // a corrected stream is exact, never double-counted.
    let sake = server.ok("GET", &format!("/contexts/{}", server.cx("sake")), None);
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
    server.create_with_id(
        "cef2e28b-43f0-4b6c-8201-abab0785399f",
        json!({"name": "sake", "description": "d"}),
    );
    let stream = format!(
        "{first}{second}",
        first = schema_line("cef2e28b-43f0-4b6c-8201-abab0785399f", "warn"),
        second = schema_line("ead6ef03-d61e-460c-933d-6d450c50a1e5", "warn"),
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
    let installed = server.ok(
        "GET",
        &format!("/contexts/{}/schema", server.cx("sake")),
        None,
    );
    assert_eq!(installed["mode"], "warn", "{installed}");
}

/// `taguru export --url` / `taguru import --url`: a schema installed
/// on the server rides the fetched stream as a `schema` record
/// and reinstalls on the other side — the CLI round trip
/// `remote_import.rs`'s own full-stream test proves for batches/
/// groups, extended to cover a schema.
#[test]
fn cli_export_and_import_url_round_trip_a_schema_record() {
    let source = Server::start("schema-cli-source");
    source.create_with_id(
        "cef2e28b-43f0-4b6c-8201-abab0785399f",
        json!({"name": "sake", "description": "酒蔵の知識"}),
    );
    source.ok(
        "PUT",
        &format!("/contexts/{}/schema", source.cx("sake")),
        Some(json!({
            "type": "schema",
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
    assert!(
        stream.find("\"type\":\"schema\"").unwrap()
            < stream.find("\"type\":\"source\"").unwrap_or(usize::MAX),
        "the schema record must ride first — {stream}"
    );

    let target = Server::start("schema-cli-target");
    target.create_with_id(
        "cef2e28b-43f0-4b6c-8201-abab0785399f",
        json!({"name": "sake", "description": "酒蔵の知識"}),
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

    let installed = target.ok(
        "GET",
        &format!("/contexts/{}/schema", target.cx("sake")),
        None,
    );
    assert_eq!(installed["mode"], "warn", "{installed}");
    assert_eq!(
        installed["types"],
        json!({"Brewery": {"is_a": []}}),
        "{installed}"
    );

    let _ = std::fs::remove_dir_all(&batches);
}
