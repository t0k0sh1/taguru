//! Graph-path promotion (ADR 0018), end to end: named scratch sources
//! move whole — passage, date, tags, only their share of every edge —
//! into an established destination, idempotently, with the landing
//! zone's audit riding back; previews write nothing; refusals name
//! their cause before anything applies.

use serde_json::json;

use crate::support::*;

/// Scratch under the runbook conventions: two promotable sessions, one
/// that stays behind, corroboration crossing the boundary, and one
/// alias per fate (carried / left behind).
fn seed(server: &Server) {
    server.ok(
        "PUT",
        "/contexts/scratch-claude",
        Some(json!({"description": "session notes"})),
    );
    server.ok(
        "PUT",
        "/contexts/perm",
        Some(json!({"description": "permanent"})),
    );
    server.ok(
        "POST",
        "/contexts/scratch-claude/associations",
        Some(json!([
            {"subject": "DB", "label": "採用", "object": "PostgreSQL 16", "weight": 1.0,
             "source": "session:claude:a/note", "paragraph": 0},
            // Corroborated across the promotion boundary: only the
            // promoted source's share may travel.
            {"subject": "CI", "label": "テストランナー", "object": "cargo-nextest", "weight": 1.0,
             "source": "session:claude:b"},
            {"subject": "CI", "label": "テストランナー", "object": "cargo-nextest", "weight": 1.0,
             "source": "session:claude:stay"},
            {"subject": "旧鍵", "label": "管理者", "object": "山科", "weight": 1.0,
             "source": "session:claude:stay"},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/scratch-claude/sources",
        Some(json!({
            "passages": {"session:claude:a/note": "DB は PostgreSQL 16。"},
            "dates": {"session:claude:a/note": 1785974400},
            "tags": {"session:claude:a/note": ["infra", "決定"]}
        })),
    );
    server.ok(
        "POST",
        "/contexts/scratch-claude/aliases",
        Some(json!({"concepts": {
            "Postgres": "PostgreSQL 16",
            "従来鍵": "旧鍵"
        }})),
    );
}

#[test]
fn promotion_moves_named_sources_whole_and_is_idempotent() {
    let server = Server::start("promote-moves");
    seed(&server);

    let promoted = server.ok(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(json!({
            "into": "perm",
            "sources": ["session:claude:a/note", "session:claude:b"]
        })),
    );
    // One batch per source, source-id order, the import outcome shape.
    let batches = promoted["batches"].as_array().expect("batches");
    assert_eq!(batches.len(), 2, "{promoted}");
    assert_eq!(batches[0]["source"], json!("session:claude:a/note"));
    assert_eq!(batches[1]["source"], json!("session:claude:b"));
    assert_eq!(batches[0]["context"], json!("perm"));
    assert_eq!(
        batches[0]["created"],
        json!(false),
        "promotion must never create the destination: {promoted}"
    );
    assert_eq!(batches[0]["passage_stored"], json!(true));
    // The 旧鍵 alias's canonical has no live edge in the promoted
    // slice — left behind, and counted rather than silent.
    assert_eq!(promoted["aliases_dropped"], json!(1), "{promoted}");
    // The landing zone's audit rides back by default, all sections.
    assert_eq!(promoted["audit"]["detector"], json!("consolidation/1"));
    assert!(
        promoted["audit"]["merge"].is_object()
            && promoted["audit"]["contradiction"].is_object()
            && promoted["audit"]["staleness"].is_object(),
        "all three sections run by default: {promoted}"
    );

    // Provenance travels whole: the passage with its date and tags,
    // under the SAME session source id.
    let looked_up = server.ok(
        "POST",
        "/contexts/perm/sources/lookup",
        Some(json!({"sources": ["session:claude:a/note"]})),
    );
    assert_eq!(
        looked_up["passages"]["session:claude:a/note"],
        json!("DB は PostgreSQL 16。"),
        "{looked_up}"
    );
    let sources = server.ok("GET", "/contexts/perm/sources", None);
    let entry = &sources["entries"][0];
    assert_eq!(entry["name"], json!("session:claude:a/note"), "{sources}");
    assert_eq!(entry["date"], json!(1785974400), "{sources}");
    assert_eq!(entry["tags"], json!(["infra", "決定"]), "{sources}");

    // Only the promoted source's share of a corroborated edge travels:
    // the edge exists in perm attributed to session b alone, while the
    // stay-behind source's share and its own fact never left scratch.
    let runner = server.ok(
        "POST",
        "/contexts/perm/query",
        Some(json!({"subject": "CI", "label": "テストランナー"})),
    );
    assert_eq!(runner["matches"][0]["count"], json!(1), "{runner}");
    let attributions = runner["matches"][0]["attributions"]
        .as_array()
        .expect("attributions");
    assert_eq!(attributions.len(), 1, "{runner}");
    assert_eq!(attributions[0]["source"], json!("session:claude:b"));
    let stayed = server.ok(
        "POST",
        "/contexts/perm/query",
        Some(json!({"subject": "旧鍵"})),
    );
    assert_eq!(stayed["total"], json!(0), "{stayed}");
    // The carried alias resolves in perm; the paragraph locator rides.
    let db = server.ok(
        "POST",
        "/contexts/perm/query",
        Some(json!({"subject": "DB", "label": "採用"})),
    );
    assert_eq!(db["matches"][0]["attributions"][0]["paragraph"], json!(0));
    let resolved = server.ok(
        "POST",
        "/contexts/perm/resolve",
        Some(json!({"cue": "Postgres"})),
    );
    assert_eq!(
        resolved[0]["name"],
        json!("PostgreSQL 16"),
        "the live-canonical alias must be carried: {resolved}"
    );
    assert_eq!(resolved[0]["kind"], json!("alias"), "{resolved}");

    // The scratch is untouched — retiring it stays an explicit step.
    let scratch = server.ok(
        "POST",
        "/contexts/scratch-claude/query",
        Some(json!({"subject": "旧鍵"})),
    );
    assert_eq!(scratch["total"], json!(1), "{scratch}");

    // Re-promoting the same sources is retract-then-apply: weights do
    // not double, and the outcome says what was replaced.
    let again = server.ok(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(json!({
            "into": "perm",
            "sources": ["session:claude:a/note", "session:claude:b"],
            "audit": false
        })),
    );
    assert!(
        again["batches"][1]["retracted"].as_u64().unwrap() > 0,
        "{again}"
    );
    assert!(
        again.get("audit").is_none(),
        "audit: false opts out: {again}"
    );
    let runner = server.ok(
        "POST",
        "/contexts/perm/query",
        Some(json!({"subject": "CI", "label": "テストランナー"})),
    );
    assert_eq!(runner["matches"][0]["count"], json!(1), "{runner}");
    assert_eq!(runner["matches"][0]["weight"], json!(1.0), "{runner}");
}

#[test]
fn a_dry_run_previews_the_same_shape_and_writes_nothing() {
    let server = Server::start("promote-dry-run");
    seed(&server);

    let preview = server.ok(
        "POST",
        "/contexts/scratch-claude/promote?dry_run=true",
        Some(json!({
            "into": "perm",
            "sources": ["session:claude:a/note", "session:claude:b"]
        })),
    );
    assert_eq!(preview["batches"].as_array().map(Vec::len), Some(2));
    assert!(
        preview.get("audit").is_none() && preview.get("audit_skipped").is_none(),
        "nothing landed, so there is nothing to audit: {preview}"
    );

    let sources = server.ok("GET", "/contexts/perm/sources", None);
    assert_eq!(sources["total"], json!(0), "{sources}");
    let db = server.ok(
        "POST",
        "/contexts/perm/query",
        Some(json!({"subject": "DB"})),
    );
    assert_eq!(db["total"], json!(0), "a preview must write nothing: {db}");
}

#[test]
fn promotion_refusals_name_their_cause_before_anything_applies() {
    let server = Server::start("promote-refusals");
    seed(&server);

    // No sources: choosing the keepers is the caller's judgment.
    let (status, refused) = server.call(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(json!({"into": "perm", "sources": []})),
    );
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("invalid_argument"), "{refused}");

    // Promoting into itself moves nothing anywhere.
    let (status, refused) = server.call(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(json!({"into": "scratch-claude", "sources": ["session:claude:b"]})),
    );
    assert_eq!(status, 400, "{refused}");
    assert!(
        refused["error"].as_str().unwrap().contains("DIFFERENT"),
        "{refused}"
    );

    // A missing destination refuses — promote never creates one.
    let (status, refused) = server.call(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(json!({"into": "nope", "sources": ["session:claude:b"]})),
    );
    assert_eq!(status, 404, "{refused}");
    assert_eq!(refused["code"], json!("no_context"), "{refused}");
    assert_eq!(refused["integrity"], json!("nothing_written"), "{refused}");

    // A mistyped source id refuses whole, path-addressed — under
    // retract-then-apply it would otherwise no-op silently.
    let (status, refused) = server.call(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(json!({
            "into": "perm",
            "sources": ["session:claude:b", "session:claude:typo"]
        })),
    );
    assert_eq!(status, 404, "{refused}");
    assert_eq!(refused["code"], json!("no_source"), "{refused}");
    assert_eq!(
        refused["issues"][0]["path"],
        json!("sources[1]"),
        "{refused}"
    );
    assert_eq!(refused["integrity"], json!("nothing_written"), "{refused}");
    let db = server.ok(
        "POST",
        "/contexts/perm/query",
        Some(json!({"subject": "DB"})),
    );
    assert_eq!(db["total"], json!(0), "nothing may have applied: {db}");
}

#[test]
fn the_destination_schema_judges_the_promoted_batches() {
    let server = Server::start("promote-schema");
    seed(&server);
    // A strict, closed destination that never declared 採用: the
    // scratch's a/note batch must refuse whole, exactly as an import.
    server.ok(
        "PUT",
        "/contexts/perm/schema",
        Some(json!({
            "schema": 1, "mode": "strict", "closed_labels": true,
            "types": {},
            "relations": {"テストランナー": {"domain": [], "range": []}}
        })),
    );

    let (status, refused) = server.call(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(json!({"into": "perm", "sources": ["session:claude:a/note"]})),
    );
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["integrity"], json!("nothing_written"), "{refused}");
    let db = server.ok(
        "POST",
        "/contexts/perm/query",
        Some(json!({"subject": "DB"})),
    );
    assert_eq!(
        db["total"],
        json!(0),
        "strict refuses the whole batch: {db}"
    );

    // The scratch's OWN (absent) schema never installs into the
    // destination: the strict document above must survive a promotion
    // of the schema-clean source.
    server.ok(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(json!({"into": "perm", "sources": ["session:claude:b"], "audit": false})),
    );
    let schema = server.ok("GET", "/contexts/perm/schema", None);
    assert_eq!(schema["mode"], json!("strict"), "{schema}");
}

/// The cross-batch preview fix the promote dry run exposed, pinned on
/// `/import?dry_run=true` itself: export puts every alias on the LAST
/// batch while its canonical is interned by an earlier one, so a dry
/// run of a restore into a fresh name must seed each batch's check
/// with what the batches before it would intern — without that, the
/// preview refuses (spurious `UnknownCanonical`) a stream the real
/// import applies cleanly.
#[test]
fn an_export_stream_dry_runs_clean_when_aliases_trail_their_canonicals() {
    let server = Server::start("promote-preview-seeds");
    seed(&server);

    let (status, exported) = server.call("GET", "/contexts/scratch-claude/export", None);
    assert_eq!(status, 200, "{exported}");
    let stream = exported.as_str().expect("NDJSON body").replace(
        "\"context\":\"scratch-claude\"",
        "\"context\":\"fresh-restore\"",
    );
    assert!(
        stream.lines().last().unwrap().contains("\"alias\""),
        "the premise: aliases ride the last batch: {stream}"
    );

    let (status, previewed) = server.call_raw(
        "POST",
        "/import?dry_run=true",
        Some(&stream),
        Some("application/x-ndjson"),
    );
    assert_eq!(
        status, 200,
        "a dry run must refuse exactly what the real import would: {previewed}"
    );
    assert_eq!(
        previewed["result"]["batches"].as_array().map(Vec::len),
        Some(3),
        "{previewed}"
    );
    let (status, _) = server.call("GET", "/contexts/fresh-restore", None);
    assert_eq!(status, 404, "the preview must create nothing");
}

/// The advertised MCP tool drives the same endpoint through `/mcp` —
/// the runbook's graph path is reachable without leaving the agent's
/// tool surface.
#[test]
fn the_mcp_promote_tool_reaches_the_endpoint() {
    let server = Server::start("promote-mcp");
    seed(&server);

    let result = server.call_tool(
        7,
        "promote",
        json!({
            "context": "scratch-claude",
            "into": "perm",
            "sources": ["session:claude:a/note"],
            "audit": false,
            "dry_run": true
        }),
    );
    let text = result["content"][0]["text"].as_str().expect("text");
    assert!(text.contains("\"batches\""), "{result}");
    let sources = server.ok("GET", "/contexts/perm/sources", None);
    assert_eq!(
        sources["total"],
        json!(0),
        "dry_run through the tool writes nothing"
    );
}
