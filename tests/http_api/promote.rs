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
    // A passage-less source into a destination that never stored one:
    // the preview must not report a phantom passage drop.
    assert_eq!(
        preview["batches"][1]["passage_dropped"],
        json!(false),
        "{preview}"
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

    // Past the shared list ceiling the request refuses up front —
    // before the per-id validation work its size would otherwise buy.
    let over_cap: Vec<String> = (0..1001).map(|i| format!("session:claude:{i}")).collect();
    let (status, refused) = server.call(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(json!({"into": "perm", "sources": over_cap})),
    );
    assert_eq!(status, 400, "{refused}");
    assert_eq!(refused["code"], json!("over_limit"), "{refused}");
    assert!(
        refused["error"]
            .as_str()
            .unwrap()
            .contains("split the request"),
        "{refused}"
    );

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
    assert_eq!(
        refused["retryable_after_correction"],
        json!(true),
        "{refused}"
    );

    // The reserved export ids are stream artifacts, never promotable —
    // sourceless weight cannot travel with a promotion.
    let (status, refused) = server.call(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(json!({"into": "perm", "sources": ["export:unsourced"]})),
    );
    assert_eq!(status, 400, "{refused}");
    assert!(
        refused["error"].as_str().unwrap().contains("reserved"),
        "{refused}"
    );

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
    // The true count survives (issue #623 finding 5), and the message
    // above already names it once — not twice.
    assert_eq!(refused["issues_total"], json!(1), "{refused}");
    assert_eq!(refused["integrity"], json!("nothing_written"), "{refused}");
    assert_eq!(
        refused["retryable_after_correction"],
        json!(true),
        "{refused}"
    );
    let db = server.ok(
        "POST",
        "/contexts/perm/query",
        Some(json!({"subject": "DB"})),
    );
    assert_eq!(db["total"], json!(0), "nothing may have applied: {db}");

    // Past MAX_LISTED_ISSUES (20) mistyped ids: `issues_total` (25)
    // must survive `truncate_issues`, not collapse to the listed
    // count (20) — the two agree at a single mistyped id above, so
    // that case alone cannot tell `issues_total` apart from
    // `issues.len()`.
    let fake_sources: Vec<String> = (0..25).map(|i| format!("session:claude:typo{i}")).collect();
    let (status, refused) = server.call(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(json!({"into": "perm", "sources": fake_sources})),
    );
    assert_eq!(status, 404, "{refused}");
    assert_eq!(refused["issues_total"], json!(25), "{refused}");
    assert_eq!(refused["issues"].as_array().unwrap().len(), 20, "{refused}");

    // A fully-retracted source is no longer promotable: its dead
    // attribution rows (count 0) must not count as "exists here".
    server.ok(
        "POST",
        "/contexts/scratch-claude/sources/retract",
        Some(json!({"source": "session:claude:b"})),
    );
    let (status, refused) = server.call(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(json!({"into": "perm", "sources": ["session:claude:b"]})),
    );
    assert_eq!(
        status, 404,
        "a retracted source has nothing left to promote: {refused}"
    );
    assert_eq!(refused["code"], json!("no_source"), "{refused}");
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

/// The destination lives in the body, out of the route check's reach:
/// a context-scoped key needs `into` in its grant too, and a grant
/// covering both contexts clears the gate.
#[test]
fn a_scoped_key_needs_the_destination_in_its_grant() {
    let server = Server::start_with_env(
        "promote-scopes",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,pair:ptok,half:htok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"pair": {"role": "write", "contexts": ["scratch-claude", "perm"]},
                    "half": {"role": "write", "contexts": ["scratch-claude"]}}"#,
            ),
        ],
    );
    let call = |method: &str, path: &str, body: Option<serde_json::Value>, token: &str| {
        server.call_with_token(method, path, body, Some(token))
    };
    for (context, description) in [("scratch-claude", "notes"), ("perm", "permanent")] {
        let (status, body) = call(
            "PUT",
            &format!("/contexts/{context}"),
            Some(json!({"description": description})),
            "atok",
        );
        assert_eq!(status, 200, "{body}");
    }
    let (status, body) = call(
        "POST",
        "/contexts/scratch-claude/associations",
        Some(
            json!([{"subject": "DB", "label": "採用", "object": "PostgreSQL 16",
                     "weight": 1.0, "source": "session:claude:a"}]),
        ),
        "atok",
    );
    assert_eq!(status, 200, "{body}");

    let request = json!({"into": "perm", "sources": ["session:claude:a"], "audit": false});
    let (status, refused) = call(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(request.clone()),
        "htok",
    );
    assert_eq!(status, 403, "{refused}");
    assert!(
        refused["error"].as_str().unwrap().contains("'into'"),
        "the refusal names which grant is missing: {refused}"
    );
    assert_eq!(refused["integrity"], json!("nothing_written"), "{refused}");

    let (status, promoted) = call(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(request),
        "ptok",
    );
    assert_eq!(
        status, 200,
        "a grant covering both contexts clears the gate: {promoted}"
    );
}

/// The destination's storage quota gates promoted growth exactly as
/// `/import` gates a stream: batch-granular, checked before the batch
/// is attempted, and never on a dry run (a preview writes nothing to
/// gate — its capacity answers are advisory by documented contract).
#[test]
fn the_destination_quota_gates_growth_before_the_batch_is_attempted() {
    let server = Server::start_with_env(
        "promote-quota",
        &[(
            "TAGURU_CONTEXT_QUOTAS",
            r#"{"perm": {"storage_bytes": 1, "cache_bytes": 1048576}}"#,
        )],
    );
    seed(&server);
    // Put the destination at its ceiling before the promotion — the
    // quotas.rs pattern: the direct write's WAL bytes are what the
    // live-lane pre-check reads.
    server.ok(
        "POST",
        "/contexts/perm/associations",
        Some(json!([{"subject": "蔵", "label": "杜氏", "object": "高瀬",
                     "weight": 1.0, "source": "keep.md"}])),
    );

    // A preview writes nothing, so the ceiling has nothing to gate.
    let preview = server.ok(
        "POST",
        "/contexts/scratch-claude/promote?dry_run=true",
        Some(json!({
            "into": "perm",
            "sources": ["session:claude:a/note", "session:claude:b"]
        })),
    );
    assert_eq!(preview["batches"].as_array().map(Vec::len), Some(2));

    // For real: the first batch is stopped BEFORE it is attempted —
    // nothing written, and the fields say so machine-readably.
    let (status, refused) = server.call(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(json!({
            "into": "perm",
            "sources": ["session:claude:a/note", "session:claude:b"]
        })),
    );
    assert_eq!(status, 507, "{refused}");
    assert_eq!(refused["code"], json!("storage_full"), "{refused}");
    assert_eq!(refused["integrity"], json!("nothing_written"), "{refused}");
    let message = refused["error"].as_str().unwrap();
    assert!(message.contains("not attempted"), "{message}");
    assert!(message.contains("storage quota"), "{message}");
}

/// Sums the five on-disk lanes `AppState::storage_quota_excess` gates
/// on (`src/registry/engine.rs`) straight off `/metrics`, requiring
/// `TAGURU_METRICS_PER_CONTEXT`.
fn disk_total_bytes(server: &Server, context: &str) -> u64 {
    let (status, body) = server.call("GET", "/metrics", None);
    assert_eq!(status, 200);
    let text = body.as_str().expect("metrics body is text, not JSON");
    ["image", "passages", "passages_wal", "sidecars", "wal"]
        .iter()
        .map(|file| {
            let prefix =
                format!("taguru_context_disk_bytes{{context=\"{context}\",file=\"{file}\"}} ");
            text.lines()
                .find_map(|line| line.strip_prefix(prefix.as_str()))
                .and_then(|value| value.trim().parse::<u64>().ok())
                .unwrap_or(0)
        })
        .sum()
}

/// The other half of the ceiling test above: when the destination
/// clears its ceiling for the FIRST promoted batch but that batch's
/// own growth tips it over, the SECOND batch's refusal reports a
/// durable prefix — `quota_refusal`'s `landed > 0` shape, unreachable
/// from the `nothing_written` path both this file's earlier test and
/// `quotas.rs` already cover. `storage_quota_refusal` is a pure
/// function of the destination's CURRENT disk usage, never wall-clock
/// time, so the boundary is fully deterministic: measure what landing
/// batch 0 alone costs on an uncapped probe server, then reproduce the
/// identical scenario on a fresh server whose ceiling sits exactly at
/// that measured usage.
#[test]
fn quota_refusal_reports_a_durable_prefix_when_the_first_landed_batch_tips_the_ceiling() {
    let probe = Server::start_with_env(
        "promote-quota-probe",
        &[("TAGURU_METRICS_PER_CONTEXT", "1")],
    );
    seed(&probe);
    probe.ok(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(json!({"into": "perm", "sources": ["session:claude:a/note"]})),
    );
    let ceiling = disk_total_bytes(&probe, "perm");
    assert!(
        ceiling > 0,
        "the landed batch must have grown the destination"
    );

    let quotas = format!(r#"{{"perm": {{"storage_bytes": {ceiling}, "cache_bytes": 1048576}}}}"#);
    let server = Server::start_with_env(
        "promote-quota-durable",
        &[("TAGURU_CONTEXT_QUOTAS", quotas.as_str())],
    );
    seed(&server);

    let (status, refused) = server.call(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(json!({
            "into": "perm",
            "sources": ["session:claude:a/note", "session:claude:b"]
        })),
    );
    assert_eq!(status, 507, "{refused}");
    assert_eq!(refused["code"], json!("storage_full"), "{refused}");
    assert_eq!(refused["integrity"], json!("durable_prefix"), "{refused}");
    assert_eq!(refused["durable_batches"], json!(1), "{refused}");
    let message = refused["error"].as_str().unwrap();
    assert!(message.contains("batch 2 of 2"), "{message}");
    assert!(message.contains("storage quota"), "{message}");

    // The first batch landed for real before the second was refused.
    let sources = server.ok("GET", "/contexts/perm/sources", None);
    assert_eq!(sources["total"], json!(1), "{sources}");
}

/// A `warn`-mode destination schema lets the promoted batches land and
/// reports the violations in the success envelope, `/import`'s own
/// accounting — the exact true count, not a truncation artifact.
#[test]
fn a_warn_mode_destination_reports_schema_violations_in_the_envelope() {
    let server = Server::start("promote-warn-schema");
    seed(&server);
    server.ok(
        "PUT",
        "/contexts/perm/schema",
        Some(json!({
            "schema": 1, "mode": "warn", "closed_labels": true,
            "types": {},
            "relations": {"テストランナー": {"domain": [], "range": []}}
        })),
    );

    let (status, body) = server.call(
        "POST",
        "/contexts/scratch-claude/promote",
        Some(json!({"into": "perm", "sources": ["session:claude:a/note"], "audit": false})),
    );
    assert_eq!(status, 200, "warn lets the batch land: {body}");
    assert_eq!(body["schema_violations"], json!(1), "{body}");
    assert_eq!(
        body["issues"][0]["path"]
            .as_str()
            .map(|path| path.starts_with("batches[0].")),
        Some(true),
        "{body}"
    );
    assert_eq!(
        body["result"]["batches"][0]["schema_violations"],
        json!(1),
        "{body}"
    );
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

/// The per-context half of the same preview fix: a stream can
/// interleave contexts, and a name interned into a SIBLING context
/// vouches for nothing — the real import refuses the alias, so the
/// dry run must refuse it identically instead of letting the sibling's
/// vocabulary leak across.
#[test]
fn a_cross_context_stream_does_not_let_sibling_vocabulary_vouch() {
    let server = Server::start("promote-preview-contexts");
    // alpha interns Foo; beta's alias names Foo without interning it.
    let stream = concat!(
        "{\"taguru_batch\": 1, \"context\": \"alpha\", \"source\": \"a.md\", ",
        "\"create\": {\"description\": \"a\"}}\n",
        "{\"subject\": \"Foo\", \"label\": \"関連\", \"object\": \"Bar\", \"weight\": 1.0}\n",
        "{\"taguru_batch\": 1, \"context\": \"beta\", \"source\": \"b.md\", ",
        "\"create\": {\"description\": \"b\"}}\n",
        "{\"subject\": \"X\", \"label\": \"関連\", \"object\": \"Y\", \"weight\": 1.0}\n",
        "{\"alias\": \"ふー\", \"canonical\": \"Foo\", \"kind\": \"concept\"}\n",
    );

    let (status, previewed) = server.call_raw(
        "POST",
        "/import?dry_run=true",
        Some(stream),
        Some("application/x-ndjson"),
    );
    assert_eq!(
        status, 409,
        "the preview must refuse exactly what the real import refuses: {previewed}"
    );
    let (status, refused) = server.call_raw(
        "POST",
        "/import",
        Some(stream),
        Some("application/x-ndjson"),
    );
    assert_eq!(status, 409, "{refused}");
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
