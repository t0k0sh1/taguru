//! The `taguru import` CLI path and the /import HTTP endpoint.

use serde_json::{Value, json};

use crate::support::*;

#[test]
fn an_offline_import_lands_facts_passage_and_aliases_the_server_serves() {
    let batches = batch_dir("import-serve");
    let file = batches.join("guide.jsonl");
    std::fs::write(
        &file,
        r#"{"taguru_batch": 1, "context": "sake", "source": "doc-guide", "create": {"description": "酒蔵の記憶"}}
{"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 2.0}
{"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0}
{"alias": "Aomine", "canonical": "青嶺酒造", "kind": "concept"}
{"passage": "青嶺酒造の杜氏は高瀬。1907年創業。"}
"#,
    )
    .unwrap();

    let data_dir =
        std::env::temp_dir().join(format!("taguru-http-import-serve-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
    let (code, stdout, stderr) = run_import(&data_dir, &[file.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("+2 association(s)"), "{stdout}");
    assert!(stdout.contains("(created)"), "{stdout}");

    // The server boots on the imported directory and serves it all:
    // the facts, the alias entry point, and the original passage.
    let server = Server::start_on("import-serve", data_dir);
    let brewer = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "Aomine", "label": "杜氏"})),
    );
    assert_eq!(brewer["matches"][0]["subject"], json!("青嶺酒造"));
    assert_eq!(brewer["matches"][0]["object"], json!("高瀬"));
    assert_eq!(brewer["matches"][0]["weight"], json!(2.0));
    let passages = server.ok(
        "POST",
        "/contexts/sake/sources/lookup",
        Some(json!({"sources": ["doc-guide"]})),
    );
    assert_eq!(
        passages["passages"]["doc-guide"],
        json!("青嶺酒造の杜氏は高瀬。1907年創業。")
    );
    let _ = std::fs::remove_dir_all(&batches);
}

/// `apply_batch`'s question bookkeeping (src/ingest.rs) had no coverage
/// of its own — only the direct `POST .../sources` path (see
/// `store_passages_accepts_questions_and_reports_the_bookkeeping`) was
/// exercised. This drives the same stored/dropped counters through a
/// batch file and confirms a stored question actually rides the
/// passage into the search index under the paragraph it names.
#[test]
fn an_offline_import_carries_questions_through_to_the_search_index() {
    let batches = batch_dir("import-questions");
    let file = batches.join("guide.jsonl");
    std::fs::write(
        &file,
        r#"{"taguru_batch": 1, "context": "sake", "source": "doc-guide", "create": {"description": "酒蔵の記憶"}}
{"passage": "青嶺酒造は1907年に創業した。\n\n杜氏は高瀬。"}
{"question": "杜氏は誰?", "paragraph": 1}
{"question": "存在しない段落への質問?", "paragraph": 9}
"#,
    )
    .unwrap();

    let data_dir = std::env::temp_dir().join(format!(
        "taguru-http-import-questions-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&data_dir);
    let (code, stdout, stderr) = run_import(&data_dir, &[file.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("+1 question(s) (1 dropped: no such paragraph)"),
        "{stdout}"
    );

    // The stored question rides the passage into the search index: a
    // query using the QUESTION's wording, not the paragraph's own,
    // still finds the paragraph it names.
    let server = Server::start_on("import-questions", data_dir);
    let hits = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "杜氏は誰?", "limit": 3})),
    );
    let hit = &hits["hits"][0];
    assert_eq!(hit["source"], "doc-guide");
    assert_eq!(hit["paragraph"], 1, "the hit names the answering PARAGRAPH");
    let _ = std::fs::remove_dir_all(&batches);
}

/// The import-time bookkeeping for storing/dropping section markers —
/// the same drop-and-count convention as questions. Resolving a stored
/// section back out through recall/query is covered separately by
/// `an_attributions_section_label_resolves_on_read_but_is_never_fabricated`.
#[test]
fn an_offline_import_carries_sections_through_and_drops_out_of_range_ones() {
    let batches = batch_dir("import-sections");
    let file = batches.join("guide.jsonl");
    std::fs::write(
        &file,
        r#"{"taguru_batch": 1, "context": "sake", "source": "doc-guide", "create": {"description": "酒蔵の記憶"}}
{"passage": "青嶺酒造は1907年に創業した。\n\n杜氏は高瀬。"}
{"paragraph": 1, "section": "杜氏"}
{"paragraph": 9, "section": "存在しない段落"}
"#,
    )
    .unwrap();

    let data_dir = std::env::temp_dir().join(format!(
        "taguru-http-import-sections-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&data_dir);
    let (code, stdout, stderr) = run_import(&data_dir, &[file.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("+1 section(s) (1 dropped: no such paragraph)"),
        "{stdout}"
    );
    let _ = std::fs::remove_dir_all(&batches);
}

/// Unlike questions/sections above, an association's paragraph is
/// incidental metadata, not its reason for existing — so an out-of-range
/// one clears just the locator and keeps the whole fact, where a
/// question or section drops entirely. But the drop is still surfaced
/// with its own count and report line, symmetric with those two: a
/// locator silently vanishing is exactly the kind of loss the report
/// exists to name.
#[test]
fn an_offline_import_drops_an_out_of_range_association_paragraph_but_keeps_the_fact() {
    let batches = batch_dir("import-assoc-paragraph");
    let file = batches.join("guide.jsonl");
    std::fs::write(
        &file,
        r#"{"taguru_batch": 1, "context": "sake", "source": "doc-guide", "create": {"description": "酒蔵の記憶"}}
{"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "paragraph": 0}
{"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0, "paragraph": 9}
{"passage": "青嶺酒造は1907年に創業した。\n\n杜氏は高瀬。"}
"#,
    )
    .unwrap();

    let data_dir = std::env::temp_dir().join(format!(
        "taguru-http-import-assoc-paragraph-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&data_dir);
    let (code, stdout, stderr) = run_import(&data_dir, &[file.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    // Both facts land — the out-of-range locator does not cost its
    // association — and the one dropped locator is named, not silent.
    assert!(stdout.contains("+2 association(s)"), "{stdout}");
    assert!(
        stdout.contains("1 association paragraph locator(s) dropped: no such paragraph"),
        "{stdout}"
    );

    let server = Server::start_on("import-assoc-paragraph", data_dir);
    let founding = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "創業年"})),
    );
    assert_eq!(
        founding["matches"][0]["attributions"][0]["paragraph"],
        json!(0),
        "an in-range paragraph must survive: {founding}"
    );
    let brewer = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "杜氏"})),
    );
    assert_eq!(
        brewer["matches"][0]["weight"],
        json!(1.0),
        "the fact itself must survive even though its locator did not: {brewer}"
    );
    assert_eq!(
        brewer["matches"][0]["attributions"][0]["paragraph"],
        json!(null),
        "an out-of-range paragraph must be cleared, not left dangling: {brewer}"
    );
    let _ = std::fs::remove_dir_all(&batches);
}

/// The same silent clamp, reached through the direct HTTP path instead
/// of an import batch: nothing between `store_passages` and
/// `associations` has the passage text in hand, so
/// `AppState::add_associations` must check the resident passage store
/// itself before honoring a paragraph locator.
#[test]
fn http_associations_drops_an_out_of_range_paragraph_against_a_stored_passage() {
    let server = Server::start("http-assoc-paragraph");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "酒蔵の記憶"})),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {"doc-guide": "青嶺酒造は1907年に創業した。\n\n杜氏は高瀬。"}
        })),
    );
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "source": "doc-guide", "paragraph": 0},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "doc-guide", "paragraph": 9},
        ])),
    );

    let founding = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "創業年"})),
    );
    assert_eq!(
        founding["matches"][0]["attributions"][0]["paragraph"],
        json!(0),
        "an in-range paragraph must survive: {founding}"
    );

    let brewer = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "杜氏"})),
    );
    assert_eq!(
        brewer["matches"][0]["weight"],
        json!(1.0),
        "the fact itself must survive even though its locator did not: {brewer}"
    );
    assert_eq!(
        brewer["matches"][0]["attributions"][0]["paragraph"],
        json!(null),
        "an out-of-range paragraph must be cleared, not left dangling: {brewer}"
    );
}

/// Issue #11: an attribution whose paragraph locator falls inside a
/// stored section resolves that section's label on read — recall,
/// query, and (through the same conversion) explore, activate, and
/// unreachable_from all carry it. A paragraph that exists but sits
/// before every marker, and an attribution with no paragraph locator
/// at all, must both report `section: null` — resolution never
/// fabricates a label. Those two null outcomes come out of the one
/// shared `attribution_out` conversion every endpoint routes through,
/// so recall and query (which exhaust them) are sufficient — repeating
/// them per endpoint would not catch a broken wiring the way a
/// resolved assertion does (null passes whether or not resolution ran
/// at all).
#[test]
fn an_attributions_section_label_resolves_on_read_but_is_never_fabricated() {
    let batches = batch_dir("import-section-resolution");
    let file = batches.join("guide.jsonl");
    std::fs::write(
        &file,
        r#"{"taguru_batch": 1, "context": "sake", "source": "doc-guide", "create": {"description": "酒蔵の記憶"}}
{"passage": "青嶺酒造は1907年に創業した。\n\n杜氏は高瀬。\n\n仕込み水は雲居山の伏流水である。"}
{"paragraph": 1, "section": "杜氏"}
{"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "paragraph": 0}
{"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0, "paragraph": 1}
{"subject": "青嶺酒造", "label": "仕込み水源", "object": "雲居山", "weight": 1.0}
{"subject": "廻船問屋", "label": "取引先", "object": "山田", "weight": 1.0, "paragraph": 1}
"#,
    )
    .unwrap();

    let data_dir = std::env::temp_dir().join(format!(
        "taguru-http-import-section-resolution-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&data_dir);
    let (code, stdout, stderr) = run_import(&data_dir, &[file.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let server = Server::start_on("import-section-resolution", data_dir);

    // Paragraph 1 sits inside the "杜氏" section (which starts at
    // paragraph 1 and runs to the passage's end): recall must resolve
    // it, matching what a manual read of the source would show.
    let recalled = server.ok(
        "POST",
        "/contexts/sake/recall",
        Some(json!({"cue": "青嶺酒造", "limit": 10})),
    );
    let brewer = recalled["matches"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["label"] == "杜氏")
        .expect("the 杜氏 association must recall from its subject");
    assert_eq!(
        brewer["attributions"][0]["section"],
        json!("杜氏"),
        "a paragraph inside a stored section must resolve its label: {brewer}"
    );

    // Paragraph 0 exists but sits BEFORE the first section marker: no
    // section governs it, so resolution must report null rather than
    // guessing at the nearest one.
    let founding = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "創業年"})),
    );
    assert_eq!(
        founding["matches"][0]["attributions"][0]["paragraph"],
        json!(0),
        "sanity: the paragraph locator itself must still be present: {founding}"
    );
    assert_eq!(
        founding["matches"][0]["attributions"][0]["section"],
        json!(null),
        "a paragraph before every marker must not resolve to a section: {founding}"
    );

    // No paragraph locator at all: section is null with nothing to
    // resolve from.
    let water = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "仕込み水源"})),
    );
    assert_eq!(
        water["matches"][0]["attributions"][0]["paragraph"],
        json!(null),
        "sanity: this attribution carries no locator: {water}"
    );
    assert_eq!(
        water["matches"][0]["attributions"][0]["section"],
        json!(null),
        "an attribution without a paragraph must not resolve to a section: {water}"
    );

    // explore nests associations one level deeper
    // (matches[].association.attributions[]) — check the same
    // resolution reaches that shape too.
    let explored = server.ok(
        "POST",
        "/contexts/sake/explore",
        Some(json!({"origins": ["青嶺酒造"], "max_depth": 1})),
    );
    let brewer_hop = explored["matches"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["association"]["label"] == "杜氏")
        .expect("the 杜氏 association must be one hop from 青嶺酒造");
    assert_eq!(
        brewer_hop["association"]["attributions"][0]["section"],
        json!("杜氏"),
        "explore's nested association must resolve sections too: {brewer_hop}"
    );

    // activate nests associations the same one level deep
    // (matches[].association) — check the same resolution reaches that
    // shape too.
    let activated = server.ok(
        "POST",
        "/contexts/sake/activate",
        Some(json!({"origins": ["青嶺酒造"], "limit": 10})),
    );
    let brewer_activation = activated["matches"]
        .as_array()
        .unwrap()
        .iter()
        .find(|a| a["association"]["label"] == "杜氏")
        .expect("the 杜氏 association must activate from 青嶺酒造");
    assert_eq!(
        brewer_activation["association"]["attributions"][0]["section"],
        json!("杜氏"),
        "activate's nested association must resolve sections too: {brewer_activation}"
    );

    // unreachable_from returns associations no walk from the origin can
    // reach — 廻船問屋 is isolated from 青嶺酒造's graph, so it
    // qualifies; check its section resolves too.
    let orphaned = server.ok(
        "POST",
        "/contexts/sake/unreachable_from",
        Some(json!({"origins": ["青嶺酒造"]})),
    );
    let orphan_match = orphaned["matches"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["subject"] == "廻船問屋")
        .expect("the isolated association must be unreachable from 青嶺酒造");
    assert_eq!(
        orphan_match["attributions"][0]["section"],
        json!("杜氏"),
        "unreachable_from's associations must resolve sections too: {orphan_match}"
    );

    let _ = std::fs::remove_dir_all(&batches);
}

#[test]
fn reimporting_a_source_replaces_it_instead_of_doubling() {
    let batches = batch_dir("import-idem");
    let file = batches.join("facts.jsonl");
    let header = r#"{"taguru_batch": 1, "context": "sake", "source": "doc-1", "create": {"description": "d"}}"#;
    std::fs::write(
        &file,
        format!(
            "{header}\n{}\n",
            r#"{"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 2.0}"#
        ),
    )
    .unwrap();

    let data_dir =
        std::env::temp_dir().join(format!("taguru-http-import-idem-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
    // Twice: the weight must not accumulate across identical imports.
    for _ in 0..2 {
        let (code, stdout, stderr) = run_import(&data_dir, &[file.to_str().unwrap()]);
        assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    }
    let server = Server::start_on("import-idem", data_dir);
    let edge = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "杜氏"})),
    );
    assert_eq!(edge["matches"][0]["weight"], json!(2.0));
    assert_eq!(
        edge["matches"][0]["attributions"].as_array().unwrap().len(),
        1
    );
    let data_dir = server.stop_gracefully();

    // A revised file for the same source: its truth replaces, never
    // stacks onto, the old one.
    std::fs::write(
        &file,
        format!(
            "{header}\n{}\n",
            r#"{"subject": "蔵", "label": "杜氏", "object": "高瀬", "weight": 5.0}"#
        ),
    )
    .unwrap();
    let (code, _, stderr) = run_import(&data_dir, &[file.to_str().unwrap()]);
    assert_eq!(code, 0, "{stderr}");
    let server = Server::start_on("import-idem-2", data_dir);
    let edge = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "杜氏"})),
    );
    assert_eq!(edge["matches"][0]["weight"], json!(5.0));
    let _ = std::fs::remove_dir_all(&batches);
}

/// A predicted alias rejection is caught before the first mutation, so
/// a batch that pairs a valid association with a conflicting alias
/// must not touch the context at all — not the association, not the
/// alias, and the run's summary (which the embeddings-refresh pass
/// mirrors) must not claim the context needs a tick either, since
/// nothing new landed for it to embed.
#[test]
fn a_predicted_alias_rejection_touches_no_context_for_the_refresh_pass() {
    let batches = batch_dir("import-partial-touch");
    let setup = batches.join("setup.jsonl");
    std::fs::write(
        &setup,
        r#"{"taguru_batch": 1, "context": "sake", "source": "doc-1", "create": {"description": "d"}}
{"subject": "青嶺酒造", "label": "所在地", "object": "京都酒造", "weight": 1.0}
{"alias": "kyo", "canonical": "京都酒造", "kind": "concept"}
"#,
    )
    .unwrap();
    let data_dir = std::env::temp_dir().join(format!(
        "taguru-http-import-partial-touch-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&data_dir);
    let (code, stdout, stderr) = run_import(&data_dir, &[setup.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    // BTreeMap order: "aomine" would resolve fine, but "kyo" would
    // conflict by re-pointing an existing alias — predicted before
    // either the association or "aomine" ever lands, so the whole
    // batch is refused up front.
    let partial = batches.join("partial.jsonl");
    std::fs::write(
        &partial,
        r#"{"taguru_batch": 1, "context": "sake", "source": "doc-2"}
{"subject": "新蔵", "label": "特徴", "object": "辛口", "weight": 1.0}
{"alias": "aomine", "canonical": "青嶺酒造", "kind": "concept"}
{"alias": "kyo", "canonical": "青嶺酒造", "kind": "concept"}
"#,
    )
    .unwrap();
    let (code, stdout, stderr) = run_import(&data_dir, &[partial.to_str().unwrap()]);
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("nothing was applied"), "{stderr}");
    assert!(
        stdout.contains("across 0 context(s)"),
        "a predicted rejection touches nothing: {stdout}"
    );

    // And the association the rejected batch carried really is
    // nowhere to be found.
    let server = Server::start_on("import-partial-touch", data_dir);
    let edge = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "新蔵", "label": "特徴"})),
    );
    assert_eq!(
        edge["matches"],
        json!([]),
        "the rejected batch's association must not have landed"
    );
    let _ = std::fs::remove_dir_all(&batches);
}

/// The CLI-facing test above confirms nothing NEW landed; this
/// confirms the same thing from the alias catalog's own side — the
/// "valid" alias must not register just because it sorts before the
/// conflicting one in the batch's `BTreeMap`. Predicting one bad entry
/// refuses the whole alias step, valid entries included.
#[test]
fn a_batch_with_one_valid_and_one_conflicting_alias_registers_neither() {
    let server = Server::start("http-import-mixed-aliases");
    let (status, _) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\", \
         \"create\": {\"description\": \"d\"}}\n\
         {\"subject\": \"青嶺酒造\", \"label\": \"所在地\", \"object\": \"京都酒造\", \"weight\": 1.0}\n\
         {\"alias\": \"kyo\", \"canonical\": \"京都酒造\", \"kind\": \"concept\"}\n",
        None,
    );
    assert_eq!(status, 200);

    // BTreeMap order: "avalid" sorts first and would register cleanly
    // on its own; "kyo" sorts after it and conflicts by re-pointing
    // the alias set up above. Predicting the second entry's rejection
    // must refuse the first entry too.
    let (status, body) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-2\"}\n\
         {\"alias\": \"avalid\", \"canonical\": \"青嶺酒造\", \"kind\": \"concept\"}\n\
         {\"alias\": \"kyo\", \"canonical\": \"青嶺酒造\", \"kind\": \"concept\"}\n",
        None,
    );
    assert_eq!(status, 409, "{body}");

    let listing = server.ok("GET", "/contexts/sake/aliases", None);
    assert_eq!(
        listing["concepts"],
        json!({"kyo": "京都酒造"}),
        "the valid alias must not register just because it sorts first: {listing}"
    );
}

#[test]
fn import_refuses_a_data_directory_a_live_server_holds() {
    let server = Server::start("import-locked");
    let batches = batch_dir("import-locked");
    let file = batches.join("late.jsonl");
    std::fs::write(
        &file,
        "{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"s\", \"create\": {}}\n",
    )
    .unwrap();
    let (code, _, stderr) = run_import(&server.data_dir, &[file.to_str().unwrap()]);
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("another taguru process"), "{stderr}");
    let _ = std::fs::remove_dir_all(&batches);
}

#[test]
fn a_malformed_file_refuses_the_whole_import_before_any_write() {
    let batches = batch_dir("import-refuse");
    let good = batches.join("good.jsonl");
    std::fs::write(
        &good,
        "{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"s\", \"create\": {}}\n",
    )
    .unwrap();
    let bad = batches.join("bad.jsonl");
    std::fs::write(
        &bad,
        "{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"t\"}\n\n{\"foo\": 1}\n",
    )
    .unwrap();

    let data_dir =
        std::env::temp_dir().join(format!("taguru-http-import-refuse-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
    let (code, _, stderr) = run_import(&data_dir, &[good.to_str().unwrap(), bad.to_str().unwrap()]);
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("line 3"), "{stderr}");
    assert!(stderr.contains("nothing was applied"), "{stderr}");
    // Refused during validation: the good file was NOT applied either —
    // the data directory was never even created.
    assert!(!data_dir.exists(), "validation must not touch the disk");

    // The same holds for a clean --dry-run.
    let (code, stdout, stderr) = run_import(&data_dir, &["--dry-run", good.to_str().unwrap()]);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("dry run"), "{stdout}");
    assert!(!data_dir.exists(), "a dry run must not touch the disk");
    let _ = std::fs::remove_dir_all(&batches);
}

#[test]
fn the_import_endpoint_applies_batches_to_a_live_server() {
    let server = Server::start_with_env("http-import", &[("TAGURU_API_TOKEN", "opskey")]);
    let batch = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-live\", \
                 \"create\": {\"description\": \"d\"}}\n\
                 {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 2.0}\n\
                 {\"passage\": \"蔵の杜氏は高瀬。\"}\n";

    // The endpoint sits behind bearer auth like any other write.
    let (status, _) = post_import(&server, batch, None);
    assert_eq!(status, 401);

    let (status, first) = post_import(&server, batch, Some("opskey"));
    assert_eq!(status, 200, "{first}");
    // A single-batch body answers the same {batches: [...]} shape a
    // stream does — one shape to parse, with one entry here.
    let outcome = &first["result"]["batches"][0];
    assert_eq!(first["result"]["batches"].as_array().map(Vec::len), Some(1));
    assert_eq!(outcome["created"], json!(true));
    assert_eq!(outcome["associations"], json!(1));
    assert_eq!(outcome["passage_stored"], json!(true));
    assert_eq!(outcome["retracted"], json!(0));

    // Same batch again: the source is replaced, not doubled — the
    // no-downtime spelling of the CLI's idempotency.
    let (status, second) = post_import(&server, batch, Some("opskey"));
    assert_eq!(status, 200, "{second}");
    assert_eq!(second["result"]["batches"][0]["retracted"], json!(1));
    let (status, edge) = server.call_with_token(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "杜氏"})),
        Some("opskey"),
    );
    assert_eq!(status, 200);
    assert_eq!(edge["result"]["matches"][0]["weight"], json!(2.0));
}

/// `?dry_run=true` reports the same shape a real import would without
/// writing anything — the counts it previews match exactly what the
/// same batch applies for real right after.
#[test]
fn import_dry_run_previews_without_writing_anything() {
    let server = Server::start("http-import-dry-run");
    let batch = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-preview\", \
                 \"create\": {\"description\": \"d\"}}\n\
                 {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 2.0}\n\
                 {\"passage\": \"蔵の杜氏は高瀬。\"}\n";

    let (status, preview) = post_import_dry_run(&server, batch, None);
    assert_eq!(status, 200, "{preview}");
    let outcome = &preview["result"]["batches"][0];
    assert_eq!(outcome["created"], json!(true));
    assert_eq!(outcome["associations"], json!(1));
    assert_eq!(outcome["passage_stored"], json!(true));
    assert_eq!(outcome["retracted"], json!(0));

    // A preview writes nothing: the context it would create doesn't
    // exist yet.
    let (status, _) = server.call("GET", "/contexts/sake", None);
    assert_eq!(status, 404, "dry_run must not create the context");

    // The real import right after previews identically — same counts,
    // this time durable.
    let (status, real) = post_import(&server, batch, None);
    assert_eq!(status, 200, "{real}");
    assert_eq!(real["result"]["batches"][0], *outcome, "{real}");
    let (status, _) = server.call("GET", "/contexts/sake", None);
    assert_eq!(status, 200, "the real import did create the context");

    // Previewing a source replacement counts the retraction without
    // performing it — the edge the first import landed is still live.
    let (status, replace_preview) = post_import_dry_run(&server, batch, None);
    assert_eq!(status, 200, "{replace_preview}");
    assert_eq!(
        replace_preview["result"]["batches"][0]["created"],
        json!(false)
    );
    assert_eq!(
        replace_preview["result"]["batches"][0]["retracted"],
        json!(1),
        "{replace_preview}"
    );
    let (status, edge) = server.call(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "杜氏"})),
    );
    assert_eq!(status, 200);
    assert_eq!(
        edge["result"]["matches"][0]["weight"],
        json!(2.0),
        "the preview did not actually retract anything"
    );
}

/// A predicted alias rejection reaches the identical decision whether
/// or not the caller is previewing — `dry_run=true` must not let a
/// batch through that a real import would refuse, nor invent a
/// refusal a real import would not hit. Builds on the happy-path
/// parity above, this time down the refusal path.
#[test]
fn dry_run_and_a_real_import_reach_the_same_predicted_alias_rejection() {
    let server = Server::start("http-import-dry-run-parity");
    let (status, _) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\", \
         \"create\": {\"description\": \"d\"}}\n\
         {\"subject\": \"青嶺酒造\", \"label\": \"所在地\", \"object\": \"京都酒造\", \"weight\": 1.0}\n\
         {\"alias\": \"kyo\", \"canonical\": \"京都酒造\", \"kind\": \"concept\"}\n",
        None,
    );
    assert_eq!(status, 200);

    let conflicting = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-2\"}\n\
                        {\"alias\": \"kyo\", \"canonical\": \"青嶺酒造\", \"kind\": \"concept\"}\n";
    let (dry_status, dry_body) = post_import_dry_run(&server, conflicting, None);
    assert_eq!(dry_status, 409, "{dry_body}");
    let (real_status, real_body) = post_import(&server, conflicting, None);
    assert_eq!(real_status, 409, "{real_body}");
    assert_eq!(
        dry_body["error"], real_body["error"],
        "a preview must reach the exact same refusal text as the real import"
    );
    assert_eq!(dry_body["code"], real_body["code"]);
}

/// A stream that mixes batches with a `taguru_group` record previews
/// the batches but skips the group record entirely — the response
/// omits `groups`, and no group is created.
#[test]
fn import_dry_run_skips_group_records() {
    let server = Server::start("http-import-dry-run-groups");
    let stream = "{\"taguru_group\": 1, \"name\": \"kura\", \"contexts\": [\"sake\"]}\n\
                  {\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
                   \"create\": {\"description\": \"d\"}}\n";
    let (status, preview) = post_import_dry_run(&server, stream, None);
    assert_eq!(status, 200, "{preview}");
    assert_eq!(
        preview["result"]["batches"].as_array().map(Vec::len),
        Some(1)
    );
    assert!(
        preview["result"].get("groups").is_none(),
        "dry_run previews no group records: {preview}"
    );

    let (status, _) = server.call("GET", "/groups/kura", None);
    assert_eq!(status, 404, "dry_run must not create the group");
    let (status, _) = server.call("GET", "/contexts/sake", None);
    assert_eq!(status, 404, "dry_run must not create the context either");
}

/// A live import whose alias step is predicted to fail is refused
/// before the first mutation — no context created, no batch-open
/// marker written, so #59's tear detection has nothing to name.
/// Marker survival across a genuine mid-batch tear (one prediction
/// cannot catch, e.g. a disk fault) and its repairs — re-importing the
/// corrected batch, or retracting the source — are covered beside
/// `apply_batch` itself, since this endpoint and the CLI import share
/// that one function.
#[test]
fn a_predicted_alias_rejection_leaves_no_marker_on_a_live_import() {
    let server = Server::start("http-import-marker");
    // An alias whose canonical nothing interned is caught by
    // predicting the alias step's outcome before anything runs.
    let torn = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-torn\", \
                 \"create\": {\"description\": \"d\"}}\n\
                 {\"alias\": \"Aomine\", \"canonical\": \"存在しない\", \"kind\": \"concept\"}\n";
    let (status, body) = post_import(&server, torn, None);
    assert_eq!(
        status, 409,
        "the batch must be refused as a conflict: {body}"
    );

    let markers = |dir: &std::path::Path| -> Vec<std::path::PathBuf> {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("importing"))
            .collect()
    };
    assert!(
        markers(&server.data_dir).is_empty(),
        "a predicted rejection opens no marker"
    );
    let (status, _) = server.call("GET", "/contexts/sake", None);
    assert_eq!(
        status, 404,
        "a predicted rejection must not create the context"
    );
}

/// The issue #187 regression check: an import batch for a brand-new
/// source, refused because its own alias step is predicted to
/// conflict, must leave that source entirely absent from `GET
/// /contexts/{name}/sources` — not applied-with-an-error, not
/// partially there, not there at all.
#[test]
fn a_rejected_new_source_import_never_appears_in_list_sources() {
    let server = Server::start("http-import-reject-list-sources");
    let (status, _) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-old\", \
         \"create\": {\"description\": \"d\"}}\n\
         {\"passage\": \"蔵の杜氏は高瀬。\"}\n",
        None,
    );
    assert_eq!(status, 200);

    let (status, body) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-new\"}\n\
         {\"subject\": \"新蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n\
         {\"passage\": \"新しい文章。\"}\n\
         {\"alias\": \"Aomine\", \"canonical\": \"存在しない\", \"kind\": \"concept\"}\n",
        None,
    );
    assert_eq!(status, 409, "{body}");

    let sources = server.ok("GET", "/contexts/sake/sources", None);
    assert_eq!(
        sources["sources"],
        json!(["doc-old"]),
        "the rejected batch's new source must never appear: {sources}"
    );
}

/// The multi-batch "durable prefix" guarantee is a different axis from
/// this batch's own atomicity, and the fix must not blur the two:
/// batch 1 landing durably and batch 2 being refused before its first
/// mutation are both true of the same stream at once.
#[test]
fn a_rejected_batch_in_a_stream_leaves_the_earlier_batch_durable_and_its_own_source_absent() {
    let server = Server::start("http-import-stream-prefix");
    let stream = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\", \
                   \"create\": {\"description\": \"d\"}}\n\
                  {\"subject\": \"青嶺酒造\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n\
                  {\"passage\": \"青嶺酒造の杜氏は高瀬。\"}\n\
                  {\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-2\"}\n\
                  {\"alias\": \"Aomine\", \"canonical\": \"存在しない\", \"kind\": \"concept\"}\n";
    let (status, body) = post_import(&server, stream, None);
    assert_eq!(status, 409, "{body}");
    let message = body["error"].as_str().unwrap();
    assert!(message.contains("batch 2 of 2"), "{message}");
    assert!(
        message.contains("landed durably"),
        "the note must credit the earlier batch as durable: {message}"
    );

    // Batch 1's write stands...
    let edge = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "杜氏"})),
    );
    assert_eq!(edge["matches"][0]["object"], json!("高瀬"));

    // ...but batch 2's own source never appears at all.
    let sources = server.ok("GET", "/contexts/sake/sources", None);
    assert_eq!(
        sources["sources"],
        json!(["doc-1"]),
        "batch 2's own source must never appear: {sources}"
    );
}

#[test]
fn the_import_endpoint_reports_section_bookkeeping() {
    let server = Server::start("http-import-sections");
    let batch = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-sections\", \
                 \"create\": {\"description\": \"d\"}}\n\
                 {\"passage\": \"蔵の杜氏は高瀬。\\n\\n創業は1907年。\"}\n\
                 {\"paragraph\": 1, \"section\": \"沿革\"}\n\
                 {\"paragraph\": 9, \"section\": \"存在しない段落\"}\n";

    let (status, result) = post_import(&server, batch, None);
    assert_eq!(status, 200, "{result}");
    assert_eq!(
        result["result"]["batches"][0]["sections_stored"],
        json!(1),
        "{result}"
    );
    assert_eq!(
        result["result"]["batches"][0]["sections_dropped"],
        json!(1),
        "{result}"
    );
}

/// The import endpoint surfaces a dropped association paragraph locator
/// in its JSON just as the CLI report does: the fact still lands (unlike
/// a dropped question or section, which vanishes whole), only the
/// out-of-range locator is cleared — and counted, never silent.
#[test]
fn the_import_endpoint_reports_dropped_association_paragraphs() {
    let server = Server::start("http-import-assoc-drop");
    let batch = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-guide\", \
                 \"create\": {\"description\": \"d\"}}\n\
                 {\"passage\": \"蔵の杜氏は高瀬。\\n\\n創業は1907年。\"}\n\
                 {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0, \"paragraph\": 9}\n";

    let (status, result) = post_import(&server, batch, None);
    assert_eq!(status, 200, "{result}");
    assert_eq!(
        result["result"]["batches"][0]["associations"],
        json!(1),
        "the fact lands: {result}"
    );
    assert_eq!(
        result["result"]["batches"][0]["association_paragraphs_dropped"],
        json!(1),
        "the out-of-range locator is cleared and counted: {result}"
    );
}

/// The backup loop, live: build a context over the API, pull it back
/// out at GET /contexts/{name}/export, delete the context, and restore
/// it by POSTing the stream to /import — facts, aliases, passages,
/// questions, and sections all round-trip, and the stream response
/// reports one outcome per batch.
#[test]
fn a_context_round_trips_through_the_export_endpoint_and_import() {
    let server = Server::start("http-export-roundtrip");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "酒蔵の知識", "dice_floor": 0.25})),
    );
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "青嶺酒造", "label": "代表銘柄", "object": "青嶺",
             "weight": 1.0, "source": "a.md", "paragraph": 0},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬",
             "weight": 2.0, "source": "b.md"},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {"a.md": "青嶺酒造の紹介。\n\n代表銘柄は青嶺。"},
            "questions": {"a.md": [{"paragraph": 0, "question": "どこの蔵?"}]},
            "sections": {"a.md": [{"paragraph": 0, "section": "概要"}]},
        })),
    );
    server.ok(
        "POST",
        "/contexts/sake/aliases",
        Some(json!({"concepts": {"Aomine": "青嶺酒造"}})),
    );

    let (status, exported) = server.call("GET", "/contexts/sake/export", None);
    assert_eq!(status, 200, "{exported}");
    let stream = exported
        .as_str()
        .expect("the export body is JSONL, not the envelope");
    // The doc2query question rides the export stream (its round trip is
    // otherwise invisible in a lexical-only test).
    assert!(
        stream.contains("どこの蔵?"),
        "export must emit the question: {stream}"
    );
    assert_eq!(
        stream.matches("\"taguru_batch\":1").count(),
        2,
        "one batch per source: {stream}"
    );
    assert!(
        stream.contains("\"description\":\"酒蔵の知識\""),
        "{stream}"
    );

    // Exporting a context that does not exist is the ordinary 404.
    let (status, missing) = server.call("GET", "/contexts/ghost/export", None);
    assert_eq!(status, 404, "{missing}");

    server.ok("DELETE", "/contexts/sake", None);
    let (status, restored) = post_import(&server, stream, None);
    assert_eq!(status, 200, "{restored}");
    let outcomes = restored["result"]["batches"]
        .as_array()
        .expect("a stream answers one outcome per batch");
    assert_eq!(outcomes.len(), 2, "{restored}");
    assert_eq!(outcomes[0]["created"], json!(true), "{restored}");

    let facts = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造"})),
    );
    assert_eq!(facts["total"], json!(2), "{facts}");
    let citation = server.ok(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "a.md", "paragraph": 0})),
    );
    assert_eq!(citation["text"], json!("青嶺酒造の紹介。"), "{citation}");
    assert_eq!(citation["section"], json!("概要"), "{citation}");
    let aliases = server.ok("GET", "/contexts/sake/aliases", None);
    assert_eq!(
        aliases["concepts"]["Aomine"],
        json!("青嶺酒造"),
        "{aliases}"
    );
    let row = server.ok("GET", "/contexts/sake", None);
    assert_eq!(row["dice_floor"], json!(0.25), "{row}");
    // The question survived the delete+restore: re-exporting the
    // restored context still carries it.
    let (status, re_exported) = server.call("GET", "/contexts/sake/export", None);
    assert_eq!(status, 200);
    assert!(
        re_exported.as_str().unwrap().contains("どこの蔵?"),
        "the question must survive the round trip: {re_exported}"
    );

    // Restoring over the restored context is a per-source replace, not
    // a doubling — the same idempotency the CLI import promises.
    let (status, again) = post_import(&server, stream, None);
    assert_eq!(status, 200, "{again}");
    let facts = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造"})),
    );
    assert_eq!(facts["total"], json!(2), "no doubling: {facts}");
}

/// Group records ride the import stream and restore AFTER every batch
/// — one body can create the member contexts and the groups bundling
/// them, in any order — as a create-or-replace of the whole record,
/// reported under `groups: [...]` (absent when the stream carried
/// none, so the pre-group response shape is untouched).
#[test]
fn import_restores_group_records_after_the_batches() {
    let server = Server::start("http-import-groups");
    // The group records sit FIRST: apply order is batches-then-groups,
    // not stream order — and `kura` names `kid`, which only this same
    // stream brings.
    let stream = "{\"taguru_group\": 1, \"name\": \"kura\", \"description\": \"蔵まとめ\", \
                   \"contexts\": [\"sake\", \"bunko\"], \"groups\": [\"kid\"]}\n\
                  {\"taguru_group\": 1, \"name\": \"kid\", \"contexts\": [\"bunko\"]}\n\
                  {\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
                   \"create\": {\"description\": \"d\"}}\n\
                  {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n\
                  {\"taguru_batch\": 1, \"context\": \"bunko\", \"source\": \"b.md\", \
                   \"create\": {\"description\": \"d\"}}\n";
    let (status, first) = post_import(&server, stream, None);
    assert_eq!(status, 200, "{first}");
    assert_eq!(first["result"]["batches"].as_array().map(Vec::len), Some(2));
    let restored = first["result"]["groups"]
        .as_array()
        .expect("group outcomes ride the response");
    assert_eq!(restored.len(), 2, "{first}");
    assert_eq!(restored[0]["name"], json!("kura"), "{first}");
    assert_eq!(restored[0]["outcome"], json!("created"), "{first}");
    assert_eq!(restored[0]["contexts"], json!(2), "{first}");
    assert_eq!(restored[0]["groups"], json!(1), "{first}");
    let row = server.ok("GET", "/groups/kura", None);
    assert_eq!(row["contexts"], json!(["bunko", "sake"]), "{row}");
    assert_eq!(row["groups"], json!(["kid"]), "{row}");
    assert_eq!(row["description"], json!("蔵まとめ"), "{row}");

    // Re-POSTing converges: the records already stand.
    let (status, second) = post_import(&server, stream, None);
    assert_eq!(status, 200, "{second}");
    assert_eq!(
        second["result"]["groups"][0]["outcome"],
        json!("unchanged"),
        "{second}"
    );

    // A stream with no group records keeps the pre-group shape: no
    // `groups` field at all.
    let batches_only = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"c.md\"}\n";
    let (status, plain) = post_import(&server, batches_only, None);
    assert_eq!(status, 200, "{plain}");
    assert!(plain["result"].get("groups").is_none(), "{plain}");

    // A restore REPLACES the record: whatever it omits drops.
    let shrunk = "{\"taguru_group\": 1, \"name\": \"kura\", \"contexts\": [\"sake\"]}\n";
    let (status, third) = post_import(&server, shrunk, None);
    assert_eq!(status, 200, "{third}");
    assert_eq!(
        third["result"]["groups"][0]["outcome"],
        json!("replaced"),
        "{third}"
    );
    let row = server.ok("GET", "/groups/kura", None);
    assert_eq!(row["contexts"], json!(["sake"]), "{row}");
    assert_eq!(row["groups"], json!([]), "{row}");
    assert_eq!(
        row["description"],
        json!(""),
        "a replace is the whole record: {row}"
    );
}

/// A group record that would dangle or misshape refuses every group
/// record — with the batches already durable — under the API's usual
/// codes: `no_context` for a missing member, `no_group` for a missing
/// child, `invalid_argument` for a cycle.
#[test]
fn import_refuses_group_records_that_would_dangle_or_misshape() {
    let server = Server::start("http-import-group-refuse");
    let stream = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
                   \"create\": {\"description\": \"d\"}}\n\
                  {\"taguru_group\": 1, \"name\": \"kura\", \"contexts\": [\"sake\", \"ghost\"]}\n";
    let (status, refusal) = post_import(&server, stream, None);
    assert_eq!(status, 404, "{refusal}");
    assert_eq!(refusal["code"], json!("no_context"), "{refusal}");
    assert!(
        refusal["error"].as_str().unwrap().contains("ghost"),
        "{refusal}"
    );
    // The batch landed; no group did.
    let (status, _) = server.call("GET", "/contexts/sake", None);
    assert_eq!(
        status, 200,
        "the batches before the group refusal are durable"
    );
    let (status, gone) = server.call("GET", "/groups/kura", None);
    assert_eq!(status, 404, "{gone}");

    // A child that neither exists nor rides the stream: no_group.
    let stream = "{\"taguru_group\": 1, \"name\": \"kura\", \"contexts\": [\"sake\"], \
                   \"groups\": [\"nope\"]}\n";
    let (status, refusal) = post_import(&server, stream, None);
    assert_eq!(status, 404, "{refusal}");
    assert_eq!(refusal["code"], json!("no_group"), "{refusal}");

    // A cycle the incoming set closes with itself: the request's own
    // shape, 400.
    let stream = "{\"taguru_group\": 1, \"name\": \"a\", \"groups\": [\"b\"]}\n\
                  {\"taguru_group\": 1, \"name\": \"b\", \"groups\": [\"a\"]}\n";
    let (status, refusal) = post_import(&server, stream, None);
    assert_eq!(status, 400, "{refusal}");
    assert_eq!(refusal["code"], json!("invalid_argument"), "{refusal}");

    // Restating one group in one stream is a parse-stage refusal.
    let stream = "{\"taguru_group\": 1, \"name\": \"a\"}\n{\"taguru_group\": 1, \"name\": \"a\"}\n";
    let (status, refusal) = post_import(&server, stream, None);
    assert_eq!(status, 400, "{refusal}");
    assert!(
        refusal["error"]
            .as_str()
            .unwrap()
            .contains("one record owns one group's truth"),
        "{refusal}"
    );
}

/// A scoped key's group records are judged like any group write — by
/// the transitive context closure, the standing record's and the
/// prospective one's both — before anything at all applies.
#[test]
fn a_scoped_key_cannot_import_group_records_beyond_its_grant() {
    let server = Server::start_with_env(
        "http-import-group-scope",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,curator:ctok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"curator": {"role": "admin", "contexts": ["sake"]}}"#,
            ),
        ],
    );
    let call = |method: &str, path: &str, body: Option<Value>, token: &str| {
        server.call_with_token(method, path, body, Some(token))
    };
    assert_eq!(
        call(
            "PUT",
            "/contexts/sake",
            Some(json!({"description": "d"})),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "PUT",
            "/contexts/bunko",
            Some(json!({"description": "d"})),
            "atok"
        )
        .0,
        200
    );

    // Out of grant through the record's own members: the whole request
    // refuses — the in-grant batch beside it included.
    let stream = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s.md\"}\n\
                  {\"taguru_group\": 1, \"name\": \"kura\", \"contexts\": [\"sake\", \"bunko\"]}\n";
    let (status, refusal) = post_import(&server, stream, Some("ctok"));
    assert_eq!(status, 403, "{refusal}");
    assert!(
        refusal["error"].as_str().unwrap().contains("bunko"),
        "{refusal}"
    );
    let (status, _) = call("GET", "/groups/kura", None, "atok");
    assert_eq!(status, 404, "the refusal precedes every apply");
    let (_, sources) = call("GET", "/contexts/sake/sources", None, "atok");
    assert!(
        !sources.to_string().contains("s.md"),
        "the batch must not land either: {sources}"
    );

    // Inside the grant the same key restores normally.
    let stream = "{\"taguru_group\": 1, \"name\": \"kura\", \"contexts\": [\"sake\"]}\n";
    let (status, applied) = post_import(&server, stream, Some("ctok"));
    assert_eq!(status, 200, "{applied}");

    // The replace side is judged too: shrinking a standing group that
    // bundles an out-of-grant member would release that member, so the
    // scoped replace refuses.
    let wide = "{\"taguru_group\": 1, \"name\": \"wide\", \"contexts\": [\"sake\", \"bunko\"]}\n";
    let (status, seeded) = post_import(&server, wide, Some("atok"));
    assert_eq!(status, 200, "{seeded}");
    let shrink = "{\"taguru_group\": 1, \"name\": \"wide\", \"contexts\": [\"sake\"]}\n";
    let (status, refusal) = post_import(&server, shrink, Some("ctok"));
    assert_eq!(status, 403, "{refusal}");
}

/// The destination of `POST /contexts/{name}/rename` rides in the
/// body, out of the authorization middleware's reach — so a
/// context-scoped key must not be able to move its data to a name
/// beyond its grant, whether that name is already someone else's
/// context or brand new.
#[test]
fn a_scoped_key_cannot_rename_a_context_to_a_destination_beyond_its_grant() {
    let server = Server::start_with_env(
        "http-rename-scope",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,curator:ctok,wide:wtok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"curator": {"role": "admin", "contexts": ["sake"]},
                    "wide": {"role": "admin", "contexts": ["sake", "shochu"]}}"#,
            ),
        ],
    );
    let call = |method: &str, path: &str, body: Option<Value>, token: &str| {
        server.call_with_token(method, path, body, Some(token))
    };
    assert_eq!(
        call(
            "PUT",
            "/contexts/sake",
            Some(json!({"description": "d"})),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "PUT",
            "/contexts/bunko",
            Some(json!({"description": "d"})),
            "atok"
        )
        .0,
        200
    );

    // Destination already exists, but beyond the grant: refused, the
    // message naming the destination so the caller knows what to fix.
    let (status, refusal) = call(
        "POST",
        "/contexts/sake/rename",
        Some(json!({"to": "bunko"})),
        "ctok",
    );
    assert_eq!(status, 403, "{refusal}");
    assert!(
        refusal["error"].as_str().unwrap().contains("bunko"),
        "{refusal}"
    );

    // Destination is brand new, still beyond the grant: refused the
    // same way — existence is not what is being checked.
    let (status, refusal) = call(
        "POST",
        "/contexts/sake/rename",
        Some(json!({"to": "shochu"})),
        "ctok",
    );
    assert_eq!(status, 403, "{refusal}");

    // Neither refusal moved anything.
    assert_eq!(call("GET", "/contexts/sake", None, "atok").0, 200);
    assert_eq!(call("GET", "/contexts/shochu", None, "atok").0, 404);

    // A key scoped to BOTH names may rename between them — the grant
    // check is about the names involved, not a blanket ban.
    let (status, applied) = call(
        "POST",
        "/contexts/sake/rename",
        Some(json!({"to": "shochu"})),
        "wtok",
    );
    assert_eq!(status, 200, "{applied}");
    assert_eq!(call("GET", "/contexts/shochu", None, "atok").0, 200);
    assert_eq!(call("GET", "/contexts/sake", None, "atok").0, 404);
}

/// `GET /groups/{name}/export` serves one `taguru_group` record that
/// `POST /import` restores whole — and a scoped key exports exactly
/// the slice its grant lets it read.
#[test]
fn a_group_exports_as_one_import_record() {
    let server = Server::start_with_env(
        "http-group-export",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,curator:ctok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"curator": {"role": "read", "contexts": ["sake"]}}"#,
            ),
        ],
    );
    let call = |method: &str, path: &str, body: Option<Value>, token: &str| {
        server.call_with_token(method, path, body, Some(token))
    };
    assert_eq!(
        call(
            "PUT",
            "/contexts/sake",
            Some(json!({"description": "d"})),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(
        call(
            "PUT",
            "/contexts/bunko",
            Some(json!({"description": "d"})),
            "atok"
        )
        .0,
        200
    );
    assert_eq!(call("PUT", "/groups/kid", Some(json!({})), "atok").0, 200);
    assert_eq!(
        call(
            "PUT",
            "/groups/kura",
            Some(
                json!({"description": "蔵まとめ", "contexts": ["sake", "bunko"],
                        "groups": ["kid"]})
            ),
            "atok"
        )
        .0,
        200
    );

    // One JSONL line is itself valid JSON, so the harness hands it
    // back parsed — assert the record's shape (the byte-exact
    // rendering, field order and omitted empties, is pinned by
    // export's own unit test).
    let (status, exported) = call("GET", "/groups/kura/export", None, "atok");
    assert_eq!(status, 200, "{exported}");
    assert_eq!(
        exported,
        json!({"taguru_group": 1, "name": "kura", "description": "蔵まとめ",
               "contexts": ["bunko", "sake"], "groups": ["kid"]})
    );

    // Deleting and re-importing the record restores the group whole.
    assert_eq!(call("DELETE", "/groups/kura", None, "atok").0, 200);
    let line = format!("{exported}\n");
    let (status, restored) = post_import(&server, &line, Some("atok"));
    assert_eq!(status, 200, "{restored}");
    assert_eq!(
        restored["result"]["groups"][0]["outcome"],
        json!("created"),
        "{restored}"
    );
    let (_, row) = call("GET", "/groups/kura", None, "atok");
    assert_eq!(row["result"]["contexts"], json!(["bunko", "sake"]), "{row}");

    // A scoped key exports its grant's slice — the row it can read IS
    // the record it takes away.
    let (status, sliced) = call("GET", "/groups/kura/export", None, "ctok");
    assert_eq!(status, 200, "{sliced}");
    assert_eq!(sliced["contexts"], json!(["sake"]), "{sliced}");

    // Unknown group: the ordinary 404.
    let (status, missing) = call("GET", "/groups/ghost/export", None, "atok");
    assert_eq!(status, 404, "{missing}");
}

#[test]
fn the_import_endpoint_refuses_with_the_cli_wording_and_api_statuses() {
    let server = Server::start("http-import-refuse");

    // Malformed op line: 400, named by line number.
    let (status, refusal) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"s\"}\n\n{\"foo\": 1}\n",
        None,
    );
    assert_eq!(status, 400);
    assert!(
        refusal["error"].as_str().unwrap().contains("line 3"),
        "{refusal}"
    );

    // Absent context, no create block: 404.
    let (status, refusal) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"ghost\", \"source\": \"s\"}\n",
        None,
    );
    assert_eq!(status, 404);
    assert!(
        refusal["error"].as_str().unwrap().contains("create block"),
        "{refusal}"
    );

    // Re-pointing an existing alias is the API's usual conflict: 409,
    // predicted before anything in the batch runs — its association
    // never lands either.
    let (status, _) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"s1\", \"create\": {}}\n\
         {\"subject\": \"X\", \"label\": \"l\", \"object\": \"Z\", \"weight\": 1.0}\n\
         {\"alias\": \"A\", \"canonical\": \"X\", \"kind\": \"concept\"}\n",
        None,
    );
    assert_eq!(status, 200);
    let (status, refusal) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"s2\"}\n\
         {\"subject\": \"Y\", \"label\": \"l\", \"object\": \"Z\", \"weight\": 1.0}\n\
         {\"alias\": \"A\", \"canonical\": \"Y\", \"kind\": \"concept\"}\n",
        None,
    );
    assert_eq!(status, 409, "{refusal}");
    assert!(
        refusal["error"]
            .as_str()
            .unwrap()
            .contains("nothing was applied"),
        "{refusal}"
    );
}

#[test]
fn aliases_withdraw_and_the_spelling_is_reusable() {
    let server = Server::start("alias-remove");
    server.ok("PUT", "/contexts/c", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/c/associations",
        Some(json!([
            {"subject": "X", "label": "l", "object": "Z", "weight": 1.0},
            {"subject": "Y", "label": "l", "object": "Z", "weight": 1.0},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/c/aliases",
        Some(json!({"concepts": {"A": "X"}})),
    );

    let removed = server.ok(
        "DELETE",
        "/contexts/c/aliases",
        Some(json!({"concepts": ["A"]})),
    );
    assert_eq!(removed, json!(1));
    let listing = server.ok("GET", "/contexts/c/aliases", None);
    assert_eq!(listing["concepts"], json!({}));

    // The spelling is free to point elsewhere — the un-wedging move.
    server.ok(
        "POST",
        "/contexts/c/aliases",
        Some(json!({"concepts": {"A": "Y"}})),
    );
    let via = server.ok("POST", "/contexts/c/query", Some(json!({"subject": "A"})));
    assert_eq!(via["matches"][0]["subject"], json!("Y"));

    // Refusals: absent spellings and canonical names are conflicts,
    // and an empty withdrawal is malformed rather than a silent no-op.
    let (status, body) = server.call(
        "DELETE",
        "/contexts/c/aliases",
        Some(json!({"concepts": ["ghost"]})),
    );
    assert_eq!(status, 409, "{body}");
    let (status, _) = server.call(
        "DELETE",
        "/contexts/c/aliases",
        Some(json!({"concepts": ["X"]})),
    );
    assert_eq!(status, 409);
    let (status, _) = server.call("DELETE", "/contexts/c/aliases", Some(json!({})));
    assert_eq!(status, 400);
}

#[test]
fn an_import_alias_conflict_heals_with_a_withdrawal_then_reimport() {
    let server = Server::start("alias-heal");
    let (status, _) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"s1\", \"create\": {}}\n\
         {\"subject\": \"X\", \"label\": \"l\", \"object\": \"Z\", \"weight\": 1.0}\n\
         {\"alias\": \"A\", \"canonical\": \"X\", \"kind\": \"concept\"}\n",
        None,
    );
    assert_eq!(status, 200);
    let revised = "{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"s2\"}\n\
         {\"subject\": \"Y\", \"label\": \"l\", \"object\": \"Z\", \"weight\": 1.0}\n\
         {\"alias\": \"A\", \"canonical\": \"Y\", \"kind\": \"concept\"}\n";
    let (status, _) = post_import(&server, revised, None);
    assert_eq!(status, 409);

    // The heal the import docs prescribe: withdraw the old
    // registration deliberately, then re-import — retract-then-apply
    // makes the second attempt exact.
    server.ok(
        "DELETE",
        "/contexts/c/aliases",
        Some(json!({"concepts": ["A"]})),
    );
    let (status, body) = post_import(&server, revised, None);
    assert_eq!(status, 200, "{body}");
    let listing = server.ok("GET", "/contexts/c/aliases", None);
    assert_eq!(listing["concepts"]["A"], json!("Y"));
}

/// A REPLACEMENT batch (same source name as one already on file) that
/// is refused by a predicted alias conflict must not even reach the
/// retraction step — the old passage, its association, and the
/// context's aliases (which live independently of any one source) all
/// stay exactly as they were.
#[test]
fn a_rejected_replacement_batch_leaves_the_old_version_of_its_source_untouched() {
    let server = Server::start("http-import-reject-replacement");
    let (status, _) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\", \
         \"create\": {\"description\": \"d\"}}\n\
         {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n\
         {\"passage\": \"元の文章。\"}\n\
         {\"alias\": \"kyo\", \"canonical\": \"蔵\", \"kind\": \"concept\"}\n",
        None,
    );
    assert_eq!(status, 200);

    // Same source, a revised truth — but its alias step re-points
    // "kyo" from the existing 蔵 to a name this very batch freshly
    // interns, a predicted conflict that must refuse before the
    // retraction that would otherwise clear the old passage/association.
    let (status, body) = post_import(
        &server,
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\"}\n\
         {\"subject\": \"新蔵\", \"label\": \"特徴\", \"object\": \"辛口\", \"weight\": 1.0}\n\
         {\"passage\": \"新しい文章。\"}\n\
         {\"alias\": \"kyo\", \"canonical\": \"新蔵\", \"kind\": \"concept\"}\n",
        None,
    );
    assert_eq!(status, 409, "{body}");

    // The old passage is exactly as it was — never retracted, never
    // overwritten.
    let passages = server.ok(
        "POST",
        "/contexts/sake/sources/lookup",
        Some(json!({"sources": ["doc-1"]})),
    );
    assert_eq!(passages["passages"]["doc-1"], json!("元の文章。"));

    // The old association still stands...
    let old = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "蔵", "label": "杜氏"})),
    );
    assert_eq!(old["matches"][0]["object"], json!("高瀬"));

    // ...and the replacement's own association never landed.
    let new = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "新蔵", "label": "特徴"})),
    );
    assert_eq!(new["matches"], json!([]));

    // The alias, which lives independently of any one source, is
    // untouched too.
    let aliases = server.ok("GET", "/contexts/sake/aliases", None);
    assert_eq!(aliases["concepts"]["kyo"], json!("蔵"));
}

#[test]
fn importing_into_an_absent_context_needs_a_create_block() {
    let batches = batch_dir("import-nocreate");
    let file = batches.join("orphan.jsonl");
    std::fs::write(
        &file,
        "{\"taguru_batch\": 1, \"context\": \"ghost\", \"source\": \"s\"}\n",
    )
    .unwrap();
    let data_dir = std::env::temp_dir().join(format!(
        "taguru-http-import-nocreate-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&data_dir);
    let (code, _, stderr) = run_import(&data_dir, &[file.to_str().unwrap()]);
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("no create block"), "{stderr}");
    let _ = std::fs::remove_dir_all(&batches);
    let _ = std::fs::remove_dir_all(&data_dir);
}
