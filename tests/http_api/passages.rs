//! Passage storage, citation, and paragraph-level search.

use serde_json::json;

use crate::support::*;

/// The hybrid-search wire contract on a lexical-only server: hits are
/// paragraph-granular, every hit carries its lane evidence, the
/// top-level score stays the raw BM25 number (no semantic lane ran),
/// and the vector key is absent rather than null.
#[test]
fn passage_search_serves_paragraph_hits_with_lane_evidence() {
    let server = Server::start("passage-lanes");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "蔵の知識"})),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {
            "docs/aomine.md": "青嶺酒造は雲居県霧沢町の蔵元である。\n\n\
                原料米には山田錦を使い、精米歩合は50パーセントまで磨く。"
        }})),
    );

    let hits = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "精米歩合はどこまで磨く?", "limit": 3})),
    );
    let hit = &hits[0];
    assert_eq!(hit["source"], "docs/aomine.md");
    assert_eq!(hit["paragraph"], 1, "the hit names the answering PARAGRAPH");
    assert!(hit["text"].as_str().unwrap().starts_with("原料米"), "{hit}");
    assert_eq!(hit["lanes"]["bm25"]["rank"], 1, "{hit}");
    assert_eq!(
        hit["score"], hit["lanes"]["bm25"]["score"],
        "lexical-only deployments keep raw BM25 score semantics"
    );
    assert!(
        hit["lanes"].get("vector").is_none(),
        "no provider, no vector key: {hit}"
    );

    // A zero limit asks for nothing and gets nothing.
    let none = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "精米", "limit": 0})),
    );
    assert_eq!(none.as_array().unwrap().len(), 0);
}

/// The citation endpoint's wire contract: given a known (source,
/// paragraph), it returns the exact verbatim excerpt `search_passages`
/// would show for that same paragraph — sliced through the one shared
/// `PassageRecord::paragraph` accessor, so the two can never disagree —
/// plus the source, and `section` always present as a key (never
/// omitted). This source stored no sections, so it resolves to `null`;
/// `citation_resolves_the_section_governing_its_paragraph` covers the
/// case where a section is actually stored.
#[test]
fn citation_returns_the_verbatim_paragraph_named_by_source_and_paragraph() {
    let server = Server::start("citation-hit");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "蔵の知識"})),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {
            "docs/aomine.md": "青嶺酒造は雲居県霧沢町の蔵元である。\n\n\
                原料米には山田錦を使い、精米歩合は50パーセントまで磨く。"
        }})),
    );

    let citation = server.ok(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "docs/aomine.md", "paragraph": 1})),
    );
    assert_eq!(
        citation["text"], "原料米には山田錦を使い、精米歩合は50パーセントまで磨く。",
        "{citation}"
    );
    assert_eq!(citation["source"], "docs/aomine.md");
    assert!(
        citation.as_object().unwrap().contains_key("section") && citation["section"].is_null(),
        "section is present and null, never omitted: {citation}"
    );
}

/// Unknown source, an out-of-range paragraph position, and an unknown
/// context all speak the same `ApiError` shape (never a panic), each
/// with a message naming what was not found.
#[test]
fn citation_reports_clear_errors_for_unknown_source_paragraph_and_context() {
    let server = Server::start("citation-miss");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "蔵の知識"})),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"docs/aomine.md": "一段落だけ。"}})),
    );

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "docs/ghost.md", "paragraph": 0})),
    );
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
    assert!(
        body["error"].as_str().unwrap().contains("docs/ghost.md"),
        "{body}"
    );

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "docs/aomine.md", "paragraph": 9})),
    );
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
    assert!(
        body["error"].as_str().unwrap().contains("docs/aomine.md"),
        "{body}"
    );

    let (status, body) = server.call(
        "POST",
        "/contexts/ghost/citations",
        Some(json!({"source": "docs/aomine.md", "paragraph": 0})),
    );
    assert_eq!(status, 404, "{body}");
    assert_eq!(body["status"], json!("error"), "{body}");
    assert!(body["error"].as_str().unwrap().contains("ghost"), "{body}");
}

/// A section stored via import resolves on the citation endpoint too:
/// `AppState::citation` reads it off the very same `PassageRecord` via
/// `section_for`, the accessor `resolve_sections` uses for association
/// attributions, so both report the same label for the same paragraph.
/// The paragraph preceding the first marker still resolves to `null`.
#[test]
fn citation_resolves_the_section_governing_its_paragraph() {
    let server = Server::start("citation-section");
    let batch = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-sections\", \
                 \"create\": {\"description\": \"d\"}}\n\
                 {\"passage\": \"蔵の杜氏は高瀬。\\n\\n創業は1907年。\"}\n\
                 {\"paragraph\": 1, \"section\": \"沿革\"}\n";
    let (status, result) = post_import(&server, batch, None);
    assert_eq!(status, 200, "{result}");
    assert_eq!(
        result["result"]["batches"][0]["sections_stored"],
        json!(1),
        "{result}"
    );

    let before = server.ok(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "doc-sections", "paragraph": 0})),
    );
    assert_eq!(before["text"], "蔵の杜氏は高瀬。", "{before}");
    assert!(
        before["section"].is_null(),
        "paragraph 0 precedes the first marker: {before}"
    );

    let after = server.ok(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "doc-sections", "paragraph": 1})),
    );
    assert_eq!(after["text"], "創業は1907年。", "{after}");
    assert_eq!(after["section"], json!("沿革"), "{after}");
}

/// doc2query over HTTP: questions ride the store request per source,
/// out-of-range ones are dropped with their count reported (never
/// failing the passage), and questions for a source the request does
/// not carry are refused outright.
#[test]
fn store_passages_accepts_questions_and_reports_the_bookkeeping() {
    let server = Server::start("passage-questions");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "蔵の知識"})),
    );
    let result = server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {"doc": "一つ目。\n\n二つ目。"},
            "questions": {"doc": [
                {"paragraph": 1, "question": "二つ目は何?"},
                {"paragraph": 9, "question": "存在しない段落への質問?"}
            ]}
        })),
    );
    assert_eq!(result["stored"], 1, "{result}");
    assert_eq!(result["questions_stored"], 1, "{result}");
    assert_eq!(result["questions_dropped"], 1, "{result}");

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {},
            "questions": {"ghost": [{"paragraph": 0, "question": "誰の質問?"}]}
        })),
    );
    assert_eq!(status, 400, "{body}");
}

/// Section markers over HTTP: they ride the store request per source
/// exactly like questions, out-of-range ones are dropped with their
/// count reported (never failing the passage), and sections for a
/// source the request does not carry are refused outright.
#[test]
fn store_passages_accepts_sections_and_reports_the_bookkeeping() {
    let server = Server::start("passage-sections");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "蔵の知識"})),
    );
    let result = server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {"doc": "一つ目。\n\n二つ目。"},
            "sections": {"doc": [
                {"paragraph": 1, "section": "沿革"},
                {"paragraph": 9, "section": "存在しない段落"}
            ]}
        })),
    );
    assert_eq!(result["stored"], 1, "{result}");
    assert_eq!(result["sections_stored"], 1, "{result}");
    assert_eq!(result["sections_dropped"], 1, "{result}");

    let (status, body) = server.call(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {},
            "sections": {"ghost": [{"paragraph": 0, "section": "幽霊"}]}
        })),
    );
    assert_eq!(status, 400, "{body}");
}

/// The live path proves the same resolution as import: a section set
/// via `POST /sources` (not the batch importer) governs its paragraph
/// on the citation endpoint too, the same way
/// `citation_resolves_the_section_governing_its_paragraph` proves it
/// for import — and the paragraph preceding the first marker still
/// resolves to `null`.
#[test]
fn a_section_stored_via_store_passages_resolves_on_citation() {
    let server = Server::start("store-passages-citation-section");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "蔵の知識"})),
    );
    let result = server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {"doc-sections": "蔵の杜氏は高瀬。\n\n創業は1907年。"},
            "sections": {"doc-sections": [{"paragraph": 1, "section": "沿革"}]}
        })),
    );
    assert_eq!(result["sections_stored"], 1, "{result}");

    let before = server.ok(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "doc-sections", "paragraph": 0})),
    );
    assert_eq!(before["text"], "蔵の杜氏は高瀬。", "{before}");
    assert!(
        before["section"].is_null(),
        "paragraph 0 precedes the first marker: {before}"
    );

    let after = server.ok(
        "POST",
        "/contexts/sake/citations",
        Some(json!({"source": "doc-sections", "paragraph": 1})),
    );
    assert_eq!(after["text"], "創業は1907年。", "{after}");
    assert_eq!(after["section"], json!("沿革"), "{after}");
}

/// /live is the pure liveness probe: 200 whenever the process answers,
/// unauthenticated, unconditional — /health keeps the readiness
/// (write-path) signal.
#[test]
fn live_answers_unauthenticated_even_with_auth_on() {
    let server = Server::start_with_env("http-live", &[("TAGURU_API_TOKEN", "opskey")]);
    let (status, body) = server.call("GET", "/live", None);
    assert_eq!(status, 200);
    assert_eq!(body, json!("ok"));
}
