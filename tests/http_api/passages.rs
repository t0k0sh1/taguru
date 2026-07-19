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

    let page = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "精米歩合はどこまで磨く?", "limit": 3})),
    );
    let hit = &page["hits"][0];
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

    // The response-level plan (#151): the same story the per-hit
    // evidence cannot tell — the semantic lane never ran here, and why.
    let plan = &page["plan"]["contexts"][0];
    assert_eq!(plan["context"], "sake", "{page}");
    assert_eq!(plan["lanes"]["bm25"]["ran"], json!(true), "{page}");
    assert_eq!(plan["lanes"]["vector"]["ran"], json!(false), "{page}");
    assert_eq!(
        plan["lanes"]["vector"]["reason"],
        json!("no embedding provider is configured"),
        "{page}"
    );

    // A zero limit asks for nothing and gets nothing — but still says
    // what it did (nothing, on both lanes).
    let none = server.ok(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "精米", "limit": 0})),
    );
    assert_eq!(none["hits"].as_array().unwrap().len(), 0);
    assert_eq!(
        none["plan"]["contexts"][0]["lanes"]["bm25"],
        json!({"ran": false, "reason": "the requested limit is 0"}),
        "{none}"
    );
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

/// A minimal OpenAI-style embeddings endpoint on a local port: every
/// `input` text gets a fixed unit vector by content — りんご texts sit
/// on the first axis, みかん queries at cosine 0.28 to them, anything
/// else orthogonal — so floor arithmetic over the real HTTP transport
/// is as deterministic as the registry tests' in-process mock.
fn spawn_fruity_embeddings() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            std::thread::spawn(move || {
                use std::io::{Read, Write};
                let mut buffer = Vec::new();
                let mut chunk = [0u8; 4096];
                let body_start = loop {
                    match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                    }
                    if let Some(at) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                        break at + 4;
                    }
                };
                let headers = String::from_utf8_lossy(&buffer[..body_start]).to_string();
                let length: usize = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse().ok())?
                    })
                    .unwrap_or(0);
                while buffer.len() < body_start + length {
                    match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                    }
                }
                let request: serde_json::Value =
                    serde_json::from_slice(&buffer[body_start..body_start + length]).unwrap();
                let data: Vec<serde_json::Value> = request["input"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|text| {
                        let text = text.as_str().unwrap();
                        let vector = if text.contains("みかん") {
                            let x = 0.28f64;
                            vec![x, (1.0 - x * x).sqrt(), 0.0]
                        } else if text.contains("りんご") {
                            vec![1.0, 0.0, 0.0]
                        } else {
                            vec![0.0, 0.0, 1.0]
                        };
                        json!({ "embedding": vector })
                    })
                    .collect();
                let body = json!({ "data": data }).to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
            });
        }
    });
    format!("http://{addr}/v1/embeddings")
}

/// The per-request `semantic_floor` override rides every search
/// surface — the single-context handler, the explain handler (which
/// must account for the call actually made), and the cross-context
/// fan-out — through a real embedding round trip: a 0.28-cosine
/// paraphrase hidden by the 0.35 default floor is admitted by an
/// override of 0.2, and the explanation names the floor it ran under.
#[test]
fn search_semantic_floor_override_rides_every_search_surface() {
    let provider = spawn_fruity_embeddings();
    let server = Server::start_with_env(
        "passage-floor-override",
        &[
            ("TAGURU_EMBED_URL", provider.as_str()),
            ("TAGURU_EMBED_MODEL", "fruity-model"),
            ("TAGURU_EMBED_PASSAGES", "1"),
        ],
    );
    server.ok(
        "PUT",
        "/contexts/fruit",
        Some(json!({"description": "果樹園"})),
    );
    server.ok(
        "POST",
        "/contexts/fruit/sources",
        Some(json!({"passages": {"docs/apple.md": "りんごは真っ赤に実った。"}})),
    );
    server.ok("POST", "/contexts/fruit/embeddings/refresh", None);

    // Under the default floor the paraphrase stays hidden — and the
    // query shares no bigram with the text, so no lexical rescue. The
    // plan names the floor that hid it: a zero-hit response is now
    // distinguishable from a lane that never ran.
    let hidden = server.ok(
        "POST",
        "/contexts/fruit/sources/search",
        Some(json!({"query": "みかん"})),
    );
    assert_eq!(hidden["hits"].as_array().unwrap().len(), 0, "{hidden}");
    let vector_plan = &hidden["plan"]["contexts"][0]["lanes"]["vector"];
    assert_eq!(vector_plan["ran"], json!(true), "{hidden}");
    assert!(
        (vector_plan["floor"].as_f64().unwrap() - 0.35).abs() < 1e-6,
        "the plan reports the server-default floor: {hidden}"
    );

    // The override admits it: a vector-only hit at cosine 0.28.
    let page = server.ok(
        "POST",
        "/contexts/fruit/sources/search",
        Some(json!({"query": "みかん", "semantic_floor": 0.2})),
    );
    assert_eq!(page["hits"].as_array().unwrap().len(), 1, "{page}");
    let hit = &page["hits"][0];
    assert_eq!(hit["source"], "docs/apple.md");
    assert!(hit["lanes"].get("bm25").is_none(), "no shared term: {hit}");
    let cosine = hit["lanes"]["vector"]["score"].as_f64().unwrap();
    assert!((cosine - 0.28).abs() < 1e-4, "{hit}");
    let overridden = &page["plan"]["contexts"][0]["lanes"]["vector"];
    assert!(
        (overridden["floor"].as_f64().unwrap() - 0.2).abs() < 1e-6,
        "the plan reports the override, not the default: {page}"
    );

    // The explanation reports the floor of the call being explained.
    let explained = server.ok(
        "POST",
        "/contexts/fruit/sources/search/explain",
        Some(json!({"query": "みかん", "source": "docs/apple.md", "semantic_floor": 0.2})),
    );
    assert_eq!(explained["verdict"], json!("served"), "{explained}");
    assert_eq!(explained["vector"]["ran"], json!(true), "{explained}");
    let floor = explained["vector"]["floor"].as_f64().unwrap();
    assert!((floor - 0.2).abs() < 1e-6, "{explained}");

    // The cross-context fan-out forwards the same override, and its
    // plan carries the per-target account under the same key.
    let across = server.ok(
        "POST",
        "/sources/search",
        Some(json!({"contexts": ["fruit"], "query": "みかん", "semantic_floor": 0.2})),
    );
    assert_eq!(across["hits"].as_array().unwrap().len(), 1, "{across}");
    assert_eq!(across["hits"][0]["context"], json!("fruit"), "{across}");
    assert_eq!(
        across["hits"][0]["source"],
        json!("docs/apple.md"),
        "{across}"
    );
    let cross_plan = &across["plan"]["contexts"][0];
    assert_eq!(cross_plan["context"], json!("fruit"), "{across}");
    assert!(
        (cross_plan["lanes"]["vector"]["floor"].as_f64().unwrap() - 0.2).abs() < 1e-6,
        "{across}"
    );

    // Two targets, two effective floors: a context's own setting beats
    // the server default only for itself, and the plan reports each —
    // the resolution chain was per-context all along, invisibly.
    server.ok(
        "PUT",
        "/contexts/veggie",
        Some(json!({"description": "菜園"})),
    );
    server.ok(
        "POST",
        "/contexts/veggie/sources",
        Some(json!({"passages": {"docs/tomato.md": "りんごの隣にトマトが実った。"}})),
    );
    server.ok("POST", "/contexts/veggie/embeddings/refresh", None);
    server.ok(
        "PATCH",
        "/contexts/veggie",
        Some(json!({"semantic_floor": 0.6})),
    );
    let split = server.ok(
        "POST",
        "/sources/search",
        Some(json!({"contexts": ["fruit", "veggie"], "query": "みかん"})),
    );
    let floors: Vec<f64> = split["plan"]["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry["lanes"]["vector"]["floor"].as_f64().unwrap())
        .collect();
    assert!(
        (floors[0] - 0.35).abs() < 1e-6 && (floors[1] - 0.6).abs() < 1e-6,
        "per-context effective floors, in target order: {split}"
    );
}
