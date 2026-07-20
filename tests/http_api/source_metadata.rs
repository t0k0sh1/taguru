//! Source metadata and pre-lane search filters (#167), end to end:
//! store stamps and lists the metadata back, it survives a hard-kill
//! restart and an export → import round trip (stored_at preserved),
//! filtered search serves only the eligible set with an honest plan,
//! the retrieval cache keys on the filter (normalized), and explain
//! names the `filtered_out` verdict.

use serde_json::{Value, json};

use crate::support::*;

/// One counter value scraped off /metrics — the same technique the
/// retrieval-cache tests use; there is no other cache introspection.
fn metric(server: &Server, line_start: &str) -> u64 {
    let (status, body) = server.call("GET", "/metrics", None);
    assert_eq!(status, 200);
    let text = body.as_str().expect("metrics body is text");
    text.lines()
        .find_map(|line| {
            line.strip_prefix(line_start)
                .and_then(|rest| rest.trim().parse::<u64>().ok())
        })
        .unwrap_or_else(|| panic!("metric '{line_start}' not found in scrape"))
}

fn search_cache(server: &Server, outcome: &str) -> u64 {
    metric(
        server,
        &format!("taguru_retrieval_cache_total{{op=\"search_passages\",outcome=\"{outcome}\"}}"),
    )
}

/// Three sources sharing a term so every search ranks them all: one
/// tagged 酒 and dated 1000, one tagged 蔵 and dated 2000, one bare
/// (no tags, no date — its only timestamp is the server's stamp).
fn seed(server: &Server, context: &str) {
    server.ok(
        "PUT",
        &format!("/contexts/{context}"),
        Some(json!({"description": "metadata fixture"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{context}/sources"),
        Some(json!({
            "passages": {
                "a.md": "共通語の資料。\n\n酒の由来について。",
                "b.md": "共通語の資料。\n\n蔵開きの祭りについて。",
                "c.md": "共通語の資料。\n\n杜氏の紹介について。"
            },
            "tags": {"a.md": ["酒"], "b.md": ["蔵"]},
            "dates": {"a.md": 1000, "b.md": 2000}
        })),
    );
}

fn entries_by_name(page: &Value) -> Vec<(String, Value)> {
    page["entries"]
        .as_array()
        .expect("entries must be present")
        .iter()
        .map(|entry| (entry["name"].as_str().unwrap().to_string(), entry.clone()))
        .collect()
}

#[test]
fn store_stamps_metadata_and_lists_it_back() {
    let server = Server::start("meta-list");
    let before = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    seed(&server, "sake");
    let after = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let page = server.ok("GET", "/contexts/sake/sources", None);
    assert_eq!(page["total"], 3);
    assert_eq!(page["sources"], json!(["a.md", "b.md", "c.md"]));
    let entries = entries_by_name(&page);
    assert_eq!(entries.len(), 3, "entries ride the same page window");
    let (name, a) = &entries[0];
    assert_eq!(name, "a.md");
    assert_eq!(a["tags"], json!(["酒"]));
    assert_eq!(a["date"], 1000);
    let stamped = a["stored_at"]
        .as_u64()
        .expect("the server stamps stored_at");
    assert!(
        (before..=after).contains(&stamped),
        "stored_at {stamped} outside [{before}, {after}]"
    );
    let (_, c) = &entries[2];
    assert!(
        c.get("tags").is_none() && c.get("date").is_none(),
        "absent metadata omits its keys: {c}"
    );
    assert!(c["stored_at"].is_u64(), "every fresh store is stamped");

    // The page window applies to entries exactly as to sources.
    let page = server.ok("GET", "/contexts/sake/sources?after=a.md&limit=1", None);
    assert_eq!(page["sources"], json!(["b.md"]));
    assert_eq!(entries_by_name(&page).len(), 1);
    assert_eq!(entries_by_name(&page)[0].0, "b.md");

    // Validation: tags must name a passage in THIS request, stay under
    // the caps, and never be empty strings.
    for (body, needle) in [
        (
            json!({"passages": {"x": "本文"}, "tags": {"ghost": ["酒"]}}),
            "ghost",
        ),
        (
            json!({"passages": {"x": "本文"}, "tags": {"x": [""]}}),
            "empty",
        ),
        (
            json!({"passages": {"x": "本文"}, "tags": {"x": ["t".repeat(129)]}}),
            "128",
        ),
        (
            json!({"passages": {"x": "本文"}, "tags": {"x": (0..33).map(|i| format!("t{i}")).collect::<Vec<_>>()}}),
            "32",
        ),
        (
            json!({"passages": {"x": "本文"}, "dates": {"ghost": 1}}),
            "ghost",
        ),
    ] {
        let (status, answer) = server.call("POST", "/contexts/sake/sources", Some(body));
        assert_eq!(status, 400, "{answer}");
        assert!(
            answer["error"]
                .as_str()
                .unwrap_or_default()
                .contains(needle),
            "expected '{needle}' in {answer}"
        );
    }
}

#[test]
fn metadata_survives_a_restart_and_an_export_import_round_trip() {
    let server = Server::start("meta-durable");
    seed(&server, "sake");
    let first = server.ok("GET", "/contexts/sake/sources", None);

    // Hard kill: whatever survives comes off the disk alone (the WAL —
    // nothing here waited for a flush tick or a compaction).
    let data_dir = server.stop_hard();
    let server = Server::start_on("meta-durable", data_dir);
    let replayed = server.ok("GET", "/contexts/sake/sources", None);
    assert_eq!(
        replayed["entries"], first["entries"],
        "WAL replay must reproduce the metadata, stored_at stamp included"
    );

    // Export carries the metadata on the passage line…
    let (status, body) = server.call("GET", "/contexts/sake/export", None);
    assert_eq!(status, 200);
    let stream = body.as_str().expect("an export is a JSON Lines body");
    let passage_line = stream
        .lines()
        .find(|line| line.contains("\"passage\"") && line.contains("酒の由来"))
        .expect("a.md's passage line");
    for needle in ["\"stored_at\":", "\"date\":1000", "\"tags\":[\"酒\"]"] {
        assert!(
            passage_line.contains(needle),
            "missing {needle}: {passage_line}"
        );
    }

    // …and import preserves it — stored_at especially: a restore must
    // not re-date the corpus to its restore time.
    let restored = Server::start("meta-restored");
    let (status, outcome) = post_import(&restored, stream, None);
    assert_eq!(status, 200, "{outcome}");
    let after_import = restored.ok("GET", "/contexts/sake/sources", None);
    assert_eq!(
        after_import["entries"], first["entries"],
        "an export → import round trip is metadata-lossless"
    );

    // The batch format validates tags with the same vocabulary as the
    // HTTP store: an empty tag refuses the batch, naming the line.
    let bad = concat!(
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"bad.md\"}\n",
        "{\"passage\": \"本文。\", \"tags\": [\"\"]}\n",
    );
    let (status, refusal) = post_import(&restored, bad, None);
    assert_ne!(status, 200, "{refusal}");
    assert!(
        refusal.to_string().contains("tag"),
        "the refusal names the tag: {refusal}"
    );
}

#[test]
fn filtered_search_serves_only_eligible_sources_with_an_honest_plan() {
    let server = Server::start("meta-filter");
    seed(&server, "sake");
    let search = |body: Value| server.ok("POST", "/contexts/sake/sources/search", Some(body));
    let hit_sources = |page: &Value| -> Vec<String> {
        page["hits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|hit| hit["source"].as_str().unwrap().to_string())
            .collect()
    };

    // Unfiltered: all three rank, and the plan carries no filter block.
    let all = search(json!({"query": "共通語の資料"}));
    assert_eq!(hit_sources(&all).len(), 3);
    assert!(
        all["plan"]["contexts"][0].get("filter").is_none(),
        "no filter requested → no filter block: {all}"
    );

    // Tags are any-of; the plan reports the eligibility split.
    let tagged = search(json!({"query": "共通語の資料", "tags": ["酒"]}));
    assert_eq!(hit_sources(&tagged), vec!["a.md"]);
    assert_eq!(
        tagged["plan"]["contexts"][0]["filter"],
        json!({"eligible_sources": 1, "total_sources": 3})
    );
    let either = search(json!({"query": "共通語の資料", "tags": ["酒", "蔵"]}));
    assert_eq!(hit_sources(&either).len(), 2);

    // A tag nothing carries: the bare source has no tags, so the page
    // is honestly empty with the split to prove it.
    let none = search(json!({"query": "共通語の資料", "tags": ["杜氏"]}));
    assert!(hit_sources(&none).is_empty());
    assert_eq!(
        none["plan"]["contexts"][0]["filter"],
        json!({"eligible_sources": 0, "total_sources": 3})
    );

    // The window is half-open over date ?? stored_at: `since` is
    // inclusive, `until` exclusive, and the bare source's stamp (now)
    // sits far past both fixture dates.
    let until = search(json!({"query": "共通語の資料", "until": 2000}));
    assert_eq!(hit_sources(&until), vec!["a.md"], "until is exclusive");
    let window = search(json!({"query": "共通語の資料", "since": 2000, "until": 2001}));
    assert_eq!(hit_sources(&window), vec!["b.md"], "since is inclusive");

    // Tag and time constraints AND together.
    let both = search(json!({"query": "共通語の資料", "tags": ["酒", "蔵"], "until": 1500}));
    assert_eq!(hit_sources(&both), vec!["a.md"]);

    // `limit` counts eligible hits — the filter never starves it.
    let limited = search(json!({"query": "共通語の資料", "tags": ["酒", "蔵"], "limit": 1}));
    assert_eq!(hit_sources(&limited).len(), 1);

    // An empty window is refused, not silently empty.
    let (status, answer) = server.call(
        "POST",
        "/contexts/sake/sources/search",
        Some(json!({"query": "共通語の資料", "since": 5, "until": 5})),
    );
    assert_eq!(status, 400, "{answer}");

    // The cross variant applies the same filter to every target, each
    // with its own eligibility split.
    server.ok(
        "PUT",
        "/contexts/beer",
        Some(json!({"description": "untagged neighbor"})),
    );
    server.ok(
        "POST",
        "/contexts/beer/sources",
        Some(json!({"passages": {"z.md": "共通語の資料。\n\n麦芽の話。"}})),
    );
    let cross = server.ok(
        "POST",
        "/sources/search",
        Some(json!({"contexts": ["sake", "beer"], "query": "共通語の資料", "tags": ["酒"]})),
    );
    let contexts: Vec<&str> = cross["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|hit| hit["context"].as_str().unwrap())
        .collect();
    assert!(!contexts.is_empty() && contexts.iter().all(|context| *context == "sake"));
    assert_eq!(
        cross["plan"]["contexts"][0]["filter"],
        json!({"eligible_sources": 1, "total_sources": 3})
    );
    assert_eq!(
        cross["plan"]["contexts"][1]["filter"],
        json!({"eligible_sources": 0, "total_sources": 1})
    );
}

#[test]
fn filters_key_the_retrieval_cache_apart_and_normalized_spellings_share() {
    let server = Server::start("meta-cache");
    seed(&server, "sake");
    let search = |body: Value| server.ok("POST", "/contexts/sake/sources/search", Some(body));

    // Same query, same filter: the second call is a cache hit.
    let misses = search_cache(&server, "miss");
    let hits = search_cache(&server, "hit");
    search(json!({"query": "共通語", "tags": ["酒", "蔵"]}));
    search(json!({"query": "共通語", "tags": ["酒", "蔵"]}));
    assert_eq!(search_cache(&server, "miss"), misses + 1);
    assert_eq!(search_cache(&server, "hit"), hits + 1);

    // Two spellings of ONE filter (order, duplicates) normalize onto
    // one key and share the entry.
    let hits = search_cache(&server, "hit");
    search(json!({"query": "共通語", "tags": ["蔵", "酒", "酒"]}));
    assert_eq!(
        search_cache(&server, "hit"),
        hits + 1,
        "a reordered, duplicated tag list is the same filter"
    );

    // A different filter never shares: dropping the filter, changing
    // the tag set, and adding a window each miss.
    let misses = search_cache(&server, "miss");
    search(json!({"query": "共通語"}));
    search(json!({"query": "共通語", "tags": ["酒"]}));
    search(json!({"query": "共通語", "tags": ["酒"], "until": 1500}));
    assert_eq!(
        search_cache(&server, "miss"),
        misses + 3,
        "same query, different filters must be three distinct entries"
    );

    // A metadata-changing write (a re-store is the only one) re-keys
    // the same filtered request.
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({"passages": {"a.md": "共通語の資料。\n\n改訂版。"}, "tags": {"a.md": ["酒"]}})),
    );
    let misses = search_cache(&server, "miss");
    search(json!({"query": "共通語", "tags": ["酒", "蔵"]}));
    assert_eq!(
        search_cache(&server, "miss"),
        misses + 1,
        "the write moved the passages lane; the cached page must not survive it"
    );
}

/// The semantic tier's bucket carries the filter (both sans-query
/// tuples do): a paraphrase under one filter must never serve a page
/// filled under another — `queries_agree` compares query text only, so
/// this separation exists exactly because the filter rides the bucket
/// params. Within one filter the paraphrase pairing still works.
#[test]
fn the_semantic_tier_never_pairs_requests_across_different_filters() {
    let provider = crate::semantic_cache::spawn_paired_embeddings();
    let env = crate::semantic_cache::semantic_env(&provider);
    let env: Vec<(&str, &str)> = env.iter().map(|(k, v)| (*k, v.as_str())).collect();
    let server = Server::start_with_env("meta-semantic", &env);
    let semantic = |outcome: &str| {
        metric(
            &server,
            &format!("taguru_semantic_cache_total{{outcome=\"{outcome}\"}}"),
        )
    };
    server.ok("PUT", "/contexts/mill", Some(json!({"description": "d"})));
    // Two sources matching the paired queries; only one is tagged, so
    // the filtered and unfiltered pages differ observably.
    server.ok(
        "POST",
        "/contexts/mill/sources",
        Some(json!({
            "passages": {
                "a.md": "The mill produces fresh oysters.",
                "b.md": "The mill produces oysters in bulk."
            },
            "tags": {"a.md": ["酒"]}
        })),
    );
    let search = |body: Value| server.ok("POST", "/contexts/mill/sources/search", Some(body));
    let sources = |page: &Value| -> Vec<String> {
        page["hits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|hit| hit["source"].as_str().unwrap().to_string())
            .collect()
    };

    // Fill under the filter, then serve its paraphrase under the SAME
    // filter — the tier works within a bucket.
    let filtered = search(json!({"query": "does the mill produce oysters", "tags": ["酒"]}));
    assert_eq!(sources(&filtered), vec!["a.md"]);
    assert_eq!(semantic("miss"), 1);
    let paraphrased = search(json!({"query": "is the mill producing oysters", "tags": ["酒"]}));
    assert_eq!(paraphrased, filtered, "the canonical's page, unchanged");
    assert_eq!(semantic("hit"), 1);

    // The same paraphrase WITHOUT the filter must not serve that page:
    // its bucket differs, so it opens its own cluster and answers from
    // the full corpus.
    let unfiltered = search(json!({"query": "is the mill producing oysters"}));
    assert_eq!(
        sources(&unfiltered).len(),
        2,
        "the unfiltered page answers from the whole corpus: {unfiltered}"
    );
    assert_eq!(
        semantic("hit"),
        1,
        "a filter-A claim must never serve a filter-less lookup"
    );
    assert_eq!(semantic("miss"), 2, "the unfiltered bucket starts empty");

    // Positive control: within the unfiltered bucket the pair works.
    let canonical = search(json!({"query": "does the mill produce oysters"}));
    assert_eq!(canonical, unfiltered, "{canonical}");
    assert_eq!(semantic("hit"), 2);
}

#[test]
fn explain_names_filtered_out_before_the_lanes() {
    let server = Server::start("meta-explain");
    seed(&server, "sake");

    // Unfiltered, the source explains normally (served here).
    let plain = server.ok(
        "POST",
        "/contexts/sake/sources/search/explain",
        Some(json!({"query": "共通語の資料", "source": "c.md"})),
    );
    assert_eq!(plain["verdict"], "served", "{plain}");

    // Under a filter that excludes it, the verdict is the filter — not
    // a misleading lane diagnosis about a search that never saw it.
    let filtered = server.ok(
        "POST",
        "/contexts/sake/sources/search/explain",
        Some(json!({"query": "共通語の資料", "source": "c.md", "tags": ["酒"]})),
    );
    assert_eq!(filtered["verdict"], "filtered_out", "{filtered}");
    assert!(
        filtered["summary"]
            .as_str()
            .unwrap_or_default()
            .contains("filter"),
        "{filtered}"
    );

    // An eligible target under the same filter is ranked against the
    // ELIGIBLE field only: unfiltered, all three sources' matching
    // paragraphs rank; filtered to a.md's tag, only a.md's does.
    let eligible = server.ok(
        "POST",
        "/contexts/sake/sources/search/explain",
        Some(json!({"query": "共通語の資料", "source": "a.md", "tags": ["酒"]})),
    );
    assert_eq!(eligible["verdict"], "served", "{eligible}");
    assert_eq!(eligible["ranking"]["ranked"], 1, "{eligible}");
    assert_eq!(
        plain["ranking"]["ranked"], 3,
        "the same query unfiltered ranks every source's matching paragraph: {plain}"
    );
}

#[test]
fn the_mcp_tools_route_metadata_and_filters() {
    let server = Server::start("meta-mcp");
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "mcp metadata"})),
    );

    let stored = server.call_tool(
        1,
        "store_passages",
        json!({
            "context": "sake",
            "passages": {"a.md": "共通語の資料。\n\n酒の話。", "b.md": "共通語の資料。\n\n蔵の話。"},
            "tags": {"a.md": ["酒"]},
            "dates": {"a.md": 1000}
        }),
    );
    assert_ne!(stored["isError"], json!(true), "{stored}");

    let searched = server.call_tool(
        2,
        "search_passages",
        json!({"context": "sake", "query": "共通語の資料", "tags": ["酒"]}),
    );
    assert_ne!(searched["isError"], json!(true), "{searched}");
    // Tool content is the API envelope as JSON text; the payload sits
    // under `result`.
    let envelope: Value = serde_json::from_str(
        searched["content"][0]["text"]
            .as_str()
            .expect("tool content is JSON text"),
    )
    .expect("tool content parses");
    let page = &envelope["result"];
    let hits = page["hits"]
        .as_array()
        .unwrap_or_else(|| panic!("no hits in {page}"));
    assert!(
        !hits.is_empty() && hits.iter().all(|hit| hit["source"] == "a.md"),
        "{page}"
    );
    assert_eq!(
        page["plan"]["contexts"][0]["filter"],
        json!({"eligible_sources": 1, "total_sources": 2})
    );

    let listed = server.call_tool(3, "list_sources", json!({"context": "sake"}));
    assert_ne!(listed["isError"], json!(true), "{listed}");
    let envelope: Value =
        serde_json::from_str(listed["content"][0]["text"].as_str().unwrap()).unwrap();
    let page = &envelope["result"];
    assert_eq!(page["entries"][0]["tags"], json!(["酒"]), "{page}");
    assert_eq!(page["entries"][0]["date"], 1000, "{page}");

    // explain_search forwards the same filter — an excluded source
    // verdicts filtered_out through MCP exactly as over HTTP.
    let explained = server.call_tool(
        4,
        "explain_search",
        json!({"context": "sake", "query": "共通語の資料", "source": "b.md", "tags": ["酒"]}),
    );
    assert_ne!(explained["isError"], json!(true), "{explained}");
    let envelope: Value =
        serde_json::from_str(explained["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(envelope["result"]["verdict"], "filtered_out", "{envelope}");
}
