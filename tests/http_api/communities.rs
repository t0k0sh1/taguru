//! The community verbs and `taguru communities` (issue #166) end to
//! end: the analysis stream carries its revision snapshot, search
//! refuses honestly before an artifact exists and verdicts staleness
//! after one does, the MCP tool routes, and the CLI derives
//! incrementally — an unchanged graph re-runs with zero LLM calls.

use std::io::{BufRead, BufReader, Read, Write};
use std::process::Command;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use crate::support::*;

/// A raw-body GET — the analysis stream is JSON Lines, which the JSON
/// envelope helpers cannot parse.
fn get_ndjson(server: &Server, path: &str) -> (u16, String) {
    let request = ureq::http::Request::builder()
        .method("GET")
        .uri(format!("{}{path}", server.base))
        .body(())
        .expect("request must assemble");
    let mut response = test_agent().run(request).expect("request must run");
    let status = response.status().as_u16();
    let body = response
        .body_mut()
        .read_to_string()
        .expect("body must read");
    (status, body)
}

/// Two 4-cliques with NO bridge: exactly two leaf communities and —
/// with nothing to merge above them — exactly one level, so every
/// LLM-call count below is deterministic.
fn seed_two_cliques(server: &Server, name: &str) {
    server.ok("PUT", &format!("/contexts/{name}"), None);
    let mut ops = Vec::new();
    for group in ["a", "b"] {
        let members: Vec<String> = (1..=4).map(|index| format!("{group}{index}")).collect();
        for (index, subject) in members.iter().enumerate() {
            for object in &members[index + 1..] {
                ops.push(json!({
                    "subject": subject,
                    "label": "近い",
                    "object": object,
                    "weight": 2.0,
                }));
            }
        }
    }
    server.ok(
        "POST",
        &format!("/contexts/{name}/associations"),
        Some(Value::Array(ops)),
    );
}

#[test]
fn the_analysis_stream_carries_the_partition_and_its_revision_snapshot() {
    let server = Server::start("communities-analysis");
    seed_two_cliques(&server, "corpus");
    let revision = server.ok("GET", "/contexts/corpus", None)["revision"]["graph"]
        .as_u64()
        .expect("a revision");

    let (status, body) = get_ndjson(&server, "/contexts/corpus/communities");
    assert_eq!(status, 200, "{body}");
    let lines: Vec<Value> = body
        .lines()
        .map(|line| serde_json::from_str(line).expect("every line parses"))
        .collect();
    let header = &lines[0];
    assert_eq!(header["taguru_communities"], 1);
    assert_eq!(header["algorithm"], "louvain-cc/1");
    assert_eq!(header["revision"]["graph"].as_u64(), Some(revision));
    assert_eq!(header["concept_count"], 8);
    assert_eq!(header["levels"], 1);
    let communities = &lines[1..];
    assert_eq!(
        communities.len(),
        header["communities"].as_u64().unwrap() as usize
    );
    assert_eq!(communities.len(), 2);
    for community in communities {
        assert_eq!(community["level"], 0);
        assert_eq!(community["concept_count"], 4);
        assert_eq!(community["members"].as_array().unwrap().len(), 4);
        let fingerprint = community["fingerprint"].as_str().unwrap();
        assert_eq!(fingerprint.len(), 16, "an fnv64 hex digest: {fingerprint}");
        assert!(
            community["members"][0]["strength"].as_f64().unwrap() > 0.0,
            "members carry their intra-community strength"
        );
    }

    let (status, _) = server.call("GET", "/contexts/nowhere/communities", None);
    assert_eq!(status, 404);
}

#[test]
fn search_refuses_without_an_artifact_and_verdicts_staleness_with_one() {
    let server = Server::start("communities-search");
    seed_two_cliques(&server, "sci");

    // No artifact: a refusal that names the build command — absence of
    // analysis must never read as an empty corpus.
    let (status, refusal) = server.call(
        "POST",
        "/contexts/sci/communities/search",
        Some(json!({"query": "何がテーマか"})),
    );
    assert_eq!(status, 404, "{refusal}");
    assert!(
        refusal["error"]
            .as_str()
            .unwrap()
            .contains("taguru communities"),
        "{refusal}"
    );

    // Build the artifact by hand through the same API the CLI uses.
    let revision = server.ok("GET", "/contexts/sci", None)["revision"].clone();
    server.ok("PUT", "/contexts/sci::communities", None);
    let manifest = json!({
        "taguru_communities": 1,
        "algorithm": "louvain-cc/1",
        "source_context": "sci",
        "revision": revision,
        "levels": 1,
        "communities": [
            {"id": "L0-0", "level": 0, "fingerprint": "00aa00aa00aa00aa", "concept_count": 4},
        ],
    });
    server.ok(
        "POST",
        "/contexts/sci::communities/sources",
        Some(json!({"passages": {
            "community:L0-0": "夏目漱石と明治の文学者たちの交流についての要約。",
            "communities:manifest": manifest.to_string(),
        }})),
    );
    server.ok(
        "POST",
        "/contexts/sci::communities/associations",
        Some(json!([
            {"subject": "community:L0-0", "label": "contains", "object": "a1", "weight": 6.0},
            {"subject": "community:L0-0", "label": "contains", "object": "a2", "weight": 4.0},
        ])),
    );

    let page = server.ok(
        "POST",
        "/contexts/sci/communities/search",
        Some(json!({"query": "夏目漱石"})),
    );
    assert_eq!(page["derived"], "sci::communities");
    assert_eq!(page["stale"], false, "{page}");
    assert_eq!(page["algorithm"], "louvain-cc/1");
    let hit = &page["hits"][0];
    assert_eq!(hit["community"], "L0-0");
    assert_eq!(hit["level"], 0);
    assert_eq!(hit["concept_count"], 4);
    assert!(hit["text"].as_str().unwrap().contains("夏目漱石"));
    let members = hit["members"].as_array().unwrap();
    assert_eq!(members[0]["name"], "a1", "strongest member first");
    assert_eq!(page["plan"]["contexts"][0]["context"], "sci");

    // A source-graph write flips the verdict IMMEDIATELY — the cached
    // entry cannot answer, because the source's current graph revision
    // is part of the cache key's params.
    server.ok(
        "POST",
        "/contexts/sci/associations",
        Some(json!([
            {"subject": "a1", "label": "新事実", "object": "z9", "weight": 1.0},
        ])),
    );
    let page = server.ok(
        "POST",
        "/contexts/sci/communities/search",
        Some(json!({"query": "夏目漱石"})),
    );
    assert_eq!(page["stale"], true, "{page}");
    let recorded = page["revision"]["recorded_graph"].as_u64().unwrap();
    let current = page["revision"]["current_graph"].as_u64().unwrap();
    assert!(current > recorded, "{page}");

    // A `derived` override pointing nowhere is the same honest refusal.
    let (status, refusal) = server.call(
        "POST",
        "/contexts/sci/communities/search",
        Some(json!({"query": "夏目漱石", "derived": "elsewhere"})),
    );
    assert_eq!(status, 404, "{refusal}");
    assert!(refusal["error"].as_str().unwrap().contains("elsewhere"));
}

/// #562 item 7: one `search_communities` call must bump the aggregate
/// `taguru_searches_total{op="search_communities"}` exactly once, even
/// though it touches two contexts (the real source and the derived
/// artifact) — each still gets its own per-context `usage.reads` row.
#[test]
fn a_single_search_counts_the_aggregate_once_and_both_contexts_reads() {
    let server = Server::start("communities-search-counts");
    seed_two_cliques(&server, "sci");

    let revision = server.ok("GET", "/contexts/sci", None)["revision"].clone();
    server.ok("PUT", "/contexts/sci::communities", None);
    let manifest = json!({
        "taguru_communities": 1,
        "algorithm": "louvain-cc/1",
        "source_context": "sci",
        "revision": revision,
        "levels": 1,
        "communities": [
            {"id": "L0-0", "level": 0, "fingerprint": "00aa00aa00aa00aa", "concept_count": 4},
        ],
    });
    server.ok(
        "POST",
        "/contexts/sci::communities/sources",
        Some(json!({"passages": {
            "community:L0-0": "夏目漱石と明治の文学者たちの交流についての要約。",
            "communities:manifest": manifest.to_string(),
        }})),
    );
    server.ok(
        "POST",
        "/contexts/sci::communities/associations",
        Some(json!([
            {"subject": "community:L0-0", "label": "contains", "object": "a1", "weight": 6.0},
            {"subject": "community:L0-0", "label": "contains", "object": "a2", "weight": 4.0},
        ])),
    );

    server.ok(
        "POST",
        "/contexts/sci/communities/search",
        Some(json!({"query": "夏目漱石"})),
    );

    let (status, body) = server.call("GET", "/metrics", None);
    assert_eq!(status, 200);
    let text = body.as_str().expect("metrics body is text, not JSON");
    assert!(
        text.contains("taguru_searches_total{op=\"search_communities\",outcome=\"hit\"} 1"),
        "one search over two contexts must bump the aggregate exactly once: {text}"
    );

    let source = server.ok("GET", "/contexts/sci", None);
    assert_eq!(source["usage"]["reads"], json!(1), "{source}");
    let derived = server.ok("GET", "/contexts/sci::communities", None);
    assert_eq!(derived["usage"]["reads"], json!(1), "{derived}");
}

#[test]
fn the_search_communities_tool_routes_through_mcp() {
    let server = Server::start("communities-mcp");
    seed_two_cliques(&server, "mcp-src");
    let result = server.call_tool(
        1,
        "search_communities",
        json!({"context": "mcp-src", "query": "テーマ"}),
    );
    // No artifact yet: the tool surfaces the server's refusal — with
    // the build command — as a tool error, not an empty result.
    assert_eq!(result["isError"], true, "{result}");
    assert!(
        result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("taguru communities"),
        "{result}"
    );
}

/// A chat stub that answers every completion with a unique, searchable
/// summary and records each request body.
fn stub_chat(replies: Arc<Mutex<Vec<String>>>) -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut content_length = 0usize;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
                    break;
                }
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    content_length = value.trim().parse().unwrap_or(0);
                }
            }
            let mut body = vec![0u8; content_length];
            if reader.read_exact(&mut body).is_err() {
                continue;
            }
            let index = {
                let mut replies = replies.lock().unwrap();
                replies.push(String::from_utf8_lossy(&body).into_owned());
                replies.len()
            };
            let content = format!("要約: この共同体のテーマは主題{index}です。");
            let payload =
                json!({"choices": [{"message": {"role": "assistant", "content": content}}]})
                    .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                payload.len(),
            );
            let _ = stream.write_all(response.as_bytes());
        }
    });
    url
}

/// One `taguru communities` run, hermetic like every other binary
/// spawn.
fn run_communities(args: &[&str], env: &[(&str, &str)]) -> (i32, String, String) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    common::scrub_taguru_env(&mut command)
        .arg("communities")
        .args(args);
    for (key, value) in env {
        command.env(key, value);
    }
    let output = command.output().expect("communities must run");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn the_cli_derives_incrementally_and_dry_run_writes_nothing() {
    let server = Server::start("communities-cli");
    seed_two_cliques(&server, "corp");
    let requests = Arc::new(Mutex::new(Vec::new()));
    let chat_url = stub_chat(Arc::clone(&requests));
    let extract_env = [
        ("TAGURU_EXTRACT_URL", chat_url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];

    // First run: two leaf communities, two summaries, one artifact.
    let (code, stdout, stderr) =
        run_communities(&["--context", "corp", &server.base], &extract_env);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("2 generated, 0 reused"), "{stdout}");
    assert_eq!(requests.lock().unwrap().len(), 2);
    // The chat body carries exactly the base keys: communities rides
    // extract's ChatClient without ever engaging its structured-output
    // or output-budget options, and a defaulted request must not grow.
    let first: serde_json::Value =
        serde_json::from_str(&requests.lock().unwrap()[0]).expect("a JSON chat body");
    assert_eq!(
        first
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["messages", "model", "temperature"],
        "{first}"
    );
    let prompts = requests.lock().unwrap().join("\n");
    assert!(
        prompts.contains("近い"),
        "leaf prompts carry the induced relations: {prompts}"
    );

    let page = server.ok(
        "POST",
        "/contexts/corp/communities/search",
        Some(json!({"query": "共同体のテーマ"})),
    );
    assert_eq!(page["stale"], false, "{page}");
    assert!(!page["hits"].as_array().unwrap().is_empty());
    let hit = &page["hits"][0];
    assert!(hit["text"].as_str().unwrap().contains("要約"));
    assert_eq!(hit["members"].as_array().unwrap().len(), 4);
    assert_eq!(hit["concept_count"], 4);

    // Membership landed as queryable edges on the artifact.
    let community = hit["community"].as_str().unwrap();
    let members = server.ok(
        "POST",
        "/contexts/corp::communities/query",
        Some(json!({"subject": format!("community:{community}"), "label": "contains"})),
    );
    assert_eq!(members["total"], 4, "{members}");

    // Unchanged graph: the re-run reuses every summary — zero LLM
    // calls is the whole point of the fingerprints.
    let (code, stdout, _) = run_communities(&["--context", "corp", &server.base], &extract_env);
    assert_eq!(code, 0, "{stdout}");
    assert!(stdout.contains("0 generated, 2 reused"), "{stdout}");
    assert_eq!(requests.lock().unwrap().len(), 2);

    // One clique's content moves: exactly that community re-summarizes.
    server.ok(
        "POST",
        "/contexts/corp/associations",
        Some(json!([
            {"subject": "a1", "label": "近い", "object": "a2", "weight": 1.0},
        ])),
    );

    // --dry-run sees the pending work but writes nothing and calls
    // nobody — it must succeed with no extract env at all.
    let derived_before = server.ok("GET", "/contexts/corp::communities", None)["revision"].clone();
    let (code, stdout, _) = run_communities(&["--context", "corp", "--dry-run", &server.base], &[]);
    assert_eq!(code, 0, "{stdout}");
    assert!(stdout.contains("1 would generate"), "{stdout}");
    assert_eq!(requests.lock().unwrap().len(), 2);
    let derived_after = server.ok("GET", "/contexts/corp::communities", None)["revision"].clone();
    assert_eq!(derived_before, derived_after, "a dry run must not write");

    let (code, stdout, _) = run_communities(&["--context", "corp", &server.base], &extract_env);
    assert_eq!(code, 0, "{stdout}");
    assert!(stdout.contains("1 generated, 1 reused"), "{stdout}");
    assert_eq!(requests.lock().unwrap().len(), 3);
    let page = server.ok(
        "POST",
        "/contexts/corp/communities/search",
        Some(json!({"query": "共同体のテーマ"})),
    );
    assert_eq!(
        page["stale"], false,
        "the refreshed artifact is current: {page}"
    );

    // A torn artifact: the manifest promises a summary the store no
    // longer holds. The graph is unchanged (zero fresh communities),
    // so this is exactly the case where the chat client must still
    // come up — the run repairs the hole with one LLM call.
    let community = page["hits"][0]["community"].as_str().unwrap().to_string();
    server.ok(
        "POST",
        "/contexts/corp::communities/sources/retract",
        Some(json!({"source": format!("community:{community}")})),
    );
    let (code, stdout, _) = run_communities(&["--context", "corp", &server.base], &extract_env);
    assert_eq!(code, 0, "{stdout}");
    assert!(stdout.contains("1 generated, 1 reused"), "{stdout}");
    assert_eq!(requests.lock().unwrap().len(), 4);
    let page = server.ok(
        "POST",
        "/contexts/corp/communities/search",
        Some(json!({"query": "共同体のテーマ"})),
    );
    assert!(
        !page["hits"].as_array().unwrap().is_empty(),
        "the repaired artifact serves again: {page}"
    );
}

// --- community_hits: untested refusal/degradation branches -----------------

/// The artifact's own manifest builder, `search_refuses_without_an_
/// artifact_and_verdicts_staleness_with_one`'s setup factored out so
/// the branches below can each mutate one field of an otherwise valid
/// artifact.
fn seed_manifest_artifact(server: &Server, derived: &str, manifest: &Value) {
    server.ok("PUT", &format!("/contexts/{derived}"), None);
    server.ok(
        "POST",
        &format!("/contexts/{derived}/sources"),
        Some(json!({"passages": {
            "community:L0-0": "夏目漱石と明治の文学者たちの交流についての要約。",
            "communities:manifest": manifest.to_string(),
        }})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{derived}/associations"),
        Some(json!([
            {"subject": "community:L0-0", "label": "contains", "object": "a1", "weight": 6.0},
            {"subject": "community:L0-0", "label": "contains", "object": "a2", "weight": 4.0},
        ])),
    );
}

/// A `communities:manifest` record that fails to parse as JSON — the
/// artifact exists, but its identity record is corrupt, distinct from
/// no artifact at all (which answers "build one", not a 409).
#[test]
fn search_reports_conflict_when_the_manifest_record_does_not_parse() {
    let server = Server::start("communities-manifest-corrupt");
    seed_two_cliques(&server, "sci");
    server.ok("PUT", "/contexts/sci::communities", None);
    server.ok(
        "POST",
        "/contexts/sci::communities/sources",
        Some(json!({"passages": {
            "communities:manifest": "{not valid json",
        }})),
    );

    let (status, refused) = server.call(
        "POST",
        "/contexts/sci/communities/search",
        Some(json!({"query": "夏目漱石"})),
    );
    assert_eq!(status, 409, "{refused}");
    assert_eq!(refused["code"], json!("conflict"), "{refused}");
    assert!(
        refused["error"]
            .as_str()
            .unwrap()
            .contains("does not parse"),
        "{refused}"
    );
}

/// A manifest whose `source_context` names a DIFFERENT context than
/// the one the search was made against — the artifact answers for
/// somebody else's graph, so serving it would silently mislabel it.
#[test]
fn search_reports_conflict_when_the_manifest_names_a_different_source_context() {
    let server = Server::start("communities-manifest-mismatch");
    seed_two_cliques(&server, "sci");
    let revision = server.ok("GET", "/contexts/sci", None)["revision"].clone();
    let manifest = json!({
        "taguru_communities": 1,
        "algorithm": "louvain-cc/1",
        "source_context": "elsewhere",
        "revision": revision,
        "levels": 1,
        "communities": [
            {"id": "L0-0", "level": 0, "fingerprint": "00aa00aa00aa00aa", "concept_count": 4},
        ],
    });
    seed_manifest_artifact(&server, "sci::communities", &manifest);

    let (status, refused) = server.call(
        "POST",
        "/contexts/sci/communities/search",
        Some(json!({"query": "夏目漱石"})),
    );
    assert_eq!(status, 409, "{refused}");
    assert_eq!(refused["code"], json!("conflict"), "{refused}");
    let message = refused["error"].as_str().unwrap();
    assert!(message.contains("elsewhere"), "{message}");
    assert!(message.contains("sci"), "{message}");
}

/// `check_derived_scope`'s Forbidden: the auth middleware already
/// cleared the PATH context ('sci'), but the DERIVED artifact context
/// ('sci::communities') is a second read target named in the body, and
/// a scoped key without a grant on it must be refused just as it would
/// be for the path context — otherwise a scoped key could read any
/// context by aiming a search's `derived` field at it.
#[test]
fn search_reports_forbidden_when_the_scoped_key_has_no_grant_on_the_derived_context() {
    let server = Server::start_with_env(
        "communities-derived-scope",
        &[
            ("TAGURU_API_TOKENS", "boss:atok,reader:rtok"),
            (
                "TAGURU_KEY_SCOPES",
                r#"{"reader": {"role": "read", "contexts": ["sci"]}}"#,
            ),
        ],
    );
    let admin = |method: &str, path: &str, body: Option<Value>| {
        server.call_with_token(method, path, body, Some("atok"))
    };
    admin("PUT", "/contexts/sci", None);
    // The same 4-clique graph `seed_two_cliques` seeds, over an
    // authenticated admin token so the derived-scope grant below can
    // be scoped to a real, non-empty source graph.
    let members: Vec<String> = (1..=4).map(|index| format!("a{index}")).collect();
    let mut ops = Vec::new();
    for (index, subject) in members.iter().enumerate() {
        for object in &members[index + 1..] {
            ops.push(json!({"subject": subject, "label": "近い", "object": object, "weight": 2.0}));
        }
    }
    admin(
        "POST",
        "/contexts/sci/associations",
        Some(Value::Array(ops)),
    );
    let revision = admin("GET", "/contexts/sci", None).1["revision"].clone();
    let manifest = json!({
        "taguru_communities": 1,
        "algorithm": "louvain-cc/1",
        "source_context": "sci",
        "revision": revision,
        "levels": 1,
        "communities": [
            {"id": "L0-0", "level": 0, "fingerprint": "00aa00aa00aa00aa", "concept_count": 4},
        ],
    });
    admin("PUT", "/contexts/sci::communities", None);
    admin(
        "POST",
        "/contexts/sci::communities/sources",
        Some(json!({"passages": {
            "community:L0-0": "夏目漱石と明治の文学者たちの交流についての要約。",
            "communities:manifest": manifest.to_string(),
        }})),
    );
    admin(
        "POST",
        "/contexts/sci::communities/associations",
        Some(json!([
            {"subject": "community:L0-0", "label": "contains", "object": "a1", "weight": 6.0},
        ])),
    );

    let (status, refused) = server.call_with_token(
        "POST",
        "/contexts/sci/communities/search",
        Some(json!({"query": "夏目漱石"})),
        Some("rtok"),
    );
    assert_eq!(status, 403, "{refused}");
    assert_eq!(refused["code"], json!("forbidden"), "{refused}");
    let message = refused["error"].as_str().unwrap();
    assert!(message.contains("sci::communities"), "{message}");
}

/// `MEMBERS_PER_HIT` (12): a community's `contains` membership beyond
/// the cap is truncated, `members_truncated` says so rather than
/// silently dropping the tail — `community_hits` reads membership
/// straight off the artifact's own graph, so this needs no community
/// DETECTION run at all, just `contains` edges asserted directly.
#[test]
fn search_truncates_membership_past_members_per_hit_and_flags_it() {
    let server = Server::start("communities-members-cap");
    seed_two_cliques(&server, "sci");
    let revision = server.ok("GET", "/contexts/sci", None)["revision"].clone();
    let manifest = json!({
        "taguru_communities": 1,
        "algorithm": "louvain-cc/1",
        "source_context": "sci",
        "revision": revision,
        "levels": 1,
        "communities": [
            {"id": "L0-0", "level": 0, "fingerprint": "00aa00aa00aa00aa", "concept_count": 13},
        ],
    });
    server.ok("PUT", "/contexts/sci::communities", None);
    server.ok(
        "POST",
        "/contexts/sci::communities/sources",
        Some(json!({"passages": {
            "community:L0-0": "夏目漱石と明治の文学者たちの交流についての要約。",
            "communities:manifest": manifest.to_string(),
        }})),
    );
    // 13 members, strongest first by weight — one past MEMBERS_PER_HIT.
    let members: Vec<Value> = (0..13)
        .map(|i| {
            json!({
                "subject": "community:L0-0", "label": "contains",
                "object": format!("m{i:02}"), "weight": (13 - i) as f64,
            })
        })
        .collect();
    server.ok(
        "POST",
        "/contexts/sci::communities/associations",
        Some(Value::Array(members)),
    );

    let page = server.ok(
        "POST",
        "/contexts/sci/communities/search",
        Some(json!({"query": "夏目漱石"})),
    );
    let hit = &page["hits"][0];
    assert_eq!(hit["members_truncated"], json!(true), "{page}");
    let served = hit["members"].as_array().unwrap();
    let names: Vec<&str> = served
        .iter()
        .map(|member| member["name"].as_str().unwrap())
        .collect();
    let expected: Vec<String> = (0..12).map(|i| format!("m{i:02}")).collect();
    assert_eq!(
        names, expected,
        "strongest-first order, m12 excluded: {page}"
    );
}

/// A community summary is searchable (its passage and `contains`
/// membership exist in the artifact) but the manifest's `communities`
/// array does not list it — a torn artifact, one write behind the
/// other. `level`/`parent`/`concept_count` all come from the manifest
/// fact lookup (`facts.get(summary.id)`), so absence there must
/// degrade those three fields to null rather than panic or fabricate
/// zeros.
#[test]
fn search_omits_manifest_facts_for_a_community_the_manifest_does_not_list() {
    let server = Server::start("communities-manifest-torn");
    seed_two_cliques(&server, "sci");
    let revision = server.ok("GET", "/contexts/sci", None)["revision"].clone();
    // The manifest's own `communities` array is empty — L0-0 is
    // searchable (passage + contains edges below) but unlisted.
    let manifest = json!({
        "taguru_communities": 1,
        "algorithm": "louvain-cc/1",
        "source_context": "sci",
        "revision": revision,
        "levels": 1,
        "communities": [],
    });
    seed_manifest_artifact(&server, "sci::communities", &manifest);

    let page = server.ok(
        "POST",
        "/contexts/sci/communities/search",
        Some(json!({"query": "夏目漱石"})),
    );
    let hit = &page["hits"][0];
    assert_eq!(hit["community"], json!("L0-0"), "{page}");
    assert!(hit["level"].is_null(), "{page}");
    assert!(hit["parent"].is_null(), "{page}");
    assert!(hit["concept_count"].is_null(), "{page}");
    // Membership itself is unaffected — it comes straight off the
    // graph, not the manifest.
    assert_eq!(hit["members"].as_array().unwrap().len(), 2, "{page}");
}

// --- #753: derive-flow branches the #551 sweep found untested -------------

/// Replaces the artifact's manifest passage wholesale through the same
/// retract-then-apply door everything else uses.
fn overwrite_manifest(server: &Server, derived: &str, manifest_text: &str) {
    let batch = format!(
        "{}\n{}\n",
        json!({"taguru_batch": 1, "context": derived, "source": "communities:manifest"}),
        json!({"passage": manifest_text}),
    );
    let (status, body) = server.call_raw(
        "POST",
        "/import",
        Some(&batch),
        Some("application/x-ndjson"),
    );
    assert_eq!(status, 200, "{body}");
}

/// A vanished community — present in the previous manifest, absent
/// from the new analysis — retracts its stored source; the survivor's
/// batch is rewritten in place and stays queryable. An unchanged
/// re-run afterwards needs no extract env at all.
#[test]
fn a_vanished_community_is_retracted_and_the_survivor_kept() {
    let server = Server::start("communities-vanish");
    server.ok("PUT", "/contexts/corp", None);
    let mut ops = Vec::new();
    for (group, source) in [("a", "a.md"), ("b", "b.md")] {
        let members: Vec<String> = (1..=4).map(|index| format!("{group}{index}")).collect();
        for (index, subject) in members.iter().enumerate() {
            for object in &members[index + 1..] {
                ops.push(json!({
                    "subject": subject, "label": "近い", "object": object,
                    "weight": 2.0, "source": source,
                }));
            }
        }
    }
    server.ok(
        "POST",
        "/contexts/corp/associations",
        Some(Value::Array(ops)),
    );

    let requests = Arc::new(Mutex::new(Vec::new()));
    let chat_url = stub_chat(Arc::clone(&requests));
    let extract_env = [
        ("TAGURU_EXTRACT_URL", chat_url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) =
        run_communities(&["--context", "corp", &server.base], &extract_env);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("2 generated, 0 reused"), "{stdout}");
    assert!(
        !stdout.contains("vanished") && !stdout.contains("algorithm changed"),
        "a first run neither retracts nor rebuilds: {stdout}"
    );

    // An unchanged graph re-runs WITHOUT the extract env — the chat
    // client only comes up when something actually needs summarizing.
    let (code, stdout, stderr) = run_communities(&["--context", "corp", &server.base], &[]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("0 generated, 2 reused"), "{stdout}");

    // Clique b's whole source retracts: the new analysis holds one
    // community, and the run must retract exactly the vanished one.
    server.ok(
        "POST",
        "/contexts/corp/sources/retract",
        Some(json!({"source": "b.md"})),
    );
    let (code, stdout, stderr) =
        run_communities(&["--context", "corp", &server.base], &extract_env);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("1 vanished community retracted"),
        "{stdout}"
    );
    assert!(
        stdout.contains("0 generated, 1 reused"),
        "the survivor's fingerprint is unchanged — no re-summarizing: {stdout}"
    );
    assert_eq!(
        requests.lock().unwrap().len(),
        2,
        "retraction costs zero LLM calls"
    );

    // The survivor's membership still answers; the vanished id's
    // source is gone from the artifact.
    let survivors = server.ok(
        "POST",
        "/contexts/corp::communities/query",
        Some(json!({"subject": "community:L0-0", "label": "contains"})),
    );
    assert_eq!(survivors["total"], 4, "{survivors}");
    let looked = server.ok(
        "POST",
        "/contexts/corp::communities/sources/lookup",
        Some(json!({"sources": ["community:L0-1"]})),
    );
    assert!(
        looked["passages"]["community:L0-1"].is_null(),
        "the vanished community's summary must be retracted: {looked}"
    );
}

/// A manifest whose algorithm differs marks every fingerprint
/// incomparable: the report says so and every summary regenerates. A
/// manifest that does not PARSE is a refusal that names the way out —
/// never a silent full rebuild.
#[test]
fn a_changed_algorithm_rebuilds_and_a_mangled_manifest_refuses() {
    let server = Server::start("communities-rebuild");
    seed_two_cliques(&server, "corp");
    let requests = Arc::new(Mutex::new(Vec::new()));
    let chat_url = stub_chat(Arc::clone(&requests));
    let extract_env = [
        ("TAGURU_EXTRACT_URL", chat_url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, _) = run_communities(&["--context", "corp", &server.base], &extract_env);
    assert_eq!(code, 0, "{stdout}");
    assert_eq!(requests.lock().unwrap().len(), 2);

    // The REAL stored manifest with ONLY its algorithm changed — the
    // fingerprints inside would all match, so only the comparability
    // rule forces the rebuild.
    let stored = server.ok(
        "POST",
        "/contexts/corp::communities/sources/lookup",
        Some(json!({"sources": ["communities:manifest"]})),
    );
    let mut manifest: Value =
        serde_json::from_str(stored["passages"]["communities:manifest"].as_str().unwrap())
            .expect("the stored manifest parses");
    manifest["algorithm"] = json!("other/1");
    overwrite_manifest(&server, "corp::communities", &manifest.to_string());
    let (code, stdout, stderr) =
        run_communities(&["--context", "corp", &server.base], &extract_env);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("algorithm changed: previous fingerprints incomparable"),
        "{stdout}"
    );
    assert!(stdout.contains("2 generated, 0 reused"), "{stdout}");
    assert_eq!(requests.lock().unwrap().len(), 4);

    // Mangled bytes where the manifest should be: a human's problem,
    // said out loud.
    overwrite_manifest(&server, "corp::communities", "not a manifest at all");
    let (code, _stdout, stderr) =
        run_communities(&["--context", "corp", &server.base], &extract_env);
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("does not parse"), "{stderr}");
    assert!(
        stderr.contains("delete the artifact context to rebuild"),
        "{stderr}"
    );
}

/// A graph Louvain aggregates twice — two super-clusters of three
/// 3-cliques each, ring-bridged inside — exercises the parent branch
/// end to end: parent prompts quote child summaries, `includes` edges
/// land in the artifact, and search serves parents alongside leaves.
#[test]
fn a_two_level_graph_summarizes_parents_from_their_children() {
    let server = Server::start("communities-levels");
    server.ok("PUT", "/contexts/deep", None);
    let mut ops = Vec::new();
    let sides = [["a", "b", "c"], ["d", "e", "f"]];
    for cliques in &sides {
        for prefix in cliques {
            let members: Vec<String> = (1..=3).map(|index| format!("{prefix}{index}")).collect();
            for (index, subject) in members.iter().enumerate() {
                for object in &members[index + 1..] {
                    ops.push(json!({
                        "subject": subject, "label": "近い", "object": object, "weight": 1.0,
                    }));
                }
            }
        }
        for index in 0..cliques.len() {
            let (p, q) = (cliques[index], cliques[(index + 1) % cliques.len()]);
            ops.push(json!({
                "subject": format!("{p}1"), "label": "橋", "object": format!("{q}1"),
                "weight": 1.5,
            }));
        }
    }
    server.ok(
        "POST",
        "/contexts/deep/associations",
        Some(Value::Array(ops)),
    );

    // The graph really is two levels — the test's premise, pinned so a
    // future algorithm change fails HERE and not in a prompt assert.
    let (status, body) = get_ndjson(&server, "/contexts/deep/communities");
    assert_eq!(status, 200, "{body}");
    let header: Value = serde_json::from_str(body.lines().next().unwrap()).unwrap();
    assert_eq!(header["levels"], 2, "{header}");
    let parents = body
        .lines()
        .skip(1)
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|community| community["level"] == json!(1))
        .collect::<Vec<_>>();
    assert_eq!(parents.len(), 2, "{body}");
    let total = header["communities"].as_u64().unwrap() as usize;

    let requests = Arc::new(Mutex::new(Vec::new()));
    let chat_url = stub_chat(Arc::clone(&requests));
    let extract_env = [
        ("TAGURU_EXTRACT_URL", chat_url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) =
        run_communities(&["--context", "deep", &server.base], &extract_env);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains(&format!("{total} generated, 0 reused")),
        "{stdout}"
    );

    // Parent prompts are built FROM the child summaries — the leaves
    // summarize first, and each parent prompt quotes their text.
    let prompts = requests.lock().unwrap().clone();
    assert_eq!(prompts.len(), total);
    let parent_prompts: Vec<&String> = prompts
        .iter()
        .filter(|prompt| prompt.contains("sub-communities"))
        .collect();
    assert_eq!(parent_prompts.len(), 2, "one prompt per parent");
    assert!(
        parent_prompts.iter().all(|prompt| prompt.contains("主題")),
        "a parent prompt quotes its children's summaries: {parent_prompts:?}"
    );

    // Hierarchy landed as queryable `includes` edges.
    let parent_id = parents[0]["id"].as_str().unwrap();
    let children = parents[0]["children"].as_array().unwrap();
    let includes = server.ok(
        "POST",
        "/contexts/deep::communities/query",
        Some(json!({"subject": format!("community:{parent_id}"), "label": "includes"})),
    );
    assert_eq!(includes["total"], json!(children.len()), "{includes}");
    let mut edge_objects: Vec<String> = includes["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| edge["object"].as_str().unwrap().to_string())
        .collect();
    edge_objects.sort();
    let mut child_sources: Vec<String> = children
        .iter()
        .map(|child| format!("community:{}", child.as_str().unwrap()))
        .collect();
    child_sources.sort();
    assert_eq!(
        edge_objects, child_sources,
        "every child id has exactly its own includes edge"
    );
}

/// `--group` derives one artifact per member context, child groups
/// included transitively; a group reaching no contexts refuses by
/// name; `--into` renames a single-context artifact.
#[test]
fn group_traversal_derives_each_member_and_into_renames() {
    let server = Server::start("communities-group");
    seed_two_cliques(&server, "m1");
    seed_two_cliques(&server, "m2");
    server.ok("PUT", "/groups/child", Some(json!({"contexts": ["m2"]})));
    server.ok(
        "PUT",
        "/groups/kura",
        Some(json!({"contexts": ["m1"], "groups": ["child"]})),
    );
    server.ok("PUT", "/groups/hollow", Some(json!({"contexts": []})));

    let requests = Arc::new(Mutex::new(Vec::new()));
    let chat_url = stub_chat(Arc::clone(&requests));
    let extract_env = [
        ("TAGURU_EXTRACT_URL", chat_url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];

    let (code, _stdout, stderr) =
        run_communities(&["--group", "hollow", &server.base], &extract_env);
    assert_eq!(code, 1, "{stderr}");
    assert!(
        stderr.contains("group 'hollow' reaches no contexts"),
        "{stderr}"
    );

    let (code, stdout, stderr) = run_communities(&["--group", "kura", &server.base], &extract_env);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("context 'm1'"), "{stdout}");
    assert!(stdout.contains("context 'm2'"), "{stdout}");
    for derived in ["m1::communities", "m2::communities"] {
        let (status, _) = server.call("GET", &format!("/contexts/{derived}"), None);
        assert_eq!(status, 200, "the member artifact '{derived}' must exist");
    }

    // --into aims one context's artifact at a chosen name.
    let (code, stdout, stderr) = run_communities(
        &["--context", "m1", "--into", "renamed", &server.base],
        &extract_env,
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("artifact 'renamed'"), "{stdout}");
    let (status, _) = server.call("GET", "/contexts/renamed", None);
    assert_eq!(status, 200);
}

/// `--json` serializes the reports themselves — one object per
/// context, machine-readable counts, nothing human-formatted on
/// stdout.
#[test]
fn json_mode_emits_the_report_structs() {
    let server = Server::start("communities-json");
    seed_two_cliques(&server, "corp");
    let requests = Arc::new(Mutex::new(Vec::new()));
    let chat_url = stub_chat(Arc::clone(&requests));
    let extract_env = [
        ("TAGURU_EXTRACT_URL", chat_url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) =
        run_communities(&["--context", "corp", "--json", &server.base], &extract_env);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let reports: Value = serde_json::from_str(&stdout).expect("stdout must be the JSON reports");
    let report = &reports[0];
    assert_eq!(report["context"], json!("corp"), "{report}");
    assert_eq!(report["derived"], json!("corp::communities"), "{report}");
    assert_eq!(report["summaries_generated"], json!(2), "{report}");
    assert_eq!(report["summaries_reused"], json!(0), "{report}");
    assert_eq!(report["dry_run"], json!(false), "{report}");
}
