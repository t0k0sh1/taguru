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
