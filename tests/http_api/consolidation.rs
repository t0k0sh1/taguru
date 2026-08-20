//! The consolidation audit (ADR 0012), end to end: caller-selected
//! sections, merge candidates corroborated structurally with types
//! attached, contradiction groups joined with assertion times and
//! ranked by measured functional tendency, contested edges with both
//! sides named, staleness gaps with an honest undatable count — and
//! fingerprints that hold still until the evidence moves.

use std::io::{BufRead, BufReader, Read, Write};
use std::process::Command;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use crate::support::*;

/// One corpus exercising every section: spelling twins sharing
/// structure, a dated supersession under a functional-tendency label,
/// a sign-contested edge, and an undatable associations-only source.
fn seed(server: &Server) {
    server.ok(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "整理"})),
    );
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            // The twins: two spellings of one brewery, two shared facts,
            // one distinct fact each.
            {"subject": "青嶺酒造", "label": "銘柄", "object": "青嶺", "weight": 1.0, "source": "doc-a"},
            {"subject": "青嶺酒蔵", "label": "銘柄", "object": "青嶺", "weight": 1.0, "source": "doc-b"},
            {"subject": "青嶺酒造", "label": "所在地", "object": "霧沢町", "weight": 1.0, "source": "doc-a"},
            {"subject": "青嶺酒蔵", "label": "所在地", "object": "霧沢町", "weight": 1.0, "source": "doc-b"},
            {"subject": "青嶺酒造", "label": "創業", "object": "1907", "weight": 1.0, "source": "doc-a"},
            {"subject": "青嶺酒蔵", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "doc-b"},
            // The supersession: 杜氏 is one-object for two subjects,
            // two-object for this one — old fact dated 1000, new 2000.
            {"subject": "蔵A", "label": "杜氏", "object": "高瀬", "weight": 1.0, "source": "doc-2019"},
            {"subject": "蔵A", "label": "杜氏", "object": "青山", "weight": 1.0, "source": "doc-2024"},
            {"subject": "蔵B", "label": "杜氏", "object": "田中", "weight": 1.0, "source": "doc-2019"},
            // The dispute: one edge, two sources, opposite signs.
            {"subject": "蔵A", "label": "行う", "object": "大量生産", "weight": 1.0, "source": "宣伝"},
            {"subject": "蔵A", "label": "行う", "object": "大量生産", "weight": -2.0, "source": "doc-2024"},
            // Staleness: 蔵A's 銘柄 fact is only attested at 1000 while
            // its neighborhood reaches 2000; this one is undatable.
            {"subject": "蔵A", "label": "銘柄", "object": "初霜", "weight": 1.0, "source": "doc-2019"},
            {"subject": "蔵C", "label": "銘柄", "object": "幻", "weight": 1.0, "source": "doc-undated"},
        ])),
    );
    server.ok(
        "POST",
        "/contexts/sake/sources",
        Some(json!({
            "passages": {
                "doc-2019": "旧情報。", "doc-2024": "新情報。", "宣伝": "宣伝文。",
                "doc-a": "表記Aの資料。", "doc-b": "表記Bの資料。"
            },
            "dates": {"doc-2019": 1000, "doc-2024": 2000, "宣伝": 1500}
        })),
    );
}

fn audit(server: &Server, body: Value) -> Value {
    server.ok("POST", "/contexts/sake/consolidation/audit", Some(body))
}

#[test]
fn sections_detect_join_and_fingerprint_their_candidates() {
    let server = Server::start("consolidation-audit");
    seed(&server);

    let report = audit(
        &server,
        json!({"checks": ["merge", "contradiction", "staleness"]}),
    );
    assert_eq!(report["detector"], json!("consolidation/1"));

    // Merge: the spelling twins surface with their shared structure.
    let merge = &report["merge"];
    assert_eq!(merge["total"], json!(1), "{merge}");
    let pair = &merge["candidates"][0];
    assert_eq!(pair["tier"], json!("lexical"));
    assert_eq!(pair["shared_total"], json!(2), "銘柄 and 所在地");
    assert_eq!(pair["only_a_total"], json!(1));
    assert_eq!(pair["only_b_total"], json!(1));
    assert_eq!(pair["overlap"], json!(0.5));

    // Contradiction: the grouped kind leads (ranked by measured
    // functional tendency), rows dated; the contested edge follows
    // with both sides named.
    let contradiction = &report["contradiction"];
    assert_eq!(contradiction["total"], json!(2), "{contradiction}");
    let grouped = &contradiction["candidates"][0];
    assert_eq!(grouped["kind"], json!("objects"));
    assert_eq!(grouped["subject"], json!("蔵A"));
    assert_eq!(grouped["label"], json!("杜氏"));
    // Three 杜氏 subjects (蔵A with two objects, 蔵B and 青嶺酒蔵 with
    // one each): tendency 2/3.
    assert_eq!(grouped["functional_tendency"], json!(2.0 / 3.0));
    let objects = grouped["objects"].as_array().unwrap();
    assert_eq!(objects[0]["object"], json!("高瀬"));
    assert_eq!(objects[0]["latest"], json!(1000));
    assert_eq!(objects[1]["object"], json!("青山"));
    assert_eq!(objects[1]["latest"], json!(2000));
    let contested = &contradiction["candidates"][1];
    assert_eq!(contested["kind"], json!("contested"));
    assert_eq!(contested["supporting_sources"], json!(["宣伝"]));
    assert_eq!(contested["disputing_sources"], json!(["doc-2024"]));

    // Staleness: 蔵A's 銘柄 fact (latest 1000) trails its own
    // neighborhood (2000); the undatable edge is counted, not guessed.
    let staleness = &report["staleness"];
    assert_eq!(staleness["undatable"], json!(1), "{staleness}");
    let stale = staleness["candidates"].as_array().unwrap();
    assert!(
        stale
            .iter()
            .any(|candidate| candidate["label"] == json!("銘柄")
                && candidate["gap"] == json!(1000)),
        "{staleness}"
    );

    // Fingerprints hold still across identical audits…
    let again = audit(&server, json!({"checks": ["contradiction"]}));
    assert_eq!(
        again["contradiction"]["candidates"][0]["fingerprint"],
        grouped["fingerprint"]
    );
    // …and move when the evidence moves.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "蔵A", "label": "杜氏", "object": "青山", "weight": 1.0, "source": "doc-2025"}
        ])),
    );
    let moved = audit(&server, json!({"checks": ["contradiction"]}));
    assert_ne!(
        moved["contradiction"]["candidates"][0]["fingerprint"],
        grouped["fingerprint"]
    );

    // Truncation is honest: limit cuts the page, never the total.
    let cut = audit(&server, json!({"checks": ["contradiction"], "limit": 1}));
    assert_eq!(cut["contradiction"]["total"], json!(2), "{cut}");
    assert_eq!(
        cut["contradiction"]["candidates"].as_array().unwrap().len(),
        1
    );

    // The selector is honest: unrequested sections are absent, an
    // empty or unknown selector refuses.
    let only_merge = audit(&server, json!({"checks": ["merge"]}));
    assert!(only_merge.get("contradiction").is_none());
    assert!(only_merge.get("staleness").is_none());
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/consolidation/audit",
        Some(json!({"checks": []})),
    );
    assert_eq!(status, 400, "{body}");
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/consolidation/audit",
        Some(json!({"checks": ["typo"]})),
    );
    assert_eq!(status, 400, "{body}");
    // Past the shared list ceiling: `dedup` folds only consecutive
    // repeats, so an alternating list this long would otherwise pass
    // both selector guards.
    let alternating: Vec<&str> = ["merge", "contradiction"]
        .into_iter()
        .cycle()
        .take(1001)
        .collect();
    let (status, body) = server.call(
        "POST",
        "/contexts/sake/consolidation/audit",
        Some(json!({"checks": alternating})),
    );
    assert_eq!(status, 400, "{body}");
    assert_eq!(body["code"], json!("over_limit"), "{body}");
    let (status, _) = server.call(
        "POST",
        "/contexts/nope/consolidation/audit",
        Some(json!({"checks": ["staleness"]})),
    );
    assert_eq!(status, 404);
}

/// `floor_secs` is inclusive (issue #620): 蔵A's 銘柄 fact gaps its
/// neighborhood by exactly 1000 (latest 1000 vs. neighborhood 2000,
/// per the seed comment above), so a floor of exactly 1000 must still
/// surface it — only a floor PAST the gap excludes it. Pins the `<`
/// boundary `staleness_section`'s gap filter runs on.
#[test]
fn floor_secs_is_inclusive_of_a_gap_exactly_at_the_floor() {
    let server = Server::start("consolidation-audit-floor-secs");
    seed(&server);

    let at_floor = audit(
        &server,
        json!({"checks": ["staleness"], "floor_secs": 1000}),
    );
    let stale = at_floor["staleness"]["candidates"].as_array().unwrap();
    assert!(
        stale
            .iter()
            .any(|candidate| candidate["label"] == json!("銘柄") && candidate["gap"] == json!(1000)),
        "a gap exactly at floor_secs must still be a candidate — {at_floor}"
    );

    let past_floor = audit(
        &server,
        json!({"checks": ["staleness"], "floor_secs": 1001}),
    );
    let stale = past_floor["staleness"]["candidates"].as_array().unwrap();
    assert!(
        !stale
            .iter()
            .any(|candidate| candidate["label"] == json!("銘柄") && candidate["gap"] == json!(1000)),
        "a gap one below floor_secs must be excluded — {past_floor}"
    );
}

/// A chat stub answering every completion with a valid judgment —
/// fenced, because real models decorate — and counting the calls.
fn stub_judge(calls: Arc<Mutex<usize>>) -> String {
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
            *calls.lock().unwrap() += 1;
            let content = "```json\n{\"verdict\": \"apply\", \"action\": \"alias\", \
                           \"rationale\": \"同一の蔵\"}\n```";
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

fn run_consolidation(args: &[&str], env: &[(&str, &str)]) -> (i32, String, String) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    common::scrub_taguru_env(&mut command)
        .arg("consolidation")
        .args(args);
    for (key, value) in env {
        command.env(key, value);
    }
    let output = command.output().expect("consolidation must run");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// The judge verb end to end (ADR 0012 §5): judgments are keyed by
/// fingerprint, dismissals and re-runs reuse them with zero LLM
/// calls, and only moved evidence re-judges.
#[test]
fn the_cli_judges_incrementally_by_fingerprint() {
    let server = Server::start("consolidation-cli");
    seed(&server);
    let calls = Arc::new(Mutex::new(0usize));
    let chat_url = stub_judge(Arc::clone(&calls));
    let extract_env = [
        ("TAGURU_EXTRACT_URL", chat_url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];

    // Dry-run first: names the work, calls nothing, writes nothing.
    let (code, stdout, stderr) = run_consolidation(
        &["--context", "sake", "--dry-run", &server.base],
        &extract_env,
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("5 to judge, 0 reused"), "{stdout}");
    assert_eq!(*calls.lock().unwrap(), 0);

    // First real run: every candidate judged once, artifact written.
    let (code, stdout, stderr) =
        run_consolidation(&["--context", "sake", &server.base], &extract_env);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("judged 5 (5 apply, 0 dismiss), 0 reused"),
        "{stdout}"
    );
    assert_eq!(*calls.lock().unwrap(), 5);
    let stored = server.ok(
        "POST",
        "/contexts/sake::consolidation/sources/lookup",
        Some(json!({"sources": ["consolidation:manifest"]})),
    );
    let manifest: Value = serde_json::from_str(
        stored["passages"]["consolidation:manifest"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(manifest["taguru_consolidation"], json!(1));
    assert_eq!(manifest["detector"], json!("consolidation/1"));

    // Second run over the unchanged graph: zero LLM calls.
    let (code, stdout, _) = run_consolidation(&["--context", "sake", &server.base], &extract_env);
    assert_eq!(code, 0);
    assert!(
        stdout.contains("judgments up to date (5 reused, no LLM calls)"),
        "{stdout}"
    );
    assert_eq!(*calls.lock().unwrap(), 5);

    // Moved evidence re-judges exactly the moved candidate.
    server.ok(
        "POST",
        "/contexts/sake/associations",
        Some(json!([
            {"subject": "蔵A", "label": "杜氏", "object": "青山", "weight": 1.0, "source": "doc-2024b"}
        ])),
    );
    let (code, stdout, _) = run_consolidation(&["--context", "sake", &server.base], &extract_env);
    assert_eq!(code, 0, "{stdout}");
    assert!(
        stdout.contains("judged 1 (1 apply, 0 dismiss), 4 reused"),
        "{stdout}"
    );
    assert_eq!(*calls.lock().unwrap(), 6);
}

/// A chat stub that never answers the required JSON shape.
fn stub_junk() -> String {
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
            let payload = json!({"choices": [{"message": {"role": "assistant",
                "content": "判定できませんでした。"}}]})
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

/// A model that never answers the JSON shape fails the run loudly —
/// exit 1, no artifact written — instead of storing junk judgments.
#[test]
fn a_shapeless_judgment_fails_the_run_and_writes_nothing() {
    let server = Server::start("consolidation-cli-junk");
    seed(&server);
    let chat_url = stub_junk();
    let (code, stdout, stderr) = run_consolidation(
        &["--context", "sake", &server.base],
        &[
            ("TAGURU_EXTRACT_URL", chat_url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("not the required JSON shape"), "{stderr}");
    let (status, _) = server.call("GET", "/contexts/sake::consolidation", None);
    assert_eq!(
        status, 404,
        "a failed run must not have created the artifact"
    );
}

/// Issue #751: a base URL no request could leave on — unparseable, or
/// a scheme ureq does not speak — is a usage error (exit 2), caught
/// before drive() prints its "consolidation → URL" target line, the
/// same upfront refusal every other client verb gives it.
#[test]
fn a_malformed_url_is_a_usage_error_caught_before_the_target_line() {
    let (code, _stdout, stderr) =
        run_consolidation(&["--context", "sake", "--url", "not a url at all"], &[]);
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("is not a usable base URL"), "{stderr}");
    assert!(
        !stderr.contains("consolidation →"),
        "the refusal must land before the target line: {stderr}"
    );

    let (code, _stdout, stderr) =
        run_consolidation(&["--context", "sake", "--url", "ftp://127.0.0.1:9"], &[]);
    assert_eq!(code, 2, "{stderr}");
    assert!(
        stderr.contains("--url only supports http/https"),
        "{stderr}"
    );
}

// --- #753: judge-flow branches the #551 sweep found untested ---------------

/// A chat stub that dismisses everything — the counter twin of
/// `stub_judge`'s all-apply answers.
fn stub_dismiss(calls: Arc<Mutex<usize>>) -> String {
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
            *calls.lock().unwrap() += 1;
            let content = "{\"verdict\": \"dismiss\", \"action\": \"benign twins\", \
                           \"rationale\": \"別の蔵\"}";
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

/// Replaces the judgment artifact's manifest passage wholesale — the
/// same retract-then-apply door the CLI itself writes through.
fn overwrite_manifest(server: &Server, manifest_text: &str) {
    let batch = format!(
        "{}\n{}\n",
        json!({"taguru_batch": 1, "context": "sake::consolidation",
               "source": "consolidation:manifest"}),
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

/// A dismissal is a first-class judgment: counted as one in the
/// report and reused without an LLM call exactly like an apply.
#[test]
fn a_dismissal_is_counted_and_reused_like_any_judgment() {
    let server = Server::start("consolidation-dismiss");
    seed(&server);
    let calls = Arc::new(Mutex::new(0usize));
    let chat_url = stub_dismiss(Arc::clone(&calls));
    let extract_env = [
        ("TAGURU_EXTRACT_URL", chat_url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) =
        run_consolidation(&["--context", "sake", &server.base], &extract_env);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("judged 5 (0 apply, 5 dismiss), 0 reused"),
        "{stdout}"
    );
    assert_eq!(*calls.lock().unwrap(), 5);

    let (code, stdout, _) = run_consolidation(&["--context", "sake", &server.base], &extract_env);
    assert_eq!(code, 0, "{stdout}");
    assert!(
        stdout.contains("judgments up to date (5 reused, no LLM calls)"),
        "{stdout}"
    );
    assert_eq!(*calls.lock().unwrap(), 5);
}

/// A server whose audit answers a different detector is refused
/// before anything is judged or written — fingerprints from another
/// detector would be incomparable.
#[test]
fn a_foreign_server_detector_is_refused_outright() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let responses = [
            // GET /health for the skew preflight, then the audit.
            json!({"status": "ok"}).to_string(),
            json!({"result": {"detector": "consolidation/999"}}).to_string(),
        ];
        for body in responses {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut buffer = [0u8; 4096];
            let _ = stream.read(&mut buffer);
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        }
    });
    let (code, _stdout, stderr) = run_consolidation(
        &["--context", "sake", "--url", &format!("http://{addr}")],
        &[],
    );
    assert_eq!(code, 1, "{stderr}");
    assert!(
        stderr.contains("is not this build's (consolidation/1)"),
        "{stderr}"
    );
}

/// A stored manifest naming another detector re-judges everything,
/// loudly; a wrong or missing format stamp refuses outright — a
/// mangled artifact must never be silently diffed against.
#[test]
fn a_changed_stored_detector_rejudges_and_a_bad_stamp_refuses() {
    let server = Server::start("consolidation-restamp");
    seed(&server);
    let calls = Arc::new(Mutex::new(0usize));
    let chat_url = stub_judge(Arc::clone(&calls));
    let extract_env = [
        ("TAGURU_EXTRACT_URL", chat_url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, _) = run_consolidation(&["--context", "sake", &server.base], &extract_env);
    assert_eq!(code, 0, "{stdout}");
    assert_eq!(*calls.lock().unwrap(), 5);

    // Same format, different detector: every judgment re-costs, and
    // stderr says why.
    overwrite_manifest(
        &server,
        &json!({"taguru_consolidation": 1, "detector": "consolidation/0", "context": "sake"})
            .to_string(),
    );
    let (code, stdout, stderr) =
        run_consolidation(&["--context", "sake", &server.base], &extract_env);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stderr.contains("detector changed (consolidation/0 → consolidation/1)"),
        "{stderr}"
    );
    assert!(
        stdout.contains("judged 5 (5 apply, 0 dismiss), 0 reused"),
        "{stdout}"
    );
    assert_eq!(*calls.lock().unwrap(), 10);

    // A format stamp from another program: refused by number.
    overwrite_manifest(
        &server,
        &json!({"taguru_consolidation": 2, "detector": "consolidation/1", "context": "sake"})
            .to_string(),
    );
    let (code, _stdout, stderr) =
        run_consolidation(&["--context", "sake", &server.base], &extract_env);
    assert_eq!(code, 1, "{stderr}");
    assert!(
        stderr.contains("judgment artifact format 2 is not this build's 1"),
        "{stderr}"
    );

    // No stamp at all: refused by name.
    overwrite_manifest(
        &server,
        &json!({"detector": "consolidation/1", "context": "sake"}).to_string(),
    );
    let (code, _stdout, stderr) =
        run_consolidation(&["--context", "sake", &server.base], &extract_env);
    assert_eq!(code, 1, "{stderr}");
    assert!(
        stderr.contains("carries no taguru_consolidation stamp"),
        "{stderr}"
    );
}
