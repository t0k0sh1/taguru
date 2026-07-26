//! `taguru benchmark extract` against a stub OpenAI-compatible chat
//! endpoint that also answers (or 404s) the provider-probe paths
//! (`/api/show`, `/v1/models`) `taguru benchmark` itself calls before
//! any cell runs.

use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use crate::support::*;

/// Same request-reading shape as `extract.rs`'s own stub helper (one
/// HTTP/1.1 request off `stream`, headers up to the blank line then the
/// body per `Content-Length`) — duplicated here rather than shared,
/// since each cluster file owns its own protocol stub in this tree.
fn read_http_request(stream: &mut std::net::TcpStream) -> Option<(String, String)> {
    use std::io::Read;

    let mut buffer = Vec::new();
    let mut chunk = [0u8; 8192];
    let header_end = loop {
        let read = stream.read(&mut chunk).unwrap();
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(position) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break position + 4;
        }
    };
    let headers = String::from_utf8_lossy(&buffer[..header_end]).to_string();
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())?
        })
        .unwrap_or(0);
    while buffer.len() < header_end + content_length {
        let read = stream.read(&mut chunk).unwrap();
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
    }
    let body = String::from_utf8_lossy(&buffer[header_end..]).to_string();
    Some((headers, body))
}

fn request_path(headers: &str) -> String {
    headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or_default()
        .to_string()
}

fn chat_ok(content: &str) -> String {
    let payload = json!({
        "choices": [{"message": {"role": "assistant", "content": content}}]
    })
    .to_string();
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    )
}

fn http_response(status: u16, reason: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// One stub origin serving both the provider probe (`/api/show` 404s,
/// so `probe_model` falls back to `/v1/models`, which answers 200 with
/// an empty list) and the chat-completions path every spawned `extract`
/// child's `TAGURU_EXTRACT_URL` points at — the queued replies are
/// consumed in request order by chat calls only.
fn stub_provider(replies: Vec<String>) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let queue = Arc::new(Mutex::new(VecDeque::from(replies)));
    let captured_bodies = Arc::new(Mutex::new(Vec::new()));
    let captured_for_acceptor = Arc::clone(&captured_bodies);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let queue = Arc::clone(&queue);
            let captured = Arc::clone(&captured_for_acceptor);
            std::thread::spawn(move || {
                let Some((headers, body)) = read_http_request(&mut stream) else {
                    return;
                };
                let path = request_path(&headers);
                let response = match path.as_str() {
                    "/api/show" => http_response(404, "Not Found", "{}"),
                    "/v1/models" => http_response(200, "OK", r#"{"data":[]}"#),
                    _ => {
                        captured.lock().unwrap().push(body);
                        let reply =
                            queue.lock().unwrap().pop_front().unwrap_or_else(|| {
                                r#"{"associations":[],"aliases":[]}"#.to_string()
                            });
                        chat_ok(&reply)
                    }
                };
                let _ = stream.write_all(response.as_bytes());
            });
        }
    });
    (url, captured_bodies)
}

fn write_models_json(dir: &Path, entries: &[(&str, &str)]) -> PathBuf {
    let models: Vec<Value> = entries
        .iter()
        .map(|(id, url)| {
            json!({
                "id": id,
                "model": format!("{id}-model"),
                "url": format!("{url}/v1/chat/completions"),
            })
        })
        .collect();
    let path = dir.join("models.json");
    std::fs::write(
        &path,
        json!({"taguru_benchmark_models": 1, "models": models}).to_string(),
    )
    .unwrap();
    path
}

fn corpus_dir(tag: &str, docs: &[(&str, &str)]) -> PathBuf {
    let dir = batch_dir(&format!("bench-corpus-{tag}"));
    for (name, text) in docs {
        std::fs::write(dir.join(name), text).unwrap();
    }
    dir
}

fn results_dir(tag: &str) -> PathBuf {
    batch_dir(&format!("bench-results-{tag}"))
}

fn run_benchmark(args: &[&str]) -> (i32, String, String) {
    let mut full = vec!["benchmark", "extract"];
    full.extend_from_slice(args);
    run_cli(&full, &[])
}

#[test]
fn a_happy_path_matrix_produces_the_full_layout_and_runs_kind_sequence() {
    let (url, _captured) = stub_provider(vec![
        r#"{"associations":[{"subject":"青嶺酒造","label":"創業年","object":"1907年"}],"aliases":[]}"#
            .to_string(),
    ]);
    let corpus = corpus_dir("happy", &[("brewery.md", "青嶺酒造は1907年創業。")]);
    let out = results_dir("happy");
    let models = write_models_json(&out, &[("stub-a", &url)]);

    let (code, stdout, stderr) = run_benchmark(&[
        "--models",
        models.to_str().unwrap(),
        "--context",
        "bench",
        "--out",
        out.to_str().unwrap(),
        corpus.to_str().unwrap(),
    ]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["taguru_benchmark_manifest"], 1);
    assert_eq!(manifest["harness"]["execution"], "subprocess");
    assert_eq!(manifest["documents"].as_array().unwrap().len(), 1);
    assert_eq!(manifest["documents"][0]["document_id"], "brewery");
    assert_eq!(manifest["cells"].as_array().unwrap().len(), 1);
    assert_eq!(manifest["cells"][0]["outcome"], "complete");
    assert!(manifest["finished_at"].is_string());

    assert!(out.join("models.lock.json").is_file());
    let cell_dir = out.join("cells/stub-a/run01");
    assert!(cell_dir.join("diagnostics.jsonl").is_file());
    assert!(cell_dir.join("stdout.log").is_file());
    assert!(cell_dir.join("stderr.log").is_file());
    assert_eq!(
        std::fs::read_to_string(cell_dir.join("exit_code")).unwrap(),
        "0"
    );

    let runs_path = out.join("runs/stub-a.run01.jsonl");
    let runs_text = std::fs::read_to_string(&runs_path).unwrap();
    let lines: Vec<Value> = runs_text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let kinds: Vec<&str> = lines
        .iter()
        .map(|line| line["kind"].as_str().unwrap())
        .collect();
    assert_eq!(kinds.first(), Some(&"header"));
    assert_eq!(kinds.last(), Some(&"cell"));
    assert!(kinds.contains(&"attempt"), "{kinds:?}");
    assert_eq!(
        kinds.iter().filter(|k| **k == "document").count(),
        2,
        "one start, one end: {kinds:?}"
    );

    let attempt = lines.iter().find(|line| line["kind"] == "attempt").unwrap();
    assert_eq!(attempt["document_id"], "brewery");
    assert_eq!(attempt["model_id"], "stub-a");
    assert_eq!(attempt["cell_id"], "stub-a.run01");
    assert!(attempt["chunk_sha256"].is_string());
    assert!(
        attempt.get("state").is_some(),
        "Layer 1 fields pass through verbatim"
    );

    let cell_line = lines.last().unwrap();
    assert_eq!(cell_line["outcome"], "complete");
    assert_eq!(cell_line["exit_code"], 0);
}

#[test]
fn resuming_a_complete_matrix_calls_the_stub_zero_times() {
    let (url, captured) = stub_provider(vec![r#"{"associations":[],"aliases":[]}"#.to_string()]);
    let corpus = corpus_dir("resume", &[("only.md", "内容だけの文書。")]);
    let out = results_dir("resume");
    let models = write_models_json(&out, &[("stub-b", &url)]);

    let bench_args = [
        "--models",
        models.to_str().unwrap(),
        "--context",
        "bench",
        "--out",
        out.to_str().unwrap(),
        corpus.to_str().unwrap(),
    ];
    let (code, stdout, stderr) = run_benchmark(&bench_args);
    assert_eq!(code, 0, "first run — stdout: {stdout}\nstderr: {stderr}");
    let calls_after_first_run = captured.lock().unwrap().len();
    assert!(
        calls_after_first_run >= 1,
        "the first run must call the stub"
    );

    let (code, stdout, stderr) = run_benchmark(&bench_args);
    assert_eq!(code, 0, "resumed run — stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(
        captured.lock().unwrap().len(),
        calls_after_first_run,
        "a fully complete cell must never be re-run"
    );

    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["cells"].as_array().unwrap().len(), 1);
}

#[test]
fn a_models_json_edited_after_the_fact_refuses_to_resume() {
    let (url, _captured) = stub_provider(vec![r#"{"associations":[],"aliases":[]}"#.to_string()]);
    let corpus = corpus_dir("driftedconfig", &[("only.md", "内容。")]);
    let out = results_dir("driftedconfig");
    let models = write_models_json(&out, &[("stub-c", &url)]);

    let bench_args = [
        "--models",
        models.to_str().unwrap(),
        "--context",
        "bench",
        "--out",
        out.to_str().unwrap(),
        corpus.to_str().unwrap(),
    ];
    let (code, ..) = run_benchmark(&bench_args);
    assert_eq!(code, 0);

    // Edit models.json after the matrix completed — a changed matrix
    // definition must never share a results directory (ADR 0003 §6).
    std::fs::write(
        &models,
        json!({
            "taguru_benchmark_models": 1,
            "models": [{"id": "stub-c", "model": "different-model", "url": format!("{url}/v1/chat/completions")}]
        })
        .to_string(),
    )
    .unwrap();

    let (code, _stdout, stderr) = run_benchmark(&bench_args);
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("models.json has changed"), "{stderr}");
}

#[test]
fn a_cell_that_fails_every_document_is_recorded_failed_with_a_synthesized_end() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            std::thread::spawn(move || {
                let Some((headers, _body)) = read_http_request(&mut stream) else {
                    return;
                };
                let path = request_path(&headers);
                let response = match path.as_str() {
                    "/api/show" => http_response(404, "Not Found", "{}"),
                    "/v1/models" => http_response(200, "OK", r#"{"data":[]}"#),
                    _ => http_response(500, "Internal Server Error", "boom"),
                };
                let _ = stream.write_all(response.as_bytes());
            });
        }
    });
    let corpus = corpus_dir("failing", &[("only.md", "内容。")]);
    let out = results_dir("failing");
    let models = write_models_json(&out, &[("stub-d", &url)]);

    let (code, stdout, stderr) = run_benchmark(&[
        "--models",
        models.to_str().unwrap(),
        "--context",
        "bench",
        "--out",
        out.to_str().unwrap(),
        corpus.to_str().unwrap(),
    ]);
    assert_eq!(
        code, 0,
        "a cell recording failed documents is still a clean matrix run"
    );
    let _ = (stdout, stderr);

    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["cells"][0]["outcome"], "failed");

    let runs_text = std::fs::read_to_string(out.join("runs/stub-d.run01.jsonl")).unwrap();
    let lines: Vec<Value> = runs_text
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let end = lines
        .iter()
        .find(|line| line["kind"] == "document" && line["phase"] == "end")
        .expect("a synthesized failed end must exist");
    assert_eq!(end["outcome"], "failed");
    assert!(end["associations"].is_null());
    let cell_line = lines.last().unwrap();
    assert_eq!(cell_line["kind"], "cell");
    assert_eq!(cell_line["outcome"], "failed");
}
