//! The `taguru extract` document-to-batch pipeline against a stub chat server.

use std::process::{Command, Stdio};

use serde_json::{Value, json};

use crate::support::*;

/// A one-shot OpenAI-compatible chat stub: answers the canned
/// assistant texts in order, one connection per request, then hands
/// back every captured request (headers + body) through the join.
/// Reads one HTTP/1.1 request off `stream`: headers up to the blank
/// line, then the body per `Content-Length` (missing or unparsable
/// counts as 0). `None` if the connection closes before the headers
/// are complete — real I/O errors still panic, matching the stub
/// servers built on this that never expect one on localhost.
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

fn stub_chat_server(replies: Vec<String>) -> (String, std::thread::JoinHandle<Vec<String>>) {
    use std::io::Write;

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        let mut captured = Vec::new();
        for reply in replies {
            let (mut stream, _) = listener.accept().unwrap();
            let Some((headers, body)) = read_http_request(&mut stream) else {
                continue;
            };
            captured.push(format!("{headers}\n{body}"));
            let payload = json!({
                "choices": [{"message": {"role": "assistant", "content": reply}}]
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                payload.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
        }
        captured
    });
    (url, handle)
}

/// A concurrent, content-keyed OpenAI-compatible chat stub for
/// `--parallel` and retry tests: unlike `stub_chat_server`, every
/// accepted connection is handled on its own thread — so simultaneous
/// client connections are actually served simultaneously — and the
/// reply is chosen by the chunk index embedded in the request body
/// (the `"part K of N"` tag `extract.rs`'s `user_message` writes into
/// the user message, K one-based), not by arrival order, since
/// concurrent workers connect in OS-scheduled, non-deterministic
/// order. `respond` is called once per request with the zero-based
/// chunk index and that index's attempt number (0 the first time that
/// index is requested, 1 on its first retry, …) and returns the raw
/// HTTP response to write back.
///
/// The acceptor thread runs for the rest of the test process's life —
/// the same spawn-and-never-join shape
/// `the_extract_timeout_knob_bounds_a_stalled_provider` already uses —
/// because every caller here drives the client to completion before
/// inspecting the returned capture, so nothing is ever left waiting on
/// it.
fn stub_chat_server_concurrent<F>(
    respond: F,
) -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>)
where
    F: Fn(usize, usize) -> String + Send + Sync + 'static,
{
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let captured = Arc::new(Mutex::new(Vec::new()));
    let attempts: Arc<Mutex<std::collections::HashMap<usize, usize>>> =
        Arc::new(Mutex::new(std::collections::HashMap::new()));
    let respond = Arc::new(respond);
    let captured_for_acceptor = Arc::clone(&captured);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let respond = Arc::clone(&respond);
            let captured = Arc::clone(&captured_for_acceptor);
            let attempts = Arc::clone(&attempts);
            std::thread::spawn(move || {
                let Some((_headers, body)) = read_http_request(&mut stream) else {
                    return;
                };
                let index = chunk_index_of(&body);
                let attempt = {
                    let mut attempts = attempts.lock().unwrap();
                    let counter = attempts.entry(index).or_insert(0);
                    let seen = *counter;
                    *counter += 1;
                    seen
                };
                captured.lock().unwrap().push(body);
                let response = respond(index, attempt);
                let _ = stream.write_all(response.as_bytes());
            });
        }
    });
    (url, captured)
}

/// Pulls the zero-based chunk index out of the last user-role
/// message's `"part K of N"` tag; a single-chunk document carries no
/// such tag, so it is index 0.
fn chunk_index_of(body: &str) -> usize {
    let value: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let messages = value["messages"].as_array().cloned().unwrap_or_default();
    let content = messages
        .iter()
        .rev()
        .find(|message| message["role"] == "user")
        .and_then(|message| message["content"].as_str())
        .unwrap_or_default();
    content
        .find("part ")
        .and_then(|start| {
            let rest = &content[start + "part ".len()..];
            let end = rest.find(" of ")?;
            rest[..end].parse::<usize>().ok()
        })
        .and_then(|one_based: usize| one_based.checked_sub(1))
        .unwrap_or(0)
}

/// A 200 OK chat-completion response carrying `content` as the
/// assistant's answer — the same wire shape `stub_chat_server` sends.
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

/// Same wire shape as [`chat_ok`], but the choice also carries
/// `finish_reason` — drives extract.rs's `ChatClient::complete`
/// `finish_reason` plumbing (truncation-aware correction).
fn chat_ok_with_finish_reason(content: &str, finish_reason: &str) -> String {
    let payload = json!({
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }]
    })
    .to_string();
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    )
}

/// Same wire shape as [`chat_ok`], but the response also carries a
/// top-level `usage` object — drives `ChatClient::complete`'s token
/// capture for the `--diagnostics-out` sidecar (issue #200).
fn chat_ok_with_usage(content: &str, prompt_tokens: u64, completion_tokens: u64) -> String {
    let payload = json!({
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    })
    .to_string();
    format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        payload.len()
    )
}

/// An error chat-completion response — any status, an optional extra
/// header line (e.g. `"Retry-After: 1\r\n"`), and a plain-text body.
fn chat_error(status: u16, reason: &str, extra_header: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 {status} {reason}\r\n{extra_header}Content-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    )
}

/// A document long enough to split into many chunks: `count`
/// paragraphs, eight or so of which alone exceed `extract.rs`'s 24 KiB
/// chunk cap.
fn multi_chunk_document(count: usize) -> String {
    // "s value-" attests the stub answers' names (subject "S", object
    // "value-N") under the occurrence check (ADR 0013): "value-" plus
    // any digits covers ≥ 3/4 of "value-N" whichever chunk answers.
    (0..count)
        .map(|i| format!("Paragraph {i}: s value- {}", "x".repeat(3000)))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Pulls the chunk count out of `--dry-run`'s own report line:
/// `"{source}: would extract (N bytes, M chunk(s)) → {out_path}"`.
fn chunk_count_from_dry_run(stdout: &str) -> usize {
    let marker = " chunk(s))";
    let end = stdout
        .find(marker)
        .unwrap_or_else(|| panic!("no chunk count in: {stdout}"));
    let start = stdout[..end].rfind(' ').map(|i| i + 1).unwrap_or(0);
    stdout[start..end]
        .parse()
        .unwrap_or_else(|_| panic!("no chunk count in: {stdout}"))
}

/// The JSON body of one captured request. [`stub_chat_server`] stores
/// `"{headers}\n{body}"` while the concurrent stub stores the body
/// alone; chat bodies are single-line JSON, so the body is always the
/// text after the last newline either way.
fn json_body_of(request: &str) -> Value {
    let body = request.rsplit_once('\n').map_or(request, |(_, body)| body);
    serde_json::from_str(body).unwrap_or_else(|_| panic!("no JSON body in: {request}"))
}

/// The top-level keys of one captured request's JSON body, in wire
/// order — the "defaults add nothing" assertions compare against
/// exactly `["messages", "model", "temperature"]`.
fn top_level_keys(request: &str) -> Vec<String> {
    json_body_of(request)
        .as_object()
        .expect("a JSON object body")
        .keys()
        .cloned()
        .collect()
}

/// Every file in `dir` other than the extract manifest — a failed
/// document must leave none.
fn stray_batch_files(dir: &std::path::Path) -> Vec<std::ffi::OsString> {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name())
        .filter(|name| {
            // Both are expected, permanent siblings of the batch
            // files under `--out`, not stray output: the manifest
            // (skip-index of successes) and, since issue #179, the
            // chunk checkpoint directory (one file per document,
            // cleared but never removed itself once a document's
            // batch lands) — and, since ADR 0023, the trace directory
            // (one file per written document, the batch's sibling).
            name.to_str() != Some(".extract-manifest.json")
                && name.to_str() != Some(".extract-checkpoints")
                && name.to_str() != Some(".extract-trace")
        })
        .collect()
}

/// Parses a `--diagnostics-out` sidecar into its records, in file
/// order — one JSON object per line (issue #200). The sidecar is a
/// tagged stream since issue #262 (`kind`: `chunk`/`attempt`/
/// `document`); most callers want [`read_attempt_records`] instead.
fn read_diagnostics(path: &std::path::Path) -> Vec<Value> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("reading diagnostics file {}: {error}", path.display()));
    text.lines()
        .filter(|line| !line.is_empty())
        .map(|line| {
            serde_json::from_str(line)
                .unwrap_or_else(|error| panic!("bad diagnostics JSONL line {line:?}: {error}"))
        })
        .collect()
}

/// The `kind == "attempt"` records of a sidecar, in file order — the
/// issue #200 tests predate issue #262's `chunk`/`document` kinds and
/// reason about attempts alone, exactly as `--diagnostics-out` wrote
/// them before this issue.
fn read_attempt_records(path: &std::path::Path) -> Vec<Value> {
    read_diagnostics(path)
        .into_iter()
        .filter(|record| record["kind"] == "attempt")
        .collect()
}

/// Scrubs a developer shell's TAGURU_EXTRACT_*/TAGURU_CONFIG vars —
/// shared by [`run_extract`] and the one test that spawns its own child
/// to inspect mid-run state instead of going through it.
fn scrub_extract_env(command: &mut Command) -> &mut Command {
    command
        .env_remove("TAGURU_EXTRACT_URL")
        .env_remove("TAGURU_EXTRACT_MODEL")
        .env_remove("TAGURU_EXTRACT_API_KEY")
        .env_remove("TAGURU_EXTRACT_TIMEOUT_SECS")
        .env_remove("TAGURU_EXTRACT_PARALLEL")
        .env_remove("TAGURU_EXTRACT_FACT_BUDGET")
        .env_remove("TAGURU_EXTRACT_MAX_ATTEMPTS")
        .env_remove("TAGURU_EXTRACT_CORRECTIVE_CONTEXT_BYTES")
        .env_remove("TAGURU_EXTRACT_STRUCTURED_OUTPUT")
        .env_remove("TAGURU_EXTRACT_MAX_OUTPUT_TOKENS")
        .env_remove("TAGURU_EXTRACT_ESCALATION_FACTOR")
        .env_remove("TAGURU_EXTRACT_CHUNK_BYTES")
        .env_remove("TAGURU_EXTRACT_LOSSY")
        .env_remove("TAGURU_EXTRACT_CANDIDATES")
        .env_remove("TAGURU_EXTRACT_VOCABULARY")
        .env_remove("TAGURU_EXTRACT_COVERAGE")
        .env_remove("TAGURU_EXTRACT_DIAGNOSTICS")
        .env_remove("TAGURU_EXTRACT_TRACE_ATTEMPTS")
        .env_remove("TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES")
        .env_remove("TAGURU_EXTRACT_SCHEMA")
        .env_remove("TAGURU_EXTRACT_REPLAY")
        .env_remove("TAGURU_EXTRACT_REPLAY_FROM")
        .env_remove("TAGURU_CONFIG")
}

/// Runs `taguru extract`, hermetic like the other spawns: only the
/// given TAGURU_EXTRACT_* values reach it.
fn run_extract(
    out_dir: &std::path::Path,
    env: &[(&str, &str)],
    args: &[&str],
) -> (i32, String, String) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    scrub_extract_env(&mut command).arg("extract");
    for (key, value) in env {
        command.env(key, value);
    }
    let output = command
        .args(["--out", out_dir.to_str().unwrap()])
        .args(args)
        .output()
        .expect("extract must run");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn extraction_turns_documents_into_batches_import_applies_and_the_server_serves() {
    let docs = batch_dir("extract-docs");
    let aomine = docs.join("aomine.md");
    let takase = docs.join("takase.md");
    std::fs::write(
        &aomine,
        "青嶺酒造は1907年に創業した。\n\n杜氏は高瀬。大量生産は行わない。",
    )
    .unwrap();
    std::fs::write(&takase, "高瀬は青嶺酒造の杜氏。").unwrap();
    let aomine_src = aomine.to_str().unwrap();
    let takase_src = takase.to_str().unwrap();

    // Dry run: no provider configured, nothing called, nothing written.
    let out = batch_dir("extract-out");
    let (code, stdout, stderr) = run_extract(
        &out,
        &[],
        &["--dry-run", "--context", "sake", aomine_src, takase_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(stdout.matches("would extract").count(), 2, "{stdout}");

    // The real run. aomine answers fenced (the extractor must strip
    // markdown fences) and carries one duplicate triple, one alias
    // whose canonical exists nowhere, and one null-valued item — real
    // models emit all three. takase answers garbage first — one
    // corrective turn — then a valid object with weight omitted.
    // Paragraph 0 is the founding sentence; paragraph 1 is the brewer
    // and no-mass-production sentence — the tagged values below match
    // where each fact actually sits in the source text above.
    let aomine_reply = json!({
        "associations": [
            {"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "paragraph": 0},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0, "paragraph": 1},
            {"subject": "青嶺酒造", "label": "行う", "object": "大量生産", "weight": -1.0, "paragraph": 1},
            {"subject": "青嶺酒造", "label": "所在地", "object": null},
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0, "paragraph": 1}
        ],
        "aliases": [
            {"alias": "Aomine", "canonical": "青嶺酒造", "kind": "concept"},
            {"alias": "幽霊", "canonical": "存在しない", "kind": "concept"}
        ]
    })
    .to_string();
    // takase's reply omits paragraph entirely — the missing-tag path
    // must still leave the fact in place (asserted below via the
    // server responses, since a dropped fact wouldn't come back at all).
    let takase_reply =
        json!({"associations": [{"subject": "高瀬", "label": "所属", "object": "青嶺酒造"}]})
            .to_string();
    let (url, requests) = stub_chat_server(vec![
        format!("```json\n{aomine_reply}\n```"),
        "Sure! Here are the facts I found.".to_string(),
        takase_reply.clone(),
    ]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ("TAGURU_EXTRACT_API_KEY", "sekrit"),
    ];
    // Issue #199: the default (strict) mode would turn aomine's
    // null-valued item and dangling alias into corrective turns instead
    // of silent drops — this test is about the pipeline end to end
    // (fences, dedup, corrective-on-garbage, manifest, import, serving),
    // not #199's own corrective behavior (covered by the dedicated
    // strict-mode tests below), so it opts into the pre-#199
    // drop-and-proceed behavior explicitly.
    let (code, stdout, stderr) = run_extract(
        &out,
        &provider,
        &[
            "--lossy",
            "--context",
            "sake",
            "--description",
            "酒蔵の記憶",
            aomine_src,
            takase_src,
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("3 association(s)"), "{stdout}");
    assert!(stdout.contains("1 duplicate(s) folded"), "{stdout}");
    assert!(stdout.contains("2 item(s) dropped (--lossy)"), "{stdout}");
    assert!(stdout.contains("2 written"), "{stdout}");

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 3);
    assert!(requests[0].contains("Bearer sekrit"), "{}", requests[0]);
    assert!(
        requests[0].contains("青嶺酒造は1907年に創業した。"),
        "{}",
        requests[0]
    );
    // Every paragraph is numbered for the model now, questions or not
    // — the same indexes aomine_reply's associations tag themselves
    // with above.
    assert!(
        requests[0].contains("[0] 青嶺酒造は1907年に創業した。"),
        "{}",
        requests[0]
    );
    assert!(
        requests[0].contains("[1] 杜氏は高瀬。大量生産は行わない。"),
        "{}",
        requests[0]
    );
    // The second document's prompt carries the first document's labels…
    assert!(
        requests[1].contains("創業年"),
        "vocabulary did not accumulate: {}",
        requests[1]
    );
    // …and the corrective turn asks again after the garbage answer.
    assert!(
        requests[2].contains("only the JSON object"),
        "{}",
        requests[2]
    );

    // Import applies what extract wrote; the server serves the facts,
    // the alias entry, the negative weight, and the original passage.
    let data_dir = common::scratch_dir("http-extract");
    let (code, stdout, stderr) = run_import(&data_dir, &[out.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let server = Server::start_on("extract-serve", data_dir);
    let brewer = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "Aomine", "label": "杜氏"})),
    );
    assert_eq!(brewer["matches"][0]["object"], json!("高瀬"));
    let negated = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "青嶺酒造", "label": "行う"})),
    );
    assert_eq!(negated["matches"][0]["weight"], json!(-1.0));
    let membership = server.ok(
        "POST",
        "/contexts/sake/query",
        Some(json!({"subject": "高瀬", "label": "所属"})),
    );
    assert_eq!(membership["matches"][0]["weight"], json!(1.0));
    let passages = server.ok(
        "POST",
        "/contexts/sake/sources/lookup",
        Some(json!({"sources": [aomine_src]})),
    );
    assert_eq!(
        passages["passages"][aomine_src],
        json!("青嶺酒造は1907年に創業した。\n\n杜氏は高瀬。大量生産は行わない。")
    );
    drop(server);

    // Unchanged documents skip without a single model call: the
    // endpoint here refuses every connection, so an attempt would fail
    // loudly instead of passing. Every input the manifest keys on must
    // match the first run bit-for-bit — including --description, which
    // the manifest treats as a computation input (it is baked into the
    // batch header's create block) even though import ignores that
    // block once the context already exists; dropping it here would
    // legitimately change the manifest key and force a real re-extract.
    let dead = [
        ("TAGURU_EXTRACT_URL", "http://127.0.0.1:9"),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) = run_extract(
        &out,
        &dead,
        &[
            "--lossy",
            "--context",
            "sake",
            "--description",
            "酒蔵の記憶",
            aomine_src,
            takase_src,
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(stdout.matches("unchanged, skipped").count(), 2, "{stdout}");

    // --force re-extracts both.
    let (url, requests) = stub_chat_server(vec![aomine_reply.clone(), takase_reply.clone()]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) = run_extract(
        &out,
        &provider,
        &[
            "--lossy",
            "--force",
            "--context",
            "sake",
            aomine_src,
            takase_src,
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("2 written"), "{stdout}");
    assert_eq!(requests.join().unwrap().len(), 2);

    // A re-pointed --context re-extracts too — a skip would leave
    // files whose headers still send everything to 'sake'.
    let (url, requests) = stub_chat_server(vec![aomine_reply, takase_reply]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) = run_extract(
        &out,
        &provider,
        &["--lossy", "--context", "vats", aomine_src, takase_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(!stdout.contains("unchanged, skipped"), "{stdout}");
    assert!(stdout.contains("2 written"), "{stdout}");
    assert_eq!(requests.join().unwrap().len(), 2);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// `--parallel N` must produce stdout — and the written batch file —
/// byte-for-byte identical to the sequential (default) run of the
/// same multi-chunk document, whatever N is or however its worker
/// threads happen to race: the same property
/// `compact_parallel_output_matches_the_sequential_run_byte_for_byte`
/// guarantees for `compact --parallel`.
#[test]
fn extract_parallel_output_matches_the_sequential_run_byte_for_byte() {
    let docs = batch_dir("extract-par-docs");
    let doc = docs.join("big.md");
    std::fs::write(&doc, multi_chunk_document(50)).unwrap();
    let doc_src = doc.to_str().unwrap();

    let probe = batch_dir("extract-par-probe");
    let (code, dry_stdout, stderr) =
        run_extract(&probe, &[], &["--dry-run", "--context", "c", doc_src]);
    assert_eq!(code, 0, "stdout: {dry_stdout}\nstderr: {stderr}");
    let total_chunks = chunk_count_from_dry_run(&dry_stdout);
    assert!(
        total_chunks >= 4,
        "fixture must span several chunks to exercise concurrency: {dry_stdout}"
    );

    fn reply_for(index: usize) -> String {
        json!({"associations": [
            {"subject": "S", "label": "chunk", "object": format!("value-{index}"), "weight": 1.0}
        ]})
        .to_string()
    }

    let seq_out = batch_dir("extract-par-seq-out");
    let (seq_url, _seq_captured) =
        stub_chat_server_concurrent(|index, _attempt| chat_ok(&reply_for(index)));
    let (code, seq_stdout, stderr) = run_extract(
        &seq_out,
        &[
            ("TAGURU_EXTRACT_URL", seq_url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc_src],
    );
    assert_eq!(code, 0, "stdout: {seq_stdout}\nstderr: {stderr}");
    assert!(seq_stdout.contains("1 written"), "{seq_stdout}");

    let par_out = batch_dir("extract-par-par-out");
    let (par_url, par_captured) =
        stub_chat_server_concurrent(|index, _attempt| chat_ok(&reply_for(index)));
    let (code, par_stdout, stderr) = run_extract(
        &par_out,
        &[
            ("TAGURU_EXTRACT_URL", par_url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", "--parallel", "4", doc_src],
    );
    assert_eq!(code, 0, "stdout: {par_stdout}\nstderr: {stderr}");

    // `Run::report` embeds the run's own `--out` directory in its
    // trailing `→ {out_path}`, and the two runs necessarily write to
    // differently-named scratch directories. Normalize each run's own
    // out-dir prefix to a shared placeholder before comparing, so the
    // assertion still catches any real divergence (association/alias
    // counts, ordering, summary line) without being defeated by that
    // incidental path difference.
    let seq_stdout_normalized = seq_stdout.replace(seq_out.to_str().unwrap(), "OUT_DIR");
    let par_stdout_normalized = par_stdout.replace(par_out.to_str().unwrap(), "OUT_DIR");
    assert_eq!(
        seq_stdout_normalized, par_stdout_normalized,
        "--parallel output must match the sequential run byte for byte, modulo the out-dir path"
    );
    assert_eq!(
        par_captured.lock().unwrap().len(),
        total_chunks,
        "every chunk must be dispatched exactly once when nothing fails"
    );

    let seq_files = stray_batch_files(&seq_out);
    let par_files = stray_batch_files(&par_out);
    assert_eq!(seq_files.len(), 1, "{seq_files:?}");
    assert_eq!(par_files.len(), 1, "{par_files:?}");
    let seq_body = std::fs::read_to_string(seq_out.join(&seq_files[0])).unwrap();
    let par_body = std::fs::read_to_string(par_out.join(&par_files[0])).unwrap();
    assert_eq!(
        seq_body, par_body,
        "the written batch files must match byte for byte too"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&probe);
    let _ = std::fs::remove_dir_all(&seq_out);
    let _ = std::fs::remove_dir_all(&par_out);
}

/// A bad `--parallel` flag value is refused before anything boots — no
/// provider needs to be configured, since `Args::parse` rejects it
/// before `ChatClient::from_env` is ever reached.
#[test]
fn extract_rejects_a_non_positive_parallel_flag() {
    let docs = batch_dir("extract-badflag-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "content").unwrap();
    let out = batch_dir("extract-badflag-out");

    let (code, _, stderr) = run_extract(
        &out,
        &[],
        &["--context", "c", "--parallel", "0", doc.to_str().unwrap()],
    );
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("--parallel needs an integer"), "{stderr}");

    let (code, _, stderr) = run_extract(
        &out,
        &[],
        &[
            "--context",
            "c",
            "--parallel",
            "nope",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("--parallel needs an integer"), "{stderr}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// `TAGURU_EXTRACT_PARALLEL` is validated with the same strength as
/// the flag — this half of the check only runs once a provider is
/// configured, since it happens after `ChatClient::from_env` in `run`.
#[test]
fn extract_rejects_a_non_positive_parallel_env_var() {
    let docs = batch_dir("extract-badenv-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "content").unwrap();
    let out = batch_dir("extract-badenv-out");

    for bad in ["0", "nope"] {
        let provider = [
            ("TAGURU_EXTRACT_URL", "http://127.0.0.1:9"),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_PARALLEL", bad),
        ];
        let (code, _, stderr) =
            run_extract(&out, &provider, &["--context", "c", doc.to_str().unwrap()]);
        assert_eq!(code, 2, "{bad}: {stderr}");
        assert!(
            stderr.contains("TAGURU_EXTRACT_PARALLEL needs an integer"),
            "{bad}: {stderr}"
        );
    }

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// `--parallel` wins over `TAGURU_EXTRACT_PARALLEL` outright — the
/// flag short-circuits before the environment variable is even read,
/// so a garbage env var value must not surface as an error.
#[test]
fn extract_parallel_flag_overrides_the_environment_variable() {
    let docs = batch_dir("extract-flagwins-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-flagwins-out");

    let (url, _requests) = stub_chat_server(vec![
        json!({"associations": [{"subject": "S", "label": "L", "object": "O", "weight": 1.0}]})
            .to_string(),
    ]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ("TAGURU_EXTRACT_PARALLEL", "not-a-number"),
    ];
    let (code, stdout, stderr) = run_extract(
        &out,
        &provider,
        &["--context", "c", "--parallel", "2", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("1 written"), "{stdout}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// A chunk failure fails the whole document (no partial batch file —
/// `extract_document` propagates the error before `merge` ever runs)
/// and the worker pool must not dispatch the tail past the failing
/// index once it has been recorded.
#[test]
fn a_failing_chunk_fails_the_document_without_dispatching_the_tail() {
    let docs = batch_dir("extract-earlystop-docs");
    let doc = docs.join("big.md");
    std::fs::write(&doc, multi_chunk_document(200)).unwrap();
    let doc_src = doc.to_str().unwrap();
    let out = batch_dir("extract-earlystop-out");

    let (code, dry_stdout, stderr) =
        run_extract(&out, &[], &["--dry-run", "--context", "c", doc_src]);
    assert_eq!(code, 0, "stdout: {dry_stdout}\nstderr: {stderr}");
    let total_chunks = chunk_count_from_dry_run(&dry_stdout);
    let failing_index = 1usize;
    assert!(
        total_chunks >= 12,
        "fixture must leave a real tail past the failure to prove it was skipped: {dry_stdout}"
    );

    let (url, captured) = stub_chat_server_concurrent(move |index, _attempt| {
        if index == failing_index {
            chat_error(400, "Bad Request", "", "no thanks")
        } else {
            // The failure must be recorded before any other worker claims
            // far past it: same fix as
            // dispatch_chunks_concurrently_bounds_spillover_past_a_promptly_recorded_failure
            // in registry.rs — an instant failure raced against
            // slowed-down successes, so the bound asserted below holds
            // instead of racing the stub server's own thread scheduling
            // (without this, a CI runner busy enough to delay the failing
            // request past several successes can let a worker claim well
            // beyond `failing_index + workers`, which is exactly the
            // best-effort spillover dispatch_chunks_concurrently's own
            // doc comment says is unbounded once a failure is slow to
            // surface). 100ms rather than 20ms: the coverage job's
            // instrumented binary is slow enough, and its capped
            // --test-threads busy enough, to burn through a 20ms margin
            // on a loaded runner and flake this exact assertion.
            std::thread::sleep(std::time::Duration::from_millis(100));
            chat_ok(&json!({"associations": []}).to_string())
        }
    });
    let workers = 3usize;
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--parallel",
            &workers.to_string(),
            doc_src,
        ],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stderr.contains(&format!("chunk {}/{total_chunks}", failing_index + 1)),
        "{stderr}"
    );
    let stray = stray_batch_files(&out);
    assert!(
        stray.is_empty(),
        "a failed document must not leave a batch file behind: {stray:?}"
    );

    let seen = captured.lock().unwrap();
    let last_index = total_chunks - 1;
    assert!(
        !seen.iter().any(|body| chunk_index_of(body) == last_index),
        "the tail past the failure must never be dispatched: saw {} requests",
        seen.len()
    );
    assert!(
        seen.len() < total_chunks,
        "early stop must dispatch fewer than every chunk: saw {}/{total_chunks}",
        seen.len()
    );
    // A worker can only ever have one chunk in flight at a time, so once
    // `first_failure` is recorded, at most `workers` chunks beyond it can
    // already be claimed and running. This bound only holds because the
    // stub above stalls every success while returning the failure
    // instantly, promptly recording it — the same precondition
    // dispatch_chunks_concurrently_bounds_spillover_past_a_promptly_recorded_failure
    // in registry.rs relies on. It also catches a regression to a weaker
    // atomic ordering that lets a worker miss the failure update.
    let max_in_flight_past_failure = failing_index + workers;
    let stray_indexes: Vec<usize> = seen
        .iter()
        .map(|body| chunk_index_of(body))
        .filter(|&index| index > max_in_flight_past_failure)
        .collect();
    assert!(
        stray_indexes.is_empty(),
        "no worker may claim a chunk more than `workers` past the failure \
         once it is recorded: saw claims at {stray_indexes:?} (failure at \
         {failing_index}, {workers} workers)"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// #763: a document that fails on a later chunk keeps the earlier
/// chunks' checkpoints, the failure line says so, and a plain rerun
/// resumes from them — only the failed chunk is asked again.
#[test]
fn a_late_chunk_failure_names_its_checkpoints_and_a_rerun_resumes_from_them() {
    let docs = batch_dir("extract-resume-hint-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, format!("{}\n\n{}", "a".repeat(600), "b".repeat(600))).unwrap();
    let out = batch_dir("extract-resume-hint-out");
    let args = [
        "--context",
        "c",
        "--chunk-bytes",
        "700",
        doc.to_str().unwrap(),
    ];

    let (url, captured) = stub_chat_server_concurrent(|index, _attempt| {
        if index == 1 {
            chat_error(400, "Bad Request", "", "no thanks")
        } else {
            chat_ok(&json!({"associations": []}).to_string())
        }
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &args,
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("chunk 2/2"), "{stderr}");
    assert!(
        stderr.contains(
            "(1 extracted unit(s) are checkpointed — a rerun without --force resumes from them)"
        ),
        "{stderr}"
    );
    assert_eq!(captured.lock().unwrap().len(), 2);

    // The rerun asks only for the chunk that failed.
    let (url, captured) = stub_chat_server_concurrent(|_index, _attempt| {
        chat_ok(&json!({"associations": []}).to_string())
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &args,
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("1 written"), "{stdout}");
    let seen = captured.lock().unwrap();
    assert_eq!(seen.len(), 1, "only the failed chunk is re-asked: {seen:?}");
    assert_eq!(chunk_index_of(&seen[0]), 1);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// A transient 500 recovers on retry — RETRY_ATTEMPTS now covers more
/// than the one immediate retry the old fixed-sleep policy gave.
#[test]
fn a_transient_five_hundred_recovers_on_retry() {
    let docs = batch_dir("extract-retry500-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-retry500-out");

    let (url, _captured) = stub_chat_server_concurrent(|_index, attempt| {
        if attempt == 0 {
            chat_error(500, "Internal Server Error", "", "transient")
        } else {
            chat_ok(&json!({"associations": []}).to_string())
        }
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("1 written"), "{stdout}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// A 429 carrying `Retry-After` uses that delay verbatim instead of
/// the computed jittered backoff.
#[test]
fn a_429_with_retry_after_is_honored_before_the_retry() {
    let docs = batch_dir("extract-retryafter-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-retryafter-out");

    let (url, _captured) = stub_chat_server_concurrent(|_index, attempt| {
        if attempt == 0 {
            chat_error(429, "Too Many Requests", "Retry-After: 1\r\n", "slow down")
        } else {
            chat_ok(&json!({"associations": []}).to_string())
        }
    });
    let started = std::time::Instant::now();
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    let elapsed = started.elapsed();
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("1 written"), "{stdout}");
    assert!(
        elapsed >= std::time::Duration::from_secs(1),
        "Retry-After: 1 must be honored, took {elapsed:?}"
    );
    assert!(
        elapsed < std::time::Duration::from_secs(10),
        "the wait must not run away, took {elapsed:?}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// ADR 0014 (#496 S2): `--candidates` folds the document's own
/// segmented names into the system prompt, non-restrictively; the
/// default run sends no such block and stays byte-for-byte pre-S2.
#[test]
fn extract_candidates_flag_folds_the_document_names_into_the_system_prompt() {
    let docs = batch_dir("extract-candidates-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "CI のテストランナーは cargo-nextest。").unwrap();
    let out = batch_dir("extract-candidates-out");

    let reply = json!({"associations": [
        {"subject": "CI", "label": "テストランナー", "object": "cargo-nextest"}
    ]})
    .to_string();
    let (url, requests) = stub_chat_server(vec![reply.clone()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", "--candidates", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 1);
    // Inspect the SYSTEM message's own candidate block, not the whole
    // request — the user message quotes the document, which contains
    // the same names, so a whole-body grep would pass with candidate
    // extraction broken.
    let body_start = requests[0].find('{').expect("request carries a JSON body");
    let body: serde_json::Value = serde_json::from_str(&requests[0][body_start..]).unwrap();
    let system = body["messages"][0]["content"].as_str().unwrap();
    assert_eq!(body["messages"][0]["role"], "system");
    let block = system
        .split("Names appearing in this document")
        .nth(1)
        .unwrap_or_else(|| panic!("no candidate block in the system prompt: {system}"));
    assert!(block.contains("cargo-nextest"), "{block}");
    assert!(
        block.contains("still allowed"),
        "non-restrictive contract: {block}"
    );

    // The same run WITHOUT the flag must not send the block — and must
    // re-extract (candidates is a computation input), not skip.
    let (url, requests) = stub_chat_server(vec![reply]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(!stdout.contains("unchanged, skipped"), "{stdout}");
    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(
        !requests[0].contains("Names appearing in this document"),
        "{}",
        requests[0]
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// ADR 0016 (#496 S4): `--coverage` reports every candidate-pair
/// sentence no accepted association covers — one stderr line each, a
/// count on the report line, an `uncovered` count in the diagnostics
/// document record — and judges a manifest-skipped document from its
/// already-written batch, calling no model at all.
#[test]
fn extract_coverage_reports_uncovered_candidate_pair_sentences() {
    let docs = batch_dir("extract-coverage-docs");
    let doc = docs.join("ops.md");
    std::fs::write(
        &doc,
        "バックアップはS3へ保存する。\n\n- 頻度: 日次\n- 保持期間: 30日",
    )
    .unwrap();
    let out = batch_dir("extract-coverage-out");
    let diagnostics = out.join("diag.jsonl");

    // The reply covers the first sentence (subject+object) and the
    // retention line (label+object) but never the frequency line —
    // the systematically-dropped fact shape the check exists to name.
    let reply = json!({"associations": [
        {"subject": "バックアップ", "label": "保存先", "object": "S3", "weight": 1.0, "paragraph": 0},
        {"subject": "バックアップ", "label": "保持期間", "object": "30日", "weight": 1.0, "paragraph": 1}
    ]})
    .to_string();
    let (url, requests) = stub_chat_server(vec![reply]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "ops",
            "--coverage",
            "--diagnostics-out",
            diagnostics.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("1 sentence(s) uncovered (coverage)"),
        "{stdout}"
    );
    assert!(
        stderr.contains("uncovered: [paragraph 1] - 頻度: 日次"),
        "{stderr}"
    );
    let document = read_diagnostics(&diagnostics)
        .into_iter()
        .find(|record| record["kind"] == "document")
        .expect("a document record");
    assert_eq!(document["uncovered"], 1);
    requests.join().unwrap();

    // ADR 0026 (#787): the trace carries the same gap with the FULL
    // sentence and the paragraph's text, plus one `paragraph` record
    // per canonical paragraph — text attached exactly when no kept
    // item cites it — so coverage rate (count- or byte-weighted) is a
    // fold over the records.
    // (`read_trace` expects the batch to be --out's only stray file;
    // this test also parks diag.jsonl there, so address the trace by
    // the batch name directly.)
    let batch_name = stray_batch_files(&out)
        .into_iter()
        .map(|name| name.to_string_lossy().into_owned())
        .find(|name| name != "diag.jsonl")
        .unwrap();
    let trace: Vec<Value> = std::fs::read_to_string(out.join(".extract-trace").join(&batch_name))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let coverage: Vec<&Value> = trace.iter().filter(|r| r["kind"] == "paragraph").collect();
    assert_eq!(coverage.len(), 2, "{coverage:?}");
    assert_eq!(coverage[0]["paragraph"], 0);
    assert_eq!(coverage[0]["covered"], true);
    assert_eq!(coverage[0]["items"], 1);
    assert!(coverage[0].get("text").is_none());
    assert_eq!(coverage[0]["bytes"], "バックアップはS3へ保存する。".len());
    assert_eq!(coverage[1]["covered"], true, "paragraph 1 is cited too");
    let gaps: Vec<&Value> = trace.iter().filter(|r| r["kind"] == "uncovered").collect();
    assert_eq!(gaps.len(), 1, "{gaps:?}");
    assert_eq!(gaps[0]["paragraph"], 1);
    assert_eq!(gaps[0]["sentence"], "- 頻度: 日次");
    assert_eq!(
        gaps[0]["text"],
        "- 頻度: 日次
- 保持期間: 30日"
    );
    let chunk = trace.iter().find(|r| r["kind"] == "chunk").unwrap();
    assert_eq!(gaps[0]["chunk_index"], chunk["chunk_index"]);
    assert_eq!(gaps[0]["chunk_sha256"], chunk["chunk_sha256"]);

    // Report-only: the flag is not a computation input, so a rerun
    // skips — and still judges the gap from the batch it already
    // wrote. It must call nothing: the stub above accepted exactly
    // one connection, so a second request would fail this run.
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "ops", "--coverage", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("unchanged, skipped"), "{stdout}");
    assert!(
        stderr.contains("uncovered: [paragraph 1] - 頻度: 日次"),
        "{stderr}"
    );
    // EXACTLY one gap: the skip path must judge with the batch's real
    // associations — an empty or fabricated triple set would flag the
    // two covered sentences as well.
    assert_eq!(stderr.matches("uncovered:").count(), 1, "{stderr}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// The `/{stem}` suffix can push a within-cap `--source-id` over the
/// 1024-byte source cap only at extract time — exactly at the cap
/// both documents pass; one byte over fails them before any model
/// call.
#[test]
fn extract_suffixed_source_ids_respect_the_name_cap() {
    let docs = batch_dir("extract-source-cap-docs");
    std::fs::write(docs.join("aa.md"), "壱。").unwrap();
    std::fs::write(docs.join("bb.md"), "弐。").unwrap();
    let out = batch_dir("extract-source-cap-out");

    // 1022 + "/" + 2-byte stem = 1025: one over MAX_NAME_BYTES (1024).
    let over = "s".repeat(1022);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", "http://127.0.0.1:9/v1"),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "ops",
            "--source-id",
            &over,
            docs.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("source cap"), "{stderr}");

    // 1021 + "/" + 2 = 1024: exactly at the cap, both documents land.
    let at_cap = "s".repeat(1021);
    let reply = |name: &str, text: &str| {
        json!({"associations": [
            {"subject": name, "label": "rel", "object": text, "weight": 1.0, "paragraph": 0}
        ]})
        .to_string()
    };
    let (url, requests) = stub_chat_server(vec![reply("壱", "壱。"), reply("弐", "弐。")]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "ops",
            "--source-id",
            &at_cap,
            docs.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    requests.join().unwrap();

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// The `--source-id` usage errors name their actual cause: a
/// duplicate says "given twice", an empty id says "must not be
/// empty" — both exit 2, so only the wording tells the operator
/// which mistake they made.
#[test]
fn extract_source_id_usage_errors_name_their_cause() {
    let out = batch_dir("extract-source-id-usage-out");
    let (code, _, stderr) = run_extract(
        &out,
        &[],
        &[
            "--context",
            "c",
            "--source-id",
            "a",
            "--source-id",
            "b",
            "doc.md",
        ],
    );
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("--source-id given twice"), "{stderr}");
    let (code, _, stderr) =
        run_extract(&out, &[], &["--context", "c", "--source-id", "", "doc.md"]);
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("--source-id must not be empty"), "{stderr}");
    let _ = std::fs::remove_dir_all(&out);
}

/// #466 S1 (ADR 0017): `--source-id`/`--date`/`--tag` bake the
/// promotion runbook's conventions into the written batch — the
/// session source id (with the `/{doc}` stem suffix across several
/// documents), the passage line's date and tags — and all three are
/// manifest computation inputs: same flags skip, a changed date
/// rewrites. A source-id collision between two documents fails the
/// second instead of letting import fold them into one another.
#[test]
fn extract_bakes_the_runbook_conventions_into_the_batch() {
    let docs = batch_dir("extract-runbook-docs");
    std::fs::write(docs.join("s1.md"), "青嶺酒造は1907年に創業した。").unwrap();
    std::fs::write(docs.join("s2.md"), "杜氏は高瀬。").unwrap();
    let out = batch_dir("extract-runbook-out");

    let replies = [
        json!({"associations": [
            {"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0, "paragraph": 0}
        ]})
        .to_string(),
        json!({"associations": [
            {"subject": "高瀬", "label": "役職", "object": "杜氏", "weight": 1.0, "paragraph": 0}
        ]})
        .to_string(),
    ];
    let flags = [
        "--context",
        "ops",
        "--source-id",
        "session:claude:abc",
        "--date",
        "2026-08-06",
        "--tag",
        "ops",
        "--tag",
        "リリース",
    ];
    let (url, requests) = stub_chat_server(replies.to_vec());
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let mut args: Vec<&str> = flags.to_vec();
    let docs_arg = docs.to_str().unwrap().to_string();
    args.push(&docs_arg);
    let (code, stdout, stderr) = run_extract(&out, &provider, &args);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    requests.join().unwrap();

    // Several documents: each header carries ID/{stem}; the passage
    // line carries the date (2026-08-06 UTC midnight) and the tags.
    for (file, expected_source) in [
        ("s1.md", "session:claude:abc/s1"),
        ("s2.md", "session:claude:abc/s2"),
    ] {
        let batch_file = std::fs::read_dir(&out)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| {
                path.file_name()
                    .is_some_and(|name| name.to_string_lossy().contains(&file.replace(".md", "")))
                    && path.extension().is_some_and(|ext| ext == "jsonl")
            })
            .unwrap_or_else(|| panic!("a batch file for {file}"));
        let body = std::fs::read_to_string(&batch_file).unwrap();
        let header: Value = serde_json::from_str(body.lines().next().unwrap()).unwrap();
        assert_eq!(header["source"], expected_source, "{body}");
        let passage: Value = serde_json::from_str(body.lines().nth(1).unwrap()).unwrap();
        assert_eq!(passage["date"], 1785974400u64, "{body}");
        assert_eq!(passage["tags"], json!(["ops", "リリース"]), "{body}");
    }

    // Same flags again: both documents skip (the metadata is in the
    // fingerprint and unchanged). No model call — the stub above
    // accepted exactly two connections.
    let (code, stdout, stderr) = run_extract(&out, &provider, &args);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(stdout.matches("unchanged, skipped").count(), 2, "{stdout}");

    // A changed date must rewrite — a skip would leave the old date in
    // the emitted file.
    let (url, requests) = stub_chat_server(replies.to_vec());
    let mut redated: Vec<&str> = args.clone();
    let position = redated.iter().position(|a| *a == "2026-08-06").unwrap();
    redated[position] = "2026-08-07";
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &redated,
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(!stdout.contains("unchanged, skipped"), "{stdout}");
    requests.join().unwrap();

    // A single document takes the id verbatim — no suffix to invent.
    let (url, requests) = stub_chat_server(vec![replies[0].clone()]);
    let single = docs.join("s1.md");
    let mut single_args: Vec<&str> = flags.to_vec();
    let single_arg = single.to_str().unwrap().to_string();
    single_args.push(&single_arg);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &single_args,
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    requests.join().unwrap();
    // Batch file names carry an unconditional hash suffix (issue
    // #730), so the file is found by its header's source id rather
    // than by recomputing the name.
    std::fs::read_dir(&out)
        .unwrap()
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .filter_map(|path| std::fs::read_to_string(path).ok())
        .find(|body| {
            serde_json::from_str::<Value>(body.lines().next().unwrap_or(""))
                .ok()
                .is_some_and(|header| header["source"] == "session:claude:abc")
        })
        .expect("a single document takes the --source-id verbatim into its batch header");

    // Two documents whose stems collide would land on ONE source id —
    // import's per-source retract-then-apply would fold them, so the
    // second fails before any call is spent on it.
    let nested_a = docs.join("a");
    let nested_b = docs.join("b");
    std::fs::create_dir_all(&nested_a).unwrap();
    std::fs::create_dir_all(&nested_b).unwrap();
    std::fs::write(nested_a.join("x.md"), "壱。").unwrap();
    std::fs::write(nested_b.join("x.md"), "弐。").unwrap();
    let (url, requests) = stub_chat_server(vec![
        json!({"associations": [
            {"subject": "壱", "label": "rel", "object": "壱。", "weight": 1.0, "paragraph": 0}
        ]})
        .to_string(),
    ]);
    let collide_out = batch_dir("extract-runbook-collide-out");
    let (code, stdout, stderr) = run_extract(
        &collide_out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "ops",
            "--source-id",
            "session:claude:abc",
            nested_a.join("x.md").to_str().unwrap(),
            nested_b.join("x.md").to_str().unwrap(),
        ],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("collides"), "{stderr}");
    requests.join().unwrap();

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&collide_out);
}

/// TAGURU_EXTRACT_COVERAGE resolves like its boolean siblings: `1`
/// turns the report on, `0` keeps it off (and keeps every report line
/// free of an "uncovered" note — the count note must not render at
/// zero either), anything else is a hard usage error.
#[test]
fn extract_coverage_env_resolves_like_its_boolean_siblings() {
    let docs = batch_dir("extract-coverage-env-docs");
    let covered = docs.join("covered.md");
    std::fs::write(&covered, "テストランナーはnextestを使う。").unwrap();
    let gapped = docs.join("gapped.md");
    std::fs::write(
        &gapped,
        "バックアップはS3へ保存する。\n\n- 頻度: 日次\n- 保持期間: 30日",
    )
    .unwrap();
    let out = batch_dir("extract-coverage-env-out");

    // Directory expansion sorts by name: covered.md answers first.
    let covered_reply = json!({"associations": [
        {"subject": "テストランナー", "label": "採用", "object": "nextest", "weight": 1.0, "paragraph": 0}
    ]})
    .to_string();
    let gapped_reply = json!({"associations": [
        {"subject": "バックアップ", "label": "保存先", "object": "S3", "weight": 1.0, "paragraph": 0},
        {"subject": "バックアップ", "label": "保持期間", "object": "30日", "weight": 1.0, "paragraph": 1}
    ]})
    .to_string();
    let (url, requests) = stub_chat_server(vec![covered_reply, gapped_reply]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_COVERAGE", "1"),
        ],
        &["--context", "ops", docs.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("1 sentence(s) uncovered (coverage)"),
        "{stdout}"
    );
    // The fully-covered document's own report line carries NO
    // uncovered note — a zero count must not render as ", 0
    // sentence(s) uncovered".
    let covered_line = stdout
        .lines()
        .find(|line| line.contains("covered.md:"))
        .expect("covered.md earns a report line");
    assert!(!covered_line.contains("uncovered"), "{covered_line}");
    requests.join().unwrap();

    // `0` (both documents now unchanged): off means no uncovered
    // report anywhere, and no usage error.
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_COVERAGE", "0"),
        ],
        &["--context", "ops", docs.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("unchanged, skipped"), "{stdout}");
    assert!(!stderr.contains("uncovered"), "{stderr}");

    // Anything else is a usage error, not a silent default.
    let (code, _, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_COVERAGE", "banana"),
        ],
        &["--context", "ops", docs.to_str().unwrap()],
    );
    assert_eq!(code, 2, "{stderr}");
    assert!(
        stderr.contains("TAGURU_EXTRACT_COVERAGE takes 1/true or 0/false"),
        "{stderr}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// ADR 0015 (#496 S3): `--vocabulary` loads an exported batch stream,
/// offers its concept names and labels in the system prompt, admits a
/// context spelling through the occurrence check, and re-extracts when
/// the digest changes. A bad path is a hard startup error.
#[test]
fn extract_vocabulary_steers_spellings_and_is_a_computation_input() {
    let docs = batch_dir("extract-vocab-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "CI のテストランナーは cargo-nextest。").unwrap();
    let vocab = docs.join("export.jsonl");
    std::fs::write(
        &vocab,
        concat!(
            r#"{"taguru_batch":1,"context":"ops","source":"s0"}"#,
            "\n",
            r#"{"subject":"CI","label":"テストランナー","object":"nextest","weight":1.0}"#,
            "\n",
        ),
    )
    .unwrap();
    let out = batch_dir("extract-vocab-out");

    // The model follows the steering: the CONTEXT spelling `nextest`
    // for an entity the document spells `cargo-nextest` — kept only
    // because the vocabulary allowlists it past the occurrence check.
    let reply = json!({"associations": [
        {"subject": "CI", "label": "テストランナー", "object": "nextest"}
    ]})
    .to_string();
    let (url, requests) = stub_chat_server(vec![reply.clone()]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) = run_extract(
        &out,
        &provider,
        &[
            "--context",
            "c",
            "--vocabulary",
            vocab.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("1 association(s)"), "{stdout}");
    assert!(!stdout.contains("removed"), "{stdout}");

    let requests = requests.join().unwrap();
    let body_start = requests[0].find('{').unwrap();
    let body: serde_json::Value = serde_json::from_str(&requests[0][body_start..]).unwrap();
    let system = body["messages"][0]["content"].as_str().unwrap();
    let block = system
        .split("Names already in use in the target context")
        .nth(1)
        .unwrap_or_else(|| panic!("no context-names block: {system}"));
    assert!(block.contains("nextest"), "{block}");
    // The exported labels seed the run vocabulary from document one.
    assert!(
        system.contains("Relation labels already in use"),
        "{system}"
    );
    assert!(system.contains("テストランナー"), "{system}");

    // A changed vocabulary is a computation input: same document, new
    // name set → re-extract, not an "unchanged" skip.
    std::fs::write(
        &vocab,
        concat!(
            r#"{"taguru_batch":1,"context":"ops","source":"s0"}"#,
            "\n",
            r#"{"subject":"CI","label":"テストランナー","object":"cargo-nextest","weight":1.0}"#,
            "\n",
        ),
    )
    .unwrap();
    let (url, requests) = stub_chat_server(vec![reply]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) = run_extract(
        &out,
        &provider,
        &[
            "--context",
            "c",
            "--vocabulary",
            vocab.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(!stdout.contains("unchanged, skipped"), "{stdout}");
    assert_eq!(requests.join().unwrap().len(), 1);

    // A path that loads nothing is a hard startup error.
    let (code, _, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", "http://127.0.0.1:9"),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--vocabulary",
            docs.join("missing.jsonl").to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("--vocabulary"), "{stderr}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}
/// #758: document A settles on a concept spelling; document C offers
/// that same spelling as an alias of a different name. Import would
/// refuse the rewire (409, the stream stopping with A already
/// applied), so extract removes the alias mechanically with accounting
/// and the directory imports whole. A rerun with a new document D
/// claims A's names from A's skipped batch the same way.
#[test]
fn extract_prunes_an_alias_that_would_rewire_an_earlier_documents_concept() {
    let docs = batch_dir("extract-claimed-docs");
    let a = docs.join("a.md");
    let c = docs.join("c.md");
    std::fs::write(&a, "東雲電機株式会社(架空)は新潟にある。").unwrap();
    std::fs::write(&c, "東雲電機株式会社の製品 SN-SEN70。").unwrap();
    let out = batch_dir("extract-claimed-out");

    let reply_a = json!({"associations": [
        {"subject": "東雲電機株式会社(架空)", "label": "所在地", "object": "新潟"}
    ]})
    .to_string();
    let reply_c = json!({
        "associations": [
            {"subject": "東雲電機株式会社", "label": "製品", "object": "SN-SEN70"}
        ],
        "aliases": [
            {"alias": "東雲電機株式会社(架空)", "canonical": "東雲電機株式会社", "kind": "concept"}
        ]
    })
    .to_string();
    let (url, requests) = stub_chat_server(vec![reply_a, reply_c.clone()]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) = run_extract(
        &out,
        &provider,
        &[
            "--context",
            "spec",
            "--description",
            "product spec sheets",
            a.to_str().unwrap(),
            c.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    // No corrective turn was spent: one request per document.
    assert_eq!(requests.join().unwrap().len(), 2);
    assert!(
        stdout.contains("1 item(s) removed (mechanical validation)"),
        "{stdout}"
    );
    let expected = format!(
        "taguru: extract: {}: removed: aliases[0]: alias \"東雲電機株式会社(架空)\" already \
         names a concept an earlier document or the target context settled on; an alias \
         cannot rewire it (import would refuse the batch)",
        c.display()
    );
    assert!(stderr.contains(&expected), "{stderr}");
    let batches: Vec<String> = std::fs::read_dir(&out)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .map(|path| std::fs::read_to_string(path).unwrap())
        .collect();
    assert_eq!(batches.len(), 2, "{batches:?}");
    assert!(
        batches.iter().all(|batch| !batch.contains("\"alias\"")),
        "the rewiring alias must not be written: {batches:?}"
    );

    // The whole directory imports — the 409 that motivated this never
    // fires.
    let data_dir = common::scratch_dir("http-extract-claimed");
    let (code, stdout, stderr) = run_import(&data_dir, &[out.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(!stderr.contains("already resolves"), "{stderr}");

    // Rerun: A and C are unchanged (skipped), D is new and offers A's
    // concept as an alias again — claimed from A's batch file, not
    // from a fresh extraction.
    let d = docs.join("d.md");
    std::fs::write(&d, "東雲電機株式会社の拠点は東京。").unwrap();
    let reply_d = json!({
        "associations": [
            {"subject": "東雲電機株式会社", "label": "拠点", "object": "東京"}
        ],
        "aliases": [
            {"alias": "東雲電機株式会社(架空)", "canonical": "東雲電機株式会社", "kind": "concept"}
        ]
    })
    .to_string();
    let (url, requests) = stub_chat_server(vec![reply_d]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) = run_extract(
        &out,
        &provider,
        &[
            "--context",
            "spec",
            "--description",
            "product spec sheets",
            a.to_str().unwrap(),
            c.to_str().unwrap(),
            d.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(requests.join().unwrap().len(), 1);
    assert_eq!(stdout.matches("unchanged, skipped").count(), 2, "{stdout}");
    let expected = format!(
        "taguru: extract: {}: removed: aliases[0]: alias \"東雲電機株式会社(架空)\" already",
        d.display()
    );
    assert!(stderr.contains(&expected), "{stderr}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// TAGURU_EXTRACT_CANDIDATES engages the block without the flag, and a
/// bad value is a hard usage error — the --lossy env conventions.
#[test]
fn extract_candidates_env_var_enables_the_block_and_rejects_bad_values() {
    let docs = batch_dir("extract-candidates-env-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "CI のテストランナーは cargo-nextest。").unwrap();
    let out = batch_dir("extract-candidates-env-out");

    let (url, requests) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_CANDIDATES", "true"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let requests = requests.join().unwrap();
    assert!(
        requests[0].contains("Names appearing in this document"),
        "{}",
        requests[0]
    );

    let (code, _, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", "http://127.0.0.1:9"),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_CANDIDATES", "nope"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 2, "{stderr}");
    assert!(
        stderr.contains("TAGURU_EXTRACT_CANDIDATES takes 1/true or 0/false"),
        "{stderr}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}
/// A non-retryable 4xx (anything but 429) fails on the first attempt
/// without spending the retry budget.
#[test]
fn a_non_retryable_four_hundred_fails_without_spending_the_retry_budget() {
    let docs = batch_dir("extract-fail400-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-fail400-out");

    let (url, captured) =
        stub_chat_server_concurrent(|_index, _attempt| chat_error(400, "Bad Request", "", "nope"));
    let started = std::time::Instant::now();
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    let elapsed = started.elapsed();
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("400"), "{stderr}");
    assert!(
        elapsed < std::time::Duration::from_secs(5),
        "a non-retryable failure must not spend the retry budget, took {elapsed:?}"
    );
    assert_eq!(
        captured.lock().unwrap().len(),
        1,
        "a non-retryable failure must not be retried"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn the_extract_timeout_knob_bounds_a_stalled_provider() {
    let docs = batch_dir("extract-stall-docs");
    let doc = docs.join("slow.md");
    std::fs::write(&doc, "content").unwrap();

    // A provider that accepts and never answers — the local-model
    // failure mode (a thinking model grinding away) as seen from the
    // client. All four attempts' connections are held open, unanswered,
    // well past the client's worst-case retry budget below.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let mut held = Vec::new();
        for _ in 0..4 {
            if let Ok((stream, _)) = listener.accept() {
                held.push(stream);
            }
        }
        std::thread::sleep(std::time::Duration::from_secs(30));
    });

    let out = batch_dir("extract-stall-out");
    let started = std::time::Instant::now();
    let (code, _, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_TIMEOUT_SECS", "1"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 1, "{stderr}");
    // ureq 3 renders its timeout error as "timeout: <phase>".
    assert!(stderr.contains("timeout"), "{stderr}");
    // Four 1-second attempts plus the jittered backoff between them
    // (up to 1 + 2 + 4 = 7s worst case) — nowhere near the 300-second
    // default this knob overrides.
    assert!(
        started.elapsed() < std::time::Duration::from_secs(25),
        "took {:?}",
        started.elapsed()
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn extract_persists_the_manifest_after_each_document_not_only_at_the_end() {
    let docs = batch_dir("extract-manifest-durability-docs");
    let fast = docs.join("fast.md");
    let slow = docs.join("slow.md");
    std::fs::write(&fast, "青嶺酒造は1907年に創業した。").unwrap();
    std::fs::write(&slow, "高瀬は青嶺酒造の杜氏。").unwrap();
    let fast_src = fast.to_str().unwrap().to_string();

    // fast.md's one request gets a real answer; every later connection
    // (slow.md's) is accepted and then never answered — standing in
    // for the interruption (Ctrl+C, a CI timeout's SIGKILL, a panic on
    // a later document) that this test triggers itself by killing the
    // child while it hangs there. fast.md's progress must already be
    // on disk by that point, not deferred to a final save this kill
    // prevents from ever running.
    let reply = json!({
        "associations": [
            {"subject": "青嶺酒造", "label": "創業年", "object": "1907年", "weight": 1.0}
        ]
    })
    .to_string();
    let response = chat_ok(&reply);
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        use std::io::Write;
        let mut held = Vec::new();
        for (index, stream) in listener.incoming().enumerate() {
            let Ok(mut stream) = stream else { continue };
            if index == 0 {
                let _ = read_http_request(&mut stream);
                let _ = stream.write_all(response.as_bytes());
            } else {
                held.push(stream);
            }
        }
    });

    let out = batch_dir("extract-manifest-durability-out");
    let manifest_path = out.join(".extract-manifest.json");
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    scrub_extract_env(&mut command)
        .arg("extract")
        .env("TAGURU_EXTRACT_URL", &url)
        .env("TAGURU_EXTRACT_MODEL", "stub-model")
        .args(["--out", out.to_str().unwrap(), "--context", "c"])
        .arg(&fast)
        .arg(&slow)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut child = command.spawn().expect("extract must spawn");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut saved = String::new();
    while std::time::Instant::now() < deadline {
        if let Ok(text) = std::fs::read_to_string(&manifest_path)
            && text.contains(&fast_src)
        {
            saved = text;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let _ = child.kill();
    let _ = child.wait();
    assert!(
        saved.contains(&fast_src),
        "manifest never recorded the completed document before the run was killed: {saved:?}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn extract_fact_budget_flag_is_folded_into_the_system_prompt() {
    let docs = batch_dir("extract-factbudget-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-factbudget-out");

    let (url, requests) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--fact-budget",
            "3",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(
        requests[0].contains("Keep this answer to at most 3 association(s) total"),
        "{}",
        requests[0]
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn extract_schema_flag_folds_the_type_and_relation_block_into_the_system_prompt() {
    let docs = batch_dir("extract-schema-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let schema_path = docs.join("sake.schema.json");
    std::fs::write(
        &schema_path,
        json!({
            "schema": 1,
            "mode": "warn",
            "closed_labels": false,
            "types": {
                "Brewery": {"is_a": []},
                "Person": {"is_a": []}
            },
            "relations": {
                "杜氏": {"domain": ["Brewery"], "range": ["Person"]}
            }
        })
        .to_string(),
    )
    .unwrap();
    let out = batch_dir("extract-schema-out");

    let (url, requests) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--schema",
            schema_path.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].contains("Brewery"), "{}", requests[0]);
    assert!(
        requests[0].contains("\\u675c\\u6c0f: Brewery \\u2192 Person")
            || requests[0].contains("杜氏: Brewery → Person"),
        "{}",
        requests[0]
    );
    assert!(requests[0].contains("schema:type"), "{}", requests[0]);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn extract_schema_flag_fails_the_run_at_startup_when_the_file_does_not_parse() {
    let docs = batch_dir("extract-schema-invalid-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let schema_path = docs.join("broken.schema.json");
    std::fs::write(&schema_path, "not json").unwrap();
    let out = batch_dir("extract-schema-invalid-out");

    // --dry-run calls no provider, but --schema is validated at startup
    // regardless — an operator-named schema file that fails to parse
    // must never be silently treated as "no schema".
    let (code, stdout, stderr) = run_extract(
        &out,
        &[],
        &[
            "--dry-run",
            "--context",
            "c",
            "--schema",
            schema_path.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("broken.schema.json"), "{stderr}");
    assert!(!stdout.contains("would extract"), "{stdout}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn extract_schema_flag_fails_the_run_at_startup_when_the_file_does_not_exist() {
    let docs = batch_dir("extract-schema-missing-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let missing_path = docs.join("missing.schema.json");
    let out = batch_dir("extract-schema-missing-out");

    let (code, stdout, stderr) = run_extract(
        &out,
        &[],
        &[
            "--dry-run",
            "--context",
            "c",
            "--schema",
            missing_path.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("missing.schema.json"), "{stderr}");
    assert!(!stdout.contains("would extract"), "{stdout}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn extract_schema_flag_fails_the_run_at_startup_when_schema_install_refuses_the_document() {
    let docs = batch_dir("extract-schema-refused-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let schema_path = docs.join("future.schema.json");
    // Valid JSON — the parse itself succeeds — but an unknown `schema`
    // version, which only schema::install's own check refuses, distinct
    // from the JSON-parse-failure case above.
    std::fs::write(
        &schema_path,
        json!({
            "schema": 999,
            "mode": "off",
            "closed_labels": false,
            "types": {},
            "relations": {}
        })
        .to_string(),
    )
    .unwrap();
    let out = batch_dir("extract-schema-refused-out");

    let (code, stdout, stderr) = run_extract(
        &out,
        &[],
        &[
            "--dry-run",
            "--context",
            "c",
            "--schema",
            schema_path.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("future.schema.json"), "{stderr}");
    assert!(!stdout.contains("would extract"), "{stdout}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn extract_schema_env_var_fails_the_run_at_startup_the_same_as_the_flag() {
    let docs = batch_dir("extract-schema-env-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let schema_path = docs.join("broken.schema.json");
    std::fs::write(&schema_path, "not json").unwrap();
    let out = batch_dir("extract-schema-env-out");

    let (code, stdout, stderr) = run_extract(
        &out,
        &[("TAGURU_EXTRACT_SCHEMA", schema_path.to_str().unwrap())],
        &["--dry-run", "--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("broken.schema.json"), "{stderr}");
    assert!(!stdout.contains("would extract"), "{stdout}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// `TAGURU_EXTRACT_MAX_ATTEMPTS` raised past the default lets a chunk
/// survive more than one corrective turn — two bad answers followed by
/// a good one, which the default policy (2 total attempts) would never
/// reach.
#[test]
fn extract_max_attempts_env_var_extends_corrective_retries_past_the_default() {
    let docs = batch_dir("extract-maxattempts-extend-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-maxattempts-extend-out");

    let (url, requests) = stub_chat_server(vec![
        "still not json".to_string(),
        "nope, still not".to_string(),
        json!({"associations": []}).to_string(),
    ]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_MAX_ATTEMPTS", "3"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("1 written"), "{stdout}");
    assert_eq!(requests.join().unwrap().len(), 3);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// `TAGURU_EXTRACT_MAX_ATTEMPTS=1` means one attempt total — no
/// corrective turn at all, unlike the default of 2.
#[test]
fn extract_max_attempts_of_one_skips_the_corrective_turn() {
    let docs = batch_dir("extract-maxattempts-one-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-maxattempts-one-out");

    let (url, requests) = stub_chat_server(vec!["not json at all".to_string()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_MAX_ATTEMPTS", "1"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("chunk 1/1"), "{stderr}");
    assert!(
        stderr.contains("the model would not produce the JSON object"),
        "{stderr}"
    );
    assert_eq!(
        requests.join().unwrap().len(),
        1,
        "max_attempts=1 must not send a corrective turn"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn extract_rejects_a_max_attempts_env_var_outside_its_range() {
    let docs = batch_dir("extract-maxattempts-range-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "content").unwrap();
    let out = batch_dir("extract-maxattempts-range-out");

    for bad in ["0", "11", "nope"] {
        let provider = [
            ("TAGURU_EXTRACT_URL", "http://127.0.0.1:9"),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_MAX_ATTEMPTS", bad),
        ];
        let (code, _, stderr) =
            run_extract(&out, &provider, &["--context", "c", doc.to_str().unwrap()]);
        assert_eq!(code, 2, "{bad}: {stderr}");
        assert!(
            stderr.contains("TAGURU_EXTRACT_MAX_ATTEMPTS needs an integer between 1 and 10"),
            "{bad}: {stderr}"
        );
    }

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// A byte cap truncates the corrective turn's replay of the model's own
/// prior bad answer instead of resending it in full.
#[test]
fn extract_corrective_context_bytes_caps_the_replayed_bad_answer() {
    let docs = batch_dir("extract-correctivecap-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-correctivecap-out");

    let bad_answer = "not json at all, definitely not a JSON object";
    let (url, requests) = stub_chat_server(vec![
        bad_answer.to_string(),
        json!({"associations": []}).to_string(),
    ]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_CORRECTIVE_CONTEXT_BYTES", "10"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests[1].contains("[truncated to 10 bytes]"),
        "{}",
        requests[1]
    );
    assert!(
        !requests[1].contains(bad_answer),
        "the full bad answer must not be replayed under a cap: {}",
        requests[1]
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// A zero byte cap omits the replay entirely, behind a placeholder —
/// distinct from an unset cap (full replay, the default).
#[test]
fn extract_corrective_context_bytes_of_zero_omits_the_bad_answer() {
    let docs = batch_dir("extract-correctivezero-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-correctivezero-out");

    let bad_answer = "not json at all, definitely not a JSON object";
    let (url, requests) = stub_chat_server(vec![
        bad_answer.to_string(),
        json!({"associations": []}).to_string(),
    ]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_CORRECTIVE_CONTEXT_BYTES", "0"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests[1].contains("[omitted: not the requested JSON object]"),
        "{}",
        requests[1]
    );
    assert!(!requests[1].contains(bad_answer), "{}", requests[1]);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn extract_rejects_a_corrective_context_bytes_env_var_that_is_not_a_number() {
    let docs = batch_dir("extract-correctivebad-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "content").unwrap();
    let out = batch_dir("extract-correctivebad-out");

    for bad in ["nope", "-5"] {
        let provider = [
            ("TAGURU_EXTRACT_URL", "http://127.0.0.1:9"),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_CORRECTIVE_CONTEXT_BYTES", bad),
        ];
        let (code, _, stderr) =
            run_extract(&out, &provider, &["--context", "c", doc.to_str().unwrap()]);
        assert_eq!(code, 2, "{bad}: {stderr}");
        assert!(
            stderr.contains("TAGURU_EXTRACT_CORRECTIVE_CONTEXT_BYTES needs an integer"),
            "{bad}: {stderr}"
        );
    }

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// A `finish_reason: "length"` on the bad answer swaps the corrective
/// ask from "try again" to "try again shorter" and names the run's
/// `--fact-budget` — the fix for Issue #178's stall (a huge truncated
/// answer, replayed in full, re-asked for the very length it just
/// proved it couldn't fit in).
#[test]
fn extract_a_length_limited_bad_answer_asks_for_shorter_and_names_the_fact_budget() {
    let docs = batch_dir("extract-lengthlimited-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-lengthlimited-out");

    let (url, captured) = stub_chat_server_concurrent(|_index, attempt| {
        if attempt == 0 {
            chat_ok_with_finish_reason("not json, and huge", "length")
        } else {
            chat_ok(&json!({"associations": []}).to_string())
        }
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--fact-budget",
            "4",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let requests = captured.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let corrective = &requests[1];
    assert!(corrective.contains("SHORTER"), "{corrective}");
    assert!(
        corrective.contains("cut off at the output limit"),
        "{corrective}"
    );
    assert!(
        corrective.contains("Keep it to at most 4 association(s) total."),
        "{corrective}"
    );
    assert!(
        !corrective.contains("Answer again with only the JSON object."),
        "a length-limited correction must not repeat the plain ask verbatim: {corrective}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// With no new control engaged, the request body carries exactly the
/// pre-ladder keys and the run resolves no structured-output rung —
/// the wire half of "defaults byte-for-byte unchanged" (the byte half
/// is extract.rs's own `request_options_default_adds_no_keys_to_the_body`).
#[test]
fn extract_default_request_body_carries_exactly_the_base_keys() {
    let docs = batch_dir("extract-defaultbody-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-defaultbody-out");

    let (url, requests) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        !stderr.contains("structured output:"),
        "no rung resolution may run at defaults: {stderr}"
    );

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        top_level_keys(&requests[0]),
        ["messages", "model", "temperature"]
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// A pinned `--structured-output json-schema` sends the canonical
/// schema as `response_format` on the one extraction request — no
/// probe call before it, `strict` requested, the binding name from the
/// schema's own title.
#[test]
fn structured_output_json_schema_sends_the_canonical_schema_without_probing() {
    let docs = batch_dir("extract-jsonschema-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-jsonschema-out");

    let (url, requests) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--structured-output",
            "json-schema",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stderr.contains("structured output: json_schema (pinned)"),
        "{stderr}"
    );

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 1, "a pinned mode must not probe");
    let body = json_body_of(&requests[0]);
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(
        body["response_format"]["json_schema"]["name"],
        "ModelOutput"
    );
    assert_eq!(body["response_format"]["json_schema"]["strict"], true);
    let schema = &body["response_format"]["json_schema"]["schema"];
    assert_eq!(schema["title"], "ModelOutput");
    assert_eq!(
        schema["required"],
        json!(["associations", "aliases"]),
        "{schema}"
    );
    assert!(
        body.get("max_tokens").is_none(),
        "no budget was configured: {body}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn structured_output_json_object_sends_json_mode_without_probing() {
    let docs = batch_dir("extract-jsonobject-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-jsonobject-out");

    let (url, requests) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--structured-output",
            "json-object",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stderr.contains("structured output: json_object (pinned)"),
        "{stderr}"
    );

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        json_body_of(&requests[0])["response_format"],
        json!({"type": "json_object"})
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// `--max-output-tokens` alone engages the budget without any
/// `response_format`: the two controls are orthogonal, and the mode's
/// default stays off even when the budget is set.
#[test]
fn structured_output_off_with_a_budget_sends_max_tokens_and_no_response_format() {
    let docs = batch_dir("extract-budgetonly-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-budgetonly-out");

    let (url, requests) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--max-output-tokens",
            "512",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(!stderr.contains("structured output:"), "{stderr}");

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 1);
    let body = json_body_of(&requests[0]);
    assert_eq!(body["max_tokens"], 512);
    assert!(body.get("response_format").is_none(), "{body}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn extract_rejects_a_bad_structured_output_value() {
    let docs = batch_dir("extract-badmode-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "content").unwrap();
    let out = batch_dir("extract-badmode-out");

    // The flag and the env var reject the same closed vocabulary the
    // same way — including near-misses in the wrong spelling.
    for bad in ["json_schema", "schema", "on", ""] {
        let (code, _, stderr) = run_extract(
            &out,
            &[
                ("TAGURU_EXTRACT_URL", "http://127.0.0.1:9"),
                ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ],
            &[
                "--context",
                "c",
                "--structured-output",
                bad,
                doc.to_str().unwrap(),
            ],
        );
        assert_eq!(code, 2, "{bad}: {stderr}");
        assert!(
            stderr.contains("--structured-output takes auto, json-schema, json-object, or off"),
            "{bad}: {stderr}"
        );
    }
    let (code, _, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", "http://127.0.0.1:9"),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_STRUCTURED_OUTPUT", "json_schema"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 2, "{stderr}");
    assert!(
        stderr.contains("TAGURU_EXTRACT_STRUCTURED_OUTPUT takes auto, json-schema"),
        "{stderr}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn extract_rejects_a_bad_max_output_tokens_value() {
    let docs = batch_dir("extract-badbudget-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "content").unwrap();
    let out = batch_dir("extract-badbudget-out");

    for bad in ["0", "-1", "nope"] {
        let (code, _, stderr) = run_extract(
            &out,
            &[
                ("TAGURU_EXTRACT_URL", "http://127.0.0.1:9"),
                ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ],
            &[
                "--context",
                "c",
                "--max-output-tokens",
                bad,
                doc.to_str().unwrap(),
            ],
        );
        assert_eq!(code, 2, "{bad}: {stderr}");
        assert!(
            stderr.contains("--max-output-tokens needs an integer of at least 1"),
            "{bad}: {stderr}"
        );
    }
    let (code, _, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", "http://127.0.0.1:9"),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_MAX_OUTPUT_TOKENS", "0"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 2, "{stderr}");
    assert!(
        stderr.contains("TAGURU_EXTRACT_MAX_OUTPUT_TOKENS needs an integer of at least 1"),
        "{stderr}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// With a budget engaged, a `length`-terminated answer escalates
/// exactly once: the next request raises `max_tokens` to the factored
/// cap (ADR 0019: 2× by default) and resends the base ask NEUTRALLY —
/// no corrective turn, no replay of the truncated answer, none of the
/// legacy SHORTER wording (which asks for less than the budget could
/// now hold).
#[test]
fn length_limited_escalates_once_with_a_neutral_resend_when_a_budget_is_set() {
    let docs = batch_dir("extract-escalate-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-escalate-out");

    let (url, captured) = stub_chat_server_concurrent(|_index, attempt| {
        if attempt == 0 {
            chat_ok_with_finish_reason("truncated garbage", "length")
        } else {
            chat_ok(&json!({"associations": []}).to_string())
        }
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--max-output-tokens",
            "512",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("1 written"), "{stdout}");

    let requests = captured.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(json_body_of(&requests[0])["max_tokens"], 512);
    let escalated = json_body_of(&requests[1]);
    assert_eq!(
        escalated["max_tokens"], 1024,
        "escalation must raise the budget (2× by default), never re-ask under it: {escalated}"
    );
    assert_eq!(
        escalated["messages"].as_array().unwrap().len(),
        2,
        "escalation is a neutral resend of the base ask: {escalated}"
    );
    assert!(
        !requests[1].contains("truncated garbage"),
        "{}",
        requests[1]
    );
    assert!(!requests[1].contains("SHORTER"), "{}", requests[1]);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// ADR 0020 (#762): `--chunk-bytes` / TAGURU_EXTRACT_CHUNK_BYTES set the
/// chunk cap — visible in --dry-run's chunk count — the flag winning
/// over the variable, and both refusing anything under the split floor.
#[test]
fn chunk_bytes_flag_and_env_set_the_chunk_cap() {
    let docs = batch_dir("extract-chunkbytes-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, format!("{}\n\n{}", "a".repeat(600), "b".repeat(600))).unwrap();
    let out = batch_dir("extract-chunkbytes-out");
    let dry = |env: &[(&str, &str)], extra: &[&str]| {
        let mut args = vec!["--dry-run", "--context", "c"];
        args.extend_from_slice(extra);
        args.push(doc.to_str().unwrap());
        run_extract(&out, env, &args)
    };

    let (code, stdout, stderr) = dry(&[], &[]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(chunk_count_from_dry_run(&stdout), 1, "{stdout}");

    let (code, stdout, _) = dry(&[], &["--chunk-bytes", "700"]);
    assert_eq!(code, 0, "{stdout}");
    assert_eq!(chunk_count_from_dry_run(&stdout), 2, "{stdout}");

    let (code, stdout, _) = dry(&[("TAGURU_EXTRACT_CHUNK_BYTES", "700")], &[]);
    assert_eq!(code, 0, "{stdout}");
    assert_eq!(chunk_count_from_dry_run(&stdout), 2, "{stdout}");

    // The flag wins over the variable.
    let (code, stdout, _) = dry(
        &[("TAGURU_EXTRACT_CHUNK_BYTES", "700")],
        &["--chunk-bytes", "4096"],
    );
    assert_eq!(code, 0, "{stdout}");
    assert_eq!(chunk_count_from_dry_run(&stdout), 1, "{stdout}");

    let (code, _, stderr) = dry(&[], &["--chunk-bytes", "511"]);
    assert_eq!(code, 2, "{stderr}");
    assert!(
        stderr.contains("--chunk-bytes needs an integer of at least 512"),
        "{stderr}"
    );
    for bad in ["big", "511", "0"] {
        let (code, _, stderr) = dry(&[("TAGURU_EXTRACT_CHUNK_BYTES", bad)], &[]);
        assert_eq!(code, 2, "{bad}: {stderr}");
        assert!(
            stderr.contains("TAGURU_EXTRACT_CHUNK_BYTES needs an integer of at least 512"),
            "{bad}: {stderr}"
        );
    }
    // The floor itself is accepted from the variable too — and at 512
    // each 600-byte paragraph is itself over the cap, so it splits.
    let (code, stdout, _) = dry(&[("TAGURU_EXTRACT_CHUNK_BYTES", "512")], &[]);
    assert_eq!(code, 0, "{stdout}");
    assert_eq!(chunk_count_from_dry_run(&stdout), 4, "{stdout}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// ADR 0020 (#762): under the ladder a timeout descends to the split
/// rung — one stalled ask, then one answer per half — instead of four
/// same-size attempts and a failed source; at the split floor it fails
/// after one attempt with the timeout named.
#[test]
fn a_timeout_under_the_ladder_splits_instead_of_retrying_at_the_same_size() {
    let docs = batch_dir("extract-timeoutsplit-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, format!("{}\n\n{}", "a".repeat(600), "b".repeat(600))).unwrap();
    let out = batch_dir("extract-timeoutsplit-out");

    let (url, captured) = stub_chat_server_concurrent(|_index, attempt| {
        if attempt == 0 {
            // Outlive the client's 1 s timeout, then answer into a
            // connection the client has already abandoned.
            std::thread::sleep(std::time::Duration::from_secs(3));
        }
        chat_ok(&json!({"associations": []}).to_string())
    });
    let started = std::time::Instant::now();
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_TIMEOUT_SECS", "1"),
        ],
        &[
            "--context",
            "c",
            "--max-output-tokens",
            "512",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("1 written"), "{stdout}");
    assert!(!stderr.contains("after 4 attempts"), "{stderr}");
    assert_eq!(
        captured.lock().unwrap().len(),
        3,
        "the stalled ask, then one per half"
    );
    assert!(
        started.elapsed() < std::time::Duration::from_secs(20),
        "took {:?}",
        started.elapsed()
    );
    // ADR 0029: a timeout split's move record says it was a timeout,
    // not an output cap.
    let records = read_attempts_log(&out);
    let split = records
        .iter()
        .find(|r| r["kind"] == "move" && r["move"] == "split")
        .unwrap_or_else(|| panic!("{records:?}"));
    assert!(
        split["reason"].as_str().unwrap().contains("timed out"),
        "{split}"
    );
    assert_eq!(split["sub_pieces"], 2);
    // The timed-out attempt is on record with zero transport retries —
    // fail-fast under the ladder returned the first timeout.
    let timeout = records
        .iter()
        .find(|r| r["kind"] == "attempt" && r["state"] == "timeout")
        .unwrap_or_else(|| panic!("{records:?}"));
    assert_eq!(timeout["transport_retries"], 0);

    // The floor: one paragraph that cannot split, always stalling.
    let floor = docs.join("floor.md");
    std::fs::write(&floor, "content").unwrap();
    let (url, captured) = stub_chat_server_concurrent(|_index, _attempt| {
        std::thread::sleep(std::time::Duration::from_secs(3));
        chat_ok(&json!({"associations": []}).to_string())
    });
    let started = std::time::Instant::now();
    let (code, _, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_TIMEOUT_SECS", "1"),
        ],
        &[
            "--context",
            "c",
            "--max-output-tokens",
            "512",
            floor.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 1, "{stderr}");
    assert!(
        stderr.contains("timed out") && stderr.contains("cannot split further"),
        "{stderr}"
    );
    assert!(!stderr.contains("after 4 attempts"), "{stderr}");
    assert_eq!(captured.lock().unwrap().len(), 1, "one attempt, no retries");
    assert!(
        started.elapsed() < std::time::Duration::from_secs(10),
        "took {:?}",
        started.elapsed()
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// ADR 0019 (#761): TAGURU_EXTRACT_ESCALATION_FACTOR sets the escalated
/// resend's cap as a multiple of the budget; 0 restores the uncapped
/// resend; anything else is a usage error whether or not a budget is
/// configured.
#[test]
fn escalation_factor_env_caps_the_resend_and_zero_uncaps_it() {
    let docs = batch_dir("extract-escfactor-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-escfactor-out");

    for (factor, expected) in [("3", Some(1536)), ("0", None)] {
        let (url, captured) = stub_chat_server_concurrent(|_index, attempt| {
            if attempt == 0 {
                chat_ok_with_finish_reason("truncated garbage", "length")
            } else {
                chat_ok(&json!({"associations": []}).to_string())
            }
        });
        let (code, stdout, stderr) = run_extract(
            &out,
            &[
                ("TAGURU_EXTRACT_URL", url.as_str()),
                ("TAGURU_EXTRACT_MODEL", "stub-model"),
                ("TAGURU_EXTRACT_ESCALATION_FACTOR", factor),
            ],
            &[
                "--context",
                "c",
                "--force",
                "--max-output-tokens",
                "512",
                doc.to_str().unwrap(),
            ],
        );
        assert_eq!(
            code, 0,
            "factor {factor}: stdout: {stdout}\nstderr: {stderr}"
        );
        let requests = captured.lock().unwrap();
        assert_eq!(requests.len(), 2, "factor {factor}");
        assert_eq!(json_body_of(&requests[0])["max_tokens"], 512);
        let escalated = json_body_of(&requests[1]);
        match expected {
            Some(cap) => assert_eq!(escalated["max_tokens"], cap, "factor {factor}: {escalated}"),
            None => assert!(
                escalated.get("max_tokens").is_none(),
                "factor 0 is the uncapped resend: {escalated}"
            ),
        }
    }

    // A bad value is a usage error even without a budget.
    let (code, _, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", "http://127.0.0.1:9"),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_ESCALATION_FACTOR", "two"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 2, "{stderr}");
    assert!(
        stderr.contains("TAGURU_EXTRACT_ESCALATION_FACTOR needs an integer of at least 0"),
        "{stderr}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// The core regression from ADR 0001: `finish_reason: "length"` on an
/// answer whose prefix happens to parse is still truncation. The valid
/// prefix must never be imported — the piece regenerates at the
/// escalated budget and only THAT answer lands.
#[test]
fn a_length_terminated_answer_that_happens_to_parse_is_never_treated_as_success() {
    let docs = batch_dir("extract-validprefix-docs");
    let doc = docs.join("a.md");
    // "whole answer"/"half answer" attest the stub names under the
    // occurrence check without putting the literal "half_answer"
    // token into the passage the batch assertion greps for.
    std::fs::write(&doc, "whole answer half answer x small document").unwrap();
    let out = batch_dir("extract-validprefix-out");

    let prefix = json!({"associations":
        [{"subject": "half_answer", "label": "rel", "object": "x", "weight": 1.0}]});
    let complete = json!({"associations":
        [{"subject": "whole_answer", "label": "rel", "object": "x", "weight": 1.0}]});
    let (url, captured) = stub_chat_server_concurrent(move |_index, attempt| {
        if attempt == 0 {
            chat_ok_with_finish_reason(&prefix.to_string(), "length")
        } else {
            chat_ok(&complete.to_string())
        }
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--max-output-tokens",
            "512",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(captured.lock().unwrap().len(), 2);

    let batches = stray_batch_files(&out);
    assert_eq!(batches.len(), 1, "{batches:?}");
    let batch = std::fs::read_to_string(out.join(&batches[0])).unwrap();
    assert!(batch.contains("whole_answer"), "{batch}");
    assert!(
        !batch.contains("half_answer"),
        "a truncated answer's valid prefix must never import: {batch}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// Without a configured budget there is nothing to escalate: `length`
/// goes straight to the split rung, and each sub-piece runs its own
/// ladder from the top.
#[test]
fn length_limited_without_a_configured_budget_splits_instead_of_escalating() {
    let docs = batch_dir("extract-splitnobudget-docs");
    let doc = docs.join("a.md");
    // Two paragraphs, comfortably splittable at the halved cap.
    std::fs::write(&doc, format!("{}\n\n{}", "a".repeat(600), "b".repeat(600))).unwrap();
    let out = batch_dir("extract-splitnobudget-out");

    let (url, captured) = stub_chat_server_concurrent(|_index, attempt| {
        if attempt == 0 {
            chat_ok_with_finish_reason("truncated garbage", "length")
        } else {
            chat_ok(&json!({"associations": []}).to_string())
        }
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--structured-output",
            "json-object",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("1 written"), "{stdout}");

    let requests = captured.lock().unwrap();
    assert_eq!(
        requests.len(),
        3,
        "one length-limited ask, then one per split half"
    );
    for request in requests.iter() {
        let body = json_body_of(request);
        assert!(body.get("max_tokens").is_none(), "{body}");
        assert_eq!(body["response_format"], json!({"type": "json_object"}));
    }
    assert!(requests[1].contains("[0] a"), "{}", requests[1]);
    assert!(!requests[1].contains("[1] b"), "{}", requests[1]);
    assert!(requests[2].contains("[1] b"), "{}", requests[2]);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// Budget set, escalation exhausted: the piece splits, and each
/// sub-piece's ladder starts back at the CONFIGURED budget — the
/// halved input is expected to fit it, and an uncapped first ask would
/// give away the budget the operator set.
#[test]
fn length_limited_after_escalation_splits_the_piece_and_sub_pieces_restart_at_the_budget() {
    let docs = batch_dir("extract-escalatesplit-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, format!("{}\n\n{}", "a".repeat(600), "b".repeat(600))).unwrap();
    let out = batch_dir("extract-escalatesplit-out");

    let (url, captured) = stub_chat_server_concurrent(|_index, attempt| {
        if attempt <= 1 {
            chat_ok_with_finish_reason("truncated garbage", "length")
        } else {
            chat_ok(&json!({"associations": []}).to_string())
        }
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--max-output-tokens",
            "512",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let requests = captured.lock().unwrap();
    assert_eq!(
        requests.len(),
        4,
        "budgeted ask, escalated ask, then one per split half"
    );
    assert_eq!(json_body_of(&requests[0])["max_tokens"], 512);
    assert_eq!(json_body_of(&requests[1])["max_tokens"], 1024);
    assert_eq!(json_body_of(&requests[2])["max_tokens"], 512);
    assert_eq!(json_body_of(&requests[3])["max_tokens"], 512);

    // ADR 0029 (#791): the two ladder moves land in the attempts log
    // as records — the escalation with both budgets, then the split
    // with size, cap, and sub-piece count — id-joined to the piece.
    drop(requests);
    let records = read_attempts_log(&out);
    let moves: Vec<&Value> = records.iter().filter(|r| r["kind"] == "move").collect();
    assert_eq!(moves.len(), 2, "{moves:?}");
    let run_id = records[0]["run_id"].as_str().unwrap();
    assert_eq!(moves[0]["move"], "escalate");
    assert_eq!(moves[0]["run_id"], run_id);
    assert_eq!(moves[0]["chunk_index"], 0);
    assert_eq!(moves[0]["from_max_tokens"], 512);
    assert_eq!(moves[0]["to_max_tokens"], 1024);
    assert!(moves[0].get("from_rung").is_none());
    assert_eq!(moves[1]["move"], "split");
    assert_eq!(moves[1]["piece_id"], moves[0]["piece_id"]);
    assert!(
        moves[1]["reason"].as_str().unwrap().contains("output cap"),
        "{}",
        moves[1]
    );
    assert!(moves[1]["piece_bytes"].as_u64().unwrap() > 1000);
    assert!(moves[1]["split_cap"].as_u64().unwrap() >= 512);
    assert_eq!(moves[1]["sub_pieces"], 2);
    // The two length-limited attempts both name the same piece the
    // moves do; the sub-pieces' attempts name their own.
    let attempts: Vec<&Value> = records.iter().filter(|r| r["kind"] == "attempt").collect();
    assert_eq!(attempts.len(), 4);
    assert_eq!(attempts[0]["piece_id"], moves[0]["piece_id"]);
    assert_eq!(attempts[1]["state"], "length_limited");
    assert_ne!(attempts[2]["piece_id"], moves[0]["piece_id"]);
    // A clean HTTP conversation: zero transport retries everywhere.
    assert!(attempts.iter().all(|a| a["transport_retries"] == 0));

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// ADR 0029 (#791): the transport-layer retries ADR 0001 §10 folds
/// into one attempt are now counted on it — a 500 answered twice
/// before success is one `attempt` record with `transport_retries: 2`,
/// in the sidecar and the attempts log alike.
#[test]
fn transport_retries_are_counted_on_the_one_attempt_record() {
    let docs = batch_dir("extract-retrycount-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "alpha relates to beta").unwrap();
    let out = batch_dir("extract-retrycount-out");
    let diag_dir = batch_dir("extract-retrycount-diag");
    let diag = diag_dir.join("diag.jsonl");

    let good = chat_ok(
        &json!({"associations": [
            {"subject": "alpha", "label": "relates to", "object": "beta", "weight": 1.0}
        ]})
        .to_string(),
    );
    let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let calls_in = std::sync::Arc::clone(&calls);
    let (url, _captured) = stub_chat_server_concurrent(move |_index, _attempt| {
        // Connection-order fault injection: the first two tries get a
        // 500, the third the real answer.
        let call = calls_in.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if call < 2 {
            "HTTP/1.1 500 Internal Server Error
content-length: 0

"
            .to_string()
        } else {
            good.clone()
        }
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--diagnostics-out",
            diag.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(
        code, 0,
        "stdout: {stdout}
stderr: {stderr}"
    );
    let records = read_attempt_records(&diag);
    assert_eq!(records.len(), 1, "one attempt, retries folded in");
    assert_eq!(records[0]["state"], "stop_valid");
    assert_eq!(records[0]["transport_retries"], 2, "{:?}", records[0]);
    let log = read_attempts_log(&out);
    let attempt = log.iter().find(|r| r["kind"] == "attempt").unwrap();
    assert_eq!(attempt["transport_retries"], 2);
    assert!(!log.iter().any(|r| r["kind"] == "move"), "no ladder move");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// A piece too small to split that still overruns the escalated
/// budget fails the source with the named diagnosis — never a partial
/// import, never a prefix salvage, never an unbounded loop.
#[test]
fn a_minimum_unit_that_still_hits_length_after_escalation_fails_the_source() {
    let docs = batch_dir("extract-minunit-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-minunit-out");

    let (url, captured) = stub_chat_server_concurrent(|_index, _attempt| {
        chat_ok_with_finish_reason("truncated garbage", "length")
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--max-output-tokens",
            "512",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("cannot split further"), "{stderr}");
    assert_eq!(
        captured.lock().unwrap().len(),
        2,
        "the budgeted ask and one escalation — then fail, no loop"
    );
    assert!(stray_batch_files(&out).is_empty());

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// `finish_reason: "content_filter"` is terminal: no corrective turn
/// can argue with a policy refusal, so exactly one request goes out
/// and the source fails with the named class.
#[test]
fn refusal_is_terminal_with_no_corrective_turn() {
    let docs = batch_dir("extract-refusal-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-refusal-out");

    let (url, captured) = stub_chat_server_concurrent(|_index, _attempt| {
        chat_ok_with_finish_reason("", "content_filter")
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--max-output-tokens",
            "512",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stderr.contains("the provider refused this content"),
        "{stderr}"
    );
    assert!(stderr.contains("content_filter"), "{stderr}");
    assert_eq!(captured.lock().unwrap().len(), 1);
    assert!(stray_batch_files(&out).is_empty());

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// Under the ladder an empty answer gets exactly one corrective —
/// however high TAGURU_EXTRACT_MAX_ATTEMPTS is — then the named
/// diagnosis: a model that answers nothing twice will not answer on
/// the fifth try either.
#[test]
fn an_empty_answer_gets_exactly_one_corrective_however_high_max_attempts_is() {
    let docs = batch_dir("extract-emptycap-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-emptycap-out");

    let (url, captured) = stub_chat_server_concurrent(|_index, _attempt| chat_ok(""));
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_MAX_ATTEMPTS", "5"),
        ],
        &[
            "--context",
            "c",
            "--max-output-tokens",
            "512",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("the answer was empty"), "{stderr}");
    assert_eq!(captured.lock().unwrap().len(), 2);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// A malformed `stop` answer under the ladder still gets the ordinary
/// corrective loop — with the plain ask, never the legacy SHORTER
/// wording — and a constrained answer that failed validation is
/// called out as provider non-conformance.
#[test]
fn stop_malformed_still_runs_the_ordinary_corrective_loop_under_ladder_mode() {
    let docs = batch_dir("extract-laddermalformed-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-laddermalformed-out");

    let (url, captured) = stub_chat_server_concurrent(|_index, attempt| {
        if attempt == 0 {
            chat_ok("not json at all")
        } else {
            chat_ok(&json!({"associations": []}).to_string())
        }
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--structured-output",
            "json-object",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stderr.contains("provider non-conformance"),
        "a constrained answer that fails validation earns the named line: {stderr}"
    );

    let requests = captured.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let corrective = &requests[1];
    assert!(
        corrective.contains("Answer again with only the JSON object."),
        "{corrective}"
    );
    assert!(!corrective.contains("SHORTER"), "{corrective}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// `auto` sends one startup probe carrying EXACTLY the extraction
/// `response_format`; an answer in the canonical shape verifies the
/// json_schema rung and every extraction request keeps that format.
/// The probe ask must not say "json" — a prompted model answers it
/// with prose, which is precisely what tells the rungs apart.
#[test]
fn auto_probe_resolves_to_json_schema_when_the_backend_honors_it() {
    let docs = batch_dir("extract-probeschema-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-probeschema-out");

    let (url, requests) = stub_chat_server(vec![
        json!({"associations": [], "aliases": []}).to_string(),
        json!({"associations": []}).to_string(),
    ]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--structured-output",
            "auto",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stderr.contains("structured output: json_schema (probe verified)"),
        "{stderr}"
    );

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 2, "one probe, one extraction");
    let probe = json_body_of(&requests[0]);
    assert_eq!(probe["response_format"]["type"], "json_schema");
    assert_eq!(probe["max_tokens"], 256);
    let probe_ask = probe["messages"][1]["content"].as_str().unwrap();
    assert!(
        !probe_ask.to_ascii_lowercase().contains("json"),
        "the json_schema probe must invite prose: {probe_ask}"
    );
    let extraction = json_body_of(&requests[1]);
    assert_eq!(extraction["response_format"]["type"], "json_schema");
    assert!(extraction.get("max_tokens").is_none(), "{extraction}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// ADR 0021 (#760): a probe-verified json_schema rung that loops on a
/// real document (`length` at the budget and again at the escalated
/// resend) demotes the run to json_object — reported on stderr — and
/// the piece restarts; the next document starts on the demoted rung.
#[test]
fn auto_demotes_json_schema_after_a_looping_piece_and_reports_it() {
    let docs = batch_dir("extract-demote-docs");
    let a = docs.join("a.md");
    let b = docs.join("b.md");
    std::fs::write(&a, "small document").unwrap();
    std::fs::write(&b, "another small document").unwrap();
    let out = batch_dir("extract-demote-out");

    // Every request here is "chunk 0": the probe, a's rounds, b's
    // round — so `attempt` counts them in order.
    let (url, captured) = stub_chat_server_concurrent(|_index, attempt| match attempt {
        0 => chat_ok(&json!({"associations": [], "aliases": []}).to_string()),
        1 | 2 => chat_ok_with_finish_reason("looping garbage", "length"),
        _ => chat_ok(&json!({"associations": []}).to_string()),
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--structured-output",
            "auto",
            "--max-output-tokens",
            "512",
            a.to_str().unwrap(),
            b.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("2 written"), "{stdout}");
    assert!(
        stderr.contains("structured output: json_schema (probe verified)"),
        "{stderr}"
    );
    let expected = format!(
        "taguru: extract: {}: structured output: json_schema demoted to json_object — the \
         answer ended at the output cap even after the escalated resend under the \
         json_schema rung; the piece restarts at the ladder's top",
        a.display()
    );
    assert!(stderr.contains(&expected), "{stderr}");

    let requests = captured.lock().unwrap();
    assert_eq!(
        requests.len(),
        5,
        "probe, a at 512, a at 1024, a restarted at 512 under json_object, b"
    );
    let body = |i: usize| json_body_of(&requests[i]);
    assert_eq!(body(1)["response_format"]["type"], "json_schema");
    assert_eq!(body(1)["max_tokens"], 512);
    assert_eq!(body(2)["response_format"]["type"], "json_schema");
    assert_eq!(body(2)["max_tokens"], 1024);
    assert_eq!(body(3)["response_format"]["type"], "json_object");
    assert_eq!(body(3)["max_tokens"], 512);
    assert_eq!(
        body(4)["response_format"]["type"],
        "json_object",
        "the demotion is run-wide: {}",
        body(4)
    );

    // ADR 0029 (#791): the escalation and the demotion are records in
    // a.md's attempts log, with the stderr line's reason and both
    // rungs — the restart under json_object then succeeds, so no
    // split record follows.
    let a_batch = stray_batch_files(&out)
        .into_iter()
        .map(|entry| entry.to_string_lossy().into_owned())
        .find(|name| name.contains("a.md"))
        .unwrap();
    let a_records: Vec<Value> = std::fs::read_to_string(out.join(".extract-trace").join(format!(
        "{}.attempts.jsonl",
        a_batch.trim_end_matches(".jsonl")
    )))
    .unwrap()
    .lines()
    .map(|line| serde_json::from_str::<Value>(line).unwrap())
    .collect();
    let moves: Vec<&Value> = a_records
        .iter()
        .filter(|record| record["kind"] == "move")
        .collect();
    let kinds: Vec<&str> = moves.iter().map(|m| m["move"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["escalate", "demote"], "{moves:?}");
    assert_eq!(moves[1]["from_rung"], "json_schema");
    assert_eq!(moves[1]["to_rung"], "json_object");
    assert!(
        moves[1]["reason"]
            .as_str()
            .unwrap()
            .contains("even after the escalated resend"),
        "{}",
        moves[1]
    );
    assert_eq!(moves[1]["piece_id"], moves[0]["piece_id"]);

    // ADR 0031 §3.2: every attempt names the rung it was asked under —
    // the two under json_schema (base + escalated resend), the restart
    // under json_object.
    let attempts: Vec<&Value> = a_records
        .iter()
        .filter(|record| record["kind"] == "attempt")
        .collect();
    assert_eq!(attempts.len(), 3, "{attempts:?}");
    assert_eq!(attempts[0]["rung"], "json_schema", "{}", attempts[0]);
    assert_eq!(attempts[1]["rung"], "json_schema", "{}", attempts[1]);
    assert_eq!(attempts[2]["rung"], "json_object", "{}", attempts[2]);
    // The settings record names the rung `--structured-output auto`
    // resolved at startup — the probe's own verdict, before any demote.
    let settings = a_records
        .iter()
        .find(|record| record["kind"] == "settings")
        .unwrap();
    assert_eq!(settings["rung"], "json_schema", "{settings:?}");
    assert_eq!(settings["structured_output"], "auto");
    assert_eq!(settings["max_output_tokens"], 512);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// A prose answer to the json_schema probe fails that rung; JSON of
/// any shape to the second probe verifies json_object, and extraction
/// proceeds under it.
#[test]
fn auto_probe_falls_back_to_json_object_when_json_schema_is_not_honored() {
    let docs = batch_dir("extract-probeobject-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-probeobject-out");

    let (url, requests) = stub_chat_server(vec![
        "The sky is blue.".to_string(),
        json!({"color": "blue"}).to_string(),
        json!({"associations": []}).to_string(),
    ]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--structured-output",
            "auto",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stderr.contains("structured output: json_object"),
        "{stderr}"
    );

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 3, "two probes, one extraction");
    let object_probe = json_body_of(&requests[1]);
    assert_eq!(
        object_probe["response_format"],
        json!({"type": "json_object"})
    );
    // OpenAI's json_object mode refuses requests that never say
    // "json", so this probe's ask must.
    assert!(
        object_probe["messages"][1]["content"]
            .as_str()
            .unwrap()
            .to_ascii_lowercase()
            .contains("json"),
        "{object_probe}"
    );
    assert_eq!(
        json_body_of(&requests[2])["response_format"],
        json!({"type": "json_object"})
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// Neither probe verified: extraction runs exactly as it always has —
/// bare prompted JSON, no response_format at all — and says so.
#[test]
fn auto_probe_falls_back_to_prompted_json_when_neither_probe_succeeds() {
    let docs = batch_dir("extract-probebare-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-probebare-out");

    let (url, requests) = stub_chat_server(vec![
        "The sky is blue.".to_string(),
        "Sure! The sky is blue.".to_string(),
        json!({"associations": []}).to_string(),
    ]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--structured-output",
            "auto",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stderr.contains("structured output: prompted JSON only"),
        "{stderr}"
    );

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 3);
    assert_eq!(
        top_level_keys(&requests[2]),
        ["messages", "model", "temperature"],
        "an unverified endpoint gets exactly the request it always got"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// `--dry-run` calls nothing, so it probes nothing — `auto` resolves
/// no rung and reports none.
#[test]
fn auto_probe_is_skipped_under_dry_run() {
    let docs = batch_dir("extract-probedryrun-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-probedryrun-out");

    let (code, stdout, stderr) = run_extract(
        &out,
        &[],
        &[
            "--dry-run",
            "--context",
            "c",
            "--structured-output",
            "auto",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("would extract"), "{stdout}");
    assert!(!stderr.contains("structured output:"), "{stderr}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// The requested mode is a manifest computation input: changing it
/// re-extracts once, keeping it skips — exactly the --context /
/// --fact-budget discipline.
#[test]
fn changing_structured_output_mode_forces_a_re_extraction() {
    let docs = batch_dir("extract-modemanifest-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-modemanifest-out");

    let (url, requests) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, _) =
        run_extract(&out, &provider, &["--context", "c", doc.to_str().unwrap()]);
    assert_eq!(code, 0, "{stdout}");
    assert!(stdout.contains("1 written"), "{stdout}");
    assert_eq!(requests.join().unwrap().len(), 1);

    let (url, requests) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let mode_args = [
        "--context",
        "c",
        "--structured-output",
        "json-object",
        doc.to_str().unwrap(),
    ];
    let (code, stdout, _) = run_extract(&out, &provider, &mode_args);
    assert_eq!(code, 0, "{stdout}");
    assert!(
        stdout.contains("1 written"),
        "a changed mode must re-extract: {stdout}"
    );
    assert_eq!(requests.join().unwrap().len(), 1);

    // Same mode again: the manifest matches, nothing is called — the
    // dead endpoint would fail loudly if anything were.
    let provider = [
        ("TAGURU_EXTRACT_URL", "http://127.0.0.1:9"),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, _) = run_extract(&out, &provider, &mode_args);
    assert_eq!(code, 0, "{stdout}");
    assert!(stdout.contains("1 unchanged"), "{stdout}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn changing_max_output_tokens_forces_a_re_extraction() {
    let docs = batch_dir("extract-budgetmanifest-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-budgetmanifest-out");

    let (url, requests) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, _) =
        run_extract(&out, &provider, &["--context", "c", doc.to_str().unwrap()]);
    assert_eq!(code, 0, "{stdout}");
    assert_eq!(requests.join().unwrap().len(), 1);

    let (url, requests) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let budget_args = [
        "--context",
        "c",
        "--max-output-tokens",
        "512",
        doc.to_str().unwrap(),
    ];
    let (code, stdout, _) = run_extract(&out, &provider, &budget_args);
    assert_eq!(code, 0, "{stdout}");
    assert!(
        stdout.contains("1 written"),
        "a changed budget must re-extract: {stdout}"
    );
    assert_eq!(requests.join().unwrap().len(), 1);

    let provider = [
        ("TAGURU_EXTRACT_URL", "http://127.0.0.1:9"),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, _) = run_extract(&out, &provider, &budget_args);
    assert_eq!(code, 0, "{stdout}");
    assert!(stdout.contains("1 unchanged"), "{stdout}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

// Issue #199: merge-level silent item drop replaced by path-addressed
// corrective retry. The default (strict) mode earns a targeted
// corrective turn for a business-rule-invalid item instead of dropping
// it; `--lossy` restores the pre-#199 behavior exactly.

/// A single invalid weight earns one corrective turn naming its exact
/// path; when the model corrects it, every item survives and nothing
/// is reported dropped.
#[test]
fn strict_default_corrects_an_invalid_weight_and_keeps_every_item() {
    let docs = batch_dir("extract-strict-weight-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "a small b document").unwrap();
    let out = batch_dir("extract-strict-weight-out");

    let bad_reply = json!({
        "associations": [
            {"subject": "a", "label": "rel", "object": "b", "weight": "strong"}
        ],
        "aliases": []
    })
    .to_string();
    let good_reply = json!({
        "associations": [
            {"subject": "a", "label": "rel", "object": "b", "weight": 0.9}
        ],
        "aliases": []
    })
    .to_string();
    let (url, requests) = stub_chat_server(vec![bad_reply, good_reply]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) =
        run_extract(&out, &provider, &["--context", "c", doc.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("1 association(s)"), "{stdout}");
    assert!(!stdout.contains("dropped"), "{stdout}");

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests[1].contains("associations[0].weight: expected finite non-zero number, got string"),
        "{}",
        requests[1]
    );
    assert!(requests[1].contains("keep every item"), "{}", requests[1]);
    assert!(
        requests[1].contains("correct the fields listed above instead of deleting"),
        "{}",
        requests[1]
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// When the corrected answer is still invalid, the source fails
/// outright — no batch is written, matching the never-silent-drop
/// ruling (ADR 0001 §8).
#[test]
fn strict_default_fails_the_source_when_the_corrected_answer_is_still_invalid() {
    let docs = batch_dir("extract-strict-fail-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "a small b document").unwrap();
    let out = batch_dir("extract-strict-fail-out");

    let bad_reply = json!({
        "associations": [
            {"subject": "a", "label": "rel", "object": "b", "weight": "strong"}
        ],
        "aliases": []
    })
    .to_string();
    // Default max_attempts is 2 — both attempts answer the same
    // invalid weight, so the corrective turn cannot save it.
    let (url, requests) = stub_chat_server(vec![bad_reply.clone(), bad_reply]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) =
        run_extract(&out, &provider, &["--context", "c", doc.to_str().unwrap()]);
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stderr.contains("associations[0].weight: expected finite non-zero number"),
        "{stderr}"
    );
    assert!(
        stray_batch_files(&out).is_empty(),
        "a failed source must not write a batch file"
    );
    assert_eq!(requests.join().unwrap().len(), 2);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// A failed re-extraction (`--force` against an always-invalid stub)
/// must leave a previously written batch byte-for-byte untouched —
/// the server-side atomicity guarantee (#187) extended to the
/// producer side.
#[test]
fn a_failed_reextraction_leaves_the_existing_batch_untouched() {
    let docs = batch_dir("extract-strict-untouched-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "a small b document").unwrap();
    let out = batch_dir("extract-strict-untouched-out");

    let good_reply = json!({
        "associations": [
            {"subject": "a", "label": "rel", "object": "b", "weight": 1.0}
        ],
        "aliases": []
    })
    .to_string();
    let (url, _) = stub_chat_server(vec![good_reply]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) =
        run_extract(&out, &provider, &["--context", "c", doc.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let written = stray_batch_files(&out);
    assert_eq!(written.len(), 1, "{written:?}");
    let batch_path = out.join(&written[0]);
    let original_bytes = std::fs::read(&batch_path).unwrap();

    let bad_reply = json!({
        "associations": [
            {"subject": "a", "label": "rel", "object": "b", "weight": "strong"}
        ],
        "aliases": []
    })
    .to_string();
    let (url, _) = stub_chat_server(vec![bad_reply.clone(), bad_reply]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) = run_extract(
        &out,
        &provider,
        &["--force", "--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(
        std::fs::read(&batch_path).unwrap(),
        original_bytes,
        "a failed re-extraction must not touch the existing batch"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// Stage 2 (cross-chunk alias validation): a shadowing alias earns
/// its own corrective turn naming the exact alias path.
#[test]
fn a_shadowing_alias_earns_a_cross_chunk_corrective_turn() {
    let docs = batch_dir("extract-strict-shadowing-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "a small b document").unwrap();
    let out = batch_dir("extract-strict-shadowing-out");

    let bad_reply = json!({
        "associations": [
            {"subject": "a", "label": "rel", "object": "b"}
        ],
        "aliases": [
            {"alias": "a", "canonical": "b", "kind": "concept"}
        ]
    })
    .to_string();
    let good_reply = json!({
        "associations": [
            {"subject": "a", "label": "rel", "object": "b"}
        ],
        "aliases": [
            {"alias": "x", "canonical": "a", "kind": "concept"}
        ]
    })
    .to_string();
    let (url, requests) = stub_chat_server(vec![bad_reply, good_reply]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) =
        run_extract(&out, &provider, &["--context", "c", doc.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(!stdout.contains("dropped"), "{stdout}");
    assert!(!stdout.contains("removed"), "{stdout}");

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests[1].contains("aliases[0].alias: names something the associations already contain"),
        "{}",
        requests[1]
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// ADR 0022 (#763): a shadowing alias the corrective turn does not fix
/// is removed with accounting — named on stderr, counted on the report
/// line — and the document is written, instead of one uncorrectable
/// alias costing every fact the document held.
#[test]
fn an_uncorrected_shadowing_alias_is_removed_and_the_document_still_lands() {
    let docs = batch_dir("extract-uncorrected-shadow-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "a small b document").unwrap();
    let out = batch_dir("extract-uncorrected-shadow-out");

    let bad_reply = json!({
        "associations": [
            {"subject": "a", "label": "rel", "object": "b"}
        ],
        "aliases": [
            {"alias": "a", "canonical": "b", "kind": "concept"}
        ]
    })
    .to_string();
    // The model answers the corrective turn with the very same thing.
    let (url, requests) = stub_chat_server(vec![bad_reply.clone(), bad_reply]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) =
        run_extract(&out, &provider, &["--context", "c", doc.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("1 association(s), 0 alias(es)"), "{stdout}");
    assert!(
        stdout.contains("1 item(s) removed (mechanical validation)"),
        "{stdout}"
    );
    let expected = format!(
        "taguru: extract: {}: removed: aliases[0].alias: names something the associations \
         already contain — still so after the corrective turn; removed",
        doc.display()
    );
    assert!(stderr.contains(&expected), "{stderr}");
    assert_eq!(
        requests.join().unwrap().len(),
        2,
        "one ask, one corrective turn"
    );
    let batches = stray_batch_files(&out);
    assert_eq!(batches.len(), 1, "{batches:?}");
    let batch = std::fs::read_to_string(out.join(&batches[0])).unwrap();
    assert!(batch.contains("\"subject\":\"a\""), "{batch}");
    assert!(!batch.contains("\"alias\""), "{batch}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// ADR 0013 (#496 S1): a dangling canonical is no longer a corrective
/// issue — it is pruned mechanically after Stage 2, with the removal
/// named on stderr and counted on the report line, and the run spends
/// zero corrective turns on it.
#[test]
fn a_dangling_alias_is_pruned_mechanically_with_zero_corrective_turns() {
    let docs = batch_dir("extract-strict-dangling-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "a small b document").unwrap();
    let out = batch_dir("extract-strict-dangling-out");

    let reply = json!({
        "associations": [
            {"subject": "a", "label": "rel", "object": "b"}
        ],
        "aliases": [
            {"alias": "x", "canonical": "存在しない", "kind": "concept"}
        ]
    })
    .to_string();
    let (url, requests) = stub_chat_server(vec![reply]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) =
        run_extract(&out, &provider, &["--context", "c", doc.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(!stdout.contains("dropped"), "{stdout}");
    assert!(
        stdout.contains("1 item(s) removed (mechanical validation)"),
        "{stdout}"
    );
    assert!(
        stderr.contains(
            "removed: aliases[0]: canonical \"存在しない\" names nothing the associations contain"
        ),
        "{stderr}"
    );
    assert_eq!(
        requests.join().unwrap().len(),
        1,
        "a dangling canonical must not spend a corrective turn"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// A `--questions` answer citing a paragraph out of the document's
/// canonical range earns a corrective turn, not a silent drop.
#[test]
fn an_out_of_range_question_paragraph_earns_a_corrective_turn() {
    let docs = batch_dir("extract-strict-question-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "段落0の文です。").unwrap();
    let out = batch_dir("extract-strict-question-out");

    let bad_reply = json!({
        "associations": [],
        "aliases": [],
        "questions": [{"paragraph": 9, "question": "何?"}]
    })
    .to_string();
    let good_reply = json!({
        "associations": [],
        "aliases": [],
        "questions": [{"paragraph": 0, "question": "何?"}]
    })
    .to_string();
    let (url, requests) = stub_chat_server(vec![bad_reply, good_reply]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) = run_extract(
        &out,
        &provider,
        &["--questions", "1", "--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("1 question(s)"), "{stdout}");

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests[1].contains("questions[0].paragraph: must cite a paragraph below 1, got 9"),
        "{}",
        requests[1]
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// An alias in an early chunk whose canonical only shows up in a later
/// chunk is exactly today's `merge()` comment: Stage 2 must resolve it
/// against the FULL merged name set and never spend a corrective turn
/// on it.
#[test]
fn a_chunk_1_alias_resolved_by_a_later_chunk_needs_no_corrective_turn() {
    let docs = batch_dir("extract-strict-crosschunk-docs");
    let doc = docs.join("big.md");
    // The final paragraph carries the names the last chunk's answer
    // uses, so the occurrence check (ADR 0013) attests them.
    std::fs::write(
        &doc,
        format!("{}\n\n青嶺酒造の杜氏は高瀬。", multi_chunk_document(20)),
    )
    .unwrap();
    let doc_src = doc.to_str().unwrap();
    let out = batch_dir("extract-strict-crosschunk-out");

    let (code, dry_stdout, stderr) =
        run_extract(&out, &[], &["--dry-run", "--context", "c", doc_src]);
    assert_eq!(code, 0, "stdout: {dry_stdout}\nstderr: {stderr}");
    let total_chunks = chunk_count_from_dry_run(&dry_stdout);
    assert!(
        total_chunks >= 2,
        "fixture must split into at least 2 chunks to prove cross-chunk resolution: {dry_stdout}"
    );

    let mut replies: Vec<String> = (0..total_chunks)
        .map(|_| json!({"associations": [], "aliases": []}).to_string())
        .collect();
    replies[0] = json!({
        "associations": [],
        "aliases": [{"alias": "Aomine", "canonical": "青嶺酒造", "kind": "concept"}]
    })
    .to_string();
    *replies.last_mut().unwrap() = json!({
        "associations": [{"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬"}],
        "aliases": []
    })
    .to_string();

    let (url, requests) = stub_chat_server(replies);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) = run_extract(&out, &provider, &["--context", "c", doc_src]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(!stdout.contains("dropped"), "{stdout}");
    assert_eq!(
        requests.join().unwrap().len(),
        total_chunks,
        "a canonical resolved by a later chunk must not trigger any corrective turn"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// `--lossy` restores the pre-#199 drop-and-proceed behavior exactly:
/// no corrective turn is spent on a validity issue, and the report
/// marks the drop explicitly so it is never confused with a policy
/// trim.
#[test]
fn lossy_flag_skips_correction_and_marks_the_drop_explicitly() {
    let docs = batch_dir("extract-lossy-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-lossy-out");

    // weight 0 (not a wrong-typed weight): merge()'s lenient default
    // for a MISSING/malformed weight is 1.0 (a plain assertion, not a
    // drop) — only a well-typed business-rule violation like zero
    // actually gets dropped, so this is a faithful pre-#199 drop case.
    let bad_reply = json!({
        "associations": [
            {"subject": "a", "label": "rel", "object": "b", "weight": 0}
        ],
        "aliases": []
    })
    .to_string();
    let (url, requests) = stub_chat_server(vec![bad_reply]);
    let provider = [
        ("TAGURU_EXTRACT_URL", url.as_str()),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
    ];
    let (code, stdout, stderr) = run_extract(
        &out,
        &provider,
        &["--lossy", "--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("1 item(s) dropped (--lossy)"), "{stdout}");
    assert_eq!(
        requests.join().unwrap().len(),
        1,
        "--lossy must never spend a corrective turn on a validity issue"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// `TAGURU_EXTRACT_MAX_ATTEMPTS=1` bounds Stage 1's validity corrective
/// turn exactly like it bounds the syntax corrective turn: one attempt
/// total, no correction, straight to failure.
#[test]
fn strict_default_with_max_attempts_of_one_skips_the_validity_corrective_turn() {
    let docs = batch_dir("extract-strict-maxone-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-strict-maxone-out");

    let bad_reply = json!({
        "associations": [
            {"subject": "a", "label": "rel", "object": "b", "weight": 0}
        ],
        "aliases": []
    })
    .to_string();
    let (url, requests) = stub_chat_server(vec![bad_reply]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_MAX_ATTEMPTS", "1"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stderr.contains("associations[0].weight: expected finite non-zero number"),
        "{stderr}"
    );
    assert_eq!(
        requests.join().unwrap().len(),
        1,
        "max_attempts=1 must not send a corrective turn even for a validity issue"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn extract_rejects_a_bad_lossy_env_var_value() {
    let docs = batch_dir("extract-lossy-env-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "content").unwrap();
    let out = batch_dir("extract-lossy-env-out");

    let provider = [
        ("TAGURU_EXTRACT_URL", "http://127.0.0.1:9"),
        ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ("TAGURU_EXTRACT_LOSSY", "nope"),
    ];
    let (code, _, stderr) =
        run_extract(&out, &provider, &["--context", "c", doc.to_str().unwrap()]);
    assert_eq!(code, 2, "{stderr}");
    assert!(
        stderr.contains("TAGURU_EXTRACT_LOSSY takes 1/true or 0/false"),
        "{stderr}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// TAGURU_EXTRACT_LOSSY alone (no `--lossy` flag) must engage lossy
/// mode — the same flag-defers-to-env pattern as every other
/// TAGURU_EXTRACT_* control.
#[test]
fn extract_lossy_env_var_enables_lossy_mode_without_the_flag() {
    let docs = batch_dir("extract-lossy-envon-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-lossy-envon-out");

    let bad_reply = json!({
        "associations": [
            {"subject": "a", "label": "rel", "object": "b", "weight": 0}
        ],
        "aliases": []
    })
    .to_string();
    let (url, requests) = stub_chat_server(vec![bad_reply]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_LOSSY", "true"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("(--lossy)"), "{stdout}");
    assert_eq!(requests.join().unwrap().len(), 1);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

// ---------------------------------------------------------------------
// Issue #200: `--diagnostics-out` / TAGURU_EXTRACT_DIAGNOSTICS
// ---------------------------------------------------------------------

/// One JSONL record per attempt, sharing the ADR 0001 §7 state
/// vocabulary and the Python event field names (issue #200).
#[test]
fn diagnostics_out_writes_one_record_per_attempt_with_the_shared_state_vocabulary() {
    let docs = batch_dir("extract-diag-basic-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-diag-basic-out");
    let diag_dir = batch_dir("extract-diag-basic-diag");
    let diag = diag_dir.join("diag.jsonl");

    let (url, _requests) = stub_chat_server(vec![
        "not json".to_string(),
        json!({"associations": []}).to_string(),
    ]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--diagnostics-out",
            diag.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("1 written"), "{stdout}");

    let records = read_attempt_records(&diag);
    assert_eq!(records.len(), 2, "{records:?}");

    assert_eq!(records[0]["kind"], "attempt");
    assert_eq!(records[0]["source"], doc.to_str().unwrap());
    assert_eq!(records[0]["stage"], "item");
    assert_eq!(records[0]["chunk_index"], 0);
    assert_eq!(records[0]["attempt"], 1);
    assert_eq!(records[0]["max_attempts"], 2);
    assert_eq!(records[0]["state"], "stop_malformed");
    assert_eq!(records[0]["length_limited"], false);
    assert!(records[0]["elapsed_seconds"].as_f64().unwrap() >= 0.0);
    assert!(!records[0]["parse_error"].is_null(), "{:?}", records[0]);
    assert!(records[0]["validation_issues"].is_null());
    assert!(!records[0]["provider_metadata"].is_null());
    assert!(
        records[0].get("piece_bytes").is_none(),
        "the legacy (non-ladder) path never sends piece_bytes: {:?}",
        records[0]
    );
    assert!(
        records[0].get("requested_max_tokens").is_none(),
        "{:?}",
        records[0]
    );
    assert!(
        records[0].get("response_text").is_none(),
        "metadata only by default: {:?}",
        records[0]
    );

    assert_eq!(records[1]["attempt"], 2);
    assert_eq!(records[1]["state"], "stop_valid");
    assert!(records[1]["parse_error"].is_null());
    assert!(records[1]["validation_issues"].is_null());

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// `--diagnostics-out` wins over a conflicting TAGURU_EXTRACT_DIAGNOSTICS
/// path — the same flag-over-environment precedence every other control
/// follows.
#[test]
fn diagnostics_out_flag_wins_over_the_environment_variable() {
    let docs = batch_dir("extract-diag-precedence-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-diag-precedence-out");
    let env_dir = batch_dir("extract-diag-precedence-env");
    let env_path = env_dir.join("env.jsonl");
    let flag_dir = batch_dir("extract-diag-precedence-flag");
    let flag_path = flag_dir.join("flag.jsonl");

    let (url, _requests) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_DIAGNOSTICS", env_path.to_str().unwrap()),
        ],
        &[
            "--context",
            "c",
            "--diagnostics-out",
            flag_path.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(flag_path.is_file(), "the flag must win: {flag_path:?}");
    assert!(
        !env_path.exists(),
        "the env var's path must be ignored once the flag is given: {env_path:?}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&env_dir);
    let _ = std::fs::remove_dir_all(&flag_dir);
}

/// TAGURU_EXTRACT_DIAGNOSTICS alone (no flag) also opens the sidecar.
#[test]
fn diagnostics_env_var_alone_opens_the_sidecar() {
    let docs = batch_dir("extract-diag-envonly-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-diag-envonly-out");
    let diag_dir = batch_dir("extract-diag-envonly-diag");
    let diag = diag_dir.join("diag.jsonl");

    let (url, _requests) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_DIAGNOSTICS", diag.to_str().unwrap()),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let records = read_attempt_records(&diag);
    assert_eq!(records.len(), 1, "{records:?}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// A non-numeric TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES is a hard usage
/// error, not a silently ignored knob — the same discipline every other
/// TAGURU_EXTRACT_* env var follows.
#[test]
fn diagnostics_raw_bytes_env_var_rejects_a_non_integer() {
    let docs = batch_dir("extract-diag-rawbad-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-diag-rawbad-out");

    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", "http://127.0.0.1:1"),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES", "not-a-number"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 2, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stderr.contains("TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES"),
        "{stderr}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// The #188 acceptance criterion: a length-terminated attempt, an
/// empty answer, and a policy refusal — all reached through the ADR
/// 0001 §7 ladder — each earn a distinct `state`, not the same generic
/// failure.
#[test]
fn diagnostics_distinguishes_length_limited_empty_and_refusal_states() {
    let docs = batch_dir("extract-diag-states-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();

    {
        // length_limited: the ladder escalates once, so two requests go
        // out — the first cut off at the budget, the second accepted.
        let out = batch_dir("extract-diag-states-length-out");
        let diag_dir = batch_dir("extract-diag-states-length-diag");
        let diag = diag_dir.join("diag.jsonl");
        let (url, _captured) = stub_chat_server_concurrent(|_index, attempt| {
            if attempt == 0 {
                chat_ok_with_finish_reason("truncated garbage", "length")
            } else {
                chat_ok(&json!({"associations": []}).to_string())
            }
        });
        let (code, stdout, stderr) = run_extract(
            &out,
            &[
                ("TAGURU_EXTRACT_URL", url.as_str()),
                ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ],
            &[
                "--context",
                "c",
                "--max-output-tokens",
                "512",
                "--diagnostics-out",
                diag.to_str().unwrap(),
                doc.to_str().unwrap(),
            ],
        );
        assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
        let records = read_attempt_records(&diag);
        assert_eq!(records.len(), 2, "{records:?}");
        assert_eq!(records[0]["state"], "length_limited");
        assert_eq!(records[0]["length_limited"], true);
        assert_eq!(records[0]["requested_max_tokens"], 512);
        assert!(!records[0]["parse_error"].is_null());
        assert_eq!(records[1]["state"], "stop_valid");
        assert_eq!(
            records[1]["requested_max_tokens"], 1024,
            "escalation records the factored cap it resent at: {:?}",
            records[1]
        );

        let _ = std::fs::remove_dir_all(&out);
        let _ = std::fs::remove_dir_all(&diag_dir);
    }

    {
        // empty: exactly one corrective under the ladder — two
        // attempts, both diagnosed "empty".
        let out = batch_dir("extract-diag-states-empty-out");
        let diag_dir = batch_dir("extract-diag-states-empty-diag");
        let diag = diag_dir.join("diag.jsonl");
        let (url, _captured) = stub_chat_server_concurrent(|_index, _attempt| chat_ok(""));
        let (code, stdout, stderr) = run_extract(
            &out,
            &[
                ("TAGURU_EXTRACT_URL", url.as_str()),
                ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ],
            &[
                "--context",
                "c",
                "--max-output-tokens",
                "512",
                "--diagnostics-out",
                diag.to_str().unwrap(),
                doc.to_str().unwrap(),
            ],
        );
        assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
        let all = read_diagnostics(&diag);
        assert!(
            !all.iter().any(|record| record["kind"] == "document"),
            "a document that never lands earns no summary record: {all:?}"
        );
        let records = read_attempt_records(&diag);
        assert_eq!(records.len(), 2, "{records:?}");
        assert_eq!(records[0]["state"], "empty");
        assert_eq!(records[1]["state"], "empty");

        let _ = std::fs::remove_dir_all(&out);
        let _ = std::fs::remove_dir_all(&diag_dir);
    }

    {
        // refusal: terminal on the first attempt, no corrective turn.
        let out = batch_dir("extract-diag-states-refusal-out");
        let diag_dir = batch_dir("extract-diag-states-refusal-diag");
        let diag = diag_dir.join("diag.jsonl");
        let (url, _captured) = stub_chat_server_concurrent(|_index, _attempt| {
            chat_ok_with_finish_reason("", "content_filter")
        });
        let (code, stdout, stderr) = run_extract(
            &out,
            &[
                ("TAGURU_EXTRACT_URL", url.as_str()),
                ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ],
            &[
                "--context",
                "c",
                "--max-output-tokens",
                "512",
                "--diagnostics-out",
                diag.to_str().unwrap(),
                doc.to_str().unwrap(),
            ],
        );
        assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
        let all = read_diagnostics(&diag);
        assert!(
            !all.iter().any(|record| record["kind"] == "document"),
            "a document that never lands earns no summary record: {all:?}"
        );
        let records = read_attempt_records(&diag);
        assert_eq!(records.len(), 1, "{records:?}");
        assert_eq!(records[0]["state"], "refusal");
        assert!(!records[0]["parse_error"].is_null());

        let _ = std::fs::remove_dir_all(&out);
        let _ = std::fs::remove_dir_all(&diag_dir);
    }

    let _ = std::fs::remove_dir_all(&docs);
}

/// A timeout is one attempt, not one per transport retry — the four
/// `RETRY_ATTEMPTS` inside `ChatClient::complete` are a single
/// extraction-level attempt from the diagnostics sink's point of view.
#[test]
fn diagnostics_records_a_timeout_as_a_single_attempt_with_no_provider_metadata() {
    let docs = batch_dir("extract-diag-timeout-docs");
    let doc = docs.join("slow.md");
    std::fs::write(&doc, "content").unwrap();
    let out = batch_dir("extract-diag-timeout-out");
    let diag_dir = batch_dir("extract-diag-timeout-diag");
    let diag = diag_dir.join("diag.jsonl");

    // Same stalled-provider shape as the_extract_timeout_knob_bounds_a_
    // stalled_provider: every retry's connection is accepted and held
    // open, unanswered, well past the client's worst-case retry budget.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let mut held = Vec::new();
        for _ in 0..4 {
            if let Ok((stream, _)) = listener.accept() {
                held.push(stream);
            }
        }
        std::thread::sleep(std::time::Duration::from_secs(30));
    });

    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_TIMEOUT_SECS", "1"),
        ],
        &[
            "--context",
            "c",
            "--diagnostics-out",
            diag.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");

    let records = read_attempt_records(&diag);
    assert_eq!(
        records.len(),
        1,
        "transport retries must not each earn their own record: {records:?}"
    );
    assert_eq!(records[0]["state"], "timeout");
    assert!(records[0]["provider_metadata"].is_null());
    assert!(!records[0]["parse_error"].is_null());

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// A non-retryable 4xx is TRANSPORT, not TIMEOUT — the provider
/// answered, it just refused the request outright.
#[test]
fn diagnostics_records_a_non_retryable_http_error_as_transport() {
    let docs = batch_dir("extract-diag-transport-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-diag-transport-out");
    let diag_dir = batch_dir("extract-diag-transport-diag");
    let diag = diag_dir.join("diag.jsonl");

    let (url, _captured) =
        stub_chat_server_concurrent(|_index, _attempt| chat_error(400, "Bad Request", "", "nope"));
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--diagnostics-out",
            diag.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");

    let records = read_attempt_records(&diag);
    assert_eq!(records.len(), 1, "{records:?}");
    assert_eq!(records[0]["state"], "transport");
    assert!(records[0]["provider_metadata"].is_null());
    assert!(
        records[0]["parse_error"].as_str().unwrap().contains("400"),
        "{:?}",
        records[0]
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// Token counts the provider reports land in `provider_metadata`,
/// translated to the shared (Python `ProviderMetadata`) field names.
#[test]
fn diagnostics_reports_provider_token_usage_when_present() {
    let docs = batch_dir("extract-diag-usage-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-diag-usage-out");
    let diag_dir = batch_dir("extract-diag-usage-diag");
    let diag = diag_dir.join("diag.jsonl");

    let (url, _captured) = stub_chat_server_concurrent(|_index, _attempt| {
        chat_ok_with_usage(&json!({"associations": []}).to_string(), 123, 45)
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--diagnostics-out",
            diag.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let records = read_attempt_records(&diag);
    assert_eq!(records.len(), 1, "{records:?}");
    let metadata = &records[0]["provider_metadata"];
    assert_eq!(metadata["input_tokens"], 123);
    assert_eq!(metadata["output_tokens"], 45);
    assert_eq!(metadata["total_tokens"], 168);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES opts into a byte-capped raw
/// answer, truncated at capture exactly like `corrective_context_bytes`.
#[test]
fn diagnostics_raw_bytes_attaches_a_capped_response_text() {
    let docs = batch_dir("extract-diag-raw-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-diag-raw-out");
    let diag_dir = batch_dir("extract-diag-raw-diag");
    let diag = diag_dir.join("diag.jsonl");

    let (url, _requests) = stub_chat_server(vec![
        "this reply is definitely longer than eight bytes".to_string(),
    ]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_MAX_ATTEMPTS", "1"),
            ("TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES", "8"),
        ],
        &[
            "--context",
            "c",
            "--diagnostics-out",
            diag.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");

    let records = read_attempt_records(&diag);
    assert_eq!(records.len(), 1, "{records:?}");
    let text = records[0]["response_text"]
        .as_str()
        .expect("response_text must be present when RAW_BYTES is set");
    assert!(text.starts_with("this rep"), "{text:?}");
    assert!(text.contains("[truncated to 8 bytes]"), "{text:?}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// Without TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES, `response_text` never
/// appears — metadata only by default (ADR 0001 §10).
#[test]
fn diagnostics_omits_response_text_when_raw_bytes_is_unset() {
    let docs = batch_dir("extract-diag-noraw-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-diag-noraw-out");
    let diag_dir = batch_dir("extract-diag-noraw-diag");
    let diag = diag_dir.join("diag.jsonl");

    let (url, _requests) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--diagnostics-out",
            diag.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let records = read_attempt_records(&diag);
    assert_eq!(records.len(), 1, "{records:?}");
    assert!(
        records[0].get("response_text").is_none(),
        "metadata-only by default: {:?}",
        records[0]
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// A killed run keeps every diagnostics record already written — the
/// same incremental-persistence contract as the manifest
/// (`extract_persists_the_manifest_after_each_document_not_only_at_the_end`),
/// applied to the sidecar.
#[test]
fn diagnostics_is_written_incrementally_and_survives_a_kill() {
    let docs = batch_dir("extract-diag-kill-docs");
    let fast = docs.join("fast.md");
    let slow = docs.join("slow.md");
    std::fs::write(&fast, "青嶺酒造は1907年に創業した。").unwrap();
    std::fs::write(&slow, "高瀬は青嶺酒造の杜氏。").unwrap();
    let fast_src = fast.to_str().unwrap().to_string();

    let reply = chat_ok(&json!({"associations": []}).to_string());
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        use std::io::Write;
        let mut held = Vec::new();
        for (index, stream) in listener.incoming().enumerate() {
            let Ok(mut stream) = stream else { continue };
            if index == 0 {
                let _ = read_http_request(&mut stream);
                let _ = stream.write_all(reply.as_bytes());
            } else {
                held.push(stream);
            }
        }
    });

    let out = batch_dir("extract-diag-kill-out");
    let diag_dir = batch_dir("extract-diag-kill-diag");
    let diag = diag_dir.join("diag.jsonl");
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    scrub_extract_env(&mut command)
        .arg("extract")
        .env("TAGURU_EXTRACT_URL", &url)
        .env("TAGURU_EXTRACT_MODEL", "stub-model")
        .args([
            "--out",
            out.to_str().unwrap(),
            "--context",
            "c",
            "--diagnostics-out",
            diag.to_str().unwrap(),
        ])
        .arg(&fast)
        .arg(&slow)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut child = command.spawn().expect("extract must spawn");

    // The first line to land is now a "chunk" provenance record (issue
    // #262) — poll until an "attempt" record has landed too, not merely
    // until the file is non-empty.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut saved = String::new();
    while std::time::Instant::now() < deadline {
        if let Ok(text) = std::fs::read_to_string(&diag)
            && text.lines().filter(|line| !line.is_empty()).any(|line| {
                serde_json::from_str::<Value>(line)
                    .map(|value| value["kind"] == "attempt")
                    .unwrap_or(false)
            })
        {
            saved = text;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let _ = child.kill();
    let _ = child.wait();
    assert!(
        !saved.trim().is_empty(),
        "no diagnostics record landed before the run was killed"
    );
    // The poll above tolerates an unparseable trailing line (the run
    // may still be mid-write when `saved` is captured) — this second
    // pass over the same snapshot must too, or a torn last line fails
    // the test spuriously instead of exercising the ordering invariant.
    let records: Vec<Value> = saved
        .lines()
        .filter(|line| !line.is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    assert!(!records.is_empty(), "no complete diagnostics line survived");
    // ADR 0023 §3.3: the sidecar's first line names the run; the
    // chunk record still lands before that chunk's first attempt.
    assert_eq!(records[0]["kind"], "run", "{records:?}");
    assert_eq!(
        records[1]["kind"], "chunk",
        "the chunk record lands before that chunk's first attempt: {records:?}"
    );
    let record = records
        .iter()
        .find(|value| value["kind"] == "attempt")
        .expect("an attempt record must have landed");
    assert_eq!(record["state"], "stop_valid");
    assert_eq!(record["source"], fast_src.as_str());

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// Without `--diagnostics-out`/TAGURU_EXTRACT_DIAGNOSTICS, extract
/// never writes a sidecar — off by default (requirement 4).
#[test]
fn extract_without_diagnostics_out_writes_no_sidecar() {
    let docs = batch_dir("extract-diag-off-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-diag-off-out");
    let diag_dir = batch_dir("extract-diag-off-diag");
    let phantom = diag_dir.join("would-be.jsonl");

    let (url, _requests) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("1 written"), "{stdout}");
    assert!(
        !phantom.exists(),
        "extract must never write a diagnostics sidecar without \
         --diagnostics-out/TAGURU_EXTRACT_DIAGNOSTICS"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// `--dry-run` calls nothing, so it opens no sidecar either (the usage
/// text says so) — not even the `run` record ADR 0023 puts first, and
/// not the trace directory.
#[test]
fn dry_run_opens_no_diagnostics_sidecar_and_no_trace() {
    let docs = batch_dir("extract-diag-dryrun-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-diag-dryrun-out");
    let diag_dir = batch_dir("extract-diag-dryrun-diag");
    let diag = diag_dir.join("diag.jsonl");

    let (code, stdout, stderr) = run_extract(
        &out,
        &[],
        &[
            "--dry-run",
            "--context",
            "c",
            "--diagnostics-out",
            diag.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("would extract"), "{stdout}");
    assert!(!diag.exists(), "--dry-run must not open the sidecar");
    assert!(!out.join(".extract-trace").exists());

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// Stage 2's cross-chunk alias correction earns its own diagnostics
/// record, `stage: "cross_chunk"` — distinct from the item-stage
/// records the same run also emits.
#[test]
fn diagnostics_records_the_stage_two_cross_chunk_correction() {
    let docs = batch_dir("extract-diag-stage2-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "a small b document").unwrap();
    let out = batch_dir("extract-diag-stage2-out");
    let diag_dir = batch_dir("extract-diag-stage2-diag");
    let diag = diag_dir.join("diag.jsonl");

    // A SHADOWING alias: since ADR 0013 a dangling canonical is pruned
    // mechanically instead of corrected, so shadowing is what still
    // exercises the cross_chunk corrective record.
    let bad_reply = json!({
        "associations": [
            {"subject": "a", "label": "rel", "object": "b"}
        ],
        "aliases": [
            {"alias": "a", "canonical": "b", "kind": "concept"}
        ]
    })
    .to_string();
    let good_reply = json!({
        "associations": [
            {"subject": "a", "label": "rel", "object": "b"}
        ],
        "aliases": [
            {"alias": "x", "canonical": "a", "kind": "concept"}
        ]
    })
    .to_string();
    let (url, _requests) = stub_chat_server(vec![bad_reply, good_reply]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--diagnostics-out",
            diag.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let records = read_attempt_records(&diag);
    assert_eq!(records.len(), 2, "{records:?}");
    assert_eq!(records[0]["stage"], "item");
    assert_eq!(records[0]["state"], "stop_valid");
    assert_eq!(records[1]["stage"], "cross_chunk");
    assert_eq!(records[1]["state"], "stop_valid");
    assert_eq!(records[1]["attempt"], 1);
    assert_eq!(records[1]["max_attempts"], 1);
    // ADR 0023: both completions are numbered in issue order and name
    // the same piece; the trace's piece record names the CORRECTIVE
    // completion — the answer the batch actually came from — and the
    // alias the correction introduced is traced to that piece.
    assert_eq!(records[0]["attempt_seq"], 1);
    assert_eq!(records[1]["attempt_seq"], 2);
    assert_eq!(records[1]["piece_id"], records[0]["piece_id"]);
    // ADR 0028: Stage 2 corrects the ACCEPTED Stage 1 attempt whose
    // output it replays.
    assert!(records[0].get("corrects").is_none());
    assert_eq!(records[1]["corrects"]["run_id"], records[0]["run_id"]);
    assert_eq!(
        records[1]["corrects"]["attempt_seq"],
        records[0]["attempt_seq"]
    );
    let (_, trace) = read_trace(&out);
    let piece = trace.iter().find(|r| r["kind"] == "piece").unwrap();
    assert_eq!(piece["piece_id"], records[0]["piece_id"]);
    assert_eq!(piece["attempt"]["run_id"], records[1]["run_id"]);
    assert_eq!(piece["attempt"]["attempt_seq"], 2, "{piece:?}");
    let alias = trace
        .iter()
        .find(|r| r["kind"] == "item" && r["item"] == "concept")
        .unwrap();
    assert_eq!(alias["alias"], "x");
    assert_eq!(alias["piece_id"], piece["piece_id"]);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// `--parallel` dispatches chunk workers concurrently onto the same
/// sidecar — every attempt still earns exactly one well-formed line.
#[test]
fn diagnostics_records_every_chunk_attempt_under_parallel() {
    let docs = batch_dir("extract-diag-parallel-docs");
    let doc = docs.join("big.md");
    std::fs::write(&doc, multi_chunk_document(20)).unwrap();
    let out = batch_dir("extract-diag-parallel-out");
    let diag_dir = batch_dir("extract-diag-parallel-diag");
    let diag = diag_dir.join("diag.jsonl");

    let (url, captured) = stub_chat_server_concurrent(|_index, _attempt| {
        chat_ok(&json!({"associations": []}).to_string())
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--parallel",
            "2",
            "--diagnostics-out",
            diag.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let request_count = captured.lock().unwrap().len();
    let records = read_attempt_records(&diag);
    assert_eq!(records.len(), request_count, "{records:?}");
    assert!(
        records.len() > 1,
        "the document must actually split into multiple chunks"
    );
    for record in &records {
        assert_eq!(record["state"], "stop_valid");
        assert_eq!(record["stage"], "item");
    }

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// Issue #262, ADR 0003 §7: one `kind: "chunk"` record per chunk,
/// written before that chunk's first attempt — with correct paragraph
/// provenance a reader can join against the canonical document without
/// re-implementing `chunk()`'s packing rule.
#[test]
fn diagnostics_writes_one_chunk_record_per_chunk_before_any_attempt() {
    let docs = batch_dir("extract-diag-chunkrec-docs");
    let doc = docs.join("big.md");
    std::fs::write(&doc, multi_chunk_document(20)).unwrap();
    let out = batch_dir("extract-diag-chunkrec-out");
    let diag_dir = batch_dir("extract-diag-chunkrec-diag");
    let diag = diag_dir.join("diag.jsonl");

    let (url, captured) = stub_chat_server_concurrent(|_index, _attempt| {
        chat_ok(&json!({"associations": []}).to_string())
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--diagnostics-out",
            diag.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let all = read_diagnostics(&diag);
    let last_chunk_position = all.iter().rposition(|record| record["kind"] == "chunk");
    let first_attempt_position = all.iter().position(|record| record["kind"] == "attempt");
    assert!(
        last_chunk_position < first_attempt_position,
        "every chunk record must land before any attempt record: {all:?}"
    );

    let chunks: Vec<&Value> = all
        .iter()
        .filter(|record| record["kind"] == "chunk")
        .collect();
    let request_count = captured.lock().unwrap().len();
    assert_eq!(
        chunks.len(),
        request_count,
        "one chunk record per chunk sent: {chunks:?}"
    );
    assert!(
        chunks.len() > 1,
        "the document must actually split into multiple chunks"
    );
    for (index, record) in chunks.iter().enumerate() {
        assert_eq!(record["source"], doc.to_str().unwrap());
        assert_eq!(record["chunk_index"], index);
        assert_eq!(record["chunk_total"], chunks.len());
        let sha = record["chunk_sha256"].as_str().unwrap();
        assert_eq!(sha.len(), 64, "{sha:?} must be a sha256 hex digest");
        assert!(sha.chars().all(|c| c.is_ascii_hexdigit()));
        let bytes = record["chunk_bytes"].as_u64().unwrap();
        assert!(
            (1..=24576).contains(&bytes),
            "{bytes} out of CHUNK_BYTES range"
        );
        let first = record["paragraph_first"].as_u64().unwrap();
        let last = record["paragraph_last"].as_u64().unwrap();
        assert!(first <= last);
    }
    // No oversized paragraph in this document, so ranges never overlap
    // and cover every canonical paragraph exactly once, in order.
    assert_eq!(chunks[0]["paragraph_first"], 0);
    for pair in chunks.windows(2) {
        assert_eq!(
            pair[1]["paragraph_first"].as_u64().unwrap(),
            pair[0]["paragraph_last"].as_u64().unwrap() + 1,
            "{:?} must pick up exactly where {:?} left off",
            pair[1],
            pair[0]
        );
    }

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// Issue #262, ADR 0003 §7: one `kind: "document"` record per document
/// written, a structured twin of `Run::report`'s human-readable line —
/// `concepts`/`labels` counted separately rather than combined into one
/// "alias(es)" figure.
#[test]
fn diagnostics_writes_a_document_record_whose_counts_match_the_written_batch() {
    let docs = batch_dir("extract-diag-docrec-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "a small b document").unwrap();
    let out = batch_dir("extract-diag-docrec-out");
    let diag_dir = batch_dir("extract-diag-docrec-diag");
    let diag = diag_dir.join("diag.jsonl");

    // Two exact-duplicate triples fold into one association, plus one
    // concept alias — association(s)=1, concepts=1, labels=0,
    // duplicates=1, dropped=0.
    let reply = json!({
        "associations": [
            {"subject": "a", "label": "rel", "object": "b", "weight": 1.0},
            {"subject": "a", "label": "rel", "object": "b", "weight": 1.0}
        ],
        "aliases": [
            {"alias": "X", "canonical": "a", "kind": "concept"}
        ]
    })
    .to_string();
    let (url, _requests) = stub_chat_server(vec![reply]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--diagnostics-out",
            diag.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("1 association(s), 1 alias(es)")
            && stdout.contains("1 duplicate(s) folded"),
        "{stdout}"
    );

    let all = read_diagnostics(&diag);
    let documents: Vec<&Value> = all
        .iter()
        .filter(|record| record["kind"] == "document")
        .collect();
    assert_eq!(documents.len(), 1, "{all:?}");
    assert_eq!(
        all.last().unwrap()["kind"],
        "document",
        "the document record lands only once its document is fully written: {all:?}"
    );
    let record = documents[0];
    assert_eq!(record["source"], doc.to_str().unwrap());
    assert_eq!(record["associations"], 1);
    assert_eq!(record["concepts"], 1);
    assert_eq!(record["labels"], 0);
    assert_eq!(record["questions"], 0);
    assert_eq!(record["duplicates"], 1);
    assert_eq!(record["dropped"], 0);
    assert_eq!(record["removed"], 0);

    let written = stray_batch_files(&out);
    assert_eq!(written.len(), 1, "{written:?}");
    let expected_path = out.join(&written[0]);
    assert_eq!(
        record["batch_path"].as_str().unwrap(),
        expected_path.display().to_string()
    );
    assert!(expected_path.is_file());

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// `--lossy` wins over a conflicting TAGURU_EXTRACT_LOSSY=false, the
/// same flag-over-environment precedence every other control follows.
#[test]
fn extract_lossy_flag_overrides_the_environment_variable() {
    let docs = batch_dir("extract-lossy-override-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-lossy-override-out");

    let bad_reply = json!({
        "associations": [
            {"subject": "a", "label": "rel", "object": "b", "weight": 0}
        ],
        "aliases": []
    })
    .to_string();
    let (url, requests) = stub_chat_server(vec![bad_reply]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_LOSSY", "false"),
        ],
        &["--lossy", "--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("(--lossy)"), "{stdout}");
    assert_eq!(requests.join().unwrap().len(), 1);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

// -- issue #179: durable chunk checkpoints, cooperative stop, and resume ------------------

/// The on-disk path of one document's chunk checkpoint file — the
/// flatten-then-hash-suffix naming scheme `extract.rs`'s private
/// `checkpoint_file_name` uses, replicated here since an integration
/// test only ever sees the compiled binary's filesystem effects. Every
/// source gets the 16-hex-character hash suffix unconditionally
/// (issue #227); the >120-byte truncation of the flattened prefix
/// isn't replicated since every source used in these tests is short.
fn checkpoint_file_path(out: &std::path::Path, source: &str) -> std::path::PathBuf {
    use sha2::{Digest, Sha256};
    let name = source.replace(['/', '\\', ':'], "__");
    let hash: String = Sha256::digest(source.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    out.join(".extract-checkpoints")
        .join(format!("{name}-{}.json", &hash[..16]))
}

/// The number of units recorded in one document's checkpoint file, or
/// 0 if it doesn't exist (never created yet, or already cleared once
/// that document's batch landed).
fn checkpoint_units_count(out: &std::path::Path, source: &str) -> usize {
    let Ok(text) = std::fs::read_to_string(checkpoint_file_path(out, source)) else {
        return 0;
    };
    let value: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    value["units"].as_object().map_or(0, |units| units.len())
}

/// Runs one 2-top-level-chunk document (`multi_chunk_document(9)`, see
/// its chunk math note) where chunk 0 always succeeds and chunk 1 never
/// produces valid JSON — the document fails after chunk 1 exhausts its
/// corrective attempts, but only after chunk 0 already landed
/// durably in the checkpoint file. Confirms the 2-chunk shape via
/// `--dry-run` first rather than assuming it. Returns the document path
/// and its freshly created `--out` directory for the caller's own
/// follow-up run(s).
fn setup_one_checkpointed_chunk_and_one_failure(
    tag: &str,
) -> (std::path::PathBuf, std::path::PathBuf) {
    let docs = batch_dir(&format!("{tag}-docs"));
    let doc = docs.join("a.md");
    std::fs::write(&doc, multi_chunk_document(9)).unwrap();
    let doc_src = doc.to_str().unwrap().to_string();
    let out = batch_dir(&format!("{tag}-out"));

    let (code, dry_stdout, stderr) =
        run_extract(&out, &[], &["--dry-run", "--context", "c", &doc_src]);
    assert_eq!(code, 0, "stdout: {dry_stdout}\nstderr: {stderr}");
    assert_eq!(
        chunk_count_from_dry_run(&dry_stdout),
        2,
        "multi_chunk_document(9) must pack into exactly 2 top-level chunks: {dry_stdout}"
    );

    // Sequential (parallel=1, the default) request order: chunk 0's one
    // successful call, then chunk 1's two corrective attempts (default
    // max_attempts=2), both malformed — never valid JSON, so chunk 1
    // exhausts its attempts and the document fails, but only after
    // chunk 0 already landed in the checkpoint file. `stub_chat_server`
    // (connection-order-keyed) is used deliberately instead of
    // `stub_chat_server_concurrent` (content-parsed "part K of N"
    // index): a corrective retry's last user turn is the corrective ask
    // text, not the original part tag, so the latter can't tell chunk
    // 1's retries apart from chunk 0.
    let chunk0_reply = json!({"associations": [
        {"subject": "S", "label": "rel", "object": "chunk0", "weight": 1.0}
    ]})
    .to_string();
    let (url, requests) = stub_chat_server(vec![
        chunk0_reply,
        "not json".to_string(),
        "not json".to_string(),
    ]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", &doc_src],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(
        requests.join().unwrap().len(),
        3,
        "chunk 0's one call plus chunk 1's two corrective attempts"
    );
    assert!(
        stray_batch_files(&out).is_empty(),
        "a failed document must not leave a batch file behind"
    );
    assert_eq!(
        checkpoint_units_count(&out, &doc_src),
        1,
        "chunk 0's checkpoint must survive chunk 1's failure"
    );

    (doc, out)
}

/// The checkpoint's whole point: a chunk that already succeeded is
/// reused by the very next attempt, even though the two attempts are
/// entirely separate process invocations (no kill involved — the first
/// one just fails outright on chunk 1, the ordinary way).
#[test]
fn checkpoint_reuses_a_completed_chunk_after_a_failed_document_without_recalling_the_model() {
    let (doc, out) = setup_one_checkpointed_chunk_and_one_failure("extract-checkpoint-reuse");
    let doc_src = doc.to_str().unwrap();

    // Only chunk 1 should ever connect — chunk 0 comes from its
    // checkpoint. One reply, so the server thread's join (implicitly,
    // via requests.join() below) proves exactly one connection arrived.
    let reply = json!({"associations": [
        {"subject": "S", "label": "rel", "object": "chunk1", "weight": 1.0}
    ]})
    .to_string();
    let (url, requests) = stub_chat_server(vec![reply]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("1 written"), "{stdout}");
    assert_eq!(
        requests.join().unwrap().len(),
        1,
        "chunk 0 must be served from its checkpoint, not re-requested"
    );
    assert_eq!(
        checkpoint_units_count(&out, doc_src),
        0,
        "the checkpoint file must be cleared once the batch lands"
    );

    let _ = std::fs::remove_dir_all(doc.parent().unwrap());
    let _ = std::fs::remove_dir_all(&out);
}

/// `--dry-run` (issue #179's requirement) reports a nonzero reusable
/// count from a prior incomplete run's checkpoint, without calling the
/// model.
#[test]
fn dry_run_reports_a_reusable_count_from_a_prior_incomplete_run() {
    let (doc, out) = setup_one_checkpointed_chunk_and_one_failure("extract-checkpoint-dryrun");
    let doc_src = doc.to_str().unwrap();

    // The checkpoint's fingerprint recorded "stub-model" (the setup
    // run's TAGURU_EXTRACT_MODEL) — matching it here is required for
    // the checkpoint to be considered compatible at all, exactly like
    // the existing manifest skip-check.
    let (code, stdout, stderr) = run_extract(
        &out,
        &[("TAGURU_EXTRACT_MODEL", "stub-model")],
        &["--dry-run", "--context", "c", doc_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("2 chunk(s), 1 reusable from checkpoint"),
        "{stdout}"
    );

    let _ = std::fs::remove_dir_all(doc.parent().unwrap());
    let _ = std::fs::remove_dir_all(&out);
}

/// `--force` extends its existing "redo this document" meaning one
/// level deeper: every chunk is re-asked, even one whose checkpoint
/// would otherwise be perfectly reusable.
#[test]
fn force_ignores_existing_checkpoints_and_recalls_every_chunk() {
    let (doc, out) = setup_one_checkpointed_chunk_and_one_failure("extract-checkpoint-force");
    let doc_src = doc.to_str().unwrap();

    // Both chunks must connect this time, in order — --force discards
    // chunk 0's checkpoint too.
    let (url, requests) = stub_chat_server(vec![
        json!({"associations": [
            {"subject": "S", "label": "rel", "object": "chunk0", "weight": 1.0}
        ]})
        .to_string(),
        json!({"associations": [
            {"subject": "S", "label": "rel", "object": "chunk1", "weight": 1.0}
        ]})
        .to_string(),
    ]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--force", "--context", "c", doc_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(
        requests.join().unwrap().len(),
        2,
        "--force must re-call every chunk despite an existing, fingerprint-compatible checkpoint"
    );

    let _ = std::fs::remove_dir_all(doc.parent().unwrap());
    let _ = std::fs::remove_dir_all(&out);
}

/// `--resume-from read/plan/steer` (#822) extends `--force`'s "redo
/// this document" reach the same way: every chunk is re-asked, even
/// one whose checkpoint would otherwise be perfectly reusable — a
/// deliberate resume ask must not be silently answered by stale
/// checkpoint content either.
#[test]
fn resume_from_read_plan_steer_also_ignore_existing_checkpoints() {
    for step in ["read", "plan", "steer"] {
        let (doc, out) = setup_one_checkpointed_chunk_and_one_failure(&format!(
            "extract-resume-checkpoint-{step}"
        ));
        let doc_src = doc.to_str().unwrap();

        let (url, requests) = stub_chat_server(vec![
            json!({"associations": [
                {"subject": "S", "label": "rel", "object": "chunk0", "weight": 1.0}
            ]})
            .to_string(),
            json!({"associations": [
                {"subject": "S", "label": "rel", "object": "chunk1", "weight": 1.0}
            ]})
            .to_string(),
        ]);
        let (code, stdout, stderr) = run_extract(
            &out,
            &[
                ("TAGURU_EXTRACT_URL", url.as_str()),
                ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ],
            &["--context", "c", "--resume-from", step, doc_src],
        );
        assert_eq!(code, 0, "{step}: stdout: {stdout}\nstderr: {stderr}");
        assert_eq!(
            requests.join().unwrap().len(),
            2,
            "{step}: --resume-from {step} must re-call every chunk despite an existing, \
             fingerprint-compatible checkpoint"
        );

        let _ = std::fs::remove_dir_all(doc.parent().unwrap());
        let _ = std::fs::remove_dir_all(&out);
    }
}

/// Changing a compute-shaping setting (here `--fact-budget`) between
/// attempts must invalidate chunk 0's checkpoint even though the
/// document's own content is byte-for-byte unchanged — never a silent
/// reuse of an output computed under different rules.
#[test]
fn a_changed_fact_budget_invalidates_checkpoints_even_though_content_is_unchanged() {
    let (doc, out) = setup_one_checkpointed_chunk_and_one_failure("extract-checkpoint-factbudget");
    let doc_src = doc.to_str().unwrap();

    // Both chunks must connect this time, in order — the changed
    // --fact-budget invalidates chunk 0's checkpoint too.
    let (url, requests) = stub_chat_server(vec![
        json!({"associations": [
            {"subject": "S", "label": "rel", "object": "chunk0", "weight": 1.0}
        ]})
        .to_string(),
        json!({"associations": [
            {"subject": "S", "label": "rel", "object": "chunk1", "weight": 1.0}
        ]})
        .to_string(),
    ]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--fact-budget", "3", "--context", "c", doc_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(
        requests.join().unwrap().len(),
        2,
        "a changed --fact-budget must invalidate chunk 0's checkpoint, not silently reuse it"
    );

    let _ = std::fs::remove_dir_all(doc.parent().unwrap());
    let _ = std::fs::remove_dir_all(&out);
}

/// Issue #179's core durability claim, the hard way: a chunk that
/// already landed in the checkpoint file survives the process being
/// killed outright (SIGKILL — no cooperative stop involved, unlike the
/// SIGINT test below), and a rerun does not re-ask the model for it.
#[test]
fn checkpoint_resumes_a_killed_multi_chunk_document_without_recalling_completed_chunks() {
    let docs = batch_dir("extract-checkpoint-kill-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, multi_chunk_document(9)).unwrap();
    let doc_src = doc.to_str().unwrap().to_string();
    let out = batch_dir("extract-checkpoint-kill-out");

    let response0 = chat_ok(
        &json!({"associations": [
            {"subject": "S", "label": "rel", "object": "chunk0", "weight": 1.0}
        ]})
        .to_string(),
    );
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        use std::io::Write;
        let mut held = Vec::new();
        for (index, stream) in listener.incoming().enumerate() {
            let Ok(mut stream) = stream else { continue };
            if index == 0 {
                let _ = read_http_request(&mut stream);
                let _ = stream.write_all(response0.as_bytes());
            } else {
                held.push(stream);
            }
        }
    });

    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    scrub_extract_env(&mut command)
        .arg("extract")
        .env("TAGURU_EXTRACT_URL", &url)
        .env("TAGURU_EXTRACT_MODEL", "stub-model")
        .args(["--out", out.to_str().unwrap(), "--context", "c"])
        .arg(&doc)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut child = command.spawn().expect("extract must spawn");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut landed = false;
    while std::time::Instant::now() < deadline {
        if checkpoint_units_count(&out, &doc_src) >= 1 {
            landed = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let _ = child.kill();
    let _ = child.wait();
    assert!(landed, "chunk 0's checkpoint never landed before the kill");

    let (url, captured) = stub_chat_server_concurrent(|index, _attempt| {
        chat_ok(
            &json!({"associations": [
                {"subject": "S", "label": "rel", "object": format!("chunk{index}"), "weight": 1.0}
            ]})
            .to_string(),
        )
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", &doc_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("1 written"), "{stdout}");
    assert_eq!(
        captured.lock().unwrap().len(),
        1,
        "chunk 0 must not be re-requested after resuming from a killed run's checkpoint"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// Issue #179's amendment: ADR 0001 §7's split rung (Option D) can
/// change a chunk's boundaries mid-run. A sub-piece that already
/// succeeded must be reused on resume even though the ORIGINAL
/// (pre-split) piece never itself succeeds and so is never itself
/// cacheable — only a per-unit content hash, never `chunk_index` alone,
/// can tell the two apart correctly.
#[test]
fn checkpoint_resumes_the_not_yet_completed_sub_piece_after_a_kill_mid_split() {
    let docs = batch_dir("extract-checkpoint-split-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, format!("{}\n\n{}", "a".repeat(600), "b".repeat(600))).unwrap();
    let doc_src = doc.to_str().unwrap().to_string();
    let out = batch_dir("extract-checkpoint-split-out");

    let length_reply = chat_ok_with_finish_reason("truncated garbage", "length");
    let sub_a_reply = chat_ok(&json!({"associations": []}).to_string());
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        use std::io::Write;
        let mut held = Vec::new();
        for (index, stream) in listener.incoming().enumerate() {
            let Ok(mut stream) = stream else { continue };
            match index {
                0 | 1 => {
                    let _ = read_http_request(&mut stream);
                    let _ = stream.write_all(length_reply.as_bytes());
                }
                2 => {
                    let _ = read_http_request(&mut stream);
                    let _ = stream.write_all(sub_a_reply.as_bytes());
                }
                _ => held.push(stream),
            }
        }
    });

    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    scrub_extract_env(&mut command)
        .arg("extract")
        .env("TAGURU_EXTRACT_URL", &url)
        .env("TAGURU_EXTRACT_MODEL", "stub-model")
        .args([
            "--out",
            out.to_str().unwrap(),
            "--context",
            "c",
            "--max-output-tokens",
            "512",
        ])
        .arg(&doc)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let mut child = command.spawn().expect("extract must spawn");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut landed = false;
    while std::time::Instant::now() < deadline {
        if checkpoint_units_count(&out, &doc_src) >= 1 {
            landed = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let _ = child.kill();
    let _ = child.wait();
    assert!(
        landed,
        "the completed sub-piece's checkpoint never landed before the kill"
    );
    assert_eq!(
        checkpoint_units_count(&out, &doc_src),
        1,
        "only the completed sub-piece, never the pre-split piece, should be checkpointed"
    );

    let (url, captured) = stub_chat_server_concurrent(|_index, attempt| {
        if attempt <= 1 {
            chat_ok_with_finish_reason("truncated garbage", "length")
        } else {
            chat_ok(&json!({"associations": []}).to_string())
        }
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", "--max-output-tokens", "512", &doc_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(
        captured.lock().unwrap().len(),
        3,
        "budgeted ask, escalation, and only the not-yet-completed sub-piece — never the \
         already-checkpointed one"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// Blocks until `stderr` prints the line `StopSignal::install()` emits
/// once its background signal handlers are registered, then spawns a
/// thread draining whatever follows so the child never blocks writing
/// to a full stderr pipe (mirrors `support.rs`'s
/// `read_listen_line_and_drain` for the server's own "listening on"
/// line). Replaces a fixed startup sleep: sending a signal before
/// registration completes would hit the process's default disposition
/// (immediate termination) instead of the cooperative stop path,
/// which a fixed margin can only ever guess at under variable CI load.
fn wait_for_stop_signal_handlers(stderr: std::process::ChildStderr) {
    use std::io::{BufRead, BufReader};
    let mut lines = BufReader::new(stderr).lines();
    loop {
        let line = lines
            .next()
            .expect("extract exited before installing its stop signal handlers")
            .expect("extract stderr must be readable");
        if line.contains("stop signal handlers installed") {
            break;
        }
    }
    std::thread::spawn(move || for _ in lines {});
}

/// [`std::process::Child::wait_with_output`], but force-kills the child
/// and panics with a clear message if it hasn't exited within
/// `deadline` — issue #213: an earlier fix attempt here hung silently
/// until the CI job's own (much longer) timeout killed it instead of
/// failing this test directly, which cost a full 15-minute round trip
/// to even notice. Stdout is drained on a background thread the whole
/// time it waits, exactly like `wait_with_output` would, so a chatty
/// child still can't deadlock on a full pipe while this polls.
fn wait_with_deadline(
    mut child: std::process::Child,
    deadline: std::time::Duration,
) -> std::process::Output {
    use std::io::Read;
    let stdout_handle = child.stdout.take().map(|mut pipe| {
        std::thread::spawn(move || {
            let mut buffer = Vec::new();
            let _ = pipe.read_to_end(&mut buffer);
            buffer
        })
    });
    let start = std::time::Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().expect("polling the child must not fail") {
            break status;
        }
        if start.elapsed() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("extract did not exit within {deadline:?} after SIGINT — see issue #213");
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    let stdout = stdout_handle
        .map(|handle| handle.join().unwrap())
        .unwrap_or_default();
    std::process::Output {
        status,
        stdout,
        stderr: Vec::new(),
    }
}

/// Issue #179's cooperative stop: SIGINT lets an in-flight chunk finish
/// (and get checkpointed) before the process exits with code 130 —
/// distinct from a hard failure — and a rerun resumes without
/// re-asking the model for what already landed. Chunk 0's response is
/// deliberately delayed server-side so SIGINT can be sent any time
/// before it lands. That alone doesn't bound when SIGINT is safe to
/// send, though (issue #213): `StopSignal::install()` (and its
/// readiness marker) runs before the file loop even starts, well
/// before chunk 0's request is dispatched — on a loaded CI runner,
/// signal delivery can beat the main thread to its first stop check,
/// interrupting before chunk 0 ever starts. So the mock server also
/// signals over a channel once it has actually received chunk 0's
/// request, and the test waits on that too before sending SIGINT —
/// chunk 0 is then guaranteed in flight (mid-sleep, awaiting its
/// response) no matter how the two processes are scheduled.
///
/// Every wait below carries its own bound (`recv_timeout`, a read
/// timeout on the stream, [`wait_with_deadline`] instead of a bare
/// `wait_with_output`) rather than trusting a fixed sleep: an earlier
/// attempt at this fix closed the original race but then hung for the
/// full CI job timeout instead of failing fast when something
/// unexpected happened — these bounds turn any future surprise into a
/// quick, diagnosable failure instead of a repeat of that.
#[test]
fn cooperative_sigint_stops_between_chunks_and_a_rerun_resumes() {
    use std::time::Duration;

    let docs = batch_dir("extract-checkpoint-sigint-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, multi_chunk_document(9)).unwrap();
    let doc_src = doc.to_str().unwrap().to_string();
    let out = batch_dir("extract-checkpoint-sigint-out");

    let response0 = chat_ok(
        &json!({"associations": [
            {"subject": "S", "label": "rel", "object": "chunk0", "weight": 1.0}
        ]})
        .to_string(),
    );
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (chunk0_received_tx, chunk0_received_rx) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        use std::io::Write;
        let mut held = Vec::new();
        for (index, stream) in listener.incoming().enumerate() {
            let Ok(mut stream) = stream else { continue };
            if index == 0 {
                // A read timeout here means an unexpected connection
                // (anything that isn't a complete HTTP request within
                // 10s) fails this thread's `read_http_request` call
                // loudly instead of blocking it forever — which would
                // otherwise silently starve `chunk0_received_tx` and
                // rely entirely on the receiver's own timeout below.
                let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
                let request = read_http_request(&mut stream);
                assert!(
                    request.is_some(),
                    "chunk 0's connection must carry a complete HTTP request within 10s"
                );
                let _ = chunk0_received_tx.send(());
                std::thread::sleep(Duration::from_millis(4000));
                let _ = stream.write_all(response0.as_bytes());
            } else {
                held.push(stream);
            }
        }
    });

    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    scrub_extract_env(&mut command)
        .arg("extract")
        .env("TAGURU_EXTRACT_URL", &url)
        .env("TAGURU_EXTRACT_MODEL", "stub-model")
        .args(["--out", out.to_str().unwrap(), "--context", "c"])
        .arg(&doc)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().expect("extract must spawn");
    wait_for_stop_signal_handlers(child.stderr.take().unwrap());
    chunk0_received_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("chunk 0's request must reach the mock server before SIGINT is sent");

    let pid = child.id().to_string();
    Command::new("kill")
        .args(["-INT", &pid])
        .status()
        .expect("kill must run");

    let output = wait_with_deadline(child, Duration::from_secs(30));
    assert_eq!(output.status.code(), Some(130), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("stopped early"), "{stdout}");
    assert!(
        stray_batch_files(&out).is_empty(),
        "an interrupted document must not leave a batch file behind"
    );
    assert_eq!(
        checkpoint_units_count(&out, &doc_src),
        1,
        "chunk 0 must be checkpointed before the stop takes effect"
    );

    let (url, captured) = stub_chat_server_concurrent(|index, _attempt| {
        chat_ok(
            &json!({"associations": [
                {"subject": "S", "label": "rel", "object": format!("chunk{index}"), "weight": 1.0}
            ]})
            .to_string(),
        )
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", &doc_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(
        captured.lock().unwrap().len(),
        1,
        "chunk 0 must not be re-requested on resume after the cooperative stop"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// The escape hatch: a SECOND SIGINT forces an immediate exit even
/// while the process is permanently blocked inside a chunk's request
/// (a stub that never answers at all) — mirroring the server's own
/// `shutdown_signal` two-stage semantics exactly.
///
/// Issue #570: the stop-signal-handlers marker (awaited by
/// `wait_for_stop_signal_handlers`) is printed before the file loop
/// even starts — same as issue #213's race in
/// `cooperative_sigint_stops_between_chunks_and_a_rerun_resumes`
/// above. Sending the first SIGINT right after that marker leaves a
/// window where, on a loaded CI runner, the signal can beat the main
/// thread to its first stop check (`extract.rs`'s per-document check,
/// `run.rs`'s post-checkpoint-load check, or its per-chunk check) —
/// the cooperative stop then succeeds on the FIRST signal instead of
/// requiring a second one, so the process exits before the 500ms
/// `try_wait` assertion, which is a flake, not a bug. This is the
/// same fix as #213: the mock listener signals over a channel once it
/// has actually read chunk 0's full HTTP request, and the test waits
/// on that before sending the first SIGINT — chunk 0 is then
/// guaranteed in flight (blocked awaiting a response that never
/// comes) no matter how the two processes are scheduled.
#[test]
fn a_second_sigint_forces_an_immediate_exit_even_while_permanently_blocked() {
    use std::time::Duration;

    let docs = batch_dir("extract-checkpoint-doublesigint-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let doc_src = doc.to_str().unwrap().to_string();
    let out = batch_dir("extract-checkpoint-doublesigint-out");

    // Accepts every connection and never answers any of them — the
    // one (and only) chunk's request blocks forever. The first
    // connection's request is read in full (and only then reported
    // over `chunk0_received_tx`) so the test knows the chunk is
    // actually in flight before it sends SIGINT; every stream is kept
    // open in `held` (never closed, never answered) so ureq has
    // nothing to reconnect over.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (chunk0_received_tx, chunk0_received_rx) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        let mut held = Vec::new();
        let mut first_connection = true;
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            if first_connection {
                first_connection = false;
                let _ = stream.set_read_timeout(Some(Duration::from_secs(10)));
                let request = read_http_request(&mut stream);
                assert!(
                    request.is_some(),
                    "chunk 0's connection must carry a complete HTTP request within 10s"
                );
                let _ = chunk0_received_tx.send(());
            }
            held.push(stream);
        }
    });

    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    scrub_extract_env(&mut command)
        .arg("extract")
        .env("TAGURU_EXTRACT_URL", &url)
        .env("TAGURU_EXTRACT_MODEL", "stub-model")
        .args(["--out", out.to_str().unwrap(), "--context", "c"])
        .arg(&doc)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().expect("extract must spawn");
    wait_for_stop_signal_handlers(child.stderr.take().unwrap());
    chunk0_received_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("chunk 0's request must reach the mock server before SIGINT is sent");

    let pid = child.id().to_string();
    // The first signal only sets the cooperative flag — the process
    // stays blocked inside the one chunk's never-answered request, so
    // it must still be running here.
    Command::new("kill")
        .args(["-INT", &pid])
        .status()
        .expect("kill must run");
    std::thread::sleep(Duration::from_millis(500));
    assert!(
        child.try_wait().unwrap().is_none(),
        "the first SIGINT must not exit the process while a chunk is still in flight"
    );

    Command::new("kill")
        .args(["-INT", &pid])
        .status()
        .expect("kill must run");
    let output = wait_with_deadline(child, Duration::from_secs(30));
    assert_eq!(
        output.status.code(),
        Some(130),
        "extract must exit after the second SIGINT: {output:?}"
    );
    assert_eq!(
        checkpoint_units_count(&out, &doc_src),
        0,
        "the one chunk never completed, so nothing should be checkpointed"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

// =====================================================================
// ADR 0023 (#785): the per-document trace file under
// `--out/.extract-trace/`, joining every batch item to its piece and
// completion.
// =====================================================================

/// Reads `--out/.extract-trace/<batch name>` for the one batch in
/// `out`, as records in file order.
fn read_trace(out: &std::path::Path) -> (String, Vec<Value>) {
    let batches = stray_batch_files(out);
    assert_eq!(batches.len(), 1, "{batches:?}");
    let name = batches[0].to_str().unwrap().to_string();
    let path = out.join(".extract-trace").join(&name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading trace {}: {error}", path.display()));
    let records = text
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    (name, records)
}

/// A two-chunk document: every batch item has an `item` record naming
/// the `piece_id` of the chunk whose answer carried it; every `piece`
/// names the `(run_id, attempt_seq)` the diagnostics sidecar recorded
/// for the same completion; an unsplit chunk's `piece_id` is its
/// `chunk_sha256`; and a triple both chunks answered is attributed to
/// the first (the copy `merge` kept).
#[test]
fn trace_joins_every_batch_item_to_its_piece_and_the_sidecar_attempt() {
    let docs = batch_dir("extract-trace-join-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, multi_chunk_document(9)).unwrap();
    let out = batch_dir("extract-trace-join-out");
    let diag_dir = batch_dir("extract-trace-join-diag");
    let diag = diag_dir.join("diag.jsonl");

    let (url, _captured) = stub_chat_server_concurrent(|index, _attempt| {
        let object = format!("value-{index}");
        chat_ok(
            &json!({
                "associations": [
                    {"subject": "S", "label": "rel", "object": object, "weight": 1.0, "paragraph": 0},
                    {"subject": "S", "label": "rel", "object": "value-9", "weight": 1.0}
                ],
                "aliases": [{"alias": "s", "canonical": "S", "kind": "concept"}],
                "questions": [{"paragraph": 0, "question": format!("what is value-{index}?")}]
            })
            .to_string(),
        )
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--questions",
            "2",
            "--diagnostics-out",
            diag.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let sidecar = read_diagnostics(&diag);
    assert_eq!(sidecar[0]["kind"], "run", "{sidecar:?}");
    let run_id = sidecar[0]["run_id"].as_str().unwrap().to_string();
    assert_eq!(run_id.len(), 16, "{run_id:?}");
    assert!(run_id.chars().all(|c| c.is_ascii_hexdigit()), "{run_id:?}");

    let (batch_name, trace) = read_trace(&out);
    assert_eq!(trace[0]["kind"], "document", "{trace:?}");
    assert_eq!(trace[0]["run_id"], run_id.as_str());
    assert_eq!(trace[0]["source"], doc.to_str().unwrap());
    assert_eq!(trace[0]["chunk_total"], 2);
    assert_eq!(trace[0]["document_sha256"].as_str().unwrap().len(), 64);
    assert!(
        trace[0]["batch_path"]
            .as_str()
            .unwrap()
            .ends_with(&batch_name),
        "{:?}",
        trace[0]
    );

    let chunks: Vec<&Value> = trace.iter().filter(|r| r["kind"] == "chunk").collect();
    let pieces: Vec<&Value> = trace.iter().filter(|r| r["kind"] == "piece").collect();
    let items: Vec<&Value> = trace.iter().filter(|r| r["kind"] == "item").collect();
    assert_eq!(chunks.len(), 2, "{trace:?}");
    assert_eq!(pieces.len(), 2, "{trace:?}");
    // The diagnostics chunk records and the trace chunk records agree
    // field for field (ADR 0003 §7's provenance, in both places).
    for (index, chunk) in chunks.iter().enumerate() {
        let twin = sidecar
            .iter()
            .find(|r| r["kind"] == "chunk" && r["chunk_index"] == index)
            .unwrap();
        for field in [
            "chunk_sha256",
            "chunk_bytes",
            "paragraph_first",
            "paragraph_last",
            "chunk_total",
        ] {
            assert_eq!(chunk[field], twin[field], "{field}: {chunk:?} vs {twin:?}");
        }
    }
    // Unsplit: piece == chunk, fresh (not reused), and its attempt is
    // exactly the sidecar's stop_valid attempt for that piece.
    for (index, piece) in pieces.iter().enumerate() {
        assert_eq!(piece["chunk_index"], index);
        assert_eq!(piece["piece_id"], chunks[index]["chunk_sha256"]);
        assert_eq!(piece["chunk_sha256"], chunks[index]["chunk_sha256"]);
        assert_eq!(piece["piece_bytes"], chunks[index]["chunk_bytes"]);
        assert_eq!(piece["paragraph_first"], chunks[index]["paragraph_first"]);
        assert_eq!(piece["paragraph_last"], chunks[index]["paragraph_last"]);
        assert_eq!(piece["reused"], false);
        assert_eq!(piece["attempt"]["run_id"], run_id.as_str());
        let seq = piece["attempt"]["attempt_seq"].as_u64().unwrap();
        let attempt = sidecar
            .iter()
            .find(|r| r["kind"] == "attempt" && r["attempt_seq"] == seq)
            .unwrap_or_else(|| panic!("no sidecar attempt {seq}: {sidecar:?}"));
        assert_eq!(attempt["run_id"], run_id.as_str());
        assert_eq!(attempt["piece_id"], piece["piece_id"]);
        assert_eq!(attempt["chunk_index"], index);
        assert_eq!(attempt["state"], "stop_valid");
    }
    // attempt_seq is 1-based and dense over the run's completions.
    let mut seqs: Vec<u64> = sidecar
        .iter()
        .filter(|r| r["kind"] == "attempt")
        .map(|r| r["attempt_seq"].as_u64().unwrap())
        .collect();
    seqs.sort_unstable();
    assert_eq!(seqs, vec![1, 2]);

    // Every batch item has exactly one item record, keyed by content.
    let batch = std::fs::read_to_string(out.join(&batch_name)).unwrap();
    let batch_lines: Vec<Value> = batch
        .lines()
        .skip(1) // header
        .map(|line| serde_json::from_str(line).unwrap())
        .filter(|line: &Value| line.get("passage").is_none())
        .collect();
    assert_eq!(items.len(), batch_lines.len(), "{items:?}\n{batch_lines:?}");
    let piece_of = |item: &Value| -> usize {
        let id = item["piece_id"].as_str().expect("every item names a piece");
        pieces
            .iter()
            .position(|p| p["piece_id"] == id)
            .expect("an item's piece_id names a piece record")
    };
    for line in &batch_lines {
        let item = items
            .iter()
            .find(|item| {
                if line.get("subject").is_some() {
                    item["item"] == "association"
                        && item["subject"] == line["subject"]
                        && item["label"] == line["label"]
                        && item["object"] == line["object"]
                } else if line.get("alias").is_some() {
                    item["item"] == line["kind"] && item["alias"] == line["alias"]
                } else {
                    item["item"] == "question"
                        && item["paragraph"] == line["paragraph"]
                        && item["question"] == line["question"]
                }
            })
            .unwrap_or_else(|| panic!("no item record for {line}: {items:?}"));
        // Each chunk's own object came from that chunk.
        if line["object"] == "value-0" {
            assert_eq!(piece_of(item), 0, "{item:?}");
        }
        if line["object"] == "value-1" {
            assert_eq!(piece_of(item), 1, "{item:?}");
        }
        if line["question"] == "what is value-1?" {
            assert_eq!(piece_of(item), 1, "{item:?}");
        }
    }
    // The triple both chunks answered is attributed to the kept copy —
    // the first output's — and the alias likewise.
    let shared = items
        .iter()
        .find(|item| item["object"] == "value-9")
        .unwrap();
    assert_eq!(piece_of(shared), 0, "{shared:?}");
    let alias = items.iter().find(|item| item["item"] == "concept").unwrap();
    assert_eq!(piece_of(alias), 0, "{alias:?}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// ADR 0023 §3.5: a unit reused from a checkpoint keeps the attempt
/// of the run that produced it — the resumed run's trace marks it
/// `reused: true` with the EARLIER run's id, while the freshly
/// extracted chunk carries the resumed run's own id.
#[test]
fn trace_marks_a_checkpoint_reused_piece_with_the_producing_runs_attempt() {
    // The same shape as `setup_one_checkpointed_chunk_and_one_failure`,
    // with objects the occurrence check (ADR 0013) attests in
    // `multi_chunk_document` ("value-N"), so both items reach the
    // batch and the trace.
    let docs = batch_dir("extract-trace-reuse-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, multi_chunk_document(9)).unwrap();
    let doc_src = doc.to_str().unwrap();
    let out = batch_dir("extract-trace-reuse-out");
    let diag_dir = batch_dir("extract-trace-reuse-diag");
    let diag = diag_dir.join("diag.jsonl");

    let chunk0_reply = json!({"associations": [
        {"subject": "S", "label": "rel", "object": "value-0", "weight": 1.0}
    ]})
    .to_string();
    let (url, requests) = stub_chat_server(vec![
        chunk0_reply,
        "not json".to_string(),
        "not json".to_string(),
    ]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc_src],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(requests.join().unwrap().len(), 3);
    assert_eq!(checkpoint_units_count(&out, doc_src), 1);
    // A failed document writes no trace file — only its attempts log
    // (ADR 0025), which is the directory's one entry here.
    let entries: Vec<String> = std::fs::read_dir(out.join(".extract-trace"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(entries.len(), 1, "{entries:?}");
    assert!(entries[0].ends_with(".attempts.jsonl"), "{entries:?}");

    let reply = json!({"associations": [
        {"subject": "S", "label": "rel", "object": "value-1", "weight": 1.0}
    ]})
    .to_string();
    let (url, requests) = stub_chat_server(vec![reply]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--diagnostics-out",
            diag.to_str().unwrap(),
            doc_src,
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(requests.join().unwrap().len(), 1);

    let sidecar = read_diagnostics(&diag);
    let run_id = sidecar[0]["run_id"].as_str().unwrap();
    let (_, trace) = read_trace(&out);
    let pieces: Vec<&Value> = trace.iter().filter(|r| r["kind"] == "piece").collect();
    assert_eq!(pieces.len(), 2, "{trace:?}");
    let reused = pieces.iter().find(|p| p["chunk_index"] == 0).unwrap();
    let fresh = pieces.iter().find(|p| p["chunk_index"] == 1).unwrap();
    assert_eq!(reused["reused"], true, "{reused:?}");
    assert_ne!(reused["attempt"]["run_id"], run_id, "{reused:?}");
    assert_eq!(reused["attempt"]["attempt_seq"], 1, "{reused:?}");
    assert_eq!(fresh["reused"], false, "{fresh:?}");
    assert_eq!(fresh["attempt"]["run_id"], run_id, "{fresh:?}");
    assert_eq!(fresh["attempt"]["attempt_seq"], 1, "{fresh:?}");
    let items: Vec<&Value> = trace.iter().filter(|r| r["kind"] == "item").collect();
    let chunk0 = items.iter().find(|i| i["object"] == "value-0").unwrap();
    let chunk1 = items.iter().find(|i| i["object"] == "value-1").unwrap();
    assert_eq!(chunk0["piece_id"], reused["piece_id"]);
    assert_eq!(chunk1["piece_id"], fresh["piece_id"]);

    // An unchanged rerun skips the document and leaves its trace as
    // written (the batch's lifecycle, not the run's).
    let before = std::fs::read_to_string(
        out.join(".extract-trace")
            .join(stray_batch_files(&out)[0].to_str().unwrap()),
    )
    .unwrap();
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", "http://127.0.0.1:9"),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("unchanged, skipped"), "{stdout}");
    let after = std::fs::read_to_string(
        out.join(".extract-trace")
            .join(stray_batch_files(&out)[0].to_str().unwrap()),
    )
    .unwrap();
    assert_eq!(before, after);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// #786 / ADR 0023 §3.7: every item the model wrote that the batch
/// does not hold lands in the trace as a `loss` with the original text
/// — a mechanically removed fabrication (against its cited paragraph),
/// a triple both chunks answered (the kept piece named), and a
/// question over the `--questions` cap — and the document record's
/// `removed`/`duplicates`/`dropped` counts are those records' lengths.
#[test]
fn trace_records_every_lost_item_with_its_original_text() {
    let docs = batch_dir("extract-trace-loss-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, multi_chunk_document(9)).unwrap();
    let out = batch_dir("extract-trace-loss-out");
    let diag_dir = batch_dir("extract-trace-loss-diag");
    let diag = diag_dir.join("diag.jsonl");

    let (url, _captured) = stub_chat_server_concurrent(|index, _attempt| {
        chat_ok(
            &json!({
                "associations": [
                    {"subject": "S", "label": "rel", "object": "value-9", "weight": 1.0},
                    {"subject": "ghost", "label": "rel", "object": "value-9", "weight": 1.0,
                     "paragraph": 0}
                ],
                "questions": [
                    {"paragraph": 0, "question": "one?"},
                    {"paragraph": 0, "question": format!("two-{index}?")}
                ]
            })
            .to_string(),
        )
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--questions",
            "1",
            "--diagnostics-out",
            diag.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stderr.contains("removed: chunk 1/2 associations[1]: subject \"ghost\" does not appear"),
        "{stderr}"
    );

    let (_, trace) = read_trace(&out);
    let pieces: Vec<&Value> = trace.iter().filter(|r| r["kind"] == "piece").collect();
    let losses: Vec<&Value> = trace.iter().filter(|r| r["kind"] == "loss").collect();
    let document = read_diagnostics(&diag)
        .into_iter()
        .find(|r| r["kind"] == "document")
        .unwrap();
    let count = |reason: &str| losses.iter().filter(|l| l["reason"] == reason).count();
    assert_eq!(document["removed"], count("removed"), "{losses:?}");
    assert_eq!(document["duplicates"], count("duplicate"), "{losses:?}");
    assert_eq!(document["dropped"], count("dropped"), "{losses:?}");
    assert_eq!(count("removed"), 2, "one fabrication per chunk: {losses:?}");

    // The fabrication cited paragraph 0, so its text is that paragraph
    // — the original, not the `[0] `-labeled rendering.
    let removed: Vec<&&Value> = losses.iter().filter(|l| l["reason"] == "removed").collect();
    for (index, loss) in removed.iter().enumerate() {
        assert_eq!(loss["item"], "association");
        assert_eq!(loss["path"], "associations[1]");
        assert!(
            loss["rule"].as_str().unwrap().contains("\"ghost\""),
            "{loss}"
        );
        assert_eq!(loss["raw"]["subject"], "ghost");
        assert_eq!(loss["paragraph"], 0);
        assert!(
            loss["text"]
                .as_str()
                .unwrap()
                .starts_with("Paragraph 0: s value-"),
            "{loss}"
        );
        assert_eq!(loss["piece_id"], pieces[index]["piece_id"]);
        assert_eq!(loss["attempt"], pieces[index]["attempt"]);
    }
    // The shared triple: chunk 1's copy is the duplicate, chunk 0's
    // is kept; no citation → the piece text.
    let duplicates: Vec<&&Value> = losses
        .iter()
        .filter(|l| l["reason"] == "duplicate" && l["item"] == "association")
        .collect();
    assert_eq!(duplicates.len(), 1, "{losses:?}");
    assert_eq!(duplicates[0]["raw"]["object"], "value-9");
    assert_eq!(duplicates[0]["piece_id"], pieces[1]["piece_id"]);
    assert_eq!(duplicates[0]["kept_piece_id"], pieces[0]["piece_id"]);
    assert_eq!(duplicates[0]["paragraph"], Value::Null);
    let piece_text = duplicates[0]["text"].as_str().unwrap();
    assert_eq!(
        piece_text.len() as u64,
        pieces[1]["piece_bytes"].as_u64().unwrap(),
        "no citation → the whole piece, as sent"
    );
    assert!(piece_text.starts_with('['), "{piece_text:.40}");
    // Questions: "one?" twice (chunk 1's is a duplicate), "two-0?" over
    // the cap of 1, "two-1?" over the cap too.
    let questions: Vec<&&Value> = losses.iter().filter(|l| l["item"] == "question").collect();
    assert_eq!(questions.len(), 3, "{questions:?}");
    assert_eq!(
        questions
            .iter()
            .filter(|l| l["reason"] == "duplicate")
            .count(),
        1
    );
    assert!(
        questions
            .iter()
            .filter(|l| l["reason"] == "dropped")
            .all(|l| l["rule"].as_str().unwrap().contains("--questions cap of 1")),
        "{questions:?}"
    );
    // Every loss is readable: text is never empty.
    assert!(
        losses
            .iter()
            .all(|l| !l["text"].as_str().unwrap().is_empty())
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

// =====================================================================
// ADR 0025 (#788): the per-document attempts log — every completion's
// full prompt and full answer, on by default.
// =====================================================================

/// Reads `--out/.extract-trace/<batch stem>.attempts.jsonl` for the one
/// document whose attempts log exists there.
fn read_attempts_log(out: &std::path::Path) -> Vec<Value> {
    let dir = out.join(".extract-trace");
    let mut logs: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.to_string_lossy().ends_with(".attempts.jsonl"))
        .collect();
    assert_eq!(logs.len(), 1, "{logs:?}");
    let text = std::fs::read_to_string(logs.remove(0)).unwrap();
    text.lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

/// A corrective retry: the log holds the document record, the system
/// prompt once (by hash), and every completion's full conversation and
/// full answer — the retry's replayed bad answer and corrective ask
/// included — joinable to the sidecar by `(run_id, attempt_seq)`, with
/// no environment variable set.
#[test]
fn attempts_log_keeps_every_completions_full_prompt_and_answer() {
    let docs = batch_dir("extract-attempts-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "alpha relates to beta").unwrap();
    let out = batch_dir("extract-attempts-out");
    let diag_dir = batch_dir("extract-attempts-diag");
    let diag = diag_dir.join("diag.jsonl");

    let good = json!({"associations": [
        {"subject": "alpha", "label": "relates to", "object": "beta", "weight": 1.0}
    ]})
    .to_string();
    let (url, requests) = stub_chat_server(vec!["not json at all".to_string(), good.clone()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--diagnostics-out",
            diag.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let sent = requests.join().unwrap();
    assert_eq!(sent.len(), 2);

    let (batch_name, _) = read_trace(&out);
    let log_path = out.join(".extract-trace").join(format!(
        "{}.attempts.jsonl",
        batch_name.trim_end_matches(".jsonl")
    ));
    assert!(log_path.is_file(), "{}", log_path.display());
    let records = read_attempts_log(&out);
    let kinds: Vec<&str> = records
        .iter()
        .map(|r| r["kind"].as_str().unwrap())
        .collect();
    assert_eq!(
        kinds,
        ["document", "settings", "system", "attempt", "attempt"],
        "{records:?}"
    );

    let sidecar = read_diagnostics(&diag);
    let run_id = sidecar[0]["run_id"].as_str().unwrap();
    assert_eq!(records[0]["run_id"], run_id);
    assert_eq!(records[0]["source"], doc.to_str().unwrap());
    assert_eq!(records[0]["resumed"], false);
    assert_eq!(records[0]["document_sha256"].as_str().unwrap().len(), 64);

    // ADR 0031 §3.2/§3.9: one settings record right after `document`,
    // a diagnostic snapshot of this run's compute inputs.
    let settings = &records[1];
    assert_eq!(settings["model"], "stub-model");
    assert_eq!(settings["prompt_version"], 3);
    assert_eq!(settings["questions_n"], 0);
    assert_eq!(settings["fact_budget"], 0);
    assert_eq!(settings["structured_output"], "");
    assert_eq!(settings["max_output_tokens"], 0);
    assert_eq!(settings["chunk_bytes"], "");
    assert_eq!(settings["lossy"], false);
    assert_eq!(settings["schema_digest"], "");
    assert_eq!(settings["candidates"], "");
    assert_eq!(settings["vocabulary_digest"], "");
    // No `--structured-output` engaged: no ladder, so no rung to name.
    assert!(settings.get("rung").is_none(), "{settings:?}");

    let system = &records[2];
    let system_text = system["content"].as_str().unwrap();
    assert!(system_text.contains("associations"), "{system_text:.80}");
    assert_eq!(system["bytes"], system_text.len());
    let sha = system["sha256"].as_str().unwrap();
    assert_eq!(sha.len(), 64);

    // The first attempt: base conversation, malformed answer in full.
    let first = &records[3];
    assert_eq!(first["attempt_seq"], 1);
    assert_eq!(first["run_id"], run_id);
    assert_eq!(first["stage"], "item");
    assert_eq!(first["attempt"], 1);
    assert_eq!(first["state"], "stop_malformed");
    assert_eq!(first["answer"], "not json at all");
    assert!(
        first.get("corrects").is_none(),
        "a base attempt corrects nothing"
    );
    let turns = first["messages"].as_array().unwrap();
    assert_eq!(turns.len(), 2, "{turns:?}");
    assert_eq!(turns[0]["role"], "system");
    assert_eq!(turns[0]["system_sha256"], sha);
    assert!(turns[0].get("content").is_none(), "system rides by hash");
    assert_eq!(turns[1]["role"], "user");
    let user = turns[1]["content"].as_str().unwrap();
    assert!(user.ends_with("[0] alpha relates to beta"), "{user:?}");
    assert!(!first["parse_error"].is_null());
    // What the log says was sent IS what the stub received.
    // `stub_chat_server` captures "{headers}\n{body}" — the body is the
    // last line.
    let sent_first: Value = serde_json::from_str(sent[0].lines().last().unwrap()).unwrap();
    assert_eq!(sent_first["messages"][1]["content"], user);
    assert_eq!(
        sha256_hex_of(sent_first["messages"][0]["content"].as_str().unwrap()),
        sha
    );

    // The corrective attempt: the replayed bad answer and the ask, in
    // full; the accepted answer in full; joinable to the sidecar.
    let second = &records[4];
    assert_eq!(second["attempt_seq"], 2);
    assert_eq!(second["attempt"], 2);
    assert_eq!(second["state"], "stop_valid");
    assert_eq!(second["answer"], good);
    // ADR 0028: the corrective attempt names the attempt it corrects —
    // the tuple (issue → ask → answer → adoption) joins on this.
    assert_eq!(second["corrects"]["run_id"], run_id);
    assert_eq!(second["corrects"]["attempt_seq"], 1);
    let turns = second["messages"].as_array().unwrap();
    assert_eq!(turns.len(), 4, "{turns:?}");
    assert_eq!(turns[2]["role"], "assistant");
    assert_eq!(turns[2]["content"], "not json at all");
    assert_eq!(turns[3]["role"], "user");
    assert!(
        turns[3]["content"].as_str().unwrap().contains("JSON"),
        "{}",
        turns[3]
    );
    assert_eq!(second["piece_id"], first["piece_id"]);
    let sidecar_second = sidecar
        .iter()
        .find(|r| r["kind"] == "attempt" && r["attempt_seq"] == 2)
        .unwrap();
    assert_eq!(sidecar_second["piece_id"], second["piece_id"]);
    assert_eq!(sidecar_second["state"], "stop_valid");
    assert_eq!(sidecar_second["corrects"]["attempt_seq"], 1);
    assert_eq!(sidecar_second["corrects"]["run_id"], run_id);
    // The sidecar itself still carries no raw text without the opt-in.
    assert!(sidecar_second.get("response_text").is_none());

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_dir_all(&diag_dir);
}

/// Mirrors `crate::extract::sha256_hex` without depending on the
/// library crate from an integration test — same pattern as
/// evaluate.rs's thresholds digest.
fn sha256_hex_of(text: &str) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write;
    Sha256::digest(text.as_bytes())
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

/// A document that fails keeps its attempts log (that is what the log
/// is for); the resumed run appends to it — `resumed: true` — rather
/// than truncating the earlier run's attempts, so the file spans the
/// runs that built the batch, exactly as the checkpoint does.
#[test]
fn attempts_log_survives_a_failure_and_is_appended_to_on_resume() {
    let docs = batch_dir("extract-attempts-resume-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, multi_chunk_document(9)).unwrap();
    let doc_src = doc.to_str().unwrap();
    let out = batch_dir("extract-attempts-resume-out");

    let chunk0 = json!({"associations": [
        {"subject": "S", "label": "rel", "object": "value-0", "weight": 1.0}
    ]})
    .to_string();
    let (url, _) = stub_chat_server(vec![chunk0, "not json".to_string(), "not json".to_string()]);
    let (code, _, _) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc_src],
    );
    assert_eq!(code, 1);
    let after_failure = read_attempts_log(&out);
    let kinds: Vec<&str> = after_failure
        .iter()
        .map(|r| r["kind"].as_str().unwrap())
        .collect();
    assert_eq!(
        kinds,
        [
            "document", "settings", "system", "attempt", "attempt", "attempt"
        ],
        "{after_failure:?}"
    );
    let first_run = after_failure[0]["run_id"].as_str().unwrap().to_string();
    assert!(
        after_failure[4..]
            .iter()
            .all(|r| r["state"] == "stop_malformed"),
        "{after_failure:?}"
    );

    let chunk1 = json!({"associations": [
        {"subject": "S", "label": "rel", "object": "value-1", "weight": 1.0}
    ]})
    .to_string();
    let (url, _) = stub_chat_server(vec![chunk1]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let after_resume = read_attempts_log(&out);
    let kinds: Vec<&str> = after_resume
        .iter()
        .map(|r| r["kind"].as_str().unwrap())
        .collect();
    assert_eq!(
        kinds,
        [
            "document", "settings", "system", "attempt", "attempt", "attempt", "document",
            "settings", "system", "attempt"
        ],
        "{after_resume:?}"
    );
    assert_eq!(after_resume[6]["resumed"], true);
    assert_ne!(after_resume[6]["run_id"], first_run.as_str());
    assert_eq!(after_resume[9]["run_id"], after_resume[6]["run_id"]);
    assert_eq!(after_resume[9]["attempt_seq"], 1);
    assert_eq!(after_resume[9]["chunk_index"], 1);
    // The trace's reused piece names the first run's attempt 1 — which
    // the log still holds in full.
    let (_, trace) = read_trace(&out);
    let reused = trace
        .iter()
        .find(|r| r["kind"] == "piece" && r["reused"] == true)
        .unwrap();
    assert_eq!(reused["attempt"]["run_id"], first_run.as_str());
    assert_eq!(reused["attempt"]["attempt_seq"], 1);
    assert_eq!(after_resume[3]["run_id"], first_run.as_str());
    assert_eq!(after_resume[3]["attempt_seq"], 1);
    assert_eq!(after_resume[3]["piece_id"], reused["piece_id"]);

    // A re-extraction from scratch (--force, no checkpoint) truncates.
    let chunk_any = json!({"associations": []}).to_string();
    let (url, _) = stub_chat_server_concurrent(move |_, _| chat_ok(&chunk_any));
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", "--force", doc_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let after_force = read_attempts_log(&out);
    assert_eq!(after_force[0]["resumed"], false);
    assert_ne!(after_force[0]["run_id"], first_run.as_str());
    assert_eq!(
        after_force
            .iter()
            .filter(|r| r["kind"] == "document")
            .count(),
        1,
        "{after_force:?}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// `TAGURU_EXTRACT_TRACE_ATTEMPTS=off` is the opt-out: no attempts log,
/// the trace file and the batch unchanged.
#[test]
fn attempts_log_can_be_switched_off() {
    let docs = batch_dir("extract-attempts-off-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "alpha relates to beta").unwrap();
    let out = batch_dir("extract-attempts-off-out");
    let (url, _) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_TRACE_ATTEMPTS", "off"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let (_, trace) = read_trace(&out);
    assert_eq!(trace[0]["kind"], "document");
    let logs = std::fs::read_dir(out.join(".extract-trace"))
        .unwrap()
        .filter(|entry| {
            entry
                .as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".attempts.jsonl")
        })
        .count();
    assert_eq!(logs, 0);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// #789 / ADR 0027: the trace's `steering` record shows exactly what
/// taguru put in the prompt — the second document of a run carries the
/// first document's labels with their counts (the #759 amplification
/// path, now auditable), and `--candidates` lists the offered names.
#[test]
fn trace_steering_record_carries_candidates_and_reuse_vocabulary() {
    let docs = batch_dir("extract-steering-docs");
    let first = docs.join("a.md");
    let second = docs.join("b.md");
    std::fs::write(&first, "alpha relates to beta").unwrap();
    std::fs::write(&second, "gamma relates to delta").unwrap();
    let out = batch_dir("extract-steering-out");

    let (url, _captured) = stub_chat_server_concurrent(|_, _| {
        chat_ok(
            &json!({"associations": [
                {"subject": "alpha", "label": "relates to", "object": "beta", "weight": 1.0},
                {"subject": "gamma", "label": "relates to", "object": "delta", "weight": 1.0}
            ]})
            .to_string(),
        )
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--candidates",
            first.to_str().unwrap(),
            second.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let steering_of = |name: &str| -> Value {
        let batch = stray_batch_files(&out)
            .into_iter()
            .map(|entry| entry.to_string_lossy().into_owned())
            .find(|file| file.contains(name))
            .unwrap();
        std::fs::read_to_string(out.join(".extract-trace").join(&batch))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .find(|record| record["kind"] == "steering")
            .unwrap()
    };
    // Documents run in sorted order: a.md first, with an empty
    // vocabulary; b.md second, offered a.md's labels with counts.
    let first_steering = steering_of("a.md");
    assert_eq!(first_steering["chunk_index"], Value::Null);
    assert_eq!(first_steering["vocabulary"], json!([]));
    assert_eq!(first_steering["context_names"], json!([]));
    assert_eq!(first_steering["schema"], Value::Null);
    let candidates: Vec<&str> = first_steering["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect();
    assert!(candidates.contains(&"alpha"), "{candidates:?}");
    assert!(candidates.contains(&"beta"), "{candidates:?}");
    assert!(!candidates.contains(&"gamma"), "a.md's own names only");

    let second_steering = steering_of("b.md");
    // a.md kept one association (the gamma one was mechanically
    // removed — its names never occur in a.md) — the reuse list b.md
    // saw says so, count and all.
    assert_eq!(
        second_steering["vocabulary"],
        json!([{"label": "relates to", "count": 1}])
    );
    let candidates: Vec<&str> = second_steering["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect();
    assert!(candidates.contains(&"gamma"), "{candidates:?}");
    assert!(!candidates.contains(&"alpha"), "b.md's own names only");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// ADR 0027's `null` contract, regression for the empty-schema edge: a
/// valid installed schema with no types and no constrained relations
/// prompts no schema block at all, so the steering record says
/// `schema: null` — not `{"types":[],"constrained_relations":[]}` —
/// while a schema that does prompt a block is recorded with its lists.
#[test]
fn trace_steering_schema_is_null_exactly_when_no_schema_block_was_prompted() {
    let docs = batch_dir("extract-steering-schema-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let empty_schema = docs.join("empty.schema.json");
    std::fs::write(
        &empty_schema,
        json!({
            "schema": 1,
            "mode": "warn",
            "closed_labels": false,
            "types": {},
            "relations": {"述べる": {"domain": [], "range": []}}
        })
        .to_string(),
    )
    .unwrap();
    let out = batch_dir("extract-steering-schema-out");

    let (url, _requests) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--schema",
            empty_schema.to_str().unwrap(),
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let (_, trace) = read_trace(&out);
    let steering = trace.iter().find(|r| r["kind"] == "steering").unwrap();
    assert_eq!(steering["schema"], Value::Null, "{steering}");

    // The types-only control: one list empty, the other not — the
    // block IS prompted (its type half), so the record carries the
    // lists; this is the case that separates "both empty" from
    // "either empty".
    let full_schema = docs.join("full.schema.json");
    std::fs::write(
        &full_schema,
        json!({
            "schema": 1,
            "mode": "warn",
            "closed_labels": false,
            "types": {"Brewery": {"is_a": []}},
            "relations": {"述べる": {"domain": [], "range": []}}
        })
        .to_string(),
    )
    .unwrap();
    let (url, _requests) = stub_chat_server(vec![json!({"associations": []}).to_string()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--schema",
            full_schema.to_str().unwrap(),
            "--force",
            doc.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let (_, trace) = read_trace(&out);
    let steering = trace.iter().find(|r| r["kind"] == "steering").unwrap();
    assert_eq!(
        steering["schema"]["types"],
        json!(["Brewery"]),
        "{steering}"
    );
    assert_eq!(
        steering["schema"]["constrained_relations"],
        json!([]),
        "{steering}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// #792: `scripts/extract_metrics.py` aggregates the trace and
/// attempts-log records this crate writes — running it against a REAL
/// extract run pins the two sides together, so a record-shape change
/// that breaks the aggregation fails here, not in the field. (The
/// script's own arithmetic is covered by its `--self-test`, also run
/// here.)
#[test]
fn extract_metrics_script_aggregates_a_real_run() {
    let script =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/extract_metrics.py");
    let self_test = std::process::Command::new("python3")
        .arg(&script)
        .arg("--self-test")
        .output()
        .expect("python3 must be available");
    assert!(
        self_test.status.success(),
        "{}{}",
        String::from_utf8_lossy(&self_test.stdout),
        String::from_utf8_lossy(&self_test.stderr)
    );

    let docs = batch_dir("extract-metrics-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, multi_chunk_document(9)).unwrap();
    let out = batch_dir("extract-metrics-out");
    let (url, _captured) = stub_chat_server_concurrent(|index, attempt| {
        if index == 0 && attempt == 0 {
            // One corrective round, with usage: this attempt fails, so
            // its tokens land in the lost-cost buckets.
            chat_ok_with_usage("not json", 40, 7)
        } else {
            chat_ok_with_usage(
                &json!({"associations": [
                    {"subject": "S", "label": "rel", "object": format!("value-{index}"),
                     "weight": 1.0, "paragraph": 0},
                    {"subject": "ghost", "label": "rel", "object": "value-9", "weight": 1.0}
                ]})
                .to_string(),
                100,
                50,
            )
        }
    });
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let ledger = docs.join("ledger.json");
    std::fs::write(
        &ledger,
        json!({"sources": {doc.to_str().unwrap(): {"context": "ch1", "groups": ["book"]}}})
            .to_string(),
    )
    .unwrap();
    let report_path = docs.join("report.json");
    let run = std::process::Command::new("python3")
        .arg(&script)
        .arg(&out)
        .args(["--ledger", ledger.to_str().unwrap()])
        .args(["--json", report_path.to_str().unwrap()])
        .args(["--price-in", "100", "--price-out", "200"])
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let report: Value =
        serde_json::from_str(&std::fs::read_to_string(&report_path).unwrap()).unwrap();
    let metrics = &report["run"];
    assert_eq!(metrics["documents"], 1);
    // 2 chunks kept one "S rel value-N" each; the shared "ghost"
    // association was removed per chunk (fabricated subject) and the
    // duplicate of chunk 1's copy... ghost is removed mechanically in
    // both chunks (2 removed losses), the two value-N facts survive.
    assert_eq!(metrics["loss"]["association"]["kept"], 2, "{metrics}");
    assert_eq!(
        metrics["loss"]["association"]["by_reason"]["removed"], 2,
        "{metrics}"
    );
    assert_eq!(metrics["corrections"]["attempted"], 1);
    assert_eq!(metrics["corrections"]["success_rate"], 1.0);
    assert_eq!(metrics["attempts"]["total"], 3, "base+corrective+chunk2");
    assert!(metrics["coverage"]["covered_rate"].as_f64().unwrap() > 0.0);
    // The stub's usage sums: 40+100+100 in, 7+50+50 out; the malformed
    // attempt's 40/7 are the lost share; money at 100/200 per 1M.
    assert_eq!(metrics["cost"]["input_tokens"], 240, "{metrics}");
    assert_eq!(metrics["cost"]["output_tokens"], 107, "{metrics}");
    assert_eq!(metrics["cost"]["lost_input_tokens"], 40, "{metrics}");
    assert_eq!(metrics["cost"]["lost_output_tokens"], 7, "{metrics}");
    assert_eq!(metrics["cost"]["money"], 0.0454, "{metrics}");
    assert_eq!(report["contexts"]["ch1"]["documents"], 1);
    assert_eq!(report["groups"]["book"]["documents"], 1);

    // Compare mode against itself: everything unchanged.
    let compared = std::process::Command::new("python3")
        .arg(&script)
        .arg(&out)
        .args(["--compare", report_path.to_str().unwrap()])
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&compared.stdout);
    assert!(compared.status.success(), "{text}");
    assert!(text.contains("Compared to baseline"), "{text}");
    assert!(text.contains("| assoc loss | 0 | 0 | 1 |"), "{text}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// #793: `taguru anchoring` judges a real extract run's batches — the
/// strict and alias-group anchoring rates and locator validity — and
/// `scripts/extract_metrics.py --anchoring` rolls the JSON into its
/// tables. The alias-dependent case is exercised end to end: the model
/// asserts a fact under a spelling the passage never uses, anchored
/// only through the alias it also emitted.
#[test]
fn anchoring_command_rates_a_real_run_and_the_script_folds_it_in() {
    let docs = batch_dir("extract-anchoring-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "青嶺酒造の杜氏は高瀬。\n\n蔵は山にある。").unwrap();
    let out = batch_dir("extract-anchoring-out");
    let reply = json!({
        "associations": [
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0,
             "paragraph": 0}
        ]
    })
    .to_string();
    let (url, _requests) = stub_chat_server(vec![reply]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc.to_str().unwrap()],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    // A second, hand-written batch (anchoring parses, never imports):
    // its subject appears in the passage only under the alias's
    // canonical spelling — anchored with aliases, not strictly. This
    // is 0.9.3-shaped input too: no trace beside it.
    std::fs::write(
        out.join("b.jsonl"),
        "{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"b.md\"}\n\
         {\"passage\":\"青嶺酒造の杜氏は高瀬。\\n\\n蔵は山にある。\"}\n\
         {\"subject\":\"あおみね\",\"label\":\"所在\",\"object\":\"山\",\"weight\":1.0}\n\
         {\"alias\":\"あおみね\",\"canonical\":\"青嶺酒造\",\"kind\":\"concept\"}\n",
    )
    .unwrap();

    let report_path = docs.join("anchoring.json");
    let output = Command::new(env!("CARGO_BIN_EXE_taguru"))
        .args(["anchoring", out.to_str().unwrap()])
        .args(["--json", report_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let table = String::from_utf8_lossy(&output.stdout);
    assert!(
        !table.contains("skipped"),
        "nothing was skipped, so no skipped line: {table}"
    );
    let report: Value =
        serde_json::from_str(&std::fs::read_to_string(&report_path).unwrap()).unwrap();
    let totals = &report["totals"];
    assert_eq!(totals["associations"], 2, "{report}");
    assert_eq!(totals["anchored_strict"], 1, "あおみね is not in the text");
    assert_eq!(
        totals["anchored_with_aliases"], 2,
        "…but its alias's canonical is"
    );
    assert_eq!(totals["rate_strict"], 0.5);
    assert_eq!(totals["rate_with_aliases"], 1.0);
    assert_eq!(totals["cited"], 1, "only a.md's association cites");
    assert_eq!(totals["locator_valid"], 1);
    assert_eq!(
        report["documents"]["b.md"]["anchored_strict"], 0,
        "{report}"
    );
    assert_eq!(report["documents"]["b.md"]["anchored_with_aliases"], 1);
    let a_key = doc.to_str().unwrap();
    assert_eq!(report["documents"][a_key]["context"], "c");

    // The aggregation script folds the matched document in and warns
    // about the trace-less one instead of inventing a row.
    let script =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/extract_metrics.py");
    let folded = Command::new("python3")
        .arg(&script)
        .arg(&out)
        .args(["--anchoring", report_path.to_str().unwrap()])
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&folded.stdout);
    let warnings = String::from_utf8_lossy(&folded.stderr);
    assert!(folded.status.success(), "{text}{warnings}");
    assert!(text.contains("## Anchoring"), "{text}");
    assert!(
        text.contains("| run | 1 | 1.000 | 1.000 | 0.000 | 1.000 |"),
        "{text}"
    );
    assert!(warnings.contains("'b.md'"), "{warnings}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// The anchoring CLI's edges: a usage error exits 2 with the unknown
/// argument named; `--vocabulary` context aliases actually widen the
/// alias groups (without the flag the same association is unanchored);
/// a `--no-passage`-shaped batch is skipped and counted; the table
/// prints its TOTAL row.
#[test]
fn anchoring_cli_usage_vocabulary_and_skip_edges() {
    let dir = batch_dir("extract-anchoring-cli-docs");
    // Usage error: exit 2, the argument named, usage shown.
    let bogus = Command::new(env!("CARGO_BIN_EXE_taguru"))
        .args(["anchoring", "--bogus"])
        .output()
        .unwrap();
    assert_eq!(bogus.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&bogus.stderr);
    assert!(stderr.contains("unknown argument '--bogus'"), "{stderr}");
    assert!(stderr.contains("usage: taguru anchoring"), "{stderr}");

    // One judgeable batch whose subject anchors only through a
    // CONTEXT alias, one passage-less batch (skipped, counted).
    std::fs::write(
        dir.join("c.jsonl"),
        "{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"c.md\"}\n\
         {\"passage\":\"青嶺酒造の杜氏は高瀬。\"}\n\
         {\"subject\":\"あおみね\",\"label\":\"杜氏\",\"object\":\"高瀬\",\"weight\":1.0}\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("nopassage.jsonl"),
        "{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"n.md\"}\n\
         {\"subject\":\"a\",\"label\":\"l\",\"object\":\"b\",\"weight\":1.0}\n",
    )
    .unwrap();
    let vocabulary = dir.join("vocabulary.jsonl");
    std::fs::write(
        &vocabulary,
        "{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"prior.md\"}\n\
         {\"subject\":\"青嶺酒造\",\"label\":\"杜氏\",\"object\":\"高瀬\",\"weight\":1.0}\n\
         {\"alias\":\"あおみね\",\"canonical\":\"青嶺酒造\",\"kind\":\"concept\"}\n",
    )
    .unwrap();

    let run = |vocab: bool| -> (String, String) {
        let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
        command
            .arg("anchoring")
            .arg(dir.join("c.jsonl"))
            .arg(dir.join("nopassage.jsonl"));
        if vocab {
            command.args(["--vocabulary", vocabulary.to_str().unwrap()]);
        }
        let output = command.output().unwrap();
        assert!(output.status.success());
        (
            String::from_utf8_lossy(&output.stdout).into_owned(),
            String::from_utf8_lossy(&output.stderr).into_owned(),
        )
    };
    let (with, with_err) = run(true);
    assert!(with.contains("TOTAL\t1\t0.000\t1.000\t-"), "{with}");
    assert!(
        with.contains("(1 batch(es) without a passage skipped)"),
        "{with}"
    );
    assert!(with_err.contains("no passage"), "{with_err}");
    let (without, _) = run(false);
    assert!(
        without.contains("TOTAL\t1\t0.000\t0.000\t-"),
        "without --vocabulary nothing supplies the alias: {without}"
    );

    // Every batch skipped: nothing to report, exit 1 (the module
    // doc's contract).
    let empty = Command::new(env!("CARGO_BIN_EXE_taguru"))
        .arg("anchoring")
        .arg(dir.join("nopassage.jsonl"))
        .output()
        .unwrap();
    assert_eq!(empty.status.code(), Some(1));
    let empty_err = String::from_utf8_lossy(&empty.stderr);
    assert!(
        empty_err.contains("no batch with a passage to judge"),
        "{empty_err}"
    );

    // The same source judged twice (two inputs): both rows survive,
    // the second under a disambiguated key, with a warning.
    let twin = dir.join("twin");
    std::fs::create_dir(&twin).unwrap();
    std::fs::copy(dir.join("c.jsonl"), twin.join("c.jsonl")).unwrap();
    let doubled = Command::new(env!("CARGO_BIN_EXE_taguru"))
        .arg("anchoring")
        .arg(dir.join("c.jsonl"))
        .arg(twin.join("c.jsonl"))
        .output()
        .unwrap();
    assert!(doubled.status.success());
    let stdout = String::from_utf8_lossy(&doubled.stdout);
    let stderr = String::from_utf8_lossy(&doubled.stderr);
    assert!(stdout.contains("TOTAL\t2\t"), "both counted: {stdout}");
    assert!(stdout.contains("c.md (twin)"), "{stdout}");
    assert!(stderr.contains("source 'c.md' already judged"), "{stderr}");

    let _ = std::fs::remove_dir_all(&dir);
}

/// ADR 0031: the two-machine scenario (§3.8) end to end — record with
/// a live endpoint, then replay with none configured at all. `--replay
/// strict` with no `TAGURU_EXTRACT_URL` makes a live call physically
/// impossible, so "the stub server saw zero requests" cannot be a
/// counting mistake the way it could be for `--replay auto`.
#[test]
fn replay_strict_reuses_a_recorded_run_with_no_model_endpoint_at_all() {
    let docs = batch_dir("extract-replay-strict-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "alpha relates to beta").unwrap();
    let doc_src = doc.to_str().unwrap();
    let out = batch_dir("extract-replay-strict-out");

    let good = json!({"associations": [
        {"subject": "alpha", "label": "relates to", "object": "beta", "weight": 1.0}
    ]})
    .to_string();
    let (url, requests) = stub_chat_server(vec![good.clone()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(requests.join().unwrap().len(), 1);

    let (batch_name, _) = read_trace(&out);
    let batch_path = out.join(&batch_name);
    let original_batch = std::fs::read(&batch_path).unwrap();
    let records_before = read_attempts_log(&out);
    let first_run_id = records_before
        .iter()
        .find(|r| r["kind"] == "document")
        .unwrap()["run_id"]
        .as_str()
        .unwrap()
        .to_string();

    let (code2, _stdout2, stderr2) = run_extract(
        &out,
        &[("TAGURU_EXTRACT_MODEL", "stub-model")],
        &["--context", "c", "--replay", "strict", doc_src],
    );
    assert_eq!(code2, 0, "stderr: {stderr2}");
    assert!(
        stderr2.contains("replayed 1/1 completions (0 live)"),
        "{stderr2}"
    );
    assert!(
        !stderr2.contains("different settings"),
        "nothing changed between the two runs: {stderr2}"
    );
    // ADR 0031 §3.6: nothing changed, so the pinned system prompt must
    // match this run's own recomputation exactly — no mismatch line.
    assert!(
        !stderr2.contains("the recorded system prompt differs from this run's"),
        "{stderr2}"
    );

    let replayed_batch = std::fs::read(&batch_path).unwrap();
    assert_eq!(
        original_batch, replayed_batch,
        "a replayed batch must be byte-identical to the one the live run wrote"
    );

    // The trace's `steering` record names the system prompt actually
    // sent and, since it was pinned, the run_id it was pinned from —
    // its hash must be the recorded run's own, the pinned text.
    let recorded_system_sha256 = records_before
        .iter()
        .find(|r| r["kind"] == "system")
        .expect("the live run must record its system prompt")["sha256"]
        .as_str()
        .unwrap()
        .to_string();
    let (_, trace_records) = read_trace(&out);
    let steering = trace_records
        .iter()
        .find(|record| record["kind"] == "steering")
        .expect("a steering record must exist");
    assert_eq!(
        steering["system_sha256"], recorded_system_sha256,
        "the pinned prompt's hash must be the recorded one: {steering}"
    );
    assert_eq!(steering["pinned_from"], first_run_id, "{steering}");

    let records_after = read_attempts_log(&out);
    assert!(records_after.len() > records_before.len());
    assert_eq!(
        &records_after[..records_before.len()],
        records_before.as_slice(),
        "the first run's records must survive, never be truncated"
    );
    let second_run = &records_after[records_before.len()..];
    let kinds: Vec<&str> = second_run
        .iter()
        .map(|r| r["kind"].as_str().unwrap())
        .collect();
    assert_eq!(kinds[0], "document");
    assert_eq!(second_run[0]["resumed"], true);
    assert!(kinds.contains(&"replay"), "{kinds:?}");
    assert!(kinds.contains(&"replay_summary"), "{kinds:?}");
    let replay_record = second_run.iter().find(|r| r["kind"] == "replay").unwrap();
    assert_eq!(replay_record["mode"], "strict");
    let summary = second_run
        .iter()
        .find(|r| r["kind"] == "replay_summary")
        .unwrap();
    assert_eq!(summary["replayed"], 1);
    assert_eq!(summary["live"], 0);

    // #823: the replay run's own re-emitted `attempt` record names the
    // original one its answer came from — so extract_metrics.py can
    // skip it (its elapsed_seconds/tokens describe the replay, not the
    // real completion) without losing the join back to the real one.
    let original_attempt = records_before
        .iter()
        .find(|r| r["kind"] == "attempt")
        .expect("the live run must record its one attempt");
    let replayed_attempt = second_run
        .iter()
        .find(|r| r["kind"] == "attempt")
        .expect("the replay run must also emit an attempt record");
    assert_eq!(
        replayed_attempt["replayed_from"]["run_id"], original_attempt["run_id"],
        "{replayed_attempt}"
    );
    assert_eq!(
        replayed_attempt["replayed_from"]["attempt_seq"], original_attempt["attempt_seq"],
        "{replayed_attempt}"
    );
    assert!(
        original_attempt.get("replayed_from").is_none(),
        "the original, live attempt never names an origin: {original_attempt}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// `--replay auto`: a settings change that only touches the system
/// prompt (`--fact-budget`) is reported on stderr field by field, same
/// as before ADR 0031 §3.6's pin — but the pin now absorbs it: this
/// document's one recorded `system` record is reused verbatim, so the
/// conversation still matches and the completion still replays, with
/// zero live calls. The recorded-vs-recomputed mismatch itself is also
/// named on stderr (ADR 0031 §3.6).
#[test]
fn replay_auto_pins_the_system_prompt_across_a_settings_change_and_reports_both_mismatches() {
    let docs = batch_dir("extract-replay-auto-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "alpha relates to beta").unwrap();
    let doc_src = doc.to_str().unwrap();
    let out = batch_dir("extract-replay-auto-out");

    let good = json!({"associations": [
        {"subject": "alpha", "label": "relates to", "object": "beta", "weight": 1.0}
    ]})
    .to_string();
    let (url, _requests) = stub_chat_server(vec![good]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    // No replies at all: if the pin failed to absorb the settings
    // change, the stub thread would be blocked in `accept()` waiting
    // for a live call that never comes, and `join()` would hang the
    // test instead of failing it.
    let (url2, requests2) = stub_chat_server(vec![]);
    let (code2, _stdout2, stderr2) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url2.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--replay",
            "auto",
            "--fact-budget",
            "3",
            doc_src,
        ],
    );
    assert_eq!(code2, 0, "stderr: {stderr2}");
    assert_eq!(
        requests2.join().unwrap().len(),
        0,
        "the pinned system prompt must still match, no live call needed"
    );
    assert!(
        stderr2.contains("replayed 1/1 completions (0 live)"),
        "{stderr2}"
    );
    assert!(
        stderr2.contains("different settings") && stderr2.contains("fact_budget"),
        "{stderr2}"
    );
    assert!(
        stderr2.contains("the recorded system prompt differs from this run's"),
        "{stderr2}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// A genuine conversation change — the document text itself, which
/// drives the user turn the pin never touches (ADR 0031 §3.6) — still
/// falls through to a live call under `--replay auto`, and still fails
/// the document under `--replay strict`, with the miss diagnostic
/// (piece id, recorded count) on stderr.
#[test]
fn replay_strict_fails_on_a_changed_document_with_the_miss_reason_on_stderr() {
    let docs = batch_dir("extract-replay-strict-miss-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "alpha relates to beta").unwrap();
    let doc_src = doc.to_str().unwrap();
    let recorded_out = batch_dir("extract-replay-strict-miss-recorded");

    let good = json!({"associations": [
        {"subject": "alpha", "label": "relates to", "object": "beta", "weight": 1.0}
    ]})
    .to_string();
    let (url, _requests) = stub_chat_server(vec![good]);
    let (code, stdout, stderr) = run_extract(
        &recorded_out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    // The document itself changed — the pin never touches the user
    // turn, so this must still miss regardless of the system prompt.
    std::fs::write(&doc, "alpha relates to gamma").unwrap();

    let replay_from = recorded_out.join(".extract-trace");
    let strict_out = batch_dir("extract-replay-strict-miss-out");
    let (code2, _stdout2, stderr2) = run_extract(
        &strict_out,
        &[("TAGURU_EXTRACT_MODEL", "stub-model")],
        &[
            "--context",
            "c",
            "--replay",
            "strict",
            "--replay-from",
            replay_from.to_str().unwrap(),
            doc_src,
        ],
    );
    assert_ne!(code2, 0, "stderr: {stderr2}");
    assert!(stderr2.contains("--replay strict"), "{stderr2}");
    assert!(stderr2.contains("recorded attempt"), "{stderr2}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&recorded_out);
    let _ = std::fs::remove_dir_all(&strict_out);
}

/// ADR 0031 §3.6's core motivation: a multi-document run where only
/// document 1 changes text between record and replay still lets
/// document 2 — whose own recorded system prompt reflects the
/// vocabulary document 1's *original* extraction accumulated — replay
/// with zero live calls, because document 2's pin is scoped to its own
/// `ReplayIndex` and never depends on what document 1 replays to.
/// Without the pin, document 2's recomputed system prompt would follow
/// whatever vocabulary this run's own document 1 accumulates and could
/// drift from what was recorded.
#[test]
fn replay_auto_pins_each_documents_own_system_prompt_independently() {
    let docs = batch_dir("extract-replay-multi-doc-docs");
    let doc_a = docs.join("a.md");
    let doc_b = docs.join("b.md");
    std::fs::write(&doc_a, "alpha relates to beta").unwrap();
    std::fs::write(&doc_b, "gamma relates to delta").unwrap();
    let doc_a_src = doc_a.to_str().unwrap();
    let doc_b_src = doc_b.to_str().unwrap();
    let out = batch_dir("extract-replay-multi-doc-out");

    let good_a = json!({"associations": [
        {"subject": "alpha", "label": "relates to", "object": "beta", "weight": 1.0}
    ]})
    .to_string();
    let good_b = json!({"associations": [
        {"subject": "gamma", "label": "relates to", "object": "delta", "weight": 1.0}
    ]})
    .to_string();
    let (url, _requests) = stub_chat_server(vec![good_a, good_b]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc_a_src, doc_b_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    // A settings change touching every document's system prompt: were
    // either document's pin to fail (or to leak from the other), a
    // live call would be needed and this empty-reply stub would hang.
    let (url2, requests2) = stub_chat_server(vec![]);
    let (code2, _stdout2, stderr2) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url2.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--replay",
            "auto",
            "--fact-budget",
            "3",
            doc_a_src,
            doc_b_src,
        ],
    );
    assert_eq!(code2, 0, "stderr: {stderr2}");
    assert_eq!(
        requests2.join().unwrap().len(),
        0,
        "both documents' own system prompts must be pinned, no live calls"
    );
    assert_eq!(
        stderr2.matches("replayed 1/1 completions (0 live)").count(),
        2,
        "one report line per document: {stderr2}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// When a document's log names two distinct `system` records (here, by
/// hand-editing a second one in after a normal recorded run — the real
/// cause is a checkpoint-resumed document spanning a run whose
/// vocabulary differed), the pin declines rather than guessing: it is
/// reported once on stderr, and the run falls back to computing its
/// own system prompt for that document, hitting or missing on the
/// conversation content exactly as an unpinned replay would.
#[test]
fn replay_does_not_pin_when_the_log_names_two_distinct_system_prompts() {
    let docs = batch_dir("extract-replay-ambiguous-system-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "alpha relates to beta").unwrap();
    let doc_src = doc.to_str().unwrap();
    let out = batch_dir("extract-replay-ambiguous-system-out");

    let good = json!({"associations": [
        {"subject": "alpha", "label": "relates to", "object": "beta", "weight": 1.0}
    ]})
    .to_string();
    let (url, _requests) = stub_chat_server(vec![good.clone()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    // Splice a second, distinct `system` record into the recorded
    // document's attempts log — the log now names two.
    let trace_dir = out.join(".extract-trace");
    let attempts_path = std::fs::read_dir(&trace_dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.to_string_lossy().ends_with(".attempts.jsonl"))
        .expect("the recorded run must leave exactly one attempts log");
    let mut text = std::fs::read_to_string(&attempts_path).unwrap();
    let other = "a completely different system prompt";
    let other_sha256 = sha256_hex_of(other);
    text.push_str(&format!(
        "{}\n",
        json!({
            "kind": "system",
            "sha256": other_sha256,
            "bytes": other.len(),
            "content": other,
        })
    ));
    std::fs::write(&attempts_path, text).unwrap();

    // The same settings, unchanged: without the pin, the recomputed
    // system prompt matches the original recording exactly, so this
    // must still hit — proving the ambiguity fell back to a plain
    // recompute rather than failing the completion outright.
    let (url2, requests2) = stub_chat_server(vec![]);
    let (code2, _stdout2, stderr2) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url2.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", "--replay", "auto", doc_src],
    );
    assert_eq!(code2, 0, "stderr: {stderr2}");
    assert_eq!(requests2.join().unwrap().len(), 0, "{stderr2}");
    assert!(
        stderr2.contains("2 distinct system prompts recorded") && stderr2.contains("not pinning"),
        "{stderr2}"
    );

    // Not pinned: the trace's `steering` record still names the system
    // prompt actually sent, but `pinned_from` is absent.
    let (_, trace_records) = read_trace(&out);
    let steering = trace_records
        .iter()
        .find(|record| record["kind"] == "steering")
        .expect("a steering record must exist");
    assert_eq!(
        steering["system_sha256"].as_str().unwrap().len(),
        64,
        "{steering}"
    );
    assert!(steering.get("pinned_from").is_none(), "{steering}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// ADR 0031 §3.2's `--parallel` determinism claim: two independent
/// `--replay strict` runs against the same recorded log, each under
/// `--parallel 4`, land on the identical batch — whatever race order
/// either the original recording or either replay's own workers
/// happened to run in. FIFO consumption only ever collides within one
/// piece's own ladder, and different pieces always key differently.
#[test]
fn replay_is_deterministic_under_parallel() {
    let docs = batch_dir("extract-replay-parallel-docs");
    let doc = docs.join("big.md");
    std::fs::write(&doc, multi_chunk_document(50)).unwrap();
    let doc_src = doc.to_str().unwrap();

    let probe = batch_dir("extract-replay-parallel-probe");
    let (code, dry_stdout, stderr) =
        run_extract(&probe, &[], &["--dry-run", "--context", "c", doc_src]);
    assert_eq!(code, 0, "stdout: {dry_stdout}\nstderr: {stderr}");
    let total_chunks = chunk_count_from_dry_run(&dry_stdout);
    assert!(
        total_chunks >= 4,
        "fixture must span several chunks: {dry_stdout}"
    );

    fn reply_for(index: usize) -> String {
        json!({"associations": [
            {"subject": "S", "label": "chunk", "object": format!("value-{index}"), "weight": 1.0}
        ]})
        .to_string()
    }

    let recorded_out = batch_dir("extract-replay-parallel-recorded");
    let (url, _captured) =
        stub_chat_server_concurrent(|index, _attempt| chat_ok(&reply_for(index)));
    let (code, stdout, stderr) = run_extract(
        &recorded_out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", "--parallel", "4", doc_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let (recorded_batch_name, _) = read_trace(&recorded_out);
    let recorded_batch = std::fs::read(recorded_out.join(&recorded_batch_name)).unwrap();

    let replay_from = recorded_out.join(".extract-trace");
    let mut replayed_batches = Vec::new();
    for tag in ["a", "b"] {
        let out = batch_dir(&format!("extract-replay-parallel-out-{tag}"));
        let (code, stdout, stderr) = run_extract(
            &out,
            &[("TAGURU_EXTRACT_MODEL", "stub-model")],
            &[
                "--context",
                "c",
                "--replay",
                "strict",
                "--replay-from",
                replay_from.to_str().unwrap(),
                "--parallel",
                "4",
                doc_src,
            ],
        );
        assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
        let (batch_name, _) = read_trace(&out);
        let batch = std::fs::read(out.join(&batch_name)).unwrap();
        replayed_batches.push((out, batch));
    }
    assert_eq!(
        replayed_batches[0].1, recorded_batch,
        "replay a vs. recorded"
    );
    assert_eq!(
        replayed_batches[0].1, replayed_batches[1].1,
        "replay a vs. replay b"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&probe);
    let _ = std::fs::remove_dir_all(&recorded_out);
    for (out, _) in &replayed_batches {
        let _ = std::fs::remove_dir_all(out);
    }
}

/// `TAGURU_EXTRACT_REPLAY_FROM` (the env fallback, no `--replay-from`
/// flag at all) must be honored exactly like the flag — a run pointed
/// at a recorded log purely through the environment still replays.
#[test]
fn replay_from_env_var_is_honored_with_no_flag() {
    let docs = batch_dir("extract-replay-from-env-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "alpha relates to beta").unwrap();
    let doc_src = doc.to_str().unwrap();
    let recorded_out = batch_dir("extract-replay-from-env-recorded");

    let good = json!({"associations": [
        {"subject": "alpha", "label": "relates to", "object": "beta", "weight": 1.0}
    ]})
    .to_string();
    let (url, _requests) = stub_chat_server(vec![good]);
    let (code, stdout, stderr) = run_extract(
        &recorded_out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let replay_from = recorded_out.join(".extract-trace");
    let strict_out = batch_dir("extract-replay-from-env-out");
    let (code2, _stdout2, stderr2) = run_extract(
        &strict_out,
        &[
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ("TAGURU_EXTRACT_REPLAY_FROM", replay_from.to_str().unwrap()),
        ],
        &["--context", "c", "--replay", "strict", doc_src],
    );
    assert_eq!(code2, 0, "stderr: {stderr2}");
    assert!(
        stderr2.contains("replayed 1/1 completions (0 live)"),
        "{stderr2}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&recorded_out);
    let _ = std::fs::remove_dir_all(&strict_out);
}

/// #820: `--replay strict` relaxes `TAGURU_EXTRACT_URL`, never
/// `TAGURU_EXTRACT_MODEL` — the model name is still a manifest
/// computation input, so writing an empty one would be a lie the
/// manifest believes on its next live run.
#[test]
fn replay_strict_still_requires_the_model_env_var() {
    let docs = batch_dir("extract-replay-strict-needs-model-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "alpha relates to beta").unwrap();
    let doc_src = doc.to_str().unwrap();
    let out = batch_dir("extract-replay-strict-needs-model-out");

    let (code, _stdout, stderr) = run_extract(
        &out,
        &[],
        &["--context", "c", "--replay", "strict", doc_src],
    );
    assert_eq!(code, 2, "stderr: {stderr}");
    assert!(stderr.contains("TAGURU_EXTRACT_MODEL"), "{stderr}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// `--structured-output auto` needs a live probe by construction (ADR
/// 0021) — `--replay strict` with no `TAGURU_EXTRACT_URL` can never
/// resolve it, so combining the two is a usage error rather than a
/// silently-skipped probe.
#[test]
fn replay_strict_with_structured_output_auto_and_no_url_is_a_usage_error() {
    let docs = batch_dir("extract-replay-strict-auto-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "alpha relates to beta").unwrap();
    let doc_src = doc.to_str().unwrap();
    let out = batch_dir("extract-replay-strict-auto-out");

    let (code, _stdout, stderr) = run_extract(
        &out,
        &[("TAGURU_EXTRACT_MODEL", "stub-model")],
        &[
            "--context",
            "c",
            "--replay",
            "strict",
            "--structured-output",
            "auto",
            doc_src,
        ],
    );
    assert_eq!(code, 2, "stderr: {stderr}");
    assert!(stderr.contains("--structured-output auto"), "{stderr}");
    assert!(stderr.contains("rung"), "{stderr}");

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// #822: `--resume-from` names one of ADR 0030's step names and folds
/// onto a `--replay` mode instead of taking one directly. `call`
/// through `verify` all fold onto `--replay auto` — a zero-reply stub
/// proves the completion still replays with no live call for every one
/// of them, not just a single representative.
#[test]
fn resume_from_call_through_verify_all_fold_into_replay_auto() {
    for step in [
        "call",
        "parse",
        "validate",
        "reconcile",
        "merge",
        "render",
        "verify",
    ] {
        let docs = batch_dir(&format!("extract-resume-from-{step}-docs"));
        let doc = docs.join("a.md");
        std::fs::write(&doc, "alpha relates to beta").unwrap();
        let doc_src = doc.to_str().unwrap();
        let out = batch_dir(&format!("extract-resume-from-{step}-out"));

        let good = json!({"associations": [
            {"subject": "alpha", "label": "relates to", "object": "beta", "weight": 1.0}
        ]})
        .to_string();
        let (url, _requests) = stub_chat_server(vec![good]);
        let (code, stdout, stderr) = run_extract(
            &out,
            &[
                ("TAGURU_EXTRACT_URL", url.as_str()),
                ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ],
            &["--context", "c", doc_src],
        );
        assert_eq!(code, 0, "{step}: stdout: {stdout}\nstderr: {stderr}");

        let (url2, requests2) = stub_chat_server(vec![]);
        let (code2, _stdout2, stderr2) = run_extract(
            &out,
            &[
                ("TAGURU_EXTRACT_URL", url2.as_str()),
                ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ],
            &["--context", "c", "--resume-from", step, doc_src],
        );
        assert_eq!(code2, 0, "{step}: stderr: {stderr2}");
        assert_eq!(
            requests2.join().unwrap().len(),
            0,
            "{step}: --resume-from {step} must fold onto --replay auto, no live call"
        );
        assert!(
            stderr2.contains("replayed 1/1 completions (0 live)"),
            "{step}: {stderr2}"
        );

        let _ = std::fs::remove_dir_all(&docs);
        let _ = std::fs::remove_dir_all(&out);
    }
}

/// `read`/`plan`/`steer` have no usable record in this version (only
/// `prompt`/`call` are recorded at all) — `--resume-from` on any of
/// them folds onto a plain, unreplayed run: a prior recorded run
/// exists, but the second run still pays for a live call.
#[test]
fn resume_from_read_plan_steer_all_fold_into_an_unreplayed_run() {
    for step in ["read", "plan", "steer"] {
        let docs = batch_dir(&format!("extract-resume-from-{step}-docs"));
        let doc = docs.join("a.md");
        std::fs::write(&doc, "alpha relates to beta").unwrap();
        let doc_src = doc.to_str().unwrap();
        let out = batch_dir(&format!("extract-resume-from-{step}-out"));

        let good = json!({"associations": [
            {"subject": "alpha", "label": "relates to", "object": "beta", "weight": 1.0}
        ]})
        .to_string();
        let (url, _requests) = stub_chat_server(vec![good.clone()]);
        let (code, stdout, stderr) = run_extract(
            &out,
            &[
                ("TAGURU_EXTRACT_URL", url.as_str()),
                ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ],
            &["--context", "c", doc_src],
        );
        assert_eq!(code, 0, "{step}: stdout: {stdout}\nstderr: {stderr}");

        // No --force: --resume-from must bypass the manifest skip on
        // its own (the whole point of naming a resume step is a
        // deliberate redo — a silent "unchanged, skipped" would defeat
        // that intent exactly as much here as it would under
        // --replay).
        let (url2, requests2) = stub_chat_server(vec![good]);
        let (code2, stdout2, stderr2) = run_extract(
            &out,
            &[
                ("TAGURU_EXTRACT_URL", url2.as_str()),
                ("TAGURU_EXTRACT_MODEL", "stub-model"),
            ],
            &["--context", "c", "--resume-from", step, doc_src],
        );
        assert_eq!(code2, 0, "{step}: stderr: {stderr2}");
        assert!(
            !stdout2.contains("unchanged, skipped"),
            "{step}: --resume-from must bypass the manifest skip: {stdout2}"
        );
        assert_eq!(
            requests2.join().unwrap().len(),
            1,
            "{step}: --resume-from {step} must never consult the log, one live call"
        );
        assert!(
            !stderr2.contains("replayed"),
            "{step}: no replay engaged at all: {stderr2}"
        );

        let _ = std::fs::remove_dir_all(&docs);
        let _ = std::fs::remove_dir_all(&out);
    }
}

/// `--resume-from prompt` is `--replay auto` with ADR 0031 §3.6's
/// system-prompt pin turned off (#821, #822): a settings change that a
/// plain `--replay auto` run absorbs via the pin (proven in
/// `replay_auto_pins_the_system_prompt_across_a_settings_change_and_reports_both_mismatches`)
/// instead falls through to a live call here, because `resolve_system`
/// never even asks whether the log's one recorded `system` could be
/// pinned.
#[test]
fn resume_from_prompt_disables_the_system_pin_so_a_settings_change_falls_through() {
    let docs = batch_dir("extract-resume-from-prompt-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "alpha relates to beta").unwrap();
    let doc_src = doc.to_str().unwrap();
    let out = batch_dir("extract-resume-from-prompt-out");

    let good = json!({"associations": [
        {"subject": "alpha", "label": "relates to", "object": "beta", "weight": 1.0}
    ]})
    .to_string();
    let (url, _requests) = stub_chat_server(vec![good.clone()]);
    let (code, stdout, stderr) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &["--context", "c", doc_src],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let (url2, requests2) = stub_chat_server(vec![good]);
    let (code2, _stdout2, stderr2) = run_extract(
        &out,
        &[
            ("TAGURU_EXTRACT_URL", url2.as_str()),
            ("TAGURU_EXTRACT_MODEL", "stub-model"),
        ],
        &[
            "--context",
            "c",
            "--resume-from",
            "prompt",
            "--fact-budget",
            "3",
            doc_src,
        ],
    );
    assert_eq!(code2, 0, "stderr: {stderr2}");
    assert_eq!(
        requests2.join().unwrap().len(),
        1,
        "the pin is off, so the changed system prompt must miss and fall through live"
    );
    assert!(
        stderr2.contains("replayed 0/1 completions (1 live)"),
        "{stderr2}"
    );
    assert!(
        !stderr2.contains("the recorded system prompt differs from this run's"),
        "no pin was ever attempted, so there is nothing to report a mismatch about: {stderr2}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}

/// An unknown step name is a usage error naming the closed vocabulary,
/// exactly like `--replay`'s own closed set of values.
#[test]
fn resume_from_rejects_an_unknown_step_name() {
    let docs = batch_dir("extract-resume-from-unknown-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "alpha relates to beta").unwrap();
    let doc_src = doc.to_str().unwrap();
    let out = batch_dir("extract-resume-from-unknown-out");

    let (code, _stdout, stderr) = run_extract(
        &out,
        &[("TAGURU_EXTRACT_MODEL", "stub-model")],
        &["--context", "c", "--resume-from", "escalate", doc_src],
    );
    assert_eq!(code, 2, "{stderr}");
    assert!(
        stderr.contains(
            "--resume-from takes one of: read, plan, steer, prompt, call, parse, validate, \
             reconcile, merge, render, verify"
        ),
        "must list every one of ADR 0030's 11 step names, not just a prefix: {stderr}"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}
