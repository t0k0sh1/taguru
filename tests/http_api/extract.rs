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
    (0..count)
        .map(|i| format!("Paragraph {i}: {}", "x".repeat(3000)))
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
            // batch lands).
            name.to_str() != Some(".extract-manifest.json")
                && name.to_str() != Some(".extract-checkpoints")
        })
        .collect()
}

/// Parses a `--diagnostics-out` sidecar into its records, in file
/// order — one JSON object per line (issue #200).
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
        .env_remove("TAGURU_EXTRACT_LOSSY")
        .env_remove("TAGURU_EXTRACT_DIAGNOSTICS")
        .env_remove("TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES")
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
    let data_dir = std::env::temp_dir().join(format!("taguru-http-extract-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
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
/// exactly once: the next request drops `max_tokens` and resends the
/// base ask NEUTRALLY — no corrective turn, no replay of the truncated
/// answer, none of the legacy SHORTER wording (which asks for less
/// than the budget could now hold).
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
    assert!(
        escalated.get("max_tokens").is_none(),
        "escalation must drop the budget, never re-ask under it: {escalated}"
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

/// The core regression from ADR 0001: `finish_reason: "length"` on an
/// answer whose prefix happens to parse is still truncation. The valid
/// prefix must never be imported — the piece regenerates at the
/// escalated budget and only THAT answer lands.
#[test]
fn a_length_terminated_answer_that_happens_to_parse_is_never_treated_as_success() {
    let docs = batch_dir("extract-validprefix-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-validprefix-out");

    let prefix = json!({"associations":
        [{"subject": "half_answer", "label": "l", "object": "x", "weight": 1.0}]});
    let complete = json!({"associations":
        [{"subject": "whole_answer", "label": "l", "object": "x", "weight": 1.0}]});
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
    assert!(json_body_of(&requests[1]).get("max_tokens").is_none());
    assert_eq!(json_body_of(&requests[2])["max_tokens"], 512);
    assert_eq!(json_body_of(&requests[3])["max_tokens"], 512);

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
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
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-strict-weight-out");

    let bad_reply = json!({
        "associations": [
            {"subject": "a", "label": "l", "object": "b", "weight": "strong"}
        ],
        "aliases": []
    })
    .to_string();
    let good_reply = json!({
        "associations": [
            {"subject": "a", "label": "l", "object": "b", "weight": 0.9}
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
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-strict-fail-out");

    let bad_reply = json!({
        "associations": [
            {"subject": "a", "label": "l", "object": "b", "weight": "strong"}
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
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-strict-untouched-out");

    let good_reply = json!({
        "associations": [
            {"subject": "a", "label": "l", "object": "b", "weight": 1.0}
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
            {"subject": "a", "label": "l", "object": "b", "weight": "strong"}
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

/// Stage 2 (cross-chunk alias validation): a dangling canonical earns
/// its own corrective turn naming the exact alias path.
#[test]
fn a_dangling_alias_earns_a_cross_chunk_corrective_turn() {
    let docs = batch_dir("extract-strict-dangling-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-strict-dangling-out");

    let bad_reply = json!({
        "associations": [
            {"subject": "a", "label": "l", "object": "b"}
        ],
        "aliases": [
            {"alias": "x", "canonical": "存在しない", "kind": "concept"}
        ]
    })
    .to_string();
    let good_reply = json!({
        "associations": [
            {"subject": "a", "label": "l", "object": "b"}
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

    let requests = requests.join().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests[1].contains("aliases[0].canonical: names nothing the associations contain"),
        "{}",
        requests[1]
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
    std::fs::write(&doc, multi_chunk_document(20)).unwrap();
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
            {"subject": "a", "label": "l", "object": "b", "weight": 0}
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
            {"subject": "a", "label": "l", "object": "b", "weight": 0}
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
            {"subject": "a", "label": "l", "object": "b", "weight": 0}
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

    let records = read_diagnostics(&diag);
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
    let records = read_diagnostics(&diag);
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
        let records = read_diagnostics(&diag);
        assert_eq!(records.len(), 2, "{records:?}");
        assert_eq!(records[0]["state"], "length_limited");
        assert_eq!(records[0]["length_limited"], true);
        assert_eq!(records[0]["requested_max_tokens"], 512);
        assert!(!records[0]["parse_error"].is_null());
        assert_eq!(records[1]["state"], "stop_valid");
        assert!(
            records[1].get("requested_max_tokens").is_none(),
            "escalation drops the budget: {:?}",
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
        let records = read_diagnostics(&diag);
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
        let records = read_diagnostics(&diag);
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

    let records = read_diagnostics(&diag);
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

    let records = read_diagnostics(&diag);
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
    let records = read_diagnostics(&diag);
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

    let records = read_diagnostics(&diag);
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
    let records = read_diagnostics(&diag);
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

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut saved = String::new();
    while std::time::Instant::now() < deadline {
        if let Ok(text) = std::fs::read_to_string(&diag)
            && !text.trim().is_empty()
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
    let record: Value = serde_json::from_str(saved.lines().next().unwrap())
        .unwrap_or_else(|error| panic!("the surviving line must parse: {error}\n{saved}"));
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

/// Stage 2's cross-chunk alias correction earns its own diagnostics
/// record, `stage: "cross_chunk"` — distinct from the item-stage
/// records the same run also emits.
#[test]
fn diagnostics_records_the_stage_two_cross_chunk_correction() {
    let docs = batch_dir("extract-diag-stage2-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let out = batch_dir("extract-diag-stage2-out");
    let diag_dir = batch_dir("extract-diag-stage2-diag");
    let diag = diag_dir.join("diag.jsonl");

    let bad_reply = json!({
        "associations": [
            {"subject": "a", "label": "l", "object": "b"}
        ],
        "aliases": [
            {"alias": "x", "canonical": "存在しない", "kind": "concept"}
        ]
    })
    .to_string();
    let good_reply = json!({
        "associations": [
            {"subject": "a", "label": "l", "object": "b"}
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

    let records = read_diagnostics(&diag);
    assert_eq!(records.len(), 2, "{records:?}");
    assert_eq!(records[0]["stage"], "item");
    assert_eq!(records[0]["state"], "stop_valid");
    assert_eq!(records[1]["stage"], "cross_chunk");
    assert_eq!(records[1]["state"], "stop_valid");
    assert_eq!(records[1]["attempt"], 1);
    assert_eq!(records[1]["max_attempts"], 1);

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
    let records = read_diagnostics(&diag);
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
            {"subject": "a", "label": "l", "object": "b", "weight": 0}
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
/// flatten-then-`.json` naming scheme `extract.rs`'s private
/// `checkpoint_file_name` uses, replicated here since an integration
/// test only ever sees the compiled binary's filesystem effects.
fn checkpoint_file_path(out: &std::path::Path, source: &str) -> std::path::PathBuf {
    let name = source.replace(['/', '\\', ':'], "__");
    out.join(".extract-checkpoints")
        .join(format!("{name}.json"))
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
        {"subject": "S", "label": "l", "object": "chunk0", "weight": 1.0}
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
        {"subject": "S", "label": "l", "object": "chunk1", "weight": 1.0}
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
            {"subject": "S", "label": "l", "object": "chunk0", "weight": 1.0}
        ]})
        .to_string(),
        json!({"associations": [
            {"subject": "S", "label": "l", "object": "chunk1", "weight": 1.0}
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
            {"subject": "S", "label": "l", "object": "chunk0", "weight": 1.0}
        ]})
        .to_string(),
        json!({"associations": [
            {"subject": "S", "label": "l", "object": "chunk1", "weight": 1.0}
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
            {"subject": "S", "label": "l", "object": "chunk0", "weight": 1.0}
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
                {"subject": "S", "label": "l", "object": format!("chunk{index}"), "weight": 1.0}
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

/// Issue #179's cooperative stop: SIGINT lets an in-flight chunk finish
/// (and get checkpointed) before the process exits with code 130 —
/// distinct from a hard failure — and a rerun resumes without
/// re-asking the model for what already landed. Chunk 0's response is
/// deliberately delayed server-side so SIGINT can be sent any time
/// before it, without racing the process's own startup — no matter
/// when it lands in that window, chunk 0 still completes and gets
/// checkpointed before the next iteration notices the flag.
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
            {"subject": "S", "label": "l", "object": "chunk0", "weight": 1.0}
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

    let pid = child.id().to_string();
    Command::new("kill")
        .args(["-INT", &pid])
        .status()
        .expect("kill must run");

    let output = child
        .wait_with_output()
        .expect("extract must exit after SIGINT");
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
                {"subject": "S", "label": "l", "object": format!("chunk{index}"), "weight": 1.0}
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
#[test]
fn a_second_sigint_forces_an_immediate_exit_even_while_permanently_blocked() {
    use std::time::Duration;

    let docs = batch_dir("extract-checkpoint-doublesigint-docs");
    let doc = docs.join("a.md");
    std::fs::write(&doc, "small document").unwrap();
    let doc_src = doc.to_str().unwrap().to_string();
    let out = batch_dir("extract-checkpoint-doublesigint-out");

    // Accepts every connection and never answers any of them — the
    // one (and only) chunk's request blocks forever.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let mut held = Vec::new();
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
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
    let output = child
        .wait_with_output()
        .expect("extract must exit after the second SIGINT");
    assert_eq!(output.status.code(), Some(130), "{output:?}");
    assert_eq!(
        checkpoint_units_count(&out, &doc_src),
        0,
        "the one chunk never completed, so nothing should be checkpointed"
    );

    let _ = std::fs::remove_dir_all(&docs);
    let _ = std::fs::remove_dir_all(&out);
}
