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

/// Every file in `dir` other than the extract manifest — a failed
/// document must leave none.
fn stray_batch_files(dir: &std::path::Path) -> Vec<std::ffi::OsString> {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name())
        .filter(|name| name.to_str() != Some(".extract-manifest.json"))
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
    let (code, stdout, stderr) = run_extract(
        &out,
        &provider,
        &[
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
    assert!(stdout.contains("2 item(s) dropped"), "{stdout}");
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
        &["--force", "--context", "sake", aomine_src, takase_src],
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
        &["--context", "vats", aomine_src, takase_src],
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
