//! `taguru import --url`: chunked remote import via `POST /import`
//! (issue #247, ADR 0002 §6/§8/§9) — the counterpart to the local
//! path's own pins in `tests/http_api/offline_import.rs`.
//!
//! 503 shed and the heavy-ops semaphore are not reproduced here for
//! the same reason `remote_compact.rs` skips them: this harness cannot
//! force a saturation condition deterministically. `Retry-After`
//! display and the structured envelope decoding are pinned at the
//! unit level instead — `src/remote.rs`'s `import_chunk` tests.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;

use crate::support::{Server, batch_dir, run_cli, run_import};

/// Every file directly under `dir`, name → bytes — the same
/// byte-identical comparison `remote_export.rs` uses to pin a remote
/// walk against the local path of the same data.
fn dir_contents(dir: &Path) -> std::collections::BTreeMap<String, Vec<u8>> {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_file())
        .map(|entry| {
            (
                entry.file_name().into_string().unwrap(),
                std::fs::read(entry.path()).unwrap(),
            )
        })
        .collect()
}

/// A full remote import of a multi-source, multi-context stream (with
/// a passage, an alias, and an association) must land byte-identical
/// content to importing the same file locally — proven by exporting
/// both afterward and comparing the files, the same round-trip
/// `remote_export.rs`'s own full-export test uses.
#[test]
fn a_full_remote_import_matches_the_local_import_of_the_same_stream() {
    let batches = batch_dir("remote-import-full");
    let file = batches.join("seed.jsonl");
    std::fs::write(
        &file,
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
         \"create\": {\"description\": \"酒蔵の知識\"}}\n\
         {\"passage\": \"青嶺酒造の紹介。\\n\\n代表銘柄は青嶺。\", \"stored_at\": 1700000000}\n\
         {\"subject\": \"青嶺酒造\", \"label\": \"代表銘柄\", \"object\": \"青嶺\", \
          \"weight\": 1.0, \"paragraph\": 1}\n\
         {\"alias\": \"Aomine\", \"canonical\": \"青嶺酒造\", \"kind\": \"concept\"}\n\
         {\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"b.md\"}\n\
         {\"subject\": \"青嶺酒造\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 2.0}\n",
    )
    .expect("fixture must be writable");

    let server = Server::start("remote-import-full");
    let (code, stdout, stderr) = run_cli(
        &["import", "--url", &server.base, file.to_str().unwrap()],
        &[],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(
        stderr.matches("import → ").count(),
        1,
        "the target must print exactly once: {stderr}"
    );
    assert!(
        !stderr.contains("warning:"),
        "same-build CLI and server must not trip the version-skew warning: {stderr}"
    );

    let out_remote = batches.join("out-remote");
    let (code, _stdout, stderr) = run_cli(
        &[
            "export",
            "--url",
            &server.base,
            "--out",
            out_remote.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "{stderr}");

    let local_data = batches.join("local-data");
    let (code, _stdout, stderr) = run_import(&local_data, &[file.to_str().unwrap()]);
    assert_eq!(code, 0, "{stderr}");
    let out_local = batches.join("out-local");
    let (code, _stdout, stderr) = run_cli(
        &["export", "--out", out_local.to_str().unwrap()],
        &[("TAGURU_DATA_DIR", local_data.to_str().unwrap())],
    );
    assert_eq!(code, 0, "{stderr}");

    assert_eq!(
        dir_contents(&out_remote),
        dir_contents(&out_local),
        "a remote import must land the same content as the local import of the same stream"
    );

    let _ = std::fs::remove_dir_all(&batches);
}

/// `--dry-run` sends every chunk as `?dry_run=true` and writes nothing
/// — the context named in the batch must not exist afterward.
#[test]
fn a_remote_dry_run_previews_every_chunk_and_writes_nothing() {
    let batches = batch_dir("remote-import-dryrun");
    let file = batches.join("seed.jsonl");
    std::fs::write(
        &file,
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
         \"create\": {\"description\": \"d\"}}\n\
         {\"subject\": \"s\", \"label\": \"l\", \"object\": \"o\", \"weight\": 1.0}\n",
    )
    .expect("fixture must be writable");

    let server = Server::start("remote-import-dryrun");
    let (code, stdout, stderr) = run_cli(
        &[
            "import",
            "--url",
            &server.base,
            "--dry-run",
            file.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("dry run:"), "{stdout}");
    assert!(stdout.contains("nothing applied"), "{stdout}");

    let (status, _) = server.call("GET", "/contexts/sake", None);
    assert_eq!(status, 404, "a dry run must write nothing");

    let _ = std::fs::remove_dir_all(&batches);
}

/// A server whose body cap is far smaller than the client's starting
/// 4 MiB budget forces the 413-halving path repeatedly until every
/// chunk fits — and the import still lands every source, across
/// however many chunks that takes.
#[test]
fn a_small_body_cap_forces_multiple_chunks_and_the_import_still_lands() {
    let batches = batch_dir("remote-import-smallcap");
    let file = batches.join("seed.jsonl");
    let mut content = String::new();
    for i in 0..6 {
        content.push_str(&format!(
            "{{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s{i}.md\", \
             \"create\": {{\"description\": \"d\"}}}}\n\
             {{\"subject\": \"s{i}\", \"label\": \"l\", \"object\": \"o\", \"weight\": 1.0}}\n"
        ));
    }
    std::fs::write(&file, &content).expect("fixture must be writable");

    let server = Server::start_with_env(
        "remote-import-smallcap",
        &[("TAGURU_MAX_BODY_BYTES", "600")],
    );
    let (code, stdout, stderr) = run_cli(
        &["import", "--url", &server.base, file.to_str().unwrap()],
        &[],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.matches("chunk ").count() > 1,
        "a 600-byte cap must force more than one chunk: {stdout}"
    );
    assert!(
        stdout.contains("6 batch(es) applied across 1 context(s)"),
        "{stdout}"
    );

    let (_, export) = server.call("GET", "/contexts/sake/export", None);
    let export_text = export.as_str().expect("export is raw text, not JSON");
    for i in 0..6 {
        assert!(
            export_text.contains(&format!("s{i}.md")),
            "source s{i}.md must have landed: {export_text}"
        );
    }

    let _ = std::fs::remove_dir_all(&batches);
}

/// A single batch that alone exceeds the server's (tiny) body cap is a
/// hard, pre-send-style failure naming the source and both remedies —
/// the server's own 413 words ride along too, since this is the one
/// case the client cannot avoid ever sending it.
#[test]
fn a_lone_batch_the_server_still_413s_is_a_hard_error_naming_both_remedies() {
    let batches = batch_dir("remote-import-lone413");
    let file = batches.join("seed.jsonl");
    let passage: String = "青".repeat(500);
    std::fs::write(
        &file,
        format!(
            "{{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"big.md\", \
             \"create\": {{\"description\": \"d\"}}}}\n{{\"passage\": \"{passage}\"}}\n"
        ),
    )
    .expect("fixture must be writable");

    let server =
        Server::start_with_env("remote-import-lone413", &[("TAGURU_MAX_BODY_BYTES", "512")]);
    let (code, _stdout, stderr) = run_cli(
        &["import", "--url", &server.base, file.to_str().unwrap()],
        &[],
    );
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("big.md"), "{stderr}");
    assert!(stderr.contains("TAGURU_MAX_BODY_BYTES"), "{stderr}");
    assert!(stderr.contains("split"), "{stderr}");

    let (status, _) = server.call("GET", "/contexts/sake", None);
    assert_eq!(status, 404, "a refused lone batch must apply nothing");

    let _ = std::fs::remove_dir_all(&batches);
}

/// A `taguru_group` record in an earlier file, naming contexts a later
/// file's batches create, still restores — proof that group units ride
/// after every batch unit on the wire, the same order the local path
/// already applies them in.
#[test]
fn groups_ride_after_every_batch_chunk_and_restore_remotely() {
    let batches = batch_dir("remote-import-groups");
    let group_file = batches.join("00-group.jsonl");
    std::fs::write(
        &group_file,
        "{\"taguru_group\": 1, \"name\": \"kura\", \"contexts\": [\"sake\", \"beer\"]}\n",
    )
    .expect("fixture must be writable");
    let batch_file = batches.join("01-batches.jsonl");
    std::fs::write(
        &batch_file,
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
         \"create\": {\"description\": \"d\"}}\n\
         {\"subject\": \"s\", \"label\": \"l\", \"object\": \"o\", \"weight\": 1.0}\n\
         {\"taguru_batch\": 1, \"context\": \"beer\", \"source\": \"b.md\", \
         \"create\": {\"description\": \"d\"}}\n\
         {\"subject\": \"s2\", \"label\": \"l2\", \"object\": \"o2\", \"weight\": 1.0}\n",
    )
    .expect("fixture must be writable");

    let server = Server::start("remote-import-groups");
    let (code, stdout, stderr) = run_cli(
        &[
            "import",
            "--url",
            &server.base,
            group_file.to_str().unwrap(),
            batch_file.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("group 'kura'"), "{stdout}");

    let (status, group) = server.call("GET", "/groups/kura", None);
    assert_eq!(status, 200, "{group}");
    assert_eq!(
        group["result"]["contexts"].as_array().map(Vec::len),
        Some(2)
    );

    let _ = std::fs::remove_dir_all(&batches);
}

/// A batch naming a nonexistent context with no `create` block refuses
/// (404, the ordinary `NoContext` refusal) — with a tiny body cap
/// forcing each batch onto its own chunk, the batch before the refusal
/// must have landed and the batch after it must never have been sent.
#[test]
fn a_mid_stream_refusal_reports_the_prefix_and_what_was_never_sent() {
    let batches = batch_dir("remote-import-midrefusal");
    let file = batches.join("seed.jsonl");
    std::fs::write(
        &file,
        "{\"taguru_batch\": 1, \"context\": \"a\", \"source\": \"a.md\", \
         \"create\": {\"description\": \"d\"}}\n\
         {\"subject\": \"s\", \"label\": \"l\", \"object\": \"o\", \"weight\": 1.0}\n\
         {\"taguru_batch\": 1, \"context\": \"missing\", \"source\": \"bad.md\"}\n\
         {\"subject\": \"s2\", \"label\": \"l2\", \"object\": \"o2\", \"weight\": 1.0}\n\
         {\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"c.md\", \
         \"create\": {\"description\": \"d\"}}\n\
         {\"subject\": \"s3\", \"label\": \"l3\", \"object\": \"o3\", \"weight\": 1.0}\n",
    )
    .expect("fixture must be writable");

    let server = Server::start_with_env(
        "remote-import-midrefusal",
        &[("TAGURU_MAX_BODY_BYTES", "220")],
    );
    let (code, stdout, stderr) = run_cli(
        &["import", "--url", &server.base, file.to_str().unwrap()],
        &[],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("refused"), "{stderr}");
    assert!(stderr.contains("never sent"), "{stderr}");
    assert!(stderr.contains("landed durably"), "{stderr}");

    let (status, _) = server.call("GET", "/contexts/a", None);
    assert_eq!(status, 200, "the batch before the refusal must have landed");
    let (status, _) = server.call("GET", "/contexts/c", None);
    assert_eq!(
        status, 404,
        "the batch after the refusal must never have been sent"
    );

    let _ = std::fs::remove_dir_all(&batches);
}

/// ADR 0002 §5/§7: a bare `--url` with no value, and a URL carrying
/// userinfo, are both usage errors caught before any request leaves
/// the process — no server needs to be running for either check.
#[test]
fn a_userinfo_url_or_a_valueless_url_flag_is_a_usage_error() {
    let (code, _stdout, stderr) = run_cli(&["import", "--url"], &[]);
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("--url needs a server URL"), "{stderr}");

    let batches = batch_dir("remote-import-userinfo");
    let file = batches.join("seed.jsonl");
    std::fs::write(
        &file,
        "{\"taguru_batch\": 1, \"context\": \"a\", \"source\": \"a.md\", \"create\": {}}\n",
    )
    .expect("fixture must be writable");
    let (code, _stdout, stderr) = run_cli(
        &[
            "import",
            "--url",
            "http://user:tok@127.0.0.1:9",
            file.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("TAGURU_API_TOKEN"), "{stderr}");

    let _ = std::fs::remove_dir_all(&batches);
}

/// A malformed `--url`, or one using a scheme `ureq` cannot speak
/// (`file://`, `ftp://` — `url::Url::parse` alone accepts both), is a
/// usage error caught up front — exit 2, never the exit-1
/// "connection lost" shape a request that actually reached `ureq`'s
/// transport would produce. Both checked before any file is even
/// read, so a bad `--url` cannot slip past an otherwise-empty batch
/// stream and exit 0.
#[test]
fn a_malformed_or_non_http_url_is_a_usage_error_not_a_transport_failure() {
    let batches = batch_dir("remote-import-bad-url");
    let file = batches.join("seed.jsonl");
    std::fs::write(
        &file,
        "{\"taguru_batch\": 1, \"context\": \"a\", \"source\": \"a.md\", \"create\": {}}\n",
    )
    .expect("fixture must be writable");

    for (url, needle) in [
        ("not a url at all", "is not a usable base URL"),
        ("file:///etc/passwd", "only supports http/https"),
        ("ftp://127.0.0.1:9/", "only supports http/https"),
    ] {
        let (code, _stdout, stderr) =
            run_cli(&["import", "--url", url, file.to_str().unwrap()], &[]);
        assert_eq!(code, 2, "{url}: {stderr}");
        assert!(stderr.contains(needle), "{url}: {stderr}");
    }

    let _ = std::fs::remove_dir_all(&batches);
}

/// ADR 0002 §5: `--no-embed` only means something offline; combined
/// with `--url` it is a usage error caught before any request leaves
/// the process — proven by pointing `--url` at a port nothing listens
/// on and still getting the usage-error exit code, not a connection
/// failure.
#[test]
fn no_embed_combined_with_url_is_a_usage_error_before_any_request() {
    let batches = batch_dir("remote-import-noembed");
    let file = batches.join("seed.jsonl");
    std::fs::write(
        &file,
        "{\"taguru_batch\": 1, \"context\": \"a\", \"source\": \"a.md\", \"create\": {}}\n",
    )
    .expect("fixture must be writable");
    let (code, _stdout, stderr) = run_cli(
        &[
            "import",
            "--url",
            "http://127.0.0.1:1",
            "--no-embed",
            file.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("--no-embed"), "{stderr}");
    assert!(stderr.contains("--url"), "{stderr}");

    let _ = std::fs::remove_dir_all(&batches);
}

/// The bearer rides the same environment variables the server reads
/// (ADR 0002 §7): present, the import succeeds; absent, the server's
/// own 401 surfaces as the refusal it is — re-importing is idempotent,
/// so the second (token-less) attempt is safe to make against the same
/// already-landed source.
#[test]
fn the_environment_token_authenticates_and_its_absence_is_the_servers_401() {
    let server = Server::start_with_env("remote-import-auth", &[("TAGURU_API_TOKEN", "sekrit")]);

    let batches = batch_dir("remote-import-auth");
    let file = batches.join("seed.jsonl");
    std::fs::write(
        &file,
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
         \"create\": {\"description\": \"d\"}}\n",
    )
    .expect("fixture must be writable");

    let (code, stdout, stderr) = run_cli(
        &["import", "--url", &server.base, file.to_str().unwrap()],
        &[("TAGURU_API_TOKEN", "sekrit")],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let (code, _stdout, stderr) = run_cli(
        &["import", "--url", &server.base, file.to_str().unwrap()],
        &[],
    );
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("401"), "{stderr}");

    let _ = std::fs::remove_dir_all(&batches);
}

/// A minimal stub answering exactly two requests in order: `GET
/// /health` (a deliberately mismatched version, to trigger the ADR
/// 0002 §10 skew warning) then a 500 for whatever comes next — the
/// same shape `remote_export.rs`'s own stub uses, since only the
/// preflight (which runs before any real request) is under test here.
fn spawn_mismatched_health_stub() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let responses = [
            ("HTTP/1.1 200 OK", r#"{"status":"ok","version":"0.1.0"}"#),
            (
                "HTTP/1.1 500 Internal Server Error",
                r#"{"status":"error","code":"internal","error":"stub"}"#,
            ),
        ];
        for (status_line, body) in responses {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut buffer = [0u8; 2048];
            let _ = stream.read(&mut buffer);
            let response = format!(
                "{status_line}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        }
    });
    format!("http://{addr}")
}

/// A server whose `/health` names a different minor version prints the
/// skew warning exactly once, on stderr — the positive case the other
/// tests' "no warning on a matching build" assertions cannot exercise.
#[test]
fn a_mismatched_server_version_prints_the_skew_warning_once() {
    let base = spawn_mismatched_health_stub();
    let batches = batch_dir("remote-import-skew");
    let file = batches.join("seed.jsonl");
    std::fs::write(
        &file,
        "{\"taguru_batch\": 1, \"context\": \"a\", \"source\": \"a.md\", \"create\": {}}\n",
    )
    .expect("fixture must be writable");
    let (_code, _stdout, stderr) =
        run_cli(&["import", "--url", &base, file.to_str().unwrap()], &[]);
    assert_eq!(
        stderr.matches("warning:").count(),
        1,
        "a version mismatch must warn exactly once: {stderr}"
    );
    assert!(stderr.contains("0.1.0"), "{stderr}");

    let _ = std::fs::remove_dir_all(&batches);
}

/// A scripted stub: `GET /health` (matching version, no skew warning),
/// then `POST /import` answers 413 for the whole (2-batch) chunk,
/// 200 for the first half after the client halves it at the batch
/// boundary, then drops the connection on the second half without
/// answering at all — proving both halves of ADR 0002 §9's 413
/// adaptation and §8's "connection lost after chunk N/M" wording,
/// against real accept()/read()/write() rather than a fake `Api`.
fn spawn_413_then_drop_stub() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        // 1) GET /health — matching version, so no skew warning muddies
        //    the assertions below.
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buffer = [0u8; 2048];
            let _ = stream.read(&mut buffer);
            let body = format!(
                r#"{{"status":"ok","version":"{}"}}"#,
                env!("CARGO_PKG_VERSION")
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        }
        // 2) POST /import — the whole (2-batch) chunk is too large.
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buffer = [0u8; 8192];
            let _ = stream.read(&mut buffer);
            let body = r#"{"status":"error","code":"payload_too_large","error":"stub: too large"}"#;
            let response = format!(
                "HTTP/1.1 413 Payload Too Large\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        }
        // 3) POST /import — the first half, after halving at the batch
        //    boundary, lands.
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buffer = [0u8; 8192];
            let _ = stream.read(&mut buffer);
            let body = r#"{"status":"ok","result":{"batches":[{"context":"a","source":"a.md","created":true,"retracted":0,"associations":1,"aliases":0,"passage_stored":false,"passage_dropped":false,"questions_stored":0,"questions_dropped":0,"sections_stored":0,"sections_dropped":0,"locators_stored":0,"locators_dropped":0,"association_paragraphs_dropped":0}]},"time":0.0}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
        }
        // 4) POST /import — the second half's connection drops with no
        //    response at all.
        if let Ok((stream, _)) = listener.accept() {
            drop(stream);
        }
    });
    format!("http://{addr}")
}

#[test]
fn a_413_halves_at_the_batch_boundary_and_a_lost_connection_names_the_resume() {
    let base = spawn_413_then_drop_stub();
    let batches = batch_dir("remote-import-413drop");
    let file = batches.join("seed.jsonl");
    std::fs::write(
        &file,
        "{\"taguru_batch\": 1, \"context\": \"a\", \"source\": \"a.md\", \
         \"create\": {\"description\": \"d\"}}\n\
         {\"subject\": \"s\", \"label\": \"l\", \"object\": \"o\", \"weight\": 1.0}\n\
         {\"taguru_batch\": 1, \"context\": \"b\", \"source\": \"b.md\", \
         \"create\": {\"description\": \"d\"}}\n\
         {\"subject\": \"s2\", \"label\": \"l2\", \"object\": \"o2\", \"weight\": 1.0}\n",
    )
    .expect("fixture must be writable");

    let (code, stdout, stderr) = run_cli(&["import", "--url", &base, file.to_str().unwrap()], &[]);
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        !stderr.contains("warning:"),
        "the matching health version must not warn: {stderr}"
    );
    assert!(stdout.contains("chunk 1/2"), "stdout: {stdout}");
    assert!(
        stderr.contains("connection lost after chunk 1/2"),
        "{stderr}"
    );
    assert!(stderr.contains("--dry-run"), "{stderr}");

    let _ = std::fs::remove_dir_all(&batches);
}
