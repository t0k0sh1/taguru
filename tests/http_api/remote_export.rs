//! `taguru export --url`: remote export via `GET /contexts` and `GET
//! /groups` enumeration (issue #245, ADR 0002 §6/§9) — the counterpart
//! to the local path `tests/cli.rs`'s own round-trip test pins.
//!
//! Pagination itself (the keyset `after`/`limit` walk) is exercised at
//! the unit level instead of here — `src/remote.rs`'s
//! `list_names_paged` tests walk several small pages directly.
//! Provisioning 1000+ contexts in an integration test to cross the
//! server's real page ceiling would be a dishonest cost for the same
//! coverage, and a hidden CLI page-size flag added just to make that
//! provisioning cheaper would be surface added for the tests' sake
//! alone.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;

use serde_json::json;

use crate::support::{Server, batch_dir, run_cli, run_import};

/// Every file directly under `dir`, name → bytes — the same
/// byte-identical comparison `tests/http_api/replication.rs` uses to
/// pin a restored bucket against its source.
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

/// The full remote walk — enumerate, fetch every context and group,
/// write them under --out — must land the exact same files a local
/// export of the same data directory would, byte for byte. One of the
/// two contexts carries a non-ASCII name, exercising `Api::url`'s
/// percent-encoding and `file_stem`'s stem-encoding on the same real
/// request/response round trip a live remote export makes.
#[test]
fn a_full_remote_export_matches_the_local_export_of_the_same_directory() {
    let batches = batch_dir("remote-export-full");
    let file = batches.join("seed.jsonl");
    std::fs::write(
        &file,
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
         \"create\": {\"description\": \"酒蔵の知識\"}}\n\
         {\"passage\": \"青嶺酒造の紹介。\\n\\n代表銘柄は青嶺。\"}\n\
         {\"paragraph\": 0, \"section\": \"概要\"}\n\
         {\"subject\": \"青嶺酒造\", \"label\": \"代表銘柄\", \"object\": \"青嶺\", \
          \"weight\": 1.0, \"paragraph\": 1}\n\
         {\"alias\": \"Aomine\", \"canonical\": \"青嶺酒造\", \"kind\": \"concept\"}\n\
         {\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"b.md\"}\n\
         {\"subject\": \"青嶺酒造\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 2.0}\n\
         {\"taguru_batch\": 1, \"context\": \"酒蔵\", \"source\": \"c.md\", \
          \"create\": {\"description\": \"蔵元台帳\"}}\n\
         {\"subject\": \"白鶴\", \"label\": \"所在地\", \"object\": \"神戸\", \"weight\": 1.0}\n\
         {\"taguru_group\": 1, \"name\": \"kura\", \"description\": \"蔵まとめ\", \
          \"contexts\": [\"sake\", \"酒蔵\"]}\n",
    )
    .expect("fixture must be writable");

    let data_dir = crate::support::common::scratch_dir("remote-export-full");
    let (code, stdout, stderr) = run_import(&data_dir, &[file.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let server = Server::start_on("remote-export-full", data_dir);

    let out_remote = batches.join("out-remote");
    let (code, stdout, stderr) = run_cli(
        &[
            "export",
            "--url",
            &server.base,
            "--out",
            out_remote.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(
        stderr.matches("not a point-in-time snapshot").count(),
        1,
        "the consistency note must print exactly once: {stderr}"
    );
    assert!(
        !stderr.contains("warning:"),
        "same-build CLI and server must not trip the version-skew warning: {stderr}"
    );
    assert!(
        stdout.contains("and 1 of 1 group(s)"),
        "a full export must also report the group: {stdout}"
    );

    // Reclaim the data directory (releasing its lock) and re-export it
    // through the well-established local path for comparison.
    let data_dir = server.stop_gracefully();

    let out_local = batches.join("out-local");
    let (code, stdout, stderr) = run_cli(
        &["export", "--out", out_local.to_str().unwrap()],
        &[("TAGURU_DATA_DIR", data_dir.to_str().unwrap())],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    assert_eq!(
        dir_contents(&out_remote),
        dir_contents(&out_local),
        "a remote export must land byte-identical files to the local export of the same data"
    );

    let _ = std::fs::remove_dir_all(&batches);
}

/// A subset export (explicit CONTEXT names) never enumerates `GET
/// /groups` and writes no `.group.jsonl` file — the same rule
/// `run_local` already follows, since a group's truth can span
/// contexts a subset does not carry.
#[test]
fn a_subset_remote_export_skips_enumeration_and_writes_no_groups() {
    let server = Server::start("remote-export-subset");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));
    server.ok("PUT", "/contexts/bunko", Some(json!({"description": "d"})));
    server.ok(
        "PUT",
        "/groups/kura",
        Some(json!({"contexts": ["sake", "bunko"]})),
    );

    let out = std::env::temp_dir().join(format!(
        "taguru-remote-export-subset-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&out);
    let (code, stdout, stderr) = run_cli(
        &[
            "export",
            "--url",
            &server.base,
            "--out",
            out.to_str().unwrap(),
            "sake",
        ],
        &[],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(!stdout.contains("group(s)"), "{stdout}");

    let contents = dir_contents(&out);
    assert_eq!(contents.len(), 1, "{contents:?}");
    assert!(contents.contains_key("sake.jsonl"), "{contents:?}");

    // The output alone cannot tell "never enumerated" apart from
    // "enumerated, then discarded" — the request-count counter can.
    let (_, metrics_body) = server.call("GET", "/metrics", None);
    let metrics_text = metrics_body
        .as_str()
        .expect("metrics body is text, not JSON");
    assert!(
        !metrics_text.contains("route=\"/groups\""),
        "a subset export must never call GET /groups: {metrics_text}"
    );

    let _ = std::fs::remove_dir_all(&out);
}

/// The bearer rides the same environment variables the server reads
/// (ADR 0002 §7): present, the export succeeds; absent, the server's
/// own 401 surfaces as the per-item failure it is, not a hang or a
/// panic.
#[test]
fn the_environment_token_authenticates_and_its_absence_is_the_servers_401() {
    let server = Server::start_with_env("remote-export-auth", &[("TAGURU_API_TOKEN", "sekrit")]);
    let (status, _) = server.call_with_token(
        "PUT",
        "/contexts/sake",
        Some(json!({"description": "d"})),
        Some("sekrit"),
    );
    assert_eq!(status, 200);

    let out = crate::support::common::scratch_dir("remote-export-auth");

    let (code, stdout, stderr) = run_cli(
        &[
            "export",
            "--url",
            &server.base,
            "--out",
            out.to_str().unwrap(),
        ],
        &[("TAGURU_API_TOKEN", "sekrit")],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let (code, _stdout, stderr) = run_cli(
        &[
            "export",
            "--url",
            &server.base,
            "--out",
            out.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("401"), "{stderr}");

    let _ = std::fs::remove_dir_all(&out);
}

/// Enumeration and fetch are independent per item: a name that 404s
/// counts as a per-item failure, exactly like the local path's "no
/// such context", and the rest of the run still lands.
#[test]
fn an_unknown_context_counts_as_a_failure_and_the_rest_still_lands() {
    let server = Server::start("remote-export-unknown");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    let out = std::env::temp_dir().join(format!(
        "taguru-remote-export-unknown-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&out);
    let (code, stdout, stderr) = run_cli(
        &[
            "export",
            "--url",
            &server.base,
            "--out",
            out.to_str().unwrap(),
            "sake",
            "nope",
        ],
        &[],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("context 'nope'"), "{stderr}");
    assert!(stderr.contains("404"), "{stderr}");
    assert!(stdout.contains("1 of 2 context(s)"), "{stdout}");
    let contents = dir_contents(&out);
    assert!(contents.contains_key("sake.jsonl"), "{contents:?}");

    let _ = std::fs::remove_dir_all(&out);
}

/// A full export against a server with no contexts refuses the same
/// way the local path refuses an empty data directory — naming the
/// server instead of a directory.
#[test]
fn an_empty_server_refuses_the_full_export() {
    let server = Server::start("remote-export-empty");
    let out = crate::support::common::scratch_dir("remote-export-empty");
    let (code, _stdout, stderr) = run_cli(
        &[
            "export",
            "--url",
            &server.base,
            "--out",
            out.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("holds no contexts"), "{stderr}");
}

/// ADR 0002 §5/§7: a bare `--url` with no value, and a URL carrying
/// userinfo, are both usage errors caught before any request leaves
/// the process — no server needs to be running for either check.
#[test]
fn a_userinfo_url_or_a_valueless_url_flag_is_a_usage_error() {
    let (code, _stdout, stderr) = run_cli(&["export", "--url"], &[]);
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("--url needs a server URL"), "{stderr}");

    let out = std::env::temp_dir().join(format!(
        "taguru-remote-export-userinfo-{}",
        std::process::id()
    ));
    let (code, _stdout, stderr) = run_cli(
        &[
            "export",
            "--url",
            "http://user:tok@127.0.0.1:9",
            "--out",
            out.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("TAGURU_API_TOKEN"), "{stderr}");

    // Issue #751: a base no request could leave on — unparseable, or a
    // scheme ureq does not speak — is the same upfront usage error
    // `import --url` already gives it, caught before the snapshot note
    // prints (it used to be an exit-1 runtime failure after it).
    let (code, _stdout, stderr) = run_cli(
        &[
            "export",
            "--url",
            "not a url at all",
            "--out",
            out.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("is not a usable base URL"), "{stderr}");
    assert!(
        !stderr.contains("not a point-in-time snapshot"),
        "the refusal must land before the note: {stderr}"
    );

    let (code, _stdout, stderr) = run_cli(
        &[
            "export",
            "--url",
            "ftp://127.0.0.1:9",
            "--out",
            out.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 2, "{stderr}");
    assert!(
        stderr.contains("--url only supports http/https"),
        "{stderr}"
    );
}

/// Issue #751: a FULL export owns `--out`'s `*.jsonl` files — one left
/// by an earlier export whose context or group no longer exists on the
/// server is removed, so a directory import can never resurrect the
/// deleted entity. A subset export never prunes, and a file that is
/// not a stream (`notes.txt`) is never touched by either.
#[test]
fn a_full_remote_export_removes_stale_stream_files() {
    let server = Server::start("remote-export-prune");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    let out = crate::support::common::scratch_dir("remote-export-prune");
    std::fs::create_dir_all(&out).expect("out dir must be creatable");
    std::fs::write(out.join("zombie.jsonl"), b"{}").expect("stale file must be writable");
    std::fs::write(out.join("zombie.group.jsonl"), b"{}").expect("stale file must be writable");
    std::fs::write(out.join("notes.txt"), b"keep me").expect("bystander must be writable");

    // Subset export: writes its slice, removes nothing.
    let (code, stdout, stderr) = run_cli(
        &[
            "export",
            "--url",
            &server.base,
            "--out",
            out.to_str().unwrap(),
            "sake",
        ],
        &[],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(!stdout.contains("removed"), "{stdout}");
    assert!(
        out.join("zombie.jsonl").exists(),
        "a subset export must not prune"
    );

    // Full export: both stale stream files go, each named on stdout.
    let (code, stdout, stderr) = run_cli(
        &[
            "export",
            "--url",
            &server.base,
            "--out",
            out.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert_eq!(
        stdout
            .matches("removed — no longer exists at the source")
            .count(),
        2,
        "{stdout}"
    );
    let contents = dir_contents(&out);
    assert!(contents.contains_key("sake.jsonl"), "{contents:?}");
    assert!(contents.contains_key("notes.txt"), "{contents:?}");
    assert!(!contents.contains_key("zombie.jsonl"), "{contents:?}");
    assert!(!contents.contains_key("zombie.group.jsonl"), "{contents:?}");

    // An unexpected `*.jsonl` entry the prune cannot unlink — forced
    // with a DIRECTORY, which `remove_file` refuses even for root —
    // is a failure the exit code and stderr both carry.
    std::fs::create_dir_all(out.join("undead.jsonl")).expect("dir must be creatable");
    let (code, stdout, stderr) = run_cli(
        &[
            "export",
            "--url",
            &server.base,
            "--out",
            out.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("cannot remove stale"), "{stderr}");

    let _ = std::fs::remove_dir_all(&out);
}

/// Issue #751: a well-formed response naming a DIFFERENT context or
/// group than the one requested is refused, and nothing lands on
/// disk. Import applies each batch/record to its EMBEDDED name,
/// whatever file it rode in on — saved under the wrong name, a later
/// directory import would restore the wrong truth.
#[test]
fn a_response_naming_a_different_context_or_group_is_refused() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let responses = [
            // /health, /version: no skew warning, no schema refusal.
            ("HTTP/1.1 200 OK", r#"{"status":"ok"}"#.to_string()),
            ("HTTP/1.1 200 OK", r#"{}"#.to_string()),
            // GET /contexts, one page then the terminator.
            (
                "HTTP/1.1 200 OK",
                r#"{"result":{"total":1,"contexts":[{"name":"sake"}]}}"#.to_string(),
            ),
            (
                "HTTP/1.1 200 OK",
                r#"{"result":{"total":1,"contexts":[]}}"#.to_string(),
            ),
            // GET /contexts/sake/export: a valid stream — for the
            // WRONG context.
            (
                "HTTP/1.1 200 OK",
                "{\"taguru_batch\":1,\"context\":\"other\",\"source\":\"a.md\",\
                 \"create\":{\"description\":\"d\"}}\n{\"passage\":\"x\"}\n"
                    .to_string(),
            ),
            // GET /groups, one page then the terminator.
            (
                "HTTP/1.1 200 OK",
                r#"{"result":{"total":1,"groups":[{"name":"g"}]}}"#.to_string(),
            ),
            (
                "HTTP/1.1 200 OK",
                r#"{"result":{"total":1,"groups":[]}}"#.to_string(),
            ),
            // GET /groups/g/export: a valid record — for the WRONG
            // group.
            (
                "HTTP/1.1 200 OK",
                r#"{"taguru_group":1,"name":"h","description":"x","contexts":["sake"]}"#
                    .to_string(),
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
    let base = format!("http://{addr}");

    let out = crate::support::common::scratch_dir("remote-export-wrongname");
    let (code, stdout, stderr) = run_cli(
        &["export", "--url", &base, "--out", out.to_str().unwrap()],
        &[],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stderr.contains("the response carries context 'other'"),
        "{stderr}"
    );
    assert!(stderr.contains("the response names group 'h'"), "{stderr}");
    let contents = dir_contents(&out);
    assert!(
        !contents.contains_key("sake.jsonl") && !contents.contains_key("g.group.jsonl"),
        "a wrong-name response must never land on disk: {contents:?}"
    );
    let _ = std::fs::remove_dir_all(&out);
}

/// Issue #751: `GET /groups/{name}/export` answering something that is
/// not a group record — here a well-formed JSON object with a 200
/// status — is a per-item failure, and nothing lands on disk for it.
/// "Parses as JSON" alone used to let any such body through as a
/// `.group.jsonl` file reporting 0 members.
#[test]
fn a_group_export_response_that_is_not_a_group_record_is_refused() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let responses = [
            // /health, /version: no skew warning, no schema refusal.
            ("HTTP/1.1 200 OK", r#"{"status":"ok"}"#.to_string()),
            ("HTTP/1.1 200 OK", r#"{}"#.to_string()),
            // GET /contexts, first page then the terminating empty one.
            (
                "HTTP/1.1 200 OK",
                r#"{"result":{"total":1,"contexts":[{"name":"sake"}]}}"#.to_string(),
            ),
            (
                "HTTP/1.1 200 OK",
                r#"{"result":{"total":1,"contexts":[]}}"#.to_string(),
            ),
            // GET /contexts/sake/export: a real batch stream.
            (
                "HTTP/1.1 200 OK",
                "{\"taguru_batch\":1,\"context\":\"sake\",\"source\":\"a.md\",\
                 \"create\":{\"description\":\"d\"}}\n{\"passage\":\"x\"}\n"
                    .to_string(),
            ),
            // GET /groups, one page then the terminator.
            (
                "HTTP/1.1 200 OK",
                r#"{"result":{"total":1,"groups":[{"name":"g"}]}}"#.to_string(),
            ),
            (
                "HTTP/1.1 200 OK",
                r#"{"result":{"total":1,"groups":[]}}"#.to_string(),
            ),
            // GET /groups/g/export: valid JSON, but no group record.
            ("HTTP/1.1 200 OK", r#"{"status":"ok"}"#.to_string()),
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
    let base = format!("http://{addr}");

    let out = crate::support::common::scratch_dir("remote-export-badgroup");
    let (code, stdout, stderr) = run_cli(
        &["export", "--url", &base, "--out", out.to_str().unwrap()],
        &[],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stderr.contains("group 'g': not a taguru group record"),
        "{stderr}"
    );
    let contents = dir_contents(&out);
    assert!(contents.contains_key("sake.jsonl"), "{contents:?}");
    assert!(
        !contents.contains_key("g.group.jsonl"),
        "a refused response must never land on disk: {contents:?}"
    );
    let _ = std::fs::remove_dir_all(&out);
}

/// A minimal stub answering exactly two requests in order: `GET
/// /health` (a deliberately mismatched version, to trigger the ADR
/// 0002 §10 skew warning) then a 500 for whatever comes next. This
/// test only needs to prove the warning fires — `warn_on_version_skew`
/// runs before any real request the export makes — not that a
/// mismatched-version stub can complete a whole export, which would
/// need a far larger fake server for no extra coverage.
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

/// A server whose `/health` names a different minor version prints
/// the skew warning exactly once, on stderr — the positive case the
/// other tests' "no warning on a matching build" assertions cannot
/// exercise (the unit-level `skew_warning` tests already cover the
/// warning text itself; this proves `export --url` actually calls it).
#[test]
fn a_mismatched_server_version_prints_the_skew_warning_once() {
    let base = spawn_mismatched_health_stub();
    let out = crate::support::common::scratch_dir("remote-export-skew");
    let (_code, _stdout, stderr) = run_cli(
        &["export", "--url", &base, "--out", out.to_str().unwrap()],
        &[],
    );
    assert_eq!(
        stderr.matches("warning:").count(),
        1,
        "a version mismatch must warn exactly once: {stderr}"
    );
    assert!(stderr.contains("0.1.0"), "{stderr}");
}

/// When the groups enumeration itself fails on a full export, the run
/// must end nonzero and the summary must say the enumeration failed —
/// not read like a subset export that never touched groups. Only a
/// stub can force this: the real server always answers `GET /groups`.
/// A failed context fetch rides along, pinning the combined exit rule
/// (either failure kind alone must be enough).
#[test]
fn a_failed_group_enumeration_is_a_failure_the_summary_names() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let responses = [
            // /health: no version key, no skew warning.
            ("HTTP/1.1 200 OK", r#"{"status":"ok"}"#),
            // /version: no schema_formats, which export treats as safe.
            ("HTTP/1.1 200 OK", r#"{}"#),
            // GET /contexts, first page then the terminating empty one.
            (
                "HTTP/1.1 200 OK",
                r#"{"result":{"total":1,"contexts":[{"name":"sake"}]}}"#,
            ),
            ("HTTP/1.1 200 OK", r#"{"result":{"total":1,"contexts":[]}}"#),
            // GET /contexts/sake/export: a per-context failure.
            (
                "HTTP/1.1 500 Internal Server Error",
                r#"{"status":"error","code":"internal","error":"stub"}"#,
            ),
            // GET /groups: the enumeration failure under test.
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
    let base = format!("http://{addr}");

    let out = crate::support::common::scratch_dir("remote-export-groups-fail");
    // Issue #751: without the group list, a stale `.group.jsonl` cannot
    // be told from a live one — a failed enumeration must prune nothing.
    std::fs::create_dir_all(&out).expect("out dir must be creatable");
    std::fs::write(out.join("zombie.group.jsonl"), b"{}").expect("stale file must be writable");
    let (code, stdout, stderr) = run_cli(
        &["export", "--url", &base, "--out", out.to_str().unwrap()],
        &[],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("export: 0 of 1 context(s) and groups: enumeration failed"),
        "{stdout}"
    );
    assert!(stderr.contains("taguru: export: groups:"), "{stderr}");
    assert!(
        out.join("zombie.group.jsonl").exists(),
        "a failed group enumeration must not prune"
    );
    let _ = std::fs::remove_dir_all(&out);
}

/// Issue #753: per-item failures counted independently mid-run — a
/// context whose 200 body is not an export stream, and a group whose
/// own fetch 404s, are each one failure; the group after them still
/// lands, and the summary carries both denominators.
#[test]
fn per_item_failures_count_and_the_rest_still_lands() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let responses = [
            ("HTTP/1.1 200 OK", r#"{"status":"ok"}"#.to_string()),
            ("HTTP/1.1 200 OK", r#"{}"#.to_string()),
            // GET /contexts: one context, then the terminator.
            (
                "HTTP/1.1 200 OK",
                r#"{"result":{"total":1,"contexts":[{"name":"sake"}]}}"#.to_string(),
            ),
            (
                "HTTP/1.1 200 OK",
                r#"{"result":{"total":1,"contexts":[]}}"#.to_string(),
            ),
            // GET /contexts/sake/export: a 200 that is no stream.
            ("HTTP/1.1 200 OK", "not an export stream".to_string()),
            // GET /groups: two groups, then the terminator.
            (
                "HTTP/1.1 200 OK",
                r#"{"result":{"total":2,"groups":[{"name":"g"},{"name":"h"}]}}"#.to_string(),
            ),
            (
                "HTTP/1.1 200 OK",
                r#"{"result":{"total":2,"groups":[]}}"#.to_string(),
            ),
            // GET /groups/g/export: gone between enumeration and fetch.
            (
                "HTTP/1.1 404 Not Found",
                r#"{"status":"error","code":"no_context","error":"no such group"}"#.to_string(),
            ),
            // GET /groups/h/export: the survivor still lands.
            (
                "HTTP/1.1 200 OK",
                r#"{"taguru_group":1,"name":"h","description":"x","contexts":[]}"#.to_string(),
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
    let base = format!("http://{addr}");

    let out = crate::support::common::scratch_dir("remote-export-per-item");
    let (code, stdout, stderr) = run_cli(
        &["export", "--url", &base, "--out", out.to_str().unwrap()],
        &[],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stderr.contains("context 'sake': not a taguru export stream"),
        "{stderr}"
    );
    assert!(stderr.contains("group 'g'"), "{stderr}");
    assert!(
        stdout.contains("export: 0 of 1 context(s) and 1 of 2 group(s)"),
        "{stdout}"
    );
    let contents = dir_contents(&out);
    assert!(!contents.contains_key("sake.jsonl"), "{contents:?}");
    assert!(
        contents.contains_key("h.group.jsonl"),
        "the survivor must land: {contents:?}"
    );
    let _ = std::fs::remove_dir_all(&out);
}

/// Issue #753: an `--out` that cannot be created — its parent is a
/// FILE — is a clean exit-1 refusal naming the path, for the remote
/// path exactly like the local one (`tests/cli.rs`'s twin).
#[test]
fn an_uncreatable_out_directory_refuses_the_remote_export() {
    let server = Server::start("remote-export-outfail");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    let scratch = crate::support::common::scratch_dir("remote-export-outfail");
    std::fs::create_dir_all(&scratch).expect("scratch dir must be creatable");
    let blocker = scratch.join("blocker");
    std::fs::write(&blocker, b"a file where a directory must go").unwrap();
    let out = blocker.join("out");
    let (code, _stdout, stderr) = run_cli(
        &[
            "export",
            "--url",
            &server.base,
            "--out",
            out.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("cannot create"), "{stderr}");
    let _ = std::fs::remove_dir_all(&scratch);
}
