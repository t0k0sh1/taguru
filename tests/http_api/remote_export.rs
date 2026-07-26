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

    let data_dir =
        std::env::temp_dir().join(format!("taguru-remote-export-full-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
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

    let out =
        std::env::temp_dir().join(format!("taguru-remote-export-auth-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);

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
    let out =
        std::env::temp_dir().join(format!("taguru-remote-export-empty-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&out);
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
}
