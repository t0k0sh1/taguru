//! `taguru compact --url`: remote compaction via per-context `POST
//! /contexts/{name}/compact` and the whole-server `POST
//! /maintenance/compact` sweep (issue #246, ADR 0002 §6/§8) — the
//! counterpart to the local path's own pins in `tests/cli.rs`.
//!
//! 503 shed and 409 double-sweep are not reproduced here: the
//! heavy-ops semaphore and the maintenance CAS guard are both
//! saturation conditions this harness cannot force deterministically
//! without racing real threads against the test itself. `Retry-After`
//! display is pinned at the unit level instead — `src/remote.rs`'s
//! `a_shed_response_displays_its_retry_after_header` test.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;

use serde_json::json;

use crate::support::{Server, run_cli, run_import};

/// One context ("sake") carrying a single dead edge: `a.md` states a
/// fact, `b.md` restates the same source with a different fact,
/// retracting the first and leaving dead weight for compact to
/// reclaim — the same fixture shape `tests/cli.rs`'s own
/// `compact_rewrites_a_data_directory_offline` seeds. Returns the
/// `TAGURU_DATA_DIR` path; the caller cleans up its parent.
fn seed_dead_edge(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "taguru-remote-compact-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
    std::fs::write(
        dir.join("a.jsonl"),
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
         \"create\": {\"description\": \"d\"}}\n\
         {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n",
    )
    .expect("fixture must be writable");
    std::fs::write(
        dir.join("b.jsonl"),
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\"}\n\
         {\"subject\": \"蔵\", \"label\": \"銘柄\", \"object\": \"青嶺\", \"weight\": 1.0}\n",
    )
    .expect("fixture must be writable");

    let data = dir.join("data");
    let (code, _stdout, stderr) = run_import(&data, &[dir.join("a.jsonl").to_str().unwrap()]);
    assert_eq!(code, 0, "{stderr}");
    let (code, _stdout, stderr) = run_import(&data, &[dir.join("b.jsonl").to_str().unwrap()]);
    assert_eq!(code, 0, "{stderr}");
    data
}

/// Naming a context explicitly must call `POST
/// /contexts/{name}/compact` and report the exact bytes shed the local
/// path would, for the same original data — proven against a second,
/// identically seeded directory compacted locally, since compaction
/// mutates and a directory compacted once has nothing left to shed on
/// a second run.
#[test]
fn a_remote_per_context_compact_matches_the_local_run_byte_for_byte() {
    let remote_data = seed_dead_edge("bytematch-remote");
    let remote_scratch = remote_data.parent().unwrap().to_path_buf();
    let local_data = seed_dead_edge("bytematch-local");
    let local_scratch = local_data.parent().unwrap().to_path_buf();

    let server = Server::start_on("remote-compact-bytematch", remote_data);
    let (code, remote_stdout, remote_stderr) =
        run_cli(&["compact", "--url", &server.base, "sake"], &[]);
    assert_eq!(code, 0, "stdout: {remote_stdout}\nstderr: {remote_stderr}");
    assert_eq!(
        remote_stderr.matches("compact → ").count(),
        1,
        "the target must print exactly once: {remote_stderr}"
    );
    assert!(!remote_stderr.contains("warning:"), "{remote_stderr}");

    let (code, local_stdout, local_stderr) = run_cli(
        &["compact", "sake"],
        &[("TAGURU_DATA_DIR", local_data.to_str().unwrap())],
    );
    assert_eq!(code, 0, "stdout: {local_stdout}\nstderr: {local_stderr}");

    assert_eq!(
        remote_stdout, local_stdout,
        "a remote per-context compact must report the same bytes shed as the local run"
    );

    drop(server);
    let _ = std::fs::remove_dir_all(&remote_scratch);
    let _ = std::fs::remove_dir_all(&local_scratch);
}

/// No CONTEXT arguments means the whole-server sweep, `POST
/// /maintenance/compact` — never an enumerate-then-call loop over
/// `GET /contexts`, which the request-count metric can prove absent
/// the same way `remote_export.rs`'s subset test proves `GET /groups`
/// absent from a subset export.
#[test]
fn a_full_remote_sweep_hits_maintenance_compact_and_never_enumerates() {
    let data = seed_dead_edge("sweep-noenum");
    let scratch = data.parent().unwrap().to_path_buf();
    let server = Server::start_on("remote-compact-sweep", data);

    let (code, stdout, stderr) = run_cli(&["compact", "--url", &server.base], &[]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("context 'sake':"), "{stdout}");
    assert!(stdout.contains("dead edge(s) shed"), "{stdout}");
    assert!(
        stdout.contains("server sweep rewrote 1 context(s)"),
        "{stdout}"
    );

    let (_, metrics_body) = server.call("GET", "/metrics", None);
    let metrics_text = metrics_body
        .as_str()
        .expect("metrics body is text, not JSON");
    assert!(
        !metrics_text.contains("route=\"/contexts\""),
        "a full sweep must never call GET /contexts: {metrics_text}"
    );

    drop(server);
    let _ = std::fs::remove_dir_all(&scratch);
}

/// `compact --dry-run --url` with no CONTEXT arguments (issue #371)
/// must enumerate `GET /contexts` — never the sweep, and never `POST
/// .../compact` — and must leave every context exactly as compacting
/// it for real afterward still finds the same weight to shed.
#[test]
fn dry_run_url_enumerates_get_contexts_and_never_compacts() {
    let data = seed_dead_edge("dry-run-noenum");
    let scratch = data.parent().unwrap().to_path_buf();
    let server = Server::start_on("remote-compact-dry-run", data);

    let (code, stdout, stderr) = run_cli(&["compact", "--dry-run", "--url", &server.base], &[]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stdout.contains("context 'sake':"), "{stdout}");
    assert!(stdout.contains("dead edge(s)"), "{stdout}");
    assert!(
        !stdout.contains("dead edge(s) shed"),
        "a dry run must not claim anything was shed: {stdout}"
    );
    assert!(stdout.contains("dry run: 1 of 1"), "{stdout}");

    let (_, metrics_body) = server.call("GET", "/metrics", None);
    let metrics_text = metrics_body
        .as_str()
        .expect("metrics body is text, not JSON");
    assert!(
        metrics_text.contains("route=\"/contexts\""),
        "a dry run with no CONTEXT arguments must enumerate GET /contexts: {metrics_text}"
    );
    assert!(
        !metrics_text.contains("route=\"/maintenance/compact\""),
        "a dry run must never call the sweep: {metrics_text}"
    );
    assert!(
        !metrics_text.contains("route=\"/contexts/{name}/compact\""),
        "a dry run must never call POST .../compact: {metrics_text}"
    );

    // Compacting for real afterward sheds exactly what a first-ever
    // run would — proof the dry run above did not already reclaim it.
    let (code, real_stdout, real_stderr) = run_cli(&["compact", "--url", &server.base], &[]);
    assert_eq!(code, 0, "stdout: {real_stdout}\nstderr: {real_stderr}");
    assert!(real_stdout.contains("dead edge(s) shed"), "{real_stdout}");

    drop(server);
    let _ = std::fs::remove_dir_all(&scratch);
}

/// `compact --dry-run --json --url` decodes as one JSON array of the
/// dead-weight rows, same shape the offline path's own JSON test pins
/// in `tests/cli.rs`.
#[test]
fn dry_run_json_url_emits_a_single_parseable_document() {
    let data = seed_dead_edge("dry-run-json");
    let scratch = data.parent().unwrap().to_path_buf();
    let server = Server::start_on("remote-compact-dry-run-json", data);

    let (code, stdout, stderr) = run_cli(
        &["compact", "--dry-run", "--json", "--url", &server.base],
        &[],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let value: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|error| panic!("--dry-run --json must be one JSON document: {error}"));
    let rows = value.as_array().expect("--dry-run --json is an array");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["context"], "sake");
    assert!(rows[0]["dead_edges"].as_u64().unwrap() > 0);

    drop(server);
    let _ = std::fs::remove_dir_all(&scratch);
}

/// `compact --dry-run --json --url NAME` takes the single-context path
/// (`GET /contexts/{name}`) rather than the enumeration one the two
/// tests above cover — a distinct branch in `run_remote_dry_run` with
/// its own request shape, so it needs its own pin.
#[test]
fn dry_run_json_url_with_a_named_context_uses_the_single_context_path() {
    let data = seed_dead_edge("dry-run-json-named");
    let scratch = data.parent().unwrap().to_path_buf();
    let server = Server::start_on("remote-compact-dry-run-json-named", data);

    let (code, stdout, stderr) = run_cli(
        &[
            "compact",
            "--dry-run",
            "--json",
            "--url",
            &server.base,
            "sake",
        ],
        &[],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    let value: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|error| panic!("--dry-run --json must be one JSON document: {error}"));
    let rows = value.as_array().expect("--dry-run --json is an array");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["context"], "sake");
    assert!(rows[0]["dead_edges"].as_u64().unwrap() > 0);

    let (_, metrics_body) = server.call("GET", "/metrics", None);
    let metrics_text = metrics_body
        .as_str()
        .expect("metrics body is text, not JSON");
    assert!(
        metrics_text.contains("route=\"/contexts/{name}\""),
        "a named CONTEXT argument must call GET /contexts/{{name}}, not enumerate: {metrics_text}"
    );
    assert!(
        !metrics_text.contains("route=\"/contexts\""),
        "a named CONTEXT argument must not enumerate contexts: {metrics_text}"
    );
    assert!(
        !metrics_text.contains("route=\"/contexts/{name}/compact\""),
        "--dry-run must not compact the named context: {metrics_text}"
    );

    drop(server);
    let _ = std::fs::remove_dir_all(&scratch);
}

/// An unknown context name must count as a per-item failure — a
/// nonzero exit code and a stderr message naming it — while `--dry-run
/// --json` itself is unaffected by the caller naming a name the
/// server doesn't carry; the same "rest of the run still lands"
/// contract the non-dry-run per-context path already has (see
/// `an_unknown_context_counts_as_a_failure_and_the_rest_still_lands`
/// below).
#[test]
fn dry_run_json_url_reports_an_unknown_context_as_a_failure() {
    let data = seed_dead_edge("dry-run-json-unknown");
    let scratch = data.parent().unwrap().to_path_buf();
    let server = Server::start_on("remote-compact-dry-run-json-unknown", data);

    let (code, stdout, stderr) = run_cli(
        &[
            "compact",
            "--dry-run",
            "--json",
            "--url",
            &server.base,
            "sake",
            "nope",
        ],
        &[],
    );
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("context 'nope'"), "{stderr}");
    let value: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|error| panic!("--dry-run --json must be one JSON document: {error}"));
    let rows = value.as_array().expect("--dry-run --json is an array");
    assert_eq!(
        rows.len(),
        1,
        "the failed name must not appear in the array, but 'sake' still must: {value}"
    );
    assert_eq!(rows[0]["context"], "sake");

    drop(server);
    let _ = std::fs::remove_dir_all(&scratch);
}

/// Enumeration is not in play here (per-context mode never enumerates
/// either), but a name the server does not carry must still surface as
/// a per-item failure — the rest of the run lands regardless, exactly
/// like `export --url`'s own "unknown context" test.
#[test]
fn an_unknown_context_counts_as_a_failure_and_the_rest_still_lands() {
    let data = seed_dead_edge("unknown");
    let scratch = data.parent().unwrap().to_path_buf();
    let server = Server::start_on("remote-compact-unknown", data);

    let (code, stdout, stderr) = run_cli(&["compact", "--url", &server.base, "sake", "nope"], &[]);
    assert_eq!(code, 1, "stdout: {stdout}\nstderr: {stderr}");
    assert!(stderr.contains("context 'nope'"), "{stderr}");
    assert!(stderr.contains("404"), "{stderr}");
    assert!(stdout.contains("context 'sake':"), "{stdout}");
    assert!(stdout.contains("1 of 2 context(s) rewritten"), "{stdout}");

    drop(server);
    let _ = std::fs::remove_dir_all(&scratch);
}

/// ADR 0002 §5/§7: a bare `--url` with no value, and a URL carrying
/// userinfo, are both usage errors caught before any request leaves
/// the process — no server needs to be running for either check.
#[test]
fn a_userinfo_url_or_a_valueless_url_flag_is_a_usage_error() {
    let (code, _stdout, stderr) = run_cli(&["compact", "--url"], &[]);
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("--url needs a server URL"), "{stderr}");

    let (code, _stdout, stderr) = run_cli(
        &["compact", "--url", "http://user:tok@127.0.0.1:9", "sake"],
        &[],
    );
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("TAGURU_API_TOKEN"), "{stderr}");
}

/// The bearer rides the same environment variables the server reads
/// (ADR 0002 §7): present, both remote modes succeed; absent, the
/// server's own 401 surfaces as a failure, never a hang or a panic.
#[test]
fn the_environment_token_authenticates_and_its_absence_is_the_servers_401() {
    let data = seed_dead_edge("auth");
    let scratch = data.parent().unwrap().to_path_buf();
    let server = Server::start_on_with_env(
        "remote-compact-auth",
        data,
        &[("TAGURU_API_TOKEN", "sekrit")],
    );

    let (code, stdout, stderr) = run_cli(
        &["compact", "--url", &server.base, "sake"],
        &[("TAGURU_API_TOKEN", "sekrit")],
    );
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");

    let (code, _stdout, stderr) = run_cli(&["compact", "--url", &server.base, "sake"], &[]);
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("401"), "{stderr}");

    let (code, _stdout, stderr) = run_cli(&["compact", "--url", &server.base], &[]);
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("401"), "{stderr}");

    drop(server);
    let _ = std::fs::remove_dir_all(&scratch);
}

/// `--parallel N`'s remote per-context path must be reordered back to
/// the sequential run's stdout, byte for byte — the same guarantee
/// `tests/cli.rs`'s `compact_parallel_output_matches_the_sequential_run_byte_for_byte`
/// pins locally, proven here over independent HTTP calls instead of
/// independent entry locks.
#[test]
fn remote_parallel_output_matches_the_sequential_remote_run() {
    fn seed(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "taguru-remote-compact-par-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
        // Created out of alphabetical order, same as the local
        // --parallel pin: a sorted result only comes from something
        // that actually sorts it.
        std::fs::write(
            dir.join("a.jsonl"),
            "{\"taguru_batch\": 1, \"context\": \"charlie\", \"source\": \"a.md\", \
             \"create\": {\"description\": \"d\"}}\n\
             {\"subject\": \"s\", \"label\": \"l\", \"object\": \"o1\", \"weight\": 1.0}\n\
             {\"taguru_batch\": 1, \"context\": \"alpha\", \"source\": \"a.md\", \
             \"create\": {\"description\": \"d\"}}\n\
             {\"subject\": \"s\", \"label\": \"l\", \"object\": \"o1\", \"weight\": 1.0}\n\
             {\"taguru_batch\": 1, \"context\": \"bravo\", \"source\": \"a.md\", \
             \"create\": {\"description\": \"d\"}}\n\
             {\"subject\": \"s\", \"label\": \"l\", \"object\": \"o1\", \"weight\": 1.0}\n",
        )
        .expect("fixture must be writable");
        std::fs::write(
            dir.join("b.jsonl"),
            "{\"taguru_batch\": 1, \"context\": \"charlie\", \"source\": \"a.md\"}\n\
             {\"subject\": \"s\", \"label\": \"l\", \"object\": \"o2\", \"weight\": 1.0}\n\
             {\"taguru_batch\": 1, \"context\": \"alpha\", \"source\": \"a.md\"}\n\
             {\"subject\": \"s\", \"label\": \"l\", \"object\": \"o2\", \"weight\": 1.0}\n\
             {\"taguru_batch\": 1, \"context\": \"bravo\", \"source\": \"a.md\"}\n\
             {\"subject\": \"s\", \"label\": \"l\", \"object\": \"o2\", \"weight\": 1.0}\n",
        )
        .expect("fixture must be writable");
        let data = dir.join("data");
        let (code, _stdout, stderr) = run_import(&data, &[dir.join("a.jsonl").to_str().unwrap()]);
        assert_eq!(code, 0, "{stderr}");
        let (code, _stdout, stderr) = run_import(&data, &[dir.join("b.jsonl").to_str().unwrap()]);
        assert_eq!(code, 0, "{stderr}");
        data
    }

    let seq_data = seed("seq");
    let seq_scratch = seq_data.parent().unwrap().to_path_buf();
    let par_data = seed("par");
    let par_scratch = par_data.parent().unwrap().to_path_buf();

    let seq_server = Server::start_on("remote-compact-par-seq", seq_data);
    let (code, sequential_stdout, stderr) = run_cli(
        &[
            "compact",
            "--url",
            &seq_server.base,
            "charlie",
            "alpha",
            "bravo",
        ],
        &[],
    );
    assert_eq!(code, 0, "{stderr}");
    drop(seq_server);

    // The default heavy-op ceiling (2 concurrent) would shed one of
    // three simultaneous compact calls as a 503 — a real server
    // behavior this test isn't after; it's proving the client
    // reorders its own output, not the server's admission control
    // (that's `src/limits.rs`'s and `remote.rs`'s own coverage).
    let par_server = Server::start_on_with_env(
        "remote-compact-par-par",
        par_data,
        &[("TAGURU_MAX_CONCURRENT_HEAVY_OPS", "8")],
    );
    // More workers than contexts, so every worker races for the queue.
    let (code, parallel_stdout, stderr) = run_cli(
        &[
            "compact",
            "--url",
            &par_server.base,
            "--parallel",
            "8",
            "charlie",
            "alpha",
            "bravo",
        ],
        &[],
    );
    assert_eq!(code, 0, "{stderr}");
    drop(par_server);

    assert!(
        sequential_stdout.contains("3 of 3 context(s) rewritten"),
        "{sequential_stdout}"
    );
    assert_eq!(
        sequential_stdout, parallel_stdout,
        "--parallel output must match the sequential remote run byte for byte"
    );

    let _ = std::fs::remove_dir_all(&seq_scratch);
    let _ = std::fs::remove_dir_all(&par_scratch);
}

/// A server whose `/health` names a different minor version prints the
/// skew warning exactly once, on stderr — mirrors `remote_export.rs`'s
/// own positive-case test for the same ADR 0002 §10 preflight.
#[test]
fn a_mismatched_server_version_prints_the_skew_warning_once() {
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
    let base = format!("http://{addr}");

    let (_code, _stdout, stderr) = run_cli(&["compact", "--url", &base], &[]);
    assert_eq!(
        stderr.matches("warning:").count(),
        1,
        "a version mismatch must warn exactly once: {stderr}"
    );
    assert!(stderr.contains("0.1.0"), "{stderr}");
}

/// A sweep against a server with no dead weight to shed is a success,
/// not a refusal — unlike the local path's (and the remote per-context
/// path's) "holds no contexts" rejection, the sweep never enumerates a
/// list to find empty; it just reports that it rewrote nothing.
#[test]
fn an_empty_server_sweep_reports_zero_rewritten() {
    let server = Server::start("remote-compact-empty-sweep");
    server.ok("PUT", "/contexts/sake", Some(json!({"description": "d"})));

    let (code, stdout, stderr) = run_cli(&["compact", "--url", &server.base], &[]);
    assert_eq!(code, 0, "stdout: {stdout}\nstderr: {stderr}");
    assert!(
        stdout.contains("server sweep rewrote 0 context(s)"),
        "{stdout}"
    );
}
