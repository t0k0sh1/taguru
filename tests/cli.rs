//! CLI integration tests: the real binary, real arguments, real exit
//! codes. The serve default is pinned by every test in http_api.rs;
//! this file covers everything that must NOT start a server, plus the
//! configuration file.

use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

#[path = "common/spawn.rs"]
mod common;

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_taguru"))
        .args(args)
        .env_remove("TAGURU_CONFIG")
        .output()
        .expect("binary must run")
}

/// Like [`run`], but setting (rather than removing) environment
/// variables — for exercising `TAGURU_CONFIG` itself.
fn run_with_env(args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    command.args(args);
    for (key, value) in env {
        command.env(key, value);
    }
    command.output().expect("binary must run")
}

/// A scratch directory holding a config file (and doubling as the data
/// directory the file points at). Removed by the caller.
fn write_config(tag: &str, lines: &str) -> (PathBuf, PathBuf) {
    let dir = common::scratch_dir(&format!("cli-{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
    let config = dir.join("taguru.env");
    let data_dir = dir.join("data");
    let text = format!("TAGURU_DATA_DIR={}\n{lines}", data_dir.display());
    std::fs::write(&config, text).expect("config must be writable");
    (dir, config)
}

/// Spawns `taguru --config <file>` with a scrubbed environment plus
/// `extra_env`, waits for the listen line (proof the file supplied the
/// address), then stops it and returns the whole stderr.
fn serve_with_config(config: &std::path::Path, extra_env: &[(&str, &str)]) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    common::scrub_taguru_env(&mut command).args(["--config", &config.display().to_string()]);
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("server must spawn");

    let stdout = child.stdout.take().expect("stdout must be piped");
    common::read_listen_line("cli server", stdout);
    let _ = child.kill();
    let _ = child.wait();
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .expect("stderr must be piped")
        .read_to_string(&mut stderr)
        .expect("stderr must be readable");
    stderr
}

/// Spawns a live server on a free port with a scratch data dir. The
/// caller kills the child and removes the directory.
fn spawn_server(tag: &str) -> (std::process::Child, String, PathBuf) {
    let dir = common::scratch_dir(&format!("cli-{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    common::scrub_taguru_env(&mut command)
        .env("TAGURU_ADDR", "127.0.0.1:0")
        .env("TAGURU_DATA_DIR", dir.join("data"));
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("server must spawn");
    let stdout = child.stdout.take().expect("stdout must be piped");
    let (addr, _) = common::read_listen_line("cli server", stdout);
    (child, addr, dir)
}

#[test]
fn version_subcommand_prints_the_version_and_nothing_else() {
    let output = run(&["version"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout.trim(),
        format!("taguru {}", env!("CARGO_PKG_VERSION"))
    );
    // The old failure mode: any argument silently started the server.
    assert!(!stdout.contains("listening on"));
}

#[test]
fn help_prints_usage_without_starting_a_server() {
    for args in [&["--help"][..], &["-h"][..], &["help"][..]] {
        let output = run(args);
        assert!(output.status.success(), "{args:?}");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("USAGE"), "{args:?}: {stdout}");
        assert!(stdout.contains("TAGURU_DATA_DIR"), "{args:?}: {stdout}");
        assert!(!stdout.contains("listening on"), "{args:?}");
    }
}

#[test]
fn an_unknown_argument_is_refused_with_a_usage_exit() {
    for args in [&["--version"][..], &["nonsense"][..], &["serve", "-x"][..]] {
        let output = run(args);
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("--help"), "{args:?}: {stderr}");
        assert!(
            String::from_utf8_lossy(&output.stdout).is_empty(),
            "{args:?}"
        );
    }
}

#[test]
fn version_refuses_trailing_arguments() {
    let output = run(&["version", "extra"]);
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn a_config_file_supplies_what_the_environment_lacks() {
    // TAGURU_ADDR and TAGURU_DATA_DIR come ONLY from the file; the
    // server reaching its listen line proves both were applied.
    let (dir, config) = write_config("supplies", "TAGURU_ADDR=127.0.0.1:0\n");
    serve_with_config(&config, &[]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_real_environment_beats_the_config_file() {
    // The file names an unbindable address (port 1); the environment
    // overrides it with a working one. Reaching the listen line proves
    // the environment won — and the notice says so.
    let (dir, config) = write_config("envwins", "TAGURU_ADDR=127.0.0.1:1\n");
    let stderr = serve_with_config(&config, &[("TAGURU_ADDR", "127.0.0.1:0")]);
    assert!(
        stderr.contains("TAGURU_ADDR set in the environment"),
        "{stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_unknown_taguru_key_in_the_config_is_flagged_as_a_typo() {
    let (dir, config) = write_config(
        "typo",
        "TAGURU_ADDR=127.0.0.1:0\nTAGURU_CAHCE_BYTES=1048576\n",
    );
    let stderr = serve_with_config(&config, &[]);
    assert!(
        stderr.contains("TAGURU_CAHCE_BYTES is not a variable taguru reads"),
        "{stderr}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_malformed_config_line_refuses_to_boot() {
    let (dir, config) = write_config("malformed", "not a pair\n");
    let output = run(&["--config", &config.display().to_string()]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("line 2"), "{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_config_file_refuses_to_boot() {
    let output = run(&["--config", "/nonexistent/taguru.env"]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("cannot read config"), "{stderr}");
}

#[test]
fn taguru_config_variable_names_the_file_too() {
    let (dir, config) = write_config("viaenv", "TAGURU_ADDR=127.0.0.1:0\n");
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    common::scrub_taguru_env(&mut command).env("TAGURU_CONFIG", &config);
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("server must spawn");
    let stdout = child.stdout.take().expect("stdout must be piped");
    common::read_listen_line("cli server", stdout);
    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn health_prints_the_json_ok_body_against_a_live_server() {
    let (mut child, addr, dir) = spawn_server("health-ok");
    let output = run(&["health", &format!("http://{addr}")]);
    let _ = child.kill();
    let _ = child.wait();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let body: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("health prints the /health JSON body");
    assert_eq!(body["status"], "ok");
    assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn health_derives_its_target_from_taguru_addr() {
    let (mut child, addr, dir) = spawn_server("health-env");
    let output = Command::new(env!("CARGO_BIN_EXE_taguru"))
        .arg("health")
        .env_remove("TAGURU_CONFIG")
        .env("TAGURU_ADDR", &addr)
        .output()
        .expect("binary must run");
    let _ = child.kill();
    let _ = child.wait();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn health_reads_taguru_addr_from_the_config_file() {
    // The documented container shape: TAGURU_ADDR lives in a --config
    // file, serve reads it — and the HEALTHCHECK probe must read the
    // same file, or it asks the built-in default port forever and
    // reports a healthy server unhealthy.
    let (mut child, addr, dir) = spawn_server("health-config");
    let config = dir.join("probe.env");
    std::fs::write(&config, format!("TAGURU_ADDR={addr}\n")).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_taguru"))
        .arg("health")
        .arg("--config")
        .arg(&config)
        .env_remove("TAGURU_CONFIG")
        .env_remove("TAGURU_ADDR")
        .output()
        .expect("binary must run");
    let _ = child.kill();
    let _ = child.wait();
    assert_eq!(
        output.status.code(),
        Some(0),
        "a --config deployment's health check must probe the configured port: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn health_exits_nonzero_when_nothing_listens() {
    // Learn a free port, then release it: a brief race, but nothing
    // is likely to grab this exact port before the probe fires.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    drop(listener);
    let output = run(&["health", &format!("http://{addr}")]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("health"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn health_refuses_trailing_arguments() {
    let output = run(&["health", "http://127.0.0.1:1", "extra"]);
    assert_eq!(output.status.code(), Some(2));
}

// Issue #248 item 1: `--url` is an alias for the positional URL on
// `health`/`calibrate`/`communities` — either names the target, never
// both.

#[test]
fn health_url_flag_works_the_same_as_the_positional_form() {
    let (mut child, addr, dir) = spawn_server("health-url-flag");
    let output = run(&["health", "--url", &format!("http://{addr}")]);
    let _ = child.kill();
    let _ = child.wait();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn health_refuses_both_url_flag_and_positional() {
    let output = run(&[
        "health",
        "--url",
        "http://127.0.0.1:1",
        "http://127.0.0.1:2",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("not both"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn health_url_flag_does_not_swallow_a_following_flag() {
    // --url --help must not treat "--help" as the URL value (a
    // confusing "invalid URL" failure instead of the usage error an
    // operator reaching for the manual actually wants).
    let output = run(&["health", "--url", "--help"]);
    assert_eq!(
        output.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("--url needs a server URL"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn calibrate_url_flag_does_not_swallow_a_following_flag() {
    let dir = std::env::temp_dir().join(format!(
        "taguru-cli-calibrate-urlflag-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
    let probes = dir.join("probes.tsv");
    std::fs::write(&probes, "a paraphrase\texpected\n").expect("probes file must be writable");

    let output = run(&[
        "calibrate",
        "--context",
        "sake",
        "--probes",
        &probes.display().to_string(),
        "--url",
        "--json",
    ]);
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(
        output.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("--url needs a server URL"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn communities_url_flag_does_not_swallow_a_following_flag() {
    let output = run(&["communities", "--context", "sake", "--url", "--json"]);
    assert_eq!(
        output.status.code(),
        Some(2),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("--url needs a server URL"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn calibrate_refuses_both_url_flag_and_positional() {
    let dir = common::scratch_dir("cli-calibrate-both");
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
    let probes = dir.join("probes.tsv");
    std::fs::write(&probes, "a paraphrase\texpected\n").expect("probes file must be writable");

    let output = run(&[
        "calibrate",
        "--context",
        "sake",
        "--probes",
        &probes.display().to_string(),
        "--url",
        "http://127.0.0.1:1",
        "http://127.0.0.1:2",
    ]);
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("not both"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn communities_refuses_both_url_flag_and_positional() {
    let output = run(&[
        "communities",
        "--context",
        "sake",
        "--url",
        "http://127.0.0.1:1",
        "http://127.0.0.1:2",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("not both"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn communities_url_flag_reaches_the_named_server() {
    // Not a full round trip (no context populated) — just proof --url
    // actually routed the request there: a live server answers with a
    // business error (context missing), not a connection failure.
    let (mut child, addr, dir) = spawn_server("communities-url-flag");
    let output = run(&[
        "communities",
        "--context",
        "sake",
        "--url",
        &format!("http://{addr}"),
    ]);
    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&dir);
    assert_ne!(
        output.status.code(),
        Some(2),
        "a reachable server must not be treated as a usage error: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn inspect_verifies_a_directory_and_a_single_image() {
    let dir = common::scratch_dir("cli-inspect");
    std::fs::create_dir_all(&dir).unwrap();
    let mut context = taguru::context::Context::default();
    context
        .associate("青嶺酒造", "代表銘柄", "青嶺", 1.0)
        .unwrap();
    let image = dir.join("sake.ctx");
    std::fs::write(&image, context.to_bytes()).unwrap();

    let output = run(&["inspect", &dir.display().to_string()]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{stdout}");
    assert!(stdout.contains("sake: ok"), "{stdout}");
    assert!(stdout.contains("1 associations"), "{stdout}");
    assert!(stdout.contains("2 concepts"), "{stdout}");
    assert!(stdout.contains("total: 1 contexts"), "{stdout}");
    // A freshly built image carries no dead weight: no retraction, no
    // alias ever removed.
    assert!(stdout.contains("0 dead edge(s) (0.0% dead)"), "{stdout}");
    assert!(stdout.contains("0 unlinked attribution(s)"), "{stdout}");
    assert!(stdout.contains("0 B arena slack"), "{stdout}");
    // `associate` above named no source, so the one edge it created is
    // entirely unsourced weight.
    assert!(
        stdout.contains("1 unsourced edge(s) (weight 1.0)"),
        "{stdout}"
    );
    // "ok" must state HOW MUCH was proven: a current image was
    // checksum-verified, and the line says so.
    assert!(stdout.contains("checksum verified"), "{stdout}");

    let output = run(&["inspect", &image.display().to_string()]);
    assert_eq!(output.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&output.stdout).contains("ok"));

    let _ = std::fs::remove_dir_all(&dir);
}

/// `inspect --json` (issue #371) on the same directory and single
/// image as the human-readable test above — one JSON document, the
/// same facts (context stats, dead-weight numbers, the total footer),
/// nothing printed alongside it.
#[test]
fn inspect_json_reports_directory_and_image_stats_as_one_document() {
    let dir = common::scratch_dir("cli-inspect-json");
    std::fs::create_dir_all(&dir).unwrap();
    let mut context = taguru::context::Context::default();
    context
        .associate("青嶺酒造", "代表銘柄", "青嶺", 1.0)
        .unwrap();
    let image = dir.join("sake.ctx");
    std::fs::write(&image, context.to_bytes()).unwrap();

    let output = run(&["inspect", "--json", &dir.display().to_string()]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|error| panic!("--json must be one JSON document: {error}"));
    assert_eq!(report["kind"], "directory");
    assert_eq!(report["corrupt"], 0);
    let row = &report["contexts"][0];
    assert_eq!(row["name"], "sake");
    assert_eq!(row["status"], "ok");
    assert_eq!(row["associations"], 1);
    assert_eq!(row["concepts"], 2);
    assert_eq!(row["dead_edges"], 0);
    assert_eq!(row["dead_ratio"], 0.0);
    assert_eq!(row["arena_slack"], 0);
    assert_eq!(row["unsourced_edges"], 1);
    assert!(
        row["generation"]
            .as_str()
            .unwrap()
            .contains("checksum verified")
    );
    // Directory scan sidecars: absent for a bare single-file inspect,
    // present here.
    assert!(row["wal_bytes"].is_u64());
    assert_eq!(report["total"]["contexts"], 1);

    let output = run(&["inspect", "--json", &image.display().to_string()]);
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|error| panic!("--json must be one JSON document: {error}"));
    assert_eq!(report["kind"], "image");
    let row = &report["contexts"][0];
    assert_eq!(row["status"], "ok");
    assert_eq!(row["associations"], 1);
    // A bare image inspect never touches the WAL/passages sidecars.
    assert!(row.get("wal_bytes").is_none(), "{row}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn inspect_fails_an_image_whose_bytes_rotted_in_place() {
    // The backup-verification case the checksum footer exists for: one
    // flipped bit in a stored name leaves the image structurally
    // perfect — every id in range, every chain intact — so before the
    // footer this passed inspection and loaded as truth. Now it must
    // fail, loudly, BEFORE a restore spends it.
    let dir = common::scratch_dir("cli-bitrot");
    std::fs::create_dir_all(&dir).unwrap();
    let mut context = taguru::context::Context::default();
    context.associate("i", "likes", "apple", 1.0).unwrap();
    let mut image = context.to_bytes();
    let last_arena_byte = image.len() - 5; // the 4-byte footer follows
    image[last_arena_byte] ^= 0x01;
    std::fs::write(dir.join("sake.ctx"), &image).unwrap();

    let output = run(&["inspect", &dir.display().to_string()]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(1), "{stdout}");
    assert!(stdout.contains("CORRUPT image"), "{stdout}");
    assert!(stdout.contains("checksum mismatch"), "{stdout}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn inspect_flags_a_corrupt_image_and_a_corrupt_wal() {
    let dir = common::scratch_dir("cli-corrupt");
    std::fs::create_dir_all(&dir).unwrap();

    // A truncated/garbage image must fail the whole inspection.
    std::fs::write(dir.join("bad.ctx"), b"not an image").unwrap();
    let output = run(&["inspect", &dir.display().to_string()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("CORRUPT image"),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );

    // A healthy image whose WAL does not parse: the log holds
    // acknowledged writes that exist nowhere else, so this fails too.
    let context = taguru::context::Context::default();
    std::fs::write(dir.join("sake.ctx"), context.to_bytes()).unwrap();
    std::fs::write(dir.join("sake.wal.jsonl"), b"not json\n").unwrap();
    std::fs::remove_file(dir.join("bad.ctx")).unwrap();
    let output = run(&["inspect", &dir.display().to_string()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("CORRUPT WAL"),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );

    // Same strictness for the passage store: its snapshot holds
    // acknowledged text, so garbage there is a failure, not a shrug.
    std::fs::remove_file(dir.join("sake.wal.jsonl")).unwrap();
    std::fs::write(dir.join("sake.passages.bin"), b"not a snapshot").unwrap();
    let output = run(&["inspect", &dir.display().to_string()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("CORRUPT passages"),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The three corrupt statuses `--json` can report for a context row —
/// `corrupt_image`/`corrupt_wal`/`corrupt_passages` — each with the
/// error text and the overall `corrupt` count, over the same fixture
/// sequence `inspect_flags_a_corrupt_image_and_a_corrupt_wal` pins for
/// the human-readable path.
#[test]
fn inspect_json_reports_the_three_corrupt_statuses() {
    let dir = common::scratch_dir("cli-corrupt-json");
    std::fs::create_dir_all(&dir).unwrap();

    std::fs::write(dir.join("bad.ctx"), b"not an image").unwrap();
    let output = run(&["inspect", "--json", &dir.display().to_string()]);
    assert_eq!(output.status.code(), Some(1));
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|error| panic!("--json must be one JSON document: {error}"));
    assert_eq!(report["contexts"][0]["status"], "corrupt_image");
    assert!(report["contexts"][0]["error"].as_str().is_some());
    assert_eq!(report["corrupt"], 1);
    assert!(report["contexts"][0].get("associations").is_none());

    let context = taguru::context::Context::default();
    std::fs::write(dir.join("sake.ctx"), context.to_bytes()).unwrap();
    std::fs::write(dir.join("sake.wal.jsonl"), b"not json\n").unwrap();
    std::fs::remove_file(dir.join("bad.ctx")).unwrap();
    let output = run(&["inspect", "--json", &dir.display().to_string()]);
    assert_eq!(output.status.code(), Some(1));
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|error| panic!("--json must be one JSON document: {error}"));
    assert_eq!(report["contexts"][0]["status"], "corrupt_wal");

    std::fs::remove_file(dir.join("sake.wal.jsonl")).unwrap();
    std::fs::write(dir.join("sake.passages.bin"), b"not a snapshot").unwrap();
    let output = run(&["inspect", "--json", &dir.display().to_string()]);
    assert_eq!(output.status.code(), Some(1));
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|error| panic!("--json must be one JSON document: {error}"));
    assert_eq!(report["contexts"][0]["status"], "corrupt_passages");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn inspect_reports_a_torn_wal_tail_without_healing_it() {
    // Fix 3: a directory audit must never mutate what it audits. A WAL
    // whose last record was cut short by a crash mid-append is not
    // corruption — the server heals it on its next load — so inspect
    // reports it as a NOTE, still exits 0, and (the decisive part)
    // leaves the torn bytes on disk untouched. This is the read-only
    // guarantee that separates `inspect` from a boot-time replay.
    let dir = common::scratch_dir("cli-torn");
    std::fs::create_dir_all(&dir).unwrap();

    // A healthy image at watermark 0 — the WAL below carries the writes.
    let context = taguru::context::Context::default();
    std::fs::write(dir.join("sake.ctx"), context.to_bytes()).unwrap();

    // One complete acknowledged record, then a fragment with no closing
    // newline: exactly the shape a crash leaves after O_APPEND wrote
    // part of the next line. This is the same on-disk JSON the server's
    // replay reads, hand-written here the way wal.rs's own torn-tail
    // tests are.
    let wal = dir.join("sake.wal.jsonl");
    let mut bytes = br#"{"seq":1,"op":"associate","subject":"a","label":"likes","object":"apple","weight":1.0}"#
        .to_vec();
    bytes.push(b'\n');
    bytes.extend_from_slice(br#"{"seq":2,"op":"associate","subject":"b"#);
    std::fs::write(&wal, &bytes).unwrap();

    let output = run(&["inspect", &dir.display().to_string()]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        output.status.code(),
        Some(0),
        "a torn tail is a crash artifact, not a failure: {stdout}"
    );
    assert!(stdout.contains("sake: ok"), "{stdout}");
    assert!(
        stdout.contains("NOTE"),
        "the torn tail must be reported: {stdout}"
    );
    assert!(stdout.contains("WAL torn tail"), "{stdout}");
    assert!(
        stdout.contains("1 pending"),
        "the one complete record decoded and counts as pending: {stdout}"
    );

    // The decisive read-only check: inspect left the file byte-for-byte
    // as written, torn fragment and all. A boot replay would have
    // truncated it back to the last newline.
    assert_eq!(
        std::fs::read(&wal).unwrap(),
        bytes,
        "inspect must not heal (truncate) the WAL it audits"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn inspect_reports_a_torn_import_marker_without_failing() {
    // A surviving batch-open marker means an import stopped between
    // its four separately-durable steps: every store parses clean, so
    // the marker is the only witness. inspect must SAY so (with the
    // repair) yet exit 0 — the bytes are intact; the tear has a
    // documented fix.
    let dir = common::scratch_dir("cli-import-marker");
    std::fs::create_dir_all(&dir).unwrap();
    let context = taguru::context::Context::default();
    std::fs::write(dir.join("sake.ctx"), context.to_bytes()).unwrap();
    // The marker as the server writes it: {stem}.{source-hash}.importing,
    // the (context, source) pair in the content. The exact hash is
    // irrelevant to reporting — content is what gets read.
    std::fs::write(
        dir.join("sake.00000000deadbeef.importing"),
        br#"{"context":"sake","source":"doc-1"}"#,
    )
    .unwrap();
    // A marker whose context is gone: noted, not warned — the server's
    // next boot removes it.
    std::fs::write(
        dir.join("ghost.00000000deadbeef.importing"),
        br#"{"context":"ghost","source":"doc-9"}"#,
    )
    .unwrap();

    let output = run(&["inspect", &dir.display().to_string()]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{stdout}");
    assert!(
        stdout.contains("WARNING") && stdout.contains("doc-1"),
        "{stdout}"
    );
    assert!(stdout.contains("never completed"), "{stdout}");
    assert!(
        stdout.contains("re-import") || stdout.contains("retract"),
        "the repair must be named: {stdout}"
    );
    assert!(
        stdout.contains("no longer exists here"),
        "the moot marker gets its NOTE: {stdout}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// A torn WAL tail and a surviving import marker (both `inspect_reports_
/// a_torn_wal_tail_without_healing_it` and `inspect_reports_a_torn_
/// import_marker_without_failing`'s own fixtures) surface as structured
/// notices — a per-context `note`/`wal_torn_tail` on the context row,
/// and top-level `warning`/`incomplete_import` and `note`/
/// `orphan_import_marker` entries — all while `--json` still exits 0.
#[test]
fn inspect_json_reports_torn_tail_and_import_marker_notices() {
    let dir = std::env::temp_dir().join(format!(
        "taguru-cli-inspect-json-notices-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let context = taguru::context::Context::default();
    std::fs::write(dir.join("sake.ctx"), context.to_bytes()).unwrap();
    let mut bytes = br#"{"seq":1,"op":"associate","subject":"a","label":"likes","object":"apple","weight":1.0}"#
        .to_vec();
    bytes.push(b'\n');
    bytes.extend_from_slice(br#"{"seq":2,"op":"associate","subject":"b"#);
    std::fs::write(dir.join("sake.wal.jsonl"), &bytes).unwrap();
    std::fs::write(
        dir.join("sake.00000000deadbeef.importing"),
        br#"{"context":"sake","source":"doc-1"}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("ghost.00000000deadbeef.importing"),
        br#"{"context":"ghost","source":"doc-9"}"#,
    )
    .unwrap();
    // A marker whose bytes don't even parse as JSON — the third
    // branch (`unreadable_import_marker`), previously untested here.
    std::fs::write(dir.join("torn.00000000deadbeef.importing"), b"{not json").unwrap();

    let output = run(&["inspect", "--json", &dir.display().to_string()]);
    let stdout_text = String::from_utf8_lossy(&output.stdout).into_owned();
    assert_eq!(output.status.code(), Some(0), "{stdout_text}");
    let report: serde_json::Value = serde_json::from_str(&stdout_text)
        .unwrap_or_else(|error| panic!("--json must be one JSON document: {error}"));

    let notes = report["contexts"][0]["notes"].as_array().unwrap();
    assert!(
        notes
            .iter()
            .any(|note| note["kind"] == "wal_torn_tail" && note["level"] == "note"),
        "{report}"
    );

    let notices = report["notices"].as_array().unwrap();
    assert!(
        notices
            .iter()
            .any(|notice| notice["kind"] == "incomplete_import"
                && notice["level"] == "warning"
                && notice["subject"] == "sake"),
        "{report}"
    );
    assert!(
        notices
            .iter()
            .any(|notice| notice["kind"] == "orphan_import_marker" && notice["level"] == "note"),
        "{report}"
    );
    assert!(
        notices
            .iter()
            .any(|notice| notice["kind"] == "unreadable_import_marker"
                && notice["level"] == "warning"),
        "{report}"
    );

    // The WAL is left byte-for-byte untouched by --json too — the
    // read-only guarantee does not bend for the JSON rendering.
    assert_eq!(std::fs::read(dir.join("sake.wal.jsonl")).unwrap(), bytes);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn inspect_refuses_a_nonexistent_path() {
    let output = run(&["inspect", "/nonexistent/data"]);
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn inspect_help_flag_prints_usage_and_exits_zero() {
    for flag in ["--help", "-h"] {
        let output = run(&["inspect", flag]);
        assert_eq!(output.status.code(), Some(0), "{flag}");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("usage: taguru inspect PATH"),
            "{flag}: {stdout}"
        );
        assert!(String::from_utf8_lossy(&output.stderr).is_empty(), "{flag}");
    }
}

#[test]
fn inspect_help_flag_works_anywhere_in_the_argument_list() {
    // Not just as the sole argument (issue #248 item 10): an operator
    // halfway through composing `inspect PATH` reaches for --help
    // without first deleting the path they already typed.
    let output = run(&["inspect", "/nonexistent/data", "--help"]);
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("usage: taguru inspect PATH"), "{stdout}");
    assert!(String::from_utf8_lossy(&output.stderr).is_empty());
}

#[test]
fn inspect_refuses_the_wrong_number_of_arguments() {
    for args in [&["inspect"][..], &["inspect", "a", "b"][..]] {
        let output = run(args);
        assert_eq!(output.status.code(), Some(2), "{args:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("usage: taguru inspect PATH"),
            "{args:?}"
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).is_empty(),
            "{args:?}"
        );
    }
}

#[test]
fn inspect_reports_no_images_under_an_empty_directory() {
    let dir = common::scratch_dir("cli-inspect-empty");
    std::fs::create_dir_all(&dir).unwrap();

    let output = run(&["inspect", &dir.display().to_string()]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{stdout}");
    assert!(
        stdout.contains(&format!("no .ctx images under {}", dir.display())),
        "{stdout}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn inspect_warns_on_an_undecodable_stem_but_does_not_fail() {
    let dir = common::scratch_dir("cli-inspect-badstem");
    std::fs::create_dir_all(&dir).unwrap();

    // "%zz" is not valid hex — file_stem's own encoding can never
    // produce it, so this is a backup file the server would skip too.
    std::fs::write(dir.join("%zz.ctx"), b"never parsed as an image").unwrap();
    let context = taguru::context::Context::default();
    std::fs::write(dir.join("sake.ctx"), context.to_bytes()).unwrap();

    let output = run(&["inspect", &dir.display().to_string()]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{stdout}");
    assert!(
        stdout.contains("%zz.ctx: WARNING — stem does not decode"),
        "{stdout}"
    );
    assert!(stdout.contains("total: 1 contexts"), "{stdout}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn inspect_warns_on_unparseable_meta_json_but_does_not_fail() {
    let dir = common::scratch_dir("cli-inspect-badmeta");
    std::fs::create_dir_all(&dir).unwrap();

    let context = taguru::context::Context::default();
    std::fs::write(dir.join("sake.ctx"), context.to_bytes()).unwrap();
    std::fs::write(dir.join("sake.meta.json"), b"not json").unwrap();

    let output = run(&["inspect", &dir.display().to_string()]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Self-healing on the server side: a broken sidecar is noted, not fatal.
    assert_eq!(output.status.code(), Some(0), "{stdout}");
    assert!(
        stdout.contains("WARNING: meta.json unparseable"),
        "{stdout}"
    );
    assert!(stdout.contains("total: 1 contexts"), "{stdout}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn inspect_refuses_a_single_corrupt_image_file() {
    let dir = common::scratch_dir("cli-inspect-badfile");
    std::fs::create_dir_all(&dir).unwrap();

    let file = dir.join("sake.ctx");
    std::fs::write(&file, b"not a context image").unwrap();

    let output = run(&["inspect", &file.display().to_string()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stdout).is_empty());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("CORRUPT"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn estimate_reports_memory_and_disk_for_a_target_shape() {
    let output = run(&[
        "estimate",
        "--associations",
        "20000",
        "--embedding-dims",
        "3072",
    ]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{stdout}");
    assert!(stdout.contains("graph footprint"), "{stdout}");
    assert!(stdout.contains("vector store"), "{stdout}");
    assert!(stdout.contains("image"), "{stdout}");
    assert!(stdout.contains("TAGURU_CACHE_BYTES"), "{stdout}");
    assert!(stdout.contains("example benchmark"), "{stdout}");
    assert!(stdout.contains("maintenance window"), "{stdout}");
    assert!(stdout.contains("compaction peak"), "{stdout}");
}

#[test]
fn estimate_requires_the_association_count() {
    let output = run(&["estimate"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("--associations is required"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// ============================== benchmark compare ==============================

/// A minimal but complete `taguru benchmark extract` results
/// directory: one model, one run, one written document with a batch
/// and one failed document — enough for `taguru benchmark compare` to
/// exercise every artifact section without a real model endpoint.
fn write_benchmark_results_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "taguru-cli-benchmark-compare-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("runs")).unwrap();
    std::fs::create_dir_all(dir.join("cells/m/run01")).unwrap();

    std::fs::write(
        dir.join("cells/m/run01/brewery.jsonl"),
        "{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"corpus/brewery.md\"}\n\
         {\"passage\":\"text\"}\n\
         {\"subject\":\"beer co\",\"label\":\"brews\",\"object\":\"lager\",\"weight\":1.0,\"paragraph\":0}\n",
    )
    .unwrap();

    let runs_lines = [
        serde_json::json!({
            "kind": "header", "taguru_benchmark_runs": 1, "run_id": "run-1",
            "cell_id": "m.run01", "model_id": "m", "model_name": "m-model",
            "run_index": 1, "prompt_version": 1,
        }),
        serde_json::json!({
            "kind": "document", "ts": 100.0, "cell_id": "m.run01",
            "document_id": "brewery", "source": "corpus/brewery.md",
            "document_sha256": "sha-brewery", "chunk_total": 1, "phase": "start",
        }),
        serde_json::json!({
            "kind": "attempt", "source": "corpus/brewery.md", "stage": "item",
            "chunk_index": 0, "attempt": 1, "max_attempts": 2, "state": "stop_valid",
            "length_limited": false, "elapsed_seconds": 4.0,
            "provider_metadata": {"finish_reason": "stop", "input_tokens": 1000,
                "output_tokens": 200, "total_tokens": 1200},
            "parse_error": null, "validation_issues": null,
            "ts": 101.0, "cell_id": "m.run01", "model_id": "m", "run_index": 1,
            "document_id": "brewery", "document_sha256": "sha-brewery",
            "chunk_sha256": "sha-chunk0", "paragraph_first": 0, "paragraph_last": 0,
        }),
        serde_json::json!({
            "kind": "document", "ts": 110.0, "cell_id": "m.run01",
            "document_id": "brewery", "source": "corpus/brewery.md",
            "document_sha256": "sha-brewery", "phase": "end", "outcome": "written",
            "associations": 1, "concepts": 0, "labels": 0, "questions": 0,
            "duplicates": 0, "dropped": 0, "batch_path": "cells/m/run01/brewery.jsonl",
        }),
        serde_json::json!({
            "kind": "cell", "ts": 111.0, "cell_id": "m.run01", "outcome": "complete",
            "documents_written": 1, "attempts_total": 1, "exit_code": 0,
        }),
    ];
    let runs_text = runs_lines
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    std::fs::write(dir.join("runs/m.run01.jsonl"), runs_text).unwrap();

    let manifest = serde_json::json!({
        "taguru_benchmark_manifest": 1,
        "run_id": "run-1",
        "started_at": "2026-07-26T09:00:00Z",
        "finished_at": "2026-07-26T09:05:00Z",
        "taguru_version": "0.0.0",
        "sdk_versions": {},
        "harness": {},
        "extraction_settings": {},
        "documents": [
            {
                "document_id": "brewery", "path": "corpus/brewery.md", "bytes": 100,
                "sha256": "sha-brewery", "paragraph_count": 5, "chunk_total": 1, "chunks": [],
            },
        ],
        "models": [
            {
                "model_id": "m", "model_name": "m-model", "endpoint": "http://x",
                "digest": null, "quantization": null, "context_window": null,
                "structured_output_requested": "auto", "timeout_secs": 60,
                "provider_probe": {"attempted": [], "ok": true, "note": null},
            },
        ],
        "cells": [
            {
                "cell_id": "m.run01", "model_id": "m", "run_index": 1,
                "runs_file": "runs/m.run01.jsonl", "cell_dir": "cells/m/run01",
                "structured_output_resolved": "json_schema",
                "started_at": "2026-07-26T09:00:01Z",
                "finished_at": "2026-07-26T09:04:00Z", "outcome": "complete",
            },
        ],
        "environment": {"os": "linux", "arch": "x86_64"},
    });
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    dir
}

/// Checks object *keys* only, never string values — `percentile_method:
/// "nearest-rank"` legitimately contains "rank" as a value, which must
/// not trip this check the way a `rank` key would.
fn assert_no_banned_keys(value: &serde_json::Value) {
    const BANNED: [&str; 7] = [
        "rank",
        "score",
        "winner",
        "best",
        "recommended",
        "overall",
        "delta_vs",
    ];
    match value {
        serde_json::Value::Object(map) => {
            for (key, v) in map {
                for banned in BANNED {
                    assert!(
                        !key.to_lowercase().contains(banned),
                        "banned key fragment '{banned}' found in key '{key}'"
                    );
                }
                assert_no_banned_keys(v);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                assert_no_banned_keys(item);
            }
        }
        _ => {}
    }
}

#[test]
fn benchmark_compare_derives_measurements_from_a_results_directory() {
    let dir = write_benchmark_results_dir("smoke");

    let output = run(&["benchmark", "compare", &dir.display().to_string()]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let json_text = std::fs::read_to_string(dir.join("measurements.json")).unwrap();
    let measurements: serde_json::Value = serde_json::from_str(&json_text).unwrap();
    assert_eq!(measurements["taguru_benchmark_measurements"], 1);
    assert_eq!(measurements["percentile_method"], "nearest-rank");
    assert!(measurements["cells"]["m.run01"].is_object());
    assert!(measurements["models"]["m"].is_object());
    assert!(measurements["documents"]["m"]["brewery"]["run01"].is_object());
    assert_no_banned_keys(&measurements);

    // issue #258: the same-ness parameters every stability metric was
    // computed under, recorded so the artifact stays re-derivable
    // without reading benchmark::identity's source (ADR 0003 §9.4).
    assert_eq!(measurements["matching"]["module"], "benchmark::identity");
    assert_eq!(measurements["matching"]["case_fold"], true);
    assert_eq!(measurements["matching"]["unicode_normalization"], "NFKC");
    assert_eq!(measurements["matching"]["alias_expansion"], "batch-local");
    assert!(measurements["models"]["m"]["stability.run_pair_jaccard"].is_object());

    let csv_text = std::fs::read_to_string(dir.join("measurements.csv")).unwrap();
    assert_eq!(
        csv_text.lines().next(),
        Some("scope,model_id,run_index,document_id,metric,stat,value,unit,n")
    );

    // Re-running is a pure function of the (unchanged) results
    // directory: same bytes out, generated_at aside.
    let output2 = run(&["benchmark", "compare", &dir.display().to_string()]);
    assert_eq!(output2.status.code(), Some(0));
    let csv_text2 = std::fs::read_to_string(dir.join("measurements.csv")).unwrap();
    assert_eq!(
        csv_text, csv_text2,
        "measurements.csv must be byte-identical across reruns"
    );

    let json_text2 = std::fs::read_to_string(dir.join("measurements.json")).unwrap();
    let measurements2: serde_json::Value = serde_json::from_str(&json_text2).unwrap();
    let mut a = measurements.clone();
    let mut b = measurements2.clone();
    a.as_object_mut().unwrap().remove("generated_at");
    b.as_object_mut().unwrap().remove("generated_at");
    assert_eq!(
        a, b,
        "measurements.json must match across reruns aside from generated_at"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn benchmark_compare_requires_exactly_one_results_dir_argument() {
    let output = run(&["benchmark", "compare"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("RESULTS_DIR"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let output = run(&["benchmark", "compare", "a", "b"]);
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn benchmark_compare_rejects_a_directory_missing_a_manifest() {
    let dir = std::env::temp_dir().join(format!(
        "taguru-cli-benchmark-compare-no-manifest-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let output = run(&["benchmark", "compare", &dir.display().to_string()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("manifest.json"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn benchmark_reports_all_subcommands_on_an_unknown_one() {
    let output = run(&["benchmark", "bogus"]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("'extract', 'compare', or 'search'"),
        "{stderr}"
    );
}

// ============================== benchmark compare: differences.jsonl (issue #259) ==============================

/// A two-model results directory: `m1` and `m2`, one run each, one
/// document (`brewery`) both complete. `beer co`/`brews`/`lager`
/// disagrees in weight sign between the two models — enough to exercise
/// `differences.jsonl`'s `association_shared` and `polarity_difference`
/// records without duplicating `benchmark::compare::tests`' much larger
/// synthetic fixture (every record kind, every fire condition) that
/// already covers the module's logic in depth.
fn write_two_model_benchmark_results_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "taguru-cli-benchmark-differences-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("runs")).unwrap();
    std::fs::create_dir_all(dir.join("cells/m1/run01")).unwrap();
    std::fs::create_dir_all(dir.join("cells/m2/run01")).unwrap();

    std::fs::write(
        dir.join("cells/m1/run01/brewery.jsonl"),
        "{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"corpus/brewery.md\"}\n\
         {\"passage\":\"text\"}\n\
         {\"subject\":\"beer co\",\"label\":\"brews\",\"object\":\"lager\",\"weight\":1.0,\"paragraph\":0}\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("cells/m2/run01/brewery.jsonl"),
        "{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"corpus/brewery.md\"}\n\
         {\"passage\":\"text\"}\n\
         {\"subject\":\"beer co\",\"label\":\"brews\",\"object\":\"lager\",\"weight\":-1.0,\"paragraph\":0}\n",
    )
    .unwrap();

    fn cell_lines(cell_id: &str, model_id: &str, batch_path: &str) -> Vec<serde_json::Value> {
        vec![
            serde_json::json!({
                "kind": "header", "taguru_benchmark_runs": 1, "run_id": "run-diff-cli",
                "cell_id": cell_id, "model_id": model_id, "model_name": format!("{model_id}-model"),
                "run_index": 1, "prompt_version": 1,
            }),
            serde_json::json!({
                "kind": "document", "ts": 100.0, "cell_id": cell_id,
                "document_id": "brewery", "source": "corpus/brewery.md",
                "document_sha256": "sha-brewery", "chunk_total": 1, "phase": "start",
            }),
            serde_json::json!({
                "kind": "document", "ts": 110.0, "cell_id": cell_id,
                "document_id": "brewery", "source": "corpus/brewery.md",
                "document_sha256": "sha-brewery", "phase": "end", "outcome": "written",
                "associations": 1, "concepts": 0, "labels": 0, "questions": 0,
                "duplicates": 0, "dropped": 0, "batch_path": batch_path,
            }),
            serde_json::json!({
                "kind": "cell", "ts": 111.0, "cell_id": cell_id, "outcome": "complete",
                "documents_written": 1, "attempts_total": 0, "exit_code": 0,
            }),
        ]
    }
    for (cell_id, model_id, batch_path) in [
        ("m1.run01", "m1", "cells/m1/run01/brewery.jsonl"),
        ("m2.run01", "m2", "cells/m2/run01/brewery.jsonl"),
    ] {
        let text = cell_lines(cell_id, model_id, batch_path)
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        std::fs::write(dir.join(format!("runs/{cell_id}.jsonl")), text).unwrap();
    }

    let manifest = serde_json::json!({
        "taguru_benchmark_manifest": 1,
        "run_id": "run-diff-cli",
        "started_at": "2026-07-26T09:00:00Z",
        "finished_at": "2026-07-26T09:05:00Z",
        "taguru_version": "0.0.0",
        "sdk_versions": {},
        "harness": {},
        "extraction_settings": {},
        "documents": [
            {
                "document_id": "brewery", "path": "corpus/brewery.md", "bytes": 100,
                "sha256": "sha-brewery", "paragraph_count": 5, "chunk_total": 1, "chunks": [],
            },
        ],
        "models": [
            {
                "model_id": "m1", "model_name": "m1-model", "endpoint": "http://x",
                "digest": null, "quantization": null, "context_window": null,
                "structured_output_requested": "auto", "timeout_secs": 60,
                "provider_probe": {"attempted": [], "ok": true, "note": null},
            },
            {
                "model_id": "m2", "model_name": "m2-model", "endpoint": "http://y",
                "digest": null, "quantization": null, "context_window": null,
                "structured_output_requested": "auto", "timeout_secs": 60,
                "provider_probe": {"attempted": [], "ok": true, "note": null},
            },
        ],
        "cells": [
            {
                "cell_id": "m1.run01", "model_id": "m1", "run_index": 1,
                "runs_file": "runs/m1.run01.jsonl", "cell_dir": "cells/m1/run01",
                "structured_output_resolved": "json_schema",
                "started_at": "2026-07-26T09:00:01Z",
                "finished_at": "2026-07-26T09:04:00Z", "outcome": "complete",
            },
            {
                "cell_id": "m2.run01", "model_id": "m2", "run_index": 1,
                "runs_file": "runs/m2.run01.jsonl", "cell_dir": "cells/m2/run01",
                "structured_output_resolved": "json_schema",
                "started_at": "2026-07-26T09:00:01Z",
                "finished_at": "2026-07-26T09:04:00Z", "outcome": "complete",
            },
        ],
        "environment": {"os": "linux", "arch": "x86_64"},
    });
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    dir
}

fn read_differences_lines(dir: &std::path::Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(dir.join("differences.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn benchmark_compare_derives_differences_for_each_model_pair() {
    let dir = write_two_model_benchmark_results_dir("pair");

    let output = run(&["benchmark", "compare", &dir.display().to_string()]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let lines = read_differences_lines(&dir);
    assert_eq!(lines[0]["kind"], "header");
    assert_eq!(lines[0]["taguru_benchmark_differences"], 2);
    assert_eq!(lines[0]["text_included"], false);
    assert_eq!(
        lines[0]["pairs"],
        serde_json::json!([{"pair_id": "2:m1__m2", "a": "m1", "b": "m2"}])
    );
    for line in &lines {
        assert_no_banned_keys(line);
    }
    assert!(
        lines
            .iter()
            .any(|l| l["kind"] == "association_shared" && l["key"]["object"] == "lager"),
        "{lines:?}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l["kind"] == "polarity_difference" && l["key"]["object"] == "lager"),
        "{lines:?}"
    );

    // Re-running is a pure function of the (unchanged) results
    // directory: differences.jsonl carries no generated_at, so it must
    // be byte-identical across reruns, not merely equal aside from one
    // field.
    let differences_text = std::fs::read_to_string(dir.join("differences.jsonl")).unwrap();
    let output2 = run(&["benchmark", "compare", &dir.display().to_string()]);
    assert_eq!(output2.status.code(), Some(0));
    let differences_text2 = std::fs::read_to_string(dir.join("differences.jsonl")).unwrap();
    assert_eq!(
        differences_text, differences_text2,
        "differences.jsonl must be byte-identical across reruns"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn benchmark_compare_with_text_embeds_paragraph_text() {
    let dir = write_two_model_benchmark_results_dir("with-text");
    std::fs::create_dir_all(dir.join("corpus")).unwrap();
    let text = "para0 is the one lager points at";
    std::fs::write(dir.join("corpus/brewery.md"), text).unwrap();
    let real_sha = {
        // Mirrors `crate::extract::sha256_hex` (src/extract.rs) without
        // depending on the library crate from an integration test —
        // sha2 is already a direct dependency (Cargo.toml).
        use sha2::{Digest, Sha256};
        use std::fmt::Write;
        Sha256::digest(text.as_bytes())
            .iter()
            .fold(String::with_capacity(64), |mut hex, byte| {
                let _ = write!(hex, "{byte:02x}");
                hex
            })
    };

    let manifest_path = dir.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["documents"][0]["path"] =
        serde_json::json!(dir.join("corpus/brewery.md").to_string_lossy());
    manifest["documents"][0]["sha256"] = serde_json::json!(real_sha);
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let output = run(&[
        "benchmark",
        "compare",
        "--with-text",
        &dir.display().to_string(),
    ]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let lines = read_differences_lines(&dir);
    assert_eq!(lines[0]["text_included"], true);
    let polarity = lines
        .iter()
        .find(|l| l["kind"] == "polarity_difference")
        .expect("lager fires polarity_difference");
    assert_eq!(polarity["locator"]["text"], text);
    assert_eq!(polarity["locator"]["text_truncated"], false);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn benchmark_compare_with_text_refuses_corpus_drift() {
    let dir = write_two_model_benchmark_results_dir("with-text-drift");
    std::fs::create_dir_all(dir.join("corpus")).unwrap();
    std::fs::write(dir.join("corpus/brewery.md"), "drifted since the run").unwrap();

    let manifest_path = dir.join("manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["documents"][0]["path"] =
        serde_json::json!(dir.join("corpus/brewery.md").to_string_lossy());
    // sha256 stays the fixture's placeholder — deliberately not
    // matching the corpus file written above.
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let output = run(&[
        "benchmark",
        "compare",
        "--with-text",
        &dir.display().to_string(),
    ]);
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("brewery.md"), "{stderr}");
    // A refused --with-text run must publish none of the three
    // artifacts — measurements.json/.csv included, since one shared
    // `load_results` call feeds both and the refusal happens before any
    // artifact is staged.
    assert!(!dir.join("differences.jsonl").exists());
    assert!(!dir.join("measurements.json").exists());
    assert!(!dir.join("measurements.csv").exists());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn benchmark_compare_with_text_flag_is_accepted_alongside_the_positional() {
    let dir = write_benchmark_results_dir("with-text-flag-order");
    let output = run(&[
        "benchmark",
        "compare",
        "--with-text",
        &dir.display().to_string(),
    ]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_mcp_bridge_answers_initialize_despite_a_stalled_protocol_probe() {
    use std::io::Write;

    // A listener that accepts (localhost handshakes complete via the
    // backlog) but never responds: the worst startup case — a server
    // that is not dead, just silent. The bridge's protocol probe must
    // give up on its own short ceiling, not hold stdio hostage for the
    // full 75-second tool-call timeout an MCP client's handshake
    // budget never survives.
    let stall = std::net::TcpListener::bind("127.0.0.1:0").expect("stall listener must bind");
    let addr = stall.local_addr().unwrap();

    let mut child = Command::new(env!("CARGO_BIN_EXE_taguru-mcp"))
        .env("TAGURU_URL", format!("http://{addr}"))
        .env_remove("TAGURU_API_TOKEN")
        .env_remove("TAGURU_MCP_TIMEOUT_SECS")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("bridge must spawn");

    let mut stdin = child.stdin.take().unwrap();
    writeln!(
        stdin,
        r#"{{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {{}}}}"#
    )
    .unwrap();

    // Read the reply on a side thread so a hung bridge fails this test
    // by timeout rather than hanging the harness with it.
    let stdout = child.stdout.take().unwrap();
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut lines = BufReader::new(stdout).lines();
        let _ = sender.send(lines.next().and_then(Result::ok));
    });
    let reply = receiver
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("initialize must be answered within the probe ceiling, not the tool timeout")
        .expect("one JSON-RPC response line");
    assert!(reply.contains(r#""id":1"#), "{reply}");
    // The probe failed, so the bundled protocol copy is what serves.
    assert!(reply.contains("instructions"), "{reply}");

    let _ = child.kill();
    let _ = child.wait();
}

/// #62 item 1: the stdio bridge's `Bridge::call` must carry the
/// `import` tool's NDJSON stream as raw text, exactly like the HTTP
/// transport's `call_inner` — naively JSON-encoding a string argument
/// would escape every newline and collapse a multi-line stream onto
/// one unparsable line. Verified end to end against a live server
/// rather than at the routing layer alone, since the routing test
/// cannot see how the bridge actually serializes the request body.
#[test]
fn the_mcp_bridge_applies_a_multi_line_import_stream_through_a_live_server() {
    use std::io::Write;

    let (mut server, addr, dir) = spawn_server("mcp-bridge-import");

    let mut bridge = Command::new(env!("CARGO_BIN_EXE_taguru-mcp"))
        .env("TAGURU_URL", format!("http://{addr}"))
        .env_remove("TAGURU_API_TOKEN")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("bridge must spawn");

    let stream = "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-bridge\", \
                 \"create\": {\"description\": \"d\"}}\n\
                 {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n";
    let request = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "import", "arguments": {"stream": stream}}
    });

    let mut stdin = bridge.stdin.take().unwrap();
    writeln!(stdin, "{request}").unwrap();
    drop(stdin);

    let stdout = bridge.stdout.take().unwrap();
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut lines = BufReader::new(stdout).lines();
        let _ = sender.send(lines.next().and_then(Result::ok));
    });
    let reply = receiver
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the bridge must answer the tool call")
        .expect("one JSON-RPC response line");

    let _ = bridge.kill();
    let _ = bridge.wait();
    let _ = server.kill();
    let _ = server.wait();
    let _ = std::fs::remove_dir_all(&dir);

    let answer: serde_json::Value = serde_json::from_str(&reply).expect("reply must be JSON");
    assert!(
        answer["result"].get("isError").is_none(),
        "the multi-line stream must not be mangled into one unparsable line: {reply}"
    );
    let text = answer["result"]["content"][0]["text"].as_str().unwrap();
    let envelope: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(
        envelope["result"]["batches"][0]["created"],
        serde_json::json!(true)
    );
    assert_eq!(
        envelope["result"]["batches"][0]["associations"],
        serde_json::json!(1)
    );
}

/// issue #182: a rejected ingestion tool call carries the same
/// structured JSON detail over the stdio bridge as it does over
/// `POST /mcp` — `Bridge::call` parses the downstream HTTP error body
/// itself and attaches it as `structuredContent`, alongside the
/// unchanged prose in `content[0].text`.
#[test]
fn the_mcp_bridge_carries_structured_content_on_a_rejected_write() {
    use std::io::Write;

    let (mut server, addr, dir) = spawn_server("mcp-bridge-structured");

    let mut bridge = Command::new(env!("CARGO_BIN_EXE_taguru-mcp"))
        .env("TAGURU_URL", format!("http://{addr}"))
        .env_remove("TAGURU_API_TOKEN")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("bridge must spawn");

    let create = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "create_context", "arguments": {"name": "sake"}}
    });
    let invalid = serde_json::json!({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "add_associations", "arguments": {"context": "sake", "associations": [
            {"subject": "s", "label": "l", "object": "o", "weight": "strong"}
        ]}}
    });

    let mut stdin = bridge.stdin.take().unwrap();
    writeln!(stdin, "{create}").unwrap();
    writeln!(stdin, "{invalid}").unwrap();
    drop(stdin);

    let stdout = bridge.stdout.take().unwrap();
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut lines = BufReader::new(stdout).lines();
        let _ = sender.send(lines.next().and_then(Result::ok));
        let _ = sender.send(lines.next().and_then(Result::ok));
    });
    // The bridge dispatches queued `tools/call` requests onto a worker
    // pool, so the two replies are not guaranteed to arrive in request
    // order — match by `id` instead of position.
    let first = receiver
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the bridge must answer the first call")
        .expect("one JSON-RPC response line");
    let second = receiver
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the bridge must answer the second call")
        .expect("one JSON-RPC response line");

    let _ = bridge.kill();
    let _ = bridge.wait();
    let _ = server.kill();
    let _ = server.wait();
    let _ = std::fs::remove_dir_all(&dir);

    let first: serde_json::Value = serde_json::from_str(&first).expect("reply must be JSON");
    let second: serde_json::Value = serde_json::from_str(&second).expect("reply must be JSON");
    let answer = if first["id"] == serde_json::json!(2) {
        first
    } else {
        second
    };
    assert_eq!(
        answer["result"]["isError"],
        serde_json::json!(true),
        "{answer}"
    );
    let text = answer["result"]["content"][0]["text"].as_str().unwrap();
    assert!(text.contains("associations[0].weight"), "{text}");
    let structured = &answer["result"]["structuredContent"];
    assert_eq!(
        structured["integrity"],
        serde_json::json!("nothing_written"),
        "{answer}"
    );
    assert_eq!(
        structured["issues"][0]["path"],
        serde_json::json!("associations[0].weight"),
        "{answer}"
    );
    assert_eq!(
        structured["issues"][0]["kind"],
        serde_json::json!("type"),
        "{answer}"
    );
}

/// A client that pipelines far more `tools/call` requests than
/// `TAGURU_MCP_MAX_CONCURRENT_TOOLS` must still get every one of them
/// answered, each matched to its own id — proving the fixed-size worker
/// pool actually queues a backlog rather than losing or wedging requests
/// past its own concurrency ceiling (the failure mode a one-thread-per-call
/// design would never hit, since it never has a queue to get stuck in).
#[test]
fn the_mcp_bridge_drains_a_pipelined_backlog_through_a_small_worker_pool() {
    use std::io::Write;

    let (mut server, addr, dir) = spawn_server("mcp-bridge-backlog");

    let mut bridge = Command::new(env!("CARGO_BIN_EXE_taguru-mcp"))
        .env("TAGURU_URL", format!("http://{addr}"))
        .env_remove("TAGURU_API_TOKEN")
        .env("TAGURU_MCP_MAX_CONCURRENT_TOOLS", "2")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("bridge must spawn");

    let calls: i64 = 50;
    let mut stdin = bridge.stdin.take().unwrap();
    for id in 0..calls {
        let request = serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": "tools/call",
            "params": {"name": "get_protocol", "arguments": {}}
        });
        writeln!(stdin, "{request}").unwrap();
    }
    drop(stdin);

    let stdout = bridge.stdout.take().unwrap();
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut lines = BufReader::new(stdout).lines();
        for _ in 0..calls {
            match lines.next() {
                Some(Ok(line)) => {
                    if sender.send(line).is_err() {
                        break;
                    }
                }
                _ => break,
            }
        }
    });

    let mut seen_ids = std::collections::HashSet::new();
    for _ in 0..calls {
        let reply = receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect(
                "every pipelined call must eventually be answered, not dropped or wedged \
                 behind a 2-worker pool",
            );
        let answer: serde_json::Value = serde_json::from_str(&reply).expect("reply must be JSON");
        assert!(answer["result"].get("isError").is_none(), "{reply}");
        seen_ids.insert(
            answer["id"]
                .as_i64()
                .expect("id must echo back as a number"),
        );
    }

    let _ = bridge.kill();
    let _ = bridge.wait();
    let _ = server.kill();
    let _ = server.wait();
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(
        seen_ids.len(),
        calls as usize,
        "each of the {calls} pipelined ids must get exactly one reply"
    );
}

#[test]
fn estimate_prints_usage_for_help_in_any_position() {
    // The other subcommands answer --help wherever it appears; an
    // operator halfway through composing flags gets the manual, not
    // "unknown flag '--help'".
    let output = run(&["estimate", "--associations", "100", "--help"]);
    assert_eq!(output.status.code(), Some(0));
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("usage: taguru estimate"),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
}

/// The offline backup loop end to end: import seeds a data directory,
/// export writes it back out as batch streams, import --dry-run
/// validates the streams untouched, a second import restores them into
/// a fresh directory, and inspect vouches for the restored family.
/// Re-exporting the restored directory reproduces the streams byte for
/// byte — the format is deterministic, so backups diff cleanly.
#[test]
fn export_round_trips_a_data_directory_through_batch_streams() {
    let dir = common::scratch_dir("cli-export");
    std::fs::create_dir_all(dir.join("batches")).expect("scratch dir must be creatable");
    std::fs::write(
        dir.join("batches/a.jsonl"),
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
         \"create\": {\"description\": \"酒蔵の知識\"}}\n\
         {\"passage\": \"青嶺酒造の紹介。\\n\\n代表銘柄は青嶺。\"}\n\
         {\"paragraph\": 0, \"section\": \"概要\"}\n\
         {\"subject\": \"青嶺酒造\", \"label\": \"代表銘柄\", \"object\": \"青嶺\", \
          \"weight\": 1.0, \"paragraph\": 1}\n\
         {\"alias\": \"Aomine\", \"canonical\": \"青嶺酒造\", \"kind\": \"concept\"}\n",
    )
    .expect("fixture must be writable");
    std::fs::write(
        dir.join("batches/b.jsonl"),
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"b.md\"}\n\
         {\"subject\": \"青嶺酒造\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 2.0}\n",
    )
    .expect("fixture must be writable");
    // A group record beside the batches: groups restore after every
    // batch of the run, so the file order never matters.
    std::fs::write(
        dir.join("batches/kura.jsonl"),
        "{\"taguru_group\": 1, \"name\": \"kura\", \"description\": \"蔵まとめ\", \
          \"contexts\": [\"sake\"]}\n",
    )
    .expect("fixture must be writable");

    let run_in = |data_dir: &std::path::Path, args: &[&str]| -> Output {
        Command::new(env!("CARGO_BIN_EXE_taguru"))
            .args(args)
            .env("TAGURU_DATA_DIR", data_dir)
            .env_remove("TAGURU_CONFIG")
            .env_remove("TAGURU_EMBED_URL")
            .output()
            .expect("binary must run")
    };

    let data_a = dir.join("data-a");
    let seeded = run_in(
        &data_a,
        &["import", &dir.join("batches").display().to_string()],
    );
    assert_eq!(seeded.status.code(), Some(0), "{seeded:?}");

    let exports = dir.join("exports");
    let exported = run_in(
        &data_a,
        &["export", "--out", &exports.display().to_string()],
    );
    assert_eq!(exported.status.code(), Some(0), "{exported:?}");
    let stdout = String::from_utf8_lossy(&exported.stdout);
    assert!(stdout.contains("sake.jsonl"), "{stdout}");
    assert!(stdout.contains("2 batch(es)"), "{stdout}");
    // The full export carries the group as its own record file.
    assert!(stdout.contains("group 'kura'"), "{stdout}");
    assert!(stdout.contains("1 of 1 group(s)"), "{stdout}");
    let stream =
        std::fs::read_to_string(exports.join("sake.jsonl")).expect("the stream must exist");
    assert!(
        stream.contains("\"description\":\"酒蔵の知識\""),
        "{stream}"
    );
    let group_stream = std::fs::read_to_string(exports.join("kura.group.jsonl"))
        .expect("the group record must exist");
    assert!(
        group_stream.contains("\"taguru_group\":1") && group_stream.contains("蔵まとめ"),
        "{group_stream}"
    );

    // --dry-run validates the export without a data directory or lock.
    let checked = run_in(
        &data_a,
        &["import", "--dry-run", &exports.display().to_string()],
    );
    assert_eq!(checked.status.code(), Some(0), "{checked:?}");
    assert!(
        String::from_utf8_lossy(&checked.stdout)
            .contains("2 batch(es) and 1 group record(s) valid"),
        "{}",
        String::from_utf8_lossy(&checked.stdout)
    );

    let data_b = dir.join("data-b");
    let restored = run_in(&data_b, &["import", &exports.display().to_string()]);
    assert_eq!(restored.status.code(), Some(0), "{restored:?}");
    assert!(
        String::from_utf8_lossy(&restored.stdout).contains("1 of 1 group record(s) restored"),
        "{}",
        String::from_utf8_lossy(&restored.stdout)
    );
    let inspected = run_in(&data_b, &["inspect", &data_b.display().to_string()]);
    assert_eq!(inspected.status.code(), Some(0), "{inspected:?}");
    let inspected_out = String::from_utf8_lossy(&inspected.stdout);
    assert!(inspected_out.contains("kura: ok"), "{inspected_out}");
    assert!(
        inspected_out.contains("total: 1 contexts · 1 groups"),
        "{inspected_out}"
    );

    let re_exports = dir.join("exports-b");
    let re_exported = run_in(
        &data_b,
        &["export", "--out", &re_exports.display().to_string()],
    );
    assert_eq!(re_exported.status.code(), Some(0), "{re_exported:?}");
    let re_stream =
        std::fs::read_to_string(re_exports.join("sake.jsonl")).expect("the stream must exist");
    assert_eq!(
        stream, re_stream,
        "a restore must re-export byte-identically"
    );
    let re_group_stream = std::fs::read_to_string(re_exports.join("kura.group.jsonl"))
        .expect("the group record must exist");
    assert_eq!(
        group_stream, re_group_stream,
        "the group record must re-export byte-identically"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// An unknown context in an explicit `export` list is a per-item
/// failure: the named survivors still land, the summary counts them,
/// and the exit code is nonzero — the offline twin of
/// `remote_export::an_unknown_context_counts_as_a_failure_and_the_rest_still_lands`.
#[test]
fn export_counts_an_unknown_context_as_a_failure() {
    let dir = common::scratch_dir("cli-export-unknown");
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
    std::fs::write(
        dir.join("a.jsonl"),
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
         \"create\": {\"description\": \"d\"}}\n\
         {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n",
    )
    .expect("fixture must be writable");
    let data = dir.join("data");
    let run_in = |args: &[&str]| -> Output {
        Command::new(env!("CARGO_BIN_EXE_taguru"))
            .args(args)
            .env("TAGURU_DATA_DIR", &data)
            .env_remove("TAGURU_CONFIG")
            .env_remove("TAGURU_EMBED_URL")
            .output()
            .expect("binary must run")
    };
    let seeded = run_in(&["import", &dir.join("a.jsonl").display().to_string()]);
    assert_eq!(seeded.status.code(), Some(0), "{seeded:?}");

    let out = dir.join("out");
    let output = run_in(&[
        "export",
        "--out",
        &out.display().to_string(),
        "sake",
        "nope",
    ]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("export: 1 of 2 context(s) written"),
        "{output:?}"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("context 'nope': no such context"),
        "{output:?}"
    );
    assert!(out.join("sake.jsonl").exists(), "the survivor must land");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A group record that cannot be written is a failure the summary and
/// the exit code must both carry — forced by pre-creating a DIRECTORY
/// where the group file would land, so `write_atomic`'s rename cannot.
#[test]
fn export_counts_an_unwritable_group_file_as_a_failure() {
    let dir = common::scratch_dir("cli-export-group-unwritable");
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
    std::fs::write(
        dir.join("a.jsonl"),
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
         \"create\": {\"description\": \"d\"}}\n\
         {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n",
    )
    .expect("fixture must be writable");
    std::fs::write(
        dir.join("kura.jsonl"),
        "{\"taguru_group\": 1, \"name\": \"kura\", \"description\": \"蔵まとめ\", \
          \"contexts\": [\"sake\"]}\n",
    )
    .expect("fixture must be writable");
    let data = dir.join("data");
    let run_in = |args: &[&str]| -> Output {
        Command::new(env!("CARGO_BIN_EXE_taguru"))
            .args(args)
            .env("TAGURU_DATA_DIR", &data)
            .env_remove("TAGURU_CONFIG")
            .env_remove("TAGURU_EMBED_URL")
            .output()
            .expect("binary must run")
    };
    for file in ["a.jsonl", "kura.jsonl"] {
        let seeded = run_in(&["import", &dir.join(file).display().to_string()]);
        assert_eq!(seeded.status.code(), Some(0), "{seeded:?}");
    }

    let out = dir.join("out");
    std::fs::create_dir_all(out.join("kura.group.jsonl")).expect("the decoy must be creatable");
    let output = run_in(&["export", "--out", &out.display().to_string()]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("export: 1 of 1 context(s) and 0 of 1 group(s)"),
        "{stdout}"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("group 'kura'"),
        "{output:?}"
    );
    assert!(
        out.join("sake.jsonl").exists(),
        "the context stream the summary counts as written must actually remain"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Pass 1's summary counts refused FILES, not refused batches: a
/// stream file (`taguru export` output, or any hand-built one) can
/// carry several batches, and each one that restates a source an
/// earlier file already claimed logs its own conflict line — but the
/// file itself must still add at most 1 to the tally, or "N of M
/// file(s) refused" could report N > M from a single offending file.
#[test]
fn a_multi_batch_stream_restating_earlier_sources_counts_as_one_refused_file() {
    let dir = std::env::temp_dir().join(format!(
        "taguru-cli-import-refused-count-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");

    // first.jsonl claims three sources in one stream — all novel.
    std::fs::write(
        dir.join("first.jsonl"),
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s1\"}\n\
         {\"subject\": \"a\", \"label\": \"l\", \"object\": \"o1\", \"weight\": 1.0}\n\
         {\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s2\"}\n\
         {\"subject\": \"a\", \"label\": \"l\", \"object\": \"o2\", \"weight\": 1.0}\n\
         {\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s3\"}\n\
         {\"subject\": \"a\", \"label\": \"l\", \"object\": \"o3\", \"weight\": 1.0}\n",
    )
    .expect("fixture must be writable");
    // second.jsonl restates the same three sources in one stream of its
    // own — one refused FILE, but three separate ownership conflicts.
    std::fs::write(
        dir.join("second.jsonl"),
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s1\"}\n\
         {\"subject\": \"a\", \"label\": \"l\", \"object\": \"o1b\", \"weight\": 1.0}\n\
         {\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s2\"}\n\
         {\"subject\": \"a\", \"label\": \"l\", \"object\": \"o2b\", \"weight\": 1.0}\n\
         {\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s3\"}\n\
         {\"subject\": \"a\", \"label\": \"l\", \"object\": \"o3b\", \"weight\": 1.0}\n",
    )
    .expect("fixture must be writable");

    let output = run(&[
        "import",
        &dir.join("first.jsonl").display().to_string(),
        &dir.join("second.jsonl").display().to_string(),
    ]);
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Each conflicting batch is still named individually…
    assert_eq!(
        stderr
            .matches("is already stated by an earlier file")
            .count(),
        3,
        "{stderr}"
    );
    // …but only one of the two files actually failed — never 3 of 2.
    assert!(
        stderr.contains("1 of 2 file(s) refused during validation"),
        "{stderr}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Inspect covers the group files: parse trouble fails the check (a
/// boot would reset the record, and membership is acknowledged data),
/// an unreadable file fails it (a boot refuses outright), and what
/// boot's reconciliation would merely drop — dangling members,
/// ill-shaped nesting — warns without failing, since the server
/// accepts the directory and heals it.
#[test]
fn inspect_flags_group_trouble_and_previews_boot_repairs() {
    let dir = common::scratch_dir("cli-inspect-groups");
    std::fs::create_dir_all(&dir).unwrap();
    let context = taguru::context::Context::default();
    std::fs::write(dir.join("sake.ctx"), context.to_bytes()).unwrap();
    std::fs::write(
        dir.join("kura.group"),
        "{\"description\": \"\", \"contexts\": [\"sake\", \"ghost\"], \"groups\": []}",
    )
    .unwrap();

    // A record that parses is ok; its dangling member is the preview
    // of what boot would drop — a warning, never a failure.
    let output = run(&["inspect", &dir.display().to_string()]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(0), "{stdout}");
    assert!(stdout.contains("kura: ok"), "{stdout}");
    assert!(
        stdout.contains("member context(s) have no context here"),
        "{stdout}"
    );
    assert!(stdout.contains("total: 1 contexts · 1 groups"), "{stdout}");

    // A nesting the validator refuses warns the same way — the
    // preview runs the real repair, so EVERY doomed edge is named in
    // one run (a cycle and an over-deep chain at once), not just the
    // first violation a walk happens to hit.
    std::fs::write(dir.join("cyc-a.group"), "{\"groups\": [\"cyc-b\"]}").unwrap();
    std::fs::write(dir.join("cyc-b.group"), "{\"groups\": [\"cyc-a\"]}").unwrap();
    for (parent, child) in [("n1", "n2"), ("n2", "n3"), ("n3", "n4"), ("n4", "")] {
        let children = if child.is_empty() {
            "[]".to_string()
        } else {
            format!("[\"{child}\"]")
        };
        std::fs::write(
            dir.join(format!("{parent}.group")),
            format!("{{\"groups\": {children}}}"),
        )
        .unwrap();
    }
    let output = run(&["inspect", &dir.display().to_string()]);
    assert_eq!(
        output.status.code(),
        Some(0),
        "shape trouble warns, never fails"
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Deterministic, name-order repair: the second cycle edge and the
    // chain's deepest edge are exactly what boot would drop.
    assert!(
        stdout.contains("boot drops the nesting edge 'cyc-b' → 'cyc-a'"),
        "{stdout}"
    );
    assert!(
        stdout.contains("boot drops the nesting edge 'n3' → 'n4'"),
        "{stdout}"
    );
    for stale in ["n1", "n2", "n3", "n4"] {
        std::fs::remove_file(dir.join(format!("{stale}.group"))).unwrap();
    }
    std::fs::remove_file(dir.join("cyc-a.group")).unwrap();
    std::fs::remove_file(dir.join("cyc-b.group")).unwrap();

    // Bytes that do not parse fail the inspection — restoring this
    // backup would reset the record.
    std::fs::write(dir.join("bad.group"), b"{not json").unwrap();
    let output = run(&["inspect", &dir.display().to_string()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("CORRUPT group"),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    std::fs::remove_file(dir.join("bad.group")).unwrap();

    // An unreadable file fails it too — a boot refuses to start. A
    // directory wearing the extension fails fs::read on every platform.
    std::fs::create_dir(dir.join("locked.group")).unwrap();
    let output = run(&["inspect", &dir.display().to_string()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("UNREADABLE group"),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    std::fs::remove_dir(dir.join("locked.group")).unwrap();

    // Single-file mode answers for one record's parse, both ways.
    let output = run(&["inspect", &dir.join("kura.group").display().to_string()]);
    assert_eq!(output.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&output.stdout).contains("ok"));
    std::fs::write(dir.join("kura.group"), b"{not json").unwrap();
    let output = run(&["inspect", &dir.join("kura.group").display().to_string()]);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("CORRUPT"));

    let _ = std::fs::remove_dir_all(&dir);
}

/// `--json`'s group-side reporting: a dangling member-context reference
/// as a `GroupRow` note, a dropped nesting edge as a top-level notice,
/// and single-file `.group` inspect answering the same `GroupRow` shape
/// — over the same fixtures
/// `inspect_flags_group_trouble_and_previews_boot_repairs` pins for the
/// human-readable path.
#[test]
fn inspect_json_reports_group_notes_and_nesting_drops() {
    let dir = std::env::temp_dir().join(format!(
        "taguru-cli-inspect-groups-json-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let context = taguru::context::Context::default();
    std::fs::write(dir.join("sake.ctx"), context.to_bytes()).unwrap();
    std::fs::write(
        dir.join("kura.group"),
        "{\"description\": \"\", \"contexts\": [\"sake\", \"ghost\"], \"groups\": []}",
    )
    .unwrap();

    let output = run(&["inspect", "--json", &dir.display().to_string()]);
    let stdout_text = String::from_utf8_lossy(&output.stdout).into_owned();
    assert_eq!(output.status.code(), Some(0), "{stdout_text}");
    let report: serde_json::Value = serde_json::from_str(&stdout_text)
        .unwrap_or_else(|error| panic!("--json must be one JSON document: {error}"));
    let kura = report["groups"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["name"] == "kura")
        .expect("kura must be reported");
    assert_eq!(kura["status"], "ok");
    assert_eq!(kura["contexts"], 2);
    assert!(
        kura["notes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|note| note["kind"] == "dangling_context_reference" && note["level"] == "warning"),
        "{report}"
    );

    // A cycle: boot's repair drops one edge, reported as a top-level
    // notice (it names an edge between two groups, not one group's own
    // fact).
    std::fs::write(dir.join("cyc-a.group"), "{\"groups\": [\"cyc-b\"]}").unwrap();
    std::fs::write(dir.join("cyc-b.group"), "{\"groups\": [\"cyc-a\"]}").unwrap();
    let output = run(&["inspect", "--json", &dir.display().to_string()]);
    let stdout_text = String::from_utf8_lossy(&output.stdout).into_owned();
    assert_eq!(output.status.code(), Some(0), "{stdout_text}");
    let report: serde_json::Value = serde_json::from_str(&stdout_text)
        .unwrap_or_else(|error| panic!("--json must be one JSON document: {error}"));
    assert!(
        report["notices"].as_array().unwrap().iter().any(|notice| {
            notice["kind"] == "dropped_nesting_edge"
                && notice["message"]
                    .as_str()
                    .unwrap()
                    .contains("'cyc-b' → 'cyc-a'")
        }),
        "{report}"
    );
    std::fs::remove_file(dir.join("cyc-a.group")).unwrap();
    std::fs::remove_file(dir.join("cyc-b.group")).unwrap();

    // Directory-scan path: a corrupt .group is registered under an
    // empty placeholder record for reference checks (inspect_groups'
    // own comment explains why), but the JSON preview loop must not
    // ALSO emit an "ok" row for that same name — exactly one row,
    // status "corrupt". `find` alone would pass even with a stray
    // duplicate "ok" row present, so this counts rows by name instead.
    std::fs::write(dir.join("bad.group"), b"{not json").unwrap();
    let output = run(&["inspect", "--json", &dir.display().to_string()]);
    let stdout_text = String::from_utf8_lossy(&output.stdout).into_owned();
    assert_eq!(output.status.code(), Some(1), "{stdout_text}");
    let report: serde_json::Value = serde_json::from_str(&stdout_text)
        .unwrap_or_else(|error| panic!("--json must be one JSON document: {error}"));
    let bad_rows: Vec<&serde_json::Value> = report["groups"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["name"] == "bad")
        .collect();
    assert_eq!(
        bad_rows.len(),
        1,
        "a corrupt group must report exactly one row, not a duplicate 'ok' row too: {report}"
    );
    assert_eq!(bad_rows[0]["status"], "corrupt");
    assert!(report["corrupt"].as_u64().unwrap() >= 1, "{report}");
    std::fs::remove_file(dir.join("bad.group")).unwrap();

    // Single-file `.group` inspect answers the same GroupRow shape.
    let output = run(&[
        "inspect",
        "--json",
        &dir.join("kura.group").display().to_string(),
    ]);
    assert_eq!(output.status.code(), Some(0));
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|error| panic!("--json must be one JSON document: {error}"));
    assert_eq!(report["kind"], "group");
    assert_eq!(
        report["groups"][0]["name"],
        dir.join("kura.group").display().to_string()
    );
    assert_eq!(report["groups"][0]["contexts"], 2);

    std::fs::write(dir.join("kura.group"), b"{not json").unwrap();
    let output = run(&[
        "inspect",
        "--json",
        &dir.join("kura.group").display().to_string(),
    ]);
    assert_eq!(output.status.code(), Some(1));
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|error| panic!("--json must be one JSON document: {error}"));
    assert_eq!(report["groups"][0]["status"], "corrupt");

    let _ = std::fs::remove_dir_all(&dir);
}

/// A corrupt `.group` file is still a group at boot — `scan_groups`
/// registers its name with an empty record rather than dropping it, so
/// a sibling naming it as a child must not get a false "boot drops this
/// reference" warning, and it must still be counted in the total.
#[test]
fn inspect_does_not_flag_a_corrupt_child_group_as_a_dangling_reference() {
    let dir = std::env::temp_dir().join(format!(
        "taguru-cli-inspect-corrupt-child-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("parent.group"),
        "{\"description\": \"\", \"contexts\": [], \"groups\": [\"child\"]}",
    )
    .unwrap();
    std::fs::write(dir.join("child.group"), b"{not json").unwrap();

    let output = run(&["inspect", &dir.display().to_string()]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    // The corrupt child still fails the check (restoring it would reset
    // the record), but that is not a dangling-reference problem for
    // `parent` — boot keeps the edge, just to an empty group.
    assert_eq!(output.status.code(), Some(1), "{stdout}");
    assert!(stdout.contains("child: CORRUPT group"), "{stdout}");
    assert!(
        !stdout.contains("child group(s) have no group here"),
        "{stdout}"
    );
    assert!(
        stdout.contains("parent: ok  0 member context(s) · 1 child group(s)"),
        "{stdout}"
    );
    assert!(
        stdout.contains("child: ok  0 member context(s) · 0 child group(s)"),
        "{stdout}"
    );
    assert!(stdout.contains("total: 0 contexts · 2 groups"), "{stdout}");

    let _ = std::fs::remove_dir_all(&dir);
}

/// `taguru compact` offline: the report names the shrink, and inspect
/// vouches for the rewritten family.
#[test]
fn compact_rewrites_a_data_directory_offline() {
    let dir = common::scratch_dir("cli-compact");
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
    std::fs::write(
        dir.join("a.jsonl"),
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
         \"create\": {\"description\": \"d\"}}\n\
         {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n",
    )
    .expect("fixture must be writable");
    // A revision that drops the fact leaves dead records behind…
    std::fs::write(
        dir.join("b.jsonl"),
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\"}\n\
         {\"subject\": \"蔵\", \"label\": \"銘柄\", \"object\": \"青嶺\", \"weight\": 1.0}\n",
    )
    .expect("fixture must be writable");

    let data = dir.join("data");
    let run_in = |args: &[&str]| -> Output {
        Command::new(env!("CARGO_BIN_EXE_taguru"))
            .args(args)
            .env("TAGURU_DATA_DIR", &data)
            .env_remove("TAGURU_CONFIG")
            .env_remove("TAGURU_EMBED_URL")
            .output()
            .expect("binary must run")
    };
    let first = run_in(&["import", &dir.join("a.jsonl").display().to_string()]);
    assert_eq!(first.status.code(), Some(0), "{first:?}");
    let second = run_in(&["import", &dir.join("b.jsonl").display().to_string()]);
    assert_eq!(second.status.code(), Some(0), "{second:?}");

    // …which compact reclaims.
    let compacted = run_in(&["compact"]);
    assert_eq!(compacted.status.code(), Some(0), "{compacted:?}");
    let stdout = String::from_utf8_lossy(&compacted.stdout);
    assert!(stdout.contains("dead edge(s) shed"), "{stdout}");
    assert!(stdout.contains("1 of 1 context(s) rewritten"), "{stdout}");

    let inspected = run_in(&["inspect", &data.display().to_string()]);
    assert_eq!(inspected.status.code(), Some(0), "{inspected:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `compact --dry-run` (issue #371) reports the standing dead weight
/// and touches nothing — proven by compacting for real afterward and
/// seeing the exact same shed counts a fresh (never-previewed) run
/// would report.
#[test]
fn compact_dry_run_reports_dead_weight_without_rewriting() {
    let dir = common::scratch_dir("cli-compact-dry");
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
    let run_in = |args: &[&str]| -> Output {
        Command::new(env!("CARGO_BIN_EXE_taguru"))
            .args(args)
            .env("TAGURU_DATA_DIR", &data)
            .env_remove("TAGURU_CONFIG")
            .env_remove("TAGURU_EMBED_URL")
            .output()
            .expect("binary must run")
    };
    let first = run_in(&["import", &dir.join("a.jsonl").display().to_string()]);
    assert_eq!(first.status.code(), Some(0), "{first:?}");
    let second = run_in(&["import", &dir.join("b.jsonl").display().to_string()]);
    assert_eq!(second.status.code(), Some(0), "{second:?}");

    // Snapshot the image bytes before the dry run so a mutation, if
    // one somehow happened, would be caught even if the report text
    // itself looked plausible.
    let image = data.join("sake.ctx");
    let before = std::fs::read(&image).expect("image must exist after import");

    let dry = run_in(&["compact", "--dry-run"]);
    assert_eq!(dry.status.code(), Some(0), "{dry:?}");
    let dry_stdout = String::from_utf8_lossy(&dry.stdout);
    assert!(dry_stdout.contains("dead edge(s)"), "{dry_stdout}");
    assert!(dry_stdout.contains("dry run: 1 of 1"), "{dry_stdout}");
    assert!(
        !dry_stdout.contains("dead edge(s) shed"),
        "a dry run must not claim anything was shed: {dry_stdout}"
    );

    let after = std::fs::read(&image).expect("image must still exist");
    assert_eq!(before, after, "--dry-run must not rewrite the image");

    // A real compact afterward sheds exactly what a first-ever run
    // would — proof the dry run above did not already reclaim it.
    let real = run_in(&["compact"]);
    assert_eq!(real.status.code(), Some(0), "{real:?}");
    let real_stdout = String::from_utf8_lossy(&real.stdout);
    assert!(real_stdout.contains("dead edge(s) shed"), "{real_stdout}");
    assert!(
        real_stdout.contains("1 of 1 context(s) rewritten"),
        "{real_stdout}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// `compact --dry-run --json` and `compact --json` each emit exactly
/// one parseable JSON document on stdout — no human-readable line
/// mixed in — with the fields the plan promises.
#[test]
fn compact_json_emits_a_single_parseable_document_dry_run_and_real() {
    let dir = common::scratch_dir("cli-compact-json");
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
    let run_in = |args: &[&str]| -> Output {
        Command::new(env!("CARGO_BIN_EXE_taguru"))
            .args(args)
            .env("TAGURU_DATA_DIR", &data)
            .env_remove("TAGURU_CONFIG")
            .env_remove("TAGURU_EMBED_URL")
            .output()
            .expect("binary must run")
    };
    let first = run_in(&["import", &dir.join("a.jsonl").display().to_string()]);
    assert_eq!(first.status.code(), Some(0), "{first:?}");
    let second = run_in(&["import", &dir.join("b.jsonl").display().to_string()]);
    assert_eq!(second.status.code(), Some(0), "{second:?}");

    let dry = run_in(&["compact", "--dry-run", "--json"]);
    assert_eq!(dry.status.code(), Some(0), "{dry:?}");
    let dry_value: serde_json::Value = serde_json::from_slice(&dry.stdout)
        .unwrap_or_else(|error| panic!("--dry-run --json must be one JSON document: {error}"));
    let dry_rows = dry_value.as_array().expect("--dry-run --json is an array");
    assert_eq!(dry_rows.len(), 1);
    assert_eq!(dry_rows[0]["context"], "sake");
    assert!(dry_rows[0]["dead_edges"].as_u64().unwrap() > 0);
    assert!(dry_rows[0]["dead_ratio"].as_f64().unwrap() > 0.0);
    // A freshly booted CLI process hasn't loaded the context into
    // memory yet (nothing in this run has read or written it before
    // the dry run), so its stats are the last-saved snapshot, not a
    // live recomputation — correctly flagged as such.
    assert_eq!(dry_rows[0]["stats_are_snapshot"], true);

    let real = run_in(&["compact", "--json"]);
    assert_eq!(real.status.code(), Some(0), "{real:?}");
    let real_value: serde_json::Value = serde_json::from_slice(&real.stdout)
        .unwrap_or_else(|error| panic!("--json must be one JSON document: {error}"));
    let real_rows = real_value.as_array().expect("--json is an array");
    assert_eq!(real_rows.len(), 1);
    assert_eq!(real_rows[0]["name"], "sake");
    assert!(real_rows[0]["dead_edges"].as_u64().unwrap() > 0);
    assert!(real_rows[0]["bytes_before"].as_u64().unwrap() > 0);

    let _ = std::fs::remove_dir_all(&dir);
}

/// `--parallel N` must produce stdout byte-for-byte identical to the
/// sequential (default) run, whatever N is or however its worker
/// threads happen to race — the property the shared-queue reordering
/// in `compact.rs` exists to guarantee.
#[test]
fn compact_parallel_output_matches_the_sequential_run_byte_for_byte() {
    fn seed(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "taguru-cli-compact-par-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
        // Three contexts, each carrying one fact from a.md — created
        // in an order (charlie, alpha, bravo) that is NOT alphabetical,
        // so a run over "every context" only comes out sorted if
        // something actually sorts it.
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
        // Restating a.md per context with a different fact retracts
        // the first, leaving dead edges for compact to reclaim.
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
        dir
    }

    fn run_in(dir: &std::path::Path, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_taguru"))
            .args(args)
            .env("TAGURU_DATA_DIR", dir.join("data"))
            .env_remove("TAGURU_CONFIG")
            .env_remove("TAGURU_EMBED_URL")
            .output()
            .expect("binary must run")
    }

    let seq_dir = seed("seq");
    let par_dir = seed("par");
    for dir in [&seq_dir, &par_dir] {
        let first = run_in(dir, &["import", &dir.join("a.jsonl").display().to_string()]);
        assert_eq!(first.status.code(), Some(0), "{first:?}");
        let second = run_in(dir, &["import", &dir.join("b.jsonl").display().to_string()]);
        assert_eq!(second.status.code(), Some(0), "{second:?}");
    }

    let sequential = run_in(&seq_dir, &["compact"]);
    assert_eq!(sequential.status.code(), Some(0), "{sequential:?}");
    // More workers than contexts, so every worker races for the queue.
    let parallel = run_in(&par_dir, &["compact", "--parallel", "8"]);
    assert_eq!(parallel.status.code(), Some(0), "{parallel:?}");

    let sequential_stdout = String::from_utf8_lossy(&sequential.stdout).into_owned();
    let parallel_stdout = String::from_utf8_lossy(&parallel.stdout).into_owned();
    assert!(
        sequential_stdout.contains("3 of 3 context(s) rewritten"),
        "{sequential_stdout}"
    );
    assert_eq!(
        sequential_stdout, parallel_stdout,
        "--parallel output must match the sequential run byte for byte"
    );

    let _ = std::fs::remove_dir_all(&seq_dir);
    let _ = std::fs::remove_dir_all(&par_dir);
}

/// A bad `--parallel` value is refused with the usual usage-error
/// shape, before anything boots.
#[test]
fn compact_rejects_a_non_positive_parallel_value() {
    let output = run(&["compact", "--parallel", "0"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("--parallel needs an integer"),
        "{output:?}"
    );

    let output = run(&["compact", "--parallel", "nope"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("--parallel needs an integer"),
        "{output:?}"
    );
}

/// An unknown context is a per-item failure on every local path —
/// sequential, `--parallel`, and `--dry-run` — while the known
/// context's work still lands: the summary counts the failure and the
/// exit code is nonzero. The remote twin
/// (`remote_compact::an_unknown_context_counts_as_a_failure_and_the_rest_still_lands`)
/// already pins this contract for `--url`; these are the offline
/// halves.
#[test]
fn compact_counts_an_unknown_context_as_a_failure_on_every_local_path() {
    let dir = common::scratch_dir("cli-compact-unknown");
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
    std::fs::write(
        dir.join("a.jsonl"),
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
         \"create\": {\"description\": \"d\"}}\n\
         {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n",
    )
    .expect("fixture must be writable");
    let data = dir.join("data");
    let run_in = |args: &[&str]| -> Output {
        Command::new(env!("CARGO_BIN_EXE_taguru"))
            .args(args)
            .env("TAGURU_DATA_DIR", &data)
            .env_remove("TAGURU_CONFIG")
            .env_remove("TAGURU_EMBED_URL")
            .output()
            .expect("binary must run")
    };
    let seeded = run_in(&["import", &dir.join("a.jsonl").display().to_string()]);
    assert_eq!(seeded.status.code(), Some(0), "{seeded:?}");

    for args in [
        &["compact", "sake", "nope"][..],
        &["compact", "--parallel", "2", "sake", "nope"][..],
    ] {
        let output = run_in(args);
        assert_eq!(output.status.code(), Some(1), "{args:?}: {output:?}");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            stdout.contains("1 of 2 context(s) rewritten"),
            "{args:?}: {stdout}"
        );
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("context 'nope': no such context"),
            "{args:?}: {output:?}"
        );
    }

    let dry = run_in(&["compact", "--dry-run", "sake", "nope"]);
    assert_eq!(dry.status.code(), Some(1), "{dry:?}");
    assert!(
        String::from_utf8_lossy(&dry.stdout).contains("of 2 context(s) carry dead weight"),
        "{dry:?}"
    );
    assert!(
        String::from_utf8_lossy(&dry.stderr).contains("context 'nope': no such context"),
        "{dry:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `--config` given ONCE loads the file — the given-twice guard
/// rejects the second flag, never the first.
#[test]
fn compact_accepts_a_single_config_flag() {
    let dir = common::scratch_dir("cli-compact-config-once");
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
    std::fs::write(
        dir.join("a.jsonl"),
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
         \"create\": {\"description\": \"d\"}}\n\
         {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n",
    )
    .expect("fixture must be writable");
    let data = dir.join("data");
    let seeded = Command::new(env!("CARGO_BIN_EXE_taguru"))
        .args(["import", &dir.join("a.jsonl").display().to_string()])
        .env("TAGURU_DATA_DIR", &data)
        .env_remove("TAGURU_CONFIG")
        .env_remove("TAGURU_EMBED_URL")
        .output()
        .expect("binary must run");
    assert_eq!(seeded.status.code(), Some(0), "{seeded:?}");

    let config = dir.join("taguru.env");
    std::fs::write(&config, format!("TAGURU_DATA_DIR={}\n", data.display()))
        .expect("config must be writable");
    let mut command = Command::new(env!("CARGO_BIN_EXE_taguru"));
    common::scrub_taguru_env(&mut command);
    let output = command
        .args([
            "compact",
            "--dry-run",
            "--config",
            &config.display().to_string(),
        ])
        .output()
        .expect("binary must run");
    assert_eq!(output.status.code(), Some(0), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("of 1 context(s) carry dead weight"),
        "{output:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ============================== evaluate --thresholds (issue #276) ==============================
//
// Every case here is a usage/input error (exit 2), reported before any
// network call — [`crate::evaluate::run_evaluate`] validates
// `--thresholds` right after `eval.jsonl` itself, ahead of URL/config
// resolution, so none of these need a running server (ADR 0004 §9.3).

/// A scratch directory holding one `eval.jsonl` with a single case
/// declaring no expectations, ready for a `--thresholds FILE` next to
/// it. Removed by the caller.
fn eval_scratch_dir(tag: &str) -> PathBuf {
    let dir = common::scratch_dir(&format!("cli-evaluate-{tag}"));
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
    let eval_path = dir.join("eval.jsonl");
    std::fs::write(
        &eval_path,
        "{\"taguru_eval\":1}\n{\"case_id\":\"c1\",\"query\":\"q\"}\n",
    )
    .expect("eval.jsonl must be writable");
    dir
}

fn write_thresholds(dir: &std::path::Path, contents: &str) -> PathBuf {
    let path = dir.join("thresholds.json");
    std::fs::write(&path, contents).expect("thresholds.json must be writable");
    path
}

#[test]
fn evaluate_rejects_a_thresholds_file_with_the_wrong_stamp() {
    let dir = eval_scratch_dir("bad-stamp");
    let thresholds = write_thresholds(&dir, "{\"taguru_evaluate_thresholds\":2}");
    let output = run(&[
        "evaluate",
        "--eval",
        dir.join("eval.jsonl").to_str().unwrap(),
        "--context",
        "sake",
        "--thresholds",
        thresholds.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("taguru_evaluate_thresholds"), "{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn evaluate_rejects_a_thresholds_file_naming_an_unknown_aggregate_metric() {
    let dir = eval_scratch_dir("unknown-metric");
    let thresholds = write_thresholds(
        &dir,
        "{\"taguru_evaluate_thresholds\":1,\
         \"aggregate\":{\"not.a.real.metric\":{\"min\":0.5}}}",
    );
    let output = run(&[
        "evaluate",
        "--eval",
        dir.join("eval.jsonl").to_str().unwrap(),
        "--context",
        "sake",
        "--thresholds",
        thresholds.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not.a.real.metric"), "{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn evaluate_rejects_a_thresholds_file_naming_an_unknown_case_id_override() {
    let dir = eval_scratch_dir("unknown-case-id");
    let thresholds = write_thresholds(
        &dir,
        "{\"taguru_evaluate_thresholds\":1,\
         \"cases\":{\"overrides\":{\"no-such-case\":{\"recall.recall_at_k\":{\"min\":0.5}}}}}",
    );
    let output = run(&[
        "evaluate",
        "--eval",
        dir.join("eval.jsonl").to_str().unwrap(),
        "--context",
        "sake",
        "--thresholds",
        thresholds.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no-such-case"), "{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn evaluate_rejects_a_thresholds_file_naming_a_non_case_scoped_metric_in_cases_default() {
    let dir = eval_scratch_dir("non-case-scoped");
    let thresholds = write_thresholds(
        &dir,
        "{\"taguru_evaluate_thresholds\":1,\
         \"cases\":{\"default\":{\"latency.resolve_ms\":{\"max\":100.0}}}}",
    );
    let output = run(&[
        "evaluate",
        "--eval",
        dir.join("eval.jsonl").to_str().unwrap(),
        "--context",
        "sake",
        "--thresholds",
        thresholds.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("latency.resolve_ms"), "{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn evaluate_rejects_a_bound_with_neither_min_nor_max() {
    let dir = eval_scratch_dir("empty-bound");
    let thresholds = write_thresholds(
        &dir,
        "{\"taguru_evaluate_thresholds\":1,\
         \"aggregate\":{\"recall.recall_at_k\":{}}}",
    );
    let output = run(&[
        "evaluate",
        "--eval",
        dir.join("eval.jsonl").to_str().unwrap(),
        "--context",
        "sake",
        "--thresholds",
        thresholds.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn evaluate_rejects_a_bound_with_min_greater_than_max() {
    let dir = eval_scratch_dir("inverted-bound");
    let thresholds = write_thresholds(
        &dir,
        "{\"taguru_evaluate_thresholds\":1,\
         \"aggregate\":{\"recall.recall_at_k\":{\"min\":0.9,\"max\":0.1}}}",
    );
    let output = run(&[
        "evaluate",
        "--eval",
        dir.join("eval.jsonl").to_str().unwrap(),
        "--context",
        "sake",
        "--thresholds",
        thresholds.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn evaluate_thresholds_flag_given_twice_is_a_usage_error() {
    let dir = eval_scratch_dir("dup-thresholds");
    let thresholds = write_thresholds(
        &dir,
        "{\"taguru_evaluate_thresholds\":1,\"aggregate\":{\"recall.recall_at_k\":{\"min\":0.5}}}",
    );
    let output = run(&[
        "evaluate",
        "--eval",
        dir.join("eval.jsonl").to_str().unwrap(),
        "--context",
        "sake",
        "--thresholds",
        thresholds.to_str().unwrap(),
        "--thresholds",
        thresholds.to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--thresholds given twice"), "{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Every value-taking flag rejects a second occurrence the same way —
/// `compact` and `extract` silently took the last value until this
/// guard, letting a malformed script run against the wrong config,
/// context, or output directory.
#[test]
fn compact_and_extract_flags_given_twice_are_usage_errors() {
    for (args, flag) in [
        (
            vec!["compact", "--config", "a.env", "--config", "b.env"],
            "--config",
        ),
        (
            vec!["compact", "--url", "http://a", "--url", "http://b"],
            "--url",
        ),
        (
            vec!["compact", "--parallel", "2", "--parallel", "3"],
            "--parallel",
        ),
        (
            vec!["extract", "--context", "a", "--context", "b"],
            "--context",
        ),
        (vec!["extract", "--out", "dir1", "--out", "dir2"], "--out"),
        (
            vec!["extract", "--config", "a.env", "--config", "b.env"],
            "--config",
        ),
    ] {
        let output = run(&args);
        assert_eq!(output.status.code(), Some(2), "{args:?}: {output:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!("{flag} given twice")),
            "{args:?}: {stderr}"
        );
    }
}

/// `export --out`/`--config`/`--url` reject a second occurrence, same
/// convention as `compact`/`extract` above — and a value that itself
/// looks like a flag (`--out --out DIR`, the first `--out` swallowing
/// the second AS ITS PATH) does not silently slip past that guard by
/// masquerading as a legitimate first value.
#[test]
fn export_flags_given_twice_are_usage_errors() {
    for (args, flag) in [
        (vec!["export", "--out", "dir1", "--out", "dir2"], "--out"),
        (
            vec!["export", "--config", "a.env", "--config", "b.env"],
            "--config",
        ),
        (
            vec!["export", "--url", "http://a", "--url", "http://b"],
            "--url",
        ),
    ] {
        let output = run(&args);
        assert_eq!(output.status.code(), Some(2), "{args:?}: {output:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!("{flag} given twice")),
            "{args:?}: {stderr}"
        );
    }
}

#[test]
fn export_flag_values_that_look_like_flags_are_rejected_not_swallowed() {
    for (args, needs) in [
        (vec!["export", "--out", "--out", "dir"], "--out needs"),
        (
            vec!["export", "--config", "--config", "a.env"],
            "--config needs",
        ),
        (vec!["export", "--url", "--url", "http://a"], "--url needs"),
    ] {
        let output = run(&args);
        assert_eq!(output.status.code(), Some(2), "{args:?}: {output:?}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(needs), "{args:?}: {stderr}");
    }
}

#[test]
fn evaluate_thresholds_file_that_cannot_be_read_is_a_usage_error() {
    let dir = eval_scratch_dir("missing-thresholds-file");
    let output = run(&[
        "evaluate",
        "--eval",
        dir.join("eval.jsonl").to_str().unwrap(),
        "--context",
        "sake",
        "--thresholds",
        dir.join("does-not-exist.json").to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn top_level_help_documents_exit_code_3_for_threshold_violations() {
    let output = run(&["--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("EXIT CODES"), "{stdout}");
    assert!(stdout.contains('3'), "{stdout}");
}

/// Issue #248 item 6: the top-level --help must name every
/// sub-subcommand `benchmark`/`communities` has, not just the first
/// one — a regression guard, not a re-derivation of the USAGE text.
#[test]
fn top_level_help_covers_every_sub_subcommand_and_flag() {
    let output = run(&["--help"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    for needle in [
        "taguru benchmark extract",
        "taguru benchmark compare",
        "taguru benchmark search",
        "taguru evaluate compare",
        "--group",
        "--into",
    ] {
        assert!(stdout.contains(needle), "missing {needle:?}: {stdout}");
    }
}

// ADR 0008 §10: the stdio bridge propagates trace context on its
// outbound calls, in both directions — injecting from its own active
// span, and adopting a parent an MCP client attached via `params._meta`.

/// A raw HTTP listener that records every request it receives (request
/// line plus headers, header names lower-cased) and answers each with
/// a fixed 200 JSON body. Multiple connections on purpose: the bridge
/// makes an unauthenticated `GET /protocol` probe before the stdio
/// loop even starts (no active span at that point, so correctly
/// carries no `traceparent`), and a test that only captured the first
/// connection would silently assert on the probe instead of the tool
/// call under test. The bridge does not parse or validate a tool's
/// response shape beyond its status code, so any valid JSON body is a
/// passing "server."
type CapturedRequests =
    std::sync::Arc<std::sync::Mutex<Vec<(String, std::collections::HashMap<String, String>)>>>;

struct HeaderCapture {
    addr: std::net::SocketAddr,
    requests: CapturedRequests,
}

impl HeaderCapture {
    fn start() -> Self {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("capture must bind");
        let addr = listener.local_addr().unwrap();
        let requests = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = requests.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut buffer = Vec::new();
                let mut chunk = [0u8; 4096];
                let header_end = loop {
                    match stream.read(&mut chunk) {
                        Ok(0) | Err(_) => break None,
                        Ok(n) => buffer.extend_from_slice(&chunk[..n]),
                    }
                    if let Some(at) = buffer.windows(4).position(|w| w == b"\r\n\r\n") {
                        break Some(at + 4);
                    }
                };
                if let Some(header_end) = header_end {
                    let text = String::from_utf8_lossy(&buffer[..header_end]).to_string();
                    let mut lines = text.lines();
                    let request_line = lines.next().unwrap_or_default().to_string();
                    let mut parsed = std::collections::HashMap::new();
                    for line in lines {
                        if let Some((name, value)) = line.split_once(':') {
                            parsed
                                .insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
                        }
                    }
                    sink.lock().unwrap().push((request_line, parsed));
                }
                let _ = stream.write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                      Content-Length: 2\r\nConnection: close\r\n\r\n{}",
                );
            }
        });
        Self { addr, requests }
    }

    /// Blocks (bounded) for a request whose request line contains
    /// `path`, then returns its headers — skipping the startup probe
    /// (`GET /protocol`) and any other unrelated connection.
    fn wait_for(&self, path: &str) -> std::collections::HashMap<String, String> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if let Some((_, headers)) = self
                .requests
                .lock()
                .unwrap()
                .iter()
                .find(|(line, _)| line.contains(path))
            {
                return headers.clone();
            }
            assert!(
                std::time::Instant::now() < deadline,
                "no request matching {path:?} reached the capture listener within 10s"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
}

/// One `tools/call` sent over the bridge's stdin, its first reply line
/// read back with a bounded wait — the same pattern every other bridge
/// test in this file uses.
fn call_bridge_tool(bridge: &mut std::process::Child, request: serde_json::Value) -> String {
    use std::io::Write;
    let mut stdin = bridge.stdin.take().unwrap();
    writeln!(stdin, "{request}").unwrap();
    drop(stdin);
    let stdout = bridge.stdout.take().unwrap();
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut lines = BufReader::new(stdout).lines();
        let _ = sender.send(lines.next().and_then(Result::ok));
    });
    receiver
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("the bridge must answer the tool call")
        .expect("one JSON-RPC response line")
}

#[test]
fn the_stdio_bridge_injects_traceparent_into_its_outbound_calls() {
    let capture = HeaderCapture::start();

    let mut bridge = Command::new(env!("CARGO_BIN_EXE_taguru-mcp"))
        .env("TAGURU_URL", format!("http://{}", capture.addr))
        .env_remove("TAGURU_API_TOKEN")
        // The endpoint need not be reachable — building the exporter
        // (what makes `trace::enabled()` true) does not require
        // connectivity, only actually delivering a batch would.
        .env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("bridge must spawn");

    let reply = call_bridge_tool(
        &mut bridge,
        serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "list_contexts", "arguments": {}}
        }),
    );
    let headers = capture.wait_for("/contexts");

    let _ = bridge.kill();
    let _ = bridge.wait();

    assert!(reply.contains(r#""id":1"#), "{reply}");
    let traceparent = headers
        .get("traceparent")
        .unwrap_or_else(|| panic!("no traceparent header among {headers:?}"));
    // `{version}-{trace-id:32}-{parent-id:16}-{flags:2}` — the bridge
    // minted a fresh trace (no `_meta` was sent), so only the shape is
    // checked, not a specific id.
    let parts: Vec<&str> = traceparent.split('-').collect();
    assert_eq!(parts.len(), 4, "{traceparent}");
    assert_eq!(parts[0], "00", "{traceparent}");
    assert_eq!(parts[1].len(), 32, "{traceparent}");
    assert_eq!(parts[2].len(), 16, "{traceparent}");
}

#[test]
fn the_stdio_bridge_adopts_a_parent_from_meta() {
    let capture = HeaderCapture::start();

    let mut bridge = Command::new(env!("CARGO_BIN_EXE_taguru-mcp"))
        .env("TAGURU_URL", format!("http://{}", capture.addr))
        .env_remove("TAGURU_API_TOKEN")
        .env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("bridge must spawn");

    let inbound_trace_id = "4bf92f3577b34da6a3ce929d0e0e4736";
    let reply = call_bridge_tool(
        &mut bridge,
        serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "list_contexts", "arguments": {},
                "_meta": {
                    "traceparent": format!("00-{inbound_trace_id}-00f067aa0ba902b7-01"),
                }
            }
        }),
    );
    let headers = capture.wait_for("/contexts");

    let _ = bridge.kill();
    let _ = bridge.wait();

    assert!(reply.contains(r#""id":1"#), "{reply}");
    let traceparent = headers
        .get("traceparent")
        .unwrap_or_else(|| panic!("no traceparent header among {headers:?}"));
    // Same trace as the client attached via `_meta` — proof the
    // bridge adopted it as `taguru.tool_call`'s parent rather than
    // starting a fresh trace.
    assert!(
        traceparent.starts_with(&format!("00-{inbound_trace_id}-")),
        "expected trace id {inbound_trace_id} to propagate, got {traceparent}"
    );
    // The span id must NOT be the caller's own `00f067aa0ba902b7` —
    // the outbound call is parented under the bridge's OWN
    // `taguru.tool_call` span, not a bare copy of the inbound header.
    assert!(
        !traceparent.contains("00f067aa0ba902b7"),
        "span id must be the bridge's own span, not the caller's: {traceparent}"
    );
}

#[test]
fn the_stdio_bridge_starts_a_fresh_trace_on_a_malformed_meta_traceparent() {
    let capture = HeaderCapture::start();

    let mut bridge = Command::new(env!("CARGO_BIN_EXE_taguru-mcp"))
        .env("TAGURU_URL", format!("http://{}", capture.addr))
        .env_remove("TAGURU_API_TOKEN")
        .env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("bridge must spawn");

    let reply = call_bridge_tool(
        &mut bridge,
        serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "list_contexts", "arguments": {},
                "_meta": {"traceparent": "not-a-traceparent"}
            }
        }),
    );
    let headers = capture.wait_for("/contexts");

    let _ = bridge.kill();
    let _ = bridge.wait();

    assert!(reply.contains(r#""id":1"#), "{reply}");
    // Garbage in `_meta` is "no parent", not an error — the bridge
    // still answers and still injects a `traceparent` of its own, just
    // rooted in a fresh trace rather than the malformed one.
    let traceparent = headers
        .get("traceparent")
        .unwrap_or_else(|| panic!("no traceparent header among {headers:?}"));
    let parts: Vec<&str> = traceparent.split('-').collect();
    assert_eq!(parts.len(), 4, "{traceparent}");
    assert_eq!(parts[0], "00", "{traceparent}");
    assert_eq!(parts[1].len(), 32, "{traceparent}");
    assert_eq!(parts[2].len(), 16, "{traceparent}");
}

#[test]
fn the_stdio_bridge_forwards_tracestate_from_meta() {
    let capture = HeaderCapture::start();

    let mut bridge = Command::new(env!("CARGO_BIN_EXE_taguru-mcp"))
        .env("TAGURU_URL", format!("http://{}", capture.addr))
        .env_remove("TAGURU_API_TOKEN")
        .env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("bridge must spawn");

    let inbound_trace_id = "4bf92f3577b34da6a3ce929d0e0e4736";
    let reply = call_bridge_tool(
        &mut bridge,
        serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {
                "name": "list_contexts", "arguments": {},
                "_meta": {
                    "traceparent": format!("00-{inbound_trace_id}-00f067aa0ba902b7-01"),
                    "tracestate": "vendor=abc123",
                }
            }
        }),
    );
    let headers = capture.wait_for("/contexts");

    let _ = bridge.kill();
    let _ = bridge.wait();

    assert!(reply.contains(r#""id":1"#), "{reply}");
    assert_eq!(
        headers.get("tracestate").map(String::as_str),
        Some("vendor=abc123"),
        "{headers:?}"
    );
}

// Issue #248 item 8: `calibrate`/`communities` gain the same
// unparsable-URL rejection `evaluate`/`benchmark search` already have
// (issue #289 / #281 / #288) — a string that fails to parse as a URL,
// but isn't caught by `reject_userinfo` (which deliberately leaves an
// unparsable `base` alone), must not reach `Api::new` or any report.

#[test]
fn calibrate_rejects_a_url_that_does_not_parse() {
    let dir = common::scratch_dir("cli-calibrate");
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
    let probes = dir.join("probes.tsv");
    std::fs::write(&probes, "a paraphrase\texpected\n").expect("probes file must be writable");

    let output = run(&[
        "calibrate",
        "--context",
        "sake",
        "--probes",
        &probes.display().to_string(),
        "not-a-url",
    ]);
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("could not be parsed as a URL"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// Issue #248 item 9: `route` is renamed to `router` (every design doc
// and the module itself already called it "the router"); `route`
// keeps working as a deprecated alias, warning on stderr.

#[test]
fn router_help_flag_prints_usage_and_exits_zero_with_no_warning() {
    let output = run(&["router", "--help"]);
    assert_eq!(output.status.code(), Some(0));
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("taguru router"),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(String::from_utf8_lossy(&output.stderr).is_empty());
}

#[test]
fn route_still_works_as_a_deprecated_alias_for_router() {
    let router_output = run(&["router", "--help"]);
    let route_output = run(&["route", "--help"]);

    assert_eq!(route_output.status.code(), Some(0));
    // Identical usage text either name is invoked under — the alias
    // changes nothing about what gets printed, only that it warns.
    assert_eq!(
        route_output.stdout, router_output.stdout,
        "'route' and 'router' --help must print the exact same usage text"
    );
    let stderr = String::from_utf8_lossy(&route_output.stderr);
    assert!(
        stderr.contains("'route' is a deprecated alias for 'router'"),
        "{stderr}"
    );
}

// Issue #248 item 2: `import`/`export`/`compact`/`extract` now fall
// back to TAGURU_CONFIG when --config is absent, like every other
// subcommand already does — previously these four silently ignored
// it. A config file with a misspelled TAGURU_* key still earns the
// typo warning, which only fires if the file was actually loaded.

#[test]
fn import_honors_taguru_config_when_the_flag_is_absent() {
    let (dir, config) = write_config("import-envcfg", "TAGURU_CAHCE_BYTES=1\n");
    let output = run_with_env(
        &["import", "/nonexistent-item"],
        &[("TAGURU_CONFIG", &config.display().to_string())],
    );
    let _ = std::fs::remove_dir_all(&dir);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("TAGURU_CAHCE_BYTES is not a variable taguru reads"),
        "{stderr}"
    );
}

#[test]
fn export_honors_taguru_config_when_the_flag_is_absent() {
    let (dir, config) = write_config("export-envcfg", "TAGURU_CAHCE_BYTES=1\n");
    let out_dir = dir.join("out");
    let output = run_with_env(
        &["export", "--out", &out_dir.display().to_string()],
        &[("TAGURU_CONFIG", &config.display().to_string())],
    );
    let _ = std::fs::remove_dir_all(&dir);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("TAGURU_CAHCE_BYTES is not a variable taguru reads"),
        "{stderr}"
    );
}

#[test]
fn compact_honors_taguru_config_when_the_flag_is_absent() {
    let (dir, config) = write_config("compact-envcfg", "TAGURU_CAHCE_BYTES=1\n");
    let output = run_with_env(
        &["compact"],
        &[("TAGURU_CONFIG", &config.display().to_string())],
    );
    let _ = std::fs::remove_dir_all(&dir);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("TAGURU_CAHCE_BYTES is not a variable taguru reads"),
        "{stderr}"
    );
}

#[test]
fn extract_honors_taguru_config_when_the_flag_is_absent() {
    let (dir, config) = write_config("extract-envcfg", "TAGURU_CAHCE_BYTES=1\n");
    let out_dir = dir.join("out");
    let output = run_with_env(
        &[
            "extract",
            "--context",
            "sake",
            "--out",
            &out_dir.display().to_string(),
            "/nonexistent-item",
        ],
        &[("TAGURU_CONFIG", &config.display().to_string())],
    );
    let _ = std::fs::remove_dir_all(&dir);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("TAGURU_CAHCE_BYTES is not a variable taguru reads"),
        "{stderr}"
    );
}

// Issue #248 item 3: `restore`'s usage errors now name the subcommand
// and point at its own --help, like every other subcommand, instead
// of the bare `usage_error` format that pointed at `taguru --help`.

#[test]
fn restore_usage_errors_name_the_subcommand_and_its_own_help() {
    let output = run(&["restore", "--out", "/tmp/wherever", "one-url", "two-url"]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("taguru: restore:") && stderr.contains("taguru restore --help"),
        "{stderr}"
    );
}

#[test]
fn restore_help_flag_prints_usage_and_exits_zero() {
    let output = run(&["restore", "--help"]);
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("usage: taguru restore --out DIR"),
        "{stdout}"
    );
}

#[test]
fn communities_rejects_a_url_that_does_not_parse() {
    let output = run(&["communities", "--context", "sake", "not-a-url"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("could not be parsed as a URL"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Issue #728: a group record naming a member that STILL does not
/// exist after every batch of the run applied refuses the whole group
/// set — the batches stay landed, the exit code says failure, and
/// `--json` carries the whole-set refusal in `error` (the ingest-layer
/// glue over the registry's own whole-set judgment, which
/// `restore_groups_judges_the_whole_set_before_writing_anything`
/// already pins at its own layer).
#[test]
fn import_refuses_the_whole_group_set_but_keeps_batches_landed() {
    let dir = common::scratch_dir("cli-group-set-refusal");
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
    let batch = dir.join("a.jsonl");
    std::fs::write(
        &batch,
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"a.md\", \
          \"create\": {\"description\": \"酒\"}}\n\
         {\"subject\": \"青嶺\", \"label\": \"銘柄\", \"object\": \"酒\", \"weight\": 1.0}\n",
    )
    .expect("fixture must be writable");
    let group = dir.join("g.jsonl");
    // A perfectly valid sibling rides the SAME set: the whole-set
    // judgment must refuse it too — writing the valid record before
    // judging the set would be exactly the defect this pins.
    std::fs::write(
        &group,
        "{\"taguru_group\": 1, \"name\": \"kura\", \"contexts\": [\"sake\", \"ghost\"]}\n\
         {\"taguru_group\": 1, \"name\": \"valid\", \"contexts\": [\"sake\"]}\n",
    )
    .expect("fixture must be writable");
    let data_dir = dir.join("data");

    let refused = run_with_env(
        &[
            "import",
            "--json",
            &batch.display().to_string(),
            &group.display().to_string(),
        ],
        &[("TAGURU_DATA_DIR", &data_dir.display().to_string())],
    );
    assert_eq!(refused.status.code(), Some(1), "{refused:?}");
    let report: serde_json::Value =
        serde_json::from_slice(&refused.stdout).expect("--json must answer one document");
    assert!(
        report["error"]
            .as_str()
            .is_some_and(|error| error.contains("ghost")),
        "the whole-set refusal must name the missing member: {report}"
    );
    assert_eq!(
        report["batches"][0]["context"],
        serde_json::json!("sake"),
        "the batch before the group refusal stays landed: {report}"
    );
    assert!(
        report.get("groups").is_none() || report["groups"].as_array().is_some_and(Vec::is_empty),
        "no group may report an outcome when the set refused whole: {report}"
    );

    // The batches really landed: create the missing member, re-run the
    // same group record alone, and the restore now succeeds against
    // the state the first run left behind.
    let ghost = dir.join("ghost.jsonl");
    std::fs::write(
        &ghost,
        "{\"taguru_batch\": 1, \"context\": \"ghost\", \"source\": \"g.md\", \"create\": {}}\n",
    )
    .expect("fixture must be writable");
    let healed = run_with_env(
        &[
            "import",
            "--json",
            &ghost.display().to_string(),
            &group.display().to_string(),
        ],
        &[("TAGURU_DATA_DIR", &data_dir.display().to_string())],
    );
    assert_eq!(healed.status.code(), Some(0), "{healed:?}");
    let healed_report: serde_json::Value =
        serde_json::from_slice(&healed.stdout).expect("--json must answer one document");
    let outcomes: Vec<(&str, &str)> = healed_report["groups"]
        .as_array()
        .expect("both groups restore")
        .iter()
        .map(|group| {
            (
                group["name"].as_str().unwrap(),
                group["outcome"].as_str().unwrap(),
            )
        })
        .collect();
    // "created" for BOTH proves the refused run wrote neither — the
    // valid sibling included, which a set-judged-after-writes defect
    // would have left behind as "replaced"/"unchanged" here.
    assert_eq!(
        outcomes,
        vec![("kura", "created"), ("valid", "created")],
        "{healed_report}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// Issue #728: Pass 2 applies file by file — one file's APPLY-stage
/// refusal (a context that does not exist and no create meta to make
/// it, which Pass 1's parse-only validation cannot see) must not stop
/// the files after it.
#[test]
fn a_failing_files_apply_does_not_stop_the_files_after_it() {
    let dir = common::scratch_dir("cli-pass2-continues");
    std::fs::create_dir_all(&dir).expect("scratch dir must be creatable");
    let broken = dir.join("a.jsonl");
    std::fs::write(
        &broken,
        "{\"taguru_batch\": 1, \"context\": \"missing\", \"source\": \"a.md\"}\n\
         {\"subject\": \"x\", \"label\": \"y\", \"object\": \"z\", \"weight\": 1.0}\n",
    )
    .expect("fixture must be writable");
    let healthy = dir.join("b.jsonl");
    std::fs::write(
        &healthy,
        "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"b.md\", \"create\": {}}\n\
         {\"subject\": \"青嶺\", \"label\": \"銘柄\", \"object\": \"酒\", \"weight\": 1.0}\n",
    )
    .expect("fixture must be writable");
    let data_dir = dir.join("data");

    let output = run_with_env(
        &[
            "import",
            &broken.display().to_string(),
            &healthy.display().to_string(),
        ],
        &[("TAGURU_DATA_DIR", &data_dir.display().to_string())],
    );
    assert_eq!(output.status.code(), Some(1), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("missing"), "{stderr}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("1 of 2 batch(es) applied"),
        "the healthy file after the failure must still apply: {stdout}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
