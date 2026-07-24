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

/// A scratch directory holding a config file (and doubling as the data
/// directory the file points at). Removed by the caller.
fn write_config(tag: &str, lines: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("taguru-cli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
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
    let dir = std::env::temp_dir().join(format!("taguru-cli-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
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
fn health_answers_ok_against_a_live_server() {
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
    assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "ok");
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

#[test]
fn inspect_verifies_a_directory_and_a_single_image() {
    let dir = std::env::temp_dir().join(format!("taguru-cli-inspect-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
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

#[test]
fn inspect_fails_an_image_whose_bytes_rotted_in_place() {
    // The backup-verification case the checksum footer exists for: one
    // flipped bit in a stored name leaves the image structurally
    // perfect — every id in range, every chain intact — so before the
    // footer this passed inspection and loaded as truth. Now it must
    // fail, loudly, BEFORE a restore spends it.
    let dir = std::env::temp_dir().join(format!("taguru-cli-bitrot-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
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
    let dir = std::env::temp_dir().join(format!("taguru-cli-corrupt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
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

#[test]
fn inspect_reports_a_torn_wal_tail_without_healing_it() {
    // Fix 3: a directory audit must never mutate what it audits. A WAL
    // whose last record was cut short by a crash mid-append is not
    // corruption — the server heals it on its next load — so inspect
    // reports it as a NOTE, still exits 0, and (the decisive part)
    // leaves the torn bytes on disk untouched. This is the read-only
    // guarantee that separates `inspect` from a boot-time replay.
    let dir = std::env::temp_dir().join(format!("taguru-cli-torn-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
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
    let dir = std::env::temp_dir().join(format!("taguru-cli-import-marker-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
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
    let dir = std::env::temp_dir().join(format!("taguru-cli-inspect-empty-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
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
    let dir =
        std::env::temp_dir().join(format!("taguru-cli-inspect-badstem-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
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
    let dir =
        std::env::temp_dir().join(format!("taguru-cli-inspect-badmeta-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
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
    let dir =
        std::env::temp_dir().join(format!("taguru-cli-inspect-badfile-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
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
    let dir = std::env::temp_dir().join(format!("taguru-cli-export-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
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
    let dir =
        std::env::temp_dir().join(format!("taguru-cli-inspect-groups-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
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
    let dir = std::env::temp_dir().join(format!("taguru-cli-compact-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
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
