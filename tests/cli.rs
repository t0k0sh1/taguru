//! CLI integration tests: the real binary, real arguments, real exit
//! codes. The serve default is pinned by every test in http_api.rs;
//! this file covers everything that must NOT start a server, plus the
//! configuration file.

use std::io::{BufRead, BufReader, Read};
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};

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
    command
        .args(["--config", &config.display().to_string()])
        .env_remove("TAGURU_ADDR")
        .env_remove("TAGURU_DATA_DIR")
        .env_remove("TAGURU_EMBED_URL")
        .env_remove("TAGURU_API_TOKEN")
        .env_remove("TAGURU_CONFIG")
        .env_remove("TAGURU_LOG_SEARCHES")
        .env_remove("OTEL_EXPORTER_OTLP_ENDPOINT")
        .env_remove("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        .env_remove("OTEL_EXPORTER_OTLP_PROTOCOL");
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("server must spawn");

    let stdout = child.stdout.take().expect("stdout must be piped");
    let mut lines = BufReader::new(stdout).lines();
    loop {
        let line = lines
            .next()
            .expect("server must reach its listen line")
            .expect("stdout must be readable");
        if line.starts_with("listening on ") {
            break;
        }
    }
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
    let mut child = Command::new(env!("CARGO_BIN_EXE_taguru"))
        .env_remove("TAGURU_CONFIG")
        .env_remove("TAGURU_API_TOKEN")
        .env_remove("TAGURU_EMBED_URL")
        .env("TAGURU_ADDR", "127.0.0.1:0")
        .env("TAGURU_DATA_DIR", dir.join("data"))
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("server must spawn");
    let stdout = child.stdout.take().expect("stdout must be piped");
    let mut lines = BufReader::new(stdout).lines();
    let addr = loop {
        let line = lines
            .next()
            .expect("server must reach its listen line")
            .expect("stdout must be readable");
        if let Some(addr) = line.strip_prefix("listening on ") {
            break addr.to_string();
        }
    };
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
    command
        .env_remove("TAGURU_ADDR")
        .env_remove("TAGURU_DATA_DIR")
        .env_remove("TAGURU_EMBED_URL")
        .env_remove("TAGURU_API_TOKEN")
        .env("TAGURU_CONFIG", &config);
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("server must spawn");
    let stdout = child.stdout.take().expect("stdout must be piped");
    let mut lines = BufReader::new(stdout).lines();
    loop {
        let line = lines
            .next()
            .expect("server must reach its listen line")
            .expect("stdout must be readable");
        if line.starts_with("listening on ") {
            break;
        }
    }
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

    let output = run(&["inspect", &image.display().to_string()]);
    assert_eq!(output.status.code(), Some(0));
    assert!(String::from_utf8_lossy(&output.stdout).contains("ok"));

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
        assert!(
            String::from_utf8_lossy(&output.stderr).is_empty(),
            "{flag}"
        );
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
    let dir =
        std::env::temp_dir().join(format!("taguru-cli-inspect-empty-{}", std::process::id()));
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
