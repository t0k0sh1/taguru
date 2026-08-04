//! Shared harness for spawning the real `taguru` binary hermetically
//! (http_api/support.rs, cli.rs): scrubbing a developer shell's
//! TAGURU_*/OTEL_*/RUST_LOG vars before a spawn, and reading a child's
//! stdout for its `"listening on <addr>"` line.

use std::io::{BufRead, BufReader, Lines};
use std::process::{ChildStdout, Command};

/// Every TAGURU_*/OTEL_*/RUST_LOG var a hermetic spawn should scrub so
/// a developer's live shell (a configured embed provider, a live OTel
/// collector, stray auth tokens or a config file, a `warn`-level log
/// filter that would silence the info-level audit/access/search
/// events log-assertion tests parse) never leaks into a test run. A
/// spawn that needs one of these set does so with `Command::env`
/// afterward — it always wins over an earlier `env_remove`.
pub fn scrub_taguru_env(command: &mut Command) -> &mut Command {
    for key in [
        "RUST_LOG",
        "TAGURU_ADDR",
        "TAGURU_DATA_DIR",
        "TAGURU_EMBED_URL",
        "TAGURU_EMBED_MODEL",
        "TAGURU_EMBED_AUTO",
        "TAGURU_EMBED_PASSAGES",
        "TAGURU_PASSAGE_VECTOR_LIMIT",
        "TAGURU_PASSAGES_WAL_MAX_BYTES",
        "TAGURU_SEMANTIC_FLOOR",
        "TAGURU_RERANK_URL",
        "TAGURU_RERANK_MODEL",
        "TAGURU_RERANK_API_KEY",
        "TAGURU_RERANK_TIMEOUT_SECS",
        "TAGURU_API_TOKEN",
        "TAGURU_API_TOKENS",
        "TAGURU_KEY_SCOPES",
        "TAGURU_EXTRACT_URL",
        "TAGURU_EXTRACT_MODEL",
        "TAGURU_EXTRACT_API_KEY",
        "TAGURU_EXTRACT_TIMEOUT_SECS",
        "TAGURU_EXTRACT_PARALLEL",
        "TAGURU_RATE_LIMIT_PER_MIN",
        "TAGURU_PUBLIC_URL",
        "TAGURU_LOG_SEARCHES",
        "TAGURU_ROUTE_MAP",
        "TAGURU_CONFIG",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
    ] {
        command.env_remove(key);
    }
    command
}

/// A fresh scratch directory under the OS temp dir, named
/// `taguru-<prefix>-<pid>`: any previous run's leftovers under this
/// exact name are removed first. Does not create the directory —
/// callers that need it to exist call `std::fs::create_dir_all`
/// themselves (some want the bare directory, some a subdirectory,
/// some let the spawned server create it on first write).
pub fn scratch_dir(prefix: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("taguru-{prefix}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Reads `stdout` until a `"listening on <addr>"` line appears,
/// returning the address plus the still-open line iterator so the
/// caller can decide whether to keep draining it. `label` names the
/// process in the panic message if it exits before listening.
pub fn read_listen_line(
    label: &str,
    stdout: ChildStdout,
) -> (String, Lines<BufReader<ChildStdout>>) {
    let mut lines = BufReader::new(stdout).lines();
    let addr = loop {
        let line = lines
            .next()
            .unwrap_or_else(|| panic!("{label} exited before listening"))
            .expect("server stdout must be readable");
        if let Some(addr) = line.strip_prefix("listening on ") {
            break addr.to_string();
        }
    };
    (addr, lines)
}
