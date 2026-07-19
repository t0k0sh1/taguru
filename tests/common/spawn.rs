//! Shared harness for spawning the real `taguru` binary hermetically
//! (http_api/support.rs, cli.rs): scrubbing a developer shell's
//! TAGURU_*/OTEL_* vars before a spawn, and reading a child's stdout for
//! its `"listening on <addr>"` line.

use std::io::{BufRead, BufReader, Lines};
use std::process::{ChildStdout, Command};

/// Every TAGURU_*/OTEL_* var a hermetic spawn should scrub so a
/// developer's live shell (a configured embed provider, a live OTel
/// collector, stray auth tokens or a config file) never leaks into a
/// test run. A spawn that needs one of these set does so with
/// `Command::env` afterward — it always wins over an earlier
/// `env_remove`.
pub fn scrub_taguru_env(command: &mut Command) -> &mut Command {
    for key in [
        "TAGURU_ADDR",
        "TAGURU_DATA_DIR",
        "TAGURU_EMBED_URL",
        "TAGURU_EMBED_MODEL",
        "TAGURU_EMBED_AUTO",
        "TAGURU_EMBED_PASSAGES",
        "TAGURU_PASSAGE_VECTOR_LIMIT",
        "TAGURU_PASSAGES_WAL_MAX_BYTES",
        "TAGURU_SEMANTIC_FLOOR",
        "TAGURU_API_TOKEN",
        "TAGURU_API_TOKENS",
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
