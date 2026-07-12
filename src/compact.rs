//! `taguru compact`: rewrite context images without their dead
//! weight. The image is append-only by design — retraction unlinks
//! attribution records but never reclaims them, alias removal leaves
//! arena bytes behind — so a long-lived context with heavy revision
//! traffic grows monotonically. Compaction rebuilds each image from
//! its live content alone ([`taguru::context::Context::compacted`])
//! and persists the result; a running server serves the same at
//! `POST /contexts/{name}/compact`. Passages need none of this: their
//! store compacts itself ratio-triggered.

use std::path::PathBuf;

use taguru::deadline::Deadline;

use crate::registry::AccessError;

const USAGE: &str = "\
usage: taguru compact [--config FILE] [CONTEXT...]

Rewrites context images in TAGURU_DATA_DIR without the dead weight the
append-only format accumulates (retracted edges, unlinked attribution
records, arena slack) — offline; the directory lock refuses to run
beside a live server. No CONTEXT arguments means every context. Live
systems use POST /contexts/{name}/compact instead (admin role; the
context's own requests wait out the rebuild). Content is preserved:
counts and paragraph locators exactly, per-source weights within
float re-accumulation error; aliases whose canonical no longer
carries any live association are dropped and counted.

  --config F   read KEY=VALUE environment from F (same dialect as serve)
";

pub(crate) fn run(args: &[String]) -> i32 {
    let mut config: Option<PathBuf> = None;
    let mut names: Vec<String> = Vec::new();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{USAGE}");
                return 0;
            }
            "--config" => match rest.next() {
                Some(path) => config = Some(PathBuf::from(path)),
                None => return usage_error("--config needs a file path"),
            },
            other if other.starts_with('-') => {
                return usage_error(&format!("unknown flag '{other}'"));
            }
            name => names.push(name.to_string()),
        }
    }
    // SAFETY (same contract as serve/import/export): applied while the
    // process is still single-threaded — no runtime ever starts here.
    if let Some(path) = &config {
        crate::cli::load_config(path);
    }

    crate::ingest::init_logging();
    let state = match crate::registry::BootConfig::from_env().boot(None) {
        Ok(state) => state,
        Err(error) => {
            eprintln!("taguru: compact: {error}");
            return 1;
        }
    };

    let names = if names.is_empty() {
        let all: Vec<String> = state
            .directory()
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        if all.is_empty() {
            eprintln!("taguru: compact: the data directory holds no contexts");
            return 1;
        }
        all
    } else {
        names
    };

    let mut failures = 0usize;
    for name in &names {
        match state.compact_context(name, Deadline::unbounded()) {
            Ok(outcome) => println!(
                "context '{name}': {} → {} ({} dead edge(s) shed{})",
                crate::cli::fmt_bytes(outcome.bytes_before as u64),
                crate::cli::fmt_bytes(outcome.bytes_after as u64),
                outcome.dead_edges,
                match outcome.aliases_dropped {
                    0 => String::new(),
                    dropped =>
                        format!(", {dropped} alias(es) dropped (canonical has no live association)"),
                },
            ),
            Err(failure) => {
                let message = match failure {
                    AccessError::NotFound => "no such context".to_string(),
                    AccessError::Load(error) => error,
                    AccessError::Unpersisted(error) => error,
                    // The CLI runs with Deadline::unbounded(), which
                    // never expires — unreachable in practice, kept for
                    // exhaustiveness.
                    AccessError::DeadlineExceeded => "deadline exceeded".to_string(),
                };
                eprintln!("taguru: compact: context '{name}': {message}");
                failures += 1;
            }
        }
    }
    println!(
        "compact: {} of {} context(s) rewritten",
        names.len() - failures,
        names.len()
    );
    if failures > 0 { 1 } else { 0 }
}

fn usage_error(message: &str) -> i32 {
    eprintln!("taguru: compact: {message} — try 'taguru compact --help'");
    2
}
