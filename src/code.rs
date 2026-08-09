//! `taguru-code`: offline codebase ingestion and lookup. `sync` walks
//! a git repository's worktree — exactly ripgrep's universe: tracked
//! plus untracked files, .gitignore excluded, bytes as they are on
//! disk (staged and unstaged edits included), with HEAD serving only
//! as the incremental anchor — parses source files into
//! location/structure facts deterministically (tree-sitter, no LLM
//! anywhere), and applies them to a data directory at
//! `$PROJECT_ROOT/.taguru` through the same batch/import contract
//! every other write uses. `find`/`tree` answer "where is X"
//! in-process from that directory — no server, no network, nothing
//! to configure.
//!
//! This is a spike (plan: prototype → evaluate against graphify →
//! decide); module split follows the plan: `grammar` is the
//! language-agnostic contract, `grammars/` its per-language
//! implementations, `facts` the pure fact builder, with `repo_walk`,
//! `sync`, `query`, `eval` following.

// Explicit `#[path]` on every child, same reason as `mcp.rs`: this
// file is itself loaded via `#[path = "../code.rs"]` from the binary,
// and a `#[path]`-loaded file's own unpathed child mods would resolve
// beside the loader instead of under `src/code/`.
#[path = "code/eval.rs"]
pub(crate) mod eval;
#[path = "code/facts.rs"]
pub(crate) mod facts;
#[path = "code/grammar.rs"]
pub(crate) mod grammar;
#[path = "code/grammars.rs"]
pub(crate) mod grammars;
#[path = "code/query.rs"]
pub(crate) mod query;
#[path = "code/repo_walk.rs"]
pub(crate) mod repo_walk;
#[path = "code/sync.rs"]
pub(crate) mod sync;
#[path = "code/usage_log.rs"]
pub(crate) mod usage_log;
#[path = "code/watch.rs"]
pub(crate) mod watch;

const USAGE: &str = "\
taguru-code: offline codebase map for coding agents

USAGE:
  taguru-code sync [PATH] [--dry-run]   ingest the repo at PATH (default .)
  taguru-code watch [PATH] [--interval-ms N]  keep syncing as the tree changes
  taguru-code find <cue> [--json] [--limit N]  locate a symbol: kind, name, file:line (default 10 rows)
  taguru-code tree [PATH]               list what a directory/file/symbol contains
  taguru-code status                    show sync state
  taguru-code evalset --out FILE [--sample N]  generate an eval set (default 200 cases)
  taguru-code eval --eval FILE [--thresholds FILE]  measure find accuracy (exit 3 on regression)
  taguru-code models [--json]           list local embedding models, best pick first

  Every verb except models also takes --context NAME (default 'code')
  and --data-dir PATH (default: the repo root's .taguru).

ENVIRONMENT:
  TAGURU_USAGE_LOG            0/false/off disables the per-invocation usage log
  TAGURU_USAGE_LOG_DIR        where usage records go (default $HOME/.taguru/logs)
  TAGURU_USAGE_LOG_MAX_BYTES  total cap across usage-*.jsonl, oldest days deleted
                              first (default 52428800 = 50 MiB, 0 = uncapped)
  TAGURU_EMBED_URL            'local' embeds in-process during sync (default
                              builds carry it; slim --no-default-features
                              builds warn and keep the lane off); any other
                              value is an OpenAI-compatible /embeddings endpoint
  TAGURU_EMBED_MODEL          which model — see `taguru-code models`
";

/// Dispatches one invocation; returns the process exit code.
pub(crate) fn run(args: &[String]) -> i32 {
    // `help`, `--help`, or `-h` — bare or after a verb — answers the
    // usage and exits 0, checked once before dispatch so every verb
    // behaves identically and no per-verb parser can forget it
    // (`--help` used to fall into each parser as an unknown flag).
    // Scanning the whole argv is safe: no flag takes a value that
    // could legitimately be "-h" (cues and paths starting with '-'
    // are refused as unknown flags), so the conventional reading
    // always wins.
    if args.first().map(String::as_str) == Some("help")
        || args.iter().any(|arg| arg == "--help" || arg == "-h")
    {
        print!("{USAGE}");
        return 0;
    }
    match args.first().map(String::as_str) {
        Some("sync") => usage_log::record("sync", &args[1..], || sync::run(&args[1..])),
        Some("watch") => usage_log::record("watch", &args[1..], || watch::run(&args[1..])),
        Some("status") => usage_log::record("status", &args[1..], || sync::status(&args[1..])),
        Some("find") => usage_log::record("find", &args[1..], || query::find(&args[1..])),
        Some("tree") => usage_log::record("tree", &args[1..], || query::tree(&args[1..])),
        Some("evalset") => usage_log::record("evalset", &args[1..], || eval::evalset(&args[1..])),
        Some("eval") => usage_log::record("eval", &args[1..], || eval::eval(&args[1..])),
        Some("models") => usage_log::record("models", &args[1..], || models(&args[1..])),
        Some(other) => {
            eprintln!("taguru-code: unknown argument '{other}' — try 'taguru-code --help'");
            2
        }
        None => {
            print!("{USAGE}");
            2
        }
    }
}

/// The `models` verb: what `TAGURU_EMBED_URL=local` can run, in
/// recommendation order (the table's own order). Static on purpose —
/// the listing must work, and teach, even on a default build that
/// cannot run any of them; whether THIS binary can is one line at the
/// end, not a reason to hide the catalog.
fn models(args: &[String]) -> i32 {
    let json = match args.first().map(String::as_str) {
        None => false,
        Some("--json") if args.len() == 1 => true,
        Some(other) => {
            eprintln!("taguru-code: models: unknown argument '{other}' — only --json is accepted");
            return 2;
        }
    };
    let runnable = cfg!(feature = "local-embed");
    if json {
        let listed: Vec<serde_json::Value> = crate::embedding::LOCAL_MODELS
            .iter()
            .map(|info| {
                serde_json::json!({
                    "name": info.name,
                    "download_mib": info.download_mib,
                    "dims": info.dims,
                    "multilingual": info.multilingual,
                    "license": info.license,
                    "note": info.note,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "models": listed,
                "runnable": runnable,
                "cache_dir": crate::embedding::model_cache_dir()
                    .map(|dir| dir.display().to_string()),
            })
        );
        return 0;
    }
    println!("local embedding models, best pick first:");
    println!("(TAGURU_EMBED_URL=local TAGURU_EMBED_MODEL=<name> taguru-code sync)\n");
    for info in &crate::embedding::LOCAL_MODELS {
        println!(
            "  {:<41} {:>4} MiB  {:>3}d  {:<12} {:<11} {}",
            info.name,
            info.download_mib,
            info.dims,
            if info.multilingual {
                "multilingual"
            } else {
                "English"
            },
            info.license,
            info.note,
        );
    }
    // The effective cache, not a hardcoded promise: fastembed lets
    // HF_HOME override the directory this binary asks for.
    let cache = crate::embedding::model_cache_dir()
        .map(|dir| dir.display().to_string())
        .unwrap_or_else(|| "(no home directory found)".to_string());
    println!("\ndownloaded once into {cache}; offline afterwards");
    if !runnable {
        println!(
            "note: this slim build lacks the local-embed feature — a default `cargo build` \
             carries it"
        );
    }
    0
}

#[cfg(test)]
mod tests {
    use super::USAGE;

    /// The usage-log knobs live outside `config::KNOWN_KEYS` (that
    /// list feeds the server's typo detection and help-text test), so
    /// this USAGE text is their only documentation — keep it honest.
    #[test]
    fn usage_documents_every_usage_log_knob() {
        for knob in [
            "TAGURU_USAGE_LOG",
            "TAGURU_USAGE_LOG_DIR",
            "TAGURU_USAGE_LOG_MAX_BYTES",
        ] {
            assert!(USAGE.contains(knob), "USAGE must document {knob}");
        }
    }
}
