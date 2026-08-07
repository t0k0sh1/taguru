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
  taguru-code find <cue> [--json]       locate a symbol: kind, name, file:line
  taguru-code tree [PATH]               list what a directory/file/symbol contains
  taguru-code status                    show sync state
  taguru-code evalset --out FILE        generate an eval set from the synced repo
  taguru-code eval --eval FILE          measure find accuracy (exit 3 on regression)

ENVIRONMENT:
  TAGURU_USAGE_LOG            0/false/off disables the per-invocation usage log
  TAGURU_USAGE_LOG_DIR        where usage records go (default $HOME/.taguru/logs)
  TAGURU_USAGE_LOG_MAX_BYTES  total cap across usage-*.jsonl, oldest days deleted
                              first (default 52428800 = 50 MiB, 0 = uncapped)
";

/// Dispatches one invocation; returns the process exit code.
pub(crate) fn run(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("help") | Some("--help") | Some("-h") => {
            print!("{USAGE}");
            0
        }
        Some("sync") => usage_log::record("sync", &args[1..], || sync::run(&args[1..])),
        Some("watch") => usage_log::record("watch", &args[1..], || watch::run(&args[1..])),
        Some("status") => usage_log::record("status", &args[1..], || sync::status(&args[1..])),
        Some("find") => usage_log::record("find", &args[1..], || query::find(&args[1..])),
        Some("tree") => usage_log::record("tree", &args[1..], || query::tree(&args[1..])),
        Some("evalset") => usage_log::record("evalset", &args[1..], || eval::evalset(&args[1..])),
        Some("eval") => usage_log::record("eval", &args[1..], || eval::eval(&args[1..])),
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
