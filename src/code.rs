//! `taguru-code`: offline codebase ingestion and lookup. `sync` walks
//! a git repository (committed state only — `git ls-files` is the
//! .gitignore authority), parses source files into location/structure
//! facts deterministically (tree-sitter, no LLM anywhere), and applies
//! them to a data directory at `$PROJECT_ROOT/.taguru` through the
//! same batch/import contract every other write uses. `find`/`tree`
//! answer "where is X" in-process from that directory — no server, no
//! network, nothing to configure.
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

const USAGE: &str = "\
taguru-code: offline codebase map for coding agents

USAGE:
  taguru-code sync [PATH] [--dry-run]   ingest the repo at PATH (default .)
  taguru-code find <cue> [--json]       locate a symbol: kind, name, file:line
  taguru-code tree [PATH]               list what a directory/file/symbol contains
  taguru-code status                    show sync state
  taguru-code evalset --out FILE        generate an eval set from the synced repo
  taguru-code eval --eval FILE          measure find accuracy (exit 3 on regression)
";

/// Dispatches one invocation; returns the process exit code.
pub(crate) fn run(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("help") | Some("--help") | Some("-h") => {
            print!("{USAGE}");
            0
        }
        Some("sync") => sync::run(&args[1..]),
        Some("status") => sync::status(&args[1..]),
        Some("find") => query::find(&args[1..]),
        Some("tree") => query::tree(&args[1..]),
        Some("evalset") => eval::evalset(&args[1..]),
        Some("eval") => eval::eval(&args[1..]),
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
