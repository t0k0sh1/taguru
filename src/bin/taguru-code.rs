//! taguru-code: offline codebase ingestion and lookup for coding
//! agents — `sync` turns a git repository's worktree (ripgrep's
//! universe: tracked + untracked, .gitignore excluded, disk bytes;
//! HEAD is only the incremental anchor) into location/structure
//! facts in `$PROJECT_ROOT/.taguru`, `find`/`tree` answer "where is
//! X" from that directory in-process. No server, no LLM, no
//! configuration: the substance lives in the shared [`code`] module;
//! this file is the process entry plus the dual-inclusion of the
//! server modules the offline apply path reuses (`registry` boot,
//! `ingest`'s batch parse/apply) — the same `#[path]` shape
//! `taguru-mcp` uses, scaled to the import web's closure.

// The inclusion set below is the import web's actual closure (#443
// item 3): reference-error-minimal, so a module here is one some
// retained module names, not a leftover. What keeps these two allows:
// the retained modules are used only through the narrow surface
// `code::sync`/`code::query` call, so most of what they export —
// `registry`'s full pub surface, `api.rs`'s handler shapes — is dead
// weight in THIS binary, by the nature of dual inclusion rather than
// by oversight.
#![allow(dead_code)]
#![allow(unused_imports)]

#[path = "../api.rs"]
mod api;
#[path = "../auth.rs"]
mod auth;
#[path = "../bm25.rs"]
mod bm25;
#[path = "../breaker.rs"]
mod breaker;
#[path = "../clock.rs"]
mod clock;
#[path = "../config.rs"]
mod config;
#[path = "../context_proptest.rs"]
#[cfg(test)]
pub(crate) mod context_proptest;
#[path = "../crc32c.rs"]
mod crc32c;
#[path = "../embedding.rs"]
mod embedding;
#[path = "../env.rs"]
mod env;
#[path = "../export.rs"]
mod export;
#[path = "../groups.rs"]
mod groups;
#[path = "../hash.rs"]
mod hash;
#[path = "../hydrate.rs"]
mod hydrate;
#[path = "../ingest.rs"]
mod ingest;
#[path = "../limits.rs"]
mod limits;
// Narrow on purpose: `api.rs`'s /version facts quote the MCP protocol
// constants, so `crate::mcp::SUPPORTED_PROTOCOL_VERSIONS` must resolve
// — but including the full `mcp.rs` would drag the whole tool-routing
// surface into a binary that never speaks MCP. In THIS crate,
// `crate::mcp` is just the protocol constants and the tool schemas
// they lean on (the outer `#[path]` re-bases the inner declarations
// onto the real `src/mcp/`).
#[path = "../mcp"]
mod mcp {
    pub(crate) mod protocol;
    pub(crate) mod schema;
    pub(crate) use protocol::SUPPORTED_PROTOCOL_VERSIONS;
}
#[path = "../metrics.rs"]
mod metrics;
#[path = "../oauth.rs"]
mod oauth;
#[path = "../paragraph.rs"]
mod paragraph;
#[path = "../passages.rs"]
mod passages;
#[path = "../registry.rs"]
mod registry;
#[path = "../remote.rs"]
mod remote;
#[path = "../replica.rs"]
mod replica;
#[path = "../schema.rs"]
mod schema;
#[path = "../sensitive.rs"]
mod sensitive;
#[path = "../sha256.rs"]
mod sha256;
#[path = "../ship.rs"]
mod ship;
#[path = "../storage.rs"]
mod storage;
#[path = "../trace.rs"]
mod trace;
#[path = "../wal.rs"]
mod wal;

#[path = "../code.rs"]
mod code;

fn main() {
    // Rust's startup ignores SIGPIPE, so `taguru-code status | head -3`
    // ends in a println! panic when the downstream closes the pipe.
    // This binary's stdout exists to be piped (agents post-process
    // find/tree/status), so restore the default disposition: die
    // quietly on a closed pipe, like grep does (issue #453).
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(code::run(&args));
}
