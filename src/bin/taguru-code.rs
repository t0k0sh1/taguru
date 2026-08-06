//! taguru-code: offline codebase ingestion and lookup for coding
//! agents — `sync` turns a git repository's committed state into
//! location/structure facts in `$PROJECT_ROOT/.taguru`, `find`/`tree`
//! answer "where is X" from that directory in-process. No server, no
//! LLM, no configuration: the substance lives in the shared [`code`]
//! module; this file is the process entry plus the dual-inclusion of
//! the server modules the offline apply path reuses (`registry` boot,
//! `ingest`'s batch parse/apply) — the same `#[path]` shape
//! `taguru-mcp` uses, scaled to the import web's closure.

// The included server modules are used only through the narrow
// surface `code::sync`/`code::query` call; everything else they
// export is dead weight in THIS binary by design (spike posture —
// trimming the inclusion set is follow-up work if the prototype
// graduates).
#![allow(dead_code)]
// Same spike posture for the hub re-exports (`api.rs`'s handler
// surface exists for main.rs's router, which this binary never
// builds).
#![allow(unused_imports)]

#[path = "../api.rs"]
mod api;
#[path = "../auth.rs"]
mod auth;
#[path = "../benchmark.rs"]
mod benchmark;
#[path = "../bm25.rs"]
mod bm25;
#[path = "../breaker.rs"]
mod breaker;
#[path = "../calibrate.rs"]
mod calibrate;
#[path = "../cli.rs"]
mod cli;
#[path = "../clock.rs"]
mod clock;
#[path = "../communities.rs"]
mod communities;
#[path = "../compact.rs"]
mod compact;
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
#[path = "../estimate.rs"]
mod estimate;
#[path = "../evalset.rs"]
mod evalset;
#[path = "../evaluate.rs"]
mod evaluate;
#[path = "../export.rs"]
mod export;
#[path = "../extract.rs"]
mod extract;
#[path = "../groups.rs"]
mod groups;
#[path = "../hash.rs"]
mod hash;
#[path = "../hydrate.rs"]
mod hydrate;
#[path = "../ingest.rs"]
mod ingest;
#[path = "../inspect.rs"]
mod inspect;
#[path = "../limits.rs"]
mod limits;
#[path = "../mcp.rs"]
mod mcp;
#[path = "../measure.rs"]
mod measure;
#[path = "../metrics.rs"]
mod metrics;
#[path = "../oauth.rs"]
mod oauth;
#[path = "../oauth_http.rs"]
mod oauth_http;
#[path = "../paragraph.rs"]
mod paragraph;
#[path = "../passages.rs"]
mod passages;
#[path = "../registry.rs"]
mod registry;
#[path = "../remote.rs"]
mod remote;
#[path = "../remote_mcp.rs"]
mod remote_mcp;
#[path = "../replica.rs"]
mod replica;
#[path = "../schema.rs"]
mod schema;
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
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(code::run(&args));
}
