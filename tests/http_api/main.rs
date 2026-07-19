//! HTTP integration tests: the real binary, spawned on a free port with
//! a scratch data directory, driven through the same retrieval loop the
//! protocol documents. Everything here was once verified by hand with
//! curl; this pins it so handler wiring and response shapes cannot
//! regress silently.
//!
//! Split by concern across this directory's modules; `support` is the
//! shared server harness, everything else is one test cluster.

mod support;

mod auth;
mod calibrate;
mod directory;
mod directory_labels_compact;
mod errors;
mod explore_audit;
mod extract;
mod groups;
mod groups_cross_mcp;
mod key_scopes_cross_context;
mod mcp_basics;
mod metrics;
mod observability;
mod offline_import;
mod passages;
mod replication;
mod resolve_match;
mod retrieval_cache;
mod retrieval_core;
mod routing;
mod search_log;
mod search_plan;
mod semantic_cache;
mod server;
mod server_ops;
mod width_probe;
