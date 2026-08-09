//! `taguru router` (`route` is a deprecated alias, issue #248 item 9):
//! a stateless scatter-gather router over sharded
//! instances (issue #130) — the write-scaling leg beside the replica
//! pool's read scaling. One URL serves the whole HTTP surface over a
//! static context→shard map; groups and multi-context search span
//! shards with the exact single-instance merge semantics.
//!
//! **What the router is.** A mode of the same binary with no data
//! directory, no lock, and no durable state of any kind — run as many
//! as the load balancer wants. Config is one file (`TAGURU_ROUTE_MAP`):
//! `context = shard-url` lines plus an optional `* = shard-url`
//! fallback for contexts the map does not name. Editing the map takes
//! a router restart — restarts of a stateless process behind an LB
//! are a rolling non-event. Moving a context between shards, in
//! order: quiesce its writes, `taguru export` it, DELETE it through
//! the router (the old shard drops it — and sweeps it from its group
//! projections), edit the map and roll the routers, then re-import
//! through the router (which now routes it to the new shard). The
//! delete must precede the re-import (a copy left on the old shard
//! keeps answering that shard's slice of every group fan-out —
//! duplicate hits, not just a stale listing), and the restart must
//! finish first too: a router still holding the old map would route
//! the re-import back to the old shard.
//!
//! **Routing.** Context-scoped verbs proxy verbatim (streamed both
//! ways) to the owning shard, so their responses — including error
//! shapes, 404s, and exports — are the shard's own bytes. `/import`
//! splits the batch stream by each batch's `context` header, validates
//! the WHOLE stream first with the same parser the shard runs, and
//! dry-run-preflights every chunk so a stream the single instance
//! would refuse whole is refused whole here too, with nothing applied.
//!
//! **Scatter-gather.** `POST /recall`, `/query`, and `/sources/search`
//! fan out to the shards owning the named contexts (all shards when
//! groups are named) and merge exactly as one instance merges its own
//! contexts: the graph verbs by [`crate::api::cross_rank`] (one weight
//! scale, context/subject/label/object tiebreak) with `total` summed;
//! the passage verb by per-context rank interleaving. Cursors need no
//! composition at all: the `after` cursor is anchored on the last
//! match itself, not on any per-instance position, so the router
//! forwards it verbatim and every shard resumes past the same point.
//!
//! **Groups.** Every group exists on every shard; each shard's copy
//! holds the member contexts the map assigns to that shard, while
//! child-group edges are broadcast whole — identical nesting structure
//! everywhere, so cycle and depth verdicts cannot differ, and a
//! group's transitive closure on one shard is exactly the global
//! closure's slice for that shard. Group writes rewrite the member
//! lists per shard and broadcast sequentially; reads union the
//! projections. A search naming a group therefore just names it to
//! every shard — no expansion round trip.
//!
//! **Partial failure.** A shard that ANSWERS an error fails the whole
//! request, exactly as one failing context fails a single instance's
//! cross-context search. A shard that cannot be REACHED (connect,
//! timeout, mid-body) degrades the fan-out verbs to labeled partial
//! results: the envelope gains `"unreached": [{shard, contexts,
//! error}]`, omitted entirely when every shard answered — the same
//! field reaches MCP tool results, whose text is this JSON. Group reads
//! and writes never serve partials (a partially-unioned group would
//! look complete); they answer 502 `shard_unreachable` instead.
//!
//! **Auth is pass-through.** The router forwards `Authorization`
//! verbatim and holds no key store — shards keep enforcing keys,
//! scopes, and rate limits, so keyrings must agree across shards.
//! Setting TAGURU_API_TOKEN(S)/TAGURU_KEY_SCOPES on the router is a
//! boot refusal, not a silent no-op: an operator who set them expected
//! enforcement that would not happen. OAuth (TAGURU_PUBLIC_URL) is
//! refused the same way: consent and registration are durable state a
//! stateless fleet cannot hold. `POST /mcp` itself works fully — the
//! same in-process dispatch the server uses, over these proxy routes,
//! with the caller's bearer re-attached to every dispatched call.
//!
//! **Known divergences from one instance, on purpose:**
//! - A scoped-key group write refused by shard k leaves shards <k
//!   applied (deltas converge on retry); a single instance applies
//!   nothing. Import does NOT share this gap — its preflight catches
//!   refusals before anything lands.
//! - Multi-shard import refusals that survive preflight (mid-apply IO)
//!   number batches within the failing chunk, not the whole stream.
//! - A scoped key naming an UNMAPPED context in a cross-search gets
//!   `no_context` from the map's own truth; a single instance checks
//!   the scope first. The router cannot evaluate scopes (it holds no
//!   keyring), and what the earlier 404 reveals is deployment
//!   topology, not data.
//! - `/metrics` is router-shaped (`taguru_router_*`), not server-shaped.
//! - Renaming a context through the router works but leaves the map
//!   pointing at the old name until the operator edits it.
//! - `/contexts/{name}/promote` (ADR 0018) proxies whole to the shard
//!   owning the scratch `{name}`, so a destination mapped to another
//!   shard refuses there (`no_context`) — promotion through the
//!   router requires the pair on one shard. Cross-shard moves stay
//!   what they were: export, delete, remap, re-import.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{MatchedPath, Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use taguru::deadline::Deadline;
use tokio::net::TcpListener;
use tower_http::catch_panic::CatchPanicLayer;
use tracing::{Instrument as _, info, warn};

use crate::api::{self, ErrorCode};
use crate::env::{
    DEFAULT_MAX_BODY_BYTES, DEFAULT_MCP_MAX_RESULT_BYTES, env_number, resolve_body_bytes,
    resolve_mcp_max_result_bytes, resolve_timeout_secs,
};

#[path = "route/config.rs"]
mod config;
#[path = "route/cross.rs"]
mod cross;
#[path = "route/endpoints.rs"]
mod endpoints;
#[path = "route/groups.rs"]
mod groups;
#[path = "route/import.rs"]
mod import;
#[path = "route/maintenance.rs"]
mod maintenance;
#[path = "route/proxy.rs"]
mod proxy;
#[path = "route/scatter.rs"]
mod scatter;
#[path = "route/server.rs"]
mod server;
#[path = "route/state.rs"]
mod state;

use config::RouteMap;
use cross::{cross_query, cross_recall, cross_search_passages, merge_contexts};
use endpoints::{health, proxy_protocol, render_metrics, urlencode};
use groups::{
    create_group_broadcast, delete_group_broadcast, export_group_union, full_path, merge_groups,
    rename_group_broadcast, union_group, update_group_broadcast,
};
use import::route_import;
use maintenance::{broadcast_flush, broadcast_maintenance};
use proxy::{proxy_context_root, proxy_context_sub};
#[cfg(test)]
use scatter::abort_rank;
use scatter::{gather, plan_scatter, shard_body};
use server::budget;
pub(crate) use server::run;
use state::{RouterInner, RouterMetrics, RouterState};

/// One reached shard's answer, body buffered — the fan-out verbs all
/// carry small JSON bodies. The streaming path ([`proxy_to_shard`])
/// never builds one of these.
struct ShardAnswer {
    status: StatusCode,
    body: Bytes,
}

/// The envelope every shard success wraps its result in; the router
/// re-wraps merged results in its own (see [`RouterResponse`]).
#[derive(Deserialize)]
struct ShardEnvelope<T> {
    result: T,
}

/// The single-instance `{result, status, time}` envelope plus the one
/// router-only field: which shards could not be reached. Serializes
/// byte-identically to the single instance whenever `unreached` is
/// empty — the field vanishes entirely.
#[derive(Serialize)]
struct RouterResponse<T: Serialize> {
    result: T,
    status: &'static str,
    time: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    unreached: Vec<Unreached>,
}

/// One unreachable shard in a fan-out: its URL, the directly-named
/// contexts that routed to it (group members it may also have held are
/// not enumerable while it is down), and the transport error.
#[derive(Serialize, Clone)]
struct Unreached {
    shard: String,
    contexts: Vec<String>,
    error: String,
}

fn router_ok<T: Serialize>(result: T, unreached: Vec<Unreached>, started_at: Instant) -> Response {
    (
        StatusCode::OK,
        axum::Json(RouterResponse {
            result,
            status: "ok",
            time: started_at.elapsed().as_secs_f64(),
            // Field order matches the shard envelope's `result, status,
            // time` prefix; `unreached` rides last, when present at all.
            unreached,
        }),
    )
        .into_response()
}

/// 502 for a shard the router could not reach on a path that cannot
/// serve partial results.
fn unreachable_refusal(unreached: &[Unreached], started_at: Instant) -> Response {
    let names: Vec<&str> = unreached.iter().map(|entry| entry.shard.as_str()).collect();
    let first = unreached
        .first()
        .map(|entry| entry.error.as_str())
        .unwrap_or("unreachable");
    api::error(
        ErrorCode::ShardUnreachable,
        format!(
            "shard {} is unreachable ({first}); this request needs every shard it names — \
             retry when the shard (or its LB) answers again",
            names.join(", ")
        ),
        started_at,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_route_map_parses_contexts_a_fallback_and_comments() {
        let map = RouteMap::parse(
            "# fleet\nsake = http://a:8248/\nbreweries = http://a:8248\n\nglossary = http://b:8248\n* = http://a:8248\n",
        )
        .expect("a well-formed map parses");
        assert_eq!(map.shards, vec!["http://a:8248", "http://b:8248"]);
        assert_eq!(map.shard_of("sake"), Some(0));
        assert_eq!(map.shard_of("glossary"), Some(1));
        // Unmapped falls to '*'.
        assert_eq!(map.shard_of("brand-new"), Some(0));
    }

    #[test]
    fn the_route_map_refuses_duplicates_and_malformed_lines() {
        let duplicated = RouteMap::parse("sake = http://a:1\nsake = http://b:1\n")
            .expect_err("a context mapped twice is a config bug");
        assert!(duplicated.contains("line 2"), "{duplicated}");
        let starless = RouteMap::parse("* = http://a:1\n* = http://b:1\n")
            .expect_err("two fallbacks contradict each other");
        assert!(starless.contains("line 2"), "{starless}");
        let bare = RouteMap::parse("sake http://a:1\n").expect_err("no '=' is not a mapping");
        assert!(bare.contains("line 1"), "{bare}");
        let scheme = RouteMap::parse("sake = ftp://a:1\n").expect_err("shards speak http(s) only");
        assert!(scheme.contains("http(s)"), "{scheme}");
        let empty =
            RouteMap::parse("# nothing\n").expect_err("a map with no shards routes nothing");
        assert!(empty.contains("no shards"), "{empty}");
        // Without a fallback, unmapped contexts have no shard at all.
        let map = RouteMap::parse("sake = http://a:1\n").unwrap();
        assert_eq!(map.shard_of("unmapped"), None);
    }

    #[test]
    fn the_projection_splits_members_by_owner_and_keeps_children_whole() {
        let map = RouteMap::parse("a = http://a:1\nb = http://b:1\n").unwrap();
        assert_eq!(map.project(["a", "b"], 0), vec!["a".to_string()]);
        assert_eq!(map.project(["a", "b"], 1), vec!["b".to_string()]);
        // A member no shard owns projects nowhere — the owning-shard
        // refusal downstream is what reports it.
        assert!(map.project(["stray"], 0).is_empty());
    }

    #[test]
    fn abort_precedence_matches_the_single_instance_check_order() {
        assert!(abort_rank(Some("forbidden")) < abort_rank(Some("no_context")));
        assert!(abort_rank(Some("no_context")) < abort_rank(Some("no_group")));
        assert!(abort_rank(Some("no_group")) < abort_rank(Some("timeout")));
        assert_eq!(abort_rank(None), 3);
    }

    #[test]
    fn urlencode_round_trips_a_multibyte_context_name() {
        assert_eq!(urlencode("sake"), "sake");
        assert_eq!(urlencode("日本酒"), "%E6%97%A5%E6%9C%AC%E9%85%92");
        assert_eq!(urlencode("a b/c"), "a%20b%2Fc");
    }
}
