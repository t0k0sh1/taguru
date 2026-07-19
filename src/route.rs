//! `taguru route`: a stateless scatter-gather router over sharded
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
use tracing::{info, warn};

use crate::api::{self, ErrorCode};
use crate::env::{
    DEFAULT_MAX_BODY_BYTES, DEFAULT_MCP_MAX_RESULT_BYTES, env_number, resolve_body_bytes,
    resolve_mcp_max_result_bytes, resolve_timeout_secs,
};

/// The context→shard map, parsed from `TAGURU_ROUTE_MAP`. Shards are
/// deduped by URL in file order; `fallback` is the `*` entry.
#[derive(Debug)]
pub(crate) struct RouteMap {
    shards: Vec<String>,
    contexts: BTreeMap<String, usize>,
    fallback: Option<usize>,
}

impl RouteMap {
    /// One `context = url` per line (`*` for the fallback), `#`
    /// comments and blank lines ignored — the same boring dialect as
    /// the config file. Refused whole on the first malformed line:
    /// a route map with a silent hole misroutes forever.
    pub(crate) fn parse(text: &str) -> Result<Self, String> {
        let mut shards: Vec<String> = Vec::new();
        let mut contexts = BTreeMap::new();
        let mut fallback = None;
        let shard_index = |url: &str, shards: &mut Vec<String>| -> usize {
            let url = url.trim_end_matches('/').to_string();
            match shards.iter().position(|known| *known == url) {
                Some(index) => index,
                None => {
                    shards.push(url);
                    shards.len() - 1
                }
            }
        };
        for (number, line) in text.lines().enumerate() {
            let number = number + 1;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((name, url)) = line.split_once('=') else {
                return Err(format!(
                    "line {number}: expected 'context = shard-url' (or '* = shard-url')"
                ));
            };
            let (name, url) = (name.trim(), url.trim());
            if name.is_empty() {
                return Err(format!("line {number}: the context name is empty"));
            }
            if !url.starts_with("http://") && !url.starts_with("https://") {
                return Err(format!(
                    "line {number}: '{url}' is not an http(s) shard URL"
                ));
            }
            if name == "*" {
                if fallback.is_some() {
                    return Err(format!("line {number}: '*' fallback given twice"));
                }
                fallback = Some(shard_index(url, &mut shards));
            } else if contexts
                .insert(name.to_string(), shard_index(url, &mut shards))
                .is_some()
            {
                return Err(format!("line {number}: context '{name}' is mapped twice"));
            }
        }
        if shards.is_empty() {
            return Err("the route map names no shards".to_string());
        }
        Ok(Self {
            shards,
            contexts,
            fallback,
        })
    }

    fn shard_of(&self, context: &str) -> Option<usize> {
        self.contexts.get(context).copied().or(self.fallback)
    }

    fn all(&self) -> impl Iterator<Item = usize> {
        0..self.shards.len()
    }

    fn url(&self, shard: usize) -> &str {
        &self.shards[shard]
    }

    /// The map's member-list projection for one shard — what a group
    /// write sends there. A member no shard owns keeps flowing to the
    /// owning-shard check downstream, which refuses it exactly as a
    /// single instance refuses a nonexistent member.
    fn project<'a>(&self, members: impl IntoIterator<Item = &'a str>, shard: usize) -> Vec<String> {
        members
            .into_iter()
            .filter(|name| self.shard_of(name) == Some(shard))
            .map(str::to_string)
            .collect()
    }
}

/// Router-mode counters, rendered at `GET /metrics` as
/// `taguru_router_*` — deliberately router-shaped, not server-shaped:
/// a stateless proxy has no flusher, no WAL, no cache to report on.
#[derive(Default)]
struct RouterMetrics {
    http: Mutex<BTreeMap<(String, u16), u64>>,
    shard: Mutex<BTreeMap<(usize, &'static str), u64>>,
}

impl RouterMetrics {
    fn record_http(&self, route: &str, status: u16) {
        *self
            .http
            .lock()
            .entry((route.to_string(), status))
            .or_insert(0) += 1;
    }

    fn record_shard(&self, shard: usize, outcome: &'static str) {
        *self.shard.lock().entry((shard, outcome)).or_insert(0) += 1;
    }
}

struct RouterInner {
    map: RouteMap,
    client: reqwest::Client,
    metrics: RouterMetrics,
    /// The MCP `initialize` manual, fetched once from the first shard
    /// that answers `GET /protocol` — a cache of immutable-per-deploy
    /// text, not state. Until a shard answers, initialize falls back
    /// to the local manual without the shard's configuration trailer.
    instructions: OnceLock<Arc<String>>,
}

#[derive(Clone)]
pub(crate) struct RouterState {
    inner: Arc<RouterInner>,
}

impl RouterState {
    fn map(&self) -> &RouteMap {
        &self.inner.map
    }
}

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

// ---------------------------------------------------------------------------
// Boot

#[tokio::main]
pub(crate) async fn run(config: Option<PathBuf>) {
    let tracer_provider = crate::init_telemetry();
    let _ = &config; // config was loaded into the environment by main()

    // Misconfiguration refuses to boot, replica-mode style: a router
    // that silently ignored a keyring would advertise enforcement that
    // never happens.
    for var in ["TAGURU_API_TOKEN", "TAGURU_API_TOKENS", "TAGURU_KEY_SCOPES"] {
        if std::env::var(var).is_ok_and(|value| !value.trim().is_empty()) {
            tracing::error!(
                "{var} is set, but the router holds no key store — it forwards \
                 Authorization verbatim and the shards enforce it. Configure keys on \
                 every shard and unset {var} here"
            );
            std::process::exit(1);
        }
    }
    if std::env::var("TAGURU_PUBLIC_URL").is_ok_and(|value| !value.trim().is_empty()) {
        tracing::error!(
            "TAGURU_PUBLIC_URL enables OAuth, and OAuth registrations/consents are durable \
             state a stateless router fleet cannot hold — terminate OAuth on a single \
             instance, or use bearer keys through the router"
        );
        std::process::exit(1);
    }
    // TAGURU_DATA_DIR stays un-warned on purpose: the container image
    // bakes it in, so the warning would fire on every correct
    // containerized router and teach operators to ignore warnings.
    for var in ["TAGURU_REPLICATE_URL", "TAGURU_REPLICA"] {
        if std::env::var(var).is_ok() {
            warn!("{var} is set but means nothing to a router — ignoring it");
        }
    }

    let map_path = match std::env::var("TAGURU_ROUTE_MAP") {
        Ok(path) if !path.trim().is_empty() => PathBuf::from(path),
        _ => {
            tracing::error!(
                "route mode needs TAGURU_ROUTE_MAP: a file of 'context = shard-url' lines \
                 (plus an optional '* = shard-url' fallback) that says where contexts live"
            );
            std::process::exit(1);
        }
    };
    let map = match std::fs::read_to_string(&map_path)
        .map_err(|error| error.to_string())
        .and_then(|text| RouteMap::parse(&text))
    {
        Ok(map) => map,
        Err(error) => {
            tracing::error!(path = %map_path.display(), %error, "TAGURU_ROUTE_MAP is not usable");
            std::process::exit(1);
        }
    };
    info!(
        shards = map.shards.len(),
        contexts = map.contexts.len(),
        fallback = map.fallback.map(|index| map.shards[index].clone()),
        "route map loaded"
    );

    let timeout_secs = resolve_timeout_secs(env_number("TAGURU_REQUEST_TIMEOUT_SECS", 30));
    let max_body_bytes =
        resolve_body_bytes(env_number("TAGURU_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES));
    let mcp_max_result_bytes = resolve_mcp_max_result_bytes(env_number(
        "TAGURU_MCP_MAX_RESULT_BYTES",
        DEFAULT_MCP_MAX_RESULT_BYTES,
    ));

    // Redirects must pass THROUGH a proxy, not be chased by it; the
    // shards' own request timeout still applies per call via the
    // deadline, so no global client timeout here.
    let client = match reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            tracing::error!(%error, "could not build the outbound HTTP client");
            std::process::exit(1);
        }
    };

    let state = RouterState {
        inner: Arc::new(RouterInner {
            map,
            client,
            metrics: RouterMetrics::default(),
            instructions: OnceLock::new(),
        }),
    };

    let app = routes(state.clone()).with_state(state.clone());
    // The same in-process dispatch trick serve() uses for POST /mcp:
    // a handle to the routes with no outer layers, so a dispatched
    // tool call is not re-charged — but here it must also re-carry the
    // caller's bearer, since the dispatched request is synthetic and
    // the shards (not this process) are what enforce it.
    let mcp_dispatch = app
        .clone()
        .layer(axum::extract::DefaultBodyLimit::disable());
    let mcp_state = state.clone();
    let app = app.route(
        "/mcp",
        post(
            move |axum::Extension(deadline): axum::Extension<Deadline>,
                  headers: HeaderMap,
                  body: Bytes| {
                let state = mcp_state.clone();
                let dispatch = mcp_dispatch
                    .clone()
                    .layer(axum::middleware::from_fn_with_state(
                        headers.get(header::AUTHORIZATION).cloned(),
                        reattach_authorization,
                    ));
                async move {
                    // Only `initialize` reads the manual — a tool call
                    // must never spend its budget probing shards for
                    // text it will not use (with every shard down, the
                    // probes would eat the whole deadline first).
                    let instructions = if wants_instructions(&body) {
                        state.mcp_instructions(deadline).await
                    } else {
                        Arc::new(String::new())
                    };
                    crate::remote_mcp::serve(
                        dispatch,
                        instructions,
                        None,
                        body,
                        mcp_max_result_bytes,
                        deadline,
                    )
                    .await
                }
            },
        ),
    );
    let app = app
        .layer({
            let state = state.clone();
            CatchPanicLayer::custom(move |payload| router_panic_response(payload, &state))
        })
        .layer(axum::extract::DefaultBodyLimit::max(max_body_bytes))
        .layer(axum::middleware::from_fn_with_state(
            Duration::from_secs(timeout_secs as u64),
            crate::limits::enforce_timeout,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            track_router_http,
        ));

    let addr = std::env::var("TAGURU_ADDR").unwrap_or_else(|_| "127.0.0.1:8248".to_string());
    let listener = match TcpListener::bind(&addr).await {
        Ok(listener) => listener,
        Err(bind_error) => {
            tracing::error!(
                %addr,
                error = %bind_error,
                "cannot bind — is the port already in use, or the address not local? \
                 Set TAGURU_ADDR to change where the router listens"
            );
            std::process::exit(1);
        }
    };
    // The same stdout contract line serve() prints; spawners read it.
    println!("listening on {}", listener.local_addr().unwrap());
    info!(
        addr = %listener.local_addr().unwrap(),
        shards = state.map().shards.len(),
        mapped_contexts = state.map().contexts.len(),
        timeout_secs,
        max_body_mib = max_body_bytes / (1024 * 1024),
        "router ready — stateless; auth, scopes, and rate limits are enforced by the shards",
    );
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(crate::shutdown_signal())
    .await
    .unwrap();

    if let Some(provider) = tracer_provider
        && let Err(error) = tokio::task::block_in_place(|| provider.shutdown())
    {
        warn!(error = %error, "trace export flush on shutdown failed");
    }
}

/// Every route the router answers. The two `/contexts/{name}` entries
/// proxy ANY method so a shard's own 405/404 shapes pass through
/// untouched — and so context verbs added to the server later route
/// without touching this table.
fn routes(state: RouterState) -> Router<RouterState> {
    Router::new()
        .route("/health", get(health))
        .route("/live", get(crate::metrics::live))
        .route("/metrics", get(render_metrics))
        .route("/protocol", get(proxy_protocol))
        .route("/flush", post(broadcast_flush))
        .route("/maintenance/compact", post(broadcast_maintenance))
        .route("/import", post(route_import))
        .route("/contexts", get(merge_contexts))
        .route("/contexts/{name}", any(proxy_context_root))
        .route("/contexts/{name}/{*rest}", any(proxy_context_sub))
        .route("/groups", get(merge_groups))
        .route(
            "/groups/{name}",
            get(union_group)
                .put(create_group_broadcast)
                .patch(update_group_broadcast)
                .delete(delete_group_broadcast),
        )
        .route("/groups/{name}/export", get(export_group_union))
        .route("/groups/{name}/rename", post(rename_group_broadcast))
        .route("/recall", post(cross_recall))
        .route("/query", post(cross_query))
        .route("/sources/search", post(cross_search_passages))
        .fallback(api::unknown_path)
        .method_not_allowed_fallback(api::method_not_allowed)
        .layer(CatchPanicLayer::custom({
            let state = state.clone();
            move |payload| router_panic_response(payload, &state)
        }))
}

/// Whether an MCP message is an `initialize` — the one method whose
/// reply carries the manual. A cheap peek, not a validation: anything
/// unparseable goes to `remote_mcp::serve` for its own refusal.
fn wants_instructions(body: &Bytes) -> bool {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|message| {
            message
                .get("method")
                .and_then(Value::as_str)
                .map(|method| method == "initialize")
        })
        .unwrap_or(false)
}

/// The dispatched-call twin of the server's auth story: the outer
/// bearer, re-attached header-for-header to every in-process /mcp
/// dispatch, because the SHARDS are what verify it here.
async fn reattach_authorization(
    State(auth): State<Option<HeaderValue>>,
    mut request: Request,
    next: Next,
) -> Response {
    if let Some(value) = auth.clone() {
        request.headers_mut().insert(header::AUTHORIZATION, value);
    }
    next.run(request).await
}

/// `api::panic_response` without the registry: same JSON 500, panic
/// counted in the router's own metrics.
fn router_panic_response(payload: Box<dyn std::any::Any + Send>, state: &RouterState) -> Response {
    let message = if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else {
        "handler panicked with a non-string payload".to_string()
    };
    tracing::error!(%message, "router handler panicked");
    state.inner.metrics.record_http("<panic>", 500);
    api::error(
        ErrorCode::Internal,
        "internal error: the handler panicked (this is a bug worth reporting)",
        Instant::now(),
    )
}

/// The router's access log + RED counters — the thin twin of
/// `metrics::track_http`, without the registry the full version needs.
async fn track_router_http(
    State(state): State<RouterState>,
    matched: Option<MatchedPath>,
    request: Request,
    next: Next,
) -> Response {
    let method = request.method().clone();
    let route = matched
        .as_ref()
        .map(|matched| matched.as_str().to_string())
        .unwrap_or_else(|| "<unmatched>".to_string());
    let started = Instant::now();
    let response = next.run(request).await;
    let status = response.status().as_u16();
    state.inner.metrics.record_http(&route, status);
    info!(
        method = %method,
        route = %route,
        status,
        latency_ms = started.elapsed().as_secs_f64() * 1000.0,
        "http",
    );
    response
}

// ---------------------------------------------------------------------------
// Outbound

/// Strips what must not cross a proxy hop: the RFC 9110 hop-by-hop
/// set, `Host` (reqwest recomputes it for the shard), and the length/
/// framing headers the outbound client re-derives from the body it
/// actually sends.
fn hop_headers(headers: &HeaderMap) -> HeaderMap {
    let mut forwarded = headers.clone();
    for name in [
        header::CONNECTION,
        header::PROXY_AUTHENTICATE,
        header::PROXY_AUTHORIZATION,
        header::TE,
        header::TRAILER,
        header::TRANSFER_ENCODING,
        header::UPGRADE,
        header::HOST,
        header::CONTENT_LENGTH,
    ] {
        forwarded.remove(name);
    }
    forwarded.remove("keep-alive");
    forwarded
}

/// The per-call slice of the request budget. `None` means unbounded —
/// only the CLI paths produce that; served requests always carry the
/// timeout middleware's deadline.
fn budget(deadline: Deadline) -> Option<Duration> {
    let remaining = deadline.remaining();
    (remaining != Duration::MAX).then_some(remaining)
}

impl RouterState {
    /// One buffered round trip to a shard — the fan-out building
    /// block. `Err` is TRANSPORT failure only (connect, timeout, torn
    /// body); an HTTP error status is an answer, not an `Err`.
    async fn call_shard(
        &self,
        shard: usize,
        method: Method,
        path_and_query: &str,
        headers: &HeaderMap,
        body: Option<Bytes>,
        deadline: Deadline,
    ) -> Result<ShardAnswer, String> {
        let url = format!("{}{}", self.map().url(shard), path_and_query);
        let mut request = self
            .inner
            .client
            .request(method, url)
            .headers(forward_headers(headers));
        if let Some(limit) = budget(deadline) {
            request = request.timeout(limit);
        }
        if let Some(body) = body {
            request = request
                .header(header::CONTENT_TYPE, "application/json")
                .body(body);
        }
        let outcome = async {
            let response = request.send().await?;
            let status = response.status();
            let body = response.bytes().await?;
            Ok::<ShardAnswer, reqwest::Error>(ShardAnswer { status, body })
        }
        .await;
        match outcome {
            Ok(answer) => {
                self.inner.metrics.record_shard(
                    shard,
                    if answer.status.is_success() {
                        "ok"
                    } else {
                        "http_error"
                    },
                );
                Ok(answer)
            }
            Err(error) => {
                self.inner.metrics.record_shard(shard, "unreached");
                Err(error.to_string())
            }
        }
    }

    /// [`Self::call_shard`] across a shard set, concurrently; answers
    /// come back labeled by shard index.
    async fn fan_out<F>(
        &self,
        shards: &[usize],
        method: Method,
        path_and_query: &str,
        headers: &HeaderMap,
        body_for: F,
        deadline: Deadline,
    ) -> Vec<(usize, Result<ShardAnswer, String>)>
    where
        F: Fn(usize) -> Option<Bytes>,
    {
        let calls = shards.iter().map(|&shard| {
            let method = method.clone();
            let body = body_for(shard);
            async move {
                (
                    shard,
                    self.call_shard(shard, method, path_and_query, headers, body, deadline)
                        .await,
                )
            }
        });
        futures_util::future::join_all(calls).await
    }

    /// The MCP manual: the first shard's `GET /protocol`, cached for
    /// the process lifetime once one answers; the local text (no
    /// configuration trailer) until then.
    async fn mcp_instructions(&self, deadline: Deadline) -> Arc<String> {
        if let Some(cached) = self.inner.instructions.get() {
            return Arc::clone(cached);
        }
        for shard in self.map().all() {
            let fetch = self
                .call_shard(
                    shard,
                    Method::GET,
                    "/protocol",
                    &HeaderMap::new(),
                    None,
                    deadline,
                )
                .await;
            if let Ok(answer) = fetch
                && answer.status.is_success()
                && let Ok(text) = std::str::from_utf8(&answer.body)
            {
                let manual = Arc::new(text.to_string());
                let _ = self.inner.instructions.set(Arc::clone(&manual));
                return Arc::clone(self.inner.instructions.get().unwrap_or(&manual));
            }
        }
        Arc::new(api::protocol_text(None))
    }
}

/// Request headers a NON-proxy (fan-out) call forwards: the caller's
/// identity and trace context; everything else is the router's own
/// call.
fn forward_headers(headers: &HeaderMap) -> HeaderMap {
    let mut forwarded = HeaderMap::new();
    for name in [
        header::AUTHORIZATION,
        header::HeaderName::from_static("traceparent"),
        header::HeaderName::from_static("x-amzn-trace-id"),
    ] {
        if let Some(value) = headers.get(&name) {
            forwarded.insert(name, value.clone());
        }
    }
    forwarded
}

// ---------------------------------------------------------------------------
// The streaming proxy: context-scoped verbs

async fn proxy_context_root(
    State(state): State<RouterState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    request: Request,
) -> Response {
    proxy_context(state, name, deadline, request).await
}

async fn proxy_context_sub(
    State(state): State<RouterState>,
    axum::extract::Path((name, _rest)): axum::extract::Path<(String, String)>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    request: Request,
) -> Response {
    proxy_context(state, name, deadline, request).await
}

/// The transparent hop: same method, same path and query, headers
/// minus the hop-by-hop set, body streamed out and the shard's answer
/// streamed back — the response is the shard's own bytes, error
/// shapes included.
async fn proxy_context(
    state: RouterState,
    name: String,
    deadline: Deadline,
    request: Request,
) -> Response {
    let started_at = Instant::now();
    let Some(shard) = state.map().shard_of(&name) else {
        // No entry and no fallback: for a read this context cannot
        // exist anywhere the router routes — the single-instance
        // not-found, byte for byte. A PUT is asking to CREATE it, and
        // the honest answer is that the map decides where new contexts
        // go, not that something wasn't found.
        return if request.method() == Method::PUT {
            api::error(
                ErrorCode::InvalidArgument,
                format!(
                    "no shard owns context '{name}': add a route-map entry for it, or a \
                     '*' fallback for unmapped contexts (TAGURU_ROUTE_MAP)"
                ),
                started_at,
            )
        } else {
            api::error(
                ErrorCode::NoContext,
                format!("context '{name}' not found"),
                started_at,
            )
        };
    };
    let (parts, body) = request.into_parts();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|paq| paq.as_str().to_string())
        .unwrap_or_else(|| parts.uri.path().to_string());
    let url = format!("{}{}", state.map().url(shard), path_and_query);
    let mut outbound = state
        .inner
        .client
        .request(parts.method.clone(), url)
        .headers(hop_headers(&parts.headers))
        .body(reqwest::Body::wrap_stream(body.into_data_stream()));
    if let Some(limit) = budget(deadline) {
        outbound = outbound.timeout(limit);
    }
    match outbound.send().await {
        Ok(answer) => {
            state.inner.metrics.record_shard(
                shard,
                if answer.status().is_success() {
                    "ok"
                } else {
                    "http_error"
                },
            );
            let status = answer.status();
            let headers = hop_headers(answer.headers());
            let mut response = Response::builder().status(status);
            if let Some(response_headers) = response.headers_mut() {
                *response_headers = headers;
            }
            response
                .body(Body::from_stream(answer.bytes_stream()))
                .unwrap_or_else(|error| {
                    api::error(
                        ErrorCode::Internal,
                        format!("could not assemble the proxied response: {error}"),
                        started_at,
                    )
                })
        }
        Err(error) => {
            state.inner.metrics.record_shard(shard, "unreached");
            api::error(
                ErrorCode::ShardUnreachable,
                format!(
                    "shard {} (owning context '{name}') is unreachable: {error}",
                    state.map().url(shard)
                ),
                started_at,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Scatter-gather: /recall, /query, /sources/search

/// The shared front half of every fan-out search: the single-instance
/// pre-checks (byte for byte), the direct-name dedup, and the
/// shard-set/per-shard-body computation. `direct` preserves first-
/// appearance order — the same order `cross_targets` seats direct
/// names in.
struct Scatter {
    direct: Vec<String>,
    /// direct contexts per shard, order preserved within each shard.
    per_shard: BTreeMap<usize, Vec<String>>,
    shards: Vec<usize>,
}

fn plan_scatter(
    state: &RouterState,
    contexts: &[String],
    groups: &[String],
    started_at: Instant,
) -> Result<Scatter, Box<Response>> {
    if contexts.is_empty() && groups.is_empty() {
        return Err(Box::new(api::error(
            ErrorCode::InvalidArgument,
            "'contexts' or 'groups' must name at least one target",
            started_at,
        )));
    }
    for (field, count) in [("contexts", contexts.len()), ("groups", groups.len())] {
        if let Some(refusal) = api::overlong(field, count, started_at) {
            return Err(Box::new(refusal));
        }
    }
    let mut seen = BTreeSet::new();
    let direct: Vec<String> = contexts
        .iter()
        .filter(|name| seen.insert((*name).clone()))
        .cloned()
        .collect();
    let mut per_shard: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for name in &direct {
        let Some(shard) = state.map().shard_of(name) else {
            // Unmapped with no fallback cannot exist anywhere the
            // router reaches — the same first-missing-name refusal a
            // single instance gives, in the same list order.
            return Err(Box::new(api::error(
                ErrorCode::NoContext,
                format!("context '{name}' not found"),
                started_at,
            )));
        };
        per_shard.entry(shard).or_default().push(name.clone());
    }
    let shards: Vec<usize> = if groups.is_empty() {
        per_shard.keys().copied().collect()
    } else {
        // Groups live on every shard (the projected-broadcast
        // invariant), so naming one fans out everywhere.
        state.map().all().collect()
    };
    Ok(Scatter {
        direct,
        per_shard,
        shards,
    })
}

/// Sorts multi-shard failures into the single-instance refusal order:
/// scope refusals over direct names come before existence, existence
/// before group resolution — tie-broken by where each shard's first
/// direct target sits in the request's own order.
fn abort_rank(code: Option<&str>) -> u8 {
    match code {
        Some("forbidden") => 0,
        Some("no_context") => 1,
        Some("no_group") => 2,
        _ => 3,
    }
}

/// The fan-out outcome, split three ways: HTTP-answered failures abort
/// the whole request (a shard that answered an error is a context that
/// failed, and one failing context fails a single instance's search
/// whole); transport failures become the labeled `unreached` partials;
/// the rest merge.
struct Gathered {
    answers: Vec<(usize, Bytes)>,
    unreached: Vec<Unreached>,
}

fn gather(
    state: &RouterState,
    scatter: &Scatter,
    outcomes: Vec<(usize, Result<ShardAnswer, String>)>,
) -> Result<Gathered, Box<Response>> {
    let mut answers = Vec::new();
    let mut unreached = Vec::new();
    let mut aborts: Vec<(u8, usize, usize, ShardAnswer)> = Vec::new();
    for (shard, outcome) in outcomes {
        match outcome {
            Ok(answer) if answer.status.is_success() => answers.push((shard, answer.body)),
            Ok(answer) => {
                let code = serde_json::from_slice::<Value>(&answer.body)
                    .ok()
                    .and_then(|body| body.get("code").and_then(Value::as_str).map(str::to_string));
                let first_direct = scatter
                    .per_shard
                    .get(&shard)
                    .and_then(|targets| targets.first())
                    .and_then(|name| scatter.direct.iter().position(|direct| direct == name))
                    .unwrap_or(usize::MAX);
                aborts.push((abort_rank(code.as_deref()), first_direct, shard, answer));
            }
            Err(error) => unreached.push(Unreached {
                shard: state.map().url(shard).to_string(),
                contexts: scatter.per_shard.get(&shard).cloned().unwrap_or_default(),
                error,
            }),
        }
    }
    if let Some((_, _, _, answer)) = aborts
        .into_iter()
        .min_by_key(|(rank, position, shard, _)| (*rank, *position, *shard))
    {
        // The shard's own bytes pass through — same code, same
        // message, same status a single instance would have answered.
        return Err(Box::new(
            (
                answer.status,
                [(header::CONTENT_TYPE, "application/json")],
                answer.body,
            )
                .into_response(),
        ));
    }
    if answers.is_empty() && !unreached.is_empty() {
        return Err(Box::new(unreachable_refusal(&unreached, Instant::now())));
    }
    Ok(Gathered { answers, unreached })
}

/// Builds each shard's request body: the caller's own body with the
/// `contexts` list cut down to what that shard owns. Everything else —
/// groups, cue, limit, the verbatim `after` cursor — is forwarded
/// untouched.
fn shard_body(base: &Value, targets: Option<&Vec<String>>) -> Bytes {
    let mut body = base.clone();
    body["contexts"] = json!(targets.cloned().unwrap_or_default());
    Bytes::from(body.to_string())
}

async fn cross_recall(
    State(state): State<RouterState>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    api::AppJson(request): api::AppJson<api::CrossRecallRequest>,
) -> Response {
    let limit = request.limit;
    let contexts = request.contexts.clone();
    let groups = request.groups.clone();
    let base = match serde_json::to_value(&request) {
        Ok(base) => base,
        Err(error) => {
            return api::error(
                ErrorCode::Internal,
                format!("could not re-serialize the request: {error}"),
                Instant::now(),
            );
        }
    };
    merge_matches(
        state, headers, deadline, "/recall", contexts, groups, limit, base,
    )
    .await
}

async fn cross_query(
    State(state): State<RouterState>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    api::AppJson(request): api::AppJson<api::CrossQueryRequest>,
) -> Response {
    let limit = request.limit;
    let contexts = request.contexts.clone();
    let groups = request.groups.clone();
    let base = match serde_json::to_value(&request) {
        Ok(base) => base,
        Err(error) => {
            return api::error(
                ErrorCode::Internal,
                format!("could not re-serialize the request: {error}"),
                Instant::now(),
            );
        }
    };
    merge_matches(
        state, headers, deadline, "/query", contexts, groups, limit, base,
    )
    .await
}

/// The graph-verb merge: concatenate every shard's already-cursored
/// top page, rank with the exact single-instance comparator
/// ([`api::cross_rank`]), cut at the same clamp, sum the totals. Each
/// shard's page is its own top-`limit` past the cursor under the same
/// total order, so the union's top-`limit` is the global page.
#[allow(clippy::too_many_arguments)]
async fn merge_matches(
    state: RouterState,
    headers: HeaderMap,
    deadline: Deadline,
    path: &str,
    contexts: Vec<String>,
    groups: Vec<String>,
    limit: Option<usize>,
    base: Value,
) -> Response {
    let started_at = Instant::now();
    let scatter = match plan_scatter(&state, &contexts, &groups, started_at) {
        Ok(scatter) => scatter,
        Err(refusal) => return *refusal,
    };
    let outcomes = state
        .fan_out(
            &scatter.shards,
            Method::POST,
            path,
            &headers,
            |shard| Some(shard_body(&base, scatter.per_shard.get(&shard))),
            deadline,
        )
        .await;
    let gathered = match gather(&state, &scatter, outcomes) {
        Ok(gathered) => gathered,
        Err(refusal) => return *refusal,
    };
    let mut total = 0usize;
    let mut matches: Vec<api::CrossMatch<api::AssociationOut>> = Vec::new();
    let mut searched: BTreeSet<String> = BTreeSet::new();
    for (shard, body) in gathered.answers {
        match serde_json::from_slice::<ShardEnvelope<api::CrossMatchPage>>(&body) {
            Ok(page) => {
                total += page.result.total;
                matches.extend(page.result.matches);
                if let Some(plan) = page.result.plan {
                    searched.extend(plan.contexts);
                }
            }
            Err(error) => {
                return api::error(
                    ErrorCode::Internal,
                    format!(
                        "shard {} answered an unreadable page: {error}",
                        state.map().url(shard)
                    ),
                    started_at,
                );
            }
        }
    }
    matches.sort_by(|a, b| {
        api::cross_rank(
            (
                a.inner.weight,
                a.context.as_str(),
                a.inner.subject.as_str(),
                a.inner.label.as_str(),
                a.inner.object.as_str(),
            ),
            (
                b.inner.weight,
                b.context.as_str(),
                b.inner.subject.as_str(),
                b.inner.label.as_str(),
                b.inner.object.as_str(),
            ),
        )
    });
    matches.truncate(api::clamp(
        limit,
        api::DEFAULT_MATCH_LIMIT,
        api::MAX_MATCH_LIMIT,
    ));
    // The merged plan re-seats the union of the shard plans into the
    // single-instance effective order: direct names in request order,
    // group-resolved members after them in name order (the shards
    // resolved the groups — the router only reorders). A dead shard's
    // contexts are honestly absent: they were not searched, and the
    // `unreached` labels beside the plan say why.
    let mut contexts: Vec<String> = scatter
        .direct
        .iter()
        .filter(|name| searched.remove(name.as_str()))
        .cloned()
        .collect();
    contexts.extend(searched);
    router_ok(
        api::CrossMatchPage {
            total,
            matches,
            plan: Some(api::MatchPlan { contexts }),
        },
        gathered.unreached,
        started_at,
    )
}

async fn cross_search_passages(
    State(state): State<RouterState>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    api::AppJson(request): api::AppJson<api::CrossSearchPassagesRequest>,
) -> Response {
    let started_at = Instant::now();
    let base = match serde_json::to_value(&request) {
        Ok(base) => base,
        Err(error) => {
            return api::error(
                ErrorCode::Internal,
                format!("could not re-serialize the request: {error}"),
                started_at,
            );
        }
    };
    let scatter = match plan_scatter(&state, &request.contexts, &request.groups, started_at) {
        Ok(scatter) => scatter,
        Err(refusal) => return *refusal,
    };
    let outcomes = state
        .fan_out(
            &scatter.shards,
            Method::POST,
            "/sources/search",
            &headers,
            |shard| Some(shard_body(&base, scatter.per_shard.get(&shard))),
            deadline,
        )
        .await;
    let gathered = match gather(&state, &scatter, outcomes) {
        Ok(gathered) => gathered,
        Err(refusal) => return *refusal,
    };
    // Passage scores don't share a scale across contexts, so the
    // single-instance merge is rank interleaving: (per-context rank,
    // target-list position). Rank is recovered from each shard's page
    // — within it, one context's hits appear in rank order — and the
    // target-list position needs only the RELATIVE order of the
    // searched contexts: direct names first, in request order, then
    // group-resolved members in name order, which is exactly how
    // `cross_targets` seats them. The shard plans name every searched
    // context (hits alone would miss the empty-handed ones), so the
    // seat map and the merged plan both come from them.
    let mut pool: Vec<(usize, api::CrossMatch<api::PassageHit>)> = Vec::new();
    let mut plan_entries: Vec<api::SearchContextPlan> = Vec::new();
    for (shard, body) in gathered.answers {
        match serde_json::from_slice::<ShardEnvelope<api::CrossPassagePage>>(&body) {
            Ok(page) => {
                let mut rank_of: BTreeMap<String, usize> = BTreeMap::new();
                for hit in page.result.hits {
                    let rank = rank_of.entry(hit.context.clone()).or_insert(0);
                    let seat = *rank;
                    *rank += 1;
                    pool.push((seat, hit));
                }
                plan_entries.extend(page.result.plan.contexts);
            }
            Err(error) => {
                return api::error(
                    ErrorCode::Internal,
                    format!(
                        "shard {} answered an unreadable page: {error}",
                        state.map().url(shard)
                    ),
                    started_at,
                );
            }
        }
    }
    let mut seat: BTreeMap<String, usize> = scatter
        .direct
        .iter()
        .enumerate()
        .map(|(position, name)| (name.clone(), position))
        .collect();
    let resolved: BTreeSet<String> = plan_entries
        .iter()
        .map(|entry| entry.context.clone())
        .filter(|name| !seat.contains_key(name))
        .collect();
    for (position, name) in resolved.into_iter().enumerate() {
        seat.insert(name, scatter.direct.len() + position);
    }
    pool.sort_by_key(|(rank, hit)| (*rank, seat.get(&hit.context).copied().unwrap_or(usize::MAX)));
    pool.truncate(api::clamp(request.limit, 5, api::MAX_MATCH_LIMIT));
    plan_entries.sort_by_key(|entry| seat.get(&entry.context).copied().unwrap_or(usize::MAX));
    router_ok(
        api::CrossPassagePage {
            plan: api::SearchPlan {
                contexts: plan_entries,
            },
            hits: pool.into_iter().map(|(_, hit)| hit).collect(),
        },
        gathered.unreached,
        started_at,
    )
}

// ---------------------------------------------------------------------------
// Directory merges: GET /contexts, GET /groups

async fn merge_contexts(
    State(state): State<RouterState>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    api::AppQuery(query): api::AppQuery<api::ListContextsQuery>,
    request: Request,
) -> Response {
    let started_at = Instant::now();
    let path = full_path(&request);
    let shards: Vec<usize> = state.map().all().collect();
    let outcomes = state
        .fan_out(&shards, Method::GET, &path, &headers, |_| None, deadline)
        .await;
    let mut unreached = Vec::new();
    let mut rows: BTreeMap<String, (usize, crate::registry::DirectoryEntry)> = BTreeMap::new();
    let mut total = 0usize;
    for (shard, outcome) in outcomes {
        match outcome {
            Ok(answer) if answer.status.is_success() => {
                match serde_json::from_slice::<ShardEnvelope<api::ContextPage>>(&answer.body) {
                    Ok(page) => {
                        total += page.result.total;
                        for entry in page.result.contexts {
                            match rows.get(&entry.name) {
                                // A context answered by two shards is a
                                // mid-move stray: the map's owner wins the
                                // row, and the duplicate leaves the total.
                                Some((held_shard, _))
                                    if state.map().shard_of(&entry.name) != Some(shard)
                                        || *held_shard == shard =>
                                {
                                    total = total.saturating_sub(1);
                                    warn!(
                                        context = %entry.name,
                                        "context answered by more than one shard — \
                                         mid-move stray? the route map's owner wins"
                                    );
                                }
                                Some(_) => {
                                    total = total.saturating_sub(1);
                                    warn!(
                                        context = %entry.name,
                                        "context answered by more than one shard — \
                                         mid-move stray? the route map's owner wins"
                                    );
                                    rows.insert(entry.name.clone(), (shard, entry));
                                }
                                None => {
                                    rows.insert(entry.name.clone(), (shard, entry));
                                }
                            }
                        }
                    }
                    Err(error) => {
                        return api::error(
                            ErrorCode::Internal,
                            format!(
                                "shard {} answered an unreadable page: {error}",
                                state.map().url(shard)
                            ),
                            started_at,
                        );
                    }
                }
            }
            Ok(answer) => {
                return (
                    answer.status,
                    [(header::CONTENT_TYPE, "application/json")],
                    answer.body,
                )
                    .into_response();
            }
            Err(error) => unreached.push(Unreached {
                shard: state.map().url(shard).to_string(),
                contexts: Vec::new(),
                error,
            }),
        }
    }
    if rows.is_empty() && !unreached.is_empty() && unreached.len() == shards.len() {
        return unreachable_refusal(&unreached, started_at);
    }
    let contexts: Vec<crate::registry::DirectoryEntry> = rows
        .into_values()
        .map(|(_, entry)| entry)
        .take(api::clamp(
            query.limit,
            api::MAX_MATCH_LIMIT,
            api::MAX_MATCH_LIMIT,
        ))
        .collect();
    router_ok(api::ContextPage { total, contexts }, unreached, started_at)
}

async fn merge_groups(
    State(state): State<RouterState>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    api::AppQuery(query): api::AppQuery<api::KeysetQuery>,
    request: Request,
) -> Response {
    let started_at = Instant::now();
    let path = full_path(&request);
    let shards: Vec<usize> = state.map().all().collect();
    let outcomes = state
        .fan_out(&shards, Method::GET, &path, &headers, |_| None, deadline)
        .await;
    let mut rows: BTreeMap<String, api::GroupEntry> = BTreeMap::new();
    let mut total = 0usize;
    for (shard, outcome) in outcomes {
        match outcome {
            Ok(answer) if answer.status.is_success() => {
                match serde_json::from_slice::<ShardEnvelope<api::GroupPage>>(&answer.body) {
                    Ok(page) => {
                        // Every shard holds every group, so any one
                        // shard's directory count is the true count;
                        // max rides out a half-created record.
                        total = total.max(page.result.total);
                        for entry in page.result.groups {
                            merge_group_entry(&mut rows, entry);
                        }
                    }
                    Err(error) => {
                        return api::error(
                            ErrorCode::Internal,
                            format!(
                                "shard {} answered an unreadable page: {error}",
                                state.map().url(shard)
                            ),
                            started_at,
                        );
                    }
                }
            }
            Ok(answer) => {
                return (
                    answer.status,
                    [(header::CONTENT_TYPE, "application/json")],
                    answer.body,
                )
                    .into_response();
            }
            Err(error) => {
                // A partially-unioned group row would look complete;
                // the group surfaces refuse rather than thin out.
                return unreachable_refusal(
                    &[Unreached {
                        shard: state.map().url(shard).to_string(),
                        contexts: Vec::new(),
                        error,
                    }],
                    started_at,
                );
            }
        }
    }
    let groups: Vec<api::GroupEntry> = rows
        .into_values()
        .take(api::clamp(
            query.limit,
            api::MAX_MATCH_LIMIT,
            api::MAX_MATCH_LIMIT,
        ))
        .collect();
    router_ok(api::GroupPage { total, groups }, Vec::new(), started_at)
}

/// Unions one shard's row into the merged directory: member contexts
/// are per-shard projections (disjoint by the map), children are
/// broadcast whole, the description is identical everywhere a
/// non-drifted record lives. Fingerprints are folded together — each
/// shard's token covers the members that shard holds, so the union's
/// token must move whenever ANY shard's does; the fold order is the
/// fan-out's shard order (stable across requests), so an unchanged
/// fleet keeps an unchanged token.
fn merge_group_entry(rows: &mut BTreeMap<String, api::GroupEntry>, entry: api::GroupEntry) {
    match rows.get_mut(&entry.name) {
        Some(held) => {
            let mut members: BTreeSet<String> = held.contexts.drain(..).collect();
            members.extend(entry.contexts);
            held.contexts = members.into_iter().collect();
            held.groups.extend(entry.groups);
            let mut digest = crate::hash::fnv1a_fold(
                crate::hash::FNV1A_OFFSET,
                held.fingerprint.bytes().chain([0xff]),
            );
            digest = crate::hash::fnv1a_fold(digest, entry.fingerprint.bytes());
            held.fingerprint = format!("{digest:016x}");
        }
        None => {
            rows.insert(entry.name.clone(), entry);
        }
    }
}

fn full_path(request: &Request) -> String {
    request
        .uri()
        .path_and_query()
        .map(|paq| paq.as_str().to_string())
        .unwrap_or_else(|| request.uri().path().to_string())
}

// ---------------------------------------------------------------------------
// Group verbs: projected broadcast

/// Runs one group write against every shard IN ORDER, the request's
/// member lists projected per shard by the map. The first refusal
/// stops the broadcast and passes through as-is — shards before it
/// have applied (deltas converge on retry; documented divergence).
async fn broadcast_group_write<F>(
    state: &RouterState,
    method: Method,
    path: &str,
    headers: &HeaderMap,
    body_for: F,
    deadline: Deadline,
    started_at: Instant,
) -> Result<Vec<Bytes>, Box<Response>>
where
    F: Fn(usize) -> Option<Bytes>,
{
    let mut answers = Vec::new();
    for shard in state.map().all() {
        match state
            .call_shard(
                shard,
                method.clone(),
                path,
                headers,
                body_for(shard),
                deadline,
            )
            .await
        {
            Ok(answer) if answer.status.is_success() => answers.push(answer.body),
            Ok(answer) => {
                return Err(Box::new(
                    (
                        answer.status,
                        [(header::CONTENT_TYPE, "application/json")],
                        answer.body,
                    )
                        .into_response(),
                ));
            }
            Err(error) => {
                return Err(Box::new(unreachable_refusal(
                    &[Unreached {
                        shard: state.map().url(shard).to_string(),
                        contexts: Vec::new(),
                        error,
                    }],
                    started_at,
                )));
            }
        }
    }
    Ok(answers)
}

/// Projects the named member-list fields of a JSON body per shard —
/// any member the map does not place is refused up front with the
/// single-instance nonexistent-member message, since a context no
/// shard owns cannot exist on any of them.
fn project_body(
    state: &RouterState,
    base: &Value,
    fields: &[&str],
    started_at: Instant,
) -> Result<impl Fn(usize) -> Option<Bytes> + use<>, Box<Response>> {
    let mut lists: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for field in fields {
        let members: Vec<String> = base
            .get(*field)
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        for member in &members {
            if state.map().shard_of(member).is_none() {
                return Err(Box::new(api::error(
                    ErrorCode::NoContext,
                    format!("context '{member}' not found; nothing was applied"),
                    started_at,
                )));
            }
        }
        lists.insert((*field).to_string(), members);
    }
    let base = base.clone();
    let shards_projection: Vec<BTreeMap<String, Vec<String>>> = state
        .map()
        .all()
        .map(|shard| {
            lists
                .iter()
                .map(|(field, members)| {
                    (
                        field.clone(),
                        state
                            .map()
                            .project(members.iter().map(String::as_str), shard),
                    )
                })
                .collect()
        })
        .collect();
    Ok(move |shard: usize| {
        let mut body = base.clone();
        for (field, members) in &shards_projection[shard] {
            body[field.as_str()] = json!(members);
        }
        Some(Bytes::from(body.to_string()))
    })
}

async fn create_group_broadcast(
    State(state): State<RouterState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    body: Bytes,
) -> Response {
    let started_at = Instant::now();
    let base: Value = if body.is_empty() {
        json!({})
    } else {
        match serde_json::from_slice(&body) {
            Ok(base) => base,
            Err(_) => {
                // Malformed bodies go to one shard untouched so the
                // refusal is the shard's own extractor shape.
                return forward_group_probe(&state, Method::PUT, &name, headers, body, deadline)
                    .await;
            }
        }
    };
    let path = format!("/groups/{}", urlencode(&name));
    let body_for = match project_body(&state, &base, &["contexts"], started_at) {
        Ok(body_for) => body_for,
        Err(refusal) => return *refusal,
    };
    match broadcast_group_write(
        &state,
        Method::PUT,
        &path,
        &headers,
        body_for,
        deadline,
        started_at,
    )
    .await
    {
        Ok(_) => api::ok(true, started_at),
        Err(refusal) => *refusal,
    }
}

async fn update_group_broadcast(
    State(state): State<RouterState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    body: Bytes,
) -> Response {
    let started_at = Instant::now();
    let base: Value = match serde_json::from_slice(&body) {
        Ok(base) => base,
        Err(_) => {
            return forward_group_probe(&state, Method::PATCH, &name, headers, body, deadline)
                .await;
        }
    };
    let path = format!("/groups/{}", urlencode(&name));
    let body_for = match project_body(
        &state,
        &base,
        &["add_contexts", "remove_contexts"],
        started_at,
    ) {
        Ok(body_for) => body_for,
        Err(refusal) => return *refusal,
    };
    match broadcast_group_write(
        &state,
        Method::PATCH,
        &path,
        &headers,
        body_for,
        deadline,
        started_at,
    )
    .await
    {
        Ok(answers) => {
            let mut rows: BTreeMap<String, api::GroupEntry> = BTreeMap::new();
            for body in answers {
                if let Ok(envelope) =
                    serde_json::from_slice::<ShardEnvelope<api::GroupEntry>>(&body)
                {
                    merge_group_entry(&mut rows, envelope.result);
                }
            }
            match rows.into_values().next() {
                Some(entry) => api::ok(entry, started_at),
                None => api::error(
                    ErrorCode::Internal,
                    "every shard applied the update but none answered a readable entry",
                    started_at,
                ),
            }
        }
        Err(refusal) => *refusal,
    }
}

/// Sends an unparseable body to the group's first shard verbatim, so
/// the refusal (shape, status, message) is the single-instance
/// extractor's own.
async fn forward_group_probe(
    state: &RouterState,
    method: Method,
    name: &str,
    headers: HeaderMap,
    body: Bytes,
    deadline: Deadline,
) -> Response {
    let started_at = Instant::now();
    let path = format!("/groups/{}", urlencode(name));
    match state
        .call_shard(0, method, &path, &headers, Some(body), deadline)
        .await
    {
        Ok(answer) => (
            answer.status,
            [(header::CONTENT_TYPE, "application/json")],
            answer.body,
        )
            .into_response(),
        Err(error) => unreachable_refusal(
            &[Unreached {
                shard: state.map().url(0).to_string(),
                contexts: Vec::new(),
                error,
            }],
            started_at,
        ),
    }
}

async fn delete_group_broadcast(
    State(state): State<RouterState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    group_broadcast_simple(state, Method::DELETE, name, None, headers, deadline).await
}

async fn rename_group_broadcast(
    State(state): State<RouterState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    body: Bytes,
) -> Response {
    group_broadcast_simple(
        state,
        Method::POST,
        format!("{name}/rename"),
        Some(body),
        headers,
        deadline,
    )
    .await
}

/// Delete and rename broadcast the identical request everywhere; a
/// 404 from EVERY shard is the single-instance not-found, while a
/// mixed answer (drift healing) succeeds with the successes.
async fn group_broadcast_simple(
    state: RouterState,
    method: Method,
    name_path: String,
    body: Option<Bytes>,
    headers: HeaderMap,
    deadline: Deadline,
) -> Response {
    let started_at = Instant::now();
    let (encoded_name, suffix) = match name_path.split_once('/') {
        Some((name, suffix)) => (urlencode(name), format!("/{suffix}")),
        None => (urlencode(&name_path), String::new()),
    };
    let path = format!("/groups/{encoded_name}{suffix}");
    let mut not_found: Option<ShardAnswer> = None;
    let mut succeeded = false;
    for shard in state.map().all() {
        match state
            .call_shard(
                shard,
                method.clone(),
                &path,
                &headers,
                body.clone(),
                deadline,
            )
            .await
        {
            Ok(answer) if answer.status.is_success() => succeeded = true,
            Ok(answer) if answer.status == StatusCode::NOT_FOUND => not_found = Some(answer),
            Ok(answer) => {
                return (
                    answer.status,
                    [(header::CONTENT_TYPE, "application/json")],
                    answer.body,
                )
                    .into_response();
            }
            Err(error) => {
                return unreachable_refusal(
                    &[Unreached {
                        shard: state.map().url(shard).to_string(),
                        contexts: Vec::new(),
                        error,
                    }],
                    started_at,
                );
            }
        }
    }
    match (succeeded, not_found) {
        (true, _) => api::ok(true, started_at),
        (false, Some(answer)) => (
            answer.status,
            [(header::CONTENT_TYPE, "application/json")],
            answer.body,
        )
            .into_response(),
        (false, None) => api::ok(true, started_at),
    }
}

async fn union_group(
    State(state): State<RouterState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    let path = format!("/groups/{}", urlencode(&name));
    let shards: Vec<usize> = state.map().all().collect();
    let outcomes = state
        .fan_out(&shards, Method::GET, &path, &headers, |_| None, deadline)
        .await;
    let mut rows: BTreeMap<String, api::GroupEntry> = BTreeMap::new();
    let mut not_found: Option<ShardAnswer> = None;
    for (shard, outcome) in outcomes {
        match outcome {
            Ok(answer) if answer.status.is_success() => {
                match serde_json::from_slice::<ShardEnvelope<api::GroupEntry>>(&answer.body) {
                    Ok(envelope) => merge_group_entry(&mut rows, envelope.result),
                    Err(error) => {
                        return api::error(
                            ErrorCode::Internal,
                            format!(
                                "shard {} answered an unreadable group: {error}",
                                state.map().url(shard)
                            ),
                            started_at,
                        );
                    }
                }
            }
            Ok(answer) if answer.status == StatusCode::NOT_FOUND => not_found = Some(answer),
            Ok(answer) => {
                return (
                    answer.status,
                    [(header::CONTENT_TYPE, "application/json")],
                    answer.body,
                )
                    .into_response();
            }
            Err(error) => {
                return unreachable_refusal(
                    &[Unreached {
                        shard: state.map().url(shard).to_string(),
                        contexts: Vec::new(),
                        error,
                    }],
                    started_at,
                );
            }
        }
    }
    match (rows.into_values().next(), not_found) {
        (Some(entry), _) => api::ok(entry, started_at),
        (None, Some(answer)) => (
            answer.status,
            [(header::CONTENT_TYPE, "application/json")],
            answer.body,
        )
            .into_response(),
        (None, None) => api::error(
            ErrorCode::NoGroup,
            format!("group '{name}' not found"),
            started_at,
        ),
    }
}

/// `GET /groups/{name}/export`: every shard's record line names its
/// own projection; the union record — one line, importable — is what
/// the group actually is.
async fn export_group_union(
    State(state): State<RouterState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    let path = format!("/groups/{}/export", urlencode(&name));
    let shards: Vec<usize> = state.map().all().collect();
    let outcomes = state
        .fan_out(&shards, Method::GET, &path, &headers, |_| None, deadline)
        .await;
    let mut merged: Option<crate::groups::GroupRecord> = None;
    let mut not_found: Option<ShardAnswer> = None;
    for (shard, outcome) in outcomes {
        match outcome {
            Ok(answer) if answer.status.is_success() => {
                let Some(record) = parse_group_export(&answer.body) else {
                    return api::error(
                        ErrorCode::Internal,
                        format!(
                            "shard {} answered an unreadable group record",
                            state.map().url(shard)
                        ),
                        started_at,
                    );
                };
                match &mut merged {
                    Some(held) => {
                        held.contexts.extend(record.contexts);
                        held.groups.extend(record.groups);
                    }
                    None => merged = Some(record),
                }
            }
            Ok(answer) if answer.status == StatusCode::NOT_FOUND => not_found = Some(answer),
            Ok(answer) => {
                return (
                    answer.status,
                    [(header::CONTENT_TYPE, "application/json")],
                    answer.body,
                )
                    .into_response();
            }
            Err(error) => {
                return unreachable_refusal(
                    &[Unreached {
                        shard: state.map().url(shard).to_string(),
                        contexts: Vec::new(),
                        error,
                    }],
                    started_at,
                );
            }
        }
    }
    match (merged, not_found) {
        (Some(record), _) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/x-ndjson; charset=utf-8")],
            crate::export::render_group(&name, &record),
        )
            .into_response(),
        (None, Some(answer)) => (
            answer.status,
            [(header::CONTENT_TYPE, "application/json")],
            answer.body,
        )
            .into_response(),
        (None, None) => api::error(
            ErrorCode::NoGroup,
            format!("group '{name}' not found"),
            started_at,
        ),
    }
}

/// One shard's export body back into a record: a single
/// `taguru_group` line, the same shape `parse_group` reads.
fn parse_group_export(body: &Bytes) -> Option<crate::groups::GroupRecord> {
    let text = std::str::from_utf8(body).ok()?;
    let line = text.lines().find(|line| !line.trim().is_empty())?;
    let value: Value = serde_json::from_str(line).ok()?;
    let object = value.as_object()?;
    let string_set = |key: &str| -> BTreeSet<String> {
        object
            .get(key)
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default()
    };
    Some(crate::groups::GroupRecord {
        description: object
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        contexts: string_set("contexts"),
        groups: string_set("groups"),
    })
}

// ---------------------------------------------------------------------------
// Broadcast operator verbs

async fn broadcast_flush(
    State(state): State<RouterState>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    let shards: Vec<usize> = state.map().all().collect();
    let outcomes = state
        .fan_out(
            &shards,
            Method::POST,
            "/flush",
            &headers,
            |_| None,
            deadline,
        )
        .await;
    let mut flushed: Vec<String> = Vec::new();
    let mut unreached = Vec::new();
    for (shard, outcome) in outcomes {
        match outcome {
            Ok(answer) if answer.status.is_success() => {
                if let Ok(envelope) =
                    serde_json::from_slice::<ShardEnvelope<Vec<String>>>(&answer.body)
                {
                    flushed.extend(envelope.result);
                }
            }
            Ok(answer) => {
                return (
                    answer.status,
                    [(header::CONTENT_TYPE, "application/json")],
                    answer.body,
                )
                    .into_response();
            }
            Err(error) => unreached.push(Unreached {
                shard: state.map().url(shard).to_string(),
                contexts: Vec::new(),
                error,
            }),
        }
    }
    if flushed.is_empty() && unreached.len() == shards.len() && !shards.is_empty() {
        return unreachable_refusal(&unreached, started_at);
    }
    router_ok(flushed, unreached, started_at)
}

async fn broadcast_maintenance(
    State(state): State<RouterState>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    request: Request,
) -> Response {
    let started_at = Instant::now();
    let path = full_path(&request);
    let shards: Vec<usize> = state.map().all().collect();
    // Sequential on purpose: each shard's sweep drains its own
    // traffic; running them one at a time keeps the fleet from
    // pausing everywhere at once.
    let mut contexts: Vec<Value> = Vec::new();
    let mut deadline_exceeded = false;
    let mut unreached = Vec::new();
    for shard in shards {
        match state
            .call_shard(shard, Method::POST, &path, &headers, None, deadline)
            .await
        {
            Ok(answer) if answer.status.is_success() => {
                if let Ok(envelope) = serde_json::from_slice::<ShardEnvelope<Value>>(&answer.body) {
                    if let Some(swept) = envelope.result.get("contexts").and_then(Value::as_array) {
                        contexts.extend(swept.iter().cloned());
                    }
                    deadline_exceeded |= envelope
                        .result
                        .get("deadline_exceeded")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                }
            }
            Ok(answer) => {
                return (
                    answer.status,
                    [(header::CONTENT_TYPE, "application/json")],
                    answer.body,
                )
                    .into_response();
            }
            Err(error) => unreached.push(Unreached {
                shard: state.map().url(shard).to_string(),
                contexts: Vec::new(),
                error,
            }),
        }
    }
    router_ok(
        json!({"contexts": contexts, "deadline_exceeded": deadline_exceeded}),
        unreached,
        started_at,
    )
}

// ---------------------------------------------------------------------------
// /import: split by batch header, preflight, dispatch in stream order

async fn route_import(
    State(state): State<RouterState>,
    headers: HeaderMap,
    axum::Extension(deadline): axum::Extension<Deadline>,
    api::AppQuery(query): api::AppQuery<api::ImportQuery>,
    api::AppBytes(body): api::AppBytes,
) -> Response {
    let started_at = Instant::now();
    // The same whole-stream validation a shard would run, so a
    // malformed stream refuses here with the same message and the same
    // stream-level line numbers.
    let stream = match crate::ingest::parse_stream(&body[..]) {
        Ok(stream) => stream,
        Err(message) => return api::error(ErrorCode::MalformedRequest, message, started_at),
    };
    let slices = split_batches(&body);
    debug_assert_eq!(slices.len(), stream.batches.len());
    // Route every batch before anything ships: a stream naming an
    // unroutable context refuses whole, like any other invalid stream.
    let mut chunks: Vec<(usize, Vec<std::ops::Range<usize>>)> = Vec::new();
    for (batch, range) in stream.batches.iter().zip(slices) {
        let Some(shard) = state.map().shard_of(&batch.context) else {
            return api::error(
                ErrorCode::NoContext,
                format!(
                    "batch source '{}': context '{}' has no route-map entry and no '*' \
                     fallback (TAGURU_ROUTE_MAP); nothing was applied",
                    batch.source, batch.context
                ),
                started_at,
            );
        };
        match chunks.last_mut() {
            Some((last_shard, ranges)) if *last_shard == shard => ranges.push(range),
            _ => chunks.push((shard, vec![range])),
        }
    }
    let assemble = |ranges: &[std::ops::Range<usize>]| {
        let mut chunk = Vec::new();
        for range in ranges {
            chunk.extend_from_slice(&body[range.clone()]);
        }
        Bytes::from(chunk)
    };
    // Each shard's projected group-record stream, rendered once —
    // preflighted below beside the batch chunks, then applied last.
    let group_lines_for = |shard: usize| -> String {
        let mut lines = String::new();
        for (name, record) in &stream.groups {
            let projected = crate::groups::GroupRecord {
                description: record.description.clone(),
                contexts: record
                    .contexts
                    .iter()
                    .filter(|member| state.map().shard_of(member) == Some(shard))
                    .cloned()
                    .collect(),
                groups: record.groups.clone(),
            };
            lines.push_str(&crate::export::render_group(name, &projected));
        }
        lines
    };
    // Preflight: every batch chunk AND every projected group stream
    // dry-runs first, so a refusal a single instance would answer with
    // nothing applied — malformed batches, a scoped key beyond its
    // grant on a batch context or on a group record's closure — is
    // answered the same way here, with nothing applied on any shard.
    // (A real dry_run request IS its own preflight. Group VALIDATION
    // against live state still runs after the batches land, exactly
    // where the single instance runs it.)
    let preflight_query = "/import?dry_run=true";
    let real_query = if query.dry_run {
        "/import?dry_run=true"
    } else {
        "/import"
    };
    if !query.dry_run {
        let group_shards: Vec<usize> = if stream.groups.is_empty() {
            Vec::new()
        } else {
            state.map().all().collect()
        };
        let preflights = chunks
            .iter()
            .map(|(shard, ranges)| (*shard, assemble(ranges)))
            .chain(
                group_shards
                    .iter()
                    .map(|&shard| (shard, Bytes::from(group_lines_for(shard)))),
            );
        for (shard, chunk) in preflights {
            match state
                .call_shard(
                    shard,
                    Method::POST,
                    preflight_query,
                    &headers,
                    Some(chunk),
                    deadline,
                )
                .await
            {
                Ok(answer) if answer.status.is_success() => {}
                Ok(answer) => {
                    return (
                        answer.status,
                        [(header::CONTENT_TYPE, "application/json")],
                        answer.body,
                    )
                        .into_response();
                }
                Err(error) => {
                    return unreachable_refusal(
                        &[Unreached {
                            shard: state.map().url(shard).to_string(),
                            contexts: Vec::new(),
                            error,
                        }],
                        started_at,
                    );
                }
            }
        }
    }
    // The real run, chunk by chunk in stream order. A refusal that
    // survived preflight (mid-apply IO, a budget spent partway) stops
    // here exactly as it stops a single instance partway: the batches
    // before it landed durably, and re-POSTing the stream is exact.
    let mut batches: Vec<Value> = Vec::new();
    for (index, (shard, ranges)) in chunks.iter().enumerate() {
        match state
            .call_shard(
                *shard,
                Method::POST,
                real_query,
                &headers,
                Some(assemble(ranges)),
                deadline,
            )
            .await
        {
            Ok(answer) if answer.status.is_success() => {
                if let Ok(envelope) = serde_json::from_slice::<ShardEnvelope<Value>>(&answer.body)
                    && let Some(outcomes) = envelope.result.get("batches").and_then(Value::as_array)
                {
                    batches.extend(outcomes.iter().cloned());
                }
            }
            Ok(answer) => {
                return rewrap_import_refusal(answer, batches.len(), index > 0, started_at);
            }
            Err(error) => {
                return unreachable_refusal(
                    &[Unreached {
                        shard: state.map().url(*shard).to_string(),
                        contexts: Vec::new(),
                        error,
                    }],
                    started_at,
                );
            }
        }
    }
    // Group records apply LAST, after every batch — the same order the
    // stream contract promises — projected per shard and broadcast. A
    // dry run dispatches them too (with dry_run forwarded): the shards
    // still parse and scope-check the records exactly as a single
    // instance's dry run does, while omitting them from the preview.
    let mut groups: Vec<Value> = Vec::new();
    if !stream.groups.is_empty() {
        let mut per_group: BTreeMap<usize, Vec<GroupOutcomeWire>> = BTreeMap::new();
        for shard in state.map().all() {
            match state
                .call_shard(
                    shard,
                    Method::POST,
                    real_query,
                    &headers,
                    Some(Bytes::from(group_lines_for(shard))),
                    deadline,
                )
                .await
            {
                Ok(answer) if answer.status.is_success() => {
                    if let Ok(envelope) =
                        serde_json::from_slice::<ShardEnvelope<GroupsOnlyWire>>(&answer.body)
                    {
                        for (position, outcome) in envelope.result.groups.into_iter().enumerate() {
                            per_group.entry(position).or_default().push(outcome);
                        }
                    }
                }
                Ok(answer) => {
                    return rewrap_import_refusal(
                        answer,
                        batches.len(),
                        !query.dry_run && !chunks.is_empty(),
                        started_at,
                    );
                }
                Err(error) => {
                    return unreachable_refusal(
                        &[Unreached {
                            shard: state.map().url(shard).to_string(),
                            contexts: Vec::new(),
                            error,
                        }],
                        started_at,
                    );
                }
            }
        }
        for (_, outcomes) in per_group {
            let Some(first) = outcomes.first() else {
                continue;
            };
            let outcome = if outcomes.iter().all(|entry| entry.outcome == "unchanged") {
                "unchanged"
            } else if outcomes.iter().all(|entry| entry.outcome == "created") {
                "created"
            } else {
                "replaced"
            };
            groups.push(json!({
                "name": first.name,
                "outcome": outcome,
                // Projections partition the member set, so the true
                // count is their sum; the child-group list is
                // broadcast whole, so any shard's count is the count.
                "contexts": outcomes.iter().map(|entry| entry.contexts).sum::<usize>(),
                "groups": first.groups,
            }));
        }
    }
    let mut result = json!({"batches": batches});
    if !groups.is_empty() {
        result["groups"] = json!(groups);
    }
    api::ok(result, started_at)
}

#[derive(Deserialize)]
struct GroupsOnlyWire {
    #[serde(default)]
    groups: Vec<GroupOutcomeWire>,
}

#[derive(Deserialize)]
struct GroupOutcomeWire {
    name: String,
    outcome: String,
    contexts: usize,
    groups: usize,
}

/// A shard refusal from the real (post-preflight) import run, with the
/// cross-chunk truth prepended when earlier chunks already landed —
/// the shard's own note counts only its chunk.
fn rewrap_import_refusal(
    answer: ShardAnswer,
    batches_landed: usize,
    other_chunks_landed: bool,
    started_at: Instant,
) -> Response {
    if !other_chunks_landed {
        return (
            answer.status,
            [(header::CONTENT_TYPE, "application/json")],
            answer.body,
        )
            .into_response();
    }
    let (code, message) = match serde_json::from_slice::<Value>(&answer.body) {
        Ok(body) => (
            body.get("code")
                .and_then(Value::as_str)
                .unwrap_or("internal")
                .to_string(),
            body.get("error")
                .and_then(Value::as_str)
                .unwrap_or("shard refusal with an unreadable body")
                .to_string(),
        ),
        Err(_) => (
            "internal".to_string(),
            String::from_utf8_lossy(&answer.body).into_owned(),
        ),
    };
    let rewrapped = json!({
        "status": "error",
        "code": code,
        "error": format!(
            "{batches_landed} batch(es) landed durably on earlier shards before this \
             refusal (re-POSTing the whole stream is exact — each batch replaces its own \
             source); the refusing shard says: {message}"
        ),
        "time": started_at.elapsed().as_secs_f64(),
    });
    (
        answer.status,
        [(header::CONTENT_TYPE, "application/json")],
        rewrapped.to_string(),
    )
        .into_response()
}

/// Byte ranges of each batch in a stream `parse_stream` already
/// validated: a batch runs from its `taguru_batch` header line to the
/// next stream-level record (header or `taguru_group` line) or EOF.
/// Group-record bytes belong to no batch — they are re-rendered from
/// the parsed records instead of sliced.
fn split_batches(body: &[u8]) -> Vec<std::ops::Range<usize>> {
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
    let mut current_start: Option<usize> = None;
    let mut offset = 0usize;
    for line in body.split_inclusive(|byte| *byte == b'\n') {
        let start = offset;
        offset += line.len();
        let mut text = line;
        if start == 0 && text.starts_with(&[0xEF, 0xBB, 0xBF]) {
            text = &text[3..];
        }
        let Ok(text) = std::str::from_utf8(text) else {
            continue;
        };
        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };
        let Some(object) = value.as_object() else {
            continue;
        };
        if object.contains_key("taguru_batch") || object.contains_key("taguru_group") {
            if let Some(batch_start) = current_start.take() {
                ranges.push(batch_start..start);
            }
            if object.contains_key("taguru_batch") {
                current_start = Some(start);
            }
        }
    }
    if let Some(batch_start) = current_start {
        ranges.push(batch_start..body.len());
    }
    ranges
}

// ---------------------------------------------------------------------------
// Router-owned endpoints

async fn health(State(state): State<RouterState>) -> Response {
    // The router itself has no degraded write path to report; shard
    // health belongs to the shards' own probes (and surfaces here per
    // request as pass-through errors and `unreached` labels).
    axum::Json(json!({
        "status": "ok",
        "router": true,
        "shards": state.map().shards.len(),
    }))
    .into_response()
}

async fn render_metrics(State(state): State<RouterState>) -> Response {
    let mut out = String::new();
    out.push_str("# TYPE taguru_router gauge\ntaguru_router 1\n");
    out.push_str(&format!(
        "# TYPE taguru_router_shards gauge\ntaguru_router_shards {}\n",
        state.map().shards.len()
    ));
    out.push_str("# TYPE taguru_router_requests_total counter\n");
    for ((route, status), count) in state.inner.metrics.http.lock().iter() {
        out.push_str(&format!(
            "taguru_router_requests_total{{route=\"{route}\",status=\"{status}\"}} {count}\n"
        ));
    }
    out.push_str("# TYPE taguru_router_shard_requests_total counter\n");
    for ((shard, outcome), count) in state.inner.metrics.shard.lock().iter() {
        out.push_str(&format!(
            "taguru_router_shard_requests_total{{shard=\"{}\",outcome=\"{outcome}\"}} {count}\n",
            state.map().url(*shard)
        ));
    }
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        out,
    )
        .into_response()
}

/// `GET /protocol`, proxied from the first shard that answers — the
/// manual is deploy-wide and shards are expected to be homogeneous
/// (same version, same embedding configuration); the router itself
/// has no embedding tier to describe.
async fn proxy_protocol(
    State(state): State<RouterState>,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    let mut unreached = Vec::new();
    for shard in state.map().all() {
        match state
            .call_shard(
                shard,
                Method::GET,
                "/protocol",
                &HeaderMap::new(),
                None,
                deadline,
            )
            .await
        {
            Ok(answer) if answer.status.is_success() => {
                return (
                    answer.status,
                    [(header::CONTENT_TYPE, "text/markdown; charset=utf-8")],
                    answer.body,
                )
                    .into_response();
            }
            Ok(answer) => {
                return (
                    answer.status,
                    [(header::CONTENT_TYPE, "application/json")],
                    answer.body,
                )
                    .into_response();
            }
            Err(error) => unreached.push(Unreached {
                shard: state.map().url(shard).to_string(),
                contexts: Vec::new(),
                error,
            }),
        }
    }
    unreachable_refusal(&unreached, started_at)
}

/// Percent-encodes one path segment the way the stdio bridge does:
/// enough for a context/group name to survive the round trip.
fn urlencode(segment: &str) -> String {
    let mut encoded = String::with_capacity(segment.len());
    for byte in segment.bytes() {
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char)
            }
            other => encoded.push_str(&format!("%{other:02X}")),
        }
    }
    encoded
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
    fn split_batches_slices_exactly_the_bytes_between_stream_level_records() {
        let body = concat!(
            "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s1\"}\n",
            "{\"assoc\": [\"a\", \"likes\", \"b\"]}\n",
            "\n",
            "{\"taguru_group\": 1, \"name\": \"g\", \"contexts\": [\"sake\"]}\n",
            "{\"taguru_batch\": 1, \"context\": \"beer\", \"source\": \"s2\"}\n",
            "{\"assoc\": [\"c\", \"likes\", \"d\"]}",
        )
        .as_bytes();
        let ranges = split_batches(body);
        assert_eq!(ranges.len(), 2);
        let first = std::str::from_utf8(&body[ranges[0].clone()]).unwrap();
        assert!(first.starts_with("{\"taguru_batch\": 1, \"context\": \"sake\""));
        // The batch's ops (and the blank line) ride along; the group
        // record between the batches belongs to neither.
        assert!(first.contains("likes"));
        assert!(!first.contains("taguru_group"));
        let second = std::str::from_utf8(&body[ranges[1].clone()]).unwrap();
        assert!(second.starts_with("{\"taguru_batch\": 1, \"context\": \"beer\""));
        assert!(second.ends_with("\"d\"]}"), "EOF closes the last batch");
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
