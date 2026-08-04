//! Boot and the route table: `run` builds the router's `RouterState`
//! and axum `Router`, wires the in-process `/mcp` dispatch, and starts
//! serving; `routes` names every path the router answers.

use super::*;

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
                    // No key and no scope: the router authenticates
                    // nobody — the bearer rides the reattached header
                    // and the SHARDS judge it.
                    crate::remote_mcp::serve(
                        dispatch,
                        instructions,
                        None,
                        None,
                        headers,
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
        // Shared verbatim with `serve` mode (ADR 0005 §6: router mode
        // answers under the same "shards are homogeneous" assumption
        // `/health` and `/protocol` already proxy under) — every field
        // is a compile-time constant, so there is nothing router-mode
        // needs to add, unlike `/health`'s own `router`/`shards` facts.
        .route("/version", get(crate::metrics::version))
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

/// Whether an MCP message is an `initialize` or a `server/discover` —
/// the two methods whose reply carries the manual (one per era). A
/// cheap peek, not a validation: anything unparseable goes to
/// `remote_mcp::serve` for its own refusal.
fn wants_instructions(body: &Bytes) -> bool {
    serde_json::from_slice::<Value>(body)
        .ok()
        .and_then(|message| {
            message
                .get("method")
                .and_then(Value::as_str)
                .map(|method| method == "initialize" || method == "server/discover")
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
/// Shares `crate::trace::traced_request`/`normalized_method` with that
/// function (ADR 0008 §5) — router mode calls `init_telemetry` at boot
/// like `serve` does, but until now made no span of its own, so its
/// access log carried no `trace_id` to correlate with a collector.
async fn track_router_http(
    State(state): State<RouterState>,
    matched: Option<MatchedPath>,
    request: Request,
    next: Next,
) -> Response {
    let method = crate::trace::normalized_method(request.method());
    let route = matched
        .as_ref()
        .map(|matched| matched.as_str().to_string())
        .unwrap_or_else(|| "<unmatched>".to_string());
    let started = Instant::now();

    let (response, trace_id) = if crate::trace::enabled() {
        crate::trace::traced_request(method, &route, request, next).await
    } else {
        (next.run(request).await, None)
    };

    let status = response.status().as_u16();
    state.inner.metrics.record_http(&route, status);
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    match trace_id {
        Some(trace_id) => info!(
            method = %method,
            route = %route,
            status,
            latency_ms,
            trace_id = %trace_id,
            "http",
        ),
        None => info!(
            method = %method,
            route = %route,
            status,
            latency_ms,
            "http",
        ),
    }
    response
}

/// The per-call slice of the request budget. `None` means unbounded —
/// only the CLI paths produce that; served requests always carry the
/// timeout middleware's deadline.
pub(super) fn budget(deadline: Deadline) -> Option<Duration> {
    let remaining = deadline.remaining();
    (remaining != Duration::MAX).then_some(remaining)
}
