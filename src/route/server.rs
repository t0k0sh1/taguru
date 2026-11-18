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
    for var in ["TAGURU_API_TOKEN", "TAGURU_API_TOKENS", "TAGURU_KEY_GRANTS"] {
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
    let map_text = match std::fs::read_to_string(&map_path) {
        Ok(text) => text,
        Err(error) => {
            tracing::error!(path = %map_path.display(), %error, "TAGURU_ROUTE_MAP is not usable");
            std::process::exit(1);
        }
    };
    let map = match RouteMap::parse(&map_text) {
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
            map: parking_lot::RwLock::new(Arc::new(map)),
            client,
            metrics: RouterMetrics::default(),
            instructions: OnceLock::new(),
        }),
    };
    // Editing the map no longer takes a router restart (issue #515):
    // SIGHUP and the map-file watch swap it at runtime, keyring-style.
    // The watch's change baseline is the digest of the bytes boot just
    // APPLIED — not a fresh read of the file: a rewrite landing in the
    // gap between boot's read and the watch task's first read would
    // otherwise be recorded as already-seen without ever being applied.
    let background = spawn_route_map_reload_tasks(
        state.clone(),
        map_path,
        crate::sha256::sha256_hex(map_text.as_bytes()),
    );

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
                    // Only `initialize` and `server/discover` read the
                    // manual — both carry it as `instructions` — so only
                    // they pay the shard probe (once; the result is
                    // cached). A tool call must never spend its budget
                    // probing shards for text it will not use (with
                    // every shard down, the probes would eat the whole
                    // deadline first).
                    let instructions = if wants_instructions(&body) {
                        state.mcp_instructions(deadline).await
                    } else {
                        Arc::new(String::new())
                    };
                    // No key and no grant: the router authenticates
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
    let booted_map = state.map();
    info!(
        addr = %listener.local_addr().unwrap(),
        shards = booted_map.shards.len(),
        mapped_contexts = booted_map.contexts.len(),
        timeout_secs,
        max_body_mib = max_body_bytes / (1024 * 1024),
        "router ready — stateless; auth, grants, and rate limits are enforced by the shards",
    );
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(crate::shutdown_signal())
    .await
    .unwrap();

    // The watch tasks first, before the runtime is dropped (#898).
    crate::stop_background_tasks(background).await;

    if let Some(provider) = tracer_provider
        && let Err(error) = tokio::task::block_in_place(|| provider.shutdown())
    {
        warn!(error = %error, "trace export flush on shutdown failed");
    }
}

/// One reload attempt from the map file's bytes: parse whole, swap on
/// success, refuse on failure — the old map keeps serving, exactly as
/// a boot-time refusal keeps the process from serving a broken one.
/// The keyring's reload semantics (issue #134), applied to the route
/// map (issue #515). Returns whether the swap happened. Both runtime
/// triggers reach this through [`MapReloads`], which reads `bytes`
/// under its lock.
pub(super) fn reload_route_map(state: &RouterState, bytes: &[u8], trigger: &str) -> bool {
    let parsed = std::str::from_utf8(bytes)
        .map_err(|error| format!("not UTF-8: {error}"))
        .and_then(RouteMap::parse);
    match parsed {
        Ok(map) => {
            info!(
                trigger,
                shards = map.shards.len(),
                contexts = map.contexts.len(),
                fallback = map.fallback.map(|index| map.shards[index].clone()),
                "route map reloaded"
            );
            state.swap_map(map);
            state.inner.metrics.record_map_reload(true);
            true
        }
        Err(error) => {
            warn!(
                trigger,
                %error,
                "route map reload refused — the map already serving stays in effect"
            );
            state.inner.metrics.record_map_reload(false);
            false
        }
    }
}

/// The route map's reload ledger, shared by both triggers (SIGHUP and
/// the map-file watch). Every reload reads the file UNDER `judged`'s
/// lock and swaps before releasing it, so reloads are serialized end
/// to end — read → parse → swap — and none can swap in bytes another
/// reload already superseded. Read outside the lock, a SIGHUP's v1
/// could land after a watch tick armed the newer v2; with v2 already
/// recorded as judged, the stale map would keep serving with nothing
/// left to re-trigger. The keyring's `SharedKeyring::reload_serial`
/// discipline, applied to the map.
struct MapReloads {
    path: PathBuf,
    /// The digest of the newest map content any reload judged —
    /// applied or refused — seeded with the bytes boot applied, not a
    /// fresh read (#942), so an edit racing the watch's startup still
    /// registers as a change. The watch's change detector, shared with
    /// SIGHUP on purpose: a watch-private digest goes stale once a
    /// SIGHUP arms other content, and a file then reverted to what the
    /// watch last judged would read as unchanged. Refusals count: one
    /// loud line per change, not one per tick.
    judged: parking_lot::Mutex<String>,
}

impl MapReloads {
    fn new(path: PathBuf, boot_digest: String) -> Self {
        Self {
            path,
            judged: parking_lot::Mutex::new(boot_digest),
        }
    }

    /// SIGHUP's reload: an operator asked, so whatever the file says is
    /// judged, already-judged content included. `None`: the file is
    /// unreadable, and the map already serving stays in effect.
    fn reload(&self, state: &RouterState) -> Option<bool> {
        self.judge(state, "sighup", false)
    }

    /// A watch tick's reload, for content no reload has judged yet — a
    /// digest, not `(mtime, len)`, for the same reasons the keyring
    /// watch hashes (issue #309). `None` when nothing was judged: an
    /// unreadable file (transient volume states must neither trigger a
    /// reload nor poison the digest), or content already judged.
    fn reload_if_changed(&self, state: &RouterState) -> Option<bool> {
        self.judge(state, "map-watch", true)
    }

    /// Blocking: the read happens under the lock, so the reload tasks
    /// run this on an off-worker thread (`read_off_worker` for a tick,
    /// `read_off_worker_queued` for a SIGHUP), never on a worker.
    fn judge(&self, state: &RouterState, trigger: &str, skip_judged: bool) -> Option<bool> {
        let mut judged = self.judged.lock();
        let bytes = std::fs::read(&self.path).ok()?;
        let digest = crate::sha256::sha256_hex(&bytes);
        if skip_judged && *judged == digest {
            return None;
        }
        *judged = digest;
        Some(reload_route_map(state, &bytes, trigger))
    }
}

/// Arms the route-map hot-reload triggers (issue #515) — the same two
/// triggers, same swap discipline, as the keyring's
/// `spawn_keyring_reload_tasks`:
///
/// - **SIGHUP** (unix only; a failed registration degrades to "no
///   signal reload" with a warning, never a crash): for an operator
///   who wants the new shard assignment NOW.
/// - **The map-file watch** (every platform): a poll of the file's
///   content digest every `CONFIG_WATCH_INTERVAL` — a digest, not
///   `(mtime, len)`, for the same reasons the keyring watch hashes
///   (issue #309: same-length rewrites, symlink swaps inside one
///   filesystem-clock tick). A ConfigMap-style atomic symlink swap is
///   picked up with no signal delivered.
///
/// Requests already in flight keep the snapshot they took at entry
/// ([`RouterState::map`]); only new requests see a swapped map.
/// `boot_digest` is the sha256 of the bytes boot applied — the
/// reloads' judged baseline (see [`MapReloads`]).
fn spawn_route_map_reload_tasks(
    state: RouterState,
    path: PathBuf,
    boot_digest: String,
) -> Vec<tokio::task::JoinHandle<()>> {
    let mut tasks = Vec::new();
    // Shared with the keyring watch — see `crate::read_off_worker`
    // (issue #900): off a plain thread, not tokio's blocking pool, so
    // a stalled read on a network mount can neither wedge this loop
    // nor outlive graceful shutdown — and the whole reload with it,
    // since each one reads under `MapReloads`' lock. SIGHUP and the
    // periodic tick both read this same path, so they share one
    // `in_flight` flag too: a SIGHUP landing mid-stall never piles a
    // second blocked thread behind the tick's, and — unlike a tick —
    // it is never skipped either, but queues until the path is free.
    use crate::{ReadInFlight, read_off_worker};
    let in_flight = Arc::new(ReadInFlight::default());
    let reloads = Arc::new(MapReloads::new(path, boot_digest));
    #[cfg(unix)]
    {
        use crate::read_off_worker_queued;

        let state = state.clone();
        let reloads = Arc::clone(&reloads);
        let in_flight = Arc::clone(&in_flight);
        // Registered before the task is spawned (#892, the same window
        // `shutdown_signal` closes): a SIGHUP before the task's first
        // poll would terminate the router instead of reloading the map.
        let hangup = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup()) {
            Ok(stream) => Some(stream),
            Err(error) => {
                warn!(%error, "could not register the SIGHUP route-map-reload handler");
                None
            }
        };
        if let Some(mut hangup) = hangup {
            tasks.push(tokio::spawn(async move {
                while hangup.recv().await.is_some() {
                    let (state, reload) = (state.clone(), Arc::clone(&reloads));
                    let judged =
                        read_off_worker_queued(&in_flight, move || reload.reload(&state), || {
                            warn!(
                                path = %reloads.path.display(),
                                "route map SIGHUP reload is waiting on a read of the map that \
                                 has not returned yet — it runs once that read does; the map \
                                 already serving stays in effect meanwhile"
                            );
                        })
                        .await;
                    if judged.is_none() {
                        warn!(
                            path = %reloads.path.display(),
                            "route map unreadable on SIGHUP — the map already serving stays in effect"
                        );
                    }
                }
            }));
        }
    }
    tasks.push(tokio::spawn(async move {
        let mut ticker = tokio::time::interval(crate::CONFIG_WATCH_INTERVAL);
        ticker.tick().await; // fires immediately; boot already read the file
        loop {
            ticker.tick().await;
            let (state, reload) = (state.clone(), Arc::clone(&reloads));
            let _ = read_off_worker(&in_flight, move || reload.reload_if_changed(&state)).await;
        }
    }));
    tasks
}

/// Every route the router answers. The two `/contexts/{id}` entries
/// proxy ANY method so a shard's own 405/404 shapes pass through
/// untouched — and so `context` verbs added to the server later route
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
        .route(
            "/contexts",
            get(merge_contexts).post(proxy::route_create_context),
        )
        .route("/contexts/{id}", any(proxy_context_root))
        .route("/contexts/{id}/{*rest}", any(proxy_context_sub))
        .route("/groups", get(merge_groups).post(create_group_broadcast))
        .route(
            "/groups/{id}",
            get(union_group)
                .patch(update_group_broadcast)
                .delete(delete_group_broadcast),
        )
        .route("/groups/{id}/export", get(export_group_union))
        .route("/groups/{id}/rename", post(rename_group_broadcast))
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
        api::PANIC_RESPONSE_MESSAGE,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> RouterState {
        RouterState {
            inner: Arc::new(RouterInner {
                map: parking_lot::RwLock::new(Arc::new(
                    RouteMap::parse("sake = http://a:1\n").unwrap(),
                )),
                client: reqwest::Client::new(),
                metrics: RouterMetrics::default(),
                instructions: OnceLock::new(),
            }),
        }
    }

    /// The router's panic handler answers the shared 500 envelope with
    /// the fixed sentence — never the payload's text — and counts the
    /// panic in the router's own HTTP metrics; a non-string payload
    /// takes the same path.
    #[tokio::test]
    async fn a_router_panic_answers_the_fixed_sentence_and_is_counted() {
        let state = state();
        let payloads: Vec<Box<dyn std::any::Any + Send>> = vec![
            Box::new("kaboom-string".to_string()),
            Box::new("kaboom-str"),
            Box::new(42u32),
        ];
        for payload in payloads {
            let response = router_panic_response(payload, &state);
            assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
            let bytes = axum::body::to_bytes(response.into_body(), 1 << 16)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["status"], "error");
            assert_eq!(body["code"], "internal");
            assert_eq!(body["error"], api::PANIC_RESPONSE_MESSAGE, "{body}");
            assert!(!bytes.windows(6).any(|w| w == b"kaboom"), "{body}");
        }
        assert_eq!(
            state
                .inner
                .metrics
                .http
                .lock()
                .get(&("<panic>".to_string(), 500))
                .copied(),
            Some(3)
        );
    }

    /// Where the serving map sends `sake` — the one context every map
    /// in these tests names.
    fn sake_url(state: &RouterState) -> String {
        let map = state.map();
        map.url(map.shard_of("sake").unwrap()).to_string()
    }

    /// A map file under a fresh scratch dir holding `boot`, and the
    /// ledger boot would hand the reload tasks for it.
    fn booted_map(tag: &str, boot: &str) -> (PathBuf, PathBuf, MapReloads) {
        let dir = crate::registry::test_support::scratch_dir(tag);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("route-map");
        std::fs::write(&path, boot).unwrap();
        let reloads = MapReloads::new(path.clone(), crate::sha256::sha256_hex(boot.as_bytes()));
        (dir, path, reloads)
    }

    /// Every map reload — SIGHUP's and the watch's alike — reads the
    /// file UNDER `MapReloads`' lock, not before taking it. Read first,
    /// a SIGHUP's v1 could be swapped in after a watch tick armed the
    /// newer v2 (recording v2 as judged): a stale map serving with
    /// nothing left to re-trigger.
    ///
    /// Deterministic via a FIFO as the map file: the reader's `open`
    /// blocks until this test opens the write end, so the moment our
    /// `open` returns the reload thread is provably INSIDE its read —
    /// and at that moment the lock must already be held.
    #[cfg(unix)]
    #[test]
    fn every_map_reload_reads_the_file_while_holding_the_reload_lock() {
        use std::io::Write;

        let (dir, path, reloads) =
            booted_map("route-reload-read-under-lock", "sake = http://a:1\n");
        std::fs::remove_file(&path).unwrap();
        let status = std::process::Command::new("mkfifo")
            .arg(&path)
            .status()
            .unwrap();
        assert!(status.success(), "mkfifo failed");
        let state = state();

        let sighup = || reloads.reload(&state);
        let watch = || reloads.reload_if_changed(&state);
        let triggers: [(&str, &(dyn Fn() -> Option<bool> + Sync)); 2] =
            [("sighup", &sighup), ("map-watch", &watch)];
        for (port, (trigger, reload)) in (2..).zip(triggers) {
            let (locked_during_read, swapped) = std::thread::scope(|scope| {
                let reload = scope.spawn(reload);
                // Returns only once the reload thread has opened the
                // read end, i.e. is mid-read.
                let mut writer = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
                let locked = reloads.judged.is_locked();
                // Finish the read either way, so a failing assertion
                // below reports instead of hanging the reload thread.
                writer
                    .write_all(format!("sake = http://a:{port}\n").as_bytes())
                    .unwrap();
                drop(writer);
                (locked, reload.join().unwrap())
            });

            assert!(
                locked_during_read,
                "{trigger}: the map read ran outside the reload lock, so it can race another \
                 reload's swap"
            );
            assert_eq!(swapped, Some(true), "{trigger}");
            assert_eq!(sake_url(&state), format!("http://a:{port}"), "{trigger}");
        }

        let _ = std::fs::remove_dir_all(dir);
    }

    /// The watch's change detector is shared with SIGHUP: content a
    /// SIGHUP judged counts as judged, and — what a watch-private
    /// digest gets wrong — content the watch judged EARLIER is a
    /// change again once a SIGHUP has armed something else. A rewrite
    /// picked up by SIGHUP and reverted before the next tick must not
    /// leave the rewrite serving against a file that no longer says it.
    #[test]
    fn a_map_reverted_after_a_sighup_reload_still_reads_as_a_change() {
        let boot = "sake = http://a:1\n";
        let (dir, path, reloads) = booted_map("route-reload-revert", boot);
        let state = state();

        std::fs::write(&path, "sake = http://b:2\n").unwrap();
        assert_eq!(reloads.reload(&state), Some(true));
        // The SIGHUP's judgment is the watch's too: no second reload.
        assert_eq!(reloads.reload_if_changed(&state), None);

        std::fs::write(&path, boot).unwrap();
        assert_eq!(
            reloads.reload_if_changed(&state),
            Some(true),
            "the reverted map must still read as a change"
        );
        assert_eq!(sake_url(&state), "http://a:1");

        let _ = std::fs::remove_dir_all(dir);
    }

    /// The watch's change detector: boot's own bytes (#942) and content
    /// already judged are no reload, an unreadable file neither reloads
    /// nor moves the baseline, and each content change is judged
    /// exactly once — a refusal included, one loud line per change
    /// rather than per tick — while a SIGHUP judges whatever the file
    /// says, and the serving map survives every refusal.
    #[test]
    fn the_map_watch_judges_each_content_change_once() {
        let boot = "sake = http://a:1\n";
        let (dir, path, reloads) = booted_map("route-reload-change-detection", boot);
        let state = state();
        assert_eq!(reloads.reload_if_changed(&state), None);

        std::fs::remove_file(&path).unwrap();
        assert_eq!(reloads.reload_if_changed(&state), None);
        assert_eq!(reloads.reload(&state), None);
        std::fs::write(&path, boot).unwrap();
        assert_eq!(reloads.reload_if_changed(&state), None);

        std::fs::write(&path, "sake = http://b:2\n").unwrap();
        assert_eq!(reloads.reload_if_changed(&state), Some(true));
        assert_eq!(sake_url(&state), "http://b:2");
        assert_eq!(reloads.reload_if_changed(&state), None);

        std::fs::write(&path, "sake http://no-equals\n").unwrap();
        assert_eq!(reloads.reload_if_changed(&state), Some(false));
        assert_eq!(reloads.reload_if_changed(&state), None);
        assert_eq!(reloads.reload(&state), Some(false));
        assert_eq!(sake_url(&state), "http://b:2");

        let _ = std::fs::remove_dir_all(dir);
    }
}
