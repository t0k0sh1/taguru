mod api;
mod auth;
mod bm25;
mod cli;
mod compact;
#[cfg(test)]
mod context_proptest;
// The same file lib.rs includes — see src/crc32c.rs for why the
// checksum primitive is dual-included instead of exported.
mod crc32c;
mod embedding;
mod estimate;
mod export;
mod extract;
mod groups;
mod ingest;
mod inspect;
mod limits;
mod mcp;
mod metrics;
mod oauth;
mod oauth_http;
mod paragraph;
mod passages;
mod registry;
mod remote_mcp;
mod trace;
mod wal;

use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::routing::{get, post};
use embedding::EmbeddingProvider;
use registry::AppState;
use taguru::deadline::Deadline;
use tokio::net::TcpListener;
use tower_http::catch_panic::CatchPanicLayer;
use tracing::{info, warn};

/// Configuration comes from the environment — or from a KEY=VALUE file
/// (`--config FILE` / `TAGURU_CONFIG=FILE`, the `docker --env-file`
/// dialect; real environment variables win over the file):
/// - `TAGURU_DATA_DIR`: where context images and sidecars live (default
///   `data`). Disk is the source of truth; memory is a cache over it.
/// - `TAGURU_CACHE_BYTES`: resident budget for unpinned loaded contexts
///   (default 512 MiB). Past it, least-recently-used contexts are
///   flushed and dropped; pinned contexts live outside the budget.
/// - `TAGURU_FLUSH_SECS`: how often dirty contexts are persisted (default
///   5). With the WAL on this is image-freshness cadence; writes also
///   persist on eviction and on graceful shutdown (SIGINT/SIGTERM).
/// - `TAGURU_WAL`: per-context write-ahead log for acknowledged graph
///   writes (default on). `0`/`false` restores the flush-interval
///   crash-loss window.
/// - `TAGURU_ADDR`: bind address (default 127.0.0.1:8248 — "TAGU" on a
///   phone keypad, clear of the defaults Taguru tends to sit beside:
///   3000 Next.js/Grafana, 8000 FastAPI/Chroma, 8080 Weaviate, 11434
///   Ollama, 6333 Qdrant. Port 0 picks a free port and the resolved
///   address is printed).
/// - `RUST_LOG`: log filter (default `info`), standard EnvFilter syntax.
/// - `TAGURU_LOG_FORMAT`: `json` for one JSON object per log line;
///   anything else keeps the human-readable format. Logs go to stderr.
/// - `TAGURU_LOG_SEARCHES`: `1` emits one `taguru::search` event line
///   per retrieval (context, op, cue, hits) for keyword analysis in
///   the log pipeline. Off by default: cues are memory content, and
///   the standard log stream carries no content.
/// - `TAGURU_MAX_CONCURRENT_HEAVY_OPS`: shared ceiling for concurrent
///   vocabulary audits (including a drift audit's `include_twins`) and
///   per-context compactions (default 2; 0 disables). Excess calls are
///   shed immediately with 503 + `Retry-After`.
/// - `OTEL_EXPORTER_OTLP_ENDPOINT` (or the `_TRACES_` variant): turns
///   on OTLP/HTTP span export — one span per request, parented from an
///   inbound `traceparent` or `X-Amzn-Trace-Id`, `trace_id` stamped
///   into the access log. The other standard `OTEL_*` variables
///   (service name, headers, batch cadence) apply. Unset: no tracing.
fn main() {
    // Argument handling runs before ANY runtime, listener, or
    // telemetry exists: `taguru version` and friends must never start
    // a server (`--help` used to).
    let serve_args = cli::dispatch();
    // The config file becomes environment variables HERE, while the
    // process is still single-threaded (set_var's soundness condition)
    // and before init_telemetry reads RUST_LOG/OTEL_* — file values
    // must steer those too.
    if let Some(path) = &serve_args.config {
        cli::load_config(path);
    }
    serve();
}

#[tokio::main]
async fn serve() {
    // The subscriber must exist before anything can log — the
    // env_number warnings just below would otherwise be dropped
    // silently (tracing has no default subscriber and no buffering).
    let tracer_provider = init_telemetry();

    let config = registry::BootConfig::from_env();
    let flush_secs = resolve_flush_secs(env_number("TAGURU_FLUSH_SECS", 5));
    let max_body_bytes =
        resolve_body_bytes(env_number("TAGURU_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES));
    let mcp_max_result_bytes = resolve_mcp_max_result_bytes(env_number(
        "TAGURU_MCP_MAX_RESULT_BYTES",
        DEFAULT_MCP_MAX_RESULT_BYTES,
    ));
    let timeout_secs = resolve_timeout_secs(env_number("TAGURU_REQUEST_TIMEOUT_SECS", 30));
    let rate_per_minute = resolve_per_minute(
        "TAGURU_RATE_LIMIT_PER_MIN",
        env_number("TAGURU_RATE_LIMIT_PER_MIN", 0),
    );
    let rate_limiter = Arc::new(limits::RateLimiter::new(rate_per_minute));
    let rate_limit_disabled = rate_limiter.is_disabled();
    if !rate_limit_disabled {
        info!(per_minute = rate_per_minute, "per-key rate limit enabled");
    }
    // The in-flight ceiling defaults ON: without it nothing bounds
    // concurrent request growth under overload but the OS itself. 256
    // is far above any healthy load for a single node while still
    // being a real valve; 0 disables.
    let max_concurrent = env_number("TAGURU_MAX_CONCURRENT_REQUESTS", 256);
    if max_concurrent > 0 {
        info!(
            max_in_flight = max_concurrent,
            "in-flight request ceiling enabled (past it: 503 + Retry-After)"
        );
    }
    let max_concurrent_heavy = resolve_heavy_ops(env_number("TAGURU_MAX_CONCURRENT_HEAVY_OPS", 2));
    if max_concurrent_heavy > 0 {
        info!(
            max_concurrent_heavy,
            "heavy-operation ceiling enabled for vocabulary audits and context compactions"
        );
    }
    // Failed-auth attempts are throttled per source IP so the gate
    // cannot be brute-forced for free (default 10/min; 0 disables).
    let auth_fail_per_minute = resolve_per_minute(
        "TAGURU_AUTH_FAIL_LIMIT_PER_MIN",
        env_number("TAGURU_AUTH_FAIL_LIMIT_PER_MIN", 10),
    );
    let fail_limiter = Arc::new(limits::RateLimiter::new(auth_fail_per_minute));
    if !fail_limiter.is_disabled() {
        info!(
            per_minute = auth_fail_per_minute,
            "failed-auth throttle enabled (per source IP)"
        );
    }

    // Misconfigured credentials refuse to boot: a keyring that
    // silently dropped an entry would surface as an auth hole.
    let keyring = match auth::Keyring::parse(
        std::env::var("TAGURU_API_TOKEN").ok(),
        std::env::var("TAGURU_API_TOKENS").ok(),
    )
    .and_then(|mut keyring| {
        // Scopes are part of the same credential story: a grant that
        // silently failed to arm is an authorization hole.
        keyring.apply_scopes(std::env::var("TAGURU_KEY_SCOPES").ok().as_deref())?;
        Ok(keyring)
    }) {
        Ok(keyring) => Arc::new(keyring),
        Err(error) => {
            tracing::error!(%error, "refusing to start with broken credentials");
            std::process::exit(1);
        }
    };
    let auth_configured = !keyring.is_disabled();
    if auth_configured {
        info!(
            keys = keyring.key_count(),
            scoped = keyring.scoped_key_count(),
            "bearer auth enabled"
        );
    } else {
        warn!(
            "TAGURU_API_TOKEN(S) is not set: the API accepts UNAUTHENTICATED requests \
             (fine on localhost, never on an exposed address)"
        );
    }

    // TAGURU_PUBLIC_URL turns on OAuth for the remote MCP transport —
    // the URL doubles as issuer and audience, so OAuth without it
    // would mint tokens for an address nobody can verify.
    let oauth = match std::env::var("TAGURU_PUBLIC_URL") {
        Ok(url) if !url.trim().is_empty() => {
            if !auth_configured {
                tracing::error!(
                    "TAGURU_PUBLIC_URL is set but no API key is: the OAuth consent \
                     step delegates an EXISTING key, so configure TAGURU_API_TOKEN(S) first"
                );
                std::process::exit(1);
            }
            let oauth = Arc::new(oauth::Oauth::open(&url, &config.data_dir));
            info!(issuer = %oauth.public_url(), "oauth enabled for remote MCP");
            Some(oauth)
        }
        _ => None,
    };

    let embedder = embedding::HttpEmbeddings::from_env();
    if let Some(embedder) = &embedder {
        info!(model = embedder.model(), "semantic entry tier enabled");
        // The provider call runs in block_in_place, which the request
        // timeout cannot preempt — a budget under the provider ceiling
        // 408s spuriously whenever the provider is merely slow. And one
        // embed() makes up to MAX_EMBED_ATTEMPTS sequential attempts on
        // transient failures, so the worst-case hold is that many times
        // the per-attempt ceiling, not one. This trap used to live only
        // in a code comment; size it and say it at boot.
        let embed_timeout = env_number("TAGURU_EMBED_TIMEOUT_SECS", 60).max(1);
        let worst_case = embed_timeout.saturating_mul(embedding::MAX_EMBED_ATTEMPTS);
        if timeout_secs <= worst_case {
            warn!(
                request_timeout_secs = timeout_secs,
                embed_timeout_secs = embed_timeout,
                embed_worst_case_secs = worst_case,
                embed_attempts = embedding::MAX_EMBED_ATTEMPTS,
                "TAGURU_REQUEST_TIMEOUT_SECS is at or under the embedding provider's \
                 worst-case hold (TAGURU_EMBED_TIMEOUT_SECS × retries) — raise it above \
                 that, or slow provider calls will answer 408 after the work was \
                 already done"
            );
        }
    }
    let auto_embed = embedder.is_some()
        && std::env::var("TAGURU_EMBED_AUTO")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
    if auto_embed {
        info!("auto embedding refresh enabled (runs with each flush)");
    }
    let embedder = embedder.map(|provider| Arc::new(provider) as Arc<dyn EmbeddingProvider>);
    // /protocol tells connecting agents which optional tiers are live;
    // the model name is captured here because the provider itself moves
    // into the state.
    let protocol_trailer = api::protocol_trailer(
        embedder.as_ref().map(|provider| provider.model()),
        auto_embed,
    );
    // MCP initialize hands out exactly what GET /protocol serves —
    // both transports, one manual.
    let mcp_instructions = Arc::new(api::protocol_text(protocol_trailer.as_deref()));
    if let Some(floor) = config.semantic_floor {
        info!(floor, "semantic floor default recalibrated");
    }

    let state = match config.boot(embedder) {
        Ok(state) => state,
        // A held lock or an unreadable directory is an operator
        // problem, not a bug: one line, no backtrace.
        Err(error) => {
            tracing::error!(%error, "data directory is not usable");
            std::process::exit(1);
        }
    };

    spawn_flusher(state.clone(), flush_secs, auto_embed);

    let app = routes(
        protocol_trailer,
        limits::HeavyOpsLimiter::new(max_concurrent_heavy),
    )
    .with_state(state.clone());

    // POST /mcp speaks the MCP Streamable HTTP transport over these
    // same routes. The dispatch handle is captured BEFORE the outer
    // middleware stack goes on, so a tool call that already passed
    // auth, the timeout, and the body cap at /mcp is not re-charged
    // inside — one client request, one budget, one log line — but it
    // DOES carry its own authorization layer, so a scoped key's grant
    // is enforced on each dispatched tool call exactly as on the raw
    // API. ("No body cap inside" is explicit: an extractor that finds
    // no DefaultBodyLimit extension falls back to axum's hardcoded
    // 2 MiB default, which would silently cap dispatched tool calls
    // below the operator's TAGURU_MAX_BODY_BYTES.)
    let mcp_dispatch = app
        .clone()
        .layer(axum::extract::DefaultBodyLimit::disable())
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&keyring),
            auth::enforce_authorization,
        ));
    let app = app.route(
        "/mcp",
        post(
            move |deadline: axum::Extension<Deadline>,
                  key: Option<axum::Extension<auth::AuthKey>>,
                  body: axum::body::Bytes| {
                remote_mcp::serve(
                    mcp_dispatch.clone(),
                    Arc::clone(&mcp_instructions),
                    key.map(|extension| extension.0),
                    body,
                    mcp_max_result_bytes,
                    deadline.0,
                )
            },
        ),
    );
    // OAuth's discovery and grant endpoints ride the same stack (rate
    // limited, logged) but are exempted from bearer auth inside the
    // gate — they exist to create credentials.
    let app = match &oauth {
        Some(oauth) => app.merge(oauth_http::router(oauth_http::OauthState {
            oauth: Arc::clone(oauth),
            keyring: Arc::clone(&keyring),
            // The consent form shares the bearer gate's failed-auth
            // throttle: a wrong key there is a wrong credential too.
            fail_limiter: Arc::clone(&fail_limiter),
        })),
        None => app,
    };
    // Authorization wraps every OUTER route — the context API, the
    // /mcp endpoint itself, and the merged OAuth routes — now that all
    // are registered. `.layer()` only covers routes already on the
    // router, so this MUST come after the /mcp route and the oauth
    // merge above; layering it earlier (as this once did) silently
    // left both surfaces unchecked. Idempotent on auth-exempt routes:
    // enforce_authorization no-ops when no AuthKey extension is present.
    let app = app.layer(axum::middleware::from_fn_with_state(
        Arc::clone(&keyring),
        auth::enforce_authorization,
    ));
    let gate = Arc::new(auth::Gate {
        keyring,
        oauth: oauth.clone(),
        fail_limiter,
    });
    let app = app
        // Layers wrap only the routes registered above and nest by
        // call order — later .layer() calls sit outside earlier ones.
        // Body limit innermost, then the per-key rate gate (inside
        // auth: budget is spent only by authenticated keys), then
        // auth, then the timeout (its budget covers auth and body
        // handling too), and metrics outermost: every response — 401,
        // 408, 413, 429 — lands in the access log and the RED metrics.
        .layer(axum::extract::DefaultBodyLimit::max(max_body_bytes))
        .layer(axum::middleware::from_fn_with_state(
            rate_limiter,
            limits::enforce_rate_limit,
        ))
        .layer(axum::middleware::from_fn_with_state(
            gate,
            auth::require_bearer,
        ))
        .layer(axum::middleware::from_fn_with_state(
            Duration::from_secs(timeout_secs as u64),
            limits::enforce_timeout,
        ))
        // The in-flight ceiling sits outside the timeout: a shed must
        // be the cheapest possible response — no auth, no body read,
        // no handler — while still landing in the access log below.
        .layer(axum::middleware::from_fn_with_state(
            (max_concurrent, state.clone()),
            limits::enforce_concurrency,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            metrics::track_http,
        ));

    let addr = std::env::var("TAGURU_ADDR").unwrap_or_else(|_| "127.0.0.1:8248".to_string());
    let listener = match TcpListener::bind(&addr).await {
        Ok(listener) => listener,
        Err(bind_error) => {
            // The one boot failure an operator hits routinely deserves
            // a diagnosis, not a panic backtrace.
            tracing::error!(
                %addr,
                error = %bind_error,
                "cannot bind — is the port already in use, or the address not local? \
                 Set TAGURU_ADDR to change where the server listens"
            );
            std::process::exit(1);
        }
    };
    // Print the RESOLVED address: with port 0 the OS picks one, and
    // whoever spawned us (integration tests included) reads it here.
    // This stdout line is a contract — logging changes must not touch it.
    println!("listening on {}", listener.local_addr().unwrap());
    // Off-loopback with no per-key budget: one chatty client owns the
    // whole server. README says to turn the limit on when the server
    // leaves localhost — say it where the operator is looking.
    if !listener.local_addr().unwrap().ip().is_loopback() && rate_limit_disabled {
        warn!(
            "listening beyond loopback with TAGURU_RATE_LIMIT_PER_MIN off — set a \
             per-key budget whenever the server leaves localhost"
        );
    }
    // "ready" only after the socket exists: everything an operator
    // needs to sanity-check the deployment, in one line, at the moment
    // requests can actually arrive.
    info!(
        addr = %listener.local_addr().unwrap(),
        contexts = state.context_count(),
        data_dir = %config.data_dir.display(),
        cache_mib = config.cache_bytes / (1024 * 1024),
        flush_secs,
        wal_enabled = config.wal_enabled,
        wal_max_mib = config.wal_max_bytes / (1024 * 1024),
        passages_wal_max_mib = config.passages_wal_max_bytes / (1024 * 1024),
        max_body_mib = max_body_bytes / (1024 * 1024),
        mcp_max_result_mib = mcp_max_result_bytes / (1024 * 1024),
        timeout_secs,
        auth_enabled = auth_configured,
        "server ready",
    );
    // `with_connect_info` puts each caller's SocketAddr in the request
    // extensions — the per-source-IP throttles (failed-auth in the gate,
    // the anonymous request budget) read it from there. Without it those
    // fall back to a single shared bucket.
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .unwrap();

    // Nothing dirty may outlive the process: one final flush.
    tokio::task::block_in_place(|| state.flush_dirty());
    // Usage counters skip the WAL by design; the graceful stop is
    // where purely-read contexts get theirs onto disk.
    tokio::task::block_in_place(|| state.persist_usage());
    info!("flushed dirty contexts on shutdown");
    // The batch worker still owns the last spans; hand them to the
    // collector before the process ends.
    if let Some(provider) = tracer_provider
        && let Err(error) = tokio::task::block_in_place(|| provider.shutdown())
    {
        warn!(error = %error, "trace export flush on shutdown failed");
    }
}

/// Every HTTP route the server answers, mapped to its api:: handler —
/// the same table `POST /mcp` dispatches into. Off-axis errors answer
/// in the ApiError shape too: unknown paths, and known paths hit with
/// the wrong verb.
fn routes(
    protocol_trailer: Option<String>,
    heavy_ops_limiter: limits::HeavyOpsLimiter,
) -> Router<AppState> {
    let heavy_routes = Router::new()
        .route("/contexts/{name}/compact", post(api::compact_context))
        .route(
            "/contexts/{name}/vocabulary/audit",
            post(api::audit_vocabulary),
        )
        // `include_twins` runs the same CPU-bound pairwise scan
        // audit_vocabulary does, so this shares its gate rather than
        // the ordinary route table.
        .route("/contexts/{name}/drift/audit", post(api::audit_drift))
        .route_layer(axum::middleware::from_fn_with_state(
            heavy_ops_limiter,
            limits::enforce_heavy_ops,
        ));

    Router::new()
        .route("/health", get(metrics::health))
        .route("/live", get(metrics::live))
        .route("/metrics", get(metrics::render))
        .route(
            "/protocol",
            get(move || api::protocol(protocol_trailer.clone())),
        )
        .route("/flush", post(api::flush_all))
        .route("/maintenance/compact", post(api::maintenance_compact))
        .route("/import", post(api::import_batch))
        .route("/contexts", get(api::list_contexts))
        .route(
            "/contexts/{name}",
            get(api::get_context)
                .put(api::create_context)
                .patch(api::update_context)
                .delete(api::delete_context),
        )
        .route("/groups", get(api::list_groups))
        .route(
            "/groups/{name}",
            get(api::get_group)
                .put(api::create_group)
                .patch(api::update_group)
                .delete(api::delete_group),
        )
        .route("/groups/{name}/export", get(api::export_group))
        .route("/groups/{name}/rename", post(api::rename_group))
        // The cross-context searches: the per-context operation of the
        // same name, run across several contexts named in the body.
        .route("/recall", post(api::cross_recall))
        .route("/query", post(api::cross_query))
        .route("/sources/search", post(api::cross_search_passages))
        .route("/contexts/{name}/export", get(api::export_context))
        .route("/contexts/{name}/rename", post(api::rename_context))
        .route("/contexts/{name}/associations", post(api::add_associations))
        .route(
            "/contexts/{name}/associations/retract",
            post(api::retract_association),
        )
        .route("/contexts/{name}/recall", post(api::recall))
        .route("/contexts/{name}/query", post(api::query))
        .route("/contexts/{name}/describe", post(api::describe))
        .route("/contexts/{name}/explore", post(api::explore))
        .route("/contexts/{name}/activate", post(api::activate))
        .route("/contexts/{name}/resolve", post(api::resolve))
        .route(
            "/contexts/{name}/resolve/explain",
            post(api::explain_resolve),
        )
        .route("/contexts/{name}/resolve_label", post(api::resolve_label))
        .route(
            "/contexts/{name}/resolve_label/explain",
            post(api::explain_resolve_label),
        )
        .route("/contexts/{name}/labels", get(api::labels))
        .route(
            "/contexts/{name}/aliases",
            get(api::list_aliases)
                .post(api::add_aliases)
                .delete(api::remove_aliases),
        )
        .route(
            "/contexts/{name}/sources",
            get(api::list_sources).post(api::store_passages),
        )
        .route(
            "/contexts/{name}/sources/lookup",
            post(api::lookup_passages),
        )
        .route(
            "/contexts/{name}/sources/search",
            post(api::search_passages),
        )
        .route(
            "/contexts/{name}/sources/search/explain",
            post(api::explain_search_passages),
        )
        .route(
            "/contexts/{name}/sources/retract",
            post(api::retract_source),
        )
        .route("/contexts/{name}/citations", post(api::citation))
        .route(
            "/contexts/{name}/embeddings/refresh",
            post(api::refresh_embeddings),
        )
        .route(
            "/contexts/{name}/unreachable_from",
            post(api::unreachable_from),
        )
        .merge(heavy_routes)
        .fallback(api::unknown_path)
        .method_not_allowed_fallback(api::method_not_allowed)
        // Innermost on purpose: `routes()` is cloned for the /mcp
        // in-process dispatch too (see `mcp_dispatch` below), so a
        // panic inside a dispatched tool call is caught right here,
        // same as one from an ordinary HTTP handler — neither takes
        // down the connection task out from under the metrics/
        // access-log/trace middleware layered on further out.
        .layer(CatchPanicLayer::custom(api::panic_response))
}

/// The periodic flusher: every `flush_secs`, persist what is dirty —
/// and, when auto embedding is on, refresh what just flushed. Best
/// effort: a failed refresh is retried the next time a write dirties
/// the context (the gloss diff is idempotent), and the manual endpoint
/// always remains.
fn spawn_flusher(state: AppState, flush_secs: usize, auto_embed: bool) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(flush_secs as u64));
        ticker.tick().await; // the first tick fires immediately; skip it
        loop {
            ticker.tick().await;
            // A maintenance sweep is already rewriting images under its
            // own lock discipline; skip this tick rather than race it.
            // Safe either way (the entry's `image_generation` guards a
            // stale republish), but skipping avoids flushing an entry
            // mid-compaction for nothing.
            if state.metrics().maintenance_active() {
                continue;
            }
            // Serialization and fsyncs are blocking work; keep the
            // async workers free while images land on disk.
            let flushed = tokio::task::block_in_place(|| state.flush_dirty());
            // Opt-in: a flushed context just changed on disk, so its
            // glosses may have changed too — re-embed the difference.
            if auto_embed && !flushed.is_empty() {
                tokio::task::block_in_place(|| {
                    for name in &flushed {
                        match state.refresh_embeddings(name, Deadline::unbounded()) {
                            None | Some(Ok((0, _))) => {}
                            Some(Ok((embedded, _))) => {
                                info!(context = %name, embedded, "auto-embedded glosses");
                            }
                            Some(Err(error)) => {
                                warn!(context = %name, error, "auto embedding refresh failed");
                            }
                        }
                    }
                });
            }
            // Passages ride their own dirty flag: storing text never
            // marks the GRAPH dirty, so the flush list alone would
            // miss passage-only ingest.
            if auto_embed && state.passage_embedding_enabled() {
                let stale = state.passage_embed_dirty_names();
                if !stale.is_empty() {
                    tokio::task::block_in_place(|| {
                        for name in &stale {
                            match state.refresh_passage_embeddings(name, Deadline::unbounded()) {
                                None => {}
                                Some(Ok(outcome)) if outcome.embedded == 0 => {}
                                Some(Ok(outcome)) => {
                                    info!(
                                        context = %name,
                                        embedded = outcome.embedded,
                                        "auto-embedded paragraphs"
                                    );
                                }
                                Some(Err(error)) => {
                                    warn!(
                                        context = %name,
                                        error,
                                        "auto passage embedding failed"
                                    );
                                }
                            }
                        }
                    });
                }
            }
        }
    });
}

/// Installs the global tracing subscriber: `RUST_LOG` filtering
/// (default `info`), human-readable or JSON lines per
/// `TAGURU_LOG_FORMAT`, written to stderr — stdout stays reserved for
/// the bootstrap contract lines tests parse. When an OTLP endpoint is
/// configured, an export layer rides alongside; `RUST_LOG` shapes only
/// the stderr log, never what is exported.
fn init_telemetry() -> Option<opentelemetry_sdk::trace::SdkTracerProvider> {
    use tracing_subscriber::Layer as _;
    use tracing_subscriber::layer::SubscriberExt as _;
    use tracing_subscriber::util::SubscriberInitExt as _;

    let filter = tracing_subscriber::EnvFilter::try_from_env("RUST_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let json = std::env::var("TAGURU_LOG_FORMAT").is_ok_and(|v| v.eq_ignore_ascii_case("json"));

    let (provider, exporter_error) = trace::provider();
    let otel_layer = provider.as_ref().map(|provider| {
        use opentelemetry::trace::TracerProvider as _;
        // INFO keeps the export layer from re-enabling debug/trace
        // callsites the stderr filter would otherwise leave off.
        tracing_opentelemetry::layer()
            .with_tracer(provider.tracer("taguru"))
            .with_filter(tracing::level_filters::LevelFilter::INFO)
    });

    let registry = tracing_subscriber::registry().with(otel_layer);
    let stderr_layer = tracing_subscriber::fmt::layer().with_writer(std::io::stderr);
    if json {
        registry
            .with(stderr_layer.json().with_filter(filter))
            .init();
    } else {
        registry.with(stderr_layer.with_filter(filter)).init();
    }

    // Deferred from trace::provider(): logging works only now.
    if let Some(error) = exporter_error {
        warn!(
            error,
            "span export disabled: the OTLP exporter failed to build"
        );
    }
    if provider.is_some() {
        info!("OTLP span export enabled");
    }
    provider
}

/// Resolves on the first SIGINT (Ctrl+C) or SIGTERM — the signal that
/// starts axum's graceful drain — and, before returning, arms a one-shot
/// task that force-exits on the SECOND such signal. The drain waits on
/// in-flight requests; if one hangs, an impatient operator's second
/// Ctrl+C would otherwise be swallowed (tokio's installed handler
/// overrides the default terminate disposition, and nothing awaits it),
/// leaving SIGKILL as the only way out. The re-armed listener restores
/// the familiar escape hatch: press again to quit now. A forced exit
/// skips the final image flush, but every write is already in the WAL and
/// replays on the next boot — only unwritten usage counters are lost,
/// the price the operator chose by not waiting for the clean stop.
///
/// The signal streams are registered ONCE and the second wait reuses
/// them. Re-registering per wait (the old shape) dropped the SIGTERM
/// stream when the first signal fired and installed a new one only after
/// the spawn — a window in which a second signal would land on no handler
/// and be lost, defeating the very escape hatch this arms.
async fn shutdown_signal() {
    let mut signals = TerminateSignals::install();
    signals.recv().await;
    tokio::spawn(async move {
        signals.recv().await;
        warn!("second shutdown signal received — forcing an immediate exit");
        // 128 + SIGINT(2), the shell convention for signal-terminated.
        std::process::exit(130);
    });
}

/// The SIGINT/SIGTERM streams axum's graceful drain listens on, held for
/// the whole shutdown sequence so [`recv`](Self::recv) can be awaited
/// again for a second signal without a re-registration gap.
struct TerminateSignals {
    #[cfg(unix)]
    interrupt: Option<tokio::signal::unix::Signal>,
    #[cfg(unix)]
    terminate: Option<tokio::signal::unix::Signal>,
}

impl TerminateSignals {
    /// Registers the terminate signals once. A stream that fails to
    /// register degrades to one that never fires — the other signal (and
    /// a second Ctrl+C) still works — rather than crashing the server as
    /// it starts serving.
    fn install() -> Self {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            let register = |kind: SignalKind| match signal(kind) {
                Ok(stream) => Some(stream),
                Err(error) => {
                    warn!(%error, "could not register a shutdown signal handler");
                    None
                }
            };
            Self {
                interrupt: register(SignalKind::interrupt()),
                terminate: register(SignalKind::terminate()),
            }
        }
        #[cfg(not(unix))]
        {
            Self {}
        }
    }

    /// Completes on the next SIGINT or SIGTERM (Ctrl+C on non-Unix).
    async fn recv(&mut self) {
        #[cfg(unix)]
        {
            let Self {
                interrupt,
                terminate,
            } = self;
            let interrupt = wait_signal(interrupt.as_mut());
            let terminate = wait_signal(terminate.as_mut());
            tokio::select! {
                () = interrupt => {}
                () = terminate => {}
            }
        }
        #[cfg(not(unix))]
        {
            // No persistent Unix stream to drop here; ctrl_c()'s handler
            // is a global that stays installed, so re-awaiting is safe.
            let _ = tokio::signal::ctrl_c().await;
        }
    }
}

/// Awaits an optional signal stream, or pends forever when its
/// registration failed so the surrounding select falls to the other arm.
#[cfg(unix)]
async fn wait_signal(stream: Option<&mut tokio::signal::unix::Signal>) {
    match stream {
        Some(stream) => {
            stream.recv().await;
        }
        None => std::future::pending().await,
    }
}

pub(crate) fn env_number(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(value) => value.parse().unwrap_or_else(|_| {
            warn!("ignoring {key}={value}: not a number; using {default}");
            default
        }),
        Err(_) => default,
    }
}

/// An optional 0..=1 fraction from the environment; anything else
/// (including NaN) is ignored with a warning, keeping the built-in
/// calibration.
pub(crate) fn env_floor(key: &str) -> Option<f32> {
    let value = std::env::var(key).ok()?;
    match value.parse::<f32>() {
        Ok(floor) if (0.0..=1.0).contains(&floor) => Some(floor),
        _ => {
            warn!("ignoring {key}={value}: not a number between 0 and 1");
            None
        }
    }
}

/// `tokio::time::interval` panics on a zero period — with
/// `TAGURU_FLUSH_SECS=0` that panic fires inside the spawned flusher
/// task, not the main thread, so the server keeps listening and
/// answering requests while dirty contexts silently stop persisting
/// forever. Floor to 1 instead, loudly, the same "never silent" rule
/// `env_number` already applies to unparseable input.
pub(crate) fn resolve_flush_secs(requested: usize) -> usize {
    if requested == 0 {
        warn!("TAGURU_FLUSH_SECS=0 would never fire (and would panic the flusher task); using 1");
        1
    } else {
        requested
    }
}

/// The limiter holds its budget in a u32; a bigger env value would be
/// silently clamped inside the constructor while the boot line logged
/// the raw number — the logged limit and the enforced limit must be
/// the same number, and the clamp must be loud like every other
/// out-of-range env here.
pub(crate) fn resolve_per_minute(name: &str, requested: usize) -> u32 {
    u32::try_from(requested).unwrap_or_else(|_| {
        warn!(
            "{name}={requested} exceeds the limiter's ceiling; clamping to {}",
            u32::MAX
        );
        u32::MAX
    })
}

/// Tokio reserves a few high bits in its semaphore permit counter and
/// panics when constructed above `MAX_PERMITS`. An operator-provided usize
/// must therefore be clamped before it reaches `HeavyOpsLimiter::new`.
pub(crate) fn resolve_heavy_ops(requested: usize) -> usize {
    if requested > tokio::sync::Semaphore::MAX_PERMITS {
        warn!(
            "TAGURU_MAX_CONCURRENT_HEAVY_OPS={requested} exceeds the semaphore's ceiling; \
             clamping to {}",
            tokio::sync::Semaphore::MAX_PERMITS
        );
        tokio::sync::Semaphore::MAX_PERMITS
    } else {
        requested
    }
}

const DEFAULT_MAX_BODY_BYTES: usize = 8 * 1024 * 1024;

/// `TAGURU_REQUEST_TIMEOUT_SECS=0` reads as "no timeout" — its
/// neighbor in the usage text documents 0 = off — but would wrap every
/// request in a zero-length budget: anything that doesn't resolve on
/// its very first poll (most writes, under real network latency)
/// answers 408 while the server boots and logs as if healthy. Floor to
/// 1, loudly; a huge value is how to effectively disable the budget.
pub(crate) fn resolve_timeout_secs(requested: usize) -> usize {
    if requested == 0 {
        warn!(
            "TAGURU_REQUEST_TIMEOUT_SECS=0 would 408 every request; using 1 \
             (set a large value to effectively disable the budget)"
        );
        1
    } else {
        requested
    }
}

/// The same trap for `TAGURU_MAX_BODY_BYTES=0`: by analogy with the
/// WAL ceilings' "0 = off" it reads as "no cap", but a zero
/// DefaultBodyLimit refuses every request that carries a body — all
/// writes 413, no startup complaint. And a truly uncapped body would
/// hand an allocation lever to whoever can reach the port, so 0 gets
/// the default back instead, loudly.
pub(crate) fn resolve_body_bytes(requested: usize) -> usize {
    if requested == 0 {
        warn!(
            "TAGURU_MAX_BODY_BYTES=0 would refuse every write; using the 8 MiB default \
             (set an explicit larger cap for bigger bodies)"
        );
        DEFAULT_MAX_BODY_BYTES
    } else {
        requested
    }
}

const DEFAULT_MCP_MAX_RESULT_BYTES: usize = 8 * 1024 * 1024;

/// Below this a cap is almost certainly a fat-fingered value (a byte
/// count typed where a kibibyte count was meant) rather than a
/// deliberate small budget — still honored, just flagged.
const MCP_MAX_RESULT_BYTES_WARN_FLOOR: usize = 64 * 1024;

/// The same "0 = unbounded is a trap" reasoning as
/// [`resolve_body_bytes`]: an uncapped `POST /mcp` tool result hands
/// an allocation lever to whoever can reach a tool that returns a lot
/// of data (`export_context` on a large context, chiefly), so 0
/// floors to the default instead of disabling the cap. Anything
/// nonzero but under 64 KiB is small enough to likely be a mistake —
/// obeyed regardless, just logged.
pub(crate) fn resolve_mcp_max_result_bytes(requested: usize) -> usize {
    if requested == 0 {
        warn!(
            "TAGURU_MCP_MAX_RESULT_BYTES=0 would refuse every tool result; using the 8 MiB \
             default (set an explicit larger cap for bigger results)"
        );
        return DEFAULT_MCP_MAX_RESULT_BYTES;
    }
    if requested < MCP_MAX_RESULT_BYTES_WARN_FLOOR {
        warn!(
            requested,
            "TAGURU_MCP_MAX_RESULT_BYTES is under 64 KiB — most tool results will fit, but \
             double check this wasn't meant to be a larger unit"
        );
    }
    requested
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flush_secs_zero_is_floored_to_one_instead_of_panicking_the_flusher() {
        assert_eq!(resolve_flush_secs(0), 1);
        assert_eq!(resolve_flush_secs(5), 5);
    }

    #[test]
    fn zero_timeout_and_body_cap_are_floored_loudly_not_obeyed() {
        assert_eq!(resolve_timeout_secs(0), 1);
        assert_eq!(resolve_timeout_secs(30), 30);
        assert_eq!(resolve_body_bytes(0), DEFAULT_MAX_BODY_BYTES);
        assert_eq!(resolve_body_bytes(1024), 1024);
    }

    #[test]
    fn zero_mcp_result_cap_is_floored_but_a_small_one_is_still_obeyed() {
        assert_eq!(
            resolve_mcp_max_result_bytes(0),
            DEFAULT_MCP_MAX_RESULT_BYTES
        );
        assert_eq!(resolve_mcp_max_result_bytes(1024), 1024);
        assert_eq!(
            resolve_mcp_max_result_bytes(DEFAULT_MCP_MAX_RESULT_BYTES),
            DEFAULT_MCP_MAX_RESULT_BYTES
        );
    }

    #[test]
    fn per_minute_rates_clamp_at_the_limiter_ceiling() {
        assert_eq!(resolve_per_minute("X", 0), 0);
        assert_eq!(resolve_per_minute("X", 600), 600);
        assert_eq!(resolve_per_minute("X", u32::MAX as usize + 1), u32::MAX);
    }

    #[test]
    fn heavy_operation_limits_clamp_before_semaphore_construction() {
        assert_eq!(resolve_heavy_ops(0), 0);
        assert_eq!(resolve_heavy_ops(2), 2);
        assert_eq!(
            resolve_heavy_ops(usize::MAX),
            tokio::sync::Semaphore::MAX_PERMITS
        );
    }
}
