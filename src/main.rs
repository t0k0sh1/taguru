mod api;
mod auth;
mod cli;
mod embedding;
mod limits;
mod metrics;
mod registry;
mod trace;
mod wal;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::routing::{get, post};
use embedding::EmbeddingProvider;
use registry::AppState;
use tokio::net::TcpListener;
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

    let data_dir =
        PathBuf::from(std::env::var("TAGURU_DATA_DIR").unwrap_or_else(|_| "data".into()));
    let cache_bytes = env_number("TAGURU_CACHE_BYTES", 512 * 1024 * 1024);
    let flush_secs = env_number("TAGURU_FLUSH_SECS", 5);
    let max_body_bytes = env_number("TAGURU_MAX_BODY_BYTES", 8 * 1024 * 1024);
    let timeout_secs = env_number("TAGURU_REQUEST_TIMEOUT_SECS", 30);

    let api_token = Arc::new(std::env::var("TAGURU_API_TOKEN").ok());
    let auth_configured = api_token.is_some();
    if api_token.is_none() {
        warn!(
            "TAGURU_API_TOKEN is not set: the API accepts UNAUTHENTICATED requests \
             (fine on localhost, never on an exposed address)"
        );
    }

    let embedder = embedding::HttpEmbeddings::from_env();
    if let Some(embedder) = &embedder {
        info!(model = embedder.model(), "semantic entry tier enabled");
    }
    let auto_embed = embedder.is_some()
        && std::env::var("TAGURU_EMBED_AUTO")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
    if auto_embed {
        info!("auto embedding refresh enabled (runs with each flush)");
    }
    let embedder = embedder.map(|provider| Arc::new(provider) as Arc<dyn EmbeddingProvider>);

    // The WAL closes the flush-interval loss window; opting out
    // (TAGURU_WAL=0) restores the old posture for benchmarks or
    // explicit risk acceptance.
    let wal_enabled = std::env::var("TAGURU_WAL")
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(true);
    // Backstop for a persistently failing flush: past this, writes are
    // refused rather than growing the log without bound (0 = no cap).
    let wal_max_bytes = env_number("TAGURU_WAL_MAX_BYTES", registry::DEFAULT_WAL_MAX_BYTES);

    let state = AppState::boot_with(
        data_dir.clone(),
        cache_bytes,
        embedder,
        wal_enabled,
        wal_max_bytes,
    )
    .expect("data directory must be usable");

    let flusher = state.clone();
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(flush_secs as u64));
        ticker.tick().await; // the first tick fires immediately; skip it
        loop {
            ticker.tick().await;
            // Serialization and fsyncs are blocking work; keep the
            // async workers free while images land on disk.
            let flushed = tokio::task::block_in_place(|| flusher.flush_dirty());
            // Opt-in: a flushed context just changed on disk, so its
            // glosses may have changed too — re-embed the difference.
            // Best effort: a failed refresh is retried the next time a
            // write dirties the context (the gloss diff is idempotent),
            // and the manual endpoint always remains.
            if auto_embed && !flushed.is_empty() {
                tokio::task::block_in_place(|| {
                    for name in &flushed {
                        match flusher.refresh_embeddings(name) {
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
        }
    });

    let app = Router::new()
        .route("/health", get(metrics::health))
        .route("/metrics", get(metrics::render))
        .route("/protocol", get(api::protocol))
        .route("/flush", post(api::flush_all))
        .route("/contexts", get(api::list_contexts))
        .route(
            "/contexts/{name}",
            get(api::get_context)
                .put(api::create_context)
                .patch(api::update_context)
                .delete(api::delete_context),
        )
        .route("/contexts/{name}/associations", post(api::add_associations))
        .route("/contexts/{name}/recall", post(api::recall))
        .route("/contexts/{name}/query", post(api::query))
        .route("/contexts/{name}/describe", post(api::describe))
        .route("/contexts/{name}/explore", post(api::explore))
        .route("/contexts/{name}/activate", post(api::activate))
        .route("/contexts/{name}/resolve", post(api::resolve))
        .route("/contexts/{name}/resolve_label", post(api::resolve_label))
        .route("/contexts/{name}/labels", get(api::labels))
        .route(
            "/contexts/{name}/aliases",
            get(api::list_aliases).post(api::add_aliases),
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
            "/contexts/{name}/sources/retract",
            post(api::retract_source),
        )
        .route(
            "/contexts/{name}/embeddings/refresh",
            post(api::refresh_embeddings),
        )
        .route(
            "/contexts/{name}/unreachable_from",
            post(api::unreachable_from),
        )
        .route(
            "/contexts/{name}/vocabulary/audit",
            post(api::audit_vocabulary),
        )
        // Off-axis errors answer in the ApiError shape too: unknown
        // paths, and known paths hit with the wrong verb.
        .fallback(api::unknown_path)
        .method_not_allowed_fallback(api::method_not_allowed)
        // Layers wrap only the routes registered above and nest by
        // call order — later .layer() calls sit outside earlier ones.
        // Body limit innermost, then auth, then the timeout (its
        // budget covers auth and body handling too), and metrics
        // outermost: every response — 401, 408, 413 — lands in the
        // access log and the RED metrics.
        .layer(axum::extract::DefaultBodyLimit::max(max_body_bytes))
        .layer(axum::middleware::from_fn_with_state(
            api_token,
            auth::require_bearer,
        ))
        .layer(axum::middleware::from_fn_with_state(
            Duration::from_secs(timeout_secs as u64),
            limits::enforce_timeout,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            metrics::track_http,
        ))
        .with_state(state.clone());

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
    // "ready" only after the socket exists: everything an operator
    // needs to sanity-check the deployment, in one line, at the moment
    // requests can actually arrive.
    info!(
        addr = %listener.local_addr().unwrap(),
        contexts = state.context_count(),
        data_dir = %data_dir.display(),
        cache_mib = cache_bytes / (1024 * 1024),
        flush_secs,
        wal_enabled,
        wal_max_mib = wal_max_bytes / (1024 * 1024),
        max_body_mib = max_body_bytes / (1024 * 1024),
        timeout_secs,
        auth_enabled = auth_configured,
        "server ready",
    );
    axum::serve(listener, app)
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

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut stream) => {
                stream.recv().await;
            }
            Err(_) => std::future::pending().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
}

fn env_number(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(value) => value.parse().unwrap_or_else(|_| {
            warn!("ignoring {key}={value}: not a number; using {default}");
            default
        }),
        Err(_) => default,
    }
}
