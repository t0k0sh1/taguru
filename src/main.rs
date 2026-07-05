mod api;
mod auth;
mod embedding;
mod limits;
mod metrics;
mod registry;
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

/// Configuration comes from the environment:
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
#[tokio::main]
async fn main() {
    // The subscriber must exist before anything can log — the
    // env_number warnings just below would otherwise be dropped
    // silently (tracing has no default subscriber and no buffering).
    init_tracing();

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
    info!("flushed dirty contexts on shutdown");
}

/// Installs the global tracing subscriber: `RUST_LOG` filtering
/// (default `info`), human-readable or JSON lines per
/// `TAGURU_LOG_FORMAT`, written to stderr — stdout stays reserved for
/// the bootstrap contract lines tests parse.
fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_env("RUST_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let json = std::env::var("TAGURU_LOG_FORMAT").is_ok_and(|v| v.eq_ignore_ascii_case("json"));
    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr);
    if json {
        builder.json().init();
    } else {
        builder.init();
    }
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
