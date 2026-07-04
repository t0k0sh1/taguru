mod api;
mod registry;

use std::path::PathBuf;
use std::time::Duration;

use axum::Router;
use axum::routing::{get, post, put};
use registry::AppState;
use tokio::net::TcpListener;

/// Configuration comes from the environment:
/// - `ARAG_DATA_DIR`: where context images and sidecars live (default
///   `data`). Disk is the source of truth; memory is a cache over it.
/// - `ARAG_CACHE_BYTES`: resident budget for unpinned loaded contexts
///   (default 512 MiB). Past it, least-recently-used contexts are
///   flushed and dropped; pinned contexts live outside the budget.
/// - `ARAG_FLUSH_SECS`: how often dirty contexts are persisted (default
///   5). This is the crash-loss window; writes also persist on eviction
///   and on graceful shutdown (SIGINT/SIGTERM).
#[tokio::main]
async fn main() {
    let data_dir = PathBuf::from(std::env::var("ARAG_DATA_DIR").unwrap_or_else(|_| "data".into()));
    let cache_bytes = env_number("ARAG_CACHE_BYTES", 512 * 1024 * 1024);
    let flush_secs = env_number("ARAG_FLUSH_SECS", 5);

    let state =
        AppState::boot(data_dir.clone(), cache_bytes).expect("data directory must be usable");
    println!(
        "{} context(s) registered from {} (cache budget {} MiB, flush every {flush_secs}s)",
        state.context_count(),
        data_dir.display(),
        cache_bytes / (1024 * 1024),
    );

    let flusher = state.clone();
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(flush_secs as u64));
        ticker.tick().await; // the first tick fires immediately; skip it
        loop {
            ticker.tick().await;
            flusher.flush_dirty();
        }
    });

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/contexts", get(api::list_contexts))
        .route(
            "/contexts/{name}",
            put(api::create_context)
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
            "/contexts/{name}/unreachable_from",
            post(api::unreachable_from),
        )
        .with_state(state.clone());

    let addr = "127.0.0.1:3000";
    let listener = TcpListener::bind(addr).await.unwrap();
    println!("listening on {addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();

    // Nothing dirty may outlive the process: one final flush.
    state.flush_dirty();
    println!("flushed dirty contexts on shutdown");
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
            eprintln!("ignoring {key}={value}: not a number; using {default}");
            default
        }),
        Err(_) => default,
    }
}
