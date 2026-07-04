mod api;
mod registry;

use std::path::PathBuf;

use axum::Router;
use axum::routing::{get, post, put};
use registry::AppState;
use tokio::net::TcpListener;

/// Configuration comes from the environment:
/// - `ARAG_DATA_DIR`: where context images and sidecars live (default
///   `data`). Disk is the source of truth; the registry is loaded from
///   here on boot.
#[tokio::main]
async fn main() {
    let data_dir = PathBuf::from(std::env::var("ARAG_DATA_DIR").unwrap_or_else(|_| "data".into()));
    let state = AppState::boot(data_dir.clone()).expect("data directory must be usable");
    println!(
        "loaded {} context(s) from {}",
        state.context_count(),
        data_dir.display()
    );

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
        .route("/contexts/{name}/explore", post(api::explore))
        .route("/contexts/{name}/activate", post(api::activate))
        .route("/contexts/{name}/resolve", post(api::resolve))
        .route("/contexts/{name}/resolve_label", post(api::resolve_label))
        .route("/contexts/{name}/labels", get(api::labels))
        .route(
            "/contexts/{name}/unreachable_from",
            post(api::unreachable_from),
        )
        .with_state(state);

    let addr = "127.0.0.1:3000";
    let listener = TcpListener::bind(addr).await.unwrap();
    println!("listening on {addr}");
    axum::serve(listener, app).await.unwrap();
}
