mod api;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use associative_rag::context::Context;
use axum::Router;
use axum::routing::{get, post, put};
use tokio::net::TcpListener;

/// The context registry. The outer lock guards only the name → context
/// map and is held just long enough to look up, insert, or remove an
/// entry; every `Context` sits behind its own lock. A handler that works
/// on one context clones its `Arc` and releases the registry immediately,
/// so a slow retrieval or write on one context never blocks the others —
/// and a panic while holding a context's lock poisons only that context,
/// not the whole registry.
pub type SharedContexts = Arc<RwLock<HashMap<String, Arc<RwLock<Context>>>>>;

#[tokio::main]
async fn main() {
    let contexts: SharedContexts = Arc::new(RwLock::new(HashMap::new()));

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route(
            "/contexts/{name}",
            put(api::create_context).delete(api::delete_context),
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
        .with_state(contexts);

    let addr = "127.0.0.1:3000";
    let listener = TcpListener::bind(addr).await.unwrap();
    println!("listening on {addr}");
    axum::serve(listener, app).await.unwrap();
}
