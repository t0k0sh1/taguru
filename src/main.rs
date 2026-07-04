use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use associative_rag::context::Context;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;

#[derive(Debug, Deserialize)]
struct CreateContextRequest {}

/// The context registry. The outer lock guards only the name → context
/// map and is held just long enough to look up, insert, or remove an
/// entry; every `Context` sits behind its own lock. A handler that works
/// on one context clones its `Arc` and releases the registry immediately,
/// so a slow retrieval or write on one context never blocks the others —
/// and a panic while holding a context's lock poisons only that context,
/// not the whole registry.
type SharedContexts = Arc<RwLock<HashMap<String, Arc<RwLock<Context>>>>>;

#[derive(Serialize)]
struct ApiResponse<T> {
    result: T,
    status: &'static str,
    time: f64,
}

impl<T> ApiResponse<T> {
    fn ok(result: T, started_at: Instant) -> Self {
        Self {
            result,
            status: "ok",
            time: started_at.elapsed().as_secs_f64(),
        }
    }
}

#[derive(Serialize)]
struct ApiError {
    status: &'static str,
    error: String,
    time: f64,
}

impl ApiError {
    fn new(error: impl Into<String>, started_at: Instant) -> Self {
        Self {
            status: "error",
            error: error.into(),
            time: started_at.elapsed().as_secs_f64(),
        }
    }
}

async fn create_context(
    State(contexts): State<SharedContexts>,
    Path(name): Path<String>,
    Json(_body): Json<CreateContextRequest>,
) -> impl IntoResponse {
    let started_at = Instant::now();
    let mut contexts = contexts.write().unwrap();

    if contexts.contains_key(&name) {
        let body = ApiError::new(format!("context '{name}' already exists"), started_at);
        return (StatusCode::CONFLICT, Json(body)).into_response();
    }

    contexts.insert(name, Arc::new(RwLock::new(Context::default())));
    (StatusCode::OK, Json(ApiResponse::ok(true, started_at))).into_response()
}

async fn delete_context(
    State(contexts): State<SharedContexts>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let started_at = Instant::now();
    let mut contexts = contexts.write().unwrap();

    if contexts.remove(&name).is_none() {
        let body = ApiError::new(format!("context '{name}' not found"), started_at);
        return (StatusCode::NOT_FOUND, Json(body)).into_response();
    }

    (StatusCode::OK, Json(ApiResponse::ok(true, started_at))).into_response()
}

#[tokio::main]
async fn main() {
    let contexts: SharedContexts = Arc::new(RwLock::new(HashMap::new()));

    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .route(
            "/contexts/{name}",
            put(create_context).delete(delete_context),
        )
        .with_state(contexts);

    let addr = "127.0.0.1:3000";
    let listener = TcpListener::bind(addr).await.unwrap();
    println!("listening on {addr}");
    axum::serve(listener, app).await.unwrap();
}
