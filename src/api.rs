//! HTTP surface of the retrieval service: thin JSON adapters, one per
//! `Context` operation. The server adds transport, naming, and lifecycle
//! around the library — never retrieval semantics of its own — so each
//! handler is a lock, a library call, and a serialized reply.

use std::time::Instant;

use associative_rag::context::Context;
use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use crate::registry::{AppState, ContextMeta, CreateError};

#[derive(Serialize)]
pub struct ApiResponse<T> {
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
pub struct ApiError {
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

fn ok<T: Serialize>(result: T, started_at: Instant) -> Response {
    (StatusCode::OK, Json(ApiResponse::ok(result, started_at))).into_response()
}

fn error(status: StatusCode, message: impl Into<String>, started_at: Instant) -> Response {
    (status, Json(ApiError::new(message, started_at))).into_response()
}

fn not_found(name: &str, started_at: Instant) -> Response {
    error(
        StatusCode::NOT_FOUND,
        format!("context '{name}' not found"),
        started_at,
    )
}

/// The routing directory, skills-style: name, prose description, and
/// stats for every context, so an LLM client can decide where to search
/// (and where to ingest) without the server owning that judgement.
pub async fn list_contexts(State(state): State<AppState>) -> Response {
    let started_at = Instant::now();
    ok(state.directory(), started_at)
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct CreateContextRequest {
    pub description: String,
    pub pinned: bool,
}

pub async fn create_context(
    State(state): State<AppState>,
    Path(name): Path<String>,
    body: Option<Json<CreateContextRequest>>,
) -> Response {
    let started_at = Instant::now();
    let Json(request) = body.unwrap_or_default();
    let meta = ContextMeta {
        description: request.description,
        pinned: request.pinned,
    };
    match state.create(&name, meta) {
        Ok(()) => ok(true, started_at),
        Err(CreateError::AlreadyExists) => error(
            StatusCode::CONFLICT,
            format!("context '{name}' already exists"),
            started_at,
        ),
        Err(CreateError::Io(io_error)) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("context '{name}' could not be persisted: {io_error}"),
            started_at,
        ),
    }
}

#[derive(Debug, Deserialize)]
pub struct UpdateContextRequest {
    pub description: Option<String>,
    pub pinned: Option<bool>,
}

pub async fn update_context(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<UpdateContextRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.update_meta(&name, request.description, request.pinned) {
        None => not_found(&name, started_at),
        Some(Ok(meta)) => ok(meta, started_at),
        Some(Err(io_error)) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("metadata updated in memory but not persisted: {io_error}"),
            started_at,
        ),
    }
}

pub async fn delete_context(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let started_at = Instant::now();
    match state.delete(&name) {
        None => not_found(&name, started_at),
        Some(Ok(())) => ok(true, started_at),
        Some(Err(io_error)) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("context '{name}' removed but its files were not: {io_error}"),
            started_at,
        ),
    }
}

/// One assertion in a batch: the arguments of `associate` /
/// `associate_from` as JSON fields. A batch is the natural write unit —
/// one document's extracted facts arrive as one request, one lock
/// acquisition, one persisted image.
#[derive(Debug, Deserialize)]
pub struct AssociationInput {
    pub subject: String,
    pub label: String,
    pub object: String,
    pub weight: f64,
    pub source: Option<String>,
}

pub async fn add_associations(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(associations): Json<Vec<AssociationInput>>,
) -> Response {
    let started_at = Instant::now();
    let outcome = state.write_context(&name, |context| {
        for (index, input) in associations.iter().enumerate() {
            let result = match &input.source {
                Some(source) => context.associate_from(
                    input.subject.as_str(),
                    input.label.as_str(),
                    input.object.as_str(),
                    input.weight,
                    source.as_str(),
                ),
                None => context.associate(
                    input.subject.as_str(),
                    input.label.as_str(),
                    input.object.as_str(),
                    input.weight,
                ),
            };
            if let Err(full) = result {
                return Err((index, full));
            }
        }
        Ok(associations.len())
    });

    let Some((applied, persisted)) = outcome else {
        return not_found(&name, started_at);
    };
    match (applied, persisted) {
        (Ok(count), Ok(())) => ok(count, started_at),
        // The batch landed in memory but the image write failed: the
        // client must know the data is not yet durable.
        (Ok(count), Err(io_error)) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("applied {count} associations in memory, but persisting failed: {io_error}"),
            started_at,
        ),
        // Items before the failing one are applied (each item is
        // all-or-nothing in the library); report how far the batch got.
        (Err((index, full)), persisted) => {
            let mut message = format!(
                "applied {index} of {} associations, then: {full}",
                associations.len()
            );
            if let Err(io_error) = persisted {
                message.push_str(&format!("; persisting also failed: {io_error}"));
            }
            error(StatusCode::INSUFFICIENT_STORAGE, message, started_at)
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct RecallRequest {
    pub cue: String,
}

pub async fn recall(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<RecallRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| context.recall(&request.cue)) {
        Some(result) => ok(result, started_at),
        None => not_found(&name, started_at),
    }
}

#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub subject: Option<String>,
    pub label: Option<String>,
    pub object: Option<String>,
}

pub async fn query(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<QueryRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| {
        context.query(
            request.subject.as_deref(),
            request.label.as_deref(),
            request.object.as_deref(),
        )
    }) {
        Some(result) => ok(result, started_at),
        None => not_found(&name, started_at),
    }
}

#[derive(Debug, Deserialize)]
pub struct ExploreRequest {
    pub origins: Vec<String>,
    /// Omitted means [`Context::UNBOUNDED`]: the whole component.
    pub max_depth: Option<usize>,
}

pub async fn explore(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<ExploreRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| {
        let origins: Vec<&str> = request.origins.iter().map(String::as_str).collect();
        context.explore(&origins, request.max_depth.unwrap_or(Context::UNBOUNDED))
    }) {
        Some(result) => ok(result, started_at),
        None => not_found(&name, started_at),
    }
}

#[derive(Debug, Deserialize)]
pub struct ActivateRequest {
    pub origins: Vec<String>,
    /// Omitted means 0.5 — the halving-per-hop default the examples use.
    pub decay: Option<f64>,
    /// Omitted means 20 results.
    pub limit: Option<usize>,
}

pub async fn activate(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<ActivateRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| {
        let origins: Vec<&str> = request.origins.iter().map(String::as_str).collect();
        context.activate(
            &origins,
            request.decay.unwrap_or(0.5),
            request.limit.unwrap_or(20),
        )
    }) {
        Some(result) => ok(result, started_at),
        None => not_found(&name, started_at),
    }
}

#[derive(Debug, Deserialize)]
pub struct ResolveRequest {
    pub cue: String,
}

pub async fn resolve(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<ResolveRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| context.resolve(&request.cue)) {
        Some(result) => ok(result, started_at),
        None => not_found(&name, started_at),
    }
}

pub async fn resolve_label(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<ResolveRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| context.resolve_label(&request.cue)) {
        Some(result) => ok(result, started_at),
        None => not_found(&name, started_at),
    }
}

pub async fn labels(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| {
        let labels: Vec<String> = context.labels().into_iter().map(String::from).collect();
        labels
    }) {
        Some(result) => ok(result, started_at),
        None => not_found(&name, started_at),
    }
}

#[derive(Debug, Deserialize)]
pub struct UnreachableFromRequest {
    pub origins: Vec<String>,
}

pub async fn unreachable_from(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<UnreachableFromRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| {
        let origins: Vec<&str> = request.origins.iter().map(String::as_str).collect();
        context.unreachable_from(&origins)
    }) {
        Some(result) => ok(result, started_at),
        None => not_found(&name, started_at),
    }
}
