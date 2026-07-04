//! HTTP surface of the retrieval service: thin JSON adapters, one per
//! `Context` operation. The server adds transport, naming, and lifecycle
//! around the library — never retrieval semantics of its own — so each
//! handler is a lock, a library call, and a serialized reply.

use std::collections::BTreeMap;
use std::time::Instant;

use associative_rag::context::{AliasError, Association, Context};
use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use crate::registry::{AccessError, AppState, ContextMeta, CreateError};

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

fn access_error(failure: AccessError, name: &str, started_at: Instant) -> Response {
    match failure {
        AccessError::NotFound => not_found(name, started_at),
        AccessError::Load(message) => error(StatusCode::INTERNAL_SERVER_ERROR, message, started_at),
    }
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
            format!("metadata update not persisted: {io_error}"),
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
/// acquisition. The write becomes durable at the next flush (periodic,
/// on eviction, or on shutdown).
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

    match outcome {
        Err(failure) => access_error(failure, &name, started_at),
        Ok(Ok(applied)) => ok(applied, started_at),
        // Items before the failing one are applied (each item is
        // all-or-nothing in the library); report how far the batch got.
        Ok(Err((applied, full))) => error(
            StatusCode::INSUFFICIENT_STORAGE,
            format!(
                "applied {applied} of {} associations, then: {full}",
                associations.len()
            ),
            started_at,
        ),
    }
}

/// Cap applied to recall/query matches when the request names no limit:
/// a hub concept must not flood an LLM client's prompt by default.
const DEFAULT_MATCH_LIMIT: usize = 100;

/// A bounded set of matches. `total` is the full match count before the
/// limit was applied, so a client can see that it is looking at a
/// truncated view and narrow the query (or raise the limit).
#[derive(Serialize)]
pub struct MatchPage {
    pub total: usize,
    pub matches: Vec<Association>,
}

/// Bounds a match list: within the limit the library's insertion order
/// is preserved; past it, the strongest knowledge (|weight|, the same
/// magnitude-ranks philosophy as activate) survives the cut, ties in
/// insertion order.
fn page(mut matches: Vec<Association>, limit: Option<usize>) -> MatchPage {
    let total = matches.len();
    let limit = limit.unwrap_or(DEFAULT_MATCH_LIMIT);
    if total > limit {
        matches.sort_by(|a, b| b.weight.abs().total_cmp(&a.weight.abs()));
        matches.truncate(limit);
    }
    MatchPage { total, matches }
}

/// One name or several: query positions accept `"住所"` and
/// `["住所", "職歴"]` interchangeably.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany {
    One(String),
    Many(Vec<String>),
}

fn as_refs(position: &Option<OneOrMany>) -> Vec<&str> {
    match position {
        None => Vec::new(),
        Some(OneOrMany::One(name)) => vec![name.as_str()],
        Some(OneOrMany::Many(names)) => names.iter().map(String::as_str).collect(),
    }
}

/// Alias registrations, alias → canonical per namespace. Applied in
/// sorted order (BTreeMap), aborting at the first failure with the
/// applied count reported — like association batches, each item is
/// all-or-nothing in the library.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct AliasRequest {
    pub concepts: BTreeMap<String, String>,
    pub labels: BTreeMap<String, String>,
}

/// The full alias vocabulary of one context — the exportable unit for
/// bulk workflows (dump names, translate, re-import).
#[derive(Serialize)]
pub struct AliasExport {
    pub concepts: BTreeMap<String, String>,
    pub labels: BTreeMap<String, String>,
}

pub async fn add_aliases(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<AliasRequest>,
) -> Response {
    let started_at = Instant::now();
    let outcome = state.write_context(&name, |context| {
        let mut applied = 0usize;
        for (alias, canonical) in &request.concepts {
            if let Err(alias_error) = context.add_concept_alias(alias.as_str(), canonical) {
                let full = matches!(alias_error, AliasError::Full(_));
                return Err((
                    applied,
                    format!("concept alias '{alias}' → '{canonical}': {alias_error}"),
                    full,
                ));
            }
            applied += 1;
        }
        for (alias, canonical) in &request.labels {
            if let Err(alias_error) = context.add_label_alias(alias.as_str(), canonical) {
                let full = matches!(alias_error, AliasError::Full(_));
                return Err((
                    applied,
                    format!("label alias '{alias}' → '{canonical}': {alias_error}"),
                    full,
                ));
            }
            applied += 1;
        }
        Ok(applied)
    });

    match outcome {
        Err(failure) => access_error(failure, &name, started_at),
        Ok(Ok(applied)) => ok(applied, started_at),
        Ok(Err((applied, message, full))) => {
            let status = if full {
                StatusCode::INSUFFICIENT_STORAGE
            } else {
                StatusCode::CONFLICT
            };
            error(
                status,
                format!("applied {applied} aliases, then {message}"),
                started_at,
            )
        }
    }
}

pub async fn list_aliases(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| AliasExport {
        concepts: context
            .concept_aliases()
            .into_iter()
            .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
            .collect(),
        labels: context
            .label_aliases()
            .into_iter()
            .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
            .collect(),
    }) {
        Ok(result) => ok(result, started_at),
        Err(failure) => access_error(failure, &name, started_at),
    }
}

/// Original text passages keyed by source id — the same opaque ids that
/// appear on attributions.
#[derive(Debug, Deserialize)]
pub struct StorePassagesRequest {
    pub passages: BTreeMap<String, String>,
}

pub async fn store_passages(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<StorePassagesRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.store_passages(&name, request.passages) {
        None => not_found(&name, started_at),
        Some(Ok(stored)) => ok(stored, started_at),
        Some(Err(io_error)) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("passages could not be persisted: {io_error}"),
            started_at,
        ),
    }
}

#[derive(Debug, Deserialize)]
pub struct LookupPassagesRequest {
    pub sources: Vec<String>,
}

/// The dereference half of "find with the graph, answer from the text":
/// attributions name sources, this returns the original passages behind
/// them (and which sources have none registered).
#[derive(Serialize)]
pub struct PassageLookup {
    pub passages: BTreeMap<String, String>,
    pub missing: Vec<String>,
}

pub async fn lookup_passages(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<LookupPassagesRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.lookup_passages(&name, &request.sources) {
        None => not_found(&name, started_at),
        Some((passages, missing)) => ok(PassageLookup { passages, missing }, started_at),
    }
}

pub async fn list_sources(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let started_at = Instant::now();
    match state.passage_sources(&name) {
        None => not_found(&name, started_at),
        Some(sources) => ok(sources, started_at),
    }
}

#[derive(Debug, Deserialize)]
pub struct RecallRequest {
    pub cue: String,
    /// Omitted means 100.
    pub limit: Option<usize>,
}

pub async fn recall(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<RecallRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| context.recall(&request.cue)) {
        Ok(result) => ok(page(result, request.limit), started_at),
        Err(failure) => access_error(failure, &name, started_at),
    }
}

#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub subject: Option<OneOrMany>,
    pub label: Option<OneOrMany>,
    pub object: Option<OneOrMany>,
    /// Omitted means 100.
    pub limit: Option<usize>,
}

pub async fn query(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<QueryRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| {
        context.query_any(
            &as_refs(&request.subject),
            &as_refs(&request.label),
            &as_refs(&request.object),
        )
    }) {
        Ok(result) => ok(page(result, request.limit), started_at),
        Err(failure) => access_error(failure, &name, started_at),
    }
}

#[derive(Debug, Deserialize)]
pub struct DescribeRequest {
    pub concept: String,
}

/// The staged-read entry point: what kinds of knowledge exist about a
/// concept (labels and counts, per role) without materializing a single
/// association. Check the outline, then `query` just the labels that
/// matter. An unknown concept comes back as a null result.
pub async fn describe(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<DescribeRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| context.describe(&request.concept)) {
        Ok(result) => ok(result, started_at),
        Err(failure) => access_error(failure, &name, started_at),
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
        Ok(result) => ok(result, started_at),
        Err(failure) => access_error(failure, &name, started_at),
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
        Ok(result) => ok(result, started_at),
        Err(failure) => access_error(failure, &name, started_at),
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
        Ok(result) => ok(result, started_at),
        Err(failure) => access_error(failure, &name, started_at),
    }
}

pub async fn resolve_label(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<ResolveRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| context.resolve_label(&request.cue)) {
        Ok(result) => ok(result, started_at),
        Err(failure) => access_error(failure, &name, started_at),
    }
}

pub async fn labels(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| {
        let labels: Vec<String> = context.labels().into_iter().map(String::from).collect();
        labels
    }) {
        Ok(result) => ok(result, started_at),
        Err(failure) => access_error(failure, &name, started_at),
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
        Ok(result) => ok(result, started_at),
        Err(failure) => access_error(failure, &name, started_at),
    }
}
