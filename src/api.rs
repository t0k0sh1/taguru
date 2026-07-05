//! HTTP surface of the retrieval service: thin JSON adapters, one per
//! `Context` operation. The server adds transport, naming, and lifecycle
//! around the library — never retrieval semantics of its own — so each
//! handler is a lock, a library call, and a serialized reply.

use std::collections::BTreeMap;
use std::time::Instant;

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Path, Request, State};
use axum::http::{Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use taguru::context::{Association, Context, Recollection, Resolution};

use crate::metrics::{ErrorKind, ResolveTier, SearchOp};
use crate::registry::{AccessError, AppState, AssocOp, ContextMeta, CreateError};

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

pub(crate) fn error(
    status: StatusCode,
    message: impl Into<String>,
    started_at: Instant,
) -> Response {
    (status, Json(ApiError::new(message, started_at))).into_response()
}

fn not_found(name: &str, started_at: Instant) -> Response {
    error(
        StatusCode::NOT_FOUND,
        format!("context '{name}' not found"),
        started_at,
    )
}

/// axum's Json extractor with its rejections reshaped into the
/// [`ApiError`] body: a machine client parses ONE error shape on every
/// axis, malformed JSON included. The status codes stay axum's — 400
/// for broken syntax, 415 for the wrong media type, 422 for
/// well-formed JSON of the wrong type — only the body changes.
pub struct AppJson<T>(pub T);

impl<S, T> FromRequest<S> for AppJson<T>
where
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(request, state).await {
            Ok(Json(value)) => Ok(Self(value)),
            Err(rejection) => Err(error(
                rejection.status(),
                rejection.body_text(),
                Instant::now(),
            )),
        }
    }
}

/// axum's Query extractor with rejections reshaped into the
/// [`ApiError`] body, exactly as [`AppJson`] does for request bodies.
pub struct AppQuery<T>(pub T);

impl<S, T> axum::extract::FromRequestParts<S> for AppQuery<T>
where
    T: serde::de::DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        match axum::extract::Query::<T>::from_request_parts(parts, state).await {
            Ok(axum::extract::Query(value)) => Ok(Self(value)),
            Err(rejection) => Err(error(
                rejection.status(),
                rejection.body_text(),
                Instant::now(),
            )),
        }
    }
}

/// The router-wide 404: paths outside the API answer in the error
/// shape too, with a pointer at the self-describing endpoint.
pub async fn unknown_path(method: Method, uri: Uri) -> Response {
    error(
        StatusCode::NOT_FOUND,
        format!("no route for {method} {uri}; GET /protocol lists the API"),
        Instant::now(),
    )
}

/// A known path hit with the wrong verb — same story, 405.
pub async fn method_not_allowed(method: Method, uri: Uri) -> Response {
    error(
        StatusCode::METHOD_NOT_ALLOWED,
        format!("{method} is not supported on {uri}; GET /protocol lists the API"),
        Instant::now(),
    )
}

fn access_error(
    state: &AppState,
    failure: AccessError,
    name: &str,
    started_at: Instant,
) -> Response {
    match failure {
        AccessError::NotFound => not_found(name, started_at),
        AccessError::Load(message) => {
            state.metrics().record_error(ErrorKind::Load);
            error(StatusCode::INTERNAL_SERVER_ERROR, message, started_at)
        }
        AccessError::Unpersisted(message) => {
            state.metrics().record_error(ErrorKind::WalRefused);
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("write not persisted (nothing was applied): {message}"),
                started_at,
            )
        }
    }
}

/// Keyset paging over the name-sorted directory.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct ListContextsQuery {
    /// Page size; omitted means the ceiling (1000) — the directory is
    /// the routing surface, and a sane deployment fits one page.
    pub limit: Option<usize>,
    /// Only contexts whose name sorts strictly after this one.
    pub after: Option<String>,
}

/// A bounded directory page: like every other listing, `total` names
/// the full count so a truncated view is visible.
#[derive(Serialize)]
pub struct ContextPage {
    pub total: usize,
    pub contexts: Vec<crate::registry::DirectoryEntry>,
}

/// The routing directory, skills-style: name, prose description, and
/// stats for every context, so an LLM client can decide where to search
/// (and where to ingest) without the server owning that judgement.
/// Paged like every other listing — thousands of contexts must not
/// mean a megabytes-large response on every routing decision.
pub async fn list_contexts(
    State(state): State<AppState>,
    AppQuery(query): AppQuery<ListContextsQuery>,
) -> Response {
    let started_at = Instant::now();
    let directory = state.directory();
    let total = directory.len();
    let contexts: Vec<_> = directory
        .into_iter()
        .filter(|entry| {
            query
                .after
                .as_deref()
                .is_none_or(|after| entry.name.as_str() > after)
        })
        .take(clamp(query.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT))
        .collect();
    ok(ContextPage { total, contexts }, started_at)
}

/// POST /flush: persist every dirty context NOW and answer with the
/// names that flushed — the quiescing move before a file-level backup,
/// instead of "stop the server or wait out the flush interval".
pub async fn flush_all(State(state): State<AppState>) -> Response {
    let started_at = Instant::now();
    let flushed = tokio::task::block_in_place(|| state.flush_dirty());
    ok(flushed, started_at)
}

/// One directory row by name — the cheap existence-and-stats check,
/// without listing anything else.
pub async fn get_context(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let started_at = Instant::now();
    match state.directory_entry(&name) {
        Some(entry) => ok(entry, started_at),
        None => not_found(&name, started_at),
    }
}

/// The LLM-facing manual — ingest discipline, retrieval loop, API
/// reference — served by the API itself so a client can learn the
/// protocol at connect time, the way it would read a skill.
pub async fn protocol() -> Response {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/markdown; charset=utf-8",
        )],
        include_str!("../docs/llm-protocol.md"),
    )
        .into_response()
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct CreateContextRequest {
    pub description: String,
    pub pinned: bool,
    /// Per-context fuzzy-entry floor for resolve; omitted means the
    /// default (0.3).
    pub dice_floor: Option<f64>,
    /// Per-context semantic floor; omitted means the default (0.35).
    pub semantic_floor: Option<f32>,
}

pub async fn create_context(
    State(state): State<AppState>,
    Path(name): Path<String>,
    body: axum::body::Bytes,
) -> Response {
    let started_at = Instant::now();
    let request: CreateContextRequest = match optional_body(&body, started_at) {
        Ok(request) => request,
        Err(refusal) => return *refusal,
    };
    if let Some(refusal) = oversized(
        "the context name",
        &name,
        MAX_CONTEXT_NAME_BYTES,
        started_at,
    ) {
        return refusal;
    }
    if let Some(refusal) = oversized(
        "the description",
        &request.description,
        MAX_DESCRIPTION_BYTES,
        started_at,
    ) {
        return refusal;
    }
    let meta = ContextMeta {
        description: request.description,
        pinned: request.pinned,
        dice_floor: request.dice_floor.map(|floor| floor.clamp(0.0, 1.0)),
        semantic_floor: request.semantic_floor.map(|floor| floor.clamp(0.0, 1.0)),
    };
    match state.create(&name, meta) {
        Ok(()) => ok(true, started_at),
        Err(CreateError::AlreadyExists) => error(
            StatusCode::CONFLICT,
            format!("context '{name}' already exists"),
            started_at,
        ),
        Err(CreateError::Io(io_error)) => {
            state.metrics().record_error(ErrorKind::Io);
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("context '{name}' could not be persisted: {io_error}"),
                started_at,
            )
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct UpdateContextRequest {
    pub description: Option<String>,
    pub pinned: Option<bool>,
    pub dice_floor: Option<f64>,
    pub semantic_floor: Option<f32>,
}

pub async fn update_context(
    State(state): State<AppState>,
    Path(name): Path<String>,
    AppJson(request): AppJson<UpdateContextRequest>,
) -> Response {
    let started_at = Instant::now();
    if let Some(description) = &request.description
        && let Some(refusal) = oversized(
            "the description",
            description,
            MAX_DESCRIPTION_BYTES,
            started_at,
        )
    {
        return refusal;
    }
    match state.update_meta(
        &name,
        request.description,
        request.pinned,
        request.dice_floor,
        request.semantic_floor,
    ) {
        None => not_found(&name, started_at),
        Some(Ok(meta)) => ok(meta, started_at),
        Some(Err(io_error)) => {
            state.metrics().record_error(ErrorKind::Io);
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("metadata update not persisted: {io_error}"),
                started_at,
            )
        }
    }
}

pub async fn delete_context(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let started_at = Instant::now();
    match state.delete(&name) {
        None => not_found(&name, started_at),
        Some(Ok(())) => ok(true, started_at),
        Some(Err(io_error)) => {
            state.metrics().record_error(ErrorKind::Io);
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("context '{name}' removed but its files were not: {io_error}"),
                started_at,
            )
        }
    }
}

/// A batch of assertions — the arguments of `associate` /
/// `associate_from` as JSON fields ([`AssocOp`], shared with the WAL).
/// A batch is the natural write unit: one document's extracted facts,
/// one request, one lock acquisition, one WAL fsync.
pub async fn add_associations(
    State(state): State<AppState>,
    Path(name): Path<String>,
    AppJson(associations): AppJson<Vec<AssocOp>>,
) -> Response {
    let started_at = Instant::now();
    // Refused before the write lock is even taken: nothing of an
    // oversized batch is applied.
    if associations.len() > MAX_ASSOCIATIONS_PER_REQUEST {
        return error(
            StatusCode::BAD_REQUEST,
            format!(
                "batch of {} associations exceeds the per-request limit of \
                 {MAX_ASSOCIATIONS_PER_REQUEST}; split the ingest",
                associations.len()
            ),
            started_at,
        );
    }
    // Also refused whole: a weight the graph must never accumulate.
    // JSON cannot carry Infinity or NaN, but it carries 1e300 just
    // fine, and saturation never washes out of an edge.
    for (index, op) in associations.iter().enumerate() {
        if !op.weight.is_finite() || op.weight.abs() > MAX_ASSOCIATION_WEIGHT {
            return error(
                StatusCode::BAD_REQUEST,
                format!(
                    "associations[{index}].weight {} is outside the accepted range \
                     (finite, |weight| <= {MAX_ASSOCIATION_WEIGHT}); nothing was applied",
                    op.weight
                ),
                started_at,
            );
        }
        // And a name the graph must never intern — see MAX_NAME_BYTES.
        let source = op.source.as_deref().unwrap_or("");
        for (field, value) in [
            ("subject", op.subject.as_str()),
            ("label", op.label.as_str()),
            ("object", op.object.as_str()),
            ("source", source),
        ] {
            if let Some(refusal) = oversized(
                &format!("associations[{index}].{field}"),
                value,
                MAX_NAME_BYTES,
                started_at,
            ) {
                return refusal;
            }
        }
    }
    let total = associations.len();
    match state.add_associations(&name, associations) {
        Err(failure) => access_error(&state, failure, &name, started_at),
        Ok(Ok(applied)) => ok(applied, started_at),
        // Items before the failing one are applied (each item is
        // all-or-nothing in the library); report how far the batch got.
        Ok(Err(partial)) => error(
            StatusCode::INSUFFICIENT_STORAGE,
            format!(
                "applied {} of {total} associations, then: {}",
                partial.applied, partial.message
            ),
            started_at,
        ),
    }
}

/// Cap applied to recall/query matches when the request names no limit:
/// a hub concept must not flood an LLM client's prompt by default.
const DEFAULT_MATCH_LIMIT: usize = 100;

/// The hard ceiling every per-request result limit is clamped to,
/// whatever the request asks for — result sizing is a server
/// protection, not a client entitlement.
const MAX_MATCH_LIMIT: usize = 1000;

/// Hop ceiling for explore, applied even when `max_depth` is omitted:
/// the walk's cost is bounded by depth, so the depth is what the
/// server caps. Ten hops of association-following covers any sane
/// retrieval; audits that truly need the whole component use
/// unreachable_from.
const MAX_EXPLORE_DEPTH: usize = 10;

/// Per-request association batch cap — one document's facts arrive as
/// one request; anything past this is asked to split.
const MAX_ASSOCIATIONS_PER_REQUEST: usize = 10_000;

/// Per-op weight ceiling (absolute value). Weights accumulate on an
/// edge across writes and floats saturate: two f64::MAX writes made an
/// edge +Infinity, and a later retract minted Inf − Inf = NaN — an
/// unreadable, unresettable fact. At ±1e6 per op, saturating would
/// take ~1.8e302 acknowledged writes.
const MAX_ASSOCIATION_WEIGHT: f64 = 1e6;

/// Byte cap on every name-shaped write: subject/label/object, source
/// ids, aliases. Names are entry keys, not documents — the graph
/// interns every distinct spelling forever, and the top-concepts
/// snapshot carries them into every directory listing (one 4 MB
/// subject made every GET /contexts response 4 MB, resident outside
/// the cache budget).
const MAX_NAME_BYTES: usize = 1024;

/// Byte cap on a context name: it becomes a file stem, percent-encoded
/// at up to 3× — 64 bytes keeps the longest sidecar filename well
/// under every filesystem's 255-byte limit.
const MAX_CONTEXT_NAME_BYTES: usize = 64;

/// Byte cap on a context description — it too rides in every
/// directory listing.
const MAX_DESCRIPTION_BYTES: usize = 4096;

/// The optional-body contract of create and audit: an ABSENT body
/// means defaults, but a PRESENT body must parse as JSON — whatever
/// the Content-Type header says. `Option<Json<T>>` answered a
/// missing or mismatched header with `None` even when a body was
/// present, so a client that forgot the header (Python's
/// `requests.put(url, data=...)` territory) had its whole payload
/// silently replaced by defaults, under a 200.
fn optional_body<T: Default + serde::de::DeserializeOwned>(
    body: &axum::body::Bytes,
    started_at: Instant,
) -> Result<T, Box<Response>> {
    if body.is_empty() {
        return Ok(T::default());
    }
    serde_json::from_slice(body).map_err(|parse| {
        Box::new(error(
            StatusCode::BAD_REQUEST,
            format!("the request body is not valid JSON: {parse}"),
            started_at,
        ))
    })
}

/// The one gate every persisted name goes through: `Some(400)` when
/// `value` runs over `cap`, with `what` naming the offender precisely
/// enough to fix the request. The value itself is never echoed — it
/// is the oversized thing.
fn oversized(what: &str, value: &str, cap: usize, started_at: Instant) -> Option<Response> {
    (value.len() > cap).then(|| {
        error(
            StatusCode::BAD_REQUEST,
            format!(
                "{what} is {} bytes, over the {cap}-byte cap; nothing was applied",
                value.len()
            ),
            started_at,
        )
    })
}

/// The one clamp every numeric cap in this file goes through: an
/// omitted value takes the default, and nothing exceeds the ceiling.
fn clamp(value: Option<usize>, default: usize, ceiling: usize) -> usize {
    value.unwrap_or(default).min(ceiling)
}

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
    let limit = clamp(limit, DEFAULT_MATCH_LIMIT, MAX_MATCH_LIMIT);
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
    AppJson(request): AppJson<AliasRequest>,
) -> Response {
    let started_at = Instant::now();
    // Aliases intern names on both sides; the same cap as every other
    // name-shaped write.
    for (namespace, pairs) in [("concepts", &request.concepts), ("labels", &request.labels)] {
        for (alias, canonical) in pairs {
            for (role, value) in [("alias", alias.as_str()), ("canonical", canonical.as_str())] {
                if let Some(refusal) = oversized(
                    &format!("a {namespace} {role}"),
                    value,
                    MAX_NAME_BYTES,
                    started_at,
                ) {
                    return refusal;
                }
            }
        }
    }
    match state.add_aliases(&name, &request.concepts, &request.labels) {
        Err(failure) => access_error(&state, failure, &name, started_at),
        Ok(Ok(applied)) => ok(applied, started_at),
        Ok(Err(partial)) => {
            let status = if partial.full {
                StatusCode::INSUFFICIENT_STORAGE
            } else {
                StatusCode::CONFLICT
            };
            error(
                status,
                format!(
                    "applied {} aliases, then {}",
                    partial.applied, partial.message
                ),
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
        Err(failure) => access_error(&state, failure, &name, started_at),
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
    AppJson(request): AppJson<StorePassagesRequest>,
) -> Response {
    let started_at = Instant::now();
    // Source ids are names (the passage text itself is a document and
    // rides under the body cap instead).
    for source in request.passages.keys() {
        if let Some(refusal) = oversized("a passage source id", source, MAX_NAME_BYTES, started_at)
        {
            return refusal;
        }
    }
    match state.store_passages(&name, request.passages) {
        None => not_found(&name, started_at),
        Some(Ok(stored)) => ok(stored, started_at),
        Some(Err(io_error)) => {
            state.metrics().record_error(ErrorKind::Io);
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("passages could not be persisted: {io_error}"),
                started_at,
            )
        }
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
    AppJson(request): AppJson<LookupPassagesRequest>,
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

/// Vocabulary audit request: floors for the two fork detectors.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct VocabularyAuditRequest {
    /// Lexical (spelling) detector floor; omitted means 0.6.
    pub dice_floor: Option<f64>,
    /// Semantic (gloss cosine) detector floor; omitted means 0.6.
    pub cosine_floor: Option<f32>,
}

#[derive(Serialize)]
pub struct TwinPair<S> {
    pub a: String,
    pub b: String,
    pub score: S,
}

/// The vocabulary health report: fork CANDIDATES for review, not
/// verdicts. Lexical twins are spelling drift (青嶺酒蔵/青嶺酒造);
/// semantic twins are synonym drift (創業年/設立年) visible only
/// through gloss embeddings.
#[derive(Serialize)]
pub struct VocabularyAudit {
    pub lexical_concepts: Vec<TwinPair<f64>>,
    pub lexical_labels: Vec<TwinPair<f64>>,
    pub semantic_concepts: Vec<TwinPair<f32>>,
    pub semantic_labels: Vec<TwinPair<f32>>,
    /// Why the semantic half was skipped, when it was.
    pub semantic_note: Option<String>,
}

fn twin_pairs<S>(pairs: Vec<(String, String, S)>) -> Vec<TwinPair<S>> {
    pairs
        .into_iter()
        .map(|(a, b, score)| TwinPair { a, b, score })
        .collect()
}

pub async fn audit_vocabulary(
    State(state): State<AppState>,
    Path(name): Path<String>,
    body: axum::body::Bytes,
) -> Response {
    let started_at = Instant::now();
    let request: VocabularyAuditRequest = match optional_body(&body, started_at) {
        Ok(request) => request,
        Err(refusal) => return *refusal,
    };
    let dice_floor = request.dice_floor.unwrap_or(0.6);
    let cosine_floor = request.cosine_floor.unwrap_or(0.6);

    // BOTH halves are CPU-bound pairwise sweeps — the lexical one is
    // O(Σ posting_len²) over the whole vocabulary, seconds at tens of
    // thousands of concepts. Neither may run on an async worker: with
    // the lexical half inline, a handful of concurrent audits pinned
    // every worker and starved every other request, /health included.
    let lexical = match tokio::task::block_in_place(|| {
        state.read_context(&name, |context| {
            (
                context.similar_concepts(dice_floor),
                context.similar_labels(dice_floor),
            )
        })
    }) {
        Ok(lexical) => lexical,
        Err(failure) => return access_error(&state, failure, &name, started_at),
    };
    let semantic = tokio::task::block_in_place(|| state.semantic_twins(&name, cosine_floor));
    let Some((semantic_concepts, semantic_labels, semantic_note)) = semantic else {
        return not_found(&name, started_at);
    };

    ok(
        VocabularyAudit {
            lexical_concepts: twin_pairs(lexical.0),
            lexical_labels: twin_pairs(lexical.1),
            semantic_concepts: twin_pairs(semantic_concepts),
            semantic_labels: twin_pairs(semantic_labels),
            semantic_note,
        },
        started_at,
    )
}

#[derive(Debug, Deserialize)]
pub struct RetractSourceRequest {
    pub source: String,
}

/// What one retraction accomplished: how many associations lost this
/// source's contribution, and whether its passage went with it.
#[derive(Serialize)]
pub struct RetractOutcome {
    pub associations_touched: usize,
    pub passage_removed: bool,
}

pub async fn retract_source(
    State(state): State<AppState>,
    Path(name): Path<String>,
    AppJson(request): AppJson<RetractSourceRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.retract_source(&name, &request.source) {
        Err(failure) => access_error(&state, failure, &name, started_at),
        Ok((associations_touched, passage_removed)) => ok(
            RetractOutcome {
                associations_touched,
                passage_removed,
            },
            started_at,
        ),
    }
}

#[derive(Debug, Deserialize)]
pub struct SearchPassagesRequest {
    pub query: String,
    /// Omitted means 5.
    pub limit: Option<usize>,
}

/// One passage matched by full-text search: the second retrieval lane,
/// for knowledge that never decomposed into triples.
#[derive(Serialize)]
pub struct PassageHit {
    pub source: String,
    pub score: f32,
    pub text: String,
}

pub async fn search_passages(
    State(state): State<AppState>,
    Path(name): Path<String>,
    AppJson(request): AppJson<SearchPassagesRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.search_passages(
        &name,
        &request.query,
        clamp(request.limit, 5, MAX_MATCH_LIMIT),
    ) {
        None => not_found(&name, started_at),
        Some(hits) => {
            state
                .metrics()
                .record_search(SearchOp::SearchPassages, hits.is_empty());
            ok(
                hits.into_iter()
                    .map(|(source, score, text)| PassageHit {
                        source,
                        score,
                        text,
                    })
                    .collect::<Vec<_>>(),
                started_at,
            )
        }
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
    AppJson(request): AppJson<RecallRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| context.recall(&request.cue)) {
        Ok(result) => {
            let paged = page(result, request.limit);
            state
                .metrics()
                .record_search(SearchOp::Recall, paged.total == 0);
            ok(paged, started_at)
        }
        Err(failure) => access_error(&state, failure, &name, started_at),
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
    AppJson(request): AppJson<QueryRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| {
        context.query_any(
            &as_refs(&request.subject),
            &as_refs(&request.label),
            &as_refs(&request.object),
        )
    }) {
        Ok(result) => {
            let paged = page(result, request.limit);
            state
                .metrics()
                .record_search(SearchOp::Query, paged.total == 0);
            ok(paged, started_at)
        }
        Err(failure) => access_error(&state, failure, &name, started_at),
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
    AppJson(request): AppJson<DescribeRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| context.describe(&request.concept)) {
        Ok(result) => ok(result, started_at),
        Err(failure) => access_error(&state, failure, &name, started_at),
    }
}

#[derive(Debug, Deserialize)]
pub struct ExploreRequest {
    pub origins: Vec<String>,
    /// Hop ceiling. Omitted — and everything above it — means the
    /// server maximum ([`MAX_EXPLORE_DEPTH`]).
    pub max_depth: Option<usize>,
    /// Result cap. Omitted means 100, ceiling 1000 — depth bounds the
    /// walk, this bounds the response.
    pub limit: Option<usize>,
}

/// A bounded explore result: the same `{total, matches}` shape as
/// [`MatchPage`], but the cut keeps the CLOSEST structure — explore
/// is a neighbourhood walk, so past the limit the nearest hops
/// survive, not the heaviest weights. The library already returns
/// matches sorted by distance, so truncation is the whole cut.
#[derive(Serialize)]
pub struct ExplorePage {
    pub total: usize,
    pub matches: Vec<Recollection>,
}

pub async fn explore(
    State(state): State<AppState>,
    Path(name): Path<String>,
    AppJson(request): AppJson<ExploreRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| {
        let origins: Vec<&str> = request.origins.iter().map(String::as_str).collect();
        // The clamp turns "omitted = the whole component" into
        // "omitted = the server's hop ceiling".
        context.explore(
            &origins,
            clamp(request.max_depth, Context::UNBOUNDED, MAX_EXPLORE_DEPTH),
        )
    }) {
        Ok(mut matches) => {
            // Depth alone does not bound the response: one dense hub
            // can put a million edges within a single hop, and explore
            // used to return them all in one body.
            let total = matches.len();
            matches.truncate(clamp(request.limit, DEFAULT_MATCH_LIMIT, MAX_MATCH_LIMIT));
            state.metrics().record_search(SearchOp::Explore, total == 0);
            ok(ExplorePage { total, matches }, started_at)
        }
        Err(failure) => access_error(&state, failure, &name, started_at),
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
    AppJson(request): AppJson<ActivateRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| {
        let origins: Vec<&str> = request.origins.iter().map(String::as_str).collect();
        context.activate(
            &origins,
            request.decay.unwrap_or(0.5),
            clamp(request.limit, 20, MAX_MATCH_LIMIT),
        )
    }) {
        Ok(result) => {
            state
                .metrics()
                .record_search(SearchOp::Activate, result.is_empty());
            ok(result, started_at)
        }
        Err(failure) => access_error(&state, failure, &name, started_at),
    }
}

#[derive(Debug, Deserialize)]
pub struct ResolveRequest {
    pub cue: String,
    /// One-call override of the fuzzy-entry floor — the loosen-and-retry
    /// move after a miss. Omitted means the context's setting.
    pub dice_floor: Option<f64>,
    /// One-call override of the semantic-tier floor, same story.
    pub semantic_floor: Option<f32>,
}

/// One resolve candidate plus the tier that produced it. Lexical scores
/// are coverage/Dice, semantic scores are cosine similarities — ordinal
/// within a tier, never comparable across tiers.
#[derive(Serialize)]
pub struct TieredResolution {
    pub name: String,
    pub score: f64,
    pub tier: &'static str,
}

fn lexical_tier(resolutions: Vec<Resolution>) -> Vec<TieredResolution> {
    resolutions
        .into_iter()
        .map(|resolution| TieredResolution {
            name: resolution.name,
            score: resolution.score,
            tier: "lexical",
        })
        .collect()
}

/// Lexical coverage at or above this is a confident entry — the cue is
/// (at least half of) a stored spelling — and the semantic tier is
/// skipped. Below it, a lexical hit is a fragment collision (蔵 inside
/// 祭りを主催する蔵元 scores 0.11) and MUST NOT silence the semantic
/// tier: in a dense vocabulary some substring almost always matches
/// something, and letting it gate the entry buried 0.55-cosine answers
/// behind 0.11-coverage noise.
const LEXICAL_CONFIDENCE: f64 = 0.5;

/// Merges the two entry tiers: lexical candidates keep the front (their
/// scores are string evidence, best first), semantic candidates append
/// for names not already present. Scales stay incomparable, which is
/// what the tier field is for.
fn merge_tiers(lexical: Vec<Resolution>, semantic: Vec<(String, f32)>) -> Vec<TieredResolution> {
    let mut merged = lexical_tier(lexical);
    for (name, score) in semantic {
        if merged.iter().any(|candidate| candidate.name == name) {
            continue;
        }
        merged.push(TieredResolution {
            name,
            score: f64::from(score),
            tier: "semantic",
        });
    }
    merged
}

/// The full entry ladder: lexical tiers first; the semantic tier runs
/// whenever they came back empty OR merely fragment-weak (best score
/// under [`LEXICAL_CONFIDENCE`]), and its candidates are appended after
/// the lexical ones.
fn resolve_with_fallback(
    state: &AppState,
    name: &str,
    request: &ResolveRequest,
    labels: bool,
    started_at: Instant,
) -> Response {
    let lexical = match state.read_context(name, |context| match (labels, request.dice_floor) {
        (false, Some(floor)) => context.resolve_with_floor(&request.cue, floor),
        (false, None) => context.resolve(&request.cue),
        (true, Some(floor)) => context.resolve_label_with_floor(&request.cue, floor),
        (true, None) => context.resolve_label(&request.cue),
    }) {
        Ok(result) => result,
        Err(failure) => return access_error(state, failure, name, started_at),
    };
    let confident = lexical
        .first()
        .is_some_and(|best| best.score >= LEXICAL_CONFIDENCE);
    let served = if confident {
        lexical_tier(lexical)
    } else {
        // The provider round trip can take hundreds of milliseconds;
        // tell the runtime this thread will block so other tasks
        // migrate off it.
        match tokio::task::block_in_place(|| {
            state.semantic_resolve(name, &request.cue, labels, request.semantic_floor)
        }) {
            None => return not_found(name, started_at),
            Some(Ok(semantic)) => merge_tiers(lexical, semantic),
            // The semantic tier is enrichment once ANY lexical
            // candidate exists: degrade to the weak lexical results
            // and log, rather than failing an answerable request over
            // a provider hiccup.
            Some(Err(message)) if !lexical.is_empty() => {
                tracing::warn!("semantic entry failed (serving weak lexical results): {message}");
                lexical_tier(lexical)
            }
            Some(Err(message)) => {
                tracing::warn!(
                    "semantic entry failed with nothing lexical to fall back on: {message}"
                );
                return error(
                    StatusCode::BAD_GATEWAY,
                    format!("semantic entry failed: {message}"),
                    started_at,
                );
            }
        }
    };
    let op = if labels {
        SearchOp::ResolveLabel
    } else {
        SearchOp::Resolve
    };
    state.metrics().record_search(op, served.is_empty());
    state
        .metrics()
        .record_resolve_tier(resolve_tier_of(&served));
    ok(served, started_at)
}

/// Classifies a resolve by what was actually served, so every serve
/// path lands in exactly one bucket: any semantic candidate means the
/// semantic tier answered; otherwise the best (first) lexical score
/// decides confident versus fragment-weak; nothing at all is a miss.
fn resolve_tier_of(served: &[TieredResolution]) -> ResolveTier {
    if served.is_empty() {
        ResolveTier::Miss
    } else if served.iter().any(|candidate| candidate.tier == "semantic") {
        ResolveTier::Semantic
    } else if served[0].score >= LEXICAL_CONFIDENCE {
        ResolveTier::Lexical
    } else {
        ResolveTier::WeakLexical
    }
}

pub async fn resolve(
    State(state): State<AppState>,
    Path(name): Path<String>,
    AppJson(request): AppJson<ResolveRequest>,
) -> Response {
    let started_at = Instant::now();
    resolve_with_fallback(&state, &name, &request, false, started_at)
}

pub async fn resolve_label(
    State(state): State<AppState>,
    Path(name): Path<String>,
    AppJson(request): AppJson<ResolveRequest>,
) -> Response {
    let started_at = Instant::now();
    resolve_with_fallback(&state, &name, &request, true, started_at)
}

/// What one embedding refresh accomplished.
#[derive(Serialize)]
pub struct RefreshOutcome {
    pub embedded: usize,
    pub total: usize,
}

pub async fn refresh_embeddings(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Response {
    let started_at = Instant::now();
    if !state.embeddings_configured() {
        return error(
            StatusCode::NOT_IMPLEMENTED,
            "no embedding provider is configured (set TAGURU_EMBED_URL and TAGURU_EMBED_MODEL)",
            started_at,
        );
    }
    // Refresh batches can talk to the provider for seconds; keep the
    // runtime's workers unstarved while this one blocks.
    match tokio::task::block_in_place(|| state.refresh_embeddings(&name)) {
        None => not_found(&name, started_at),
        Some(Ok((embedded, total))) => ok(RefreshOutcome { embedded, total }, started_at),
        Some(Err(message)) => error(
            StatusCode::BAD_GATEWAY,
            format!("embedding refresh failed: {message}"),
            started_at,
        ),
    }
}

pub async fn labels(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| {
        context
            .labels()
            .into_iter()
            .map(String::from)
            .collect::<Vec<String>>()
    }) {
        Ok(result) => ok(result, started_at),
        Err(failure) => access_error(&state, failure, &name, started_at),
    }
}

#[derive(Debug, Deserialize)]
pub struct UnreachableFromRequest {
    pub origins: Vec<String>,
    /// Omitted means 100, capped at 1000 — the audit pages exactly
    /// like recall and query, `total` telling the whole story.
    pub limit: Option<usize>,
}

pub async fn unreachable_from(
    State(state): State<AppState>,
    Path(name): Path<String>,
    AppJson(request): AppJson<UnreachableFromRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| {
        let origins: Vec<&str> = request.origins.iter().map(String::as_str).collect();
        context.unreachable_from(&origins)
    }) {
        Ok(result) => ok(page(result, request.limit), started_at),
        Err(failure) => access_error(&state, failure, &name, started_at),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assoc(object: &str, weight: f64) -> Association {
        Association {
            subject: "s".to_string(),
            label: "l".to_string(),
            object: object.to_string(),
            weight,
            attributions: Vec::new(),
        }
    }

    #[test]
    fn resolve_tier_classification_reads_the_served_payload() {
        let candidate = |score: f64, tier: &'static str| TieredResolution {
            name: "n".to_string(),
            score,
            tier,
        };

        assert_eq!(resolve_tier_of(&[]), ResolveTier::Miss);
        // Any semantic candidate marks the whole answer semantic, even
        // behind weak lexical ones (lexical always keeps the front).
        assert_eq!(
            resolve_tier_of(&[candidate(0.2, "lexical"), candidate(0.55, "semantic")]),
            ResolveTier::Semantic
        );
        assert_eq!(
            resolve_tier_of(&[candidate(0.9, "lexical"), candidate(0.2, "lexical")]),
            ResolveTier::Lexical
        );
        // The confidence boundary itself counts as confident.
        assert_eq!(
            resolve_tier_of(&[candidate(LEXICAL_CONFIDENCE, "lexical")]),
            ResolveTier::Lexical
        );
        assert_eq!(
            resolve_tier_of(&[candidate(0.11, "lexical")]),
            ResolveTier::WeakLexical
        );
    }

    #[test]
    fn clamp_fills_the_default_and_never_exceeds_the_ceiling() {
        assert_eq!(clamp(None, 100, 1000), 100);
        assert_eq!(clamp(Some(5), 100, 1000), 5);
        assert_eq!(clamp(Some(1_000_000_000), 100, 1000), 1000);
        // explore's exact case: an UNBOUNDED default is itself capped.
        assert_eq!(clamp(None, usize::MAX, 10), 10);
    }

    #[test]
    fn page_clamps_an_explicit_limit_to_the_hard_ceiling() {
        let matches: Vec<Association> = (0..MAX_MATCH_LIMIT + 5)
            .map(|i| assoc(&format!("o{i}"), 1.0))
            .collect();
        let paged = page(matches, Some(1_000_000_000));
        assert_eq!(paged.total, MAX_MATCH_LIMIT + 5);
        assert_eq!(paged.matches.len(), MAX_MATCH_LIMIT);
    }

    #[test]
    fn page_keeps_insertion_order_within_limit_and_strongest_past_it() {
        let matches = vec![assoc("a", 1.0), assoc("b", -3.0), assoc("c", 2.0)];

        // Within the limit nothing is reordered or dropped.
        let within = page(matches.clone(), Some(3));
        assert_eq!(within.total, 3);
        let objects: Vec<&str> = within.matches.iter().map(|m| m.object.as_str()).collect();
        assert_eq!(objects, vec!["a", "b", "c"]);

        // Past it, magnitude ranks — the negative fact is the strongest
        // knowledge — and total still reports the untruncated count.
        let truncated = page(matches, Some(2));
        assert_eq!(truncated.total, 3);
        let objects: Vec<&str> = truncated
            .matches
            .iter()
            .map(|m| m.object.as_str())
            .collect();
        assert_eq!(objects, vec!["b", "c"]);
    }

    #[test]
    fn merge_keeps_lexical_first_and_deduplicates_names() {
        let lexical = vec![
            Resolution {
                name: "杜氏の職".to_string(),
                score: 0.33,
            },
            Resolution {
                name: "蔵".to_string(),
                score: 0.25,
            },
        ];
        let semantic = vec![
            ("蔵人".to_string(), 0.55_f32),
            ("蔵".to_string(), 0.48),
            ("杜氏の職".to_string(), 0.41),
        ];
        let merged = merge_tiers(lexical, semantic);
        let view: Vec<(&str, &str)> = merged
            .iter()
            .map(|candidate| (candidate.name.as_str(), candidate.tier))
            .collect();
        assert_eq!(
            view,
            vec![
                ("杜氏の職", "lexical"),
                ("蔵", "lexical"),
                ("蔵人", "semantic"),
            ]
        );
    }
}
