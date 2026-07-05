//! HTTP surface of the retrieval service: thin JSON adapters, one per
//! `Context` operation. The server adds transport, naming, and lifecycle
//! around the library — never retrieval semantics of its own — so each
//! handler is a lock, a library call, and a serialized reply.

use std::collections::BTreeMap;
use std::time::Instant;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use taguru::context::{Association, Context, Recollection, Resolution};

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

fn access_error(failure: AccessError, name: &str, started_at: Instant) -> Response {
    match failure {
        AccessError::NotFound => not_found(name, started_at),
        AccessError::Load(message) => error(StatusCode::INTERNAL_SERVER_ERROR, message, started_at),
        AccessError::Unpersisted(message) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("write not persisted (nothing was applied): {message}"),
            started_at,
        ),
    }
}

/// The routing directory, skills-style: name, prose description, and
/// stats for every context, so an LLM client can decide where to search
/// (and where to ingest) without the server owning that judgement.
pub async fn list_contexts(State(state): State<AppState>) -> Response {
    let started_at = Instant::now();
    ok(state.directory(), started_at)
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
    body: Option<Json<CreateContextRequest>>,
) -> Response {
    let started_at = Instant::now();
    let Json(request) = body.unwrap_or_default();
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
    pub dice_floor: Option<f64>,
    pub semantic_floor: Option<f32>,
}

pub async fn update_context(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<UpdateContextRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.update_meta(
        &name,
        request.description,
        request.pinned,
        request.dice_floor,
        request.semantic_floor,
    ) {
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

/// A batch of assertions — the arguments of `associate` /
/// `associate_from` as JSON fields ([`AssocOp`], shared with the WAL).
/// A batch is the natural write unit: one document's extracted facts,
/// one request, one lock acquisition, one WAL fsync.
pub async fn add_associations(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(associations): Json<Vec<AssocOp>>,
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
    }
    let total = associations.len();
    match state.add_associations(&name, associations) {
        Err(failure) => access_error(failure, &name, started_at),
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
    Json(request): Json<AliasRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.add_aliases(&name, &request.concepts, &request.labels) {
        Err(failure) => access_error(failure, &name, started_at),
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
    body: Option<Json<VocabularyAuditRequest>>,
) -> Response {
    let started_at = Instant::now();
    let Json(request) = body.unwrap_or_default();
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
        Err(failure) => return access_error(failure, &name, started_at),
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
    Json(request): Json<RetractSourceRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.retract_source(&name, &request.source) {
        Err(failure) => access_error(failure, &name, started_at),
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
    Json(request): Json<SearchPassagesRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.search_passages(
        &name,
        &request.query,
        clamp(request.limit, 5, MAX_MATCH_LIMIT),
    ) {
        None => not_found(&name, started_at),
        Some(hits) => ok(
            hits.into_iter()
                .map(|(source, score, text)| PassageHit {
                    source,
                    score,
                    text,
                })
                .collect::<Vec<_>>(),
            started_at,
        ),
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
    Json(request): Json<ExploreRequest>,
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
            ok(ExplorePage { total, matches }, started_at)
        }
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
            clamp(request.limit, 20, MAX_MATCH_LIMIT),
        )
    }) {
        Ok(result) => ok(result, started_at),
        Err(failure) => access_error(failure, &name, started_at),
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
        Err(failure) => return access_error(failure, name, started_at),
    };
    let confident = lexical
        .first()
        .is_some_and(|best| best.score >= LEXICAL_CONFIDENCE);
    if confident {
        return ok(lexical_tier(lexical), started_at);
    }

    // The provider round trip can take hundreds of milliseconds; tell
    // the runtime this thread will block so other tasks migrate off it.
    match tokio::task::block_in_place(|| {
        state.semantic_resolve(name, &request.cue, labels, request.semantic_floor)
    }) {
        None => not_found(name, started_at),
        Some(Ok(semantic)) => ok(merge_tiers(lexical, semantic), started_at),
        // The semantic tier is enrichment once ANY lexical candidate
        // exists: degrade to the weak lexical results and log, rather
        // than failing an answerable request over a provider hiccup.
        Some(Err(message)) if !lexical.is_empty() => {
            tracing::warn!("semantic entry failed (serving weak lexical results): {message}");
            ok(lexical_tier(lexical), started_at)
        }
        Some(Err(message)) => {
            tracing::warn!("semantic entry failed with nothing lexical to fall back on: {message}");
            error(
                StatusCode::BAD_GATEWAY,
                format!("semantic entry failed: {message}"),
                started_at,
            )
        }
    }
}

pub async fn resolve(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<ResolveRequest>,
) -> Response {
    let started_at = Instant::now();
    resolve_with_fallback(&state, &name, &request, false, started_at)
}

pub async fn resolve_label(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(request): Json<ResolveRequest>,
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
        Err(failure) => access_error(failure, &name, started_at),
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
    Json(request): Json<UnreachableFromRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.read_context(&name, |context| {
        let origins: Vec<&str> = request.origins.iter().map(String::as_str).collect();
        context.unreachable_from(&origins)
    }) {
        Ok(result) => ok(page(result, request.limit), started_at),
        Err(failure) => access_error(failure, &name, started_at),
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
