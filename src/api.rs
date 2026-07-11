//! HTTP surface of the retrieval service: thin JSON adapters, one per
//! `Context` operation. The server adds transport, naming, and lifecycle
//! around the library — never retrieval semantics of its own — so each
//! handler is a lock, a library call, and a serialized reply.

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Path, Request, State};
use axum::http::{Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use taguru::context::{Activation, Association, Attribution, Context, Recollection, Resolution};

use crate::metrics::{ErrorKind, ResolveTier, SearchOp};
use crate::registry::{AccessError, AppState, AssocOp, ContextMeta, CreateError};

mod sources;
pub use sources::{citation, list_sources, lookup_passages, retract_source, search_passages};

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

/// The credential behind a request, for the `taguru::audit` lines —
/// "-" when auth is off, mirroring the access log's key column. The
/// auth layer stamps [`crate::auth::AuthKey`] onto the request
/// extensions, so any handler that must say WHO can take it as an
/// optional `Extension`.
pub(crate) fn key_name(key: &Option<axum::Extension<crate::auth::AuthKey>>) -> &str {
    key.as_ref().map_or("-", |extension| extension.0.0.as_ref())
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

/// Whether each retrieval emits a `taguru::search` event line
/// (`TAGURU_LOG_SEARCHES=1`). Off by default ON PURPOSE: cues are the
/// user's memory content, and until now the log stream carried no
/// content at all — copying queries into the log pipeline is a data
/// decision the operator must make, not inherit. When on, the lines
/// feed keyword/phrase analysis downstream (what do clients actually
/// ask, and which cues come back empty) — aggregation belongs to the
/// log system, which is why there is no in-server top-K.
fn search_log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("TAGURU_LOG_SEARCHES")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    })
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

/// A bounded directory page. `total` names the whole directory's
/// count, deliberately cursor-independent — "how many contexts exist",
/// not "how many remain past `after`" — so a truncated view is visible
/// and the figure is stable across pages. (The search endpoints' `total`
/// counts post-filter matches instead: there the query itself defines
/// the population of interest.)
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
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    AppQuery(query): AppQuery<ListContextsQuery>,
) -> Response {
    let started_at = Instant::now();
    let directory = state.directory();
    // A context-scoped key's world IS its grant: rows outside it are
    // filtered rather than refused, and `total` counts the visible
    // world so pagination stays coherent for that caller.
    let directory: Vec<_> = match &scope {
        Some(axum::Extension(scope)) => directory
            .into_iter()
            .filter(|entry| scope.allows_context(&entry.name))
            .collect(),
        None => directory,
    };
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
/// protocol at connect time, the way it would read a skill. The static
/// text carries this server's live configuration as a trailer, so the
/// agent also learns which optional tiers are actually on.
pub async fn protocol(trailer: Option<String>) -> Response {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/markdown; charset=utf-8",
        )],
        protocol_text(trailer.as_deref()),
    )
        .into_response()
}

/// The manual as served: static text plus the live-configuration
/// trailer. Shared with the MCP transports, whose `initialize` must
/// hand out exactly what GET /protocol serves.
pub fn protocol_text(trailer: Option<&str>) -> String {
    let mut body = include_str!("llm-protocol.md").to_string();
    if let Some(trailer) = trailer {
        body.push_str(trailer);
    }
    body
}

/// The `## This server` trailer behind [`protocol`]: the runtime facts
/// an agent acts on differently — today, whether the semantic tier is
/// live and who runs the gloss refresh. The static manual cannot say
/// this ("servers with embeddings only"), and an agent that cannot see
/// that embeddings are on never calls `refresh_embeddings`, leaving
/// the tier dark over a fully configured provider.
pub fn protocol_trailer(embed_model: Option<&str>, auto_embed: bool) -> Option<String> {
    let model = embed_model?;
    let refresh_note = if auto_embed {
        "This server auto-refreshes embeddings shortly after each write \
         settles; calling `refresh_embeddings` yourself only buys \
         immediacy."
    } else {
        "Nothing embeds glosses automatically here: finish every ingest \
         and alias fix by calling `refresh_embeddings` on the context \
         you touched, or its new names stay invisible to the semantic \
         tier."
    };
    Some(format!(
        "\n---\n\n## This server\n\nSemantic entry is ON (embedding model `{model}`): `resolve` falls \
         back to embedded glosses, and `refresh_embeddings` and \
         `audit_vocabulary`'s semantic pass are live. {refresh_note}\n"
    ))
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
        Err(CreateError::InvalidName) => error(
            StatusCode::BAD_REQUEST,
            "the context name must not be empty".to_string(),
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

pub async fn delete_context(
    State(state): State<AppState>,
    Path(name): Path<String>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
) -> Response {
    let started_at = Instant::now();
    match state.delete(&name) {
        None => not_found(&name, started_at),
        Some(outcome) => {
            // Every destructive operation leaves one self-contained
            // `taguru::audit` line — who, what, to which object — so an
            // incident review greps one target instead of reconstructing
            // objects from route templates. Logged on the failed-unlink
            // arm too: the context is gone from the API either way.
            tracing::info!(
                target: "taguru::audit",
                key = %key_name(&key),
                context = %name,
                files_removed = outcome.is_ok(),
                "context deleted",
            );
            match outcome {
                Ok(()) => ok(true, started_at),
                Err(io_error) => {
                    state.metrics().record_error(ErrorKind::Io);
                    error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!(
                            "context '{name}' removed but its files were not: {io_error} \
                             (a deletion marker remains; the next boot resumes the removal)"
                        ),
                        started_at,
                    )
                }
            }
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
        // subject/label/object name the triple itself and must carry
        // something. A source may be OMITTED — that is the ordinary
        // unsourced-association case (see AssocOp::source) — but a
        // source that is PRESENT is a name like any other: an empty
        // string would intern a real, permanent source id that every
        // later attribution list displays and that unrelated callers'
        // mistakes silently merge into.
        for (field, value) in [
            ("subject", Some(op.subject.as_str())),
            ("label", Some(op.label.as_str())),
            ("object", Some(op.object.as_str())),
            ("source", op.source.as_deref()),
        ] {
            let Some(value) = value else {
                continue;
            };
            if let Some(refusal) =
                empty(&format!("associations[{index}].{field}"), value, started_at)
            {
                return refusal;
            }
        }
    }
    let total = associations.len();
    // The write stages ops in the WAL and fsyncs before returning, so
    // keep it off the async worker like `store_passages` and the flush.
    match tokio::task::block_in_place(|| state.add_associations(&name, associations)) {
        Err(failure) => access_error(&state, failure, &name, started_at),
        Ok(Ok(applied)) => {
            // An empty batch reaches here as `applied == 0`: nothing
            // was written, so the counter must not move either — the
            // same rule the partial-write arm below already applies.
            if applied > 0 {
                state.note_write(&name);
            }
            ok(applied, started_at)
        }
        // Items before the failing one are applied (each item is
        // all-or-nothing in the library); report how far the batch got.
        Ok(Err(partial)) => {
            if partial.applied > 0 {
                state.note_write(&name);
            }
            error(
                StatusCode::INSUFFICIENT_STORAGE,
                format!(
                    "applied {} of {total} associations, then: {}",
                    partial.applied, partial.message
                ),
                started_at,
            )
        }
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
/// one request; anything past this is asked to split. (`pub(crate)`:
/// the offline import chunks its applies at the same size.)
pub(crate) const MAX_ASSOCIATIONS_PER_REQUEST: usize = 10_000;

/// Per-request cap on list-shaped read inputs: origins, query terms,
/// source lists. The per-item work happens under the context's lock,
/// and for `lookup_passages` the response body scales with the list
/// too — the output-side clamps cannot bound what the request itself
/// carries, so the input is refused up front like an oversized
/// association batch. It matches the largest page any read endpoint
/// serves, so a paged bulk workflow (list_sources → lookup_passages)
/// fits exactly.
const MAX_INPUT_ITEMS: usize = 1000;

/// Refuses a list-shaped input field longer than [`MAX_INPUT_ITEMS`],
/// before any lock is taken. `None` means the length is fine.
fn overlong(field: &str, len: usize, started_at: Instant) -> Option<Response> {
    (len > MAX_INPUT_ITEMS).then(|| {
        error(
            StatusCode::BAD_REQUEST,
            format!(
                "{field} carries {len} items, past the per-request limit of \
                 {MAX_INPUT_ITEMS}; split the request"
            ),
            started_at,
        )
    })
}

/// Per-op weight ceiling (absolute value). Weights accumulate on an
/// edge across writes and floats saturate: two f64::MAX writes made an
/// edge +Infinity, and a later retract minted Inf − Inf = NaN — an
/// unreadable, unresettable fact. At ±1e6 per op, saturating would
/// take ~1.8e302 acknowledged writes. (`pub(crate)`: the offline
/// import enforces the same caps, file lines instead of requests.)
pub(crate) const MAX_ASSOCIATION_WEIGHT: f64 = 1e6;

/// Byte cap on every name-shaped write: subject/label/object, source
/// ids, aliases. Names are entry keys, not documents — the graph
/// interns every distinct spelling forever, and the top-concepts
/// snapshot carries them into every directory listing (one 4 MB
/// subject made every GET /contexts response 4 MB, resident outside
/// the cache budget).
pub(crate) const MAX_NAME_BYTES: usize = 1024;

/// Byte cap on a context name: it becomes a file stem, percent-encoded
/// at up to 3× — 64 bytes keeps the longest sidecar filename well
/// under every filesystem's 255-byte limit.
pub(crate) const MAX_CONTEXT_NAME_BYTES: usize = 64;

/// Byte cap on a context description — it too rides in every
/// directory listing.
pub(crate) const MAX_DESCRIPTION_BYTES: usize = 4096;

/// Byte cap on one doc2query question — a search phrase, not a
/// document; anything longer is misuse of the field.
pub(crate) const MAX_QUESTION_BYTES: usize = 512;

/// How many stored questions one paragraph may carry (shared with
/// `taguru extract --questions`, whose N cannot exceed it). Each
/// question is an embedded row spending the vector-limit budget;
/// past a handful they stop adding recall and start crowding it out.
pub(crate) const MAX_QUESTIONS_PER_PARAGRAPH: usize = 8;

/// Byte cap on one section label — a heading, not a document; same
/// bound as a question since both are short strings riding a
/// paragraph index. Unlike questions, sections are never embedded, so
/// there is no per-paragraph cap to match `MAX_QUESTIONS_PER_PARAGRAPH`.
pub(crate) const MAX_SECTION_BYTES: usize = 512;

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

/// Companion to `oversized`, at the other end of the range: `Some(400)`
/// when `value` is empty. An empty subject/label/object is not a
/// degenerate name, it is no name — silently interning it would seed
/// the graph with an unaddressable concept that every future listing
/// and export carries but no query can usefully name.
fn empty(what: &str, value: &str, started_at: Instant) -> Option<Response> {
    value.is_empty().then(|| {
        error(
            StatusCode::BAD_REQUEST,
            format!("{what} must not be empty; nothing was applied"),
            started_at,
        )
    })
}

/// Companion to `oversized`/`empty`: `Some(400)` when `source` names a
/// passage the request itself does not carry alongside it — a question
/// or section cannot attach to text the request does not carry.
fn orphaned_source(
    what: &str,
    source: &str,
    passages: &BTreeMap<String, String>,
    started_at: Instant,
) -> Option<Response> {
    (!passages.contains_key(source)).then(|| {
        error(
            StatusCode::BAD_REQUEST,
            format!("{what} for '{source}' arrived without a passage for it in this request"),
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
    pub matches: Vec<AssociationOut>,
}

/// Bounds a match list: within the limit the library's insertion order
/// is preserved; past it, the strongest knowledge (|weight|, the same
/// magnitude-ranks philosophy as activate) survives the cut, ties in
/// insertion order. Returns the raw library `Association`s, not yet
/// resolved to their wire shape — callers still need to run them
/// through `resolve_sections`/`association_out`, and `page` itself has
/// no context name to resolve against.
fn page(mut matches: Vec<Association>, limit: Option<usize>) -> (usize, Vec<Association>) {
    let total = matches.len();
    let limit = clamp(limit, DEFAULT_MATCH_LIMIT, MAX_MATCH_LIMIT);
    if total > limit {
        matches.sort_by(|a, b| b.weight.abs().total_cmp(&a.weight.abs()));
        matches.truncate(limit);
    }
    (total, matches)
}

/// `Attribution`'s wire shape: everything the library exposes, plus the
/// section label the server resolves from `paragraph` via
/// `AppState::resolve_sections`. `null` when the attribution carries no
/// paragraph locator, or when the locator falls outside every section
/// marker the ingest batch recorded for that source — never a
/// fabricated label.
#[derive(Serialize)]
pub struct AttributionOut {
    pub source: String,
    /// This source's raw cumulative contribution, NOT averaged — divide
    /// by `count` for its per-assertion average. Contrast with the
    /// enclosing [`AssociationOut::weight`], which is averaged.
    pub weight: f64,
    pub count: u64,
    pub paragraph: Option<u32>,
    pub section: Option<String>,
}

/// `Association`'s wire shape: identical, except its attributions carry
/// a resolved section label (see [`AttributionOut`]).
#[derive(Serialize)]
pub struct AssociationOut {
    pub subject: String,
    pub label: String,
    pub object: String,
    /// The averaged weight per assertion (`sum / count`) — unlike
    /// [`AttributionOut::weight`], which is each source's raw sum.
    pub weight: f64,
    pub count: u64,
    pub attributions: Vec<AttributionOut>,
}

fn attribution_out(
    attribution: Attribution,
    sections: &HashMap<(String, u32), String>,
) -> AttributionOut {
    let section = attribution
        .paragraph
        .and_then(|paragraph| sections.get(&(attribution.source.clone(), paragraph)))
        .cloned();
    AttributionOut {
        source: attribution.source,
        weight: attribution.weight,
        count: attribution.count,
        paragraph: attribution.paragraph,
        section,
    }
}

fn association_out(
    association: Association,
    sections: &HashMap<(String, u32), String>,
) -> AssociationOut {
    AssociationOut {
        subject: association.subject,
        label: association.label,
        object: association.object,
        weight: association.weight,
        count: association.count,
        attributions: association
            .attributions
            .into_iter()
            .map(|attribution| attribution_out(attribution, sections))
            .collect(),
    }
}

/// `Recollection`'s wire shape: identical, with `association` reshaped
/// to carry resolved section labels (see [`AssociationOut`]).
#[derive(Serialize)]
pub struct RecollectionOut {
    pub distance: usize,
    pub path: Vec<String>,
    pub association: AssociationOut,
}

fn recollection_out(
    recollection: Recollection,
    sections: &HashMap<(String, u32), String>,
) -> RecollectionOut {
    RecollectionOut {
        distance: recollection.distance,
        path: recollection.path,
        association: association_out(recollection.association, sections),
    }
}

/// `Activation`'s wire shape: identical, with `association` reshaped to
/// carry resolved section labels (see [`AssociationOut`]).
#[derive(Serialize)]
pub struct ActivationOut {
    pub strength: f64,
    pub path: Vec<String>,
    pub association: AssociationOut,
}

fn activation_out(
    activation: Activation,
    sections: &HashMap<(String, u32), String>,
) -> ActivationOut {
    ActivationOut {
        strength: activation.strength,
        path: activation.path,
        association: association_out(activation.association, sections),
    }
}

/// Every `(source, paragraph)` locator across a batch of associations'
/// attributions — the key set for one `resolve_sections` call, so a
/// page of N matches loads the passage store once rather than once per
/// attribution. Attributions without a paragraph locator contribute
/// nothing; they can never resolve to a section.
fn locator_keys<'a>(
    associations: impl Iterator<Item = &'a Association> + 'a,
) -> impl Iterator<Item = (String, u32)> + 'a {
    associations.flat_map(|association| {
        association.attributions.iter().filter_map(|attribution| {
            attribution
                .paragraph
                .map(|paragraph| (attribution.source.clone(), paragraph))
        })
    })
}

/// Resolve section labels for a page of associations and convert each
/// to its wire shape — the `resolve_sections` + `locator_keys` + map
/// sequence every association-returning endpoint needs.
fn associations_out(
    state: &AppState,
    name: &str,
    matches: Vec<Association>,
) -> Vec<AssociationOut> {
    let sections = state.resolve_sections(name, locator_keys(matches.iter()));
    matches
        .into_iter()
        .map(|association| association_out(association, &sections))
        .collect()
}

/// Same as [`associations_out`], for recollections (explore's results).
fn recollections_out(
    state: &AppState,
    name: &str,
    matches: Vec<Recollection>,
) -> Vec<RecollectionOut> {
    let sections = state.resolve_sections(
        name,
        locator_keys(matches.iter().map(|recollection| &recollection.association)),
    );
    matches
        .into_iter()
        .map(|recollection| recollection_out(recollection, &sections))
        .collect()
}

/// Same as [`associations_out`], for activations (activate's results).
fn activations_out(state: &AppState, name: &str, matches: Vec<Activation>) -> Vec<ActivationOut> {
    let sections = state.resolve_sections(
        name,
        locator_keys(matches.iter().map(|activation| &activation.association)),
    );
    matches
        .into_iter()
        .map(|activation| activation_out(activation, &sections))
        .collect()
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

/// One page of the alias table: `total` counts BOTH namespaces before
/// paging, the maps carry this page's entries. The page cursor spans
/// the namespaces in order — concepts (sorted by alias), then labels —
/// so the next page starts after the last entry shown:
/// `?after=concept:<alias>` or `?after=label:<alias>`.
#[derive(Serialize)]
pub struct AliasExport {
    pub total: usize,
    pub concepts: BTreeMap<String, String>,
    pub labels: BTreeMap<String, String>,
}

/// `?limit=&after=` — the keyset page every unbounded listing takes
/// (default and ceiling [`MAX_MATCH_LIMIT`], like the directory).
#[derive(Debug, Deserialize)]
pub struct KeysetQuery {
    pub limit: Option<usize>,
    pub after: Option<String>,
}

pub async fn add_aliases(
    State(state): State<AppState>,
    Path(name): Path<String>,
    AppJson(request): AppJson<AliasRequest>,
) -> Response {
    let started_at = Instant::now();
    // Refused before the write lock like an association batch — each
    // pair is one WAL op in a single lock-hold/fsync, the same cost
    // shape MAX_ASSOCIATIONS_PER_REQUEST bounds there.
    let pairs = request.concepts.len() + request.labels.len();
    if pairs > MAX_ASSOCIATIONS_PER_REQUEST {
        return error(
            StatusCode::BAD_REQUEST,
            format!(
                "batch of {pairs} aliases exceeds the per-request limit of \
                 {MAX_ASSOCIATIONS_PER_REQUEST}; split the ingest"
            ),
            started_at,
        );
    }
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
                // An empty spelling is worse than an unaddressable
                // name: `str::contains("")` is always true, so a
                // zero-length alias would containment-match every cue
                // and plant a phantom hit in every resolution from
                // then on.
                if let Some(refusal) = empty(&format!("a {namespace} {role}"), value, started_at) {
                    return refusal;
                }
            }
        }
    }
    // Same fsync-bearing WAL write as add_associations; keep it off the
    // async worker.
    match tokio::task::block_in_place(|| {
        state.add_aliases(&name, &request.concepts, &request.labels)
    }) {
        Err(failure) => access_error(&state, failure, &name, started_at),
        Ok(Ok(applied)) => {
            // Same rule as add_associations: an empty batch applies
            // nothing, so it must not bump the write counter either.
            if applied > 0 {
                state.note_write(&name);
            }
            ok(applied, started_at)
        }
        Ok(Err(partial)) => {
            if partial.applied > 0 {
                state.note_write(&name);
            }
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

/// Alias withdrawals — the exact registered spellings, per namespace.
/// Withdrawal is the undo for a mis-registered alias: the spelling
/// stops resolving and is free to register again; canonicals and
/// edges are untouched, and a canonical name is refused (removal
/// must never unname a record).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct RemoveAliasesRequest {
    pub concepts: Vec<String>,
    pub labels: Vec<String>,
}

pub async fn remove_aliases(
    State(state): State<AppState>,
    Path(name): Path<String>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    AppJson(request): AppJson<RemoveAliasesRequest>,
) -> Response {
    let started_at = Instant::now();
    // An empty withdrawal is a malformed request, not a silent no-op.
    if request.concepts.is_empty() && request.labels.is_empty() {
        return error(
            StatusCode::BAD_REQUEST,
            "the request names no aliases to remove",
            started_at,
        );
    }
    // Same WAL-op-per-item cost shape as add_aliases; same cap.
    let names = request.concepts.len() + request.labels.len();
    if names > MAX_ASSOCIATIONS_PER_REQUEST {
        return error(
            StatusCode::BAD_REQUEST,
            format!(
                "batch of {names} alias removals exceeds the per-request limit of \
                 {MAX_ASSOCIATIONS_PER_REQUEST}; split the request"
            ),
            started_at,
        );
    }
    // Same fsync-bearing WAL write; keep it off the async worker.
    match tokio::task::block_in_place(|| {
        state.remove_aliases(&name, &request.concepts, &request.labels)
    }) {
        Err(failure) => access_error(&state, failure, &name, started_at),
        Ok(Ok(removed)) => {
            // Withdrawn spellings live in the body; counts (not the
            // spellings — a batch may run to thousands) reach the
            // audit line, the access log names the context and key.
            tracing::info!(
                target: "taguru::audit",
                key = %key_name(&key),
                context = %name,
                concepts = request.concepts.len(),
                labels = request.labels.len(),
                removed,
                "aliases removed",
            );
            state.note_write(&name);
            ok(removed, started_at)
        }
        Ok(Err(partial)) => {
            if partial.applied > 0 {
                state.note_write(&name);
            }
            // `full` is unreachable for removals (they free, never
            // fill), but the shared mapping stays uniform.
            let status = if partial.full {
                StatusCode::INSUFFICIENT_STORAGE
            } else {
                StatusCode::CONFLICT
            };
            error(
                status,
                format!(
                    "removed {} aliases, then {}",
                    partial.applied, partial.message
                ),
                started_at,
            )
        }
    }
}

pub async fn list_aliases(
    State(state): State<AppState>,
    Path(name): Path<String>,
    AppQuery(query): AppQuery<KeysetQuery>,
) -> Response {
    let started_at = Instant::now();
    // The cursor names a namespace and an alias; anything else is a
    // malformed request, not an empty page.
    let after = match query.after.as_deref() {
        None => None,
        Some(cursor) => match cursor.split_once(':') {
            Some((kind @ ("concept" | "label"), alias)) => Some((kind == "label", alias)),
            _ => {
                return error(
                    StatusCode::BAD_REQUEST,
                    "after must be 'concept:<alias>' or 'label:<alias>' — the last \
                     entry of the previous page",
                    started_at,
                );
            }
        },
    };
    let limit = clamp(query.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT);
    match state.read_context(&name, |context| {
        let mut concepts: Vec<(String, String)> = context
            .concept_aliases()
            .into_iter()
            .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
            .collect();
        concepts.sort();
        let mut labels: Vec<(String, String)> = context
            .label_aliases()
            .into_iter()
            .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
            .collect();
        labels.sort();
        let total = concepts.len() + labels.len();
        // One ordered sequence — concepts, then labels — filtered past
        // the cursor and cut at the limit, then split back into maps.
        let page: Vec<(bool, (String, String))> = concepts
            .into_iter()
            .map(|entry| (false, entry))
            .chain(labels.into_iter().map(|entry| (true, entry)))
            .filter(|(is_label, (alias, _))| match after {
                None => true,
                Some((after_is_label, after_alias)) => {
                    (*is_label, alias.as_str()) > (after_is_label, after_alias)
                }
            })
            .take(limit)
            .collect();
        let mut export = AliasExport {
            total,
            concepts: BTreeMap::new(),
            labels: BTreeMap::new(),
        };
        for (is_label, (alias, canonical)) in page {
            if is_label {
                export.labels.insert(alias, canonical);
            } else {
                export.concepts.insert(alias, canonical);
            }
        }
        export
    }) {
        Ok(result) => ok(result, started_at),
        Err(failure) => access_error(&state, failure, &name, started_at),
    }
}

/// Original text passages keyed by source id — the same opaque ids that
/// appear on attributions — plus, optionally, doc2query questions and
/// section markers per source, each naming a paragraph of that
/// source's text IN THIS REQUEST (a question or section cannot attach
/// to text the request does not carry: storage replaces per source,
/// wholesale).
#[derive(Debug, Deserialize)]
pub struct StorePassagesRequest {
    pub passages: BTreeMap<String, String>,
    #[serde(default)]
    pub questions: BTreeMap<String, Vec<QuestionSpec>>,
    #[serde(default)]
    pub sections: BTreeMap<String, Vec<SectionSpec>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuestionSpec {
    pub paragraph: u32,
    pub question: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SectionSpec {
    pub paragraph: u32,
    pub section: String,
}

/// What a passage store accomplished. `stored` counts the batch (the
/// historical number, now named); the question and section tallies
/// report doc2query/section bookkeeping — a dropped question or
/// section named a paragraph that does not exist in the text it rode
/// in with.
#[derive(Serialize)]
pub struct StoredPassages {
    pub stored: usize,
    pub questions_stored: usize,
    pub questions_dropped: usize,
    pub sections_stored: usize,
    pub sections_dropped: usize,
}

pub async fn store_passages(
    State(state): State<AppState>,
    Path(name): Path<String>,
    AppJson(request): AppJson<StorePassagesRequest>,
) -> Response {
    let started_at = Instant::now();
    // Source ids are names (the passage text itself is a document and
    // rides under the body cap instead) — and like every name, an
    // empty one is refused: it would list as a blank entry in GET
    // /sources and answer lookups nothing sensible ever asks.
    for source in request.passages.keys() {
        if let Some(refusal) = oversized("a passage source id", source, MAX_NAME_BYTES, started_at)
        {
            return refusal;
        }
        if let Some(refusal) = empty("a passage source id", source, started_at) {
            return refusal;
        }
    }
    // Question structure is the request's to get right: sources must
    // name passages in THIS request, sizes and per-paragraph counts
    // stay under the shared caps. (Whether a paragraph index exists in
    // the text is settled at store time, one rule for every entrance.)
    for (source, questions) in &request.questions {
        if let Some(refusal) = orphaned_source("questions", source, &request.passages, started_at) {
            return refusal;
        }
        let mut per_paragraph: BTreeMap<u32, usize> = BTreeMap::new();
        for spec in questions {
            if spec.question.len() > MAX_QUESTION_BYTES {
                return error(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "a question for '{source}' is {} bytes; questions are capped at \
                         {MAX_QUESTION_BYTES} bytes",
                        spec.question.len()
                    ),
                    started_at,
                );
            }
            // Embedded verbatim on the next refresh, where providers
            // refuse zero-length input — which would stall the whole
            // refresh pass at this row, every pass.
            if let Some(refusal) = empty(
                &format!("a question for '{source}'"),
                &spec.question,
                started_at,
            ) {
                return refusal;
            }
            let count = per_paragraph.entry(spec.paragraph).or_insert(0);
            *count += 1;
            if *count > MAX_QUESTIONS_PER_PARAGRAPH {
                return error(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "paragraph {} of '{source}' carries more than \
                         {MAX_QUESTIONS_PER_PARAGRAPH} questions",
                        spec.paragraph
                    ),
                    started_at,
                );
            }
        }
    }
    // Section structure follows questions' rule: sources must name
    // passages in THIS request, sizes stay under the shared cap.
    // (Whether a paragraph index exists in the text is settled at
    // store time, same as questions; ingest's batch format has no
    // per-paragraph section count limit either, so neither does this.)
    for (source, sections) in &request.sections {
        if let Some(refusal) = orphaned_source("sections", source, &request.passages, started_at) {
            return refusal;
        }
        for spec in sections {
            if spec.section.len() > MAX_SECTION_BYTES {
                return error(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "a section label for '{source}' is {} bytes; section labels are \
                         capped at {MAX_SECTION_BYTES} bytes",
                        spec.section.len()
                    ),
                    started_at,
                );
            }
            if let Some(refusal) = empty(
                &format!("a section label for '{source}'"),
                &spec.section,
                started_at,
            ) {
                return refusal;
            }
        }
    }
    let mut questions = request.questions;
    let mut sections = request.sections;
    let passages: BTreeMap<String, crate::passages::PassageSubmission> = request
        .passages
        .into_iter()
        .map(|(source, text)| {
            let questions = questions
                .remove(&source)
                .unwrap_or_default()
                .into_iter()
                .map(|spec| (spec.paragraph, spec.question))
                .collect();
            let sections = sections
                .remove(&source)
                .unwrap_or_default()
                .into_iter()
                .map(|spec| (spec.paragraph, spec.section))
                .collect();
            (
                source,
                crate::passages::PassageSubmission {
                    text,
                    questions,
                    sections,
                },
            )
        })
        .collect();
    // Off the async worker: the store fsyncs its log, and folding the
    // new paragraphs into a resident index tokenizes them.
    let outcome = tokio::task::block_in_place(|| state.store_passages(&name, passages));
    match outcome {
        None => not_found(&name, started_at),
        Some(Ok(outcome)) => {
            state.note_write(&name);
            ok(
                StoredPassages {
                    stored: outcome.stored,
                    questions_stored: outcome.questions_stored,
                    questions_dropped: outcome.questions_dropped,
                    sections_stored: outcome.sections_stored,
                    sections_dropped: outcome.sections_dropped,
                },
                started_at,
            )
        }
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

/// What `POST /import` accomplished — the same numbers the CLI's
/// per-file report line carries.
#[derive(Serialize)]
pub struct ImportOutcome {
    pub context: String,
    pub source: String,
    pub created: bool,
    pub retracted: usize,
    pub associations: usize,
    pub aliases: usize,
    pub passage_stored: bool,
    /// A stored passage for this source was retracted with no replacement
    /// in the batch — `passage_stored: false` alone cannot tell "never
    /// had one" from "just erased one", so callers get this explicitly.
    pub passage_dropped: bool,
    pub questions_stored: usize,
    pub questions_dropped: usize,
    pub sections_stored: usize,
    pub sections_dropped: usize,
    /// Association paragraph locators dropped for naming a spot the
    /// batch's passage split does not have — the association itself
    /// still landed. Reported like `questions_dropped`/`sections_dropped`.
    pub association_paragraphs_dropped: usize,
}

/// What a multi-batch `POST /import` stream accomplished — one
/// [`ImportOutcome`] per batch, in stream order. Only streams answer
/// this shape; a single-batch body keeps the original bare outcome.
#[derive(Serialize)]
pub struct ImportStreamOutcome {
    pub batches: Vec<ImportOutcome>,
}

fn import_outcome(batch: &crate::ingest::Batch, applied: &crate::ingest::Applied) -> ImportOutcome {
    ImportOutcome {
        context: batch.context.clone(),
        source: batch.source.clone(),
        created: applied.created,
        retracted: applied.retracted,
        associations: applied.associations,
        aliases: applied.aliases,
        passage_stored: applied.passage_stored,
        passage_dropped: applied.passage_dropped,
        questions_stored: applied.questions_stored,
        questions_dropped: applied.questions_dropped,
        sections_stored: applied.sections_stored,
        sections_dropped: applied.sections_dropped,
        association_paragraphs_dropped: applied.association_paragraphs_dropped,
    }
}

/// Maps one batch's [`ApplyRefusal`](crate::ingest::ApplyRefusal) onto
/// the response, `note` naming which batch of a stream refused (empty
/// for a single-batch body, keeping that path's responses exactly as
/// they were).
fn import_refusal(
    state: &AppState,
    batch: &crate::ingest::Batch,
    refusal: crate::ingest::ApplyRefusal,
    note: &str,
    started_at: Instant,
) -> Response {
    match refusal {
        crate::ingest::ApplyRefusal::Access(failure) if note.is_empty() => {
            access_error(state, failure, &batch.context, started_at)
        }
        // The prefixed twin of access_error's arms — same statuses,
        // same metrics, the batch note in front.
        crate::ingest::ApplyRefusal::Access(AccessError::NotFound) => error(
            StatusCode::NOT_FOUND,
            format!("{note}context '{}' not found", batch.context),
            started_at,
        ),
        crate::ingest::ApplyRefusal::Access(AccessError::Load(message)) => {
            state.metrics().record_error(ErrorKind::Load);
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("{note}{message}"),
                started_at,
            )
        }
        crate::ingest::ApplyRefusal::Access(AccessError::Unpersisted(message)) => {
            state.metrics().record_error(ErrorKind::WalRefused);
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("{note}write not persisted (nothing was applied): {message}"),
                started_at,
            )
        }
        refusal @ crate::ingest::ApplyRefusal::NoContext(_) => error(
            StatusCode::NOT_FOUND,
            format!("{note}{}", refusal.text()),
            started_at,
        ),
        refusal @ crate::ingest::ApplyRefusal::Io(_) => {
            state.metrics().record_error(ErrorKind::Io);
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("{note}{}", refusal.text()),
                started_at,
            )
        }
        crate::ingest::ApplyRefusal::Partial {
            applied,
            message,
            full,
        } => {
            // The batch got partway: what landed counts as a write,
            // and the status keeps the capacity/conflict distinction
            // every partial write reports.
            if applied > 0 {
                state.note_write(&batch.context);
            }
            let status = if full {
                StatusCode::INSUFFICIENT_STORAGE
            } else {
                StatusCode::CONFLICT
            };
            error(status, format!("{note}{message}"), started_at)
        }
    }
}

/// `POST /import` — the batch-file contract (docs/import.html) over
/// HTTP: the body IS one batch file — or a whole stream of batches,
/// as `GET /contexts/{name}/export` renders — applied to the live
/// server with the same validate-first, retract-then-apply semantics
/// as `taguru import`. Each batch states one source's complete truth,
/// so bulk loads and restores reach a running server without a
/// downtime window. The body cap, auth, timeout, and rate limit apply
/// as on any endpoint; embeddings ride the next flush
/// (`TAGURU_EMBED_AUTO`) exactly as live writes do.
///
/// A single-batch body answers with that batch's outcome; a stream
/// answers `{batches: [...]}` in stream order. A refusal partway
/// through a stream stops there — the batches before it landed
/// durably, and because every batch is retract-then-apply,
/// re-POSTing the whole corrected stream is exact, never
/// double-counted.
pub async fn import_batch(
    State(state): State<AppState>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    body: axum::body::Bytes,
) -> Response {
    let started_at = Instant::now();
    let batches = match crate::ingest::parse_batches(&body[..]) {
        Ok(batches) => batches,
        // Line-numbered, like the CLI's validation pass.
        Err(message) => return error(StatusCode::BAD_REQUEST, message, started_at),
    };
    // Import's contexts live in the BODY, out of the route-level
    // authorization check's reach — a context-scoped key is judged
    // here instead, before anything applies.
    if let Some(axum::Extension(scope)) = &scope
        && let Some(refused) = batches
            .iter()
            .find(|batch| !scope.allows_context(&batch.context))
    {
        return error(
            StatusCode::FORBIDDEN,
            format!(
                "key '{}' has no grant on context '{}' (batch source '{}'); nothing \
                 was applied",
                key.as_ref().map_or("-", |extension| extension.0.0.as_ref()),
                refused.context,
                refused.source
            ),
            started_at,
        );
    }
    let total = batches.len();
    let mut outcomes: Vec<ImportOutcome> = Vec::with_capacity(total);
    for (index, batch) in batches.iter().enumerate() {
        match crate::ingest::apply_batch(&state, batch) {
            Ok(applied) => {
                // Import is retract-then-apply — destructive — and both
                // the context and the replaced source live in the BODY,
                // out of the access log's reach. One audit line each.
                tracing::info!(
                    target: "taguru::audit",
                    key = %key_name(&key),
                    context = %batch.context,
                    source = %batch.source,
                    created = applied.created,
                    retracted = applied.retracted,
                    associations = applied.associations,
                    "import batch applied",
                );
                outcomes.push(import_outcome(batch, &applied));
            }
            Err(refusal) => {
                let note = if total > 1 {
                    format!(
                        "batch {} of {total} (context '{}', source '{}') refused — the {} \
                         batch(es) before it landed durably; fixing the stream and \
                         re-POSTing it whole is exact (each batch replaces its own \
                         source): ",
                        index + 1,
                        batch.context,
                        batch.source,
                        outcomes.len(),
                    )
                } else {
                    String::new()
                };
                return import_refusal(&state, batch, refusal, &note, started_at);
            }
        }
    }
    if total > 1 {
        ok(ImportStreamOutcome { batches: outcomes }, started_at)
    } else {
        let outcome = outcomes.pop().expect("parse_batches refuses empty streams");
        ok(outcome, started_at)
    }
}

/// `POST /contexts/{name}/compact` — rebuild the image without the
/// dead weight the append-only format accumulates (retracted edges,
/// unlinked attributions, arena slack), persisting the result before
/// answering. An admin verb (the role table's fail-closed default);
/// the context's own requests wait out the rebuild, every other
/// context is untouched. Content is preserved — the response says
/// what was shed and what the footprint became.
pub async fn compact_context(
    State(state): State<AppState>,
    Path(name): Path<String>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
) -> Response {
    let started_at = Instant::now();
    match tokio::task::block_in_place(|| state.compact_context(&name)) {
        Ok(outcome) => {
            // Maintenance that rewrites the image is audit-worthy even
            // though no knowledge changes.
            tracing::info!(
                target: "taguru::audit",
                key = %key_name(&key),
                context = %name,
                bytes_before = outcome.bytes_before,
                bytes_after = outcome.bytes_after,
                dead_edges = outcome.dead_edges,
                aliases_dropped = outcome.aliases_dropped,
                "context compacted",
            );
            ok(outcome, started_at)
        }
        Err(failure) => access_error(&state, failure, &name, started_at),
    }
}

/// `GET /contexts/{name}/export` — the context back out as the import
/// batch stream (docs/import.html): one batch per source in source-id
/// order, the create block on the first, the alias table on the last,
/// sourceless weight in a reserved `export:unsourced` batch. The
/// response body IS the stream (JSON Lines, not the JSON envelope):
/// save it, and `taguru import` or `POST /import` restores it —
/// per-source retract-then-apply, so re-importing is idempotent.
/// Materialized under one registry fence (a concurrent write cannot
/// shear the graph against the passages) and rendered off the async
/// runtime, the way vocabulary/audit steps aside.
pub async fn export_context(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    let started_at = Instant::now();
    let rendered = tokio::task::block_in_place(|| {
        state
            .export_context(&name)
            .map(|snapshot| crate::export::render(&name, &snapshot))
    });
    match rendered {
        Ok(Ok(rendered)) => (
            StatusCode::OK,
            [(
                axum::http::header::CONTENT_TYPE,
                "application/x-ndjson; charset=utf-8",
            )],
            rendered.stream,
        )
            .into_response(),
        // A real source colliding with a reserved export id — the one
        // thing a context can hold that the stream cannot say.
        Ok(Err(message)) => error(StatusCode::CONFLICT, message, started_at),
        Err(failure) => access_error(&state, failure, &name, started_at),
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
            let (total, matches) = page(result, request.limit);
            state.note_search(SearchOp::Recall, &name, total == 0);
            if search_log_enabled() {
                tracing::info!(
                    target: "taguru::search",
                    context = %name,
                    op = "recall",
                    cue = %request.cue,
                    hits = total,
                    "search",
                );
            }
            let matches = associations_out(&state, &name, matches);
            ok(MatchPage { total, matches }, started_at)
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
    for (field, position) in [
        ("subject", &request.subject),
        ("label", &request.label),
        ("object", &request.object),
    ] {
        if let Some(refusal) = overlong(field, as_refs(position).len(), started_at) {
            return refusal;
        }
    }
    match state.read_context(&name, |context| {
        context.query_any(
            &as_refs(&request.subject),
            &as_refs(&request.label),
            &as_refs(&request.object),
        )
    }) {
        Ok(result) => {
            let (total, matches) = page(result, request.limit);
            state.note_search(SearchOp::Query, &name, total == 0);
            if search_log_enabled() {
                tracing::info!(
                    target: "taguru::search",
                    context = %name,
                    op = "query",
                    subject = %as_refs(&request.subject).join(","),
                    label = %as_refs(&request.label).join(","),
                    object = %as_refs(&request.object).join(","),
                    hits = total,
                    "search",
                );
            }
            let matches = associations_out(&state, &name, matches);
            ok(MatchPage { total, matches }, started_at)
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
        Ok(result) => {
            state.note_read(&name, result.is_none());
            ok(result, started_at)
        }
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
    pub matches: Vec<RecollectionOut>,
}

pub async fn explore(
    State(state): State<AppState>,
    Path(name): Path<String>,
    AppJson(request): AppJson<ExploreRequest>,
) -> Response {
    let started_at = Instant::now();
    if let Some(refusal) = overlong("origins", request.origins.len(), started_at) {
        return refusal;
    }
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
            state.note_search(SearchOp::Explore, &name, total == 0);
            if search_log_enabled() {
                tracing::info!(
                    target: "taguru::search",
                    context = %name,
                    op = "explore",
                    origins = %request.origins.join(","),
                    hits = total,
                    "search",
                );
            }
            let matches = recollections_out(&state, &name, matches);
            ok(ExplorePage { total, matches }, started_at)
        }
        Err(failure) => access_error(&state, failure, &name, started_at),
    }
}

/// A bounded activation result: the same `{total, matches}` shape as
/// [`MatchPage`], but `total` comes straight from `Context::activate`,
/// which already sorts and truncates internally rather than going
/// through the `page` helper.
#[derive(Serialize)]
pub struct ActivationPage {
    pub total: usize,
    pub matches: Vec<ActivationOut>,
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
    if let Some(refusal) = overlong("origins", request.origins.len(), started_at) {
        return refusal;
    }
    match state.read_context(&name, |context| {
        let origins: Vec<&str> = request.origins.iter().map(String::as_str).collect();
        context.activate(
            &origins,
            request.decay.unwrap_or(0.5),
            clamp(request.limit, 20, MAX_MATCH_LIMIT),
        )
    }) {
        Ok((total, matches)) => {
            state.note_search(SearchOp::Activate, &name, total == 0);
            if search_log_enabled() {
                tracing::info!(
                    target: "taguru::search",
                    context = %name,
                    op = "activate",
                    origins = %request.origins.join(","),
                    hits = total,
                    "search",
                );
            }
            let matches = activations_out(&state, &name, matches);
            ok(ActivationPage { total, matches }, started_at)
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
    /// Candidate-count cap, clamped to the shared ceiling like every
    /// other match endpoint. Omitted means the ceiling itself — resolve
    /// historically served every qualifying candidate, so the default
    /// only stops the flood a short cue raises against a large
    /// vocabulary (every name containing it, in one response body).
    pub limit: Option<usize>,
}

/// One resolve candidate plus the tier that produced it. Lexical scores
/// are coverage/Dice, semantic scores are cosine similarities — ordinal
/// within a tier, never comparable across tiers.
#[derive(Serialize)]
pub struct TieredResolution {
    pub name: String,
    pub score: f64,
    pub tier: &'static str,
    /// The lexical string relation behind the score (exact / alias /
    /// containment / fuzzy) — the caller's warning that a high score
    /// may be a lookalike, not the thing (possible inside impossible
    /// scores 0.8). Absent on semantic candidates, whose score is a
    /// cosine, not a string overlap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<&'static str>,
    /// The candidate's own gloss — its name plus its heaviest facts,
    /// the same text the semantic tier embeds. This is the evidence
    /// that tells lookalike candidates apart: string overlap says
    /// 東京都 and 京都 are near, their facts say they are different
    /// things. Attached to the top candidates only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gloss: Option<String>,
}

fn lexical_tier(resolutions: Vec<Resolution>) -> Vec<TieredResolution> {
    resolutions
        .into_iter()
        .map(|resolution| TieredResolution {
            name: resolution.name,
            score: resolution.score,
            tier: "lexical",
            kind: Some(resolution.kind.as_str()),
            gloss: None,
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
            kind: None,
            gloss: None,
        });
    }
    merged
}

/// Enforces the resolve cap after the tiers merge. Semantic candidates
/// sit at the tail, appended precisely because the lexical tier was
/// weak — so overflow comes out of the lexical segment's own tail,
/// never out of the semantic candidates the fallback just earned. Only
/// a limit smaller than the semantic tier itself cuts into it (from
/// its tail too; both tiers arrive best-first).
fn trim_to_limit(mut served: Vec<TieredResolution>, limit: usize) -> Vec<TieredResolution> {
    let overflow = served.len().saturating_sub(limit);
    if overflow == 0 {
        return served;
    }
    let semantic = served
        .iter()
        .rev()
        .take_while(|candidate| candidate.tier == "semantic")
        .count();
    let lexical = served.len() - semantic;
    served.drain(lexical - overflow.min(lexical)..lexical);
    served.truncate(limit);
    served
}

/// How many resolve candidates carry their gloss. The head of the list
/// is where a caller weighs lookalikes against each other; decorating a
/// long fuzzy tail would only bloat the response it has already
/// stopped reading.
const GLOSSED_CANDIDATES: usize = 8;

/// Attaches each top candidate's gloss — the evidence a caller needs
/// to tell lookalike names apart without a second round trip. One
/// shared read after the tiers settle, covering lexical and semantic
/// candidates alike; if the context vanished in between, the
/// candidates simply keep gloss = None (the entry answer itself
/// already succeeded, and a decoration must not turn it into an
/// error).
fn attach_glosses(
    state: &AppState,
    name: &str,
    labels: bool,
    mut served: Vec<TieredResolution>,
) -> Vec<TieredResolution> {
    let head = served.len().min(GLOSSED_CANDIDATES);
    if head == 0 {
        return served;
    }
    let glosses = state.read_context(name, |context| {
        served[..head]
            .iter()
            .map(|candidate| {
                if labels {
                    context.label_gloss(&candidate.name, Context::GLOSS_EXAMPLES)
                } else {
                    context.concept_gloss(&candidate.name, Context::GLOSS_FACTS)
                }
            })
            .collect::<Vec<_>>()
    });
    if let Ok(glosses) = glosses {
        for (candidate, gloss) in served.iter_mut().zip(glosses) {
            candidate.gloss = gloss;
        }
    }
    served
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
    let limit = clamp(request.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT);
    let mut lexical = match state.read_context(name, |context| match (labels, request.dice_floor) {
        (false, Some(floor)) => context.resolve_with_floor(&request.cue, floor),
        (false, None) => context.resolve(&request.cue),
        (true, Some(floor)) => context.resolve_label_with_floor(&request.cue, floor),
        (true, None) => context.resolve_label(&request.cue),
    }) {
        Ok(result) => result,
        Err(failure) => return access_error(state, failure, name, started_at),
    };
    // Resolutions arrive best-first, so bounding the flood right here
    // keeps the strongest candidates and spares everything downstream
    // (the semantic dedup scan, gloss reads, serialization) the
    // pathological tail. The confidence probe below reads only the
    // first entry, which truncation never touches.
    lexical.truncate(limit);
    let confident = lexical
        .first()
        .is_some_and(|best| best.score >= LEXICAL_CONFIDENCE);
    let served = if confident {
        lexical_tier(lexical)
    } else {
        match semantic_fallback(state, name, request, labels, lexical, started_at) {
            Ok(served) => served,
            Err(response) => return response,
        }
    };
    let served = attach_glosses(state, name, labels, trim_to_limit(served, limit));
    let op = if labels {
        SearchOp::ResolveLabel
    } else {
        SearchOp::Resolve
    };
    let tier = resolve_tier_of(&served);
    state.note_search(op, name, served.is_empty());
    state.metrics().record_resolve_tier(tier);
    if search_log_enabled() {
        tracing::info!(
            target: "taguru::search",
            context = %name,
            op = if labels { "resolve_label" } else { "resolve" },
            cue = %request.cue,
            hits = served.len(),
            tier = tier.as_str(),
            top_score = served.first().map_or(0.0, |best| best.score),
            "search",
        );
    }
    ok(served, started_at)
}

/// The semantic tier, entered when the lexical tiers came back empty
/// or fragment-weak. `Err` is a response to serve immediately: unknown
/// context, or a provider failure with nothing lexical to degrade to.
#[allow(clippy::result_large_err)] // the Err IS the response served next
fn semantic_fallback(
    state: &AppState,
    name: &str,
    request: &ResolveRequest,
    labels: bool,
    lexical: Vec<Resolution>,
    started_at: Instant,
) -> Result<Vec<TieredResolution>, Response> {
    // The provider round trip can take hundreds of milliseconds; tell
    // the runtime this thread will block so other tasks migrate off it.
    match tokio::task::block_in_place(|| {
        state.semantic_resolve(name, &request.cue, labels, request.semantic_floor)
    }) {
        None => Err(not_found(name, started_at)),
        Some(Ok(semantic)) => Ok(merge_tiers(lexical, semantic)),
        // The semantic tier is enrichment once ANY lexical candidate
        // exists: degrade to the weak lexical results and log, rather
        // than failing an answerable request over a provider hiccup.
        Some(Err(message)) if !lexical.is_empty() => {
            tracing::warn!("semantic entry failed (serving weak lexical results): {message}");
            Ok(lexical_tier(lexical))
        }
        Some(Err(message)) => {
            tracing::warn!("semantic entry failed with nothing lexical to fall back on: {message}");
            Err(error(
                StatusCode::BAD_GATEWAY,
                format!("semantic entry failed: {message}"),
                started_at,
            ))
        }
    }
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

/// What one embedding refresh accomplished. `embedded`/`total` stay
/// the all-up numbers older clients read; the breakdowns appear only
/// when the passage lane ran, so a gloss-only deployment keeps its
/// exact historical shape.
#[derive(Serialize)]
pub struct RefreshOutcome {
    pub embedded: usize,
    pub total: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub glosses: Option<RefreshBreakdown>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub passages: Option<RefreshBreakdown>,
}

#[derive(Serialize)]
pub struct RefreshBreakdown {
    pub embedded: usize,
    pub total: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skipped_over_limit: Option<usize>,
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
    let glosses = match tokio::task::block_in_place(|| state.refresh_embeddings(&name)) {
        None => return not_found(&name, started_at),
        Some(Ok(counts)) => counts,
        Some(Err(message)) => {
            return error(
                StatusCode::BAD_GATEWAY,
                format!("embedding refresh failed: {message}"),
                started_at,
            );
        }
    };
    if !state.passage_embedding_enabled() {
        let (embedded, total) = glosses;
        return ok(
            RefreshOutcome {
                embedded,
                total,
                glosses: None,
                passages: None,
            },
            started_at,
        );
    }
    match tokio::task::block_in_place(|| state.refresh_passage_embeddings(&name)) {
        None => not_found(&name, started_at),
        Some(Ok(passages)) => ok(
            RefreshOutcome {
                embedded: glosses.0 + passages.embedded,
                total: glosses.1 + passages.total,
                glosses: Some(RefreshBreakdown {
                    embedded: glosses.0,
                    total: glosses.1,
                    skipped_over_limit: None,
                }),
                passages: Some(RefreshBreakdown {
                    embedded: passages.embedded,
                    total: passages.total,
                    skipped_over_limit: Some(passages.skipped_over_limit),
                }),
            },
            started_at,
        ),
        // The gloss half already succeeded and partial passage progress
        // is persisted — but the caller asked for a refresh and did not
        // fully get one; say so.
        Some(Err(message)) => error(
            StatusCode::BAD_GATEWAY,
            format!("passage embedding refresh failed partway (progress is saved): {message}"),
            started_at,
        ),
    }
}

/// One page of the relation vocabulary, keyset by label — the
/// vocabulary is client-minted, so like every listing it pages
/// instead of promising to fit in one response.
#[derive(Serialize)]
pub struct LabelPage {
    pub total: usize,
    pub labels: Vec<String>,
}

pub async fn labels(
    State(state): State<AppState>,
    Path(name): Path<String>,
    AppQuery(query): AppQuery<KeysetQuery>,
) -> Response {
    let started_at = Instant::now();
    let limit = clamp(query.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT);
    match state.read_context(&name, |context| {
        let mut labels: Vec<String> = context.labels().into_iter().map(String::from).collect();
        labels.sort();
        let total = labels.len();
        let labels: Vec<String> = labels
            .into_iter()
            .filter(|label| {
                query
                    .after
                    .as_deref()
                    .is_none_or(|after| label.as_str() > after)
            })
            .take(limit)
            .collect();
        LabelPage { total, labels }
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
    if let Some(refusal) = overlong("origins", request.origins.len(), started_at) {
        return refusal;
    }
    match state.read_context(&name, |context| {
        let origins: Vec<&str> = request.origins.iter().map(String::as_str).collect();
        context.unreachable_from(&origins)
    }) {
        Ok(result) => {
            let (total, matches) = page(result, request.limit);
            // A graph read like recall/query/explore/activate — the
            // usage counters must agree with that grouping. Zero
            // orphans is the audit SUCCEEDING, though, not a miss, so
            // it never counts as an empty read.
            state.note_read(&name, false);
            let matches = associations_out(&state, &name, matches);
            ok(MatchPage { total, matches }, started_at)
        }
        Err(failure) => access_error(&state, failure, &name, started_at),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use taguru::context::MatchKind;

    /// Absent halves of the refresh response shape must OMIT their
    /// keys, never serialize as null: older clients of
    /// /embeddings/refresh keep the exact historical two-key body.
    #[test]
    fn refresh_shapes_omit_absent_keys_rather_than_nulling_them() {
        let legacy = serde_json::to_value(RefreshOutcome {
            embedded: 3,
            total: 9,
            glosses: None,
            passages: None,
        })
        .unwrap();
        assert_eq!(
            legacy,
            serde_json::json!({"embedded": 3, "total": 9}),
            "passage lane off = the historical shape, byte for byte"
        );

        let broken_down = serde_json::to_value(RefreshOutcome {
            embedded: 5,
            total: 12,
            glosses: Some(RefreshBreakdown {
                embedded: 2,
                total: 4,
                skipped_over_limit: None,
            }),
            passages: Some(RefreshBreakdown {
                embedded: 3,
                total: 8,
                skipped_over_limit: Some(1),
            }),
        })
        .unwrap();
        assert_eq!(
            broken_down["glosses"],
            serde_json::json!({"embedded": 2, "total": 4}),
            "a gloss breakdown never grows a skipped_over_limit key"
        );
        assert_eq!(broken_down["passages"]["skipped_over_limit"], 1);
    }

    #[test]
    fn protocol_trailer_names_the_model_and_the_refresh_owner() {
        assert!(protocol_trailer(None, false).is_none());
        // No provider means no trailer even with the auto flag stuck on.
        assert!(protocol_trailer(None, true).is_none());

        let manual = protocol_trailer(Some("test-model"), false).unwrap();
        assert!(manual.contains("`test-model`"));
        assert!(manual.contains("calling `refresh_embeddings`"));

        let auto = protocol_trailer(Some("test-model"), true).unwrap();
        assert!(auto.contains("auto-refreshes"));
    }

    fn assoc(object: &str, weight: f64) -> Association {
        Association {
            subject: "s".to_string(),
            label: "l".to_string(),
            object: object.to_string(),
            weight,
            count: 1,
            attributions: Vec::new(),
        }
    }

    #[test]
    fn resolve_tier_classification_reads_the_served_payload() {
        let candidate = |score: f64, tier: &'static str| TieredResolution {
            name: "n".to_string(),
            score,
            tier,
            kind: None,
            gloss: None,
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
    fn trim_to_limit_spends_overflow_on_the_lexical_tail_never_the_semantic_one() {
        let candidate = |name: &str, tier: &'static str| TieredResolution {
            name: name.to_string(),
            score: 0.2,
            tier,
            kind: None,
            gloss: None,
        };
        let names = |served: &[TieredResolution]| -> Vec<String> {
            served.iter().map(|c| c.name.clone()).collect()
        };

        // Overflow comes out of the weak lexical tail; the semantic
        // candidates the fallback earned all survive.
        let served = vec![
            candidate("l1", "lexical"),
            candidate("l2", "lexical"),
            candidate("l3", "lexical"),
            candidate("s1", "semantic"),
            candidate("s2", "semantic"),
        ];
        assert_eq!(names(&trim_to_limit(served, 3)), vec!["l1", "s1", "s2"]);

        // A limit under the semantic tier itself: the lexical segment
        // goes first, then semantic trims from its own (best-first)
        // tail.
        let served = vec![
            candidate("l1", "lexical"),
            candidate("s1", "semantic"),
            candidate("s2", "semantic"),
            candidate("s3", "semantic"),
        ];
        assert_eq!(names(&trim_to_limit(served, 2)), vec!["s1", "s2"]);

        // Under the limit nothing moves.
        let served = vec![candidate("l1", "lexical"), candidate("s1", "semantic")];
        assert_eq!(names(&trim_to_limit(served, 2)), vec!["l1", "s1"]);
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
        let (total, matches) = page(matches, Some(1_000_000_000));
        assert_eq!(total, MAX_MATCH_LIMIT + 5);
        assert_eq!(matches.len(), MAX_MATCH_LIMIT);
    }

    #[test]
    fn page_keeps_insertion_order_within_limit_and_strongest_past_it() {
        let matches = vec![assoc("a", 1.0), assoc("b", -3.0), assoc("c", 2.0)];

        // Within the limit nothing is reordered or dropped.
        let (total, within) = page(matches.clone(), Some(3));
        assert_eq!(total, 3);
        let objects: Vec<&str> = within.iter().map(|m| m.object.as_str()).collect();
        assert_eq!(objects, vec!["a", "b", "c"]);

        // Past it, magnitude ranks — the negative fact is the strongest
        // knowledge — and total still reports the untruncated count.
        let (total, truncated) = page(matches, Some(2));
        assert_eq!(total, 3);
        let objects: Vec<&str> = truncated.iter().map(|m| m.object.as_str()).collect();
        assert_eq!(objects, vec!["b", "c"]);
    }

    #[test]
    fn merge_keeps_lexical_first_and_deduplicates_names() {
        let lexical = vec![
            Resolution {
                name: "杜氏の職".to_string(),
                score: 0.33,
                kind: MatchKind::Containment,
            },
            Resolution {
                name: "蔵".to_string(),
                score: 0.25,
                kind: MatchKind::Fuzzy,
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
