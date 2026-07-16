//! HTTP surface of the retrieval service: thin JSON adapters, one per
//! `Context` operation. The server adds transport, naming, and lifecycle
//! around the library — never retrieval semantics of its own — so each
//! handler is a lock, a library call, and a serialized reply.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request, State};
use axum::http::{Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use taguru::context::{
    Activation, Association, Attribution, Context, MatchKind, Recollection, Resolution,
};
use taguru::deadline::Deadline;

use crate::groups::{GroupRecord, MAX_GROUP_DEPTH, MAX_GROUP_MEMBERS, NestingViolation};
use crate::limits::HeavyOpsLimiter;
use crate::metrics::{ErrorKind, ResolveTier, SearchOp};
use crate::registry::{
    AccessError, AppState, AssocOp, ContextMeta, CreateError, CreateGroupError, RenameContextError,
    RenameGroupError, UpdateGroupError,
};

mod sources;
pub use sources::{
    citation, cross_search_passages, explain_search_passages, list_sources, lookup_passages,
    retract_source, search_passages,
};

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

/// The machine-readable failure kind riding every JSON error response
/// as `code` — a STABLE vocabulary (documented in llm-protocol.md) for
/// clients that must branch without parsing the human `error` text.
/// The status stays the transport truth; the code names what the
/// status alone leaves ambiguous (which 400, which 404, which 409,
/// which 503). Renaming or repurposing a variant is a breaking change
/// and belongs in the CHANGELOG like a response-shape change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ErrorCode {
    /// The request never parsed: broken JSON, wrong media type,
    /// well-formed JSON of the wrong shape, an unreadable query
    /// string, or an import stream that fails line validation.
    MalformedRequest,
    /// The request parsed, but a value was refused: an empty or
    /// oversized name, a non-finite or over-cap weight, a malformed
    /// cursor, a question with no passage to attach to.
    InvalidArgument,
    /// A batch or list-shaped input over its per-request cap — the
    /// one refusal where splitting the same content and resending it
    /// works.
    OverLimit,
    Unauthorized,
    Forbidden,
    NoContext,
    NoSource,
    NoParagraph,
    NoGroup,
    UnknownPath,
    MethodNotAllowed,
    Timeout,
    /// `PUT` on a resource — context or group — that already exists.
    AlreadyExists,
    /// Every other 409: alias conflicts, non-capacity partial writes,
    /// a real source colliding with a reserved export id.
    Conflict,
    PayloadTooLarge,
    RateLimited,
    Internal,
    EmbeddingsUnconfigured,
    EmbeddingsFailed,
    /// 503 shed at the in-flight ceiling — `Retry-After` rides along.
    Overloaded,
    /// 503 from the readiness probe: the write path is degraded.
    Unhealthy,
    StorageFull,
    /// 503 while a `POST /maintenance/compact` sweep holds the server
    /// closed to ordinary traffic — an intentional pause, not a fault.
    Maintenance,
}

impl ErrorCode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::MalformedRequest => "malformed_request",
            Self::InvalidArgument => "invalid_argument",
            Self::OverLimit => "over_limit",
            Self::Unauthorized => "unauthorized",
            Self::Forbidden => "forbidden",
            Self::NoContext => "no_context",
            Self::NoSource => "no_source",
            Self::NoParagraph => "no_paragraph",
            Self::NoGroup => "no_group",
            Self::UnknownPath => "unknown_path",
            Self::MethodNotAllowed => "method_not_allowed",
            Self::Timeout => "timeout",
            Self::AlreadyExists => "already_exists",
            Self::Conflict => "conflict",
            Self::PayloadTooLarge => "payload_too_large",
            Self::RateLimited => "rate_limited",
            Self::Internal => "internal",
            Self::EmbeddingsUnconfigured => "embeddings_unconfigured",
            Self::EmbeddingsFailed => "embeddings_failed",
            Self::Overloaded => "overloaded",
            Self::Unhealthy => "unhealthy",
            Self::StorageFull => "storage_full",
            Self::Maintenance => "maintenance",
        }
    }

    /// The status each code answers with — the code is the single
    /// source of truth at every error site, so a site cannot pair a
    /// code with a contradicting status. Codes sharing a status
    /// (`no_context`/`no_source`/..., `overloaded`/`unhealthy`) are
    /// exactly why the code exists. The extractor rejections are the
    /// one exception: they keep axum's own status (400/413/415/422)
    /// and pick the code FROM it — see [`AppJson`].
    fn status(self) -> StatusCode {
        match self {
            Self::MalformedRequest | Self::InvalidArgument | Self::OverLimit => {
                StatusCode::BAD_REQUEST
            }
            Self::Unauthorized => StatusCode::UNAUTHORIZED,
            Self::Forbidden => StatusCode::FORBIDDEN,
            Self::NoContext
            | Self::NoSource
            | Self::NoParagraph
            | Self::NoGroup
            | Self::UnknownPath => StatusCode::NOT_FOUND,
            Self::MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED,
            Self::Timeout => StatusCode::REQUEST_TIMEOUT,
            Self::AlreadyExists | Self::Conflict => StatusCode::CONFLICT,
            Self::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Self::RateLimited => StatusCode::TOO_MANY_REQUESTS,
            Self::Internal => StatusCode::INTERNAL_SERVER_ERROR,
            Self::EmbeddingsUnconfigured => StatusCode::NOT_IMPLEMENTED,
            Self::EmbeddingsFailed => StatusCode::BAD_GATEWAY,
            Self::Overloaded | Self::Unhealthy | Self::Maintenance => {
                StatusCode::SERVICE_UNAVAILABLE
            }
            Self::StorageFull => StatusCode::INSUFFICIENT_STORAGE,
        }
    }

    /// The code an extractor rejection maps to, keyed on the status
    /// axum chose: the body cap breach is its own kind (chunk the
    /// payload); everything else is a request that never parsed.
    fn for_rejection(status: StatusCode) -> Self {
        if status == StatusCode::PAYLOAD_TOO_LARGE {
            Self::PayloadTooLarge
        } else {
            Self::MalformedRequest
        }
    }
}

#[derive(Serialize)]
pub struct ApiError {
    status: &'static str,
    code: &'static str,
    error: String,
    time: f64,
}

impl ApiError {
    fn new(code: ErrorCode, error: impl Into<String>, started_at: Instant) -> Self {
        Self {
            status: "error",
            code: code.as_str(),
            error: error.into(),
            time: started_at.elapsed().as_secs_f64(),
        }
    }
}

fn ok<T: Serialize>(result: T, started_at: Instant) -> Response {
    (StatusCode::OK, Json(ApiResponse::ok(result, started_at))).into_response()
}

/// The one constructor every JSON error response goes through, status
/// derived from the code. The extractor reshapes ([`AppJson`],
/// [`AppQuery`], [`AppBytes`]) call [`coded`] instead, keeping axum's
/// own rejection status.
pub(crate) fn error(code: ErrorCode, message: impl Into<String>, started_at: Instant) -> Response {
    coded(code.status(), code, message, started_at)
}

fn coded(
    status: StatusCode,
    code: ErrorCode,
    message: impl Into<String>,
    started_at: Instant,
) -> Response {
    (status, Json(ApiError::new(code, message, started_at))).into_response()
}

/// `CatchPanicLayer`'s panic handler: a handler that unwinds still owes
/// the caller the same one JSON error shape every other failure returns
/// — and, just as importantly, an ordinary [`Response`] handed back to
/// `next.run` is what lets the metrics/access-log/trace middleware
/// wrapping the router see this request at all. Without it, a panic
/// unwinds straight through them and the request vanishes from every
/// signal at once.
pub(crate) fn panic_response(payload: Box<dyn std::any::Any + Send>) -> Response {
    let message = if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        "handler panicked with a non-string payload".to_string()
    };
    // A bug surfacing at runtime, not one of the foreseen degraded
    // states this codebase otherwise logs with warn! — worth the same
    // loud signal as a boot-time fatal.
    tracing::error!(%message, "handler panicked; responding 500 instead of dropping the connection");
    coded(
        StatusCode::INTERNAL_SERVER_ERROR,
        ErrorCode::Internal,
        format!("internal error: {message}"),
        Instant::now(),
    )
}

fn not_found(name: &str, started_at: Instant) -> Response {
    error(
        ErrorCode::NoContext,
        format!("context '{name}' not found"),
        started_at,
    )
}

/// The entry guard every `block_in_place` site checks before paying
/// for its blocking work: a budget already spent (queueing, an
/// earlier retry, a slow body read) means the operation would run
/// past `enforce_timeout`'s race anyway, so refuse now instead of
/// starting work nobody will read the result of.
fn deadline_exceeded(started_at: Instant) -> Response {
    error(
        ErrorCode::Timeout,
        "request exceeded its budget before this operation could start; narrow the query or raise TAGURU_REQUEST_TIMEOUT_SECS",
        started_at,
    )
}

fn group_not_found(name: &str, started_at: Instant) -> Response {
    error(
        ErrorCode::NoGroup,
        format!("group '{name}' not found"),
        started_at,
    )
}

/// The one place the two nesting violations map onto the error
/// vocabulary: a cycle is a malformed request (`invalid_argument`)
/// while a too-tall stack trips a cap (`over_limit`) — depth is
/// policy, not syntax. [`nesting_refusal`] and the import restore's
/// refusal both read it here, so the two paths cannot drift onto
/// different codes for the same condition.
fn nesting_error_code(violation: &NestingViolation) -> ErrorCode {
    match violation {
        NestingViolation::Cycle(_) => ErrorCode::InvalidArgument,
        NestingViolation::TooDeep(_) => ErrorCode::OverLimit,
    }
}

/// The two shape refusals a nesting write can hear, one wire response
/// each; both name a group on the offending path.
fn nesting_refusal(violation: NestingViolation, started_at: Instant) -> Response {
    let code = nesting_error_code(&violation);
    let message = match violation {
        NestingViolation::Cycle(group) => {
            format!("the nesting would loop back into group '{group}'; nothing was applied")
        }
        NestingViolation::TooDeep(group) => format!(
            "the nesting would stack more than {MAX_GROUP_DEPTH} groups \
             (through group '{group}'); nothing was applied"
        ),
    };
    error(code, message, started_at)
}

/// A group write whose RESULT would bundle more than
/// [`MAX_GROUP_MEMBERS`] names in one set — `field` says which
/// ("member contexts" / "child groups"). The delta caps already bound
/// each request; this bounds what the deltas accumulate to.
fn over_cap_refusal(field: &'static str, started_at: Instant) -> Response {
    error(
        ErrorCode::OverLimit,
        format!(
            "the group would bundle more than {MAX_GROUP_MEMBERS} {field}; nothing was \
             applied — split into nested child groups"
        ),
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
            Err(rejection) => Err(coded(
                rejection.status(),
                ErrorCode::for_rejection(rejection.status()),
                rejection.body_text(),
                Instant::now(),
            )),
        }
    }
}

/// axum's Bytes extractor with its rejection reshaped into the
/// [`ApiError`] body: the raw-body routes (import, and the
/// optional-body handlers) answer a body-cap breach in the same one
/// shape every other axis speaks, instead of axum's plain-text 413.
pub struct AppBytes(pub axum::body::Bytes);

impl<S> FromRequest<S> for AppBytes
where
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(request: Request, state: &S) -> Result<Self, Self::Rejection> {
        match axum::body::Bytes::from_request(request, state).await {
            Ok(bytes) => Ok(Self(bytes)),
            Err(rejection) => Err(coded(
                rejection.status(),
                ErrorCode::for_rejection(rejection.status()),
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
            Err(rejection) => Err(coded(
                rejection.status(),
                ErrorCode::for_rejection(rejection.status()),
                rejection.body_text(),
                Instant::now(),
            )),
        }
    }
}

/// axum's Path extractor with rejections reshaped into the
/// [`ApiError`] body, exactly as [`AppQuery`] does. A percent-encoded
/// path segment that decodes to invalid UTF-8 is the one client-driven
/// Path rejection; without this wrapper it would answer axum's
/// plain-text 400 — the lone axis still off the shared error shape.
pub struct AppPath<T>(pub T);

impl<S, T> axum::extract::FromRequestParts<S> for AppPath<T>
where
    T: serde::de::DeserializeOwned + Send,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        match axum::extract::Path::<T>::from_request_parts(parts, state).await {
            Ok(axum::extract::Path(value)) => Ok(Self(value)),
            Err(rejection) => Err(coded(
                rejection.status(),
                ErrorCode::for_rejection(rejection.status()),
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
        ErrorCode::UnknownPath,
        format!("no route for {method} {uri}; GET /protocol lists the API"),
        Instant::now(),
    )
}

/// A known path hit with the wrong verb — same story, 405.
pub async fn method_not_allowed(method: Method, uri: Uri) -> Response {
    error(
        ErrorCode::MethodNotAllowed,
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
    *ENABLED.get_or_init(|| crate::env_bool("TAGURU_LOG_SEARCHES", false))
}

fn access_error(
    state: &AppState,
    failure: AccessError,
    name: &str,
    started_at: Instant,
) -> Response {
    access_error_noted(state, failure, name, "", started_at)
}

/// [`access_error`] with a leading `note` — the one place the three
/// `AccessError` arms map to statuses and metrics, so the per-batch
/// import path (which prefixes each refusal with which batch failed)
/// shares them instead of hand-copying the mapping. `note` is empty
/// for every ordinary caller.
fn access_error_noted(
    state: &AppState,
    failure: AccessError,
    name: &str,
    note: &str,
    started_at: Instant,
) -> Response {
    match failure {
        AccessError::NotFound => error(
            ErrorCode::NoContext,
            format!("{note}context '{name}' not found"),
            started_at,
        ),
        AccessError::Load(message) => {
            state.metrics().record_error(ErrorKind::Load);
            error(ErrorCode::Internal, format!("{note}{message}"), started_at)
        }
        AccessError::Unpersisted(message) => {
            state.metrics().record_error(ErrorKind::WalRefused);
            error(
                ErrorCode::Internal,
                format!("{note}write not persisted (nothing was applied): {message}"),
                started_at,
            )
        }
        AccessError::DeadlineExceeded => error(
            ErrorCode::Timeout,
            format!(
                "{note}request exceeded its budget; narrow the query \
                 (TAGURU_REQUEST_TIMEOUT_SECS tunes this)"
            ),
            started_at,
        ),
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
    /// Only contexts with this pinned state. Defines the population of
    /// interest rather than a cursor, so — unlike `after`/`limit` — it
    /// is applied before `total` is counted.
    pub pinned: Option<bool>,
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
    let limit = clamp(query.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT);
    let after = query.after.as_deref();
    // world so pagination stays coherent for that caller. An explicit
    // allow-list has no relation to name order, so seeking a page of
    // the full directory and filtering afterward could come back short
    // even when more allowed contexts exist further along — instead,
    // the (typically small, operator-configured) allow-list is paged
    // as its own sorted collection. An unscoped key (or one scoped to
    // "every context") pages the registry directly — unless `pinned`
    // is set, which (like the allow-list) defines the population
    // rather than a cursor and so also forces the whole-directory path:
    // the BTreeMap-seeking `directory_page` fast path has no way to
    // know in advance how many pinned entries lie within any range.
    let allowed = match &scope {
        Some(axum::Extension(scope)) => scope.contexts.clone(),
        None => None,
    };
    let (total, contexts) = if allowed.is_none() && query.pinned.is_none() {
        state.directory_page(after, limit)
    } else {
        let mut directory: Vec<_> = match &allowed {
            Some(allowed) => allowed
                .iter()
                .filter_map(|name| state.directory_entry(name))
                .collect(),
            None => state.directory(),
        };
        directory.retain(|entry| query.pinned.is_none_or(|pinned| entry.pinned == pinned));
        directory.sort_by(|a, b| a.name.cmp(&b.name));
        let total = directory.len();
        let contexts = directory
            .into_iter()
            .filter(|entry| after.is_none_or(|after| entry.name.as_str() > after))
            .take(limit)
            .collect();
        (total, contexts)
    };
    ok(ContextPage { total, contexts }, started_at)
}

/// POST /flush: persist every dirty context NOW and answer with the
/// names that flushed — the quiescing move before a file-level backup,
/// instead of "stop the server or wait out the flush interval".
///
/// Flush is inherently whole-of-server — that IS its job — so it has
/// no `{name}` for the middleware's per-context grant to key on, and
/// its response NAMES every flushed context. A context-scoped key must
/// therefore be refused outright rather than handed the full list
/// (which `GET /contexts` would hide from it): filtering flush would
/// silently defeat its one purpose, quiescing everything before a
/// backup.
pub async fn flush_all(
    State(state): State<AppState>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    if let Some(axum::Extension(scope)) = &scope
        && scope.contexts.is_some()
    {
        return error(
            ErrorCode::Forbidden,
            "flush is server-wide (it names every flushed context); a context-scoped \
             key cannot call it",
            started_at,
        );
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let flushed = tokio::task::block_in_place(|| state.flush_dirty());
    ok(flushed, started_at)
}

/// Optional tuning knob for [`maintenance_compact`]: only contexts whose
/// live dead ratio strictly exceeds this qualify. Omitted (and `0.0`)
/// means "any dead weight at all".
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct MaintenanceCompactQuery {
    pub min_dead_ratio: Option<f64>,
}

/// `POST /maintenance/compact` — closes the server to ordinary traffic
/// just long enough to rebuild every context whose dead ratio clears
/// `min_dead_ratio` (`GET /contexts/{name}` and `/metrics` show the
/// live ratios that inform the choice), worst ratio first, then reopens.
/// `/health` answers 503 `maintenance` and `enforce_concurrency` sheds
/// new work early for the duration, but the one real guarantee against
/// two sweeps overlapping is the CAS in [`AppState::try_enter_maintenance`]
/// taken here: a second call while one is running answers 409, not a
/// queued wait. Server-wide like `flush`, so a context-scoped key is
/// refused outright rather than silently filtered.
pub async fn maintenance_compact(
    State(state): State<AppState>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    AppQuery(query): AppQuery<MaintenanceCompactQuery>,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    if let Some(axum::Extension(scope)) = &scope
        && scope.contexts.is_some()
    {
        return error(
            ErrorCode::Forbidden,
            "maintenance is server-wide (it may compact any context); a \
             context-scoped key cannot call it",
            started_at,
        );
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let Some(_guard) = state.try_enter_maintenance() else {
        return error(
            ErrorCode::Conflict,
            "a maintenance compaction sweep is already running",
            started_at,
        );
    };
    // New admissions are already being shed (best effort) now that the
    // flag is set, so this count can only fall — except for our own
    // request, which is why it drains to 1 rather than 0.
    while state.metrics().inflight_count() > 1 {
        if deadline.expired() {
            return deadline_exceeded(started_at);
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let min_dead_ratio = query.min_dead_ratio.unwrap_or(0.0).clamp(0.0, 1.0);
    let outcome =
        tokio::task::block_in_place(|| state.run_maintenance_compaction(min_dead_ratio, deadline));
    // Maintenance that rewrites images is audit-worthy even though no
    // knowledge changes (mirrors the per-context compact endpoint).
    tracing::info!(
        target: "taguru::audit",
        key = %key_name(&key),
        contexts_compacted = outcome.contexts.len(),
        deadline_exceeded = outcome.deadline_exceeded,
        "maintenance compaction swept the server",
    );
    ok(outcome, started_at)
}

/// One directory row by name — the cheap existence-and-stats check,
/// without listing anything else.
pub async fn get_context(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
) -> Response {
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
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppBytes(body): AppBytes,
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
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Writes the sidecar (fsync + rename) like every other mutating
    // endpoint; keep it off the async worker.
    match tokio::task::block_in_place(|| state.create(&name, meta)) {
        Ok(()) => ok(true, started_at),
        Err(CreateError::AlreadyExists) => error(
            ErrorCode::AlreadyExists,
            format!("context '{name}' already exists"),
            started_at,
        ),
        Err(CreateError::InvalidName) => error(
            ErrorCode::InvalidArgument,
            "the context name must not be empty".to_string(),
            started_at,
        ),
        Err(CreateError::Io(io_error)) => {
            state.metrics().record_error(ErrorKind::Io);
            error(
                ErrorCode::Internal,
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
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
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
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Writes the sidecar (fsync + rename) like every other mutating
    // endpoint, and a pin toggle can additionally load the context from
    // disk (`ensure_hot`); keep both off the async worker.
    match tokio::task::block_in_place(|| {
        state.update_meta(
            &name,
            request.description,
            request.pinned,
            request.dice_floor,
            request.semantic_floor,
        )
    }) {
        None => not_found(&name, started_at),
        Some(Ok(meta)) => ok(meta, started_at),
        Some(Err(io_error)) => {
            state.metrics().record_error(ErrorKind::Io);
            error(
                ErrorCode::Internal,
                format!("metadata update not persisted: {io_error}"),
                started_at,
            )
        }
    }
}

pub async fn delete_context(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Writes a durable marker (fsync) then unlinks every sidecar file;
    // keep it off the async worker like every other mutating endpoint.
    match tokio::task::block_in_place(|| state.delete(&name)) {
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
                        ErrorCode::Internal,
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

#[derive(Debug, Deserialize)]
pub struct RenameRequest {
    pub to: String,
}

/// `POST /contexts/{name}/rename` — the whole file family moves to
/// `to` and every group naming `name` is rewritten to match. Admin
/// role (unclassified in [`crate::auth::required_role`], so it fails
/// closed there); `{name}` is a context name like every other
/// `/contexts/{name}...` route, so the authorization middleware's own
/// per-context grant check already covers the SOURCE. The
/// DESTINATION lives in the body, out of that middleware's reach —
/// same discipline as `import_batch` — so this handler gates it with
/// [`scope_refusal`] before renaming: otherwise a context-scoped key
/// could move its data to an unscoped name.
pub async fn rename_context(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<RenameRequest>,
) -> Response {
    let started_at = Instant::now();
    if let Some(refusal) = oversized(
        "the destination name",
        &request.to,
        MAX_CONTEXT_NAME_BYTES,
        started_at,
    ) {
        return refusal;
    }
    if let Some(refusal) = scope_refusal(&scope, &key, [&request.to], started_at) {
        return refusal;
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Drains any unflushed Hot state to disk, moves the whole file
    // family, and rewrites group membership; keep it off the async
    // worker like every other mutating endpoint.
    match tokio::task::block_in_place(|| state.rename_context(&name, &request.to)) {
        Ok(()) => {
            tracing::info!(
                target: "taguru::audit",
                key = %key_name(&key),
                from = %name,
                to = %request.to,
                "context renamed",
            );
            ok(true, started_at)
        }
        Err(RenameContextError::NotFound) => not_found(&name, started_at),
        Err(RenameContextError::AlreadyExists) => error(
            ErrorCode::AlreadyExists,
            format!("context '{}' already exists", request.to),
            started_at,
        ),
        Err(RenameContextError::InvalidName) => error(
            ErrorCode::InvalidArgument,
            "the destination name must not be empty".to_string(),
            started_at,
        ),
        Err(RenameContextError::Busy) => error(
            ErrorCode::Conflict,
            format!(
                "context '{name}' or '{}' is mid-rename, -create, or -delete; retry shortly",
                request.to
            ),
            started_at,
        ),
        Err(RenameContextError::Io(io_error)) => {
            state.metrics().record_error(ErrorKind::Io);
            error(
                ErrorCode::Internal,
                format!(
                    "context '{name}' rename not fully persisted: {io_error} \
                     (a rename marker remains; the next boot resumes it)"
                ),
                started_at,
            )
        }
    }
}

/// A bounded group-directory page; `total` counts the whole directory,
/// cursor-independent, exactly as [`ContextPage`]'s does.
#[derive(Serialize)]
pub struct GroupPage {
    pub total: usize,
    pub groups: Vec<GroupEntry>,
}

/// One group as served — the directory row, the single GET, and the
/// PATCH response are all this one shape, as with [`DirectoryEntry`].
#[derive(Serialize)]
pub struct GroupEntry {
    pub name: String,
    pub description: String,
    /// Member context names, sorted. For a context-scoped key this
    /// carries only the members its grant allows.
    pub contexts: Vec<String>,
    /// Child group names, sorted — never scope-filtered (so the set
    /// moves straight from the record): like the row itself, a group's
    /// name is an organizational label, not context content, and the
    /// contexts BEHIND a child stay filtered wherever they are served.
    pub groups: BTreeSet<String>,
}

/// Whether the key's grant lets it see the named context — no scope
/// means everything is visible. The one predicate behind every place
/// that FILTERS to the grant rather than refusing ([`group_entry`],
/// [`cross_targets`]'s group resolution), so "the slice a scoped key
/// sees" is defined exactly once and the two surfaces cannot drift.
fn scope_allows(scope: &Option<axum::Extension<crate::auth::KeyScope>>, name: &str) -> bool {
    scope
        .as_ref()
        .is_none_or(|axum::Extension(scope)| scope.allows_context(name))
}

/// The scope cut on one group row. Deliberately different from
/// `list_contexts`, which hides whole rows: a group is an
/// organizational label over contexts, not context content, and hiding
/// the row would also hide it from the very key that may still add or
/// remove its own contexts there. The members are what a grant is
/// about, so the members are what gets filtered.
fn group_entry(
    name: String,
    record: GroupRecord,
    scope: &Option<axum::Extension<crate::auth::KeyScope>>,
) -> GroupEntry {
    GroupEntry {
        name,
        description: record.description,
        contexts: scoped_member_contexts(record.contexts, scope),
        groups: record.groups,
    }
}

/// [`group_entry`]'s member filter on its own — the one loop behind
/// every surface that serves a group's members (the row, the export),
/// generic over the collection each output shape wants, so the
/// surfaces cannot drift in what a scoped key sees.
fn scoped_member_contexts<C: FromIterator<String>>(
    contexts: BTreeSet<String>,
    scope: &Option<axum::Extension<crate::auth::KeyScope>>,
) -> C {
    contexts
        .into_iter()
        .filter(|context| scope_allows(scope, context))
        .collect()
}

/// The gate for a scoped key on any operation whose context names ride
/// the body or the stored record rather than the path — group writes
/// (through [`scoped_group_refusal`], at membership granularity, the
/// import gate's pre-apply judgement) and the cross-context searches:
/// one involved context beyond the grant refuses the request whole.
/// Checked BEFORE existence on purpose: existence-first would answer
/// 404 for a missing out-of-scope name and 403 for a live one, handing
/// a scoped key an oracle for which context names exist beyond its
/// grant.
fn scope_refusal<'a>(
    scope: &Option<axum::Extension<crate::auth::KeyScope>>,
    key: &Option<axum::Extension<crate::auth::AuthKey>>,
    involved: impl IntoIterator<Item = &'a String>,
    started_at: Instant,
) -> Option<Response> {
    let Some(axum::Extension(scope)) = scope else {
        return None;
    };
    let refused = involved
        .into_iter()
        .find(|context| !scope.allows_context(context))?;
    Some(error(
        ErrorCode::Forbidden,
        format!(
            "key '{}' has no grant on context '{refused}'; nothing was applied",
            key_name(key),
        ),
        started_at,
    ))
}

/// The gate every group write runs, wrapped around
/// [`scope_refusal`]: resolves what the operation involves — the
/// transitive context closures of the `closure_roots` groups plus the
/// `direct` context names — and refuses if any of it sits beyond the
/// grant. An unscoped key passes immediately, without paying for the
/// closure read.
fn scoped_group_refusal<'r, 'd>(
    state: &AppState,
    scope: &Option<axum::Extension<crate::auth::KeyScope>>,
    key: &Option<axum::Extension<crate::auth::AuthKey>>,
    closure_roots: impl IntoIterator<Item = &'r str>,
    direct: impl IntoIterator<Item = &'d String>,
    started_at: Instant,
) -> Option<Response> {
    if scope.is_none() {
        return None;
    }
    let mut involved = state.group_context_closures(closure_roots);
    involved.extend(direct.into_iter().cloned());
    scope_refusal(scope, key, &involved, started_at)
}

/// The group directory: every group's name, description, member
/// contexts, and child groups, name-ordered and paged like
/// `GET /contexts`. Groups bundle contexts and may nest child groups —
/// a shallow DAG, at most [`MAX_GROUP_DEPTH`] storeys and never cyclic
/// — as the addressing unit that cross-context retrieval will build
/// on.
pub async fn list_groups(
    State(state): State<AppState>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    AppQuery(query): AppQuery<KeysetQuery>,
) -> Response {
    let started_at = Instant::now();
    let (total, page) = state.group_page(
        query.after.as_deref(),
        clamp(query.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT),
    );
    let groups: Vec<_> = page
        .into_iter()
        .map(|(name, record)| group_entry(name, record, &scope))
        .collect();
    ok(GroupPage { total, groups }, started_at)
}

pub async fn get_group(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
) -> Response {
    let started_at = Instant::now();
    match state.group(&name) {
        Some(record) => ok(group_entry(name, record, &scope), started_at),
        None => group_not_found(&name, started_at),
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct CreateGroupRequest {
    pub description: String,
    /// Initial member context names; every one must already exist.
    pub contexts: Vec<String>,
    /// Initial child group names; every one must already exist, and
    /// the nesting that results must stay acyclic and at most
    /// [`MAX_GROUP_DEPTH`] groups tall.
    pub groups: Vec<String>,
}

pub async fn create_group(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppBytes(body): AppBytes,
) -> Response {
    let started_at = Instant::now();
    let request: CreateGroupRequest = match optional_body(&body, started_at) {
        Ok(request) => request,
        Err(refusal) => return *refusal,
    };
    if let Some(refusal) = oversized("the group name", &name, MAX_CONTEXT_NAME_BYTES, started_at) {
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
    if let Some(refusal) = overlong("contexts", request.contexts.len(), started_at) {
        return refusal;
    }
    if let Some(refusal) = overlong("groups", request.groups.len(), started_at) {
        return refusal;
    }
    // A scoped key is judged against everything the new group would
    // address: the listed contexts plus every context reachable
    // through the listed children.
    if let Some(refusal) = scoped_group_refusal(
        &state,
        &scope,
        &key,
        request.groups.iter().map(String::as_str),
        &request.contexts,
        started_at,
    ) {
        return refusal;
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Writes the group file (fsync + rename) like every other mutating
    // endpoint; keep it off the async worker.
    match tokio::task::block_in_place(|| {
        state.create_group(
            &name,
            request.description,
            request.contexts.into_iter().collect(),
            request.groups.into_iter().collect(),
        )
    }) {
        Ok(()) => ok(true, started_at),
        Err(CreateGroupError::AlreadyExists) => error(
            ErrorCode::AlreadyExists,
            format!("group '{name}' already exists"),
            started_at,
        ),
        Err(CreateGroupError::InvalidName) => error(
            ErrorCode::InvalidArgument,
            "the group name must not be empty".to_string(),
            started_at,
        ),
        Err(CreateGroupError::NoSuchContext(context)) => error(
            ErrorCode::NoContext,
            format!("context '{context}' not found; nothing was applied"),
            started_at,
        ),
        Err(CreateGroupError::NoSuchGroup(group)) => error(
            ErrorCode::NoGroup,
            format!("group '{group}' not found; nothing was applied"),
            started_at,
        ),
        Err(CreateGroupError::Nesting(violation)) => nesting_refusal(violation, started_at),
        Err(CreateGroupError::OverCap(field)) => over_cap_refusal(field, started_at),
        Err(CreateGroupError::Io(io_error)) => {
            state.metrics().record_error(ErrorKind::Io);
            error(
                ErrorCode::Internal,
                format!("group '{name}' could not be persisted: {io_error}"),
                started_at,
            )
        }
    }
}

/// Membership updates are DELTAS, not a replacement list: two clients
/// adding different contexts concurrently must both land, and "add
/// this context here" is the natural operation for an LLM client —
/// the add/remove split aliases already use. A name in both lists ends
/// up a member (removals apply first). Removing a non-member is an
/// idempotent no-op; only additions demand the context — or, for
/// `add_groups`, the child group — exists. Child additions must also
/// leave the nesting acyclic and within [`MAX_GROUP_DEPTH`] storeys.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct UpdateGroupRequest {
    pub description: Option<String>,
    pub add_contexts: Vec<String>,
    pub remove_contexts: Vec<String>,
    pub add_groups: Vec<String>,
    pub remove_groups: Vec<String>,
}

pub async fn update_group(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<UpdateGroupRequest>,
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
    if let Some(refusal) = overlong("add_contexts", request.add_contexts.len(), started_at) {
        return refusal;
    }
    if let Some(refusal) = overlong("remove_contexts", request.remove_contexts.len(), started_at) {
        return refusal;
    }
    if let Some(refusal) = overlong("add_groups", request.add_groups.len(), started_at) {
        return refusal;
    }
    if let Some(refusal) = overlong("remove_groups", request.remove_groups.len(), started_at) {
        return refusal;
    }
    // A scoped key is judged against every context this update touches:
    // the group's transitive members plus every name the request
    // carries — context names directly, group names through their own
    // closures.
    if let Some(refusal) = scoped_group_refusal(
        &state,
        &scope,
        &key,
        [name.as_str()]
            .into_iter()
            .chain(request.add_groups.iter().map(String::as_str))
            .chain(request.remove_groups.iter().map(String::as_str)),
        request.add_contexts.iter().chain(&request.remove_contexts),
        started_at,
    ) {
        return refusal;
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Writes the group file (fsync + rename); keep it off the async
    // worker.
    match tokio::task::block_in_place(|| {
        state.update_group(
            &name,
            request.description,
            request.add_contexts.into_iter().collect(),
            request.remove_contexts.into_iter().collect(),
            request.add_groups.into_iter().collect(),
            request.remove_groups.into_iter().collect(),
        )
    }) {
        Ok(record) => ok(group_entry(name, record, &scope), started_at),
        Err(UpdateGroupError::NotFound) => group_not_found(&name, started_at),
        Err(UpdateGroupError::NoSuchContext(context)) => error(
            ErrorCode::NoContext,
            format!("context '{context}' not found; nothing was applied"),
            started_at,
        ),
        Err(UpdateGroupError::NoSuchGroup(group)) => error(
            ErrorCode::NoGroup,
            format!("group '{group}' not found; nothing was applied"),
            started_at,
        ),
        Err(UpdateGroupError::Nesting(violation)) => nesting_refusal(violation, started_at),
        Err(UpdateGroupError::OverCap(field)) => over_cap_refusal(field, started_at),
        Err(UpdateGroupError::Io(io_error)) => {
            state.metrics().record_error(ErrorKind::Io);
            error(
                ErrorCode::Internal,
                format!("group update not persisted (nothing was applied): {io_error}"),
                started_at,
            )
        }
    }
}

pub async fn delete_group(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    // Deleting the bundling touches every member's grant — nested
    // members included: judged like any other group write.
    if let Some(refusal) = scoped_group_refusal(
        &state,
        &scope,
        &key,
        [name.as_str()],
        std::iter::empty(),
        started_at,
    ) {
        return refusal;
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Unlinks the group file; keep it off the async worker.
    match tokio::task::block_in_place(|| state.delete_group(&name)) {
        None => group_not_found(&name, started_at),
        Some(outcome) => {
            // Destructive, so it leaves a `taguru::audit` line like
            // delete_context — the member contexts themselves are
            // untouched and say so via their own lines only when THEY
            // are deleted.
            tracing::info!(
                target: "taguru::audit",
                key = %key_name(&key),
                group = %name,
                file_removed = outcome.is_ok(),
                "group deleted",
            );
            match outcome {
                Ok(()) => ok(true, started_at),
                Err(io_error) => {
                    state.metrics().record_error(ErrorKind::Io);
                    error(
                        ErrorCode::Internal,
                        format!(
                            "group '{name}' removed but its file was not: {io_error} \
                             (if the file survives, the group reappears at the next restart)"
                        ),
                        started_at,
                    )
                }
            }
        }
    }
}

/// `POST /groups/{name}/rename` — the group's file moves to `to` and
/// every OTHER group naming `name` as a child is rewritten to match.
/// Unlike `rename_context`, `{name}` here is a GROUP name, so it is
/// one of the routes the authorization middleware exempts from its
/// per-context grant check — the scope gate belongs to this handler,
/// exactly as `delete_group`'s does.
pub async fn rename_group(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<RenameRequest>,
) -> Response {
    let started_at = Instant::now();
    if let Some(refusal) = oversized(
        "the destination name",
        &request.to,
        MAX_CONTEXT_NAME_BYTES,
        started_at,
    ) {
        return refusal;
    }
    // Renaming the bundling touches every member's grant — nested
    // members included — exactly like deleting it.
    if let Some(refusal) = scoped_group_refusal(
        &state,
        &scope,
        &key,
        [name.as_str()],
        std::iter::empty(),
        started_at,
    ) {
        return refusal;
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Writes the group file (fsync + rename); keep it off the async
    // worker.
    match tokio::task::block_in_place(|| state.rename_group(&name, &request.to)) {
        Ok(()) => {
            tracing::info!(
                target: "taguru::audit",
                key = %key_name(&key),
                from = %name,
                to = %request.to,
                "group renamed",
            );
            ok(true, started_at)
        }
        Err(RenameGroupError::NotFound) => group_not_found(&name, started_at),
        Err(RenameGroupError::AlreadyExists) => error(
            ErrorCode::AlreadyExists,
            format!("group '{}' already exists", request.to),
            started_at,
        ),
        Err(RenameGroupError::InvalidName) => error(
            ErrorCode::InvalidArgument,
            "the destination name must not be empty".to_string(),
            started_at,
        ),
        Err(RenameGroupError::Io(io_error)) => {
            state.metrics().record_error(ErrorKind::Io);
            error(
                ErrorCode::Internal,
                format!(
                    "group '{name}' rename not fully persisted: {io_error} \
                     (a rename marker remains; the next boot resumes it)"
                ),
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
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(associations): AppJson<Vec<AssocOp>>,
) -> Response {
    let started_at = Instant::now();
    // Refused before the write lock is even taken: nothing of an
    // oversized batch is applied.
    if associations.len() > MAX_ASSOCIATIONS_PER_REQUEST {
        return error(
            ErrorCode::OverLimit,
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
                ErrorCode::InvalidArgument,
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
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let total = associations.len();
    // The write stages ops in the WAL and fsyncs before returning, so
    // keep it off the async worker like `store_passages` and the flush.
    match tokio::task::block_in_place(|| state.add_associations(&name, associations, deadline)) {
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
            // The capacity/conflict split every other batch write
            // reports — association writes only fail on capacity
            // today, but the mapping must not assume that.
            let code = if partial.full {
                ErrorCode::StorageFull
            } else {
                ErrorCode::Conflict
            };
            error(
                code,
                format!(
                    "applied {} of {total} associations, then: {}",
                    partial.applied, partial.message
                ),
                started_at,
            )
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct RetractAssociationRequest {
    pub subject: String,
    pub label: String,
    pub object: String,
}

/// What one association retraction accomplished. `retracted: false`
/// means the triple named no live edge — unknown names, no such edge,
/// or one already fully retracted — and nothing was changed, the same
/// found-nothing honesty `retract_source` answers with.
#[derive(Serialize)]
pub struct RetractAssociationOutcome {
    pub retracted: bool,
    /// How many per-source attribution records were unlinked with the
    /// edge (0 for an edge carrying only unsourced weight).
    pub attributions_removed: usize,
}

/// Withdraws one `(subject, label, object)` association outright —
/// the surgical correction for a fact that should never have been
/// asserted, where `retract_source` would discard the whole
/// document's contribution. A fact that is merely CONTESTED wants a
/// negative-weight assertion instead, which preserves the dispute.
pub async fn retract_association(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<RetractAssociationRequest>,
) -> Response {
    let started_at = Instant::now();
    for (field, value) in [
        ("subject", &request.subject),
        ("label", &request.label),
        ("object", &request.object),
    ] {
        if let Some(refusal) = empty(field, value, started_at) {
            return refusal;
        }
        if let Some(refusal) = oversized(field, value, MAX_NAME_BYTES, started_at) {
            return refusal;
        }
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // The write stages a WAL op and fsyncs before returning; keep it
    // off the async worker like every other write path.
    let outcome = tokio::task::block_in_place(|| {
        state.retract_association(&name, &request.subject, &request.label, &request.object)
    });
    match outcome {
        Err(failure) => access_error(&state, failure, &name, started_at),
        Ok(unlinked) => {
            // The retracted TRIPLE lives in the body, so the access log
            // alone cannot say what was withdrawn — the audit line can
            // (destructive, like retract_source and the deletes).
            tracing::info!(
                target: "taguru::audit",
                key = %key_name(&key),
                context = %name,
                subject = %request.subject,
                label = %request.label,
                object = %request.object,
                retracted = unlinked.is_some(),
                attributions_removed = unlinked.unwrap_or(0),
                "association retracted",
            );
            // A retraction that found nothing changed nothing; only an
            // effective one counts as a write.
            if unlinked.is_some() {
                state.note_write(&name);
            }
            ok(
                RetractAssociationOutcome {
                    retracted: unlinked.is_some(),
                    attributions_removed: unlinked.unwrap_or(0),
                },
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
            ErrorCode::OverLimit,
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

/// Per-request cap on the number of passage sources one store may
/// carry. Each source is a whole document tokenized and folded into the
/// resident index under the context's lock — heavier per item than an
/// association, so a tenth of [`MAX_ASSOCIATIONS_PER_REQUEST`]. The body
/// cap bounds the request's total bytes; this bounds how many documents
/// do that per-item work in a single lock-hold, the same reason an
/// association batch is refused up front. (The offline import calls the
/// registry's `store_passages` directly and is not bound by this.)
pub(crate) const MAX_PASSAGES_PER_REQUEST: usize = 1_000;

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
            ErrorCode::MalformedRequest,
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
            ErrorCode::InvalidArgument,
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
            ErrorCode::InvalidArgument,
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
            ErrorCode::InvalidArgument,
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

/// One cross-context result: the per-context wire shape, tagged with
/// the context it came from — the tag is what makes the result
/// actionable, since every follow-up (citations, lookups, activate)
/// is a per-context call.
#[derive(Serialize)]
pub struct CrossMatch<T> {
    pub context: String,
    #[serde(flatten)]
    pub inner: T,
}

/// [`MatchPage`], cross-context: same `total`-above-count truncation
/// contract, every match tagged.
#[derive(Serialize)]
pub struct CrossMatchPage {
    pub total: usize,
    pub matches: Vec<CrossMatch<AssociationOut>>,
}

/// A match page's resume point: the rank key of the last item on the
/// previous page. `(subject, label, object)` alone already uniquely
/// identifies an edge within one context (the `edge_ids` bijection),
/// so `weight` plus that triple is a total order with no possible tie
/// — a client can always build the next `after` from the last match it
/// received.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatchCursor {
    pub weight: f64,
    pub subject: String,
    pub label: String,
    pub object: String,
}

/// The total order every match page sorts by: strongest |weight| first
/// (the same magnitude-ranks philosophy as activate), ties broken
/// lexicographically on `(subject, label, object)` — deliberately NOT
/// insertion order, since a cursor can only resume a client-visible
/// order.
fn rank(a: (f64, &str, &str, &str), b: (f64, &str, &str, &str)) -> std::cmp::Ordering {
    b.0.abs()
        .total_cmp(&a.0.abs())
        .then_with(|| a.1.cmp(b.1))
        .then_with(|| a.2.cmp(b.2))
        .then_with(|| a.3.cmp(b.3))
}

/// Bounds a match list to one page: past `after` when given, ranked by
/// [`rank`], cut at the limit. Returns the raw library `Association`s,
/// not yet resolved to their wire shape — callers still need to run
/// them through `resolve_sections`/`association_out`, and `page` itself
/// has no context name to resolve against.
fn page(
    matches: Vec<Association>,
    limit: Option<usize>,
    after: Option<&MatchCursor>,
) -> (usize, Vec<Association>) {
    page_by(matches, limit, after, |association| {
        (
            association.weight,
            association.subject.as_str(),
            association.label.as_str(),
            association.object.as_str(),
        )
    })
}

/// [`page`]'s bound over any match shape, the rank key read through an
/// accessor — the cross-context pages carry `(context, association)`
/// pairs rather than bare associations. `total` is captured before the
/// cursor filter and the truncate: it names the query's whole result
/// set, not what remains past `after`, so it stays constant across
/// every page (the same convention `labels`/`ContextPage` use).
fn page_by<T>(
    mut matches: Vec<T>,
    limit: Option<usize>,
    after: Option<&MatchCursor>,
    key: impl Fn(&T) -> (f64, &str, &str, &str),
) -> (usize, Vec<T>) {
    let total = matches.len();
    let limit = clamp(limit, DEFAULT_MATCH_LIMIT, MAX_MATCH_LIMIT);
    if let Some(cursor) = after {
        let seat = (
            cursor.weight,
            cursor.subject.as_str(),
            cursor.label.as_str(),
            cursor.object.as_str(),
        );
        matches.retain(|item| rank(key(item), seat) == std::cmp::Ordering::Greater);
    }
    matches.sort_by(|a, b| rank(key(a), key(b)));
    matches.truncate(limit);
    (total, matches)
}

/// [`explore`]'s resume point: `Context::explore` already produces a
/// full deterministic order via `(distance, edge_id)`, but `edge_id` is
/// internal and never reaches the client — `(subject, label, object)`
/// (the same externally-visible triple [`MatchCursor`] uses) stands in
/// for it. `distance` stays in the key even though the triple alone
/// already disambiguates the row: dropping it would let "ranks after
/// the cursor" compare across distance bands incorrectly.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExploreCursor {
    pub distance: usize,
    pub subject: String,
    pub label: String,
    pub object: String,
}

/// [`ExploreCursor`]'s rank key for one [`Recollection`]: nearest hop
/// first, ties broken lexicographically on `(subject, label, object)`
/// — `path` needs no part in it, since `Context::explore`'s
/// first-write-wins `reached.entry(edge_id)` already guarantees exactly
/// one `Recollection` per `(subject, label, object)`.
fn explore_rank(recollection: &Recollection) -> (usize, &str, &str, &str) {
    let association = &recollection.association;
    (
        recollection.distance,
        association.subject.as_str(),
        association.label.as_str(),
        association.object.as_str(),
    )
}

/// [`page`], for `Context::explore`'s output: same cursor-then-cut
/// contract, ranked by [`explore_rank`] instead of [`rank`]. The
/// re-sort is NOT optional — `Context::explore` orders by
/// `(distance, edge_id)`, and `edge_id` tracks roughly insertion order,
/// not the lexicographic order the cursor resumes, so a same-distance
/// tie's insertion order commonly differs from `explore_rank`'s.
fn explore_page(
    mut matches: Vec<Recollection>,
    after: Option<&ExploreCursor>,
    limit: Option<usize>,
) -> (usize, Vec<Recollection>) {
    let total = matches.len();
    if let Some(cursor) = after {
        let seat = (
            cursor.distance,
            cursor.subject.as_str(),
            cursor.label.as_str(),
            cursor.object.as_str(),
        );
        matches.retain(|recollection| explore_rank(recollection) > seat);
    }
    matches.sort_by(|a, b| explore_rank(a).cmp(&explore_rank(b)));
    matches.truncate(clamp(limit, DEFAULT_MATCH_LIMIT, MAX_MATCH_LIMIT));
    (total, matches)
}

/// Runs `job` for every index in `0..count` on the blocking thread
/// pool, at most `permits` at once, and returns the results in index
/// order (not completion order) — so callers can zip them back against
/// whatever list `count` came from. Built for cross-context fan-out:
/// each `job` call is one context's blocking read, and bounding
/// concurrency keeps a large `contexts`/`groups` list from opening one
/// blocking thread per target at once.
async fn bounded_parallel_map<R: Send + 'static>(
    count: usize,
    permits: usize,
    job: impl Fn(usize) -> R + Send + Sync + 'static,
) -> Vec<R> {
    if count == 0 {
        return Vec::new();
    }
    let permits = permits.clamp(1, count);
    let semaphore = Arc::new(tokio::sync::Semaphore::new(permits));
    let job = Arc::new(job);
    let mut set = tokio::task::JoinSet::new();
    for index in 0..count {
        let permit = Arc::clone(&semaphore)
            .acquire_owned()
            .await
            .expect("semaphore is never closed");
        let job = Arc::clone(&job);
        set.spawn_blocking(move || {
            let _permit = permit;
            (index, job(index))
        });
    }
    let mut slots: Vec<Option<R>> = (0..count).map(|_| None).collect();
    while let Some(outcome) = set.join_next().await {
        let (index, value) = outcome.expect("cross-search job panicked");
        slots[index] = Some(value);
    }
    slots
        .into_iter()
        .map(|slot| slot.expect("every index was joined"))
        .collect()
}

/// Concurrency ceiling for [`bounded_parallel_map`]'s cross-context fan
/// out, `TAGURU_CROSS_SEARCH_CONCURRENCY`-overridable (default 4) —
/// read once and cached, the same `OnceLock` shape as
/// [`search_log_enabled`], since it governs a fan-out shape, not a
/// per-request value.
fn cross_search_concurrency() -> usize {
    static CONCURRENCY: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CONCURRENCY.get_or_init(|| {
        std::env::var("TAGURU_CROSS_SEARCH_CONCURRENCY")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&value: &usize| value > 0)
            .unwrap_or(4)
    })
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

/// [`associations_out`] for a cross-context page: section labels
/// resolve against the context each match came from — one
/// `resolve_sections` call per distinct context on the page, not one
/// per match (and none for a context whose page entries carry no
/// paragraph locator; `resolve_sections` short-circuits on an empty
/// key set).
fn cross_associations_out(
    state: &AppState,
    page: Vec<(String, Association)>,
) -> Vec<CrossMatch<AssociationOut>> {
    let mut locators: BTreeMap<String, Vec<(String, u32)>> = BTreeMap::new();
    for (context, association) in &page {
        locators
            .entry(context.clone())
            .or_default()
            .extend(locator_keys(std::iter::once(association)));
    }
    let sections: BTreeMap<String, HashMap<(String, u32), String>> = locators
        .into_iter()
        .map(|(context, keys)| {
            let resolved = state.resolve_sections(&context, keys.into_iter());
            (context, resolved)
        })
        .collect();
    page.into_iter()
        .map(|(context, association)| {
            let inner = association_out(association, &sections[&context]);
            CrossMatch { context, inner }
        })
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

/// The shared cap on a query's three positions: subject, label, and
/// object are each list-shaped read input, bounded like every other
/// list ([`overlong`]) — `query` and `cross_query` run one check.
fn overlong_positions(
    subject: &Option<OneOrMany>,
    label: &Option<OneOrMany>,
    object: &Option<OneOrMany>,
    started_at: Instant,
) -> Option<Response> {
    [("subject", subject), ("label", label), ("object", object)]
        .into_iter()
        .find_map(|(field, position)| overlong(field, as_refs(position).len(), started_at))
}

/// Refuses a query that pins nothing at all: subject, label, and
/// object empty together would materialize and rank every edge in the
/// context (or, cross-context, every edge in every named target)
/// before the limit ever trims it — treated as a client bug, not a
/// deliberate "give me everything", the same stance [`cross_targets`]
/// takes on an empty `contexts`/`groups` pair.
fn empty_positions(
    subject: &Option<OneOrMany>,
    label: &Option<OneOrMany>,
    object: &Option<OneOrMany>,
    started_at: Instant,
) -> Option<Response> {
    let nothing_pinned =
        as_refs(subject).is_empty() && as_refs(label).is_empty() && as_refs(object).is_empty();
    nothing_pinned.then(|| {
        error(
            ErrorCode::InvalidArgument,
            "'subject', 'label', or 'object' must pin at least one value",
            started_at,
        )
    })
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
/// `prefix` narrows the population of interest itself (like `total` on
/// the search endpoints, not a cursor) — endpoints that support it
/// apply it before counting `total`; `list_groups` ignores it.
#[derive(Debug, Deserialize)]
pub struct KeysetQuery {
    pub limit: Option<usize>,
    pub after: Option<String>,
    pub prefix: Option<String>,
}

pub async fn add_aliases(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<AliasRequest>,
) -> Response {
    let started_at = Instant::now();
    // Refused before the write lock like an association batch — each
    // pair is one WAL op in a single lock-hold/fsync, the same cost
    // shape MAX_ASSOCIATIONS_PER_REQUEST bounds there.
    let pairs = request.concepts.len() + request.labels.len();
    if pairs > MAX_ASSOCIATIONS_PER_REQUEST {
        return error(
            ErrorCode::OverLimit,
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
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Same fsync-bearing WAL write as add_associations; keep it off the
    // async worker.
    match tokio::task::block_in_place(|| {
        state.add_aliases(&name, &request.concepts, &request.labels)
    }) {
        Err(failure) => access_error(&state, failure, &name, started_at),
        Ok(Ok(applied)) => {
            // Counts (not the spellings — a batch may run to
            // thousands) reach the audit line, the access log names
            // the context and key. Unconditional, symmetric with
            // remove_aliases: even an empty batch (applied == 0)
            // leaves a line, so an operator reconstructing a bad
            // alias's live window sees every registration attempt.
            tracing::info!(
                target: "taguru::audit",
                key = %key_name(&key),
                context = %name,
                concepts = request.concepts.len(),
                labels = request.labels.len(),
                applied,
                "aliases registered",
            );
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
            let code = if partial.full {
                ErrorCode::StorageFull
            } else {
                ErrorCode::Conflict
            };
            error(
                code,
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
    AppPath(name): AppPath<String>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<RemoveAliasesRequest>,
) -> Response {
    let started_at = Instant::now();
    // An empty withdrawal is a malformed request, not a silent no-op.
    if request.concepts.is_empty() && request.labels.is_empty() {
        return error(
            ErrorCode::InvalidArgument,
            "the request names no aliases to remove",
            started_at,
        );
    }
    // Same WAL-op-per-item cost shape as add_aliases; same cap, and the
    // same `OverLimit` code add_aliases uses for the over-cap batch — an
    // oversized request is not a malformed one.
    let names = request.concepts.len() + request.labels.len();
    if names > MAX_ASSOCIATIONS_PER_REQUEST {
        return error(
            ErrorCode::OverLimit,
            format!(
                "batch of {names} alias removals exceeds the per-request limit of \
                 {MAX_ASSOCIATIONS_PER_REQUEST}; split the request"
            ),
            started_at,
        );
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
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
            let code = if partial.full {
                ErrorCode::StorageFull
            } else {
                ErrorCode::Conflict
            };
            error(
                code,
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
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
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
                    ErrorCode::InvalidArgument,
                    "after must be 'concept:<alias>' or 'label:<alias>' — the last \
                     entry of the previous page",
                    started_at,
                );
            }
        },
    };
    let limit = clamp(query.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT);
    // Namespace order is concepts-then-labels, same as the wire cursor's
    // `(is_label, alias)` ordering. A cursor already inside the label
    // namespace means every concept alias is behind it, so the concept
    // seek is skipped outright rather than run and immediately discarded.
    let (concept_after, label_after, skip_concepts) = match after {
        None => (None, None, false),
        Some((false, alias)) => (Some(alias), None, false),
        Some((true, alias)) => (None, Some(alias), true),
    };
    // A `prefix` filter forces the whole-namespace scan below; a bare
    // cursor stays on the cheap BTreeMap-seeking path regardless.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    match state.read_context(&name, |context| {
        // A `prefix` filter defines the population rather than a
        // cursor, so — like `pinned` on `list_contexts` — it forces the
        // whole-namespace path: the BTreeMap-seeking `*_alias_page` fast
        // path has no way to know in advance how many prefix-matching
        // aliases lie within any given range.
        if let Some(prefix) = query.prefix.as_deref() {
            let mut concepts: Vec<(String, String)> = context
                .concept_aliases()
                .into_iter()
                .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
                .collect();
            concepts.sort();
            concepts.retain(|(alias, _)| alias.starts_with(prefix));
            let mut labels: Vec<(String, String)> = context
                .label_aliases()
                .into_iter()
                .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
                .collect();
            labels.sort();
            labels.retain(|(alias, _)| alias.starts_with(prefix));
            let total = concepts.len() + labels.len();
            // One ordered sequence — concepts, then labels — filtered
            // past the cursor and cut at the limit, then split back
            // into maps.
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
            return export;
        }
        let total = context.concept_alias_count() + context.label_alias_count();
        let mut export = AliasExport {
            total,
            concepts: BTreeMap::new(),
            labels: BTreeMap::new(),
        };
        let mut remaining = limit;
        if !skip_concepts {
            let (_, page) = context.concept_alias_page(concept_after, remaining);
            remaining -= page.len();
            export.concepts.extend(page);
        }
        // A concept page shorter than what was asked for means that
        // namespace ran dry, so the leftover budget spills into labels,
        // started fresh — reached only when `label_after` is `None`,
        // since a label-namespace cursor takes the `skip_concepts`
        // branch above and never sets `remaining` here.
        if remaining > 0 {
            let (_, page) = context.label_alias_page(label_after, remaining);
            export.labels.extend(page);
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
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<StorePassagesRequest>,
) -> Response {
    let started_at = Instant::now();
    // Refused before any tokenization or lock is taken: nothing of an
    // oversized batch is stored.
    if request.passages.len() > MAX_PASSAGES_PER_REQUEST {
        return error(
            ErrorCode::OverLimit,
            format!(
                "batch of {} passages exceeds the per-request limit of \
                 {MAX_PASSAGES_PER_REQUEST}; split the store",
                request.passages.len()
            ),
            started_at,
        );
    }
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
                    ErrorCode::InvalidArgument,
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
                    ErrorCode::InvalidArgument,
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
                    ErrorCode::InvalidArgument,
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
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
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
                ErrorCode::Internal,
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

/// Shared body of `audit_vocabulary` and `audit_drift`'s `include_twins`
/// section: lexical fork candidates always, semantic ones when
/// embeddings are configured and the deadline allows. Callers run this
/// inside `block_in_place` — it does its own CPU-bound pairwise sweeps
/// and must never run on an async worker (see the comment this carried
/// forward from `audit_vocabulary`, below).
fn vocabulary_audit(
    state: &AppState,
    name: &str,
    dice_floor: f64,
    cosine_floor: f32,
    deadline: Deadline,
) -> Result<VocabularyAudit, AccessError> {
    // BOTH halves are CPU-bound pairwise sweeps — the lexical one is
    // O(Σ posting_len²) over the whole vocabulary, seconds at tens of
    // thousands of concepts. Neither may run on an async worker: with
    // the lexical half inline, a handful of concurrent audits pinned
    // every worker and starved every other request, /health included.
    let lexical = state
        .read_context(name, |context| {
            let concepts = context
                .similar_concepts(dice_floor, deadline)
                .map_err(|_| AccessError::DeadlineExceeded)?;
            let labels = context
                .similar_labels(dice_floor, deadline)
                .map_err(|_| AccessError::DeadlineExceeded)?;
            Ok((concepts, labels))
        })
        .and_then(std::convert::identity)?;

    // The lexical half already spent the budget checking its own
    // deadline; skip the semantic half rather than fail a request that
    // did real, useful work — semantic_twins also checks this same
    // deadline internally (its own pairwise sweep can still be large),
    // so this is a courtesy early-out on top of that, not the only guard.
    if deadline.expired() {
        return Ok(VocabularyAudit {
            lexical_concepts: twin_pairs(lexical.0),
            lexical_labels: twin_pairs(lexical.1),
            semantic_concepts: Vec::new(),
            semantic_labels: Vec::new(),
            semantic_note: Some("意味的検出はスキップ (期限切れ)".to_string()),
        });
    }
    let Some((semantic_concepts, semantic_labels, semantic_note)) =
        state.semantic_twins(name, cosine_floor, deadline)
    else {
        return Err(AccessError::NotFound);
    };

    Ok(VocabularyAudit {
        lexical_concepts: twin_pairs(lexical.0),
        lexical_labels: twin_pairs(lexical.1),
        semantic_concepts: twin_pairs(semantic_concepts),
        semantic_labels: twin_pairs(semantic_labels),
        semantic_note,
    })
}

pub async fn audit_vocabulary(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppBytes(body): AppBytes,
) -> Response {
    let started_at = Instant::now();
    let request: VocabularyAuditRequest = match optional_body(&body, started_at) {
        Ok(request) => request,
        Err(refusal) => return *refusal,
    };
    let dice_floor = request.dice_floor.unwrap_or(0.6);
    let cosine_floor = request.cosine_floor.unwrap_or(0.6);

    // vocabulary_audit's lexical half builds an O(n) per-span bigram
    // table before its own inner loop ever checks this deadline — fail
    // a spent request before that table gets built, the way every other
    // heavy handler pre-flights.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    match tokio::task::block_in_place(|| {
        vocabulary_audit(&state, &name, dice_floor, cosine_floor, deadline)
    }) {
        Ok(audit) => ok(audit, started_at),
        Err(failure) => access_error(&state, failure, &name, started_at),
    }
}

/// One edge [`Context::unsourced_edges`] flagged, reshaped to the wire
/// (section-resolved association, same as every other match page).
#[derive(Serialize)]
pub struct UnsourcedEdgeOut {
    pub unsourced_weight: f64,
    pub unsourced_count: u64,
    pub association: AssociationOut,
}

/// Drift audit request: an optional floor for the unsourced-weight
/// section, paging over it, and an opt-in for the vocabulary-twins
/// section (skipped by default — it's the same CPU-bound pairwise
/// sweep `audit_vocabulary` runs, not something every drift check
/// should pay for).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct DriftAuditRequest {
    /// Minimum unsourced weight (compared by magnitude) to include;
    /// omitted means any amount at all.
    pub unsourced_floor: Option<f64>,
    /// Omitted means 100, capped at 1000 — pages exactly like recall,
    /// query, and unreachable_from.
    pub limit: Option<usize>,
    /// Resume past a previous page's last match — see [`MatchCursor`].
    pub after: Option<MatchCursor>,
    /// Also run the lexical/semantic fork-candidate sweep
    /// (`vocabulary_audit`) and include it as `twins`.
    pub include_twins: bool,
    /// Lexical (spelling) detector floor; omitted means 0.6. Ignored
    /// unless `include_twins` is set.
    pub dice_floor: Option<f64>,
    /// Semantic (gloss cosine) detector floor; omitted means 0.6.
    /// Ignored unless `include_twins` is set.
    pub cosine_floor: Option<f32>,
}

/// The drift audit: three independent read-only checks in one
/// response, each answering a different way the graph's current shape
/// can have drifted from what actually happened. `unsourced`/`total`
/// page like every other match list; the alias and twins sections are
/// small enough in practice to return whole.
#[derive(Serialize)]
pub struct DriftAudit {
    pub total: usize,
    pub unsourced: Vec<UnsourcedEdgeOut>,
    pub dead_concept_aliases: BTreeMap<String, String>,
    pub dead_label_aliases: BTreeMap<String, String>,
    /// Present only when `include_twins` was set.
    pub twins: Option<VocabularyAudit>,
}

pub async fn audit_drift(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    axum::Extension(heavy_ops): axum::Extension<HeavyOpsLimiter>,
    AppBytes(body): AppBytes,
) -> Response {
    let started_at = Instant::now();
    let request: DriftAuditRequest = match optional_body(&body, started_at) {
        Ok(request) => request,
        Err(refusal) => return *refusal,
    };
    let floor = request.unsourced_floor.unwrap_or(0.0);

    // The two sweeps below each check this deadline only once they are
    // already inside their own edge loop — fail a spent request before
    // read_context even takes the registry lock, the way every other
    // heavy handler pre-flights.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let loaded = tokio::task::block_in_place(|| {
        state
            .read_context(&name, |context| {
                let unsourced = context
                    .unsourced_edges(floor, deadline)
                    .map_err(|_| AccessError::DeadlineExceeded)?;
                let aliases = context
                    .dead_canonical_aliases(deadline)
                    .map_err(|_| AccessError::DeadlineExceeded)?;
                Ok((unsourced, aliases))
            })
            .and_then(std::convert::identity)
    });
    let (unsourced, (dead_concept_aliases, dead_label_aliases)) = match loaded {
        Ok(loaded) => loaded,
        Err(failure) => return access_error(&state, failure, &name, started_at),
    };

    let (total, unsourced) = page_by(unsourced, request.limit, request.after.as_ref(), |edge| {
        (
            edge.weight,
            edge.association.subject.as_str(),
            edge.association.label.as_str(),
            edge.association.object.as_str(),
        )
    });
    // A graph read like unreachable_from — zero drift is the audit
    // succeeding, not a miss, so it never counts as an empty read.
    state.note_read(&name, false);
    let sections = state.resolve_sections(
        &name,
        locator_keys(unsourced.iter().map(|edge| &edge.association)),
    );
    let unsourced = unsourced
        .into_iter()
        .map(|edge| UnsourcedEdgeOut {
            unsourced_weight: edge.weight,
            unsourced_count: edge.count,
            association: association_out(edge.association, &sections),
        })
        .collect();

    let twins = if request.include_twins {
        // Only this branch runs the CPU-bound pairwise scan
        // `audit_vocabulary` also runs, so it alone spends a
        // heavy-ops permit — the unsourced/alias sweeps above are a
        // cheap O(n) pass that a full heavy-ops ceiling would
        // otherwise gate for no reason.
        let _permit = match heavy_ops.try_acquire() {
            Ok(permit) => permit,
            Err(shed_response) => return *shed_response,
        };
        let dice_floor = request.dice_floor.unwrap_or(0.6);
        let cosine_floor = request.cosine_floor.unwrap_or(0.6);
        match tokio::task::block_in_place(|| {
            vocabulary_audit(&state, &name, dice_floor, cosine_floor, deadline)
        }) {
            Ok(audit) => Some(audit),
            Err(failure) => return access_error(&state, failure, &name, started_at),
        }
    } else {
        None
    };

    ok(
        DriftAudit {
            total,
            unsourced,
            dead_concept_aliases,
            dead_label_aliases,
            twins,
        },
        started_at,
    )
}

/// `POST /import`'s query string.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct ImportQuery {
    /// Report what the stream would do without writing anything — no
    /// context created, no source retracted, nothing stored. See
    /// [`crate::ingest::preview_batch`] for which counts are exact and
    /// which are advisory.
    pub dry_run: bool,
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

/// What a `POST /import` body accomplished — one [`ImportOutcome`] per
/// batch, in stream order. Every import answers this shape, a
/// single-batch body included: one shape to parse, not two.
#[derive(Serialize)]
pub struct ImportStreamOutcome {
    pub batches: Vec<ImportOutcome>,
    /// One entry per `taguru_group` record, stream order — absent
    /// entirely for a stream that carried none, keeping the pre-group
    /// response byte-identical.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<GroupImportOutcome>,
}

/// What restoring one group record accomplished. A restore is a
/// replace of the whole record; the label says what it replaced.
#[derive(Serialize)]
pub struct GroupImportOutcome {
    pub name: String,
    /// `"created"` | `"replaced"` | `"unchanged"` — the last says the
    /// standing record already matched, so nothing was rewritten.
    pub outcome: &'static str,
    /// Member counts of the record as restored.
    pub contexts: usize,
    pub groups: usize,
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
        // The three AccessError arms — status, metric, message — live
        // in access_error_noted; import just supplies the batch note.
        crate::ingest::ApplyRefusal::Access(failure) => {
            access_error_noted(state, failure, &batch.context, note, started_at)
        }
        refusal @ crate::ingest::ApplyRefusal::NoContext(_) => error(
            ErrorCode::NoContext,
            format!("{note}{}", refusal.text()),
            started_at,
        ),
        refusal @ crate::ingest::ApplyRefusal::Io(_) => {
            state.metrics().record_error(ErrorKind::Io);
            error(
                ErrorCode::Internal,
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
            let code = if full {
                ErrorCode::StorageFull
            } else {
                ErrorCode::Conflict
            };
            error(code, format!("{note}{message}"), started_at)
        }
    }
}

/// Maps a refused group restore onto the response — the group half of
/// [`import_refusal`]. Groups apply after every batch, so the note
/// says what already landed; re-POSTing the corrected stream is exact
/// (batches replace their sources, records their groups).
fn restore_refusal(
    state: &AppState,
    refusal: crate::registry::RestoreGroupsError,
    batches_landed: usize,
    started_at: Instant,
) -> Response {
    use crate::registry::RestoreGroupsError;
    let code = match &refusal {
        RestoreGroupsError::InvalidName | RestoreGroupsError::Duplicate(_) => {
            ErrorCode::InvalidArgument
        }
        RestoreGroupsError::NoSuchContext { .. } => ErrorCode::NoContext,
        RestoreGroupsError::NoSuchChild { .. } => ErrorCode::NoGroup,
        RestoreGroupsError::OverCap { .. } => ErrorCode::OverLimit,
        RestoreGroupsError::Nesting(violation) => nesting_error_code(violation),
        RestoreGroupsError::Io { .. } => {
            state.metrics().record_error(ErrorKind::Io);
            ErrorCode::Internal
        }
        // A spent budget is not a server fault — the batch-loop timeout
        // records no error either, and the client resumes by re-POSTing.
        RestoreGroupsError::Timeout { .. } => ErrorCode::Timeout,
    };
    // A budget that ran out mid-restore is a resumable prefix, not a
    // rejected set: some records may have landed, and the batches
    // before them did. The validation arms, by contrast, wrote nothing
    // in the group phase.
    let message = match &refusal {
        RestoreGroupsError::Timeout { .. } => format!(
            "group restore exceeded its budget with {batches_landed} batch(es) durable \
             (TAGURU_REQUEST_TIMEOUT_SECS tunes this); {}",
            refusal.text()
        ),
        _ => format!(
            "group records refused with every batch landed ({batches_landed} durable); \
             fixing the stream and re-POSTing it whole is exact: {}",
            refusal.text()
        ),
    };
    error(code, message, started_at)
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
/// `taguru_group` records ride the same stream and apply LAST — after
/// every batch, wherever they sat — so a group and the member
/// contexts it names can travel in one body in any order. Restoring a
/// record replaces the whole group; the set is validated whole and a
/// refusal applies no group, with every batch already durable.
///
/// The response is `{batches: [...]}` in stream order — a single-batch
/// body answers the same shape with one entry, and a stream that
/// carried group records adds `groups: [...]`. A refusal partway
/// through a stream stops there — the batches before it landed
/// durably, and because every batch is retract-then-apply,
/// re-POSTing the whole corrected stream is exact, never
/// double-counted.
///
/// `?dry_run=true` reports the same `{batches: [...]}` shape without
/// writing anything — parsing and scope checks still run in full, so a
/// malformed or forbidden stream is refused exactly as it would be for
/// real. Two counts per batch, `associations` and `aliases`, are
/// optimistic (see [`crate::ingest::preview_batch`]); every other
/// field is exact. `taguru_group` records are a known gap: they apply
/// through a separate path (`restore_groups`) that dry-run does not
/// preview, so a stream carrying any is parsed and scope-checked like
/// normal but its `groups` are silently not previewed — the response
/// omits `groups` entirely rather than report a guess.
pub async fn import_batch(
    State(state): State<AppState>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppQuery(query): AppQuery<ImportQuery>,
    AppBytes(body): AppBytes,
) -> Response {
    let started_at = Instant::now();
    let stream = match crate::ingest::parse_stream(&body[..]) {
        Ok(stream) => stream,
        // Line-numbered, like the CLI's validation pass.
        Err(message) => return error(ErrorCode::MalformedRequest, message, started_at),
    };
    // Import's contexts live in the BODY, out of the route-level
    // authorization check's reach — a context-scoped key is judged
    // here instead, before anything applies.
    if let Some(axum::Extension(scope)) = &scope
        && let Some(refused) = stream
            .batches
            .iter()
            .find(|batch| !scope.allows_context(&batch.context))
    {
        return error(
            ErrorCode::Forbidden,
            format!(
                "key '{}' has no grant on context '{}' (batch source '{}'); nothing \
                 was applied",
                key_name(&key),
                refused.context,
                refused.source
            ),
            started_at,
        );
    }
    // Group records are judged the way every group write is: by the
    // context closure — the standing record's and the prospective
    // one's both — before anything applies. Gated on the scope first,
    // [`scoped_group_refusal`]'s discipline: an unscoped key never
    // pays for the closure read.
    if scope.is_some()
        && !stream.groups.is_empty()
        && let Some(refusal) = scope_refusal(
            &scope,
            &key,
            &state.group_restore_involves(&stream.groups),
            started_at,
        )
    {
        return refusal;
    }
    let total = stream.batches.len();
    // Each batch is a create/retract_source/store_passages/add_associations/
    // add_aliases sequence — fsync-bearing writes, same as every other
    // mutating endpoint — run back to back; keep the whole loop off the
    // async worker rather than just one call in it.
    let outcome = tokio::task::block_in_place(|| {
        let mut outcomes: Vec<ImportOutcome> = Vec::with_capacity(total);
        for (index, batch) in stream.batches.iter().enumerate() {
            // Each landed batch is durable (retract-then-apply), so a
            // budget that runs out partway is safe to report as a
            // resumable prefix rather than an all-or-nothing failure.
            if deadline.expired() {
                let note = if total > 1 && query.dry_run {
                    format!(
                        "batch {} of {total} (context '{}', source '{}') not previewed — \
                         the {} batch(es) before it previewed clean (dry_run=true; \
                         nothing is written); re-running the preview with more time or \
                         a narrower stream is exact: ",
                        index + 1,
                        batch.context,
                        batch.source,
                        outcomes.len(),
                    )
                } else if total > 1 {
                    format!(
                        "batch {} of {total} (context '{}', source '{}') not attempted — \
                         the {} batch(es) before it landed durably; re-POSTing the \
                         remaining stream is exact (each batch replaces its own source): ",
                        index + 1,
                        batch.context,
                        batch.source,
                        outcomes.len(),
                    )
                } else {
                    String::new()
                };
                return Err(Box::new(error(
                    ErrorCode::Timeout,
                    format!(
                        "{note}request exceeded its budget partway through a multi-batch \
                         import (TAGURU_REQUEST_TIMEOUT_SECS tunes this)"
                    ),
                    started_at,
                )));
            }
            let applied = if query.dry_run {
                crate::ingest::preview_batch(&state, batch)
            } else {
                crate::ingest::apply_batch(&state, batch)
            };
            match applied {
                Ok(applied) => {
                    // Import is retract-then-apply — destructive — and both
                    // the context and the replaced source live in the BODY,
                    // out of the access log's reach. One audit line each.
                    // `dry_run` marks the ones that wrote nothing.
                    tracing::info!(
                        target: "taguru::audit",
                        key = %key_name(&key),
                        context = %batch.context,
                        source = %batch.source,
                        created = applied.created,
                        retracted = applied.retracted,
                        associations = applied.associations,
                        dry_run = query.dry_run,
                        "import batch applied",
                    );
                    outcomes.push(import_outcome(batch, &applied));
                }
                Err(refusal) => {
                    let note = if total > 1 && query.dry_run {
                        format!(
                            "batch {} of {total} (context '{}', source '{}') would be \
                             refused — the {} batch(es) before it previewed clean \
                             (dry_run=true; nothing is written); fixing the stream and \
                             re-running the preview is exact: ",
                            index + 1,
                            batch.context,
                            batch.source,
                            outcomes.len(),
                        )
                    } else if total > 1 {
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
                    return Err(Box::new(import_refusal(
                        &state, batch, refusal, &note, started_at,
                    )));
                }
            }
        }
        // Groups apply LAST — after every batch — so a record and the
        // member contexts it names can ride one stream in any order.
        // `taguru_group` records apply through a separate path
        // (`restore_groups`) that has no read-only twin, so a dry run
        // skips them entirely — the response omits `groups` rather than
        // report a guess.
        let mut group_outcomes: Vec<GroupImportOutcome> = Vec::new();
        if !stream.groups.is_empty() && !query.dry_run {
            match state.restore_groups(&stream.groups, deadline) {
                Ok(restored) => {
                    for ((name, record), (_, applied)) in stream.groups.iter().zip(&restored) {
                        // A restore replaces the whole record —
                        // destructive like the batches above, and just
                        // as far out of the access log's reach.
                        tracing::info!(
                            target: "taguru::audit",
                            key = %key_name(&key),
                            group = %name,
                            outcome = applied.as_str(),
                            contexts = record.contexts.len(),
                            children = record.groups.len(),
                            "import group record applied",
                        );
                        group_outcomes.push(GroupImportOutcome {
                            name: name.clone(),
                            outcome: applied.as_str(),
                            contexts: record.contexts.len(),
                            groups: record.groups.len(),
                        });
                    }
                }
                Err(refusal) => {
                    return Err(Box::new(restore_refusal(
                        &state, refusal, total, started_at,
                    )));
                }
            }
        }
        Ok((outcomes, group_outcomes))
    });
    let (outcomes, group_outcomes) = match outcome {
        Ok(applied) => applied,
        Err(refusal) => return *refusal,
    };
    ok(
        ImportStreamOutcome {
            batches: outcomes,
            groups: group_outcomes,
        },
        started_at,
    )
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
    AppPath(name): AppPath<String>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    match tokio::task::block_in_place(|| state.compact_context(&name, deadline)) {
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
pub async fn export_context(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let rendered = tokio::task::block_in_place(|| {
        state
            .export_context(&name, deadline)
            .map(|snapshot| crate::export::render(&name, &snapshot, deadline))
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
        // Mirrors search_passages: render()'s own loop over a large
        // context's associations/aliases can outlast the budget after
        // the entry check above already passed — reclassify by asking
        // the same deadline again rather than by matching render()'s
        // message text.
        Ok(Err(_)) if deadline.expired() => deadline_exceeded(started_at),
        // A real source colliding with a reserved export id — the one
        // thing a context can hold that the stream cannot say.
        Ok(Err(message)) => error(ErrorCode::Conflict, message, started_at),
        Err(failure) => access_error(&state, failure, &name, started_at),
    }
}

/// `GET /groups/{name}/export` — the group back out as its import
/// record: one `taguru_group` line (JSON Lines body, not the JSON
/// envelope), [`export_context`]'s twin one storey up. `POST /import`
/// (or `taguru import`) restores it by replacing the whole record, so
/// re-importing is idempotent; the members must exist at import time,
/// and batches of the same stream apply first. For a context-scoped
/// key the members are the grant's slice, exactly as
/// `GET /groups/{name}` serves them — the export IS that key's view,
/// and restoring it elsewhere carries only what the key could see.
pub async fn export_group(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
) -> Response {
    let started_at = Instant::now();
    let Some(record) = state.group(&name) else {
        return group_not_found(&name, started_at);
    };
    let filtered = GroupRecord {
        description: record.description,
        contexts: scoped_member_contexts(record.contexts, &scope),
        // Child names stay whole, as on the row: labels, not content.
        groups: record.groups,
    };
    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "application/x-ndjson; charset=utf-8",
        )],
        crate::export::render_group(&name, &filtered),
    )
        .into_response()
}

#[derive(Debug, Deserialize)]
pub struct RecallRequest {
    pub cue: String,
    /// Omitted means 100.
    pub limit: Option<usize>,
    /// Resume past a previous page's last match — see [`MatchCursor`].
    pub after: Option<MatchCursor>,
}

pub async fn recall(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<RecallRequest>,
) -> Response {
    let started_at = Instant::now();
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    match state.read_context(&name, |context| context.recall(&request.cue)) {
        Ok(result) => {
            let (total, matches) = page(result, request.limit, request.after.as_ref());
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

/// Vets a cross-context search's target list — the directly named
/// contexts plus every context the named groups reach, nested children
/// included — and returns it deduped: direct names lead in
/// first-appearance order, group-resolved members follow in name
/// order (the tie order the passage merge documents). Refused, in
/// order: naming nothing at all (a search of nothing is a client bug,
/// not an empty result — and emphatically not "every context"); either
/// list over the input-items cap; a direct name beyond the key's grant
/// ([`scope_refusal`] — whole-request, and before existence, so grants
/// cannot probe names); the first direct name that does not exist
/// (`no_context`, before any context is searched); and the first group
/// name that is not a group (`no_group` — group rows are visible to
/// every key, so that refusal probes nothing). Group-RESOLVED members
/// beyond the grant are dropped, not refused: a scoped key searches
/// its slice of a group exactly as `GET /groups` shows it that slice
/// ([`group_entry`]) — refusing would name out-of-grant members and
/// leak what the listing hides. The slice can come up empty; a legal
/// request that resolves to nothing is an empty result, not an error.
fn cross_targets(
    state: &AppState,
    scope: &Option<axum::Extension<crate::auth::KeyScope>>,
    key: &Option<axum::Extension<crate::auth::AuthKey>>,
    contexts: Vec<String>,
    groups: Vec<String>,
    started_at: Instant,
) -> Result<Arc<[String]>, Box<Response>> {
    if contexts.is_empty() && groups.is_empty() {
        return Err(Box::new(error(
            ErrorCode::InvalidArgument,
            "'contexts' or 'groups' must name at least one target",
            started_at,
        )));
    }
    for (field, count) in [("contexts", contexts.len()), ("groups", groups.len())] {
        if let Some(refusal) = overlong(field, count, started_at) {
            return Err(Box::new(refusal));
        }
    }
    let mut seen = BTreeSet::new();
    let mut targets: Vec<String> = contexts
        .into_iter()
        .filter(|name| seen.insert(name.clone()))
        .collect();
    if let Some(refusal) = scope_refusal(scope, key, &targets, started_at) {
        return Err(Box::new(refusal));
    }
    if let Some(missing) = targets.iter().find(|name| !state.context_exists(name)) {
        return Err(Box::new(error(
            ErrorCode::NoContext,
            format!("context '{missing}' not found"),
            started_at,
        )));
    }
    // Resolution is skipped outright when no groups were named: a
    // context-only search must never queue behind a group write's
    // fsync on the groups lock (see the registry field's doc).
    if !groups.is_empty() {
        let resolved = match state.resolve_groups(&groups) {
            Ok(resolved) => resolved,
            Err(missing) => return Err(Box::new(group_not_found(&missing, started_at))),
        };
        targets.extend(
            resolved
                .into_iter()
                .filter(|name| scope_allows(scope, name) && seen.insert(name.clone())),
        );
    }
    Ok(targets.into())
}

/// One cross-context result page: the pre-cut total, and the surviving
/// `(context, association)` pairs in page order.
type CrossPage = (usize, Vec<(String, Association)>);

/// [`MatchCursor`], cross-context: `(subject, label, object)` only
/// identifies an edge *within* one context's `edge_ids` map, so two
/// different target contexts can each hold an edge with the identical
/// triple — `context` joins the key as a fifth field to keep the
/// merged pool's order total. Every wire match already carries
/// `context` ([`CrossMatch`]'s flattened shape), so a client builds
/// this from the last match it received exactly as it builds
/// [`MatchCursor`].
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossMatchCursor {
    pub weight: f64,
    pub context: String,
    pub subject: String,
    pub label: String,
    pub object: String,
}

/// [`CrossMatchCursor`]'s rank key for one pooled `(context,
/// association)` pair.
fn cross_key<'a>(
    context: &'a str,
    found: &'a Association,
) -> (f64, &'a str, &'a str, &'a str, &'a str) {
    (
        found.weight,
        context,
        found.subject.as_str(),
        found.label.as_str(),
        found.object.as_str(),
    )
}

/// [`rank`], cross-context: the same strongest-|weight|-first order
/// with `context` spliced in ahead of the `(subject, label, object)`
/// tiebreak. A separate small function rather than one generalized
/// N-tuple comparator — the two key shapes are concretely different
/// arities, and this codebase prefers explicit small functions over
/// generic machinery built for two call sites.
fn cross_rank(
    a: (f64, &str, &str, &str, &str),
    b: (f64, &str, &str, &str, &str),
) -> std::cmp::Ordering {
    b.0.abs()
        .total_cmp(&a.0.abs())
        .then_with(|| a.1.cmp(b.1))
        .then_with(|| a.2.cmp(b.2))
        .then_with(|| a.3.cmp(b.3))
        .then_with(|| a.4.cmp(b.4))
}

/// [`page_by`], cross-context: same cursor-then-sort-then-cut contract,
/// ranked by [`cross_rank`] instead of [`rank`]. Not generic over the
/// pooled shape the way `page_by` is — `targets` is taken directly
/// rather than folded into a `key` closure, since a closure generic
/// enough to cover any `T` would have to name its output's borrows
/// with a single elided lifetime tied to the per-item `&T` it is
/// handed, and `targets` (this function's own parameter, borrowed for
/// its whole call) does not fit that shape: it outlives any one
/// comparison, but a `for<'r> Fn(&'r T) -> (..., &'r str, ...)` bound
/// cannot say so.
fn cross_page_by(
    mut matches: Vec<(usize, Association)>,
    limit: Option<usize>,
    after: Option<&CrossMatchCursor>,
    targets: &[String],
) -> (usize, Vec<(usize, Association)>) {
    let total = matches.len();
    let limit = clamp(limit, DEFAULT_MATCH_LIMIT, MAX_MATCH_LIMIT);
    if let Some(cursor) = after {
        let seat = (
            cursor.weight,
            cursor.context.as_str(),
            cursor.subject.as_str(),
            cursor.label.as_str(),
            cursor.object.as_str(),
        );
        matches.retain(|(index, found)| {
            cross_rank(cross_key(&targets[*index], found), seat) == std::cmp::Ordering::Greater
        });
    }
    matches.sort_by(|(ia, a), (ib, b)| {
        cross_rank(cross_key(&targets[*ia], a), cross_key(&targets[*ib], b))
    });
    matches.truncate(limit);
    (total, matches)
}

/// The shared middle of the cross-context graph searches: gather every
/// target's search concurrently ([`bounded_parallel_map`], bounded by
/// [`cross_search_concurrency`]), pool the matches, cut past the
/// limit, and only then tag the survivors with their context names —
/// naming every match up front would allocate thousands of strings
/// just to throw them away. [`cross_page_by`] makes every cut, so
/// there is exactly one comparator.
///
/// The in-loop cut holds the MEMORY bound: it fires at twice the limit
/// and comes back to the limit, so the pool never holds more than
/// twice the limit plus one target's matches, and each firing discards
/// at least `limit` entries. It filters by the *same* `after` cursor as
/// the closing cut, not `None` — a cursor-blind mid-loop cut would,
/// whenever the cursor sits deep in the ranking, keep only the
/// strongest raw-weight items (exactly the ones that rank *before* the
/// cursor and get discarded at the close anyway) while prematurely
/// discarding the weaker, legitimately-after-cursor items a deep page
/// actually needs. This is safe and exact: "ranks strictly after the
/// cursor" is a pure per-item predicate independent of pool
/// membership, so `retain` commutes with the incremental union the
/// loop performs, and top-K-after-filter stays monotonic under
/// superset growth — whatever a prefix pool's filtered-and-cut
/// survivors exclude, every superset's does too.
///
/// Every target's fetch now lands concurrently rather than in list
/// order, so "the first per-context failure aborts the whole response"
/// means the first failure in target-list order once every fetch has
/// landed, not the first one hit in real time — the response is
/// identical either way, since a read has nothing to half-apply and
/// every fetch has to land before any cut can run.
async fn cross_matches(
    state: &AppState,
    targets: &Arc<[String]>,
    op: SearchOp,
    limit: Option<usize>,
    after: Option<&CrossMatchCursor>,
    search: impl Fn(&Context) -> Vec<Association> + Send + Sync + 'static,
    started_at: Instant,
) -> Result<CrossPage, Box<Response>> {
    let limit = clamp(limit, DEFAULT_MATCH_LIMIT, MAX_MATCH_LIMIT);
    let permits = cross_search_concurrency().min(targets.len().max(1));
    let owned_targets = Arc::clone(targets);
    let job_state = state.clone();
    let fetched = bounded_parallel_map(targets.len(), permits, move |index| {
        job_state.read_context(&owned_targets[index], &search)
    })
    .await;

    let mut total = 0;
    let mut pool: Vec<(usize, Association)> = Vec::new();
    for (index, outcome) in fetched.into_iter().enumerate() {
        match outcome {
            Ok(matches) => {
                state.note_search(op, &targets[index], matches.is_empty());
                total += matches.len();
                pool.extend(matches.into_iter().map(|found| (index, found)));
                if pool.len() >= limit * 2 {
                    pool = cross_page_by(pool, Some(limit), after, targets).1;
                }
            }
            Err(failure) => {
                return Err(Box::new(access_error(
                    state,
                    failure,
                    &targets[index],
                    started_at,
                )));
            }
        }
    }
    let (_, pool) = cross_page_by(pool, Some(limit), after, targets);
    let tagged = pool
        .into_iter()
        .map(|(index, association)| (targets[index].clone(), association))
        .collect();
    Ok((total, tagged))
}

#[derive(Debug, Deserialize)]
pub struct CrossRecallRequest {
    /// Full context names — no patterns.
    #[serde(default)]
    pub contexts: Vec<String>,
    /// Group names — each adds every context it reaches, nested
    /// children included. Overlaps, with `contexts` or between groups,
    /// dedupe silently: a context is searched once however many ways
    /// it was named.
    #[serde(default)]
    pub groups: Vec<String>,
    pub cue: String,
    /// Omitted means 100.
    pub limit: Option<usize>,
    /// Resume past a previous page's last match — see
    /// [`CrossMatchCursor`].
    pub after: Option<CrossMatchCursor>,
}

/// [`recall`] across several named contexts at once, every match
/// tagged with the context it came from. `total` sums the per-context
/// match counts, and past the limit the strongest |weight| survives
/// exactly as within one context — weights share one scale (evidence
/// mass), so the cut means the same thing across contexts. Contexts
/// are searched concurrently (bounded by [`cross_search_concurrency`]);
/// the first per-context failure aborts the whole response (a read has
/// nothing to half-apply).
pub async fn cross_recall(
    State(state): State<AppState>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<CrossRecallRequest>,
) -> Response {
    let started_at = Instant::now();
    let targets = match cross_targets(
        &state,
        &scope,
        &key,
        request.contexts,
        request.groups,
        started_at,
    ) {
        Ok(targets) => targets,
        Err(refusal) => return *refusal,
    };
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Computed before `search` moves `cue` out of `request`.
    let cue_log = request.cue.clone();
    let outcome = cross_matches(
        &state,
        &targets,
        SearchOp::Recall,
        request.limit,
        request.after.as_ref(),
        move |context| context.recall(&request.cue),
        started_at,
    )
    .await;
    let (total, page) = match outcome {
        Ok(result) => result,
        Err(refusal) => return *refusal,
    };
    if search_log_enabled() {
        tracing::info!(
            target: "taguru::search",
            contexts = %targets.join(","),
            op = "recall",
            cue = %cue_log,
            hits = total,
            "search",
        );
    }
    let matches = cross_associations_out(&state, page);
    ok(CrossMatchPage { total, matches }, started_at)
}

#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    pub subject: Option<OneOrMany>,
    pub label: Option<OneOrMany>,
    pub object: Option<OneOrMany>,
    /// Omitted means 100.
    pub limit: Option<usize>,
    /// Resume past a previous page's last match — see [`MatchCursor`].
    pub after: Option<MatchCursor>,
}

pub async fn query(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<QueryRequest>,
) -> Response {
    let started_at = Instant::now();
    if let Some(refusal) = overlong_positions(
        &request.subject,
        &request.label,
        &request.object,
        started_at,
    ) {
        return refusal;
    }
    if let Some(refusal) = empty_positions(
        &request.subject,
        &request.label,
        &request.object,
        started_at,
    ) {
        return refusal;
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    match state.read_context(&name, |context| {
        context.query_any(
            &as_refs(&request.subject),
            &as_refs(&request.label),
            &as_refs(&request.object),
        )
    }) {
        Ok(result) => {
            let (total, matches) = page(result, request.limit, request.after.as_ref());
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
pub struct CrossQueryRequest {
    /// Full context names — no patterns.
    #[serde(default)]
    pub contexts: Vec<String>,
    /// Group names, resolved and deduped as in [`CrossRecallRequest`].
    #[serde(default)]
    pub groups: Vec<String>,
    pub subject: Option<OneOrMany>,
    pub label: Option<OneOrMany>,
    pub object: Option<OneOrMany>,
    /// Omitted means 100.
    pub limit: Option<usize>,
    /// Resume past a previous page's last match — see
    /// [`CrossMatchCursor`].
    pub after: Option<CrossMatchCursor>,
}

/// [`query`] across several named contexts at once — the same
/// cross-context contract as [`cross_recall`]: tagged matches, summed
/// `total`, strongest |weight| past the limit, first per-context
/// failure aborts.
pub async fn cross_query(
    State(state): State<AppState>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<CrossQueryRequest>,
) -> Response {
    let started_at = Instant::now();
    if let Some(refusal) = overlong_positions(
        &request.subject,
        &request.label,
        &request.object,
        started_at,
    ) {
        return refusal;
    }
    if let Some(refusal) = empty_positions(
        &request.subject,
        &request.label,
        &request.object,
        started_at,
    ) {
        return refusal;
    }
    let targets = match cross_targets(
        &state,
        &scope,
        &key,
        request.contexts,
        request.groups,
        started_at,
    ) {
        Ok(targets) => targets,
        Err(refusal) => return *refusal,
    };
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Computed before `search` moves `subject`/`label`/`object` out of
    // `request` — `OneOrMany` has no `Clone` to fall back on.
    let subject_log = as_refs(&request.subject).join(",");
    let label_log = as_refs(&request.label).join(",");
    let object_log = as_refs(&request.object).join(",");
    let outcome = cross_matches(
        &state,
        &targets,
        SearchOp::Query,
        request.limit,
        request.after.as_ref(),
        move |context| {
            context.query_any(
                &as_refs(&request.subject),
                &as_refs(&request.label),
                &as_refs(&request.object),
            )
        },
        started_at,
    )
    .await;
    let (total, page) = match outcome {
        Ok(result) => result,
        Err(refusal) => return *refusal,
    };
    if search_log_enabled() {
        tracing::info!(
            target: "taguru::search",
            contexts = %targets.join(","),
            op = "query",
            subject = %subject_log,
            label = %label_log,
            object = %object_log,
            hits = total,
            "search",
        );
    }
    let matches = cross_associations_out(&state, page);
    ok(CrossMatchPage { total, matches }, started_at)
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
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<DescribeRequest>,
) -> Response {
    let started_at = Instant::now();
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
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
    /// Resume past a previous page's last match — see [`ExploreCursor`].
    pub after: Option<ExploreCursor>,
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
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<ExploreRequest>,
) -> Response {
    let started_at = Instant::now();
    if let Some(refusal) = overlong("origins", request.origins.len(), started_at) {
        return refusal;
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
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
        Ok(matches) => {
            // Depth alone does not bound the response: one dense hub
            // can put a million edges within a single hop, and explore
            // used to return them all in one body.
            let (total, matches) = explore_page(matches, request.after.as_ref(), request.limit);
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
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<ActivateRequest>,
) -> Response {
    let started_at = Instant::now();
    if let Some(refusal) = overlong("origins", request.origins.len(), started_at) {
        return refusal;
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
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
fn merge_tiers(lexical: Vec<Resolution>, semantic: &[(String, f32)]) -> Vec<TieredResolution> {
    let mut merged = lexical_tier(lexical);
    for (name, score) in semantic {
        if merged.iter().any(|candidate| &candidate.name == name) {
            continue;
        }
        merged.push(TieredResolution {
            name: name.clone(),
            score: f64::from(*score),
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

/// The resolve pipeline's data half — the same tiers, floors, merge
/// inputs, and truncation the endpoint serves, shared verbatim with
/// the explain twin so the two cannot disagree about what would be
/// served. `bounded` is the lexical tier after the flood bound;
/// `overflow` is the tail that bound cut (empty in every non-
/// pathological call), kept so explain can locate a candidate past it.
struct ResolvedTiers {
    bounded: Vec<Resolution>,
    overflow: Vec<Resolution>,
    confident: bool,
    /// What the semantic tier contributed: empty when the lexical tier
    /// was confident, the provider is off, nothing is embedded — or
    /// when a provider failure degraded to lexical-only (logged).
    semantic: Vec<(String, f32)>,
}

/// Runs the entry ladder up to (but not through) the final merge:
/// lexical tiers first; the semantic tier joins whenever they came
/// back empty OR merely fragment-weak (best score under
/// [`LEXICAL_CONFIDENCE`]). `Err` is a response to serve immediately:
/// unknown context, or a provider failure with nothing lexical to
/// degrade to.
#[allow(clippy::result_large_err)] // the Err IS the response served next
fn resolve_tiers(
    state: &AppState,
    name: &str,
    request: &ResolveRequest,
    labels: bool,
    deadline: Deadline,
    started_at: Instant,
) -> Result<ResolvedTiers, Response> {
    let limit = clamp(request.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT);
    let mut bounded = match state.read_context(name, |context| match (labels, request.dice_floor) {
        (false, Some(floor)) => context.resolve_with_floor(&request.cue, floor),
        (false, None) => context.resolve(&request.cue),
        (true, Some(floor)) => context.resolve_label_with_floor(&request.cue, floor),
        (true, None) => context.resolve_label(&request.cue),
    }) {
        Ok(result) => result,
        Err(failure) => return Err(access_error(state, failure, name, started_at)),
    };
    // Resolutions arrive best-first, so bounding the flood right here
    // keeps the strongest candidates and spares everything downstream
    // (the semantic dedup scan, gloss reads, serialization) the
    // pathological tail. The confidence probe below reads only the
    // first entry, which the bound never touches.
    let overflow = bounded.split_off(bounded.len().min(limit));
    let confident = bounded
        .first()
        .is_some_and(|best| best.score >= LEXICAL_CONFIDENCE);
    let semantic = if confident {
        Vec::new()
    } else {
        // The provider round trip can take hundreds of milliseconds;
        // tell the runtime this thread will block so other tasks
        // migrate off it.
        match tokio::task::block_in_place(|| {
            state.semantic_resolve(name, &request.cue, labels, request.semantic_floor, deadline)
        }) {
            None => return Err(not_found(name, started_at)),
            Some(Ok(semantic)) => semantic,
            // The semantic tier is enrichment once ANY lexical
            // candidate exists: degrade to the weak lexical results
            // and log, rather than failing an answerable request over
            // a provider hiccup.
            Some(Err(message)) if !bounded.is_empty() => {
                tracing::warn!("semantic entry failed (serving weak lexical results): {message}");
                Vec::new()
            }
            Some(Err(message)) if deadline.expired() => {
                tracing::warn!(
                    "semantic entry failed with nothing lexical to fall back on: {message}"
                );
                return Err(error(
                    ErrorCode::Timeout,
                    format!("semantic entry failed: {message}"),
                    started_at,
                ));
            }
            Some(Err(message)) => {
                tracing::warn!(
                    "semantic entry failed with nothing lexical to fall back on: {message}"
                );
                return Err(error(
                    ErrorCode::EmbeddingsFailed,
                    format!("semantic entry failed: {message}"),
                    started_at,
                ));
            }
        }
    };
    Ok(ResolvedTiers {
        bounded,
        overflow,
        confident,
        semantic,
    })
}

/// The full entry ladder: [`resolve_tiers`], then the merge, the trim,
/// and the gloss decoration of what survived.
fn resolve_with_fallback(
    state: &AppState,
    name: &str,
    request: &ResolveRequest,
    labels: bool,
    deadline: Deadline,
    started_at: Instant,
) -> Response {
    // Both resolve handlers land here first, and the lexical read alone
    // takes the registry lock and sweeps the vocabulary before the
    // semantic tier's own deadline checks ever run — fail a spent
    // request fast, the way every other handler pre-flights.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let limit = clamp(request.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT);
    let tiers = match resolve_tiers(state, name, request, labels, deadline, started_at) {
        Ok(tiers) => tiers,
        Err(response) => return response,
    };
    let served = trim_to_limit(merge_tiers(tiers.bounded, &tiers.semantic), limit);
    let served = attach_glosses(state, name, labels, served);
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
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<ResolveRequest>,
) -> Response {
    let started_at = Instant::now();
    resolve_with_fallback(&state, &name, &request, false, deadline, started_at)
}

pub async fn resolve_label(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<ResolveRequest>,
) -> Response {
    let started_at = Instant::now();
    resolve_with_fallback(&state, &name, &request, true, deadline, started_at)
}

#[derive(Debug, Deserialize)]
pub struct ExplainResolveRequest {
    pub cue: String,
    /// The name the caller expected among the candidates.
    pub expected: String,
    /// The same one-call overrides resolve takes — explain answers for
    /// the exact call being questioned.
    pub dice_floor: Option<f64>,
    pub semantic_floor: Option<f32>,
    pub limit: Option<usize>,
}

/// How many nearest stored spellings a `not_in_vocabulary` verdict
/// attaches per tier — enough to spot the fork, not a listing.
const NEAREST_SPELLINGS: usize = 5;

/// One verdict for "why didn't (or did) this name come back for this
/// cue": machine-readable in `verdict`, human-readable in `summary`,
/// evidence attached. Every verdict is a 200 — a diagnosed miss is
/// this endpoint's success.
#[derive(Serialize)]
pub struct ResolveExplanation {
    /// `not_in_vocabulary` | `served` | `cue_resolved_exactly` |
    /// `below_floor` | `below_cutoff` | `semantic_not_run` |
    /// `semantic_below_floor`, first match wins in that order.
    pub verdict: &'static str,
    pub summary: String,
    pub cue: String,
    pub expected: String,
    pub in_vocabulary: bool,
    /// The canonical name `expected` maps to (differs when `expected`
    /// is an alias), and how it maps (`exact` / `alias`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected_kind: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lexical: Option<LexicalExplain>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic: Option<SemanticExplain>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranking: Option<ResolveRanking>,
    /// Nearest stored spellings, attached to `not_in_vocabulary` —
    /// the fix (register an alias) is one step away.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nearest: Option<NearestSpellings>,
}

/// The lexical tier's account: the Dice/coverage score the resolver
/// actually gave (cue → canonical) next to the floor in effect —
/// "scored 0.42, floor 0.6" answers "what floor would have shown it".
#[derive(Serialize)]
pub struct LexicalExplain {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<&'static str>,
    pub floor: f64,
    /// Whether the tier's best candidate was confident (≥ 0.5) — the
    /// predicate that decides if the semantic tier joins at all.
    pub confident: bool,
}

/// The semantic tier's account: whether it joined this call, why not
/// when it could not have, and the expected name's own gloss cosine
/// against the floor when the sweep could run.
#[derive(Serialize)]
pub struct SemanticExplain {
    pub entered: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub floor: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cosine: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    /// The tier serves its top `cap` only (a constant, not a request
    /// knob) — present whenever the sweep ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cap: Option<usize>,
}

/// Where the canonical stands against the served list: its rank when
/// present, and a `limit_to_reach` VERIFIED by rerunning the real
/// serve computation.
#[derive(Serialize)]
pub struct ResolveRanking {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tier: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    pub limit: usize,
    pub served: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit_to_reach: Option<usize>,
}

#[derive(Serialize)]
pub struct NearestSpellings {
    pub lexical: Vec<Resolution>,
    pub semantic: Vec<NearestGloss>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub semantic_note: Option<String>,
}

#[derive(Serialize)]
pub struct NearestGloss {
    pub name: String,
    pub cosine: f32,
}

/// What one `read_context` pass gathers for an explain call — the
/// membership probe, the floor in effect, and the cue's whole lexical
/// neighborhood, all under the same lock so they describe one moment.
struct VocabularyView {
    /// `expected` at floor 1.0: only normalized-exact spellings, so
    /// `None` means no entry, no alias, no canonical.
    canonical: Option<Resolution>,
    dice_floor: f64,
    /// The cue at floor 0.0: every lexically related name, best-first
    /// — where the canonical's actual score is read off, floor or no
    /// floor.
    sweep: Vec<Resolution>,
    /// `expected` at floor 0.0 (head only) — the nearest stored
    /// spellings when membership failed.
    nearest: Vec<Resolution>,
}

fn explain_resolve_verdict(
    state: &AppState,
    name: &str,
    request: &ExplainResolveRequest,
    labels: bool,
    deadline: Deadline,
    started_at: Instant,
) -> Response {
    // Both explain handlers land here first, and the view alone runs
    // several lexical sweeps inside read_context (membership, the cue
    // sweep, and possibly the nearest-spellings scan) before the
    // semantic tier's own deadline checks would ever run — fail a spent
    // request fast, the way resolve_with_fallback already does for its
    // own lexical read.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let view = match state.read_context(name, |context| {
        let resolve_floor = |cue: &str, floor: f64| {
            if labels {
                context.resolve_label_with_floor(cue, floor)
            } else {
                context.resolve_with_floor(cue, floor)
            }
        };
        // Membership means the expected string IS a stored spelling —
        // exact or alias, nothing looser. The floor cannot express
        // that (it gates only the fuzzy tier; containment sails past
        // any floor), so the kind is the test.
        let canonical = resolve_floor(&request.expected, 1.0)
            .into_iter()
            .find(|candidate| matches!(candidate.kind, MatchKind::Exact | MatchKind::Alias));
        let dice_floor = request.dice_floor.map_or_else(
            || context.dice_floor(),
            // NaN falls to the strictest floor, exactly as the scorer
            // treats it (`clamp_unit_or`'s nan_fallback).
            |floor| {
                if floor.is_nan() {
                    1.0
                } else {
                    floor.clamp(0.0, 1.0)
                }
            },
        );
        let sweep = resolve_floor(&request.cue, 0.0);
        let nearest = if canonical.is_none() {
            let mut nearest = resolve_floor(&request.expected, 0.0);
            nearest.truncate(NEAREST_SPELLINGS);
            nearest
        } else {
            Vec::new()
        };
        VocabularyView {
            canonical,
            dice_floor,
            sweep,
            nearest,
        }
    }) {
        Ok(view) => view,
        Err(failure) => return access_error(state, failure, name, started_at),
    };

    let namespace = if labels { "label" } else { "concept" };
    let Some(membership) = view.canonical else {
        // Verdict 1: not in the vocabulary at all. Attach the nearest
        // stored spellings, lexical and semantic alike, so the repair
        // (register an alias) needs no second round trip.
        let (glosses, semantic_note) = match tokio::task::block_in_place(|| {
            state.semantic_resolve(name, &request.expected, labels, Some(0.0), deadline)
        }) {
            None => return not_found(name, started_at),
            Some(Ok(matches)) => (matches, None),
            Some(Err(message)) => (Vec::new(), Some(message)),
        };
        let explanation = ResolveExplanation {
            verdict: "not_in_vocabulary",
            summary: format!(
                "'{}' is not in this context's {namespace} vocabulary — no entry, no \
                 alias, no canonical spelling; the nearest stored spellings are attached",
                request.expected
            ),
            cue: request.cue.clone(),
            expected: request.expected.clone(),
            in_vocabulary: false,
            canonical: None,
            expected_kind: None,
            lexical: None,
            semantic: None,
            ranking: None,
            nearest: Some(NearestSpellings {
                lexical: view.nearest,
                semantic: glosses
                    .into_iter()
                    .map(|(name, cosine)| NearestGloss { name, cosine })
                    .collect(),
                semantic_note,
            }),
        };
        state.note_read(name, true);
        return ok(explanation, started_at);
    };
    let canonical = membership.name.clone();
    let expected_kind = membership.kind.as_str();

    // The pipeline itself, with the request's own overrides — the
    // serve boundary below is exactly the resolve call being explained.
    let resolve_request = ResolveRequest {
        cue: request.cue.clone(),
        dice_floor: request.dice_floor,
        semantic_floor: request.semantic_floor,
        limit: request.limit,
    };
    let limit = clamp(resolve_request.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT);
    let tiers = match resolve_tiers(state, name, &resolve_request, labels, deadline, started_at) {
        Ok(tiers) => tiers,
        Err(response) => return response,
    };
    let served = trim_to_limit(merge_tiers(tiers.bounded.clone(), &tiers.semantic), limit);
    let full: Vec<Resolution> = tiers
        .bounded
        .iter()
        .chain(tiers.overflow.iter())
        .cloned()
        .collect();
    let merged = merge_tiers(full.clone(), &tiers.semantic);

    let served_at = served
        .iter()
        .position(|candidate| candidate.name == canonical);
    let merged_at = merged
        .iter()
        .position(|candidate| candidate.name == canonical);
    // Whether the effective-floor resolve admitted the canonical at
    // all — the floor gates ONLY the fuzzy tier (exact and containment
    // pass it by construction), so "scored under the floor" is
    // precisely "visible at floor 0, absent here".
    let admitted = full.iter().any(|candidate| candidate.name == canonical);
    let target = view
        .sweep
        .iter()
        .find(|candidate| candidate.name == canonical);
    // The exact tier answers alone when the cue IS a stored spelling —
    // nothing else is ever scored, however close. An all-exact sweep
    // is that early return's signature.
    let exact_shortcut = target.is_none()
        && !view.sweep.is_empty()
        && view
            .sweep
            .iter()
            .all(|candidate| matches!(candidate.kind, MatchKind::Exact | MatchKind::Alias));

    // The gloss lane's account — the cue embedding is already cached
    // when the pipeline entered the tier, so this adds no provider
    // round trip beyond the one targeted scoring explain budgets for.
    let lane = match tokio::task::block_in_place(|| {
        state.explain_semantic_resolve(
            name,
            &request.cue,
            &canonical,
            labels,
            request.semantic_floor,
            deadline,
        )
    }) {
        Some(lane) => lane,
        None => return not_found(name, started_at),
    };
    let entered = !tiers.confident;
    let semantic = {
        use crate::registry::GlossLaneReport;
        let mut explain = SemanticExplain {
            entered,
            reason: None,
            floor: None,
            cosine: None,
            rank: None,
            cap: None,
        };
        match &lane {
            GlossLaneReport::Off => {
                explain.reason = Some("no embedding provider is configured".to_string());
            }
            GlossLaneReport::ModelChanged { stored, current } => {
                explain.reason = Some(format!(
                    "gloss vectors belong to model '{stored}' but the provider is \
                     '{current}' — awaiting re-embed"
                ));
            }
            GlossLaneReport::EmptyTable => {
                explain.reason = Some(format!(
                    "no {namespace} gloss vectors exist yet — no embedding refresh has run"
                ));
            }
            GlossLaneReport::QueryEmbeddingFailed(error) => {
                explain.reason = Some(format!("the cue embedding failed: {error}"));
            }
            GlossLaneReport::Ran {
                floor,
                cosine,
                rank,
                cap,
                ..
            } => {
                explain.floor = Some(*floor);
                explain.cosine = *cosine;
                explain.rank = *rank;
                explain.cap = Some(*cap);
                if cosine.is_none() {
                    explain.reason = Some(format!(
                        "'{canonical}' has no gloss vector yet — added after the last refresh"
                    ));
                }
            }
        }
        if !entered && explain.reason.is_none() {
            explain.reason = Some(format!(
                "the lexical tier was confident (best score ≥ {LEXICAL_CONFIDENCE}), \
                 so the semantic tier never joined"
            ));
        }
        explain
    };

    // The smallest VERIFIED limit that serves the canonical: rerun the
    // real serve computation (bound, merge, trim), growing on a miss —
    // the trim takes overflow out of the lexical tail first, so the
    // unbounded rank alone is a floor, not an answer.
    let serve_at = |limit: usize| -> Vec<TieredResolution> {
        let mut bounded = full.clone();
        bounded.truncate(limit);
        trim_to_limit(merge_tiers(bounded, &tiers.semantic), limit)
    };
    let limit_to_reach = if served_at.is_some() {
        Some(limit)
    } else {
        merged_at.map(|at| at + 1).and_then(|first| {
            let mut candidate = first;
            for _ in 0..8 {
                if candidate > merged.len() {
                    return None;
                }
                if serve_at(candidate)
                    .iter()
                    .any(|served| served.name == canonical)
                {
                    return Some(candidate);
                }
                candidate = candidate.saturating_mul(2).min(merged.len());
            }
            None
        })
    };

    let target_score = target.map(|candidate| candidate.score);
    let target_kind = target.map(|candidate| candidate.kind.as_str());
    use crate::registry::GlossLaneReport;
    let (verdict, summary) = if let Some(at) = served_at {
        (
            "served",
            format!(
                "served: '{canonical}' ranked {} of {} at limit {limit} ({} tier)",
                at + 1,
                served.len(),
                served[at].tier
            ),
        )
    } else if exact_shortcut {
        (
            "cue_resolved_exactly",
            format!(
                "the cue is an exact spelling of '{}' — the exact tier answers alone, \
                 so '{canonical}' was never scored against it; a reworded cue is the \
                 way in",
                view.sweep[0].name
            ),
        )
    } else if let Some(score) = target_score.filter(|_| !admitted) {
        (
            "below_floor",
            format!(
                "the resolver scored '{canonical}' {score:.4} ({}) against dice_floor \
                 {:.4} — a floor at or under {score:.4} would have shown it",
                target_kind.unwrap_or("fuzzy"),
                view.dice_floor
            ),
        )
    } else if target_score.is_some() {
        // At or above the floor yet not served: lost on the limit.
        let reach = match limit_to_reach {
            Some(reach) => format!("limit {reach} reaches it"),
            None => "no limit reaches it".to_string(),
        };
        (
            "below_cutoff",
            format!(
                "'{canonical}' passed the floor but ranked {} of {} — the request \
                 served {limit}; {reach}",
                merged_at.map_or(0, |at| at + 1),
                merged.len()
            ),
        )
    } else if !entered {
        (
            "semantic_not_run",
            format!(
                "'{canonical}' shares no spelling with the cue, and the semantic tier \
                 never joined: the lexical tier was confident (best score ≥ \
                 {LEXICAL_CONFIDENCE})"
            ),
        )
    } else {
        match &lane {
            GlossLaneReport::Ran {
                floor,
                cosine: Some(cosine),
                rank,
                passing,
                cap,
            } => {
                if cosine < floor {
                    (
                        "semantic_below_floor",
                        format!(
                            "'{canonical}' shares no spelling with the cue; its gloss \
                             cosine {cosine:.4} sits under the semantic floor {floor:.4} \
                             — a floor at or under {cosine:.4} would have shown it"
                        ),
                    )
                } else if rank.is_some_and(|rank| rank > *cap) {
                    (
                        "below_cutoff",
                        format!(
                            "'{canonical}' cleared the semantic floor (cosine \
                             {cosine:.4}) but ranked {} of {passing} — the semantic \
                             tier serves its top {cap} only",
                            rank.unwrap_or_default()
                        ),
                    )
                } else {
                    // Cleared the floor, within the cap, yet missing:
                    // the request's own limit trimmed it out.
                    let reach = match limit_to_reach {
                        Some(reach) => format!("limit {reach} reaches it"),
                        None => "no limit reaches it".to_string(),
                    };
                    (
                        "below_cutoff",
                        format!(
                            "'{canonical}' earned a semantic seat (cosine {cosine:.4}) \
                             but the request served only {limit}; {reach}"
                        ),
                    )
                }
            }
            _ => (
                "semantic_not_run",
                format!(
                    "'{canonical}' shares no spelling with the cue, and the semantic \
                     tier could not score it: {}",
                    semantic.reason.as_deref().unwrap_or("it did not run")
                ),
            ),
        }
    };

    let explanation = ResolveExplanation {
        verdict,
        summary,
        cue: request.cue.clone(),
        expected: request.expected.clone(),
        in_vocabulary: true,
        canonical: Some(canonical.clone()),
        expected_kind: Some(expected_kind),
        lexical: Some(LexicalExplain {
            score: target_score,
            kind: target_kind,
            floor: view.dice_floor,
            confident: tiers.confident,
        }),
        semantic: Some(semantic),
        ranking: Some(ResolveRanking {
            rank: merged_at.map(|at| at + 1),
            tier: merged_at.map(|at| merged[at].tier),
            score: merged_at.map(|at| merged[at].score),
            limit,
            served: served_at.is_some(),
            limit_to_reach,
        }),
        nearest: None,
    };
    state.note_read(name, false);
    ok(explanation, started_at)
}

/// `POST /contexts/{name}/resolve/explain` — "why didn't this concept
/// come back for this cue" in one call: the same tiers, floors, and
/// trims the resolve endpoint runs, with the expected name located in
/// (or placed against) each of them. Read-only.
pub async fn explain_resolve(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<ExplainResolveRequest>,
) -> Response {
    let started_at = Instant::now();
    explain_resolve_verdict(&state, &name, &request, false, deadline, started_at)
}

/// `POST /contexts/{name}/resolve_label/explain` — explain_resolve,
/// for relation labels.
pub async fn explain_resolve_label(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<ExplainResolveRequest>,
) -> Response {
    let started_at = Instant::now();
    explain_resolve_verdict(&state, &name, &request, true, deadline, started_at)
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
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    if !state.embeddings_configured() {
        return error(
            ErrorCode::EmbeddingsUnconfigured,
            "no embedding provider is configured (set TAGURU_EMBED_URL and TAGURU_EMBED_MODEL)",
            started_at,
        );
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Refresh batches can talk to the provider for seconds; keep the
    // runtime's workers unstarved while this one blocks.
    let glosses = match tokio::task::block_in_place(|| state.refresh_embeddings(&name, deadline)) {
        None => return not_found(&name, started_at),
        Some(Ok(counts)) => counts,
        Some(Err(message)) if deadline.expired() => {
            return error(
                ErrorCode::Timeout,
                format!("embedding refresh failed: {message}"),
                started_at,
            );
        }
        Some(Err(message)) => {
            return error(
                ErrorCode::EmbeddingsFailed,
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
    match tokio::task::block_in_place(|| state.refresh_passage_embeddings(&name, deadline)) {
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
        Some(Err(message)) if deadline.expired() => error(
            ErrorCode::Timeout,
            format!("passage embedding refresh failed partway (progress is saved): {message}"),
            started_at,
        ),
        Some(Err(message)) => error(
            ErrorCode::EmbeddingsFailed,
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
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppQuery(query): AppQuery<KeysetQuery>,
) -> Response {
    let started_at = Instant::now();
    let limit = clamp(query.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT);
    // A `prefix` filter forces the whole-vocabulary scan below; a bare
    // cursor stays on the cheap BTreeMap-seeking path regardless.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    match state.read_context(&name, |context| {
        // A `prefix` filter defines the population rather than a
        // cursor, so — like `pinned` on `list_contexts` — it forces the
        // whole-vocabulary path instead of the BTreeMap-seeking
        // `label_page` fast path, which has no way to know in advance
        // how many prefix-matching labels lie within any given range.
        let (total, labels) = match query.prefix.as_deref() {
            Some(prefix) => {
                let mut labels: Vec<String> =
                    context.labels().into_iter().map(String::from).collect();
                labels.sort();
                labels.retain(|label| label.starts_with(prefix));
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
                (total, labels)
            }
            None => context.label_page(query.after.as_deref(), limit),
        };
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
    /// Resume past a previous page's last match — see [`MatchCursor`].
    pub after: Option<MatchCursor>,
}

pub async fn unreachable_from(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<UnreachableFromRequest>,
) -> Response {
    let started_at = Instant::now();
    if let Some(refusal) = overlong("origins", request.origins.len(), started_at) {
        return refusal;
    }
    // Always walks every edge in the context (see Context::unreachable_from)
    // — the same unconditional-full-scan cost as audit_drift's
    // unsourced_edges, which already pre-flights this.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    match state.read_context(&name, |context| {
        let origins: Vec<&str> = request.origins.iter().map(String::as_str).collect();
        context.unreachable_from(&origins)
    }) {
        Ok(result) => {
            let (total, matches) = page(result, request.limit, request.after.as_ref());
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

    /// A percent-encoded path segment that decodes to invalid UTF-8 is
    /// the one client-reachable `Path` rejection (`%ff` → the lone byte
    /// 0xFF). `AppPath` must answer it in the shared [`ApiError`] JSON
    /// shape — a 400 carrying `status: "error"` — not axum's bare
    /// `text/plain` "Invalid URL", which was the last off-shape axis.
    #[tokio::test]
    async fn apppath_reshapes_an_invalid_utf8_segment_into_the_api_error_body() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use axum::routing::get;
        use tower::util::ServiceExt;

        async fn echo(AppPath(name): AppPath<String>) -> String {
            name
        }
        let app = Router::new().route("/x/{name}", get(echo));

        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/x/%ff")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            response.headers()[axum::http::header::CONTENT_TYPE],
            "application/json"
        );
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["status"], "error");
        assert_eq!(body["code"], ErrorCode::MalformedRequest.as_str());
        assert!(
            body["error"].is_string() && body["time"].is_number(),
            "{body}"
        );
    }

    /// The happy path is untouched: a well-formed segment still reaches
    /// the handler decoded, so wrapping every route in `AppPath` only
    /// changes the error axis, never a valid request.
    #[tokio::test]
    async fn apppath_passes_a_valid_segment_through_decoded() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use axum::routing::get;
        use tower::util::ServiceExt;

        async fn echo(AppPath(name): AppPath<String>) -> String {
            name
        }
        let app = Router::new().route("/x/{name}", get(echo));

        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/x/hello%20world")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), "hello world");
    }

    /// `CatchPanicLayer::custom(panic_response)` is the exact wiring
    /// `routes()` uses in `main.rs` — exercised together, not
    /// `panic_response` in isolation, since the bug this guards
    /// against is the request vanishing from the metrics/access-log/
    /// trace middleware wrapping the router, not the response shape
    /// on its own.
    #[tokio::test]
    async fn a_panicking_handler_still_answers_with_the_shared_api_error_shape() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use axum::routing::get;
        use tower::util::ServiceExt;
        use tower_http::catch_panic::CatchPanicLayer;

        async fn boom() -> &'static str {
            panic!("kaboom")
        }
        let app = Router::new()
            .route("/boom", get(boom))
            .layer(CatchPanicLayer::custom(panic_response));

        let response = app
            .oneshot(
                HttpRequest::builder()
                    .uri("/boom")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            response.headers()[axum::http::header::CONTENT_TYPE],
            "application/json"
        );
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["status"], "error");
        assert_eq!(body["code"], ErrorCode::Internal.as_str());
        assert!(body["error"].as_str().unwrap().contains("kaboom"), "{body}");
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
        let (total, matches) = page(matches, Some(1_000_000_000), None);
        assert_eq!(total, MAX_MATCH_LIMIT + 5);
        assert_eq!(matches.len(), MAX_MATCH_LIMIT);
    }

    #[test]
    fn page_orders_by_magnitude_then_lexicographic_tiebreak_never_insertion_order() {
        // Equal |weight|: (subject, label, object) breaks the tie, not
        // the order matches arrived in — insertion here is c, a, b.
        let matches = vec![assoc("c", 2.0), assoc("a", 2.0), assoc("b", 2.0)];
        let (total, ranked) = page(matches, Some(3), None);
        assert_eq!(total, 3);
        let objects: Vec<&str> = ranked.iter().map(|m| m.object.as_str()).collect();
        assert_eq!(objects, vec!["a", "b", "c"]);

        // Past the limit, magnitude ranks first — the negative fact is
        // the strongest knowledge — and total still reports the
        // untruncated count.
        let matches = vec![assoc("a", 1.0), assoc("b", -3.0), assoc("c", 2.0)];
        let (total, truncated) = page(matches, Some(2), None);
        assert_eq!(total, 3);
        let objects: Vec<&str> = truncated.iter().map(|m| m.object.as_str()).collect();
        assert_eq!(objects, vec!["b", "c"]);
    }

    #[test]
    fn page_resumes_past_a_cursor_and_keeps_total_constant() {
        let matches = vec![
            assoc("a", 3.0),
            assoc("b", 2.0),
            assoc("c", 1.0),
            assoc("d", 1.0),
        ];
        let (total, first) = page(matches.clone(), Some(2), None);
        assert_eq!(total, 4);
        let objects: Vec<&str> = first.iter().map(|m| m.object.as_str()).collect();
        assert_eq!(objects, vec!["a", "b"]);

        let last = first.last().unwrap();
        let cursor = MatchCursor {
            weight: last.weight,
            subject: last.subject.clone(),
            label: last.label.clone(),
            object: last.object.clone(),
        };
        let (total, second) = page(matches, Some(2), Some(&cursor));
        assert_eq!(total, 4, "total is the pre-cursor, pre-truncate count");
        let objects: Vec<&str> = second.iter().map(|m| m.object.as_str()).collect();
        assert_eq!(objects, vec!["c", "d"]);
    }

    fn recollection(object: &str, distance: usize) -> Recollection {
        Recollection {
            distance,
            path: vec!["origin".to_string()],
            association: assoc(object, 1.0),
        }
    }

    #[test]
    fn explore_page_resorts_same_distance_ties_lexicographically_not_by_arrival() {
        // `Context::explore` orders same-distance ties by internal
        // edge_id, which tracks arrival order (c, a, b here) — a
        // retain-only cut would keep this exact wrong order.
        let matches = vec![
            recollection("c", 1),
            recollection("a", 1),
            recollection("b", 1),
        ];
        let (total, ranked) = explore_page(matches, None, Some(3));
        assert_eq!(total, 3);
        let objects: Vec<&str> = ranked
            .iter()
            .map(|r| r.association.object.as_str())
            .collect();
        assert_eq!(objects, vec!["a", "b", "c"]);
    }

    #[test]
    fn explore_page_resumes_past_a_cursor_and_keeps_total_constant() {
        let matches = vec![
            recollection("a", 1),
            recollection("b", 1),
            recollection("z", 2),
        ];
        let (total, first) = explore_page(matches.clone(), None, Some(2));
        assert_eq!(total, 3);
        let objects: Vec<&str> = first
            .iter()
            .map(|r| r.association.object.as_str())
            .collect();
        assert_eq!(objects, vec!["a", "b"]);

        let last = first.last().unwrap();
        let cursor = ExploreCursor {
            distance: last.distance,
            subject: last.association.subject.clone(),
            label: last.association.label.clone(),
            object: last.association.object.clone(),
        };
        let (total, second) = explore_page(matches, Some(&cursor), Some(2));
        assert_eq!(total, 3, "total is the pre-cursor, pre-truncate count");
        let objects: Vec<&str> = second
            .iter()
            .map(|r| r.association.object.as_str())
            .collect();
        assert_eq!(objects, vec!["z"]);
    }

    #[test]
    fn cross_page_by_breaks_a_same_triple_tie_on_context() {
        // Two different target contexts each hold an edge with the
        // identical (subject, label, object) and weight — (subject,
        // label, object) alone cannot order them, so `context` must.
        let targets = vec!["zeta".to_string(), "alpha".to_string()];
        let matches = vec![(0, assoc("青嶺", 1.0)), (1, assoc("青嶺", 1.0))];
        let (total, ranked) = cross_page_by(matches, Some(2), None, &targets);
        assert_eq!(total, 2);
        let contexts: Vec<&str> = ranked.iter().map(|(i, _)| targets[*i].as_str()).collect();
        assert_eq!(
            contexts,
            vec!["alpha", "zeta"],
            "context breaks the tie lexicographically, not target-list order"
        );
    }

    #[test]
    fn cross_page_by_resumes_past_a_cursor_and_keeps_total_constant() {
        let targets = vec!["alpha".to_string(), "zeta".to_string()];
        let matches = vec![(0, assoc("青嶺", 1.0)), (1, assoc("青嶺", 1.0))];
        let (total, first) = cross_page_by(matches.clone(), Some(1), None, &targets);
        assert_eq!(total, 2);
        assert_eq!(targets[first[0].0], "alpha");

        let last = &first[0].1;
        let cursor = CrossMatchCursor {
            weight: last.weight,
            context: targets[first[0].0].clone(),
            subject: last.subject.clone(),
            label: last.label.clone(),
            object: last.object.clone(),
        };
        let (total, second) = cross_page_by(matches, Some(1), Some(&cursor), &targets);
        assert_eq!(total, 2, "total is the pre-cursor, pre-truncate count");
        assert_eq!(second.len(), 1);
        assert_eq!(targets[second[0].0], "zeta");
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
        let merged = merge_tiers(lexical, &semantic);
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

    #[tokio::test]
    async fn restore_refusal_frames_a_spent_budget_as_a_resumable_timeout() {
        use crate::registry::RestoreGroupsError;

        // A group restore that runs out of budget is a resumable prefix, not
        // a rejected set: it answers with the Timeout code and a message
        // naming the durable batch count and the timeout knob — distinct from
        // the generic "every batch landed" refusal the validation arms emit.
        // Guards the Timeout arm of the message match against deletion, which
        // would fall the Timeout case through to that generic wording.
        let dir =
            std::env::temp_dir().join(format!("taguru-api-restore-refusal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();

        let response = restore_refusal(
            &state,
            RestoreGroupsError::Timeout { applied: 2 },
            2,
            Instant::now(),
        );
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["code"], ErrorCode::Timeout.as_str());
        let message = body["error"].as_str().expect("error is a string");
        assert!(
            message.contains("group restore exceeded its budget with 2 batch(es) durable"),
            "{message}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `include_twins` is the only expensive branch `audit_drift` runs
    /// — the unsourced/alias sweeps ahead of it are a cheap O(n) pass.
    /// The heavy-ops permit must therefore only be spent on that
    /// branch: a plain drift audit succeeds even with the limiter's
    /// sole permit held elsewhere, and an `include_twins: true` audit
    /// sheds once that permit is unavailable rather than running the
    /// pairwise scan unpermitted.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn audit_drift_only_spends_a_heavy_ops_permit_for_include_twins() {
        use crate::registry::ContextMeta;
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use axum::routing::post;
        use tower::util::ServiceExt;

        let dir = std::env::temp_dir().join(format!(
            "taguru-api-audit-drift-permit-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();

        let limiter = HeavyOpsLimiter::new(1);
        let held = limiter.try_acquire().unwrap();

        let app = Router::new()
            .route("/contexts/{name}/drift/audit", post(audit_drift))
            .layer(axum::Extension(limiter))
            .layer(axum::Extension(Deadline::unbounded()))
            .with_state(state);

        let request = |body: &'static str| {
            HttpRequest::builder()
                .method("POST")
                .uri("/contexts/sake/drift/audit")
                .body(Body::from(body))
                .unwrap()
        };

        let cheap = app.clone().oneshot(request("{}")).await.unwrap();
        assert_eq!(
            cheap.status(),
            StatusCode::OK,
            "the sole permit was never touched"
        );

        let heavy = app
            .clone()
            .oneshot(request(r#"{"include_twins": true}"#))
            .await
            .unwrap();
        assert_eq!(
            heavy.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "include_twins needs the permit the cheap audit above left untouched"
        );

        drop(held);
        let recovered = app
            .oneshot(request(r#"{"include_twins": true}"#))
            .await
            .unwrap();
        assert_eq!(recovered.status(), StatusCode::OK);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
