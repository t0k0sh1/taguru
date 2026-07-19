//! HTTP surface of the retrieval service: thin JSON adapters, one per
//! `Context` operation. The server adds transport, naming, and lifecycle
//! around the library — never retrieval semantics of its own — so each
//! handler is a lock, a library call, and a serialized reply.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request};
use axum::http::{Method, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use taguru::context::{Activation, Association, Attribution, Recollection};

use crate::groups::{MAX_GROUP_DEPTH, MAX_GROUP_MEMBERS, NestingViolation};
use crate::metrics::ErrorKind;
use crate::registry::{AccessError, AppState, CachedRetrieval, PartialWrite, RetrievalKey};

mod aliases;
mod associations;
mod contexts;
mod coverage;
mod explore;
mod groups;
mod import;
mod recall;
mod resolve;
mod sources;
mod vocabulary;

pub use aliases::{add_aliases, list_aliases, remove_aliases};
pub use associations::{add_associations, retract_association};
pub use contexts::{
    create_context, delete_context, flush_all, get_context, list_contexts, maintenance_compact,
    protocol, protocol_text, protocol_trailer, rename_context, update_context,
};
pub use coverage::{labels, refresh_embeddings, unreachable_from};
pub use explore::{activate, describe, explore};
pub use groups::{create_group, delete_group, get_group, list_groups, rename_group, update_group};
pub use import::{compact_context, export_context, export_group, import_batch};
pub use recall::{cross_query, cross_recall, query, recall};
pub use resolve::{explain_resolve, explain_resolve_label, resolve, resolve_label};
pub use sources::{
    citation, cross_search_passages, explain_search_passages, list_sources, lookup_passages,
    retract_source, search_passages, store_passages,
};
pub use vocabulary::{audit_drift, audit_vocabulary};

// The router mode's imports: the wire shapes it re-serializes when
// merging shard pages, the exact merge comparator, and the query
// shapes it pre-parses (`crate::route`).
pub(crate) use aliases::KeysetQuery;
pub(crate) use contexts::{ContextPage, ListContextsQuery};
pub(crate) use groups::{GroupEntry, GroupPage};
pub(crate) use import::ImportQuery;
pub(crate) use recall::{CrossQueryRequest, CrossRecallRequest, cross_rank};
pub(crate) use sources::{CrossSearchPassagesRequest, PassageHit};

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
    /// 403 from a read replica for any mutating verb: writes go to the
    /// writer, which the message names as precisely as the replica can
    /// (`TAGURU_WRITER_URL`, plus the bucket's fence holder). A 4xx on
    /// purpose — a deliberate refusal that retrying cannot change, so
    /// no SDK or client retry loop ever forms around it.
    ReadOnlyReplica,
    /// 502 from `taguru route` when a shard the request needs cannot
    /// be REACHED (connect, timeout, torn body) — distinct from a
    /// shard that answered an error, which passes through with the
    /// shard's own code. Retryable once the shard (or its LB) answers.
    ShardUnreachable,
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
            Self::ReadOnlyReplica => "read_only_replica",
            Self::ShardUnreachable => "shard_unreachable",
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
            Self::Forbidden | Self::ReadOnlyReplica => StatusCode::FORBIDDEN,
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
            Self::EmbeddingsFailed | Self::ShardUnreachable => StatusCode::BAD_GATEWAY,
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

pub(crate) fn ok<T: Serialize>(result: T, started_at: Instant) -> Response {
    (StatusCode::OK, Json(ApiResponse::ok(result, started_at))).into_response()
}

/// The one constructor every JSON error response goes through, status
/// derived from the code. The extractor reshapes ([`AppJson`],
/// [`AppQuery`], [`AppBytes`]) call [`coded`] instead, keeping axum's
/// own rejection status.
pub(crate) fn error(code: ErrorCode, message: impl Into<String>, started_at: Instant) -> Response {
    coded(code.status(), code, message, started_at)
}

/// Refuses every mutating verb on a read replica (issue #129), crisply
/// and with the writer named — a no-op layer on a writer. Layered
/// inside `routes()` on purpose: the `/mcp` in-process dispatch clones
/// that router, so a write TOOL call is judged here exactly like the
/// raw HTTP verb it dispatches into, while `/mcp` itself (Role::Read —
/// the transport, not a mutation) passes and its read tools work
/// unchanged.
///
/// What passes: GET/HEAD (this API keeps them side-effect-free — the
/// probes, the listings, the exports, OAuth discovery), plus every
/// route [`crate::auth::required_role`] classifies `Read` (the
/// retrieval POSTs). Everything else — the ingest loop, the operator
/// verbs, OAuth's grant endpoints, and any FUTURE unclassified POST —
/// refuses, which is the same deny-by-default posture the role table
/// itself keeps. Requests no route matched fall through to the
/// ordinary 404/405 shapes: "unknown path" is a more useful answer
/// than "read only".
pub(crate) async fn replica_gate(
    axum::extract::State(state): axum::extract::State<crate::registry::AppState>,
    matched: Option<axum::extract::MatchedPath>,
    request: Request,
    next: axum::middleware::Next,
) -> Response {
    let Some(replica) = state.replica() else {
        return next.run(request).await;
    };
    let Some(matched) = matched else {
        return next.run(request).await;
    };
    let method = request.method();
    if method == Method::GET
        || method == Method::HEAD
        || crate::auth::required_role(method, matched.as_str()) == crate::auth::Role::Read
    {
        return next.run(request).await;
    }
    error(
        ErrorCode::ReadOnlyReplica,
        replica.refusal(),
        Instant::now(),
    )
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
///
/// Takes `state` to record `ErrorKind::Panic` here too: `record_error`
/// is how every other 500 cause reaches `taguru_errors_total`, and a
/// panic is as legitimate a cause as `Load` or `Io` — omitting it would
/// leave the one 500 cause that is always a bug invisible in the metric
/// meant to surface exactly that. This also covers panics from a tool
/// call dispatched through `/mcp`: `routes()` layers this same handler
/// as its innermost layer specifically so a dispatched call's panic is
/// caught here (see that function's doc comment), even though the
/// resulting 500 never becomes the outer HTTP response — the JSON-RPC
/// envelope answers 200 either way, so `taguru_errors_total` is the
/// only signal that call ever left behind.
pub(crate) fn panic_response(payload: Box<dyn std::any::Any + Send>, state: &AppState) -> Response {
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
    state.metrics().record_error(ErrorKind::Panic);
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

/// Pulls a named path parameter out of request parts by hand —
/// `RawPathParams` rejects param-less routes, so it cannot ride a
/// handler's signature as an ordinary extractor the way `MatchedPath`
/// (which supports optional extraction) does. Shared by the auth
/// middleware's per-context grant check and the access-log middleware's
/// logged name, both of which run ahead of routing proper and so must
/// extract by hand from the parts they hold.
pub(crate) async fn path_param(
    parts: &mut axum::http::request::Parts,
    name: &str,
) -> Option<String> {
    use axum::extract::FromRequestParts as _;
    axum::extract::RawPathParams::from_request_parts(parts, &())
        .await
        .ok()
        .and_then(|params| {
            params
                .iter()
                .find(|(param, _)| *param == name)
                .map(|(_, value)| value.to_string())
        })
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
/// log system, which is why there is no in-server top-K. A line
/// served from the retrieval cache carries an extra `cached=true`
/// field; the ask is just as real, the search path just didn't run.
fn search_log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| crate::env::env_bool("TAGURU_LOG_SEARCHES", false))
}

/// Replays a retrieval-cache hit's observable side effects: the same
/// `note_search` per target (usage counters and the searches family)
/// and the same passage-lane contributions the fresh path recorded
/// when it computed the entry. Metrics describe served responses; the
/// cache only changes how a response was computed, so a hit must not
/// read as a quiet server. The hit/miss split itself lives in its own
/// family (`taguru_retrieval_cache_total`), recorded at lookup.
fn replay_cached_search(state: &AppState, key: &RetrievalKey, found: &CachedRetrieval) {
    let op = key.op.search_op();
    for (target, empty) in key.targets.iter().zip(found.target_empty.iter()) {
        state.note_search(op, &target.name, *empty);
    }
    let [bm25_only, both_lanes, vector_only] = found.lane_hits;
    if bm25_only > 0 || both_lanes > 0 || vector_only > 0 {
        state
            .metrics()
            .record_passage_hit_counts(bm25_only, both_lanes, vector_only);
    }
}

/// The retrieval fill path's tail: serialize the payload exactly once,
/// file those bytes under `key` when one was minted, and serve them —
/// so a later hit is byte-identical to the fill that produced it
/// (round-tripping through `serde_json::Value` instead would widen
/// `f32` scores to `f64` and change the wire text). `target_empty`,
/// `lane_hits`, and the two log numbers are what a hit replays; see
/// [`replay_cached_search`].
#[allow(clippy::too_many_arguments)] // one call-shape per cached surface, not an API
fn cache_and_serve<T: Serialize>(
    state: &AppState,
    key: Option<RetrievalKey>,
    payload: &T,
    target_empty: Vec<bool>,
    lane_hits: [u64; 3],
    log_hits: usize,
    log_top_score: f32,
    started_at: Instant,
) -> Response {
    let Ok(raw) = serde_json::value::to_raw_value(payload) else {
        // Unreachable for these payload types (no non-string map keys,
        // no non-finite floats) — serve uncached rather than fail a
        // response the search already computed.
        return ok(payload, started_at);
    };
    let raw: Arc<serde_json::value::RawValue> = raw.into();
    if let Some(key) = key {
        state.retrieval_store(
            key,
            CachedRetrieval {
                payload: Arc::clone(&raw),
                target_empty: target_empty.into(),
                lane_hits,
                log_hits,
                log_top_score,
            },
        );
    }
    ok(raw.as_ref(), started_at)
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

/// The capacity/conflict split every batch write reports on a partial
/// failure, plus the write-counter bump for whatever landed before it —
/// shared by `add_associations`, `add_aliases`, and `remove_aliases`.
/// `describe` renders the applied count and the library's own message
/// into each caller's phrasing.
fn partial_write_error(
    state: &AppState,
    name: &str,
    partial: PartialWrite,
    started_at: Instant,
    describe: impl FnOnce(usize, String) -> String,
) -> Response {
    if partial.applied > 0 {
        state.note_write(name);
    }
    let code = if partial.full {
        ErrorCode::StorageFull
    } else {
        ErrorCode::Conflict
    };
    error(code, describe(partial.applied, partial.message), started_at)
}

/// Cap applied to recall/query matches when the request names no limit:
/// a hub concept must not flood an LLM client's prompt by default.
pub(crate) const DEFAULT_MATCH_LIMIT: usize = 100;

/// The hard ceiling every per-request result limit is clamped to,
/// whatever the request asks for — result sizing is a server
/// protection, not a client entitlement.
pub(crate) const MAX_MATCH_LIMIT: usize = 1000;

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
pub(crate) fn overlong(field: &str, len: usize, started_at: Instant) -> Option<Response> {
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
pub(crate) fn clamp(value: Option<usize>, default: usize, ceiling: usize) -> usize {
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
#[derive(Serialize, Deserialize)]
pub struct CrossMatch<T> {
    pub context: String,
    #[serde(flatten)]
    pub inner: T,
}

/// [`MatchPage`], cross-context: same `total`-above-count truncation
/// contract, every match tagged.
#[derive(Serialize, Deserialize)]
pub struct CrossMatchPage {
    pub total: usize,
    pub matches: Vec<CrossMatch<AssociationOut>>,
}

/// A match page's resume point: the rank key of the last item on the
/// previous page. `(subject, label, object)` alone already uniquely
/// identifies an edge within one context (the `edge_ids` bijection),
/// so `weight` plus that triple is a total order with no possible tie
/// — a client can always build the next `after` from the last match it
/// received. (`Serialize` is for the retrieval cache's key, where the
/// cursor is a result-affecting parameter like any other.)
#[derive(Debug, Clone, Serialize, Deserialize)]
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
        matches.retain(|item| {
            let k = key(item);
            // The cursor names an edge by identity — `weight` only
            // ranks it, and a concurrent `associate` call can re-average
            // that weight between pages. Comparing on the current
            // weight alone would then be able to rank this exact edge
            // as still ahead of the cursor it itself produced, handing
            // it back a second time. The triple alone already uniquely
            // identifies an edge (the `edge_ids` bijection this
            // module's docs describe), so that exact edge is always
            // excluded regardless of what its weight now reads.
            if (k.1, k.2, k.3) == (seat.1, seat.2, seat.3) {
                return false;
            }
            rank(k, seat) == std::cmp::Ordering::Greater
        });
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
        matches.retain(|recollection| {
            let k = explore_rank(recollection);
            // The cursor names a row by identity — `distance` only ranks
            // it, and it's recomputed by a fresh BFS on every call, so a
            // concurrent structural change (e.g. retracting a bridging
            // edge) can change the distance this same edge gets between
            // pages. Comparing on the current distance alone would then
            // be able to rank this exact edge as still ahead of the
            // cursor it itself produced, handing it back a second time —
            // the same hazard page_by guards against for weight.
            if (k.1, k.2, k.3) == (seat.1, seat.2, seat.3) {
                return false;
            }
            k > seat
        });
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
        let value = crate::env::env_number("TAGURU_CROSS_SEARCH_CONCURRENCY", 4);
        if value == 0 {
            tracing::warn!("TAGURU_CROSS_SEARCH_CONCURRENCY=0 would fan out no work; using 4");
            4
        } else {
            value
        }
    })
}

/// `Attribution`'s wire shape: everything the library exposes, plus the
/// section label the server resolves from `paragraph` via
/// `AppState::resolve_sections`. `null` when the attribution carries no
/// paragraph locator, or when the locator falls outside every section
/// marker the ingest batch recorded for that source — never a
/// fabricated label.
#[derive(Serialize, Deserialize)]
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
#[derive(Serialize, Deserialize)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use taguru::context::{MatchKind, Resolution};
    use taguru::deadline::Deadline;

    use crate::limits::HeavyOpsLimiter;
    use crate::metrics::ResolveTier;

    use super::coverage::*;
    use super::import::*;
    use super::recall::*;
    use super::resolve::*;

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

        let dir = std::env::temp_dir().join(format!(
            "taguru-api-panic-response-shape-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();

        let app = Router::new().route("/boom", get(boom)).layer({
            let state = state.clone();
            CatchPanicLayer::custom(move |payload| panic_response(payload, &state))
        });

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

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The real layer order (CatchPanicLayer inside metrics::track_http,
    /// exactly as main.rs's routes()/serve() wire it) must let a
    /// panic-induced 500 land in BOTH the RED metrics (by status code,
    /// via `track_http`, which never needed this fix) AND
    /// `taguru_errors_total{kind="panic"}` (via `record_error`, which
    /// `panic_response` used to skip entirely — see its doc comment).
    #[tokio::test]
    async fn a_panicking_handler_is_visible_in_both_the_red_metrics_and_the_error_counter() {
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use axum::routing::get;
        use tower::util::ServiceExt;
        use tower_http::catch_panic::CatchPanicLayer;

        async fn boom() -> &'static str {
            panic!("kaboom")
        }

        let dir = std::env::temp_dir().join(format!(
            "taguru-api-panic-metrics-visibility-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let state = AppState::boot(dir.clone(), 1 << 20, None).unwrap();

        let app = Router::new()
            .route("/boom", get(boom))
            .layer({
                let state = state.clone();
                CatchPanicLayer::custom(move |payload| panic_response(payload, &state))
            })
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                crate::metrics::track_http,
            ))
            .with_state(state.clone());

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

        let rendered = state.metrics().render_prometheus(&state.gauge_snapshot());
        assert!(
            rendered.contains(
                "taguru_http_requests_total{method=\"GET\",route=\"/boom\",status=\"500\"} 1"
            ),
            "{rendered}"
        );
        assert!(
            rendered.contains("taguru_errors_total{kind=\"panic\"} 1"),
            "{rendered}"
        );

        let _ = std::fs::remove_dir_all(&dir);
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

    #[test]
    fn page_never_returns_the_cursors_own_edge_even_if_its_weight_moved() {
        // The cursor names "b" at weight 2.0. Between pages, a
        // concurrent `associate` call re-averages b's weight down to
        // 0.5. Ranking on that new weight alone would place b's
        // current record ahead of the cursor it itself produced (its
        // magnitude no longer matches what the cursor recorded), and
        // hand the same edge back a second time.
        let matches = vec![
            assoc("a", 3.0),
            assoc("b", 2.0),
            assoc("c", 1.0),
            assoc("d", 1.0),
        ];
        let (_, first) = page(matches, Some(2), None);
        let last = first.last().unwrap();
        assert_eq!(last.object, "b");
        let cursor = MatchCursor {
            weight: last.weight,
            subject: last.subject.clone(),
            label: last.label.clone(),
            object: last.object.clone(),
        };

        let mutated = vec![
            assoc("a", 3.0),
            assoc("b", 0.5),
            assoc("c", 1.0),
            assoc("d", 1.0),
        ];
        let (_, second) = page(mutated, Some(10), Some(&cursor));
        let objects: Vec<&str> = second.iter().map(|m| m.object.as_str()).collect();
        assert!(
            !objects.contains(&"b"),
            "b must never come back just because its own weight moved: got {objects:?}"
        );
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
    fn explore_page_never_returns_the_cursors_own_edge_even_if_its_distance_moved() {
        // The cursor names "b" at distance 1. Between pages, a
        // concurrent structural change (e.g. a bridging edge appearing
        // or being retracted) moves b's distance to 2 in the fresh BFS.
        // Ranking on that new distance alone would place b's current
        // record ahead of the cursor it itself produced, and hand the
        // same edge back a second time.
        let matches = vec![
            recollection("a", 1),
            recollection("b", 1),
            recollection("z", 2),
        ];
        let (_, first) = explore_page(matches, None, Some(2));
        let last = first.last().unwrap();
        assert_eq!(last.association.object, "b");
        let cursor = ExploreCursor {
            distance: last.distance,
            subject: last.association.subject.clone(),
            label: last.association.label.clone(),
            object: last.association.object.clone(),
        };

        let mutated = vec![
            recollection("a", 1),
            recollection("b", 2),
            recollection("z", 2),
        ];
        let (_, second) = explore_page(mutated, Some(&cursor), Some(10));
        let objects: Vec<&str> = second
            .iter()
            .map(|r| r.association.object.as_str())
            .collect();
        assert!(
            !objects.contains(&"b"),
            "b must never come back just because its own distance moved: got {objects:?}"
        );
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
    fn cross_page_by_never_returns_the_cursors_own_edge_even_if_its_weight_moved() {
        let targets = vec!["alpha".to_string(), "zeta".to_string()];
        let matches = vec![(0, assoc("青嶺", 1.0)), (1, assoc("青嶺", 1.0))];
        let (_, first) = cross_page_by(matches, Some(1), None, &targets);
        let (index, last) = &first[0];
        assert_eq!(targets[*index], "alpha");
        let cursor = CrossMatchCursor {
            weight: last.weight,
            context: targets[*index].clone(),
            subject: last.subject.clone(),
            label: last.label.clone(),
            object: last.object.clone(),
        };

        // alpha's edge re-averages down between pages — same edge,
        // moved weight, must not rank ahead of the cursor it produced.
        let mutated = vec![(0, assoc("青嶺", 0.2)), (1, assoc("青嶺", 1.0))];
        let (_, second) = cross_page_by(mutated, Some(10), Some(&cursor), &targets);
        let contexts: Vec<&str> = second.iter().map(|(i, _)| targets[*i].as_str()).collect();
        assert!(
            !contexts.contains(&"alpha"),
            "alpha's edge must never come back just because its own weight moved: got {contexts:?}"
        );
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

    /// `limit_to_reach` must be the smallest limit VERIFIED to serve the
    /// canonical, not just the first one a doubling search from
    /// `merged_at + 1` happens to hit. This reproduces the exact shape
    /// the fix at `limit_to_reach`'s definition targets: a canonical
    /// that ties every decoy's lexical (containment) score — so
    /// alphabetical order buries it last in the lexical tier — while
    /// ALSO being the semantic tier's rank-1 hit. At `limit = 12`
    /// (one past its lexical-only rank 12), `trim_to_limit` drops the
    /// canonical's OWN lexical occurrence to make room, and the
    /// dedup in `merge_tiers` means no semantic duplicate is left to
    /// carry it — a below_cutoff verdict whose true `limit_to_reach`
    /// is 1 (semantic tier alone already serves it), not the 13 an
    /// unconditional doubling search from 12 would report.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn limit_to_reach_reports_the_smallest_verified_limit_not_a_stale_doubling_hit() {
        use crate::embedding::{EmbedPurpose, EmbeddingProvider};
        use crate::registry::ContextMeta;
        use axum::Router;
        use axum::body::Body;
        use axum::http::Request as HttpRequest;
        use axum::routing::post;
        use tower::util::ServiceExt;

        struct PurposeSplitEmbeddings {
            index: Vec<(&'static str, Vec<f32>)>,
            query: Vec<(&'static str, Vec<f32>)>,
        }
        impl EmbeddingProvider for PurposeSplitEmbeddings {
            fn model(&self) -> &str {
                "mock-purpose-split"
            }
            fn embed(
                &self,
                texts: &[&str],
                purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                let keys = match purpose {
                    EmbedPurpose::Index => &self.index,
                    EmbedPurpose::Query => &self.query,
                };
                Ok(texts
                    .iter()
                    .map(|text| {
                        keys.iter()
                            .find(|(key, _)| text.starts_with(key))
                            .map(|(_, vector)| vector.clone())
                            .unwrap_or_else(|| vec![0.0, 0.0, 1.0])
                    })
                    .collect())
            }
        }

        let dir = std::env::temp_dir().join(format!(
            "taguru-api-limit-to-reach-nonmonotonic-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        // rank-1 cosine 1.0 for the canonical, rank-2 cosine 0.9 for a
        // decoy that never touches the lexical tier — two semantic
        // candidates are what actually produce the post-merged_at dip
        // (one alone would leave nothing behind after the dedup drops
        // it, per the comment at the fix site).
        let embedder = Some(Arc::new(PurposeSplitEmbeddings {
            index: vec![
                ("AAA0012", vec![1.0, 0.0, 0.0]),
                (
                    "ZZZ_SEMANTIC_ONLY",
                    vec![0.9, (1.0f32 - 0.81f32).sqrt(), 0.0],
                ),
            ],
            query: vec![("AAA", vec![1.0, 0.0, 0.0])],
        }) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), 1 << 20, embedder).unwrap();
        state.create("ctx", ContextMeta::default()).unwrap();
        state
            .write_context("ctx", |context| {
                // 12 names all containing "AAA", all the same length, so
                // containment scores them identically and alphabetical
                // order (sort_resolutions's tiebreak) puts AAA0012 dead
                // last in the lexical tier.
                for i in 1..=12 {
                    let name = format!("AAA{i:04}");
                    context.associate(&name, "分類", "何か", 1.0).unwrap();
                }
                context
                    .associate("ZZZ_SEMANTIC_ONLY", "分類", "何か", 1.0)
                    .unwrap();
            })
            .unwrap();
        state
            .refresh_embeddings("ctx", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let app = Router::new()
            .route("/contexts/{name}/resolve/explain", post(explain_resolve))
            .layer(axum::Extension(Deadline::unbounded()))
            .with_state(state);

        let response = app
            .oneshot(
                HttpRequest::builder()
                    .method("POST")
                    .uri("/contexts/ctx/resolve/explain")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"cue": "AAA", "expected": "AAA0012", "limit": 12}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let result = &body["result"];

        assert_eq!(result["verdict"], "below_cutoff", "{result}");
        assert_eq!(result["ranking"]["rank"], 12, "{result}");
        assert_eq!(result["ranking"]["served"], false, "{result}");
        assert_eq!(
            result["ranking"]["limit_to_reach"], 1,
            "the canonical is ALSO the top semantic hit, so limit 1 already \
             serves it in semantic-only mode; a doubling search that starts \
             past merged_at and never looks back would report 13 — {result}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
