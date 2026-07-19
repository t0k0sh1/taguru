use std::time::{Duration, Instant};

use axum::extract::State;
use axum::response::{IntoResponse, Response};

use serde::{Deserialize, Serialize};

use taguru::deadline::Deadline;

use crate::metrics::ErrorKind;
use crate::registry::{AppState, ContextMeta, CreateError, RenameContextError};

use super::groups::scope_refusal;
use super::{
    AppBytes, AppJson, AppPath, AppQuery, ErrorCode, MAX_CONTEXT_NAME_BYTES, MAX_DESCRIPTION_BYTES,
    MAX_MATCH_LIMIT, clamp, deadline_exceeded, error, key_name, not_found, ok, optional_body,
    oversized,
};

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
#[derive(Serialize, Deserialize)]
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
    axum::Extension(deadline): axum::Extension<Deadline>,
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
        // An allow-list or `pinned` forces the whole-directory scan
        // below instead of `directory_page`'s O(log n + k) seek — gate
        // it on the deadline like every other handler whose read cost
        // scales with the directory rather than the page (35f5ead).
        if deadline.expired() {
            return deadline_exceeded(started_at);
        }
        let mut directory: Vec<_> = tokio::task::block_in_place(|| match &allowed {
            Some(allowed) => allowed
                .iter()
                .filter_map(|name| state.directory_entry(name))
                .collect(),
            None => state.directory(),
        });
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
    let mut body = include_str!("../llm-protocol.md").to_string();
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
