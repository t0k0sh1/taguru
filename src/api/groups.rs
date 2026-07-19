use std::collections::BTreeSet;
use std::time::Instant;

use axum::extract::State;
use axum::response::Response;

use serde::{Deserialize, Serialize};

use taguru::deadline::Deadline;

use crate::groups::GroupRecord;
use crate::metrics::ErrorKind;
use crate::registry::{AppState, CreateGroupError, RenameGroupError, UpdateGroupError};

use super::aliases::KeysetQuery;
use super::contexts::RenameRequest;
use super::{
    AppBytes, AppJson, AppPath, AppQuery, ErrorCode, MAX_CONTEXT_NAME_BYTES, MAX_DESCRIPTION_BYTES,
    MAX_MATCH_LIMIT, clamp, deadline_exceeded, error, group_not_found, key_name, nesting_refusal,
    ok, optional_body, over_cap_refusal, overlong, oversized,
};

/// A bounded group-directory page; `total` counts the whole directory,
/// cursor-independent, exactly as [`ContextPage`]'s does.
#[derive(Serialize, Deserialize)]
pub struct GroupPage {
    pub total: usize,
    pub groups: Vec<GroupEntry>,
}

/// One group as served — the directory row, the single GET, and the
/// PATCH response are all this one shape, as with [`DirectoryEntry`].
#[derive(Serialize, Deserialize)]
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
pub(super) fn scope_allows(
    scope: &Option<axum::Extension<crate::auth::KeyScope>>,
    name: &str,
) -> bool {
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
pub(super) fn scoped_member_contexts<C: FromIterator<String>>(
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
pub(super) fn scope_refusal<'a>(
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
