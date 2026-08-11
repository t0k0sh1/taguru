use std::collections::BTreeSet;
use std::time::Instant;

use axum::extract::State;
use axum::response::Response;

use serde::{Deserialize, Serialize};

use taguru::deadline::{Deadline, DeadlineExceeded};

use crate::groups::GroupRecord;
use crate::metrics::ErrorKind;
use crate::registry::{AppState, CreateGroupError, RenameGroupError, UpdateGroupError};

use super::aliases::KeysetQuery;
use super::contexts::RenameRequest;
use super::{
    AppBytes, AppJson, AppPath, AppQuery, ErrorCode, MAX_CONTEXT_NAME_BYTES, MAX_DESCRIPTION_BYTES,
    MAX_MATCH_LIMIT, clamp_page, deadline_exceeded, error, group_not_found, key_name,
    nesting_refusal, ok, optional_body, over_cap_refusal, overlong, oversized,
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
    /// Change token over the group's transitive member contexts: a
    /// stable hash of each visible member's name and revision counters
    /// (see `ContextRevision`), so a group-level cache invalidates
    /// exactly when a relevant member changed — a member write, an
    /// embedding refresh, a rename, or a membership edit all move it;
    /// anything else leaves it alone. Computed over the slice the
    /// caller's key can see, so it leaks no change-signal about
    /// contexts beyond a scoped grant. Compare for equality only, and
    /// only against the same server process — the member revisions
    /// behind it carry the same restart caveats they do individually.
    /// `serde(default)` so a router merging rows from an older shard
    /// reads an empty token instead of refusing the row.
    #[serde(default)]
    pub fingerprint: String,
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
///
/// Propagates [`DeadlineExceeded`] straight from [`group_fingerprint`]:
/// the fingerprint walk is the only expensive part of assembling a
/// row, so there is nothing left worth finishing once it gives up.
fn group_entry(
    state: &AppState,
    name: String,
    record: GroupRecord,
    scope: &Option<axum::Extension<crate::auth::KeyScope>>,
    deadline: &Deadline,
) -> Result<GroupEntry, DeadlineExceeded> {
    let fingerprint = group_fingerprint(state, &name, scope, deadline)?;
    Ok(GroupEntry {
        name,
        description: record.description,
        contexts: scoped_member_contexts(record.contexts, scope),
        groups: record.groups,
        fingerprint,
    })
}

/// A single group's [`group_entry`] response, gated on the deadline and
/// kept off the async worker — the fingerprint walks the group's
/// transitive member closure, cost that scales with nesting/membership
/// rather than with this being a single row. Shared by every handler
/// that returns one group's entry after `list_groups`' own inline
/// (page-of-many) version of the same gate.
fn deadline_gated_group_entry(
    state: &AppState,
    name: String,
    record: GroupRecord,
    scope: &Option<axum::Extension<crate::auth::KeyScope>>,
    deadline: &Deadline,
    started_at: Instant,
) -> Response {
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    match tokio::task::block_in_place(|| group_entry(state, name, record, scope, deadline)) {
        Ok(entry) => ok(entry, started_at),
        Err(DeadlineExceeded) => deadline_exceeded(started_at),
    }
}

/// The change token on one group row: FNV-1a over the scope-visible
/// transitive context closure, each member as its length-prefixed name
/// followed by the three revision counters — structurally unambiguous,
/// so distinct closures cannot collide by concatenation. The closure
/// (not the direct members) because a group search fans out through
/// nested children, and sorted iteration (the closure is a `BTreeSet`)
/// makes the hash deterministic. A member registered but mid-delete
/// contributes nothing, exactly as it answers no search.
///
/// `deadline` is checked on EVERY iteration of the closure walk, not
/// only once before the walk starts: each iteration takes a
/// per-context revision lock, so cost scales with the group's nesting
/// and membership rather than with this being "one row" — a deeply
/// nested or large group could otherwise run well past the point
/// where the HTTP client's own timeout gave up on the response,
/// burning worker capacity on a result nobody will read (#318). A
/// caller that must finish regardless (`update_group`, after its
/// write already committed) passes `Deadline::unbounded()`.
fn group_fingerprint(
    state: &AppState,
    name: &str,
    scope: &Option<axum::Extension<crate::auth::KeyScope>>,
    deadline: &Deadline,
) -> Result<String, DeadlineExceeded> {
    let mut digest = crate::hash::FNV1A_OFFSET;
    for context in state.group_context_closures([name]) {
        if deadline.expired() || injected_fingerprint_loop_expiry() {
            return Err(DeadlineExceeded);
        }
        if !scope_allows(scope, &context) {
            continue;
        }
        let Some(revision) = state.context_revision(&context) else {
            continue;
        };
        digest = crate::hash::fnv1a_fold(digest, (context.len() as u64).to_le_bytes());
        digest = crate::hash::fnv1a_fold(digest, context.bytes());
        digest = crate::hash::fnv1a_fold(digest, revision.graph.to_le_bytes());
        digest = crate::hash::fnv1a_fold(digest, revision.passages.to_le_bytes());
        digest = crate::hash::fnv1a_fold(digest, revision.config.to_le_bytes());
    }
    Ok(format!("{digest:016x}"))
}

// Test-only deterministic fault injection for `group_fingerprint`'s
// per-iteration deadline check.
#[cfg(test)]
thread_local! {
    static EXPIRE_FINGERPRINT_LOOP_AFTER: std::cell::Cell<Option<u32>> =
        const { std::cell::Cell::new(None) };
}

/// Test-only deterministic fault injection for [`group_fingerprint`]'s
/// per-iteration deadline check: mirrors
/// [`crate::storage::injected_persistence_failure`]'s "counted successes,
/// then fire" shape, but for a `Deadline` instead of an `io::Error`.
/// Real wall-clock expiry inside a fingerprint walk is hard to land
/// deterministically (each iteration is an in-memory lock read, done
/// in well under a microsecond), so a regression test that needs the
/// check to fire partway through — not before the walk starts, not
/// after it finishes — arms this instead of racing a `Duration`.
/// Thread-local, so parallel tests stay independent; self-clearing
/// once it fires, so one arm never leaks into an unrelated later test
/// on a reused thread.
///
/// Arms the fault: the next `remaining` per-iteration checks report
/// "not yet expired" as usual, and the one after that reports expired.
#[cfg(test)]
fn expire_fingerprint_loop_after(remaining: u32) {
    EXPIRE_FINGERPRINT_LOOP_AFTER.with(|cell| cell.set(Some(remaining)));
}

#[cfg(test)]
fn injected_fingerprint_loop_expiry() -> bool {
    EXPIRE_FINGERPRINT_LOOP_AFTER.with(|cell| match cell.get() {
        Some(0) => {
            cell.set(None);
            true
        }
        Some(remaining) => {
            cell.set(Some(remaining - 1));
            false
        }
        None => false,
    })
}

#[cfg(not(test))]
fn injected_fingerprint_loop_expiry() -> bool {
    false
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
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppQuery(query): AppQuery<KeysetQuery>,
) -> Response {
    let started_at = Instant::now();
    let (total, page) = state.group_page(
        query.after.as_deref(),
        clamp_page(query.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT),
    );
    // Each row's fingerprint walks its transitive member closure and
    // takes a per-context revision lock — cost that scales with group
    // nesting/membership, not with the page size alone. This upfront
    // check (like `list_contexts`' whole-directory path, 35f5ead) only
    // catches a budget already spent before the page loop starts; the
    // per-ROW gate lives inside `group_entry`/`group_fingerprint` now
    // (#318), so a page of many groups stops promptly at whichever row
    // is in flight when the deadline lands instead of finishing every
    // row that happened to be queued first. Kept off the async worker,
    // the same shape as every other fsync/scan-bearing handler here.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let groups: Result<Vec<_>, DeadlineExceeded> = tokio::task::block_in_place(|| {
        page.into_iter()
            .map(|(name, record)| group_entry(&state, name, record, &scope, &deadline))
            .collect()
    });
    match groups {
        Ok(groups) => ok(GroupPage { total, groups }, started_at),
        Err(DeadlineExceeded) => deadline_exceeded(started_at),
    }
}

pub async fn get_group(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    match state.group(&name) {
        Some(record) => {
            deadline_gated_group_entry(&state, name, record, &scope, &deadline, started_at)
        }
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
        Ok(record) => {
            // The write above already committed (fsync + rename) — a
            // deadline expiring from here on must not turn a durable
            // mutation into an apparent failure. Returning a timeout
            // here (as `deadline_gated_group_entry` would) invites a
            // client to retry a write that already landed; finish
            // computing the fingerprint and report success instead,
            // same as `get_group` would for the record `update_group`
            // just wrote. `group_fingerprint`'s per-iteration deadline
            // check (#318) would otherwise cut this short on the
            // caller's now-possibly-spent request budget, so this
            // deliberately hands it a fresh, unbounded one instead of
            // the request's own — the only caller in this file that
            // does, and only because the mutation is already durable.
            let entry = tokio::task::block_in_place(|| {
                group_entry(&state, name, record, &scope, &Deadline::unbounded())
            });
            match entry {
                Ok(entry) => ok(entry, started_at),
                // `Deadline::unbounded()` never expires; kept for
                // exhaustiveness (same shape as `compact.rs`'s
                // `report_outcome`).
                Err(DeadlineExceeded) => deadline_exceeded(started_at),
            }
        }
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::registry::ContextMeta;

    use super::*;

    /// A fresh, on-disk-backed [`AppState`] — same construction
    /// [`crate::api`]'s own `restore_refusal_frames_a_spent_budget_as_a_resumable_timeout`
    /// test uses, kept local rather than reaching into
    /// `crate::registry::test_support` (a private module of `registry`
    /// that this file, outside that module tree, cannot name).
    fn scratch_state(tag: &str) -> AppState {
        let dir =
            std::env::temp_dir().join(format!("taguru-api-groups-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        AppState::boot(dir, usize::MAX, None).unwrap()
    }

    /// A one-context group, ready for the deadline tests below.
    fn one_context_group(state: &AppState, group: &str, context: &str) {
        state.create(context, ContextMeta::default()).unwrap();
        state
            .create_group(
                group,
                "d".to_string(),
                BTreeSet::from([context.to_string()]),
                BTreeSet::new(),
            )
            .unwrap();
    }

    /// A deadline that has already elapsed by the time it is checked —
    /// mirrors `deadline.rs`'s own `a_zero_budget_is_already_expired`:
    /// the clock advances between construction and observation even
    /// under load, so a zero budget cannot be observed as anything but
    /// expired.
    fn already_expired_deadline() -> Deadline {
        let deadline = Deadline::after(Duration::ZERO);
        std::thread::sleep(Duration::from_millis(1));
        assert!(deadline.expired(), "a zero budget must read as expired");
        deadline
    }

    /// Regression for #318: a deadline spent before `group_fingerprint`
    /// is ever called must refuse without walking a single context —
    /// covers both `list_groups`' and `deadline_gated_group_entry`'s
    /// pre-loop gate (unchanged by this fix) and `group_fingerprint`
    /// itself now returning the same `Err` a caller that skipped the
    /// pre-check would need.
    #[test]
    fn group_fingerprint_refuses_immediately_once_the_deadline_has_already_expired() {
        let state = scratch_state("fp-expired");
        one_context_group(&state, "kura", "sake");

        let deadline = already_expired_deadline();
        let result = group_fingerprint(&state, "kura", &None, &deadline);
        assert_eq!(result, Err(DeadlineExceeded));
    }

    /// Regression for #318: the per-context deadline check inside
    /// `group_fingerprint`'s loop must fire on a LATER iteration, not
    /// only once before the whole walk starts. A real
    /// `Deadline::after(tiny duration)` cannot land this
    /// deterministically — each iteration is an in-memory lock read
    /// that completes in well under a microsecond, so a real clock
    /// would need to race an unpredictable number of iterations. The
    /// test instead arms `expire_fingerprint_loop_after`, the
    /// thread-local fault this file defines for exactly this shape
    /// (mirroring `crate::storage`'s `fail_persistence_ops_after`):
    /// the deadline passed here never actually expires, so this can
    /// only return `Err` if the loop consults the hook on more than
    /// one iteration — the "before the pre-#318 fix, this hook did not
    /// even exist" case that would otherwise return `Ok`.
    #[test]
    fn group_fingerprint_checks_the_deadline_on_a_later_iteration_not_only_before_the_walk_starts()
    {
        let state = scratch_state("fp-mid-loop");
        for context in ["sake", "bunko", "cha"] {
            state.create(context, ContextMeta::default()).unwrap();
        }
        state
            .create_group(
                "kura",
                "d".to_string(),
                BTreeSet::from(["sake".to_string(), "bunko".to_string(), "cha".to_string()]),
                BTreeSet::new(),
            )
            .unwrap();

        // The first per-iteration check reports "not yet expired"; the
        // second reports expired — proof the check runs more than
        // once across the three-context closure.
        expire_fingerprint_loop_after(1);
        let result = group_fingerprint(&state, "kura", &None, &Deadline::unbounded());
        assert_eq!(result, Err(DeadlineExceeded));
    }

    /// Regression for #318: `get_group` with an already-expired
    /// deadline answers the standard timeout envelope instead of
    /// computing a fingerprint — the HTTP-handler-level twin of the
    /// unit test above, exercised the same way `api.rs`'s own
    /// `restore_refusal_frames_a_spent_budget_as_a_resumable_timeout`
    /// calls a handler function directly and inspects the JSON body.
    #[tokio::test]
    async fn get_group_with_an_already_expired_deadline_answers_timeout() {
        let state = scratch_state("get-expired");
        one_context_group(&state, "kura", "sake");
        let deadline = already_expired_deadline();

        let response = get_group(
            State(state),
            AppPath("kura".to_string()),
            None,
            axum::Extension(deadline),
        )
        .await;
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["status"], "error", "{body}");
        assert_eq!(body["code"], ErrorCode::Timeout.as_str(), "{body}");
    }

    /// Regression for #318: on a page of several groups, a deadline
    /// that lands MID-PAGE — after the first row's fingerprint has
    /// already run to completion, before the second row's has even
    /// started — must answer the standard timeout envelope rather than
    /// a partial or a complete page. `list_groups`' own `Deadline`
    /// never actually expires here; the injected fault fires inside
    /// the second row's `group_fingerprint` call, standing in for a
    /// real deadline landing between two rows of the same page — the
    /// scenario the pre-#318 code (one check before the whole loop)
    /// could not detect at all.
    // `list_groups` runs its work under `tokio::task::block_in_place`,
    // which panics off a single-threaded runtime — unlike the
    // already-expired-deadline tests above, this one's deadline never
    // actually expires, so it reaches that call.
    #[tokio::test(flavor = "multi_thread")]
    async fn list_groups_answers_timeout_not_a_partial_or_complete_page_when_the_deadline_lands_mid_page()
     {
        let state = scratch_state("list-mid-page");
        // Name order matters: "aaa" must be paged before "bbb" so its
        // fingerprint is the one that completes before the fault
        // fires on the row after it.
        one_context_group(&state, "aaa", "ctx-a");
        one_context_group(&state, "bbb", "ctx-b");

        expire_fingerprint_loop_after(1);
        let response = list_groups(
            State(state),
            None,
            axum::Extension(Deadline::unbounded()),
            AppQuery(KeysetQuery {
                limit: None,
                after: None,
                prefix: None,
            }),
        )
        .await;
        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["status"], "error", "{body}");
        assert_eq!(body["code"], ErrorCode::Timeout.as_str(), "{body}");
    }
}
