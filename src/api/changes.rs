//! `GET /contexts/{name}/changes` (#422): the polling change feed —
//! one page of recent content-change events after an opaque cursor,
//! served from the bounded in-memory ring `crate::registry::changes`
//! keeps per context. A lost position (restart, recreate, ring
//! overflow) answers `stale_cursor` (410) rather than a silently
//! incomplete page; the client's move is a full resync, then tailing
//! again from a fresh cursor (a call without `since`).

use std::time::Instant;

use axum::extract::State;
use axum::response::Response;

use serde::{Deserialize, Serialize};

use taguru::deadline::Deadline;

use crate::registry::{AppState, ChangeEvent, ChangesOutcome};

use super::{
    AppPath, AppQuery, DEFAULT_MATCH_LIMIT, ErrorCode, MAX_MATCH_LIMIT, access_error, clamp,
    deadline_exceeded, error, ok,
};

/// `?since=&limit=`. `since` is the opaque cursor a previous page's
/// `next` handed back; omitted means "start tailing now" — an empty
/// page whose `next` is the current position, the bootstrap for a
/// client that just finished a full sync. `limit` follows the match
/// endpoints' clamp (default 100, ceiling 1000).
#[derive(Debug, Deserialize)]
pub struct ChangesQuery {
    pub since: Option<String>,
    pub limit: Option<usize>,
}

/// One page of the feed. `next` is always present — pass it as the
/// next call's `since`. `more: true` means events past `limit` are
/// already waiting: poll again immediately instead of waiting out the
/// interval.
#[derive(Serialize)]
pub struct ChangesPage {
    pub events: Vec<ChangeEvent>,
    pub next: String,
    pub more: bool,
}

pub async fn changes(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppQuery(query): AppQuery<ChangesQuery>,
) -> Response {
    let started_at = Instant::now();
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    match state.context_changes(
        &name,
        query.since.as_deref(),
        clamp(query.limit, DEFAULT_MATCH_LIMIT, MAX_MATCH_LIMIT),
    ) {
        Ok(ChangesOutcome::Page { events, next, more }) => {
            state.note_read(&name, events.is_empty());
            ok(ChangesPage { events, next, more }, started_at)
        }
        Ok(ChangesOutcome::Stale) => error(
            ErrorCode::StaleCursor,
            "the cursor's history is no longer held (a restart, a recreate, or more \
             changes than the feed retains) — run a full resync, then tail again from \
             a fresh cursor (GET /changes without since)",
            started_at,
        ),
        Err(failure) => access_error(&state, failure, &name, started_at),
    }
}
