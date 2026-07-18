use std::time::Instant;

use axum::extract::State;
use axum::response::Response;

use serde::{Deserialize, Serialize};

use taguru::context::Context;
use taguru::deadline::Deadline;

use crate::metrics::SearchOp;
use crate::registry::AppState;

use super::{
    ActivationOut, AppJson, AppPath, ExploreCursor, MAX_EXPLORE_DEPTH, MAX_MATCH_LIMIT,
    RecollectionOut, access_error, activations_out, clamp, deadline_exceeded, explore_page, ok,
    overlong, recollections_out, search_log_enabled,
};

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
