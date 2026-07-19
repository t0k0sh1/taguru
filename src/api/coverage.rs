use std::time::Instant;

use axum::extract::State;
use axum::response::Response;

use serde::{Deserialize, Serialize};

use taguru::deadline::Deadline;

use crate::registry::{AccessError, AppState};

use super::aliases::KeysetQuery;
use super::{
    AppJson, AppPath, AppQuery, ErrorCode, MAX_MATCH_LIMIT, MatchCursor, MatchPage, access_error,
    associations_out, clamp, deadline_exceeded, error, not_found, ok, overlong, page,
};

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
    // The gloss half above can spend the whole budget by itself; recheck
    // before the passage half's own synchronous work (snapshot clone,
    // sidecar load, full paragraph scan) runs unconditionally ahead of
    // its first internal deadline check.
    if deadline.expired() {
        return deadline_exceeded(started_at);
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
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // A `prefix` filter defines the population rather than a cursor, so
    // — like `pinned` on `list_contexts` — it forces the whole-vocabulary
    // path instead of the BTreeMap-seeking `label_page` fast path, which
    // has no way to know in advance how many prefix-matching labels lie
    // within any given range. That path clones and sorts every label in
    // the context — the same unconditional whole-table cost as
    // `unreachable_from`'s full scan — so it runs under block_in_place;
    // the bare-cursor path below stays off it, same as
    // `recall`/`describe`, since its BTreeMap seek is already bounded by
    // `limit`.
    let outcome = match query.prefix.as_deref() {
        Some(prefix) => tokio::task::block_in_place(|| {
            state.read_context(&name, |context| {
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
                LabelPage { total, labels }
            })
        }),
        None => state.read_context(&name, |context| {
            let (total, labels) = context.label_page(query.after.as_deref(), limit);
            LabelPage { total, labels }
        }),
    };
    match outcome {
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
    // unsourced_edges, so it gets the same block_in_place + deadline
    // treatment rather than running the scan straight on the async task.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let loaded = tokio::task::block_in_place(|| {
        state
            .read_context(&name, |context| {
                let origins: Vec<&str> = request.origins.iter().map(String::as_str).collect();
                context
                    .unreachable_from(&origins, deadline)
                    .map_err(|_| AccessError::DeadlineExceeded)
            })
            .and_then(std::convert::identity)
    });
    match loaded {
        Ok(result) => {
            let (total, matches) = page(result, request.limit, request.after.as_ref());
            // A graph read like recall/query/explore/activate — the
            // usage counters must agree with that grouping. Zero
            // orphans is the audit SUCCEEDING, though, not a miss, so
            // it never counts as an empty read.
            state.note_read(&name, false);
            let matches = associations_out(&state, &name, matches);
            // No plan: an orphan audit is not a search — the field
            // stays off the wire here.
            ok(
                MatchPage {
                    total,
                    matches,
                    plan: None,
                },
                started_at,
            )
        }
        Err(failure) => access_error(&state, failure, &name, started_at),
    }
}
