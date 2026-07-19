use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::Response;

use serde::{Deserialize, Serialize};

use taguru::context::{Association, Context};
use taguru::deadline::Deadline;

use crate::metrics::SearchOp;
use crate::registry::AppState;

use super::aliases::{OneOrMany, as_refs, validate_positions};
use super::groups::{scope_allows, scope_refusal};
use super::{
    AppJson, AppPath, CrossMatchPage, DEFAULT_MATCH_LIMIT, ErrorCode, MAX_MATCH_LIMIT, MatchCursor,
    MatchPage, access_error, associations_out, bounded_parallel_map, clamp, cross_associations_out,
    cross_search_concurrency, deadline_exceeded, error, group_not_found, ok, overlong, page,
    search_log_enabled,
};

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
pub(super) fn cross_targets(
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
pub(crate) fn cross_rank(
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
pub(super) fn cross_page_by(
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
            let k = cross_key(&targets[*index], found);
            // Same reasoning as `page_by`: `(context, subject, label,
            // object)` alone already identifies the edge the cursor
            // names, so a weight that moved between pages can never
            // make that same edge outrank its own cursor a second time.
            if (k.1, k.2, k.3, k.4) == (seat.1, seat.2, seat.3, seat.4) {
                return false;
            }
            cross_rank(k, seat) == std::cmp::Ordering::Greater
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

#[derive(Debug, Serialize, Deserialize)]
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
    if let Some(refusal) = validate_positions(
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

#[derive(Debug, Serialize, Deserialize)]
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
    if let Some(refusal) = validate_positions(
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
