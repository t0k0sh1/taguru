//! `POST /contexts/{name}/evidence` (#305, ADR 0006 §5.1, §5.4, §10,
//! §11, §13.1-13.2): the opt-in HTTP surface over #303's candidate
//! normalization and #304's budgeted selection. This is the first real
//! caller of both — `super::EvidenceCandidate::from_*`, `super::fuse`,
//! and `select::select` were exercised only by tests until this
//! handler existed.
//!
//! One request runs the same five-lane fan-out the composed MCP
//! `retrieve` tool (`src/mcp/retrieve.rs`) runs as separate HTTP round
//! trips, but server-side and in one call: resolve each origin cue,
//! query (only when `labels` pins the facets) and activate (always)
//! for graph associations, search passages, optionally search
//! community summaries, and resolve every located citation — then
//! normalize, fuse, and budget-select the combined pool. `retrieve`
//! itself is untouched; ADR 0006 §3 A rejected extending it in favor
//! of this new, server-implemented endpoint (§2.1: `retrieve` has no
//! server-side implementation of its own to extend).

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::Instant;

use axum::extract::State;
use axum::response::Response;

use serde::{Deserialize, Serialize};

use taguru::deadline::Deadline;

use crate::metrics::SearchOp;
use crate::registry::{AppState, PassageSearchLanes};

use crate::api::aliases::OneOrMany;
use crate::api::communities::{
    CommunityLaneOutcome, check_derived_scope, community_hits, derived_context_name,
};
use crate::api::resolve::{ResolveRequest, resolve_served};
use crate::api::sources::{
    Citation, LanePlan, NO_QUERY_TERMS_REASON, PassageHit, ZERO_LIMIT_REASON,
};
use crate::api::{
    AppJson, AppPath, MAX_MATCH_LIMIT, access_error, activations_out, associations_out, clamp,
    deadline_exceeded, not_found, ok, overlong, page,
};

use super::budget::{BudgetLimits, BudgetRequest, BudgetUsage};
use super::rerank::{self, RerankRequest, RerankerPlan};
use super::select::{self, EvidenceItem, OmittedCandidate, SelectionPlan, select_with_reorder};
use super::{CitationEntry, EvidenceCandidate, FusedCandidate, fuse};

/// `search_limit`'s own ceiling — deliberately lower than the house
/// `MAX_MATCH_LIMIT` (1000) every other endpoint's list-limit reuses.
/// `search_limit` bounds both the passage and (opt-in) community
/// lanes, and every passage candidate it admits is later compared
/// pairwise against every other kept passage by
/// [`select::suppress_near_duplicate_passages`] — an O(n²) step by
/// construction (ADR 0006 §9 fixes the near-duplicate detector's
/// *rule*, not a request-tunable pool size to run it over). At the
/// house ceiling, a caller could ask for up to 2000 combined passage/
/// community candidates and drive on the order of two million pairwise
/// comparisons on one request from an ordinary Read-scoped key. 200
/// keeps the worst case two orders of magnitude smaller while staying
/// well above the default (5) and any realistic caller's real need —
/// nothing in ADR 0006 requires `search_limit` to share
/// `MAX_MATCH_LIMIT`'s ceiling, only that it default to 5 (§5.1).
const MAX_EVIDENCE_SEARCH_LIMIT: usize = 200;

/// `origins`/`labels` accept `"..."` and `["...", ...]` interchangeably
/// (the same contract `retrieve`'s own arguments use); flattened to an
/// owned list once, up front, so every lane below works with a plain
/// `Vec<String>`.
fn cues(value: OneOrMany) -> Vec<String> {
    match value {
        OneOrMany::One(cue) => vec![cue],
        OneOrMany::Many(cues) => cues,
    }
}

#[derive(Debug, Deserialize)]
pub struct AssembleEvidenceRequest {
    /// Required — the only non-optional field (ADR 0006 §5.1). Reuses
    /// `retrieve`'s own cue-list contract, `origins` cap included
    /// ([`overlong`]'s `MAX_INPUT_ITEMS`, same value as `retrieve`'s
    /// own `MAX_ORIGIN_CUES`).
    pub origins: OneOrMany,
    /// Relation labels to query on, alongside the always-run activate
    /// — same as `retrieve`'s `labels`.
    pub labels: Option<OneOrMany>,
    pub dice_floor: Option<f64>,
    pub semantic_floor: Option<f32>,
    pub resolve_limit: Option<usize>,
    pub activate_decay: Option<f64>,
    pub activate_limit: Option<usize>,
    /// The passage/community lanes' query text. Omitted means the
    /// canonical query ADR 0006 §12 already defines for the reranker
    /// boundary: `origins` joined with `"; "` in request order — so a
    /// caller who never names an explicit fallback query still gets
    /// mixed graph/passage evidence by default, and the two lanes
    /// agree with whatever query a future configured reranker sees.
    pub text_fallback_query: Option<String>,
    pub search_limit: Option<usize>,
    #[serde(default)]
    pub include_communities: bool,
    pub budget: Option<BudgetRequest>,
    /// Opt-in reranking (#307, ADR 0006 §12): absent means the §7
    /// deterministic RRF order is used untouched, at no network or
    /// credential cost. Present but no provider configured, a model
    /// mismatch, or any provider failure all degrade to that same
    /// deterministic order — `plan.reranker.reason` names why, and the
    /// call still answers 200 either way (ADR 0006 §11).
    pub rerank: Option<RerankRequest>,
}

/// ADR 0006 §10's `EvidencePlan`: everything [`select::SelectedEvidence`]
/// does not already carry, because it only exists once a live corpus
/// and an HTTP/MCP request are in the picture.
#[derive(Debug, Serialize, Deserialize)]
pub struct EvidencePlan {
    pub lanes: EvidenceLanesPlan,
    pub selection: SelectionPlan,
    pub reranker: RerankerPlan,
}

/// One [`LanePlan`] per retrieval lane this endpoint fans out to (ADR
/// 0006 §10). Reuses the existing `LanePlan { ran, reason, floor }`
/// shape (`src/api/sources.rs`) — the same shape `sources/search`'s
/// own `plan.contexts[].lanes` already uses — rather than a
/// coarser bool, so a caller already familiar with that response reads
/// this one the same way.
#[derive(Debug, Serialize, Deserialize)]
pub struct EvidenceLanesPlan {
    pub resolve: LanePlan,
    pub query: LanePlan,
    pub activate: LanePlan,
    pub passages: LanePlan,
    pub communities: LanePlan,
    pub citations: LanePlan,
}

/// ADR 0006 §10's `EvidencePackage`: [`select::SelectedEvidence`] with
/// its `selection` folded into the fuller [`EvidencePlan`] this
/// endpoint can build (lanes + reranker) but the pure `select` function
/// cannot.
#[derive(Debug, Serialize, Deserialize)]
pub struct EvidencePackage {
    pub items: Vec<EvidenceItem>,
    pub citations: Vec<CitationEntry>,
    pub budget: BudgetUsage,
    pub omitted: Vec<OmittedCandidate>,
    pub omitted_total: usize,
    pub omitted_by_reason: BTreeMap<String, usize>,
    pub plan: EvidencePlan,
}

pub async fn assemble_evidence(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<AssembleEvidenceRequest>,
) -> Response {
    let started_at = Instant::now();

    let origins = cues(request.origins);
    if let Some(refusal) = overlong("origins", origins.len(), started_at) {
        return refusal;
    }
    let labels = request.labels.map(cues).unwrap_or_default();
    if let Some(refusal) = overlong("labels", labels.len(), started_at) {
        return refusal;
    }

    // The server-composed twin of `taguru.retrieve` (ADR 0008 §5, §6):
    // same root shape, one child span per `EvidenceLanesPlan` field.
    // Synchronous below (every lane's I/O is a `block_in_place`
    // closure, never a real `.await`), so holding the guard across
    // the whole handler is correct — matching `taguru.passage_search`
    // in `src/api/sources.rs`.
    let root = crate::trace::span!(
        "taguru.assemble_evidence",
        otel.kind = "internal",
        taguru.operation = "assemble_evidence",
        taguru.origin.count = origins.len() as i64,
        taguru.label.count = labels.len() as i64,
        taguru.anchor.count = tracing::field::Empty,
        taguru.association.count = tracing::field::Empty,
        taguru.citation.returned = tracing::field::Empty,
    );
    let _entered = root.enter();

    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // A single existence check up front, rather than leaning on
    // whichever lane happens to run first — `origins: []` is
    // degenerate-but-valid input (§8: budgets/inputs at their floor
    // are ordinary, not an error) and must not skip this check the
    // way an empty resolve loop otherwise would.
    let Some(current) = state.context_revision(&name) else {
        return not_found(&name, started_at);
    };

    let limits = BudgetLimits::resolve(request.budget);

    // The canonical query ADR 0006 §12 defines for the reranker
    // boundary, reused here as the passage/community lanes' own query
    // text: `text_fallback_query` verbatim when given, otherwise
    // `origins` joined with `"; "` in request order.
    let canonical_query = request
        .text_fallback_query
        .unwrap_or_else(|| origins.join("; "));

    // --- Step 1: resolve each origin cue, auto-picking the top
    // candidate into a deduplicated anchor list (retrieve's own Step
    // 1, `src/mcp/retrieve.rs:86-155`). `origins` is consumed here —
    // nothing below needs it again, `canonical_query` above already
    // read it.
    let origins_is_empty = origins.is_empty();
    let resolve_span = crate::trace::span!(
        "taguru.resolve",
        otel.kind = "internal",
        taguru.op = "resolve",
        taguru.origin.count = origins.len() as i64,
        taguru.anchor.count = tracing::field::Empty,
    );
    let mut anchors: Vec<String> = Vec::new();
    {
        let _guard = resolve_span.enter();
        for cue in origins {
            if deadline.expired() {
                return deadline_exceeded(started_at);
            }
            let resolve_request = ResolveRequest {
                cue,
                dice_floor: request.dice_floor,
                semantic_floor: request.semantic_floor,
                limit: request.resolve_limit,
            };
            let served = match resolve_served(
                &state,
                &name,
                &resolve_request,
                false,
                deadline,
                started_at,
            ) {
                Ok(served) => served,
                Err(response) => return response,
            };
            if let Some(top) = served.first()
                && !anchors.contains(&top.name)
            {
                anchors.push(top.name.clone());
            }
        }
        resolve_span.record("taguru.anchor.count", anchors.len() as i64);
    }
    let resolve_plan = if origins_is_empty {
        tracing::info!(taguru.reason = "origins_empty", "taguru.skip");
        LanePlan::skipped("origins was empty")
    } else {
        LanePlan::ran()
    };
    root.record("taguru.anchor.count", anchors.len() as i64);
    let anchor_refs: Vec<&str> = anchors.iter().map(String::as_str).collect();
    const NO_ANCHORS_REASON: &str = "no anchors resolved from 'origins'";

    let mut association_pool: Vec<EvidenceCandidate> = Vec::new();

    // --- Step 2: query — only when `labels` pins the facets (retrieve's
    // own Step 3a, `src/mcp/retrieve.rs:178-207`).
    let query_plan = if anchors.is_empty() {
        tracing::info!(taguru.reason = "no_anchors", "taguru.skip");
        LanePlan::skipped(NO_ANCHORS_REASON)
    } else if labels.is_empty() {
        tracing::info!(taguru.reason = "labels_absent", "taguru.skip");
        LanePlan::skipped("no 'labels' given")
    } else {
        if deadline.expired() {
            return deadline_exceeded(started_at);
        }
        let query_span = crate::trace::span!(
            "taguru.query",
            otel.kind = "internal",
            taguru.op = "query",
            taguru.anchor.count = anchors.len() as i64,
            taguru.association.count = tracing::field::Empty,
        );
        let _guard = query_span.enter();
        let label_refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        match state.read_context(&name, |context| {
            context.query_any(&anchor_refs, &label_refs, &[])
        }) {
            Ok(matches) => {
                let (total, matches) = page(matches, None, None);
                state.note_search(SearchOp::Query, &name, total == 0);
                let matches = associations_out(&state, &name, matches);
                for (rank, association) in matches.into_iter().enumerate() {
                    association_pool.push(EvidenceCandidate::from_association(
                        &name,
                        association,
                        rank + 1,
                    ));
                }
                query_span.record("taguru.association.count", association_pool.len() as i64);
                LanePlan::ran()
            }
            Err(failure) => return access_error(&state, failure, &name, started_at),
        }
    };

    // --- Step 3: activate — always, when there is at least one anchor
    // (retrieve's own Step 3, `src/mcp/retrieve.rs:207-233`).
    let activate_plan = if anchors.is_empty() {
        tracing::info!(taguru.reason = "no_anchors", "taguru.skip");
        LanePlan::skipped(NO_ANCHORS_REASON)
    } else {
        if deadline.expired() {
            return deadline_exceeded(started_at);
        }
        let activate_span = crate::trace::span!(
            "taguru.activate",
            otel.kind = "internal",
            taguru.op = "activate",
            taguru.anchor.count = anchors.len() as i64,
            taguru.activation.count = tracing::field::Empty,
        );
        let _guard = activate_span.enter();
        // ADR 0009 §6.3 exclusion 1, same as `POST /contexts/{name}/activate`
        // — resolved before `read_context`, per `AppState::hidden_label`'s
        // own doc. Its slow path is real disk I/O under a write lock,
        // so — like every other lane in this handler — it runs off the
        // async worker.
        let hidden = tokio::task::block_in_place(|| state.hidden_label(&name));
        let excluded: Vec<&str> = hidden.into_iter().collect();
        match state.read_context(&name, |context| {
            context.activate_excluding(
                &anchor_refs,
                request.activate_decay.unwrap_or(0.5),
                clamp(request.activate_limit, 20, MAX_MATCH_LIMIT),
                &excluded,
            )
        }) {
            Ok((total, matches)) => {
                state.note_search(SearchOp::Activate, &name, total == 0);
                let matches = activations_out(&state, &name, matches);
                activate_span.record("taguru.activation.count", matches.len() as i64);
                for (rank, activation) in matches.into_iter().enumerate() {
                    association_pool.push(EvidenceCandidate::from_activation(
                        &name,
                        activation,
                        rank + 1,
                    ));
                }
                LanePlan::ran()
            }
            Err(failure) => return access_error(&state, failure, &name, started_at),
        }
    };
    root.record("taguru.association.count", association_pool.len() as i64);

    // --- Step 4: passages — always, over the canonical query. Unlike
    // `retrieve`'s own text-fallback lane (opt-in, and only when
    // associations came back empty), this lane runs unconditionally so
    // a caller gets mixed graph/passage evidence by default.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let search_limit = clamp(request.search_limit, 5, MAX_EVIDENCE_SEARCH_LIMIT);
    let mut passage_candidates: Vec<EvidenceCandidate> = Vec::new();
    // Named `taguru.passages`, not `taguru.passage_search` — this lane
    // is unconditional here (unlike `sources/search`'s handler, which
    // owns that name), so a distinct name keeps "always ran" and "ran
    // as a fallback" tellable apart in a trace (ADR 0008 §5).
    let passages_span = crate::trace::span!(
        "taguru.passages",
        otel.kind = "internal",
        taguru.op = "search_passages",
        taguru.limit = search_limit as i64,
        taguru.passage.hit_count = tracing::field::Empty,
    );
    // Scoped tightly to Step 4 alone — Steps 5/6 below must not nest
    // under this span.
    let passages_plan;
    {
        let _guard = passages_span.enter();
        // A residency's first search tokenizes the whole corpus into
        // the index (`search_passages`'s own rule) — keep it off the
        // async worker, like every other passage-search entry.
        passages_plan = match tokio::task::block_in_place(|| {
            state.search_passages(
                &name,
                &canonical_query,
                search_limit,
                request.semantic_floor,
                None,
                deadline,
            )
        }) {
            None => return not_found(&name, started_at),
            // Logged, not discarded (issue #620): same reasoning as
            // `search_passages`'s own budget/io-error race.
            Some(Err(io_error)) if deadline.expired() || crate::api::injected_deadline_race() => {
                tracing::warn!(kind = ?io_error.kind(), "passage read failed under a spent budget");
                return deadline_exceeded(started_at);
            }
            Some(Err(io_error)) => {
                return crate::api::sources::passages_unreadable(&state, io_error, started_at);
            }
            Some(Ok(found)) => {
                state.note_search(SearchOp::SearchPassages, &name, found.hits.is_empty());
                passages_span.record("taguru.passage.hit_count", found.hits.len() as i64);
                for hit in &found.hits {
                    state
                        .metrics()
                        .record_passage_hit(hit.bm25.is_some(), hit.vector.is_some());
                }
                let plan = match &found.lanes {
                    PassageSearchLanes::NoQueryTerms => {
                        tracing::info!(taguru.reason = "no_query_terms", "taguru.skip");
                        LanePlan::skipped(NO_QUERY_TERMS_REASON)
                    }
                    PassageSearchLanes::ZeroLimit => {
                        tracing::info!(taguru.reason = "zero_limit", "taguru.skip");
                        LanePlan::skipped(ZERO_LIMIT_REASON)
                    }
                    PassageSearchLanes::Ran { .. } => LanePlan::ran(),
                };
                for (rank, hit) in found.hits.into_iter().enumerate() {
                    passage_candidates.push(EvidenceCandidate::from_passage(
                        &name,
                        PassageHit::from(hit),
                        rank + 1,
                    ));
                }
                plan
            }
        };
    }

    // --- Step 5: communities — opt-in, over the same canonical query
    // (ADR 0006 §6: `include_communities`, default false). A missing
    // artifact is a degrade here, never the refusal
    // `communities/search` itself gives (ADR 0006 §11): community
    // evidence is one opt-in input among several, not the entire point
    // of this call.
    let mut community_candidates: Vec<EvidenceCandidate> = Vec::new();
    let communities_plan = if !request.include_communities {
        tracing::info!(taguru.reason = "communities_disabled", "taguru.skip");
        LanePlan::skipped("include_communities was false")
    } else {
        let derived = derived_context_name(&name);
        if let Some(refusal) = check_derived_scope(&scope, &name, &derived, started_at) {
            return refusal;
        }
        if deadline.expired() {
            return deadline_exceeded(started_at);
        }
        // Same shape as the sibling `taguru.passages` (issue #697):
        // op, limit, and a hit count — the hits ARE passages of the
        // derived artifact context, so the count reuses
        // `taguru.passage.hit_count` rather than minting a
        // communities-only spelling.
        let communities_span = crate::trace::span!(
            "taguru.communities",
            otel.kind = "internal",
            taguru.op = "search_communities",
            taguru.limit = search_limit as i64,
            taguru.passage.hit_count = tracing::field::Empty,
        );
        let _guard = communities_span.enter();
        match community_hits(
            &state,
            &name,
            &derived,
            &canonical_query,
            search_limit,
            request.semantic_floor,
            current.graph,
            deadline,
            started_at,
        ) {
            Ok(CommunityLaneOutcome::Found(found)) => {
                communities_span.record("taguru.passage.hit_count", found.hits.len() as i64);
                for (rank, hit) in found.hits.into_iter().enumerate() {
                    community_candidates.push(EvidenceCandidate::from_community(
                        &name,
                        hit,
                        rank + 1,
                    ));
                }
                LanePlan::ran()
            }
            Ok(CommunityLaneOutcome::NoArtifact(reason)) => {
                // Every other skip arm in this function emits its
                // event (issue #697); a fixed code, never `reason` —
                // that text names the context (forbidden on any
                // span/event, ADR 0008 §8) and belongs to the
                // response's plan alone.
                tracing::info!(taguru.reason = "no_communities_artifact", "taguru.skip");
                LanePlan::skipped(reason)
            }
            // A real failure (malformed manifest, IO error, deadline,
            // access) — not the "no artifact" degrade — aborts the
            // whole call, the same severity a direct
            // `communities/search` call would answer with.
            Err(response) => return response,
        }
    };

    // --- Step 6: citations — every attribution locator any association
    // candidate carries, deduplicated by (source, paragraph) (retrieve's
    // own Step 4, `src/mcp/retrieve.rs:238-280`). Unlike `retrieve`,
    // this step has no opt-out — #216's evidence package is citation-
    // complete by construction (ADR 0006 §6), not by a caller
    // remembering to ask for it.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let mut wanted: BTreeSet<(String, u32)> = BTreeSet::new();
    for candidate in &association_pool {
        wanted.extend(candidate.origins.iter().cloned());
    }
    let citations_span = crate::trace::span!(
        "taguru.citations",
        otel.kind = "internal",
        taguru.citation.requested = wanted.len() as i64,
        taguru.citation.returned = tracing::field::Empty,
    );
    let citation_lookup;
    {
        let _guard = citations_span.enter();
        // Same cold-load path as the direct `citation` endpoint; one
        // block_in_place around every locator this call needs, not one
        // per locator.
        #[allow(clippy::result_large_err)] // the Err IS the response served next
        {
            citation_lookup = match tokio::task::block_in_place(|| {
                resolve_citations(&state, &name, wanted, started_at)
            }) {
                Ok(citation_lookup) => citation_lookup,
                Err(response) => return response,
            };
        }
        citations_span.record("taguru.citation.returned", citation_lookup.len() as i64);
        root.record("taguru.citation.returned", citation_lookup.len() as i64);
    }
    let citations_plan = LanePlan::ran();

    // --- Fuse and select (#303 §7, #304 §8-§9), with an optional
    // reorder pass (#307, ADR 0006 §12) applied by `select_with_reorder`
    // itself, immediately before diversity-aware admission. `rerank`
    // absent is the overwhelmingly common case and costs nothing beyond
    // this one `is_none()` check — no provider touched, no `block_in_place`,
    // no metrics.
    let mut pool = association_pool;
    pool.extend(passage_candidates);
    pool.extend(community_candidates);
    let (fused, dedup_dropped) = fuse(pool);

    let reranker_configured = state.reranker().is_some();
    let (selected, reranker) = match &request.rerank {
        None => (
            select::select(fused, dedup_dropped, &limits, &citation_lookup),
            RerankerPlan::not_requested(reranker_configured),
        ),
        Some(rerank_request) => {
            let mut reranker_plan = None;
            let mut reorder = |survivors: &[FusedCandidate]| {
                let (order, plan, outcome) = tokio::task::block_in_place(|| {
                    rerank::drive(
                        state.reranker().as_deref(),
                        rerank_request,
                        &canonical_query,
                        survivors,
                        deadline,
                    )
                });
                state.record_rerank(outcome);
                reranker_plan = Some(plan);
                order
            };
            let selected = select_with_reorder(
                fused,
                dedup_dropped,
                &limits,
                &citation_lookup,
                Some(&mut reorder),
            );
            let reranker = reranker_plan.unwrap_or(RerankerPlan {
                configured: reranker_configured,
                ran: false,
                model: None,
                reason: None,
            });
            (selected, reranker)
        }
    };

    let package = EvidencePackage {
        items: selected.items,
        citations: selected.citations,
        budget: selected.budget,
        omitted: selected.omitted,
        omitted_total: selected.omitted_total,
        omitted_by_reason: selected.omitted_by_reason,
        plan: EvidencePlan {
            lanes: EvidenceLanesPlan {
                resolve: resolve_plan,
                query: query_plan,
                activate: activate_plan,
                passages: passages_plan,
                communities: communities_plan,
                citations: citations_plan,
            },
            selection: selected.selection,
            reranker,
        },
    };

    ok(package, started_at)
}

/// Step 6's citation resolution, split out so the whole locator set
/// runs inside one `block_in_place` call rather than one per locator
/// — an early `Err` here is the whole `assemble_evidence` call's own
/// failure, matching what the direct `citation` endpoint would answer
/// for the same (source, paragraph).
#[allow(clippy::result_large_err)] // the Err IS the response served next
fn resolve_citations(
    state: &AppState,
    name: &str,
    wanted: BTreeSet<(String, u32)>,
    started_at: Instant,
) -> Result<HashMap<(String, u32), Citation>, Response> {
    let mut citation_lookup = HashMap::new();
    for (source, paragraph) in wanted {
        match state.citation(name, &source, paragraph) {
            None => return Err(not_found(name, started_at)),
            Some(Err(io_error)) => {
                return Err(crate::api::sources::passages_unreadable(
                    state, io_error, started_at,
                ));
            }
            // A located attribution whose passage was never stored (or
            // was retracted) is skipped, exactly like `retrieve`'s own
            // citation loop — the graph fact still stands.
            Some(Ok(crate::registry::CitationLookup::UnknownSource))
            | Some(Ok(crate::registry::CitationLookup::IndexOutOfRange)) => {}
            Some(Ok(crate::registry::CitationLookup::Found {
                text,
                section,
                locator,
            })) => {
                citation_lookup.insert(
                    (source.clone(), paragraph),
                    Citation {
                        text,
                        source,
                        section,
                        locator,
                    },
                );
            }
        }
    }
    Ok(citation_lookup)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ContextMeta;

    fn scratch_state(tag: &str) -> (AppState, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "taguru-api-evidence-assemble-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        (state, dir)
    }

    /// Forces `context`'s next passage read to fail with a genuine
    /// `io::Error` — the same trick `api::sources`'s own io-error
    /// tests use (issue #620): a snapshot file `PassageStore::load`
    /// cannot parse, written before the context's first passage touch.
    fn corrupt_passages_snapshot(dir: &std::path::Path, context: &str) {
        let stem = crate::registry::file_stem(context);
        let path = crate::registry::passages_path(dir, &stem);
        std::fs::write(path, b"not a valid passages snapshot").unwrap();
    }

    /// A deadline that has already elapsed by the time it is checked —
    /// mirrors `api::recall`/`api::groups`'s own `already_expired_deadline`.
    fn already_expired_deadline() -> Deadline {
        let deadline = Deadline::after(std::time::Duration::ZERO);
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(deadline.expired(), "a zero budget must read as expired");
        deadline
    }

    fn minimal_request() -> AssembleEvidenceRequest {
        AssembleEvidenceRequest {
            origins: OneOrMany::Many(Vec::new()),
            labels: None,
            dice_floor: None,
            semantic_floor: None,
            resolve_limit: None,
            activate_decay: None,
            activate_limit: None,
            // Non-empty on purpose: an empty query short-circuits
            // `search_passages` before it ever touches the passage
            // store, which would never reach the corrupted snapshot.
            text_fallback_query: Some("AAA".to_string()),
            search_limit: None,
            include_communities: false,
            budget: None,
            rerank: None,
        }
    }

    /// issue #620 (所見3): `assemble_evidence`'s own twin of
    /// `search_passages`'s io-error/deadline race tests.
    #[tokio::test(flavor = "multi_thread")]
    async fn assemble_evidence_reports_a_genuine_io_error_as_unreadable_not_timeout() {
        let (state, dir) = scratch_state("io-error");
        state.create("sake", ContextMeta::default()).unwrap();
        corrupt_passages_snapshot(&dir, "sake");

        let response = assemble_evidence(
            State(state),
            AppPath("sake".to_string()),
            None,
            axum::Extension(Deadline::unbounded()),
            AppJson(minimal_request()),
        )
        .await;

        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            body["code"],
            crate::api::ErrorCode::Internal.as_str(),
            "an unexpired deadline must never reclassify a real disk fault as a \
             timeout — {body}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn assemble_evidence_reclassifies_an_io_error_as_timeout_once_the_budget_is_spent() {
        let (state, dir) = scratch_state("io-error-timeout");
        state.create("sake", ContextMeta::default()).unwrap();
        corrupt_passages_snapshot(&dir, "sake");
        crate::api::expire_deadline_race();

        let response = assemble_evidence(
            State(state),
            AppPath("sake".to_string()),
            None,
            axum::Extension(Deadline::unbounded()),
            AppJson(minimal_request()),
        )
        .await;

        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            body["code"],
            crate::api::ErrorCode::Timeout.as_str(),
            "a budget spent by the time the read failed must reclassify as a \
             timeout — {body}"
        );
    }

    /// The front-of-handler deadline gate (line 190, before `resolve`
    /// ever runs): none of the seven `deadline.expired()` early returns
    /// in this handler had a direct test — the two tests above only
    /// exercise `expire_deadline_race`'s io-error reclassification,
    /// a different mechanism entirely. Naming a context that does not
    /// exist makes the ordering observable without a fault-injection
    /// hook: if this gate ran only after the context-existence check
    /// (the pre-fix order for a bug of this shape elsewhere in the
    /// tree, #620), this would answer 404 `no_context` instead of the
    /// timeout asserted below.
    #[tokio::test(flavor = "multi_thread")]
    async fn assemble_evidence_refuses_an_already_expired_deadline_before_any_lane_runs() {
        let (state, _dir) = scratch_state("expired-deadline");
        let deadline = already_expired_deadline();

        let response = assemble_evidence(
            State(state),
            AppPath("ghost".to_string()),
            None,
            axum::Extension(deadline),
            AppJson(minimal_request()),
        )
        .await;

        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            body["code"],
            crate::api::ErrorCode::Timeout.as_str(),
            "an already-expired deadline must be refused before the \
             context-existence check ever runs — {body}"
        );
    }
}
