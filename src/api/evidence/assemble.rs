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
    let mut anchors: Vec<String> = Vec::new();
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
        let served =
            match resolve_served(&state, &name, &resolve_request, false, deadline, started_at) {
                Ok(served) => served,
                Err(response) => return response,
            };
        if let Some(top) = served.first()
            && !anchors.contains(&top.name)
        {
            anchors.push(top.name.clone());
        }
    }
    let resolve_plan = if origins_is_empty {
        LanePlan::skipped("origins was empty")
    } else {
        LanePlan::ran()
    };
    let anchor_refs: Vec<&str> = anchors.iter().map(String::as_str).collect();
    const NO_ANCHORS_REASON: &str = "no anchors resolved from 'origins'";

    let mut association_pool: Vec<EvidenceCandidate> = Vec::new();

    // --- Step 2: query — only when `labels` pins the facets (retrieve's
    // own Step 3a, `src/mcp/retrieve.rs:178-207`).
    let query_plan = if anchors.is_empty() {
        LanePlan::skipped(NO_ANCHORS_REASON)
    } else if labels.is_empty() {
        LanePlan::skipped("no 'labels' given")
    } else {
        if deadline.expired() {
            return deadline_exceeded(started_at);
        }
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
                LanePlan::ran()
            }
            Err(failure) => return access_error(&state, failure, &name, started_at),
        }
    };

    // --- Step 3: activate — always, when there is at least one anchor
    // (retrieve's own Step 3, `src/mcp/retrieve.rs:207-233`).
    let activate_plan = if anchors.is_empty() {
        LanePlan::skipped(NO_ANCHORS_REASON)
    } else {
        if deadline.expired() {
            return deadline_exceeded(started_at);
        }
        match state.read_context(&name, |context| {
            context.activate(
                &anchor_refs,
                request.activate_decay.unwrap_or(0.5),
                clamp(request.activate_limit, 20, MAX_MATCH_LIMIT),
            )
        }) {
            Ok((total, matches)) => {
                state.note_search(SearchOp::Activate, &name, total == 0);
                let matches = activations_out(&state, &name, matches);
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

    // --- Step 4: passages — always, over the canonical query. Unlike
    // `retrieve`'s own text-fallback lane (opt-in, and only when
    // associations came back empty), this lane runs unconditionally so
    // a caller gets mixed graph/passage evidence by default.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let search_limit = clamp(request.search_limit, 5, MAX_EVIDENCE_SEARCH_LIMIT);
    let mut passage_candidates: Vec<EvidenceCandidate> = Vec::new();
    // A residency's first search tokenizes the whole corpus into the
    // index (`search_passages`'s own rule) — keep it off the async
    // worker, like every other passage-search entry.
    let passages_plan = match tokio::task::block_in_place(|| {
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
        Some(Err(_)) if deadline.expired() => return deadline_exceeded(started_at),
        Some(Err(io_error)) => {
            return crate::api::sources::passages_unreadable(&state, io_error, started_at);
        }
        Some(Ok(found)) => {
            state.note_search(SearchOp::SearchPassages, &name, found.hits.is_empty());
            for hit in &found.hits {
                state
                    .metrics()
                    .record_passage_hit(hit.bm25.is_some(), hit.vector.is_some());
            }
            let plan = match &found.lanes {
                PassageSearchLanes::NoQueryTerms => LanePlan::skipped(NO_QUERY_TERMS_REASON),
                PassageSearchLanes::ZeroLimit => LanePlan::skipped(ZERO_LIMIT_REASON),
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

    // --- Step 5: communities — opt-in, over the same canonical query
    // (ADR 0006 §6: `include_communities`, default false). A missing
    // artifact is a degrade here, never the refusal
    // `communities/search` itself gives (ADR 0006 §11): community
    // evidence is one opt-in input among several, not the entire point
    // of this call.
    let mut community_candidates: Vec<EvidenceCandidate> = Vec::new();
    let communities_plan = if !request.include_communities {
        LanePlan::skipped("include_communities was false")
    } else {
        let derived = derived_context_name(&name);
        if let Some(refusal) = check_derived_scope(&scope, &name, &derived, started_at) {
            return refusal;
        }
        if deadline.expired() {
            return deadline_exceeded(started_at);
        }
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
                for (rank, hit) in found.hits.into_iter().enumerate() {
                    community_candidates.push(EvidenceCandidate::from_community(
                        &name,
                        hit,
                        rank + 1,
                    ));
                }
                LanePlan::ran()
            }
            Ok(CommunityLaneOutcome::NoArtifact(reason)) => LanePlan::skipped(reason),
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
    // Same cold-load path as the direct `citation` endpoint; one
    // block_in_place around every locator this call needs, not one
    // per locator.
    #[allow(clippy::result_large_err)] // the Err IS the response served next
    let citation_lookup = match tokio::task::block_in_place(|| {
        resolve_citations(&state, &name, wanted, started_at)
    }) {
        Ok(citation_lookup) => citation_lookup,
        Err(response) => return response,
    };
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
            Some(Ok(crate::registry::CitationLookup::Found(text, section))) => {
                citation_lookup.insert(
                    (source.clone(), paragraph),
                    Citation {
                        text,
                        source,
                        section,
                    },
                );
            }
        }
    }
    Ok(citation_lookup)
}
