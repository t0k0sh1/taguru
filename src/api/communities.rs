//! The community verbs (issue #166). Two surfaces with one artifact
//! between them:
//!
//! - `GET /contexts/{name}/communities` — detection on the live graph
//!   ([`taguru::context::Context::communities`]), streamed as JSON
//!   Lines like the export: a header line carrying the revision
//!   snapshot the analysis was cut at, then one line per community.
//!   This is the derivation half `taguru communities` orchestrates —
//!   compute-heavy, so it rides the heavy-ops gate beside compact and
//!   the vocabulary audit, and uncached (an analysis dump would evict
//!   the retrieval cache's working set for one offline caller).
//! - `POST /contexts/{name}/communities/search` — the global-search
//!   surface over a previously derived artifact: ranked community
//!   summaries with membership, and an honest staleness verdict. The
//!   artifact is an ordinary context (default `{name}::communities`),
//!   so the ranking IS `search_passages` against it — both lanes,
//!   floors, plan and all — and the response rides the retrieval
//!   cache under its own op.
//!
//! The revision-before-analysis order in the compute verb is
//! deliberate and mirrors `retrieval_key`'s key-before-search rule: a
//! write landing between the two leaves the recorded revision OLDER
//! than the analyzed graph, so the derived artifact can only err
//! toward reporting itself stale — never toward claiming freshness it
//! does not have.

use std::collections::BTreeMap;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};

use taguru::context::CommunityAnalysis;
use taguru::deadline::Deadline;

use crate::metrics::{RetrievalCacheOp, SearchOp};
use crate::registry::{AppState, ContextRevision};

use super::sources::{SearchContextPlan, SearchPlan, passages_unreadable};
use super::{
    AppJson, AppPath, ErrorCode, MAX_MATCH_LIMIT, access_error, cache_and_serve, clamp,
    deadline_exceeded, error, not_found, ok, replay_cached_search, search_log_enabled,
};

/// The derived context a source context's artifact lives in by
/// default. `search` accepts an override for artifacts built with
/// `taguru communities --into`.
pub(crate) fn derived_context_name(source: &str) -> String {
    format!("{source}::communities")
}

/// The reserved source id carrying the derivation record inside a
/// derived context.
pub(crate) const MANIFEST_SOURCE: &str = "communities:manifest";

/// Every community's summary is stored under `community:{id}`.
pub(crate) const COMMUNITY_SOURCE_PREFIX: &str = "community:";

/// Membership edges in a derived context: `community:{id} —contains→
/// member concept`, weighted by the member's intra-community strength.
pub(crate) const CONTAINS_LABEL: &str = "contains";

/// Hierarchy edges in a derived context: `community:{parent}
/// —includes→ community:{child}`.
pub(crate) const INCLUDES_LABEL: &str = "includes";

/// Member concepts served per hit — a digest for the reading LLM; the
/// full membership stays queryable on the artifact itself.
const MEMBERS_PER_HIT: usize = 12;

/// The derivation record — the passage text of [`MANIFEST_SOURCE`],
/// written by `taguru communities`, read back by `search` for the
/// staleness verdict and per-hit community facts. The version key
/// follows the batch/group convention (`taguru_batch`,
/// `taguru_group`).
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CommunitiesManifest {
    pub taguru_communities: u64,
    pub algorithm: String,
    /// The context the artifact was derived from — `search` refuses a
    /// `derived` override pointing at another context's artifact.
    pub source_context: String,
    /// The source context's revision at derivation time (snapshotted
    /// BEFORE the analysis — see the module doc for why that order).
    pub revision: ContextRevision,
    pub levels: usize,
    pub communities: Vec<ManifestCommunity>,
}

/// One community's identity in the manifest: what `search` needs per
/// hit, and what the next derivation run diffs fingerprints against.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ManifestCommunity {
    pub id: String,
    pub level: usize,
    pub fingerprint: String,
    pub concept_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
}

/// Wire version of the manifest and of the analysis stream's header
/// line.
pub(crate) const COMMUNITIES_FORMAT: u64 = 1;

/// `GET /contexts/{name}/communities` — the analysis stream. The body
/// is JSON Lines, not the JSON envelope (the export's rule): a header
/// object first, then one line per community, leaves first.
pub async fn analyze_communities(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
) -> Response {
    let started_at = Instant::now();
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Revision BEFORE analysis — the safe-staleness order (module doc).
    let Some(revision) = state.context_revision(&name) else {
        return not_found(&name, started_at);
    };
    // Off the async worker: detection sweeps every edge, and a cold
    // context loads first — the audit endpoints' rule.
    let outcome = tokio::task::block_in_place(|| {
        state.read_context(&name, |context| context.communities(deadline))
    });
    match outcome {
        Err(failure) => access_error(&state, failure, &name, started_at),
        Ok(Err(_)) => deadline_exceeded(started_at),
        Ok(Ok(analysis)) => match render_analysis(&name, revision, &analysis) {
            Some(body) => (
                StatusCode::OK,
                [(
                    axum::http::header::CONTENT_TYPE,
                    "application/x-ndjson; charset=utf-8",
                )],
                body,
            )
                .into_response(),
            // Unreachable for these types (no non-string keys, no
            // non-finite floats survive the weight cap) — refuse
            // loudly rather than stream half a body.
            None => error(
                ErrorCode::Internal,
                "the analysis did not serialize",
                started_at,
            ),
        },
    }
}

/// The analysis stream's body: header line, then one community per
/// line. `None` only if serialization itself refuses.
fn render_analysis(
    name: &str,
    revision: ContextRevision,
    analysis: &CommunityAnalysis,
) -> Option<String> {
    #[derive(Serialize)]
    struct Header<'a> {
        taguru_communities: u64,
        context: &'a str,
        algorithm: &'a str,
        revision: ContextRevision,
        concept_count: usize,
        edge_count: usize,
        levels: usize,
        communities: usize,
    }
    let mut body = serde_json::to_string(&Header {
        taguru_communities: COMMUNITIES_FORMAT,
        context: name,
        algorithm: analysis.algorithm,
        revision,
        concept_count: analysis.concept_count,
        edge_count: analysis.edge_count,
        levels: analysis.levels,
        communities: analysis.communities.len(),
    })
    .ok()?;
    body.push('\n');
    for community in &analysis.communities {
        body.push_str(&serde_json::to_string(community).ok()?);
        body.push('\n');
    }
    Some(body)
}

#[derive(Debug, Deserialize)]
pub struct SearchCommunitiesRequest {
    pub query: String,
    /// Omitted means 5, `search_passages`' default.
    pub limit: Option<usize>,
    /// One-call override of the artifact's semantic-lane floor — the
    /// same knob `search_passages` takes, applied to the same search.
    pub semantic_floor: Option<f32>,
    /// The artifact context to search; omitted means
    /// `{name}::communities`. For artifacts built with `--into`.
    pub derived: Option<String>,
}

/// [`search_communities`]' result: the staleness verdict beside the
/// ranked summaries it qualifies.
#[derive(Serialize, Deserialize)]
pub struct CommunityPage {
    /// The artifact context that answered.
    pub derived: String,
    /// The algorithm that built it — comparable against the analysis
    /// verb's current one.
    pub algorithm: String,
    /// True when the source context's graph moved since derivation —
    /// the summaries describe an older graph; re-run
    /// `taguru communities` to refresh them.
    pub stale: bool,
    pub revision: CommunityRevisions,
    pub plan: SearchPlan,
    pub hits: Vec<CommunityHit>,
}

/// The two graph revisions the staleness verdict compares.
#[derive(Serialize, Deserialize)]
pub struct CommunityRevisions {
    pub recorded_graph: u64,
    pub current_graph: u64,
}

/// One ranked community: the matched summary paragraph plus the
/// community's manifest facts and its strongest members. Manifest
/// fields are absent when the artifact is mid-rewrite and the summary
/// landed before its manifest line — served honestly rather than
/// dropped.
#[derive(Serialize, Deserialize)]
pub struct CommunityHit {
    /// Community id (`L0-3`) — its summary source is
    /// `community:L0-3` on the artifact.
    pub community: String,
    pub score: f32,
    /// The matched paragraph of the summary; the whole summary is one
    /// `sources/lookup` away on the artifact.
    pub text: String,
    pub paragraph: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concept_count: Option<usize>,
    /// Strongest member concepts (leaf communities; parents carry
    /// their children as `includes` edges on the artifact instead).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub members: Vec<CommunityHitMember>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub members_truncated: bool,
}

#[derive(Serialize, Deserialize)]
pub struct CommunityHitMember {
    pub name: String,
    pub strength: f64,
}

pub async fn search_communities(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<SearchCommunitiesRequest>,
) -> Response {
    let started_at = Instant::now();
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let limit = clamp(request.limit, 5, MAX_MATCH_LIMIT);
    let derived = request
        .derived
        .clone()
        .unwrap_or_else(|| derived_context_name(&name));
    // The auth middleware checked the PATH context; `derived` is a
    // second read target and gets the same per-context grant check —
    // otherwise a scoped key could read any context by naming it here.
    if let Some(axum::Extension(scope)) = &scope
        && !scope.allows_context(&derived)
    {
        return error(
            ErrorCode::Forbidden,
            format!(
                "this key's scope does not extend to the artifact context \
                 '{derived}' — grant it alongside '{name}'"
            ),
            started_at,
        );
    }
    // The source context anchors the staleness verdict; its absence is
    // this verb's 404 whatever the artifact holds.
    let Some(current) = state.context_revision(&name) else {
        return not_found(&name, started_at);
    };
    // Minted before manifest read and search — see `retrieval_key`.
    // The artifact rides the key as a target (passages/config lanes);
    // the SOURCE graph revision rides the params, so a source write
    // re-keys the staleness verdict the cached payload states.
    let key = state.retrieval_key(
        RetrievalCacheOp::SearchCommunities,
        &[derived.clone(), name.clone()],
        serde_json::to_string(&(
            "search_communities",
            &request.query,
            limit,
            request.semantic_floor,
            current.graph,
        ))
        .ok(),
    );
    if let Some(key) = &key
        && let Some(found) = state.retrieval_lookup(key)
    {
        replay_cached_search(&state, key, &found);
        if search_log_enabled() {
            tracing::info!(
                target: "taguru::search",
                context = %name,
                op = "search_communities",
                cue = %request.query,
                hits = found.log_hits,
                top_score = f64::from(found.log_top_score),
                cached = true,
                "search",
            );
        }
        return ok(found.payload.as_ref(), started_at);
    }

    // The manifest is the artifact's identity: no artifact context, or
    // an artifact without its record, both answer "build one" rather
    // than an empty result — absence of analysis is not an empty
    // corpus.
    let manifest = tokio::task::block_in_place(|| {
        state.lookup_passages(&derived, std::slice::from_ref(&MANIFEST_SOURCE.to_string()))
    });
    let manifest = match manifest {
        None => {
            return error(
                ErrorCode::NoContext,
                format!(
                    "no communities artifact for '{name}': context '{derived}' does not \
                     exist — run `taguru communities` to build it"
                ),
                started_at,
            );
        }
        Some(Err(io_error)) => return passages_unreadable(&state, io_error, started_at),
        Some(Ok((mut passages, _missing))) => match passages.remove(MANIFEST_SOURCE) {
            None => {
                return error(
                    ErrorCode::NoSource,
                    format!(
                        "context '{derived}' holds no '{MANIFEST_SOURCE}' record — run \
                         `taguru communities` to (re)build the artifact"
                    ),
                    started_at,
                );
            }
            Some(text) => match serde_json::from_str::<CommunitiesManifest>(&text) {
                Err(parse_error) => {
                    return error(
                        ErrorCode::Conflict,
                        format!(
                            "the '{MANIFEST_SOURCE}' record in '{derived}' does not parse \
                             ({parse_error}) — rebuild the artifact with `taguru communities`"
                        ),
                        started_at,
                    );
                }
                Ok(manifest) => manifest,
            },
        },
    };
    if manifest.source_context != name {
        return error(
            ErrorCode::Conflict,
            format!(
                "artifact '{derived}' was derived from '{}', not '{name}'",
                manifest.source_context
            ),
            started_at,
        );
    }
    let stale = manifest.revision.graph != current.graph;
    let facts: BTreeMap<&str, &ManifestCommunity> = manifest
        .communities
        .iter()
        .map(|community| (community.id.as_str(), community))
        .collect();

    // The ranking IS search_passages against the artifact — one extra
    // slot absorbs the manifest source ranking for the query, which is
    // filtered out below. No source filter: the artifact's sources are
    // synthetic `community:{id}` rows that carry no user metadata, so
    // a filter here could only ever exclude everything.
    let outcome = tokio::task::block_in_place(|| {
        state.search_passages(
            &derived,
            &request.query,
            limit + 1,
            request.semantic_floor,
            None,
            deadline,
        )
    });
    let found = match outcome {
        None => {
            return error(
                ErrorCode::NoContext,
                format!(
                    "no communities artifact for '{name}': context '{derived}' does not \
                     exist — run `taguru communities` to build it"
                ),
                started_at,
            );
        }
        Some(Err(_)) if deadline.expired() => return deadline_exceeded(started_at),
        Some(Err(io_error)) => return passages_unreadable(&state, io_error, started_at),
        Some(Ok(found)) => found,
    };

    // One passage hit kept for the response, pre-join: its community
    // id, the ranking evidence, and which lanes surfaced it.
    struct RankedSummary {
        id: String,
        score: f32,
        text: String,
        paragraph: u32,
        bm25: bool,
        vector: bool,
    }
    let mut lane_hits = [0u64; 3];
    let mut ranked: Vec<RankedSummary> = Vec::new();
    for hit in &found.hits {
        let Some(id) = hit.source.strip_prefix(COMMUNITY_SOURCE_PREFIX) else {
            continue;
        };
        ranked.push(RankedSummary {
            id: id.to_string(),
            score: hit.score,
            text: hit.text.clone(),
            paragraph: hit.index,
            bm25: hit.bm25.is_some(),
            vector: hit.vector.is_some(),
        });
    }
    ranked.truncate(limit);
    for summary in &ranked {
        state
            .metrics()
            .record_passage_hit(summary.bm25, summary.vector);
        match (summary.bm25, summary.vector) {
            (true, false) => lane_hits[0] += 1,
            (true, true) => lane_hits[1] += 1,
            (false, true) => lane_hits[2] += 1,
            (false, false) => {}
        }
    }

    // Membership for the served hits, straight off the artifact's own
    // graph — `contains` edges, strongest members first.
    let members = tokio::task::block_in_place(|| {
        state.read_context(&derived, |context| {
            ranked
                .iter()
                .map(|summary| {
                    let subject = format!("{COMMUNITY_SOURCE_PREFIX}{}", summary.id);
                    let mut all: Vec<(String, f64)> = context
                        .query(Some(&subject), Some(CONTAINS_LABEL), None)
                        .into_iter()
                        .map(|association| (association.object, association.weight))
                        .collect();
                    all.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()).then_with(|| a.0.cmp(&b.0)));
                    let truncated = all.len() > MEMBERS_PER_HIT;
                    all.truncate(MEMBERS_PER_HIT);
                    (all, truncated)
                })
                .collect::<Vec<_>>()
        })
    });
    let members = match members {
        Ok(members) => members,
        Err(failure) => return access_error(&state, failure, &derived, started_at),
    };

    let hits: Vec<CommunityHit> = ranked
        .into_iter()
        .zip(members)
        .map(|(summary, (members, members_truncated))| {
            let fact = facts.get(summary.id.as_str());
            CommunityHit {
                level: fact.map(|fact| fact.level),
                parent: fact.and_then(|fact| fact.parent.clone()),
                concept_count: fact.map(|fact| fact.concept_count),
                community: summary.id,
                score: summary.score,
                text: summary.text,
                paragraph: summary.paragraph,
                members: members
                    .into_iter()
                    .map(|(name, strength)| CommunityHitMember { name, strength })
                    .collect(),
                members_truncated,
            }
        })
        .collect();

    let empty = hits.is_empty();
    state.note_search(SearchOp::SearchCommunities, &derived, empty);
    state.note_search(SearchOp::SearchCommunities, &name, empty);
    if search_log_enabled() {
        tracing::info!(
            target: "taguru::search",
            context = %name,
            op = "search_communities",
            cue = %request.query,
            hits = hits.len(),
            top_score = hits.first().map_or(0.0, |hit| f64::from(hit.score)),
            stale,
            "search",
        );
    }
    // A transiently degraded semantic lane must not be pinned — the
    // same rule as search_passages.
    let key = key.filter(|_| !found.lanes.embedding_failed());
    let payload = CommunityPage {
        derived,
        algorithm: manifest.algorithm,
        stale,
        revision: CommunityRevisions {
            recorded_graph: manifest.revision.graph,
            current_graph: current.graph,
        },
        // The plan entry names the SOURCE context — the caller asked
        // about `{name}`; which artifact answered is `derived`'s job.
        plan: SearchPlan {
            contexts: vec![SearchContextPlan::of(&name, &found.lanes, None)],
        },
        hits,
    };
    let top_score = payload.hits.first().map_or(0.0, |hit| hit.score);
    let log_hits = payload.hits.len();
    cache_and_serve(
        &state,
        key,
        &payload,
        vec![empty, empty],
        lane_hits,
        log_hits,
        top_score,
        None,
        started_at,
    )
}
