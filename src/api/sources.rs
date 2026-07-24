use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::Response;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use taguru::deadline::Deadline;

use crate::metrics::{ErrorKind, RetrievalCacheOp, SearchOp};
use crate::registry::{AppState, CitationLookup, PassageExplainLookup, SemanticFill};

use super::aliases::KeysetQuery;
use super::recall::cross_targets;
use super::{
    AppJson, AppPath, AppQuery, CrossMatch, ErrorCode, Issue, MAX_MATCH_LIMIT, MAX_NAME_BYTES,
    MAX_PASSAGES_PER_REQUEST, MAX_QUESTION_BYTES, MAX_QUESTIONS_PER_PARAGRAPH, MAX_SECTION_BYTES,
    MAX_TAG_BYTES, MAX_TAGS_PER_SOURCE, RefusalDetail, access_error, bounded_parallel_map,
    cache_and_serve, clamp, collected_validation_message, cross_search_concurrency,
    deadline_exceeded, empty, error, not_found, ok, overlong, oversized, replay_cached_search,
    search_log_enabled, truncate_issues, validation_error,
};

#[derive(Debug, Deserialize)]
pub struct LookupPassagesRequest {
    pub sources: Vec<String>,
}

/// The dereference half of "find with the graph, answer from the text":
/// attributions name sources, this returns the original passages behind
/// them (and which sources have none registered).
#[derive(Serialize)]
pub struct PassageLookup {
    pub passages: BTreeMap<String, String>,
    pub missing: Vec<String>,
}

pub async fn lookup_passages(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<LookupPassagesRequest>,
) -> Response {
    let started_at = Instant::now();
    // Each requested source returns its whole passage: the response
    // scales with this list, so the list itself is what gets bounded.
    if let Some(refusal) = overlong("sources", request.sources.len(), started_at) {
        return refusal;
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // A residency's first passage access loads the store from disk
    // (sources.json/passages.bin/WAL replay); keep that off the async
    // worker like every other passage-search entry.
    match tokio::task::block_in_place(|| state.lookup_passages(&name, &request.sources)) {
        None => not_found(&name, started_at),
        Some(Ok((passages, missing))) => {
            state.note_read(&name, passages.is_empty());
            ok(PassageLookup { passages, missing }, started_at)
        }
        Some(Err(io_error)) => passages_unreadable(&state, io_error, started_at),
    }
}

#[derive(Debug, Deserialize)]
pub struct CitationRequest {
    pub source: String,
    /// `index` is the pre-#35 name; still accepted so direct HTTP callers
    /// who haven't migrated aren't broken by the rename.
    #[serde(alias = "index")]
    pub paragraph: u32,
}

/// One located, verbatim excerpt: the citation counterpart of
/// `PassageLookup`'s whole-document dereference — text plus exactly
/// enough provenance to attribute it. `section` is the label governing
/// this paragraph (see `PassageRecord::section_for`), `null` when the
/// paragraph falls outside every section the source has stored, or when
/// it stored none at all; the key is never omitted, so callers can
/// rely on it always being present.
#[derive(Serialize)]
pub struct Citation {
    pub text: String,
    pub source: String,
    pub section: Option<String>,
}

pub async fn citation(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<CitationRequest>,
) -> Response {
    let started_at = Instant::now();
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Same cold-load path as lookup_passages; keep it off the async
    // worker.
    match tokio::task::block_in_place(|| state.citation(&name, &request.source, request.paragraph))
    {
        None => not_found(&name, started_at),
        Some(Err(io_error)) => passages_unreadable(&state, io_error, started_at),
        Some(Ok(CitationLookup::UnknownSource)) => {
            state.note_read(&name, true);
            error(
                ErrorCode::NoSource,
                format!("source '{}' not found in context '{name}'", request.source),
                started_at,
            )
        }
        Some(Ok(CitationLookup::IndexOutOfRange)) => {
            state.note_read(&name, true);
            error(
                ErrorCode::NoParagraph,
                format!(
                    "paragraph {} out of range for source '{}' in context '{name}'",
                    request.paragraph, request.source
                ),
                started_at,
            )
        }
        Some(Ok(CitationLookup::Found(text, section))) => {
            state.note_read(&name, false);
            ok(
                Citation {
                    text,
                    source: request.source,
                    section,
                },
                started_at,
            )
        }
    }
}

/// One page of registered source ids, keyset by id — the list grows
/// with every ingested document, so it pages like the directory.
/// `entries` carries each listed source's metadata (#167) over the
/// same page window; `sources` stays the bare id list it always was,
/// so existing consumers keep parsing.
#[derive(Serialize)]
pub struct SourcePage {
    pub total: usize,
    pub sources: Vec<String>,
    pub entries: Vec<SourceEntry>,
}

/// One listed source with its metadata (#167). Absent metadata omits
/// its key — a source stored before metadata existed lists as bare.
#[derive(Serialize)]
pub struct SourceEntry {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stored_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub date: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

pub async fn list_sources(
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
    // Same cold-load path as lookup_passages; keep it off the async
    // worker.
    match tokio::task::block_in_place(|| state.passage_source_entries(&name)) {
        None => not_found(&name, started_at),
        // `passage_source_entries` already yields BTreeMap-key order —
        // no sort. One read feeds both `sources` and `entries`, so the
        // two views of the page can never disagree.
        Some(Ok(entries)) => {
            let entries: Vec<(String, crate::passages::SourceMeta)> = match query.prefix.as_deref()
            {
                Some(prefix) => entries
                    .into_iter()
                    .filter(|(source, _)| source.starts_with(prefix))
                    .collect(),
                None => entries,
            };
            let total = entries.len();
            let entries: Vec<SourceEntry> = entries
                .into_iter()
                .filter(|(source, _)| {
                    query
                        .after
                        .as_deref()
                        .is_none_or(|after| source.as_str() > after)
                })
                .take(limit)
                .map(|(name, meta)| SourceEntry {
                    name,
                    stored_at: meta.stored_at,
                    date: meta.date,
                    tags: meta.tags,
                })
                .collect();
            let sources = entries.iter().map(|entry| entry.name.clone()).collect();
            ok(
                SourcePage {
                    total,
                    sources,
                    entries,
                },
                started_at,
            )
        }
        Some(Err(io_error)) => passages_unreadable(&state, io_error, started_at),
    }
}

/// The passage store exists but could not be loaded — its snapshot and
/// log hold acknowledged writes, so this is a 500 pointing at disk,
/// never a silent empty answer.
pub(crate) fn passages_unreadable(
    state: &AppState,
    io_error: std::io::Error,
    started_at: Instant,
) -> Response {
    state.metrics().record_error(ErrorKind::Io);
    error(
        ErrorCode::Internal,
        format!("passages could not be read: {io_error}"),
        started_at,
    )
}

#[derive(Debug, Deserialize)]
pub struct RetractSourceRequest {
    pub source: String,
}

/// What one retraction accomplished: how many associations lost this
/// source's contribution, and whether its passage went with it.
#[derive(Serialize)]
pub struct RetractOutcome {
    pub associations_touched: usize,
    pub passage_removed: bool,
}

pub async fn retract_source(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<RetractSourceRequest>,
) -> Response {
    let started_at = Instant::now();
    // Same gate every other name-shaped write goes through (add_associations'
    // source, retract_association's subject/label/object): an empty or
    // oversized source would otherwise reach the lookup below unchecked,
    // paying for a marker fsync and a WAL fsync before failing to find it.
    if let Some(refusal) = empty("source", &request.source, started_at) {
        return refusal;
    }
    if let Some(refusal) = oversized("source", &request.source, MAX_NAME_BYTES, started_at) {
        return refusal;
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Retraction stages a WAL op and fsyncs before returning; keep that
    // synchronous write off the async worker like every other write path.
    match tokio::task::block_in_place(|| state.retract_source(&name, &request.source)) {
        Err(failure) => access_error(&state, failure, &name, started_at),
        Ok((associations_touched, passage_removed)) => {
            // The retracted SOURCE lives in the body, so the access log
            // alone cannot say what was withdrawn — the audit line can.
            tracing::info!(
                target: "taguru::audit",
                key = %crate::api::key_name(&key),
                context = %name,
                source = %request.source,
                associations_touched,
                passage_removed,
                "source retracted",
            );
            // A retraction that found nothing changed nothing; only an
            // effective one counts as a write.
            if associations_touched > 0 || passage_removed {
                state.note_write(&name);
            }
            // Retracting the source is the second documented repair for
            // a torn import (beside re-importing the batch): its truth
            // is now consistently absent, so a surviving batch-open
            // marker stops describing a tear. `state.retract_source`
            // already cleared it (its own or a leftover from a torn
            // batch — the marker is keyed by context and source alone).
            ok(
                RetractOutcome {
                    associations_touched,
                    passage_removed,
                },
                started_at,
            )
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct SearchPassagesRequest {
    pub query: String,
    /// Omitted means 5.
    pub limit: Option<usize>,
    /// One-call override of the vector lane's cosine floor — beats the
    /// context setting, which beats the server default. Clamped to
    /// [0, 1]. Floors only the semantic lane: cosine is the one scale
    /// with absolute meaning here (the fused score is rank arithmetic,
    /// and raw BM25 is corpus-local).
    pub semantic_floor: Option<f32>,
    /// Pre-lane source filter (#167): only sources carrying at least
    /// one of these tags may answer. Empty constrains nothing.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Pre-lane time window (#167), epoch seconds, half-open
    /// `[since, until)` over each source's `date ?? stored_at` —
    /// sources with neither (stored before metadata existed) never
    /// match a time filter.
    pub since: Option<u64>,
    pub until: Option<u64>,
}

/// Validates and normalizes one request's filter fields into the
/// registry's [`SourceFilter`] — `Ok(None)` when nothing constrains
/// (no tags, no window), so an unfiltered request stays on exactly the
/// pre-#167 path. Tags are sorted and deduplicated HERE, before the
/// filter reaches any cache key: two spellings of one filter must
/// mint one key. Shared by search, cross search, and explain, so the
/// three surfaces cannot drift on what a legal filter is.
fn source_filter(
    tags: &[String],
    since: Option<u64>,
    until: Option<u64>,
    started_at: Instant,
) -> Result<Option<crate::passages::SourceFilter>, Box<Response>> {
    if tags.len() > super::MAX_TAGS_PER_SOURCE {
        return Err(Box::new(error(
            ErrorCode::InvalidArgument,
            format!(
                "{} filter tags where at most {} may be named",
                tags.len(),
                super::MAX_TAGS_PER_SOURCE
            ),
            started_at,
        )));
    }
    for tag in tags {
        if let Some(refusal) = empty("a filter tag", tag, started_at) {
            return Err(Box::new(refusal));
        }
        if let Some(refusal) = oversized("a filter tag", tag, super::MAX_TAG_BYTES, started_at) {
            return Err(Box::new(refusal));
        }
    }
    if let (Some(since), Some(until)) = (since, until)
        && since >= until
    {
        return Err(Box::new(error(
            ErrorCode::InvalidArgument,
            format!(
                "since {since} is not before until {until} — the window is half-open \
                 [since, until) and this one selects nothing"
            ),
            started_at,
        )));
    }
    let mut tags = tags.to_vec();
    tags.sort();
    tags.dedup();
    if tags.is_empty() && since.is_none() && until.is_none() {
        return Ok(None);
    }
    Ok(Some(crate::passages::SourceFilter { tags, since, until }))
}

/// The filter fields exactly as they enter a retrieval cache key —
/// one value serialized into BOTH the main key and the sans-query
/// semantic bucket of each search variant, so the two hand-built
/// tuples can never disagree about the filter (the semantic tier
/// compares query text only; a filter missing from its bucket params
/// would pair filter-A registrations with filter-B lookups). `None`
/// (no filter) serializes as `null`, distinct from every real filter.
fn filter_key_params(
    filter: &Option<crate::passages::SourceFilter>,
) -> Option<(&Vec<String>, Option<u64>, Option<u64>)> {
    filter
        .as_ref()
        .map(|filter| (&filter.tags, filter.since, filter.until))
}

/// One PARAGRAPH matched by passage search: the text lane, for
/// knowledge that never decomposed into triples. `paragraph` is its
/// position within the source (0-based, this split); `text` is that
/// paragraph alone — cite it, or dereference the whole source through
/// the lookup endpoint. `score` is the fused reciprocal-rank number
/// when the semantic lane ran, the raw BM25 score otherwise; `lanes`
/// carries each lane's own rank and raw score — evidence for the
/// reading LLM, the same posture as resolve's tiers.
#[derive(Serialize, Deserialize)]
pub struct PassageHit {
    pub source: String,
    pub paragraph: u32,
    pub score: f32,
    pub text: String,
    pub lanes: PassageLanes,
}

#[derive(Serialize, Deserialize)]
pub struct PassageLanes {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25: Option<LaneEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<LaneEvidence>,
}

/// Where one lane put this hit: 1-based rank within the lane's own
/// candidate pool, and that lane's raw score (BM25, or cosine).
#[derive(Serialize, Deserialize)]
pub struct LaneEvidence {
    pub rank: usize,
    pub score: f32,
}

impl LaneEvidence {
    fn from_lane(lane: Option<(usize, f32)>) -> Option<Self> {
        lane.map(|(rank, score)| Self { rank, score })
    }
}

impl From<crate::registry::PassageSearchHit> for PassageHit {
    fn from(hit: crate::registry::PassageSearchHit) -> Self {
        Self {
            source: hit.source,
            paragraph: hit.index,
            score: hit.score,
            text: hit.text,
            lanes: PassageLanes {
                bm25: LaneEvidence::from_lane(hit.bm25),
                vector: LaneEvidence::from_lane(hit.vector),
            },
        }
    }
}

/// The response-level execution plan of one passage search (#151): one
/// entry per context actually searched, in effective order — for the
/// cross variant, the resolved target list (groups expanded, grants
/// applied), the same order the merge breaks ties by and the retrieval
/// cache keys on. What the per-hit `lanes` evidence cannot say — "the
/// semantic lane never ran here, and this is why" — lives here, so a
/// caller can tell a lexical-only answer from a fused one without a
/// separate explain call. The plan describes the computation that
/// produced these hits; a cache tier may replay both together, and
/// every event that could change the plan (a corpus write, a vector
/// publish, a floor change) also moves the cache key.
#[derive(Serialize, Deserialize)]
pub struct SearchPlan {
    pub contexts: Vec<SearchContextPlan>,
}

/// One searched context's account, mirroring the per-hit `lanes` shape.
/// `filter` is present exactly when the request carried a source
/// filter (#167): how many sources were eligible to answer, out of how
/// many the context stores — so an empty page under a narrow filter is
/// diagnosable from the response alone.
#[derive(Serialize, Deserialize)]
pub struct SearchContextPlan {
    pub context: String,
    pub lanes: SearchLanesPlan,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter: Option<FilterPlan>,
}

/// The source filter's account for one searched context (#167).
#[derive(Serialize, Deserialize)]
pub struct FilterPlan {
    pub eligible_sources: usize,
    pub total_sources: usize,
}

impl FilterPlan {
    fn of(report: Option<crate::registry::SourceFilterReport>) -> Option<Self> {
        report.map(|report| Self {
            eligible_sources: report.eligible,
            total_sources: report.total,
        })
    }
}

#[derive(Serialize, Deserialize)]
pub struct SearchLanesPlan {
    pub bm25: LanePlan,
    pub vector: LanePlan,
}

/// One lane's verdict for the whole call: it ran (the vector lane also
/// names the effective cosine `floor` it swept under — the resolved
/// override → context setting → server default chain), or it did not
/// and `reason` says why, in the same prose the explain endpoint uses.
#[derive(Serialize, Deserialize)]
pub struct LanePlan {
    pub ran: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub floor: Option<f32>,
}

impl LanePlan {
    fn ran() -> Self {
        Self {
            ran: true,
            reason: None,
            floor: None,
        }
    }

    fn skipped(reason: String) -> Self {
        Self {
            ran: false,
            reason: Some(reason),
            floor: None,
        }
    }
}

impl SearchContextPlan {
    /// One context's plan entry from the registry's account of its
    /// search — the vector arm maps through the same reason strings
    /// explain emits, so the two surfaces cannot drift apart in prose.
    /// Shared with `search_communities`, whose ranking IS this search
    /// (it passes no filter — its sources are synthetic `community:`
    /// rows that carry no user metadata).
    pub(crate) fn of(
        context: &str,
        lanes: &crate::registry::PassageSearchLanes,
        filter: Option<FilterPlan>,
    ) -> Self {
        use crate::registry::{PassageSearchLanes, VectorLaneStatus};

        let both = |reason: &str| SearchLanesPlan {
            bm25: LanePlan::skipped(reason.to_string()),
            vector: LanePlan::skipped(reason.to_string()),
        };
        let lanes = match lanes {
            PassageSearchLanes::NoQueryTerms => both("the query yields no searchable terms"),
            PassageSearchLanes::ZeroLimit => both("the requested limit is 0"),
            PassageSearchLanes::Ran { vector } => SearchLanesPlan {
                bm25: LanePlan::ran(),
                vector: match vector {
                    VectorLaneStatus::Off {
                        provider_configured,
                    } => LanePlan::skipped(vector_off_reason(*provider_configured)),
                    VectorLaneStatus::QueryEmbeddingFailed(error) => {
                        LanePlan::skipped(vector_failed_reason(error))
                    }
                    VectorLaneStatus::NoVectors => LanePlan::skipped(vector_empty_reason()),
                    VectorLaneStatus::ModelChanged { stored, current } => {
                        LanePlan::skipped(vector_model_changed_reason(stored, current))
                    }
                    VectorLaneStatus::WidthChanged { stored, current } => {
                        LanePlan::skipped(vector_width_changed_reason(*stored, *current))
                    }
                    VectorLaneStatus::Ran { floor } => LanePlan {
                        floor: Some(*floor),
                        ..LanePlan::ran()
                    },
                },
            },
        };
        Self {
            context: context.to_string(),
            lanes,
            filter,
        }
    }
}

/// [`search_passages`]' result: the plan beside the hits it accounts
/// for. The hits array is unchanged from the pre-#151 bare-array shape
/// — it moved under `hits`.
#[derive(Serialize, Deserialize)]
pub struct PassagePage {
    pub plan: SearchPlan,
    pub hits: Vec<PassageHit>,
}

/// [`cross_search_passages`]' result — the same wrap, context-tagged
/// hits. The router mode re-merges this shape across shards.
#[derive(Serialize, Deserialize)]
pub struct CrossPassagePage {
    pub plan: SearchPlan,
    pub hits: Vec<CrossMatch<PassageHit>>,
}

/// The one set of wire reason strings for a semantic lane that did not
/// run — shared by the explain report and the search plan.
fn vector_off_reason(provider_configured: bool) -> String {
    if provider_configured {
        "passage embedding is off (TAGURU_EMBED_PASSAGES)".to_string()
    } else {
        "no embedding provider is configured".to_string()
    }
}

fn vector_failed_reason(error: &str) -> String {
    format!("the query embedding failed: {error}")
}

fn vector_empty_reason() -> String {
    "no paragraph vectors exist yet — the embedding refresh has not covered this context"
        .to_string()
}

fn vector_model_changed_reason(stored: &str, current: &str) -> String {
    format!(
        "stored vectors belong to model '{stored}' but the provider is \
         '{current}' — they are never served, and the next refresh re-embeds"
    )
}

fn vector_width_changed_reason(stored: usize, current: usize) -> String {
    format!(
        "stored vectors are {stored}-dimensional but the model now answers \
         {current} (a dimensions setting changed behind its name) — they are \
         never served, and the next refresh re-embeds"
    )
}

pub async fn search_passages(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<SearchPassagesRequest>,
) -> Response {
    let started_at = Instant::now();
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let limit = clamp(request.limit, 5, MAX_MATCH_LIMIT);
    let filter = match source_filter(&request.tags, request.since, request.until, started_at) {
        Ok(filter) => filter,
        Err(refusal) => return *refusal,
    };
    // The one filter value BOTH key tuples below serialize — see
    // `filter_key_params` for why sharing it is load-bearing.
    let filter_params = filter_key_params(&filter);
    // Minted before the search — see `retrieval_key`. The raw
    // `semantic_floor` goes in unclamped: two spellings of one
    // effective floor just occupy two entries, which is only a hit-rate
    // cost, never a correctness one.
    let key = state.retrieval_key(
        RetrievalCacheOp::SearchPassages,
        std::slice::from_ref(&name),
        serde_json::to_string(&(
            "search_passages",
            &request.query,
            limit,
            request.semantic_floor,
            &filter_params,
        ))
        .ok(),
    );
    let mut semantic_fill = None;
    if let Some(key) = &key {
        if let Some(found) = state.retrieval_lookup(key) {
            replay_cached_search(&state, key, &found);
            if search_log_enabled() {
                tracing::info!(
                    target: "taguru::search",
                    context = %name,
                    op = "search_passages",
                    cue = %request.query,
                    hits = found.log_hits,
                    top_score = f64::from(found.log_top_score),
                    cached = true,
                    "search",
                );
            }
            return ok(found.payload.as_ref(), started_at);
        }
        // The semantic tier (see `semantic_retrieval`): the bucket is
        // the key params with the query stripped, so equivalence can
        // only pair requests that agree on everything else — the
        // filter included. Blocking section — the probe may pay the
        // query embedding the search below would otherwise pay (same
        // cue cache, one provider call either way).
        if let Ok(sans_query) = serde_json::to_string(&(
            "search_passages",
            limit,
            request.semantic_floor,
            &filter_params,
        )) && let Some(probe) = tokio::task::block_in_place(|| {
            state.semantic_retrieval(key, &sans_query, &request.query, deadline)
        }) {
            if let Some(served) = probe.served {
                replay_cached_search(&state, key, &served.value);
                if search_log_enabled() {
                    tracing::info!(
                        target: "taguru::search",
                        context = %name,
                        op = "search_passages",
                        cue = %request.query,
                        hits = served.value.log_hits,
                        top_score = f64::from(served.value.log_top_score),
                        cached = true,
                        similarity = f64::from(served.similarity),
                        matched = %served.canonical,
                        "search",
                    );
                }
                return ok(served.value.payload.as_ref(), started_at);
            }
            semantic_fill = Some(SemanticFill {
                params: sans_query,
                query: request.query.clone(),
                embedding: probe.embedding,
            });
        }
    }
    // Off the async worker: a residency's first search tokenizes the
    // whole corpus into the index (the audit endpoints' rule).
    let outcome = tokio::task::block_in_place(|| {
        state.search_passages(
            &name,
            &request.query,
            limit,
            request.semantic_floor,
            filter.as_ref(),
            deadline,
        )
    });
    match outcome {
        None => not_found(&name, started_at),
        // A rebuild the lexical lane needed refused to start once the
        // budget was already gone — the same "before it could start"
        // shape as the entry check above, just discovered later, past
        // the embedding call this search's semantic lane also makes.
        Some(Err(_)) if deadline.expired() => deadline_exceeded(started_at),
        Some(Err(io_error)) => passages_unreadable(&state, io_error, started_at),
        Some(Ok(found)) => {
            state.note_search(SearchOp::SearchPassages, &name, found.hits.is_empty());
            let target_empty = vec![found.hits.is_empty()];
            let mut lane_hits = [0u64; 3];
            for hit in &found.hits {
                state
                    .metrics()
                    .record_passage_hit(hit.bm25.is_some(), hit.vector.is_some());
                match (hit.bm25.is_some(), hit.vector.is_some()) {
                    (true, false) => lane_hits[0] += 1,
                    (true, true) => lane_hits[1] += 1,
                    (false, true) => lane_hits[2] += 1,
                    (false, false) => {}
                }
            }
            if search_log_enabled() {
                tracing::info!(
                    target: "taguru::search",
                    context = %name,
                    op = "search_passages",
                    cue = %request.query,
                    hits = found.hits.len(),
                    top_score = found.hits.first().map_or(0.0, |hit| f64::from(hit.score)),
                    "search",
                );
            }
            // A transiently degraded fill must not be pinned — see
            // `PassageSearchLanes::embedding_failed` (the semantic
            // claim is skipped with it; both live behind the key).
            let key = key.filter(|_| !found.lanes.embedding_failed());
            let payload = PassagePage {
                plan: SearchPlan {
                    contexts: vec![SearchContextPlan::of(
                        &name,
                        &found.lanes,
                        FilterPlan::of(found.filter),
                    )],
                },
                hits: found.hits.into_iter().map(PassageHit::from).collect(),
            };
            let top_score = payload.hits.first().map_or(0.0, |hit| hit.score);
            let log_hits = payload.hits.len();
            cache_and_serve(
                &state,
                key,
                &payload,
                target_empty,
                lane_hits,
                log_hits,
                top_score,
                semantic_fill,
                started_at,
            )
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ExplainSearchRequest {
    pub query: String,
    /// The thing the caller expected to see.
    pub source: String,
    /// Which of the source's paragraphs (0-based). Omitted means "its
    /// best showing": the best-ranked paragraph, or the one sharing
    /// the most query terms when nothing ranked.
    #[serde(default, alias = "index")]
    pub paragraph: Option<u32>,
    /// The search call being explained; omitted means 5, the same
    /// default `sources/search` applies.
    pub limit: Option<usize>,
    /// The floor override of the search call being explained — pass
    /// the same value, or the explanation accounts for a call nobody
    /// made.
    pub semantic_floor: Option<f32>,
    /// The source filter of the search being explained (#167) — same
    /// rule as the floor: pass the same values, or the explanation
    /// accounts for a call nobody made.
    #[serde(default)]
    pub tags: Vec<String>,
    pub since: Option<u64>,
    pub until: Option<u64>,
}

/// One verdict for "why didn't (or did) this source appear for this
/// query": the first that applies, machine-readable in `verdict`,
/// human-readable in `summary`, evidence attached for the skeptical.
/// Every verdict is a 200 — a diagnosed miss is this endpoint's
/// success, not its failure.
#[derive(Serialize)]
pub struct SearchExplanation {
    /// `not_stored` | `paragraph_out_of_range` | `filtered_out` |
    /// `no_query_terms` | `no_term_overlap` | `below_cutoff` |
    /// `served`, first match wins in that order (a served paragraph is
    /// served, whatever else is true of it). `filtered_out` (#167)
    /// means the source exists but the request's source filter
    /// excludes it — the search being explained never considered it.
    pub verdict: &'static str,
    pub summary: String,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paragraph: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paragraphs: Option<usize>,
    /// Present (and false) when the endpoint picked the paragraph
    /// because the request named none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paragraph_named: Option<bool>,
    /// The query's terms as strings — which words, which character
    /// bigrams — exactly what both lanes matched against.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_terms: Option<Vec<String>>,
    /// The paragraph's own terms (doc2query questions included) — only
    /// on `no_term_overlap`, where seeing both sides IS the diagnosis
    /// (query says 酒造, paragraph spells 酒蔵).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub paragraph_terms: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25: Option<Bm25Explain>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<VectorExplain>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranking: Option<RankingExplain>,
}

/// The lexical lane's evidence for the target: its rank in that lane,
/// its BM25 score, and the score's per-term addends.
#[derive(Serialize)]
pub struct Bm25Explain {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    pub score: f32,
    pub terms: Vec<TermContribution>,
}

/// One query term against the target paragraph: `df` paragraphs carry
/// it corpus-wide (its `idf` follows), the target carries it `tf`
/// times, contributing `contribution` to the BM25 score. `tf` 0 with a
/// high `df` is the "matched only ubiquitous bigrams" signature.
#[derive(Serialize)]
pub struct TermContribution {
    pub term: String,
    pub tf: f32,
    pub df: usize,
    pub idf: f32,
    pub contribution: f32,
}

/// The vector lane's evidence — or the reason there is none.
#[derive(Serialize)]
pub struct VectorExplain {
    pub ran: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub floor: Option<f32>,
    /// The target's best cosine across its rows, floor or no floor —
    /// "scored 0.31 against floor 0.35" is the actionable half.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cosine: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
}

/// Where the target stands in the fused ranking `sources/search`
/// truncates: its rank against `ranked` scored candidates, the
/// `cutoff_score` the request's `limit` served down to, and a
/// `limit_to_reach` VERIFIED by rerunning the real serve computation
/// (pool caps included), not read off the unbounded ranking.
#[derive(Serialize)]
pub struct RankingExplain {
    pub fused: bool,
    pub ranked: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    pub limit: usize,
    pub served: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cutoff_score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit_to_reach: Option<usize>,
}

impl SearchExplanation {
    /// A verdict-only shell for the arms that end before scoring.
    fn shell(verdict: &'static str, summary: String, source: &str) -> Self {
        Self {
            verdict,
            summary,
            source: source.to_string(),
            paragraph: None,
            paragraphs: None,
            paragraph_named: None,
            query_terms: None,
            paragraph_terms: None,
            bm25: None,
            vector: None,
            ranking: None,
        }
    }

    fn from_lookup(
        name: &str,
        request: &ExplainSearchRequest,
        lookup: PassageExplainLookup,
    ) -> Self {
        match lookup {
            PassageExplainLookup::UnknownSource => Self::shell(
                "not_stored",
                format!(
                    "source '{}' has no passages in context '{name}' — never stored here, \
                     or stored and later retracted (the store keeps no tombstone history \
                     to tell which)",
                    request.source
                ),
                &request.source,
            ),
            PassageExplainLookup::IndexOutOfRange { paragraphs } => {
                let mut explanation = Self::shell(
                    "paragraph_out_of_range",
                    format!(
                        "paragraph {} is out of range for source '{}' — it stores {} \
                         paragraph(s), 0-based",
                        request.paragraph.unwrap_or_default(),
                        request.source,
                        paragraphs
                    ),
                    &request.source,
                );
                explanation.paragraph = request.paragraph;
                explanation.paragraphs = Some(paragraphs);
                explanation
            }
            PassageExplainLookup::FilteredOut => Self::shell(
                "filtered_out",
                format!(
                    "source '{}' is excluded by the request's source filter (tags/since/until) \
                     — the search being explained never considered it",
                    request.source
                ),
                &request.source,
            ),
            PassageExplainLookup::NoQueryTerms => Self::shell(
                "no_query_terms",
                "the query yields no searchable terms — a search of it answers the empty \
                 list before either lane runs"
                    .to_string(),
                &request.source,
            ),
            PassageExplainLookup::Explained(explanation) => {
                Self::from_explanation(&request.source, *explanation)
            }
        }
    }

    fn from_explanation(
        source: &str,
        explanation: crate::registry::PassageSearchExplanation,
    ) -> Self {
        use crate::registry::VectorLaneReport;

        let verdict = if explanation.served {
            "served"
        } else if explanation.rank.is_some() {
            "below_cutoff"
        } else {
            "no_term_overlap"
        };

        let vector = match &explanation.vector {
            VectorLaneReport::Off {
                provider_configured,
            } => VectorExplain {
                ran: false,
                reason: Some(vector_off_reason(*provider_configured)),
                floor: None,
                cosine: None,
                rank: None,
            },
            VectorLaneReport::QueryEmbeddingFailed(error) => VectorExplain {
                ran: false,
                reason: Some(vector_failed_reason(error)),
                floor: None,
                cosine: None,
                rank: None,
            },
            VectorLaneReport::NoVectors => VectorExplain {
                ran: false,
                reason: Some(vector_empty_reason()),
                floor: None,
                cosine: None,
                rank: None,
            },
            VectorLaneReport::ModelChanged { stored, current } => VectorExplain {
                ran: false,
                reason: Some(vector_model_changed_reason(stored, current)),
                floor: None,
                cosine: None,
                rank: None,
            },
            VectorLaneReport::WidthChanged { stored, current } => VectorExplain {
                ran: false,
                reason: Some(vector_width_changed_reason(*stored, *current)),
                floor: None,
                cosine: None,
                rank: None,
            },
            VectorLaneReport::Ran { floor, cosine } => VectorExplain {
                ran: true,
                reason: None,
                floor: Some(*floor),
                cosine: *cosine,
                rank: explanation.vector_lane.map(|(rank, _)| rank),
            },
        };

        let summary = match verdict {
            "served" => format!(
                "served: paragraph {} of '{source}' ranked {} of {} at limit {}",
                explanation.paragraph,
                explanation.rank.unwrap_or_default(),
                explanation.ranked,
                explanation.limit
            ),
            "below_cutoff" => {
                let reach = match explanation.limit_to_reach {
                    Some(limit) => format!("limit {limit} reaches it"),
                    None => format!(
                        "no limit up to {} reaches it (pool interplay)",
                        explanation.ranked
                    ),
                };
                format!(
                    "paragraph {} of '{source}' ranked {} of {} — the cutoff at limit {} \
                     was score {}; {reach}",
                    explanation.paragraph,
                    explanation.rank.unwrap_or_default(),
                    explanation.ranked,
                    explanation.limit,
                    explanation
                        .cutoff_score
                        .map_or_else(|| "-".to_string(), |score| format!("{score:.4}")),
                )
            }
            _ => {
                let vector_clause = match (&explanation.vector, &vector) {
                    (VectorLaneReport::Ran { floor, cosine }, _) => match cosine {
                        Some(cosine) => format!(
                            "; the vector lane scored it {cosine:.4} against floor {floor:.4}"
                        ),
                        None => "; the vector lane ran but this paragraph has no current \
                                 embedding yet"
                            .to_string(),
                    },
                    (_, vector_explain) => format!(
                        " and the vector lane did not run ({})",
                        vector_explain.reason.as_deref().unwrap_or("off")
                    ),
                };
                format!(
                    "paragraph {} of '{source}' shares no term with the query{vector_clause}",
                    explanation.paragraph
                )
            }
        };

        // The per-term table marries the registry's evidence (query-
        // gram order) to its spellings (same order, same dedup rule).
        let bm25 = explanation.lexical.map(|lexical| Bm25Explain {
            rank: explanation.bm25_lane.map(|(rank, _)| rank),
            score: lexical.score,
            terms: lexical
                .terms
                .into_iter()
                .zip(explanation.query_terms.iter())
                .map(|(term, (spelling, _))| TermContribution {
                    term: spelling.clone(),
                    tf: term.tf,
                    df: term.carriers as usize,
                    idf: term.idf,
                    contribution: term.contribution,
                })
                .collect(),
        });

        Self {
            verdict,
            summary,
            source: source.to_string(),
            paragraph: Some(explanation.paragraph),
            paragraphs: Some(explanation.paragraphs),
            paragraph_named: Some(explanation.paragraph_named),
            query_terms: Some(
                explanation
                    .query_terms
                    .into_iter()
                    .map(|(spelling, _)| spelling)
                    .collect(),
            ),
            paragraph_terms: explanation.paragraph_terms,
            bm25,
            vector: Some(vector),
            ranking: Some(RankingExplain {
                fused: explanation.fused,
                ranked: explanation.ranked,
                rank: explanation.rank,
                score: explanation.score,
                limit: explanation.limit,
                served: explanation.served,
                cutoff_score: explanation.cutoff_score,
                limit_to_reach: explanation.limit_to_reach,
            }),
        }
    }
}

/// `POST /contexts/{name}/sources/search/explain` — one call instead
/// of "orchestrate four endpoints and cross-reference by hand": name
/// the query and the source (optionally the paragraph) you expected to
/// see, get the first verdict that applies with its evidence. Runs the
/// same lanes the search runs (read-only, roughly one query plus one
/// targeted scoring); the serve boundary is recomputed exactly as
/// `sources/search` computes it, so the two cannot disagree.
pub async fn explain_search_passages(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<ExplainSearchRequest>,
) -> Response {
    let started_at = Instant::now();
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let filter = match source_filter(&request.tags, request.since, request.until, started_at) {
        Ok(filter) => filter,
        Err(refusal) => return *refusal,
    };
    // Off the async worker: a residency's first search tokenizes the
    // whole corpus into the index (the audit endpoints' rule).
    let outcome = tokio::task::block_in_place(|| {
        state.explain_passage_search(
            &name,
            &request.query,
            &request.source,
            request.paragraph,
            clamp(request.limit, 5, MAX_MATCH_LIMIT),
            request.semantic_floor,
            filter.as_ref(),
            deadline,
        )
    });
    match outcome {
        None => not_found(&name, started_at),
        // Mirrors search_passages: a rebuild the lexical lane needed
        // refused to start once the budget was already gone.
        Some(Err(_)) if deadline.expired() => deadline_exceeded(started_at),
        Some(Err(io_error)) => passages_unreadable(&state, io_error, started_at),
        Some(Ok(lookup)) => {
            // A lookup that never reached scoring is the unproductive
            // read; a diagnosed miss is exactly what was asked for.
            state.note_read(&name, !matches!(lookup, PassageExplainLookup::Explained(_)));
            ok(
                SearchExplanation::from_lookup(&name, &request, lookup),
                started_at,
            )
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CrossSearchPassagesRequest {
    /// Full context names — no patterns.
    #[serde(default)]
    pub contexts: Vec<String>,
    /// Group names, resolved and deduped as in
    /// [`super::CrossRecallRequest`].
    #[serde(default)]
    pub groups: Vec<String>,
    pub query: String,
    /// Omitted means 5.
    pub limit: Option<usize>,
    /// One-call override of every target's vector-lane cosine floor —
    /// beats each context's own setting, which beats the server
    /// default. Clamped to [0, 1]. One value for all targets: cosine
    /// shares a scale across contexts (unlike BM25 and the fused
    /// number, which is why the merge interleaves by rank).
    pub semantic_floor: Option<f32>,
    /// Pre-lane source filter (#167), one value for all targets —
    /// same shape and semantics as the single-context search's.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub since: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub until: Option<u64>,
}

/// [`search_passages`] across several named contexts at once, every
/// hit tagged with its context. Unlike the graph lanes' weights,
/// passage scores do NOT share a scale across contexts (BM25
/// statistics are corpus-local; fusion numbers are rank arithmetic),
/// so the merged order is rank interleaving — every context's best
/// hit, then every second hit, ties broken by target-list order: the
/// same rank-fusion posture the endpoint already takes across its two
/// lanes. `score` stays what it was, per-context evidence. Every
/// target's search runs concurrently, bounded by
/// [`cross_search_concurrency`] — on a stone-cold cache, up to that
/// many targets may each pay for the query embedding before the first
/// resolution lands in the cue cache, instead of exactly one target
/// paying and the rest reusing it; the cache is still the single
/// source of truth (a `Mutex`), so this is wasted provider calls, not
/// a correctness risk, and every request after the first is unaffected.
pub async fn cross_search_passages(
    State(state): State<AppState>,
    scope: Option<axum::Extension<crate::auth::KeyScope>>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<CrossSearchPassagesRequest>,
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
    let limit = clamp(request.limit, 5, MAX_MATCH_LIMIT);
    let filter = match source_filter(&request.tags, request.since, request.until, started_at) {
        Ok(filter) => filter,
        Err(refusal) => return *refusal,
    };
    // The one filter value BOTH key tuples below serialize — see
    // `filter_key_params` for why sharing it is load-bearing.
    let filter_params = filter_key_params(&filter);
    // One rank cut for both sites below: inside the loop it holds the
    // memory bound (hits carry their full paragraph text) — firing at
    // twice the limit and coming back to the limit, so each firing
    // discards at least `limit` hits instead of re-sorting per
    // target — and after the loop it produces the page. Exact both
    // times: (rank, index) keys are unique, later contexts only
    // append larger indexes, and whatever sits outside a prefix
    // pool's best `limit` sits outside every superset's.
    let cut = |pool: &mut Vec<_>| {
        pool.sort_by_key(|(index, rank, _)| (*rank, *index));
        pool.truncate(limit);
    };
    // A budget already spent when the request arrived shouldn't pay to
    // tokenize even one context — checked once before the fan-out
    // starts, mirroring the single-context handler's pre-flight cost
    // discipline.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Resolved-target keying, minted before any fetch — the same
    // contract as `cross_recall`'s key (see there) with this op's
    // lanes.
    let key = state.retrieval_key(
        RetrievalCacheOp::SearchPassages,
        &targets,
        serde_json::to_string(&(
            "cross_search_passages",
            &request.query,
            limit,
            request.semantic_floor,
            &filter_params,
        ))
        .ok(),
    );
    let mut semantic_fill = None;
    if let Some(key) = &key {
        if let Some(found) = state.retrieval_lookup(key) {
            replay_cached_search(&state, key, &found);
            if search_log_enabled() {
                tracing::info!(
                    target: "taguru::search",
                    contexts = %targets.join(","),
                    op = "search_passages",
                    cue = %request.query,
                    hits = found.log_hits,
                    top_score = f64::from(found.log_top_score),
                    cached = true,
                    "search",
                );
            }
            return ok(found.payload.as_ref(), started_at);
        }
        // The semantic tier, keyed on the resolved target list like
        // the exact key above. A side effect worth having: the probe
        // warms the cue cache before the fan-out, so a cold cross
        // search no longer has up to `permits` targets each paying
        // for the same query embedding.
        if let Ok(sans_query) = serde_json::to_string(&(
            "cross_search_passages",
            limit,
            request.semantic_floor,
            &filter_params,
        )) && let Some(probe) = tokio::task::block_in_place(|| {
            state.semantic_retrieval(key, &sans_query, &request.query, deadline)
        }) {
            if let Some(served) = probe.served {
                replay_cached_search(&state, key, &served.value);
                if search_log_enabled() {
                    tracing::info!(
                        target: "taguru::search",
                        contexts = %targets.join(","),
                        op = "search_passages",
                        cue = %request.query,
                        hits = served.value.log_hits,
                        top_score = f64::from(served.value.log_top_score),
                        cached = true,
                        similarity = f64::from(served.similarity),
                        matched = %served.canonical,
                        "search",
                    );
                }
                return ok(served.value.payload.as_ref(), started_at);
            }
            semantic_fill = Some(SemanticFill {
                params: sans_query,
                query: request.query.clone(),
                embedding: probe.embedding,
            });
        }
    }
    // Each residency's first search tokenizes its whole corpus, so the
    // fetch belongs on the blocking pool — bounded and concurrent, the
    // same `bounded_parallel_map` shape as `cross_matches`'s gather.
    // `deadline` is `Copy`, so every job carries its own value and can
    // bail out mid-tokenize the same way the single-context handler
    // does.
    let permits = cross_search_concurrency().min(targets.len().max(1));
    let owned_targets = Arc::clone(&targets);
    let query = request.query.clone();
    let semantic_floor = request.semantic_floor;
    let job_filter = filter.clone();
    let job_state = state.clone();
    let fetched = bounded_parallel_map(targets.len(), permits, move |index| {
        job_state.search_passages(
            &owned_targets[index],
            &query,
            limit,
            semantic_floor,
            job_filter.as_ref(),
            deadline,
        )
    })
    .await;

    // Sequential merge: every fetch has already landed, so nothing
    // here blocks. The first per-context failure aborts the whole
    // response (a read has nothing to half-apply) — in target-list
    // order now that every fetch runs concurrently, not the first one
    // hit in real time, though the response is identical either way.
    // Mirrors the single-context handler: an error surfacing once the
    // budget is gone is reported as a timeout, not as whatever shape
    // the abandoned work happened to fail with.
    let mut pool = Vec::new();
    let mut target_empty = Vec::with_capacity(targets.len());
    let mut plans = Vec::with_capacity(targets.len());
    let mut embedding_failed = false;
    let mut lane_hits = [0u64; 3];
    for (index, outcome) in fetched.into_iter().enumerate() {
        let name = &targets[index];
        match outcome {
            None => return not_found(name, started_at),
            Some(Err(_)) if deadline.expired() => return deadline_exceeded(started_at),
            Some(Err(io_error)) => return passages_unreadable(&state, io_error, started_at),
            Some(Ok(found)) => {
                state.note_search(SearchOp::SearchPassages, name, found.hits.is_empty());
                target_empty.push(found.hits.is_empty());
                plans.push(SearchContextPlan::of(
                    name,
                    &found.lanes,
                    FilterPlan::of(found.filter),
                ));
                embedding_failed |= found.lanes.embedding_failed();
                for hit in &found.hits {
                    state
                        .metrics()
                        .record_passage_hit(hit.bm25.is_some(), hit.vector.is_some());
                    match (hit.bm25.is_some(), hit.vector.is_some()) {
                        (true, false) => lane_hits[0] += 1,
                        (true, true) => lane_hits[1] += 1,
                        (false, true) => lane_hits[2] += 1,
                        (false, false) => {}
                    }
                }
                pool.extend(
                    found
                        .hits
                        .into_iter()
                        .enumerate()
                        .map(|(rank, hit)| (index, rank, hit)),
                );
                if pool.len() >= limit * 2 {
                    cut(&mut pool);
                }
            }
        }
    }
    cut(&mut pool);
    if search_log_enabled() {
        tracing::info!(
            target: "taguru::search",
            contexts = %targets.join(","),
            op = "search_passages",
            cue = %request.query,
            hits = pool.len(),
            top_score = pool.first().map_or(0.0, |(_, _, hit)| f64::from(hit.score)),
            "search",
        );
    }
    // One target's transient embedding failure uncaches the whole
    // response — see `PassageSearchLanes::embedding_failed`.
    let key = key.filter(|_| !embedding_failed);
    let payload = CrossPassagePage {
        plan: SearchPlan { contexts: plans },
        hits: pool
            .into_iter()
            .map(|(index, _, hit)| CrossMatch {
                context: targets[index].clone(),
                inner: PassageHit::from(hit),
            })
            .collect(),
    };
    let top_score = payload.hits.first().map_or(0.0, |found| found.inner.score);
    let log_hits = payload.hits.len();
    cache_and_serve(
        &state,
        key,
        &payload,
        target_empty,
        lane_hits,
        log_hits,
        top_score,
        semantic_fill,
        started_at,
    )
}

/// Original text passages keyed by source id — the same opaque ids that
/// appear on attributions — plus, optionally, doc2query questions and
/// section markers per source, each naming a paragraph of that
/// source's text IN THIS REQUEST (a question or section cannot attach
/// to text the request does not carry: storage replaces per source,
/// wholesale). Built by [`interpret_store_passages`]'s raw-JSON walk
/// (issue #182) rather than derived `Deserialize`, so a wrong-typed or
/// orphaned field anywhere in the body is diagnosed alongside every
/// other issue in one pass instead of rejecting the whole request at
/// the JSON-extractor layer.
#[derive(Debug)]
pub struct StorePassagesRequest {
    pub passages: BTreeMap<String, String>,
    pub questions: BTreeMap<String, Vec<QuestionSpec>>,
    pub sections: BTreeMap<String, Vec<SectionSpec>>,
    /// Source tags (#167), per source in THIS request — the same
    /// source-must-name-a-passage rule as questions/sections, and the
    /// same wholesale-replace semantics: a re-store without tags
    /// clears them.
    pub tags: BTreeMap<String, Vec<String>>,
    /// User-supplied document dates (#167), epoch seconds per source
    /// in THIS request — the document's own time, which time filters
    /// prefer over the server's `stored_at` stamp.
    pub dates: BTreeMap<String, u64>,
}

#[derive(Debug)]
pub struct QuestionSpec {
    pub paragraph: u32,
    pub question: String,
}

#[derive(Debug)]
pub struct SectionSpec {
    pub paragraph: u32,
    pub section: String,
}

/// One `{paragraph, <text-field>}` item shared by questions and
/// sections — reads the paragraph index, then the named bounded text
/// field, collecting an [`Issue`] per problem instead of stopping at
/// the first (issue #182).
fn interpret_paragraph_and_text(
    item: &Value,
    path: &str,
    text_key: &str,
    text_cap: usize,
    issues: &mut Vec<Issue>,
) -> (u32, String) {
    let Some(obj) = item.as_object() else {
        issues.push(Issue::wrong_type(path, "an object", item));
        return (0, String::new());
    };
    let paragraph = match obj.get("paragraph") {
        None | Some(Value::Null) => {
            issues.push(Issue::missing(
                format!("{path}.paragraph"),
                "a non-negative integer paragraph index",
            ));
            0
        }
        Some(value @ Value::Number(number)) => {
            match number.as_u64().and_then(|value| u32::try_from(value).ok()) {
                Some(paragraph) => paragraph,
                None => {
                    issues.push(Issue::wrong_type(
                        format!("{path}.paragraph"),
                        "a non-negative integer paragraph index",
                        value,
                    ));
                    0
                }
            }
        }
        Some(other) => {
            issues.push(Issue::wrong_type(
                format!("{path}.paragraph"),
                "a non-negative integer paragraph index",
                other,
            ));
            0
        }
    };
    let text = interpret_bounded_text(obj, text_key, path, text_cap, issues);
    (paragraph, text)
}

/// A required, non-empty, within-cap string field read from a JSON
/// object — shared by question text, section labels, and tags (issue
/// #182), each a short string riding a paragraph index or a source.
fn interpret_bounded_text(
    obj: &serde_json::Map<String, Value>,
    key: &str,
    path: &str,
    cap: usize,
    issues: &mut Vec<Issue>,
) -> String {
    let full_path = format!("{path}.{key}");
    match obj.get(key) {
        None | Some(Value::Null) => {
            issues.push(Issue::missing(full_path, "a non-empty string"));
            String::new()
        }
        Some(Value::String(text)) if text.is_empty() => {
            issues.push(Issue::empty(full_path));
            String::new()
        }
        Some(Value::String(text)) if text.len() > cap => {
            issues.push(Issue::too_long(full_path, cap, text.len()));
            String::new()
        }
        Some(Value::String(text)) => text.clone(),
        Some(other) => {
            issues.push(Issue::wrong_type(full_path, "a non-empty string", other));
            String::new()
        }
    }
}

/// `source`'s orphan rule shared by questions/sections/tags/dates: a
/// source must name a passage carried alongside it IN THIS REQUEST — a
/// question or section cannot attach to text the request does not
/// carry.
fn check_orphaned_source(
    path: String,
    source: &str,
    passages: &BTreeMap<String, String>,
    issues: &mut Vec<Issue>,
) {
    if !passages.contains_key(source) {
        issues.push(Issue::unknown_reference(
            path,
            "a source id present in this request's own `passages`",
        ));
    }
}

/// Interprets `passages`: an object of source id → text. Source ids are
/// names like any other (empty or oversized refused), the text itself
/// rides under the body cap instead.
fn interpret_passages(
    obj: &serde_json::Map<String, Value>,
    issues: &mut Vec<Issue>,
) -> BTreeMap<String, String> {
    let mut passages = BTreeMap::new();
    match obj.get("passages") {
        None | Some(Value::Null) => {
            issues.push(Issue::missing("passages", "an object of source id -> text"));
        }
        Some(Value::Object(map)) => {
            for (source, text) in map {
                if source.is_empty() {
                    issues.push(Issue::empty("a passage source id"));
                } else if source.len() > MAX_NAME_BYTES {
                    issues.push(Issue::too_long(
                        "a passage source id",
                        MAX_NAME_BYTES,
                        source.len(),
                    ));
                }
                match text {
                    Value::String(text) => {
                        passages.insert(source.clone(), text.clone());
                    }
                    other => issues.push(Issue::wrong_type(
                        format!("passages['{source}']"),
                        "a string",
                        other,
                    )),
                }
            }
        }
        Some(other) => issues.push(Issue::wrong_type(
            "passages",
            "an object of source id -> text",
            other,
        )),
    }
    passages
}

/// Interprets `questions`: an object of source → `[{paragraph,
/// question}]` — sources must name passages in THIS request, sizes and
/// per-paragraph counts stay under the shared caps (whether a
/// paragraph index exists in the text is settled at store time, one
/// rule for every entrance).
fn interpret_questions(
    obj: &serde_json::Map<String, Value>,
    passages: &BTreeMap<String, String>,
    issues: &mut Vec<Issue>,
) -> BTreeMap<String, Vec<QuestionSpec>> {
    let mut questions = BTreeMap::new();
    match obj.get("questions") {
        None | Some(Value::Null) => {}
        Some(Value::Object(map)) => {
            for (source, list) in map {
                let path = format!("questions['{source}']");
                let Some(array) = list.as_array() else {
                    issues.push(Issue::wrong_type(path, "an array", list));
                    continue;
                };
                check_orphaned_source(path.clone(), source, passages, issues);
                let mut per_paragraph: BTreeMap<u32, usize> = BTreeMap::new();
                let mut specs = Vec::with_capacity(array.len());
                for (index, item) in array.iter().enumerate() {
                    let item_path = format!("{path}[{index}]");
                    let (paragraph, question) = interpret_paragraph_and_text(
                        item,
                        &item_path,
                        "question",
                        MAX_QUESTION_BYTES,
                        issues,
                    );
                    let count = per_paragraph.entry(paragraph).or_insert(0);
                    *count += 1;
                    if *count > MAX_QUESTIONS_PER_PARAGRAPH {
                        issues.push(Issue::over_limit(
                            format!("{item_path}.paragraph"),
                            format!("at most {MAX_QUESTIONS_PER_PARAGRAPH} questions per paragraph"),
                            format!("paragraph {paragraph} carries more than {MAX_QUESTIONS_PER_PARAGRAPH} questions"),
                        ));
                    }
                    specs.push(QuestionSpec {
                        paragraph,
                        question,
                    });
                }
                questions.insert(source.clone(), specs);
            }
        }
        Some(other) => issues.push(Issue::wrong_type(
            "questions",
            "an object of source -> [{paragraph, question}]",
            other,
        )),
    }
    questions
}

/// Interprets `sections`: an object of source → `[{paragraph,
/// section}]`, the same orphan/size rule as questions (no per-paragraph
/// count cap — ingest's batch format has none either).
fn interpret_sections(
    obj: &serde_json::Map<String, Value>,
    passages: &BTreeMap<String, String>,
    issues: &mut Vec<Issue>,
) -> BTreeMap<String, Vec<SectionSpec>> {
    let mut sections = BTreeMap::new();
    match obj.get("sections") {
        None | Some(Value::Null) => {}
        Some(Value::Object(map)) => {
            for (source, list) in map {
                let path = format!("sections['{source}']");
                let Some(array) = list.as_array() else {
                    issues.push(Issue::wrong_type(path, "an array", list));
                    continue;
                };
                check_orphaned_source(path.clone(), source, passages, issues);
                let mut specs = Vec::with_capacity(array.len());
                for (index, item) in array.iter().enumerate() {
                    let item_path = format!("{path}[{index}]");
                    let (paragraph, section) = interpret_paragraph_and_text(
                        item,
                        &item_path,
                        "section",
                        MAX_SECTION_BYTES,
                        issues,
                    );
                    specs.push(SectionSpec { paragraph, section });
                }
                sections.insert(source.clone(), specs);
            }
        }
        Some(other) => issues.push(Issue::wrong_type(
            "sections",
            "an object of source -> [{paragraph, section}]",
            other,
        )),
    }
    sections
}

/// Interprets `tags`: an object of source → `[tag]` (#167) — the same
/// orphan rule as questions/sections, plus the shared per-source count
/// cap and per-tag byte cap.
fn interpret_tags(
    obj: &serde_json::Map<String, Value>,
    passages: &BTreeMap<String, String>,
    issues: &mut Vec<Issue>,
) -> BTreeMap<String, Vec<String>> {
    let mut tags = BTreeMap::new();
    match obj.get("tags") {
        None | Some(Value::Null) => {}
        Some(Value::Object(map)) => {
            for (source, list) in map {
                let path = format!("tags['{source}']");
                let Some(array) = list.as_array() else {
                    issues.push(Issue::wrong_type(path, "an array of strings", list));
                    continue;
                };
                check_orphaned_source(path.clone(), source, passages, issues);
                if array.len() > MAX_TAGS_PER_SOURCE {
                    issues.push(Issue::over_limit(
                        path,
                        format!("at most {MAX_TAGS_PER_SOURCE} tags"),
                        format!("{} tags", array.len()),
                    ));
                }
                let mut values = Vec::with_capacity(array.len());
                for (index, item) in array.iter().enumerate() {
                    let item_path = format!("tags['{source}'][{index}]");
                    match item {
                        Value::String(text) if text.is_empty() => {
                            issues.push(Issue::empty(item_path));
                        }
                        Value::String(text) if text.len() > MAX_TAG_BYTES => {
                            issues.push(Issue::too_long(item_path, MAX_TAG_BYTES, text.len()));
                        }
                        Value::String(text) => values.push(text.clone()),
                        other => {
                            issues.push(Issue::wrong_type(item_path, "a non-empty string", other))
                        }
                    }
                }
                tags.insert(source.clone(), values);
            }
        }
        Some(other) => issues.push(Issue::wrong_type(
            "tags",
            "an object of source -> [tag]",
            other,
        )),
    }
    tags
}

/// Interprets `dates`: an object of source → epoch seconds (#167) — the
/// same orphan rule as questions/sections/tags.
fn interpret_dates(
    obj: &serde_json::Map<String, Value>,
    passages: &BTreeMap<String, String>,
    issues: &mut Vec<Issue>,
) -> BTreeMap<String, u64> {
    let mut dates = BTreeMap::new();
    match obj.get("dates") {
        None | Some(Value::Null) => {}
        Some(Value::Object(map)) => {
            for (source, value) in map {
                let path = format!("dates['{source}']");
                check_orphaned_source(path.clone(), source, passages, issues);
                match value {
                    Value::Number(number) => match number.as_u64() {
                        Some(epoch) => {
                            dates.insert(source.clone(), epoch);
                        }
                        None => issues.push(Issue::wrong_type(
                            path,
                            "a non-negative integer (epoch seconds)",
                            value,
                        )),
                    },
                    other => issues.push(Issue::wrong_type(
                        path,
                        "a non-negative integer (epoch seconds)",
                        other,
                    )),
                }
            }
        }
        Some(other) => issues.push(Issue::wrong_type(
            "dates",
            "an object of source -> epoch seconds",
            other,
        )),
    }
    dates
}

/// Interprets the `store_passages` request body as a lenient JSON walk
/// (issue #182), collecting every field's issues in one pass instead of
/// rejecting the whole request at the first bad value — mirroring, for
/// this REST write, the same collect-all discipline ADR 0001 §8 already
/// gives a retrying LLM's answer. The built-so-far request is discarded
/// the moment any issue is found, since the whole write is refused
/// together — `nothing_written`.
fn interpret_store_passages(value: &Value) -> Result<StorePassagesRequest, Vec<Issue>> {
    let Some(obj) = value.as_object() else {
        return Err(vec![Issue::wrong_type("", "an object", value)]);
    };
    let mut issues = Vec::new();
    let passages = interpret_passages(obj, &mut issues);
    let questions = interpret_questions(obj, &passages, &mut issues);
    let sections = interpret_sections(obj, &passages, &mut issues);
    let tags = interpret_tags(obj, &passages, &mut issues);
    let dates = interpret_dates(obj, &passages, &mut issues);
    if issues.is_empty() {
        Ok(StorePassagesRequest {
            passages,
            questions,
            sections,
            tags,
            dates,
        })
    } else {
        Err(issues)
    }
}

/// What a passage store accomplished. `stored` counts the batch (the
/// historical number, now named); the question and section tallies
/// report doc2query/section bookkeeping — a dropped question or
/// section named a paragraph that does not exist in the text it rode
/// in with, or (sections only) lost out to a later marker claiming the
/// same paragraph.
#[derive(Serialize)]
pub struct StoredPassages {
    pub stored: usize,
    pub questions_stored: usize,
    pub questions_dropped: usize,
    pub sections_stored: usize,
    pub sections_dropped: usize,
}

pub async fn store_passages(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(body): AppJson<Value>,
) -> Response {
    let started_at = Instant::now();
    // Refused before any tokenization or lock is taken, and before the
    // per-field walk below: nothing of an oversized batch is stored,
    // and there is no point diagnosing thousands of sources nothing
    // will ever write.
    if let Some(passages) = body.get("passages").and_then(Value::as_object)
        && passages.len() > MAX_PASSAGES_PER_REQUEST
    {
        return error(
            ErrorCode::OverLimit,
            format!(
                "batch of {} passages exceeds the per-request limit of \
                 {MAX_PASSAGES_PER_REQUEST}; split the store",
                passages.len()
            ),
            started_at,
        );
    }
    let request = match interpret_store_passages(&body) {
        Ok(request) => request,
        Err(issues) => {
            let (issues, total) = truncate_issues(issues);
            let message = collected_validation_message("the passage store", &issues, total);
            return validation_error(
                ErrorCode::InvalidArgument,
                message,
                RefusalDetail {
                    issues,
                    integrity: Some("nothing_written"),
                    retryable_after_correction: Some(true),
                    ..Default::default()
                },
                started_at,
            );
        }
    };
    let mut questions = request.questions;
    let mut sections = request.sections;
    let mut tags = request.tags;
    let mut dates = request.dates;
    let passages: BTreeMap<String, crate::passages::PassageSubmission> = request
        .passages
        .into_iter()
        .map(|(source, text)| {
            let questions = questions
                .remove(&source)
                .unwrap_or_default()
                .into_iter()
                .map(|spec| (spec.paragraph, spec.question))
                .collect();
            let sections = sections
                .remove(&source)
                .unwrap_or_default()
                .into_iter()
                .map(|spec| (spec.paragraph, spec.section))
                .collect();
            // `stored_at: None` on purpose: the HTTP path never
            // supplies a stamp — the store takes it once, at the
            // write (only import restores an existing one).
            let meta = crate::passages::SourceMeta {
                stored_at: None,
                date: dates.remove(&source),
                tags: tags.remove(&source).unwrap_or_default(),
            };
            (
                source,
                crate::passages::PassageSubmission {
                    text,
                    questions,
                    sections,
                    meta,
                },
            )
        })
        .collect();
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    // Off the async worker: the store fsyncs its log, and folding the
    // new paragraphs into a resident index tokenizes them.
    let outcome = tokio::task::block_in_place(|| state.store_passages(&name, passages));
    match outcome {
        None => not_found(&name, started_at),
        Some(Ok(outcome)) => {
            state.note_write(&name);
            ok(
                StoredPassages {
                    stored: outcome.stored,
                    questions_stored: outcome.questions_stored,
                    questions_dropped: outcome.questions_dropped,
                    sections_stored: outcome.sections_stored,
                    sections_dropped: outcome.sections_dropped,
                },
                started_at,
            )
        }
        // No error-kind counter for the quota arm: a refusal at the
        // declared ceiling is the policy working, not the server
        // failing — `taguru_storage_quota_refusals_total` counts it.
        Some(Err(crate::registry::PassagesWriteError::QuotaExceeded(message))) => {
            error(ErrorCode::StorageFull, message, started_at)
        }
        Some(Err(crate::registry::PassagesWriteError::Io(io_error))) => {
            state.metrics().record_error(ErrorKind::Io);
            error(
                ErrorCode::Internal,
                format!("passages could not be persisted: {io_error}"),
                started_at,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct HTTP callers still on the pre-#35 field name aren't broken by
    /// the rename to `paragraph`.
    #[test]
    fn citation_request_accepts_the_pre_35_index_field_name() {
        let request: CitationRequest =
            serde_json::from_value(serde_json::json!({"source": "s", "index": 3})).unwrap();
        assert_eq!(request.paragraph, 3);
    }

    /// `#[serde(alias)]` maps both names onto one field, not onto a
    /// "prefer paragraph" merge: sending both is a duplicate-field error,
    /// same as sending `paragraph` twice. The MCP path's `pick_with_alias`
    /// resolves a same-request clash by preference instead; direct HTTP
    /// callers get this stricter, but still well-defined, rejection.
    #[test]
    fn citation_request_rejects_both_names_at_once_as_a_duplicate_field() {
        let result: Result<CitationRequest, _> =
            serde_json::from_value(serde_json::json!({"source": "s", "paragraph": 1, "index": 2}));
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains("duplicate field"),
            "expected a duplicate-field error, got: {error}"
        );
    }

    /// An absent lane must OMIT its key, never serialize as null:
    /// lane consumers test key presence.
    #[test]
    fn lane_shapes_omit_absent_keys_rather_than_nulling_them() {
        let lanes = serde_json::to_value(PassageLanes {
            bm25: Some(LaneEvidence {
                rank: 1,
                score: 2.5,
            }),
            vector: None,
        })
        .unwrap();
        assert_eq!(
            lanes,
            serde_json::json!({"bm25": {"rank": 1, "score": 2.5}}),
            "an absent lane omits its key"
        );
    }
}
