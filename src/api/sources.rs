use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::Response;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use taguru::deadline::Deadline;

use crate::metrics::{ErrorKind, RetrievalCacheOp, SearchOp};
use crate::registry::{
    AppState, CachedRetrieval, CitationLookup, PassageExplainLookup, RetrievalKey, SemanticFill,
    SemanticServe,
};

use super::aliases::{KeysetQuery, keyset_bounds};
use super::recall::cross_targets;
use super::{
    AppJson, AppPath, AppQuery, CrossMatch, ErrorCode, Issue, MAX_LOCATOR_KIND_BYTES,
    MAX_LOCATOR_VALUE_BYTES, MAX_MATCH_LIMIT, MAX_NAME_BYTES, MAX_PASSAGES_PER_REQUEST,
    MAX_QUESTION_BYTES, MAX_QUESTIONS_PER_PARAGRAPH, MAX_SECTION_BYTES, MAX_TAG_BYTES,
    MAX_TAGS_PER_SOURCE, RefusalDetail, access_error, bounded_parallel_map, cache_and_serve,
    check_bounded_len, clamp, clamp_page, collected_validation_message, cross_job_panic,
    cross_search_concurrency, deadline_exceeded, empty, error, interpret_bounded_text, not_found,
    ok, overlong, oversized, replay_cached_search, search_log_enabled, truncate_issues,
    validation_error,
};

#[derive(Debug, Deserialize)]
pub struct LookupPassagesRequest {
    pub sources: Vec<String>,
}

/// The `empty`/`oversized` gate every source-shaped field goes through
/// before it reaches a lookup or write that would otherwise pay for a
/// disk read (or fsync) before failing to find it. Shared by
/// `lookup_passages`, `citation`, and `retract_source`.
fn invalid_source(source: &str, started_at: Instant) -> Option<Response> {
    if let Some(refusal) = empty("source", source, started_at) {
        return Some(refusal);
    }
    oversized("source", source, MAX_NAME_BYTES, started_at)
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
    for source in &request.sources {
        if let Some(refusal) = invalid_source(source, started_at) {
            return refusal;
        }
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
/// it stored none at all; `locator` is the typed citation locator (a
/// page/slide/sheet position, ADR 0007 §7, see `PassageRecord::
/// locator_for`) — independent of `section`, and `null` under the same
/// rule. Neither key is ever omitted, so callers can rely on both
/// always being present.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub text: String,
    pub source: String,
    pub section: Option<String>,
    pub locator: Option<crate::passages::Locator>,
}

pub async fn citation(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppJson(request): AppJson<CitationRequest>,
) -> Response {
    let started_at = Instant::now();
    if let Some(refusal) = invalid_source(&request.source, started_at) {
        return refusal;
    }
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
        Some(Ok(CitationLookup::Found {
            text,
            section,
            locator,
        })) => {
            state.note_read(&name, false);
            ok(
                Citation {
                    text,
                    source: request.source,
                    section,
                    locator,
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
    if let Some(refusal) = keyset_bounds(&query, started_at) {
        return refusal;
    }
    let limit = clamp_page(query.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT);
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

/// `?dry_run=true` (#437): report what the retraction WOULD do —
/// the same `{associations_touched, passage_removed}` shape — with
/// nothing written: no WAL op, no import marker, no audit line, no
/// usage bump. The same preview `/import?dry_run=true` already runs
/// before its own per-source replace, exposed on the standalone
/// entrance.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct RetractSourceQuery {
    pub dry_run: bool,
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
    AppQuery(query): AppQuery<RetractSourceQuery>,
    AppJson(request): AppJson<RetractSourceRequest>,
) -> Response {
    let started_at = Instant::now();
    // Same gate every other name-shaped write goes through (add_associations'
    // source, retract_association's subject/label/object): an empty or
    // oversized source would otherwise reach the lookup below unchecked,
    // paying for a marker fsync and a WAL fsync before failing to find it.
    if let Some(refusal) = invalid_source(&request.source, started_at) {
        return refusal;
    }
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    if query.dry_run {
        // Reads only — off the async worker like every store-loading
        // read, but with none of the write path's marker/WAL/audit.
        return match tokio::task::block_in_place(|| {
            state.retract_source_preview(&name, &request.source)
        }) {
            Err(failure) => access_error(&state, failure, &name, started_at),
            Ok((associations_touched, passage_removed)) => ok(
                RetractOutcome {
                    associations_touched,
                    passage_removed,
                },
                started_at,
            ),
        };
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
///
/// [`MAX_TAGS_PER_SOURCE`](super::MAX_TAGS_PER_SOURCE)'s own doc reads
/// "how many tags one search filter may name" — that is the EFFECTIVE
/// filter, so it is checked AFTER dedup (issue #620): naming the same
/// tag 33 times names one filter tag, not 33, and refusing it anyway
/// would reject a request whose stored filter is legal. The raw input
/// still has its own door before dedup — [`overlong`], the same
/// `MAX_INPUT_ITEMS` cap `cross_targets` applies to `contexts`/
/// `groups` — so an unbounded body cannot force this function to sort
/// and dedup an arbitrarily large list before refusing it.
pub(super) fn source_filter(
    tags: &[String],
    since: Option<u64>,
    until: Option<u64>,
    started_at: Instant,
) -> Result<Option<crate::passages::SourceFilter>, Box<Response>> {
    if let Some(refusal) = overlong("tags", tags.len(), started_at) {
        return Err(Box::new(refusal));
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
    if tags.len() > super::MAX_TAGS_PER_SOURCE {
        // `ErrorCode::OverLimit` (issue #623 finding 2), matching the
        // raw-input cap `overlong` applies just above (via the door at
        // this function's own top): both are "a list-shaped field
        // carries too many items," and a client branching on
        // `over_limit` to pick a retry strategy must not see one of the
        // two silently fall back to `invalid_argument`. `tags` is an
        // any-of filter (`SourceFilter::matches`, `src/passages.rs`),
        // so splitting the distinct set across two calls and unioning
        // the results client-side IS equivalent to one call with the
        // full set — `OverLimit`'s "split and resend" contract holds,
        // the same as `cross_targets`' own any-of `contexts`/`groups`
        // cap (`recall.rs`) already reusing `overlong`. Confirmed
        // against this exact constant's OTHER enforcement point too:
        // `interpret_tags` (below, a source's own stored tag list,
        // write side) already reports an over-cap count as
        // `Issue::over_limit`, not a type/argument issue — read and
        // write sides of `MAX_TAGS_PER_SOURCE` agreeing on the kind is
        // the more direct precedent. Not reused as `overlong(...)`
        // itself: that helper hardcodes `MAX_INPUT_ITEMS` (no cap
        // parameter) and its advice ("split the request") would read
        // oddly here — the message below already names the right fix
        // (name fewer distinct tags) more precisely than "split".
        return Err(Box::new(error(
            ErrorCode::OverLimit,
            format!(
                "{} distinct filter tags where at most {} may be named",
                tags.len(),
                super::MAX_TAGS_PER_SOURCE
            ),
            started_at,
        )));
    }
    if tags.is_empty() && since.is_none() && until.is_none() {
        return Ok(None);
    }
    Ok(Some(crate::passages::SourceFilter { tags, since, until }))
}

/// Resolves a graph-lane request's optional assertion-time window into
/// the source names it admits (ADR 0011 §4 steps 1–2). `Ok(None)` when
/// no window was asked for, so the caller stays on the exact unwindowed
/// path; the window contract itself is [`source_filter`]'s — the shared
/// validator, so the graph lanes and the passage lanes cannot drift on
/// what a legal window is (`since >= until` refused up front). The
/// metadata join runs off the async worker before `read_context`, the
/// same pre-resolution rule `hidden_label`/`schema_of` follow, and a
/// store that cannot be read answers like a schema that cannot be
/// loaded: a logged `Internal`, never a silently unwindowed result.
pub(super) fn resolve_window(
    state: &AppState,
    name: &str,
    since: Option<u64>,
    until: Option<u64>,
    started_at: Instant,
) -> Result<Option<std::collections::BTreeSet<String>>, Box<Response>> {
    let Some(filter) = source_filter(&[], since, until, started_at)? else {
        return Ok(None);
    };
    match tokio::task::block_in_place(|| state.window_source_names(name, &filter)) {
        None => Err(Box::new(not_found(name, started_at))),
        Some(Ok(names)) => Ok(Some(names)),
        Some(Err(io_error)) => {
            // ADR 0008 §7: never a span field named `error`; the detail
            // rides the message text, matching `query`'s schema-load
            // failure arm.
            tracing::warn!(context = %name, "window source join failed: {io_error}");
            state.metrics().record_error(ErrorKind::Load);
            Err(Box::new(error(
                ErrorCode::Internal,
                format!("context '{name}' source metadata could not be read — see server logs"),
                started_at,
            )))
        }
    }
}

/// One passage-search variant's full set of result-affecting
/// parameters — the request query plus everything else that can
/// change the page served. The exact cache key and the semantic
/// tier's sans-query bucket ([`Self::exact`], [`Self::sans_query`])
/// are both DERIVED from one value here, so a field added to a search
/// request reaches both automatically. Before this type, the two were
/// hand-written tuple literals with no shared type or test binding
/// them — a field added to one and not the other would let requests
/// that disagree on that field share a [`super::semantic_cache::SemanticBucket`],
/// and a rewrite would splice in the canonical's full params, serving
/// a page computed for a different limit/floor/filter (#602 item 1).
/// `None` (no filter) serializes as `null`, distinct from every real
/// filter.
#[derive(Clone, Copy, Serialize)]
struct PassageKeyParams<'a> {
    /// Distinguishes `search_passages` from `cross_search_passages`,
    /// which otherwise share every other field's shape.
    op: &'static str,
    /// `None` in the bucket half — see [`Self::sans_query`].
    #[serde(skip_serializing_if = "Option::is_none")]
    query: Option<&'a str>,
    limit: usize,
    semantic_floor: Option<f32>,
    filter: Option<(&'a [String], Option<u64>, Option<u64>)>,
}

impl<'a> PassageKeyParams<'a> {
    fn new(
        op: &'static str,
        query: &'a str,
        limit: usize,
        semantic_floor: Option<f32>,
        filter: &'a Option<crate::passages::SourceFilter>,
    ) -> Self {
        Self {
            op,
            query: Some(query),
            limit,
            semantic_floor,
            filter: filter
                .as_ref()
                .map(|filter| (filter.tags.as_slice(), filter.since, filter.until)),
        }
    }

    /// The exact key's full params, query included.
    fn exact(&self) -> Option<String> {
        serde_json::to_string(self).ok()
    }

    /// The semantic tier's bucket params: the same value with the
    /// query stripped, so the two serializations can never disagree
    /// about anything else (#602 item 1).
    fn sans_query(&self) -> Option<String> {
        serde_json::to_string(&Self {
            query: None,
            ..*self
        })
        .ok()
    }
}

/// One PARAGRAPH matched by passage search: the text lane, for
/// knowledge that never decomposed into triples. `paragraph` is its
/// position within the source (0-based, this split); `text` is that
/// paragraph alone — cite it, or dereference the whole source through
/// the lookup endpoint. `score` is the fused reciprocal-rank number
/// when the semantic lane ran, the raw BM25 score otherwise; `lanes`
/// carries each lane's own rank and raw score — evidence for the
/// reading LLM, the same posture as resolve's tiers.
#[derive(Debug, Serialize, Deserialize)]
pub struct PassageHit {
    pub source: String,
    pub paragraph: u32,
    pub score: f32,
    pub text: String,
    pub lanes: PassageLanes,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PassageLanes {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25: Option<LaneEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<LaneEvidence>,
}

/// Where one lane put this hit: 1-based rank within the lane's own
/// candidate pool, and that lane's raw score (BM25, or cosine).
#[derive(Debug, Serialize, Deserialize)]
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
#[derive(Debug, Serialize, Deserialize)]
pub struct LanePlan {
    pub ran: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub floor: Option<f32>,
}

impl LanePlan {
    pub(crate) fn ran() -> Self {
        Self {
            ran: true,
            reason: None,
            floor: None,
        }
    }

    pub(crate) fn skipped(reason: impl Into<String>) -> Self {
        Self {
            ran: false,
            reason: Some(reason.into()),
            floor: None,
        }
    }
}

/// The two reasons neither lane ran at all — before either one could
/// start, so there is nothing lane-specific to say (ADR 0006 §10:
/// evidence assembly's own single-lane `plan.lanes.passages` account
/// shares this exact wording via [`SearchContextPlan::of`]'s own use
/// of them, so the two surfaces cannot drift apart in prose).
pub(crate) const NO_QUERY_TERMS_REASON: &str = "the query yields no searchable terms";
pub(crate) const ZERO_LIMIT_REASON: &str = "the requested limit is 0";

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

        let both = |reason: &'static str| SearchLanesPlan {
            bm25: LanePlan::skipped(reason),
            vector: LanePlan::skipped(reason),
        };
        let lanes = match lanes {
            PassageSearchLanes::NoQueryTerms => both(NO_QUERY_TERMS_REASON),
            PassageSearchLanes::ZeroLimit => both(ZERO_LIMIT_REASON),
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

/// [`passage_search_cache_probe`]'s outcome recorder gets called with:
/// which of the two ways a probe can answer directly fired, so a
/// caller that tracks cache-outcome span attributes (ADR 0008 §6)
/// knows which to record.
enum CacheProbeHit {
    Exact,
    Semantic,
}

/// [`passage_search_cache_probe`]'s result: it already answered the
/// request, or nothing did and the caller's own fresh search should
/// run, carrying whatever [`SemanticFill`] the probe warmed (if any)
/// to file alongside its own result.
enum CacheProbe {
    Answered(Response),
    Fresh(Option<SemanticFill>),
}

/// The lookup → semantic-probe → fill block `search_passages` and
/// `cross_search_passages` both open with, character-for-character
/// identical but for the search-log line's one differing field name
/// (issue #605): an exact-cache hit or a semantic-tier hit answers
/// immediately; otherwise the caller's own fresh search must run.
/// `log_hit`/`log_semantic_hit` stay per-caller because `tracing`
/// field names are macro-time identifiers, not runtime strings, so
/// `context = %name` and `contexts = %targets.join(",")` cannot share
/// one call; `on_hit` likewise stays per-caller because only
/// `search_passages` carries a `taguru.passage_search` span to
/// attribute a cache outcome onto — `cross_search_passages` has none
/// today (found while extracting this block, not fixed here: giving
/// cross searches their own span is a bigger design question than
/// this duplication cleanup covers).
#[allow(clippy::too_many_arguments)]
fn passage_search_cache_probe(
    state: &AppState,
    key: Option<&RetrievalKey>,
    key_params: &PassageKeyParams,
    query: &str,
    deadline: Deadline,
    started_at: Instant,
    log_hit: impl FnOnce(&CachedRetrieval),
    log_semantic_hit: impl FnOnce(&SemanticServe),
    on_hit: impl FnOnce(CacheProbeHit),
) -> CacheProbe {
    let Some(key) = key else {
        return CacheProbe::Fresh(None);
    };
    if let Some(found) = state.retrieval_lookup(key) {
        replay_cached_search(state, key, &found);
        if search_log_enabled() {
            log_hit(&found);
        }
        on_hit(CacheProbeHit::Exact);
        return CacheProbe::Answered(ok(found.payload.as_ref(), started_at));
    }
    // The semantic tier (see `semantic_retrieval`): the bucket is the
    // key params with the query stripped, so equivalence can only pair
    // requests that agree on everything else — the filter included.
    // Blocking section — the probe may pay the query embedding the
    // fresh search below would otherwise pay (same cue cache, one
    // provider call either way).
    if let Some(sans_query) = key_params.sans_query()
        && let Some(probe) = tokio::task::block_in_place(|| {
            state.semantic_retrieval(key, &sans_query, query, deadline)
        })
    {
        if let Some(served) = probe.served {
            replay_cached_search(state, key, &served.value);
            if search_log_enabled() {
                log_semantic_hit(&served);
            }
            on_hit(CacheProbeHit::Semantic);
            return CacheProbe::Answered(ok(served.value.payload.as_ref(), started_at));
        }
        return CacheProbe::Fresh(Some(SemanticFill {
            params: sans_query,
            query: query.to_string(),
            embedding: probe.embedding,
        }));
    }
    CacheProbe::Fresh(None)
}

/// Tallies [`Metrics::record_passage_hit`]'s lane split across a batch
/// of SERVED hits — calling it once per hit and returning the same
/// `[bm25_only, both_lanes, vector_only]` shape `cache_and_serve`
/// stores. Callers MUST feed only hits that will actually reach the
/// client (issue #621): the metric and its Prometheus counter
/// (`taguru_passage_lane_contributions_total`) are documented as
/// counting served hits, not everything a lane found before
/// truncation or a cross-context merge. `communities.rs`'s
/// `community_hits` truncates its own ranked list before tallying for
/// the same reason; this helper exists so `search_passages` and
/// `cross_search_passages` don't each reimplement the three-way lane
/// match (and so both get the same unit-test coverage of it).
fn record_lane_hits(
    metrics: &crate::metrics::Metrics,
    hits: impl IntoIterator<Item = (bool, bool)>,
) -> [u64; 3] {
    let mut lane_hits = [0u64; 3];
    for (bm25, vector) in hits {
        metrics.record_passage_hit(bm25, vector);
        match (bm25, vector) {
            (true, false) => lane_hits[0] += 1,
            (true, true) => lane_hits[1] += 1,
            (false, true) => lane_hits[2] += 1,
            (false, false) => {}
        }
    }
    lane_hits
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
    // One params value the exact key and the semantic bucket below
    // both derive from — see `PassageKeyParams` for why that sharing
    // is load-bearing.
    let key_params = PassageKeyParams::new(
        "search_passages",
        &request.query,
        limit,
        request.semantic_floor,
        &filter,
    );
    // One span for the whole handler — cache hits included, so a hit
    // is a childless span of its own rather than invisible (ADR 0008
    // §5, §6). Synchronous for the rest of this function (every I/O
    // point below is a `block_in_place` closure, never a real
    // `.await`), so holding the guard across it is correct.
    let span = crate::trace::span!(
        "taguru.passage_search",
        otel.kind = "internal",
        taguru.op = SearchOp::SearchPassages.as_str(),
        taguru.limit = limit,
        taguru.cache.result = tracing::field::Empty,
        taguru.cache.semantic = tracing::field::Empty,
        taguru.search.lanes = tracing::field::Empty,
        taguru.search.vector.outcome = tracing::field::Empty,
        taguru.passage.hit_count = tracing::field::Empty,
        taguru.passage.bm25_only = tracing::field::Empty,
        taguru.passage.both_lanes = tracing::field::Empty,
        taguru.passage.vector_only = tracing::field::Empty,
        taguru.filter.eligible = tracing::field::Empty,
        taguru.filter.total = tracing::field::Empty,
    );
    let _entered = span.enter();
    // Minted before the search — see `retrieval_key`. The raw
    // `semantic_floor` goes in unclamped: two spellings of one
    // effective floor just occupy two entries, which is only a hit-rate
    // cost, never a correctness one.
    let key = state.retrieval_key(
        RetrievalCacheOp::SearchPassages,
        std::slice::from_ref(&name),
        key_params.exact(),
    );
    let semantic_fill = match passage_search_cache_probe(
        &state,
        key.as_ref(),
        &key_params,
        &request.query,
        deadline,
        started_at,
        |found| {
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
        },
        |served| {
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
        },
        |hit| match hit {
            // A hit answers with THIS span and no lane children — that
            // zero-duration childless span is the signal (ADR 0008
            // §6.1).
            CacheProbeHit::Exact => {
                span.record("taguru.cache.result", "hit");
                tracing::info!(taguru.reason = "retrieval_cache_hit", "taguru.cache");
            }
            // The exact tier missed (we are past that early return)
            // but the semantic tier served — the two outcomes are
            // independent attributes on purpose (ADR 0008 §6).
            // `semantic_retrieval` already emitted its own
            // `taguru.cache` event for the outcome.
            CacheProbeHit::Semantic => {
                span.record("taguru.cache.result", "miss");
                span.record(
                    "taguru.cache.semantic",
                    crate::metrics::SemanticCacheOutcome::Hit.as_str(),
                );
            }
        },
    ) {
        CacheProbe::Answered(response) => return response,
        CacheProbe::Fresh(fill) => fill,
    };
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
        // The client answer stays a timeout either way (issue #620):
        // only the log gets the underlying io::Error, so a real disk
        // fault isn't indistinguishable from an ordinary budget cut.
        Some(Err(io_error)) if deadline.expired() || crate::api::injected_deadline_race() => {
            tracing::warn!(kind = ?io_error.kind(), "passage read failed under a spent budget");
            deadline_exceeded(started_at)
        }
        Some(Err(io_error)) => passages_unreadable(&state, io_error, started_at),
        Some(Ok(found)) => {
            state.note_search(SearchOp::SearchPassages, &name, found.hits.is_empty());
            let target_empty = vec![found.hits.is_empty()];
            let lane_hits = record_lane_hits(
                state.metrics(),
                found
                    .hits
                    .iter()
                    .map(|hit| (hit.bm25.is_some(), hit.vector.is_some())),
            );
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
            // Both tiers missed (an exact hit or a semantic hit would
            // already have returned above) — this is a fresh compute.
            span.record("taguru.cache.result", "miss");
            span.record("taguru.search.lanes", found.lanes.code());
            if let crate::registry::PassageSearchLanes::Ran { vector } = &found.lanes {
                span.record("taguru.search.vector.outcome", vector.code());
            }
            span.record("taguru.passage.hit_count", found.hits.len());
            span.record("taguru.passage.bm25_only", lane_hits[0]);
            span.record("taguru.passage.both_lanes", lane_hits[1]);
            span.record("taguru.passage.vector_only", lane_hits[2]);
            if let Some(filter_report) = found.filter {
                span.record("taguru.filter.eligible", filter_report.eligible);
                span.record("taguru.filter.total", filter_report.total);
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
/// (pool caps included), not read off the unbounded ranking. `None`
/// alone cannot tell "never ranked at all" from "the probe exhausted
/// its search space without reaching it" — `limit_to_reach_reason`
/// names the latter (`"unreachable"`) when it applies; the former
/// never reaches this struct with `rank` set, so it needs no reason
/// of its own (#601 item 4).
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit_to_reach_reason: Option<&'static str>,
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
        use crate::registry::{LimitToReach, VectorLaneReport};

        // Wire shape once, up front: `limit_to_reach` moving out of
        // `explanation` here (it is not `Copy` — three distinct
        // endings, #601 item 4) would otherwise conflict with the
        // struct build further down, which needs the same value.
        let (limit_to_reach, limit_to_reach_reason) = match explanation.limit_to_reach {
            LimitToReach::At(limit) => (Some(limit), None),
            LimitToReach::NotRanked => (None, None),
            LimitToReach::Unreachable => (None, Some("unreachable")),
        };

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
                let reach = match limit_to_reach {
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
                limit_to_reach,
                limit_to_reach_reason,
            }),
        }
    }
}

/// `POST /contexts/{name}/sources/search/explain` — one call instead
/// of "orchestrate four endpoints and cross-reference by hand": name
/// the query and the source (optionally the paragraph) you expected to
/// see, get the first verdict that applies with its evidence. Runs the
/// same lanes the search runs, read-only — one unbounded sweep per
/// lane (the vector lane's own ANN-vs-exact choice included, so a
/// large corpus is not swept exactly just because this is explain),
/// then O(log(raw row count)) reruns of the ranking alone to verify
/// `limit_to_reach`; the serve boundary is recomputed exactly as
/// `sources/search` computes it, so the two cannot disagree.
///
/// Counts as a READ (`state.note_read`) but not a SEARCH — no
/// `note_search`, no `record_passage_hit`, no `taguru::search` log
/// line (issue #621's finding 2, deliberate). `resolve`'s
/// `explain_resolve_verdict` documents the same rule; see
/// [`crate::api::resolve::ResolveExplanation`] for why.
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
        // refused to start once the budget was already gone — logged,
        // not discarded (issue #620), the same reasoning as there.
        Some(Err(io_error)) if deadline.expired() || crate::api::injected_deadline_race() => {
            tracing::warn!(kind = ?io_error.kind(), "passage read failed under a spent budget");
            deadline_exceeded(started_at)
        }
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
/// [`cross_search_concurrency`] — with the retrieval cache enabled (the
/// default), a probe warms the cue cache before the fan-out starts, so
/// only that probe pays for the query embedding. With the cache
/// disabled, there is no probe to warm anything, so up to that many
/// targets may each pay for the query embedding independently; the
/// cache is still the single source of truth when it exists (a
/// `Mutex`), so even the disabled-cache case is wasted provider calls,
/// not a correctness risk.
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
    // One params value the exact key and the semantic bucket below
    // both derive from — see `PassageKeyParams` for why that sharing
    // is load-bearing.
    let key_params = PassageKeyParams::new(
        "cross_search_passages",
        &request.query,
        limit,
        request.semantic_floor,
        &filter,
    );
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
        key_params.exact(),
    );
    // The semantic tier is keyed on the resolved target list like the
    // exact key above. A side effect worth having: the probe warms the
    // cue cache before the fan-out, so a cold cross search no longer
    // has up to `permits` targets each paying for the same query
    // embedding.
    let semantic_fill = match passage_search_cache_probe(
        &state,
        key.as_ref(),
        &key_params,
        &request.query,
        deadline,
        started_at,
        |found| {
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
        },
        |served| {
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
        },
        // No span to record a cache outcome onto (issue #605): unlike
        // `search_passages`, this handler carries none today.
        |_hit| {},
    ) {
        CacheProbe::Answered(response) => return response,
        CacheProbe::Fresh(fill) => fill,
    };
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
    let fetched = match bounded_parallel_map(targets.len(), permits, move |index| {
        job_state.search_passages(
            &owned_targets[index],
            &query,
            limit,
            semantic_floor,
            job_filter.as_ref(),
            deadline,
        )
    })
    .await
    {
        Ok(fetched) => fetched,
        Err(panicked) => {
            return cross_job_panic(&state, &targets[panicked.index], started_at);
        }
    };

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
    for (index, outcome) in fetched.into_iter().enumerate() {
        let name = &targets[index];
        match outcome {
            None => return not_found(name, started_at),
            // Logged, not discarded (issue #620): the client still
            // gets a timeout, but the underlying io::Error survives in
            // the log instead of vanishing behind it.
            Some(Err(io_error)) if deadline.expired() || crate::api::injected_deadline_race() => {
                tracing::warn!(kind = ?io_error.kind(), "passage read failed under a spent budget");
                return deadline_exceeded(started_at);
            }
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
    // Tallied AFTER the final cut, against the merged/truncated pool —
    // not per-target inside the loop above (issue #621): each target
    // can return up to `limit` hits, so counting before the merge
    // could over-report served hits by up to a factor of `targets.len()`
    // relative to what the client actually receives.
    let lane_hits = record_lane_hits(
        state.metrics(),
        pool.iter()
            .map(|(_, _, hit)| (hit.bm25.is_some(), hit.vector.is_some())),
    );
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
    /// Typed citation locators (ADR 0007 §7), per source in THIS
    /// request — same orphan rule and wholesale-replace semantics as
    /// `sections`, but independent of it: a locator does not extend to
    /// the next paragraph.
    pub locators: BTreeMap<String, Vec<LocatorSpec>>,
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

#[derive(Debug)]
pub struct LocatorSpec {
    pub paragraph: u32,
    pub locator: crate::passages::Locator,
}

/// Reads the `paragraph` field alone — the index-only half of
/// [`interpret_paragraph_and_text`], factored out so `locators` (whose
/// payload is a nested `{kind, value}` object, not a single bounded-text
/// field) can reuse the same paragraph-index validation.
fn interpret_paragraph(
    obj: &serde_json::Map<String, Value>,
    path: &str,
    issues: &mut Vec<Issue>,
) -> u32 {
    match obj.get("paragraph") {
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
    }
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
    let paragraph = interpret_paragraph(obj, path, issues);
    let text = interpret_bounded_text(obj, text_key, path, text_cap, issues);
    (paragraph, text)
}

/// One `{paragraph, locator: {kind, value}}` item (ADR 0007 §7.1) —
/// `locators`' own shape, since a locator's payload is a nested object
/// rather than the single bounded-text field `interpret_paragraph_and_text`
/// handles for questions/sections. Always returns a full spec (empty
/// strings where a field failed validation), matching the collect-all
/// convention every other field here follows.
fn interpret_locator_item(item: &Value, path: &str, issues: &mut Vec<Issue>) -> LocatorSpec {
    let Some(obj) = item.as_object() else {
        issues.push(Issue::wrong_type(path, "an object", item));
        return LocatorSpec {
            paragraph: 0,
            locator: crate::passages::Locator::default(),
        };
    };
    let paragraph = interpret_paragraph(obj, path, issues);
    let locator_path = format!("{path}.locator");
    let locator = match obj.get("locator") {
        None | Some(Value::Null) => {
            issues.push(Issue::missing(locator_path, "an object {kind, value}"));
            crate::passages::Locator::default()
        }
        Some(Value::Object(locator_obj)) => crate::passages::Locator {
            kind: interpret_bounded_text(
                locator_obj,
                "kind",
                &locator_path,
                MAX_LOCATOR_KIND_BYTES,
                issues,
            ),
            value: interpret_bounded_text(
                locator_obj,
                "value",
                &locator_path,
                MAX_LOCATOR_VALUE_BYTES,
                issues,
            ),
        },
        Some(other) => {
            issues.push(Issue::wrong_type(
                locator_path,
                "an object {kind, value}",
                other,
            ));
            crate::passages::Locator::default()
        }
    };
    LocatorSpec { paragraph, locator }
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
                check_bounded_len(
                    source,
                    format!("passages['{source}']"),
                    MAX_NAME_BYTES,
                    issues,
                );
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

/// Interprets `locators`: an object of source → `[{paragraph, locator:
/// {kind, value}}]` (ADR 0007 §7) — same orphan rule as questions/
/// sections, independent of `sections` (a locator never extends to the
/// next paragraph).
fn interpret_locators(
    obj: &serde_json::Map<String, Value>,
    passages: &BTreeMap<String, String>,
    issues: &mut Vec<Issue>,
) -> BTreeMap<String, Vec<LocatorSpec>> {
    let mut locators = BTreeMap::new();
    match obj.get("locators") {
        None | Some(Value::Null) => {}
        Some(Value::Object(map)) => {
            for (source, list) in map {
                let path = format!("locators['{source}']");
                let Some(array) = list.as_array() else {
                    issues.push(Issue::wrong_type(path, "an array", list));
                    continue;
                };
                check_orphaned_source(path.clone(), source, passages, issues);
                let mut specs = Vec::with_capacity(array.len());
                for (index, item) in array.iter().enumerate() {
                    let item_path = format!("{path}[{index}]");
                    specs.push(interpret_locator_item(item, &item_path, issues));
                }
                locators.insert(source.clone(), specs);
            }
        }
        Some(other) => issues.push(Issue::wrong_type(
            "locators",
            "an object of source -> [{paragraph, locator: {kind, value}}]",
            other,
        )),
    }
    locators
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
                        Value::String(text) => {
                            if check_bounded_len(text, item_path, MAX_TAG_BYTES, issues) {
                                values.push(text.clone());
                            }
                        }
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
    let locators = interpret_locators(obj, &passages, &mut issues);
    let tags = interpret_tags(obj, &passages, &mut issues);
    let dates = interpret_dates(obj, &passages, &mut issues);
    if issues.is_empty() {
        Ok(StorePassagesRequest {
            passages,
            questions,
            sections,
            locators,
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
    pub locators_stored: usize,
    pub locators_dropped: usize,
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
                    issues_total: Some(total),
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
    let mut locators = request.locators;
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
            let locators = locators
                .remove(&source)
                .unwrap_or_default()
                .into_iter()
                .map(|spec| (spec.paragraph, spec.locator))
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
                    locators,
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
                    locators_stored: outcome.locators_stored,
                    locators_dropped: outcome.locators_dropped,
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
    use crate::registry::ContextMeta;

    /// A fresh, on-disk-backed [`AppState`], its data directory
    /// alongside it so tests can reach into the on-disk shape directly
    /// (the corruption trick below needs to write straight to a
    /// passages snapshot file `AppState` has no API for).
    fn scratch_state(tag: &str) -> (AppState, std::path::PathBuf) {
        let dir =
            std::env::temp_dir().join(format!("taguru-api-sources-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        (state, dir)
    }

    /// Forces `context`'s NEXT passage read to fail with a genuine
    /// `io::Error` (issue #620): writes bytes `PassageStore::load`
    /// cannot parse as a snapshot to the exact path it reads on a
    /// context's first passage touch. Deterministic and needs no prior
    /// successful load to "then corrupt" — the snapshot file does not
    /// exist yet for a context that has never stored a passage, so
    /// this is simply that file's first-ever write.
    fn corrupt_passages_snapshot(dir: &std::path::Path, context: &str) {
        let stem = crate::registry::file_stem(context);
        let path = crate::registry::passages_path(dir, &stem);
        std::fs::write(path, b"not a valid passages snapshot").unwrap();
    }

    /// issue #620 (所見3): `search_passages` must classify a genuine
    /// io::Error correctly against the real deadline — logged either
    /// way, but the CLIENT response must still pick the right one of
    /// "disk fault" (500) vs. "budget spent" (408). Both directions of
    /// the match guard are exercised: the corrupted snapshot alone
    /// (deadline untouched) proves the guard does not fire when it
    /// should not; `expire_deadline_race` then proves it fires when it
    /// should, without racing a real `Duration`.
    #[tokio::test(flavor = "multi_thread")]
    async fn search_passages_reports_a_genuine_io_error_as_unreadable_not_timeout() {
        let (state, dir) = scratch_state("search-io-error");
        state.create("sake", ContextMeta::default()).unwrap();
        corrupt_passages_snapshot(&dir, "sake");

        let request = SearchPassagesRequest {
            query: "AAA".to_string(),
            limit: None,
            semantic_floor: None,
            tags: Vec::new(),
            since: None,
            until: None,
        };
        let response = search_passages(
            State(state),
            AppPath("sake".to_string()),
            axum::Extension(Deadline::unbounded()),
            AppJson(request),
        )
        .await;

        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            body["code"],
            ErrorCode::Internal.as_str(),
            "an unexpired deadline must never reclassify a real disk fault as a \
             timeout — {body}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn search_passages_reclassifies_an_io_error_as_timeout_once_the_budget_is_spent() {
        let (state, dir) = scratch_state("search-io-error-timeout");
        state.create("sake", ContextMeta::default()).unwrap();
        corrupt_passages_snapshot(&dir, "sake");
        crate::api::expire_deadline_race();

        let request = SearchPassagesRequest {
            query: "AAA".to_string(),
            limit: None,
            semantic_floor: None,
            tags: Vec::new(),
            since: None,
            until: None,
        };
        let response = search_passages(
            State(state),
            AppPath("sake".to_string()),
            axum::Extension(Deadline::unbounded()),
            AppJson(request),
        )
        .await;

        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            body["code"],
            ErrorCode::Timeout.as_str(),
            "a budget spent by the time the read failed must reclassify as a \
             timeout, matching the single-context error path's own rule — {body}"
        );
    }

    /// issue #620 (所見3): `explain_search_passages`'s own twin of the
    /// two tests above.
    #[tokio::test(flavor = "multi_thread")]
    async fn explain_search_passages_reports_a_genuine_io_error_as_unreadable_not_timeout() {
        let (state, dir) = scratch_state("explain-io-error");
        state.create("sake", ContextMeta::default()).unwrap();
        corrupt_passages_snapshot(&dir, "sake");

        let request = ExplainSearchRequest {
            query: "AAA".to_string(),
            source: "a.md".to_string(),
            paragraph: None,
            limit: None,
            semantic_floor: None,
            tags: Vec::new(),
            since: None,
            until: None,
        };
        let response = explain_search_passages(
            State(state),
            AppPath("sake".to_string()),
            axum::Extension(Deadline::unbounded()),
            AppJson(request),
        )
        .await;

        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["code"], ErrorCode::Internal.as_str(), "{body}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn explain_search_passages_reclassifies_an_io_error_as_timeout_once_the_budget_is_spent()
    {
        let (state, dir) = scratch_state("explain-io-error-timeout");
        state.create("sake", ContextMeta::default()).unwrap();
        corrupt_passages_snapshot(&dir, "sake");
        crate::api::expire_deadline_race();

        let request = ExplainSearchRequest {
            query: "AAA".to_string(),
            source: "a.md".to_string(),
            paragraph: None,
            limit: None,
            semantic_floor: None,
            tags: Vec::new(),
            since: None,
            until: None,
        };
        let response = explain_search_passages(
            State(state),
            AppPath("sake".to_string()),
            axum::Extension(Deadline::unbounded()),
            AppJson(request),
        )
        .await;

        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["code"], ErrorCode::Timeout.as_str(), "{body}");
    }

    /// issue #620 (所見3): the other direction of the same guard — an
    /// unexpired deadline must never reclassify a target's genuine
    /// disk fault as a timeout.
    #[tokio::test(flavor = "multi_thread")]
    async fn cross_search_passages_reports_a_genuine_io_error_as_unreadable_not_timeout() {
        let (state, dir) = scratch_state("cross-search-io-error");
        state.create("sake", ContextMeta::default()).unwrap();
        corrupt_passages_snapshot(&dir, "sake");

        let request = CrossSearchPassagesRequest {
            contexts: vec!["sake".to_string()],
            groups: Vec::new(),
            query: "AAA".to_string(),
            limit: None,
            semantic_floor: None,
            tags: Vec::new(),
            since: None,
            until: None,
        };
        let response = cross_search_passages(
            State(state),
            None,
            None,
            axum::Extension(Deadline::unbounded()),
            AppJson(request),
        )
        .await;

        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["code"], ErrorCode::Internal.as_str(), "{body}");
    }

    /// issue #620 (所見3, 所見1's non-panicking twin): `cross_search_passages`
    /// must reclassify a target's genuine io::Error as a timeout once
    /// the budget is spent, same rule as the single-context handler.
    #[tokio::test(flavor = "multi_thread")]
    async fn cross_search_passages_reclassifies_an_io_error_as_timeout_once_the_budget_is_spent() {
        let (state, dir) = scratch_state("cross-search-io-error-timeout");
        state.create("sake", ContextMeta::default()).unwrap();
        corrupt_passages_snapshot(&dir, "sake");
        crate::api::expire_deadline_race();

        let request = CrossSearchPassagesRequest {
            contexts: vec!["sake".to_string()],
            groups: Vec::new(),
            query: "AAA".to_string(),
            limit: None,
            semantic_floor: None,
            tags: Vec::new(),
            since: None,
            until: None,
        };
        let response = cross_search_passages(
            State(state),
            None,
            None,
            axum::Extension(Deadline::unbounded()),
            AppJson(request),
        )
        .await;

        let bytes = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["code"], ErrorCode::Timeout.as_str(), "{body}");
    }

    /// issue #620 (所見4): the tag-count cap must gate the EFFECTIVE
    /// filter, not the raw input — 33 spellings of one tag store as
    /// one filter tag, well under `MAX_TAGS_PER_SOURCE`, and refusing
    /// it anyway would reject a request whose stored filter is legal.
    #[test]
    fn source_filter_accepts_the_same_tag_repeated_past_the_cap() {
        let tags: Vec<String> = (0..super::super::MAX_TAGS_PER_SOURCE + 1)
            .map(|_| "dup".to_string())
            .collect();
        let filter = source_filter(&tags, None, None, Instant::now())
            .unwrap()
            .expect("a non-empty tag list must produce Some filter");
        assert_eq!(filter.tags, vec!["dup".to_string()]);
    }

    /// The cap is inclusive: exactly `MAX_TAGS_PER_SOURCE` distinct
    /// tags is still a legal filter, only one past it refuses — pins
    /// the `>` boundary the cap check runs on.
    #[test]
    fn source_filter_accepts_exactly_the_cap_in_distinct_tags() {
        let tags: Vec<String> = (0..super::super::MAX_TAGS_PER_SOURCE)
            .map(|index| format!("tag{index}"))
            .collect();
        let filter = source_filter(&tags, None, None, Instant::now())
            .unwrap()
            .expect("exactly the cap in distinct tags must still be a legal filter");
        assert_eq!(filter.tags.len(), super::super::MAX_TAGS_PER_SOURCE);
    }

    /// The cap still refuses a filter whose DISTINCT tag count exceeds
    /// it — dedup must not become a loophole for an unbounded filter.
    /// `OverLimit`, not `InvalidArgument` (issue #623 finding 2) — the
    /// same code the raw-input door below returns, since both are a
    /// list-shaped field carrying too many items.
    #[tokio::test]
    async fn source_filter_refuses_too_many_distinct_tags() {
        let tags: Vec<String> = (0..super::super::MAX_TAGS_PER_SOURCE + 1)
            .map(|index| format!("tag{index}"))
            .collect();
        let refusal = source_filter(&tags, None, None, Instant::now()).unwrap_err();
        let bytes = axum::body::to_bytes(refusal.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["code"], ErrorCode::OverLimit.as_str());
        assert!(
            body["error"]
                .as_str()
                .unwrap()
                .contains(&format!("{}", super::super::MAX_TAGS_PER_SOURCE + 1)),
            "{body}"
        );
    }

    /// The raw-input door still refuses an unbounded body before dedup
    /// ever runs — the same `MAX_INPUT_ITEMS` cap `cross_targets`
    /// applies to `contexts`/`groups`.
    #[tokio::test]
    async fn source_filter_refuses_an_oversized_raw_tag_list_before_dedup() {
        let tags: Vec<String> = (0..crate::api::MAX_INPUT_ITEMS + 1)
            .map(|index| format!("tag{index}"))
            .collect();
        let refusal = source_filter(&tags, None, None, Instant::now()).unwrap_err();
        let bytes = axum::body::to_bytes(refusal.into_body(), 4096)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["code"], ErrorCode::OverLimit.as_str());
    }

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

    /// The semantic tier's bucket must be exactly the exact key's
    /// params with the query field removed — never a separately
    /// hand-written shape that could drift from it (#602 item 1).
    #[test]
    fn the_semantic_bucket_params_are_the_exact_params_minus_the_query() {
        let filter = Some(crate::passages::SourceFilter {
            tags: vec!["a".to_string(), "b".to_string()],
            since: Some(1),
            until: Some(2),
        });
        let params = PassageKeyParams::new("search_passages", "the query", 7, Some(0.5), &filter);
        let exact: serde_json::Value = serde_json::from_str(&params.exact().unwrap()).unwrap();
        let sans_query: serde_json::Value =
            serde_json::from_str(&params.sans_query().unwrap()).unwrap();
        assert_eq!(
            exact.get("query").and_then(Value::as_str),
            Some("the query"),
            "the exact key must actually carry the query"
        );
        let mut exact_minus_query = exact.as_object().unwrap().clone();
        exact_minus_query.remove("query");
        assert_eq!(
            Value::Object(exact_minus_query),
            sans_query,
            "the bucket must be exactly the exact params minus the query"
        );
    }

    /// Every result-affecting field must move the semantic bucket —
    /// otherwise two requests that disagree on that field could share
    /// a bucket, and a rewrite would serve one's page for the other's
    /// parameters (#602 item 1).
    #[test]
    fn every_result_affecting_field_moves_the_bucket() {
        let base_filter = Some(crate::passages::SourceFilter {
            tags: vec!["a".to_string()],
            since: Some(1),
            until: Some(2),
        });
        let base = PassageKeyParams::new("search_passages", "q", 5, Some(0.5), &base_filter)
            .sans_query()
            .unwrap();

        let other_op =
            PassageKeyParams::new("cross_search_passages", "q", 5, Some(0.5), &base_filter)
                .sans_query()
                .unwrap();
        assert_ne!(base, other_op, "op must move the bucket");

        let other_limit = PassageKeyParams::new("search_passages", "q", 6, Some(0.5), &base_filter)
            .sans_query()
            .unwrap();
        assert_ne!(base, other_limit, "limit must move the bucket");

        let other_floor = PassageKeyParams::new("search_passages", "q", 5, Some(0.6), &base_filter)
            .sans_query()
            .unwrap();
        assert_ne!(base, other_floor, "semantic_floor must move the bucket");

        let other_tags_filter = Some(crate::passages::SourceFilter {
            tags: vec!["c".to_string()],
            since: Some(1),
            until: Some(2),
        });
        let other_tags =
            PassageKeyParams::new("search_passages", "q", 5, Some(0.5), &other_tags_filter)
                .sans_query()
                .unwrap();
        assert_ne!(base, other_tags, "filter tags must move the bucket");

        let other_since_filter = Some(crate::passages::SourceFilter {
            tags: vec!["a".to_string()],
            since: Some(9),
            until: Some(2),
        });
        let other_since =
            PassageKeyParams::new("search_passages", "q", 5, Some(0.5), &other_since_filter)
                .sans_query()
                .unwrap();
        assert_ne!(base, other_since, "filter since must move the bucket");

        let other_until_filter = Some(crate::passages::SourceFilter {
            tags: vec!["a".to_string()],
            since: Some(1),
            until: Some(9),
        });
        let other_until =
            PassageKeyParams::new("search_passages", "q", 5, Some(0.5), &other_until_filter)
                .sans_query()
                .unwrap();
        assert_ne!(base, other_until, "filter until must move the bucket");

        let no_filter: Option<crate::passages::SourceFilter> = None;
        let without_filter =
            PassageKeyParams::new("search_passages", "q", 5, Some(0.5), &no_filter)
                .sans_query()
                .unwrap();
        assert_ne!(
            base, without_filter,
            "presence of a filter must move the bucket"
        );
    }

    /// The `below_cutoff` verdict's summary text comes from its own
    /// match arm, picked by a string comparison nothing else in the
    /// wire shape (verdict, ranking.rank, ranking.served) would notice
    /// falling through to the generic "shares no term" phrasing
    /// instead (#601).
    #[test]
    fn below_cutoff_summary_reports_the_rank_and_cutoff_not_the_no_term_overlap_text() {
        let explanation = crate::registry::PassageSearchExplanation {
            paragraph: 0,
            paragraphs: 1,
            paragraph_named: false,
            query_terms: Vec::new(),
            lexical: None,
            paragraph_terms: None,
            vector: crate::registry::VectorLaneReport::Off {
                provider_configured: false,
            },
            fused: false,
            ranked: 5,
            rank: Some(4),
            score: Some(0.1),
            bm25_lane: Some((4, 0.1)),
            vector_lane: None,
            limit: 3,
            served: false,
            cutoff_score: Some(0.5),
            limit_to_reach: crate::registry::LimitToReach::At(4),
        };
        let result = SearchExplanation::from_explanation("doc-a", explanation);
        assert_eq!(result.verdict, "below_cutoff", "{}", result.summary);
        assert!(
            result.summary.contains("ranked 4 of 5"),
            "must use the below_cutoff phrasing, not the no-term-overlap \
             fallback: {}",
            result.summary
        );
        assert!(
            result.summary.contains("limit 4 reaches it"),
            "{}",
            result.summary
        );
    }

    /// `LimitToReach`'s three endings must stay distinguishable on the
    /// wire: `Unreachable` sends `limit_to_reach_reason: "unreachable"`
    /// with `limit_to_reach` itself omitted — never confusable with
    /// `NotRanked`, which omits both (CodeRabbit, PR #609).
    #[test]
    fn limit_to_reach_wire_shape_distinguishes_unreachable_from_not_ranked() {
        let base = |rank, limit_to_reach| crate::registry::PassageSearchExplanation {
            paragraph: 0,
            paragraphs: 1,
            paragraph_named: false,
            query_terms: Vec::new(),
            lexical: None,
            paragraph_terms: None,
            vector: crate::registry::VectorLaneReport::Off {
                provider_configured: false,
            },
            fused: false,
            ranked: 5,
            rank,
            score: None,
            bm25_lane: None,
            vector_lane: None,
            limit: 3,
            served: false,
            cutoff_score: Some(0.5),
            limit_to_reach,
        };

        let unreachable = serde_json::to_value(SearchExplanation::from_explanation(
            "doc-a",
            base(Some(4), crate::registry::LimitToReach::Unreachable),
        ))
        .unwrap();
        assert_eq!(
            unreachable["ranking"]["limit_to_reach_reason"],
            "unreachable"
        );
        assert!(
            unreachable["ranking"].get("limit_to_reach").is_none(),
            "{unreachable}"
        );

        let not_ranked = serde_json::to_value(SearchExplanation::from_explanation(
            "doc-a",
            base(None, crate::registry::LimitToReach::NotRanked),
        ))
        .unwrap();
        assert!(
            not_ranked["ranking"].get("limit_to_reach").is_none(),
            "{not_ranked}"
        );
        assert!(
            not_ranked["ranking"].get("limit_to_reach_reason").is_none(),
            "NotRanked carries no reason — a bare absence, distinct from \
             Unreachable's explicit one: {not_ranked}"
        );
    }

    /// issue #621: `record_lane_hits` must sort each hit into exactly
    /// one of the three named lanes, matching `record_passage_hit`'s
    /// own "counted nowhere rather than inventing a fourth label" rule
    /// for a hit with neither lane set (which cannot occur from a real
    /// search, but the tally must not silently misclassify it either).
    #[test]
    fn record_lane_hits_sorts_each_combination_into_its_own_lane() {
        let metrics = crate::metrics::Metrics::default();
        let lane_hits = record_lane_hits(
            &metrics,
            [
                (true, false),  // bm25_only
                (true, false),  // bm25_only
                (true, true),   // both_lanes
                (false, true),  // vector_only
                (false, false), // neither — counted nowhere
            ],
        );
        assert_eq!(
            lane_hits,
            [2, 1, 1],
            "[bm25_only, both_lanes, vector_only], neither-lane hits excluded"
        );
    }

    #[test]
    fn record_lane_hits_reports_all_zeros_for_an_empty_batch() {
        let metrics = crate::metrics::Metrics::default();
        assert_eq!(record_lane_hits(&metrics, []), [0, 0, 0]);
    }

    /// issue #621: the tally is not just a local count — it must also
    /// drive `Metrics::record_passage_hit` so the Prometheus counter
    /// agrees with the returned array. A mutant that dropped the
    /// `metrics.record_passage_hit(..)` call inside the loop would
    /// still pass the two tests above.
    #[test]
    fn record_lane_hits_also_drives_the_prometheus_counter() {
        let metrics = crate::metrics::Metrics::default();
        record_lane_hits(&metrics, [(true, false), (true, true), (false, true)]);
        let gauges = crate::metrics::GaugeSnapshot {
            contexts_registered: 0,
            groups_registered: 0,
            contexts_resident: 0,
            resident_bytes: 0,
            wal_bytes: 0,
            passages_wal_bytes: 0,
            dead_edges_total: 0,
            dead_attributions_total: 0,
            arena_slack_total: 0,
            unsourced_edges_total: 0,
            unsourced_weight_total: 0.0,
            embed_breaker: None,
            rerank_breaker: None,
            retrieval_cache_entries: 0,
            retrieval_cache_bytes: 0,
            semantic_cache_entries: 0,
            embed_slot_waiters: 0,
            per_context: Vec::new(),
        };
        let rendered = metrics.render_prometheus(&gauges);
        assert!(rendered.contains("taguru_passage_lane_contributions_total{lane=\"bm25_only\"} 1"));
        assert!(
            rendered.contains("taguru_passage_lane_contributions_total{lane=\"both_lanes\"} 1")
        );
        assert!(
            rendered.contains("taguru_passage_lane_contributions_total{lane=\"vector_only\"} 1")
        );
    }
}
