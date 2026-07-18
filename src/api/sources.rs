use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::response::Response;
use serde::{Deserialize, Serialize};
use taguru::deadline::Deadline;

use crate::metrics::{ErrorKind, SearchOp};
use crate::registry::{AppState, CitationLookup, PassageExplainLookup};

use super::aliases::KeysetQuery;
use super::recall::cross_targets;
use super::{
    AppJson, AppPath, AppQuery, CrossMatch, ErrorCode, MAX_MATCH_LIMIT, MAX_NAME_BYTES,
    MAX_PASSAGES_PER_REQUEST, MAX_QUESTION_BYTES, MAX_QUESTIONS_PER_PARAGRAPH, MAX_SECTION_BYTES,
    access_error, bounded_parallel_map, clamp, cross_search_concurrency, deadline_exceeded, empty,
    error, not_found, ok, orphaned_source, overlong, oversized, search_log_enabled,
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
#[derive(Serialize)]
pub struct SourcePage {
    pub total: usize,
    pub sources: Vec<String>,
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
    match tokio::task::block_in_place(|| state.passage_sources(&name)) {
        None => not_found(&name, started_at),
        // `passage_sources` already yields BTreeMap-key order — no sort.
        Some(Ok(sources)) => {
            let sources: Vec<String> = match query.prefix.as_deref() {
                Some(prefix) => sources
                    .into_iter()
                    .filter(|source| source.starts_with(prefix))
                    .collect(),
                None => sources,
            };
            let total = sources.len();
            let sources: Vec<String> = sources
                .into_iter()
                .filter(|source| {
                    query
                        .after
                        .as_deref()
                        .is_none_or(|after| source.as_str() > after)
                })
                .take(limit)
                .collect();
            ok(SourcePage { total, sources }, started_at)
        }
        Some(Err(io_error)) => passages_unreadable(&state, io_error, started_at),
    }
}

/// The passage store exists but could not be loaded — its snapshot and
/// log hold acknowledged writes, so this is a 500 pointing at disk,
/// never a silent empty answer.
fn passages_unreadable(
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
}

/// One PARAGRAPH matched by passage search: the text lane, for
/// knowledge that never decomposed into triples. `paragraph` is its
/// position within the source (0-based, this split); `text` is that
/// paragraph alone — cite it, or dereference the whole source through
/// the lookup endpoint. `score` is the fused reciprocal-rank number
/// when the semantic lane ran, the raw BM25 score otherwise; `lanes`
/// carries each lane's own rank and raw score — evidence for the
/// reading LLM, the same posture as resolve's tiers.
#[derive(Serialize)]
pub struct PassageHit {
    pub source: String,
    pub paragraph: u32,
    pub score: f32,
    pub text: String,
    pub lanes: PassageLanes,
}

#[derive(Serialize)]
pub struct PassageLanes {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25: Option<LaneEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<LaneEvidence>,
}

/// Where one lane put this hit: 1-based rank within the lane's own
/// candidate pool, and that lane's raw score (BM25, or cosine).
#[derive(Serialize)]
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
    // Off the async worker: a residency's first search tokenizes the
    // whole corpus into the index (the audit endpoints' rule).
    let outcome = tokio::task::block_in_place(|| {
        state.search_passages(
            &name,
            &request.query,
            clamp(request.limit, 5, MAX_MATCH_LIMIT),
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
        Some(Ok(hits)) => {
            state.note_search(SearchOp::SearchPassages, &name, hits.is_empty());
            for hit in &hits {
                state
                    .metrics()
                    .record_passage_hit(hit.bm25.is_some(), hit.vector.is_some());
            }
            if search_log_enabled() {
                tracing::info!(
                    target: "taguru::search",
                    context = %name,
                    op = "search_passages",
                    cue = %request.query,
                    hits = hits.len(),
                    top_score = hits.first().map_or(0.0, |hit| f64::from(hit.score)),
                    "search",
                );
            }
            ok(
                hits.into_iter().map(PassageHit::from).collect::<Vec<_>>(),
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
}

/// One verdict for "why didn't (or did) this source appear for this
/// query": the first that applies, machine-readable in `verdict`,
/// human-readable in `summary`, evidence attached for the skeptical.
/// Every verdict is a 200 — a diagnosed miss is this endpoint's
/// success, not its failure.
#[derive(Serialize)]
pub struct SearchExplanation {
    /// `not_stored` | `paragraph_out_of_range` | `no_query_terms` |
    /// `no_term_overlap` | `below_cutoff` | `served`, first match wins
    /// in that order (a served paragraph is served, whatever else is
    /// true of it).
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
                reason: Some(if *provider_configured {
                    "passage embedding is off (TAGURU_EMBED_PASSAGES)".to_string()
                } else {
                    "no embedding provider is configured".to_string()
                }),
                floor: None,
                cosine: None,
                rank: None,
            },
            VectorLaneReport::QueryEmbeddingFailed(error) => VectorExplain {
                ran: false,
                reason: Some(format!("the query embedding failed: {error}")),
                floor: None,
                cosine: None,
                rank: None,
            },
            VectorLaneReport::NoVectors => VectorExplain {
                ran: false,
                reason: Some(
                    "no paragraph vectors exist yet — the embedding refresh has not \
                     covered this context"
                        .to_string(),
                ),
                floor: None,
                cosine: None,
                rank: None,
            },
            VectorLaneReport::ModelChanged { stored, current } => VectorExplain {
                ran: false,
                reason: Some(format!(
                    "stored vectors belong to model '{stored}' but the provider is \
                     '{current}' — they are never served, and the next refresh re-embeds"
                )),
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
    // Off the async worker: a residency's first search tokenizes the
    // whole corpus into the index (the audit endpoints' rule).
    let outcome = tokio::task::block_in_place(|| {
        state.explain_passage_search(
            &name,
            &request.query,
            &request.source,
            request.paragraph,
            clamp(request.limit, 5, MAX_MATCH_LIMIT),
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

#[derive(Debug, Deserialize)]
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
    // Each residency's first search tokenizes its whole corpus, so the
    // fetch belongs on the blocking pool — bounded and concurrent, the
    // same `bounded_parallel_map` shape as `cross_matches`'s gather.
    // `deadline` is `Copy`, so every job carries its own value and can
    // bail out mid-tokenize the same way the single-context handler
    // does.
    let permits = cross_search_concurrency().min(targets.len().max(1));
    let owned_targets = Arc::clone(&targets);
    let query = request.query.clone();
    let job_state = state.clone();
    let fetched = bounded_parallel_map(targets.len(), permits, move |index| {
        job_state.search_passages(&owned_targets[index], &query, limit, deadline)
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
    for (index, outcome) in fetched.into_iter().enumerate() {
        let name = &targets[index];
        match outcome {
            None => return not_found(name, started_at),
            Some(Err(_)) if deadline.expired() => return deadline_exceeded(started_at),
            Some(Err(io_error)) => return passages_unreadable(&state, io_error, started_at),
            Some(Ok(hits)) => {
                state.note_search(SearchOp::SearchPassages, name, hits.is_empty());
                for hit in &hits {
                    state
                        .metrics()
                        .record_passage_hit(hit.bm25.is_some(), hit.vector.is_some());
                }
                pool.extend(
                    hits.into_iter()
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
    ok(
        pool.into_iter()
            .map(|(index, _, hit)| CrossMatch {
                context: targets[index].clone(),
                inner: PassageHit::from(hit),
            })
            .collect::<Vec<_>>(),
        started_at,
    )
}

/// Original text passages keyed by source id — the same opaque ids that
/// appear on attributions — plus, optionally, doc2query questions and
/// section markers per source, each naming a paragraph of that
/// source's text IN THIS REQUEST (a question or section cannot attach
/// to text the request does not carry: storage replaces per source,
/// wholesale).
#[derive(Debug, Deserialize)]
pub struct StorePassagesRequest {
    pub passages: BTreeMap<String, String>,
    #[serde(default)]
    pub questions: BTreeMap<String, Vec<QuestionSpec>>,
    #[serde(default)]
    pub sections: BTreeMap<String, Vec<SectionSpec>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuestionSpec {
    pub paragraph: u32,
    pub question: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SectionSpec {
    pub paragraph: u32,
    pub section: String,
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
    AppJson(request): AppJson<StorePassagesRequest>,
) -> Response {
    let started_at = Instant::now();
    // Refused before any tokenization or lock is taken: nothing of an
    // oversized batch is stored.
    if request.passages.len() > MAX_PASSAGES_PER_REQUEST {
        return error(
            ErrorCode::OverLimit,
            format!(
                "batch of {} passages exceeds the per-request limit of \
                 {MAX_PASSAGES_PER_REQUEST}; split the store",
                request.passages.len()
            ),
            started_at,
        );
    }
    // Source ids are names (the passage text itself is a document and
    // rides under the body cap instead) — and like every name, an
    // empty one is refused: it would list as a blank entry in GET
    // /sources and answer lookups nothing sensible ever asks.
    for source in request.passages.keys() {
        if let Some(refusal) = oversized("a passage source id", source, MAX_NAME_BYTES, started_at)
        {
            return refusal;
        }
        if let Some(refusal) = empty("a passage source id", source, started_at) {
            return refusal;
        }
    }
    // Question structure is the request's to get right: sources must
    // name passages in THIS request, sizes and per-paragraph counts
    // stay under the shared caps. (Whether a paragraph index exists in
    // the text is settled at store time, one rule for every entrance.)
    for (source, questions) in &request.questions {
        if let Some(refusal) = orphaned_source("questions", source, &request.passages, started_at) {
            return refusal;
        }
        let mut per_paragraph: BTreeMap<u32, usize> = BTreeMap::new();
        for spec in questions {
            if spec.question.len() > MAX_QUESTION_BYTES {
                return error(
                    ErrorCode::InvalidArgument,
                    format!(
                        "a question for '{source}' is {} bytes; questions are capped at \
                         {MAX_QUESTION_BYTES} bytes",
                        spec.question.len()
                    ),
                    started_at,
                );
            }
            // Embedded verbatim on the next refresh, where providers
            // refuse zero-length input — which would stall the whole
            // refresh pass at this row, every pass.
            if let Some(refusal) = empty(
                &format!("a question for '{source}'"),
                &spec.question,
                started_at,
            ) {
                return refusal;
            }
            let count = per_paragraph.entry(spec.paragraph).or_insert(0);
            *count += 1;
            if *count > MAX_QUESTIONS_PER_PARAGRAPH {
                return error(
                    ErrorCode::InvalidArgument,
                    format!(
                        "paragraph {} of '{source}' carries more than \
                         {MAX_QUESTIONS_PER_PARAGRAPH} questions",
                        spec.paragraph
                    ),
                    started_at,
                );
            }
        }
    }
    // Section structure follows questions' rule: sources must name
    // passages in THIS request, sizes stay under the shared cap.
    // (Whether a paragraph index exists in the text is settled at
    // store time, same as questions; ingest's batch format has no
    // per-paragraph section count limit either, so neither does this.)
    for (source, sections) in &request.sections {
        if let Some(refusal) = orphaned_source("sections", source, &request.passages, started_at) {
            return refusal;
        }
        for spec in sections {
            if spec.section.len() > MAX_SECTION_BYTES {
                return error(
                    ErrorCode::InvalidArgument,
                    format!(
                        "a section label for '{source}' is {} bytes; section labels are \
                         capped at {MAX_SECTION_BYTES} bytes",
                        spec.section.len()
                    ),
                    started_at,
                );
            }
            if let Some(refusal) = empty(
                &format!("a section label for '{source}'"),
                &spec.section,
                started_at,
            ) {
                return refusal;
            }
        }
    }
    let mut questions = request.questions;
    let mut sections = request.sections;
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
            (
                source,
                crate::passages::PassageSubmission {
                    text,
                    questions,
                    sections,
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
        Some(Err(io_error)) => {
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
