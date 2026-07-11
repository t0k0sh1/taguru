use std::collections::BTreeMap;
use std::time::Instant;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Response;
use serde::{Deserialize, Serialize};

use crate::metrics::{ErrorKind, SearchOp};
use crate::registry::{AppState, CitationLookup};

use super::{
    AppJson, AppQuery, KeysetQuery, MAX_MATCH_LIMIT, access_error, clamp, error, not_found, ok,
    overlong, search_log_enabled,
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
    Path(name): Path<String>,
    AppJson(request): AppJson<LookupPassagesRequest>,
) -> Response {
    let started_at = Instant::now();
    // Each requested source returns its whole passage: the response
    // scales with this list, so the list itself is what gets bounded.
    if let Some(refusal) = overlong("sources", request.sources.len(), started_at) {
        return refusal;
    }
    match state.lookup_passages(&name, &request.sources) {
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
    Path(name): Path<String>,
    AppJson(request): AppJson<CitationRequest>,
) -> Response {
    let started_at = Instant::now();
    match state.citation(&name, &request.source, request.paragraph) {
        None => not_found(&name, started_at),
        Some(Err(io_error)) => passages_unreadable(&state, io_error, started_at),
        Some(Ok(CitationLookup::UnknownSource)) => {
            state.note_read(&name, true);
            error(
                StatusCode::NOT_FOUND,
                format!("source '{}' not found in context '{name}'", request.source),
                started_at,
            )
        }
        Some(Ok(CitationLookup::IndexOutOfRange)) => {
            state.note_read(&name, true);
            error(
                StatusCode::NOT_FOUND,
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
    Path(name): Path<String>,
    AppQuery(query): AppQuery<KeysetQuery>,
) -> Response {
    let started_at = Instant::now();
    let limit = clamp(query.limit, MAX_MATCH_LIMIT, MAX_MATCH_LIMIT);
    match state.passage_sources(&name) {
        None => not_found(&name, started_at),
        Some(Ok(mut sources)) => {
            sources.sort();
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
        StatusCode::INTERNAL_SERVER_ERROR,
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
    Path(name): Path<String>,
    key: Option<axum::Extension<crate::auth::AuthKey>>,
    AppJson(request): AppJson<RetractSourceRequest>,
) -> Response {
    let started_at = Instant::now();
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

pub async fn search_passages(
    State(state): State<AppState>,
    Path(name): Path<String>,
    AppJson(request): AppJson<SearchPassagesRequest>,
) -> Response {
    let started_at = Instant::now();
    // Off the async worker: a residency's first search tokenizes the
    // whole corpus into the index (the audit endpoints' rule).
    let outcome = tokio::task::block_in_place(|| {
        state.search_passages(
            &name,
            &request.query,
            clamp(request.limit, 5, MAX_MATCH_LIMIT),
        )
    });
    match outcome {
        None => not_found(&name, started_at),
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
                hits.into_iter()
                    .map(|hit| PassageHit {
                        source: hit.source,
                        paragraph: hit.index,
                        score: hit.score,
                        text: hit.text,
                        lanes: PassageLanes {
                            bm25: LaneEvidence::from_lane(hit.bm25),
                            vector: LaneEvidence::from_lane(hit.vector),
                        },
                    })
                    .collect::<Vec<_>>(),
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
