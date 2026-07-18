use std::collections::BTreeMap;
use std::time::Instant;

use axum::extract::State;
use axum::response::Response;

use serde::{Deserialize, Serialize};

use taguru::deadline::Deadline;

use crate::limits::HeavyOpsLimiter;
use crate::registry::{AccessError, AppState};

use super::{
    AppBytes, AppPath, AssociationOut, MatchCursor, access_error, association_out,
    deadline_exceeded, locator_keys, ok, optional_body, page_by,
};

/// Vocabulary audit request: floors for the two fork detectors.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct VocabularyAuditRequest {
    /// Lexical (spelling) detector floor; omitted means 0.6.
    pub dice_floor: Option<f64>,
    /// Semantic (gloss cosine) detector floor; omitted means 0.6.
    pub cosine_floor: Option<f32>,
}

#[derive(Serialize)]
pub struct TwinPair<S> {
    pub a: String,
    pub b: String,
    pub score: S,
}

/// The vocabulary health report: fork CANDIDATES for review, not
/// verdicts. Lexical twins are spelling drift (青嶺酒蔵/青嶺酒造);
/// semantic twins are synonym drift (創業年/設立年) visible only
/// through gloss embeddings.
#[derive(Serialize)]
pub struct VocabularyAudit {
    pub lexical_concepts: Vec<TwinPair<f64>>,
    pub lexical_labels: Vec<TwinPair<f64>>,
    pub semantic_concepts: Vec<TwinPair<f32>>,
    pub semantic_labels: Vec<TwinPair<f32>>,
    /// Why the semantic half was skipped, when it was.
    pub semantic_note: Option<String>,
}

fn twin_pairs<S>(pairs: Vec<(String, String, S)>) -> Vec<TwinPair<S>> {
    pairs
        .into_iter()
        .map(|(a, b, score)| TwinPair { a, b, score })
        .collect()
}

/// Shared body of `audit_vocabulary` and `audit_drift`'s `include_twins`
/// section: lexical fork candidates always, semantic ones when
/// embeddings are configured and the deadline allows. Callers run this
/// inside `block_in_place` — it does its own CPU-bound pairwise sweeps
/// and must never run on an async worker (see the comment this carried
/// forward from `audit_vocabulary`, below).
fn vocabulary_audit(
    state: &AppState,
    name: &str,
    dice_floor: f64,
    cosine_floor: f32,
    deadline: Deadline,
) -> Result<VocabularyAudit, AccessError> {
    // BOTH halves are CPU-bound pairwise sweeps — the lexical one is
    // O(Σ posting_len²) over the whole vocabulary, seconds at tens of
    // thousands of concepts. Neither may run on an async worker: with
    // the lexical half inline, a handful of concurrent audits pinned
    // every worker and starved every other request, /health included.
    let lexical = state
        .read_context(name, |context| {
            let concepts = context
                .similar_concepts(dice_floor, deadline)
                .map_err(|_| AccessError::DeadlineExceeded)?;
            let labels = context
                .similar_labels(dice_floor, deadline)
                .map_err(|_| AccessError::DeadlineExceeded)?;
            Ok((concepts, labels))
        })
        .and_then(std::convert::identity)?;

    // The lexical half already spent the budget checking its own
    // deadline; skip the semantic half rather than fail a request that
    // did real, useful work — semantic_twins also checks this same
    // deadline internally (its own pairwise sweep can still be large),
    // so this is a courtesy early-out on top of that, not the only guard.
    if deadline.expired() {
        return Ok(VocabularyAudit {
            lexical_concepts: twin_pairs(lexical.0),
            lexical_labels: twin_pairs(lexical.1),
            semantic_concepts: Vec::new(),
            semantic_labels: Vec::new(),
            semantic_note: Some("意味的検出はスキップ (期限切れ)".to_string()),
        });
    }
    let Some((semantic_concepts, semantic_labels, semantic_note)) =
        state.semantic_twins(name, cosine_floor, deadline)
    else {
        return Err(AccessError::NotFound);
    };

    Ok(VocabularyAudit {
        lexical_concepts: twin_pairs(lexical.0),
        lexical_labels: twin_pairs(lexical.1),
        semantic_concepts: twin_pairs(semantic_concepts),
        semantic_labels: twin_pairs(semantic_labels),
        semantic_note,
    })
}

pub async fn audit_vocabulary(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    AppBytes(body): AppBytes,
) -> Response {
    let started_at = Instant::now();
    let request: VocabularyAuditRequest = match optional_body(&body, started_at) {
        Ok(request) => request,
        Err(refusal) => return *refusal,
    };
    let dice_floor = request.dice_floor.unwrap_or(0.6);
    let cosine_floor = request.cosine_floor.unwrap_or(0.6);

    // vocabulary_audit's lexical half builds an O(n) per-span bigram
    // table before its own inner loop ever checks this deadline — fail
    // a spent request before that table gets built, the way every other
    // heavy handler pre-flights.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    match tokio::task::block_in_place(|| {
        vocabulary_audit(&state, &name, dice_floor, cosine_floor, deadline)
    }) {
        Ok(audit) => ok(audit, started_at),
        Err(failure) => access_error(&state, failure, &name, started_at),
    }
}

/// One edge [`Context::unsourced_edges`] flagged, reshaped to the wire
/// (section-resolved association, same as every other match page).
#[derive(Serialize)]
pub struct UnsourcedEdgeOut {
    pub unsourced_weight: f64,
    pub unsourced_count: u64,
    pub association: AssociationOut,
}

/// Drift audit request: an optional floor for the unsourced-weight
/// section, paging over it, and an opt-in for the vocabulary-twins
/// section (skipped by default — it's the same CPU-bound pairwise
/// sweep `audit_vocabulary` runs, not something every drift check
/// should pay for).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct DriftAuditRequest {
    /// Minimum unsourced weight (compared by magnitude) to include;
    /// omitted means any amount at all.
    pub unsourced_floor: Option<f64>,
    /// Omitted means 100, capped at 1000 — pages exactly like recall,
    /// query, and unreachable_from.
    pub limit: Option<usize>,
    /// Resume past a previous page's last match — see [`MatchCursor`].
    pub after: Option<MatchCursor>,
    /// Also run the lexical/semantic fork-candidate sweep
    /// (`vocabulary_audit`) and include it as `twins`.
    pub include_twins: bool,
    /// Lexical (spelling) detector floor; omitted means 0.6. Ignored
    /// unless `include_twins` is set.
    pub dice_floor: Option<f64>,
    /// Semantic (gloss cosine) detector floor; omitted means 0.6.
    /// Ignored unless `include_twins` is set.
    pub cosine_floor: Option<f32>,
}

/// The drift audit: three independent read-only checks in one
/// response, each answering a different way the graph's current shape
/// can have drifted from what actually happened. `unsourced`/`total`
/// page like every other match list; the alias and twins sections are
/// small enough in practice to return whole.
#[derive(Serialize)]
pub struct DriftAudit {
    pub total: usize,
    pub unsourced: Vec<UnsourcedEdgeOut>,
    pub dead_concept_aliases: BTreeMap<String, String>,
    pub dead_label_aliases: BTreeMap<String, String>,
    /// Present only when `include_twins` was set.
    pub twins: Option<VocabularyAudit>,
}

pub async fn audit_drift(
    State(state): State<AppState>,
    AppPath(name): AppPath<String>,
    axum::Extension(deadline): axum::Extension<Deadline>,
    axum::Extension(heavy_ops): axum::Extension<HeavyOpsLimiter>,
    AppBytes(body): AppBytes,
) -> Response {
    let started_at = Instant::now();
    let request: DriftAuditRequest = match optional_body(&body, started_at) {
        Ok(request) => request,
        Err(refusal) => return *refusal,
    };
    let floor = request.unsourced_floor.unwrap_or(0.0);

    // The two sweeps below each check this deadline only once they are
    // already inside their own edge loop — fail a spent request before
    // read_context even takes the registry lock, the way every other
    // heavy handler pre-flights.
    if deadline.expired() {
        return deadline_exceeded(started_at);
    }
    let loaded = tokio::task::block_in_place(|| {
        state
            .read_context(&name, |context| {
                let unsourced = context
                    .unsourced_edges(floor, deadline)
                    .map_err(|_| AccessError::DeadlineExceeded)?;
                let aliases = context
                    .dead_canonical_aliases(deadline)
                    .map_err(|_| AccessError::DeadlineExceeded)?;
                Ok((unsourced, aliases))
            })
            .and_then(std::convert::identity)
    });
    let (unsourced, (dead_concept_aliases, dead_label_aliases)) = match loaded {
        Ok(loaded) => loaded,
        Err(failure) => return access_error(&state, failure, &name, started_at),
    };

    let (total, unsourced) = page_by(unsourced, request.limit, request.after.as_ref(), |edge| {
        (
            edge.weight,
            edge.association.subject.as_str(),
            edge.association.label.as_str(),
            edge.association.object.as_str(),
        )
    });
    // A graph read like unreachable_from — zero drift is the audit
    // succeeding, not a miss, so it never counts as an empty read.
    state.note_read(&name, false);
    let sections = state.resolve_sections(
        &name,
        locator_keys(unsourced.iter().map(|edge| &edge.association)),
    );
    let unsourced = unsourced
        .into_iter()
        .map(|edge| UnsourcedEdgeOut {
            unsourced_weight: edge.weight,
            unsourced_count: edge.count,
            association: association_out(edge.association, &sections),
        })
        .collect();

    let twins = if request.include_twins {
        // Only this branch runs the CPU-bound pairwise scan
        // `audit_vocabulary` also runs, so it alone spends a
        // heavy-ops permit — the unsourced/alias sweeps above are a
        // cheap O(n) pass that a full heavy-ops ceiling would
        // otherwise gate for no reason.
        let _permit = match heavy_ops.try_acquire() {
            Ok(permit) => permit,
            Err(shed_response) => return *shed_response,
        };
        let dice_floor = request.dice_floor.unwrap_or(0.6);
        let cosine_floor = request.cosine_floor.unwrap_or(0.6);
        match tokio::task::block_in_place(|| {
            vocabulary_audit(&state, &name, dice_floor, cosine_floor, deadline)
        }) {
            Ok(audit) => Some(audit),
            Err(failure) => return access_error(&state, failure, &name, started_at),
        }
    } else {
        None
    };

    ok(
        DriftAudit {
            total,
            unsourced,
            dead_concept_aliases,
            dead_label_aliases,
            twins,
        },
        started_at,
    )
}
