//! #308 (ADR 0006 §14): the `--assembly` passage-lane substitute and
//! the `--max-items`/`--max-bytes`/`--max-tokens` equal-budget
//! machinery it shares with the unmodified `baseline` mode. Nothing
//! here touches the structural lane (`resolve` → `query`) — that runs
//! identically in both modes, so a `baseline`/`assembly` pair of runs
//! stays comparable on coverage and on `lanes.structural_hit`
//! specifically. `lanes.passage_hit` is **not** identical across
//! modes: `hits_from_evidence_items` (`src/evaluate.rs`) unions each
//! admitted item's own locator with its `citation_refs` — a graph
//! association's corroborating attributions — into the same `hits`
//! recall/lane-cross scoring reads, so a case whose evidence is
//! entirely structural can read `passage_hit: true` under `--assembly`
//! where `baseline` (a pure passage-lane call) would read it `false`.
//!
//! **Assembly mode** replaces `POST /contexts/{name}/sources/search`
//! with `POST /contexts/{name}/evidence` ([`run_evidence_lane`]),
//! reusing the same request/response wire types #305/#307 already
//! shipped (`crate::api::evidence::assemble::EvidencePackage`) rather
//! than minting a parallel client-side mirror.
//!
//! **Baseline mode**, when a budget flag is given, truncates the
//! already-fetched `sources/search` hits to the identical three
//! ceilings ([`truncate_to_budget`]) using the *exact* accounting
//! `crate::api::evidence::budget`/`select` computes server-side —
//! never a reimplemented approximation of it, since ADR 0006 §8 fixes
//! the token estimator itself as part of the wire contract this
//! equal-budget comparison depends on agreeing with. Concretely, each
//! baseline `PassageHit` is wrapped in the same `EvidenceItem` envelope
//! ([`envelope_for`]) a real assembled package would produce for it —
//! `candidate_id`, `kind`, `fused_rank`, `lane_ranks`, empty
//! `citation_refs` (a passage item always self-cites, ADR 0006 §6) —
//! and measured with the real, server-side `content_metrics` function,
//! not the bare `PassageHit` alone. A bare-`PassageHit` measurement
//! undercounts every hit by the envelope's own overhead (the
//! `candidate_id` string alone routinely runs over a hundred bytes),
//! which would let baseline admit roughly twice as many hits as
//! assembly under the identical nominal `--max-bytes` — the opposite
//! of "equal budget". A baseline hit array never carries a `citations`
//! array of its own (unlike an assembled package's `items` +
//! `citations` pair — a passage item's `citation_refs` is always
//! empty, so a real assembled package built from passage-only
//! candidates would carry an empty `citations` array too), so
//! truncation accounts for one array (`hits`) plus a
//! permanently-empty `citations` array's `array_overhead(0)` floor —
//! the same floor an assembled package pays even when it selects zero
//! citations, which is what makes the two modes' ceilings comparable
//! at all despite their different response shapes.

use std::time::Instant;

use serde_json::json;

use crate::api::evidence::assemble::EvidencePackage;
use crate::api::evidence::budget::{
    BudgetLimits, BudgetRequest, BudgetUsage, array_overhead, content_metrics, tokens_from_quarters,
};
use crate::api::evidence::select::EvidenceItem;
use crate::api::evidence::{CandidatePayload, EvidenceCandidate, LaneRank};
use crate::api::sources::PassageHit;
use crate::evalset::EvalCase;
use crate::remote::Api;

use super::{elapsed_ms, truncate_message};

/// The three `--max-items`/`--max-bytes`/`--max-tokens` CLI values,
/// still unresolved (each may be absent) — resolved once per run via
/// [`EvidenceBudgetArgs::resolve`], the same `clamp`-through-defaults
/// funnel the server itself uses ([`BudgetLimits::resolve`]), so a
/// caller who names only one flag gets the other two defaults, not an
/// error.
#[derive(Debug)]
pub(super) struct EvidenceBudgetArgs {
    pub(super) max_items: Option<usize>,
    pub(super) max_bytes: Option<usize>,
    pub(super) max_tokens: Option<usize>,
}

impl EvidenceBudgetArgs {
    pub(super) fn resolve(&self) -> BudgetLimits {
        BudgetLimits::resolve(Some(BudgetRequest {
            max_bytes: self.max_bytes,
            max_tokens: self.max_tokens,
            max_items: self.max_items,
        }))
    }
}

/// [`run_evidence_lane`]'s outcome — deliberately not `EvidenceOutcome`
/// (the artifact schema type, `super::EvidenceOutcome`): this is the
/// unshaped HTTP result, before `evaluate.rs` turns it into both the
/// scoring-facing `PassageOutcome` projection and the diagnostic
/// `EvidenceOutcome` block.
pub(super) enum LaneResult {
    Assembled {
        package: Box<EvidencePackage>,
        latency_ms: u64,
    },
    Failed {
        message: String,
        latency_ms: u64,
    },
}

/// `POST /contexts/{context}/evidence` (ADR 0006 §5.1). `origins` is
/// `cues` when the case declares any, else the bare `query` — the same
/// fallback rule the structural lane already applies when resolving a
/// cueless case. `text_fallback_query` is always the case's own
/// `query`, so the passage/community lanes and a configured reranker
/// all search the same text regardless of `origins`.
pub(super) fn run_evidence_lane(
    api: &Api,
    context: &str,
    case: &EvalCase,
    limit: usize,
    limits: &BudgetLimits,
    rerank_model: Option<&str>,
) -> LaneResult {
    let origins = if case.cues.is_empty() {
        vec![case.query.clone()]
    } else {
        case.cues.clone()
    };
    let mut body = json!({
        "origins": origins,
        "text_fallback_query": case.query,
        "search_limit": limit,
        "semantic_floor": case.options.floor,
        "budget": {
            "max_items": limits.max_items,
            "max_bytes": limits.max_bytes,
            "max_tokens": limits.max_tokens,
        },
    });
    if let Some(model) = rerank_model {
        body["rerank"] = json!({ "model": model });
    }

    let started_at = Instant::now();
    match api.post(&["contexts", context, "evidence"], &body) {
        Ok(value) => {
            let latency_ms = elapsed_ms(started_at);
            match serde_json::from_value::<EvidencePackage>(value) {
                Ok(package) => LaneResult::Assembled {
                    package: Box::new(package),
                    latency_ms,
                },
                Err(error) => LaneResult::Failed {
                    message: format!("response carries no recognizable evidence package ({error})"),
                    latency_ms,
                },
            }
        }
        Err(message) => LaneResult::Failed {
            message: truncate_message(&message),
            latency_ms: elapsed_ms(started_at),
        },
    }
}

/// [`truncate_to_budget`]'s result: the admitted prefix of `hits`
/// (rank order preserved), the resulting [`BudgetUsage`] computed with
/// the identical formula `crate::api::evidence::budget`/`select` uses
/// server-side, and how many of the input hits did not fit.
pub(super) struct TruncatedHits {
    pub(super) hits: Vec<PassageHit>,
    pub(super) usage: BudgetUsage,
    pub(super) omitted_total: usize,
}

/// Wraps a baseline `PassageHit` in the same `EvidenceItem` envelope a
/// real assembled package would produce for it — built by calling
/// [`EvidenceCandidate::from_passage`] itself (the same constructor
/// `src/api/evidence/assemble.rs`'s live pipeline calls) for
/// `candidate_id`/`kind`/`lane` rather than re-deriving that NUL-joined
/// format and lane-selection rule a second time: one definition, two
/// consumers, the same discipline ADR 0006 §7 already applies to
/// `canonical_key`. `citation_refs` is always empty — a passage item
/// always self-cites (ADR 0006 §6), so it never contributes a
/// `citations` array entry, matching `truncate_to_budget`'s own
/// permanently-empty `citations` floor. `rank` is the hit's 1-based
/// position in the incoming (already fused-order) list, matching how a
/// single-lane pool's `fused_rank` would read with nothing upstream to
/// re-rank it against.
fn envelope_for(context: &str, hit: PassageHit, rank: usize) -> EvidenceItem {
    let EvidenceCandidate {
        kind,
        candidate_id,
        lane,
        lane_rank,
        payload,
        ..
    } = EvidenceCandidate::from_passage(context, hit, rank);
    let CandidatePayload::Passage(hit) = payload else {
        unreachable!("EvidenceCandidate::from_passage always produces a Passage payload")
    };
    EvidenceItem {
        candidate_id,
        kind,
        fused_rank: rank,
        lane_ranks: vec![LaneRank {
            lane,
            rank: lane_rank,
        }],
        citation_refs: Vec::new(),
        corroboration: None,
        contradicts: Vec::new(),
        bytes: None,
        estimated_tokens: None,
        association: None,
        passage: Some(hit),
        community: None,
    }
}

/// Walks `hits` in the order `sources/search` already ranked them,
/// admitting each into the budget if it still fits — skipping an
/// over-budget hit and continuing to the next, never stopping at the
/// first one that doesn't fit (ADR 0006 §3 D/§9's "skip, don't stop"
/// rule, applied here so a baseline run's own budget floor never
/// looks artificially exhausted by one large early hit that a smaller
/// later one would have fit around). `context` is the searched context
/// name — needed only to build each hit's [`envelope_for`] `candidate_id`
/// with the identical string a real assembled package would carry.
pub(super) fn truncate_to_budget(
    context: &str,
    hits: Vec<PassageHit>,
    limits: &BudgetLimits,
) -> TruncatedHits {
    let mut admitted: Vec<PassageHit> = Vec::new();
    let mut items_bytes_sum = 0usize;
    let mut items_quarters_sum: u64 = 0u64;
    let mut omitted_total = 0usize;

    // A baseline hit array never carries a `citations` array of its
    // own — `sources/search` has no such concept — so the permanently-
    // empty array's `array_overhead(0)` floor (2 bytes: `"[]"`) is
    // added on every check, matching what an assembled package pays
    // even when it selects zero citations.
    let empty_citations_bytes = array_overhead(0);
    let empty_citations_quarters = array_overhead(0) as u64;

    for (index, hit) in hits.into_iter().enumerate() {
        let envelope = envelope_for(context, hit, index + 1);
        let (bytes, quarters) = content_metrics(&envelope);
        let candidate_items = admitted.len() + 1;
        let candidate_bytes =
            array_overhead(candidate_items) + items_bytes_sum + bytes + empty_citations_bytes;
        let candidate_quarters = array_overhead(candidate_items) as u64
            + items_quarters_sum
            + quarters
            + empty_citations_quarters;
        let candidate_tokens = tokens_from_quarters(candidate_quarters);

        let fits = candidate_items <= limits.max_items
            && candidate_bytes <= limits.max_bytes
            && candidate_tokens <= limits.max_tokens;
        if fits {
            items_bytes_sum += bytes;
            items_quarters_sum += quarters;
            admitted.push(envelope.passage.expect("envelope_for always sets passage"));
        } else {
            omitted_total += 1;
        }
    }

    let items_used = admitted.len();
    let bytes_used = array_overhead(items_used) + items_bytes_sum + empty_citations_bytes;
    let quarters_used =
        array_overhead(items_used) as u64 + items_quarters_sum + empty_citations_quarters;

    TruncatedHits {
        hits: admitted,
        usage: BudgetUsage {
            items_used,
            bytes_used,
            tokens_used: tokens_from_quarters(quarters_used),
            limits: *limits,
        },
        omitted_total,
    }
}
