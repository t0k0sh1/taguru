//! #308 (ADR 0006 §14): the `--assembly` passage-lane substitute and
//! the `--max-items`/`--max-bytes`/`--max-tokens` equal-budget
//! machinery it shares with the unmodified `baseline` mode. Nothing
//! here touches the structural lane (`resolve` → `query`) — that runs
//! identically in both modes, so a `baseline`/`assembly` pair of runs
//! stays comparable on coverage and lane cross-tab.
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
//! `crate::api::evidence::budget` computes server-side — never a
//! reimplemented approximation of it, since ADR 0006 §8 fixes the
//! token estimator itself as part of the wire contract this equal-
//! budget comparison depends on agreeing with. A baseline hit array
//! carries no `citations` array of its own (unlike an assembled
//! package's `items` + `citations` pair), so truncation accounts for
//! one array (`hits`) plus a permanently-empty `citations` array's
//! `array_overhead(0)` floor — the same floor an assembled package
//! pays even when it selects zero citations, which is what makes the
//! two modes' ceilings comparable at all despite their different
//! response shapes.

use std::time::Instant;

use serde_json::json;

use crate::api::evidence::assemble::EvidencePackage;
use crate::api::evidence::budget::{
    BudgetLimits, BudgetRequest, BudgetUsage, array_overhead, content_metrics, tokens_from_quarters,
};
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

/// Walks `hits` in the order `sources/search` already ranked them,
/// admitting each into the budget if it still fits — skipping an
/// over-budget hit and continuing to the next, never stopping at the
/// first one that doesn't fit (ADR 0006 §3 D/§9's "skip, don't stop"
/// rule, applied here so a baseline run's own budget floor never
/// looks artificially exhausted by one large early hit that a smaller
/// later one would have fit around).
pub(super) fn truncate_to_budget(hits: Vec<PassageHit>, limits: &BudgetLimits) -> TruncatedHits {
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

    for hit in hits {
        let (bytes, quarters) = content_metrics(&hit);
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
            admitted.push(hit);
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
