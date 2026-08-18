//! Deterministic budgeted evidence selection (ADR 0006 §9, for #304).
//!
//! [`select`] consumes the output of [`super::fuse`] — an already
//! exact-key-deduped, RRF-ranked pool (ADR 0006 §7) — and runs the
//! rest of ADR 0006 §9's pipeline over it: contradiction grouping,
//! near-duplicate suppression, diversity-aware tiering, and greedy
//! admission under the three independent [`BudgetLimits`] hard
//! ceilings (ADR 0006 §8). It knows nothing about HTTP, MCP, or a
//! reranker — those are #305/#307's job; this module is a pure,
//! synchronous function of its inputs, which is what lets ADR 0006 §9
//! invariant I3 ("the same request against the same corpus revision
//! produces byte-identical JSON … every time") hold without needing to
//! reason about anything outside this file.
//!
//! Contradiction grouping happens here, after [`super::fuse`], not
//! before it as ADR 0006 §9's prose pipeline lists it — behaviorally
//! identical, since grouping only inspects `(subject, label, object,
//! signed_weight)`, never a rank or score `fuse` produces, so doing it
//! after fusion produces the same groups doing it before would have.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::budget::{
    BudgetLimits, BudgetUsage, array_overhead, content_metrics, tokens_from_quarters,
};
use super::{
    CandidatePayload, CitationCollector, CitationEntry, EvidenceCandidate, FusedCandidate,
    KIND_ASSOCIATION, LaneRank,
};
use crate::api::communities::CommunityHit;
use crate::api::sources::{Citation, PassageHit};
use crate::api::{AssociationOut, MAX_LISTED_ISSUES};

/// A `(source, paragraph)` locator with no citation text attached — the
/// shape an [`EvidenceItem`]'s own `citation_refs`/`corroboration`
/// carry (ADR 0006 §10); the text itself lives exactly once, in the
/// package's top-level `citations` ([`CitationEntry`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CitationRef {
    pub(crate) source: String,
    pub(crate) paragraph: u32,
}

/// Every independent source an admitted association/activation item's
/// underlying evidence traces back to (ADR 0006 §9 I4) — `None` for
/// every other item, so dedup/near-duplicate suppression can never be
/// accused of silently collapsing corroboration down to a single
/// opaque count: the field is either fully populated or absent, never
/// truncated.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Corroboration {
    pub(crate) sources: Vec<String>,
    pub(crate) attributions: Vec<CitationRef>,
}

/// One admitted piece of evidence (ADR 0006 §10). Embeds the
/// *existing* wire type verbatim as the kind-specific payload — no
/// parallel type is minted — with exactly one of
/// `association`/`passage`/`community` present, selected by `kind`.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct EvidenceItem {
    pub(crate) candidate_id: String,
    pub(crate) kind: String,
    pub(crate) fused_rank: usize,
    pub(crate) lane_ranks: Vec<LaneRank>,
    pub(crate) citation_refs: Vec<CitationRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) corroboration: Option<Corroboration>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) contradicts: Vec<String>,
    /// This item's own content-only §8 byte contribution — excludes
    /// this field and `estimated_tokens` themselves (ADR 0006 §8, to
    /// avoid a field whose value depends on its own length). `None`
    /// only in the instant between construction and measurement in
    /// [`prepare_item`]: `content_metrics` measures the struct with
    /// both fields still absent (via `skip_serializing_if`) so the
    /// measured bytes are the real, f32-shortest wire text — not
    /// `serde_json::Value`'s f64-widened one with these two keys
    /// removed after the fact — then both are filled in before the
    /// item is ever returned; every item this module hands out has
    /// both `Some`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) estimated_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) association: Option<AssociationOut>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) passage: Option<PassageHit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) community: Option<CommunityHit>,
}

/// A candidate that did not make it into `items` (ADR 0006 §10).
/// `duplicate_of` is present only for `reason: "duplicate_passage"`
/// and names the *surviving* `candidate_id`, never a locator, so an
/// unsourced candidate can appear on either side of the reference.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct OmittedCandidate {
    pub(crate) candidate_id: String,
    pub(crate) kind: String,
    pub(crate) reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) duplicate_of: Option<String>,
}

/// The selection pipeline's own account of what it did, beyond the
/// per-item/per-omission detail (ADR 0006 §10).
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SelectionPlan {
    /// How many raw candidates [`super::fuse`]'s exact-key dedup folded
    /// into an existing group before selection ever ran.
    pub(crate) dedup_dropped: usize,
    /// How many contradiction groups (ADR 0006 §9) existed in the pool
    /// — regardless of whether they were ultimately admitted or
    /// omitted together.
    pub(crate) contradiction_groups: usize,
    pub(crate) diversity_tier_width: usize,
}

/// The full result of one [`select`] call (ADR 0006 §10's
/// `EvidencePackage`, minus the fields only #305's HTTP/MCP surface
/// knows how to fill in — `plan.lanes`/`plan.reranker`).
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SelectedEvidence {
    pub(crate) items: Vec<EvidenceItem>,
    pub(crate) citations: Vec<CitationEntry>,
    pub(crate) budget: BudgetUsage,
    /// Capped at [`MAX_LISTED_ISSUES`], the same way `Issue` lists are
    /// (`src/api.rs:437`) — `omitted_total`/`omitted_by_reason` below
    /// are never capped, so the itemized list's own truncation is
    /// always observable.
    pub(crate) omitted: Vec<OmittedCandidate>,
    pub(crate) omitted_total: usize,
    pub(crate) omitted_by_reason: BTreeMap<String, usize>,
    pub(crate) selection: SelectionPlan,
}

const REASON_DUPLICATE_PASSAGE: &str = "duplicate_passage";
/// `pub(crate)` so `evaluate`'s tests can build an assembly-mode
/// `omitted_by_reason` against the real vocabulary instead of a
/// literal that could silently drift from it.
pub(crate) const REASON_BUDGET_EXCEEDED: &str = "budget_exceeded";
const REASON_CONTRADICTION_GROUP_EXCEEDS_BUDGET: &str = "contradiction_group_exceeds_budget";

/// How many of `omitted_by_reason`'s entries a budget ceiling
/// specifically dropped — excludes [`REASON_DUPLICATE_PASSAGE`], which
/// runs before budget admission ever does and would drop the same
/// candidate at any budget. `pub(crate)` so `src/evaluate.rs`'s
/// `budget.omitted_rate` metric (#308) can separate a budget-driven
/// omission from a near-duplicate one when summing an assembly-mode
/// case's `omitted_by_reason` — `omitted_total` alone conflates the
/// two and would report a rate the metric's own documented definition
/// ("share … a budget ceiling dropped") does not describe — without
/// that caller needing to know this module's own reason-string
/// vocabulary itself.
pub(crate) fn budget_only_omitted(omitted_by_reason: &BTreeMap<String, usize>) -> usize {
    omitted_by_reason
        .get(REASON_BUDGET_EXCEEDED)
        .copied()
        .unwrap_or(0)
        + omitted_by_reason
            .get(REASON_CONTRADICTION_GROUP_EXCEEDS_BUDGET)
            .copied()
            .unwrap_or(0)
}

/// Character-bigram Dice-coefficient threshold a passage pair must
/// meet or exceed to be treated as near-duplicate (ADR 0006 §9: "#304
/// fixes and documents one deterministic similarity function … not
/// request-configurable"). Chosen conservatively high so only
/// near-identical text — not merely topically related passages — is
/// suppressed; no wire field ever exposes the raw coefficient, so
/// nothing outside this module needs the exact value, only that it be
/// fixed and documented once.
const NEAR_DUPLICATE_DICE_THRESHOLD: f64 = 0.9;

/// Runs the whole ADR 0006 §9 pipeline over an already fused pool.
///
/// `dedup_dropped` is [`super::fuse`]'s own return value, folded
/// straight into [`SelectionPlan`] so the caller does not have to
/// thread it through separately. `citation_lookup` resolves an
/// association candidate's origin locators to the citation text that
/// belongs in the package's top-level `citations` array — a plain map
/// rather than a live corpus call, since this function is otherwise
/// pure; #305's HTTP handler is what will populate it from a real
/// `cite_passage`-equivalent lookup before calling this function. A
/// locator this map has no entry for is silently dropped from that
/// item's own `citation_refs` rather than ever creating an orphan
/// reference (ADR 0006 §9 I1) — in the real pipeline every origin
/// locator traces back to an attribution the graph engine itself
/// produced, so this path is a defensive guarantee, not a real degrade
/// mode.
///
/// `order` names a complete reordering of `0..len`: same length, every
/// index in range, no index repeated. A private copy of the same check
/// [`super::rerank`] itself already runs on a provider's raw response —
/// duplicated deliberately, so [`select_with_reorder`] never trusts a
/// permutation's validity to whichever caller happened to construct it.
fn is_valid_permutation(order: &[usize], len: usize) -> bool {
    if order.len() != len {
        return false;
    }
    let mut seen = vec![false; len];
    for &index in order {
        if index >= len || seen[index] {
            return false;
        }
        seen[index] = true;
    }
    true
}

pub(crate) fn select(
    fused: Vec<FusedCandidate>,
    dedup_dropped: usize,
    limits: &BudgetLimits,
    citation_lookup: &HashMap<(String, u32), Citation>,
) -> SelectedEvidence {
    select_with_reorder(fused, dedup_dropped, limits, citation_lookup, None)
}

/// [`select_with_reorder`]'s optional reorder callback: given the
/// survivor pool in its current order, returns a permutation to apply,
/// or `None` to leave it untouched. A named alias purely to keep the
/// function signature below readable — `clippy::type_complexity` flags
/// the unaliased form.
pub(crate) type ReorderHook<'a> = &'a mut dyn FnMut(&[FusedCandidate]) -> Option<Vec<usize>>;

/// [`select`], plus one optional reorder hook (#307, ADR 0006 §12): a
/// reranker's entire effect on this pipeline. `reorder`, when present,
/// is called exactly once with the survivor pool in its §7
/// RRF/near-duplicate-suppressed order — the same pool `meta`/
/// `groups_by_key`/`units` below are about to be derived from — and may
/// return a permutation of `0..survivors.len()` to apply before that
/// derivation happens. Grouping and diversity tiering then run over the
/// *reordered* pool, so a promoted candidate can become its
/// contradiction group's representative and can land in an earlier
/// diversity tier — reordering the input to an otherwise-unchanged,
/// invariant-preserving admission process, exactly ADR 0006 §12's
/// boundary ("a reranker may only reorder").
///
/// A `None` return, or anything [`is_valid_permutation`] rejects
/// (wrong length, a repeated index, an out-of-range index), leaves
/// `survivors` untouched — the caller is expected to have already
/// classified that failure into `plan.reranker.reason`
/// (`invalid_permutation` or otherwise); this is a second, defensive
/// check so a malformed permutation can never reach `meta` regardless
/// of what produced it.
pub(crate) fn select_with_reorder(
    fused: Vec<FusedCandidate>,
    dedup_dropped: usize,
    limits: &BudgetLimits,
    citation_lookup: &HashMap<(String, u32), Citation>,
    reorder: Option<ReorderHook<'_>>,
) -> SelectedEvidence {
    let (mut survivors, mut omitted, mut omitted_by_reason) =
        suppress_near_duplicate_passages(fused);

    if let Some(reorder) = reorder
        && let Some(order) = reorder(&survivors)
        && is_valid_permutation(&order, survivors.len())
    {
        let mut slots: Vec<Option<FusedCandidate>> = survivors.into_iter().map(Some).collect();
        survivors = order
            .into_iter()
            .map(|index| {
                slots[index]
                    .take()
                    .expect("permutation visits every index once")
            })
            .collect();
    }

    let meta: Vec<SurvivorMeta> = survivors.iter().map(SurvivorMeta::of).collect();
    let groups_by_key = group_by_subject_label(&meta);
    let units = build_admission_units(&meta, &groups_by_key);
    let contradiction_groups = units.iter().filter(|unit| unit.is_group()).count();

    let tier_width = std::cmp::max(1, limits.max_items / 4);
    let ordered_units = diversity_reorder(units, &meta, tier_width);

    // Every survivor's other-group-members' candidate_ids, computed
    // once from `meta` before `survivors` is consumed below — a group
    // member needs every *other* member's id (ADR 0006 §10's
    // `contradicts`), not its own.
    let mut contradicts_by_index: Vec<Vec<String>> = vec![Vec::new(); survivors.len()];
    for idxs in groups_by_key.values() {
        if idxs.len() < 2 {
            continue;
        }
        let ids: Vec<String> = idxs.iter().map(|&i| meta[i].candidate_id.clone()).collect();
        for &i in idxs {
            contradicts_by_index[i] = ids
                .iter()
                .filter(|id| **id != meta[i].candidate_id)
                .cloned()
                .collect();
        }
    }

    let mut prepared: Vec<Option<PreparedItem>> = survivors
        .into_iter()
        .zip(contradicts_by_index)
        .map(|(fc, contradicts)| Some(prepare_item(fc, contradicts, citation_lookup)))
        .collect();

    let mut items_out = Vec::new();
    let mut citations = CitationCollector::new();
    let mut totals = Totals::default();

    for unit in &ordered_units {
        let idxs = unit.indices();

        let mut unit_items_bytes = 0usize;
        let mut unit_items_quarters: u64 = 0;
        // The new citation entries this unit would introduce — a
        // scratch collector so a locator two items in the same group
        // both cite is only counted once, checked against the package
        // built so far (`citations`) before it is even considered new.
        let mut scratch = CitationCollector::new();
        let mut new_citations_bytes = 0usize;
        let mut new_citations_quarters: u64 = 0;
        for &i in idxs {
            let candidate = prepared[i]
                .as_ref()
                .expect("each survivor belongs to exactly one unit");
            unit_items_bytes += candidate.item_bytes;
            unit_items_quarters += candidate.item_quarters;
            for contribution in &candidate.citation_contributions {
                if citations.contains(&contribution.source, contribution.paragraph) {
                    continue;
                }
                let newly_added = scratch.insert(
                    contribution.source.clone(),
                    contribution.paragraph,
                    contribution.citation.clone(),
                );
                if newly_added {
                    new_citations_bytes += contribution.bytes;
                    new_citations_quarters += contribution.quarters;
                }
            }
        }

        let candidate_totals = totals.plus_unit(UnitCost {
            items: idxs.len(),
            items_bytes: unit_items_bytes,
            items_quarters: unit_items_quarters,
            citations: scratch.len(),
            citations_bytes: new_citations_bytes,
            citations_quarters: new_citations_quarters,
        });

        if !candidate_totals.fits(limits) {
            // Skip this unit and keep walking the rest of the order
            // (ADR 0006 §3 D) — never stop at the first over-budget
            // candidate.
            let reason = if idxs.len() > 1 {
                REASON_CONTRADICTION_GROUP_EXCEEDS_BUDGET
            } else {
                REASON_BUDGET_EXCEEDED
            };
            for &i in idxs {
                let candidate = prepared[i].as_ref().expect("not yet taken");
                omitted.push(OmittedCandidate {
                    candidate_id: candidate.item.candidate_id.clone(),
                    kind: candidate.item.kind.clone(),
                    reason: reason.to_string(),
                    duplicate_of: None,
                });
                *omitted_by_reason.entry(reason.to_string()).or_insert(0) += 1;
            }
            continue;
        }

        totals = candidate_totals;
        for entry in scratch.into_entries() {
            citations.insert(entry.source, entry.paragraph, entry.citation);
        }
        for &i in idxs {
            let prepared_item = prepared[i]
                .take()
                .expect("each survivor committed at most once");
            items_out.push(prepared_item.item);
        }
    }

    let mut citations_out = citations.into_entries();
    citations_out
        .sort_by(|a, b| (a.source.as_str(), a.paragraph).cmp(&(b.source.as_str(), b.paragraph)));

    let omitted_total = omitted.len();
    omitted.truncate(MAX_LISTED_ISSUES);

    SelectedEvidence {
        items: items_out,
        citations: citations_out,
        budget: BudgetUsage {
            items_used: totals.items_count,
            bytes_used: totals.bytes(),
            tokens_used: totals.tokens(),
            limits: *limits,
        },
        omitted,
        omitted_total,
        omitted_by_reason,
        selection: SelectionPlan {
            dedup_dropped,
            contradiction_groups,
            diversity_tier_width: tier_width,
        },
    }
}

/// One admission unit's own cost contribution — what [`Totals::plus_unit`]
/// needs to compute the candidate post-admission state, without that
/// state itself.
struct UnitCost {
    items: usize,
    items_bytes: usize,
    items_quarters: u64,
    citations: usize,
    citations_bytes: usize,
    citations_quarters: u64,
}

/// The three-budget running account the greedy admission walk updates
/// as it commits units (ADR 0006 §8). Bundles what would otherwise be
/// six separately-threaded counters (and their six `new_*` shadows per
/// iteration) into one value, and gives the in-loop "would this still
/// fit" check and the final `BudgetUsage` the same `bytes()`/
/// `tokens()` formula instead of two independent copies of it.
#[derive(Clone, Copy, Default)]
struct Totals {
    items_count: usize,
    items_bytes_sum: usize,
    items_quarters_sum: u64,
    citations_count: usize,
    citations_bytes_sum: usize,
    citations_quarters_sum: u64,
}

impl Totals {
    /// The `items` array's own compact-JSON byte length plus the
    /// `citations` array's (ADR 0006 §8: "exactly the items array plus
    /// the citations array").
    fn bytes(&self) -> usize {
        array_overhead(self.items_count)
            + self.items_bytes_sum
            + array_overhead(self.citations_count)
            + self.citations_bytes_sum
    }

    /// The combined quarter-tokens both arrays' text contributes — the
    /// ceiling to whole tokens is taken once, over this sum, not once
    /// per array (ADR 0006 §8: "the estimated total is the ceiling of
    /// the sum").
    fn quarters(&self) -> u64 {
        array_overhead(self.items_count) as u64
            + self.items_quarters_sum
            + array_overhead(self.citations_count) as u64
            + self.citations_quarters_sum
    }

    fn tokens(&self) -> usize {
        tokens_from_quarters(self.quarters())
    }

    fn fits(&self, limits: &BudgetLimits) -> bool {
        self.items_count <= limits.max_items
            && self.bytes() <= limits.max_bytes
            && self.tokens() <= limits.max_tokens
    }

    /// The state after admitting one more unit, computed without
    /// mutating `self` — the caller checks [`Totals::fits`] on the
    /// result before deciding whether to keep it (ADR 0006 §9's
    /// "skip and continue" admission rule needs the candidate state
    /// available without committing to it).
    fn plus_unit(&self, cost: UnitCost) -> Self {
        Self {
            items_count: self.items_count + cost.items,
            items_bytes_sum: self.items_bytes_sum + cost.items_bytes,
            items_quarters_sum: self.items_quarters_sum + cost.items_quarters,
            citations_count: self.citations_count + cost.citations,
            citations_bytes_sum: self.citations_bytes_sum + cost.citations_bytes,
            citations_quarters_sum: self.citations_quarters_sum + cost.citations_quarters,
        }
    }
}

/// Lightweight, cheaply-cloned facts about one survivor, extracted
/// before `survivors` is consumed into owned [`EvidenceItem`]s —
/// contradiction grouping and diversity tiering only ever need these,
/// never the full candidate/payload.
struct SurvivorMeta {
    candidate_id: String,
    source: Option<String>,
    subject_label: Option<(String, String)>,
}

impl SurvivorMeta {
    fn of(fc: &FusedCandidate) -> Self {
        let candidate = &fc.candidate;
        // Every constructor that produces a `CandidatePayload::Association`
        // (`from_association`, `from_activation`) also sets
        // `kind: KIND_ASSOCIATION`, and no other constructor produces
        // that payload variant — matching on the payload alone already
        // implies the kind, so no separate `kind` check is needed here.
        let subject_label = match &candidate.payload {
            CandidatePayload::Association(association) => {
                Some((association.subject.clone(), association.label.clone()))
            }
            _ => None,
        };
        Self {
            candidate_id: candidate.candidate_id.clone(),
            source: candidate.source.clone(),
            subject_label,
        }
    }
}

/// Groups association-candidate indices sharing `(subject, label)`
/// (ADR 0006 §9's contradiction rule). Every post-[`super::fuse`]
/// candidate has a unique `(subject, label, object)` triple — `fuse`'s
/// own exact-key dedup already collapsed any duplicate — so two
/// candidates sharing `(subject, label)` necessarily disagree on
/// `object`, which is exactly ADR 0006 §9's contradiction condition;
/// no further check is needed, and a same-triple/opposite-sign pair
/// (§9's other contradiction case) can never arise as two distinct
/// post-fuse candidates for the same reason, so this function does not
/// special-case it.
fn group_by_subject_label(meta: &[SurvivorMeta]) -> HashMap<(String, String), Vec<usize>> {
    let mut groups: HashMap<(String, String), Vec<usize>> = HashMap::new();
    for (i, item) in meta.iter().enumerate() {
        if let Some(key) = item.subject_label.clone() {
            groups.entry(key).or_default().push(i);
        }
    }
    groups
}

/// One admission unit: an ordinary candidate (a length-1 vec), or a
/// whole contradiction group counted as one (ADR 0006 §9 — "never
/// tiered or admitted separately"). Member indices are in the order
/// [`build_admission_units`] encountered them, so index `0` is always
/// the lowest-fused-rank (first-encountered) member — what makes it
/// double as `representative()` below.
#[derive(Clone)]
struct AdmissionUnit(Vec<usize>);

impl AdmissionUnit {
    fn indices(&self) -> &[usize] {
        &self.0
    }

    fn representative(&self) -> usize {
        self.0[0]
    }

    /// Whether this unit is a whole contradiction group rather than an
    /// ordinary single candidate.
    fn is_group(&self) -> bool {
        self.0.len() > 1
    }
}

/// Walks `meta` in its already fused-rank order, emitting one
/// [`AdmissionUnit`] per candidate — a whole [`AdmissionUnit::Group`]
/// the first time any of its members is reached, so the unit's
/// position is always its lowest-fused-rank member's position.
fn build_admission_units(
    meta: &[SurvivorMeta],
    groups_by_key: &HashMap<(String, String), Vec<usize>>,
) -> Vec<AdmissionUnit> {
    let mut consumed = vec![false; meta.len()];
    let mut units = Vec::new();
    for i in 0..meta.len() {
        if consumed[i] {
            continue;
        }
        let group = meta[i]
            .subject_label
            .as_ref()
            .and_then(|key| groups_by_key.get(key))
            .filter(|idxs| idxs.len() >= 2);
        match group {
            Some(idxs) => {
                for &j in idxs {
                    consumed[j] = true;
                }
                units.push(AdmissionUnit(idxs.clone()));
            }
            None => {
                consumed[i] = true;
                units.push(AdmissionUnit(vec![i]));
            }
        }
    }
    units
}

/// Tier-based round-robin reordering (ADR 0006 §9): a pure ordering
/// pass, independent of whatever budget admission later decides — it
/// changes admission *order*, never which candidates end up in
/// `items`. Units are chunked into consecutive tiers of `tier_width`;
/// within a tier, a unit whose primary source has not yet appeared —
/// anywhere earlier in the order this function has built so far,
/// across every prior tier and every earlier unit in the current tier
/// — is placed ahead of one whose source has. An unsourced unit
/// (`source: None`) is always treated as novel and never marked seen.
fn diversity_reorder(
    units: Vec<AdmissionUnit>,
    meta: &[SurvivorMeta],
    tier_width: usize,
) -> Vec<AdmissionUnit> {
    let mut seen_sources: HashSet<String> = HashSet::new();
    let mut output = Vec::with_capacity(units.len());
    for chunk in units.chunks(tier_width) {
        let mut novel = Vec::new();
        let mut repeat = Vec::new();
        let mut newly_seen_this_tier: HashSet<String> = HashSet::new();
        for unit in chunk {
            match &meta[unit.representative()].source {
                None => novel.push(unit.clone()),
                Some(source) => {
                    if seen_sources.contains(source) || newly_seen_this_tier.contains(source) {
                        repeat.push(unit.clone());
                    } else {
                        newly_seen_this_tier.insert(source.clone());
                        novel.push(unit.clone());
                    }
                }
            }
        }
        seen_sources.extend(newly_seen_this_tier);
        output.extend(novel);
        output.extend(repeat);
    }
    output
}

/// A borrowing view of [`CitationEntry`], serialized identically (same
/// field names, same order) purely to measure a citation's content
/// bytes/quarters without cloning the `Citation` it borrows — the
/// clone `content_metrics` would otherwise need is exactly what
/// [`CitationContribution`] already owns a moment later.
#[derive(Serialize)]
struct CitationEntryRef<'a> {
    source: &'a str,
    paragraph: u32,
    citation: &'a Citation,
}

/// One resolved citation locator a survivor would contribute if it (or
/// its containing group) is admitted, with its own content bytes/
/// quarters precomputed once (ADR 0006 §8's "computes every
/// candidate's content byte count first … uses that fixed number for
/// every admission decision").
struct CitationContribution {
    source: String,
    paragraph: u32,
    citation: Citation,
    bytes: usize,
    quarters: u64,
}

struct PreparedItem {
    item: EvidenceItem,
    item_bytes: usize,
    item_quarters: u64,
    citation_contributions: Vec<CitationContribution>,
}

/// Builds one survivor's [`EvidenceItem`] and precomputes everything
/// the greedy admission walk needs to decide, without yet knowing
/// whether it will be admitted.
fn prepare_item(
    fc: FusedCandidate,
    mut contradicts: Vec<String>,
    citation_lookup: &HashMap<(String, u32), Citation>,
) -> PreparedItem {
    contradicts.sort();

    let FusedCandidate {
        candidate,
        lane_ranks,
        fused_rank,
        ..
    } = fc;
    let raw_refs = candidate.citation_refs();
    let corroborating_sources = candidate.corroborating_sources();
    let EvidenceCandidate {
        kind,
        candidate_id,
        payload,
        ..
    } = candidate;

    let mut resolved: Vec<(String, u32, Citation)> = Vec::new();
    for (source, paragraph) in raw_refs {
        if let Some(citation) = citation_lookup.get(&(source.clone(), paragraph)) {
            resolved.push((source, paragraph, citation.clone()));
        }
    }
    resolved.sort_by(|a, b| (a.0.as_str(), a.1).cmp(&(b.0.as_str(), b.1)));

    let citation_refs: Vec<CitationRef> = resolved
        .iter()
        .map(|(source, paragraph, _)| CitationRef {
            source: source.clone(),
            paragraph: *paragraph,
        })
        .collect();

    // Gates on `corroborating_sources`, not `citation_refs`: the two
    // differ exactly when an attribution names a source but no
    // paragraph (so it never entered `origins`/`citation_refs`, ADR
    // 0006 §6) or when a located locator's citation text failed to
    // resolve (`citation_lookup` had no entry). `corroborating_sources`
    // already reads every attribution directly for this reason
    // (`EvidenceCandidate::corroborating_sources`'s own doc) — gating
    // on `citation_refs` here silently dropped that guarantee and made
    // a graph-only, paragraph-less fact (or one whose citation could
    // not be located) report no corroboration at all, even with two or
    // more independent sources, violating ADR 0006 §9 I4 ("corroboration
    // is never silently collapsed"). `attributions` still only ever
    // lists what actually resolved to citation text — a source with no
    // resolvable locator is named in `sources` but contributes no
    // `attributions` entry.
    let corroboration = if kind == KIND_ASSOCIATION && !corroborating_sources.is_empty() {
        Some(Corroboration {
            sources: corroborating_sources.into_iter().collect(),
            attributions: citation_refs.clone(),
        })
    } else {
        None
    };

    let (association, passage, community) = match payload {
        CandidatePayload::Association(association) => (Some(association), None, None),
        CandidatePayload::Passage(passage) => (None, Some(passage), None),
        CandidatePayload::Community(community) => (None, None, Some(community)),
    };

    let mut item = EvidenceItem {
        candidate_id,
        kind,
        fused_rank,
        lane_ranks,
        citation_refs,
        corroboration,
        contradicts,
        bytes: None,
        estimated_tokens: None,
        association,
        passage,
        community,
    };
    let (item_bytes, item_quarters) = content_metrics(&item);
    item.bytes = Some(item_bytes);
    item.estimated_tokens = Some(tokens_from_quarters(item_quarters));

    let citation_contributions = resolved
        .into_iter()
        .map(|(source, paragraph, citation)| {
            // Measures through a borrowing view so the (potentially
            // large, if the citation text is a full passage excerpt)
            // `Citation` is never cloned just to size it — `source`/
            // `citation` below are moved into `CitationContribution`
            // once, not cloned.
            let probe = CitationEntryRef {
                source: source.as_str(),
                paragraph,
                citation: &citation,
            };
            let (bytes, quarters) = content_metrics(&probe);
            CitationContribution {
                source,
                paragraph,
                citation,
                bytes,
                quarters,
            }
        })
        .collect();

    PreparedItem {
        item,
        item_bytes,
        item_quarters,
        citation_contributions,
    }
}

/// Drops textually near-duplicate *passage* candidates from the pool
/// (ADR 0006 §9), staged over the already fused-rank-ordered input so
/// "keep the higher-ranked (lower `fused_rank`) candidate" is
/// well-defined. Association/community candidates always pass through
/// unchanged — this detector only ever compares passage text.
fn suppress_near_duplicate_passages(
    fused: Vec<FusedCandidate>,
) -> (
    Vec<FusedCandidate>,
    Vec<OmittedCandidate>,
    BTreeMap<String, usize>,
) {
    let mut survivors = Vec::with_capacity(fused.len());
    let mut omitted = Vec::new();
    let mut omitted_by_reason: BTreeMap<String, usize> = BTreeMap::new();
    // (candidate_id, normalized-text bigrams) for every passage kept so
    // far, in fused-rank order.
    let mut kept_passages: Vec<(String, HashSet<(char, char)>)> = Vec::new();

    for fc in fused {
        let bigrams = match &fc.candidate.payload {
            CandidatePayload::Passage(hit) => {
                Some(char_bigrams(&taguru::context::normalize_entry(&hit.text)))
            }
            _ => None,
        };

        let Some(bigrams) = bigrams else {
            survivors.push(fc);
            continue;
        };

        let duplicate_of = kept_passages
            .iter()
            .find(|(_, kept_bigrams)| could_be_near_duplicate(kept_bigrams, &bigrams))
            .map(|(id, _)| id.clone());

        match duplicate_of {
            Some(survivor_id) => {
                omitted.push(OmittedCandidate {
                    candidate_id: fc.candidate.candidate_id.clone(),
                    kind: fc.candidate.kind.clone(),
                    reason: REASON_DUPLICATE_PASSAGE.to_string(),
                    duplicate_of: Some(survivor_id),
                });
                *omitted_by_reason
                    .entry(REASON_DUPLICATE_PASSAGE.to_string())
                    .or_insert(0) += 1;
            }
            None => {
                kept_passages.push((fc.candidate.candidate_id.clone(), bigrams));
                survivors.push(fc);
            }
        }
    }

    (survivors, omitted, omitted_by_reason)
}

/// Character bigrams of a normalized string, as a set (not a
/// multiset) — good enough for a fixed, documented near-duplicate
/// threshold (ADR 0006 §9) without pulling in a second, unrelated
/// weighting scheme. A string under two scalars has no bigram; it is
/// paired with itself instead, so two identical one-character strings
/// still compare as an exact match rather than the undefined 0/0 case.
fn char_bigrams(normalized: &str) -> HashSet<(char, char)> {
    let chars: Vec<char> = normalized.chars().collect();
    if chars.len() < 2 {
        return chars.iter().map(|&c| (c, c)).collect();
    }
    chars.windows(2).map(|pair| (pair[0], pair[1])).collect()
}

/// Dice's coefficient over two bigram sets: `2|A∩B| / (|A|+|B|)`, `1.0`
/// when both are empty (two empty strings are identical, not
/// incomparable).
fn dice_coefficient(a: &HashSet<(char, char)>, b: &HashSet<(char, char)>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let shared = a.intersection(b).count();
    2.0 * shared as f64 / (a.len() + b.len()) as f64
}

/// A sound, O(1) upper bound on [`dice_coefficient`] — `|A∩B| <=
/// min(|A|,|B|)`, so `dice <= 2*min(|A|,|B|) / (|A|+|B|)` always — used
/// to skip the O(min(|A|,|B|)) real intersection for a pair whose
/// bound alone already falls under the threshold. Never a false
/// negative: this can only over-estimate the real coefficient, so
/// [`suppress_near_duplicate_passages`]'s O(n²) candidate-pair walk
/// pays the real intersection only for pairs that could plausibly
/// match, without changing which pairs it calls near-duplicate.
fn could_be_near_duplicate(a: &HashSet<(char, char)>, b: &HashSet<(char, char)>) -> bool {
    let denom = a.len() + b.len();
    if denom == 0 {
        return true; // two empty strings: dice_coefficient's own 1.0 case.
    }
    let upper_bound = 2.0 * a.len().min(b.len()) as f64 / denom as f64;
    upper_bound >= NEAR_DUPLICATE_DICE_THRESHOLD
        && dice_coefficient(a, b) >= NEAR_DUPLICATE_DICE_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::communities::CommunityHitMember;
    use crate::api::sources::{LaneEvidence, PassageLanes};
    use crate::api::{ActivationOut, AttributionOut};

    fn limits(max_items: usize, max_bytes: usize, max_tokens: usize) -> BudgetLimits {
        BudgetLimits {
            max_items,
            max_bytes,
            max_tokens,
        }
    }

    fn generous_limits() -> BudgetLimits {
        limits(1000, 10_000_000, 10_000_000)
    }

    fn association_out(subject: &str, label: &str, object: &str, weight: f64) -> AssociationOut {
        AssociationOut {
            subject: subject.to_string(),
            label: label.to_string(),
            object: object.to_string(),
            weight,
            count: 1,
            attributions: Vec::new(),
        }
    }

    fn sourced_association(
        subject: &str,
        label: &str,
        object: &str,
        weight: f64,
        source: &str,
        paragraph: u32,
    ) -> AssociationOut {
        let mut association = association_out(subject, label, object, weight);
        association.attributions = vec![AttributionOut {
            source: source.to_string(),
            weight,
            count: 1,
            paragraph: Some(paragraph),
            section: None,
            locator: None,
        }];
        association
    }

    fn passage_hit(source: &str, paragraph: u32, text: &str) -> PassageHit {
        PassageHit {
            source: source.to_string(),
            paragraph,
            score: 1.0,
            text: text.to_string(),
            lanes: PassageLanes {
                bm25: Some(LaneEvidence {
                    rank: 1,
                    score: 1.0,
                }),
                vector: None,
            },
        }
    }

    fn community_hit(id: &str, paragraph: u32, text: &str) -> CommunityHit {
        CommunityHit {
            community: id.to_string(),
            score: 1.0,
            text: text.to_string(),
            paragraph,
            level: None,
            parent: None,
            concept_count: None,
            members: vec![CommunityHitMember {
                name: "member".to_string(),
                strength: 1.0,
            }],
            members_truncated: false,
        }
    }

    fn citation_for(source: &str, section: Option<&str>) -> Citation {
        Citation {
            text: format!("excerpt of {source}"),
            source: source.to_string(),
            section: section.map(|s| s.to_string()),
            locator: None,
        }
    }

    fn empty_lookup() -> HashMap<(String, u32), Citation> {
        HashMap::new()
    }

    fn select_all(
        pool: Vec<EvidenceCandidate>,
        citation_lookup: &HashMap<(String, u32), Citation>,
    ) -> SelectedEvidence {
        let (fused, dedup_dropped) = super::super::fuse(pool);
        select(fused, dedup_dropped, &generous_limits(), citation_lookup)
    }

    /// Independently recomputes `BudgetUsage.bytes_used`/`tokens_used`
    /// from a finished `items`/`citations` pair, through the same
    /// public `array_overhead`/`content_metrics`/`tokens_from_quarters`
    /// helpers `select` itself uses internally — the shared
    /// recomputation both `budgets_are_hard_ceilings_and_every_
    /// omission_is_counted` and `reorder_can_change_admission_but_
    /// budgets_still_hold` below check the reported `BudgetUsage`
    /// against, catching any internal running-sum drift from what the
    /// output actually serializes to. Each item is measured with its
    /// own `bytes`/`estimated_tokens` set back to `None` first — the
    /// same state [`prepare_item`] measures under — so this goes
    /// through the real f32-shortest serialization `content_metrics`
    /// requires, never a `serde_json::Value`-widened approximation of
    /// it (the regression this recomputation exists to catch: see
    /// `budget::tests::content_metrics_does_not_widen_f32_to_f64_precision`).
    fn recomputed_budget_usage(
        items: Vec<EvidenceItem>,
        citations: &[CitationEntry],
    ) -> (usize, usize) {
        let items_len = items.len();
        let citations_len = citations.len();

        let (items_bytes, items_quarters) = items
            .into_iter()
            .map(|item| {
                let EvidenceItem {
                    candidate_id,
                    kind,
                    fused_rank,
                    lane_ranks,
                    citation_refs,
                    corroboration,
                    contradicts,
                    association,
                    passage,
                    community,
                    ..
                } = item;
                let probe = EvidenceItem {
                    candidate_id,
                    kind,
                    fused_rank,
                    lane_ranks,
                    citation_refs,
                    corroboration,
                    contradicts,
                    bytes: None,
                    estimated_tokens: None,
                    association,
                    passage,
                    community,
                };
                content_metrics(&probe)
            })
            .fold((0usize, 0u64), |(b, q), (ib, iq)| (b + ib, q + iq));

        let (citations_bytes, citations_quarters) = citations
            .iter()
            .map(content_metrics)
            .fold((0usize, 0u64), |(b, q), (ib, iq)| (b + ib, q + iq));

        let bytes_used = array_overhead(items_len)
            + items_bytes
            + array_overhead(citations_len)
            + citations_bytes;
        let total_quarters = array_overhead(items_len) as u64
            + items_quarters
            + array_overhead(citations_len) as u64
            + citations_quarters;
        (bytes_used, tokens_from_quarters(total_quarters))
    }

    // --- `could_be_near_duplicate`'s O(1) bound never disagrees with
    // the real `dice_coefficient` it stands in for ---

    #[test]
    fn could_be_near_duplicate_agrees_with_dice_coefficient_below_threshold() {
        // Same size, zero overlap: the O(1) size bound alone (1.0)
        // cannot prune this pair, so `could_be_near_duplicate` falls
        // through to the real intersection — which correctly finds
        // none — and must still agree with `dice_coefficient` on the
        // verdict.
        let a = char_bigrams("aa");
        let b = char_bigrams("bb");
        assert!(dice_coefficient(&a, &b) < NEAR_DUPLICATE_DICE_THRESHOLD);
        assert!(!could_be_near_duplicate(&a, &b));
    }

    #[test]
    fn could_be_near_duplicate_prunes_a_size_mismatched_pair_without_the_real_intersection() {
        // |a| = 20 distinct bigrams, |b| = 1: upper bound =
        // 2*1/21 ≈ 0.095, well under 0.9 — the O(1) bound alone must
        // reject this pair, short-circuiting before `dice_coefficient`
        // ever runs its O(min(|a|,|b|)) intersection.
        let a: HashSet<(char, char)> = ('a'..='t').zip('b'..='u').collect();
        assert_eq!(a.len(), 20);
        let mut b = HashSet::new();
        b.insert(('x', 'y'));
        assert!(dice_coefficient(&a, &b) < NEAR_DUPLICATE_DICE_THRESHOLD);
        assert!(!could_be_near_duplicate(&a, &b));
    }

    #[test]
    fn could_be_near_duplicate_agrees_with_dice_coefficient_above_threshold() {
        let a = char_bigrams(&taguru::context::normalize_entry("identical passage text"));
        let b = char_bigrams(&taguru::context::normalize_entry("identical passage text"));
        assert!(dice_coefficient(&a, &b) >= NEAR_DUPLICATE_DICE_THRESHOLD);
        assert!(could_be_near_duplicate(&a, &b));
    }

    #[test]
    fn could_be_near_duplicate_treats_two_empty_strings_as_a_match() {
        let a = char_bigrams("");
        let b = char_bigrams("");
        assert_eq!(dice_coefficient(&a, &b), 1.0);
        assert!(could_be_near_duplicate(&a, &b));
    }

    // --- `budget_only_omitted` isolates budget-driven reasons ---

    #[test]
    fn budget_only_omitted_excludes_duplicate_passage() {
        let mut by_reason = BTreeMap::new();
        by_reason.insert(REASON_DUPLICATE_PASSAGE.to_string(), 5);
        by_reason.insert(REASON_BUDGET_EXCEEDED.to_string(), 2);
        by_reason.insert(REASON_CONTRADICTION_GROUP_EXCEEDS_BUDGET.to_string(), 3);
        assert_eq!(budget_only_omitted(&by_reason), 5);
    }

    #[test]
    fn budget_only_omitted_of_an_empty_map_is_zero() {
        assert_eq!(budget_only_omitted(&BTreeMap::new()), 0);
    }

    /// Regression case for the bug fixed alongside `budget::content_
    /// metrics`'s own `content_metrics_does_not_widen_f32_to_f64_
    /// precision`: a passage carrying a score whose f32-shortest form
    /// (`0.86304635`) is shorter than its f64-widened one
    /// (`0.8630463480949402`) must have its admitted item's `bytes`/
    /// `estimated_tokens` reflect the former, end to end through
    /// `select` — not just at the `content_metrics` unit level.
    /// Verified against the item's own REAL wire serialization (`bytes`/
    /// `estimated_tokens` populated, as a caller actually receives it):
    /// strip the exact `,"bytes":N,"estimated_tokens":M` span — the two
    /// fields are always adjacent, in field-declaration order — and the
    /// remainder's length must equal what `select` reported as `N`.
    #[test]
    fn admitted_item_bytes_reflect_the_real_f32_shortest_score_not_an_f64_widened_one() {
        let hit = PassageHit {
            source: "s".to_string(),
            paragraph: 0,
            score: 0.86304635,
            text: "hit".to_string(),
            lanes: PassageLanes {
                bm25: Some(LaneEvidence {
                    rank: 1,
                    score: 0.86304635,
                }),
                vector: None,
            },
        };
        let candidate = EvidenceCandidate::from_passage("ctx", hit, 1);
        let result = select_all(vec![candidate], &empty_lookup());
        let item = result.items.into_iter().next().expect("one item survives");
        let reported_bytes = item.bytes.expect("select always measures admitted items");
        let reported_tokens = item
            .estimated_tokens
            .expect("select always measures admitted items");

        let full_text = serde_json::to_string(&item).unwrap();
        let marker = format!(",\"bytes\":{reported_bytes},\"estimated_tokens\":{reported_tokens}");
        assert!(
            full_text.contains(&marker),
            "expected {marker:?} in {full_text}"
        );
        let content_only = full_text.replacen(&marker, "", 1);
        assert_eq!(content_only.len(), reported_bytes);

        assert!(
            content_only.contains("0.86304635"),
            "f32's own shortest decimal form: {content_only}"
        );
        assert!(
            !content_only.contains("0.8630463480949402"),
            "must never widen through f64: {content_only}"
        );
    }

    // --- Mixed pool, determinism ---

    #[test]
    fn mixed_pool_produces_a_deterministic_package() {
        let pool = vec![
            EvidenceCandidate::from_association(
                "ctx",
                sourced_association("cats", "is_a", "mammals", 1.0, "book.txt", 1),
                1,
            ),
            EvidenceCandidate::from_passage("ctx", passage_hit("s1", 0, "cats are mammals"), 1),
            EvidenceCandidate::from_community("ctx", community_hit("L0-1", 0, "summary text"), 1),
        ];
        let mut lookup = empty_lookup();
        lookup.insert(("book.txt".to_string(), 1), citation_for("book.txt", None));

        let first = select_all(pool, &lookup);
        assert_eq!(first.items.len(), 3);
        assert_eq!(first.citations.len(), 1);
        assert_eq!(first.omitted_total, 0);

        // Rebuild the identical pool from scratch and select again —
        // I3 requires byte-identical JSON.
        let pool2 = vec![
            EvidenceCandidate::from_association(
                "ctx",
                sourced_association("cats", "is_a", "mammals", 1.0, "book.txt", 1),
                1,
            ),
            EvidenceCandidate::from_passage("ctx", passage_hit("s1", 0, "cats are mammals"), 1),
            EvidenceCandidate::from_community("ctx", community_hit("L0-1", 0, "summary text"), 1),
        ];
        let second = select_all(pool2, &lookup);
        assert_eq!(
            serde_json::to_string(&first).unwrap(),
            serde_json::to_string(&second).unwrap()
        );
    }

    // --- Budget: zero/tiny, never an error ---

    #[test]
    fn zero_item_budget_yields_an_empty_package_not_an_error() {
        let pool = vec![EvidenceCandidate::from_passage(
            "ctx",
            passage_hit("s", 0, "hello"),
            1,
        )];
        let (fused, dropped) = super::super::fuse(pool);
        let result = select(
            fused,
            dropped,
            &limits(0, 100_000, 100_000),
            &empty_lookup(),
        );
        assert!(result.items.is_empty());
        assert_eq!(result.omitted_total, 1);
        assert_eq!(result.omitted[0].reason, REASON_BUDGET_EXCEEDED);
        assert_eq!(
            result.omitted_by_reason.get(REASON_BUDGET_EXCEEDED),
            Some(&1)
        );
    }

    #[test]
    fn one_byte_budget_yields_an_empty_package_not_an_error() {
        let pool = vec![EvidenceCandidate::from_passage(
            "ctx",
            passage_hit("s", 0, "hello"),
            1,
        )];
        let (fused, dropped) = super::super::fuse(pool);
        let result = select(fused, dropped, &limits(100, 1, 100_000), &empty_lookup());
        assert!(result.items.is_empty());
        assert_eq!(result.omitted_total, 1);
    }

    #[test]
    fn zero_token_budget_yields_an_empty_package_not_an_error() {
        let pool = vec![EvidenceCandidate::from_passage(
            "ctx",
            passage_hit("s", 0, "hello"),
            1,
        )];
        let (fused, dropped) = super::super::fuse(pool);
        let result = select(fused, dropped, &limits(100, 100_000, 0), &empty_lookup());
        assert!(result.items.is_empty());
        assert_eq!(result.omitted_total, 1);
    }

    // --- Budget: skip over-budget, keep going (§3 D) ---

    #[test]
    fn an_over_budget_candidate_is_skipped_so_a_later_smaller_one_still_fits() {
        let big =
            EvidenceCandidate::from_passage("ctx", passage_hit("big", 0, &"x".repeat(500)), 1);
        let small = EvidenceCandidate::from_passage("ctx", passage_hit("small", 0, "tiny"), 2);
        let (fused, dropped) = super::super::fuse(vec![big, small]);
        // A byte budget that fits the small item alone but not the big
        // one first in the walk.
        let result = select(fused, dropped, &limits(10, 400, 100_000), &empty_lookup());
        assert_eq!(result.items.len(), 1);
        assert!(result.items[0].candidate_id.contains("small"));
        assert!(
            result
                .omitted
                .iter()
                .any(|o| o.candidate_id.contains("big") && o.reason == REASON_BUDGET_EXCEEDED)
        );
    }

    // --- I1: no orphan citations, both directions ---

    #[test]
    fn every_citation_ref_has_a_matching_entry_and_vice_versa() {
        let association = sourced_association("cats", "is_a", "mammals", 1.0, "book.txt", 1);
        let pool = vec![EvidenceCandidate::from_association("ctx", association, 1)];
        let mut lookup = empty_lookup();
        lookup.insert(("book.txt".to_string(), 1), citation_for("book.txt", None));
        let result = select_all(pool, &lookup);

        assert_eq!(result.items.len(), 1);
        let refs = &result.items[0].citation_refs;
        assert_eq!(refs.len(), 1);
        assert!(
            result
                .citations
                .iter()
                .any(|c| c.source == refs[0].source && c.paragraph == refs[0].paragraph)
        );
        for entry in &result.citations {
            assert!(result.items.iter().any(|item| {
                item.citation_refs
                    .iter()
                    .any(|r| r.source == entry.source && r.paragraph == entry.paragraph)
            }));
        }
    }

    #[test]
    fn an_unresolvable_origin_locator_is_dropped_rather_than_left_orphaned() {
        let association = sourced_association("cats", "is_a", "mammals", 1.0, "missing.txt", 1);
        let pool = vec![EvidenceCandidate::from_association("ctx", association, 1)];
        // Empty lookup: "missing.txt" paragraph 1 has no citation text.
        let result = select_all(pool, &empty_lookup());
        assert_eq!(result.items.len(), 1);
        assert!(result.items[0].citation_refs.is_empty());
        assert!(result.citations.is_empty());
    }

    /// ADR 0006 §9 Admission's last sentence: "a citation locator
    /// already present in the package … costs nothing extra when a
    /// later unit reuses it." Two distinct association candidates
    /// (different `(subject, label, object)`, so two separate admission
    /// units) that cite the same `(source, paragraph)` must only add
    /// that citation's bytes/tokens once — a budget sized exactly to
    /// the correctly-deduped usage still admits both, which would fail
    /// if the second unit's citation were billed again.
    #[test]
    fn a_shared_citation_locator_is_not_double_counted_across_two_admission_units() {
        fn two_candidates_sharing_one_citation() -> Vec<EvidenceCandidate> {
            let a = sourced_association("cats", "is_a", "mammals", 1.0, "book.txt", 1);
            let b = sourced_association("dogs", "is_a", "mammals", 1.0, "book.txt", 1);
            vec![
                EvidenceCandidate::from_association("ctx", a, 1),
                EvidenceCandidate::from_association("ctx", b, 2),
            ]
        }
        let mut lookup = empty_lookup();
        lookup.insert(("book.txt".to_string(), 1), citation_for("book.txt", None));

        let generous = select_all(two_candidates_sharing_one_citation(), &lookup);
        assert_eq!(generous.items.len(), 2);
        assert_eq!(
            generous.citations.len(),
            1,
            "one shared citation entry, not two"
        );

        // A byte budget set to exactly the generous run's usage: if the
        // second item's shared citation were billed again, this tight
        // budget would only have room for one item.
        let (fused, dropped) = super::super::fuse(two_candidates_sharing_one_citation());
        let tight = select(
            fused,
            dropped,
            &limits(10, generous.budget.bytes_used, 100_000),
            &lookup,
        );
        assert_eq!(
            tight.items.len(),
            2,
            "both items still fit — the reused citation cost nothing extra"
        );
        assert_eq!(tight.budget.bytes_used, generous.budget.bytes_used);
    }

    // --- I4: corroboration never truncated ---

    #[test]
    fn corroboration_names_every_independent_source() {
        let mut association = association_out("cats", "is_a", "mammals", 1.0);
        association.attributions = vec![
            AttributionOut {
                source: "a.txt".to_string(),
                weight: 1.0,
                count: 1,
                paragraph: Some(1),
                section: None,
                locator: None,
            },
            AttributionOut {
                source: "b.txt".to_string(),
                weight: 1.0,
                count: 1,
                paragraph: Some(2),
                section: None,
                locator: None,
            },
        ];
        let pool = vec![EvidenceCandidate::from_association("ctx", association, 1)];
        let mut lookup = empty_lookup();
        lookup.insert(("a.txt".to_string(), 1), citation_for("a.txt", None));
        lookup.insert(("b.txt".to_string(), 2), citation_for("b.txt", None));
        let result = select_all(pool, &lookup);

        let corroboration = result.items[0]
            .corroboration
            .as_ref()
            .expect("association with origins has corroboration");
        assert_eq!(
            corroboration.sources,
            vec!["a.txt".to_string(), "b.txt".to_string()]
        );
        assert_eq!(corroboration.attributions.len(), 2);
    }

    #[test]
    fn corroboration_survives_a_paragraph_less_attribution() {
        // An attribution naming a source but no paragraph never enters
        // `origins`/`citation_refs` (`origins_of`'s own filter, ADR
        // 0006 §6) — but it still corroborates the fact and must still
        // be named in `corroboration.sources`. Regression for gating
        // `corroboration` on `citation_refs.is_empty()`, which dropped
        // this source (and every sibling attribution's source) entirely
        // whenever none of an association's attributions carried a
        // paragraph.
        let mut association = association_out("cats", "is_a", "mammals", 1.0);
        association.attributions = vec![AttributionOut {
            source: "a.txt".to_string(),
            weight: 1.0,
            count: 1,
            paragraph: None,
            section: None,
            locator: None,
        }];
        let pool = vec![EvidenceCandidate::from_association("ctx", association, 1)];
        let result = select_all(pool, &empty_lookup());

        let corroboration = result.items[0]
            .corroboration
            .as_ref()
            .expect("a paragraph-less attribution still corroborates");
        assert_eq!(corroboration.sources, vec!["a.txt".to_string()]);
        assert!(
            corroboration.attributions.is_empty(),
            "no paragraph locator to name — attributions stays empty, not the whole field"
        );
    }

    #[test]
    fn corroboration_survives_an_unresolvable_citation() {
        // A real paragraph locator whose citation text never resolved
        // (the passage was retracted, say) still corroborates the
        // fact — the missing citation text is `citation_refs`'s
        // problem, not `corroboration`'s.
        let association = sourced_association("cats", "is_a", "mammals", 1.0, "a.txt", 1);
        let pool = vec![EvidenceCandidate::from_association("ctx", association, 1)];
        // Empty lookup: "a.txt" paragraph 1 has no citation text.
        let result = select_all(pool, &empty_lookup());

        let corroboration = result.items[0]
            .corroboration
            .as_ref()
            .expect("an unresolvable citation still corroborates");
        assert_eq!(corroboration.sources, vec!["a.txt".to_string()]);
        assert!(corroboration.attributions.is_empty());
    }

    #[test]
    fn an_unsourced_association_has_no_corroboration() {
        let pool = vec![EvidenceCandidate::from_association(
            "ctx",
            association_out("a", "l", "b", 1.0),
            1,
        )];
        let result = select_all(pool, &empty_lookup());
        assert!(result.items[0].corroboration.is_none());
    }

    // --- I5: contradiction groups are atomic ---

    #[test]
    fn a_contradiction_group_is_admitted_or_omitted_as_a_whole() {
        let a = EvidenceCandidate::from_association(
            "ctx",
            sourced_association("cats", "is_a", "mammals", 1.0, "s1", 1),
            1,
        );
        let b = EvidenceCandidate::from_association(
            "ctx",
            sourced_association("cats", "is_a", "reptiles", 1.0, "s2", 1),
            2,
        );
        let (fused, dropped) = super::super::fuse(vec![a, b]);
        let mut lookup = empty_lookup();
        lookup.insert(("s1".to_string(), 1), citation_for("s1", None));
        lookup.insert(("s2".to_string(), 1), citation_for("s2", None));

        // Plenty of room: both members admitted together.
        let admitted = select(fused, dropped, &generous_limits(), &lookup);
        assert_eq!(admitted.items.len(), 2);
        assert_eq!(admitted.items[0].contradicts.len(), 1);
        assert_eq!(admitted.items[1].contradicts.len(), 1);

        // Rebuild and re-run with a budget that fits exactly one
        // member's bytes but not both — the whole group must still be
        // omitted together, never half-admitted.
        let a2 = EvidenceCandidate::from_association(
            "ctx",
            sourced_association("cats", "is_a", "mammals", 1.0, "s1", 1),
            1,
        );
        let b2 = EvidenceCandidate::from_association(
            "ctx",
            sourced_association("cats", "is_a", "reptiles", 1.0, "s2", 1),
            2,
        );
        let (fused2, dropped2) = super::super::fuse(vec![a2, b2]);
        let one_item_budget = limits(1, 10_000_000, 10_000_000);
        let tight = select(fused2, dropped2, &one_item_budget, &lookup);
        assert!(tight.items.is_empty());
        assert_eq!(tight.omitted_total, 2);
        assert!(
            tight
                .omitted
                .iter()
                .all(|o| o.reason == REASON_CONTRADICTION_GROUP_EXCEEDS_BUDGET)
        );
    }

    #[test]
    fn three_way_contradiction_is_one_group() {
        let a = EvidenceCandidate::from_association(
            "ctx",
            association_out("cats", "is_a", "mammals", 1.0),
            1,
        );
        let b = EvidenceCandidate::from_association(
            "ctx",
            association_out("cats", "is_a", "reptiles", 1.0),
            2,
        );
        let c = EvidenceCandidate::from_association(
            "ctx",
            association_out("cats", "is_a", "fish", 1.0),
            3,
        );
        let result = select_all(vec![a, b, c], &empty_lookup());
        assert_eq!(result.items.len(), 3);
        for item in &result.items {
            assert_eq!(item.contradicts.len(), 2);
        }
        assert_eq!(result.selection.contradiction_groups, 1);
    }

    // --- Activation and negative-weight candidates preserved ---

    #[test]
    fn activation_uses_the_inner_association_and_keeps_kind_association() {
        let mut inner = association_out("cats", "is_a", "mammals", 1.0);
        inner.attributions = vec![AttributionOut {
            source: "book.txt".to_string(),
            weight: 1.0,
            count: 1,
            paragraph: Some(1),
            section: None,
            locator: None,
        }];
        let activation = ActivationOut {
            strength: 0.5,
            path: vec!["cats".to_string(), "mammals".to_string()],
            association: inner,
        };
        let pool = vec![EvidenceCandidate::from_activation("ctx", activation, 1)];
        let result = select_all(pool, &empty_lookup());
        assert_eq!(result.items.len(), 1);
        assert_eq!(result.items[0].kind, KIND_ASSOCIATION);
        assert!(result.items[0].association.is_some());
    }

    // --- Near-duplicate suppression ---

    #[test]
    fn near_identical_passages_keep_the_higher_ranked_one() {
        let survivor = EvidenceCandidate::from_passage(
            "ctx",
            passage_hit("s1", 0, "the quick brown fox jumps"),
            1,
        );
        let dup = EvidenceCandidate::from_passage(
            "ctx",
            passage_hit("s2", 0, "the quick brown fox jump"),
            2,
        );
        let result = select_all(vec![survivor, dup], &empty_lookup());
        assert_eq!(result.items.len(), 1);
        assert!(result.items[0].candidate_id.contains("s1"));
        assert_eq!(result.omitted_total, 1);
        assert_eq!(result.omitted[0].reason, REASON_DUPLICATE_PASSAGE);
        assert!(
            result.omitted[0]
                .duplicate_of
                .as_deref()
                .unwrap()
                .contains("s1")
        );
    }

    #[test]
    fn distinct_passages_are_not_suppressed() {
        let a = EvidenceCandidate::from_passage(
            "ctx",
            passage_hit("s1", 0, "cats are independent animals"),
            1,
        );
        let b = EvidenceCandidate::from_passage(
            "ctx",
            passage_hit("s2", 0, "quarterly revenue exceeded forecasts"),
            2,
        );
        let result = select_all(vec![a, b], &empty_lookup());
        assert_eq!(result.items.len(), 2);
        assert_eq!(result.omitted_total, 0);
    }

    // --- Diversity ---

    #[test]
    fn diversity_tiering_prefers_a_novel_source_within_a_tier() {
        // tier_width for max_items=4 is 1 — every unit is its own
        // tier, so diversity cannot move anything across tiers; use
        // max_items=12 so tier_width=3 puts all three candidates in
        // one tier, where the same-source repeat can be pushed back.
        let dominant_1 = EvidenceCandidate::from_passage(
            "ctx",
            passage_hit("dominant", 0, "alpha content one"),
            1,
        );
        let dominant_2 = EvidenceCandidate::from_passage(
            "ctx",
            passage_hit("dominant", 1, "beta content two"),
            2,
        );
        let other = EvidenceCandidate::from_passage(
            "ctx",
            passage_hit("other", 0, "gamma content three"),
            3,
        );
        let (fused, dropped) = super::super::fuse(vec![dominant_1, dominant_2, other]);
        let result = select(
            fused,
            dropped,
            &limits(12, 10_000_000, 10_000_000),
            &empty_lookup(),
        );
        assert_eq!(result.items.len(), 3);
        // The second "dominant" item must be pushed behind "other"
        // within its tier: order should be dominant#0, other, dominant#1.
        let sources: Vec<&str> = result
            .items
            .iter()
            .map(|item| item.passage.as_ref().unwrap().source.as_str())
            .collect();
        assert_eq!(sources, vec!["dominant", "other", "dominant"]);
    }

    /// ADR 0006 §12: `select_with_reorder` re-checks a reorder hook's
    /// permutation itself (`is_valid_permutation`, doc comment above it
    /// says this duplicates rerank.rs's own check "so this function
    /// never trusts a permutation's validity to whichever caller
    /// happened to construct it"). rerank.rs's tests exercise its own
    /// copy of this check with all three invalid shapes, but nothing
    /// exercised select.rs's independent copy of the same defense — a
    /// proptest reorder hook here only ever returns a genuine
    /// permutation. Each invalid shape below must be silently ignored,
    /// falling back to the un-reordered baseline.
    #[test]
    fn an_invalid_reorder_permutation_falls_back_to_the_unreordered_order() {
        fn three_candidates() -> Vec<EvidenceCandidate> {
            vec![
                EvidenceCandidate::from_passage("ctx", passage_hit("a", 0, "alpha"), 1),
                EvidenceCandidate::from_passage("ctx", passage_hit("b", 0, "beta"), 2),
                EvidenceCandidate::from_passage("ctx", passage_hit("c", 0, "gamma"), 3),
            ]
        }
        fn candidate_ids(result: &SelectedEvidence) -> Vec<String> {
            result
                .items
                .iter()
                .map(|item| item.candidate_id.clone())
                .collect()
        }

        let baseline = select_all(three_candidates(), &empty_lookup());
        assert_eq!(baseline.items.len(), 3);

        let cases: Vec<(&str, Vec<usize>)> = vec![
            ("wrong length", vec![0, 1]),
            ("out of range index", vec![0, 1, 99]),
            ("repeated index", vec![0, 0, 1]),
        ];
        for (name, order) in cases {
            let (fused, dropped) = super::super::fuse(three_candidates());
            let result = select_with_reorder(
                fused,
                dropped,
                &generous_limits(),
                &empty_lookup(),
                Some(&mut |_survivors: &[FusedCandidate]| Some(order.clone())),
            );
            assert_eq!(
                candidate_ids(&result),
                candidate_ids(&baseline),
                "invalid permutation ({name}) must fall back to the unreordered order"
            );
        }
    }

    /// ADR 0006 §9: a source counts as "seen" for diversity purposes
    /// once it has appeared "anywhere in the package-so-far" — across
    /// every prior tier, not just push-backs within the current one.
    /// `diversity_tiering_prefers_a_novel_source_within_a_tier` only
    /// ever tests one tier; this pins the cross-tier case, where
    /// "alpha" was already admitted in tier 0 and must still be pushed
    /// behind a fresh source when it reappears in tier 1.
    #[test]
    fn diversity_tiering_remembers_a_source_seen_in_an_earlier_tier() {
        // tier_width for max_items=8 is 2: tier 0 = [alpha#0, beta],
        // tier 1 = [alpha#1, gamma]. Within tier 0 both sources are
        // novel, so order is unchanged. In tier 1, "alpha" was already
        // admitted in tier 0 — it must be pushed behind "gamma", the
        // tier's only genuinely novel source.
        let alpha_0 =
            EvidenceCandidate::from_passage("ctx", passage_hit("alpha", 0, "alpha content"), 1);
        let beta =
            EvidenceCandidate::from_passage("ctx", passage_hit("beta", 0, "beta content"), 2);
        let alpha_1 =
            EvidenceCandidate::from_passage("ctx", passage_hit("alpha", 1, "alpha content two"), 3);
        let gamma =
            EvidenceCandidate::from_passage("ctx", passage_hit("gamma", 0, "gamma content"), 4);
        let (fused, dropped) = super::super::fuse(vec![alpha_0, beta, alpha_1, gamma]);
        let result = select(
            fused,
            dropped,
            &limits(8, 10_000_000, 10_000_000),
            &empty_lookup(),
        );
        assert_eq!(result.items.len(), 4);
        let sources: Vec<&str> = result
            .items
            .iter()
            .map(|item| item.passage.as_ref().unwrap().source.as_str())
            .collect();
        assert_eq!(sources, vec!["alpha", "beta", "gamma", "alpha"]);
    }

    #[test]
    fn unsourced_candidates_never_count_as_repeats_of_each_other() {
        let a = EvidenceCandidate::from_association("ctx", association_out("a", "l1", "x", 1.0), 1);
        let b = EvidenceCandidate::from_association("ctx", association_out("b", "l2", "y", 1.0), 2);
        let result = select_all(vec![a, b], &empty_lookup());
        assert_eq!(result.items.len(), 2);
    }

    // --- I6: every omission is counted ---

    #[test]
    fn every_non_admitted_candidate_is_counted_in_omitted_total() {
        let candidates: Vec<EvidenceCandidate> = (0..25)
            .map(|i| {
                EvidenceCandidate::from_passage(
                    "ctx",
                    passage_hit(
                        &format!("s{i}"),
                        0,
                        &format!("distinct text body number {i}"),
                    ),
                    i + 1,
                )
            })
            .collect();
        let (fused, dropped) = super::super::fuse(candidates);
        let total = fused.len();
        let result = select(
            fused,
            dropped,
            &limits(3, 10_000_000, 10_000_000),
            &empty_lookup(),
        );
        assert_eq!(result.items.len() + result.omitted_total, total);
        // Listing itself is capped at MAX_LISTED_ISSUES even though
        // omitted_total is exact.
        assert_eq!(
            result.omitted.len(),
            MAX_LISTED_ISSUES.min(result.omitted_total)
        );
    }

    // --- Community candidates ---

    #[test]
    fn community_candidates_self_cite_and_contribute_no_citation_entries() {
        let pool = vec![EvidenceCandidate::from_community(
            "ctx",
            community_hit("L0-1", 0, "summary"),
            1,
        )];
        let result = select_all(pool, &empty_lookup());
        assert_eq!(result.items.len(), 1);
        assert!(result.items[0].citation_refs.is_empty());
        assert!(result.citations.is_empty());
    }

    // --- Serde round-trip ---

    #[test]
    fn an_item_with_no_contradictions_round_trips_through_serde() {
        // `contradicts` is skipped on the wire when empty
        // (`skip_serializing_if = "Vec::is_empty"`), so the JSON a real
        // admitted item without a contradiction produces omits the key
        // entirely — deserializing that JSON back must not fail with a
        // "missing field" error.
        let pool = vec![EvidenceCandidate::from_passage(
            "ctx",
            passage_hit("s", 0, "no contradictions here"),
            1,
        )];
        let result = select_all(pool, &empty_lookup());
        let json = serde_json::to_string(&result.items[0]).expect("serialize");
        assert!(
            !json.contains("contradicts"),
            "empty contradicts must be omitted from the wire, not asserting a bug in this test: {json}"
        );
        let restored: EvidenceItem = serde_json::from_str(&json).expect("deserialize");
        assert!(restored.contradicts.is_empty());
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn proptest_config() -> ProptestConfig {
            let cases = std::env::var("PROPTEST_CASES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64);
            ProptestConfig {
                cases,
                ..ProptestConfig::default()
            }
        }

        /// A recipe, not a built [`EvidenceCandidate`] — the candidate
        /// itself has no `Clone` (its payload embeds wire types that
        /// don't derive one either), so "the same pool in a different
        /// order" (for the determinism property below) is built by
        /// calling the real constructors twice from the same owned
        /// recipe data rather than cloning built candidates.
        #[derive(Clone, Debug)]
        enum CandidateRecipe {
            Association {
                subject: &'static str,
                label: &'static str,
                object: &'static str,
                weight: f64,
                source: Option<(&'static str, u32)>,
            },
            Passage {
                source: &'static str,
                paragraph: u32,
                text: &'static str,
            },
            Community {
                id: &'static str,
                paragraph: u32,
                text: &'static str,
            },
        }

        /// A small, mixed ASCII/Japanese vocabulary — deliberately
        /// reused across subject/label/object/source/text so generated
        /// pools frequently collide on `(subject, label)` (exercising
        /// contradiction grouping) and on near-identical text
        /// (exercising near-duplicate suppression), not just on
        /// disjoint, never-interacting candidates.
        fn word() -> impl Strategy<Value = &'static str> {
            prop_oneof![
                Just("alpha"),
                Just("beta"),
                Just("gamma"),
                Just("猫"),
                Just("犬"),
            ]
        }

        fn recipe_strategy() -> impl Strategy<Value = CandidateRecipe> {
            prop_oneof![
                (
                    word(),
                    word(),
                    word(),
                    -5.0f64..5.0,
                    prop::option::of((word(), 0u32..3))
                )
                    .prop_map(|(subject, label, object, weight, source)| {
                        CandidateRecipe::Association {
                            subject,
                            label,
                            object,
                            weight,
                            source,
                        }
                    }),
                (word(), 0u32..3, word()).prop_map(|(source, paragraph, text)| {
                    CandidateRecipe::Passage {
                        source,
                        paragraph,
                        text,
                    }
                }),
                (word(), 0u32..3, word()).prop_map(|(id, paragraph, text)| {
                    CandidateRecipe::Community {
                        id,
                        paragraph,
                        text,
                    }
                }),
            ]
        }

        fn pool_strategy() -> impl Strategy<Value = Vec<CandidateRecipe>> {
            prop::collection::vec(recipe_strategy(), 0..12)
        }

        fn build_one(recipe: &CandidateRecipe, lane_rank: usize) -> EvidenceCandidate {
            match recipe {
                CandidateRecipe::Association {
                    subject,
                    label,
                    object,
                    weight,
                    source,
                } => {
                    let mut association = association_out(subject, label, object, *weight);
                    if let Some((src, paragraph)) = source {
                        association.attributions = vec![AttributionOut {
                            source: src.to_string(),
                            weight: *weight,
                            count: 1,
                            paragraph: Some(*paragraph),
                            section: None,
                            locator: None,
                        }];
                    }
                    EvidenceCandidate::from_association("ctx", association, lane_rank)
                }
                CandidateRecipe::Passage {
                    source,
                    paragraph,
                    text,
                } => EvidenceCandidate::from_passage(
                    "ctx",
                    passage_hit(source, *paragraph, text),
                    lane_rank,
                ),
                CandidateRecipe::Community {
                    id,
                    paragraph,
                    text,
                } => EvidenceCandidate::from_community(
                    "ctx",
                    community_hit(id, *paragraph, text),
                    lane_rank,
                ),
            }
        }

        /// Builds the pool in `order`, but each candidate still gets
        /// the lane rank tied to its *original* position in `recipes`
        /// (`i + 1`, never its position in `order`), so this only
        /// changes the arrival order `fuse` sees, never any candidate's
        /// own rank. Reassigning rank-by-arrival-position instead would
        /// not be a fair reordering test at all: two same-triple
        /// candidates with different weights are *supposed* to fuse to
        /// a different representative when whichever one holds rank 1
        /// changes — that is RRF working correctly, not an I3
        /// violation.
        fn build_pool_in_order(
            recipes: &[CandidateRecipe],
            order: &[usize],
        ) -> Vec<EvidenceCandidate> {
            order
                .iter()
                .map(|&i| build_one(&recipes[i], i + 1))
                .collect()
        }

        /// [`build_pool_in_order`] with the identity order — one
        /// candidate per recipe, each getting the 1-based lane rank
        /// matching its own position in `recipes`.
        fn build_pool(recipes: &[CandidateRecipe]) -> Vec<EvidenceCandidate> {
            build_pool_in_order(recipes, &(0..recipes.len()).collect::<Vec<_>>())
        }

        fn citation_lookup_for(recipes: &[CandidateRecipe]) -> HashMap<(String, u32), Citation> {
            let mut lookup = HashMap::new();
            for recipe in recipes {
                if let CandidateRecipe::Association {
                    source: Some((source, paragraph)),
                    ..
                } = recipe
                {
                    lookup.insert((source.to_string(), *paragraph), citation_for(source, None));
                }
            }
            lookup
        }

        proptest! {
            #![proptest_config(proptest_config())]

            /// I2: the three budgets are hard ceilings for *any*
            /// candidate pool, including tiny/zero budgets — checked by
            /// independently recomputing the byte/token totals from the
            /// final `items`/`citations` output through the same public
            /// budget helpers `select` itself uses internally, both
            /// against the ceiling and against `select`'s own reported
            /// `BudgetUsage` (catching any internal running-sum drift
            /// from what the output actually serializes to). An empty
            /// `items`/`citations` pair still costs 4 bytes on the wire
            /// (`"[]"` twice) — a structural floor no admission
            /// decision can shrink further, so a `max_bytes`/
            /// `max_tokens` set below that floor is the one case the
            /// ceiling check exempts, matching ADR 0006 §8's own
            /// framing of a tiny budget as "ordinary, valid input," not
            /// a promise the reported usage always fits under it.
            ///
            /// I6: every candidate not present in `items` is counted
            /// in `omitted_total`, exactly — the two must always sum to
            /// the fused pool's own size.
            #[test]
            fn budgets_are_hard_ceilings_and_every_omission_is_counted(
                recipes in pool_strategy(),
                max_items in 0usize..8,
                max_bytes in 0usize..2000,
                max_tokens in 0usize..400,
            ) {
                let lookup = citation_lookup_for(&recipes);
                let (fused, dropped) = super::super::super::fuse(build_pool(&recipes));
                let fused_count = fused.len();
                let budget_limits = limits(max_items, max_bytes, max_tokens);
                let result = select(fused, dropped, &budget_limits, &lookup);

                prop_assert!(result.items.len() <= max_items);
                let items_len = result.items.len();
                let SelectedEvidence { items, citations, budget, omitted_total, .. } = result;

                let (bytes_used, tokens_used) = recomputed_budget_usage(items, &citations);
                prop_assert!(bytes_used <= max_bytes || items_len == 0);
                prop_assert_eq!(bytes_used, budget.bytes_used);
                prop_assert!(tokens_used <= max_tokens || items_len == 0);
                prop_assert_eq!(tokens_used, budget.tokens_used);

                prop_assert_eq!(items_len + omitted_total, fused_count);
            }

            /// I3: the same logical pool — same candidates, same ranks
            /// — fed to `fuse` in reverse arrival order selects
            /// byte-identical JSON. Each candidate keeps the rank tied
            /// to its own original position ([`build_pool_in_order`]);
            /// only the order `fuse`'s internal grouping visits them in
            /// changes.
            #[test]
            fn selection_is_deterministic_regardless_of_pool_order(recipes in pool_strategy()) {
                let lookup = citation_lookup_for(&recipes);
                let forward_order: Vec<usize> = (0..recipes.len()).collect();
                let reverse_order: Vec<usize> = forward_order.iter().rev().copied().collect();

                let (fused_a, dropped_a) = super::super::super::fuse(build_pool_in_order(&recipes, &forward_order));
                let (fused_b, dropped_b) = super::super::super::fuse(build_pool_in_order(&recipes, &reverse_order));
                let generous = limits(20, 100_000, 20_000);
                let result_a = select(fused_a, dropped_a, &generous, &lookup);
                let result_b = select(fused_b, dropped_b, &generous, &lookup);

                prop_assert_eq!(
                    serde_json::to_string(&result_a).unwrap(),
                    serde_json::to_string(&result_b).unwrap()
                );
            }

            /// #307, ADR 0006 §12: an arbitrary reorder hook — standing
            /// in for a reranker's permutation — can change WHICH
            /// candidates get admitted (a promoted candidate can win a
            /// tie-break, an omission budget can land differently), but
            /// can never break the invariants that hold with no
            /// reorder at all: every budget ceiling, the items/omitted
            /// count reconciling to the fused pool's size, and every
            /// admitted item's `bytes`/`estimated_tokens` still
            /// matching its own independently recomputed content
            /// metrics. Reordering the input to an invariant-preserving
            /// admission process cannot itself violate an invariant.
            #[test]
            fn reordering_the_pool_before_admission_preserves_every_selection_invariant(
                recipes in pool_strategy(),
                seed: u64,
                max_items in 0usize..8,
                max_bytes in 0usize..2000,
                max_tokens in 0usize..400,
            ) {
                let lookup = citation_lookup_for(&recipes);
                let (fused, dropped) = super::super::super::fuse(build_pool(&recipes));
                let fused_count = fused.len();
                let budget_limits = limits(max_items, max_bytes, max_tokens);

                // A self-contained splitmix64-seeded Fisher-Yates shuffle
                // — no `rand` dev-dependency exists in this tree, and the
                // permutation only needs to be arbitrary, not
                // cryptographically random. Sized against `survivors`
                // (post near-duplicate-suppression) at call time, inside
                // the closure, rather than pre-built against
                // `fused_count` — suppression can drop specific
                // candidates rather than only truncate a tail, so a
                // pre-built permutation would not line up.
                let mut state = seed;
                let mut next_u64 = move || {
                    state = state.wrapping_add(0x9E3779B97F4A7C15);
                    let mut z = state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                    z ^ (z >> 31)
                };

                let result = select_with_reorder(
                    fused,
                    dropped,
                    &budget_limits,
                    &lookup,
                    Some(&mut |survivors: &[FusedCandidate]| {
                        let len = survivors.len();
                        let mut order: Vec<usize> = (0..len).collect();
                        for i in (1..order.len()).rev() {
                            let j = (next_u64() as usize) % (i + 1);
                            order.swap(i, j);
                        }
                        Some(order)
                    }),
                );

                prop_assert!(result.items.len() <= max_items);
                let items_len = result.items.len();
                let SelectedEvidence { items, citations, budget, omitted_total, .. } = result;

                let (bytes_used, tokens_used) = recomputed_budget_usage(items, &citations);
                prop_assert!(bytes_used <= max_bytes || items_len == 0);
                prop_assert_eq!(bytes_used, budget.bytes_used);
                prop_assert!(tokens_used <= max_tokens || items_len == 0);
                prop_assert_eq!(tokens_used, budget.tokens_used);

                prop_assert_eq!(items_len + omitted_total, fused_count);
            }
        }
    }
}
