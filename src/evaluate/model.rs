//! The evaluation.json output data model: `EvaluationFile` and every
//! block nested under it, plus the handful of pure, non-network
//! projection functions (`evidence_locators`, `hits_from_evidence_items`)
//! that shape an assembled `EvidenceItem` into the same locator shapes
//! the baseline passage lane produces. The lane runners in
//! `super::lanes` construct these; `super::run_evaluate`,
//! `super::build_metrics`, and `super::print_summary` read them back.

use super::*;

// ============================= Value shapes =============================

// `Debug`/`Clone` are dropped from this and every struct that
// transitively embeds `SearchContextPlan`/`PassageLanes` (the real
// server response types, reused verbatim per #282): neither derives
// them, matching this codebase's convention of keeping wire-response
// DTOs lean. Nothing here is ever cloned; `Serialize` is all a
// write-once artifact needs.
#[derive(Serialize)]
pub(crate) struct EvaluationFile {
    pub(crate) taguru_evaluation: u64,
    pub(crate) generated_at: String,
    pub(crate) matching: MatchingBlock,
    pub(crate) inputs: InputsBlock,
    pub(crate) corpus: CorpusBlock,
    /// `None` when `--thresholds` was not given (a report-only run);
    /// `Some` — the loaded bounds checked against this run — otherwise
    /// (ADR 0004 §9.1's threshold identity, extended by #276 with the
    /// pass/fail verdict; see [`thresholds::ThresholdReport`]).
    pub(crate) thresholds: Option<ThresholdReport>,
    pub(crate) definitions: BTreeMap<String, MetricDef>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) warnings: Vec<String>,
    pub(crate) cases: Vec<CaseBlock>,
    pub(crate) metrics: MetricsMap,
}

/// Records the identity-matching choice this run made, following ADR
/// 0003 §9.4's precedent for recording such choices in an artifact
/// header. ADR 0004 §8: `evaluate` matches with
/// `taguru::context::normalize_entry` — the same folding the passage
/// index itself uses — never `benchmark::identity::normalize_term`,
/// whose deliberate katakana exception exists for a cross-model
/// comparison this verb does not do.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct MatchingBlock {
    pub(crate) normalization: &'static str,
    /// What `normalization` is applied to, both sides, before an exact
    /// match: `expected_concepts`/`expected_labels` against a cue's
    /// `resolved_names[]`. `expected_associations` is deliberately
    /// absent — its coverage never does a client-side string
    /// comparison at all; `/resolve`/`/resolve_label` pin each
    /// position server-side (ADR 0004 §7 step 2), and coverage is then
    /// just whether the pinned `/query` call returned `total >= 1`
    /// (see [`association_coverage`]).
    pub(crate) normalized: &'static [&'static str],
    /// `expected_sources` match by exact `(source, paragraph)` instead
    /// — the source preflight already requires an exact manifest path
    /// match, and a paragraph index is not text to fold.
    pub(crate) sources: &'static str,
}

impl Default for MatchingBlock {
    fn default() -> Self {
        Self {
            normalization: "taguru::context::normalize_entry",
            normalized: &["expected_concepts", "expected_labels"],
            sources: "exact (source, paragraph) match — no normalization",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct InputsBlock {
    pub(crate) eval: EvalInputsBlock,
    pub(crate) context: String,
    /// scheme + host + port only (ADR 0004 §11) — never the literal
    /// `--url` value.
    pub(crate) url: String,
    pub(crate) out: String,
    pub(crate) default_limit: usize,
    pub(crate) resolve_limit: usize,
    /// #308 (ADR 0006 §14): `"baseline"` (unchanged passage lane) or
    /// `"assembly"` (`--assembly`, ADR 0006's evidence-assembly
    /// passage lane) — an open string, not a closed Rust enum, per
    /// this codebase's convention for wire-visible mode tags.
    pub(crate) mode: String,
    /// The equal-budget ceilings this run enforced. Always present in
    /// `assembly` mode — `POST /contexts/{name}/evidence` has no
    /// unbudgeted mode, so this records the server's own defaults
    /// (`max_items: 40, max_bytes: 65536, max_tokens: 4000`) even when
    /// no `--max-*` flag was given. `None` only in `baseline` mode,
    /// when no `--max-items`/`--max-bytes`/`--max-tokens` flag was
    /// given — that mode alone has a genuinely unbudgeted state
    /// (unchanged from before this flag existed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) budget: Option<BudgetLimits>,
    /// The `--rerank MODEL` value, when given — `None` in `baseline`
    /// mode, and `None` in `assembly` mode when `--rerank` was not
    /// given (a fully deterministic assembly run, ADR 0006 §14
    /// configuration 2). Whether the requested model actually reranked
    /// is per-case, in `CaseBlock.evidence`'s `reranker` block — a
    /// server with no `TAGURU_RERANK_URL`/`_MODEL` configured, or a
    /// model mismatch, degrades every case the same way.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) rerank: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct EvalInputsBlock {
    pub(crate) path: String,
    pub(crate) name: Option<String>,
    pub(crate) cases: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CorpusBlock {
    pub(crate) revision_before: ContextRevision,
    pub(crate) revision_after: ContextRevision,
    /// Equality across all three `ContextRevision` lanes, never
    /// ordering (ADR 0004 §12) — a write landing mid-run flips this to
    /// `false` without aborting the run itself.
    pub(crate) stable: bool,
    pub(crate) last_write_epoch_before: u64,
    pub(crate) last_write_epoch_after: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) embeddings: Option<EmbeddingsBlock>,
    pub(crate) sources_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct EmbeddingsBlock {
    pub(crate) provider_model: Option<String>,
}

#[derive(Serialize)]
pub(crate) struct CaseBlock {
    pub(crate) case_id: String,
    pub(crate) query: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) cues: Vec<String>,
    pub(crate) limit: usize,
    pub(crate) passage: PassageOutcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) structural: Option<StructuralBlock>,
    /// `Some` only when the case declares at least one `relevance >= 1`
    /// `expected_sources` entry and the passage lane completed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) recall: Option<RecallBlock>,
    /// `Some` only when the case declares at least one of
    /// `expected_concepts`/`expected_labels`/`expected_associations`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) coverage: Option<CoverageBlock>,
    /// The ADR 0004 §7 lane cross-tab input for this one case — `Some`
    /// only when the case declares both a structural and a source
    /// expectation and the passage lane completed; aggregated into
    /// `metrics`' `lanes.*` ratios (§9.1).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) lane_cross: Option<LaneCrossBlock>,
    /// ADR 0004 §8's two citation measurements — `Some` only when the
    /// case declares at least one `expected_citations` entry. Runs
    /// independent of every other lane, including a passage lane that
    /// missed outright.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) citations: Option<CitationsBlock>,
    /// Unmet expectations (ADR 0004 §11): sources, then concepts, then
    /// labels, then associations, then citations, capped at 3 entries.
    /// Silent (empty) when the passage lane failed outright — see
    /// [`build_missed`].
    pub(crate) missed: Vec<String>,
    pub(crate) missed_truncated: usize,
    /// #308 (ADR 0006 §14): the assembly lane's own diagnostic detail
    /// — selection trace, reranker outcome, omission breakdown. `Some`
    /// only in `assembly` mode; `passage` above (built from the same
    /// admitted package, see [`hits_from_evidence_items`]) is what recall/
    /// citation/lane-cross scoring actually reads, so this block never
    /// duplicates scoring logic — only what `passage`'s `HitLocator`
    /// projection cannot carry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) evidence: Option<EvidenceOutcome>,
    /// #308's equal-budget accounting — `Some` in both modes when a
    /// budget flag was given, `None` when none was (both modes then
    /// run untruncated, matching every run before this flag existed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) budget: Option<BudgetAccounting>,
    /// #308's `diversity.sources` metric input (ADR 0006 §14's own new
    /// metric): the count of distinct `source` locators among this
    /// case's `passage.hits` — `baseline`'s (possibly truncated)
    /// passage hits, or `assembly`'s admitted items' own locators
    /// union their `citation_refs` (the identical set
    /// [`hits_from_evidence_items`] builds for recall/citation
    /// scoring, so this reads the same evidence those metrics do).
    /// `None` when the passage/evidence lane failed outright, matching
    /// `recall`/`coverage`'s own not-applicable convention.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) diversity_sources: Option<usize>,
}

/// Recall@k/MRR/nDCG against `expected_sources`' graded relevance (ADR
/// 0004 §274) — see [`score_recall`].
#[derive(Debug, Clone, Serialize)]
pub(crate) struct RecallBlock {
    pub(crate) recall_at_k: f64,
    pub(crate) mrr: f64,
    pub(crate) ndcg: f64,
    pub(crate) expected_total: usize,
    pub(crate) matched: usize,
}

/// One expectation category's coverage count — see [`coverage_counts`]/
/// [`association_coverage`].
#[derive(Debug, Clone, Serialize)]
pub(crate) struct CoverageCounts {
    pub(crate) expected: usize,
    pub(crate) matched: usize,
    pub(crate) value: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CoverageBlock {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) concepts: Option<CoverageCounts>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) labels: Option<CoverageCounts>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) associations: Option<CoverageCounts>,
}

/// One case's contribution to the ADR 0004 §7 lane cross-tab —
/// aggregated across every case that has one into `metrics`'
/// `lanes.structural_hit`/`lanes.passage_hit`/`lanes.both`/
/// `lanes.neither`.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct LaneCrossBlock {
    pub(crate) structural_hit: bool,
    pub(crate) passage_hit: bool,
}

/// ADR 0004 §8's two, deliberately un-merged citation measurements for
/// one case, plus the per-entry checks both are computed from.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct CitationsBlock {
    /// Citation recall: the fraction of `checks` whose `(source,
    /// paragraph)` appeared among this case's served results (passage
    /// hits, plus the structural lane's `AttributionOut` locators when
    /// it ran) — see [`served_locators`]. Independent of `validity`.
    pub(crate) recall: CitationRecallBlock,
    /// Locator validity: the fraction of `checks` whose
    /// `POST /contexts/{name}/citations` call resolved with a matching
    /// `section` (when declared) and `quote` (when declared) — see
    /// [`citation_is_valid`]. Computed even for a case whose passage
    /// lane missed outright.
    pub(crate) validity: CitationValidityBlock,
    /// One entry per `expected_citations[]`, in request order (the
    /// endpoint does not echo `paragraph` back, so requests and
    /// responses correlate positionally).
    pub(crate) checks: Vec<CitationCheck>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CitationRecallBlock {
    pub(crate) expected_total: usize,
    pub(crate) matched: usize,
    pub(crate) value: f64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CitationValidityBlock {
    pub(crate) expected_total: usize,
    pub(crate) valid: usize,
    pub(crate) value: f64,
}

/// One `expected_citations[]` entry's outcome from both measurements:
/// `served` is citation recall's own per-entry bit, `outcome` is
/// locator validity's `POST /contexts/{name}/citations` result.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct CitationCheck {
    pub(crate) source: String,
    pub(crate) paragraph: u32,
    pub(crate) served: bool,
    #[serde(flatten)]
    pub(crate) outcome: CitationOutcome,
    pub(crate) latency_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub(crate) enum CitationOutcome {
    Resolved {
        section: SectionCheck,
        #[serde(skip_serializing_if = "Option::is_none")]
        quote: Option<QuoteCheck>,
    },
    /// `code` is the server's stable `ErrorCode` string (`no_source`/
    /// `no_paragraph`/etc., `src/api.rs`) when the failure carried one
    /// — `None` for a transport failure or an unparseable response.
    /// `message` is truncated to [`MAX_ERROR_BYTES`] (ADR 0004 §11).
    Unresolved {
        code: Option<String>,
        message: String,
    },
}

/// [`ExpectedCitation::section`]'s three-valued check (ADR 0004 §8):
/// absent stays [`SectionCheck::NotChecked`]; present — including an
/// explicit `null` — is checked against the server's `Citation.section`
/// and comes back `Matched`/`Mismatched`. The server's own `section`
/// value is never recorded on a mismatch (ADR 0004 §11's no-corpus-text
/// posture) — only the user's own declared `expected` rides along, so a
/// reader diagnoses the miss by re-running the printed reproduction
/// command, never by reading served text out of the artifact.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "check", rename_all = "snake_case")]
pub(crate) enum SectionCheck {
    NotChecked,
    Matched { expected: Option<String> },
    Mismatched { expected: Option<String> },
}

/// [`ExpectedCitation::quote`]'s check result (ADR 0004 §8). ADR 0004
/// §11's one documented exception to "no corpus body text": even on a
/// mismatch this records only the user's own declared `quote` and a
/// boolean, never the served paragraph body.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct QuoteCheck {
    pub(crate) declared: String,
    pub(crate) matched: bool,
}

#[derive(Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub(crate) enum PassageOutcome {
    Searched {
        #[serde(skip_serializing_if = "Option::is_none")]
        plan: Option<SearchContextPlan>,
        hits: Vec<HitLocator>,
        latency_ms: u64,
    },
    Failed {
        message: String,
        latency_ms: u64,
    },
}

/// A passage hit's locator, stripped of `text` — no corpus body text
/// is written into `evaluation.json` (ADR 0004 §11).
#[derive(Serialize)]
pub(crate) struct HitLocator {
    pub(crate) source: String,
    pub(crate) paragraph: u32,
    pub(crate) score: f32,
    pub(crate) lanes: PassageLanes,
}

impl From<PassageHit> for HitLocator {
    fn from(hit: PassageHit) -> Self {
        HitLocator {
            source: hit.source,
            paragraph: hit.paragraph,
            score: hit.score,
            lanes: hit.lanes,
        }
    }
}

/// #308: flattens an assembled package's admitted `items[]` into the
/// same `HitLocator` shape the baseline passage lane produces, in
/// `fused_rank` order — one `HitLocator` per `citation_refs` entry, so
/// an item with several independent attributions (ADR 0006 §9's
/// corroboration) contributes one locator per source, and an item with
/// none (a zero-attribution association — `EvidenceCandidate`'s own
/// documented case) contributes none. This is what lets
/// `score_recall`/`served_locators`/`build_missed` — every scoring
/// function the baseline passage lane already feeds — run unchanged
/// against an assembled package: they only ever compare
/// `(source, paragraph)` pairs and rank order, never a lane's own
/// score, and RRF's `fused_score` is deliberately never serialized at
/// all (ADR 0006 §7) — `score: 0.0` below is a placeholder, not a
/// discarded real value.
/// #308: the diagnostic-locator projection of an assembled package's
/// admitted `items[]` — [`EvidenceLocator`], stripped of
/// `association`/`passage`/`community` (which can carry passage body
/// text) the same way [`hits_from_evidence_items`] strips it from the
/// scoring-facing `HitLocator` projection.
/// One [`EvidenceItem`]'s own `(source, paragraph)` locator — `None`
/// for an association item, whose locator instead lives entirely in
/// `citation_refs` ([`EvidenceCandidate::citation_refs`],
/// `src/api/evidence.rs`: an association's own payload carries no
/// single source/paragraph of its own, only a set of attributions).
/// `Some` for a passage item (`PassageHit.source`/`.paragraph`
/// directly) or a community item (`community:{id}`, matching the
/// artifact-source-id convention `CommunityHit`'s own doc comment
/// names, and its own `.paragraph`).
fn evidence_item_locator(item: &EvidenceItem) -> Option<(String, u32)> {
    if let Some(passage) = &item.passage {
        return Some((passage.source.clone(), passage.paragraph));
    }
    if let Some(community) = &item.community {
        return Some((
            format!("community:{}", community.community),
            community.paragraph,
        ));
    }
    None
}

pub(crate) fn evidence_locators(items: &[EvidenceItem]) -> Vec<EvidenceLocator> {
    items
        .iter()
        .map(|item| {
            let (source, paragraph) = match evidence_item_locator(item) {
                Some((source, paragraph)) => (Some(source), Some(paragraph)),
                None => (None, None),
            };
            EvidenceLocator {
                candidate_id: item.candidate_id.clone(),
                kind: item.kind.clone(),
                fused_rank: item.fused_rank,
                source,
                paragraph,
                citation_refs: item.citation_refs.clone(),
            }
        })
        .collect()
}

/// [`HitLocator`]s for scoring: each item's own locator
/// ([`evidence_item_locator`]) union its `citation_refs` (an
/// association's corroborating attributions) — a passage/community
/// item without this union would never enter recall/citation/
/// diversity scoring at all, since `citation_refs` is empty for both
/// kinds by design (their locator lives on the payload itself, not as
/// a separate attribution list). `limit` is the case's own configured
/// limit (`options.limit`/`default_limit`, the same value the baseline
/// passage lane's own `sources/search` call already bounds its `hits`
/// response to server-side): `items[]` can carry up to `budget.max_items`
/// admitted candidates (default 40) plus one `citation_refs` entry per
/// association's attribution, structurally unrelated to `limit` — left
/// unbounded, that would size an assembly-mode case's `hits[]` (and
/// therefore its `k` for `recall_at_k`/`mrr`/`ndcg`, and its
/// `lane_cross.passage_hit` classification, ADR 0004 §8) differently
/// from the identical-`limit` baseline case it exists to compare
/// against.
pub(crate) fn hits_from_evidence_items(items: &[EvidenceItem], limit: usize) -> Vec<HitLocator> {
    items
        .iter()
        .flat_map(|item| {
            let own = evidence_item_locator(item).into_iter();
            let attributed = item
                .citation_refs
                .iter()
                .map(|reference| (reference.source.clone(), reference.paragraph));
            own.chain(attributed)
        })
        .take(limit)
        .map(|(source, paragraph)| HitLocator {
            source,
            paragraph,
            score: 0.0,
            lanes: PassageLanes {
                bm25: None,
                vector: None,
            },
        })
        .collect()
}

/// #308's diagnostic locator for one admitted assembly item — stripped
/// of `association`/`passage`/`community` (which can carry passage
/// body text) the same way [`HitLocator`] strips `PassageHit.text`
/// (ADR 0004 §11's no-corpus-text rule). `source`/`paragraph` is the
/// item's own locator ([`evidence_item_locator`]) — absent for an
/// association item, whose locators live in `citation_refs` instead.
#[derive(Serialize)]
pub(crate) struct EvidenceLocator {
    pub(crate) candidate_id: String,
    pub(crate) kind: String,
    pub(crate) fused_rank: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) paragraph: Option<u32>,
    pub(crate) citation_refs: Vec<CitationRef>,
}

/// #308's per-case assembly diagnostics (ADR 0006 §14) — everything
/// `CaseBlock.passage`'s `HitLocator` projection (built by
/// [`hits_from_evidence_items`]) cannot carry. `selection`/`reranker`
/// are the *existing* `EvidencePackage` wire blocks, embedded verbatim
/// — no parallel mirror — matching how `items[]` itself embeds
/// `AssociationOut`/`PassageHit`/`CommunityHit` verbatim upstream.
#[derive(Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub(crate) enum EvidenceOutcome {
    Assembled {
        latency_ms: u64,
        items: Vec<EvidenceLocator>,
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        omitted_by_reason: BTreeMap<String, usize>,
        selection: SelectionPlan,
        reranker: RerankerPlan,
    },
    Failed {
        message: String,
        latency_ms: u64,
    },
}

/// #308's equal-budget accounting for one case, shared by both modes —
/// `crate::api::evidence::budget::BudgetUsage` (the same wire shape a
/// configured request already returns) flattened alongside how many
/// candidates this case's budget dropped, so the artifact reads as one
/// object rather than a nested `usage` key.
#[derive(Serialize)]
pub(crate) struct BudgetAccounting {
    #[serde(flatten)]
    pub(crate) usage: BudgetUsage,
    pub(crate) omitted_total: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct StructuralBlock {
    pub(crate) cues: Vec<CueResolution>,
    pub(crate) associations: Vec<AssociationProbe>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct CueResolution {
    pub(crate) cue: String,
    pub(crate) kind: &'static str,
    pub(crate) resolved_names: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tier: Option<String>,
    pub(crate) limit: usize,
    pub(crate) latency_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct AssociationProbe {
    pub(crate) subject_cue: String,
    pub(crate) label_cue: String,
    pub(crate) object_cue: String,
    pub(crate) subject: PositionOutcome,
    pub(crate) label: PositionOutcome,
    pub(crate) object: PositionOutcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) query: Option<QueryProbe>,
}

/// One `expected_associations[]` position's resolution (ADR 0004 §7
/// step 2's stricter multi-candidate policy): exactly one top-tier
/// candidate pins the position; zero or several do not, and `query` is
/// never called in either of those cases.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub(crate) enum PositionOutcome {
    Resolved {
        name: String,
        tier: String,
        latency_ms: u64,
    },
    NotFound {
        latency_ms: u64,
    },
    Ambiguous {
        tier: String,
        candidates: Vec<String>,
        latency_ms: u64,
    },
    Errored {
        message: String,
        latency_ms: u64,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub(crate) enum QueryProbe {
    Queried {
        total: usize,
        matches: usize,
        /// The structural lane's own served citation locators (ADR 0004
        /// §8) — every `attributions[]` entry off every matched
        /// association, `source` + `paragraph` only. No corpus body
        /// text (ADR 0004 §11): `AttributionOut` never carries any.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        attributions: Vec<AttributionLocator>,
        latency_ms: u64,
    },
    Errored {
        message: String,
        latency_ms: u64,
    },
}

/// One served `(source, paragraph)` locator read off an `AttributionOut`
/// (`src/api.rs:1453-1462`) — stripped of `weight`/`count`/`section`,
/// which citation recall does not need. `paragraph: None` (an
/// attribution with no paragraph locator at all) never satisfies an
/// `expected_citations` entry, which always names one.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct AttributionLocator {
    pub(crate) source: String,
    pub(crate) paragraph: Option<u32>,
}
