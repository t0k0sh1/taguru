//! The passage/structural/citation lane runners (ADR 0004 §7-8): each
//! makes the lane's own HTTP calls and interprets the response into
//! `super::model`'s output shapes, plus the pure rank/label scoring
//! functions (recall@k/MRR/nDCG, coverage, citation recall and locator
//! validity) those responses feed — never a lane's own raw score.
//! `build_case_block` is the per-case entry point `super::run_evaluate`
//! calls once per eval case; `score_case` (kept in the hub, since it is
//! `build_case_block`'s own orchestration) calls back into this
//! module's scoring functions.

use super::*;

// ============================== Passage lane ==============================

pub(crate) fn run_passage_lane(
    api: &Api,
    context: &str,
    case: &EvalCase,
    limit: usize,
) -> PassageOutcome {
    match fetch_passage_hits(api, context, case, limit) {
        Ok((hits, plan, latency_ms)) => PassageOutcome::Searched {
            plan,
            hits: hits.into_iter().map(HitLocator::from).collect(),
            latency_ms,
        },
        Err((message, latency_ms)) => PassageOutcome::Failed {
            message,
            latency_ms,
        },
    }
}

/// The passage lane's raw fetch, kept separate from [`run_passage_lane`]
/// so #308's `baseline` truncation path
/// ([`evidence::truncate_to_budget`]) can measure each hit's real
/// `PassageHit.text` byte/token cost — the same content
/// `crate::api::evidence::budget` accounts for server-side — before
/// [`HitLocator::from`] strips it. Truncating already-stripped
/// locators would undercount every hit's true size and make the
/// "equal budget" comparison dishonest.
/// `Ok((hits, plan, latency_ms))` on success, `Err((message,
/// latency_ms))` on failure — named so [`fetch_passage_hits`]'s
/// signature reads clearly instead of tripping clippy's
/// `type_complexity` lint on the raw nested tuple type.
type PassageFetch = Result<(Vec<PassageHit>, Option<SearchContextPlan>, u64), (String, u64)>;

fn fetch_passage_hits(api: &Api, context: &str, case: &EvalCase, limit: usize) -> PassageFetch {
    let body = serde_json::json!({
        "query": case.query,
        "limit": limit,
        "semantic_floor": case.options.floor,
        "tags": case.options.tags,
        "since": case.options.since,
        "until": case.options.until,
    });
    let started_at = Instant::now();
    match api.post(&["contexts", context, "sources", "search"], &body) {
        Ok(value) => {
            let latency_ms = elapsed_ms(started_at);
            extract_passages(&value)
                .map(|(hits, plan)| (hits, plan, latency_ms))
                .map_err(|message| (message, latency_ms))
        }
        Err(message) => Err((truncate_message(&message), elapsed_ms(started_at))),
    }
}

/// Prefers the real `{plan, hits}` shape (`PassagePage`, made
/// `Deserialize` for exactly this purpose by #282); falls back to
/// pulling `source`/`paragraph`/`score` out of a bare hits array or an
/// object whose `hits` don't fit `PassageHit`'s lane-carrying shape —
/// an older or otherwise-nonconforming server, matching
/// `benchmark/search.rs`'s own `extract_hits`.
pub(crate) fn extract_passages(
    value: &Value,
) -> Result<(Vec<PassageHit>, Option<SearchContextPlan>), String> {
    if let Ok(page) = serde_json::from_value::<PassagePage>(value.clone()) {
        let plan = page.plan.contexts.into_iter().next();
        return Ok((page.hits, plan));
    }
    let raw_hits: &Vec<Value> = if let Some(array) = value.as_array() {
        array
    } else if let Some(array) = value.get("hits").and_then(Value::as_array) {
        array
    } else {
        return Err("response carries no recognizable hits (plan/hits shape mismatch)".to_string());
    };
    let mut hits = Vec::with_capacity(raw_hits.len());
    for raw in raw_hits {
        let source = raw
            .get("source")
            .and_then(Value::as_str)
            .ok_or("a hit is missing 'source'")?
            .to_string();
        let paragraph = raw
            .get("paragraph")
            .and_then(Value::as_u64)
            .ok_or("a hit is missing 'paragraph'")? as u32;
        let score = raw.get("score").and_then(Value::as_f64).unwrap_or(0.0) as f32;
        hits.push(PassageHit {
            source,
            paragraph,
            score,
            // `HitLocator::from` still strips this before anything
            // reaches evaluation.json (ADR 0004 §11 — no corpus body
            // text written to disk), but #308's `--max-bytes`/
            // `--max-tokens` truncation (`evidence::truncate_to_budget`)
            // measures this same field's real byte/token cost BEFORE
            // that stripping happens. Recovering it here when the raw
            // hit still carries it (the common case even when the
            // whole response fails to match `PassagePage`'s stricter
            // lane shape) keeps that measurement honest; a server that
            // omits `text` entirely still degrades to an undercount,
            // which is unavoidable with no data to measure.
            text: raw
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            lanes: PassageLanes {
                bm25: None,
                vector: None,
            },
        });
    }
    Ok((hits, None))
}

// ============================= Structural lane =============================

pub(crate) fn runs_structural_lane(case: &EvalCase) -> bool {
    !case.expected_concepts.is_empty()
        || !case.expected_labels.is_empty()
        || !case.expected_associations.is_empty()
}

fn build_structural_block(api: &Api, context: &str, case: &EvalCase) -> StructuralBlock {
    let cues: Vec<String> = if case.cues.is_empty() {
        vec![case.query.clone()]
    } else {
        case.cues.clone()
    };

    let mut cue_resolutions = Vec::new();
    if !case.expected_concepts.is_empty() {
        for cue in &cues {
            cue_resolutions.push(resolve_cue(api, context, cue, "concept", false));
        }
    }
    if !case.expected_labels.is_empty() {
        for cue in &cues {
            cue_resolutions.push(resolve_cue(api, context, cue, "label", true));
        }
    }

    let associations = case
        .expected_associations
        .iter()
        .map(|assoc| build_association_probe(api, context, assoc))
        .collect();

    StructuralBlock {
        cues: cue_resolutions,
        associations,
    }
}

fn call_resolve(
    api: &Api,
    context: &str,
    cue: &str,
    labels: bool,
    limit: usize,
) -> Result<(Vec<TieredResolution>, u64), String> {
    let endpoint = if labels { "resolve_label" } else { "resolve" };
    let body = serde_json::json!({ "cue": cue, "limit": limit });
    let started_at = Instant::now();
    let value = api.post(&["contexts", context, endpoint], &body)?;
    let latency_ms = elapsed_ms(started_at);
    let resolved: Vec<TieredResolution> = serde_json::from_value(value)
        .map_err(|error| format!("{endpoint} response did not parse: {error}"))?;
    Ok((resolved, latency_ms))
}

/// ADR 0004 §7 step 1: resolve tiers are not comparable, so a caller
/// only ever reads ONE tier of a response — never lexical and semantic
/// candidates mixed. Lexical candidates always sort first when both
/// tiers are present (`resolve.rs`'s own `merge_tiers`), so the rule
/// is: the lexical group when it is non-empty, the semantic group
/// otherwise.
pub(crate) fn top_tier(candidates: &[TieredResolution]) -> Vec<&TieredResolution> {
    let lexical: Vec<&TieredResolution> =
        candidates.iter().filter(|c| c.tier == "lexical").collect();
    if !lexical.is_empty() {
        lexical
    } else {
        candidates.iter().filter(|c| c.tier == "semantic").collect()
    }
}

fn resolve_cue(
    api: &Api,
    context: &str,
    cue: &str,
    kind: &'static str,
    labels: bool,
) -> CueResolution {
    let started_at = Instant::now();
    match call_resolve(api, context, cue, labels, RESOLVE_LIMIT) {
        Ok((resolved, latency_ms)) => {
            let group = top_tier(&resolved);
            CueResolution {
                cue: cue.to_string(),
                kind,
                tier: group.first().map(|candidate| candidate.tier.clone()),
                resolved_names: group.into_iter().map(|c| c.name.clone()).collect(),
                limit: RESOLVE_LIMIT,
                latency_ms,
                error: None,
            }
        }
        Err(message) => CueResolution {
            cue: cue.to_string(),
            kind,
            resolved_names: Vec::new(),
            tier: None,
            limit: RESOLVE_LIMIT,
            latency_ms: elapsed_ms(started_at),
            error: Some(truncate_message(&message)),
        },
    }
}

/// ADR 0004 §7 step 2's stricter policy: unlike coverage's "expand the
/// whole top tier," pinning a `/query` triple needs exactly one stored
/// name per position. Zero candidates in the top tier is `not_found`;
/// two or more is `ambiguous` — either way `query` is never called for
/// that position, and no combination is guessed at.
fn resolve_position(api: &Api, context: &str, cue: &str, labels: bool) -> PositionOutcome {
    let started_at = Instant::now();
    match call_resolve(api, context, cue, labels, RESOLVE_LIMIT) {
        Ok((resolved, latency_ms)) => classify_position(top_tier(&resolved), latency_ms),
        Err(message) => PositionOutcome::Errored {
            message: truncate_message(&message),
            latency_ms: elapsed_ms(started_at),
        },
    }
}

/// The pure decision behind [`resolve_position`], split out so the
/// multi-candidate policy (ADR 0004 §7 step 2) is unit-testable
/// without a network round trip: exactly one top-tier candidate pins
/// the position, zero is `not_found`, several is `ambiguous`.
pub(crate) fn classify_position(group: Vec<&TieredResolution>, latency_ms: u64) -> PositionOutcome {
    match group.len() {
        0 => PositionOutcome::NotFound { latency_ms },
        1 => PositionOutcome::Resolved {
            name: group[0].name.clone(),
            tier: group[0].tier.clone(),
            latency_ms,
        },
        _ => PositionOutcome::Ambiguous {
            tier: group[0].tier.clone(),
            candidates: group.iter().map(|c| c.name.clone()).collect(),
            latency_ms,
        },
    }
}

fn position_name(outcome: &PositionOutcome) -> Option<&str> {
    match outcome {
        PositionOutcome::Resolved { name, .. } => Some(name.as_str()),
        _ => None,
    }
}

pub(crate) fn position_latency_ms(outcome: &PositionOutcome) -> u64 {
    match outcome {
        PositionOutcome::Resolved { latency_ms, .. }
        | PositionOutcome::NotFound { latency_ms }
        | PositionOutcome::Ambiguous { latency_ms, .. }
        | PositionOutcome::Errored { latency_ms, .. } => *latency_ms,
    }
}

/// `query` pins all three positions exactly (ADR 0004 §7 step 2) —
/// `limit: 1` is enough since a pinned triple's `total` is 0 or 1; a
/// single match can still carry several `attributions[]` (the same
/// triple asserted from more than one source), which citation recall
/// (ADR 0004 §8) reads as served `(source, paragraph)` locators
/// alongside the passage lane's own hits.
fn run_query(api: &Api, context: &str, subject: &str, label: &str, object: &str) -> QueryProbe {
    let body = serde_json::json!({
        "subject": subject,
        "label": label,
        "object": object,
        "limit": 1,
    });
    let started_at = Instant::now();
    match api.post(&["contexts", context, "query"], &body) {
        Ok(value) => {
            let latency_ms = elapsed_ms(started_at);
            match serde_json::from_value::<MatchPage>(value) {
                Ok(page) => {
                    let attributions = page
                        .matches
                        .iter()
                        .flat_map(|matched| matched.attributions.iter())
                        .map(|attribution| AttributionLocator {
                            source: attribution.source.clone(),
                            paragraph: attribution.paragraph,
                        })
                        .collect();
                    QueryProbe::Queried {
                        total: page.total,
                        matches: page.matches.len(),
                        attributions,
                        latency_ms,
                    }
                }
                Err(error) => QueryProbe::Errored {
                    message: format!("query response did not parse: {error}"),
                    latency_ms,
                },
            }
        }
        Err(message) => QueryProbe::Errored {
            message: truncate_message(&message),
            latency_ms: elapsed_ms(started_at),
        },
    }
}

pub(crate) fn query_latency_ms(probe: &QueryProbe) -> u64 {
    match probe {
        QueryProbe::Queried { latency_ms, .. } | QueryProbe::Errored { latency_ms, .. } => {
            *latency_ms
        }
    }
}

fn build_association_probe(
    api: &Api,
    context: &str,
    assoc: &ExpectedAssociation,
) -> AssociationProbe {
    let subject = resolve_position(api, context, &assoc.subject, false);
    let label = resolve_position(api, context, &assoc.label, true);
    let object = resolve_position(api, context, &assoc.object, false);

    let query = match (
        position_name(&subject),
        position_name(&label),
        position_name(&object),
    ) {
        (Some(s), Some(l), Some(o)) => Some(run_query(api, context, s, l, o)),
        _ => None,
    };

    AssociationProbe {
        subject_cue: assoc.subject.clone(),
        label_cue: assoc.label.clone(),
        object_cue: assoc.object.clone(),
        subject,
        label,
        object,
        query,
    }
}

// ============================ Citation lane ============================

fn runs_citation_lane(case: &EvalCase) -> bool {
    !case.expected_citations.is_empty()
}

/// One `POST /contexts/{name}/citations` call per `expected_citations[]`
/// entry, strictly sequential — the endpoint takes exactly one locator
/// per request (`src/api/sources.rs:67-74`), never a batch, so N
/// expectations cost N round trips. Deliberately NOT preflighted the
/// way `expected_sources` is (`run_evaluate`'s missing-source check
/// above): a citation naming a source or paragraph the corpus does not
/// carry is exactly the failure ADR 0004 §8's locator-validity
/// measurement exists to catch, so it must run to completion and be
/// recorded — never abort the whole run. This also means the lane runs
/// even on a case whose passage-lane search missed entirely: citation
/// recall and locator validity are measured independently.
fn run_citation_lane(
    api: &Api,
    context: &str,
    case: &EvalCase,
    served: &BTreeSet<(String, u32)>,
) -> Vec<CitationCheck> {
    case.expected_citations
        .iter()
        .map(|expected| check_citation(api, context, expected, served))
        .collect()
}

fn check_citation(
    api: &Api,
    context: &str,
    expected: &ExpectedCitation,
    served: &BTreeSet<(String, u32)>,
) -> CitationCheck {
    let body = serde_json::json!({
        "source": expected.source,
        "paragraph": expected.paragraph,
    });
    let started_at = Instant::now();
    let outcome = match api.post_envelope(&["contexts", context, "citations"], &body) {
        Ok(value) => match serde_json::from_value::<Citation>(value) {
            Ok(citation) => {
                let section = check_section(&expected.section, &citation.section);
                let quote = expected.quote.as_ref().map(|quote| QuoteCheck {
                    matched: quote_matches(quote, &citation.text),
                    declared: quote.clone(),
                });
                CitationOutcome::Resolved { section, quote }
            }
            Err(error) => CitationOutcome::Unresolved {
                code: None,
                message: truncate_message(&format!("citation response did not parse: {error}")),
            },
        },
        Err(ApiFailure::NotFound { code, message }) => CitationOutcome::Unresolved {
            code,
            message: truncate_message(&message),
        },
        Err(ApiFailure::Other(message)) => CitationOutcome::Unresolved {
            code: None,
            message: truncate_message(&message),
        },
    };
    CitationCheck {
        source: expected.source.clone(),
        paragraph: expected.paragraph,
        served: served.contains(&(expected.source.clone(), expected.paragraph)),
        outcome,
        latency_ms: elapsed_ms(started_at),
    }
}

// ================================ Scoring ==================================
//
// Everything below reads only rank (a hit's position in `hits[]`, a
// cue's own resolved-name list, a query's `total`) and label
// (`expected_sources`' `relevance`, `expected_concepts`/
// `expected_labels`/`expected_associations`) — never a `HitLocator`'s
// or `TieredResolution`'s own `score` field (#215's "do not collapse
// incomparable raw scores from graph, BM25, and vector lanes into one
// invented scale" requirement). Pure functions, no network access, so
// every rule here is unit-testable without a server.

/// One case's recall@k/MRR/nDCG against `expected_sources` (ADR 0004
/// §274). `None` when the case declares no `relevance >= 1` source —
/// `relevance == 0` means "not evidence for this case," dropped from
/// the denominator like `benchmark::search`'s own `resolve_expected_items`
/// (`search.rs:1014-1016`).
/// Whether `hit` satisfies `expected`: same `source`, and either
/// `expected.paragraphs` is empty (any paragraph of this source
/// answers the case) or it names `hit.paragraph` explicitly. The one
/// place this policy lives — [`score_recall`] and [`build_missed`]
/// both call it, so a future change to source matching cannot diverge
/// between "how a case scores" and "why a case says it missed."
fn source_matches(expected: &ExpectedSource, hit: &HitLocator) -> bool {
    hit.source == expected.source
        && (expected.paragraphs.is_empty() || expected.paragraphs.contains(&hit.paragraph))
}

pub(crate) fn score_recall(
    expected_sources: &[ExpectedSource],
    hits: &[HitLocator],
) -> Option<RecallBlock> {
    let items: Vec<&ExpectedSource> = expected_sources
        .iter()
        .filter(|expected| expected.relevance >= 1)
        .collect();
    if items.is_empty() {
        return None;
    }

    let matched = items
        .iter()
        .filter(|item| hits.iter().any(|hit| source_matches(item, hit)))
        .count();

    let mut mrr = 0.0;
    for (rank, hit) in hits.iter().enumerate() {
        if items.iter().any(|item| source_matches(item, hit)) {
            mrr = 1.0 / (rank as f64 + 1.0);
            break;
        }
    }

    // One credit per expectation, at the rank of the first hit that
    // satisfies it — not one credit per matching hit. A wildcard entry
    // (`paragraphs` empty) that lands on several hits is not double
    // counted, so IDCG is well-defined from the label multiset alone
    // regardless of how many hits happen to carry one label.
    let dcg: f64 = items
        .iter()
        .filter_map(|item| {
            hits.iter()
                .position(|hit| source_matches(item, hit))
                .map(|rank| item.relevance as f64 / (rank as f64 + 2.0).log2())
        })
        .sum();

    let mut ideal: Vec<u8> = items.iter().map(|item| item.relevance).collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = ideal
        .iter()
        .enumerate()
        .map(|(rank, relevance)| *relevance as f64 / (rank as f64 + 2.0).log2())
        .sum();
    // IDCG assigns each expectation its own distinct rank, but DCG does
    // not: two expectations satisfied by the same hit (e.g. a wildcard
    // `paragraphs: []` entry and a paragraph-restricted entry on the
    // same source) both credit at that hit's one rank, so DCG can
    // exceed IDCG. Clamped to keep this a well-formed 0..=1 ratio.
    let ndcg = if idcg > 0.0 {
        (dcg / idcg).min(1.0)
    } else {
        0.0
    };

    Some(RecallBlock {
        recall_at_k: matched as f64 / items.len() as f64,
        mrr,
        ndcg,
        expected_total: items.len(),
        matched,
    })
}

/// Every `(source, paragraph)` this case's served results carry (ADR
/// 0004 §8's citation recall): the passage lane's hits up to its own
/// `limit`, plus the structural lane's `AttributionOut` locators when
/// it ran — never one lane's own raw score, only rank-independent
/// membership. An attribution with no `paragraph` locator at all
/// contributes nothing, since `expected_citations` always names one.
pub(crate) fn served_locators(
    passage: &PassageOutcome,
    structural: Option<&StructuralBlock>,
) -> BTreeSet<(String, u32)> {
    let mut served = BTreeSet::new();
    if let PassageOutcome::Searched { hits, .. } = passage {
        served.extend(hits.iter().map(|hit| (hit.source.clone(), hit.paragraph)));
    }
    if let Some(structural) = structural {
        served.extend(
            structural
                .associations
                .iter()
                .filter_map(|assoc| assoc.query.as_ref())
                .flat_map(|query| {
                    let attributions: &[AttributionLocator] = match query {
                        QueryProbe::Queried { attributions, .. } => attributions,
                        QueryProbe::Errored { .. } => &[],
                    };
                    attributions.iter()
                })
                .filter_map(|locator| locator.paragraph.map(|p| (locator.source.clone(), p))),
        );
    }
    served
}

/// [`ExpectedCitation::section`]'s three-valued check (ADR 0004 §8): an
/// absent key (`None`) means "don't check section" and never runs a
/// comparison; a present key — including an explicit `null` — is
/// compared against the server's `Citation.section` as-is, so
/// `Some(None)` correctly asserts "outside every stored section."
pub(crate) fn check_section(
    expected: &Option<Option<String>>,
    actual: &Option<String>,
) -> SectionCheck {
    match expected {
        None => SectionCheck::NotChecked,
        Some(expected) if expected == actual => SectionCheck::Matched {
            expected: expected.clone(),
        },
        Some(expected) => SectionCheck::Mismatched {
            expected: expected.clone(),
        },
    }
}

/// Whether `quote` is a substring of `text` after both sides go through
/// `normalize_entry` (ADR 0004 §8 — the same folding the passage index
/// itself uses, trimmed first since `normalize_entry` does not trim,
/// matching [`resolved_contains`]'s own preprocessing). `text` is
/// always exactly one paragraph (`Citation.text`, `src/api/sources.rs:
/// 84-88`), so a `quote` spanning a paragraph boundary can never match
/// here — ADR 0004 §8's documented workaround is splitting it into two
/// `expected_citations` entries.
pub(crate) fn quote_matches(quote: &str, text: &str) -> bool {
    let target = normalize_entry(quote.trim());
    normalize_entry(text.trim()).contains(&target)
}

/// A [`CitationCheck`] is valid (ADR 0004 §8's locator-validity
/// measurement) when it resolved, its `section` check never mismatched
/// (an absent expectation or a match both count), and its `quote` check
/// — when the case declared one — matched.
pub(crate) fn citation_is_valid(check: &CitationCheck) -> bool {
    match &check.outcome {
        CitationOutcome::Resolved { section, quote } => {
            !matches!(section, SectionCheck::Mismatched { .. })
                && quote.as_ref().is_none_or(|quote| quote.matched)
        }
        CitationOutcome::Unresolved { .. } => false,
    }
}

/// Citation recall (ADR 0004 §8): the fraction of `checks` whose
/// `(source, paragraph)` appeared among this case's served results.
/// Callers only call this on a non-empty `checks` — an empty
/// `expected_citations` means the citation lane never ran at all, so
/// there is nothing here to divide by.
pub(crate) fn score_citation_recall(checks: &[CitationCheck]) -> CitationRecallBlock {
    let matched = checks.iter().filter(|check| check.served).count();
    CitationRecallBlock {
        expected_total: checks.len(),
        matched,
        value: matched as f64 / checks.len() as f64,
    }
}

/// Locator validity (ADR 0004 §8): the fraction of `checks` that
/// resolved with a matching `section` (when declared) and `quote` (when
/// declared) — never merged with [`score_citation_recall`]'s own ratio.
pub(crate) fn score_citation_validity(checks: &[CitationCheck]) -> CitationValidityBlock {
    let valid = checks
        .iter()
        .filter(|check| citation_is_valid(check))
        .count();
    CitationValidityBlock {
        expected_total: checks.len(),
        valid,
        value: valid as f64 / checks.len() as f64,
    }
}

/// Whether `expected` (case-declared) appears among `resolutions`'
/// `resolved_names[]`, matched with `normalize_entry` on both sides
/// (ADR 0004 §8) — `normalize_entry` does not trim, so both sides are
/// trimmed first.
pub(crate) fn resolved_contains(resolutions: &[&CueResolution], expected: &str) -> bool {
    let target = normalize_entry(expected.trim());
    resolutions
        .iter()
        .flat_map(|resolution| &resolution.resolved_names)
        .any(|name| normalize_entry(name.trim()) == target)
}

/// Coverage of one expectation list (`expected_concepts` or
/// `expected_labels`) against the matching-kind `CueResolution`s.
/// `None` when the case declares none — coverage does not apply, as
/// distinct from applying and finding zero.
pub(crate) fn coverage_counts(
    expected: &[String],
    resolutions: &[&CueResolution],
) -> Option<CoverageCounts> {
    if expected.is_empty() {
        return None;
    }
    let matched = expected
        .iter()
        .filter(|item| resolved_contains(resolutions, item))
        .count();
    Some(CoverageCounts {
        expected: expected.len(),
        matched,
        value: matched as f64 / expected.len() as f64,
    })
}

/// Coverage of `expected_associations`: an entry counts as covered
/// only when its `/query` call ran and returned `total >= 1` — the
/// same "exactly one candidate pins a position, otherwise `query` is
/// never called" policy (ADR 0004 §7 step 2) means `not_found`/
/// `ambiguous`/`Errored` positions and a never-run `query` are all
/// uncovered, never guessed at.
pub(crate) fn association_coverage(associations: &[AssociationProbe]) -> Option<CoverageCounts> {
    if associations.is_empty() {
        return None;
    }
    let matched = associations
        .iter()
        .filter(
            |probe| matches!(&probe.query, Some(QueryProbe::Queried { total, .. }) if *total >= 1),
        )
        .count();
    Some(CoverageCounts {
        expected: associations.len(),
        matched,
        value: matched as f64 / associations.len() as f64,
    })
}

/// `missed[]`, capped at 3 entries with a count of the entries
/// *dropped* — not the total (ADR 0004 §11). Order: sources, concepts,
/// labels, associations, citations. Silent about sources when the
/// passage lane failed — its own `Failed` outcome and
/// `passage.failure_rate` already record that; guessing at which
/// sources it would have missed adds no information. Citation checks
/// run regardless of the passage lane's own outcome (ADR 0004 §8), so
/// they are never silenced that way.
pub(crate) fn build_missed(
    case: &EvalCase,
    hits: Option<&[HitLocator]>,
    concept_resolutions: &[&CueResolution],
    label_resolutions: &[&CueResolution],
    structural: Option<&StructuralBlock>,
    citation_checks: &[CitationCheck],
) -> (Vec<String>, usize) {
    let mut all = Vec::new();

    if let Some(hits) = hits {
        for expected in &case.expected_sources {
            if expected.relevance == 0 {
                continue;
            }
            let hit = hits.iter().any(|h| source_matches(expected, h));
            if !hit {
                all.push(format!(
                    "expected_sources: '{}' not found among passage hits",
                    expected.source
                ));
            }
        }
    }

    for concept in &case.expected_concepts {
        if !resolved_contains(concept_resolutions, concept) {
            all.push(format!("expected_concepts: '{concept}' not resolved"));
        }
    }
    for label in &case.expected_labels {
        if !resolved_contains(label_resolutions, label) {
            all.push(format!("expected_labels: '{label}' not resolved"));
        }
    }
    if let Some(structural) = structural {
        for assoc in &structural.associations {
            let why: Option<String> = match &assoc.query {
                Some(QueryProbe::Queried { total, .. }) if *total >= 1 => None,
                Some(QueryProbe::Queried { .. }) => {
                    Some("query returned no association".to_string())
                }
                Some(QueryProbe::Errored { .. }) => Some("query errored".to_string()),
                None => {
                    // A position's own `/resolve` call failing outright
                    // (`PositionOutcome::Errored`, a transport-level
                    // fault) is never the same fact as zero/multiple
                    // candidates coming back cleanly — the former is
                    // possibly-transient infra trouble, the latter a
                    // data-quality miss. Keep them distinguishable in
                    // `missed[]` rather than collapsing both into "a
                    // position did not pin."
                    let errored_positions: Vec<(&str, &str)> = [
                        ("subject", &assoc.subject),
                        ("label", &assoc.label),
                        ("object", &assoc.object),
                    ]
                    .into_iter()
                    .filter_map(|(name, outcome)| match outcome {
                        PositionOutcome::Errored { message, .. } => Some((name, message.as_str())),
                        _ => None,
                    })
                    .collect();
                    if errored_positions.is_empty() {
                        Some("query never ran (a position did not pin)".to_string())
                    } else {
                        Some(format!(
                            "a position could not be resolved due to a transport error: {}",
                            errored_positions
                                .iter()
                                .map(|(name, message)| format!("{name}: {message}"))
                                .collect::<Vec<_>>()
                                .join("; ")
                        ))
                    }
                }
            };
            if let Some(why) = why {
                all.push(format!(
                    "expected_associations: ({}, {}, {}) {why}",
                    assoc.subject_cue, assoc.label_cue, assoc.object_cue
                ));
            }
        }
    }

    // ADR 0004 §8's two citation measurements are never merged into one
    // score, so a bad check can add up to two distinct entries here —
    // one for recall (not served), one for validity (locator itself
    // wrong) — never collapsed into a single ambiguous message.
    for check in citation_checks {
        if !check.served {
            all.push(format!(
                "expected_citations: ({}, {}) not found among served results",
                check.source, check.paragraph
            ));
        }
        if !citation_is_valid(check) {
            all.push(format!(
                "expected_citations: ({}, {}) failed locator validity",
                check.source, check.paragraph
            ));
        }
    }

    let truncated = all.len().saturating_sub(3);
    all.truncate(3);
    (all, truncated)
}

// ================================ Per case ================================

/// #308 (ADR 0006 §14): builds the passage lane's `PassageOutcome`
/// (what every scoring function below reads) plus the two mode-
/// specific diagnostics — `EvidenceOutcome` (`assembly` only) and
/// `BudgetAccounting` (either mode, once a budget flag was given).
/// `assembly` mode always sends `run_config.limits` to
/// `POST /contexts/{name}/evidence` (that endpoint has no unbudgeted
/// mode, ADR 0006 §8); `baseline` mode only truncates — and only then
/// carries a `budget` block at all — when `run_config.budget_given`.
fn run_passage_or_evidence_lane(
    api: &Api,
    context: &str,
    case: &EvalCase,
    limit: usize,
    run_config: &RunConfig,
) -> (
    PassageOutcome,
    Option<EvidenceOutcome>,
    Option<BudgetAccounting>,
) {
    if run_config.assembly {
        match evidence::run_evidence_lane(
            api,
            context,
            case,
            limit,
            &run_config.limits,
            run_config.rerank,
        ) {
            evidence::LaneResult::Assembled {
                package,
                latency_ms,
            } => {
                let hits = hits_from_evidence_items(&package.items, limit);
                let locators = evidence_locators(&package.items);
                let passage = PassageOutcome::Searched {
                    plan: None,
                    hits,
                    latency_ms,
                };
                let budget = BudgetAccounting {
                    usage: package.budget,
                    omitted_total: package.omitted_total,
                };
                let outcome = EvidenceOutcome::Assembled {
                    latency_ms,
                    items: locators,
                    omitted_by_reason: package.omitted_by_reason,
                    selection: package.plan.selection,
                    reranker: package.plan.reranker,
                };
                (passage, Some(outcome), Some(budget))
            }
            evidence::LaneResult::Failed {
                message,
                latency_ms,
            } => {
                let passage = PassageOutcome::Failed {
                    message: message.clone(),
                    latency_ms,
                };
                let outcome = EvidenceOutcome::Failed {
                    message,
                    latency_ms,
                };
                (passage, Some(outcome), None)
            }
        }
    } else if run_config.budget_given {
        match fetch_passage_hits(api, context, case, limit) {
            Ok((hits, plan, latency_ms)) => {
                let truncated = evidence::truncate_to_budget(context, hits, &run_config.limits);
                let passage = PassageOutcome::Searched {
                    plan,
                    hits: truncated.hits.into_iter().map(HitLocator::from).collect(),
                    latency_ms,
                };
                let budget = BudgetAccounting {
                    usage: truncated.usage,
                    omitted_total: truncated.omitted_total,
                };
                (passage, None, Some(budget))
            }
            Err((message, latency_ms)) => (
                PassageOutcome::Failed {
                    message,
                    latency_ms,
                },
                None,
                None,
            ),
        }
    } else {
        (run_passage_lane(api, context, case, limit), None, None)
    }
}

pub(crate) fn build_case_block(
    api: &Api,
    context: &str,
    case: &EvalCase,
    default_limit: usize,
    run_config: &RunConfig,
) -> CaseBlock {
    let limit = case.options.limit.unwrap_or(default_limit);
    let (passage, evidence_outcome, budget) =
        run_passage_or_evidence_lane(api, context, case, limit, run_config);
    let diversity_sources = match &passage {
        PassageOutcome::Searched { hits, .. } => Some(
            hits.iter()
                .map(|hit| hit.source.as_str())
                .collect::<BTreeSet<_>>()
                .len(),
        ),
        PassageOutcome::Failed { .. } => None,
    };
    let structural = runs_structural_lane(case).then(|| build_structural_block(api, context, case));
    // Citation recall needs to know what the first two lanes already
    // served BEFORE the citation lane's own network calls run, so the
    // per-check `served` bit can be set at request time rather than
    // reconciled afterward.
    let served = served_locators(&passage, structural.as_ref());
    let citation_checks = if runs_citation_lane(case) {
        run_citation_lane(api, context, case, &served)
    } else {
        Vec::new()
    };
    let scores = score_case(case, &passage, structural.as_ref(), &citation_checks);

    CaseBlock {
        case_id: case.case_id.clone(),
        query: case.query.clone(),
        cues: case.cues.clone(),
        limit,
        passage,
        structural,
        recall: scores.recall,
        coverage: scores.coverage,
        lane_cross: scores.lane_cross,
        citations: scores.citations,
        missed: scores.missed,
        missed_truncated: scores.missed_truncated,
        evidence: evidence_outcome,
        budget,
        diversity_sources,
    }
}
