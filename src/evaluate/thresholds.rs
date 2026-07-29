//! `--thresholds FILE` (#276, ADR 0004 §9.3, §5, §12): a checked-in,
//! user-authored JSON file that turns `evaluate` from a report-only run
//! into a CI gate. Stamp `taguru_evaluate_thresholds: 1`,
//! equality-checked (`load_thresholds` below) — the same `taguru_batch`
//! posture `eval.jsonl` itself uses (`evalset.rs`'s module doc), not
//! `evaluation.json`'s own `IMAGE_VERSION` range acceptance, because a
//! human authors this file by hand.
//!
//! Three top-level keys: `aggregate` (metric name → bound, checked
//! against `metrics`), `cases` (`default` + per-`case_id` `overrides`,
//! checked against one case's own value — see [`CASE_SCOPED_METRICS`]),
//! and `allow_unstable_corpus` (ADR 0004 §12's corpus-revision gate).
//! `#[serde(deny_unknown_fields)]` catches a typo'd top-level or bound
//! key, but cannot catch an unknown *metric name* or `case_id` inside a
//! map it does not itself validate — ADR 0004 §9.3 names this
//! explicitly as the easy-to-miss part, so [`load_thresholds`] runs
//! that check itself, as a post-load set difference against
//! `evaluation.json`'s own `definitions` keys and the loaded
//! `eval.jsonl`'s `case_id`s.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::{CaseBlock, PassageOutcome};
use crate::measure::{MetricDef, MetricValue, MetricsMap};

const THRESHOLDS_VERSION: u64 = 1;

/// One `aggregate`/`cases.default`/`cases.overrides.*` entry: at least
/// one of `min`/`max`, both finite, and (when both present) `min <=
/// max` — checked in [`validate_bound`], since `deny_unknown_fields`
/// only rejects a stray key, not an incoherent pair of values.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Bound {
    #[serde(default)]
    min: Option<f64>,
    #[serde(default)]
    max: Option<f64>,
}

fn validate_bound(bound: &Bound) -> Result<(), String> {
    if bound.min.is_none() && bound.max.is_none() {
        return Err("needs at least one of min/max".to_string());
    }
    if let Some(min) = bound.min
        && !min.is_finite()
    {
        return Err(format!("min {min} must be finite"));
    }
    if let Some(max) = bound.max
        && !max.is_finite()
    {
        return Err(format!("max {max} must be finite"));
    }
    if let (Some(min), Some(max)) = (bound.min, bound.max)
        && min > max
    {
        return Err(format!("min {min} must be <= max {max}"));
    }
    Ok(())
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct CasesThresholds {
    #[serde(default)]
    default: BTreeMap<String, Bound>,
    #[serde(default)]
    overrides: BTreeMap<String, BTreeMap<String, Bound>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThresholdsFile {
    taguru_evaluate_thresholds: u64,
    #[serde(default)]
    aggregate: BTreeMap<String, Bound>,
    #[serde(default)]
    cases: CasesThresholds,
    #[serde(default)]
    allow_unstable_corpus: bool,
}

/// Metric names [`case_metric_value`] can answer for one case — the
/// only names a `cases.default`/`cases.overrides.*` entry may use.
/// `latency.resolve_ms`/`latency.query_ms`/`latency.citation_ms` are
/// deliberately absent even though `definitions` scopes them `"case"`:
/// a single case can make more than one resolve/query/citation call,
/// so there is no one per-case value to compare against a bound —
/// only `latency.passage_ms`, whose case always makes exactly one
/// search call, qualifies.
pub(super) const CASE_SCOPED_METRICS: &[&str] = &[
    "recall.recall_at_k",
    "recall.mrr",
    "recall.ndcg",
    "coverage.concepts",
    "coverage.labels",
    "coverage.associations",
    "citations.recall",
    "citations.locator_validity",
    "latency.passage_ms",
];

/// [`CASE_SCOPED_METRICS`] minus `latency.passage_ms` — the set
/// `evaluate::compare` (#277) compares case-by-case to classify a case
/// `improved`/`regressed`/`unchanged`. Single-run latency is noisy
/// enough that including it would flag most cases on rerun alone
/// rather than on a genuine quality change, so `compare` only judges
/// the eight quality metrics; `compare_lexicon_test` (in
/// `super::compare::tests`) asserts this stays exactly
/// `CASE_SCOPED_METRICS` minus that one name, so a future addition to
/// `CASE_SCOPED_METRICS` does not silently go unconsidered here.
pub(super) const COMPARISON_METRICS: &[&str] = &[
    "recall.recall_at_k",
    "recall.mrr",
    "recall.ndcg",
    "coverage.concepts",
    "coverage.labels",
    "coverage.associations",
    "citations.recall",
    "citations.locator_validity",
];

/// One case's value for a [`CASE_SCOPED_METRICS`] name — `None` when
/// the case never computed one (it did not declare the matching
/// expectation, or the lane that would have produced it did not
/// complete), which a caller treats as "not applicable," never as a
/// violation.
fn case_metric_value(case: &CaseBlock, metric: &str) -> Option<f64> {
    match metric {
        "recall.recall_at_k" => case.recall.as_ref().map(|r| r.recall_at_k),
        "recall.mrr" => case.recall.as_ref().map(|r| r.mrr),
        "recall.ndcg" => case.recall.as_ref().map(|r| r.ndcg),
        "coverage.concepts" => case.coverage.as_ref()?.concepts.as_ref().map(|c| c.value),
        "coverage.labels" => case.coverage.as_ref()?.labels.as_ref().map(|c| c.value),
        "coverage.associations" => case
            .coverage
            .as_ref()?
            .associations
            .as_ref()
            .map(|c| c.value),
        "citations.recall" => case.citations.as_ref().map(|c| c.recall.value),
        "citations.locator_validity" => case.citations.as_ref().map(|c| c.validity.value),
        "latency.passage_ms" => Some(match &case.passage {
            PassageOutcome::Searched { latency_ms, .. } => *latency_ms as f64,
            PassageOutcome::Failed { latency_ms, .. } => *latency_ms as f64,
        }),
        _ => None,
    }
}

/// A loaded, fully validated `--thresholds FILE`, ready to check
/// against one run's `cases`/`metrics`/`stable` — see [`Self::evaluate`].
#[derive(Debug)]
pub(super) struct LoadedThresholds {
    sha256: String,
    file: ThresholdsFile,
}

/// Reads, parses, and validates `path` (ADR 0004 §9.3). `definitions`
/// is `evaluate`'s own `build_definitions()` output — the set of
/// metric names `aggregate` may legally name; `case_ids` is the loaded
/// `eval.jsonl`'s own `case_id`s — the set `cases.overrides` may
/// legally name. Every problem found is collected and reported
/// together, not just the first, since a hand-edited thresholds file
/// commonly has more than one typo at once.
pub(super) fn load_thresholds(
    path: &Path,
    definitions: &BTreeMap<String, MetricDef>,
    case_ids: &BTreeSet<String>,
) -> Result<LoadedThresholds, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let file: ThresholdsFile =
        serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()))?;
    if file.taguru_evaluate_thresholds != THRESHOLDS_VERSION {
        return Err(format!(
            "{}: taguru_evaluate_thresholds must be {THRESHOLDS_VERSION}, got {}",
            path.display(),
            file.taguru_evaluate_thresholds
        ));
    }

    let mut problems = Vec::new();

    // ADR 0004 §9.3: `deny_unknown_fields` cannot reject an unknown
    // metric name or case_id inside a map it does not itself validate
    // — this is that check, run explicitly against `evaluation.json`'s
    // own `definitions` keys and the loaded dataset's `case_id`s.
    let known_metrics: BTreeSet<&str> = definitions.keys().map(String::as_str).collect();
    let case_scoped: BTreeSet<&str> = CASE_SCOPED_METRICS.iter().copied().collect();

    for name in file.aggregate.keys() {
        if !known_metrics.contains(name.as_str()) {
            problems.push(format!("aggregate: unknown metric name '{name}'"));
        }
    }
    for name in file.cases.default.keys() {
        if !case_scoped.contains(name.as_str()) {
            problems.push(format!(
                "cases.default: unknown (or not case-scoped) metric name '{name}'"
            ));
        }
    }
    for (case_id, bounds) in &file.cases.overrides {
        if !case_ids.contains(case_id.as_str()) {
            problems.push(format!("cases.overrides: unknown case_id '{case_id}'"));
        }
        for name in bounds.keys() {
            if !case_scoped.contains(name.as_str()) {
                problems.push(format!(
                    "cases.overrides.{case_id}: unknown (or not case-scoped) metric name '{name}'"
                ));
            }
        }
    }

    let every_bound = file
        .aggregate
        .iter()
        .map(|(name, bound)| ("aggregate".to_string(), name, bound))
        .chain(
            file.cases
                .default
                .iter()
                .map(|(name, bound)| ("cases.default".to_string(), name, bound)),
        )
        .chain(file.cases.overrides.iter().flat_map(|(case_id, bounds)| {
            bounds
                .iter()
                .map(move |(name, bound)| (format!("cases.overrides.{case_id}"), name, bound))
        }));
    for (scope, name, bound) in every_bound {
        if let Err(reason) = validate_bound(bound) {
            problems.push(format!("{scope}: '{name}': {reason}"));
        }
    }

    if !problems.is_empty() {
        return Err(format!("{}: {}", path.display(), problems.join("; ")));
    }

    Ok(LoadedThresholds {
        sha256: crate::extract::sha256_hex(&bytes),
        file,
    })
}

/// One threshold violation (ADR 0004 §9.1/§11: never a URL, a
/// credential, or corpus body text — only a metric name, a bound, a
/// number, and an optional `case_id`, so masking is structural rather
/// than a rule someone has to remember to apply here).
#[derive(Debug, Clone, Serialize)]
pub(super) struct Violation {
    pub(super) scope: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) case_id: Option<String>,
    pub(super) metric: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) bound: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) threshold: Option<f64>,
    pub(super) actual: Option<f64>,
    pub(super) reason: String,
}

/// `evaluation.json`'s `thresholds` block once a run has one (ADR 0004
/// §9.1's threshold identity, extended with the pass/fail verdict and
/// the violations/skips that produced it).
#[derive(Debug, Clone, Serialize)]
pub(super) struct ThresholdReport {
    pub(super) sha256: String,
    pub(super) passed: bool,
    pub(super) allow_unstable_corpus: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) violations: Vec<Violation>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) skipped: Vec<String>,
}

impl LoadedThresholds {
    /// Checks this run's `cases`/`metrics`/`stable` against the loaded
    /// bounds. `aggregate` compares [`MetricValue::scalar`] against
    /// `metrics[name]`; a `case` bound is `cases.default` merged with
    /// `cases.overrides[case_id]` per metric name (override wins), read
    /// through [`case_metric_value`]; a metric a case never computed is
    /// `skipped`, not violated. `!stable && !allow_unstable_corpus`
    /// (ADR 0004 §12) adds one more `corpus`-scoped violation —
    /// `corpus.stable` is not a `definitions` entry, so
    /// `allow_unstable_corpus` is the only lever, never an `aggregate`
    /// bound.
    pub(super) fn evaluate(
        &self,
        cases: &[CaseBlock],
        metrics: &MetricsMap,
        stable: bool,
    ) -> ThresholdReport {
        let mut violations = Vec::new();
        let mut skipped = Vec::new();

        for (metric, bound) in &self.file.aggregate {
            let actual = metrics.get(metric).and_then(MetricValue::scalar);
            check_bound("aggregate", None, metric, bound, actual, &mut violations);
        }

        for case in cases {
            let mut merged: BTreeMap<&str, Bound> = self
                .file
                .cases
                .default
                .iter()
                .map(|(name, bound)| (name.as_str(), *bound))
                .collect();
            if let Some(overrides) = self.file.cases.overrides.get(&case.case_id) {
                for (name, bound) in overrides {
                    merged.insert(name.as_str(), *bound);
                }
            }
            for (metric, bound) in merged {
                match case_metric_value(case, metric) {
                    Some(actual) => check_bound(
                        "case",
                        Some(case.case_id.clone()),
                        metric,
                        &bound,
                        Some(actual),
                        &mut violations,
                    ),
                    None => skipped.push(format!(
                        "case '{}': '{metric}' not applicable — the case did not declare the \
                         matching expectation, or its lane did not complete",
                        case.case_id
                    )),
                }
            }
        }

        if !stable && !self.file.allow_unstable_corpus {
            violations.push(Violation {
                scope: "corpus",
                case_id: None,
                metric: "corpus.stable".to_string(),
                bound: None,
                threshold: None,
                actual: None,
                reason: "the context's revision changed during this run \
                         (corpus.revision_before != corpus.revision_after) and \
                         allow_unstable_corpus is false"
                    .to_string(),
            });
        }

        let passed = violations.is_empty();
        ThresholdReport {
            sha256: self.sha256.clone(),
            passed,
            allow_unstable_corpus: self.file.allow_unstable_corpus,
            violations,
            skipped,
        }
    }
}

fn check_bound(
    scope: &'static str,
    case_id: Option<String>,
    metric: &str,
    bound: &Bound,
    actual: Option<f64>,
    violations: &mut Vec<Violation>,
) {
    let Some(value) = actual else {
        violations.push(Violation {
            scope,
            case_id,
            metric: metric.to_string(),
            bound: None,
            threshold: None,
            actual: None,
            reason: "no samples for this metric (n=0)".to_string(),
        });
        return;
    };
    if let Some(min) = bound.min
        && value < min
    {
        violations.push(Violation {
            scope,
            case_id,
            metric: metric.to_string(),
            bound: Some("min"),
            threshold: Some(min),
            actual: Some(value),
            reason: format!("{value:.6} < min {min:.6}"),
        });
        return;
    }
    if let Some(max) = bound.max
        && value > max
    {
        violations.push(Violation {
            scope,
            case_id,
            metric: metric.to_string(),
            bound: Some("max"),
            threshold: Some(max),
            actual: Some(value),
            reason: format!("{value:.6} > max {max:.6}"),
        });
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::evaluate::{
        CitationRecallBlock, CitationValidityBlock, CitationsBlock, CoverageBlock, CoverageCounts,
        RecallBlock, build_definitions,
    };
    use crate::measure::{Distribution, ratio_metric};

    #[test]
    fn case_scoped_metrics_are_all_known_definitions() {
        let definitions = build_definitions();
        for metric in CASE_SCOPED_METRICS {
            assert!(
                definitions.contains_key(*metric),
                "'{metric}' is in CASE_SCOPED_METRICS but not build_definitions()"
            );
        }
    }

    #[test]
    fn latency_resolve_query_citation_ms_are_not_case_scoped() {
        for excluded in [
            "latency.resolve_ms",
            "latency.query_ms",
            "latency.citation_ms",
        ] {
            assert!(
                !CASE_SCOPED_METRICS.contains(&excluded),
                "'{excluded}' can be called more than once per case and must stay excluded"
            );
        }
    }

    // ============================ load_thresholds ============================

    fn write_temp(tag: &str, contents: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "taguru-thresholds-unit-{tag}-{}.json",
            std::process::id()
        ));
        fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn rejects_a_stamp_mismatch() {
        let path = write_temp("stamp", "{\"taguru_evaluate_thresholds\":2}");
        let error = load_thresholds(&path, &build_definitions(), &BTreeSet::new()).unwrap_err();
        assert!(error.contains("taguru_evaluate_thresholds"), "{error}");
    }

    #[test]
    fn rejects_an_unknown_top_level_key() {
        let path = write_temp(
            "top-level",
            "{\"taguru_evaluate_thresholds\":1,\"bogus\":true}",
        );
        let error = load_thresholds(&path, &build_definitions(), &BTreeSet::new()).unwrap_err();
        assert!(error.to_lowercase().contains("bogus"), "{error}");
    }

    #[test]
    fn rejects_an_unknown_aggregate_metric_name() {
        let path = write_temp(
            "agg-unknown",
            "{\"taguru_evaluate_thresholds\":1,\"aggregate\":{\"not.a.metric\":{\"min\":0.1}}}",
        );
        let error = load_thresholds(&path, &build_definitions(), &BTreeSet::new()).unwrap_err();
        assert!(error.contains("not.a.metric"), "{error}");
    }

    #[test]
    fn rejects_a_non_case_scoped_metric_in_cases_default() {
        let path = write_temp(
            "case-default-bad",
            "{\"taguru_evaluate_thresholds\":1,\
             \"cases\":{\"default\":{\"latency.resolve_ms\":{\"max\":1.0}}}}",
        );
        let error = load_thresholds(&path, &build_definitions(), &BTreeSet::new()).unwrap_err();
        assert!(error.contains("latency.resolve_ms"), "{error}");
    }

    #[test]
    fn rejects_an_unknown_case_id_in_overrides() {
        let path = write_temp(
            "case-id-bad",
            "{\"taguru_evaluate_thresholds\":1,\
             \"cases\":{\"overrides\":{\"ghost\":{\"recall.recall_at_k\":{\"min\":0.1}}}}}",
        );
        let case_ids: BTreeSet<String> = ["real-case".to_string()].into_iter().collect();
        let error = load_thresholds(&path, &build_definitions(), &case_ids).unwrap_err();
        assert!(error.contains("ghost"), "{error}");
    }

    #[test]
    fn rejects_a_bound_with_neither_min_nor_max() {
        let path = write_temp(
            "empty-bound",
            "{\"taguru_evaluate_thresholds\":1,\"aggregate\":{\"recall.recall_at_k\":{}}}",
        );
        let error = load_thresholds(&path, &build_definitions(), &BTreeSet::new()).unwrap_err();
        assert!(error.contains("min/max"), "{error}");
    }

    #[test]
    fn rejects_min_greater_than_max() {
        let path = write_temp(
            "inverted",
            "{\"taguru_evaluate_thresholds\":1,\
             \"aggregate\":{\"recall.recall_at_k\":{\"min\":0.9,\"max\":0.1}}}",
        );
        let error = load_thresholds(&path, &build_definitions(), &BTreeSet::new()).unwrap_err();
        assert!(error.contains("<="), "{error}");
    }

    #[test]
    fn overflowing_json_number_is_rejected_at_parse_time() {
        // Not `NaN`/`Infinity` — those are not legal JSON tokens, and
        // serde_json itself refuses a numeric literal that overflows f64
        // rather than silently producing `inf` — so this never reaches
        // [`validate_bound`]'s own finiteness check at all. The exact
        // wording is serde_json's own, so this only asserts that
        // loading fails, not what it says.
        let path = write_temp(
            "overflow",
            "{\"taguru_evaluate_thresholds\":1,\
             \"aggregate\":{\"recall.recall_at_k\":{\"min\":1e400}}}",
        );
        assert!(load_thresholds(&path, &build_definitions(), &BTreeSet::new()).is_err());
    }

    #[test]
    fn validate_bound_rejects_a_non_finite_min_or_max() {
        assert!(
            validate_bound(&Bound {
                min: Some(f64::INFINITY),
                max: None,
            })
            .is_err()
        );
        assert!(
            validate_bound(&Bound {
                min: None,
                max: Some(f64::NAN),
            })
            .is_err()
        );
    }

    #[test]
    fn sha256_hashes_the_raw_file_bytes() {
        let contents = "{\"taguru_evaluate_thresholds\":1}";
        let path = write_temp("sha", contents);
        let loaded = load_thresholds(&path, &build_definitions(), &BTreeSet::new()).unwrap();
        assert_eq!(
            loaded.sha256,
            crate::extract::sha256_hex(contents.as_bytes())
        );
    }

    #[test]
    fn every_problem_in_a_file_is_reported_together_not_just_the_first() {
        let path = write_temp(
            "many-problems",
            "{\"taguru_evaluate_thresholds\":1,\
             \"aggregate\":{\"not.a.metric\":{\"min\":0.1}},\
             \"cases\":{\"overrides\":{\"ghost\":{\"recall.recall_at_k\":{\"min\":0.1}}}}}",
        );
        let error = load_thresholds(&path, &build_definitions(), &BTreeSet::new()).unwrap_err();
        assert!(error.contains("not.a.metric"), "{error}");
        assert!(error.contains("ghost"), "{error}");
    }

    // =============================== evaluate() ===============================

    fn minimal_case(case_id: &str) -> CaseBlock {
        CaseBlock {
            case_id: case_id.to_string(),
            query: "q".to_string(),
            cues: Vec::new(),
            limit: 10,
            passage: PassageOutcome::Failed {
                message: "x".to_string(),
                latency_ms: 5,
            },
            structural: None,
            recall: None,
            coverage: None,
            lane_cross: None,
            citations: None,
            missed: Vec::new(),
            missed_truncated: 0,
        }
    }

    fn loaded(file: ThresholdsFile) -> LoadedThresholds {
        LoadedThresholds {
            sha256: "test-sha".to_string(),
            file,
        }
    }

    fn bounds(entries: &[(&str, Bound)]) -> BTreeMap<String, Bound> {
        entries
            .iter()
            .map(|(name, bound)| (name.to_string(), *bound))
            .collect()
    }

    #[test]
    fn aggregate_compares_a_distribution_by_mean_and_a_ratio_by_value() {
        let mut metrics: MetricsMap = BTreeMap::new();
        metrics.insert(
            "recall.recall_at_k".to_string(),
            MetricValue::Distribution(Distribution::from_samples(vec![0.5, 1.0])), // mean 0.75
        );
        metrics.insert(
            "citations.resolved".to_string(),
            MetricValue::Ratio(ratio_metric(3, 4)), // value 0.75
        );
        let file = ThresholdsFile {
            taguru_evaluate_thresholds: 1,
            aggregate: bounds(&[
                (
                    "recall.recall_at_k",
                    Bound {
                        min: Some(0.75),
                        max: None,
                    },
                ),
                (
                    "citations.resolved",
                    Bound {
                        min: Some(0.75),
                        max: None,
                    },
                ),
            ]),
            cases: CasesThresholds::default(),
            allow_unstable_corpus: false,
        };
        let report = loaded(file).evaluate(&[], &metrics, true);
        assert!(report.passed, "{report:?}");
    }

    #[test]
    fn a_metric_with_no_samples_is_a_violation_with_no_actual_value() {
        let metrics: MetricsMap = BTreeMap::new(); // recall.recall_at_k never inserted (n=0)
        let file = ThresholdsFile {
            taguru_evaluate_thresholds: 1,
            aggregate: bounds(&[(
                "recall.recall_at_k",
                Bound {
                    min: Some(0.5),
                    max: None,
                },
            )]),
            cases: CasesThresholds::default(),
            allow_unstable_corpus: false,
        };
        let report = loaded(file).evaluate(&[], &metrics, true);
        assert!(!report.passed);
        assert_eq!(report.violations.len(), 1);
        assert_eq!(report.violations[0].actual, None);
    }

    #[test]
    fn a_case_that_never_declared_the_expectation_is_skipped_not_violated() {
        let case = minimal_case("c1"); // no recall block at all
        let file = ThresholdsFile {
            taguru_evaluate_thresholds: 1,
            aggregate: BTreeMap::new(),
            cases: CasesThresholds {
                default: bounds(&[(
                    "recall.recall_at_k",
                    Bound {
                        min: Some(0.5),
                        max: None,
                    },
                )]),
                overrides: BTreeMap::new(),
            },
            allow_unstable_corpus: false,
        };
        let report = loaded(file).evaluate(std::slice::from_ref(&case), &BTreeMap::new(), true);
        assert!(report.passed, "{report:?}");
        assert_eq!(report.violations.len(), 0);
        assert_eq!(report.skipped.len(), 1);
    }

    #[test]
    fn an_override_replaces_the_default_bound_for_that_metric_and_case_only() {
        let mut low = minimal_case("low");
        low.recall = Some(RecallBlock {
            recall_at_k: 0.2,
            mrr: 0.2,
            ndcg: 0.2,
            expected_total: 1,
            matched: 0,
        });
        let mut high = minimal_case("high");
        high.recall = Some(RecallBlock {
            recall_at_k: 0.9,
            mrr: 0.9,
            ndcg: 0.9,
            expected_total: 1,
            matched: 1,
        });
        let file = ThresholdsFile {
            taguru_evaluate_thresholds: 1,
            aggregate: BTreeMap::new(),
            cases: CasesThresholds {
                default: bounds(&[(
                    "recall.recall_at_k",
                    Bound {
                        min: Some(0.5),
                        max: None,
                    },
                )]),
                overrides: [(
                    "low".to_string(),
                    bounds(&[(
                        "recall.recall_at_k",
                        Bound {
                            min: Some(0.0),
                            max: None,
                        },
                    )]),
                )]
                .into_iter()
                .collect(),
            },
            allow_unstable_corpus: false,
        };
        let report = loaded(file).evaluate(&[low, high], &BTreeMap::new(), true);
        // 'low' is overridden down to min 0.0 (its 0.2 satisfies it);
        // 'high' still uses the 0.5 default (its 0.9 satisfies it too)
        // — the override must not leak onto the case that did not name it.
        assert!(report.passed, "{report:?}");
    }

    #[test]
    fn an_unstable_corpus_violates_the_gate_by_default() {
        let file = ThresholdsFile {
            taguru_evaluate_thresholds: 1,
            aggregate: BTreeMap::new(),
            cases: CasesThresholds::default(),
            allow_unstable_corpus: false,
        };
        let report = loaded(file).evaluate(&[], &BTreeMap::new(), false);
        assert!(!report.passed);
        assert_eq!(report.violations[0].scope, "corpus");
        assert_eq!(report.violations[0].metric, "corpus.stable");
    }

    #[test]
    fn allow_unstable_corpus_opts_out_of_the_corpus_stability_gate() {
        let file = ThresholdsFile {
            taguru_evaluate_thresholds: 1,
            aggregate: BTreeMap::new(),
            cases: CasesThresholds::default(),
            allow_unstable_corpus: true,
        };
        let report = loaded(file).evaluate(&[], &BTreeMap::new(), false);
        assert!(report.passed, "{report:?}");
    }

    #[test]
    fn a_violation_never_carries_a_url_credential_or_passage_body_text() {
        let case = {
            let mut case = minimal_case("c1");
            // Plant the needles in fields a regression would most
            // plausibly copy into a `Violation` — the query text and
            // the passage lane's own failure message — so this test
            // actually fails if `Violation` starts echoing case fields.
            case.query = "TOP-SECRET query".to_string();
            case.passage = PassageOutcome::Failed {
                message: "GET https://user:Bearer-token@host/search failed".to_string(),
                latency_ms: 5,
            };
            case.citations = Some(CitationsBlock {
                recall: CitationRecallBlock {
                    expected_total: 1,
                    matched: 0,
                    value: 0.0,
                },
                validity: CitationValidityBlock {
                    expected_total: 1,
                    valid: 0,
                    value: 0.0,
                },
                checks: Vec::new(),
            });
            case
        };
        let file = ThresholdsFile {
            taguru_evaluate_thresholds: 1,
            aggregate: BTreeMap::new(),
            cases: CasesThresholds {
                default: bounds(&[(
                    "citations.recall",
                    Bound {
                        min: Some(1.0),
                        max: None,
                    },
                )]),
                overrides: BTreeMap::new(),
            },
            allow_unstable_corpus: false,
        };
        let report = loaded(file).evaluate(std::slice::from_ref(&case), &BTreeMap::new(), true);
        assert!(!report.passed);
        let json = serde_json::to_string(&report).unwrap();
        for needle in ["http://", "https://", "Bearer", "token", "TOP-SECRET"] {
            assert!(!json.contains(needle), "leaked '{needle}' in {json}");
        }
    }

    #[test]
    fn coverage_case_metric_values_read_the_declared_category_only() {
        let mut case = minimal_case("c1");
        case.coverage = Some(CoverageBlock {
            concepts: Some(CoverageCounts {
                expected: 2,
                matched: 1,
                value: 0.5,
            }),
            labels: None,
            associations: None,
        });
        assert_eq!(case_metric_value(&case, "coverage.concepts"), Some(0.5));
        assert_eq!(case_metric_value(&case, "coverage.labels"), None);
        assert_eq!(case_metric_value(&case, "coverage.associations"), None);
    }
}
