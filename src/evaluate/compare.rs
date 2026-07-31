//! `taguru evaluate compare` (issue #277, ADR 0004 §5, §9.2): reads two
//! already-written `evaluation.json` runs and writes `changes.jsonl` —
//! one record per case whose outcome moved, `kind` ∈ `improved |
//! regressed | added | removed`. `unchanged` is counted in the header
//! only, never emitted per case (ADR 0004 §9.2). Verdict vocabulary is
//! legitimate here, unlike inside `benchmark`'s artifacts — this is the
//! one place in the tree whose whole job is to say which run was
//! better.
//!
//! **Read-side types are a separate mirror, not `EvaluationFile`
//! reused.** `EvaluationFile` and its per-case blocks are
//! `Serialize`-only by construction (`HitLocator.lanes: PassageLanes`,
//! `CueResolution.kind: &'static str`, and friends carry no
//! `Deserialize` impl, several irreversibly so). `EvaluationView` below
//! is `compare`'s own read-only projection: every field
//! `#[serde(default)]`, matching ADR 0004 §9.1's range-acceptance
//! posture for `evaluation.json` — a field a newer writer added is read
//! as absent by an older `compare`, never a hard parse error.
//!
//! **Classification only ever looks at quality metrics.**
//! [`thresholds::COMPARISON_METRICS`] is [`thresholds::CASE_SCOPED_METRICS`]
//! minus `latency.passage_ms` — single-run latency is too noisy to
//! drive an improved/regressed verdict. A case with at least one
//! metric that got worse (`head < base`, past a small epsilon) is
//! `regressed`, even if another metric improved in the same case — a
//! quality-gate-first posture, not a net-score one. The record still
//! carries every changed metric's own `sides`, so a reader can see the
//! improved metric too.
//!
//! **Mismatches warn, never refuse** (ADR 0002 §10): a different
//! `context`, `corpus.revision_after`, threshold-file identity
//! (`thresholds.sha256`, persisted in each `evaluation.json` — this
//! module never reads the original thresholds file), or
//! `taguru_evaluation` stamp between `BASE` and `HEAD` all land in
//! `warnings`, printed to stderr and echoed in the header, without
//! changing the exit code.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::evidence::budget::BudgetLimits;
use crate::clock::{iso8601_utc, now_unix_secs};
use crate::config::subcommand_usage_error;
use crate::measure::{Count, Distribution, MetricValue, Ratio};
use crate::registry::ContextRevision;
use crate::storage;

use super::EVALUATION_VERSION;
use super::thresholds::COMPARISON_METRICS;

const CHANGES_VERSION: u64 = 1;
const DEFAULT_OUT: &str = "changes.jsonl";
/// A per-case metric delta at or below this magnitude is `unchanged`,
/// not a floating-point artifact of the same underlying value —
/// `evaluation.json`'s own scores are `f64` ratios in `[0, 1]` or
/// small counts, so this is comfortably below any real change.
const EPSILON: f64 = 1e-9;
/// Regressed cases printed in full in the terminal summary before
/// falling back to "... and N more" (ADR 0004 §11's no-silent-
/// truncation discipline, the same one `print_threshold_summary`
/// already follows).
const SUMMARY_CASE_CAP: usize = 10;

const USAGE: &str = "\
usage: taguru evaluate compare BASE.json HEAD.json [--out FILE]

Reads two evaluation.json runs (ADR 0004 §9.1) and writes changes.jsonl
(ADR 0004 §9.2): one record per case whose outcome moved between BASE
and HEAD — kind improved/regressed/added/removed — classified on
recall@k/MRR/nDCG, concept/label/association coverage, and citation
recall/locator validity (latency is excluded: too noisy run-to-run to
drive a verdict). unchanged cases are counted in the header only. A
mismatched context, corpus revision, threshold-file identity, or
taguru_evaluation stamp between the two files is a warning, never a
refusal. A human-readable terminal summary accompanies the file.

  BASE.json   an evaluation.json from an earlier run
  HEAD.json   an evaluation.json from a later run
  --out FILE  where to write the artifact (changes.jsonl)

EXIT CODES: 0 comparison completed (regardless of regressions found —
the pass/fail gate is `taguru evaluate --thresholds`, not this
subcommand) · 1 changes.jsonl could not be written · 2 usage error, a
file could not be read, or malformed/out-of-range JSON.

Contract and discipline: docs/evaluate.html,
adr/0004-retrieval-citation-quality-gate.md §5, §9.2.
";

pub(super) fn run(args: &[String]) -> i32 {
    let compare_args = match parse_args(args) {
        Ok(compare_args) => compare_args,
        Err(code) => return code,
    };

    let base = match load_report(&compare_args.base) {
        Ok(report) => report,
        Err(message) => {
            eprintln!("taguru: evaluate: BASE {message}");
            return 2;
        }
    };
    let head = match load_report(&compare_args.head) {
        Ok(report) => report,
        Err(message) => {
            eprintln!("taguru: evaluate: HEAD {message}");
            return 2;
        }
    };

    let report = build_report(&base, &head);
    for warning in &report.header.warnings {
        eprintln!("taguru: evaluate: {warning}");
    }

    let text = match render_jsonl(&report) {
        Ok(text) => text,
        Err(message) => {
            eprintln!("taguru: evaluate: {message}");
            return 1;
        }
    };
    if let Err(error) = storage::write_atomic(&compare_args.out, text.as_bytes()) {
        eprintln!(
            "taguru: evaluate: writing {}: {error}",
            compare_args.out.display()
        );
        return 1;
    }

    print_summary(&report, &compare_args.out);
    0
}

// ============================== Arguments ==============================

#[derive(Debug)]
struct CompareArgs {
    base: PathBuf,
    head: PathBuf,
    out: PathBuf,
}

fn parse_args(args: &[String]) -> Result<CompareArgs, i32> {
    let usage = |message: &str| subcommand_usage_error("evaluate", message);
    let mut positional: Vec<&str> = Vec::new();
    let mut out: Option<PathBuf> = None;

    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{USAGE}");
                return Err(0);
            }
            "--out" => match rest.next() {
                Some(path) if out.is_none() => out = Some(PathBuf::from(path)),
                Some(_) => return Err(usage("--out given twice")),
                None => return Err(usage("--out needs a file path")),
            },
            flag if flag.starts_with("--") => {
                return Err(usage(&format!(
                    "unknown flag '{flag}' for evaluate compare"
                )));
            }
            other => positional.push(other),
        }
    }

    if positional.len() != 2 {
        return Err(usage(&format!(
            "evaluate compare takes exactly two positional arguments, BASE.json and \
             HEAD.json (got {})",
            positional.len()
        )));
    }

    Ok(CompareArgs {
        base: PathBuf::from(positional[0]),
        head: PathBuf::from(positional[1]),
        out: out.unwrap_or_else(|| PathBuf::from(DEFAULT_OUT)),
    })
}

// ============================== Reading evaluation.json ==============================

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct EvaluationView {
    taguru_evaluation: u64,
    generated_at: String,
    inputs: InputsView,
    corpus: CorpusView,
    thresholds: Option<ThresholdIdentityView>,
    definitions: BTreeMap<String, DefinitionView>,
    metrics: BTreeMap<String, Value>,
    cases: Vec<CaseView>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct InputsView {
    context: String,
    /// #308 (ADR 0006 §14): `None` when the run carried no
    /// `--max-items`/`--max-bytes`/`--max-tokens` — read (never
    /// written) here so [`mismatch_warnings`] can tell a genuine
    /// equal-budget comparison from one that silently wasn't.
    budget: Option<BudgetLimits>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct CorpusView {
    revision_after: ContextRevision,
    stable: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct ThresholdIdentityView {
    sha256: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct DefinitionView {
    statistic: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct CaseView {
    case_id: String,
    recall: Option<RecallView>,
    coverage: Option<CoverageView>,
    citations: Option<CitationsView>,
    /// #308 (ADR 0006 §14): `CaseBlock.diversity_sources`, read as
    /// `f64` the same way every other `case_metric_value` field is —
    /// an older `evaluation.json` with no such field parses as `None`
    /// (`#[serde(default)]`), never a hard error.
    diversity_sources: Option<f64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct RecallView {
    recall_at_k: Option<f64>,
    mrr: Option<f64>,
    ndcg: Option<f64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct CoverageView {
    concepts: Option<CoverageCountsView>,
    labels: Option<CoverageCountsView>,
    associations: Option<CoverageCountsView>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct CoverageCountsView {
    value: Option<f64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct CitationsView {
    recall: CitationScoreView,
    validity: CitationScoreView,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct CitationScoreView {
    value: Option<f64>,
}

#[derive(Debug)]
struct LoadedReport {
    /// The path exactly as given on the command line — recorded
    /// verbatim in the header, matching `InputsBlock.eval.path`'s own
    /// posture in `evaluate.rs`.
    path: String,
    view: EvaluationView,
}

fn load_report(path: &Path) -> Result<LoadedReport, String> {
    let text =
        fs::read_to_string(path).map_err(|error| format!("reading {}: {error}", path.display()))?;
    let view: EvaluationView = serde_json::from_str(&text)
        .map_err(|error| format!("{}: malformed evaluation.json: {error}", path.display()))?;
    if !(1..=EVALUATION_VERSION).contains(&view.taguru_evaluation) {
        return Err(format!(
            "{}: taguru_evaluation must be within 1..={EVALUATION_VERSION}, got {}",
            path.display(),
            view.taguru_evaluation
        ));
    }
    Ok(LoadedReport {
        path: path.display().to_string(),
        view,
    })
}

/// One [`COMPARISON_METRICS`] name's value for one case — `None` when
/// the case never computed it (the matching expectation was not
/// declared, or its lane did not complete). Mirrors
/// `thresholds::case_metric_value`'s field paths, over the read-only
/// view types instead of `CaseBlock`.
fn case_metric_value(case: &CaseView, metric: &str) -> Option<f64> {
    match metric {
        "recall.recall_at_k" => case.recall.as_ref()?.recall_at_k,
        "recall.mrr" => case.recall.as_ref()?.mrr,
        "recall.ndcg" => case.recall.as_ref()?.ndcg,
        "coverage.concepts" => case.coverage.as_ref()?.concepts.as_ref()?.value,
        "coverage.labels" => case.coverage.as_ref()?.labels.as_ref()?.value,
        "coverage.associations" => case.coverage.as_ref()?.associations.as_ref()?.value,
        "citations.recall" => case.citations.as_ref()?.recall.value,
        "citations.locator_validity" => case.citations.as_ref()?.validity.value,
        "diversity.sources" => case.diversity_sources,
        _ => None,
    }
}

// ============================== Classification ==============================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Improved,
    Regressed,
}

/// One case present in both runs. `None` means unchanged — no
/// `changes.jsonl` record, only a header count.
fn classify_matched(
    case_id: &str,
    base: &CaseView,
    head: &CaseView,
    warnings: &mut Vec<String>,
) -> Option<(Verdict, Vec<MetricDiffRecord>)> {
    let mut changed = Vec::new();
    let mut any_regressed = false;
    let mut any_improved = false;

    for &metric in COMPARISON_METRICS {
        match (
            case_metric_value(base, metric),
            case_metric_value(head, metric),
        ) {
            (Some(base_value), Some(head_value)) => {
                let delta = head_value - base_value;
                if delta > EPSILON {
                    any_improved = true;
                    changed.push(MetricDiffRecord {
                        metric: metric.to_string(),
                        sides: Sides {
                            base: Some(base_value),
                            head: Some(head_value),
                        },
                        delta: Some(delta),
                        direction: "improved",
                    });
                } else if delta < -EPSILON {
                    any_regressed = true;
                    changed.push(MetricDiffRecord {
                        metric: metric.to_string(),
                        sides: Sides {
                            base: Some(base_value),
                            head: Some(head_value),
                        },
                        delta: Some(delta),
                        direction: "regressed",
                    });
                }
            }
            (Some(_), None) | (None, Some(_)) => {
                warnings.push(format!(
                    "case '{case_id}': metric '{metric}' has a value on only one side — \
                     excluded from classification"
                ));
            }
            (None, None) => {}
        }
    }

    if any_regressed {
        Some((Verdict::Regressed, changed))
    } else if any_improved {
        Some((Verdict::Improved, changed))
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy)]
enum OneSide {
    Base,
    Head,
}

/// An `added`/`removed` case's own metric readout: every
/// [`COMPARISON_METRICS`] entry it has a value for, `direction:
/// "not_comparable"` since there is no other side to diff against.
fn one_sided_metrics(case: &CaseView, side: OneSide) -> Vec<MetricDiffRecord> {
    COMPARISON_METRICS
        .iter()
        .filter_map(|&metric| {
            let value = case_metric_value(case, metric)?;
            let sides = match side {
                OneSide::Base => Sides {
                    base: Some(value),
                    head: None,
                },
                OneSide::Head => Sides {
                    base: None,
                    head: Some(value),
                },
            };
            Some(MetricDiffRecord {
                metric: metric.to_string(),
                sides,
                delta: None,
                direction: "not_comparable",
            })
        })
        .collect()
}

fn index_cases<'a>(
    cases: &'a [CaseView],
    label: &str,
    warnings: &mut Vec<String>,
) -> BTreeMap<String, &'a CaseView> {
    let mut map = BTreeMap::new();
    for case in cases {
        if map.insert(case.case_id.clone(), case).is_some() {
            warnings.push(format!(
                "{label}: duplicate case_id '{}' — using the last occurrence",
                case.case_id
            ));
        }
    }
    map
}

// ============================== Aggregate metric diffs ==============================

/// One `metrics[name]` value, read through `definitions[name]
/// .statistic` first — never through `MetricValue`'s own
/// `#[serde(untagged)]`, which is unsound to deserialize through
/// directly (ADR 0004 §9.1: a `Ratio` payload can silently parse as a
/// `Distribution`, since every `Distribution` field but `n` is
/// optional). `Ok(None)` when the metric simply is not present in this
/// run; `Err` — turned into a warning by the caller, never fatal — when
/// it is present but its shape does not match its own declared
/// statistic, or it has no `definitions` entry at all.
fn resolve_aggregate_scalar(view: &EvaluationView, metric: &str) -> Result<Option<f64>, String> {
    let Some(raw) = view.metrics.get(metric) else {
        return Ok(None);
    };
    let Some(definition) = view.definitions.get(metric) else {
        return Err(format!(
            "metric '{metric}' has no definitions entry — skipped from the aggregate diff"
        ));
    };
    let value = match definition.statistic.as_str() {
        "distribution" => {
            serde_json::from_value::<Distribution>(raw.clone()).map(MetricValue::Distribution)
        }
        "ratio" => serde_json::from_value::<Ratio>(raw.clone()).map(MetricValue::Ratio),
        "count" => serde_json::from_value::<Count>(raw.clone()).map(MetricValue::Count),
        other => {
            return Err(format!(
                "metric '{metric}': definitions names unknown statistic '{other}'"
            ));
        }
    }
    .map_err(|error| format!("metric '{metric}': {error}"))?;
    Ok(value.scalar())
}

fn aggregate_diffs(
    base: &EvaluationView,
    head: &EvaluationView,
    warnings: &mut Vec<String>,
) -> Vec<AggregateDiffRecord> {
    let mut names: BTreeSet<String> = BTreeSet::new();
    names.extend(base.metrics.keys().cloned());
    names.extend(head.metrics.keys().cloned());

    let mut diffs = Vec::new();
    for name in names {
        let base_value = resolve_aggregate_scalar(base, &name).unwrap_or_else(|message| {
            warnings.push(format!("BASE: {message}"));
            None
        });
        let head_value = resolve_aggregate_scalar(head, &name).unwrap_or_else(|message| {
            warnings.push(format!("HEAD: {message}"));
            None
        });
        if base_value.is_none() && head_value.is_none() {
            continue;
        }
        let delta = match (base_value, head_value) {
            (Some(base_value), Some(head_value)) => Some(head_value - base_value),
            _ => None,
        };
        let statistic = head
            .definitions
            .get(&name)
            .or_else(|| base.definitions.get(&name))
            .map(|definition| definition.statistic.clone());
        diffs.push(AggregateDiffRecord {
            metric: name,
            statistic,
            sides: Sides {
                base: base_value,
                head: head_value,
            },
            delta,
        });
    }
    diffs
}

// ============================== Building the report ==============================

struct ChangesReport {
    header: ChangesHeader,
    records: Vec<ChangeRecord>,
}

fn build_report(base: &LoadedReport, head: &LoadedReport) -> ChangesReport {
    let mut warnings = Vec::new();
    mismatch_warnings(base, head, &mut warnings);

    let base_cases = index_cases(&base.view.cases, "BASE", &mut warnings);
    let head_cases = index_cases(&head.view.cases, "HEAD", &mut warnings);

    let mut ids: BTreeSet<&String> = BTreeSet::new();
    ids.extend(base_cases.keys());
    ids.extend(head_cases.keys());

    let mut records = Vec::new();
    let mut improved = 0usize;
    let mut regressed = 0usize;
    let mut added = 0usize;
    let mut removed = 0usize;
    let mut unchanged = 0usize;

    for case_id in ids {
        match (base_cases.get(case_id), head_cases.get(case_id)) {
            (Some(base_case), Some(head_case)) => {
                match classify_matched(case_id, base_case, head_case, &mut warnings) {
                    Some((Verdict::Improved, metrics)) => {
                        improved += 1;
                        records.push(ChangeRecord::Improved {
                            case_id: case_id.clone(),
                            metrics,
                        });
                    }
                    Some((Verdict::Regressed, metrics)) => {
                        regressed += 1;
                        records.push(ChangeRecord::Regressed {
                            case_id: case_id.clone(),
                            metrics,
                        });
                    }
                    None => unchanged += 1,
                }
            }
            (Some(base_case), None) => {
                removed += 1;
                records.push(ChangeRecord::Removed {
                    case_id: case_id.clone(),
                    metrics: one_sided_metrics(base_case, OneSide::Base),
                });
            }
            (None, Some(head_case)) => {
                added += 1;
                records.push(ChangeRecord::Added {
                    case_id: case_id.clone(),
                    metrics: one_sided_metrics(head_case, OneSide::Head),
                });
            }
            (None, None) => unreachable!("case_id came from the union of both key sets"),
        }
    }

    let metrics = aggregate_diffs(&base.view, &head.view, &mut warnings);

    let header = ChangesHeader {
        kind: "header",
        taguru_evaluation_changes: CHANGES_VERSION,
        generated_at: iso8601_utc(now_unix_secs()),
        base: report_identity(base),
        head: report_identity(head),
        counts: Counts {
            improved,
            regressed,
            added,
            removed,
            unchanged,
            cases_base: base_cases.len(),
            cases_head: head_cases.len(),
        },
        metrics,
        warnings,
    };

    ChangesReport { header, records }
}

fn report_identity(loaded: &LoadedReport) -> ReportIdentity {
    ReportIdentity {
        path: loaded.path.clone(),
        context: loaded.view.inputs.context.clone(),
        taguru_evaluation: loaded.view.taguru_evaluation,
        generated_at: loaded.view.generated_at.clone(),
        corpus_revision_after: loaded.view.corpus.revision_after,
        corpus_stable: loaded.view.corpus.stable,
        thresholds_sha256: loaded.view.thresholds.as_ref().map(|t| t.sha256.clone()),
    }
}

/// ADR 0004 §9.2's four named mismatches, plus a corpus that was
/// already unstable within one of the two runs on its own. None of
/// these change the exit code (ADR 0002 §10: a warning never blocks).
fn mismatch_warnings(base: &LoadedReport, head: &LoadedReport, warnings: &mut Vec<String>) {
    if base.view.inputs.context != head.view.inputs.context {
        warnings.push(format!(
            "context differs: BASE '{}' vs HEAD '{}'",
            base.view.inputs.context, head.view.inputs.context
        ));
    }
    if base.view.corpus.revision_after != head.view.corpus.revision_after {
        warnings.push("corpus.revision_after differs between BASE and HEAD".to_string());
    }
    if !base.view.corpus.stable {
        warnings.push(format!(
            "{}: corpus.stable is false — a write landed mid-run",
            base.path
        ));
    }
    if !head.view.corpus.stable {
        warnings.push(format!(
            "{}: corpus.stable is false — a write landed mid-run",
            head.path
        ));
    }
    let base_sha = base.view.thresholds.as_ref().map(|t| t.sha256.as_str());
    let head_sha = head.view.thresholds.as_ref().map(|t| t.sha256.as_str());
    if base_sha != head_sha {
        warnings.push(format!(
            "thresholds file identity differs: BASE {} vs HEAD {}",
            base_sha.map_or("none", |sha| &sha[..12.min(sha.len())]),
            head_sha.map_or("none", |sha| &sha[..12.min(sha.len())]),
        ));
    }
    if base.view.taguru_evaluation != head.view.taguru_evaluation {
        warnings.push(format!(
            "taguru_evaluation stamp differs: BASE {} vs HEAD {}",
            base.view.taguru_evaluation, head.view.taguru_evaluation
        ));
    }
    // #308 (ADR 0006 §14): comparing two runs whose --max-items/
    // --max-bytes/--max-tokens differ (including one side having no
    // budget flag at all) is exactly the dishonest comparison an
    // equal-budget evaluation exists to catch — warned, never refused,
    // matching every other mismatch above (ADR 0002 §10).
    let base_budget = base.view.inputs.budget;
    let head_budget = head.view.inputs.budget;
    if budget_key(base_budget) != budget_key(head_budget) {
        warnings.push(format!(
            "budget differs: BASE {} vs HEAD {} — not an equal-budget comparison",
            describe_budget(base_budget),
            describe_budget(head_budget),
        ));
    }
}

/// A `PartialEq`-able projection of [`BudgetLimits`] — that type
/// itself derives no `PartialEq` (it is #304/#305's wire type, not
/// this module's to extend), so [`mismatch_warnings`] compares this
/// instead.
fn budget_key(limits: Option<BudgetLimits>) -> Option<(usize, usize, usize)> {
    limits.map(|limits| (limits.max_items, limits.max_bytes, limits.max_tokens))
}

fn describe_budget(limits: Option<BudgetLimits>) -> String {
    match limits {
        Some(limits) => format!(
            "max_items={} max_bytes={} max_tokens={}",
            limits.max_items, limits.max_bytes, limits.max_tokens
        ),
        None => "none".to_string(),
    }
}

// ============================== Output shapes ==============================

/// A metric's value on each side of the comparison — `None` when that
/// side has no matching value, never a count of absence, matching
/// `benchmark::compare::differences::Sides<T>`'s same posture. Field
/// names are `base`/`head` here rather than that type's `a`/`b`: this
/// is the one artifact in the tree whose whole job is to say which run
/// was better, so naming a side by its role is legitimate (ADR 0004
/// §9.2).
#[derive(Debug, Clone, Copy, Serialize)]
struct Sides<T> {
    base: Option<T>,
    head: Option<T>,
}

#[derive(Debug, Clone, Serialize)]
struct MetricDiffRecord {
    metric: String,
    sides: Sides<f64>,
    delta: Option<f64>,
    direction: &'static str,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ChangeRecord {
    Improved {
        case_id: String,
        metrics: Vec<MetricDiffRecord>,
    },
    Regressed {
        case_id: String,
        metrics: Vec<MetricDiffRecord>,
    },
    Added {
        case_id: String,
        metrics: Vec<MetricDiffRecord>,
    },
    Removed {
        case_id: String,
        metrics: Vec<MetricDiffRecord>,
    },
}

#[derive(Debug, Clone, Serialize)]
struct ReportIdentity {
    path: String,
    context: String,
    taguru_evaluation: u64,
    generated_at: String,
    corpus_revision_after: ContextRevision,
    corpus_stable: bool,
    thresholds_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct Counts {
    improved: usize,
    regressed: usize,
    added: usize,
    removed: usize,
    unchanged: usize,
    cases_base: usize,
    cases_head: usize,
}

#[derive(Debug, Clone, Serialize)]
struct AggregateDiffRecord {
    metric: String,
    statistic: Option<String>,
    sides: Sides<f64>,
    delta: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct ChangesHeader {
    kind: &'static str,
    taguru_evaluation_changes: u64,
    generated_at: String,
    base: ReportIdentity,
    head: ReportIdentity,
    counts: Counts,
    metrics: Vec<AggregateDiffRecord>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<String>,
}

fn render_jsonl(report: &ChangesReport) -> Result<String, String> {
    let mut lines: Vec<String> = Vec::with_capacity(report.records.len() + 1);
    lines.push(
        serde_json::to_string(&report.header)
            .map_err(|error| format!("serializing changes header: {error}"))?,
    );
    for record in &report.records {
        lines.push(
            serde_json::to_string(record)
                .map_err(|error| format!("serializing a change record: {error}"))?,
        );
    }
    let mut text = lines.join("\n");
    text.push('\n');
    Ok(text)
}

// ============================== Terminal summary ==============================

fn print_summary(report: &ChangesReport, out: &Path) {
    let counts = &report.header.counts;
    println!(
        "evaluate compare: {} case(s) in BASE, {} in HEAD — {} improved, {} regressed, \
         {} added, {} removed, {} unchanged",
        counts.cases_base,
        counts.cases_head,
        counts.improved,
        counts.regressed,
        counts.added,
        counts.removed,
        counts.unchanged
    );

    if !report.header.metrics.is_empty() {
        println!("  aggregate:");
        for entry in &report.header.metrics {
            match (entry.sides.base, entry.sides.head) {
                (Some(base_value), Some(head_value)) => println!(
                    "    {}: {base_value:.3} -> {head_value:.3} (Δ {:+.3})",
                    entry.metric,
                    entry.delta.unwrap_or(0.0)
                ),
                (Some(base_value), None) => {
                    println!("    {}: {base_value:.3} -> n/a", entry.metric)
                }
                (None, Some(head_value)) => {
                    println!("    {}: n/a -> {head_value:.3}", entry.metric)
                }
                (None, None) => {}
            }
        }
    }

    let regressed: Vec<&ChangeRecord> = report
        .records
        .iter()
        .filter(|record| matches!(record, ChangeRecord::Regressed { .. }))
        .collect();
    for record in regressed.iter().take(SUMMARY_CASE_CAP) {
        if let ChangeRecord::Regressed { case_id, metrics } = record {
            let detail: Vec<String> = metrics
                .iter()
                .filter(|metric| metric.direction == "regressed")
                .map(|metric| format!("{} {:+.3}", metric.metric, metric.delta.unwrap_or(0.0)))
                .collect();
            println!("    regressed: {case_id} ({})", detail.join(", "));
        }
    }
    if regressed.len() > SUMMARY_CASE_CAP {
        println!(
            "    ... and {} more regressed case(s)",
            regressed.len() - SUMMARY_CASE_CAP
        );
    }

    println!("  wrote {}", out.display());
}

#[cfg(test)]
mod tests;
