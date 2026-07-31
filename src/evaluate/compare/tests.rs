use std::collections::BTreeSet;

use serde_json::Value;

use super::*;

// ============================== Argument parsing ==============================

fn args(words: &[&str]) -> Result<CompareArgs, i32> {
    parse_args(&words.iter().map(|s| s.to_string()).collect::<Vec<_>>())
}

#[test]
fn missing_positionals_is_a_usage_error() {
    assert_eq!(args(&[]).unwrap_err(), 2);
}

#[test]
fn one_positional_is_a_usage_error() {
    assert_eq!(args(&["base.json"]).unwrap_err(), 2);
}

#[test]
fn three_positionals_is_a_usage_error() {
    assert_eq!(args(&["a.json", "b.json", "c.json"]).unwrap_err(), 2);
}

#[test]
fn out_given_twice_is_a_usage_error() {
    assert_eq!(
        args(&["a.json", "b.json", "--out", "x.jsonl", "--out", "y.jsonl"]).unwrap_err(),
        2
    );
}

#[test]
fn out_without_a_value_is_a_usage_error() {
    assert_eq!(args(&["a.json", "b.json", "--out"]).unwrap_err(), 2);
}

#[test]
fn an_unknown_flag_is_refused() {
    assert_eq!(args(&["a.json", "b.json", "--bogus"]).unwrap_err(), 2);
}

#[test]
fn out_defaults_to_changes_jsonl() {
    let parsed = args(&["base.json", "head.json"]).unwrap();
    assert_eq!(parsed.base, PathBuf::from("base.json"));
    assert_eq!(parsed.head, PathBuf::from("head.json"));
    assert_eq!(parsed.out, PathBuf::from("changes.jsonl"));
}

#[test]
fn out_is_captured_when_given() {
    let parsed = args(&["base.json", "head.json", "--out", "custom.jsonl"]).unwrap();
    assert_eq!(parsed.out, PathBuf::from("custom.jsonl"));
}

#[test]
fn help_exits_zero_without_requiring_positionals() {
    assert_eq!(args(&["--help"]).unwrap_err(), 0);
}

// ============================== Test fixtures ==============================

fn base_case(id: &str) -> CaseView {
    CaseView {
        case_id: id.to_string(),
        recall: None,
        coverage: None,
        citations: None,
        diversity_sources: None,
    }
}

fn with_recall(mut case: CaseView, recall_at_k: f64, mrr: f64, ndcg: f64) -> CaseView {
    case.recall = Some(RecallView {
        recall_at_k: Some(recall_at_k),
        mrr: Some(mrr),
        ndcg: Some(ndcg),
    });
    case
}

fn with_citation_recall(mut case: CaseView, value: f64) -> CaseView {
    case.citations = Some(CitationsView {
        recall: CitationScoreView { value: Some(value) },
        validity: CitationScoreView { value: Some(1.0) },
    });
    case
}

fn evaluation_view(context: &str, taguru_evaluation: u64, cases: Vec<CaseView>) -> EvaluationView {
    EvaluationView {
        taguru_evaluation,
        generated_at: "2026-07-29T00:00:00Z".to_string(),
        inputs: InputsView {
            context: context.to_string(),
            budget: None,
        },
        corpus: CorpusView {
            revision_after: ContextRevision::default(),
            stable: true,
        },
        thresholds: None,
        definitions: BTreeMap::new(),
        metrics: BTreeMap::new(),
        cases,
    }
}

fn loaded(path: &str, view: EvaluationView) -> LoadedReport {
    LoadedReport {
        path: path.to_string(),
        view,
    }
}

// ============================== classify_matched ==============================

#[test]
fn a_single_metric_increase_is_improved() {
    let base = with_recall(base_case("c1"), 0.5, 0.5, 0.5);
    let head = with_recall(base_case("c1"), 0.8, 0.5, 0.5);
    let mut warnings = Vec::new();
    let (verdict, metrics) = classify_matched("c1", &base, &head, &mut warnings).unwrap();
    assert_eq!(verdict, Verdict::Improved);
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].metric, "recall.recall_at_k");
    assert_eq!(metrics[0].direction, "improved");
    assert!((metrics[0].delta.unwrap() - 0.3).abs() < 1e-9);
    assert!(warnings.is_empty());
}

#[test]
fn a_single_metric_decrease_is_regressed() {
    let base = with_recall(base_case("c1"), 0.8, 0.5, 0.5);
    let head = with_recall(base_case("c1"), 0.5, 0.5, 0.5);
    let mut warnings = Vec::new();
    let (verdict, metrics) = classify_matched("c1", &base, &head, &mut warnings).unwrap();
    assert_eq!(verdict, Verdict::Regressed);
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0].direction, "regressed");
}

#[test]
fn a_mixed_case_with_any_regression_is_regressed_and_keeps_both_directions() {
    let base = with_citation_recall(with_recall(base_case("c1"), 0.5, 0.5, 0.5), 1.0);
    let mut head = with_recall(base_case("c1"), 0.75, 0.5, 0.5);
    head.citations = Some(CitationsView {
        recall: CitationScoreView { value: Some(0.5) },
        validity: CitationScoreView { value: Some(1.0) },
    });
    let mut warnings = Vec::new();
    let (verdict, metrics) = classify_matched("c1", &base, &head, &mut warnings).unwrap();
    assert_eq!(verdict, Verdict::Regressed);
    let directions: BTreeSet<&str> = metrics.iter().map(|m| m.direction).collect();
    assert!(directions.contains("improved"));
    assert!(directions.contains("regressed"));
    assert_eq!(metrics.len(), 2);
}

#[test]
fn identical_values_are_unchanged() {
    let base = with_recall(base_case("c1"), 0.6, 0.6, 0.6);
    let head = with_recall(base_case("c1"), 0.6, 0.6, 0.6);
    let mut warnings = Vec::new();
    assert!(classify_matched("c1", &base, &head, &mut warnings).is_none());
    assert!(warnings.is_empty());
}

#[test]
fn a_sub_epsilon_delta_is_unchanged() {
    let base = with_recall(base_case("c1"), 0.6, 0.6, 0.6);
    let head = with_recall(base_case("c1"), 0.6 + 1e-12, 0.6, 0.6);
    let mut warnings = Vec::new();
    assert!(classify_matched("c1", &base, &head, &mut warnings).is_none());
}

#[test]
fn a_metric_present_on_only_one_side_does_not_drive_classification_but_warns() {
    let base = base_case("c1");
    let head = with_recall(base_case("c1"), 0.9, 0.9, 0.9);
    let mut warnings = Vec::new();
    let verdict = classify_matched("c1", &base, &head, &mut warnings);
    assert!(verdict.is_none());
    assert_eq!(warnings.len(), 3);
    assert!(warnings.iter().all(|w| w.contains("only one side")));
}

// ============================== index_cases ==============================

#[test]
fn duplicate_case_id_within_one_side_warns_and_keeps_the_last() {
    let cases = vec![
        base_case("dup"),
        with_recall(base_case("dup"), 0.9, 0.9, 0.9),
    ];
    let mut warnings = Vec::new();
    let indexed = index_cases(&cases, "BASE", &mut warnings);
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("duplicate case_id 'dup'"));
    assert!(indexed["dup"].recall.is_some());
}

// ============================== aggregate diffs ==============================

fn distribution_value(mean: f64, n: u64) -> Value {
    serde_json::json!({
        "n": n, "min": null, "p50": null, "p90": null, "p99": null,
        "max": null, "mean": mean, "sum": null
    })
}

fn ratio_value(value: f64, n: u64, numerator: u64) -> Value {
    serde_json::json!({"value": value, "n": n, "numerator": numerator})
}

#[test]
fn a_ratio_metric_is_read_as_a_ratio_not_misread_as_a_distribution() {
    // ADR 0004 §9.1's untagged-deserialize trap: a Ratio payload's only
    // required field (n) also satisfies Distribution, so reading
    // through `definitions.statistic` first — not MetricValue's own
    // untagged enum — must still recover Ratio's own `value`, not a
    // `Distribution` silently parsed with `mean: None`.
    let mut view = evaluation_view("ctx", 1, vec![]);
    view.definitions.insert(
        "citations.recall".to_string(),
        DefinitionView {
            statistic: "ratio".to_string(),
        },
    );
    view.metrics
        .insert("citations.recall".to_string(), ratio_value(0.0968, 31, 3));
    let scalar = resolve_aggregate_scalar(&view, "citations.recall").unwrap();
    assert!((scalar.unwrap() - 0.0968).abs() < 1e-9);
}

#[test]
fn a_metric_without_a_definitions_entry_warns_and_is_skipped() {
    let mut base = evaluation_view("ctx", 1, vec![]);
    base.metrics
        .insert("mystery.metric".to_string(), ratio_value(1.0, 1, 1));
    let head = evaluation_view("ctx", 1, vec![]);
    let mut warnings = Vec::new();
    let diffs = aggregate_diffs(&base, &head, &mut warnings);
    assert!(diffs.is_empty());
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("mystery.metric"));
    assert!(warnings[0].contains("no definitions entry"));
}

#[test]
fn an_unknown_statistic_name_warns_and_is_skipped() {
    let mut base = evaluation_view("ctx", 1, vec![]);
    base.definitions.insert(
        "weird".to_string(),
        DefinitionView {
            statistic: "histogram".to_string(),
        },
    );
    base.metrics
        .insert("weird".to_string(), serde_json::json!({"n": 1}));
    let head = evaluation_view("ctx", 1, vec![]);
    let mut warnings = Vec::new();
    let diffs = aggregate_diffs(&base, &head, &mut warnings);
    assert!(diffs.is_empty());
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("unknown statistic"));
}

#[test]
fn aggregate_delta_is_head_minus_base() {
    let mut base = evaluation_view("ctx", 1, vec![]);
    base.definitions.insert(
        "recall.recall_at_k".to_string(),
        DefinitionView {
            statistic: "distribution".to_string(),
        },
    );
    base.metrics
        .insert("recall.recall_at_k".to_string(), distribution_value(0.6, 5));
    let mut head = evaluation_view("ctx", 1, vec![]);
    head.definitions.insert(
        "recall.recall_at_k".to_string(),
        DefinitionView {
            statistic: "distribution".to_string(),
        },
    );
    head.metrics
        .insert("recall.recall_at_k".to_string(), distribution_value(0.8, 5));
    let mut warnings = Vec::new();
    let diffs = aggregate_diffs(&base, &head, &mut warnings);
    assert_eq!(diffs.len(), 1);
    assert!((diffs[0].delta.unwrap() - 0.2).abs() < 1e-9);
    assert!(warnings.is_empty());
}

// ============================== mismatch_warnings ==============================

#[test]
fn a_context_mismatch_warns() {
    let base = loaded("base.json", evaluation_view("ctx-a", 1, vec![]));
    let head = loaded("head.json", evaluation_view("ctx-b", 1, vec![]));
    let mut warnings = Vec::new();
    mismatch_warnings(&base, &head, &mut warnings);
    assert!(warnings.iter().any(|w| w.contains("context differs")));
}

#[test]
fn a_revision_after_mismatch_warns() {
    let mut base_view = evaluation_view("ctx", 1, vec![]);
    base_view.corpus.revision_after = ContextRevision {
        graph: 1,
        passages: 1,
        config: 1,
    };
    let mut head_view = evaluation_view("ctx", 1, vec![]);
    head_view.corpus.revision_after = ContextRevision {
        graph: 2,
        passages: 1,
        config: 1,
    };
    let base = loaded("base.json", base_view);
    let head = loaded("head.json", head_view);
    let mut warnings = Vec::new();
    mismatch_warnings(&base, &head, &mut warnings);
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("corpus.revision_after differs"))
    );
}

#[test]
fn an_unstable_corpus_on_either_side_warns() {
    let mut base_view = evaluation_view("ctx", 1, vec![]);
    base_view.corpus.stable = false;
    let base = loaded("base.json", base_view);
    let head = loaded("head.json", evaluation_view("ctx", 1, vec![]));
    let mut warnings = Vec::new();
    mismatch_warnings(&base, &head, &mut warnings);
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("corpus.stable is false"))
    );
}

#[test]
fn a_thresholds_sha256_mismatch_including_one_null_side_warns() {
    let mut head_view = evaluation_view("ctx", 1, vec![]);
    head_view.thresholds = Some(ThresholdIdentityView {
        sha256: "abc123".to_string(),
    });
    let base = loaded("base.json", evaluation_view("ctx", 1, vec![]));
    let head = loaded("head.json", head_view);
    let mut warnings = Vec::new();
    mismatch_warnings(&base, &head, &mut warnings);
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("thresholds file identity differs"))
    );
}

#[test]
fn a_taguru_evaluation_stamp_mismatch_warns() {
    let base = loaded("base.json", evaluation_view("ctx", 1, vec![]));
    let head = loaded("head.json", evaluation_view("ctx", 2, vec![]));
    let mut warnings = Vec::new();
    mismatch_warnings(&base, &head, &mut warnings);
    assert!(
        warnings
            .iter()
            .any(|w| w.contains("taguru_evaluation stamp differs"))
    );
}

#[test]
fn a_budget_mismatch_including_one_unbudgeted_side_warns() {
    let mut head_view = evaluation_view("ctx", 1, vec![]);
    head_view.inputs.budget = Some(BudgetLimits {
        max_items: 40,
        max_bytes: 65536,
        max_tokens: 4000,
    });
    let base = loaded("base.json", evaluation_view("ctx", 1, vec![]));
    let head = loaded("head.json", head_view);
    let mut warnings = Vec::new();
    mismatch_warnings(&base, &head, &mut warnings);
    assert!(
        warnings.iter().any(|w| w.contains("budget differs")),
        "{warnings:?}"
    );
}

#[test]
fn a_baseline_vs_assembly_pair_under_the_same_budget_does_not_warn() {
    let budget = Some(BudgetLimits {
        max_items: 40,
        max_bytes: 65536,
        max_tokens: 4000,
    });
    let mut base_view = evaluation_view("ctx", 1, vec![]);
    base_view.inputs.budget = budget;
    let mut head_view = evaluation_view("ctx", 1, vec![]);
    head_view.inputs.budget = budget;
    let base = loaded("base.json", base_view);
    let head = loaded("head.json", head_view);
    let mut warnings = Vec::new();
    mismatch_warnings(&base, &head, &mut warnings);
    assert!(
        !warnings.iter().any(|w| w.contains("budget")),
        "{warnings:?}"
    );
}

// ============================== build_report / render_jsonl ==============================

#[test]
fn a_case_only_in_head_is_added_and_only_in_base_is_removed() {
    let base = loaded(
        "base.json",
        evaluation_view("ctx", 1, vec![base_case("only-base")]),
    );
    let head = loaded(
        "head.json",
        evaluation_view(
            "ctx",
            1,
            vec![with_recall(base_case("only-head"), 0.5, 0.5, 0.5)],
        ),
    );
    let report = build_report(&base, &head);
    assert_eq!(report.header.counts.added, 1);
    assert_eq!(report.header.counts.removed, 1);
    assert_eq!(report.records.len(), 2);
}

#[test]
fn an_added_case_lists_its_available_metrics_as_not_comparable() {
    let base = loaded("base.json", evaluation_view("ctx", 1, vec![]));
    let head_case = with_recall(base_case("new-case"), 0.7, 0.7, 0.7);
    let head = loaded("head.json", evaluation_view("ctx", 1, vec![head_case]));
    let report = build_report(&base, &head);
    assert_eq!(report.records.len(), 1);
    match &report.records[0] {
        ChangeRecord::Added { case_id, metrics } => {
            assert_eq!(case_id, "new-case");
            assert_eq!(metrics.len(), 3);
            for metric in metrics {
                assert_eq!(metric.direction, "not_comparable");
                assert!(metric.delta.is_none());
                assert!(metric.sides.base.is_none());
                assert!(metric.sides.head.is_some());
            }
        }
        other => panic!("expected Added, got {other:?}"),
    }
}

#[test]
fn identical_case_across_both_runs_counts_as_unchanged_and_emits_no_record() {
    let case = with_recall(base_case("same"), 0.5, 0.5, 0.5);
    let base = loaded("base.json", evaluation_view("ctx", 1, vec![case.clone()]));
    let head = loaded("head.json", evaluation_view("ctx", 1, vec![case]));
    let report = build_report(&base, &head);
    assert_eq!(report.header.counts.unchanged, 1);
    assert!(report.records.is_empty());
}

#[test]
fn changes_jsonl_lines_are_each_independently_valid_json_and_the_first_is_the_header() {
    let base = loaded(
        "base.json",
        evaluation_view("ctx", 1, vec![with_recall(base_case("c1"), 0.5, 0.5, 0.5)]),
    );
    let head = loaded(
        "head.json",
        evaluation_view("ctx", 1, vec![with_recall(base_case("c1"), 0.9, 0.9, 0.9)]),
    );
    let report = build_report(&base, &head);
    let text = render_jsonl(&report).unwrap();
    let lines: Vec<&str> = text.trim_end().lines().collect();
    assert_eq!(lines.len(), 2, "header + one improved case");
    let header: Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(header["kind"], "header");
    assert_eq!(header["taguru_evaluation_changes"], 1);
    for line in &lines {
        let _: Value =
            serde_json::from_str(line).expect("every line must be standalone valid JSON");
    }
}

#[test]
fn an_unchanged_case_produces_no_per_case_line() {
    let case = with_recall(base_case("same"), 0.5, 0.5, 0.5);
    let base = loaded("base.json", evaluation_view("ctx", 1, vec![case.clone()]));
    let head = loaded("head.json", evaluation_view("ctx", 1, vec![case]));
    let report = build_report(&base, &head);
    let text = render_jsonl(&report).unwrap();
    let lines: Vec<&str> = text.trim_end().lines().collect();
    assert_eq!(lines.len(), 1, "header only, unchanged case is silent");
    let header: Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(header["counts"]["unchanged"], 1);
}

#[test]
fn changes_jsonl_never_carries_corpus_body_text() {
    let base = loaded(
        "base.json",
        evaluation_view("ctx", 1, vec![with_recall(base_case("c1"), 0.5, 0.5, 0.5)]),
    );
    let head = loaded(
        "head.json",
        evaluation_view("ctx", 1, vec![with_recall(base_case("c1"), 0.9, 0.9, 0.9)]),
    );
    let report = build_report(&base, &head);
    let text = render_jsonl(&report).unwrap();
    for needle in ["\"text\"", "\"quote\"", "\"section\"", "\"hits\""] {
        assert!(
            !text.contains(needle),
            "found {needle} in changes.jsonl output"
        );
    }
}

// ============================== load_report ==============================

fn write_temp(tag: &str, contents: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "taguru-evaluate-compare-unit-{tag}-{}.json",
        std::process::id()
    ));
    fs::write(&path, contents).unwrap();
    path
}

#[test]
fn load_report_rejects_a_missing_file() {
    let error = load_report(Path::new("/does/not/exist/evaluation.json")).unwrap_err();
    assert!(error.contains("reading"));
}

#[test]
fn load_report_rejects_malformed_json() {
    let path = write_temp("malformed", "{not json");
    let error = load_report(&path).unwrap_err();
    assert!(error.contains("malformed evaluation.json"));
}

#[test]
fn load_report_rejects_a_stamp_above_evaluation_version() {
    let path = write_temp(
        "stamp",
        &format!(r#"{{"taguru_evaluation":{}}}"#, EVALUATION_VERSION + 1),
    );
    let error = load_report(&path).unwrap_err();
    assert!(error.contains("taguru_evaluation must be within"));
}

#[test]
fn load_report_accepts_a_well_formed_report() {
    let path = write_temp(
        "ok",
        r#"{"taguru_evaluation":1,"generated_at":"2026-07-29T00:00:00Z",
           "inputs":{"context":"demo"},
           "corpus":{"revision_after":{"graph":1,"passages":1,"config":1},"stable":true},
           "cases":[]}"#,
    );
    let report = load_report(&path).unwrap();
    assert_eq!(report.view.inputs.context, "demo");
}

// ============================== COMPARISON_METRICS lexicon ==============================

#[test]
fn compare_lexicon_test() {
    // COMPARISON_METRICS must stay exactly CASE_SCOPED_METRICS minus
    // latency.passage_ms — see the doc comment on both consts
    // (thresholds.rs) for why. A future addition to
    // CASE_SCOPED_METRICS must be a deliberate choice here too, not a
    // silent omission.
    let mut expected: Vec<&str> = crate::evaluate::thresholds::CASE_SCOPED_METRICS
        .iter()
        .copied()
        .filter(|&metric| metric != "latency.passage_ms")
        .collect();
    let mut actual: Vec<&str> = COMPARISON_METRICS.to_vec();
    expected.sort_unstable();
    actual.sort_unstable();
    assert_eq!(actual, expected);
}
