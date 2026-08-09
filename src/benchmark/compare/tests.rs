use super::*;

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "taguru-benchmark-compare-{tag}-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

// ============================== ordering & banned keys ==============================
//
// nearest-rank and Distribution/Ratio n==0-shape tests moved to
// `crate::measure`'s own test module (issue #280) — they exercise
// `nearest_rank`/`Distribution::from_samples`/`ratio_metric` directly
// and belong beside the types they test.

#[test]
fn per_model_maps_serialize_in_lexicographic_order() {
    let mut models: BTreeMap<String, MetricsMap> = BTreeMap::new();
    models.insert("zeta".to_string(), MetricsMap::new());
    models.insert("alpha".to_string(), MetricsMap::new());
    let json = serde_json::to_string(&models).unwrap();
    assert!(
        json.find("alpha").unwrap() < json.find("zeta").unwrap(),
        "{json}"
    );
}

const BANNED_KEYS: [&str; 7] = [
    "rank",
    "score",
    "winner",
    "best",
    "recommended",
    "overall",
    "delta_vs",
];

fn assert_no_banned_keys(value: &Value, path: &str) {
    match value {
        Value::Object(map) => {
            for (key, v) in map {
                for banned in BANNED_KEYS {
                    assert!(
                        !key.to_lowercase().contains(banned),
                        "banned key fragment '{banned}' found in key '{key}' at {path}"
                    );
                }
                assert_no_banned_keys(v, &format!("{path}.{key}"));
            }
        }
        Value::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                assert_no_banned_keys(v, &format!("{path}[{i}]"));
            }
        }
        _ => {}
    }
}

#[test]
fn no_banned_key_appears_anywhere_in_the_artifact() {
    let dir = synthetic_results_dir("banned-keys");
    let measurements = compute_measurements(&dir).expect("computes");
    let value = serde_json::to_value(&measurements).unwrap();
    assert_no_banned_keys(&value, "$");

    let csv = render_csv(&measurements);
    for line in csv.lines().skip(1) {
        let metric = line.split(',').nth(4).unwrap_or("");
        for banned in BANNED_KEYS {
            assert!(
                !metric.to_lowercase().contains(banned),
                "banned key fragment '{banned}' found in CSV metric column '{metric}'"
            );
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

/// The 12 metric names issue #258 adds. Kept as an explicit list (not
/// derived from `build_definitions`) so a rename or an accidental drop
/// shows up as a failing assertion here rather than silently narrowing
/// what [`stability_lexicon_test_covers_every_new_metric_name`] checks.
const STABILITY_METRIC_NAMES: [&str; 12] = [
    "stability.run_pair_jaccard",
    "stability.keys_distinct",
    "stability.keys_in_all_runs_ratio",
    "stability.keys_in_single_run_ratio",
    "stability.key_presence_ratio",
    "stability.polarity_variation_ratio",
    "stability.weight_variation_ratio",
    "stability.attribution_variation_ratio",
    "stability.alias_canonical_variation_ratio",
    "run.associations_total",
    "run.elapsed_seconds_total",
    "run.documents_written",
];

/// ADR 0003 §9.4's stricter lexicon is stated for `differences.jsonl`
/// (#259), but #258 sets the naming precedent #259 inherits — so its
/// own new metric names are held to the same bar here, on top of
/// `no_banned_key_appears_anywhere_in_the_artifact`'s §9.3 rank-word
/// check above.
const ADR_9_4_LEXICON: [&str; 13] = [
    "miss",
    "error",
    "wrong",
    "incorrect",
    "fail",
    "omit",
    "expected",
    "gold",
    "truth",
    "recall",
    "precision",
    "better",
    "worse",
];

#[test]
fn stability_lexicon_test_covers_every_new_metric_name() {
    let dir = synthetic_multi_run_results_dir("stability-name-coverage");
    let measurements = compute_measurements(&dir).expect("computes");
    let model = &measurements.models["m"];
    let mut emitted: BTreeSet<&str> = BTreeSet::new();
    for name in model.keys() {
        if name.starts_with("stability.") || name.starts_with("run.") {
            emitted.insert(name.as_str());
        }
    }
    let expected: BTreeSet<&str> = STABILITY_METRIC_NAMES.into_iter().collect();
    assert_eq!(
        emitted, expected,
        "STABILITY_METRIC_NAMES has drifted from what compute_measurements actually emits"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn no_adr_0003_lexicon_word_appears_in_a_stability_metric_name() {
    for name in STABILITY_METRIC_NAMES {
        for banned in ADR_9_4_LEXICON {
            assert!(
                !name.to_lowercase().contains(banned),
                "banned lexicon fragment '{banned}' found in metric name '{name}'"
            );
        }
    }
}

#[test]
fn every_emitted_metric_keys_definitions_and_units_match_csv() {
    let dir = synthetic_results_dir("definitions-coverage");
    let measurements = compute_measurements(&dir).expect("computes");

    let mut all_metric_names: BTreeSet<String> = BTreeSet::new();
    for block in measurements.cells.values() {
        all_metric_names.extend(block.metrics.keys().cloned());
    }
    for metrics in measurements.models.values() {
        all_metric_names.extend(metrics.keys().cloned());
    }
    for by_doc in measurements.documents.values() {
        for by_run in by_doc.values() {
            for metrics in by_run.values() {
                all_metric_names.extend(metrics.keys().cloned());
            }
        }
    }

    for name in &all_metric_names {
        assert!(
            measurements.definitions.contains_key(name),
            "metric {name} has no definitions entry"
        );
    }

    let csv = render_csv(&measurements);
    let mut header = true;
    for line in csv.lines() {
        if header {
            header = false;
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        let metric = fields[4];
        let unit = fields[7];
        let expected = measurements.definitions.get(metric).unwrap().unit();
        assert_eq!(unit, expected, "unit mismatch for {metric}");
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn known_caveats_mention_what_they_conflate() {
    let d = build_definitions(&BTreeSet::new());
    let stop_malformed = d.get("attempt.state_rate.stop_malformed").unwrap();
    assert!(
        stop_malformed
            .caveat()
            .unwrap()
            .contains("validation_issues")
    );
    let length_limited = d.get("attempt.state_rate.length_limited").unwrap();
    assert!(
        length_limited
            .caveat()
            .unwrap()
            .contains("length_limited: true")
    );
}

// ============================== CSV projection ==============================

#[test]
fn csv_fields_with_commas_quotes_or_newlines_are_escaped() {
    assert_eq!(csv_field("plain"), "plain");
    assert_eq!(csv_field("a,b"), "\"a,b\"");
    assert_eq!(csv_field("a\"b"), "\"a\"\"b\"");
    assert_eq!(csv_field("a\nb"), "\"a\nb\"");
}

#[test]
fn csv_is_an_exact_value_projection_of_the_json() {
    let dir = synthetic_results_dir("csv-projection");
    let measurements = compute_measurements(&dir).expect("computes");
    let csv = render_csv(&measurements);

    let mut checked = 0;
    for line in csv.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        let (scope, model_id, run_index, document_id, metric, stat, value, _unit, n) = (
            fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], fields[6], fields[7],
            fields[8],
        );
        let metric_value = match scope {
            "cell" => {
                let cell_id = format!("{model_id}.run{:0>2}", run_index);
                &measurements.cells[&cell_id].metrics[metric]
            }
            "model" => &measurements.models[model_id][metric],
            "document" => {
                &measurements.documents[model_id][document_id][&format!("run{:0>2}", run_index)]
                    [metric]
            }
            other => panic!("unexpected scope {other}"),
        };
        let (json_n, rows) = (metric_value.sample_size(), metric_csv_rows(metric_value));
        assert_eq!(n.parse::<u64>().unwrap(), json_n);
        let expected = rows.iter().find(|(s, _)| *s == stat).unwrap().1;
        match expected {
            None => assert!(
                value.is_empty(),
                "expected empty for {metric}/{stat}, got {value}"
            ),
            Some(v) => {
                let parsed: f64 = value.parse().unwrap();
                assert!(
                    (parsed - v).abs() < 1e-9,
                    "{metric}/{stat}: {parsed} != {v}"
                );
            }
        }
        checked += 1;
    }
    assert!(checked > 0, "the synthetic fixture produced no CSV rows");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn csv_is_an_exact_value_projection_of_the_json_with_two_runs() {
    // Same check as csv_is_an_exact_value_projection_of_the_json, but
    // over synthetic_multi_run_results_dir — the fixture that actually
    // exercises stability.run_pair_jaccard and the run.* distributions
    // with n > 1, unlike the single-run fixture above.
    let dir = synthetic_multi_run_results_dir("csv-projection-multi-run");
    let measurements = compute_measurements(&dir).expect("computes");
    let csv = render_csv(&measurements);

    let mut checked = 0;
    let mut saw_stability_metric = false;
    for line in csv.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        let (scope, model_id, run_index, document_id, metric, stat, value, _unit, n) = (
            fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], fields[6], fields[7],
            fields[8],
        );
        if metric.starts_with("stability.") || metric.starts_with("run.") {
            saw_stability_metric = true;
        }
        let metric_value = match scope {
            "cell" => {
                let cell_id = format!("{model_id}.run{:0>2}", run_index);
                &measurements.cells[&cell_id].metrics[metric]
            }
            "model" => &measurements.models[model_id][metric],
            "document" => {
                &measurements.documents[model_id][document_id][&format!("run{:0>2}", run_index)]
                    [metric]
            }
            other => panic!("unexpected scope {other}"),
        };
        let (json_n, rows) = (metric_value.sample_size(), metric_csv_rows(metric_value));
        assert_eq!(n.parse::<u64>().unwrap(), json_n);
        let expected = rows.iter().find(|(s, _)| *s == stat).unwrap().1;
        match expected {
            None => assert!(
                value.is_empty(),
                "expected empty for {metric}/{stat}, got {value}"
            ),
            Some(v) => {
                let parsed: f64 = value.parse().unwrap();
                assert!(
                    (parsed - v).abs() < 1e-9,
                    "{metric}/{stat}: {parsed} != {v}"
                );
            }
        }
        checked += 1;
    }
    assert!(checked > 0, "the synthetic fixture produced no CSV rows");
    assert!(
        saw_stability_metric,
        "the multi-run fixture must exercise at least one stability.*/run.* row"
    );
    let _ = fs::remove_dir_all(&dir);
}

// ============================== grouping from synthetic runs ==============================

#[test]
fn chunk_seconds_sums_retries_and_excludes_cross_chunk() {
    let attempts = [
        AttemptRow {
            cell_id: "m.run01".into(),
            model_id: "m".into(),
            document_id: "doc".into(),
            chunk_index: 0,
            stage: "item".into(),
            state: "stop_malformed".into(),
            attempt_no: 1,
            length_limited: false,
            elapsed_seconds: 2.0,
            finish_reason: None,
            input_tokens: None,
            output_tokens: None,
            total_tokens: None,
            provider_metadata_present: false,
            parse_error_present: true,
            validation_rejected: false,
        },
        AttemptRow {
            cell_id: "m.run01".into(),
            model_id: "m".into(),
            document_id: "doc".into(),
            chunk_index: 0,
            stage: "item".into(),
            state: "stop_valid".into(),
            attempt_no: 2,
            length_limited: false,
            elapsed_seconds: 3.0,
            finish_reason: Some("stop".into()),
            input_tokens: Some(10),
            output_tokens: Some(5),
            total_tokens: Some(15),
            provider_metadata_present: true,
            parse_error_present: false,
            validation_rejected: false,
        },
        AttemptRow {
            cell_id: "m.run01".into(),
            model_id: "m".into(),
            document_id: "doc".into(),
            chunk_index: 0,
            stage: "cross_chunk".into(),
            state: "stop_valid".into(),
            attempt_no: 1,
            length_limited: false,
            elapsed_seconds: 100.0,
            finish_reason: Some("stop".into()),
            input_tokens: None,
            output_tokens: None,
            total_tokens: None,
            provider_metadata_present: true,
            parse_error_present: false,
            validation_rejected: false,
        },
    ];
    let refs: Vec<&AttemptRow> = attempts.iter().collect();
    let d = chunk_seconds_distribution(&refs);
    assert_eq!(d.n(), 1, "one chunk, two item-stage retries summed");
    assert_eq!(
        d.sum(),
        Some(5.0),
        "2.0 + 3.0, cross_chunk's 100.0 excluded"
    );

    let retry_rate = attempt_rate_metrics(&refs);
    let MetricValue::Ratio(r) = &retry_rate["attempt.retry_rate"] else {
        panic!()
    };
    assert_eq!(
        r.numerator(),
        Some(1),
        "only the second item-stage attempt is a retry"
    );
}

#[test]
fn wall_seconds_needs_both_start_and_end() {
    let started_only = DocRow {
        cell_id: "m.run01".into(),
        model_id: "m".into(),
        run_index: 1,
        document_id: "doc".into(),
        start_ts: Some(10.0),
        end_ts: None,
        outcome: None,
        associations: None,
        concepts: None,
        labels: None,
        questions: None,
        duplicates: None,
        dropped: None,
        elapsed_seconds_sum: 0.0,
        input_tokens_sum: None,
        batch: None,
    };
    assert_eq!(wall_seconds(&started_only), None);

    let complete = DocRow {
        end_ts: Some(25.0),
        ..started_only
    };
    assert_eq!(wall_seconds(&complete), Some(15.0));
}

#[test]
fn wall_seconds_drops_an_end_stamped_before_its_start() {
    let backwards = DocRow {
        cell_id: "m.run01".into(),
        model_id: "m".into(),
        run_index: 1,
        document_id: "doc".into(),
        start_ts: Some(10.0),
        end_ts: Some(9.0),
        outcome: None,
        associations: None,
        concepts: None,
        labels: None,
        questions: None,
        duplicates: None,
        dropped: None,
        elapsed_seconds_sum: 0.0,
        input_tokens_sum: None,
        batch: None,
    };
    assert_eq!(
        wall_seconds(&backwards),
        None,
        "a clock-stepped sample must not sink the distribution's min"
    );
    // A zero-length span is a real (if instant) measurement, not a
    // broken one.
    let instant = DocRow {
        end_ts: Some(10.0),
        ..backwards
    };
    assert_eq!(wall_seconds(&instant), Some(0.0));
}

#[test]
fn document_outcome_rates_counts_interrupted_in_the_denominator_only() {
    fn doc(outcome: Option<&str>) -> DocRow {
        DocRow {
            cell_id: "m.run01".into(),
            model_id: "m".into(),
            run_index: 1,
            document_id: "doc".into(),
            start_ts: Some(0.0),
            end_ts: outcome.map(|_| 1.0),
            outcome: outcome.map(str::to_string),
            associations: None,
            concepts: None,
            labels: None,
            questions: None,
            duplicates: None,
            dropped: None,
            elapsed_seconds_sum: 0.0,
            input_tokens_sum: None,
            batch: None,
        }
    }
    let rows = [doc(Some("written")), doc(Some("failed")), doc(None)];
    let refs: Vec<&DocRow> = rows.iter().collect();
    let rates = document_outcome_rates(&refs);
    let MetricValue::Ratio(written) = &rates["document.written_rate"] else {
        panic!()
    };
    assert_eq!(written.n(), 3);
    assert_eq!(written.numerator(), Some(1));
    let MetricValue::Ratio(failed) = &rates["document.failed_rate"] else {
        panic!()
    };
    assert_eq!(failed.numerator(), Some(1));
}

// ============================== batch analysis ==============================

#[test]
fn analyze_batch_counts_every_vocabulary_shape_once() {
    let batch = "\
{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"s\"}
{\"passage\":\"lorem ipsum\"}
{\"paragraph\":0,\"question\":\"who?\"}
{\"subject\":\"alice\",\"label\":\"knows\",\"object\":\"bob\",\"weight\":1.0,\"paragraph\":0}
{\"subject\":\"alice\",\"label\":\"knows\",\"object\":\"carol\",\"weight\":-1.0}
{\"subject\":\"bob\",\"label\":\"knows\",\"object\":\"carol\",\"weight\":1.0,\"paragraph\":99}
{\"alias\":\"Bob\",\"canonical\":\"bob\",\"kind\":\"concept\"}
{\"alias\":\"nope\",\"canonical\":\"ghost\",\"kind\":\"concept\"}
this is not json
{\"unexpected\":\"shape\"}
";
    let stats = analyze_batch(batch, 5);
    assert_eq!(stats.assoc_lines, 3);
    assert_eq!(stats.weight_positive, 2);
    assert_eq!(stats.weight_negative, 1);
    assert_eq!(
        stats.paragraph_attributed, 2,
        "two association lines carry a paragraph"
    );
    assert_eq!(
        stats.subjects,
        BTreeSet::from(["alice".to_string(), "bob".to_string()])
    );
    assert_eq!(stats.relations, BTreeSet::from(["knows".to_string()]));
    assert_eq!(
        stats.alias_orphans, 1,
        "canonical 'ghost' matches no subject/object"
    );
    assert_eq!(
        stats.paragraph_out_of_range, 1,
        "paragraph 99 >= paragraph_count 5"
    );
    assert_eq!(
        stats.invalid_lines, 2,
        "the non-JSON line and the unrecognized-shape line"
    );
}

#[test]
fn analyze_batch_captures_raw_rows_for_identity_keying() {
    let batch = "\
{\"subject\":\"Alice\",\"label\":\"knows\",\"object\":\"Bob\",\"weight\":1.0,\"paragraph\":0}
{\"alias\":\"Bob\",\"canonical\":\"bob\",\"kind\":\"concept\"}
{\"alias\":\"x\",\"canonical\":\"y\",\"kind\":\"mystery\"}
";
    let stats = analyze_batch(batch, 5);
    assert_eq!(stats.rows.associations.len(), 1);
    assert_eq!(stats.rows.associations[0].subject, "Alice");
    assert_eq!(stats.rows.associations[0].weight, 1.0);
    assert_eq!(
        stats.rows.aliases.len(),
        1,
        "the unrecognized 'mystery' kind is excluded from identity's alias rows"
    );
    assert_eq!(stats.rows.aliases[0].alias, "Bob");
    assert_eq!(stats.rows.aliases[0].canonical, "bob");
}

#[test]
fn analyze_batch_treats_an_unattributed_alias_kind_as_not_this_metrics_call() {
    let batch = "{\"alias\":\"x\",\"canonical\":\"y\",\"kind\":\"mystery\"}\n";
    let stats = analyze_batch(batch, 10);
    assert_eq!(stats.alias_orphans, 0);
}

#[test]
fn analyze_batch_does_not_wrap_a_paragraph_locator_past_u32_max() {
    // A paragraph value beyond u32::MAX must not silently truncate to a
    // small, in-range number (e.g. 4294967296 wrapping to 0) — it must
    // still be counted as out-of-range.
    let batch = "{\"subject\":\"a\",\"label\":\"knows\",\"object\":\"b\",\"weight\":1.0,\
                 \"paragraph\":4294967296}\n";
    let stats = analyze_batch(batch, 5);
    assert_eq!(stats.paragraph_attributed, 1);
    assert_eq!(
        stats.paragraph_out_of_range, 1,
        "4294967296 must not wrap to 0 and pass as in-range"
    );
}

// ============================== token exclusion ==============================

#[test]
fn attempts_with_no_provider_metadata_are_excluded_from_token_metrics() {
    let with_tokens = AttemptRow {
        cell_id: "m.run01".into(),
        model_id: "m".into(),
        document_id: "doc".into(),
        chunk_index: 0,
        stage: "item".into(),
        state: "stop_valid".into(),
        attempt_no: 1,
        length_limited: false,
        elapsed_seconds: 1.0,
        finish_reason: Some("stop".into()),
        input_tokens: Some(100),
        output_tokens: Some(50),
        total_tokens: Some(150),
        provider_metadata_present: true,
        parse_error_present: false,
        validation_rejected: false,
    };
    let timed_out = AttemptRow {
        state: "timeout".into(),
        finish_reason: None,
        input_tokens: None,
        output_tokens: None,
        total_tokens: None,
        provider_metadata_present: false,
        ..clone_attempt(&with_tokens)
    };
    let refs = vec![&with_tokens, &timed_out];
    let metrics = attempt_distribution_metrics(&refs);
    let MetricValue::Distribution(d) = &metrics["tokens.input_per_attempt"] else {
        panic!()
    };
    assert_eq!(
        d.n(),
        1,
        "the timeout attempt is excluded, not counted as 0"
    );
    assert_eq!(d.sum(), Some(100.0));

    let missing = attempt_rate_metrics(&refs);
    let MetricValue::Ratio(r) = &missing["attempt.provider_metadata_missing_rate"] else {
        panic!()
    };
    assert_eq!(r.numerator(), Some(1));
    assert_eq!(r.n(), 2);
}

fn clone_attempt(a: &AttemptRow) -> AttemptRow {
    AttemptRow {
        cell_id: a.cell_id.clone(),
        model_id: a.model_id.clone(),
        document_id: a.document_id.clone(),
        chunk_index: a.chunk_index,
        stage: a.stage.clone(),
        state: a.state.clone(),
        attempt_no: a.attempt_no,
        length_limited: a.length_limited,
        elapsed_seconds: a.elapsed_seconds,
        finish_reason: a.finish_reason.clone(),
        input_tokens: a.input_tokens,
        output_tokens: a.output_tokens,
        total_tokens: a.total_tokens,
        provider_metadata_present: a.provider_metadata_present,
        parse_error_present: a.parse_error_present,
        validation_rejected: a.validation_rejected,
    }
}

// ============================== argument parsing ==============================

#[test]
fn parse_args_requires_exactly_one_positional() {
    assert!(parse_args(&[]).is_err());
    assert!(parse_args(&["a".to_string(), "b".to_string()]).is_err());
    assert!(parse_args(&["--bogus".to_string()]).is_err());
    let args = parse_args(&["dir".to_string()]).unwrap();
    assert_eq!(args.dir, PathBuf::from("dir"));
    assert!(!args.with_text);
}

#[test]
fn parse_args_recognizes_with_text() {
    let args = parse_args(&["--with-text".to_string(), "dir".to_string()]).unwrap();
    assert_eq!(args.dir, PathBuf::from("dir"));
    assert!(args.with_text);

    // Flag order does not matter.
    let args = parse_args(&["dir".to_string(), "--with-text".to_string()]).unwrap();
    assert_eq!(args.dir, PathBuf::from("dir"));
    assert!(args.with_text);
}

// ============================== end-to-end synthetic fixture ==============================

/// Builds a minimal but realistic results directory: one model, one
/// run, two documents (one written with a batch, one failed), enough
/// to exercise every metric family without a real `taguru extract`
/// child process.
fn synthetic_results_dir(tag: &str) -> PathBuf {
    let dir = temp_dir(tag);
    fs::create_dir_all(dir.join("runs")).unwrap();
    fs::create_dir_all(dir.join("cells/m/run01")).unwrap();

    let batch = "\
{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"corpus/brewery.md\"}
{\"passage\":\"text\"}
{\"subject\":\"beer co\",\"label\":\"brews\",\"object\":\"lager\",\"weight\":1.0,\"paragraph\":0}
{\"subject\":\"beer co\",\"label\":\"brews\",\"object\":\"ale\",\"weight\":1.0,\"paragraph\":0}
{\"alias\":\"BeerCo\",\"canonical\":\"beer co\",\"kind\":\"concept\"}
";
    fs::write(dir.join("cells/m/run01/brewery.jsonl"), batch).unwrap();

    let runs_lines = [
        serde_json::json!({
            "kind": "header", "taguru_benchmark_runs": 1, "run_id": "run-1",
            "cell_id": "m.run01", "model_id": "m", "model_name": "m-model",
            "run_index": 1, "prompt_version": 1,
        }),
        serde_json::json!({
            "kind": "document", "ts": 100.0, "cell_id": "m.run01",
            "document_id": "brewery", "source": "corpus/brewery.md",
            "document_sha256": "sha-brewery", "chunk_total": 1, "phase": "start",
        }),
        serde_json::json!({
            "kind": "attempt", "source": "corpus/brewery.md", "stage": "item",
            "chunk_index": 0, "attempt": 1, "max_attempts": 2, "state": "stop_valid",
            "length_limited": false, "elapsed_seconds": 4.0,
            "provider_metadata": {"finish_reason": "stop", "input_tokens": 1000,
                "output_tokens": 200, "total_tokens": 1200},
            "parse_error": null, "validation_issues": null,
            "ts": 101.0, "cell_id": "m.run01", "model_id": "m", "run_index": 1,
            "document_id": "brewery", "document_sha256": "sha-brewery",
            "chunk_sha256": "sha-chunk0", "paragraph_first": 0, "paragraph_last": 0,
        }),
        serde_json::json!({
            "kind": "document", "ts": 110.0, "cell_id": "m.run01",
            "document_id": "brewery", "source": "corpus/brewery.md",
            "document_sha256": "sha-brewery", "phase": "end", "outcome": "written",
            "associations": 2, "concepts": 1, "labels": 0, "questions": 0,
            "duplicates": 0, "dropped": 0, "batch_path": "cells/m/run01/brewery.jsonl",
        }),
        serde_json::json!({
            "kind": "document", "ts": 111.0, "cell_id": "m.run01",
            "document_id": "sake", "source": "corpus/sake.md",
            "document_sha256": "sha-sake", "chunk_total": 1, "phase": "start",
        }),
        serde_json::json!({
            "kind": "attempt", "source": "corpus/sake.md", "stage": "item",
            "chunk_index": 0, "attempt": 1, "max_attempts": 2, "state": "timeout",
            "length_limited": false, "elapsed_seconds": 30.0,
            "provider_metadata": null, "parse_error": "timed out", "validation_issues": null,
            "ts": 141.0, "cell_id": "m.run01", "model_id": "m", "run_index": 1,
            "document_id": "sake", "document_sha256": "sha-sake",
            "chunk_sha256": "sha-chunk0", "paragraph_first": 0, "paragraph_last": 0,
        }),
        serde_json::json!({
            "kind": "document", "ts": 142.0, "cell_id": "m.run01",
            "document_id": "sake", "source": "corpus/sake.md",
            "document_sha256": "sha-sake", "phase": "end", "outcome": "failed",
            "associations": null, "concepts": null, "labels": null, "questions": null,
            "duplicates": null, "dropped": null, "batch_path": null,
        }),
        serde_json::json!({
            "kind": "cell", "ts": 143.0, "cell_id": "m.run01", "outcome": "complete",
            "documents_written": 1, "attempts_total": 2, "exit_code": 0,
        }),
    ];
    let runs_text = runs_lines
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(dir.join("runs/m.run01.jsonl"), runs_text).unwrap();

    let manifest = serde_json::json!({
        "taguru_benchmark_manifest": 1,
        "run_id": "run-1",
        "started_at": "2026-07-26T09:00:00Z",
        "finished_at": "2026-07-26T09:05:00Z",
        "taguru_version": "0.0.0",
        "sdk_versions": {},
        "harness": {},
        "extraction_settings": {},
        "documents": [
            {
                "document_id": "brewery", "path": "corpus/brewery.md", "bytes": 100,
                "sha256": "sha-brewery", "paragraph_count": 5, "chunk_total": 1, "chunks": [],
            },
            {
                "document_id": "sake", "path": "corpus/sake.md", "bytes": 50,
                "sha256": "sha-sake", "paragraph_count": 3, "chunk_total": 1, "chunks": [],
            },
        ],
        "models": [
            {
                "model_id": "m", "model_name": "m-model", "endpoint": "http://x",
                "digest": null, "quantization": null, "context_window": null,
                "structured_output_requested": "auto", "timeout_secs": 60,
                "provider_probe": {"attempted": [], "ok": true, "note": null},
            },
        ],
        "cells": [
            {
                "cell_id": "m.run01", "model_id": "m", "run_index": 1,
                "runs_file": "runs/m.run01.jsonl", "cell_dir": "cells/m/run01",
                "structured_output_resolved": "json_schema",
                "started_at": "2026-07-26T09:00:01Z",
                "finished_at": "2026-07-26T09:04:00Z", "outcome": "complete",
            },
        ],
        "environment": {"os": "linux", "arch": "x86_64"},
    });
    fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    dir
}

/// Builds a two-run results directory for one model, exercising every
/// `stability.*`/`run.*` metric (issue #258) against hand-computable
/// expected values:
///
/// - `brewery` completes in both runs. Its `beer co`/`brews`/`lager`
///   association is declared with different casing, an opposite
///   weight sign, and a different paragraph in run02 than in run01 —
///   the same association key (case folding merges the spellings), so
///   it exercises polarity and attribution variation. Its
///   `beer co`/`brews`/`ale` association is identical (case aside) in
///   both runs — a control key that must show *no* variation. run02
///   alone adds a `beer co`/`founded in`/`1990` association — a
///   run-local key. The `BrewCo Group` concept alias resolves to
///   `beer co` in run01 but to `brewer` in run02 — alias variation,
///   independent of any association's own key.
/// - `sake` completes only in run01 (run02's attempt times out) — its
///   keys must be excluded from every ratio that requires 2+ completed
///   runs for the same document, and from every run-pair Jaccard
///   sample (no document is shared between the pair's two completed
///   sets).
fn synthetic_multi_run_results_dir(tag: &str) -> PathBuf {
    let dir = temp_dir(tag);
    fs::create_dir_all(dir.join("runs")).unwrap();
    fs::create_dir_all(dir.join("cells/m/run01")).unwrap();
    fs::create_dir_all(dir.join("cells/m/run02")).unwrap();

    fs::write(
        dir.join("cells/m/run01/brewery.jsonl"),
        "\
{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"corpus/brewery.md\"}
{\"passage\":\"text\"}
{\"subject\":\"Beer Co\",\"label\":\"brews\",\"object\":\"Lager\",\"weight\":1.0,\"paragraph\":0}
{\"subject\":\"Beer Co\",\"label\":\"brews\",\"object\":\"Ale\",\"weight\":1.0,\"paragraph\":0}
{\"alias\":\"Beer Co\",\"canonical\":\"beer co\",\"kind\":\"concept\"}
{\"alias\":\"Lager\",\"canonical\":\"lager\",\"kind\":\"concept\"}
{\"alias\":\"Ale\",\"canonical\":\"ale\",\"kind\":\"concept\"}
{\"alias\":\"BrewCo Group\",\"canonical\":\"beer co\",\"kind\":\"concept\"}
",
    )
    .unwrap();
    fs::write(
        dir.join("cells/m/run01/sake.jsonl"),
        "\
{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"corpus/sake.md\"}
{\"passage\":\"text\"}
{\"subject\":\"Sake Co\",\"label\":\"brews\",\"object\":\"Junmai\",\"weight\":1.0,\"paragraph\":0}
{\"subject\":\"Sake Co\",\"label\":\"brews\",\"object\":\"Ginjo\",\"weight\":1.0,\"paragraph\":1}
",
    )
    .unwrap();
    fs::write(
        dir.join("cells/m/run02/brewery.jsonl"),
        "\
{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"corpus/brewery.md\"}
{\"passage\":\"text\"}
{\"subject\":\"BEER CO\",\"label\":\"brews\",\"object\":\"LAGER\",\"weight\":-1.0,\"paragraph\":5}
{\"subject\":\"BEER CO\",\"label\":\"brews\",\"object\":\"ALE\",\"weight\":1.0,\"paragraph\":0}
{\"subject\":\"Beer Co\",\"label\":\"founded in\",\"object\":\"1990\",\"weight\":1.0,\"paragraph\":2}
{\"alias\":\"BEER CO\",\"canonical\":\"beer co\",\"kind\":\"concept\"}
{\"alias\":\"LAGER\",\"canonical\":\"lager\",\"kind\":\"concept\"}
{\"alias\":\"ALE\",\"canonical\":\"ale\",\"kind\":\"concept\"}
{\"alias\":\"BrewCo Group\",\"canonical\":\"brewer\",\"kind\":\"concept\"}
",
    )
    .unwrap();

    fn attempt(
        cell_id: &str,
        run_index: usize,
        document_id: &str,
        document_sha256: &str,
        elapsed_seconds: f64,
        state: &str,
    ) -> Value {
        serde_json::json!({
            "kind": "attempt", "source": format!("corpus/{document_id}.md"), "stage": "item",
            "chunk_index": 0, "attempt": 1, "max_attempts": 2, "state": state,
            "length_limited": false, "elapsed_seconds": elapsed_seconds,
            "provider_metadata": if state == "stop_valid" {
                serde_json::json!({"finish_reason": "stop", "input_tokens": 100,
                    "output_tokens": 20, "total_tokens": 120})
            } else {
                Value::Null
            },
            "parse_error": if state == "stop_valid" { Value::Null } else { Value::String("timed out".into()) },
            "validation_issues": null,
            "ts": 0.0, "cell_id": cell_id, "model_id": "m", "run_index": run_index,
            "document_id": document_id, "document_sha256": document_sha256,
            "chunk_sha256": "sha-chunk0", "paragraph_first": 0, "paragraph_last": 0,
        })
    }
    fn doc_start(cell_id: &str, document_id: &str, source: &str, document_sha256: &str) -> Value {
        serde_json::json!({
            "kind": "document", "ts": 0.0, "cell_id": cell_id,
            "document_id": document_id, "source": source,
            "document_sha256": document_sha256, "chunk_total": 1, "phase": "start",
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn doc_end_written(
        cell_id: &str,
        document_id: &str,
        source: &str,
        document_sha256: &str,
        associations: u64,
        batch_path: &str,
    ) -> Value {
        serde_json::json!({
            "kind": "document", "ts": 1.0, "cell_id": cell_id,
            "document_id": document_id, "source": source,
            "document_sha256": document_sha256, "phase": "end", "outcome": "written",
            "associations": associations, "concepts": 0, "labels": 0, "questions": 0,
            "duplicates": 0, "dropped": 0, "batch_path": batch_path,
        })
    }
    fn doc_end_failed(
        cell_id: &str,
        document_id: &str,
        source: &str,
        document_sha256: &str,
    ) -> Value {
        serde_json::json!({
            "kind": "document", "ts": 1.0, "cell_id": cell_id,
            "document_id": document_id, "source": source,
            "document_sha256": document_sha256, "phase": "end", "outcome": "failed",
            "associations": null, "concepts": null, "labels": null, "questions": null,
            "duplicates": null, "dropped": null, "batch_path": null,
        })
    }

    let run01_lines = [
        serde_json::json!({
            "kind": "header", "taguru_benchmark_runs": 1, "run_id": "run-multi",
            "cell_id": "m.run01", "model_id": "m", "model_name": "m-model",
            "run_index": 1, "prompt_version": 1,
        }),
        doc_start("m.run01", "brewery", "corpus/brewery.md", "sha-brewery"),
        attempt("m.run01", 1, "brewery", "sha-brewery", 4.0, "stop_valid"),
        doc_end_written(
            "m.run01",
            "brewery",
            "corpus/brewery.md",
            "sha-brewery",
            2,
            "cells/m/run01/brewery.jsonl",
        ),
        doc_start("m.run01", "sake", "corpus/sake.md", "sha-sake"),
        attempt("m.run01", 1, "sake", "sha-sake", 5.0, "stop_valid"),
        doc_end_written(
            "m.run01",
            "sake",
            "corpus/sake.md",
            "sha-sake",
            2,
            "cells/m/run01/sake.jsonl",
        ),
        serde_json::json!({
            "kind": "cell", "ts": 6.0, "cell_id": "m.run01", "outcome": "complete",
            "documents_written": 2, "attempts_total": 2, "exit_code": 0,
        }),
    ];
    fs::write(
        dir.join("runs/m.run01.jsonl"),
        run01_lines
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .unwrap();

    let run02_lines = [
        serde_json::json!({
            "kind": "header", "taguru_benchmark_runs": 1, "run_id": "run-multi",
            "cell_id": "m.run02", "model_id": "m", "model_name": "m-model",
            "run_index": 2, "prompt_version": 1,
        }),
        doc_start("m.run02", "brewery", "corpus/brewery.md", "sha-brewery"),
        attempt("m.run02", 2, "brewery", "sha-brewery", 6.0, "stop_valid"),
        doc_end_written(
            "m.run02",
            "brewery",
            "corpus/brewery.md",
            "sha-brewery",
            3,
            "cells/m/run02/brewery.jsonl",
        ),
        doc_start("m.run02", "sake", "corpus/sake.md", "sha-sake"),
        attempt("m.run02", 2, "sake", "sha-sake", 7.0, "timeout"),
        doc_end_failed("m.run02", "sake", "corpus/sake.md", "sha-sake"),
        serde_json::json!({
            "kind": "cell", "ts": 8.0, "cell_id": "m.run02", "outcome": "complete",
            "documents_written": 1, "attempts_total": 2, "exit_code": 0,
        }),
    ];
    fs::write(
        dir.join("runs/m.run02.jsonl"),
        run02_lines
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .unwrap();

    let manifest = serde_json::json!({
        "taguru_benchmark_manifest": 1,
        "run_id": "run-multi",
        "started_at": "2026-07-26T09:00:00Z",
        "finished_at": "2026-07-26T09:10:00Z",
        "taguru_version": "0.0.0",
        "sdk_versions": {},
        "harness": {},
        "extraction_settings": {},
        "documents": [
            {
                "document_id": "brewery", "path": "corpus/brewery.md", "bytes": 100,
                "sha256": "sha-brewery", "paragraph_count": 10, "chunk_total": 1, "chunks": [],
            },
            {
                "document_id": "sake", "path": "corpus/sake.md", "bytes": 50,
                "sha256": "sha-sake", "paragraph_count": 5, "chunk_total": 1, "chunks": [],
            },
        ],
        "models": [
            {
                "model_id": "m", "model_name": "m-model", "endpoint": "http://x",
                "digest": null, "quantization": null, "context_window": null,
                "structured_output_requested": "auto", "timeout_secs": 60,
                "provider_probe": {"attempted": [], "ok": true, "note": null},
            },
        ],
        "cells": [
            {
                "cell_id": "m.run01", "model_id": "m", "run_index": 1,
                "runs_file": "runs/m.run01.jsonl", "cell_dir": "cells/m/run01",
                "structured_output_resolved": "json_schema",
                "started_at": "2026-07-26T09:00:01Z",
                "finished_at": "2026-07-26T09:04:00Z", "outcome": "complete",
            },
            {
                "cell_id": "m.run02", "model_id": "m", "run_index": 2,
                "runs_file": "runs/m.run02.jsonl", "cell_dir": "cells/m/run02",
                "structured_output_resolved": "json_schema",
                "started_at": "2026-07-26T09:05:00Z",
                "finished_at": "2026-07-26T09:09:00Z", "outcome": "complete",
            },
        ],
        "environment": {"os": "linux", "arch": "x86_64"},
    });
    fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    dir
}

#[test]
fn stability_metrics_over_two_runs_match_hand_computed_values() {
    let dir = synthetic_multi_run_results_dir("stability-multi-run");
    let measurements = compute_measurements(&dir).expect("computes");
    let model = &measurements.models["m"];

    // stability.run_pair_jaccard: the only pair (run01, run02),
    // restricted to `brewery` (the only document both runs completed).
    // run01 keys = {(beer co,brews,lager), (beer co,brews,ale)};
    // run02 keys = {(beer co,brews,lager), (beer co,brews,ale),
    // (beer co,founded in,1990)}. |intersection|=2, |union|=3.
    let MetricValue::Distribution(jaccard) = &model["stability.run_pair_jaccard"] else {
        panic!()
    };
    assert_eq!(jaccard.n(), 1);
    assert!((jaccard.sum().unwrap() - 2.0 / 3.0).abs() < 1e-9);

    // 5 distinct keys: brewery's {lager, ale, founded in/1990}, sake's
    // {junmai, ginjo} — over 3 completed (run, document) batches
    // (sake never completed in run02).
    let MetricValue::Count(keys_distinct) = &model["stability.keys_distinct"] else {
        panic!()
    };
    assert_eq!(keys_distinct.value(), Some(5.0));
    assert_eq!(keys_distinct.n(), 3);

    // Eligible keys (document completed in 2+ runs) are brewery's 3
    // keys only — sake completed in just 1 run. lager/ale are in both
    // runs (n_present=2 of 2); founded-in/1990 is run02-only
    // (n_present=1 of 2).
    let MetricValue::Ratio(in_all) = &model["stability.keys_in_all_runs_ratio"] else {
        panic!()
    };
    assert_eq!(in_all.n(), 3);
    assert_eq!(in_all.numerator(), Some(2));
    let MetricValue::Ratio(in_single) = &model["stability.keys_in_single_run_ratio"] else {
        panic!()
    };
    assert_eq!(in_single.n(), 3);
    assert_eq!(in_single.numerator(), Some(1));

    let MetricValue::Distribution(presence) = &model["stability.key_presence_ratio"] else {
        panic!()
    };
    assert_eq!(presence.n(), 3);
    assert_eq!(presence.min(), Some(0.5));
    assert_eq!(presence.max(), Some(1.0));
    assert!((presence.sum().unwrap() - 2.5).abs() < 1e-9);

    // Keys observed in 2+ runs: lager (polarity + weight + attribution
    // all vary) and ale (nothing varies) — 2 keys, 1 of each variation.
    let MetricValue::Ratio(polarity) = &model["stability.polarity_variation_ratio"] else {
        panic!()
    };
    assert_eq!(polarity.n(), 2);
    assert_eq!(polarity.numerator(), Some(1));
    let MetricValue::Ratio(weight) = &model["stability.weight_variation_ratio"] else {
        panic!()
    };
    assert_eq!(weight.n(), 2);
    assert_eq!(weight.numerator(), Some(1));
    let MetricValue::Ratio(attribution) = &model["stability.attribution_variation_ratio"] else {
        panic!()
    };
    assert_eq!(attribution.n(), 2);
    assert_eq!(attribution.numerator(), Some(1));

    // 4 alias spellings declared in both runs (beer co, lager, ale,
    // brewco group); only "BrewCo Group" resolves to a different
    // canonical (beer co in run01, brewer in run02).
    let MetricValue::Ratio(alias_variation) = &model["stability.alias_canonical_variation_ratio"]
    else {
        panic!()
    };
    assert_eq!(alias_variation.n(), 4);
    assert_eq!(alias_variation.numerator(), Some(1));

    // run.*: one sample per run (run_indexes come from manifest.cells,
    // not from which documents happened to complete).
    let MetricValue::Distribution(run_assoc) = &model["run.associations_total"] else {
        panic!()
    };
    assert_eq!(run_assoc.n(), 2);
    assert_eq!(
        run_assoc.min(),
        Some(3.0),
        "run02: 3 (brewery) + 0 (sake failed)"
    );
    assert_eq!(run_assoc.max(), Some(4.0), "run01: 2 (brewery) + 2 (sake)");

    let MetricValue::Distribution(run_written) = &model["run.documents_written"] else {
        panic!()
    };
    assert_eq!(run_written.min(), Some(1.0), "run02: only brewery written");
    assert_eq!(run_written.max(), Some(2.0), "run01: both written");

    let MetricValue::Distribution(run_elapsed) = &model["run.elapsed_seconds_total"] else {
        panic!()
    };
    assert_eq!(run_elapsed.n(), 2);
    assert_eq!(run_elapsed.min(), Some(9.0), "run01: 4.0 + 5.0");
    assert_eq!(run_elapsed.max(), Some(13.0), "run02: 6.0 + 7.0");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn compute_measurements_over_a_synthetic_results_directory() {
    let dir = synthetic_results_dir("smoke");
    let measurements = compute_measurements(&dir).expect("computes");

    assert_eq!(measurements.taguru_benchmark_measurements, 1);
    assert_eq!(measurements.run_id, "run-1");
    assert_eq!(measurements.percentile_method, "nearest-rank");
    assert_eq!(
        measurements.inputs.runs,
        vec!["runs/m.run01.jsonl".to_string()]
    );

    let cell = &measurements.cells["m.run01"];
    assert_eq!(cell.model_id, "m");
    assert_eq!(cell.run_index, 1);
    let MetricValue::Distribution(latency) = &cell.metrics["latency.attempt_seconds"] else {
        panic!()
    };
    assert_eq!(
        latency.n(),
        2,
        "both the written and the timed-out attempt count"
    );

    let MetricValue::Ratio(written_rate) = &cell.metrics["document.written_rate"] else {
        panic!()
    };
    assert_eq!(written_rate.n(), 2);
    assert_eq!(written_rate.numerator(), Some(1));

    let model = &measurements.models["m"];
    let MetricValue::Ratio(complete_rate) = &model["cell.complete_rate"] else {
        panic!()
    };
    assert_eq!(complete_rate.value(), Some(1.0));

    let brewery_run01 = &measurements.documents["m"]["brewery"]["run01"];
    let MetricValue::Count(associations) = &brewery_run01["extraction.associations"] else {
        panic!()
    };
    assert_eq!(associations.value(), Some(2.0));
    let MetricValue::Count(subjects) = &brewery_run01["extraction.subjects_distinct"] else {
        panic!()
    };
    assert_eq!(
        subjects.value(),
        Some(1.0),
        "one distinct subject: 'beer co'"
    );

    let sake_run01 = &measurements.documents["m"]["sake"]["run01"];
    let MetricValue::Count(sake_associations) = &sake_run01["extraction.associations"] else {
        panic!()
    };
    assert_eq!(
        sake_associations.value(),
        None,
        "sake failed — no associations"
    );
    assert_eq!(sake_associations.n(), 0);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_reprocessed_documents_second_end_record_supersedes_the_first() {
    let dir = synthetic_results_dir("duplicate-end");
    // A resumed cell that re-processed `brewery` logs a fresh `end`;
    // the cell finished in THAT state, so the later record must win —
    // mirroring `start`'s keep-the-earliest, not repeating it.
    let runs_path = dir.join("runs/m.run01.jsonl");
    let mut runs = fs::read_to_string(&runs_path).unwrap();
    runs.push_str(
        &serde_json::json!({
            "kind": "document", "ts": 120.0, "cell_id": "m.run01",
            "document_id": "brewery", "source": "corpus/brewery.md",
            "document_sha256": "sha-brewery", "phase": "end", "outcome": "written",
            "associations": 5, "concepts": 1, "labels": 0, "questions": 0,
            "duplicates": 0, "dropped": 0, "batch_path": "cells/m/run01/brewery.jsonl",
        })
        .to_string(),
    );
    runs.push('\n');
    fs::write(&runs_path, runs).unwrap();

    let measurements = compute_measurements(&dir).expect("computes");
    let brewery_run01 = &measurements.documents["m"]["brewery"]["run01"];
    let MetricValue::Count(associations) = &brewery_run01["extraction.associations"] else {
        panic!()
    };
    assert_eq!(associations.value(), Some(5.0));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn stability_metrics_with_a_single_run_are_the_defined_zero_shape() {
    // synthetic_results_dir has exactly one run (m.run01) — every
    // cross-run stability.* metric has nothing to compare and must
    // emit n:0/null (ADR 0003 §9.3's defined shape), while run.* still
    // reports its one sample.
    let dir = synthetic_results_dir("stability-n1");
    let measurements = compute_measurements(&dir).expect("computes");
    let model = &measurements.models["m"];

    let MetricValue::Distribution(jaccard) = &model["stability.run_pair_jaccard"] else {
        panic!()
    };
    assert_eq!(jaccard.n(), 0, "a single run has no pair to compare");

    let MetricValue::Count(keys_distinct) = &model["stability.keys_distinct"] else {
        panic!()
    };
    assert_eq!(
        keys_distinct.value(),
        Some(2.0),
        "brewery's 2 associations are 2 distinct keys"
    );
    assert_eq!(keys_distinct.n(), 1, "one completed (run, document) batch");

    for ratio_metric_name in [
        "stability.keys_in_all_runs_ratio",
        "stability.keys_in_single_run_ratio",
        "stability.polarity_variation_ratio",
        "stability.weight_variation_ratio",
        "stability.attribution_variation_ratio",
        "stability.alias_canonical_variation_ratio",
    ] {
        let MetricValue::Ratio(r) = &model[ratio_metric_name] else {
            panic!("{ratio_metric_name} is not a Ratio")
        };
        assert_eq!(
            r.n(),
            0,
            "{ratio_metric_name}: no key has a second run to compare against"
        );
        assert_eq!(r.value(), None);
        assert_eq!(r.numerator(), None);
    }

    let MetricValue::Distribution(presence) = &model["stability.key_presence_ratio"] else {
        panic!()
    };
    assert_eq!(presence.n(), 0);

    let MetricValue::Distribution(run_assoc) = &model["run.associations_total"] else {
        panic!()
    };
    assert_eq!(run_assoc.n(), 1, "one run this model has a cell for");
    assert_eq!(
        run_assoc.sum(),
        Some(2.0),
        "brewery's 2 associations; sake failed and contributes 0"
    );

    let MetricValue::Distribution(run_written) = &model["run.documents_written"] else {
        panic!()
    };
    assert_eq!(run_written.sum(), Some(1.0), "only brewery was written");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn compute_measurements_records_the_default_matching_block() {
    let dir = synthetic_results_dir("matching-block");
    let measurements = compute_measurements(&dir).expect("computes");
    assert_eq!(measurements.matching, identity::Matching::default());
    let value = serde_json::to_value(&measurements).unwrap();
    assert_eq!(
        value["matching"],
        serde_json::json!({
            "module": "benchmark::identity",
            "case_fold": true,
            "unicode_normalization": "NFKC",
            "alias_expansion": "batch-local",
            "weight_tolerance": 0.0,
        }),
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn compute_measurements_is_deterministic_across_two_runs() {
    let dir = synthetic_results_dir("determinism");
    let first = compute_measurements(&dir).expect("computes");
    let second = compute_measurements(&dir).expect("computes");
    assert_eq!(render_csv(&first), render_csv(&second));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn stability_metrics_are_deterministic_across_two_runs() {
    // The multi-run fixture's alias/presence bookkeeping iterates
    // BTreeMaps throughout (issue #258) — this pins that down the same
    // way compute_measurements_is_deterministic_across_two_runs does
    // for the rest of the artifact.
    let dir = synthetic_multi_run_results_dir("stability-determinism");
    let first = compute_measurements(&dir).expect("computes");
    let second = compute_measurements(&dir).expect("computes");
    assert_eq!(render_csv(&first), render_csv(&second));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn a_manifest_naming_an_unreadable_runs_file_is_an_error() {
    let dir = temp_dir("missing-runs-file");
    let manifest = serde_json::json!({
        "taguru_benchmark_manifest": 1,
        "run_id": "run-1",
        "cells": [
            {"cell_id": "m.run01", "model_id": "m", "run_index": 1,
             "runs_file": "runs/does-not-exist.jsonl", "cell_dir": "cells/m/run01",
             "outcome": "complete"},
        ],
    });
    fs::write(
        dir.join("manifest.json"),
        serde_json::to_string(&manifest).unwrap(),
    )
    .unwrap();
    let error = compute_measurements(&dir).unwrap_err();
    assert!(error.contains("m.run01"), "{error}");
    let _ = fs::remove_dir_all(&dir);
}

// ============================== differences.jsonl (issue #259) ==============================

/// Builds a two-model results directory — `alpha` (2 runs) and `beta`
/// (1 run) — engineered so `differences::compute_differences` exercises
/// every record kind exactly where named, against hand-computable
/// values:
///
/// Document `brewery` completes for both models (`alpha` in both its
/// runs, `beta` in its one), so it is eligible for association-level
/// records. Its keys:
/// - `beer co`/`brews`/`ale`: identical weight, paragraph, and spelling
///   on both sides — a control key that must fire nothing but its own
///   `association_shared`.
/// - `beer co`/`brews`/`lager`: `alpha` always positive, `beta` always
///   negative — disjoint sign sets, `polarity_difference`.
/// - `beer co`/`founded in`/`1990`: `alpha` always paragraph 2, `beta`
///   always paragraph 7 — disjoint paragraph sets, both `Some`,
///   `attribution_difference`.
/// - `cafe co`/`brews`/`latte` (case-folds identically): `alpha` always
///   spells it `Cafe Co`, `beta` always `CAFE CO` — 2 distinct raw
///   spellings, `surface_form_variation` on a *shared* key.
/// - `tea house`/`brews`/`matcha`: only `alpha` ever writes it (`beta`
///   never does), and `alpha` itself spells it two ways across its own
///   two runs (`Tea House` / `TEA HOUSE`) — `association_single_side`
///   *and* `surface_form_variation`, proving the latter fires on a
///   single-side key too.
/// - `town hall`/`hosts`/`meeting`: `alpha` never attributes a
///   paragraph (the field is omitted, `None`), `beta` always attributes
///   paragraph 5 — `{None}` vs. `{Some(5)}` are disjoint sets,
///   `attribution_difference` (proving `None` is its own category, not
///   "no signal").
/// - `library`/`hosts`/`reading`: neither side ever attributes a
///   paragraph — `{None}` vs. `{None}` are *not* disjoint, a control key
///   that must show no attribution difference despite neither side
///   locating it.
///
/// The `BrewCo Group` concept alias resolves to `beer co` in every
/// `alpha` run but to `brewer` in `beta`'s — disjoint canonicals,
/// `alias_resolution_difference`.
///
/// Document `sake` completes only for `alpha` (`beta` has no cell entry
/// for it at all) — `document_coverage` with `present_in: ["alpha"]`
/// and no association-level records, the coverage-exclusion case.
fn synthetic_two_model_results_dir(tag: &str) -> PathBuf {
    let dir = temp_dir(tag);
    fs::create_dir_all(dir.join("runs")).unwrap();
    fs::create_dir_all(dir.join("cells/alpha/run01")).unwrap();
    fs::create_dir_all(dir.join("cells/alpha/run02")).unwrap();
    fs::create_dir_all(dir.join("cells/beta/run01")).unwrap();

    fs::write(
        dir.join("cells/alpha/run01/brewery.jsonl"),
        "\
{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"corpus/brewery.md\"}
{\"passage\":\"text\"}
{\"subject\":\"beer co\",\"label\":\"brews\",\"object\":\"ale\",\"weight\":1.0,\"paragraph\":0}
{\"subject\":\"beer co\",\"label\":\"brews\",\"object\":\"lager\",\"weight\":1.0,\"paragraph\":1}
{\"subject\":\"beer co\",\"label\":\"founded in\",\"object\":\"1990\",\"weight\":1.0,\"paragraph\":2}
{\"subject\":\"Cafe Co\",\"label\":\"brews\",\"object\":\"latte\",\"weight\":1.0,\"paragraph\":3}
{\"subject\":\"Tea House\",\"label\":\"brews\",\"object\":\"matcha\",\"weight\":1.0,\"paragraph\":4}
{\"subject\":\"Town Hall\",\"label\":\"hosts\",\"object\":\"meeting\",\"weight\":1.0}
{\"subject\":\"Library\",\"label\":\"hosts\",\"object\":\"reading\",\"weight\":1.0}
{\"alias\":\"BrewCo Group\",\"canonical\":\"beer co\",\"kind\":\"concept\"}
",
    )
    .unwrap();
    fs::write(
        dir.join("cells/alpha/run01/sake.jsonl"),
        "\
{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"corpus/sake.md\"}
{\"passage\":\"text\"}
{\"subject\":\"sake co\",\"label\":\"brews\",\"object\":\"junmai\",\"weight\":1.0,\"paragraph\":0}
",
    )
    .unwrap();
    fs::write(
        dir.join("cells/alpha/run02/brewery.jsonl"),
        "\
{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"corpus/brewery.md\"}
{\"passage\":\"text\"}
{\"subject\":\"beer co\",\"label\":\"brews\",\"object\":\"ale\",\"weight\":1.0,\"paragraph\":0}
{\"subject\":\"beer co\",\"label\":\"brews\",\"object\":\"lager\",\"weight\":1.0,\"paragraph\":1}
{\"subject\":\"beer co\",\"label\":\"founded in\",\"object\":\"1990\",\"weight\":1.0,\"paragraph\":2}
{\"subject\":\"Cafe Co\",\"label\":\"brews\",\"object\":\"latte\",\"weight\":1.0,\"paragraph\":3}
{\"subject\":\"TEA HOUSE\",\"label\":\"brews\",\"object\":\"matcha\",\"weight\":1.0,\"paragraph\":4}
{\"subject\":\"Town Hall\",\"label\":\"hosts\",\"object\":\"meeting\",\"weight\":1.0}
{\"subject\":\"Library\",\"label\":\"hosts\",\"object\":\"reading\",\"weight\":1.0}
{\"alias\":\"BrewCo Group\",\"canonical\":\"beer co\",\"kind\":\"concept\"}
",
    )
    .unwrap();
    fs::write(
        dir.join("cells/beta/run01/brewery.jsonl"),
        "\
{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"corpus/brewery.md\"}
{\"passage\":\"text\"}
{\"subject\":\"beer co\",\"label\":\"brews\",\"object\":\"ale\",\"weight\":1.0,\"paragraph\":0}
{\"subject\":\"beer co\",\"label\":\"brews\",\"object\":\"lager\",\"weight\":-1.0,\"paragraph\":1}
{\"subject\":\"beer co\",\"label\":\"founded in\",\"object\":\"1990\",\"weight\":1.0,\"paragraph\":7}
{\"subject\":\"CAFE CO\",\"label\":\"brews\",\"object\":\"latte\",\"weight\":1.0,\"paragraph\":3}
{\"subject\":\"Town Hall\",\"label\":\"hosts\",\"object\":\"meeting\",\"weight\":1.0,\"paragraph\":5}
{\"subject\":\"Library\",\"label\":\"hosts\",\"object\":\"reading\",\"weight\":1.0}
{\"alias\":\"BrewCo Group\",\"canonical\":\"brewer\",\"kind\":\"concept\"}
",
    )
    .unwrap();

    fn doc_start(cell_id: &str, document_id: &str, source: &str, document_sha256: &str) -> Value {
        serde_json::json!({
            "kind": "document", "ts": 0.0, "cell_id": cell_id,
            "document_id": document_id, "source": source,
            "document_sha256": document_sha256, "chunk_total": 1, "phase": "start",
        })
    }
    fn doc_end_written(
        cell_id: &str,
        document_id: &str,
        source: &str,
        document_sha256: &str,
        batch_path: &str,
    ) -> Value {
        serde_json::json!({
            "kind": "document", "ts": 1.0, "cell_id": cell_id,
            "document_id": document_id, "source": source,
            "document_sha256": document_sha256, "phase": "end", "outcome": "written",
            "associations": 1, "concepts": 0, "labels": 0, "questions": 0,
            "duplicates": 0, "dropped": 0, "batch_path": batch_path,
        })
    }

    let alpha_run01_lines = [
        serde_json::json!({
            "kind": "header", "taguru_benchmark_runs": 1, "run_id": "run-diff",
            "cell_id": "alpha.run01", "model_id": "alpha", "model_name": "alpha-model",
            "run_index": 1, "prompt_version": 1,
        }),
        doc_start("alpha.run01", "brewery", "corpus/brewery.md", "sha-brewery"),
        doc_end_written(
            "alpha.run01",
            "brewery",
            "corpus/brewery.md",
            "sha-brewery",
            "cells/alpha/run01/brewery.jsonl",
        ),
        doc_start("alpha.run01", "sake", "corpus/sake.md", "sha-sake"),
        doc_end_written(
            "alpha.run01",
            "sake",
            "corpus/sake.md",
            "sha-sake",
            "cells/alpha/run01/sake.jsonl",
        ),
        serde_json::json!({
            "kind": "cell", "ts": 2.0, "cell_id": "alpha.run01", "outcome": "complete",
            "documents_written": 2, "attempts_total": 0, "exit_code": 0,
        }),
    ];
    fs::write(
        dir.join("runs/alpha.run01.jsonl"),
        alpha_run01_lines
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .unwrap();

    let alpha_run02_lines = [
        serde_json::json!({
            "kind": "header", "taguru_benchmark_runs": 1, "run_id": "run-diff",
            "cell_id": "alpha.run02", "model_id": "alpha", "model_name": "alpha-model",
            "run_index": 2, "prompt_version": 1,
        }),
        doc_start("alpha.run02", "brewery", "corpus/brewery.md", "sha-brewery"),
        doc_end_written(
            "alpha.run02",
            "brewery",
            "corpus/brewery.md",
            "sha-brewery",
            "cells/alpha/run02/brewery.jsonl",
        ),
        serde_json::json!({
            "kind": "cell", "ts": 2.0, "cell_id": "alpha.run02", "outcome": "complete",
            "documents_written": 1, "attempts_total": 0, "exit_code": 0,
        }),
    ];
    fs::write(
        dir.join("runs/alpha.run02.jsonl"),
        alpha_run02_lines
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .unwrap();

    let beta_run01_lines = [
        serde_json::json!({
            "kind": "header", "taguru_benchmark_runs": 1, "run_id": "run-diff",
            "cell_id": "beta.run01", "model_id": "beta", "model_name": "beta-model",
            "run_index": 1, "prompt_version": 1,
        }),
        doc_start("beta.run01", "brewery", "corpus/brewery.md", "sha-brewery"),
        doc_end_written(
            "beta.run01",
            "brewery",
            "corpus/brewery.md",
            "sha-brewery",
            "cells/beta/run01/brewery.jsonl",
        ),
        serde_json::json!({
            "kind": "cell", "ts": 2.0, "cell_id": "beta.run01", "outcome": "complete",
            "documents_written": 1, "attempts_total": 0, "exit_code": 0,
        }),
    ];
    fs::write(
        dir.join("runs/beta.run01.jsonl"),
        beta_run01_lines
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .unwrap();

    let manifest = serde_json::json!({
        "taguru_benchmark_manifest": 1,
        "run_id": "run-diff",
        "started_at": "2026-07-26T09:00:00Z",
        "finished_at": "2026-07-26T09:10:00Z",
        "taguru_version": "0.0.0",
        "sdk_versions": {},
        "harness": {},
        "extraction_settings": {},
        "documents": [
            {
                "document_id": "brewery", "path": "corpus/brewery.md", "bytes": 500,
                "sha256": "sha-brewery", "paragraph_count": 10, "chunk_total": 2,
                "chunks": [
                    {"chunk_index": 0, "chunk_sha256": "sha-chunk0", "chunk_bytes": 200,
                     "paragraph_first": 0, "paragraph_last": 4},
                    {"chunk_index": 1, "chunk_sha256": "sha-chunk1", "chunk_bytes": 200,
                     "paragraph_first": 5, "paragraph_last": 9},
                ],
            },
            {
                "document_id": "sake", "path": "corpus/sake.md", "bytes": 50,
                "sha256": "sha-sake", "paragraph_count": 3, "chunk_total": 1, "chunks": [],
            },
        ],
        "models": [
            {
                "model_id": "alpha", "model_name": "alpha-model", "endpoint": "http://x",
                "digest": null, "quantization": null, "context_window": null,
                "structured_output_requested": "auto", "timeout_secs": 60,
                "provider_probe": {"attempted": [], "ok": true, "note": null},
            },
            {
                "model_id": "beta", "model_name": "beta-model", "endpoint": "http://y",
                "digest": null, "quantization": null, "context_window": null,
                "structured_output_requested": "auto", "timeout_secs": 60,
                "provider_probe": {"attempted": [], "ok": true, "note": null},
            },
        ],
        "cells": [
            {
                "cell_id": "alpha.run01", "model_id": "alpha", "run_index": 1,
                "runs_file": "runs/alpha.run01.jsonl", "cell_dir": "cells/alpha/run01",
                "structured_output_resolved": "json_schema",
                "started_at": "2026-07-26T09:00:01Z",
                "finished_at": "2026-07-26T09:04:00Z", "outcome": "complete",
            },
            {
                "cell_id": "alpha.run02", "model_id": "alpha", "run_index": 2,
                "runs_file": "runs/alpha.run02.jsonl", "cell_dir": "cells/alpha/run02",
                "structured_output_resolved": "json_schema",
                "started_at": "2026-07-26T09:05:00Z",
                "finished_at": "2026-07-26T09:09:00Z", "outcome": "complete",
            },
            {
                "cell_id": "beta.run01", "model_id": "beta", "run_index": 1,
                "runs_file": "runs/beta.run01.jsonl", "cell_dir": "cells/beta/run01",
                "structured_output_resolved": "json_schema",
                "started_at": "2026-07-26T09:00:01Z",
                "finished_at": "2026-07-26T09:04:00Z", "outcome": "complete",
            },
        ],
        "environment": {"os": "linux", "arch": "x86_64"},
    });
    fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    dir
}

/// Loads `manifest.json` + `runs/*.jsonl` + `cells/**` for a results
/// directory and returns the parsed differences.jsonl lines as `Value`s
/// (line 0 is the header) — the test-side equivalent of
/// `compute_measurements`, going through the same `load_results` this
/// module's production code shares between both artifacts.
fn compute_differences_lines(dir: &Path, with_text: bool) -> Result<Vec<Value>, String> {
    let manifest = super::super::load_bench_manifest(&dir.join("manifest.json"))?;
    let loaded = load_results(dir, &manifest)?;
    let text = differences::compute_differences(
        &manifest,
        &loaded.doc_rows,
        &differences::DifferencesOptions { with_text },
    )?;
    text.lines()
        .map(|line| serde_json::from_str(line).map_err(|e| e.to_string()))
        .collect()
}

fn records_of_kind<'a>(lines: &'a [Value], kind: &str) -> Vec<&'a Value> {
    lines
        .iter()
        .filter(|v| v["kind"] == kind)
        .collect::<Vec<_>>()
}

#[test]
fn differences_header_matches_the_adr_shape() {
    let dir = synthetic_two_model_results_dir("differences-header");
    let lines = compute_differences_lines(&dir, false).expect("computes");
    let header = &lines[0];
    assert_eq!(header["kind"], "header");
    assert_eq!(header["taguru_benchmark_differences"], 2);
    assert_eq!(header["run_id"], "run-diff");
    assert_eq!(header["text_included"], false);
    assert_eq!(
        header["pairs"],
        serde_json::json!([{"pair_id": "5:alpha__beta", "a": "alpha", "b": "beta"}])
    );
    assert_eq!(
        header["matching"],
        serde_json::json!({
            "module": "benchmark::identity",
            "case_fold": true,
            "unicode_normalization": "NFKC",
            "alias_expansion": "batch-local",
            "weight_tolerance": 0.0,
        }),
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn differences_document_coverage_marks_a_document_only_one_side_completed() {
    let dir = synthetic_two_model_results_dir("differences-coverage");
    let lines = compute_differences_lines(&dir, false).expect("computes");
    let coverage = records_of_kind(&lines, "document_coverage");
    assert_eq!(coverage.len(), 2, "brewery and sake, one record each");

    let brewery = coverage
        .iter()
        .find(|r| r["sides"]["a"]["n_present"] == 2 && r["sides"]["b"]["n_present"] == 1)
        .expect("brewery: alpha completed 2 runs, beta 1");
    assert_eq!(brewery["present_in"], serde_json::json!(["alpha", "beta"]));
    assert_eq!(brewery["sides"]["a"]["runs"], serde_json::json!([1, 2]));
    assert_eq!(brewery["sides"]["b"]["runs"], serde_json::json!([1]));

    let sake = coverage
        .iter()
        .find(|r| r["present_in"] == serde_json::json!(["alpha"]))
        .expect("sake: alpha only");
    assert!(sake["sides"]["b"].is_null());
    assert_eq!(sake["sides"]["a"]["n_present"], 1);

    // sake is excluded from every key-level record (association_shared,
    // single_side, and every difference kind), since it is not eligible
    // (beta never completed it).
    for kind in [
        "association_shared",
        "association_single_side",
        "polarity_difference",
        "attribution_difference",
        "surface_form_variation",
    ] {
        for record in records_of_kind(&lines, kind) {
            assert_ne!(record["locator"]["document_id"], "sake", "kind={kind}");
        }
    }

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn differences_control_key_fires_nothing_but_association_shared() {
    let dir = synthetic_two_model_results_dir("differences-control");
    let lines = compute_differences_lines(&dir, false).expect("computes");
    let shared = records_of_kind(&lines, "association_shared");
    // "beer co" also keys lager and founded-in/1990 — disambiguate on object.
    let ale = shared
        .iter()
        .find(|r| r["key"]["object"] == "ale")
        .expect("ale is a shared key");
    assert_eq!(ale["present_in"], serde_json::json!(["alpha", "beta"]));
    assert_eq!(
        ale["sides"]["a"],
        serde_json::json!({"runs": [1, 2], "n_present": 2})
    );
    assert_eq!(
        ale["sides"]["b"],
        serde_json::json!({"runs": [1], "n_present": 1})
    );

    for kind in [
        "polarity_difference",
        "attribution_difference",
        "surface_form_variation",
    ] {
        assert!(
            records_of_kind(&lines, kind)
                .iter()
                .all(|r| r["key"]["object"] != "ale"),
            "the control key must not fire {kind}"
        );
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn differences_polarity_difference_fires_only_on_disjoint_sign_sets() {
    let dir = synthetic_two_model_results_dir("differences-polarity");
    let lines = compute_differences_lines(&dir, false).expect("computes");
    let polarity = records_of_kind(&lines, "polarity_difference");
    assert_eq!(polarity.len(), 1, "only lager disagrees in sign");
    let record = polarity[0];
    assert_eq!(record["key"]["object"], "lager");
    assert_eq!(record["present_in"], serde_json::json!(["alpha", "beta"]));
    assert_eq!(record["sides"]["a"]["weight_signs"], serde_json::json!([1]));
    assert_eq!(
        record["sides"]["b"]["weight_signs"],
        serde_json::json!([-1])
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn differences_attribution_difference_treats_none_as_its_own_category() {
    let dir = synthetic_two_model_results_dir("differences-attribution");
    let lines = compute_differences_lines(&dir, false).expect("computes");
    let attribution = records_of_kind(&lines, "attribution_difference");
    // founded-in/1990 (Some(2) vs Some(7)) and town hall/meeting (None
    // vs Some(5)) both fire; library/reading (None vs None) must not.
    assert_eq!(attribution.len(), 2);

    let founded_in = attribution
        .iter()
        .find(|r| r["key"]["label"] == "founded in")
        .expect("founded-in fires");
    assert_eq!(
        founded_in["sides"]["a"]["paragraphs"],
        serde_json::json!([2])
    );
    assert_eq!(
        founded_in["sides"]["b"]["paragraphs"],
        serde_json::json!([7])
    );

    let town_hall = attribution
        .iter()
        .find(|r| r["key"]["subject"] == "town hall")
        .expect("town hall fires: {None} vs {Some(5)} are disjoint");
    assert_eq!(
        town_hall["sides"]["a"]["paragraphs"],
        serde_json::json!([null])
    );
    assert_eq!(
        town_hall["sides"]["b"]["paragraphs"],
        serde_json::json!([5])
    );

    assert!(
        attribution.iter().all(|r| r["key"]["subject"] != "library"),
        "library must not fire: {{None}} vs {{None}} are not disjoint"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn differences_surface_form_variation_fires_on_shared_and_single_side_keys() {
    let dir = synthetic_two_model_results_dir("differences-surface");
    let lines = compute_differences_lines(&dir, false).expect("computes");
    let surface = records_of_kind(&lines, "surface_form_variation");
    assert_eq!(
        surface.len(),
        2,
        "cafe co (shared) and tea house (single-side)"
    );

    let cafe = surface
        .iter()
        .find(|r| r["key"]["subject"] == "cafe co")
        .expect("cafe co: alpha spells it differently from beta");
    assert_eq!(cafe["present_in"], serde_json::json!(["alpha", "beta"]));
    assert_eq!(
        cafe["sides"]["a"]["surface_forms"],
        serde_json::json!([{"subject": "Cafe Co", "label": "brews", "object": "latte"}]),
    );
    assert_eq!(
        cafe["sides"]["b"]["surface_forms"],
        serde_json::json!([{"subject": "CAFE CO", "label": "brews", "object": "latte"}]),
    );

    let tea_house = surface
        .iter()
        .find(|r| r["key"]["subject"] == "tea house")
        .expect("tea house: alpha alone spelled it two ways across its own 2 runs");
    assert_eq!(tea_house["present_in"], serde_json::json!(["alpha"]));
    assert!(tea_house["sides"]["b"].is_null());
    assert_eq!(
        tea_house["sides"]["a"]["surface_forms"],
        serde_json::json!([
            {"subject": "TEA HOUSE", "label": "brews", "object": "matcha"},
            {"subject": "Tea House", "label": "brews", "object": "matcha"},
        ]),
        "sorted (BTreeSet) order of the two raw spellings"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn differences_association_single_side_has_a_null_absent_side() {
    let dir = synthetic_two_model_results_dir("differences-single-side");
    let lines = compute_differences_lines(&dir, false).expect("computes");
    let single_side = records_of_kind(&lines, "association_single_side");
    assert_eq!(single_side.len(), 1);
    let record = single_side[0];
    assert_eq!(record["key"]["subject"], "tea house");
    assert_eq!(record["present_in"], serde_json::json!(["alpha"]));
    assert!(
        record["sides"]["b"].is_null(),
        "absent side is null, never a count"
    );
    assert_eq!(record["sides"]["a"]["n_present"], 2);
    assert_eq!(record["sides"]["a"]["runs"], serde_json::json!([1, 2]));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn differences_alias_resolution_difference_requires_both_sides_and_disjoint_canonicals() {
    let dir = synthetic_two_model_results_dir("differences-alias");
    let lines = compute_differences_lines(&dir, false).expect("computes");
    let alias = records_of_kind(&lines, "alias_resolution_difference");
    assert_eq!(alias.len(), 1);
    let record = alias[0];
    assert_eq!(record["alias_kind"], "concept");
    assert_eq!(record["spelling"], "brewco group");
    assert_eq!(record["present_in"], serde_json::json!(["alpha", "beta"]));
    assert_eq!(
        record["sides"]["a"]["canonicals"],
        serde_json::json!(["beer co"])
    );
    assert_eq!(record["sides"]["a"]["n_present"], 2);
    assert_eq!(
        record["sides"]["b"]["canonicals"],
        serde_json::json!(["brewer"])
    );
    assert_eq!(record["sides"]["b"]["n_present"], 1);
    // Alias lines carry no paragraph locator of their own.
    assert!(record["locator"]["paragraph"].is_null());
    assert!(record["locator"]["chunk_index"].is_null());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn differences_locator_selects_the_minimum_paragraph_and_derives_its_chunk() {
    let dir = synthetic_two_model_results_dir("differences-locator");
    let lines = compute_differences_lines(&dir, false).expect("computes");

    // founded-in/1990: alpha's paragraph is 2, beta's is 7 — the
    // locator must point at the minimum (2), inside chunk 0
    // (paragraph_first=0, paragraph_last=4), not chunk 1.
    let attribution = records_of_kind(&lines, "attribution_difference");
    let founded_in = attribution
        .iter()
        .find(|r| r["key"]["label"] == "founded in")
        .unwrap();
    let locator = &founded_in["locator"];
    assert_eq!(locator["document_id"], "brewery");
    assert_eq!(locator["source"], "corpus/brewery.md");
    assert_eq!(locator["document_sha256"], "sha-brewery");
    assert_eq!(locator["paragraph"], 2);
    assert_eq!(locator["chunk_index"], 0);
    assert_eq!(locator["chunk_sha256"], "sha-chunk0");
    assert!(locator["text"].is_null(), "--with-text was not given");
    assert_eq!(locator["text_truncated"], false);

    // town hall/meeting: alpha never attributes a paragraph, beta
    // attributes 5 — the union is {None, Some(5)}, so the minimum
    // Some(.) is 5, inside chunk 1.
    let town_hall = attribution
        .iter()
        .find(|r| r["key"]["subject"] == "town hall")
        .unwrap();
    assert_eq!(town_hall["locator"]["paragraph"], 5);
    assert_eq!(town_hall["locator"]["chunk_index"], 1);
    assert_eq!(town_hall["locator"]["chunk_sha256"], "sha-chunk1");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn differences_lexicon_test_bans_the_adr_0003_9_4_vocabulary_everywhere() {
    let dir = synthetic_two_model_results_dir("differences-lexicon");
    let lines = compute_differences_lines(&dir, false).expect("computes");
    let extended_lexicon: Vec<&str> = ADR_9_4_LEXICON
        .iter()
        .copied()
        .chain(["false_positive", "falsepositive"])
        .collect();
    for (i, line) in lines.iter().enumerate() {
        assert_no_banned_keys(line, &format!("differences.jsonl[{i}]"));
        for banned in BANNED_KEYS {
            let kind = line["kind"].as_str().unwrap_or("");
            assert!(
                !kind.to_lowercase().contains(banned),
                "banned key fragment '{banned}' found in kind '{kind}'"
            );
        }
        for banned in &extended_lexicon {
            let kind = line["kind"].as_str().unwrap_or("");
            assert!(
                !kind.to_lowercase().contains(*banned),
                "banned ADR §9.4 lexicon fragment '{banned}' found in kind '{kind}'"
            );
            fn walk_values(value: &Value, banned: &str, path: &str) {
                match value {
                    Value::Object(map) => {
                        for (k, v) in map {
                            assert!(
                                !k.to_lowercase().contains(banned),
                                "banned ADR §9.4 lexicon fragment '{banned}' found in key '{k}' at {path}"
                            );
                            walk_values(v, banned, &format!("{path}.{k}"));
                        }
                    }
                    Value::Array(items) => {
                        for (idx, v) in items.iter().enumerate() {
                            walk_values(v, banned, &format!("{path}[{idx}]"));
                        }
                    }
                    _ => {}
                }
            }
            walk_values(line, banned, &format!("differences.jsonl[{i}]"));
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn differences_a_single_model_directory_yields_a_header_with_no_pairs() {
    let dir = synthetic_results_dir("differences-single-model");
    let lines = compute_differences_lines(&dir, false).expect("computes");
    assert_eq!(lines.len(), 1, "header only — no pair to compare");
    assert_eq!(lines[0]["pairs"], serde_json::json!([]));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn compute_differences_is_deterministic_across_two_runs() {
    let dir = synthetic_two_model_results_dir("differences-determinism");
    let first = compute_differences_lines(&dir, false).expect("computes");
    let second = compute_differences_lines(&dir, false).expect("computes");
    assert_eq!(first, second);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn with_text_embeds_the_exact_paragraph_bytes() {
    let dir = synthetic_two_model_results_dir("differences-with-text");
    // The corpus file compare will actually read for --with-text: its
    // sha256 must match manifest.json's pinned "sha-brewery" — since
    // that fixture uses a placeholder hash, point the manifest at a
    // real file and recompute a real hash for this one test.
    fs::create_dir_all(dir.join("corpus")).unwrap();
    let paragraphs = "para0\n\npara1\n\npara2 is this one\n\npara3\n\npara4\n\npara5\n\npara6\n\npara7\n\npara8\n\npara9";
    fs::write(dir.join("corpus/brewery.md"), paragraphs).unwrap();
    let real_sha = crate::sha256::sha256_hex(paragraphs.as_bytes());

    let manifest_path = dir.join("manifest.json");
    let mut manifest: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["documents"][0]["path"] =
        serde_json::json!(dir.join("corpus/brewery.md").to_string_lossy());
    manifest["documents"][0]["sha256"] = serde_json::json!(real_sha);
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let lines = compute_differences_lines(&dir, true).expect("computes");
    assert_eq!(lines[0]["text_included"], true);

    let attribution = records_of_kind(&lines, "attribution_difference");
    let founded_in = attribution
        .iter()
        .find(|r| r["key"]["label"] == "founded in")
        .unwrap();
    // paragraph 2 (0-indexed, blank-line separated) is "para2 is this one".
    assert_eq!(founded_in["locator"]["text"], "para2 is this one");
    assert_eq!(founded_in["locator"]["text_truncated"], false);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn with_text_truncates_at_the_cap_on_a_char_boundary() {
    let dir = synthetic_two_model_results_dir("differences-with-text-truncate");
    fs::create_dir_all(dir.join("corpus")).unwrap();
    // founded-in/1990's locator resolves to paragraph 2 (alpha's value,
    // the minimum of {2, 7}) — put the oversized paragraph there,
    // blank-line separated from 9 short filler paragraphs so the
    // document still has 10 paragraphs (matching paragraph_count: 10)
    // and every OTHER locator in this fixture keeps resolving to a
    // short paragraph as usual. Each "あ" is 3 bytes, so 2000 of them
    // is 6000 bytes, and 4096 does not land on a character boundary
    // (4096 % 3 != 0), exercising the char-boundary floor.
    let long_paragraph = "あ".repeat(2000);
    let text = format!("p0\n\np1\n\n{long_paragraph}\n\np3\n\np4\n\np5\n\np6\n\np7\n\np8\n\np9");
    fs::write(dir.join("corpus/brewery.md"), &text).unwrap();
    let real_sha = crate::sha256::sha256_hex(text.as_bytes());

    let manifest_path = dir.join("manifest.json");
    let mut manifest: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["documents"][0]["path"] =
        serde_json::json!(dir.join("corpus/brewery.md").to_string_lossy());
    manifest["documents"][0]["sha256"] = serde_json::json!(real_sha);
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let lines = compute_differences_lines(&dir, true).expect("computes");
    let attribution = records_of_kind(&lines, "attribution_difference");
    let founded_in = attribution
        .iter()
        .find(|r| r["key"]["label"] == "founded in")
        .unwrap();
    let text = founded_in["locator"]["text"].as_str().unwrap();
    assert!(text.len() <= 4096);
    assert!(text.chars().all(|c| c == 'あ'), "no partial character");
    assert_eq!(founded_in["locator"]["text_truncated"], true);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn with_text_refuses_a_document_sha256_drift() {
    let dir = synthetic_two_model_results_dir("differences-with-text-drift");
    fs::create_dir_all(dir.join("corpus")).unwrap();
    fs::write(dir.join("corpus/brewery.md"), "drifted content").unwrap();
    // manifest.json still pins the placeholder "sha-brewery" — deliberately
    // left unmatched, standing in for a corpus that changed after the
    // results directory was created.
    let manifest_path = dir.join("manifest.json");
    let mut manifest: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["documents"][0]["path"] =
        serde_json::json!(dir.join("corpus/brewery.md").to_string_lossy());
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let error = compute_differences_lines(&dir, true).unwrap_err();
    assert!(error.contains("brewery.md"), "{error}");
    assert!(error.contains("sha-brewery"), "{error}");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn with_text_refuses_an_unreadable_corpus_file() {
    let dir = synthetic_two_model_results_dir("differences-with-text-unreadable");
    let manifest_path = dir.join("manifest.json");
    let mut manifest: Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).unwrap()).unwrap();
    manifest["documents"][0]["path"] =
        serde_json::json!(dir.join("corpus/does-not-exist.md").to_string_lossy());
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let error = compute_differences_lines(&dir, true).unwrap_err();
    assert!(error.contains("does-not-exist.md"), "{error}");
    let _ = fs::remove_dir_all(&dir);
}

// ============================== atomic write cleanup ==============================

#[test]
fn write_artifacts_cleans_up_staged_files_when_a_later_commit_fails() {
    // Persistence op sequence: stage(differences), stage(csv),
    // stage(json), commit(differences), commit(csv), commit(json) —
    // allowing the first 3 through and failing the 4th exercises the
    // case where every artifact was already staged and the first
    // commit is what fails.
    let dir = synthetic_results_dir("write-cleanup");
    let measurements = compute_measurements(&dir).expect("computes");

    crate::storage::fail_persistence_ops_after(3);
    let result = write_artifacts(&dir, &measurements, "differences-fixture\n");
    let past_end = crate::storage::clear_persistence_fault();

    assert!(result.is_err(), "the injected commit failure must surface");
    assert!(
        !past_end,
        "the fault must have fired for this test to be meaningful"
    );
    assert!(!dir.join("differences.jsonl").exists());
    assert!(!dir.join("measurements.csv").exists());
    assert!(!dir.join("measurements.json").exists());

    let leftover: Vec<String> = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains(".tmp"))
        .collect();
    assert!(
        leftover.is_empty(),
        "staged temp files were left behind: {leftover:?}"
    );

    let _ = fs::remove_dir_all(&dir);
}
