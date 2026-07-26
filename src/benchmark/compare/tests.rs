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

// ============================== nearest-rank ==============================

#[test]
fn nearest_rank_matches_the_adr_formula() {
    let one = vec![7.0];
    assert_eq!(nearest_rank(&one, 50), 7.0);
    assert_eq!(nearest_rank(&one, 99), 7.0);

    let two = vec![1.0, 2.0];
    assert_eq!(nearest_rank(&two, 50), 1.0, "ceil(50/100*2)=1 -> x[0]");
    assert_eq!(nearest_rank(&two, 90), 2.0, "ceil(90/100*2)=2 -> x[1]");

    let four = vec![1.0, 2.0, 3.0, 4.0];
    assert_eq!(nearest_rank(&four, 50), 2.0, "ceil(50/100*4)=2 -> x[1]");

    let ten: Vec<f64> = (1..=10).map(f64::from).collect();
    assert_eq!(nearest_rank(&ten, 90), 9.0, "ceil(90/100*10)=9 -> x[8]");

    let hundred: Vec<f64> = (1..=100).map(f64::from).collect();
    assert_eq!(
        nearest_rank(&hundred, 99),
        99.0,
        "ceil(99/100*100)=99 -> x[98]"
    );

    let all_same = vec![5.0, 5.0, 5.0, 5.0, 5.0];
    assert_eq!(nearest_rank(&all_same, 50), 5.0);
    assert_eq!(nearest_rank(&all_same, 99), 5.0);
}

#[test]
fn distribution_from_samples_sorts_first() {
    let d = Distribution::from_samples(vec![3.0, 1.0, 2.0]);
    assert_eq!(d.min, Some(1.0));
    assert_eq!(d.max, Some(3.0));
    assert_eq!(d.n, 3);
}

// ============================== n == 0 shapes ==============================

#[test]
fn distribution_with_zero_samples_serializes_all_stats_null() {
    let d = Distribution::from_samples(vec![]);
    let value = serde_json::to_value(&d).unwrap();
    let obj = value.as_object().unwrap();
    assert_eq!(obj["n"], 0);
    for key in ["min", "p50", "p90", "p99", "max", "mean", "sum"] {
        assert!(obj.contains_key(key), "{key} must be present");
        assert!(obj[key].is_null(), "{key} must be null, got {:?}", obj[key]);
    }
}

#[test]
fn ratio_with_zero_denominator_serializes_value_and_numerator_null() {
    let r = ratio_metric(0, 0);
    let value = serde_json::to_value(&r).unwrap();
    assert!(value["value"].is_null());
    assert!(value["numerator"].is_null());
    assert_eq!(value["n"], 0);
}

#[test]
fn ratio_with_a_denominator_computes_the_share() {
    let r = ratio_metric(3, 31);
    assert!((r.value.unwrap() - 3.0 / 31.0).abs() < 1e-12);
    assert_eq!(r.n, 31);
    assert_eq!(r.numerator, Some(3));
}

// ============================== ordering & banned keys ==============================

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
        let expected = measurements.definitions.get(metric).unwrap().unit;
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
            .caveat
            .as_ref()
            .unwrap()
            .contains("validation_issues")
    );
    let length_limited = d.get("attempt.state_rate.length_limited").unwrap();
    assert!(
        length_limited
            .caveat
            .as_ref()
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
    assert_eq!(d.n, 1, "one chunk, two item-stage retries summed");
    assert_eq!(d.sum, Some(5.0), "2.0 + 3.0, cross_chunk's 100.0 excluded");

    let retry_rate = attempt_rate_metrics(&refs);
    let MetricValue::Ratio(r) = &retry_rate["attempt.retry_rate"] else {
        panic!()
    };
    assert_eq!(
        r.numerator,
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
    assert_eq!(written.n, 3);
    assert_eq!(written.numerator, Some(1));
    let MetricValue::Ratio(failed) = &rates["document.failed_rate"] else {
        panic!()
    };
    assert_eq!(failed.numerator, Some(1));
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
    assert_eq!(d.n, 1, "the timeout attempt is excluded, not counted as 0");
    assert_eq!(d.sum, Some(100.0));

    let missing = attempt_rate_metrics(&refs);
    let MetricValue::Ratio(r) = &missing["attempt.provider_metadata_missing_rate"] else {
        panic!()
    };
    assert_eq!(r.numerator, Some(1));
    assert_eq!(r.n, 2);
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
    assert_eq!(
        parse_args(&["dir".to_string()]).unwrap(),
        PathBuf::from("dir")
    );
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
        latency.n, 2,
        "both the written and the timed-out attempt count"
    );

    let MetricValue::Ratio(written_rate) = &cell.metrics["document.written_rate"] else {
        panic!()
    };
    assert_eq!(written_rate.n, 2);
    assert_eq!(written_rate.numerator, Some(1));

    let model = &measurements.models["m"];
    let MetricValue::Ratio(complete_rate) = &model["cell.complete_rate"] else {
        panic!()
    };
    assert_eq!(complete_rate.value, Some(1.0));

    let brewery_run01 = &measurements.documents["m"]["brewery"]["run01"];
    let MetricValue::Count(associations) = &brewery_run01["extraction.associations"] else {
        panic!()
    };
    assert_eq!(associations.value, Some(2.0));
    let MetricValue::Count(subjects) = &brewery_run01["extraction.subjects_distinct"] else {
        panic!()
    };
    assert_eq!(subjects.value, Some(1.0), "one distinct subject: 'beer co'");

    let sake_run01 = &measurements.documents["m"]["sake"]["run01"];
    let MetricValue::Count(sake_associations) = &sake_run01["extraction.associations"] else {
        panic!()
    };
    assert_eq!(
        sake_associations.value, None,
        "sake failed — no associations"
    );
    assert_eq!(sake_associations.n, 0);

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

// ============================== atomic write cleanup ==============================

#[test]
fn write_measurements_cleans_up_staged_files_when_a_later_commit_fails() {
    // Persistence op sequence: stage(csv), stage(json), commit(csv),
    // commit(json) — allowing the first 2 through and failing the 3rd
    // exercises the exact case where `staged_csv` was already staged
    // and its own commit is what fails.
    let dir = synthetic_results_dir("write-cleanup");
    let measurements = compute_measurements(&dir).expect("computes");

    crate::storage::fail_persistence_ops_after(2);
    let result = write_measurements(&dir, &measurements);
    let past_end = crate::storage::clear_persistence_fault();

    assert!(result.is_err(), "the injected commit failure must surface");
    assert!(
        !past_end,
        "the fault must have fired for this test to be meaningful"
    );
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
