//! `taguru benchmark search` (issue #260, ADR 0003 §11) against a real
//! running server: per-model corpus import from a synthetic
//! `taguru benchmark extract` results directory, search over the
//! shared `eval.jsonl` question set, and `retrieval.json`'s lane,
//! recall@k/MRR, and pair-overlap output. No stub LLM is needed here
//! (unlike `benchmark.rs`'s own cluster) — `taguru benchmark search`
//! never spawns `taguru extract`; it only reads a finished results
//! directory's already-written batch files.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::support::*;

/// A synthetic two-model results directory, one run each, one document
/// each: `m1`'s document names "青嶺酒造" (the concept an `eval.jsonl`
/// case below expects) and lives at `corpus/brewery.md`; `m2`'s is a
/// different document at a different source that also contains the
/// query term "青嶺" but not the expected concept — so recall/MRR and
/// pair overlap have something real, asymmetric, and non-vacuous to
/// measure instead of two identical corpora.
fn write_results_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "taguru-benchmark-search-results-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("cells/m1/run01")).unwrap();
    std::fs::create_dir_all(dir.join("cells/m2/run01")).unwrap();

    std::fs::write(
        dir.join("cells/m1/run01/brewery.jsonl"),
        "{\"taguru_batch\":1,\"context\":\"sake\",\"source\":\"corpus/brewery.md\"}\n\
         {\"passage\":\"青嶺は青嶺酒造が造る銘柄です。\"}\n\
         {\"subject\":\"青嶺\",\"label\":\"醸造元\",\"object\":\"青嶺酒造\",\"weight\":1.0,\"paragraph\":0}\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("cells/m2/run01/history.jsonl"),
        "{\"taguru_batch\":1,\"context\":\"sake\",\"source\":\"corpus/history.md\"}\n\
         {\"passage\":\"青嶺という言葉の歴史は古い。\"}\n\
         {\"subject\":\"青嶺\",\"label\":\"由来\",\"object\":\"不明\",\"weight\":1.0,\"paragraph\":0}\n",
    )
    .unwrap();

    let manifest = json!({
        "taguru_benchmark_manifest": 1,
        "run_id": "run-search-1",
        "started_at": "2026-07-26T09:00:00Z",
        "finished_at": "2026-07-26T09:05:00Z",
        "taguru_version": "0.0.0",
        "sdk_versions": {},
        "harness": {},
        "extraction_settings": {"context": "sake"},
        "documents": [
            {"document_id": "brewery", "path": "corpus/brewery.md", "bytes": 10,
             "sha256": "sha-brewery", "paragraph_count": 1, "chunk_total": 1, "chunks": []},
            {"document_id": "history", "path": "corpus/history.md", "bytes": 10,
             "sha256": "sha-history", "paragraph_count": 1, "chunk_total": 1, "chunks": []},
        ],
        "models": [
            {"model_id": "m1", "model_name": "m1-model", "endpoint": "http://x",
             "digest": null, "quantization": null, "context_window": null,
             "structured_output_requested": "auto", "timeout_secs": 60,
             "provider_probe": {"attempted": [], "ok": true, "note": null}},
            {"model_id": "m2", "model_name": "m2-model", "endpoint": "http://x",
             "digest": null, "quantization": null, "context_window": null,
             "structured_output_requested": "auto", "timeout_secs": 60,
             "provider_probe": {"attempted": [], "ok": true, "note": null}},
        ],
        "cells": [
            {"cell_id": "m1.run01", "model_id": "m1", "run_index": 1,
             "runs_file": "runs/m1.run01.jsonl", "cell_dir": "cells/m1/run01",
             "structured_output_resolved": "json_schema",
             "started_at": "2026-07-26T09:00:01Z", "finished_at": "2026-07-26T09:04:00Z",
             "outcome": "complete"},
            {"cell_id": "m2.run01", "model_id": "m2", "run_index": 1,
             "runs_file": "runs/m2.run01.jsonl", "cell_dir": "cells/m2/run01",
             "structured_output_resolved": "json_schema",
             "started_at": "2026-07-26T09:00:01Z", "finished_at": "2026-07-26T09:04:00Z",
             "outcome": "complete"},
        ],
        "environment": {"os": "linux", "arch": "x86_64"},
    });
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();

    dir
}

/// One case: `expected_sources` names `m1`'s document (any paragraph),
/// `expected_concepts` names a string only `m1`'s passage text
/// contains — `m1` should score full recall/MRR, `m2` zero.
fn write_eval_file(dir: &Path) -> PathBuf {
    let path = dir.join("eval.jsonl");
    std::fs::write(
        &path,
        "{\"taguru_eval\":1,\"name\":\"sake retrieval cases\"}\n\
         {\"case_id\":\"brand-origin-001\",\"query\":\"青嶺\",\
         \"expected_sources\":[{\"source\":\"corpus/brewery.md\",\"relevance\":3}],\
         \"expected_concepts\":[\"青嶺酒造\"],\"options\":{\"limit\":10}}\n",
    )
    .unwrap();
    path
}

fn associations_count(server: &Server, context: &str) -> u64 {
    server.ok("GET", &format!("/contexts/{context}"), None)["stats"]["associations"]
        .as_u64()
        .unwrap()
}

#[test]
fn benchmark_search_builds_corpora_searches_them_and_writes_retrieval_json() {
    let server = Server::start("search-smoke");
    let results_dir = write_results_dir("smoke");
    let eval_path = write_eval_file(&results_dir);

    let run = || {
        run_cli(
            &[
                "benchmark",
                "search",
                "--eval",
                eval_path.to_str().unwrap(),
                "--url",
                &server.base,
                results_dir.to_str().unwrap(),
            ],
            &[],
        )
    };

    let (code, _stdout, stderr) = run();
    assert_eq!(code, 0, "{stderr}");

    // Both per-model corpora exist under prefix::model_id (the
    // extraction context "sake" recorded in manifest.json is the
    // default --context-prefix).
    let contexts = server.ok("GET", "/contexts", None);
    let names: Vec<&str> = contexts["contexts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"sake::m1"), "{names:?}");
    assert!(names.contains(&"sake::m2"), "{names:?}");

    let retrieval: Value =
        serde_json::from_str(&std::fs::read_to_string(results_dir.join("retrieval.json")).unwrap())
            .unwrap();
    assert_eq!(retrieval["taguru_benchmark_retrieval"], 1);
    assert_eq!(retrieval["run_id"], "run-search-1");
    assert_eq!(retrieval["corpus"]["m1"]["outcome"], "built", "{retrieval}");
    assert_eq!(retrieval["corpus"]["m2"]["outcome"], "built", "{retrieval}");

    let case = &retrieval["cases"][0];
    assert_eq!(case["case_id"], "brand-origin-001");

    // No embedding provider is configured on this server: the bm25
    // lane ran, the vector lane honestly did not — a documented
    // degradation (ADR 0003 §11), not a search failure.
    let m1 = &case["models"]["m1"];
    assert_eq!(m1["outcome"], "searched", "{m1}");
    assert_eq!(m1["plan"]["bm25"]["ran"], true, "{m1}");
    assert_eq!(m1["plan"]["vector"]["ran"], false, "{m1}");
    assert_eq!(
        m1["plan"]["vector"]["reason"], "no embedding provider is configured",
        "{m1}"
    );

    // m1's corpus carries both the expected source and the expected
    // concept; m2's carries neither.
    assert_eq!(m1["recall"]["recall_at_k"], 1.0, "{m1}");
    assert_eq!(m1["recall"]["mrr"], 1.0, "{m1}");
    let m2 = &case["models"]["m2"];
    assert_eq!(m2["outcome"], "searched", "{m2}");
    assert_eq!(m2["recall"]["recall_at_k"], 0.0, "{m2}");
    assert_eq!(m2["recall"]["mrr"], 0.0, "{m2}");

    // Both models found exactly one hit for "青嶺", but at disjoint
    // sources (each corpus holds one, different, document) — a real,
    // non-vacuous overlap value, not the `None` two empty hit lists
    // would produce.
    let pair = &case["pairs"]["m1__m2"];
    assert_eq!(pair["outcome"], "compared", "{pair}");
    assert_eq!(pair["jaccard"], 0.0, "{pair}");
    assert_eq!(pair["shared_hits"], 0, "{pair}");

    let before = associations_count(&server, "sake::m1");

    // Re-running against the same, unchanged results directory is
    // idempotent: import is a per-source create-or-replace, so a
    // second run must not double the association count.
    let (code2, _out2, stderr2) = run();
    assert_eq!(code2, 0, "{stderr2}");
    let after = associations_count(&server, "sake::m1");
    assert_eq!(before, after, "a re-run must not duplicate associations");

    let _ = std::fs::remove_dir_all(&results_dir);
}

#[test]
fn benchmark_search_rejects_a_run_the_manifest_never_recorded() {
    let server = Server::start("search-bad-run");
    let results_dir = write_results_dir("bad-run");
    let eval_path = write_eval_file(&results_dir);

    let (code, _stdout, stderr) = run_cli(
        &[
            "benchmark",
            "search",
            "--eval",
            eval_path.to_str().unwrap(),
            "--url",
            &server.base,
            "--run",
            "2",
            results_dir.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 2);
    assert!(stderr.contains("no cell recorded for run 2"), "{stderr}");

    let _ = std::fs::remove_dir_all(&results_dir);
}

#[test]
fn benchmark_search_skip_import_searches_existing_corpora_and_flags_a_missing_one() {
    let server = Server::start("search-skip-import");
    let results_dir = write_results_dir("skip-import");
    let eval_path = write_eval_file(&results_dir);

    // Pre-create only m1's corpus by hand, under the exact
    // prefix::model_id naming --skip-import expects to find; m2's is
    // left absent on purpose.
    server.ok(
        "PUT",
        "/contexts/sake::m1",
        Some(json!({"description": "pre-existing"})),
    );
    server.ok(
        "POST",
        "/contexts/sake::m1/sources",
        Some(json!({"passages": {"corpus/brewery.md": "青嶺は青嶺酒造が造る銘柄です。"}})),
    );

    let (code, _stdout, stderr) = run_cli(
        &[
            "benchmark",
            "search",
            "--eval",
            eval_path.to_str().unwrap(),
            "--url",
            &server.base,
            "--skip-import",
            results_dir.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "{stderr}");

    let retrieval: Value =
        serde_json::from_str(&std::fs::read_to_string(results_dir.join("retrieval.json")).unwrap())
            .unwrap();
    assert_eq!(
        retrieval["corpus"]["m1"]["outcome"], "search_only",
        "{retrieval}"
    );
    assert_eq!(
        retrieval["corpus"]["m2"]["outcome"], "failed",
        "{retrieval}"
    );
    let case = &retrieval["cases"][0];
    assert_eq!(case["models"]["m1"]["outcome"], "searched", "{case}");
    assert_eq!(case["models"]["m2"]["outcome"], "failed", "{case}");

    let _ = std::fs::remove_dir_all(&results_dir);
}

#[test]
fn benchmark_search_requires_eval_and_exactly_one_results_dir() {
    let output_missing_eval = run_cli(&["benchmark", "search", "out"], &[]);
    assert_eq!(output_missing_eval.0, 2);

    let output_missing_dir = run_cli(&["benchmark", "search", "--eval", "e.jsonl"], &[]);
    assert_eq!(output_missing_dir.0, 2);
}
