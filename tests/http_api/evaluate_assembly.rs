//! #308 (ADR 0006 §14): `taguru evaluate --assembly` and the
//! equal-budget comparison it exists to make honest — against a real
//! running server, offline and deterministic by default (no
//! `TAGURU_EMBED_*`/`TAGURU_RERANK_*` unless a test says so
//! explicitly).
//!
//! `tests/http_api/evaluate_fixture.rs` already covers the `baseline`
//! passage/structural/citation lanes in depth (#278); this file is
//! deliberately scoped to what #308 itself adds: the `--assembly`
//! flag, `--max-items`/`--max-bytes`/`--max-tokens`'s equal-budget
//! accounting in both modes, `diversity.sources`, an --rerank run's
//! degrade path, run-to-run reproducibility, and `evaluate compare`'s
//! new budget-mismatch warning. It does not re-prove
//! `POST /contexts/{name}/evidence`'s own selection algorithm
//! (dedup/contradiction/near-duplicate/diversity-tiering) — `src/api/
//! evidence/select.rs` and `tests/http_api/evidence.rs` already do
//! that at the endpoint level.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::support::*;

fn eval_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "taguru-evaluate-assembly-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("eval scratch dir must be creatable");
    dir
}

fn write_eval_file(dir: &Path, contents: &str) -> PathBuf {
    let path = dir.join("eval.jsonl");
    std::fs::write(&path, contents).unwrap();
    path
}

fn write_thresholds(dir: &Path, contents: &str) -> PathBuf {
    let path = dir.join("thresholds.json");
    std::fs::write(&path, contents).unwrap();
    path
}

/// Three passage sources plus one association, deliberately shaped so
/// `baseline` (passage-only) and `assembly` (fused graph+passage) can
/// never see the same set of distinct sources for the same query,
/// regardless of ranking or budget-tiering subtleties:
///
/// - `corpus/kura-a.md`/`corpus/kura-b.md` both contain the query term
///   "青嶺酒造" and so are ordinary BM25 passage hits in either mode.
/// - `corpus/kura-c.md`'s text shares no term with "青嶺酒造" at all —
///   BM25 never surfaces it as a passage hit in `baseline` mode — but
///   it is the association's own citation locator, so `assembly`
///   mode's always-run `activate` lane admits it as a third, distinct
///   source no passage-only search can ever reach.
/// - `corpus/unrelated.md` shares no term with the query either, and
///   exists purely as extra rank-order filler for the tiny-budget
///   test.
fn seed_assembly_corpus(server: &Server, context: &str) {
    server.ok(
        "PUT",
        &format!("/contexts/{context}"),
        Some(json!({"description": "#308 equal-budget fixture"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{context}/sources"),
        Some(json!({"passages": {
            "corpus/kura-a.md": "青嶺酒造は雲居県霧沢町の蔵元である。",
            "corpus/kura-b.md": "青嶺酒造の仕込み水は霧沢川の伏流水を使う。",
            "corpus/kura-c.md": "杜氏見習いの日誌帳を几帳面につける。",
            "corpus/unrelated.md": "梅雨明けの空に入道雲が広がった。",
        }})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{context}/associations"),
        Some(json!([
            {"subject": "青嶺酒造", "label": "杜氏", "object": "高瀬", "weight": 1.0,
             "source": "corpus/kura-c.md", "paragraph": 0},
        ])),
    );
}

/// One case: `expected_sources` names the two passage-reachable
/// sources (graded so `recall.recall_at_k`/`mrr`/`ndcg` are all
/// non-trivial), and `expected_citations` exercises the citation lane
/// (unchanged by `--assembly` — it always calls
/// `POST /contexts/{name}/citations` directly) so
/// `citations.recall`/`citations.locator_validity` compute in both
/// modes too.
fn write_shared_eval(dir: &Path) -> PathBuf {
    let lines = [
        r#"{"taguru_eval":1,"name":"evaluate assembly fixture: #308 equal budget"}"#,
        r#"{"case_id":"diversity-001","query":"青嶺酒造","cues":["青嶺酒造"],"expected_sources":[{"source":"corpus/kura-a.md","relevance":3},{"source":"corpus/kura-b.md","relevance":2}],"expected_citations":[{"source":"corpus/kura-a.md","paragraph":0}]}"#,
    ];
    write_eval_file(dir, &(lines.join("\n") + "\n"))
}

/// Runs `taguru evaluate` (or, with `--assembly` among `extra`, the
/// `--assembly` mode) and returns `(exit_code, stderr, parsed
/// evaluation.json)`.
fn run_evaluate(
    base: &str,
    eval_path: &Path,
    context: &str,
    out_path: &Path,
    extra: &[&str],
) -> (i32, String, Value) {
    let mut args: Vec<String> = vec![
        "evaluate".to_string(),
        "--eval".to_string(),
        eval_path.to_str().unwrap().to_string(),
        "--context".to_string(),
        context.to_string(),
        "--url".to_string(),
        base.to_string(),
        "--out".to_string(),
        out_path.to_str().unwrap().to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
    let (code, _stdout, stderr) = run_cli(&arg_refs, &[]);
    let evaluation: Value = if out_path.exists() {
        serde_json::from_str(&std::fs::read_to_string(out_path).unwrap()).unwrap_or(Value::Null)
    } else {
        Value::Null
    };
    (code, stderr, evaluation)
}

fn case<'a>(evaluation: &'a Value, id: &str) -> &'a Value {
    evaluation["cases"]
        .as_array()
        .unwrap_or_else(|| panic!("no cases array: {evaluation}"))
        .iter()
        .find(|c| c["case_id"] == id)
        .unwrap_or_else(|| panic!("missing case '{id}': {evaluation}"))
}

// ============================ Equal-budget accounting ============================

#[test]
fn equal_budget_baseline_and_assembly_record_the_same_limits() {
    let server = Server::start("evaluate-assembly-equal-budget");
    seed_assembly_corpus(&server, "sake");
    let dir = eval_dir("equal-budget");
    let eval_path = write_shared_eval(&dir);

    let baseline_out = dir.join("baseline.json");
    let (code, stderr, baseline) = run_evaluate(
        &server.base,
        &eval_path,
        "sake",
        &baseline_out,
        &[
            "--max-items",
            "3",
            "--max-bytes",
            "4096",
            "--max-tokens",
            "200",
        ],
    );
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(baseline["inputs"]["mode"], "baseline", "{baseline}");
    assert_eq!(
        baseline["inputs"]["budget"],
        json!({"max_items": 3, "max_bytes": 4096, "max_tokens": 200}),
        "{baseline}"
    );
    let baseline_budget = &case(&baseline, "diversity-001")["budget"];
    assert_eq!(
        baseline_budget["limits"]["max_items"], 3,
        "{baseline_budget}"
    );

    let assembly_out = dir.join("assembly.json");
    let (code, stderr, assembly) = run_evaluate(
        &server.base,
        &eval_path,
        "sake",
        &assembly_out,
        &[
            "--assembly",
            "--max-items",
            "3",
            "--max-bytes",
            "4096",
            "--max-tokens",
            "200",
        ],
    );
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(assembly["inputs"]["mode"], "assembly", "{assembly}");
    assert_eq!(
        assembly["inputs"]["budget"],
        json!({"max_items": 3, "max_bytes": 4096, "max_tokens": 200}),
        "{assembly}"
    );
    let assembly_budget = &case(&assembly, "diversity-001")["budget"];
    assert_eq!(
        assembly_budget["limits"]["max_items"], 3,
        "{assembly_budget}"
    );
}

/// Without any `--max-*` flag, neither mode truncates and neither
/// mode's artifact carries a `budget` block at all — the same
/// evaluation.json this run would have produced before #308.
#[test]
fn without_a_budget_flag_neither_mode_truncates_or_reports_a_budget_block() {
    let server = Server::start("evaluate-assembly-no-budget");
    seed_assembly_corpus(&server, "sake");
    let dir = eval_dir("no-budget");
    let eval_path = write_shared_eval(&dir);

    let baseline_out = dir.join("baseline.json");
    let (code, stderr, baseline) =
        run_evaluate(&server.base, &eval_path, "sake", &baseline_out, &[]);
    assert_eq!(code, 0, "{stderr}");
    assert!(baseline["inputs"].get("budget").is_none(), "{baseline}");
    assert!(
        case(&baseline, "diversity-001").get("budget").is_none(),
        "{baseline}"
    );

    let assembly_out = dir.join("assembly.json");
    let (code, stderr, assembly) = run_evaluate(
        &server.base,
        &eval_path,
        "sake",
        &assembly_out,
        &["--assembly"],
    );
    assert_eq!(code, 0, "{stderr}");
    // Assembly always sends *some* budget to POST .../evidence (the
    // endpoint has no unbudgeted mode, ADR 0006 §8) — but that is the
    // server's own default, not a caller-requested ceiling, so
    // InputsBlock.budget stays populated (it always is in assembly
    // mode) while `--max-items`/`--max-bytes`/`--max-tokens` were
    // never given.
    assert_eq!(assembly["inputs"]["mode"], "assembly", "{assembly}");
    assert_eq!(
        assembly["inputs"]["budget"],
        json!({"max_items": 40, "max_bytes": 65536, "max_tokens": 4000}),
        "{assembly}"
    );
}

// ============================== Source diversity ==============================

/// `baseline` can never surface `corpus/kura-c.md` (no term overlap
/// with the query) — its diversity is capped at the two passage
/// sources. `assembly` fuses in the association's own citation locator
/// via the always-run `activate` lane, reaching a third, distinct
/// source no passage-only search could ever find. Generous (unset)
/// budget on both sides, so the comparison is about what each mode can
/// *reach*, not which candidate a tight budget happened to keep.
#[test]
fn assembly_reaches_a_source_diversity_baseline_structurally_cannot() {
    let server = Server::start("evaluate-assembly-diversity");
    seed_assembly_corpus(&server, "sake");
    let dir = eval_dir("diversity");
    let eval_path = write_shared_eval(&dir);

    let baseline_out = dir.join("baseline.json");
    let (code, stderr, baseline) =
        run_evaluate(&server.base, &eval_path, "sake", &baseline_out, &[]);
    assert_eq!(code, 0, "{stderr}");
    let baseline_diversity = case(&baseline, "diversity-001")["diversity_sources"]
        .as_u64()
        .unwrap_or_else(|| panic!("{baseline}"));
    assert!(baseline_diversity <= 2, "{baseline}");

    let assembly_out = dir.join("assembly.json");
    let (code, stderr, assembly) = run_evaluate(
        &server.base,
        &eval_path,
        "sake",
        &assembly_out,
        &["--assembly"],
    );
    assert_eq!(code, 0, "{stderr}");
    let assembly_diversity = case(&assembly, "diversity-001")["diversity_sources"]
        .as_u64()
        .unwrap_or_else(|| panic!("{assembly}"));
    assert_eq!(assembly_diversity, 3, "{assembly}");
    assert!(
        assembly_diversity > baseline_diversity,
        "assembly ({assembly_diversity}) must exceed baseline ({baseline_diversity})"
    );

    let diversity_metric = &assembly["metrics"]["diversity.sources"];
    assert_eq!(diversity_metric["n"], 1, "{diversity_metric}");
}

// ============================ Regression gate (thresholds) ============================

#[test]
fn assembly_passes_a_checked_in_recall_and_diversity_threshold() {
    let server = Server::start("evaluate-assembly-thresholds");
    seed_assembly_corpus(&server, "sake");
    let dir = eval_dir("thresholds");
    let eval_path = write_shared_eval(&dir);
    let thresholds_path = write_thresholds(
        &dir,
        r#"{"taguru_evaluate_thresholds":1,
            "aggregate":{
                "recall.recall_at_k":{"min":0.5},
                "citations.recall":{"min":1.0},
                "diversity.sources":{"min":3}
            }}"#,
    );

    let out_path = dir.join("assembly.json");
    let (code, stderr, evaluation) = run_evaluate(
        &server.base,
        &eval_path,
        "sake",
        &out_path,
        &[
            "--assembly",
            "--thresholds",
            thresholds_path.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(evaluation["thresholds"]["passed"], true, "{evaluation}");
}

/// The same thresholds file's `diversity.sources: {"min": 3}` bound —
/// satisfiable by `assembly`, structurally never by `baseline` (capped
/// at 2, see `assembly_reaches_a_source_diversity_baseline_structurally_cannot`)
/// — fails the gate (exit 3) against `baseline`, proving the threshold
/// actually discriminates between the two configurations rather than
/// passing either way.
#[test]
fn the_same_diversity_threshold_fails_the_gate_against_baseline() {
    let server = Server::start("evaluate-assembly-thresholds-baseline");
    seed_assembly_corpus(&server, "sake");
    let dir = eval_dir("thresholds-baseline");
    let eval_path = write_shared_eval(&dir);
    let thresholds_path = write_thresholds(
        &dir,
        r#"{"taguru_evaluate_thresholds":1,
            "aggregate":{"diversity.sources":{"min":3}}}"#,
    );

    let out_path = dir.join("baseline.json");
    let (code, stderr, evaluation) = run_evaluate(
        &server.base,
        &eval_path,
        "sake",
        &out_path,
        &["--thresholds", thresholds_path.to_str().unwrap()],
    );
    assert_eq!(code, 3, "{stderr}");
    assert_eq!(evaluation["thresholds"]["passed"], false, "{evaluation}");
}

// ============================== Reproducibility ==============================

#[test]
fn assembly_is_reproducible_across_repeated_runs() {
    let server = Server::start("evaluate-assembly-reproducible");
    seed_assembly_corpus(&server, "sake");
    let dir = eval_dir("reproducible");
    let eval_path = write_shared_eval(&dir);

    let first_out = dir.join("first.json");
    let (code, stderr, first) = run_evaluate(
        &server.base,
        &eval_path,
        "sake",
        &first_out,
        &["--assembly", "--max-items", "3"],
    );
    assert_eq!(code, 0, "{stderr}");

    let second_out = dir.join("second.json");
    let (code, stderr, second) = run_evaluate(
        &server.base,
        &eval_path,
        "sake",
        &second_out,
        &["--assembly", "--max-items", "3"],
    );
    assert_eq!(code, 0, "{stderr}");

    // `generated_at`, `inputs.out`, and every real wall-clock
    // `latency_ms`/`latency.*` measurement are the only fields
    // expected to differ between two otherwise-identical runs — ADR
    // 0006 §9's own determinism invariant (I3) is what makes
    // everything else — selection order, budget accounting, admitted
    // items — byte-identical JSON.
    assert_eq!(
        without_latency(first["cases"].clone()),
        without_latency(second["cases"].clone()),
        "{first}\n---\n{second}"
    );
    assert_eq!(
        without_latency(first["metrics"].clone()),
        without_latency(second["metrics"].clone()),
        "{first}\n---\n{second}"
    );
}

/// Recursively drops every `latency_ms` key and blanks every top-level
/// `metrics` key starting with `"latency."` — the only fields a
/// determinism comparison must ignore, since they measure real
/// wall-clock time and are never claimed to repeat (see
/// `assembly_is_reproducible_across_repeated_runs`).
fn without_latency(mut value: Value) -> Value {
    strip_latency(&mut value);
    value
}

fn strip_latency(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.remove("latency_ms");
            for (key, nested) in map.iter_mut() {
                if key.starts_with("latency.") {
                    *nested = Value::Null;
                } else {
                    strip_latency(nested);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                strip_latency(item);
            }
        }
        _ => {}
    }
}

// ================================ Tiny budget ================================

#[test]
fn a_tiny_budget_reports_every_omission_in_both_modes() {
    let server = Server::start("evaluate-assembly-tiny-budget");
    seed_assembly_corpus(&server, "sake");
    let dir = eval_dir("tiny-budget");
    let eval_path = write_shared_eval(&dir);

    let baseline_out = dir.join("baseline.json");
    let (code, stderr, baseline) = run_evaluate(
        &server.base,
        &eval_path,
        "sake",
        &baseline_out,
        &["--max-items", "1"],
    );
    assert_eq!(code, 0, "{stderr}");
    let baseline_budget = &case(&baseline, "diversity-001")["budget"];
    assert_eq!(baseline_budget["items_used"], 1, "{baseline_budget}");
    assert!(
        baseline_budget["omitted_total"].as_u64().unwrap_or(0) >= 1,
        "{baseline_budget}"
    );

    let assembly_out = dir.join("assembly.json");
    let (code, stderr, assembly) = run_evaluate(
        &server.base,
        &eval_path,
        "sake",
        &assembly_out,
        &["--assembly", "--max-items", "1"],
    );
    assert_eq!(code, 0, "{stderr}");
    let assembly_case = case(&assembly, "diversity-001");
    let assembly_budget = &assembly_case["budget"];
    assert_eq!(assembly_budget["items_used"], 1, "{assembly_budget}");
    assert!(
        assembly_budget["omitted_total"].as_u64().unwrap_or(0) >= 1,
        "{assembly_budget}"
    );
    let omitted_by_reason = &assembly_case["evidence"]["omitted_by_reason"];
    assert!(
        omitted_by_reason
            .as_object()
            .map(|m| !m.is_empty())
            .unwrap_or(false),
        "{assembly_case}"
    );
}

// ================================ Reranker degrade ================================

/// A configured-but-unreachable reranker (the same deliberately-dead-
/// port trick `tests/http_api/evidence.rs` uses) degrades every
/// `--rerank` case to the deterministic order, `reason:
/// "provider_error"` — and `rerank.ran`'s aggregate reports the
/// degrade as `0.0`, `n: 1`, so a CI reader can see the fallback rate
/// without opening the per-case detail.
#[test]
fn a_failing_reranker_degrades_and_the_degrade_rate_is_reported() {
    let server = Server::start_with_env(
        "evaluate-assembly-rerank-degrade",
        &[
            ("TAGURU_RERANK_URL", "http://127.0.0.1:9"),
            ("TAGURU_RERANK_MODEL", "stub-reranker"),
            ("TAGURU_RERANK_TIMEOUT_SECS", "2"),
        ],
    );
    seed_assembly_corpus(&server, "sake");
    let dir = eval_dir("rerank-degrade");
    let eval_path = write_shared_eval(&dir);

    let out_path = dir.join("assembly.json");
    let (code, stderr, evaluation) = run_evaluate(
        &server.base,
        &eval_path,
        "sake",
        &out_path,
        &["--assembly", "--rerank", "stub-reranker"],
    );
    assert_eq!(code, 0, "{stderr}");
    let reranker = &case(&evaluation, "diversity-001")["evidence"]["reranker"];
    assert_eq!(reranker["configured"], true, "{reranker}");
    assert_eq!(reranker["ran"], false, "{reranker}");
    assert!(reranker["reason"].is_string(), "{reranker}");

    let rerank_ran = &evaluation["metrics"]["rerank.ran"];
    assert_eq!(rerank_ran["n"], 1, "{rerank_ran}");
    assert_eq!(rerank_ran["value"], 0.0, "{rerank_ran}");
}

/// `--rerank` requires a server-side provider it never got configured
/// with here: `--assembly` alone, no `TAGURU_RERANK_*`, degrades with
/// `reason` absent entirely (`RerankerPlan::not_requested` covers both
/// "never asked" and "no provider" the same shape) — the fully
/// deterministic ADR 0006 §14 configuration 2, and proof the whole run
/// completed with no provider of any kind, embedding or reranker.
#[test]
fn an_offline_run_records_no_embedding_provider_and_no_reranker() {
    let server = Server::start("evaluate-assembly-offline");
    seed_assembly_corpus(&server, "sake");
    let dir = eval_dir("offline");
    let eval_path = write_shared_eval(&dir);

    let out_path = dir.join("assembly.json");
    let (code, stderr, evaluation) =
        run_evaluate(&server.base, &eval_path, "sake", &out_path, &["--assembly"]);
    assert_eq!(code, 0, "{stderr}");
    assert!(
        evaluation["corpus"]["embeddings"]["provider_model"].is_null(),
        "{evaluation}"
    );
    let reranker = &case(&evaluation, "diversity-001")["evidence"]["reranker"];
    assert_eq!(
        reranker,
        &json!({"configured": false, "ran": false}),
        "{reranker}"
    );
}

// ================================ compare ================================

/// `taguru evaluate compare` still runs to completion (exit 0) on a
/// `baseline`/`assembly` pair under different budgets — the mismatch
/// is a warning, never a refusal (ADR 0002 §10) — but names the
/// mismatch explicitly, so a reader does not mistake it for an
/// apples-to-apples comparison.
#[test]
fn compare_warns_when_the_two_runs_used_different_budgets() {
    let server = Server::start("evaluate-assembly-compare-budget-warn");
    seed_assembly_corpus(&server, "sake");
    let dir = eval_dir("compare-budget-warn");
    let eval_path = write_shared_eval(&dir);

    let base_out = dir.join("base.json");
    let (code, stderr, _) = run_evaluate(&server.base, &eval_path, "sake", &base_out, &[]);
    assert_eq!(code, 0, "{stderr}");

    let head_out = dir.join("head.json");
    let (code, stderr, _) = run_evaluate(
        &server.base,
        &eval_path,
        "sake",
        &head_out,
        &["--assembly", "--max-items", "2"],
    );
    assert_eq!(code, 0, "{stderr}");

    let changes_out = dir.join("changes.jsonl");
    let (code, _stdout, stderr) = run_cli(
        &[
            "evaluate",
            "compare",
            base_out.to_str().unwrap(),
            head_out.to_str().unwrap(),
            "--out",
            changes_out.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "{stderr}");
    assert!(stderr.contains("budget differs"), "{stderr}");
}
