//! `taguru evaluate` (issue #273, ADR 0004) against a real running
//! server: the passage lane's plan/hits echo, the structural lane's
//! resolve/query coverage and its multi-candidate policy, corpus
//! revision bracketing, and the source preflight — the same style
//! `benchmark_search.rs` uses for #260's own harness.

use std::path::{Path, PathBuf};

use serde_json::{Value, json};

use crate::support::*;

fn eval_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("taguru-evaluate-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("eval scratch dir must be creatable");
    dir
}

fn write_eval_file(dir: &Path, contents: &str) -> PathBuf {
    let path = dir.join("eval.jsonl");
    std::fs::write(&path, contents).unwrap();
    path
}

/// One passage naming "青嶺酒造" (matched by the coverage cue "青嶺"
/// via containment) and one association whose three positions —
/// "青嶺酒造", "醸造元", "蔵元" — are each the ONLY stored concept/label
/// of that spelling, so every resolve call pins to exactly one
/// candidate: a clean, non-ambiguous fixture.
fn seed_context(server: &Server, context: &str) {
    server.ok(
        "PUT",
        &format!("/contexts/{context}"),
        Some(json!({"description": "d"})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{context}/sources"),
        Some(json!({"passages": {"corpus/brewery.md": "青嶺は青嶺酒造が造る銘柄です。"}})),
    );
    server.ok(
        "POST",
        &format!("/contexts/{context}/associations"),
        Some(json!([
            {"subject": "青嶺酒造", "label": "醸造元", "object": "蔵元",
             "weight": 1.0, "source": "corpus/brewery.md", "paragraph": 0},
        ])),
    );
}

/// [`seed_context`], authenticated — for the read-only-key completion
/// test, where the server enforces bearer auth from boot.
fn seed_context_with_token(server: &Server, context: &str, token: &str) {
    let call = |method: &str, path: String, body: Value| {
        let (status, parsed) = server.call_with_token(method, &path, Some(body), Some(token));
        assert_eq!(status, 200, "{method} {path} -> {parsed}");
    };
    call(
        "PUT",
        format!("/contexts/{context}"),
        json!({"description": "d"}),
    );
    call(
        "POST",
        format!("/contexts/{context}/sources"),
        json!({"passages": {"corpus/brewery.md": "青嶺は青嶺酒造が造る銘柄です。"}}),
    );
    call(
        "POST",
        format!("/contexts/{context}/associations"),
        json!([
            {"subject": "青嶺酒造", "label": "醸造元", "object": "蔵元",
             "weight": 1.0, "source": "corpus/brewery.md", "paragraph": 0},
        ]),
    );
}

/// One case: `expected_sources`/`expected_concepts` exercise the
/// passage lane's coverage echo, `expected_associations` exercises the
/// structural lane's query pin, `expected_citations` exercises the
/// citation lane (#275) — matching [`seed_context`]'s fixture. The
/// citation's `quote` is a substring of the fixture's own paragraph 0
/// text ("青嶺は青嶺酒造が造る銘柄です。"), so both citation recall
/// (the (source, paragraph) is among the passage lane's own hits) and
/// locator validity (the citations endpoint resolves it and the quote
/// matches) are expected to be perfect for this case.
fn write_smoke_eval(dir: &Path) -> PathBuf {
    write_eval_file(
        dir,
        "{\"taguru_eval\":1,\"name\":\"evaluate smoke\"}\n\
         {\"case_id\":\"brand-origin-001\",\"query\":\"青嶺\",\"cues\":[\"青嶺\"],\
         \"expected_sources\":[{\"source\":\"corpus/brewery.md\",\"relevance\":3}],\
         \"expected_concepts\":[\"青嶺酒造\"],\
         \"expected_associations\":[{\"subject\":\"青嶺酒造\",\"label\":\"醸造元\",\"object\":\"蔵元\"}],\
         \"expected_citations\":[{\"source\":\"corpus/brewery.md\",\"paragraph\":0,\"quote\":\"青嶺酒造\"}],\
         \"options\":{\"limit\":10}}\n",
    )
}

#[test]
fn evaluate_runs_both_lanes_and_writes_evaluation_json() {
    let server = Server::start("evaluate-smoke");
    seed_context(&server, "sake");
    let dir = eval_dir("smoke");
    let eval_path = write_smoke_eval(&dir);
    let out_path = dir.join("evaluation.json");

    let (code, _stdout, stderr) = run_cli(
        &[
            "evaluate",
            "--eval",
            eval_path.to_str().unwrap(),
            "--context",
            "sake",
            "--url",
            &server.base,
            "--out",
            out_path.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "{stderr}");
    // report-only announcement (ADR 0004 §5) — no --thresholds was given.
    assert!(stderr.contains("report-only"), "{stderr}");

    let evaluation: Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).unwrap()).unwrap();
    assert_eq!(evaluation["taguru_evaluation"], 1);
    assert_eq!(evaluation["thresholds"], Value::Null, "{evaluation}");
    assert_eq!(evaluation["corpus"]["stable"], true, "{evaluation}");
    assert_eq!(
        evaluation["corpus"]["revision_before"], evaluation["corpus"]["revision_after"],
        "{evaluation}"
    );
    // Never the literal --url (ADR 0004 §11): no userinfo was given
    // here, but the scheme+host+port shape is what matters.
    assert!(
        evaluation["inputs"]["url"]
            .as_str()
            .unwrap()
            .starts_with("http://"),
        "{evaluation}"
    );

    let case = &evaluation["cases"][0];
    assert_eq!(case["case_id"], "brand-origin-001");
    assert_eq!(case["passage"]["outcome"], "searched", "{case}");
    // No embedding provider is configured on this server: the bm25
    // lane ran, the vector lane honestly did not.
    assert_eq!(
        case["passage"]["plan"]["lanes"]["vector"]["ran"], false,
        "{case}"
    );
    assert_eq!(
        case["passage"]["plan"]["lanes"]["vector"]["reason"], "no embedding provider is configured",
        "{case}"
    );
    assert!(case["passage"]["hits"][0].get("text").is_none(), "{case}");

    let assoc = &case["structural"]["associations"][0];
    assert_eq!(assoc["subject"]["outcome"], "resolved", "{assoc}");
    assert_eq!(assoc["query"]["outcome"], "queried", "{assoc}");
    assert_eq!(assoc["query"]["total"], 1, "{assoc}");

    // #274: recall@k/MRR/nDCG against expected_sources, and structural
    // coverage against expected_concepts/expected_associations — the
    // fixture's one hit and one queried association both fully satisfy
    // this case's single expectation of each kind.
    assert_eq!(case["recall"]["recall_at_k"], 1.0, "{case}");
    assert_eq!(case["recall"]["mrr"], 1.0, "{case}");
    assert_eq!(case["recall"]["ndcg"], 1.0, "{case}");
    assert_eq!(case["coverage"]["concepts"]["value"], 1.0, "{case}");
    assert_eq!(case["coverage"]["associations"]["value"], 1.0, "{case}");
    assert_eq!(case["lane_cross"]["structural_hit"], true, "{case}");
    assert_eq!(case["lane_cross"]["passage_hit"], true, "{case}");
    assert!(case["missed"].as_array().unwrap().is_empty(), "{case}");

    // #275: citation recall (the locator is among the passage lane's
    // own hits) and locator validity (the citations endpoint resolves
    // it and the declared quote matches) — never merged into one score.
    assert_eq!(case["citations"]["recall"]["value"], 1.0, "{case}");
    assert_eq!(case["citations"]["validity"]["value"], 1.0, "{case}");
    let check = &case["citations"]["checks"][0];
    assert_eq!(check["source"], "corpus/brewery.md", "{check}");
    assert_eq!(check["paragraph"], 0, "{check}");
    assert_eq!(check["served"], true, "{check}");
    assert_eq!(check["outcome"], "resolved", "{check}");
    assert_eq!(check["quote"]["matched"], true, "{check}");
    // ADR 0004 §11: never the served paragraph body, even alongside a
    // matched quote.
    assert!(check.get("text").is_none(), "{check}");

    let metrics = &evaluation["metrics"];
    assert_eq!(metrics["recall.recall_at_k"]["n"], 1, "{metrics}");
    assert_eq!(metrics["lanes.both"]["numerator"], 1, "{metrics}");
    assert_eq!(metrics["lanes.both"]["n"], 1, "{metrics}");
    assert_eq!(metrics["citations.recall"]["n"], 1, "{metrics}");
    assert_eq!(metrics["citations.resolved"]["numerator"], 1, "{metrics}");
    assert_eq!(
        metrics["citations.quote_match"]["numerator"], 1,
        "{metrics}"
    );
    assert_eq!(
        evaluation["matching"]["normalization"], "taguru::context::normalize_entry",
        "{evaluation}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn evaluate_completes_on_a_read_only_api_key() {
    let server = Server::start_with_env(
        "evaluate-readonly",
        &[
            ("TAGURU_API_TOKENS", "admin:tok-admin,ci:tok-read"),
            ("TAGURU_KEY_SCOPES", r#"{"ci": "read"}"#),
        ],
    );
    seed_context_with_token(&server, "sake", "tok-admin");
    let dir = eval_dir("readonly");
    let eval_path = write_smoke_eval(&dir);
    let out_path = dir.join("evaluation.json");

    let (code, _stdout, stderr) = run_cli(
        &[
            "evaluate",
            "--eval",
            eval_path.to_str().unwrap(),
            "--context",
            "sake",
            "--url",
            &server.base,
            "--out",
            out_path.to_str().unwrap(),
        ],
        &[("TAGURU_API_TOKEN", "tok-read")],
    );
    assert_eq!(code, 0, "{stderr}");
    assert!(out_path.exists(), "evaluation.json must be written");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn evaluate_refuses_an_expected_source_the_context_does_not_carry() {
    let server = Server::start("evaluate-missing-source");
    seed_context(&server, "sake");
    let dir = eval_dir("missing-source");
    let eval_path = write_eval_file(
        &dir,
        "{\"taguru_eval\":1,\"name\":\"missing source\"}\n\
         {\"case_id\":\"ghost-001\",\"query\":\"青嶺\",\
         \"expected_sources\":[{\"source\":\"corpus/does-not-exist.md\",\"relevance\":1}]}\n",
    );
    let out_path = dir.join("evaluation.json");

    let (code, _stdout, stderr) = run_cli(
        &[
            "evaluate",
            "--eval",
            eval_path.to_str().unwrap(),
            "--context",
            "sake",
            "--url",
            &server.base,
            "--out",
            out_path.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 2, "{stderr}");
    assert!(stderr.contains("ghost-001"), "{stderr}");
    assert!(stderr.contains("corpus/does-not-exist.md"), "{stderr}");
    assert!(
        !out_path.exists(),
        "evaluation.json must not be written on a preflight failure"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn evaluate_marks_an_ambiguous_position_and_never_calls_query_for_it() {
    let server = Server::start("evaluate-ambiguous");
    // Same lookalike pair `resolve_match.rs` uses: "京都" is both a
    // stored concept in its own right and a substring of "東京都" —
    // resolving cue "京都" surfaces both at the lexical tier.
    server.ok("PUT", "/contexts/looks", Some(json!({"description": "d"})));
    server.ok(
        "POST",
        "/contexts/looks/associations",
        Some(json!([
            {"subject": "京都", "label": "位置", "object": "関西", "weight": 1.0},
            {"subject": "東京都", "label": "分類", "object": "日本の首都", "weight": 1.0},
        ])),
    );
    let dir = eval_dir("ambiguous");
    let eval_path = write_eval_file(
        &dir,
        "{\"taguru_eval\":1,\"name\":\"ambiguous subject\"}\n\
         {\"case_id\":\"kyoto-001\",\"query\":\"京都\",\
         \"expected_associations\":[{\"subject\":\"京都\",\"label\":\"位置\",\"object\":\"関西\"}]}\n",
    );
    let out_path = dir.join("evaluation.json");

    let (code, _stdout, stderr) = run_cli(
        &[
            "evaluate",
            "--eval",
            eval_path.to_str().unwrap(),
            "--context",
            "looks",
            "--url",
            &server.base,
            "--out",
            out_path.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "{stderr}");

    let evaluation: Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).unwrap()).unwrap();
    let assoc = &evaluation["cases"][0]["structural"]["associations"][0];
    assert_eq!(assoc["subject"]["outcome"], "ambiguous", "{assoc}");
    let candidates: Vec<&str> = assoc["subject"]["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(candidates.contains(&"京都"), "{assoc}");
    assert!(candidates.contains(&"東京都"), "{assoc}");
    // Neither guessed at nor fanned out over (ADR 0004 §7 step 2):
    // query is simply never called for an ambiguous position.
    assert!(assoc["query"].is_null(), "{assoc}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn evaluate_runs_the_citation_lane_without_preflighting_it_and_distinguishes_no_source_from_no_paragraph()
 {
    let server = Server::start("evaluate-citations");
    seed_context(&server, "sake");
    let dir = eval_dir("citations");
    // No `expected_sources` at all — the passage-lane search for this
    // query surfaces nothing relevant, exercising ADR 0004 §8's
    // orthogonality claim: the citation lane runs and is scored
    // regardless of what (if anything) the passage lane found. Neither
    // expected_citations entry names a source/paragraph declared in
    // `expected_sources`, so the startup preflight (which only reads
    // expected_sources) never sees them and never aborts the run.
    let eval_path = write_eval_file(
        &dir,
        "{\"taguru_eval\":1,\"name\":\"citation lane\"}\n\
         {\"case_id\":\"citations-001\",\"query\":\"存在しないクエリ\",\
         \"expected_citations\":[\
           {\"source\":\"corpus/does-not-exist.md\",\"paragraph\":0},\
           {\"source\":\"corpus/brewery.md\",\"paragraph\":99}\
         ]}\n",
    );
    let out_path = dir.join("evaluation.json");

    let (code, _stdout, stderr) = run_cli(
        &[
            "evaluate",
            "--eval",
            eval_path.to_str().unwrap(),
            "--context",
            "sake",
            "--url",
            &server.base,
            "--out",
            out_path.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 0, "{stderr}");
    assert!(
        out_path.exists(),
        "an expected_citations entry naming a source the corpus lacks must not abort the run"
    );

    let evaluation: Value =
        serde_json::from_str(&std::fs::read_to_string(&out_path).unwrap()).unwrap();
    let checks = &evaluation["cases"][0]["citations"]["checks"];
    assert_eq!(checks[0]["source"], "corpus/does-not-exist.md", "{checks}");
    assert_eq!(checks[0]["outcome"], "unresolved", "{checks}");
    assert_eq!(checks[0]["code"], "no_source", "{checks}");
    assert_eq!(checks[1]["source"], "corpus/brewery.md", "{checks}");
    assert_eq!(checks[1]["outcome"], "unresolved", "{checks}");
    assert_eq!(checks[1]["code"], "no_paragraph", "{checks}");
    assert_eq!(
        evaluation["cases"][0]["citations"]["validity"]["value"], 0.0,
        "{evaluation}"
    );
    assert_eq!(
        evaluation["cases"][0]["citations"]["recall"]["value"], 0.0,
        "{evaluation}"
    );

    let metrics = &evaluation["metrics"];
    assert_eq!(metrics["citations.no_source"]["numerator"], 1, "{metrics}");
    assert_eq!(
        metrics["citations.no_paragraph"]["numerator"], 1,
        "{metrics}"
    );
    assert_eq!(metrics["citations.resolved"]["numerator"], 0, "{metrics}");
    assert_eq!(metrics["citations.resolved"]["n"], 2, "{metrics}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn evaluate_exits_1_when_the_server_is_unreachable() {
    let dir = eval_dir("unreachable");
    let eval_path = write_smoke_eval(&dir);
    let out_path = dir.join("evaluation.json");

    let (code, _stdout, stderr) = run_cli(
        &[
            "evaluate",
            "--eval",
            eval_path.to_str().unwrap(),
            "--context",
            "sake",
            "--url",
            "http://127.0.0.1:1",
            "--out",
            out_path.to_str().unwrap(),
        ],
        &[],
    );
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("not reachable"), "{stderr}");
    assert!(!out_path.exists());

    let _ = std::fs::remove_dir_all(&dir);
}
