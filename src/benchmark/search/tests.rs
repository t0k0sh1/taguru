use super::*;
use crate::evalset::EvalOptions;

fn args(words: &[&str]) -> Result<SearchArgs, i32> {
    parse_args(&words.iter().map(|s| s.to_string()).collect::<Vec<_>>())
}

fn hit(source: &str, paragraph: u32, text: &str) -> PassageHit {
    PassageHit {
        source: source.to_string(),
        paragraph,
        score: 1.0,
        text: text.to_string(),
        lanes: PassageLanes {
            bm25: None,
            vector: None,
        },
    }
}

fn locator(rank: usize, source: &str, paragraph: u32) -> HitLocator {
    HitLocator {
        rank,
        source: source.to_string(),
        paragraph,
    }
}

fn doc(document_id: &str, path: &str) -> DocumentInfo {
    DocumentInfo {
        document_id: document_id.to_string(),
        path: path.to_string(),
        ..Default::default()
    }
}

fn case_with_limit(limit: Option<usize>) -> EvalCase {
    EvalCase {
        case_id: "c".to_string(),
        query: "q".to_string(),
        cues: Vec::new(),
        expected_sources: Vec::new(),
        expected_concepts: Vec::new(),
        options: EvalOptions {
            limit,
            ..EvalOptions::default()
        },
        expected_labels: Vec::new(),
        expected_associations: Vec::new(),
        expected_citations: Vec::new(),
    }
}

// ============================== Argument parsing ==============================

#[test]
fn eval_and_results_dir_are_both_required() {
    assert_eq!(args(&[]).unwrap_err(), 2);
    assert_eq!(
        args(&["--eval", "e.jsonl"]).unwrap_err(),
        2,
        "no RESULTS_DIR"
    );
    assert_eq!(args(&["out"]).unwrap_err(), 2, "no --eval");
}

#[test]
fn a_full_argument_set_parses() {
    let parsed = args(&[
        "--eval",
        "e.jsonl",
        "--url",
        "http://localhost:8248/",
        "--config",
        "c.json",
        "--run",
        "2",
        "--context-prefix",
        "bench",
        "--skip-import",
        "out",
    ])
    .unwrap();
    assert_eq!(parsed.eval, PathBuf::from("e.jsonl"));
    assert_eq!(parsed.dir, PathBuf::from("out"));
    // trailing slash trimmed, matching every other --url verb
    assert_eq!(parsed.url.as_deref(), Some("http://localhost:8248"));
    assert_eq!(parsed.config, Some(PathBuf::from("c.json")));
    assert_eq!(parsed.run, 2);
    assert_eq!(parsed.context_prefix.as_deref(), Some("bench"));
    assert!(parsed.skip_import);
}

#[test]
fn run_must_be_a_positive_integer() {
    assert_eq!(
        args(&["--eval", "e.jsonl", "--run", "0", "out"]).unwrap_err(),
        2
    );
    assert_eq!(
        args(&["--eval", "e.jsonl", "--run", "nope", "out"]).unwrap_err(),
        2
    );
}

#[test]
fn an_unknown_flag_is_a_usage_error() {
    assert_eq!(
        args(&["--eval", "e.jsonl", "--bogus", "out"]).unwrap_err(),
        2
    );
}

#[test]
fn exactly_one_results_dir_is_required() {
    assert_eq!(
        args(&["--eval", "e.jsonl", "out1", "out2"]).unwrap_err(),
        2,
        "two positionals is a usage error"
    );
}

#[test]
fn help_short_circuits_with_exit_zero() {
    assert_eq!(args(&["--help"]).unwrap_err(), 0);
}

// ============================== URL masking ==============================

#[test]
fn mask_url_drops_userinfo_path_and_query_but_keeps_scheme_host_port() {
    assert_eq!(
        mask_url("https://user:token@host:8443/x?y=1"),
        "https://host:8443"
    );
}

#[test]
fn mask_url_without_a_port_omits_it() {
    assert_eq!(mask_url("http://example.com/a"), "http://example.com");
}

#[test]
fn mask_url_redacts_an_unparsable_base_instead_of_echoing_it() {
    assert_eq!(mask_url("not a url"), "<unparseable-url>");
}

// ============================== Limit validation ==============================

#[test]
fn a_zero_or_over_ceiling_limit_is_refused() {
    assert!(validate_limits(&[case_with_limit(Some(0))]).is_err());
    assert!(validate_limits(&[case_with_limit(Some(MAX_SEARCH_LIMIT + 1))]).is_err());
}

#[test]
fn a_limit_within_range_or_absent_is_accepted() {
    assert!(validate_limits(&[case_with_limit(Some(1))]).is_ok());
    assert!(validate_limits(&[case_with_limit(Some(MAX_SEARCH_LIMIT))]).is_ok());
    assert!(validate_limits(&[case_with_limit(None)]).is_ok());
}

// ============================== Context naming ==============================

#[test]
fn corpus_context_name_joins_prefix_and_model_with_a_double_colon() {
    assert_eq!(corpus_context_name("sake", "gpt-4o"), "sake::gpt-4o");
}

#[test]
fn pair_key_is_sorted_and_symmetric() {
    assert_eq!(pair_key("b", "a"), "a__b");
    assert_eq!(pair_key("a", "b"), "a__b");
}

// ============================== Ownership marker ==============================

#[test]
fn ownership_marker_differs_across_run_index_for_the_same_run_id_and_model() {
    // run_id names the whole extract invocation, constant across every
    // --run N within it — without run_index in the marker, two
    // different runs of the same results directory would produce the
    // same marker and silently merge into one corpus.
    let run1 = ownership_marker("run-abc", "m1", 1);
    let run2 = ownership_marker("run-abc", "m1", 2);
    assert_ne!(run1, run2, "{run1} vs {run2}");
    assert!(run1.contains("run_index 1"), "{run1}");
    assert!(run2.contains("run_index 2"), "{run2}");
}

// ============================== Batch header rewriting ==============================

#[test]
fn rewrite_batch_header_replaces_context_and_stamps_the_marker() {
    let original = "{\"taguru_batch\":1,\"context\":\"sake\",\"source\":\"docs/a.md\"}\n\
                     {\"subject\":\"s\",\"label\":\"l\",\"object\":\"o\",\"weight\":1.0}\n";
    let rewritten = rewrite_batch_header(
        original,
        "sake::gpt-4o",
        "taguru benchmark search corpus: run r, model m",
    )
    .unwrap();
    let mut lines = rewritten.lines();
    let header: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    assert_eq!(header["context"], "sake::gpt-4o");
    assert_eq!(header["source"], "docs/a.md", "source is left untouched");
    assert_eq!(
        header["create"]["description"],
        "taguru benchmark search corpus: run r, model m"
    );
    assert_eq!(
        lines.next().unwrap(),
        "{\"subject\":\"s\",\"label\":\"l\",\"object\":\"o\",\"weight\":1.0}",
        "every other line rides through byte-for-byte"
    );
    assert!(rewritten.ends_with('\n'));
}

#[test]
fn rewrite_batch_header_overwrites_any_existing_create_block() {
    let original = "{\"taguru_batch\":1,\"context\":\"sake\",\"source\":\"docs/a.md\",\"create\":{\"description\":\"whatever the cell wrote\"}}\n";
    let rewritten = rewrite_batch_header(original, "sake::gpt-4o", "owner-marker").unwrap();
    let header: Value = serde_json::from_str(rewritten.lines().next().unwrap()).unwrap();
    assert_eq!(header["create"]["description"], "owner-marker");
}

#[test]
fn rewrite_batch_header_refuses_an_empty_file() {
    assert!(rewrite_batch_header("", "ctx", "marker").is_err());
}

#[test]
fn rewrite_batch_header_refuses_a_non_json_header() {
    assert!(rewrite_batch_header("not json\n", "ctx", "marker").is_err());
}

#[test]
fn rewrite_batch_header_drops_blank_lines() {
    let original =
        "{\"taguru_batch\":1,\"context\":\"c\",\"source\":\"s\"}\n\n{\"passage\":\"x\"}\n\n";
    let rewritten = rewrite_batch_header(original, "c2", "m").unwrap();
    assert_eq!(rewritten.lines().count(), 2);
}

// ============================== Response extraction ==============================

#[test]
fn extract_hits_prefers_the_typed_plan_and_hits_shape() {
    let value = serde_json::json!({
        "plan": {"contexts": [{"context": "ctx1", "lanes": {
            "bm25": {"ran": true},
            "vector": {"ran": false, "reason": "no embedding provider is configured"}
        }}]},
        "hits": [{"source": "docs/a.md", "paragraph": 0, "score": 1.0, "text": "hello",
                   "lanes": {"bm25": {"rank": 1, "score": 1.0}}}]
    });
    let (hits, plan) = extract_hits(&value).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].source, "docs/a.md");
    assert!(hits[0].lanes.bm25.is_some());
    assert!(hits[0].lanes.vector.is_none());
    let plan = plan.unwrap();
    assert_eq!(plan.context, "ctx1");
    assert!(plan.lanes.bm25.ran);
    assert!(!plan.lanes.vector.ran);
    assert_eq!(
        plan.lanes.vector.reason.as_deref(),
        Some("no embedding provider is configured")
    );
}

#[test]
fn extract_hits_falls_back_to_a_bare_hit_array() {
    let value = serde_json::json!([
        {"source": "docs/a.md", "paragraph": 0, "score": 0.5, "text": "hello"}
    ]);
    let (hits, plan) = extract_hits(&value).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].text, "hello");
    assert!(plan.is_none());
}

#[test]
fn extract_hits_falls_back_to_an_object_wrapped_hits_array_without_a_plan() {
    let value = serde_json::json!({"hits": [{"source": "docs/a.md", "paragraph": 1, "text": "x"}]});
    let (hits, plan) = extract_hits(&value).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].paragraph, 1);
    assert!(plan.is_none());
}

#[test]
fn extract_hits_refuses_a_response_with_no_recognizable_hits() {
    assert!(extract_hits(&serde_json::json!({"foo": 1})).is_err());
}

// ============================== Lane accounting ==============================

#[test]
fn lane_hit_counts_splits_by_which_lanes_evidenced_each_hit() {
    let mut both_hit = hit("a", 0, "");
    both_hit.lanes = PassageLanes {
        bm25: Some(crate::api::sources::LaneEvidence {
            rank: 1,
            score: 1.0,
        }),
        vector: Some(crate::api::sources::LaneEvidence {
            rank: 2,
            score: 0.9,
        }),
    };
    let mut bm25_hit = hit("b", 0, "");
    bm25_hit.lanes.bm25 = Some(crate::api::sources::LaneEvidence {
        rank: 1,
        score: 1.0,
    });
    let neither_hit = hit("c", 0, "");

    let counts = lane_hit_counts(&[both_hit, bm25_hit, neither_hit], true);
    assert_eq!(counts.both, 1);
    assert_eq!(counts.bm25_only, 1);
    assert_eq!(counts.vector_only, 0);
    assert_eq!(counts.neither, 1);
    assert_eq!(counts.unknown, 0);
}

#[test]
fn lane_hit_counts_reports_unknown_when_lanes_were_not_recoverable() {
    let hits = [hit("a", 0, ""), hit("b", 0, "")];
    let counts = lane_hit_counts(&hits, false);
    assert_eq!(counts.unknown, 2);
    assert_eq!(
        counts.both + counts.bm25_only + counts.vector_only + counts.neither,
        0
    );
}

// ============================== Expected-source resolution ==============================

#[test]
fn resolve_expected_source_path_prefers_an_exact_path_match() {
    let docs = [doc("d1", "corpus/a.md"), doc("d2", "corpus/b.md")];
    let (resolved, warning) = resolve_expected_source_path("corpus/a.md", &docs);
    assert_eq!(resolved, "corpus/a.md");
    assert!(warning.is_none());
}

#[test]
fn resolve_expected_source_path_falls_back_to_a_document_id_match() {
    let docs = [doc("brand-001", "corpus/a.md")];
    let (resolved, warning) = resolve_expected_source_path("brand-001", &docs);
    assert_eq!(resolved, "corpus/a.md");
    assert!(warning.is_none());
}

#[test]
fn resolve_expected_source_path_falls_back_to_a_unique_suffix_match() {
    let docs = [doc("d1", "/abs/path/corpus/brewery.md")];
    let (resolved, warning) = resolve_expected_source_path("corpus/brewery.md", &docs);
    assert_eq!(resolved, "/abs/path/corpus/brewery.md");
    assert!(warning.is_none());
}

#[test]
fn resolve_expected_source_path_warns_and_falls_back_literally_when_unresolved() {
    let docs = [doc("d1", "corpus/a.md")];
    let (resolved, warning) = resolve_expected_source_path("corpus/does-not-exist.md", &docs);
    assert_eq!(resolved, "corpus/does-not-exist.md");
    assert!(warning.unwrap().contains("matches no document"));
}

#[test]
fn resolve_expected_source_path_warns_on_an_ambiguous_suffix() {
    let docs = [doc("d1", "a/brewery.md"), doc("d2", "b/brewery.md")];
    let (resolved, warning) = resolve_expected_source_path("brewery.md", &docs);
    assert_eq!(resolved, "brewery.md");
    assert!(warning.unwrap().contains("ambiguous"));
}

// ============================== recall@k / MRR ==============================

#[test]
fn compute_recall_matches_a_source_regardless_of_paragraph_when_none_are_named() {
    let matching = identity::Matching::default();
    let expected = vec![ExpectedSource {
        source: "corpus/a.md".to_string(),
        paragraphs: vec![],
        relevance: 1,
    }];
    let docs = [doc("d1", "corpus/a.md")];
    let hits = [hit("corpus/a.md", 7, "irrelevant text")];
    let (result, warnings) = compute_recall(&matching, "c1", &expected, &[], &docs, &hits);
    assert!(warnings.is_empty());
    assert_eq!(result.matched, 1);
    assert_eq!(result.expected_total, 1);
    assert_eq!(result.recall_at_k, 1.0);
    assert_eq!(result.mrr, 1.0);
}

#[test]
fn compute_recall_requires_the_named_paragraph_when_paragraphs_are_given() {
    let matching = identity::Matching::default();
    let expected = vec![ExpectedSource {
        source: "corpus/a.md".to_string(),
        paragraphs: vec![2],
        relevance: 1,
    }];
    let docs = [doc("d1", "corpus/a.md")];
    let hits = [hit("corpus/a.md", 0, "")];
    let (result, _) = compute_recall(&matching, "c1", &expected, &[], &docs, &hits);
    assert_eq!(
        result.matched, 0,
        "paragraph 0 does not satisfy an expectation pinned to 2"
    );
}

#[test]
fn compute_recall_excludes_zero_relevance_entries_from_the_denominator() {
    let matching = identity::Matching::default();
    let expected = vec![ExpectedSource {
        source: "corpus/a.md".to_string(),
        paragraphs: vec![],
        relevance: 0,
    }];
    let docs = [doc("d1", "corpus/a.md")];
    let hits = [hit("corpus/a.md", 0, "")];
    let (result, _) = compute_recall(&matching, "c1", &expected, &[], &docs, &hits);
    assert_eq!(result.expected_total, 0);
    assert_eq!(result.recall_at_k, 0.0);
}

#[test]
fn compute_recall_matches_a_concept_by_folded_substring_in_hit_text() {
    let matching = identity::Matching::default();
    let docs: [DocumentInfo; 0] = [];
    // Fullwidth + uppercase in the expectation, halfwidth + lowercase
    // in the hit text — only matches if NFKC and case folding both run
    // (identity::normalize_term, ADR 0003 §9.4's own precedent).
    let hits = [hit("corpus/a.md", 0, "the brewery is called aomine")];
    let (result, _) = compute_recall(
        &matching,
        "c1",
        &[],
        &["ＡＯＭＩＮＥ".to_string()],
        &docs,
        &hits,
    );
    assert_eq!(result.matched, 1);
    assert_eq!(result.recall_at_k, 1.0);
}

#[test]
fn compute_recall_unions_source_and_concept_expectations_into_one_denominator() {
    let matching = identity::Matching::default();
    let expected_sources = vec![ExpectedSource {
        source: "corpus/a.md".to_string(),
        paragraphs: vec![],
        relevance: 1,
    }];
    let docs = [doc("d1", "corpus/a.md")];
    let hits = [hit("corpus/a.md", 0, "青嶺酒造の話")];
    let (result, _) = compute_recall(
        &matching,
        "c1",
        &expected_sources,
        &["青嶺酒造".to_string()],
        &docs,
        &hits,
    );
    assert_eq!(
        result.expected_total, 2,
        "one source entry plus one concept entry"
    );
    assert_eq!(result.matched, 2, "the single hit satisfies both");
}

#[test]
fn compute_recall_mrr_reflects_the_first_hit_that_satisfies_anything() {
    let matching = identity::Matching::default();
    let expected = vec![ExpectedSource {
        source: "corpus/b.md".to_string(),
        paragraphs: vec![],
        relevance: 1,
    }];
    let docs = [doc("d1", "corpus/a.md"), doc("d2", "corpus/b.md")];
    let hits = [hit("corpus/a.md", 0, ""), hit("corpus/b.md", 0, "")];
    let (result, _) = compute_recall(&matching, "c1", &expected, &[], &docs, &hits);
    assert_eq!(result.mrr, 0.5, "the match lands at rank 2");
}

#[test]
fn compute_recall_is_zero_when_nothing_matches() {
    let matching = identity::Matching::default();
    let expected = vec![ExpectedSource {
        source: "corpus/z.md".to_string(),
        paragraphs: vec![],
        relevance: 1,
    }];
    let docs = [doc("d1", "corpus/z.md")];
    let hits = [hit("corpus/a.md", 0, "")];
    let (result, _) = compute_recall(&matching, "c1", &expected, &[], &docs, &hits);
    assert_eq!(result.matched, 0);
    assert_eq!(result.recall_at_k, 0.0);
    assert_eq!(result.mrr, 0.0);
}

#[test]
fn resolve_expected_items_runs_once_and_score_recall_reuses_it_across_models() {
    // The regression this guards: resolving expectations must happen
    // once per case, not once per model — resolve_expected_items is
    // the only place a resolution warning is produced, so calling it
    // once and scoring each model's hits against the same items is
    // what keeps an unresolvable source from warning N times for N
    // models. Two models' hits, scored from the one resolved list,
    // must each get their own independent recall result.
    let matching = identity::Matching::default();
    let expected = vec![ExpectedSource {
        source: "corpus/does-not-exist.md".to_string(),
        paragraphs: vec![],
        relevance: 1,
    }];
    let docs = [doc("d1", "corpus/a.md")];
    let (items, warnings) = resolve_expected_items(&matching, "c1", &expected, &[], &docs);
    assert_eq!(warnings.len(), 1, "{warnings:?}");

    let hits_m1 = [hit("corpus/does-not-exist.md", 0, "")];
    let hits_m2 = [hit("corpus/a.md", 0, "")];
    let result_m1 = score_recall(&matching, &items, &hits_m1);
    let result_m2 = score_recall(&matching, &items, &hits_m2);
    assert_eq!(
        result_m1.matched, 1,
        "m1's hit matches the literal fallback path"
    );
    assert_eq!(result_m2.matched, 0, "m2's hit does not");
}

// ============================== Pair overlap ==============================

#[test]
fn pair_overlap_of_two_empty_hit_lists_is_none_not_a_vacuous_value() {
    let (jaccard, shared, mean_rank_difference) = pair_overlap(&[], &[]);
    assert_eq!(jaccard, None);
    assert_eq!(shared, 0);
    assert_eq!(mean_rank_difference, None);
}

#[test]
fn pair_overlap_computes_jaccard_and_mean_rank_difference_over_shared_locators() {
    let a = vec![
        locator(1, "s1", 0),
        locator(2, "s2", 0),
        locator(3, "s3", 0),
    ];
    let b = vec![locator(1, "s2", 0), locator(2, "s4", 0)];
    let (jaccard, shared, mean_rank_difference) = pair_overlap(&a, &b);
    // intersection {s2}=1, union {s1,s2,s3,s4}=4
    assert_eq!(jaccard, Some(0.25));
    assert_eq!(shared, 1);
    // s2 is rank 2 in a, rank 1 in b => |2-1| = 1
    assert_eq!(mean_rank_difference, Some(1.0));
}

#[test]
fn pair_overlap_of_disjoint_hit_sets_has_no_shared_locators() {
    let a = vec![locator(1, "s1", 0)];
    let b = vec![locator(1, "s2", 0)];
    let (jaccard, shared, mean_rank_difference) = pair_overlap(&a, &b);
    assert_eq!(jaccard, Some(0.0));
    assert_eq!(shared, 0);
    assert_eq!(mean_rank_difference, None);
}

// ============================== Aggregation ==============================

#[test]
fn aggregate_model_keeps_unknown_lanes_apart_from_genuine_zero_hit_lanes() {
    // A case whose lanes could not be recovered (the legacy fallback
    // in extract_hits) must not be indistinguishable, in the
    // aggregate, from a case whose lanes genuinely evidenced nothing
    // — bm25_only/vector_only/both are 0 in both situations, so
    // lanes.unknown is what tells them apart.
    let mut models = BTreeMap::new();
    models.insert(
        "m1".to_string(),
        SearchOutcome::Searched {
            hit_count: 2,
            empty: false,
            distinct_sources: 1,
            lanes: LaneHitCounts {
                bm25_only: 0,
                vector_only: 0,
                both: 0,
                neither: 0,
                unknown: 2,
            },
            plan: None,
            hits: vec![],
            recall: None,
        },
    );
    let cases = vec![CaseBlock {
        case_id: "c1".to_string(),
        query: "q".to_string(),
        cues: vec![],
        limit: 10,
        has_expectations: false,
        models,
        pairs: BTreeMap::new(),
    }];

    let metrics = aggregate_model(&cases, "m1");
    let value = serde_json::to_value(&metrics).unwrap();
    assert_eq!(value["lanes.unknown"]["n"], 1, "{value}");
    assert_eq!(value["lanes.unknown"]["sum"], 2.0, "{value}");
    assert_eq!(
        value["lanes.bm25_only"]["sum"], 0.0,
        "still recorded, but lanes.unknown is what marks this case's zeros as \
         unrecoverable rather than observed — {value}"
    );
}

// ============================== No cross-model verdict vocabulary ==============================

/// Words that would imply this artifact ranks or judges models against
/// each other — legitimate for #215's own quality gate, never here
/// (this module's own doc comment explains why `rank`/`score` are
/// exempt: they name an ordinary IR concept — a hit's own position —
/// never a cross-model ranking).
const VERDICT_WORDS: [&str; 5] = ["winner", "best", "recommended", "overall", "delta_vs"];

fn assert_no_verdict_words(value: &Value, path: &str) {
    match value {
        Value::Object(map) => {
            for (key, v) in map {
                for banned in VERDICT_WORDS {
                    assert!(
                        !key.to_lowercase().contains(banned),
                        "verdict word '{banned}' found in key '{key}' at {path}"
                    );
                }
                assert_no_verdict_words(v, &format!("{path}.{key}"));
            }
        }
        Value::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                assert_no_verdict_words(v, &format!("{path}[{i}]"));
            }
        }
        _ => {}
    }
}

#[test]
fn no_verdict_word_appears_anywhere_in_the_artifact() {
    let mut cases = Vec::new();
    let mut models = BTreeMap::new();
    models.insert(
        "m1".to_string(),
        SearchOutcome::Searched {
            hit_count: 1,
            empty: false,
            distinct_sources: 1,
            lanes: LaneHitCounts::default(),
            plan: None,
            hits: vec![locator(1, "corpus/a.md", 0)],
            recall: Some(RecallResult {
                recall_at_k: 1.0,
                mrr: 1.0,
                expected_total: 1,
                matched: 1,
            }),
        },
    );
    let mut pairs = BTreeMap::new();
    pairs.insert(
        "m1__m2".to_string(),
        PairOutcome::Compared {
            jaccard: Some(0.5),
            shared_hits: 1,
            mean_rank_difference: Some(1.0),
        },
    );
    cases.push(CaseBlock {
        case_id: "c1".to_string(),
        query: "q".to_string(),
        cues: vec![],
        limit: 10,
        has_expectations: true,
        models,
        pairs,
    });
    let (models_agg, pairs_agg) = aggregate(&cases);

    let retrieval = RetrievalFile {
        taguru_benchmark_retrieval: BENCHMARK_RETRIEVAL_VERSION,
        run_id: "r1".to_string(),
        generated_at: "2026-01-01T00:00:00Z".to_string(),
        matching: identity::Matching::default(),
        inputs: InputsBlock {
            results_dir: "out".to_string(),
            eval: EvalInputsBlock {
                path: "e.jsonl".to_string(),
                name: None,
                cases: 1,
            },
            url: "http://localhost:8248".to_string(),
            run_index: 1,
            default_limit: DEFAULT_LIMIT,
        },
        definitions: build_definitions(),
        warnings: vec![],
        corpus: BTreeMap::new(),
        cases,
        models: models_agg,
        pairs: pairs_agg,
    };

    let value = serde_json::to_value(&retrieval).unwrap();
    assert_no_verdict_words(&value, "$");
}
