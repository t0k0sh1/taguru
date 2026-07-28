use axum::http::Method;

use super::*;

// ============================ Argument parsing ============================

fn args(words: &[&str]) -> Result<EvaluateArgs, i32> {
    parse_args(&words.iter().map(|s| s.to_string()).collect::<Vec<_>>())
}

#[test]
fn missing_eval_is_a_usage_error() {
    assert_eq!(args(&["--context", "sake"]).unwrap_err(), 2);
}

#[test]
fn missing_context_is_a_usage_error() {
    assert_eq!(args(&["--eval", "eval.jsonl"]).unwrap_err(), 2);
}

#[test]
fn a_duplicate_flag_is_refused() {
    let error = args(&[
        "--eval",
        "a.jsonl",
        "--eval",
        "b.jsonl",
        "--context",
        "sake",
    ])
    .unwrap_err();
    assert_eq!(error, 2);
}

#[test]
fn an_unknown_flag_is_refused() {
    let error = args(&["--eval", "a.jsonl", "--context", "sake", "--bogus"]).unwrap_err();
    assert_eq!(error, 2);
}

#[test]
fn a_positional_argument_is_refused() {
    let error = args(&["--eval", "a.jsonl", "--context", "sake", "extra"]).unwrap_err();
    assert_eq!(error, 2);
}

#[test]
fn out_defaults_to_evaluation_json_and_url_config_default_to_none() {
    let parsed =
        args(&["--eval", "a.jsonl", "--context", "sake"]).expect("minimal flags must parse");
    assert_eq!(parsed.out, PathBuf::from("evaluation.json"));
    assert!(parsed.url.is_none());
    assert!(parsed.config.is_none());
}

#[test]
fn every_flag_is_captured_and_a_trailing_url_slash_is_trimmed() {
    let parsed = args(&[
        "--eval",
        "a.jsonl",
        "--context",
        "sake",
        "--url",
        "http://h:1/",
        "--config",
        "c.env",
        "--out",
        "out.json",
    ])
    .expect("a fully-specified invocation must parse");
    assert_eq!(parsed.eval, PathBuf::from("a.jsonl"));
    assert_eq!(parsed.context, "sake");
    assert_eq!(parsed.url.as_deref(), Some("http://h:1"));
    assert_eq!(parsed.config, Some(PathBuf::from("c.env")));
    assert_eq!(parsed.out, PathBuf::from("out.json"));
}

// ================================ Dispatch ================================

#[test]
fn run_with_no_args_is_a_usage_error() {
    assert_eq!(run(&[]), 2);
}

#[test]
fn run_with_an_unknown_subcommand_is_a_usage_error() {
    assert_eq!(run(&["bogus".to_string()]), 2);
}

#[test]
fn run_help_exits_zero_without_touching_the_network() {
    assert_eq!(run(&["--help".to_string()]), 0);
}

// =============================== URL masking ===============================

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
fn mask_url_tolerates_an_unparsable_base() {
    assert_eq!(mask_url("not a url"), "not a url");
}

// ============================= Message truncation =============================

#[test]
fn truncate_message_leaves_short_messages_alone() {
    assert_eq!(truncate_message("short"), "short");
}

#[test]
fn truncate_message_cuts_long_multibyte_messages_without_panicking() {
    let long = "あ".repeat(200);
    let truncated = truncate_message(&long);
    assert!(truncated.ends_with('…'), "{truncated}");
    assert!(truncated.len() < long.len());
}

// ================================ Tier choice ================================

fn candidate(name: &str, tier: &str) -> TieredResolution {
    TieredResolution {
        name: name.to_string(),
        score: 1.0,
        tier: tier.to_string(),
        kind: None,
        gloss: None,
    }
}

#[test]
fn top_tier_prefers_lexical_when_present() {
    let candidates = vec![candidate("a", "lexical"), candidate("b", "semantic")];
    let group = top_tier(&candidates);
    assert_eq!(group.len(), 1);
    assert_eq!(group[0].name, "a");
}

#[test]
fn top_tier_falls_back_to_semantic_when_no_lexical_candidate_exists() {
    let candidates = vec![candidate("a", "semantic"), candidate("b", "semantic")];
    let group = top_tier(&candidates);
    assert_eq!(group.len(), 2);
}

#[test]
fn top_tier_of_an_empty_response_is_empty() {
    assert!(top_tier(&[]).is_empty());
}

// ========================= Multi-candidate policy (ADR §7 step 2) =========================

#[test]
fn classify_position_with_zero_candidates_is_not_found() {
    let outcome = classify_position(Vec::new(), 12);
    assert!(matches!(
        outcome,
        PositionOutcome::NotFound { latency_ms: 12 }
    ));
}

#[test]
fn classify_position_with_exactly_one_candidate_is_resolved() {
    let a = candidate("青嶺酒造", "lexical");
    let outcome = classify_position(vec![&a], 7);
    match outcome {
        PositionOutcome::Resolved {
            name,
            tier,
            latency_ms,
        } => {
            assert_eq!(name, "青嶺酒造");
            assert_eq!(tier, "lexical");
            assert_eq!(latency_ms, 7);
        }
        _ => panic!("expected Resolved, got a different outcome"),
    }
}

#[test]
fn classify_position_with_two_or_more_candidates_is_ambiguous() {
    let a = candidate("a", "semantic");
    let b = candidate("b", "semantic");
    let outcome = classify_position(vec![&a, &b], 3);
    match outcome {
        PositionOutcome::Ambiguous {
            tier, candidates, ..
        } => {
            assert_eq!(tier, "semantic");
            assert_eq!(candidates, vec!["a".to_string(), "b".to_string()]);
        }
        _ => panic!("expected Ambiguous, got a different outcome"),
    }
}

// ========================= Passage response degrade path =========================

fn passage_page_value() -> Value {
    serde_json::json!({
        "plan": {
            "contexts": [
                {
                    "context": "sake",
                    "lanes": {
                        "bm25": {"ran": true},
                        "vector": {"ran": false, "reason": "no embedding provider is configured"}
                    }
                }
            ]
        },
        "hits": [
            {
                "source": "corpus/brewery.md",
                "paragraph": 0,
                "score": 1.0,
                "text": "青嶺は青嶺酒造が造る銘柄です。",
                "lanes": {"bm25": {"rank": 1, "score": 1.0}}
            }
        ]
    })
}

#[test]
fn extract_passages_reads_the_real_plan_and_hits_shape() {
    let value = passage_page_value();
    let (hits, plan) = extract_passages(&value).expect("a well-formed PassagePage must parse");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].source, "corpus/brewery.md");
    let plan = plan.expect("plan.contexts must carry the one searched context");
    assert_eq!(plan.context, "sake");
}

#[test]
fn extract_passages_falls_back_to_a_bare_hits_array() {
    let value = serde_json::json!([
        {"source": "corpus/brewery.md", "paragraph": 0, "score": 0.5}
    ]);
    let (hits, plan) = extract_passages(&value).expect("a bare hits array must still parse");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].paragraph, 0);
    assert!(plan.is_none());
}

#[test]
fn extract_passages_falls_back_to_an_object_wrapped_hits_array_without_a_plan() {
    let value = serde_json::json!({"hits": [{"source": "s", "paragraph": 1, "score": 0.1}]});
    let (hits, plan) =
        extract_passages(&value).expect("an object without plan must still parse via hits");
    assert_eq!(hits.len(), 1);
    assert!(plan.is_none());
}

#[test]
fn extract_passages_rejects_a_response_with_no_recognizable_hits() {
    let value = serde_json::json!({"foo": 1});
    assert!(extract_passages(&value).is_err());
}

// ========================= No corpus body text in hits[] =========================

fn assert_no_body_text(value: &Value, needle: &str, path: &str) {
    match value {
        Value::String(text) => {
            assert!(!text.contains(needle), "body text leaked at {path}: {text}")
        }
        Value::Object(map) => {
            for (key, v) in map {
                assert_no_body_text(v, needle, &format!("{path}.{key}"));
            }
        }
        Value::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                assert_no_body_text(v, needle, &format!("{path}[{i}]"));
            }
        }
        _ => {}
    }
}

#[test]
fn hit_locator_never_carries_the_passage_body_text() {
    let hit = PassageHit {
        source: "corpus/brewery.md".to_string(),
        paragraph: 0,
        score: 1.0,
        text: "TOP-SECRET-PASSAGE-BODY-TEXT".to_string(),
        lanes: PassageLanes {
            bm25: None,
            vector: None,
        },
    };
    let locator: HitLocator = hit.into();
    let value = serde_json::to_value(&locator).unwrap();
    assert_no_body_text(&value, "TOP-SECRET-PASSAGE-BODY-TEXT", "$");
}

// ========================= metrics <-> definitions agreement =========================

#[test]
fn every_metric_key_has_a_matching_definition_and_vice_versa() {
    let cases: Vec<CaseBlock> = Vec::new();
    let metrics = build_metrics(&cases);
    let definitions = build_definitions();
    let metric_keys: BTreeSet<&String> = metrics.keys().collect();
    let definition_keys: BTreeSet<&String> = definitions.keys().collect();
    assert_eq!(metric_keys, definition_keys);
}

// ========================= No answer-generation LLM seam (ADR §12, AC 8) =========================

#[test]
fn evaluate_module_never_names_an_extraction_or_embedding_seam() {
    // Read straight from disk rather than `include_str!`, so this
    // check's own source text is never itself scanned (it lives in a
    // sibling file, tests.rs, which this assertion does not touch).
    let source = include_str!("../evaluate.rs");
    // Built by concatenation so this assertion's own literals never
    // accidentally match themselves.
    let extract_prefix = concat!("TAGURU_", "EXTRACT_");
    let embed_prefix = concat!("TAGURU_", "EMBED_");
    assert!(
        !source.contains(extract_prefix),
        "found {extract_prefix} in evaluate.rs"
    );
    assert!(
        !source.contains(embed_prefix),
        "found {embed_prefix} in evaluate.rs"
    );
    assert!(
        !source.contains("crate::extract"),
        "found a crate::extract import in evaluate.rs"
    );
    assert!(
        !source.contains("crate::embedding"),
        "found a crate::embedding import in evaluate.rs"
    );
}

// ========================= Endpoint role (ADR §11) =========================

#[test]
fn evaluate_only_touches_read_role_endpoints() {
    for (method, route) in [
        (Method::POST, "/contexts/{name}/sources/search"),
        (Method::POST, "/contexts/{name}/resolve"),
        (Method::POST, "/contexts/{name}/resolve_label"),
        (Method::POST, "/contexts/{name}/query"),
        (Method::GET, "/contexts/{name}"),
        (Method::GET, "/contexts/{name}/sources"),
        (Method::GET, "/contexts/{name}/embeddings"),
    ] {
        assert_eq!(
            crate::auth::required_role(&method, route),
            crate::auth::Role::Read,
            "{route}"
        );
    }
}
