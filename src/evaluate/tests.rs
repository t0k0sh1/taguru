use axum::http::Method;

use super::*;
use crate::api::evidence::select::REASON_BUDGET_EXCEEDED;

#[test]
fn every_usage_variable_is_a_known_key() {
    // This command's own USAGE is invisible to cli.rs's consistency
    // tests: a variable documented here but missing from KNOWN_KEYS
    // would make --config warn "typo?" on a perfectly valid setting.
    crate::config::assert_usage_vars_are_known_keys(USAGE);
}

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

// ============================= #308: --assembly ============================

#[test]
fn without_assembly_or_budget_flags_the_run_config_is_untouched() {
    let parsed = args(&["--eval", "a.jsonl", "--context", "sake"]).unwrap();
    assert!(!parsed.assembly);
    assert!(parsed.budget.is_none());
    assert!(parsed.rerank.is_none());
}

#[test]
fn assembly_alone_is_accepted_with_no_budget() {
    let parsed = args(&["--eval", "a.jsonl", "--context", "sake", "--assembly"]).unwrap();
    assert!(parsed.assembly);
    assert!(parsed.budget.is_none());
}

#[test]
fn one_budget_flag_is_enough_to_populate_budget() {
    let parsed = args(&["--eval", "a.jsonl", "--context", "sake", "--max-items", "5"]).unwrap();
    let budget = parsed.budget.expect("--max-items alone must set budget");
    assert_eq!(budget.max_items, Some(5));
    assert_eq!(budget.max_bytes, None);
    assert_eq!(budget.max_tokens, None);
}

#[test]
fn zero_or_non_numeric_budget_values_are_rejected_for_every_flag() {
    for flag in ["--max-items", "--max-bytes", "--max-tokens"] {
        for value in ["0", "abc"] {
            assert_eq!(
                args(&["--eval", "a.jsonl", "--context", "sake", flag, value]).unwrap_err(),
                2,
                "{flag} {value}"
            );
        }
    }
}

#[test]
fn a_duplicate_assembly_flag_is_refused() {
    assert_eq!(
        args(&[
            "--eval",
            "a.jsonl",
            "--context",
            "sake",
            "--assembly",
            "--assembly"
        ])
        .unwrap_err(),
        2
    );
}

#[test]
fn a_duplicate_budget_or_rerank_flag_is_refused() {
    for flag in ["--max-items", "--max-bytes", "--max-tokens"] {
        assert_eq!(
            args(&[
                "--eval",
                "a.jsonl",
                "--context",
                "sake",
                flag,
                "5",
                flag,
                "5"
            ])
            .unwrap_err(),
            2,
            "{flag}"
        );
    }
    assert_eq!(
        args(&[
            "--eval",
            "a.jsonl",
            "--context",
            "sake",
            "--assembly",
            "--rerank",
            "m",
            "--rerank",
            "m"
        ])
        .unwrap_err(),
        2
    );
}

#[test]
fn rerank_without_assembly_is_a_usage_error() {
    assert_eq!(
        args(&[
            "--eval",
            "a.jsonl",
            "--context",
            "sake",
            "--rerank",
            "some-model"
        ])
        .unwrap_err(),
        2
    );
}

#[test]
fn rerank_with_assembly_is_accepted() {
    let parsed = args(&[
        "--eval",
        "a.jsonl",
        "--context",
        "sake",
        "--assembly",
        "--rerank",
        "some-model",
    ])
    .unwrap();
    assert_eq!(parsed.rerank.as_deref(), Some("some-model"));
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
fn mask_url_redacts_an_unparsable_base_instead_of_echoing_it() {
    assert_eq!(mask_url("not a url"), "<unparseable-url>");
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
        types: None,
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

/// #308's `--max-bytes`/`--max-tokens` baseline truncation
/// (`evidence::truncate_to_budget`) measures each hit's real
/// `PassageHit.text` byte/token cost — the fallback path must recover
/// that text when the raw hit still carries it, not silently zero it
/// out the way the (now-removed) "never read downstream" shortcut did.
/// Regression for a bug that made every fallback-path hit measure as
/// ~55-60 bytes regardless of its real size, letting `--max-bytes`
/// admit far more hits than the ceiling should allow.
#[test]
fn extract_passages_fallback_recovers_text_when_the_raw_hit_still_carries_it() {
    let value = serde_json::json!([
        {"source": "corpus/brewery.md", "paragraph": 0, "score": 0.5, "text": "青嶺酒造は雲居県霧沢町の蔵元である。"}
    ]);
    let (hits, _plan) = extract_passages(&value).expect("a bare hits array must still parse");
    assert_eq!(hits[0].text, "青嶺酒造は雲居県霧沢町の蔵元である。");
}

/// A server whose bare hit objects carry no `text` field at all
/// degrades to an empty string — unavoidable with no data to measure,
/// but a documented degrade rather than a panic or an error.
#[test]
fn extract_passages_fallback_degrades_to_empty_text_when_absent() {
    let value = serde_json::json!([
        {"source": "corpus/brewery.md", "paragraph": 0, "score": 0.5}
    ]);
    let (hits, _plan) = extract_passages(&value).expect("a bare hits array must still parse");
    assert_eq!(hits[0].text, "");
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

/// #308: an assembled package's `items[]` embeds the *existing*
/// `PassageHit` (with its full body `text`) verbatim as its
/// kind-specific payload (ADR 0006 §10) — neither of #308's two
/// projections of that item, [`evidence_locators`] (the diagnostic
/// `EvidenceOutcome.items` block) nor [`hits_from_evidence_items`]
/// (the scoring-facing `HitLocator` projection), may let that text
/// reach `evaluation.json` (ADR 0004 §11's no-corpus-text rule).
#[test]
fn evidence_locator_never_carries_the_passage_body_text() {
    let item = EvidenceItem {
        candidate_id: "passage::corpus/brewery.md::0".to_string(),
        kind: "passage".to_string(),
        fused_rank: 1,
        lane_ranks: vec![crate::api::evidence::LaneRank {
            lane: "passage_bm25".to_string(),
            rank: 1,
        }],
        citation_refs: vec![CitationRef {
            source: "corpus/brewery.md".to_string(),
            paragraph: 0,
        }],
        corroboration: None,
        contradicts: Vec::new(),
        bytes: Some(0),
        estimated_tokens: Some(0),
        association: None,
        passage: Some(PassageHit {
            source: "corpus/brewery.md".to_string(),
            paragraph: 0,
            score: 1.0,
            text: "TOP-SECRET-PASSAGE-BODY-TEXT".to_string(),
            lanes: PassageLanes {
                bm25: None,
                vector: None,
            },
        }),
        community: None,
    };

    let locators = evidence_locators(std::slice::from_ref(&item));
    let locators_value = serde_json::to_value(&locators).unwrap();
    assert_no_body_text(&locators_value, "TOP-SECRET-PASSAGE-BODY-TEXT", "$");

    let hits = hits_from_evidence_items(std::slice::from_ref(&item), 10);
    let hits_value = serde_json::to_value(&hits).unwrap();
    assert_no_body_text(&hits_value, "TOP-SECRET-PASSAGE-BODY-TEXT", "$");
}

/// A minimal passage-kind `EvidenceItem` — one `HitLocator` when
/// flattened, since a passage's `citation_refs` is always empty
/// (ADR 0006 §6, self-citing).
fn passage_evidence_item(source: &str, paragraph: u32, rank: usize) -> EvidenceItem {
    EvidenceItem {
        candidate_id: format!("passage\u{0}ctx\u{0}{source}\u{0}{paragraph}"),
        kind: "passage".to_string(),
        fused_rank: rank,
        lane_ranks: vec![crate::api::evidence::LaneRank {
            lane: "passage_bm25".to_string(),
            rank,
        }],
        citation_refs: Vec::new(),
        corroboration: None,
        contradicts: Vec::new(),
        bytes: Some(0),
        estimated_tokens: Some(0),
        association: None,
        passage: Some(PassageHit {
            source: source.to_string(),
            paragraph,
            score: 1.0,
            text: String::new(),
            lanes: PassageLanes {
                bm25: None,
                vector: None,
            },
        }),
        community: None,
    }
}

/// Regression: an assembly-mode case's `items[]` can carry up to
/// `budget.max_items` (default 40) admitted candidates — structurally
/// unrelated to the case's own `options.limit`/`default_limit` — so
/// `hits_from_evidence_items` must cap its output at `limit` the same
/// way baseline's `sources/search` call already bounds its `hits`
/// response server-side. Without the cap, `hits.len()` could exceed
/// `limit`, changing the effective `k` for `recall_at_k`/`mrr`/`ndcg`
/// between a `baseline`/`assembly` run pair at the identical `--limit`.
#[test]
fn hits_from_evidence_items_never_exceeds_the_case_limit() {
    let items: Vec<EvidenceItem> = (0..5)
        .map(|i| passage_evidence_item("corpus/brewery.md", i, i as usize + 1))
        .collect();
    let hits = hits_from_evidence_items(&items, 3);
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].paragraph, 0);
    assert_eq!(hits[2].paragraph, 2);
}

// ========================= rerank.ran denominator =========================

/// A one-case `CaseBlock` whose passage lane searched and found
/// nothing — the twelve fields every metrics/summary test here shares,
/// with only the three assembly-lane fields varying per test.
fn searched_case(
    evidence: Option<EvidenceOutcome>,
    budget: Option<BudgetAccounting>,
    diversity_sources: Option<usize>,
) -> CaseBlock {
    CaseBlock {
        case_id: "c1".to_string(),
        query: "q".to_string(),
        cues: Vec::new(),
        limit: 10,
        passage: PassageOutcome::Searched {
            plan: None,
            hits: Vec::new(),
            latency_ms: 1,
        },
        structural: None,
        recall: None,
        coverage: None,
        lane_cross: None,
        citations: None,
        missed: Vec::new(),
        missed_truncated: 0,
        evidence,
        budget,
        diversity_sources,
    }
}

/// Regression: `RerankerPlan::not_requested` (`assemble.rs`) echoes the
/// *server's* own configuration state (`state.reranker().is_some()`)
/// into `configured`, even for a request whose body carried no
/// `rerank` field at all — a run against a rerank-configured server
/// that never itself passed `--rerank` still gets back
/// `{configured: true, ran: false, reason: None}` for every case.
/// `rerank.ran`'s denominator must gate on `rerank_requested_this_run`
/// (this run's own `--rerank` flag), never on that echoed
/// `reranker.configured`, or a `--rerank`-less run would wrongly count
/// toward "empty when `--rerank` was not given" (`build_definitions`'
/// own documented contract for this metric).
#[test]
fn rerank_ran_ignores_a_server_side_configured_reranker_when_this_run_never_asked_for_one() {
    let case = searched_case(
        Some(EvidenceOutcome::Assembled {
            latency_ms: 1,
            items: Vec::new(),
            omitted_by_reason: BTreeMap::new(),
            selection: SelectionPlan {
                dedup_dropped: 0,
                contradiction_groups: 0,
                diversity_tier_width: 10,
            },
            reranker: RerankerPlan {
                configured: true,
                ran: false,
                model: None,
                reason: None,
            },
        }),
        None,
        None,
    );

    let metrics = build_metrics(&[case], false);
    let rerank_ran = metrics.get("rerank.ran").expect("metric key exists");
    assert_eq!(rerank_ran.sample_size(), 0, "{rerank_ran:?}");
}

// ========================= assembly stdout summary =========================

fn inputs_block(budget: Option<BudgetLimits>, rerank: Option<&str>) -> InputsBlock {
    InputsBlock {
        eval: EvalInputsBlock {
            path: "eval.jsonl".to_string(),
            name: None,
            cases: 1,
        },
        context: "sake".to_string(),
        url: "http://localhost:8080".to_string(),
        out: "evaluation.json".to_string(),
        default_limit: 10,
        resolve_limit: 5,
        mode: if budget.is_some() {
            "assembly"
        } else {
            "baseline"
        }
        .to_string(),
        budget,
        rerank: rerank.map(str::to_string),
    }
}

/// #308's data (budget usage, `diversity.sources`, `rerank.ran`) must
/// reach stdout, not only `evaluation.json` — a CI operator reading
/// the console alone should see the equal-budget and reranker outcome
/// of an `--assembly` run.
#[test]
fn assembly_summary_prints_budget_diversity_and_rerank_lines() {
    let limits = BudgetLimits {
        max_items: 40,
        max_bytes: 65536,
        max_tokens: 4000,
    };
    let mut omitted_by_reason = BTreeMap::new();
    omitted_by_reason.insert(REASON_BUDGET_EXCEEDED.to_string(), 1);
    let case = searched_case(
        Some(EvidenceOutcome::Assembled {
            latency_ms: 1,
            items: Vec::new(),
            omitted_by_reason,
            selection: SelectionPlan {
                dedup_dropped: 0,
                contradiction_groups: 0,
                diversity_tier_width: 10,
            },
            reranker: RerankerPlan {
                configured: true,
                ran: true,
                model: Some("model-x".to_string()),
                reason: None,
            },
        }),
        Some(BudgetAccounting {
            usage: BudgetUsage {
                items_used: 3,
                bytes_used: 500,
                tokens_used: 100,
                limits,
            },
            omitted_total: 1,
        }),
        Some(2),
    );

    let metrics = build_metrics(&[case], true);
    let lines = assembly_summary_lines(&inputs_block(Some(limits), Some("model-x")), &metrics);
    assert_eq!(
        lines,
        vec![
            "  budget over 1 case(s): mean 3.0 item(s) / 500.0 byte(s) / 100.0 token(s) used, \
             budget-omitted rate 0.250"
                .to_string(),
            "  diversity over 1 case(s): mean 2.0 distinct source(s) in admitted evidence"
                .to_string(),
            "  rerank 'model-x' over 1 attempted case(s): 1 ran, 0 degraded".to_string(),
        ]
    );
}

/// A plain baseline run (no budget flag, no `--rerank`, no assembly
/// lane) must add nothing to the summary — the pre-#308 stdout shape
/// is unchanged.
#[test]
fn a_plain_baseline_run_adds_no_assembly_summary_lines() {
    let case = searched_case(None, None, None);

    let metrics = build_metrics(&[case], false);
    let lines = assembly_summary_lines(&inputs_block(None, None), &metrics);
    assert!(lines.is_empty(), "{lines:?}");
}

// ========================= metrics <-> definitions agreement =========================

#[test]
fn every_metric_key_has_a_matching_definition_and_vice_versa() {
    let cases: Vec<CaseBlock> = Vec::new();
    let metrics = build_metrics(&cases, false);
    let definitions = build_definitions();
    let metric_keys: BTreeSet<&String> = metrics.keys().collect();
    let definition_keys: BTreeSet<&String> = definitions.keys().collect();
    assert_eq!(metric_keys, definition_keys);
}

// ========================= No answer-generation LLM seam (ADR §12, AC 8) =========================

/// Cuts a `include_str!`-ed source at its first `#[cfg(test)]`, so a
/// file whose test module lives inline (`thresholds.rs`, which embeds
/// `mod tests { ... }` rather than declaring a sibling file) never has
/// its own test fixtures scanned — only production code is asserted
/// clean. Files whose tests live in a sibling module (`evaluate.rs`,
/// `compare.rs`, both `mod tests;`) are untouched by this: the two
/// lines declaring the sibling carry no banned literal themselves.
fn production_only(source: &str) -> &str {
    match source.find("#[cfg(test)]") {
        Some(at) => &source[..at],
        None => source,
    }
}

/// Whether `source` references `crate::{module}` through any import
/// shape: a plain `crate::{module}::...` path, or `module` appearing
/// inside a single-level grouped import like `use crate::{extract,
/// embedding};`. Not a full parser — no nested groups, no `as`-alias
/// tracking — deliberately a step up from a pure substring search, not
/// a syntax tree: this guard exists to catch an INADVERTENT seam, and
/// matches the rest of this codebase's plain string-based source
/// checks (`cli.rs`'s `every_documented_variable_is_a_known_key`)
/// rather than pulling in a parser crate for one test.
fn references_module(source: &str, module: &str) -> bool {
    // A raw substring search on `crate::{module}` alone would false-
    // positive on an unrelated module sharing the prefix (a
    // hypothetical `crate::extraction` for `module: "extract"`), so
    // every direct-path match must be followed by a non-identifier
    // byte (`::`, `;`, whitespace, …) or end of input — the same
    // boundary the grouped-import branch below already checks via
    // `member == module`.
    let plain = format!("crate::{module}");
    let mut search_from = 0;
    while let Some(relative) = source[search_from..].find(plain.as_str()) {
        let at = search_from + relative;
        let after = at + plain.len();
        let boundary = source[after..]
            .chars()
            .next()
            .is_none_or(|ch| !(ch.is_alphanumeric() || ch == '_'));
        if boundary {
            return true;
        }
        search_from = at + 1;
    }
    let mut rest = source;
    while let Some(start) = rest.find("crate::{") {
        let after = &rest[start + "crate::{".len()..];
        let Some(end) = after.find('}') else {
            break;
        };
        let group = &after[..end];
        if group.split(',').any(|member| {
            let member = member.trim();
            member == module || member.starts_with(&format!("{module}::"))
        }) {
            return true;
        }
        rest = &after[end + 1..];
    }
    false
}

#[test]
fn evaluate_module_never_names_an_extraction_or_embedding_seam() {
    // Scans evaluate.rs and its submodules — ADR 0004 §12's structural
    // enforcement of AC 8 covers the whole `mod evaluate` tree, not
    // just its top file, so a future submodule can't quietly grow the
    // seam this check exists to rule out. None of these three
    // `include_str!`s pulls in this file itself (tests.rs), so this
    // check's own literals below are never scanned.
    let sources = [
        production_only(include_str!("../evaluate.rs")),
        production_only(include_str!("../evaluate/compare.rs")),
        production_only(include_str!("../evaluate/evidence.rs")),
        production_only(include_str!("../evaluate/thresholds.rs")),
    ];
    // Built by concatenation so this assertion's own literals never
    // accidentally match themselves.
    let extract_prefix = concat!("TAGURU_", "EXTRACT_");
    let embed_prefix = concat!("TAGURU_", "EMBED_");
    for source in sources {
        assert!(
            !source.contains(extract_prefix),
            "found {extract_prefix} in the evaluate module tree"
        );
        assert!(
            !source.contains(embed_prefix),
            "found {embed_prefix} in the evaluate module tree"
        );
        assert!(
            !references_module(source, "extract"),
            "found a crate::extract reference (plain or grouped import) in the evaluate module \
             tree"
        );
        assert!(
            !references_module(source, "embedding"),
            "found a crate::embedding reference (plain or grouped import) in the evaluate \
             module tree"
        );
    }
}

/// Guards against the `sources` array above silently drifting behind
/// `evaluate.rs`'s own submodule list: a future `mod foo;` added there
/// without a matching `include_str!` above would otherwise never be
/// scanned for the extract/embedding seam at all, despite the seam
/// test's own comment claiming whole-tree coverage. `tests` is
/// declared under `#[cfg(test)]` and is not production code the seam
/// scan needs to cover; every other declared submodule must be named
/// here.
#[test]
fn the_seam_scan_names_every_production_submodule_evaluate_rs_declares() {
    let source = include_str!("../evaluate.rs");
    let declared: BTreeSet<&str> = source
        .lines()
        .map(str::trim)
        .filter_map(|line| line.strip_prefix("mod ")?.strip_suffix(';'))
        .filter(|name| *name != "tests")
        .collect();
    assert_eq!(
        declared,
        BTreeSet::from(["compare", "evidence", "thresholds"]),
        "evaluate.rs declares a submodule the seam-scan test's `sources` array does not name — \
         add it there too"
    );
}

#[test]
fn references_module_catches_a_grouped_import_the_plain_substring_check_would_miss() {
    assert!(references_module(
        "use crate::{extract, embedding};",
        "extract"
    ));
    assert!(references_module(
        "use crate::{extract, embedding};",
        "embedding"
    ));
    assert!(references_module(
        "use crate::{other, extract::sha256_hex};",
        "extract"
    ));
    assert!(references_module(
        "use crate::extract::sha256_hex;",
        "extract"
    ));
    assert!(!references_module("use crate::{sha256, hash};", "extract"));
    // A name merely containing "extract" as a substring (e.g. a future
    // `extraction` module) must not false-positive — neither in a
    // grouped import nor, since it shares the same "crate::extract"
    // prefix, in a direct path.
    assert!(!references_module("use crate::{extraction};", "extract"));
    assert!(!references_module("use crate::extraction::Foo;", "extract"));
    // But a real direct-path reference immediately followed by more of
    // the SAME name (not a `::` continuation) is exactly what the
    // boundary check above exists to tell apart from this case.
    assert!(references_module(
        "use crate::extract::sha256_hex;",
        "extract"
    ));
}

// ========================= Scoring test fixtures (#274) =========================

fn base_case(case_id: &str) -> EvalCase {
    EvalCase {
        case_id: case_id.to_string(),
        query: "q".to_string(),
        cues: Vec::new(),
        expected_sources: Vec::new(),
        expected_concepts: Vec::new(),
        options: evalset::EvalOptions::default(),
        expected_labels: Vec::new(),
        expected_associations: Vec::new(),
        expected_citations: Vec::new(),
    }
}

fn expected_source(source: &str, paragraphs: &[u32], relevance: u8) -> ExpectedSource {
    ExpectedSource {
        source: source.to_string(),
        paragraphs: paragraphs.to_vec(),
        relevance,
    }
}

fn hit_with_score(source: &str, paragraph: u32, score: f32) -> HitLocator {
    HitLocator {
        source: source.to_string(),
        paragraph,
        score,
        lanes: PassageLanes {
            bm25: None,
            vector: None,
        },
    }
}

fn hit(source: &str, paragraph: u32) -> HitLocator {
    hit_with_score(source, paragraph, 0.0)
}

fn cue_resolution(kind: &'static str, names: &[&str]) -> CueResolution {
    CueResolution {
        cue: "cue".to_string(),
        kind,
        resolved_names: names.iter().map(|s| s.to_string()).collect(),
        tier: Some("lexical".to_string()),
        limit: RESOLVE_LIMIT,
        latency_ms: 1,
        error: None,
    }
}

fn position_resolved(name: &str) -> PositionOutcome {
    PositionOutcome::Resolved {
        name: name.to_string(),
        tier: "lexical".to_string(),
        latency_ms: 1,
    }
}

fn association_probe(query: Option<QueryProbe>) -> AssociationProbe {
    AssociationProbe {
        subject_cue: "s".to_string(),
        label_cue: "l".to_string(),
        object_cue: "o".to_string(),
        subject: position_resolved("s"),
        label: position_resolved("l"),
        object: position_resolved("o"),
        query,
    }
}

fn expected_citation(
    source: &str,
    paragraph: u32,
    section: Option<Option<&str>>,
    quote: Option<&str>,
) -> ExpectedCitation {
    ExpectedCitation {
        source: source.to_string(),
        paragraph,
        section: section.map(|inner| inner.map(str::to_string)),
        quote: quote.map(str::to_string),
    }
}

fn resolved_outcome(section: SectionCheck, quote: Option<QuoteCheck>) -> CitationOutcome {
    CitationOutcome::Resolved { section, quote }
}

fn unresolved_outcome(code: Option<&str>) -> CitationOutcome {
    CitationOutcome::Unresolved {
        code: code.map(str::to_string),
        message: "boom".to_string(),
    }
}

fn citation_check(
    source: &str,
    paragraph: u32,
    served: bool,
    outcome: CitationOutcome,
) -> CitationCheck {
    CitationCheck {
        source: source.to_string(),
        paragraph,
        served,
        outcome,
        latency_ms: 1,
    }
}

// ========================= score_recall: recall@k / MRR / graded nDCG =========================

#[test]
fn score_recall_matches_the_worked_example_from_the_274_design_discussion() {
    // A(rel=3, no paragraph restriction) matched at rank 0; B(rel=1)
    // matched at rank 2 — the plan's own worked example.
    let expected = vec![
        expected_source("corpus/a.md", &[], 3),
        expected_source("corpus/b.md", &[], 1),
    ];
    let hits = vec![
        hit("corpus/a.md", 0),
        hit("corpus/a.md", 1),
        hit("corpus/b.md", 4),
    ];
    let recall = score_recall(&expected, &hits).expect("a relevance >= 1 entry exists");
    assert_eq!(recall.expected_total, 2);
    assert_eq!(recall.matched, 2);
    assert!((recall.recall_at_k - 1.0).abs() < 1e-9, "{recall:?}");
    assert!((recall.mrr - 1.0).abs() < 1e-9, "{recall:?}");
    // DCG = 3/log2(2) + 1/log2(4) = 3.5; IDCG = 3/log2(2) + 1/log2(3) ≈ 3.6309
    assert!((recall.ndcg - 0.9639).abs() < 1e-3, "{recall:?}");
}

#[test]
fn score_recall_is_one_when_hits_arrive_in_ideal_relevance_order() {
    let expected = vec![
        expected_source("corpus/a.md", &[], 3),
        expected_source("corpus/b.md", &[], 1),
    ];
    let hits = vec![hit("corpus/a.md", 0), hit("corpus/b.md", 0)];
    let recall = score_recall(&expected, &hits).unwrap();
    assert!((recall.ndcg - 1.0).abs() < 1e-9, "{recall:?}");
}

#[test]
fn score_recall_drops_below_one_when_the_higher_relevance_hit_ranks_lower() {
    let expected = vec![
        expected_source("corpus/a.md", &[], 3),
        expected_source("corpus/b.md", &[], 1),
    ];
    // Reversed from ideal: the low-relevance source now outranks the
    // high-relevance one.
    let hits = vec![hit("corpus/b.md", 0), hit("corpus/a.md", 0)];
    let recall = score_recall(&expected, &hits).unwrap();
    assert!(recall.ndcg < 1.0, "{recall:?}");
}

#[test]
fn score_recall_drops_a_relevance_zero_entry_from_the_denominator() {
    let expected = vec![
        expected_source("corpus/a.md", &[], 1),
        expected_source("corpus/ignored.md", &[], 0),
    ];
    let hits = vec![hit("corpus/a.md", 0)];
    let recall = score_recall(&expected, &hits).unwrap();
    assert_eq!(recall.expected_total, 1);
}

#[test]
fn score_recall_is_none_when_every_entry_has_relevance_zero() {
    let expected = vec![expected_source("corpus/ignored.md", &[], 0)];
    assert!(score_recall(&expected, &[]).is_none());
}

#[test]
fn score_recall_honors_a_paragraph_restricted_expectation() {
    let expected = vec![expected_source("corpus/a.md", &[2], 1)];
    let wrong_paragraph = vec![hit("corpus/a.md", 0)];
    assert_eq!(
        score_recall(&expected, &wrong_paragraph).unwrap().matched,
        0
    );
    let right_paragraph = vec![hit("corpus/a.md", 2)];
    assert_eq!(
        score_recall(&expected, &right_paragraph).unwrap().matched,
        1
    );
}

#[test]
fn score_recall_ignores_a_hit_locators_own_score_field() {
    let expected = vec![expected_source("corpus/a.md", &[], 2)];
    let low = vec![hit_with_score("corpus/a.md", 0, 0.01)];
    let high = vec![hit_with_score("corpus/a.md", 0, 99.0)];
    let low_result = score_recall(&expected, &low).unwrap();
    let high_result = score_recall(&expected, &high).unwrap();
    assert_eq!(low_result.recall_at_k, high_result.recall_at_k);
    assert_eq!(low_result.mrr, high_result.mrr);
    assert_eq!(low_result.ndcg, high_result.ndcg);
}

#[test]
fn score_recall_clamps_ndcg_to_one_when_two_expectations_share_a_hit() {
    // Both a wildcard (`paragraphs: []`) entry and a second entry on
    // the same source are satisfied by the single hit at rank 0 — DCG
    // credits both at that one rank, while IDCG spreads them across
    // ranks 0 and 1, so the raw ratio would exceed 1.0 without a
    // clamp.
    let expected = vec![
        expected_source("corpus/a.md", &[], 3),
        expected_source("corpus/a.md", &[], 2),
    ];
    let hits = vec![hit("corpus/a.md", 0)];
    let recall = score_recall(&expected, &hits).unwrap();
    assert!((recall.ndcg - 1.0).abs() < 1e-9, "{recall:?}");
}

// ========================= coverage_counts / resolved_contains =========================

#[test]
fn coverage_counts_folds_katakana_and_hiragana_via_normalize_entry() {
    let resolutions = [cue_resolution("concept", &["りんご"])];
    let refs: Vec<&CueResolution> = resolutions.iter().collect();
    let expected = vec!["リンゴ".to_string()];
    let coverage = coverage_counts(&expected, &refs).expect("a non-empty expectation list");
    assert_eq!(coverage.expected, 1);
    assert_eq!(coverage.matched, 1);
    assert!((coverage.value - 1.0).abs() < 1e-9);
}

#[test]
fn coverage_counts_folds_full_width_romaji_via_normalize_entry() {
    let resolutions = [cue_resolution("label", &["apple"])];
    let refs: Vec<&CueResolution> = resolutions.iter().collect();
    let expected = vec!["Ａｐｐｌｅ".to_string()];
    let coverage = coverage_counts(&expected, &refs).unwrap();
    assert_eq!(coverage.matched, 1);
}

#[test]
fn coverage_counts_is_none_when_the_expectation_list_is_empty() {
    assert!(coverage_counts(&[], &[]).is_none());
}

#[test]
fn coverage_counts_reports_a_partial_match() {
    let resolutions = [cue_resolution("concept", &["青嶺酒造"])];
    let refs: Vec<&CueResolution> = resolutions.iter().collect();
    let expected = vec!["青嶺酒造".to_string(), "存在しない".to_string()];
    let coverage = coverage_counts(&expected, &refs).unwrap();
    assert_eq!(coverage.expected, 2);
    assert_eq!(coverage.matched, 1);
    assert!((coverage.value - 0.5).abs() < 1e-9);
}

// ========================= association_coverage =========================

#[test]
fn association_coverage_counts_only_queried_entries_with_a_hit() {
    let associations = vec![
        association_probe(Some(QueryProbe::Queried {
            total: 1,
            matches: 1,
            attributions: Vec::new(),
            latency_ms: 1,
        })),
        association_probe(Some(QueryProbe::Queried {
            total: 0,
            matches: 0,
            attributions: Vec::new(),
            latency_ms: 1,
        })),
        association_probe(None),
    ];
    let coverage = association_coverage(&associations).expect("a non-empty association list");
    assert_eq!(coverage.expected, 3);
    assert_eq!(coverage.matched, 1);
}

#[test]
fn association_coverage_is_none_when_there_are_no_associations() {
    assert!(association_coverage(&[]).is_none());
}

// ========================= Citation lane (ADR §8) =========================

#[test]
fn check_section_is_not_checked_when_the_key_is_absent() {
    assert!(matches!(
        check_section(&None, &Some("沿革".to_string())),
        SectionCheck::NotChecked
    ));
}

#[test]
fn check_section_matches_an_explicit_null_against_no_stored_section() {
    // `Some(None)` asserts "outside every stored section" — a real,
    // checkable claim (ADR 0004 §8), not a serde artifact.
    assert!(matches!(
        check_section(&Some(None), &None),
        SectionCheck::Matched { expected: None }
    ));
}

#[test]
fn check_section_flags_an_explicit_null_mismatched_against_a_real_section() {
    assert!(matches!(
        check_section(&Some(None), &Some("沿革".to_string())),
        SectionCheck::Mismatched { expected: None }
    ));
}

#[test]
fn check_section_matches_an_equal_declared_value() {
    let expected = Some(Some("沿革".to_string()));
    assert!(matches!(
        check_section(&expected, &Some("沿革".to_string())),
        SectionCheck::Matched { .. }
    ));
}

#[test]
fn check_section_flags_a_differing_declared_value() {
    let expected = Some(Some("沿革".to_string()));
    assert!(matches!(
        check_section(&expected, &Some("製品ラインナップ".to_string())),
        SectionCheck::Mismatched { .. }
    ));
}

#[test]
fn quote_matches_a_plain_substring() {
    assert!(quote_matches(
        "1897年に創業",
        "青嶺は1897年に創業した蔵元です。"
    ));
}

#[test]
fn quote_matches_folds_katakana_hiragana_and_width_via_normalize_entry() {
    assert!(quote_matches("ｱｯﾌﾟﾙ", "りんご、あっぷる、青嶺"));
}

#[test]
fn quote_matches_is_false_for_unrelated_text() {
    assert!(!quote_matches(
        "存在しない引用",
        "青嶺は1897年に創業した蔵元です。"
    ));
}

#[test]
fn quote_matches_never_succeeds_across_a_paragraph_boundary() {
    // `Citation.text` is exactly one paragraph (`sources.rs:84-88`); a
    // quote that spans two paragraphs cannot be a substring of either
    // one alone. ADR 0004 §8's documented workaround is splitting it
    // into two `expected_citations` entries.
    let paragraph_one = "青嶺は1897年に創業した蔵元";
    let paragraph_two = "です。醸造元は青嶺酒造。";
    let spanning_quote = "創業した蔵元です";
    assert!(!quote_matches(spanning_quote, paragraph_one));
    assert!(!quote_matches(spanning_quote, paragraph_two));
}

#[test]
fn citation_is_valid_requires_resolution_and_no_section_mismatch_and_a_matched_quote() {
    let resolved_clean = citation_check(
        "corpus/a.md",
        3,
        true,
        resolved_outcome(SectionCheck::NotChecked, None),
    );
    assert!(citation_is_valid(&resolved_clean));

    let section_mismatch = citation_check(
        "corpus/a.md",
        3,
        true,
        resolved_outcome(
            SectionCheck::Mismatched {
                expected: Some("沿革".to_string()),
            },
            None,
        ),
    );
    assert!(!citation_is_valid(&section_mismatch));

    let quote_mismatch = citation_check(
        "corpus/a.md",
        3,
        true,
        resolved_outcome(
            SectionCheck::NotChecked,
            Some(QuoteCheck {
                declared: "存在しない".to_string(),
                matched: false,
            }),
        ),
    );
    assert!(!citation_is_valid(&quote_mismatch));

    let unresolved = citation_check(
        "corpus/a.md",
        3,
        false,
        unresolved_outcome(Some("no_source")),
    );
    assert!(!citation_is_valid(&unresolved));
}

#[test]
fn score_citation_recall_counts_only_served_checks() {
    let checks = vec![
        citation_check(
            "a.md",
            0,
            true,
            resolved_outcome(SectionCheck::NotChecked, None),
        ),
        citation_check(
            "b.md",
            1,
            false,
            resolved_outcome(SectionCheck::NotChecked, None),
        ),
    ];
    let recall = score_citation_recall(&checks);
    assert_eq!(recall.expected_total, 2);
    assert_eq!(recall.matched, 1);
    assert!((recall.value - 0.5).abs() < 1e-9);
}

#[test]
fn score_citation_validity_counts_only_valid_checks_independent_of_served() {
    let checks = vec![
        // Not served, but a perfectly valid locator — recall and
        // validity are independent, never merged into one score.
        citation_check(
            "a.md",
            0,
            false,
            resolved_outcome(SectionCheck::NotChecked, None),
        ),
        citation_check("b.md", 1, true, unresolved_outcome(Some("no_paragraph"))),
    ];
    let validity = score_citation_validity(&checks);
    assert_eq!(validity.expected_total, 2);
    assert_eq!(validity.valid, 1);
    assert!((validity.value - 0.5).abs() < 1e-9);
}

#[test]
fn served_locators_unions_passage_hits_and_structural_attributions() {
    let passage = PassageOutcome::Searched {
        plan: None,
        hits: vec![hit("a.md", 0)],
        latency_ms: 1,
    };
    let structural = StructuralBlock {
        cues: Vec::new(),
        associations: vec![association_probe(Some(QueryProbe::Queried {
            total: 1,
            matches: 1,
            attributions: vec![AttributionLocator {
                source: "b.md".to_string(),
                paragraph: Some(2),
            }],
            latency_ms: 1,
        }))],
    };
    let served = served_locators(&passage, Some(&structural));
    assert!(served.contains(&("a.md".to_string(), 0)));
    assert!(served.contains(&("b.md".to_string(), 2)));
    assert_eq!(served.len(), 2);
}

#[test]
fn served_locators_ignores_an_attribution_with_no_paragraph() {
    let passage = PassageOutcome::Failed {
        message: "boom".to_string(),
        latency_ms: 1,
    };
    let structural = StructuralBlock {
        cues: Vec::new(),
        associations: vec![association_probe(Some(QueryProbe::Queried {
            total: 1,
            matches: 1,
            attributions: vec![AttributionLocator {
                source: "b.md".to_string(),
                paragraph: None,
            }],
            latency_ms: 1,
        }))],
    };
    assert!(served_locators(&passage, Some(&structural)).is_empty());
}

#[test]
fn score_case_recall_is_computed_from_citation_checks_even_when_the_passage_lane_failed() {
    // ADR 0004 §8: citation recall and locator validity are computed
    // independent of whether that case's own search happened to
    // surface anything — a failed passage lane still yields a
    // (well-defined, zero) citation recall, unlike expected_sources
    // recall, which is `None` in that situation.
    let mut case = base_case("c1");
    case.expected_citations = vec![expected_citation("a.md", 0, None, None)];
    let passage = PassageOutcome::Failed {
        message: "boom".to_string(),
        latency_ms: 1,
    };
    let checks = vec![citation_check(
        "a.md",
        0,
        false,
        resolved_outcome(SectionCheck::NotChecked, None),
    )];
    let scores = score_case(&case, &passage, None, &checks);
    let citations = scores.citations.expect("expected_citations was declared");
    assert_eq!(citations.recall.matched, 0);
    assert_eq!(citations.validity.valid, 1);
}

#[test]
fn score_case_citations_is_none_when_no_expected_citations_are_declared() {
    let case = base_case("c1");
    let passage = PassageOutcome::Searched {
        plan: None,
        hits: Vec::new(),
        latency_ms: 1,
    };
    let scores = score_case(&case, &passage, None, &[]);
    assert!(scores.citations.is_none());
}

#[test]
fn build_missed_reports_a_missed_citation_recall_and_a_failed_locator_validity_separately() {
    let case = base_case("c1");
    let checks = vec![
        // Not served, but a valid locator: only a recall miss.
        citation_check(
            "a.md",
            0,
            false,
            resolved_outcome(SectionCheck::NotChecked, None),
        ),
    ];
    let (missed, _) = build_missed(&case, None, &[], &[], None, &checks);
    assert_eq!(missed.len(), 1, "{missed:?}");
    assert!(
        missed[0].contains("not found among served results"),
        "{missed:?}"
    );
}

#[test]
fn build_missed_reports_both_failure_modes_for_a_check_that_is_neither_served_nor_valid() {
    let case = base_case("c1");
    let checks = vec![citation_check(
        "a.md",
        0,
        false,
        unresolved_outcome(Some("no_source")),
    )];
    let (missed, _) = build_missed(&case, None, &[], &[], None, &checks);
    assert_eq!(missed.len(), 2, "{missed:?}");
    assert!(
        missed[0].contains("not found among served results"),
        "{missed:?}"
    );
    assert!(missed[1].contains("failed locator validity"), "{missed:?}");
}

/// The one ADR 0004 §11 exception to "no corpus body text": even a
/// `quote` mismatch records only the user's own declared quote and a
/// boolean, never the served paragraph body.
#[test]
fn citation_check_never_carries_the_served_paragraph_text() {
    let check = citation_check(
        "corpus/brewery.md",
        0,
        true,
        resolved_outcome(
            SectionCheck::NotChecked,
            Some(QuoteCheck {
                declared: "存在しない引用".to_string(),
                matched: false,
            }),
        ),
    );
    let value = serde_json::to_value(&check).unwrap();
    assert_no_body_text(&value, "TOP-SECRET-PASSAGE-BODY-TEXT", "$");
    // The served text itself is never on `CitationCheck` at all — this
    // is a schema check, not a string-scan for a placeholder that was
    // never constructed with any served text to begin with.
    assert!(value.get("text").is_none(), "{value}");
}

// ========================= missed[] capping (ADR §11) =========================

#[test]
fn build_missed_caps_at_three_and_counts_the_rest_as_truncated() {
    let mut case = base_case("c1");
    case.expected_sources = vec![
        expected_source("a.md", &[], 1),
        expected_source("b.md", &[], 1),
    ];
    case.expected_concepts = vec!["x".to_string(), "y".to_string()];
    let hits: Vec<HitLocator> = Vec::new();
    let (missed, truncated) = build_missed(&case, Some(&hits), &[], &[], None, &[]);
    assert_eq!(missed.len(), 3, "{missed:?}");
    assert_eq!(truncated, 1, "4 misses total, 3 kept, 1 dropped");
}

#[test]
fn build_missed_distinguishes_a_queried_zero_from_a_never_run_query() {
    let case = base_case("c1");
    let structural = StructuralBlock {
        cues: Vec::new(),
        associations: vec![
            association_probe(Some(QueryProbe::Queried {
                total: 0,
                matches: 0,
                attributions: Vec::new(),
                latency_ms: 1,
            })),
            association_probe(None),
        ],
    };
    let (missed, _) = build_missed(&case, None, &[], &[], Some(&structural), &[]);
    assert_eq!(missed.len(), 2, "{missed:?}");
    assert!(
        missed[0].contains("query returned no association"),
        "{missed:?}"
    );
    assert!(
        missed[1].contains("query never ran (a position did not pin)"),
        "{missed:?}"
    );
    assert_ne!(
        missed[0], missed[1],
        "the two failure modes must not collapse into the same message"
    );
}

#[test]
fn build_missed_is_silent_about_sources_when_the_passage_lane_failed() {
    let mut case = base_case("c1");
    case.expected_sources = vec![expected_source("a.md", &[], 1)];
    let (missed, truncated) = build_missed(&case, None, &[], &[], None, &[]);
    assert!(missed.is_empty(), "{missed:?}");
    assert_eq!(truncated, 0);
}

// ========================= score_case: lane cross-tab denominator (ADR §7) =========================

#[test]
fn score_case_recall_and_lane_cross_are_none_when_the_passage_lane_failed() {
    let mut case = base_case("c1");
    case.expected_sources = vec![expected_source("a.md", &[], 1)];
    case.expected_concepts = vec!["x".to_string()];
    let passage = PassageOutcome::Failed {
        message: "boom".to_string(),
        latency_ms: 1,
    };
    let structural = StructuralBlock {
        cues: vec![cue_resolution("concept", &["x"])],
        associations: Vec::new(),
    };
    let scores = score_case(&case, &passage, Some(&structural), &[]);
    assert!(scores.recall.is_none());
    assert!(scores.lane_cross.is_none());
    assert!(
        scores.coverage.is_some(),
        "structural coverage does not depend on the passage lane"
    );
}

#[test]
fn score_case_reports_lane_cross_only_when_both_expectation_kinds_are_declared() {
    let mut case = base_case("c1");
    case.expected_sources = vec![expected_source("a.md", &[], 1)];
    case.expected_concepts = vec!["x".to_string()];
    let passage = PassageOutcome::Searched {
        plan: None,
        hits: vec![hit("a.md", 0)],
        latency_ms: 1,
    };
    let structural = StructuralBlock {
        cues: vec![cue_resolution("concept", &["x"])],
        associations: Vec::new(),
    };
    let scores = score_case(&case, &passage, Some(&structural), &[]);
    let cross = scores
        .lane_cross
        .expect("both a structural and a source expectation were declared");
    assert!(cross.structural_hit);
    assert!(cross.passage_hit);
}

#[test]
fn score_case_lane_cross_is_none_with_only_a_source_expectation() {
    let mut case = base_case("c1");
    case.expected_sources = vec![expected_source("a.md", &[], 1)];
    let passage = PassageOutcome::Searched {
        plan: None,
        hits: vec![hit("a.md", 0)],
        latency_ms: 1,
    };
    let scores = score_case(&case, &passage, None, &[]);
    assert!(scores.lane_cross.is_none());
    assert!(scores.recall.is_some());
}

// ========================= No fused cross-lane scale (#215's own requirement) =========================

/// Words that would imply a case's/run's outcome folds graph, BM25, and
/// vector scores into one invented scale — banned from every metric key
/// and its own `MetricDef`. `rank`/`score` are not banned: a hit's own
/// position and a `HitLocator`'s own per-lane evidence are ordinary IR
/// concepts, never a cross-lane composite.
const FUSED_SCALE_WORDS: [&str; 6] = [
    "fused",
    "combined",
    "unified",
    "composite",
    "blended",
    "normalized_score",
];

fn assert_no_fused_scale_words(value: &Value, path: &str) {
    match value {
        Value::Object(map) => {
            for (key, v) in map {
                for banned in FUSED_SCALE_WORDS {
                    assert!(
                        !key.to_lowercase().contains(banned),
                        "fused-scale word '{banned}' found in key '{key}' at {path}"
                    );
                }
                assert_no_fused_scale_words(v, &format!("{path}.{key}"));
            }
        }
        Value::String(text) => {
            for banned in FUSED_SCALE_WORDS {
                assert!(
                    !text.to_lowercase().contains(banned),
                    "fused-scale word '{banned}' found in a string at {path}: {text}"
                );
            }
        }
        Value::Array(items) => {
            for (i, v) in items.iter().enumerate() {
                assert_no_fused_scale_words(v, &format!("{path}[{i}]"));
            }
        }
        _ => {}
    }
}

#[test]
fn no_fused_scale_word_appears_in_a_metric_key_or_definition() {
    let metrics = build_metrics(&[], false);
    let definitions = build_definitions();
    assert_no_fused_scale_words(&serde_json::to_value(&metrics).unwrap(), "$.metrics");
    assert_no_fused_scale_words(
        &serde_json::to_value(&definitions).unwrap(),
        "$.definitions",
    );
}

// ========================= matching block accuracy =========================

#[test]
fn matching_block_does_not_claim_a_normalize_entry_comparison_for_associations() {
    // association_coverage never runs a client-side string comparison
    // — coverage is decided by the server's own /resolve+/query round
    // trip (ADR 0004 §7 step 2) — so `normalized` must not list
    // expected_associations alongside the two fields that really are
    // folded through normalize_entry.
    let matching = MatchingBlock::default();
    assert_eq!(
        matching.normalized,
        &["expected_concepts", "expected_labels"]
    );
}

// ========================= Endpoint role (ADR §11) =========================

#[test]
fn evaluate_only_touches_read_role_endpoints() {
    for (method, route) in [
        (Method::POST, "/contexts/{name}/sources/search"),
        // #308 (ADR 0006 §5.4): --assembly's own passage-lane
        // substitute, `POST /contexts/{name}/evidence`.
        (Method::POST, "/contexts/{name}/evidence"),
        (Method::POST, "/contexts/{name}/resolve"),
        (Method::POST, "/contexts/{name}/resolve_label"),
        (Method::POST, "/contexts/{name}/query"),
        (Method::POST, "/contexts/{name}/citations"),
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
