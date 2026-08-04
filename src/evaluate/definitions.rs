//! The metric catalog: `build_definitions` assembles the `MetricDef`
//! entry for every key `super::build_metrics` inserts into a run's
//! `MetricsMap`, read back by `super::run_evaluate` for the artifact's
//! own `definitions` block and by `--thresholds` loading to validate a
//! bound names a real metric.

use super::*;

pub(crate) fn build_definitions() -> BTreeMap<String, MetricDef> {
    let mut d = BTreeMap::new();
    d.insert(
        "latency.passage_ms".to_string(),
        def(
            "ms",
            "distribution",
            &["case"],
            "Wall-clock round trip of the passage lane's own \
             POST /contexts/{name}/sources/search call.",
            "POST /contexts/{name}/sources/search",
            None,
        ),
    );
    d.insert(
        "latency.resolve_ms".to_string(),
        def(
            "ms",
            "distribution",
            &["case"],
            "Wall-clock round trip of every /resolve and /resolve_label call \
             the structural lane made — coverage cues and association \
             positions alike.",
            "POST /contexts/{name}/resolve, POST /contexts/{name}/resolve_label",
            None,
        ),
    );
    d.insert(
        "latency.query_ms".to_string(),
        def(
            "ms",
            "distribution",
            &["case"],
            "Wall-clock round trip of each expected_associations[] entry's \
             POST /contexts/{name}/query call.",
            "POST /contexts/{name}/query",
            Some(
                "empty when no case declares expected_associations, or none \
                 of them resolved every position to exactly one name",
            ),
        ),
    );
    d.insert(
        "passage.failure_rate".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of cases whose passage lane call did not complete \
             (transport error or an unparseable response).",
            "POST /contexts/{name}/sources/search",
            None,
        ),
    );
    d.insert(
        "structural.case_rate".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of cases that declared expected_concepts, \
             expected_labels, or expected_associations and so ran the \
             structural lane.",
            "eval.jsonl case fields",
            None,
        ),
    );
    d.insert(
        "recall.recall_at_k".to_string(),
        def(
            "ratio",
            "distribution",
            &["case"],
            "Fraction of a case's expected_sources (relevance >= 1) found \
             among the passage lane's hits, up to the case's own limit.",
            "eval.jsonl expected_sources, POST /contexts/{name}/sources/search",
            Some(
                "empty when no case declares a relevance >= 1 expected_sources \
                 entry, or the passage lane failed outright",
            ),
        ),
    );
    d.insert(
        "recall.mrr".to_string(),
        def(
            "ratio",
            "distribution",
            &["case"],
            "1 / rank of the first passage hit that satisfies any of a \
             case's expected_sources entries, 0 if none does.",
            "eval.jsonl expected_sources, POST /contexts/{name}/sources/search",
            Some(
                "empty when no case declares a relevance >= 1 expected_sources \
                 entry, or the passage lane failed outright",
            ),
        ),
    );
    d.insert(
        "recall.ndcg".to_string(),
        def(
            "ratio",
            "distribution",
            &["case"],
            "Graded-relevance nDCG over expected_sources: each expectation \
             contributes its own relevance (0..=3) once, at the rank of the \
             first hit that satisfies it; IDCG orders the case's own \
             expected relevances descending.",
            "eval.jsonl expected_sources.relevance, POST /contexts/{name}/sources/search",
            Some(
                "empty when no case declares a relevance >= 1 expected_sources \
                 entry, or the passage lane failed outright",
            ),
        ),
    );
    d.insert(
        "coverage.concepts".to_string(),
        def(
            "ratio",
            "distribution",
            &["case"],
            "Fraction of a case's expected_concepts found, after \
             normalize_entry folding, among the structural lane's \
             concept-cue resolved_names[].",
            "eval.jsonl expected_concepts, POST /contexts/{name}/resolve",
            Some("empty when no case declares expected_concepts"),
        ),
    );
    d.insert(
        "coverage.labels".to_string(),
        def(
            "ratio",
            "distribution",
            &["case"],
            "Fraction of a case's expected_labels found, after \
             normalize_entry folding, among the structural lane's \
             label-cue resolved_names[].",
            "eval.jsonl expected_labels, POST /contexts/{name}/resolve_label",
            Some("empty when no case declares expected_labels"),
        ),
    );
    d.insert(
        "coverage.associations".to_string(),
        def(
            "ratio",
            "distribution",
            &["case"],
            "Fraction of a case's expected_associations whose /query call \
             ran (all three positions pinned) and returned total >= 1.",
            "eval.jsonl expected_associations, POST /contexts/{name}/query",
            Some(
                "empty when no case declares expected_associations; a \
                 not_found/ambiguous position never runs query and so \
                 counts as uncovered, never guessed at",
            ),
        ),
    );
    d.insert(
        "citations.recall".to_string(),
        def(
            "ratio",
            "distribution",
            &["case"],
            "Fraction of a case's expected_citations whose (source, \
             paragraph) appeared among that case's served results — \
             passage hits up to limit, plus the structural lane's \
             AttributionOut locators when it ran (ADR 0004 §8). Never \
             merged with citations.locator_validity.",
            "eval.jsonl expected_citations, POST /contexts/{name}/sources/search, \
             POST /contexts/{name}/query",
            Some("empty when no case declares expected_citations"),
        ),
    );
    d.insert(
        "citations.locator_validity".to_string(),
        def(
            "ratio",
            "distribution",
            &["case"],
            "Fraction of a case's expected_citations whose \
             POST /contexts/{name}/citations call resolved with a \
             matching section (when declared) and quote (when declared) \
             — computed even for a case whose passage lane missed \
             outright (ADR 0004 §8). Never merged with citations.recall.",
            "POST /contexts/{name}/citations",
            Some("empty when no case declares expected_citations"),
        ),
    );
    d.insert(
        "citations.resolved".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of every POST /contexts/{name}/citations call made \
             across all cases that resolved (neither no_source nor \
             no_paragraph).",
            "POST /contexts/{name}/citations",
            None,
        ),
    );
    d.insert(
        "citations.no_source".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of every citation call that failed with ErrorCode \
             no_source — the expected_citations entry names a source \
             this context does not carry.",
            "POST /contexts/{name}/citations",
            None,
        ),
    );
    d.insert(
        "citations.no_paragraph".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of every citation call that failed with ErrorCode \
             no_paragraph — the expected_citations entry names a \
             paragraph index out of range for its source.",
            "POST /contexts/{name}/citations",
            None,
        ),
    );
    d.insert(
        "citations.other".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of every citation call that neither resolved nor \
             failed with ErrorCode no_source/no_paragraph — a transport \
             failure, an unparsable response, or any other ErrorCode. \
             resolved + no_source + no_paragraph + other always sums \
             to 1.0.",
            "POST /contexts/{name}/citations",
            None,
        ),
    );
    d.insert(
        "citations.section_match".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Of the resolved citation calls whose expected_citations \
             entry declared a section key (an explicit null included), \
             the share whose declared value matched the server's own \
             Citation.section.",
            "eval.jsonl expected_citations.section, POST /contexts/{name}/citations",
            Some("n is 0 when no expected_citations entry declares section"),
        ),
    );
    d.insert(
        "citations.quote_match".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Of the resolved citation calls whose expected_citations \
             entry declared a quote, the share whose declared quote was \
             a normalize_entry-folded substring of the returned text. A \
             quote spanning a paragraph boundary can never match here, \
             since Citation.text is exactly one paragraph.",
            "eval.jsonl expected_citations.quote, POST /contexts/{name}/citations",
            Some("n is 0 when no expected_citations entry declares quote"),
        ),
    );
    d.insert(
        "latency.citation_ms".to_string(),
        def(
            "ms",
            "distribution",
            &["case"],
            "Wall-clock round trip of each expected_citations[] entry's \
             own POST /contexts/{name}/citations call.",
            "POST /contexts/{name}/citations",
            Some("empty when no case declares expected_citations"),
        ),
    );
    d.insert(
        "lanes.structural_hit".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of cases declaring BOTH a structural expectation and a \
             relevance >= 1 source expectation whose structural coverage \
             matched at least one expectation (ADR 0004 §7).",
            "derived from coverage.* and recall.recall_at_k",
            Some(
                "denominator excludes cases with only a structural or only a \
                 source expectation, and any case whose passage lane failed",
            ),
        ),
    );
    d.insert(
        "lanes.passage_hit".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of cases declaring BOTH a structural expectation and a \
             relevance >= 1 source expectation whose passage lane matched \
             at least one expected_sources entry (ADR 0004 §7).",
            "derived from coverage.* and recall.recall_at_k",
            Some(
                "denominator excludes cases with only a structural or only a \
                 source expectation, and any case whose passage lane failed",
            ),
        ),
    );
    d.insert(
        "lanes.both".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of the same denominator as lanes.structural_hit/ \
             lanes.passage_hit where both lanes hit (ADR 0004 §7).",
            "derived from coverage.* and recall.recall_at_k",
            None,
        ),
    );
    d.insert(
        "lanes.neither".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of the same denominator as lanes.structural_hit/ \
             lanes.passage_hit where neither lane hit (ADR 0004 §7). A \
             case counts here purely by rank/label outcome, never by a \
             graph/BM25/vector score value.",
            "derived from coverage.* and recall.recall_at_k",
            None,
        ),
    );
    // #308 (ADR 0006 §14): equal-budget comparison metrics —
    // `diversity.sources` is the one metric #216/ADR 0006 name
    // explicitly ("source diversity at equal evidence budget") that
    // ADR 0004 does not already define; the rest let a `--thresholds`
    // file and `evaluate compare` see budget consumption and reranker
    // degrade rate the same way they already see recall/citations.
    d.insert(
        "diversity.sources".to_string(),
        def(
            "source",
            "distribution",
            &["case"],
            "Count of distinct source locators among a case's admitted \
             evidence — read from the same hits[] recall/citation \
             scoring itself uses: baseline's (possibly budget-truncated) \
             passage hits, or assembly's admitted items' own locators \
             union their citation_refs (an association's corroborating \
             attributions; a passage/community item's own locator, \
             since its citation_refs is always empty by design).",
            "POST /contexts/{name}/sources/search, POST /contexts/{name}/evidence",
            Some("empty when the passage/evidence lane failed outright"),
        ),
    );
    d.insert(
        "latency.evidence_ms".to_string(),
        def(
            "ms",
            "distribution",
            &["case"],
            "Wall-clock round trip of an --assembly run's own \
             POST /contexts/{name}/evidence call.",
            "POST /contexts/{name}/evidence",
            Some("empty in baseline mode"),
        ),
    );
    d.insert(
        "budget.items_used".to_string(),
        def(
            "item",
            "distribution",
            &["case"],
            "A case's own items_used against its resolved budget ceiling \
             — baseline's client-side truncation accounting or \
             assembly's server-returned BudgetUsage, computed with the \
             identical ADR 0006 §8 formula either way.",
            "crate::api::evidence::budget",
            Some("empty when no --max-items/--max-bytes/--max-tokens flag was given"),
        ),
    );
    d.insert(
        "budget.bytes_used".to_string(),
        def(
            "byte",
            "distribution",
            &["case"],
            "A case's own bytes_used against its resolved budget ceiling \
             (ADR 0006 §8: the items array's compact JSON length plus \
             the citations array's, excluding each item's own bytes/ \
             estimated_tokens fields).",
            "crate::api::evidence::budget",
            Some("empty when no --max-items/--max-bytes/--max-tokens flag was given"),
        ),
    );
    d.insert(
        "budget.tokens_used".to_string(),
        def(
            "token",
            "distribution",
            &["case"],
            "A case's own tokens_used against its resolved budget \
             ceiling — ADR 0006 §8's fixed estimator (0.25 tokens per \
             Basic Latin scalar, 1.0 otherwise), never a real tokenizer \
             count.",
            "crate::api::evidence::budget",
            Some("empty when no --max-items/--max-bytes/--max-tokens flag was given"),
        ),
    );
    d.insert(
        "budget.omitted_rate".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of every candidate a budget-truncated case considered \
             (admitted plus omitted, either mode) that a budget ceiling \
             dropped — the equal-budget \"how much got left out\" \
             counterpart to items_used/bytes_used/tokens_used.",
            "crate::api::evidence::budget, POST /contexts/{name}/evidence",
            Some("empty when no --max-items/--max-bytes/--max-tokens flag was given"),
        ),
    );
    d.insert(
        "rerank.ran".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of --rerank cases whose configured reranker actually \
             reordered the pool (plan.reranker.ran); the complement is \
             the degrade rate — no provider configured, a model \
             mismatch, or any other ADR 0006 §12 fallback reason.",
            "POST /contexts/{name}/evidence plan.reranker",
            Some("empty when --rerank was not given"),
        ),
    );
    d
}
