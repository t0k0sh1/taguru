//! `benchmark::identity` (issue #258, ADR 0003 §9.4): the association
//! same-ness core that `taguru benchmark compare`'s stability metrics
//! use, and that #259's model-pair `differences.jsonl` will reuse for
//! its own `association_shared`/`association_single_side` matching.
//!
//! [`Matching`] is the parameter set ADR 0003 §9.4 requires every
//! `differences.jsonl` header to carry (`matching` block) so a
//! difference set stays re-derivable. Its field set and serialized
//! spelling are exactly the ADR's header example — additive changes
//! (e.g. a future `kana_fold` flag) are fine, but nothing here should
//! rename or drop from that set without updating the ADR.
//!
//! **Deliberately not `crate::context::normalize_entry`**: that
//! function folds katakana to hiragana in addition to lowercasing and
//! NFKC, because two spellings of one *stored graph entry* should
//! resolve to the same node. Here the question is different — did two
//! *model runs* produce the same triple — and a model's choice of
//! katakana vs. hiragana is exactly the kind of run-to-run variation
//! §258 exists to measure, not something to fold away before
//! measuring it. [`normalize_term`] applies only case folding and
//! NFKC, gated by [`Matching`]'s own flags.
//!
//! Alias resolution here is "batch-local" (ADR 0003 §9.4's
//! `alias_expansion` value): only the alias/canonical declarations
//! inside the one batch a triple came from are consulted, mirroring
//! the conflict-resolution and chain-following semantics
//! `crate::context::alias::add_alias`/`check_aliases` already use for
//! the live graph — first-seen alias spelling wins a conflict, and a
//! chain (`a -> b -> c`) resolves to its final canonical regardless of
//! declaration order.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

// ============================== Matching parameters ==============================

/// ADR 0003 §9.4's `matching.unicode_normalization` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) enum UnicodeNormalization {
    #[serde(rename = "NFKC")]
    Nfkc,
    #[serde(rename = "none")]
    None,
}

/// ADR 0003 §9.4's `matching.alias_expansion` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) enum AliasExpansion {
    #[serde(rename = "batch-local")]
    BatchLocal,
    #[serde(rename = "none")]
    None,
}

/// The parameter set a same-ness judgment was made under — recorded
/// verbatim in `measurements.json`'s own `matching` block (and, by
/// #259, in every `differences.jsonl` header) so the judgment stays
/// re-derivable without re-reading this module's source.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct Matching {
    pub(crate) module: &'static str,
    pub(crate) case_fold: bool,
    pub(crate) unicode_normalization: UnicodeNormalization,
    pub(crate) alias_expansion: AliasExpansion,
    pub(crate) weight_tolerance: f64,
}

impl Default for Matching {
    fn default() -> Self {
        Self {
            module: "benchmark::identity",
            case_fold: true,
            unicode_normalization: UnicodeNormalization::Nfkc,
            alias_expansion: AliasExpansion::BatchLocal,
            weight_tolerance: 0.0,
        }
    }
}

/// Trim, then fold under `matching`'s own flags. Idempotent for every
/// flag combination `Matching` can express:
///
/// - neither flag set: trimming alone is idempotent on already-trimmed
///   text.
/// - `case_fold` only: Unicode case folding is idempotent on its own
///   (`char::to_lowercase` never produces text that lowercases
///   differently the second time).
/// - `unicode_normalization: NFKC` only: NFKC is idempotent by
///   definition — normalizing already-normalized text is a no-op.
/// - both: lowercasing must run both before AND after `nfkc()`, for the
///   same reason `crate::context`'s `normalize` does — NFKC's
///   decomposition can introduce fresh uppercase text (its own doc
///   comment at src/context.rs:900 walks the counterexamples), and only
///   a trailing lowercase re-folds that back to a fixed point.
pub(crate) fn normalize_term(matching: &Matching, raw: &str) -> String {
    let trimmed = raw.trim();
    if !matching.case_fold && matching.unicode_normalization == UnicodeNormalization::None {
        return trimmed.to_string();
    }
    let mut folded: String = if matching.case_fold {
        trimmed.chars().flat_map(char::to_lowercase).collect()
    } else {
        trimmed.to_string()
    };
    if matching.unicode_normalization == UnicodeNormalization::Nfkc {
        use unicode_normalization::UnicodeNormalization as _;
        folded = folded.nfkc().collect();
        if matching.case_fold {
            folded = folded.chars().flat_map(char::to_lowercase).collect();
        }
    }
    folded
}

// ============================== Alias resolution (batch-local) ==============================

/// The two alias namespaces `render_batch` (src/extract.rs) ever
/// writes — kept independent, since a concept spelling and a label
/// spelling can coincide without meaning the same thing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum AliasKind {
    Concept,
    Label,
}

/// Parses the batch format's `kind` string. `None` for anything other
/// than `"concept"`/`"label"` — the same two shapes
/// `crate::benchmark::compare::analyze_batch`'s orphan check already
/// recognizes and lets through unclassified.
pub(crate) fn parse_alias_kind(kind: &str) -> Option<AliasKind> {
    match kind {
        "concept" => Some(AliasKind::Concept),
        "label" => Some(AliasKind::Label),
        _ => None,
    }
}

/// One `{"alias", "canonical", "kind"}` batch line, pre-normalization.
#[derive(Debug, Clone)]
pub(crate) struct AliasLine {
    pub(crate) alias: String,
    pub(crate) canonical: String,
    pub(crate) kind: AliasKind,
}

/// A batch-local alias resolver: normalized alias spelling ->
/// normalized, chain-resolved canonical. Unregistered spellings
/// resolve to themselves (`resolve` returns its input unchanged).
#[derive(Debug, Default)]
pub(crate) struct AliasMap {
    concepts: BTreeMap<String, String>,
    labels: BTreeMap<String, String>,
}

impl AliasMap {
    /// Builds the resolver from one batch's alias lines, in the order
    /// they appear in the file. `alias_expansion: none` yields an empty
    /// resolver (identity resolution for everything).
    ///
    /// Conflicts (the same normalized alias declared with two
    /// different canonicals) keep the first declaration, matching
    /// `crate::context::alias::add_alias`'s conflict-is-rejected
    /// posture — a later, conflicting line is simply not seen. Chains
    /// (`a -> b`, `b -> c`) resolve to their final canonical regardless
    /// of which line came first in the file, since resolution walks the
    /// completed map rather than the file's line order. A cycle
    /// (`a -> b`, `b -> a`) has no canonical to walk to; every member
    /// resolves to the cycle's lexicographically smallest spelling,
    /// deterministically and without erroring — this module counts and
    /// compares, it does not validate a harness's output.
    pub(crate) fn from_batch(matching: &Matching, lines: &[AliasLine]) -> Self {
        if matching.alias_expansion == AliasExpansion::None {
            return Self::default();
        }
        let mut raw_concepts: BTreeMap<String, String> = BTreeMap::new();
        let mut raw_labels: BTreeMap<String, String> = BTreeMap::new();
        for line in lines {
            let alias = normalize_term(matching, &line.alias);
            let canonical = normalize_term(matching, &line.canonical);
            if alias == canonical {
                continue; // a self-mapping alias is a no-op, not a fact worth recording
            }
            let raw = match line.kind {
                AliasKind::Concept => &mut raw_concepts,
                AliasKind::Label => &mut raw_labels,
            };
            raw.entry(alias).or_insert(canonical);
        }
        Self {
            concepts: resolve_chains(&raw_concepts),
            labels: resolve_chains(&raw_labels),
        }
    }

    /// Resolves one already-normalized spelling to its canonical form,
    /// or returns it unchanged if this map has no entry for it.
    pub(crate) fn resolve(&self, kind: AliasKind, normalized: &str) -> String {
        let map = match kind {
            AliasKind::Concept => &self.concepts,
            AliasKind::Label => &self.labels,
        };
        map.get(normalized)
            .cloned()
            .unwrap_or_else(|| normalized.to_string())
    }
}

/// Follows every alias -> canonical chain in `raw` to its terminal,
/// short-circuiting through spellings already resolved by an earlier
/// starting point, and folding any cycle to its lexicographically
/// smallest member. `raw`'s own iteration order (a `BTreeMap`, so
/// always sorted by spelling) drives which chains get walked first,
/// but the terminal each chain reaches is independent of that order —
/// only the *first-seen-wins* conflict choice above depends on file
/// order, never this walk.
fn resolve_chains(raw: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    let mut resolved: BTreeMap<String, String> = BTreeMap::new();
    for start in raw.keys() {
        if resolved.contains_key(start) {
            continue;
        }
        let mut path: Vec<String> = Vec::new();
        let mut index_of: BTreeMap<String, usize> = BTreeMap::new();
        let mut current = start.clone();
        let terminal = loop {
            if let Some(known) = resolved.get(&current) {
                break known.clone();
            }
            if let Some(&pos) = index_of.get(&current) {
                let mut cycle: Vec<String> = path[pos..].to_vec();
                cycle.sort();
                break cycle.into_iter().next().expect("cycle is non-empty");
            }
            index_of.insert(current.clone(), path.len());
            path.push(current.clone());
            match raw.get(&current) {
                Some(next) => current = next.clone(),
                None => break current.clone(),
            }
        };
        for member in path {
            resolved.entry(member).or_insert_with(|| terminal.clone());
        }
    }
    resolved
}

// ============================== Keying ==============================

/// A same-ness key, scoped to one document (ADR 0003 §9.4's
/// `differences.jsonl` records are per-document too, via `locator`):
/// normalized and alias-resolved subject/label/object. Two batch lines
/// that fold to the same key are the same association for every metric
/// in this module, however differently the model spelled them.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub(crate) struct AssocKey {
    pub(crate) document_id: String,
    pub(crate) subject: String,
    pub(crate) label: String,
    pub(crate) object: String,
}

/// One `{"subject","label","object","weight"[,"paragraph"]}` batch
/// line, pre-normalization.
#[derive(Debug, Clone)]
pub(crate) struct RawAssociation {
    pub(crate) subject: String,
    pub(crate) label: String,
    pub(crate) object: String,
    pub(crate) weight: f64,
    pub(crate) paragraph: Option<u64>,
}

/// One document's worth of one batch's raw material — everything
/// [`keyed_associations`] and alias-variation accounting need, kept
/// alongside `crate::benchmark::compare::BatchStats`' existing
/// vocabulary-shape counts rather than duplicating a batch read.
#[derive(Debug, Clone, Default)]
pub(crate) struct BatchRows {
    pub(crate) associations: Vec<RawAssociation>,
    pub(crate) aliases: Vec<AliasLine>,
}

/// What was actually observed at one key within one run: every raw
/// weight that folded into this key (almost always one, unless
/// normalization merged distinct spellings), and the distinct
/// paragraph locators attached to it.
#[derive(Debug, Clone, Default)]
pub(crate) struct KeyObservation {
    pub(crate) weights: Vec<f64>,
    pub(crate) paragraphs: BTreeSet<Option<u64>>,
}

/// Keys one document's raw associations under `matching`'s rules,
/// resolving subject/object through the concept namespace and label
/// through the label namespace.
pub(crate) fn keyed_associations(
    matching: &Matching,
    document_id: &str,
    associations: &[RawAssociation],
    aliases: &AliasMap,
) -> BTreeMap<AssocKey, KeyObservation> {
    let mut out: BTreeMap<AssocKey, KeyObservation> = BTreeMap::new();
    for assoc in associations {
        let subject = aliases.resolve(
            AliasKind::Concept,
            &normalize_term(matching, &assoc.subject),
        );
        let label = aliases.resolve(AliasKind::Label, &normalize_term(matching, &assoc.label));
        let object = aliases.resolve(AliasKind::Concept, &normalize_term(matching, &assoc.object));
        let key = AssocKey {
            document_id: document_id.to_string(),
            subject,
            label,
            object,
        };
        let observation = out.entry(key).or_default();
        observation.weights.push(assoc.weight);
        observation.paragraphs.insert(assoc.paragraph);
    }
    out
}

// ============================== N-run presence ==============================

/// One key's observations across every run it appeared in — the
/// material ADR 0003 §9.4's `sides.{a,b}.runs`/`n_present` are built
/// from (#259 reuses this directly for its own per-model presence).
#[derive(Debug, Default)]
pub(crate) struct KeyPresence {
    by_run: BTreeMap<usize, KeyObservation>,
}

impl KeyPresence {
    pub(crate) fn n_present(&self) -> usize {
        self.by_run.len()
    }

    #[allow(dead_code)] // ADR 0003 §9.4's sides.{a,b}.runs — consumed once #259 writes differences.jsonl
    pub(crate) fn runs(&self) -> impl Iterator<Item = usize> + '_ {
        self.by_run.keys().copied()
    }

    pub(crate) fn observations(&self) -> impl Iterator<Item = &KeyObservation> {
        self.by_run.values()
    }
}

/// Every key's [`KeyPresence`] across every run inserted so far —
/// typically built one run (one completed document's batch) at a time
/// via [`Self::insert_run`].
#[derive(Debug, Default)]
pub(crate) struct PresenceTable {
    entries: BTreeMap<AssocKey, KeyPresence>,
}

impl PresenceTable {
    pub(crate) fn insert_run(
        &mut self,
        run_index: usize,
        keyed: BTreeMap<AssocKey, KeyObservation>,
    ) {
        for (key, observation) in keyed {
            self.entries
                .entry(key)
                .or_default()
                .by_run
                .insert(run_index, observation);
        }
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&AssocKey, &KeyPresence)> {
        self.entries.iter()
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

// ============================== Variation & overlap ==============================

/// `> 0` / `== 0` / `< 0`, independent categories — the same three-way
/// split `crate::benchmark::compare::analyze_batch` already uses for
/// `weight_positive`/`weight_negative` (a weight of exactly 0 counts
/// toward neither polarity).
fn weight_sign(weight: f64) -> i8 {
    if weight > 0.0 {
        1
    } else if weight < 0.0 {
        -1
    } else {
        0
    }
}

/// Whether this key's weight sign disagreed across its observations —
/// within a run (normalization merged distinct spellings) or across
/// runs.
pub(crate) fn polarity_varies(presence: &KeyPresence) -> bool {
    let mut signs: BTreeSet<i8> = BTreeSet::new();
    for observation in presence.observations() {
        signs.extend(observation.weights.iter().copied().map(weight_sign));
    }
    signs.len() > 1
}

/// Whether this key's observed weights spread by more than
/// `matching.weight_tolerance` — an exact-equal spread (`max - min ==
/// tolerance`) does not count as variation, only a strict excess does.
pub(crate) fn weight_varies(presence: &KeyPresence, tolerance: f64) -> bool {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut any = false;
    for observation in presence.observations() {
        for &weight in &observation.weights {
            any = true;
            min = min.min(weight);
            max = max.max(weight);
        }
    }
    any && (max - min) > tolerance
}

/// Whether this key's paragraph-locator set disagreed across its
/// observations.
pub(crate) fn attribution_varies(presence: &KeyPresence) -> bool {
    let mut sets = presence.observations().map(|o| &o.paragraphs);
    let Some(first) = sets.next() else {
        return false;
    };
    sets.any(|set| set != first)
}

/// `|A ∩ B| / |A ∪ B|`. `None` when both sides are empty — no ratio is
/// synthesized for an empty-over-empty comparison, matching this
/// module's neighbors' "measured, zero samples is not the same as a
/// value" posture (ADR 0003 §9.3) rather than reporting a vacuous
/// `1.0`.
pub(crate) fn jaccard(a: &BTreeSet<AssocKey>, b: &BTreeSet<AssocKey>) -> Option<f64> {
    if a.is_empty() && b.is_empty() {
        return None;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    Some(intersection as f64 / union as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn key(document_id: &str, subject: &str, label: &str, object: &str) -> AssocKey {
        AssocKey {
            document_id: document_id.to_string(),
            subject: subject.to_string(),
            label: label.to_string(),
            object: object.to_string(),
        }
    }

    // ---------- normalize_term ----------

    #[test]
    fn matching_default_matches_the_adr_header_example() {
        let value = serde_json::to_value(Matching::default()).unwrap();
        assert_eq!(
            value,
            serde_json::json!({
                "module": "benchmark::identity",
                "case_fold": true,
                "unicode_normalization": "NFKC",
                "alias_expansion": "batch-local",
                "weight_tolerance": 0.0,
            }),
        );
    }

    #[test]
    fn normalize_term_trims_and_case_folds() {
        let m = Matching::default();
        assert_eq!(normalize_term(&m, "  Apple  "), "apple");
    }

    #[test]
    fn normalize_term_folds_fullwidth_forms_via_nfkc() {
        let m = Matching::default();
        assert_eq!(normalize_term(&m, "Ａｐｐｌｅ"), "apple");
    }

    #[test]
    fn normalize_term_does_not_fold_katakana_to_hiragana() {
        // Unlike crate::context::normalize_entry — a model's choice of
        // script is exactly the run-to-run variation this module
        // exists to observe, not something to fold away.
        let m = Matching::default();
        assert_eq!(normalize_term(&m, "リンゴ"), "リンゴ");
        assert_ne!(normalize_term(&m, "リンゴ"), "りんご");
    }

    #[test]
    fn normalize_term_handles_the_context_rs_idempotence_counterexamples() {
        // src/context.rs:907-918: neither lowercase-then-NFKC nor
        // NFKC-then-lowercase alone is idempotent; both orders combined
        // (as normalize_term does) must be.
        let m = Matching::default();
        for raw in ["J\u{030C}", "\u{1F110}", "Ａｐｐｌｅ", "  trim me  "] {
            let once = normalize_term(&m, raw);
            let twice = normalize_term(&m, &once);
            assert_eq!(once, twice, "not idempotent for {raw:?}");
        }
    }

    #[test]
    fn normalize_term_with_every_flag_off_only_trims() {
        let m = Matching {
            case_fold: false,
            unicode_normalization: UnicodeNormalization::None,
            ..Matching::default()
        };
        assert_eq!(normalize_term(&m, "  Apple  "), "Apple");
    }

    proptest! {
        #[test]
        fn normalize_term_is_idempotent_for_every_flag_combination(
            raw in ".*",
            case_fold in any::<bool>(),
            nfkc in any::<bool>(),
        ) {
            let m = Matching {
                case_fold,
                unicode_normalization: if nfkc { UnicodeNormalization::Nfkc } else { UnicodeNormalization::None },
                ..Matching::default()
            };
            let once = normalize_term(&m, &raw);
            let twice = normalize_term(&m, &once);
            prop_assert_eq!(once, twice);
        }
    }

    // ---------- AliasMap ----------

    #[test]
    fn alias_map_resolves_a_chain_regardless_of_declaration_order() {
        let m = Matching::default();
        // Declared out of chain order (b->c before a->b): resolution
        // walks the completed map, not the file's line order.
        let lines = vec![
            AliasLine {
                alias: "b".into(),
                canonical: "c".into(),
                kind: AliasKind::Concept,
            },
            AliasLine {
                alias: "a".into(),
                canonical: "b".into(),
                kind: AliasKind::Concept,
            },
        ];
        let aliases = AliasMap::from_batch(&m, &lines);
        assert_eq!(aliases.resolve(AliasKind::Concept, "a"), "c");
        assert_eq!(aliases.resolve(AliasKind::Concept, "b"), "c");
    }

    #[test]
    fn alias_map_resolves_a_cycle_to_its_lexicographically_smallest_member() {
        let m = Matching::default();
        let lines = vec![
            AliasLine {
                alias: "zeta".into(),
                canonical: "alpha".into(),
                kind: AliasKind::Concept,
            },
            AliasLine {
                alias: "alpha".into(),
                canonical: "zeta".into(),
                kind: AliasKind::Concept,
            },
        ];
        let aliases = AliasMap::from_batch(&m, &lines);
        assert_eq!(aliases.resolve(AliasKind::Concept, "alpha"), "alpha");
        assert_eq!(aliases.resolve(AliasKind::Concept, "zeta"), "alpha");
    }

    #[test]
    fn alias_map_resolves_a_chain_leading_into_a_cycle() {
        let m = Matching::default();
        // x -> a -> b -> a: the cycle is {a, b}; x must land on the
        // cycle's representative too, not stop one hop short of it.
        let lines = vec![
            AliasLine {
                alias: "x".into(),
                canonical: "a".into(),
                kind: AliasKind::Concept,
            },
            AliasLine {
                alias: "a".into(),
                canonical: "b".into(),
                kind: AliasKind::Concept,
            },
            AliasLine {
                alias: "b".into(),
                canonical: "a".into(),
                kind: AliasKind::Concept,
            },
        ];
        let aliases = AliasMap::from_batch(&m, &lines);
        assert_eq!(aliases.resolve(AliasKind::Concept, "x"), "a");
        assert_eq!(aliases.resolve(AliasKind::Concept, "a"), "a");
        assert_eq!(aliases.resolve(AliasKind::Concept, "b"), "a");
    }

    #[test]
    fn alias_map_self_mapping_is_a_no_op() {
        let m = Matching::default();
        let lines = vec![AliasLine {
            alias: "Apple".into(),
            canonical: "apple".into(), // normalizes to the same spelling
            kind: AliasKind::Concept,
        }];
        let aliases = AliasMap::from_batch(&m, &lines);
        assert_eq!(aliases.resolve(AliasKind::Concept, "apple"), "apple");
    }

    #[test]
    fn alias_map_conflict_keeps_the_first_declaration() {
        let m = Matching::default();
        let lines = vec![
            AliasLine {
                alias: "a".into(),
                canonical: "first".into(),
                kind: AliasKind::Concept,
            },
            AliasLine {
                alias: "a".into(),
                canonical: "second".into(),
                kind: AliasKind::Concept,
            },
        ];
        let aliases = AliasMap::from_batch(&m, &lines);
        assert_eq!(aliases.resolve(AliasKind::Concept, "a"), "first");
    }

    #[test]
    fn alias_map_keeps_concept_and_label_namespaces_independent() {
        let m = Matching::default();
        let lines = vec![
            AliasLine {
                alias: "x".into(),
                canonical: "concept-canonical".into(),
                kind: AliasKind::Concept,
            },
            AliasLine {
                alias: "x".into(),
                canonical: "label-canonical".into(),
                kind: AliasKind::Label,
            },
        ];
        let aliases = AliasMap::from_batch(&m, &lines);
        assert_eq!(
            aliases.resolve(AliasKind::Concept, "x"),
            "concept-canonical"
        );
        assert_eq!(aliases.resolve(AliasKind::Label, "x"), "label-canonical");
    }

    #[test]
    fn alias_expansion_none_resolves_everything_to_itself() {
        let m = Matching {
            alias_expansion: AliasExpansion::None,
            ..Matching::default()
        };
        let lines = vec![AliasLine {
            alias: "a".into(),
            canonical: "b".into(),
            kind: AliasKind::Concept,
        }];
        let aliases = AliasMap::from_batch(&m, &lines);
        assert_eq!(aliases.resolve(AliasKind::Concept, "a"), "a");
    }

    #[test]
    fn parse_alias_kind_recognizes_only_the_two_batch_kinds() {
        assert_eq!(parse_alias_kind("concept"), Some(AliasKind::Concept));
        assert_eq!(parse_alias_kind("label"), Some(AliasKind::Label));
        assert_eq!(parse_alias_kind("mystery"), None);
    }

    // ---------- jaccard ----------

    #[test]
    fn jaccard_of_identical_sets_is_one() {
        let a: BTreeSet<AssocKey> = [key("d", "s", "l", "o")].into_iter().collect();
        assert_eq!(jaccard(&a, &a), Some(1.0));
    }

    #[test]
    fn jaccard_of_disjoint_sets_is_zero() {
        let a: BTreeSet<AssocKey> = [key("d", "s1", "l", "o")].into_iter().collect();
        let b: BTreeSet<AssocKey> = [key("d", "s2", "l", "o")].into_iter().collect();
        assert_eq!(jaccard(&a, &b), Some(0.0));
    }

    #[test]
    fn jaccard_of_a_partial_overlap() {
        let a: BTreeSet<AssocKey> = [key("d", "s1", "l", "o"), key("d", "s2", "l", "o")]
            .into_iter()
            .collect();
        let b: BTreeSet<AssocKey> = [key("d", "s2", "l", "o"), key("d", "s3", "l", "o")]
            .into_iter()
            .collect();
        // intersection={s2}, union={s1,s2,s3} -> 1/3
        assert_eq!(jaccard(&a, &b), Some(1.0 / 3.0));
    }

    #[test]
    fn jaccard_of_two_empty_sets_is_none() {
        let a: BTreeSet<AssocKey> = BTreeSet::new();
        let b: BTreeSet<AssocKey> = BTreeSet::new();
        assert_eq!(jaccard(&a, &b), None);
    }

    // ---------- variation ----------

    fn presence_from(runs: Vec<(usize, Vec<f64>, Vec<Option<u64>>)>) -> KeyPresence {
        let mut presence = KeyPresence::default();
        for (run_index, weights, paragraphs) in runs {
            presence.by_run.insert(
                run_index,
                KeyObservation {
                    weights,
                    paragraphs: paragraphs.into_iter().collect(),
                },
            );
        }
        presence
    }

    #[test]
    fn polarity_varies_when_signs_disagree_across_runs() {
        let same = presence_from(vec![(1, vec![1.0], vec![]), (2, vec![2.0], vec![])]);
        assert!(!polarity_varies(&same));
        let different = presence_from(vec![(1, vec![1.0], vec![]), (2, vec![-1.0], vec![])]);
        assert!(polarity_varies(&different));
    }

    #[test]
    fn polarity_treats_zero_as_its_own_category() {
        let presence = presence_from(vec![(1, vec![0.0], vec![]), (2, vec![1.0], vec![])]);
        assert!(polarity_varies(&presence));
    }

    #[test]
    fn weight_varies_only_past_the_tolerance() {
        let presence = presence_from(vec![(1, vec![1.0], vec![]), (2, vec![1.5], vec![])]);
        assert!(
            !weight_varies(&presence, 0.5),
            "spread exactly equal to tolerance is not variation"
        );
        assert!(weight_varies(&presence, 0.49));
    }

    #[test]
    fn attribution_varies_when_paragraph_sets_disagree() {
        let same = presence_from(vec![
            (1, vec![1.0], vec![Some(0)]),
            (2, vec![1.0], vec![Some(0)]),
        ]);
        assert!(!attribution_varies(&same));
        let different = presence_from(vec![
            (1, vec![1.0], vec![Some(0)]),
            (2, vec![1.0], vec![Some(1)]),
        ]);
        assert!(attribution_varies(&different));
    }

    // ---------- keyed_associations / PresenceTable ----------

    #[test]
    fn keyed_associations_merges_spellings_that_normalize_to_the_same_key() {
        let m = Matching::default();
        let aliases = AliasMap::from_batch(&m, &[]);
        let associations = vec![
            RawAssociation {
                subject: "Apple".into(),
                label: "grows in".into(),
                object: "Orchard".into(),
                weight: 1.0,
                paragraph: Some(0),
            },
            RawAssociation {
                subject: "apple".into(),
                label: "grows in".into(),
                object: "orchard".into(),
                weight: -1.0,
                paragraph: Some(1),
            },
        ];
        let keyed = keyed_associations(&m, "doc", &associations, &aliases);
        assert_eq!(keyed.len(), 1, "both lines fold to the same key");
        let observation = keyed.values().next().unwrap();
        assert_eq!(observation.weights, vec![1.0, -1.0]);
        assert_eq!(observation.paragraphs, BTreeSet::from([Some(0), Some(1)]));
    }

    #[test]
    fn presence_table_tracks_n_present_and_runs_across_inserts() {
        let mut table = PresenceTable::default();
        let k = key("doc", "s", "l", "o");
        let mut first = BTreeMap::new();
        first.insert(
            k.clone(),
            KeyObservation {
                weights: vec![1.0],
                paragraphs: BTreeSet::new(),
            },
        );
        table.insert_run(1, first);
        let mut third = BTreeMap::new();
        third.insert(
            k.clone(),
            KeyObservation {
                weights: vec![1.0],
                paragraphs: BTreeSet::new(),
            },
        );
        table.insert_run(3, third);

        assert_eq!(table.len(), 1);
        let presence = &table.iter().next().unwrap().1;
        assert_eq!(presence.n_present(), 2);
        assert_eq!(presence.runs().collect::<Vec<_>>(), vec![1, 3]);
    }
}
