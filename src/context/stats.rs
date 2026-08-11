use super::{
    AliasRecord, AttributionId, AttributionLocatorRecord, AttributionRecord, ConceptId,
    ConceptRecord, Context, EdgeId, EdgeRecord, LabelId, LabelRecord, SourceId, SourceRecord,
    clamp_unit_or, dead_ratio_of,
};

/// The default floor for bigram-overlap matches: Dice below this is
/// noise, not a near-miss spelling, and is dropped rather than surfacing
/// distant concepts on every shared 2-gram. Vocabularies differ in how
/// much fuzz they can afford, so this is only the default — tunable per
/// context via [`Context::set_dice_floor`] and per call via
/// [`Context::resolve_with_floor`].
pub(super) const DICE_FLOOR: f64 = 0.3;

impl Context {
    /// The durability watermark stored in this context's image header
    /// — see the field's documentation; the `Context` never interprets
    /// it.
    pub fn applied_seq(&self) -> u64 {
        self.applied_seq
    }

    /// Sets the durability watermark the next [`Context::to_bytes`]
    /// will persist. The caller (the server's WAL) owns the meaning.
    pub fn set_applied_seq(&mut self, seq: u64) {
        self.applied_seq = seq;
    }

    /// The fuzzy-entry floor this context applies when a call does not
    /// name one: bigram-Dice matches below it are dropped as noise.
    pub fn dice_floor(&self) -> f64 {
        self.dice_floor.unwrap_or(DICE_FLOOR)
    }

    /// Tunes the fuzzy-entry floor for this context — lower admits more
    /// distant near-miss spellings (vocabularies with heavy 表記ゆれ),
    /// higher keeps entry strict (curated glossaries). `None` returns to
    /// the default. Clamped into [0, 1]. Config, not knowledge: the
    /// value is not part of the persistent image, so whoever restores a
    /// context from bytes must re-apply it.
    pub fn set_dice_floor(&mut self, dice_floor: Option<f64>) {
        self.dice_floor = dice_floor.map(|floor| clamp_unit_or(floor, 1.0));
    }

    /// How many distinct (subject, label, object) associations are stored.
    pub fn association_count(&self) -> usize {
        self.edges.len()
    }

    /// How many distinct concepts are interned.
    pub fn concept_count(&self) -> usize {
        self.concepts.len()
    }

    /// How many distinct relation labels are interned.
    pub fn label_count(&self) -> usize {
        self.labels.len()
    }

    /// How many distinct sources have been named by `associate_from`.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Live count of edges with `count == 0` — dead weight `compacted()`
    /// would shed right now. See the field doc for how this differs from
    /// [`CompactionStats::dead_edges`].
    pub fn dead_edges(&self) -> usize {
        self.dead_edges
    }

    /// Live count of attribution records unlinked from every chain but
    /// not yet reclaimed.
    pub fn dead_attributions(&self) -> usize {
        self.dead_attributions
    }

    /// Lower-bound count of arena bytes occupied by removed aliases'
    /// spellings. See the field doc for why this is a lower bound.
    pub fn arena_slack(&self) -> usize {
        self.arena_slack
    }

    /// Fraction of associations that are currently dead weight — the
    /// signal for deciding whether a context is due for compaction.
    pub fn dead_ratio(&self) -> f64 {
        dead_ratio_of(self.dead_edges, self.association_count())
    }

    /// The most connected concepts — name plus total degree, most
    /// connected first, ties toward earlier interning. This is the
    /// mechanical "what is this context about" signal: unlike a
    /// hand-written summary it cannot go stale, so a routing directory
    /// can show it next to the prose description.
    ///
    /// Degree computation below is unavoidably O(concepts + edges) —
    /// every concept's chains have to be walked at least once — but
    /// selecting the top `limit` out of that does not need a full
    /// O(n log n) sort of the whole concept table: `select_nth_unstable_by`
    /// partitions around the `limit`th element in O(n), and only the
    /// kept prefix (usually `limit`, a small directory-display count)
    /// pays the final O(limit log limit) sort. Exactly
    /// result-equivalent to sorting-then-truncating, not merely
    /// close: the comparator's `(id)` tiebreak makes it a total order
    /// over `ranked`, so there are no ties left for
    /// `select_nth_unstable_by` to resolve arbitrarily — the set of
    /// elements it leaves in the front partition is the same set a
    /// full sort's first `limit` would be.
    pub fn top_concepts(&self, limit: usize) -> Vec<(&str, usize)> {
        let mut ranked: Vec<(usize, ConceptId)> = self
            .concepts
            .iter()
            .enumerate()
            .map(|(id, _)| {
                let id = id as u32;
                // outgoing_count/incoming_count are chain LENGTHS — every
                // edge ever created, including ones retract_association/
                // retract_source zeroed without unlinking. Walk the chains
                // and count only count > 0 edges instead, the same
                // dead-edge test describe/explore/adjacent apply, so a
                // concept that has had every association withdrawn can't
                // still rank as "most connected".
                let degree = self
                    .outgoing(id)
                    .filter(|&edge_id| self.edges[edge_id as usize].count > 0)
                    .count()
                    + self
                        .incoming(id)
                        .filter(|&edge_id| self.edges[edge_id as usize].count > 0)
                        .count();
                (degree, id)
            })
            .collect();
        let order = |a: &(usize, ConceptId), b: &(usize, ConceptId)| {
            b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1))
        };
        // Clamped rather than guarded separately for "limit past the
        // table": `select_nth_unstable_by(limit - 1, ..)` at exactly
        // `limit == ranked.len()` is valid (index `limit - 1` is the
        // last one) and harmless — the truncate below is then a no-op
        // and the final sort re-derives the same order regardless —
        // so a SEPARATE `limit < ranked.len()` branch would only ever
        // skip a redundant call, never change the answer; folding the
        // "past the table" case into `.min()` here means every path
        // through this function calls `select_nth_unstable_by` at
        // most once, uniformly.
        let limit = limit.min(ranked.len());
        if limit == 0 {
            return Vec::new();
        }
        ranked.select_nth_unstable_by(limit - 1, order);
        ranked.truncate(limit);
        ranked.sort_unstable_by(order);
        ranked
            .into_iter()
            .map(|(degree, id)| (self.concept_name(id), degree))
            .collect()
    }

    /// Rough resident-memory estimate: exact bytes for the arena and the
    /// five record tables, plus estimates for the derived indexes (the
    /// name → id maps own a second copy of every interned name, the
    /// triple index and the (edge, source) attribution index each cost
    /// their key and value, and each hash-map entry carries bookkeeping
    /// overhead; the lowercase shadows mirror the arena again). Intended
    /// for cache budgeting — deciding what stays in memory — not
    /// accounting-grade measurement.
    pub fn footprint(&self) -> usize {
        let tables = self.arena.len()
            + self.concepts.len() * size_of::<ConceptRecord>()
            + self.labels.len() * size_of::<LabelRecord>()
            + self.sources.len() * size_of::<SourceRecord>()
            + self.edges.len() * size_of::<EdgeRecord>()
            + self.attributions.len() * size_of::<AttributionRecord>()
            + (self.concept_aliases.len() + self.label_aliases.len()) * size_of::<AliasRecord>()
            + self.attribution_locators.len() * size_of::<AttributionLocatorRecord>();

        const MAP_ENTRY_OVERHEAD: usize = 48;
        let name_entries = self.concepts.len() + self.labels.len() + self.sources.len();
        let triple_entry = size_of::<(ConceptId, LabelId, ConceptId)>() + size_of::<EdgeId>();
        let attribution_entry = size_of::<(EdgeId, SourceId)>() + size_of::<AttributionId>();
        // The three keyset indexes behind `concept_alias_page`/
        // `label_alias_page`/`label_page` each own a second copy of a
        // subset of names already counted once above (aliases, or
        // canonical labels) — same coarse per-entry overhead, no
        // separate byte count for the duplicated keys.
        let keyset_index_entries =
            self.concept_aliases.len() + self.label_aliases.len() + self.labels.len();
        let derived = self.arena.len() // owned keys of the name → id maps
            + name_entries * MAP_ENTRY_OVERHEAD
            + self.edges.len() * (triple_entry + MAP_ENTRY_OVERHEAD)
            + self.attribution_ids.len() * (attribution_entry + MAP_ENTRY_OVERHEAD)
            // The source → edges reverse index: one map entry per
            // attributing source, one Vec element per live attribution
            // (the same population `attribution_ids` counts).
            + self.source_edges.len() * (size_of::<SourceId>() + size_of::<Vec<EdgeId>>() + MAP_ENTRY_OVERHEAD)
            + self.attribution_ids.len() * size_of::<EdgeId>()
            + keyset_index_entries * MAP_ENTRY_OVERHEAD
            + self.concept_index.footprint()
            + self.label_index.footprint();

        tables + derived
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::test_support::associate_examples;

    /// `set_dice_floor` clamps into [0, 1] via the same helper as
    /// `activate`'s decay, but a NaN floor here must land on 1.0 (the
    /// strictest admission bar), not 0.0: excluding every fuzzy match
    /// beats admitting every one of them.
    #[test]
    fn set_dice_floor_maps_nan_to_the_strictest_floor() {
        let mut context = Context::default();
        context.set_dice_floor(Some(f64::NAN));
        assert_eq!(context.dice_floor(), 1.0);
    }

    #[test]
    fn counts_and_top_concepts_expose_directory_stats() {
        let mut context = Context::default();
        associate_examples(&mut context);
        context
            .associate_from("私", "食べられる", "りんご", -0.2, "文書1", None)
            .unwrap();

        assert_eq!(context.association_count(), 4);
        assert_eq!(context.concept_count(), 4);
        assert_eq!(context.label_count(), 2);
        assert_eq!(context.source_count(), 1);

        // 私 carries 4 edges, りんご 2, the rest 1 each.
        let top = context.top_concepts(2);
        assert_eq!(top, vec![("私", 4), ("りんご", 2)]);
        // A limit beyond the table is just everything.
        assert_eq!(context.top_concepts(100).len(), 4);
    }

    #[test]
    fn top_concepts_excludes_retracted_associations() {
        let mut context = Context::default();
        associate_examples(&mut context);
        context
            .associate_from("私", "食べられる", "りんご", -0.2, "文書1", None)
            .unwrap();

        // Retracting zeroes the edge's count but leaves it linked into
        // both concepts' chains — outgoing_count/incoming_count (chain
        // length) don't shrink. top_concepts must not still credit 私 and
        // りんご for a withdrawn association.
        context
            .retract_association("私", "食べられる", "りんご")
            .unwrap();

        let top = context.top_concepts(4);
        assert_eq!(top.iter().find(|&&(n, _)| n == "私").unwrap().1, 3);
        assert_eq!(top.iter().find(|&&(n, _)| n == "りんご").unwrap().1, 1);
    }

    /// `top_concepts` selects its top `limit` with `select_nth_unstable_by`
    /// plus a bounded final sort rather than sorting the whole concept
    /// table — this pins that the cut is exactly what a full
    /// sort-then-truncate would give, including the one case the two
    /// could disagree on if the comparator's tiebreak were not a total
    /// order: a `limit` that falls INSIDE a tied-degree group. c1, c2,
    /// c3 all carry degree 2, inserted in that order (so their
    /// `ConceptId` tiebreak is c1 < c2 < c3); `limit=2` cuts between
    /// c2 and c3, keeping only the two lowest-id members of the tie.
    #[test]
    fn top_concepts_selects_the_same_prefix_as_a_full_sort_across_a_tied_boundary() {
        let mut context = Context::default();
        context.associate("c0", "r0", "p0", 1.0).unwrap();
        context.associate("c0", "r1", "p1", 1.0).unwrap();
        context.associate("c0", "r2", "p2", 1.0).unwrap();
        for (name, a, b) in [("c1", "p3", "p4"), ("c2", "p5", "p6"), ("c3", "p7", "p8")] {
            context.associate(name, "r3", a, 1.0).unwrap();
            context.associate(name, "r4", b, 1.0).unwrap();
        }

        assert!(
            context.top_concepts(0).is_empty(),
            "a zero limit must be empty, not the whole table"
        );

        let full = context.top_concepts(100);
        assert_eq!(
            full[..4],
            [("c0", 3), ("c1", 2), ("c2", 2), ("c3", 2)],
            "sanity: the intended degree-3-then-tied-degree-2 shape"
        );
        assert_eq!(
            context.top_concepts(2),
            full[..2],
            "a limit cutting inside a tied-degree group must match a full sort's own prefix"
        );
    }

    /// `select_nth_unstable_by(limit - 1, ..)` — a boundary an
    /// off-by-one (`limit` instead of `limit - 1`) would not visibly
    /// break on a small table: Rust's unstable sort/select falls back
    /// to a full sort for small slices, so a handful of concepts stays
    /// correctly ordered even under the wrong index. This builds 25
    /// concepts with distinct, unique degrees — comfortably past that
    /// small-slice threshold — and checks several `limit`s against a
    /// full sort's own prefix, so an off-by-one has nowhere to hide
    /// behind an accidentally-still-sorted small array.
    #[test]
    fn top_concepts_matches_a_full_sort_prefix_at_a_scale_past_the_small_array_fallback() {
        let mut context = Context::default();
        const CONCEPTS: usize = 25;
        for degree in 1..=CONCEPTS {
            let name = format!("c{degree}");
            for partner in 0..degree {
                context
                    .associate(&name, "r", format!("c{degree}p{partner}"), 1.0)
                    .unwrap();
            }
        }

        // Each partner concept ("cDpP") has degree exactly 1 — degree
        // ties with "c1" itself, but every limit checked below stays
        // above that tie: ranks 1..24 are exactly the 24 named
        // concepts of degree 2..25, so no partner (or "c1") is in
        // play yet.
        let full = context.top_concepts(CONCEPTS + 100);
        assert_eq!(
            full[0],
            (format!("c{CONCEPTS}").as_str(), CONCEPTS),
            "sanity: the highest-degree concept must lead"
        );

        for limit in [1, 5, 12, 13, 20, 24] {
            assert_eq!(
                context.top_concepts(limit),
                full[..limit],
                "limit={limit} must match a full sort's own prefix exactly"
            );
        }
    }

    #[test]
    fn footprint_grows_with_content() {
        let mut context = Context::default();
        let empty = context.footprint();
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書", Some(2))
            .unwrap();
        context.add_concept_alias("自分", "私").unwrap();
        context.add_label_alias("好む", "好き").unwrap();
        assert!(context.footprint() > empty);

        // Pin the budgeting formula, not merely monotonic growth: a wrong
        // arithmetic operator can still produce a larger positive estimate.
        const MAP_ENTRY_OVERHEAD: usize = 48;
        let tables = context.arena.len()
            + context.concepts.len() * size_of::<ConceptRecord>()
            + context.labels.len() * size_of::<LabelRecord>()
            + context.sources.len() * size_of::<SourceRecord>()
            + context.edges.len() * size_of::<EdgeRecord>()
            + context.attributions.len() * size_of::<AttributionRecord>()
            + (context.concept_aliases.len() + context.label_aliases.len())
                * size_of::<AliasRecord>()
            + context.attribution_locators.len() * size_of::<AttributionLocatorRecord>();
        let name_entries = context.concepts.len() + context.labels.len() + context.sources.len();
        let triple_entry = size_of::<(ConceptId, LabelId, ConceptId)>() + size_of::<EdgeId>();
        let attribution_entry = size_of::<(EdgeId, SourceId)>() + size_of::<AttributionId>();
        let keyset_index_entries =
            context.concept_aliases.len() + context.label_aliases.len() + context.labels.len();
        let derived = context.arena.len()
            + name_entries * MAP_ENTRY_OVERHEAD
            + context.edges.len() * (triple_entry + MAP_ENTRY_OVERHEAD)
            + context.attribution_ids.len() * (attribution_entry + MAP_ENTRY_OVERHEAD)
            + context.source_edges.len()
                * (size_of::<SourceId>() + size_of::<Vec<EdgeId>>() + MAP_ENTRY_OVERHEAD)
            + context.attribution_ids.len() * size_of::<EdgeId>()
            + keyset_index_entries * MAP_ENTRY_OVERHEAD
            + context.concept_index.footprint()
            + context.label_index.footprint();
        assert_eq!(context.footprint(), tables + derived);
    }
}
