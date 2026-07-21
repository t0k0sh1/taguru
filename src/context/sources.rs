use std::collections::HashSet;

use crate::deadline::{Deadline, DeadlineExceeded};

use super::{
    Context, EdgeRecord, NIL, SourceId, UNSOURCED_SOURCE, UnsourcedEdge, accumulate_saturating,
};

impl Context {
    /// The unsourced share of one edge: weight/count no non-reserved,
    /// live attribution explains. `None` when every unit of `count` is
    /// covered by a real source. `unattributed` is UNSOURCED_SOURCE's
    /// SourceId, pre-resolved once by the caller (avoids a per-attribution
    /// string compare across a potentially large chain).
    fn unsourced_share(
        &self,
        edge: &EdgeRecord,
        unattributed: Option<SourceId>,
    ) -> Option<(f64, u64)> {
        if edge.count == 0 {
            return None;
        }
        let mut attributed_count = 0u64;
        let mut attributed_sum = 0.0f64;
        for (_, record) in self.attribution_chain(edge.first_attribution) {
            if record.count == 0 || Some(record.source) == unattributed {
                continue;
            }
            attributed_count += record.count;
            accumulate_saturating(&mut attributed_sum, record.sum);
        }
        (attributed_count < edge.count).then(|| {
            // edge.sum and attributed_sum are each finite, but their
            // difference can still overflow (f64::MAX − (−f64::MAX)):
            // saturate it like every other cross-record sum.
            let mut residual = edge.sum;
            accumulate_saturating(&mut residual, -attributed_sum);
            (residual, edge.count - attributed_count)
        })
    }

    /// Context-wide unsourced-weight summary: `(edges, total weight)`.
    /// `total` sums `residual.abs()` across every flagged edge — summing
    /// signed residuals across edges would let an over- and under-counted
    /// edge cancel out and hide both. This is a new aggregation policy,
    /// not a port of export.rs's per-edge-only residual math (export.rs
    /// never sums across edges), so don't expect bit-exact equality with
    /// it in tests — compare with a small epsilon tolerance instead.
    /// `ContextStats`, `taguru inspect`, and the `/metrics` gauges all
    /// read through this one method so the three surfaces can't drift
    /// from each other.
    pub fn unsourced_summary(&self) -> (usize, f64) {
        let unattributed = self.source_ids.get(UNSOURCED_SOURCE).copied();
        let mut edges = 0usize;
        let mut total = 0.0f64;
        for edge in &self.edges {
            if let Some((residual, _)) = self.unsourced_share(edge, unattributed) {
                edges += 1;
                accumulate_saturating(&mut total, residual.abs());
            }
        }
        (edges, total)
    }

    /// Every edge with unsourced weight at or past `floor` (compared by
    /// magnitude), in edge-id order. CPU-bound over the whole edge table;
    /// checks `deadline` periodically like `compacted`/`similar_concepts`.
    pub fn unsourced_edges(
        &self,
        floor: f64,
        deadline: Deadline,
    ) -> Result<Vec<UnsourcedEdge>, DeadlineExceeded> {
        // Not `clamp_unit_or`: this floor is a magnitude threshold on
        // accumulated edge weight, not a [0, 1] similarity score (a
        // caller can legitimately ask for 2.5), so only NaN needs a
        // fallback. This is a drift-audit surface, where silently
        // excluding every edge (weight.abs() >= NaN is always false)
        // reads as "the corpus is clean" — the unsafe direction to
        // fail toward. 0.0 keeps every edge in play instead.
        let floor = if floor.is_nan() { 0.0 } else { floor.abs() };
        let unattributed = self.source_ids.get(UNSOURCED_SOURCE).copied();
        let mut out = Vec::new();
        for edge_id in 0..self.edges.len() as u32 {
            if deadline.expired() {
                return Err(DeadlineExceeded);
            }
            let edge = &self.edges[edge_id as usize];
            if let Some((weight, count)) = self.unsourced_share(edge, unattributed)
                && weight.abs() >= floor
            {
                out.push(UnsourcedEdge {
                    association: self.association(edge_id),
                    weight,
                    count,
                });
            }
        }
        Ok(out)
    }

    /// Withdraws one source's contributions: every attribution it made
    /// is removed from its edge's chain and its weight subtracted from
    /// the edge's total — the differential-sync move when a document
    /// changes (retract the old version, re-ingest the new one), instead
    /// of rebuilding the whole context.
    ///
    /// Returns how many associations were touched, or `None` for a
    /// source this context never saw. What retraction deliberately does
    /// NOT do: concepts, labels, and edges minted by the document stay
    /// (the storage is append-only; an edge whose weight nets to 0.0
    /// simply stops carrying knowledge — `activate` already skips it),
    /// unsourced weight on shared edges stays, and the source's name
    /// stays interned so re-asserting from it later just works. The
    /// unlinked attribution records remain as dead space in their
    /// append-only table.
    pub fn retract_source(&mut self, source: &str) -> Option<usize> {
        let &source_id = self.source_ids.get(source)?;
        // The reverse index hands over exactly the edges this source
        // touches: retraction costs the document's own footprint, not a
        // scan of every edge in the context. Processed in edge order,
        // the same order the old full scan visited them.
        let mut edges_of_source = self.source_edges.remove(&source_id).unwrap_or_default();
        edges_of_source.sort_unstable();
        let mut touched = 0usize;
        for edge_id in edges_of_source {
            let edge_index = edge_id as usize;
            // Locate the source's attribution and its predecessor.
            let mut previous = NIL;
            let mut cursor = self.edges[edge_index].first_attribution;
            let mut found = None;
            while cursor != NIL {
                let record = &self.attributions[cursor as usize];
                if record.source == source_id {
                    found = Some((previous, cursor, record.sum, record.count));
                    break;
                }
                previous = cursor;
                cursor = record.next;
            }
            let Some((previous, cursor, sum, count)) = found else {
                continue;
            };

            let next = self.attributions[cursor as usize].next;
            let edge = &mut self.edges[edge_index];
            // Subtraction accumulates too: a saturated edge sum minus a
            // saturated attribution sum can overshoot the representable
            // range just like two additions can.
            accumulate_saturating(&mut edge.sum, -sum);
            edge.count = edge.count.saturating_sub(count);
            // A linked attribution always has a positive count, and a
            // live edge cannot reach zero more than once.
            if edge.count == 0 {
                self.dead_edges += 1;
            }
            if previous == NIL {
                edge.first_attribution = next;
            }
            if edge.last_attribution == cursor {
                edge.last_attribution = previous;
            }
            if previous != NIL {
                self.attributions[previous as usize].next = next;
            }
            // The record is off the chain and now dead space; drop its
            // index entry too, or a later re-assertion from this source
            // would fold into the unlinked record instead of appending a
            // fresh one — resurrecting retracted weight.
            self.attribution_ids.remove(&(edge_id, source_id));
            touched += 1;
        }
        self.dead_attributions += touched;
        Some(touched)
    }

    /// The read-only twin of [`Self::retract_source`]'s count: how
    /// many distinct edges this source touches, without unlinking
    /// anything — `POST /import?dry_run=true`'s preview of what a
    /// retraction would report.
    ///
    /// `source_edges` is not pruned by [`Self::retract_association`],
    /// which can unlink this source's attribution on one of its edges
    /// without touching the reverse index — so a raw `Vec::len` would
    /// overcount past that edge. Worse, if the source later re-asserts
    /// onto that same edge, `attribute` has no way to tell the stale
    /// reverse-index entry apart from a fresh one and appends a second
    /// one, so the same live edge can appear twice. Each candidate is
    /// confirmed live against `attribution_ids` (the same check
    /// `retract_source` itself relies on to skip a dead entry) and
    /// deduplicated so a retract-then-reassert cycle is still counted
    /// once.
    pub fn count_source_edges(&self, source: &str) -> usize {
        let Some(&source_id) = self.source_ids.get(source) else {
            return 0;
        };
        let Some(edges) = self.source_edges.get(&source_id) else {
            return 0;
        };
        let mut seen = HashSet::new();
        edges
            .iter()
            .filter(|&&edge_id| self.attribution_ids.contains_key(&(edge_id, source_id)))
            .filter(|&&edge_id| seen.insert(edge_id))
            .count()
    }

    /// Withdraws one association outright: the `(subject, label,
    /// object)` edge — the names resolve through aliases, like every
    /// other entry point — has every attribution unlinked and its
    /// total zeroed, so it stops carrying knowledge entirely,
    /// sourceless weight included. This is the surgical correction for
    /// a fact that should never have been asserted (an extraction
    /// error, a merge mistake); a fact that is CONTESTED wants a
    /// negative-weight assertion instead, which preserves the dispute
    /// as evidence.
    ///
    /// Returns how many attributions were unlinked, or `None` when the
    /// triple names no live edge — unknown names, no such edge, or an
    /// edge already fully retracted — so a replayed retraction is a
    /// no-op, never an error.
    ///
    /// What it deliberately does NOT do mirrors [`Context::
    /// retract_source`]: the concepts, the label, the edge record, and
    /// every source name stay; the edge remains visible to `query` at
    /// weight 0.0 / count 0 until compaction sheds it (`activate`
    /// already skips it); the unlinked attribution records stay as
    /// dead space in their append-only table; and re-asserting the
    /// same triple later just works.
    pub fn retract_association(
        &mut self,
        subject: &str,
        label: &str,
        object: &str,
    ) -> Option<usize> {
        let &subject_id = self.concept_ids.get(subject)?;
        let &label_id = self.label_ids.get(label)?;
        let &object_id = self.concept_ids.get(object)?;
        let &edge_id = self.edge_ids.get(&(subject_id, label_id, object_id))?;
        let edge_index = edge_id as usize;
        if self.edges[edge_index].count == 0 {
            // Already fully retracted — `count == 0` is the dead-edge
            // test everywhere else too (`compacted`, the weight
            // getter, the export): nothing left to withdraw.
            return None;
        }

        // Unlink the whole attribution chain, dropping each (edge,
        // source) index entry — the same resurrection hazard
        // retract_source guards: a later re-assertion must append a
        // fresh record, never fold into an unlinked one.
        let mut cursor = self.edges[edge_index].first_attribution;
        let mut unlinked = 0usize;
        while cursor != NIL {
            let record = &self.attributions[cursor as usize];
            self.attribution_ids.remove(&(edge_id, record.source));
            cursor = record.next;
            unlinked += 1;
        }
        let edge = &mut self.edges[edge_index];
        edge.first_attribution = NIL;
        edge.last_attribution = NIL;
        edge.sum = 0.0;
        edge.count = 0;
        // The early return above guarantees this edge was live on entry,
        // so this is unconditionally a live→dead transition.
        self.dead_edges += 1;
        self.dead_attributions += unlinked;
        Some(unlinked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::Attribution;
    use crate::context::test_support::weight_between;

    #[test]
    fn retracting_an_extreme_source_saturates_instead_of_reaching_infinity() {
        // Retraction subtracts an attribution sum from the edge sum; with
        // saturated magnitudes on both sides that difference can overshoot
        // f64's range exactly like an addition can. A raw subtraction here
        // once produced an infinite edge sum — a context that saved but
        // never loaded again.
        let mut context = Context::default();
        context
            .associate_from("a", "r", "b", -f64::MAX, "源Y", None)
            .unwrap();
        context
            .associate_from("a", "r", "b", -f64::MAX, "源Y", None)
            .unwrap();
        // Fold the edge sum back to the positive extreme while 源Y's
        // attribution stays saturated at -f64::MAX.
        context
            .associate_from("a", "r", "b", f64::MAX, "源Z", None)
            .unwrap();
        context
            .associate_from("a", "r", "b", f64::MAX, "源Z", None)
            .unwrap();

        // edge.sum - (-f64::MAX) would round to +Infinity unsaturated.
        assert_eq!(context.retract_source("源Y"), Some(1));

        assert!(weight_between(&context, "a", "r", "b").is_finite());
        Context::from_bytes(&context.to_bytes())
            .expect("a context whose retraction saturated must still round-trip");
    }
    #[test]
    fn unsourced_residuals_saturate_across_opposite_extremes() {
        // The unsourced residual is `edge.sum - attributed_sum`. With a
        // positive-saturated edge sum and a negative-saturated attributed
        // sum, that difference (f64::MAX − (−f64::MAX)) rounds to +inf
        // unless it saturates like every other cross-record sum — and the
        // context-wide total then sums those residuals, so two saturated
        // edges would overflow it too. An infinite residual would poison
        // the /metrics gauges and `taguru inspect` that read the summary.
        let mut context = Context::default();
        for subject in ["a", "c"] {
            // A real source drives the attributed sum to −f64::MAX...
            context
                .associate_from(subject, "r", "b", -f64::MAX, "src", None)
                .unwrap();
            context
                .associate_from(subject, "r", "b", -f64::MAX, "src", None)
                .unwrap();
            // ...while sourceless folds pull the edge sum back to +f64::MAX,
            // leaving unsourced count for the residual to describe.
            context.associate(subject, "r", "b", f64::MAX).unwrap();
            context.associate(subject, "r", "b", f64::MAX).unwrap();
        }

        // Per-edge: each residual stays finite instead of reaching +inf.
        let flagged = context.unsourced_edges(0.0, Deadline::unbounded()).unwrap();
        assert_eq!(flagged.len(), 2);
        assert!(
            flagged.iter().all(|edge| edge.weight.is_finite()),
            "each residual must saturate rather than reach infinity"
        );

        // Context-wide: summing two saturated residuals must saturate too,
        // not overflow the total to +inf.
        let (edges, total) = context.unsourced_summary();
        assert_eq!(edges, 2);
        assert!(
            total.is_finite() && total > 0.0,
            "the summary total must stay finite across saturated residuals, got {total}"
        );
    }
    #[test]
    fn retract_source_withdraws_its_contributions() {
        let mut context = Context::default();
        context
            .associate_from("a", "r", "b", 1.0, "旧版", None)
            .unwrap();
        // 新版 and 第三者 carry locators, so their retraction/round-trip
        // below also exercises the attribution-locator side table.
        context
            .associate_from("a", "r", "b", 2.0, "新版", Some(4))
            .unwrap();
        context
            .associate_from("a", "r", "b", 0.5, "第三者", Some(7))
            .unwrap();
        context
            .associate_from("a", "r", "c", 1.0, "旧版", None)
            .unwrap();
        context.associate("a", "r", "d", 1.0).unwrap(); // unsourced stays

        // 旧版 contributed to two edges; both lose exactly its share.
        // "a"-"r"-"b" had sum 3.5 across 3 attributions (1.0 + 2.0 +
        // 0.5); losing 旧版's (1.0, count 1) leaves sum 2.5, count 2.
        assert_eq!(context.retract_source("旧版"), Some(2));
        assert_eq!(weight_between(&context, "a", "r", "b"), 1.25);
        assert_eq!(
            context.query(Some("a"), None, Some("b"))[0].attributions,
            vec![
                Attribution {
                    source: "新版".to_string(),
                    weight: 2.0,
                    count: 1,
                    paragraph: Some(4),
                },
                Attribution {
                    source: "第三者".to_string(),
                    weight: 0.5,
                    count: 1,
                    paragraph: Some(7),
                },
            ]
        );
        // A fully retracted edge nets to zero: still queryable, no
        // longer knowledge activate would carry.
        assert_eq!(weight_between(&context, "a", "r", "c"), 0.0);
        assert!(
            context.query(Some("a"), None, Some("c"))[0]
                .attributions
                .is_empty()
        );
        assert_eq!(weight_between(&context, "a", "r", "d"), 1.0);
        // Of the two edges 旧版 touched, only "a"-"r"-"c" died (its only
        // source); "a"-"r"-"b" survives on 新版 and 第三者.
        assert_eq!(context.dead_edges(), 1);
        assert_eq!(context.dead_attributions(), 2);
        // Retracting an unknown source is a no-op — no phantom counting.
        assert_eq!(context.retract_source("存在しない出典"), None);
        assert_eq!(context.dead_edges(), 1);
        assert_eq!(context.dead_attributions(), 2);

        // Unlinking the tail keeps chains appendable, and the image —
        // now carrying orphaned attribution records — must round-trip.
        // Losing 第三者's (0.5, count 1) leaves sum 2.0, count 1 — a
        // weight of 2.0 that happens to match the pre-retraction figure.
        assert_eq!(context.retract_source("第三者"), Some(1));
        assert_eq!(weight_between(&context, "a", "r", "b"), 2.0);
        // "a"-"r"-"b" still carries 新版 alone — it does not die, so
        // dead_edges holds at 1 while dead_attributions grows to 3.
        assert_eq!(context.dead_edges(), 1);
        assert_eq!(context.dead_attributions(), 3);
        context
            .associate_from("a", "r", "b", 0.5, "旧版", None)
            .unwrap();
        // Back up to sum 2.5, count 2: weight 1.25, not the old 2.5.
        assert_eq!(weight_between(&context, "a", "r", "b"), 1.25);
        // A fresh attribution record on a never-dead edge — the counters
        // are exactly what they were before this re-assertion.
        assert_eq!(context.dead_edges(), 1);
        assert_eq!(context.dead_attributions(), 3);
        let restored = Context::from_bytes(&context.to_bytes()).expect("image must load");
        assert_eq!(
            restored.query(Some("a"), None, None),
            context.query(Some("a"), None, None)
        );
        assert_eq!(restored.dead_edges(), context.dead_edges());
        assert_eq!(restored.dead_attributions(), context.dead_attributions());
    }
    #[test]
    fn count_source_edges_previews_retract_source_without_touching_anything() {
        let mut context = Context::default();
        context
            .associate_from("a", "r", "b", 1.0, "旧版", None)
            .unwrap();
        context
            .associate_from("a", "r", "c", 1.0, "旧版", None)
            .unwrap();
        context.associate("a", "r", "d", 1.0).unwrap();

        assert_eq!(context.count_source_edges("旧版"), 2);
        assert_eq!(context.count_source_edges("存在しない出典"), 0);
        // A read, not a retraction: calling it again reports the same
        // count, and the real retraction afterward still sees both edges.
        assert_eq!(context.count_source_edges("旧版"), 2);
        assert_eq!(context.retract_source("旧版"), Some(2));
        assert_eq!(context.count_source_edges("旧版"), 0);
    }
    #[test]
    fn count_source_edges_ignores_an_edge_retracted_via_retract_association() {
        let mut context = Context::default();
        context
            .associate_from("a", "r", "b", 1.0, "旧版", None)
            .unwrap();
        context
            .associate_from("a", "r", "c", 1.0, "旧版", None)
            .unwrap();
        assert_eq!(context.count_source_edges("旧版"), 2);

        // retract_association unlinks (a, r, b) outright, every source
        // included — the reverse index still lists the edge under
        // "旧版", but the attribution itself is gone.
        assert_eq!(context.retract_association("a", "r", "b"), Some(1));
        assert_eq!(context.count_source_edges("旧版"), 1);

        // retract_source must agree with the preview: only the one
        // still-live edge is actually touched.
        assert_eq!(context.retract_source("旧版"), Some(1));
        assert_eq!(context.count_source_edges("旧版"), 0);
    }
    #[test]
    fn count_source_edges_does_not_double_count_a_retract_then_reassert_cycle() {
        let mut context = Context::default();
        context
            .associate_from("a", "r", "b", 1.0, "旧版", None)
            .unwrap();
        assert_eq!(context.count_source_edges("旧版"), 1);

        // retract_association unlinks the attribution but leaves the
        // reverse-index entry for (a, r, b) under "旧版" in place.
        assert_eq!(context.retract_association("a", "r", "b"), Some(1));
        assert_eq!(context.count_source_edges("旧版"), 0);

        // Re-asserting from the same source onto the same edge appends a
        // second reverse-index entry — attribute() cannot tell it apart
        // from the stale one. Both now resolve to the same, single live
        // attribution, so the edge must still count once, not twice.
        context
            .associate_from("a", "r", "b", 1.0, "旧版", None)
            .unwrap();
        assert_eq!(context.count_source_edges("旧版"), 1);
        assert_eq!(context.retract_source("旧版"), Some(1));
    }
    /// Retracting twice is idempotent — the second call finds no live
    /// attributions (the source → edges reverse index was emptied) and
    /// reports zero — and the name stays usable for a fresh assertion,
    /// whose retraction works again.
    #[test]
    fn retracting_a_source_twice_reports_zero_the_second_time() {
        let mut context = Context::default();
        context
            .associate_from("A", "r", "B", 2.0, "doc", None)
            .unwrap();
        assert_eq!(context.retract_source("doc"), Some(1));
        assert_eq!(context.retract_source("doc"), Some(0));
        context
            .associate_from("A", "r", "B", 3.0, "doc", None)
            .unwrap();
        assert_eq!(context.retract_source("doc"), Some(1));
    }
    /// The single-association counterpart of retract_source: the named
    /// edge nets to zero — every source's contribution withdrawn at
    /// once, unsourced weight included — while its siblings, each
    /// document's other facts, and the vocabulary stay untouched.
    #[test]
    fn retract_association_withdraws_one_edge_outright() {
        let mut context = Context::default();
        context
            .associate_from("a", "r", "b", 1.0, "doc1", Some(0))
            .unwrap();
        context
            .associate_from("a", "r", "b", 2.0, "doc2", Some(3))
            .unwrap();
        context.associate("a", "r", "b", 0.5).unwrap(); // unsourced share
        context
            .associate_from("a", "r", "c", 1.0, "doc1", None)
            .unwrap();

        // Two attribution records (doc1, doc2) — the unsourced share
        // never had one; it vanishes with the edge total.
        assert_eq!(context.retract_association("a", "r", "b"), Some(2));
        assert_eq!(weight_between(&context, "a", "r", "b"), 0.0);
        let dead = &context.query(Some("a"), None, Some("b"))[0];
        assert_eq!(dead.count, 0);
        assert!(dead.attributions.is_empty());
        // The same document's OTHER fact is untouched — the point of
        // edge-granular retraction.
        assert_eq!(weight_between(&context, "a", "r", "c"), 1.0);
        // "a"-"r"-"b" is the only one of the two edges that died.
        assert_eq!(context.dead_edges(), 1);
        assert_eq!(context.dead_attributions(), 2);

        // Idempotent: a second retraction finds nothing live, and must
        // not double-count an edge that was already dead.
        assert_eq!(context.retract_association("a", "r", "b"), None);
        assert_eq!(context.dead_edges(), 1);
        assert_eq!(context.dead_attributions(), 2);

        // Re-assertion appends FRESH records — never folds into the
        // unlinked dead space (the resurrection hazard retract_source
        // also guards) — and revives the edge: dead_edges drops back
        // down, while dead_attributions (never reclaimed outside
        // compaction) stays at 2.
        context
            .associate_from("a", "r", "b", 3.0, "doc1", None)
            .unwrap();
        assert_eq!(weight_between(&context, "a", "r", "b"), 3.0);
        assert_eq!(
            context.query(Some("a"), None, Some("b"))[0].attributions,
            vec![Attribution {
                source: "doc1".to_string(),
                weight: 3.0,
                count: 1,
                paragraph: None,
            }]
        );
        assert_eq!(context.dead_edges(), 0);
        assert_eq!(context.dead_attributions(), 2);

        // And the image — orphaned attribution records behind — must
        // round-trip, dead-weight counters included (seeded fresh from
        // the reloaded chains rather than persisted).
        let restored = Context::from_bytes(&context.to_bytes()).expect("image must load");
        assert_eq!(
            restored.query(Some("a"), None, None),
            context.query(Some("a"), None, None)
        );
        assert_eq!(restored.dead_edges(), context.dead_edges());
        assert_eq!(restored.dead_attributions(), context.dead_attributions());
    }
    /// Entry resolution matches every other entry point: aliases in
    /// both namespaces resolve, unknown names answer None — and a
    /// contested edge (net zero, still live) is retractable, while a
    /// fully retracted one is not.
    #[test]
    fn retract_association_resolves_aliases_and_reports_absence() {
        let mut context = Context::default();
        context
            .associate_from("青嶺酒造", "代表銘柄", "青嶺", 1.0, "doc", None)
            .unwrap();
        context
            .add_concept_alias("Aomine Brewery", "青嶺酒造")
            .unwrap();
        context.add_label_alias("flagship", "代表銘柄").unwrap();

        assert_eq!(
            context.retract_association("未知", "代表銘柄", "青嶺"),
            None
        );
        assert_eq!(
            context.retract_association("青嶺酒造", "未知", "青嶺"),
            None
        );
        assert_eq!(
            context.retract_association("青嶺酒造", "代表銘柄", "未知"),
            None
        );

        // Aliases from both namespaces land on the canonical edge.
        assert_eq!(
            context.retract_association("Aomine Brewery", "flagship", "青嶺"),
            Some(1)
        );
        assert_eq!(
            weight_between(&context, "青嶺酒造", "代表銘柄", "青嶺"),
            0.0
        );

        // Contested is not dead: sum 0.0 with live assertions still
        // carries the dispute, and retraction takes all of it.
        context
            .associate_from("x", "r", "y", 1.0, "for", None)
            .unwrap();
        context
            .associate_from("x", "r", "y", -1.0, "against", None)
            .unwrap();
        assert_eq!(weight_between(&context, "x", "r", "y"), 0.0);
        assert_eq!(context.retract_association("x", "r", "y"), Some(2));
        assert_eq!(context.retract_association("x", "r", "y"), None);
    }
    #[test]
    fn retract_source_subtracts_its_full_re_assertion_count() {
        // A source that re-asserted the same edge multiple times folds
        // into one attribution record whose `count` tracks how many
        // times it re-asserted. Retracting it must subtract that whole
        // count, not just 1, or repeated retract/re-assert cycles would
        // leak count into the edge forever.
        let mut context = Context::default();
        context
            .associate_from("私", "好き", "りんご", 1.0, "A", None)
            .unwrap();
        context
            .associate_from("私", "好き", "りんご", 1.0, "A", None)
            .unwrap();
        context
            .associate_from("私", "好き", "りんご", 2.0, "A", None)
            .unwrap();
        context
            .associate_from("私", "好き", "りんご", 2.0, "B", None)
            .unwrap();

        // A's attribution record: sum 4.0, count 3. B's: sum 2.0, count
        // 1. Edge: sum 6.0, count 4, weight 1.5.
        let before = &context.recall("私")[0];
        assert_eq!(before.count, 4);
        assert_eq!(before.weight, 1.5);

        // Retracting A drops its whole (sum 4.0, count 3): edge becomes
        // sum 2.0, count 1, weight 2.0 — B's contribution alone.
        assert_eq!(context.retract_source("A"), Some(1));
        let after = &context.recall("私")[0];
        assert_eq!(after.count, 1);
        assert_eq!(after.weight, 2.0);
    }
    #[test]
    fn retracting_every_source_of_a_migrated_edge_does_not_underflow_count() {
        // A v4 image's edge count is synthesized as the attribution
        // chain length (see
        // `v4_images_synthesize_count_from_the_attribution_chain`), not
        // as a flat 1 — retracting every one of several sources in turn
        // must land exactly at 0, never wrap past it via u64 underflow.
        let mut context = Context::default();
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書1", None)
            .unwrap();
        context
            .associate_from("私", "好き", "りんご", 2.0, "文書2", None)
            .unwrap();
        let v4 = context.to_bytes_as_version(4);
        let mut restored = Context::from_bytes(&v4).expect("v4 image must load");

        assert_eq!(restored.retract_source("文書1"), Some(1));
        assert_eq!(restored.retract_source("文書2"), Some(1));

        let after = &restored.recall("私")[0];
        assert_eq!(after.count, 0);
        assert_eq!(after.weight, 0.0);
    }
    #[test]
    fn unsourced_summary_and_edges_count_only_the_unattributed_share() {
        let mut context = Context::default();
        // Fully sourceless: the whole edge is unsourced.
        context.associate("a", "r", "b", 3.0).unwrap();
        // Fully sourced: no unsourced share at all.
        context
            .associate_from("c", "r", "d", 2.0, "src1", None)
            .unwrap();
        // Mixed: one sourced assertion plus one sourceless one on the
        // same edge — only the sourceless portion counts.
        context
            .associate_from("e", "r", "f", 1.0, "src1", None)
            .unwrap();
        context.associate("e", "r", "f", 4.0).unwrap();

        let (edges, total) = context.unsourced_summary();
        assert_eq!(edges, 2, "c-r-d is fully sourced and must not be flagged");
        assert!((total - 7.0).abs() < 1e-9, "3.0 + 4.0, got {total}");

        let mut flagged = context.unsourced_edges(0.0, Deadline::unbounded()).unwrap();
        flagged.sort_by(|x, y| x.association.subject.cmp(&y.association.subject));
        assert_eq!(flagged.len(), 2);
        assert_eq!(flagged[0].association.subject, "a");
        assert_eq!(flagged[0].weight, 3.0);
        assert_eq!(flagged[0].count, 1);
        assert_eq!(flagged[1].association.subject, "e");
        assert_eq!(flagged[1].weight, 4.0);
        assert_eq!(flagged[1].count, 1);
    }
    #[test]
    fn unsourced_edges_preserves_sign_and_filters_by_magnitude() {
        let mut context = Context::default();
        context.associate("g", "r", "h", -2.0).unwrap();

        assert!(
            context
                .unsourced_edges(2.5, Deadline::unbounded())
                .unwrap()
                .is_empty(),
            "floor compares by magnitude, 2.0 < 2.5"
        );
        let flagged = context.unsourced_edges(2.0, Deadline::unbounded()).unwrap();
        assert_eq!(flagged.len(), 1);
        assert_eq!(
            flagged[0].weight, -2.0,
            "sign survives the floor comparison"
        );
    }
    #[test]
    fn unsourced_edges_treats_a_nan_floor_as_admitting_everything() {
        let mut context = Context::default();
        context.associate("g", "r", "h", -2.0).unwrap();

        let flagged = context
            .unsourced_edges(f64::NAN, Deadline::unbounded())
            .unwrap();
        assert_eq!(
            flagged.len(),
            1,
            "a NaN floor must not silently read as \"nothing unsourced\""
        );
    }
    #[test]
    fn unsourced_edges_treats_the_reserved_source_as_unattributed() {
        // A round trip (export → import) can hand a live attribution the
        // reserved UNSOURCED_SOURCE id back; it must still count as
        // unsourced, not as a named source that happens to explain the
        // edge.
        let mut context = Context::default();
        context
            .associate_from("i", "r", "j", 5.0, UNSOURCED_SOURCE, None)
            .unwrap();

        let (edges, total) = context.unsourced_summary();
        assert_eq!(edges, 1);
        assert_eq!(total, 5.0);
        let flagged = context.unsourced_edges(0.0, Deadline::unbounded()).unwrap();
        assert_eq!(flagged.len(), 1);
        assert_eq!(flagged[0].weight, 5.0);
    }
}
