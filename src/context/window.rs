//! Windowed graph reads (ADR 0011): the graph lanes behind an
//! eligibility view — the set of sources whose effective time fell
//! inside the caller's `[since, until)` window. `Context` stays
//! metadata-agnostic by design: the registry computes WHO qualifies
//! (the passage store's `SourceFilter` over `SourceMeta`) and hands
//! the surviving names down; nothing in this module knows what a date
//! is. An edge is visible to a windowed read iff at least one
//! in-window source attests it, and its weight is recomputed from the
//! in-window attributions alone — filtering eligibility but conducting
//! full weight would quietly lie at exactly the margins a temporal
//! query exists to probe (ADR 0011 §4).
//!
//! The unwindowed reads are untouched: every `*_within` method pairs
//! with its existing sibling through a lens the compiler monomorphizes
//! per call site, so the plain lanes compile down to their exact
//! pre-window code — the same reasoning (and the same measured
//! regression risk) `explore_excluding`'s doc records for the
//! `visible` closure.

use std::collections::HashSet;

use super::{
    Activation, Association, Attribution, Context, EdgeId, EdgeRecord, LabelId, Recollection,
    SourceId, accumulate_saturating,
};

/// The eligibility view a windowed read carries: window-qualifying
/// sources, resolved to this context's interned ids once at
/// construction. Names never interned here simply drop out — the same
/// "unknown names contribute nothing" convention `query_any` fixes —
/// so an empty view (no eligible source known to this graph) makes
/// every windowed read answer empty, which is the honest reading of a
/// window nothing falls into.
pub struct SourceWindow {
    eligible: HashSet<SourceId>,
}

/// How a read sees edges: whether an edge is eligible at all, what
/// magnitude it conducts, and how it materializes. The unwindowed
/// lens is the identity (`alive` constant-true, the stored totals);
/// the windowed lens restricts all three to the window's sources.
/// A trait rather than closures because three functions travel
/// together through two walks and a query, and each call site
/// monomorphizes — the unwindowed instantiations fold to the exact
/// pre-lens code.
pub(super) trait EdgeLens {
    fn alive(&self, edge_id: EdgeId, edge: &EdgeRecord) -> bool;
    fn magnitude(&self, edge_id: EdgeId, edge: &EdgeRecord) -> f64;
    fn association(&self, edge_id: EdgeId) -> Association;
}

/// The unwindowed view: every edge as stored. `alive` is
/// unconditionally true — the dead-edge (`count == 0`) test stays
/// where each lane always had it, so `query_any`'s
/// dead-edges-included contract is untouched.
pub(super) struct FullLens<'a> {
    pub(super) context: &'a Context,
}

impl EdgeLens for FullLens<'_> {
    #[inline]
    fn alive(&self, _edge_id: EdgeId, _edge: &EdgeRecord) -> bool {
        true
    }
    #[inline]
    fn magnitude(&self, _edge_id: EdgeId, edge: &EdgeRecord) -> f64 {
        edge.sum.abs()
    }
    fn association(&self, edge_id: EdgeId) -> Association {
        self.context.association(edge_id)
    }
}

/// The windowed view: an edge exists to the degree its in-window
/// sources attest it. `alive` covers the ghost rule for free — a
/// retracted edge's attribution chain is empty, so its windowed count
/// is zero whatever the window holds (ADR 0011 §4 step 3).
pub(super) struct WindowLens<'a> {
    pub(super) context: &'a Context,
    pub(super) window: &'a SourceWindow,
}

impl EdgeLens for WindowLens<'_> {
    fn alive(&self, _edge_id: EdgeId, edge: &EdgeRecord) -> bool {
        self.context.windowed_totals(edge, self.window).0 > 0
    }
    fn magnitude(&self, _edge_id: EdgeId, edge: &EdgeRecord) -> f64 {
        self.context.windowed_totals(edge, self.window).1.abs()
    }
    fn association(&self, edge_id: EdgeId) -> Association {
        self.context.windowed_association(edge_id, self.window)
    }
}

impl Context {
    /// Resolves window-qualifying source names into the view the
    /// `*_within` reads take. The caller owns the temporal judgment —
    /// which sources' effective time falls in the window — and this
    /// resolves spellings, nothing more; the registry is the one
    /// caller today (ADR 0011 §4 step 1-2).
    pub fn source_window<'a>(&self, sources: impl IntoIterator<Item = &'a str>) -> SourceWindow {
        SourceWindow {
            eligible: sources
                .into_iter()
                .filter_map(|name| self.source_ids.get(name).copied())
                .collect(),
        }
    }

    /// One edge's (count, sum) restricted to the window — exact, from
    /// the per-source attribution records (the same per-source figures
    /// compaction preserves "within float re-accumulation error").
    fn windowed_totals(&self, edge: &EdgeRecord, window: &SourceWindow) -> (u64, f64) {
        let mut count = 0u64;
        let mut sum = 0.0f64;
        for (_, record) in self.attribution_chain(edge.first_attribution) {
            if window.eligible.contains(&record.source) {
                count += record.count;
                accumulate_saturating(&mut sum, record.sum);
            }
        }
        (count, sum)
    }

    /// [`Context::association`] restricted to the window: weight and
    /// count from in-window attributions alone, and only those
    /// attributions listed — a citation the window excludes must not
    /// leak back through the edge it once fed.
    fn windowed_association(&self, edge_id: EdgeId, window: &SourceWindow) -> Association {
        let edge = &self.edges[edge_id as usize];
        let (count, sum) = self.windowed_totals(edge, window);
        Association {
            subject: self.concept_name(edge.subject).to_string(),
            label: self.label_name(edge.label).to_string(),
            object: self.concept_name(edge.object).to_string(),
            weight: if count == 0 { 0.0 } else { sum / count as f64 },
            count,
            attributions: self
                .attribution_chain(edge.first_attribution)
                .filter(|(_, record)| window.eligible.contains(&record.source))
                .map(|(attribution_id, record)| Attribution {
                    source: self.source_name(record.source).to_string(),
                    weight: record.sum,
                    count: record.count,
                    paragraph: self.locator_for(attribution_id),
                })
                .collect(),
        }
    }

    /// [`Context::recall`] behind a source window: only associations at
    /// least one in-window source attests, each materialized from its
    /// in-window attributions alone.
    pub fn recall_within(&self, cue: &str, window: &SourceWindow) -> Vec<Association> {
        self.recall_impl(
            cue,
            &WindowLens {
                context: self,
                window,
            },
        )
    }

    /// [`Context::query_any`] behind a source window. Unlike the plain
    /// read — whose dead-edges-included contract predates the window
    /// and stays frozen — a windowed query never returns an edge the
    /// window cannot attest, dead ones included for free.
    pub fn query_any_within(
        &self,
        subjects: &[&str],
        labels: &[&str],
        objects: &[&str],
        window: &SourceWindow,
    ) -> Vec<Association> {
        self.query_any_impl(
            subjects,
            labels,
            objects,
            &WindowLens {
                context: self,
                window,
            },
        )
    }

    /// [`Context::explore_excluding`] behind a source window: an edge
    /// the window cannot attest neither reports nor bridges — the
    /// eligibility test runs inside the walk, so activation and
    /// reachability cannot conduct through out-of-window evidence
    /// (ADR 0011 §5).
    pub fn explore_within(
        &self,
        origins: &[&str],
        max_depth: usize,
        excluded: &[&str],
        window: &SourceWindow,
    ) -> Vec<Recollection> {
        let excluded_ids: HashSet<LabelId> = excluded
            .iter()
            .filter_map(|name| self.label_ids.get(*name).copied())
            .collect();
        self.explore_impl(
            origins,
            max_depth,
            move |label: LabelId| !excluded_ids.contains(&label),
            &WindowLens {
                context: self,
                window,
            },
        )
    }

    /// [`Context::activate_excluding`] behind a source window: both the
    /// fan-normalization total and the conducted magnitude use the
    /// windowed sums, so an edge cannot rank — or leak activation —
    /// on the strength of out-of-window evidence (ADR 0011 §4 step 4).
    pub fn activate_within(
        &self,
        origins: &[&str],
        decay: f64,
        limit: usize,
        excluded: &[&str],
        window: &SourceWindow,
    ) -> (usize, Vec<Activation>) {
        let excluded_ids: HashSet<LabelId> = excluded
            .iter()
            .filter_map(|name| self.label_ids.get(*name).copied())
            .collect();
        self.activate_impl(
            origins,
            decay,
            limit,
            move |label: LabelId| !excluded_ids.contains(&label),
            &WindowLens {
                context: self,
                window,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two dated sources, one edge each, plus one edge they share.
    /// "old" stands for a source whose effective time is in-window,
    /// "new" for one outside it — the lib never sees the dates, only
    /// the registry's verdict, which these tests hand-roll.
    fn corpus() -> Context {
        let mut context = Context::default();
        context
            .associate_from("蔵", "杜氏", "高瀬", 1.0, "doc-old", None)
            .unwrap();
        context
            .associate_from("蔵", "杜氏", "青山", 1.0, "doc-new", None)
            .unwrap();
        // Corroborated by both: the windowed view must keep the edge
        // but re-derive weight/count from the qualifying half alone.
        context
            .associate_from("蔵", "銘柄", "青嶺", 2.0, "doc-old", None)
            .unwrap();
        context
            .associate_from("蔵", "銘柄", "青嶺", 4.0, "doc-new", None)
            .unwrap();
        context
    }

    #[test]
    fn a_window_hides_edges_only_out_of_window_sources_attest() {
        let context = corpus();
        let window = context.source_window(["doc-old"]);

        let hits = context.query_any_within(&["蔵"], &["杜氏"], &[], &window);
        assert_eq!(hits.len(), 1, "the doc-new-only fact must vanish");
        assert_eq!(hits[0].object, "高瀬");

        // The unwindowed read still sees both.
        assert_eq!(context.query_any(&["蔵"], &["杜氏"], &[]).len(), 2);
    }

    #[test]
    fn a_windowed_association_recomputes_weight_and_lists_only_eligible_attributions() {
        let context = corpus();
        let window = context.source_window(["doc-old"]);

        let hits = context.query_any_within(&["蔵"], &["銘柄"], &[], &window);
        assert_eq!(hits.len(), 1);
        let hit = &hits[0];
        assert_eq!(
            hit.weight, 2.0,
            "windowed weight is the in-window sum/count"
        );
        assert_eq!(hit.count, 1);
        assert_eq!(hit.attributions.len(), 1);
        assert_eq!(hit.attributions[0].source, "doc-old");

        // Unwindowed, the same edge reports the accumulated view.
        let full = context.query_any(&["蔵"], &["銘柄"], &[]);
        assert_eq!(full[0].weight, 3.0);
        assert_eq!(full[0].count, 2);
        assert_eq!(full[0].attributions.len(), 2);
    }

    #[test]
    fn recall_within_filters_by_the_window() {
        let context = corpus();
        let window = context.source_window(["doc-old"]);
        let hits = context.recall_within("蔵", &window);
        assert_eq!(hits.len(), 2, "杜氏→高瀬 and 銘柄→青嶺");
        assert!(hits.iter().all(|hit| {
            hit.attributions
                .iter()
                .all(|attribution| attribution.source == "doc-old")
        }));
    }

    #[test]
    fn windowed_walks_do_not_bridge_through_out_of_window_edges() {
        let mut context = Context::default();
        // a --(doc-new)--> b --(doc-old)--> c: with only doc-old
        // eligible, the first hop is invisible, so nothing beyond it
        // may be reached — eligibility must gate the walk itself, not
        // post-filter its results.
        context
            .associate_from("a", "r", "b", 1.0, "doc-new", None)
            .unwrap();
        context
            .associate_from("b", "r", "c", 1.0, "doc-old", None)
            .unwrap();
        let window = context.source_window(["doc-old"]);

        let reached = context.explore_within(&["a"], Context::UNBOUNDED, &[], &window);
        assert!(
            reached.is_empty(),
            "a's only edge is out-of-window; nothing is reachable: {reached:?}"
        );

        let (total, matches) = context.activate_within(&["a"], 1.0, 10, &[], &window);
        assert_eq!(total, 0, "activation must not conduct either: {matches:?}");

        // From b the in-window half of the graph is fully visible.
        let (total, matches) = context.activate_within(&["b"], 1.0, 10, &[], &window);
        assert_eq!(total, 1);
        assert_eq!(matches[0].association.object, "c");
    }

    #[test]
    fn windowed_activation_ranks_on_windowed_sums() {
        let mut context = Context::default();
        // Same anchor, two facts: the one whose IN-WINDOW evidence is
        // stronger must outrank, even though the other's accumulated
        // sum is larger overall.
        context
            .associate_from("蔵", "銘柄", "青嶺", 3.0, "doc-old", None)
            .unwrap();
        context
            .associate_from("蔵", "杜氏", "高瀬", 1.0, "doc-old", None)
            .unwrap();
        context
            .associate_from("蔵", "杜氏", "高瀬", 9.0, "doc-new", None)
            .unwrap();
        let window = context.source_window(["doc-old"]);

        let (_, windowed) = context.activate_within(&["蔵"], 1.0, 10, &[], &window);
        assert_eq!(
            windowed[0].association.label, "銘柄",
            "3.0 beats 1.0 in-window"
        );

        let (_, full) = context.activate(&["蔵"], 1.0, 10);
        assert_eq!(full[0].association.label, "杜氏", "10.0 beats 3.0 overall");
    }

    #[test]
    fn a_retracted_edge_stays_invisible_even_when_its_source_is_eligible() {
        let mut context = Context::default();
        context
            .associate_from("蔵", "杜氏", "高瀬", 1.0, "doc-old", None)
            .unwrap();
        context.retract_source("doc-old");
        let window = context.source_window(["doc-old"]);
        assert!(
            context
                .query_any_within(&["蔵"], &[], &[], &window)
                .is_empty(),
            "the ghost rule composes: an empty attribution chain has no in-window sources"
        );
    }

    #[test]
    fn an_empty_window_answers_empty_everywhere() {
        let context = corpus();
        let window = context.source_window([]);
        assert!(context.recall_within("蔵", &window).is_empty());
        assert!(context.query_any_within(&[], &[], &[], &window).is_empty());
        let (total, _) = context.activate_within(&["蔵"], 1.0, 10, &[], &window);
        assert_eq!(total, 0);
    }
}
