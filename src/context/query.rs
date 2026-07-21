use std::collections::{HashMap, HashSet};

use super::{
    Association, ConceptDescription, Context, EdgeFollow, EdgeId, LabelId, LabelUsage,
    keep_narrowest_anchor,
};

impl Context {
    /// Recalls every association touching `cue`, whether it appears as the
    /// subject, the relation label, or the object. This lets a relation
    /// label (e.g. "好き") act as a search cue in its own right, not just a
    /// concept. `recall` cannot tell which position matched — use `query`
    /// when the role of the cue matters (e.g. "私が好きなもの" vs "私を好き
    /// な人"). Results come back in insertion order.
    pub fn recall(&self, cue: &str) -> Vec<Association> {
        let concept_edges = self
            .concept_ids
            .get(cue)
            .map(|&id| self.outgoing(id).chain(self.incoming(id)));
        let label_edges = self.label_ids.get(cue).map(|&id| self.labeled(id));

        let mut edge_ids: Vec<EdgeId> = concept_edges
            .into_iter()
            .flatten()
            .chain(label_edges.into_iter().flatten())
            .collect();
        edge_ids.sort_unstable();
        edge_ids.dedup();

        edge_ids
            .into_iter()
            .map(|edge_id| self.association(edge_id))
            .collect()
    }

    /// Recalls associations matching a fixed pattern: each `Some` position
    /// pins that exact subject/label/object, each `None` leaves it
    /// unconstrained. This is what tells apart queries `recall` cannot
    /// express on its own — "私が好きなもの" is
    /// `query(Some("私"), Some("好き"), None)`, "りんごを好きな人" is
    /// `query(None, Some("好き"), Some("りんご"))`, and "好きと思われている
    /// もの全部" is `query(None, Some("好き"), None)` — they differ only in
    /// which position is pinned. Results come back in insertion order.
    pub fn query(
        &self,
        subject: Option<&str>,
        label: Option<&str>,
        object: Option<&str>,
    ) -> Vec<Association> {
        self.query_any(subject.as_slice(), label.as_slice(), object.as_slice())
    }

    /// [`Context::query`] with OR-sets: each non-empty list pins its
    /// position to ANY of the given names, each empty list leaves the
    /// position unconstrained. This is the "narrowed re-query" of the
    /// two-step read pattern — [`Context::describe`] shows which labels a
    /// concept carries, then `query_any(&["山田太郎"], &["住所", "職歴"],
    /// &[])` fetches just the chosen facets instead of the whole profile.
    ///
    /// Names that were never interned contribute nothing; a constrained
    /// position whose names are ALL unknown can match nothing at all.
    /// Results come back in insertion order.
    pub fn query_any(
        &self,
        subjects: &[&str],
        labels: &[&str],
        objects: &[&str],
    ) -> Vec<Association> {
        if subjects.is_empty() && labels.is_empty() && objects.is_empty() {
            return (0..self.edges.len() as u32)
                .map(|edge_id| self.association(edge_id))
                .collect();
        }

        // Resolve every constrained position to its id set up front; a
        // constrained position with no known names can match nothing.
        let resolve_set = |map: &HashMap<String, u32>, names: &[&str]| -> Option<HashSet<u32>> {
            (!names.is_empty()).then(|| {
                names
                    .iter()
                    .filter_map(|name| map.get(*name).copied())
                    .collect()
            })
        };
        let subject_ids = resolve_set(&self.concept_ids, subjects);
        let label_ids = resolve_set(&self.label_ids, labels);
        let object_ids = resolve_set(&self.concept_ids, objects);
        for constrained in [&subject_ids, &label_ids, &object_ids]
            .into_iter()
            .flatten()
        {
            if constrained.is_empty() {
                return Vec::new();
            }
        }

        // Narrow the scan with whichever constrained position anchors the
        // fewest chained edges in total (each record carries its chain's
        // length), then filter the remaining constraints in memory; this
        // walks a few small chains instead of every edge.
        let mut narrowest: Option<(u64, Vec<EdgeId>, EdgeFollow)> = None;
        if let Some(ids) = &subject_ids {
            keep_narrowest_anchor(
                &mut narrowest,
                ids.iter()
                    .map(|&id| u64::from(self.concepts[id as usize].outgoing_count))
                    .sum(),
                ids.iter()
                    .map(|&id| self.concepts[id as usize].first_outgoing)
                    .collect(),
                |edge| edge.next_outgoing,
            );
        }
        if let Some(ids) = &label_ids {
            keep_narrowest_anchor(
                &mut narrowest,
                ids.iter()
                    .map(|&id| u64::from(self.labels[id as usize].edge_count))
                    .sum(),
                ids.iter()
                    .map(|&id| self.labels[id as usize].first_edge)
                    .collect(),
                |edge| edge.next_labeled,
            );
        }
        if let Some(ids) = &object_ids {
            keep_narrowest_anchor(
                &mut narrowest,
                ids.iter()
                    .map(|&id| u64::from(self.concepts[id as usize].incoming_count))
                    .sum(),
                ids.iter()
                    .map(|&id| self.concepts[id as usize].first_incoming)
                    .collect(),
                |edge| edge.next_incoming,
            );
        }
        let Some((_, firsts, follow)) = narrowest else {
            return Vec::new();
        };

        // Chains of one position never share an edge, but sorting restores
        // global insertion order across the walked chains.
        let mut edge_ids: Vec<EdgeId> = firsts
            .into_iter()
            .flat_map(|first| self.edge_chain(first, follow))
            .collect();
        edge_ids.sort_unstable();

        edge_ids
            .into_iter()
            .filter(|&edge_id| {
                let edge = &self.edges[edge_id as usize];
                subject_ids
                    .as_ref()
                    .is_none_or(|ids| ids.contains(&edge.subject))
                    && label_ids
                        .as_ref()
                        .is_none_or(|ids| ids.contains(&edge.label))
                    && object_ids
                        .as_ref()
                        .is_none_or(|ids| ids.contains(&edge.object))
            })
            .map(|edge_id| self.association(edge_id))
            .collect()
    }

    /// The outline of one concept: which relation labels its edges carry
    /// and how often, split by role, most frequent first (ties in label
    /// insertion order). Returns `None` for an unknown concept.
    ///
    /// This is the cheap first step of a staged read — a caller checks
    /// what KINDS of knowledge exist about a concept (O(degree), no
    /// association materialized), then fetches only the relevant labels
    /// via [`Context::query_any`].
    pub fn describe(&self, concept: &str) -> Option<ConceptDescription> {
        let &id = self.concept_ids.get(concept)?;
        let tally = |edges: &mut dyn Iterator<Item = EdgeId>| -> Vec<LabelUsage> {
            let mut counts: HashMap<LabelId, usize> = HashMap::new();
            for edge_id in edges {
                let edge = &self.edges[edge_id as usize];
                // Skip retracted edges (count == 0): describe reports how
                // often a label is used by LIVE facts, so a withdrawn one
                // must not inflate the tally. Same dead-edge test as
                // heaviest/compacted/the export.
                if edge.count == 0 {
                    continue;
                }
                *counts.entry(edge.label).or_insert(0) += 1;
            }
            let mut usages: Vec<(LabelId, usize)> = counts.into_iter().collect();
            usages.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            usages
                .into_iter()
                .map(|(label_id, count)| LabelUsage {
                    label: self.label_name(label_id).to_string(),
                    count,
                })
                .collect()
        };
        Some(ConceptDescription {
            concept: self.concept_name(id).to_string(),
            as_subject: tally(&mut self.outgoing(id)),
            as_object: tally(&mut self.incoming(id)),
        })
    }

    /// The full relation-label vocabulary in insertion order — the
    /// governance view an ingester consults (e.g. pasted into its
    /// extraction prompt) so relation spellings stay consistent across
    /// documents instead of forking per document.
    pub fn labels(&self) -> Vec<&str> {
        self.labels
            .iter()
            .map(|record| self.arena_str(record.name_offset, record.name_len))
            .collect()
    }

    /// One name-ordered page of the relation-label vocabulary plus the
    /// cursor-independent total, seeked in O(log n + k) against
    /// [`Context::label_name_index`] instead of collecting and sorting
    /// the whole vocabulary on every call — the paged sibling of
    /// [`Context::labels`].
    pub fn label_page(&self, after: Option<&str>, limit: usize) -> (usize, Vec<String>) {
        use std::ops::Bound;

        let start = match after {
            Some(after) => Bound::Excluded(after),
            None => Bound::Unbounded,
        };
        let page = self
            .label_name_index
            .range::<str, _>((start, Bound::Unbounded))
            .take(limit)
            .cloned()
            .collect();
        (self.label_name_index.len(), page)
    }

    /// Every canonical concept spelling in insertion order — the
    /// vocabulary an external entry tier (e.g. an embedding index over
    /// names) enumerates to stay in sync with the network.
    pub fn concept_names(&self) -> Vec<&str> {
        self.concepts
            .iter()
            .map(|record| self.arena_str(record.name_offset, record.name_len))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::test_support::{assoc, associate_examples};

    #[test]
    fn recalls_associations_by_any_cue_word() {
        let mut context = Context::default();
        associate_examples(&mut context);

        assert_eq!(context.recall("私").len(), 3);
        assert_eq!(context.recall("好き").len(), 3);

        let by_object = context.recall("りんご");
        assert_eq!(by_object, vec![assoc("私", "好き", "りんご", 1.0)]);

        assert!(context.recall("存在しない単語").is_empty());
    }
    #[test]
    fn positional_query_pins_the_role_that_recall_conflates() {
        let mut context = Context::default();

        // 私はりんごが好きです (私 is the subject here)
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        // りんごは私が好きです (私 is the object here instead)
        context.associate("りんご", "好き", "私", 1.0).unwrap();

        assert_eq!(
            context.query(Some("私"), None, None),
            vec![assoc("私", "好き", "りんご", 1.0)]
        );
        assert_eq!(
            context.query(None, None, Some("私")),
            vec![assoc("りんご", "好き", "私", 1.0)]
        );

        // recall cannot distinguish the two roles above; it reports both.
        assert_eq!(context.recall("私").len(), 2);
    }
    #[test]
    fn positional_query_combines_label_with_subject_or_object() {
        let mut context = Context::default();
        associate_examples(&mut context);

        // 私が好きなもの
        let liked_by_me = context.query(Some("私"), Some("好き"), None);
        assert_eq!(liked_by_me.len(), 3);
        assert!(liked_by_me.contains(&assoc("私", "好き", "りんご", 1.0)));

        // りんごを好きな人
        assert_eq!(
            context.query(None, Some("好き"), Some("りんご")),
            vec![assoc("私", "好き", "りんご", 1.0)]
        );

        // 好きと思われているもの全部
        assert_eq!(context.query(None, Some("好き"), None).len(), 3);
    }
    #[test]
    fn query_with_no_constraints_returns_everything() {
        let mut context = Context::default();
        associate_examples(&mut context);

        assert_eq!(context.query(None, None, None).len(), 3);
    }
    #[test]
    fn query_returns_nothing_for_an_unknown_bound_value() {
        let mut context = Context::default();
        associate_examples(&mut context);

        assert!(context.query(Some("存在しない概念"), None, None).is_empty());
        assert!(context.query(None, Some("存在しない関係"), None).is_empty());
    }
    #[test]
    fn distinct_relation_labels_between_the_same_pair_stay_independent() {
        let mut context = Context::default();

        // 私はりんごが好きです / 私はりんごを食べられません — two distinct
        // labels between the same (subject, object) pair.
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        context
            .associate("私", "食べられる", "りんご", -1.0)
            .unwrap();

        let between_me_and_ringo = context.query(Some("私"), None, Some("りんご"));
        assert_eq!(between_me_and_ringo.len(), 2);
        assert!(between_me_and_ringo.contains(&assoc("私", "好き", "りんご", 1.0)));
        assert!(between_me_and_ringo.contains(&assoc("私", "食べられる", "りんご", -1.0)));
    }
    #[test]
    fn label_page_seeks_a_page_instead_of_resorting_the_whole_vocabulary() {
        let mut context = Context::default();
        // Interned out of alphabetical order — proves `label_page` seeks
        // its own sorted index rather than replaying insertion order.
        context.associate("蔵", "founded", "1907", 1.0).unwrap();
        context.associate("蔵", "brand", "青嶺", 1.0).unwrap();
        context.associate("蔵", "location", "霧沢町", 1.0).unwrap();
        context.associate("蔵", "brewer", "高瀬", 1.0).unwrap();

        assert_eq!(context.label_count(), 4);
        let (total, first) = context.label_page(None, 2);
        assert_eq!(total, 4);
        assert_eq!(first, vec!["brand".to_string(), "brewer".to_string()]);

        let (total, second) = context.label_page(Some("brewer"), 2);
        assert_eq!(total, 4, "total stays constant across pages");
        assert_eq!(second, vec!["founded".to_string(), "location".to_string()]);

        let (_, exhausted) = context.label_page(Some("location"), 2);
        assert!(exhausted.is_empty());
    }
    #[test]
    fn labels_lists_the_relation_vocabulary_in_insertion_order() {
        let mut context = Context::default();
        context.associate("a", "r2", "b", 1.0).unwrap();
        context.associate("b", "r1", "c", 1.0).unwrap();
        context.associate("c", "r2", "d", 1.0).unwrap(); // reuse must not duplicate

        assert_eq!(context.labels(), vec!["r2", "r1"]);
    }

    /// Builds a small profile: 山田太郎 with two addresses' worth of
    /// facts, three career entries, and one incoming recommendation.
    fn associate_profile(context: &mut Context) {
        context.associate("山田太郎", "住所", "東京", 1.0).unwrap();
        context
            .associate("山田太郎", "職歴", "営業部", 1.0)
            .unwrap();
        context
            .associate("山田太郎", "職歴", "開発部", 1.0)
            .unwrap();
        context
            .associate("山田太郎", "職歴", "企画部", 1.0)
            .unwrap();
        context
            .associate("山田太郎", "アピールポイント", "英語", 1.0)
            .unwrap();
        context
            .associate("佐藤", "推薦する", "山田太郎", 1.0)
            .unwrap();
    }

    #[test]
    fn describe_outlines_a_concept_without_materializing_facts() {
        let mut context = Context::default();
        associate_profile(&mut context);

        let description = context.describe("山田太郎").unwrap();
        assert_eq!(description.concept, "山田太郎");
        // Most frequent first; ties (住所/アピールポイント, 1 each) in
        // label insertion order.
        assert_eq!(
            description.as_subject,
            vec![
                LabelUsage {
                    label: "職歴".to_string(),
                    count: 3,
                },
                LabelUsage {
                    label: "住所".to_string(),
                    count: 1,
                },
                LabelUsage {
                    label: "アピールポイント".to_string(),
                    count: 1,
                },
            ]
        );
        assert_eq!(
            description.as_object,
            vec![LabelUsage {
                label: "推薦する".to_string(),
                count: 1,
            }]
        );

        assert!(context.describe("存在しない概念").is_none());
    }
    #[test]
    fn query_any_pins_a_position_to_any_of_several_names() {
        let mut context = Context::default();
        associate_profile(&mut context);

        // The staged read: describe showed 職歴/住所/アピールポイント;
        // fetch just two of those facets.
        let narrowed = context.query_any(&["山田太郎"], &["住所", "アピールポイント"], &[]);
        assert_eq!(narrowed.len(), 2);
        assert_eq!(narrowed[0].label, "住所"); // insertion order
        assert_eq!(narrowed[1].label, "アピールポイント");

        // Unknown names inside a set contribute nothing but do not
        // poison the rest; an all-unknown set can match nothing.
        assert_eq!(
            context
                .query_any(&["山田太郎"], &["住所", "存在しない関係"], &[])
                .len(),
            1
        );
        assert!(
            context
                .query_any(&["山田太郎"], &["存在しない関係"], &[])
                .is_empty()
        );

        // Multiple subjects at once, and parity with the single-name query.
        assert_eq!(context.query_any(&["山田太郎", "佐藤"], &[], &[]).len(), 6);
        assert_eq!(
            context.query_any(&["山田太郎"], &["職歴"], &[]),
            context.query(Some("山田太郎"), Some("職歴"), None)
        );

        // All positions unconstrained returns everything.
        assert_eq!(context.query_any(&[], &[], &[]).len(), 6);
    }
}
