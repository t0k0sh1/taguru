use super::{ConceptId, Context, EdgeId};

impl Context {
    /// Facts a concept gloss carries when the caller has no reason to
    /// choose: enough graph context to place the name, few enough that
    /// one new heavy fact does not churn every stored vector. Shared
    /// by the embedding refresh and the resolve response so the text a
    /// caller reads is exactly the text the concept's vector encodes.
    pub const GLOSS_FACTS: usize = 4;
    /// Example triples a label gloss carries, same story.
    pub const GLOSS_EXAMPLES: usize = 3;

    /// A compact textual gloss of one concept: its name followed by its
    /// heaviest facts phrased as minimal sentences, most established
    /// first (ranked by raw cumulative |sum|, not the averaged public
    /// weight, ties in insertion order), negatives phrased as denials.
    /// This is what an external embedding tier
    /// should embed instead of the bare name — a lone word carries too
    /// little signal for sentence-trained embedding models (measured on
    /// text-embedding-3-large: 醸造責任者×杜氏 lands at cosine 0.28
    /// bare but 0.53 glossed) — and the graph already owns the context.
    /// The same text rides along on resolve candidates, where it is
    /// the evidence that tells lookalike names apart (東京都 is not
    /// 京都; their facts are disjoint even when their spellings
    /// overlap). Returns `None` for an unknown concept.
    pub fn concept_gloss(&self, concept: &str, facts: usize) -> Option<String> {
        let &id = self.concept_ids.get(concept)?;
        let edges = self.heaviest(
            self.outgoing(id)
                .chain(self.incoming(id).filter(|&edge_id| {
                    // A self-loop threads BOTH of this concept's chains; kept
                    // from both it would claim two of the `facts` slots for
                    // one fact — the same sentence twice, crowding out
                    // another fact that deserved the second slot.
                    let edge = &self.edges[edge_id as usize];
                    edge.subject != edge.object
                })),
            facts,
        );
        Some(self.gloss_text(self.concept_name(id), &edges, Some(id)))
    }

    /// [`Context::concept_gloss`] for a relation label: the label's name
    /// followed by up to `examples` of the heaviest triples that use it,
    /// so the embedding sees what the relation relates.
    pub fn label_gloss(&self, label: &str, examples: usize) -> Option<String> {
        let &id = self.label_ids.get(label)?;
        let edges = self.heaviest(self.labeled(id), examples);
        Some(self.gloss_text(self.label_name(id), &edges, None))
    }

    /// The `keep` heaviest edges of a chain walk, ranked by the raw
    /// cumulative |sum| descending (not the averaged public weight),
    /// ties toward insertion order.
    fn heaviest(&self, edges: impl Iterator<Item = EdgeId>, keep: usize) -> Vec<EdgeId> {
        // Retracted edges linger in the chain walk with count == 0 and a
        // zeroed sum (retract_association unlinks only the attribution
        // records, not the edge itself). Their |sum| of 0 sorts last, so
        // they surface only when a concept has fewer than `keep` live
        // facts — but then gloss_text renders a withdrawn fact as current.
        // count == 0 is the dead-edge test everywhere else (compacted, the
        // export); apply it here too.
        let mut edges: Vec<EdgeId> = edges
            .filter(|&id| self.edges[id as usize].count > 0)
            .collect();
        edges.sort_by(|&a, &b| {
            self.edges[b as usize]
                .sum
                .abs()
                .total_cmp(&self.edges[a as usize].sum.abs())
                .then_with(|| a.cmp(&b))
        });
        edges.truncate(keep);
        edges
    }

    /// Renders a name plus fact sentences: `名前。AのBはC。…`, negatives
    /// as `…ではない。` — mechanical rather than fluent, which is all an
    /// embedding needs. A fact's subject is dropped when it is `own`
    /// (the gloss's own concept — so a concept with several outgoing
    /// facts states its own name once, not once per sentence) or when
    /// it repeats the previous sentence's subject; otherwise the
    /// subject is stated and becomes the new one to compare against.
    fn gloss_text(&self, name: &str, edges: &[EdgeId], own: Option<ConceptId>) -> String {
        let mut gloss = String::from(name);
        gloss.push('。');
        let mut prev_subject = None;
        for &edge_id in edges {
            let edge = &self.edges[edge_id as usize];
            if Some(edge.subject) != own && Some(edge.subject) != prev_subject {
                gloss.push_str(self.concept_name(edge.subject));
                gloss.push('の');
            }
            prev_subject = Some(edge.subject);
            gloss.push_str(self.label_name(edge.label));
            gloss.push('は');
            gloss.push_str(self.concept_name(edge.object));
            if edge.sum < 0.0 {
                gloss.push_str("ではない");
            }
            gloss.push('。');
        }
        gloss
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glosses_phrase_the_heaviest_context_around_a_name() {
        let mut context = Context::default();
        context.associate("青嶺酒造", "杜氏", "高瀬", 2.0).unwrap();
        context.associate("高瀬", "出身", "南部杜氏", 1.0).unwrap();
        context
            .associate("高瀬", "監督する", "麹造り", 1.0)
            .unwrap();
        context
            .associate("青嶺酒造", "行う", "大量生産", -1.0)
            .unwrap();

        // Heaviest facts first, capped at `facts`; ties in insertion
        // order (出身 before 監督する). The second sentence's subject
        // (高瀬, the gloss's own concept) is dropped.
        assert_eq!(
            context.concept_gloss("高瀬", 2).unwrap(),
            "高瀬。青嶺酒造の杜氏は高瀬。出身は南部杜氏。"
        );
        // A label gloss shows what the relation relates.
        assert_eq!(
            context.label_gloss("杜氏", 3).unwrap(),
            "杜氏。青嶺酒造の杜氏は高瀬。"
        );
        // Negative facts read as denials.
        assert_eq!(
            context.concept_gloss("大量生産", 1).unwrap(),
            "大量生産。青嶺酒造の行うは大量生産ではない。"
        );
        assert!(context.concept_gloss("存在しない概念", 1).is_none());
        assert!(context.label_gloss("存在しない関係", 1).is_none());
    }

    #[test]
    fn one_distinctness_edge_warns_in_both_lookalikes_glosses() {
        // The ingest protocol adjudicates lookalike twins that are NOT
        // the same thing by recording the distinction as an ordinary
        // fact. That advice leans on glosses carrying incoming edges:
        // one directed edge must surface in BOTH names' evidence, even
        // when neither concept has any other fact yet.
        let mut context = Context::default();
        context
            .associate("株式会社青嶺", "別物", "青嶺株式会社", 1.0)
            .unwrap();

        assert_eq!(
            context
                .concept_gloss("株式会社青嶺", Context::GLOSS_FACTS)
                .unwrap(),
            "株式会社青嶺。別物は青嶺株式会社。"
        );
        assert_eq!(
            context
                .concept_gloss("青嶺株式会社", Context::GLOSS_FACTS)
                .unwrap(),
            "青嶺株式会社。株式会社青嶺の別物は青嶺株式会社。"
        );
    }

    #[test]
    fn gloss_states_its_own_name_once_across_mixed_labels_and_polarity() {
        let mut context = Context::default();
        context
            .associate("青嶺酒造", "代表銘柄", "青嶺", 4.0)
            .unwrap();
        context
            .associate("青嶺酒造", "創業年", "1907年", 3.0)
            .unwrap();
        context
            .associate("青嶺酒造", "行う", "大量生産", -2.0)
            .unwrap();

        // Every fact's subject is the gloss's own concept, so it is
        // stated once (as the leading name) and dropped from every
        // sentence after, even across different labels and a negative
        // weight.
        assert_eq!(
            context.concept_gloss("青嶺酒造", 3).unwrap(),
            "青嶺酒造。代表銘柄は青嶺。創業年は1907年。行うは大量生産ではない。"
        );
    }

    #[test]
    fn gloss_does_not_phrase_zero_weight_as_negative() {
        let mut context = Context::default();
        context.associate("A", "関係", "B", 0.0).unwrap();

        assert_eq!(context.concept_gloss("A", 1).unwrap(), "A。関係はB。");
    }

    #[test]
    fn gloss_tracks_the_subject_across_incoming_and_outgoing_facts() {
        let mut context = Context::default();
        context.associate("青嶺酒造", "杜氏", "高瀬", 4.0).unwrap();
        context.associate("青嶺酒造", "顧問", "高瀬", 3.0).unwrap();
        context.associate("高瀬", "出身", "南部杜氏", 2.0).unwrap();
        context.associate("南部酒造", "杜氏", "高瀬", 1.0).unwrap();

        // 青嶺酒造 is stated once and dropped from the immediately
        // following sentence that repeats it; 高瀬 (this gloss's own
        // concept) is dropped outright; 南部酒造 is a new subject, so
        // it is restated even though it shares 高瀬's own name as the
        // object and 杜氏 as the label with the first sentence.
        assert_eq!(
            context.concept_gloss("高瀬", 4).unwrap(),
            "高瀬。青嶺酒造の杜氏は高瀬。顧問は高瀬。出身は南部杜氏。南部酒造の杜氏は高瀬。"
        );
    }

    #[test]
    fn concept_gloss_lists_a_self_loop_once_not_twice() {
        let mut context = Context::default();
        context.associate("蒼月堂", "自称", "蒼月堂", 4.0).unwrap();
        context.associate("蒼月堂", "創業地", "京都", 3.0).unwrap();
        context
            .associate("蒼月堂", "看板商品", "朝霧", 2.0)
            .unwrap();

        // The self-loop threads both the outgoing and incoming chains;
        // with room for exactly three facts, it must surface once, not
        // once per chain — which would double it and crowd out 看板商品.
        assert_eq!(
            context.concept_gloss("蒼月堂", 3).unwrap(),
            "蒼月堂。自称は蒼月堂。創業地は京都。看板商品は朝霧。"
        );
    }

    #[test]
    fn label_gloss_omits_a_repeated_example_subject() {
        let mut context = Context::default();
        context.associate("青嶺酒造", "杜氏", "高瀬", 2.0).unwrap();
        context.associate("青嶺酒造", "杜氏", "山田", 1.0).unwrap();

        assert_eq!(
            context.label_gloss("杜氏", 2).unwrap(),
            "杜氏。青嶺酒造の杜氏は高瀬。杜氏は山田。"
        );
    }
}
