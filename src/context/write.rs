use crate::deadline::Deadline;

use super::{
    AliasError, AttributionLocatorRecord, AttributionRecord, CompactionError, CompactionStats,
    Context, ContextFull, EdgeId, EdgeRecord, NIL, SourceId, accumulate_saturating,
    append_to_chain, arena_fits, claim_id, ids_left,
};

impl Context {
    /// Adds a directed, labeled, signed edge from `subject` to `object`, or
    /// folds `weight` into an edge already there for the same (subject,
    /// label, object) key.
    ///
    /// This and [`Context::associate_from`] are the only ways facts enter a
    /// `Context`. It takes an already-resolved subject, label, object, and
    /// signed weight — no tokenizing, part-of-speech tagging, or sentence
    /// structure of any language. Reducing a natural-language utterance
    /// (e.g. "私はりんごが好きですが、たくさんは食べられません") down to
    /// that shape is entirely the caller's job; `Context` only ever sees
    /// the result, as two independent calls:
    /// `associate("私", "好き", "りんご", 1.0)` and
    /// `associate("私", "食べられる", "りんご", -0.2)`.
    ///
    /// The first time a (subject, label, object) key is seen, `weight`
    /// seeds its running sum and [`Association::count`] becomes 1. Every
    /// later call for the same key folds `weight` into that sum and
    /// increments `count`; the `weight` a caller sees back is always the
    /// average `sum / count`. Repeated agreeing evidence of the same
    /// magnitude therefore leaves the average unchanged — corroboration
    /// reads as more `count` behind the same weight, not a bigger one —
    /// while contradicting evidence nets against the sum before it is
    /// averaged, and strong enough contradiction can still overturn its
    /// sign. Nothing about the mechanism treats agreement and contradiction
    /// differently; both are just addition to the underlying sum.
    ///
    /// # Panics
    ///
    /// Panics when `weight` is not finite. A NaN sum would own the top
    /// gloss slot forever (`f64::total_cmp` sorts it as the maximum)
    /// and the image would stop round-tripping — `from_bytes` rejects
    /// non-finite sums as corruption — so the write boundary refuses
    /// the value outright, exactly as the HTTP layer does before ever
    /// calling in.
    ///
    /// # Errors
    ///
    /// Returns [`ContextFull`] when the write would need a record or name
    /// bytes beyond the context's u32 id/offset space; the context is
    /// left unchanged.
    pub fn associate(
        &mut self,
        subject: impl Into<String>,
        label: impl Into<String>,
        object: impl Into<String>,
        weight: f64,
    ) -> Result<(), ContextFull> {
        self.upsert(
            subject.into(),
            label.into(),
            object.into(),
            weight,
            None,
            None,
        )
    }

    /// Like [`Context::associate`], but records which source asserted the
    /// fact. The same sum/count accumulation applies to the edge's total;
    /// in addition, the contribution is tallied per source, so a retrieved
    /// [`Association`] can show whether its weight came from many
    /// independent sources (corroboration) or one source repeating itself,
    /// and the caller can follow any attribution back to the original text.
    ///
    /// Discipline note: paraphrases of one fact inside one document should
    /// still NOT be re-asserted — same-magnitude re-assertions leave the
    /// public `weight` unchanged (it is an average, not a pile-up), but each
    /// one still inflates `count`, and its source's own attribution count,
    /// without adding independent evidence. Assert once per document;
    /// re-assert across documents (real corroboration). Readers that care
    /// about independence should still look at `count` and the attribution
    /// list rather than trust weight alone: same-source repetition and
    /// genuine multi-source corroboration of equal magnitude average out to
    /// the same weight.
    ///
    /// `paragraph` optionally locates the fact within `source` (e.g. a
    /// paragraph index) and is first-write-wins: it is only recorded the
    /// first time this source is attributed to this edge, so a later
    /// re-assertion of the same source cannot change where the first one
    /// pointed.
    ///
    /// # Panics
    ///
    /// Panics on a non-finite `weight`, as [`Context::associate`] does.
    ///
    /// # Errors
    ///
    /// Returns [`ContextFull`] exactly as [`Context::associate`] does; a
    /// sourced write may additionally need room for one attribution
    /// record.
    pub fn associate_from(
        &mut self,
        subject: impl Into<String>,
        label: impl Into<String>,
        object: impl Into<String>,
        weight: f64,
        source: impl Into<String>,
        paragraph: Option<u32>,
    ) -> Result<(), ContextFull> {
        self.upsert(
            subject.into(),
            label.into(),
            object.into(),
            weight,
            Some(source.into()),
            paragraph,
        )
    }

    fn upsert(
        &mut self,
        subject: String,
        label: String,
        object: String,
        weight: f64,
        source: Option<String>,
        paragraph: Option<u32>,
    ) -> Result<(), ContextFull> {
        // A non-finite weight is a contract violation, not data: one NaN
        // sum sorts as the maximum under `total_cmp` and owns the top
        // gloss slot forever, and `from_bytes` (rightly) refuses the
        // resulting image as corrupt. The HTTP and import layers
        // validate before calling in; a library caller meets the same
        // rule here. (WAL replay cannot reach this: JSON has no literal
        // for NaN/Infinity and serde_json rejects out-of-range floats.)
        assert!(
            weight.is_finite(),
            "association weight must be finite, got {weight}"
        );
        self.ensure_room(&subject, &label, &object, source.as_deref())?;

        // Infallible from here on: every record and arena byte this write
        // can need has been checked above.
        let subject_id = self.intern_concept(subject);
        let object_id = self.intern_concept(object);
        let label_id = self.intern_label(label);
        let source_id = source.map(|name| self.intern_source(name));

        let key = (subject_id, label_id, object_id);
        let edge_id = match self.edge_ids.get(&key).copied() {
            Some(edge_id) => {
                let edge = &mut self.edges[edge_id as usize];
                // `edge_ids` is never pruned, so a triple that died
                // (count fell to zero) can still be found here and
                // revived by a fresh assertion — dead_edges must track
                // that revival, not just deaths, to stay a live count.
                let was_dead = edge.count == 0;
                accumulate_saturating(&mut edge.sum, weight);
                edge.count += 1;
                if was_dead {
                    self.dead_edges -= 1;
                }
                edge_id
            }
            None => {
                let edge_id = claim_id(self.edges.len(), "edge");
                self.edge_ids.insert(key, edge_id);
                self.edges.push(EdgeRecord {
                    subject: subject_id,
                    label: label_id,
                    object: object_id,
                    next_outgoing: NIL,
                    next_incoming: NIL,
                    next_labeled: NIL,
                    first_attribution: NIL,
                    last_attribution: NIL,
                    count: 1,
                    sum: weight,
                });
                let node = &mut self.concepts[subject_id as usize];
                append_to_chain(
                    &mut self.edges,
                    &mut node.first_outgoing,
                    &mut node.last_outgoing,
                    &mut node.outgoing_count,
                    edge_id,
                    |edge| &mut edge.next_outgoing,
                );
                let node = &mut self.concepts[object_id as usize];
                append_to_chain(
                    &mut self.edges,
                    &mut node.first_incoming,
                    &mut node.last_incoming,
                    &mut node.incoming_count,
                    edge_id,
                    |edge| &mut edge.next_incoming,
                );
                let label = &mut self.labels[label_id as usize];
                append_to_chain(
                    &mut self.edges,
                    &mut label.first_edge,
                    &mut label.last_edge,
                    &mut label.edge_count,
                    edge_id,
                    |edge| &mut edge.next_labeled,
                );
                edge_id
            }
        };

        if let Some(source_id) = source_id {
            self.attribute(edge_id, source_id, weight, paragraph);
        }
        Ok(())
    }

    /// Pre-flight for one `upsert`: verifies, without mutating anything,
    /// that every record and arena byte the write could need still fits
    /// below the u32 id/offset ceilings. Checking everything up front is
    /// what makes a capacity failure all-or-nothing — an `Err` leaves the
    /// context exactly as it was.
    // Exercising these branches through a Context would require allocating
    // u32::MAX records or 4 GiB of interned text. The pure ceiling helpers
    // and claim_id backstop are tested directly at their exact boundaries.
    #[mutants::skip]
    fn ensure_room(
        &self,
        subject: &str,
        label: &str,
        object: &str,
        source: Option<&str>,
    ) -> Result<(), ContextFull> {
        let subject_new = !self.concept_ids.contains_key(subject);
        let object_new = object != subject && !self.concept_ids.contains_key(object);
        let label_new = !self.label_ids.contains_key(label);
        let source_new = source.is_some_and(|name| !self.source_ids.contains_key(name));

        let new_concepts = usize::from(subject_new) + usize::from(object_new);
        if !ids_left(self.concepts.len(), new_concepts) {
            return Err(ContextFull("the concept table is out of u32 ids"));
        }
        if label_new && !ids_left(self.labels.len(), 1) {
            return Err(ContextFull("the label table is out of u32 ids"));
        }
        if source_new && !ids_left(self.sources.len(), 1) {
            return Err(ContextFull("the source table is out of u32 ids"));
        }

        // The exact triple can only already exist if all three names do.
        let existing_edge = match (
            self.concept_ids.get(subject),
            self.label_ids.get(label),
            self.concept_ids.get(object),
        ) {
            (Some(&subject_id), Some(&label_id), Some(&object_id)) => self
                .edge_ids
                .get(&(subject_id, label_id, object_id))
                .copied(),
            _ => None,
        };
        if existing_edge.is_none() && !ids_left(self.edges.len(), 1) {
            return Err(ContextFull("the edge table is out of u32 ids"));
        }

        // A sourced write appends an attribution record unless this source
        // already attributes this exact edge — an O(1) index hit, not a
        // walk of the edge's attribution chain.
        if let Some(name) = source {
            let already_attributed = existing_edge.is_some_and(|edge_id| {
                self.source_ids.get(name).is_some_and(|&source_id| {
                    self.attribution_ids.contains_key(&(edge_id, source_id))
                })
            });
            if !already_attributed && !ids_left(self.attributions.len(), 1) {
                return Err(ContextFull("the attribution table is out of u32 ids"));
            }
        }

        let mut arena_growth = 0usize;
        if subject_new {
            arena_growth += subject.len();
        }
        if object_new {
            arena_growth += object.len();
        }
        if label_new {
            arena_growth += label.len();
        }
        if source_new {
            arena_growth += source.map_or(0, str::len);
        }
        if !arena_fits(self.arena.len(), arena_growth) {
            return Err(ContextFull("the string arena is out of offset space"));
        }
        Ok(())
    }

    /// Adds `weight` onto the edge's existing attribution for `source_id`,
    /// or appends a new one at the chain's tail — one record per distinct
    /// source, however often it re-asserts, in first-assertion order.
    /// `paragraph` locates the new record within its source and is only
    /// ever used here, on creation: a weight-only merge into an existing
    /// record never touches its locator (first-write-wins).
    fn attribute(
        &mut self,
        edge_id: EdgeId,
        source_id: SourceId,
        weight: f64,
        paragraph: Option<u32>,
    ) {
        // The (edge, source) index finds this source's existing record in
        // O(1); folding into it leaves the chain and the locator untouched.
        if let Some(&existing) = self.attribution_ids.get(&(edge_id, source_id)) {
            let record = &mut self.attributions[existing as usize];
            accumulate_saturating(&mut record.sum, weight);
            record.count += 1;
            return;
        }

        let attribution_id = claim_id(self.attributions.len(), "attribution");
        self.attributions.push(AttributionRecord {
            source: source_id,
            next: NIL,
            count: 1,
            sum: weight,
        });
        self.attribution_ids
            .insert((edge_id, source_id), attribution_id);
        self.source_edges
            .entry(source_id)
            .or_default()
            .push(edge_id);
        if let Some(paragraph) = paragraph {
            self.attribution_locators.push(AttributionLocatorRecord {
                attribution: attribution_id,
                paragraph,
            });
        }
        let edge = &mut self.edges[edge_id as usize];
        let tail = edge.last_attribution;
        if tail == NIL {
            edge.first_attribution = attribution_id;
        }
        edge.last_attribution = attribution_id;
        if tail != NIL {
            self.attributions[tail as usize].next = attribution_id;
        }
    }

    /// Installs one brand-new edge at its FINAL totals in one shot —
    /// the compaction primitive. Where `upsert` folds a single
    /// assertion (count += 1) and so costs one call per historical
    /// assertion, this sets the edge's `count`/`sum` and each
    /// attribution's `count`/`sum`/locator directly, so rebuilding a
    /// context is O(distinct edges + distinct attributions), not
    /// O(total assertions) — a heavily corroborated edge no longer
    /// turns compaction into millions of redundant interning probes.
    ///
    /// The caller ([`Context::compacted`]) builds a subset of an
    /// already-valid image, so capacity cannot be exceeded; the
    /// `ensure_room` call is the capacity guard that also produces the
    /// `ContextFull` arm the signature promises. `attributions` carries
    /// `(source, sum, count, paragraph)` per distinct source; the edge
    /// total may exceed their sum when sourceless weight rode in.
    fn install_edge(
        &mut self,
        subject: &str,
        label: &str,
        object: &str,
        edge_count: u64,
        edge_sum: f64,
        attributions: &[(String, f64, u64, Option<u32>)],
    ) -> Result<(), ContextFull> {
        debug_assert!(edge_sum.is_finite(), "compaction must carry finite sums");
        // One room check covers the concepts, label, edge, and the
        // first source; every later source is one already present in the
        // superset this context is compacting, so the arena and id
        // ceilings it needs are provably already cleared.
        let first_source = attributions.first().map(|(source, ..)| source.as_str());
        self.ensure_room(subject, label, object, first_source)?;

        let subject_id = self.intern_concept(subject.to_string());
        let object_id = self.intern_concept(object.to_string());
        let label_id = self.intern_label(label.to_string());

        let edge_id = claim_id(self.edges.len(), "edge");
        self.edge_ids
            .insert((subject_id, label_id, object_id), edge_id);
        self.edges.push(EdgeRecord {
            subject: subject_id,
            label: label_id,
            object: object_id,
            next_outgoing: NIL,
            next_incoming: NIL,
            next_labeled: NIL,
            first_attribution: NIL,
            last_attribution: NIL,
            count: edge_count,
            sum: edge_sum,
        });
        let node = &mut self.concepts[subject_id as usize];
        append_to_chain(
            &mut self.edges,
            &mut node.first_outgoing,
            &mut node.last_outgoing,
            &mut node.outgoing_count,
            edge_id,
            |edge| &mut edge.next_outgoing,
        );
        let node = &mut self.concepts[object_id as usize];
        append_to_chain(
            &mut self.edges,
            &mut node.first_incoming,
            &mut node.last_incoming,
            &mut node.incoming_count,
            edge_id,
            |edge| &mut edge.next_incoming,
        );
        let label_rec = &mut self.labels[label_id as usize];
        append_to_chain(
            &mut self.edges,
            &mut label_rec.first_edge,
            &mut label_rec.last_edge,
            &mut label_rec.edge_count,
            edge_id,
            |edge| &mut edge.next_labeled,
        );

        for (source, sum, count, paragraph) in attributions {
            let source_id = self.intern_source(source.clone());
            let attribution_id = claim_id(self.attributions.len(), "attribution");
            self.attributions.push(AttributionRecord {
                source: source_id,
                next: NIL,
                count: *count,
                sum: *sum,
            });
            self.attribution_ids
                .insert((edge_id, source_id), attribution_id);
            self.source_edges
                .entry(source_id)
                .or_default()
                .push(edge_id);
            if let Some(paragraph) = paragraph {
                self.attribution_locators.push(AttributionLocatorRecord {
                    attribution: attribution_id,
                    paragraph: *paragraph,
                });
            }
            let edge = &mut self.edges[edge_id as usize];
            let tail = edge.last_attribution;
            if tail == NIL {
                edge.first_attribution = attribution_id;
            }
            edge.last_attribution = attribution_id;
            if tail != NIL {
                self.attributions[tail as usize].next = attribution_id;
            }
        }
        Ok(())
    }

    /// A fresh context holding exactly this one's LIVE content — the
    /// offline answer to append-only storage: fully retracted edges,
    /// their unlinked attribution records, arena bytes behind removed
    /// aliases and dead names, and index slack all stay behind. Every
    /// live edge is re-asserted assertion by assertion (per-source
    /// sums re-accumulate within float re-addition error; counts and
    /// first-assertion paragraph locators are exact), sourceless
    /// weight included. An alias whose canonical no longer carries any
    /// live edge cannot re-intern its target and is dropped — counted,
    /// never silent. The caller re-applies configuration the image
    /// never holds (`applied_seq`, `dice_floor`).
    ///
    /// `deadline` is checked once per association and once per alias —
    /// not inside `query_any` itself, which collects every association
    /// up front (its fast path for an all-wildcard query) before this loop ever
    /// runs, so a deadline that is already tight when this is called
    /// cannot shorten that initial O(edges) collection.
    ///
    /// # Errors
    ///
    /// [`ContextFull`] is structurally unreachable — the rebuild holds
    /// a subset of what this context already held — but the write API
    /// says it, so this signature does too.
    pub fn compacted(
        &self,
        deadline: Deadline,
    ) -> Result<(Context, CompactionStats), CompactionError> {
        let mut fresh = Context::default();
        let mut stats = CompactionStats::default();
        for association in self.query_any(&[], &[], &[]) {
            if deadline.expired() {
                return Err(CompactionError::DeadlineExceeded);
            }
            if association.count == 0 {
                stats.dead_edges += 1;
                continue;
            }
            // The edge's final totals are known up front: its count is
            // the association's, and its sum is the average weight times
            // that count (`weight` is the average `sum / count`). Each
            // attribution carries its own cumulative sum and count
            // verbatim. Sourceless weight is whatever the edge total
            // exceeds the attributed share — it needs no record, only to
            // be inside the edge sum/count, which it already is.
            let attributions: Vec<(String, f64, u64, Option<u32>)> = association
                .attributions
                .iter()
                .map(|attribution| {
                    (
                        attribution.source.clone(),
                        attribution.weight,
                        attribution.count,
                        attribution.paragraph,
                    )
                })
                .collect();
            // The edge total is its average times its count by
            // construction, but `sum / count` rounds up for some values,
            // so the product can tip just past f64::MAX to ±inf even
            // though the original (saturated) sum was finite. install_edge
            // needs a finite sum — the image invariant its debug_assert
            // stands in for — so clamp the reconstruction the same way the
            // per-source sums are re-accumulated.
            let mut edge_sum = association.weight * association.count as f64;
            if !edge_sum.is_finite() {
                edge_sum = f64::MAX.copysign(edge_sum);
            }
            fresh.install_edge(
                &association.subject,
                &association.label,
                &association.object,
                association.count,
                edge_sum,
                &attributions,
            )?;
        }
        for (alias, canonical) in self.concept_aliases() {
            if deadline.expired() {
                return Err(CompactionError::DeadlineExceeded);
            }
            match fresh.add_concept_alias(alias, canonical) {
                Ok(()) => {}
                Err(AliasError::Full(full)) => return Err(full.into()),
                Err(_) => stats.aliases_dropped += 1,
            }
        }
        for (alias, canonical) in self.label_aliases() {
            if deadline.expired() {
                return Err(CompactionError::DeadlineExceeded);
            }
            match fresh.add_label_alias(alias, canonical) {
                Ok(()) => {}
                Err(AliasError::Full(full)) => return Err(full.into()),
                Err(_) => stats.aliases_dropped += 1,
            }
        }
        Ok((fresh, stats))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::context::Attribution;
    use crate::context::test_support::{associate_examples, weight_between};

    /// Compaction reproduces every live edge — count, weight, and the
    /// full per-source attribution breakdown including the paragraph
    /// locator and sourceless residual — via the O(distinct) install
    /// path, not the old O(total-assertion) replay. A heavily
    /// corroborated edge (asserted many times) must round-trip exactly.
    #[test]
    fn compaction_reproduces_edges_at_their_final_totals() {
        let mut context = Context::default();
        // One edge corroborated 500 times by a.md (locator on the first),
        // plus twice by b.md, plus one sourceless assertion.
        for index in 0..500 {
            context
                .associate_from("蔵", "杜氏", "高瀬", 1.0, "a.md", (index == 0).then_some(3))
                .unwrap();
        }
        context
            .associate_from("蔵", "杜氏", "高瀬", 2.0, "b.md", None)
            .unwrap();
        context
            .associate_from("蔵", "杜氏", "高瀬", 2.0, "b.md", None)
            .unwrap();
        context.associate("蔵", "杜氏", "高瀬", 5.0).unwrap();
        // A second, independent edge and a fully retracted one.
        context
            .associate_from("蔵", "銘柄", "青嶺", 1.0, "a.md", None)
            .unwrap();
        context
            .associate_from("蔵", "廃", "旧", 1.0, "x.md", None)
            .unwrap();
        context.retract_source("x.md");

        let before = context.query(Some("蔵"), Some("杜氏"), Some("高瀬"));
        // The retraction above is already live-tracked as dead weight,
        // ahead of any compaction.
        assert_eq!(context.dead_edges(), 1);
        assert_eq!(context.dead_attributions(), 1);
        let (fresh, stats) = context.compacted(Deadline::unbounded()).unwrap();
        assert_eq!(stats.dead_edges, 1, "the retracted edge sheds");
        // Compaction carries forward only what's live, so the fresh
        // context's dead-weight counters start over at zero — nothing
        // survives for them to count.
        assert_eq!(fresh.dead_edges(), 0);
        assert_eq!(fresh.dead_attributions(), 0);
        assert_eq!(fresh.arena_slack(), 0);
        let after = fresh.query(Some("蔵"), Some("杜氏"), Some("高瀬"));
        assert_eq!(
            before, after,
            "the corroborated edge must round-trip exactly"
        );

        let edge = &after[0];
        assert_eq!(edge.count, 503, "500 + 2 + 1 assertions");
        // Live edges survive; the dead one is gone.
        assert_eq!(fresh.association_count(), 2);
        let a_md = edge
            .attributions
            .iter()
            .find(|attribution| attribution.source == "a.md")
            .expect("a.md attribution survives");
        assert_eq!(a_md.count, 500);
        assert_eq!(a_md.paragraph, Some(3), "first-assertion locator preserved");
    }

    #[test]
    fn associate_registers_concepts_and_signed_weights() {
        let mut context = Context::default();
        associate_examples(&mut context);

        let concept_names: HashSet<String> = context.concept_ids.keys().cloned().collect();
        assert_eq!(
            concept_names,
            HashSet::from(["私", "りんご", "みかん", "バナナ"].map(String::from))
        );
        // Labels are interned separately — "好き" is a relation, not a concept.
        assert!(!context.concept_ids.contains_key("好き"));

        assert_eq!(weight_between(&context, "私", "好き", "りんご"), 1.0);
        assert_eq!(weight_between(&context, "私", "好き", "みかん"), 2.0);
        assert_eq!(weight_between(&context, "私", "好き", "バナナ"), -1.0);
    }

    #[test]
    fn compacting_a_saturated_edge_reconstructs_a_finite_sum() {
        // compaction rebuilds each edge from its average weight and count,
        // then hands the product `weight * count` to install_edge. An edge
        // whose sum saturated at f64::MAX carries a count that need not
        // divide it evenly, so the average rounds up just enough that the
        // product tips back past f64::MAX to +inf — and install_edge's
        // debug_assert demands the finite sum the image invariant promises.
        // The reconstruction must re-clamp, exactly as the live folds do.
        let mut context = Context::default();
        // Three folds of f64::MAX leave the edge sum saturated at f64::MAX
        // with count 3; (f64::MAX / 3) * 3 rounds to +inf.
        context.associate("a", "r", "b", f64::MAX).unwrap();
        context.associate("a", "r", "b", f64::MAX).unwrap();
        context.associate("a", "r", "b", f64::MAX).unwrap();

        let (fresh, _) = context
            .compacted(Deadline::unbounded())
            .expect("compacting a saturated edge must not reconstruct a non-finite sum");
        assert!(
            weight_between(&fresh, "a", "r", "b").is_finite(),
            "the reconstructed edge weight must stay finite"
        );
        // The clamped image must still round-trip, the same guarantee the
        // live saturation path gives.
        Context::from_bytes(&fresh.to_bytes())
            .expect("a compacted context with a clamped edge sum must round-trip");
    }

    #[test]
    fn repeated_associations_average_rather_than_sum() {
        let mut context = Context::default();

        // The first mention seeds the weight directly.
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        assert_eq!(weight_between(&context, "私", "好き", "りんご"), 1.0);

        // A later agreeing repeat of the SAME magnitude leaves weight
        // unchanged — it's averaged in, not piled on top — which is what
        // lets a caller tell "this was corroborated again" apart from "this
        // was asserted once, more emphatically".
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        assert_eq!(weight_between(&context, "私", "好き", "りんご"), 1.0);

        // A more emphatic restatement (e.g. "大好き" carrying a bigger
        // weight) still pulls the average toward it.
        context.associate("私", "好き", "りんご", 5.0).unwrap();
        assert_eq!(weight_between(&context, "私", "好き", "りんご"), 7.0 / 3.0);
    }

    #[test]
    fn opposite_signed_evidence_nets_against_the_existing_weight() {
        let mut context = Context::default();

        context.associate("私", "好き", "りんご", 2.0).unwrap();
        // Contradicts, but not enough to overturn it.
        context.associate("私", "好き", "りんご", -0.5).unwrap();
        assert_eq!(weight_between(&context, "私", "好き", "りんご"), 1.5 / 2.0);

        // Contradicts hard enough this time to flip the sign outright.
        context.associate("私", "好き", "りんご", -3.0).unwrap();
        assert_eq!(weight_between(&context, "私", "好き", "りんご"), -1.5 / 3.0);
    }

    #[test]
    fn a_weight_can_cross_zero_and_keep_accumulating() {
        let mut context = Context::default();

        context.associate("私", "好き", "りんご", 1.0).unwrap();
        context.associate("私", "好き", "りんご", -1.0).unwrap();
        assert_eq!(weight_between(&context, "私", "好き", "りんご"), 0.0);

        // Landing on exactly 0.0 is just another value to average in, not
        // a dead end — the next call keeps contributing from there like
        // any other.
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        assert_eq!(weight_between(&context, "私", "好き", "りんご"), 1.0 / 3.0);
    }

    #[test]
    fn associate_from_records_which_source_contributed_what() {
        let mut context = Context::default();
        context
            .associate_from("決定", "手段", "投票", 1.0, "IPA公式", None)
            .unwrap();
        context
            .associate_from("決定", "手段", "投票", 1.0, "解説記事", None)
            .unwrap();

        let recalled = context.recall("投票");
        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].weight, 1.0);
        assert_eq!(recalled[0].count, 2);
        assert_eq!(
            recalled[0].attributions,
            vec![
                Attribution {
                    source: "IPA公式".to_string(),
                    weight: 1.0,
                    count: 1,
                    paragraph: None,
                },
                Attribution {
                    source: "解説記事".to_string(),
                    weight: 1.0,
                    count: 1,
                    paragraph: None,
                },
            ]
        );
    }

    #[test]
    fn attributions_tell_corroboration_apart_from_single_source_emphasis() {
        let mut context = Context::default();
        // Same total, different evidence story: "a" is corroborated by two
        // independent sources each asserting 1.0; "x" is asserted by one
        // source emphatically asserting 2.0 a single time.
        context
            .associate_from("a", "r", "b", 1.0, "文書1", None)
            .unwrap();
        context
            .associate_from("a", "r", "b", 1.0, "文書2", None)
            .unwrap();
        context
            .associate_from("x", "r", "y", 2.0, "文書1", None)
            .unwrap();

        let corroborated = &context.query(Some("a"), None, None)[0];
        let emphatic = &context.query(Some("x"), None, None)[0];
        // Averaging tells the two stories apart: corroboration settles at
        // the per-assertion weight (1.0), while a single emphatic
        // assertion keeps its full weight (2.0).
        assert_eq!(corroborated.weight, 1.0);
        assert_eq!(emphatic.weight, 2.0);
        assert_eq!(corroborated.attributions.len(), 2);
        assert_eq!(emphatic.attributions.len(), 1);
    }

    #[test]
    fn repeated_assertions_from_one_source_accumulate_into_one_attribution() {
        let mut context = Context::default();
        context
            .associate_from("a", "r", "b", 1.0, "文書1", None)
            .unwrap();
        context
            .associate_from("a", "r", "b", 0.5, "文書1", None)
            .unwrap();

        let recalled = context.recall("a");
        assert_eq!(recalled[0].weight, 0.75);
        assert_eq!(recalled[0].count, 2);
        assert_eq!(
            recalled[0].attributions,
            vec![Attribution {
                source: "文書1".to_string(),
                weight: 1.5,
                count: 2,
                paragraph: None,
            }]
        );
    }

    #[test]
    fn first_locator_wins_on_repeated_assertions_from_one_source() {
        let mut context = Context::default();
        context
            .associate_from("a", "r", "b", 1.0, "文書1", Some(2))
            .unwrap();
        // A later re-assertion from the same source accumulates weight
        // but must not overwrite the paragraph the source was first
        // located at.
        context
            .associate_from("a", "r", "b", 0.5, "文書1", Some(9))
            .unwrap();

        let recalled = context.recall("a");
        assert_eq!(recalled[0].weight, 0.75);
        assert_eq!(
            recalled[0].attributions,
            vec![Attribution {
                source: "文書1".to_string(),
                weight: 1.5,
                count: 2,
                paragraph: Some(2),
            }]
        );

        // The same rule holds in the other direction: a first assertion
        // with no locator stays unlocated even when a later re-assertion
        // from the same source supplies one.
        let mut unlocated_first = Context::default();
        unlocated_first
            .associate_from("c", "r", "d", 1.0, "文書1", None)
            .unwrap();
        unlocated_first
            .associate_from("c", "r", "d", 0.5, "文書1", Some(9))
            .unwrap();
        assert_eq!(
            unlocated_first.recall("c")[0].attributions,
            vec![Attribution {
                source: "文書1".to_string(),
                weight: 1.5,
                count: 2,
                paragraph: None,
            }]
        );
    }

    #[test]
    fn unsourced_and_sourced_assertions_mix_on_one_association() {
        let mut context = Context::default();
        context.associate("a", "r", "b", 1.0).unwrap();
        context
            .associate_from("a", "r", "b", 0.5, "文書1", None)
            .unwrap();

        // Total weight counts both; only the sourced part is attributed.
        let recalled = context.recall("a");
        assert_eq!(recalled[0].weight, 0.75);
        assert_eq!(recalled[0].count, 2);
        assert_eq!(
            recalled[0].attributions,
            vec![Attribution {
                source: "文書1".to_string(),
                weight: 0.5,
                count: 1,
                paragraph: None,
            }]
        );
    }

    /// Non-finite weights corrupt ranking (`total_cmp` sorts NaN as
    /// the maximum) and break the to_bytes/from_bytes round trip, so
    /// the write boundary refuses them outright.
    #[test]
    #[should_panic(expected = "must be finite")]
    fn a_non_finite_weight_is_refused_at_the_boundary() {
        let mut context = Context::default();
        let _ = context.associate("A", "rel", "B", f64::NAN);
    }

    #[test]
    fn associate_returns_ok_while_room_remains() {
        let mut context = Context::default();
        assert_eq!(context.associate("私", "好き", "りんご", 1.0), Ok(()));
        assert_eq!(
            context.associate_from("私", "好き", "りんご", 1.0, "文書1", None),
            Ok(())
        );
    }
}
