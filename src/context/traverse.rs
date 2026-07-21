use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use crate::deadline::{Deadline, DeadlineExceeded};

use super::{
    Activation, Association, ConceptId, Context, EdgeId, EdgeRecord, Recollection,
    accumulate_saturating, clamp_unit_or,
};

impl Context {
    /// Depth for [`Context::explore`] meaning "no limit": the walk covers
    /// the origins' whole connected component.
    pub const UNBOUNDED: usize = usize::MAX;

    /// Activation below which [`Context::activate`] stops propagating.
    /// Origins start at 1.0 and every hop multiplies by ≤ decay, so this
    /// is a fraction of an origin's signal: knowledge reached with less
    /// than a millionth of it cannot influence any realistic ranking,
    /// and cutting there bounds a call to the origins' effective
    /// neighborhood instead of their entire connected component.
    const ACTIVATION_FLOOR: f64 = 1e-6;

    /// Walks the network outward from `origins` and returns every
    /// association reachable within `max_depth` hops, each annotated with
    /// the hop count at which it was first reached.
    ///
    /// This is the structural sweep: it treats every edge as equally worth
    /// returning and bounds the walk purely by distance. Use
    /// [`Context::activate`] instead when results should be *ranked* — by
    /// how near and how heavily weighted they are — rather than enumerated.
    ///
    /// Traversal follows edges in both directions — an association is
    /// directional in meaning, not in reachability. It can start from
    /// several origins at once, and each association reports its distance
    /// from the nearest origin. It does NOT travel through shared relation
    /// labels: labels annotate edges rather than connect them, precisely so
    /// that two unrelated facts using the same label (e.g. "好き") don't
    /// leak into each other's neighborhoods.
    ///
    /// Results are ordered by distance, then by insertion order within the
    /// same distance, so output is deterministic for a given `Context`
    /// history. Each result carries the concept path from its origin to
    /// where it was reached. Origins naming unknown concepts contribute
    /// nothing, and `max_depth == 0` returns nothing — zero hops of
    /// association-following reaches no associations.
    pub fn explore(&self, origins: &[&str], max_depth: usize) -> Vec<Recollection> {
        let mut node_distances: HashMap<ConceptId, usize> = HashMap::new();
        let mut parents: HashMap<ConceptId, ConceptId> = HashMap::new();
        let mut frontier: VecDeque<ConceptId> = VecDeque::new();
        for &origin in origins {
            if let Some(&id) = self.concept_ids.get(origin)
                && let Entry::Vacant(entry) = node_distances.entry(id)
            {
                entry.insert(0);
                frontier.push_back(id);
            }
        }

        // Breadth-first, so a node comes off the frontier with its minimal
        // distance already settled, and an edge's first sighting is from
        // its nearer endpoint — which is the endpoint the path leads to.
        let mut reached: HashMap<EdgeId, (usize, ConceptId)> = HashMap::new();
        while let Some(concept_id) = frontier.pop_front() {
            let hop = node_distances[&concept_id] + 1;
            if hop > max_depth {
                continue;
            }
            for edge_id in self.outgoing(concept_id).chain(self.incoming(concept_id)) {
                let edge = &self.edges[edge_id as usize];
                // A retracted edge (count == 0) still hangs off the
                // adjacency chains — retract_association unlinks only the
                // attribution records — so it must not act as a bridge to
                // otherwise-unrelated live facts. count == 0 is the
                // dead-edge test everywhere else (heaviest, compacted, the
                // export); apply it here too.
                if edge.count == 0 {
                    continue;
                }
                reached.entry(edge_id).or_insert((hop, concept_id));
                for neighbor in [edge.subject, edge.object] {
                    if let Entry::Vacant(entry) = node_distances.entry(neighbor) {
                        entry.insert(hop);
                        parents.insert(neighbor, concept_id);
                        frontier.push_back(neighbor);
                    }
                }
            }
        }

        let mut ordered: Vec<(usize, EdgeId, ConceptId)> = reached
            .into_iter()
            .map(|(edge_id, (distance, from))| (distance, edge_id, from))
            .collect();
        ordered.sort_unstable();

        ordered
            .into_iter()
            .map(|(distance, edge_id, from)| Recollection {
                distance,
                path: self.path_from(&parents, from),
                association: self.association(edge_id),
            })
            .collect()
    }

    /// Reconstructs the concept-name trail from an origin down to
    /// `concept` by following the walk's parent pointers, origin first.
    fn path_from(
        &self,
        parents: &HashMap<ConceptId, ConceptId>,
        concept: ConceptId,
    ) -> Vec<String> {
        let mut trail = vec![concept];
        let mut cursor = concept;
        while let Some(&parent) = parents.get(&cursor) {
            trail.push(parent);
            cursor = parent;
        }
        trail.reverse();
        trail
            .into_iter()
            .map(|id| self.concept_name(id).to_string())
            .collect()
    }

    /// Spreads activation from `origins` through the network and returns
    /// the full pre-truncation match count alongside the `limit` most
    /// strongly activated associations, best first — mirroring what
    /// `api::page` does for `Vec<Association>`, so a caller can tell a
    /// complete result from a truncated one. This is the ranked
    /// counterpart of [`Context::explore`] — the retrieval that reads the
    /// weights `associate` has been writing.
    ///
    /// Each origin starts with activation 1.0, and a concept's activation
    /// is that of its single strongest incoming path. Two formulas share
    /// that activation but do different jobs. Both rank on `sum`, the
    /// edge's raw cumulative total — NOT the averaged [`Association::weight`]
    /// returned to callers — so that corroboration (more evidence summing
    /// higher) keeps outranking a single assertion of the same average
    /// intensity:
    ///
    /// - An association's returned strength is
    ///   `activation(nearer endpoint) * decay * |sum|` — deliberately
    ///   independent of how many OTHER associations its endpoints carry.
    ///   A fact about an anchor must not sink because the anchor is well
    ///   documented, and facts of several origins must compete fairly
    ///   however lopsided the origins' degrees are.
    /// - What flows ON to a neighboring concept is fan-normalized:
    ///   `activation * decay * |sum| / Σ|sum|` over the concept's
    ///   associations. A promiscuous hub splits its activation thinly, so
    ///   everything BEYOND a hub fades — connection through a busy concept
    ///   is weak evidence of relatedness. At equal depth and sum, an
    ///   association just past a hub therefore ties with one past a
    ///   dedicated relay; the hub's penalty lands on all knowledge beyond.
    ///
    /// Magnitude ranks, sign is content: sum -2.0 is strong knowledge
    /// ("firmly not the case") and outranks +1.0, while associations
    /// netted out to 0.0 are not returned at all — a fully contested fact
    /// carries no reliable knowledge. `decay` shrinks the signal each hop
    /// (clamped into [0, 1]). Strengths are ordinal: compare them within
    /// one call's results, not across calls or corpus versions. Callers
    /// that care about independent corroboration rather than accumulated
    /// sum can re-rank the returned associations by their number of
    /// attributions. Results are deterministic — strength descending, then
    /// insertion order — and origins naming unknown concepts contribute
    /// nothing.
    ///
    /// Activation that decays below [`Context::ACTIVATION_FLOOR`] (one
    /// millionth of an origin's) stops propagating, so a call costs the
    /// origins' effective neighborhood, not their whole connected
    /// component; associations only reachable with less activation than
    /// that are simply absent from the result. The cutoff only bites
    /// where decay and fan-out actually attenuate the signal — with
    /// `decay == 1.0` through dedicated relays, activation barely decays
    /// and the walk still covers whatever stays above the floor.
    pub fn activate(&self, origins: &[&str], decay: f64, limit: usize) -> (usize, Vec<Activation>) {
        // NaN must shrink every signal to nothing (like decay == 0.0),
        // not propagate — clamp alone would let it through, since the
        // score gate below is a `<=` that a NaN score never satisfies.
        let decay = clamp_unit_or(decay, 0.0);

        let mut best: HashMap<ConceptId, f64> = HashMap::new();
        let mut parents: HashMap<ConceptId, ConceptId> = HashMap::new();
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();
        for &origin in origins {
            if let Some(&id) = self.concept_ids.get(origin) {
                best.insert(id, 1.0);
                heap.push(Candidate {
                    activation: 1.0,
                    concept: id,
                });
            }
        }

        // Best-first (Dijkstra-style): every propagation factor is <= 1,
        // so the first time a concept is popped it carries its maximal
        // activation and can be settled for good. Each strength remembers
        // which settled endpoint scored it, so the strongest activation
        // path can be reconstructed for the result.
        let mut settled: HashSet<ConceptId> = HashSet::new();
        let mut strengths: HashMap<EdgeId, (f64, ConceptId)> = HashMap::new();
        while let Some(Candidate {
            activation,
            concept,
        }) = heap.pop()
        {
            if !settled.insert(concept) {
                continue;
            }
            let total: f64 = self
                .outgoing(concept)
                .chain(self.incoming(concept).filter(|&edge_id| {
                    // A self-loop threads BOTH of this concept's chains;
                    // summed once per chain it would dilute every
                    // neighbor's share as if the loop were two edges.
                    let edge = &self.edges[edge_id as usize];
                    edge.subject != edge.object
                }))
                // Retracted edges (count == 0) linger in the chain walk
                // until compaction; a fully-withdrawn association must
                // not weigh in the total or propagate. Same dead-edge
                // test as `heaviest`, `describe`, and the export.
                .filter(|&edge_id| self.edges[edge_id as usize].count > 0)
                .map(|edge_id| self.edges[edge_id as usize].sum.abs())
                .fold(0.0, |mut acc, magnitude| {
                    // Individually-finite magnitudes can sum past f64's
                    // range; an infinite total would zero every flow and
                    // silently stop propagation through this concept.
                    accumulate_saturating(&mut acc, magnitude);
                    acc
                });
            if total == 0.0 {
                continue;
            }
            for edge_id in self
                .outgoing(concept)
                .chain(self.incoming(concept))
                .filter(|&edge_id| self.edges[edge_id as usize].count > 0)
            {
                let edge = &self.edges[edge_id as usize];
                // Ranking: fan-independent, so a busy endpoint doesn't
                // sink its own facts. Netted-out (or zero-decay) signals
                // carry nothing and are skipped entirely.
                let score = activation * decay * edge.sum.abs();
                if score <= 0.0 {
                    continue;
                }
                let strength = strengths.entry(edge_id).or_insert((0.0, concept));
                if score > strength.0 {
                    *strength = (score, concept);
                }

                // Propagation: fan-normalized, so a busy node dilutes
                // everything beyond itself. Signals worn down below the
                // floor stop here — that bound is what keeps a call from
                // sweeping the whole connected component.
                let flow = score / total;
                if flow < Self::ACTIVATION_FLOOR {
                    continue;
                }
                for neighbor in [edge.subject, edge.object] {
                    if settled.contains(&neighbor) {
                        continue;
                    }
                    if flow > best.get(&neighbor).copied().unwrap_or(0.0) {
                        best.insert(neighbor, flow);
                        parents.insert(neighbor, concept);
                        heap.push(Candidate {
                            activation: flow,
                            concept: neighbor,
                        });
                    }
                }
            }
        }

        let mut ranked: Vec<(f64, EdgeId, ConceptId)> = strengths
            .into_iter()
            .map(|(edge_id, (strength, from))| (strength, edge_id, from))
            .collect();
        ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let total = ranked.len();
        ranked.truncate(limit);

        let matches = ranked
            .into_iter()
            .map(|(strength, edge_id, from)| Activation {
                strength,
                path: self.path_from(&parents, from),
                association: self.association(edge_id),
            })
            .collect();
        (total, matches)
    }

    /// Whether any single association directly connects the two
    /// concepts, in either direction under any label. Unknown names are
    /// simply not adjacent. Audits use this to tell RELATED apart from
    /// DUPLICATE: concepts joined by an edge legitimately resemble each
    /// other (their glosses quote the same fact) and are not forks.
    pub fn adjacent(&self, a: &str, b: &str) -> bool {
        let (Some(&id_a), Some(&id_b)) = (self.concept_ids.get(a), self.concept_ids.get(b)) else {
            return false;
        };
        // Checked separately, not `.chain(...).any(|e| subject == id_b ||
        // object == id_b)`: every `outgoing` edge already has `subject ==
        // id_a`, so when `a` and `b` name the same concept that combined
        // condition is trivially true off `subject` alone for ANY edge
        // touching `id_a` — not just a genuine self-loop. Testing only
        // the far endpoint on each side keeps `id_a == id_b` meaning "a
        // self-loop exists" instead of "this concept has any edge at
        // all".
        let live = |edge: &EdgeRecord| edge.count > 0;
        // Retracted edges (count == 0) linger in the chain walk until
        // compaction; a withdrawn association must not read as a live
        // adjacency. Same dead-edge test as `heaviest`, `describe`, and
        // the export.
        self.outgoing(id_a).any(|edge_id| {
            let edge = &self.edges[edge_id as usize];
            live(edge) && edge.object == id_b
        }) || self.incoming(id_a).any(|edge_id| {
            let edge = &self.edges[edge_id as usize];
            live(edge) && edge.subject == id_b
        })
    }

    /// Whether the two relation labels are ever used on one common
    /// subject. Genuinely distinct labels tend to co-occur on a subject
    /// (単数形にする条件 and 複数形にする条件 on 配列名); accidental
    /// forks tend NOT to — they were minted in parallel and used apart —
    /// so co-occurrence marks a pair as related rather than duplicate.
    pub fn labels_share_subject(&self, a: &str, b: &str) -> bool {
        let (Some(&id_a), Some(&id_b)) = (self.label_ids.get(a), self.label_ids.get(b)) else {
            return false;
        };
        // Skip retracted edges (count == 0) on both sides: they linger on
        // the label chains until compaction, and a withdrawn fact must
        // not make two labels look co-occurrent. Same dead-edge test as
        // `heaviest`, `describe`, and the export.
        let subjects: HashSet<ConceptId> = self
            .labeled(id_a)
            .filter(|&edge_id| self.edges[edge_id as usize].count > 0)
            .map(|edge_id| self.edges[edge_id as usize].subject)
            .collect();
        self.labeled(id_b)
            .filter(|&edge_id| self.edges[edge_id as usize].count > 0)
            .any(|edge_id| subjects.contains(&self.edges[edge_id as usize].subject))
    }

    /// Lists every association that no walk from `origins` can ever reach
    /// — the post-ingest coverage audit. Unreachable knowledge fails
    /// silently: nothing errors, retrieval simply never returns it. Run
    /// this after ingesting a document, anchored at the document's main
    /// entities; a non-empty result means the decomposition left facts
    /// disconnected (usually an implicit membership that never became an
    /// edge) and names exactly which ones. Reachability is bidirectional
    /// and does not travel through labels, exactly as in
    /// [`Context::explore`]. CPU-bound over the whole edge table; checks
    /// `deadline` periodically like `unsourced_edges`/`similar_concepts`.
    pub fn unreachable_from(
        &self,
        origins: &[&str],
        deadline: Deadline,
    ) -> Result<Vec<Association>, DeadlineExceeded> {
        let mut visited: HashSet<ConceptId> = HashSet::new();
        let mut frontier: VecDeque<ConceptId> = VecDeque::new();
        for &origin in origins {
            if let Some(&id) = self.concept_ids.get(origin)
                && visited.insert(id)
            {
                frontier.push_back(id);
            }
        }
        while let Some(concept_id) = frontier.pop_front() {
            if deadline.expired() {
                return Err(DeadlineExceeded);
            }
            for edge_id in self.outgoing(concept_id).chain(self.incoming(concept_id)) {
                let edge = &self.edges[edge_id as usize];
                // A retracted edge (count == 0) lingers in the adjacency
                // chains and must not act as a bridge between otherwise
                // disconnected live facts — same dead-edge test as
                // `explore` and `heaviest`.
                if edge.count == 0 {
                    continue;
                }
                for neighbor in [edge.subject, edge.object] {
                    if visited.insert(neighbor) {
                        frontier.push_back(neighbor);
                    }
                }
            }
        }

        // An edge's endpoints reach each other through it, so checking one
        // endpoint decides the whole edge. Retracted edges (count == 0)
        // are no longer facts at all, so they're excluded rather than
        // reported as unreachable ones.
        let mut out = Vec::new();
        for edge_id in 0..self.edges.len() as u32 {
            if deadline.expired() {
                return Err(DeadlineExceeded);
            }
            let edge = &self.edges[edge_id as usize];
            if edge.count > 0 && !visited.contains(&edge.subject) {
                out.push(self.association(edge_id));
            }
        }
        Ok(out)
    }
}

/// Heap entry for [`Context::activate`]: max-ordered by activation, ties
/// broken toward the lower concept id so pop order is deterministic.
struct Candidate {
    activation: f64,
    concept: ConceptId,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.activation
            .total_cmp(&other.activation)
            .then_with(|| other.concept.cmp(&self.concept))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::test_support::assoc;

    #[test]
    fn activation_still_propagates_through_extreme_fan_out_sums() {
        // The fan-out normalizer sums |edge.sum| across a concept's
        // incident edges. Two saturated edges would push an unsaturated
        // total to +Infinity, zeroing every flow — the hub's own edges
        // still rank (strength is fan-independent), but nothing beyond
        // the hub would ever be reached.
        let mut context = Context::default();
        context.associate("hub", "r1", "leaf1", f64::MAX).unwrap();
        context.associate("hub", "r2", "leaf2", f64::MAX).unwrap();
        context.associate("leaf1", "r3", "far", 1.0).unwrap();

        let (_, activations) = context.activate(&["hub"], 0.5, 10);
        assert!(
            activations
                .iter()
                .any(|activation| activation.association.object == "far"),
            "activation must propagate past the hub to leaf1's own association"
        );
    }

    #[test]
    fn explore_walks_hop_by_hop_and_reports_distance() {
        let mut context = Context::default();
        // A chain: 私 → りんご → 果物 → ビタミン
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        context.associate("りんご", "分類", "果物", 1.0).unwrap();
        context.associate("果物", "含む", "ビタミン", 1.0).unwrap();

        // Zero hops of association-following reaches no associations.
        assert!(context.explore(&["私"], 0).is_empty());

        let one_hop = context.explore(&["私"], 1);
        assert_eq!(one_hop.len(), 1);
        assert_eq!(one_hop[0].distance, 1);
        assert_eq!(one_hop[0].association, assoc("私", "好き", "りんご", 1.0));

        let two_hops = context.explore(&["私"], 2);
        assert_eq!(two_hops.len(), 2);
        assert_eq!(two_hops[1].distance, 2);
        assert_eq!(
            two_hops[1].association,
            assoc("りんご", "分類", "果物", 1.0)
        );
        // The path names the intermediate concept the walk ran through.
        assert_eq!(two_hops[0].path, vec!["私"]);
        assert_eq!(two_hops[1].path, vec!["私", "りんご"]);

        // Depth beyond the component's diameter just returns the whole
        // connected component, no more.
        assert_eq!(context.explore(&["私"], 3).len(), 3);
        assert_eq!(context.explore(&["私"], 100).len(), 3);
    }

    #[test]
    fn explore_traverses_against_edge_direction_too() {
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        context.associate("農家", "育てる", "りんご", 1.0).unwrap();

        // りんご is the *object* of both edges; both must still be reachable
        // from it, and reaching 農家's fact from 私 runs against 育てる's
        // direction.
        assert_eq!(context.explore(&["りんご"], 1).len(), 2);

        let from_me = context.explore(&["私"], 2);
        assert_eq!(from_me.len(), 2);
        assert!(from_me.contains(&Recollection {
            distance: 2,
            path: vec!["私".to_string(), "りんご".to_string()],
            association: assoc("農家", "育てる", "りんご", 1.0),
        }));
    }

    #[test]
    fn explore_does_not_leak_through_shared_labels() {
        let mut context = Context::default();
        // Two facts share only the label "好き" — that must NOT connect
        // them: labels annotate edges, they are not nodes of the network.
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        context.associate("犬", "好き", "骨", 1.0).unwrap();

        let neighborhood = context.explore(&["私"], 100);
        assert_eq!(neighborhood.len(), 1);
        assert_eq!(
            neighborhood[0].association,
            assoc("私", "好き", "りんご", 1.0)
        );
    }

    #[test]
    fn explore_from_multiple_origins_keeps_the_nearest_distance() {
        let mut context = Context::default();
        // A chain: a → b → c → d → e
        context.associate("a", "r1", "b", 1.0).unwrap();
        context.associate("b", "r2", "c", 1.0).unwrap();
        context.associate("c", "r3", "d", 1.0).unwrap();
        context.associate("d", "r4", "e", 1.0).unwrap();

        let both_ends = context.explore(&["a", "e"], 2);
        assert_eq!(both_ends.len(), 4);

        // Ordered by distance first: the two end edges, then the middle
        // ones — each middle edge 2 hops from its NEAREST end, not 3 from
        // the far one.
        let distances: Vec<usize> = both_ends.iter().map(|r| r.distance).collect();
        assert_eq!(distances, vec![1, 1, 2, 2]);
        assert!(both_ends.contains(&Recollection {
            distance: 2,
            path: vec!["a".to_string(), "b".to_string()],
            association: assoc("b", "r2", "c", 1.0),
        }));
    }

    #[test]
    fn explore_ignores_unknown_origins() {
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();

        assert!(context.explore(&["存在しない概念"], 3).is_empty());

        // A known origin still works alongside an unknown one.
        assert_eq!(context.explore(&["存在しない概念", "私"], 1).len(), 1);
    }

    #[test]
    fn explore_does_not_bridge_through_a_retracted_edge() {
        let mut context = Context::default();
        // 私 -- りんご -- 果物: retracting the middle edge must sever the
        // path, not leave it standing as a free bridge to 果物's facts.
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        context.associate("りんご", "分類", "果物", 1.0).unwrap();
        context.associate("果物", "含む", "ビタミン", 1.0).unwrap();
        context
            .retract_association("りんご", "分類", "果物")
            .unwrap();

        let reached = context.explore(&["私"], Context::UNBOUNDED);
        assert_eq!(
            reached.len(),
            1,
            "the retracted edge must not surface, nor bridge to 果物's facts"
        );
        assert_eq!(reached[0].association, assoc("私", "好き", "りんご", 1.0));
    }

    #[test]
    fn activate_ranks_direct_strong_edges_above_weak_ones() {
        let mut context = Context::default();
        context.associate("起点", "強い関係", "A", 3.0).unwrap();
        context.associate("起点", "弱い関係", "B", 1.0).unwrap();

        let (_, ranked) = context.activate(&["起点"], 0.5, 10);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].association.object, "A");
        assert_eq!(ranked[0].strength, 1.5); // 1.0 * 0.5 * |3.0|
        assert_eq!(ranked[1].association.object, "B");
        assert_eq!(ranked[1].strength, 0.5); // 1.0 * 0.5 * |1.0|
    }

    #[test]
    fn activate_keeps_the_first_deterministic_path_when_strengths_tie() {
        let mut context = Context::default();
        context.associate("first", "r", "second", 1.0).unwrap();

        // Heap ties settle the lower concept id first, independently of
        // caller order. Reaching the same edge later at equal strength must
        // not replace that stable path.
        let (_, ranked) = context.activate(&["second", "first"], 0.5, 10);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].strength, 0.5);
        assert_eq!(ranked[0].path, vec!["first"]);
    }

    #[test]
    fn activation_candidates_obey_their_ordering_equality() {
        let a = Candidate {
            activation: 0.5,
            concept: 7,
        };
        let same = Candidate {
            activation: 0.5,
            concept: 7,
        };
        let other = Candidate {
            activation: 0.5,
            concept: 8,
        };
        assert!(a == same);
        assert!(a != other);
    }

    #[test]
    fn activate_propagates_a_signal_exactly_at_the_floor() {
        let mut context = Context::default();
        context.associate("origin", "r", "relay", 1.0).unwrap();
        context
            .associate("origin", "r", "distractor", 999_999.0)
            .unwrap();
        context.associate("relay", "r", "far", 1.0).unwrap();

        let (_, ranked) = context.activate(&["origin"], 1.0, 10);
        let far = ranked
            .iter()
            .find(|hit| hit.association.object == "far")
            .expect("a signal equal to the floor still reaches the relay");
        assert_eq!(far.strength, Context::ACTIVATION_FLOOR);
    }

    #[test]
    fn activate_keeps_the_first_parent_when_equal_flows_meet() {
        let mut context = Context::default();
        context.associate("origin", "r", "a", 1.0).unwrap();
        context.associate("origin", "r", "b", 1.0).unwrap();
        context.associate("a", "r", "join", 1.0).unwrap();
        context.associate("b", "r", "join", 1.0).unwrap();
        context.associate("join", "r", "far", 1.0).unwrap();

        let (_, ranked) = context.activate(&["origin"], 1.0, 20);
        let far = ranked
            .iter()
            .find(|hit| hit.association.object == "far")
            .expect("the joined signal reaches the far edge");
        assert_eq!(far.path, vec!["origin", "a", "join"]);
    }

    #[test]
    fn activate_decays_with_distance() {
        let mut context = Context::default();
        // A chain of equal weights: nearer must outrank farther.
        context.associate("起点", "r", "近い", 1.0).unwrap();
        context.associate("近い", "r", "遠い", 1.0).unwrap();

        let (_, ranked) = context.activate(&["起点"], 0.5, 10);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].association.object, "近い");
        assert_eq!(ranked[0].strength, 0.5); // 1.0 * 0.5 * |1.0|
        assert_eq!(ranked[1].association.object, "遠い");
        assert_eq!(ranked[1].strength, 0.25); // a(近い) = 0.5, then 0.5 * 0.5 * |1.0|

        // Each result carries the activation path that scored it.
        assert_eq!(ranked[0].path, vec!["起点"]);
        assert_eq!(ranked[1].path, vec!["起点", "近い"]);
    }

    #[test]
    fn activate_dilutes_through_promiscuous_hubs() {
        let mut context = Context::default();
        // Two 3-hop chains of identical weights; one passes through a hub
        // with many other associations, one through a dedicated relay.
        context.associate("起点", "r", "中継", 1.0).unwrap();
        context.associate("中継", "r", "中継の先", 1.0).unwrap();
        context.associate("中継の先", "r", "中継の奥", 1.0).unwrap();
        context.associate("起点", "r", "ハブ", 1.0).unwrap();
        context.associate("ハブ", "r", "ハブの先", 1.0).unwrap();
        context.associate("ハブの先", "r", "ハブの奥", 1.0).unwrap();
        context.associate("ハブ", "r", "雑多1", 1.0).unwrap();
        context.associate("ハブ", "r", "雑多2", 1.0).unwrap();
        context.associate("ハブ", "r", "雑多3", 1.0).unwrap();

        let (_, ranked) = context.activate(&["起点"], 0.5, 100);
        let strength_of = |object: &str| {
            ranked
                .iter()
                .find(|a| a.association.object == object)
                .map(|a| a.strength)
                .unwrap()
        };

        // The hub's own facts are not penalized for the hub being busy:
        // at equal depth and weight, the edge just past the hub ties with
        // the one just past the dedicated relay.
        assert_eq!(strength_of("ハブの先"), strength_of("中継の先"));
        assert_eq!(strength_of("中継の先"), 0.125); // a(中継)=0.25, * 0.5 * |1.0|

        // But everything BEYOND the hub inherits the split: the hub passes
        // each neighbor only its share of the flow, so the next hop on
        // diverges — dedicated relay 1/2 vs hub 1/5.
        assert_eq!(strength_of("中継の奥"), 0.03125); // a(中継の先)=0.0625
        assert!((strength_of("ハブの奥") - 0.0125).abs() < 1e-12); // a(ハブの先)=0.025
        assert!(strength_of("中継の奥") > strength_of("ハブの奥"));
    }

    #[test]
    fn activate_treats_negative_weight_as_strong_knowledge() {
        let mut context = Context::default();
        // "Firmly not the case" is strong knowledge: magnitude ranks, the
        // sign stays visible on the returned association.
        context.associate("私", "好き", "バナナ", -2.0).unwrap();
        context.associate("私", "好き", "りんご", 1.0).unwrap();

        let (_, ranked) = context.activate(&["私"], 0.5, 10);
        assert_eq!(ranked[0].association.object, "バナナ");
        assert_eq!(ranked[0].association.weight, -2.0);
        assert!(ranked[0].strength > ranked[1].strength);
    }

    #[test]
    fn activate_drops_netted_out_associations() {
        let mut context = Context::default();
        context.associate("a", "r", "b", 1.0).unwrap();
        context.associate("a", "r", "b", -1.0).unwrap(); // fully contested → 0.0
        context.associate("a", "r", "c", 1.0).unwrap();

        let (_, ranked) = context.activate(&["a"], 0.5, 10);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].association.object, "c");
    }

    /// 0.1 + 0.2, then retracting 0.1 and 0.2 in that order, leaves a
    /// nonzero residual (~2.78e-17) in `sum` even though `count` reaches
    /// 0 — floating-point subtraction doesn't exactly invert addition.
    /// A fully retracted edge must read as dead, not as a residual
    /// signal with a score just barely above zero.
    #[test]
    fn activate_ignores_a_fully_retracted_edges_floating_point_residue() {
        let mut context = Context::default();
        context
            .associate_from("a", "r", "b", 0.1, "s1", None)
            .unwrap();
        context
            .associate_from("a", "r", "b", 0.2, "s2", None)
            .unwrap();
        context.associate("a", "r", "c", 1.0).unwrap();

        context.retract_source("s1");
        context.retract_source("s2");
        assert_eq!(context.dead_edges(), 1);

        let (_, ranked) = context.activate(&["a"], 0.5, 10);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].association.object, "c");
    }

    #[test]
    fn activate_truncates_to_the_strongest_limit_results() {
        let mut context = Context::default();
        context.associate("起点", "r1", "a", 3.0).unwrap();
        context.associate("起点", "r2", "b", 2.0).unwrap();
        context.associate("起点", "r3", "c", 1.0).unwrap();

        let (total, top_two) = context.activate(&["起点"], 0.5, 2);
        assert_eq!(total, 3, "total must reflect the pre-truncation count");
        assert_eq!(top_two.len(), 2);
        assert_eq!(top_two[0].association.object, "a");
        assert_eq!(top_two[1].association.object, "b");
    }

    /// `f64::clamp` returns NaN when `self` is NaN, so a bare
    /// `.clamp(0.0, 1.0)` on a NaN decay would leave the score gate's
    /// `score <= 0.0` comparison false for every edge (NaN compares
    /// false against everything) — never skipping, so every edge would
    /// settle with a NaN strength that `total_cmp` then sorts as the
    /// maximum. A NaN decay must instead shrink every signal to
    /// nothing, exactly like `decay == 0.0`.
    #[test]
    fn activate_treats_a_nan_decay_as_no_signal() {
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();

        let (total, matches) = context.activate(&["私"], f64::NAN, 10);
        assert_eq!(total, 0);
        assert!(matches.is_empty());
    }

    /// A self-loop sits in both the outgoing and the incoming chain of
    /// its one endpoint; an unfiltered fan-out sum counted it twice,
    /// diluting every neighbor's propagated activation as if the loop
    /// were two edges. It must dilute exactly like one ordinary edge.
    #[test]
    fn a_self_loop_dilutes_fan_out_like_one_edge_not_two() {
        let mut looped = Context::default();
        looped.associate("X", "係る", "Y", 1.0).unwrap();
        looped.associate("X", "自己", "X", 1.0).unwrap();
        looped.associate("Y", "係る", "Z", 1.0).unwrap();

        let mut fanned = Context::default();
        fanned.associate("X", "係る", "Y", 1.0).unwrap();
        fanned.associate("X", "分岐", "W", 1.0).unwrap();
        fanned.associate("Y", "係る", "Z", 1.0).unwrap();

        let z_strength = |ranked: &[Activation]| {
            ranked
                .iter()
                .find(|activation| activation.association.object == "Z")
                .expect("Z must be reached")
                .strength
        };
        let via_loop = z_strength(&looped.activate(&["X"], 0.5, 10).1);
        let via_fan = z_strength(&fanned.activate(&["X"], 0.5, 10).1);
        // a(Y) = (1.0 * 0.5 * |1.0|) / 2 edges = 0.25 in both graphs,
        // then strength(Y→Z) = 0.25 * 0.5 * |1.0|.
        assert_eq!(via_loop, 0.125);
        assert_eq!(via_loop, via_fan);
    }

    #[test]
    fn activate_ignores_unknown_origins() {
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();

        assert!(context.activate(&["存在しない概念"], 0.5, 10).1.is_empty());
        assert_eq!(
            context.activate(&["存在しない概念", "私"], 0.5, 10).1.len(),
            1
        );
    }

    #[test]
    fn explore_unbounded_covers_the_whole_component() {
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        context.associate("りんご", "分類", "果物", 1.0).unwrap();
        context.associate("犬", "好き", "骨", 1.0).unwrap(); // separate component

        let everything = context.explore(&["私"], Context::UNBOUNDED);
        assert_eq!(everything.len(), 2);
    }

    #[test]
    fn activate_scores_do_not_sink_as_the_origin_gains_unrelated_facts() {
        let mut context = Context::default();
        context
            .associate("青嶺酒造", "仕込み水", "伏流水", 2.0)
            .unwrap();
        let before = context.activate(&["青嶺酒造"], 0.5, 10).1[0].strength;
        assert_eq!(before, 1.0); // 1.0 * 0.5 * |2.0|

        // Five more facts about the same origin. Under fan-normalized
        // scoring these used to drag every strength down; the water fact
        // itself has not changed, so its score must not move.
        for i in 0..5 {
            context
                .associate("青嶺酒造", format!("関係{i}"), format!("事実{i}"), 1.0)
                .unwrap();
        }
        let (_, after) = context.activate(&["青嶺酒造"], 0.5, 10);
        assert_eq!(after[0].association.object, "伏流水");
        assert_eq!(after[0].strength, before);
    }

    #[test]
    fn activate_ranks_multi_origin_facts_by_weight_not_origin_degree() {
        let mut context = Context::default();
        // A talkative origin with five facts and a terse one with a single
        // heavier fact. Degree must not tilt the ranking between origins.
        for i in 0..5 {
            context
                .associate("多弁", format!("r{i}"), format!("x{i}"), 1.0)
                .unwrap();
        }
        context.associate("寡黙", "r", "y", 2.0).unwrap();

        let (_, ranked) = context.activate(&["多弁", "寡黙"], 0.5, 10);
        assert_eq!(ranked.len(), 6);
        assert_eq!(ranked[0].association.object, "y");
        assert_eq!(ranked[0].strength, 1.0); // weight wins...
        assert!(ranked[1..].iter().all(|a| a.strength == 0.5)); // ...and equal weights tie
    }

    #[test]
    fn unreachable_from_lists_orphaned_knowledge() {
        let mut context = Context::default();
        context
            .associate("青嶺酒造", "代表銘柄", "青嶺", 1.0)
            .unwrap();
        context.associate("高瀬", "役職", "杜氏", 1.0).unwrap(); // island

        let orphans = context
            .unreachable_from(&["青嶺酒造"], Deadline::unbounded())
            .unwrap();
        assert_eq!(orphans, vec![assoc("高瀬", "役職", "杜氏", 1.0)]);

        // The membership edge repairs the island; the audit comes back
        // clean.
        context.associate("青嶺酒造", "杜氏", "高瀬", 1.0).unwrap();
        assert!(
            context
                .unreachable_from(&["青嶺酒造"], Deadline::unbounded())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn unreachable_from_an_unknown_origin_reports_everything() {
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();

        assert_eq!(
            context
                .unreachable_from(&["存在しない概念"], Deadline::unbounded())
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            context
                .unreachable_from(&[], Deadline::unbounded())
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn unreachable_from_ignores_retracted_edges_on_both_sides_of_the_audit() {
        let mut context = Context::default();
        // 青嶺酒造 -- 杜氏 -- 高瀬 is the only path to 高瀬's island; once
        // retracted, 高瀬's remaining fact is a genuine orphan again — the
        // dead edge itself must not (a) stand in as a live bridge that
        // hides the orphan, nor (b) be reported as an "unreachable"
        // association in its own right, since it is no longer a fact.
        context
            .associate("青嶺酒造", "代表銘柄", "青嶺", 1.0)
            .unwrap();
        context.associate("高瀬", "役職", "杜氏", 1.0).unwrap();
        context.associate("青嶺酒造", "杜氏", "高瀬", 1.0).unwrap();
        assert!(
            context
                .unreachable_from(&["青嶺酒造"], Deadline::unbounded())
                .unwrap()
                .is_empty()
        );

        context
            .retract_association("青嶺酒造", "杜氏", "高瀬")
            .unwrap();
        let orphans = context
            .unreachable_from(&["青嶺酒造"], Deadline::unbounded())
            .unwrap();
        assert_eq!(orphans, vec![assoc("高瀬", "役職", "杜氏", 1.0)]);
    }

    #[test]
    fn adjacency_requires_a_real_edge_in_either_direction() {
        let mut context = Context::default();
        context.associate("a", "r", "b", 1.0).unwrap();
        context.associate("c", "r", "d", 1.0).unwrap();

        assert!(context.adjacent("a", "b"));
        assert!(context.adjacent("b", "a"));
        assert!(!context.adjacent("a", "c"));
        assert!(!context.adjacent("a", "unknown"));
    }

    #[test]
    fn adjacent_to_itself_requires_an_actual_self_loop() {
        let mut context = Context::default();
        context.associate("a", "r", "b", 1.0).unwrap();
        // `a` participates in a live edge, but not one that loops back to
        // itself — `outgoing("a")` and `incoming("a")` each hold that one
        // edge, and checking `subject == id_b || object == id_b` against
        // BOTH chains (rather than just the far endpoint each chain
        // implies) used to read that as `a` being adjacent to itself,
        // since `subject == id_a` is trivially true on its own outgoing
        // edge once `id_b == id_a`.
        assert!(!context.adjacent("a", "a"));

        context.associate("x", "r", "x", 1.0).unwrap();
        assert!(context.adjacent("x", "x"));
    }

    #[test]
    fn labels_share_a_subject_only_when_their_edges_do() {
        let mut context = Context::default();
        context.associate("shared", "r1", "a", 1.0).unwrap();
        context.associate("shared", "r2", "b", 1.0).unwrap();
        context.associate("other", "r3", "c", 1.0).unwrap();

        assert!(context.labels_share_subject("r1", "r2"));
        assert!(!context.labels_share_subject("r1", "r3"));
        assert!(!context.labels_share_subject("r1", "unknown"));
    }

    #[test]
    fn a_retracted_edge_is_no_longer_adjacent() {
        let mut context = Context::default();
        context.associate("a", "r", "b", 1.0).unwrap();
        assert!(context.adjacent("a", "b"));
        // Retracting the only edge tombstones it (count == 0) but leaves
        // it on the adjacency chain until compaction. A withdrawn fact
        // must stop counting as a live adjacency, or the vocabulary audit
        // reads it as RELATED evidence that no longer exists. (The Some
        // payload is the attribution count unlinked — 0 here, since this
        // edge carries sourceless weight — but the edge still goes dead.)
        assert!(context.retract_association("a", "r", "b").is_some());
        assert!(!context.adjacent("a", "b"));
        assert!(!context.adjacent("b", "a"));
    }

    #[test]
    fn a_retracted_edge_no_longer_shares_a_subject() {
        let mut context = Context::default();
        context.associate("shared", "r1", "a", 1.0).unwrap();
        context.associate("shared", "r2", "b", 1.0).unwrap();
        assert!(context.labels_share_subject("r1", "r2"));
        // Withdraw r1's only edge: the tombstone lingers on the subject's
        // label chain, but a dead edge must not make the labels look
        // co-occurrent (the audit would mis-read the fork as related).
        assert!(context.retract_association("shared", "r1", "a").is_some());
        assert!(!context.labels_share_subject("r1", "r2"));
    }

    #[test]
    fn activate_stops_propagating_below_the_activation_floor() {
        // A 30-edge chain of equal weights. From c1 onward every hop
        // multiplies the activation by decay/2 = 0.25, so it sinks below
        // the 1e-6 floor when settling c10: edges e0..e10 come back, the
        // 19 beyond are cut off instead of being walked to the end of the
        // component.
        let mut context = Context::default();
        for i in 0..30 {
            context
                .associate(format!("c{i}"), "r", format!("c{}", i + 1), 1.0)
                .unwrap();
        }

        let (_, ranked) = context.activate(&["c0"], 0.5, 100);
        assert_eq!(ranked.len(), 11);
        assert_eq!(ranked[0].association.subject, "c0");
        assert_eq!(ranked.last().unwrap().association.subject, "c10");

        // explore is the structural sweep and must stay unbounded.
        assert_eq!(context.explore(&["c0"], Context::UNBOUNDED).len(), 30);
    }
}
