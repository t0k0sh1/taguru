use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use crate::deadline::{Deadline, DeadlineExceeded};

use super::{
    Activation, Association, ConceptId, Context, EdgeId, EdgeRecord, LabelId, PathsResult,
    Recollection, Trail, accumulate_saturating, clamp_unit_or,
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
        self.explore_impl(origins, max_depth, |_| true)
    }

    /// [`Context::explore`] with a set of relation labels hidden from the
    /// walk entirely — never reported, never a bridge between the concepts
    /// it touches. This is ADR 0009 §6.3's traversal exclusion for the
    /// reserved `schema:type` label: a hub label (many concepts sharing
    /// one type) would otherwise put every instance within a couple of
    /// hops of every other, distorting a neighborhood search that has
    /// nothing to do with typing. The skip happens before a hidden edge's
    /// endpoints ever enter `node_distances`/`parents`, so — unlike a
    /// post-filter — it cannot leave a hidden label as the sole path
    /// connecting two results.
    ///
    /// Additive alongside [`Context::explore`] rather than a signature
    /// change, since `Context` is published API (ADR 0009 §6.3's own
    /// framing: exclusions must not touch a schema-free caller).
    /// [`Context::explore`] and this share [`Context::explore_impl`],
    /// monomorphized per call site on the `visible` closure — `explore`'s
    /// own instantiation passes the constant `|_| true`, which the
    /// compiler can see is unconditionally true and fold the whole check
    /// away, rather than a runtime-empty `HashSet` a single shared
    /// non-generic body would still have to branch on per edge. That
    /// distinction is not cosmetic: an earlier version filtered
    /// unconditionally and regressed `bench_activate`'s instruction count
    /// by several percent even with the exclusion set empty, purely from
    /// the extra iterator-adaptor/closure-call machinery in the hot loop.
    pub fn explore_excluding(
        &self,
        origins: &[&str],
        max_depth: usize,
        excluded: &[&str],
    ) -> Vec<Recollection> {
        let excluded_ids: HashSet<LabelId> = excluded
            .iter()
            .filter_map(|name| self.label_ids.get(*name).copied())
            .collect();
        self.explore_impl(origins, max_depth, move |label: LabelId| {
            !excluded_ids.contains(&label)
        })
    }

    fn explore_impl(
        &self,
        origins: &[&str],
        max_depth: usize,
        visible: impl Fn(LabelId) -> bool,
    ) -> Vec<Recollection> {
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
                if edge.count == 0 || !visible(edge.label) {
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
        self.activate_impl(origins, decay, limit, |_| true)
    }

    /// [`Context::activate`] with a set of relation labels hidden from
    /// both the fan-normalization total and the propagation/ranking
    /// walk — ADR 0009 §6.3's traversal exclusion for `schema:type`,
    /// covering the ranked sibling of [`Context::explore_excluding`].
    /// Filtering only the propagation loop and leaving the fan total
    /// alone would still let a heavily-typed concept's real facts be
    /// diluted by its hidden type edges — half an exclusion is exactly
    /// the "quietly distort unrelated features" ADR 0009 §6.3 rules out
    /// — so both chains are filtered identically. Additive alongside
    /// [`Context::activate`] for the same published-API reason
    /// [`Context::explore_excluding`] documents, and — like
    /// [`Context::explore`]/[`Context::explore_excluding`] sharing
    /// [`Context::explore_impl`] — the two share [`Context::activate_impl`],
    /// monomorphized per call so `activate`'s own instantiation compiles
    /// down to the exact pre-exclusion code: see `explore_impl`'s doc for
    /// why a single shared body checking a possibly-empty `HashSet` at
    /// runtime is not equivalent (it measurably regressed
    /// `bench_activate`'s instruction count even with nothing excluded).
    pub fn activate_excluding(
        &self,
        origins: &[&str],
        decay: f64,
        limit: usize,
        excluded: &[&str],
    ) -> (usize, Vec<Activation>) {
        let excluded_ids: HashSet<LabelId> = excluded
            .iter()
            .filter_map(|name| self.label_ids.get(*name).copied())
            .collect();
        self.activate_impl(origins, decay, limit, move |label: LabelId| {
            !excluded_ids.contains(&label)
        })
    }

    fn activate_impl(
        &self,
        origins: &[&str],
        decay: f64,
        limit: usize,
        visible: impl Fn(LabelId) -> bool,
    ) -> (usize, Vec<Activation>) {
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
                // test as `heaviest`, `describe`, and the export. A
                // hidden label (ADR 0009 §6.3) is excluded from the same
                // fold — it must not dilute a neighbor's share of real
                // facts, the same reason `activate_excluding` filters
                // the propagation loop below identically.
                .filter(|&edge_id| {
                    let edge = &self.edges[edge_id as usize];
                    edge.count > 0 && visible(edge.label)
                })
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
                .filter(|&edge_id| {
                    let edge = &self.edges[edge_id as usize];
                    edge.count > 0 && visible(edge.label)
                })
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

    /// Edge-expansion budget for one [`Context::paths`] call. Simple-path
    /// enumeration is combinatorial in the worst case (a dense clique
    /// holds factorially many simple paths between any two members), so
    /// unlike `explore` — whose work is bounded by the component's edge
    /// count — `paths` needs an explicit ceiling on how many edges the
    /// forward walk may examine. 100k expansions is far beyond any trail
    /// set a caller could digest while keeping the worst case bounded;
    /// hitting it is reported honestly as `capped` rather than silently
    /// returning a result that looks exhaustive.
    const PATHS_EXPANSION_BUDGET: usize = 100_000;

    /// Walks the network from `origins` to `targets` and returns up to
    /// `limit` simple paths connecting them — the 手繰り itself: not what
    /// lies near one concept, but through which concrete chain of
    /// associations two concepts relate, each hop carrying its full
    /// citation data.
    ///
    /// Traversal is structural, with exactly [`Context::explore`]'s
    /// discipline: edges are followed in both directions (an association
    /// is directional in meaning, not in reachability), shared relation
    /// labels never bridge, retracted edges never bridge, and origins or
    /// targets naming unknown concepts contribute nothing. Only simple
    /// paths are enumerated — no concept repeats within one trail — so a
    /// concept that is both origin and target yields no trail to itself
    /// (`distance` is never 0), while a trail may legitimately pass
    /// through one target on its way to another, and both prefixes are
    /// reported.
    ///
    /// Ranking is deterministic: distance ascending (the shortest thread
    /// is the strongest claim of relatedness), then weakest-link strength
    /// descending — see [`Trail::strength`] — then insertion order, the
    /// same final tie-break `explore` uses. `max_depth` bounds the hops
    /// per trail and `limit` bounds the trails returned; the
    /// pre-truncation count comes back as [`PathsResult::total`].
    /// Enumeration that would exceed [`Context::PATHS_EXPANSION_BUDGET`]
    /// stops early and says so via [`PathsResult::capped`].
    pub fn paths(
        &self,
        origins: &[&str],
        targets: &[&str],
        max_depth: usize,
        limit: usize,
    ) -> PathsResult {
        self.paths_impl(origins, targets, max_depth, limit, |_| true)
    }

    /// [`Context::paths`] with a set of relation labels hidden from the
    /// walk entirely — never a hop in a trail, never a bridge. ADR 0009
    /// §6.3's traversal exclusion for the reserved `schema:type` label,
    /// on the same reasoning as [`Context::explore_excluding`]: a hub
    /// label would put every typed instance a couple of hops from every
    /// other, and "A relates to B because both are typed 会社" is exactly
    /// the non-answer a paths query exists to avoid. Additive alongside
    /// [`Context::paths`] for the same published-API reason the other
    /// `_excluding` variants document, sharing [`Context::paths_impl`]
    /// monomorphized per call site on the `visible` closure (see
    /// [`Context::explore_excluding`] for why that is not cosmetic).
    pub fn paths_excluding(
        &self,
        origins: &[&str],
        targets: &[&str],
        max_depth: usize,
        limit: usize,
        excluded: &[&str],
    ) -> PathsResult {
        let excluded_ids: HashSet<LabelId> = excluded
            .iter()
            .filter_map(|name| self.label_ids.get(*name).copied())
            .collect();
        self.paths_impl(origins, targets, max_depth, limit, move |label: LabelId| {
            !excluded_ids.contains(&label)
        })
    }

    fn paths_impl(
        &self,
        origins: &[&str],
        targets: &[&str],
        max_depth: usize,
        limit: usize,
        visible: impl Fn(LabelId) -> bool,
    ) -> PathsResult {
        let empty = PathsResult {
            total: 0,
            capped: false,
            trails: Vec::new(),
        };
        let target_ids: HashSet<ConceptId> = targets
            .iter()
            .filter_map(|name| self.concept_ids.get(*name).copied())
            .collect();
        // Origins keep caller order (deduplicated) so enumeration order —
        // which decides WHAT gets found when the budget bites — is a
        // function of the request, not of hash iteration.
        let mut seen_origins: HashSet<ConceptId> = HashSet::new();
        let origin_ids: Vec<ConceptId> = origins
            .iter()
            .filter_map(|name| self.concept_ids.get(*name).copied())
            .filter(|&id| seen_origins.insert(id))
            .collect();
        if origin_ids.is_empty() || target_ids.is_empty() || max_depth == 0 || limit == 0 {
            return empty;
        }

        // Reverse breadth-first sweep from every target: each node's
        // distance to its NEAREST target, the admissible bound that lets
        // the forward walk prune any branch that can no longer reach a
        // target within `max_depth`. Without this the walk would wander
        // the whole component before discovering a branch is hopeless.
        let mut to_target: HashMap<ConceptId, usize> = HashMap::new();
        let mut frontier: VecDeque<ConceptId> = VecDeque::new();
        for &id in &target_ids {
            to_target.insert(id, 0);
            frontier.push_back(id);
        }
        while let Some(concept) = frontier.pop_front() {
            let hop = to_target[&concept] + 1;
            if hop > max_depth {
                continue;
            }
            for edge_id in self.outgoing(concept).chain(self.incoming(concept)) {
                let edge = &self.edges[edge_id as usize];
                if edge.count == 0 || !visible(edge.label) {
                    continue;
                }
                for neighbor in [edge.subject, edge.object] {
                    if let Entry::Vacant(entry) = to_target.entry(neighbor) {
                        entry.insert(hop);
                        frontier.push_back(neighbor);
                    }
                }
            }
        }

        // Depth-first enumeration of simple paths, pruned by the reverse
        // distances. Iterative — with the budget rather than `max_depth`
        // as the real recursion bound, a deep chain could otherwise grow
        // the call stack past its limit.
        let mut found: Vec<(Vec<ConceptId>, Vec<EdgeId>)> = Vec::new();
        let mut capped = false;
        let mut budget = Self::PATHS_EXPANSION_BUDGET;
        let mut node_stack: Vec<ConceptId> = Vec::new();
        let mut edge_stack: Vec<EdgeId> = Vec::new();
        let mut on_path: HashSet<ConceptId> = HashSet::new();
        'origins: for &origin in &origin_ids {
            if !to_target.contains_key(&origin) {
                continue;
            }
            node_stack.clear();
            edge_stack.clear();
            on_path.clear();
            node_stack.push(origin);
            on_path.insert(origin);
            let mut frames = Vec::new();
            frames.push(self.outgoing(origin).chain(self.incoming(origin)));
            while let Some(frame) = frames.last_mut() {
                let Some(edge_id) = frame.next() else {
                    frames.pop();
                    if let Some(done) = node_stack.pop() {
                        on_path.remove(&done);
                    }
                    edge_stack.pop();
                    continue;
                };
                if budget == 0 {
                    capped = true;
                    break 'origins;
                }
                budget -= 1;
                let edge = &self.edges[edge_id as usize];
                // Same dead-edge and hidden-label discipline as `explore`:
                // a retracted association is not a fact, a hidden label is
                // invisible — neither may be a hop, and skipping here (not
                // post-filtering) means neither can bridge either.
                if edge.count == 0 || !visible(edge.label) {
                    continue;
                }
                let here = *node_stack.last().expect("stack tracks frames");
                let next = if edge.subject == here {
                    edge.object
                } else {
                    edge.subject
                };
                // A self-loop leads nowhere new, and revisiting a concept
                // already on the trail would make the path non-simple.
                if next == here || on_path.contains(&next) {
                    continue;
                }
                let depth = edge_stack.len() + 1;
                let Some(&remaining) = to_target.get(&next) else {
                    continue;
                };
                if depth + remaining > max_depth {
                    continue;
                }
                if target_ids.contains(&next) {
                    let mut nodes = node_stack.clone();
                    nodes.push(next);
                    let mut edges = edge_stack.clone();
                    edges.push(edge_id);
                    found.push((nodes, edges));
                }
                if depth < max_depth {
                    node_stack.push(next);
                    on_path.insert(next);
                    edge_stack.push(edge_id);
                    frames.push(self.outgoing(next).chain(self.incoming(next)));
                }
            }
        }

        // Rank: distance, then weakest link descending, then ids — which
        // are insertion order, `explore`'s own same-distance tie-break —
        // so the order is deterministic for a given `Context` history.
        let mut ranked: Vec<(usize, f64, Vec<ConceptId>, Vec<EdgeId>)> = found
            .into_iter()
            .map(|(nodes, edges)| {
                let strength = edges
                    .iter()
                    .map(|&edge_id| self.edges[edge_id as usize].sum.abs())
                    .fold(f64::INFINITY, f64::min);
                (edges.len(), strength, nodes, edges)
            })
            .collect();
        ranked.sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| b.1.total_cmp(&a.1))
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.3.cmp(&b.3))
        });
        let total = ranked.len();
        ranked.truncate(limit);

        let trails = ranked
            .into_iter()
            .map(|(distance, strength, nodes, edges)| Trail {
                distance,
                path: nodes
                    .into_iter()
                    .map(|id| self.concept_name(id).to_string())
                    .collect(),
                strength,
                associations: edges
                    .into_iter()
                    .map(|edge_id| self.association(edge_id))
                    .collect(),
            })
            .collect();
        PathsResult {
            total,
            capped,
            trails,
        }
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
        self.unreachable_from_impl(origins, deadline, |_| true)
    }

    /// [`Context::unreachable_from`] with a set of relation labels hidden
    /// from the audit entirely — never a bridge in the reachability walk,
    /// never reported as an orphan. This is ADR 0009 §6.3's traversal
    /// exclusion applied to the coverage audit, for the same reason
    /// [`Context::explore_excluding`] exists: a hub label (many concepts
    /// sharing one type) would otherwise connect every typed instance to
    /// every other, silently under-reporting facts that are genuinely
    /// disconnected from the origins in every non-typing sense. The
    /// report side is excluded too — a hidden edge is invisible to this
    /// audit, not a fact it should flag — mirroring `explore_excluding`'s
    /// "never reported, never a bridge" contract rather than inventing a
    /// third semantics.
    ///
    /// Additive alongside [`Context::unreachable_from`] rather than a
    /// signature change, since `Context` is published API — the same
    /// framing as [`Context::explore_excluding`], with which it also
    /// shares the monomorphized-`visible`-closure structure (the
    /// unfiltered caller's `|_| true` folds away entirely).
    pub fn unreachable_from_excluding(
        &self,
        origins: &[&str],
        deadline: Deadline,
        excluded: &[&str],
    ) -> Result<Vec<Association>, DeadlineExceeded> {
        let excluded_ids: HashSet<LabelId> = excluded
            .iter()
            .filter_map(|name| self.label_ids.get(*name).copied())
            .collect();
        self.unreachable_from_impl(origins, deadline, move |label: LabelId| {
            !excluded_ids.contains(&label)
        })
    }

    fn unreachable_from_impl(
        &self,
        origins: &[&str],
        deadline: Deadline,
        visible: impl Fn(LabelId) -> bool,
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
                if edge.count == 0 || !visible(edge.label) {
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
        // are no longer facts at all, and hidden-label edges are invisible
        // to the audit, so both are excluded rather than reported as
        // unreachable ones.
        let mut out = Vec::new();
        for edge_id in 0..self.edges.len() as u32 {
            if deadline.expired() {
                return Err(DeadlineExceeded);
            }
            let edge = &self.edges[edge_id as usize];
            if edge.count > 0 && visible(edge.label) && !visited.contains(&edge.subject) {
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
    fn paths_pulls_the_thread_end_to_end() {
        let mut context = Context::default();
        // A chain: 私 → りんご → 果物 → ビタミン
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        context.associate("りんご", "分類", "果物", 1.0).unwrap();
        context.associate("果物", "含む", "ビタミン", 1.0).unwrap();

        let direct = context.paths(&["私"], &["りんご"], 10, 10);
        assert!(!direct.capped);
        assert_eq!(direct.total, 1);
        assert_eq!(direct.trails.len(), 1);
        assert_eq!(direct.trails[0].distance, 1);
        assert_eq!(direct.trails[0].path, vec!["私", "りんご"]);
        assert_eq!(
            direct.trails[0].associations,
            vec![assoc("私", "好き", "りんご", 1.0)]
        );

        let far = context.paths(&["私"], &["ビタミン"], 10, 10);
        assert_eq!(far.total, 1);
        assert_eq!(far.trails[0].distance, 3);
        assert_eq!(far.trails[0].path, vec!["私", "りんご", "果物", "ビタミン"]);
        assert_eq!(far.trails[0].associations.len(), 3);
        // The trail's associations come back in walk order.
        assert_eq!(far.trails[0].associations[0].subject, "私");
        assert_eq!(far.trails[0].associations[2].object, "ビタミン");

        // A depth ceiling below the shortest connection finds nothing.
        assert_eq!(context.paths(&["私"], &["ビタミン"], 2, 10).total, 0);
    }

    #[test]
    fn paths_walks_against_edge_direction_too() {
        let mut context = Context::default();
        // りんご is the *object* of both edges; connecting 私 to 農家 runs
        // against 育てる's direction on the second hop.
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        context.associate("農家", "育てる", "りんご", 1.0).unwrap();

        let result = context.paths(&["私"], &["農家"], 10, 10);
        assert_eq!(result.total, 1);
        assert_eq!(result.trails[0].path, vec!["私", "りんご", "農家"]);
    }

    #[test]
    fn paths_orders_shorter_trails_before_stronger_long_ones() {
        let mut context = Context::default();
        // A weak direct edge and a heavyweight detour: the direct thread
        // still comes first — distance ranks before strength.
        context.associate("起点", "弱い関係", "目標", 0.5).unwrap();
        context.associate("起点", "強い関係", "中継", 5.0).unwrap();
        context.associate("中継", "強い関係", "目標", 5.0).unwrap();

        let result = context.paths(&["起点"], &["目標"], 10, 10);
        assert_eq!(result.total, 2);
        assert_eq!(result.trails[0].distance, 1);
        assert_eq!(result.trails[0].strength, 0.5);
        assert_eq!(result.trails[1].distance, 2);
        assert_eq!(result.trails[1].strength, 5.0);
    }

    #[test]
    fn paths_ranks_equal_length_trails_by_their_weakest_link() {
        let mut context = Context::default();
        // Two 2-hop routes; the first is weaker overall despite one strong
        // hop, because a chain is only as reliable as its weakest link.
        context.associate("起点", "r", "弱い中継", 1.0).unwrap();
        context.associate("弱い中継", "r", "目標", 5.0).unwrap();
        context.associate("起点", "r", "強い中継", 3.0).unwrap();
        context.associate("強い中継", "r", "目標", 5.0).unwrap();

        let result = context.paths(&["起点"], &["目標"], 10, 10);
        assert_eq!(result.total, 2);
        assert_eq!(result.trails[0].path[1], "強い中継");
        assert_eq!(result.trails[0].strength, 3.0);
        assert_eq!(result.trails[1].path[1], "弱い中継");
        assert_eq!(result.trails[1].strength, 1.0);
    }

    #[test]
    fn paths_strength_rewards_corroboration_over_average_weight() {
        let mut context = Context::default();
        // The corroborated hop sums to 2.0 (two assertions of 1.0) while
        // its averaged weight stays 1.0; the single emphatic 1.5 has the
        // higher average. Ranking on the raw sum keeps corroboration
        // ahead — the same discipline activate documents.
        context.associate("起点", "r", "裏取り", 1.0).unwrap();
        context.associate("起点", "r", "裏取り", 1.0).unwrap();
        context.associate("裏取り", "r", "目標", 5.0).unwrap();
        context.associate("起点", "r", "単発", 1.5).unwrap();
        context.associate("単発", "r", "目標", 5.0).unwrap();

        let result = context.paths(&["起点"], &["目標"], 10, 10);
        assert_eq!(result.total, 2);
        assert_eq!(result.trails[0].path[1], "裏取り");
        assert_eq!(result.trails[0].strength, 2.0);
        assert_eq!(result.trails[1].strength, 1.5);
    }

    #[test]
    fn paths_does_not_bridge_through_a_retracted_edge() {
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        context.associate("りんご", "分類", "果物", 1.0).unwrap();
        context
            .retract_association("りんご", "分類", "果物")
            .unwrap();

        assert_eq!(context.paths(&["私"], &["果物"], 10, 10).total, 0);
    }

    #[test]
    fn paths_enumerates_simple_paths_and_reports_total_past_the_limit() {
        let mut context = Context::default();
        // A diamond: exactly two simple paths, never an a↔b zigzag.
        context.associate("o", "r", "a", 1.0).unwrap();
        context.associate("o", "r", "b", 1.0).unwrap();
        context.associate("a", "r", "t", 1.0).unwrap();
        context.associate("b", "r", "t", 1.0).unwrap();

        let all = context.paths(&["o"], &["t"], 10, 10);
        assert_eq!(all.total, 2);
        assert_eq!(all.trails.len(), 2);

        let cut = context.paths(&["o"], &["t"], 10, 1);
        assert_eq!(cut.total, 2, "total still counts what the limit cut");
        assert_eq!(cut.trails.len(), 1);
    }

    #[test]
    fn paths_records_the_prefix_when_a_trail_passes_through_a_target() {
        let mut context = Context::default();
        context.associate("o", "r", "t1", 1.0).unwrap();
        context.associate("t1", "r", "t2", 1.0).unwrap();

        let result = context.paths(&["o"], &["t1", "t2"], 10, 10);
        assert_eq!(result.total, 2);
        assert_eq!(result.trails[0].path, vec!["o", "t1"]);
        assert_eq!(result.trails[1].path, vec!["o", "t1", "t2"]);
    }

    #[test]
    fn paths_returns_nothing_for_unknowns_zero_depth_or_self() {
        let mut context = Context::default();
        context.associate("a", "r", "b", 1.0).unwrap();

        assert_eq!(context.paths(&["未知"], &["b"], 10, 10).total, 0);
        assert_eq!(context.paths(&["a"], &["未知"], 10, 10).total, 0);
        assert_eq!(context.paths(&["a"], &["b"], 0, 10).total, 0);
        assert_eq!(context.paths(&["a"], &["b"], 10, 0).trails.len(), 0);
        // A concept is never connected to itself by an empty trail, and a
        // cycle back to the start is not a simple path.
        assert_eq!(context.paths(&["a"], &["a"], 10, 10).total, 0);
    }

    #[test]
    fn paths_excluding_hides_the_label_as_hop_and_bridge() {
        let mut context = Context::default();
        // The only connection runs over the hidden label — once as the
        // sole hop, once as the middle of a longer thread.
        context.associate("a", "schema:type", "会社", 1.0).unwrap();
        context.associate("b", "schema:type", "会社", 1.0).unwrap();

        assert_eq!(context.paths(&["a"], &["b"], 10, 10).total, 1);
        let excluded = context.paths_excluding(&["a"], &["b"], 10, 10, &["schema:type"]);
        assert_eq!(
            excluded.total, 0,
            "a hidden label must be neither a hop nor a bridge"
        );
    }

    #[test]
    fn paths_reports_capped_when_enumeration_exhausts_the_budget() {
        // A 12-clique holds millions of simple paths between any two
        // members; enumeration must stop at the budget and say so instead
        // of pretending the answer is complete.
        let mut context = Context::default();
        let names: Vec<String> = (0..12).map(|i| format!("n{i}")).collect();
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                context.associate(&names[i], "r", &names[j], 1.0).unwrap();
            }
        }

        let result = context.paths(&["n0"], &["n11"], Context::UNBOUNDED, 5);
        assert!(result.capped, "a clique walk must hit the budget");
        assert_eq!(result.trails.len(), 5);
        assert!(result.total >= 5);
        // Whatever was found before the budget bit still ranks shortest
        // first.
        assert!(
            result
                .trails
                .windows(2)
                .all(|w| w[0].distance <= w[1].distance)
        );
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
