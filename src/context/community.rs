//! Community detection over the concept graph — the analysis substrate
//! behind `GET /contexts/{name}/communities` and, through it, the
//! `taguru communities` derivation flow (issue #166).
//!
//! The graph communities partition is the CONCEPT graph: concepts are
//! the nodes, live associations the edges, and an edge's strength is
//! `|sum|` — the same magnitude-not-sign convention `activate` ranks
//! by, because a firmly contested relation still binds its endpoints
//! into one topic. Labels stay off the node set on purpose (two facts
//! sharing a label must not become one community through it — the same
//! anti-leakage rule the traversal verbs follow); they ride along on
//! the induced edges, into fingerprints and summary prompts.
//!
//! The algorithm is hand-rolled deterministic Louvain plus a
//! connected-component split after every level — `louvain-cc/1`:
//!
//! - **Louvain, not Leiden, not label propagation.** Greedy modularity
//!   gives usable topic clusters and its aggregation levels ARE the
//!   hierarchy the summaries want — no second mechanism. Label
//!   propagation is cheaper but oscillates and carries no hierarchy;
//!   full Leiden's refinement phase buys guaranteed well-connectedness
//!   at ~3× the code. The component split below closes Louvain's one
//!   structural artifact (internally disconnected communities) for the
//!   price of a BFS, which is the part of Leiden's guarantee that
//!   matters for summarization.
//! - **Deterministic by construction, matching the codebase's
//!   conventions**: nodes are visited in dense-id order, the best move
//!   ties toward the lower community id (the `Candidate` convention),
//!   community ids renumber by first member, and no RNG exists
//!   anywhere. Same graph in, same communities out — CI asserts on
//!   fixtures, and re-runs only re-summarize what actually changed.
//! - **Fingerprints are the incremental contract.** Every community
//!   carries an FNV-1a digest of its content (leaf: sorted members +
//!   sorted induced live edges with their signed sums and counts;
//!   parent: sorted child digests). The derivation CLI re-summarizes a
//!   community only when its fingerprint moved — detection is cheap
//!   and re-runs whole, LLM calls are the expensive part and re-run
//!   only for changed content.
//!
//! A full pass over every edge can run long on a big context, so the
//! entry point takes a [`Deadline`] and answers `DeadlineExceeded`
//! mid-flight, the same contract as `unreachable_from` and the audit
//! sweeps.

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;

use crate::deadline::{Deadline, DeadlineExceeded};
use crate::hash::{FNV1A_OFFSET, fnv1a_fold};

use super::{ConceptId, Context, accumulate_saturating};

/// Algorithm identifier recorded in every analysis and every derived
/// manifest. Bump the suffix on any change that can alter the
/// partition — a stale artifact built by an older algorithm should
/// say so rather than silently diff against incomparable fingerprints.
pub const COMMUNITY_ALGORITHM: &str = "louvain-cc/1";

/// Ceiling on aggregation levels. Louvain converges in a handful of
/// levels on real graphs; this is a runaway backstop, not a tunable.
const MAX_LEVELS: usize = 8;

/// Induced associations reported per leaf community, strongest first —
/// sized for a summary prompt, not for completeness (the full set is
/// reachable through `query` on the source context).
const TOP_ASSOCIATIONS_PER_COMMUNITY: usize = 24;

/// How many node visits go between deadline checks inside the moving
/// passes — one `Instant` compare per node would be noise, one per
/// pass too coarse to honor a deadline on a huge level.
const DEADLINE_STRIDE: usize = 1024;

/// One detected partition of a context's concept graph: every level's
/// communities in one flat list, leaves first.
#[derive(Debug, Serialize)]
pub struct CommunityAnalysis {
    /// [`COMMUNITY_ALGORITHM`] — recorded so a derived artifact can
    /// refuse to diff fingerprints across algorithm versions.
    pub algorithm: &'static str,
    /// Concepts that entered detection: at least one live,
    /// non-self-loop association. Isolated concepts form no community.
    pub concept_count: usize,
    /// Undirected concept pairs after collapsing parallel labels —
    /// detection's edge count, not the association count.
    pub edge_count: usize,
    /// Hierarchy depth. 0 means the graph had nothing to cluster.
    pub levels: usize,
    pub communities: Vec<Community>,
}

/// One community at one level. Leaves (`level == 0`) carry concept
/// members and their strongest induced associations; parents carry
/// child community ids instead — membership edges are written once, at
/// the leaves, and the hierarchy is edges between community records.
#[derive(Debug, Serialize)]
pub struct Community {
    /// `"L{level}-{index}"` — stable only within one analysis; the
    /// fingerprint, not the id, is the cross-run identity.
    pub id: String,
    pub level: usize,
    /// Id of the community one level up that contains this one, absent
    /// on the top level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// FNV-1a content digest, `{:016x}` — the incremental-derivation
    /// key: equal fingerprint means equal summarizable content.
    pub fingerprint: String,
    /// Leaf concepts under this community, transitively — a parent's
    /// count sums its children's.
    pub concept_count: usize,
    /// Leaf only: member concepts, strongest first (strength = the
    /// member's share of intra-community edge weight).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub members: Vec<CommunityMember>,
    /// Parent only: ids of the communities one level down.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<String>,
    /// Leaf only: the strongest live associations with both endpoints
    /// inside the community — the summary prompt's raw material.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub top_associations: Vec<CommunityAssociation>,
}

#[derive(Debug, Serialize)]
pub struct CommunityMember {
    pub name: String,
    /// Sum of `|sum|` over the member's live intra-community edges —
    /// ranks the community's central concepts first, and becomes the
    /// membership edge's weight in a derived artifact.
    pub strength: f64,
}

/// One induced association, without the attribution chain — summary
/// prompts and analysis payloads want the fact, not its provenance
/// (the source context still has the full [`super::Association`]).
#[derive(Debug, Serialize)]
pub struct CommunityAssociation {
    pub subject: String,
    pub label: String,
    pub object: String,
    pub weight: f64,
    pub count: u64,
}

/// The working graph one Louvain level runs on: undirected weighted
/// adjacency over dense node indices, plus per-node self-loop weight
/// (zero at the leaf level, the aggregated internal weight above it).
/// Neighbor lists are sorted by node index — determinism leans on it.
struct WorkGraph {
    neighbors: Vec<Vec<(usize, f64)>>,
    self_weight: Vec<f64>,
}

impl WorkGraph {
    fn len(&self) -> usize {
        self.neighbors.len()
    }

    /// A node's weighted degree, self-loop counted twice — the Louvain
    /// convention that keeps modularity's `2m` normalization honest.
    fn strength(&self, node: usize) -> f64 {
        let mut total = 2.0 * self.self_weight[node];
        for &(_, weight) in &self.neighbors[node] {
            accumulate_saturating(&mut total, weight);
        }
        total
    }

    /// Total edge weight `m` (each undirected pair once, self-loops
    /// once) — the modularity denominator.
    fn total_weight(&self) -> f64 {
        let mut total = 0.0;
        for (node, list) in self.neighbors.iter().enumerate() {
            accumulate_saturating(&mut total, self.self_weight[node]);
            for &(neighbor, weight) in list {
                if neighbor > node {
                    accumulate_saturating(&mut total, weight);
                }
            }
        }
        total
    }
}

impl Context {
    /// Detects the concept graph's communities: deterministic Louvain
    /// with a component split per level, hierarchical via aggregation.
    /// See the module doc for the algorithm contract; see
    /// [`CommunityAnalysis`] for the result shape. An empty or
    /// edge-free context answers an empty analysis rather than an
    /// error — no communities is a true statement about it.
    pub fn communities(&self, deadline: Deadline) -> Result<CommunityAnalysis, DeadlineExceeded> {
        let (nodes, leaf_graph) = self.leaf_graph(deadline)?;
        let edge_count = leaf_graph
            .neighbors
            .iter()
            .map(|list| list.len())
            .sum::<usize>()
            / 2;
        if nodes.is_empty() {
            return Ok(CommunityAnalysis {
                algorithm: COMMUNITY_ALGORITHM,
                concept_count: 0,
                edge_count: 0,
                levels: 0,
                communities: Vec::new(),
            });
        }

        // One assignment per level: level_assignments[k] maps a
        // level-k node (leaf concepts at k == 0, communities above)
        // to its community index one level up the loop's ladder.
        let mut level_assignments: Vec<Vec<usize>> = Vec::new();
        let mut graph = leaf_graph;
        while level_assignments.len() < MAX_LEVELS {
            let (assignment, community_count) = one_level(&graph, deadline)?;
            if community_count == graph.len() {
                // Nothing merged — the ladder has converged; an
                // identity level would add nothing but noise.
                break;
            }
            graph = aggregate(&graph, &assignment, community_count);
            level_assignments.push(assignment);
            if community_count == 1 {
                break;
            }
        }
        if level_assignments.is_empty() {
            // Every concept stayed alone: a graph of disconnected
            // pairs' worth of structure. Treat each node as its own
            // leaf community so the artifact still names them.
            level_assignments.push((0..nodes.len()).collect());
        }

        let communities = self.assemble(&nodes, &level_assignments, deadline)?;
        Ok(CommunityAnalysis {
            algorithm: COMMUNITY_ALGORITHM,
            concept_count: nodes.len(),
            edge_count,
            levels: level_assignments.len(),
            communities,
        })
    }

    /// Builds the leaf working graph: live (`count > 0`),
    /// non-self-loop, non-netted (`|sum| > 0`) edges, collapsed to
    /// undirected concept pairs with parallel labels' magnitudes
    /// summed. Returns the dense-index → `ConceptId` table alongside.
    fn leaf_graph(
        &self,
        deadline: Deadline,
    ) -> Result<(Vec<ConceptId>, WorkGraph), DeadlineExceeded> {
        let mut merged: HashMap<(ConceptId, ConceptId), f64> = HashMap::new();
        for (index, edge) in self.edges.iter().enumerate() {
            if index.is_multiple_of(DEADLINE_STRIDE) && deadline.expired() {
                return Err(DeadlineExceeded);
            }
            if edge.count == 0 || edge.subject == edge.object {
                continue;
            }
            let magnitude = edge.sum.abs();
            if magnitude == 0.0 {
                continue;
            }
            let pair = (edge.subject.min(edge.object), edge.subject.max(edge.object));
            accumulate_saturating(merged.entry(pair).or_insert(0.0), magnitude);
        }

        let mut concept_ids: Vec<ConceptId> = merged.keys().flat_map(|&(a, b)| [a, b]).collect();
        concept_ids.sort_unstable();
        concept_ids.dedup();
        let index_of: HashMap<ConceptId, usize> = concept_ids
            .iter()
            .enumerate()
            .map(|(index, &concept)| (concept, index))
            .collect();

        let mut pairs: Vec<((ConceptId, ConceptId), f64)> = merged.into_iter().collect();
        pairs.sort_unstable_by_key(|&(pair, _)| pair);
        let mut neighbors = vec![Vec::new(); concept_ids.len()];
        for ((a, b), weight) in pairs {
            let a = index_of[&a];
            let b = index_of[&b];
            neighbors[a].push((b, weight));
            neighbors[b].push((a, weight));
        }
        for list in &mut neighbors {
            list.sort_unstable_by_key(|&(neighbor, _)| neighbor);
        }
        let self_weight = vec![0.0; concept_ids.len()];
        Ok((
            concept_ids,
            WorkGraph {
                neighbors,
                self_weight,
            },
        ))
    }

    /// Materializes the level assignments into the flat, leaves-first
    /// community list: names, strengths, induced associations, and
    /// fingerprints.
    fn assemble(
        &self,
        nodes: &[ConceptId],
        level_assignments: &[Vec<usize>],
        deadline: Deadline,
    ) -> Result<Vec<Community>, DeadlineExceeded> {
        // A leaf concept's community index at every level: fold the
        // assignment ladder upward once so parent membership needs no
        // repeated walks.
        let levels = level_assignments.len();
        let mut per_level: Vec<Vec<usize>> = Vec::with_capacity(levels);
        per_level.push(level_assignments[0].clone());
        for assignment in &level_assignments[1..] {
            let previous = per_level.last().expect("pushed above");
            per_level.push(previous.iter().map(|&c| assignment[c]).collect());
        }
        let count_at = |level: usize| -> usize {
            per_level[level]
                .iter()
                .copied()
                .max()
                .map_or(0, |max| max + 1)
        };

        // Leaf members and strengths. Strength sums the member's live
        // intra-community magnitudes — self-loops included here (they
        // are content about the member) even though detection ignored
        // them for topology.
        let index_of: HashMap<ConceptId, usize> = nodes
            .iter()
            .enumerate()
            .map(|(index, &concept)| (concept, index))
            .collect();
        let leaf_count = count_at(0);
        let mut members: Vec<Vec<(ConceptId, f64)>> = vec![Vec::new(); leaf_count];
        for (index, &concept) in nodes.iter().enumerate() {
            members[per_level[0][index]].push((concept, 0.0));
        }
        let mut induced: Vec<Vec<(u32, f64, u64)>> = vec![Vec::new(); leaf_count];
        for (edge_index, edge) in self.edges.iter().enumerate() {
            if edge_index.is_multiple_of(DEADLINE_STRIDE) && deadline.expired() {
                return Err(DeadlineExceeded);
            }
            if edge.count == 0 {
                continue;
            }
            let (Some(&subject), Some(&object)) =
                (index_of.get(&edge.subject), index_of.get(&edge.object))
            else {
                continue;
            };
            let community = per_level[0][subject];
            if per_level[0][object] != community {
                continue;
            }
            induced[community].push((edge_index as u32, edge.sum, edge.count));
        }
        for (community, list) in induced.iter().enumerate() {
            let mut strength: HashMap<ConceptId, f64> = HashMap::new();
            for &(edge_id, sum, _) in list {
                let edge = &self.edges[edge_id as usize];
                accumulate_saturating(strength.entry(edge.subject).or_insert(0.0), sum.abs());
                if edge.subject != edge.object {
                    accumulate_saturating(strength.entry(edge.object).or_insert(0.0), sum.abs());
                }
            }
            for member in &mut members[community] {
                member.1 = strength.get(&member.0).copied().unwrap_or(0.0);
            }
        }

        let mut communities = Vec::new();
        // Leaves: members strongest-first (name-tied for determinism),
        // induced associations strongest-first, fingerprint over the
        // sorted content.
        let mut leaf_fingerprints = Vec::with_capacity(leaf_count);
        for (community, mut member_list) in members.into_iter().enumerate() {
            if deadline.expired() {
                return Err(DeadlineExceeded);
            }
            member_list.sort_by(|a, b| {
                b.1.total_cmp(&a.1)
                    .then_with(|| self.concept_name(a.0).cmp(self.concept_name(b.0)))
            });

            let mut edges: Vec<&(u32, f64, u64)> = induced[community].iter().collect();
            edges.sort_by(|a, b| {
                let ea = &self.edges[a.0 as usize];
                let eb = &self.edges[b.0 as usize];
                self.concept_name(ea.subject)
                    .cmp(self.concept_name(eb.subject))
                    .then_with(|| self.label_name(ea.label).cmp(self.label_name(eb.label)))
                    .then_with(|| {
                        self.concept_name(ea.object)
                            .cmp(self.concept_name(eb.object))
                    })
            });
            let mut digest = FNV1A_OFFSET;
            let mut names: Vec<&str> = member_list
                .iter()
                .map(|&(concept, _)| self.concept_name(concept))
                .collect();
            names.sort_unstable();
            for name in names {
                digest = fnv1a_fold(digest, (name.len() as u64).to_le_bytes());
                digest = fnv1a_fold(digest, name.bytes());
            }
            for &&(edge_id, sum, count) in &edges {
                let edge = &self.edges[edge_id as usize];
                // Canonicalize the arithmetic zero: -0.0 and 0.0 mean
                // the same netted edge and must not split fingerprints.
                let sum = if sum == 0.0 { 0.0 } else { sum };
                for name in [
                    self.concept_name(edge.subject),
                    self.label_name(edge.label),
                    self.concept_name(edge.object),
                ] {
                    digest = fnv1a_fold(digest, (name.len() as u64).to_le_bytes());
                    digest = fnv1a_fold(digest, name.bytes());
                }
                digest = fnv1a_fold(digest, sum.to_bits().to_le_bytes());
                digest = fnv1a_fold(digest, count.to_le_bytes());
            }
            leaf_fingerprints.push(digest);

            let mut top: Vec<&(u32, f64, u64)> = induced[community]
                .iter()
                .filter(|(_, sum, _)| sum.abs() > 0.0)
                .collect();
            top.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()).then_with(|| a.0.cmp(&b.0)));
            top.truncate(TOP_ASSOCIATIONS_PER_COMMUNITY);

            communities.push(Community {
                id: format!("L0-{community}"),
                level: 0,
                parent: (levels > 1).then(|| format!("L1-{}", level_assignments[1][community])),
                fingerprint: format!("{digest:016x}"),
                concept_count: member_list.len(),
                members: member_list
                    .into_iter()
                    .map(|(concept, strength)| CommunityMember {
                        name: self.concept_name(concept).to_string(),
                        strength,
                    })
                    .collect(),
                children: Vec::new(),
                top_associations: top
                    .into_iter()
                    .map(|&(edge_id, sum, count)| {
                        let edge = &self.edges[edge_id as usize];
                        CommunityAssociation {
                            subject: self.concept_name(edge.subject).to_string(),
                            label: self.label_name(edge.label).to_string(),
                            object: self.concept_name(edge.object).to_string(),
                            weight: sum / count as f64,
                            count,
                        }
                    })
                    .collect(),
            });
        }

        // Parents: children by id order, concept counts summed
        // transitively, fingerprint over sorted child digests — a
        // moved leaf changes every ancestor's digest, which is exactly
        // when their summaries go stale.
        let mut fingerprints = leaf_fingerprints;
        for level in 1..levels {
            let count = count_at(level);
            let below = count_at(level - 1);
            let assignment = &level_assignments[level];
            let mut children: Vec<Vec<usize>> = vec![Vec::new(); count];
            for child in 0..below {
                children[assignment[child]].push(child);
            }
            let mut concept_counts = vec![0usize; count];
            for &concept_community in &per_level[level] {
                concept_counts[concept_community] += 1;
            }
            let mut level_fingerprints = Vec::with_capacity(count);
            for (community, child_list) in children.iter().enumerate() {
                let mut digests: Vec<u64> = child_list
                    .iter()
                    .map(|&child| fingerprints[child])
                    .collect();
                digests.sort_unstable();
                let mut digest = FNV1A_OFFSET;
                digest = fnv1a_fold(digest, (digests.len() as u64).to_le_bytes());
                for child_digest in digests {
                    digest = fnv1a_fold(digest, child_digest.to_le_bytes());
                }
                level_fingerprints.push(digest);
                communities.push(Community {
                    id: format!("L{level}-{community}"),
                    level,
                    parent: (level + 1 < levels).then(|| {
                        format!("L{}-{}", level + 1, level_assignments[level + 1][community])
                    }),
                    fingerprint: format!("{digest:016x}"),
                    concept_count: concept_counts[community],
                    members: Vec::new(),
                    children: child_list
                        .iter()
                        .map(|&child| format!("L{}-{child}", level - 1))
                        .collect(),
                    top_associations: Vec::new(),
                });
            }
            fingerprints = level_fingerprints;
        }
        Ok(communities)
    }
}

/// One Louvain level: greedy modularity moves to convergence, then the
/// component split. Returns each node's community (dense, renumbered
/// by first member) and the community count.
fn one_level(
    graph: &WorkGraph,
    deadline: Deadline,
) -> Result<(Vec<usize>, usize), DeadlineExceeded> {
    let n = graph.len();
    let m = graph.total_weight();
    if m == 0.0 {
        return Ok(((0..n).collect(), n));
    }
    let strengths: Vec<f64> = (0..n).map(|node| graph.strength(node)).collect();
    let mut community: Vec<usize> = (0..n).collect();
    let mut community_total: Vec<f64> = strengths.clone();

    let mut moved = true;
    let mut visits = 0usize;
    while moved {
        moved = false;
        for node in 0..n {
            visits += 1;
            if visits.is_multiple_of(DEADLINE_STRIDE) && deadline.expired() {
                return Err(DeadlineExceeded);
            }
            let own = community[node];
            // Weight from `node` into each adjacent community, own
            // included (without the node's self-loop — moving with
            // yourself is not a connection).
            let mut adjacent: BTreeMap<usize, f64> = BTreeMap::new();
            adjacent.insert(own, 0.0);
            for &(neighbor, weight) in &graph.neighbors[node] {
                accumulate_saturating(adjacent.entry(community[neighbor]).or_insert(0.0), weight);
            }
            // Lift the node out before comparing, so its own strength
            // doesn't count against candidate communities (nor its
            // current one).
            community_total[own] -= strengths[node];
            let gain = |target: usize, link: f64| -> f64 {
                link - strengths[node] * community_total[target] / (2.0 * m)
            };
            let stay = gain(own, adjacent[&own]);
            // BTreeMap iteration is ascending by community id, and the
            // strict `>` keeps the first (lowest-id) best — the
            // deterministic tie-break.
            let mut best = (own, stay);
            for (&target, &link) in &adjacent {
                if target == own {
                    continue;
                }
                let candidate = gain(target, link);
                if candidate > best.1 {
                    best = (target, candidate);
                }
            }
            community[node] = best.0;
            community_total[best.0] += strengths[node];
            if best.0 != own {
                moved = true;
            }
        }
    }

    renumber(&mut community);
    let split = split_components(graph, &mut community);
    Ok((community, split))
}

/// Renumbers an assignment densely in first-member order (node 0's
/// community becomes 0, the next unseen community 1, …). Returns the
/// community count.
fn renumber(assignment: &mut [usize]) -> usize {
    let mut map: HashMap<usize, usize> = HashMap::new();
    for value in assignment.iter_mut() {
        let next = map.len();
        *value = *map.entry(*value).or_insert(next);
    }
    map.len()
}

/// Splits every community into its connected components (intra-
/// community edges only) — the cheap half of Leiden's correction:
/// Louvain can glue two subgraphs whose only bond was a node that
/// later moved away, and a disconnected "community" summarizes into
/// nonsense. BFS from the lowest member keeps it deterministic.
/// Renumbers again and returns the final count.
fn split_components(graph: &WorkGraph, assignment: &mut [usize]) -> usize {
    let mut component: Vec<Option<usize>> = vec![None; assignment.len()];
    let mut next = 0usize;
    for start in 0..assignment.len() {
        if component[start].is_some() {
            continue;
        }
        let id = next;
        next += 1;
        component[start] = Some(id);
        let mut frontier = vec![start];
        while let Some(node) = frontier.pop() {
            for &(neighbor, _) in &graph.neighbors[node] {
                if assignment[neighbor] == assignment[node] && component[neighbor].is_none() {
                    component[neighbor] = Some(id);
                    frontier.push(neighbor);
                }
            }
        }
    }
    for (value, part) in assignment.iter_mut().zip(component) {
        *value = part.expect("every node was visited");
    }
    renumber(assignment)
}

/// Collapses a level's communities into the next level's working
/// graph: inter-community weights summed into undirected edges,
/// intra-community weight (member self-loops included) into the
/// super-node's self-loop.
fn aggregate(graph: &WorkGraph, assignment: &[usize], communities: usize) -> WorkGraph {
    let mut self_weight = vec![0.0; communities];
    let mut between: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for node in 0..graph.len() {
        let home = assignment[node];
        accumulate_saturating(&mut self_weight[home], graph.self_weight[node]);
        for &(neighbor, weight) in &graph.neighbors[node] {
            let there = assignment[neighbor];
            if home == there {
                if node < neighbor {
                    accumulate_saturating(&mut self_weight[home], weight);
                }
            } else if node < neighbor {
                let pair = (home.min(there), home.max(there));
                accumulate_saturating(between.entry(pair).or_insert(0.0), weight);
            }
        }
    }
    let mut neighbors = vec![Vec::new(); communities];
    for ((a, b), weight) in between {
        neighbors[a].push((b, weight));
        neighbors[b].push((a, weight));
    }
    for list in &mut neighbors {
        list.sort_unstable_by_key(|&(neighbor, _)| neighbor);
    }
    WorkGraph {
        neighbors,
        self_weight,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clique(context: &mut Context, names: &[&str], weight: f64) {
        for (index, &a) in names.iter().enumerate() {
            for &b in &names[index + 1..] {
                context.associate(a, "近い", b, weight).unwrap();
            }
        }
    }

    fn leaf_ids(analysis: &CommunityAnalysis) -> Vec<Vec<String>> {
        let mut leaves: Vec<Vec<String>> = analysis
            .communities
            .iter()
            .filter(|community| community.level == 0)
            .map(|community| {
                let mut names: Vec<String> = community
                    .members
                    .iter()
                    .map(|member| member.name.clone())
                    .collect();
                names.sort();
                names
            })
            .collect();
        leaves.sort();
        leaves
    }

    #[test]
    fn two_cliques_with_a_weak_bridge_split_into_two_communities() {
        let mut context = Context::default();
        clique(&mut context, &["a1", "a2", "a3", "a4"], 2.0);
        clique(&mut context, &["b1", "b2", "b3", "b4"], 2.0);
        context.associate("a1", "橋", "b1", 0.1).unwrap();

        let analysis = context.communities(Deadline::unbounded()).unwrap();
        assert_eq!(analysis.algorithm, COMMUNITY_ALGORITHM);
        assert_eq!(analysis.concept_count, 8);
        assert_eq!(
            leaf_ids(&analysis),
            vec![vec!["a1", "a2", "a3", "a4"], vec!["b1", "b2", "b3", "b4"],]
        );
    }

    #[test]
    fn same_graph_in_same_communities_out() {
        let build = || {
            let mut context = Context::default();
            clique(&mut context, &["x1", "x2", "x3"], 1.0);
            clique(&mut context, &["y1", "y2", "y3"], 1.0);
            clique(&mut context, &["z1", "z2", "z3"], 1.0);
            context.associate("x1", "r", "y1", 0.2).unwrap();
            context.associate("y2", "r", "z1", 0.2).unwrap();
            context
        };
        let first = build().communities(Deadline::unbounded()).unwrap();
        let second = build().communities(Deadline::unbounded()).unwrap();
        let render = |analysis: &CommunityAnalysis| {
            analysis
                .communities
                .iter()
                .map(|community| {
                    format!(
                        "{}:{}:{}",
                        community.id, community.fingerprint, community.concept_count
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(render(&first), render(&second));
    }

    #[test]
    fn retracted_and_netted_edges_do_not_bind_communities() {
        let mut context = Context::default();
        clique(&mut context, &["a1", "a2", "a3"], 2.0);
        clique(&mut context, &["b1", "b2", "b3"], 2.0);
        // A strong bond that is then fully withdrawn…
        context.associate("a1", "絆", "b1", 5.0).unwrap();
        context.retract_association("a1", "絆", "b1").unwrap();
        // …and one that nets to zero: both must not glue the cliques.
        context.associate("a2", "論争", "b2", 3.0).unwrap();
        context.associate("a2", "論争", "b2", -3.0).unwrap();

        let analysis = context.communities(Deadline::unbounded()).unwrap();
        assert_eq!(
            leaf_ids(&analysis),
            vec![vec!["a1", "a2", "a3"], vec!["b1", "b2", "b3"]]
        );
    }

    #[test]
    fn a_contested_negative_relation_still_binds_its_endpoints() {
        let mut context = Context::default();
        clique(&mut context, &["a1", "a2", "a3"], 1.0);
        // Firmly negative is strong knowledge — |sum| binds, so n1
        // enters detection and lands in a1's community (a -3.0 bond
        // outweighs the clique's 1.0 edges; whether it pulls a1 out of
        // the clique is modularity's call, not this test's).
        context.associate("a1", "否定", "n1", -3.0).unwrap();

        let analysis = context.communities(Deadline::unbounded()).unwrap();
        assert_eq!(analysis.concept_count, 4);
        let with_n1 = leaf_ids(&analysis)
            .into_iter()
            .find(|members| members.iter().any(|name| name == "n1"))
            .expect("n1 must be clustered");
        assert!(with_n1.iter().any(|name| name == "a1"));
    }

    #[test]
    fn isolated_concepts_and_pure_self_loops_stay_out() {
        let mut context = Context::default();
        clique(&mut context, &["a1", "a2", "a3"], 1.0);
        // A concept whose only association is a self-loop has no
        // topology to cluster by.
        context.associate("島", "自認", "島", 1.0).unwrap();

        let analysis = context.communities(Deadline::unbounded()).unwrap();
        assert_eq!(analysis.concept_count, 3);
        assert_eq!(leaf_ids(&analysis), vec![vec!["a1", "a2", "a3"]]);
    }

    #[test]
    fn fingerprints_move_with_content_and_hold_without() {
        let mut context = Context::default();
        clique(&mut context, &["a1", "a2", "a3", "a4"], 2.0);
        clique(&mut context, &["b1", "b2", "b3", "b4"], 2.0);
        context.associate("a1", "橋", "b1", 0.1).unwrap();
        let before = context.communities(Deadline::unbounded()).unwrap();

        // Strengthen one clique's internal edge: only that community's
        // fingerprint may move.
        context.associate("a1", "近い", "a2", 1.0).unwrap();
        let after = context.communities(Deadline::unbounded()).unwrap();

        let fingerprint_of = |analysis: &CommunityAnalysis, member: &str| {
            analysis
                .communities
                .iter()
                .find(|community| {
                    community.level == 0 && community.members.iter().any(|m| m.name == member)
                })
                .map(|community| community.fingerprint.clone())
                .unwrap()
        };
        assert_ne!(fingerprint_of(&before, "a1"), fingerprint_of(&after, "a1"));
        assert_eq!(fingerprint_of(&before, "b1"), fingerprint_of(&after, "b1"));
    }

    #[test]
    fn members_rank_by_intra_community_strength() {
        let mut context = Context::default();
        context.associate("中心", "r", "a", 3.0).unwrap();
        context.associate("中心", "r", "b", 3.0).unwrap();
        context.associate("a", "r", "b", 0.5).unwrap();

        let analysis = context.communities(Deadline::unbounded()).unwrap();
        let leaf = analysis
            .communities
            .iter()
            .find(|community| community.level == 0)
            .unwrap();
        assert_eq!(leaf.members[0].name, "中心");
        assert!(leaf.members[0].strength > leaf.members[1].strength);
    }

    #[test]
    fn top_associations_stay_inside_the_community_and_rank_by_magnitude() {
        let mut context = Context::default();
        clique(&mut context, &["a1", "a2", "a3"], 1.0);
        clique(&mut context, &["b1", "b2", "b3"], 1.0);
        context.associate("a1", "最強", "a2", 9.0).unwrap();
        context.associate("a1", "橋", "b1", 0.1).unwrap();

        let analysis = context.communities(Deadline::unbounded()).unwrap();
        let community_of = |member: &str| {
            analysis
                .communities
                .iter()
                .find(|community| {
                    community.level == 0 && community.members.iter().any(|m| m.name == member)
                })
                .unwrap()
        };
        let a = community_of("a1");
        assert_eq!(a.top_associations[0].label, "最強");
        for association in &a.top_associations {
            // The bridge crosses communities — induced means induced.
            assert_ne!(association.label, "橋");
        }
    }

    #[test]
    fn an_empty_context_answers_an_empty_analysis() {
        let context = Context::default();
        let analysis = context.communities(Deadline::unbounded()).unwrap();
        assert_eq!(analysis.levels, 0);
        assert_eq!(analysis.concept_count, 0);
        assert!(analysis.communities.is_empty());
    }

    #[test]
    fn an_expired_deadline_answers_deadline_exceeded() {
        let mut context = Context::default();
        clique(&mut context, &["a1", "a2", "a3"], 1.0);
        let result = context.communities(Deadline::after(std::time::Duration::ZERO));
        assert!(result.is_err());
    }

    #[test]
    fn hierarchy_parents_contain_their_children() {
        // Three cliques chained pairwise: enough structure for at
        // least one aggregation level on most parameterizations —
        // but the invariant below must hold at ANY depth.
        let mut context = Context::default();
        for group in ["p", "q", "r", "s"] {
            let names: Vec<String> = (1..=4).map(|index| format!("{group}{index}")).collect();
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            clique(&mut context, &refs, 2.0);
        }
        context.associate("p1", "橋", "q1", 0.5).unwrap();
        context.associate("r1", "橋", "s1", 0.5).unwrap();

        let analysis = context.communities(Deadline::unbounded()).unwrap();
        for community in &analysis.communities {
            if let Some(parent_id) = &community.parent {
                let parent = analysis
                    .communities
                    .iter()
                    .find(|candidate| &candidate.id == parent_id)
                    .expect("a named parent must exist");
                assert_eq!(parent.level, community.level + 1);
                assert!(parent.children.contains(&community.id));
                assert!(parent.concept_count >= community.concept_count);
            }
        }
        // Concept counts are conserved per level.
        for level in 0..analysis.levels {
            let total: usize = analysis
                .communities
                .iter()
                .filter(|community| community.level == level)
                .map(|community| community.concept_count)
                .sum();
            assert_eq!(total, analysis.concept_count);
        }
    }
}
