//! Consolidation candidates, the graph-only half (ADR 0012 §4): the
//! structural corroboration behind a merge candidate, the grouped
//! contradiction pass, and the sign-conflict scan. Everything here is
//! detection over live edges — candidates, never verdicts — and
//! everything temporal (assertion times, staleness) stays with the
//! registry, where `SourceMeta` lives; these methods hand back source
//! NAMES so that join can happen above (the ADR 0011 §4 layering).

use std::collections::{BTreeSet, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::deadline::{Deadline, DeadlineExceeded};
use crate::hash::{FNV1A_OFFSET, fnv1a_fold, fold_field};

use super::{ConceptId, Context, LabelId};

/// One fact of a concept's neighborhood, as merge evidence reports it:
/// `out` means the concept is the subject of `(label, other)`, `in`
/// means it is the object of `(other, label)`. Ordered so evidence
/// lists are deterministic.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NeighborFact {
    /// `"out"` or `"in"` — which chain the fact sits on.
    pub direction: String,
    pub label: String,
    pub other: String,
}

/// The structural half of one merge candidate (ADR 0012 §4 `merge`):
/// how much of two concepts' live neighborhoods coincide, with the
/// evidence itself — the judge's question ("same thing, differently
/// spelled?") is answered by exactly these lists, so they ride along,
/// capped with honest totals rather than silently truncated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeEvidence {
    /// Jaccard over the two live neighbor sets (0.0 when both are
    /// empty — no structure is no corroboration, not perfect overlap).
    pub overlap: f64,
    pub shared_total: usize,
    pub only_a_total: usize,
    pub only_b_total: usize,
    pub shared: Vec<NeighborFact>,
    pub only_a: Vec<NeighborFact>,
    pub only_b: Vec<NeighborFact>,
    /// FNV-1a over the pair's names and its FULL evidence sets
    /// (never the capped lists), pair-order canonicalized — the
    /// judgment artifact's identity (ADR 0012 §5). Serialized by the
    /// audit's own wire shape, not here.
    #[serde(skip)]
    pub fingerprint: u64,
}

/// One object's standing in a contradiction group: its windowless
/// weight/count and the sources that attest it — names, not dates,
/// because dates live in the registry's passage store and join there.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectRow {
    pub object: String,
    pub weight: f64,
    pub count: u64,
    pub sources: Vec<String>,
}

/// One `(subject, label)` holding two or more live distinct objects
/// (ADR 0012 §4 `contradiction`, the between-objects kind), with the
/// label's measured functional tendency riding along so the caller
/// can rank: a label that is one-object for most of its subjects
/// makes its multi-object outliers strong candidates.
#[derive(Debug, Clone, Serialize)]
pub struct ContradictionGroup {
    pub subject: String,
    pub label: String,
    /// Fraction of this label's subjects holding exactly one live
    /// object — measured over the same pass, never declared (declared
    /// cardinality is still ADR 0009 §9.2's deferral).
    pub functional_tendency: f64,
    pub objects: Vec<ObjectRow>,
    /// FNV-1a over subject, label, and every object row (name, count,
    /// sum bits, sorted sources) — see [`MergeEvidence::fingerprint`].
    #[serde(skip)]
    pub fingerprint: u64,
}

/// One edge whose per-source attributions disagree in sign (ADR 0012
/// §4 `contradiction`, the within-edge kind) — the store's own
/// "contested" vocabulary, surfaced with both sides' sources so the
/// judge reads the dispute, not just its net.
#[derive(Debug, Clone, Serialize)]
pub struct SignConflict {
    pub subject: String,
    pub label: String,
    pub object: String,
    /// The net view (`sum / count`) the plain read serves — what the
    /// dispute currently resolves to.
    pub weight: f64,
    pub count: u64,
    pub supporting_sources: Vec<String>,
    pub disputing_sources: Vec<String>,
    /// FNV-1a over the edge's names, net, and both sides' sorted
    /// sources — see [`MergeEvidence::fingerprint`].
    #[serde(skip)]
    pub fingerprint: u64,
}

// Test-only deterministic fault injection for `merge_evidence`'s own
// per-iteration deadline checks (issue #620): mirrors `api::groups`'s
// `expire_fingerprint_loop_after` — a real deadline landing mid-walk
// through one concept's adjacency list cannot be raced reliably (each
// step is an in-memory lookup, done in well under a microsecond), so a
// regression test that needs the check to fire partway through arms
// this instead. Shared by `neighbor_set`'s two loops and
// `merge_evidence`'s own fingerprint fold, since all three sit behind
// one caller-visible contract ("checked throughout", not just before
// the call).
#[cfg(test)]
thread_local! {
    static EXPIRE_MERGE_EVIDENCE_LOOP_AFTER: std::cell::Cell<Option<u32>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn expire_merge_evidence_loop_after(remaining: u32) {
    EXPIRE_MERGE_EVIDENCE_LOOP_AFTER.with(|cell| cell.set(Some(remaining)));
}

#[cfg(test)]
fn injected_merge_evidence_loop_expiry() -> bool {
    EXPIRE_MERGE_EVIDENCE_LOOP_AFTER.with(|cell| match cell.get() {
        Some(0) => {
            cell.set(None);
            true
        }
        Some(remaining) => {
            cell.set(Some(remaining - 1));
            false
        }
        None => false,
    })
}

#[cfg(not(test))]
#[mutants::skip] // dead under cfg(test) — mirrors api.rs's injected_deadline_race
fn injected_merge_evidence_loop_expiry() -> bool {
    false
}

impl Context {
    /// The live neighborhood of one concept as a set of
    /// (direction, label, other) triples, hidden labels excluded —
    /// the comparable unit behind [`Context::merge_evidence`].
    /// Deadline-checked per edge (issue #620): a hub concept's degree
    /// is unbounded by anything `merge_evidence`'s own caller controls
    /// (`evidence_cap` only bounds what gets SERVED, not this walk),
    /// so a single pair naming one can otherwise run past the budget
    /// entirely between two of `merge_section`'s own per-pair checks.
    fn neighbor_set(
        &self,
        id: ConceptId,
        hidden: &HashSet<LabelId>,
        deadline: Deadline,
    ) -> Result<BTreeSet<(u8, LabelId, ConceptId)>, DeadlineExceeded> {
        let mut set = BTreeSet::new();
        for edge_id in self.outgoing(id) {
            if deadline.expired() || injected_merge_evidence_loop_expiry() {
                return Err(DeadlineExceeded);
            }
            let edge = &self.edges[edge_id as usize];
            if edge.count > 0 && !hidden.contains(&edge.label) {
                set.insert((0u8, edge.label, edge.object));
            }
        }
        for edge_id in self.incoming(id) {
            if deadline.expired() || injected_merge_evidence_loop_expiry() {
                return Err(DeadlineExceeded);
            }
            let edge = &self.edges[edge_id as usize];
            if edge.count > 0 && !hidden.contains(&edge.label) {
                set.insert((1u8, edge.label, edge.subject));
            }
        }
        Ok(set)
    }

    /// Structural corroboration for one candidate pair (ADR 0012 §4):
    /// Jaccard over the two concepts' live neighborhoods, with the
    /// shared and distinct facts listed (capped at `cap` per list,
    /// totals always exact). `excluded` labels — the reserved
    /// `schema:type` when a schema is installed — count toward
    /// nothing: two same-typed concepts sharing only their type edge
    /// have no *structural* overlap, and the type itself is reported
    /// separately by the audit (ADR 0012 §7). `Ok(None)` when either
    /// spelling is unknown. Deadline-checked throughout (issue #620),
    /// not just before/after the call: `neighbor_set`'s own walk and
    /// the fingerprint fold below both scale with a concept's degree,
    /// not with `cap`.
    pub fn merge_evidence(
        &self,
        a: &str,
        b: &str,
        cap: usize,
        excluded: &[&str],
        deadline: Deadline,
    ) -> Result<Option<MergeEvidence>, DeadlineExceeded> {
        let (Some(&a_id), Some(&b_id)) = (self.concept_ids.get(a), self.concept_ids.get(b)) else {
            return Ok(None);
        };
        let hidden: HashSet<LabelId> = excluded
            .iter()
            .filter_map(|name| self.label_ids.get(*name).copied())
            .collect();
        let a_set = self.neighbor_set(a_id, &hidden, deadline)?;
        let b_set = self.neighbor_set(b_id, &hidden, deadline)?;

        let materialize = |(direction, label, other): &(u8, LabelId, ConceptId)| NeighborFact {
            direction: if *direction == 0 { "out" } else { "in" }.to_string(),
            label: self.label_name(*label).to_string(),
            other: self.concept_name(*other).to_string(),
        };
        let shared: Vec<_> = a_set.intersection(&b_set).collect();
        let only_a: Vec<_> = a_set.difference(&b_set).collect();
        let only_b: Vec<_> = b_set.difference(&a_set).collect();
        let union = shared.len() + only_a.len() + only_b.len();

        // The fingerprint folds NAMES over the FULL evidence, pair-order
        // canonicalized: swapping (a, b) must not mint a new identity,
        // and dense ids — reassigned by compaction — must never enter.
        let fact_strings =
            |facts: &[&(u8, LabelId, ConceptId)]| -> Result<Vec<String>, DeadlineExceeded> {
                let mut strings = Vec::with_capacity(facts.len());
                for &&(direction, label, other) in facts {
                    if deadline.expired() || injected_merge_evidence_loop_expiry() {
                        return Err(DeadlineExceeded);
                    }
                    strings.push(format!(
                        "{}|{}|{}",
                        direction,
                        self.label_name(label),
                        self.concept_name(other)
                    ));
                }
                strings.sort_unstable();
                Ok(strings)
            };
        let (first, second, first_only, second_only) = if a <= b {
            (a, b, &only_a, &only_b)
        } else {
            (b, a, &only_b, &only_a)
        };
        let mut digest = fold_field(FNV1A_OFFSET, "merge");
        digest = fold_field(digest, first);
        digest = fold_field(digest, second);
        for (tag, facts) in [("s", &shared), ("f", first_only), ("g", second_only)] {
            digest = fold_field(digest, tag);
            for fact in fact_strings(facts)? {
                digest = fold_field(digest, &fact);
            }
        }

        Ok(Some(MergeEvidence {
            overlap: if union == 0 {
                0.0
            } else {
                shared.len() as f64 / union as f64
            },
            shared_total: shared.len(),
            only_a_total: only_a.len(),
            only_b_total: only_b.len(),
            shared: shared.into_iter().take(cap).map(materialize).collect(),
            only_a: only_a.into_iter().take(cap).map(materialize).collect(),
            only_b: only_b.into_iter().take(cap).map(materialize).collect(),
            fingerprint: digest,
        }))
    }

    /// Every `(subject, label)` holding at least two live distinct
    /// objects, sorted by (label id, subject id) — the grouping map's
    /// own iteration order is nondeterministic, so this sort is what
    /// makes the output deterministic — each label carrying its
    /// measured functional tendency; one O(edges) grouped pass
    /// (ADR 0012 §4). Deadline-checked like every other full-graph
    /// audit.
    pub fn contradiction_groups(
        &self,
        deadline: Deadline,
    ) -> Result<Vec<ContradictionGroup>, DeadlineExceeded> {
        // (subject, label) → live distinct objects, in first-seen order.
        let mut groups: HashMap<(ConceptId, LabelId), Vec<u32>> = HashMap::new();
        for edge_id in 0..self.edges.len() as u32 {
            if deadline.expired() {
                return Err(DeadlineExceeded);
            }
            let edge = &self.edges[edge_id as usize];
            if edge.count == 0 {
                continue;
            }
            groups
                .entry((edge.subject, edge.label))
                .or_default()
                .push(edge_id);
        }

        // Per label: how many subjects hold exactly one live object —
        // the tendency the candidates rank on.
        let mut label_subjects: HashMap<LabelId, (usize, usize)> = HashMap::new();
        for (&(_, label), edges) in &groups {
            let entry = label_subjects.entry(label).or_default();
            entry.1 += 1;
            if edges.len() == 1 {
                entry.0 += 1;
            }
        }

        let mut candidates: Vec<(&(ConceptId, LabelId), &Vec<u32>)> = groups
            .iter()
            .filter(|(_, edges)| edges.len() >= 2)
            .collect();
        candidates.sort_unstable_by_key(|((subject, label), _)| (*label, *subject));
        Ok(candidates
            .into_iter()
            .map(|(&(subject, label), edges)| {
                let (single, total) = label_subjects[&label];
                let objects: Vec<ObjectRow> = edges
                    .iter()
                    .map(|&edge_id| {
                        let edge = &self.edges[edge_id as usize];
                        ObjectRow {
                            object: self.concept_name(edge.object).to_string(),
                            weight: edge.sum / edge.count as f64,
                            count: edge.count,
                            // Zero-sum records stay chain-linked (only a
                            // retraction unlinks) but attest nothing —
                            // the same posture `sign_conflicts` takes,
                            // whose zero-sum sources sit on neither side
                            // of a dispute. Listing one here would hand
                            // the judge a source that never net-asserted
                            // the triple.
                            sources: self
                                .attribution_chain(edge.first_attribution)
                                .filter(|(_, record)| record.sum != 0.0)
                                .map(|(_, record)| self.source_name(record.source).to_string())
                                .collect(),
                        }
                    })
                    .collect();
                let mut digest = fold_field(FNV1A_OFFSET, "objects");
                digest = fold_field(digest, self.concept_name(subject));
                digest = fold_field(digest, self.label_name(label));
                let mut rows: Vec<&ObjectRow> = objects.iter().collect();
                rows.sort_unstable_by(|a, b| a.object.cmp(&b.object));
                for row in rows {
                    digest = fold_field(digest, &row.object);
                    digest = fnv1a_fold(digest, row.count.to_le_bytes());
                    digest = fnv1a_fold(digest, row.weight.to_bits().to_le_bytes());
                    let mut sources: Vec<&String> = row.sources.iter().collect();
                    sources.sort_unstable();
                    for source in sources {
                        digest = fold_field(digest, source);
                    }
                }
                ContradictionGroup {
                    subject: self.concept_name(subject).to_string(),
                    label: self.label_name(label).to_string(),
                    functional_tendency: single as f64 / total as f64,
                    objects,
                    fingerprint: digest,
                }
            })
            .collect())
    }

    /// Every live edge whose per-source attributions disagree in sign,
    /// in edge (insertion) order — a source whose own accumulated sum
    /// is exactly zero sits on neither side of the dispute. One
    /// O(edges + attributions) scan, deadline-checked.
    pub fn sign_conflicts(
        &self,
        deadline: Deadline,
    ) -> Result<Vec<SignConflict>, DeadlineExceeded> {
        let mut conflicts = Vec::new();
        for edge_id in 0..self.edges.len() as u32 {
            if deadline.expired() {
                return Err(DeadlineExceeded);
            }
            let edge = &self.edges[edge_id as usize];
            if edge.count == 0 {
                continue;
            }
            let mut supporting = Vec::new();
            let mut disputing = Vec::new();
            for (_, record) in self.attribution_chain(edge.first_attribution) {
                if record.sum > 0.0 {
                    supporting.push(self.source_name(record.source).to_string());
                } else if record.sum < 0.0 {
                    disputing.push(self.source_name(record.source).to_string());
                }
            }
            if !supporting.is_empty() && !disputing.is_empty() {
                let mut digest = fold_field(FNV1A_OFFSET, "contested");
                digest = fold_field(digest, self.concept_name(edge.subject));
                digest = fold_field(digest, self.label_name(edge.label));
                digest = fold_field(digest, self.concept_name(edge.object));
                digest = fnv1a_fold(digest, edge.count.to_le_bytes());
                digest = fnv1a_fold(digest, edge.sum.to_bits().to_le_bytes());
                for (tag, side) in [("+", &supporting), ("-", &disputing)] {
                    digest = fold_field(digest, tag);
                    let mut sorted: Vec<&String> = side.iter().collect();
                    sorted.sort_unstable();
                    for source in sorted {
                        digest = fold_field(digest, source);
                    }
                }
                conflicts.push(SignConflict {
                    subject: self.concept_name(edge.subject).to_string(),
                    label: self.label_name(edge.label).to_string(),
                    object: self.concept_name(edge.object).to_string(),
                    weight: edge.sum / edge.count as f64,
                    count: edge.count,
                    supporting_sources: supporting,
                    disputing_sources: disputing,
                    fingerprint: digest,
                });
            }
        }
        Ok(conflicts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_evidence_measures_live_structural_overlap() {
        let mut context = Context::default();
        // 青嶺酒造 and its spelling twin 青嶺酒蔵 share two facts, and
        // each has one of its own; a retracted fact must not count.
        for subject in ["青嶺酒造", "青嶺酒蔵"] {
            context.associate(subject, "銘柄", "青嶺", 1.0).unwrap();
            context.associate(subject, "所在地", "霧沢町", 1.0).unwrap();
        }
        context.associate("青嶺酒造", "杜氏", "高瀬", 1.0).unwrap();
        context
            .associate("高瀬", "勤務先", "青嶺酒蔵", 1.0)
            .unwrap();
        context
            .associate_from("青嶺酒蔵", "創業", "1907", 1.0, "doc", None)
            .unwrap();
        context.retract_source("doc");

        let evidence = context
            .merge_evidence("青嶺酒造", "青嶺酒蔵", 10, &[], Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(evidence.shared_total, 2);
        assert_eq!(evidence.only_a_total, 1, "杜氏→高瀬");
        assert_eq!(
            evidence.only_b_total, 1,
            "incoming 勤務先, retracted 創業 excluded"
        );
        assert_eq!(evidence.overlap, 2.0 / 4.0);
        assert!(
            evidence
                .shared
                .iter()
                .any(|fact| fact.direction == "out" && fact.label == "銘柄")
        );
        assert!(
            evidence
                .only_b
                .iter()
                .any(|fact| fact.direction == "in" && fact.other == "高瀬")
        );
        assert!(
            context
                .merge_evidence("青嶺酒造", "未知", 10, &[], Deadline::unbounded())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn merge_evidence_excludes_hidden_labels_from_overlap() {
        let mut context = Context::default();
        context
            .associate("a", "schema:type", "Person", 1.0)
            .unwrap();
        context
            .associate("b", "schema:type", "Person", 1.0)
            .unwrap();
        let bare = context
            .merge_evidence("a", "b", 10, &["schema:type"], Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(bare.overlap, 0.0, "a shared type edge is not structure");
        assert_eq!(bare.shared_total, 0);
        let unhidden = context
            .merge_evidence("a", "b", 10, &[], Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(unhidden.overlap, 1.0);
    }

    #[test]
    fn merge_evidence_caps_lists_never_totals_and_fingerprints_ignore_pair_order_and_cap() {
        let mut context = Context::default();
        for n in 0..5 {
            context
                .associate("a", "shares", format!("x{n}"), 1.0)
                .unwrap();
            context
                .associate("b", "shares", format!("x{n}"), 1.0)
                .unwrap();
        }
        context.associate("a", "own", "only-a", 1.0).unwrap();

        let capped = context
            .merge_evidence("a", "b", 2, &[], Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(capped.shared_total, 5, "totals stay exact past the cap");
        assert_eq!(capped.shared.len(), 2, "lists cap");
        assert_eq!(capped.only_a_total, 1);

        // The fingerprint covers the FULL evidence, so neither the cap
        // nor the pair order may move it — it is the judgment
        // artifact's identity (ADR 0012 §5).
        let uncapped = context
            .merge_evidence("a", "b", 100, &[], Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(capped.fingerprint, uncapped.fingerprint);
        let swapped = context
            .merge_evidence("b", "a", 2, &[], Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(capped.fingerprint, swapped.fingerprint);
        assert_eq!(swapped.only_a_total, 0, "sides follow the arguments");
        assert_eq!(swapped.only_b_total, 1);
    }

    #[test]
    fn full_graph_passes_honor_an_expired_deadline() {
        let mut context = Context::default();
        context.associate("a", "r", "b", 1.0).unwrap();
        let expired = Deadline::after(std::time::Duration::ZERO);
        assert!(context.contradiction_groups(expired).is_err());
        assert!(context.sign_conflicts(expired).is_err());
    }

    /// issue #620: `merge_evidence` must refuse an already-expired
    /// deadline like every other full-graph pass.
    #[test]
    fn merge_evidence_honors_an_expired_deadline() {
        let mut context = Context::default();
        context.associate("a", "r", "x", 1.0).unwrap();
        context.associate("b", "r", "x", 1.0).unwrap();
        let expired = Deadline::after(std::time::Duration::ZERO);
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(expired.expired(), "a zero budget must read as expired");
        assert!(context.merge_evidence("a", "b", 10, &[], expired).is_err());
    }

    /// issue #620: the check inside `neighbor_set`'s own walk must
    /// fire on a LATER iteration, not only before `merge_evidence` is
    /// called — a hub concept's degree is unbounded by `evidence_cap`
    /// (which only bounds what gets SERVED), so `merge_section`'s own
    /// per-pair check cannot catch a budget dying partway through one
    /// pair's adjacency walk. A real `Duration` cannot land this
    /// deterministically (each step is an in-memory lookup, done in
    /// well under a microsecond) — `expire_merge_evidence_loop_after`
    /// arms the fault instead, mirroring `api::groups`'s own
    /// `expire_fingerprint_loop_after` for the same reason.
    #[test]
    fn merge_evidence_checks_the_deadline_mid_walk_not_only_before_it_starts() {
        let mut context = Context::default();
        for n in 0..5 {
            context
                .associate("a", "shares", format!("x{n}"), 1.0)
                .unwrap();
        }
        context.associate("b", "shares", "x0", 1.0).unwrap();

        // The first per-edge check reports "not yet expired"; the
        // second reports expired — proof the check runs more than
        // once across "a"'s five-edge walk, not only once before it.
        expire_merge_evidence_loop_after(1);
        let result = context.merge_evidence("a", "b", 10, &[], Deadline::unbounded());
        assert!(
            result.is_err(),
            "the check must fire on a later iteration of neighbor_set's walk"
        );
    }

    /// issue #620: `neighbor_set`'s INCOMING-edge loop has its own
    /// deadline check, independent of the outgoing loop above — a
    /// concept that is mostly (or only) an OBJECT of other assertions
    /// has an unbounded incoming degree the outgoing loop's own check
    /// never walks through. Calls `neighbor_set` directly rather than
    /// through `merge_evidence`: `deadline.expired() || injected(..)`
    /// short-circuits, so if the incoming loop's own check were ever
    /// downgraded to `&&`, an unconsumed armed fault would silently
    /// leak forward into `merge_evidence`'s later checkpoints (`b`'s
    /// own walk, then the fingerprint fold) and still be consumed
    /// there — making the overall call return `Err` either way and
    /// hiding exactly the regression this test exists to catch.
    #[test]
    fn merge_evidence_checks_the_deadline_on_the_incoming_side_too() {
        let mut context = Context::default();
        // "a" has zero outgoing edges and five incoming edges — the
        // outgoing loop consults the fault zero times, so the first
        // consult the fault sees can only come from the incoming loop.
        for n in 0..5 {
            context.associate(format!("z{n}"), "r", "a", 1.0).unwrap();
        }
        let a_id = *context.concept_ids.get("a").unwrap();

        // The first per-edge check reports "not yet expired"; the
        // second reports expired — proof the check runs more than
        // once across the incoming walk, not only once before it.
        expire_merge_evidence_loop_after(1);
        let result = context.neighbor_set(a_id, &HashSet::new(), Deadline::unbounded());
        assert!(
            result.is_err(),
            "the incoming-edge loop must also consult the injected fault"
        );
    }

    /// issue #620: the fingerprint fold over the full evidence
    /// (`fact_strings`, after both sides' `neighbor_set` calls
    /// already succeeded) has its own deadline check too — the fold
    /// is unbounded in the number of shared/distinct facts, same
    /// reasoning as the adjacency walk itself.
    #[test]
    fn merge_evidence_checks_the_deadline_during_the_fingerprint_fold_too() {
        let mut context = Context::default();
        context.associate("a", "r", "only-a", 1.0).unwrap();
        context.associate("b", "r", "only-b", 1.0).unwrap();

        // Exactly 2 fault consults happen inside `neighbor_set("a")`
        // and `neighbor_set("b")` (one outgoing edge each, no
        // incoming ones) before either side's walk could possibly
        // fail; the 3rd consult is `fact_strings`' own first
        // iteration over the (non-shared) evidence.
        expire_merge_evidence_loop_after(2);
        let result = context.merge_evidence("a", "b", 10, &[], Deadline::unbounded());
        assert!(
            result.is_err(),
            "the fingerprint fold must also consult the injected fault"
        );
    }

    #[test]
    fn contradiction_groups_find_multi_object_labels_and_measure_tendency() {
        let mut context = Context::default();
        // 杜氏 is functional in tendency: three subjects, two with one
        // object, one with two — the candidate.
        context
            .associate_from("蔵A", "杜氏", "高瀬", 1.0, "old", None)
            .unwrap();
        context
            .associate_from("蔵A", "杜氏", "青山", 1.0, "new", None)
            .unwrap();
        context.associate("蔵B", "杜氏", "田中", 1.0).unwrap();
        context.associate("蔵C", "杜氏", "佐藤", 1.0).unwrap();
        // 銘柄 is naturally many-valued and must not float up: one
        // subject, two objects, tendency 0.
        context.associate("蔵B", "銘柄", "初霜", 1.0).unwrap();
        context.associate("蔵B", "銘柄", "冬雲", 1.0).unwrap();

        let groups = context.contradiction_groups(Deadline::unbounded()).unwrap();
        assert_eq!(groups.len(), 2);
        let toji = groups.iter().find(|g| g.label == "杜氏").unwrap();
        assert_eq!(toji.subject, "蔵A");
        assert_eq!(toji.functional_tendency, 2.0 / 3.0);
        assert_eq!(toji.objects.len(), 2);
        assert_eq!(toji.objects[0].sources, vec!["old"]);
        let brand = groups.iter().find(|g| g.label == "銘柄").unwrap();
        assert_eq!(brand.functional_tendency, 0.0);

        // A retraction dissolves the candidate: one live object left.
        context.retract_source("new");
        let after = context.contradiction_groups(Deadline::unbounded()).unwrap();
        assert!(after.iter().all(|g| g.label != "杜氏"), "{after:?}");
    }

    /// A source that asserted a triple and then cancelled its own
    /// assertion (net sum exactly zero, record still chain-linked —
    /// only a retraction unlinks) attests nothing: the evidence rows
    /// handed to the judge must not list it, the same posture
    /// `sign_conflicts` already takes for its zero-sum records.
    #[test]
    fn contradiction_group_sources_exclude_a_source_whose_own_sum_cancelled_to_zero() {
        let mut context = Context::default();
        context
            .associate_from("蔵A", "杜氏", "高瀬", 1.0, "old", None)
            .unwrap();
        context
            .associate_from("蔵A", "杜氏", "青山", 1.0, "new", None)
            .unwrap();
        // 撤回記事 asserts 高瀬, then cancels itself with the exact
        // negative — chain-linked at sum 0.0, attesting nothing.
        context
            .associate_from("蔵A", "杜氏", "高瀬", 2.0, "撤回記事", None)
            .unwrap();
        context
            .associate_from("蔵A", "杜氏", "高瀬", -2.0, "撤回記事", None)
            .unwrap();
        // Tendency needs company to make 杜氏 a candidate.
        context.associate("蔵B", "杜氏", "田中", 1.0).unwrap();

        let groups = context.contradiction_groups(Deadline::unbounded()).unwrap();
        let toji = groups.iter().find(|g| g.label == "杜氏").unwrap();
        let takase = toji
            .objects
            .iter()
            .find(|row| row.object == "高瀬")
            .unwrap();
        assert_eq!(
            takase.sources,
            vec!["old"],
            "the cancelled-out source must not read as attesting: {takase:?}"
        );
    }

    #[test]
    fn sign_conflicts_split_the_dispute_by_source() {
        let mut context = Context::default();
        context
            .associate_from("蔵", "行う", "大量生産", 1.0, "宣伝", None)
            .unwrap();
        context
            .associate_from("蔵", "行う", "大量生産", -2.0, "検証記事", None)
            .unwrap();
        // An uncontested edge stays out.
        context
            .associate_from("蔵", "銘柄", "青嶺", 1.0, "宣伝", None)
            .unwrap();

        let conflicts = context.sign_conflicts(Deadline::unbounded()).unwrap();
        assert_eq!(conflicts.len(), 1);
        let conflict = &conflicts[0];
        assert_eq!(conflict.object, "大量生産");
        assert_eq!(conflict.supporting_sources, vec!["宣伝"]);
        assert_eq!(conflict.disputing_sources, vec!["検証記事"]);
        assert_eq!(conflict.weight, -0.5, "the net the plain read serves");
    }
}
