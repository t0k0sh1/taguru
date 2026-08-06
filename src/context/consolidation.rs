//! Consolidation candidates, the graph-only half (ADR 0012 §4): the
//! structural corroboration behind a merge candidate, the grouped
//! contradiction pass, and the sign-conflict scan. Everything here is
//! detection over live edges — candidates, never verdicts — and
//! everything temporal (assertion times, staleness) stays with the
//! registry, where `SourceMeta` lives; these methods hand back source
//! NAMES so that join can happen above (the ADR 0011 §4 layering).

use std::collections::{BTreeSet, HashMap, HashSet};

use serde::Serialize;

use crate::deadline::{Deadline, DeadlineExceeded};
use crate::hash::{FNV1A_OFFSET, fnv1a_fold};

use super::{ConceptId, Context, LabelId};

/// One string folded into a candidate fingerprint, NUL-terminated so
/// field boundaries cannot alias ("ab"+"c" ≠ "a"+"bc"). Fingerprints
/// are the judgment artifact's identity (ADR 0012 §5), so they fold
/// NAMES, never dense ids — ids are reassigned by compaction, names
/// are the stable evidence.
fn fold_field(digest: u64, field: &str) -> u64 {
    fnv1a_fold(fnv1a_fold(digest, field.bytes()), [0u8])
}

/// One fact of a concept's neighborhood, as merge evidence reports it:
/// `out` means the concept is the subject of `(label, other)`, `in`
/// means it is the object of `(other, label)`. Ordered so evidence
/// lists are deterministic.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct NeighborFact {
    /// `"out"` or `"in"` — which chain the fact sits on.
    pub direction: &'static str,
    pub label: String,
    pub other: String,
}

/// The structural half of one merge candidate (ADR 0012 §4 `merge`):
/// how much of two concepts' live neighborhoods coincide, with the
/// evidence itself — the judge's question ("same thing, differently
/// spelled?") is answered by exactly these lists, so they ride along,
/// capped with honest totals rather than silently truncated.
#[derive(Debug, Clone, Serialize)]
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
#[derive(Debug, Clone, Serialize)]
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

impl Context {
    /// The live neighborhood of one concept as a set of
    /// (direction, label, other) triples, hidden labels excluded —
    /// the comparable unit behind [`Context::merge_evidence`].
    fn neighbor_set(
        &self,
        id: ConceptId,
        hidden: &HashSet<LabelId>,
    ) -> BTreeSet<(u8, LabelId, ConceptId)> {
        let mut set = BTreeSet::new();
        for edge_id in self.outgoing(id) {
            let edge = &self.edges[edge_id as usize];
            if edge.count > 0 && !hidden.contains(&edge.label) {
                set.insert((0u8, edge.label, edge.object));
            }
        }
        for edge_id in self.incoming(id) {
            let edge = &self.edges[edge_id as usize];
            if edge.count > 0 && !hidden.contains(&edge.label) {
                set.insert((1u8, edge.label, edge.subject));
            }
        }
        set
    }

    /// Structural corroboration for one candidate pair (ADR 0012 §4):
    /// Jaccard over the two concepts' live neighborhoods, with the
    /// shared and distinct facts listed (capped at `cap` per list,
    /// totals always exact). `excluded` labels — the reserved
    /// `schema:type` when a schema is installed — count toward
    /// nothing: two same-typed concepts sharing only their type edge
    /// have no *structural* overlap, and the type itself is reported
    /// separately by the audit (ADR 0012 §7). `None` when either
    /// spelling is unknown.
    pub fn merge_evidence(
        &self,
        a: &str,
        b: &str,
        cap: usize,
        excluded: &[&str],
    ) -> Option<MergeEvidence> {
        let &a_id = self.concept_ids.get(a)?;
        let &b_id = self.concept_ids.get(b)?;
        let hidden: HashSet<LabelId> = excluded
            .iter()
            .filter_map(|name| self.label_ids.get(*name).copied())
            .collect();
        let a_set = self.neighbor_set(a_id, &hidden);
        let b_set = self.neighbor_set(b_id, &hidden);

        let materialize = |(direction, label, other): &(u8, LabelId, ConceptId)| NeighborFact {
            direction: if *direction == 0 { "out" } else { "in" },
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
        let fact_strings = |facts: &[&(u8, LabelId, ConceptId)]| -> Vec<String> {
            let mut strings: Vec<String> = facts
                .iter()
                .map(|&&(direction, label, other)| {
                    format!(
                        "{}|{}|{}",
                        direction,
                        self.label_name(label),
                        self.concept_name(other)
                    )
                })
                .collect();
            strings.sort_unstable();
            strings
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
            for fact in fact_strings(facts) {
                digest = fold_field(digest, &fact);
            }
        }

        Some(MergeEvidence {
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
        })
    }

    /// Every `(subject, label)` holding at least two live distinct
    /// objects, in (label, subject) insertion order, each label
    /// carrying its measured functional tendency — one O(edges)
    /// grouped pass (ADR 0012 §4). Deadline-checked like every other
    /// full-graph audit.
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
                            sources: self
                                .attribution_chain(edge.first_attribution)
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
            .merge_evidence("青嶺酒造", "青嶺酒蔵", 10, &[])
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
                .merge_evidence("青嶺酒造", "未知", 10, &[])
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
            .merge_evidence("a", "b", 10, &["schema:type"])
            .unwrap();
        assert_eq!(bare.overlap, 0.0, "a shared type edge is not structure");
        assert_eq!(bare.shared_total, 0);
        let unhidden = context.merge_evidence("a", "b", 10, &[]).unwrap();
        assert_eq!(unhidden.overlap, 1.0);
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
