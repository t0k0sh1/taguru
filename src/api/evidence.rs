//! Evidence candidate normalization and ranking (#303, ADR 0006 §6-§7).
//!
//! #216's evidence-assembly layer needs one typed candidate that graph
//! associations, graph activations, passage hits, and community hits
//! all normalize into, because none of those four lanes' own score
//! fields (`AssociationOut::weight`, `ActivationOut::strength`,
//! `PassageHit::score`, `CommunityHit::score`) share a unit, a sign
//! convention, or a range — ADR 0006 §2.2 documents this as already
//! true of the existing wire types, not a new constraint this module
//! invents. [`EvidenceCandidate`] carries every lane's provenance
//! (context, source, paragraph, section, lane, lane rank, graph path,
//! signed weight, attribution set) without ever comparing those four
//! scales against each other; [`fuse`] is the one place cross-lane
//! comparison is allowed, and it compares ranks, never raw scores
//! (ADR 0006 §7).
//!
//! #305's `assemble` submodule is this module's first real, non-test
//! caller: its `POST /contexts/{name}/evidence` handler builds
//! [`EvidenceCandidate`]s from a live corpus, runs them through
//! [`fuse`] and `select::select`, and serializes the result as ADR
//! 0006 §10's `EvidencePackage`.

pub(crate) mod assemble;
pub(crate) mod budget;
pub(crate) mod rerank;
pub(crate) mod select;

use std::collections::{BTreeSet, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::communities::CommunityHit;
use super::sources::{Citation, PassageHit};
use super::{ActivationOut, AssociationOut};

/// Every candidate `kind` this module produces. A plain, open string on
/// the wire — never a closed Rust enum — per ADR 0005 §8's third
/// bullet ("any … field with a closed set of values is designed open
/// from the start"), which ADR 0006 §10 restates for this feature
/// specifically ("`kind` … [is a] plain, open string").
pub(crate) const KIND_ASSOCIATION: &str = "association";
pub(crate) const KIND_PASSAGE: &str = "passage";
pub(crate) const KIND_COMMUNITY: &str = "community";

/// Every `lane` name this module produces (ADR 0006 §6). Open strings,
/// same reasoning as `KIND_*` above.
pub(crate) const LANE_GRAPH_QUERY: &str = "graph_query";
pub(crate) const LANE_GRAPH_ACTIVATE: &str = "graph_activate";
pub(crate) const LANE_PASSAGE_BM25: &str = "passage_bm25";
pub(crate) const LANE_PASSAGE_VECTOR: &str = "passage_vector";
pub(crate) const LANE_PASSAGE_FUSED: &str = "passage_fused";
pub(crate) const LANE_COMMUNITY: &str = "community";

/// Reciprocal rank fusion constant (ADR 0006 §7). Matches the existing
/// passage-lane fusion's own `RRF_K` (`src/registry/search.rs:740`) —
/// there is no requirement the two agree (§7 says the constant is
/// "not required to match any particular value"), but reusing the
/// house number avoids introducing a second, unrelated-looking
/// constant with no stated reason to differ.
const RRF_K: f64 = 60.0;

/// The kind-specific payload an [`EvidenceCandidate`] carries,
/// embedding the *existing* wire types verbatim (ADR 0006 §10: "no
/// parallel type is minted"). An activation's own `ActivationOut` is
/// never stored — only its inner `AssociationOut` — because ADR 0006
/// §10's `EvidenceItem` embeds `AssociationOut`, never `ActivationOut`,
/// for both `association` and `activation`-lane candidates.
///
/// Deliberately does not derive `Debug`: [`EvidenceCandidate`]'s own
/// hand-written `Debug` impl relies on this type staying un-`Debug`-able
/// so corpus/citation text can never reach a log line through it (ADR
/// 0006 §12). Do not add a `Debug` derive here without also updating
/// that impl to keep redacting `payload`.
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum CandidatePayload {
    Association(AssociationOut),
    Passage(PassageHit),
    Community(CommunityHit),
}

/// One normalized piece of evidence from any lane, carrying full
/// provenance (ADR 0006 §6). Nothing outside [`fuse`] may compare two
/// candidates' `signed_weight`/payload score fields against each other
/// when they come from different `kind`s — comparing `lane_rank` is
/// only valid within the same `lane`.
///
/// `Debug` is implemented by hand, not derived, so it never renders
/// `payload`'s corpus text or attribution locators beyond the
/// identifying fields below — the same discipline ADR 0006 §12
/// requires of the reranker boundary ("candidate text … nowhere else
/// … never into a log line"), made structural here rather than
/// something every future caller has to remember not to log.
///
/// `kind`/`lane` are owned `String`s, not `&'static str`, purely so
/// this type can round-trip through `Deserialize` (a borrowed
/// `'static` string can never come from a deserializer reading an
/// owned buffer). [`KIND_ASSOCIATION`]/[`LANE_GRAPH_QUERY`]-and-siblings
/// stay the single source of truth for the actual values; every
/// constructor below assigns from one of those constants.
#[derive(Serialize, Deserialize)]
pub(crate) struct EvidenceCandidate {
    pub(crate) context: String,
    pub(crate) kind: String,
    /// `(subject, label, object)` NUL-joined for an association/
    /// activation candidate, `(context, source, paragraph)` NUL-joined
    /// for a passage/community candidate — one definition, two
    /// consumers: [`EvidenceCandidate::candidate_id`] and [`fuse`]'s
    /// tie-break (ADR 0006 §7).
    pub(crate) canonical_key: String,
    /// The opaque, deterministic identity every candidate has, even one
    /// with no locator at all (ADR 0006 §6): `kind` NUL-joined with
    /// `canonical_key`. The NUL delimiter is the same convention
    /// `citationKey` (`sdk/typescript/src/models.ts:833-839`) already
    /// uses so no field value can collide with it.
    pub(crate) candidate_id: String,
    /// Primary attribution's source (association/activation), or the
    /// hit's own source (passage/community). `None` only for an
    /// unsourced graph fact with zero attributions.
    pub(crate) source: Option<String>,
    pub(crate) paragraph: Option<u32>,
    pub(crate) section: Option<String>,
    pub(crate) lane: String,
    /// 1-based rank within that lane's own pool — the one number ADR
    /// 0006 §7 allows comparing across lanes, and only after RRF.
    pub(crate) lane_rank: usize,
    /// `ActivationOut::path` verbatim when this candidate came from
    /// `graph_activate`; `None` otherwise. Never invented for a lane
    /// that has no path concept.
    pub(crate) graph_path: Option<Vec<String>>,
    /// `AssociationOut::weight` / `ActivationOut::strength` for graph
    /// candidates; `None` for passage/community, which have no signed
    /// weight at all.
    pub(crate) signed_weight: Option<f64>,
    /// The full set of `(source, paragraph)` pairs this candidate's
    /// evidence is attributed to (ADR 0006 §6). Never serialized to
    /// the wire as `origins` itself — [`EvidenceCandidate::citation_refs`]
    /// and a future `corroboration` projection (#305) are its two
    /// public projections.
    pub(crate) origins: BTreeSet<(String, u32)>,
    pub(crate) payload: CandidatePayload,
}

impl EvidenceCandidate {
    /// One `graph_query` candidate per `AssociationOut` (ADR 0006 §6).
    /// `lane_rank` is the association's 1-based position in the page
    /// order the caller already produced (`api::rank`, `src/api.rs:1309`)
    /// — never recomputed here.
    pub(crate) fn from_association(
        context: &str,
        association: AssociationOut,
        lane_rank: usize,
    ) -> Self {
        let AssociationProvenance {
            source,
            paragraph,
            section,
            origins,
            canonical_key,
        } = association_provenance(&association);
        let signed_weight = association.weight;
        Self {
            context: context.to_string(),
            kind: KIND_ASSOCIATION.to_string(),
            candidate_id: candidate_id(KIND_ASSOCIATION, &canonical_key),
            canonical_key,
            source,
            paragraph,
            section,
            lane: LANE_GRAPH_QUERY.to_string(),
            lane_rank,
            graph_path: None,
            signed_weight: Some(signed_weight),
            origins,
            payload: CandidatePayload::Association(association),
        }
    }

    /// One `graph_activate` candidate per `ActivationOut` (ADR 0006
    /// §6). `lane_rank` is the activation's 1-based position in
    /// `ActivationOut`'s own strength-sorted order — never recomputed
    /// here. The inner `AssociationOut` becomes the payload; the
    /// `ActivationOut` wrapper itself is not kept (see
    /// [`CandidatePayload`]'s doc comment).
    pub(crate) fn from_activation(
        context: &str,
        activation: ActivationOut,
        lane_rank: usize,
    ) -> Self {
        let ActivationOut {
            strength,
            path,
            association,
        } = activation;
        let AssociationProvenance {
            source,
            paragraph,
            section,
            origins,
            canonical_key,
        } = association_provenance(&association);
        Self {
            context: context.to_string(),
            kind: KIND_ASSOCIATION.to_string(),
            candidate_id: candidate_id(KIND_ASSOCIATION, &canonical_key),
            canonical_key,
            source,
            paragraph,
            section,
            lane: LANE_GRAPH_ACTIVATE.to_string(),
            lane_rank,
            graph_path: Some(path),
            signed_weight: Some(strength),
            origins,
            payload: CandidatePayload::Association(association),
        }
    }

    /// One `passage_bm25`/`passage_vector`/`passage_fused` candidate
    /// per `PassageHit` (ADR 0006 §6), lane chosen from which of
    /// `PassageLanes`'s two lanes actually ran. `lane_rank` is the
    /// hit's 1-based position in the incoming `PassageHit` list — this
    /// does not re-fuse `PassageLanes.bm25`/`.vector`; it treats the
    /// whole hit as one lane, matching ADR 0006 §7's instruction.
    /// `fuse_passage_lanes` (`src/registry/search.rs:775-777`) never
    /// emits a hit with neither lane populated, so that case is
    /// unreachable from the real pipeline; it is handled defensively
    /// here as `passage_bm25` rather than panicking.
    pub(crate) fn from_passage(context: &str, hit: PassageHit, lane_rank: usize) -> Self {
        let lane = match (hit.lanes.bm25.is_some(), hit.lanes.vector.is_some()) {
            (true, true) => LANE_PASSAGE_FUSED,
            (false, true) => LANE_PASSAGE_VECTOR,
            _ => LANE_PASSAGE_BM25,
        };
        let source = hit.source.clone();
        let paragraph = hit.paragraph;
        let canonical_key = passage_key(context, &source, paragraph);
        let origins = BTreeSet::from([(source.clone(), paragraph)]);
        Self {
            context: context.to_string(),
            kind: KIND_PASSAGE.to_string(),
            candidate_id: candidate_id(KIND_PASSAGE, &canonical_key),
            canonical_key,
            source: Some(source),
            paragraph: Some(paragraph),
            section: None,
            lane: lane.to_string(),
            lane_rank,
            graph_path: None,
            signed_weight: None,
            origins,
            payload: CandidatePayload::Passage(hit),
        }
    }

    /// One `community` candidate per `CommunityHit` (ADR 0006 §6).
    /// `lane_rank` is the hit's 1-based position in the incoming
    /// `CommunityHit` list. `source` is built as `community:{id}`
    /// (`src/api/communities.rs:236-238`'s own convention), which is
    /// exactly why ADR 0006 §9 can assert a passage hit and a community
    /// hit can never share a `(source, paragraph)` locator — community
    /// summaries live in a source-id space passage sources never use.
    pub(crate) fn from_community(context: &str, hit: CommunityHit, lane_rank: usize) -> Self {
        let source = format!("community:{}", hit.community);
        let paragraph = hit.paragraph;
        let canonical_key = passage_key(context, &source, paragraph);
        let origins = BTreeSet::from([(source.clone(), paragraph)]);
        Self {
            context: context.to_string(),
            kind: KIND_COMMUNITY.to_string(),
            candidate_id: candidate_id(KIND_COMMUNITY, &canonical_key),
            canonical_key,
            source: Some(source),
            paragraph: Some(paragraph),
            section: None,
            lane: LANE_COMMUNITY.to_string(),
            lane_rank,
            graph_path: None,
            signed_weight: None,
            origins,
            payload: CandidatePayload::Community(hit),
        }
    }

    /// Every attribution locator this candidate's underlying evidence
    /// carries — one of `origins`'s two public projections (ADR 0006
    /// §6). Empty for an unsourced graph fact or a passage/community
    /// hit whose own `(source, paragraph)` is already the item's own
    /// locator (self-citing — no separate reference needed).
    pub(crate) fn citation_refs(&self) -> Vec<(String, u32)> {
        match &self.payload {
            CandidatePayload::Association(_) => self.origins.iter().cloned().collect(),
            CandidatePayload::Passage(_) | CandidatePayload::Community(_) => Vec::new(),
        }
    }

    /// The distinct sources this candidate's evidence is attributed to
    /// — `origins`'s other public projection, and the basis for a
    /// future `corroboration.sources` wire field (#305). Reads
    /// straight off the embedded `AssociationOut::attributions` rather
    /// than `origins` itself, so a source whose attribution carries no
    /// `paragraph` (and therefore cannot enter `origins`, a
    /// `(source, u32)` set) still counts as a corroborating source.
    pub(crate) fn corroborating_sources(&self) -> BTreeSet<String> {
        match &self.payload {
            CandidatePayload::Association(association) => association
                .attributions
                .iter()
                .map(|attribution| attribution.source.clone())
                .collect(),
            _ => self.source.iter().cloned().collect::<BTreeSet<_>>(),
        }
    }
}

impl std::fmt::Debug for EvidenceCandidate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvidenceCandidate")
            .field("candidate_id", &self.candidate_id)
            .field("kind", &self.kind)
            .field("lane", &self.lane)
            .field("lane_rank", &self.lane_rank)
            .finish_non_exhaustive()
    }
}

/// `kind` NUL-joined with `canonical_key` (ADR 0006 §6).
fn candidate_id(kind: &str, canonical_key: &str) -> String {
    format!("{kind}\u{0}{canonical_key}")
}

/// `{subject}\0{label}\0{object}` (ADR 0006 §6-§7) — matches ADR §7's
/// literal definition of `canonical_key` for an association/activation
/// candidate. `kind` is deliberately not embedded here; it is folded in
/// exactly once, at the `candidate_id` layer above, so a candidate's id
/// never carries its own kind twice.
fn association_key(subject: &str, label: &str, object: &str) -> String {
    format!("{subject}\u{0}{label}\u{0}{object}")
}

/// `{context}\0{source}\0{paragraph}` for a passage or community
/// candidate (ADR 0006 §6-§7) — matches ADR §7's literal definition of
/// `canonical_key`, the same as `association_key` above: `kind` is
/// folded in only once, at the `candidate_id` layer, never here. A
/// passage and a community candidate sharing one `(context, source,
/// paragraph)` is structurally impossible anyway (ADR 0006 §9 —
/// `CommunityHit`'s source always carries a `community:` prefix), so
/// this key alone only ever needs to be unique within one kind, which
/// it already is.
fn passage_key(context: &str, source: &str, paragraph: u32) -> String {
    format!("{context}\u{0}{source}\u{0}{paragraph}")
}

/// The provenance block `from_association`/`from_activation` both need
/// from the same `AssociationOut`, computed once so the two
/// constructors don't duplicate the same three calls.
struct AssociationProvenance {
    source: Option<String>,
    paragraph: Option<u32>,
    section: Option<String>,
    origins: BTreeSet<(String, u32)>,
    canonical_key: String,
}

fn association_provenance(association: &AssociationOut) -> AssociationProvenance {
    let (source, paragraph, section) = primary_attribution(&association.attributions);
    let origins = origins_of(&association.attributions);
    let canonical_key = association_key(
        &association.subject,
        &association.label,
        &association.object,
    );
    AssociationProvenance {
        source,
        paragraph,
        section,
        origins,
        canonical_key,
    }
}

/// The primary attribution for a graph candidate carrying zero or more
/// attributions (ADR 0006 §6): deterministically the attribution with
/// the greatest `|weight|`, ties broken by the lexicographically
/// smallest `source`, preferring a real `paragraph` locator over none,
/// then smallest `paragraph`. A candidate with zero attributions
/// returns `(None, None, None)`.
fn primary_attribution(
    attributions: &[super::AttributionOut],
) -> (Option<String>, Option<u32>, Option<String>) {
    attributions
        .iter()
        .max_by(|a, b| {
            a.weight
                .abs()
                .total_cmp(&b.weight.abs())
                .then_with(|| b.source.cmp(&a.source))
                // `None < Some(_)`, and every comparison here is
                // reversed (`b.cmp(&a)`) to prefer the smallest value
                // — which would make a paragraph-less attribution beat
                // one with a real locator. Compare presence first so a
                // locator is never discarded in favor of none.
                .then_with(|| a.paragraph.is_some().cmp(&b.paragraph.is_some()))
                .then_with(|| b.paragraph.cmp(&a.paragraph))
        })
        .map(|attribution| {
            (
                Some(attribution.source.clone()),
                attribution.paragraph,
                attribution.section.clone(),
            )
        })
        .unwrap_or((None, None, None))
}

/// Every `(source, paragraph)` pair a graph candidate's attributions
/// carry — attributions without a paragraph contribute nothing, the
/// same rule `locator_keys` (`src/api.rs:1622-1630`) already applies.
fn origins_of(attributions: &[super::AttributionOut]) -> BTreeSet<(String, u32)> {
    attributions
        .iter()
        .filter_map(|attribution| {
            attribution
                .paragraph
                .map(|paragraph| (attribution.source.clone(), paragraph))
        })
        .collect()
}

/// One `(lane, rank)` pair a fused candidate's underlying evidence
/// contributed (ADR 0006 §10's `lane_ranks`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LaneRank {
    pub(crate) lane: String,
    pub(crate) rank: usize,
}

/// One candidate after reciprocal rank fusion (ADR 0006 §7): the
/// surviving representative of every lane appearance the same
/// underlying evidence produced, its 1-based `fused_rank`, and which
/// lanes (at which rank) contributed.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct FusedCandidate {
    pub(crate) candidate: EvidenceCandidate,
    pub(crate) lane_ranks: Vec<LaneRank>,
    pub(crate) fused_rank: usize,
    /// The raw RRF sum `fused_rank`/`fused_rank_order` were computed
    /// from — kept so tests and callers can observe the score directly
    /// instead of recomputing it from `lane_ranks`, but never
    /// serialized: ADR 0006 §7 is explicit that `fused_score` itself
    /// "is never serialized (only the ordinal `fused_rank` reaches the
    /// wire)". `#[serde(skip)]` makes that a property of the type
    /// rather than something #305's eventual wire projection has to
    /// remember to leave out.
    #[serde(skip)]
    #[allow(dead_code)] // read by tests only — never serialized, and #305 has no other reader
    pub(crate) fused_score: f64,
}

/// One `(source, paragraph)` citation, collected once regardless of how
/// many admitted items reference it (ADR 0006 §3 E, §6): the
/// package's `citations` array holds each citation's text exactly
/// once, never duplicated per referencing item.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CitationEntry {
    pub(crate) source: String,
    pub(crate) paragraph: u32,
    pub(crate) citation: Citation,
}

/// Collects citation entries, deduplicated by `(source, paragraph)` —
/// the same locator pair `run_retrieve`'s own citation dedup already
/// keys on (`src/mcp/retrieve.rs:245-265`). Insertion order of first
/// appearance is preserved via `entries`, so `seen`'s only job is
/// membership testing — a `HashSet`, not a `BTreeSet`, since nothing
/// reads it in sorted order; callers that need a stable wire order
/// sort by `(source, paragraph)` separately.
#[derive(Default)]
pub(crate) struct CitationCollector {
    seen: HashSet<(String, u32)>,
    entries: Vec<CitationEntry>,
}

impl CitationCollector {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Adds one citation locator's text if `(source, paragraph)` was
    /// not already collected. Returns whether it was newly added.
    pub(crate) fn insert(&mut self, source: String, paragraph: u32, citation: Citation) -> bool {
        if !self.seen.insert((source.clone(), paragraph)) {
            return false;
        }
        self.entries.push(CitationEntry {
            source,
            paragraph,
            citation,
        });
        true
    }

    /// Whether `(source, paragraph)` was already collected — lets a
    /// caller check membership without attempting (and discarding) an
    /// insert, e.g. to cost a not-yet-committed candidate against what
    /// a package already carries.
    pub(crate) fn contains(&self, source: &str, paragraph: u32) -> bool {
        self.seen.contains(&(source.to_string(), paragraph))
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn into_entries(self) -> Vec<CitationEntry> {
        self.entries
    }
}

/// The total order every fused candidate list sorts by (ADR 0006 §7):
/// `fused_score` descending via `f64::total_cmp`, ties broken on
/// `(kind, canonical_key)` lexicographic order — fully deterministic,
/// matching the graph page's own existing tie-break discipline
/// (`api::rank`, `src/api.rs:1309-1315`). RRF terms are always finite
/// (`1.0 / (RRF_K + rank)` over a `usize` rank), so `total_cmp` never
/// meets a NaN here the way a raw, unvalidated score could.
fn fused_rank_order(a: &(f64, &str, &str), b: &(f64, &str, &str)) -> std::cmp::Ordering {
    b.0.total_cmp(&a.0)
        .then_with(|| a.1.cmp(b.1))
        .then_with(|| a.2.cmp(b.2))
}

/// Reciprocal rank fusion over a pool of normalized candidates (ADR
/// 0006 §7). Candidates sharing a `candidate_id` are grouped into one
/// — this grouping *is* ADR 0006 §9's exact-key dedup (association on
/// `(subject, label, object)`; passage/community on
/// `(context, source, paragraph)`), which is what lets §7 say a
/// candidate's fused score sums every lane's contribution: two lane
/// appearances of the same evidence are one candidate by the time this
/// function groups them, never two.
///
/// Returns the fused, deterministically ordered candidates plus how
/// many raw candidates were folded into an existing group (the input
/// count minus the output count) — the basis for #304's
/// `SelectionPlan.dedup_dropped`.
pub(crate) fn fuse(pool: Vec<EvidenceCandidate>) -> (Vec<FusedCandidate>, usize) {
    let input_len = pool.len();

    // Group by candidate_id, keeping every group's lowest-lane_rank
    // member as the surviving representative (so graph_path/
    // signed_weight provenance survives) and unioning every member's
    // origins (so corroboration is never truncated — I4). Written as
    // an explicit get_mut/insert match, not entry().and_modify()/
    // or_insert_with(), so `candidate` is moved into exactly one arm
    // instead of needing a Clone impl the embedded wire types (e.g.
    // PassageHit, CommunityHit) don't have — and so a merge (the
    // common case once cross-lane overlap exists) costs zero extra key
    // clones, unlike `entry()`, which always clones its key up front
    // regardless of hit or miss. Iteration order of `groups`/
    // `best_rank_by_lane` never reaches the wire — the explicit sorts
    // below are what make the result deterministic — so plain
    // `HashMap`s are enough; nothing here needs `BTreeMap`'s
    // sorted-key bookkeeping.
    struct Group {
        representative: EvidenceCandidate,
        best_rank_by_lane: HashMap<String, usize>,
    }

    let mut groups: HashMap<String, Group> = HashMap::new();
    for mut candidate in pool {
        match groups.get_mut(&candidate.candidate_id) {
            Some(group) => {
                // A lane appearing twice in one group keeps its best
                // (lowest) rank — mirrors fuse_passage_lanes's "first
                // arrival is its best showing" (src/registry/search.rs:747-754).
                group
                    .best_rank_by_lane
                    .entry(candidate.lane.clone())
                    .and_modify(|rank| *rank = (*rank).min(candidate.lane_rank))
                    .or_insert(candidate.lane_rank);
                // Union the two origin sets by moving elements rather
                // than cloning them (`BTreeSet::append` drains `other`
                // into `self`) — nothing downstream reads either set's
                // pre-merge contents, so a clone here would be pure
                // waste.
                //
                // Strict `<` on `lane_rank` alone leaves an equal-rank
                // tie (e.g. the same triple arriving from both
                // `graph_query` and `graph_activate` at rank 1) decided
                // by pool order — whichever copy this loop happens to
                // see first. Breaking the tie on `lane` (deterministic,
                // and already this candidate's own field) makes the
                // survivor — and with it `graph_path`/`signed_weight` —
                // a function of the pool's contents, not its order.
                let challenger = (candidate.lane_rank, candidate.lane.as_str());
                let incumbent = (
                    group.representative.lane_rank,
                    group.representative.lane.as_str(),
                );
                if challenger < incumbent {
                    let mut merged = std::mem::take(&mut group.representative.origins);
                    merged.append(&mut candidate.origins);
                    candidate.origins = merged;
                    group.representative = candidate;
                } else {
                    group.representative.origins.append(&mut candidate.origins);
                }
            }
            None => {
                let mut best_rank_by_lane = HashMap::new();
                best_rank_by_lane.insert(candidate.lane.clone(), candidate.lane_rank);
                groups.insert(
                    candidate.candidate_id.clone(),
                    Group {
                        representative: candidate,
                        best_rank_by_lane,
                    },
                );
            }
        }
    }

    let dedup_dropped = input_len - groups.len();

    let mut fused: Vec<(f64, EvidenceCandidate, Vec<LaneRank>)> = groups
        .into_values()
        .map(|group| {
            let fused_score: f64 = group
                .best_rank_by_lane
                .values()
                .map(|&rank| 1.0 / (RRF_K + rank as f64))
                .sum();
            let mut lane_ranks: Vec<LaneRank> = group
                .best_rank_by_lane
                .into_iter()
                .map(|(lane, rank)| LaneRank { lane, rank })
                .collect();
            lane_ranks.sort_by(|a, b| a.lane.cmp(&b.lane));
            (fused_score, group.representative, lane_ranks)
        })
        .collect();

    fused.sort_by(|(score_a, candidate_a, _), (score_b, candidate_b, _)| {
        fused_rank_order(
            &(
                *score_a,
                candidate_a.kind.as_str(),
                candidate_a.canonical_key.as_str(),
            ),
            &(
                *score_b,
                candidate_b.kind.as_str(),
                candidate_b.canonical_key.as_str(),
            ),
        )
    });

    let fused = fused
        .into_iter()
        .enumerate()
        .map(
            |(index, (fused_score, candidate, lane_ranks))| FusedCandidate {
                candidate,
                lane_ranks,
                fused_rank: index + 1,
                fused_score,
            },
        )
        .collect();

    (fused, dedup_dropped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn association_out(subject: &str, label: &str, object: &str, weight: f64) -> AssociationOut {
        AssociationOut {
            subject: subject.to_string(),
            label: label.to_string(),
            object: object.to_string(),
            weight,
            count: 1,
            attributions: Vec::new(),
        }
    }

    fn attribution_out(
        source: &str,
        weight: f64,
        paragraph: Option<u32>,
    ) -> super::super::AttributionOut {
        super::super::AttributionOut {
            source: source.to_string(),
            weight,
            count: 1,
            paragraph,
            section: None,
            locator: None,
        }
    }

    fn passage_hit(source: &str, paragraph: u32, bm25: bool, vector: bool) -> PassageHit {
        use super::super::sources::{LaneEvidence, PassageLanes};
        PassageHit {
            source: source.to_string(),
            paragraph,
            score: 1.0,
            text: "text".to_string(),
            lanes: PassageLanes {
                bm25: bm25.then_some(LaneEvidence {
                    rank: 1,
                    score: 1.0,
                }),
                vector: vector.then_some(LaneEvidence {
                    rank: 1,
                    score: 1.0,
                }),
            },
        }
    }

    fn community_hit(id: &str, paragraph: u32) -> CommunityHit {
        CommunityHit {
            community: id.to_string(),
            score: 1.0,
            text: "summary".to_string(),
            paragraph,
            level: None,
            parent: None,
            concept_count: None,
            members: Vec::new(),
            members_truncated: false,
        }
    }

    // --- Normalization ---

    #[test]
    fn association_normalizes_kind_lane_and_keys() {
        let association = association_out("cats", "is_a", "mammals", 2.5);
        let candidate = EvidenceCandidate::from_association("ctx", association, 3);
        assert_eq!(candidate.kind, KIND_ASSOCIATION);
        assert_eq!(candidate.lane, LANE_GRAPH_QUERY);
        assert_eq!(candidate.lane_rank, 3);
        assert_eq!(candidate.signed_weight, Some(2.5));
        assert_eq!(candidate.graph_path, None);
        assert_eq!(candidate.canonical_key, "cats\u{0}is_a\u{0}mammals");
        assert_eq!(
            candidate.candidate_id,
            "association\u{0}cats\u{0}is_a\u{0}mammals"
        );
    }

    #[test]
    fn activation_keeps_path_and_strength_and_inner_association() {
        let mut association = association_out("cats", "is_a", "mammals", 2.5);
        association.attributions = vec![attribution_out("book.txt", 2.5, Some(4))];
        let activation = ActivationOut {
            strength: 0.75,
            path: vec!["cats".to_string(), "mammals".to_string()],
            association,
        };
        let candidate = EvidenceCandidate::from_activation("ctx", activation, 1);
        assert_eq!(candidate.kind, KIND_ASSOCIATION);
        assert_eq!(candidate.lane, LANE_GRAPH_ACTIVATE);
        assert_eq!(
            candidate.graph_path,
            Some(vec!["cats".to_string(), "mammals".to_string()])
        );
        assert_eq!(candidate.signed_weight, Some(0.75));
        match &candidate.payload {
            CandidatePayload::Association(inner) => assert_eq!(inner.subject, "cats"),
            _ => panic!("expected an association payload"),
        }
    }

    #[test]
    fn primary_attribution_prefers_greatest_magnitude_including_negative() {
        let mut association = association_out("cats", "is_a", "mammals", -1.0);
        association.attributions = vec![
            attribution_out("small.txt", 0.4, Some(1)),
            attribution_out("big.txt", -3.0, Some(2)),
        ];
        let candidate = EvidenceCandidate::from_association("ctx", association, 1);
        assert_eq!(candidate.source, Some("big.txt".to_string()));
        assert_eq!(candidate.paragraph, Some(2));
    }

    #[test]
    fn primary_attribution_ties_break_on_source_then_paragraph() {
        let mut association = association_out("cats", "is_a", "mammals", 1.0);
        association.attributions = vec![
            attribution_out("same.txt", 1.0, Some(9)),
            attribution_out("same.txt", 1.0, Some(2)),
            attribution_out("zzz.txt", 1.0, Some(1)),
        ];
        let candidate = EvidenceCandidate::from_association("ctx", association, 1);
        assert_eq!(candidate.source, Some("same.txt".to_string()));
        assert_eq!(candidate.paragraph, Some(2));
    }

    #[test]
    fn primary_attribution_prefers_a_real_locator_over_none_on_a_tie() {
        let mut association = association_out("cats", "is_a", "mammals", 1.0);
        association.attributions = vec![
            attribution_out("same.txt", 1.0, None),
            attribution_out("same.txt", 1.0, Some(4)),
        ];
        let candidate = EvidenceCandidate::from_association("ctx", association, 1);
        assert_eq!(candidate.source, Some("same.txt".to_string()));
        assert_eq!(candidate.paragraph, Some(4));
    }

    #[test]
    fn zero_attributions_leave_locator_and_origins_empty() {
        let association = association_out("cats", "is_a", "mammals", 1.0);
        let candidate = EvidenceCandidate::from_association("ctx", association, 1);
        assert_eq!(candidate.source, None);
        assert_eq!(candidate.paragraph, None);
        assert_eq!(candidate.section, None);
        assert!(candidate.origins.is_empty());
        assert!(candidate.citation_refs().is_empty());
    }

    #[test]
    fn passage_lane_labeling_covers_all_reachable_shapes() {
        let bm25_only = EvidenceCandidate::from_passage("ctx", passage_hit("s", 0, true, false), 1);
        let vector_only =
            EvidenceCandidate::from_passage("ctx", passage_hit("s", 1, false, true), 1);
        let fused = EvidenceCandidate::from_passage("ctx", passage_hit("s", 2, true, true), 1);
        assert_eq!(bm25_only.lane, LANE_PASSAGE_BM25);
        assert_eq!(vector_only.lane, LANE_PASSAGE_VECTOR);
        assert_eq!(fused.lane, LANE_PASSAGE_FUSED);
    }

    #[test]
    fn community_candidate_self_cites_and_uses_community_source_prefix() {
        let candidate = EvidenceCandidate::from_community("ctx", community_hit("L0-3", 5), 1);
        assert_eq!(candidate.source, Some("community:L0-3".to_string()));
        assert_eq!(candidate.paragraph, Some(5));
        assert!(candidate.citation_refs().is_empty());
        assert_eq!(
            candidate.origins,
            BTreeSet::from([("community:L0-3".to_string(), 5)])
        );
    }

    #[test]
    fn candidate_id_does_not_collide_across_field_boundaries() {
        // Without a delimiter, ("ab", "c") and ("a", "bc") would both
        // render as "abc" — the NUL join must keep them apart.
        let left = association_key("ab", "c", "x");
        let right = association_key("a", "bc", "x");
        assert_ne!(left, right);
    }

    // --- Ranking ---

    #[test]
    fn fusion_is_deterministic_under_reordering() {
        // EvidenceCandidate intentionally has no Clone impl (its
        // payload embeds wire types that don't derive one either), so
        // "the same pool in a different order" is built by calling the
        // same constructors twice rather than cloning one pool.
        fn build_pool() -> Vec<EvidenceCandidate> {
            vec![
                EvidenceCandidate::from_association("ctx", association_out("a", "l", "b", 1.0), 1),
                EvidenceCandidate::from_passage("ctx", passage_hit("s1", 0, true, false), 2),
                EvidenceCandidate::from_community("ctx", community_hit("L0-1", 0), 1),
            ]
        }
        let mut pool_b = build_pool();
        pool_b.reverse();

        let (fused_a, _) = fuse(build_pool());
        let (fused_b, _) = fuse(pool_b);

        let ids_a: Vec<_> = fused_a
            .iter()
            .map(|f| f.candidate.candidate_id.clone())
            .collect();
        let ids_b: Vec<_> = fused_b
            .iter()
            .map(|f| f.candidate.candidate_id.clone())
            .collect();
        assert_eq!(ids_a, ids_b);
    }

    #[test]
    fn same_evidence_across_two_lanes_outranks_a_single_lane_candidate_at_same_rank() {
        let query =
            EvidenceCandidate::from_association("ctx", association_out("a", "l", "b", 1.0), 1);
        let mut activation_assoc = association_out("a", "l", "b", 1.0);
        activation_assoc.attributions = Vec::new();
        let activation = EvidenceCandidate::from_activation(
            "ctx",
            ActivationOut {
                strength: 0.5,
                path: vec!["a".to_string(), "b".to_string()],
                association: activation_assoc,
            },
            1,
        );
        let single_lane =
            EvidenceCandidate::from_association("ctx", association_out("c", "l", "d", 1.0), 1);

        let (fused, dedup_dropped) = fuse(vec![query, activation, single_lane]);
        assert_eq!(dedup_dropped, 1); // query + activate merge into one candidate

        let merged = fused
            .iter()
            .find(|f| f.candidate.canonical_key.contains("a\u{0}l\u{0}b"))
            .expect("merged candidate present");
        let solo = fused
            .iter()
            .find(|f| f.candidate.canonical_key.contains("c\u{0}l\u{0}d"))
            .expect("solo candidate present");
        assert!(merged.fused_rank < solo.fused_rank);
        assert_eq!(merged.lane_ranks.len(), 2);
    }

    #[test]
    fn equal_rank_tie_across_lanes_picks_the_same_representative_regardless_of_pool_order() {
        // graph_query and graph_activate both rank this triple 1st —
        // an equal-rank tie the merge loop's lane_rank comparison alone
        // cannot break. Without a tie-break on lane, whichever
        // candidate this loop visits first would survive, making
        // `graph_path`/`signed_weight` a function of pool order rather
        // than pool contents.
        fn query() -> EvidenceCandidate {
            EvidenceCandidate::from_association("ctx", association_out("a", "l", "b", 1.0), 1)
        }
        fn activate() -> EvidenceCandidate {
            let mut association = association_out("a", "l", "b", 1.0);
            association.attributions = Vec::new();
            EvidenceCandidate::from_activation(
                "ctx",
                ActivationOut {
                    strength: 0.5,
                    path: vec!["a".to_string(), "b".to_string()],
                    association,
                },
                1,
            )
        }

        let (query_first, _) = fuse(vec![query(), activate()]);
        let (activate_first, _) = fuse(vec![activate(), query()]);

        assert_eq!(query_first.len(), 1);
        assert_eq!(activate_first.len(), 1);
        assert_eq!(
            query_first[0].candidate.lane,
            activate_first[0].candidate.lane
        );
        assert_eq!(
            query_first[0].candidate.graph_path,
            activate_first[0].candidate.graph_path
        );
        assert_eq!(
            query_first[0].candidate.signed_weight,
            activate_first[0].candidate.signed_weight
        );
    }

    #[test]
    fn ties_break_on_kind_then_canonical_key_not_insertion_order() {
        let second =
            EvidenceCandidate::from_association("ctx", association_out("z", "l", "z", 1.0), 5);
        let first =
            EvidenceCandidate::from_association("ctx", association_out("a", "l", "a", 1.0), 5);
        // Insertion order is deliberately "second, first" — the fused
        // order must not follow it.
        let (fused, _) = fuse(vec![second, first]);
        assert_eq!(fused[0].candidate.canonical_key, "a\u{0}l\u{0}a");
        assert_eq!(fused[1].candidate.canonical_key, "z\u{0}l\u{0}z");
    }

    #[test]
    fn rank_one_passage_and_rank_one_association_fuse_identically() {
        let association =
            EvidenceCandidate::from_association("ctx", association_out("a", "l", "b", 999.0), 1);
        let passage = EvidenceCandidate::from_passage("ctx", passage_hit("s", 0, true, false), 1);
        let (fused, _) = fuse(vec![association, passage]);
        // Equal lane_rank (1) on two different lanes must fuse to the
        // same score regardless of the wildly different raw scores
        // each carries (999.0 signed_weight vs. 1.0 passage score) —
        // neither raw score may leak into the ordering. Compares the
        // actual computed `fused_score`, not a value re-derived from
        // `lane_ranks`, so a change to the RRF formula itself would
        // fail this test rather than silently agreeing with it.
        assert_eq!(fused[0].fused_score, fused[1].fused_score);
    }

    #[test]
    fn dedup_keeps_union_of_origins_and_reports_dropped_count() {
        let mut first = association_out("a", "l", "b", 1.0);
        first.attributions = vec![attribution_out("book.txt", 1.0, Some(1))];
        let mut second = association_out("a", "l", "b", 1.0);
        second.attributions = vec![attribution_out("other.txt", 1.0, Some(2))];

        let candidate_a = EvidenceCandidate::from_association("ctx", first, 1);
        let candidate_b = EvidenceCandidate::from_association("ctx", second, 2);
        let (fused, dedup_dropped) = fuse(vec![candidate_a, candidate_b]);

        assert_eq!(dedup_dropped, 1);
        assert_eq!(fused.len(), 1);
        assert_eq!(
            fused[0].candidate.origins,
            BTreeSet::from([("book.txt".to_string(), 1), ("other.txt".to_string(), 2),])
        );
    }

    // --- Citations ---

    #[test]
    fn citation_collector_dedups_by_source_and_paragraph() {
        fn citation() -> Citation {
            Citation {
                text: "excerpt".to_string(),
                source: "book.txt".to_string(),
                section: None,
                locator: None,
            }
        }
        let mut collector = CitationCollector::new();
        assert!(collector.insert("book.txt".to_string(), 3, citation()));
        assert!(!collector.insert("book.txt".to_string(), 3, citation()));
        assert_eq!(collector.into_entries().len(), 1);
    }

    // --- Serde ---

    #[test]
    fn each_candidate_kind_round_trips_through_serde() {
        let candidates = vec![
            EvidenceCandidate::from_association("ctx", association_out("a", "l", "b", 1.0), 1),
            EvidenceCandidate::from_passage("ctx", passage_hit("s", 0, true, false), 1),
            EvidenceCandidate::from_community("ctx", community_hit("L0-1", 0), 1),
        ];
        for candidate in candidates {
            let json = serde_json::to_string(&candidate).expect("serialize");
            let restored: EvidenceCandidate = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(restored.candidate_id, candidate.candidate_id);
            assert_eq!(restored.kind, candidate.kind);
            assert_eq!(restored.lane, candidate.lane);
        }
    }

    #[test]
    fn absent_optionals_are_null_and_empty_until_a_wire_projection_skips_them() {
        let candidate =
            EvidenceCandidate::from_association("ctx", association_out("a", "l", "b", 1.0), 1);
        let json = serde_json::to_value(&candidate).expect("serialize");
        // graph_path and section are None for a plain graph_query
        // candidate with zero attributions; origins is empty. This
        // module has no #[serde(skip_serializing_if...)] annotations
        // yet since it is not itself on the wire (#305 designs the
        // wire projection) — this test locks in that the underlying
        // Option/BTreeSet values are correctly absent/empty so a
        // future wire projection has accurate inputs to skip on.
        assert!(json["graph_path"].is_null());
        assert_eq!(json["origins"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn debug_rendering_never_includes_payload_text() {
        let candidate = EvidenceCandidate::from_passage("ctx", passage_hit("s", 0, true, false), 1);
        let rendered = format!("{candidate:?}");
        assert!(rendered.contains("candidate_id"));
        // `passage_hit`'s payload text is the literal "text" — the
        // hand-written Debug impl must never surface it.
        assert!(!rendered.contains("\"text\""));
    }
}
