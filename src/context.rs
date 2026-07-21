use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

mod alias;
mod community;
mod entry_index;
mod gloss;
mod image;
#[cfg(test)]
mod proptests;
mod query;
mod resolve;
mod sources;
mod stats;
#[cfg(test)]
mod test_support;
mod traverse;
mod write;

pub use community::{
    COMMUNITY_ALGORITHM, Community, CommunityAnalysis, CommunityAssociation, CommunityMember,
};
use entry_index::EntryIndex;

/// Dense id of an interned concept string within one `Context`.
type ConceptId = u32;
/// Dense id of an interned relation-label string within one `Context`.
type LabelId = u32;
/// Dense id of an interned source string within one `Context`.
type SourceId = u32;
/// Dense id of an edge, assigned in insertion order.
type EdgeId = u32;
/// Dense id of one per-source attribution record.
type AttributionId = u32;

/// Chain terminator and "no entry" sentinel shared by every id space.
/// Ids are dense from 0 and [`claim_id`] refuses to mint this value, so it
/// can never collide with a real record.
const NIL: u32 = u32::MAX;

/// Hands out the next dense id in a table. Writes that would not fit are
/// turned into [`ContextFull`] errors by `ensure_room` before anything
/// mutates, so this panicking backstop only fires on an accounting bug,
/// never on a merely full context.
fn claim_id(len: usize, table: &str) -> u32 {
    match u32::try_from(len) {
        Ok(id) if id != NIL => id,
        _ => panic!("{table} table exceeds the u32 id space"),
    }
}

/// True when a table of `len` records can still mint `needed` more dense
/// ids: ids run from 0 and `NIL` is reserved, so a table holds at most
/// `NIL` records.
fn ids_left(len: usize, needed: usize) -> bool {
    (NIL as usize).saturating_sub(len) >= needed
}

/// True when the string arena can grow by `growth` bytes with every
/// offset — including one-past-the-end — still representable as u32.
fn arena_fits(len: usize, growth: usize) -> bool {
    len.checked_add(growth)
        .is_some_and(|end| end <= u32::MAX as usize)
}

/// Folds one more finite `weight` into a cumulative sum, saturating at
/// ±`f64::MAX` instead of overflowing to infinity. The associate
/// boundary asserts each INDIVIDUAL weight finite, but two finite
/// values can still sum past the representable range — and a non-finite
/// sum would sort as the maximum under `total_cmp` forever and make
/// `from_bytes` refuse the next image as corrupt. Saturation keeps the
/// invariant the load-time check depends on without turning a pair of
/// individually-valid library calls into a poisoned context.
fn accumulate_saturating(sum: &mut f64, weight: f64) {
    *sum += weight;
    if !sum.is_finite() {
        *sum = f64::MAX.copysign(*sum);
    }
}

/// Shared definition of "dead ratio" so [`Context::dead_ratio`] and
/// `ContextStats::dead_ratio` (registry.rs) can never drift apart: 0.0
/// when there are no associations at all, rather than dividing by zero.
pub fn dead_ratio_of(dead_edges: usize, total_edges: usize) -> f64 {
    if total_edges == 0 {
        0.0
    } else {
        dead_edges as f64 / total_edges as f64
    }
}

/// Error returned by [`Context::associate`] and [`Context::associate_from`]
/// when a write would need a record or name bytes beyond the context's u32
/// id/offset space (~4.29 billion records per table, 4 GiB of interned
/// text). The failed write is not applied — an `Err` leaves the context
/// exactly as it was — and the context stays usable: reads are unaffected,
/// and writes that still fit (e.g. accumulating weight into an existing
/// edge) keep succeeding. Knowledge that no longer fits belongs in a new
/// `Context`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextFull(&'static str);

impl fmt::Display for ContextFull {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "context is full: {}", self.0)
    }
}

impl Error for ContextFull {}

/// Error returned by [`Context::add_concept_alias`] and
/// [`Context::add_label_alias`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AliasError {
    /// The canonical spelling is not interned in this namespace.
    UnknownCanonical,
    /// The alias spelling already resolves to a different record. One
    /// spelling is one referent within a namespace — aliases included —
    /// so an alias can never shadow an existing name or alias. In
    /// particular, two spellings that BOTH already exist as concepts
    /// cannot be aliased together: that is a merge, which does not
    /// exist; rebuild the context instead.
    Conflict,
    /// The alias table or the string arena is out of space; the alias
    /// was not added and the context is unchanged.
    Full(ContextFull),
}

/// What [`Context::compacted`] left behind: the dead records shed and
/// the aliases that could not survive (their canonical carries no live
/// edge to re-intern it) — numbers for the report line, so nothing is
/// dropped silently.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CompactionStats {
    pub dead_edges: usize,
    pub aliases_dropped: usize,
}

/// Why [`Context::compacted`] did not finish. Either way the source
/// context is untouched — the rebuild only ever writes into a fresh
/// `Context` and is swapped in by the caller on full success.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactionError {
    /// Structurally unreachable — the rebuild holds a subset of what
    /// the source context already held — but the write API this
    /// delegates to says it, so this signature does too.
    Full(ContextFull),
    /// The deadline elapsed partway through the rebuild.
    DeadlineExceeded,
}

impl fmt::Display for CompactionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompactionError::Full(full) => full.fmt(f),
            CompactionError::DeadlineExceeded => write!(f, "compaction exceeded its deadline"),
        }
    }
}

impl Error for CompactionError {}

impl From<ContextFull> for CompactionError {
    fn from(full: ContextFull) -> Self {
        CompactionError::Full(full)
    }
}

impl fmt::Display for AliasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AliasError::UnknownCanonical => {
                write!(f, "the canonical spelling is not interned")
            }
            AliasError::Conflict => {
                write!(f, "the alias already resolves to a different record")
            }
            AliasError::Full(full) => full.fmt(f),
        }
    }
}

impl Error for AliasError {}

/// Reserved source id for weight that reached the graph with no named
/// source, or that an export/import round trip preserved as a bucket.
/// Never a real ingest source name.
pub const UNSOURCED_SOURCE: &str = "export:unsourced";

/// One source's accumulated contribution to an association's weight.
///
/// `source` is an opaque identifier chosen by the caller — a document id, a
/// URL, a chunk reference — that lets whoever retrieved the association go
/// fetch the original text behind it. The `Context` never interprets it,
/// except for one reserved spelling: [`UNSOURCED_SOURCE`] marks weight
/// that arrived with no source at all, and `unsourced_summary`/
/// `unsourced_edges` key off exactly that value.
/// `paragraph` optionally locates the fact within `source` (e.g. the
/// paragraph index it was read from); it is `null` rather than omitted
/// when absent, so callers can rely on the key always being present —
/// the same contract the citation endpoint's `section` field makes. It
/// reflects only the first assertion of this source (first-write-wins);
/// later re-assertions accumulate into `weight` but never change it.
/// `weight` is this source's raw cumulative total (not averaged) — the
/// invariant `sum(attributions.weight) + unsourced == association.weight *
/// association.count` holds regardless of how many times this source
/// re-asserted. `count` is how many times this source asserted the
/// association.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Attribution {
    pub source: String,
    pub weight: f64,
    pub count: u64,
    pub paragraph: Option<u32>,
}

/// A single subject-label-object association, its signed weight, and the
/// per-source breakdown of where that weight came from.
///
/// Returned instead of a positional tuple so that once this crosses a
/// serialization boundary (e.g. JSON returned to an LLM client over HTTP),
/// the field names carry the meaning of each value inline — a client doesn't
/// have to be told out-of-band that position 0 is the subject.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Association {
    pub subject: String,
    pub label: String,
    pub object: String,
    /// The average weight per assertion (`sum / count`), not the raw
    /// cumulative total. Two independent attributions of 1.0 each
    /// (corroboration) average to a `weight` of 1.0; one attribution
    /// asserting 2.0 a single time (emphasis) keeps a `weight` of 2.0 —
    /// the two cases are no longer indistinguishable the way a plain sum
    /// would make them.
    pub weight: f64,
    /// How many assertions (sourced and unsourced) contributed to `weight`.
    pub count: u64,
    /// Which sources asserted this association and how much weight each
    /// contributed. Total `weight * count` can exceed the attributed sum
    /// when some assertions came in unsourced.
    pub attributions: Vec<Attribution>,
}

/// One edge `Context::unsourced_edges` flagged: the association
/// itself, plus the weight/count no named source explains. `weight`
/// can be negative.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnsourcedEdge {
    pub association: Association,
    pub weight: f64,
    pub count: u64,
}

/// One association reached by [`Context::explore`], annotated with how many
/// hops of association-following it took to reach it from the nearest
/// origin: 1 touches an origin concept directly, 2 was reached through one
/// intermediate concept, and so on. The distance is a relevance hint for
/// whoever rebuilds prose from the result — nearer associations are more
/// central to what was asked about.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Recollection {
    pub distance: usize,
    /// Concept names from the origin that reached this association down
    /// to the endpoint it was reached at, origin first. This is the
    /// connective tissue a caller needs to recompose a multi-hop answer
    /// as prose — not just that a far fact is related, but through which
    /// intermediate concepts the relation runs.
    pub path: Vec<String>,
    pub association: Association,
}

/// One association returned by [`Context::activate`], carrying the
/// activation strength that reached it — the ranking signal that combines
/// how near the association is to the origins with how heavy it is relative
/// to its neighbors.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Activation {
    pub strength: f64,
    /// Concept names from the origin down to the endpoint whose
    /// activation scored this association, origin first — the strongest
    /// activation path, for recomposing multi-hop answers.
    pub path: Vec<String>,
    pub association: Association,
}

/// One candidate name produced by [`Context::resolve`] (concept names) or
/// [`Context::resolve_label`] (relation labels), scored by how much of the
/// longer string the lexical overlap covers (1.0 = exact match).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Resolution {
    pub name: String,
    pub score: f64,
    /// The string relation that earned the score — see [`MatchKind`].
    pub kind: MatchKind,
}

/// How a cue lexically met one of a record's spellings. The score alone
/// hides this: a containment hit on a lookalike (possible inside
/// impossible scores 0.8, 京都 inside 東京都 scores 0.67) reads like a
/// strong match while being a different thing entirely. Naming the
/// relation lets a caller weigh "this IS a stored spelling" (exact,
/// alias) against "this merely overlaps one" (containment, fuzzy)
/// before adopting a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchKind {
    /// The cue is the record's own name (normalized).
    Exact,
    /// The cue is one of the record's alias spellings — the same
    /// certainty as `Exact`, and why the returned name differs from
    /// the cue.
    Alias,
    /// The cue contains a spelling or is contained by one; the score
    /// is character coverage of the longer side.
    Containment,
    /// The cue merely shares informative bigrams with a spelling (the
    /// near-miss tier); the score is the Dice overlap.
    Fuzzy,
}

impl MatchKind {
    pub fn as_str(self) -> &'static str {
        match self {
            MatchKind::Exact => "exact",
            MatchKind::Alias => "alias",
            MatchKind::Containment => "containment",
            MatchKind::Fuzzy => "fuzzy",
        }
    }
}

/// How often one relation label appears on a concept's edges — one row of
/// a [`ConceptDescription`]. Also the wire shape for [`Context::top_concepts`]
/// as served by the routing directory, so "name plus occurrence count" has
/// one JSON representation across the whole surface.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LabelUsage {
    pub label: String,
    pub count: usize,
}

/// The outline of one concept produced by [`Context::describe`]: which
/// relation labels its edges carry and how often, split by role, without
/// materializing a single association. This is the "what is known about
/// X" overview a caller reads BEFORE fetching facts — check the outline,
/// pick the relevant labels, then [`Context::query_any`] just those —
/// so a hub concept never floods the caller with its whole profile.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConceptDescription {
    /// The stored spelling of the described concept.
    pub concept: String,
    /// Labels on edges where the concept is the subject (what it does /
    /// has), most frequent first.
    pub as_subject: Vec<LabelUsage>,
    /// Labels on edges where the concept is the object (what is said
    /// about it), most frequent first.
    pub as_object: Vec<LabelUsage>,
}

/// A concept node in fixed-width form: where its interned name lives in the
/// string arena, plus the heads, tails, and lengths of the two edge chains
/// it participates in. The chains are what make the structure a walkable
/// network rather than a flat table of rows; they thread through the edge
/// records themselves (see [`EdgeRecord`]), so this record never grows.
///
/// Layout: 8 × u32 = 32 bytes, alignment 4, no padding.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ConceptRecord {
    name_offset: u32,
    name_len: u32,
    /// Chain of edges where this concept is the subject, in insertion
    /// order: head, tail (for O(1) append), and length (so a query can
    /// pick its narrowest scan without walking anything).
    first_outgoing: EdgeId,
    last_outgoing: EdgeId,
    outgoing_count: u32,
    /// Chain of edges where this concept is the object, in insertion order.
    first_incoming: EdgeId,
    last_incoming: EdgeId,
    incoming_count: u32,
}

/// A relation label in fixed-width form: its interned name plus the chain
/// of every edge that uses it. Labels annotate edges; they are deliberately
/// NOT nodes of the network, so two unrelated facts that happen to share a
/// label (e.g. "好き") never become reachable from each other through it.
///
/// Layout: 5 × u32 = 20 bytes, alignment 4, no padding.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct LabelRecord {
    name_offset: u32,
    name_len: u32,
    first_edge: EdgeId,
    last_edge: EdgeId,
    edge_count: u32,
}

/// A source in fixed-width form: just where its interned name lives in the
/// string arena.
///
/// Layout: 2 × u32 = 8 bytes, alignment 4, no padding.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct SourceRecord {
    name_offset: u32,
    name_len: u32,
}

/// One alternative spelling in fixed-width form: where the alias string
/// lives in the arena and which record it resolves to. Aliases are
/// entry-only — they feed the lookup maps and the entry index, never
/// appear in results, and never touch the graph — so this record carries
/// no chains.
///
/// Layout: 3 × u32 = 12 bytes, alignment 4, no padding.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct AliasRecord {
    name_offset: u32,
    name_len: u32,
    target: u32,
}

/// One directed, weighted edge in fixed-width form. The three `next_*`
/// links replace per-node `Vec<EdgeId>` adjacency lists: every edge is a
/// member of exactly three chains — its subject's outgoing chain, its
/// object's incoming chain, and its label's chain — and carries the
/// successor link for each, so adjacency grows by appending edge records
/// while every record stays fixed-width. Per-source attributions hang off
/// the edge the same way, as a chain through the attribution table.
///
/// `weight` is `sum / count`, not stored directly: the record keeps the
/// raw cumulative `sum` and the assertion `count` so corroboration
/// (several assertions of the same weight) can be told apart from a single
/// emphatic assertion of the same total — see [`Association`].
///
/// Layout: 8 × u32 + 1 × u64 + 1 × f64 = 48 bytes, alignment 8, no padding
/// (the eight u32 fields put `count` at offset 32 and `sum` at offset 40).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct EdgeRecord {
    subject: ConceptId,
    label: LabelId,
    object: ConceptId,
    next_outgoing: EdgeId,
    next_incoming: EdgeId,
    next_labeled: EdgeId,
    first_attribution: AttributionId,
    last_attribution: AttributionId,
    count: u64,
    sum: f64,
}

type EdgeFollow = fn(&EdgeRecord) -> EdgeId;

// All anchors produce the same associations after filtering; this helper
// chooses only which adjacency chains cost least to scan. Comparator mutants
// are therefore behaviorally equivalent, not missing correctness assertions.
#[mutants::skip]
fn keep_narrowest_anchor(
    narrowest: &mut Option<(u64, Vec<EdgeId>, EdgeFollow)>,
    total: u64,
    firsts: Vec<EdgeId>,
    follow: EdgeFollow,
) {
    if narrowest.as_ref().is_none_or(|&(best, ..)| total < best) {
        *narrowest = Some((total, firsts, follow));
    }
}

/// One source's weight contribution to one edge, in fixed-width form; a
/// link in that edge's attribution chain, in insertion order, one record
/// per distinct source.
///
/// Layout: 2 × u32 + 1 × u64 + 1 × f64 = 24 bytes, alignment 8, no padding.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct AttributionRecord {
    source: SourceId,
    next: AttributionId,
    count: u64,
    sum: f64,
}

/// One paragraph locator for an attribution record, in fixed-width form.
/// A sparse side table: present only for attributions given a locator
/// when first asserted. Always sorted by `attribution` ascending — a
/// locator is only ever appended alongside a brand-new `AttributionRecord`
/// push, and attribution ids are monotonically increasing, so append
/// order is sort order for free; lookups are a binary search rather than
/// a dense parallel `Vec` with a sentinel.
///
/// Layout: 2 × u32 = 8 bytes, alignment 4, no padding.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct AttributionLocatorRecord {
    attribution: AttributionId,
    paragraph: u32,
}

/// Error returned by [`Context::from_bytes`] when an image is truncated,
/// wrongly versioned, or internally inconsistent. The message names the
/// first check that failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorruptImage(&'static str);

impl fmt::Display for CorruptImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "corrupt context image: {}", self.0)
    }
}

impl Error for CorruptImage {}

/// A weighted, labeled, directed graph of concepts — a many-to-many
/// associative network. Every concept, label, and source string is interned
/// exactly once and given a dense id; edges reference ids, and every
/// concept node anchors chains of the edges it participates in, so a
/// concept can fan out to (and be fanned into by) any number of others and
/// its neighborhood is directly walkable — which is what
/// [`Context::explore`] and [`Context::activate`] build on.
///
/// `Context` has no notion of natural language, sentences, or grammar — it
/// only knows subjects, relation labels, objects, signed weights, and
/// opaque source ids. Reducing an utterance in any language down to that
/// shape (deciding what the concepts are, what relates them, and how
/// strongly) is entirely the caller's job, upstream of this crate — e.g. an
/// LLM extracting structured facts from text before ever calling
/// `associate`. The network is a retrieval index over knowledge, not the
/// storage of record: attributions point back at the original text, which
/// lives wherever the caller keeps it.
///
/// One `Context` is one 文脈, and that is a contract: within a single
/// `Context`, one spelling is one referent. A spelling shared by two
/// different real-world things (the fruit "Apple" vs. the company "Apple")
/// must be stored in two different `Context`s, because nothing inside a
/// `Context` can tell two referents behind one spelling apart. Linking
/// across `Context`s does not exist (yet).
///
/// # Storage layout
///
/// Everything a `Context` knows lives in nine flat buffers: one UTF-8
/// string arena plus eight tables of fixed-width, naturally aligned,
/// pointer-free `#[repr(C)]` records (u32 ids and offsets, f64 weights) —
/// concepts, labels, sources, edges, attributions, the two alias tables,
/// and the sparse attribution-locator table. Variable-length structure is
/// expressed inside the fixed widths: strings live in the arena and
/// records hold (offset, len) pairs; adjacency lists are intrusive
/// singly-linked chains threaded through the edge records
/// (`next_outgoing` / `next_incoming` / `next_labeled`, terminated by a
/// `NIL` sentinel) and appended at the
/// tail, which is what preserves the insertion-order guarantees of the
/// read API. Every mutation is an append or an in-place field update —
/// records never move — so the whole state dumps and restores as one
/// contiguous image via [`Context::to_bytes`] / [`Context::from_bytes`].
///
/// The hash maps and normalized entry indexes are derived read-path
/// structures over those buffers (spelling → id with aliases folded in,
/// exact triple → edge, normalized forms and bigram postings for
/// `resolve`). They are not part of the persistent image; `from_bytes`
/// rebuilds them while validating it.
///
/// Capacity: ids and arena offsets are u32, so one `Context` holds at most
/// ~4.29 billion records per table and 4 GiB of interned text; a write
/// that would not fit returns [`ContextFull`] and changes nothing.
#[derive(Debug, Default)]
pub struct Context {
    /// Every interned string's UTF-8 bytes, back to back, in intern order.
    /// Records point into this with (offset, len) pairs; nothing is ever
    /// removed or moved.
    arena: Vec<u8>,
    concepts: Vec<ConceptRecord>,
    labels: Vec<LabelRecord>,
    sources: Vec<SourceRecord>,
    edges: Vec<EdgeRecord>,
    attributions: Vec<AttributionRecord>,
    /// Alternative concept spellings, resolving to canonical concepts at
    /// every entry point. Persisted.
    concept_aliases: Vec<AliasRecord>,
    /// Alternative label spellings. Persisted.
    label_aliases: Vec<AliasRecord>,
    /// Sparse paragraph locators, one per attribution that was given one
    /// when first asserted. Sorted by `attribution` ascending. Persisted.
    attribution_locators: Vec<AttributionLocatorRecord>,
    /// Derived index: interned name → concept id, aliases included. Not
    /// persisted.
    concept_ids: HashMap<String, ConceptId>,
    /// Derived index: interned name → label id. Not persisted.
    label_ids: HashMap<String, LabelId>,
    /// Derived index: interned name → source id. Not persisted.
    source_ids: HashMap<String, SourceId>,
    /// Derived, name-sorted index over the concept-alias namespace only
    /// (a subset of `concept_ids`, which also carries canonical names) —
    /// lets `concept_alias_page` seek a page with `.range()` instead of
    /// collecting and sorting every alias on each call. Not persisted.
    concept_alias_index: BTreeMap<String, ConceptId>,
    /// [`Context::concept_alias_index`] for the label-alias namespace.
    /// Not persisted.
    label_alias_index: BTreeMap<String, LabelId>,
    /// Derived, name-sorted set of every canonical label spelling — the
    /// same population `labels()` enumerates, kept sorted so `label_page`
    /// can seek instead of re-sorting the whole vocabulary on each call.
    /// No id resolution needed: a label's name IS its public identity.
    /// Not persisted.
    label_name_index: BTreeSet<String>,
    /// Derived exact-triple index so a repeated `associate` accumulates
    /// into the existing edge instead of growing a parallel one. Not
    /// persisted.
    edge_ids: HashMap<(ConceptId, LabelId, ConceptId), EdgeId>,
    /// Derived index: (edge, source) → that source's attribution record on
    /// that edge. One entry per LIVE (on-chain) attribution, so the write
    /// path finds — or rules out — an existing attribution in O(1) instead
    /// of walking the edge's whole chain; without it, asserting S sources
    /// onto one edge one at a time is O(S²). Retraction drops the entry as
    /// it unlinks the record, keeping the map from pointing at dead space.
    /// Not persisted — rebuilt from the chains on load.
    attribution_ids: HashMap<(EdgeId, SourceId), AttributionId>,
    /// Derived reverse index: source → the edges carrying a LIVE
    /// attribution from it. This is what lets `retract_source` cost the
    /// document's own footprint instead of a walk over every edge — the
    /// differential-sync workflow retracts a source on every document
    /// update, and that must not scale with the whole graph. Maintained
    /// beside `attribution_ids`: one entry pushed when a live
    /// attribution is appended, the whole key dropped on retraction.
    /// Not persisted — rebuilt from the chains on load.
    source_edges: HashMap<SourceId, Vec<EdgeId>>,
    /// Derived entry index over concept spellings — normalized forms and
    /// a bigram posting index behind `resolve`. Not persisted.
    concept_index: EntryIndex,
    /// Derived entry index over label spellings, behind `resolve_label`.
    /// Not persisted.
    label_index: EntryIndex,
    /// Tuning override for the fuzzy-entry floor; `None` means
    /// [`DICE_FLOOR`]. This is config, not knowledge, so it is NOT part
    /// of the persistent image — whoever loads an image re-applies it
    /// (the server keeps it in the context's sidecar).
    dice_floor: Option<f64>,
    /// An opaque durability watermark, persisted in the image header
    /// and meaningful only to the caller that set it: the sequence
    /// number, in the caller's own write-ahead log, of the last
    /// operation this image already reflects. It lives IN the image —
    /// not in a sidecar — precisely so image bytes and watermark are
    /// indivisible: written, fsynced, and renamed as one file, a crash
    /// can never observe one updated without the other. Zero means
    /// "nothing logged is reflected" (also what v1/v2 images load as).
    applied_seq: u64,
    /// Live count of edges whose `count` has fallen to zero — dead weight
    /// [`Context::compacted`] would shed. Tracked incrementally (unlike
    /// [`CompactionStats::dead_edges`], which reports a one-time count
    /// from an actual compaction run) so it reflects the current instant,
    /// including revivals: an edge that dies and is reasserted decrements
    /// this back down. Not persisted — seeded from the chains on load.
    dead_edges: usize,
    /// Live count of attribution records unlinked from every chain but
    /// not yet reclaimed by compaction. Unlike `dead_edges` this only
    /// grows between compactions: retraction never reuses an unlinked
    /// attribution record. Not persisted — seeded from the chains on
    /// load.
    dead_attributions: usize,
    /// Lower-bound count of arena bytes occupied by spellings of removed
    /// aliases — bytes `compacted()` would not carry forward. A lower
    /// bound because the names of concepts/labels/sources that lose
    /// their last live association also become dead weight, but arena
    /// bytes are never attributed back to a specific record once
    /// interned, so only alias removal (which frees a whole record) can
    /// be tracked here. Only grows between compactions: arena bytes are
    /// never reused. Not persisted — seeded from the residual on load.
    arena_slack: usize,
}

impl Context {
    fn intern_concept(&mut self, name: String) -> ConceptId {
        if let Some(&id) = self.concept_ids.get(&name) {
            return id;
        }
        let id = claim_id(self.concepts.len(), "concept");
        let (name_offset, name_len) = intern_name(&mut self.arena, &name);
        self.concepts.push(ConceptRecord {
            name_offset,
            name_len,
            first_outgoing: NIL,
            last_outgoing: NIL,
            outgoing_count: 0,
            first_incoming: NIL,
            last_incoming: NIL,
            incoming_count: 0,
        });
        self.concept_index.push(&name, id);
        self.concept_ids.insert(name, id);
        id
    }

    fn intern_label(&mut self, name: String) -> LabelId {
        if let Some(&id) = self.label_ids.get(&name) {
            return id;
        }
        let id = claim_id(self.labels.len(), "label");
        let (name_offset, name_len) = intern_name(&mut self.arena, &name);
        self.labels.push(LabelRecord {
            name_offset,
            name_len,
            first_edge: NIL,
            last_edge: NIL,
            edge_count: 0,
        });
        self.label_index.push(&name, id);
        self.label_name_index.insert(name.clone());
        self.label_ids.insert(name, id);
        id
    }

    fn intern_source(&mut self, name: String) -> SourceId {
        if let Some(&id) = self.source_ids.get(&name) {
            return id;
        }
        let id = claim_id(self.sources.len(), "source");
        let (name_offset, name_len) = intern_name(&mut self.arena, &name);
        self.sources.push(SourceRecord {
            name_offset,
            name_len,
        });
        self.source_ids.insert(name, id);
        id
    }

    /// Reads one interned string back out of the arena. Ranges are
    /// validated when interned and when an image is loaded, so this cannot
    /// fail on a `Context` that exists.
    fn arena_str(&self, offset: u32, len: u32) -> &str {
        let start = offset as usize;
        let end = start + len as usize;
        std::str::from_utf8(&self.arena[start..end])
            .expect("arena ranges are validated on intern and on load")
    }

    fn concept_name(&self, id: ConceptId) -> &str {
        let record = &self.concepts[id as usize];
        self.arena_str(record.name_offset, record.name_len)
    }

    fn label_name(&self, id: LabelId) -> &str {
        let record = &self.labels[id as usize];
        self.arena_str(record.name_offset, record.name_len)
    }

    fn source_name(&self, id: SourceId) -> &str {
        let record = &self.sources[id as usize];
        self.arena_str(record.name_offset, record.name_len)
    }

    /// Walks one linked chain of edges from `first`, yielding edge ids in
    /// chain (= insertion) order until the `NIL` terminator.
    fn edge_chain(
        &self,
        first: EdgeId,
        follow: fn(&EdgeRecord) -> EdgeId,
    ) -> impl Iterator<Item = EdgeId> {
        std::iter::successors((first != NIL).then_some(first), move |&edge_id| {
            let next = follow(&self.edges[edge_id as usize]);
            (next != NIL).then_some(next)
        })
    }

    /// Edges where `concept` is the subject, in insertion order.
    fn outgoing(&self, concept: ConceptId) -> impl Iterator<Item = EdgeId> {
        self.edge_chain(self.concepts[concept as usize].first_outgoing, |edge| {
            edge.next_outgoing
        })
    }

    /// Edges where `concept` is the object, in insertion order.
    fn incoming(&self, concept: ConceptId) -> impl Iterator<Item = EdgeId> {
        self.edge_chain(self.concepts[concept as usize].first_incoming, |edge| {
            edge.next_incoming
        })
    }

    /// Edges carrying `label`, in insertion order.
    fn labeled(&self, label: LabelId) -> impl Iterator<Item = EdgeId> {
        self.edge_chain(self.labels[label as usize].first_edge, |edge| {
            edge.next_labeled
        })
    }

    /// Walks one edge's attribution chain in first-assertion order,
    /// paired with each record's id so callers can look up its locator.
    fn attribution_chain(
        &self,
        first: AttributionId,
    ) -> impl Iterator<Item = (AttributionId, &AttributionRecord)> {
        std::iter::successors(
            (first != NIL).then(|| (first, &self.attributions[first as usize])),
            |&(_, record)| {
                (record.next != NIL)
                    .then(|| (record.next, &self.attributions[record.next as usize]))
            },
        )
    }

    /// Looks up the paragraph locator recorded for one attribution, if
    /// any. `attribution_locators` is sorted by `attribution` ascending —
    /// a locator is only ever appended alongside a brand-new
    /// `AttributionRecord`, and attribution ids are monotonically
    /// increasing — so a binary search finds it in O(log n).
    fn locator_for(&self, attribution: AttributionId) -> Option<u32> {
        self.attribution_locators
            .binary_search_by_key(&attribution, |record| record.attribution)
            .ok()
            .map(|index| self.attribution_locators[index].paragraph)
    }

    /// Materializes one edge back into owned strings for the caller.
    fn association(&self, edge_id: EdgeId) -> Association {
        let edge = &self.edges[edge_id as usize];
        Association {
            subject: self.concept_name(edge.subject).to_string(),
            label: self.label_name(edge.label).to_string(),
            object: self.concept_name(edge.object).to_string(),
            weight: if edge.count == 0 {
                0.0
            } else {
                edge.sum / edge.count as f64
            },
            count: edge.count,
            attributions: self
                .attribution_chain(edge.first_attribution)
                .map(|(attribution_id, record)| Attribution {
                    source: self.source_name(record.source).to_string(),
                    weight: record.sum,
                    count: record.count,
                    paragraph: self.locator_for(attribution_id),
                })
                .collect(),
        }
    }
}

/// Copies `name`'s UTF-8 bytes onto the end of the arena and returns the
/// (offset, len) pair that records store in place of the string. Capacity
/// is pre-checked by `Context::ensure_room`, so the assert here is an
/// invariant backstop, not the public failure path.
fn intern_name(arena: &mut Vec<u8>, name: &str) -> (u32, u32) {
    let offset = arena.len();
    let end = offset
        .checked_add(name.len())
        .expect("string arena size overflows usize");
    assert!(
        end <= u32::MAX as usize,
        "string arena exceeds its 4 GiB offset space"
    );
    arena.extend_from_slice(name.as_bytes());
    (offset as u32, name.len() as u32)
}

/// Appends `edge_id` at the tail of one chain, updating the chain's
/// anchor fields (head, tail, count) and the previous tail's next-link —
/// `follow` picks which next-link field of an edge record that chain
/// threads through. Appending at the tail is what keeps every chain in
/// insertion order.
fn append_to_chain(
    edges: &mut [EdgeRecord],
    first: &mut EdgeId,
    last: &mut EdgeId,
    count: &mut u32,
    edge_id: EdgeId,
    follow: fn(&mut EdgeRecord) -> &mut EdgeId,
) {
    let tail = std::mem::replace(last, edge_id);
    *count += 1;
    if tail == NIL {
        *first = edge_id;
    } else {
        *follow(&mut edges[tail as usize]) = edge_id;
    }
}

/// `f64::clamp` returns NaN when `self` is NaN — it clamps the RANGE,
/// not the value — so every "clamped into [0, 1]" call site below fed
/// a bare `.clamp(0.0, 1.0)` a NaN straight through. A NaN decay then
/// makes every downstream `<= 0.0` / `< floor` comparison false rather
/// than true, flipping fail-closed filters open (or, for `activate`'s
/// score gate, leaving a phantom zero-strength entry where "no signal"
/// should have skipped it outright). `nan_fallback` lets each call site
/// pick the safe side: 0.0 where the floor is a shrink-to-nothing decay
/// (no propagation beats fabricated propagation), 1.0 where it is an
/// admission floor (excluding everything beats flooding with noise).
fn clamp_unit_or(value: f64, nan_fallback: f64) -> f64 {
    if value.is_nan() {
        nan_fallback
    } else {
        value.clamp(0.0, 1.0)
    }
}

/// The one normalization both sides of every entry comparison go
/// through: lowercasing, Unicode NFKC (folding full-width romaji,
/// half-width kana, and compatibility forms), and katakana → hiragana,
/// so "Ａｐｐｌｅ" meets "apple" and "リンゴ" meets "りんご". Applying the
/// same function to stored spellings and cues is what makes the folds
/// safe — neither side is ever compared raw against a folded form.
///
/// Lowercasing runs both before AND after nfkc(), because either order
/// alone has a counterexample that leaves this function not idempotent:
/// - NFKC-then-lowercase: "J" + COMBINING CARON (U+030C) has no
///   precomposed NFKC form, so nfkc() leaves it as two code points;
///   lowercasing afterward yields "j" + U+030C, still two code points
///   — not a fixed point, since folding THAT again composes straight
///   to the single precomposed U+01F0.
/// - Lowercase-then-NFKC only: "🄐" (U+1F110, PARENTHESIZED LATIN
///   CAPITAL LETTER A) has no lowercase mapping of its own, so
///   lowercasing first is a no-op; its NFKC compatibility decomposition
///   is the literal, uppercase "(A)" — not a fixed point, since folding
///   THAT again lowercases to "(a)".
///
/// A trailing lowercase after nfkc() closes both gaps: it re-folds
/// whatever casing nfkc()'s decomposition introduced, so the result is
/// always a fixed point of this same function.
fn normalize(name: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    name.chars()
        .flat_map(char::to_lowercase)
        .nfkc()
        .flat_map(char::to_lowercase)
        .map(fold_kana)
        .collect()
}

/// [`normalize`] for companion text layers: a passage search over the
/// same corpus must fold text exactly the way the entry index folds
/// names, or the two disagree about what matches.
pub fn normalize_entry(text: &str) -> String {
    normalize(text)
}

/// Katakana → hiragana (U+30A1..=U+30F6 sit 0x60 above U+3041..=U+3096).
fn fold_kana(ch: char) -> char {
    match ch {
        '\u{30A1}'..='\u{30F6}' => char::from_u32(ch as u32 - 0x60).unwrap_or(ch),
        _ => ch,
    }
}

/// Shared ordering of [`Context::resolve`] / [`Context::resolve_label`]
/// output: best first, ties broken alphabetically.
fn sort_resolutions(resolutions: &mut [Resolution]) {
    resolutions.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.name.cmp(&b.name))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dead_ratio_of_treats_no_edges_as_zero_not_nan() {
        assert_eq!(dead_ratio_of(0, 0), 0.0);
        assert_eq!(dead_ratio_of(1, 4), 0.25);
        assert_eq!(dead_ratio_of(4, 4), 1.0);
    }

    // A real overflow would need ~4.29 billion records, so the boundary
    // logic behind `ensure_room` is tested directly on the pure helpers.
    #[test]
    fn ids_left_stops_exactly_at_the_nil_sentinel() {
        // Ids are dense from 0 and NIL itself is never minted, so a table
        // holds at most NIL records.
        assert!(ids_left(NIL as usize - 1, 1));
        assert!(!ids_left(NIL as usize, 1));
        assert!(ids_left(NIL as usize, 0)); // accumulate-only writes still fit
        assert!(!ids_left(NIL as usize - 1, 2));
    }

    #[test]
    fn claim_id_never_mints_the_reserved_nil_sentinel() {
        assert_eq!(claim_id(NIL as usize - 1, "concept"), NIL - 1);
        assert!(
            std::panic::catch_unwind(|| claim_id(NIL as usize, "concept")).is_err(),
            "NIL is reserved for the linked-list terminator"
        );
    }

    #[test]
    fn public_errors_keep_their_diagnostic_messages() {
        let full = ContextFull("concept ids");
        assert_eq!(full.to_string(), "context is full: concept ids");
        assert_eq!(
            CompactionError::Full(full.clone()).to_string(),
            "context is full: concept ids"
        );
        assert_eq!(
            CompactionError::DeadlineExceeded.to_string(),
            "compaction exceeded its deadline"
        );
        assert_eq!(
            AliasError::UnknownCanonical.to_string(),
            "the canonical spelling is not interned"
        );
        assert_eq!(
            AliasError::Conflict.to_string(),
            "the alias already resolves to a different record"
        );
        assert_eq!(
            AliasError::Full(full).to_string(),
            "context is full: concept ids"
        );
        assert_eq!(
            CorruptImage("bad header").to_string(),
            "corrupt context image: bad header"
        );
    }

    #[test]
    fn arena_fits_stops_exactly_at_the_offset_ceiling() {
        assert!(arena_fits(u32::MAX as usize - 5, 5));
        assert!(!arena_fits(u32::MAX as usize - 5, 6));
        assert!(arena_fits(u32::MAX as usize, 0)); // full arena, nothing new needed
        assert!(!arena_fits(usize::MAX, 1)); // must refuse, not overflow
    }

    #[test]
    fn a_retracted_fact_is_excluded_from_gloss_and_describe() {
        let mut context = Context::default();
        context.associate("A", "関係", "B", 1.0).unwrap();
        context.associate("A", "別関係", "C", 1.0).unwrap();
        // Fully retract the first fact: its count falls to 0 but the edge
        // stays linked in A's outgoing chain (retract unlinks only the
        // attribution records, not the edge itself).
        context.retract_association("A", "関係", "B");

        // Even asking for more facts than remain live, the withdrawn one
        // (|sum| 0, so it would otherwise sort in at the tail) must not
        // surface as current knowledge.
        assert_eq!(context.concept_gloss("A", 5).unwrap(), "A。別関係はC。");

        // describe tallies live facts only — the dead label is gone.
        let description = context.describe("A").unwrap();
        assert_eq!(
            description.as_subject,
            vec![LabelUsage {
                label: "別関係".to_string(),
                count: 1,
            }]
        );
        assert!(description.as_object.is_empty());
    }
}
