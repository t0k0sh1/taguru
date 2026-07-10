use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

mod image;

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

/// One source's accumulated contribution to an association's weight.
///
/// `source` is an opaque identifier chosen by the caller — a document id, a
/// URL, a chunk reference — that lets whoever retrieved the association go
/// fetch the original text behind it. The `Context` never interprets it.
/// `paragraph` optionally locates the fact within `source` (e.g. the
/// paragraph index it was read from); it is `null` rather than omitted
/// when absent, so callers can rely on the key always being present —
/// the same contract the citation endpoint's `section` field makes. It
/// reflects only the first assertion of this source (first-write-wins);
/// later re-assertions accumulate into `weight` but never change it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Attribution {
    pub source: String,
    pub weight: f64,
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
    pub weight: f64,
    /// Which sources asserted this association and how much weight each
    /// contributed. Total `weight` can exceed the attributed sum when some
    /// assertions came in unsourced. Two attributions of 1.0 (independent
    /// corroboration) and one attribution of 2.0 (a single emphatic
    /// assertion) are distinguishable here even though `weight` is 2.0 in
    /// both cases.
    pub attributions: Vec<Attribution>,
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
/// Layout: 8 × u32 + 1 × f64 = 40 bytes, alignment 8, no padding (the
/// eight u32 fields put `weight` at offset 32).
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
    weight: f64,
}

/// One source's weight contribution to one edge, in fixed-width form; a
/// link in that edge's attribution chain, in insertion order, one record
/// per distinct source.
///
/// Layout: 2 × u32 + 1 × f64 = 16 bytes, alignment 8, no padding.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct AttributionRecord {
    source: SourceId,
    next: AttributionId,
    weight: f64,
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
    /// Derived exact-triple index so a repeated `associate` accumulates
    /// into the existing edge instead of growing a parallel one. Not
    /// persisted.
    edge_ids: HashMap<(ConceptId, LabelId, ConceptId), EdgeId>,
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
}

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

    /// Adds a directed, labeled, signed edge from `subject` to `object`, or
    /// accumulates `weight` into an edge already there for the same
    /// (subject, label, object) key.
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
    /// seeds it directly. Every later call for the same key adds `weight`
    /// onto whatever is already there: repeated agreeing evidence
    /// accumulates into a stronger weight of the same sign, and
    /// contradicting evidence nets against it — strong enough contradiction
    /// can overturn the sign outright. Nothing about the mechanism treats
    /// agreement and contradiction differently; both are just addition.
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
    /// fact. The same accumulation semantics apply to the edge's total
    /// weight; in addition, the contribution is tallied per source, so a
    /// retrieved [`Association`] can show whether its weight came from many
    /// independent sources (corroboration) or one source repeating itself,
    /// and the caller can follow any attribution back to the original text.
    ///
    /// Discipline note: paraphrases of one fact inside one document should
    /// NOT be re-asserted at full weight — that inflates `weight` without
    /// adding independent evidence. Assert once per document; re-assert
    /// across documents (real corroboration). Readers that care about
    /// independence should count distinct attributions rather than trust
    /// raw weight.
    ///
    /// `paragraph` optionally locates the fact within `source` (e.g. a
    /// paragraph index) and is first-write-wins: it is only recorded the
    /// first time this source is attributed to this edge, so a later
    /// re-assertion of the same source cannot change where the first one
    /// pointed.
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
                self.edges[edge_id as usize].weight += weight;
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
                    weight,
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
        // already attributes this exact edge.
        if let Some(name) = source {
            let already_attributed = existing_edge.is_some_and(|edge_id| {
                self.source_ids.get(name).is_some_and(|&source_id| {
                    self.attribution_chain(self.edges[edge_id as usize].first_attribution)
                        .any(|(_, record)| record.source == source_id)
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
        let mut cursor = self.edges[edge_id as usize].first_attribution;
        while cursor != NIL {
            let record = &mut self.attributions[cursor as usize];
            if record.source == source_id {
                record.weight += weight;
                return;
            }
            cursor = record.next;
        }

        let attribution_id = claim_id(self.attributions.len(), "attribution");
        self.attributions.push(AttributionRecord {
            source: source_id,
            next: NIL,
            weight,
        });
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
        type Follow = fn(&EdgeRecord) -> EdgeId;
        let mut narrowest: Option<(u64, Vec<EdgeId>, Follow)> = None;
        let mut consider = |total: u64, firsts: Vec<EdgeId>, follow: Follow| {
            if narrowest.as_ref().is_none_or(|&(best, ..)| total < best) {
                narrowest = Some((total, firsts, follow));
            }
        };
        if let Some(ids) = &subject_ids {
            consider(
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
            consider(
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
            consider(
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
                *counts
                    .entry(self.edges[edge_id as usize].label)
                    .or_insert(0) += 1;
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
                reached.entry(edge_id).or_insert((hop, concept_id));
                let edge = &self.edges[edge_id as usize];
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
    /// the `limit` most strongly activated associations, best first. This
    /// is the ranked counterpart of [`Context::explore`] — the retrieval
    /// that reads the weights `associate` has been writing.
    ///
    /// Each origin starts with activation 1.0, and a concept's activation
    /// is that of its single strongest incoming path. Two formulas share
    /// that activation but do different jobs:
    ///
    /// - An association's returned strength is
    ///   `activation(nearer endpoint) * decay * |weight|` — deliberately
    ///   independent of how many OTHER associations its endpoints carry.
    ///   A fact about an anchor must not sink because the anchor is well
    ///   documented, and facts of several origins must compete fairly
    ///   however lopsided the origins' degrees are.
    /// - What flows ON to a neighboring concept is fan-normalized:
    ///   `activation * decay * |weight| / Σ|weight|` over the concept's
    ///   associations. A promiscuous hub splits its activation thinly, so
    ///   everything BEYOND a hub fades — connection through a busy concept
    ///   is weak evidence of relatedness. At equal depth and weight, an
    ///   association just past a hub therefore ties with one past a
    ///   dedicated relay; the hub's penalty lands on all knowledge beyond.
    ///
    /// Magnitude ranks, sign is content: weight -2.0 is strong knowledge
    /// ("firmly not the case") and outranks +1.0, while associations
    /// netted out to 0.0 are not returned at all — a fully contested fact
    /// carries no reliable knowledge. `decay` shrinks the signal each hop
    /// (clamped into [0, 1]). Strengths are ordinal: compare them within
    /// one call's results, not across calls or corpus versions. Callers
    /// that care about independent corroboration rather than accumulated
    /// weight can re-rank the returned associations by their number of
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
    pub fn activate(&self, origins: &[&str], decay: f64, limit: usize) -> Vec<Activation> {
        let decay = decay.clamp(0.0, 1.0);

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
                .chain(self.incoming(concept))
                .map(|edge_id| self.edges[edge_id as usize].weight.abs())
                .sum();
            if total == 0.0 {
                continue;
            }
            for edge_id in self.outgoing(concept).chain(self.incoming(concept)) {
                let edge = &self.edges[edge_id as usize];
                // Ranking: fan-independent, so a busy endpoint doesn't
                // sink its own facts. Netted-out (or zero-decay) signals
                // carry nothing and are skipped entirely.
                let score = activation * decay * edge.weight.abs();
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
        ranked.truncate(limit);

        ranked
            .into_iter()
            .map(|(strength, edge_id, from)| Activation {
                strength,
                path: self.path_from(&parents, from),
                association: self.association(edge_id),
            })
            .collect()
    }

    /// Maps free-form wording onto stored concept names, returning scored
    /// candidates, best first. This is the entry point for retrieval:
    /// `explore` and `activate` need exact concept names as origins, and a
    /// querying LLM rarely repeats a stored spelling exactly ("選考会" for
    /// "10大脅威選考会").
    ///
    /// The same check belongs on the write path: before minting a new
    /// concept spelling, an ingester should ask what similar spellings
    /// already exist and reuse one — check before mint.
    ///
    /// This tier is lexical, in two layers sharing one score scale: names
    /// are compared in normalized form (Unicode NFKC, case-folded,
    /// katakana folded to hiragana — so full-width romaji and kana
    /// variants land), containment either way scores by how much of the
    /// longer string the shorter one covers (exact match = 1.0), and
    /// near-miss spellings that containment cannot catch (青嶺酒蔵 for
    /// 青嶺酒造) match by character-bigram overlap through a posting
    /// index. It deliberately does NOT guess at semantic similarity —
    /// that belongs to the tiers above it: an LLM normalizing its query
    /// wording, or, at corpus scale, an embedding index over concept
    /// names (only the *entry* needs fuzziness; the knowledge retrieval
    /// itself stays structural). The scan is O(number of concepts) and
    /// allocates nothing per candidate.
    pub fn resolve(&self, cue: &str) -> Vec<Resolution> {
        self.resolve_with_floor(cue, self.dice_floor())
    }

    /// [`Context::resolve`] with an explicit fuzzy floor for this one
    /// call, overriding the context's setting — the loosen-and-retry
    /// move after a miss, or a tightening when a cue is known-exact.
    /// Only the fuzzy tier is affected; exact and containment matches
    /// are never floored.
    pub fn resolve_with_floor(&self, cue: &str, dice_floor: f64) -> Vec<Resolution> {
        self.scored_resolutions(&self.concept_index, Self::concept_name, cue, dice_floor)
    }

    /// [`Context::resolve`] for relation labels instead of concepts — the
    /// two namespaces never mix. This exists chiefly for the write path:
    /// label vocabulary forks silently ("創業年" vs "創業" vs "設立年"),
    /// and once forked, label-pinned queries stop seeing half the facts.
    /// An ingester should call this (or review [`Context::labels`]) before
    /// coining a relation spelling, and reuse a close existing label
    /// instead.
    pub fn resolve_label(&self, cue: &str) -> Vec<Resolution> {
        self.resolve_label_with_floor(cue, self.dice_floor())
    }

    /// [`Context::resolve_label`] with an explicit fuzzy floor for this
    /// one call, as [`Context::resolve_with_floor`].
    pub fn resolve_label_with_floor(&self, cue: &str, dice_floor: f64) -> Vec<Resolution> {
        self.scored_resolutions(&self.label_index, Self::label_name, cue, dice_floor)
    }

    /// The shared scoring behind both resolve entry points: score the
    /// cue against one namespace's entry index, materialize the winning
    /// records as named resolutions, best first. An exact hit whose
    /// record answers to a different canonical spelling was an alias
    /// match — refined here, where the names are known.
    fn scored_resolutions(
        &self,
        index: &EntryIndex,
        name_of: fn(&Self, u32) -> &str,
        cue: &str,
        dice_floor: f64,
    ) -> Vec<Resolution> {
        let needle = normalize(cue);
        let mut resolutions: Vec<Resolution> = index
            .resolve(cue, dice_floor.clamp(0.0, 1.0))
            .into_iter()
            .map(|(id, (score, kind))| {
                let name = name_of(self, id).to_string();
                let kind = if kind == MatchKind::Exact && normalize(&name) != needle {
                    MatchKind::Alias
                } else {
                    kind
                };
                Resolution { name, score, kind }
            })
            .collect();
        sort_resolutions(&mut resolutions);
        resolutions
    }

    /// The durability watermark stored in this context's image header
    /// — see the field's documentation; the `Context` never interprets
    /// it.
    pub fn applied_seq(&self) -> u64 {
        self.applied_seq
    }

    /// Sets the durability watermark the next [`Context::to_bytes`]
    /// will persist. The caller (the server's WAL) owns the meaning.
    pub fn set_applied_seq(&mut self, seq: u64) {
        self.applied_seq = seq;
    }

    /// The fuzzy-entry floor this context applies when a call does not
    /// name one: bigram-Dice matches below it are dropped as noise.
    pub fn dice_floor(&self) -> f64 {
        self.dice_floor.unwrap_or(DICE_FLOOR)
    }

    /// Tunes the fuzzy-entry floor for this context — lower admits more
    /// distant near-miss spellings (vocabularies with heavy 表記ゆれ),
    /// higher keeps entry strict (curated glossaries). `None` returns to
    /// the default. Clamped into [0, 1]. Config, not knowledge: the
    /// value is not part of the persistent image, so whoever restores a
    /// context from bytes must re-apply it.
    pub fn set_dice_floor(&mut self, dice_floor: Option<f64>) {
        self.dice_floor = dice_floor.map(|floor| floor.clamp(0.0, 1.0));
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

    /// Every canonical concept spelling in insertion order — the
    /// vocabulary an external entry tier (e.g. an embedding index over
    /// names) enumerates to stay in sync with the network.
    pub fn concept_names(&self) -> Vec<&str> {
        self.concepts
            .iter()
            .map(|record| self.arena_str(record.name_offset, record.name_len))
            .collect()
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
        self.outgoing(id_a)
            .chain(self.incoming(id_a))
            .any(|edge_id| {
                let edge = &self.edges[edge_id as usize];
                edge.subject == id_b || edge.object == id_b
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
        let subjects: HashSet<ConceptId> = self
            .labeled(id_a)
            .map(|edge_id| self.edges[edge_id as usize].subject)
            .collect();
        self.labeled(id_b)
            .any(|edge_id| subjects.contains(&self.edges[edge_id as usize].subject))
    }

    /// Concept pairs whose spellings look like accidental forks of one
    /// referent — the lexical half of a vocabulary audit. Spelling drift
    /// fails silently in this system (two spellings = two referents =
    /// queries see half the facts), so this surfaces (name_a, name_b,
    /// dice) candidates, strongest first, for review. Candidates, not
    /// verdicts: containment pairs are often legitimately distinct
    /// (青嶺 the brand vs 青嶺酒造 the brewery). Aliases pointing at one
    /// record are intentional and never reported.
    pub fn similar_concepts(&self, dice_floor: f64) -> Vec<(String, String, f64)> {
        self.scored_twins(&self.concept_index, Self::concept_name, dice_floor)
    }

    /// [`Context::similar_concepts`] for relation labels — where forks
    /// hurt most, since label-pinned queries silently miss the twin.
    pub fn similar_labels(&self, dice_floor: f64) -> Vec<(String, String, f64)> {
        self.scored_twins(&self.label_index, Self::label_name, dice_floor)
    }

    /// The shared sweep behind both twin detectors: run one namespace's
    /// entry index name-against-name and materialize the flagged pairs.
    fn scored_twins(
        &self,
        index: &EntryIndex,
        name_of: fn(&Self, u32) -> &str,
        dice_floor: f64,
    ) -> Vec<(String, String, f64)> {
        index
            .twins(dice_floor.clamp(0.0, 1.0))
            .into_iter()
            .map(|(a, b, dice)| {
                (
                    name_of(self, a).to_string(),
                    name_of(self, b).to_string(),
                    dice,
                )
            })
            .collect()
    }

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
    /// first (|weight| descending, ties in insertion order), negatives
    /// phrased as denials. This is what an external embedding tier
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
        let edges = self.heaviest(self.outgoing(id).chain(self.incoming(id)), facts);
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

    /// The `keep` heaviest edges of a chain walk, |weight| descending,
    /// ties toward insertion order.
    fn heaviest(&self, edges: impl Iterator<Item = EdgeId>, keep: usize) -> Vec<EdgeId> {
        let mut edges: Vec<EdgeId> = edges.collect();
        edges.sort_by(|&a, &b| {
            self.edges[b as usize]
                .weight
                .abs()
                .total_cmp(&self.edges[a as usize].weight.abs())
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
            if edge.weight < 0.0 {
                gloss.push_str("ではない");
            }
            gloss.push('。');
        }
        gloss
    }

    /// How many distinct (subject, label, object) associations are stored.
    pub fn association_count(&self) -> usize {
        self.edges.len()
    }

    /// How many distinct concepts are interned.
    pub fn concept_count(&self) -> usize {
        self.concepts.len()
    }

    /// How many distinct relation labels are interned.
    pub fn label_count(&self) -> usize {
        self.labels.len()
    }

    /// How many distinct sources have been named by `associate_from`.
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// The most connected concepts — name plus total degree, most
    /// connected first, ties toward earlier interning. This is the
    /// mechanical "what is this context about" signal: unlike a
    /// hand-written summary it cannot go stale, so a routing directory
    /// can show it next to the prose description.
    pub fn top_concepts(&self, limit: usize) -> Vec<(&str, usize)> {
        let mut ranked: Vec<(usize, ConceptId)> = self
            .concepts
            .iter()
            .enumerate()
            .map(|(id, record)| {
                let degree = (record.outgoing_count + record.incoming_count) as usize;
                (degree, id as u32)
            })
            .collect();
        ranked.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        ranked.truncate(limit);
        ranked
            .into_iter()
            .map(|(degree, id)| (self.concept_name(id), degree))
            .collect()
    }

    /// Rough resident-memory estimate: exact bytes for the arena and the
    /// five record tables, plus estimates for the derived indexes (the
    /// name → id maps own a second copy of every interned name, the
    /// triple index costs its key and value, and each hash-map entry
    /// carries bookkeeping overhead; the lowercase shadows mirror the
    /// arena again). Intended for cache budgeting — deciding what stays
    /// in memory — not accounting-grade measurement.
    pub fn footprint(&self) -> usize {
        let tables = self.arena.len()
            + self.concepts.len() * size_of::<ConceptRecord>()
            + self.labels.len() * size_of::<LabelRecord>()
            + self.sources.len() * size_of::<SourceRecord>()
            + self.edges.len() * size_of::<EdgeRecord>()
            + self.attributions.len() * size_of::<AttributionRecord>()
            + (self.concept_aliases.len() + self.label_aliases.len()) * size_of::<AliasRecord>()
            + self.attribution_locators.len() * size_of::<AttributionLocatorRecord>();

        const MAP_ENTRY_OVERHEAD: usize = 48;
        let name_entries = self.concepts.len() + self.labels.len() + self.sources.len();
        let triple_entry = size_of::<(ConceptId, LabelId, ConceptId)>() + size_of::<EdgeId>();
        let derived = self.arena.len() // owned keys of the name → id maps
            + name_entries * MAP_ENTRY_OVERHEAD
            + self.edges.len() * (triple_entry + MAP_ENTRY_OVERHEAD)
            + self.concept_index.footprint()
            + self.label_index.footprint();

        tables + derived
    }

    /// Lists every association that no walk from `origins` can ever reach
    /// — the post-ingest coverage audit. Unreachable knowledge fails
    /// silently: nothing errors, retrieval simply never returns it. Run
    /// this after ingesting a document, anchored at the document's main
    /// entities; a non-empty result means the decomposition left facts
    /// disconnected (usually an implicit membership that never became an
    /// edge) and names exactly which ones. Reachability is bidirectional
    /// and does not travel through labels, exactly as in
    /// [`Context::explore`].
    pub fn unreachable_from(&self, origins: &[&str]) -> Vec<Association> {
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
            for edge_id in self.outgoing(concept_id).chain(self.incoming(concept_id)) {
                let edge = &self.edges[edge_id as usize];
                for neighbor in [edge.subject, edge.object] {
                    if visited.insert(neighbor) {
                        frontier.push_back(neighbor);
                    }
                }
            }
        }

        // An edge's endpoints reach each other through it, so checking one
        // endpoint decides the whole edge.
        (0..self.edges.len() as u32)
            .filter(|&edge_id| !visited.contains(&self.edges[edge_id as usize].subject))
            .map(|edge_id| self.association(edge_id))
            .collect()
    }

    /// Registers an alternative spelling for an existing concept. Aliases
    /// are entry-only: every lookup — `query`, `recall`, `describe`,
    /// walk origins, `resolve` candidates, and interning on the write
    /// path — resolves the alias to its canonical concept, but results
    /// always carry the canonical spelling and the graph never grows an
    /// alias node. Registering an alias is therefore the post-hoc repair
    /// for "the knowledge exists but this wording misses it": queries
    /// with the new spelling start landing, and future ingests using it
    /// accumulate into the canonical concept instead of forking a new
    /// one.
    ///
    /// `canonical` may itself be an alias — the new alias resolves to
    /// the true canonical record. Re-registering an existing alias of
    /// the same record is a no-op `Ok`, so alias imports are idempotent.
    ///
    /// # Errors
    ///
    /// [`AliasError::UnknownCanonical`] when `canonical` is not
    /// interned; [`AliasError::Conflict`] when `alias` already resolves
    /// to a different record (aliasing two existing concepts together
    /// would be a merge, which does not exist — rebuild instead);
    /// [`AliasError::Full`] when the alias table or arena is out of
    /// space. The context is unchanged on every error.
    pub fn add_concept_alias(
        &mut self,
        alias: impl Into<String>,
        canonical: &str,
    ) -> Result<(), AliasError> {
        add_alias(
            &mut self.arena,
            &mut self.concept_aliases,
            &mut self.concept_index,
            &mut self.concept_ids,
            alias.into(),
            canonical,
            "the concept alias table is out of u32 ids",
        )
    }

    /// [`Context::add_concept_alias`] for relation labels — the label
    /// vocabulary is where spellings fork most often ("創業年" vs
    /// "設立年"), and a label alias heals exactly that: label-pinned
    /// queries and future ingests using either spelling land on one
    /// relation.
    ///
    /// # Errors
    ///
    /// As [`Context::add_concept_alias`], within the label namespace.
    pub fn add_label_alias(
        &mut self,
        alias: impl Into<String>,
        canonical: &str,
    ) -> Result<(), AliasError> {
        add_alias(
            &mut self.arena,
            &mut self.label_aliases,
            &mut self.label_index,
            &mut self.label_ids,
            alias.into(),
            canonical,
            "the label alias table is out of u32 ids",
        )
    }

    /// Withdraws one alias spelling from the concept namespace — the
    /// undo for a mis-registered alias. The spelling stops resolving
    /// and becomes free to register again; the canonical record, its
    /// edges, and every other spelling stay untouched. Returns the
    /// canonical name the alias pointed at, or `None` when the exact
    /// spelling is not a concept alias — in particular a CANONICAL
    /// name is refused this way, because removal must never be able
    /// to unname a record. The spelling's arena bytes stay behind as
    /// slack (append-only storage; a few bytes per removal).
    pub fn remove_concept_alias(&mut self, alias: &str) -> Option<String> {
        let position = self
            .concept_aliases
            .iter()
            .position(|record| self.arena_str(record.name_offset, record.name_len) == alias)?;
        let record = self.concept_aliases.remove(position);
        self.concept_ids.remove(alias);
        self.rebuild_concept_index();
        Some(self.concept_name(record.target).to_string())
    }

    /// [`Context::remove_concept_alias`] for relation labels.
    pub fn remove_label_alias(&mut self, alias: &str) -> Option<String> {
        let position = self
            .label_aliases
            .iter()
            .position(|record| self.arena_str(record.name_offset, record.name_len) == alias)?;
        let record = self.label_aliases.remove(position);
        self.label_ids.remove(alias);
        self.rebuild_label_index();
        Some(self.label_name(record.target).to_string())
    }

    /// Rebuilds the concept entry index from the records. The index is
    /// append-only (arena + bigram postings), so removal is a rebuild
    /// by design: alias curation is rare, a rebuild costs milliseconds,
    /// and resolve keeps a structure with no dead entries to skip.
    fn rebuild_concept_index(&mut self) {
        let mut index = EntryIndex::default();
        for (id, record) in self.concepts.iter().enumerate() {
            index.push(
                self.arena_str(record.name_offset, record.name_len),
                id as u32,
            );
        }
        for record in &self.concept_aliases {
            index.push(
                self.arena_str(record.name_offset, record.name_len),
                record.target,
            );
        }
        self.concept_index = index;
    }

    /// [`Context::rebuild_concept_index`] for the label namespace.
    fn rebuild_label_index(&mut self) {
        let mut index = EntryIndex::default();
        for (id, record) in self.labels.iter().enumerate() {
            index.push(
                self.arena_str(record.name_offset, record.name_len),
                id as u32,
            );
        }
        for record in &self.label_aliases {
            index.push(
                self.arena_str(record.name_offset, record.name_len),
                record.target,
            );
        }
        self.label_index = index;
    }

    /// Withdraws one source's contributions: every attribution it made
    /// is removed from its edge's chain and its weight subtracted from
    /// the edge's total — the differential-sync move when a document
    /// changes (retract the old version, re-ingest the new one), instead
    /// of rebuilding the whole context.
    ///
    /// Returns how many associations were touched, or `None` for a
    /// source this context never saw. What retraction deliberately does
    /// NOT do: concepts, labels, and edges minted by the document stay
    /// (the storage is append-only; an edge whose weight nets to 0.0
    /// simply stops carrying knowledge — `activate` already skips it),
    /// unsourced weight on shared edges stays, and the source's name
    /// stays interned so re-asserting from it later just works. The
    /// unlinked attribution records remain as dead space in their
    /// append-only table.
    pub fn retract_source(&mut self, source: &str) -> Option<usize> {
        let &source_id = self.source_ids.get(source)?;
        let mut touched = 0usize;
        for edge_index in 0..self.edges.len() {
            // Locate the source's attribution and its predecessor.
            let mut previous = NIL;
            let mut cursor = self.edges[edge_index].first_attribution;
            let mut found = None;
            while cursor != NIL {
                let record = &self.attributions[cursor as usize];
                if record.source == source_id {
                    found = Some((previous, cursor, record.weight));
                    break;
                }
                previous = cursor;
                cursor = record.next;
            }
            let Some((previous, cursor, weight)) = found else {
                continue;
            };

            let next = self.attributions[cursor as usize].next;
            let edge = &mut self.edges[edge_index];
            edge.weight -= weight;
            if previous == NIL {
                edge.first_attribution = next;
            }
            if edge.last_attribution == cursor {
                edge.last_attribution = previous;
            }
            if previous != NIL {
                self.attributions[previous as usize].next = next;
            }
            touched += 1;
        }
        Some(touched)
    }

    /// Every concept alias as (alias, canonical) pairs in registration
    /// order — one coherent table, so workflows that treat the alias
    /// vocabulary as a unit (export names, translate, re-import) never
    /// have to walk the records.
    pub fn concept_aliases(&self) -> Vec<(&str, &str)> {
        self.alias_pairs(&self.concept_aliases, Self::concept_name)
    }

    /// Every label alias as (alias, canonical) pairs in registration
    /// order.
    pub fn label_aliases(&self) -> Vec<(&str, &str)> {
        self.alias_pairs(&self.label_aliases, Self::label_name)
    }

    /// Materializes one namespace's alias table as (alias, canonical)
    /// name pairs in registration order.
    fn alias_pairs(
        &self,
        records: &[AliasRecord],
        name_of: fn(&Self, u32) -> &str,
    ) -> Vec<(&str, &str)> {
        records
            .iter()
            .map(|record| {
                (
                    self.arena_str(record.name_offset, record.name_len),
                    name_of(self, record.target),
                )
            })
            .collect()
    }

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
            weight: edge.weight,
            attributions: self
                .attribution_chain(edge.first_attribution)
                .map(|(attribution_id, record)| Attribution {
                    source: self.source_name(record.source).to_string(),
                    weight: record.weight,
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

/// Registers one alternative spelling in one namespace — the shared
/// mechanics behind [`Context::add_concept_alias`] and
/// [`Context::add_label_alias`], which differ only in which alias
/// table, entry index, and lookup map make up their namespace.
/// `full_message` names the alias table in the capacity error. All
/// alias semantics live here: resolution of `canonical` through the
/// lookup map (so aliasing to an alias lands on the true canonical
/// record), idempotent re-registration, conflict refusal, and the
/// all-or-nothing capacity checks before anything mutates.
fn add_alias(
    arena: &mut Vec<u8>,
    aliases: &mut Vec<AliasRecord>,
    index: &mut EntryIndex,
    ids: &mut HashMap<String, u32>,
    alias: String,
    canonical: &str,
    full_message: &'static str,
) -> Result<(), AliasError> {
    let Some(&target) = ids.get(canonical) else {
        return Err(AliasError::UnknownCanonical);
    };
    if let Some(&existing) = ids.get(&alias) {
        return if existing == target {
            Ok(())
        } else {
            Err(AliasError::Conflict)
        };
    }
    if !ids_left(aliases.len(), 1) {
        return Err(AliasError::Full(ContextFull(full_message)));
    }
    if !arena_fits(arena.len(), alias.len()) {
        return Err(AliasError::Full(ContextFull(
            "the string arena is out of offset space",
        )));
    }
    let (name_offset, name_len) = intern_name(arena, &alias);
    aliases.push(AliasRecord {
        name_offset,
        name_len,
        target,
    });
    index.push(&alias, target);
    ids.insert(alias, target);
    Ok(())
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

/// One namespace's entry index — the derived, allocation-free machinery
/// behind [`Context::resolve`]. Every entry spelling (canonical names
/// now; aliases are designed to join the same index) is stored in
/// normalized form in one arena, and a character-bigram posting index
/// over those forms catches near-miss spellings that containment cannot.
/// Extended on every intern, rebuilt from the canonical names on load,
/// never persisted. Offsets are usize rather than u32 because
/// normalization can outgrow the offset space the canonical arena is
/// held to.
#[derive(Debug, Default)]
struct EntryIndex {
    arena: String,
    spans: Vec<EntrySpan>,
    /// Bigram (two chars packed) → indexes into `spans` whose normalized
    /// form contains it, each span listed once per distinct bigram.
    bigrams: HashMap<u64, Vec<u32>>,
    /// Total posting-list entries, kept for O(1) footprint estimates.
    posting_entries: usize,
}

/// One entry spelling: where its normalized form sits in the arena, its
/// precomputed char count, and which record it resolves to.
#[derive(Debug, Clone, Copy)]
struct EntrySpan {
    start: usize,
    end: usize,
    chars: usize,
    /// The record this spelling resolves to: its own id for a canonical
    /// name, the canonical's id for an alias.
    target: u32,
}

/// The default floor for bigram-overlap matches: Dice below this is
/// noise, not a near-miss spelling, and is dropped rather than surfacing
/// distant concepts on every shared 2-gram. Vocabularies differ in how
/// much fuzz they can afford, so this is only the default — tunable per
/// context via [`Context::set_dice_floor`] and per call via
/// [`Context::resolve_with_floor`].
const DICE_FLOOR: f64 = 0.3;

impl EntryIndex {
    fn push(&mut self, spelling: &str, target: u32) {
        let normalized = normalize(spelling);
        let span_index = claim_id(self.spans.len(), "entry index");
        let start = self.arena.len();
        self.arena.push_str(&normalized);

        let mut seen: HashSet<u64> = HashSet::new();
        for bigram in bigrams_of(&normalized) {
            if seen.insert(bigram) {
                self.bigrams.entry(bigram).or_default().push(span_index);
                self.posting_entries += 1;
            }
        }
        self.spans.push(EntrySpan {
            start,
            end: self.arena.len(),
            chars: normalized.chars().count(),
            target,
        });
    }

    /// Scores every spelling against `cue` and returns the best score per
    /// target record, with the [`MatchKind`] that earned it. Two tiers
    /// share the [0, 1] scale: exact normalized match is 1.0 and
    /// containment either way scores by character coverage of the longer
    /// string; spellings that fail containment can still match by bigram
    /// Dice overlap (near-misses like 青嶺酒蔵 for 青嶺酒造), floored at
    /// `dice_floor`. A target keeps the best score any of its spellings
    /// earned. Exact means "some spelling" — whether that spelling was
    /// the canonical name or an alias is the caller's refinement, since
    /// only it knows the names.
    fn resolve(&self, cue: &str, dice_floor: f64) -> HashMap<u32, (f64, MatchKind)> {
        let needle = normalize(cue);
        if needle.is_empty() {
            return HashMap::new();
        }
        let needle_chars = needle.chars().count();

        let mut best: HashMap<u32, (f64, MatchKind)> = HashMap::new();
        let record = |target: u32,
                      score: f64,
                      kind: MatchKind,
                      best: &mut HashMap<u32, (f64, MatchKind)>| {
            let slot = best.entry(target).or_insert((0.0, kind));
            if score > slot.0 {
                *slot = (score, kind);
            }
        };

        // Containment tier: a linear scan of the packed normalized forms.
        // A spelling matched here keeps its coverage score — the fuzzy
        // tier below is a fallback for spellings containment cannot
        // catch, not a second opinion on ones it already scored.
        let mut contained: HashSet<u32> = HashSet::new();
        let mut exact = false;
        for (span_index, span) in self.spans.iter().enumerate() {
            let haystack = &self.arena[span.start..span.end];
            let (score, kind) = if haystack == needle {
                exact = true;
                (1.0, MatchKind::Exact)
            } else if haystack.contains(needle.as_str()) || needle.contains(haystack) {
                let shorter = needle_chars.min(span.chars);
                let longer = needle_chars.max(span.chars);
                (shorter as f64 / longer as f64, MatchKind::Containment)
            } else {
                continue;
            };
            contained.insert(span_index as u32);
            record(span.target, score, kind, &mut best);
        }
        // An exact hit means the entry job is done: the cue IS a stored
        // spelling. Near-miss hunting would only spend posting-list time
        // widening a cue that already landed.
        if exact {
            return best;
        }

        // Fuzzy tier: only spellings sharing at least one bigram with the
        // cue are ever touched, via the posting lists. Bigrams carried by
        // a large share of all spellings (think a 株式会社 prefix on
        // every company name) discriminate nothing and would make every
        // lookup scan the whole namespace, so they are stop-grams:
        // their postings are never scanned AND they are excluded from
        // both sides of the Dice denominator, so the score becomes an
        // overlap over informative bigrams only. Boilerplate neither
        // costs time nor drags scores down — a typo in the distinctive
        // part of a name still lands however long the shared prefix is.
        let stop_gram = (self.spans.len() / 20).max(64);
        let is_stop = |bigram: u64| {
            self.bigrams
                .get(&bigram)
                .is_some_and(|postings| postings.len() > stop_gram)
        };
        let needle_bigrams: HashSet<u64> = bigrams_of(&needle).collect();
        let informative_needle = needle_bigrams
            .iter()
            .filter(|&&bigram| !is_stop(bigram))
            .count();
        if informative_needle > 0 {
            let mut shared: HashMap<u32, u32> = HashMap::new();
            for bigram in &needle_bigrams {
                if let Some(postings) = self.bigrams.get(bigram)
                    && postings.len() <= stop_gram
                {
                    for &span_index in postings {
                        if !contained.contains(&span_index) {
                            *shared.entry(span_index).or_insert(0) += 1;
                        }
                    }
                }
            }
            for (span_index, count) in shared {
                let span = self.spans[span_index as usize];
                // The candidate set is small (≥ 1 shared informative
                // bigram), so recounting its informative bigrams here is
                // cheap — and unlike a stored count, it cannot go stale
                // as bigrams cross the stop threshold while the
                // namespace grows.
                let text = &self.arena[span.start..span.end];
                let informative_span: HashSet<u64> = bigrams_of(text)
                    .filter(|&bigram| !is_stop(bigram))
                    .collect();
                if informative_span.is_empty() {
                    continue;
                }
                let dice =
                    2.0 * f64::from(count) / (informative_needle + informative_span.len()) as f64;
                if dice >= dice_floor {
                    record(span.target, dice, MatchKind::Fuzzy, &mut best);
                }
            }
        }
        best
    }

    /// Pairs of DIFFERENT records whose spellings overlap suspiciously —
    /// fork candidates for a vocabulary audit. The same posting-list
    /// machinery as resolve, turned name-against-name: stop-grams are
    /// skipped and Dice runs over informative bigrams on both sides.
    /// Alias spellings participate (a fork can hide behind one), but
    /// pairs resolving to a single target are intentional duplicates
    /// and are excluded. Returns (target_a, target_b, dice), strongest
    /// first. Cost is O(Σ posting_len²) — an explicit-audit price, not
    /// a query-path one.
    fn twins(&self, dice_floor: f64) -> Vec<(u32, u32, f64)> {
        let stop_gram = (self.spans.len() / 20).max(64);
        let is_stop = |bigram: u64| {
            self.bigrams
                .get(&bigram)
                .is_some_and(|postings| postings.len() > stop_gram)
        };
        let informative: Vec<usize> = self
            .spans
            .iter()
            .map(|span| {
                let text = &self.arena[span.start..span.end];
                let unique: HashSet<u64> = bigrams_of(text).collect();
                unique.iter().filter(|&&bigram| !is_stop(bigram)).count()
            })
            .collect();

        let mut shared: HashMap<(u32, u32), u32> = HashMap::new();
        for postings in self.bigrams.values() {
            if postings.len() > stop_gram {
                continue;
            }
            // Postings are appended in span order, so a < b holds.
            for (index, &a) in postings.iter().enumerate() {
                for &b in &postings[index + 1..] {
                    *shared.entry((a, b)).or_insert(0) += 1;
                }
            }
        }

        let mut best: HashMap<(u32, u32), f64> = HashMap::new();
        for ((a, b), count) in shared {
            let target_a = self.spans[a as usize].target;
            let target_b = self.spans[b as usize].target;
            if target_a == target_b {
                continue;
            }
            let denominator = informative[a as usize] + informative[b as usize];
            if denominator == 0 {
                continue;
            }
            let dice = 2.0 * f64::from(count) / denominator as f64;
            if dice < dice_floor {
                continue;
            }
            let key = (target_a.min(target_b), target_a.max(target_b));
            let slot = best.entry(key).or_insert(0.0);
            if dice > *slot {
                *slot = dice;
            }
        }
        let mut twins: Vec<(u32, u32, f64)> = best
            .into_iter()
            .map(|((a, b), dice)| (a, b, dice))
            .collect();
        twins.sort_by(|x, y| {
            y.2.total_cmp(&x.2)
                .then_with(|| (x.0, x.1).cmp(&(y.0, y.1)))
        });
        twins
    }

    /// Rough resident bytes of this index, for cache budgeting.
    fn footprint(&self) -> usize {
        const MAP_ENTRY_OVERHEAD: usize = 48;
        self.arena.len()
            + self.spans.len() * size_of::<EntrySpan>()
            + self.bigrams.len() * (size_of::<u64>() + MAP_ENTRY_OVERHEAD)
            + self.posting_entries * size_of::<u32>()
    }
}

/// The one normalization both sides of every entry comparison go
/// through: Unicode NFKC (folding full-width romaji, half-width kana,
/// and compatibility forms), lowercasing, and katakana → hiragana, so
/// "Ａｐｐｌｅ" meets "apple" and "リンゴ" meets "りんご". Applying the
/// same function to stored spellings and cues is what makes the folds
/// safe — neither side is ever compared raw against a folded form.
fn normalize(name: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    name.nfkc()
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

/// Adjacent character pairs of a normalized form, packed into u64 keys.
fn bigrams_of(text: &str) -> impl Iterator<Item = u64> {
    text.chars()
        .zip(text.chars().skip(1))
        .map(|(a, b)| ((a as u64) << 32) | b as u64)
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
    use std::collections::HashSet;

    use super::*;

    fn assoc(subject: &str, label: &str, object: &str, weight: f64) -> Association {
        Association {
            subject: subject.to_string(),
            label: label.to_string(),
            object: object.to_string(),
            weight,
            attributions: Vec::new(),
        }
    }

    /// Reads the stored weight of one exact triple through the public API.
    fn weight_between(context: &Context, subject: &str, label: &str, object: &str) -> f64 {
        let matches = context.query(Some(subject), Some(label), Some(object));
        assert_eq!(
            matches.len(),
            1,
            "expected exactly one association for {subject}/{label}/{object}"
        );
        matches[0].weight
    }

    fn associate_examples(context: &mut Context) {
        // 私はりんごが好きです
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        // 私もみかんは大好きです
        context.associate("私", "好き", "みかん", 2.0).unwrap();
        // 私はバナナが好きではありません
        context.associate("私", "好き", "バナナ", -1.0).unwrap();
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
    fn repeated_associations_accumulate_additively() {
        let mut context = Context::default();

        // The first mention seeds the weight directly.
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        assert_eq!(weight_between(&context, "私", "好き", "りんご"), 1.0);

        // Every later agreeing repeat adds its own magnitude onto the
        // existing weight, so repetition strengthens an association and a
        // more emphatic restatement (e.g. "大好き" carrying a bigger weight)
        // moves it further than a mild one would.
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        assert_eq!(weight_between(&context, "私", "好き", "りんご"), 2.0);

        context.associate("私", "好き", "りんご", 5.0).unwrap();
        assert_eq!(weight_between(&context, "私", "好き", "りんご"), 7.0);
    }

    #[test]
    fn opposite_signed_evidence_nets_against_the_existing_weight() {
        let mut context = Context::default();

        context.associate("私", "好き", "りんご", 2.0).unwrap();
        // Contradicts, but not enough to overturn it.
        context.associate("私", "好き", "りんご", -0.5).unwrap();
        assert_eq!(weight_between(&context, "私", "好き", "りんご"), 1.5);

        // Contradicts hard enough this time to flip the sign outright.
        context.associate("私", "好き", "りんご", -3.0).unwrap();
        assert_eq!(weight_between(&context, "私", "好き", "りんご"), -1.5);
    }

    #[test]
    fn a_weight_can_cross_zero_and_keep_accumulating() {
        let mut context = Context::default();

        context.associate("私", "好き", "りんご", 1.0).unwrap();
        context.associate("私", "好き", "りんご", -1.0).unwrap();
        assert_eq!(weight_between(&context, "私", "好き", "りんご"), 0.0);

        // Landing on exactly 0.0 is just another value to add onto, not a
        // dead end — the next call accumulates from there like any other.
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        assert_eq!(weight_between(&context, "私", "好き", "りんご"), 1.0);
    }

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
        assert_eq!(recalled[0].weight, 2.0);
        assert_eq!(
            recalled[0].attributions,
            vec![
                Attribution {
                    source: "IPA公式".to_string(),
                    weight: 1.0,
                    paragraph: None,
                },
                Attribution {
                    source: "解説記事".to_string(),
                    weight: 1.0,
                    paragraph: None,
                },
            ]
        );
    }

    #[test]
    fn attributions_tell_corroboration_apart_from_single_source_emphasis() {
        let mut context = Context::default();
        // Same total weight, different evidence story.
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
        assert_eq!(corroborated.weight, emphatic.weight);
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
        assert_eq!(recalled[0].weight, 1.5);
        assert_eq!(
            recalled[0].attributions,
            vec![Attribution {
                source: "文書1".to_string(),
                weight: 1.5,
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
        assert_eq!(recalled[0].weight, 1.5);
        assert_eq!(
            recalled[0].attributions,
            vec![Attribution {
                source: "文書1".to_string(),
                weight: 1.5,
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
        assert_eq!(recalled[0].weight, 1.5);
        assert_eq!(
            recalled[0].attributions,
            vec![Attribution {
                source: "文書1".to_string(),
                weight: 0.5,
                paragraph: None,
            }]
        );
    }

    #[test]
    fn activate_ranks_direct_strong_edges_above_weak_ones() {
        let mut context = Context::default();
        context.associate("起点", "強い関係", "A", 3.0).unwrap();
        context.associate("起点", "弱い関係", "B", 1.0).unwrap();

        let ranked = context.activate(&["起点"], 0.5, 10);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].association.object, "A");
        assert_eq!(ranked[0].strength, 1.5); // 1.0 * 0.5 * |3.0|
        assert_eq!(ranked[1].association.object, "B");
        assert_eq!(ranked[1].strength, 0.5); // 1.0 * 0.5 * |1.0|
    }

    #[test]
    fn activate_decays_with_distance() {
        let mut context = Context::default();
        // A chain of equal weights: nearer must outrank farther.
        context.associate("起点", "r", "近い", 1.0).unwrap();
        context.associate("近い", "r", "遠い", 1.0).unwrap();

        let ranked = context.activate(&["起点"], 0.5, 10);
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

        let ranked = context.activate(&["起点"], 0.5, 100);
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

        let ranked = context.activate(&["私"], 0.5, 10);
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

        let ranked = context.activate(&["a"], 0.5, 10);
        assert_eq!(ranked.len(), 1);
        assert_eq!(ranked[0].association.object, "c");
    }

    #[test]
    fn activate_truncates_to_the_strongest_limit_results() {
        let mut context = Context::default();
        context.associate("起点", "r1", "a", 3.0).unwrap();
        context.associate("起点", "r2", "b", 2.0).unwrap();
        context.associate("起点", "r3", "c", 1.0).unwrap();

        let top_two = context.activate(&["起点"], 0.5, 2);
        assert_eq!(top_two.len(), 2);
        assert_eq!(top_two[0].association.object, "a");
        assert_eq!(top_two[1].association.object, "b");
    }

    #[test]
    fn activate_ignores_unknown_origins() {
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();

        assert!(context.activate(&["存在しない概念"], 0.5, 10).is_empty());
        assert_eq!(
            context.activate(&["存在しない概念", "私"], 0.5, 10).len(),
            1
        );
    }

    #[test]
    fn resolve_finds_exact_and_partial_concept_names() {
        let mut context = Context::default();
        context
            .associate("10大脅威選考会", "選出する", "脅威候補", 1.0)
            .unwrap();

        // Exact match scores 1.0.
        let exact = context.resolve("脅威候補");
        assert_eq!(exact[0].name, "脅威候補");
        assert_eq!(exact[0].score, 1.0);

        // A fragment resolves to the concept containing it, scored by how
        // much of the full name it covers.
        let partial = context.resolve("選考会");
        assert_eq!(partial.len(), 1);
        assert_eq!(partial[0].name, "10大脅威選考会");
        assert_eq!(partial[0].score, 3.0 / 8.0);
    }

    #[test]
    fn resolve_matches_when_the_cue_contains_the_concept() {
        let mut context = Context::default();
        context
            .associate("10大脅威選考会", "選出する", "脅威候補", 1.0)
            .unwrap();

        // A long query phrase still lands on the concept buried inside it.
        let hits = context.resolve("10大脅威選考会について教えて");
        assert_eq!(hits[0].name, "10大脅威選考会");
        assert!(hits[0].score > 0.0 && hits[0].score < 1.0);
    }

    #[test]
    fn resolve_ranks_tighter_matches_first() {
        let mut context = Context::default();
        context
            .associate("10大脅威選考会", "選出する", "脅威候補", 1.0)
            .unwrap();
        context.associate("脅威", "分類", "リスク", 1.0).unwrap();

        // "脅威" matches itself exactly, 脅威候補 half, the committee less.
        let hits = context.resolve("脅威");
        let concepts: Vec<&str> = hits.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(concepts, vec!["脅威", "脅威候補", "10大脅威選考会"]);
        assert_eq!(hits[1].score, 0.5); // 2 chars of 4
    }

    #[test]
    fn resolve_is_ascii_case_insensitive() {
        let mut context = Context::default();
        context.associate("Apple", "分類", "果物", 1.0).unwrap();

        let hits = context.resolve("apple");
        assert_eq!(hits[0].name, "Apple");
        assert_eq!(hits[0].score, 1.0);
    }

    #[test]
    fn resolve_returns_nothing_for_unrelated_or_empty_cues() {
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();

        assert!(context.resolve("ぶどう").is_empty());
        assert!(context.resolve("").is_empty());
    }

    #[test]
    fn aliases_resolve_at_every_entry_point() {
        let mut context = Context::default();
        context
            .associate("青嶺酒造", "創業年", "1907年", 1.0)
            .unwrap();
        context
            .add_concept_alias("Aomine Brewery", "青嶺酒造")
            .unwrap();
        context.add_label_alias("設立年", "創業年").unwrap();

        // Reads through the alias land on canonical knowledge, and the
        // results carry the canonical spelling.
        assert_eq!(
            context.query(Some("Aomine Brewery"), Some("設立年"), None),
            vec![assoc("青嶺酒造", "創業年", "1907年", 1.0)]
        );
        assert_eq!(context.recall("Aomine Brewery").len(), 1);
        assert_eq!(
            context.describe("Aomine Brewery").unwrap().concept,
            "青嶺酒造"
        );
        assert_eq!(context.explore(&["Aomine Brewery"], 1).len(), 1);

        // resolve surfaces the canonical name for an alias hit.
        assert_eq!(context.resolve("aomine")[0].name, "青嶺酒造");
        assert_eq!(context.resolve_label("設立")[0].name, "創業年");

        // Writes through the alias accumulate into the canonical concept
        // instead of forking a new one.
        context
            .associate("Aomine Brewery", "設立年", "1907年", 1.0)
            .unwrap();
        assert_eq!(context.concept_count(), 2);
        assert_eq!(
            weight_between(&context, "青嶺酒造", "創業年", "1907年"),
            2.0
        );

        // Vocabulary views stay canonical-only; aliases live in their
        // own exportable tables.
        assert_eq!(context.labels(), vec!["創業年"]);
        assert_eq!(
            context.concept_aliases(),
            vec![("Aomine Brewery", "青嶺酒造")]
        );
        assert_eq!(context.label_aliases(), vec![("設立年", "創業年")]);
    }

    #[test]
    fn alias_conflicts_and_unknowns_are_rejected() {
        let mut context = Context::default();
        context
            .associate("IPA", "公開する", "10大脅威", 1.0)
            .unwrap();
        context
            .associate("情報処理推進機構", "所在地", "東京", 1.0)
            .unwrap();

        assert_eq!(
            context.add_concept_alias("独法", "存在しない概念"),
            Err(AliasError::UnknownCanonical)
        );
        // Two spellings that both already exist as concepts cannot be
        // aliased together — that would be a merge.
        assert_eq!(
            context.add_concept_alias("IPA", "情報処理推進機構"),
            Err(AliasError::Conflict)
        );
        // Re-registering the same mapping is idempotent; re-pointing the
        // alias elsewhere is a conflict.
        assert_eq!(
            context.add_concept_alias("機構", "情報処理推進機構"),
            Ok(())
        );
        assert_eq!(
            context.add_concept_alias("機構", "情報処理推進機構"),
            Ok(())
        );
        assert_eq!(
            context.add_concept_alias("機構", "IPA"),
            Err(AliasError::Conflict)
        );
        // Aliasing to an alias resolves to the true canonical record.
        assert_eq!(context.add_concept_alias("kikou", "機構"), Ok(()));
        assert_eq!(
            context.concept_aliases(),
            vec![("機構", "情報処理推進機構"), ("kikou", "情報処理推進機構")]
        );
    }

    #[test]
    fn a_removed_alias_stops_resolving_and_frees_its_spelling() {
        let mut context = Context::default();
        context
            .associate("青嶺酒造", "創業年", "1907年", 1.0)
            .unwrap();
        context.associate("高瀬", "役職", "杜氏", 1.0).unwrap();
        context.add_concept_alias("Aomine", "青嶺酒造").unwrap();
        context.add_label_alias("設立年", "創業年").unwrap();

        // Withdrawal names what the alias pointed at; the spelling
        // stops resolving while the canonical keeps its knowledge.
        assert_eq!(
            context.remove_concept_alias("Aomine").as_deref(),
            Some("青嶺酒造")
        );
        assert!(context.query(Some("Aomine"), None, None).is_empty());
        assert!(context.resolve("aomine").is_empty());
        assert!(context.concept_aliases().is_empty());
        assert_eq!(context.recall("青嶺酒造").len(), 1);

        // Not-an-alias refusals: unknown spellings, and canonical
        // names — removal must never be able to unname a record.
        assert_eq!(context.remove_concept_alias("Aomine"), None);
        assert_eq!(context.remove_concept_alias("青嶺酒造"), None);

        // The spelling is free again, pointing elsewhere this time —
        // the un-wedging move a mis-registration needs.
        context.add_concept_alias("Aomine", "高瀬").unwrap();
        assert_eq!(context.describe("Aomine").unwrap().concept, "高瀬");

        // Labels mirror, and the removal survives an image roundtrip
        // (the rebuilt entry indexes included).
        assert_eq!(
            context.remove_label_alias("設立年").as_deref(),
            Some("創業年")
        );
        let reborn = Context::from_bytes(&context.to_bytes()).unwrap();
        assert_eq!(reborn.describe("Aomine").unwrap().concept, "高瀬");
        assert!(reborn.label_aliases().is_empty());
        assert_eq!(reborn.resolve("青嶺")[0].name, "青嶺酒造");
    }

    #[test]
    fn label_alias_conflicts_and_unknowns_are_rejected() {
        let mut context = Context::default();
        context
            .associate("青嶺酒造", "創業年", "1907年", 1.0)
            .unwrap();
        context
            .associate("青嶺酒造", "設立地", "霧沢町", 1.0)
            .unwrap();

        assert_eq!(
            context.add_label_alias("操業開始", "存在しないラベル"),
            Err(AliasError::UnknownCanonical)
        );
        // Two spellings that both already exist as labels cannot be
        // aliased together — that would be a merge.
        assert_eq!(
            context.add_label_alias("創業年", "設立地"),
            Err(AliasError::Conflict)
        );
        // Re-registering the same mapping is idempotent; re-pointing the
        // alias elsewhere is a conflict.
        assert_eq!(context.add_label_alias("設立年", "創業年"), Ok(()));
        assert_eq!(context.add_label_alias("設立年", "創業年"), Ok(()));
        assert_eq!(
            context.add_label_alias("設立年", "設立地"),
            Err(AliasError::Conflict)
        );
        // Aliasing to an alias resolves to the true canonical record.
        assert_eq!(context.add_label_alias("founded", "設立年"), Ok(()));
        assert_eq!(
            context.label_aliases(),
            vec![("設立年", "創業年"), ("founded", "創業年")]
        );
        // The namespaces stay separate: a label spelling is not a
        // concept spelling, so it cannot anchor a concept alias.
        assert_eq!(
            context.add_concept_alias("蔵の誕生", "創業年"),
            Err(AliasError::UnknownCanonical)
        );
    }

    #[test]
    fn resolve_label_floor_is_tunable_per_context_and_per_call() {
        let mut context = Context::default();
        context
            .associate("青嶺酒造", "仕込み水源", "雲居山", 1.0)
            .unwrap();

        // One shared informative bigram of 4+4: Dice = 0.25 — under the
        // 0.3 default, so the fuzzy cue misses without an override.
        let cue = "水源はどこ";
        assert!(
            !context
                .resolve_label(cue)
                .iter()
                .any(|r| r.name == "仕込み水源")
        );
        assert!(
            context
                .resolve_label_with_floor(cue, 0.2)
                .iter()
                .any(|r| r.name == "仕込み水源")
        );
        // The context floor governs the label namespace exactly as it
        // does concepts.
        context.set_dice_floor(Some(0.2));
        assert!(
            context
                .resolve_label(cue)
                .iter()
                .any(|r| r.name == "仕込み水源")
        );
    }

    #[test]
    fn aliases_survive_the_image_roundtrip_and_v1_images_still_load() {
        let mut context = Context::default();
        context
            .associate("青嶺酒造", "創業年", "1907年", 1.0)
            .unwrap();
        context.add_concept_alias("Aomine", "青嶺酒造").unwrap();
        context.add_label_alias("設立年", "創業年").unwrap();

        let restored = Context::from_bytes(&context.to_bytes()).expect("v2 image must load");
        assert_eq!(restored.concept_aliases(), context.concept_aliases());
        assert_eq!(restored.label_aliases(), context.label_aliases());
        assert_eq!(
            restored.query(Some("Aomine"), Some("設立年"), None),
            context.query(Some("Aomine"), Some("設立年"), None)
        );

        // A version-1 image is the current image minus the watermark
        // (8 header bytes), the two alias sections, and the attribution-
        // locator section. For this aliasless context: header 24 (magic
        // 8, version 4, padding 4, watermark 8), edges 8+40, attributions
        // 8, concepts 8+64, labels 8+20, sources 8 → the alias counts sit
        // at 188..204, and the (empty) locator count sits at 204..212.
        // Keep the 16-byte v1 header, cut the watermark, the alias
        // counts, and the locator count, set the version back to 1; the
        // image must load with no aliases.
        let mut aliasless = Context::default();
        aliasless.associate("私", "好き", "りんご", 1.0).unwrap();
        let v3 = aliasless.to_bytes();
        let mut v1 = Vec::new();
        v1.extend_from_slice(&v3[..16]);
        v1.extend_from_slice(&v3[24..188]);
        v1.extend_from_slice(&v3[212..]);
        v1[8..12].copy_from_slice(&1u32.to_le_bytes());
        let loaded = Context::from_bytes(&v1).expect("v1 image must still load");
        assert_eq!(loaded.recall("私").len(), 1);
        assert!(loaded.concept_aliases().is_empty());

        // An alias record pointing at a nonexistent concept is caught.
        // Same section math, plus one 12-byte alias record whose target
        // field is its last 4 bytes: 188 + 8 → record at 196, target at
        // 204..208.
        let mut with_alias = Context::default();
        with_alias.associate("私", "好き", "りんご", 1.0).unwrap();
        with_alias.add_concept_alias("わたし", "私").unwrap();
        let mut corrupt = with_alias.to_bytes();
        corrupt[204..208].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Context::from_bytes(&corrupt).is_err());
    }

    #[test]
    fn applied_seq_round_trips_through_the_image() {
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        assert_eq!(context.applied_seq(), 0, "fresh contexts start at 0");
        context.set_applied_seq(42);

        let restored = Context::from_bytes(&context.to_bytes()).unwrap();
        assert_eq!(restored.applied_seq(), 42);
        assert_eq!(restored.recall("私").len(), 1);
    }

    #[test]
    fn v2_images_load_with_a_zero_watermark() {
        // A v2 image is the current image minus the 8 watermark bytes
        // and the attribution-locator section (v2 predates both), with
        // the version field set back to 2 — this pins BOTH the
        // backward-compat read and the version RANGE check (a two-value
        // check like `!=1 && !=current` would reject exactly this).
        // Section math for this context (1 edge, 1 concept alias, no
        // sources, no locators): header 24, edges 8+40, attributions 8,
        // concepts 8+64, labels 8+20, sources 8, concept_aliases 8+12,
        // label_aliases 8 → the alias tables end at 216, and the
        // (empty) locator count occupies 216..224.
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        context.add_concept_alias("わたし", "私").unwrap();
        context.set_applied_seq(7); // must NOT survive into the v2 bytes
        let v3 = context.to_bytes();
        let mut v2 = Vec::new();
        v2.extend_from_slice(&v3[..16]);
        v2.extend_from_slice(&v3[24..216]);
        v2.extend_from_slice(&v3[224..]);
        v2[8..12].copy_from_slice(&2u32.to_le_bytes());

        let loaded = Context::from_bytes(&v2).expect("v2 image must load");
        assert_eq!(loaded.applied_seq(), 0);
        assert_eq!(loaded.concept_aliases(), vec![("わたし", "私")]);
    }

    #[test]
    fn v3_images_load_with_no_locators() {
        // A v3 image is the current image minus only the attribution-
        // locator section (v3 already has the watermark and both alias
        // tables — it just predates locators). Section math for this
        // context (1 edge, 1 sourced attribution, no aliases): header
        // 24, edges 8+40, attributions 8+16, concepts 8+64, labels
        // 8+20, sources 8+8, concept_aliases 8, label_aliases 8 → the
        // (empty) locator count occupies 228..236.
        let mut context = Context::default();
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書1", None)
            .unwrap();
        let v4 = context.to_bytes();
        let mut v3 = Vec::new();
        v3.extend_from_slice(&v4[..228]);
        v3.extend_from_slice(&v4[236..]);
        v3[8..12].copy_from_slice(&3u32.to_le_bytes());

        let loaded = Context::from_bytes(&v3).expect("v3 image must load");
        assert_eq!(
            loaded.recall("私")[0].attributions,
            vec![Attribution {
                source: "文書1".to_string(),
                weight: 1.0,
                paragraph: None,
            }]
        );
    }

    #[test]
    fn retract_source_withdraws_its_contributions() {
        let mut context = Context::default();
        context
            .associate_from("a", "r", "b", 1.0, "旧版", None)
            .unwrap();
        // 新版 and 第三者 carry locators, so their retraction/round-trip
        // below also exercises the attribution-locator side table.
        context
            .associate_from("a", "r", "b", 2.0, "新版", Some(4))
            .unwrap();
        context
            .associate_from("a", "r", "b", 0.5, "第三者", Some(7))
            .unwrap();
        context
            .associate_from("a", "r", "c", 1.0, "旧版", None)
            .unwrap();
        context.associate("a", "r", "d", 1.0).unwrap(); // unsourced stays

        // 旧版 contributed to two edges; both lose exactly its share.
        assert_eq!(context.retract_source("旧版"), Some(2));
        assert_eq!(weight_between(&context, "a", "r", "b"), 2.5);
        assert_eq!(
            context.query(Some("a"), None, Some("b"))[0].attributions,
            vec![
                Attribution {
                    source: "新版".to_string(),
                    weight: 2.0,
                    paragraph: Some(4),
                },
                Attribution {
                    source: "第三者".to_string(),
                    weight: 0.5,
                    paragraph: Some(7),
                },
            ]
        );
        // A fully retracted edge nets to zero: still queryable, no
        // longer knowledge activate would carry.
        assert_eq!(weight_between(&context, "a", "r", "c"), 0.0);
        assert!(
            context.query(Some("a"), None, Some("c"))[0]
                .attributions
                .is_empty()
        );
        assert_eq!(weight_between(&context, "a", "r", "d"), 1.0);
        assert_eq!(context.retract_source("存在しない出典"), None);

        // Unlinking the tail keeps chains appendable, and the image —
        // now carrying orphaned attribution records — must round-trip.
        assert_eq!(context.retract_source("第三者"), Some(1));
        assert_eq!(weight_between(&context, "a", "r", "b"), 2.0);
        context
            .associate_from("a", "r", "b", 0.5, "旧版", None)
            .unwrap();
        assert_eq!(weight_between(&context, "a", "r", "b"), 2.5);
        let restored = Context::from_bytes(&context.to_bytes()).expect("image must load");
        assert_eq!(
            restored.query(Some("a"), None, None),
            context.query(Some("a"), None, None)
        );
    }

    #[test]
    fn dice_floor_is_tunable_per_context_and_per_call() {
        let mut context = Context::default();
        context.associate("青嶺酒造", "分類", "酒蔵", 1.0).unwrap();

        // One shared informative bigram of 4+3: Dice = 2/7 ≈ 0.286 —
        // just under the 0.3 default, so the fuzzy cue misses.
        let cue = "青嶺の純米";
        assert!(!context.resolve(cue).iter().any(|r| r.name == "青嶺酒造"));

        // A one-call override loosens exactly that call ...
        assert!(
            context
                .resolve_with_floor(cue, 0.25)
                .iter()
                .any(|r| r.name == "青嶺酒造")
        );
        assert!(!context.resolve(cue).iter().any(|r| r.name == "青嶺酒造"));

        // ... and the context setting changes the default until reset.
        context.set_dice_floor(Some(0.25));
        assert_eq!(context.dice_floor(), 0.25);
        assert!(context.resolve(cue).iter().any(|r| r.name == "青嶺酒造"));
        context.set_dice_floor(None);
        assert!(!context.resolve(cue).iter().any(|r| r.name == "青嶺酒造"));
    }

    #[test]
    fn fuzzy_matching_survives_boilerplate_heavy_namespaces() {
        let mut context = Context::default();
        // 70 filler companies push the 株式会社 prefix bigrams past the
        // stop-gram threshold (max(spans/20, 64)).
        let fillers = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほ\
                       まみむめもやゆよらりるれろわをんアイウエオカキクケコサシスセ\
                       ソタチツテトナニヌネノハヒフヘ";
        for ch in fillers.chars().filter(|c| !c.is_whitespace()).take(70) {
            context
                .associate(format!("株式会社{ch}"), "業種", "その他", 1.0)
                .unwrap();
        }
        context
            .associate("株式会社青嶺", "業種", "酒造", 1.0)
            .unwrap();

        // The typo sits in the distinctive part (峰 for 嶺). Boilerplate
        // bigrams are stop-grams on BOTH sides of the Dice, so the one
        // shared informative bigram is enough: 2·1/(2+2) = 0.5. With the
        // boilerplate left in the denominator this was 2·1/(5+5) = 0.2 —
        // below the floor, a silent miss.
        let hits = context.resolve("株式会社青峰");
        assert!(
            hits.iter().any(|hit| hit.name == "株式会社青嶺"),
            "typo in the distinctive part must land: {hits:?}"
        );
    }

    #[test]
    fn resolve_normalizes_width_case_and_kana() {
        let mut context = Context::default();
        context.associate("Apple", "分類", "果物", 1.0).unwrap();
        context.associate("りんご", "分類", "果物", 1.0).unwrap();

        // Full-width romaji folds onto the stored ASCII spelling ...
        let fullwidth = context.resolve("Ａｐｐｌｅ");
        assert_eq!(fullwidth[0].name, "Apple");
        assert_eq!(fullwidth[0].score, 1.0);

        // ... and katakana folds onto hiragana. The stored spelling is
        // returned, not the cue's variant.
        let katakana = context.resolve("リンゴ");
        assert_eq!(katakana[0].name, "りんご");
        assert_eq!(katakana[0].score, 1.0);
    }

    #[test]
    fn resolve_finds_near_miss_spellings_by_bigram_overlap() {
        let mut context = Context::default();
        context
            .associate("青嶺酒造", "代表銘柄", "青嶺", 1.0)
            .unwrap();

        // 蔵 for 造: containment fails both ways, but two of three
        // bigrams survive the typo — and that fuzzy hit outranks the
        // loose containment match on the brand name 青嶺 (0.5).
        let hits = context.resolve("青嶺酒蔵");
        assert_eq!(hits[0].name, "青嶺酒造");
        assert!((hits[0].score - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(hits[1].name, "青嶺");

        // Below the Dice floor nothing surfaces: an unrelated spelling
        // sharing no bigrams stays out entirely.
        assert!(context.resolve("辛口純米").is_empty());
    }

    #[test]
    fn resolutions_name_the_string_relation_behind_each_score() {
        let mut context = Context::default();
        context.associate("東京都", "分類", "首都", 1.0).unwrap();
        context.associate("京都", "所在", "関西", 1.0).unwrap();
        context.associate("青嶺酒造", "分類", "酒蔵", 1.0).unwrap();
        context.add_concept_alias("Kyoto", "京都").unwrap();

        // The cue IS a stored name: exact. The lookalike containing it
        // scores a strong-looking 2/3 — and says it is only containment.
        let hits = context.resolve("京都");
        assert_eq!(
            (hits[0].name.as_str(), hits[0].kind),
            ("京都", MatchKind::Exact)
        );
        assert_eq!(
            (hits[1].name.as_str(), hits[1].kind),
            ("東京都", MatchKind::Containment)
        );

        // An alias hit carries exact's certainty and explains why the
        // returned spelling differs from the cue.
        let via_alias = context.resolve("Kyoto");
        assert_eq!(via_alias[0].name, "京都");
        assert_eq!(via_alias[0].score, 1.0);
        assert_eq!(via_alias[0].kind, MatchKind::Alias);

        // A near-miss lands through the bigram tier and is labeled so.
        let typo = context.resolve("青嶺酒蔵");
        assert_eq!(typo[0].name, "青嶺酒造");
        assert_eq!(typo[0].kind, MatchKind::Fuzzy);
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
        let before = context.activate(&["青嶺酒造"], 0.5, 10)[0].strength;
        assert_eq!(before, 1.0); // 1.0 * 0.5 * |2.0|

        // Five more facts about the same origin. Under fan-normalized
        // scoring these used to drag every strength down; the water fact
        // itself has not changed, so its score must not move.
        for i in 0..5 {
            context
                .associate("青嶺酒造", format!("関係{i}"), format!("事実{i}"), 1.0)
                .unwrap();
        }
        let after = context.activate(&["青嶺酒造"], 0.5, 10);
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

        let ranked = context.activate(&["多弁", "寡黙"], 0.5, 10);
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

        let orphans = context.unreachable_from(&["青嶺酒造"]);
        assert_eq!(orphans, vec![assoc("高瀬", "役職", "杜氏", 1.0)]);

        // The membership edge repairs the island; the audit comes back
        // clean.
        context.associate("青嶺酒造", "杜氏", "高瀬", 1.0).unwrap();
        assert!(context.unreachable_from(&["青嶺酒造"]).is_empty());
    }

    #[test]
    fn unreachable_from_an_unknown_origin_reports_everything() {
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();

        assert_eq!(context.unreachable_from(&["存在しない概念"]).len(), 1);
        assert_eq!(context.unreachable_from(&[]).len(), 1);
    }

    #[test]
    fn resolve_label_finds_similar_relation_labels() {
        let mut context = Context::default();
        context
            .associate("蔵人", "住み込む場所", "蔵", 1.0)
            .unwrap();
        context
            .associate("蔵人", "住み込む期間", "冬", 1.0)
            .unwrap();

        // Check before mint: an ingester about to coin "住み込む" first
        // asks what similar labels exist, and reuses instead of forking
        // the vocabulary.
        let hits = context.resolve_label("住み込む");
        let names: Vec<&str> = hits.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["住み込む場所", "住み込む期間"]);
        assert_eq!(hits[0].score, 4.0 / 6.0);

        assert!(context.resolve_label("経験").is_empty());
    }

    #[test]
    fn resolve_and_resolve_label_are_separate_namespaces() {
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();

        // "好き" exists only as a label, "りんご" only as a concept.
        assert!(context.resolve("好き").is_empty());
        assert!(context.resolve_label("りんご").is_empty());
        assert_eq!(context.resolve_label("好き")[0].score, 1.0);
    }

    #[test]
    fn labels_lists_the_relation_vocabulary_in_insertion_order() {
        let mut context = Context::default();
        context.associate("a", "r2", "b", 1.0).unwrap();
        context.associate("b", "r1", "c", 1.0).unwrap();
        context.associate("c", "r2", "d", 1.0).unwrap(); // reuse must not duplicate

        assert_eq!(context.labels(), vec!["r2", "r1"]);
    }

    #[test]
    fn image_roundtrip_preserves_every_read_path() {
        let mut context = Context::default();
        associate_examples(&mut context);
        context.associate("りんご", "分類", "果物", 1.5).unwrap();
        context
            .associate_from("決定", "手段", "投票", 1.0, "IPA公式", None)
            .unwrap();
        context
            .associate_from("決定", "手段", "投票", 0.5, "解説記事", None)
            .unwrap();
        context.associate("犬", "好き", "骨", 1.0).unwrap(); // separate component

        let restored = Context::from_bytes(&context.to_bytes()).expect("image must load");

        assert_eq!(restored.recall("私"), context.recall("私"));
        assert_eq!(restored.recall("投票"), context.recall("投票"));
        assert_eq!(
            restored.query(None, None, None),
            context.query(None, None, None)
        );
        assert_eq!(
            restored.query(Some("私"), Some("好き"), None),
            context.query(Some("私"), Some("好き"), None)
        );
        assert_eq!(
            restored.explore(&["私"], Context::UNBOUNDED),
            context.explore(&["私"], Context::UNBOUNDED)
        );
        assert_eq!(
            restored.activate(&["私"], 0.5, 10),
            context.activate(&["私"], 0.5, 10)
        );
        assert_eq!(restored.resolve("りんご"), context.resolve("りんご"));
        assert_eq!(restored.labels(), context.labels());
        assert_eq!(
            restored.unreachable_from(&["私"]),
            context.unreachable_from(&["私"])
        );
    }

    #[test]
    fn image_roundtrip_keeps_accepting_writes() {
        let mut context = Context::default();
        context
            .associate_from("a", "r", "b", 1.0, "文書1", None)
            .unwrap();

        let mut restored = Context::from_bytes(&context.to_bytes()).expect("image must load");

        // Accumulation must land on the restored edge and its restored
        // attribution — the rebuilt indexes and chain tails must all point
        // at the right records.
        restored
            .associate_from("a", "r", "b", 0.5, "文書1", None)
            .unwrap();
        assert_eq!(weight_between(&restored, "a", "r", "b"), 1.5);
        assert_eq!(
            restored.recall("a")[0].attributions,
            vec![Attribution {
                source: "文書1".to_string(),
                weight: 1.5,
                paragraph: None,
            }]
        );

        // And chains must keep extending in insertion order.
        restored.associate("a", "r", "c", 1.0).unwrap();
        restored.associate("d", "r2", "a", 1.0).unwrap();
        assert_eq!(restored.recall("a").len(), 3);
        assert_eq!(restored.labels(), vec!["r", "r2"]);
    }

    #[test]
    fn attribution_locators_survive_the_image_roundtrip() {
        let mut context = Context::default();
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書1", Some(3))
            .unwrap();
        // A second, unlocated source on the same edge must round-trip
        // as `None` alongside the first source's `Some`.
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書2", None)
            .unwrap();

        let restored = Context::from_bytes(&context.to_bytes()).expect("image must load");
        assert_eq!(restored.recall("私"), context.recall("私"));
        assert_eq!(
            restored.recall("私")[0].attributions,
            vec![
                Attribution {
                    source: "文書1".to_string(),
                    weight: 1.0,
                    paragraph: Some(3),
                },
                Attribution {
                    source: "文書2".to_string(),
                    weight: 1.0,
                    paragraph: None,
                },
            ]
        );
    }

    #[test]
    fn from_bytes_rejects_corrupt_locators() {
        // One sourced, located attribution: header 24, edges 8+40,
        // attributions 8+16, concepts 8+64, labels 8+20, sources 8+8,
        // concept_aliases 8, label_aliases 8 → the locator table starts
        // at 228 (its count, 8 bytes), and the lone record's
        // `attribution` field sits at 236..240 (same math as
        // `v3_images_load_with_no_locators`, which strips this same
        // section). Pointing it at a nonexistent attribution must be
        // caught.
        let mut context = Context::default();
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書1", Some(3))
            .unwrap();
        let mut dangling = context.to_bytes();
        dangling[236..240].copy_from_slice(&u32::MAX.to_le_bytes());
        let error = Context::from_bytes(&dangling).unwrap_err();
        assert!(error.to_string().contains("unknown attribution"), "{error}");

        // Two sourced, located attributions from two distinct sources on
        // the same edge (two attribution records, two locator records):
        // the extra 8-byte source and 16-byte attribution record shift
        // the locator table to 252..276, putting the second record's
        // `attribution` field at 268..272. Setting it to 0 — equal to
        // the first record's — breaks the strictly-increasing invariant
        // without pointing outside the attribution table.
        let mut two_sources = Context::default();
        two_sources
            .associate_from("私", "好き", "りんご", 1.0, "文書1", Some(3))
            .unwrap();
        two_sources
            .associate_from("私", "好き", "りんご", 1.0, "文書2", Some(5))
            .unwrap();
        let mut unsorted = two_sources.to_bytes();
        unsorted[268..272].copy_from_slice(&0u32.to_le_bytes());
        let error = Context::from_bytes(&unsorted).unwrap_err();
        assert!(error.to_string().contains("not sorted"), "{error}");
    }

    #[test]
    fn empty_context_roundtrips() {
        let restored =
            Context::from_bytes(&Context::default().to_bytes()).expect("image must load");
        assert!(restored.query(None, None, None).is_empty());
    }

    #[test]
    fn from_bytes_rejects_malformed_images() {
        assert!(Context::from_bytes(b"").is_err());
        assert!(Context::from_bytes(b"not an image at all").is_err());

        let mut context = Context::default();
        context
            .associate_from("私", "好き", "りんご", 1.0, "文書1", None)
            .unwrap();
        let image = context.to_bytes();

        // Truncation anywhere must be caught by section bounds.
        for len in [image.len() - 1, image.len() / 2, 9] {
            assert!(Context::from_bytes(&image[..len]).is_err());
        }
        // Trailing garbage is not silently ignored.
        let mut padded = image.clone();
        padded.push(0);
        assert!(Context::from_bytes(&padded).is_err());
        // A wrong magic or version is refused outright.
        let mut wrong_magic = image.clone();
        wrong_magic[0] ^= 0xFF;
        assert!(Context::from_bytes(&wrong_magic).is_err());
        let mut wrong_version = image.clone();
        wrong_version[8] = 0xFF;
        assert!(Context::from_bytes(&wrong_version).is_err());
    }

    #[test]
    fn from_bytes_rejects_inconsistent_records() {
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        let image = context.to_bytes();

        // The first edge record sits right after the 16-byte header and
        // the edge table's u64 count; its first field is `subject`.
        // Pointing it at a nonexistent concept must be caught.
        let mut dangling_subject = image.clone();
        dangling_subject[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(Context::from_bytes(&dangling_subject).is_err());

        // 私's `outgoing_count` sits at offset 96: header 16, edge table
        // 8 + 40, empty attribution table 8, concept count 8, then the
        // fifth u32 field of the first concept record. A count that
        // disagrees with the actual chain must be caught.
        let mut wrong_count = image.clone();
        wrong_count[96..100].copy_from_slice(&5u32.to_le_bytes());
        assert!(Context::from_bytes(&wrong_count).is_err());
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
    fn arena_fits_stops_exactly_at_the_offset_ceiling() {
        assert!(arena_fits(u32::MAX as usize - 5, 5));
        assert!(!arena_fits(u32::MAX as usize - 5, 6));
        assert!(arena_fits(u32::MAX as usize, 0)); // full arena, nothing new needed
        assert!(!arena_fits(usize::MAX, 1)); // must refuse, not overflow
    }

    #[test]
    fn counts_and_top_concepts_expose_directory_stats() {
        let mut context = Context::default();
        associate_examples(&mut context);
        context
            .associate_from("私", "食べられる", "りんご", -0.2, "文書1", None)
            .unwrap();

        assert_eq!(context.association_count(), 4);
        assert_eq!(context.concept_count(), 4);
        assert_eq!(context.label_count(), 2);
        assert_eq!(context.source_count(), 1);

        // 私 carries 4 edges, りんご 2, the rest 1 each.
        let top = context.top_concepts(2);
        assert_eq!(top, vec![("私", 4), ("りんご", 2)]);
        // A limit beyond the table is just everything.
        assert_eq!(context.top_concepts(100).len(), 4);
    }

    #[test]
    fn vocabulary_twins_surface_spelling_forks_but_not_aliases() {
        let mut context = Context::default();
        context
            .associate("青嶺酒造", "代表銘柄", "青嶺", 1.0)
            .unwrap();
        // The fork: a typo variant minted as its own concept.
        context
            .associate("青嶺酒蔵", "所在地", "霧沢町", 1.0)
            .unwrap();
        context
            .associate("蔵人", "住み込む場所", "蔵", 1.0)
            .unwrap();
        context
            .associate("蔵人", "住み込む期間", "冬", 1.0)
            .unwrap();
        // An alias is an INTENTIONAL second spelling of one record.
        context.add_concept_alias("青嶺酒屋", "青嶺酒造").unwrap();

        // 青嶺酒蔵×青嶺酒造 share 2 of 3 bigrams each: dice 2/3.
        let twins = context.similar_concepts(0.6);
        assert_eq!(twins.len(), 1, "{twins:?}");
        assert_eq!(
            (twins[0].0.as_str(), twins[0].1.as_str()),
            ("青嶺酒造", "青嶺酒蔵")
        );
        assert!((twins[0].2 - 2.0 / 3.0).abs() < 1e-9);
        // The brand/brewery containment pair (dice 0.5) stays below the
        // floor, and the alias never pairs with its own canonical.
        assert!(context.similar_concepts(0.55).len() == 1);

        // Label forks are the costliest kind; the shared 住み込む stem
        // pushes the pair over the floor for review.
        let labels = context.similar_labels(0.6);
        assert_eq!(
            (labels[0].0.as_str(), labels[0].1.as_str()),
            ("住み込む場所", "住み込む期間")
        );
    }

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
    fn label_gloss_omits_a_repeated_example_subject() {
        let mut context = Context::default();
        context.associate("青嶺酒造", "杜氏", "高瀬", 2.0).unwrap();
        context.associate("青嶺酒造", "杜氏", "山田", 1.0).unwrap();

        assert_eq!(
            context.label_gloss("杜氏", 2).unwrap(),
            "杜氏。青嶺酒造の杜氏は高瀬。杜氏は山田。"
        );
    }

    #[test]
    fn footprint_grows_with_content() {
        let mut context = Context::default();
        let empty = context.footprint();
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        assert!(context.footprint() > empty);
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

        let ranked = context.activate(&["c0"], 0.5, 100);
        assert_eq!(ranked.len(), 11);
        assert_eq!(ranked[0].association.subject, "c0");
        assert_eq!(ranked.last().unwrap().association.subject, "c10");

        // explore is the structural sweep and must stay unbounded.
        assert_eq!(context.explore(&["c0"], Context::UNBOUNDED).len(), 30);
    }
}
