use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt;

use serde::Serialize;

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

/// One source's accumulated contribution to an association's weight.
///
/// `source` is an opaque identifier chosen by the caller — a document id, a
/// URL, a chunk reference — that lets whoever retrieved the association go
/// fetch the original text behind it. The `Context` never interprets it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Attribution {
    pub source: String,
    pub weight: f64,
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
    pub association: Association,
}

/// One association returned by [`Context::activate`], carrying the
/// activation strength that reached it — the ranking signal that combines
/// how near the association is to the origins with how heavy it is relative
/// to its neighbors.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Activation {
    pub strength: f64,
    pub association: Association,
}

/// One candidate name produced by [`Context::resolve`] (concept names) or
/// [`Context::resolve_label`] (relation labels), scored by how much of the
/// longer string the lexical overlap covers (1.0 = exact match).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Resolution {
    pub name: String,
    pub score: f64,
}

/// How often one relation label appears on a concept's edges — one row of
/// a [`ConceptDescription`].
#[derive(Debug, Clone, PartialEq, Serialize)]
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

// The persistence format depends on these exact widths. A field added to
// any record must keep its table fixed-width, naturally aligned, and
// padding-free — and must bump `IMAGE_VERSION`.
const _: () = {
    assert!(size_of::<ConceptRecord>() == ConceptRecord::SIZE && align_of::<ConceptRecord>() == 4);
    assert!(size_of::<LabelRecord>() == LabelRecord::SIZE && align_of::<LabelRecord>() == 4);
    assert!(size_of::<SourceRecord>() == SourceRecord::SIZE && align_of::<SourceRecord>() == 4);
    assert!(size_of::<EdgeRecord>() == EdgeRecord::SIZE && align_of::<EdgeRecord>() == 8);
    assert!(
        size_of::<AttributionRecord>() == AttributionRecord::SIZE
            && align_of::<AttributionRecord>() == 8
    );
};

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
/// Everything a `Context` knows lives in six flat buffers: one UTF-8 string
/// arena plus five tables of fixed-width, naturally aligned, pointer-free
/// `#[repr(C)]` records (u32 ids and offsets, f64 weights). Variable-length
/// structure is expressed inside the fixed widths: strings live in the
/// arena and records hold (offset, len) pairs; adjacency lists are
/// intrusive singly-linked chains threaded through the edge records
/// (`next_outgoing` / `next_incoming` / `next_labeled`, terminated by a
/// `NIL` sentinel) and appended at the tail, which is what preserves the
/// insertion-order guarantees of the read API. Every mutation is an append
/// or an in-place field update — records never move — so the whole state
/// dumps and restores as one contiguous image via [`Context::to_bytes`] /
/// [`Context::from_bytes`].
///
/// The hash maps and lowercase name shadows are derived read-path indexes
/// over those buffers (string → id, exact triple → edge, case-folded names
/// for `resolve`). They are not part of the persistent image; `from_bytes`
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
    /// Derived index: interned name → concept id. Not persisted.
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
        self.upsert(subject.into(), label.into(), object.into(), weight, None)
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
    ) -> Result<(), ContextFull> {
        self.upsert(
            subject.into(),
            label.into(),
            object.into(),
            weight,
            Some(source.into()),
        )
    }

    fn upsert(
        &mut self,
        subject: String,
        label: String,
        object: String,
        weight: f64,
        source: Option<String>,
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
            self.attribute(edge_id, source_id, weight);
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
                        .any(|record| record.source == source_id)
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
    fn attribute(&mut self, edge_id: EdgeId, source_id: SourceId, weight: f64) {
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
    /// history. Origins naming unknown concepts contribute nothing, and
    /// `max_depth == 0` returns nothing — zero hops of association-following
    /// reaches no associations.
    pub fn explore(&self, origins: &[&str], max_depth: usize) -> Vec<Recollection> {
        let mut node_distances: HashMap<ConceptId, usize> = HashMap::new();
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
        // its nearer endpoint.
        let mut reached: HashMap<EdgeId, usize> = HashMap::new();
        while let Some(concept_id) = frontier.pop_front() {
            let hop = node_distances[&concept_id] + 1;
            if hop > max_depth {
                continue;
            }
            for edge_id in self.outgoing(concept_id).chain(self.incoming(concept_id)) {
                reached.entry(edge_id).or_insert(hop);
                let edge = &self.edges[edge_id as usize];
                for neighbor in [edge.subject, edge.object] {
                    if let Entry::Vacant(entry) = node_distances.entry(neighbor) {
                        entry.insert(hop);
                        frontier.push_back(neighbor);
                    }
                }
            }
        }

        let mut ordered: Vec<(usize, EdgeId)> = reached
            .into_iter()
            .map(|(edge_id, distance)| (distance, edge_id))
            .collect();
        ordered.sort_unstable();

        ordered
            .into_iter()
            .map(|(distance, edge_id)| Recollection {
                distance,
                association: self.association(edge_id),
            })
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
        // activation and can be settled for good.
        let mut settled: HashSet<ConceptId> = HashSet::new();
        let mut strengths: HashMap<EdgeId, f64> = HashMap::new();
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
                let strength = strengths.entry(edge_id).or_insert(0.0);
                if score > *strength {
                    *strength = score;
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
                        heap.push(Candidate {
                            activation: flow,
                            concept: neighbor,
                        });
                    }
                }
            }
        }

        let mut ranked: Vec<(f64, EdgeId)> = strengths
            .into_iter()
            .map(|(edge_id, strength)| (strength, edge_id))
            .collect();
        ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        ranked.truncate(limit);

        ranked
            .into_iter()
            .map(|(strength, edge_id)| Activation {
                strength,
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
        let mut resolutions: Vec<Resolution> = self
            .concept_index
            .resolve(cue)
            .into_iter()
            .map(|(id, score)| Resolution {
                name: self.concept_name(id).to_string(),
                score,
            })
            .collect();
        sort_resolutions(&mut resolutions);
        resolutions
    }

    /// [`Context::resolve`] for relation labels instead of concepts — the
    /// two namespaces never mix. This exists chiefly for the write path:
    /// label vocabulary forks silently ("創業年" vs "創業" vs "設立年"),
    /// and once forked, label-pinned queries stop seeing half the facts.
    /// An ingester should call this (or review [`Context::labels`]) before
    /// coining a relation spelling, and reuse a close existing label
    /// instead.
    pub fn resolve_label(&self, cue: &str) -> Vec<Resolution> {
        let mut resolutions: Vec<Resolution> = self
            .label_index
            .resolve(cue)
            .into_iter()
            .map(|(id, score)| Resolution {
                name: self.label_name(id).to_string(),
                score,
            })
            .collect();
        sort_resolutions(&mut resolutions);
        resolutions
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
            + self.attributions.len() * size_of::<AttributionRecord>();

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

    /// Walks one edge's attribution chain in first-assertion order.
    fn attribution_chain(&self, first: AttributionId) -> impl Iterator<Item = &AttributionRecord> {
        std::iter::successors(
            (first != NIL).then(|| &self.attributions[first as usize]),
            |record| (record.next != NIL).then(|| &self.attributions[record.next as usize]),
        )
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
                .map(|record| Attribution {
                    source: self.source_name(record.source).to_string(),
                    weight: record.weight,
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

/// One entry spelling: where its normalized form sits in the arena, the
/// precomputed counts its scores need, and which record it resolves to.
#[derive(Debug, Clone, Copy)]
struct EntrySpan {
    start: usize,
    end: usize,
    chars: usize,
    /// Distinct bigrams in this spelling — half the Dice denominator.
    bigram_count: u32,
    /// The record this spelling resolves to: its own id for a canonical
    /// name, the canonical's id for an alias.
    target: u32,
}

/// The floor for bigram-overlap matches: Dice below this is noise, not a
/// near-miss spelling, and is dropped rather than surfacing distant
/// concepts on every shared 2-gram.
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
            bigram_count: seen.len() as u32,
            target,
        });
    }

    /// Scores every spelling against `cue` and returns the best score per
    /// target record. Two tiers share the [0, 1] scale: exact normalized
    /// match is 1.0 and containment either way scores by character
    /// coverage of the longer string; spellings that fail containment can
    /// still match by bigram Dice overlap (near-misses like 青嶺酒蔵 for
    /// 青嶺酒造), floored at [`DICE_FLOOR`]. A target keeps the best
    /// score any of its spellings earned.
    fn resolve(&self, cue: &str) -> HashMap<u32, f64> {
        let needle = normalize(cue);
        if needle.is_empty() {
            return HashMap::new();
        }
        let needle_chars = needle.chars().count();

        let mut best: HashMap<u32, f64> = HashMap::new();
        let record = |target: u32, score: f64, best: &mut HashMap<u32, f64>| {
            let slot = best.entry(target).or_insert(0.0);
            if score > *slot {
                *slot = score;
            }
        };

        // Containment tier: a linear scan of the packed normalized forms.
        // A spelling matched here keeps its coverage score — the fuzzy
        // tier below is a fallback for spellings containment cannot
        // catch, not a second opinion on ones it already scored.
        let mut contained: HashSet<u32> = HashSet::new();
        for (span_index, span) in self.spans.iter().enumerate() {
            let haystack = &self.arena[span.start..span.end];
            let score = if haystack == needle {
                1.0
            } else if haystack.contains(needle.as_str()) || needle.contains(haystack) {
                let shorter = needle_chars.min(span.chars);
                let longer = needle_chars.max(span.chars);
                shorter as f64 / longer as f64
            } else {
                continue;
            };
            contained.insert(span_index as u32);
            record(span.target, score, &mut best);
        }

        // Fuzzy tier: only spellings sharing at least one bigram with the
        // cue are ever touched, via the posting lists.
        let needle_bigrams: HashSet<u64> = bigrams_of(&needle).collect();
        if !needle_bigrams.is_empty() {
            let mut shared: HashMap<u32, u32> = HashMap::new();
            for bigram in &needle_bigrams {
                if let Some(postings) = self.bigrams.get(bigram) {
                    for &span_index in postings {
                        if !contained.contains(&span_index) {
                            *shared.entry(span_index).or_insert(0) += 1;
                        }
                    }
                }
            }
            for (span_index, count) in shared {
                let span = self.spans[span_index as usize];
                let dice = 2.0 * f64::from(count)
                    / (needle_bigrams.len() + span.bigram_count as usize) as f64;
                if dice >= DICE_FLOOR {
                    record(span.target, dice, &mut best);
                }
            }
        }
        best
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

// ---------------------------------------------------------------------------
// Persistence: dumping and restoring the storage buffers as one image.
// ---------------------------------------------------------------------------

/// First 8 bytes of every image.
const IMAGE_MAGIC: [u8; 8] = *b"ARAGCTX\0";
/// Format version; bump whenever any record layout or section changes.
const IMAGE_VERSION: u32 = 1;
/// Magic + version + 4 bytes of padding, so the first section starts
/// 8-byte aligned.
const IMAGE_HEADER_SIZE: usize = 16;

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

impl Context {
    /// Serializes the whole network into one contiguous byte image.
    ///
    /// The image is the storage buffers written back to back: a 16-byte
    /// header, then each record table as a u64 count followed by its
    /// fixed-width records field by field in little-endian, then the
    /// string arena. Sections are ordered by descending alignment (the
    /// f64-bearing tables first), so every record sits naturally aligned
    /// within the image as well — the layout stays open to zero-copy
    /// mapping later. The derived hash indexes are not written;
    /// [`Context::from_bytes`] rebuilds them.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut image = Vec::with_capacity(
            IMAGE_HEADER_SIZE
                + 6 * size_of::<u64>()
                + self.edges.len() * EdgeRecord::SIZE
                + self.attributions.len() * AttributionRecord::SIZE
                + self.concepts.len() * ConceptRecord::SIZE
                + self.labels.len() * LabelRecord::SIZE
                + self.sources.len() * SourceRecord::SIZE
                + self.arena.len(),
        );
        image.extend_from_slice(&IMAGE_MAGIC);
        image.extend_from_slice(&IMAGE_VERSION.to_le_bytes());
        image.extend_from_slice(&[0; 4]);
        store_table(&self.edges, &mut image);
        store_table(&self.attributions, &mut image);
        store_table(&self.concepts, &mut image);
        store_table(&self.labels, &mut image);
        store_table(&self.sources, &mut image);
        image.extend_from_slice(&(self.arena.len() as u64).to_le_bytes());
        image.extend_from_slice(&self.arena);
        image
    }

    /// Restores a `Context` from an image produced by
    /// [`Context::to_bytes`], rebuilding the derived indexes.
    ///
    /// The image is fully validated before anything is trusted — magic,
    /// version, and section bounds first, then arena ranges and UTF-8,
    /// id ranges, duplicate names and triples, and every adjacency and
    /// attribution chain (ownership, length, tail, cycles) — so a
    /// truncated or tampered image comes back as an error rather than a
    /// `Context` that panics or corrupts itself later. A restored
    /// `Context` is behaviorally identical to the original, including
    /// insertion-order guarantees, and keeps accepting new associations.
    pub fn from_bytes(image: &[u8]) -> Result<Self, CorruptImage> {
        let mut reader = Reader {
            bytes: image,
            pos: 0,
        };
        if reader.take(IMAGE_MAGIC.len())? != IMAGE_MAGIC {
            return Err(CorruptImage("image does not start with the context magic"));
        }
        if reader.read_u32()? != IMAGE_VERSION {
            return Err(CorruptImage("image format version is not supported"));
        }
        reader.take(4)?; // header padding

        let edges = load_table::<EdgeRecord>(&mut reader)?;
        let attributions = load_table::<AttributionRecord>(&mut reader)?;
        let concepts = load_table::<ConceptRecord>(&mut reader)?;
        let labels = load_table::<LabelRecord>(&mut reader)?;
        let sources = load_table::<SourceRecord>(&mut reader)?;
        let arena_len = usize::try_from(reader.read_u64()?)
            .map_err(|_| CorruptImage("arena length overflows this platform"))?;
        let arena = reader.take(arena_len)?.to_vec();
        if reader.pos != image.len() {
            return Err(CorruptImage("image carries trailing bytes"));
        }

        let mut context = Context {
            arena,
            concepts,
            labels,
            sources,
            edges,
            attributions,
            concept_ids: HashMap::new(),
            label_ids: HashMap::new(),
            source_ids: HashMap::new(),
            edge_ids: HashMap::new(),
            concept_index: EntryIndex::default(),
            label_index: EntryIndex::default(),
        };
        context.rebuild_indexes()?;
        Ok(context)
    }

    /// Validates a freshly loaded image and rebuilds the derived indexes
    /// from the flat buffers. Called only by `from_bytes`, on a `Context`
    /// whose index maps are still empty.
    fn rebuild_indexes(&mut self) -> Result<(), CorruptImage> {
        // Strings: every name range must be a valid arena slice, and names
        // must be unique per namespace or lookups would be ambiguous.
        for (id, record) in self.concepts.iter().enumerate() {
            let name = checked_arena_str(&self.arena, record.name_offset, record.name_len)?;
            self.concept_index.push(name, id as u32);
            if self
                .concept_ids
                .insert(name.to_string(), id as u32)
                .is_some()
            {
                return Err(CorruptImage("two concept records share one name"));
            }
        }
        for (id, record) in self.labels.iter().enumerate() {
            let name = checked_arena_str(&self.arena, record.name_offset, record.name_len)?;
            self.label_index.push(name, id as u32);
            if self.label_ids.insert(name.to_string(), id as u32).is_some() {
                return Err(CorruptImage("two label records share one name"));
            }
        }
        for (id, record) in self.sources.iter().enumerate() {
            let name = checked_arena_str(&self.arena, record.name_offset, record.name_len)?;
            if self
                .source_ids
                .insert(name.to_string(), id as u32)
                .is_some()
            {
                return Err(CorruptImage("two source records share one name"));
            }
        }

        // Edges: endpoints must exist, and triples must be unique — the
        // accumulate-on-repeat contract depends on it.
        for (id, edge) in self.edges.iter().enumerate() {
            if edge.subject as usize >= self.concepts.len()
                || edge.object as usize >= self.concepts.len()
                || edge.label as usize >= self.labels.len()
            {
                return Err(CorruptImage("edge references an unknown concept or label"));
            }
            let key = (edge.subject, edge.label, edge.object);
            if self.edge_ids.insert(key, id as u32).is_some() {
                return Err(CorruptImage("two edge records share one triple"));
            }
        }

        // Chains: each must contain exactly its stored count of edges, all
        // owned by the anchoring record, ending at the stored tail, with
        // no cycles — and together the chains of a kind must cover every
        // edge, or some knowledge would be silently unreachable.
        for (id, record) in self.concepts.iter().enumerate() {
            let id = id as u32;
            validate_edge_chain(
                &self.edges,
                record.first_outgoing,
                record.last_outgoing,
                record.outgoing_count,
                |edge| edge.next_outgoing,
                |edge| edge.subject == id,
            )?;
            validate_edge_chain(
                &self.edges,
                record.first_incoming,
                record.last_incoming,
                record.incoming_count,
                |edge| edge.next_incoming,
                |edge| edge.object == id,
            )?;
        }
        for (id, record) in self.labels.iter().enumerate() {
            let id = id as u32;
            validate_edge_chain(
                &self.edges,
                record.first_edge,
                record.last_edge,
                record.edge_count,
                |edge| edge.next_labeled,
                |edge| edge.label == id,
            )?;
        }
        let edge_total = self.edges.len() as u64;
        let outgoing_total: u64 = self
            .concepts
            .iter()
            .map(|r| u64::from(r.outgoing_count))
            .sum();
        let incoming_total: u64 = self
            .concepts
            .iter()
            .map(|r| u64::from(r.incoming_count))
            .sum();
        let labeled_total: u64 = self.labels.iter().map(|r| u64::from(r.edge_count)).sum();
        if outgoing_total != edge_total
            || incoming_total != edge_total
            || labeled_total != edge_total
        {
            return Err(CorruptImage("edge chains do not cover the edge table"));
        }

        // Attribution chains: in range, sources known, acyclic, ending at
        // the stored tail, and disjoint across edges — a shared record
        // would let one edge's accumulation corrupt another's.
        let mut claimed = vec![false; self.attributions.len()];
        for edge in &self.edges {
            let mut cursor = edge.first_attribution;
            let mut tail = NIL;
            while cursor != NIL {
                let record = self
                    .attributions
                    .get(cursor as usize)
                    .ok_or(CorruptImage("attribution link is out of range"))?;
                if std::mem::replace(&mut claimed[cursor as usize], true) {
                    return Err(CorruptImage("attribution record belongs to two chains"));
                }
                if record.source as usize >= self.sources.len() {
                    return Err(CorruptImage("attribution references an unknown source"));
                }
                tail = cursor;
                cursor = record.next;
            }
            if tail != edge.last_attribution {
                return Err(CorruptImage("attribution chain does not end at its tail"));
            }
        }
        if claimed.iter().any(|&used| !used) {
            return Err(CorruptImage("attribution chains do not cover their table"));
        }
        Ok(())
    }
}

/// Reads one interned string out of an untrusted arena, validating the
/// range and UTF-8 — the load-time counterpart of [`Context::arena_str`].
fn checked_arena_str(arena: &[u8], offset: u32, len: u32) -> Result<&str, CorruptImage> {
    let start = offset as usize;
    let end = start
        .checked_add(len as usize)
        .ok_or(CorruptImage("name range overflows"))?;
    let bytes = arena
        .get(start..end)
        .ok_or(CorruptImage("name range escapes the arena"))?;
    std::str::from_utf8(bytes).map_err(|_| CorruptImage("name is not valid UTF-8"))
}

/// Checks that one linked chain of edges is exactly `count` records long,
/// stays in bounds, contains only edges anchored by its owner, ends at
/// `last`, and cannot cycle (a chain longer than the whole table must
/// repeat a record).
fn validate_edge_chain(
    edges: &[EdgeRecord],
    first: EdgeId,
    last: EdgeId,
    count: u32,
    follow: fn(&EdgeRecord) -> EdgeId,
    owned: impl Fn(&EdgeRecord) -> bool,
) -> Result<(), CorruptImage> {
    let mut cursor = first;
    let mut tail = NIL;
    let mut steps: usize = 0;
    while cursor != NIL {
        steps += 1;
        if steps > count as usize || steps > edges.len() {
            return Err(CorruptImage("edge chain overruns its stored count"));
        }
        let record = edges
            .get(cursor as usize)
            .ok_or(CorruptImage("edge chain link is out of range"))?;
        if !owned(record) {
            return Err(CorruptImage("edge chain contains another record's edge"));
        }
        tail = cursor;
        cursor = follow(record);
    }
    if steps != count as usize {
        return Err(CorruptImage("edge chain is shorter than its stored count"));
    }
    if tail != last {
        return Err(CorruptImage("edge chain does not end at its stored tail"));
    }
    Ok(())
}

/// One fixed-width table row: how many image bytes it spans and how to
/// store/load it, field by field in declaration order, little-endian.
trait Record: Sized {
    const SIZE: usize;
    fn store(&self, image: &mut Vec<u8>);
    fn load(reader: &mut Reader) -> Result<Self, CorruptImage>;
}

/// Writes one table as a u64 record count followed by its records.
fn store_table<T: Record>(records: &[T], image: &mut Vec<u8>) {
    image.extend_from_slice(&(records.len() as u64).to_le_bytes());
    for record in records {
        record.store(image);
    }
}

/// Reads one table written by [`store_table`], bounding the record count
/// by the bytes actually present so a hostile count cannot balloon memory.
fn load_table<T: Record>(reader: &mut Reader) -> Result<Vec<T>, CorruptImage> {
    let count = reader.read_u64()?;
    if count >= u64::from(NIL) {
        return Err(CorruptImage("table exceeds the u32 id space"));
    }
    let count = count as usize;
    let bytes_needed = count
        .checked_mul(T::SIZE)
        .ok_or(CorruptImage("table byte size overflows"))?;
    if bytes_needed > reader.remaining() {
        return Err(CorruptImage("table is truncated"));
    }
    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        records.push(T::load(reader)?);
    }
    Ok(records)
}

impl Record for ConceptRecord {
    const SIZE: usize = 32;

    fn store(&self, image: &mut Vec<u8>) {
        for field in [
            self.name_offset,
            self.name_len,
            self.first_outgoing,
            self.last_outgoing,
            self.outgoing_count,
            self.first_incoming,
            self.last_incoming,
            self.incoming_count,
        ] {
            image.extend_from_slice(&field.to_le_bytes());
        }
    }

    fn load(reader: &mut Reader) -> Result<Self, CorruptImage> {
        Ok(Self {
            name_offset: reader.read_u32()?,
            name_len: reader.read_u32()?,
            first_outgoing: reader.read_u32()?,
            last_outgoing: reader.read_u32()?,
            outgoing_count: reader.read_u32()?,
            first_incoming: reader.read_u32()?,
            last_incoming: reader.read_u32()?,
            incoming_count: reader.read_u32()?,
        })
    }
}

impl Record for LabelRecord {
    const SIZE: usize = 20;

    fn store(&self, image: &mut Vec<u8>) {
        for field in [
            self.name_offset,
            self.name_len,
            self.first_edge,
            self.last_edge,
            self.edge_count,
        ] {
            image.extend_from_slice(&field.to_le_bytes());
        }
    }

    fn load(reader: &mut Reader) -> Result<Self, CorruptImage> {
        Ok(Self {
            name_offset: reader.read_u32()?,
            name_len: reader.read_u32()?,
            first_edge: reader.read_u32()?,
            last_edge: reader.read_u32()?,
            edge_count: reader.read_u32()?,
        })
    }
}

impl Record for SourceRecord {
    const SIZE: usize = 8;

    fn store(&self, image: &mut Vec<u8>) {
        image.extend_from_slice(&self.name_offset.to_le_bytes());
        image.extend_from_slice(&self.name_len.to_le_bytes());
    }

    fn load(reader: &mut Reader) -> Result<Self, CorruptImage> {
        Ok(Self {
            name_offset: reader.read_u32()?,
            name_len: reader.read_u32()?,
        })
    }
}

impl Record for EdgeRecord {
    const SIZE: usize = 40;

    fn store(&self, image: &mut Vec<u8>) {
        for field in [
            self.subject,
            self.label,
            self.object,
            self.next_outgoing,
            self.next_incoming,
            self.next_labeled,
            self.first_attribution,
            self.last_attribution,
        ] {
            image.extend_from_slice(&field.to_le_bytes());
        }
        image.extend_from_slice(&self.weight.to_le_bytes());
    }

    fn load(reader: &mut Reader) -> Result<Self, CorruptImage> {
        Ok(Self {
            subject: reader.read_u32()?,
            label: reader.read_u32()?,
            object: reader.read_u32()?,
            next_outgoing: reader.read_u32()?,
            next_incoming: reader.read_u32()?,
            next_labeled: reader.read_u32()?,
            first_attribution: reader.read_u32()?,
            last_attribution: reader.read_u32()?,
            weight: reader.read_f64()?,
        })
    }
}

impl Record for AttributionRecord {
    const SIZE: usize = 16;

    fn store(&self, image: &mut Vec<u8>) {
        image.extend_from_slice(&self.source.to_le_bytes());
        image.extend_from_slice(&self.next.to_le_bytes());
        image.extend_from_slice(&self.weight.to_le_bytes());
    }

    fn load(reader: &mut Reader) -> Result<Self, CorruptImage> {
        Ok(Self {
            source: reader.read_u32()?,
            next: reader.read_u32()?,
            weight: reader.read_f64()?,
        })
    }
}

/// Cursor over an image's bytes; every read is bounds-checked so a
/// truncated or hostile image fails with [`CorruptImage`], never a panic.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, len: usize) -> Result<&'a [u8], CorruptImage> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or(CorruptImage("section length overflows"))?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or(CorruptImage("image ends mid-section"))?;
        self.pos = end;
        Ok(slice)
    }

    fn read_u32(&mut self) -> Result<u32, CorruptImage> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, CorruptImage> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64, CorruptImage> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
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
            .associate_from("決定", "手段", "投票", 1.0, "IPA公式")
            .unwrap();
        context
            .associate_from("決定", "手段", "投票", 1.0, "解説記事")
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
                },
                Attribution {
                    source: "解説記事".to_string(),
                    weight: 1.0,
                },
            ]
        );
    }

    #[test]
    fn attributions_tell_corroboration_apart_from_single_source_emphasis() {
        let mut context = Context::default();
        // Same total weight, different evidence story.
        context.associate_from("a", "r", "b", 1.0, "文書1").unwrap();
        context.associate_from("a", "r", "b", 1.0, "文書2").unwrap();
        context.associate_from("x", "r", "y", 2.0, "文書1").unwrap();

        let corroborated = &context.query(Some("a"), None, None)[0];
        let emphatic = &context.query(Some("x"), None, None)[0];
        assert_eq!(corroborated.weight, emphatic.weight);
        assert_eq!(corroborated.attributions.len(), 2);
        assert_eq!(emphatic.attributions.len(), 1);
    }

    #[test]
    fn repeated_assertions_from_one_source_accumulate_into_one_attribution() {
        let mut context = Context::default();
        context.associate_from("a", "r", "b", 1.0, "文書1").unwrap();
        context.associate_from("a", "r", "b", 0.5, "文書1").unwrap();

        let recalled = context.recall("a");
        assert_eq!(recalled[0].weight, 1.5);
        assert_eq!(
            recalled[0].attributions,
            vec![Attribution {
                source: "文書1".to_string(),
                weight: 1.5,
            }]
        );
    }

    #[test]
    fn unsourced_and_sourced_assertions_mix_on_one_association() {
        let mut context = Context::default();
        context.associate("a", "r", "b", 1.0).unwrap();
        context.associate_from("a", "r", "b", 0.5, "文書1").unwrap();

        // Total weight counts both; only the sourced part is attributed.
        let recalled = context.recall("a");
        assert_eq!(recalled[0].weight, 1.5);
        assert_eq!(
            recalled[0].attributions,
            vec![Attribution {
                source: "文書1".to_string(),
                weight: 0.5,
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
            .associate_from("決定", "手段", "投票", 1.0, "IPA公式")
            .unwrap();
        context
            .associate_from("決定", "手段", "投票", 0.5, "解説記事")
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
        context.associate_from("a", "r", "b", 1.0, "文書1").unwrap();

        let mut restored = Context::from_bytes(&context.to_bytes()).expect("image must load");

        // Accumulation must land on the restored edge and its restored
        // attribution — the rebuilt indexes and chain tails must all point
        // at the right records.
        restored
            .associate_from("a", "r", "b", 0.5, "文書1")
            .unwrap();
        assert_eq!(weight_between(&restored, "a", "r", "b"), 1.5);
        assert_eq!(
            restored.recall("a")[0].attributions,
            vec![Attribution {
                source: "文書1".to_string(),
                weight: 1.5,
            }]
        );

        // And chains must keep extending in insertion order.
        restored.associate("a", "r", "c", 1.0).unwrap();
        restored.associate("d", "r2", "a", 1.0).unwrap();
        assert_eq!(restored.recall("a").len(), 3);
        assert_eq!(restored.labels(), vec!["r", "r2"]);
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
            .associate_from("私", "好き", "りんご", 1.0, "文書1")
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
            context.associate_from("私", "好き", "りんご", 1.0, "文書1"),
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
            .associate_from("私", "食べられる", "りんご", -0.2, "文書1")
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
