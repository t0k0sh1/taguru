//! Resident BM25 index over paragraph postings — the lexical lane's
//! answer to "stop re-tokenizing every passage on every query". The
//! passage store stays the source of truth; this is a derived
//! structure, rebuilt from it whenever it is missing, so losing it is
//! a rebuild cost and never an outage.
//!
//! Updates are incremental at source granularity (the store's own
//! replacement unit): re-storing or retracting a source tombstones its
//! old paragraphs in O(paragraphs touched) instead of walking every
//! posting list — common bigrams appear in a large fraction of all
//! paragraphs, so eager physical deletion would make one retraction
//! cost O(total postings). Tombstones are reclaimed by an in-place
//! rebuild once they outnumber a quarter of the live paragraphs
//! (amortized O(1) per mutation, same argument as `Vec` doubling).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;
use std::sync::Arc;

use crate::hash::{FNV1A_OFFSET, fnv1a_fold};
use crate::passages::PassageRecord;
use crate::registry::passage_terms;

// B1 → B2: slots grew a question hash (doc2query questions index into
// this lane; the digest must notice a question-only change). An old
// sidecar fails the magic and rebuilds — a cost, never an outage.
const INDEX_MAGIC: &[u8; 8] = b"TAGURUB2";

/// BM25 constants, shared with nothing: the paragraph is the document.
const K1: f32 = 1.2;
const B: f32 = 0.75;

/// Tombstones are reclaimed past max(this floor, live / 4).
const COMPACT_DEAD_FLOOR: u32 = 1024;

/// Pass-through hasher for keys that are ALREADY hashes (the u64 FNV
/// terms `passage_terms` emits). Hashing a hash through SipHash would
/// spend most of a lookup's time re-mixing perfectly good entropy.
/// Only `write_u64` is meaningful; any other key type is a bug.
#[derive(Default)]
pub(crate) struct TermHasher(u64);

impl Hasher for TermHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _: &[u8]) {
        unreachable!("TermHasher only hashes u64 term keys");
    }

    fn write_u64(&mut self, key: u64) {
        self.0 = key;
    }
}

type TermMap<V> = HashMap<u64, V, BuildHasherDefault<TermHasher>>;

/// One indexed paragraph. `alive: false` is the tombstone — the slot
/// stays so postings need no eager rewrite.
struct Slot {
    source_id: u32,
    index: u32,
    length: f32,
    /// The paragraph TEXT hash — what search hands back for the
    /// staleness check against the store's current record. Questions
    /// deliberately stay out: they affect scoring, not which text the
    /// hit points at.
    hash: u64,
    /// Fold of the paragraph's attached doc2query questions — indexed
    /// content beyond the text, so the drift digest must carry it: a
    /// question-only change re-upserts the source at load instead of
    /// serving the old scoring forever.
    question_hash: u64,
    alive: bool,
}

struct Posting {
    slot: u32,
    tf: f32,
}

pub(crate) struct Bm25Index {
    /// Interned source names; a slot carries the id, search hands the
    /// name back out.
    sources: Vec<String>,
    source_ids: HashMap<String, u32>,
    /// Which slots belong to each source — the wholesale-replacement
    /// unit, so upsert/remove touch exactly these.
    by_source: HashMap<u32, Vec<u32>>,
    slots: Vec<Slot>,
    postings: TermMap<Vec<Posting>>,
    live_count: u32,
    /// f64 on purpose: an incrementally maintained f32 sum would drift
    /// as paragraphs come and go.
    live_total_length: f64,
    dead_count: u32,
}

/// One search hit: (source name, paragraph index, paragraph hash,
/// score). The hash lets the caller drop hits whose paragraph changed
/// between the index's view and the store's current record.
pub(crate) type IndexHit = (String, u32, u64, f32);

/// One query term's evidence against one paragraph, in query-gram
/// order (the position IS the term key — the caller holds the gram
/// list it asked about, spellings included). `carriers` (the df) and
/// `idf` are corpus-wide, `tf` and the `contribution` are the
/// paragraph's own. A term the paragraph lacks reports tf 0 and
/// contribution 0.0; a term nothing carries reports carriers 0:
/// "matched only two high-df bigrams contributing ~0" must be visible,
/// not inferred.
#[cfg_attr(test, derive(Debug))]
pub(crate) struct TermEvidence {
    pub(crate) tf: f32,
    pub(crate) carriers: u32,
    pub(crate) idf: f32,
    pub(crate) contribution: f32,
}

/// The index's whole account of one paragraph against one query:
/// per-term evidence plus its sum, which is bit-for-bit the score
/// [`Bm25Index::search`] gives the same paragraph — both sides add the
/// same [`contribution`] values in query-gram order.
#[cfg_attr(test, derive(Debug))]
pub(crate) struct IndexEvidence {
    /// The indexed TEXT hash — the caller's staleness check against
    /// the store's current record, exactly like a search hit's.
    pub(crate) hash: u64,
    pub(crate) score: f32,
    pub(crate) terms: Vec<TermEvidence>,
}

/// The two factors of one term's BM25 addend, factored out of the
/// search loop so [`Bm25Index::explain`] cannot drift from it — the
/// expression shapes are the loop's originals, so explain reports the
/// very numbers search summed.
fn idf(live_total: f32, carriers: f32) -> f32 {
    (1.0 + (live_total - carriers + 0.5) / (carriers + 0.5)).ln()
}

fn contribution(idf: f32, tf: f32, length: f32, average_length: f32) -> f32 {
    idf * (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * length / average_length))
}

impl Bm25Index {
    pub(crate) fn empty() -> Self {
        Self {
            sources: Vec::new(),
            source_ids: HashMap::new(),
            by_source: HashMap::new(),
            slots: Vec::new(),
            postings: TermMap::default(),
            live_count: 0,
            live_total_length: 0.0,
            dead_count: 0,
        }
    }

    /// Builds from a store snapshot — the cold path (first search of a
    /// residency), and the body of tombstone reclamation.
    pub(crate) fn build(records: &[(String, Arc<PassageRecord>)]) -> Self {
        let mut index = Self::empty();
        for (source, record) in records {
            index.upsert_source(source, record);
        }
        index
    }

    /// Replaces one source's paragraphs with `record`'s — tombstone the
    /// old, append the new. Cost is proportional to the paragraphs
    /// touched, never to the posting lists they sit in.
    ///
    /// A paragraph's attached doc2query questions index INTO it, terms
    /// and length both — the doc2query move itself (append the
    /// generated queries to the document before indexing), and the
    /// lexical mirror of the vector lane's question rows, so a
    /// question-shaped query lands on its answer-shaped paragraph even
    /// on a deployment with no embedding provider at all.
    ///
    /// Returns whether this call actually changed the index — a real
    /// tombstone, a paragraph appended, or both. `false` only for the
    /// otherwise-inert case CodeRabbit caught on #574 (issue #563 item
    /// 2's own review): an empty or whitespace-only `record` upserted
    /// for a source with nothing live to tombstone either — `intern`
    /// still registers the source name, but nothing search-observable
    /// moved, so callers deciding whether to mark the index dirty must
    /// not read this as a change.
    pub(crate) fn upsert_source(&mut self, source: &str, record: &PassageRecord) -> bool {
        let source_id = self.intern(source);
        let mut changed = self.tombstone(source_id);
        let slot_list = self.by_source.entry(source_id).or_default();
        // The record's questions are sorted by paragraph, so one cursor
        // walks them in lockstep with the paragraphs — O(paragraphs +
        // questions), and the terms and the question hash come out of
        // the same pass.
        let mut questions = record.questions.iter().peekable();
        for (span, text) in record.paragraph_texts() {
            changed = true;
            let slot = self.slots.len() as u32;
            let mut frequencies: HashMap<u64, f32> = HashMap::new();
            let mut length = 0f32;
            let mut count = |gram: u64| {
                *frequencies.entry(gram).or_insert(0.0) += 1.0;
                length += 1.0;
            };
            for gram in passage_terms(text) {
                count(gram);
            }
            let question_hash = take_questions_fold(&mut questions, span.index, |question| {
                for gram in passage_terms(question) {
                    count(gram);
                }
            });
            self.slots.push(Slot {
                source_id,
                index: span.index,
                length,
                hash: span.hash,
                question_hash,
                alive: true,
            });
            slot_list.push(slot);
            for (gram, tf) in frequencies {
                self.postings
                    .entry(gram)
                    .or_default()
                    .push(Posting { slot, tf });
            }
            self.live_count += 1;
            self.live_total_length += f64::from(length);
        }
        self.reclaim_if_due();
        changed
    }

    /// Tombstones one source's paragraphs (a retraction). Returns
    /// whether any slot was actually live to tombstone — false for a
    /// never-interned source or one already fully dead, which callers
    /// (`AppState::refresh_bm25`, issue #563 item 2) need to tell apart
    /// from a real change: retracting a source this index never held
    /// anything for must not mark the index dirty.
    pub(crate) fn remove_source(&mut self, source: &str) -> bool {
        let Some(&source_id) = self.source_ids.get(source) else {
            return false;
        };
        let changed = self.tombstone(source_id);
        self.reclaim_if_due();
        changed
    }

    /// Returns whether any slot flipped from alive to dead.
    fn tombstone(&mut self, source_id: u32) -> bool {
        let mut changed = false;
        if let Some(slot_list) = self.by_source.get_mut(&source_id) {
            for &slot in slot_list.iter() {
                let slot = &mut self.slots[slot as usize];
                if slot.alive {
                    slot.alive = false;
                    self.live_count -= 1;
                    self.live_total_length -= f64::from(slot.length);
                    self.dead_count += 1;
                    changed = true;
                }
            }
            slot_list.clear();
        }
        changed
    }

    /// In-place tombstone reclamation: rebuild the whole structure from
    /// the live slots. Postings carry only (slot, tf), so this needs
    /// the paragraphs' terms again — which the slots do not keep — so
    /// reclamation is driven from OUTSIDE with the store's records (see
    /// `needs_reclaim`); here we only report the need.
    fn reclaim_if_due(&mut self) {
        // Bookkeeping hook: the actual rebuild happens in the registry
        // (it holds the records). Nothing to do eagerly.
    }

    /// Whether tombstones have outgrown their allowance and the caller
    /// should rebuild from the store's current snapshot.
    pub(crate) fn needs_reclaim(&self) -> bool {
        self.dead_count > COMPACT_DEAD_FLOOR.max(self.live_count / 4)
    }

    /// Top `limit` live paragraphs by BM25, ties broken by (source
    /// name, paragraph index) for deterministic output. `query_grams`
    /// must already be deduplicated.
    ///
    /// `eligible` is the pre-lane source filter (#167): only slots
    /// whose source is in the set are scored and served. The corpus
    /// statistics — idf's carrier counts and the average length —
    /// deliberately stay CORPUS-GLOBAL: the filter gates which
    /// paragraphs may answer, it does not re-weight the collection
    /// they are scored against (an eligible paragraph scores exactly
    /// what it scores unfiltered, so explain's evidence stays true
    /// under any filter).
    ///
    /// A returned score can be exactly 0.0: at corpus sizes around
    /// 2^23 live paragraphs sharing a query gram, f32 `idf` underflows
    /// to 0.0, and a matching paragraph is still a match. Such hits
    /// tie-break by name and index like any other; a touched slot is
    /// never dropped for scoring 0.0 — an empty result still means
    /// there is no eligible, in-limit candidate to return (nothing
    /// touched, `limit` is 0, or every touched slot's source was
    /// filtered out by `eligible`), never that everything touched
    /// scored 0.0.
    pub(crate) fn search(
        &self,
        query_grams: &[u64],
        limit: usize,
        eligible: Option<&std::collections::BTreeSet<String>>,
    ) -> Vec<IndexHit> {
        if self.live_count == 0 || query_grams.is_empty() {
            return Vec::new();
        }
        // One membership bit per interned source, so the posting loop
        // pays an index, never a string hash.
        let allowed: Option<Vec<bool>> = eligible.map(|set| {
            self.sources
                .iter()
                .map(|source| set.contains(source))
                .collect()
        });
        let total = self.live_count as f32;
        let average_length = (self.live_total_length / f64::from(self.live_count)).max(1.0) as f32;

        // Keyed by touched slot only, so cost tracks how many paragraphs
        // the query actually matches rather than the index's total slot
        // count (which includes tombstones awaiting reclamation). `entry`
        // establishes "first hit" by key presence, so — unlike a flat
        // `scores` array — this stays correct even when a `contribution`
        // underflows to exactly 0.0 (a gram carried by nearly every live
        // paragraph): the slot is never double-added on a later gram,
        // AND every key this loop inserts is a real match, score 0.0
        // included — nothing downstream may drop a key for being 0.0,
        // or a query whose only matching grams all underflow gets read
        // as "no match" instead of "matched, contributed nothing".
        let mut scores: HashMap<u32, f32> = HashMap::new();
        // Reused across grams: each gram's alive postings, so idf's
        // carrier count and the scoring pass both walk them without a
        // second `.alive` lookup per posting.
        let mut alive_postings: Vec<&Posting> = Vec::new();
        for gram in query_grams {
            let Some(postings) = self.postings.get(gram) else {
                continue;
            };
            alive_postings.clear();
            alive_postings.extend(
                postings
                    .iter()
                    .filter(|posting| self.slots[posting.slot as usize].alive),
            );
            if alive_postings.is_empty() {
                continue;
            }
            let carriers = alive_postings.len() as f32;
            let idf = idf(total, carriers);
            for &posting in &alive_postings {
                let slot = &self.slots[posting.slot as usize];
                if let Some(allowed) = &allowed
                    && !allowed[slot.source_id as usize]
                {
                    continue;
                }
                *scores.entry(posting.slot).or_insert(0.0) +=
                    contribution(idf, posting.tf, slot.length, average_length);
            }
        }

        // No `score > 0.0` filter: every key in `scores` is a slot the
        // query actually touched (a live posting on at least one
        // query gram), so it belongs in the results regardless of
        // what it scored. At corpus sizes around 2^23 live paragraphs
        // sharing a gram, `idf` (and so `contribution`) rounds to
        // exactly 0.0 in f32 — dropping those slots here would read a
        // real, ubiquitous-term match as "no match", indistinguishable
        // from a query that touched nothing at all, and would
        // disagree with `explain`, which reports the same paragraph
        // at the same score without ever hiding it.
        let mut hits: Vec<IndexHit> = scores
            .into_iter()
            .map(|(slot_id, score)| {
                let slot = &self.slots[slot_id as usize];
                (
                    self.sources[slot.source_id as usize].clone(),
                    slot.index,
                    slot.hash,
                    score,
                )
            })
            .collect();
        hits.sort_by(|a, b| {
            b.3.total_cmp(&a.3)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.cmp(&b.1))
        });
        hits.truncate(limit);
        hits
    }

    /// One live paragraph's evidence against `query_grams` — what
    /// [`Bm25Index::search`] computed about it and threw away with the
    /// truncation, plus the per-term breakdown search never
    /// materializes. `None` when the index holds no live slot for the
    /// (source, paragraph) pair. `query_grams` deduplicated, as for
    /// `search`.
    pub(crate) fn explain(
        &self,
        query_grams: &[u64],
        source: &str,
        index: u32,
    ) -> Option<IndexEvidence> {
        let source_id = *self.source_ids.get(source)?;
        let slot_id = *self.by_source.get(&source_id)?.iter().find(|&&slot| {
            let slot = &self.slots[slot as usize];
            slot.alive && slot.index == index
        })?;
        let slot = &self.slots[slot_id as usize];
        let total = self.live_count as f32;
        let average_length = (self.live_total_length / f64::from(self.live_count)).max(1.0) as f32;

        let mut terms = Vec::with_capacity(query_grams.len());
        let mut score = 0f32;
        for &gram in query_grams {
            let (tf, carriers) = match self.postings.get(&gram) {
                Some(postings) => (
                    postings
                        .iter()
                        .find(|posting| posting.slot == slot_id)
                        .map_or(0.0, |posting| posting.tf),
                    postings
                        .iter()
                        .filter(|posting| self.slots[posting.slot as usize].alive)
                        .count() as u32,
                ),
                None => (0.0, 0),
            };
            // carriers 0 is "the corpus has never seen this term" — an
            // idf for it would be an answer to a question nobody asked.
            let idf = if carriers == 0 {
                0.0
            } else {
                idf(total, carriers as f32)
            };
            let addend = if tf > 0.0 {
                contribution(idf, tf, slot.length, average_length)
            } else {
                0.0
            };
            score += addend;
            terms.push(TermEvidence {
                tf,
                carriers,
                idf,
                contribution: addend,
            });
        }
        Some(IndexEvidence {
            hash: slot.hash,
            score,
            terms,
        })
    }

    /// Per-source digest over (paragraph index, paragraph hash,
    /// question fold) of the LIVE slots, in index order — the load-time
    /// drift detector: a source whose digest disagrees with the passage
    /// store's current record gets re-upserted instead of costing a
    /// full rebuild.
    pub(crate) fn source_digests(&self) -> HashMap<String, u64> {
        let mut digests = HashMap::new();
        for (&source_id, slot_list) in &self.by_source {
            let mut digest = FNV1A_OFFSET;
            let mut any = false;
            for &slot in slot_list {
                let slot = &self.slots[slot as usize];
                if slot.alive {
                    digest = digest_fold(digest, slot.index, slot.hash, slot.question_hash);
                    any = true;
                }
            }
            if any {
                digests.insert(self.sources[source_id as usize].clone(), digest);
            }
        }
        digests
    }

    /// Reads the sidecar, `None` on any problem — a corrupt or missing
    /// index costs a rebuild, never an outage. A read failure other
    /// than "not written yet" (permission, I/O) is warned by
    /// [`crate::storage::read_sidecar`] — otherwise a permanently
    /// unreadable sidecar pays a full re-tokenization every residency
    /// with nothing in the logs to say why.
    pub(crate) fn load(path: &Path) -> Option<Self> {
        let bytes = crate::storage::read_sidecar(path, "BM25 index")?;
        let parsed = Self::from_bytes(&bytes);
        if parsed.is_none() {
            tracing::warn!("ignoring corrupt BM25 index at {}", path.display());
        }
        parsed
    }

    /// Serializes the LIVE slots in canonical order (sources sorted,
    /// slots by (source, index), terms sorted, postings slot-ascending)
    /// — byte-stable for identical content, and saving IS a compaction:
    /// tombstones never reach the disk.
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        // Canonical live view: sorted source names, renumbered ids.
        let mut live_sources: Vec<&String> = self
            .by_source
            .iter()
            .filter(|(_, slots)| slots.iter().any(|&slot| self.slots[slot as usize].alive))
            .map(|(&id, _)| &self.sources[id as usize])
            .collect();
        live_sources.sort();
        let new_source_id: HashMap<&String, u32> = live_sources
            .iter()
            .enumerate()
            .map(|(id, name)| (*name, id as u32))
            .collect();

        // Live slots in canonical order, and old-slot → new-slot map.
        let mut order: Vec<u32> = (0..self.slots.len() as u32)
            .filter(|&slot| self.slots[slot as usize].alive)
            .collect();
        order.sort_by_key(|&slot| {
            let slot = &self.slots[slot as usize];
            (&self.sources[slot.source_id as usize], slot.index)
        });
        let new_slot: HashMap<u32, u32> = order
            .iter()
            .enumerate()
            .map(|(new, &old)| (old, new as u32))
            .collect();

        let mut out = Vec::new();
        out.extend_from_slice(INDEX_MAGIC);
        out.extend_from_slice(&(live_sources.len() as u32).to_le_bytes());
        for source in &live_sources {
            out.extend_from_slice(&(source.len() as u32).to_le_bytes());
            out.extend_from_slice(source.as_bytes());
        }
        out.extend_from_slice(&(order.len() as u32).to_le_bytes());
        for &old in &order {
            let slot = &self.slots[old as usize];
            out.extend_from_slice(
                &new_source_id[&self.sources[slot.source_id as usize]].to_le_bytes(),
            );
            out.extend_from_slice(&slot.index.to_le_bytes());
            out.extend_from_slice(&slot.length.to_le_bytes());
            out.extend_from_slice(&slot.hash.to_le_bytes());
            out.extend_from_slice(&slot.question_hash.to_le_bytes());
        }
        let mut terms: Vec<u64> = self
            .postings
            .iter()
            .filter(|(_, list)| {
                list.iter()
                    .any(|posting| self.slots[posting.slot as usize].alive)
            })
            .map(|(&term, _)| term)
            .collect();
        terms.sort_unstable();
        out.extend_from_slice(&(terms.len() as u32).to_le_bytes());
        for term in terms {
            let list = &self.postings[&term];
            let mut live: Vec<(u32, f32)> = list
                .iter()
                .filter(|posting| self.slots[posting.slot as usize].alive)
                .map(|posting| (new_slot[&posting.slot], posting.tf))
                .collect();
            live.sort_by_key(|&(slot, _)| slot);
            out.extend_from_slice(&term.to_le_bytes());
            out.extend_from_slice(&(live.len() as u32).to_le_bytes());
            for (slot, tf) in live {
                out.extend_from_slice(&slot.to_le_bytes());
                out.extend_from_slice(&tf.to_le_bytes());
            }
        }
        out
    }

    /// Rejects a well-formed-looking but structurally invalid image
    /// (bounds are fine, but the content violates the invariants
    /// [`Self::to_bytes`] always writes) exactly like the corruption
    /// checks above: refuse and let the caller rebuild. There is no
    /// CRC on these sidecars (see the comment on the `length` check
    /// below), so this is the only line between a torn write and a
    /// silently wrong corpus for the shapes bounds-checking alone
    /// cannot see:
    /// - a source name repeated (`to_bytes` writes each live source
    ///   once, sorted and deduplicated by construction),
    /// - a live (source, paragraph) pair repeated — `search`'s
    ///   tie-break key `(score, source, index)` stops being unique,
    ///   and the final order falls back to `HashMap` iteration order
    ///   (non-deterministic ranking),
    /// - a term repeated across postings blocks,
    /// - a posting list not slot-ascending (`to_bytes` always sorts a
    ///   term's live postings by slot) — this also rejects the same
    ///   slot posted twice for one term, which would make `carriers`
    ///   exceed `live_count` and `idf` go negative (the only path to a
    ///   negative idf in this design): `search` sums both postings and
    ///   double-counts the contribution, but `explain`'s `.find()`
    ///   only sees the first, breaking the "bit-for-bit the same
    ///   score" contract between them.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut pos = 0usize;
        if bytes.get(..8)? != INDEX_MAGIC {
            return None;
        }
        pos += 8;
        let source_count = read_u32(bytes, &mut pos)? as usize;
        let mut index = Self::empty();
        for _ in 0..source_count {
            let len = read_u32(bytes, &mut pos)? as usize;
            let slice = bytes.get(pos..pos.checked_add(len)?)?;
            pos += len;
            let name = std::str::from_utf8(slice).ok()?;
            if index.source_ids.contains_key(name) {
                return None;
            }
            // intern() assigns ids in insertion order = file order.
            index.intern(name);
        }
        let slot_count = read_u32(bytes, &mut pos)? as usize;
        // Capped exactly like `posting_count` below: an untrusted
        // `slot_count` must not turn a corrupt sidecar into an
        // oversized up-front allocation before the loop even gets a
        // chance to reject it.
        let mut seen_slots: std::collections::HashSet<(u32, u32)> =
            std::collections::HashSet::with_capacity(slot_count.min(1 << 20));
        for _ in 0..slot_count {
            let source_id = read_u32(bytes, &mut pos)?;
            if source_id as usize >= index.sources.len() {
                return None;
            }
            let paragraph = read_u32(bytes, &mut pos)?;
            if !seen_slots.insert((source_id, paragraph)) {
                return None;
            }
            let length = f32::from_le_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?);
            pos += 4;
            // A non-finite or negative length is corruption: added into
            // `live_total_length` it poisons the average that every
            // document's BM25 length normalization divides by, turning
            // every score NaN. Refuse the whole index so the caller
            // rebuilds it, exactly as the magic and bounds checks do —
            // the sidecars carry no CRC, so this content check is the
            // only line between a torn write and a silently wrong corpus.
            if !length.is_finite() || length < 0.0 {
                return None;
            }
            let hash = u64::from_le_bytes(bytes.get(pos..pos + 8)?.try_into().ok()?);
            pos += 8;
            let question_hash = u64::from_le_bytes(bytes.get(pos..pos + 8)?.try_into().ok()?);
            pos += 8;
            let slot = index.slots.len() as u32;
            index.slots.push(Slot {
                source_id,
                index: paragraph,
                length,
                hash,
                question_hash,
                alive: true,
            });
            index.by_source.entry(source_id).or_default().push(slot);
            index.live_count += 1;
            index.live_total_length += f64::from(length);
        }
        let term_count = read_u32(bytes, &mut pos)? as usize;
        for _ in 0..term_count {
            let term = u64::from_le_bytes(bytes.get(pos..pos + 8)?.try_into().ok()?);
            pos += 8;
            let posting_count = read_u32(bytes, &mut pos)? as usize;
            let mut list = Vec::with_capacity(posting_count.min(1 << 20));
            // `to_bytes` always writes one term's postings slot-ascending
            // — enforcing strict monotonicity here rejects an
            // out-of-order list AND a slot posted twice for this term in
            // one check (a repeat can only fail the `>` test).
            let mut prev_slot: Option<u32> = None;
            for _ in 0..posting_count {
                let slot = read_u32(bytes, &mut pos)?;
                if slot as usize >= index.slots.len() {
                    return None;
                }
                if prev_slot.is_some_and(|prev| slot <= prev) {
                    return None;
                }
                prev_slot = Some(slot);
                let tf = f32::from_le_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?);
                pos += 4;
                // Same reasoning as `length` above: a non-finite or
                // non-positive term frequency flows straight into the
                // BM25 numerator. `to_bytes` only ever writes a posting
                // for a gram the paragraph actually carries (`count`
                // starts every gram's frequency at 1.0), so a genuine
                // tf is always positive — a 0.0 posting is a term
                // "occurring" zero times, a contradiction no encoder
                // emits, and (since #603 dropped `search`'s score>0.0
                // filter) would otherwise surface as a spurious hit for
                // a term the paragraph doesn't actually carry. Reject
                // the index and rebuild rather than trust an
                // unchecksummed sidecar's word.
                if !tf.is_finite() || tf <= 0.0 {
                    return None;
                }
                list.push(Posting { slot, tf });
            }
            if index.postings.insert(term, list).is_some() {
                return None;
            }
        }
        (pos == bytes.len()).then_some(index)
    }

    /// Rough resident bytes, for the cache budget and the gauges.
    pub(crate) fn footprint(&self) -> usize {
        const POSTING: usize = std::mem::size_of::<Posting>();
        const SLOT: usize = std::mem::size_of::<Slot>();
        let names: usize = self.sources.iter().map(|s| s.len() * 2 + 64).sum();
        let posting_lists: usize = self
            .postings
            .values()
            .map(|list| 8 + 24 + list.len() * POSTING)
            .sum();
        names + self.slots.len() * SLOT + posting_lists + self.by_source.len() * 40
    }

    fn intern(&mut self, source: &str) -> u32 {
        if let Some(&id) = self.source_ids.get(source) {
            return id;
        }
        let id = self.sources.len() as u32;
        self.sources.push(source.to_string());
        self.source_ids.insert(source.to_string(), id);
        id
    }
}

/// One (paragraph index, paragraph hash, question fold) step of a
/// source digest — FNV-1a-shaped so the fold depends on order and
/// content both.
fn digest_fold(digest: u64, index: u32, hash: u64, question_hash: u64) -> u64 {
    fnv1a_fold(
        digest,
        index
            .to_le_bytes()
            .into_iter()
            .chain(hash.to_le_bytes())
            .chain(question_hash.to_le_bytes()),
    )
}

/// Consumes the questions attached to `paragraph` off the
/// record-ordered cursor, hands each to `each`, and returns their fold
/// (a separator byte after every question so adjacent ones cannot
/// blend) — the questions' share of the drift digest, computed the
/// same way on both sides of the comparison.
fn take_questions_fold<'r>(
    questions: &mut std::iter::Peekable<std::slice::Iter<'r, (u32, String)>>,
    paragraph: u32,
    mut each: impl FnMut(&'r str),
) -> u64 {
    let mut digest = FNV1A_OFFSET;
    while let Some((_, question)) = questions.next_if(|&&(index, _)| index == paragraph) {
        each(question);
        digest = fnv1a_fold(digest, question.bytes().chain([0xff]));
    }
    digest
}

/// The passage store's side of the drift comparison: the digest the
/// index WOULD have for this record if it were fresh. The same
/// lockstep cursor as `upsert_source`, so both sides cost
/// O(paragraphs + questions).
///
/// `None` for a record with zero paragraphs (an empty or
/// whitespace-only passage, which `paragraph::split` turns into zero
/// spans) — matching [`Bm25Index::source_digests`]'s own rule that a
/// source with no LIVE slot has no entry at all. Returning
/// `Some(FNV1A_OFFSET)` here instead would disagree with that absence
/// on every comparison: `bm25_index`'s repair loop would read the
/// source as permanently drifted (`upsert_source` on an empty record
/// changes nothing, so the sidecar never catches up) and mark the
/// index dirty every residency for a source that never had anything
/// to be dirty about.
pub(crate) fn record_digest(record: &PassageRecord) -> Option<u64> {
    if record.paragraphs.is_empty() {
        return None;
    }
    let mut questions = record.questions.iter().peekable();
    Some(record.paragraphs.iter().fold(FNV1A_OFFSET, |digest, span| {
        let question_hash = take_questions_fold(&mut questions, span.index, |_| {});
        digest_fold(digest, span.index, span.hash, question_hash)
    }))
}

fn read_u32(bytes: &[u8], pos: &mut usize) -> Option<u32> {
    let slice = bytes.get(*pos..*pos + 4)?;
    *pos += 4;
    Some(u32::from_le_bytes(slice.try_into().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(text: &str) -> Arc<PassageRecord> {
        PassageRecord::for_tests(text)
    }

    fn corpus() -> Vec<(String, Arc<PassageRecord>)> {
        vec![
            (
                "docs/aomine.md".to_string(),
                record(
                    "青嶺酒造は雲居県霧沢町の蔵元である。\n\n原料米には山田錦を使い、\
                     精米歩合は50パーセントまで磨く。\n\n蔵開きの祭りでは新酒がふるまわれる。",
                ),
            ),
            (
                "docs/takase.md".to_string(),
                record("杜氏の高瀬は南部杜氏の出身で、経験は30年を超える。"),
            ),
            (
                "docs/code.md".to_string(),
                record("impl AppState { pub fn boot_with(dir: PathBuf) -> Self { todo!() } }"),
            ),
        ]
    }

    /// The naive scorer the index replaced, verbatim at paragraph
    /// granularity — the oracle for the equivalence test.
    fn full_rescan(
        records: &[(String, Arc<PassageRecord>)],
        query_grams: &[u64],
        limit: usize,
    ) -> Vec<(String, u32, f32)> {
        let paragraphs: Vec<(&String, u32, HashMap<u64, f32>, f32)> = records
            .iter()
            .flat_map(|(source, record)| {
                record.paragraph_texts().map(move |(span, text)| {
                    let mut frequencies: HashMap<u64, f32> = HashMap::new();
                    let mut length = 0f32;
                    for gram in passage_terms(text) {
                        *frequencies.entry(gram).or_insert(0.0) += 1.0;
                        length += 1.0;
                    }
                    (source, span.index, frequencies, length)
                })
            })
            .collect();
        if paragraphs.is_empty() {
            return Vec::new();
        }
        let total = paragraphs.len() as f32;
        let average_length =
            (paragraphs.iter().map(|(.., length)| *length).sum::<f32>() / total).max(1.0);
        let mut scored: Vec<(String, u32, f32)> = paragraphs
            .iter()
            .map(|(source, index, frequencies, length)| {
                let mut score = 0f32;
                for gram in query_grams {
                    let Some(&frequency) = frequencies.get(gram) else {
                        continue;
                    };
                    let carriers = paragraphs
                        .iter()
                        .filter(|(_, _, f, _)| f.contains_key(gram))
                        .count() as f32;
                    let idf = (1.0 + (total - carriers + 0.5) / (carriers + 0.5)).ln();
                    score += idf * (frequency * (K1 + 1.0))
                        / (frequency + K1 * (1.0 - B + B * length / average_length));
                }
                ((*source).clone(), *index, score)
            })
            .filter(|&(_, _, score)| score > 0.0)
            .collect();
        scored.sort_by(|a, b| {
            b.2.total_cmp(&a.2)
                .then_with(|| a.0.cmp(&b.0))
                .then_with(|| a.1.cmp(&b.1))
        });
        scored.truncate(limit);
        scored
    }

    fn grams(query: &str) -> Vec<u64> {
        let mut seen = std::collections::HashSet::new();
        passage_terms(query)
            .into_iter()
            .filter(|gram| seen.insert(*gram))
            .collect()
    }

    /// Hand-builds a `from_bytes` image in [`Bm25Index::to_bytes`]'s
    /// exact wire format, for constructing the structurally invalid
    /// shapes bounds-checking alone lets through (#600 item 2) —
    /// bytes no real `to_bytes` call would ever produce, but a torn
    /// write could, since the sidecar carries no CRC.
    fn image(
        sources: &[&str],
        // (source_id, paragraph, length, hash, question_hash)
        slots: &[(u32, u32, f32, u64, u64)],
        // (term, [(slot, tf)])
        terms: &[(u64, &[(u32, f32)])],
    ) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(INDEX_MAGIC);
        out.extend_from_slice(&(sources.len() as u32).to_le_bytes());
        for source in sources {
            out.extend_from_slice(&(source.len() as u32).to_le_bytes());
            out.extend_from_slice(source.as_bytes());
        }
        out.extend_from_slice(&(slots.len() as u32).to_le_bytes());
        for &(source_id, paragraph, length, hash, question_hash) in slots {
            out.extend_from_slice(&source_id.to_le_bytes());
            out.extend_from_slice(&paragraph.to_le_bytes());
            out.extend_from_slice(&length.to_le_bytes());
            out.extend_from_slice(&hash.to_le_bytes());
            out.extend_from_slice(&question_hash.to_le_bytes());
        }
        out.extend_from_slice(&(terms.len() as u32).to_le_bytes());
        for &(term, postings) in terms {
            out.extend_from_slice(&term.to_le_bytes());
            out.extend_from_slice(&(postings.len() as u32).to_le_bytes());
            for &(slot, tf) in postings {
                out.extend_from_slice(&slot.to_le_bytes());
                out.extend_from_slice(&tf.to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn from_bytes_accepts_a_canonical_hand_built_image() {
        // The positive control for the rejection tests below: if this
        // ever fails to decode, `image` itself drifted from
        // `to_bytes`'s format and every rejection test below is
        // meaningless.
        let bytes = image(
            &["a.md", "b.md"],
            &[(0, 0, 2.0, 1, 0), (1, 0, 3.0, 2, 0)],
            &[(10, &[(0, 1.0), (1, 1.0)]), (20, &[(1, 2.0)])],
        );
        let index =
            Bm25Index::from_bytes(&bytes).expect("a well-formed canonical image must decode");
        assert_eq!(index.live_count, 2);
        assert_eq!(index.sources, vec!["a.md".to_string(), "b.md".to_string()]);
    }

    #[test]
    fn from_bytes_rejects_a_source_name_repeated() {
        // `to_bytes` writes each live source once (sorted, deduplicated
        // by construction) — a repeat is a shape it never produces.
        let bytes = image(&["a.md", "a.md"], &[], &[]);
        assert!(Bm25Index::from_bytes(&bytes).is_none());
    }

    #[test]
    fn from_bytes_rejects_a_live_paragraph_repeated_for_one_source() {
        // Two slots both claiming (source 0, paragraph 0): `search`'s
        // tie-break key (score, source, index) stops being unique, so
        // the final order falls back to `HashMap` iteration order —
        // non-deterministic ranking.
        let bytes = image(&["a.md"], &[(0, 0, 1.0, 1, 0), (0, 0, 1.0, 2, 0)], &[]);
        assert!(Bm25Index::from_bytes(&bytes).is_none());
    }

    #[test]
    fn from_bytes_rejects_a_term_repeated_across_postings_blocks() {
        // `postings.insert` would silently overwrite the first block —
        // a repeat is corruption, not a redundant-but-harmless write.
        let bytes = image(
            &["a.md"],
            &[(0, 0, 1.0, 1, 0)],
            &[(10, &[(0, 1.0)]), (10, &[(0, 1.0)])],
        );
        assert!(Bm25Index::from_bytes(&bytes).is_none());
    }

    #[test]
    fn from_bytes_rejects_a_posting_repeated_for_the_same_term() {
        // Slot 0 posted twice for term 10: `carriers` would exceed
        // `live_count` and `idf` would go negative (the only path to a
        // negative idf in this design). `search` sums both postings
        // and double-counts the contribution, but `explain`'s
        // `.find()` only sees the first — breaking the "bit-for-bit
        // the same score" contract between them.
        let bytes = image(
            &["a.md"],
            &[(0, 0, 1.0, 1, 0)],
            &[(10, &[(0, 1.0), (0, 1.0)])],
        );
        assert!(Bm25Index::from_bytes(&bytes).is_none());
    }

    #[test]
    fn from_bytes_rejects_a_posting_list_not_slot_ascending() {
        // `to_bytes` always writes one term's live postings sorted by
        // slot — a descending pair is a shape it never produces.
        let bytes = image(
            &["a.md"],
            &[(0, 0, 1.0, 1, 0), (0, 1, 1.0, 2, 0)],
            &[(10, &[(1, 1.0), (0, 1.0)])],
        );
        assert!(Bm25Index::from_bytes(&bytes).is_none());
    }

    #[test]
    fn passage_search_via_index_matches_full_rescan() {
        let records = corpus();
        let index = Bm25Index::build(&records);
        for query in [
            "精米歩合はどこまで磨く?",
            "杜氏の経験",
            "state",
            "祭り 新酒",
        ] {
            let query_grams = grams(query);
            let expected = full_rescan(&records, &query_grams, 10);
            let got = index.search(&query_grams, 10, None);
            assert_eq!(
                got.iter()
                    .map(|(source, index, _, _)| (source.as_str(), *index))
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|(source, index, _)| (source.as_str(), *index))
                    .collect::<Vec<_>>(),
                "ranking must match the rescan oracle (query {query:?})"
            );
            for (hit, oracle) in got.iter().zip(&expected) {
                assert!(
                    (hit.3 - oracle.2).abs() <= 1e-4 * oracle.2.abs().max(1.0),
                    "score drifted from the oracle: {} vs {} (query {query:?})",
                    hit.3,
                    oracle.2
                );
            }
        }
    }

    #[test]
    fn search_does_not_duplicate_a_slot_whose_first_gram_underflows_idf_to_zero() {
        // A gram carried by every live paragraph pushes idf's `ln(1 +
        // epsilon)` below f32's precision floor around 1.0 — at 2^23
        // carriers it rounds to exactly 0.0, so `contribution` does
        // too, matching `scores`' own starting value. If the first
        // matching gram's "first hit" test reads that score instead of
        // an explicit visited flag, a second matching gram duplicates
        // the slot into `touched`.
        const CARRIERS: u32 = 8_388_608; // 2^23
        let mut index = Bm25Index::empty();
        assert_eq!(
            idf(CARRIERS as f32, CARRIERS as f32),
            0.0,
            "test setup must actually underflow idf to zero"
        );

        index.sources.push("only".to_string());
        index.source_ids.insert("only".to_string(), 0);
        index.slots.push(Slot {
            source_id: 0,
            index: 0,
            length: 2.0,
            hash: 42,
            question_hash: 0,
            alive: true,
        });
        index.by_source.insert(0, vec![0]);
        index.live_count = CARRIERS;
        index.live_total_length = f64::from(CARRIERS) * 2.0;

        let underflowing_gram = 1u64;
        let normal_gram = 2u64;
        let common_postings = index.postings.entry(underflowing_gram).or_default();
        common_postings.reserve(CARRIERS as usize);
        for _ in 0..CARRIERS {
            common_postings.push(Posting { slot: 0, tf: 1.0 });
        }
        index
            .postings
            .entry(normal_gram)
            .or_default()
            .push(Posting { slot: 0, tf: 1.0 });

        let hits = index.search(&[underflowing_gram, normal_gram], 10, None);
        assert_eq!(
            hits.len(),
            1,
            "the slot must appear once even though its first matching gram's \
             contribution underflowed to 0.0: {hits:?}"
        );
    }

    #[test]
    fn search_reports_a_hit_whose_every_matching_gram_underflowed_idf_to_zero() {
        // #603 item 3: unlike the sibling test above, the query here
        // has NO normal gram to keep the total score above 0.0 — every
        // gram it matches underflows `idf` to exactly 0.0 (same 2^23
        // carrier scale). `search` must still report the paragraph:
        // dropping a touched-but-0.0 slot makes an ubiquitous-term
        // query indistinguishable from one that matched nothing, and
        // disagrees with `explain`, which never hides a 0.0 addend.
        const CARRIERS: u32 = 8_388_608; // 2^23
        let mut index = Bm25Index::empty();
        assert_eq!(
            idf(CARRIERS as f32, CARRIERS as f32),
            0.0,
            "test setup must actually underflow idf to zero"
        );

        index.sources.push("only".to_string());
        index.source_ids.insert("only".to_string(), 0);
        index.slots.push(Slot {
            source_id: 0,
            index: 0,
            length: 2.0,
            hash: 42,
            question_hash: 0,
            alive: true,
        });
        index.by_source.insert(0, vec![0]);
        index.live_count = CARRIERS;
        index.live_total_length = f64::from(CARRIERS) * 2.0;

        let underflowing_gram = 1u64;
        let postings = index.postings.entry(underflowing_gram).or_default();
        postings.reserve(CARRIERS as usize);
        for _ in 0..CARRIERS {
            postings.push(Posting { slot: 0, tf: 1.0 });
        }

        let hits = index.search(&[underflowing_gram], 10, None);
        assert_eq!(
            hits,
            vec![("only".to_string(), 0, 42, 0.0)],
            "a slot the query actually touched must be reported even at score 0.0"
        );
    }

    #[test]
    fn idf_underflows_to_zero_only_at_near_total_carrier_scale() {
        // Pins the trigger scale the two `search` tests above rely on:
        // a small corpus (a handful of live paragraphs, a query gram
        // most of them carry) keeps idf comfortably above zero, while
        // 2^23 carriers over 2^23 live paragraphs is where f32's `ln(1
        // + epsilon)` rounds all the way down.
        assert!(idf(10.0, 9.0) > 0.0, "a small corpus must not underflow");
        assert_eq!(
            idf(8_388_608.0, 8_388_608.0),
            0.0,
            "2^23 carriers over 2^23 live paragraphs is the underflow scale search relies on"
        );
    }

    #[test]
    fn explain_reports_the_addends_search_summed() {
        let records = corpus();
        let mut index = Bm25Index::build(&records);
        let query = grams("精米歩合はどこまで磨く?");

        // Every hit's score is reproduced exactly: same addends, same
        // order, bit for bit — explain and search share the formula.
        let hits = index.search(&query, 10, None);
        assert!(!hits.is_empty());
        for (source, paragraph, hash, score) in &hits {
            let evidence = index.explain(&query, source, *paragraph).unwrap();
            assert_eq!(
                evidence.score, *score,
                "explain must sum exactly what search summed ({source}:{paragraph})"
            );
            assert_eq!(evidence.hash, *hash);
            assert_eq!(evidence.terms.len(), query.len());
            let sum: f32 = evidence.terms.iter().map(|term| term.contribution).sum();
            assert_eq!(sum, evidence.score);
            assert!(
                evidence
                    .terms
                    .iter()
                    .any(|term| term.tf > 0.0 && term.carriers > 0 && term.contribution > 0.0)
            );
        }

        // A paragraph sharing nothing with the query still gets a full
        // per-term table — all-zero tf IS the no_term_overlap evidence.
        let evidence = index.explain(&query, "docs/code.md", 0).unwrap();
        assert_eq!(evidence.score, 0.0);
        assert_eq!(evidence.terms.len(), query.len());
        assert!(
            evidence
                .terms
                .iter()
                .all(|term| term.tf == 0.0 && term.contribution == 0.0)
        );

        // Question terms are the paragraph's own evidence: the text
        // never says 削る, the attached question does.
        let questioned = PassageRecord::for_tests_with_questions(
            "精米歩合は50パーセントまで磨く。",
            vec![(0, "米はどれくらい削るのか".to_string())],
        );
        let with_question = Bm25Index::build(&[("doc".to_string(), questioned)]);
        let evidence = with_question
            .explain(&grams("米をどれくらい削る?"), "doc", 0)
            .unwrap();
        assert!(evidence.score > 0.0);

        // Unknown paragraphs, unknown sources, and tombstoned sources
        // have no index evidence to report.
        assert!(index.explain(&query, "docs/aomine.md", 99).is_none());
        assert!(index.explain(&query, "missing.md", 0).is_none());
        index.remove_source("docs/aomine.md");
        assert!(index.explain(&query, "docs/aomine.md", 0).is_none());
    }

    #[test]
    fn retracted_paragraph_never_matches_again() {
        let records = corpus();
        let mut index = Bm25Index::build(&records);
        let query = grams("杜氏の経験");
        assert!(
            index
                .search(&query, 5, None)
                .iter()
                .any(|(source, ..)| source == "docs/takase.md")
        );
        index.remove_source("docs/takase.md");
        assert!(
            index
                .search(&query, 5, None)
                .iter()
                .all(|(source, ..)| source != "docs/takase.md"),
            "a tombstoned source must not resurface"
        );
    }

    /// The #167 pre-lane filter: only eligible sources are served, and
    /// an eligible paragraph's score is EXACTLY its unfiltered score —
    /// the filter gates eligibility, it never re-weights (idf and the
    /// average length stay corpus-global).
    #[test]
    fn an_eligible_set_gates_hits_without_reweighting_their_scores() {
        let records = corpus();
        let index = Bm25Index::build(&records);
        // Terms straddle two sources (霧沢町 in aomine, 杜氏 in
        // takase), so the unfiltered ranking has real competition.
        let query = grams("霧沢町の杜氏");
        let unfiltered = index.search(&query, 10, None);
        assert!(
            unfiltered
                .iter()
                .any(|(source, ..)| source == "docs/takase.md")
        );
        assert!(unfiltered.len() > 1, "the corpus must offer competition");

        let only_takase: std::collections::BTreeSet<String> =
            ["docs/takase.md".to_string()].into_iter().collect();
        let filtered = index.search(&query, 10, Some(&only_takase));
        assert!(
            filtered
                .iter()
                .all(|(source, ..)| source == "docs/takase.md"),
            "nothing outside the eligible set may be served"
        );
        for (source, paragraph, _, score) in &filtered {
            let (.., unfiltered_score) = unfiltered
                .iter()
                .find(|(s, p, ..)| s == source && p == paragraph)
                .expect("an eligible hit also ranks unfiltered");
            assert_eq!(
                score, unfiltered_score,
                "the filter must not change an eligible paragraph's score"
            );
        }

        let nobody = std::collections::BTreeSet::new();
        assert!(
            index.search(&query, 10, Some(&nobody)).is_empty(),
            "an empty eligible set serves nothing"
        );
    }

    #[test]
    fn upserting_a_source_replaces_its_paragraphs_wholesale() {
        let records = corpus();
        let mut index = Bm25Index::build(&records);
        let updated = record("高瀬は引退し、後任の杜氏は佐伯となった。");
        index.upsert_source("docs/takase.md", &updated);

        let hits = index.search(&grams("後任の杜氏"), 5, None);
        assert_eq!(hits[0].0, "docs/takase.md");
        assert!(
            index
                .search(&grams("経験は30年"), 5, None)
                .iter()
                .all(|(source, ..)| source != "docs/takase.md"),
            "the old paragraph is gone with the upsert"
        );
    }

    /// Issue #563 item 2: `AppState::refresh_bm25` uses this return to
    /// decide whether a retraction actually changed the resident index
    /// — wrong here means the sidecar gets rewritten on every flush
    /// tick even when nothing moved. Three shapes: a source never
    /// interned, a source already fully tombstoned, and a source with
    /// live paragraphs still to kill.
    #[test]
    fn remove_source_reports_whether_it_actually_tombstoned_anything() {
        let records = vec![("a".to_string(), record("霧沢町の湧き水。"))];
        let mut index = Bm25Index::build(&records);

        assert!(
            !index.remove_source("never-interned"),
            "a source this index never saw must report no change"
        );
        assert!(
            index.remove_source("a"),
            "a source with live paragraphs must report a change"
        );
        assert!(
            !index.remove_source("a"),
            "retracting an already-tombstoned source a second time must report no change"
        );
    }

    /// Caught in review on #574 (issue #563 item 2 itself): an empty
    /// or whitespace-only `PassageRecord` — a legitimate submission,
    /// `PassageStore` accepts one — has zero paragraphs, so upserting
    /// it for a source with nothing live to tombstone either leaves
    /// the index untouched. `AppState::refresh_bm25`'s dirty gate
    /// trusts this return now instead of assuming every `Some(record)`
    /// arm is a change.
    #[test]
    fn upsert_source_reports_no_change_for_an_empty_record_with_nothing_to_tombstone() {
        let mut index = Bm25Index::empty();

        assert!(
            !index.upsert_source("empty", &record("")),
            "a brand-new source with zero paragraphs and nothing to tombstone \
             must report no change"
        );
        assert!(
            index.upsert_source("real", &record("霧沢町の湧き水。")),
            "a record with actual paragraphs is a real change"
        );
        assert!(
            index.upsert_source("real", &record("")),
            "replacing it with an empty record still tombstones what was live"
        );
        assert!(
            !index.upsert_source("real", &record("")),
            "and once nothing is left live, upserting empty again is inert"
        );
    }

    #[test]
    fn tombstoned_postings_do_not_inflate_document_frequency() {
        // Two paragraphs share a term; kill one. If df still counted
        // the corpse, the survivor's idf would sink measurably.
        let records = vec![
            ("a".to_string(), record("霧沢町の湧き水。")),
            ("b".to_string(), record("霧沢町の祭り。")),
        ];
        let mut index = Bm25Index::build(&records);
        index.remove_source("b");

        let survivors = vec![records[0].clone()];
        let oracle = full_rescan(&survivors, &grams("霧沢町"), 5);
        let got = index.search(&grams("霧沢町"), 5, None);
        assert_eq!(got.len(), 1);
        assert!(
            (got[0].3 - oracle[0].2).abs() <= 1e-4,
            "df must count live paragraphs only: {} vs {}",
            got[0].3,
            oracle[0].2
        );
    }

    #[test]
    fn reclaim_is_due_once_tombstones_outnumber_a_quarter_of_the_living() {
        let mut index = Bm25Index::empty();
        // Below the floor nothing is ever due, however lopsided.
        index.upsert_source("only", &record("ひとつだけ。"));
        index.remove_source("only");
        assert!(!index.needs_reclaim(), "the floor absorbs small counts");
        index.dead_count = COMPACT_DEAD_FLOOR + 1;
        index.live_count = 1;
        assert!(index.needs_reclaim());
        index.live_count = (COMPACT_DEAD_FLOOR + 1) * 4;
        assert!(!index.needs_reclaim(), "a big live set earns more slack");
    }

    #[test]
    fn index_round_trips_through_bytes_and_tombstones_stay_behind() {
        let records = corpus();
        let mut index = Bm25Index::build(&records);
        index.remove_source("docs/takase.md");

        let bytes = index.to_bytes();
        let reborn = Bm25Index::from_bytes(&bytes).unwrap();
        assert_eq!(reborn.dead_count, 0, "saving IS a compaction");
        for query in ["精米歩合はどこまで磨く?", "state", "杜氏の経験"] {
            let grams = grams(query);
            assert_eq!(
                reborn.search(&grams, 10, None),
                index.search(&grams, 10, None),
                "the reborn index must answer exactly like the live one (query {query:?})"
            );
        }
        assert_eq!(
            index.source_digests(),
            reborn.source_digests(),
            "digests survive the round trip — the drift detector depends on it"
        );
        // Canonical serialization: same content, same bytes.
        assert_eq!(bytes, reborn.to_bytes());

        assert!(Bm25Index::from_bytes(b"garbage").is_none());
        assert!(Bm25Index::from_bytes(&bytes[..bytes.len() - 1]).is_none());
        let mut padded = bytes.clone();
        padded.push(0);
        assert!(
            Bm25Index::from_bytes(&padded).is_none(),
            "trailing bytes are corruption, not slack"
        );
    }

    #[test]
    fn from_bytes_rejects_non_finite_or_negative_length_and_tf() {
        // One source, one paragraph → one slot, at least one posting.
        let records = vec![("s".to_string(), record("あい"))];
        let index = Bm25Index::build(&records);
        let good = index.to_bytes();
        assert!(
            Bm25Index::from_bytes(&good).is_some(),
            "the pristine bytes must decode — the corruption tests below mean nothing otherwise"
        );

        // Slot layout: magic(8) + source_count(4) + name_len(4) + "s"(1)
        // + slot_count(4) + source_id(4) + paragraph(4) = 29, then length.
        const LENGTH_OFF: usize = 29;
        let stored = f32::from_le_bytes(good[LENGTH_OFF..LENGTH_OFF + 4].try_into().unwrap());
        assert_eq!(
            stored, index.slots[0].length,
            "offset check: bytes[29..33] must really be the slot length"
        );
        // The final field written is the last posting's tf, so the tail
        // four bytes are always a tf — no offset math needed.
        let tf_off = good.len() - 4;
        let stored_tf = f32::from_le_bytes(good[tf_off..].try_into().unwrap());
        assert!(
            stored_tf.is_finite() && stored_tf > 0.0,
            "the tail is a live tf"
        );

        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
            let mut length_torn = good.clone();
            length_torn[LENGTH_OFF..LENGTH_OFF + 4].copy_from_slice(&poison.to_le_bytes());
            assert!(
                Bm25Index::from_bytes(&length_torn).is_none(),
                "a {poison} paragraph length poisons the average — reject and rebuild"
            );

            let mut tf_torn = good.clone();
            tf_torn[tf_off..].copy_from_slice(&poison.to_le_bytes());
            assert!(
                Bm25Index::from_bytes(&tf_torn).is_none(),
                "a {poison} term frequency makes the score NaN — reject and rebuild"
            );
        }

        // 0.0 is not in the poison list above: unlike a negative or
        // non-finite length, a paragraph legitimately CAN have length
        // 0.0 (no grams at all). A 0.0 tf has no such legitimate
        // reading — `to_bytes` never writes one (every posted gram's
        // frequency starts at 1.0) — so it gets its own check, not a
        // shared loop with length's poison values.
        let mut zero_tf = good.clone();
        zero_tf[tf_off..].copy_from_slice(&0.0f32.to_le_bytes());
        assert!(
            Bm25Index::from_bytes(&zero_tf).is_none(),
            "a term posted with zero frequency is self-contradictory — reject and rebuild, \
             not surface as a spurious no-overlap hit now that #603 removed search's \
             score>0.0 filter"
        );
    }

    #[test]
    fn record_digest_matches_the_index_side_fold() {
        let records = corpus();
        let index = Bm25Index::build(&records);
        let digests = index.source_digests();
        for (source, record) in &records {
            assert_eq!(
                Some(digests[source]),
                record_digest(record),
                "both sides of the drift comparison must compute the same digest ({source})"
            );
        }
    }

    #[test]
    fn record_digest_is_none_for_a_paragraph_less_record_and_absent_from_source_digests() {
        // Whitespace-only text splits to zero paragraphs (paragraph.rs)
        // — the same shape `interpret_passages` lets through the public
        // API uncaught. Both sides of the drift comparison must agree
        // there is nothing to digest, or the repair loop in
        // `PassageSearch::bm25_index` reads the source as permanently
        // drifted.
        let blank = record("   \n\n\t \n");
        assert_eq!(record_digest(&blank), None);

        let index = Bm25Index::build(&[("blank".to_string(), blank)]);
        assert!(
            !index.source_digests().contains_key("blank"),
            "a source with no live slot must have no digest entry"
        );
    }

    #[test]
    fn doc2query_questions_index_into_the_lexical_lane() {
        // The paragraph never says 「削る」; only its attached question
        // does. Landing the hit proves the question's terms joined the
        // paragraph's postings — the lexical mirror of the vector
        // lane's question rows.
        let bare = record("精米歩合は50パーセントまで磨く。");
        let questioned = PassageRecord::for_tests_with_questions(
            "精米歩合は50パーセントまで磨く。",
            vec![(0, "米はどれくらい削るのか".to_string())],
        );
        let query = grams("米をどれくらい削る?");

        let without = Bm25Index::build(&[("doc".to_string(), bare)]);
        let with = Bm25Index::build(&[("doc".to_string(), questioned.clone())]);
        let baseline: f32 = without
            .search(&query, 5, None)
            .first()
            .map(|hit| hit.3)
            .unwrap_or(0.0);
        let hits = with.search(&query, 5, None);
        assert_eq!(hits.len(), 1, "the question's terms must land the hit");
        assert_eq!((hits[0].0.as_str(), hits[0].1), ("doc", 0));
        assert!(
            hits[0].3 > baseline,
            "question terms must add scoring evidence: {} vs {baseline}",
            hits[0].3
        );

        // The staleness handshake is untouched: the hit still hands
        // back the paragraph TEXT hash the store validates against.
        assert_eq!(hits[0].2, questioned.paragraphs[0].hash);
    }

    #[test]
    fn a_question_only_change_moves_the_drift_digest() {
        let text = "精米歩合は50パーセントまで磨く。";
        let bare = record(text);
        let questioned = PassageRecord::for_tests_with_questions(
            text,
            vec![(0, "米はどれくらい削るのか".to_string())],
        );
        let reworded = PassageRecord::for_tests_with_questions(
            text,
            vec![(0, "何パーセントまで磨くのか".to_string())],
        );

        // Same text, different questions: three distinct digests, so
        // the load-time repair re-upserts instead of trusting a stale
        // sidecar.
        let digests = [
            record_digest(&bare),
            record_digest(&questioned),
            record_digest(&reworded),
        ];
        assert_ne!(digests[0], digests[1]);
        assert_ne!(digests[1], digests[2]);
        assert_ne!(digests[0], digests[2]);

        // Both sides of the comparison agree on a questioned record —
        // through the byte round trip too, or the sidecar would
        // re-upsert every boot.
        let index = Bm25Index::build(&[("doc".to_string(), questioned.clone())]);
        assert_eq!(
            Some(index.source_digests()["doc"]),
            record_digest(&questioned)
        );
        let reborn = Bm25Index::from_bytes(&index.to_bytes()).unwrap();
        assert_eq!(
            Some(reborn.source_digests()["doc"]),
            record_digest(&questioned)
        );
    }

    #[test]
    fn search_tie_breaks_deterministically_by_source_then_index() {
        // Identical twin paragraphs in two sources: equal scores, so
        // the order must come from the names.
        let records = vec![
            ("b-doc".to_string(), record("同じ本文。")),
            ("a-doc".to_string(), record("同じ本文。")),
        ];
        let index = Bm25Index::build(&records);
        let hits = index.search(&grams("同じ本文"), 5, None);
        assert_eq!(hits[0].0, "a-doc");
        assert_eq!(hits[1].0, "b-doc");
        assert_eq!(hits[0].3, hits[1].3);
    }

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn proptest_config() -> ProptestConfig {
            let cases = std::env::var("PROPTEST_CASES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32);
            ProptestConfig {
                cases,
                ..ProptestConfig::default()
            }
        }

        fn source_name_strategy() -> impl Strategy<Value = String> {
            "[a-z]{1,8}\\.md"
        }

        /// A small vocabulary mixing Japanese and ASCII tokens, so
        /// generated text exercises the same tokenizer paths as the
        /// hand-written corpus without depending on proptest's unicode
        /// string-regex support.
        fn word_strategy() -> impl Strategy<Value = &'static str> {
            prop_oneof![
                Just("蔵元"),
                Just("山田錦"),
                Just("精米歩合"),
                Just("杜氏"),
                Just("南部"),
                Just("state"),
                Just("impl"),
                Just("boot"),
                Just("query"),
                Just("index"),
            ]
        }

        fn text_strategy() -> impl Strategy<Value = String> {
            prop::collection::vec(
                prop::collection::vec(word_strategy(), 0..8).prop_map(|words| words.join("")),
                0..5,
            )
            .prop_map(|paragraphs| paragraphs.join("\n\n"))
        }

        fn corpus_strategy() -> impl Strategy<Value = Vec<(String, Arc<PassageRecord>)>> {
            prop::collection::vec((source_name_strategy(), text_strategy()), 0..6).prop_map(
                |entries| {
                    entries
                        .into_iter()
                        .map(|(source, text)| (source, record(&text)))
                        .collect()
                },
            )
        }

        proptest! {
            #![proptest_config(proptest_config())]

            /// `to_bytes` is documented as canonical, so a fresh decode must
            /// re-encode identically — for arbitrary corpora, including ones
            /// with tombstones left behind by removals.
            #[test]
            fn index_round_trips_through_bytes_for_arbitrary_corpora(
                records in corpus_strategy(),
                removals in prop::collection::vec(any::<prop::sample::Index>(), 0..4),
            ) {
                let mut index = Bm25Index::build(&records);
                if !records.is_empty() {
                    for pick in removals {
                        let i = pick.index(records.len());
                        index.remove_source(&records[i].0);
                    }
                }

                let bytes = index.to_bytes();
                let reborn = Bm25Index::from_bytes(&bytes)
                    .expect("a freshly serialized index must always decode");
                prop_assert_eq!(bytes, reborn.to_bytes());
                prop_assert_eq!(reborn.dead_count, 0, "saving IS a compaction");
            }

            /// However malformed, `from_bytes` must reject rather than panic.
            #[test]
            fn from_bytes_never_panics_on_arbitrary_bytes(
                bytes in prop::collection::vec(any::<u8>(), 0..512),
            ) {
                let _ = Bm25Index::from_bytes(&bytes);
            }

            /// Single-byte mutations of an otherwise-valid image are the
            /// realistic corruption case (bit flips, truncated writes) —
            /// still must never panic.
            #[test]
            fn from_bytes_never_panics_on_mutated_valid_bytes(
                records in corpus_strategy(),
                mutations in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 0..16),
            ) {
                let index = Bm25Index::build(&records);
                let mut bytes = index.to_bytes();
                for (pick, value) in mutations {
                    *pick.get_mut(&mut bytes) = value;
                }
                let _ = Bm25Index::from_bytes(&bytes);
            }
        }
    }
}
