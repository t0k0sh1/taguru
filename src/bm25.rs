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
    pub(crate) fn upsert_source(&mut self, source: &str, record: &PassageRecord) {
        let source_id = self.intern(source);
        self.tombstone(source_id);
        let slot_list = self.by_source.entry(source_id).or_default();
        for (span, text) in record.paragraph_texts() {
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
            for (_, question) in record
                .questions
                .iter()
                .filter(|&&(paragraph, _)| paragraph == span.index)
            {
                for gram in passage_terms(question) {
                    count(gram);
                }
            }
            self.slots.push(Slot {
                source_id,
                index: span.index,
                length,
                hash: span.hash,
                question_hash: questions_fold(record, span.index),
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
    }

    /// Tombstones one source's paragraphs (a retraction).
    pub(crate) fn remove_source(&mut self, source: &str) {
        if let Some(&source_id) = self.source_ids.get(source) {
            self.tombstone(source_id);
            self.reclaim_if_due();
        }
    }

    fn tombstone(&mut self, source_id: u32) {
        if let Some(slot_list) = self.by_source.get_mut(&source_id) {
            for &slot in slot_list.iter() {
                let slot = &mut self.slots[slot as usize];
                if slot.alive {
                    slot.alive = false;
                    self.live_count -= 1;
                    self.live_total_length -= f64::from(slot.length);
                    self.dead_count += 1;
                }
            }
            slot_list.clear();
        }
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
    pub(crate) fn search(&self, query_grams: &[u64], limit: usize) -> Vec<IndexHit> {
        if self.live_count == 0 || query_grams.is_empty() {
            return Vec::new();
        }
        let total = self.live_count as f32;
        let average_length = (self.live_total_length / f64::from(self.live_count)).max(1.0) as f32;

        let mut scores: Vec<f32> = vec![0.0; self.slots.len()];
        let mut touched: Vec<u32> = Vec::new();
        for gram in query_grams {
            let Some(postings) = self.postings.get(gram) else {
                continue;
            };
            let carriers = postings
                .iter()
                .filter(|posting| self.slots[posting.slot as usize].alive)
                .count() as f32;
            if carriers == 0.0 {
                continue;
            }
            let idf = (1.0 + (total - carriers + 0.5) / (carriers + 0.5)).ln();
            for posting in postings {
                let slot = &self.slots[posting.slot as usize];
                if !slot.alive {
                    continue;
                }
                if scores[posting.slot as usize] == 0.0 {
                    touched.push(posting.slot);
                }
                scores[posting.slot as usize] += idf * (posting.tf * (K1 + 1.0))
                    / (posting.tf + K1 * (1.0 - B + B * slot.length / average_length));
            }
        }

        let mut hits: Vec<IndexHit> = touched
            .into_iter()
            .filter(|&slot| scores[slot as usize] > 0.0)
            .map(|slot_id| {
                let slot = &self.slots[slot_id as usize];
                (
                    self.sources[slot.source_id as usize].clone(),
                    slot.index,
                    slot.hash,
                    scores[slot_id as usize],
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

    /// Per-source digest over (paragraph index, paragraph hash,
    /// question fold) of the LIVE slots, in index order — the load-time
    /// drift detector: a source whose digest disagrees with the passage
    /// store's current record gets re-upserted instead of costing a
    /// full rebuild.
    pub(crate) fn source_digests(&self) -> HashMap<String, u64> {
        let mut digests = HashMap::new();
        for (&source_id, slot_list) in &self.by_source {
            let mut digest = DIGEST_OFFSET;
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
    /// index costs a rebuild, never an outage.
    pub(crate) fn load(path: &Path) -> Option<Self> {
        let bytes = std::fs::read(path).ok()?;
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
            // intern() assigns ids in insertion order = file order.
            index.intern(name);
        }
        let slot_count = read_u32(bytes, &mut pos)? as usize;
        for _ in 0..slot_count {
            let source_id = read_u32(bytes, &mut pos)?;
            if source_id as usize >= index.sources.len() {
                return None;
            }
            let paragraph = read_u32(bytes, &mut pos)?;
            let length = f32::from_le_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?);
            pos += 4;
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
            for _ in 0..posting_count {
                let slot = read_u32(bytes, &mut pos)?;
                if slot as usize >= index.slots.len() {
                    return None;
                }
                let tf = f32::from_le_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?);
                pos += 4;
                list.push(Posting { slot, tf });
            }
            index.postings.insert(term, list);
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

const DIGEST_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const DIGEST_PRIME: u64 = 0x0000_0100_0000_01b3;

/// One (paragraph index, paragraph hash, question fold) step of a
/// source digest — FNV-1a-shaped so the fold depends on order and
/// content both.
fn digest_fold(digest: u64, index: u32, hash: u64, question_hash: u64) -> u64 {
    let mut digest = digest;
    for byte in index
        .to_le_bytes()
        .into_iter()
        .chain(hash.to_le_bytes())
        .chain(question_hash.to_le_bytes())
    {
        digest ^= u64::from(byte);
        digest = digest.wrapping_mul(DIGEST_PRIME);
    }
    digest
}

/// FNV-1a over one paragraph's attached doc2query questions, in stored
/// order, a separator byte after each so adjacent questions cannot
/// blend — the questions' share of the drift digest, computed the same
/// way on both sides of the comparison.
fn questions_fold(record: &PassageRecord, paragraph: u32) -> u64 {
    let mut digest = DIGEST_OFFSET;
    for (_, question) in record
        .questions
        .iter()
        .filter(|&&(index, _)| index == paragraph)
    {
        for byte in question.as_bytes().iter().copied().chain([0xff]) {
            digest ^= u64::from(byte);
            digest = digest.wrapping_mul(DIGEST_PRIME);
        }
    }
    digest
}

/// The passage store's side of the drift comparison: the digest the
/// index WOULD have for this record if it were fresh.
pub(crate) fn record_digest(record: &PassageRecord) -> u64 {
    record
        .paragraphs
        .iter()
        .fold(DIGEST_OFFSET, |digest, span| {
            digest_fold(
                digest,
                span.index,
                span.hash,
                questions_fold(record, span.index),
            )
        })
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
            let got = index.search(&query_grams, 10);
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
    fn retracted_paragraph_never_matches_again() {
        let records = corpus();
        let mut index = Bm25Index::build(&records);
        let query = grams("杜氏の経験");
        assert!(
            index
                .search(&query, 5)
                .iter()
                .any(|(source, ..)| source == "docs/takase.md")
        );
        index.remove_source("docs/takase.md");
        assert!(
            index
                .search(&query, 5)
                .iter()
                .all(|(source, ..)| source != "docs/takase.md"),
            "a tombstoned source must not resurface"
        );
    }

    #[test]
    fn upserting_a_source_replaces_its_paragraphs_wholesale() {
        let records = corpus();
        let mut index = Bm25Index::build(&records);
        let updated = record("高瀬は引退し、後任の杜氏は佐伯となった。");
        index.upsert_source("docs/takase.md", &updated);

        let hits = index.search(&grams("後任の杜氏"), 5);
        assert_eq!(hits[0].0, "docs/takase.md");
        assert!(
            index
                .search(&grams("経験は30年"), 5)
                .iter()
                .all(|(source, ..)| source != "docs/takase.md"),
            "the old paragraph is gone with the upsert"
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
        let got = index.search(&grams("霧沢町"), 5);
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
                reborn.search(&grams, 10),
                index.search(&grams, 10),
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
    fn record_digest_matches_the_index_side_fold() {
        let records = corpus();
        let index = Bm25Index::build(&records);
        let digests = index.source_digests();
        for (source, record) in &records {
            assert_eq!(
                digests[source],
                record_digest(record),
                "both sides of the drift comparison must compute the same digest ({source})"
            );
        }
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
            .search(&query, 5)
            .first()
            .map(|hit| hit.3)
            .unwrap_or(0.0);
        let hits = with.search(&query, 5);
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
        assert_eq!(index.source_digests()["doc"], record_digest(&questioned));
        let reborn = Bm25Index::from_bytes(&index.to_bytes()).unwrap();
        assert_eq!(reborn.source_digests()["doc"], record_digest(&questioned));
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
        let hits = index.search(&grams("同じ本文"), 5);
        assert_eq!(hits[0].0, "a-doc");
        assert_eq!(hits[1].0, "b-doc");
        assert_eq!(hits[0].3, hits[1].3);
    }
}
