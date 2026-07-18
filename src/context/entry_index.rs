use std::collections::{HashMap, HashSet};

use crate::deadline::{Deadline, DeadlineExceeded};

use super::{MatchKind, claim_id, normalize};

/// One namespace's entry index — the derived, allocation-free machinery
/// behind [`super::Context::resolve`]. Every entry spelling (canonical
/// names now; aliases are designed to join the same index) is stored in
/// normalized form in one arena, and a character-bigram posting index
/// over those forms catches near-miss spellings that containment cannot.
/// Extended on every intern, rebuilt from the canonical names on load,
/// never persisted. Offsets are usize rather than u32 because
/// normalization can outgrow the offset space the canonical arena is
/// held to.
#[derive(Debug, Default)]
pub(super) struct EntryIndex {
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

impl EntryIndex {
    pub(super) fn push(&mut self, spelling: &str, target: u32) {
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
    pub(super) fn resolve(&self, cue: &str, dice_floor: f64) -> HashMap<u32, (f64, MatchKind)> {
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
            // A zero-length spelling would containment-match every cue
            // (`str::contains("")` is always true) and plant a phantom
            // hit in every resolution. The write paths refuse empty
            // spellings, but an image written before they did can
            // still carry one — skip it rather than serve it.
            if haystack.is_empty() {
                continue;
            }
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
    pub(super) fn twins(
        &self,
        dice_floor: f64,
        deadline: Deadline,
    ) -> Result<Vec<(u32, u32, f64)>, DeadlineExceeded> {
        let stop_gram = (self.spans.len() / 20).max(64);
        let is_stop = |bigram: u64| {
            self.bigrams
                .get(&bigram)
                .is_some_and(|postings| postings.len() > stop_gram)
        };
        // Built with an explicit loop rather than spans.iter().map(..)
        // so a deadline check can run per span: each span's own cost is
        // bounded (it scans only its own bigrams), but a caller with a
        // huge vocabulary and a short deadline must still be able to
        // bail before this whole pass finishes, not just once the
        // O(len²) pass below starts.
        let mut informative: Vec<usize> = Vec::with_capacity(self.spans.len());
        for span in &self.spans {
            if deadline.expired() {
                return Err(DeadlineExceeded);
            }
            let text = &self.arena[span.start..span.end];
            let unique: HashSet<u64> = bigrams_of(text).collect();
            informative.push(unique.iter().filter(|&&bigram| !is_stop(bigram)).count());
        }

        let mut shared: HashMap<(u32, u32), u32> = HashMap::new();
        for postings in self.bigrams.values() {
            if deadline.expired() {
                return Err(DeadlineExceeded);
            }
            if postings.len() > stop_gram {
                continue;
            }
            // Postings are appended in span order, so a < b holds.
            let mut tail = postings.as_slice();
            while let Some((&a, rest)) = tail.split_first() {
                // A single posting list can run up to stop_gram long,
                // and stop_gram scales with vocabulary size (spans.len()
                // / 20) — so this list's own O(len²) pass can outlast
                // `deadline` entirely between outer-loop checks on a
                // large enough vocabulary. Checking once per outer
                // element (O(len) checks) instead of once per pair
                // keeps the gap between checks to O(len), not O(len²).
                if deadline.expired() {
                    return Err(DeadlineExceeded);
                }
                for &b in rest {
                    *shared.entry((a, b)).or_insert(0) += 1;
                }
                tail = rest;
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
            *slot = slot.max(dice);
        }
        let mut twins: Vec<(u32, u32, f64)> = best
            .into_iter()
            .map(|((a, b), dice)| (a, b, dice))
            .collect();
        twins.sort_by(|x, y| {
            y.2.total_cmp(&x.2)
                .then_with(|| (x.0, x.1).cmp(&(y.0, y.1)))
        });
        Ok(twins)
    }

    /// Rough resident bytes of this index, for cache budgeting.
    pub(super) fn footprint(&self) -> usize {
        const MAP_ENTRY_OVERHEAD: usize = 48;
        self.arena.len()
            + self.spans.len() * size_of::<EntrySpan>()
            + self.bigrams.len() * (size_of::<u64>() + MAP_ENTRY_OVERHEAD)
            + self.posting_entries * size_of::<u32>()
    }
}

/// Adjacent character pairs of a normalized form, packed into u64 keys.
fn bigrams_of(text: &str) -> impl Iterator<Item = u64> {
    text.chars()
        .zip(text.chars().skip(1))
        .map(|(a, b)| ((a as u64) << 32).wrapping_add(b as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_index_counts_unique_postings_per_spelling() {
        let mut index = EntryIndex::default();
        index.push("abcd", 0); // ab, bc, cd
        assert_eq!(index.posting_entries, 3);
        index.push("aaaa", 1); // aa repeats, but one spelling posts once
        assert_eq!(index.posting_entries, 4);
        assert_eq!(
            index.footprint(),
            index.arena.len()
                + index.spans.len() * size_of::<EntrySpan>()
                + index.bigrams.len() * (size_of::<u64>() + 48)
                + index.posting_entries * size_of::<u32>()
        );
    }
}
