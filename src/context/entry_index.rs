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
    use crate::context::{Context, normalize_entry};

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

    /// An empty spelling containment-matches every cue —
    /// `str::contains("")` is always true. The HTTP and import surfaces
    /// refuse empty aliases, but an image written before they did can
    /// still carry one; resolution must treat it as inert, not as a
    /// phantom hit on every query.
    #[test]
    fn an_empty_alias_spelling_never_resolves() {
        let mut context = Context::default();
        context.associate("私", "好き", "りんご", 1.0).unwrap();
        context.add_concept_alias("", "私").unwrap();

        // Unrelated cues stay empty-handed, and a real cue must not
        // grow a second resolution through the zero-length span.
        assert!(context.resolve("ぶどう").is_empty());
        let hits = context.resolve("りんご");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].name, "りんご");
    }

    #[test]
    fn removing_a_label_alias_rebuilds_the_entry_index() {
        let mut context = Context::default();
        context.associate("A", "関係", "B", 1.0).unwrap();
        context.add_label_alias("xyzzy", "関係").unwrap();
        assert_eq!(context.resolve_label("xyzzy")[0].name, "関係");

        assert_eq!(context.remove_label_alias("xyzzy").as_deref(), Some("関係"));
        assert!(context.resolve_label("xyzzy").is_empty());
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

    /// `EntryIndex::twins` excludes a candidate pair with `if dice <
    /// dice_floor { continue; }` — a bare `.clamp(0.0, 1.0)` lets a NaN
    /// floor through unchanged, and `dice < NaN` is false for every
    /// dice score, so the exclusion would never fire: every pair in the
    /// namespace, however dissimilar, would flood out as a "similar"
    /// candidate. Mapping NaN onto the strictest floor (1.0) keeps the
    /// filter fail-closed instead.
    #[test]
    fn similar_concepts_treats_a_nan_floor_as_the_strictest_admission_bar() {
        let mut context = Context::default();
        context.associate("青嶺酒造", "分類", "酒蔵", 1.0).unwrap();
        context.associate("青嶺酒蔵", "分類", "酒蔵", 1.0).unwrap();
        let is_the_pair = |pairs: &[(String, String, f64)]| {
            pairs.iter().any(|(a, b, _)| {
                (a == "青嶺酒造" && b == "青嶺酒蔵") || (a == "青嶺酒蔵" && b == "青嶺酒造")
            })
        };

        // The two spellings share 2 of 3 informative bigrams each:
        // Dice = 2·2/(3+3) ≈ 0.667 — admitted by a lax floor ...
        assert!(is_the_pair(
            &context
                .similar_concepts(0.5, Deadline::unbounded())
                .unwrap()
        ));
        // ... excluded by a floor stricter than their score ...
        assert!(!is_the_pair(
            &context
                .similar_concepts(1.0, Deadline::unbounded())
                .unwrap()
        ));
        // ... and a NaN floor must exclude it the same way, not flood
        // it back in the way an unclamped `dice < NaN` comparison would.
        assert!(!is_the_pair(
            &context
                .similar_concepts(f64::NAN, Deadline::unbounded())
                .unwrap()
        ));
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
        let twins = context
            .similar_concepts(0.6, Deadline::unbounded())
            .unwrap();
        assert_eq!(twins.len(), 1, "{twins:?}");
        assert_eq!(
            (twins[0].0.as_str(), twins[0].1.as_str()),
            ("青嶺酒造", "青嶺酒蔵")
        );
        assert!((twins[0].2 - 2.0 / 3.0).abs() < 1e-9);
        // The brand/brewery containment pair (dice 0.5) stays below the
        // floor, and the alias never pairs with its own canonical.
        assert!(
            context
                .similar_concepts(0.55, Deadline::unbounded())
                .unwrap()
                .len()
                == 1
        );

        // Label forks are the costliest kind; the shared 住み込む stem
        // pushes the pair over the floor for review.
        let labels = context.similar_labels(0.6, Deadline::unbounded()).unwrap();
        assert_eq!(
            (labels[0].0.as_str(), labels[0].1.as_str()),
            ("住み込む場所", "住み込む期間")
        );
    }

    /// Before the fix, `twins()`'s `informative` pass had no deadline
    /// check at all, and its O(len²) pass over one large posting list
    /// was checked only once per OUTER bigram — not once per pair — so
    /// an already-expired deadline could still cost hundreds of
    /// milliseconds on a large enough shared vocabulary instead of
    /// returning almost immediately. `shared_count` sets one posting
    /// list as long as `stop_gram` allows before `twins()` skips it
    /// outright, so this exercises both O(len²) passes at once.
    #[test]
    fn similar_concepts_honors_an_expired_deadline_promptly_despite_a_large_shared_vocabulary() {
        let n: usize = 40_000;
        let stop_gram = (n / 20).max(64); // mirrors twins()'s formula
        let shared_count = stop_gram; // as large as possible while still under the skip threshold

        let mut context = Context::default();
        for i in 0..shared_count {
            context
                .associate(
                    format!("sharedprefixbigrams{i:06}"),
                    "r",
                    format!("obj{i:06}"),
                    1.0,
                )
                .unwrap();
        }
        for i in shared_count..n {
            context
                .associate(format!("unique{i:06}"), "r", format!("uobj{i:06}"), 1.0)
                .unwrap();
        }

        let deadline = Deadline::after(std::time::Duration::from_millis(1));
        std::thread::sleep(std::time::Duration::from_millis(5)); // guarantee it's already expired
        let start = std::time::Instant::now();
        let result = context.similar_concepts(0.0, deadline);
        let elapsed = start.elapsed();

        assert!(
            result.is_err(),
            "an already-expired deadline must be honored"
        );
        assert!(
            elapsed < std::time::Duration::from_millis(200),
            "deadline check should return in tens of ms, not scale with vocabulary size; took {elapsed:?}"
        );
    }

    #[test]
    fn entry_index_normalization_preserves_distinct_spellings() {
        assert_eq!(normalize_entry("ＡＬＰＨＡ"), "alpha");
        assert_ne!(normalize_entry("alpha"), normalize_entry("beta"));

        let mut index = EntryIndex::default();
        index.push("alpha", 0);
        index.push("beta", 1);

        let resolved = index.resolve("ALPHA", 0.1);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved.get(&0), Some(&(1.0, MatchKind::Exact)));
    }

    #[test]
    fn match_kind_as_str_matches_each_variant() {
        assert_eq!(MatchKind::Exact.as_str(), "exact");
        assert_eq!(MatchKind::Alias.as_str(), "alias");
        assert_eq!(MatchKind::Containment.as_str(), "containment");
        assert_eq!(MatchKind::Fuzzy.as_str(), "fuzzy");
    }

    #[test]
    fn entry_index_keeps_the_first_match_kind_when_scores_tie() {
        let mut index = EntryIndex::default();
        index.push("ab", 0); // containment against "abcd": 2 / 4
        index.push("abXcdY", 0); // fuzzy: 2 * 2 shared / (3 + 5)

        let resolved = index.resolve("abcd", 0.3);
        assert_eq!(resolved.get(&0), Some(&(0.5, MatchKind::Containment)));
    }

    #[test]
    fn entry_index_keeps_a_posting_at_the_stop_gram_threshold() {
        let mut index = EntryIndex::default();
        for target in 0..64 {
            index.push(&format!("ab{target:04}"), target);
        }

        // 64 spans floors the threshold at 64; a posting becomes a stop
        // gram only after it exceeds that threshold, not when equal.
        let resolved = index.resolve("abzz", 0.1);
        assert_eq!(resolved.get(&0), Some(&(1.0 / 3.0, MatchKind::Fuzzy)));
    }

    #[test]
    fn entry_index_scales_the_stop_gram_threshold_with_the_vocabulary() {
        let mut index = EntryIndex::default();
        for target in 0..2_000 {
            let spelling = if target < 80 {
                format!("abzz{target:04}")
            } else {
                format!("qqzz{target:04}")
            };
            index.push(&spelling, target);
        }

        // At 2,000 spans the threshold is 100: the 80-entry "ab"/"bz"
        // postings remain informative while the ubiquitous "zz" is
        // removed from both Dice denominators.
        let resolved = index.resolve("abzz!", 0.1);
        assert_eq!(resolved.get(&0), Some(&(0.8, MatchKind::Fuzzy)));
    }

    #[test]
    fn entry_index_twins_use_the_scaled_stop_gram_threshold() {
        let mut index = EntryIndex::default();
        for target in 0..1_300 {
            let spelling = if target < 65 {
                format!("abzz{target:04}")
            } else {
                format!("qqzz{target:04}")
            };
            index.push(&spelling, target);
        }

        // The scaled threshold is exactly 65. "ab"/"bz" remain useful
        // at that boundary and ubiquitous "zz" is excluded.
        let twins = index.twins(0.1, Deadline::unbounded()).unwrap();
        let pair = twins
            .iter()
            .find(|&&(a, b, _)| (a, b) == (0, 1))
            .expect("the two distinctive-prefix spellings are compared");
        assert_eq!(pair.2, 1.0);
    }
}
