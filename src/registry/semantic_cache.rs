//! The semantic retrieval tier (issue #153): a paraphrased passage
//! query answers from the exact-match cache entry of an equivalent
//! earlier query. This module stores no payloads and invalidates
//! nothing — it holds only *equivalence claims*: "query B asks what
//! canonical query A asked", proven by embedding cosine plus a
//! text-level guard. A claim that holds rewrites the request's exact
//! key ([`super::RetrievalKey`]) to the canonical's parameters — same
//! operation, same resolved targets, and crucially the request's own
//! CURRENT [`super::retrieval_cache::TargetFingerprint`]s — and serves
//! whatever the exact tier holds under that key, byte-identical.
//!
//! That shape buys two properties the obvious "second cache keyed by
//! similarity" would lose:
//!
//! - **No invalidation, by construction.** An equivalence claim is
//!   revision-independent — if B meant A yesterday it means A after
//!   every write too. Freshness rides entirely on the rewritten key's
//!   revision lanes and identity nonce: a write bumps a lane, the
//!   rewritten key misses (the `stale` outcome), and the fresh fill
//!   re-registers the cluster's canonical. No walk, no TTL, no window
//!   where an invalidation races a fill — inherited wholesale from the
//!   exact tier's proof.
//! - **One payload store.** A semantic hit serves the canonical
//!   entry's `Arc`'d bytes; nothing is duplicated, and the exact
//!   tier's "identical is literal" contract stays pure — a semantic
//!   hit never files the served bytes under the request's own key, so
//!   ongoing semantic reliance stays visible in
//!   `taguru_semantic_cache_total` instead of vanishing after the
//!   first serve.
//!
//! Matching is deliberately conjunctive: cosine over the two query
//! embeddings (computed through the same [`super::CueCache`] the
//! search itself uses, so the fresh path still pays exactly one
//! provider call) must clear `TAGURU_SEMANTIC_CACHE_THRESHOLD`, AND
//! [`queries_agree`] must find no negation/numeric/entity asymmetry
//! between the two raw query strings. High cosine alone routinely
//! conflates a question with its negation, a changed number, or a
//! swapped name — the guard exists because those sit exactly where an
//! embedding cannot separate them.
//!
//! The guard is a tripwire, not a parser. It compares three multisets
//! over NFKC-folded text and refuses on ANY asymmetry, so its false
//! positives only cost a cache miss while its false negatives are the
//! dangerous direction. Known blind spots, accepted as out of scope
//! (`#153`): spelled-out numbers ("two" vs "three") and kanji numerals
//! (二十 vs 三十) are not numeric tokens; an English entity in
//! sentence-initial position ("Acme makes…" vs "Globex makes…") hides
//! behind the capitalization exemption; entities in unsegmented
//! scripts carry no case marker at all. The cosine threshold is the
//! primary filter for all of these; the default posture is the safest
//! one — the tier is OFF until an operator sets the threshold.

use std::collections::HashMap;
use std::sync::Arc;

use taguru::deadline::Deadline;
use unicode_normalization::UnicodeNormalization;

use super::{AppState, CachedRetrieval, RetrievalKey};
use crate::embedding::similarity;
use crate::metrics::{RetrievalCacheOp, SemanticCacheOutcome};

/// One equivalence class's scope: everything of the exact key that is
/// NOT the query text. Two requests can only share a canonical when
/// their operation, their resolved target list (in effective order —
/// the same grants-resolve-alike posture the exact tier keys on), and
/// every other result-affecting parameter agree exactly.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct SemanticBucket {
    op: RetrievalCacheOp,
    targets: Box<[String]>,
    /// The handler's params serialization WITHOUT the query field.
    params: String,
}

impl SemanticBucket {
    fn of(key: &RetrievalKey, sans_query_params: &str) -> Self {
        Self {
            op: key.op,
            targets: key.targets.iter().map(|t| t.name.clone()).collect(),
            params: sans_query_params.to_string(),
        }
    }
}

/// One canonical query: its raw text (the guard's comparand), its
/// embedding (the `Arc` shared with the cue cache — no copy), and the
/// FULL exact-cache params its results were filed under.
struct SemanticSlot {
    query: String,
    embedding: Arc<Vec<f32>>,
    params: String,
    tick: u64,
}

/// A bucket scan's yield, similarity descending: the caller walks it
/// and the first candidate that passes [`queries_agree`] decides the
/// outcome (a live rewritten entry serves; a stale one falls through
/// to a fresh fill that will heal the cluster).
struct Candidate {
    query: String,
    params: String,
    similarity: f32,
}

/// The claims index: `TAGURU_SEMANTIC_CACHE_THRESHOLD` (unset =
/// disabled, everything no-ops) plus the bucketed slots under a
/// [`super::CueCache`]-style tick-LRU — same cap, same
/// scan-for-the-oldest eviction, fine at a fixed size for the same
/// reason.
pub(crate) struct SemanticCache {
    /// The cosine floor for query-vs-query equivalence, in the same
    /// `[0, 1]`-clamped space `effective_semantic_floor` uses
    /// ([`crate::env::env_floor`] enforces the range). `None` = off.
    threshold: Option<f32>,
    buckets: HashMap<SemanticBucket, Vec<SemanticSlot>>,
    tick: u64,
}

impl SemanticCache {
    const CAP: usize = 1024;

    pub(crate) fn new(threshold: Option<f32>) -> Self {
        Self {
            threshold,
            buckets: HashMap::new(),
            tick: 0,
        }
    }

    fn len(&self) -> usize {
        self.buckets.values().map(Vec::len).sum()
    }

    /// Every slot in the bucket at or above the threshold, similarity
    /// descending, recency-touched (a candidate consulted is a
    /// candidate in use, whatever the guard later says).
    fn candidates(&mut self, bucket: &SemanticBucket, embedding: &[f32]) -> Vec<Candidate> {
        let Some(threshold) = self.threshold else {
            return Vec::new();
        };
        self.tick += 1;
        let tick = self.tick;
        let Some(slots) = self.buckets.get_mut(bucket) else {
            return Vec::new();
        };
        let mut found: Vec<Candidate> = Vec::new();
        for slot in slots {
            let cosine = similarity(&slot.embedding, embedding);
            if cosine >= threshold {
                slot.tick = tick;
                found.push(Candidate {
                    query: slot.query.clone(),
                    params: slot.params.clone(),
                    similarity: cosine,
                });
            }
        }
        found.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
        found
    }

    /// Files one fresh fill as its cluster's canonical: a slot the new
    /// query would itself have matched (threshold AND guard) is
    /// replaced outright — the newest fill's exact entry is the one
    /// most likely to be live, and re-registering the same query after
    /// a revision bump lands here too (cosine 1.0 against itself).
    /// Anything else opens a new cluster; guard-refused neighbors stay
    /// separate on purpose, that split IS the guard's verdict.
    fn register(
        &mut self,
        bucket: SemanticBucket,
        query: &str,
        embedding: &Arc<Vec<f32>>,
        params: &str,
    ) {
        let Some(threshold) = self.threshold else {
            return;
        };
        self.tick += 1;
        let tick = self.tick;
        let slot = SemanticSlot {
            query: query.to_string(),
            embedding: Arc::clone(embedding),
            params: params.to_string(),
            tick,
        };
        let slots = self.buckets.entry(bucket).or_default();
        if let Some(existing) = slots.iter_mut().find(|existing| {
            similarity(&existing.embedding, embedding) >= threshold
                && queries_agree(&existing.query, query)
        }) {
            *existing = slot;
            return;
        }
        slots.push(slot);
        if self.len() > Self::CAP {
            self.evict_stalest();
        }
    }

    fn evict_stalest(&mut self) {
        let Some((bucket, at)) = self
            .buckets
            .iter()
            .flat_map(|(bucket, slots)| {
                slots
                    .iter()
                    .enumerate()
                    .map(move |(at, slot)| (slot.tick, bucket, at))
            })
            .min_by_key(|&(tick, ..)| tick)
            .map(|(_, bucket, at)| (bucket.clone(), at))
        else {
            return;
        };
        let slots = self.buckets.get_mut(&bucket).expect("bucket was just seen");
        // Within-bucket order carries no meaning (candidates sort by
        // similarity), so the cheap removal is fine.
        slots.swap_remove(at);
        if slots.is_empty() {
            self.buckets.remove(&bucket);
        }
    }
}

/// What one probe hands back to the handler: the query's embedding
/// (for [`AppState::semantic_register`] after a fresh fill — computing
/// it is the probe's one real cost, never paid twice), and the served
/// entry when a claim held and its rewritten key was live.
pub(crate) struct SemanticProbe {
    pub embedding: Arc<Vec<f32>>,
    pub served: Option<SemanticServe>,
}

/// A semantic hit: the canonical's cached entry plus what the search
/// log line reports about the match itself — the two numbers an
/// operator tunes the threshold with.
pub(crate) struct SemanticServe {
    pub value: CachedRetrieval,
    pub canonical: String,
    pub similarity: f32,
}

/// The fresh path's registration material, threaded through
/// `cache_and_serve` so a fill and its claim cannot drift apart.
pub(crate) struct SemanticFill {
    /// The bucket half of the params (query stripped).
    pub params: String,
    pub query: String,
    pub embedding: Arc<Vec<f32>>,
}

impl AppState {
    /// One semantic consultation, run only after the request's own
    /// exact key missed. `None` means the tier did not run at all (no
    /// threshold configured, the passage embedding lane is off, the
    /// query is blank, or the provider refused the embedding) — no
    /// outcome is counted and nothing registers later, mirroring the
    /// exact tier's silence when disabled. `Some` always carries the
    /// embedding for the fill path; `served` is the hit.
    ///
    /// Outcomes, counted exactly once per probe: `hit` (a claim held
    /// and its rewritten key was live), `stale` (a claim held but the
    /// corpus moved on — the fresh fill about to happen re-canonicalizes
    /// the cluster), `guarded` (cosine cleared the threshold but the
    /// text guard refused — the tuning signal), `miss` (no candidate
    /// cleared the threshold). The rewritten lookup is deliberately
    /// NOT counted in `taguru_retrieval_cache_total`: one request must
    /// not read as two exact consultations.
    ///
    /// The walk stops at the first guard-passing candidate even if its
    /// entry is stale — lower-cosine siblings are usually the same
    /// cluster's older claims, and the fill this stale falls through
    /// to replaces them anyway.
    pub(crate) fn semantic_retrieval(
        &self,
        key: &RetrievalKey,
        sans_query_params: &str,
        query: &str,
        deadline: Deadline,
    ) -> Option<SemanticProbe> {
        self.0.semantic_cache.lock().threshold?;
        if !self.passage_embedding_enabled() || query.trim().is_empty() {
            return None;
        }
        let embedder = self.0.embedder.clone().expect("enabled implies a provider");
        let embedding = self.cue_vector(&*embedder, query, deadline).ok()?;
        let bucket = SemanticBucket::of(key, sans_query_params);
        let candidates = self.0.semantic_cache.lock().candidates(&bucket, &embedding);
        if candidates.is_empty() {
            self.0
                .metrics
                .record_semantic_cache(SemanticCacheOutcome::Miss);
            return Some(SemanticProbe {
                embedding,
                served: None,
            });
        }
        for candidate in candidates {
            if !queries_agree(query, &candidate.query) {
                continue;
            }
            let rewritten = RetrievalKey {
                op: key.op,
                targets: key.targets.clone(),
                params: candidate.params,
            };
            return Some(match self.retrieval_lookup_uncounted(&rewritten) {
                Some(value) => {
                    self.0
                        .metrics
                        .record_semantic_cache(SemanticCacheOutcome::Hit);
                    SemanticProbe {
                        embedding,
                        served: Some(SemanticServe {
                            value,
                            canonical: candidate.query,
                            similarity: candidate.similarity,
                        }),
                    }
                }
                None => {
                    self.0
                        .metrics
                        .record_semantic_cache(SemanticCacheOutcome::Stale);
                    SemanticProbe {
                        embedding,
                        served: None,
                    }
                }
            });
        }
        self.0
            .metrics
            .record_semantic_cache(SemanticCacheOutcome::Guarded);
        Some(SemanticProbe {
            embedding,
            served: None,
        })
    }

    /// Files a fresh fill's equivalence claim next to its exact-cache
    /// store. `key` is the fill's own exact key — its full params are
    /// what a future rewrite must reproduce.
    pub(crate) fn semantic_register(&self, key: &RetrievalKey, fill: SemanticFill) {
        let bucket = SemanticBucket::of(key, &fill.params);
        self.0
            .semantic_cache
            .lock()
            .register(bucket, &fill.query, &fill.embedding, &key.params);
    }

    /// The scrape-time gauge.
    pub(crate) fn semantic_cache_entries(&self) -> u64 {
        self.0.semantic_cache.lock().len() as u64
    }
}

/// English negation cue words, compared as whole (case-folded) tokens.
/// "cannot" vs "can not" land in different bins and refuse — a false
/// refusal is only a miss.
const NEGATION_WORDS: [&str; 9] = [
    "no", "not", "never", "none", "neither", "nor", "without", "cannot", "non",
];

/// Substring-counted negation cues: `n't` for English contractions
/// (typographic apostrophe folded first), plus a minimal Japanese list
/// — this deployment's queries are Japanese-heavy, and an English-only
/// guard would false-negative exactly where it matters most. 不/無 are
/// common in non-negating compounds too (不動産, 無事); over-matching
/// only splits clusters, which is the safe direction.
const NEGATION_SUBSTRINGS: [&str; 7] = ["n't", "ない", "ません", "なかった", "せず", "不", "無"];

/// The go/no-go text guard between two queries whose embeddings
/// already cleared the cosine threshold: no negation, numeric, or
/// entity-ish asymmetry may separate them. Symmetric; any difference
/// refuses. See the module doc for the documented blind spots.
fn queries_agree(a: &str, b: &str) -> bool {
    let (fa, fb) = (folded(a), folded(b));
    negation_marks(&fa) == negation_marks(&fb)
        && digit_runs(&fa) == digit_runs(&fb)
        && entityish_tokens(&fa) == entityish_tokens(&fb)
}

/// NFKC-folded view (full-width digits and letters become ASCII, so
/// ２０ and 20 are the same number), typographic apostrophe folded so
/// don’t and don't carry the same `n't`.
fn folded(query: &str) -> String {
    query.nfkc().collect::<String>().replace('\u{2019}', "'")
}

/// Per-cue occurrence counts, in one fixed order.
fn negation_marks(folded: &str) -> Vec<usize> {
    let lower = folded.to_lowercase();
    let mut marks = Vec::with_capacity(NEGATION_WORDS.len() + NEGATION_SUBSTRINGS.len());
    for word in NEGATION_WORDS {
        marks.push(
            lower
                .split(|ch: char| !ch.is_alphanumeric())
                .filter(|&token| token == word)
                .count(),
        );
    }
    for cue in NEGATION_SUBSTRINGS {
        marks.push(lower.matches(cue).count());
    }
    marks
}

/// The sorted multiset of maximal ASCII digit runs. "v2" contributes
/// "2"; "01" and "1" differ on purpose (a version is not its numeric
/// value).
fn digit_runs(folded: &str) -> Vec<String> {
    let mut runs = Vec::new();
    let mut current = String::new();
    for ch in folded.chars() {
        if ch.is_ascii_digit() {
            current.push(ch);
        } else if !current.is_empty() {
            runs.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        runs.push(current);
    }
    runs.sort();
    runs
}

/// The sorted, case-folded multiset of capitalization-marked tokens —
/// the "named-entity-ish" signal. The sentence-initial token is exempt
/// unless it is an all-caps acronym ("Does…" is position, "NASA…" is a
/// name); everything later with an uppercase letter counts.
fn entityish_tokens(folded: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for (position, token) in folded
        .split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .enumerate()
    {
        if !token.chars().any(|ch| ch.is_ascii_uppercase()) {
            continue;
        }
        if position == 0
            && !(token.chars().count() >= 2 && token.chars().all(|ch| ch.is_ascii_uppercase()))
        {
            continue;
        }
        tokens.push(token.to_lowercase());
    }
    tokens.sort();
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------- the guard: #153's verify checkboxes as pure-function
    // pins, false-refusal (safe) cases next to each catch.

    #[test]
    fn a_paraphrase_with_no_tripwires_agrees() {
        assert!(queries_agree(
            "does the mill mass-produce oysters",
            "is the mill mass-producing oysters"
        ));
        // Sentence-initial capitalization is position, not identity.
        assert!(queries_agree("Does the mill run", "is the mill running"));
        assert!(queries_agree("川舟の仕事は何か", "川舟は何の仕事をするか"));
    }

    #[test]
    fn a_negation_flip_refuses_in_english_and_japanese() {
        assert!(!queries_agree(
            "does it mass-produce",
            "does it not mass-produce"
        ));
        assert!(!queries_agree("is it produced", "isn't it produced"));
        assert!(!queries_agree("it works", "it never works"));
        assert!(!queries_agree("弟は死んだか", "弟は死ななかったか"));
        assert!(!queries_agree("許可がある", "許可がない"));
        // Typographic apostrophe carries the same n't.
        assert!(!queries_agree("is it produced", "isn\u{2019}t it produced"));
        // Negation on both sides in the same form agrees — the guard
        // compares, it does not detect.
        assert!(queries_agree(
            "why is it not produced",
            "why was it not produced"
        ));
        // ...but a form change ("not" vs "n't") lands in different
        // bins and refuses: a false refusal is only a miss, and
        // "cannot"/"can not" need the same conservatism anyway.
        assert!(!queries_agree(
            "why is it not produced",
            "why isn't it produced"
        ));
    }

    #[test]
    fn a_changed_number_refuses_and_a_folded_width_does_not() {
        assert!(!queries_agree(
            "did it sell 20 units",
            "did it sell 30 units"
        ));
        assert!(!queries_agree("is v2 stable", "is v3 stable"));
        assert!(!queries_agree("in 2024", "in 2025"));
        // NFKC folds full-width digits: same number, same guard view.
        assert!(queries_agree(
            "did it sell ２０ units",
            "did it sell 20 units"
        ));
        // Leading zeros are a different spelling of a different thing.
        assert!(!queries_agree("chapter 01", "chapter 1"));
    }

    #[test]
    fn a_swapped_entity_refuses_where_capitalization_marks_it() {
        assert!(!queries_agree(
            "does Acme mass-produce widgets",
            "does Globex mass-produce widgets"
        ));
        // An all-caps sentence-initial acronym is a name, not position.
        assert!(!queries_agree(
            "NASA launch schedule",
            "ESA launch schedule"
        ));
        // The same entity in both queries agrees, case-folded.
        assert!(queries_agree(
            "when did Acme ship widgets",
            "when did ACME ship widgets"
        ));
    }

    // ------- the index.

    fn bucket(tag: &str) -> SemanticBucket {
        SemanticBucket {
            op: RetrievalCacheOp::SearchPassages,
            targets: Box::new(["c".to_string()]),
            params: tag.to_string(),
        }
    }

    fn unit(x: f32, y: f32) -> Arc<Vec<f32>> {
        let norm = (x * x + y * y).sqrt();
        Arc::new(vec![x / norm, y / norm])
    }

    #[test]
    fn candidates_come_back_similarity_descending_and_floored() {
        let mut cache = SemanticCache::new(Some(0.9));
        // Both clear the 0.9 floor against the probe below, but sit
        // ~32° apart from EACH OTHER (cosine ≈ 0.85), so registration
        // keeps them as two clusters instead of collapsing them.
        cache.register(bucket("p"), "close", &unit(0.9135, 0.4067), "params-close");
        cache.register(
            bucket("p"),
            "closer",
            &unit(0.9903, -0.1392),
            "params-closer",
        );
        cache.register(bucket("p"), "far away", &unit(0.0, 1.0), "params-far");
        let found = cache.candidates(&bucket("p"), &unit(1.0, 0.0));
        assert_eq!(
            found.iter().map(|c| c.query.as_str()).collect::<Vec<_>>(),
            ["closer", "close"],
            "the orthogonal slot never surfaces"
        );
        assert!(found[0].similarity > found[1].similarity);
        // A different bucket is a different scope: nothing crosses.
        assert!(cache.candidates(&bucket("q"), &unit(1.0, 0.0)).is_empty());
    }

    #[test]
    fn registering_an_equivalent_query_replaces_its_cluster_canonical() {
        let mut cache = SemanticCache::new(Some(0.9));
        cache.register(
            bucket("p"),
            "the old wording",
            &unit(1.0, 0.0),
            "old-params",
        );
        cache.register(
            bucket("p"),
            "the new wording",
            &unit(0.99, 0.1),
            "new-params",
        );
        assert_eq!(cache.len(), 1, "one cluster, one canonical");
        let found = cache.candidates(&bucket("p"), &unit(1.0, 0.0));
        assert_eq!(found[0].query, "the new wording");
        assert_eq!(found[0].params, "new-params");
    }

    #[test]
    fn a_guard_refused_neighbor_opens_its_own_cluster() {
        let mut cache = SemanticCache::new(Some(0.9));
        cache.register(bucket("p"), "does it produce", &unit(1.0, 0.0), "a");
        // Same direction, negated: cosine says same, guard says split.
        cache.register(bucket("p"), "does it not produce", &unit(1.0, 0.0), "b");
        assert_eq!(cache.len(), 2, "negation stays a separate claim");
    }

    #[test]
    fn the_cap_evicts_the_stalest_slot_across_buckets() {
        let mut cache = SemanticCache::new(Some(0.9));
        for at in 0..=SemanticCache::CAP {
            cache.register(
                bucket(&format!("scope-{at}")),
                "q",
                &unit(1.0, 0.0),
                "params",
            );
        }
        assert_eq!(cache.len(), SemanticCache::CAP);
        assert!(
            cache
                .candidates(&bucket("scope-0"), &unit(1.0, 0.0))
                .is_empty(),
            "the first-registered slot is the one evicted"
        );
        assert!(
            !cache
                .candidates(&bucket("scope-1"), &unit(1.0, 0.0))
                .is_empty()
        );
    }

    #[test]
    fn an_unset_threshold_disables_everything() {
        let mut cache = SemanticCache::new(None);
        cache.register(bucket("p"), "q", &unit(1.0, 0.0), "params");
        assert_eq!(cache.len(), 0);
        assert!(cache.candidates(&bucket("p"), &unit(1.0, 0.0)).is_empty());
    }
}
