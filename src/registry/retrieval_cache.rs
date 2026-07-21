//! The exact-match retrieval cache: an identical request against an
//! identical corpus state answers from the previously computed
//! response instead of re-running the search (issue #150). "Identical"
//! is literal — same operation, same resolved target list in the same
//! order, same parameters — so there is no similarity judgment and no
//! way to serve a plausible-but-wrong answer for a differently-worded
//! question.
//!
//! Invalidation is the key, not a mechanism: every key carries, per
//! target, the context's identity nonce and the pair of revision lanes
//! the operation's response can depend on ([`op_lanes`]). A write
//! bumps a lane, future lookups compute a different key, and the stale
//! entry is unreachable until the LRU evicts it — no invalidation
//! walk, no TTL, and no window where an invalidation races a fill.
//! Two orderings make that sound:
//!
//! - The key is read BEFORE the search runs ([`AppState::retrieval_key`]).
//!   Revisions bump after publish (see [`super::ContextRevision`]), so
//!   key-then-search keeps every cached entry's content at least as
//!   new as its key — a reader whose own key computation observed
//!   revision R can only ever hit content from R or later, the same
//!   ladder the revision counters promise everywhere else.
//! - Revisions restart at zero when a name is deleted and recreated,
//!   so the key also carries the entry's identity
//!   ([`super::EntryInner::cache_identity`]): process-unique per
//!   incarnation, re-minted when a replica's tailer refreshes an entry
//!   in place (the one path where content can switch lineage while
//!   `fetch_max` holds the revision counters still). An old
//!   incarnation's entries become unreachable the instant the new one
//!   registers, with no purge hook anywhere.
//!
//! The bound is a byte budget with the cue cache's tick-LRU shape —
//! bytes rather than the cue cache's entry count because response
//! payloads vary from a few hundred bytes to half a megabyte, where
//! cue vectors are all the same size.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use serde_json::value::RawValue;

use super::{AppState, ContextRevision};
use crate::metrics::RetrievalCacheOp;

/// Default byte budget (`TAGURU_RETRIEVAL_CACHE_BYTES` overrides;
/// `0` disables the cache entirely).
pub(crate) const DEFAULT_RETRIEVAL_CACHE_BYTES: usize = 32 * 1024 * 1024;

/// One target's contribution to a cache key: which context, which
/// incarnation of it, and the values of the two revision lanes this
/// operation's response can depend on.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct TargetFingerprint {
    pub name: String,
    /// [`super::EntryInner::cache_identity`] at key time.
    pub identity: u64,
    /// [`op_lanes`]' pair, in that fixed per-op order.
    pub lanes: [u64; 2],
}

/// The whole key: operation, the resolved target list in effective
/// search order, and every other result-affecting parameter serialized
/// canonically. The list order is part of the key on purpose — the
/// cross passage merge breaks rank ties by target-list order, so two
/// orderings of the same set are two different results.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) struct RetrievalKey {
    pub op: RetrievalCacheOp,
    pub targets: Box<[TargetFingerprint]>,
    pub params: String,
}

/// What a hit replays: the response payload byte-for-byte, plus the
/// observable side effects the fresh path would have produced — usage
/// and search counters per target, passage lane contributions, and the
/// numbers the opt-in search log line reports. Metrics describe served
/// responses; the cache only changes how a response was computed.
#[derive(Clone)]
pub(crate) struct CachedRetrieval {
    pub payload: Arc<RawValue>,
    /// Aligned with the key's targets: whether each target's own
    /// search came back empty — `note_search`'s replay input.
    pub target_empty: Box<[bool]>,
    /// `(bm25_only, both_lanes, vector_only)` tallies for the passage
    /// ops' lane-contribution replay; zeros for the graph ops.
    pub lane_hits: [u64; 3],
    /// The hit count the search log line reports.
    pub log_hits: usize,
    /// The passage log line's `top_score`; 0.0 for the graph ops.
    pub log_top_score: f32,
}

struct CacheSlot {
    /// The same `Arc` the maps key on — carried here so a recency bump
    /// can re-file the entry under its new tick without cloning any
    /// key material.
    key: Arc<RetrievalKey>,
    value: CachedRetrieval,
    cost: usize,
    tick: u64,
}

/// The map itself: byte-budgeted, recency-tracked by a tick counter
/// like `CueCache` (a dedicated counter, not `AppState::clock`, for
/// the same separation-of-concerns reason). Unlike `CueCache`'s
/// scan-for-the-oldest eviction — fine at a fixed 1024 entries — the
/// budget here is operator-configurable, and revision churn parks
/// unreachable entries as evictable dead weight, so eviction order is
/// kept as a side index: `order` maps tick → key, the two maps move in
/// lockstep, and evicting the stalest entry is a first-key pop instead
/// of a scan. Ticks are unique (one per cache operation), so the
/// `BTreeMap` never collides.
pub(crate) struct RetrievalCache {
    entries: HashMap<Arc<RetrievalKey>, CacheSlot>,
    /// tick → key, ascending = stalest first. Invariant: exactly one
    /// row per resident slot, at that slot's current `tick`.
    order: BTreeMap<u64, Arc<RetrievalKey>>,
    /// Sum of every resident slot's `cost`.
    bytes: usize,
    /// `0` = disabled: `lookup` and `insert` both no-op, and
    /// [`AppState::retrieval_key`] refuses to mint keys at all so the
    /// hit/miss counters stay silent.
    budget: usize,
    tick: u64,
}

impl RetrievalCache {
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: BTreeMap::new(),
            bytes: 0,
            budget,
            tick: 0,
        }
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.budget != 0
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    fn lookup(&mut self, key: &RetrievalKey) -> Option<CachedRetrieval> {
        self.tick += 1;
        let tick = self.tick;
        let slot = self.entries.get_mut(key)?;
        self.order.remove(&slot.tick);
        slot.tick = tick;
        self.order.insert(tick, Arc::clone(&slot.key));
        Some(slot.value.clone())
    }

    fn insert(&mut self, key: RetrievalKey, value: CachedRetrieval) {
        if !self.is_enabled() {
            return;
        }
        let cost = slot_cost(&key, &value);
        // One giant page must not evict the whole population to seat
        // itself: past a quarter of the budget it is served uncached.
        if cost > self.budget / 4 {
            return;
        }
        // Replacing (two identical misses racing) re-accounts the old
        // slot's bytes; distinct keys evict stalest-first until the new
        // slot fits.
        if let Some(previous) = self.entries.remove(&key) {
            self.bytes -= previous.cost;
            self.order.remove(&previous.tick);
        }
        while self.bytes + cost > self.budget {
            let Some((_, stalest)) = self.order.pop_first() else {
                break;
            };
            if let Some(evicted) = self.entries.remove(stalest.as_ref()) {
                self.bytes -= evicted.cost;
            }
        }
        self.tick += 1;
        self.bytes += cost;
        let key = Arc::new(key);
        self.order.insert(self.tick, Arc::clone(&key));
        self.entries.insert(
            Arc::clone(&key),
            CacheSlot {
                key,
                value,
                cost,
                tick: self.tick,
            },
        );
    }
}

/// A slot's byte accounting: the serialized payload dominates; the key
/// strings and a fixed per-entry overhead cover the map's own weight
/// approximately — the budget is a resource bound, not an audit.
fn slot_cost(key: &RetrievalKey, value: &CachedRetrieval) -> usize {
    let key_bytes: usize = key
        .targets
        .iter()
        .map(|target| target.name.len() + 24)
        .sum::<usize>()
        + key.params.len();
    value.payload.get().len() + key_bytes + 64
}

/// The pair of revision lanes an operation's response can depend on —
/// the whole point of keeping three counters instead of one scalar.
fn op_lanes(op: RetrievalCacheOp, revision: ContextRevision) -> [u64; 2] {
    match op {
        // Graph matches, plus section labels resolved through the
        // passage store (`resolve_sections` enriches every
        // association-bearing response) — config writes (a PATCH, an
        // embedding refresh) change neither.
        RetrievalCacheOp::Recall | RetrievalCacheOp::Query => [revision.graph, revision.passages],
        // Passage text and BM25 ride the passages lane; the vector
        // lane's published embeddings and the context's own
        // `semantic_floor` are both config — graph writes change
        // neither. Community search reads the DERIVED context's
        // summaries through the same two lanes; its dependence on the
        // SOURCE context's graph is carried in the key params (the
        // source's current graph revision), not in a lane.
        RetrievalCacheOp::SearchPassages | RetrievalCacheOp::SearchCommunities => {
            [revision.passages, revision.config]
        }
    }
}

impl AppState {
    /// Mints the cache key for one retrieval request, or `None` when
    /// the response must be computed fresh and served uncounted: the
    /// cache is disabled, the parameters would not serialize, or a
    /// named target vanished between resolution and now (the fresh
    /// path answers that race's error).
    ///
    /// MUST run BEFORE the search: revisions bump after publish, so
    /// key-then-search bounds every cached entry's content from below
    /// by its key. The reverse order could pair a fresh revision with
    /// a search that ran before the write it names published.
    pub(crate) fn retrieval_key(
        &self,
        op: RetrievalCacheOp,
        targets: &[String],
        params: Option<String>,
    ) -> Option<RetrievalKey> {
        if !self.0.retrieval_cache.lock().is_enabled() {
            return None;
        }
        let params = params?;
        let targets = targets
            .iter()
            .map(|name| {
                let entry = self.lookup(name)?;
                let inner = entry.read_unless_deleted()?;
                Some(TargetFingerprint {
                    name: name.clone(),
                    identity: inner.cache_identity,
                    lanes: op_lanes(op, entry.revision_snapshot(&inner)),
                })
            })
            .collect::<Option<Box<[_]>>>()?;
        Some(RetrievalKey {
            op,
            targets,
            params,
        })
    }

    /// One cache consultation, counted hit or miss.
    pub(crate) fn retrieval_lookup(&self, key: &RetrievalKey) -> Option<CachedRetrieval> {
        let found = self.0.retrieval_cache.lock().lookup(key);
        self.0
            .metrics
            .record_retrieval_cache(key.op, found.is_some());
        found
    }

    /// The semantic tier's delegated read of a rewritten key: same
    /// map, same recency touch, NOT counted in
    /// `taguru_retrieval_cache_total` — one request must not read as
    /// two exact consultations; `taguru_semantic_cache_total` owns
    /// this outcome (see [`AppState::semantic_retrieval`]).
    pub(crate) fn retrieval_lookup_uncounted(&self, key: &RetrievalKey) -> Option<CachedRetrieval> {
        self.0.retrieval_cache.lock().lookup(key)
    }

    /// Files a computed response under its key. Only full successes
    /// arrive here — error responses are never cached.
    pub(crate) fn retrieval_store(&self, key: RetrievalKey, value: CachedRetrieval) {
        self.0.retrieval_cache.lock().insert(key, value);
    }

    /// `(entries, bytes)` for the scrape-time gauges.
    pub(crate) fn retrieval_cache_gauges(&self) -> (u64, u64) {
        let cache = self.0.retrieval_cache.lock();
        (cache.len() as u64, cache.bytes() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::CueCache;

    fn key(params: &str) -> RetrievalKey {
        RetrievalKey {
            op: RetrievalCacheOp::Recall,
            targets: Box::new([TargetFingerprint {
                name: "c".to_string(),
                identity: 1,
                lanes: [0, 0],
            }]),
            params: params.to_string(),
        }
    }

    fn value(payload_bytes: usize) -> CachedRetrieval {
        let text = "x".repeat(payload_bytes);
        CachedRetrieval {
            payload: serde_json::value::to_raw_value(&text).unwrap().into(),
            target_empty: Box::new([false]),
            lane_hits: [0, 0, 0],
            log_hits: 1,
            log_top_score: 0.0,
        }
    }

    #[test]
    fn eviction_drops_the_least_recently_used_entry_and_keeps_the_bytes_honest() {
        // The quarter-of-budget guard means at least four equal slots
        // always fit before eviction can fire — so budget four and a
        // half slots and let the fifth insert force the choice.
        let cost = slot_cost(&key("a"), &value(400));
        let mut cache = RetrievalCache::new(cost * 4 + cost / 2);
        for params in ["a", "b", "c", "d"] {
            cache.insert(key(params), value(400));
        }
        assert_eq!(cache.len(), 4);
        // Touch everything but "b", making it the eviction candidate.
        for touched in ["a", "c", "d"] {
            assert!(cache.lookup(&key(touched)).is_some());
        }
        cache.insert(key("e"), value(400));
        assert!(cache.lookup(&key("b")).is_none(), "stalest entry evicted");
        for live in ["a", "c", "d", "e"] {
            assert!(cache.lookup(&key(live)).is_some(), "'{live}' survives");
        }
        assert_eq!(cache.bytes(), cost * 4, "bytes track the survivors");
        assert_eq!(
            cache.order.len(),
            cache.entries.len(),
            "the eviction-order index stays in lockstep"
        );
    }

    #[test]
    fn an_entry_past_a_quarter_of_the_budget_is_served_uncached() {
        let mut cache = RetrievalCache::new(2000);
        cache.insert(key("big"), value(600));
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.bytes(), 0);
    }

    #[test]
    fn replacing_an_entry_reaccounts_its_bytes_and_order_row() {
        let mut cache = RetrievalCache::new(10_000);
        cache.insert(key("a"), value(400));
        cache.insert(key("a"), value(100));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.bytes(), slot_cost(&key("a"), &value(100)));
        // A dangling order row would make a later eviction pop the
        // LIVE slot in the stale row's place — the replace must drop it.
        assert_eq!(cache.order.len(), 1, "exactly one order row survives");
    }

    #[test]
    fn a_zero_budget_disables_the_cache() {
        let mut cache = RetrievalCache::new(0);
        assert!(!cache.is_enabled());
        cache.insert(key("a"), value(10));
        assert!(cache.lookup(&key("a")).is_none());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn cue_cache_get_promotes_recency_so_eviction_spares_the_touched_entry() {
        let mut cache = CueCache::default();
        for i in 0..CueCache::CAP {
            cache.insert(format!("cue{i}"), Arc::new(vec![i as f32]));
        }
        // Touching cue0 makes it the most recently used entry, even
        // though it was the first one inserted.
        assert!(cache.get("cue0").is_some());
        // The cache is at capacity, so this eviction must reach for the
        // least recently used entry — cue1, never touched again after
        // its insert — not the oldest insertion, which is cue0.
        cache.insert("fresh-cue".to_string(), Arc::new(vec![-1.0]));
        assert!(
            cache.get("cue0").is_some(),
            "a touched entry must survive the next eviction"
        );
        assert!(
            cache.get("cue1").is_none(),
            "the least recently used entry must be the one evicted"
        );
        assert!(cache.get("fresh-cue").is_some());
    }

    #[test]
    fn cue_cache_insert_does_not_overwrite_an_existing_key() {
        let mut cache = CueCache::default();
        cache.insert("cue".to_string(), Arc::new(vec![1.0]));
        cache.insert("cue".to_string(), Arc::new(vec![2.0]));
        assert_eq!(*cache.get("cue").unwrap(), vec![1.0]);
    }
}
