use std::collections::HashMap;
use std::io;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use taguru::deadline::{Deadline, DeadlineExceeded};

use super::{
    AppState, ContextMeta, Entry, FusedHit, PassageExplainLookup, PassageSearch,
    PassageSearchExplanation, PassageSearchHit, PassageSearchLanes, PassageVectorGate,
    VectorLaneReport, VectorLaneStatus, bm25_path, file_stem, passage_terms, spelled_passage_terms,
};

impl AppState {
    /// Full-text search over the registered passages — the second lane
    /// beside the graph, for knowledge that does not decompose into
    /// triples (procedures, conditions, discourse). Scored per
    /// PARAGRAPH: the ranking unit is what an answer actually cites,
    /// and a long document no longer buries its best paragraph inside
    /// its own length normalization.
    ///
    /// Two ranking lanes, fused: the lexical one runs on the resident
    /// [`crate::bm25::Bm25Index`] (built once per residency, updated in
    /// place by store/retract — a query never re-tokenizes the corpus),
    /// and, where paragraph embedding is on, a semantic one sweeps the
    /// paragraph vectors with the query embedded as
    /// [`EmbedPurpose::Query`]. Fusion is reciprocal rank (k = 60):
    /// rank-based, so the two lanes' incomparable score scales never
    /// need reconciling. Each hit reports its per-lane rank and raw
    /// score — the server presents evidence, the reading LLM judges,
    /// same as resolve's tiers. With the vector lane off or its
    /// provider failing, results degrade to pure BM25 (raw score,
    /// evidence intact) — a broken decoration never breaks the answer.
    ///
    /// Texts resolve through the store afterwards, hash-checked: a
    /// paragraph that changed between a lane's view and now is dropped
    /// rather than served with a stale score against fresh text.
    ///
    /// `floor_override` is the one-call override of the vector lane's
    /// cosine floor — the same chain resolve's `semantic_floor` walks:
    /// override beats context setting beats server default. It floors
    /// the only lane with an absolute scale; the fused score is rank
    /// arithmetic (raw BM25 in lexical-only deployments) and carries
    /// no floorable meaning.
    ///
    /// `filter` is the pre-lane source filter (#167): the eligible
    /// source set is computed ONCE from the store's metadata, before
    /// either lane runs, and both lanes serve only from it — BM25
    /// skips ineligible slots, the vector sweep skips ineligible rows
    /// (its ANN probe widening until the oversample target is met
    /// among eligible rows). The returned `filter` report carries the
    /// eligible/total counts the response plan surfaces. Collection
    /// statistics stay corpus-global — see [`crate::bm25::Bm25Index::search`].
    pub fn search_passages(
        &self,
        name: &str,
        query: &str,
        limit: usize,
        floor_override: Option<f32>,
        filter: Option<&crate::passages::SourceFilter>,
        deadline: Deadline,
    ) -> Option<io::Result<PassageSearch>> {
        let entry = self.lookup(name)?;
        if limit == 0 {
            return entry.read_unless_deleted().map(|_| {
                Ok(PassageSearch {
                    hits: Vec::new(),
                    lanes: PassageSearchLanes::ZeroLimit,
                    filter: None,
                })
            });
        }
        let query_grams = deduped_query_grams(query);
        if query_grams.is_empty() {
            return entry.read_unless_deleted().map(|_| {
                Ok(PassageSearch {
                    hits: Vec::new(),
                    lanes: PassageSearchLanes::NoQueryTerms,
                    filter: None,
                })
            });
        }
        let pool = lane_pool(limit);

        // The semantic lane's query embedding runs BEFORE any lock: a
        // provider round trip must never extend the fence below.
        let cue = self.passage_query_cue(query, deadline);
        if let Err(error) = &cue {
            // Degrade, loudly: the lexical lane still answers, and the
            // plan hands the caller the same account this line logs.
            tracing::warn!(
                context = %name,
                error,
                "passage query embedding failed; serving the lexical lane alone"
            );
        }

        // Everything below holds the read fence: eviction and deletion
        // are excluded for the whole search, so `store` IS the resident
        // store throughout — an index built from a handle that predates
        // an eviction would silently hide writes that landed in the
        // freshly reloaded store until the next rebuild.
        let fence = entry.read_unless_deleted()?;
        let store = match self.entry_passages(&entry, &file_stem(name)) {
            Ok(store) => store,
            Err(error) => return Some(Err(error)),
        };

        // The eligibility set, decided once before either lane runs —
        // both lanes then serve from exactly this set, so they cannot
        // disagree about who was allowed to answer.
        let eligibility = filter.map(|filter| store.eligible_sources(filter));
        let eligible = eligibility.as_ref().map(|(set, _)| set);
        let filter_report = eligibility
            .as_ref()
            .map(|(set, total)| super::SourceFilterReport {
                eligible: set.len(),
                total: *total,
            });

        if self
            .ensure_bm25_index(&entry, &store, name, deadline)
            .is_err()
        {
            return Some(Err(io::Error::other(DeadlineExceeded)));
        }
        let lexical = {
            let guard = entry.bm25.read();
            let index = guard.as_ref().expect("index was just built");
            index.search(&query_grams, pool, eligible)
        };

        // Semantic lane: sweep the paragraph vectors with the
        // pre-embedded query, then drop candidates below the same floor
        // semantic_resolve applies to its own cosine matches — the
        // caller's one-call override beats the context setting beats
        // the server default (`fence` is already this entry's read
        // lock, taken above). Every arm also names what it did: the
        // status the handler serializes into the response's plan.
        let (semantic, vector): (Vec<(String, u32, u64, f32)>, VectorLaneStatus) = match &cue {
            Err(error) => (
                Vec::new(),
                VectorLaneStatus::QueryEmbeddingFailed(error.clone()),
            ),
            Ok(None) => (
                Vec::new(),
                VectorLaneStatus::Off {
                    provider_configured: self.0.embedder.is_some(),
                },
            ),
            Ok(Some(cue)) => match self.passage_vector_gate(&entry, &file_stem(name)) {
                // The gate checks the model NAME; the width can still
                // disagree (a dimensions setting changed behind a
                // stable name, #133). Swept anyway, every row would be
                // `similarity`'s silent 0.0 — an empty lane the plan
                // would then call "ran" — so the mismatch is named
                // instead, exactly like a model change.
                PassageVectorGate::Ready(vectors) if cue.len() != vectors.dim() => (
                    Vec::new(),
                    VectorLaneStatus::WidthChanged {
                        stored: vectors.dim(),
                        current: cue.len(),
                    },
                ),
                PassageVectorGate::Ready(vectors) => {
                    let floor = self.effective_semantic_floor(floor_override, &fence.meta);
                    (
                        semantic_lane_hits(
                            vectors.top_matches(cue, pool, deadline, eligible),
                            floor,
                        ),
                        VectorLaneStatus::Ran { floor },
                    )
                }
                // Unreachable in practice (a cue exists only when the
                // lane is on), kept for the same defensiveness as
                // explain's mapping.
                PassageVectorGate::Disabled => (
                    Vec::new(),
                    VectorLaneStatus::Off {
                        provider_configured: self.0.embedder.is_some(),
                    },
                ),
                PassageVectorGate::Empty => (Vec::new(), VectorLaneStatus::NoVectors),
                PassageVectorGate::ModelChanged { stored, current } => (
                    Vec::new(),
                    VectorLaneStatus::ModelChanged { stored, current },
                ),
            },
        };

        Some(Ok(PassageSearch {
            hits: fuse_passage_lanes(&store, lexical, semantic, limit),
            lanes: PassageSearchLanes::Ran { vector },
            filter: filter_report,
        }))
    }

    /// The whole account of one source (or one of its paragraphs)
    /// against one query — the same search re-run with nothing
    /// truncated, the target located in it, and each lane's reason
    /// when it has no evidence. `paragraph: None` settles on the
    /// source's best showing: its best-ranked paragraph, or the one
    /// sharing the most query terms when nothing ranked. `limit` is
    /// the search call being explained — the serve/cutoff boundary is
    /// recomputed exactly as `search_passages(limit)` computes it,
    /// pool caps included, `floor_override` too: an explanation under
    /// a different floor would account for a call nobody made.
    /// Read-only, and bounded like one normal query plus one targeted
    /// scoring.
    ///
    /// `filter` is the source filter of the search being explained
    /// (#167), applied exactly as `search_passages` applies it: an
    /// ineligible TARGET verdicts `filtered_out` before any scoring,
    /// and an eligible target is ranked against the eligible field
    /// only — an explanation against the unfiltered field would
    /// account for a call nobody made.
    // One parameter per field of the wire request being explained —
    // bundling them would just rename the same arity.
    #[allow(clippy::too_many_arguments)]
    pub fn explain_passage_search(
        &self,
        name: &str,
        query: &str,
        source: &str,
        paragraph: Option<u32>,
        limit: usize,
        floor_override: Option<f32>,
        filter: Option<&crate::passages::SourceFilter>,
        deadline: Deadline,
    ) -> Option<io::Result<PassageExplainLookup>> {
        let entry = self.lookup(name)?;
        let query_terms = deduped_spelled_query_terms(query);
        let query_grams: Vec<u64> = query_terms.iter().map(|&(_, gram)| gram).collect();

        // Mirror search_passages: the embedding runs only when a search
        // would have reached it, and before the fence.
        let cue = if query_grams.is_empty() {
            Ok(None)
        } else {
            self.passage_query_cue(query, deadline)
        };

        let fence = entry.read_unless_deleted()?;
        let store = match self.entry_passages(&entry, &file_stem(name)) {
            Ok(store) => store,
            Err(error) => return Some(Err(error)),
        };

        let Some(record) = store.get(source) else {
            return Some(Ok(PassageExplainLookup::UnknownSource));
        };
        let paragraphs = record.paragraphs.len();
        if let Some(index) = paragraph
            && (index as usize) >= paragraphs
        {
            return Some(Ok(PassageExplainLookup::IndexOutOfRange { paragraphs }));
        }
        if paragraphs == 0 {
            return Some(Ok(PassageExplainLookup::IndexOutOfRange { paragraphs }));
        }
        // The filter verdict sits between the target checks and the
        // lane verdicts: the target exists and is validly addressed,
        // but the search being explained never considered it.
        if let Some(filter) = filter
            && !filter.matches(&record)
        {
            return Some(Ok(PassageExplainLookup::FilteredOut));
        }
        if query_grams.is_empty() {
            return Some(Ok(PassageExplainLookup::NoQueryTerms));
        }

        let eligibility = filter.map(|filter| store.eligible_sources(filter));
        let eligible = eligibility.as_ref().map(|(set, _)| set);

        if self
            .ensure_bm25_index(&entry, &store, name, deadline)
            .is_err()
        {
            return Some(Err(io::Error::other(DeadlineExceeded)));
        }
        let guard = entry.bm25.read();
        let index = guard.as_ref().expect("index was just built");

        // Sweep both lanes whole, once: a lane's pool cap is a prefix
        // of the same deterministically ordered sweep, so every pool
        // size fusion needs below is a `take` away.
        let lexical_full = index.search(&query_grams, usize::MAX, eligible);
        let floor = self.effective_semantic_floor(floor_override, &fence.meta);
        let gate = match &cue {
            Ok(Some(_)) => Some(self.passage_vector_gate(&entry, &file_stem(name))),
            _ => None,
        };
        let vector_rows: Vec<(String, u32, u64, f32)> = match (&cue, &gate) {
            (Ok(Some(cue)), Some(PassageVectorGate::Ready(vectors))) => vectors
                .top_matches(cue, usize::MAX, deadline, eligible)
                .into_iter()
                .map(|(key, score)| (key.source.clone(), key.index, key.hash, score))
                .collect(),
            _ => Vec::new(),
        };
        let lexical_lane = |pool: usize| -> Vec<crate::bm25::IndexHit> {
            lexical_full.iter().take(pool).cloned().collect()
        };
        // Pool first, floor second — the order search_passages applies
        // them in (`top_matches(cue, pool)` then the filter).
        let semantic_lane = |pool: usize| -> Vec<(String, u32, u64, f32)> {
            vector_rows
                .iter()
                .take(pool)
                .filter(|&&(.., score)| score >= floor)
                .cloned()
                .collect()
        };

        let full = fuse_passage_lanes(
            &store,
            lexical_lane(usize::MAX),
            semantic_lane(usize::MAX),
            usize::MAX,
        );
        // The served list exactly as `search_passages(limit)` builds it.
        let served_hits = fuse_passage_lanes(
            &store,
            lexical_lane(lane_pool(limit)),
            semantic_lane(lane_pool(limit)),
            limit,
        );

        let chosen = paragraph
            .or_else(|| {
                full.iter()
                    .find(|hit| hit.source == source)
                    .map(|hit| hit.index)
            })
            .unwrap_or_else(|| {
                // Nothing of the source ranked at all: the best showing
                // is the paragraph sharing the most query terms (the
                // first one, when they tie — including at zero).
                let mut best = (0u32, 0usize);
                for at in 0..paragraphs as u32 {
                    let shared = index.explain(&query_grams, source, at).map_or(0, |lex| {
                        lex.terms.iter().filter(|term| term.tf > 0.0).count()
                    });
                    if shared > best.1 {
                        best = (at, shared);
                    }
                }
                best.0
            });
        let (span, text) = record
            .paragraph(chosen as usize)
            .expect("the chosen paragraph is within the record");

        // The index answered for the text it saw; evidence about an
        // edited paragraph would explain the wrong bytes, so it is
        // withheld exactly like a stale search hit.
        let lexical = index
            .explain(&query_grams, source, chosen)
            .filter(|lex| lex.hash == span.hash);
        let overlap = lexical
            .as_ref()
            .is_some_and(|lex| lex.terms.iter().any(|term| term.tf > 0.0));
        let paragraph_terms = (!overlap).then(|| {
            // Both sides of the 表記ゆれ verdict on one table: the
            // paragraph's own spellings, questions included, deduped.
            let mut seen = std::collections::HashSet::new();
            let mut terms: Vec<String> = Vec::new();
            let mut take = |raw: &str| {
                for (spelling, gram) in spelled_passage_terms(raw) {
                    if seen.insert(gram) {
                        terms.push(spelling);
                    }
                }
            };
            take(text);
            for (at, question) in &record.questions {
                if *at == chosen {
                    take(question);
                }
            }
            terms
        });

        let vector = match (cue, gate) {
            (Err(error), _) => VectorLaneReport::QueryEmbeddingFailed(error),
            (Ok(None), _) | (Ok(Some(_)), Some(PassageVectorGate::Disabled)) => {
                VectorLaneReport::Off {
                    provider_configured: self.0.embedder.is_some(),
                }
            }
            (Ok(Some(_)), Some(PassageVectorGate::Empty)) => VectorLaneReport::NoVectors,
            (Ok(Some(_)), Some(PassageVectorGate::ModelChanged { stored, current })) => {
                VectorLaneReport::ModelChanged { stored, current }
            }
            // Same width guard as the search itself: `vector_rows` is
            // already empty (top_matches refuses a mismatched query),
            // and without this arm that silence would be reported as
            // `Ran { cosine: None }` — "not yet embedded", the wrong
            // diagnosis with the wrong repair.
            (Ok(Some(cue)), Some(PassageVectorGate::Ready(vectors)))
                if cue.len() != vectors.dim() =>
            {
                VectorLaneReport::WidthChanged {
                    stored: vectors.dim(),
                    current: cue.len(),
                }
            }
            (Ok(Some(_)), Some(PassageVectorGate::Ready(_))) => {
                // The target's best cosine across its rows (text row
                // and doc2query question rows alike), current-text rows
                // only — a stale row IS "not yet re-embedded". The
                // sweep is score-descending, so the first row is the
                // best one, floor or no floor.
                let cosine = vector_rows
                    .iter()
                    .find(|&&(ref row_source, row_index, hash, _)| {
                        row_source == source && row_index == chosen && hash == span.hash
                    })
                    .map(|&(.., score)| score);
                VectorLaneReport::Ran { floor, cosine }
            }
            (Ok(Some(_)), None) => unreachable!("the gate is read whenever a cue exists"),
        };

        let is_target = |hit: &PassageSearchHit| hit.source == source && hit.index == chosen;
        let rank = full.iter().position(is_target);
        let target = rank.map(|at| &full[at]);
        let served = served_hits.iter().any(is_target);

        // The smallest VERIFIED limit that serves the target: start at
        // its full-ranking rank (and low enough lane pools), rerun the
        // real serve computation, and grow on a miss — RRF against
        // capped pools can seat late double-lane candidates above a
        // mid-pool single-lane hit, so the unbounded rank alone is a
        // floor, not an answer.
        //
        // The starting pool must cover the WORSE of the two lane
        // ranks, not the better one: a dual-lane target's rerun score
        // only matches its full-ranking score once both lanes are
        // in-pool, and a pool sized off the better rank routinely
        // truncates away the worse lane's contribution, understating
        // the target and forcing extra doublings (or exhausting the
        // retry budget below) to reach a candidate this same target
        // would have cleared on the first try.
        let limit_to_reach = if served {
            Some(limit)
        } else {
            rank.map(|at| at + 1).and_then(|first| {
                let lane_need = target
                    .and_then(|hit| match (hit.bm25, hit.vector) {
                        (Some((bm25, _)), Some((vector, _))) => Some(bm25.max(vector)),
                        (Some((lane, _)), None) | (None, Some((lane, _))) => Some(lane),
                        (None, None) => None,
                    })
                    .map_or(1, |lane| lane.div_ceil(4));
                // `lane_need` sizes the pool a RAW lane rank would need,
                // unlike `first` it is not bounded by `full.len()` — two
                // different things can inflate a raw rank past it. Under
                // heavy staleness, many raw lane hits get filtered out
                // before `full` is built, so `lane_need` overshoots what
                // is actually necessary there and `full.len()` — always
                // a legal, still-untried candidate — is the right first
                // try. But a paragraph can also own several LIVE rows in
                // a lane (doc2query question rows in the vector lane),
                // and then `lane_need` is exactly right while
                // `full.len()`'s own pool (`lane_pool(full.len())`) is
                // too small to ever reach it — no amount of retrying at
                // or below `full.len()` would. So: try the cheap
                // `full.len()`-bounded candidate first (keeps the
                // staleness case minimal), and only once that is
                // exhausted widen to the raw row ceiling `lane_need` was
                // actually sized for.
                let raw_ceiling = lexical_full.len().max(vector_rows.len()).max(full.len());
                let mut candidate = first.max(lane_need).min(full.len());
                let mut widened = false;
                for _ in 0..8 {
                    let rerun = fuse_passage_lanes(
                        &store,
                        lexical_lane(lane_pool(candidate)),
                        semantic_lane(lane_pool(candidate)),
                        candidate,
                    );
                    if rerun.iter().any(is_target) {
                        return Some(candidate);
                    }
                    if candidate >= raw_ceiling {
                        return None;
                    }
                    candidate = if !widened && candidate >= full.len() {
                        widened = true;
                        lane_need.max(candidate + 1).min(raw_ceiling)
                    } else {
                        (candidate.saturating_mul(2)).min(raw_ceiling)
                    };
                }
                None
            })
        };

        Some(Ok(PassageExplainLookup::Explained(Box::new(
            PassageSearchExplanation {
                paragraph: chosen,
                paragraphs,
                paragraph_named: paragraph.is_some(),
                query_terms,
                lexical,
                paragraph_terms,
                vector,
                fused: !semantic_lane(usize::MAX).is_empty(),
                ranked: full.len(),
                rank: rank.map(|at| at + 1),
                score: target.map(|hit| hit.score),
                bm25_lane: target.and_then(|hit| hit.bm25),
                vector_lane: target.and_then(|hit| hit.vector),
                limit,
                served,
                cutoff_score: served_hits.last().map(|hit| hit.score),
                limit_to_reach,
            },
        ))))
    }

    /// A resident BM25 index for this entry: build on the residency's
    /// first search, repair a drifted sidecar per source, rebuild when
    /// tombstones have piled up. Double-checked so concurrent first
    /// searches build once. Called under the entry's read fence with
    /// every passage-store lock released — the documented order for
    /// `Entry::bm25` (holding `bm25` while READING the store is fine,
    /// and is how the build works). `Err` when `deadline` expires
    /// before a needed rebuild starts; the index is left as it was.
    fn ensure_bm25_index(
        &self,
        entry: &Entry,
        store: &crate::passages::PassageStore,
        name: &str,
        deadline: Deadline,
    ) -> Result<(), DeadlineExceeded> {
        let stale = {
            let guard = entry.bm25.read();
            match &*guard {
                None => true,
                Some(index) => index.needs_reclaim(),
            }
        };
        if stale {
            let mut guard = entry.bm25.write();
            let rebuild = match &*guard {
                None => true,
                Some(index) => index.needs_reclaim(),
            };
            if rebuild {
                if deadline.expired() {
                    return Err(DeadlineExceeded);
                }
                let records = store.snapshot();
                let built_at = std::time::Instant::now();
                let index = if guard.take().is_some() {
                    // Tombstone reclamation: rebuild fresh from the store.
                    entry.bm25_dirty.store(true, Ordering::Relaxed);
                    crate::bm25::Bm25Index::build(&records)
                } else if let Some(mut loaded) =
                    crate::bm25::Bm25Index::load(&bm25_path(&self.0.data_dir, &file_stem(name)))
                {
                    // A sidecar spares the re-tokenization, but its save
                    // cadence is the flush tick — repair whatever drifted
                    // (per source, both directions) instead of trusting
                    // or rebuilding wholesale.
                    let mut disk = loaded.source_digests();
                    let mut drifted = 0usize;
                    for (source, record) in &records {
                        if disk.remove(source) != Some(crate::bm25::record_digest(record)) {
                            loaded.upsert_source(source, record);
                            drifted += 1;
                        }
                    }
                    drifted += disk.len();
                    for source in disk.keys() {
                        loaded.remove_source(source);
                    }
                    if drifted > 0 {
                        entry.bm25_dirty.store(true, Ordering::Relaxed);
                    }
                    loaded
                } else {
                    entry.bm25_dirty.store(true, Ordering::Relaxed);
                    crate::bm25::Bm25Index::build(&records)
                };
                *guard = Some(index);
                tracing::info!(
                    context = %name,
                    sources = records.len(),
                    ms = built_at.elapsed().as_millis() as u64,
                    "BM25 index ready",
                );
            }
        }
        Ok(())
    }

    /// The semantic lane's query embedding, run BEFORE any lock — a
    /// provider round trip must never extend an entry fence. `Ok(None)`
    /// when the lane is off; `Err` carries the provider's refusal for
    /// the caller to log (search) or report (explain).
    fn passage_query_cue(
        &self,
        query: &str,
        deadline: Deadline,
    ) -> Result<Option<Arc<Vec<f32>>>, String> {
        if !self.passage_embedding_enabled() {
            return Ok(None);
        }
        let embedder = self.0.embedder.clone().expect("enabled implies a provider");
        self.cue_vector(&*embedder, query, deadline).map(Some)
    }

    /// Why the vector lane can or cannot sweep this entry's paragraphs.
    /// Search takes the `Ready` arm and silently skips the rest;
    /// explain names them.
    fn passage_vector_gate(&self, entry: &Entry, stem: &str) -> PassageVectorGate {
        let Some(embedder) = self.0.embedder.as_ref().filter(|_| self.0.embed_passages) else {
            return PassageVectorGate::Disabled;
        };
        let vectors = self.entry_passage_vectors(entry, stem);
        if vectors.is_empty() {
            return PassageVectorGate::Empty;
        }
        if vectors.model != embedder.model() {
            return PassageVectorGate::ModelChanged {
                stored: vectors.model.clone(),
                current: embedder.model().to_string(),
            };
        }
        PassageVectorGate::Ready(vectors)
    }

    /// The floor the semantic lane drops cosine matches below — the
    /// same chain `semantic_resolve` walks: the caller's one-call
    /// override beats the context setting beats the server default.
    fn effective_semantic_floor(&self, floor_override: Option<f32>, meta: &ContextMeta) -> f32 {
        floor_override
            .or(meta.semantic_floor)
            .unwrap_or(self.0.default_semantic_floor)
            .clamp(0.0, 1.0)
    }
}

/// Each lane's over-fetch for one served `limit`: fusion can promote a
/// hit neither lane put in its own top `limit`, and the staleness
/// checks drop stragglers.
fn lane_pool(limit: usize) -> usize {
    limit.saturating_mul(4).max(50)
}

/// The deduplicated term keys of one query — first occurrence of each
/// key wins, so the stream keeps [`passage_terms`] order.
fn deduped_query_grams(query: &str) -> Vec<u64> {
    let mut seen = std::collections::HashSet::new();
    passage_terms(query)
        .into_iter()
        .filter(|gram| seen.insert(*gram))
        .collect()
}

/// [`deduped_query_grams`] with each key's spelling — the explain
/// path's view of the same stream (same walker, same dedup rule).
fn deduped_spelled_query_terms(query: &str) -> Vec<(String, u64)> {
    let mut seen = std::collections::HashSet::new();
    spelled_passage_terms(query)
        .into_iter()
        .filter(|&(_, gram)| seen.insert(gram))
        .collect()
}

/// The floor-filtered lane view of one vector sweep, in sweep order —
/// the pool cap is the caller's (`top_matches` already applied it).
fn semantic_lane_hits(
    rows: Vec<(&crate::embedding::PassageKey, f32)>,
    floor: f32,
) -> Vec<(String, u32, u64, f32)> {
    rows.into_iter()
        .filter(|&(_, score)| score >= floor)
        .map(|(key, score)| (key.source.clone(), key.index, key.hash, score))
        .collect()
}

/// Fuses the two lanes' pools into the served ranking. Fuse by rank,
/// then validate EACH LANE against the store's current paragraph:
/// every lane scored the text it saw, and vectors routinely lag the
/// text between refreshes — a stale lane must neither smuggle its
/// outdated score onto fresh text nor veto the other lane's fresh
/// match, so each loses exactly its own evidence (and its fusion
/// term). The top-level score stays the raw BM25 number when no
/// semantic lane ran, so a lexical-only deployment keeps its
/// historical score semantics.
fn fuse_passage_lanes(
    store: &crate::passages::PassageStore,
    lexical: Vec<crate::bm25::IndexHit>,
    semantic: Vec<(String, u32, u64, f32)>,
    limit: usize,
) -> Vec<PassageSearchHit> {
    const RRF_K: f32 = 60.0;
    let fused = !semantic.is_empty();
    let mut accumulated: HashMap<(String, u32), FusedHit> = HashMap::new();
    for (rank, (source, index, hash, score)) in lexical.into_iter().enumerate() {
        accumulated.entry((source, index)).or_default().bm25 = Some((rank + 1, score, hash));
    }
    for (rank, (source, index, hash, score)) in semantic.into_iter().enumerate() {
        // A paragraph can hit this lane several times (its own text
        // row plus its doc2query question rows); ranks ascend, so
        // the first arrival is its best showing and later ones must
        // not overwrite it.
        let slot = accumulated.entry((source, index)).or_default();
        if slot.vector.is_none() {
            slot.vector = Some((rank + 1, score, hash));
        }
    }

    let rrf =
        |lane: &Option<(usize, f32)>| lane.map_or(0.0, |(rank, _)| 1.0 / (RRF_K + rank as f32));
    let mut hits: Vec<PassageSearchHit> = Vec::new();
    for ((source, index), lanes) in accumulated {
        let Some(record) = store.get(&source) else {
            continue;
        };
        let Some((span, text)) = record.paragraph(index as usize) else {
            continue;
        };
        let bm25 = lanes
            .bm25
            .filter(|&(.., hash)| hash == span.hash)
            .map(|(rank, score, _)| (rank, score));
        let vector = lanes
            .vector
            .filter(|&(.., hash)| hash == span.hash)
            .map(|(rank, score, _)| (rank, score));
        if bm25.is_none() && vector.is_none() {
            continue;
        }
        let score = if fused {
            rrf(&bm25) + rrf(&vector)
        } else {
            bm25.map(|(_, score)| score).unwrap_or(0.0)
        };
        hits.push(PassageSearchHit {
            source,
            index,
            score,
            text: text.to_string(),
            bm25,
            vector,
        });
    }
    hits.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.source.cmp(&b.source))
            .then_with(|| a.index.cmp(&b.index))
    });
    hits.truncate(limit);
    hits
}
