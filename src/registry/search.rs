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
        // The two lanes touch disjoint locks — `entry.bm25` vs. the
        // `passage_vectors` slot behind `passage_vector_gate` — and
        // neither reads the other's output, so they run on their own
        // scoped threads instead of back to back (same `thread::scope`
        // shape `preload_pinned` uses, sized to exactly the two lanes
        // there are) — but only when the semantic lane has real work
        // to do. A failed or absent query embedding resolves to a
        // fixed status with no lock and no sweep, so a lexical-only
        // deployment (or any request whose embedding call failed)
        // would spend a thread-spawn on that fixed lookup for nothing;
        // both lanes run inline on the calling thread instead.

        // Semantic lane: sweep the paragraph vectors with the
        // pre-embedded query, then drop candidates below the same
        // floor semantic_resolve applies to its own cosine matches
        // — the caller's one-call override beats the context
        // setting beats the server default (`fence` is already
        // this entry's read lock, taken above). Every arm also
        // names what it did: the status the handler serializes
        // into the response's plan.
        let semantic_work = || -> (Vec<(String, u32, u64, f32)>, VectorLaneStatus) {
            match &cue {
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
                    // The gate checks the model NAME; the width can
                    // still disagree (a dimensions setting changed
                    // behind a stable name, #133). Swept anyway,
                    // every row would be `similarity`'s silent 0.0 —
                    // an empty lane the plan would then call "ran" —
                    // so the mismatch is named instead, exactly like
                    // a model change.
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
                    // Unreachable in practice (a cue exists only
                    // when the lane is on), kept for the same
                    // defensiveness as explain's mapping.
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
            }
        };

        let (lexical, (semantic, vector)) = if matches!(cue, Ok(Some(_))) {
            std::thread::scope(|scope| {
                let lexical_lane = scope.spawn(|| {
                    let guard = entry.bm25.read();
                    let index = guard.as_ref().expect("index was just built");
                    index.search(&query_grams, pool, eligible)
                });
                let semantic_lane = scope.spawn(semantic_work);

                // A lane panic surfaces exactly like it would have before
                // this ran on two threads — propagate, not degrade (unlike
                // hydrate's ensure_context, nothing here waits on a condvar
                // that a swallowed panic would otherwise strand forever).
                (
                    lexical_lane.join().expect("lexical lane thread panicked"),
                    semantic_lane.join().expect("semantic lane thread panicked"),
                )
            })
        } else {
            let lexical = {
                let guard = entry.bm25.read();
                let index = guard.as_ref().expect("index was just built");
                index.search(&query_grams, pool, eligible)
            };
            (lexical, semantic_work())
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;

    use super::*;
    use crate::embedding::{EmbedPurpose, EmbeddingProvider};
    use crate::registry::test_support::{
        MockEmbeddings, boot_for_passage_embedding, plain, scratch_dir,
    };

    #[test]
    fn passage_search_ranks_the_answering_text_first() {
        let dir = scratch_dir("bm25");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "第2段落".to_string(),
            "原料米には主に山田錦を使い、精米歩合は50パーセントまで磨く。".to_string(),
        );
        passages.insert(
            "第3段落".to_string(),
            "杜氏の高瀬は南部杜氏の出身で、経験は30年を超える。".to_string(),
        );
        passages.insert(
            "第5段落".to_string(),
            "蔵開きの祭りでは、雲居山の伏流水で仕込んだ新酒がふるまわれる。".to_string(),
        );
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        // The procedural question never became a triple; the text lane
        // must still hand back the passage that answers it, first.
        let hits = state
            .search_passages(
                "sake",
                "精米歩合はどこまで磨く?",
                3,
                None,
                None,
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap()
            .hits;
        assert_eq!(hits[0].source, "第2段落");
        assert!(hits[0].score > 0.0);

        // No shared bigrams at all → nothing, not noise.
        assert!(
            state
                .search_passages(
                    "sake",
                    "unrelated english words",
                    3,
                    None,
                    None,
                    Deadline::unbounded()
                )
                .unwrap()
                .unwrap()
                .hits
                .is_empty()
        );
        assert!(
            state
                .search_passages("nope", "x", 3, None, None, Deadline::unbounded())
                .is_none()
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn passage_search_discriminates_english_by_words() {
        let dir = scratch_dir("bm25-english");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("papers", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // English prose shares nearly every character pair across
        // documents; only word terms can tell these two apart. The
        // real-world case: a famous quote had to be found in the essay
        // that contains it, not in whichever essay mentions the topic.
        let mut passages = BTreeMap::new();
        passages.insert(
            "第51篇".to_string(),
            "The great security against a gradual concentration of the several powers \
             in the same department consists in giving to those who administer each \
             department the necessary constitutional means and personal motives to \
             resist encroachments of the others. Ambition must be made to counteract \
             ambition."
                .to_string(),
        );
        passages.insert(
            "第70篇".to_string(),
            "Energy in the executive is a leading character in the definition of good \
             government. It is essential to the protection of the community against \
             foreign attacks and to the security of liberty against the enterprises \
             and assaults of ambition, of faction, and of anarchy."
                .to_string(),
        );
        state
            .store_passages("papers", plain(passages))
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages(
                "papers",
                "ambition must be made to counteract ambition",
                2,
                None,
                None,
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap()
            .hits;
        assert_eq!(hits[0].source, "第51篇");
        assert!(
            hits.len() < 2 || hits[0].score > hits[1].score,
            "the containing passage must win decisively, not by tie-break"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_camel_case_piece_finds_the_passage() {
        let dir = scratch_dir("camel");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("code", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // "AppState" and "PathBuf" occur only as camelCase tokens here —
        // none of their pieces appears as a standalone word.
        let mut passages = BTreeMap::new();
        passages.insert(
            "src/registry.rs:AppState".to_string(),
            "impl AppState { pub fn boot_with(dir: PathBuf) -> Self { todo!() } }".to_string(),
        );
        state
            .store_passages("code", plain(passages))
            .unwrap()
            .unwrap();

        for query in ["state", "State", "app", "path"] {
            let hits = state
                .search_passages("code", query, 3, None, None, Deadline::unbounded())
                .unwrap()
                .unwrap()
                .hits;
            assert_eq!(
                hits.first().map(|hit| hit.source.as_str()),
                Some("src/registry.rs:AppState"),
                "a piece of a camelCase identifier must reach its passage (query {query:?})"
            );
        }

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn passage_search_returns_the_answering_paragraph_not_the_whole_document() {
        let dir = scratch_dir("bm25-paragraph");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // One document, three paragraphs; only the middle one answers.
        let mut passages = BTreeMap::new();
        passages.insert(
            "docs/aomine.md".to_string(),
            "青嶺酒造は雲居県霧沢町の蔵元である。\n\n原料米には山田錦を使い、精米歩合は50パーセントまで磨く。\n\n蔵開きの祭りでは新酒がふるまわれる。".to_string(),
        );
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages(
                "sake",
                "精米歩合はどこまで磨く?",
                3,
                None,
                None,
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap()
            .hits;
        assert_eq!(
            (hits[0].source.as_str(), hits[0].index),
            ("docs/aomine.md", 1),
            "the hit names the paragraph, not just the document"
        );
        assert_eq!(
            hits[0].text, "原料米には山田錦を使い、精米歩合は50パーセントまで磨く。",
            "the text is the answering paragraph alone"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_store_after_the_first_search_updates_the_index_in_place() {
        let dir = scratch_dir("bm25-incremental");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("第1章".to_string(), "青嶺酒造の創業は1907年。".to_string());
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        // First search builds the resident index.
        assert!(
            !state
                .search_passages("sake", "創業はいつ", 3, None, None, Deadline::unbounded())
                .unwrap()
                .unwrap()
                .hits
                .is_empty()
        );

        // A later store must reach searches through the in-place
        // update — no reclaim is due, so a rebuild cannot be the
        // reason this passes.
        let mut more = BTreeMap::new();
        more.insert(
            "第2章".to_string(),
            "杜氏の高瀬は南部杜氏の出身。".to_string(),
        );
        state.store_passages("sake", plain(more)).unwrap().unwrap();
        let hits = state
            .search_passages("sake", "杜氏の出身", 3, None, None, Deadline::unbounded())
            .unwrap()
            .unwrap()
            .hits;
        assert_eq!(hits[0].source, "第2章");

        // And a retraction disappears the same way.
        state.retract_source("sake", "第2章").unwrap();
        assert!(
            state
                .search_passages("sake", "杜氏の出身", 3, None, None, Deadline::unbounded())
                .unwrap()
                .unwrap()
                .hits
                .iter()
                .all(|hit| hit.source != "第2章"),
            "a retracted source must leave the index too"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn passage_search_survives_a_restart_without_retokenizing() {
        let dir = scratch_dir("bm25-persist");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            let mut passages = BTreeMap::new();
            passages.insert("第1章".to_string(), "青嶺酒造の創業は1907年。".to_string());
            state
                .store_passages("sake", plain(passages))
                .unwrap()
                .unwrap();
            // First search builds and marks dirty; the tick persists.
            state
                .search_passages("sake", "創業はいつ", 3, None, None, Deadline::unbounded())
                .unwrap()
                .unwrap();
            state.flush_dirty();
            assert!(bm25_path(&dir, &file_stem("sake")).exists());
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let hits = state
            .search_passages("sake", "創業はいつ", 3, None, None, Deadline::unbounded())
            .unwrap()
            .unwrap()
            .hits;
        assert_eq!(hits[0].source, "第1章");
        let entry = state.lookup("sake").unwrap();
        assert!(
            !entry.bm25_dirty.load(Ordering::Relaxed),
            "a clean sidecar loads as-is — nothing drifted, nothing re-tokenized"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_stale_index_sidecar_is_repaired_by_the_source_digest_mismatch() {
        let dir = scratch_dir("bm25-stale");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            let mut passages = BTreeMap::new();
            passages.insert("第1章".to_string(), "杜氏は高瀬である。".to_string());
            passages.insert("第2章".to_string(), "仕込み水は伏流水。".to_string());
            state
                .store_passages("sake", plain(passages))
                .unwrap()
                .unwrap();
            state
                .search_passages("sake", "杜氏", 3, None, None, Deadline::unbounded())
                .unwrap()
                .unwrap();
            state.flush_dirty(); // the sidecar now says 高瀬
        }

        // A new run edits 第1章 and searches BEFORE any flush: the
        // sidecar on disk still carries the old paragraph.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let mut edited = BTreeMap::new();
        edited.insert("第1章".to_string(), "杜氏は佐伯に交代した。".to_string());
        state
            .store_passages("sake", plain(edited))
            .unwrap()
            .unwrap();
        let hits = state
            .search_passages("sake", "杜氏は誰", 3, None, None, Deadline::unbounded())
            .unwrap()
            .unwrap()
            .hits;
        assert!(
            hits[0].text.contains("佐伯"),
            "the digest mismatch must repair 第1章 from the store, got {:?}",
            hits[0].text
        );
        let entry = state.lookup("sake").unwrap();
        assert!(
            entry.bm25_dirty.load(Ordering::Relaxed),
            "a repair leaves the sidecar stale until the next tick"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_corrupt_index_sidecar_falls_back_to_a_full_rebuild_not_an_outage() {
        let dir = scratch_dir("bm25-corrupt");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "第1章".to_string(),
            "蔵開きの祭りでは新酒がふるまわれる。".to_string(),
        );
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        fs::write(bm25_path(&dir, &file_stem("sake")), b"not an index").unwrap();

        let hits = state
            .search_passages("sake", "蔵開きの祭り", 3, None, None, Deadline::unbounded())
            .unwrap()
            .unwrap()
            .hits;
        assert_eq!(hits[0].source, "第1章");
        state.flush_dirty();
        assert!(
            crate::bm25::Bm25Index::load(&bm25_path(&dir, &file_stem("sake"))).is_some(),
            "the tick replaces the corpse with a valid sidecar"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn eviction_drops_the_resident_index_and_a_later_search_rebuilds_it() {
        let dir = scratch_dir("bm25-evict");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "第1章".to_string(),
            "蔵開きの祭りでは新酒がふるまわれる。".to_string(),
        );
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        assert!(
            !state
                .search_passages("sake", "蔵開きの祭り", 3, None, None, Deadline::unbounded())
                .unwrap()
                .unwrap()
                .hits
                .is_empty()
        );

        let entry = state.lookup("sake").unwrap();
        assert!(state.evict_entry("sake", &entry));
        assert!(
            entry.bm25.read().is_none(),
            "eviction must drop the resident index"
        );
        assert_eq!(
            state
                .search_passages("sake", "蔵開きの祭り", 3, None, None, Deadline::unbounded())
                .unwrap()
                .unwrap()
                .hits[0]
                .source,
            "第1章",
            "the next search rebuilds and still answers"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// A regression for a false-negative `limit_to_reach`: a raw semantic
    /// lane rank counts every row the vector store holds for a source,
    /// stale ones included, so it can run far ahead of `full.len()` (the
    /// count that actually survives the staleness filter). `lane_need`
    /// derives from that raw rank; before the fix it sized the starting
    /// candidate past `full.len()` and the search bailed out with "no
    /// limit reaches it" without ever trying `full.len()` itself — which,
    /// being the target's own rank in `full`, always would have.
    #[test]
    fn limit_to_reach_is_not_a_false_negative_when_stale_rows_inflate_the_raw_lane_rank() {
        struct MarkerEmbeddings;
        impl EmbeddingProvider for MarkerEmbeddings {
            fn model(&self) -> &str {
                "marker"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                Ok(texts
                    .iter()
                    .map(|text| {
                        if text.starts_with("TARGETMARKER") {
                            vec![0.5, 0.8660254]
                        } else {
                            vec![1.0, 0.0]
                        }
                    })
                    .collect())
            }
        }

        let dir = scratch_dir("limit-to-reach-stale");
        let state = boot_for_passage_embedding(&dir, Arc::new(MarkerEmbeddings), 20_000);
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        let mut passages = BTreeMap::new();
        passages.insert("fresh-doc".to_string(), "FRESHMARKER".to_string());
        passages.insert("target-doc".to_string(), "TARGETMARKER".to_string());
        for i in 0..15 {
            passages.insert(format!("decoy-{i:02}"), "DECOYMARKER".to_string());
        }
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();

        // Edit every decoy without re-embedding: their vector rows stay
        // in the store at the old hash, tied with `fresh-doc` at the top
        // of the raw cosine sweep, but staleness drops all 15 out of
        // `full` — so they inflate the target's raw rank without ever
        // occupying a `full` slot themselves.
        let mut edited = BTreeMap::new();
        for i in 0..15 {
            edited.insert(format!("decoy-{i:02}"), "DECOYMARKER-EDITED".to_string());
        }
        state
            .store_passages("sake", plain(edited))
            .unwrap()
            .unwrap();

        let explanation = state
            .explain_passage_search(
                "sake",
                "QUERYMARKER",
                "target-doc",
                None,
                1,
                None,
                None,
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        let PassageExplainLookup::Explained(explanation) = explanation else {
            panic!("expected an Explained verdict");
        };
        assert!(
            !explanation.served,
            "fresh-doc alone fills the one requested seat"
        );
        assert_eq!(
            explanation.ranked, 2,
            "only fresh-doc and target-doc survive the staleness filter"
        );
        assert_eq!(
            explanation.limit_to_reach,
            Some(2),
            "full.len() (2) is itself a valid, untried candidate — it must not \
             be skipped as unreachable just because lane_need overshoots it"
        );

        let hits = state
            .search_passages("sake", "QUERYMARKER", 2, None, None, Deadline::unbounded())
            .unwrap()
            .unwrap()
            .hits;
        assert!(
            hits.iter().any(|hit| hit.source == "target-doc"),
            "limit_to_reach must actually reach it: {hits:?}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// A false-negative twin of the regression above, this time from a
    /// healthy cause: doc2query gives one paragraph several vector rows
    /// (its text plus each stored question), so a raw semantic-lane rank
    /// can run far past `full.len()` even when every row is current. The
    /// fix above clamps the starting candidate to `full.len()`, but
    /// `full.len()` is only a valid candidate when ITS OWN lane pool
    /// (`full.len() * 4`, floored at 50) actually reaches the target's
    /// raw row — here two decoys carrying 30 questions each plant 62
    /// perfect-cosine rows ahead of the target's one, past that 50-row
    /// floor, so clamping to `full.len()` (3) still bails out on a limit
    /// that would have served it.
    #[test]
    fn limit_to_reach_is_not_a_false_negative_when_doc2query_questions_inflate_the_raw_lane_rank() {
        struct MarkerEmbeddings;
        impl EmbeddingProvider for MarkerEmbeddings {
            fn model(&self) -> &str {
                "marker"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                Ok(texts
                    .iter()
                    .map(|text| {
                        if text.starts_with("TARGETMARKER") {
                            vec![0.5, 0.8660254]
                        } else {
                            vec![1.0, 0.0]
                        }
                    })
                    .collect())
            }
        }

        let dir = scratch_dir("limit-to-reach-doc2query");
        let state = boot_for_passage_embedding(&dir, Arc::new(MarkerEmbeddings), 20_000);
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        let mut passages = BTreeMap::new();
        passages.insert(
            "target-doc".to_string(),
            crate::passages::PassageSubmission {
                text: "TARGETMARKER".to_string(),
                questions: Vec::new(),
                sections: Vec::new(),
                meta: crate::passages::SourceMeta::default(),
            },
        );
        for i in 0..2 {
            let questions = (0..30)
                .map(|q| (0, format!("DECOYQUESTION{i}X{q}")))
                .collect();
            passages.insert(
                format!("decoy-{i:02}"),
                crate::passages::PassageSubmission {
                    text: "DECOYMARKER".to_string(),
                    questions,
                    sections: Vec::new(),
                    meta: crate::passages::SourceMeta::default(),
                },
            );
        }
        state.store_passages("sake", passages).unwrap().unwrap();
        state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let explanation = state
            .explain_passage_search(
                "sake",
                "QUERYMARKER",
                "target-doc",
                None,
                3,
                None,
                None,
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        let PassageExplainLookup::Explained(explanation) = explanation else {
            panic!("expected an Explained verdict");
        };
        assert!(
            !explanation.served,
            "the two decoys alone fill all three requested seats"
        );
        assert_eq!(
            explanation.ranked, 3,
            "target-doc and both decoys all survive — nothing here is stale"
        );
        assert_eq!(
            explanation.limit_to_reach,
            Some(16),
            "target-doc's raw vector-lane rank is 63rd (62 decoy rows \
             ahead of it); lane_pool(16) = 64 is the smallest pool that \
             reaches it, and full.len() (3) must widen to it rather than \
             give up once its own 50-row pool comes up short"
        );

        let hits = state
            .search_passages(
                "sake",
                "QUERYMARKER",
                explanation.limit_to_reach.unwrap(),
                None,
                None,
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap()
            .hits;
        assert!(
            hits.iter().any(|hit| hit.source == "target-doc"),
            "limit_to_reach must actually reach it: {hits:?}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_stored_question_row_answers_a_question_shaped_query_at_its_paragraph() {
        let dir = scratch_dir("doc2query-hit");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "doc".to_string(),
            crate::passages::PassageSubmission {
                text: "りんごは真っ赤に実った。".to_string(),
                questions: vec![(0, "アップルはどんな色?".to_string())],
                sections: Vec::new(),
                meta: crate::passages::SourceMeta::default(),
            },
        );
        state.store_passages("fruit", passages).unwrap().unwrap();
        let outcome = state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (outcome.embedded, outcome.total),
            (2, 2),
            "the paragraph row and its question row both embed"
        );

        // The query matches the QUESTION row's wording, not the text's;
        // both rows point at the same paragraph, so the lane must fold
        // them into one hit at the question row's better rank.
        let hits = state
            .search_passages("fruit", "アップル", 3, None, None, Deadline::unbounded())
            .unwrap()
            .unwrap()
            .hits;
        assert_eq!(hits.len(), 1, "one paragraph, one hit — never a dup");
        assert_eq!((hits[0].source.as_str(), hits[0].index), ("doc", 0));
        assert!(
            hits[0].text.contains("りんご"),
            "the text served is the PARAGRAPH"
        );
        let (rank, cosine) = hits[0].vector.expect("found via the question row");
        assert_eq!(rank, 1);
        assert!(cosine > 0.99, "{cosine}");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn hybrid_search_surfaces_a_vector_only_hit_that_shares_no_letters() {
        // The Q→A gap in miniature: the query shares no bigrams with
        // the stored paragraph, so the lexical lane alone returns
        // nothing — the vector lane is what finds it, and the response
        // says so.
        let dir = scratch_dir("hybrid-vector-only");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごは真っ赤に実った。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();
        state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages("fruit", "アップル", 3, None, None, Deadline::unbounded())
            .unwrap()
            .unwrap()
            .hits;
        assert_eq!(hits[0].source, "doc-a");
        assert!(
            hits[0].bm25.is_none(),
            "no shared bigram — the lexical lane must not have seen this"
        );
        let (rank, cosine) = hits[0].vector.expect("the vector lane found it");
        assert_eq!(rank, 1);
        assert!(cosine > 0.9, "{cosine}");
        assert!(
            (hits[0].score - 1.0 / 61.0).abs() < 1e-6,
            "one lane at rank 1 fuses to 1/(60+1), got {}",
            hits[0].score
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn hybrid_search_reports_both_lanes_when_both_agree() {
        let dir = scratch_dir("hybrid-both");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごは真っ赤に実った。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();
        state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages(
                "fruit",
                "りんごは真っ赤",
                3,
                None,
                None,
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap()
            .hits;
        assert_eq!(hits[0].source, "doc-a");
        let (bm25_rank, bm25_score) = hits[0].bm25.expect("shared bigrams");
        let (vector_rank, _) = hits[0].vector.expect("same fruit, same axis");
        assert_eq!((bm25_rank, vector_rank), (1, 1));
        assert!(bm25_score > 0.0);
        assert!(
            (hits[0].score - 2.0 / 61.0).abs() < 1e-6,
            "two lanes at rank 1 fuse to 2/(60+1), got {}",
            hits[0].score
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_lane_that_scored_outdated_text_loses_its_evidence_not_the_hit() {
        // Vectors refresh on their own cadence and routinely lag the
        // text. After an edit, the in-place-updated BM25 lane must
        // keep answering while the vector lane's stale evidence is
        // dropped — never attached to text it did not score, and never
        // vetoing the fresh lexical match.
        let dir = scratch_dir("hybrid-stale-lane");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごは真っ赤に実った。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();
        state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        // The text changes; the vector sidecar is NOT refreshed.
        let mut edited = BTreeMap::new();
        edited.insert(
            "doc-a".to_string(),
            "りんごは青森の名産である。".to_string(),
        );
        state
            .store_passages("fruit", plain(edited))
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages("fruit", "りんご", 3, None, None, Deadline::unbounded())
            .unwrap()
            .unwrap()
            .hits;
        assert_eq!(hits[0].source, "doc-a");
        assert!(hits[0].text.contains("青森"), "the CURRENT text is served");
        assert!(
            hits[0].bm25.is_some(),
            "the lexical lane scored the fresh text and survives"
        );
        assert!(
            hits[0].vector.is_none(),
            "the vector lane scored the OLD text; its evidence must drop"
        );
        let (bm25_rank, _) = hits[0].bm25.unwrap();
        assert!(
            (hits[0].score - 1.0 / (60.0 + bm25_rank as f32)).abs() < 1e-6,
            "the fused score counts surviving lanes only, got {}",
            hits[0].score
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn hybrid_search_degrades_to_bm25_when_the_query_embedding_fails() {
        struct FlakyEmbeddings {
            calls: std::sync::atomic::AtomicUsize,
            fail_on: usize,
        }
        impl EmbeddingProvider for FlakyEmbeddings {
            fn model(&self) -> &str {
                "flaky"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                let call = self.calls.fetch_add(1, Ordering::Relaxed);
                if call == self.fail_on {
                    return Err("provider hiccup".to_string());
                }
                Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
            }
        }

        let dir = scratch_dir("hybrid-degrade");
        let state = boot_for_passage_embedding(
            &dir,
            Arc::new(FlakyEmbeddings {
                calls: std::sync::atomic::AtomicUsize::new(0),
                fail_on: 1, // the refresh succeeds; the QUERY embed fails
            }),
            20_000,
        );
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごは真っ赤に実った。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();
        state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages("fruit", "りんご", 3, None, None, Deadline::unbounded())
            .unwrap()
            .unwrap()
            .hits;
        assert_eq!(
            hits[0].source, "doc-a",
            "a broken decoration must not break the answer"
        );
        assert!(hits[0].vector.is_none());
        let (_, bm25_score) = hits[0].bm25.unwrap();
        assert_eq!(
            hits[0].score, bm25_score,
            "with no semantic lane the score stays the raw BM25 number"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn lexical_only_deployments_keep_raw_bm25_scores() {
        let dir = scratch_dir("hybrid-lexical-only");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごは真っ赤に実った。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages("fruit", "りんご", 3, None, None, Deadline::unbounded())
            .unwrap()
            .unwrap()
            .hits;
        assert!(hits[0].vector.is_none());
        let (rank, bm25_score) = hits[0].bm25.unwrap();
        assert_eq!(rank, 1);
        assert_eq!(hits[0].score, bm25_score);
        assert!(hits[0].score > 0.1, "raw BM25, not a tiny RRF quotient");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn search_passages_vector_lane_drops_candidates_below_the_default_semantic_floor() {
        // みかん×りんご sits at cosine 0.28 — under the 0.35 default
        // floor — and the query shares no bigram with the stored text,
        // so a fusion that ignored the floor would surface a
        // near-irrelevant paragraph on the vector lane alone.
        let dir = scratch_dir("passages-floor-default");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごは真っ赤に実った。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();
        state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages("fruit", "みかん", 3, None, None, Deadline::unbounded())
            .unwrap()
            .unwrap()
            .hits;
        assert!(
            hits.is_empty(),
            "cosine 0.28 sits under the default floor: {hits:?}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn search_passages_vector_lane_honors_a_lowered_context_semantic_floor() {
        // Same 0.28 cosine as the default-floor test, but the context's
        // floor is lowered under it: the candidate must now clear the
        // vector lane and contribute its RRF term, same as
        // semantic_resolve honors the context setting over the server
        // default.
        let dir = scratch_dir("passages-floor-context");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごは真っ赤に実った。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();
        state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        state
            .update_meta("fruit", None, None, None, Some(0.2))
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages("fruit", "みかん", 3, None, None, Deadline::unbounded())
            .unwrap()
            .unwrap()
            .hits;
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].source, "doc-a");
        let (rank, cosine) = hits[0].vector.expect("cleared the lowered floor");
        assert_eq!(rank, 1);
        assert!((cosine - 0.28).abs() < 1e-6, "{cosine}");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn search_passages_request_floor_override_beats_the_context_and_server_floors() {
        // The same 0.28 cosine as the two floor tests above, pushed
        // through the third rung of the chain: a request override of
        // 0.2 admits it past the untouched 0.35 server default, and a
        // request override of 0.5 then drops it even though the
        // context's own floor was lowered under it — the caller's word
        // beats both, exactly as resolve's semantic_floor override.
        let dir = scratch_dir("passages-floor-override");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごは真っ赤に実った。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();
        state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let hits = state
            .search_passages("fruit", "みかん", 3, Some(0.2), None, Deadline::unbounded())
            .unwrap()
            .unwrap()
            .hits;
        assert_eq!(
            hits.len(),
            1,
            "an override under the cosine admits it past the server default: {hits:?}"
        );
        assert_eq!(hits[0].source, "doc-a");
        let (rank, cosine) = hits[0].vector.expect("cleared the request floor");
        assert_eq!(rank, 1);
        assert!((cosine - 0.28).abs() < 1e-6, "{cosine}");

        state
            .update_meta("fruit", None, None, None, Some(0.2))
            .unwrap()
            .unwrap();
        let hits = state
            .search_passages("fruit", "みかん", 3, Some(0.5), None, Deadline::unbounded())
            .unwrap()
            .unwrap()
            .hits;
        assert!(
            hits.is_empty(),
            "an override above the cosine drops it even under a context floor lowered below it: {hits:?}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn explain_passage_search_reports_the_request_floor_override_it_ran_under() {
        // The explanation accounts for the call actually made: under
        // the override the vector report carries the overridden floor,
        // and the 0.28 cosine the default floor would have filtered is
        // scored and shown.
        let dir = scratch_dir("passages-explain-floor-override");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごは真っ赤に実った。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();
        state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let explanation = state
            .explain_passage_search(
                "fruit",
                "みかん",
                "doc-a",
                None,
                3,
                Some(0.2),
                None,
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        let PassageExplainLookup::Explained(explanation) = explanation else {
            panic!("expected an Explained verdict");
        };
        let VectorLaneReport::Ran { floor, cosine } = explanation.vector else {
            panic!("expected the vector lane to have run");
        };
        assert!((floor - 0.2).abs() < 1e-6, "{floor}");
        let cosine = cosine.expect("the target's cosine is scored, not filtered");
        assert!((cosine - 0.28).abs() < 1e-6, "{cosine}");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn eviction_persists_a_dirty_index_for_the_next_residency() {
        let dir = scratch_dir("bm25-evict-save");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "第1章".to_string(),
            "麹室の湿度は五十パーセント。".to_string(),
        );
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        state
            .search_passages("sake", "麹室の湿度", 3, None, None, Deadline::unbounded())
            .unwrap()
            .unwrap();

        let entry = state.lookup("sake").unwrap();
        assert!(state.evict_entry("sake", &entry));
        assert!(
            bm25_path(&dir, &file_stem("sake")).exists(),
            "a dirty index rides out with the eviction"
        );
        // The next residency loads it clean instead of re-tokenizing.
        state
            .search_passages("sake", "麹室の湿度", 3, None, None, Deadline::unbounded())
            .unwrap()
            .unwrap();
        let entry = state.lookup("sake").unwrap();
        assert!(!entry.bm25_dirty.load(Ordering::Relaxed));

        let _ = fs::remove_dir_all(dir);
    }
}
