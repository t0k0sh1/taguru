use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use taguru::context::Context;
use taguru::deadline::{Deadline, DeadlineExceeded};

use crate::embedding::{
    EmbeddingProvider, PassageKey, PassageVectorStore, VectorStore, VectorTable, similarity,
};
use crate::hash::fnv1a;

use super::{
    AccessError, AppState, EmbeddingsStatus, Entry, GlossLaneReport, GlossSidecarStatus,
    PassageRefreshOutcome, PassageSidecarStatus, SEMANTIC_RESOLVE_LIMIT,
    dispatch_chunks_concurrently, file_stem, pvectors_path, vectors_path,
};

impl AppState {
    /// Whether the semantic entry tier has a provider at all.
    pub fn embeddings_configured(&self) -> bool {
        self.0.embedder.is_some()
    }

    /// The embedding identity in one read: the provider this server is
    /// configured to call beside what each vector sidecar actually
    /// holds. `None` when the context does not exist. Backs
    /// `GET /contexts/{name}/embeddings` — the identity a calibration
    /// report stamps its floor with (#131).
    pub fn embeddings_status(&self, name: &str) -> Option<EmbeddingsStatus> {
        let entry = self.lookup(name)?;
        let stem = file_stem(name);
        let store = self.entry_vectors(&entry, &stem);
        // width() is Some exactly when anything is stored, so it doubles
        // as the emptiness gate.
        let glosses = store.width().map(|width| GlossSidecarStatus {
            model: store.model.clone(),
            width,
            concepts: store.concepts.len(),
            labels: store.labels.len(),
        });
        let passages = self.entry_passage_vectors(&entry, &stem);
        let passages = (!passages.is_empty()).then(|| PassageSidecarStatus {
            model: passages.model.clone(),
            width: passages.dim(),
            rows: passages.len(),
        });
        Some(EmbeddingsStatus {
            provider_model: self
                .0
                .embedder
                .as_ref()
                .map(|embedder| embedder.model().to_string()),
            glosses,
            passages,
        })
    }

    /// Name pairs whose GLOSSES sit close in embedding space — the
    /// synonym-fork candidates (創業年 vs 設立年) that no spelling
    /// comparison can see. Works off the stored vector sidecar alone
    /// (no provider round trip), so it runs even when the provider is
    /// gone, and is skipped with a note when no vectors exist or a
    /// namespace is too large for the O(N²) sweep. Returns
    /// (concept_pairs, label_pairs, skipped_note).
    #[allow(clippy::type_complexity)]
    pub fn semantic_twins(
        &self,
        name: &str,
        cosine_floor: f32,
        deadline: Deadline,
    ) -> Option<(
        Vec<(String, String, f32)>,
        Vec<(String, String, f32)>,
        Option<String>,
    )> {
        /// Past this many names a namespace's pairwise sweep is skipped.
        const SWEEP_CAP: usize = 2000;
        /// At most this many pairs per namespace come back.
        const PAIR_CAP: usize = 100;

        let entry = self.lookup(name)?;
        let floor = cosine_floor.clamp(0.0, 1.0);
        let store = self.entry_vectors(&entry, &file_stem(name));
        if store.concepts.is_empty() && store.labels.is_empty() {
            return Some((
                Vec::new(),
                Vec::new(),
                Some(
                    "ベクトル未生成のため意味的検出はスキップ (POST embeddings/refresh を実行)"
                        .to_string(),
                ),
            ));
        }

        let mut skipped = None;
        let sweep = |table: &VectorTable,
                     skipped: &mut Option<String>|
         -> Vec<(String, String, f32)> {
            if table.len() > SWEEP_CAP {
                *skipped = Some(format!(
                    "語彙が {} 名を超えるためこの名前空間の意味的検出はスキップ",
                    SWEEP_CAP
                ));
                return Vec::new();
            }
            let entries: Vec<(&String, &Vec<f32>)> = {
                let mut entries: Vec<_> = table.iter().map(|(name, (_, v))| (name, v)).collect();
                entries.sort_by_key(|(name, _)| name.as_str());
                entries
            };
            let mut pairs = Vec::new();
            for (i, (name_a, vector_a)) in entries.iter().enumerate() {
                if deadline.expired() {
                    *skipped = Some(
                        "意味的検出は期限切れのため途中で打ち切り (一部の結果のみ)".to_string(),
                    );
                    break;
                }
                for (name_b, vector_b) in &entries[i + 1..] {
                    let score = similarity(vector_a, vector_b);
                    if score >= floor {
                        pairs.push(((*name_a).clone(), (*name_b).clone(), score));
                    }
                }
            }
            pairs.sort_by(|x, y| {
                y.2.total_cmp(&x.2)
                    .then_with(|| (&x.0, &x.1).cmp(&(&y.0, &y.1)))
            });
            pairs.truncate(PAIR_CAP);
            pairs
        };
        let mut concepts = sweep(&store.concepts, &mut skipped);
        let mut labels = sweep(&store.labels, &mut skipped);
        // Related is not duplicate: concepts joined by an edge and labels
        // co-used on one subject resemble each other BECAUSE they are
        // related (glosses quote shared facts), and would bury the real
        // fork candidates in noise. Filtering needs the graph, so the
        // context loads if cold — acceptable for an explicit audit.
        match self.read_context(name, |context| {
            concepts.retain(|(a, b, _)| !context.adjacent(a, b));
            labels.retain(|(a, b, _)| !context.labels_share_subject(a, b));
        }) {
            Ok(()) => {}
            Err(AccessError::NotFound) => return None,
            Err(AccessError::Load(message))
            | Err(AccessError::Unpersisted(message))
            | Err(AccessError::QuotaExceeded(message)) => {
                // Vectors were readable but the graph was not: serve the
                // unfiltered pairs and say why they are noisier. (A
                // read never yields Unpersisted or QuotaExceeded; the
                // arms are for the type, not a path.)
                skipped = Some(format!(
                    "関連ペアの除外はスキップ (グラフ未ロード: {message})"
                ));
            }
            // read_context never consults a deadline itself — the
            // caller checks its own budget before calling in —
            // unreachable in practice, kept for exhaustiveness.
            Err(AccessError::DeadlineExceeded) => {
                skipped = Some("関連ペアの除外はスキップ (期限切れ)".to_string());
            }
        }
        Some((concepts, labels, skipped))
    }

    /// Embeds the GLOSS of every canonical concept and label — the name
    /// plus its heaviest facts — and persists the vector sidecar. Bare
    /// names carry too little signal for sentence-trained embedding
    /// models; the graph supplies the context itself. Each vector
    /// remembers the hash of the gloss it was computed from, so a
    /// refresh re-embeds exactly the names that are new or whose graph
    /// context changed. Explicit rather than automatic — an agent or
    /// operator calls this after ingesting, so embedding spend stays
    /// intentional. Returns (newly embedded, total vectors), or `None`
    /// for an unknown context.
    pub fn refresh_embeddings(
        &self,
        name: &str,
        deadline: Deadline,
    ) -> Option<Result<(usize, usize), String>> {
        let Some(embedder) = self.0.embedder.clone() else {
            return Some(Err(
                "no embedding provider is configured (set TAGURU_EMBED_URL and TAGURU_EMBED_MODEL)"
                    .to_string(),
            ));
        };
        let entry = self.lookup(name)?;
        // One refresh per context at a time (see Entry::vectors_refresh
        // for why); held across the gloss read too, not just the embed
        // and merge, so no overlapping refresh can be mid-flight against
        // a gloss state this one hasn't seen yet.
        let _serial = entry.vectors_refresh.lock();
        let glosses = match self.read_context(name, |context| {
            let concepts: Vec<(String, String)> = context
                .concept_names()
                .into_iter()
                .map(|name| {
                    let gloss = context
                        .concept_gloss(name, Context::GLOSS_FACTS)
                        .unwrap_or_else(|| name.to_string());
                    (name.to_string(), gloss)
                })
                .collect();
            let labels: Vec<(String, String)> = context
                .labels()
                .into_iter()
                .map(|name| {
                    let gloss = context
                        .label_gloss(name, Context::GLOSS_EXAMPLES)
                        .unwrap_or_else(|| name.to_string());
                    (name.to_string(), gloss)
                })
                .collect();
            (concepts, labels)
        }) {
            Ok(glosses) => glosses,
            Err(AccessError::NotFound) => return None,
            // A read never yields Unpersisted or QuotaExceeded; the
            // arms are for the type, not a path.
            Err(AccessError::Load(message))
            | Err(AccessError::Unpersisted(message))
            | Err(AccessError::QuotaExceeded(message)) => {
                return Some(Err(message));
            }
            // read_context never consults a deadline itself — the
            // caller checks its own budget before calling in —
            // unreachable in practice, kept for exhaustiveness.
            Err(AccessError::DeadlineExceeded) => {
                return Some(Err("request deadline exceeded".to_string()));
            }
        };
        let (concepts, labels) = glosses;
        let path = vectors_path(&self.0.data_dir, &file_stem(name));

        // Diff and embed while still holding `_serial`, not the entry's
        // data lock — provider round trips can take seconds and must
        // not block graph reads/writes. `_serial` (not this) is what
        // keeps two overlapping refreshes from racing: only one can be
        // here at a time, so the diff below always runs against
        // whatever the previous refresh (if any) already published.
        //
        // Read through the memory cache, not straight off disk: a prior
        // refresh's save can fail after the provider already sold it
        // the rows (see the tail of this function), and the cache is
        // where those survive even though the sidecar does not. Empty
        // and disk agree whenever nothing failed, so this changes
        // nothing on the common path.
        let existing = self.entry_vectors(&entry, &file_stem(name));
        // Claim the save-pending flag up front: it only ever reflects a
        // prior pass's save failure (this pass owns the whole write
        // side via `_serial`, so nothing else can set it mid-flight),
        // and tells this pass to retry the write below even if its own
        // diff buys nothing new.
        let was_pending = entry.vectors_save_pending.swap(false, Ordering::Relaxed);
        let mut fresh_model = existing.model != embedder.model();
        // ONE width agreement across both tables: without it, a provider
        // mid-migration could answer the concept call at one width and
        // the label call at another, and the merged store would persist
        // mixed — a file the loader now refuses outright (#133). The
        // first vector either call lands settles it; disagreeing rows
        // drop loudly and stay stale, exactly like a chunk disagreeing
        // within one call.
        let mut settled_width: Option<usize> = None;
        let (mut embedded_concepts, concept_failure) = self.embed_stale(
            &*embedder,
            &existing.concepts,
            &concepts,
            fresh_model,
            &mut settled_width,
            deadline,
        );
        let (mut embedded_labels, label_failure) = self.embed_stale(
            &*embedder,
            &existing.labels,
            &labels,
            fresh_model,
            &mut settled_width,
            deadline,
        );
        // Persist whatever either table bought even when the other fails:
        // losing already-billed vectors to a sibling's provider error is
        // the bug this mirrors from the passage refresh. A partial failure
        // does skip the width probe just below — spending more provider
        // budget on a pass that already reports Err and gets retried buys
        // nothing — but not the carried-vs-fresh reconciliation after it:
        // that one decides whether what already landed this pass is fit to
        // persist at all.
        let mut failure = concept_failure.or(label_failure);
        // The model NAME is the staleness discriminator, but a provider
        // can change output width behind a stable name (a backend swap
        // behind the same proxy or gateway). Old-width rows carried next
        // to new-width ones would feed `similarity` mismatched
        // dimensions — no error, no score — so a width disagreement
        // stales the whole table, exactly as if the model were renamed.
        // Concepts and labels are sampled and compared independently —
        // collapsing to "whichever table is non-empty, concepts first"
        // would miss a width change confined to whichever table that
        // fallback didn't happen to sample.
        let width = |table: &VectorTable| table.values().map(|(_, vector)| vector.len()).next();
        let carried_concepts_width = width(&existing.concepts);
        let carried_labels_width = width(&existing.labels);
        let mut fresh_width = width(&embedded_concepts).or_else(|| width(&embedded_labels));
        // Unchanged hashes embed nothing, which would leave the width
        // change of exactly this scenario — backend swap, no gloss
        // edits — undetectable forever. One probe embedding per no-op
        // refresh keeps that from hiding.
        if failure.is_none()
            && !fresh_model
            && (carried_concepts_width.is_some() || carried_labels_width.is_some())
            && fresh_width.is_none()
            && let Some((_, gloss)) = concepts.first().or_else(|| labels.first())
        {
            match self.timed_embed_for_refresh(embedder.as_ref(), &[gloss.as_str()], deadline) {
                Ok(vectors) => {
                    fresh_width = vectors.first().map(Vec::len);
                }
                Err(error) => {
                    failure = Some(error);
                }
            }
        }
        // Not gated on `failure.is_none()`: a sibling table's provider
        // error must not excuse persisting this pass's already-landed
        // vectors at a width that disagrees with what is carried —
        // that mismatch is decided below, then reconciled regardless of
        // what else failed.
        let width_mismatch = fresh_width.is_some_and(|fresh| {
            carried_concepts_width.is_some_and(|carried| carried != fresh)
                || carried_labels_width.is_some_and(|carried| carried != fresh)
        });
        if !fresh_model && width_mismatch {
            tracing::warn!(
                context = name,
                model = embedder.model(),
                carried_concepts = ?carried_concepts_width,
                carried_labels = ?carried_labels_width,
                fresh = fresh_width,
                "embedding width changed under an unchanged model name; re-embedding every gloss"
            );
            self.0.metrics.record_gloss_width_rebuild();
            fresh_model = true;
            // A fresh agreement for the redo: the old width was just
            // declared dead, and the redo's own first vector settles
            // the new one for both tables.
            let mut settled_width: Option<usize> = None;
            let (concepts_reembedded, concept_failure) = self.embed_stale(
                &*embedder,
                &existing.concepts,
                &concepts,
                true,
                &mut settled_width,
                deadline,
            );
            embedded_concepts = concepts_reembedded;
            let (labels_reembedded, label_failure) = self.embed_stale(
                &*embedder,
                &existing.labels,
                &labels,
                true,
                &mut settled_width,
                deadline,
            );
            embedded_labels = labels_reembedded;
            failure = concept_failure.or(label_failure);
        }
        let newly_embedded = embedded_concepts.len() + embedded_labels.len();

        // Publish under the entry's tombstone fence (a delete that may
        // have won it must not see its sidecar recreated) — `_serial`
        // above, held since before the gloss read, is what makes this
        // read-modify-write race-free, not this lock by itself. A SHARED
        // fence is enough: nothing below touches the entry's own data,
        // only `entry.vectors` (its own lock) and the sidecar file on
        // disk, so there is no reason to block concurrent graph reads for
        // the length of this save — the same trade `flush_bm25` makes
        // for its sidecar.
        let _fence = entry.read_unless_deleted()?;
        // Same basis `existing` was diffed against above, not a fresh
        // disk read: `_serial` has excluded every other refresh of
        // this context since before `existing` was taken, so nothing
        // could have changed the sidecar (or the cache backing it) in
        // between — re-reading here would only risk losing rows a
        // still-unpersisted prior pass bought and cached but a
        // straight disk read cannot see.
        //
        // `fresh_model` also covers the width change above: rows for
        // names that have since left the graph must not linger at the
        // old width either.
        let mut store = if fresh_model || existing.model != embedder.model() {
            VectorStore {
                model: embedder.model().to_string(),
                ..Default::default()
            }
        } else {
            (*existing).clone()
        };
        store.concepts.extend(embedded_concepts);
        store.labels.extend(embedded_labels);
        // Prune ghost rows: a name dropped by compaction leaves the live
        // gloss lists, so nothing above re-embeds or carries it, yet its
        // stored vector would linger here forever and
        // semantic_resolve/semantic_twins would keep surfacing a name the
        // graph no longer holds. A model/width wipe above already dropped
        // such rows wholesale; this covers ordinary retraction, the way
        // the passage refresh gets for free by rebuilding.
        let live_concepts: HashSet<&str> = concepts.iter().map(|(name, _)| name.as_str()).collect();
        let live_labels: HashSet<&str> = labels.iter().map(|(name, _)| name.as_str()).collect();
        let before_prune = store.concepts.len() + store.labels.len();
        store
            .concepts
            .retain(|name, _| live_concepts.contains(name.as_str()));
        store
            .labels
            .retain(|name, _| live_labels.contains(name.as_str()));
        let total = store.concepts.len() + store.labels.len();
        let pruned = before_prune - total;
        // `was_pending` covers a prior save that failed after already
        // buying rows: this pass's own diff can land on newly_embedded
        // == 0 and pruned == 0 (everything it needs is already carried
        // from `existing`, which reads the memory cache — see above)
        // while the disk image is still whatever the failed save left
        // it as. Without this, that state would never retry the write.
        let save_error = if newly_embedded > 0 || pruned > 0 || was_pending {
            store.save(&path).err()
        } else {
            None
        };
        if save_error.is_some() {
            entry.vectors_save_pending.store(true, Ordering::Relaxed);
        }
        // Publish the fresh store so queries never re-read the sidecar.
        // On a failed save too: the provider already sold
        // `embedded_concepts`/`embedded_labels`, and caching the merged
        // store is what keeps the next refresh's `existing` (read from
        // this same cache, not the disk — see above) from buying them a
        // second time. Only the sidecar write failed, not this.
        *entry.vectors.lock() = Some(Arc::new(store));
        drop(_fence);
        // Served content changed the moment the merge published above —
        // save success or not — so the config revision moves with it. A
        // `was_pending`-only rewrite republished bytes the cache already
        // served and bumps nothing. After the fence: the bump takes the
        // entry's write lock, and its own tombstone check covers the
        // delete race the fence covered here.
        if newly_embedded > 0 || pruned > 0 {
            self.bump_config_revision(name, &entry);
        }
        if let Some(error) = save_error {
            return Some(Err(format!("vector store not persisted: {error}")));
        }
        // What landed is durable; a provider failure still returns Err so
        // the caller sees the pass was partial, and the stale rows it
        // skipped stay stale for the next refresh to retry.
        match failure {
            Some(error) => Some(Err(error)),
            None => Some(Ok((newly_embedded, total))),
        }
    }

    /// Diffs one gloss table against its stored vectors and embeds what
    /// is new or changed, 128 glosses per provider call. Each vector
    /// remembers the hash of the gloss it came from; `fresh_model`
    /// marks everything stale. Returns the vectors that landed alongside
    /// the first provider error, if any — the caller persists the former
    /// so a sibling table's failure never discards billed work, and the
    /// stale rows the error skipped stay stale for the next refresh to
    /// retry. Chunks dispatch concurrently, so a provider mid-migration
    /// can answer two chunks of the very same call with different
    /// widths; unlike `PassageVectorStore::push`, `VectorTable` has no
    /// dimension of its own to enforce, so a vector that disagrees with
    /// `settled_width` — the ONE width agreement the whole refresh
    /// shares across its concept and label calls, claimed by the first
    /// vector any of them lands — is dropped here — loudly, and left
    /// stale for the next refresh — rather than merged into a store
    /// that would persist mixed widths, which the loader refuses whole
    /// and `similarity` would silently stop matching against.
    fn embed_stale(
        &self,
        embedder: &dyn EmbeddingProvider,
        stored: &VectorTable,
        entries: &[(String, String)],
        fresh_model: bool,
        settled_width: &mut Option<usize>,
        deadline: Deadline,
    ) -> (VectorTable, Option<String>) {
        let stale: Vec<(String, String, u64)> = entries
            .iter()
            .filter_map(|(name, gloss)| {
                let hash = fnv1a(gloss);
                let outdated =
                    fresh_model || stored.get(name).is_none_or(|&(hashed, _)| hashed != hash);
                outdated.then(|| (name.clone(), gloss.clone(), hash))
            })
            .collect();
        let stale_chunks: Vec<&[(String, String, u64)]> = stale.chunks(128).collect();
        let outcomes =
            dispatch_chunks_concurrently(&stale_chunks, self.0.embed_parallel, |chunk| {
                if deadline.expired() {
                    return Err(DeadlineExceeded.to_string());
                }
                let texts: Vec<&str> = chunk.iter().map(|(_, gloss, _)| gloss.as_str()).collect();
                self.timed_embed_for_refresh(embedder, &texts, deadline)
            });
        let mut embedded = VectorTable::new();
        let mut failure: Option<String> = None;
        for (chunk, outcome) in stale_chunks.iter().zip(outcomes) {
            match outcome {
                Some(Ok(vectors)) => {
                    for ((name, _, hash), vector) in chunk.iter().zip(vectors) {
                        let expected = *settled_width.get_or_insert(vector.len());
                        if vector.len() != expected {
                            tracing::warn!(
                                name = name.as_str(),
                                expected,
                                got = vector.len(),
                                "dropping a gloss vector whose width disagrees with what \
                                 this refresh already settled on — a provider mid-migration; \
                                 it stays stale for the next refresh to retry"
                            );
                            continue;
                        }
                        embedded.insert(name.clone(), (*hash, vector));
                    }
                }
                // Keep the vectors that did land so the caller can persist
                // them; report the first error. Stale rows this failure
                // skipped stay stale in the diff for the next refresh.
                Some(Err(error)) => failure = failure.or(Some(error)),
                None => {}
            }
        }
        (embedded, failure)
    }

    /// Whether the vector lane over paragraphs is on at all: a provider
    /// is configured AND the operator opted the corpus in
    /// (`TAGURU_EMBED_PASSAGES`).
    pub fn passage_embedding_enabled(&self) -> bool {
        self.0.embed_passages && self.0.embedder.is_some()
    }

    /// Contexts whose passages changed since their last embedding
    /// refresh — the auto-refresh ticker's work list. Claiming is the
    /// caller's job via [`AppState::refresh_passage_embeddings`].
    pub fn passage_embed_dirty_names(&self) -> Vec<String> {
        self.snapshot()
            .into_iter()
            .filter(|(_, entry)| entry.passages_embed_dirty.load(Ordering::Relaxed))
            .map(|(name, _)| name)
            .collect()
    }

    /// Embeds every stored paragraph (`EmbedPurpose::Index`) into the
    /// `{stem}.pvectors.bin` sidecar: the vector lane's index side.
    /// Diff-driven like the gloss refresh — a paragraph whose FNV-1a
    /// hash already has a row under the current model is carried
    /// forward, a vanished paragraph's row is dropped (retraction
    /// pruning falls out of the rebuild), and only the rest go to the
    /// provider, 128 per call. The sidecar is written AT MOST ONCE per
    /// refresh: writing per batch would multiply a large store's bytes
    /// across the whole backfill. A provider failure partway persists
    /// what did land and reports the error — the next refresh continues
    /// from there instead of re-buying the same vectors.
    pub fn refresh_passage_embeddings(
        &self,
        name: &str,
        deadline: Deadline,
    ) -> Option<Result<PassageRefreshOutcome, String>> {
        let Some(embedder) = self.0.embedder.clone() else {
            return Some(Err(
                "no embedding provider is configured (set TAGURU_EMBED_URL and TAGURU_EMBED_MODEL)"
                    .to_string(),
            ));
        };
        if !self.0.embed_passages {
            return Some(Err(
                "passage embedding is disabled (set TAGURU_EMBED_PASSAGES=1)".to_string(),
            ));
        }
        let entry = self.lookup(name)?;
        // One refresh per context at a time (see Entry::passage_refresh
        // for why); the diff below makes the loser's pass a no-op.
        let _serial = entry.passage_refresh.lock();
        // Claim the dirty flag up front: work that lands mid-refresh
        // re-marks it, so the ticker returns — never lost, never
        // double-claimed. The prior value still matters: besides a
        // fresh passage store/retract, it is also how a save that
        // failed after buying rows (see `changed` below) tells this
        // pass to retry the write even if its own diff finds nothing
        // new — otherwise a would-be-`changed: false` pass would never
        // flush the cache the failed save left behind onto disk.
        let was_dirty = entry.passages_embed_dirty.swap(false, Ordering::Relaxed);
        let store = {
            let _fence = entry.read_unless_deleted()?;
            match self.entry_passages(&entry, &file_stem(name)) {
                Ok(store) => store,
                Err(error) => {
                    // The claim above must not eat the work: a store
                    // that cannot load now still needs its refresh once
                    // it can.
                    entry.passages_embed_dirty.store(true, Ordering::Relaxed);
                    return Some(Err(error.to_string()));
                }
            }
        };
        let records = store.snapshot();
        let path = pvectors_path(&self.0.data_dir, &file_stem(name));
        // Read through the memory cache, not straight off disk: a prior
        // refresh's save can fail after the provider already sold it
        // the rows (see the tail of this function), and the cache is
        // where those survive even though the sidecar does not. Empty
        // and disk agree whenever nothing failed, so this changes
        // nothing on the common path.
        let existing = self.entry_passage_vectors(&entry, &file_stem(name));
        let mut fresh_model = existing.model != embedder.model();
        // A provider can change output width behind a stable model name
        // (a backend swap behind the same proxy). Old-width rows carried
        // next to new-width ones would let PassageVectorStore::push drop
        // every new row this pass embeds — a stale store that also
        // over-reports what it stored — so a width disagreement stales
        // the whole table, exactly as a model rename does. Detected the
        // way the concept refresh detects it; the redo re-walks `records`
        // (still in scope) so it carries no extra memory. `dim` is
        // private, so the carried width is the first stored row's length.
        let carried_width = existing.iter().next().map(|(_, row)| row.len());
        let (fresh, embedded, skipped_over_limit, failure) = loop {
            let carried: HashMap<(&str, u32, u64, Option<u64>), &[f32]> = if fresh_model {
                HashMap::new()
            } else {
                existing
                    .iter()
                    .map(|(key, row)| {
                        (
                            (key.source.as_str(), key.index, key.hash, key.question_hash),
                            row,
                        )
                    })
                    .collect()
            };

            // Deterministic walk — snapshot() is sorted by source, spans by
            // position, questions by paragraph — so the same rows win the
            // limit run after run. Each paragraph offers its own text row
            // and then one row per stored question, every one keyed to the
            // PARAGRAPH (hash included) with the question's own hash as the
            // discriminator.
            let mut fresh = PassageVectorStore::new(embedder.model());
            let mut to_embed: Vec<(PassageKey, String)> = Vec::new();
            let mut skipped_over_limit = 0usize;
            for (source, record) in &records {
                for (span, text) in record.paragraph_texts() {
                    let question_rows = record
                        .questions
                        .iter()
                        .filter(|&&(paragraph, _)| paragraph == span.index)
                        .map(|(_, question)| (Some(fnv1a(question)), question.as_str()));
                    for (question_hash, row_text) in
                        std::iter::once((None, text)).chain(question_rows)
                    {
                        // Stored before the write surfaces refused empty
                        // question text, an empty row would be sent to the
                        // provider verbatim — and providers refuse
                        // zero-length input, failing that row's whole
                        // chunk and abandoning the pass at the same spot
                        // on every retry. Empty text retrieves nothing
                        // anyway: skip it.
                        if row_text.is_empty() {
                            continue;
                        }
                        if fresh.len() + to_embed.len() >= self.0.passage_vector_limit {
                            skipped_over_limit += 1;
                            continue;
                        }
                        let key = PassageKey {
                            source: source.clone(),
                            index: span.index,
                            hash: span.hash,
                            question_hash,
                        };
                        match carried.get(&(source.as_str(), span.index, span.hash, question_hash))
                        {
                            Some(row) => fresh.push(key, row.to_vec()),
                            None => to_embed.push((key, row_text.to_string())),
                        }
                    }
                }
            }

            let to_embed_chunks: Vec<&[(PassageKey, String)]> = to_embed.chunks(128).collect();
            let outcomes =
                dispatch_chunks_concurrently(&to_embed_chunks, self.0.embed_parallel, |chunk| {
                    if deadline.expired() {
                        return Err(DeadlineExceeded.to_string());
                    }
                    let texts: Vec<&str> = chunk.iter().map(|(_, text)| text.as_str()).collect();
                    self.timed_embed_for_refresh(embedder.as_ref(), &texts, deadline)
                });
            let mut embedded = 0usize;
            let mut failure: Option<String> = None;
            let mut fresh_width: Option<usize> = None;
            for (chunk, outcome) in to_embed_chunks.iter().zip(outcomes) {
                match outcome {
                    Some(Ok(vectors)) => {
                        for ((key, _), vector) in chunk.iter().zip(vectors) {
                            fresh_width.get_or_insert(vector.len());
                            // `push` silently drops a row whose width
                            // disagrees with the dimension `fresh` already
                            // settled on (the same provider-mid-migration
                            // hazard `embed_stale` guards against for
                            // glosses) — count only the rows that actually
                            // landed, or `embedded` over-reports what
                            // `total_rows` below can already prove didn't
                            // all land.
                            let before = fresh.len();
                            fresh.push(key.clone(), vector);
                            embedded += fresh.len() - before;
                        }
                    }
                    Some(Err(error)) => failure = failure.or(Some(error)),
                    None => {}
                }
            }
            // Unchanged hashes embed nothing, which would leave the width
            // change of exactly this scenario — backend swap, no passage
            // edits — undetectable. One probe embedding per no-op refresh
            // keeps it from hiding, matching the concept refresh.
            if failure.is_none()
                && !fresh_model
                && carried_width.is_some()
                && fresh_width.is_none()
                && let Some(probe) = records
                    .iter()
                    .flat_map(|(_, record)| record.paragraph_texts())
                    .map(|(_, text)| text)
                    .find(|text| !text.is_empty())
            {
                match self.timed_embed_for_refresh(embedder.as_ref(), &[probe], deadline) {
                    Ok(vectors) => fresh_width = vectors.first().map(Vec::len),
                    Err(error) => failure = Some(error),
                }
            }
            // Not gated on `failure.is_none()`: a chunk that failed must
            // not excuse persisting this pass's already-landed rows at a
            // width that disagrees with what is carried — that mismatch
            // is decided here and reconciled regardless of what else
            // failed.
            if !fresh_model
                && let (Some(carried_w), Some(fresh_w)) = (carried_width, fresh_width)
                && carried_w != fresh_w
            {
                tracing::warn!(
                    context = name,
                    model = embedder.model(),
                    carried = carried_w,
                    fresh = fresh_w,
                    "passage embedding width changed under an unchanged model name; re-embedding every passage"
                );
                self.0.metrics.record_passage_width_rebuild();
                fresh_model = true;
                continue;
            }
            break (fresh, embedded, skipped_over_limit, failure);
        };

        // Publish under the entry's tombstone fence (a delete that won
        // it must not see its files recreated), and only when something
        // changed — an all-carried refresh is a no-op, not a rewrite. A
        // SHARED fence, exactly like the read phase above: nothing here
        // touches the entry's own data, only `entry.passage_vectors`
        // (its own lock) and the sidecar file, so graph reads need not
        // block for the length of this save.
        // `was_dirty` covers a prior save that failed after already
        // buying rows: this pass's own diff can land on `changed:
        // false` (everything it needs is already carried from
        // `existing`, which reads the memory cache — see above) while
        // the disk image is still whatever the failed save left it
        // as. Without this, that state would never retry the write.
        let changed = embedded > 0
            || fresh.len() != existing.len()
            || (fresh_model && !fresh.is_empty())
            || was_dirty;
        // `changed` minus the `was_dirty`-only save retry: whether the
        // rows about to publish differ from what was being SERVED — the
        // config-revision signal, as opposed to the rewrite-the-sidecar
        // signal above.
        let published_change =
            embedded > 0 || fresh.len() != existing.len() || (fresh_model && !fresh.is_empty());
        let _fence = entry.read_unless_deleted()?;
        let total_rows = fresh.len();
        let save_error = if changed {
            fresh.save(&path).err()
        } else {
            None
        };
        if save_error.is_some() {
            entry.passages_embed_dirty.store(true, Ordering::Relaxed);
        }
        // Publish on a failed save too: the provider already sold
        // `embedded` of these rows, and caching them is what keeps the
        // next refresh's `existing` (read from this same cache, not the
        // disk — see above) from buying them a second time. Only the
        // sidecar write failed, not this.
        *entry.passage_vectors.lock() = Some(Arc::new(fresh));
        drop(_fence);
        // Served content changed with the publish above, so the config
        // revision moves with it — after the fence, exactly as the
        // gloss refresh does (the bump re-checks the tombstone itself).
        if published_change {
            self.bump_config_revision(name, &entry);
        }
        if let Some(error) = save_error {
            return Some(Err(format!("passage vectors not persisted: {error}")));
        }
        match failure {
            Some(error) => {
                // What landed is durable; the rest stays claimed as work.
                entry.passages_embed_dirty.store(true, Ordering::Relaxed);
                Some(Err(error))
            }
            None => Some(Ok(PassageRefreshOutcome {
                embedded,
                total: total_rows,
                skipped_over_limit,
            })),
        }
    }

    /// The semantic fallback behind resolve: nearest stored names by
    /// cosine over the vector sidecar. Meant to run only after the
    /// lexical tiers found nothing; scores are cosine similarities — a
    /// different scale from lexical scores, which the API marks by tier.
    /// Empty when no provider is configured, no refresh has run, or the
    /// sidecar belongs to another model.
    pub fn semantic_resolve(
        &self,
        name: &str,
        cue: &str,
        labels: bool,
        floor_override: Option<f32>,
        deadline: Deadline,
    ) -> Option<Result<Vec<(String, f32)>, String>> {
        let Some(embedder) = self.0.embedder.clone() else {
            return Some(Ok(Vec::new()));
        };
        let entry = self.lookup(name)?;
        // One-call override beats the context setting beats the server
        // default (see [`DEFAULT_SEMANTIC_FLOOR`] for the calibration).
        let context_floor = entry.inner.read().meta.semantic_floor;
        let floor = floor_override
            .or(context_floor)
            .unwrap_or(self.0.default_semantic_floor)
            .clamp(0.0, 1.0);
        let store = self.entry_vectors(&entry, &file_stem(name));
        if store.model != embedder.model() {
            return Some(Ok(Vec::new()));
        }
        let table = if labels {
            &store.labels
        } else {
            &store.concepts
        };
        if table.is_empty() {
            return Some(Ok(Vec::new()));
        }
        let cue_vector = match self.cue_vector(&*embedder, cue, deadline) {
            Ok(vector) => vector,
            Err(error) => return Some(Err(error)),
        };
        // A width mismatch (a dimensions setting changed behind a
        // stable model name, #133) folds to the same empty answer as a
        // model change — this tier is deliberately best-effort — and
        // explain tells the states apart (`GlossLaneReport`). Swept
        // anyway, every cosine would be `similarity`'s silent 0.0.
        if store
            .width()
            .is_some_and(|stored| stored != cue_vector.len())
        {
            return Some(Ok(Vec::new()));
        }
        let mut scored: Vec<(String, f32)> = table
            .iter()
            .map(|(name, (_, vector))| (name.clone(), similarity(&cue_vector, vector)))
            .filter(|&(_, score)| score >= floor)
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(SEMANTIC_RESOLVE_LIMIT);
        Some(Ok(scored))
    }

    /// The gloss lane's account of one (cue, expected) pair — why
    /// [`AppState::semantic_resolve`] could not have surfaced the
    /// expected name (it folds provider-off, model-changed, and
    /// nothing-embedded into one empty answer; explain needs them
    /// apart), or exactly where it stood when the sweep could run:
    /// its own gloss cosine against the floor in effect, and its rank
    /// in the very ordering `semantic_resolve` truncates. `None` when
    /// the context does not exist.
    pub fn explain_semantic_resolve(
        &self,
        name: &str,
        cue: &str,
        expected: &str,
        labels: bool,
        floor_override: Option<f32>,
        deadline: Deadline,
    ) -> Option<GlossLaneReport> {
        let Some(embedder) = self.0.embedder.clone() else {
            return Some(GlossLaneReport::Off);
        };
        let entry = self.lookup(name)?;
        let context_floor = entry.inner.read().meta.semantic_floor;
        let floor = floor_override
            .or(context_floor)
            .unwrap_or(self.0.default_semantic_floor)
            .clamp(0.0, 1.0);
        let store = self.entry_vectors(&entry, &file_stem(name));
        // A never-refreshed sidecar is empty, whatever model string it
        // carries — report the missing refresh, not a model change.
        if store.concepts.is_empty() && store.labels.is_empty() {
            return Some(GlossLaneReport::EmptyTable);
        }
        if store.model != embedder.model() {
            return Some(GlossLaneReport::ModelChanged {
                stored: store.model.clone(),
                current: embedder.model().to_string(),
            });
        }
        let table = if labels {
            &store.labels
        } else {
            &store.concepts
        };
        if table.is_empty() {
            return Some(GlossLaneReport::EmptyTable);
        }
        let cue_vector = match self.cue_vector(&*embedder, cue, deadline) {
            Ok(vector) => vector,
            Err(error) => return Some(GlossLaneReport::QueryEmbeddingFailed(error)),
        };
        // Without this arm the sweep below would report a measured-
        // looking cosine of 0.0 — `similarity`'s width-mismatch
        // sentinel — and the verdict would prescribe lowering a floor
        // that no value could satisfy.
        if let Some(stored) = store.width()
            && stored != cue_vector.len()
        {
            return Some(GlossLaneReport::WidthChanged {
                stored,
                current: cue_vector.len(),
            });
        }
        let cosine = table
            .get(expected)
            .map(|(_, vector)| similarity(&cue_vector, vector));
        // The expected name's 1-based rank in semantic_resolve's exact
        // ordering (cosine desc, name asc): candidates strictly ahead
        // of it, plus one. Counted, not sorted — one sweep.
        let mut passing = 0usize;
        let mut ahead = 0usize;
        for (candidate, (_, vector)) in table.iter() {
            let score = similarity(&cue_vector, vector);
            if score < floor {
                continue;
            }
            passing += 1;
            if let Some(cosine) = cosine
                && (score > cosine || (score == cosine && candidate.as_str() < expected))
            {
                ahead += 1;
            }
        }
        let rank = cosine.filter(|&cosine| cosine >= floor).map(|_| ahead + 1);
        Some(GlossLaneReport::Ran {
            floor,
            cosine,
            rank,
            passing,
            cap: SEMANTIC_RESOLVE_LIMIT,
        })
    }

    /// The entry's vector store, loaded from its sidecar on first use
    /// and held until refresh replaces it or eviction clears it.
    fn entry_vectors(&self, entry: &Entry, stem: &str) -> Arc<VectorStore> {
        let mut cached = entry.vectors.lock();
        match &*cached {
            Some(store) => Arc::clone(store),
            None => {
                let store = Arc::new(VectorStore::load(&vectors_path(&self.0.data_dir, stem)));
                *cached = Some(Arc::clone(&store));
                store
            }
        }
    }
}
