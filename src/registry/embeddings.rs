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
    dispatch_chunks_concurrently, file_stem, pvectors_path, still_quarantined, vectors_path,
};

/// Rows per provider call, shared by both the gloss (`embed_stale`) and
/// passage (`refresh_passage_embeddings`) refresh pipelines so the two
/// stay in sync instead of drifting apart as separately hardcoded `128`s.
const EMBED_CHUNK_SIZE: usize = 128;

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
        // Both sidecar loads sit under the entry's tombstone fence: a
        // delete that won the race must read as the same 404 the
        // context endpoint gives, not a 200 built from unlinked files'
        // empty defaults (or a successor generation's sidecars).
        let (store, passages) = {
            let _fence = entry.read_unless_deleted()?;
            (
                self.entry_vectors(&entry, &stem),
                self.entry_passage_vectors(&entry, &stem),
            )
        };
        // width() is Some exactly when anything is stored, so it doubles
        // as the emptiness gate.
        let glosses = store.width().map(|width| GlossSidecarStatus {
            model: store.model.clone(),
            width,
            concepts: store.concepts.len(),
            labels: store.labels.len(),
        });
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
        /// The sweep below is O(N²) similarity comparisons with no index
        /// to narrow it (unlike [`PassageVectorStore::top_matches`], this
        /// is an explicit audit call, not a hot query path, so no ANN
        /// structure exists for it) — at 2000 names that is ~2M
        /// comparisons, the point past which an unbounded request could
        /// tie up a request thread for an unpredictable stretch. No
        /// calibration sweep backs this exact number (unlike
        /// [`crate::embedding::PASSAGE_ANN_THRESHOLD`]); it is a
        /// deliberately conservative round bound, not a measured knee.
        const SWEEP_CAP: usize = 2000;
        /// At most this many pairs per namespace come back — a response-size
        /// bound, not a quality one: `pairs` is already sorted by score
        /// descending before this truncates, so raising it only ever adds
        /// weaker matches. Round number, not calibrated against a measured
        /// cost like [`crate::embedding::PASSAGE_ANN_THRESHOLD`].
        const PAIR_CAP: usize = 100;

        let entry = self.lookup(name)?;
        let floor = cosine_floor.clamp(0.0, 1.0);
        // Scoped tombstone fence: it covers the sidecar load (a lost
        // race with delete must answer `None`, not sweep a stale or
        // successor sidecar) and is dropped before the O(N²) sweep so
        // a delete never waits on it.
        let store = {
            let _fence = entry.read_unless_deleted()?;
            self.entry_vectors(&entry, &file_stem(name))
        };
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
    ///
    /// Always pays for its own width probe when one is needed — see
    /// [`AppState::auto_refresh_embeddings`] for the throttled variant
    /// the auto-embed ticker uses instead. An explicit call (this one)
    /// is rare and deliberate, often an operator diagnosing a stale
    /// result, so it must reliably detect and heal a width change in
    /// ONE call — the contract `tests/http_api/width_probe.rs` pins.
    pub fn refresh_embeddings(
        &self,
        name: &str,
        deadline: Deadline,
    ) -> Option<Result<(usize, usize), String>> {
        self.refresh_embeddings_inner(name, deadline, false)
    }

    /// The auto-embed ticker's variant of [`AppState::refresh_embeddings`]
    /// (issue #677 item 2): identical, except its width probe is
    /// skipped when a recent embed from ANY context already confirmed
    /// the provider's current width (`provider_width_recently_confirmed`
    /// — the width is the provider's property, not this context's). A
    /// busy, gloss-stable context would otherwise pay one provider
    /// round trip per flush tick forever; an explicit caller still gets
    /// the unthrottled [`AppState::refresh_embeddings`] instead, so a
    /// deliberate refresh always heals a width change in one call.
    pub(crate) fn auto_refresh_embeddings(
        &self,
        name: &str,
        deadline: Deadline,
    ) -> Option<Result<(usize, usize), String>> {
        self.refresh_embeddings_inner(name, deadline, true)
    }

    fn refresh_embeddings_inner(
        &self,
        name: &str,
        deadline: Deadline,
        throttle_probe: bool,
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
        // edits — undetectable forever. A probe embedding keeps that
        // from hiding, UNLESS this is the throttled ticker path AND a
        // recent embed from any context (this one or another) already
        // confirmed the provider is still producing `carried`'s width
        // — the width is the provider's property, not this context's,
        // so that confirmation is just as good as one bought here. An
        // explicit caller (`throttle_probe: false`) always probes, so
        // it reliably heals a width change in this one call. Both
        // tables must be confirmed (whichever carry a width), matching
        // the mismatch check just below: a probe confined to labels
        // alone would miss a change confined to concepts, so a skip
        // confined to labels alone must not happen either.
        let width_confirmed = |carried: Option<usize>| {
            carried.is_none_or(|w| self.provider_width_recently_confirmed(w))
        };
        if failure.is_none()
            && !fresh_model
            && (carried_concepts_width.is_some() || carried_labels_width.is_some())
            && fresh_width.is_none()
            && !(throttle_probe
                && width_confirmed(carried_concepts_width)
                && width_confirmed(carried_labels_width))
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
            // The redo's agreement is already settled: `fresh_width` came
            // from this same pass's own rows, so a name already landed in
            // `embedded_concepts`/`embedded_labels` is already bought at
            // that exact width and must not be re-purchased — only names
            // still carried at the stale old width (or dropped by the
            // first pass's own `settled_width` disagreement) need a redo.
            let mut settled_width: Option<usize> = fresh_width;
            let concepts_redo: Vec<(String, String)> = concepts
                .iter()
                .filter(|(name, _)| !embedded_concepts.contains_key(name))
                .cloned()
                .collect();
            let (concepts_reembedded, concept_failure) = self.embed_stale(
                &*embedder,
                &existing.concepts,
                &concepts_redo,
                true,
                &mut settled_width,
                deadline,
            );
            embedded_concepts.extend(concepts_reembedded);
            let labels_redo: Vec<(String, String)> = labels
                .iter()
                .filter(|(name, _)| !embedded_labels.contains_key(name))
                .cloned()
                .collect();
            let (labels_reembedded, label_failure) = self.embed_stale(
                &*embedder,
                &existing.labels,
                &labels_redo,
                true,
                &mut settled_width,
                deadline,
            );
            embedded_labels.extend(labels_reembedded);
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
    /// is new or changed, `EMBED_CHUNK_SIZE` glosses per provider call. Each vector
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
        let stale_chunks: Vec<&[(String, String, u64)]> = stale.chunks(EMBED_CHUNK_SIZE).collect();
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

    /// The worker-pool size (`TAGURU_EMBED_PARALLEL`) each refresh
    /// dispatches its stale chunks under — see the field's own doc for
    /// why this is sized to the provider's rate limit, not the
    /// machine's core count. A caller fanning out ACROSS contexts, not
    /// just within one, sizes its pool by this same number too: with
    /// `embed_provider_slots` as the actual global ceiling, threads
    /// beyond it in either pool just queue for a permit rather than
    /// pushing real concurrency past what the provider was configured
    /// to take.
    pub fn embed_parallel(&self) -> usize {
        self.0.embed_parallel
    }

    /// The scrape-time gauge behind `taguru_embed_slot_waiters`:
    /// threads currently queued for a permit on `embed_provider_slots`
    /// (issue #563 item 4) — read lock-free, the same shape as
    /// `retrieval_cache_gauges`/`semantic_cache_entries`.
    pub fn embed_slot_waiters(&self) -> u64 {
        self.0.embed_provider_slots.waiting() as u64
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
    /// provider, `EMBED_CHUNK_SIZE` per call. The sidecar is written AT MOST ONCE per
    /// refresh: writing per batch would multiply a large store's bytes
    /// across the whole backfill. A provider failure partway persists
    /// what did land and reports the error — the next refresh continues
    /// from there instead of re-buying the same vectors.
    ///
    /// Always pays for its own width probe when one is needed — see
    /// [`AppState::auto_refresh_passage_embeddings`] for the throttled
    /// variant the auto-embed ticker uses instead, and
    /// [`AppState::refresh_embeddings`]'s doc for why the split exists.
    pub fn refresh_passage_embeddings(
        &self,
        name: &str,
        deadline: Deadline,
    ) -> Option<Result<PassageRefreshOutcome, String>> {
        self.refresh_passage_embeddings_inner(name, deadline, false)
    }

    /// The auto-embed ticker's variant of
    /// [`AppState::refresh_passage_embeddings`] — see
    /// [`AppState::auto_refresh_embeddings`]'s doc for the throttle
    /// this mirrors (issue #677 item 2).
    pub(crate) fn auto_refresh_passage_embeddings(
        &self,
        name: &str,
        deadline: Deadline,
    ) -> Option<Result<PassageRefreshOutcome, String>> {
        self.refresh_passage_embeddings_inner(name, deadline, true)
    }

    fn refresh_passage_embeddings_inner(
        &self,
        name: &str,
        deadline: Deadline,
        throttle_probe: bool,
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
        // Rows this pass already bought from the provider, kept across a
        // width-mismatch redo so the redo's `carried` lookup below can
        // reuse them instead of re-purchasing what already landed at the
        // (now-settled) fresh width — only names still carried at the
        // stale old width via `existing` need buying again.
        let mut bought: HashMap<(String, u32, u64, Option<u64>), Vec<f32>> = HashMap::new();
        // Once a redo fires, every row in the final store is considered
        // freshly embedded from the caller's perspective — carried rows
        // reused from `bought` above are not re-purchased, but they are
        // still new at this width, exactly as if they had been (matching
        // the pre-reuse contract every caller already relies on).
        let mut redo_triggered = false;
        let (fresh, embedded, skipped_over_limit, failure) = loop {
            let carried: HashMap<(&str, u32, u64, Option<u64>), &[f32]> = if fresh_model {
                bought
                    .iter()
                    .map(|(key, row)| ((key.0.as_str(), key.1, key.2, key.3), row.as_slice()))
                    .collect()
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

            let to_embed_chunks: Vec<&[(PassageKey, String)]> =
                to_embed.chunks(EMBED_CHUNK_SIZE).collect();
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
                            let settled = *fresh_width.get_or_insert(vector.len());
                            // `push` silently drops a row whose width
                            // disagrees with the dimension `fresh` already
                            // settled on (the same provider-mid-migration
                            // hazard `embed_stale` guards against for
                            // glosses) — count only the rows that actually
                            // landed, or `embedded` over-reports what
                            // `total_rows` below can already prove didn't
                            // all land.
                            let before = fresh.len();
                            fresh.push(key.clone(), vector.clone());
                            embedded += fresh.len() - before;
                            // Worth keeping for a width-mismatch redo to
                            // reuse even on the iteration where `fresh`
                            // itself just dropped it: that happens when a
                            // carried old-width row got pushed earlier in
                            // this same walk and locked `fresh`'s
                            // dimension first — exactly the row a redo
                            // discards, making this fetched vector the
                            // one it needs. Gated on agreement with this
                            // batch's own settled width, not on whether it
                            // landed in `fresh`, so a genuine
                            // provider-mid-migration split (disagreeing
                            // with this batch's first vector) still stays
                            // unbought.
                            if vector.len() == settled {
                                bought.insert(
                                    (key.source.clone(), key.index, key.hash, key.question_hash),
                                    vector,
                                );
                            }
                        }
                    }
                    Some(Err(error)) => failure = failure.or(Some(error)),
                    None => {}
                }
            }
            // Unchanged hashes embed nothing, which would leave the width
            // change of exactly this scenario — backend swap, no passage
            // edits — undetectable. A probe embedding keeps it from
            // hiding, matching the concept refresh — including the same
            // ticker-only skip when a recent embed from any context
            // already confirmed the provider is still producing
            // `carried_width` (see `provider_width_recently_confirmed`'s
            // doc: the width is the provider's property, not this
            // context's). An explicit caller always probes.
            if failure.is_none()
                && !fresh_model
                && carried_width
                    .is_some_and(|w| !(throttle_probe && self.provider_width_recently_confirmed(w)))
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
                redo_triggered = true;
                continue;
            }
            break (fresh, embedded, skipped_over_limit, failure);
        };
        let embedded = if redo_triggered {
            fresh.len()
        } else {
            embedded
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
        // Floor read and sidecar load share one scoped tombstone fence
        // (the guard doubles as the `meta` read — a second
        // `inner.read()` under it could deadlock behind a queued
        // writer). Dropped before the provider round trip below, which
        // must never make a delete wait on the network.
        let (context_floor, store) = {
            let fence = entry.read_unless_deleted()?;
            (
                fence.meta.semantic_floor,
                self.entry_vectors(&entry, &file_stem(name)),
            )
        };
        // One-call override beats the context setting beats the server
        // default (see [`DEFAULT_SEMANTIC_FLOOR`] for the calibration).
        let floor = floor_override
            .or(context_floor)
            .unwrap_or(self.0.default_semantic_floor)
            .clamp(0.0, 1.0);
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
        // Same scoped fence as `semantic_resolve` — floor and sidecar
        // under one guard, dropped before any provider call.
        let (context_floor, store) = {
            let fence = entry.read_unless_deleted()?;
            (
                fence.meta.semantic_floor,
                self.entry_vectors(&entry, &file_stem(name)),
            )
        };
        let floor = floor_override
            .or(context_floor)
            .unwrap_or(self.0.default_semantic_floor)
            .clamp(0.0, 1.0);
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
    ///
    /// A genuine read failure (not simply "nothing embedded yet") is
    /// quarantined exactly like `entry_passages`' load failure (issue
    /// #677 item 3): the failed attempt is NOT cached into `vectors` —
    /// only an empty, uncached default is handed back, and the next
    /// call retries the disk once `LOAD_FAILURE_RETRY` has passed.
    /// Before this, a transient disk hiccup at load time cached an
    /// empty store forever (until an explicit refresh or eviction), so
    /// semantic search silently found nothing for the rest of that
    /// residency even after the disk recovered.
    fn entry_vectors(&self, entry: &Entry, stem: &str) -> Arc<VectorStore> {
        let mut cached = entry.vectors.lock();
        if let Some(store) = &*cached {
            return Arc::clone(store);
        }
        {
            let failure = entry.vectors_load_failure.lock();
            if let Some(failed_at) = &*failure
                && still_quarantined(failed_at)
            {
                return Arc::new(VectorStore::default());
            }
        }
        match VectorStore::load_checked(&vectors_path(&self.0.data_dir, stem)) {
            Ok(store) => {
                *entry.vectors_load_failure.lock() = None;
                let store = Arc::new(store);
                *cached = Some(Arc::clone(&store));
                store
            }
            Err(()) => {
                *entry.vectors_load_failure.lock() = Some(std::time::Instant::now());
                Arc::new(VectorStore::default())
            }
        }
    }
}

#[cfg(test)]
#[path = "embeddings/gloss_tests.rs"]
mod gloss_tests;
#[path = "embeddings/passage_tests.rs"]
#[cfg(test)]
mod passage_tests;
