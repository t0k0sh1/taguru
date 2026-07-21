#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use parking_lot::Mutex;
    use taguru::deadline::Deadline;

    use crate::embedding::{EmbedPurpose, EmbeddingProvider, VectorStore};
    use crate::registry::test_support::{
        MockEmbeddings, SlowEmbeddings, assoc_op, rendered, scratch_dir,
    };
    use crate::registry::{
        AppState, BootOptions, ContextMeta, GlossLaneReport, SEMANTIC_RESOLVE_LIMIT, file_stem,
        vectors_path,
    };

    /// An embedding refresh that published something bumps the config
    /// counter; the idempotent second pass must not churn caches
    /// (#149).
    #[test]
    fn embedding_refresh_bumps_config_only_when_it_publishes() {
        let dir = scratch_dir("revision-refresh");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let embedder = Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state.create("fruit", ContextMeta::default()).unwrap();
        state
            .add_associations(
                "fruit",
                vec![assoc_op("りんご", "分類", "果物", 1.0, None)],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(state.context_revision("fruit").unwrap().config, 0);
        state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            state.context_revision("fruit").unwrap().config,
            1,
            "vectors the semantic lane now serves are a config change"
        );
        state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            state.context_revision("fruit").unwrap().config,
            1,
            "a refresh that embedded nothing new bumps nothing"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn embedding_calls_record_success_and_failure() {
        /// Same model name as the mock, so stored vectors stay usable,
        /// but every provider round trip fails.
        struct FailingEmbeddings;
        impl EmbeddingProvider for FailingEmbeddings {
            fn model(&self) -> &str {
                "mock"
            }
            fn embed(
                &self,
                _texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                Err("provider down".to_string())
            }
        }

        let dir = scratch_dir("m-embed");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        {
            let embedder =
                Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
            let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
            state
                .create("fruit", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .write_context("fruit", |context| {
                    context.associate("りんご", "分類", "果物", 1.0).unwrap();
                })
                .map_err(|_| "write")
                .unwrap();
            state
                .refresh_embeddings("fruit", Deadline::unbounded())
                .unwrap()
                .unwrap();
            // One batch per namespace: two successful provider calls.
            assert!(rendered(&state).contains(
                "taguru_embedding_requests_total{operation=\"refresh\",outcome=\"ok\"} 2"
            ));
            state.flush_dirty();
        }

        // Same data, failing provider: the resolve-path cue embedding
        // fails and is counted as such.
        let embedder = Some(Arc::new(FailingEmbeddings) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        assert!(
            state
                .semantic_resolve("fruit", "アップル", false, None, Deadline::unbounded())
                .unwrap()
                .is_err()
        );
        let body = rendered(&state);
        assert!(
            body.contains(
                "taguru_embedding_requests_total{operation=\"resolve\",outcome=\"failed\"} 1"
            ),
            "{body}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// `semantic_resolve` deliberately folds provider-off, model-changed,
    /// and nothing-embedded into one empty answer; its explain twin must
    /// hold them apart, and must place an expected name in exactly the
    /// ordering `semantic_resolve` truncates.
    #[test]
    fn explain_semantic_resolve_names_what_semantic_resolve_folds() {
        let dir = scratch_dir("sem-explain");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        {
            let embedder =
                Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
            let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
            state
                .create("fruit", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .write_context("fruit", |context| {
                    context.associate("りんご", "分類", "果物", 1.0).unwrap();
                })
                .map_err(|_| "write")
                .unwrap();

            // Before any refresh: nothing is embedded, and the report
            // says that — not "empty answer", not "model changed".
            assert!(matches!(
                state
                    .explain_semantic_resolve(
                        "fruit",
                        "アップル",
                        "りんご",
                        false,
                        None,
                        Deadline::unbounded()
                    )
                    .unwrap(),
                GlossLaneReport::EmptyTable
            ));

            state
                .refresh_embeddings("fruit", Deadline::unbounded())
                .unwrap()
                .unwrap();

            // The expected name's own cosine and its rank in the very
            // ordering semantic_resolve serves.
            let Some(GlossLaneReport::Ran {
                floor,
                cosine: Some(cosine),
                rank,
                passing,
                cap,
            }) = state.explain_semantic_resolve(
                "fruit",
                "アップル",
                "りんご",
                false,
                None,
                Deadline::unbounded(),
            )
            else {
                panic!("the sweep should have run with a cosine for りんご");
            };
            assert!((cosine - 0.96).abs() < 1e-6);
            assert_eq!(rank, Some(1));
            assert_eq!(passing, 1, "果物's cosine 0.0 sits under the floor");
            assert_eq!(cap, SEMANTIC_RESOLVE_LIMIT);
            assert!(floor > 0.0);

            // A below-floor name reports its cosine with no rank — the
            // "scored 0.0, floor 0.35" evidence — and a floor override
            // seats it, in semantic_resolve's exact order.
            let Some(GlossLaneReport::Ran {
                cosine: Some(low),
                rank: None,
                ..
            }) = state.explain_semantic_resolve(
                "fruit",
                "アップル",
                "果物",
                false,
                None,
                Deadline::unbounded(),
            )
            else {
                panic!("果物 has a vector; its cosine must be reported");
            };
            assert!(low.abs() < 1e-6);
            let Some(GlossLaneReport::Ran {
                rank: Some(rank),
                passing,
                ..
            }) = state.explain_semantic_resolve(
                "fruit",
                "アップル",
                "果物",
                false,
                Some(0.0),
                Deadline::unbounded(),
            )
            else {
                panic!("floor 0.0 must seat 果物");
            };
            assert_eq!((rank, passing), (2, 2));
            let served = state
                .semantic_resolve("fruit", "アップル", false, Some(0.0), Deadline::unbounded())
                .unwrap()
                .unwrap();
            assert_eq!(
                served[rank - 1].0,
                "果物",
                "rank must match the serve order"
            );

            // A name added after the refresh has no vector yet: the
            // sweep runs, its cosine does not exist.
            state
                .write_context("fruit", |context| {
                    context.associate("バナナ", "分類", "果物", 1.0).unwrap();
                })
                .map_err(|_| "write")
                .unwrap();
            assert!(matches!(
                state
                    .explain_semantic_resolve(
                        "fruit",
                        "アップル",
                        "バナナ",
                        false,
                        None,
                        Deadline::unbounded(),
                    )
                    .unwrap(),
                GlossLaneReport::Ran { cosine: None, .. }
            ));
            state.flush_dirty();
        }

        // Same sidecar, another model: named as the reason.
        struct OtherEmbeddings;
        impl EmbeddingProvider for OtherEmbeddings {
            fn model(&self) -> &str {
                "other-model"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                Ok(texts.iter().map(|_| vec![1.0, 0.0, 0.0]).collect())
            }
        }
        let state = AppState::boot(
            dir.clone(),
            usize::MAX,
            Some(Arc::new(OtherEmbeddings) as Arc<dyn EmbeddingProvider>),
        )
        .unwrap();
        assert!(matches!(
            state
                .explain_semantic_resolve(
                    "fruit",
                    "アップル",
                    "りんご",
                    false,
                    None,
                    Deadline::unbounded(),
                )
                .unwrap(),
            GlossLaneReport::ModelChanged { .. }
        ));
        // A context that does not exist is the outer None — but only
        // once a provider exists to get past the Off arm.
        assert!(
            state
                .explain_semantic_resolve(
                    "nazo",
                    "アップル",
                    "りんご",
                    false,
                    None,
                    Deadline::unbounded(),
                )
                .is_none()
        );

        // No provider at all: Off before any lookup, exactly where
        // semantic_resolve answers its empty list. (Shadowing keeps the
        // previous state — and its data-dir lock — alive to scope end,
        // so release it by hand.)
        drop(state);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(matches!(
            state
                .explain_semantic_resolve(
                    "fruit",
                    "アップル",
                    "りんご",
                    false,
                    None,
                    Deadline::unbounded(),
                )
                .unwrap(),
            GlossLaneReport::Off
        ));

        let _ = fs::remove_dir_all(dir);
    }

    /// The two provider call sites declare opposite purposes: gloss
    /// refresh embeds as `Index`, live cue resolution as `Query` — the
    /// distinction an asymmetric-model proxy keys `input_type` on.
    #[test]
    fn refresh_embeds_as_index_and_cue_resolution_as_query() {
        struct RecordingEmbeddings(Arc<Mutex<Vec<EmbedPurpose>>>);
        impl EmbeddingProvider for RecordingEmbeddings {
            fn model(&self) -> &str {
                "recorder"
            }
            fn embed(
                &self,
                texts: &[&str],
                purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                self.0.lock().push(purpose);
                Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
            }
        }

        let dir = scratch_dir("purpose");
        let purposes = Arc::new(Mutex::new(Vec::new()));
        let embedder = Some(
            Arc::new(RecordingEmbeddings(Arc::clone(&purposes))) as Arc<dyn EmbeddingProvider>
        );
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("p", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .write_context("p", |context| {
                context.associate("a", "l", "b", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();
        state
            .refresh_embeddings("p", Deadline::unbounded())
            .unwrap()
            .unwrap();
        state
            .semantic_resolve("p", "cue", false, None, Deadline::unbounded())
            .unwrap()
            .unwrap();

        let seen = purposes.lock().clone();
        let (cue_call, refresh_calls) = seen.split_last().unwrap();
        assert!(!refresh_calls.is_empty());
        assert!(refresh_calls.iter().all(|p| *p == EmbedPurpose::Index));
        assert_eq!(*cue_call, EmbedPurpose::Query);

        let _ = fs::remove_dir_all(dir);
    }

    /// A provider that changes output width behind a stable model name
    /// (a backend swap behind the same proxy) must stale the whole
    /// carried table: gloss hashes are unchanged, so without the width
    /// check nothing re-embeds and old-width rows sit next to new-width
    /// ones — which `similarity` scores as nothing, silently.
    #[test]
    fn a_width_change_under_the_same_model_name_re_embeds_everything() {
        struct WidthEmbeddings(usize);
        impl EmbeddingProvider for WidthEmbeddings {
            fn model(&self) -> &str {
                "stable-name"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                Ok(texts
                    .iter()
                    .map(|_| {
                        let mut vector = vec![0.0; self.0];
                        vector[0] = 1.0;
                        vector
                    })
                    .collect())
            }
        }

        let dir = scratch_dir("width-change");
        {
            let embedder = Some(Arc::new(WidthEmbeddings(2)) as Arc<dyn EmbeddingProvider>);
            let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
            state
                .create("w", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .write_context("w", |context| {
                    context.associate("a", "l", "b", 1.0).unwrap();
                })
                .map_err(|_| "write")
                .unwrap();
            let (embedded, total) = state
                .refresh_embeddings("w", Deadline::unbounded())
                .unwrap()
                .unwrap();
            assert_eq!((embedded, total), (3, 3)); // a, b, and the label l
            state.flush_dirty();
        }

        // Same model name, wider vectors: every gloss must re-embed
        // (hashes alone would say "nothing to do") and the published
        // sidecar must be uniformly the new width.
        let embedder = Some(Arc::new(WidthEmbeddings(3)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        let (embedded, total) = state
            .refresh_embeddings("w", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!((embedded, total), (3, 3));
        let store = VectorStore::load(&vectors_path(&dir, &file_stem("w")));
        assert!(
            store
                .concepts
                .values()
                .chain(store.labels.values())
                .all(|(_, vector)| vector.len() == 3),
            "old-width rows must not survive the width change"
        );

        // A no-op refresh against the same-width provider stays a no-op
        // (the probe embeds one gloss but re-embeds nothing).
        let (embedded, total) = state
            .refresh_embeddings("w", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!((embedded, total), (0, 3));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_width_drift_confined_to_the_label_table_is_still_caught() {
        struct FixedWidthEmbeddings(usize);
        impl EmbeddingProvider for FixedWidthEmbeddings {
            fn model(&self) -> &str {
                "stable-name"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                Ok(texts
                    .iter()
                    .map(|_| {
                        let mut vector = vec![0.0; self.0];
                        vector[0] = 1.0;
                        vector
                    })
                    .collect())
            }
        }

        let dir = scratch_dir("label-only-width-drift");
        let path = vectors_path(&dir, &file_stem("w"));
        {
            let embedder = Some(Arc::new(FixedWidthEmbeddings(3)) as Arc<dyn EmbeddingProvider>);
            let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
            state
                .create("w", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .write_context("w", |context| {
                    context.associate("a", "l", "b", 1.0).unwrap();
                })
                .map_err(|_| "write")
                .unwrap();
            state
                .refresh_embeddings("w", Deadline::unbounded())
                .unwrap()
                .unwrap();
            state.flush_dirty();
        }

        // Shrink only the label table's vectors in place, keeping their
        // hash unchanged — the shape a width change confined to one
        // table (a partial backend rollout, or a prior pass that only
        // reconciled concepts) would leave on disk. `carried_width`
        // sampling concepts first — as this used to — would see
        // concepts already at width 3, call that "no drift", and never
        // look at labels at all.
        let mut store = VectorStore::load(&path);
        for (_, vector) in store.labels.values_mut() {
            vector.truncate(2);
        }
        store.save(&path).unwrap();

        // Same model name, same provider width (3): a no-op content
        // diff, so nothing re-embeds and only the probe/reconciliation
        // path can notice the label table is still stuck at width 2.
        let embedder = Some(Arc::new(FixedWidthEmbeddings(3)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        let (embedded, total) = state
            .refresh_embeddings("w", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (embedded, total),
            (3, 3),
            "a width drift confined to the label table must still force a full re-embed"
        );
        let reloaded = VectorStore::load(&path);
        assert!(
            reloaded
                .concepts
                .values()
                .chain(reloaded.labels.values())
                .all(|(_, vector)| vector.len() == 3),
            "the label table's stale width must not survive reconciliation"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    #[cfg(unix)]
    fn refresh_embeddings_does_not_rebuy_rows_a_failed_save_already_bought() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("gvec-save-fail");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let embedder = Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .write_context("fruit", |context| {
                context.associate("りんご", "l", "アップル", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();

        // The disk goes bad right before the save: the provider still
        // gets paid (embed happens before the write), but the sidecar
        // write fails.
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
        let error = state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap_err();
        assert!(error.contains("not persisted"), "{error}");
        let calls_after_failure = calls.load(Ordering::Relaxed);
        assert!(
            calls_after_failure > 0,
            "the provider must have been paid before the save failed"
        );
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();

        // The disk recovers: the retry must not re-embed the rows the
        // failed save already bought (a width probe still spends one
        // call on a no-op refresh, same as any other — see
        // a_width_change_under_the_same_model_name_re_embeds_everything),
        // yet it must still retry the write so those rows do not stay
        // unpersisted forever.
        let (embedded, total) = state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            embedded, 0,
            "must not re-embed what the failed save already cached"
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            calls_after_failure + 1,
            "only the width probe's one call, not a re-embed of the cached rows"
        );

        let store = VectorStore::load(&vectors_path(&dir, &file_stem("fruit")));
        assert_eq!(
            store.concepts.len() + store.labels.len(),
            total,
            "the retried save must have actually landed on disk"
        );
        assert!(total > 0);

        let _ = fs::remove_dir_all(dir);
    }

    /// Chunks within one `embed_stale` call dispatch concurrently, so a
    /// provider mid-migration can answer two chunks of the very same
    /// call with different widths. `VectorTable` has no dimension of its
    /// own to enforce (unlike `PassageVectorStore`), so without a guard
    /// in the merge loop the disagreeing chunk would land right next to
    /// the rest, corrupting the persisted table with no error — just a
    /// `similarity` that silently stops matching for those rows.
    #[test]
    fn embed_stale_drops_a_chunk_whose_width_disagrees_with_the_rest_of_the_batch() {
        struct SplitWidthEmbeddings;
        impl EmbeddingProvider for SplitWidthEmbeddings {
            fn model(&self) -> &str {
                "split-width"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                // The full 128-entry chunk answers at width 2; any
                // smaller chunk (the trailing remainder, or the
                // single-label call) answers at width 3 — a provider
                // mid-migration serving two backend versions to
                // concurrent connections.
                let width = if texts.len() >= 128 { 2 } else { 3 };
                Ok(texts.iter().map(|_| vec![0.0; width]).collect())
            }
        }

        let dir = scratch_dir("gloss-split-width");
        let embedder = Some(Arc::new(SplitWidthEmbeddings) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            embedder,
            BootOptions {
                embed_parallel: 2,
                ..BootOptions::default()
            },
        )
        .unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .write_context("fruit", |context| {
                for i in 0..129 {
                    context
                        .associate(format!("c{i}"), "属性", "値", 1.0)
                        .unwrap();
                }
            })
            .map_err(|_| "write")
            .unwrap();

        state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let store = VectorStore::load(&vectors_path(&dir, &file_stem("fruit")));
        assert_eq!(
            store.concepts.len(),
            128,
            "the 128-item chunk lands; the remainder disagreed on width and was dropped"
        );
        assert!(
            store.concepts.values().all(|(_, v)| v.len() == 2),
            "a disagreeing vector must never reach the persisted concept table"
        );
        // The width agreement spans BOTH tables: the label call answered
        // width 3, which disagrees with the width the concept call
        // already settled, so it drops too — a store persisting concepts
        // at one width and labels at another is exactly the mixed file
        // the loader refuses whole (#133).
        assert!(
            store.labels.is_empty(),
            "a label at a width the refresh did not settle on must stay stale"
        );
        // Flush before the reboot below: write_context's association is
        // otherwise only durable on the next periodic flush, and the
        // reboot must see it. Then release the data-directory lock so
        // the reboot can open the same directory again.
        state.flush_dirty();
        drop(state);

        // The dropped remainder is still stale; once the provider
        // stops disagreeing with itself, the next refresh picks it up.
        struct ConsistentEmbeddings;
        impl EmbeddingProvider for ConsistentEmbeddings {
            fn model(&self) -> &str {
                "split-width"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                Ok(texts.iter().map(|_| vec![0.0; 2]).collect())
            }
        }
        let embedder = Some(Arc::new(ConsistentEmbeddings) as Arc<dyn EmbeddingProvider>);
        let state =
            AppState::boot_with(dir.clone(), usize::MAX, embedder, BootOptions::default()).unwrap();
        state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        let store = VectorStore::load(&vectors_path(&dir, &file_stem("fruit")));
        assert!(
            store.concepts.len() > 128,
            "the previously dropped remainder must still be stale and get embedded now"
        );
        assert!(store.concepts.values().all(|(_, v)| v.len() == 2));
        assert_eq!(
            store.labels.len(),
            1,
            "the dropped label was stale all along and lands once the provider settles"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gloss_refresh_prunes_vectors_for_a_concept_dropped_by_compaction() {
        let dir = scratch_dir("gloss-prune");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let embedder = Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "sake",
                vec![
                    assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md")),
                    assoc_op("蔵", "廃止銘柄", "旧銘", 1.0, Some("gone.md")),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();

        let (_, total) = state
            .refresh_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            total, 5,
            "concepts 蔵/高瀬/旧銘 plus labels 杜氏/廃止銘柄 all embed"
        );

        // Retract the only source behind 旧銘/廃止銘柄, then compact so
        // those names actually leave the graph.
        state.retract_source("sake", "gone.md").unwrap();
        state
            .compact_context("sake", Deadline::unbounded())
            .unwrap();

        let (_, total) = state
            .refresh_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            total, 3,
            "the vanished concept 旧銘 and label 廃止銘柄 must not linger as ghost rows"
        );
        let sidecar = VectorStore::load(&vectors_path(&dir, &file_stem("sake")));
        assert!(
            !sidecar.concepts.contains_key("旧銘"),
            "the dropped concept's row reached neither memory nor disk"
        );
        assert!(
            !sidecar.labels.contains_key("廃止銘柄"),
            "the dropped label's row reached neither memory nor disk"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gloss_refresh_keeps_concept_vectors_when_the_label_table_fails() {
        /// Succeeds except on exactly its `fail_on`-th call (0-based).
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

        let dir = scratch_dir("gloss-partial");
        // Concepts embed on call 0 (success); the labels table is call 1,
        // the one that fails.
        let embedder = Some(Arc::new(FlakyEmbeddings {
            calls: std::sync::atomic::AtomicUsize::new(0),
            fail_on: 1,
        }) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "sake",
                vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md"))],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();

        let error = state
            .refresh_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap_err();
        assert!(error.contains("hiccup"), "{error}");
        let sidecar = VectorStore::load(&vectors_path(&dir, &file_stem("sake")));
        assert_eq!(
            sidecar.concepts.len(),
            2,
            "the concepts the provider already billed for stay durable despite the label failure"
        );
        assert!(
            sidecar.labels.is_empty(),
            "the failed label table wrote nothing"
        );

        // The next refresh buys only the labels the first pass missed.
        let (embedded, total) = state
            .refresh_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!((embedded, total), (1, 3));

        let _ = fs::remove_dir_all(dir);
    }

    /// A width change behind a stable model name must still force a
    /// full re-embed even when the *other* table's call fails in the
    /// same pass — that failure must not excuse persisting this pass's
    /// concepts at the new width right next to labels still at the old
    /// one.
    #[test]
    fn gloss_width_reconciliation_fires_even_when_a_sibling_table_fails() {
        /// Succeeds except on exactly its `fail_on`-th call (0-based);
        /// every successful call answers at `width`.
        struct FlakyWidthEmbeddings {
            calls: AtomicUsize,
            fail_on: usize,
            width: usize,
        }
        impl EmbeddingProvider for FlakyWidthEmbeddings {
            fn model(&self) -> &str {
                "flaky-width"
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
                Ok(texts.iter().map(|_| vec![0.0; self.width]).collect())
            }
        }

        let dir = scratch_dir("gloss-width-reconcile");
        // First boot: establish a carried width of 2.
        {
            let embedder = Some(Arc::new(FlakyWidthEmbeddings {
                calls: AtomicUsize::new(0),
                fail_on: usize::MAX,
                width: 2,
            }) as Arc<dyn EmbeddingProvider>);
            let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
            state
                .create("w", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .write_context("w", |context| {
                    context.associate("a", "l", "b", 1.0).unwrap();
                })
                .map_err(|_| "write")
                .unwrap();
            state
                .refresh_embeddings("w", Deadline::unbounded())
                .unwrap()
                .unwrap();
            state.flush_dirty();
        }

        // Second boot: same model name, width now 3, plus a brand-new
        // association so both tables carry genuinely stale content —
        // an unchanged-content reboot would leave nothing stale and
        // fall to the single probe call instead, never exercising two
        // independent per-table calls in the same pass. Concepts embed
        // on call 0 (succeeds, proving the width changed); labels are
        // call 1, the one that fails.
        let embedder = Some(Arc::new(FlakyWidthEmbeddings {
            calls: AtomicUsize::new(0),
            fail_on: 1,
            width: 3,
        }) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .write_context("w", |context| {
                context.associate("c", "m", "d", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();
        let (embedded, total) = state
            .refresh_embeddings("w", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (embedded, total),
            (6, 6),
            "the reconciliation retry re-embeds everything, old and new alike, and succeeds"
        );
        let store = VectorStore::load(&vectors_path(&dir, &file_stem("w")));
        assert!(
            store
                .concepts
                .values()
                .chain(store.labels.values())
                .all(|(_, v)| v.len() == 3),
            "a sibling table's transient failure must not leave a mixed-width store live"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// `existing`/`embed_stale` run before the entry's data lock is
    /// ever taken — provider round trips can take seconds and must not
    /// block graph reads — so two concurrent first-time refreshes would
    /// both diff against the same empty sidecar and both call the
    /// provider. Unless `vectors_refresh` excludes them for the whole
    /// refresh (not just the merge), those two provider calls overlap;
    /// whichever refresh then merges last silently wins over the
    /// other's, with no ordering guarantee that the winner saw the
    /// newer gloss. This pins the observable down directly: the
    /// provider must never see two calls in flight at once.
    #[test]
    fn concurrent_gloss_refreshes_serialize_their_provider_calls() {
        use std::thread;

        let dir = scratch_dir("refresh-serialize");
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let embedder = Some(Arc::new(SlowEmbeddings {
            in_flight: Arc::clone(&in_flight),
            peak: Arc::clone(&peak),
        }) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .write_context("fruit", |context| {
                context.associate("りんご", "分類", "果物", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();

        let mut refreshers = Vec::new();
        for _ in 0..2 {
            let state = state.clone();
            refreshers.push(thread::spawn(move || {
                state
                    .refresh_embeddings("fruit", Deadline::unbounded())
                    .unwrap()
                    .unwrap();
            }));
        }
        for refresher in refreshers {
            refresher.join().unwrap();
        }

        assert_eq!(
            peak.load(Ordering::SeqCst),
            1,
            "two first-time refreshes both diff against an empty sidecar; without \
             vectors_refresh serializing the whole refresh, their provider calls \
             overlap and whichever merges last can clobber a fresher gloss with a \
             staler one"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn gloss_refresh_dispatches_chunks_concurrently_when_embed_parallel_is_raised() {
        let dir = scratch_dir("gloss-parallel");
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let embedder = Some(Arc::new(SlowEmbeddings {
            in_flight: Arc::clone(&in_flight),
            peak: Arc::clone(&peak),
        }) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            embedder,
            BootOptions {
                embed_parallel: 2,
                ..BootOptions::default()
            },
        )
        .unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .write_context("fruit", |context| {
                // 129 stale concepts split into a 128-item chunk and a
                // 1-item chunk; with embed_parallel=2 both dispatch at
                // once.
                for i in 0..129 {
                    context
                        .associate(format!("c{i}"), "属性", "値", 1.0)
                        .unwrap();
                }
            })
            .map_err(|_| "write")
            .unwrap();

        state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        assert!(
            peak.load(Ordering::SeqCst) > 1,
            "129 stale concepts split into two chunks; with embed_parallel=2 both \
             should reach the provider at once"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// Two contexts refreshing at once (what the flush tick's outer
    /// `parallel_map` does) each also split into two chunks internally
    /// (what `dispatch_chunks_concurrently` does within one refresh) —
    /// nested, `embed_parallel=2` on both axes could reach 4 concurrent
    /// provider calls without a shared ceiling. `embed_provider_slots`
    /// must hold the true peak at `embed_parallel`, not its square.
    #[test]
    fn embed_provider_slots_cap_concurrency_across_contexts_and_chunks() {
        let dir = scratch_dir("embed-global-ceiling");
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let embedder = Some(Arc::new(SlowEmbeddings {
            in_flight: Arc::clone(&in_flight),
            peak: Arc::clone(&peak),
        }) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            embedder,
            BootOptions {
                embed_parallel: 2,
                ..BootOptions::default()
            },
        )
        .unwrap();
        for name in ["fruit", "veg"] {
            state
                .create(name, ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .write_context(name, |context| {
                    // 129 stale concepts per context, same as the
                    // single-context test above: a 128-item chunk and a
                    // 1-item chunk, so each context's own refresh fans
                    // out to two inner threads.
                    for i in 0..129 {
                        context
                            .associate(format!("c{i}"), "属性", "値", 1.0)
                            .unwrap();
                    }
                })
                .map_err(|_| "write")
                .unwrap();
        }

        std::thread::scope(|scope| {
            for name in ["fruit", "veg"] {
                scope.spawn(|| {
                    state
                        .refresh_embeddings(name, Deadline::unbounded())
                        .unwrap()
                        .unwrap();
                });
            }
        });

        assert_eq!(
            peak.load(Ordering::SeqCst),
            2,
            "two contexts each fanning out two chunks must still cap at \
             embed_parallel=2 concurrent provider calls process-wide, not \
             the 4 a per-pool-only ceiling would allow"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn semantic_fallback_lands_paraphrases_after_refresh() {
        let dir = scratch_dir("embed");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let embedder = Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .write_context("fruit", |context| {
                context.associate("りんご", "分類", "果物", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();

        // アップル shares no normalized characters with りんご: every
        // lexical tier misses, and before a refresh so does semantics.
        let lexical = state
            .read_context("fruit", |context| context.resolve("アップル"))
            .map_err(|_| "read")
            .unwrap();
        assert!(lexical.is_empty());
        assert!(
            state
                .semantic_resolve("fruit", "アップル", false, None, Deadline::unbounded())
                .unwrap()
                .unwrap()
                .is_empty()
        );

        // Refresh embeds every canonical name's gloss once; a second run
        // is a no-op.
        let (embedded, total) = state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(embedded, 3); // りんご, 果物 + label 分類
        assert_eq!(total, 3);
        assert_eq!(
            state
                .refresh_embeddings("fruit", Deadline::unbounded())
                .unwrap()
                .unwrap()
                .0,
            0
        );

        // Now the paraphrase lands on the stored spelling by cosine, and
        // unrelated names stay under the floor.
        let hits = state
            .semantic_resolve("fruit", "アップル", false, None, Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].0, "りんご");
        assert!(hits[0].1 > 0.9);

        // A new fact changes りんご's gloss: the next refresh re-embeds
        // exactly what changed — りんご plus the new 青森 and 産地 —
        // while 果物 and 分類, whose glosses are untouched, are not
        // re-sent to the provider.
        state
            .write_context("fruit", |context| {
                context.associate("りんご", "産地", "青森", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();
        let (embedded, total) = state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(embedded, 3);
        assert_eq!(total, 5);

        assert!(
            state
                .semantic_resolve("nope", "x", false, None, Deadline::unbounded())
                .is_none()
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn semantic_path_caches_cue_vectors_and_the_sidecar() {
        let dir = scratch_dir("semcache");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let embedder = Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), 1, embedder).unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .write_context("fruit", |context| {
                context.associate("りんご", "分類", "果物", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();
        state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        // One batch per namespace: concepts, then labels.
        assert_eq!(calls.load(Ordering::Relaxed), 2);

        // First query embeds the cue; repeating the wording does not.
        let first = state
            .semantic_resolve("fruit", "アップル", false, None, Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(first[0].0, "りんご");
        assert_eq!(calls.load(Ordering::Relaxed), 3);
        state
            .semantic_resolve("fruit", "アップル", false, None, Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 3, "cue must come from cache");

        // The sidecar is held in memory after first use: even with the
        // file gone, the same query keeps answering.
        fs::remove_file(vectors_path(&dir, &file_stem("fruit"))).unwrap();
        let held = state
            .semantic_resolve("fruit", "アップル", false, None, Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(held[0].0, "りんご");

        // Eviction clears the cached store (budget is one byte, and the
        // vector cache counts): after touching another context, the
        // deleted sidecar means no vectors — proving the memory copy
        // was dropped rather than leaked.
        state
            .create("other", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .read_context("other", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert!(
            state
                .semantic_resolve("fruit", "アップル", false, None, Deadline::unbounded())
                .unwrap()
                .unwrap()
                .is_empty()
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn semantic_twins_surface_synonym_forks_from_stored_vectors() {
        let dir = scratch_dir("twins");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        // Two label glosses embed close together: a synonym fork that
        // no spelling comparison could see.
        let embedder = MockEmbeddings {
            keys: vec![
                ("創業年".to_string(), vec![1.0, 0.0, 0.0]),
                ("設立年".to_string(), vec![0.95, 0.31, 0.0]),
            ],
            calls: Arc::clone(&calls),
        };
        let embedder = Some(Arc::new(embedder) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .write_context("sake", |context| {
                context
                    .associate("青嶺酒造", "創業年", "1907年", 1.0)
                    .unwrap();
                context
                    .associate("別の蔵", "設立年", "1950年", 1.0)
                    .unwrap();
            })
            .map_err(|_| "write")
            .unwrap();

        // Before any vectors exist the semantic half is skipped, loudly.
        let (concepts, labels, note) = state
            .semantic_twins("sake", 0.6, Deadline::unbounded())
            .unwrap();
        assert!(concepts.is_empty() && labels.is_empty());
        assert!(note.is_some());

        state
            .refresh_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        let (concepts, labels, note) = state
            .semantic_twins("sake", 0.6, Deadline::unbounded())
            .unwrap();
        assert!(note.is_none());
        // Directly connected concepts (青嶺酒造 —創業年→ 1907年) are
        // related, not duplicates, and must be filtered out however
        // similar their vectors are.
        let pairs_up = |a: &str, b: &str, x: &str, y: &str| a.contains(x) && b.contains(y);
        assert!(
            concepts
                .iter()
                .all(|(a, b, _)| !pairs_up(a, b, "青嶺酒造", "1907年")
                    && !pairs_up(a, b, "1907年", "青嶺酒造")),
            "{concepts:?}"
        );
        assert_eq!(labels.len(), 1, "{labels:?}");
        assert_eq!(
            (labels[0].0.as_str(), labels[0].1.as_str()),
            ("創業年", "設立年")
        );
        assert!(labels[0].2 > 0.9);

        // No provider round trip happens for the sweep itself: the two
        // audits above added no embed calls beyond the refresh batches
        // (2 namespaces) — stored vectors are compared directly.
        assert_eq!(calls.load(Ordering::Relaxed), 2);

        assert!(
            state
                .semantic_twins("nope", 0.6, Deadline::unbounded())
                .is_none()
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn semantic_floor_is_tunable_per_context_and_per_call() {
        let dir = scratch_dir("semfloor");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let embedder = Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .write_context("fruit", |context| {
                context.associate("りんご", "分類", "果物", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();
        state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        // みかん×りんご sits at cosine 0.28 — under the 0.35 default.
        let miss = |floor: Option<f32>| {
            state
                .semantic_resolve("fruit", "みかん", false, floor, Deadline::unbounded())
                .unwrap()
                .unwrap()
        };
        assert!(miss(None).is_empty());
        // A one-call override admits it without changing the context ...
        assert_eq!(miss(Some(0.2))[0].0, "りんご");
        assert!(miss(None).is_empty());
        // ... and the context setting changes the default, persisting
        // in the sidecar across a reboot.
        state
            .update_meta("fruit", None, None, None, Some(0.2))
            .unwrap()
            .unwrap();
        assert_eq!(miss(None)[0].0, "りんご");
        assert_eq!(state.directory()[0].semantic_floor, Some(0.2));

        let _ = fs::remove_dir_all(dir);
    }

    /// TAGURU_SEMANTIC_FLOOR reaches boot as a server-wide default that
    /// sits UNDER the per-context setting and the per-call override —
    /// it recalibrates the floor for the configured embedding model
    /// without touching any context.
    #[test]
    fn semantic_floor_server_default_recalibrates_under_context_and_call() {
        let dir = scratch_dir("semfloor-srv");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let embedder = Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            embedder,
            BootOptions {
                default_semantic_floor: Some(0.2),
                ..BootOptions::default()
            },
        )
        .unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .write_context("fruit", |context| {
                context.associate("りんご", "分類", "果物", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();
        state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let hits = |floor: Option<f32>| {
            state
                .semantic_resolve("fruit", "みかん", false, floor, Deadline::unbounded())
                .unwrap()
                .unwrap()
        };
        // みかん×りんご = cosine 0.28: lost under the built-in 0.35,
        // admitted by the recalibrated server default.
        assert_eq!(hits(None)[0].0, "りんご");
        // The context setting still beats the server default ...
        state
            .update_meta("fruit", None, None, None, Some(0.9))
            .unwrap()
            .unwrap();
        assert!(hits(None).is_empty());
        // ... and the one-call override still beats them both.
        assert_eq!(hits(Some(0.1))[0].0, "りんご");

        let _ = fs::remove_dir_all(dir);
    }
}
