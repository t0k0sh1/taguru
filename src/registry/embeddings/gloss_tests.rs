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

    /// The mid-race view a delete leaves behind: the entry still
    /// reachable (a status request's `lookup` won the race) but its
    /// tombstone already planted. Every embeddings read must answer
    /// "no such context" — the same 404 the context endpoint gives —
    /// not a status built from unlinked (or a successor's) sidecars.
    #[test]
    fn a_tombstoned_entry_answers_none_on_every_embeddings_read() {
        let dir = scratch_dir("tombstone-fence");
        let calls = Arc::new(AtomicUsize::new(0));
        let embedder = Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        let entry = state.lookup("sake").unwrap();
        {
            let mut inner = entry.inner.write();
            state.tombstone_locked(&mut inner, &entry);
        }
        assert!(state.embeddings_status("sake").is_none());
        assert!(
            state
                .semantic_twins("sake", 0.5, Deadline::unbounded())
                .is_none()
        );
        assert!(
            state
                .semantic_resolve("sake", "りんご", false, None, Deadline::unbounded())
                .is_none()
        );
        assert!(
            state
                .explain_semantic_resolve(
                    "sake",
                    "りんご",
                    "果物",
                    false,
                    None,
                    Deadline::unbounded()
                )
                .is_none()
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

    /// Issue #563 item 1: a provider answering `Ok(vec![])` for a cue
    /// must never poison the process-lifetime cue cache. Before the
    /// fix, `cue_vector` cached the empty vector unconditionally —
    /// every later resolve for this exact cue would then hit the
    /// cache and silently score 0.0 forever, with no way to
    /// invalidate it short of a restart. This pins both halves: the
    /// call is reported as a failure, not a hit with nothing in it,
    /// and — the part a naive `is_err()` check alone would miss — the
    /// provider is called again on the very next resolve for the same
    /// cue, proving nothing landed in the cache.
    #[test]
    fn cue_vector_rejects_an_empty_embedding_and_never_caches_it() {
        /// Same model name as the mock so the gloss sidecar stays
        /// usable, but every round trip answers an empty vector per
        /// text — the malformed-provider shape this guards against.
        struct EmptyEmbeddings {
            calls: Arc<AtomicUsize>,
        }
        impl EmbeddingProvider for EmptyEmbeddings {
            fn model(&self) -> &str {
                "mock"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                Ok(texts.iter().map(|_| Vec::new()).collect())
            }
        }

        let dir = scratch_dir("empty-cue");
        let calls = Arc::new(AtomicUsize::new(0));
        let embedder = Some(Arc::new(EmptyEmbeddings {
            calls: Arc::clone(&calls),
        }) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state.create("fruit", ContextMeta::default()).unwrap();
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
        let baseline = calls.load(Ordering::Relaxed);

        assert!(
            state
                .semantic_resolve("fruit", "アップル", false, None, Deadline::unbounded())
                .unwrap()
                .is_err(),
            "an empty provider vector must resolve as a failure, not an empty-but-ok answer"
        );
        let after_first = calls.load(Ordering::Relaxed);
        assert_eq!(
            after_first,
            baseline + 1,
            "the first resolve for a new cue must call the provider"
        );

        assert!(
            state
                .semantic_resolve("fruit", "アップル", false, None, Deadline::unbounded())
                .unwrap()
                .is_err()
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            after_first + 1,
            "a cached empty vector would answer this without a round trip; the provider \
             must be called again every time"
        );

        let body = rendered(&state);
        assert!(
            body.contains(
                "taguru_embedding_requests_total{operation=\"resolve\",outcome=\"failed\"} 2"
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

    /// Issue #677 item 2: a busy, gloss-stable context used to pay one
    /// provider round trip (the width probe) on every single no-op
    /// refresh, forever — expensive under a write-driven auto-embed
    /// ticker. The real embed a pass just paid for already answers
    /// "what width does the provider speak right now" just as well as
    /// a dedicated probe would, so the very next no-op TICKER refresh
    /// must not pay for one of its own. This throttle is scoped to
    /// `auto_refresh_embeddings` only — the public `refresh_embeddings`
    /// an explicit caller uses always probes, so it reliably heals a
    /// width change in one call (see that function's own doc, and
    /// `tests/http_api/width_probe.rs`, for why that contract must
    /// hold).
    #[test]
    fn a_no_op_auto_refresh_right_after_a_real_one_makes_no_provider_call_at_all() {
        struct CountingEmbeddings(usize, Arc<AtomicUsize>);
        impl EmbeddingProvider for CountingEmbeddings {
            fn model(&self) -> &str {
                "stable-name"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                self.1.fetch_add(1, Ordering::Relaxed);
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

        let dir = scratch_dir("no-double-charge");
        let calls = Arc::new(AtomicUsize::new(0));
        let embedder =
            Some(Arc::new(CountingEmbeddings(2, Arc::clone(&calls))) as Arc<dyn EmbeddingProvider>);
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

        let (embedded, _) = state
            .auto_refresh_embeddings("w", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert!(
            embedded > 0,
            "the first pass must genuinely embed something"
        );
        let after_real = calls.load(Ordering::Relaxed);

        let (embedded_again, _) = state
            .auto_refresh_embeddings("w", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(embedded_again, 0);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            after_real,
            "the no-op ticker refresh's own width probe must be skipped \
             entirely, not just cheap: the pass just above already \
             confirmed this width"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// The throttle only ever defers detection, never cancels it: once
    /// a confirmed observation ages past `WIDTH_OBSERVATION_TRUST`, the
    /// ticker's next no-op refresh probes again and still catches a
    /// genuine backend swap — the same swap an explicit refresh would
    /// have caught immediately, just later.
    #[test]
    fn a_ticker_probe_re_arms_once_its_observation_ages_past_the_trust_window() {
        struct SwappableWidthEmbeddings(Arc<AtomicUsize>, Arc<AtomicUsize>);
        impl EmbeddingProvider for SwappableWidthEmbeddings {
            fn model(&self) -> &str {
                "stable-name"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                self.1.fetch_add(1, Ordering::Relaxed);
                let width = self.0.load(Ordering::Relaxed);
                Ok(texts
                    .iter()
                    .map(|_| {
                        let mut vector = vec![0.0; width];
                        vector[0] = 1.0;
                        vector
                    })
                    .collect())
            }
        }

        let dir = scratch_dir("ticker-probe-re-arms");
        let width = Arc::new(AtomicUsize::new(2));
        let calls = Arc::new(AtomicUsize::new(0));
        let embedder = Some(Arc::new(SwappableWidthEmbeddings(
            Arc::clone(&width),
            Arc::clone(&calls),
        )) as Arc<dyn EmbeddingProvider>);
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
        let (embedded, _) = state
            .auto_refresh_embeddings("w", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert!(embedded > 0);

        // The provider now speaks width 3 behind the same model name.
        // Within the trust window, the ticker's probe stays skipped —
        // this is the cost saving item 2 exists for.
        width.store(3, Ordering::Relaxed);
        let calls_before_stale_no_op = calls.load(Ordering::Relaxed);
        let (embedded, total) = state
            .auto_refresh_embeddings("w", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!((embedded, total), (0, 3), "still within the trust window");
        assert_eq!(
            calls.load(Ordering::Relaxed),
            calls_before_stale_no_op,
            "no probe yet — the observation is still trusted"
        );

        // Once the observation ages out, the very next ticker refresh
        // probes again and this time catches the swap.
        state.age_width_observation(crate::registry::WIDTH_OBSERVATION_TRUST);
        let (embedded, total) = state
            .auto_refresh_embeddings("w", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (embedded, total),
            (3, 3),
            "the aged-out observation must not suppress detection forever"
        );
        let store = VectorStore::load(&vectors_path(&dir, &file_stem("w")));
        assert!(
            store
                .concepts
                .values()
                .chain(store.labels.values())
                .all(|(_, vector)| vector.len() == 3),
            "the swap must actually heal, not just get noticed"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// The width observation is shared by the whole process, not scoped
    /// to one context — because the width is the PROVIDER's property.
    /// A context that has never itself embedded anything this boot
    /// still skips its own ticker probe once some OTHER context's
    /// ticker probe has already confirmed the width.
    #[test]
    fn a_probe_confirmed_by_one_context_lets_a_sibling_skip_its_own() {
        struct CountingWidthEmbeddings(usize, Arc<AtomicUsize>);
        impl EmbeddingProvider for CountingWidthEmbeddings {
            fn model(&self) -> &str {
                "stable-name"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                self.1.fetch_add(1, Ordering::Relaxed);
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

        let dir = scratch_dir("cross-context-width-confirm");
        {
            let embedder = Some(
                Arc::new(CountingWidthEmbeddings(2, Arc::new(AtomicUsize::new(0))))
                    as Arc<dyn EmbeddingProvider>,
            );
            let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
            for name in ["a", "b"] {
                state
                    .create(name, ContextMeta::default())
                    .map_err(|_| "create")
                    .unwrap();
                state
                    .write_context(name, |context| {
                        context.associate("x", "l", "y", 1.0).unwrap();
                    })
                    .map_err(|_| "write")
                    .unwrap();
                state
                    .refresh_embeddings(name, Deadline::unbounded())
                    .unwrap()
                    .unwrap();
            }
            state.flush_dirty();
        }

        // Fresh boot: no observation yet. Both contexts already carry
        // width 2 on disk and neither has new content this pass.
        let calls = Arc::new(AtomicUsize::new(0));
        let embedder =
            Some(Arc::new(CountingWidthEmbeddings(2, Arc::clone(&calls)))
                as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();

        // "a"'s own no-op ticker refresh has no observation to lean on
        // yet, so it pays for its own probe.
        let (embedded_a, _) = state
            .auto_refresh_embeddings("a", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(embedded_a, 0);
        assert_eq!(calls.load(Ordering::Relaxed), 1, "a's own probe");

        // "b" was never touched this boot — but the width is the
        // provider's, not "a"'s, so "a"'s probe already answers for
        // "b" too: no second provider call.
        let (embedded_b, _) = state
            .auto_refresh_embeddings("b", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(embedded_b, 0);
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "b's probe is skipped: a's already confirmed the width this boot"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// A width change that rides alongside genuinely new content (rather
    /// than being caught only by the no-op probe) is noticed from the
    /// freshly embedded rows directly — but the redo it triggers must
    /// reuse what that same pass already bought, not re-purchase it: "c"
    /// and "d" are new concepts, embedded once at the new width before
    /// the mismatch is even noticed; "a", "b", and label "l" are
    /// unchanged hashes still carried at the old width and are the only
    /// ones the redo needs to buy.
    #[test]
    fn a_width_change_alongside_new_content_reuses_the_rows_that_pass_already_bought() {
        struct WidthEmbeddings {
            width: usize,
            texts_requested: Arc<AtomicUsize>,
        }
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
                self.texts_requested
                    .fetch_add(texts.len(), Ordering::Relaxed);
                Ok(texts
                    .iter()
                    .map(|_| {
                        let mut vector = vec![0.0; self.width];
                        vector[0] = 1.0;
                        vector
                    })
                    .collect())
            }
        }

        let dir = scratch_dir("width-change-reuse");
        {
            let embedder = Some(Arc::new(WidthEmbeddings {
                width: 2,
                texts_requested: Arc::new(AtomicUsize::new(0)),
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
            let (embedded, total) = state
                .refresh_embeddings("w", Deadline::unbounded())
                .unwrap()
                .unwrap();
            assert_eq!((embedded, total), (3, 3)); // a, b, and the label l
            state.flush_dirty();
        }

        let texts_requested = Arc::new(AtomicUsize::new(0));
        let embedder = Some(Arc::new(WidthEmbeddings {
            width: 3,
            texts_requested: Arc::clone(&texts_requested),
        }) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .write_context("w", |context| {
                context.associate("c", "l", "d", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();
        let (embedded, total) = state
            .refresh_embeddings("w", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (embedded, total),
            (5, 5),
            "a, b, c, d, and the label l all land at the new width"
        );
        assert_eq!(
            texts_requested.load(Ordering::Relaxed),
            5,
            "c/d bought once, before the mismatch was even noticed, must not \
             be bought again by the redo it triggers"
        );
        let store = VectorStore::load(&vectors_path(&dir, &file_stem("w")));
        assert!(
            store
                .concepts
                .values()
                .chain(store.labels.values())
                .all(|(_, vector)| vector.len() == 3),
            "old-width rows must not survive the width change"
        );

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

    /// Issue #563 item 5: `BootOptions { embed_parallel: 0 }` built
    /// directly (bypassing the env-boundary floor in
    /// `resolve_embed_parallel`) must not construct a permanently
    /// starved `embed_provider_slots` — `boot_with` normalizes it to 1
    /// itself now (see the field's own doc). Before that normalization
    /// was shared between the field and the semaphore's construction,
    /// this exact `BootOptions` value would have zero-sized the
    /// semaphore and hung every refresh forever.
    #[test]
    fn a_zero_embed_parallel_boot_option_does_not_starve_the_refresh_semaphore() {
        let dir = scratch_dir("embed-parallel-zero");
        let calls = Arc::new(AtomicUsize::new(0));
        let embedder = Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            embedder,
            BootOptions {
                embed_parallel: 0,
                ..BootOptions::default()
            },
        )
        .unwrap();
        assert_eq!(
            state.embed_parallel(),
            1,
            "boot_with must floor a zero embed_parallel the same way the env \
             boundary does, not carry it through unchanged"
        );
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
        // A hang here (the pre-fix behavior) would time out the test
        // binary itself rather than fail an assertion — the point of
        // this test is that it completes at all.
        let embedded = state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert!(embedded.0 > 0, "the one stale concept must actually embed");

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

    /// `semantic_twins` works off the stored sidecar alone, so a store
    /// whose LABEL table is empty must still sweep the populated
    /// concept table — only both-empty means "nothing embedded yet".
    /// And the pairwise sweep starts at the NEXT entry: a name must
    /// never come back paired with itself (a self-pair scores a
    /// perfect 1.0 and would drown every real twin).
    #[test]
    fn semantic_twins_sweeps_a_concepts_only_store_without_self_pairs() {
        let dir = scratch_dir("twins-concepts-only");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut store = VectorStore {
            model: "mock".to_string(),
            ..Default::default()
        };
        store
            .concepts
            .insert("りんご".to_string(), (1, vec![1.0, 0.0]));
        store
            .concepts
            .insert("アップル".to_string(), (2, vec![0.96, 0.28]));
        store
            .concepts
            .insert("直交".to_string(), (3, vec![0.0, 1.0]));
        store
            .save(&vectors_path(&dir, &file_stem("fruit")))
            .unwrap();

        let (concepts, labels, note) = state
            .semantic_twins("fruit", 0.9, Deadline::unbounded())
            .unwrap();
        assert!(note.is_none(), "{note:?}");
        assert!(labels.is_empty());
        assert_eq!(
            concepts.len(),
            1,
            "only りんご×アップル clears 0.9: {concepts:?}"
        );
        let (a, b, score) = &concepts[0];
        assert!(a != b, "a name paired with itself: {concepts:?}");
        assert_eq!((a.as_str(), b.as_str()), ("りんご", "アップル"));
        assert!(*score > 0.9 && *score < 1.0, "{score}");

        let _ = fs::remove_dir_all(dir);
    }

    /// The O(N²) sweep cap turns away a namespace just PAST the cap,
    /// not at it — 2000 names sweep, 2001 skip with the note.
    #[test]
    fn semantic_twins_sweep_cap_skips_past_the_cap_not_at_it() {
        let dir = scratch_dir("twins-sweep-cap");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // Only the skip NOTE is asserted on — near-parallel f32
        // vectors round their cosines to (even past) 1.0, so which
        // pairs clear a floor is fp noise this test must not ride on.
        let mut store = VectorStore {
            model: "mock".to_string(),
            ..Default::default()
        };
        for i in 0..2000u32 {
            let angle = i as f32 * 1.0e-4;
            store.concepts.insert(
                format!("c{i:04}"),
                (u64::from(i), vec![angle.cos(), angle.sin()]),
            );
        }
        let path = vectors_path(&dir, &file_stem("fruit"));
        store.save(&path).unwrap();

        let (_, _, note) = state
            .semantic_twins("fruit", 1.0, Deadline::unbounded())
            .unwrap();
        assert!(note.is_none(), "2000 names must still sweep: {note:?}");

        // One more name crosses the cap. The cached store is bypassed
        // by rewriting the sidecar and re-booting: entry_vectors holds
        // the first read for the state's lifetime (and the re-boot
        // needs the first state's data-dir lock released).
        store
            .concepts
            .insert("c2000".to_string(), (2000, vec![0.0, 1.0]));
        store.save(&path).unwrap();
        drop(state);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let (pairs, _, note) = state
            .semantic_twins("fruit", 1.0, Deadline::unbounded())
            .unwrap();
        assert!(
            note.as_deref().is_some_and(|note| note.contains("2000")),
            "2001 names must skip with the cap note: {note:?}"
        );
        assert!(
            pairs.is_empty(),
            "a skipped namespace returns no pairs: {pairs:?}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// The explain lane must run whenever the QUERIED table has rows —
    /// an empty sibling table is not "nothing embedded". And its rank
    /// reproduces `semantic_resolve`'s exact ordering, cosine ties
    /// broken by name ascending.
    #[test]
    fn explain_runs_on_a_concepts_only_store_and_breaks_cosine_ties_by_name() {
        /// Every text — cue and gloss alike — lands on the same unit
        /// vector, forcing a perfect cosine tie between candidates.
        struct ConstantEmbeddings;
        impl EmbeddingProvider for ConstantEmbeddings {
            fn model(&self) -> &str {
                "mock"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
            }
        }

        let dir = scratch_dir("explain-tie");
        let embedder = Some(Arc::new(ConstantEmbeddings) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut store = VectorStore {
            model: "mock".to_string(),
            ..Default::default()
        };
        store
            .concepts
            .insert("aaa".to_string(), (1, vec![1.0, 0.0]));
        store
            .concepts
            .insert("bbb".to_string(), (2, vec![1.0, 0.0]));
        store
            .save(&vectors_path(&dir, &file_stem("fruit")))
            .unwrap();

        let explain = |expected: &str| {
            state
                .explain_semantic_resolve(
                    "fruit",
                    "cue",
                    expected,
                    false,
                    Some(0.5),
                    Deadline::unbounded(),
                )
                .unwrap()
        };
        let GlossLaneReport::Ran { rank, passing, .. } = explain("aaa") else {
            panic!("an empty label table must not read as EmptyTable");
        };
        assert_eq!(passing, 2);
        assert_eq!(rank, Some(1), "aaa wins its tie with bbb by name");
        let GlossLaneReport::Ran { rank, .. } = explain("bbb") else {
            panic!("an empty label table must not read as EmptyTable");
        };
        assert_eq!(rank, Some(2), "bbb sits behind aaa on the same cosine");

        let _ = fs::remove_dir_all(dir);
    }

    /// Every text lands on the same vector at a fixed width — the
    /// knob the width-reconciliation tests below turn.
    struct FixedWidth(usize);
    impl EmbeddingProvider for FixedWidth {
        fn model(&self) -> &str {
            "mock"
        }
        fn embed(
            &self,
            texts: &[&str],
            _purpose: EmbedPurpose,
            _deadline: Deadline,
        ) -> Result<Vec<Vec<f32>>, String> {
            Ok(texts.iter().map(|_| vec![1.0; self.0]).collect())
        }
    }

    /// Fails call number `fail_nth` (0-based), answers every other
    /// call at `width` — the knob that manufactures single-table
    /// stores and mid-redo failures.
    struct FailsNth {
        calls: AtomicUsize,
        fail_nth: usize,
        width: usize,
    }
    impl EmbeddingProvider for FailsNth {
        fn model(&self) -> &str {
            "mock"
        }
        fn embed(
            &self,
            texts: &[&str],
            _purpose: EmbedPurpose,
            _deadline: Deadline,
        ) -> Result<Vec<Vec<f32>>, String> {
            if self.calls.fetch_add(1, Ordering::Relaxed) == self.fail_nth {
                return Err("provider down".to_string());
            }
            Ok(texts.iter().map(|_| vec![1.0; self.width]).collect())
        }
    }

    /// A width drift visible only through the LABEL table must still
    /// stale the whole store — the mismatch check is either-table, not
    /// both. (A mixed-width sidecar cannot exist — the loader refuses
    /// one outright (#133) — so a one-sided drift arises from a
    /// SINGLE-table store: here, a first pass whose concept call
    /// failed, leaving labels alone carried at width 2.) And the
    /// redo's wipe must be a real wipe even though the model NAME
    /// never changed: a table whose redo failed stays absent (stale,
    /// for the next refresh), never carried at the dead width — which
    /// would persist exactly the mixed file the loader refuses.
    #[test]
    fn a_label_only_width_drift_rebuilds_and_a_failed_redo_carries_nothing_stale() {
        let dir = scratch_dir("width-label-drift");
        {
            // Pass 1: the concept call (first) fails, labels land at
            // width 2 → a labels-only store under model "mock".
            let embedder = Some(Arc::new(FailsNth {
                calls: AtomicUsize::new(0),
                fail_nth: 0,
                width: 2,
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
            assert!(
                state
                    .refresh_embeddings("fruit", Deadline::unbounded())
                    .unwrap()
                    .is_err(),
                "the concept call's failure must be reported"
            );
            state.flush_dirty();
        }
        let path = vectors_path(&dir, &file_stem("fruit"));
        let carried = VectorStore::load(&path);
        assert!(carried.concepts.is_empty(), "{:?}", carried.concepts.keys());
        assert_eq!(carried.labels.len(), 1, "{:?}", carried.labels.keys());

        // Pass 2, same model at width 4: the (all-stale) concept call
        // lands first (call 0) and settles the fresh width; 分類 is
        // carried at width 2, so the label-side clause alone declares
        // the drift. The redo does not re-buy the concepts it already
        // bought this same pass — only labels (call 1), which fails.
        let embedder = Some(Arc::new(FailsNth {
            calls: AtomicUsize::new(0),
            fail_nth: 1,
            width: 4,
        }) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        assert!(
            state
                .refresh_embeddings("fruit", Deadline::unbounded())
                .unwrap()
                .is_err(),
            "the redo's label failure must be reported"
        );

        let store = VectorStore::load(&path);
        assert_eq!(
            store.concepts.len(),
            2,
            "the redo's concepts must land at the fresh width (a mixed file \
             would load back as empty): {:?}",
            store.concepts.keys()
        );
        assert!(
            store.concepts.values().all(|(_, vector)| vector.len() == 4),
            "a concept landed at the dead width"
        );
        assert!(
            store.labels.is_empty(),
            "the failed label redo must stay stale, not linger at width 2: {:?}",
            store.labels.keys()
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// A pass that buys nothing, prunes nothing, and owes no retry
    /// must not rewrite the sidecar — the no-op refresh is the hot
    /// path the auto-ticker hits forever.
    #[test]
    #[cfg(unix)]
    fn a_no_op_gloss_refresh_leaves_the_sidecar_file_untouched() {
        use std::os::unix::fs::MetadataExt;

        let dir = scratch_dir("no-op-no-save");
        let embedder = Some(Arc::new(FixedWidth(2)) as Arc<dyn EmbeddingProvider>);
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
        let path = vectors_path(&dir, &file_stem("fruit"));
        let before = fs::metadata(&path).unwrap().ino();

        assert_eq!(
            state
                .refresh_embeddings("fruit", Deadline::unbounded())
                .unwrap()
                .unwrap()
                .0,
            0
        );
        assert_eq!(
            fs::metadata(&path).unwrap().ino(),
            before,
            "a no-op refresh rewrote the sidecar (write_atomic mints a new inode)"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// A refresh whose only work is pruning retracted names must still
    /// hit the disk AND bump the config revision: the ghost rows are
    /// gone from what the semantic lane serves, and a reboot must not
    /// resurrect them from a stale sidecar.
    #[test]
    fn a_prune_only_refresh_rewrites_the_sidecar_and_bumps_config() {
        let dir = scratch_dir("prune-only");
        let embedder = Some(Arc::new(FixedWidth(2)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // Two DISCONNECTED edges: retracting s2 (and compacting the
        // dead edge away — a retracted edge lingers with its gloss
        // merely changed until compaction removes the names) leaves
        // every surviving gloss byte-identical, so the second refresh
        // embeds nothing and prunes c/d/l2 — the prune-only pass.
        state
            .add_associations(
                "fruit",
                vec![
                    assoc_op("a", "l1", "b", 1.0, Some("s1")),
                    assoc_op("c", "l2", "d", 1.0, Some("s2")),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        state.retract_source("fruit", "s2").unwrap();
        state
            .compact_context("fruit", Deadline::unbounded())
            .unwrap();
        let config = state.context_revision("fruit").unwrap().config;

        let (newly, total) = state
            .refresh_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (newly, total),
            (0, 3),
            "a, b and l1 survive; nothing re-embeds"
        );
        let store = VectorStore::load(&vectors_path(&dir, &file_stem("fruit")));
        assert!(
            !store.concepts.contains_key("c") && !store.concepts.contains_key("d"),
            "retracted concepts survived on disk: {:?}",
            store.concepts.keys()
        );
        assert!(
            !store.labels.contains_key("l2"),
            "{:?}",
            store.labels.keys()
        );
        assert_eq!(store.concepts.len(), 2);
        assert_eq!(
            state.context_revision("fruit").unwrap().config,
            config + 1,
            "served vectors changed; the revision must move"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// The accessor hands back the CONFIGURED slot count — main's
    /// auto-refresh pool is sized by this, so a constant here would
    /// silently serialize (or over-parallelize) every deployment.
    #[test]
    fn embed_parallel_reports_the_configured_slot_count() {
        let dir = scratch_dir("embed-parallel");
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            None,
            BootOptions {
                embed_parallel: 3,
                ..BootOptions::default()
            },
        )
        .unwrap();
        assert_eq!(state.embed_parallel(), 3);
        let _ = fs::remove_dir_all(dir);
    }

    /// #678: `PAIR_CAP` truncates a namespace's returned pairs at
    /// exactly 100, the same asymmetry `semantic_twins_sweep_cap_skips_past_the_cap_not_at_it`
    /// already covers for the sibling `SWEEP_CAP`. A hub vector paired
    /// with 101 spokes at strictly increasing, distinct cosines makes
    /// the truncated-away pair unambiguous: spoke-spoke cosines are
    /// engineered (via each spoke's own orthogonal axis) to stay under
    /// the floor, so only the 101 hub-spoke pairs ever clear it, and
    /// `truncate(100)` after the descending sort drops exactly the
    /// single lowest-scoring one.
    #[test]
    fn semantic_twins_pair_cap_truncates_at_exactly_one_hundred_pairs() {
        let dir = scratch_dir("twins-pair-cap");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        const SPOKES: usize = 101;
        let dims = SPOKES + 1;
        let mut store = VectorStore {
            model: "mock".to_string(),
            ..Default::default()
        };
        let mut hub = vec![0.0f32; dims];
        hub[0] = 1.0;
        store.concepts.insert("hub".to_string(), (0, hub));
        let mut scores = Vec::with_capacity(SPOKES);
        for i in 0..SPOKES {
            // Strictly increasing, all inside [0.5, 0.55): any two
            // spokes' pairwise cosine is their product (only the hub
            // axis overlaps between them), which tops out around
            // 0.55*0.55 ≈ 0.30 — comfortably under the 0.5 floor —
            // while each hub-spoke pair clears it alone.
            let cosine = 0.50003 + i as f32 * 0.00049;
            let theta = cosine.acos();
            let mut spoke = vec![0.0f32; dims];
            spoke[0] = cosine;
            spoke[1 + i] = theta.sin();
            store
                .concepts
                .insert(format!("spoke{i:03}"), (i as u64 + 1, spoke));
            scores.push(cosine);
        }
        store
            .save(&vectors_path(&dir, &file_stem("fruit")))
            .unwrap();

        let (concepts, labels, note) = state
            .semantic_twins("fruit", 0.5, Deadline::unbounded())
            .unwrap();
        assert!(note.is_none(), "{note:?}");
        assert!(labels.is_empty());
        assert_eq!(
            concepts.len(),
            100,
            "PAIR_CAP must truncate 101 floor-clearing pairs down to 100: {concepts:?}"
        );
        assert!(
            concepts.iter().all(|(a, b, _)| a == "hub" || b == "hub"),
            "no spoke-spoke pair should ever clear the floor: {concepts:?}"
        );
        let min_score = scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            !concepts
                .iter()
                .any(|(_, _, score)| (*score - min_score).abs() < 1e-6),
            "the single lowest-scoring pair is exactly what truncate(100) must drop: {concepts:?}"
        );
        assert!(
            concepts
                .iter()
                .any(|(_, _, score)| (*score - max_score).abs() < 1e-6),
            "the highest-scoring pair must survive the truncation"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// #678: the sentinel guard between `ModelChanged` and `Ran` — a
    /// same-named model that quietly started answering wider vectors
    /// must not let `similarity`'s width-mismatch 0.0 read as a
    /// measured cosine (the doc comment on `WidthChanged`'s own
    /// variant). `tests/http_api/width_probe.rs` covers this
    /// end-to-end via `semantic.reason`; this pins the report type
    /// itself at the registry layer.
    #[test]
    fn explain_semantic_resolve_reports_a_width_change_under_the_same_model_name() {
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

        let dir = scratch_dir("explain-width-change");
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
            state
                .refresh_embeddings("w", Deadline::unbounded())
                .unwrap()
                .unwrap();
            state.flush_dirty();
        }

        // Same model name, wider query vectors: the sidecar is still
        // width 2, the cue now comes back at width 3.
        let embedder = Some(Arc::new(WidthEmbeddings(3)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        let Some(GlossLaneReport::WidthChanged { stored, current }) =
            state.explain_semantic_resolve("w", "cue", "a", false, None, Deadline::unbounded())
        else {
            panic!("a same-named model's wider query vectors must report WidthChanged");
        };
        assert_eq!((stored, current), (2, 3));

        let _ = fs::remove_dir_all(dir);
    }

    /// #678: the provider-refusal arm of `explain_semantic_resolve` —
    /// distinct from `cue_vector` itself being unit-tested
    /// (`cue_vector_rejects_an_empty_embedding_and_never_caches_it`)
    /// in that this pins the `Err` actually surfacing as
    /// `QueryEmbeddingFailed` rather than silently folding into some
    /// other arm.
    #[test]
    fn explain_semantic_resolve_reports_a_failed_cue_embedding() {
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

        let dir = scratch_dir("explain-query-embed-failed");
        let calls = Arc::new(AtomicUsize::new(0));
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
            state.flush_dirty();
        }

        let embedder = Some(Arc::new(FailingEmbeddings) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        let Some(GlossLaneReport::QueryEmbeddingFailed(message)) = state.explain_semantic_resolve(
            "fruit",
            "アップル",
            "りんご",
            false,
            None,
            Deadline::unbounded(),
        ) else {
            panic!("a failed cue embedding must be reported, not folded elsewhere");
        };
        assert!(message.contains("provider down"), "{message}");

        let _ = fs::remove_dir_all(dir);
    }
}
