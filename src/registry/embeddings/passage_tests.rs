#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use parking_lot::Mutex;
    use taguru::deadline::Deadline;

    use crate::embedding::{EmbedPurpose, EmbeddingProvider, PassageVectorStore};
    use crate::registry::test_support::{
        MockEmbeddings, SlowEmbeddings, boot_for_passage_embedding, plain, scratch_dir,
    };
    use crate::registry::{AppState, BootOptions, ContextMeta, file_stem, pvectors_path};

    #[test]
    fn refresh_passage_embeddings_embeds_every_paragraph_once_then_nothing() {
        let dir = scratch_dir("pvec-first");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "doc-a".to_string(),
            "最初の段落。\n\n二番目の段落。".to_string(),
        );
        passages.insert("doc-b".to_string(), "三番目の段落。".to_string());
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        assert_eq!(
            state.passage_embed_dirty_names(),
            vec!["sake".to_string()],
            "a store marks the context for the auto ticker"
        );

        let outcome = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (outcome.embedded, outcome.total, outcome.skipped_over_limit),
            (3, 3, 0)
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "three paragraphs fit one provider batch"
        );
        assert!(
            state.passage_embed_dirty_names().is_empty(),
            "the refresh claims the dirty flag"
        );

        // Unchanged corpus: nothing re-embeds. The one provider call is
        // the width probe that guards against a silent backend swap (a
        // changed vector width behind an unchanged model name), the same
        // one-embedding-per-no-op cost the gloss refresh pays.
        let again = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!((again.embedded, again.total), (0, 3));
        assert_eq!(calls.load(Ordering::Relaxed), 2);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn refresh_passage_embeddings_re_embeds_only_the_changed_paragraph() {
        let dir = scratch_dir("pvec-diff");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "doc-a".to_string(),
            "変わらない段落。\n\n古い版の段落。".to_string(),
        );
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();

        let mut updated = BTreeMap::new();
        updated.insert(
            "doc-a".to_string(),
            "変わらない段落。\n\n新しい版の段落。".to_string(),
        );
        state
            .store_passages("sake", plain(updated))
            .unwrap()
            .unwrap();
        let outcome = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            outcome.embedded, 1,
            "the unchanged paragraph rides its hash, only the edit re-embeds"
        );
        assert_eq!(outcome.total, 2);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn refresh_passage_embeddings_prunes_vectors_for_a_retracted_source() {
        let dir = scratch_dir("pvec-prune");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "残る段落。".to_string());
        passages.insert("doc-b".to_string(), "消える段落。".to_string());
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();

        state.retract_source("sake", "doc-b").unwrap();
        let outcome = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (outcome.embedded, outcome.total),
            (0, 1),
            "the retracted source's row is gone without any re-embedding"
        );
        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("sake")));
        assert_eq!(sidecar.len(), 1, "the prune reached the disk too");

        let _ = fs::remove_dir_all(dir);
    }

    /// A provider that changes output width behind a stable model name (a
    /// backend swap behind the same proxy) must stale the whole carried
    /// passage table, exactly as the gloss refresh does: paragraph hashes
    /// are unchanged, so without a width check old-width rows would pin
    /// the store's dimension and `PassageVectorStore::push` would drop
    /// every new-width row this pass embeds — silently, at a warn.
    #[test]
    fn a_passage_width_change_under_the_same_model_name_re_embeds_everything() {
        struct WidthEmbeddings(Arc<std::sync::atomic::AtomicUsize>);
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

        let dir = scratch_dir("pvec-width");
        let width = Arc::new(std::sync::atomic::AtomicUsize::new(2));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(WidthEmbeddings(Arc::clone(&width))), 20_000);
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "最初の段落。".to_string());
        passages.insert("doc-b".to_string(), "二番目の段落。".to_string());
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        let first = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!((first.embedded, first.total), (2, 2));
        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("sake")));
        assert!(
            sidecar.iter().all(|(_, row)| row.len() == 2),
            "the first pass stored 2-dim rows"
        );

        // Probe path: same passages (every hash carried, nothing else
        // reveals the width) but wider vectors. One probe embedding must
        // catch the change and re-embed every row, or the store keeps its
        // stale 2-dim rows against a provider now speaking 3.
        width.store(3, Ordering::Relaxed);
        let widened = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (widened.embedded, widened.total),
            (2, 2),
            "an unchanged corpus still re-embeds every row on a width change"
        );
        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("sake")));
        assert_eq!(sidecar.len(), 2);
        assert!(
            sidecar.iter().all(|(_, row)| row.len() == 3),
            "every stored row is the new width — none dropped, none left at the old one"
        );

        // Embedded-rows path: a width change that rides alongside an edit
        // is caught from the freshly embedded rows directly, no probe. The
        // one unchanged paragraph must not survive at the old width.
        width.store(4, Ordering::Relaxed);
        let mut edited = BTreeMap::new();
        edited.insert("doc-a".to_string(), "改訂された段落。".to_string());
        edited.insert("doc-b".to_string(), "二番目の段落。".to_string());
        state
            .store_passages("sake", plain(edited))
            .unwrap()
            .unwrap();
        let mixed = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (mixed.embedded, mixed.total),
            (2, 2),
            "the edit's new width stales the carried row too"
        );
        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("sake")));
        assert_eq!(sidecar.len(), 2);
        assert!(
            sidecar.iter().all(|(_, row)| row.len() == 4),
            "the carried old-width row was re-embedded, not dropped"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// `PassageVectorStore::push` already drops whichever chunk lands at
    /// a width disagreeing with the dimension the store already
    /// settled on — but the merge loop counted every attempted push as
    /// `embedded` regardless, over-reporting work that `push` silently
    /// threw away. `embedded` must count only rows that actually landed.
    #[test]
    fn passage_refresh_reports_an_accurate_embedded_count_when_a_chunk_disagrees_on_width() {
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
                // The full 128-paragraph chunk answers at width 2; the
                // trailing 44-paragraph chunk answers at width 3 — a
                // provider mid-migration serving two backend versions to
                // concurrent connections.
                let width = if texts.len() >= 128 { 2 } else { 3 };
                Ok(texts.iter().map(|_| vec![0.0; width]).collect())
            }
        }

        let dir = scratch_dir("pvec-split-width");
        let embedder = Arc::new(SplitWidthEmbeddings) as Arc<dyn EmbeddingProvider>;
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            Some(embedder),
            BootOptions {
                embed_passages: true,
                passage_vector_limit: 20_000,
                embed_parallel: 2,
                ..BootOptions::default()
            },
        )
        .unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // 172 paragraphs = a 128-item chunk plus a 44-item remainder.
        let text = (0..172)
            .map(|i| format!("段落その{i}。"))
            .collect::<Vec<_>>()
            .join("\n\n");
        let mut passages = BTreeMap::new();
        passages.insert("doc-big".to_string(), text);
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        let outcome = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (outcome.embedded, outcome.total),
            (128, 128),
            "the disagreeing trailing chunk is dropped, not merely undercounted"
        );
        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("sake")));
        assert!(sidecar.iter().all(|(_, row)| row.len() == 2));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn refresh_passage_embeddings_skips_paragraphs_beyond_the_configured_limit() {
        let dir = scratch_dir("pvec-limit");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state = boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 2);
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "doc-a".to_string(),
            "一つ目。\n\n二つ目。\n\n三つ目。".to_string(),
        );
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        let outcome = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (outcome.embedded, outcome.total, outcome.skipped_over_limit),
            (2, 2, 1),
            "past the limit the lexical lane still serves; only the vector lane goes partial"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn refresh_passage_embeddings_persists_partial_progress_after_a_provider_failure() {
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

        let dir = scratch_dir("pvec-partial");
        let state = boot_for_passage_embedding(
            &dir,
            Arc::new(FlakyEmbeddings {
                calls: std::sync::atomic::AtomicUsize::new(0),
                fail_on: 1,
            }),
            20_000,
        );
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // 129 paragraphs = one full batch of 128 plus one more, so the
        // second provider call is the one that fails.
        let text = (0..129)
            .map(|i| format!("段落その{i}。"))
            .collect::<Vec<_>>()
            .join("\n\n");
        let mut passages = BTreeMap::new();
        passages.insert("doc-big".to_string(), text);
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        let error = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap_err();
        assert!(error.contains("hiccup"), "{error}");
        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("sake")));
        assert_eq!(
            sidecar.len(),
            128,
            "the batch that landed is durable despite the failure"
        );
        assert_eq!(
            state.passage_embed_dirty_names(),
            vec!["sake".to_string()],
            "unfinished work stays claimed for the ticker"
        );

        // The next refresh buys only the missing paragraph.
        let outcome = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!((outcome.embedded, outcome.total), (1, 129));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    #[cfg(unix)]
    fn refresh_passage_embeddings_does_not_rebuy_rows_a_failed_save_already_bought() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("pvec-save-fail");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("doc-a".to_string(), "りんごの段落。".to_string());
        state
            .store_passages("fruit", plain(passages))
            .unwrap()
            .unwrap();

        // The disk goes bad right before the save: the provider still
        // gets paid (embed happens before the write), but the sidecar
        // write fails.
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
        let error = state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap_err();
        assert!(error.contains("not persisted"), "{error}");
        let calls_after_failure = calls.load(Ordering::Relaxed);
        assert!(
            calls_after_failure > 0,
            "the provider must have been paid before the save failed"
        );
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();

        // The disk recovers: the retry must not re-embed the row the
        // failed save already bought (a width probe still spends one
        // call on a no-op refresh, same as any other — see
        // a_width_change_under_the_same_model_name_re_embeds_everything),
        // yet it must still retry the write so the row does not stay
        // unpersisted forever.
        let outcome = state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            outcome.embedded, 0,
            "must not re-embed what the failed save already cached"
        );
        assert_eq!(
            calls.load(Ordering::Relaxed),
            calls_after_failure + 1,
            "only the width probe's one call, not a re-embed of the cached row"
        );

        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("fruit")));
        assert_eq!(
            sidecar.len(),
            outcome.total,
            "the retried save must have actually landed on disk"
        );
        assert!(outcome.total > 0);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn refresh_re_embeds_only_the_question_whose_text_changed() {
        let dir = scratch_dir("doc2query-diff");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state =
            boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 20_000);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let submission = |question: &str| {
            let mut passages = BTreeMap::new();
            passages.insert(
                "doc".to_string(),
                crate::passages::PassageSubmission {
                    text: "りんごは真っ赤に実った。".to_string(),
                    questions: vec![(0, question.to_string())],
                    sections: Vec::new(),
                    meta: crate::passages::SourceMeta::default(),
                },
            );
            passages
        };
        state
            .store_passages("fruit", submission("アップルはどんな色?"))
            .unwrap()
            .unwrap();
        state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();

        state
            .store_passages("fruit", submission("アップルは何色ですか?"))
            .unwrap()
            .unwrap();
        let outcome = state
            .refresh_passage_embeddings("fruit", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (outcome.embedded, outcome.total),
            (1, 2),
            "the unchanged paragraph row is carried; only the reworded question re-embeds"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn refresh_counts_question_rows_against_the_vector_limit() {
        let dir = scratch_dir("doc2query-limit");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let state = boot_for_passage_embedding(&dir, Arc::new(MockEmbeddings::fruity(&calls)), 2);
        state
            .create("fruit", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "doc".to_string(),
            crate::passages::PassageSubmission {
                text: "りんごは真っ赤に実った。".to_string(),
                questions: vec![
                    (0, "アップルはどんな色?".to_string()),
                    (0, "みかんとの違いは?".to_string()),
                ],
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
            (outcome.embedded, outcome.total, outcome.skipped_over_limit),
            (2, 2, 1),
            "a question row spends the same budget a paragraph row does"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn refresh_passage_embeddings_requires_the_opt_in() {
        let dir = scratch_dir("pvec-optin");
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let embedder = Some(Arc::new(MockEmbeddings::fruity(&calls)) as Arc<dyn EmbeddingProvider>);
        let state = AppState::boot(dir.clone(), usize::MAX, embedder).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let error = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap_err();
        assert!(error.contains("TAGURU_EMBED_PASSAGES"), "{error}");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn passage_refresh_dispatches_chunks_concurrently_when_embed_parallel_is_raised() {
        let dir = scratch_dir("passage-parallel");
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let embedder = Arc::new(SlowEmbeddings {
            in_flight: Arc::clone(&in_flight),
            peak: Arc::clone(&peak),
        }) as Arc<dyn EmbeddingProvider>;
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            Some(embedder),
            BootOptions {
                embed_passages: true,
                passage_vector_limit: 20_000,
                embed_parallel: 2,
                ..BootOptions::default()
            },
        )
        .unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // 129 paragraphs = one full batch of 128 plus one more; with
        // embed_parallel=2 both chunks dispatch at once.
        let text = (0..129)
            .map(|i| format!("段落その{i}。"))
            .collect::<Vec<_>>()
            .join("\n\n");
        let mut passages = BTreeMap::new();
        passages.insert("doc-big".to_string(), text);
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();

        assert!(
            peak.load(Ordering::SeqCst) > 1,
            "129 paragraphs split into two chunks; with embed_parallel=2 both \
             should reach the provider at once"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn refresh_passage_embeddings_persists_a_non_prefix_subset_when_parallel_dispatch_fails_early()
    {
        /// Fails every call belonging to chunk 0 (paragraphs 0..128);
        /// later chunks succeed. Which chunk a call belongs to is
        /// recovered from its first text's paragraph index, since the
        /// provider only ever sees texts, not chunk indices.
        struct FailFirstChunk {
            calls: Arc<Mutex<Vec<usize>>>,
        }
        impl EmbeddingProvider for FailFirstChunk {
            fn model(&self) -> &str {
                "fail-first"
            }
            fn embed(
                &self,
                texts: &[&str],
                _purpose: EmbedPurpose,
                _deadline: Deadline,
            ) -> Result<Vec<Vec<f32>>, String> {
                let first_index: usize = texts[0]
                    .trim_start_matches("段落その")
                    .trim_end_matches("。")
                    .parse()
                    .expect("well-formed test fixture text");
                let chunk_index = first_index / 128;
                self.calls.lock().push(chunk_index);
                if chunk_index == 0 {
                    // Delayed so the other two workers have time to
                    // claim and start their own chunks first — an
                    // immediate failure here can otherwise record
                    // `first_failure` before chunk 2's worker even
                    // calls `fetch_add`, gating a chunk that never got
                    // a chance to be "in flight."
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    return Err("boom".to_string());
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
                Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
            }
        }

        let dir = scratch_dir("pvec-non-prefix");
        let calls: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
        let embedder = Arc::new(FailFirstChunk {
            calls: Arc::clone(&calls),
        }) as Arc<dyn EmbeddingProvider>;
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            Some(embedder),
            BootOptions {
                embed_passages: true,
                passage_vector_limit: 20_000,
                embed_parallel: 3,
                ..BootOptions::default()
            },
        )
        .unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // 300 paragraphs = chunk 0 (index 0..128, fails), chunk 1
        // (128..256, succeeds), chunk 2 (256..300, succeeds) — with
        // embed_parallel=3 all three dispatch at once, so chunks 1 and 2
        // can complete and land before chunk 0's failure is even
        // recorded.
        let text = (0..300)
            .map(|i| format!("段落その{i}。"))
            .collect::<Vec<_>>()
            .join("\n\n");
        let mut passages = BTreeMap::new();
        passages.insert("doc-big".to_string(), text);
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        let error = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap_err();
        assert!(error.contains("boom"), "{error}");

        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("sake")));
        assert_eq!(
            sidecar.len(),
            172,
            "chunks 1 and 2 (128 + 44 paragraphs) persist even though the \
             earlier chunk 0 failed — the surviving subset is not a prefix \
             of the original order"
        );
        assert!(
            sidecar.iter().all(|(key, _)| key.index >= 128),
            "no paragraph from the failed first chunk (index < 128) should be present"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// A width change behind a stable model name must still force a
    /// full re-embed even when a sibling chunk's call fails in the same
    /// pass. Left gated on that failure, the carried row's old
    /// dimension stays pinned on `fresh` and `PassageVectorStore::push`
    /// silently drops every new-width row the surviving chunk just
    /// bought — reported as a mere transient error, with no sign the
    /// work was thrown away.
    #[test]
    fn passage_width_reconciliation_fires_even_when_a_later_chunk_fails() {
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

        let dir = scratch_dir("pvec-width-reconcile");
        // First boot: one paragraph at width 2, establishing the
        // carried width the second boot must reconcile against.
        {
            let embedder = Arc::new(FlakyWidthEmbeddings {
                calls: AtomicUsize::new(0),
                fail_on: usize::MAX,
                width: 2,
            }) as Arc<dyn EmbeddingProvider>;
            let state = boot_for_passage_embedding(&dir, embedder, 20_000);
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            let mut seed = BTreeMap::new();
            seed.insert("doc-seed".to_string(), "最初の段落。".to_string());
            state.store_passages("sake", plain(seed)).unwrap().unwrap();
            state
                .refresh_passage_embeddings("sake", Deadline::unbounded())
                .unwrap()
                .unwrap();
        }

        // Second boot: same model name, width now 3. 129 new paragraphs
        // split into a 128-item chunk (call 0, succeeds) and a 1-item
        // remainder (call 1, fails) — doc-seed's unchanged paragraph is
        // carried forward, not re-embedded, so it never touches the
        // provider. The surviving chunk already proves the width
        // changed; that must be reconciled regardless of its sibling's
        // failure.
        let embedder = Arc::new(FlakyWidthEmbeddings {
            calls: AtomicUsize::new(0),
            fail_on: 1,
            width: 3,
        }) as Arc<dyn EmbeddingProvider>;
        let state = boot_for_passage_embedding(&dir, embedder, 20_000);
        let text = (0..129)
            .map(|i| format!("段落その{i}。"))
            .collect::<Vec<_>>()
            .join("\n\n");
        let mut passages = BTreeMap::new();
        passages.insert("doc-big".to_string(), text);
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        let outcome = state
            .refresh_passage_embeddings("sake", Deadline::unbounded())
            .unwrap()
            .unwrap();
        assert_eq!(
            (outcome.embedded, outcome.total),
            (130, 130),
            "the reconciliation retry re-embeds every row, carried and fresh alike"
        );
        let sidecar = PassageVectorStore::load(&pvectors_path(&dir, &file_stem("sake")));
        assert!(
            sidecar.iter().all(|(_, row)| row.len() == 3),
            "a sibling chunk's transient failure must not block reconciling a width \
             disagreement the surviving chunk already proved"
        );

        let _ = fs::remove_dir_all(dir);
    }
}
