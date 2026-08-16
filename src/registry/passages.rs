use std::collections::{BTreeMap, HashMap};
use std::io;
use std::sync::atomic::Ordering;

use super::{AppState, CitationLookup, Markers, file_stem};

/// Why a passage store was refused — the write path's two failure
/// families kept apart so a handler can answer 507 for the policy
/// refusal and 500 for the disk one, instead of flattening both into
/// one `io::Error`.
#[derive(Debug)]
pub enum PassagesWriteError {
    /// The store itself failed (load, append, fsync) — an operator
    /// problem, surfaced like every other io failure here.
    Io(io::Error),
    /// The context is at or over its declared storage ceiling
    /// (`TAGURU_CONTEXT_QUOTAS`) — the same 507 `storage_full`
    /// contract as the graph side's
    /// [`super::AccessError::QuotaExceeded`].
    QuotaExceeded(String),
}

impl AppState {
    /// Registers original text passages behind source ids, merge-upsert,
    /// persisted immediately. This is the server-side "storage of
    /// record" convenience the library deliberately does not have: the
    /// graph indexes knowledge and attributions carry opaque source ids;
    /// this store lets a client dereference those ids back to original
    /// wording — find with the graph, answer from the text. Passages are
    /// optional per source; nothing requires one to exist.
    pub fn store_passages(
        &self,
        name: &str,
        passages: BTreeMap<String, crate::passages::PassageSubmission>,
    ) -> Option<Result<crate::passages::StoreOutcome, PassagesWriteError>> {
        let entry = self.lookup(name)?;
        let fence = entry.read_unless_deleted()?;
        // The storage-quota gate, before the store is even loaded: this
        // entrance only ever grows the context (retraction goes through
        // `retract_source`, which stays open at the ceiling), so no op
        // inspection is needed — the graph gate's `WalOp::grows` split
        // has no counterpart here. The admission lock is what makes
        // the gate real under the SHARED fence: without it, two
        // concurrent stores could read the same pre-write usage, both
        // pass, and only then serialize at the store's writer mutex —
        // already past the gate (see `Entry::passages_admission`).
        let admission = entry.passages_admission.lock();
        if let Some((used, ceiling)) = self.storage_quota_excess(name, &fence, &entry) {
            self.0.metrics.record_storage_quota_refusal();
            return Some(Err(PassagesWriteError::QuotaExceeded(
                super::storage_quota_message(name, used, ceiling),
            )));
        }
        let outcome = match self.entry_passages(&entry, &file_stem(name)) {
            Ok(store) => {
                let sources: Vec<String> = passages.keys().cloned().collect();
                let stored = store.store(passages);
                // The append is settled (durable or refused) and its
                // bytes are on the store's books — the next admission
                // reads them; the index folding below needs no gate.
                drop(admission);
                if stored.is_ok() {
                    // Every store lock is released again; fold the new
                    // paragraphs into the resident index.
                    self.refresh_bm25(&entry, &store, &sources);
                    entry.passages_embed_dirty.store(true, Ordering::Relaxed);
                    // Bump AFTER the batch applied (a reader observing
                    // the new value sees the new passages); fetch_max
                    // because concurrent batches finish out of order.
                    entry
                        .passage_revision
                        .fetch_max(store.watermark(), Ordering::Relaxed);
                    // The change feed's passage-side entrance (#422):
                    // one event per source, the unit a syncing client
                    // re-pulls. Under the shared fence like the store
                    // itself; the ring's own mutex orders concurrent
                    // batches.
                    entry.changes.lock().extend(sources.iter().map(|source| {
                        crate::registry::ChangeKind::SourceStored {
                            source: source.clone(),
                        }
                    }));
                }
                stored.map_err(PassagesWriteError::Io)
            }
            // The load failed before any write; the admission falls
            // with the enclosing scope.
            Err(error) => Err(PassagesWriteError::Io(error)),
        };
        drop(fence);
        // Passage text is resident now; give the budget a chance to
        // evict something (possibly this context's own cold graph).
        self.enforce_budget(name);
        Some(outcome)
    }

    /// The window→eligible-source join of a windowed graph read
    /// (ADR 0011 §4 steps 1–2): every source name whose metadata the
    /// filter admits, read from the passage store — the one place
    /// `SourceMeta` lives — for a handler to resolve into a
    /// [`taguru::context::SourceWindow`] inside its `read_context`
    /// closure. Runs BEFORE `read_context`, like `hidden_label`: the
    /// store takes its own locks, and this keeps the join out of the
    /// entry's read path. `None` when the context does not exist.
    /// Sources with no stored passage (associations-only imports) have
    /// no metadata and are absent by construction — ADR 0011 §4's
    /// documented rule for undatable sources, not an oversight.
    pub fn window_source_names(
        &self,
        name: &str,
        filter: &crate::passages::SourceFilter,
    ) -> Option<io::Result<std::collections::BTreeSet<String>>> {
        let entry = self.lookup(name)?;
        let _fence = entry.read_unless_deleted()?;
        Some(
            self.entry_passages(&entry, &file_stem(name))
                .map(|store| store.eligible_sources(filter).0),
        )
    }

    /// Every source's effective time (`date.or(stored_at)`, the
    /// `SourceFilter` rule) as one map — the consolidation audit's
    /// join input (ADR 0012 §4 `staleness`, the ADR 0011 §4 layering:
    /// dates never enter the library, so the caller joins by name).
    /// Sources with neither field — and sources with no stored passage
    /// at all — are simply absent. `None` when the context does not
    /// exist. Runs before `read_context`, like `window_source_names`.
    pub fn source_effective_times(
        &self,
        name: &str,
    ) -> Option<io::Result<std::collections::HashMap<String, u64>>> {
        let entry = self.lookup(name)?;
        let _fence = entry.read_unless_deleted()?;
        Some(self.entry_passages(&entry, &file_stem(name)).map(|store| {
            store
                .source_entries()
                .into_iter()
                .filter_map(|(source, meta)| {
                    meta.date.or(meta.stored_at).map(|time| (source, time))
                })
                .collect()
        }))
    }

    /// Dereferences source ids (as found on attributions) back to their
    /// registered passages, reporting the ids that have none.
    #[allow(clippy::type_complexity)]
    pub fn lookup_passages(
        &self,
        name: &str,
        sources: &[String],
    ) -> Option<io::Result<(BTreeMap<String, String>, Vec<String>)>> {
        let entry = self.lookup(name)?;
        let _fence = entry.read_unless_deleted()?;
        let store = match self.entry_passages(&entry, &file_stem(name)) {
            Ok(store) => store,
            Err(error) => return Some(Err(error)),
        };
        let mut passages = BTreeMap::new();
        let mut missing = Vec::new();
        for source in sources {
            match store.get(source) {
                Some(record) => {
                    passages.insert(source.clone(), record.text.to_string());
                }
                None => missing.push(source.clone()),
            }
        }
        Some(Ok((passages, missing)))
    }

    /// Resolves one `(source, paragraph index)` pair to its verbatim
    /// excerpt — the located counterpart of `lookup_passages`'
    /// whole-document dereference. Reuses `PassageRecord::paragraph`,
    /// the same slice `search_passages` goes through for its hits, so
    /// the two can never disagree about what a paragraph's text is.
    /// The section label and locator (ADR 0007 §7) come from the same
    /// resident record via `section_for`/`locator_for`, each `None`
    /// when the index falls outside what the source's import stored.
    pub fn citation(
        &self,
        name: &str,
        source: &str,
        index: u32,
    ) -> Option<io::Result<CitationLookup>> {
        let entry = self.lookup(name)?;
        let _fence = entry.read_unless_deleted()?;
        let store = match self.entry_passages(&entry, &file_stem(name)) {
            Ok(store) => store,
            Err(error) => return Some(Err(error)),
        };
        let Some(record) = store.get(source) else {
            return Some(Ok(CitationLookup::UnknownSource));
        };
        let Some((_, text)) = record.paragraph(index as usize) else {
            return Some(Ok(CitationLookup::IndexOutOfRange));
        };
        let section = record.section_for(index as usize).map(str::to_string);
        let locator = record.locator_for(index as usize).cloned();
        Some(Ok(CitationLookup::Found {
            text: text.to_string(),
            section,
            locator,
        }))
    }

    /// Resolves `(source, paragraph)` locator keys — as found on
    /// attributions — to the section label and typed citation locator
    /// (ADR 0007 §7) governing each, batching every pair an
    /// association-bearing response needs into one passage-store load
    /// rather than one per attribution. Best-effort: an unknown
    /// context, a deleted entry, or a passage-store load failure all
    /// resolve to an empty map rather than an error. Association reads
    /// (recall, query, explore, activate, unreachable_from) are graph
    /// reads first; these markers are enrichment on top, not a hard
    /// dependency the way `citation`'s text lookup is. A pair with no
    /// covering marker simply carries `None` in [`Markers`] — the same
    /// null-means-nothing contract `Attribution::paragraph` already
    /// makes, never a fabricated value. An empty `keys` iterator skips
    /// the passage-store load entirely, so a graph-only response (no
    /// attribution carries a paragraph) never touches passages.
    ///
    /// Not to be confused with `(source, paragraph)` "locator" keys
    /// themselves (`keys`' own element type, also `api::locator_keys`'
    /// output) — this resolves each key to the typed citation
    /// [`crate::passages::Locator`] payload, a different sense of the
    /// word.
    pub fn resolve_markers(
        &self,
        name: &str,
        keys: impl Iterator<Item = (String, u32)>,
    ) -> HashMap<(String, u32), Markers> {
        let mut keys = keys.peekable();
        if keys.peek().is_none() {
            return HashMap::new();
        }
        let Some(entry) = self.lookup(name) else {
            return HashMap::new();
        };
        let Some(_fence) = entry.read_unless_deleted() else {
            return HashMap::new();
        };
        let store = match self.entry_passages(&entry, &file_stem(name)) {
            Ok(store) => store,
            Err(error) => {
                tracing::warn!(
                    context = %name,
                    %error,
                    "marker resolution: passage store load failed; continuing without \
                     section/locator labels"
                );
                return HashMap::new();
            }
        };
        keys.filter_map(|(source, paragraph)| {
            let record = store.get(&source)?;
            let section = record.section_for(paragraph as usize).map(str::to_string);
            let locator = record.locator_for(paragraph as usize).cloned();
            if section.is_none() && locator.is_none() {
                return None;
            }
            Some(((source, paragraph), Markers { section, locator }))
        })
        .collect()
    }

    /// The source ids that currently have a registered passage.
    pub fn passage_sources(&self, name: &str) -> Option<io::Result<Vec<String>>> {
        let entry = self.lookup(name)?;
        let _fence = entry.read_unless_deleted()?;
        Some(
            self.entry_passages(&entry, &file_stem(name))
                .map(|store| store.source_ids()),
        )
    }

    /// [`Self::passage_sources`] with each source's metadata (#167)
    /// beside it — what `list_sources` renders its `entries` from.
    #[allow(clippy::type_complexity)]
    pub fn passage_source_entries(
        &self,
        name: &str,
    ) -> Option<io::Result<Vec<(String, crate::passages::SourceMeta)>>> {
        let entry = self.lookup(name)?;
        let _fence = entry.read_unless_deleted()?;
        Some(
            self.entry_passages(&entry, &file_stem(name))
                .map(|store| store.source_entries()),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;

    use super::*;
    use crate::registry::ContextMeta;
    use crate::registry::LOAD_FAILURE_RETRY;
    use crate::registry::paths::{passages_path, passages_wal_path, sources_path};
    use crate::registry::test_support::{plain, scratch_dir};
    use crate::registry::{BootOptions, ContextQuota};

    #[test]
    fn passages_store_lookup_and_survive_restart() {
        let dir = scratch_dir("passages");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            let mut passages = BTreeMap::new();
            passages.insert(
                "第1段落".to_string(),
                "青嶺酒造は、雲居県霧沢町にある日本酒の蔵元である。".to_string(),
            );
            assert_eq!(
                state
                    .store_passages("sake", plain(passages))
                    .unwrap()
                    .unwrap()
                    .stored,
                1
            );
        }

        // A fresh boot serves the registered passage; unknown sources
        // come back as missing rather than erroring.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let (passages, missing) = state
            .lookup_passages("sake", &["第1段落".to_string(), "第9段落".to_string()])
            .unwrap()
            .unwrap();
        assert!(passages["第1段落"].starts_with("青嶺酒造は"));
        assert_eq!(missing, vec!["第9段落".to_string()]);
        assert_eq!(
            state.passage_sources("sake").unwrap().unwrap(),
            vec!["第1段落"]
        );
        assert!(state.lookup_passages("nope", &[]).is_none());

        // Deleting the context removes the whole passage file family:
        // the log the store just wrote, any snapshot, and a legacy
        // sources file left over from before the migration.
        fs::write(
            sources_path(&dir, &file_stem("sake")),
            br#"{"legacy":"remnant"}"#,
        )
        .unwrap();
        state.delete("sake").unwrap().unwrap();
        assert!(!sources_path(&dir, &file_stem("sake")).exists());
        assert!(!passages_path(&dir, &file_stem("sake")).exists());
        assert!(!passages_wal_path(&dir, &file_stem("sake")).exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_passage_write_racing_a_delete_backs_off_at_the_tombstone() {
        let dir = scratch_dir("passage-delete-race");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert("第1段落".to_string(), "本文。".to_string());
        state
            .store_passages("sake", plain(passages.clone()))
            .unwrap()
            .unwrap();

        // The racing writer's handle predates the delete — exactly the
        // window the read fence exists for.
        let entry = state.lookup("sake").unwrap();
        state.delete("sake").unwrap().unwrap();
        assert!(
            entry.read_unless_deleted().is_none(),
            "a handle from before the delete must see the tombstone"
        );
        assert!(
            state.store_passages("sake", plain(passages)).is_none(),
            "the name is gone; nothing may recreate it"
        );
        assert!(
            !passages_wal_path(&dir, &file_stem("sake")).exists(),
            "no passage file rose from the dead"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// The passage store gets the same quarantine as the graph image:
    /// a broken snapshot or log answers its remembered refusal instead
    /// of being re-read on every passage request.
    #[test]
    fn a_failed_passage_load_is_quarantined_like_the_image() {
        let dir = scratch_dir("passage-quarantine");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .store_passages(
                    "sake",
                    BTreeMap::from([(
                        "a.md".to_string(),
                        crate::passages::PassageSubmission::plain("本文。"),
                    )]),
                )
                .unwrap()
                .unwrap();
            state.flush_dirty();
        }
        let log = dir.join("sake.passages.wal.jsonl");
        let healthy = fs::read(&log).unwrap();
        let mut corrupt = healthy.clone();
        corrupt.splice(0..0, *b"not json\n"); // a corrupt INTERIOR line
        fs::write(&log, &corrupt).unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let sources = ["a.md".to_string()];
        let first = state
            .lookup_passages("sake", &sources)
            .expect("registered")
            .unwrap_err();
        assert!(!first.to_string().contains("quarantined"), "{first}");

        fs::write(&log, &healthy).unwrap();
        let second = state
            .lookup_passages("sake", &sources)
            .expect("registered")
            .unwrap_err();
        assert!(second.to_string().contains("quarantined"), "{second}");

        state.age_load_failures("sake", LOAD_FAILURE_RETRY);
        let (passages, missing) = state
            .lookup_passages("sake", &sources)
            .expect("registered")
            .unwrap();
        assert!(missing.is_empty());
        assert_eq!(passages["a.md"], "本文。");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn eviction_drops_resident_passages_and_a_later_lookup_still_answers() {
        let dir = scratch_dir("passage-evict");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "第1段落".to_string(),
            "仕込み水は雲居山の伏流水。".to_string(),
        );
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();

        let entry = state.lookup("sake").unwrap();
        assert!(state.evict_entry("sake", &entry));
        assert!(
            entry.passages.lock().is_none(),
            "eviction must drop the resident passage store"
        );
        // Durability never depended on residency: the next access
        // reloads from the log (or the snapshot the eviction wrote).
        let (found, missing) = state
            .lookup_passages("sake", &["第1段落".to_string()])
            .unwrap()
            .unwrap();
        assert!(found["第1段落"].starts_with("仕込み水"));
        assert!(missing.is_empty());

        let _ = fs::remove_dir_all(dir);
    }

    /// ADR 0011 §4 steps 1–2 at the registry seam: the passage store's
    /// metadata decides which sources a window admits, and that set —
    /// resolved into a `SourceWindow` inside `read_context` — is what a
    /// windowed graph read sees. A source with no stored passage has no
    /// metadata and is invisible to every window, by the documented
    /// rule.
    #[test]
    fn window_source_names_joins_metadata_and_feeds_windowed_reads() {
        use crate::passages::{PassageSubmission, SourceFilter, SourceMeta};
        use taguru::deadline::Deadline;

        let dir = scratch_dir("window-join");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        let dated = |text: &str, date: u64| PassageSubmission {
            meta: SourceMeta {
                date: Some(date),
                ..SourceMeta::default()
            },
            ..PassageSubmission::plain(text.to_string())
        };
        let mut passages = BTreeMap::new();
        passages.insert("doc-old".to_string(), dated("旧杜氏は高瀬。", 100));
        passages.insert("doc-new".to_string(), dated("新杜氏は青山。", 200));
        state.store_passages("sake", passages).unwrap().unwrap();

        let assert_op = |object: &str, source: &str| crate::registry::AssocOp {
            subject: "蔵".to_string(),
            label: "杜氏".to_string(),
            object: object.to_string(),
            weight: 1.0,
            source: Some(source.to_string()),
            paragraph: None,
        };
        state
            .add_associations(
                "sake",
                vec![
                    assert_op("高瀬", "doc-old"),
                    assert_op("青山", "doc-new"),
                    // An associations-only source: no passage, no
                    // metadata, invisible to every window.
                    assert_op("幽霊", "doc-undated"),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();

        let window_after = SourceFilter {
            tags: Vec::new(),
            since: Some(150),
            until: None,
        };
        let eligible = state
            .window_source_names("sake", &window_after)
            .unwrap()
            .unwrap();
        assert_eq!(
            eligible.iter().collect::<Vec<_>>(),
            vec!["doc-new"],
            "the join admits exactly the in-window, dated sources"
        );

        let hits = state
            .read_context("sake", |context| {
                let window = context.source_window(eligible.iter().map(String::as_str));
                context.query_any_within(&["蔵"], &[], &[], &window)
            })
            .unwrap();
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].object, "青山");

        // The unwindowed read still sees all three, undated included.
        let all = state
            .read_context("sake", |context| context.query_any(&["蔵"], &[], &[]))
            .unwrap();
        assert_eq!(all.len(), 3);

        assert!(
            state.window_source_names("nope", &window_after).is_none(),
            "an unknown context is None, matching lookup_passages"
        );

        // The upper bound AT the boundary: [since, until) is half-open,
        // so until == doc-new's own date (200) excludes it — the
        // operator-confusion (< vs <=) this value is the only witness
        // for at this layer.
        let window_before = SourceFilter {
            tags: Vec::new(),
            since: None,
            until: Some(200),
        };
        let eligible = state
            .window_source_names("sake", &window_before)
            .unwrap()
            .unwrap();
        assert_eq!(eligible.iter().collect::<Vec<_>>(), vec!["doc-old"]);

        // source_effective_times, the consolidation audit's join input:
        // dated sources map to their date, the undated
        // associations-only source is absent, an unknown context is
        // None.
        let times = state.source_effective_times("sake").unwrap().unwrap();
        assert_eq!(times.get("doc-old"), Some(&100));
        assert_eq!(times.get("doc-new"), Some(&200));
        assert!(!times.contains_key("doc-undated"));
        assert!(state.source_effective_times("nope").is_none());

        let _ = fs::remove_dir_all(dir);
    }

    /// #678: `store_passages`'s own comment explains why the
    /// `passages_admission` lock is load-bearing — without it, two
    /// concurrent writes could both read the same pre-write usage and
    /// both pass a ceiling only one of them should cross. A one-byte
    /// ceiling (any real write crosses it) makes the outcome
    /// deterministic under a genuine race: exactly one of two
    /// concurrent first-writes must land, whichever wins the lock, and
    /// the other must see the ceiling already crossed.
    #[test]
    fn concurrent_store_passages_calls_serialize_at_the_quota_gate() {
        use std::thread;

        let dir = scratch_dir("passages-admission-quota");
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            None,
            BootOptions {
                context_quotas: HashMap::from([(
                    "sake".to_string(),
                    ContextQuota {
                        storage_bytes: Some(1),
                        cache_bytes: None,
                    },
                )]),
                ..BootOptions::default()
            },
        )
        .unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        let state = Arc::new(state);
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let handles: Vec<_> = (0..2)
            .map(|i| {
                let state = Arc::clone(&state);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    state.store_passages(
                        "sake",
                        BTreeMap::from([(
                            format!("race-{i}.md"),
                            crate::passages::PassageSubmission::plain("本文。"),
                        )]),
                    )
                })
            })
            .collect();
        let outcomes: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        let successes = outcomes
            .iter()
            .filter(|outcome| matches!(outcome, Some(Ok(_))))
            .count();
        let refusals = outcomes
            .iter()
            .filter(|outcome| matches!(outcome, Some(Err(PassagesWriteError::QuotaExceeded(_)))))
            .count();
        assert_eq!(
            (successes, refusals),
            (1, 1),
            "the admission lock must serialize the race so exactly one of two \
             concurrent first-writes past a one-byte ceiling lands: {outcomes:?}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// #678: `citation`'s three ordinary answers — a resolvable
    /// index (with and without markers), an unknown source, and an
    /// index past the source's stored range — had no dedicated
    /// coverage; only its HTTP-layer callers were exercised.
    #[test]
    fn citation_reports_found_unknown_source_and_out_of_range_index() {
        let dir = scratch_dir("citation");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // `section_for` walks back to the nearest marker at or before
        // `index` (a heading applies to every paragraph until the
        // next one); `locator_for` is an exact match only, never
        // extending to the next paragraph. The marker placement below
        // (section on paragraph 1, locator on paragraph 0) exercises
        // both semantics distinctly.
        state
            .store_passages(
                "sake",
                BTreeMap::from([(
                    "doc".to_string(),
                    crate::passages::PassageSubmission {
                        text: "第1段落。\n\n第2段落。".to_string(),
                        questions: Vec::new(),
                        sections: vec![(1, "はじめに".to_string())],
                        locators: vec![(
                            0,
                            crate::passages::Locator {
                                kind: "page".to_string(),
                                value: "1".to_string(),
                            },
                        )],
                        meta: crate::passages::SourceMeta::default(),
                    },
                )]),
            )
            .unwrap()
            .unwrap();

        let Some(Ok(CitationLookup::Found {
            text,
            section,
            locator,
        })) = state.citation("sake", "doc", 0)
        else {
            panic!("index 0 must resolve");
        };
        assert_eq!(text, "第1段落。");
        assert!(section.is_none(), "before the first section marker");
        assert_eq!(
            locator,
            Some(crate::passages::Locator {
                kind: "page".to_string(),
                value: "1".to_string(),
            })
        );

        let Some(Ok(CitationLookup::Found {
            text,
            section,
            locator,
        })) = state.citation("sake", "doc", 1)
        else {
            panic!("index 1 must resolve");
        };
        assert_eq!(text, "第2段落。");
        assert_eq!(section.as_deref(), Some("はじめに"));
        assert!(
            locator.is_none(),
            "a locator does not extend past its own paragraph"
        );

        assert!(matches!(
            state.citation("sake", "nope", 0),
            Some(Ok(CitationLookup::UnknownSource))
        ));
        assert!(matches!(
            state.citation("sake", "doc", 99),
            Some(Ok(CitationLookup::IndexOutOfRange))
        ));
        assert!(
            state.citation("nazo", "doc", 0).is_none(),
            "an unknown context is the outer None"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// #678: `citation`'s load-failure arm — same corrupt-log
    /// technique as `a_failed_passage_load_is_quarantined_like_the_image`.
    #[test]
    fn citation_reports_a_load_failure_as_err() {
        let dir = scratch_dir("citation-load-failure");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .store_passages(
                    "sake",
                    plain(BTreeMap::from([("doc".to_string(), "本文。".to_string())])),
                )
                .unwrap()
                .unwrap();
            state.flush_dirty();
        }
        let log = dir.join("sake.passages.wal.jsonl");
        let mut corrupt = fs::read(&log).unwrap();
        corrupt.splice(0..0, *b"not json\n");
        fs::write(&log, &corrupt).unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(matches!(state.citation("sake", "doc", 0), Some(Err(_))));

        let _ = fs::remove_dir_all(&dir);
    }

    /// #678: `resolve_markers`'s best-effort degrades — an empty key
    /// iterator skips the store load entirely (the "graph-only
    /// response never touches passages" contract), an unknown context
    /// degrades to an empty map — and the null-means-nothing contract:
    /// a key with no covering marker simply never lands in the result,
    /// it is not fabricated as `Markers::default()`.
    #[test]
    fn resolve_markers_only_maps_keys_that_actually_carry_a_marker() {
        let dir = scratch_dir("resolve-markers");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        assert!(
            state.resolve_markers("sake", std::iter::empty()).is_empty(),
            "an empty key iterator must skip the store load entirely"
        );
        assert!(
            state
                .resolve_markers(
                    "no-such-context",
                    std::iter::once(("doc".to_string(), 0u32))
                )
                .is_empty(),
            "a non-empty iterator against an unknown context degrades to empty"
        );

        // `section_for` walks back to the nearest preceding marker, so
        // the marker sits on paragraph 1 (not 0): paragraph 0 then has
        // genuinely no marker of either kind, the case this test needs
        // to prove is dropped, not fabricated as an empty `Markers`.
        state
            .store_passages(
                "sake",
                BTreeMap::from([(
                    "doc".to_string(),
                    crate::passages::PassageSubmission {
                        text: "第1段落。\n\n第2段落。".to_string(),
                        questions: Vec::new(),
                        sections: vec![(1, "はじめに".to_string())],
                        locators: Vec::new(),
                        meta: crate::passages::SourceMeta::default(),
                    },
                )]),
            )
            .unwrap()
            .unwrap();

        let keys = vec![
            ("doc".to_string(), 1u32),  // carries a section marker
            ("doc".to_string(), 0u32),  // no marker at all
            ("nope".to_string(), 0u32), // unknown source
        ];
        let resolved = state.resolve_markers("sake", keys.into_iter());
        assert_eq!(
            resolved.len(),
            1,
            "only the keyed marker survives: {resolved:?}"
        );
        let markers = &resolved[&("doc".to_string(), 1)];
        assert_eq!(markers.section.as_deref(), Some("はじめに"));
        assert!(markers.locator.is_none());

        let _ = fs::remove_dir_all(&dir);
    }

    /// #678: `resolve_markers`'s load-failure degrade — an empty map,
    /// not an error, since the markers it resolves are enrichment on
    /// top of an association read, never a hard dependency.
    #[test]
    fn resolve_markers_degrades_to_an_empty_map_on_a_load_failure() {
        let dir = scratch_dir("resolve-markers-load-failure");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .store_passages(
                    "sake",
                    plain(BTreeMap::from([("doc".to_string(), "本文。".to_string())])),
                )
                .unwrap()
                .unwrap();
            state.flush_dirty();
        }
        let log = dir.join("sake.passages.wal.jsonl");
        let mut corrupt = fs::read(&log).unwrap();
        corrupt.splice(0..0, *b"not json\n");
        fs::write(&log, &corrupt).unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(
            state
                .resolve_markers("sake", std::iter::once(("doc".to_string(), 0u32)))
                .is_empty()
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// #678: `passage_source_entries` — `passage_sources` with each
    /// source's metadata beside it, what `list_sources` renders its
    /// `entries` from. Had no dedicated coverage of its own.
    #[test]
    fn passage_source_entries_pairs_ids_with_their_metadata() {
        let dir = scratch_dir("passage-source-entries");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .store_passages(
                "sake",
                BTreeMap::from([(
                    "doc".to_string(),
                    crate::passages::PassageSubmission {
                        text: "本文。".to_string(),
                        questions: Vec::new(),
                        sections: Vec::new(),
                        locators: Vec::new(),
                        meta: crate::passages::SourceMeta {
                            stored_at: None,
                            date: Some(1_700_000_000),
                            tags: vec!["tag-a".to_string()],
                        },
                    },
                )]),
            )
            .unwrap()
            .unwrap();

        let sources = state.passage_sources("sake").unwrap().unwrap();
        let entries = state.passage_source_entries("sake").unwrap().unwrap();
        assert_eq!(
            entries.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>(),
            sources,
            "passage_source_entries must name exactly what passage_sources does"
        );
        let (_, meta) = entries.iter().find(|(id, _)| id == "doc").unwrap();
        assert_eq!(meta.date, Some(1_700_000_000));
        assert_eq!(meta.tags, vec!["tag-a".to_string()]);

        assert!(
            state.passage_source_entries("nazo").is_none(),
            "an unknown context is the outer None"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
