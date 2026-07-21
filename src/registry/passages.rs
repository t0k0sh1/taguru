use std::collections::{BTreeMap, HashMap};
use std::io;
use std::sync::atomic::Ordering;

use super::{AppState, CitationLookup, file_stem};

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
    /// The section label comes from the same resident record via
    /// `section_for`, `None` when the index falls outside every
    /// section the source's import stored.
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
        Some(Ok(CitationLookup::Found(text.to_string(), section)))
    }

    /// Resolves `(source, paragraph)` locators — as found on
    /// attributions — to the section label governing each, batching
    /// every pair an association-bearing response needs into one
    /// passage-store load rather than one per attribution. Best-effort:
    /// an unknown context, a deleted entry, or a passage-store load
    /// failure all resolve to an empty map rather than an error.
    /// Association reads (recall, query, explore, activate,
    /// unreachable_from) are graph reads first; a section label is
    /// enrichment on top, not a hard dependency the way `citation`'s
    /// text lookup is. A pair with no covering marker is simply absent
    /// from the map — the same null-means-nothing contract
    /// `Attribution::paragraph` already makes, never a fabricated
    /// label. An empty `locators` skips the passage-store load
    /// entirely, so a graph-only response (no attribution carries a
    /// paragraph) never touches passages.
    pub fn resolve_sections(
        &self,
        name: &str,
        locators: impl Iterator<Item = (String, u32)>,
    ) -> HashMap<(String, u32), String> {
        let mut locators = locators.peekable();
        if locators.peek().is_none() {
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
                    "section resolution: passage store load failed; continuing without section labels"
                );
                return HashMap::new();
            }
        };
        locators
            .filter_map(|(source, paragraph)| {
                let record = store.get(&source)?;
                let section = record.section_for(paragraph as usize)?;
                Some(((source, paragraph), section.to_string()))
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

    use super::*;
    use crate::registry::ContextMeta;
    use crate::registry::LOAD_FAILURE_RETRY;
    use crate::registry::paths::{passages_path, passages_wal_path, sources_path};
    use crate::registry::test_support::{plain, scratch_dir};

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
}
