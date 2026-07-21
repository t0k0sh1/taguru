use std::collections::BTreeMap;
use std::io;
use std::sync::atomic::Ordering;

use taguru::deadline::Deadline;

use crate::storage::{remove_persisted_file, write_atomic};
use crate::wal::WalOp;

use super::{
    AccessError, AppState, AssocOp, ImportMarker, PartialWrite, applied_count, apply_in_order,
    file_stem, import_marker_path,
};

impl AppState {
    /// Withdraws one association from a context outright — the surgical
    /// correction for a single fact that should never have been
    /// asserted, where [`AppState::retract_source`] would discard the
    /// whole document's contribution. Returns how many attributions
    /// were unlinked, or `None` when the triple names no live edge
    /// (nothing was changed — the caller answers honestly instead of
    /// pretending a write happened).
    pub fn retract_association(
        &self,
        name: &str,
        subject: &str,
        label: &str,
        object: &str,
    ) -> Result<Option<usize>, AccessError> {
        let op = WalOp::RetractAssociation {
            subject: subject.to_string(),
            label: label.to_string(),
            object: object.to_string(),
        };
        self.logged_write(
            name,
            std::slice::from_ref(&op),
            |context| context.retract_association(subject, label, object),
            // The single RetractAssociation op never fails to apply.
            |_| 1,
        )
    }

    /// The read-only twin of [`Self::retract_source`]'s edge count —
    /// `POST /import?dry_run=true`'s preview of what a retraction would
    /// report, without unlinking anything.
    pub fn count_source_edges(&self, name: &str, source: &str) -> Result<usize, AccessError> {
        self.read_context(name, |context| context.count_source_edges(source))
    }

    /// Withdraws one source from a context — its graph contributions and
    /// its registered passage — the per-document differential-sync move:
    /// retract the old version of a changed document, then re-ingest the
    /// new one, instead of rebuilding the whole context. Returns how
    /// many associations were touched and whether a passage was removed.
    ///
    /// Brackets [`Self::retract_source_unmarked`]'s two independently
    /// durable writes (the graph's own WAL, then the passage store's)
    /// with the same batch-open marker `apply_batch` uses: a crash
    /// between them would otherwise leave the graph durably retracted
    /// while the passage text survives on disk, undetected by boot or
    /// `taguru inspect` — the same hazard the marker already closes for
    /// a whole import batch, at the smaller two-write scale of a
    /// standalone retraction. `apply_batch` calls
    /// [`Self::retract_source_unmarked`] directly instead of this
    /// method: its own marker already brackets that call along with the
    /// store/associate/alias steps that follow it, and clearing the
    /// marker here too would reopen the batch to the exact gap it
    /// exists to close.
    pub fn retract_source(&self, name: &str, source: &str) -> Result<(usize, bool), AccessError> {
        self.open_import_marker(name, source).map_err(|error| {
            AccessError::Unpersisted(format!(
                "import marker not persisted: {error} — nothing was retracted"
            ))
        })?;
        let (touched, passage_removed, passage_removal_errored) =
            self.retract_source_unmarked(name, source)?;
        // A genuine passage-store failure must leave the marker in
        // place: clearing it here would erase the only surviving
        // witness (surfaced by boot and `taguru inspect`) that this
        // source's truth is now half-applied — the graph side already
        // retracted, the passage still sitting on disk. "Nothing was
        // there to remove" (raced with a delete, or never had a
        // passage) is not this case and still clears normally.
        if !passage_removal_errored {
            self.clear_import_marker(name, source);
        }
        Ok((touched, passage_removed))
    }

    /// The marker-less core of [`Self::retract_source`] — see there for
    /// behavior and for why only `apply_batch` should call this
    /// directly. The third element of the returned tuple is `true`
    /// only when the passage store's own removal genuinely errored
    /// (store unavailable, or its `retract` call failed) — as opposed
    /// to `false`/`false`, which also covers "there was nothing to
    /// remove." `apply_batch` ignores it: its own `store_passages` call
    /// right after overwrites whatever stale passage a failed
    /// retraction left behind, so the failure there is self-healing.
    /// [`Self::retract_source`] is the one caller that cannot heal it
    /// the same way and uses it to decide whether clearing its marker
    /// is safe.
    pub(crate) fn retract_source_unmarked(
        &self,
        name: &str,
        source: &str,
    ) -> Result<(usize, bool, bool), AccessError> {
        let op = WalOp::RetractSource {
            source: source.to_string(),
        };
        let touched = self.logged_write(
            name,
            std::slice::from_ref(&op),
            |context| context.retract_source(source).unwrap_or(0),
            // The single RetractSource op never fails to apply.
            |_| 1,
        )?;

        let Some(entry) = self.lookup(name) else {
            // Raced with a delete; there is nothing left to clean up.
            return Ok((touched, false, false));
        };
        let Some(_fence) = entry.read_unless_deleted() else {
            // Same race, one step later: the delete beat us to the lock.
            return Ok((touched, false, false));
        };
        // The graph retraction above already succeeded; a passage-side
        // failure must not turn it into an error, only into an honest
        // `passage_removed: false` — paired with a `true` third element
        // so a marker-clearing caller can still tell "nothing to
        // remove" and "removal genuinely failed" apart.
        let (passage_removed, passage_removal_errored) =
            match self.entry_passages(&entry, &file_stem(name)) {
                Ok(store) => match store.retract(source) {
                    Ok(removed) => {
                        if removed {
                            self.refresh_bm25(
                                &entry,
                                &store,
                                std::slice::from_ref(&source.to_string()),
                            );
                            entry.passages_embed_dirty.store(true, Ordering::Relaxed);
                            // Same bump-after-apply as store_passages:
                            // the retraction landed in the log, so the
                            // watermark moved.
                            entry
                                .passage_revision
                                .fetch_max(store.watermark(), Ordering::Relaxed);
                        }
                        (removed, false)
                    }
                    Err(error) => {
                        tracing::warn!("passage for '{source}' not removed from disk: {error}");
                        (false, true)
                    }
                },
                Err(error) => {
                    tracing::warn!("passages for '{name}' unavailable during retract: {error}");
                    (false, true)
                }
            };
        Ok((touched, passage_removed, passage_removal_errored))
    }

    /// Opens the batch-open marker for one source's import — see
    /// [`import_marker_path`] for what it means while it exists. Called
    /// by `apply_batch` before the batch's first mutation, and by
    /// [`Self::retract_source`] before its own two-write sequence; an
    /// error refuses the operation, because proceeding would silently
    /// reintroduce the undetectable-tear gap the marker exists to close
    /// (and a disk that cannot land a hundred-byte marker is not going
    /// to land the writes either). `write_atomic` makes it durable,
    /// directory entry included, before any tracked write can need it.
    pub fn open_import_marker(&self, context: &str, source: &str) -> io::Result<()> {
        let marker = ImportMarker {
            context: context.to_string(),
            source: source.to_string(),
        };
        let body = serde_json::to_vec(&marker).map_err(io::Error::from)?;
        write_atomic(
            &import_marker_path(&self.0.data_dir, &file_stem(context), source),
            &body,
        )
    }

    /// Removes one source's batch-open marker: the batch completed, or
    /// the operator repaired the tear by retracting the source outright
    /// (either way the source's truth is consistent again). Best
    /// effort, loudly: a marker that cannot be removed only means boot
    /// keeps reporting a tear that is no longer one, until a re-import
    /// or a hand unlink clears it.
    pub fn clear_import_marker(&self, context: &str, source: &str) {
        let path = import_marker_path(&self.0.data_dir, &file_stem(context), source);
        if let Err(error) = remove_persisted_file(&path)
            && error.kind() != io::ErrorKind::NotFound
        {
            tracing::warn!(
                context,
                source,
                %error,
                "import marker not removed; boot will keep reporting this batch as torn",
            );
        }
    }

    /// Applies one document's extracted facts, staging them in the WAL
    /// first. `Ok(Err(PartialWrite))` reproduces the associations
    /// endpoint's historic partial semantics: items before the failing
    /// one are applied, each all-or-nothing in the library.
    pub fn add_associations(
        &self,
        name: &str,
        ops: Vec<AssocOp>,
        deadline: Deadline,
    ) -> Result<Result<usize, PartialWrite>, AccessError> {
        if deadline.expired() {
            return Err(AccessError::DeadlineExceeded);
        }
        let ops = self.clamp_out_of_range_paragraphs(name, ops);
        let wal_ops: Vec<WalOp> = ops.into_iter().map(WalOp::Associate).collect();
        self.logged_write(
            name,
            &wal_ops,
            |context| apply_in_order(context, &wal_ops),
            applied_count,
        )
    }

    /// Drops a paragraph locator that falls outside its source's
    /// stored passage, the same silent-drop posture `StoreOutcome`
    /// already applies to out-of-range questions and sections. This
    /// is the general-purpose backstop: callers that hand the batch's
    /// passage text to the ingest pipeline get a cheaper, unconditional
    /// clamp there, but a bare HTTP call or a later `add_associations`
    /// against an already-stored source has no such text in hand, so
    /// this checks the resident passage store instead.
    ///
    /// Best-effort like [`AppState::resolve_sections`]: an unknown
    /// context, a deleted entry, a source with no stored passage, or a
    /// store load failure all leave `paragraph` as given rather than
    /// fail the write — an unresolved locator is still meaningful
    /// (just without a section label), so this only removes locators
    /// it can positively prove are out of range.
    fn clamp_out_of_range_paragraphs(&self, name: &str, mut ops: Vec<AssocOp>) -> Vec<AssocOp> {
        if !ops.iter().any(|op| op.paragraph.is_some()) {
            return ops;
        }
        let Some(entry) = self.lookup(name) else {
            return ops;
        };
        let Some(_fence) = entry.read_unless_deleted() else {
            return ops;
        };
        let Ok(store) = self.entry_passages(&entry, &file_stem(name)) else {
            return ops;
        };
        for op in &mut ops {
            let Some(paragraph) = op.paragraph else {
                continue;
            };
            let Some(source) = op.source.as_deref() else {
                continue;
            };
            let Some(record) = store.get(source) else {
                continue;
            };
            if paragraph as usize >= record.paragraphs.len() {
                op.paragraph = None;
            }
        }
        ops
    }

    /// Registers alias batches (concepts then labels, in map order),
    /// staged in the WAL first — the same partial semantics as
    /// associations, with the conflict/capacity distinction preserved
    /// in [`PartialWrite::full`].
    pub fn add_aliases(
        &self,
        name: &str,
        concepts: &BTreeMap<String, String>,
        labels: &BTreeMap<String, String>,
    ) -> Result<Result<usize, PartialWrite>, AccessError> {
        let mut wal_ops = Vec::with_capacity(concepts.len() + labels.len());
        for (alias, canonical) in concepts {
            wal_ops.push(WalOp::AliasConcept {
                alias: alias.clone(),
                canonical: canonical.clone(),
            });
        }
        for (alias, canonical) in labels {
            wal_ops.push(WalOp::AliasLabel {
                alias: alias.clone(),
                canonical: canonical.clone(),
            });
        }
        self.logged_write(
            name,
            &wal_ops,
            |context| apply_in_order(context, &wal_ops),
            applied_count,
        )
    }

    /// Withdraws alias registrations (concept spellings then label
    /// spellings, in the order given), staged in the WAL first — the
    /// same partial semantics as every batch write. `Ok(Ok(n))`
    /// counts spellings withdrawn; canonical names and unknown
    /// spellings are refused as conflicts, never applied silently.
    pub fn remove_aliases(
        &self,
        name: &str,
        concepts: &[String],
        labels: &[String],
    ) -> Result<Result<usize, PartialWrite>, AccessError> {
        let mut wal_ops = Vec::with_capacity(concepts.len() + labels.len());
        for alias in concepts {
            wal_ops.push(WalOp::UnaliasConcept {
                alias: alias.clone(),
            });
        }
        for alias in labels {
            wal_ops.push(WalOp::UnaliasLabel {
                alias: alias.clone(),
            });
        }
        self.logged_write(
            name,
            &wal_ops,
            |context| apply_in_order(context, &wal_ops),
            applied_count,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::registry::ContextMeta;
    use crate::registry::paths::import_marker_paths;
    use crate::registry::test_support::{assoc_op, plain, scratch_dir};
    use crate::storage::{clear_persistence_fault, fail_persistence_ops_after};

    /// Standalone `retract_source` — the only path the HTTP endpoint and
    /// the MCP tool ever call — used to bracket its two independently
    /// durable writes (the graph's WAL, then the passage store's) with
    /// nothing: a crash between them left the graph durably retracted
    /// while the passage text survived on disk, invisible to boot or
    /// `taguru inspect`. Every fault point must now leave either a
    /// completed, marker-free retraction, or a surviving marker naming
    /// the tear — never a silent gap between the two stores.
    #[test]
    fn every_standalone_retract_persistence_failure_is_detected_or_completes() {
        let mut exhausted = false;
        for failure in 0..24 {
            let dir = scratch_dir(&format!("retract-fault-{failure}"));
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("doc"))],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            let mut passages = BTreeMap::new();
            passages.insert("doc".to_string(), "杜氏は高瀬。".to_string());
            state
                .store_passages("sake", plain(passages))
                .unwrap()
                .unwrap();

            fail_persistence_ops_after(failure);
            let first = state.retract_source("sake", "doc");
            let past_end = clear_persistence_fault();
            let marker = import_marker_path(&dir, "sake", "doc");

            if past_end {
                assert!(
                    first.is_ok(),
                    "the past-end attempt must complete: {first:?}"
                );
                assert!(!marker.exists());
            } else {
                match &first {
                    // A witness must survive whenever the graph side may
                    // already be durably retracted while the passage side
                    // never ran: any refusal other than the marker write
                    // itself failing (which leaves nothing behind to
                    // witness, since nothing happened yet).
                    Err(AccessError::Unpersisted(message)) => {
                        let before_marker = message.contains("import marker");
                        assert_eq!(
                            marker.exists(),
                            !before_marker,
                            "a stopped retraction at step {failure} lost its tear witness: {first:?}"
                        );
                    }
                    // The graph write itself never swallows a failure —
                    // the only way this call can succeed while a fault
                    // fired somewhere is a passage-side failure folded
                    // into an honest `passage_removed: false`. The
                    // witness must survive exactly that swallow, or the
                    // half-applied state it names (graph retracted,
                    // passage still on disk) becomes permanently
                    // invisible to boot and `taguru inspect`.
                    Ok(_) => {
                        assert!(
                            marker.exists(),
                            "a swallowed passage failure at step {failure} still cleared \
                             the tear witness: {first:?}"
                        );
                    }
                    Err(_) => {}
                }
                // Retracting again is the documented repair, and
                // retract_source is idempotent per-source, so it is
                // exact even when the injected failure was swallowed
                // internally or only prevented marker cleanup.
                state.retract_source("sake", "doc").unwrap();
                assert!(
                    !marker.exists(),
                    "repair did not clear failure step {failure}"
                );
            }

            // A fully retracted edge stays (storage is append-only) but
            // nets to zero attributions — the same end-state
            // `retract_source_withdraws_its_contributions` checks.
            let attributions_gone = state
                .read_context("sake", |context| {
                    context.query(Some("蔵"), None, Some("高瀬"))[0]
                        .attributions
                        .is_empty()
                })
                .unwrap();
            assert!(
                attributions_gone,
                "retry at step {failure} did not retract the association"
            );
            let (found, missing) = state
                .lookup_passages("sake", &["doc".to_string()])
                .unwrap()
                .unwrap();
            assert!(
                !found.contains_key("doc") && missing == vec!["doc".to_string()],
                "retry at step {failure} did not retract the passage"
            );

            drop(state);
            let _ = fs::remove_dir_all(&dir);

            if past_end {
                exhausted = true;
                break;
            }
        }
        assert!(exhausted, "standalone retraction exceeded the sweep bound");
    }

    /// The import batch-open marker: opened before a batch's first
    /// mutation, cleared only after its last — while it exists, boot
    /// and inspect can name a half-applied source nothing else sees.
    #[test]
    fn import_markers_open_clear_and_sweep_with_their_context() {
        let dir = scratch_dir("import-markers");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        state.open_import_marker("sake", "doc-1").unwrap();
        let marker = import_marker_path(&dir, "sake", "doc-1");
        assert!(marker.exists(), "open writes the marker");
        // Distinct sources get distinct files — concurrent imports of
        // one context never race on a shared marker.
        state.open_import_marker("sake", "doc-2").unwrap();
        assert_eq!(import_marker_paths(&dir, "sake").len(), 2);
        // The content names the pair, so reports never decode filenames.
        let parsed: ImportMarker = serde_json::from_slice(&fs::read(&marker).unwrap()).unwrap();
        assert_eq!(
            (parsed.context.as_str(), parsed.source.as_str()),
            ("sake", "doc-1")
        );

        state.clear_import_marker("sake", "doc-1");
        assert!(!marker.exists(), "clear removes exactly its own marker");
        assert_eq!(import_marker_paths(&dir, "sake").len(), 1);

        // Deletion takes the survivors with the family: a marker must
        // not have boot report a tear in a context that is gone.
        state.delete("sake").unwrap().unwrap();
        assert!(
            import_marker_paths(&dir, "sake").is_empty(),
            "delete sweeps markers"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
