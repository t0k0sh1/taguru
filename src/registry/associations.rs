use std::collections::BTreeMap;
use std::io;
use std::sync::atomic::Ordering;

use taguru::context::Context;
use taguru::deadline::Deadline;

use crate::storage::{remove_persisted_file, write_atomic};
use crate::wal::WalOp;

use super::{
    AccessError, AppState, AssocOp, ChangeKind, ImportMarker, PartialWrite, applied_count,
    apply_in_order, file_stem, import_marker_path,
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
        self.retract_single_op(name, op, |context| {
            context.retract_association(subject, label, object)
        })
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

    /// The read-only twin of [`Self::retract_source`] (#437): the same
    /// `(associations_touched, passage_removed)` the real call would
    /// report, with nothing written — no import marker, no WAL op, no
    /// graph mutation, no passage removal. The graph count is
    /// [`crate::context::Context::count_source_edges`], the exact
    /// preview `/import?dry_run=true` already trusts; the passage half
    /// is a presence check. Advisory in the way every preview is: a
    /// write landing between this and the real retraction can change
    /// the numbers.
    pub fn retract_source_preview(
        &self,
        name: &str,
        source: &str,
    ) -> Result<(usize, bool), AccessError> {
        let touched = self.read_context(name, |context| context.count_source_edges(source))?;
        let Some(entry) = self.lookup(name) else {
            return Ok((touched, false));
        };
        let Some(_fence) = entry.read_unless_deleted() else {
            return Ok((touched, false));
        };
        // A passage-store load failure degrades to "no passage" rather
        // than failing the preview: the graph half is the load-bearing
        // number, and the real retraction reports its own passage
        // failure honestly when it happens.
        let passage_present = match self.entry_passages(&entry, &file_stem(name)) {
            Ok(store) => store.get(source).is_some(),
            Err(_) => false,
        };
        Ok((touched, passage_present))
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
        // The graph side's own honest no-op/real-change signal — see
        // `retract_single_op`. Deliberately NOT widened by a
        // passage-presence check taken before this write: presence is
        // not removal, and folding a snapshot into `applied` here
        // would advance `graph_revision`/the change feed for a call
        // whose passage-side removal, below, hasn't happened yet and
        // could still fail (#676 review). The passage side instead
        // reports its own real outcome directly to the feed, below.
        let touched = self
            .retract_single_op(name, op, |context| context.retract_source(source))?
            .unwrap_or(0);

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
                            // A source whose graph edge was already
                            // gone (`touched == 0`, retracted
                            // independently earlier) but whose passage
                            // genuinely comes off now is still a real
                            // content change a polling client should
                            // see. `touched > 0` already got this event
                            // from `retract_single_op`'s own write
                            // above (`events_of_ops`) — push it again
                            // only for the touched-nothing-in-the-graph
                            // case, or a real graph retraction would
                            // double up.
                            if touched == 0 {
                                entry.changes.lock().push(ChangeKind::SourceRetracted {
                                    source: source.to_string(),
                                });
                            }
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
        // The opt-out for an idempotent offline importer (issue #443
        // item 2): with re-run-the-sync as the documented recovery,
        // tear detection buys nothing, and this `write_atomic` is 2 of
        // the ~3 fsyncs each imported file costs. Only the open is
        // gated — `clear_import_marker` stays active so a completed
        // batch still heals a stale marker from a marker-enabled run.
        if !self.0.import_markers_enabled {
            return Ok(());
        }
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
    /// already applies to out-of-range questions, sections, and
    /// locators. This
    /// is the general-purpose backstop: callers that hand the batch's
    /// passage text to the ingest pipeline get a cheaper, unconditional
    /// clamp there, but a bare HTTP call or a later `add_associations`
    /// against an already-stored source has no such text in hand, so
    /// this checks the resident passage store instead.
    ///
    /// Best-effort like [`AppState::resolve_markers`]: an unknown
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

    /// Shared shape of [`Self::retract_association`] and
    /// [`Self::retract_source_unmarked`]: a single WAL op whose
    /// `operate` closure reports whether it actually touched a live
    /// edge/source via `Option<usize>` — `None` names no live target,
    /// nothing changed. Unlike `logged_write`'s other callers (whole
    /// batches, where a fixed `applied_count` of the op count is
    /// correct because a batch write either lands wholesale or fails),
    /// a single retract op can itself be a genuine no-op, so `applied`
    /// here honestly reflects that instead of the `|_| 1` every
    /// earlier version of these two callers hardcoded — the bug this
    /// helper fixes (#676): a no-op retract must not advance
    /// `graph_revision` or emit a change-feed event for a change that
    /// never happened.
    fn retract_single_op(
        &self,
        name: &str,
        op: WalOp,
        operate: impl FnOnce(&mut Context) -> Option<usize>,
    ) -> Result<Option<usize>, AccessError> {
        self.logged_write(name, std::slice::from_ref(&op), operate, |result| {
            result.is_some() as usize
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::registry::ContextMeta;
    use crate::registry::changes::ChangesOutcome;
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

    /// CodeRabbit review on PR #681 (issue #676): a source the graph
    /// side never saw (`Context::retract_source` returns `None` for a
    /// source this context never associated anything from, per its own
    /// doc — a genuine graph-side no-op, distinct from a `Some(0)` for
    /// a source that WAS seen but now carries zero live edges) but
    /// whose passage removal genuinely fails must not advance
    /// `graph_revision` or the change feed either. An earlier version
    /// of the fix took a passage-PRESENCE snapshot before the graph
    /// write and folded that into `applied` — which would still
    /// advance both here, since presence isn't removal. The current
    /// design instead has the passage side push its own event only
    /// once its removal actually lands (see `retract_source_unmarked`),
    /// so this failure path never reaches the feed at all. Sweeps
    /// `fail_persistence_ops_after` to land on "the graph write itself
    /// lands (as a no-op), but the passage removal right after it
    /// fails."
    #[test]
    fn a_failed_passage_removal_after_an_already_gone_graph_edge_does_not_advance_the_revision_or_feed()
     {
        let mut exercised_target_case = false;
        let mut exhausted = false;
        for failure in 0..24 {
            let dir = scratch_dir(&format!("retract-source-passage-fault-{failure}"));
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            // No association ever names this source: the graph side
            // has never seen it, so `Context::retract_source` returns
            // `None`, not `Some(0)`. `store_passages` is its own store,
            // independent of the graph's `source_ids` — this is the
            // only way to get a passage present with the graph side
            // still a genuine no-op.
            let mut passages = BTreeMap::new();
            passages.insert("doc".to_string(), "杜氏は高瀬。".to_string());
            state
                .store_passages("sake", plain(passages))
                .unwrap()
                .unwrap();

            let before = state.directory_entry("sake").unwrap().revision.graph;
            let cursor = match state.context_changes("sake", None, 100).unwrap() {
                ChangesOutcome::Page { next, .. } => next,
                ChangesOutcome::Stale => panic!("a fresh context's cursor must not be stale"),
            };

            fail_persistence_ops_after(failure);
            let outcome = state.retract_source_unmarked("sake", "doc");
            let past_end = clear_persistence_fault();

            if past_end {
                exhausted = true;
                drop(state);
                let _ = fs::remove_dir_all(&dir);
                break;
            }

            // Only "the graph write landed (a genuine no-op), the
            // passage removal failed" is this test's target — a
            // graph-write failure itself short-circuits above (`?`),
            // and a case where the passage removal actually succeeded
            // isn't a failure to guard against.
            if let Ok((touched, passage_removed, passage_removal_errored)) = outcome
                && touched == 0
                && !passage_removed
                && passage_removal_errored
            {
                exercised_target_case = true;
                let after = state.directory_entry("sake").unwrap().revision.graph;
                assert_eq!(
                    before, after,
                    "failure {failure}: a failed passage removal after an \
                     already-gone graph edge must not advance graph_revision"
                );
                match state.context_changes("sake", Some(&cursor), 100).unwrap() {
                    ChangesOutcome::Page { events, .. } => assert!(
                        events.is_empty(),
                        "failure {failure}: must not emit a change-feed event"
                    ),
                    ChangesOutcome::Stale => panic!("the cursor must still be live"),
                }
            }

            drop(state);
            let _ = fs::remove_dir_all(&dir);
        }
        assert!(exhausted, "the fault sweep never reached completion");
        assert!(
            exercised_target_case,
            "the fault sweep never hit the target case (graph no-op, \
             passage removal failed) — widen the range or check the \
             fault-injection points still line up"
        );
    }

    /// #676: `retract_association` naming no live edge must report
    /// `None` (already covered by the type) AND leave `graph_revision`
    /// and the change feed untouched — before `retract_single_op`,
    /// the hardcoded `|_| 1` `applied` count advanced both for a
    /// retraction that changed nothing.
    #[test]
    fn a_no_op_retract_association_does_not_advance_the_graph_revision_or_change_feed() {
        let dir = scratch_dir("retract-association-no-op-revision");
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

        let before = state.directory_entry("sake").unwrap().revision.graph;
        let cursor = match state.context_changes("sake", None, 100).unwrap() {
            ChangesOutcome::Page { next, .. } => next,
            ChangesOutcome::Stale => panic!("a fresh context's cursor must not be stale"),
        };

        // Names no live edge: "蔵" and "杜氏" are real, but this
        // triple was never asserted.
        let outcome = state
            .retract_association("sake", "蔵", "杜氏", "存在しない")
            .unwrap();
        assert_eq!(outcome, None, "a no-op retract must report None honestly");

        let after = state.directory_entry("sake").unwrap().revision.graph;
        assert_eq!(
            before, after,
            "a no-op retract_association must not advance graph_revision"
        );
        match state.context_changes("sake", Some(&cursor), 100).unwrap() {
            ChangesOutcome::Page { events, more, .. } => {
                assert!(
                    events.is_empty(),
                    "a no-op retract_association must not emit a change-feed event"
                );
                assert!(!more);
            }
            ChangesOutcome::Stale => panic!("the cursor must still be live"),
        }

        let _ = fs::remove_dir_all(&dir);
    }

    /// The `retract_source`/`retract_source_unmarked` twin of
    /// `a_no_op_retract_association_does_not_advance_the_graph_revision_or_change_feed`
    /// (#676).
    #[test]
    fn a_no_op_retract_source_does_not_advance_the_graph_revision_or_change_feed() {
        let dir = scratch_dir("retract-source-no-op-revision");
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

        let before = state.directory_entry("sake").unwrap().revision.graph;
        let cursor = match state.context_changes("sake", None, 100).unwrap() {
            ChangesOutcome::Page { next, .. } => next,
            ChangesOutcome::Stale => panic!("a fresh context's cursor must not be stale"),
        };

        // Names a source that was never ingested.
        let (touched, passage_removed) = state.retract_source("sake", "never-existed").unwrap();
        assert_eq!(touched, 0);
        assert!(!passage_removed);

        let after = state.directory_entry("sake").unwrap().revision.graph;
        assert_eq!(
            before, after,
            "a no-op retract_source must not advance graph_revision"
        );
        match state.context_changes("sake", Some(&cursor), 100).unwrap() {
            ChangesOutcome::Page { events, more, .. } => {
                assert!(
                    events.is_empty(),
                    "a no-op retract_source must not emit a change-feed event"
                );
                assert!(!more);
            }
            ChangesOutcome::Stale => panic!("the cursor must still be live"),
        }

        let _ = fs::remove_dir_all(&dir);
    }

    /// #678: the graph half of the preview is load-bearing (its own
    /// `read_context` call fails outright on a bad context), but a
    /// passage-store load failure degrades to "no passage" rather than
    /// failing the whole preview — the doc comment's own contract.
    /// The `read_unless_deleted` early return between those two is a
    /// genuine race window (a concurrent delete landing between the
    /// graph read above it and this fresh `lookup`) that no
    /// single-threaded test can hit deterministically without a new
    /// fault-injection hook; not exercised here.
    #[test]
    fn retract_source_preview_reports_the_graph_count_and_degrades_passage_presence_on_a_load_failure()
     {
        let dir = scratch_dir("retract-source-preview");
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
        state
            .store_passages(
                "sake",
                plain(BTreeMap::from([(
                    "doc".to_string(),
                    "杜氏は高瀬。".to_string(),
                )])),
            )
            .unwrap()
            .unwrap();

        assert_eq!(
            state.retract_source_preview("sake", "doc").unwrap(),
            (1, true),
            "a resident passage store reports the edge count and presence honestly"
        );

        state.flush_dirty();
        drop(state);
        let log = dir.join("sake.passages.wal.jsonl");
        let mut corrupt = fs::read(&log).unwrap();
        corrupt.splice(0..0, *b"not json\n"); // a corrupt INTERIOR line
        fs::write(&log, &corrupt).unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert_eq!(
            state.retract_source_preview("sake", "doc").unwrap(),
            (1, false),
            "a passage-store load failure degrades to no-passage, not an error"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// #678: `add_associations` itself, standalone — every other test
    /// in this file only calls it as setup and `.unwrap().unwrap()`s
    /// past the return value.
    #[test]
    fn add_associations_refuses_an_already_expired_deadline_and_otherwise_reports_its_own_applied_count()
     {
        let dir = scratch_dir("add-associations-contract");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();

        let already_expired = Deadline::after(std::time::Duration::ZERO);
        let refused = state.add_associations(
            "sake",
            vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("doc"))],
            already_expired,
        );
        assert!(
            matches!(refused, Err(AccessError::DeadlineExceeded)),
            "an already-expired deadline must refuse before any op runs: {refused:?}"
        );
        let untouched = state
            .read_context("sake", |context| {
                context.query(Some("蔵"), None, Some("高瀬")).is_empty()
            })
            .unwrap();
        assert!(untouched, "a refused batch must not have written anything");

        let applied = state
            .add_associations(
                "sake",
                vec![
                    assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("doc")),
                    assoc_op("蔵", "銘柄", "青嶺", 1.0, Some("doc")),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(applied, 2, "a fully-applied batch reports its own op count");

        let _ = fs::remove_dir_all(&dir);
    }

    /// #678: every early return in `clamp_out_of_range_paragraphs` is
    /// best-effort per its own doc — an unknown context, a deleted
    /// entry, and a passage-store load failure all leave `paragraph`
    /// as given rather than fail the write. Only a paragraph this
    /// function can positively prove out of range against a resident,
    /// healthy store gets cleared.
    #[test]
    fn clamp_out_of_range_paragraphs_only_clears_what_it_can_positively_prove_out_of_range() {
        fn op_with_paragraph(paragraph: u32) -> AssocOp {
            AssocOp {
                subject: "蔵".to_string(),
                label: "杜氏".to_string(),
                object: "高瀬".to_string(),
                weight: 1.0,
                source: Some("doc".to_string()),
                paragraph: Some(paragraph),
            }
        }

        // (a) Unknown context: `lookup` fails, the op passes through.
        let dir = scratch_dir("clamp-paragraphs");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let unchanged =
            state.clamp_out_of_range_paragraphs("no-such-context", vec![op_with_paragraph(99)]);
        assert_eq!(
            unchanged[0].paragraph,
            Some(99),
            "an unknown context must not clamp anything"
        );

        // (b) A deleted entry: `lookup` still finds it (the directory
        // entry survives tombstoning) but `read_unless_deleted` refuses.
        state.create("sake", ContextMeta::default()).unwrap();
        {
            let entry = state.lookup("sake").unwrap();
            let mut inner = entry.inner.write();
            state.tombstone_locked(&mut inner, &entry);
        }
        let unchanged = state.clamp_out_of_range_paragraphs("sake", vec![op_with_paragraph(99)]);
        assert_eq!(
            unchanged[0].paragraph,
            Some(99),
            "a deleted entry must not clamp anything"
        );
        let _ = fs::remove_dir_all(&dir);

        // (c) A passage-store load failure: same corrupt-log technique
        // as `a_failed_passage_load_is_quarantined_like_the_image`.
        let dir = scratch_dir("clamp-paragraphs-load-failure");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            state
                .store_passages(
                    "sake",
                    plain(BTreeMap::from([(
                        "doc".to_string(),
                        "段落1\n\n段落2".to_string(),
                    )])),
                )
                .unwrap()
                .unwrap();
            state.flush_dirty();
        }
        let log = dir.join("sake.passages.wal.jsonl");
        let healthy = fs::read(&log).unwrap();
        let mut corrupt = healthy.clone();
        corrupt.splice(0..0, *b"not json\n");
        fs::write(&log, &corrupt).unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let unchanged = state.clamp_out_of_range_paragraphs("sake", vec![op_with_paragraph(1)]);
        assert_eq!(
            unchanged[0].paragraph,
            Some(1),
            "a load failure must not clamp anything"
        );
        drop(state);

        // (d) A healthy, resident store: the in-range paragraph
        // survives untouched, only the out-of-range one is cleared.
        fs::write(&log, &healthy).unwrap();
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let clamped = state.clamp_out_of_range_paragraphs(
            "sake",
            vec![op_with_paragraph(1), op_with_paragraph(2)],
        );
        assert_eq!(
            clamped[0].paragraph,
            Some(1),
            "index 1 is the store's last paragraph, still in range"
        );
        assert_eq!(
            clamped[1].paragraph, None,
            "index 2 is past the store's two paragraphs"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
