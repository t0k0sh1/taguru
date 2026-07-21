use super::*;

impl AppState {
    /// One context's current revision counters, or `None` for an
    /// unknown or deleted context — what the group fingerprint hashes
    /// per member, without loading anything.
    pub fn context_revision(&self, name: &str) -> Option<ContextRevision> {
        let entry = self.lookup(name)?;
        let inner = entry.read_unless_deleted()?;
        Some(entry.revision_snapshot(&inner))
    }

    /// The routing directory: every context's name, description, policy,
    /// residency, and stats, in name order. For large registries, prefer
    /// [`AppState::directory_page`], which seeks a page in O(log n + k)
    /// instead of describing every entry on every call.
    pub fn directory(&self) -> Vec<DirectoryEntry> {
        self.snapshot()
            .into_iter()
            .filter_map(|(name, entry)| describe_entry(name, &entry))
            .collect()
    }

    /// One name-ordered page of the routing directory plus the
    /// cursor-independent total, seeked in O(log n + k) against the
    /// `BTreeMap`-backed registry — the paged sibling of
    /// [`AppState::directory`]. Cuts the page under the registry lock
    /// (cloning only `Arc` handles, same as [`AppState::snapshot`]) and
    /// describes the survivors after dropping it: a context's
    /// `Entry::inner` lock must never be taken while the registry lock
    /// is held, the same ordering `directory`/`lookup` already keep.
    /// A page can come back shorter than `limit` if a context in it is
    /// deleted in the instant between the seek and [`describe_entry`]
    /// reading its slot — the same race `directory` already tolerates.
    /// If EVERY entry in the seek window loses that race, re-seek past
    /// it instead of reporting an empty page: callers (the SDKs' `iter`
    /// helpers among them) treat an empty page as the end of the
    /// directory, so returning one while later entries still exist
    /// would truncate the walk. Each retry's `after` strictly advances,
    /// so this terminates in at most `total` iterations.
    pub fn directory_page(
        &self,
        after: Option<&str>,
        limit: usize,
    ) -> (usize, Vec<DirectoryEntry>) {
        use std::ops::Bound;

        let mut after = after.map(str::to_string);
        loop {
            let (total, slice) = {
                let registry = self.0.registry.read();
                let start = match &after {
                    Some(after) => Bound::Excluded(after.as_str()),
                    None => Bound::Unbounded,
                };
                let slice: Vec<(String, Arc<Entry>)> = registry
                    .range::<str, _>((start, Bound::Unbounded))
                    .take(limit)
                    .map(|(name, entry)| (name.clone(), Arc::clone(entry)))
                    .collect();
                (registry.len(), slice)
            };
            let Some(last_seeked) = slice.last().map(|(name, _)| name.clone()) else {
                return (total, Vec::new());
            };
            let page: Vec<DirectoryEntry> = slice
                .into_iter()
                .filter_map(|(name, entry)| describe_entry(name, &entry))
                .collect();
            if !page.is_empty() {
                return (total, page);
            }
            after = Some(last_seeked);
        }
    }

    /// One directory row by name, or `None` for an unknown context.
    pub fn directory_entry(&self, name: &str) -> Option<DirectoryEntry> {
        let entry = self.lookup(name)?;
        describe_entry(name.to_string(), &entry)
    }

    /// Whether a context exists, by registry membership. The
    /// cross-context search entrances vet their whole target list up
    /// front, so a mistyped name refuses before any context is
    /// searched; a context deleted between this check and its read is
    /// still caught by the read itself.
    pub fn context_exists(&self, name: &str) -> bool {
        self.lookup(name).is_some()
    }
}

impl AppState {
    /// Materializes everything one context's export stream renders
    /// from — graph, aliases, meta, passages — under a single fence.
    /// The graph half is read under `inner` (shared when hot,
    /// exclusive across a cold load), which every graph write also
    /// takes exclusively, so the associations and the passage
    /// snapshot cannot shear against a retraction or a batch apply.
    /// A concurrent passage store (which runs under the SHARED fence)
    /// can still land between the two — the passage text may be
    /// newer than the graph, never torn within one source.
    ///
    /// Cost note, like `compact_context`: the whole graph is walked and
    /// materialized into owned strings while the (shared) fence is held,
    /// so on a large context writers to THAT context wait out the
    /// materialization. It is a per-context stall, off the async runtime
    /// (`block_in_place` at the HTTP layer); a streaming, lock-light
    /// export is future work, not a v1 promise.
    pub fn export_context(
        &self,
        name: &str,
        deadline: Deadline,
    ) -> Result<crate::export::ExportSnapshot, AccessError> {
        let entry = self.lookup(name).ok_or(AccessError::NotFound)?;
        let stem = file_stem(name);
        // Fast path: already resident, shared lock (mirrors read_context).
        {
            let inner = entry.inner.read();
            match &inner.slot {
                Slot::Hot(context) => {
                    self.0.metrics.record_cache_hit();
                    let snapshot =
                        self.export_snapshot(&entry, &stem, &inner.meta, context, deadline);
                    drop(inner);
                    self.touch(&entry);
                    self.enforce_budget(name);
                    return snapshot;
                }
                Slot::Deleted => return Err(AccessError::NotFound),
                Slot::Cold => {}
            }
        }
        // Slow path: load under the exclusive lock, as read_context does.
        // The `?` skips touch/enforce_budget on a load failure, matching
        // read_context and compact_context — a repeatedly-failing export
        // must not keep bumping a broken entry's LRU recency.
        let snapshot = offload(|| {
            let mut inner = entry.lock_unless_deleted().ok_or(AccessError::NotFound)?;
            ensure_hot(
                &self.0.data_dir,
                name,
                &mut inner,
                &self.0.metrics,
                self.0.hydrator.as_deref(),
            )
            .map_err(AccessError::Load)?;
            self.recount_entry(&mut inner);
            let Slot::Hot(context) = &inner.slot else {
                unreachable!("ensure_hot leaves the slot hot");
            };
            self.export_snapshot(&entry, &stem, &inner.meta, context, deadline)
        })?;
        self.touch(&entry);
        self.enforce_budget(name);
        Ok(snapshot)
    }

    /// The materialization inside [`AppState::export_context`]'s fence.
    /// Lock order: the caller holds `inner`; `entry_passages` takes
    /// `passages` — the documented `inner` → `passages` order.
    ///
    /// `deadline` is checked once, before any of it — not inside
    /// `context.query_any(&[], &[], &[])` below, which collects every
    /// association up front (its all-wildcard fast path), so a deadline
    /// that is already tight when this is called cannot shorten that
    /// initial O(edges) collection (the same limitation documented on
    /// [`Context::compacted`]).
    fn export_snapshot(
        &self,
        entry: &Entry,
        stem: &str,
        meta: &ContextMeta,
        context: &Context,
        deadline: Deadline,
    ) -> Result<crate::export::ExportSnapshot, AccessError> {
        if deadline.expired() {
            return Err(AccessError::DeadlineExceeded);
        }
        let passages = self
            .entry_passages(entry, stem)
            .map_err(|error| AccessError::Load(format!("passage store: {error}")))?
            .snapshot();
        let owned = |pairs: Vec<(&str, &str)>| -> Vec<(String, String)> {
            pairs
                .into_iter()
                .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
                .collect()
        };
        Ok(crate::export::ExportSnapshot {
            meta: meta.clone(),
            associations: context.query_any(&[], &[], &[]),
            concept_aliases: owned(context.concept_aliases()),
            label_aliases: owned(context.label_aliases()),
            passages,
        })
    }

    /// Rebuilds one context's image without its dead weight — the
    /// append-only storage's accumulated retracted edges, unlinked
    /// attributions, and arena slack (see [`Context::compacted`]) —
    /// then persists the fresh image immediately. Runs under the
    /// context's exclusive lock for the rebuild: requests to THIS
    /// context wait; every other context is untouched. Crash-safe by
    /// construction: the fresh context carries the old WAL watermark,
    /// so a crash before the flush lands simply boots the old image
    /// and replays the same log — compaction lost, nothing corrupted.
    pub fn compact_context(
        &self,
        name: &str,
        deadline: Deadline,
    ) -> Result<CompactOutcome, AccessError> {
        let entry = self.lookup(name).ok_or(AccessError::NotFound)?;
        let outcome = offload(|| {
            let mut inner = entry.lock_unless_deleted().ok_or(AccessError::NotFound)?;
            ensure_hot(
                &self.0.data_dir,
                name,
                &mut inner,
                &self.0.metrics,
                self.0.hydrator.as_deref(),
            )
            .map_err(AccessError::Load)?;
            let Slot::Hot(context) = &inner.slot else {
                unreachable!("ensure_hot leaves the slot hot");
            };
            let bytes_before = context.footprint();
            let (mut fresh, stats) =
                context
                    .compacted(deadline)
                    .map_err(|failure| match failure {
                        CompactionError::Full(full) => {
                            AccessError::Load(format!("compaction refused: {full}"))
                        }
                        CompactionError::DeadlineExceeded => AccessError::DeadlineExceeded,
                    })?;
            // Config the image never carries, re-applied exactly as a
            // load would; the watermark keeps WAL replay monotonic.
            fresh.set_applied_seq(context.applied_seq());
            fresh.set_dice_floor(inner.meta.dice_floor);
            let bytes_after = fresh.footprint();
            inner.slot = Slot::Hot(Box::new(fresh));
            inner.image_generation += 1;
            let Slot::Hot(context) = &inner.slot else {
                unreachable!("just installed");
            };
            inner.stats = ContextStats::of(context);
            entry.dirty.store(true, Ordering::Relaxed);
            self.recount_entry(&mut inner);
            Ok(CompactOutcome {
                bytes_before,
                bytes_after,
                dead_edges: stats.dead_edges,
                aliases_dropped: stats.aliases_dropped,
            })
        })?;
        // Persist the shrunken image now (flush_entry takes its own
        // locks); a failure leaves the entry dirty for the next tick,
        // which is the flusher's ordinary retry story.
        self.flush_entry(name, &entry);
        self.touch(&entry);
        self.enforce_budget(name);
        Ok(outcome)
    }

    /// Every context whose dead ratio strictly exceeds
    /// `min_dead_ratio`, worst first — read from each entry's existing
    /// bookkeeping ([`Context::dead_ratio`] while hot, the cached
    /// sidecar snapshot's while cold); nothing is loaded or rebuilt
    /// just to be asked whether it qualifies. Shared by the manual
    /// maintenance sweep and the flusher's auto-compaction so the two
    /// can never disagree about what "needs compacting" means.
    fn compaction_candidates(&self, min_dead_ratio: f64) -> Vec<(String, f64)> {
        let mut candidates: Vec<(String, f64)> = self
            .snapshot()
            .into_iter()
            .filter_map(|(name, entry)| {
                let inner = entry.inner.read();
                let ratio = match &inner.slot {
                    Slot::Hot(context) => context.dead_ratio(),
                    Slot::Cold => inner.stats.dead_ratio(),
                    Slot::Deleted => return None,
                };
                // NaN (a malformed `min_dead_ratio`) makes every
                // comparison false, so the sweep simply selects nothing
                // rather than mis-selecting.
                (ratio > min_dead_ratio).then_some((name, ratio))
            })
            .collect();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        candidates
    }

    /// The ratio-triggered auto-compaction's selection half (issue
    /// #135): the worst-ratio context past the configured trigger that
    /// is not in `skip`, or `None` — always `None` while the feature
    /// is off. `skip` holds the contexts whose rebuild already blew
    /// the flusher's compaction budget: that failure cannot heal by
    /// retrying (the context only grows), so re-selecting one would
    /// burn a budget's worth of CPU every tick forever. The flusher
    /// asks this every pass and compacts at most one candidate, so the
    /// policy stays amortized like the passages store's own (the next
    /// tick picks up the next-worst); the caller takes the heavy-ops
    /// permit before acting on the answer, which is why selection and
    /// compaction are separate steps rather than one method holding a
    /// permit it didn't need when the fleet is clean — the common
    /// case.
    pub fn auto_compact_candidate(&self, skip: &HashSet<String>) -> Option<(String, f64)> {
        let min_dead_ratio = self.0.auto_compact?;
        self.compaction_candidates(min_dead_ratio)
            .into_iter()
            .find(|(name, _)| !skip.contains(name))
    }

    /// Server-wide sweep: every context whose live dead ratio strictly
    /// exceeds `min_dead_ratio` is rebuilt via [`Self::compact_context`],
    /// worst ratio first, so a deadline that cuts the sweep short still
    /// recovers the most it could. Candidates come from
    /// [`Self::compaction_candidates`] — bookkeeping reads only.
    /// Sequential by design: the caller
    /// (`POST /maintenance/compact`) has already drained ordinary
    /// traffic before calling this, so there is no concurrency to hide
    /// behind parallelism, and one context at a time caps the sweep's
    /// peak memory at a single context's footprint.
    pub fn run_maintenance_compaction(
        &self,
        min_dead_ratio: f64,
        deadline: Deadline,
    ) -> MaintenanceCompactionOutcome {
        let candidates = self.compaction_candidates(min_dead_ratio);
        let mut contexts = Vec::with_capacity(candidates.len());
        let mut deadline_exceeded = false;
        for (name, _) in candidates {
            if deadline.expired() {
                deadline_exceeded = true;
                break;
            }
            match self.compact_context(&name, deadline) {
                Ok(outcome) => contexts.push(MaintenanceCompactionEntry { name, outcome }),
                Err(AccessError::DeadlineExceeded) => {
                    deadline_exceeded = true;
                    break;
                }
                Err(error) => {
                    tracing::warn!(
                        context = %name,
                        ?error,
                        "maintenance sweep skipped a context"
                    );
                }
            }
        }
        MaintenanceCompactionOutcome {
            contexts,
            deadline_exceeded,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_proptest::{
        AliasInput, AssocInput, RetractionInput, config as proptest_config,
        json_roundtrip_f64_strategy, scenario_strategy,
    };
    use crate::registry::test_support::{assoc_op, scratch_dir};
    use proptest::prelude::*;

    /// The three revision counters move on exactly their own lane —
    /// graph writes, passage writes, config changes — and read-only
    /// traffic and no-op updates move none of them (#149).
    #[test]
    fn revision_counts_each_lane_independently_and_ignores_reads() {
        let dir = scratch_dir("revision-lanes");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        assert_eq!(
            state.context_revision("sake").unwrap(),
            ContextRevision::default(),
            "a fresh context starts at zeros"
        );

        state
            .add_associations(
                "sake",
                vec![
                    assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("a.md")),
                    assoc_op("蔵", "創業", "1832", 1.0, Some("a.md")),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        let after_graph = state.context_revision("sake").unwrap();
        assert_eq!(after_graph.graph, 2, "one bump per applied op");
        assert_eq!((after_graph.passages, after_graph.config), (0, 0));

        let mut passages = BTreeMap::new();
        passages.insert(
            "a.md".to_string(),
            crate::passages::PassageSubmission::plain("蔵は1832年創業。"),
        );
        state.store_passages("sake", passages).unwrap().unwrap();
        let after_passage = state.context_revision("sake").unwrap();
        assert_eq!(after_passage.passages, 1, "the passage log watermark");
        assert_eq!(
            after_passage.graph, 2,
            "a passage write moves no other lane"
        );

        state
            .update_meta("sake", None, None, None, Some(0.5))
            .unwrap()
            .unwrap();
        assert_eq!(
            state.context_revision("sake").unwrap().config,
            1,
            "a metadata change bumps config"
        );
        state
            .update_meta("sake", None, None, None, Some(0.5))
            .unwrap()
            .unwrap();
        assert_eq!(
            state.context_revision("sake").unwrap().config,
            1,
            "the same floor again changed nothing, so it bumps nothing"
        );

        state
            .read_context("sake", |context| context.association_count())
            .unwrap();
        assert_eq!(
            state.context_revision("sake").unwrap(),
            ContextRevision {
                graph: 2,
                passages: 1,
                config: 1
            },
            "reads move nothing"
        );

        // One retraction is a graph op AND a passage-log record: both
        // data lanes move, config stays.
        state.retract_source("sake", "a.md").unwrap();
        assert_eq!(
            state.context_revision("sake").unwrap(),
            ContextRevision {
                graph: 3,
                passages: 2,
                config: 1
            }
        );
        let _ = fs::remove_dir_all(dir);
    }

    /// The guarantee ladder in [`ContextRevision`]'s doc: a clean
    /// restart serves the exact counters while still cold, a crashed
    /// one serves the sidecar's lagging value until the first load
    /// catches each data lane up against its own log (#149).
    #[test]
    fn revision_is_stable_across_a_clean_restart_and_catches_up_after_a_crashed_one() {
        let dir = scratch_dir("revision-restart");
        let clean = {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("a.md"))],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            let mut passages = BTreeMap::new();
            passages.insert(
                "a.md".to_string(),
                crate::passages::PassageSubmission::plain("高瀬が杜氏。"),
            );
            state.store_passages("sake", passages).unwrap().unwrap();
            state
                .update_meta("sake", None, None, None, Some(0.4))
                .unwrap()
                .unwrap();
            // The graceful-shutdown pair: the final flush, then the sweep.
            state.flush_dirty();
            state.persist_usage();
            state.context_revision("sake").unwrap()
        };

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let entry = state.directory_entry("sake").unwrap();
        assert!(!entry.loaded, "the directory answer must not load anything");
        assert_eq!(entry.revision, clean, "clean restart: exact, while cold");

        // More writes on both data lanes, then a crash: no flush, no sweep.
        state
            .add_associations(
                "sake",
                vec![assoc_op("蔵", "創業", "1832", 1.0, Some("b.md"))],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        let mut passages = BTreeMap::new();
        passages.insert(
            "b.md".to_string(),
            crate::passages::PassageSubmission::plain("創業は1832年。"),
        );
        state.store_passages("sake", passages).unwrap().unwrap();
        let live = state.context_revision("sake").unwrap();
        assert_eq!(live.graph, clean.graph + 1);
        assert_eq!(live.passages, clean.passages + 1);
        drop(state);

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert_eq!(
            state.context_revision("sake").unwrap(),
            clean,
            "cold after a crash: the sidecar's lagging values"
        );
        // A graph load replays the WAL and floors the counter with its top.
        state
            .read_context("sake", |context| context.association_count())
            .unwrap();
        assert_eq!(state.context_revision("sake").unwrap().graph, live.graph);
        // A passage-store load replays its own log the same way.
        state
            .lookup_passages("sake", &["a.md".to_string()])
            .unwrap()
            .unwrap();
        assert_eq!(
            state.context_revision("sake").unwrap().passages,
            live.passages,
            "every search path loads before computing, so a cache fill never keys on the stale seed"
        );
        let _ = fs::remove_dir_all(dir);
    }

    /// With the WAL disabled `wal_seq` never moves — the revision has
    /// its own counter precisely so that configuration still reports
    /// changes, and a reload of the (empty) log must not walk it back
    /// (#149).
    #[test]
    fn wal_off_writes_still_advance_the_graph_revision() {
        let dir = scratch_dir("revision-wal-off");
        let wal_off = || BootOptions {
            wal_enabled: false,
            ..BootOptions::default()
        };
        {
            let state = AppState::boot_with(dir.clone(), usize::MAX, None, wal_off()).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, None)],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            assert_eq!(state.context_revision("sake").unwrap().graph, 1);
            state.flush_dirty();
        }
        let state = AppState::boot_with(dir.clone(), usize::MAX, None, wal_off()).unwrap();
        assert_eq!(
            state.context_revision("sake").unwrap().graph,
            1,
            "flushed WAL-off writes keep their count across a restart"
        );
        state
            .read_context("sake", |context| context.association_count())
            .unwrap();
        assert_eq!(
            state.context_revision("sake").unwrap().graph,
            1,
            "the stale replay top (there is no log) must not regress the counter"
        );
        let _ = fs::remove_dir_all(dir);
    }

    /// A rename carries the counters (same content, new name); a
    /// delete-recreate restarts them — the lineage reset callers must
    /// treat as invalidation (#149).
    #[test]
    fn revision_rides_a_rename_and_resets_on_recreate() {
        let dir = scratch_dir("revision-rename");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state
            .add_associations(
                "sake",
                vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, None)],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        let before = state.context_revision("sake").unwrap();
        state.rename_context("sake", "brewery").unwrap();
        assert_eq!(
            state.context_revision("brewery").unwrap(),
            before,
            "a rename is the same content under a new name"
        );
        state.delete("brewery").unwrap().unwrap();
        state.create("brewery", ContextMeta::default()).unwrap();
        assert_eq!(
            state.context_revision("brewery").unwrap(),
            ContextRevision::default(),
            "a recreate is a new lineage"
        );
        let _ = fs::remove_dir_all(dir);
    }

    /// Compaction sheds the dead weight retraction leaves behind,
    /// preserves every live fact, keeps the WAL watermark monotonic —
    /// a write after the compact replays correctly across a hard crash
    /// — and the shrunken image is what a restart boots.
    #[test]
    fn compaction_shrinks_the_image_and_stays_crash_safe() {
        let dir = scratch_dir("compact");
        let live_facts = |state: &AppState| -> Vec<(String, String, String, u64)> {
            let mut facts = state
                .read_context("sake", |context| {
                    context
                        .query_any(&[], &[], &[])
                        .into_iter()
                        .filter(|association| association.count > 0)
                        .map(|association| {
                            (
                                association.subject,
                                association.label,
                                association.object,
                                association.count,
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .map_err(|_| "read")
                .unwrap();
            facts.sort();
            facts
        };
        let before;
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .add_associations(
                    "sake",
                    vec![
                        assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md")),
                        assoc_op("蔵", "銘柄", "青嶺", 1.0, Some("keep.md")),
                        assoc_op("蔵", "廃止銘柄", "旧銘", 1.0, Some("gone.md")),
                    ],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            state.retract_source("sake", "gone.md").unwrap();
            before = live_facts(&state);

            let outcome = state
                .compact_context("sake", Deadline::unbounded())
                .unwrap();
            assert!(
                outcome.bytes_after < outcome.bytes_before,
                "{outcome:?} must shrink"
            );
            assert_eq!(outcome.dead_edges, 1, "{outcome:?}");
            assert_eq!(live_facts(&state), before, "live content must survive");

            // A write AFTER the compact must replay across a crash —
            // the fresh image carries the old watermark, so the WAL
            // sequence keeps counting from where it was.
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("蔵", "創業年", "1907年", 1.0, Some("keep.md"))],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            // Drop WITHOUT flushing: the crash.
        }
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let after = live_facts(&state);
        assert_eq!(after.len(), before.len() + 1, "{after:?}");
        assert!(
            after
                .iter()
                .any(|(_, label, object, _)| label == "創業年" && object == "1907年"),
            "the post-compact write must replay: {after:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    fn apply_generated_context(
        state: &AppState,
        assoc_ops: &[AssocInput],
        alias_ops: &[AliasInput],
        retractions: &[RetractionInput],
    ) {
        let ops = assoc_ops
            .iter()
            .map(|op| AssocOp {
                subject: op.subject.to_string(),
                label: op.label.to_string(),
                object: op.object.to_string(),
                weight: op.weight,
                source: op.source.map(str::to_string),
                paragraph: op.paragraph,
            })
            .collect();
        state
            .add_associations("generated", ops, Deadline::unbounded())
            .unwrap()
            .unwrap();

        for alias_op in alias_ops {
            let (concepts, labels) = match alias_op {
                AliasInput::Concept { alias, canonical } => (
                    BTreeMap::from([(alias.to_string(), canonical.to_string())]),
                    BTreeMap::new(),
                ),
                AliasInput::Label { alias, canonical } => (
                    BTreeMap::new(),
                    BTreeMap::from([(alias.to_string(), canonical.to_string())]),
                ),
            };
            let _ = state.add_aliases("generated", &concepts, &labels).unwrap();
        }

        for retraction in retractions {
            match retraction {
                RetractionInput::Source(source) => {
                    state.retract_source("generated", source).unwrap();
                }
                RetractionInput::Association {
                    subject,
                    label,
                    object,
                } => {
                    state
                        .retract_association("generated", subject, label, object)
                        .unwrap();
                }
            }
        }
    }

    proptest! {
        #![proptest_config(proptest_config())]

        /// The operator path installs the same canonical context as the
        /// library rebuild, flushes it immediately, and re-applies state
        /// that lives outside the graph image when it is loaded again.
        #[test]
        fn registry_compaction_flushes_and_reloads_the_canonical_image(
            (assoc_ops, alias_ops, retractions) in scenario_strategy(),
            dice_floor in prop::option::of(json_roundtrip_f64_strategy(-1.0f64..2.0)),
        ) {
            let dir = scratch_dir("compact-property");
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create(
                    "generated",
                    ContextMeta {
                        dice_floor,
                        ..ContextMeta::default()
                    },
                )
                .unwrap();
            apply_generated_context(&state, &assoc_ops, &alias_ops, &retractions);
            // Project the current WAL watermark into the source image first;
            // compaction must carry this known non-zero value forward.
            prop_assert_eq!(state.flush_dirty(), vec!["generated"]);

            let (expected_image, expected_seq, expected_floor, expected_stats) = state
                .read_context("generated", |context| {
                    let (mut canonical, stats) =
                        context.compacted(Deadline::unbounded()).unwrap();
                    canonical.set_applied_seq(context.applied_seq());
                    canonical.set_dice_floor(Some(context.dice_floor()));
                    (
                        canonical.to_bytes(),
                        context.applied_seq(),
                        context.dice_floor(),
                        stats,
                    )
                })
                .unwrap();

            let outcome = state
                .compact_context("generated", Deadline::unbounded())
                .unwrap();
            prop_assert_eq!(outcome.dead_edges, expected_stats.dead_edges);
            prop_assert_eq!(outcome.aliases_dropped, expected_stats.aliases_dropped);
            state
                .read_context("generated", |context| {
                    assert_eq!(context.to_bytes(), expected_image);
                    assert_eq!(context.applied_seq(), expected_seq);
                    assert_eq!(context.dice_floor(), expected_floor);
                })
                .unwrap();
            let disk_image = fs::read(image_path(&dir, &file_stem("generated"))).unwrap();
            prop_assert_eq!(&disk_image, &expected_image);

            let second = state
                .compact_context("generated", Deadline::unbounded())
                .unwrap();
            prop_assert_eq!(second.dead_edges, 0);
            prop_assert_eq!(second.aliases_dropped, 0);
            drop(state);

            let reloaded = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            reloaded
                .read_context("generated", |context| {
                    assert_eq!(context.to_bytes(), expected_image);
                    assert_eq!(context.applied_seq(), expected_seq);
                    assert_eq!(context.dice_floor(), expected_floor);
                })
                .unwrap();
            drop(reloaded);
            let _ = fs::remove_dir_all(dir);
        }

    }

    /// [`AppState::run_maintenance_compaction`] selects worst-ratio-first
    /// and drops anything at or under the floor — the ordering and the
    /// threshold `POST /maintenance/compact` promises callers.
    #[test]
    fn run_maintenance_compaction_orders_worst_ratio_first_and_applies_the_floor() {
        let dir = scratch_dir("maint-order");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

        state
            .create("clean", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "clean",
                vec![
                    assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md")),
                    assoc_op("蔵", "銘柄", "青嶺", 1.0, Some("keep.md")),
                    assoc_op("蔵", "産地", "灘", 1.0, Some("keep.md")),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();

        state
            .create("mild", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "mild",
                vec![
                    assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md")),
                    assoc_op("蔵", "銘柄", "青嶺", 1.0, Some("keep.md")),
                    assoc_op("蔵", "産地", "灘", 1.0, Some("keep.md")),
                    assoc_op("蔵", "廃止銘柄", "旧銘", 1.0, Some("gone.md")),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state.retract_source("mild", "gone.md").unwrap();

        state
            .create("rotten", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "rotten",
                vec![
                    assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md")),
                    assoc_op("蔵", "廃止銘柄", "旧銘", 1.0, Some("gone.md")),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state.retract_source("rotten", "gone.md").unwrap();

        let outcome = state.run_maintenance_compaction(0.0, Deadline::unbounded());
        assert!(!outcome.deadline_exceeded);
        let names: Vec<&str> = outcome
            .contexts
            .iter()
            .map(|entry| entry.name.as_str())
            .collect();
        assert_eq!(names, vec!["rotten", "mild"], "{names:?}");
        assert_eq!(outcome.contexts[0].outcome.dead_edges, 1);
        assert_eq!(outcome.contexts[1].outcome.dead_edges, 1);

        let _ = fs::remove_dir_all(dir);
    }

    /// A cold context's dead ratio comes from its persisted stats
    /// snapshot, not a load — [`AppState::run_maintenance_compaction`]'s
    /// whole point is picking candidates without paying for residency.
    #[test]
    fn run_maintenance_compaction_selects_a_cold_candidate_from_its_saved_stats() {
        let dir = scratch_dir("maint-cold");
        let state = AppState::boot(dir.clone(), 1, None).unwrap();

        state
            .create("rotten", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "rotten",
                vec![
                    assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md")),
                    assoc_op("蔵", "廃止銘柄", "旧銘", 1.0, Some("gone.md")),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state.retract_source("rotten", "gone.md").unwrap();

        state
            .create("other", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // Touching "other" evicts "rotten" to cold under the one-byte budget.
        state
            .read_context("other", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert!(
            !state.directory_entry("rotten").unwrap().loaded,
            "rotten must be cold before the sweep"
        );

        let outcome = state.run_maintenance_compaction(0.0, Deadline::unbounded());
        assert_eq!(outcome.contexts.len(), 1, "{outcome:?}");
        assert_eq!(outcome.contexts[0].name, "rotten");
        assert_eq!(outcome.contexts[0].outcome.dead_edges, 1);
        assert!(!outcome.deadline_exceeded);

        let _ = fs::remove_dir_all(dir);
    }

    /// [`AppState::auto_compact_candidate`] answers with the single
    /// worst context strictly past the trigger, ignores everything at
    /// or under it, steps to the next-worst when the worst is in the
    /// oversized skip set (without ever promoting anything from under
    /// the trigger), and — because compaction zeroes the winner's dead
    /// edges — finds nothing on the next ask: the amortized loop the
    /// flusher runs terminates by itself (issue #135).
    #[test]
    fn auto_compact_candidate_picks_the_worst_past_the_trigger_then_runs_dry() {
        let dir = scratch_dir("auto-candidate");
        // The deployment default: on, at DEFAULT_AUTO_COMPACT_RATIO.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

        state
            .create("mild", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "mild",
                vec![
                    assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md")),
                    assoc_op("蔵", "銘柄", "青嶺", 1.0, Some("keep.md")),
                    assoc_op("蔵", "産地", "灘", 1.0, Some("keep.md")),
                    assoc_op("蔵", "廃止銘柄", "旧銘", 1.0, Some("gone.md")),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state.retract_source("mild", "gone.md").unwrap();

        state
            .create("semi", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "semi",
                vec![
                    assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md")),
                    assoc_op("蔵", "銘柄", "青嶺", 1.0, Some("keep.md")),
                    assoc_op("蔵", "廃止銘柄", "旧銘", 1.0, Some("gone.md")),
                    assoc_op("蔵", "廃止蔵", "旧蔵", 1.0, Some("gone.md")),
                    assoc_op("蔵", "廃止杜氏", "旧氏", 1.0, Some("gone.md")),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state.retract_source("semi", "gone.md").unwrap();

        state
            .create("rotten", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "rotten",
                vec![
                    assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md")),
                    assoc_op("蔵", "廃止銘柄", "旧銘", 1.0, Some("gone.md")),
                    assoc_op("蔵", "廃止蔵", "旧蔵", 1.0, Some("gone.md")),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state.retract_source("rotten", "gone.md").unwrap();

        // mild sits at 1/4 — under the 0.5 default; semi at 3/5,
        // rotten at 2/3: both qualify, worst first.
        let (name, ratio) = state
            .auto_compact_candidate(&HashSet::new())
            .expect("rotten is past the trigger");
        assert_eq!(name, "rotten");
        assert!((ratio - 2.0 / 3.0).abs() < 1e-9, "{ratio}");

        // An oversized rebuild remembered by the flusher steps the
        // selection to the next-worst qualifier — the loop keeps
        // working around a context it cannot finish.
        let mut skip: HashSet<String> = ["rotten".to_string()].into();
        let (name, ratio) = state
            .auto_compact_candidate(&skip)
            .expect("semi is the next-worst qualifier");
        assert_eq!(name, "semi");
        assert!((ratio - 3.0 / 5.0).abs() < 1e-9, "{ratio}");

        // Skipping every qualifier must not promote mild from under
        // the trigger.
        skip.insert("semi".to_string());
        assert_eq!(state.auto_compact_candidate(&skip), None);

        state
            .compact_context("rotten", Deadline::unbounded())
            .unwrap();
        state
            .compact_context("semi", Deadline::unbounded())
            .unwrap();
        assert_eq!(
            state.auto_compact_candidate(&HashSet::new()),
            None,
            "the compacted winners must not re-trigger, and mild stays under"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// `auto_compact: None` (TAGURU_AUTO_COMPACT=0) is a hard off:
    /// selection answers nothing however rotten the fleet, so the
    /// flusher never takes a permit — or any lock — for the feature.
    #[test]
    fn auto_compact_candidate_is_none_while_the_feature_is_off() {
        let dir = scratch_dir("auto-off");
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            None,
            BootOptions {
                auto_compact: None,
                ..BootOptions::default()
            },
        )
        .unwrap();

        state
            .create("rotten", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "rotten",
                vec![
                    assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md")),
                    assoc_op("蔵", "廃止銘柄", "旧銘", 1.0, Some("gone.md")),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state.retract_source("rotten", "gone.md").unwrap();

        assert_eq!(state.auto_compact_candidate(&HashSet::new()), None);

        let _ = fs::remove_dir_all(dir);
    }

    /// A `Slot::Deleted` entry observed inside the sweep (the tombstone
    /// left for anyone holding the entry's `Arc` from before a concurrent
    /// `delete`) is skipped, not treated as a crash or a candidate.
    #[test]
    fn run_maintenance_compaction_skips_a_deleted_entry_without_panicking() {
        let dir = scratch_dir("maint-deleted");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("ghost", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "ghost",
                vec![assoc_op("蔵", "廃止銘柄", "旧銘", 1.0, Some("gone.md"))],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state.retract_source("ghost", "gone.md").unwrap();

        // Simulates the race `delete()` can open: still a member of the
        // registry map (so the sweep's snapshot picks it up) but its
        // slot already flipped to the tombstone, as if a concurrent
        // `delete()` had reached that half of its two-step teardown.
        let entry = state.lookup("ghost").expect("just created");
        entry.inner.write().slot = Slot::Deleted;

        let outcome = state.run_maintenance_compaction(0.0, Deadline::unbounded());
        assert!(outcome.contexts.is_empty(), "{outcome:?}");
        assert!(!outcome.deadline_exceeded);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dirty_contexts_survive_flush_and_cold_boot() {
        let dir = scratch_dir("flush");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .write_context("sake", |context| {
                    context
                        .associate("青嶺酒造", "代表銘柄", "青嶺", 1.0)
                        .unwrap();
                })
                .map_err(|_| "write")
                .unwrap();
            state.flush_dirty();
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        // Cold entries serve directory stats from the sidecar snapshot.
        let directory = state.directory();
        let sake = directory.iter().find(|e| e.name == "sake").unwrap();
        assert!(!sake.loaded);
        assert_eq!(sake.stats.associations, 1);

        let recalled = state
            .read_context("sake", |context| context.recall("青嶺").len())
            .map_err(|_| "reload")
            .unwrap();
        assert_eq!(recalled, 1);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn directory_page_seeks_by_name_with_a_cursor_independent_total() {
        let dir = scratch_dir("directory-page");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        for name in ["cherry", "apple", "banana"] {
            state.create(name, ContextMeta::default()).unwrap();
        }

        let (total, first) = state.directory_page(None, 2);
        assert_eq!(total, 3);
        let names: Vec<&str> = first.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, vec!["apple", "banana"]);

        let (total, second) = state.directory_page(Some("banana"), 2);
        assert_eq!(total, 3, "total stays constant across pages");
        let names: Vec<&str> = second.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, vec!["cherry"]);

        // Deleting a context drops it from the very next page.
        state.delete("apple").unwrap().unwrap();
        let (total, page) = state.directory_page(None, 10);
        assert_eq!(total, 2);
        let names: Vec<&str> = page.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, vec!["banana", "cherry"]);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn directory_page_skips_past_a_seek_window_tombstoned_before_describe() {
        let dir = scratch_dir("directory-page-tombstoned-window");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        for name in ["apple", "banana"] {
            state.create(name, ContextMeta::default()).unwrap();
        }

        // The race directory_page's doc comment describes: a handle
        // survives because the seek already cloned its Arc out of the
        // registry, but the slot it points at is tombstoned before
        // describe_entry reads it — the same effect AppState::delete
        // has on a handle the seek already cloned, without actually
        // removing "apple" from the BTreeMap so the seek still lands
        // on it.
        {
            let registry = state.0.registry.read();
            let entry = registry.get("apple").unwrap();
            entry.inner.write().slot = Slot::Deleted;
        }

        // limit 1 makes the whole seek window ("apple" alone) lose the
        // race — a false end of directory would stop right here and
        // never see "banana".
        let (total, page) = state.directory_page(None, 1);
        assert_eq!(total, 2);
        let names: Vec<&str> = page.iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, vec!["banana"]);

        let _ = fs::remove_dir_all(dir);
    }
}
