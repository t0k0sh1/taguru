use super::*;

impl AppState {
    /// Registers an empty context and persists it immediately, so its
    /// existence (and description) survives a crash from the moment the
    /// create call returns. A persistence failure fails the create.
    ///
    /// The registry lock is NOT held across the disk work (up to seven
    /// unlinks plus save_files' fsyncs — seconds on slow storage,
    /// behind which every operation on every context would otherwise
    /// stall). The name is reserved in `pending.creates` under the
    /// registry guard, the files are written unlocked, and the entry
    /// lands in a second critical section — the create twin of
    /// delete's `pending.deletes` choreography.
    pub fn create(&self, name: &str, meta: ContextMeta) -> Result<(), CreateError> {
        // An empty name has no file stem — it would persist as a bare
        // `.ctx` and disappear from the registry on the next restart.
        // Refuse it at the lowest boundary, so no entrance (import,
        // direct call) can conjure a self-erasing context.
        if name.is_empty() {
            return Err(CreateError::InvalidName);
        }
        {
            let registry = self.0.registry.read();
            // A name mid-delete is still taken: its delete has left the
            // registry but is still unlinking files, and a create landing
            // now would have its fresh generation destroyed by the tail of
            // that loop. A name mid-create is equally taken. A name that
            // is either end of an in-flight rename is taken too — `to`
            // because a create now would collide with the files the
            // rename is about to land there, `from` because the rename
            // has not yet torn its files down. The client sees the same
            // refusal as for a live name and simply retries after the
            // other call's response.
            if registry.contains_key(name) {
                return Err(CreateError::AlreadyExists);
            }
            // Checking the other two sets and reserving this one all
            // happen under the SAME lock, in one critical section — see
            // `PendingNames`'s doc for why that atomicity is what closes
            // the gap against a concurrent `rename_context` (the only
            // sibling that, like this call, holds only `registry.read()`
            // for its own check-then-reserve).
            let mut pending = self.0.pending.lock();
            if pending.deletes.contains(name)
                || pending.renames.contains(name)
                || !pending.creates.insert(name.to_string())
            {
                return Err(CreateError::AlreadyExists);
            }
        }
        let created = self.create_files(name, &meta);
        // Success or failure, the reservation leaves in the same
        // critical section that (on success) makes the entry visible.
        let mut registry = self.0.registry.write();
        let outcome = created.map(|(stats, usage, context)| {
            registry.insert(
                name.to_string(),
                Arc::new(Entry::new(
                    meta,
                    stats,
                    Slot::Hot(Box::new(context)),
                    0,
                    0,
                    usage,
                    ContextRevision::default(),
                )),
            );
        });
        self.0.pending.lock().creates.remove(name);
        outcome
    }

    /// The disk half of [`AppState::create`], run WITHOUT the registry
    /// lock — the `pending.creates` reservation is what keeps the name
    /// taken meanwhile.
    ///
    /// A name can be reused after a delete, and a delete that failed
    /// partway (the name is unregistered first) or a half-restored
    /// backup leaves the old generation's files behind. Nothing may
    /// bleed into the new context — a stale WAL would even replay
    /// the old generation's acknowledged writes into the fresh image
    /// on its next cold load. Clear the slate — the OLD IMAGE INCLUDED —
    /// before writing the new one: `save_files` lands the image last, so
    /// removing the old image up front means a crash anywhere before the
    /// new image commits leaves NO image at all. Nothing registers (the
    /// scan keys on `.ctx`), the next attempt clears again, and the old
    /// generation's data can never resurface under the new create's
    /// metadata. Durability of the unlinks rides on save_files'
    /// parent-directory fsync just below. A leftover that cannot be
    /// removed fails the create — registering on top of it would hand out
    /// a haunted context.
    fn create_files(
        &self,
        name: &str,
        meta: &ContextMeta,
    ) -> Result<(ContextStats, ContextUsage, Context), CreateError> {
        let stem = file_stem(name);
        for stale in [
            image_path(&self.0.data_dir, &stem),
            wal_path(&self.0.data_dir, &stem),
            sources_path(&self.0.data_dir, &stem),
            passages_path(&self.0.data_dir, &stem),
            passages_wal_path(&self.0.data_dir, &stem),
            pvectors_path(&self.0.data_dir, &stem),
            bm25_path(&self.0.data_dir, &stem),
            vectors_path(&self.0.data_dir, &stem),
            // A leftover marker from an earlier delete that could not
            // finish MUST go before this new generation of files
            // lands — otherwise the next boot's resume-sweep sees the
            // marker and deletes the context we are creating right now.
            deleted_marker_path(&self.0.data_dir, &stem),
            // The same hazard for a rename that half-finished with THIS
            // name as its SOURCE: its `.renaming` marker sits at this
            // stem, and boot's resume-sweep would otherwise move the
            // generation we are about to write onto the rename's
            // destination stem, losing it silently.
            renaming_marker_path(&self.0.data_dir, &stem),
        ] {
            if let Err(error) = remove_persisted_file(&stale)
                && error.kind() != io::ErrorKind::NotFound
            {
                return Err(CreateError::Io(error));
            }
        }
        // A rename that half-finished with THIS name as its DESTINATION
        // left its marker under the SOURCE's stem — a stem we cannot
        // derive from `name`. Boot's resume-sweep would move that source
        // family onto the generation we are about to write, erasing it.
        // Scan for any marker that names us as `to` and drop it; a fresh
        // create abandons a stuck rename either way.
        for stale in rename_markers_targeting(&self.0.data_dir, name, "renaming") {
            if let Err(error) = remove_persisted_file(&stale)
                && error.kind() != io::ErrorKind::NotFound
            {
                return Err(CreateError::Io(error));
            }
        }
        // Stale import markers are part of the same earlier generation:
        // left beside the new files, boot would report the fresh
        // context as carrying a torn import it never ran.
        for stale in import_marker_paths(&self.0.data_dir, &stem) {
            if let Err(error) = remove_persisted_file(&stale)
                && error.kind() != io::ErrorKind::NotFound
            {
                return Err(CreateError::Io(error));
            }
        }
        let mut context = Context::default();
        context.set_dice_floor(meta.dice_floor);
        let stats = ContextStats::of(&context);
        let usage = ContextUsage::default();
        // A fresh context starts its revision at zeros — which also
        // means a delete-recreate of the same name RESTARTS the
        // counters; a cache keyed on them must treat that as a new
        // lineage (see ContextRevision's doc).
        save_files(
            &self.0.data_dir,
            name,
            meta,
            &stats,
            &usage,
            ContextRevision::default(),
            &context,
        )
        .map_err(CreateError::Io)?;
        Ok((stats, usage, context))
    }

    /// Removes a context from the registry and deletes its files. The
    /// entry's lock is taken after the removal — waiting out any
    /// in-flight operation — and the slot becomes a tombstone under
    /// it: a flusher, evictor, or writer whose handle predates the
    /// removal finds [`Slot::Deleted`] when it finally locks, and
    /// backs off instead of recreating the files. Any unflushed writes
    /// are discarded — deletion destroys the context.
    ///
    /// The name enters `pending.deletes` in the same critical section
    /// that unregisters it and leaves only after the unlink loop: to a
    /// concurrent create() the name stays taken for the delete's whole
    /// run, so no new generation of files can appear under the tail of
    /// this one's removals.
    pub fn delete(&self, name: &str) -> Option<io::Result<()>> {
        let entry = {
            let mut registry = self.0.registry.write();
            if !registry.contains_key(name) {
                return None;
            }
            // A name mid-rename is refused rather than torn down: its
            // marker durably promises a move-then-membership-rewrite,
            // and a delete winning the race here would either destroy
            // the files the rename is about to move (as `from`) or the
            // files it just landed (as `to`), leaving the marker to
            // resume a rename with nothing left to finish at boot.
            // Reported through the same `Option<io::Result<()>>` a
            // live name already uses — the caller sees a name that
            // exists but cannot be deleted right now, not "no such
            // context".
            if self.0.pending.lock().renames.contains(name) {
                return Some(Err(io::Error::other(format!(
                    "context '{name}' is mid-rename; retry after it completes"
                ))));
            }
            let entry = registry.remove(name)?;
            self.0.pending.lock().deletes.insert(name.to_string());
            entry
        };
        let mut in_flight = entry.inner.write();
        self.tombstone_locked(&mut in_flight, &entry);
        // The rest of this function is disk I/O (marker, group sweep,
        // unlinks) guarded by `pending.deletes`, not by `inner` — hold
        // it no longer than the in-memory teardown above needs.
        drop(in_flight);
        let stem = file_stem(name);
        // A lazy bucket boot: the bucket's copy of this family must
        // not re-materialize after the unlinks below — veto waits out
        // any in-flight hydration so the two cannot interleave.
        // Nothing needs hydrating FIRST: files that never became local
        // were never shipped into this generation, and the manifest
        // gate (`Hydrator::drained`) keeps this generation
        // un-restorable until every family settles one way or the
        // other, so the deleted family cannot resurrect from either
        // generation.
        if let Some(hydrator) = &self.0.hydrator {
            hydrator.veto(&stem);
        }
        // The durable half of the acknowledgment: while this marker
        // exists, boot resumes the unlinks — so a partial failure here
        // (a held handle, a flaky mount) can leak bytes only until the
        // next start, and a surviving `.ctx` can never resurrect a
        // context the API reported gone. Written before the first
        // unlink; removed only after the last one succeeds.
        let marker = deleted_marker_path(&self.0.data_dir, &stem);
        if let Err(error) = write_atomic(&marker, b"") {
            tracing::warn!(context = %name, %error, "deletion marker not persisted; a partial delete would not resume at boot");
        }
        // Membership must not outlive the member: drop the name from
        // every group now, before the unlink loop's disk time. Best
        // effort — the delete's own durability rides on the marker
        // alone, and a sweep that could not persist is healed by the
        // next boot's reconciliation.
        self.sweep_context_from_groups(name);
        let mut outcome = Ok(());
        for file in context_files(&stem) {
            if let Err(error) = remove_persisted_file(self.0.data_dir.join(file))
                && error.kind() != io::ErrorKind::NotFound
            {
                outcome = Err(error);
            }
        }
        // Import markers go with the family: deletion makes any
        // half-applied batch moot, and a survivor would have boot
        // report a tear in a context that no longer exists. Same
        // failure handling as the fixed files — a miss keeps the
        // `.deleted` marker, and boot finishes the job.
        for path in import_marker_paths(&self.0.data_dir, &stem) {
            if let Err(error) = remove_persisted_file(&path)
                && error.kind() != io::ErrorKind::NotFound
            {
                outcome = Err(error);
            }
        }
        if outcome.is_ok() {
            let _ = remove_persisted_file(&marker);
        }
        self.0.pending.lock().deletes.remove(name);
        Some(outcome)
    }

    /// Renames a context: its whole file family moves under the new
    /// name and group membership follows, while the OLD name becomes a
    /// tombstone exactly as `delete` leaves one — so a flusher's or
    /// evictor's handle cloned before the rename backs off instead of
    /// recreating files a name no longer owns.
    ///
    /// Unlike `delete`, a rename must not discard unflushed writes: the
    /// entry's whole current state is drained to disk under the OLD
    /// name, under one lock, before the tombstone lands (see
    /// `drain_entry_for_rename`) — so no racing write can land in the
    /// gap between "durably saved" and "this entry stops accepting
    /// writes" and be silently lost the way `delete` allows.
    ///
    /// The marker (`renaming_marker_path`) is written and durable
    /// BEFORE anything else moves, and only removed after the group
    /// membership rewrite lands — stricter than `delete`'s best-effort
    /// marker, because a rename whose files moved but whose group
    /// membership rewrite did not would otherwise have boot's
    /// `reconcile_groups` see the old name as a dangling reference and
    /// silently drop it, rather than resuming the rewrite.
    pub fn rename_context(&self, from: &str, to: &str) -> Result<(), RenameContextError> {
        if to.is_empty() {
            return Err(RenameContextError::InvalidName);
        }
        if from == to {
            return Ok(());
        }
        let entry = {
            let registry = self.0.registry.read();
            let Some(entry) = registry.get(from) else {
                return Err(RenameContextError::NotFound);
            };
            if registry.contains_key(to) {
                return Err(RenameContextError::AlreadyExists);
            }
            // Checking all three sets and reserving both names in
            // `renames` all happen under the SAME lock, in one critical
            // section — see `PendingNames`'s doc for why that atomicity
            // is what closes the gap against a concurrent `create` (the
            // only sibling that, like this call, holds only
            // `registry.read()` for its own check-then-reserve).
            let mut pending = self.0.pending.lock();
            if pending.deletes.contains(from)
                || pending.deletes.contains(to)
                || pending.creates.contains(to)
                || pending.renames.contains(from)
                || pending.renames.contains(to)
            {
                return Err(RenameContextError::Busy);
            }
            pending.renames.insert(from.to_string());
            pending.renames.insert(to.to_string());
            Arc::clone(entry)
        };
        match self.rename_context_locked(from, to, &entry) {
            RenameOutcome::Ok => {
                let mut pending = self.0.pending.lock();
                pending.renames.remove(from);
                pending.renames.remove(to);
                Ok(())
            }
            // Rolled back before the point of no return: the registry
            // and the marker are both back to their pre-call state, so
            // both names are genuinely free again.
            RenameOutcome::RolledBack(error) => {
                let mut pending = self.0.pending.lock();
                pending.renames.remove(from);
                pending.renames.remove(to);
                Err(error)
            }
            // Failed AT or AFTER the point of no return: `from` is
            // already gone from the registry, but its files (and the
            // durable `.renaming` marker) may still be sitting there
            // half-moved. `from` MUST stay reserved — releasing it
            // would let a client's create(from) sweep away the marker
            // and the old generation's files as ordinary "stale
            // leftovers" (see create_files), destroying them beyond any
            // recovery. Only a boot resume-sweep can finish or roll
            // this back, so the reservation outlives this call; `to`
            // was never touched and is safe to free now.
            RenameOutcome::Stuck(error) => {
                tracing::error!(
                    from = %from, to = %to, ?error,
                    "context rename failed after the point of no return; the \
                     source name stays reserved until the next restart resumes \
                     it from the .renaming marker"
                );
                self.0.pending.lock().renames.remove(to);
                Err(error)
            }
        }
    }

    /// The disk-and-registry half of [`AppState::rename_context`], run
    /// with `from` and `to` both reserved in `pending.renames` — see
    /// that function's doc for why the marker is strict rather than
    /// best-effort.
    ///
    /// The return type spells out what the caller may safely release on
    /// failure: [`RenameOutcome::RolledBack`] means the attempt never
    /// passed the point of no return (the registry still lists `from`,
    /// and any marker written was cleaned up), so both names are free
    /// again. [`RenameOutcome::Stuck`] means it failed after `from` was
    /// already removed from the registry — the marker survives on disk
    /// and only a boot resume-sweep (or a successful retry) can resolve
    /// it, so `from` must stay reserved in the meantime.
    fn rename_context_locked(&self, from: &str, to: &str, entry: &Arc<Entry>) -> RenameOutcome {
        let from_stem = file_stem(from);
        let to_stem = file_stem(to);
        // A lazy bucket boot: the family must be LOCAL before it can
        // move — the pivot-based move treats a missing source file as
        // already-moved, which for an un-hydrated family would "move"
        // nothing and leave the bucket copy to resurrect under the old
        // name. Hydrate first, then veto re-materialization of the
        // stem being vacated.
        if let Some(hydrator) = &self.0.hydrator {
            if let Err(error) = hydrator.ensure_context(&from_stem) {
                return RenameOutcome::RolledBack(RenameContextError::Io(error));
            }
            hydrator.veto(&from_stem);
        }
        let marker = renaming_marker_path(&self.0.data_dir, &from_stem);
        if let Err(error) = write_rename_marker(&marker, from, to) {
            return RenameOutcome::RolledBack(RenameContextError::Io(error));
        }
        if let Err(error) = self.drain_entry_for_rename(from, entry) {
            let _ = fs::remove_file(&marker);
            return RenameOutcome::RolledBack(RenameContextError::Io(error));
        }
        self.0.registry.write().remove(from);
        // POINT OF NO RETURN: memory already reflects the rename (the
        // tombstone under `from`). Every failure from here on is
        // reported as `Stuck` — see this function's doc — so the only
        // way back is finishing the move and the membership rewrite, at
        // boot if not now.
        //
        // `entry`'s usage counters stay reachable via `note_read`/
        // `note_write`'s lock-free `lookup(from)` for as long as `from`
        // sits in the registry — right up to the `remove` just above.
        // `drain_entry_for_rename` snapshotted usage earlier (to have
        // something to hand `save_files` while `from` was still Hot, or
        // nothing at all if it was already Cold), so any read/write
        // counted after that snapshot — or ever, in the Cold case — is
        // invisible to the sidecar `read_meta_file` reads back below.
        // A second snapshot taken here, once `from` can no longer be
        // found by name, cannot miss anything a same-named lookup could
        // still land: the same monotonic counters only grow between the
        // two reads, so folding it in by field-wise max recovers the
        // count without holding any lock longer than today.
        let final_usage = entry.usage.snapshot();
        if let Err(error) = move_context_files(&self.0.data_dir, &from_stem, &to_stem) {
            return RenameOutcome::Stuck(RenameContextError::Io(error));
        }
        let MetaFile {
            meta,
            stats,
            usage,
            revision,
        } = read_meta_file(&self.0.data_dir, &to_stem);
        let usage = ContextUsage {
            reads: usage.reads.max(final_usage.reads),
            empty_reads: usage.empty_reads.max(final_usage.empty_reads),
            writes: usage.writes.max(final_usage.writes),
            last_read_epoch: usage.last_read_epoch.max(final_usage.last_read_epoch),
            last_write_epoch: usage.last_write_epoch.max(final_usage.last_write_epoch),
        };
        let pinned = meta.pinned;
        let wal_bytes = fs::metadata(wal_path(&self.0.data_dir, &to_stem))
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let passages_wal_bytes = fs::metadata(passages_wal_path(&self.0.data_dir, &to_stem))
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        // The revision moves with the sidecar: a rename is the same
        // content under a new name, so the counters carry over intact
        // (the group fingerprint still changes — the member NAME is
        // part of its hash).
        let new_entry = Arc::new(Entry::new(
            meta,
            stats,
            Slot::Cold,
            wal_bytes,
            passages_wal_bytes,
            usage,
            revision,
        ));
        self.0
            .registry
            .write()
            .insert(to.to_string(), Arc::clone(&new_entry));
        if pinned {
            let mut inner = new_entry.inner.write();
            match ensure_hot(
                &self.0.data_dir,
                to,
                &mut inner,
                &self.0.metrics,
                self.0.hydrator.as_deref(),
            ) {
                Ok(()) => self.recount_entry(&mut inner),
                Err(error) => {
                    tracing::warn!(context = %to, %error, "renamed context not preloaded; it stays cold until first use");
                }
            }
        }
        {
            let mut groups = self.0.groups.write();
            rename_in_membership(&self.0.data_dir, &mut groups, from, to, |record| {
                &mut record.contexts
            });
        }
        let _ = fs::remove_file(&marker);
        RenameOutcome::Ok
    }

    /// Writes an entry's whole current state to disk under `name` —
    /// its image (if Hot), sidecar, and stats — then tombstones the
    /// slot, all under one lock: no write racing the rename can land
    /// in the gap between "durably saved" and "this entry stops
    /// accepting writes" and be silently discarded. `delete`'s
    /// in-memory teardown discards unflushed writes on purpose; a
    /// rename must carry them to the new name instead — that is the
    /// one difference from `delete`'s teardown below.
    ///
    /// Derived indexes (passages, BM25, paragraph vectors) are cleared
    /// resident-only, exactly as `delete` clears them: their sidecars
    /// already hold their own last-saved state on disk and move with
    /// the rest of the file family, so at most a not-yet-persisted
    /// refresh is lost — a rename does not owe them the graph's
    /// durability guarantee.
    fn drain_entry_for_rename(&self, name: &str, entry: &Entry) -> io::Result<()> {
        let mut inner = entry.inner.write();
        // Read everything `save_files` and the watermark need before
        // borrowing `inner.slot` mutably below — `EntryInner` sits
        // behind a lock guard, so the borrow checker cannot see the
        // two borrows as disjoint fields the way it would on a bare
        // struct.
        let watermark = inner.wal_seq.saturating_sub(1);
        let meta = inner.meta.clone();
        let usage = entry.usage.snapshot();
        let revision = entry.revision_snapshot(&inner);
        if let Slot::Hot(context) = &mut inner.slot {
            // `ensure_hot`'s replay only applies WAL entries past
            // `applied_seq`, so baking in this watermark before saving
            // the image means the log — which rides along unmodified
            // under the new name — replays as a no-op once the file
            // family moves.
            context.set_applied_seq(watermark);
            let stats = ContextStats::of(context);
            save_files(
                &self.0.data_dir,
                name,
                &meta,
                &stats,
                &usage,
                revision,
                context,
            )?;
            inner.stats = stats;
        }
        self.tombstone_locked(&mut inner, entry);
        entry.usage_dirty.store(false, Ordering::Relaxed);
        drop(inner);
        Ok(())
    }
}

impl AppState {
    /// Updates the description and/or pin flag, persisting the sidecar
    /// immediately. Pinning loads the context now (pinned means
    /// resident); unpinning subjects it to the cache budget again.
    pub fn update_meta(
        &self,
        name: &str,
        description: Option<String>,
        pinned: Option<bool>,
        dice_floor: Option<f64>,
        semantic_floor: Option<f32>,
    ) -> Option<io::Result<ContextMeta>> {
        let entry = self.lookup(name)?;
        let outcome = {
            // A `None` means a delete won the lock first: don't
            // recreate the sidecar it just removed.
            let mut guard = entry.lock_unless_deleted()?;
            let inner = &mut *guard;
            // Saved so a load or persist failure below can restore the
            // pre-call state — without it, memory would hold fields
            // that never reached the sidecar, and a later, unrelated
            // successful update would persist them as a side effect.
            let previous = inner.meta.clone();
            if let Some(description) = description {
                inner.meta.description = description;
            }
            if let Some(pinned) = pinned {
                inner.meta.pinned = pinned;
            }
            if let Some(floor) = dice_floor {
                inner.meta.dice_floor = Some(floor.clamp(0.0, 1.0));
                // A loaded context picks the new floor up immediately;
                // a cold one gets it on its next load.
                if let Slot::Hot(context) = &mut inner.slot {
                    context.set_dice_floor(inner.meta.dice_floor);
                }
            }
            if let Some(floor) = semantic_floor {
                // Read at query time from the meta; nothing to push into
                // the loaded context.
                inner.meta.semantic_floor = Some(floor.clamp(0.0, 1.0));
            }
            if inner.meta.pinned
                && let Err(error) = ensure_hot(
                    &self.0.data_dir,
                    name,
                    inner,
                    &self.0.metrics,
                    self.0.hydrator.as_deref(),
                )
            {
                rollback_meta(inner, previous);
                self.recount_entry(inner);
                return Some(Err(io::Error::other(error)));
            }
            // A pin toggle moves the entry into or out of the budget's
            // world; the estimate must follow.
            self.recount_entry(inner);
            // Bump-and-persist atomically: the config revision rides
            // the same sidecar write as the change it tracks, and both
            // roll back together below — so a served bump always means
            // the new meta is durable. A PATCH that changed nothing
            // bumps nothing: idempotent updates must not churn caches.
            let changed = inner.meta != previous;
            if changed {
                inner.config_revision += 1;
            }
            let result = write_meta(
                &self.0.data_dir,
                &file_stem(name),
                &inner.meta,
                &inner.stats,
                &entry.usage.snapshot(),
                entry.revision_snapshot(inner),
            )
            .map(|()| inner.meta.clone());
            if result.is_err() {
                if changed {
                    inner.config_revision -= 1;
                }
                rollback_meta(inner, previous);
                self.recount_entry(inner);
            }
            result
        };
        self.enforce_budget(name);
        Some(outcome)
    }
}

/// Restores `inner.meta` to `previous` after a load or persist failure
/// partway through `update_meta`. Also un-applies the floor from any
/// already-loaded context, matching the one place `update_meta` pushes
/// a field straight into the hot context instead of just the sidecar.
fn rollback_meta(inner: &mut EntryInner, previous: ContextMeta) {
    if let Slot::Hot(context) = &mut inner.slot {
        context.set_dice_floor(previous.dice_floor);
    }
    inner.meta = previous;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::paths::RenameMarker;
    use crate::registry::test_support::{assoc_op, scratch_dir};

    /// An empty context name is refused at the registry boundary — the
    /// last guard against a bare `.ctx` file that `scan_data_dir` (which
    /// keys on the file stem) would never rediscover, silently orphaning
    /// every write to it. Parse and API refuse it earlier; this locks
    /// the floor beneath them.
    #[test]
    fn an_empty_context_name_is_refused_by_create() {
        let dir = scratch_dir("empty-name");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(matches!(
            state.create("", ContextMeta::default()),
            Err(CreateError::InvalidName)
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    /// Every stage/commit/unlink position in context deletion either
    /// finishes immediately or leaves enough durable state for boot to
    /// finish it. The first index beyond the operation proves the sweep
    /// did not merely sample a few hand-picked failures.
    #[test]
    fn every_context_delete_persistence_failure_recovers_at_boot() {
        let mut exhausted = false;
        for failure in 0..64 {
            let dir = scratch_dir(&format!("delete-fault-{failure}"));
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
            state.flush_dirty();
            state
                .create_group(
                    "breweries",
                    String::new(),
                    BTreeSet::from(["sake".to_string()]),
                    BTreeSet::new(),
                )
                .unwrap();

            fail_persistence_ops_after(failure);
            let outcome = state.delete("sake").unwrap();
            let past_end = clear_persistence_fault();
            drop(state);

            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            assert!(
                state.directory_entry("sake").is_none(),
                "failure at persistence step {failure} resurrected the context: {outcome:?}"
            );
            assert!(
                state.group("breweries").unwrap().contexts.is_empty(),
                "boot did not reconcile group membership at step {failure}"
            );
            assert!(
                !deleted_marker_path(&dir, "sake").exists(),
                "boot did not finish the marker at step {failure}"
            );
            drop(state);
            let _ = fs::remove_dir_all(&dir);

            if past_end {
                assert!(outcome.is_ok());
                exhausted = true;
                break;
            }
        }
        assert!(exhausted, "context deletion exceeded the sweep bound");
    }

    /// The dangerous interleaving: a delete leaves a marker behind
    /// (partial failure), the SAME running server recreates the
    /// context, and a later restart must NOT let the stale marker
    /// destroy the freshly created files. create() clears the marker.
    #[test]
    fn recreating_a_context_clears_a_stale_deletion_marker() {
        let dir = scratch_dir("deleted-recreate");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state.delete("sake");
            // Simulate the failure mode delete() cannot fully guard: its
            // unlink loop errored before removing the marker, so the
            // marker survives on disk while the name is free again.
            fs::write(deleted_marker_path(&dir, "sake"), b"").unwrap();
            // The same server recreates the context; create() must clear
            // that stale marker so the next boot does not resume it.
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "recreate")
                .unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("a.md"))],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            state.flush_dirty();
        }
        assert!(
            !dir.join("sake.deleted").exists(),
            "recreate must clear the stale marker"
        );
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(
            state.directory_entry("sake").is_some(),
            "the recreated context must survive the restart"
        );
        let count = state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(count, 1, "its data must be intact");
        let _ = fs::remove_dir_all(&dir);
    }

    /// `.deleted`'s recreate rule, for import markers: a marker the
    /// delete sweep could not remove must not survive into a freshly
    /// created context of the same name — boot would report the new
    /// generation as carrying a tear it never ran.
    #[test]
    fn creating_a_context_clears_stale_import_markers() {
        let dir = scratch_dir("import-marker-recreate");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state.delete("sake").unwrap().unwrap();
        // The failure delete() cannot fully guard: its marker sweep
        // missed one (crash, held handle), so the file outlives the
        // name.
        fs::write(
            import_marker_path(&dir, "sake", "doc-1"),
            b"{\"context\":\"sake\",\"source\":\"doc-1\"}",
        )
        .unwrap();

        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "recreate")
            .unwrap();
        assert!(
            import_marker_paths(&dir, "sake").is_empty(),
            "create clears the earlier generation's markers"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn create_does_not_inherit_files_left_by_an_earlier_generation() {
        let dir = scratch_dir("create-clean-slate");
        fs::create_dir_all(&dir).unwrap();
        let stem = file_stem("sake");
        // Litter an earlier generation can leave when its delete fails
        // partway (the name is unregistered first) or when files are
        // restored by hand: an acknowledged-write log, passages,
        // vectors — but no image, so nothing registers at boot.
        wal::append_batch(
            &wal_path(&dir, &stem),
            1,
            &[WalOp::Associate(assoc_op(
                "幽霊",
                "正体",
                "枯れ尾花",
                1.0,
                None,
            ))],
        )
        .unwrap();
        fs::write(sources_path(&dir, &stem), br#"{"ghost":"old passage"}"#).unwrap();
        fs::write(vectors_path(&dir, &stem), b"stale").unwrap();
        wal::append_batch(
            &passages_wal_path(&dir, &stem),
            1,
            &[crate::passages::PassageOp::Store {
                source: "ghost".to_string(),
                text: "前世代の本文".to_string(),
                questions: Vec::new(),
                sections: Vec::new(),
                stored_at: None,
                date: None,
                tags: Vec::new(),
            }],
        )
        .unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert_eq!(state.context_count(), 0, "no image, nothing registers");
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        assert!(
            !sources_path(&dir, &stem).exists(),
            "stale passages survived the create"
        );
        assert!(
            !passages_wal_path(&dir, &stem).exists(),
            "the old generation's passage log survived the create"
        );
        assert!(
            !vectors_path(&dir, &stem).exists(),
            "stale vectors survived the create"
        );
        drop(state);

        // The reboot is where inheritance would bite: a cold load
        // replays whatever the WAL holds above the fresh image's
        // watermark 0.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let recalled = state
            .read_context("sake", |context| context.recall("幽霊"))
            .map_err(|_| "read")
            .unwrap();
        assert!(
            recalled.is_empty(),
            "the old generation's WAL replayed into the new context"
        );
        assert!(
            state.passage_sources("sake").unwrap().unwrap().is_empty(),
            "the old generation's passage log replayed into the new context"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    #[cfg(unix)]
    fn a_failed_persist_does_not_leave_the_failed_change_in_memory() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("meta-rollback");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        // A clean update lands on disk.
        let meta = state
            .update_meta("sake", Some("A".to_string()), None, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(meta.description, "A");

        // The disk goes bad: this update must be refused...
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
        let failed = state
            .update_meta("sake", Some("B".to_string()), None, None, None)
            .unwrap();
        assert!(failed.is_err(), "a persist failure must surface as Err");
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();

        // ...and must not have left "B" sitting in memory — a later,
        // unrelated successful update must still see and persist "A",
        // not silently resurrect the failed change.
        let meta = state
            .update_meta("sake", None, Some(true), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(
            meta.description, "A",
            "the failed update to \"B\" must not have survived in memory"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn every_latecomer_behind_a_delete_finds_the_tombstone() {
        let dir = scratch_dir("delete-tombstone");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("victim", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let (_, stale) = state
            .snapshot()
            .into_iter()
            .find(|(name, _)| name == "victim")
            .unwrap();
        state.delete("victim").unwrap().unwrap();

        // The gate every post-lookup lock acquisition goes through:
        // a handle that predates the removal must be turned away.
        assert!(
            stale.lock_unless_deleted().is_none(),
            "the tombstone must refuse a stale handle"
        );
        // And the public write path answers NotFound rather than
        // recreating the WAL file the delete just removed.
        assert!(matches!(
            state.add_associations(
                "victim",
                vec![assoc_op("幽霊", "は", "残らない", 1.0, None)],
                Deadline::unbounded(),
            ),
            Err(AccessError::NotFound)
        ));
        assert!(!wal_path(&dir, &file_stem("victim")).exists());

        let _ = fs::remove_dir_all(dir);
    }

    /// A failed create must release its `pending.creates` reservation —
    /// otherwise one disk refusal would leave the name reading as taken
    /// until restart.
    #[test]
    fn a_failed_create_releases_the_name() {
        let dir = scratch_dir("create-release");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

        // A directory where create expects at most a stale FILE:
        // remove_file refuses it with something other than NotFound,
        // failing the clear-the-slate pass after the name is reserved.
        let obstruction = wal_path(&dir, &file_stem("sake"));
        fs::create_dir_all(&obstruction).unwrap();
        assert!(matches!(
            state.create("sake", ContextMeta::default()),
            Err(CreateError::Io(_))
        ));

        // Obstruction gone, the same name must create cleanly — the
        // failed attempt's reservation may not linger.
        fs::remove_dir_all(&obstruction).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_create_racing_a_slow_delete_is_refused_not_interleaved() {
        let dir = scratch_dir("delete-create-race");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        // Stall the delete mid-flight: it unregisters the name, then
        // must wait for this read guard before it may touch files —
        // exactly the window where a create used to interleave and
        // have its new generation unlinked from under it.
        let entry = state.lookup("sake").unwrap();
        let stall = entry.inner.read();
        let deleter = {
            let state = state.clone();
            std::thread::spawn(move || state.delete("sake").unwrap().unwrap())
        };
        while state.lookup("sake").is_some() {
            std::thread::yield_now();
        }
        assert!(
            matches!(
                state.create("sake", ContextMeta::default()),
                Err(CreateError::AlreadyExists)
            ),
            "a mid-delete name must read as taken"
        );

        drop(stall);
        deleter.join().unwrap();
        // The delete has fully finished: the name is free again and the
        // recreate starts from a clean slate.
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "recreate")
            .unwrap();
        assert!(image_path(&dir, &file_stem("sake")).exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dice_floor_persists_in_the_sidecar_and_reapplies_on_load() {
        let dir = scratch_dir("floor");
        // One shared informative bigram of 4+3: Dice ≈ 0.286 — misses
        // the 0.3 default, lands once the context is tuned to 0.25.
        let fuzzy_cue = "青嶺の純米";
        let lands = |state: &AppState| {
            state
                .read_context("sake", |context| {
                    context
                        .resolve(fuzzy_cue)
                        .iter()
                        .any(|hit| hit.name == "青嶺酒造")
                })
                .map_err(|_| "read")
                .unwrap()
        };
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .write_context("sake", |context| {
                    context.associate("青嶺酒造", "分類", "酒蔵", 1.0).unwrap();
                })
                .map_err(|_| "write")
                .unwrap();

            assert!(!lands(&state), "default floor must reject the cue");

            // Tuning applies to the loaded context immediately.
            state
                .update_meta("sake", None, None, Some(0.25), None)
                .unwrap()
                .unwrap();
            assert!(lands(&state), "tuned floor must admit the cue");
            // The flusher learns which contexts it persisted — that list
            // feeds the auto embedding refresh.
            assert_eq!(state.flush_dirty(), vec!["sake".to_string()]);
            assert!(state.flush_dirty().is_empty());
        }

        // A cold boot re-applies the floor from the sidecar — the image
        // itself carries no config.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(lands(&state), "floor must survive the restart");
        assert_eq!(state.directory()[0].dice_floor, Some(0.25));

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rename_context_moves_the_family_and_rewrites_group_membership() {
        let dir = scratch_dir("rename-context-happy");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create(
                "sake",
                ContextMeta {
                    pinned: true,
                    ..ContextMeta::default()
                },
            )
            .unwrap();
        state
            .add_associations(
                "sake",
                vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("a.md"))],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state
            .create_group(
                "drinks",
                String::new(),
                BTreeSet::from(["sake".to_string()]),
                BTreeSet::new(),
            )
            .unwrap();

        state.rename_context("sake", "shochu").unwrap();

        assert!(
            state.directory_entry("sake").is_none(),
            "the old name must be gone"
        );
        let entry = state
            .directory_entry("shochu")
            .expect("the new name must answer");
        assert!(entry.pinned, "pinned carries over");
        assert!(
            entry.loaded,
            "a pinned context reloads hot under its new name"
        );
        assert!(!dir.join("sake.ctx").exists());
        assert!(dir.join("shochu.ctx").exists());
        assert_eq!(
            state.group("drinks").unwrap().contexts,
            BTreeSet::from(["shochu".to_string()]),
            "group membership follows the rename, not a stale name"
        );
        assert!(!renaming_marker_path(&dir, &file_stem("sake")).exists());
        let count = state
            .read_context("shochu", |context| context.association_count())
            .unwrap();
        assert_eq!(count, 1, "data must survive the move");

        // Persisted, not just in memory.
        drop(state);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.directory_entry("sake").is_none());
        assert!(state.directory_entry("shochu").is_some());
        assert_eq!(
            state.group("drinks").unwrap().contexts,
            BTreeSet::from(["shochu".to_string()])
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// `note_read`/`note_write` bump their atomics through a bare
    /// `lookup(name)` regardless of Hot or Cold — nothing about them
    /// checks the slot. `drain_entry_for_rename` only ever hands its
    /// usage snapshot to `save_files` inside the `Slot::Hot` branch, so
    /// a Cold context's usage — whatever was counted since its last
    /// flush, which for a Cold entry may be everything it has ever
    /// counted — was silently dropped on every rename before the fix:
    /// the new entry was seeded from whatever sidecar already happened
    /// to sit on disk, untouched by the rename.
    #[test]
    fn rename_carries_usage_counted_while_the_context_was_cold() {
        let dir = scratch_dir("rename-usage-cold");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state.note_read("sake", false);
        state.note_write("sake");

        let entry = state.lookup("sake").unwrap();
        assert!(state.evict_entry("sake", &entry));

        // Counted while Cold — no flush or eviction will ever see these
        // before the rename runs.
        state.note_read("sake", false);
        state.note_read("sake", true);
        state.note_write("sake");

        state.rename_context("sake", "sake2").unwrap();

        let usage = state
            .directory_entry("sake2")
            .expect("the new name must answer")
            .usage;
        assert_eq!(
            (usage.reads, usage.empty_reads, usage.writes),
            (3, 1, 2),
            "usage counted while the context sat Cold must survive the \
             rename, not just whatever was already on disk before it went \
             Cold"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// Regression for the critical data-loss bug in
    /// `rename_context_locked`: once the registry has forgotten `from`
    /// (the point of no return), a failed pivot move must keep `from`
    /// reserved in `pending.renames` rather than release it. Before the
    /// fix, `rename_context` unconditionally cleared the reservation on
    /// any failure, so a client's natural reaction to seeing `from`
    /// vanish — `create(from)` — sailed through `create_files`'s
    /// stale-file sweep and deleted both the untouched old generation's
    /// files AND the `.renaming` marker that boot needs to resume the
    /// move, erasing the data beyond any recovery.
    #[test]
    fn a_rename_stuck_past_the_point_of_no_return_refuses_a_recreate_and_survives_reboot() {
        let dir = scratch_dir("rename-stuck-recreate");
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

        // Block the pivot: `fs::rename` onto an existing directory
        // fails with ENOTDIR/EISDIR, deterministically breaking
        // `move_context_files`'s first (pivot) move without touching
        // permissions — which would also break the marker write that
        // must succeed first.
        let blocker = dir.join(format!("{}.ctx", file_stem("shochu")));
        fs::create_dir(&blocker).unwrap();

        let error = state.rename_context("sake", "shochu").unwrap_err();
        assert!(
            matches!(error, RenameContextError::Io(_)),
            "the pivot move must fail: {error:?}"
        );

        assert!(
            state.directory_entry("sake").is_none(),
            "memory already forgot the source name past the point of no return"
        );
        assert!(
            state.directory_entry("shochu").is_none(),
            "the destination never landed"
        );
        assert!(
            renaming_marker_path(&dir, &file_stem("sake")).exists(),
            "the marker must survive so boot can resume the move"
        );
        assert!(
            dir.join("sake.ctx").exists(),
            "the old generation's files must stay put, untouched"
        );

        // The dangerous part: a client that saw `sake` disappear (404)
        // and naturally retries with create() must be refused, not
        // handed a fresh empty context in place of the old data.
        assert!(
            matches!(
                state.create("sake", ContextMeta::default()),
                Err(CreateError::AlreadyExists)
            ),
            "a stuck rename must keep blocking create(), or create_files' \
             stale-file sweep would delete the marker and the old data"
        );
        assert!(
            renaming_marker_path(&dir, &file_stem("sake")).exists(),
            "the refused create must not have touched the marker"
        );
        assert!(
            dir.join("sake.ctx").exists(),
            "the refused create must not have touched the old data"
        );

        // Clear the obstruction and let boot's resume-sweep finish what
        // the live call could not.
        fs::remove_dir(&blocker).unwrap();
        drop(state);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.directory_entry("sake").is_none());
        let count = state
            .read_context("shochu", |context| context.association_count())
            .unwrap();
        assert_eq!(count, 1, "the resumed move must carry the old data over");
        assert!(!renaming_marker_path(&dir, &file_stem("sake")).exists());

        let _ = fs::remove_dir_all(dir);
    }

    /// The crash-shaped state: `rename_context` wrote its marker but
    /// died before the file move and the group rewrite landed. Boot
    /// must finish both, and in the right order — rewrite group
    /// membership before `reconcile_groups` runs — or reconcile sees
    /// "sake" as a plain dangling reference (nothing registered under
    /// that name any more) and drops it instead of carrying it to
    /// "shochu". This is the regression `boot_with`'s ordering exists
    /// to prevent.
    #[test]
    fn an_unfinished_context_rename_is_resumed_at_boot_before_group_reconciliation() {
        let dir = scratch_dir("rename-context-crash");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            state
                .create_group(
                    "drinks",
                    String::new(),
                    BTreeSet::from(["sake".to_string()]),
                    BTreeSet::new(),
                )
                .unwrap();
        }
        // No manual file move: `scan_data_dir` performs it itself once
        // it sees the marker, exactly as it would resuming a real crash.
        fs::write(
            renaming_marker_path(&dir, &file_stem("sake")),
            serde_json::to_vec(&RenameMarker {
                from: "sake".to_string(),
                to: "shochu".to_string(),
            })
            .unwrap(),
        )
        .unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.directory_entry("sake").is_none());
        assert!(state.directory_entry("shochu").is_some());
        assert!(!dir.join("sake.ctx").exists());
        assert!(dir.join("shochu.ctx").exists());
        assert_eq!(
            state.group("drinks").unwrap().contexts,
            BTreeSet::from(["shochu".to_string()]),
            "the membership must be REWRITTEN to the new name, not pruned as dangling"
        );
        assert!(!renaming_marker_path(&dir, &file_stem("sake")).exists());

        let _ = fs::remove_dir_all(dir);
    }

    /// A rename that half-finished with `sake` as its SOURCE leaves a
    /// `.renaming` marker at sake's stem and frees the name to be created
    /// again on the same live server. The create must strip that marker,
    /// or the next boot's resume-sweep moves the fresh generation onto
    /// the rename's destination and `sake` silently becomes `shochu`.
    #[test]
    fn creating_a_context_abandons_a_rename_marker_at_its_own_stem() {
        let dir = scratch_dir("create-ctx-clears-source-marker");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            fs::write(
                renaming_marker_path(&dir, &file_stem("sake")),
                serde_json::to_vec(&RenameMarker {
                    from: "sake".to_string(),
                    to: "shochu".to_string(),
                })
                .unwrap(),
            )
            .unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            assert!(
                !renaming_marker_path(&dir, &file_stem("sake")).exists(),
                "create must clear a rename marker sitting at its own stem"
            );
        }
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(
            state.directory_entry("sake").is_some(),
            "the freshly created context must survive, not be swept to the rename's destination"
        );
        assert!(state.directory_entry("shochu").is_none());
        assert!(dir.join("sake.ctx").exists());
        assert!(!dir.join("shochu.ctx").exists());

        let _ = fs::remove_dir_all(dir);
    }

    /// A rename that half-finished with `sake` as its DESTINATION leaves
    /// its marker under the SOURCE's stem (`beer`) — a stem the create of
    /// `sake` cannot derive from its own name. Creating `sake` must scan
    /// for markers naming it as `to` and drop them, or the next boot's
    /// resume-sweep renames the stale `beer` family onto the fresh `sake`
    /// (fs::rename overwrites), clobbering it and erasing `beer`.
    #[test]
    fn creating_a_context_abandons_a_rename_marker_naming_it_as_destination() {
        let dir = scratch_dir("create-ctx-clears-destination-marker");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("beer", ContextMeta::default()).unwrap();
            fs::write(
                renaming_marker_path(&dir, &file_stem("beer")),
                serde_json::to_vec(&RenameMarker {
                    from: "beer".to_string(),
                    to: "sake".to_string(),
                })
                .unwrap(),
            )
            .unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            assert!(
                !renaming_marker_path(&dir, &file_stem("beer")).exists(),
                "create must clear a rename marker that names it as the destination"
            );
        }
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(
            state.directory_entry("beer").is_some(),
            "the abandoned rename must leave the untouched source context intact"
        );
        assert!(
            state.directory_entry("sake").is_some(),
            "the freshly created destination context must survive, not be overwritten by the source"
        );
        assert!(dir.join("beer.ctx").exists());
        assert!(dir.join("sake.ctx").exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rename_context_error_cases() {
        let dir = scratch_dir("rename-context-errors");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state.create("beer", ContextMeta::default()).unwrap();

        assert!(matches!(
            state.rename_context("missing", "whatever"),
            Err(RenameContextError::NotFound)
        ));
        assert!(matches!(
            state.rename_context("sake", "beer"),
            Err(RenameContextError::AlreadyExists)
        ));
        assert!(matches!(
            state.rename_context("sake", ""),
            Err(RenameContextError::InvalidName)
        ));
        assert!(
            state.rename_context("sake", "sake").is_ok(),
            "renaming a name to itself is a no-op, not an error"
        );
        assert!(state.directory_entry("sake").is_some());

        let _ = fs::remove_dir_all(dir);
    }

    /// Same fence a create races against a slow delete in
    /// `a_create_racing_a_slow_delete_is_refused_not_interleaved`: a
    /// rename reserves both its names in `pending.renames` before it
    /// may touch any file, so a create for either name must be refused
    /// until the rename settles, never interleaved with it.
    #[test]
    fn a_create_racing_a_pending_context_rename_is_refused_for_both_names() {
        let dir = scratch_dir("rename-create-race");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();

        let entry = state.lookup("sake").unwrap();
        let stall = entry.inner.read();
        let renamer = {
            let state = state.clone();
            std::thread::spawn(move || state.rename_context("sake", "shochu").unwrap())
        };
        while !state.0.pending.lock().renames.contains("sake") {
            std::thread::yield_now();
        }
        assert!(
            matches!(
                state.create("sake", ContextMeta::default()),
                Err(CreateError::AlreadyExists)
            ),
            "the source name is reserved until the rename settles"
        );
        assert!(
            matches!(
                state.create("shochu", ContextMeta::default()),
                Err(CreateError::AlreadyExists)
            ),
            "the destination name is reserved too, before any file lands there"
        );

        drop(stall);
        renamer.join().unwrap();
        assert!(state.directory_entry("shochu").is_some());
        assert!(!state.0.pending.lock().renames.contains("sake"));
        assert!(!state.0.pending.lock().renames.contains("shochu"));

        let _ = fs::remove_dir_all(dir);
    }

    /// The rename twin of
    /// `a_passage_write_racing_a_delete_backs_off_at_the_tombstone`: a
    /// handle taken before the rename must see the tombstone after,
    /// not the old generation's live state, and no write may recreate
    /// the old name from under it.
    #[test]
    fn a_write_racing_a_rename_backs_off_at_the_tombstone() {
        let dir = scratch_dir("rename-write-race");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();

        let entry = state.lookup("sake").unwrap();
        state.rename_context("sake", "shochu").unwrap();
        assert!(
            entry.read_unless_deleted().is_none(),
            "a handle from before the rename must see the tombstone"
        );
        assert!(
            matches!(
                state.add_associations(
                    "sake",
                    vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("a.md"))],
                    Deadline::unbounded(),
                ),
                Err(AccessError::NotFound)
            ),
            "the old name is gone; nothing may recreate it via a write"
        );

        let _ = fs::remove_dir_all(dir);
    }
}
