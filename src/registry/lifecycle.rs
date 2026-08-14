use super::*;

/// Which of the two callers [`AppState::sweep_stale_stem_files`] is
/// clearing a stem for — the two never mix and match independently, so
/// this collapses what used to be two positional flags into one
/// self-documenting choice.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StemSweep<'a> {
    /// A brand-new generation is about to land at `stem`: clear the
    /// entire prior family, image (`.ctx`) included. There is no
    /// pivot move to lean on here — removing the old image up front is
    /// the only thing standing between an interrupted write and a
    /// resurrected old generation.
    FreshCreate,
    /// `rename_context_locked`'s destination stem: leave the image
    /// (`.ctx`) untouched — the pivot, the first `fs::rename` in
    /// `move_context_files`, already replaces whatever sits at the
    /// destination, and pre-clearing it here would turn a
    /// deterministic pivot failure (destination occupied by something
    /// `fs::rename` refuses to replace) into a sweep failure instead,
    /// discarding the in-flight `.renaming` marker on rollback for a
    /// case meant to leave it in place for boot to resume. That
    /// marker — `exclude_marker`, the caller's own, just-written one —
    /// is also excluded from the destination-targeting scan below, so
    /// this sweep does not mistake the rename now in progress for a
    /// stale, abandoned one and delete the marker out from under
    /// itself.
    RenameDestination { exclude_marker: &'a Path },
}

impl AppState {
    /// Registers an empty context and persists it immediately, so its
    /// existence (and description) survives a crash from the moment the
    /// create call returns. A persistence failure fails the create.
    ///
    /// The registry lock is NOT held across the disk work (an unlink
    /// attempt for each candidate path `sweep_stale_stem_files` removes
    /// — the stem's on-disk family minus `meta_path`, which
    /// `save_files` overwrites instead — plus one per stale rename or
    /// import marker it finds, plus save_files' fsyncs — seconds on
    /// slow storage, behind which every operation on every context
    /// would otherwise stall). The name is reserved in
    /// `pending.creates` under the registry guard, the files are
    /// written unlocked, and the entry
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
                    // A brand-new generation never has a schema: the
                    // sweep above just removed any stray file an
                    // earlier generation of this name left behind.
                    None,
                    None,
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
        self.sweep_stale_stem_files(name, &stem, StemSweep::FreshCreate)
            .map_err(CreateError::Io)?;
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
            None,
            &context,
        )
        .map_err(CreateError::Io)?;
        Ok((stats, usage, context))
    }

    /// Clears every stale leftover a half-finished delete or rename may
    /// have left sitting at `stem` (`name`'s own file family, plus its
    /// `.deleted`/`.renaming` markers and any import markers) or naming
    /// `name` as a rename's DESTINATION under a source stem this
    /// function cannot derive from `name` alone. Shared by
    /// [`AppState::create_files`] (clearing the slate before a brand
    /// new generation's files land at `stem`) and
    /// [`AppState::rename_context_locked`] (clearing the slate at
    /// `to_stem` before the moved family lands there) — both put a
    /// fresh generation at `stem` and both must not let an EARLIER
    /// generation's leftovers bleed into it or survive to mislead a
    /// later boot's resume-sweep. See [`StemSweep`] for how the two
    /// callers differ.
    fn sweep_stale_stem_files(&self, name: &str, stem: &str, mode: StemSweep) -> io::Result<()> {
        let exclude = match mode {
            StemSweep::RenameDestination { exclude_marker } => Some(exclude_marker),
            StemSweep::FreshCreate => None,
        };
        let mut stale_paths = vec![
            wal_path(&self.0.data_dir, stem),
            sources_path(&self.0.data_dir, stem),
            passages_path(&self.0.data_dir, stem),
            passages_wal_path(&self.0.data_dir, stem),
            pvectors_path(&self.0.data_dir, stem),
            bm25_path(&self.0.data_dir, stem),
            vectors_path(&self.0.data_dir, stem),
            // Neither `create_files` nor (on its own) a rename ever
            // WRITES this file — only `PUT /contexts/{name}/schema`
            // will (#380) — so unlike `meta_path` (always freshly
            // overwritten by `save_files`/the moved family) a stray one
            // left by an earlier generation at this stem would
            // otherwise silently attach to the fresh context. Swept
            // here so a reused name never inherits schema litter that
            // would fail `ensure_hot`'s digest check on the very first
            // cold load (the fresh sidecar records no digest for it).
            schema_path(&self.0.data_dir, stem),
            // A leftover marker from an earlier delete that could not
            // finish MUST go before this new generation of files
            // lands — otherwise the next boot's resume-sweep sees the
            // marker and deletes the context we are creating right now.
            deleted_marker_path(&self.0.data_dir, stem),
            // The same hazard for a rename that half-finished with THIS
            // name as its SOURCE: its `.renaming` marker sits at this
            // stem, and boot's resume-sweep would otherwise move the
            // generation we are about to write onto the rename's
            // destination stem, losing it silently.
            renaming_marker_path(&self.0.data_dir, stem),
        ];
        if mode == StemSweep::FreshCreate {
            stale_paths.push(image_path(&self.0.data_dir, stem));
        }
        for stale in stale_paths {
            if let Err(error) = remove_persisted_file(&stale)
                && error.kind() != io::ErrorKind::NotFound
            {
                return Err(error);
            }
        }
        // A rename that half-finished with THIS name as its DESTINATION
        // left its marker under the SOURCE's stem — a stem we cannot
        // derive from `name`. Boot's resume-sweep would move that source
        // family onto the generation we are about to write, erasing it.
        // Scan for any marker that names us as `to` and drop it; landing
        // a fresh generation here abandons a stuck rename either way.
        for stale in rename_markers_targeting(&self.0.data_dir, name, "renaming") {
            if exclude.is_some_and(|excluded| excluded == stale) {
                continue;
            }
            if let Err(error) = remove_persisted_file(&stale)
                && error.kind() != io::ErrorKind::NotFound
            {
                return Err(error);
            }
        }
        // Stale import markers are part of the same earlier generation:
        // left beside the new files, boot would report the fresh
        // context as carrying a torn import it never ran.
        for stale in import_marker_paths(&self.0.data_dir, stem) {
            if let Err(error) = remove_persisted_file(&stale)
                && error.kind() != io::ErrorKind::NotFound
            {
                return Err(error);
            }
        }
        Ok(())
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
    pub fn delete(&self, name: &str) -> Option<Result<(), DeleteError>> {
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
            // Reported through the same `Option<Result<...>>` a live
            // name already uses — the caller sees a name that exists
            // but cannot be deleted right now, not "no such context".
            // `MidRename` is its own variant, not `Io`: nothing was
            // touched, so the API layer must not report this as a
            // completed (if partial) deletion the way `Io` does.
            if self.0.pending.lock().renames.contains(name) {
                return Some(Err(DeleteError::MidRename));
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
        // A stuck rename naming THIS stem as its SOURCE goes with the
        // family too: `create_files`' sweep clears exactly this marker
        // for the same reason (`sweep_stale_stem_files`'s
        // `renaming_marker_path` push) — a survivor would have the
        // next boot's resume-sweep try to move a family that no longer
        // exists onto a destination stem, resurrecting neither name.
        // Reachable here: `pending.renames` only rejects this delete
        // while the ORIGINAL call is still in flight, not across a
        // restart — a stuck rename's marker survives boot with
        // `pending.renames` empty, so a `delete(from)` right after
        // boot would otherwise sail through and orphan it.
        if let Err(error) = remove_persisted_file(renaming_marker_path(&self.0.data_dir, &stem))
            && error.kind() != io::ErrorKind::NotFound
        {
            outcome = Err(error);
        }
        // The same hazard with THIS name as a stuck rename's
        // DESTINATION: its marker sits at the SOURCE's stem, which
        // `name` alone cannot derive — scanned the same way
        // `sweep_stale_stem_files` scans for `create`.
        for stale in rename_markers_targeting(&self.0.data_dir, name, "renaming") {
            if let Err(error) = remove_persisted_file(&stale)
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
        Some(outcome.map_err(DeleteError::Io))
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
        let entry = {
            let registry = self.0.registry.read();
            let Some(entry) = registry.get(from) else {
                return Err(RenameContextError::NotFound);
            };
            // Checked AFTER existence, not before: a self-rename of a
            // name that does not exist is still a `NotFound`, not a
            // silent no-op success.
            if from == to {
                return Ok(());
            }
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
            // Failed AT or AFTER the point of no return, OR a rollback
            // that could not retract its own marker (see
            // `rollback_rename`): the durable `.renaming` marker
            // survives on disk and still names BOTH `from` and `to` as
            // its pair, so BOTH names must stay reserved. `from` MUST
            // stay reserved — releasing it would let a client's
            // create(from) sweep away the marker and the old
            // generation's files as ordinary "stale leftovers" (see
            // create_files), destroying them beyond any recovery. `to`
            // is exposed to the exact same hazard: the marker's own
            // `rename_markers_targeting` scan is what `create(to)`'s
            // sweep uses to abandon a half-done rename it finds naming
            // `to` as the destination — releasing `to` while the marker
            // still exists would have that same sweep delete the marker
            // (and, if the pivot already moved, the half-migrated image
            // sitting at `to`) as though the rename had simply never
            // started, leaving `from` reserved forever with no image to
            // recover. Only a boot resume-sweep (or the marker's own
            // eventual retraction) can resolve this, so both
            // reservations outlive this call.
            RenameOutcome::Stuck(error) => {
                // Never a bare `?error`/`error` field: `tracing-opentelemetry`
                // maps a field literally named `error` to an exception
                // event and (by default) an ERROR span status — ADR 0008
                // §2.5(a)/§7 calls this out as a live defect elsewhere in
                // the tree, not a naming choice this call site gets to
                // repeat.
                tracing::error!(
                    from = %from, to = %to, rename_error = ?error,
                    "context rename failed after the point of no return; both names \
                     stay reserved until the next restart resumes it from the \
                     .renaming marker"
                );
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
    /// any marker written was retracted, and any hydrator veto was
    /// undone), so both names are free again. [`RenameOutcome::Stuck`]
    /// means either it failed after `from` was already removed from the
    /// registry, OR a rollback itself could not retract the marker —
    /// in both cases the marker survives on disk and only a boot
    /// resume-sweep (or a successful retry) can resolve it, so `from`
    /// must stay reserved in the meantime.
    fn rename_context_locked(&self, from: &str, to: &str, entry: &Arc<Entry>) -> RenameOutcome {
        let from_stem = file_stem(from);
        let to_stem = file_stem(to);
        // A lazy bucket boot: `evict_stem` hydrates the family before
        // vetoing its re-materialization — see that method's doc for
        // why the order matters. The undo token is carried through
        // every rollback below — `ensure_context` treats `Vetoed` as
        // success, so a veto left standing after ITS OWN caller failed
        // would make `from` silently unreadable until the next restart.
        let veto_undo = match &self.0.hydrator {
            Some(hydrator) => match hydrator.evict_stem(&from_stem) {
                Ok(undo) => Some(undo),
                Err(error) => return RenameOutcome::RolledBack(RenameContextError::Io(error)),
            },
            None => None,
        };
        let marker = renaming_marker_path(&self.0.data_dir, &from_stem);
        if let Err(error) = write_rename_marker(&marker, from, to) {
            // Nothing durable landed yet — no marker to retract.
            self.undo_rename_veto(&from_stem, veto_undo);
            return RenameOutcome::RolledBack(RenameContextError::Io(error));
        }
        // A half-finished delete or rename may have left stale markers
        // (or leftover files) sitting at `to_stem` — the same hazard
        // `create_files` guards against for a brand new generation. A
        // `.deleted` marker there would have boot's resume-sweep erase
        // the family we are about to move in; a `.renaming` marker
        // would have it resumed onto (and overwrite) what we land here.
        // Swept BEFORE `drain_entry_for_rename` tombstones `from`'s
        // entry (`Slot::Deleted`, in memory — not yet reflected in the
        // registry map): a sweep failure here only rolls back the
        // marker, since `from`'s entry has not been touched yet either.
        // Ordered the other way, a sweep failure after the tombstone
        // would still report `RolledBack` and free both name
        // reservations, but leave `from`'s entry tombstoned forever —
        // registered under its old name, yet permanently unusable.
        let sweep_mode = StemSweep::RenameDestination {
            exclude_marker: marker.as_path(),
        };
        if let Err(error) = self.sweep_stale_stem_files(to, &to_stem, sweep_mode) {
            return self.rollback_rename(
                &from_stem,
                &marker,
                veto_undo,
                RenameContextError::Io(error),
            );
        }
        if let Err(error) = self.drain_entry_for_rename(from, entry) {
            return self.rollback_rename(
                &from_stem,
                &marker,
                veto_undo,
                RenameContextError::Io(error),
            );
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
        let mut meta_file = read_meta_file(&self.0.data_dir, &to_stem);
        meta_file.usage = ContextUsage {
            reads: meta_file.usage.reads.max(final_usage.reads),
            empty_reads: meta_file.usage.empty_reads.max(final_usage.empty_reads),
            writes: meta_file.usage.writes.max(final_usage.writes),
            last_read_epoch: meta_file
                .usage
                .last_read_epoch
                .max(final_usage.last_read_epoch),
            last_write_epoch: meta_file
                .usage
                .last_write_epoch
                .max(final_usage.last_write_epoch),
        };
        let pinned = meta_file.meta.pinned;
        let (wal_bytes, passages_wal_bytes) = wal_lane_bytes(&self.0.data_dir, &to_stem);
        // The revision moves with the sidecar: a rename is the same
        // content under a new name, so the counters carry over intact
        // (the group fingerprint still changes — the member NAME is
        // part of its hash).
        let new_entry = Arc::new(Entry::cold_from_meta(
            meta_file,
            wal_bytes,
            passages_wal_bytes,
            // Not resolved here even though the schema file (if any)
            // moved with the rest of the family a few lines up: this
            // mirrors the hydrator-registration case above rather than
            // re-reading a file the entry is about to go Cold over
            // anyway. `AppState::schema_of` resolves it lazily on first
            // read, or `ensure_hot` does on first load.
            None,
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
        let membership_persisted = {
            let mut groups = self.0.groups.write();
            rename_in_membership(&self.0.data_dir, &mut groups, from, to, |record| {
                &mut record.contexts
            })
        };
        // The marker is removed only once BOTH the move and the
        // membership rewrite are durable — see `retire_rename_marker`'s
        // doc for why an unconditional removal here would have the
        // next boot's `reconcile_groups` see `from` as a plain dangling
        // reference and drop it instead of resuming the rewrite.
        retire_rename_marker(
            &marker,
            membership_persisted,
            from,
            to,
            "context rename's group membership rewrite",
        );
        RenameOutcome::Ok
    }

    /// Undoes a [`crate::hydrate::Hydrator::veto`] taken on `from_stem`
    /// at the top of [`Self::rename_context_locked`], if any — shared
    /// by every rollback arm above the point of no return so none of
    /// them can forget it.
    fn undo_rename_veto(&self, from_stem: &str, veto_undo: Option<crate::hydrate::VetoUndo>) {
        if let (Some(hydrator), Some(undo)) = (&self.0.hydrator, veto_undo) {
            hydrator.undo_veto(from_stem, undo);
        }
    }

    /// One rollback path for every `rename_context_locked` failure that
    /// happens AFTER the marker landed but BEFORE the point of no
    /// return: retracts the marker (through the same
    /// [`remove_persisted_file`] choke point every other unlink in this
    /// module uses, so a test's fault injector sees it), undoes the
    /// hydrator veto, and reports [`RenameOutcome::RolledBack`] — UNLESS
    /// the marker itself will not go away, in which case the durable
    /// promise it makes ("boot resumes this rename") is still live: the
    /// veto is left standing to match, and the call reports
    /// [`RenameOutcome::Stuck`] instead so the caller keeps `from`
    /// reserved rather than handing the name back while a marker still
    /// claims it.
    fn rollback_rename(
        &self,
        from_stem: &str,
        marker: &Path,
        veto_undo: Option<crate::hydrate::VetoUndo>,
        error: RenameContextError,
    ) -> RenameOutcome {
        // NotFound counts as retracted, same as everywhere else in this
        // module: nothing left to undo.
        let retracted = match remove_persisted_file(marker) {
            Ok(()) => true,
            Err(io_error) if io_error.kind() == io::ErrorKind::NotFound => true,
            Err(marker_error) => {
                // See the same note in `rename_context`'s `Stuck` arm:
                // never a bare `?error`/`error` field.
                tracing::error!(
                    from_stem, %marker_error, rename_error = ?error,
                    "rename rollback could not retract its marker; the source name and \
                     the hydrator veto both stay in place until the next boot resumes it"
                );
                false
            }
        };
        if retracted {
            self.undo_rename_veto(from_stem, veto_undo);
            RenameOutcome::RolledBack(error)
        } else {
            RenameOutcome::Stuck(error)
        }
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
        let schema_digest = inner.schema_digest.clone();
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
                schema_digest.as_deref(),
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
                inner.schema_digest.as_deref(),
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

    /// The resident schema for `name` — `Ok(None)` for a schema-free
    /// context (`GET /contexts/{name}/schema`, #380, turns that into a
    /// 404). Outer `None` means no such context.
    ///
    /// The common case is already resolved without touching disk: boot
    /// and every cold-load already ran `load_schema` (ADR 0009 §5.2's
    /// consistency check) into [`EntryInner::schema`], and a
    /// `schema_digest` of `None` — set only by `put_schema` below, under
    /// this same lock — never means anything but "no schema". Only a
    /// digest recorded but not yet checked against its bytes LOCALLY
    /// (a replica mid-hydration, or a rename's freshly registered
    /// entry — see `EntryInner::schema`'s own doc) falls through to the
    /// slow path, which reuses `ensure_hot` rather than calling
    /// `schema::load_schema` directly: `ensure_hot` is the one place
    /// that also runs the hydrator, so a replica whose family has not
    /// been fetched yet resolves correctly here too, not just a purely
    /// local rename. Heavier than strictly needed (it loads the full
    /// graph image to get there), but `GET /schema` is an infrequent
    /// management call, not a retrieval hot path.
    pub fn schema_of(
        &self,
        name: &str,
    ) -> Option<Result<Option<Arc<schema::InstalledSchema>>, String>> {
        let entry = self.lookup(name)?;
        {
            let inner = entry.read_unless_deleted()?;
            if inner.schema.is_some() || inner.schema_digest.is_none() {
                return Some(Ok(inner.schema.clone()));
            }
        }
        let outcome = {
            let mut guard = entry.lock_unless_deleted()?;
            let inner = &mut *guard;
            if let Err(error) = ensure_hot(
                &self.0.data_dir,
                name,
                inner,
                &self.0.metrics,
                self.0.hydrator.as_deref(),
            ) {
                return Some(Err(error));
            }
            self.recount_entry(inner);
            Ok(inner.schema.clone())
        };
        self.enforce_budget(name);
        Some(outcome)
    }

    /// ADR 0009 §6.3's one gate for the reserved `schema:type` label:
    /// "an installed schema document exists," never "mode != off." An
    /// operator who installs a schema but leaves it in `off` while
    /// drafting types has already committed to the reserved label
    /// meaning something — `off` only means "don't enforce domain/range
    /// yet," not "pretend the label is ordinary." `Some(SCHEMA_TYPE_LABEL)`
    /// whenever this context has ever had a schema installed, in any
    /// mode; `None` only for a context that never installed one (or an
    /// unknown/deleted name). A schema recorded but currently unreadable
    /// (`schema_of`'s `Err` arm) maps CONSERVATIVELY to "hidden" — per
    /// [`schema`]'s own module doc, every trouble case there is a hard
    /// refusal, never a silent fallback, and this helper must not be the
    /// one place that quietly un-reserves the label because a read
    /// failed.
    ///
    /// ⚠ Never call this from inside a [`AppState::read_context`]
    /// closure: the slow path (through [`AppState::schema_of`]) takes
    /// this entry's write lock, while `read_context` already holds its
    /// read lock for the whole closure — parking_lot's `RwLock` is
    /// neither reentrant nor reader-preferring, so that ordering
    /// deadlocks. Resolve the hidden label first, then pass the
    /// `Option<&str>` into the closure.
    pub fn hidden_label(&self, name: &str) -> Option<&'static str> {
        match self.schema_of(name)? {
            Ok(Some(_)) => Some(schema::SCHEMA_TYPE_LABEL),
            Ok(None) => None,
            Err(_) => Some(schema::SCHEMA_TYPE_LABEL),
        }
    }

    /// [`Self::hidden_label`] as the exclusion slice a `read_context`
    /// call site actually wants (issue #622 finding 4) — off the async
    /// worker, since [`Self::hidden_label`]'s own doc requires it to
    /// run before, never inside, a `read_context` closure. Bundles the
    /// `block_in_place` + `.into_iter().collect()` idiom five HTTP
    /// handlers each wrote out by hand.
    pub fn excluded_hidden_label(&self, name: &str) -> Vec<&'static str> {
        tokio::task::block_in_place(|| self.hidden_label(name))
            .into_iter()
            .collect()
    }

    /// ADR 0009 §6.3 guard 2's `add_label_alias` bullet: the pre-flight
    /// an alias-creating write consults before it runs, mirroring
    /// `predicted_alias_rejection`'s own read-only-prediction shape —
    /// including that shape's own known race: this check and the
    /// write it precedes take two SEPARATE lock acquisitions, not one
    /// held across both, so a `PUT /schema` install landing in the
    /// gap between them is not caught here. That gap is not new to
    /// this guard — `apply_batch` already runs
    /// `predicted_alias_rejection` and its subsequent `add_aliases`/
    /// `add_associations` the same two-lock-acquisitions way, and ADR
    /// 0009 §7.3 explicitly declines to make the write path atomic
    /// against concurrent mutation ("that is #187's scope"). Closing
    /// it here would mean re-running this check under `add_aliases`'
    /// own write lock, which — because `Context` has no schema
    /// knowledge (§7.3's own reasoning for keeping the check a layer
    /// up) — reaches into every other `add_aliases`/`logged_write`
    /// caller too; deferred as the same kind of cross-cutting
    /// atomicity work #187 already owns, not attempted piecemeal here.
    /// Only meaningful once [`AppState::hidden_label`] says a schema
    /// exists — a schema-free context's `schema:type` stays an ordinary
    /// label (guard 1), so nothing here refuses anything for it.
    /// Deliberately does not chase a multi-hop alias chain: once a
    /// schema exists, no *live* alias can ever resolve to the reserved
    /// label (this guard and `PUT /schema`'s migration-boundary check
    /// both stand in its way going forward), so a direct value
    /// comparison against `labels` is the whole check.
    pub fn reserved_alias_conflict(
        &self,
        name: &str,
        labels: &BTreeMap<String, String>,
    ) -> Option<String> {
        self.hidden_label(name)?;
        schema::reserved_aliases(
            labels
                .iter()
                .map(|(alias, canonical)| (alias.as_str(), canonical.as_str())),
        )
        .next()
        .map(str::to_string)
    }

    /// `PUT /contexts/{name}/schema` (#380): installs `installed` as
    /// `name`'s schema document, replacing whatever was there wholesale
    /// — there is no delta form, so a retry after a failure below is
    /// always safe regardless of which side of it the previous attempt
    /// reached (ADR 0009 §5.2). Does exactly what `bump_config_revision`
    /// already does for `dice_floor`, plus `invalidate_cache_identity`:
    /// a schema mutation can change what `query`'s future type filter
    /// (§12.3) returns, so a retrieval-cache key minted before this call
    /// must not keep answering with the old constraints.
    ///
    /// Outer `None` means no such context. `Ok` carries the installed
    /// document back (including when the call was a no-op — see below)
    /// so the handler can answer `GET`-shaped without a second lookup.
    pub fn put_schema(
        &self,
        name: &str,
        installed: schema::InstalledSchema,
    ) -> Option<Result<schema::SchemaDocument, PutSchemaError>> {
        let entry = self.lookup(name)?;
        let outcome = {
            let mut guard = entry.lock_unless_deleted()?;
            let inner = &mut *guard;
            // Unconditional, unlike `update_meta`'s `pinned`-gated load:
            // the migration-boundary guard just below needs the LIVE
            // label-alias table, which only a hot context has, on every
            // call — aliases can be added between one `PUT` and the
            // next, so a resolution cached from an earlier call would
            // miss one created since.
            if let Err(error) = ensure_hot(
                &self.0.data_dir,
                name,
                inner,
                &self.0.metrics,
                self.0.hydrator.as_deref(),
            ) {
                self.recount_entry(inner);
                return Some(Err(PutSchemaError::Load(error)));
            }
            self.recount_entry(inner);
            // ADR 0009 §6.3 guard 2's install-time bullet: an
            // already-persisted `label_alias` resolving to the reserved
            // type label. Guard 2's other two bullets — refusing
            // `add_label_alias`/a batch's own `batch.labels` from ever
            // CREATING such an alias once a schema exists — are
            // `AppState::reserved_alias_conflict` (the aliases handler)
            // and `schema_issues`' `SchemaCheck::reserved` (a future
            // write entrance, S4/S5) respectively; neither existed
            // before this schema's own document did, so this call site
            // is the one place a violating alias predating them could
            // still slip through, and it stays on every `PUT` (not only
            // the off-to-installed transition) for exactly that reason.
            if let Some(alias) = schema::reserved_aliases(hot_context(inner).label_aliases()).next()
            {
                return Some(Err(PutSchemaError::ReservedAlias(alias.to_string())));
            }
            let bytes = match schema::document_bytes(installed.document()) {
                Ok(bytes) => bytes,
                Err(error) => return Some(Err(PutSchemaError::Io(error))),
            };
            let digest = crate::sha256::sha256_hex(&bytes);
            // A PUT that changes nothing bumps nothing — the same
            // idempotent-update discipline `update_meta` keeps for a
            // no-op PATCH — so a retried or duplicate `PUT` of the same
            // document never churns the retrieval cache.
            if inner.schema.is_some() && inner.schema_digest.as_deref() == Some(digest.as_str()) {
                Ok(installed.document().clone())
            } else {
                let stem = file_stem(name);
                let previous_digest = inner.schema_digest.clone();
                inner.config_revision += 1;
                inner.schema_digest = Some(digest);
                // Revision-then-content (ADR 0009 §5.2): this write
                // lands BEFORE the schema file's own `write_atomic`
                // below, both under this entry's write lock, so a
                // crash between the two always fails toward extra
                // invalidation (revision advanced, content unchanged)
                // rather than a served mismatch (content changed,
                // revision stale).
                let meta_result = write_meta(
                    &self.0.data_dir,
                    &stem,
                    &inner.meta,
                    &inner.stats,
                    &entry.usage.snapshot(),
                    entry.revision_snapshot(inner),
                    inner.schema_digest.as_deref(),
                );
                match meta_result {
                    Err(error) => {
                        inner.config_revision -= 1;
                        inner.schema_digest = previous_digest;
                        Err(PutSchemaError::Io(error))
                    }
                    Ok(()) => match schema::write_schema_bytes(&self.0.data_dir, &stem, &bytes) {
                        Ok(()) => {
                            let document = installed.document().clone();
                            inner.schema = Some(Arc::new(installed));
                            inner.invalidate_cache_identity();
                            // The change feed's config-side entrance
                            // (#422): only a PUT that actually changed
                            // the document reaches here — the idempotent
                            // early return above never feeds an event.
                            entry
                                .changes
                                .lock()
                                .push(crate::registry::ChangeKind::SchemaUpdated {
                                    mode: document.mode.as_str().to_string(),
                                });
                            Ok(document)
                        }
                        Err(error) => {
                            inner.config_revision -= 1;
                            inner.schema_digest = previous_digest.clone();
                            // Best-effort restore of the sidecar to the
                            // pre-PUT digest; if this ALSO fails, the
                            // next boot's digest check (§5.2) refuses
                            // rather than silently serving the
                            // mismatch — the same fail-closed posture
                            // `load_schema` already enforces, not a new
                            // mechanism this call adds.
                            let _ = write_meta(
                                &self.0.data_dir,
                                &stem,
                                &inner.meta,
                                &inner.stats,
                                &entry.usage.snapshot(),
                                entry.revision_snapshot(inner),
                                previous_digest.as_deref(),
                            );
                            Err(PutSchemaError::Io(error))
                        }
                    },
                }
            }
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
    use crate::registry::test_support::{assoc_op, loaded_map, scratch_dir};

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
                locators: Vec::new(),
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

    /// `update_meta`'s `dice_floor`/`semantic_floor` clamps
    /// (`floor.clamp(0.0, 1.0)`) have no test: every call site in the
    /// suite already passes an in-range value, so the clamp never
    /// actually clamps anything. It is also the ONLY guard on the PATCH
    /// path — `api/contexts.rs`'s create handler clamps up front, but
    /// its PATCH handler forwards `dice_floor`/`semantic_floor` raw.
    #[test]
    fn update_meta_clamps_out_of_range_floors_into_zero_to_one() {
        let dir = scratch_dir("update-meta-floor-clamp");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        state
            .update_meta("sake", None, None, Some(2.5), Some(-1.0))
            .unwrap()
            .unwrap();

        let entry = state.directory_entry("sake").unwrap();
        assert_eq!(
            entry.dice_floor,
            Some(1.0),
            "an over-range dice_floor must clamp to the ceiling"
        );
        assert_eq!(
            entry.semantic_floor,
            Some(0.0),
            "an under-range semantic_floor must clamp to the floor"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// `update_meta`'s pinned-`ensure_hot`-failure rollback
    /// (`rollback_meta` + `recount_entry`, then `Err`) has no test —
    /// the only existing rollback test targets the sibling `write_meta`
    /// failure arm instead, with its `pinned` call made AFTER
    /// permissions are restored so `ensure_hot` there always succeeds.
    /// Here a cold context with a corrupted image is pinned: the
    /// attempt must fail closed, `meta.pinned` must roll back to
    /// `false` (not strand the context pinned-but-unloadable), and the
    /// budget's `resident_estimate` must stay in sync with that
    /// rollback rather than the failed intermediate state.
    #[test]
    fn update_meta_rolls_back_pinning_when_the_forced_preload_fails() {
        let dir = scratch_dir("update-meta-pin-rollback");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
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
        let entry = state.lookup("sake").unwrap();
        assert!(
            state.evict_entry("sake", &entry),
            "sanity: an unpinned context must evict cleanly"
        );

        let image = image_path(&dir, &file_stem("sake"));
        let mut bytes = fs::read(&image).unwrap();
        assert!(bytes.len() > 8, "sanity: the version byte must exist");
        bytes[8] = 0xFF;
        fs::write(&image, &bytes).unwrap();

        let error = state
            .update_meta("sake", None, Some(true), None, None)
            .expect("the context still exists")
            .expect_err("the forced preload must fail on the corrupt image");
        assert!(!error.to_string().is_empty());

        let after = state.directory_entry("sake").unwrap();
        assert!(
            !after.pinned,
            "a failed forced preload must roll `pinned` back to false, \
             not strand the context pinned yet cold and unloadable"
        );
        assert!(!after.loaded, "it must stay cold, not half-applied");

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

    /// The pinned re-preload's `Err` arm (`lifecycle.rs`, inside
    /// `rename_context_locked`'s tail: `Err(error) => { tracing::warn!
    /// ..."renamed context not preloaded; it stays cold until first
    /// use" }`) has no test — the happy-path test above only proves
    /// the `Ok` arm. Corrupting the image between two boots (rather
    /// than while the context is hot) is required: `drain_entry_for_rename`
    /// re-saves a HOT source's current in-memory state before the
    /// move, which would silently heal an in-place corruption.
    /// Preloading fails at boot instead, leaving "sake" cold with the
    /// corruption intact, so the rename's own re-preload attempt at
    /// the new name hits the same failure.
    #[test]
    fn a_pinned_context_s_rename_survives_a_re_preload_failure_and_stays_cold() {
        let dir = scratch_dir("rename-pinned-repreload-failure");
        {
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
            state.flush_dirty();
        }
        // The version byte — same technique `engine.rs`'s own
        // load-failure tests use.
        let image = image_path(&dir, &file_stem("sake"));
        let mut bytes = fs::read(&image).unwrap();
        assert!(bytes.len() > 8, "sanity: the version byte must exist");
        bytes[8] = 0xFF;
        fs::write(&image, &bytes).unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert_eq!(
            loaded_map(&state).get("sake"),
            Some(&false),
            "sanity: the corrupt pinned image must fail to preload at boot"
        );

        state
            .rename_context("sake", "shochu")
            .expect("the rename itself must still succeed despite the re-preload failure");

        assert!(state.directory_entry("sake").is_none());
        let shochu = state
            .directory_entry("shochu")
            .expect("the new name must answer");
        assert!(
            shochu.pinned,
            "pinned carries over even though it stays cold"
        );
        assert!(
            !shochu.loaded,
            "a pinned context whose re-preload fails must stay cold, \
             not take the whole rename down"
        );
        assert!(!dir.join("sake.ctx").exists());
        assert!(dir.join("shochu.ctx").exists());

        let _ = fs::remove_dir_all(dir);
    }

    /// The schema file family regression: #379 added `{stem}.schema.json`
    /// as `context_files`' tenth (last, best-effort) entry — this
    /// confirms `move_context_files` actually carries it, the same as
    /// every other sidecar, and that the moved sidecar's recorded digest
    /// still matches the moved content so a later boot does not refuse.
    #[test]
    fn rename_context_moves_the_schema_file_and_its_recorded_digest_too() {
        let dir = scratch_dir("rename-context-schema");
        let document =
            br#"{"schema":1,"mode":"off","closed_labels":false,"types":{},"relations":{}}"#;
        let digest = crate::sha256::sha256_hex(document);
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            state.flush_dirty();
        }
        fs::write(schema_path(&dir, "sake"), document).unwrap();
        let meta_file = meta_path(&dir, "sake");
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&meta_file).unwrap()).unwrap();
        value["schema_digest"] = serde_json::json!(digest);
        fs::write(&meta_file, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

        // A fresh boot picks up the hand-planted schema (matching the
        // digest above) before renaming it — `Entry::new`'s
        // `schema_digest` parameter is what this test is really after.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.rename_context("sake", "shochu").unwrap();

        assert!(
            !schema_path(&dir, "sake").exists(),
            "the old stem's schema file must move, not stay behind"
        );
        assert_eq!(fs::read(schema_path(&dir, "shochu")).unwrap(), document);
        drop(state);

        // If the recorded digest had not moved with the content (or had
        // been dropped to `None` along the way), this boot would refuse.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.directory_entry("shochu").is_some());
        drop(state);

        let _ = fs::remove_dir_all(dir);
    }

    /// `delete`'s unlink loop walks `context_files`, so the schema file
    /// — its tenth entry since #379 — must go with the rest of the
    /// family, never left as litter a reused name could later collide
    /// with (see `sweep_stale_stem_files`'s own schema-litter guard).
    #[test]
    fn delete_removes_the_schema_file_with_the_rest_of_the_family() {
        let dir = scratch_dir("delete-context-schema");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state.flush_dirty();
        fs::write(schema_path(&dir, "sake"), b"irrelevant to this test").unwrap();

        state.delete("sake").unwrap().unwrap();

        assert!(!schema_path(&dir, "sake").exists());
        drop(state);
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

        // Regression for issue #561's item 4: `to` ("shochu") is NOT
        // registered — the pivot never landed — so releasing its
        // reservation would let `create("shochu", ...)`'s stale-file
        // sweep (`sweep_stale_stem_files`'s `rename_markers_targeting`
        // scan, the same one `creating_a_context_abandons_a_rename_marker_naming_it_as_destination`
        // exercises live) delete the very marker boot needs to resume
        // this stuck rename, orphaning "sake" past any recovery.
        assert!(
            matches!(
                state.create("shochu", ContextMeta::default()),
                Err(CreateError::AlreadyExists)
            ),
            "a stuck rename must keep blocking create() on BOTH names, not \
             just the source"
        );
        assert!(
            renaming_marker_path(&dir, &file_stem("sake")).exists(),
            "the refused create(to) must not have touched the marker"
        );
        assert!(
            dir.join("sake.ctx").exists(),
            "the refused create(to) must not have touched the old data"
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

    /// `boot_with`'s own resume loop is the other caller of
    /// `rename_in_membership` this issue's fix touches: a membership
    /// rewrite that fails DURING a boot-time resume must keep the
    /// marker for the NEXT boot too, not just the live call's own
    /// rollback. Before the fix the loop removed the marker whenever
    /// `rename.complete` was true, regardless of whether the
    /// membership rewrite it just attempted actually persisted.
    #[test]
    fn a_resumed_renames_membership_rewrite_that_cannot_persist_keeps_the_marker() {
        let dir = scratch_dir("rename-resume-membership-fault");
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
        // The crash-shaped state: the marker survives, nothing has
        // moved yet — `scan_data_dir`'s own resume performs the move
        // during the boot under test.
        fs::write(
            renaming_marker_path(&dir, &file_stem("sake")),
            serde_json::to_vec(&RenameMarker {
                from: "sake".to_string(),
                to: "shochu".to_string(),
            })
            .unwrap(),
        )
        .unwrap();

        // `move_context_files` now routes every one of `context_files`'
        // ten renames through the same injector (#586) — it consults
        // the injector once per family slot regardless of whether that
        // slot's source file exists (a `NotFound` is only known AFTER
        // the call), so the move alone spends exactly `context_files`'
        // length worth of "successes" before the membership rewrite's
        // own `write_group` gets to run. Let all ten land, then fail
        // the very next persistence op — `write_group`'s own stage.
        fail_persistence_ops_after(context_files(&file_stem("sake")).len() as u32);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let past_end = clear_persistence_fault();
        assert!(!past_end, "the group write itself must be what failed");

        assert!(
            state.directory_entry("shochu").is_some(),
            "the move must still land even though the membership rewrite failed"
        );
        assert!(
            renaming_marker_path(&dir, &file_stem("sake")).exists(),
            "a boot-time membership rewrite failure must keep the marker, \
             or the NEXT boot's reconcile_groups sees \"sake\" as a plain \
             dangling reference and drops it instead of resuming the rewrite"
        );
        drop(state);

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert_eq!(
            state.group("drinks").unwrap().contexts,
            BTreeSet::from(["shochu".to_string()]),
            "the retried resume must finish what the first boot could not"
        );
        assert!(!renaming_marker_path(&dir, &file_stem("sake")).exists());

        let _ = fs::remove_dir_all(&dir);
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

    /// The counterpart to the happy path above: a destination-targeting
    /// marker that FAILS to unlink for a real reason (not `NotFound`)
    /// must fail the whole sweep, not be silently swallowed. Every
    /// other sweep-failure test targets a different loop in
    /// `sweep_stale_stem_files` (`a_marker_that_cannot_be_removed_fails_the_stem_sweep`
    /// hits the stale-paths loop; the import-marker tests hit the
    /// third loop) — none exercises THIS one. Calls
    /// `sweep_stale_stem_files` directly rather than through `create`
    /// so the injected fault can be counted precisely: `FreshCreate`
    /// mode's eleven always-checked stale paths (none of which exist
    /// for a brand new "sake" stem) must all resolve as ordinary
    /// `NotFound` no-ops before the twelfth call — the planted
    /// targeting marker — is the one made to fail.
    #[test]
    fn sweep_stale_stem_files_reports_a_real_removal_failure_on_a_destination_targeting_marker() {
        let dir = scratch_dir("sweep-targeting-marker-removal-fault");
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

        fail_persistence_ops_after(11);
        let error = state
            .sweep_stale_stem_files("sake", &file_stem("sake"), StemSweep::FreshCreate)
            .unwrap_err();
        assert!(
            !clear_persistence_fault(),
            "sanity: the injected failure must land on the targeting-marker \
             removal, not somewhere earlier or never at all: {error:?}"
        );
        assert!(
            renaming_marker_path(&dir, &file_stem("beer")).exists(),
            "the marker must still be there — the injected failure stood \
             in for the real unlink, so nothing actually removed it"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// Boot's straggler contract, isolated: `ResumedRename`'s `landed`
    /// (the pivot moved) and `complete` (the WHOLE move finished) are
    /// deliberately independent booleans (`paths.rs`'s own doc), and
    /// `boot_with`'s resume loop keys membership on `landed` alone
    /// while keying marker retraction on `complete` alone. Every other
    /// boot-resume test either has no group to rewrite
    /// (`delete_clears_a_stuck_rename_marker_at_its_own_stem`, pivot
    /// blocked so `landed` is false too) or completes cleanly (the
    /// happy-path resume tests). Here the pivot moves but a sidecar
    /// (`wal_path`) stays blocked: membership must still follow the
    /// pivot's new name, and the marker must still survive for the
    /// next boot to finish the straggler.
    #[test]
    fn a_boot_resume_whose_pivot_lands_but_a_sidecar_sticks_still_rewrites_membership() {
        let dir = scratch_dir("boot-resume-straggler-membership");
        {
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
            state.flush_dirty();
            state
                .create_group(
                    "drinks",
                    String::new(),
                    BTreeSet::from(["sake".to_string()]),
                    BTreeSet::new(),
                )
                .unwrap();
        }
        // Block the DESTINATION's wal lane — a post-pivot sidecar
        // (`context_files`'s index 8, not 0) — so the resume's own
        // `move_context_files` moves the pivot and every earlier file
        // successfully, then fails here and stops treating the rest as
        // best-effort. No manual pivot move: the resume performs it.
        fs::create_dir_all(wal_path(&dir, &file_stem("shochu"))).unwrap();
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

        assert!(
            dir.join("shochu.ctx").exists(),
            "sanity: the pivot must have landed"
        );
        assert!(
            dir.join("sake.wal.jsonl").exists(),
            "sanity: the blocked sidecar must still sit at the old stem"
        );
        assert_eq!(
            state.group("drinks").unwrap().contexts,
            BTreeSet::from(["shochu".to_string()]),
            "membership must follow the pivot's new name even though \
             the move as a whole is incomplete"
        );
        assert!(
            renaming_marker_path(&dir, &file_stem("sake")).exists(),
            "the marker must survive for the next boot to finish the \
             straggling sidecar — only `complete`, not `landed`, retires it"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// `delete`'s counterpart to `creating_a_context_abandons_a_rename_marker_at_its_own_stem`:
    /// a stuck rename's marker sits at ITS OWN stem too, and `delete`
    /// must strip it just as `create_files` does — reachable because
    /// `pending.renames` (the live call's own in-memory guard) is empty
    /// again after a restart, while the marker itself survives on disk.
    /// Before the fix, `delete("sake")` left the marker orphaned: the
    /// next boot's resume-sweep would try to move a family that no
    /// longer exists onto "shochu", registering nothing under either
    /// name.
    #[test]
    fn delete_clears_a_stuck_rename_marker_at_its_own_stem() {
        let dir = scratch_dir("delete-clears-source-marker");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            // Block the pivot exactly as
            // `a_rename_stuck_past_the_point_of_no_return_...` does, so
            // the marker survives past this call AND past the reboot
            // just below.
            fs::create_dir(dir.join(format!("{}.ctx", file_stem("shochu")))).unwrap();
            let error = state.rename_context("sake", "shochu").unwrap_err();
            assert!(matches!(error, RenameContextError::Io(_)));
            assert!(renaming_marker_path(&dir, &file_stem("sake")).exists());
        }
        // A restart clears `pending.renames` (process memory only) but
        // not the marker; the pivot is still blocked, so boot's own
        // resume attempt fails the same way and leaves both the marker
        // and "sake" itself registered (the pivot never moved).
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.directory_entry("sake").is_some());
        assert!(renaming_marker_path(&dir, &file_stem("sake")).exists());

        state.delete("sake").unwrap().unwrap();
        assert!(
            !renaming_marker_path(&dir, &file_stem("sake")).exists(),
            "delete must clear a stuck rename marker sitting at its own stem, \
             the same leftover create_files already sweeps for a reused name"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// `delete`'s counterpart to
    /// `creating_a_context_abandons_a_rename_marker_naming_it_as_destination`:
    /// a stuck rename's marker sits under the SOURCE's stem but names
    /// THIS context as its destination — a stem `delete("sake")` cannot
    /// derive from "sake" alone, so it must scan for markers naming it,
    /// same as `create`'s sweep does for a reused name.
    #[test]
    fn delete_clears_a_rename_marker_naming_it_as_destination() {
        let dir = scratch_dir("delete-clears-destination-marker");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("beer", ContextMeta::default()).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        // A stuck rename from an unrelated earlier "beer" naming "sake"
        // as its destination — written directly, since going through
        // `rename_context` or `create("sake", ...)` would have already
        // swept it (that is what the two `creating_a_context_...` tests
        // above cover).
        fs::write(
            renaming_marker_path(&dir, &file_stem("beer")),
            serde_json::to_vec(&RenameMarker {
                from: "beer".to_string(),
                to: "sake".to_string(),
            })
            .unwrap(),
        )
        .unwrap();

        state.delete("sake").unwrap().unwrap();
        assert!(
            !renaming_marker_path(&dir, &file_stem("beer")).exists(),
            "delete must clear a rename marker naming it as the destination, \
             or the next boot's resume-sweep would move 'beer' onto the name \
             just reported deleted"
        );
        assert!(
            state.directory_entry("beer").is_some(),
            "the untouched, unrelated source context must survive"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// A stuck rename marker sitting at the deleted context's own stem
    /// that CANNOT be removed (here: a directory wearing the marker's
    /// name, same technique as `a_marker_that_cannot_be_removed_fails_the_stem_sweep`)
    /// must surface through the delete, not be silently treated as
    /// already gone — the marker survives, unreported, for boot to
    /// stumble over.
    #[test]
    fn a_delete_that_cannot_clear_its_own_rename_marker_reports_it() {
        let dir = scratch_dir("delete-stuck-rename-marker");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        fs::create_dir_all(renaming_marker_path(&dir, &file_stem("sake"))).unwrap();

        assert!(
            matches!(state.delete("sake"), Some(Err(DeleteError::Io(_)))),
            "an unremovable rename marker must surface through the delete"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// The destination-side twin of the test above: a stuck rename
    /// marker naming "sake" as its destination sits under a DIFFERENT
    /// stem ("beer"'s) and is found by `rename_markers_targeting`
    /// rather than the direct `renaming_marker_path` lookup. Unlike the
    /// own-stem case, this marker must stay a valid, readable file for
    /// the scan to find it at all (an unreadable one is simply
    /// filtered out, not surfaced as a removal failure) — so the
    /// unremovable-marker fault is injected instead of blocked with a
    /// directory. Swept exhaustively rather than pinned to one op
    /// index: `delete`'s own op count (the `.deleted` marker write,
    /// the ten-file family sweep, the own-stem marker check) is an
    /// implementation detail this test must not hardcode.
    #[test]
    fn a_delete_that_cannot_clear_a_rename_marker_naming_it_as_destination_reports_it() {
        let mut hit_the_targeting_marker = false;
        for failure in 0..32 {
            let dir = scratch_dir(&format!("delete-stuck-destination-marker-{failure}"));
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("beer", ContextMeta::default()).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            fs::write(
                renaming_marker_path(&dir, &file_stem("beer")),
                serde_json::to_vec(&RenameMarker {
                    from: "beer".to_string(),
                    to: "sake".to_string(),
                })
                .unwrap(),
            )
            .unwrap();

            fail_persistence_ops_after(failure);
            let outcome = state.delete("sake");
            let past_end = clear_persistence_fault();
            let marker_survived = renaming_marker_path(&dir, &file_stem("beer")).exists();
            if marker_survived {
                hit_the_targeting_marker = true;
                assert!(
                    matches!(outcome, Some(Err(DeleteError::Io(_)))),
                    "failure at persistence step {failure}: the destination \
                     marker survived unremoved, so the delete must report it, \
                     not silently succeed"
                );
            }
            drop(state);
            let _ = fs::remove_dir_all(&dir);
            if past_end {
                break;
            }
        }
        assert!(
            hit_the_targeting_marker,
            "the sweep never actually failed the destination marker's own removal"
        );
    }

    /// Regression for issue #561's item 5: `rename_in_membership`'s own
    /// doc claims "the next boot's resume retries" a `write_group`
    /// failure, but before the fix `rename_context_locked` deleted the
    /// marker unconditionally right after calling it — with no marker
    /// to resume from, the next boot's `reconcile_groups` would see
    /// "sake" (gone from the registry, the rename landed) as a plain
    /// dangling reference and drop it, losing the membership for good
    /// instead of carrying it to "shochu".
    ///
    /// Swept over every persistence fault point, following the same
    /// exhaustive-sweep shape as `every_context_delete_persistence_failure_recovers_at_boot`:
    /// wherever the fault lands, the group must never end up empty (the
    /// member lost) once the renamed context is registered — either
    /// the rename never reached the point of no return (group still
    /// names "sake", which still exists), or a single boot resume
    /// finishes rewriting membership to "shochu".
    #[test]
    fn a_rename_whose_membership_rewrite_cannot_persist_keeps_its_marker() {
        let mut exhausted = false;
        for failure in 0..64 {
            let dir = scratch_dir(&format!("rename-membership-fault-{failure}"));
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

            fail_persistence_ops_after(failure);
            let outcome = state.rename_context("sake", "shochu");
            let past_end = clear_persistence_fault();
            drop(state);

            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            let sake = state.directory_entry("sake");
            let shochu = state.directory_entry("shochu");
            let members = state.group("drinks").unwrap().contexts;
            match (sake.is_some(), shochu.is_some()) {
                (true, false) => assert_eq!(
                    members,
                    BTreeSet::from(["sake".to_string()]),
                    "failure at persistence step {failure} ({outcome:?}): the \
                     rename never landed, so membership must be untouched"
                ),
                (false, true) => assert_eq!(
                    members,
                    BTreeSet::from(["shochu".to_string()]),
                    "failure at persistence step {failure} ({outcome:?}): the \
                     rename landed, so a boot resume must have finished \
                     rewriting membership — an empty set means the marker \
                     was deleted before the rewrite could be retried, \
                     losing the member for good"
                ),
                other => panic!(
                    "failure at persistence step {failure}: the context must \
                     land under exactly one name, not {other:?}"
                ),
            }
            drop(state);
            let _ = fs::remove_dir_all(&dir);

            if past_end {
                assert!(outcome.is_ok());
                exhausted = true;
                break;
            }
        }
        assert!(exhausted, "context rename exceeded the sweep bound");
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
        // The `from == to` short-circuit must not mask a NotFound: a
        // self-rename of a name that never existed is still a refusal.
        assert!(matches!(
            state.rename_context("missing", "missing"),
            Err(RenameContextError::NotFound)
        ));

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

    fn valid_schema_document() -> schema::SchemaDocument {
        schema::SchemaDocument {
            schema: schema::SCHEMA_VERSION,
            mode: schema::SchemaMode::Strict,
            closed_labels: false,
            types: BTreeMap::from([("Brewery".to_string(), schema::TypeDef::default())]),
            relations: BTreeMap::new(),
        }
    }

    /// The core #380 contract: a `PUT` bumps the `config` revision,
    /// persists the digest to the sidecar, echoes `schema_mode`, and
    /// re-mints `cache_identity` (ADR 0009 §5.2) — exactly what
    /// `bump_config_revision` already does for `dice_floor`, plus the
    /// identity re-mint.
    #[test]
    fn put_schema_bumps_config_revision_persists_the_digest_and_mints_a_fresh_cache_identity() {
        let dir = scratch_dir("put-schema-basic");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();

        let before_revision = state.directory_entry("sake").unwrap().revision.config;
        let before_identity = state.lookup("sake").unwrap().inner.read().cache_identity;

        let installed = schema::install(valid_schema_document()).unwrap();
        let document = state.put_schema("sake", installed).unwrap().unwrap();
        assert_eq!(document.mode, schema::SchemaMode::Strict);

        let entry = state.directory_entry("sake").unwrap();
        assert_eq!(entry.revision.config, before_revision + 1);
        assert_eq!(entry.schema_mode.as_deref(), Some("strict"));

        let after_identity = state.lookup("sake").unwrap().inner.read().cache_identity;
        assert_ne!(
            before_identity, after_identity,
            "a schema PUT must re-mint cache_identity so a retrieval-cache key minted \
             before it becomes unreachable (ADR 0009 §5.2)"
        );

        let sidecar = read_meta_file(&dir, &file_stem("sake"));
        let bytes = fs::read(schema_path(&dir, &file_stem("sake"))).unwrap();
        assert_eq!(
            sidecar.schema_digest.as_deref(),
            Some(crate::sha256::sha256_hex(&bytes).as_str()),
            "the recorded digest must match the bytes actually on disk"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// "A PUT that changes nothing bumps nothing" — `update_meta`'s own
    /// idempotent-update discipline, mirrored here so a retried or
    /// duplicate `PUT` of the identical document never churns the
    /// retrieval cache.
    #[test]
    fn a_repeated_put_of_the_same_document_bumps_nothing() {
        let dir = scratch_dir("put-schema-noop");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();

        let installed = schema::install(valid_schema_document()).unwrap();
        state.put_schema("sake", installed).unwrap().unwrap();
        let revision_after_first = state.directory_entry("sake").unwrap().revision.config;
        let identity_after_first = state.lookup("sake").unwrap().inner.read().cache_identity;

        let installed_again = schema::install(valid_schema_document()).unwrap();
        state.put_schema("sake", installed_again).unwrap().unwrap();

        let entry = state.directory_entry("sake").unwrap();
        assert_eq!(
            entry.revision.config, revision_after_first,
            "identical content must not bump the revision"
        );
        assert_eq!(
            state.lookup("sake").unwrap().inner.read().cache_identity,
            identity_after_first,
            "identical content must not re-mint cache_identity"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// ADR 0009 §6.3 guard 3's migration-boundary counterpart: an
    /// already-persisted `label_alias` resolving to the reserved type
    /// label refuses the `PUT` outright — nothing written, not even
    /// the sidecar digest.
    #[test]
    fn put_schema_refuses_when_a_persisted_label_alias_resolves_to_the_reserved_type_label() {
        let dir = scratch_dir("put-schema-reserved-alias");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        // Legal today — guard 1: `schema:type` is an ordinary label in
        // a context with no installed schema — and it interns the
        // label id `add_label_alias`'s canonical must resolve against.
        state
            .add_associations(
                "sake",
                vec![assoc_op(
                    "蔵",
                    schema::SCHEMA_TYPE_LABEL,
                    "Brewery",
                    1.0,
                    Some("a.md"),
                )],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state
            .add_aliases(
                "sake",
                &BTreeMap::new(),
                &BTreeMap::from([("種別".to_string(), schema::SCHEMA_TYPE_LABEL.to_string())]),
            )
            .unwrap()
            .unwrap();

        let installed = schema::install(valid_schema_document()).unwrap();
        let error = state.put_schema("sake", installed).unwrap().unwrap_err();
        assert!(
            matches!(&error, PutSchemaError::ReservedAlias(alias) if alias == "種別"),
            "{error:?}"
        );
        assert!(
            !schema_path(&dir, &file_stem("sake")).exists(),
            "a refused PUT must not write the schema file"
        );
        assert_eq!(
            read_meta_file(&dir, &file_stem("sake")).schema_digest,
            None,
            "a refused PUT must not touch the sidecar's digest"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// ADR 0009 §5.2's write order, proven rather than merely asserted
    /// in a comment: force the schema file's own write to fail right
    /// after the sidecar's `write_meta` already landed (2 persistence
    /// checkpoints — `write_meta`'s one `write_atomic` call's stage +
    /// commit — succeed, then the schema file's own stage fails). The
    /// sidecar must already be durable by the time the schema file
    /// write is even attempted, and the best-effort restore this
    /// failure triggers must bring the sidecar back to its exact
    /// pre-PUT state (the restore's own write is unfaulted, since the
    /// injector is single-shot).
    #[test]
    fn a_schema_file_write_failure_rolls_back_the_sidecar_after_the_revision_already_landed() {
        let dir = scratch_dir("put-schema-write-order");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        let before = read_meta_file(&dir, &file_stem("sake"));

        let installed = schema::install(valid_schema_document()).unwrap();
        fail_persistence_ops_after(2);
        let error = state.put_schema("sake", installed).unwrap().unwrap_err();
        let exhausted = clear_persistence_fault();
        assert!(!exhausted, "the fault must have fired, not merely run out");
        assert!(matches!(error, PutSchemaError::Io(_)), "{error:?}");

        assert!(
            !schema_path(&dir, &file_stem("sake")).exists(),
            "the schema file must never land when its own write fails"
        );
        let after = read_meta_file(&dir, &file_stem("sake"));
        assert_eq!(after.schema_digest, before.schema_digest);
        assert_eq!(after.revision.config, before.revision.config);

        let entry = state.lookup("sake").unwrap();
        let inner = entry.inner.read();
        assert_eq!(inner.schema_digest, before.schema_digest);
        assert_eq!(inner.config_revision, before.revision.config);
        assert!(inner.schema.is_none());
        drop(inner);

        let _ = fs::remove_dir_all(&dir);
    }

    /// `schema_of`'s two direct cases: a schema-free context answers
    /// `Ok(None)` without touching disk, and a missing context answers
    /// the outer `None` — both without a `PUT` ever having run.
    #[test]
    fn schema_of_reports_a_schema_free_context_and_a_missing_one() {
        let dir = scratch_dir("schema-of-absent");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();

        assert!(
            state.schema_of("sake").unwrap().unwrap().is_none(),
            "a fresh context has no schema"
        );
        assert!(state.schema_of("nope").is_none());
        assert!(
            state
                .put_schema("nope", schema::install(valid_schema_document()).unwrap())
                .is_none(),
            "a PUT against a context that never existed must answer the outer None, \
             not a PutSchemaError"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// `lookup` — the first step of both `schema_of` and `put_schema` —
    /// already answers `None` for a name `delete` has removed from the
    /// registry, so both report the outer `None` for a deleted context
    /// exactly like a never-created one, without either method needing
    /// its own tombstone-detection logic beyond the shared
    /// `read_unless_deleted`/`lock_unless_deleted` gate every other
    /// post-lookup operation in this file already goes through.
    #[test]
    fn schema_of_and_put_schema_report_the_outer_none_for_a_deleted_context() {
        let dir = scratch_dir("schema-of-deleted");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state.delete("sake").unwrap().unwrap();

        assert!(state.schema_of("sake").is_none());
        assert!(
            state
                .put_schema("sake", schema::install(valid_schema_document()).unwrap())
                .is_none()
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// The lazy-resolution path `EntryInner::schema`'s doc names: a
    /// rename's freshly registered destination entry starts with
    /// `schema: None` even though `schema_digest` carried over from
    /// the sidecar (§2 of #380's plan) — `schema_of` must resolve it
    /// rather than misreport the schema as absent.
    #[test]
    fn schema_of_lazily_resolves_after_a_rename_carried_the_digest_but_not_the_schema() {
        let dir = scratch_dir("schema-of-rename");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        let installed = schema::install(valid_schema_document()).unwrap();
        state.put_schema("sake", installed).unwrap().unwrap();

        state.rename_context("sake", "shochu").unwrap();
        assert!(
            state
                .lookup("shochu")
                .unwrap()
                .inner
                .read()
                .schema
                .is_none(),
            "the freshly registered entry must not resolve the schema up front"
        );

        let resolved = state
            .schema_of("shochu")
            .unwrap()
            .unwrap()
            .expect("the schema must resolve, not read as absent");
        assert_eq!(resolved.document().mode, schema::SchemaMode::Strict);

        let _ = fs::remove_dir_all(&dir);
    }

    /// `hidden_label`'s `Err` arm (`Err(_) => Some(SCHEMA_TYPE_LABEL)`)
    /// has no test — `hidden_label` is never called from any test in
    /// the suite. Reusing the fixture above (a rename carries the
    /// digest but not the schema, so `schema_of` must call
    /// `ensure_hot` to resolve it), a corrupted image makes that
    /// `ensure_hot` call fail, and `schema_of` itself returns `Err`.
    /// `hidden_label` must fail CLOSED on that — report hidden, the
    /// same as a schema actually present — rather than let a
    /// resolution failure silently unhide a schema-gated context.
    #[test]
    fn hidden_label_fails_closed_when_schema_resolution_errors() {
        let dir = scratch_dir("hidden-label-schema-err");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        let installed = schema::install(valid_schema_document()).unwrap();
        state.put_schema("sake", installed).unwrap().unwrap();

        state.rename_context("sake", "shochu").unwrap();
        assert!(
            state
                .lookup("shochu")
                .unwrap()
                .inner
                .read()
                .schema
                .is_none(),
            "sanity: the freshly registered entry must not resolve the schema up front"
        );

        let image = image_path(&dir, &file_stem("shochu"));
        let mut bytes = fs::read(&image).unwrap();
        assert!(bytes.len() > 8, "sanity: the version byte must exist");
        bytes[8] = 0xFF;
        fs::write(&image, &bytes).unwrap();

        assert!(
            matches!(state.schema_of("shochu"), Some(Err(_))),
            "sanity: the corrupt image must make schema_of itself fail"
        );
        assert_eq!(
            state.hidden_label("shochu"),
            Some(schema::SCHEMA_TYPE_LABEL),
            "a schema-resolution failure must report hidden, not \
             silently unhide a schema-gated context"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// `rollback_rename`'s NotFound arm, isolated: a marker that is
    /// already gone (nothing else in this call ever wrote one) must
    /// still count as retracted and report `RolledBack`, not fall
    /// through to the general failure arm and report `Stuck` — that
    /// would strand `from_stem`'s reservation forever over a marker
    /// that was never there to resume from. Called directly (a
    /// private helper, same module) rather than through the whole
    /// `rename_context_locked` dance: reaching this exact arm live
    /// would require something else deleting the marker out from under
    /// the rollback between its write and this retraction, which nothing
    /// in this single-threaded call graph does.
    #[test]
    fn a_rollback_finds_its_marker_already_gone_and_still_rolls_back() {
        let dir = scratch_dir("rollback-marker-notfound");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let marker = renaming_marker_path(&dir, "ghost");
        assert!(!marker.exists());
        let outcome = state.rollback_rename(
            "ghost",
            &marker,
            None,
            RenameContextError::Io(io::Error::other("unrelated failure")),
        );
        assert!(
            matches!(outcome, RenameOutcome::RolledBack(_)),
            "a marker already absent must count as retracted"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// Regression for issue #561's item 3/6: a rollback that cannot
    /// retract its OWN marker (the choke point every other unlink in
    /// this module goes through, `remove_persisted_file`, fails) must
    /// not report `RolledBack` — that would free `from` while a
    /// `.renaming` marker still durably claims it, letting a client's
    /// `create(from)` sweep the marker (and whatever the marker's
    /// resume would have carried over) away as ordinary stale
    /// leftovers. It must report `Stuck` instead, exactly like a
    /// failure past the point of no return.
    #[test]
    fn a_rollback_that_cannot_retract_its_marker_stays_stuck() {
        let dir = scratch_dir("rename-rollback-stuck-marker");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();

        // Block the destination sweep deterministically (a directory
        // where `sweep_stale_stem_files` expects to unlink a plain
        // file) — the same technique as
        // `a_marker_that_cannot_be_removed_fails_the_stem_sweep`, just
        // aimed at the rename's own sweep instead of create's.
        fs::create_dir_all(wal_path(&dir, &file_stem("shochu"))).unwrap();
        // Then, with the fault injector, fail exactly the 4th
        // persistence op: #1-#2 are the rename marker's own
        // `write_atomic` (stage + commit), #3 is the sweep's blocked
        // unlink attempt (the injector lets it through; the real
        // directory blocker is what fails it), and #4 is
        // `rollback_rename`'s own `remove_persisted_file(&marker)` —
        // the one this test targets.
        fail_persistence_ops_after(3);
        let error = state.rename_context("sake", "shochu").unwrap_err();
        let past_end = clear_persistence_fault();
        assert!(
            !past_end,
            "the marker retraction itself must be what failed"
        );
        assert!(matches!(error, RenameContextError::Io(_)), "{error:?}");

        assert!(
            renaming_marker_path(&dir, &file_stem("sake")).exists(),
            "the marker that could not be retracted must survive"
        );
        assert!(
            state.directory_entry("sake").is_some(),
            "the source is still registered in memory — it was never tombstoned"
        );
        // `create("sake", ...)` is NOT the right probe here: "sake"
        // never left the registry (the failure is before the point of
        // no return), so it already reports `AlreadyExists` on that
        // basis alone, whether or not the reservation below held. The
        // real question is whether `pending.renames` still reserves
        // BOTH names — checked directly by attempting the exact same
        // rename again: a `RolledBack` bug would have freed both names
        // and let this through to retry the disk work (still blocked
        // by the same directory, but for a DIFFERENT reason); a
        // correct `Stuck` refuses before touching disk at all.
        assert!(
            matches!(
                state.rename_context("sake", "shochu"),
                Err(RenameContextError::Busy)
            ),
            "a rollback stuck on its own marker must keep BOTH names reserved \
             in pending.renames, exactly like a failure past the point of no return"
        );

        // Not a permanent loss: clearing the obstruction and letting a
        // fresh attempt (or a boot resume) run again resolves it.
        fs::remove_dir(wal_path(&dir, &file_stem("shochu"))).unwrap();
        drop(state);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.directory_entry("sake").is_none());
        assert!(state.directory_entry("shochu").is_some());
        assert!(!renaming_marker_path(&dir, &file_stem("sake")).exists());

        let _ = fs::remove_dir_all(&dir);
    }

    /// A stale import marker that cannot be removed (here: a DIRECTORY
    /// wearing the marker name) fails the stem sweep — and with it the
    /// create — rather than silently leaving a marker boot will keep
    /// reporting as a torn import.
    #[test]
    fn a_marker_that_cannot_be_removed_fails_the_stem_sweep() {
        let dir = scratch_dir("sweep-stuck-marker");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let marker = dir.join(format!(
            "{}.batch.{}",
            file_stem("sake"),
            crate::registry::paths::IMPORT_MARKER_EXTENSION
        ));
        fs::create_dir_all(&marker).unwrap();
        assert!(
            state.create("sake", ContextMeta::default()).is_err(),
            "an unremovable marker must fail the create"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A delete that cannot clear the context's import marker reports
    /// the failure — the marker survives beside the tombstone and boot
    /// must get the chance to finish the job.
    #[test]
    fn a_delete_that_cannot_clear_its_import_marker_reports_it() {
        let dir = scratch_dir("delete-stuck-marker");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let marker = dir.join(format!(
            "{}.batch.{}",
            file_stem("sake"),
            crate::registry::paths::IMPORT_MARKER_EXTENSION
        ));
        fs::create_dir_all(&marker).unwrap();
        assert!(
            matches!(state.delete("sake"), Some(Err(DeleteError::Io(_)))),
            "an unremovable marker must surface through the delete"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// EVERY pending-operation membership makes a rename Busy on its
    /// own — either name, any of the three sets.
    #[test]
    fn any_pending_membership_alone_makes_a_rename_busy() {
        let dir = scratch_dir("rename-busy-guards");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("from", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        type Pick = fn(&mut PendingNames) -> &mut std::collections::HashSet<String>;
        let deletes: Pick = |pending| &mut pending.deletes;
        let creates: Pick = |pending| &mut pending.creates;
        let renames: Pick = |pending| &mut pending.renames;
        let scenarios: [(&str, Pick); 5] = [
            ("from", deletes),
            ("to", deletes),
            ("to", creates),
            ("from", renames),
            ("to", renames),
        ];
        for (name, pick) in scenarios {
            pick(&mut state.0.pending.lock()).insert(name.to_string());
            assert!(
                matches!(
                    state.rename_context("from", "to"),
                    Err(RenameContextError::Busy)
                ),
                "a pending {name} entry alone must refuse the rename"
            );
            pick(&mut state.0.pending.lock()).remove(name);
        }

        let _ = fs::remove_dir_all(&dir);
    }

    /// Regression for issue #561's item 8: a name mid-rename is a
    /// refusal, not a failure — the context is untouched, so `delete`
    /// must report [`DeleteError::MidRename`] distinctly from
    /// [`DeleteError::Io`] (the API layer maps the two very
    /// differently: a 409 with no audit line, versus a 500 with one —
    /// see `api::contexts::delete_context`).
    #[test]
    fn a_mid_rename_delete_reports_mid_rename_not_io() {
        let dir = scratch_dir("delete-mid-rename");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();

        // Same direct-seed technique as
        // `any_pending_membership_alone_makes_a_rename_busy`: a real
        // concurrent rename would hold this for a window too narrow to
        // land a second request on reliably, so this is the
        // deterministic way to exercise the exact same in-memory guard
        // `rename_context` itself takes.
        state.0.pending.lock().renames.insert("sake".to_string());
        assert!(
            matches!(state.delete("sake"), Some(Err(DeleteError::MidRename))),
            "a mid-rename delete must be refused as MidRename, not attempted"
        );
        assert!(
            state.directory_entry("sake").is_some(),
            "the refused delete must not have touched the context"
        );

        state.0.pending.lock().renames.remove("sake");
        let _ = fs::remove_dir_all(&dir);
    }

    /// A meta or schema save that fails must roll the config revision
    /// back to exactly where it stood — the served content never
    /// changed, so neither may the revision.
    #[test]
    #[cfg(unix)]
    fn a_failed_meta_or_schema_save_rolls_the_config_revision_back() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("config-rollback");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state.flush_dirty();
        let config = state.context_revision("sake").unwrap().config;

        let lock_down = || {
            let mut perms = fs::metadata(&dir).unwrap().permissions();
            perms.set_mode(0o555);
            fs::set_permissions(&dir, perms).unwrap();
        };
        let restore = || {
            let mut perms = fs::metadata(&dir).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&dir, perms).unwrap();
        };

        lock_down();
        let outcome = state.update_meta("sake", None, None, None, Some(0.9));
        restore();
        assert!(matches!(outcome, Some(Err(_))));
        assert_eq!(
            state.context_revision("sake").unwrap().config,
            config,
            "a failed meta save must leave the revision untouched"
        );

        let installed = schema::install(valid_schema_document()).unwrap();
        lock_down();
        let outcome = state.put_schema("sake", installed);
        restore();
        assert!(matches!(outcome, Some(Err(_))));
        assert_eq!(
            state.context_revision("sake").unwrap().config,
            config,
            "a failed schema save must leave the revision untouched"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// A rename whose PIVOT move fails moves NOTHING: the image is
    /// renamed first exactly so a failure there aborts before any
    /// sibling file leaves the old stem.
    #[test]
    fn a_failed_pivot_rename_moves_nothing() {
        let dir = scratch_dir("rename-pivot-fail");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state.flush_dirty();
        assert!(meta_path(&dir, &file_stem("sake")).exists());
        // A directory where the target image belongs makes the pivot
        // rename fail deterministically.
        fs::create_dir_all(image_path(&dir, &file_stem("shochu"))).unwrap();

        assert!(state.rename_context("sake", "shochu").is_err());
        assert!(
            meta_path(&dir, &file_stem("sake")).exists(),
            "a failed pivot must leave every sibling under the old stem"
        );
        assert!(
            !meta_path(&dir, &file_stem("shochu")).exists(),
            "no sibling may land under the new stem"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
