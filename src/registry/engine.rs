use super::*;

impl AppState {
    /// Persists a dirty resident index. Derived data: a failed save
    /// re-marks and warns — the next tick retries, and the worst case
    /// is one re-tokenization on some future load. The fence keeps a
    /// racing delete from finding its file recreated.
    fn flush_bm25(&self, name: &str, entry: &Entry) {
        // A replica's rebuilt index stays memory-only: its sidecar is
        // the manifest's bytes, and a local rewrite would only diverge
        // the cache into refetch churn on the next tailed diff.
        if self.is_replica() {
            return;
        }
        if !entry.bm25_dirty.swap(false, Ordering::Relaxed) {
            return;
        }
        let Some(_fence) = entry.read_unless_deleted() else {
            return;
        };
        let bytes = {
            let guard = entry.bm25.read();
            match &*guard {
                Some(index) => index.to_bytes(),
                // Dropped since (eviction): its own save path ran.
                None => return,
            }
        };
        if let Err(error) = write_atomic(&bm25_path(&self.0.data_dir, &file_stem(name)), &bytes) {
            entry.bm25_dirty.store(true, Ordering::Relaxed);
            tracing::warn!("BM25 index for '{name}' not persisted (will retry): {error}");
        }
    }
}

impl AppState {
    /// Persists every dirty context and returns the names it flushed —
    /// the periodic flusher feeds those into the auto embedding refresh
    /// when that is enabled. Called once more on graceful shutdown; a
    /// failed save is retried on the next tick (the entry stays dirty).
    pub fn flush_dirty(&self) -> Vec<String> {
        let mut flushed = Vec::new();
        for (name, entry) in self.snapshot() {
            self.flush_bm25(&name, &entry);
            if self.flush_entry(&name, &entry) {
                flushed.push(name);
            }
        }
        // After the images land: every caller of this (the flusher
        // tick, POST /flush, the shutdown sweep) is exactly when "the
        // sizes on disk changed" — refresh the per-context disk gauges
        // here (a no-op while nothing reads it — neither the gauges
        // nor a declared storage quota) so a scrape never has to.
        self.refresh_disk_usage();
        flushed
    }

    /// One entry's flush. Split out of [`AppState::flush_dirty`] so the
    /// delete-race regression test can drive the exact window a real
    /// flusher hits — snapshot taken before a delete, entry lock taken
    /// after it.
    ///
    /// The image's disk work runs with the entry UNLOCKED: serialize a
    /// consistent snapshot under the lock, stage it (write + fsync,
    /// the megabytes half) without the lock, then re-take the lock to
    /// publish. Readers and writers of the context proceed while the
    /// bytes land; before this, every flush stalled them for the whole
    /// write.
    pub(crate) fn flush_entry(&self, name: &str, entry: &Entry) -> bool {
        // Skip a clean entry without locking it (the flusher's one
        // sanctioned lock-free read of `dirty`).
        if !entry.dirty.load(Ordering::Relaxed) {
            return false;
        }
        let (bytes, meta, stats, watermark, generation) = {
            let mut guard = entry.inner.write();
            let inner = &mut *guard;
            // Claim the flush UNDER the lock. `flushing` gates concurrent
            // flushers (a tick against the shutdown flush) so the same
            // image is never staged twice, and — set before `dirty` is
            // cleared, both under this lock — it tells a racing eviction
            // "a flush of this entry is in flight, its bytes are not on
            // disk yet, persist me before dropping me." Clearing `dirty`
            // here (not out of the lock, as before) is what makes that
            // hand-off atomic: an evict that locks next sees `flushing`
            // set, never a bare "clean". A write that lands while we stage
            // re-sets `dirty`, so nothing is lost even with the WAL off,
            // where `wal_seq` does not move to flag the race.
            if entry.flushing.swap(true, Ordering::Relaxed) {
                return false;
            }
            entry.dirty.store(false, Ordering::Relaxed);
            let watermark = inner.wal_seq - 1;
            // Cold has nothing to write; Deleted must write nothing —
            // that snapshot predates a delete and the files are gone
            // for good.
            let Slot::Hot(context) = &mut inner.slot else {
                entry.flushing.store(false, Ordering::Relaxed);
                return false;
            };
            // The image about to be written reflects everything logged
            // so far: bake that in as the watermark, and those WAL
            // records are replay-inert even if truncation below never
            // happens (crash, unwritable file — doesn't matter).
            context.set_applied_seq(watermark);
            let stats = ContextStats::of(context);
            (
                context.to_bytes(),
                inner.meta.clone(),
                stats,
                watermark,
                inner.image_generation,
            )
        };

        let stem = file_stem(name);
        let image = image_path(&self.0.data_dir, &stem);
        let staged = match stage_bytes(&image, &bytes, false) {
            Ok(staged) => staged,
            Err(error) => {
                tracing::warn!("flush of context '{name}' failed (will retry): {error}");
                entry.dirty.store(true, Ordering::Relaxed);
                entry.flushing.store(false, Ordering::Relaxed);
                self.0.metrics.record_flush(name, false);
                return false;
            }
        };

        // Publication goes back under the lock: files for a name are
        // only ever created while holding its entry lock — the
        // tombstone invariant — so a delete that won the race while we
        // were staging must find us backing off, not recreating.
        let Some(mut guard) = entry.lock_unless_deleted() else {
            let _ = remove_persisted_file(&staged);
            entry.flushing.store(false, Ordering::Relaxed);
            return false;
        };
        let inner = &mut *guard;
        // An eviction cooled us to Cold while we staged. Seeing `flushing`
        // it persisted this entry itself, so our staged snapshot is at
        // best a duplicate of what it wrote and, if a write beat the
        // evict, a step behind — publishing it would regress the image.
        // Drop the staged bytes and leave `dirty` as the evict (and any
        // racing write) left it. A compaction that swapped `slot` for a
        // fresh `Context` while we staged is the same story with the
        // variant unchanged — Hot in, Hot out — so `image_generation` is
        // what catches it: publishing our snapshot now would overwrite
        // the compacted image with the pre-compaction one it just
        // replaced, and stamp the entry's stats back to match. An
        // eviction that reloaded Hot again is the same story once more:
        // it bumps `image_generation` on its way to Cold specifically so
        // this catches that case too, the same as compaction's. Either
        // way, backing off here costs nothing — the next tick flushes
        // the current image instead.
        if !matches!(inner.slot, Slot::Hot(_)) || inner.image_generation != generation {
            let _ = remove_persisted_file(&staged);
            entry.flushing.store(false, Ordering::Relaxed);
            return false;
        }
        // Claim the usage flag before snapshotting: an increment racing
        // this write either lands in the snapshot or re-marks the flag —
        // never both lost. (Advisory counters; a failed write re-marks.)
        entry.usage_dirty.store(false, Ordering::Relaxed);
        // Sidecar before image, same as `save_files`: `scan_data_dir` keys
        // existence on the image, so it must land LAST. A crash between
        // the two leaves the sidecar ahead of the image it describes —
        // Cold reporting briefly over-states this entry rather than
        // under-stating it, which errs toward a maintenance sweep
        // re-checking a context that turns out not to need it yet, never
        // toward skipping one that does. Publishing the image first (the
        // old order) risked exactly that skip, since the sidecar would
        // instead lag the image until the next successful flush.
        let outcome = write_meta(
            &self.0.data_dir,
            &stem,
            &meta,
            &stats,
            &entry.usage.snapshot(),
            // Snapshot NOW, under the publication lock, not at staging:
            // counters that moved while the image staged are exactly the
            // sidecar-ahead-of-image posture the write order already
            // tolerates, and the WAL replay re-derives the graph value
            // on the next load either way.
            entry.revision_snapshot(inner),
        )
        .and_then(|()| commit_staged(&staged, &image));
        let published = match outcome {
            Ok(()) => {
                inner.stats = stats;
                self.0.metrics.record_flush(name, true);
                // `dirty` stays as claimed: clear (nothing raced) means the
                // image is current; a racing write re-set it and will
                // re-flush. Truncation is sound only when the image covers
                // the whole log — a write that landed mid-stage sits past
                // our watermark and its records must survive.
                if inner.wal_seq - 1 == watermark {
                    self.truncate_wal(name, inner);
                }
                true
            }
            Err(error) => {
                tracing::warn!("flush of context '{name}' failed (will retry): {error}");
                let _ = remove_persisted_file(&staged);
                entry.dirty.store(true, Ordering::Relaxed);
                entry.usage_dirty.store(true, Ordering::Relaxed);
                self.0.metrics.record_flush(name, false);
                false
            }
        };
        entry.flushing.store(false, Ordering::Relaxed);
        published
    }

    /// Truncates a context's log once an image covering everything in
    /// it has published. Failure is harmless — the image's watermark
    /// already makes the logged records replay-inert — so it warns and
    /// moves on.
    ///
    /// With replication on, the reset additionally waits (by skipping,
    /// never by blocking) until the shipper has read the records it is
    /// about to discard: the retained tail is replay-inert either way,
    /// so deferring costs nothing but bytes, while resetting under the
    /// shipper's feet forces it through the series-restart path — a
    /// whole-image upload instead of a few log records. Bounded by the
    /// shipper's own deferral budget, so a dead bucket can never walk
    /// this log into its cap.
    fn truncate_wal(&self, name: &str, inner: &mut EntryInner) {
        let path = wal_path(&self.0.data_dir, &file_stem(name));
        if let Some(progress) = &self.0.ship_progress
            && !progress.allows_reset(&path, inner.wal_seq - 1, inner.wal_bytes)
        {
            return;
        }
        match wal::reset(&path) {
            Ok(()) => inner.wal_bytes = 0,
            Err(error) => {
                tracing::warn!("WAL for '{name}' not truncated (harmless): {error}");
            }
        }
    }
}

impl AppState {
    /// The used-vs-ceiling read behind every storage-quota gate: the
    /// context's whole on-disk family, summed the same way the
    /// `taguru_context_disk_bytes` gauges sum it — flush-refreshed
    /// snapshot lanes (image, passages, sidecars) plus the live WAL
    /// lanes — so enforcement and observability can never disagree
    /// about what "used" means. `Some((used, ceiling))` when the
    /// context is AT or over a declared ceiling: at, because a line
    /// the next write would cross is a line that no longer admits
    /// growth (the WAL cap compares the same way). Content growth
    /// lands in the live lanes, so a burst inside one flush interval
    /// cannot outrun this check; the snapshot lanes lag at most one
    /// flush interval and refresh the moment images land.
    pub(super) fn storage_quota_excess(
        &self,
        name: &str,
        inner: &EntryInner,
        entry: &Entry,
    ) -> Option<(u64, u64)> {
        let ceiling = self.0.context_quotas.get(name)?.storage_bytes?;
        let disk = *entry.disk.lock();
        // A resident store knows its pending log; a cold one uses the
        // scan/eviction-seeded field — the same read `gauge_snapshot`
        // does, for the same reason.
        let passages_wal = entry
            .passages
            .lock()
            .as_ref()
            .map(|store| store.pending_log_bytes())
            .unwrap_or(inner.passages_wal_bytes);
        let used = disk.image_bytes
            + disk.passages_bytes
            + disk.sidecar_bytes
            + inner.wal_bytes
            + passages_wal;
        (used >= ceiling).then_some((used, ceiling))
    }

    /// [`Self::storage_quota_excess`] for a caller holding no entry
    /// lock — the import loop's per-batch pre-check. `None` for an
    /// unknown context too: creation is never quota-gated (a declared
    /// name may not exist yet), and the growth gates inside the write
    /// path cover everything a fresh batch then writes.
    pub fn storage_quota_refusal(&self, name: &str) -> Option<(u64, u64)> {
        let entry = self.lookup(name)?;
        let inner = entry.read_unless_deleted()?;
        self.storage_quota_excess(name, &inner, &entry)
    }

    /// The write path of the HTTP mutators: stage the whole batch in
    /// the context's WAL — one fsync, group commit at exactly the
    /// granularity the API already locks at — and only then run
    /// `operate` to apply it. An append that cannot be made durable
    /// refuses the write outright ([`AccessError::Unpersisted`],
    /// nothing applied): the client must never hold an acknowledgment
    /// the disk cannot replay. With the WAL disabled the staging step
    /// is skipped and durability falls back to the flush interval.
    /// `operate` may apply fewer than `ops.len()` — `apply_in_order`
    /// stops at the first rejection, but the WAL above was already
    /// appended in full (durability can't wait on a result it doesn't
    /// have yet). Left alone, the untried tail would sit on disk
    /// looking exactly like an applied record: `ensure_hot`'s replay
    /// (`replay_op`) continues past a rejection where the live path
    /// stopped at the first one, so that tail would be tried
    /// independently — and could succeed — next time this context
    /// goes cold. `applied` reports how many ops actually landed so
    /// the excess can be trimmed back out before this returns.
    pub(crate) fn logged_write<T>(
        &self,
        name: &str,
        ops: &[WalOp],
        operate: impl FnOnce(&mut Context) -> T,
        applied: impl FnOnce(&T) -> usize,
    ) -> Result<T, AccessError> {
        let entry = self.lookup(name).ok_or(AccessError::NotFound)?;
        let mut wal_behind = false;
        let result = {
            // Same tombstone rule as with_hot: a delete that beat us to
            // this lock owns the name — appending here would recreate
            // the WAL file it just removed.
            let mut inner = entry.lock_unless_deleted().ok_or(AccessError::NotFound)?;
            ensure_hot(
                &self.0.data_dir,
                name,
                &mut inner,
                &self.0.metrics,
                self.0.hydrator.as_deref(),
            )
            .map_err(AccessError::Load)?;
            // Count the promotion NOW, before the quota, WAL-cap, and
            // append-failure early returns below. A Cold→Hot load just added
            // this context's footprint to resident memory; a refusal that
            // returns without counting it leaves the resident estimate short,
            // so the budget sweep never reclaims those bytes. recount_entry is
            // absolute, so the post-`operate` recount below just refreshes this.
            self.recount_entry(&mut inner);
            // The policy ceiling, ahead of the failure backstop below and
            // WAL on or off: only batches that GROW the context are gated —
            // a retract/unalias batch always passes, because shrinking is
            // how a tenant gets back under (the passage store's own cap
            // draws the same line). If both this and the WAL cap hold, the
            // refusal names the quota; the failing flush behind the cap is
            // already warned about from the flusher itself every tick.
            if ops.iter().any(WalOp::grows)
                && let Some((used, ceiling)) = self.storage_quota_excess(name, &inner, &entry)
            {
                self.0.metrics.record_storage_quota_refusal();
                return Err(AccessError::QuotaExceeded(storage_quota_message(
                    name, used, ceiling,
                )));
            }
            let first_seq = inner.wal_seq;
            let mut staged = None;
            if self.0.wal_enabled {
                // Backstop against unbounded growth: the log only
                // truncates after a successful image save, so a
                // persistently failing flush would grow it forever.
                // Past the cap, refuse writes — loudly — instead.
                if self.0.wal_max_bytes > 0 && inner.wal_bytes >= self.0.wal_max_bytes as u64 {
                    tracing::warn!(
                        context = %name,
                        wal_bytes = inner.wal_bytes,
                        cap = self.0.wal_max_bytes,
                        "WAL over its cap with the image failing to flush; write refused"
                    );
                    return Err(AccessError::Unpersisted(format!(
                        "the write-ahead log is at {} bytes (cap {}): the image has been \
                         failing to flush — check disk space and the server log",
                        inner.wal_bytes, self.0.wal_max_bytes
                    )));
                }
                let path = wal_path(&self.0.data_dir, &file_stem(name));
                let len_before = inner.wal_bytes;
                match wal::append_batch(&path, inner.wal_seq, ops) {
                    Ok(appended) => {
                        self.0.metrics.record_wal_append(true);
                        inner.wal_bytes += appended;
                        inner.wal_seq += ops.len() as u64;
                    }
                    Err(error) => {
                        // The client sees the refusal; the operator
                        // must too — the core durability promise just
                        // failed to engage.
                        self.0.metrics.record_wal_append(false);
                        tracing::warn!(context = %name, %error, "WAL append failed; write refused");
                        // A failed append may still have leaked complete
                        // bytes (write landed, sync then failed, rollback
                        // failed too). Memory is untouched — `operate` never
                        // ran — so mark the entry dirty: the next flush
                        // stages this pre-write image at watermark
                        // `wal_seq - 1` and, since `wal_seq` did not move,
                        // truncates the log, carrying off the leaked tail
                        // before a replay can apply it. (`replay` de-dupes
                        // by seq as the second line of defense.)
                        entry.dirty.store(true, Ordering::Relaxed);
                        return Err(AccessError::Unpersisted(error.to_string()));
                    }
                }
                staged = Some((path, len_before));
            }
            let Slot::Hot(context) = &mut inner.slot else {
                unreachable!("ensure_hot leaves the slot hot");
            };
            let result =
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operate(context))) {
                    Ok(result) => result,
                    Err(payload) => {
                        // The WAL append above already landed, but `operate`
                        // panicked before the in-memory apply it durably
                        // promises could finish, and before `dirty` below
                        // could cover it. Left Hot, this half-mutated
                        // Context would keep serving reads and accepting
                        // further writes forever — parking_lot doesn't
                        // poison. Forcing the slot back to Cold makes the
                        // next access rebuild Hot from the image plus a
                        // full WAL replay instead, which reapplies the very
                        // op that just panicked through the same validated
                        // path replay always uses. recount_entry reflects
                        // the entry's now-zero resident footprint right
                        // away, matching the promotion it counted above.
                        inner.slot = Slot::Cold;
                        self.recount_entry(&mut inner);
                        std::panic::resume_unwind(payload);
                    }
                };
            entry.dirty.store(true, Ordering::Relaxed);
            self.recount_entry(&mut inner);
            // The graph revision counts what APPLIED, in both WAL
            // modes — `wal_seq` cannot carry this: it does not move at
            // all with the WAL disabled, and the partial-apply
            // bookkeeping below rolls it back in states where the
            // applied prefix stands in memory regardless.
            let landed = applied(&result);
            inner.graph_revision += landed as u64;

            if let Some((path, len_before)) = staged
                && landed < ops.len()
            {
                match wal::truncate_to(&path, len_before) {
                    Ok(()) if landed > 0 => {
                        match wal::append_batch(&path, first_seq, &ops[..landed]) {
                            Ok(appended) => {
                                inner.wal_bytes = len_before + appended;
                                inner.wal_seq = first_seq + landed as u64;
                            }
                            Err(error) => {
                                // The applied prefix already happened
                                // in memory and will not be undone —
                                // and the truncate above already threw
                                // its records off the disk, so the
                                // caller now holds an acknowledgment
                                // the log cannot replay. Flush the
                                // image below (out of this lock) to
                                // close that crash window now rather
                                // than waiting out the flush interval.
                                tracing::warn!(
                                    context = %name, %error,
                                    "WAL re-append after a partial apply failed; \
                                     flushing the image now to re-cover memory"
                                );
                                inner.wal_bytes = len_before;
                                inner.wal_seq = first_seq;
                                wal_behind = true;
                            }
                        }
                    }
                    Ok(()) => {
                        inner.wal_bytes = len_before;
                        inner.wal_seq = first_seq;
                    }
                    Err(error) => {
                        // The untried tail is still on disk looking
                        // exactly like applied records, and replay
                        // does not stop where the live apply did —
                        // left until the next flush tick, a crash
                        // would apply ops the caller was just told
                        // failed. Same medicine as the re-append
                        // failure above: flush the image now (out
                        // of this lock). Its watermark covers the
                        // whole batch's seqs, so the tail becomes
                        // replay-inert even if the log keeps
                        // refusing to shrink. Bookkeeping stays at
                        // the full-batch values: that is what the
                        // file still holds if the truncate never
                        // landed, and a successful flush resets
                        // both anyway.
                        tracing::warn!(
                            context = %name, %error,
                            "WAL truncate after a partial apply failed; \
                             flushing the image now to retire the untried tail"
                        );
                        wal_behind = true;
                    }
                }
            }

            result
        };
        // The one state `logged_write` must never return in: ops the
        // caller is being told succeeded, present in memory only — the
        // trimmed log no longer holds them and a crash before the next
        // flush would silently lose them. An immediate image flush
        // restores the "acknowledged means replayable" contract. It can
        // come back false two ways: a real I/O failure (already logged
        // and counted by `record_flush`, which is what /health reads) or
        // a silent no-op racing a flush already in flight (claimed
        // `flushing` first) — that path never touches `record_flush`, so
        // /health stays green even though this write is, for the moment,
        // relying on the periodic flusher rather than the WAL to survive
        // a crash. `dirty` is already set (unconditionally, above)
        // either way, so the next tick still retries it — but a crash in
        // that window would replay a WAL that no longer matches memory,
        // so the gap is worth its own loud signal rather than blending
        // into an ordinary retry.
        if wal_behind && !self.flush_entry(name, &entry) {
            tracing::warn!(
                context = %name,
                "post-write recovery flush did not land immediately; the WAL for this \
                 write is no longer trustworthy and durability now depends on the next \
                 periodic flush landing before a crash"
            );
        }
        self.touch(&entry);
        self.enforce_budget(name);
        Ok(result)
    }

    /// Evicts least-recently-used, unpinned, hot contexts until their
    /// resident estimate fits the budget. `except` (the context just
    /// used) is never evicted, so a single oversized context cannot
    /// thrash. Dirty contexts are persisted before eviction; if that
    /// save fails they stay resident rather than losing writes.
    pub(crate) fn enforce_budget(&self, except: &str) {
        // Cheap gate in front of the O(contexts) sweep below: one
        // atomic load of the graph estimate, which per-entry recounts
        // keep exact. Vector-store residency is NOT tracked between
        // sweeps, so every 64th operation forces a real sweep anyway —
        // its reconciling store below bounds any staleness or drift at
        // 64 operations. Before this gate the full sweep (snapshot,
        // two lock acquisitions per context) ran on every request.
        let ops = self.0.budget_ops.fetch_add(1, Ordering::Relaxed);
        let budget = i64::try_from(self.0.cache_bytes).unwrap_or(i64::MAX);
        if !ops.is_multiple_of(64) && self.0.resident_estimate.load(Ordering::Relaxed) <= budget {
            return;
        }

        let mut candidates: Vec<(bool, u64, usize, String, Arc<Entry>)> = Vec::new();
        let mut total = 0usize;
        for (name, entry) in self.snapshot() {
            let inner = entry.inner.read();
            if inner.meta.pinned {
                continue;
            }
            let resident = match &inner.slot {
                Slot::Hot(context) => context.footprint(),
                // A deleted entry holds nothing: the tombstone dropped
                // the graph and the delete cleared the vectors.
                Slot::Cold | Slot::Deleted => 0,
            };
            drop(inner);
            // Cached vector stores, resident passages, the BM25 index,
            // and paragraph vectors count too — a cold entry can hold
            // plenty of each.
            let bytes = resident
                + entry.vectors_footprint()
                + entry.passages_footprint()
                + entry.bm25_footprint()
                + entry.passage_vectors_footprint();
            if bytes == 0 {
                continue;
            }
            // Strictly over its declared cache ceiling (issue #136):
            // evicted before any compliant candidate below, however
            // recently touched. The ceiling reorders who goes first
            // under pressure; it reserves nothing while there is slack.
            let over_ceiling = self
                .0
                .context_quotas
                .get(&name)
                .and_then(|quota| quota.cache_bytes)
                .is_some_and(|ceiling| bytes as u64 > ceiling);
            total += bytes;
            candidates.push((
                over_ceiling,
                entry.last_touch.load(Ordering::Relaxed),
                bytes,
                name,
                entry,
            ));
        }
        // Reconcile the gate with measured reality — vectors included,
        // and any drift folded away.
        self.0
            .resident_estimate
            .store(total as i64, Ordering::Relaxed);
        if total <= self.0.cache_bytes {
            return;
        }

        // Over-ceiling first (their damage to the rest is what the
        // ceiling bounds), least recently used within each half.
        candidates.sort_unstable_by_key(|&(over, touch, ..)| (std::cmp::Reverse(over), touch));
        for (_, _, bytes, name, entry) in candidates {
            if total <= self.0.cache_bytes {
                break;
            }
            if name == except {
                continue;
            }
            if self.evict_entry(&name, &entry) {
                total = total.saturating_sub(bytes);
            }
        }
        self.0
            .resident_estimate
            .store(total as i64, Ordering::Relaxed);
    }

    /// One entry's eviction: persist if dirty, drop the graph, clear
    /// the cached vectors. `false` means nothing was freed — the entry
    /// got pinned since the caller's sweep, its save failed (it stays
    /// resident rather than losing writes), or a concurrent eviction
    /// already cleared it. That last case matters: two budget sweeps
    /// snapshot the directory under a shared lock and can carry the
    /// same candidate, so the loser must report `false` or the caller
    /// subtracts its bytes from the residency estimate a second time.
    ///
    /// The common case persists BEFORE the entry's write lock below is
    /// taken, via [`Self::flush_bm25`] and [`Self::flush_entry`] —
    /// both already stage their disk work (serialize + fsync) with
    /// the lock released (see `flush_entry`'s doc comment) and re-take
    /// it only to publish. Calling them here, instead of this function
    /// doing its own lock-held save the way it once did unconditionally,
    /// means an eviction no longer stalls every reader and writer of
    /// the context for as long as the image takes to land — the same
    /// stall `flush_entry` was written to avoid in the first place.
    /// The lock-held save below still exists as the fallback for the
    /// rare case a rival flush is already mid-flight when this call
    /// starts: `flush_entry`'s own claim would just lose that race and
    /// no-op, and skipping the drop-to-Cold below in that case would
    /// mean the caller's eviction sweep might never make progress on a
    /// context under sustained write pressure.
    pub(crate) fn evict_entry(&self, name: &str, entry: &Entry) -> bool {
        // On a replica nothing is ever dirty (writes are refused), so
        // the flush attempts are structurally no-ops — skipped for
        // clarity, alongside the one genuine writer below (the passage
        // store's best-effort compaction), which must not touch a
        // cache the tailer owns.
        if !self.is_replica() {
            self.flush_bm25(name, entry);
            self.flush_entry(name, entry);
        }

        let mut guard = entry.inner.write();
        let inner = &mut *guard;
        // Re-check under the write lock; the entry may have changed
        // between the snapshot and now.
        if inner.meta.pinned {
            return false;
        }
        let mut freed = false;
        let watermark = inner.wal_seq - 1;
        if let Slot::Hot(context) = &mut inner.slot {
            // Still dirty/flushing after the attempt above: either a
            // rival flush was already mid-flight (its own claim swap
            // made `flush_entry` above a no-op) or `flush_entry` itself
            // lost a race and backed off. Either way the durable image
            // is stale or absent, so fall back to saving it here, under
            // the lock, same as this function always did — the rare
            // cost of a lock-held serialize+fsync only on the rare
            // path where flush and eviction land on the same entry at
            // the same instant.
            if entry.dirty.load(Ordering::Relaxed) || entry.flushing.load(Ordering::Relaxed) {
                context.set_applied_seq(watermark);
                let stats = ContextStats::of(context);
                if let Err(error) = save_files(
                    &self.0.data_dir,
                    name,
                    &inner.meta,
                    &stats,
                    &entry.usage.snapshot(),
                    ContextRevision {
                        graph: inner.graph_revision,
                        passages: entry.passage_revision.load(Ordering::Relaxed),
                        config: inner.config_revision,
                    },
                    context,
                ) {
                    tracing::warn!(
                        "context '{name}' stays resident, eviction save failed: {error}"
                    );
                    self.0.metrics.record_eviction(false);
                    return false;
                }
                inner.stats = stats;
                entry.dirty.store(false, Ordering::Relaxed);
                self.truncate_wal(name, inner);
            } else {
                inner.stats = ContextStats::of(context);
            }
            inner.slot = Slot::Cold;
            // Bump so a flush that staged its image before this
            // eviction can tell, once it re-locks, that the slot it
            // captured is gone — a later reload plus that flush's
            // re-publish would otherwise resurrect a stale image over
            // whatever was written after the reload.
            inner.image_generation += 1;
            // Local zero only: the caller's absolute store settles
            // the global, so a recount's delta would double-count.
            inner.counted_bytes = 0;
            self.0.metrics.record_eviction(true);
            freed = true;
        }
        drop(guard);

        // Dropping the passage store loses nothing (its log is fsynced
        // per batch); a best-effort compaction first just spares the
        // next load a replay. Failure changes neither. Not on a
        // replica: a compaction rewrites the snapshot and truncates
        // the log — files the manifest owns — so there the store is
        // dropped as-is (the log replays on the next load, as it
        // always may).
        let compacted_wal_bytes = {
            let mut passages = entry.passages.lock();
            match passages.take() {
                Some(store) => {
                    if !self.is_replica()
                        && store.pending_log_bytes() > 0
                        && let Err(error) = store.compact()
                    {
                        tracing::warn!("passages for '{name}' evicted uncompacted: {error}");
                    }
                    Some(store.pending_log_bytes())
                }
                None => None,
            }
        };
        if let Some(bytes) = compacted_wal_bytes {
            freed = true;
            // Cold from here on: `gauge_snapshot` reads this cached
            // value instead of re-`stat`ing the log on every scrape.
            entry.inner.write().passages_wal_bytes = bytes;
        }
        // Same best-effort posture for a dirty index: saving it spares
        // the next residency a re-tokenization. `flush_bm25` above
        // already persisted it if it was dirty, so `bm25_dirty` is
        // normally already clear here and this is just a `take()`.
        {
            let mut bm25 = entry.bm25.write();
            if let Some(index) = bm25.take() {
                freed = true;
                if !self.is_replica()
                    && entry.bm25_dirty.swap(false, Ordering::Relaxed)
                    && let Err(error) = write_atomic(
                        &bm25_path(&self.0.data_dir, &file_stem(name)),
                        &index.to_bytes(),
                    )
                {
                    tracing::warn!("BM25 index for '{name}' evicted unpersisted: {error}");
                }
            }
        }
        if entry.passage_vectors.lock().take().is_some() {
            freed = true;
        }
        if entry.vectors.lock().take().is_some() {
            freed = true;
        }
        freed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::test_support::{assoc_op, loaded_map, plain, rendered, scratch_dir};

    #[test]
    fn ensure_hot_records_hits_and_loads() {
        let dir = scratch_dir("m-cache");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state.flush_dirty();
        }

        // A fresh boot leaves the context cold: the first read loads
        // from disk, the second is a pure hit.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert!(rendered(&state).contains("taguru_cache_loads_total{outcome=\"ok\"} 1"));
        state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert!(rendered(&state).contains("taguru_cache_hits_total 1"));

        // Gauges come from the registry itself.
        assert!(rendered(&state).contains("taguru_contexts_registered 1"));
        assert!(rendered(&state).contains("taguru_contexts_resident 1"));

        let _ = fs::remove_dir_all(dir);
    }

    /// The `TAGURU_CONTEXT_QUOTAS` parser (issue #136) accepts the
    /// documented shape and refuses every trap — a declaration that
    /// silently failed to arm is an unbounded tenant, so anything
    /// short of exactly-as-documented must stop the boot.
    #[test]
    fn context_quota_parsing_accepts_the_shape_and_refuses_the_traps() {
        assert!(parse_context_quotas(None).unwrap().is_empty());

        let quotas = parse_context_quotas(Some(
            r#"{"tenant-a": {"storage_bytes": 1024, "cache_bytes": 2048},
                "tenant-b": {"storage_bytes": 512}}"#,
        ))
        .unwrap();
        assert_eq!(quotas["tenant-a"].storage_bytes, Some(1024));
        assert_eq!(quotas["tenant-a"].cache_bytes, Some(2048));
        assert_eq!(quotas["tenant-b"].storage_bytes, Some(512));
        assert_eq!(quotas["tenant-b"].cache_bytes, None);

        // Set-but-empty, broken JSON, a ceiling-less quota, a typo'd
        // field name, and a zero ceiling: each refuses loudly.
        for broken in [
            "",
            "   ",
            "{not json}",
            r#"{"a": {}}"#,
            r#"{"a": {"storage_byte": 1}}"#,
            r#"{"a": {"storage_bytes": 0}}"#,
            r#"{"a": {"cache_bytes": 0, "storage_bytes": 1}}"#,
        ] {
            assert!(parse_context_quotas(Some(broken)).is_err(), "{broken:?}");
        }
    }

    /// The storage ceiling refuses growth writes once the file family
    /// reaches it — through the live WAL lane before any flush, and
    /// through the flush-stamped snapshot lanes after one (gauges off,
    /// proving the quota alone keeps the disk sweep alive) — while the
    /// shrink paths stay open and an uncapped sibling never notices.
    #[test]
    fn storage_quota_refuses_growth_and_keeps_the_shrink_paths_open() {
        let dir = scratch_dir("storage-quota");
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            None,
            BootOptions {
                context_quotas: HashMap::from([(
                    "capped".to_string(),
                    ContextQuota {
                        storage_bytes: Some(1),
                        cache_bytes: None,
                    },
                )]),
                ..BootOptions::default()
            },
        )
        .unwrap();

        // Creation is never gated: a declared name may not exist yet.
        state
            .create("capped", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // The first write lands — nothing is on disk yet — and its WAL
        // append alone carries the family past the one-byte ceiling.
        state
            .add_associations(
                "capped",
                vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md"))],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        let refused = state.add_associations(
            "capped",
            vec![assoc_op("蔵", "銘柄", "青嶺", 1.0, Some("keep.md"))],
            Deadline::unbounded(),
        );
        let Err(AccessError::QuotaExceeded(message)) = refused else {
            panic!("growth past the ceiling must refuse, got {refused:?}");
        };
        assert!(message.contains("storage quota"), "{message}");
        assert!(message.contains("retract or compact"), "{message}");
        let aliases = state.add_aliases(
            "capped",
            &BTreeMap::from([("酒蔵".to_string(), "蔵".to_string())]),
            &BTreeMap::new(),
        );
        assert!(
            matches!(aliases, Err(AccessError::QuotaExceeded(_))),
            "aliases are the other growth family: {aliases:?}"
        );

        // The shrink paths stay open at the ceiling…
        state.retract_source("capped", "keep.md").unwrap();
        state
            .compact_context("capped", Deadline::unbounded())
            .unwrap();
        // …and after a flush moves the bytes from the live lane into
        // the snapshot lanes (the WAL truncates; the image and meta
        // keep the family over this ceiling), growth still refuses —
        // the flush-time bookkeeping is what the gate reads.
        state.flush_dirty();
        assert!(matches!(
            state.add_associations(
                "capped",
                vec![assoc_op("蔵", "産地", "灘", 1.0, Some("keep.md"))],
                Deadline::unbounded(),
            ),
            Err(AccessError::QuotaExceeded(_))
        ));
        // The lock-free read the import pre-check uses agrees.
        assert!(state.storage_quota_refusal("capped").is_some());

        // An uncapped sibling never notices any of this.
        state
            .create("free", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "free",
                vec![assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md"))],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(state.storage_quota_refusal("free"), None);

        let _ = fs::remove_dir_all(dir);
    }

    /// The cache ceiling's provability claim (issue #136): under
    /// pressure, a context past its declared ceiling is evicted before
    /// the LRU victim — so the eviction damage a saturating context
    /// can inflict on compliant residents is bounded by its ceiling,
    /// while recency still orders everything else.
    #[test]
    fn cache_ceiling_evicts_the_over_quota_context_before_the_lru_victim() {
        let dir = scratch_dir("cache-ceiling");
        // First boot: build three same-shaped contexts and measure one
        // footprint, so the second boot's budget can be sized to hold
        // exactly two of them.
        let footprint = {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            for name in ["old", "hog", "fresh"] {
                state
                    .create(name, ContextMeta::default())
                    .map_err(|_| "create")
                    .unwrap();
                state
                    .add_associations(
                        name,
                        vec![
                            assoc_op("蔵", "杜氏", "高瀬", 1.0, Some("keep.md")),
                            assoc_op("蔵", "銘柄", "青嶺", 1.0, Some("keep.md")),
                        ],
                        Deadline::unbounded(),
                    )
                    .unwrap()
                    .unwrap();
            }
            state.flush_dirty();
            state
                .read_context("old", |context| context.footprint())
                .map_err(|_| "read")
                .unwrap()
        };

        let state = AppState::boot_with(
            dir.clone(),
            footprint * 5 / 2,
            None,
            BootOptions {
                context_quotas: HashMap::from([(
                    "hog".to_string(),
                    ContextQuota {
                        storage_bytes: None,
                        cache_bytes: Some(1),
                    },
                )]),
                ..BootOptions::default()
            },
        )
        .unwrap();
        // old first (the would-be LRU victim), hog second (over its
        // ceiling, more recently used than old), fresh last (its load
        // pushes the total past the budget and triggers the sweep).
        for name in ["old", "hog", "fresh"] {
            state
                .read_context(name, |context| context.association_count())
                .map_err(|_| "read")
                .unwrap();
        }
        let loaded = loaded_map(&state);
        assert!(
            !loaded["hog"],
            "the over-ceiling context must go first: {loaded:?}"
        );
        assert!(
            loaded["old"],
            "plain LRU would have evicted old — the ceiling reordered it: {loaded:?}"
        );
        assert!(loaded["fresh"], "{loaded:?}");

        let _ = fs::remove_dir_all(dir);
    }

    /// A context whose load failed answers the remembered refusal
    /// without touching the disk until the retry window elapses — a
    /// permanently corrupt context must not cost a read + full parse
    /// per request under client retries — and heals by itself on the
    /// first retry after the files are restored.
    #[test]
    fn a_failed_load_is_quarantined_until_the_retry_window_elapses() {
        let dir = scratch_dir("quarantine");
        {
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
        }
        let image = dir.join("sake.ctx");
        let healthy = fs::read(&image).unwrap();
        let mut corrupt = healthy.clone();
        corrupt[8] = 0xFF; // the version field — refused by from_bytes
        fs::write(&image, &corrupt).unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let AccessError::Load(first) = state.read_context("sake", |_| ()).unwrap_err() else {
            panic!("a corrupt image must refuse the read");
        };
        assert!(first.contains("corrupt"), "{first}");

        // Repair the file. Within the window the CACHED refusal still
        // answers — proof the disk is not consulted per request.
        fs::write(&image, &healthy).unwrap();
        let AccessError::Load(second) = state.read_context("sake", |_| ()).unwrap_err() else {
            panic!("the quarantine must still refuse");
        };
        assert!(second.contains("quarantined"), "{second}");
        assert!(
            rendered(&state).contains("taguru_cache_loads_total{outcome=\"failed\"} 1"),
            "a quarantined refusal is not a second load attempt"
        );

        // Past the window, the retry sees the repaired image and heals.
        state.age_load_failures("sake", LOAD_FAILURE_RETRY);
        let count = state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(count, 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn flush_and_eviction_record_their_outcomes() {
        let dir = scratch_dir("m-flush");
        let state = AppState::boot(dir.clone(), 1, None).unwrap();
        state
            .create("a", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .create("b", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .write_context("a", |context| {
                context.associate("私", "好き", "りんご", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();
        // Touching b evicts a under the one-byte budget.
        state
            .read_context("b", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        let body = rendered(&state);
        assert!(
            body.contains("taguru_cache_evictions_total{outcome=\"ok\"} 1"),
            "{body}"
        );

        state
            .write_context("b", |context| {
                context.associate("用語", "意味", "定義", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();
        state.flush_dirty();
        // 2, not 1: evicting "a" above already flushed it (evict_entry
        // persists via flush_entry so its own write lock is never held
        // across the serialize+fsync — see evict_entry's doc comment),
        // and this flush_dirty tick adds "b"'s.
        assert!(
            rendered(&state).contains("taguru_flush_total{outcome=\"ok\"} 2"),
            "a flushed dirty context must count"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn unflushed_writes_survive_a_process_restart_via_the_wal() {
        let dir = scratch_dir("wal-restart");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op(
                        "青嶺酒造",
                        "創業年",
                        "1907年",
                        1.0,
                        Some("第1段落"),
                    )],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            // NO flush_dirty: dropping the state here is the crash.
            // The 5-second window would have eaten this write; the WAL
            // must not.
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let recalled = state
            .read_context("sake", |context| context.recall("青嶺酒造"))
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(recalled.len(), 1, "the acknowledged write must survive");
        assert_eq!(recalled[0].object, "1907年");
        assert_eq!(
            recalled[0].attributions.len(),
            1,
            "attributions ride the WAL too"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_failed_reappend_after_a_partial_apply_still_persists_acknowledged_ops() {
        // A partial apply trims the WAL back and re-appends just the
        // applied prefix. The trim durably removed the whole batch, so
        // if that re-append then fails, the ops the caller is being
        // told succeeded exist in memory only — logged_write must close
        // that crash window itself (an immediate image flush), not
        // leave it open until the next flush tick.
        let dir = scratch_dir("wal-reappend-fault");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            // Two concepts for the aliases to target, and an alias the
            // batch's second op will re-point — the conflict that makes
            // the apply partial.
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("青嶺酒造", "所在地", "京都酒造", 1.0, None)],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            state
                .add_aliases(
                    "sake",
                    &BTreeMap::from([("kyo".to_string(), "京都酒造".to_string())]),
                    &BTreeMap::new(),
                )
                .unwrap()
                .unwrap();

            // BTreeMap order: "aomine" applies first, "kyo" then
            // conflicts (re-pointing an existing alias). The batch
            // append itself succeeds; the fault fires on the re-append
            // of the applied prefix, right after the trim.
            wal::fail_appends_after(1);
            let partial = state
                .add_aliases(
                    "sake",
                    &BTreeMap::from([
                        ("aomine".to_string(), "青嶺酒造".to_string()),
                        ("kyo".to_string(), "青嶺酒造".to_string()),
                    ]),
                    &BTreeMap::new(),
                )
                .unwrap()
                .unwrap_err();
            assert_eq!(
                partial.applied, 1,
                "the first alias landed before the conflict"
            );
            // NO flush_dirty: dropping the state here is the crash.
        }

        let reborn = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let aliases = reborn
            .read_context("sake", |context| {
                context
                    .concept_aliases()
                    .into_iter()
                    .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
                    .collect::<Vec<_>>()
            })
            .map_err(|_| "read")
            .unwrap();
        assert!(
            aliases
                .iter()
                .any(|(alias, canonical)| alias == "aomine" && canonical == "青嶺酒造"),
            "the acknowledged alias must survive the crash: {aliases:?}"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_failed_wal_trim_after_a_partial_apply_keeps_the_untried_tail_from_replaying() {
        // A partial apply trims the untried tail back out of the WAL.
        // If that trim itself fails, the tail sits on disk looking
        // exactly like applied records — and replay does not stop at
        // the rejection the live apply stopped at, so a crash would
        // apply ops the caller was just told failed. logged_write must
        // close that window itself: an immediate image flush whose
        // watermark retires the whole batch's seqs.
        let dir = scratch_dir("wal-trim-fault");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("青嶺酒造", "所在地", "京都酒造", 1.0, None)],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            state
                .add_aliases(
                    "sake",
                    &BTreeMap::from([("kyo".to_string(), "京都酒造".to_string())]),
                    &BTreeMap::new(),
                )
                .unwrap()
                .unwrap();

            // BTreeMap order: "aomine" applies, "kyo" conflicts
            // (re-pointing an existing alias) and stops the batch,
            // "mine" is never tried. The batch append succeeds; the
            // fault fires on the trim that should carry "kyo" and
            // "mine" back off the disk.
            wal::fail_truncates_after(0);
            let partial = state
                .add_aliases(
                    "sake",
                    &BTreeMap::from([
                        ("aomine".to_string(), "青嶺酒造".to_string()),
                        ("kyo".to_string(), "青嶺酒造".to_string()),
                        ("mine".to_string(), "青嶺酒造".to_string()),
                    ]),
                    &BTreeMap::new(),
                )
                .unwrap()
                .unwrap_err();
            assert_eq!(
                partial.applied, 1,
                "only the first alias landed before the conflict"
            );
            // NO flush_dirty: dropping the state here is the crash.
        }

        let reborn = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let aliases = reborn
            .read_context("sake", |context| {
                context
                    .concept_aliases()
                    .into_iter()
                    .map(|(alias, canonical)| (alias.to_string(), canonical.to_string()))
                    .collect::<Vec<_>>()
            })
            .map_err(|_| "read")
            .unwrap();
        assert!(
            aliases
                .iter()
                .any(|(alias, canonical)| alias == "aomine" && canonical == "青嶺酒造"),
            "the acknowledged alias must survive the crash: {aliases:?}"
        );
        assert!(
            aliases
                .iter()
                .any(|(alias, canonical)| alias == "kyo" && canonical == "京都酒造"),
            "the conflicting alias must keep its original target: {aliases:?}"
        );
        assert!(
            !aliases.iter().any(|(alias, _)| alias == "mine"),
            "an op the caller was told failed must not replay into existence: {aliases:?}"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_flush_whose_snapshot_predates_a_delete_does_not_resurrect_the_files() {
        let dir = scratch_dir("delete-vs-flush");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("victim", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "victim",
                vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();

        // The flusher's world an instant before the delete: entry Arcs
        // cloned out of the registry, the victim still dirty.
        let stale = state.snapshot();
        state.delete("victim").unwrap().unwrap();

        // Even if a stale handle re-marks the entry dirty (delete does
        // clear the flag, but that is an optimization), the tombstone
        // is what must hold.
        for (_, entry) in &stale {
            entry.dirty.store(true, Ordering::Relaxed);
        }
        // The flusher arrives late and works through its stale snapshot.
        for (name, entry) in &stale {
            assert!(
                !state.flush_entry(name, entry),
                "a deleted context must not flush"
            );
        }

        let stem = file_stem("victim");
        for suffix in [
            "ctx",
            "meta.json",
            "sources.json",
            "vectors.bin",
            "wal.jsonl",
        ] {
            let path = dir.join(format!("{stem}.{suffix}"));
            assert!(
                !path.exists(),
                "{} came back after the delete",
                path.display()
            );
        }
        // The resurrection a user would see: a reboot re-registering
        // it. A real reboot means the old process is gone — and the
        // directory lock with it.
        drop(state);
        let reborn = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert_eq!(
            reborn.context_count(),
            0,
            "the deleted context re-registered on boot"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn wal_growth_is_visible_and_the_cap_refuses_further_writes() {
        let dir = scratch_dir("wal-gauge");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            assert_eq!(state.gauge_snapshot().wal_bytes, 0);
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            let grown = state.gauge_snapshot().wal_bytes;
            assert!(grown > 0, "an append must show in the gauge");
            assert!(rendered(&state).contains("taguru_wal_appends_total{outcome=\"ok\"} 1"));
            assert!(rendered(&state).contains(&format!("taguru_wal_bytes {grown}")));
            state.flush_dirty();
            assert_eq!(
                state.gauge_snapshot().wal_bytes,
                0,
                "truncation must zero the gauge"
            );

            // Leave an unflushed write behind for the reboot check.
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("青嶺酒造", "杜氏", "高瀬", 1.0, None)],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
        }

        // Registration alone — no touch — must already see the
        // leftover log, or the first scrapes after a reboot lie.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(
            state.gauge_snapshot().wal_bytes > 0,
            "boot must stat leftover logs"
        );

        // A tiny cap: the first write passes (the log is empty), the
        // second is refused — the backstop for a flush that never
        // succeeds again.
        let capped_dir = scratch_dir("wal-capped");
        let state = AppState::boot_with(
            capped_dir.clone(),
            usize::MAX,
            None,
            BootOptions {
                wal_max_bytes: 1,
                ..BootOptions::default()
            },
        )
        .unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "sake",
                vec![assoc_op("a", "l", "b", 1.0, None)],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        let refused = state.add_associations(
            "sake",
            vec![assoc_op("c", "l", "d", 1.0, None)],
            Deadline::unbounded(),
        );
        assert!(
            matches!(refused, Err(AccessError::Unpersisted(_))),
            "over the cap the write must be refused: {refused:?}"
        );

        let _ = fs::remove_dir_all(dir);
        let _ = fs::remove_dir_all(capped_dir);
    }

    #[test]
    #[cfg(unix)]
    fn health_follows_the_flusher_down_and_back_up() {
        use std::os::unix::fs::PermissionsExt;

        let dir = scratch_dir("health");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "sake",
                vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state.flush_dirty();
        assert!(state.metrics().flush_is_healthy());
        let stamped = state.metrics().last_flush_success_epoch();
        assert!(stamped > 0, "a successful flush must stamp the gauge");
        assert!(rendered(&state).contains(&format!(
            "taguru_last_flush_success_timestamp_seconds {stamped}"
        )));

        // The disk goes bad: the next flush fails, health turns with it.
        state
            .add_associations(
                "sake",
                vec![assoc_op("青嶺酒造", "杜氏", "高瀬", 1.0, None)],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
        assert!(state.flush_dirty().is_empty());
        assert!(!state.metrics().flush_is_healthy());

        // The disk recovers: the next tick heals the signal.
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(state.flush_dirty(), vec!["sake".to_string()]);
        assert!(state.metrics().flush_is_healthy());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn writes_racing_the_flusher_survive_a_restart() {
        use std::thread;

        let dir = scratch_dir("flush-race");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            let writer = {
                let state = state.clone();
                thread::spawn(move || {
                    for index in 0..100 {
                        state
                            .add_associations(
                                "sake",
                                vec![assoc_op(&format!("s{index}"), "l", "o", 1.0, None)],
                                Deadline::unbounded(),
                            )
                            .unwrap()
                            .unwrap();
                    }
                })
            };
            // Flush continuously while the writes land, so staging and
            // publication interleave with appends every which way. The
            // dangerous outcome is a truncation eating a record the
            // published image does not contain.
            while !writer.is_finished() {
                state.flush_dirty();
            }
            writer.join().unwrap();
            // No final flush: the drop is the crash, and some tail of
            // writes lives only in the WAL.
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let count = state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(count, 100, "every acknowledged write must survive");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn the_budget_gate_tracks_loads_writes_pins_and_deletes() {
        let dir = scratch_dir("estimate");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .add_associations(
                "sake",
                vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        let after_write = state.0.resident_estimate.load(Ordering::Relaxed);
        assert!(after_write > 0, "a resident written context must count");

        // Pinning moves it out of the budget's world…
        state
            .update_meta("sake", None, Some(true), None, None)
            .unwrap()
            .unwrap();
        assert_eq!(state.0.resident_estimate.load(Ordering::Relaxed), 0);
        // …and unpinning brings it back.
        state
            .update_meta("sake", None, Some(false), None, None)
            .unwrap()
            .unwrap();
        assert!(state.0.resident_estimate.load(Ordering::Relaxed) >= after_write);

        state.delete("sake").unwrap().unwrap();
        assert_eq!(state.0.resident_estimate.load(Ordering::Relaxed), 0);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_flusher_stalled_on_another_context_cannot_resurrect_a_delete() {
        use std::thread;
        use std::time::Duration;

        let dir = scratch_dir("delete-mid-flush");
        for round in 0..12 {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            for name in ["decoy", "victim"] {
                state
                    .create(name, ContextMeta::default())
                    .map_err(|_| "create")
                    .unwrap();
                state
                    .add_associations(
                        name,
                        vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
                        Deadline::unbounded(),
                    )
                    .unwrap()
                    .unwrap();
            }

            // A periodic flusher mid-run: it snapshots BOTH contexts,
            // then stalls on the decoy's lock — in the rounds where the
            // decoy comes first in its iteration order — while the
            // delete below runs to completion. The stalled half then
            // reaches the victim through its stale handle. Whatever the
            // iteration order, the end state must be identical: the
            // victim stays deleted.
            let decoy = state.lookup("decoy").unwrap();
            let hold = decoy.inner.write();
            let flusher = {
                let state = state.clone();
                thread::spawn(move || {
                    state.flush_dirty();
                })
            };
            thread::sleep(Duration::from_millis(20)); // flusher snapshots, then parks on the decoy
            state.delete("victim").unwrap().unwrap();
            drop(hold);
            flusher.join().unwrap();

            let stem = file_stem("victim");
            for suffix in [
                "ctx",
                "meta.json",
                "sources.json",
                "vectors.bin",
                "wal.jsonl",
            ] {
                let path = dir.join(format!("{stem}.{suffix}"));
                assert!(
                    !path.exists(),
                    "round {round}: {} survived the delete",
                    path.display()
                );
            }
            // A reboot's view — with the old generation (and its
            // directory lock) gone first, as in any real reboot.
            drop(state);
            let reborn = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            assert_eq!(
                reborn.context_count(),
                1,
                "round {round}: only the decoy may remain"
            );
            let _ = fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn alias_removal_survives_a_restart_via_the_wal() {
        let dir = scratch_dir("unalias-wal");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("青嶺酒造", "創業年", "1907年", 1.0, None)],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            state
                .add_aliases(
                    "sake",
                    &BTreeMap::from([("Aomine".to_string(), "青嶺酒造".to_string())]),
                    &BTreeMap::new(),
                )
                .unwrap()
                .unwrap();
            // Bake the alias into the image; the removal that follows
            // lives only in the WAL when the process dies.
            state.flush_dirty();
            assert_eq!(
                state
                    .remove_aliases("sake", &["Aomine".to_string()], &[])
                    .unwrap()
                    .unwrap(),
                1
            );
        }
        let reborn = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let aliases = reborn
            .read_context("sake", |context| context.concept_aliases().len())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(aliases, 0, "the un-flushed removal must replay");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn replay_does_not_double_apply_records_already_baked_into_the_image() {
        let dir = scratch_dir("wal-noreplay");
        let wal_file = wal_path(&dir, &file_stem("sake"));
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            let logged = fs::read(&wal_file).unwrap();
            assert!(!logged.is_empty());

            // The flush bakes the watermark into the image and
            // truncates the log ...
            assert_eq!(state.flush_dirty(), vec!["sake".to_string()]);
            assert_eq!(fs::metadata(&wal_file).unwrap().len(), 0);
            // ... so putting the pre-truncation bytes back simulates a
            // crash between the image rename and the truncate (or a
            // truncate that simply never ran).
            fs::write(&wal_file, logged).unwrap();
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let (weight, count) = state
            .read_context("sake", |context| {
                let assoc = &context.query(Some("青嶺酒造"), Some("代表銘柄"), Some("青嶺"))[0];
                (assoc.weight, assoc.count)
            })
            .map_err(|_| "read")
            .unwrap();
        // A wrongly replayed record would double both sum and count,
        // leaving their ratio — weight — unchanged at 1.0; count is what
        // actually catches the double-apply the watermark exists to
        // prevent.
        assert_eq!(weight, 1.0);
        assert_eq!(count, 1);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn aliases_and_retractions_ride_the_wal_across_a_restart() {
        let dir = scratch_dir("wal-ops");
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
                        assoc_op("青嶺酒造", "仕込み水", "伏流水", 1.0, Some("第2段落")),
                        assoc_op("青嶺酒造", "仕込み水", "伏流水", 1.0, Some("第5段落")),
                    ],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            state
                .add_aliases(
                    "sake",
                    &BTreeMap::from([("Aomine".to_string(), "青嶺酒造".to_string())]),
                    &BTreeMap::new(),
                )
                .unwrap()
                .unwrap();
            let (touched, _) = state.retract_source("sake", "第5段落").unwrap();
            assert_eq!(touched, 1);
            // No flush — every one of those op kinds must replay.
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let matches = state
            .read_context("sake", |context| {
                context.query(Some("Aomine"), Some("仕込み水"), None)
            })
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(matches.len(), 1, "the alias entry point must replay");
        assert_eq!(matches[0].weight, 1.0, "the retraction must replay");

        let _ = fs::remove_dir_all(dir);
    }

    /// `apply_in_order` stops at the first rejection, but durability
    /// appends the whole batch to the WAL before `operate` even runs.
    /// Without trimming, the untried tail sits on disk indistinguishable
    /// from an applied record, and `ensure_hot`'s replay (`replay_op`,
    /// which continues past a rejection instead of stopping) would try
    /// it independently on the next cold load — applying an op the
    /// client was told never ran.
    #[test]
    fn replay_does_not_apply_ops_a_partial_batch_never_tried() {
        let dir = scratch_dir("wal-partial-tail");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            state
                .add_aliases(
                    "sake",
                    &BTreeMap::from([
                        ("Aomine".to_string(), "青嶺酒造".to_string()),
                        ("AomineShuzo".to_string(), "青嶺酒造".to_string()),
                    ]),
                    &BTreeMap::new(),
                )
                .unwrap()
                .unwrap();

            // A 3-op batch whose middle op is a guaranteed, deterministic
            // rejection (withdrawing an alias that was never registered):
            // `apply_in_order` applies "Aomine", stops at "NONEXISTENT",
            // and never even attempts "AomineShuzo".
            let outcome = state
                .remove_aliases(
                    "sake",
                    &[
                        "Aomine".to_string(),
                        "NONEXISTENT".to_string(),
                        "AomineShuzo".to_string(),
                    ],
                    &[],
                )
                .unwrap();
            let partial = outcome.unwrap_err();
            assert_eq!(partial.applied, 1, "only the leading op ran");
            assert!(
                !partial.full,
                "an absent alias is a conflict, not a capacity error"
            );
            // No flush — the WAL, not the image, must carry this state
            // into the restart below.
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let untried = state
            .read_context("sake", |context| {
                context.query(Some("AomineShuzo"), Some("代表銘柄"), None)
            })
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(
            untried.len(),
            1,
            "the untried alias must survive: the live batch never touched it, so replay must not either"
        );
        let removed = state
            .read_context("sake", |context| {
                context.query(Some("Aomine"), Some("代表銘柄"), None)
            })
            .map_err(|_| "read")
            .unwrap();
        assert!(
            removed.is_empty(),
            "the removal that did apply live must still hold after replay"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_wal_that_cannot_be_written_refuses_the_write() {
        let dir = scratch_dir("wal-refuse");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        // A directory sitting where the log file belongs makes the
        // append fail deterministically.
        fs::create_dir_all(wal_path(&dir, &file_stem("sake"))).unwrap();

        let outcome = state.add_associations(
            "sake",
            vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
            Deadline::unbounded(),
        );
        assert!(matches!(outcome, Err(AccessError::Unpersisted(_))));
        // Refused cleanly: nothing reached the graph.
        let count = state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(count, 0);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn disabling_the_wal_restores_the_flush_window() {
        let dir = scratch_dir("wal-off");
        {
            let state = AppState::boot_with(
                dir.clone(),
                usize::MAX,
                None,
                BootOptions {
                    wal_enabled: false,
                    ..BootOptions::default()
                },
            )
            .unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .add_associations(
                    "sake",
                    vec![assoc_op("青嶺酒造", "代表銘柄", "青嶺", 1.0, None)],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();
            assert!(
                !wal_path(&dir, &file_stem("sake")).exists(),
                "no log may be written when disabled"
            );
            // No flush: with the WAL off, this write is the accepted
            // crash casualty — exactly the pre-WAL posture.
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let count = state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(count, 0);

        let _ = fs::remove_dir_all(dir);
    }

    /// With the WAL off, an entry's image is its ONLY durable home. A
    /// flush must therefore not clear `dirty` before it has published that
    /// image: if it did, an eviction racing the flush would read "clean",
    /// drop the hot context WITHOUT saving, and lose the acknowledged
    /// write outright. Two contexts thrash a one-byte budget — each write
    /// evicts the other while it is still dirty — as a flusher runs flat
    /// out, so staging (unlocked) and eviction interleave every which way.
    /// Every acknowledged write must still be readable, with no restart
    /// and no log to fall back on.
    #[test]
    fn a_flush_racing_an_eviction_loses_no_write_with_the_wal_off() {
        use std::thread;

        let dir = scratch_dir("flush-evict-race");
        let state = AppState::boot_with(
            dir.clone(),
            1, // one byte: only a single context stays resident at a time
            None,
            BootOptions {
                wal_enabled: false,
                ..BootOptions::default()
            },
        )
        .unwrap();
        for name in ["sake", "wine"] {
            state
                .create(name, ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
        }

        const WRITES: usize = 200;
        let writers: Vec<_> = ["sake", "wine"]
            .into_iter()
            .map(|name| {
                let state = state.clone();
                thread::spawn(move || {
                    for index in 0..WRITES {
                        state
                            .add_associations(
                                name,
                                vec![assoc_op(&format!("s{index}"), "l", "o", 1.0, None)],
                                Deadline::unbounded(),
                            )
                            .unwrap()
                            .unwrap();
                    }
                })
            })
            .collect();
        while writers.iter().any(|writer| !writer.is_finished()) {
            state.flush_dirty();
        }
        for writer in writers {
            writer.join().unwrap();
        }

        // Whatever survives lives in the images the flushes and evictions
        // wrote — there is no WAL to replay. Every write must be there.
        for name in ["sake", "wine"] {
            let count = state
                .read_context(name, |context| context.association_count())
                .map_err(|_| "read")
                .unwrap();
            assert_eq!(count, WRITES, "context '{name}' lost writes to the race");
        }

        let _ = fs::remove_dir_all(dir);
    }

    /// With the WAL on, a flush that staged its image before an eviction
    /// cooled the entry — then a later write reloaded it Hot again before
    /// the flush re-locked — used to pass both re-validation checks (slot
    /// is Hot again; eviction never bumped `image_generation`) and
    /// publish its stale snapshot, regressing the on-disk image past
    /// whatever the eviction had already persisted and truncated the WAL
    /// for. Driven directly (`flush_entry` and `evict_entry`, the same
    /// trick as the delete-race test above) instead of hoping a thrash
    /// loop stumbles into the exact interleaving: spinning on `flushing`
    /// forces the claim (and thus the snapshot the flush later
    /// republishes) to land before the write below — `thread::spawn`
    /// scheduling latency otherwise routinely loses that race to this
    /// thread's very next line, and a big seed keeps the flush busy
    /// serializing and staging afterward, which is what gives the write,
    /// the eviction, and the reload below the room to land before it
    /// re-locks to publish.
    #[test]
    fn an_eviction_racing_a_reload_and_a_stale_flush_never_regresses_the_image() {
        use std::thread;
        use std::time::{Duration, Instant};

        let dir = scratch_dir("evict-reload-flush-race");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        // Big enough that `flush_entry`'s serialize-under-lock and
        // stage-unlocked steps together take long enough for the write,
        // the direct eviction, and the reload below to all land before
        // it re-locks — with a tiny graph the flush publishes before any
        // of them get a look in, and the race never opens.
        let seed: Vec<_> = (0..50_000)
            .map(|index| {
                assoc_op(
                    &format!("seed-subject-{index}-xxxxxxxxxxxxxxxxxxxx"),
                    "seed-label-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    &format!("seed-object-{index}-xxxxxxxxxxxxxxxxxxxx"),
                    1.0,
                    None,
                )
            })
            .collect();
        state
            .add_associations("sake", seed, Deadline::unbounded())
            .unwrap()
            .unwrap();

        let entry = state.lookup("sake").unwrap();
        let flusher = {
            let state = state.clone();
            let entry = Arc::clone(&entry);
            thread::spawn(move || state.flush_entry("sake", &entry))
        };
        // `flushing` only flips true once `flush_entry` has locked the
        // entry and entered its claim section, and only flips back at
        // the very end of the call — spinning on it here proves the
        // claim landed before we go on to write "b" below. Without this,
        // `thread::spawn` scheduling latency routinely lets that write
        // land first instead, folding "b" into the flush's own snapshot
        // and closing off the interleaving this test drives at.
        let spun_since = Instant::now();
        while !entry.flushing.load(Ordering::Relaxed) {
            assert!(
                spun_since.elapsed() < Duration::from_secs(5),
                "flusher never reached its claim section"
            );
            thread::yield_now();
        }
        // `flush_entry` claims (clears `dirty`) and serializes the seeded
        // image UNDER the entry lock before it stages unlocked, so this
        // call blocks until that claim is made — it cannot land before
        // the flush captured its (soon to be stale) snapshot.
        state
            .add_associations(
                "sake",
                vec![assoc_op("b", "l", "o", 1.0, None)],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        // Queue the reload-triggering write on the entry lock now, right
        // as the eviction below begins — behind the eviction (which,
        // called directly here, reliably wins the very next acquisition
        // over a freshly spawned thread, the same head start `flushing`'s
        // spin-wait above relies on) but ahead of the flusher's own
        // re-lock attempt, which only comes later once its slower
        // `stage_bytes` finishes. Without this, the flusher — parked on
        // the lock since partway through the eviction below — is
        // `parking_lot`'s fairness-guaranteed next owner the instant the
        // eviction releases it, and it sees Cold (this write not yet
        // applied) instead of the reloaded Hot the bug needs.
        let reloader = {
            let state = state.clone();
            thread::spawn(move || {
                state.add_associations(
                    "sake",
                    vec![assoc_op("c", "l", "o", 1.0, None)],
                    Deadline::unbounded(),
                )
            })
        };
        assert!(
            state.evict_entry("sake", &entry),
            "the write above must leave `sake` dirty and evictable"
        );
        reloader.join().unwrap().unwrap().unwrap();
        let published = flusher.join().unwrap();

        const EXPECTED: usize = 50_000 + 2; // the seed, plus "b" and "c"
        let live = state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(live, EXPECTED, "the race lost a write in memory");
        drop(state);

        let reborn = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let recovered = reborn
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(
            recovered, EXPECTED,
            "a stale flush (published: {published}) regressed the image past \
             what the eviction had already made durable"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn evict_entry_does_not_hold_the_entry_lock_across_its_serialize_and_fsync() {
        use std::thread;
        use std::time::{Duration, Instant};

        let dir = scratch_dir("evict-lock-free-io");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("big", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        // Big enough that flush_entry's stage (write + fsync) step, which
        // evict_entry now goes through instead of saving under its own
        // lock, takes long enough for the try_write below to land while
        // it's still in flight — with a tiny graph the flush publishes
        // before this thread gets a look in and the assertion proves
        // nothing either way.
        let seed: Vec<_> = (0..50_000)
            .map(|index| {
                assoc_op(
                    &format!("seed-subject-{index}-xxxxxxxxxxxxxxxxxxxx"),
                    "seed-label-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                    &format!("seed-object-{index}-xxxxxxxxxxxxxxxxxxxx"),
                    1.0,
                    None,
                )
            })
            .collect();
        state
            .add_associations("big", seed, Deadline::unbounded())
            .unwrap()
            .unwrap();

        let entry = state.lookup("big").unwrap();
        let evictor = {
            let state = state.clone();
            let entry = Arc::clone(&entry);
            thread::spawn(move || state.evict_entry("big", &entry))
        };
        // flush_entry's claim section sets `flushing` AND serializes the
        // graph (context.to_bytes()) before it releases the lock, so
        // spinning on `flushing` alone would just as often land while
        // the lock is still held for that serialize step. Poll instead:
        // the lock must open up SOME time before the evictor finishes —
        // that's the stage_bytes (write + fsync) window, exactly what
        // the old evict_entry held its own write lock across instead,
        // stalling every reader and writer of "big" for as long as the
        // serialize+fsync took.
        let poll_since = Instant::now();
        let mut saw_free_lock = false;
        while !evictor.is_finished() {
            if entry.inner.try_write().is_some() {
                saw_free_lock = true;
                break;
            }
            assert!(
                poll_since.elapsed() < Duration::from_secs(5),
                "evict_entry seems to hold the entry lock for the whole eviction"
            );
            thread::yield_now();
        }
        assert!(
            saw_free_lock,
            "evict_entry must not hold the entry lock across its serialize+fsync \
             (never observed it open up before the evictor finished — if this \
             gets flaky, grow the seed so eviction takes longer)"
        );

        assert!(
            evictor.join().unwrap(),
            "the seeded write must have been evictable"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_compaction_racing_a_periodic_flush_never_loses_the_image() {
        use std::thread;

        // `compact_context` swaps `slot` for a freshly rebuilt `Context`
        // while a `flush_dirty` tick may already be mid-stage for the
        // same entry (bytes read from the OLD `Context`, disk write in
        // flight, entry unlocked). `image_generation` is what lets that
        // flush's republish recognize the swap and back off instead of
        // overwriting the fresh image with the one it just replaced.
        // Hammering writes, compactions, and flush ticks on one context
        // concurrently gives that window many chances to open; without
        // the generation check this reliably drops associations from
        // the persisted image (a stale flush's snapshot predates them).
        let dir = scratch_dir("compact-flush-race");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        const ROUNDS: usize = 200;
        let writer = {
            let state = state.clone();
            thread::spawn(move || {
                for index in 0..ROUNDS {
                    state
                        .add_associations(
                            "sake",
                            vec![assoc_op(&format!("s{index}"), "l", "o", 1.0, None)],
                            Deadline::unbounded(),
                        )
                        .unwrap()
                        .unwrap();
                }
            })
        };
        let compactor = {
            let state = state.clone();
            thread::spawn(move || {
                for _ in 0..ROUNDS {
                    let _ = state.compact_context("sake", Deadline::unbounded());
                }
            })
        };
        while !writer.is_finished() || !compactor.is_finished() {
            state.flush_dirty();
        }
        writer.join().unwrap();
        compactor.join().unwrap();

        let expected = state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(expected, ROUNDS, "the race lost associations in memory");
        // One uncontested flush so the last round's write or compaction
        // is guaranteed durable before the reboot below — the polling
        // loop above only guarantees eventual, not final, convergence.
        state.flush_dirty();
        drop(state);

        let reborn = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let recovered = reborn
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        assert_eq!(
            recovered, expected,
            "the race lost associations from the image"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// A flush has image stage/commit, sidecar stage/commit, and a WAL
    /// truncate. Fail each in turn: publish failures stay dirty and
    /// retry, while a truncate failure is harmless because the image
    /// watermark makes the retained log replay-inert.
    #[test]
    fn every_flush_persistence_failure_retries_or_replays_cleanly() {
        let mut exhausted = false;
        for failure in 0..16 {
            let dir = scratch_dir(&format!("flush-fault-{failure}"));
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

            fail_persistence_ops_after(failure);
            let first = state.flush_dirty();
            let past_end = clear_persistence_fault();
            if !first.iter().any(|name| name == "sake") {
                assert!(
                    state.flush_dirty().iter().any(|name| name == "sake"),
                    "dirty retry did not flush after failure at step {failure}"
                );
            }
            drop(state);

            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            assert_eq!(
                state
                    .read_context("sake", |context| context.association_count())
                    .unwrap(),
                1,
                "flush failure at step {failure} lost or duplicated the write"
            );
            drop(state);
            let _ = fs::remove_dir_all(&dir);

            if past_end {
                exhausted = true;
                break;
            }
        }
        assert!(exhausted, "flush exceeded the sweep bound");
    }

    #[test]
    fn budget_evicts_lru_and_reloads_transparently() {
        let dir = scratch_dir("evict");
        // A budget of one byte: at most the just-used context stays hot.
        let state = AppState::boot(dir.clone(), 1, None).unwrap();
        state
            .create("a", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        state
            .create("b", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        state
            .write_context("a", |context| {
                context.associate("私", "好き", "りんご", 1.0).unwrap();
            })
            .map_err(|_| "write")
            .unwrap();
        // Touching b evicts a (least recently used, and b is protected
        // as the context just used) — flushing a's dirty write first.
        state
            .read_context("b", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        let loaded = loaded_map(&state);
        assert!(!loaded["a"], "a must be evicted");
        assert!(loaded["b"], "the just-used context must stay");

        // The evicted write must have survived the disk roundtrip.
        let recalled = state
            .read_context("a", |context| context.recall("私").len())
            .map_err(|_| "reload")
            .unwrap();
        assert_eq!(recalled, 1);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_second_eviction_of_the_same_entry_frees_nothing() {
        // Two concurrent budget sweeps can snapshot the same candidate;
        // the loser must report `false` so the caller does not subtract
        // the freed bytes from the residency estimate a second time.
        let dir = scratch_dir("double-evict");
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
        // The first eviction does the work and reports it.
        assert!(state.evict_entry("sake", &entry));
        // The second finds the slot cold and every cache already cleared,
        // so it frees nothing — and must say so.
        assert!(
            !state.evict_entry("sake", &entry),
            "a repeat eviction must not claim a second freeing"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// Unlike `write_context` above, `logged_write` appends the whole
    /// batch to the WAL *before* running `operate` — so a panic partway
    /// through the batch leaves the tail durably logged but (absent
    /// this recovery) never applied in memory and never marked `dirty`
    /// either, since the slot stayed Hot holding whatever `operate`
    /// half-finished. A closure that panics only *after* fully applying
    /// every op would not catch this: the in-memory state would already
    /// be complete and correct, panic or not. The real risk is the
    /// batch's untried tail — durably logged, silently missing from
    /// memory. Forcing the slot back to Cold on panic must make the
    /// very next access — no restart needed — rebuild from the image
    /// plus a WAL replay that reapplies that untried tail too.
    #[test]
    fn a_panic_inside_logged_write_forces_a_cold_reload_that_replays_the_wal() {
        let dir = scratch_dir("logged-write-panic-recovery");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();

        let ops = vec![
            WalOp::Associate(assoc_op(
                "青嶺酒造",
                "創業年",
                "1907年",
                1.0,
                Some("第1段落"),
            )),
            WalOp::Associate(assoc_op("青嶺酒造", "所在地", "灘", 1.0, Some("第2段落"))),
        ];

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            state.logged_write(
                "sake",
                &ops,
                |context| {
                    // Only the first op of the batch applies before the
                    // simulated bug fires; the second is durably in the
                    // WAL (appended above, before `operate` ran) but
                    // never reaches memory here.
                    apply_op(context, &ops[0]).unwrap();
                    panic!("simulated failure partway through the batch");
                },
                applied_count,
            )
        }));
        assert!(
            panicked.is_err(),
            "the panic must propagate, not be swallowed"
        );

        let recalled = state
            .read_context("sake", |context| context.recall("青嶺酒造"))
            .expect("the context that panicked stays usable — the lock never poisoned");
        assert_eq!(
            recalled.len(),
            2,
            "both WAL-logged ops must survive the panic, in this same process, with \
             no restart — including the one `operate` never got to apply"
        );
        let objects: BTreeSet<&str> = recalled.iter().map(|fact| fact.object.as_str()).collect();
        assert_eq!(objects, BTreeSet::from(["1907年", "灘"]));

        let _ = fs::remove_dir_all(dir);
    }
}
