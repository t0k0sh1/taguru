//! The `record_*`/`note_*` family: every write side of the registry,
//! the in-flight admission gate, and the maintenance-sweep latch.
//! `render_prometheus` (the read side) lives in `prometheus.rs`.

use super::*;

impl Metrics {
    /// Counts one request in, refusing past `limit` (0 = no ceiling,
    /// count only). Compare-and-swap so two racing admissions cannot
    /// both squeeze under the ceiling.
    pub(crate) fn admit_inflight(&self, limit: usize) -> bool {
        let mut current = self.inflight.load(Ordering::Relaxed);
        loop {
            if limit != 0 && current >= limit {
                return false;
            }
            match self.inflight.compare_exchange_weak(
                current,
                current + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    pub(crate) fn release_inflight(&self) {
        self.inflight.fetch_sub(1, Ordering::Relaxed);
    }

    /// Requests currently inside the stack, this call included if it was
    /// admitted through `admit_inflight`. A maintenance sweep polls this
    /// down to 1 (itself) to know every other request has drained.
    pub(crate) fn inflight_count(&self) -> usize {
        self.inflight.load(Ordering::Relaxed)
    }

    /// Attempts to become the sole maintenance sweep; `false` means one is
    /// already running, so the caller should answer 409 rather than queue
    /// behind it.
    pub(crate) fn try_enter_maintenance(&self) -> bool {
        self.maintenance
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
    }

    /// Whether a maintenance sweep currently holds the server closed to
    /// ordinary traffic.
    pub fn maintenance_active(&self) -> bool {
        self.maintenance.load(Ordering::Relaxed)
    }

    /// Reopens the server after a maintenance sweep. Idempotent, so it is
    /// safe to call unconditionally from a `Drop` guard on every exit path
    /// — success, deadline, or panic unwind.
    pub(crate) fn exit_maintenance(&self) {
        self.maintenance.store(false, Ordering::Release);
    }

    pub(crate) fn record_embed_latency(&self, elapsed: Duration) {
        self.embed_latency.observe(elapsed);
    }

    /// One `rerank::drive` call (#307): its wall-clock duration —
    /// including a short-circuit that never touched the provider
    /// (`not_configured`/`model_mismatch`/`empty_pool`/`circuit_open`
    /// all land here too, as near-zero samples, unlike
    /// `embed_latency`'s own call site which gates the histogram on a
    /// breaker pre-check before ever timing anything) — and which of
    /// [`RerankOutcomeKind`]'s fixed labels it ended as.
    pub(crate) fn record_rerank(&self, token: &str, elapsed: Duration) {
        self.rerank_latency.observe(elapsed);
        self.rerank_outcomes[RerankOutcomeKind::from_token(token) as usize]
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_shed(&self) {
        self.requests_shed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_http(&self, method: &str, route: &str, status: u16, elapsed: Duration) {
        let stat = self.route_stat(method, route);
        stat.latency.observe(elapsed);
        if let Some(counter) = stat.by_status.read().get(&status) {
            counter.fetch_add(1, Ordering::Relaxed);
            return;
        }
        stat.by_status
            .write()
            .entry(status)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    fn route_stat(&self, method: &str, route: &str) -> Arc<RouteStat> {
        if let Some(stat) = self
            .http
            .read()
            .get(&(method.to_string(), route.to_string()))
        {
            return Arc::clone(stat);
        }
        Arc::clone(
            self.http
                .write()
                .entry((method.to_string(), route.to_string()))
                .or_default(),
        )
    }

    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cache_load(&self, ok: bool) {
        let counter = if ok {
            &self.cache_loads_ok
        } else {
            &self.cache_loads_failed
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_eviction(&self, ok: bool) {
        let counter = if ok {
            &self.evictions_ok
        } else {
            &self.evictions_failed
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_flush(&self, name: &str, ok: bool) {
        let counter = if ok {
            &self.flush_ok
        } else {
            &self.flush_failed
        };
        counter.fetch_add(1, Ordering::Relaxed);
        // Track WHICH contexts are failing, not just the last outcome:
        // a single global bit let context B's success erase context A's
        // still-unhealed failure (and one transient failure flip the whole
        // server to 503). Health stays degraded while the set is non-empty.
        {
            let mut failing = self.flush_failing.lock();
            if ok {
                failing.remove(name);
            } else {
                failing.insert(name.to_string());
            }
            self.flush_degraded
                .store(!failing.is_empty(), Ordering::Relaxed);
        }
        if ok {
            let epoch = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|since| since.as_secs())
                .unwrap_or(0);
            self.last_flush_success_epoch
                .store(epoch, Ordering::Relaxed);
        }
    }

    /// Whether every context's most recent image flush succeeded (true
    /// when none has run yet — an idle server is a healthy server), AND
    /// the flusher loop itself is still making it to a tick's end rather
    /// than panicking out from under `/health`'s only signal.
    pub fn flush_is_healthy(&self) -> bool {
        !self.flush_degraded.load(Ordering::Relaxed)
            && !self.flusher_panicked.load(Ordering::Relaxed)
    }

    /// Unix seconds of the last successful flush; 0 when none yet.
    pub fn last_flush_success_epoch(&self) -> u64 {
        self.last_flush_success_epoch.load(Ordering::Relaxed)
    }

    /// Record whether the flusher's most recent tick completed without
    /// panicking. Called from `spawn_flusher`'s `catch_unwind` boundary
    /// once per tick, success or not — so a later clean tick clears the
    /// flag the same way a later clean flush clears `flush_degraded`.
    pub fn record_flusher_tick(&self, ok: bool) {
        self.flusher_panicked.store(!ok, Ordering::Relaxed);
    }

    /// Whether the flusher's most recent tick panicked. Distinct from
    /// `flush_is_healthy` (which folds this in) so `health()` can pick a
    /// reason message that names the actual fault.
    pub fn flusher_panicked(&self) -> bool {
        self.flusher_panicked.load(Ordering::Relaxed)
    }

    /// Record one ratio-triggered auto-compaction (issue #135):
    /// `Some(bytes)` — the resident bytes the rebuild shed — counts a
    /// success and stamps the last-success clock; `None` counts a
    /// failure. Failures never latch anything: the candidate's ratio
    /// still exceeds the trigger, so the next tick retries it, and the
    /// counter pair is what makes a stuck retry loop visible.
    pub fn record_auto_compaction(&self, reclaimed_bytes: Option<u64>) {
        match reclaimed_bytes {
            Some(bytes) => {
                self.auto_compact_ok.fetch_add(1, Ordering::Relaxed);
                self.auto_compact_reclaimed_bytes
                    .fetch_add(bytes, Ordering::Relaxed);
                let epoch = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|since| since.as_secs())
                    .unwrap_or(0);
                self.auto_compact_last_success_epoch
                    .store(epoch, Ordering::Relaxed);
            }
            None => {
                self.auto_compact_failed.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Count one growth write refused at a declared storage ceiling
    /// (issue #136) — called by the gate that refuses, never by the
    /// error mapping, so a refusal is counted exactly once wherever
    /// it surfaces.
    pub fn record_storage_quota_refusal(&self) {
        self.storage_quota_refusals.fetch_add(1, Ordering::Relaxed);
    }

    /// Count one keyring reload attempt (issue #134) by whether a
    /// table (possibly identical) was armed or the previous one kept.
    pub fn record_keyring_reload(&self, applied: bool) {
        let counter = if applied {
            &self.keyring_reloads
        } else {
            &self.keyring_reload_refusals
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_wal_append(&self, ok: bool) {
        let counter = if ok {
            &self.wal_appends_ok
        } else {
            &self.wal_appends_failed
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_embed_refresh(&self, ok: bool) {
        let counter = if ok {
            &self.embed_refresh_ok
        } else {
            &self.embed_refresh_failed
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_embed_resolve(&self, ok: bool) {
        let counter = if ok {
            &self.embed_resolve_ok
        } else {
            &self.embed_resolve_failed
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// One width-triggered sidecar wipe: the provider changed vector
    /// width behind an unchanged model name, and the refresh discarded
    /// and re-embedded the whole gloss store. Counted beside the warn
    /// line so the rebuild (and its provider spend) is graphable, not
    /// just greppable.
    pub fn record_gloss_width_rebuild(&self) {
        self.gloss_width_rebuilds.fetch_add(1, Ordering::Relaxed);
    }

    /// The passage-store twin of [`Self::record_gloss_width_rebuild`].
    pub fn record_passage_width_rebuild(&self) {
        self.passage_width_rebuilds.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_error(&self, kind: ErrorKind) {
        let counter = match kind {
            ErrorKind::Load => &self.errors_load,
            ErrorKind::WalRefused => &self.errors_wal_refused,
            ErrorKind::Io => &self.errors_io,
            ErrorKind::Panic => &self.errors_panic,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }

    /// One successful retrieval, split by whether it matched anything.
    /// Error responses never land here — a 500 is not an empty search.
    pub fn record_search(&self, op: SearchOp, empty: bool) {
        self.searches[op as usize][usize::from(empty)].fetch_add(1, Ordering::Relaxed);
    }

    /// One retrieval-cache consultation. Only consultations count: a
    /// disabled cache, or a request the cache cannot key (a target
    /// vanishing mid-request), records nothing rather than a fake miss.
    pub fn record_retrieval_cache(&self, op: RetrievalCacheOp, hit: bool) {
        self.retrieval_cache[op as usize][usize::from(!hit)].fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_semantic_cache(&self, outcome: SemanticCacheOutcome) {
        self.semantic_cache[outcome as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// A cache hit's replay of [`Metrics::record_passage_hit`]: the
    /// fresh path records lane contributions per served hit, so a hit
    /// serving the same response must land the same counts — in bulk,
    /// since the cached entry stores tallies, not hits.
    pub fn record_passage_hit_counts(&self, bm25_only: u64, both_lanes: u64, vector_only: u64) {
        self.passage_hits_bm25_only
            .fetch_add(bm25_only, Ordering::Relaxed);
        self.passage_hits_both_lanes
            .fetch_add(both_lanes, Ordering::Relaxed);
        self.passage_hits_vector_only
            .fetch_add(vector_only, Ordering::Relaxed);
    }

    pub fn record_resolve_tier(&self, tier: ResolveTier) {
        self.resolve_tiers[tier as usize].fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_schema_check(&self, outcome: SchemaOutcome) {
        self.schema_checks[outcome as usize].fetch_add(1, Ordering::Relaxed);
    }

    /// One served passage-search hit, by which lane(s) put it there. A
    /// hit carries at least one lane by construction; anything else is
    /// counted nowhere rather than inventing a fourth label.
    pub fn record_passage_hit(&self, bm25: bool, vector: bool) {
        match (bm25, vector) {
            (true, true) => &self.passage_hits_both_lanes,
            (true, false) => &self.passage_hits_bm25_only,
            (false, true) => &self.passage_hits_vector_only,
            (false, false) => return,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_replication_upload(&self) {
        self.replication_uploads.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_replication_error(&self) {
        self.replication_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// One cycle that shipped everything it found. Deliberately NOT
    /// part of `/health`: a degraded bucket must page whoever watches
    /// the dashboards, never convince an orchestrator to restart a
    /// server whose local durability is fine.
    pub fn record_replication_success(&self) {
        let epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|since| since.as_secs())
            .unwrap_or(0);
        self.replication_last_success_epoch
            .store(epoch, Ordering::Relaxed);
    }

    /// Latches the fenced flag — the metric half of the shipper's
    /// fail-stop (the audit line is the other half). Never cleared:
    /// only a restart re-contests the bucket.
    pub fn record_replication_fenced(&self) {
        self.replication_fenced.store(true, Ordering::Relaxed);
    }

    /// Refreshes one lane's lag series, keyed (context, lane).
    pub fn note_replication_lane(
        &self,
        context: &str,
        lane: &'static str,
        behind_records: u64,
        age_secs: u64,
    ) {
        self.replication_lag.lock().insert(
            (context.to_string(), lane),
            ReplicationLag {
                behind_records,
                age_secs,
            },
        );
    }

    /// Drops a deleted context's lane series so the scrape does not
    /// carry ghost labels forever.
    pub fn forget_replication_lane(&self, context: &str, lane: &'static str) {
        self.replication_lag
            .lock()
            .remove(&(context.to_string(), lane));
    }

    pub(super) fn unix_now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|since| since.as_secs())
            .unwrap_or(0)
    }

    /// Flips this process's scrape into replica shape — set once at
    /// boot under `serve --replica`, never cleared (a role is a boot
    /// decision, not a runtime one).
    pub fn set_replica_mode(&self) {
        self.replica_mode.store(true, Ordering::Relaxed);
    }

    /// One tailer poll finished; errors count, successes stamp the
    /// freshness gauge. Like the shipper's counters, deliberately NOT
    /// part of `/health`: an unreachable bucket must page whoever
    /// watches dashboards, not convince an orchestrator to restart a
    /// replica that is still serving its watermark fine.
    pub fn record_replica_poll(&self, ok: bool) {
        if ok {
            self.replica_last_poll_epoch
                .store(Self::unix_now(), Ordering::Relaxed);
        } else {
            self.replica_poll_errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The generation this replica currently follows (its hydration
    /// target after the latest retarget).
    pub fn note_replica_generation(&self, generation: u64) {
        self.replica_generation.store(generation, Ordering::Relaxed);
    }

    /// The newest complete manifest's store-clock `last_modified`, as
    /// seen by the latest poll — `time() - this` plus the poll
    /// interval bounds this replica's staleness.
    pub fn note_replica_manifest(&self, modified: SystemTime) {
        let epoch = modified
            .duration_since(UNIX_EPOCH)
            .map(|since| since.as_secs())
            .unwrap_or(0);
        self.replica_manifest_epoch.store(epoch, Ordering::Relaxed);
    }

    /// One lane fully applied: applied == shipped, gap closed.
    pub fn note_replica_lane(
        &self,
        context: &str,
        lane: &'static str,
        applied_seq: u64,
        shipped_seq: u64,
    ) {
        let mut lag = self.replica_lag.lock();
        let entry = lag.entry((context.to_string(), lane)).or_default();
        entry.applied_seq = applied_seq;
        entry.shipped_seq = shipped_seq;
        entry.behind_since_epoch = if applied_seq >= shipped_seq {
            0
        } else if entry.behind_since_epoch == 0 {
            Self::unix_now()
        } else {
            entry.behind_since_epoch
        };
    }

    /// The shipped side alone — for a lane whose family could not be
    /// applied this poll: the applied seq stays where it was (or at 0
    /// for a lane never applied), and the age starts counting.
    pub fn note_replica_shipped(&self, context: &str, lane: &'static str, shipped_seq: u64) {
        let mut lag = self.replica_lag.lock();
        let entry = lag.entry((context.to_string(), lane)).or_default();
        entry.shipped_seq = shipped_seq;
        // Same three-way move as `note_replica_lane`: today shipped
        // seqs only grow within a lineage, so the caught-up arm can't
        // fire here — but the reset must not silently depend on that.
        entry.behind_since_epoch = if entry.applied_seq >= shipped_seq {
            0
        } else if entry.behind_since_epoch == 0 {
            Self::unix_now()
        } else {
            entry.behind_since_epoch
        };
    }

    /// Drops a vanished context's replica lag rows (both lanes).
    pub fn forget_replica_context(&self, context: &str) {
        let mut lag = self.replica_lag.lock();
        lag.remove(&(context.to_string(), "graph"));
        lag.remove(&(context.to_string(), "passages"));
    }

    /// Clears every replica lag row — the tailer's move on a
    /// generation switch. Applied seqs are meaningful only within one
    /// lineage: a successor that started from an older watermark (a
    /// promotion that lost the deposed writer's tail) ships LOWER
    /// seqs, and a predecessor's applied value surviving beside them
    /// would read as caught-up on a lane that has not applied the new
    /// lineage at all. The rows rebuild from the switch's own apply
    /// results, so a family that fails to land shows its gap from
    /// zero instead of a stale success.
    pub fn reset_replica_lanes(&self) {
        self.replica_lag.lock().clear();
    }
}
