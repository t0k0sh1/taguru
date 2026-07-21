use super::*;

impl AppState {
    /// The shared observability registry — the HTTP middleware records
    /// into it, GET /metrics renders it.
    pub fn metrics(&self) -> &Metrics {
        &self.0.metrics
    }

    /// Attempts to become the sole maintenance sweep; `None` means one is
    /// already running. The returned guard is what keeps the server
    /// closed to ordinary traffic — dropping it (by any means) reopens.
    pub fn try_enter_maintenance(&self) -> Option<MaintenanceGuard> {
        self.0
            .metrics
            .try_enter_maintenance()
            .then(|| MaintenanceGuard(self.clone()))
    }

    /// Counts one successful retrieval twice over: the aggregate
    /// searches family (by operation) and the context's own usage row.
    pub fn note_search(&self, op: crate::metrics::SearchOp, name: &str, empty: bool) {
        self.0.metrics.record_search(op, empty);
        self.note_read(name, empty);
    }

    /// Bumps a context's read counters — relaxed atomics only, so a
    /// read is counted without ever waiting on the entry lock. Unknown
    /// names (a delete racing the response) are silently skipped.
    pub fn note_read(&self, name: &str, empty: bool) {
        let Some(entry) = self.lookup(name) else {
            return;
        };
        entry.usage.reads.fetch_add(1, Ordering::Relaxed);
        if empty {
            entry.usage.empty_reads.fetch_add(1, Ordering::Relaxed);
        }
        entry
            .usage
            .last_read_epoch
            .store(unix_now(), Ordering::Relaxed);
        entry.usage_dirty.store(true, Ordering::Relaxed);
    }

    /// Bumps a context's write counter, same contract as
    /// [`AppState::note_read`].
    pub fn note_write(&self, name: &str) {
        let Some(entry) = self.lookup(name) else {
            return;
        };
        entry.usage.writes.fetch_add(1, Ordering::Relaxed);
        entry
            .usage
            .last_write_epoch
            .store(unix_now(), Ordering::Relaxed);
        entry.usage_dirty.store(true, Ordering::Relaxed);
    }

    /// Persists every usage snapshot the sidecars have not seen — the
    /// graceful-shutdown sweep behind the crash-loss contract on
    /// [`ContextUsage`]. Purely-read contexts never flush, so without
    /// this their counters would evaporate on every restart. Runs
    /// after the final [`AppState::flush_dirty`], so the stats written
    /// beside the counters are current.
    pub fn persist_usage(&self) {
        // A replica's usage counters are its own ephemeral view (its
        // reads, over sidecar seeds the writer shipped); persisting
        // them would diverge the cache for the next diff to claw back.
        if self.is_replica() {
            return;
        }
        for (name, entry) in self.snapshot() {
            if !entry.usage_dirty.swap(false, Ordering::Relaxed) {
                continue;
            }
            let Some(guard) = entry.lock_unless_deleted() else {
                continue;
            };
            let outcome = write_meta(
                &self.0.data_dir,
                &file_stem(&name),
                &guard.meta,
                &guard.stats,
                &entry.usage.snapshot(),
                entry.revision_snapshot(&guard),
            );
            if let Err(error) = outcome {
                entry.usage_dirty.store(true, Ordering::Relaxed);
                tracing::warn!("usage counters for '{name}' not persisted: {error}");
            }
        }
    }

    /// Re-stats every context's non-WAL files into its entry's disk
    /// cache — the flush-time half of the per-context gauges (issue
    /// #137). Free while `TAGURU_METRICS_PER_CONTEXT` is off; otherwise
    /// it runs from boot, the flusher's tick, and `POST /flush` (all
    /// via [`AppState::flush_dirty`]) — never from a scrape, which
    /// must not walk the data directory. The WAL lanes are absent on
    /// purpose: their live bookkeeping ([`EntryInner::wal_bytes`],
    /// `passages_wal_bytes`) already serves the scrape, so the
    /// per-context `wal` series sum exactly to the global gauges.
    ///
    /// Lag is the contract, not a bug: a size lands on the scrape up
    /// to one flush interval after it lands on disk, the trade that
    /// buys zero scrape-time filesystem work.
    pub fn refresh_disk_usage(&self) {
        // The gauges are one reader of the snapshot this fills; a
        // declared storage ceiling is the other — its gates sum these
        // lanes on every growth write — so quotas keep the sweep alive
        // even with the gauges off.
        let quotas_need_disk = self
            .0
            .context_quotas
            .values()
            .any(|quota| quota.storage_bytes.is_some());
        if self.0.per_context_metrics == PerContextMetrics::Off && !quotas_need_disk {
            return;
        }
        let dir = &self.0.data_dir;
        let len = |path: PathBuf| fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
        for (name, entry) in self.snapshot() {
            let stem = file_stem(&name);
            let usage = ContextDiskUsage {
                image_bytes: len(image_path(dir, &stem)),
                passages_bytes: len(passages_path(dir, &stem)),
                sidecar_bytes: len(meta_path(dir, &stem))
                    + len(sources_path(dir, &stem))
                    + len(vectors_path(dir, &stem))
                    + len(pvectors_path(dir, &stem))
                    + len(bm25_path(dir, &stem)),
            };
            // No tombstone check: stats only read, and a racing delete
            // just leaves a snapshot nobody will render — the entry is
            // already out of the registry the next scrape reads.
            *entry.disk.lock() = usage;
        }
    }

    /// Point-in-time gauges for a scrape, computed from the registry
    /// so they cannot drift: how many contexts exist, how many are
    /// resident, and the resident-bytes estimate (loaded graphs plus
    /// cached vector stores — the same accounting the cache budget
    /// uses).
    pub fn gauge_snapshot(&self) -> GaugeSnapshot {
        let snapshot = self.snapshot();
        let contexts_registered = snapshot.len() as u64;
        let mut contexts_resident = 0u64;
        let mut resident_bytes = 0u64;
        let mut wal_bytes = 0u64;
        let mut passages_wal_bytes = 0u64;
        let mut dead_edges_total = 0u64;
        let mut dead_attributions_total = 0u64;
        let mut arena_slack_total = 0u64;
        let mut unsourced_edges_total = 0u64;
        let mut unsourced_weight_total = 0.0f64;
        // Per-context rows ride the same single pass as the fleet-wide
        // sums, from the same per-entry reads — the two can never
        // disagree about one scrape's instant.
        let collect_rows = self.0.per_context_metrics != PerContextMetrics::Off;
        let mut per_context = Vec::new();
        for (name, entry) in snapshot {
            let inner = entry.inner.read();
            let mut graph_footprint = 0u64;
            // (concepts, associations, labels, sources) — live for hot,
            // the last-saved stats snapshot for cold, exactly the
            // semantics `GET /contexts` serves.
            let counts;
            if let Slot::Hot(context) = &inner.slot {
                contexts_resident += 1;
                graph_footprint = context.footprint() as u64;
                dead_edges_total += context.dead_edges() as u64;
                dead_attributions_total += context.dead_attributions() as u64;
                arena_slack_total += context.arena_slack() as u64;
                let (unsourced_edges, unsourced_weight) = context.unsourced_summary();
                unsourced_edges_total += unsourced_edges as u64;
                unsourced_weight_total += unsourced_weight;
                counts = (
                    context.concept_count() as u64,
                    context.association_count() as u64,
                    context.label_count() as u64,
                    context.source_count() as u64,
                );
            } else {
                // Cold (or mid-delete): the graph is not in memory, so
                // this stays whatever `inner.stats` last cached at
                // eviction/compact/flush — the same staleness trade
                // `resident_bytes`'s hot-only branch already accepts,
                // just from the other side (dead weight persists on
                // disk whether or not the context is loaded).
                dead_edges_total += inner.stats.dead_edges as u64;
                dead_attributions_total += inner.stats.dead_attributions as u64;
                arena_slack_total += inner.stats.arena_slack as u64;
                unsourced_edges_total += inner.stats.unsourced_edges as u64;
                unsourced_weight_total += inner.stats.unsourced_weight;
                counts = (
                    inner.stats.concepts as u64,
                    inner.stats.associations as u64,
                    inner.stats.labels as u64,
                    inner.stats.sources as u64,
                );
            }
            resident_bytes += graph_footprint;
            let entry_wal_bytes = inner.wal_bytes;
            wal_bytes += entry_wal_bytes;
            let cold_passages_wal_bytes = inner.passages_wal_bytes;
            let pinned = inner.meta.pinned;
            drop(inner);
            let cache_footprint = entry.vectors_footprint() as u64
                + entry.passages_footprint() as u64
                + entry.bm25_footprint() as u64
                + entry.passage_vectors_footprint() as u64;
            resident_bytes += cache_footprint;
            // A resident store knows its pending log; a cold one uses
            // the value `evict_entry` cached on the way down — the
            // gauge must not go blind just because a context was
            // evicted, nor re-`stat` the log on every scrape.
            let entry_passages_wal_bytes = entry
                .passages
                .lock()
                .as_ref()
                .map(|store| store.pending_log_bytes())
                .unwrap_or(cold_passages_wal_bytes);
            passages_wal_bytes += entry_passages_wal_bytes;
            if collect_rows {
                let disk = *entry.disk.lock();
                // Declared ceilings ride the row (issue #136) so
                // usage-vs-quota is one division against the sibling
                // series, under the same knob and Top-N cut.
                let quota = self.0.context_quotas.get(&name);
                per_context.push(ContextGaugeRow {
                    name,
                    pinned,
                    resident_bytes: graph_footprint + cache_footprint,
                    disk_image_bytes: disk.image_bytes,
                    disk_wal_bytes: entry_wal_bytes,
                    disk_passages_bytes: disk.passages_bytes,
                    disk_passages_wal_bytes: entry_passages_wal_bytes,
                    disk_sidecar_bytes: disk.sidecar_bytes,
                    quota_storage_bytes: quota.and_then(|quota| quota.storage_bytes),
                    quota_cache_bytes: quota.and_then(|quota| quota.cache_bytes),
                    concepts: counts.0,
                    associations: counts.1,
                    labels: counts.2,
                    sources: counts.3,
                });
            }
        }
        if let PerContextMetrics::Top(keep) = self.0.per_context_metrics {
            // Rank by total disk bytes (ties by name, so equal sizes
            // cannot flap), cut, then restore name order — the render
            // stays deterministic and diff-friendly either mode.
            per_context.sort_by(|a, b| {
                b.disk_total_bytes()
                    .cmp(&a.disk_total_bytes())
                    .then_with(|| a.name.cmp(&b.name))
            });
            per_context.truncate(keep);
            per_context.sort_by(|a, b| a.name.cmp(&b.name));
        }
        let (retrieval_cache_entries, retrieval_cache_bytes) = self.retrieval_cache_gauges();
        let semantic_cache_entries = self.semantic_cache_entries();
        GaugeSnapshot {
            contexts_registered,
            groups_registered: self.0.groups.read().len() as u64,
            contexts_resident,
            resident_bytes,
            wal_bytes,
            passages_wal_bytes,
            dead_edges_total,
            dead_attributions_total,
            arena_slack_total,
            unsourced_edges_total,
            unsourced_weight_total,
            embed_breaker: self
                .0
                .embed_breaker
                .as_ref()
                .map(crate::embedding::EmbedBreaker::snapshot),
            retrieval_cache_entries,
            retrieval_cache_bytes,
            semantic_cache_entries,
            per_context,
        }
    }

    pub fn context_count(&self) -> usize {
        self.0.registry.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::test_support::{assoc_op, plain, rendered, scratch_dir};

    #[test]
    fn usage_notes_accumulate_and_survive_a_reboot_via_the_shutdown_sweep() {
        let dir = scratch_dir("usage-sweep");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state.note_read("sake", false);
            state.note_read("sake", true);
            state.note_write("sake");
            let usage = state.directory_entry("sake").unwrap().usage;
            assert_eq!((usage.reads, usage.empty_reads, usage.writes), (2, 1, 1));
            assert!(usage.last_read_epoch > 0);
            assert!(usage.last_write_epoch > 0);
            // Nothing marked the graph dirty since create, so no flush
            // will run: the sweep alone must put the counters on disk.
            state.persist_usage();
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let usage = state.directory_entry("sake").unwrap().usage;
        assert_eq!((usage.reads, usage.empty_reads, usage.writes), (2, 1, 1));
        let _ = fs::remove_dir_all(&dir);
    }

    /// [`AppState::try_enter_maintenance`] is a one-shot CAS: a second
    /// call fails while the guard lives, and dropping it (the only way
    /// to release — success, a deadline, or a panic unwind all reach the
    /// same `Drop`) reopens the server for the next sweep.
    #[test]
    fn maintenance_guard_is_a_one_shot_cas_released_by_drop() {
        let dir = scratch_dir("maint-guard");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

        let guard = state.try_enter_maintenance().expect("first entry succeeds");
        assert!(state.metrics().maintenance_active());
        assert!(
            state.try_enter_maintenance().is_none(),
            "a second sweep must not overlap the first"
        );

        drop(guard);
        assert!(!state.metrics().maintenance_active());
        let _second = state
            .try_enter_maintenance()
            .expect("reopened once the guard drops");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn dead_weight_gauges_track_hot_and_cold_contexts() {
        let dir = scratch_dir("dead-weight-gauges");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();

        let baseline = state.gauge_snapshot();
        assert_eq!(baseline.dead_edges_total, 0);
        assert_eq!(baseline.dead_attributions_total, 0);
        assert_eq!(baseline.arena_slack_total, 0);
        assert_eq!(baseline.unsourced_edges_total, 0);
        assert_eq!(baseline.unsourced_weight_total, 0.0);

        // One sourced association, later retracted outright: one dead
        // edge, one unlinked attribution.
        state
            .add_associations(
                "sake",
                vec![assoc_op("蔵", "廃", "旧", 1.0, Some("x.md"))],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            state.retract_association("sake", "蔵", "廃", "旧").unwrap(),
            Some(1)
        );
        // One sourceless association: pure unsourced weight, nothing else.
        state
            .add_associations(
                "sake",
                vec![assoc_op("蔵", "銘柄", "青嶺", 2.5, None)],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        // One alias registered then withdrawn: its spelling's bytes
        // become arena slack.
        state
            .add_aliases(
                "sake",
                &BTreeMap::from([("Aomine".to_string(), "蔵".to_string())]),
                &BTreeMap::new(),
            )
            .unwrap()
            .unwrap();
        state
            .remove_aliases("sake", &["Aomine".to_string()], &[])
            .unwrap()
            .unwrap();

        let hot = state.gauge_snapshot();
        assert_eq!(hot.dead_edges_total, 1);
        assert_eq!(hot.dead_attributions_total, 1);
        assert_eq!(hot.arena_slack_total, "Aomine".len() as u64);
        assert_eq!(hot.unsourced_edges_total, 1);
        assert_eq!(hot.unsourced_weight_total, 2.5);
        let text = rendered(&state);
        assert!(text.contains("taguru_dead_edges 1"));
        assert!(text.contains("taguru_dead_attributions 1"));
        assert!(text.contains(&format!("taguru_arena_slack_bytes {}", "Aomine".len())));
        assert!(text.contains("taguru_unsourced_edges 1"));
        assert!(text.contains("taguru_unsourced_weight 2.5"));

        // Eviction to cold must not lose the totals — the gauge falls
        // back to the persisted `ContextStats` snapshot.
        let entry = state.lookup("sake").unwrap();
        assert!(state.evict_entry("sake", &entry));
        let cold = state.gauge_snapshot();
        assert_eq!(cold.dead_edges_total, hot.dead_edges_total);
        assert_eq!(cold.dead_attributions_total, hot.dead_attributions_total);
        assert_eq!(cold.arena_slack_total, hot.arena_slack_total);
        assert_eq!(cold.unsourced_edges_total, hot.unsourced_edges_total);
        assert_eq!(cold.unsourced_weight_total, hot.unsourced_weight_total);

        let _ = fs::remove_dir_all(dir);
    }

    /// The per-context rows (#137) measure disk at flush time — never
    /// at scrape time — carry live counts for hot contexts and saved
    /// ones for cold, and `Top(n)` cuts by total disk bytes.
    #[test]
    fn per_context_gauges_measure_at_flush_and_cut_top_n_by_disk() {
        let dir = scratch_dir("per-context-gauges");
        {
            let state = AppState::boot_with(
                dir.clone(),
                usize::MAX,
                None,
                BootOptions {
                    per_context_metrics: PerContextMetrics::All,
                    ..BootOptions::default()
                },
            )
            .unwrap();
            state
                .create("big", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .create(
                    "small",
                    ContextMeta {
                        pinned: true,
                        ..ContextMeta::default()
                    },
                )
                .map_err(|_| "create")
                .unwrap();
            let seed: Vec<_> = (0..200)
                .map(|index| {
                    assoc_op(
                        &format!("subject-{index}"),
                        "label",
                        &format!("object-{index}"),
                        1.0,
                        Some("s.md"),
                    )
                })
                .collect();
            state
                .add_associations("big", seed, Deadline::unbounded())
                .unwrap()
                .unwrap();
            state
                .add_associations(
                    "small",
                    vec![assoc_op("a", "l", "b", 1.0, None)],
                    Deadline::unbounded(),
                )
                .unwrap()
                .unwrap();

            // Before any flush: the images EXIST on disk (create
            // persisted them after the boot sweep ran), yet the disk
            // gauges still read zero — the behavioral proof that a
            // scrape never stats the data directory.
            let before = state.gauge_snapshot();
            assert_eq!(before.per_context.len(), 2);
            let row = |snapshot: &GaugeSnapshot, name: &str| -> ContextGaugeRow {
                snapshot
                    .per_context
                    .iter()
                    .find(|row| row.name == name)
                    .unwrap()
                    .clone()
            };
            assert!(
                fs::metadata(image_path(&dir, &file_stem("big")))
                    .unwrap()
                    .len()
                    > 0
            );
            let big = row(&before, "big");
            assert_eq!(big.disk_image_bytes, 0);
            // Hot rows: live counts and real residency.
            assert_eq!(big.associations, 200);
            assert_eq!(big.sources, 1);
            assert!(big.resident_bytes > 0);
            assert!(!big.pinned);
            assert!(row(&before, "small").pinned);

            // The flush sweep publishes the real file sizes — the very
            // bytes `to_bytes()` staged, which is also `estimate`'s
            // measuring stick.
            state.flush_dirty();
            let after = state.gauge_snapshot();
            let big = row(&after, "big");
            let image_len = fs::metadata(image_path(&dir, &file_stem("big")))
                .unwrap()
                .len();
            assert_eq!(big.disk_image_bytes, image_len);
            assert!(big.disk_sidecar_bytes > 0, "the meta sidecar has bytes");
            assert!(
                big.disk_total_bytes() > row(&after, "small").disk_total_bytes(),
                "200 associations outweigh 1"
            );
        }
        // Reboot with Top(1): the cut keeps the bigger context, and the
        // cold row's counts come from the saved stats snapshot.
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            None,
            BootOptions {
                per_context_metrics: PerContextMetrics::Top(1),
                ..BootOptions::default()
            },
        )
        .unwrap();
        let snapshot = state.gauge_snapshot();
        assert_eq!(snapshot.per_context.len(), 1, "Top(1) cut to one row");
        let row = &snapshot.per_context[0];
        assert_eq!(row.name, "big");
        assert_eq!(row.associations, 200, "cold counts read the sidecar");
        assert_eq!(row.resident_bytes, 0, "a cold context holds nothing");
        assert!(row.disk_image_bytes > 0, "the boot sweep seeded disk sizes");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn cache_budget_accounts_for_resident_passage_text() {
        let dir = scratch_dir("passage-budget");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .create("sake", ContextMeta::default())
            .map_err(|_| "create")
            .unwrap();
        let before = state.gauge_snapshot().resident_bytes;
        let mut passages = BTreeMap::new();
        passages.insert("大きな段落".to_string(), "あ".repeat(300_000));
        state
            .store_passages("sake", plain(passages))
            .unwrap()
            .unwrap();
        let after = state.gauge_snapshot().resident_bytes;
        assert!(
            after >= before + 900_000,
            "resident passage text must count against the budget \
             (before {before}, after {after})"
        );

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn passages_wal_bytes_stays_correct_after_eviction() {
        let dir = scratch_dir("passages-wal-gauge");
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
        assert!(
            state.gauge_snapshot().passages_wal_bytes > 0,
            "a freshly written passage log must show up as pending"
        );

        let entry = state.lookup("sake").unwrap();
        assert!(state.evict_entry("sake", &entry));
        // `evict_entry` compacts the store on the way down and caches
        // the resulting (now-zero) pending-log size on `EntryInner`;
        // the gauge must read that cached value rather than going
        // blind — or re-`stat`ing the log — once the context is cold.
        assert_eq!(
            state.gauge_snapshot().passages_wal_bytes,
            0,
            "eviction compacts the log, and the gauge must reflect that while cold"
        );

        let _ = fs::remove_dir_all(dir);
    }
}
