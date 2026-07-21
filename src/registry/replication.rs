use super::*;

impl AppState {
    /// Under `serve --replica`: the write-refusal's payload, and the
    /// flag every locally-derived persistence path checks.
    pub fn replica(&self) -> Option<&Arc<crate::replica::ReplicaInfo>> {
        self.0.replica.as_ref()
    }

    pub fn is_replica(&self) -> bool {
        self.0.replica.is_some()
    }

    /// Replica tailer: registers a context a newer manifest introduced
    /// — the runtime twin of boot's hydrator registration. Idempotent;
    /// the sidecar meta is already local (the shared hydration pass
    /// lands every meta before families are touched).
    pub(crate) fn replica_register(&self, stem: &str) {
        let Some(name) = name_from_stem(stem) else {
            return;
        };
        let mut registry = self.0.registry.write();
        registry.entry(name).or_insert_with(|| {
            let MetaFile {
                meta,
                stats,
                usage,
                revision,
            } = read_meta_file(&self.0.data_dir, stem);
            Arc::new(Entry::new(meta, stats, Slot::Cold, 0, 0, usage, revision))
        });
    }

    /// Replica tailer: the in-memory half of applying one tailed
    /// family — the files landed already (the hydrator's verified
    /// fetch); this re-reads the sidecar meta (pin flips, description
    /// and tuning edits arrive through the manifest like everything
    /// else), drops the loaded copy so the next read serves the new
    /// bytes through the ordinary load path (image plus watermark
    /// replay), and reloads immediately when pinned — pinned means
    /// resident, on a replica as anywhere.
    pub(crate) fn replica_refresh(&self, name: &str) {
        let Some(entry) = self.lookup(name) else {
            return;
        };
        let Some(mut inner) = entry.lock_unless_deleted() else {
            return;
        };
        let stem = file_stem(name);
        let MetaFile {
            meta,
            stats,
            usage: _,
            revision,
        } = read_meta_file(&self.0.data_dir, &stem);
        inner.meta = meta;
        inner.stats = stats;
        // Monotonic re-seed: the writer's sidecar lags its shipped WAL
        // by a flush interval, and this replica's own last load may
        // already have replayed past it — a tailed refresh must move
        // the counters forward, never walk them back.
        inner.graph_revision = inner.graph_revision.max(revision.graph);
        inner.config_revision = inner.config_revision.max(revision.config);
        entry
            .passage_revision
            .fetch_max(revision.passages, Ordering::Relaxed);
        // The monotonic re-seed is exactly why the retrieval cache
        // cannot key on revisions alone here: an upstream
        // delete+recreate arrives as this same in-place refresh, with
        // the counters pinned by the `max` while the content switches
        // lineage. A fresh identity makes every key minted against the
        // old bytes unreachable (see `EntryInner::cache_identity`).
        inner.cache_identity = next_cache_identity();
        inner.load_failure = None;
        if matches!(inner.slot, Slot::Hot(_)) {
            inner.slot = Slot::Cold;
            // The same bump eviction does: a flush that staged this
            // entry's image must see the slot it captured is gone.
            // Vacuous on a replica (nothing flushes) — kept so this
            // function cannot silently rot if that ever changes.
            inner.image_generation += 1;
        }
        self.recount_entry(&mut inner);
        // Derived residents follow the graph copy out — dropped, never
        // persisted: their sidecars are the manifest's business.
        *entry.passages.lock() = None;
        *entry.bm25.write() = None;
        entry.bm25_dirty.store(false, Ordering::Relaxed);
        *entry.passage_vectors.lock() = None;
        *entry.vectors.lock() = None;
        *entry.passages_load_failure.lock() = None;
        if inner.meta.pinned {
            if let Err(error) = ensure_hot(
                &self.0.data_dir,
                name,
                &mut inner,
                &self.0.metrics,
                self.0.hydrator.as_deref(),
            ) {
                tracing::warn!(context = %name, %error, "pinned context not reloaded after tailing");
            }
            self.recount_entry(&mut inner);
        }
    }

    /// Replica tailer: deregisters a context the lineage no longer
    /// carries. The in-memory teardown only — the files are the
    /// hydrator's business (its shared pass removes what the manifest
    /// does not know), and nothing here writes: no deletion marker, no
    /// group sweep (the manifest's own group files arrive already
    /// swept by the writer that deleted the context).
    pub(crate) fn replica_deregister(&self, name: &str) {
        let Some(entry) = self.0.registry.write().remove(name) else {
            return;
        };
        let mut inner = entry.inner.write();
        self.tombstone_locked(&mut inner, &entry);
    }

    /// Replica tailer: re-reads group records from disk after a shared
    /// refresh landed new group files. Read-only by design — no rename
    /// resumption, no corrupt set-aside, no reconcile persistence: the
    /// manifest is the author here, and anything odd heals when the
    /// next diff refetches the file. A record that does not parse
    /// keeps its previous in-memory version rather than vanishing
    /// mid-serve.
    pub(crate) fn replica_reload_groups(&self) {
        let entries = match fs::read_dir(&self.0.data_dir) {
            Ok(entries) => entries,
            Err(error) => {
                tracing::warn!(%error, "replica group reload could not list the data directory");
                return;
            }
        };
        let mut fresh: BTreeMap<String, groups::GroupRecord> = BTreeMap::new();
        let mut carried: Vec<String> = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("group") {
                continue;
            }
            let Some((_stem, name)) = scanned_stem_and_name(&path) else {
                continue;
            };
            let parsed = fs::read(&path).and_then(|bytes| {
                serde_json::from_slice::<groups::GroupRecord>(&bytes).map_err(io::Error::other)
            });
            match parsed {
                Ok(record) => {
                    fresh.insert(name, record);
                }
                Err(error) => {
                    tracing::warn!(group = %name, %error, "tailed group file unreadable; keeping its previous record");
                    carried.push(name);
                }
            }
        }
        let mut groups = self.0.groups.write();
        for name in carried {
            if let Some(previous) = groups.get(&name) {
                fresh.insert(name, previous.clone());
            }
        }
        groups::repair_nesting(&mut fresh);
        *groups = fresh;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::test_support::scratch_dir;

    /// The retrieval cache's guard against a replica lineage switch: a
    /// tailed refresh `max`es the revision counters (they must never
    /// walk backward), so an upstream delete+recreate can change a
    /// context's content while every lane reads unchanged — the fresh
    /// `cache_identity` is what makes keys minted against the old
    /// bytes unreachable. Recreate-on-a-writer needs no such hand:
    /// delete tears the entry down and create builds a new one, which
    /// mints its own identity.
    #[test]
    fn a_replica_refresh_mints_a_fresh_cache_identity_under_unmoved_revisions() {
        let dir = scratch_dir("revision-remint");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        let key = |state: &AppState| {
            state
                .retrieval_key(
                    crate::metrics::RetrievalCacheOp::Recall,
                    std::slice::from_ref(&"sake".to_string()),
                    Some("params".to_string()),
                )
                .expect("a live context keys")
        };
        let before = key(&state);
        state.replica_refresh("sake");
        let after = key(&state);
        assert_eq!(
            before.targets[0].lanes, after.targets[0].lanes,
            "the refresh moved no revision lane"
        );
        assert_ne!(
            before.targets[0].identity, after.targets[0].identity,
            "the identity is re-minted, so the old key can never hit again"
        );
        let _ = fs::remove_dir_all(dir);
    }
}
