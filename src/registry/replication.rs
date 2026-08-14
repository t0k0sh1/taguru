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
            // Not schema-verified here, same asymmetry as boot's
            // hydrator registration (`boot_with`): the family this
            // digest describes is not necessarily local yet, only the
            // meta is. `ensure_hot`'s own copy of ADR 0009 §5.2's check
            // runs once a load actually needs the bytes.
            Arc::new(Entry::cold_from_meta(
                read_meta_file(&self.0.data_dir, stem),
                0,
                0,
                None,
            ))
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
            schema_digest,
        } = read_meta_file(&self.0.data_dir, &stem);
        inner.meta = meta;
        inner.stats = stats;
        // Not `max`-merged like the revision counters below: a digest
        // is a content fingerprint, not a monotonic counter, so the
        // freshly re-read sidecar's value is simply the current truth
        // — same posture as `meta`/`stats` just above. `load_failure`
        // is cleared a few lines down regardless, so a stale digest
        // left over from a load that quarantined on the OLD value
        // cannot linger past this refresh either way.
        inner.schema_digest = schema_digest;
        // Dropped, not re-resolved inline: the digest may have just
        // changed and re-verifying against a schema file the shared
        // hydration pass has not necessarily re-fetched yet would risk
        // the exact stale-content-under-a-fresh-digest window ADR 0009
        // §5.2 exists to catch. `schema_of`/`ensure_hot`'s lazy
        // resolution re-checks the file against this fresh digest on
        // next read, same as any other cold entry.
        inner.schema = None;
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
        inner.invalidate_cache_identity();
        inner.load_failure = None;
        // Re-stat both WAL gauges: on a replica the bytes arrive as
        // tailed file copies, never through the writer's live
        // increments, and `ensure_hot`'s own re-stat runs only for
        // pinned entries below (or on the next local read) — without
        // this, a cold unpinned context the tailer keeps growing
        // understates `taguru_wal_bytes` indefinitely. `NotFound` is
        // the one honest zero (no WAL shipped for this lane yet); any
        // other stat failure keeps the last-known value rather than
        // walking a live gauge down to nothing on a transient error.
        let restat = |path: &std::path::Path, last: u64| match std::fs::metadata(path) {
            Ok(meta) => meta.len(),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
            Err(_) => last,
        };
        inner.wal_bytes = restat(&wal_path(&self.0.data_dir, &stem), inner.wal_bytes);
        inner.passages_wal_bytes = restat(
            &passages_wal_path(&self.0.data_dir, &stem),
            inner.passages_wal_bytes,
        );
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
    /// falls back to its previous in-memory version rather than
    /// vanishing mid-serve — except the first time this replica ever
    /// sees that name, when there IS no previous version: it simply
    /// stays absent until a later poll lands a parseable copy (see the
    /// `carried` loop's own comment).
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
                    // Whether a previous record exists is decided
                    // below, once every file is read — this line
                    // cannot yet promise "keeping its previous record"
                    // the way it used to unconditionally claim.
                    tracing::warn!(group = %name, %error, "tailed group file unreadable");
                    carried.push(name);
                }
            }
        }
        let mut groups = self.0.groups.write();
        for name in carried {
            match groups.get(&name) {
                Some(previous) => {
                    fresh.insert(name, previous.clone());
                }
                None => {
                    // No fallback exists for a name this replica has
                    // never held a good copy of — it stays absent from
                    // `fresh` (not inserted, not repaired-around) until
                    // the manifest's next diff lands a parseable file.
                    tracing::warn!(
                        group = %name,
                        "no previous record to fall back to for this unreadable group; \
                         it stays absent until a later poll lands a parseable copy"
                    );
                }
            }
        }
        // `repair_nesting` requires dangling child references dropped
        // first (its own doc comment) — the other two callers
        // (`boot::reconcile_groups`, `inspect`'s shape preview) both do
        // this before calling it; a name `carried` above but with no
        // fallback (just warned about, immediately above) is exactly
        // the kind of dangling reference this guards against, since
        // some OTHER record's `groups` set may still name it.
        let scanned = fresh.clone();
        for record in fresh.values_mut() {
            record.groups.retain(|child| scanned.contains_key(child));
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

    /// On a replica the WAL grows by tailed file copies, never through
    /// the writer's live byte accounting — the refresh must re-stat
    /// both WAL gauges itself, or a cold unpinned context the tailer
    /// keeps growing understates `taguru_wal_bytes` until some local
    /// read happens to run `ensure_hot`.
    #[test]
    fn a_replica_refresh_restats_both_wal_gauges() {
        let dir = scratch_dir("replica-refresh-wal-bytes");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        let stem = file_stem("sake");
        // The tailer's shape: bytes land as plain file writes.
        fs::write(wal_path(&dir, &stem), b"{\"tailed\":1}\n").unwrap();
        fs::write(passages_wal_path(&dir, &stem), b"{\"tailed\":2}\n").unwrap();
        state.replica_refresh("sake");
        let entry = state.lookup("sake").unwrap();
        {
            let inner = entry.inner.read();
            assert_eq!(
                inner.wal_bytes,
                fs::metadata(wal_path(&dir, &stem)).unwrap().len()
            );
            assert_eq!(
                inner.passages_wal_bytes,
                fs::metadata(passages_wal_path(&dir, &stem)).unwrap().len()
            );
        }

        // A vanished WAL is the one honest zero.
        fs::remove_file(wal_path(&dir, &stem)).unwrap();
        fs::remove_file(passages_wal_path(&dir, &stem)).unwrap();
        state.replica_refresh("sake");
        {
            let inner = entry.inner.read();
            assert_eq!(inner.wal_bytes, 0);
            assert_eq!(inner.passages_wal_bytes, 0);
        }

        // Any OTHER stat failure keeps the last-known value instead of
        // walking a live gauge down to nothing on a transient error.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::write(wal_path(&dir, &stem), b"{\"tailed\":3}\n").unwrap();
            fs::write(passages_wal_path(&dir, &stem), b"{\"tailed\":4}\n").unwrap();
            state.replica_refresh("sake");
            let before = {
                let inner = entry.inner.read();
                (inner.wal_bytes, inner.passages_wal_bytes)
            };
            assert_ne!(before, (0, 0));
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o000)).unwrap();
            state.replica_refresh("sake");
            fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
            let inner = entry.inner.read();
            assert_eq!((inner.wal_bytes, inner.passages_wal_bytes), before);
        }
        let _ = fs::remove_dir_all(dir);
    }

    /// A tailed refresh of a HOT entry drops the slot and bumps the
    /// image generation exactly like an eviction — a staged flush must
    /// see the slot it captured is gone (vacuous on today's replica,
    /// load-bearing the moment anything there ever flushes).
    #[test]
    fn a_replica_refresh_of_a_hot_entry_bumps_the_image_generation() {
        let dir = scratch_dir("replica-refresh-generation");
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state.create("sake", ContextMeta::default()).unwrap();
        state.flush_dirty();
        state
            .read_context("sake", |context| context.association_count())
            .map_err(|_| "read")
            .unwrap();
        let entry = state.lookup("sake").unwrap();
        let generation = entry.inner.read().image_generation;
        state.replica_refresh("sake");
        {
            let inner = entry.inner.read();
            assert!(
                !matches!(inner.slot, Slot::Hot(_)),
                "the refresh cools the slot"
            );
            assert_eq!(
                inner.image_generation,
                generation + 1,
                "one refresh of a hot entry, one generation"
            );
        }
        let _ = fs::remove_dir_all(dir);
    }

    /// A replica IS a replica, and its group reload actually reads the
    /// manifest-delivered .group files into the served map.
    #[test]
    fn a_replica_reload_serves_the_group_files_on_disk() {
        let dir = scratch_dir("replica-group-reload");
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            None,
            BootOptions {
                replica: Some(std::sync::Arc::new(crate::replica::ReplicaInfo::new(None))),
                ..BootOptions::default()
            },
        )
        .unwrap();
        assert!(state.is_replica(), "the replica boot must read as one");

        let record = groups::GroupRecord {
            description: "蔵まとめ".to_string(),
            contexts: std::collections::BTreeSet::from(["sake".to_string()]),
            groups: std::collections::BTreeSet::new(),
        };
        fs::write(
            dir.join(format!("{}.group", file_stem("kura"))),
            serde_json::to_vec(&record).unwrap(),
        )
        .unwrap();
        state.replica_reload_groups();
        let (_, groups) = state.group_page(None, usize::MAX);
        assert_eq!(groups.len(), 1, "{groups:?}");
        assert_eq!(groups[0].0, "kura");
        assert_eq!(groups[0].1.contexts.len(), 1);

        let _ = fs::remove_dir_all(dir);
    }

    /// #617 item 2: a group naming a child that no longer has a
    /// (parseable) file must not carry that dangling reference into
    /// `repair_nesting` — its own doc comment requires callers to drop
    /// dangling children first, and the other two callers
    /// (`boot::reconcile_groups`, `inspect`'s shape preview) both do.
    #[test]
    fn a_replica_reload_drops_a_dangling_child_reference_before_repair() {
        let dir = scratch_dir("replica-group-dangling-child");
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            None,
            BootOptions {
                replica: Some(std::sync::Arc::new(crate::replica::ReplicaInfo::new(None))),
                ..BootOptions::default()
            },
        )
        .unwrap();

        // `parent` names `child` as a nested group, but `child`'s own
        // file never lands (deleted upstream, or torn on this exact
        // poll) — the shape `repair_nesting`'s precondition exists
        // for.
        let parent = groups::GroupRecord {
            description: "parent".to_string(),
            contexts: std::collections::BTreeSet::new(),
            groups: std::collections::BTreeSet::from(["child".to_string()]),
        };
        fs::write(
            dir.join(format!("{}.group", file_stem("parent"))),
            serde_json::to_vec(&parent).unwrap(),
        )
        .unwrap();

        state.replica_reload_groups();
        let (_, groups) = state.group_page(None, usize::MAX);
        assert_eq!(groups.len(), 1, "{groups:?}");
        assert_eq!(groups[0].0, "parent");
        assert!(
            groups[0].1.groups.is_empty(),
            "the dangling child reference must be dropped, not carried into \
             repair_nesting: {groups:?}"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// #617 item 3: a group whose file fails to parse THE FIRST TIME
    /// this replica ever sees that name has no previous in-memory
    /// version to fall back to — it must stay absent rather than the
    /// fallback silently no-op-ing into an inconsistent state, and a
    /// SIBLING group that already existed with a good previous record
    /// must still be kept, unaffected.
    #[test]
    fn a_replica_reload_leaves_a_never_seen_unparseable_group_absent() {
        let dir = scratch_dir("replica-group-new-unparseable");
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            None,
            BootOptions {
                replica: Some(std::sync::Arc::new(crate::replica::ReplicaInfo::new(None))),
                ..BootOptions::default()
            },
        )
        .unwrap();

        // An existing, already-loaded group with a good record...
        let kura = groups::GroupRecord {
            description: "蔵まとめ".to_string(),
            contexts: std::collections::BTreeSet::new(),
            groups: std::collections::BTreeSet::new(),
        };
        fs::write(
            dir.join(format!("{}.group", file_stem("kura"))),
            serde_json::to_vec(&kura).unwrap(),
        )
        .unwrap();
        state.replica_reload_groups();
        assert_eq!(state.group_page(None, usize::MAX).1.len(), 1);

        // ...plus a BRAND NEW group whose file is unparseable from the
        // start — this replica has never held a good copy of it.
        fs::write(
            dir.join(format!("{}.group", file_stem("brandnew"))),
            b"{not json",
        )
        .unwrap();
        state.replica_reload_groups();

        let (_, groups) = state.group_page(None, usize::MAX);
        assert_eq!(
            groups.len(),
            1,
            "the never-seen unparseable group must stay absent: {groups:?}"
        );
        assert_eq!(
            groups[0].0, "kura",
            "the sibling with a good record survives"
        );

        let _ = fs::remove_dir_all(dir);
    }

    /// #618: `replica_register` was never directly exercised — only
    /// ever called from `Tailer::poll_once`. The hydrator's shared
    /// pass lands the meta before this runs, so registration is a
    /// pure in-memory step reading a file that already exists.
    #[test]
    fn a_replica_register_adds_a_new_context_from_its_landed_meta() {
        let dir = scratch_dir("replica-register-new");
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            None,
            BootOptions {
                replica: Some(std::sync::Arc::new(crate::replica::ReplicaInfo::new(None))),
                ..BootOptions::default()
            },
        )
        .unwrap();
        assert!(
            state.lookup("sake").is_none(),
            "nothing registered before the tailer touches it"
        );

        let stem = file_stem("sake");
        fs::write(
            dir.join(format!("{stem}.meta.json")),
            br#"{"description":"sake","pinned":false}"#,
        )
        .unwrap();
        state.replica_register(&stem);
        assert_eq!(
            state
                .directory_entry("sake")
                .expect("the stem must register")
                .description,
            "sake"
        );

        // Idempotent: a second registration of the same stem must not
        // replace the entry or error.
        state.replica_register(&stem);
        assert!(state.lookup("sake").is_some());

        let _ = fs::remove_dir_all(dir);
    }

    /// #618: an undecodable stem (never a name this server itself
    /// wrote) must be a silent no-op, not a panic — the tailer's own
    /// worklist loop already filters these via `name_from_stem`, but
    /// `replica_register` guards against it independently too.
    #[test]
    fn a_replica_register_ignores_an_undecodable_stem() {
        let dir = scratch_dir("replica-register-undecodable");
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            None,
            BootOptions {
                replica: Some(std::sync::Arc::new(crate::replica::ReplicaInfo::new(None))),
                ..BootOptions::default()
            },
        )
        .unwrap();
        state.replica_register("not a valid stem at all");
        assert_eq!(state.group_page(None, usize::MAX).1.len(), 0);
        let _ = fs::remove_dir_all(dir);
    }

    /// #618: `replica_deregister` was never directly exercised — only
    /// ever called from `Tailer::poll_once` for a vanished lineage
    /// member.
    #[test]
    fn a_replica_deregister_removes_a_registered_context() {
        let dir = scratch_dir("replica-deregister");
        let state = AppState::boot_with(
            dir.clone(),
            usize::MAX,
            None,
            BootOptions {
                replica: Some(std::sync::Arc::new(crate::replica::ReplicaInfo::new(None))),
                ..BootOptions::default()
            },
        )
        .unwrap();
        let stem = file_stem("sake");
        fs::write(
            dir.join(format!("{stem}.meta.json")),
            br#"{"description":"sake","pinned":false}"#,
        )
        .unwrap();
        state.replica_register(&stem);
        assert!(state.lookup("sake").is_some());

        state.replica_deregister("sake");
        assert!(
            state.lookup("sake").is_none(),
            "the lineage no longer carrying this context must drop it in memory"
        );

        // A name never registered: a no-op, not a panic.
        state.replica_deregister("never-registered");

        let _ = fs::remove_dir_all(dir);
    }
}
