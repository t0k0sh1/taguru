use super::*;

impl AppState {
    /// [`AppState::boot_with`] with the WAL on and the default log cap
    /// — the deployment defaults, and what the tests boot with (so the
    /// whole existing suite exercises the WAL-enabled paths).
    #[cfg(test)]
    pub fn boot(
        data_dir: PathBuf,
        cache_bytes: usize,
        embedder: Option<Arc<dyn EmbeddingProvider>>,
    ) -> io::Result<Self> {
        Self::boot_with(data_dir, cache_bytes, embedder, BootOptions::default())
    }

    /// Opens (creating if needed) the data directory and registers every
    /// context image found in it — cold, described by their sidecar
    /// snapshots. Pinned contexts are loaded eagerly; a pinned image
    /// that fails to load is left cold with a warning rather than
    /// taking the server down. `wal_enabled: false` restores the
    /// flush-interval durability window (`TAGURU_WAL=0`);
    /// `wal_max_bytes` is the per-context log ceiling (0 = unlimited);
    /// `default_semantic_floor` recalibrates the semantic entry floor
    /// for the configured embedding model (`TAGURU_SEMANTIC_FLOOR`,
    /// `None` = the text-embedding-3-large calibration).
    pub fn boot_with(
        data_dir: PathBuf,
        cache_bytes: usize,
        embedder: Option<Arc<dyn EmbeddingProvider>>,
        options: BootOptions,
    ) -> io::Result<Self> {
        fs::create_dir_all(&data_dir)?;
        // Before reading anything: two live registries over one
        // directory (a second serve, or an import against a running
        // server) would each cache and flush independently — last
        // writer wins, silently.
        let dir_lock = lock_data_dir(&data_dir)?;
        let (mut registry, resumed_context_renames) = scan_data_dir(&data_dir)?;
        // A lazy bucket boot: the manifest's contexts are real even
        // though their images are not local yet — register them cold
        // from the sidecar metas the shared hydration already landed,
        // so enumeration, description, and pinned preload see the
        // whole directory from the first request. The `.ctx` scan
        // above naturally found any family whose image IS local
        // (cache-mode reuse); `or_insert`-style entry keeps those.
        if let Some(hydrator) = &options.hydrator {
            for stem in hydrator.context_stems() {
                let Some(name) = name_from_stem(&stem) else {
                    continue;
                };
                registry.entry(name).or_insert_with(|| {
                    let MetaFile {
                        meta,
                        stats,
                        usage,
                        revision,
                    } = read_meta_file(&data_dir, &stem);
                    Arc::new(Entry::new(meta, stats, Slot::Cold, 0, 0, usage, revision))
                });
            }
        }
        // Groups scan after contexts (the context scan also sweeps
        // staging leftovers). Both scans finish moving any in-flight
        // rename's files, and hand back the (from, to) pairs whose
        // marker survived; rewrite group membership for those FIRST —
        // before reconcile, which has no notion of a rename in flight
        // and would see `from` as a plain dangling reference (nothing
        // registered under that name any more) and drop it instead of
        // carrying it to `to`. Each rewrite persists immediately (it
        // cannot rely on reconcile's own before/after diff, which
        // would see no further change to make and skip the write), so
        // the marker is safe to remove right after.
        let (mut groups, resumed_group_renames) = groups::scan_groups(&data_dir)?;
        // Rewrite membership only once the destination's pivot has
        // landed (else there is no `to` to point at, and `from` still
        // holds the files); remove the marker only once the move is
        // complete (else a straggler still needs the next boot to
        // retry). See `ResumedRename` for why these must not be one
        // condition.
        for rename in &resumed_context_renames {
            if rename.landed {
                rename_in_membership(&data_dir, &mut groups, &rename.from, &rename.to, |record| {
                    &mut record.contexts
                });
            }
            if rename.complete {
                let _ = fs::remove_file(renaming_marker_path(&data_dir, &file_stem(&rename.from)));
            }
        }
        for rename in &resumed_group_renames {
            if rename.landed {
                rename_in_membership(&data_dir, &mut groups, &rename.from, &rename.to, |record| {
                    &mut record.groups
                });
            }
            if rename.complete {
                let _ = fs::remove_file(groups::group_renaming_marker_path(
                    &data_dir,
                    &file_stem(&rename.from),
                ));
            }
        }
        // Reconcile unconditionally: whatever put a dangling member, a
        // dangling child, or an illegal nesting into a group file — a
        // crash between a deletion and the sweep's rewrite, a sweep
        // that could not persist, a hand-edited directory — boot drops
        // it and writes the fix back, so "a group names only live
        // contexts and live groups, acyclically, within the depth cap"
        // holds from the first request on, without exception.
        reconcile_groups(&data_dir, &registry, &mut groups);

        // Legitimate for small corpora, but worth one line: under this
        // configuration every semantic sweep is the exact scan, and an
        // operator wondering why the ANN index never engages should
        // not have to read the source to learn the relationship.
        if options.embed_passages
            && options.passage_vector_limit < crate::embedding::PASSAGE_ANN_THRESHOLD
        {
            tracing::info!(
                limit = options.passage_vector_limit,
                threshold = crate::embedding::PASSAGE_ANN_THRESHOLD,
                "passage vector limit sits below the ANN activation threshold; passage search will always use the exact sweep"
            );
        }
        let embed_breaker = embedder
            .as_ref()
            .and_then(|provider| provider.breaker().cloned());
        let state = Self(Arc::new(StateInner {
            data_dir,
            _dir_lock: dir_lock,
            cache_bytes,
            registry: RwLock::new(registry),
            groups: RwLock::new(groups),
            clock: AtomicU64::new(0),
            embedder,
            embed_breaker,
            default_semantic_floor: options
                .default_semantic_floor
                .unwrap_or(DEFAULT_SEMANTIC_FLOOR)
                .clamp(0.0, 1.0),
            cue_cache: Mutex::new(CueCache::default()),
            retrieval_cache: Mutex::new(retrieval_cache::RetrievalCache::new(
                crate::env::env_number(
                    "TAGURU_RETRIEVAL_CACHE_BYTES",
                    retrieval_cache::DEFAULT_RETRIEVAL_CACHE_BYTES,
                ),
            )),
            semantic_cache: Mutex::new(semantic_cache::SemanticCache::new(crate::env::env_floor(
                "TAGURU_SEMANTIC_CACHE_THRESHOLD",
            ))),
            metrics: Metrics::default(),
            wal_enabled: options.wal_enabled,
            wal_max_bytes: options.wal_max_bytes,
            passages_wal_max_bytes: options.passages_wal_max_bytes,
            embed_passages: options.embed_passages,
            passage_vector_limit: options.passage_vector_limit,
            embed_parallel: options.embed_parallel,
            embed_provider_slots: Semaphore::new(options.embed_parallel),
            per_context_metrics: options.per_context_metrics,
            auto_compact: options.auto_compact,
            context_quotas: options.context_quotas,
            ship_progress: options.ship_progress,
            hydrator: options.hydrator,
            replica: options.replica,
            pending: Mutex::new(PendingNames::default()),
            resident_estimate: AtomicI64::new(0),
            budget_ops: AtomicU64::new(0),
        }));
        state.preload_pinned();
        // Seed the per-context disk snapshot (a no-op while nothing
        // reads it) so the first scrape is not blind until the first
        // flush tick — and so a declared storage ceiling counts a
        // restarted context's true size from the first write, not
        // from zeros.
        state.refresh_disk_usage();
        Ok(state)
    }

    /// Loads every pinned context now — in parallel, because this runs
    /// before the listener binds and its wall-clock IS the downtime a
    /// single-writer deploy pays (stop-then-start; see the README's
    /// rollout note), and chatty on purpose: a boot that spends
    /// seconds loading should say what it is loading, not sit silent
    /// until "server ready". Entries have independent locks, so the
    /// workers never contend with each other.
    fn preload_pinned(&self) {
        let pinned: Vec<(String, Arc<Entry>)> = self
            .snapshot()
            .into_iter()
            .filter(|(_, entry)| entry.inner.read().meta.pinned)
            .collect();
        if pinned.is_empty() {
            return;
        }
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(pinned.len());
        let queue = Mutex::new(pinned.into_iter());
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| {
                    loop {
                        let Some((name, entry)) = queue.lock().next() else {
                            break;
                        };
                        let mut inner = entry.inner.write();
                        if !inner.meta.pinned {
                            continue;
                        }
                        let preload_started = std::time::Instant::now();
                        match ensure_hot(
                            &self.0.data_dir,
                            &name,
                            &mut inner,
                            &self.0.metrics,
                            self.0.hydrator.as_deref(),
                        ) {
                            Ok(()) => tracing::info!(
                                context = %name,
                                ms = preload_started.elapsed().as_millis() as u64,
                                "preloaded pinned context"
                            ),
                            Err(error) => {
                                tracing::warn!("pinned context '{name}' not preloaded: {error}");
                            }
                        }
                    }
                });
            }
        });
    }
}

/// One boot-time pass over the data directory: crash leftovers of
/// staged writes are deleted (never published, and nothing may linger
/// as unbounded disk litter), and every context image found is
/// registered cold, described by its sidecar snapshot.
fn scan_data_dir(data_dir: &Path) -> io::Result<(BTreeMap<String, Arc<Entry>>, ResumedRenames)> {
    // Unfinished deletions first: a `.deleted` marker means delete()
    // acknowledged the removal but could not unlink the whole family —
    // without this sweep, a surviving `.ctx` would RESURRECT a context
    // the API already reported gone (and a surviving sidecar would
    // leak forever). Resuming the deletion here makes the marker the
    // durable half of the operation: acknowledged deletes stay deleted
    // across any crash or IO failure, eventually.
    for dir_entry in fs::read_dir(data_dir)? {
        let path = dir_entry?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("deleted")
            && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
        {
            tracing::warn!(stem, "resuming an unfinished context deletion");
            let stem = stem.to_string();
            for file in context_files(&stem) {
                let target = data_dir.join(file);
                if let Err(error) = remove_persisted_file(&target)
                    && error.kind() != io::ErrorKind::NotFound
                {
                    tracing::warn!(path = %target.display(), %error, "unfinished deletion: file still held");
                }
            }
            // The marker goes last: it only leaves once the family did.
            if remove_persisted_file(&path).is_err() {
                tracing::warn!(path = %path.display(), "unfinished deletion: marker still held");
            }
        }
    }
    // Unfinished renames next, before the `.ctx` scan below: a
    // `.renaming` marker means `rename_context` moved (or was about to
    // move) the whole file family but crashed before the group
    // membership rewrite landed. Finishing the move here — repeatable,
    // since a missing source file just means it already moved — lets
    // the `.ctx` scan discover the context under its NEW name. The
    // marker itself survives this pass; `boot_with` removes it only
    // after also rewriting group membership, so a second crash still
    // has everything it needs to resume.
    let resumed_renames = resume_rename_markers(
        data_dir,
        "renaming",
        "context",
        |from_stem, to_stem| move_context_files(data_dir, from_stem, to_stem),
        // The pivot is `.ctx` — its arrival is what lets the `.ctx` scan
        // below register the context under `to`.
        |to_stem| data_dir.join(format!("{to_stem}.ctx")).exists(),
    )?;
    let mut candidates: Vec<(String, String)> = Vec::new();
    let mut import_markers: Vec<PathBuf> = Vec::new();
    for dir_entry in fs::read_dir(data_dir)? {
        let path = dir_entry?.path();
        let extension = path.extension().and_then(|e| e.to_str());
        if extension.is_some_and(|e| e.starts_with("tmp")) {
            let _ = remove_persisted_file(&path);
            continue;
        }
        // Import markers are judged after the scan, once it is known
        // which contexts exist — collect them on the way through.
        if extension == Some(IMPORT_MARKER_EXTENSION) {
            import_markers.push(path);
            continue;
        }
        if extension != Some("ctx") {
            continue;
        }
        let Some((stem, name)) = scanned_stem_and_name(&path) else {
            continue;
        };
        candidates.push((stem, name));
    }

    // The expensive part of a boot scan is the disk I/O per candidate
    // (sidecar read plus two `fs::metadata` calls), and each candidate
    // is independent — `parallel_map` pays for it in parallel the same
    // way `preload_pinned` does; arrival order cannot affect the result
    // since it only feeds a `BTreeMap`.
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let registry: BTreeMap<String, Arc<Entry>> =
        parallel_map(candidates, workers, |(stem, name)| {
            let stem = stem.as_str();
            let MetaFile {
                meta,
                stats,
                usage,
                revision,
            } = read_meta_file(data_dir, stem);
            // The gauge must see leftover logs from the first scrape,
            // not only after each context's first touch.
            let wal_bytes = fs::metadata(wal_path(data_dir, stem))
                .map(|meta| meta.len())
                .unwrap_or(0);
            let passages_wal_bytes = fs::metadata(passages_wal_path(data_dir, stem))
                .map(|meta| meta.len())
                .unwrap_or(0);
            (
                name,
                Arc::new(Entry::new(
                    meta,
                    stats,
                    Slot::Cold,
                    wal_bytes,
                    passages_wal_bytes,
                    usage,
                    revision,
                )),
            )
        })
        .into_iter()
        .collect();

    // Surviving import markers: each says a multi-store batch opened
    // and never finished — a crash (or an unretried refusal) between
    // retract_source, store_passages, add_associations, and
    // add_aliases. Every store is individually consistent, so this
    // marker is the ONLY thing that can say the source's truth is
    // half-applied. Report the live ones every boot until a re-import
    // or a retraction clears them; a marker whose context no longer
    // exists is moot (deletion destroys the batch's target) and is
    // removed here, completing delete()'s own best-effort sweep.
    for path in import_markers {
        let parsed = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<ImportMarker>(&bytes).ok());
        let Some(marker) = parsed else {
            tracing::warn!(
                path = %path.display(),
                "unreadable import marker — an import batch may be half-applied, \
                 but which source is unrecoverable; remove the file once investigated",
            );
            continue;
        };
        if registry.contains_key(&marker.context) {
            tracing::warn!(
                context = %marker.context,
                source = %marker.source,
                "an import batch for this source never completed — its truth may be \
                 half-applied (passages without associations, or associations without \
                 aliases); re-import the batch file or retract the source",
            );
        } else {
            let _ = remove_persisted_file(&path);
        }
    }
    Ok((registry, resumed_renames))
}

/// Boot-time counterpart of the delete-path sweeps: drops every group
/// member that is not a registered context, every child that is not a
/// scanned group, every name past the [`groups::MAX_GROUP_MEMBERS`]
/// per-set cap, and every nesting edge that would close a cycle or
/// stack more than [`groups::MAX_GROUP_DEPTH`] groups (hand-edits
/// only — nothing running can persist such a shape). Each fix is
/// written back to disk immediately — disk is the source of truth, and
/// a fix that only lived in memory would leave the file lying to
/// `taguru inspect` and to file-level backups until the next unrelated
/// write. Runs unconditionally: the causes it heals (a crash between a
/// deletion and the sweep's rewrite, a sweep that could not persist, a
/// hand-edited data directory) leave no marker behind, and the whole
/// collection is small enough that checking it all costs nothing.
fn reconcile_groups(
    data_dir: &Path,
    registry: &BTreeMap<String, Arc<Entry>>,
    groups: &mut BTreeMap<String, GroupRecord>,
) {
    let scanned = groups.clone();
    for record in groups.values_mut() {
        record
            .contexts
            .retain(|context| registry.contains_key(context));
        record.groups.retain(|child| scanned.contains_key(child));
    }
    // Dangling names never count toward the cap — they were just
    // dropped — so the trim runs on what actually remains…
    groups::trim_membership(groups, groups::MAX_GROUP_MEMBERS);
    // …and what remains can still be the wrong SHAPE — the repair
    // drops exactly the edges the validator refuses, deterministically.
    groups::repair_nesting(groups);
    for (name, record) in groups.iter() {
        let before = &scanned[name];
        if before == record {
            continue;
        }
        match groups::write_group(data_dir, &file_stem(name), record) {
            Ok(()) => {
                tracing::info!(
                    group = %name,
                    dropped_contexts = before.contexts.len() - record.contexts.len(),
                    dropped_children = before.groups.len() - record.groups.len(),
                    "dropped dangling, over-cap, or ill-nested group reference(s) at boot"
                );
            }
            Err(error) => {
                tracing::warn!(
                    group = %name,
                    %error,
                    "boot reconciliation not persisted; memory is correct, the file heals on the next successful group write"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::test_support::{assoc_op, loaded_map, scratch_dir};

    #[test]
    fn the_data_directory_admits_one_registry_at_a_time() {
        let dir = scratch_dir("dir-lock");
        let holder = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        // flock-style locks are per descriptor, so a second registry in
        // the SAME process is refused exactly as a second process would
        // be — which is what lets one test prove the contract.
        let error = AppState::boot(dir.clone(), usize::MAX, None)
            .map(|_| ())
            .unwrap_err();
        assert!(
            error.to_string().contains("another taguru process"),
            "{error}"
        );
        // The lock dies with its holder; the directory is reusable.
        drop(holder);
        let _reborn = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let _ = fs::remove_dir_all(&dir);
    }

    /// A `.deleted` marker is the durable half of a delete: boot
    /// resumes the unlinks it finds one for, so an acknowledged
    /// deletion can never resurrect — however the unlink loop failed.
    #[test]
    fn an_unfinished_deletion_is_resumed_at_boot() {
        let dir = scratch_dir("deleted-sweep");
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
        assert!(dir.join("sake.ctx").exists());
        // The crash-shaped state: delete() wrote its marker, then the
        // process died before (or while) the unlinks ran.
        fs::write(dir.join("sake.deleted"), b"").unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(
            state.directory_entry("sake").is_none(),
            "an acknowledged deletion must not resurrect"
        );
        assert!(!dir.join("sake.ctx").exists(), "the family must be gone");
        assert!(!dir.join("sake.wal.jsonl").exists());
        assert!(
            !dir.join("sake.deleted").exists(),
            "the marker leaves once the family did"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// Boot's marker pass: a surviving marker whose context exists is
    /// the torn-import report (and stays on disk for the next boot to
    /// repeat, until re-import or retraction); one whose context is
    /// gone is moot and is removed — it completes delete()'s own
    /// best-effort sweep.
    #[test]
    fn boot_keeps_a_live_torn_import_marker_and_removes_a_moot_one() {
        let dir = scratch_dir("import-marker-boot");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            // The crash-shaped state: batches opened their markers and
            // the process died between the four mutations.
            state.open_import_marker("sake", "doc-1").unwrap();
            state.open_import_marker("ghost", "doc-9").unwrap();
            state.flush_dirty();
        }

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(
            import_marker_path(&dir, "sake", "doc-1").exists(),
            "a live context's tear stays visible until the repair runs"
        );
        assert!(
            !import_marker_path(&dir, "ghost", "doc-9").exists(),
            "a marker without its context is moot; boot removes it"
        );
        drop(state);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn pinned_contexts_are_never_evicted_and_preload_on_boot() {
        let dir = scratch_dir("pin");
        {
            let state = AppState::boot(dir.clone(), 1, None).unwrap();
            let pinned = ContextMeta {
                description: "glossary".into(),
                pinned: true,
                ..ContextMeta::default()
            };
            state
                .create("glossary", pinned)
                .map_err(|_| "create")
                .unwrap();
            state
                .create("other", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .write_context("glossary", |context| {
                    context.associate("用語", "意味", "定義", 1.0).unwrap();
                })
                .map_err(|_| "write")
                .unwrap();

            // Churning through the other context must not push the
            // pinned one out.
            state
                .read_context("other", |context| context.association_count())
                .map_err(|_| "read")
                .unwrap();
            assert!(loaded_map(&state)["glossary"]);
            state.flush_dirty();
        }

        // A fresh boot preloads pinned contexts and leaves the rest cold.
        let state = AppState::boot(dir.clone(), 1, None).unwrap();
        let loaded = loaded_map(&state);
        assert!(loaded["glossary"], "pinned must preload");
        assert!(!loaded["other"], "unpinned must boot cold");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn scan_data_dir_discovers_every_context_and_sorts_them_by_name() {
        let dir = scratch_dir("scan-parallel");
        let names = ["delta", "alpha", "charlie", "bravo", "echo", "foxtrot"];
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            for name in names {
                state
                    .create(name, ContextMeta::default())
                    .map_err(|_| "create")
                    .unwrap();
            }
        }

        // A fresh boot re-runs `scan_data_dir`'s worker-pool scan; the
        // registry it returns must still hold every context, keyed and
        // ordered by name regardless of which worker raced to finish
        // its disk reads first.
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        let found: Vec<String> = state.directory().into_iter().map(|e| e.name).collect();
        let mut expected: Vec<String> = names.iter().map(|s| s.to_string()).collect();
        expected.sort();
        assert_eq!(found, expected);

        let _ = fs::remove_dir_all(dir);
    }
}
