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
        let (mut registry, resumed_context_renames) =
            scan_data_dir(&data_dir, options.hydrator.as_deref())?;
        // A lazy bucket boot: the manifest's contexts are real even
        // though their images are not local yet — register them cold
        // from the sidecar metas the shared hydration already landed,
        // so enumeration, description, and pinned preload see the
        // whole directory from the first request. The `.ctx` scan
        // above naturally found any family whose image IS local
        // (cache-mode reuse); `or_insert`-style entry keeps those.
        // Not schema-verified here: a lazy bucket boot registers these
        // stems from the sidecar meta ALONE — their families (schema
        // file included) are not local yet, so there is nothing on disk
        // to check a digest against. `ensure_hot`'s own copy of ADR
        // 0009 §5.2's check runs once hydration actually lands the
        // family, which is the earliest point verification is possible
        // for this path.
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
                        schema_digest,
                    } = read_meta_file(&data_dir, &stem);
                    Arc::new(Entry::new(
                        meta,
                        stats,
                        Slot::Cold,
                        0,
                        0,
                        usage,
                        revision,
                        schema_digest,
                        // Not schema-verified here (see the comment
                        // above): the family is not local yet, so
                        // there is nothing to resolve. `ensure_hot`
                        // populates this once hydration lands it.
                        None,
                    ))
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
            // `true` when there was nothing to rewrite (`!landed`): a
            // marker resumed that far is still cleared by `complete`
            // alone, same as before this field existed.
            let membership_persisted = if rename.landed {
                rename_in_membership(&data_dir, &mut groups, &rename.from, &rename.to, |record| {
                    &mut record.contexts
                })
            } else {
                true
            };
            // Removing the marker unconditionally on `complete` — the
            // move alone — is the bug `rename_in_membership`'s own doc
            // warns about; see `retire_rename_marker`'s doc for why
            // `complete` alone is not enough. Only reached once the
            // move is complete: a straggler still needs the next boot
            // to retry regardless of membership, so nothing runs here
            // at all otherwise.
            if rename.complete {
                retire_rename_marker(
                    &renaming_marker_path(&data_dir, &file_stem(&rename.from)),
                    membership_persisted,
                    &rename.from,
                    &rename.to,
                    "context rename's group membership rewrite",
                );
            }
        }
        for rename in &resumed_group_renames {
            let membership_persisted = if rename.landed {
                rename_in_membership(&data_dir, &mut groups, &rename.from, &rename.to, |record| {
                    &mut record.groups
                })
            } else {
                true
            };
            if rename.complete {
                retire_rename_marker(
                    &groups::group_renaming_marker_path(&data_dir, &file_stem(&rename.from)),
                    membership_persisted,
                    &rename.from,
                    &rename.to,
                    "group rename's nesting rewrite",
                );
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
        let rerank_breaker = options
            .reranker
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
            reranker: options.reranker,
            rerank_breaker,
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
            import_markers_enabled: options.import_markers_enabled,
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
            budget_saturated: AtomicBool::new(false),
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

/// Unlinks `path`, warning only if the removal failed for a reason
/// worth an operator's attention — `NotFound` just means nothing was
/// there to begin with, the unremarkable common case every one of the
/// `.deleted` resume sweep's removals expects on a healthy boot.
/// Shared by the sweep's file, own-marker, and targeting-marker
/// removals so the three cannot drift on wording or on which errors
/// they consider worth a log line.
///
/// Which branch runs has no bearing on program behavior — both attempt
/// the same unlink — so a mutated condition here can never be caught
/// by a behavioral test; the difference is purely how noisy the logs
/// are for the routine "no marker to begin with" case.
#[mutants::skip]
fn remove_persisted_file_quietly(path: &Path, what: &str) {
    if let Err(error) = remove_persisted_file(path)
        && error.kind() != io::ErrorKind::NotFound
    {
        // Never a bare `error` field — see the same note on
        // `rename_context`'s `Stuck` arm in `lifecycle.rs`.
        tracing::warn!(path = %path.display(), removal_error = %error, "{what}");
    }
}

/// One boot-time pass over the data directory: crash leftovers of
/// staged writes are deleted (never published, and nothing may linger
/// as unbounded disk litter), and every context image found is
/// registered cold, described by its sidecar snapshot.
fn scan_data_dir(
    data_dir: &Path,
    hydrator: Option<&crate::hydrate::Hydrator>,
) -> io::Result<(BTreeMap<String, Arc<Entry>>, ResumedRenames)> {
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
            // A lazy bucket boot: the bucket's copy of this family must
            // not re-materialize after the unlinks below — the same
            // veto `delete()` itself takes (`lifecycle.rs`'s own
            // `AppState::delete`) before its unlink loop. Without this,
            // `Hydrator::spawn_background_fill` could re-fetch a
            // `Pending` stem this acknowledged delete is still tearing
            // down, and the next boot would find a resurrected family
            // sitting behind a marker that has not yet been removed.
            if let Some(hydrator) = hydrator {
                hydrator.veto(&stem);
            }
            for file in context_files(&stem) {
                remove_persisted_file_quietly(
                    &data_dir.join(file),
                    "unfinished deletion: file still held",
                );
            }
            // A stuck rename naming this stem as its SOURCE or
            // DESTINATION goes with the family too — the boot-time
            // counterpart of `AppState::delete`'s own sweep (see that
            // function's doc for why a survivor would otherwise have
            // the rename-resume pass below try to move a family that
            // no longer exists).
            remove_persisted_file_quietly(
                &renaming_marker_path(data_dir, &stem),
                "unfinished deletion: stale rename marker still held",
            );
            // `rename_markers_targeting` matches a marker's `to` field
            // against the DECODED context name, not the file stem
            // (`RenameMarker` is written from the names `rename_context`
            // was called with, before `file_stem`'s percent-encoding) —
            // `delete`'s own equivalent scan (`lifecycle.rs`) passes
            // `name` for the same reason. A name that needed encoding
            // would otherwise never match here, leaving its stale
            // targeting marker behind for the next boot's resume to
            // move a deleted family back to life.
            if let Some(name) = name_from_stem(&stem) {
                for stale in rename_markers_targeting(data_dir, &name, "renaming") {
                    remove_persisted_file_quietly(
                        &stale,
                        "unfinished deletion: stale rename marker still held",
                    );
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
        |from_stem, to_stem| {
            // `evict_stem` hydrates the family before vetoing its
            // re-materialization — the same primitive the live path
            // uses in `AppState::rename_context_locked` before moving
            // files, so the two can never drift on the order or the
            // reasoning. The returned undo token is dropped: unlike the
            // live path, a resume that fails here just leaves the
            // marker for the next boot to retry, no rollback needed.
            //
            // Logged (like `preload_pinned`'s own timing line) because
            // this runs inside `scan_data_dir`, before the listener
            // binds and while the data-directory lock is held: a
            // permanently unreachable object turns every boot into a
            // multi-round fetch-retry wait, once per stale marker, and
            // an operator staring at a slow boot needs to see why.
            if let Some(hydrator) = hydrator {
                let hydrate_started = std::time::Instant::now();
                let result = hydrator.evict_stem(from_stem);
                tracing::info!(
                    from_stem,
                    ms = hydrate_started.elapsed().as_millis() as u64,
                    ok = result.is_ok(),
                    "boot resume: hydrated a renamed family before moving it"
                );
                result?;
            }
            move_context_files(data_dir, from_stem, to_stem)
        },
        // The pivot is `.ctx` — its arrival is what lets the `.ctx` scan
        // below register the context under `to`.
        |to_stem| image_path(data_dir, to_stem).exists(),
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
    // way `preload_pinned` does. Arrival order cannot affect a
    // SUCCESSFUL result (it only feeds a `BTreeMap`), but it can affect
    // a FAILURE: `parallel_map`'s own doc says results "come back in
    // arrival order, not input order," and the schema check below can
    // fail more than one candidate — the sort by `index` right after
    // (its own documented remedy: "callers that need input order carry
    // an index through T/R and sort afterward") is what makes the
    // FIRST reported error deterministic across repeated boots of the
    // same directory, rather than a coin flip decided by which
    // worker's thread happened to finish first.
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    type IndexedCandidate = (usize, io::Result<(String, Arc<Entry>)>);
    let mut indexed: Vec<IndexedCandidate> = parallel_map(
        candidates.into_iter().enumerate().collect(),
        workers,
        |(index, (stem, name))| {
            let stem = stem.as_str();
            let MetaFile {
                meta,
                stats,
                usage,
                revision,
                schema_digest,
            } = read_meta_file(data_dir, stem);
            // ADR 0009 §5.1/§5.2: an unreadable, malformed, invalid, or
            // digest-mismatched schema file refuses the WHOLE boot — a
            // schema is never allowed to fall back to "as if absent"
            // the way a corrupt `.meta.json` sidecar does just above,
            // since that fallback is indistinguishable from `mode:
            // off` and would silently disable `strict`. `Ok(None)`
            // (nothing recorded, nothing on disk — a schema-free
            // context) is the only outcome that lets this candidate
            // through; every other case is folded into the `?` below
            // and stops the boot for the whole directory, exactly as
            // an unreadable `.ctx` already does one candidate over.
            //
            // The two leniency postures interact: `read_meta_file`'s
            // own fallback (see its doc) drops `schema_digest` to
            // `None` for a corrupt sidecar, and a healthy
            // `{stem}.schema.json` then reads here as a digest
            // MISMATCH (recorded none, on disk something) rather than
            // as "the sidecar is the actual problem." The operator
            // symptom is a server that refuses to start at all over
            // what looks like a schema fault; the fix is to repair
            // `{stem}.meta.json`, not the schema file.
            let schema = match schema::load_schema(data_dir, stem, schema_digest.as_deref(), true) {
                Ok(schema) => schema.map(Arc::new),
                Err(error) => {
                    return (
                        index,
                        Err(io::Error::new(
                            error.kind(),
                            format!("context '{name}': {error}"),
                        )),
                    );
                }
            };
            // The gauge must see leftover logs from the first scrape,
            // not only after each context's first touch.
            let wal_bytes = fs::metadata(wal_path(data_dir, stem))
                .map(|meta| meta.len())
                .unwrap_or(0);
            let passages_wal_bytes = fs::metadata(passages_wal_path(data_dir, stem))
                .map(|meta| meta.len())
                .unwrap_or(0);
            (
                index,
                Ok((
                    name,
                    Arc::new(Entry::new(
                        meta,
                        stats,
                        Slot::Cold,
                        wal_bytes,
                        passages_wal_bytes,
                        usage,
                        revision,
                        schema_digest,
                        schema,
                    )),
                )),
            )
        },
    );
    indexed.sort_by_key(|(index, _)| *index);
    let registry: BTreeMap<String, Arc<Entry>> = indexed
        .into_iter()
        .map(|(_, result)| result)
        .collect::<io::Result<BTreeMap<String, Arc<Entry>>>>(
    )?;

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
    use crate::registry::paths::RenameMarker;
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

    /// The boot-time counterpart of `AppState::delete`'s own rename
    /// marker sweep: a `.deleted` marker means the ORIGINAL `delete`
    /// call died before finishing (this is what the resume above is
    /// for), so it never had the chance to clear a stuck rename's
    /// marker sitting at (or naming) this stem either. Boot's own
    /// resume must do it, or a stale marker survives an acknowledged
    /// deletion and has the NEXT boot try to move a family that no
    /// longer exists onto a destination stem.
    #[test]
    fn an_unfinished_deletion_clears_stray_rename_markers_too() {
        let dir = scratch_dir("deleted-sweep-rename-markers");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .create("beer", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
        }
        // The crash-shaped state: `sake`'s own delete died mid-unlink
        // (the marker survives), while it ALSO happens to sit at the
        // source stem of one stuck rename (naming "shochu" as `to`) and
        // the destination of another (from "beer"). Neither marker's
        // family is this test's concern — only that both are gone once
        // the deletion sweep finishes with "sake".
        fs::write(dir.join("sake.deleted"), b"").unwrap();
        fs::write(
            renaming_marker_path(&dir, &file_stem("sake")),
            serde_json::to_vec(&RenameMarker {
                from: "sake".to_string(),
                to: "shochu".to_string(),
            })
            .unwrap(),
        )
        .unwrap();
        fs::write(
            renaming_marker_path(&dir, &file_stem("beer")),
            serde_json::to_vec(&RenameMarker {
                from: "beer".to_string(),
                to: "sake".to_string(),
            })
            .unwrap(),
        )
        .unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.directory_entry("sake").is_none());
        assert!(
            !renaming_marker_path(&dir, &file_stem("sake")).exists(),
            "the deletion sweep must clear a stuck rename marker at its own stem"
        );
        assert!(
            !renaming_marker_path(&dir, &file_stem("beer")).exists(),
            "the deletion sweep must clear a stuck rename marker naming it as destination"
        );
        assert!(
            state.directory_entry("beer").is_some(),
            "the untouched, unrelated source context must survive"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// Regression: the destination-targeting scan above must compare
    /// against the DECODED context name, not the file stem —
    /// `RenameMarker.to` is written from the name `rename_context` was
    /// called with, before `file_stem`'s percent-encoding. A deleted
    /// context whose name needed encoding (anything outside
    /// `[A-Za-z0-9_-]`) previously never matched here, leaving its
    /// stale targeting marker behind for the next boot's resume to
    /// move a family onto a name that no longer exists.
    #[test]
    fn an_unfinished_deletion_clears_a_targeting_marker_whose_name_needed_encoding() {
        let dir = scratch_dir("deleted-sweep-encoded-targeting-marker");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake!", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state
                .create("beer", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
        }
        fs::write(dir.join(format!("{}.deleted", file_stem("sake!"))), b"").unwrap();
        fs::write(
            renaming_marker_path(&dir, &file_stem("beer")),
            serde_json::to_vec(&RenameMarker {
                from: "beer".to_string(),
                to: "sake!".to_string(),
            })
            .unwrap(),
        )
        .unwrap();

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.directory_entry("sake!").is_none());
        assert!(
            !renaming_marker_path(&dir, &file_stem("beer")).exists(),
            "the deletion sweep must clear a targeting marker even when \
             the deleted context's name needed percent-encoding"
        );
        assert!(state.directory_entry("beer").is_some());

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

    /// A context with no `{stem}.schema.json` and no recorded digest
    /// boots exactly as it did before #379 — the acceptance criterion
    /// ADR 0009 §7.1 states by construction.
    #[test]
    fn a_context_with_no_schema_boots_unchanged() {
        let dir = scratch_dir("schema-off-boots");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state.flush_dirty();
        }
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        assert!(state.directory_entry("sake").is_some());
        drop(state);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Hand-edits the meta sidecar's `schema_digest` directly — S1 ships
    /// no writer for it (`PUT /contexts/{name}/schema` is #380), so
    /// this is the only way a test can put one there; the boot refusal
    /// under test does not care how it arrived.
    fn record_schema_digest(dir: &Path, stem: &str, digest: &str) {
        let path = meta_path(dir, stem);
        let mut value: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        value["schema_digest"] = serde_json::json!(digest);
        fs::write(&path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    }

    /// ADR 0009 §5.1: a schema file that reads but does not parse
    /// refuses the WHOLE boot (not just this one context) and sets the
    /// mangled bytes aside — never a fresh-empty-record fallback, since
    /// that is indistinguishable from `mode: off` and would silently
    /// disable `strict`.
    #[test]
    fn a_corrupt_schema_file_refuses_the_boot_and_quarantines_the_bytes() {
        let dir = scratch_dir("schema-corrupt-boots");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state.flush_dirty();
        }
        let bytes = b"not json";
        fs::write(schema_path(&dir, "sake"), bytes).unwrap();
        record_schema_digest(&dir, "sake", &crate::sha256::sha256_hex(bytes));

        let error = AppState::boot(dir.clone(), usize::MAX, None)
            .map(|_| ())
            .unwrap_err();
        assert!(error.to_string().contains("does not parse"), "{error}");
        assert_eq!(fs::read(schema_corrupt_path(&dir, "sake")).unwrap(), bytes);

        let _ = fs::remove_dir_all(&dir);
    }

    /// ADR 0009 §5.2: the whole point of the recorded digest is to
    /// catch a crash between `write_meta`'s revision bump and the
    /// schema file's own separate `write_atomic` — simulated here by
    /// simply recording a digest that does not match the file's actual
    /// bytes. Refuses the boot rather than serving `strict` (or any
    /// mode) under content that does not match what the revision claims.
    #[test]
    fn a_schema_digest_mismatch_refuses_the_boot() {
        let dir = scratch_dir("schema-mismatch-boots");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state.flush_dirty();
        }
        let document =
            br#"{"schema":1,"mode":"off","closed_labels":false,"types":{},"relations":{}}"#;
        fs::write(schema_path(&dir, "sake"), document).unwrap();
        record_schema_digest(
            &dir,
            "sake",
            &crate::sha256::sha256_hex(b"a different document"),
        );

        let error = AppState::boot(dir.clone(), usize::MAX, None)
            .map(|_| ())
            .unwrap_err();
        assert!(error.to_string().contains("does not match"), "{error}");

        let _ = fs::remove_dir_all(&dir);
    }

    /// Regression for issue #561's item 7: a corrupt `.meta.json`
    /// sidecar alongside a perfectly healthy schema file is the SAME
    /// digest-mismatch refusal as the test above (`read_meta_file`'s
    /// own lenient fallback zeroes `schema_digest` to `None`), but the
    /// operator symptom and the actual fix are completely different —
    /// the message must say so instead of pointing at the schema file.
    #[test]
    fn a_corrupt_sidecar_alongside_a_live_schema_hints_at_the_real_cause() {
        let dir = scratch_dir("schema-mismatch-corrupt-sidecar");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state.flush_dirty();
        }
        let document =
            br#"{"schema":1,"mode":"off","closed_labels":false,"types":{},"relations":{}}"#;
        fs::write(schema_path(&dir, "sake"), document).unwrap();
        record_schema_digest(&dir, "sake", &crate::sha256::sha256_hex(document));
        // Corrupt the sidecar itself: `read_meta_file` falls back to
        // `MetaFile::default()`, dropping the digest it just recorded
        // — even though the schema file on disk never changed.
        fs::write(meta_path(&dir, "sake"), b"not json").unwrap();

        let error = AppState::boot(dir.clone(), usize::MAX, None)
            .map(|_| ())
            .unwrap_err();
        assert!(error.to_string().contains("does not match"), "{error}");
        assert!(
            error
                .to_string()
                .contains("sidecar itself is corrupt or missing"),
            "the message must point at the sidecar as the likely cause: {error}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// The other branch of the same hazard as the test above:
    /// `read_meta_file`'s `fs::read` failing (no sidecar at all) hits
    /// its lenient `Err(_) => MetaFile::default()` fallback exactly the
    /// same way a corrupt-but-present sidecar hits its `from_slice`
    /// fallback — same zeroed digest, same digest-mismatch refusal,
    /// same hint.
    #[test]
    fn a_missing_sidecar_alongside_a_live_schema_hints_at_the_real_cause() {
        let dir = scratch_dir("schema-mismatch-missing-sidecar");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state.flush_dirty();
        }
        let document =
            br#"{"schema":1,"mode":"off","closed_labels":false,"types":{},"relations":{}}"#;
        fs::write(schema_path(&dir, "sake"), document).unwrap();
        record_schema_digest(&dir, "sake", &crate::sha256::sha256_hex(document));
        // Remove the sidecar entirely, keeping the schema file: unlike
        // the corrupt-content case above, this exercises `fs::read`
        // itself failing rather than `serde_json::from_slice`.
        fs::remove_file(meta_path(&dir, "sake")).unwrap();

        let error = AppState::boot(dir.clone(), usize::MAX, None)
            .map(|_| ())
            .unwrap_err();
        assert!(error.to_string().contains("does not match"), "{error}");
        assert!(
            error
                .to_string()
                .contains("sidecar itself is corrupt or missing"),
            "the message must point at the sidecar as the likely cause: {error}"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// The other direction of §5.2's mismatch: a digest recorded with
    /// no corresponding file (the schema was deleted, or the write that
    /// should have landed it never reached disk).
    #[test]
    fn a_recorded_digest_with_no_schema_file_refuses_the_boot() {
        let dir = scratch_dir("schema-missing-boots");
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state.flush_dirty();
        }
        record_schema_digest(&dir, "sake", "deadbeef");

        let error = AppState::boot(dir.clone(), usize::MAX, None)
            .map(|_| ())
            .unwrap_err();
        assert!(error.to_string().contains("is missing"), "{error}");

        let _ = fs::remove_dir_all(&dir);
    }

    /// `EntryInner::schema_digest` must survive every `write_meta`
    /// call, not just the one at boot — otherwise the very next flush
    /// after a successful boot would silently drop it back to `None`,
    /// and the NEXT boot would then see a stray schema file with
    /// nothing recorded for it (a digest mismatch, per the module's own
    /// fail-closed contract). `update_meta` is a convenient write_meta
    /// trigger that does not require going through flush timing.
    #[test]
    fn schema_digest_survives_a_meta_update_write() {
        let dir = scratch_dir("schema-digest-survives");
        let document =
            br#"{"schema":1,"mode":"off","closed_labels":false,"types":{},"relations":{}}"#;
        let digest = crate::sha256::sha256_hex(document);
        {
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state
                .create("sake", ContextMeta::default())
                .map_err(|_| "create")
                .unwrap();
            state.flush_dirty();
        }
        fs::write(schema_path(&dir, "sake"), document).unwrap();
        record_schema_digest(&dir, "sake", &digest);

        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
        state
            .update_meta("sake", Some("updated".to_string()), None, None, None)
            .unwrap()
            .unwrap();
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(meta_path(&dir, "sake")).unwrap()).unwrap();
        assert_eq!(
            value["schema_digest"],
            serde_json::json!(digest),
            "a write must not silently drop the recorded schema digest"
        );
        drop(state);
        let _ = fs::remove_dir_all(&dir);
    }
}
