//! The `{stem}.meta.json` sidecar and the file-family operations built
//! beside it: the sidecar's on-disk shape (`MetaFile`), the
//! save/read pair every flush and boot goes through, and the
//! whole-family list/move helpers the delete loop, the boot sweep,
//! and rename recovery all share.

use super::*;

/// What `{name}.meta.json` holds: the meta inline plus the stats
/// snapshot as of the last save, so a directory listing can describe a
/// cold context without touching its image. `usage` rides along under
/// `#[serde(default)]`, so sidecars from before it existed load with
/// zeroed counters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub(super) struct MetaFile {
    #[serde(flatten)]
    pub(super) meta: ContextMeta,
    pub(super) stats: ContextStats,
    pub(super) usage: ContextUsage,
    /// The revision counters as of this save — what a cold entry (and
    /// a replica's tailed refresh) seeds from. Defaulted for sidecars
    /// from before the field existed, and for sidecars that simply do
    /// not exist yet (a fresh context): those report zeros until their
    /// first load or flush catches them up. A sidecar that DOES exist
    /// but cannot be read or parsed is a different case — see
    /// [`MetaFile::degraded`].
    pub(super) revision: ContextRevision,
    /// `sha256_hex` of `{stem}.schema.json`'s bytes as of this save,
    /// `None` for a schema-free context — ADR 0009 §5.2's boot-time
    /// consistency check. Written in the SAME `write_meta` call that
    /// bumps `config_revision` (never separately), so a crash between
    /// this field landing and the schema file's own `write_atomic`
    /// leaves a detectable disagreement rather than a silently stale
    /// enforcement: [`crate::schema::load_schema`] refuses to load
    /// whenever the file on disk and this recorded value disagree, in
    /// either direction. Defaulted for sidecars from before the field
    /// existed, exactly like `revision` above — a pre-#379 context has
    /// no schema file either, so `None` is also the correct fact, not
    /// just a safe default.
    pub(super) schema_digest: Option<String>,
}

impl MetaFile {
    /// [`MetaFile::default`], except `revision.config` is seeded from
    /// the wall clock instead of left at zero — [`read_meta_file`]'s
    /// fallback for a sidecar that exists but could not be read or
    /// parsed (as opposed to one that simply is not there yet).
    ///
    /// `graph`/`revision.graph` and `revision.passages` stay zero on
    /// purpose: both lanes have an independent source of truth that
    /// floors them back up on the next load (`ensure_hot`'s WAL replay
    /// top, `entry_passages`'s store watermark), so seeding them here
    /// would only fight that floor. `config` has no such source — a
    /// config change leaves no log, and the content it would have
    /// restored from is the very sidecar that just failed to read — so
    /// a time-based seed is the only way to guarantee this boot's value
    /// never collides with one a fingerprint consumer (`group_fingerprint`,
    /// `src/api/groups.rs`) saw served under the same counter before the
    /// corruption. Monotonic re-seeds elsewhere (`replica_refresh`'s
    /// `max()`) only ever pull a replica's counter forward from this
    /// value, never back down to a primary's smaller one, so the skew
    /// this introduces costs at most a spurious cache miss, never a
    /// stale hit.
    fn degraded() -> Self {
        Self {
            revision: ContextRevision {
                config: crate::clock::now_unix_secs(),
                ..ContextRevision::default()
            },
            ..Self::default()
        }
    }
}

#[allow(clippy::too_many_arguments)] // every whole-family save call site, not an API
pub(super) fn save_files(
    dir: &Path,
    name: &str,
    meta: &ContextMeta,
    stats: &ContextStats,
    usage: &ContextUsage,
    revision: ContextRevision,
    schema_digest: Option<&str>,
    context: &Context,
) -> io::Result<()> {
    let stem = file_stem(name);
    // The image is what `scan_data_dir` keys a context's existence on, so
    // it lands LAST: each `write_atomic` fully commits (fsync + rename +
    // parent-dir fsync) before returning, so by the time the `.ctx` is
    // durably in the directory its `.meta.json` companion already is too.
    // A crash between the two therefore leaves at worst an orphan sidecar
    // with no image — invisible to the scan and overwritten by the next
    // same-name create — never a durable image with a defaulted sidecar,
    // which would resurrect a context `create` told the client had failed.
    // (Image-then-meta would do exactly that; see `create`'s doc.)
    write_meta(dir, &stem, meta, stats, usage, revision, schema_digest)?;
    write_atomic(&image_path(dir, &stem), &context.to_bytes())
}

pub(super) fn write_meta(
    dir: &Path,
    stem: &str,
    meta: &ContextMeta,
    stats: &ContextStats,
    usage: &ContextUsage,
    revision: ContextRevision,
    schema_digest: Option<&str>,
) -> io::Result<()> {
    let file = MetaFile {
        meta: meta.clone(),
        stats: stats.clone(),
        usage: usage.clone(),
        revision,
        schema_digest: schema_digest.map(str::to_string),
    };
    write_atomic(&meta_path(dir, stem), &serde_json::to_vec_pretty(&file)?)
}

/// Reads the sidecar, falling back to defaults on any problem — a
/// missing or corrupt sidecar must not make the image unreachable.
///
/// The fallback distinguishes two very different problems. A sidecar
/// that simply is not there (`ErrorKind::NotFound` — a fresh context,
/// or one from before the file existed) is the ordinary, silent case:
/// [`MetaFile::default`], no log line. A sidecar that IS there but
/// could not be read (`EACCES`, `EIO`, a directory where a file should
/// be, ...) or could not be parsed is a real degradation with no other
/// diagnostic — logged at `warn`, exactly like the parse-failure arm
/// below always has been, and seeded via [`MetaFile::degraded`] rather
/// than a plain default so the revision counters it hands back can
/// never collide with a value some past, healthy save of this same
/// sidecar already handed out (see `degraded`'s doc).
///
/// That leniency has one more sharp edge: the fallback also zeroes
/// `schema_digest` to `None`, which for a context that DOES have a
/// live `{stem}.schema.json` collides with `schema::load_schema`'s own
/// fail-closed posture (ADR 0009 §5.1/§5.2, issue #561's audit) — a
/// corrupt sidecar plus a healthy schema file turns into a
/// digest-mismatch refusal that stops the WHOLE boot, not just this
/// one candidate, and the resulting message names a mismatch rather
/// than the sidecar that caused it. The fix for that case is the
/// sidecar's, not the schema check's: restore `{stem}.meta.json` (or
/// delete it if the context has no schema) so its recorded digest
/// agrees with the file on disk again.
pub(super) fn read_meta_file(dir: &Path, stem: &str) -> MetaFile {
    match fs::read(meta_path(dir, stem)) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_else(|error| {
            tracing::warn!("ignoring corrupt sidecar for '{stem}': {error}");
            MetaFile::degraded()
        }),
        Err(error) if error.kind() == io::ErrorKind::NotFound => MetaFile::default(),
        Err(error) => {
            tracing::warn!("sidecar for '{stem}' unreadable, falling back to defaults: {error}");
            MetaFile::degraded()
        }
    }
}

/// The recorded schema digest alone, for a caller (`taguru inspect`)
/// that has no use for the rest of the sidecar and must not import
/// `MetaFile` (private to this module). Same lenient fallback as
/// [`read_meta_file`]: an unreadable or corrupt sidecar reports `None`
/// here exactly as it would seed a fresh [`MetaFile::default`] at boot,
/// so inspect's schema check judges a context by the same recorded
/// value boot itself would.
pub(crate) fn schema_digest_of(dir: &Path, stem: &str) -> Option<String> {
    read_meta_file(dir, stem).schema_digest
}

/// One context's whole file family, by stem — the delete loop and the
/// boot-time deletion sweep must never disagree about what "the whole
/// family" means, so both read this one list.
///
/// A curated list, not a mechanical projection of `paths.rs`: that
/// module has 14 single-path builders, and the four not listed here
/// (`schema_corrupt_path`, `deleted_marker_path`, `renaming_marker_path`,
/// `import_marker_path`) are markers and quarantine files, deliberately
/// NOT family members. Adding an eleventh FAMILY file kind therefore
/// requires editing this array too — the test pinning its exact
/// extension set (`core_tests.rs`) fails loudly on a rename or
/// removal, but a new kind landing here is on the author, not a
/// compiler or test that catches it automatically.
pub(crate) fn context_files(stem: &str) -> [String; 10] {
    let unrooted = Path::new("");
    [
        image_path(unrooted, stem),
        meta_path(unrooted, stem),
        sources_path(unrooted, stem),
        passages_path(unrooted, stem),
        passages_wal_path(unrooted, stem),
        pvectors_path(unrooted, stem),
        bm25_path(unrooted, stem),
        vectors_path(unrooted, stem),
        wal_path(unrooted, stem),
        // Last on purpose: a missing or lagging schema file must never
        // block the pivot rename below, so it sits where a straggler is
        // already tolerated as best-effort (ADR 0009 §5.1).
        schema_path(unrooted, stem),
    ]
    .map(|path| path.to_string_lossy().into_owned())
}

/// Moves one context's whole file family from `from_stem` to
/// `to_stem`, file by file, in the fixed order [`context_files`]
/// defines — a missing source is skipped (an earlier, interrupted
/// attempt already moved it; safe to retry at boot or from a fresh
/// call). `.ctx` is index 0 and the pivot the boot scan registers a
/// context by: if IT will not move, nothing else does either (the
/// family stays wholly under `from_stem`, cleanly retried), and the
/// call fails before touching a sidecar. Once the pivot has moved, a
/// sidecar that still sticks is best-effort — the rest are moved anyway
/// so the retry has fewer orphans to chase — but the first such error
/// is returned so the caller knows the move is incomplete and keeps the
/// rename marker. All ten share `data_dir` as their parent, so one
/// fsync after every rename covers the whole family durably instead of
/// paying for it (via `commit_staged`) up to ten times. Each rename
/// itself still goes through the shared [`rename_persisted_file`] choke
/// point — this crate's highest-risk multi-step FS operation is fault-
/// injectable exactly like a single-file publish is. The fsync's own
/// failure is reported too, but only when there was no earlier
/// straggler to report first — a rename error names the file that
/// actually didn't move, which is more actionable than a directory
/// fsync failure that names nothing.
pub(super) fn move_context_files(
    data_dir: &Path,
    from_stem: &str,
    to_stem: &str,
) -> io::Result<()> {
    let mut moved_any = false;
    let mut first_error: Option<io::Error> = None;
    for (position, (from_file, to_file)) in context_files(from_stem)
        .into_iter()
        .zip(context_files(to_stem))
        .enumerate()
    {
        match rename_persisted_file(data_dir.join(from_file), data_dir.join(to_file)) {
            Ok(()) => moved_any = true,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            // The pivot: fail outright so nothing else moves.
            Err(error) if position == 0 => return Err(error),
            // A post-pivot straggler: keep going, remember the first.
            Err(error) => {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
    }
    let fsync_result = if moved_any {
        fsync_dir(data_dir)
    } else {
        Ok(())
    };
    match first_error {
        Some(error) => Err(error),
        None => fsync_result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::test_support::scratch_dir;

    /// The ordinary, silent case: no sidecar has ever existed for this
    /// stem (a brand new context, or one from before the file existed).
    /// `revision` stays all-zero — nothing here has degraded, so there
    /// is nothing to shield a fingerprint consumer from.
    #[test]
    fn a_missing_sidecar_reports_a_plain_zeroed_default() {
        let dir = scratch_dir("meta-io-missing");
        fs::create_dir_all(&dir).unwrap();
        let meta = read_meta_file(&dir, "sake");
        assert_eq!(meta.revision, ContextRevision::default());
        assert_eq!(meta.schema_digest, None);
        let _ = fs::remove_dir_all(&dir);
    }

    /// A sidecar that IS there but is not valid JSON (a torn write, a
    /// hand edit gone wrong) is the degraded case: `revision.config` is
    /// seeded from the wall clock instead of left at zero, so this
    /// boot's fingerprint can never collide with one an earlier, healthy
    /// save of the very same sidecar already handed a consumer like
    /// `group_fingerprint` (#585).
    #[test]
    fn a_corrupt_sidecar_seeds_config_revision_from_the_clock_instead_of_zero() {
        let dir = scratch_dir("meta-io-corrupt");
        fs::create_dir_all(&dir).unwrap();
        fs::write(meta_path(&dir, "sake"), b"not json").unwrap();

        let before = crate::clock::now_unix_secs();
        let meta = read_meta_file(&dir, "sake");
        let after = crate::clock::now_unix_secs();

        assert_eq!(meta.revision.graph, 0, "graph is floored by WAL replay");
        assert_eq!(
            meta.revision.passages, 0,
            "passages is floored by the store watermark"
        );
        assert!(
            (before..=after).contains(&meta.revision.config),
            "config must be a fresh clock reading, not zero: {meta:?}"
        );
        assert_eq!(meta.schema_digest, None);
        let _ = fs::remove_dir_all(&dir);
    }

    /// The other degraded case: the sidecar exists but `fs::read`
    /// itself fails with something other than `NotFound` — simulated
    /// here by putting a directory where the sidecar file should be, so
    /// the read fails without ever reaching `serde_json`. Same fallback
    /// as the corrupt-content case, exercised through the other branch.
    #[test]
    fn an_unreadable_sidecar_seeds_config_revision_from_the_clock_instead_of_zero() {
        let dir = scratch_dir("meta-io-unreadable");
        fs::create_dir_all(meta_path(&dir, "sake")).unwrap();

        let before = crate::clock::now_unix_secs();
        let meta = read_meta_file(&dir, "sake");
        let after = crate::clock::now_unix_secs();

        assert_eq!(meta.revision.graph, 0);
        assert_eq!(meta.revision.passages, 0);
        assert!(
            (before..=after).contains(&meta.revision.config),
            "config must be a fresh clock reading, not zero: {meta:?}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A healthy sidecar's own recorded revision is never overridden —
    /// the clock seed only ever applies to the degraded fallback.
    #[test]
    fn a_healthy_sidecar_reports_its_own_recorded_revision_unchanged() {
        let dir = scratch_dir("meta-io-healthy");
        fs::create_dir_all(&dir).unwrap();
        write_meta(
            &dir,
            "sake",
            &ContextMeta::default(),
            &ContextStats::default(),
            &ContextUsage::default(),
            ContextRevision {
                graph: 3,
                passages: 2,
                config: 1,
            },
            None,
        )
        .unwrap();

        let meta = read_meta_file(&dir, "sake");
        assert_eq!(
            meta.revision,
            ContextRevision {
                graph: 3,
                passages: 2,
                config: 1
            }
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// #586: `move_context_files` now routes every one of its ten
    /// renames through the same [`crate::storage::rename_persisted_file`]
    /// choke point every other publish uses — the fault injector must
    /// actually reach the pivot move, the highest-risk multi-step FS
    /// operation in the crate. A failure there must behave exactly like
    /// a real `fs::rename` failure on the pivot always has: nothing
    /// else in the family moves.
    #[test]
    fn an_injected_pivot_rename_failure_moves_nothing() {
        let dir = scratch_dir("meta-io-move-pivot-fault");
        fs::create_dir_all(&dir).unwrap();
        let from_stem = file_stem("sake");
        let to_stem = file_stem("shochu");
        fs::write(image_path(&dir, &from_stem), b"ctx bytes").unwrap();
        fs::write(meta_path(&dir, &from_stem), b"meta bytes").unwrap();

        fail_persistence_ops_after(0);
        let error = move_context_files(&dir, &from_stem, &to_stem).unwrap_err();
        let past_end = clear_persistence_fault();
        assert!(
            !past_end,
            "the very first rename (the pivot) must be what failed"
        );
        assert_eq!(
            error.kind(),
            io::ErrorKind::Other,
            "an injected failure, not a real ENOENT: {error:?}"
        );

        assert!(
            image_path(&dir, &from_stem).exists(),
            "the pivot must stay put"
        );
        assert!(
            meta_path(&dir, &from_stem).exists(),
            "a post-pivot file must never be touched once the pivot itself failed"
        );
        assert!(!image_path(&dir, &to_stem).exists());
        assert!(!meta_path(&dir, &to_stem).exists());

        let _ = fs::remove_dir_all(&dir);
    }

    /// The best-effort half of the same contract: an injected failure
    /// on a post-pivot file (here `.meta.json`, `context_files`'
    /// second entry) must behave like the existing real-FS-blocker
    /// tests (`lifecycle.rs`'s straggler tests) already prove for a
    /// genuine `fs::rename` error — the pivot has already moved, so
    /// the loop keeps going rather than stopping, and a later file in
    /// the family (here `sources_path`, the third entry) still lands
    /// at `to_stem`. The first straggler error is what
    /// `move_context_files` reports.
    #[test]
    fn an_injected_straggler_rename_failure_still_lets_later_files_move() {
        let dir = scratch_dir("meta-io-move-straggler-fault");
        fs::create_dir_all(&dir).unwrap();
        let from_stem = file_stem("sake");
        let to_stem = file_stem("shochu");
        fs::write(image_path(&dir, &from_stem), b"ctx bytes").unwrap();
        fs::write(meta_path(&dir, &from_stem), b"meta bytes").unwrap();
        fs::write(sources_path(&dir, &from_stem), b"sources bytes").unwrap();

        // One success (the pivot) then fail the very next op — the
        // `.meta.json` rename.
        fail_persistence_ops_after(1);
        let error = move_context_files(&dir, &from_stem, &to_stem).unwrap_err();
        let past_end = clear_persistence_fault();
        assert!(!past_end, "the meta rename must be what failed");
        assert_eq!(error.kind(), io::ErrorKind::Other, "{error:?}");

        assert!(
            !image_path(&dir, &from_stem).exists(),
            "the pivot already moved before the fault"
        );
        assert!(image_path(&dir, &to_stem).exists());
        assert!(
            meta_path(&dir, &from_stem).exists(),
            "the failed rename must never have touched the source"
        );
        assert!(!meta_path(&dir, &to_stem).exists());
        assert!(
            !sources_path(&dir, &from_stem).exists() && sources_path(&dir, &to_stem).exists(),
            "best-effort continues past a post-pivot straggler: a LATER \
             family member still lands at the destination"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
