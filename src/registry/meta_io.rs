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
    /// from before the field existed: those report zeros until their
    /// first load or flush catches them up.
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
/// That leniency has one sharp edge: the fallback also zeroes
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
            MetaFile::default()
        }),
        Err(_) => MetaFile::default(),
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
/// family" means, so both read this one list. Built from the same ten
/// path builders every other caller uses, so a file kind added there
/// cannot silently miss this list.
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
/// paying for it (via `commit_staged`) up to ten times. The fsync's own
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
        match fs::rename(data_dir.join(from_file), data_dir.join(to_file)) {
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
