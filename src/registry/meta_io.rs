//! The `{stem}.meta.json` sidecar and the file-family operations built
//! beside it: the sidecar's on-disk shape (`MetaFile`), the
//! save/read pair every flush and boot goes through, and the
//! whole-family list/move helpers the delete loop, the boot sweep,
//! and rename recovery all share.

use super::*;

/// What `{name}.meta.json` holds: the meta inline plus the stats
/// snapshot as of the last save, so a directory listing can describe a
/// cold `context` without touching its image. `usage` rides along under
/// `#[serde(default)]`, so sidecars from before it existed load with
/// zeroed counters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct MetaFile {
    /// The `context`'s id (ADR 0045): the UUID the server minted at
    /// create, which is also this file family's stem. Recorded so the
    /// sidecar is self-describing, and so boot can tell a pre-id data
    /// directory (`None` — refused with a migration message) from a
    /// current one. Boot cross-checks it against the stem it found
    /// the file under; the two disagreeing means the family was
    /// copied or hand-edited, and the boot refuses rather than guess.
    pub(super) id: Option<String>,
    /// The `context`'s display name (ADR 0045): a free string, NOT
    /// unique (issue #961 decision 1) and no longer derivable from
    /// the stem — this field is the one durable place it lives, and
    /// boot's scan reads it from here.
    pub(super) name: Option<String>,
    #[serde(flatten)]
    pub(super) meta: ContextMeta,
    pub(super) stats: ContextStats,
    pub(super) usage: ContextUsage,
    /// The revision counters as of this save — what a cold entry (and
    /// a replica's tailed refresh) seeds from. Defaulted for sidecars
    /// from before the field existed, and for sidecars that simply do
    /// not exist yet (a fresh `context`): those report zeros until their
    /// first load or flush catches them up. A sidecar that DOES exist
    /// but cannot be read or parsed is a different case — see
    /// [`MetaFile::degraded`].
    pub(super) revision: ContextRevision,
    /// `sha256_hex` of `{stem}.schema.json`'s bytes as of this save,
    /// `None` for a schema-free `context` — ADR 0009 §5.2's boot-time
    /// consistency check. Written in the SAME `write_meta` call that
    /// bumps `config_revision` (never separately), so a crash between
    /// this field landing and the schema file's own `write_atomic`
    /// leaves a detectable disagreement rather than a silently stale
    /// enforcement: [`crate::schema::load_schema`] refuses to load
    /// whenever the file on disk and this recorded value disagree, in
    /// either direction. Defaulted for sidecars from before the field
    /// existed, exactly like `revision` above — a pre-#379 `context` has
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
    stem: &str,
    name: &str,
    meta: &ContextMeta,
    stats: &ContextStats,
    usage: &ContextUsage,
    revision: ContextRevision,
    schema_digest: Option<&str>,
    context: &Context,
) -> io::Result<()> {
    // The image is what `scan_data_dir` keys a context's existence on, so
    // it lands LAST: each `write_atomic` fully commits (fsync + rename +
    // parent-dir fsync) before returning, so by the time the `.ctx` is
    // durably in the directory its `.meta.json` companion already is too.
    // A crash between the two therefore leaves at worst an orphan sidecar
    // with no image — invisible to the scan and overwritten by the next
    // same-name create — never a durable image with a defaulted sidecar,
    // which would resurrect a context `create` told the client had failed.
    // (Image-then-meta would do exactly that; see `create`'s doc.)
    write_meta(dir, stem, name, meta, stats, usage, revision, schema_digest)?;
    write_atomic(&image_path(dir, stem), &context.to_bytes())
}

#[allow(clippy::too_many_arguments)] // every sidecar save call site, not an API
pub(super) fn write_meta(
    dir: &Path,
    stem: &str,
    name: &str,
    meta: &ContextMeta,
    stats: &ContextStats,
    usage: &ContextUsage,
    revision: ContextRevision,
    schema_digest: Option<&str>,
) -> io::Result<()> {
    let file = MetaFile {
        // The stem IS the id (ADR 0045): every write re-records it so
        // the sidecar can never drift from the family it sits in.
        id: Some(stem.to_string()),
        name: Some(name.to_string()),
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
/// that simply is not there (`ErrorKind::NotFound` — a fresh `context`,
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
/// `schema_digest` to `None`, which for a `context` that DOES have a
/// live `{stem}.schema.json` collides with `schema::load_schema`'s own
/// fail-closed posture (ADR 0009 §5.1/§5.2, issue #561's audit) — a
/// corrupt sidecar plus a healthy schema file turns into a
/// digest-mismatch refusal that stops the WHOLE boot, not just this
/// one candidate, and the resulting message names a mismatch rather
/// than the sidecar that caused it. The fix for that case is the
/// sidecar's, not the schema check's: restore `{stem}.meta.json` (or
/// delete it if the `context` has no schema) so its recorded digest
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

/// [`read_scanned_meta`]'s classification of one candidate's sidecar —
/// the boot scan must tell a healthy current sidecar apart from a
/// pre-id one (refused with a migration message) and from plain
/// degradation (tolerated exactly as [`read_meta_file`] tolerates it).
pub(crate) enum ScannedMeta {
    /// Parsed, and its recorded `id` equals the stem it sits under —
    /// the only healthy case.
    Current(MetaFile),
    /// Parsed, but records no `id`: a data directory written by a
    /// pre-id release (before ADR 0045). Refused — the stem is a
    /// percent-encoded name there, not an id, and registering it
    /// would serve the encoding as identity. Migration is export on
    /// the old release, import on this one (§2.7: no compatibility).
    PreId,
    /// Parsed, but records an `id` other than the stem it sits under:
    /// the family was copied or hand-edited. Refused — trusting
    /// either value would silently rebind the other.
    ForeignId(String),
    /// Missing, unreadable, or corrupt — the same lenient fallback
    /// [`read_meta_file`] serves (already logged there when it is a
    /// real degradation). The id and name fall back to the stem; for
    /// the id that is even exact (the stem IS the id), for the name
    /// it is a display fallback an operator can rename away.
    Degraded(MetaFile),
}

impl ScannedMeta {
    /// The display name this classification yields for `stem`, with
    /// the same fallback the boot scan applies (the stem itself for a
    /// degraded sidecar, or one that records no name); `None` for the
    /// two refused shapes, which have no honest name to report.
    pub(crate) fn display_name(&self, stem: &str) -> Option<String> {
        match self {
            ScannedMeta::Current(meta_file) | ScannedMeta::Degraded(meta_file) => {
                Some(meta_file.name.clone().unwrap_or_else(|| stem.to_string()))
            }
            ScannedMeta::PreId | ScannedMeta::ForeignId(_) => None,
        }
    }
}

/// [`read_meta_file`] plus the id-vs-stem classification above — the
/// boot scan's one read of a candidate's sidecar.
pub(crate) fn read_scanned_meta(dir: &Path, stem: &str) -> ScannedMeta {
    match fs::read(meta_path(dir, stem)) {
        Ok(bytes) => match serde_json::from_slice::<MetaFile>(&bytes) {
            Ok(meta_file) => match meta_file.id.as_deref() {
                Some(id) if id == stem => ScannedMeta::Current(meta_file),
                Some(id) => ScannedMeta::ForeignId(id.to_string()),
                None => ScannedMeta::PreId,
            },
            Err(error) => {
                tracing::warn!("ignoring corrupt sidecar for '{stem}': {error}");
                ScannedMeta::Degraded(MetaFile::degraded())
            }
        },
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            ScannedMeta::Degraded(MetaFile::default())
        }
        Err(error) => {
            tracing::warn!("sidecar for '{stem}' unreadable, falling back to defaults: {error}");
            ScannedMeta::Degraded(MetaFile::degraded())
        }
    }
}

/// The stems whose sidecar records the display name `name` — the
/// offline twin of the registry's name index, for tools that open a
/// data directory without booting one (`taguru-code`, diagnostics).
/// Names are not unique (issue #961 decision 1), so this returns
/// every claimant and the caller decides how to refuse ambiguity.
/// Read errors on individual sidecars are skipped, matching the boot
/// scan's lenient posture; only the directory listing itself can
/// fail.
#[allow(dead_code)] // consumed by the taguru-code binary only
pub(crate) fn stems_named(dir: &Path, name: &str) -> io::Result<Vec<String>> {
    let mut stems = Vec::new();
    for dir_entry in fs::read_dir(dir)? {
        let path = dir_entry?.path();
        let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        let Some(stem) = file_name.strip_suffix(".meta.json") else {
            continue;
        };
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        let Ok(meta_file) = serde_json::from_slice::<MetaFile>(&bytes) else {
            continue;
        };
        if meta_file.name.as_deref() == Some(name) {
            stems.push(stem.to_string());
        }
    }
    stems.sort();
    Ok(stems)
}

/// The recorded schema digest alone, for a caller (`taguru inspect`)
/// that has no use for the rest of the sidecar and must not import
/// `MetaFile` (private to this module). Same lenient fallback as
/// [`read_meta_file`]: an unreadable or corrupt sidecar reports `None`
/// here exactly as it would seed a fresh [`MetaFile::default`] at boot,
/// so inspect's schema check judges a `context` by the same recorded
/// value boot itself would.
pub(crate) fn schema_digest_of(dir: &Path, stem: &str) -> Option<String> {
    read_meta_file(dir, stem).schema_digest
}

/// One `context`'s whole file family, by stem — the delete loop and the
/// boot-time deletion sweep must never disagree about what "the whole
/// family" means, so both read this one list.
///
/// A curated list, not a mechanical projection of `paths.rs`: the
/// builders not listed here (`schema_corrupt_path`,
/// `deleted_marker_path`, `import_marker_path`) are markers and
/// quarantine files, deliberately NOT family members. Adding an eleventh FAMILY file kind therefore
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::test_support::scratch_dir;

    /// The ordinary, silent case: no sidecar has ever existed for this
    /// stem (a brand new `context`, or one from before the file existed).
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
}
