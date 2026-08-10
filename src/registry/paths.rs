use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::storage::write_atomic;

/// Encodes a context name as a file stem: bytes outside [A-Za-z0-9_-]
/// become %XX. Context names arrive from URL paths and may contain path
/// separators or dots; encoding them keeps every name inside the data
/// directory (no traversal) and reversible.
pub(crate) fn file_stem(name: &str) -> String {
    let mut stem = String::new();
    for byte in name.bytes() {
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' => stem.push(byte as char),
            _ => stem.push_str(&format!("%{byte:02X}")),
        }
    }
    stem
}

/// Decodes [`file_stem`]'s encoding back into a context name.
pub(crate) fn name_from_stem(stem: &str) -> Option<String> {
    let mut bytes = Vec::with_capacity(stem.len());
    let mut cursor = stem.bytes();
    while let Some(byte) = cursor.next() {
        if byte == b'%' {
            let high = cursor.next()?;
            let low = cursor.next()?;
            let hex = [high, low];
            let hex = std::str::from_utf8(&hex).ok()?;
            bytes.push(u8::from_str_radix(hex, 16).ok()?);
        } else {
            bytes.push(byte);
        }
    }
    String::from_utf8(bytes).ok()
}

/// FNV-1a over raw bytes — the same primitive the search terms build
/// on, kept here for the one non-search need (import marker file
/// names, below).
fn fnv64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// `pub(super)` for terms.rs, which runs the same FNV-1a inline over
// its word stream rather than calling `fnv64` on a slice.
pub(super) const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
pub(super) const FNV_PRIME: u64 = 0x1_0000_01b3;

pub(crate) fn image_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.ctx"))
}

pub(crate) fn meta_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.meta.json"))
}

pub(crate) fn sources_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.sources.json"))
}

pub(crate) fn passages_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.passages.bin"))
}

pub(crate) fn passages_wal_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.passages.wal.jsonl"))
}

pub(crate) fn pvectors_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.pvectors.bin"))
}

pub(crate) fn bm25_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.bm25.bin"))
}

pub(crate) fn vectors_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.vectors.bin"))
}

pub(crate) fn wal_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.wal.jsonl"))
}

/// The optional per-context schema document (ADR 0009 §5.1). Built with
/// `format!`, not `path.with_extension`, on purpose: `{stem}.schema.json`
/// is two dot segments, and `with_extension` would replace only the
/// last one, mangling the `.schema` half. Never discovered by an
/// extension scan the way `.ctx`/`.group` are — `Path::extension()`
/// would answer `Some("json")` and hand back the wrong stem — so this
/// builder is the only way a caller reaches the file, exactly like
/// [`meta_path`], which has the same two-segment shape for the same
/// reason.
pub(crate) fn schema_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.schema.json"))
}

/// Where a schema file's bytes are set aside when they read but do not
/// parse — evidence for hand recovery, [`crate::groups::scan_groups`]'s
/// `{stem}.group.corrupt` convention applied to schema (see
/// `crate::schema`'s module doc for why the parallel stops there: a
/// schema, unlike a group, never falls back to an empty record after
/// setting the bytes aside).
pub(crate) fn schema_corrupt_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.schema.corrupt"))
}

/// The durable-deletion marker: while it exists, boot resumes the
/// unlinks (see `delete`/`scan_data_dir`). One builder so the writer,
/// the boot sweep, and the create-time cleanup can never disagree
/// about its name — a stale marker beside a freshly recreated context
/// would otherwise make the next boot delete the new context.
pub(crate) fn deleted_marker_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.deleted"))
}

/// The durable-rename marker: while it exists, boot resumes the file
/// move AND re-applies the group membership rewrite (`contexts`
/// entries naming `from`) before `reconcile_groups` runs — without
/// that ordering, a crash between the move and the rewrite would have
/// reconcile see `from` as dangling and drop it, losing the
/// membership for good rather than carrying it to `to`. Removed only
/// once both the move and the rewrite are durable.
pub(crate) fn renaming_marker_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.renaming"))
}

/// The batch-open marker's file extension — `.deleted`'s sibling for
/// imports. Shared by the path builder, the boot sweep, and inspect,
/// so the three can never disagree about what counts as a marker.
pub(crate) const IMPORT_MARKER_EXTENSION: &str = "importing";

/// One import batch's in-flight marker: `{stem}.{fnv64(source)}.importing`,
/// written before the first of the batch's four separately-durable
/// mutations (retract_source → store_passages → add_associations →
/// add_aliases) and removed only after the last. While it exists, the
/// source's truth may be HALF-APPLIED — a crash between the steps
/// leaves passages without their associations, or associations without
/// their aliases, and every store is individually consistent, so
/// nothing else can tell. Boot and `taguru inspect` report survivors;
/// the repair is the documented one (re-import the batch file, whose
/// retract-then-apply is idempotent, or retract the source).
///
/// The source's name rides INSIDE the file (see [`ImportMarker`]); the
/// file name only needs to be unique per (context, source) and safe,
/// which the hash gives without an encoding scheme. Stems contain no
/// dots, so the `{stem}.` prefix plus the extension identifies a
/// marker's context unambiguously.
pub(crate) fn import_marker_path(dir: &Path, stem: &str, source: &str) -> PathBuf {
    dir.join(format!(
        "{stem}.{:016x}.{IMPORT_MARKER_EXTENSION}",
        fnv64(source.as_bytes())
    ))
}

/// Every import marker beside `stem`'s files — the enumeration the
/// delete and create sweeps need, since markers (unlike the fixed
/// `context_files` family) exist per in-flight source. Read failures
/// yield the empty list: both sweeps treat markers as best-effort
/// hygiene backed by boot's own pass.
pub(crate) fn import_marker_paths(dir: &Path, stem: &str) -> Vec<PathBuf> {
    let prefix = format!("{stem}.");
    let suffix = format!(".{IMPORT_MARKER_EXTENSION}");
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(&prefix) && name.ends_with(&suffix))
        })
        .collect()
}

/// What an import marker file says: which source's batch was open, in
/// which context — self-describing, so boot and inspect report the
/// human-readable pair without decoding file names.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ImportMarker {
    pub(crate) context: String,
    pub(crate) source: String,
}

/// What a rename marker file says: the source and destination names,
/// self-describing so boot can resume the move and the group rewrite
/// without any other input. Shared shape for contexts (`.renaming`)
/// and groups (`.grouprenaming`) — the two use different extensions
/// (a context and a group may share a name) but the same fields.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RenameMarker {
    pub(crate) from: String,
    pub(crate) to: String,
}

/// One rename whose marker `scan_data_dir`/`groups::scan_groups` found
/// at boot and tried to finish, handed back so `boot_with` can act
/// before `reconcile_groups` runs.
///
/// The two booleans decouple the two things a resume owes, because a
/// half-done move must not do the second without the first:
/// - `landed` — the destination's pivot file (a context's `.ctx`, a
///   group's `.group`) is now in place, so the scan registered the
///   entity under `to`. Group membership naming `from` must be
///   rewritten to `to`, or `reconcile_groups` — which has no notion of
///   a rename in flight — reads `from` as dangling and drops it.
/// - `complete` — every present file moved, so the marker has done its
///   job and may be removed. If a straggler sidecar was still held,
///   this stays false and the marker survives for the next boot to
///   retry, even though `landed` (and the membership rewrite) already
///   went through. Deleting the marker on a `landed`-but-not-`complete`
///   resume was the bug: the retry vanished, orphaning the straggler.
pub(crate) struct ResumedRename {
    pub(crate) from: String,
    pub(crate) to: String,
    pub(crate) landed: bool,
    pub(crate) complete: bool,
}

/// Every rename resumed in one boot scan (see [`ResumedRename`]).
pub(crate) type ResumedRenames = Vec<ResumedRename>;

/// Serializes and durably writes a rename marker at `path` — the
/// first step of both `rename_context_locked` and `rename_group`,
/// which must land before anything else moves (see their docs for
/// why the marker comes first and is not best-effort).
pub(super) fn write_rename_marker(path: &Path, from: &str, to: &str) -> io::Result<()> {
    let body = serde_json::to_vec(&RenameMarker {
        from: from.to_string(),
        to: to.to_string(),
    })
    .expect("RenameMarker has no non-serializable field");
    write_atomic(path, &body)
}

/// Resumes every `extension` rename marker found in `dir`: reads it,
/// parses the `(from, to)` pair, moves that pair's files via
/// `move_files`, and returns every pair resumed (see [`ResumedRename`]
/// for what the two per-rename booleans mean and why the caller needs
/// both). `scan_data_dir` (`.renaming`, a ten-file context family) and
/// `groups::scan_groups` (`.grouprenaming`, one file) share this exact
/// shape and differ only in what "moving the files" means for their
/// entity — `entity` names it for the log lines (`"context"` /
/// `"group"`).
///
/// `destination_landed(to_stem)` answers "is the destination's pivot
/// file now in place?" — checked whether or not `move_files` returned
/// Ok, because a move can fail on a straggler AFTER the pivot arrived.
/// That is `landed`; `move_files` returning Ok is `complete`.
///
/// A marker whose `from` or `to` also names another marker's `from` or
/// `to` — a chain (`a→b`, `b→c`), a cycle (`a→b`, `b→a`), or two
/// markers merely colliding on one shared endpoint (`a→b`, `a→c`; or
/// `a→c`, `b→c`) — is refused rather than resumed: nothing in this
/// codebase ever produces one on a live server (`rename_context`/
/// `rename_group` each reserve both names for the whole call, so two
/// markers can never share an endpoint from ordinary operation), so
/// one can only mean hand-edited files or a corruption this function
/// has no business inventing multi-hop or last-write-wins semantics
/// for. Every marker touching a shared endpoint is left on disk,
/// untouched, for a human to resolve — moving SOME of them and not
/// others would make the eventual manual fix harder, not easier.
/// Markers are read and parsed in full BEFORE any of them acts
/// (detecting the collision needs the whole set at once), then acted
/// on in `from`-sorted order — deterministic, unlike `read_dir`'s own
/// platform-dependent iteration order.
pub(crate) fn resume_rename_markers(
    dir: &Path,
    extension: &str,
    entity: &str,
    mut move_files: impl FnMut(&str, &str) -> io::Result<()>,
    destination_landed: impl Fn(&str) -> bool,
) -> io::Result<ResumedRenames> {
    let mut markers = Vec::new();
    for dir_entry in fs::read_dir(dir)? {
        let path = dir_entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some(extension) {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else {
            tracing::warn!(path = %path.display(), entity, "unreadable rename marker; a rename may be stuck half-done");
            continue;
        };
        let Ok(marker) = serde_json::from_slice::<RenameMarker>(&bytes) else {
            tracing::warn!(path = %path.display(), entity, "rename marker does not parse; a rename may be stuck half-done");
            continue;
        };
        markers.push(marker);
    }
    markers.sort_by(|a, b| a.from.cmp(&b.from));

    // Every marker contributes both its endpoints to this count, so an
    // endpoint occurring twice catches all three hazards uniformly: a
    // chain/cycle (one marker's `to` is another's `from`), and two
    // markers merely colliding on the same `from` or the same `to`.
    let mut endpoint_counts: HashMap<&str, usize> = HashMap::new();
    for marker in &markers {
        *endpoint_counts.entry(marker.from.as_str()).or_insert(0) += 1;
        *endpoint_counts.entry(marker.to.as_str()).or_insert(0) += 1;
    }
    let conflicted: BTreeSet<String> = endpoint_counts
        .into_iter()
        .filter(|&(_, count)| count >= 2)
        .map(|(name, _)| name.to_string())
        .collect();

    let mut resumed = Vec::new();
    for marker in markers {
        if conflicted.contains(&marker.from) || conflicted.contains(&marker.to) {
            // Not a failed operation — the marker is left exactly as
            // it was, and the boot that found it still succeeds. `warn`
            // matches that: `error` is reserved for outcomes the
            // caller must actually contend with (e.g. `RenameOutcome::
            // Stuck`'s real I/O failure), not a degraded-but-handled
            // refusal like this one.
            tracing::warn!(
                from = %marker.from, to = %marker.to, entity,
                "a rename marker collides with another on a shared endpoint; \
                 refusing to resume it automatically — resolve the data \
                 directory by hand"
            );
            continue;
        }
        tracing::warn!(from = %marker.from, to = %marker.to, entity, "resuming an unfinished rename");
        let to_stem = file_stem(&marker.to);
        let complete = match move_files(&file_stem(&marker.from), &to_stem) {
            Ok(()) => true,
            Err(error) => {
                tracing::warn!(from = %marker.from, to = %marker.to, entity, %error, "unfinished rename: file still held");
                false
            }
        };
        // Ask the disk, not the move's return value: a straggler sidecar
        // can stick (complete = false) long after the pivot moved, and
        // the membership rewrite keys on the pivot, not on completeness.
        let landed = destination_landed(&to_stem);
        resumed.push(ResumedRename {
            from: marker.from,
            to: marker.to,
            landed,
            complete,
        });
    }
    Ok(resumed)
}

/// Every rename marker of `extension` in `dir` that names `context` as
/// its DESTINATION. A marker sits at its SOURCE's stem, so a create of
/// the destination name cannot find it positionally the way it clears
/// the marker at its own stem; this scan lets the create sweep abandon a
/// half-done rename that would otherwise have boot's resume move the
/// source family over the fresh generation. Unreadable or unparseable
/// markers are skipped — boot's own sweep reports them. Shared by the
/// context (`renaming`) and group (`grouprenaming`) create paths.
pub(super) fn rename_markers_targeting(dir: &Path, context: &str, extension: &str) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some(extension))
        .filter(|path| {
            fs::read(path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<RenameMarker>(&bytes).ok())
                .is_some_and(|marker| marker.to == context)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::test_support::scratch_dir;

    /// The half-done-move contract `boot_with` leans on. `landed` and
    /// `complete` must move independently: a failed move is never
    /// complete (so the marker stays for the next boot to retry), and
    /// membership may only be rewritten once the destination pivot has
    /// landed. Deleting the marker on a failed move was the bug — the
    /// retry vanished and the group association was lost with no way
    /// back.
    #[test]
    fn a_failed_resume_keeps_the_marker_and_defers_membership() {
        let dir = scratch_dir("resume-failure");
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            renaming_marker_path(&dir, &file_stem("sake")),
            serde_json::to_vec(&RenameMarker {
                from: "sake".to_string(),
                to: "shochu".to_string(),
            })
            .unwrap(),
        )
        .unwrap();

        // Move fails and no pivot appears at the destination: neither
        // bit set — boot_with rewrites no membership and keeps the
        // marker. resume_rename_markers itself never removes a marker.
        let resumed = resume_rename_markers(
            &dir,
            "renaming",
            "context",
            |_, _| Err(io::Error::other("file still held")),
            |_| false,
        )
        .unwrap();
        assert_eq!(resumed.len(), 1);
        assert_eq!(resumed[0].from, "sake");
        assert_eq!(resumed[0].to, "shochu");
        assert!(!resumed[0].complete, "a failed move is not complete");
        assert!(!resumed[0].landed, "no pivot at the destination");
        assert!(
            renaming_marker_path(&dir, &file_stem("sake")).exists(),
            "the marker must survive a failed resume so the next boot retries"
        );

        // Pivot landed but a straggler stuck: landed (rewrite
        // membership) yet not complete (keep the marker to finish).
        let resumed = resume_rename_markers(
            &dir,
            "renaming",
            "context",
            |_, _| Err(io::Error::other("sidecar still held")),
            |_| true,
        )
        .unwrap();
        assert!(resumed[0].landed, "the pivot is at the destination");
        assert!(!resumed[0].complete, "a stuck straggler is not complete");

        // Everything moved: both bits set — rewrite membership, drop marker.
        let resumed =
            resume_rename_markers(&dir, "renaming", "context", |_, _| Ok(()), |_| true).unwrap();
        assert!(resumed[0].landed && resumed[0].complete);

        let _ = fs::remove_dir_all(dir);
    }

    fn write_marker(dir: &Path, at_stem: &str, from: &str, to: &str) {
        fs::write(
            renaming_marker_path(dir, at_stem),
            serde_json::to_vec(&RenameMarker {
                from: from.to_string(),
                to: to.to_string(),
            })
            .unwrap(),
        )
        .unwrap();
    }

    /// Resumes markers in `dir` and records every attempted move, for
    /// tests that need to assert on both the returned `ResumedRenames`
    /// and which moves were (or weren't) attempted.
    fn resume_and_record(dir: &Path) -> (ResumedRenames, Vec<(String, String)>) {
        let mut moved = Vec::new();
        let resumed = resume_rename_markers(
            dir,
            "renaming",
            "context",
            |from_stem, to_stem| {
                moved.push((from_stem.to_string(), to_stem.to_string()));
                Ok(())
            },
            |_| true,
        )
        .unwrap();
        (resumed, moved)
    }

    /// Regression for issue #561's item 9: a chain (`a→b`, `b→c`) is
    /// refused wholesale rather than resumed in whatever order
    /// `read_dir` happens to return — resuming a→b before b→c would
    /// collapse the marker for the SAME intended a→c move into a
    /// deterministic-looking (but never requested) two-step, and the
    /// reverse order would strand a→b having moved nothing.
    #[test]
    fn a_chained_rename_marker_pair_is_refused_not_resumed() {
        let dir = scratch_dir("resume-chain");
        fs::create_dir_all(&dir).unwrap();
        write_marker(&dir, &file_stem("a"), "a", "b");
        write_marker(&dir, &file_stem("b"), "b", "c");

        let (resumed, moved) = resume_and_record(&dir);

        assert!(
            resumed.is_empty(),
            "a chained pair must resume neither marker: {:?}",
            resumed.iter().map(|r| (&r.from, &r.to)).collect::<Vec<_>>()
        );
        assert!(
            moved.is_empty(),
            "neither marker's move may even be attempted"
        );
        assert!(
            renaming_marker_path(&dir, &file_stem("a")).exists(),
            "a→b's marker must survive for a human to resolve"
        );
        assert!(
            renaming_marker_path(&dir, &file_stem("b")).exists(),
            "b→c's marker must survive for a human to resolve"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// The cyclic shape of the same hazard: `a→b`, `b→a`. Resuming
    /// either alone would move one family onto the other's name while
    /// the second marker still claims that exact pair, in either
    /// order — refused wholesale, same as the chain.
    #[test]
    fn a_cyclic_rename_marker_pair_is_refused() {
        let dir = scratch_dir("resume-cycle");
        fs::create_dir_all(&dir).unwrap();
        write_marker(&dir, &file_stem("a"), "a", "b");
        write_marker(&dir, &file_stem("b"), "b", "a");

        let resumed =
            resume_rename_markers(&dir, "renaming", "context", |_, _| Ok(()), |_| true).unwrap();

        assert!(
            resumed.is_empty(),
            "a cyclic pair must resume neither marker"
        );
        assert!(renaming_marker_path(&dir, &file_stem("a")).exists());
        assert!(renaming_marker_path(&dir, &file_stem("b")).exists());

        let _ = fs::remove_dir_all(&dir);
    }

    /// A shape the chain/cycle check alone would miss: two markers
    /// sharing a `from` (`a→b`, `a→c`) rather than one's `to` feeding
    /// the other's `from`. Resuming either first would move "a"'s
    /// family out from under the other marker's own claim on the same
    /// source, so both must be refused.
    #[test]
    fn markers_sharing_a_from_are_both_refused() {
        let dir = scratch_dir("resume-shared-from");
        fs::create_dir_all(&dir).unwrap();
        write_marker(&dir, &file_stem("a"), "a", "b");
        // A second marker also claiming "a" as its source can only
        // exist via hand-editing or corruption — ordinary code never
        // writes two markers for the same `from` — so it lands at an
        // unrelated file stem rather than the one `from: "a"` would
        // normally use.
        write_marker(&dir, &file_stem("z"), "a", "c");

        let (resumed, moved) = resume_and_record(&dir);

        assert!(resumed.is_empty(), "neither marker may resume");
        assert!(
            moved.is_empty(),
            "neither marker's move may even be attempted"
        );
        assert!(renaming_marker_path(&dir, &file_stem("a")).exists());
        assert!(renaming_marker_path(&dir, &file_stem("z")).exists());

        let _ = fs::remove_dir_all(&dir);
    }

    /// The mirror shape: two markers sharing a `to` (`a→c`, `b→c`)
    /// instead of a `from`. Both would land at "c", so both are
    /// refused.
    #[test]
    fn markers_sharing_a_to_are_both_refused() {
        let dir = scratch_dir("resume-shared-to");
        fs::create_dir_all(&dir).unwrap();
        write_marker(&dir, &file_stem("a"), "a", "c");
        write_marker(&dir, &file_stem("b"), "b", "c");

        let (resumed, moved) = resume_and_record(&dir);

        assert!(resumed.is_empty(), "neither marker may resume");
        assert!(
            moved.is_empty(),
            "neither marker's move may even be attempted"
        );
        assert!(renaming_marker_path(&dir, &file_stem("a")).exists());
        assert!(renaming_marker_path(&dir, &file_stem("b")).exists());

        let _ = fs::remove_dir_all(&dir);
    }

    /// The refusal must not be broader than the hazard it guards
    /// against: two markers that share no endpoint resume exactly as
    /// they always have, in a deterministic (`from`-sorted) order.
    #[test]
    fn unrelated_markers_resume_in_a_deterministic_order() {
        let dir = scratch_dir("resume-unrelated");
        fs::create_dir_all(&dir).unwrap();
        write_marker(&dir, &file_stem("x"), "x", "y");
        write_marker(&dir, &file_stem("p"), "p", "q");

        let (resumed, moved) = resume_and_record(&dir);

        assert_eq!(resumed.len(), 2, "both unrelated markers must resume");
        assert!(
            resumed
                .iter()
                .all(|rename| rename.landed && rename.complete)
        );
        // Sorted by `from`: "p" < "x".
        assert_eq!(
            moved,
            vec![
                ("p".to_string(), "q".to_string()),
                ("x".to_string(), "y".to_string()),
            ]
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
