use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::storage::write_atomic;

use super::{file_stem, fnv64};

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
/// both). `scan_data_dir` (`.renaming`, a nine-file context family) and
/// `groups::scan_groups` (`.grouprenaming`, one file) share this exact
/// shape and differ only in what "moving the files" means for their
/// entity — `entity` names it for the log lines (`"context"` /
/// `"group"`).
///
/// `destination_landed(to_stem)` answers "is the destination's pivot
/// file now in place?" — checked whether or not `move_files` returned
/// Ok, because a move can fail on a straggler AFTER the pivot arrived.
/// That is `landed`; `move_files` returning Ok is `complete`.
pub(crate) fn resume_rename_markers(
    dir: &Path,
    extension: &str,
    entity: &str,
    mut move_files: impl FnMut(&str, &str) -> io::Result<()>,
    destination_landed: impl Fn(&str) -> bool,
) -> io::Result<ResumedRenames> {
    let mut resumed = Vec::new();
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
}
