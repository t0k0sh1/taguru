//! Per-context write-ahead log: the acknowledged-write half of the
//! durability story. Every graph mutation the HTTP API accepts is
//! appended here (JSON Lines, one fsync per batch) BEFORE it touches
//! memory, so a crash between two image flushes loses nothing.
//!
//! Replay is driven entirely by sequence numbers: records carry a
//! monotonic `seq`, the image header carries the watermark of the last
//! seq already baked in (see `Context::applied_seq`), and loading
//! applies only records above it. The log's physical contents never
//! need to agree with the image — truncation after a flush is pure
//! housekeeping, and a crash between image write and truncate replays
//! nothing, because the fresh watermark already covers those records.
//! That indifference is the point: `associate` accumulates weight, so
//! a double-applied record would corrupt silently.

use std::fs;
use std::io::{self, Write};
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// One graph mutation, in the same vocabulary the `Context` write API
/// speaks — replay is just calling the same function again.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WalOp {
    Associate(crate::registry::AssocOp),
    AliasConcept { alias: String, canonical: String },
    AliasLabel { alias: String, canonical: String },
    // Additive variants keep old logs replayable forever (the tag
    // form ignores nothing it knows); a DOWNGRADED binary reading a
    // log that holds one of the newer ops refuses the boot as
    // corruption — which is the right refusal for records it cannot
    // apply.
    UnaliasConcept { alias: String },
    UnaliasLabel { alias: String },
    RetractSource { source: String },
}

/// The machinery below is generic over the op vocabulary: the graph
/// logs [`WalOp`] into `{stem}.wal.jsonl`, and any other store with its
/// own log supplies its own op enum (same internally-tagged shape) and
/// its own file. Each log is one vocabulary — mixing two op types in
/// one file would make every replay of it fail as corruption.
#[derive(Serialize, Deserialize)]
struct WalRecord<Op> {
    seq: u64,
    #[serde(flatten)]
    op: Op,
}

/// Appends `ops` numbered from `first_seq`, one line each, with a
/// single fsync after all of them — the HTTP batch is the natural
/// group-commit unit (one document, one request, one lock, one sync).
/// Returns the bytes appended, so the caller can track log growth.
/// On `Err` nothing may be assumed durable; the caller must not have
/// applied anything yet (write-ahead: log, sync, THEN apply). Callers
/// serialize appends per log (the context's entry lock), so only
/// crashes race this function, never other appenders.
pub fn append_batch<Op: Serialize>(path: &Path, first_seq: u64, ops: &[Op]) -> io::Result<u64> {
    let mut buffer = Vec::new();
    for (offset, op) in ops.iter().enumerate() {
        let record = WalRecord {
            seq: first_seq + offset as u64,
            op,
        };
        serde_json::to_writer(&mut buffer, &record)?;
        buffer.push(b'\n');
    }
    // `create_new` tells "we made the file" from "it was already there"
    // in the open itself — the distinction the directory sync below turns
    // on.
    let (mut file, created) = match fs::OpenOptions::new()
        .create_new(true)
        .append(true)
        .open(path)
    {
        Ok(file) => (file, true),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            (fs::OpenOptions::new().append(true).open(path)?, false)
        }
        Err(error) => return Err(error),
    };
    // A freshly created file adds an entry to the parent directory's own
    // data: without syncing the directory too, power loss can drop the
    // whole file even though its contents were fsynced — the same rule
    // `write_atomic` follows for renames. Sync it right after the create,
    // before any content, and if that sync fails remove the file: a
    // lingering file must never outlive its durability, because the next
    // append finds it via the `AlreadyExists` path above and does NOT
    // sync the directory again. (Deferring this sync until after the
    // content write — tied to the same `and_then` chain — let a create
    // whose content sync then failed leave an un-synced file that every
    // later append skipped forever.)
    if created && let Err(error) = crate::registry::fsync_parent_dir(path) {
        let _ = fs::remove_file(path);
        return Err(error);
    }
    let length_before = file.metadata()?.len();
    if let Err(error) = file.write_all(&buffer).and_then(|()| file.sync_all()) {
        // The caller refuses the write on `Err` and hands the same seq
        // numbers to the next batch — so any bytes that DID land here
        // would later replay as ghost records beside the real ones,
        // double-applying their seqs. Put the log back exactly as it was
        // (the directory entry is already durable, so an emptied new file
        // is a harmless, replay-inert leftover). Best effort: if even
        // this fails the disk is failing twice over, and replay's
        // torn-tail rule still absorbs the common partial-append shape.
        let _ = file.set_len(length_before);
        return Err(error);
    }
    Ok(buffer.len() as u64)
}

/// Reads the log back: every op with `seq > watermark` in file order,
/// plus the highest seq observed (or `watermark` when the file is
/// absent, empty, or entirely at/below it) — the caller numbers its
/// next write from there.
///
/// A trailing line without its `\n` is the half-written record of a
/// crash mid-append — the expected torn shape — and is dropped with a
/// warning. Any OTHER undecodable line is real corruption: unlike the
/// sidecars, this file holds acknowledged writes that exist nowhere
/// else, so skipping past it would be silent loss — it errors instead.
///
/// Dropping a torn tail from the returned ops is not enough on its own:
/// the bytes stay on disk, and the next `append_batch` opens with
/// `O_APPEND` and writes straight after them. The torn fragment and the
/// new record would then share a line with no `\n` between them, and
/// that fused line decodes as neither — turning a recoverable torn tail
/// into the fatal interior-corruption case above. So the tail is healed
/// in place here too: it was never acknowledged (its writer never got
/// `Ok`), so truncating it away loses nothing. Healing is best effort —
/// a failure only leaves the log as it was, to be retried next replay.
pub fn replay<Op: DeserializeOwned>(path: &Path, watermark: u64) -> io::Result<(Vec<Op>, u64)> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok((Vec::new(), watermark));
        }
        Err(error) => return Err(error),
    };

    let mut segments: Vec<&[u8]> = bytes.split(|&byte| byte == b'\n').collect();
    // A complete file ends in '\n', making the final segment empty; a
    // torn file's final segment is the record a crash cut short. One
    // rule covers both: the last segment is never a whole record.
    if let Some(tail) = segments.pop()
        && !tail.is_empty()
    {
        // Everything up to and including the last '\n' is intact; the
        // torn fragment is exactly the trailing `tail.len()` bytes.
        let healthy_len = (bytes.len() - tail.len()) as u64;
        tracing::warn!(
            "dropping a torn trailing WAL record at {} ({} bytes) — crash mid-append",
            path.display(),
            tail.len(),
        );
        if let Err(error) = truncate_to(path, healthy_len) {
            tracing::warn!(
                "could not heal torn WAL tail at {} (harmless, will retry next replay): {error}",
                path.display(),
            );
        }
    }

    let mut ops = Vec::new();
    let mut top = watermark;
    for (index, line) in segments.iter().enumerate() {
        let record: WalRecord<Op> = serde_json::from_slice(line).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "corrupt WAL record at {} line {}: {error}",
                    path.display(),
                    index + 1
                ),
            )
        })?;
        top = top.max(record.seq);
        if record.seq > watermark {
            ops.push(record.op);
        }
    }
    Ok((ops, top))
}

/// Empties the log in place (`set_len(0)`, same inode, no directory
/// change to sync). Housekeeping only: replay correctness rests on the
/// watermark comparison, never on what this file still contains, so a
/// failure here just leaves the log longer than necessary.
pub fn reset(path: &Path) -> io::Result<()> {
    match truncate_to(path, 0) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// Rewinds the log to exactly `len` bytes — same in-place, no-rename
/// shape as `reset`, but to an arbitrary prior length rather than
/// zero. The caller for this is a batch write whose live apply stopped
/// short of what was staged: durability appends the whole batch before
/// running it, so a partial apply leaves the tail describing ops that
/// were never actually tried. Left on disk, that tail is
/// indistinguishable from an applied record, and replay would try it
/// independently next time this context goes cold.
pub fn truncate_to(path: &Path, len: u64) -> io::Result<()> {
    let file = fs::OpenOptions::new().write(true).open(path)?;
    file.set_len(len)?;
    file.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::AssocOp;
    use std::path::PathBuf;

    fn scratch_wal(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("taguru-wal-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir.join("test.wal.jsonl")
    }

    fn associate(subject: &str) -> WalOp {
        WalOp::Associate(AssocOp {
            subject: subject.to_string(),
            label: "好き".to_string(),
            object: "りんご".to_string(),
            weight: 1.0,
            source: None,
            paragraph: None,
        })
    }

    fn subject_of(op: &WalOp) -> &str {
        match op {
            WalOp::Associate(op) => &op.subject,
            _ => panic!("test ops are associations"),
        }
    }

    #[test]
    fn append_then_replay_returns_only_ops_above_the_watermark() {
        let path = scratch_wal("filter");
        append_batch(&path, 1, &[associate("a"), associate("b")]).unwrap();
        append_batch(&path, 3, &[associate("c")]).unwrap();

        let (all, top) = replay(&path, 0).unwrap();
        assert_eq!(top, 3);
        let subjects: Vec<&str> = all.iter().map(subject_of).collect();
        assert_eq!(subjects, vec!["a", "b", "c"]);

        // A mid-point watermark keeps only the tail; the top still
        // reports the highest seq so numbering continues correctly.
        let (tail, top) = replay(&path, 2).unwrap();
        assert_eq!(top, 3);
        assert_eq!(tail.iter().map(subject_of).collect::<Vec<_>>(), vec!["c"]);

        // Fully covered: nothing to replay, top sticks to the watermark.
        let (none, top) = replay::<WalOp>(&path, 9).unwrap();
        assert!(none.is_empty());
        assert_eq!(top, 9);
    }

    #[test]
    fn a_torn_trailing_line_is_dropped_not_fatal() {
        let path = scratch_wal("torn");
        append_batch(&path, 1, &[associate("a")]).unwrap();
        let healthy_len = fs::metadata(&path).unwrap().len();
        // Simulate a crash mid-append: valid bytes, no final newline.
        let mut bytes = fs::read(&path).unwrap();
        bytes.extend_from_slice(br#"{"seq":2,"op":"associate","subject":"b""#);
        fs::write(&path, bytes).unwrap();

        let (ops, top) = replay(&path, 0).unwrap();
        assert_eq!(ops.len(), 1, "the complete record survives");
        assert_eq!(subject_of(&ops[0]), "a");
        assert_eq!(top, 1);
        // The torn bytes are healed off disk, not just filtered out of
        // the returned ops — the file ends back at its last newline.
        assert_eq!(
            fs::metadata(&path).unwrap().len(),
            healthy_len,
            "replay must truncate the torn tail in place"
        );
    }

    #[test]
    fn a_torn_tail_does_not_fuse_with_the_next_append() {
        // The regression this guards: before healing, replay dropped the
        // torn record from its result but left the bytes on disk, so the
        // next O_APPEND write landed straight after them and fused into a
        // single undecodable line — which the *next* replay then treated
        // as fatal interior corruption, bricking the context for good.
        let path = scratch_wal("fuse");
        append_batch(&path, 1, &[associate("a")]).unwrap();
        // Crash mid-append: a partial record with no closing newline.
        let mut bytes = fs::read(&path).unwrap();
        bytes.extend_from_slice(br#"{"seq":2,"op":"associate","subject":"b"#);
        fs::write(&path, bytes).unwrap();

        // Recovery replay heals the tail; numbering resumes from the
        // last intact record.
        let (ops, top) = replay(&path, 0).unwrap();
        assert_eq!(ops.iter().map(subject_of).collect::<Vec<_>>(), vec!["a"]);
        assert_eq!(top, 1);

        // A fresh append after recovery lands on a clean newline
        // boundary, so it stays a record of its own.
        append_batch(&path, top + 1, &[associate("c")]).unwrap();

        // The decisive check: the second replay succeeds instead of
        // erroring on a fused line.
        let (ops, top) = replay(&path, 0).unwrap();
        assert_eq!(
            ops.iter().map(subject_of).collect::<Vec<_>>(),
            vec!["a", "c"]
        );
        assert_eq!(top, 2);
    }

    #[test]
    fn a_corrupt_interior_line_is_an_error_not_a_skip() {
        let path = scratch_wal("corrupt");
        append_batch(&path, 1, &[associate("a")]).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes.extend_from_slice(b"garbage that is not a record\n");
        fs::write(&path, bytes).unwrap();
        append_batch(&path, 2, &[associate("b")]).unwrap();

        let error = replay::<WalOp>(&path, 0).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("line 2"), "{error}");
    }

    #[test]
    fn a_missing_file_replays_to_nothing() {
        let path = scratch_wal("missing");
        let (ops, top) = replay::<WalOp>(&path, 5).unwrap();
        assert!(ops.is_empty());
        assert_eq!(top, 5);
    }

    #[test]
    fn reset_truncates_in_place_and_tolerates_a_missing_file() {
        let path = scratch_wal("reset");
        reset(&path).unwrap(); // nothing there yet — fine
        append_batch(&path, 1, &[associate("a")]).unwrap();
        reset(&path).unwrap();
        assert_eq!(fs::metadata(&path).unwrap().len(), 0);
        // An emptied log replays to nothing and keeps the numbering.
        let (ops, top) = replay::<WalOp>(&path, 4).unwrap();
        assert!(ops.is_empty());
        assert_eq!(top, 4);
    }

    #[test]
    fn truncate_to_rewinds_to_an_arbitrary_prior_length() {
        let path = scratch_wal("truncate");
        append_batch(&path, 1, &[associate("a")]).unwrap();
        let len_before = fs::metadata(&path).unwrap().len();
        append_batch(&path, 2, &[associate("b"), associate("c")]).unwrap();

        truncate_to(&path, len_before).unwrap();

        assert_eq!(fs::metadata(&path).unwrap().len(), len_before);
        let (ops, top) = replay(&path, 0).unwrap();
        assert_eq!(
            ops.iter().map(subject_of).collect::<Vec<_>>(),
            vec!["a"],
            "the rewound tail must not survive"
        );
        assert_eq!(top, 1);
    }
}
