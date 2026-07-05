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

use serde::{Deserialize, Serialize};

/// One graph mutation, in the same vocabulary the `Context` write API
/// speaks — replay is just calling the same function again.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum WalOp {
    Associate(crate::registry::AssocOp),
    AliasConcept { alias: String, canonical: String },
    AliasLabel { alias: String, canonical: String },
    RetractSource { source: String },
}

#[derive(Serialize, Deserialize)]
struct WalRecord {
    seq: u64,
    #[serde(flatten)]
    op: WalOp,
}

/// Appends `ops` numbered from `first_seq`, one line each, with a
/// single fsync after all of them — the HTTP batch is the natural
/// group-commit unit (one document, one request, one lock, one sync).
/// On `Err` nothing may be assumed durable; the caller must not have
/// applied anything yet (write-ahead: log, sync, THEN apply). Callers
/// serialize appends per log (the context's entry lock), so only
/// crashes race this function, never other appenders.
pub fn append_batch(path: &Path, first_seq: u64, ops: &[WalOp]) -> io::Result<()> {
    let mut buffer = Vec::new();
    for (offset, op) in ops.iter().enumerate() {
        let record = WalRecord {
            seq: first_seq + offset as u64,
            op: op.clone(),
        };
        serde_json::to_writer(&mut buffer, &record)?;
        buffer.push(b'\n');
    }
    // Creating the log adds an entry to the parent directory's own
    // data: without syncing the directory too, power loss can drop the
    // whole file even though its contents were fsynced — the same rule
    // `write_atomic` follows for renames. `create_new` tells the two
    // cases apart in the open itself.
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
    let length_before = file.metadata()?.len();
    let outcome = file
        .write_all(&buffer)
        .and_then(|()| file.sync_all())
        .and_then(|()| {
            if created {
                crate::registry::fsync_parent_dir(path)
            } else {
                Ok(())
            }
        });
    if let Err(error) = outcome {
        // The caller refuses the write on `Err` and hands the same seq
        // numbers to the next batch — so any bytes that DID land here
        // would later replay as ghost records beside the real ones,
        // double-applying their seqs. Put the log back exactly as it
        // was. Best effort: if even this fails the disk is failing
        // twice over, and replay's torn-tail rule still absorbs the
        // common partial-append shape.
        let _ = file.set_len(length_before);
        return Err(error);
    }
    Ok(())
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
pub fn replay(path: &Path, watermark: u64) -> io::Result<(Vec<WalOp>, u64)> {
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
        tracing::warn!(
            "dropping a torn trailing WAL record at {} ({} bytes) — crash mid-append",
            path.display(),
            tail.len(),
        );
    }

    let mut ops = Vec::new();
    let mut top = watermark;
    for (index, line) in segments.iter().enumerate() {
        let record: WalRecord = serde_json::from_slice(line).map_err(|error| {
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
    match fs::OpenOptions::new().write(true).open(path) {
        Ok(file) => {
            file.set_len(0)?;
            file.sync_all()
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
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
        let (none, top) = replay(&path, 9).unwrap();
        assert!(none.is_empty());
        assert_eq!(top, 9);
    }

    #[test]
    fn a_torn_trailing_line_is_dropped_not_fatal() {
        let path = scratch_wal("torn");
        append_batch(&path, 1, &[associate("a")]).unwrap();
        // Simulate a crash mid-append: valid bytes, no final newline.
        let mut bytes = fs::read(&path).unwrap();
        bytes.extend_from_slice(br#"{"seq":2,"op":"associate","subject":"b""#);
        fs::write(&path, bytes).unwrap();

        let (ops, top) = replay(&path, 0).unwrap();
        assert_eq!(ops.len(), 1, "the complete record survives");
        assert_eq!(subject_of(&ops[0]), "a");
        assert_eq!(top, 1);
    }

    #[test]
    fn a_corrupt_interior_line_is_an_error_not_a_skip() {
        let path = scratch_wal("corrupt");
        append_batch(&path, 1, &[associate("a")]).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes.extend_from_slice(b"garbage that is not a record\n");
        fs::write(&path, bytes).unwrap();
        append_batch(&path, 2, &[associate("b")]).unwrap();

        let error = replay(&path, 0).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("line 2"), "{error}");
    }

    #[test]
    fn a_missing_file_replays_to_nothing() {
        let path = scratch_wal("missing");
        let (ops, top) = replay(&path, 5).unwrap();
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
        let (ops, top) = replay(&path, 4).unwrap();
        assert!(ops.is_empty());
        assert_eq!(top, 4);
    }
}
