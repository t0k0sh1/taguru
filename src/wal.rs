//! Per-context write-ahead log: the acknowledged-write half of the
//! durability story. Every graph mutation the HTTP API accepts is
//! appended here (JSON Lines, one fsync per batch) BEFORE it touches
//! memory, so a crash between two image flushes loses nothing.
//!
//! Each record carries a CRC-32C of its own canonical bytes (the
//! `crc` field, always last), verified on every replay: structural
//! parsing catches truncation and garbage, but a flipped byte that
//! stays valid JSON would otherwise replay as truth. Records written
//! before the field existed replay unchecked, and a pre-checksum
//! binary reading a checksummed log ignores the field — the change is
//! additive in both directions.
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

use std::collections::BTreeMap;
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
    AliasConcept {
        alias: String,
        canonical: String,
    },
    AliasLabel {
        alias: String,
        canonical: String,
    },
    // Additive variants keep old logs replayable forever (the tag
    // form ignores nothing it knows); a DOWNGRADED binary reading a
    // log that holds one of the newer ops refuses the boot as
    // corruption — which is the right refusal for records it cannot
    // apply.
    UnaliasConcept {
        alias: String,
    },
    UnaliasLabel {
        alias: String,
    },
    RetractSource {
        source: String,
    },
    RetractAssociation {
        subject: String,
        label: String,
        object: String,
    },
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
    /// CRC-32C of this record's own serialization WITHOUT this field —
    /// the bit-rot check structural parsing cannot give (a flipped byte
    /// that stays valid JSON replays as truth). Always the record's
    /// LAST field, spliced in by [`append_batch`]; replay re-serializes
    /// the parsed record crc-less and compares. The record shape stays
    /// readable by a pre-checksum binary — its parser routes the
    /// unknown field into the flattened op, where serde ignores it —
    /// and records WITHOUT the field (written by one) replay unchecked,
    /// so the format change is additive in both directions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    crc: Option<u32>,
}

/// One record's canonical crc-less bytes — the exact serialization the
/// checksum covers, shared by the writer (which splices the field onto
/// it) and replay (which recomputes it from the parsed record). The op
/// vocabularies keep this reconstruction byte-exact: strings, integers,
/// ryu-printed floats, and Vecs all round-trip through serde_json to
/// the same bytes; none of them may grow a map field (iteration order
/// would differ between write and verify and fail every record).
fn canonical_record_bytes<Op: Serialize>(seq: u64, op: Op) -> serde_json::Result<Vec<u8>> {
    serde_json::to_vec(&WalRecord { seq, op, crc: None })
}

/// Test-only fault injection: arms the calling thread so that, after
/// `successes` more non-empty [`append_batch`] calls succeed normally,
/// the one after them fails with an injected error and the hook
/// disarms. This is how tests reach failure points that directory
/// permissions cannot select — e.g. a re-append failing right after an
/// append and a truncate on the same file succeeded.
#[cfg(test)]
pub(crate) fn fail_appends_after(successes: u32) {
    APPEND_FAULT.with(|cell| cell.set(Some(successes)));
}

#[cfg(test)]
thread_local! {
    static APPEND_FAULT: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn injected_append_failure() -> Option<io::Error> {
    APPEND_FAULT.with(|cell| match cell.get() {
        Some(0) => {
            cell.set(None);
            Some(io::Error::other("injected append failure"))
        }
        Some(remaining) => {
            cell.set(Some(remaining - 1));
            None
        }
        None => None,
    })
}

/// [`fail_appends_after`]'s twin for [`truncate_to`]: after `successes`
/// more truncates succeed normally, the one after them fails with an
/// injected error and the hook disarms. This is how tests reach the
/// partial-apply rollback failing while the batch append before it
/// succeeded — the reverse of what directory permissions can select.
#[cfg(test)]
pub(crate) fn fail_truncates_after(successes: u32) {
    TRUNCATE_FAULT.with(|cell| cell.set(Some(successes)));
}

#[cfg(test)]
thread_local! {
    static TRUNCATE_FAULT: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn injected_truncate_failure() -> Option<io::Error> {
    TRUNCATE_FAULT.with(|cell| match cell.get() {
        Some(0) => {
            cell.set(None);
            Some(io::Error::other("injected truncate failure"))
        }
        Some(remaining) => {
            cell.set(Some(remaining - 1));
            None
        }
        None => None,
    })
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
    // Nothing to append is nothing to sync: an empty batch would otherwise
    // create the file (or open it) and fsync it — and its parent directory
    // on first touch — to land zero bytes. Return before any I/O; the next
    // real append creates and syncs the file itself.
    if ops.is_empty() {
        return Ok(0);
    }
    #[cfg(test)]
    if let Some(error) = injected_append_failure() {
        return Err(error);
    }
    let mut buffer = Vec::new();
    for (offset, op) in ops.iter().enumerate() {
        let record = canonical_record_bytes(first_seq + offset as u64, op)?;
        let crc = crate::crc32c::crc32c(&record);
        // Splice the checksum in as the final field: identical bytes to
        // serializing the record with `crc: Some(..)` (a struct field
        // after a flatten lands last), for one serialization instead of
        // two. A record is always a JSON object, so it always ends '}'.
        buffer.extend_from_slice(&record[..record.len() - 1]);
        buffer.extend_from_slice(format!(",\"crc\":{crc}}}\n").as_bytes());
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
        // double-applying their seqs. And a failed sync leaves the
        // on-disk state UNKNOWN: the batch may sit there complete, and
        // a complete batch is not a torn tail, so replay would accept
        // it. Put the log back exactly as it was and sync the truncate
        // too — left unsynced, the rollback itself could be lost to a
        // crash and resurrect the refused batch. (The directory entry
        // is already durable, so an emptied new file is a harmless,
        // replay-inert leftover.) Best effort: if even this fails the
        // disk is failing twice over, and replay's torn-tail rule
        // still absorbs the common partial-append shape.
        let _ = file.set_len(length_before).and_then(|()| file.sync_all());
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
///
/// Records above the watermark are keyed by `seq`, and a repeated seq
/// keeps the LATER one. Appends are seq-monotonic, so no duplicate ever
/// occurs in normal operation — this is a backstop for the one shape
/// that can leak one: `append_batch` wrote a complete batch, its sync
/// failed, AND the rollback truncate failed too (a disk failing twice
/// over). That leftover is a complete, valid record the torn-tail rule
/// cannot catch; a retry (or the next write) reusing the same seq would
/// otherwise replay BESIDE it and double-apply. Later-wins drops the
/// unacknowledged leftover in favor of the write that actually followed.
pub fn replay<Op: DeserializeOwned + Serialize>(
    path: &Path,
    watermark: u64,
) -> io::Result<(Vec<Op>, u64)> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok((Vec::new(), watermark));
        }
        Err(error) => return Err(error),
    };
    let (ops, top, torn, _) = parse_log(path, &bytes, watermark)?;
    // A torn trailing record is the expected crash-mid-append shape. It
    // is already left out of `ops`; heal it off disk too, so the next
    // append does not write straight after the fragment and fuse a new
    // record onto it — turning a recoverable tear into the fatal
    // interior-corruption case. It was never acknowledged, so truncating
    // it loses nothing. Best effort: a failure only leaves the log as it
    // was, to be retried next replay.
    if let Some(torn_len) = torn {
        let healthy_len = bytes.len() as u64 - torn_len;
        tracing::warn!(
            "dropping a torn trailing WAL record at {} ({torn_len} bytes) — crash mid-append",
            path.display(),
        );
        if let Err(error) = truncate_to(path, healthy_len) {
            tracing::warn!(
                "could not heal torn WAL tail at {} (harmless, will retry next replay): {error}",
                path.display(),
            );
        }
    }
    Ok((ops, top))
}

/// Like [`replay`] but read-only: it heals nothing. Returns the applied
/// ops and top seq exactly as `replay` would, plus — when the log's
/// final record was torn by a crash mid-append — the byte size of that
/// torn fragment (`Some(bytes)`), so a diagnostic caller such as `taguru
/// inspect` can REPORT the tear rather than silently truncate it (a
/// clean log returns `None`) — plus how many intact records carried no
/// checksum (written before the field existed), so the same caller can
/// say how much of the log was actually VERIFIED rather than merely
/// parsed. The torn fragment is already excluded from the returned ops,
/// so what this reports is exactly what the server's next `replay`
/// would heal away.
pub fn replay_readonly<Op: DeserializeOwned + Serialize>(
    path: &Path,
    watermark: u64,
) -> io::Result<(Vec<Op>, u64, Option<u64>, usize)> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Ok((Vec::new(), watermark, None, 0));
        }
        Err(error) => return Err(error),
    };
    parse_log(path, &bytes, watermark)
}

/// The disk-free core of replay: split the bytes, set aside a torn
/// trailing fragment (returned as its byte length, `None` when the file
/// ends clean), and decode every intact record above `watermark` in seq
/// order — a repeated seq keeps the later record (see [`replay`] for the
/// double-fault that can leak one). Every record carrying a checksum is
/// verified against its canonical bytes — a mismatch is the same fatal
/// corruption an undecodable line is, because this file holds
/// acknowledged writes that exist nowhere else — and the count of
/// records carrying NONE (pre-checksum writers) rides along last, for
/// diagnostic callers. Writes nothing; whether to heal a torn tail is
/// left to the caller.
fn parse_log<Op: DeserializeOwned + Serialize>(
    path: &Path,
    bytes: &[u8],
    watermark: u64,
) -> io::Result<(Vec<Op>, u64, Option<u64>, usize)> {
    let mut segments: Vec<&[u8]> = bytes.split(|&byte| byte == b'\n').collect();
    // A complete file ends in '\n', making the final segment empty; a
    // torn file's final segment is the record a crash cut short. One
    // rule covers both: the last segment is never a whole record.
    let torn = match segments.pop() {
        Some(tail) if !tail.is_empty() => Some(tail.len() as u64),
        _ => None,
    };

    // Keyed by seq so a duplicate resolves to the later record, and
    // drained in seq order (== append order for the monotonic tail).
    let mut pending: BTreeMap<u64, Op> = BTreeMap::new();
    let mut top = watermark;
    let mut unchecked = 0usize;
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
        match record.crc {
            // Every checksummed record is verified, at or below the
            // watermark included: a watermark-covered record replays to
            // nothing, but its corruption still says the medium is
            // eating this file — the one thing structural parsing
            // cannot notice and the whole reason the field exists.
            Some(stored) => {
                let canonical =
                    canonical_record_bytes(record.seq, &record.op).map_err(io::Error::from)?;
                let computed = crate::crc32c::crc32c(&canonical);
                if computed != stored {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "corrupt WAL record at {} line {}: checksum mismatch (stored \
                             {stored:#010x}, computed {computed:#010x}) — the bytes changed \
                             after they were written",
                            path.display(),
                            index + 1
                        ),
                    ));
                }
            }
            None => unchecked += 1,
        }
        top = top.max(record.seq);
        if record.seq > watermark && pending.insert(record.seq, record.op).is_some() {
            tracing::warn!(
                "WAL {} carries a duplicate seq {} — keeping the later record \
                 (an earlier append's failed sync left an un-rolled-back batch)",
                path.display(),
                record.seq,
            );
        }
    }
    let ops = pending.into_values().collect();
    Ok((ops, top, torn, unchecked))
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
    #[cfg(test)]
    if let Some(error) = injected_truncate_failure() {
        return Err(error);
    }
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
    fn records_carry_a_checksum_spliced_exactly_as_serde_would_write_it() {
        let path = scratch_wal("crc-shape");
        append_batch(&path, 1, &[associate("a"), associate("b")]).unwrap();

        let bytes = fs::read(&path).unwrap();
        for line in bytes.split(|&b| b == b'\n').filter(|l| !l.is_empty()) {
            // Every record ends in the checksum field...
            let record: WalRecord<WalOp> = serde_json::from_slice(line).unwrap();
            let stored = record.crc.expect("a fresh record carries its checksum");
            // ...that covers the record's own crc-less serialization...
            let canonical = canonical_record_bytes(record.seq, &record.op).unwrap();
            assert_eq!(stored, crate::crc32c::crc32c(&canonical));
            // ...and the splice is byte-identical to letting serde
            // serialize `crc: Some(..)` itself — the property replay's
            // re-serialization check rests on.
            let full = serde_json::to_vec(&WalRecord {
                seq: record.seq,
                op: &record.op,
                crc: Some(stored),
            })
            .unwrap();
            assert_eq!(line, full);
        }

        let (ops, top) = replay::<WalOp>(&path, 0).unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(top, 2);
    }

    #[test]
    fn a_bitflip_that_stays_valid_json_fails_replay_as_corruption() {
        let path = scratch_wal("crc-flip");
        append_batch(&path, 1, &[associate("aaaa")]).unwrap();
        append_batch(&path, 2, &[associate("z")]).unwrap();

        // Silent corruption's exact shape: one byte changes, the line
        // still parses, the seq and structure are intact. Only the
        // checksum can tell — and it must refuse, not skip, because the
        // record is acknowledged data with no other copy.
        let text = String::from_utf8(fs::read(&path).unwrap()).unwrap();
        assert_eq!(text.matches("aaaa").count(), 1);
        fs::write(&path, text.replace("aaaa", "aaab")).unwrap();

        let error = replay::<WalOp>(&path, 0).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("checksum mismatch"), "{error}");
        assert!(error.to_string().contains("line 1"), "{error}");
    }

    #[test]
    fn pre_checksum_records_replay_unchecked_and_are_counted() {
        let path = scratch_wal("crc-legacy");
        // A record exactly as the pre-checksum writer serialized it.
        fs::write(
            &path,
            "{\"seq\":1,\"op\":\"associate\",\"subject\":\"a\",\"label\":\"好き\",\
             \"object\":\"りんご\",\"weight\":1.0}\n",
        )
        .unwrap();
        // A checksummed record appended by the current writer rides
        // beside it — an upgraded server's log mid-transition.
        append_batch(&path, 2, &[associate("b")]).unwrap();

        let (ops, top, torn, unchecked) = replay_readonly::<WalOp>(&path, 0).unwrap();
        assert_eq!(ops.iter().map(subject_of).collect::<Vec<_>>(), ["a", "b"]);
        assert_eq!(top, 2);
        assert_eq!(torn, None);
        assert_eq!(unchecked, 1, "exactly the legacy record goes unverified");

        // The healing replay accepts the same mix.
        let (ops, _) = replay::<WalOp>(&path, 0).unwrap();
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn a_downgraded_parser_shape_reads_a_checksummed_record() {
        // The compatibility promise in the WalRecord doc: a binary
        // whose record shape predates `crc` must keep replaying new
        // logs — serde routes the unknown field into the flattened op,
        // and the op's struct ignores it.
        #[derive(Deserialize)]
        struct PreChecksumRecord {
            seq: u64,
            #[serde(flatten)]
            op: WalOp,
        }

        let path = scratch_wal("crc-downgrade");
        append_batch(&path, 7, &[associate("a")]).unwrap();
        let bytes = fs::read(&path).unwrap();
        let line = bytes.split(|&b| b == b'\n').next().unwrap();
        let record: PreChecksumRecord = serde_json::from_slice(line).unwrap();
        assert_eq!(record.seq, 7);
        assert_eq!(subject_of(&record.op), "a");
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
    fn a_duplicate_seq_keeps_the_later_record_not_both() {
        // A complete batch whose sync then failed, with a rollback that
        // failed too, leaves a valid record on disk that the torn-tail
        // rule cannot catch. The write that follows reuses the same seq
        // (the caller never advanced past a refused batch). Replay must
        // apply the LATER record only — never both, which would
        // double-apply the seq and resurrect the refused write.
        let path = scratch_wal("dupseq");
        append_batch(&path, 1, &[associate("ghost")]).unwrap();
        // The retry reuses seq 1 rather than advancing.
        append_batch(&path, 1, &[associate("real")]).unwrap();

        let (ops, top) = replay(&path, 0).unwrap();
        assert_eq!(
            ops.iter().map(subject_of).collect::<Vec<_>>(),
            vec!["real"],
            "the later record at a reused seq wins; the ghost is dropped"
        );
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
    fn an_empty_batch_creates_no_file_and_appends_nothing() {
        let path = scratch_wal("empty");
        // Nothing to append: no bytes, and no file to create or sync.
        assert_eq!(append_batch::<WalOp>(&path, 1, &[]).unwrap(), 0);
        assert!(!path.exists(), "an empty batch must not create the log");

        // The next real append still creates and syncs the file itself,
        // and replays cleanly.
        append_batch(&path, 1, &[associate("a")]).unwrap();
        let (ops, top) = replay(&path, 0).unwrap();
        assert_eq!(ops.iter().map(subject_of).collect::<Vec<_>>(), vec!["a"]);
        assert_eq!(top, 1);
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
