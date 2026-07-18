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
#[cfg_attr(test, derive(PartialEq))]
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

#[cfg(test)]
impl From<crate::context_proptest::GeneratedWalOp> for WalOp {
    fn from(op: crate::context_proptest::GeneratedWalOp) -> Self {
        use crate::context_proptest::GeneratedWalOp;

        match op {
            GeneratedWalOp::Associate(op) => Self::Associate(crate::registry::AssocOp {
                subject: op.subject.to_string(),
                label: op.label.to_string(),
                object: op.object.to_string(),
                weight: op.weight,
                source: op.source.map(str::to_string),
                paragraph: op.paragraph,
            }),
            GeneratedWalOp::AliasConcept { alias, canonical } => Self::AliasConcept {
                alias: alias.to_string(),
                canonical: canonical.to_string(),
            },
            GeneratedWalOp::AliasLabel { alias, canonical } => Self::AliasLabel {
                alias: alias.to_string(),
                canonical: canonical.to_string(),
            },
            GeneratedWalOp::UnaliasConcept(alias) => Self::UnaliasConcept {
                alias: alias.to_string(),
            },
            GeneratedWalOp::UnaliasLabel(alias) => Self::UnaliasLabel {
                alias: alias.to_string(),
            },
            GeneratedWalOp::RetractSource(source) => Self::RetractSource {
                source: source.to_string(),
            },
            GeneratedWalOp::RetractAssociation {
                subject,
                label,
                object,
            } => Self::RetractAssociation {
                subject: subject.to_string(),
                label: label.to_string(),
                object: object.to_string(),
            },
        }
    }
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
    /// LAST field, spliced in by [`append_batch`]; replay strips this
    /// suffix back off the raw line and compares (see
    /// [`canonical_bytes_of_line`] — re-deriving the bytes by
    /// re-encoding the parsed op is NOT safe, see its doc comment). The
    /// record shape stays readable by a pre-checksum binary — its
    /// parser routes the unknown field into the flattened op, where
    /// serde ignores it — and records WITHOUT the field (written by
    /// one) replay unchecked, so the format change is additive in both
    /// directions. That same leniency would also swallow a record
    /// whose crc field survived corruption everywhere except its own
    /// name — serde treats an unrecognized field name exactly like a
    /// missing one — so every replay path additionally checks whether
    /// the raw bytes hold a field the deserialized op cannot account
    /// for before trusting a `None` here (see
    /// [`line_has_a_field_the_op_cannot_explain`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    crc: Option<u32>,
}

/// One record's canonical crc-less bytes, as [`append_batch`] first
/// produces them and checksums; the writer then splices `,"crc":N}` onto
/// exactly this. Not used to reconstruct those bytes on the read side —
/// see [`canonical_bytes_of_line`] for why re-encoding the parsed op
/// cannot stand in for the original serialization.
fn canonical_record_bytes<Op: Serialize>(seq: u64, op: Op) -> serde_json::Result<Vec<u8>> {
    serde_json::to_vec(&WalRecord { seq, op, crc: None })
}

/// Recovers a checksummed record's canonical crc-less bytes straight
/// from the line that was actually read off disk, by stripping the
/// `,"crc":N}` suffix [`append_batch`] spliced on and restoring the
/// closing brace — byte-for-byte identical to what the writer
/// checksummed, no matter what.
///
/// This does NOT re-derive the bytes by parsing the record and
/// re-encoding it through [`canonical_record_bytes`] — that would be
/// unsound: serde_json's float formatter does not always round-trip
/// bit-exact (a handful of `f64` values re-serialize one ULP off from
/// how they were first written), which would fail the checksum of a
/// perfectly intact record. Working from the original bytes sidesteps
/// the question entirely.
///
/// `None` only if `crc` was somehow absent from the line despite the
/// record parsing with `crc: Some(_)` moments earlier — a format
/// invariant violation (the field is always spliced in last), not
/// something a checksum could have caught anyway.
fn canonical_bytes_of_line(line: &[u8]) -> Option<Vec<u8>> {
    const CRC_FIELD: &[u8] = b",\"crc\":";
    let pos = line
        .windows(CRC_FIELD.len())
        .rposition(|window| window == CRC_FIELD)?;
    let mut canonical = line[..pos].to_vec();
    canonical.push(b'}');
    Some(canonical)
}

/// Whether a line, already parsed with `crc: None`, carries a top-level
/// field the parsed `op` cannot account for. Only meaningful to call in
/// that situation: serde's flatten treats ANY field name it doesn't
/// recognize as simply absent (see [`WalRecord::crc`]), so a record
/// whose crc field's own name got bit-flipped into something else — say
/// `"crd"` — parses exactly like a genuine pre-checksum record that
/// never had the field at all. Naively scanning the raw bytes for a
/// `,"crc":` marker cannot tell the two apart either: the corrupted
/// name isn't `"crc"` anymore, so the marker is gone from the bytes in
/// exactly the case that needs catching.
///
/// The two situations DO differ in one place: a real pre-checksum
/// record's bytes hold nothing but `seq` and the op's own fields, while
/// a corrupted-name record has one extra field sitting alongside them.
/// Re-serializing the already-deserialized `op` recovers exactly the
/// key set it explains — without ever needing to know `Op`'s field
/// names statically — and this only ever compares KEYS, never a value,
/// so serde_json's occasional one-ULP float round-trip drift (see
/// [`canonical_bytes_of_line`]'s doc comment) cannot produce a false
/// positive here the way comparing re-encoded bytes could.
fn line_has_a_field_the_op_cannot_explain<Op: Serialize>(line: &[u8], op: &Op) -> io::Result<bool> {
    let raw: serde_json::Value = serde_json::from_slice(line)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
    let Some(raw_fields) = raw.as_object() else {
        return Ok(false);
    };
    let op_value = serde_json::to_value(op)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
    let op_fields = op_value.as_object();
    Ok(raw_fields.keys().any(|key| {
        key != "seq" && key != "crc" && !op_fields.is_some_and(|fields| fields.contains_key(key))
    }))
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

/// [`fail_appends_after`]'s twin for [`append_missing_newline`]: after
/// `successes` more newline heals succeed normally, the one after them
/// fails with an injected error and the hook disarms.
#[cfg(test)]
pub(crate) fn fail_newline_heals_after(successes: u32) {
    NEWLINE_HEAL_FAULT.with(|cell| cell.set(Some(successes)));
}

#[cfg(test)]
thread_local! {
    static NEWLINE_HEAL_FAULT: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
fn injected_newline_heal_failure() -> Option<io::Error> {
    NEWLINE_HEAL_FAULT.with(|cell| match cell.get() {
        Some(0) => {
            cell.set(None);
            Some(io::Error::other("injected newline-heal failure"))
        }
        Some(remaining) => {
            cell.set(Some(remaining - 1));
            None
        }
        None => None,
    })
}

/// Arms the calling thread so that the next `count` directory-fsyncs
/// [`append_batch`] performs right after creating a new WAL file fail
/// with an injected error; calls beyond `count` succeed normally. Unlike
/// the `_after` faults above (one shot, armed by a success count), this
/// one needs to fail an exact number of times in a row so a test can
/// drive both the first failure AND a subsequent retry's failure
/// deterministically, without relying on real directory-permission
/// tricks to fail an `fsync` specifically.
#[cfg(test)]
pub(crate) fn fail_next_create_fsyncs(count: u32) {
    CREATE_FSYNC_FAULT.with(|cell| cell.set(count));
}

#[cfg(test)]
thread_local! {
    static CREATE_FSYNC_FAULT: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn injected_create_fsync_failure() -> Option<io::Error> {
    CREATE_FSYNC_FAULT.with(|cell| {
        let remaining = cell.get();
        if remaining == 0 {
            None
        } else {
            cell.set(remaining - 1);
            Some(io::Error::other("injected create-fsync failure"))
        }
    })
}

/// [`crate::storage::fsync_parent_dir`] for the directory-entry sync
/// [`append_batch`] performs right after creating a new WAL file —
/// test-fault-aware the same way [`injected_append_failure`] is for the
/// append itself, so a regression test can fail this specific sync (and
/// a retry of it) deterministically without also catching the syncs
/// `write_atomic` performs elsewhere through the same shared function.
fn fsync_parent_dir_after_create(path: &Path) -> io::Result<()> {
    #[cfg(test)]
    if let Some(error) = injected_create_fsync_failure() {
        return Err(error);
    }
    crate::storage::fsync_parent_dir(path)
}

/// The shape of a non-empty final WAL line missing its trailing `\n` —
/// both are the same crash (a `write_all` that completed followed by a
/// crash before the batch's `sync_all`), differing only in where the
/// crash landed relative to the record itself. Reported by
/// [`replay_readonly`] and [`parse_log`], and healed by [`replay`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TornTail {
    /// The crash landed inside the record — an incomplete fragment that
    /// never parses whole. It was never acknowledged (its writer never
    /// got `Ok`), so healing by truncating it away loses nothing.
    Discarded { bytes: u64 },
    /// The record itself finished; only its own trailing `\n` was lost.
    /// Already decoded and included among the returned ops — healing
    /// appends the missing byte rather than removing the record.
    Recovered { bytes: u64 },
}

/// Heals a [`TornTail::Recovered`] record in place by appending the
/// single `\n` byte a crash took between the record's own `write_all`
/// and the batch's `sync_all`. Without this, the next `append_batch`'s
/// `O_APPEND` write would start right after this record's last byte —
/// fusing the two into one line that decodes as neither, the same
/// fatal fusion [`replay`]'s doc comment describes for a `Discarded`
/// tail, guarded against here by completing the line instead of
/// removing it.
fn append_missing_newline(path: &Path) -> io::Result<()> {
    #[cfg(test)]
    if let Some(error) = injected_newline_heal_failure() {
        return Err(error);
    }
    if let Some(error) = crate::storage::injected_persistence_failure("WAL newline heal") {
        return Err(error);
    }
    let mut file = fs::OpenOptions::new().append(true).open(path)?;
    file.write_all(b"\n")?;
    file.sync_all()
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
    if let Some(error) = crate::storage::injected_persistence_failure("WAL append") {
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
        Err(error) => match error.kind() {
            io::ErrorKind::AlreadyExists => {
                (fs::OpenOptions::new().append(true).open(path)?, false)
            }
            _ => return Err(error),
        },
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
    if created && let Err(error) = fsync_parent_dir_after_create(path) {
        // The usual recovery is removing the file so the next append
        // recreates it and syncs the directory again. But if the
        // removal ITSELF fails, dropping the error here would leave an
        // unsynced file behind that the `AlreadyExists` path above will
        // then skip syncing forever — silently, since nothing surfaces
        // that failure to a caller. Retrying the sync instead gives the
        // directory entry a second chance to actually land: on success
        // the batch proceeds with a file that is now durable after all.
        if crate::storage::remove_persisted_file(path).is_err() {
            if let Err(retry_error) = fsync_parent_dir_after_create(path) {
                // The sync failed twice AND the cleanup failed once
                // already — a disk in serious trouble. Leaving the file
                // behind now would make the poisoning permanent: every
                // later append reaches it through the `AlreadyExists`
                // path above, which never syncs the directory, so this
                // would be the last chance ever to close the hole. Best
                // effort: if this ALSO fails there is nothing left to
                // try, and the (still unsynced) leftover survives, but
                // the original disk failure is exceedingly rare to
                // begin with, and rarer still to also make the cleanup
                // fail twice over.
                let _ = crate::storage::remove_persisted_file(path);
                return Err(retry_error);
            }
        } else {
            return Err(error);
        }
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
/// A trailing line without its `\n` is a crash mid-append, classified
/// by whether the record itself finished writing
/// ([`TornTail`]): an incomplete fragment ([`TornTail::Discarded`])
/// never happened as far as the log is concerned — its writer never
/// got `Ok` — and is dropped. A complete, checksum-valid record missing
/// only the delimiter itself ([`TornTail::Recovered`]) already counts
/// among the returned ops. Any OTHER undecodable line, or a trailing
/// record that decodes whole but fails its own checksum, is real
/// corruption: unlike the sidecars, this file holds acknowledged writes
/// that exist nowhere else, so skipping past it would be silent loss —
/// it errors instead.
///
/// Leaving either shape's bytes on disk is not enough on its own: the
/// next `append_batch` opens with `O_APPEND` and writes straight after
/// them. Whichever shape it is, the old bytes and the new record would
/// then share a line with no `\n` between them, and that fused line
/// decodes as neither — turning a recoverable tail into fatal interior
/// corruption. So both shapes are healed in place here: a `Discarded`
/// fragment is truncated away (it was never acknowledged, so this loses
/// nothing); a `Recovered` record gets its missing `\n` appended
/// instead (removing it would lose an acknowledged write). Unlike the
/// old best-effort truncate, a heal failure here is now fatal: returning
/// `Ok` with the tail still un-healed would silently hand the caller a
/// context one append away from bricking on fused corruption. The
/// caller (`ensure_hot`) already quarantines a context on `Err` rather
/// than crash the server, so refusing to heal is the safe failure mode.
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
        Err(error) => match error.kind() {
            io::ErrorKind::NotFound => return Ok((Vec::new(), watermark)),
            _ => return Err(error),
        },
    };
    let (ops, top, torn, _) = parse_log(path, &bytes, watermark)?;
    // Leaving either shape's bytes on disk risks the next append fusing
    // onto them (see the doc comment above), so both are healed here —
    // and, unlike the old best-effort truncate, a heal failure is now
    // propagated instead of shrugged off: a still-torn tail is exactly
    // one append away from bricking the context on fused corruption.
    match torn {
        Some(TornTail::Discarded { bytes: torn_len }) => {
            let healthy_len = bytes.len() as u64 - torn_len;
            tracing::warn!(
                "dropping a torn trailing WAL record at {} ({torn_len} bytes) — crash mid-append",
                path.display(),
            );
            truncate_to(path, healthy_len).map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!(
                        "could not heal torn WAL tail at {}: {error}",
                        path.display(),
                    ),
                )
            })?;
        }
        Some(TornTail::Recovered {
            bytes: recovered_len,
        }) => {
            tracing::warn!(
                "recovering a WAL record at {} missing only its trailing newline \
                 ({recovered_len} bytes) — crash after the write, before the delimiter's own fsync",
                path.display(),
            );
            append_missing_newline(path).map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!(
                        "could not heal a missing WAL newline at {}: {error}",
                        path.display(),
                    ),
                )
            })?;
        }
        None => {}
    }
    Ok((ops, top))
}

/// Like [`replay`] but read-only: it heals nothing. Returns the applied
/// ops and top seq exactly as `replay` would, plus — when the log's
/// final line was missing its trailing `\n` — which [`TornTail`] shape
/// it was, so a diagnostic caller such as `taguru inspect` can REPORT
/// the tear precisely rather than silently heal it (a clean log returns
/// `None`) — plus how many intact records carried no checksum (written
/// before the field existed), so the same caller can say how much of
/// the log was actually VERIFIED rather than merely parsed. A
/// `Discarded` fragment is already excluded from the returned ops; a
/// `Recovered` one is already included in them — either way, this
/// reports exactly what the server's next `replay` would do, without
/// doing it.
pub fn replay_readonly<Op: DeserializeOwned + Serialize>(
    path: &Path,
    watermark: u64,
) -> io::Result<(Vec<Op>, u64, Option<TornTail>, usize)> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => match error.kind() {
            io::ErrorKind::NotFound => return Ok((Vec::new(), watermark, None, 0)),
            _ => return Err(error),
        },
    };
    parse_log(path, &bytes, watermark)
}

/// Whether a final line missing its `\n` is actually a complete record
/// rather than a genuine fragment — the difference between "the crash
/// took only the trailing delimiter" (safe to keep) and "the crash
/// landed mid-record" (safe to discard). A record that parses whole but
/// carries a checksum that does NOT match is neither shape: that is
/// bytes changing after they were written, the same corruption an
/// interior checksum mismatch reports, so it is surfaced as an error
/// rather than silently classified either way — a genuinely torn write
/// never reaches a matching checksum.
fn tail_record_is_complete<Op: DeserializeOwned + Serialize>(tail: &[u8]) -> io::Result<bool> {
    let record: WalRecord<Op> = match serde_json::from_slice(tail) {
        Ok(record) => record,
        Err(_) => return Ok(false),
    };
    let Some(stored) = record.crc else {
        if line_has_a_field_the_op_cannot_explain(tail, &record.op)? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "a field the op does not account for is present in the raw bytes — \
                 the crc field's name is likely corrupted, not genuinely absent",
            ));
        }
        return Ok(true);
    };
    let canonical = canonical_bytes_of_line(tail).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "parsed a crc field but could not find it in the raw line",
        )
    })?;
    let computed = crate::crc32c::crc32c(&canonical);
    if computed != stored {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "checksum mismatch (stored {stored:#010x}, computed {computed:#010x}) — \
                 the bytes changed after they were written",
            ),
        ));
    }
    Ok(true)
}

/// The disk-free core of replay: split the bytes, classify a non-empty
/// trailing fragment missing its `\n` as a [`TornTail`] (`None` when
/// the file ends clean), and decode every intact record above
/// `watermark` in seq order — a repeated seq keeps the later record
/// (see [`replay`] for the double-fault that can leak one). A
/// `Recovered` tail — a complete, checksum-valid record missing only
/// its own delimiter — rejoins the very same decode loop as every other
/// line below, rather than being handled separately, so it is verified,
/// counted, and seq-deduplicated exactly like an interior record. Every
/// record carrying a checksum is verified against its canonical bytes —
/// a mismatch is the same fatal corruption an undecodable line is,
/// because this file holds acknowledged writes that exist nowhere else
/// — and the count of records carrying NONE (pre-checksum writers)
/// rides along last, for diagnostic callers. Writes nothing; whether
/// and how to heal a torn tail is left to the caller.
fn parse_log<Op: DeserializeOwned + Serialize>(
    path: &Path,
    bytes: &[u8],
    watermark: u64,
) -> io::Result<(Vec<Op>, u64, Option<TornTail>, usize)> {
    let mut segments: Vec<&[u8]> = bytes.split(|&byte| byte == b'\n').collect();
    // A complete file ends in '\n', making the final segment empty. A
    // non-empty final segment is missing its newline — decide below
    // whether that is a genuine fragment (never happened) or a complete
    // record that lost only its own delimiter (already happened, and
    // already belongs in the ops this function returns).
    let tail = segments.pop().filter(|segment| !segment.is_empty());
    let torn = match tail {
        Some(tail) => {
            // 1-indexed, matching the per-line loop below: `segments`
            // has already had this tail popped off, so its remaining
            // length is exactly the count of lines ahead of it.
            let line_number = segments.len() + 1;
            let complete = tail_record_is_complete::<Op>(tail).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "corrupt WAL record at {} line {line_number}: {error}",
                        path.display(),
                    ),
                )
            })?;
            if complete {
                segments.push(tail);
                Some(TornTail::Recovered {
                    bytes: tail.len() as u64,
                })
            } else {
                Some(TornTail::Discarded {
                    bytes: tail.len() as u64,
                })
            }
        }
        None => None,
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
                let canonical = canonical_bytes_of_line(line).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "corrupt WAL record at {} line {}: parsed a crc field but \
                             could not find it in the raw line",
                            path.display(),
                            index + 1
                        ),
                    )
                })?;
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
            None => {
                if line_has_a_field_the_op_cannot_explain(line, &record.op)? {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "corrupt WAL record at {} line {}: a field the op does not \
                             account for is present in the raw bytes — the crc field's \
                             name is likely corrupted, not genuinely absent",
                            path.display(),
                            index + 1
                        ),
                    ));
                }
                unchecked += 1;
            }
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
        Err(error) => match error.kind() {
            io::ErrorKind::NotFound => Ok(()),
            _ => Err(error),
        },
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
    if let Some(error) = crate::storage::injected_persistence_failure("WAL truncate") {
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
            // serialize `crc: Some(..)` itself.
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
    fn a_bitflip_in_an_interior_crc_field_name_is_not_silently_treated_as_pre_checksum() {
        // A distinct corruption shape from the sibling test above: the
        // flip lands on the FIELD NAME rather than the op's data. Serde's
        // flatten treats any field name it doesn't recognize as simply
        // absent, so `record.crc` silently becomes `None` and the record
        // looks exactly like a genuine pre-checksum write — without the
        // raw-byte marker scan, this corruption would be counted as
        // unchecked instead of rejected.
        let path = scratch_wal("crc-field-name-flip-interior");
        append_batch(&path, 1, &[associate("aaaa")]).unwrap();
        append_batch(&path, 2, &[associate("z")]).unwrap();

        let text = String::from_utf8(fs::read(&path).unwrap()).unwrap();
        assert_eq!(text.matches("\"crc\":").count(), 2);
        let flipped = text.replacen("\"crc\":", "\"crd\":", 1);
        fs::write(&path, flipped).unwrap();

        let error = replay::<WalOp>(&path, 0).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(
            error
                .to_string()
                .contains("field's name is likely corrupted"),
            "{error}"
        );
        assert!(error.to_string().contains("line 1"), "{error}");
    }

    #[test]
    fn a_weight_serde_json_cannot_parse_bit_exact_still_replays_clean() {
        // serde_json (1.0.150, pinned) mis-rounds this exact literal by
        // 1 ULP on the PARSE side (`f64::from_str` gets it right —
        // confirmed as a serde_json bug, not this crate's). Before
        // `canonical_bytes_of_line` existed, replay re-derived its
        // checksum input by re-encoding the freshly reparsed op, so
        // that drift changed the bytes being hashed and a record that
        // landed on disk untouched failed its own checksum — a false
        // "corrupt" on data nothing had corrupted. Pinning the exact
        // value here catches a regression in either direction: this
        // crate reintroducing a re-encode, or (harmlessly) serde_json
        // fixing the parse bug and no longer reproducing it.
        const ULP_DRIFT_WEIGHT: f64 = -434820.72978759644;
        assert_ne!(
            serde_json::from_str::<f64>(&serde_json::to_string(&ULP_DRIFT_WEIGHT).unwrap())
                .unwrap(),
            ULP_DRIFT_WEIGHT,
            "this test's premise is a value serde_json does not parse back bit-exact; \
             if this now holds, serde_json fixed the bug this test guards against"
        );

        let path = scratch_wal("crc-serde-json-ulp");
        append_batch(
            &path,
            1,
            &[WalOp::Associate(AssocOp {
                subject: "りんご".to_string(),
                label: "りんご".to_string(),
                object: "りんご".to_string(),
                weight: ULP_DRIFT_WEIGHT,
                source: None,
                paragraph: None,
            })],
        )
        .unwrap();

        let (ops, top) = replay::<WalOp>(&path, 0).unwrap();
        assert_eq!(top, 1);
        assert_eq!(
            ops.len(),
            1,
            "an intact record must not fail its own checksum"
        );
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
    fn a_tail_missing_only_its_newline_is_recovered_not_discarded() {
        let path = scratch_wal("recovered");
        append_batch(&path, 1, &[associate("a")]).unwrap();
        let len_before_b = fs::metadata(&path).unwrap().len();
        append_batch(&path, 2, &[associate("b")]).unwrap();
        let full_len = fs::metadata(&path).unwrap().len();
        // The tail's length once its own trailing '\n' is gone.
        let tail_len = full_len - len_before_b - 1;

        // Simulate a crash after "b"'s own write_all landed but before
        // the batch's sync_all reached its trailing '\n': strip exactly
        // that one byte, leaving a complete, checksum-valid record.
        let mut bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.pop(), Some(b'\n'));
        fs::write(&path, &bytes).unwrap();

        let (ops, top, torn, unchecked) = replay_readonly::<WalOp>(&path, 0).unwrap();
        assert_eq!(
            ops.iter().map(subject_of).collect::<Vec<_>>(),
            vec!["a", "b"],
            "the newline-less record must already be included, not dropped"
        );
        assert_eq!(top, 2);
        assert_eq!(unchecked, 0);
        assert_eq!(torn, Some(TornTail::Recovered { bytes: tail_len }));

        // The healing replay recovers the same ops and repairs the file
        // back to a clean, newline-terminated state instead of
        // truncating the record away.
        let (ops, top) = replay::<WalOp>(&path, 0).unwrap();
        assert_eq!(
            ops.iter().map(subject_of).collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        assert_eq!(top, 2);
        assert_eq!(
            fs::metadata(&path).unwrap().len(),
            full_len,
            "healing must restore the missing newline, not truncate the record away"
        );
    }

    #[test]
    fn a_recovered_tail_does_not_fuse_with_the_next_append() {
        // The Recovered twin of `a_torn_tail_does_not_fuse_with_the_next_append`:
        // healing must APPEND the missing newline, not merely accept the
        // record in memory and leave the file as found — otherwise the
        // next O_APPEND write lands on this record's last byte and fuses
        // into one undecodable line, same as an un-truncated Discarded
        // fragment would.
        let path = scratch_wal("recovered-fuse");
        append_batch(&path, 1, &[associate("a")]).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.pop(), Some(b'\n'));
        fs::write(&path, &bytes).unwrap();

        let (ops, top) = replay(&path, 0).unwrap();
        assert_eq!(ops.iter().map(subject_of).collect::<Vec<_>>(), vec!["a"]);
        assert_eq!(top, 1);

        append_batch(&path, top + 1, &[associate("c")]).unwrap();

        let (ops, top) = replay(&path, 0).unwrap();
        assert_eq!(
            ops.iter().map(subject_of).collect::<Vec<_>>(),
            vec!["a", "c"],
            "healing must not leave the two records fused onto one line"
        );
        assert_eq!(top, 2);
    }

    #[test]
    fn a_tail_with_a_bad_checksum_is_fatal_not_torn() {
        // A final line that parses whole but fails its own checksum is
        // neither shape: a genuinely torn write never reaches a complete
        // record with a mismatched checksum, so this is corruption after
        // the fact, not a crash artifact — it must be fatal like any
        // other checksum mismatch, never silently classified as torn or
        // recovered.
        let path = scratch_wal("tail-bad-crc");
        append_batch(&path, 1, &[associate("aaaa")]).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.pop(), Some(b'\n'));
        let text = String::from_utf8(bytes).unwrap();
        assert_eq!(text.matches("aaaa").count(), 1);
        fs::write(&path, text.replace("aaaa", "aaab")).unwrap();

        let error = replay::<WalOp>(&path, 0).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("checksum mismatch"), "{error}");
    }

    #[test]
    fn a_bitflip_in_the_tail_crc_field_name_is_not_silently_treated_as_pre_checksum() {
        // The tail-specific counterpart to
        // `a_bitflip_in_an_interior_crc_field_name_is_not_silently_treated_as_pre_checksum`:
        // a final line missing its newline goes through
        // `tail_record_is_complete` instead of the interior loop, a
        // separate code path that needs the same raw-byte guard. Without
        // it, this corruption parses as `crc: None`, is indistinguishable
        // from a genuine pre-checksum tail, and is recovered as if intact.
        let path = scratch_wal("crc-field-name-flip-tail");
        append_batch(&path, 1, &[associate("aaaa")]).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.pop(), Some(b'\n'));
        let text = String::from_utf8(bytes).unwrap();
        assert_eq!(text.matches("\"crc\":").count(), 1);
        let flipped = text.replace("\"crc\":", "\"crd\":");
        fs::write(&path, flipped).unwrap();

        let error = replay::<WalOp>(&path, 0).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(
            error
                .to_string()
                .contains("field's name is likely corrupted"),
            "{error}"
        );
    }

    #[test]
    fn a_legacy_tail_missing_its_newline_is_recovered_and_counted_unchecked() {
        // A pre-checksum record (no crc field) missing only its
        // trailing newline is still a complete, parseable record —
        // recovered and counted unchecked, exactly like a legacy record
        // found anywhere else in the log.
        let path = scratch_wal("legacy-recovered");
        fs::write(
            &path,
            "{\"seq\":1,\"op\":\"associate\",\"subject\":\"a\",\"label\":\"好き\",\
             \"object\":\"りんご\",\"weight\":1.0}",
        )
        .unwrap();

        let (ops, top, torn, unchecked) = replay_readonly::<WalOp>(&path, 0).unwrap();
        assert_eq!(ops.iter().map(subject_of).collect::<Vec<_>>(), vec!["a"]);
        assert_eq!(top, 1);
        assert_eq!(
            unchecked, 1,
            "the recovered legacy record has no checksum to verify"
        );
        assert!(matches!(torn, Some(TornTail::Recovered { .. })));

        let (ops, top) = replay::<WalOp>(&path, 0).unwrap();
        assert_eq!(ops.iter().map(subject_of).collect::<Vec<_>>(), vec!["a"]);
        assert_eq!(top, 1);
        assert_eq!(
            fs::read(&path).unwrap().last(),
            Some(&b'\n'),
            "healing must append the missing newline"
        );
    }

    #[test]
    fn a_truncate_heal_failure_is_fatal_not_a_shrug() {
        // Before this fix, a failed heal only logged a second warning
        // and returned Ok with the tail still on disk — one append away
        // from fusing into fatal interior corruption. It must now
        // refuse instead of shrugging.
        let path = scratch_wal("truncate-heal-fails");
        append_batch(&path, 1, &[associate("a")]).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        bytes.extend_from_slice(br#"{"seq":2,"op":"associate","subject":"b"#);
        fs::write(&path, &bytes).unwrap();
        let torn_len = bytes.len() as u64;

        fail_truncates_after(0);
        let error = replay::<WalOp>(&path, 0).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert!(
            error.to_string().contains("injected truncate failure"),
            "{error}"
        );
        // The failed heal must not leave the file in some in-between
        // state: nothing was truncated, so the torn bytes are untouched.
        assert_eq!(fs::metadata(&path).unwrap().len(), torn_len);
    }

    #[test]
    fn a_newline_heal_failure_is_fatal_not_a_shrug() {
        let path = scratch_wal("newline-heal-fails");
        append_batch(&path, 1, &[associate("a")]).unwrap();
        let mut bytes = fs::read(&path).unwrap();
        assert_eq!(bytes.pop(), Some(b'\n'));
        let stripped_len = bytes.len() as u64;
        fs::write(&path, &bytes).unwrap();

        fail_newline_heals_after(0);
        let error = replay::<WalOp>(&path, 0).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert!(
            error.to_string().contains("injected newline-heal failure"),
            "{error}"
        );
        // The failed heal must not leave the file in some in-between
        // state: the missing newline was never appended.
        assert_eq!(fs::metadata(&path).unwrap().len(), stripped_len);
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
    fn a_missing_file_replays_readonly_to_nothing() {
        let path = scratch_wal("missing-readonly");
        let (ops, top, torn, unchecked) = replay_readonly::<WalOp>(&path, 5).unwrap();
        assert!(ops.is_empty());
        assert_eq!(top, 5);
        assert_eq!(torn, None);
        assert_eq!(unchecked, 0);
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
    fn a_failed_create_fsync_retries_after_a_failed_cleanup_and_succeeds() {
        // Before this fix, a failed directory fsync always tried to
        // remove the just-created file and gave up on `Err`, ignoring
        // whether the removal itself worked. If it didn't, the file
        // stayed on disk unsynced forever: the next append finds it via
        // the `AlreadyExists` path, which never syncs the directory
        // again. The fix retries the sync when cleanup fails — this
        // test proves that retry actually happens and, on success, the
        // batch completes rather than surfacing the original error.
        let path = scratch_wal("create-fsync-retry-succeeds");

        fail_next_create_fsyncs(1);
        // The registry-tracked op sequence this call reaches is: the
        // "WAL append" check (succeeds — decrements the arm), then
        // `remove_persisted_file`'s "unlink" check (fails — the arm was
        // at 0). The directory fsync itself never touches this counter,
        // so its failure/retry is driven entirely by `fail_next_create_fsyncs`.
        crate::storage::fail_persistence_ops_after(1);
        let result = append_batch(&path, 1, &[associate("a")]);
        let past_end = crate::storage::clear_persistence_fault();

        result.unwrap();
        assert!(
            !past_end,
            "the unlink fault must have fired for this test to be meaningful"
        );
        let (ops, top) = replay(&path, 0).unwrap();
        assert_eq!(ops.iter().map(subject_of).collect::<Vec<_>>(), vec!["a"]);
        assert_eq!(top, 1);
    }

    #[test]
    fn a_failed_create_fsync_surfaces_an_error_when_cleanup_and_retry_both_fail() {
        // The other half of the fix: when the retry ALSO fails, this
        // must still return an error rather than silently proceeding —
        // best effort does not paper over a disk that keeps refusing to
        // sync. But it must not leave the file behind either: that
        // leftover would be a permanent, silent durability hole, since
        // every later append reaches it through the `AlreadyExists`
        // path above, which never syncs the directory again.
        let path = scratch_wal("create-fsync-retry-fails");

        fail_next_create_fsyncs(2);
        crate::storage::fail_persistence_ops_after(1);
        let result = append_batch(&path, 1, &[associate("a")]);
        let past_end = crate::storage::clear_persistence_fault();

        let error = result.unwrap_err();
        assert!(
            error.to_string().contains("injected create-fsync failure"),
            "{error}"
        );
        assert!(
            !past_end,
            "the unlink fault must have fired for this test to be meaningful"
        );
        // The first cleanup attempt was intercepted by the injected
        // unlink fault (one-shot, already spent), so this second
        // attempt reaches the real filesystem unobstructed and actually
        // removes the file — the next append starts completely fresh
        // and gets an honest chance to sync the directory again, rather
        // than silently skipping it forever via `AlreadyExists`.
        assert!(!path.exists());
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

    mod proptests {
        use super::*;
        use crate::context_proptest::wal_op_strategy as generated_op_strategy;
        use proptest::prelude::*;

        fn shared_wal_op_strategy() -> impl Strategy<Value = WalOp> {
            generated_op_strategy().prop_map(WalOp::from)
        }

        fn proptest_config() -> ProptestConfig {
            let cases = std::env::var("PROPTEST_CASES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32);
            ProptestConfig {
                cases,
                ..ProptestConfig::default()
            }
        }

        fn text_strategy() -> impl Strategy<Value = String> {
            prop_oneof![
                Just("りんご"),
                Just("バナナ"),
                Just("好き"),
                Just("嫌い"),
                Just("alice"),
                Just("bob"),
                Just("docs/a.md"),
                Just("docs/b.md"),
            ]
            .prop_map(|s| s.to_string())
        }

        /// JSON has no NaN/Infinity literal, so a real record can never
        /// carry one — restricting to finite values keeps the generator
        /// inside the format's actual domain.
        ///
        /// Also filtered to values `serde_json` itself parses back
        /// bit-exact: the pinned `serde_json` (1.0.150) mis-rounds a
        /// minority of 17-significant-digit decimals by 1 ULP on the
        /// PARSE side (confirmed against `f64::from_str`, which gets
        /// them right). That no longer affects `parse_log`'s CHECKSUM
        /// (which now verifies the raw line straight off disk — see
        /// `canonical_bytes_of_line` — rather than re-encoding the
        /// parsed op), but the OP VALUE this test asserts equality on
        /// is still whatever `serde_json::from_slice` handed back, so a
        /// third-party parse rounding difference would still fail this
        /// property. Filtering it out keeps this property about
        /// `parse_log`'s own plumbing rather than `serde_json`'s float
        /// parser.
        fn finite_weight_strategy() -> impl Strategy<Value = f64> {
            (-1.0e6f64..1.0e6f64).prop_filter("value must survive a JSON round trip", |&w| {
                serde_json::from_str::<f64>(&serde_json::to_string(&w).unwrap()).unwrap() == w
            })
        }

        fn assoc_op_strategy() -> impl Strategy<Value = AssocOp> {
            (
                text_strategy(),
                text_strategy(),
                text_strategy(),
                finite_weight_strategy(),
                prop::option::of(text_strategy()),
                prop::option::of(any::<u32>()),
            )
                .prop_map(|(subject, label, object, weight, source, paragraph)| {
                    AssocOp {
                        subject,
                        label,
                        object,
                        weight,
                        source,
                        paragraph,
                    }
                })
        }

        fn wal_op_strategy() -> impl Strategy<Value = WalOp> {
            prop_oneof![
                assoc_op_strategy().prop_map(WalOp::Associate),
                (text_strategy(), text_strategy())
                    .prop_map(|(alias, canonical)| WalOp::AliasConcept { alias, canonical }),
                (text_strategy(), text_strategy())
                    .prop_map(|(alias, canonical)| WalOp::AliasLabel { alias, canonical }),
                text_strategy().prop_map(|alias| WalOp::UnaliasConcept { alias }),
                text_strategy().prop_map(|alias| WalOp::UnaliasLabel { alias }),
                text_strategy().prop_map(|source| WalOp::RetractSource { source }),
                (text_strategy(), text_strategy(), text_strategy()).prop_map(
                    |(subject, label, object)| WalOp::RetractAssociation {
                        subject,
                        label,
                        object,
                    }
                ),
            ]
        }

        /// The disk-free encode side of what `append_batch` writes: same
        /// canonical-bytes-then-splice-crc shape, built in memory so the
        /// property stays disk-free too.
        fn encode_batch(first_seq: u64, ops: &[WalOp]) -> Vec<u8> {
            let mut buffer = Vec::new();
            for (offset, op) in ops.iter().enumerate() {
                let record = canonical_record_bytes(first_seq + offset as u64, op).unwrap();
                let crc = crate::crc32c::crc32c(&record);
                buffer.extend_from_slice(&record[..record.len() - 1]);
                buffer.extend_from_slice(format!(",\"crc\":{crc}}}\n").as_bytes());
            }
            buffer
        }

        proptest! {
            #![proptest_config(proptest_config())]

            #[test]
            fn parse_log_round_trips_a_clean_batch(
                ops in prop::collection::vec(wal_op_strategy(), 0..8),
                first_seq in 1u64..1000,
            ) {
                let bytes = encode_batch(first_seq, &ops);
                let (parsed, top, torn, unchecked) =
                    parse_log::<WalOp>(Path::new("scratch.wal.jsonl"), &bytes, 0).unwrap();
                prop_assert_eq!(parsed, ops.clone());
                prop_assert_eq!(unchecked, 0);
                prop_assert!(torn.is_none());
                let expected_top = ops.len() as u64 + first_seq - 1;
                prop_assert_eq!(top, if ops.is_empty() { 0 } else { expected_top });
            }

            #[test]
            fn parse_log_never_panics_on_arbitrary_bytes(
                bytes in prop::collection::vec(any::<u8>(), 0..512),
                watermark in any::<u64>(),
            ) {
                let _ = parse_log::<WalOp>(Path::new("scratch.wal.jsonl"), &bytes, watermark);
            }

            #[test]
            fn parse_log_never_panics_on_mutated_valid_bytes(
                ops in prop::collection::vec(wal_op_strategy(), 1..8),
                first_seq in 1u64..1000,
                mutations in prop::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 0..16),
            ) {
                let mut bytes = encode_batch(first_seq, &ops);
                for (pick, value) in mutations {
                    *pick.get_mut(&mut bytes) = value;
                }
                let _ = parse_log::<WalOp>(Path::new("scratch.wal.jsonl"), &bytes, 0);
            }

            #[test]
            fn corrupting_any_interior_record_is_fatal(
                ops in prop::collection::vec(shared_wal_op_strategy(), 2..12),
                record_pick in any::<prop::sample::Index>(),
                cut_pick in any::<prop::sample::Index>(),
            ) {
                let path = scratch_wal("property-interior-corruption");
                append_batch(&path, 1, &ops).unwrap();
                let mut bytes = fs::read(&path).unwrap();
                let line_starts: Vec<_> = std::iter::once(0)
                    .chain(
                        bytes
                            .iter()
                            .enumerate()
                            .filter(|(_, byte)| **byte == b'\n')
                            .map(|(index, _)| index + 1),
                    )
                    .collect();
                let record = record_pick.index(ops.len() - 1);
                let start = line_starts[record];
                let end = line_starts[record + 1] - 1;
                let cut = cut_pick.index(end - start);
                // Keep the newline and every later record, but replace the
                // selected interior record with an arbitrary strict prefix.
                // The identical shape at EOF is a healable torn tail; here it
                // is acknowledged data and must be fatal.
                bytes.drain(start + cut..end);
                fs::write(&path, bytes).unwrap();

                let error = replay::<WalOp>(&path, 0).unwrap_err();
                prop_assert_eq!(error.kind(), io::ErrorKind::InvalidData);
            }

            #[test]
            fn a_generated_duplicate_sequence_number_keeps_the_later_op(
                mut ops in prop::collection::vec(shared_wal_op_strategy(), 1..12),
                replacement in shared_wal_op_strategy(),
                duplicate_pick in any::<prop::sample::Index>(),
            ) {
                let path = scratch_wal("property-duplicate-seq");
                append_batch(&path, 1, &ops).unwrap();
                let duplicate = duplicate_pick.index(ops.len());
                append_batch(&path, duplicate as u64 + 1, std::slice::from_ref(&replacement))
                    .unwrap();

                ops[duplicate] = replacement;
                let (replayed, top) = replay::<WalOp>(&path, 0).unwrap();
                prop_assert_eq!(replayed, ops.clone());
                prop_assert_eq!(top, ops.len() as u64);
            }
        }
    }
}
