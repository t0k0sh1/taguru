use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::hash::{FNV1A_OFFSET, fnv1a_fold};

/// Mints a fresh `context` id — a UUID v4 in its canonical hyphenated
/// lowercase form, which doubles as the file stem of the context's
/// whole family (ADR 0045: the id, not the name, addresses the files,
/// so a rename never touches them). Plain ASCII inside `[a-f0-9-]`,
/// so it needs no [`file_stem`] encoding and never collides with one:
/// an encoded stem cannot contain `-`-separated hex groups of this
/// shape by accident, and uniqueness comes from the 122 random bits.
pub(crate) fn mint_context_id() -> String {
    // The wire-fixture generator commits response bodies that carry
    // minted ids, so its ids must be identical run to run. This hook
    // is for THAT — a test-only sequence in the same canonical shape —
    // and for nothing else: two servers minting from it collide, and
    // a restart re-mints ids the data directory already holds.
    if std::env::var_os("TAGURU_TEST_DETERMINISTIC_IDS").is_some() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        return format!("00000000-0000-4000-8000-{n:012x}");
    }
    uuid::Uuid::new_v4().to_string()
}

/// A `context` id derived from `seed` instead of minted: the same seed
/// always yields the same canonical id. For the offline flows that must
/// name their one context in a header BEFORE any server state exists
/// (`taguru-code sync` renders its batches ahead of booting the data
/// directory) and need a re-run to find the context it made last time.
/// Shaped like a v4 UUID (the random-bytes layout, fed SHA-256 bytes
/// instead), so every reader of a context id accepts it unchanged.
pub(crate) fn derived_context_id(seed: &str) -> String {
    let hex = crate::sha256::sha256_hex(seed.as_bytes());
    let mut bytes = [0u8; 16];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16)
            .expect("sha256_hex is lowercase hex");
    }
    uuid::Builder::from_random_bytes(bytes)
        .into_uuid()
        .to_string()
}

/// Whether `value` is a `context` id in the one spelling the server
/// accepts — a UUID in its canonical hyphenated lowercase form, exactly
/// what [`mint_context_id`] produces. `Uuid::try_parse` alone also
/// takes uppercase, braces, URNs, and hyphen-less digits; two spellings
/// of one id would be two different map keys and two different file
/// stems, so the round trip through `to_string` must reproduce `value`.
/// Shared by the `/contexts/{id}` path extractor and the import
/// stream's `context_id` columns, so every entrance refuses the same
/// set of strings.
pub(crate) fn is_context_id(value: &str) -> bool {
    uuid::Uuid::try_parse(value).is_ok_and(|parsed| parsed.to_string() == value)
}

/// Encodes a display name as a file stem: bytes outside [A-Za-z0-9_-]
/// become %XX. Names may contain path separators or dots; encoding
/// them keeps every name inside its directory (no traversal) and
/// reversible. Nothing in the data directory is named this way any
/// more — `context` families and `group` files are stemmed by id (ADR
/// 0045) — but export still names its stream files after names until
/// #996 moves the `group` files to ids.
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

/// Decodes [`file_stem`]'s encoding back into a name — test-only, the
/// proof that the encoding is reversible (so two names never share a
/// stem).
#[cfg(test)]
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
/// names, below). A thin wrapper over [`crate::hash::fnv1a_fold`]
/// (issue #605): this used to re-declare the offset/prime constants
/// and hand-unroll the loop, a byte-identical duplicate of the shared
/// primitive.
fn fnv64(bytes: &[u8]) -> u64 {
    fnv1a_fold(FNV1A_OFFSET, bytes.iter().copied())
}

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

/// Both WAL lanes' on-disk sizes for `stem` — `0` for a lane with no
/// file yet, not an error: a missing log is the healthy common case
/// (a freshly created or never-written `context`), not one worth
/// failing a boot or a rename over. The boot scan and a rename's
/// re-register both seed a Cold entry with these so the residency
/// gauge sees a leftover log from the first scrape, not only after
/// the `context`'s first touch post-restart. Order is `(graph, passages)`
/// — the same order [`Entry::new`] takes `wal_bytes` /
/// `passages_wal_bytes` in.
pub(super) fn wal_lane_bytes(data_dir: &Path, stem: &str) -> (u64, u64) {
    let lane_bytes = |path: PathBuf| {
        fs::metadata(path)
            .map(|metadata| metadata.len())
            .unwrap_or(0)
    };
    (
        lane_bytes(wal_path(data_dir, stem)),
        lane_bytes(passages_wal_path(data_dir, stem)),
    )
}

/// The optional per-`context` schema document (ADR 0009 §5.1). Built with
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
/// schema, unlike a `group`, never falls back to an empty record after
/// setting the bytes aside).
pub(crate) fn schema_corrupt_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.schema.corrupt"))
}

/// The durable-deletion marker: while it exists, boot resumes the
/// unlinks (see `delete`/`scan_data_dir`). One builder so the writer,
/// the boot sweep, and the create-time cleanup can never disagree
/// about its name — a stale marker beside a freshly recreated `context`
/// would otherwise make the next boot delete the new `context`.
pub(crate) fn deleted_marker_path(dir: &Path, stem: &str) -> PathBuf {
    dir.join(format!("{stem}.deleted"))
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
/// file name only needs to be unique per (`context`, source) and safe,
/// which the hash gives without an encoding scheme. Stems contain no
/// dots, so the `{stem}.` prefix plus the extension identifies a
/// marker's `context` unambiguously.
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
/// which `context` — self-describing, so boot and inspect report the
/// human-readable pair without decoding file names.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ImportMarker {
    pub(crate) context: String,
    pub(crate) source: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `derived_context_id` is consumed only by taguru-code's sync (a
    /// module cargo-mutants does not cover), so its bytes are pinned here
    /// against the layout written out by hand: the first 16 SHA-256 bytes
    /// with the v4 version nibble and the RFC 4122 variant bits forced.
    /// A re-run of `taguru-code sync` finds last run's context only while
    /// these literals hold.
    #[test]
    fn derived_context_ids_are_pinned_deterministic_v4_shaped_and_distinct() {
        assert_eq!(
            derived_context_id("sake"),
            "cef2e28b-43f0-4b6c-8201-abab0785399f"
        );
        assert_eq!(
            derived_context_id("code"),
            "5694d08a-2e53-4fca-a0c3-103e5ad6f607"
        );
        assert_eq!(derived_context_id("sake"), derived_context_id("sake"));
        for seed in ["", "sake", "code", "酒蔵"] {
            assert!(is_context_id(&derived_context_id(seed)), "{seed:?}");
        }
    }

    /// One spelling per id: everything `Uuid::try_parse` accepts beyond
    /// the canonical lowercase hyphenated form is refused, as is a name.
    #[test]
    fn only_the_canonical_lowercase_hyphenated_uuid_is_a_context_id() {
        assert!(is_context_id("cef2e28b-43f0-4b6c-8201-abab0785399f"));
        assert!(!is_context_id("CEF2E28B-43F0-4B6C-8201-ABAB0785399F"));
        assert!(!is_context_id("cef2e28b43f04b6c8201abab0785399f"));
        assert!(!is_context_id("{cef2e28b-43f0-4b6c-8201-abab0785399f}"));
        assert!(!is_context_id(
            "urn:uuid:cef2e28b-43f0-4b6c-8201-abab0785399f"
        ));
        assert!(!is_context_id("sake"));
        assert!(!is_context_id(""));
    }
}
