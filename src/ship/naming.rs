//! Object naming, the local replication record, and the manifest
//! format: everything about identifying and describing what ships,
//! without touching the store itself.

use super::*;

pub(super) const FENCE_PREFIX: &str = "fence";
pub(super) const COMPLETE_MARKER: &str = "complete";
pub(crate) const HEARTBEAT_MARKER: &str = "heartbeat";
pub(crate) const RETIRED_MARKER: &str = "retired";

/// How often a live shipper refreshes `gen-{N}/heartbeat`. The object's
/// `last_modified` is what a later claimant's takeover guard reads to
/// ask "is this generation's writer still alive?" — pure ergonomics,
/// not correctness (the fence stays the only mutual-exclusion
/// primitive), so a coarse cadence is fine and keeps the idle-time
/// PUT rate negligible.
pub(super) const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(60);

/// How recently a generation must have shown life (its fence claim or
/// its last heartbeat) for the takeover guard to demand explicit
/// intent. Wider than the heartbeat cadence by 5× so a paused-but-live
/// writer (GC, a stalled node) is not casually deposed at the first
/// missed beat.
pub(crate) const TAKEOVER_GRACE: Duration = Duration::from_secs(300);

/// The local file remembering this data directory's relationship with
/// the bucket: which generation its writer last claimed, and — when
/// the directory was materialized FROM the bucket — which generation
/// it hydrated from (the cache-mode marker `crate::hydrate` keys on).
/// Never shipped (see [`classify`]): it describes the local replica of
/// the relationship, not the data.
pub(crate) const REPLICATION_RECORD: &str = ".taguru.replication";

/// What [`REPLICATION_RECORD`] holds.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct ReplicationRecord {
    /// The bucket URL the record speaks about — a directory re-pointed
    /// at a different bucket must not inherit the old bucket's
    /// generation arithmetic.
    pub(crate) url: String,
    /// The generation this directory's writer last claimed; `None`
    /// between a hydration creating the directory and its first claim.
    pub(crate) claimed_generation: Option<u64>,
    /// Present iff the directory began life as a materialization of
    /// the bucket — from this generation. A directory carrying this is
    /// a CACHE of the bucket lineage: any boot that cannot prove it is
    /// the lineage's own newest writer re-verifies local files against
    /// the bucket instead of trusting them (see `crate::hydrate`).
    #[serde(default)]
    pub(crate) hydrated_from: Option<u64>,
}

/// Reads the replication record, distinguishing "never written"
/// (`Ok(None)` — the pre-#128 posture, a normal state) from "written
/// but unreadable" (`Err`). The distinction is load-bearing: this
/// record is what marks a directory as a CACHE of the bucket lineage,
/// so treating a corrupt record as absent would boot a possibly
/// half-hydrated cache as independent local truth — every context not
/// yet localized would silently vanish from the registry, and the
/// claim that follows would fork the lineage. Boot paths refuse the
/// error; best-effort readers degrade explicitly at their call site.
pub(crate) fn read_replication_record(data_dir: &FsPath) -> io::Result<Option<ReplicationRecord>> {
    let path = data_dir.join(REPLICATION_RECORD);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(io::Error::new(
                error.kind(),
                format!("reading {}: {error}", path.display()),
            ));
        }
    };
    serde_json::from_slice(&bytes).map(Some).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{} exists but cannot be parsed ({error}) — this directory may be a \
                 cache of the bucket lineage, and guessing would either hide every \
                 not-yet-hydrated context or fork the lineage. Restore the record, or \
                 delete it to declare local disk the independent truth",
                path.display()
            ),
        )
    })
}

pub(crate) fn write_replication_record(
    data_dir: &FsPath,
    record: &ReplicationRecord,
) -> io::Result<()> {
    crate::storage::write_atomic(
        &data_dir.join(REPLICATION_RECORD),
        &serde_json::to_vec(record).expect("no unserializable field"),
    )
}

/// What `gen-{N}/complete` holds since issue #128: the exact shipped
/// state, so a reader can verify every downloaded byte and decide
/// whether a LOCAL file already matches without downloading anything.
/// The marker's existence still means what it always did — this
/// generation restores whole — but an empty (pre-manifest) marker now
/// names a generation from before #128, which `restore_into` refuses
/// rather than reconstructing by listing.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct Manifest {
    pub(crate) generation: u64,
    /// Published files: name → exactly the bytes last uploaded.
    pub(crate) files: BTreeMap<String, ManifestFile>,
    /// Log lanes: name → the newest series and its concatenated
    /// extent (every segment in order, as one byte string).
    pub(crate) lanes: BTreeMap<String, ManifestLane>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct ManifestFile {
    pub(crate) len: u64,
    pub(crate) crc: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct ManifestLane {
    pub(crate) series: u64,
    pub(crate) segments: u64,
    /// Length and CRC-32C of the series' segments concatenated in
    /// order — the same prefix arithmetic the shipping cursor keeps,
    /// so a local log whose prefix matches IS the shipped stream.
    pub(crate) len: u64,
    pub(crate) crc: u32,
    /// Highest record seq shipped in the series.
    pub(crate) seq: u64,
}

/// Fixed-width decimal so lexicographic object listing IS numeric
/// ordering — restore sorts names, never parses to sort.
pub(crate) fn fence_key(root: &StorePath, generation: u64) -> StorePath {
    root.clone()
        .join(FENCE_PREFIX)
        .join(format!("{generation:020}"))
}

pub(crate) fn gen_root(root: &StorePath, generation: u64) -> StorePath {
    root.clone().join(format!("gen-{generation:020}"))
}

/// The generation's `complete` key — the manifest object a replica
/// polls (its store-clock `last_modified` is the cheap "anything
/// new?" probe).
pub(crate) fn complete_key(generation_root: &StorePath) -> StorePath {
    generation_root.clone().join(COMPLETE_MARKER)
}

pub(crate) fn segment_name(series: u64, seg: u64) -> String {
    format!("{series:010}-{seg:010}.jsonl")
}

/// What a fence object says. The claim is the object's existence — the
/// body exists for operators (`who took the bucket from me?`) and as
/// the future home of lease fields (a TTL), so automation could be
/// layered on without changing the medium. Liveness already lives
/// beside it as `gen-{N}/heartbeat`, feeding the takeover guard
/// (`crate::hydrate`) — ergonomics, deliberately not lease semantics.
#[derive(serde::Serialize, serde::Deserialize)]
pub(super) struct FenceBody {
    pub(super) generation: u64,
    pub(super) holder: String,
    pub(super) claimed_at_epoch_secs: u64,
}

/// The names a directory scan sorts every entry into. Only shapes the
/// shipper understands leave the machine: an unknown file would be
/// shipped as opaque bytes just fine, but restore fidelity is easier
/// to reason about when the shipped set is exactly the family the
/// server itself reads, and staging litter (`*.tmp{N}`) must never
/// ship at all — those names are mid-write by definition.
pub(super) fn classify(name: &str) -> EntryKind {
    if name == ".taguru.lock" || name == REPLICATION_RECORD {
        return EntryKind::Skip;
    }
    // `staging_path` builds `{final}.tmp{pid}-{nonce}` names; matching
    // the exact shape through storage's own predicate (not just ".tmp")
    // keeps a hypothetical user file named exactly "x.tmp" shippable
    // while excluding every stager. The previous inline check here
    // required all-digits after "tmp" and so MISSED the hyphenated
    // names `staging_path` actually builds: a flush racing a ship cycle
    // put `{stem}.tmp{pid}-{nonce}` into the manifest, the rename then
    // dropped the object from the next cycle, and a replica hydrating
    // from the stale manifest refused to boot on the vanished object.
    if let Some(extension) = FsPath::new(name).extension().and_then(|e| e.to_str())
        && crate::storage::is_staging_extension(extension)
    {
        return EntryKind::Skip;
    }
    if name.ends_with(".wal.jsonl") {
        return EntryKind::LogLane;
    }
    EntryKind::Published
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum EntryKind {
    Published,
    LogLane,
    Skip,
}

/// The lane's parent snapshot: the file whose watermark makes a series
/// restart safe to restore from (uploaded BEFORE the new series — see
/// the module doc). The passage suffix is checked first because both
/// lanes end in `.wal.jsonl`.
pub(super) fn parent_snapshot_of(lane_name: &str) -> Option<String> {
    if let Some(stem) = lane_name.strip_suffix(".passages.wal.jsonl") {
        return Some(format!("{stem}.passages.bin"));
    }
    lane_name
        .strip_suffix(".wal.jsonl")
        .map(|stem| format!("{stem}.ctx"))
}

/// The per-lane label pair the lag metric carries: the context's
/// decoded name where the stem decodes (it always should — these files
/// were written by the server), plus which lane. The replica's lag
/// rows reuse it so the two vocabularies cannot drift.
pub(crate) fn lane_metric_labels(lane_name: &str) -> (String, &'static str) {
    if let Some(stem) = lane_name.strip_suffix(".passages.wal.jsonl") {
        (
            crate::registry::name_from_stem(stem).unwrap_or_else(|| stem.to_string()),
            "passages",
        )
    } else {
        let stem = lane_name.strip_suffix(".wal.jsonl").unwrap_or(lane_name);
        (
            crate::registry::name_from_stem(stem).unwrap_or_else(|| stem.to_string()),
            "graph",
        )
    }
}
