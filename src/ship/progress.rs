//! Shipping progress and cursor state: the flusher's window into how
//! far each log lane has shipped ([`ShipProgress`]), a published
//! file's change signature ([`FileSig`]) and last-shipped extent
//! ([`ShippedFile`]), and one log lane's shipping cursor
//! ([`LaneState`]).

use super::*;

/// The flusher's one window into shipping progress, consulted before
/// the GRAPH lane's housekeeping WAL reset (the one that follows a
/// successful image flush). Holding the reset until the tail has
/// shipped keeps the shipped stream gapless — the cheap, common path —
/// WITHOUT coupling correctness to it: a reset that proceeds anyway
/// (shipping stalled past the deferral budget, a rollback truncate,
/// or the passage log's post-compaction reset, none of which consult
/// this) just diverges the prefix, and the shipper answers with a
/// parent-snapshot-first series restart. The passage lane is excluded
/// on purpose: its reset rides a ratio-triggered compaction inside
/// the write path, and deferring it would re-fire whole-snapshot
/// rewrites on every store while the bucket lags — see the
/// `StateInner::ship_progress` doc in `registry.rs` for the full
/// argument.
///
/// Everything here is advisory and non-blocking: one mutexed map
/// touched by the shipper once per shipped segment and read by the
/// flusher once per reset decision. No write ever waits on it.
pub(crate) struct ShipProgress {
    /// Log path → highest seq shipped from it.
    lanes: Mutex<BTreeMap<PathBuf, u64>>,
    /// Past this many bytes of un-shipped log, resets proceed anyway:
    /// a dead bucket must degrade REPLICATION (to snapshot-grade RPO,
    /// via the series-restart path), never the local server — an
    /// indefinitely-deferred reset would otherwise walk `wal_bytes`
    /// into the WAL cap and start refusing writes, putting shipping
    /// back on the write path through the side door.
    defer_cap_bytes: u64,
}

/// Deferral budget before a housekeeping reset stops waiting for the
/// shipper. 64 MiB of retained-but-inert log is noise disk-wise, far
/// under the 256 MiB default WAL cap, and represents minutes of
/// backlog at any plausible write rate — a bucket outage longer than
/// that has already conceded the fine-grained RPO the deferral exists
/// to protect.
pub(super) const DEFAULT_DEFER_CAP_BYTES: u64 = 64 * 1024 * 1024;

impl ShipProgress {
    pub(crate) fn new() -> Self {
        Self {
            lanes: Mutex::new(BTreeMap::new()),
            defer_cap_bytes: DEFAULT_DEFER_CAP_BYTES,
        }
    }

    /// Records that everything up to `seq` in `log` is durably in the
    /// bucket. Called by the shipper only, after each segment PUT.
    pub(super) fn note_shipped(&self, log: &FsPath, seq: u64) {
        self.lanes.lock().insert(log.to_path_buf(), seq);
    }

    /// Forgets a lane whose local file vanished (context deleted); a
    /// re-created context must not inherit the old lane's high-water
    /// mark or its first flush would defer forever waiting for seqs
    /// the new lane will never reach — the new lane restarts at 1.
    pub(super) fn forget(&self, log: &FsPath) {
        self.lanes.lock().remove(log);
    }

    /// Whether the housekeeping reset of `log` — about to discard every
    /// record at or below `watermark` — may proceed without opening a
    /// gap in the shipped stream. `log_bytes` is the file's current
    /// size, for the deferral budget.
    pub(crate) fn allows_reset(&self, log: &FsPath, watermark: u64, log_bytes: u64) -> bool {
        if log_bytes >= self.defer_cap_bytes {
            return true;
        }
        // A lane the shipper has not reached yet ships from scratch on
        // its first visit — nothing shipped means nothing to keep
        // contiguous, EXCEPT that deferring here (returning false)
        // would be wrong forever for a server whose shipper died at
        // claim time. The shipper's own series-restart path covers the
        // gap either way, so unknown lanes defer only via the map:
        // absent entry = shipped nothing = defer until the first
        // segment lands or the budget runs out.
        let lanes = self.lanes.lock();
        lanes.get(log).copied().unwrap_or(0) >= watermark
    }
}

/// One published file's change signature. Publication is always a
/// rename (`storage::commit_staged`), so a new version is a new inode —
/// length and mtime ride along for filesystems (and platforms) where
/// inode reuse could otherwise alias two versions within one mtime
/// granule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FileSig {
    len: u64,
    mtime: Option<SystemTime>,
    ino: u64,
}

impl FileSig {
    pub(crate) fn of(metadata: &std::fs::Metadata) -> Self {
        #[cfg(unix)]
        let ino = std::os::unix::fs::MetadataExt::ino(metadata);
        #[cfg(not(unix))]
        let ino = 0;
        Self {
            len: metadata.len(),
            mtime: metadata.modified().ok(),
            ino,
        }
    }
}

/// One published file as last shipped: the change signature the scan
/// compares, plus the uploaded bytes' extent for the manifest.
#[derive(Debug, Clone, Copy)]
pub(super) struct ShippedFile {
    pub(super) sig: FileSig,
    pub(super) content: ManifestFile,
}

/// One log lane's shipping cursor — everything needed to prove the
/// next tail read is a strict continuation of what already shipped
/// (see the module doc's "Log-lane correctness").
#[derive(Clone)]
pub(super) struct LaneState {
    pub(super) series: u64,
    pub(super) next_seg: u64,
    /// Bytes of the local file already shipped (always a complete-line
    /// boundary), and the CRC-32C of exactly those bytes.
    pub(super) shipped_offset: u64,
    pub(super) shipped_crc: u32,
    /// Highest record seq shipped — [`ShipProgress`] and the lag
    /// metric speak seqs, not offsets.
    pub(super) shipped_seq: u64,
    /// When un-shipped local records were first observed, for the lag
    /// age metric; cleared when the lane catches up.
    pub(super) pending_since: Option<Instant>,
    /// The file's signature as of the last full read. Unchanged since
    /// then means that read already found everything the file
    /// currently has, so [`super::shipper::Shipper::ship_lane`] can
    /// skip straight to the lag metric instead of re-reading the file
    /// and re-hashing the whole shipped prefix.
    pub(super) last_seen_sig: Option<FileSig>,
    /// `newest_seq` as of that same read, cached for the lag metric on
    /// a cycle the signature check above short-circuits.
    pub(super) local_seq: u64,
}

impl LaneState {
    pub(super) fn fresh(series: u64) -> Self {
        Self {
            series,
            next_seg: 0,
            shipped_offset: 0,
            shipped_crc: 0,
            shipped_seq: 0,
            pending_since: None,
            last_seen_sig: None,
            local_seq: 0,
        }
    }
}
