//! Continuous replication of the data directory to object storage —
//! "WAL shipping" — and the restore path back out of a bucket
//! (issue #127). This is the second durability tier: a local crash
//! still loses nothing (the WAL's own claim, unchanged); losing the
//! machine or the volume now costs only the shipping lag, because a
//! bucket in another failure domain holds a continuously-refreshed
//! copy of every file family.
//!
//! # What ships
//!
//! Everything the data directory holds, split by how it changes:
//!
//! - **Published files** — every file that appears via
//!   stage-then-rename (`storage::write_atomic`): images, meta,
//!   sources, passage snapshots, the derived sidecars, group records,
//!   the OAuth grant store, and the crash markers. Immutable once
//!   visible, so each is uploaded whole when its signature (inode,
//!   length, mtime) changes, and deleted remotely when it disappears
//!   locally.
//! - **The two log lanes** — the graph WAL (`{stem}.wal.jsonl`) and
//!   the passage log (`{stem}.passages.wal.jsonl`) mutate in place
//!   (append, and occasionally truncate/reset), so they are tailed
//!   instead: each cycle ships the newly-appended complete records as
//!   one immutable segment object.
//!
//! The shipper POLLS. Nothing on the write path signals it, blocks on
//! it, or even knows it exists — "shipping stays off the write path"
//! is structural, not a promise. The one integration point runs the
//! other way: the flusher consults [`ShipProgress`] before its
//! housekeeping WAL reset so the tail it is about to empty has been
//! shipped first (bounded — see [`ShipProgress::allows_reset`]), which
//! keeps the shipped stream gapless without ever delaying a write.
//!
//! # Log-lane correctness
//!
//! A tailed copy must never diverge from what replay would see. The
//! lane state remembers how many bytes it has shipped and the CRC-32C
//! of exactly those bytes; a cycle re-reads the file and:
//!
//! - prefix intact (same length-or-longer, same CRC over the shipped
//!   prefix): the new complete records append as the next segment of
//!   the current SERIES;
//! - prefix gone (file shrank: a post-flush reset, a rollback of an
//!   unacknowledged batch) or rewritten (a rollback followed by new
//!   appends over the same offsets, or a delete + re-create): the
//!   local log and the shipped series have diverged, so the lane's
//!   PARENT SNAPSHOT (`.ctx` for the graph lane, `.passages.bin` for
//!   the passage lane) is uploaded FIRST, then a fresh series starts
//!   with the file's current contents.
//!
//! Restore concatenates only the newest series of each lane, so the
//! parent-snapshot-first order guarantees the snapshot's watermark
//! always reaches the series' first record: replay applies a gapless
//! suffix or nothing. Both shapes of local log surgery
//! (`wal::reset`, `wal::truncate_to`) land in the "prefix gone"
//! arm — neither needs to know shipping exists.
//!
//! # Epoch fencing
//!
//! The bucket is a second shared medium; the flock guards only its own
//! local directory, so two live processes on different volumes can
//! both hold their own lock and ship to the same bucket — a botched
//! restore, a second instance with the same env, a wedged-but-alive
//! old writer beside its replacement. Unfenced, that interleaves two
//! histories into one lineage.
//!
//! At startup the shipper claims a monotonic GENERATION: it lists
//! `fence/` and creates `fence/{N+1}` with a conditional create
//! (`PutMode::Create`), retrying upward until one create wins — the
//! only conditional primitive every backend (S3, GCS, Azure, local
//! files) supports, which is also why the fence is an immutable object
//! per generation rather than one mutated object. Everything the
//! claimant ships lives under `gen-{N+1}/`, so a deposed writer's
//! in-flight uploads land in its own superseded namespace and can
//! never corrupt the successor's. Before any cycle that has something
//! to upload, the shipper asks whether `fence/{N+2}` exists; if it
//! does, a newer claimant owns the bucket and this shipper FAIL-STOPS —
//! permanently, loudly (metric + `taguru::audit` line) — while the
//! serve path keeps running on its local truth. No TTL, no renewal, no
//! clock: failure semantics stay fail-stop and human-legible. The
//! fence object's body carries `{generation, holder, claimed_at}` so a
//! lease layer (a TTL, heartbeats) could be added later without
//! redesigning the medium — today's fence is a permanent lease with
//! TTL 0.
//!
//! # Bucket layout
//!
//! ```text
//! {prefix}/fence/{N:020}                       one immutable claim per generation
//! {prefix}/gen-{N:020}/complete                the manifest; restore requires it
//! {prefix}/gen-{N:020}/heartbeat               refreshed every minute while the writer lives
//! {prefix}/gen-{N:020}/retired                 written by a graceful stop
//! {prefix}/gen-{N:020}/files/{filename}        published files, named as on disk
//! {prefix}/gen-{N:020}/wal/{filename}/{series:010}-{seg:010}.jsonl
//! ```
//!
//! `complete` carries the [`Manifest`] — every shipped file's and
//! lane's exact extent (length + CRC-32C), refreshed after each batch
//! of uploads — so its existence still means "this generation
//! restores whole", and its body lets a reader verify every
//! downloaded byte and reuse local files without downloading at all
//! (the boot-from-bucket path, `crate::hydrate`). `heartbeat` and
//! `retired` feed that path's takeover guard: liveness and clean
//! shutdown as objects, read by the NEXT claimant's boot — pure
//! ergonomics on top of the fence, never a substitute for it.
//!
//! A restore reads the newest generation carrying `complete`:
//! download `files/*`, concatenate each lane's newest series in
//! segment order — manifest-verified when the manifest is present —
//! and the result is a data directory the ordinary boot path loads —
//! crash-consistent by the same argument as a local crash, since every
//! shipped object is either a whole published file or a gapless run
//! of acknowledged log records.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use futures_util::StreamExt;
use object_store::path::Path as StorePath;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload};
// parking_lot, not std::sync: a panic while a lock is held must not
// poison it and brick the shipper's progress tracking for the rest of
// the process (the same reasoning as the registry — see Cargo.toml).
use parking_lot::Mutex;

use crate::registry::AppState;

#[path = "ship/config.rs"]
mod config;
#[path = "ship/error.rs"]
mod error;
#[path = "ship/handle.rs"]
mod handle;
#[path = "ship/naming.rs"]
mod naming;
#[path = "ship/progress.rs"]
mod progress;
#[path = "ship/restore.rs"]
mod restore;
#[path = "ship/shipper.rs"]
mod shipper;

pub(crate) use config::{ReplicateConfig, open_store};
use error::{ShipError, store_error};
pub(crate) use handle::spawn;
use naming::{
    COMPLETE_MARKER, EntryKind, FENCE_PREFIX, FenceBody, HEARTBEAT_INTERVAL, classify,
    parent_snapshot_of,
};
pub(crate) use naming::{
    HEARTBEAT_MARKER, Manifest, ManifestFile, ManifestLane, REPLICATION_RECORD, RETIRED_MARKER,
    ReplicationRecord, TAKEOVER_GRACE, complete_key, fence_key, gen_root, lane_metric_labels,
    read_replication_record, segment_name, write_replication_record,
};
pub(crate) use progress::{FileSig, ShipProgress};
use progress::{LaneState, ShippedFile};
pub(crate) use restore::{
    fetch, fetch_lane, newest_complete_generation, read_manifest, run, verify_file_bytes,
    write_restored_file,
};
pub(crate) use shipper::{FenceInfo, Shipper, fence_holder, newest_fence};
// Re-exported only for the sibling test module below (`use super::*` in
// ship/tests.rs) — not reachable from outside `#[cfg(test)]` builds.
#[cfg(test)]
use progress::DEFAULT_DEFER_CAP_BYTES;
#[cfg(test)]
use restore::{parse_segment_name, restore_into};

#[path = "ship/tests.rs"]
#[cfg(test)]
mod tests;
