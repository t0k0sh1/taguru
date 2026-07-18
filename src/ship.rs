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
//! {prefix}/gen-{N:020}/complete                baseline finished; restore requires it
//! {prefix}/gen-{N:020}/files/{filename}        published files, named as on disk
//! {prefix}/gen-{N:020}/wal/{filename}/{series:010}-{seg:010}.jsonl
//! ```
//!
//! A restore reads the newest generation carrying `complete`: download
//! `files/*`, concatenate each lane's newest series in segment order,
//! and the result is a data directory the ordinary boot path loads —
//! crash-consistent by the same argument as a local crash, since every
//! shipped object is either a whole published file or a gapless run
//! of acknowledged log records.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path as FsPath, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

use futures_util::StreamExt;
use object_store::path::Path as StorePath;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload};

use crate::registry::AppState;

/// How the operator turns shipping on: a bucket URL, nothing else
/// required. Credentials ride each cloud's default chain (`AWS_*`,
/// `GOOGLE_*`, `AZURE_*` — whatever `object_store`'s builders read
/// from the environment), so the one variable is the whole feature.
pub(crate) struct ReplicateConfig {
    pub(crate) url: String,
    pub(crate) interval: Duration,
}

impl ReplicateConfig {
    /// `TAGURU_REPLICATE_URL` (unset = shipping off), plus the poll
    /// cadence `TAGURU_REPLICATE_INTERVAL_MS` (default 1000 — the
    /// steady-state RPO knob). A zero interval would spin the poll
    /// loop; floor to 100ms, loudly, like every other env knob.
    pub(crate) fn from_env() -> Option<Self> {
        let url = std::env::var("TAGURU_REPLICATE_URL").ok()?;
        let url = url.trim().to_string();
        if url.is_empty() {
            // The same present-but-blank trap TAGURU_PUBLIC_URL guards
            // against: almost always a templating accident, never a
            // deliberate opt-out spelled as an empty string.
            tracing::warn!(
                "TAGURU_REPLICATE_URL is set but empty: treating replication as disabled — \
                 unset the variable entirely if that's intended"
            );
            return None;
        }
        let requested = crate::env::env_number("TAGURU_REPLICATE_INTERVAL_MS", 1000);
        let interval = if requested < 100 {
            tracing::warn!(
                "TAGURU_REPLICATE_INTERVAL_MS={requested} would busy-poll the data \
                 directory; using 100"
            );
            Duration::from_millis(100)
        } else {
            Duration::from_millis(requested as u64)
        };
        Some(Self { url, interval })
    }
}

/// Opens the store a replicate URL names, with each cloud's default
/// credential chain. `parse_url` alone constructs builders WITHOUT
/// environment credentials — fine for `file://`, wrong for every
/// cloud — so the cloud schemes go through their builders' `from_env`
/// explicitly. `file://` is first-class, not a test crutch: it is how
/// the round trip is verified without cloud spend, and how an
/// air-gapped deployment ships to a mounted remote volume.
pub(crate) fn open_store(url: &str) -> io::Result<(Arc<dyn ObjectStore>, StorePath)> {
    use object_store::ObjectStoreScheme;

    let parsed = url::Url::parse(url)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, format!("{url}: {error}")))?;
    let (scheme, path) = ObjectStoreScheme::parse(&parsed)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, format!("{url}: {error}")))?;
    let root = StorePath::parse(path)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, format!("{url}: {error}")))?;
    let store: Arc<dyn ObjectStore> = match scheme {
        ObjectStoreScheme::Local => {
            // The bucket must exist before S3/GCS/Azure accept a write;
            // hold file:// to the same contract instead of silently
            // mkdir-ing a typo into a fresh empty "bucket".
            let dir = parsed.to_file_path().map_err(|()| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("{url}: not a local path"),
                )
            })?;
            if !dir.is_dir() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "{}: replication target directory does not exist",
                        dir.display()
                    ),
                ));
            }
            // Prefix handling differs local vs cloud: the URL's path IS
            // the directory, so the store roots there and the in-store
            // prefix is empty.
            let local = object_store::local::LocalFileSystem::new_with_prefix(&dir)
                .map_err(|error| io::Error::other(format!("{}: {error}", dir.display())))?;
            return Ok((Arc::new(local), StorePath::default()));
        }
        ObjectStoreScheme::AmazonS3 => Arc::new(
            object_store::aws::AmazonS3Builder::from_env()
                .with_url(url)
                .build()
                .map_err(|error| io::Error::other(format!("{url}: {error}")))?,
        ),
        ObjectStoreScheme::GoogleCloudStorage => Arc::new(
            object_store::gcp::GoogleCloudStorageBuilder::from_env()
                .with_url(url)
                .build()
                .map_err(|error| io::Error::other(format!("{url}: {error}")))?,
        ),
        ObjectStoreScheme::MicrosoftAzure => Arc::new(
            object_store::azure::MicrosoftAzureBuilder::from_env()
                .with_url(url)
                .build()
                .map_err(|error| io::Error::other(format!("{url}: {error}")))?,
        ),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{url}: unsupported replication scheme — use s3://, gs://, az://, or file://"
                ),
            ));
        }
    };
    Ok((store, root))
}

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
const DEFAULT_DEFER_CAP_BYTES: u64 = 64 * 1024 * 1024;

impl ShipProgress {
    pub(crate) fn new() -> Self {
        Self {
            lanes: Mutex::new(BTreeMap::new()),
            defer_cap_bytes: DEFAULT_DEFER_CAP_BYTES,
        }
    }

    /// Records that everything up to `seq` in `log` is durably in the
    /// bucket. Called by the shipper only, after each segment PUT.
    fn note_shipped(&self, log: &FsPath, seq: u64) {
        self.lanes.lock().unwrap().insert(log.to_path_buf(), seq);
    }

    /// Forgets a lane whose local file vanished (context deleted); a
    /// re-created context must not inherit the old lane's high-water
    /// mark or its first flush would defer forever waiting for seqs
    /// the new lane will never reach — the new lane restarts at 1.
    fn forget(&self, log: &FsPath) {
        self.lanes.lock().unwrap().remove(log);
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
        let lanes = self.lanes.lock().unwrap();
        lanes.get(log).copied().unwrap_or(0) >= watermark
    }
}

/// One published file's change signature. Publication is always a
/// rename (`storage::commit_staged`), so a new version is a new inode —
/// length and mtime ride along for filesystems (and platforms) where
/// inode reuse could otherwise alias two versions within one mtime
/// granule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileSig {
    len: u64,
    mtime: Option<SystemTime>,
    ino: u64,
}

impl FileSig {
    fn of(metadata: &std::fs::Metadata) -> Self {
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

/// One log lane's shipping cursor — everything needed to prove the
/// next tail read is a strict continuation of what already shipped
/// (see the module doc's "Log-lane correctness").
#[derive(Clone)]
struct LaneState {
    series: u64,
    next_seg: u64,
    /// Bytes of the local file already shipped (always a complete-line
    /// boundary), and the CRC-32C of exactly those bytes.
    shipped_offset: u64,
    shipped_crc: u32,
    /// Highest record seq shipped — [`ShipProgress`] and the lag
    /// metric speak seqs, not offsets.
    shipped_seq: u64,
    /// When un-shipped local records were first observed, for the lag
    /// age metric; cleared when the lane catches up.
    pending_since: Option<Instant>,
}

impl LaneState {
    fn fresh(series: u64) -> Self {
        Self {
            series,
            next_seg: 0,
            shipped_offset: 0,
            shipped_crc: 0,
            shipped_seq: 0,
            pending_since: None,
        }
    }
}

/// Why a cycle stopped early. Transient errors surface as `Io` and the
/// next cycle retries from the recorded cursors; `Fenced` is terminal
/// by design.
#[derive(Debug)]
pub(crate) enum ShipError {
    /// A newer generation claimed the bucket: fail-stop.
    Fenced {
        newer_generation: u64,
    },
    Io(io::Error),
}

impl std::fmt::Display for ShipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fenced { newer_generation } => write!(
                f,
                "fenced: generation {newer_generation} claimed the bucket after ours"
            ),
            Self::Io(error) => error.fmt(f),
        }
    }
}

impl From<io::Error> for ShipError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

fn store_error(context: &str, error: object_store::Error) -> ShipError {
    ShipError::Io(io::Error::other(format!("{context}: {error}")))
}

const FENCE_PREFIX: &str = "fence";
const COMPLETE_MARKER: &str = "complete";

/// Fixed-width decimal so lexicographic object listing IS numeric
/// ordering — restore sorts names, never parses to sort.
fn fence_key(root: &StorePath, generation: u64) -> StorePath {
    root.clone()
        .join(FENCE_PREFIX)
        .join(format!("{generation:020}"))
}

fn gen_root(root: &StorePath, generation: u64) -> StorePath {
    root.clone().join(format!("gen-{generation:020}"))
}

fn segment_name(series: u64, seg: u64) -> String {
    format!("{series:010}-{seg:010}.jsonl")
}

/// What a fence object says. The claim is the object's existence — the
/// body exists for operators (`who took the bucket from me?`) and as
/// the future home of lease fields (a TTL, a heartbeat stamp), so
/// automation could be layered on without changing the medium.
#[derive(serde::Serialize, serde::Deserialize)]
struct FenceBody {
    generation: u64,
    holder: String,
    claimed_at_epoch_secs: u64,
}

/// The names a directory scan sorts every entry into. Only shapes the
/// shipper understands leave the machine: an unknown file would be
/// shipped as opaque bytes just fine, but restore fidelity is easier
/// to reason about when the shipped set is exactly the family the
/// server itself reads, and staging litter (`*.tmp{N}`) must never
/// ship at all — those names are mid-write by definition.
fn classify(name: &str) -> EntryKind {
    if name == ".taguru.lock" {
        return EntryKind::Skip;
    }
    // `staging_path` builds `{final}.tmp{nonce}` names; matching the
    // shape (not just ".tmp") keeps a hypothetical user file named
    // exactly "x.tmp" shippable while excluding every stager.
    if let Some(extension) = FsPath::new(name).extension().and_then(|e| e.to_str())
        && let Some(rest) = extension.strip_prefix("tmp")
        && rest.chars().all(|c| c.is_ascii_digit())
    {
        return EntryKind::Skip;
    }
    if name.ends_with(".wal.jsonl") {
        return EntryKind::LogLane;
    }
    EntryKind::Published
}

#[derive(Debug, PartialEq, Eq)]
enum EntryKind {
    Published,
    LogLane,
    Skip,
}

/// The lane's parent snapshot: the file whose watermark makes a series
/// restart safe to restore from (uploaded BEFORE the new series — see
/// the module doc). The passage suffix is checked first because both
/// lanes end in `.wal.jsonl`.
fn parent_snapshot_of(lane_name: &str) -> Option<String> {
    if let Some(stem) = lane_name.strip_suffix(".passages.wal.jsonl") {
        return Some(format!("{stem}.passages.bin"));
    }
    lane_name
        .strip_suffix(".wal.jsonl")
        .map(|stem| format!("{stem}.ctx"))
}

/// The per-lane label pair the lag metric carries: the context's
/// decoded name where the stem decodes (it always should — these files
/// were written by the server), plus which lane.
fn lane_metric_labels(lane_name: &str) -> (String, &'static str) {
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

pub(crate) struct Shipper {
    store: Arc<dyn ObjectStore>,
    root: StorePath,
    generation: u64,
    data_dir: PathBuf,
    files: BTreeMap<String, FileSig>,
    lanes: BTreeMap<String, LaneState>,
    progress: Arc<ShipProgress>,
    state: AppState,
    baseline_complete: bool,
}

impl Shipper {
    /// Claims the next generation and returns a shipper whose first
    /// cycle will run the baseline sync. Everything before the claim
    /// succeeds is refusal-shaped: a bucket that cannot even take the
    /// fence write is a bucket that cannot take data.
    pub(crate) async fn claim(
        store: Arc<dyn ObjectStore>,
        root: StorePath,
        data_dir: PathBuf,
        progress: Arc<ShipProgress>,
        state: AppState,
    ) -> Result<Self, ShipError> {
        let mut generation = newest_fence(&store, &root).await?.unwrap_or(0) + 1;
        loop {
            let body = FenceBody {
                generation,
                holder: format!(
                    "{}#{}",
                    std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown-host".into()),
                    std::process::id()
                ),
                claimed_at_epoch_secs: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            };
            let payload =
                PutPayload::from(serde_json::to_vec(&body).expect("no unserializable field"));
            let create = PutOptions::from(PutMode::Create);
            match store
                .put_opts(&fence_key(&root, generation), payload, create)
                .await
            {
                Ok(_) => break,
                // Two claimants racing one generation: the loser's
                // create fails the condition and it bids one higher —
                // the loop converges because every retry is caused by
                // someone else's SUCCESSFUL claim.
                Err(object_store::Error::AlreadyExists { .. }) => generation += 1,
                Err(error) => return Err(store_error("claiming the replication fence", error)),
            }
        }
        tracing::info!(
            target: "taguru::audit",
            generation,
            "replication generation claimed",
        );
        Ok(Self {
            store,
            root,
            generation,
            data_dir,
            files: BTreeMap::new(),
            lanes: BTreeMap::new(),
            progress,
            state,
            baseline_complete: false,
        })
    }

    /// Whether a newer claimant has taken the bucket. One GET per
    /// dirty cycle: a successor claims exactly `our generation + 1`
    /// upward, and every claim is caused by a real claimant, so the
    /// first successor's fence is the one to watch for.
    async fn fenced_by(&self) -> Result<Option<u64>, ShipError> {
        match newest_fence(&self.store, &self.root).await? {
            Some(newest) if newest > self.generation => Ok(Some(newest)),
            _ => Ok(None),
        }
    }

    /// One poll cycle: scan, and if anything changed, re-check the
    /// fence and ship the difference. Returns whether anything
    /// shipped. `Fenced` is terminal — the caller stops the loop.
    pub(crate) async fn cycle(&mut self) -> Result<bool, ShipError> {
        let scan = self.scan()?;
        let dirty = !self.baseline_complete
            || !scan.changed.is_empty()
            || !scan.vanished.is_empty()
            || !scan.lanes.is_empty();
        if !dirty {
            return Ok(false);
        }
        // The fence check gates UPLOADS, not local reads: an idle
        // deposed shipper discovers its deposition on its next real
        // work, which is exactly when it matters — and never sooner,
        // because a fenced check with nothing to ship would fail-stop
        // a server whose bucket successor changes nothing about its
        // local correctness.
        if let Some(newer_generation) = self.fenced_by().await? {
            return Err(ShipError::Fenced { newer_generation });
        }
        let mut shipped = false;
        for name in &scan.vanished {
            self.retire_file(name).await?;
            shipped = true;
        }
        for name in &scan.changed {
            // The lane loop below owns log files; `scan` never lists
            // them here.
            let sig = self.ship_published(name).await?;
            self.files.insert(name.clone(), sig);
            shipped = true;
        }
        for name in &scan.lanes {
            shipped |= self.ship_lane(name).await?;
        }
        if !self.baseline_complete {
            self.put(
                &gen_root(&self.root, self.generation).join(COMPLETE_MARKER),
                Vec::new(),
            )
            .await?;
            self.baseline_complete = true;
            tracing::info!(
                target: "taguru::audit",
                generation = self.generation,
                "replication baseline complete — the bucket can restore this directory",
            );
            shipped = true;
        }
        if shipped {
            self.state.metrics().record_replication_success();
        }
        Ok(shipped)
    }

    /// Reads the directory once and buckets every entry. Log lanes are
    /// listed every dirty cycle (their tail decides itself whether
    /// anything is new); published files only when their signature
    /// moved; names that vanished since the last scan are retired.
    fn scan(&self) -> io::Result<Scan> {
        let mut scan = Scan::default();
        let mut seen = std::collections::BTreeSet::new();
        for entry in std::fs::read_dir(&self.data_dir)? {
            let entry = entry?;
            let Ok(name) = entry.file_name().into_string() else {
                // A non-UTF-8 name was not written by this server;
                // skip it rather than guess an object key for it.
                continue;
            };
            let Ok(metadata) = entry.metadata() else {
                // Vanished between readdir and stat — the next cycle
                // sees the settled state.
                continue;
            };
            if !metadata.is_file() {
                continue;
            }
            match classify(&name) {
                EntryKind::Skip => {}
                EntryKind::LogLane => {
                    seen.insert(name.clone());
                    scan.lanes.push(name);
                }
                EntryKind::Published => {
                    seen.insert(name.clone());
                    let sig = FileSig::of(&metadata);
                    if self.files.get(&name) != Some(&sig) {
                        scan.changed.push(name);
                    }
                }
            }
        }
        scan.vanished = self
            .files
            .keys()
            .chain(self.lanes.keys())
            .filter(|name| !seen.contains(*name))
            .cloned()
            .collect();
        Ok(scan)
    }

    /// Uploads one published file whole. The read races publication by
    /// design: `fs::read` holds the opened inode, so it always
    /// delivers ONE complete published version — if a newer rename
    /// lands mid-read, the next cycle's signature check ships that one
    /// too. The signature is taken BEFORE the read: taking it after
    /// could stamp version N+1's signature on version N's bytes and
    /// never re-ship N+1.
    async fn ship_published(&mut self, name: &str) -> Result<FileSig, ShipError> {
        let path = self.data_dir.join(name);
        let metadata = std::fs::metadata(&path).map_err(ShipError::Io)?;
        let sig = FileSig::of(&metadata);
        let bytes = std::fs::read(&path).map_err(ShipError::Io)?;
        let key = gen_root(&self.root, self.generation)
            .join("files")
            .join(name);
        self.put(&key, bytes).await?;
        Ok(sig)
    }

    /// Removes a vanished file's remote counterpart — and, for a log
    /// lane, its whole segment prefix, so a re-created context of the
    /// same name can never interleave with the old incarnation's
    /// records on restore.
    async fn retire_file(&mut self, name: &str) -> Result<(), ShipError> {
        let generation_root = gen_root(&self.root, self.generation);
        if self.lanes.remove(name).is_some() {
            self.progress.forget(&self.data_dir.join(name));
            let (context, lane_kind) = lane_metric_labels(name);
            self.state
                .metrics()
                .forget_replication_lane(&context, lane_kind);
            let prefix = generation_root.join("wal").join(name);
            delete_prefix(self.store.as_ref(), &prefix).await?;
        } else {
            self.files.remove(name);
            let key = generation_root.join("files").join(name);
            match self.store.delete(&key).await {
                Ok(()) => {}
                // Deleting a file the baseline never shipped (created
                // and deleted between two cycles) is a no-op, not an
                // error.
                Err(object_store::Error::NotFound { .. }) => {}
                Err(error) => {
                    self.state.metrics().record_replication_error();
                    return Err(store_error("retiring a replicated file", error));
                }
            }
        }
        Ok(())
    }

    /// Tails one log lane (see the module doc's correctness argument).
    /// Returns whether a segment (or a series restart) shipped. The
    /// cursor is worked on as a copy and written back only after the
    /// segment PUT succeeds: a failed upload re-reads and re-ships the
    /// same bytes next cycle under the same segment name — idempotent,
    /// since the name encodes the position.
    async fn ship_lane(&mut self, name: &str) -> Result<bool, ShipError> {
        let path = self.data_dir.join(name);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            // Vanished mid-cycle: the next scan retires it.
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(ShipError::Io(error)),
        };
        let mut lane = self
            .lanes
            .get(name)
            .cloned()
            .unwrap_or_else(|| LaneState::fresh(0));

        // Is the shipped prefix still literally the file's prefix?
        let prefix_intact = bytes.len() as u64 >= lane.shipped_offset
            && crate::crc32c::crc32c(&bytes[..lane.shipped_offset as usize]) == lane.shipped_crc;

        if !prefix_intact {
            // Divergence: reset, rollback, or delete + re-create. The
            // parent snapshot covers everything the divergence
            // discarded, and it must be IN THE BUCKET before the new
            // series exists there, or a restore in the window between
            // the two could see a series whose first record is beyond
            // the restored snapshot's watermark.
            let parent = parent_snapshot_of(name).expect("lane names always carry a wal suffix");
            let parent_path = self.data_dir.join(&parent);
            match std::fs::metadata(&parent_path) {
                Ok(metadata) => {
                    // Skip the upload when this cycle (or an earlier
                    // one) already shipped exactly this version — the
                    // common case, since the reset that diverged the
                    // lane follows the very flush that published it.
                    if self.files.get(&parent) != Some(&FileSig::of(&metadata)) {
                        let sig = self.ship_published(&parent).await?;
                        self.files.insert(parent, sig);
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    // No local snapshot (a re-created context that has
                    // not flushed yet): the REMOTE snapshot, if any, is
                    // the old incarnation's and its watermark would
                    // swallow the new lane's low seqs on restore.
                    // Retire it first.
                    self.retire_file(&parent).await?;
                }
                Err(error) => return Err(ShipError::Io(error)),
            }
            lane = LaneState::fresh(lane.series + 1);
        }

        // Ship only complete lines: a torn tail (crash or mid-write
        // race) becomes shippable one heal or one append later.
        let new = &bytes[lane.shipped_offset as usize..];
        let complete_end = new.iter().rposition(|&b| b == b'\n').map(|at| at + 1);
        let shipped = match complete_end {
            None => false,
            Some(complete_end) => {
                let complete = &new[..complete_end];
                let records = crate::wal::shippable_records(complete).map_err(|error| {
                    // Interior corruption in bytes replay would also
                    // refuse — shipping it would spread the rot to
                    // every restore. Leave the lane where it is and
                    // surface the error; the server's own next load of
                    // this context hits the same wall.
                    ShipError::Io(io::Error::new(
                        error.kind(),
                        format!("{}: {error}", path.display()),
                    ))
                })?;
                if records.is_empty() {
                    false
                } else {
                    let key = gen_root(&self.root, self.generation)
                        .join("wal")
                        .join(name)
                        .join(segment_name(lane.series, lane.next_seg));
                    let last_seq = records.last().expect("checked non-empty").seq;
                    self.put(&key, complete.to_vec()).await?;
                    lane.shipped_offset += complete_end as u64;
                    lane.shipped_crc =
                        crate::crc32c::crc32c(&bytes[..lane.shipped_offset as usize]);
                    lane.shipped_seq = last_seq;
                    lane.next_seg += 1;
                    self.progress.note_shipped(&path, last_seq);
                    true
                }
            }
        };

        // Lag bookkeeping, shipped or not: how far the local log's
        // newest record is beyond the shipped one, and for how long.
        let local_seq = newest_seq(&bytes).unwrap_or(lane.shipped_seq);
        if local_seq > lane.shipped_seq {
            lane.pending_since.get_or_insert_with(Instant::now);
        } else {
            lane.pending_since = None;
        }
        let age_secs = lane
            .pending_since
            .map(|since| since.elapsed().as_secs())
            .unwrap_or(0);
        let (context, lane_kind) = lane_metric_labels(name);
        self.state.metrics().note_replication_lane(
            &context,
            lane_kind,
            local_seq.saturating_sub(lane.shipped_seq),
            age_secs,
        );
        self.lanes.insert(name.to_string(), lane);
        Ok(shipped)
    }

    async fn put(&self, key: &StorePath, bytes: Vec<u8>) -> Result<(), ShipError> {
        match self.store.put(key, PutPayload::from(bytes)).await {
            Ok(_) => {
                self.state.metrics().record_replication_upload();
                Ok(())
            }
            Err(error) => {
                self.state.metrics().record_replication_error();
                Err(store_error("uploading a replicated object", error))
            }
        }
    }
}

/// The newest (highest) seq among the file's complete lines, ignoring
/// integrity: this feeds the LAG metric only, where an honest "how far
/// behind" matters more than validity — corrupt bytes will surface as
/// a shipping error, not a hidden zero lag.
fn newest_seq(bytes: &[u8]) -> Option<u64> {
    #[derive(serde::Deserialize)]
    struct SeqOnly {
        seq: u64,
    }
    let mut segments: Vec<&[u8]> = bytes.split(|&b| b == b'\n').collect();
    segments.pop();
    segments
        .iter()
        .rev()
        .find_map(|line| serde_json::from_slice::<SeqOnly>(line).ok().map(|r| r.seq))
}

#[derive(Default)]
struct Scan {
    changed: Vec<String>,
    vanished: Vec<String>,
    lanes: Vec<String>,
}

/// The highest generation with a fence object, scanned by listing the
/// fence prefix — names are fixed-width decimals, so the maximum is
/// the lexicographic maximum.
async fn newest_fence(
    store: &Arc<dyn ObjectStore>,
    root: &StorePath,
) -> Result<Option<u64>, ShipError> {
    let prefix = root.clone().join(FENCE_PREFIX);
    let mut newest = None;
    let mut listing = store.list(Some(&prefix));
    while let Some(meta) = listing.next().await {
        let meta = meta.map_err(|error| store_error("listing the replication fence", error))?;
        if let Some(name) = meta.location.filename()
            && let Ok(generation) = name.parse::<u64>()
        {
            newest = newest.max(Some(generation));
        }
    }
    Ok(newest)
}

/// Deletes every object under `prefix`, for retiring a vanished
/// lane's segments.
async fn delete_prefix(store: &dyn ObjectStore, prefix: &StorePath) -> Result<(), ShipError> {
    let keys: Vec<StorePath> = {
        let mut listing = store.list(Some(prefix));
        let mut keys = Vec::new();
        while let Some(meta) = listing.next().await {
            let meta = meta.map_err(|error| store_error("listing replicated segments", error))?;
            keys.push(meta.location);
        }
        keys
    };
    for key in keys {
        match store.delete(&key).await {
            Ok(()) | Err(object_store::Error::NotFound { .. }) => {}
            Err(error) => return Err(store_error("retiring replicated segments", error)),
        }
    }
    Ok(())
}

/// The serve-side handle: signals the shipper to stop and waits for
/// its final cycle, so the post-drain flush (`main`'s shutdown runs
/// `flush_dirty` first) reaches the bucket before the process exits.
pub(crate) struct ShipperHandle {
    stop: tokio::sync::watch::Sender<bool>,
    task: tokio::task::JoinHandle<()>,
}

impl ShipperHandle {
    pub(crate) async fn shutdown(self) {
        let _ = self.stop.send(true);
        if let Err(error) = self.task.await {
            tracing::warn!(%error, "replication task did not shut down cleanly");
        }
    }
}

/// Boots the shipper as one background task: claim a generation, then
/// poll until told to stop (one final cycle after the signal drains
/// the shutdown flush) — or until fenced, which stops it for good.
///
/// Claiming inside the task keeps a slow or unreachable bucket off
/// the serve path: the server binds and answers while the claim
/// retries; every failure is surfaced through the metric and the log
/// rather than a refused boot. The ONE thing a boot refuses on is a
/// URL that does not even parse (`open_store` in the caller) — a typo
/// should fail loudly at start, an outage should not.
pub(crate) fn spawn(
    store: Arc<dyn ObjectStore>,
    root: StorePath,
    data_dir: PathBuf,
    interval: Duration,
    progress: Arc<ShipProgress>,
    state: AppState,
) -> ShipperHandle {
    let (stop, mut stopped) = tokio::sync::watch::channel(false);
    let task = tokio::spawn(async move {
        let mut shipper = loop {
            match Shipper::claim(
                Arc::clone(&store),
                root.clone(),
                data_dir.clone(),
                Arc::clone(&progress),
                state.clone(),
            )
            .await
            {
                Ok(shipper) => break shipper,
                Err(error) => {
                    state.metrics().record_replication_error();
                    tracing::warn!(%error, "replication fence claim failed; retrying");
                    tokio::select! {
                        _ = tokio::time::sleep(interval.max(Duration::from_secs(1))) => {}
                        _ = stopped.changed() => return,
                    }
                }
            }
        };
        loop {
            let stopping = *stopped.borrow();
            match shipper.cycle().await {
                Ok(_) => {}
                Err(ShipError::Fenced { newer_generation }) => {
                    // Fail-stop, permanently: the successor owns the
                    // bucket. The serve path is untouched — it keeps
                    // answering from its local truth — but nothing
                    // more leaves this process, and both the metric
                    // and the audit line say so.
                    state.metrics().record_replication_fenced();
                    tracing::error!(
                        target: "taguru::audit",
                        generation = shipper.generation,
                        newer_generation,
                        "replication FENCED: a newer writer claimed the bucket — shipping \
                         stopped for good; this server keeps serving its local data (restart \
                         it to contest the claim, after making sure only one writer should \
                         exist)",
                    );
                    return;
                }
                Err(ShipError::Io(error)) => {
                    // Transient by assumption: the cursors did not
                    // advance past anything unshipped, so the next
                    // cycle retries exactly where this one failed.
                    tracing::warn!(%error, "replication cycle failed; will retry");
                }
            }
            if stopping {
                return;
            }
            tokio::select! {
                _ = tokio::time::sleep(interval) => {}
                _ = stopped.changed() => {}
            }
        }
    });
    ShipperHandle { stop, task }
}

// ---------------------------------------------------------------------------
// Restore: materialize a data directory from the bucket.
// ---------------------------------------------------------------------------

/// What one restore did, for the CLI's report and the tests'
/// assertions.
#[derive(Debug, Default)]
pub(crate) struct RestoreReport {
    pub(crate) generation: u64,
    pub(crate) files: usize,
    pub(crate) lanes: usize,
    pub(crate) records: usize,
}

/// `taguru restore --out DIR [URL]`: materializes a data directory
/// from the bucket's newest COMPLETE generation. Exit codes follow the
/// house rule: 0 restored · 1 bucket unusable (no fence, no complete
/// generation, corrupt segments) · 2 usage error.
pub(crate) fn run(args: &[String]) -> i32 {
    let mut out: Option<PathBuf> = None;
    let mut url: Option<String> = None;
    let mut config: Option<PathBuf> = None;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                println!(
                    "usage: taguru restore --out DIR [--config FILE] [URL]\n\
                     \n\
                     Materialize a data directory from a replication bucket (the newest\n\
                     complete generation): every published file, plus each context's two\n\
                     log lanes reassembled from their shipped segments. The URL defaults\n\
                     to TAGURU_REPLICATE_URL; credentials ride each cloud's default chain.\n\
                     DIR must not already contain a data directory — restore refuses to\n\
                     mix two histories. Verify the result with: taguru inspect DIR"
                );
                return 0;
            }
            "--out" => match rest.next() {
                Some(path) if out.is_none() => out = Some(PathBuf::from(path)),
                Some(_) => crate::config::usage_error("--out given twice"),
                None => crate::config::usage_error("--out needs a directory path"),
            },
            "--config" => match rest.next() {
                Some(path) if config.is_none() => config = Some(PathBuf::from(path)),
                Some(_) => crate::config::usage_error("--config given twice"),
                None => crate::config::usage_error("--config needs a file path"),
            },
            flag if flag.starts_with('-') => {
                crate::config::usage_error(&format!("'restore' does not take '{flag}'"))
            }
            positional => {
                if url.replace(positional.to_string()).is_some() {
                    crate::config::usage_error(&format!(
                        "'restore' takes one optional URL, got '{positional}'"
                    ));
                }
            }
        }
    }
    let config = config.or_else(|| std::env::var("TAGURU_CONFIG").ok().map(PathBuf::from));
    if let Some(path) = &config {
        crate::config::load_config(path);
    }
    let Some(out) = out else {
        crate::config::usage_error("restore needs --out DIR (the directory to materialize)");
    };
    let Some(url) = url.or_else(|| std::env::var("TAGURU_REPLICATE_URL").ok()) else {
        crate::config::usage_error(
            "restore needs a bucket URL — pass one, or set TAGURU_REPLICATE_URL",
        );
    };

    // Refuse a target that already holds data: a restore layered over
    // an existing directory would interleave two histories — exactly
    // the corruption the fence exists to prevent bucket-side. A lone
    // `.taguru.lock` does not count as data — it is the empty leftover
    // of the lock below (an earlier restore that died before writing
    // anything), and the lock file itself never ships.
    match std::fs::read_dir(&out) {
        Ok(entries) => {
            let occupied = entries
                .filter_map(|entry| entry.ok())
                .any(|entry| entry.file_name() != ".taguru.lock");
            if occupied {
                eprintln!(
                    "taguru: restore: {} is not empty — restore refuses to mix histories; \
                     point --out at a new or empty directory",
                    out.display()
                );
                return 1;
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            if let Err(error) = std::fs::create_dir_all(&out) {
                eprintln!("taguru: restore: cannot create {}: {error}", out.display());
                return 1;
            }
        }
        Err(error) => {
            eprintln!("taguru: restore: cannot read {}: {error}", out.display());
            return 1;
        }
    }
    // Hold the same advisory lock every writer takes, for the whole
    // materialization: a `taguru serve` pointed at this directory
    // mid-restore would otherwise boot over a half-written family and
    // cache the holes as truth. Import already refuses to run beside a
    // live server through this exact lock; restore is a writer too.
    let _dir_lock = match crate::storage::lock_data_dir(&out) {
        Ok(lock) => lock,
        Err(error) => {
            eprintln!("taguru: restore: {error}");
            return 1;
        }
    };

    let (store, root) = match open_store(&url) {
        Ok(opened) => opened,
        Err(error) => {
            eprintln!("taguru: restore: {error}");
            return 2;
        }
    };
    // The CLI runs with no ambient runtime (same posture as import and
    // export); the store client needs one, so restore brings its own.
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("taguru: restore: cannot start an async runtime: {error}");
            return 1;
        }
    };
    match runtime.block_on(restore_into(store.as_ref(), &root, &out)) {
        Ok(report) => {
            println!(
                "restored generation {}: {} files, {} log lanes ({} records) into {}",
                report.generation,
                report.files,
                report.lanes,
                report.records,
                out.display()
            );
            println!("verify with: taguru inspect {}", out.display());
            0
        }
        Err(error) => {
            eprintln!("taguru: restore: {error}");
            1
        }
    }
}

/// The restore body: pick the newest complete generation, download
/// `files/*` verbatim, and reassemble each log lane's newest series.
pub(crate) async fn restore_into(
    store: &dyn ObjectStore,
    root: &StorePath,
    out: &FsPath,
) -> io::Result<RestoreReport> {
    let generation = newest_complete_generation(store, root).await?;
    let generation_root = gen_root(root, generation);
    let mut report = RestoreReport {
        generation,
        ..RestoreReport::default()
    };

    // files/* — verbatim, atomically (stage + rename via the same
    // helper the server writes with, so a crash mid-restore leaves
    // whole files or nothing, never a torn image). The grant store is
    // the one secret-bearing file and keeps its owner-only mode.
    let files_prefix = generation_root.clone().join("files");
    let names = list_names_under(store, &files_prefix).await?;
    for name in names {
        let key = files_prefix.clone().join(name.as_str());
        let bytes = fetch(store, &key).await?;
        let path = out.join(&name);
        if name == "oauth.json" {
            crate::storage::write_atomic_private(&path, &bytes)?;
        } else {
            crate::storage::write_atomic(&path, &bytes)?;
        }
        report.files += 1;
    }

    // wal/{lane}/ — newest series only, segments in order, each
    // verified record-by-record before any byte lands: shipping runs
    // the same check, so a mismatch here means the bucket rotted (or
    // was edited), and a restore that "mostly worked" would be worse
    // than one that says so.
    let wal_prefix = generation_root.clone().join("wal");
    for lane in list_names_under(store, &wal_prefix).await? {
        let lane_prefix = wal_prefix.clone().join(lane.as_str());
        let mut segments: Vec<(u64, u64, StorePath)> = Vec::new();
        let mut listing = store.list(Some(&lane_prefix));
        while let Some(meta) = listing.next().await {
            let meta =
                meta.map_err(|error| io::Error::other(format!("listing lane {lane}: {error}")))?;
            let Some(segment_file) = meta.location.filename() else {
                continue;
            };
            let Some((series, seg)) = parse_segment_name(segment_file) else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("lane {lane}: unrecognized segment object '{segment_file}'"),
                ));
            };
            segments.push((series, seg, meta.location));
        }
        let Some(&(newest_series, _, _)) = segments.iter().max() else {
            continue;
        };
        let mut series_segments: Vec<(u64, StorePath)> = segments
            .into_iter()
            .filter(|&(series, _, _)| series == newest_series)
            .map(|(_, seg, key)| (seg, key))
            .collect();
        series_segments.sort();
        let mut assembled = Vec::new();
        for (position, (seg, key)) in series_segments.iter().enumerate() {
            // Segment numbers are the shipper's cursor, one PUT each:
            // a hole means an object vanished, and the records it held
            // are acknowledged writes — refuse, never skip.
            if *seg != position as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "lane {lane} series {newest_series}: segment {position} is missing \
                         (found {seg}) — the bucket lost or dropped an object"
                    ),
                ));
            }
            assembled.extend_from_slice(&fetch(store, key).await?);
        }
        let records = crate::wal::shippable_records(&assembled).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("lane {lane} series {newest_series}: {error}"),
            )
        })?;
        report.records += records.len();
        crate::storage::write_atomic(&out.join(&lane), &assembled)?;
        report.lanes += 1;
    }
    Ok(report)
}

/// The newest generation whose baseline finished (`complete` exists).
/// A newer INCOMPLETE generation is normal — a writer that claimed
/// and is mid-baseline, or died there — and restoring it would hand
/// back a directory with holes; fall back to the newest complete one.
async fn newest_complete_generation(store: &dyn ObjectStore, root: &StorePath) -> io::Result<u64> {
    let fence_prefix = root.clone().join(FENCE_PREFIX);
    let mut generations = Vec::new();
    let mut listing = store.list(Some(&fence_prefix));
    while let Some(meta) = listing.next().await {
        let meta = meta.map_err(|error| io::Error::other(format!("listing the fence: {error}")))?;
        if let Some(name) = meta.location.filename()
            && let Ok(generation) = name.parse::<u64>()
        {
            generations.push(generation);
        }
    }
    if generations.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "no replication fence in the bucket — nothing was ever shipped here \
             (check the URL and its prefix)",
        ));
    }
    generations.sort_unstable_by(|a, b| b.cmp(a));
    for generation in generations {
        let marker = gen_root(root, generation).join(COMPLETE_MARKER);
        match store.head(&marker).await {
            Ok(_) => return Ok(generation),
            Err(object_store::Error::NotFound { .. }) => continue,
            Err(error) => {
                return Err(io::Error::other(format!(
                    "checking generation {generation}: {error}"
                )));
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "no complete generation in the bucket — every claimant died before finishing \
         its baseline sync; nothing here can restore a whole directory",
    ))
}

/// The distinct first-level names under `prefix` (file names under
/// `files/`, lane names under `wal/`), via delimited listing.
async fn list_names_under(store: &dyn ObjectStore, prefix: &StorePath) -> io::Result<Vec<String>> {
    let listing = store
        .list_with_delimiter(Some(prefix))
        .await
        .map_err(|error| io::Error::other(format!("listing {prefix}: {error}")))?;
    let mut names: Vec<String> = listing
        .objects
        .iter()
        .filter_map(|meta| meta.location.filename().map(String::from))
        .chain(
            listing
                .common_prefixes
                .iter()
                .filter_map(|p| p.filename().map(String::from)),
        )
        .collect();
    names.sort();
    names.dedup();
    Ok(names)
}

fn parse_segment_name(name: &str) -> Option<(u64, u64)> {
    let body = name.strip_suffix(".jsonl")?;
    let (series, seg) = body.split_once('-')?;
    Some((series.parse().ok()?, seg.parse().ok()?))
}

async fn fetch(store: &dyn ObjectStore, key: &StorePath) -> io::Result<Vec<u8>> {
    let result = store
        .get(key)
        .await
        .map_err(|error| io::Error::other(format!("downloading {key}: {error}")))?;
    let bytes = result
        .bytes()
        .await
        .map_err(|error| io::Error::other(format!("downloading {key}: {error}")))?;
    Ok(bytes.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wal::{self, WalOp};
    use object_store::memory::InMemory;

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("taguru-ship-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn state_for(dir: &FsPath) -> AppState {
        AppState::boot(dir.to_path_buf(), 64 * 1024 * 1024, None).unwrap()
    }

    fn associate(subject: &str) -> WalOp {
        WalOp::Associate(crate::registry::AssocOp {
            subject: subject.to_string(),
            label: "好き".to_string(),
            object: "りんご".to_string(),
            weight: 1.0,
            source: None,
            paragraph: None,
        })
    }

    async fn claimed(
        store: &Arc<InMemory>,
        dir: &FsPath,
        state: &AppState,
        progress: &Arc<ShipProgress>,
    ) -> Shipper {
        Shipper::claim(
            Arc::clone(store) as Arc<dyn ObjectStore>,
            StorePath::default(),
            dir.to_path_buf(),
            Arc::clone(progress),
            state.clone(),
        )
        .await
        .unwrap()
    }

    async fn read_object(store: &Arc<InMemory>, key: &str) -> Vec<u8> {
        fetch(
            store.as_ref() as &dyn ObjectStore,
            &StorePath::parse(key).unwrap(),
        )
        .await
        .unwrap()
    }

    async fn object_names(store: &Arc<InMemory>) -> Vec<String> {
        let mut names = Vec::new();
        let mut listing = (store.as_ref() as &dyn ObjectStore).list(None);
        while let Some(meta) = listing.next().await {
            names.push(meta.unwrap().location.to_string());
        }
        names.sort();
        names
    }

    #[tokio::test]
    async fn claims_are_monotonic_and_a_race_converges_on_distinct_generations() {
        let dir = scratch_dir("claim");
        let state = state_for(&dir);
        let progress = Arc::new(ShipProgress::new());
        let store = Arc::new(InMemory::new());

        let first = claimed(&store, &dir, &state, &progress).await;
        let second = claimed(&store, &dir, &state, &progress).await;
        assert_eq!(first.generation, 1);
        assert_eq!(
            second.generation, 2,
            "a second claimant must outbid the first"
        );

        // Two claimants racing one generation: whoever loses the
        // conditional create bids one higher — both end up holding a
        // generation, and they are never equal.
        let (a, b) = tokio::join!(
            claimed(&store, &dir, &state, &progress),
            claimed(&store, &dir, &state, &progress),
        );
        assert_ne!(a.generation, b.generation);
        assert!(a.generation > 2 && b.generation > 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn ships_files_and_lane_segments_and_restore_round_trips() {
        let dir = scratch_dir("roundtrip");
        let state = state_for(&dir);
        let progress = Arc::new(ShipProgress::new());
        let store = Arc::new(InMemory::new());

        // A hand-laid family: an image stand-in and a real WAL written
        // by the real appender. The shipper reads the directory, not
        // the registry, so files are the whole fixture.
        std::fs::write(dir.join("ctx_a.ctx"), b"image-v1").unwrap();
        std::fs::write(dir.join("ctx_a.meta.json"), b"{}").unwrap();
        let wal_path = dir.join("ctx_a.wal.jsonl");
        wal::append_batch(&wal_path, 1, &[associate("a"), associate("b")]).unwrap();

        let mut shipper = claimed(&store, &dir, &state, &progress).await;
        assert!(
            shipper.cycle().await.unwrap(),
            "first cycle ships the baseline"
        );

        let names = object_names(&store).await;
        assert!(
            names.contains(&"fence/00000000000000000001".to_string()),
            "{names:?}"
        );
        assert!(names.contains(&"gen-00000000000000000001/complete".to_string()));
        assert!(names.contains(&"gen-00000000000000000001/files/ctx_a.ctx".to_string()));
        assert!(
            names.contains(
                &"gen-00000000000000000001/wal/ctx_a.wal.jsonl/0000000000-0000000000.jsonl"
                    .to_string()
            ),
            "{names:?}"
        );

        // A quiet cycle ships nothing and stays quiet.
        assert!(!shipper.cycle().await.unwrap());

        // More appends become the next segment of the same series.
        wal::append_batch(&wal_path, 3, &[associate("c")]).unwrap();
        assert!(shipper.cycle().await.unwrap());
        let segment = read_object(
            &store,
            "gen-00000000000000000001/wal/ctx_a.wal.jsonl/0000000000-0000000001.jsonl",
        )
        .await;
        assert!(segment.ends_with(b"\n"));

        // The concatenated segments are byte-identical to the local
        // log — the restore-equivalence property at its smallest.
        let restored_dir = scratch_dir("roundtrip-out");
        let report = restore_into(
            store.as_ref() as &dyn ObjectStore,
            &StorePath::default(),
            &restored_dir,
        )
        .await
        .unwrap();
        assert_eq!(report.generation, 1);
        assert_eq!(report.lanes, 1);
        assert_eq!(report.records, 3);
        assert_eq!(
            std::fs::read(restored_dir.join("ctx_a.wal.jsonl")).unwrap(),
            std::fs::read(&wal_path).unwrap(),
        );
        assert_eq!(
            std::fs::read(restored_dir.join("ctx_a.ctx")).unwrap(),
            b"image-v1"
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&restored_dir);
    }

    #[tokio::test]
    async fn a_reset_lane_restarts_its_series_behind_a_fresh_parent_snapshot() {
        let dir = scratch_dir("series");
        let state = state_for(&dir);
        let progress = Arc::new(ShipProgress::new());
        let store = Arc::new(InMemory::new());

        std::fs::write(dir.join("ctx_a.ctx"), b"image-v1").unwrap();
        let wal_path = dir.join("ctx_a.wal.jsonl");
        wal::append_batch(&wal_path, 1, &[associate("a"), associate("b")]).unwrap();

        let mut shipper = claimed(&store, &dir, &state, &progress).await;
        shipper.cycle().await.unwrap();

        // The flush shape: a fresh image lands, the log resets, new
        // appends continue at higher seqs.
        std::fs::write(dir.join("ctx_a.ctx"), b"image-v2-covers-seq-2").unwrap();
        wal::reset(&wal_path).unwrap();
        wal::append_batch(&wal_path, 3, &[associate("c")]).unwrap();
        shipper.cycle().await.unwrap();

        let names = object_names(&store).await;
        assert!(
            names.contains(
                &"gen-00000000000000000001/wal/ctx_a.wal.jsonl/0000000001-0000000000.jsonl"
                    .to_string()
            ),
            "series must restart after a reset: {names:?}"
        );
        // The restored lane is exactly the new series — the old
        // series' records are covered by the (also restored) parent.
        let restored_dir = scratch_dir("series-out");
        restore_into(
            store.as_ref() as &dyn ObjectStore,
            &StorePath::default(),
            &restored_dir,
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read(restored_dir.join("ctx_a.wal.jsonl")).unwrap(),
            std::fs::read(&wal_path).unwrap(),
        );
        assert_eq!(
            std::fs::read(restored_dir.join("ctx_a.ctx")).unwrap(),
            b"image-v2-covers-seq-2",
            "the parent snapshot must ship before (and so survive into) the new series"
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&restored_dir);
    }

    #[tokio::test]
    async fn a_rewritten_prefix_diverges_the_lane_and_restarts_its_series() {
        let dir = scratch_dir("rewrite");
        let state = state_for(&dir);
        let progress = Arc::new(ShipProgress::new());
        let store = Arc::new(InMemory::new());

        std::fs::write(dir.join("ctx_a.ctx"), b"image-v1").unwrap();
        let wal_path = dir.join("ctx_a.wal.jsonl");
        wal::append_batch(&wal_path, 1, &[associate("a"), associate("b")]).unwrap();

        let mut shipper = claimed(&store, &dir, &state, &progress).await;
        shipper.cycle().await.unwrap();

        // A rollback's shape: the file rewinds past the shipped
        // offset, then different bytes grow over the same offsets
        // (same seqs, different ops — the un-acknowledged batch was
        // replaced by the write that actually happened).
        wal::truncate_to(&wal_path, 0).unwrap();
        wal::append_batch(
            &wal_path,
            1,
            &[associate("x"), associate("y"), associate("z")],
        )
        .unwrap();
        shipper.cycle().await.unwrap();

        let restored_dir = scratch_dir("rewrite-out");
        restore_into(
            store.as_ref() as &dyn ObjectStore,
            &StorePath::default(),
            &restored_dir,
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read(restored_dir.join("ctx_a.wal.jsonl")).unwrap(),
            std::fs::read(&wal_path).unwrap(),
            "the restored lane must be the rewritten history, not the shipped-then-rolled-back one"
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&restored_dir);
    }

    #[tokio::test]
    async fn a_newer_claim_fences_the_shipper_on_its_next_dirty_cycle() {
        let dir = scratch_dir("fence");
        let state = state_for(&dir);
        let progress = Arc::new(ShipProgress::new());
        let store = Arc::new(InMemory::new());

        std::fs::write(dir.join("ctx_a.ctx"), b"image-v1").unwrap();
        let mut shipper = claimed(&store, &dir, &state, &progress).await;
        shipper.cycle().await.unwrap();

        // A second writer claims the bucket out from under us.
        let usurper = claimed(&store, &dir, &state, &progress).await;
        assert_eq!(usurper.generation, 2);

        // An idle cycle stays quiet — deposition matters only when
        // there is something to ship.
        assert!(!shipper.cycle().await.unwrap());

        // A dirty cycle discovers the fence and fail-stops.
        std::fs::write(dir.join("ctx_a.ctx"), b"image-v2").unwrap();
        match shipper.cycle().await {
            Err(ShipError::Fenced { newer_generation }) => assert_eq!(newer_generation, 2),
            other => panic!("expected Fenced, got {other:?}"),
        }
        // Nothing of v2 leaked into the bucket.
        let names = object_names(&store).await;
        assert_eq!(
            read_object(&store, "gen-00000000000000000001/files/ctx_a.ctx").await,
            b"image-v1",
            "{names:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn a_vanished_family_is_retired_remotely_including_its_segments() {
        let dir = scratch_dir("retire");
        let state = state_for(&dir);
        let progress = Arc::new(ShipProgress::new());
        let store = Arc::new(InMemory::new());

        std::fs::write(dir.join("ctx_a.ctx"), b"image-v1").unwrap();
        let wal_path = dir.join("ctx_a.wal.jsonl");
        wal::append_batch(&wal_path, 1, &[associate("a")]).unwrap();
        let mut shipper = claimed(&store, &dir, &state, &progress).await;
        shipper.cycle().await.unwrap();

        std::fs::remove_file(dir.join("ctx_a.ctx")).unwrap();
        std::fs::remove_file(&wal_path).unwrap();
        shipper.cycle().await.unwrap();

        let names = object_names(&store).await;
        assert!(
            !names.iter().any(|name| name.contains("ctx_a")),
            "every trace of the deleted family must leave the bucket: {names:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn restore_refuses_a_gapped_segment_run() {
        let dir = scratch_dir("gap");
        let state = state_for(&dir);
        let progress = Arc::new(ShipProgress::new());
        let store = Arc::new(InMemory::new());

        std::fs::write(dir.join("ctx_a.ctx"), b"image-v1").unwrap();
        let wal_path = dir.join("ctx_a.wal.jsonl");
        let mut shipper = claimed(&store, &dir, &state, &progress).await;
        for seq in 1..=3u64 {
            wal::append_batch(&wal_path, seq, &[associate("a")]).unwrap();
            shipper.cycle().await.unwrap();
        }
        (store.as_ref() as &dyn ObjectStore)
            .delete(
                &StorePath::parse(
                    "gen-00000000000000000001/wal/ctx_a.wal.jsonl/0000000000-0000000001.jsonl",
                )
                .unwrap(),
            )
            .await
            .unwrap();

        let restored_dir = scratch_dir("gap-out");
        let error = restore_into(
            store.as_ref() as &dyn ObjectStore,
            &StorePath::default(),
            &restored_dir,
        )
        .await
        .expect_err("a hole in the segment run holds acknowledged writes — refuse");
        assert!(error.to_string().contains("segment"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&restored_dir);
    }

    #[tokio::test]
    async fn restore_picks_the_newest_complete_generation_not_the_newest_claim() {
        let dir = scratch_dir("newest");
        let state = state_for(&dir);
        let progress = Arc::new(ShipProgress::new());
        let store = Arc::new(InMemory::new());

        std::fs::write(dir.join("ctx_a.ctx"), b"gen1-image").unwrap();
        let mut shipper = claimed(&store, &dir, &state, &progress).await;
        shipper.cycle().await.unwrap();

        // A second claimant exists but never finished its baseline —
        // its generation must not shadow the restorable one.
        let _mid_claim = claimed(&store, &dir, &state, &progress).await;

        let restored_dir = scratch_dir("newest-out");
        let report = restore_into(
            store.as_ref() as &dyn ObjectStore,
            &StorePath::default(),
            &restored_dir,
        )
        .await
        .unwrap();
        assert_eq!(report.generation, 1);
        assert_eq!(
            std::fs::read(restored_dir.join("ctx_a.ctx")).unwrap(),
            b"gen1-image"
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&restored_dir);
    }

    #[test]
    fn allows_reset_defers_until_shipped_and_caps_the_deferral() {
        let progress = ShipProgress::new();
        let log = FsPath::new("/data/x.wal.jsonl");

        // Nothing shipped yet: defer (the shipper will get there).
        assert!(!progress.allows_reset(log, 5, 1024));
        // Shipped past the watermark: reset freely.
        progress.note_shipped(log, 5);
        assert!(progress.allows_reset(log, 5, 1024));
        assert!(!progress.allows_reset(log, 9, 1024));
        // Past the deferral budget the reset proceeds regardless — a
        // dead bucket must never walk the log into its cap.
        assert!(progress.allows_reset(log, 9, DEFAULT_DEFER_CAP_BYTES));
        // A forgotten lane defers again from scratch.
        progress.forget(log);
        assert!(!progress.allows_reset(log, 1, 0));
        // A zero watermark has nothing to lose.
        assert!(progress.allows_reset(log, 0, 0));
    }

    #[test]
    fn classification_and_lane_parents_agree_with_the_family_layout() {
        assert_eq!(classify(".taguru.lock"), EntryKind::Skip);
        assert_eq!(classify("x.ctx.tmp42"), EntryKind::Skip);
        assert_eq!(classify("x.ctx"), EntryKind::Published);
        assert_eq!(classify("x.meta.json"), EntryKind::Published);
        assert_eq!(classify("x.wal.jsonl"), EntryKind::LogLane);
        assert_eq!(classify("x.passages.wal.jsonl"), EntryKind::LogLane);
        assert_eq!(classify("oauth.json"), EntryKind::Published);
        assert_eq!(classify("x.deleted"), EntryKind::Published);

        assert_eq!(parent_snapshot_of("x.wal.jsonl").unwrap(), "x.ctx");
        assert_eq!(
            parent_snapshot_of("x.passages.wal.jsonl").unwrap(),
            "x.passages.bin"
        );
        assert_eq!(parent_snapshot_of("x.ctx"), None);

        assert_eq!(
            parse_segment_name("0000000001-0000000002.jsonl"),
            Some((1, 2))
        );
        assert_eq!(parse_segment_name(&segment_name(3, 4)), Some((3, 4)));
        assert_eq!(parse_segment_name("junk"), None);
    }

    /// The restore-equivalence property, generated: any interleaving
    /// of appends, ship cycles, and flush-shaped resets must leave the
    /// bucket restorable to exactly the acknowledged state — the
    /// export/import fixed-point analog issue #127 asks for, phrased
    /// at the lane level where this module's correctness argument
    /// lives. The parent snapshot is a stand-in whose bytes ARE the
    /// watermark, so the assertion can replay the restored lane over
    /// the restored snapshot's watermark exactly as a real boot
    /// replays a real image.
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(crate::context_proptest::config())]

            #[test]
            fn any_ship_reset_interleaving_restores_the_acknowledged_suffix(
                generated in proptest::collection::vec(
                    crate::context_proptest::wal_op_strategy(),
                    1..16,
                ),
                schedule in proptest::collection::vec((any::<bool>(), any::<bool>()), 16),
            ) {
                let ops: Vec<WalOp> = generated.into_iter().map(WalOp::from).collect();
                let dir = scratch_dir("prop");
                let state = state_for(&dir);
                let progress = Arc::new(ShipProgress::new());
                let store = Arc::new(InMemory::new());
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap();

                let ctx_path = dir.join("ctx_a.ctx");
                let wal_path = dir.join("ctx_a.wal.jsonl");
                let mut watermark = 0u64;
                std::fs::write(&ctx_path, watermark.to_le_bytes()).unwrap();
                let mut shipper =
                    runtime.block_on(claimed(&store, &dir, &state, &progress));

                for (index, op) in ops.iter().enumerate() {
                    let seq = index as u64 + 1;
                    wal::append_batch(&wal_path, seq, std::slice::from_ref(op)).unwrap();
                    let (ship, flush) = schedule[index];
                    if ship {
                        runtime.block_on(shipper.cycle()).unwrap();
                    }
                    if flush {
                        // The flush shape: the image (stand-in) bakes
                        // everything in, then the log resets — with or
                        // without the shipper having read it first.
                        watermark = seq;
                        std::fs::write(&ctx_path, watermark.to_le_bytes()).unwrap();
                        wal::reset(&wal_path).unwrap();
                    }
                }
                runtime.block_on(shipper.cycle()).unwrap();

                let restored_dir = scratch_dir("prop-out");
                runtime
                    .block_on(restore_into(
                        store.as_ref() as &dyn ObjectStore,
                        &StorePath::default(),
                        &restored_dir,
                    ))
                    .unwrap();

                let restored_watermark = u64::from_le_bytes(
                    std::fs::read(restored_dir.join("ctx_a.ctx"))
                        .unwrap()
                        .try_into()
                        .expect("the stand-in image is exactly its watermark"),
                );
                let (restored_ops, top) = wal::replay::<WalOp>(
                    &restored_dir.join("ctx_a.wal.jsonl"),
                    restored_watermark,
                )
                .unwrap();
                // Replaying the restored lane over the restored
                // snapshot's watermark yields exactly the acknowledged
                // suffix: nothing doubled, nothing skipped, ending at
                // the newest acknowledged write.
                prop_assert_eq!(
                    &restored_ops[..],
                    &ops[restored_watermark as usize..],
                    "restored replay must be the suffix past watermark {}",
                    restored_watermark
                );
                prop_assert_eq!(top, ops.len() as u64);

                let _ = std::fs::remove_dir_all(&dir);
                let _ = std::fs::remove_dir_all(&restored_dir);
            }
        }
    }
}
