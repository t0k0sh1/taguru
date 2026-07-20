//! Boot from the bucket (issue #128): with `TAGURU_REPLICATE_URL`
//! set, an empty data directory hydrates itself from the bucket's
//! newest complete generation instead of demanding a hand-restored
//! volume — the volume demotes to a CACHE of the bucket lineage, and
//! recovery becomes "start anywhere". Hydration is lazy and
//! priority-ordered: the shared files (groups, the grant store, every
//! context's sidecar meta) land before boot so the registry can
//! enumerate and describe everything; pinned contexts hydrate eagerly
//! through the ordinary parallel preload; the rest hydrate on first
//! touch — or via the background fill, whichever reaches a context
//! first. A local file whose BYTES already match the manifest is
//! reused without a download, which is what keeps warm restarts of a
//! cache-mode volume cheap; anything stale, torn, or foreign to the
//! lineage is refetched or removed, never trusted.
//!
//! # Which boots hydrate
//!
//! Decided once, at startup, from two local facts and one bucket look:
//!
//! - The directory's [`ship::ReplicationRecord`] says whether ITS
//!   writer holds the bucket's newest generation (then local disk IS
//!   the lineage's tip — boot exactly as before, no verification), and
//!   whether the directory began life as a bucket materialization
//!   (`hydrated_from`, "cache mode" — then any OTHER newest generation
//!   means the lineage moved on without us, and local files are
//!   re-verified against the manifest instead of trusted).
//! - A directory with real data and NO cache marker never hydrates:
//!   that is the pre-#128 posture — local disk is the truth, restarting
//!   a fenced writer contests the bucket rather than submitting to it —
//!   and it stays intact for every existing deployment.
//! - An (essentially) empty directory hydrates whenever the bucket has
//!   a complete generation to offer.
//!
//! # The takeover guard
//!
//! Booting from the bucket is exactly the change that makes a
//! concurrent-writer accident PHYSICALLY possible — two writers no
//! longer need the same disk, so the volume stops being the mutex and
//! the generation claim from #127 replaces it. Claiming deposes the
//! previous writer's shipping; a misconfigured second instance must
//! not do that casually. So: when the bucket's newest generation is
//! not [retired](ship::RETIRED_MARKER) and showed life (its fence
//! claim or last [heartbeat](ship::HEARTBEAT_MARKER)) within
//! [`ship::TAKEOVER_GRACE`], a boot that is NOT that generation's own
//! writer refuses to start unless the operator states the intent —
//! `--take-over` / `TAGURU_TAKEOVER=1`. Starting a writer against a
//! bucket IS the promotion act; the guard just makes it a stated one.
//! It is ergonomics, not correctness: claims race, clocks skew, and
//! none of it matters because the fence stays the only arbiter — a
//! writer that slips past the guard still deposes cleanly, loudly,
//! through the fence.
//!
//! What a takeover costs is stated here once, honestly: the deposed
//! writer's un-shipped tail (its shipping lag, plus anything a dead
//! bucket kept it from shipping) exists only on its own volume. A
//! successor hydrating elsewhere serves the lineage WITHOUT that tail;
//! reusing the very same volume in cache mode keeps a tail whose
//! shipped prefix still matches, and discards one that diverged.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io;
use std::path::{Path as FsPath, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime};

use object_store::ObjectStore;
use object_store::path::Path as StorePath;

use crate::ship::{self, FileSig, Manifest, ManifestFile, ManifestLane};

/// How many manifest re-reads a fetch spends arbitrating a
/// verification mismatch before calling it rot, and the pause between
/// them. A live lineage legitimately advances under a reader — the
/// writer replaces an object, then refreshes the manifest — and the
/// gap between those two uploads is the window a fetch can fall into;
/// a few paced re-reads outlast it, while true rot (bytes disagreeing
/// with a manifest that is NOT moving) still refuses. Sized against
/// the shipping cadence, not the poll interval: one cycle's uploads
/// finish well inside these rounds.
const FETCH_REFRESH_ROUNDS: usize = 4;
const FETCH_REFRESH_PAUSE: Duration = Duration::from_millis(150);

/// Decides, once per boot, how this data directory relates to the
/// bucket: `None` — boot from local disk exactly as before (with the
/// shipper claiming the next generation as usual); `Some` — the
/// directory is (or becomes) a cache of the bucket lineage, the
/// returned hydrator has already landed the shared files, and boot
/// should register its contexts and hydrate the rest lazily. An error
/// refuses the boot: either the takeover guard (see the module doc)
/// or a bucket that cannot be verified while local disk has no
/// independent truth to serve.
pub(crate) async fn prepare(
    store: &Arc<dyn ObjectStore>,
    root: &StorePath,
    url: &str,
    data_dir: &FsPath,
    take_over: bool,
) -> io::Result<Option<Arc<Hydrator>>> {
    let record = ship::read_replication_record(data_dir).filter(|record| {
        if record.url != url {
            tracing::warn!(
                recorded = %record.url,
                configured = %url,
                "the data directory's replication record names a different bucket; \
                 treating this directory as new to the configured one"
            );
            return false;
        }
        true
    });
    let cache_mode = record
        .as_ref()
        .is_some_and(|record| record.hydrated_from.is_some());
    let empty = essentially_empty(data_dir)?;

    let fence = match ship::newest_fence(store, root).await {
        Ok(fence) => fence,
        Err(error) => {
            // Unreachable bucket. A directory with its own truth boots
            // degraded (the shipper retries in the background, as
            // ever); a directory with none — empty, or a cache of a
            // lineage it cannot re-verify — must not invent one.
            if !empty && !cache_mode {
                tracing::warn!(
                    error = %error,
                    "replication bucket unreachable at boot; continuing on local truth"
                );
                return Ok(None);
            }
            return Err(io::Error::other(format!(
                "cannot reach the replication bucket, and this data directory has no \
                 independent truth to serve ({}): {error}",
                if empty {
                    "it is empty"
                } else {
                    "it is a cache of the bucket lineage"
                }
            )));
        }
    };

    let Some(fence) = fence else {
        // A virgin bucket: nothing to hydrate from, nothing to depose.
        return Ok(None);
    };

    if record.as_ref().and_then(|record| record.claimed_generation) == Some(fence.generation) {
        // The bucket's newest writer was this very directory: local
        // disk is the lineage's tip (plus any un-shipped tail, which
        // WAL replay recovers) — the warm restart. No guard, no
        // verification, no downloads.
        return Ok(None);
    }

    if !take_over
        && let Some(live) = liveness(store, root, fence).await?
        && live < ship::TAKEOVER_GRACE
    {
        return Err(io::Error::other(format!(
            "the bucket's newest generation ({}) showed life {}s ago and is not retired — \
             a live writer may still own it, and starting this server would depose it \
             (starting a writer against a bucket IS the takeover). If that is intended, \
             pass --take-over or set TAGURU_TAKEOVER=1; a cleanly stopped writer never \
             trips this, and a crashed one ages out of it in {}s",
            fence.generation,
            live.as_secs(),
            ship::TAKEOVER_GRACE.as_secs(),
        )));
    }

    if !empty && !cache_mode {
        // Real data, no cache marker: the pre-#128 posture, preserved —
        // local disk is the truth and the claim will contest the
        // bucket, not submit to it.
        return Ok(None);
    }

    let generation = match ship::newest_complete_generation(store.as_ref(), root).await {
        Ok(generation) => generation,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            // Claims exist but no baseline ever finished: there is no
            // lineage to inherit. An empty directory starting fresh is
            // the honest outcome.
            tracing::warn!(
                error = %error,
                "the bucket has no complete generation; starting fresh from local disk"
            );
            return Ok(None);
        }
        Err(error) => return Err(error),
    };
    let generation_root = ship::gen_root(root, generation);
    let Some(manifest) = ship::read_manifest(store.as_ref(), &generation_root).await? else {
        return Err(io::Error::other(format!(
            "generation {generation} predates the shipping manifest, so a lazy boot \
             cannot verify local files against it — materialize it explicitly with \
             `taguru restore --out DIR` and boot from that, or let a current writer \
             ship once to refresh the bucket"
        )));
    };

    // From here the directory IS a cache of the lineage: create it if
    // this is its very first boot, then record cache mode durably
    // before touching anything else, so a crash mid-hydration resumes
    // as cache mode (re-verify everything) instead of booting a
    // half-materialized directory as truth.
    std::fs::create_dir_all(data_dir)?;
    ship::write_replication_record(
        data_dir,
        &ship::ReplicationRecord {
            url: url.to_string(),
            claimed_generation: record.as_ref().and_then(|record| record.claimed_generation),
            hydrated_from: Some(generation),
        },
    )?;

    let hydrator = Arc::new(Hydrator::new(
        Arc::clone(store),
        generation,
        generation_root,
        data_dir.to_path_buf(),
        manifest,
        LanePolicy::KeepAckedTail,
    ));
    let report = hydrator.hydrate_shared().await?;
    tracing::info!(
        generation,
        contexts = hydrator.context_stems().len(),
        shared_fetched = report.fetched,
        shared_reused = report.reused,
        removed = report.removed,
        "hydrating from the bucket — shared files landed; context families \
         hydrate on preload, first touch, or the background fill",
    );
    Ok(Some(hydrator))
}

/// The replica's boot decision — [`prepare`]'s sibling with the writer
/// posture stripped out: a replica never claims a generation, never
/// heartbeats, never deposes anyone (so the takeover guard does not
/// apply), and its directory is never "local truth" — it is a cache of
/// the bucket lineage, always re-verified, with any writer-role log
/// tail truncated ([`LanePolicy::ShippedExact`]). What remains of the
/// table:
///
/// - A directory with real data and NO replication record is refused:
///   it is somebody's truth (a pre-replication writer, a restore), and
///   a replica would verify-and-replace its files against the bucket.
///   Deliberately strict — the operator points a replica at an empty
///   directory, or at that replica's own previous cache.
/// - A directory whose record claims the bucket's NEWEST generation is
///   refused: that is the current writer's directory, and demoting it
///   would discard its un-shipped tail — data with no other home. A
///   FORMER writer (its claim deposed by a newer one) demotes freely;
///   its divergent tail is already lost to the lineage, and truncating
///   it merely says so.
/// - An unreachable bucket degrades a cache to serving its last
///   watermark (the tailer keeps trying), refuses an empty directory
///   (nothing to serve), and a virgin bucket serves empty until its
///   first complete generation appears — the tailer provisions the
///   target the moment one does.
pub(crate) async fn prepare_replica(
    store: &Arc<dyn ObjectStore>,
    root: &StorePath,
    url: &str,
    data_dir: &FsPath,
) -> io::Result<Arc<Hydrator>> {
    let record = ship::read_replication_record(data_dir).filter(|record| {
        if record.url != url {
            tracing::warn!(
                recorded = %record.url,
                configured = %url,
                "the data directory's replication record names a different bucket; \
                 treating this directory as new to the configured one"
            );
            return false;
        }
        true
    });
    let empty = essentially_empty(data_dir)?;
    if !empty && record.is_none() {
        return Err(io::Error::other(
            "this data directory holds data that is not a cache of the configured \
             bucket — a replica serves the bucket lineage and would verify (and \
             replace) local files against it. Point the replica at an empty \
             directory, or at its own previous cache; a writer's directory keeps \
             its meaning only under `serve` without --replica",
        ));
    }

    let unprovisioned = || {
        Arc::new(Hydrator::unprovisioned(
            Arc::clone(store),
            data_dir.to_path_buf(),
            LanePolicy::ShippedExact,
        ))
    };

    let fence = match ship::newest_fence(store, root).await {
        Ok(fence) => fence,
        Err(error) => {
            // Degraded boot serves only a VERIFIED cache: a directory
            // whose record carries `hydrated_from` completed a shared
            // hydration against that generation once (the second half
            // of the two-phase record write below). A directory that
            // never got that far — empty, or mid-first-conversion when
            // it crashed — has nothing this replica can honestly
            // serve, and a demoted writer's un-shipped tail must never
            // slip out through that gap.
            if !empty
                && record
                    .as_ref()
                    .is_some_and(|record| record.hydrated_from.is_some())
            {
                // A cache with local files serves its last watermark
                // while the tailer keeps knocking — read availability
                // is the whole point of a replica.
                tracing::warn!(
                    error = %error,
                    "replication bucket unreachable at boot; serving the cache's \
                     last watermark until the tailer reaches it"
                );
                return Ok(unprovisioned());
            }
            return Err(io::Error::other(format!(
                "cannot reach the replication bucket, and this replica has nothing \
                 to serve yet ({}): {error}",
                if empty {
                    "its directory is empty"
                } else {
                    "its hydration never completed, so it is not yet a verified cache"
                }
            )));
        }
    };

    if let Some(fence) = fence
        && record.as_ref().and_then(|record| record.claimed_generation) == Some(fence.generation)
    {
        return Err(io::Error::other(format!(
            "this directory is the bucket's newest writer (generation {}): a replica \
             would demote it to a cache and truncate its un-shipped log tail — data \
             with no other home. Start it with `serve` (as the writer), or point the \
             replica at a different directory",
            fence.generation,
        )));
    }

    let generation = match ship::newest_complete_generation(store.as_ref(), root).await {
        Ok(generation) => generation,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            // A virgin bucket, or claimants that all died mid-baseline:
            // nothing to tail yet. Serve what the cache holds (nothing,
            // for an empty directory) and let the tailer provision the
            // target when the first complete generation lands.
            tracing::info!(
                error = %error,
                "no complete generation to replicate yet; serving until one appears"
            );
            std::fs::create_dir_all(data_dir)?;
            ship::write_replication_record(
                data_dir,
                &ship::ReplicationRecord {
                    url: url.to_string(),
                    claimed_generation: None,
                    hydrated_from: record.as_ref().and_then(|record| record.hydrated_from),
                },
            )?;
            return Ok(unprovisioned());
        }
        Err(error) => return Err(error),
    };
    let generation_root = ship::gen_root(root, generation);
    let Some(manifest) = ship::read_manifest(store.as_ref(), &generation_root).await? else {
        return Err(io::Error::other(format!(
            "generation {generation} predates the shipping manifest, so a replica \
             cannot verify or tail it — let a current writer ship once to refresh \
             the bucket"
        )));
    };

    // From here the directory IS a cache of the lineage — and never a
    // writer's truth again until a writer-mode boot claims over it.
    // The record is written in TWO phases, deliberately inverted from
    // the writer path's write-first order (there, an early record
    // makes a crashed half-hydration re-verify instead of booting as
    // truth; here, degraded boot TRUSTS the record, so it must not
    // overpromise): phase one drops any stale claim (the newest-writer
    // refusal above is why none can be current) and marks the
    // directory as this URL's replica workspace — keeping whatever
    // generation an EARLIER completed hydration verified — and only
    // after the shared hydration lands does phase two advance
    // `hydrated_from` to this generation. A crash in between leaves a
    // record that promises no more than what was actually verified:
    // a first conversion that never finished stays un-servable while
    // the bucket is unreachable, and a demoted writer's un-shipped
    // tail cannot leak through a half-converted directory.
    std::fs::create_dir_all(data_dir)?;
    ship::write_replication_record(
        data_dir,
        &ship::ReplicationRecord {
            url: url.to_string(),
            claimed_generation: None,
            hydrated_from: record.as_ref().and_then(|record| record.hydrated_from),
        },
    )?;

    let hydrator = Arc::new(Hydrator::new(
        Arc::clone(store),
        generation,
        generation_root,
        data_dir.to_path_buf(),
        manifest,
        LanePolicy::ShippedExact,
    ));
    let report = hydrator.hydrate_shared().await?;
    ship::write_replication_record(
        data_dir,
        &ship::ReplicationRecord {
            url: url.to_string(),
            claimed_generation: None,
            hydrated_from: Some(generation),
        },
    )?;
    tracing::info!(
        generation,
        contexts = hydrator.context_stems().len(),
        shared_fetched = report.fetched,
        shared_reused = report.reused,
        removed = report.removed,
        "replica hydrating from the bucket — shared files landed; context \
         families hydrate on preload, first touch, or the tailer",
    );
    Ok(hydrator)
}

/// How recently the generation showed life: the youngest of its fence
/// claim and its last heartbeat, as an age — `None` when the writer
/// retired cleanly (which is a statement, not an age).
async fn liveness(
    store: &Arc<dyn ObjectStore>,
    root: &StorePath,
    fence: ship::FenceInfo,
) -> io::Result<Option<Duration>> {
    let generation_root = ship::gen_root(root, fence.generation);
    if let Some(_retired) =
        head_modified(store, &generation_root.clone().join(ship::RETIRED_MARKER)).await?
    {
        return Ok(None);
    }
    let heartbeat =
        head_modified(store, &generation_root.clone().join(ship::HEARTBEAT_MARKER)).await?;
    let newest = [fence.claimed, heartbeat].into_iter().flatten().max();
    // A stamp the local clock claims is in the future reads as age
    // zero: skew must widen the guard, never disarm it.
    Ok(newest.map(|at| {
        SystemTime::now()
            .duration_since(at)
            .unwrap_or(Duration::ZERO)
    }))
}

/// The object's `last_modified` by the store's clock, or `None` when
/// it does not exist.
pub(crate) async fn head_modified(
    store: &Arc<dyn ObjectStore>,
    key: &StorePath,
) -> io::Result<Option<SystemTime>> {
    use object_store::ObjectStoreExt;
    match store.head(key).await {
        Ok(meta) => Ok(Some(SystemTime::from(meta.last_modified))),
        Err(object_store::Error::NotFound { .. }) => Ok(None),
        Err(error) => Err(io::Error::other(format!("checking {key}: {error}"))),
    }
}

/// Whether the directory holds nothing but the bookkeeping files a
/// boot may leave behind (`.taguru.lock`, the replication record). A
/// missing directory is empty; anything else — one real file — is not.
fn essentially_empty(data_dir: &FsPath) -> io::Result<bool> {
    match std::fs::read_dir(data_dir) {
        Ok(entries) => {
            for entry in entries {
                let entry = entry?;
                let name = entry.file_name();
                if name != ".taguru.lock" && name != ship::REPLICATION_RECORD {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(true),
        Err(error) => Err(error),
    }
}

/// What one shared-hydration pass did, for the boot log.
#[derive(Debug, Default)]
pub(crate) struct SharedReport {
    pub(crate) fetched: usize,
    pub(crate) reused: usize,
    pub(crate) removed: usize,
}

/// What one [`Hydrator::retarget`] touched — the replica tailer's
/// work list.
#[derive(Debug, Default)]
pub(crate) struct RetargetReport {
    /// Families whose local materialization no longer matches the new
    /// manifest (new families included): re-hydrate, then drop any
    /// loaded copy so the next read serves the new bytes.
    pub(crate) stale: Vec<String>,
    /// Families the new manifest no longer carries: their files are
    /// gone from the lineage, so deregister them (each reported once).
    pub(crate) vanished: Vec<String>,
}

/// How a log lane treats local bytes beyond the manifest's shipped
/// extent. A writer-side cache keeps them (they are acknowledged
/// records this very directory appended, and WAL replay is their
/// normal recovery); a replica truncates them (it serves the shipped
/// stream and nothing else — a record the lineage cannot back must
/// not surface in reads, and the writer role is where such a tail
/// belongs).
#[derive(Debug, Clone, Copy)]
pub(crate) enum LanePolicy {
    KeepAckedTail,
    ShippedExact,
}

/// The manifest slice one context family occupies: what a fetch needs,
/// and what a retarget compares to decide staleness.
#[derive(Debug, Clone, PartialEq, Default)]
struct FamilySig {
    files: BTreeMap<String, ManifestFile>,
    lanes: BTreeMap<String, ManifestLane>,
}

impl FamilySig {
    fn of(manifest: &Manifest, stem: &str) -> Self {
        let mut sig = Self::default();
        for name in crate::registry::context_files(stem) {
            if let Some(file) = manifest.files.get(&name) {
                sig.files.insert(name, *file);
            } else if let Some(lane) = manifest.lanes.get(&name) {
                sig.lanes.insert(name, *lane);
            }
        }
        sig
    }
}

/// Where one context family stands in this hydrator's current target.
#[derive(Debug, Clone, PartialEq)]
enum StemState {
    /// Not hydrated yet — first touch, the preload, the background
    /// fill, or the replica tailer will take it.
    Pending,
    /// Someone is hydrating it right now; wait, don't duplicate.
    InFlight,
    /// Locally materialized and verified (or reused) as this exact
    /// manifest slice — looked at again only if a retarget moves the
    /// slice out from under it.
    Done(FamilySig),
    /// Deleted or renamed away locally before hydration reached it —
    /// the bucket copy must NOT be re-materialized, or the vanished
    /// family would resurrect on the next boot. A later retarget may
    /// revive the stem if a newer manifest carries it again.
    Vetoed,
}

/// The generation a hydrator is currently materializing. The writer
/// path provisions it once at boot and never moves it; the replica
/// tailer swaps it on every manifest change ([`Hydrator::retarget`]).
/// `None` is a replica serving without a lineage yet — a virgin
/// bucket, or a cache booted while the bucket was unreachable — where
/// every local load passes through untouched until the first
/// successful poll provisions the target.
#[derive(Debug, Clone)]
struct Target {
    generation: u64,
    root: StorePath,
    manifest: Manifest,
}

#[derive(Debug)]
struct HydratorInner {
    target: Option<Target>,
    states: BTreeMap<String, StemState>,
}

/// The lazy half of a bucket boot, shared by the registry (first
/// touch, the pinned preload), the background fill, the shipper
/// (whose manifest write gates on [`Hydrator::drained`]) — and, in
/// replica mode, the tailer, which re-aims the same machine at every
/// newer manifest the bucket grows.
#[derive(Debug)]
pub(crate) struct Hydrator {
    store: Arc<dyn ObjectStore>,
    data_dir: PathBuf,
    lane_policy: LanePolicy,
    inner: Mutex<HydratorInner>,
    settled: Condvar,
    /// Lanes last confirmed to match the manifest, by name — a
    /// [`FileSig`] that still matches the local file means the bytes
    /// this cached [`ManifestLane`] verified are still on disk, so a
    /// repeat call (a sibling file in the same family changed and
    /// re-queued the whole stem) can skip the read-and-CRC entirely.
    /// Only ever populated from a stat taken before that verification
    /// ran, so a miss costs at most one redundant check — never a
    /// false hit.
    lane_cache: Mutex<HashMap<String, (ManifestLane, FileSig)>>,
    /// [`Self::fetch_published_if_stale`]'s analog of `lane_cache`,
    /// same safety argument: the whole bucket shares one manifest, so
    /// any family's shipping activity re-triggers `hydrate_shared`
    /// over every shared file, most of which did not change.
    file_cache: Mutex<HashMap<String, (ManifestFile, FileSig)>>,
}

impl Hydrator {
    fn new(
        store: Arc<dyn ObjectStore>,
        generation: u64,
        root: StorePath,
        data_dir: PathBuf,
        manifest: Manifest,
        lane_policy: LanePolicy,
    ) -> Self {
        let states = manifest
            .files
            .keys()
            .filter_map(|name| name.strip_suffix(".ctx"))
            .map(|stem| (stem.to_string(), StemState::Pending))
            .collect();
        Self {
            store,
            data_dir,
            lane_policy,
            inner: Mutex::new(HydratorInner {
                target: Some(Target {
                    generation,
                    root,
                    manifest,
                }),
                states,
            }),
            settled: Condvar::new(),
            lane_cache: Mutex::new(HashMap::new()),
            file_cache: Mutex::new(HashMap::new()),
        }
    }

    /// A hydrator with no lineage to materialize yet: every touch is a
    /// pass-through and local files serve as they are, until the first
    /// [`Self::retarget`] provisions a manifest. This is the replica's
    /// degraded boot (bucket unreachable, or nothing complete in it) —
    /// it exists so the load path's hydration hooks are in place from
    /// the very first request, not bolted on when the bucket appears.
    fn unprovisioned(
        store: Arc<dyn ObjectStore>,
        data_dir: PathBuf,
        lane_policy: LanePolicy,
    ) -> Self {
        Self {
            store,
            data_dir,
            lane_policy,
            inner: Mutex::new(HydratorInner {
                target: None,
                states: BTreeMap::new(),
            }),
            settled: Condvar::new(),
            lane_cache: Mutex::new(HashMap::new()),
            file_cache: Mutex::new(HashMap::new()),
        }
    }

    /// The generation currently being materialized (`None` until a
    /// degraded replica boot first reaches its bucket).
    pub(crate) fn generation(&self) -> Option<u64> {
        let inner = self.inner.lock().unwrap();
        inner.target.as_ref().map(|target| target.generation)
    }

    /// Every context stem the current target carries, for boot's
    /// registry registration — the sidecar metas these describe are
    /// already local by the time boot runs ([`Self::hydrate_shared`]).
    pub(crate) fn context_stems(&self) -> Vec<String> {
        self.inner.lock().unwrap().states.keys().cloned().collect()
    }

    /// Whether every family is settled (hydrated or vetoed): the
    /// shipper's gate for writing its own generation's manifest — a
    /// `complete` written earlier would crown a generation that lacks
    /// every family still waiting in the predecessor.
    pub(crate) fn drained(&self) -> bool {
        self.inner
            .lock()
            .unwrap()
            .states
            .values()
            .all(|state| matches!(state, StemState::Done(_) | StemState::Vetoed))
    }

    /// Marks a family off-limits before its local files are deleted or
    /// renamed away: hydration must never re-materialize what the
    /// server just removed. Waits out an in-flight hydration first so
    /// the caller's deletion cannot interleave with a half-landed
    /// download.
    pub(crate) fn veto(&self, stem: &str) {
        let mut inner = self.inner.lock().unwrap();
        while matches!(inner.states.get(stem), Some(StemState::InFlight)) {
            inner = self.settled.wait(inner).unwrap();
        }
        if inner.states.contains_key(stem) {
            inner.states.insert(stem.to_string(), StemState::Vetoed);
        }
    }

    /// Re-aims the hydrator at a newer manifest — the replica tailer's
    /// one verb. Families whose manifest slice moved (or that are new)
    /// go back to `Pending` and are reported as `stale`; families the
    /// new manifest no longer carries are vetoed and reported as
    /// `vanished`, exactly once. Waits out in-flight fetches first so
    /// nothing lands as `Done` against a target that just moved —
    /// fetches are bounded, and a poll-cadence caller can afford them.
    pub(crate) fn retarget(
        &self,
        generation: u64,
        root: StorePath,
        manifest: Manifest,
    ) -> RetargetReport {
        let mut report = RetargetReport::default();
        let mut inner = self.inner.lock().unwrap();
        while inner
            .states
            .values()
            .any(|state| matches!(state, StemState::InFlight))
        {
            inner = self.settled.wait(inner).unwrap();
        }
        let stems: BTreeSet<String> = manifest
            .files
            .keys()
            .filter_map(|name| name.strip_suffix(".ctx"))
            .map(str::to_string)
            .collect();
        for stem in &stems {
            let sig = FamilySig::of(&manifest, stem);
            let fresh =
                matches!(inner.states.get(stem), Some(StemState::Done(have)) if *have == sig);
            if !fresh {
                inner.states.insert(stem.clone(), StemState::Pending);
                report.stale.push(stem.clone());
            }
        }
        for (stem, state) in inner.states.iter_mut() {
            if !stems.contains(stem) && !matches!(state, StemState::Vetoed) {
                *state = StemState::Vetoed;
                report.vanished.push(stem.clone());
            }
        }
        inner.target = Some(Target {
            generation,
            root,
            manifest,
        });
        report
    }

    /// Materializes one context's family before its first load —
    /// synchronous, callable from any thread (each hydration runs on
    /// its own scoped worker thread with its own small runtime, so no
    /// caller ever block_on's inside an async worker). Families the
    /// manifest does not know (created after this boot) return
    /// immediately; concurrent callers of the same stem coalesce.
    pub(crate) fn ensure_context(&self, stem: &str) -> io::Result<()> {
        let (root, sig) = {
            let mut inner = self.inner.lock().unwrap();
            loop {
                match inner.states.get(stem) {
                    None | Some(StemState::Done(_) | StemState::Vetoed) => return Ok(()),
                    Some(StemState::InFlight) => inner = self.settled.wait(inner).unwrap(),
                    Some(StemState::Pending) => break,
                }
            }
            // Snapshot the target under the same lock that grants
            // InFlight: the fetch below runs against exactly this
            // slice even if a retarget swaps the manifest meanwhile.
            let target = inner
                .target
                .as_ref()
                .expect("a pending stem implies a provisioned target");
            let snapshot = (target.root.clone(), FamilySig::of(&target.manifest, stem));
            inner.states.insert(stem.to_string(), StemState::InFlight);
            snapshot
        };

        let outcome = std::thread::scope(|scope| {
            scope
                .spawn(|| -> io::Result<(usize, usize)> {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()?;
                    runtime.block_on(self.hydrate_family(stem, &root, &sig))
                })
                .join()
                // A worker panic must land in the ordinary error arm
                // below — re-panicking here would strand the stem
                // InFlight forever, deadlocking every thread parked on
                // `settled` (a veto, a coalescing touch, the fill).
                .unwrap_or_else(|_| {
                    Err(io::Error::other(
                        "the hydration worker panicked; the family will be retried",
                    ))
                })
        });

        let mut inner = self.inner.lock().unwrap();
        match &outcome {
            Ok((fetched, reused)) => {
                // A veto cannot have landed meanwhile — veto() waits
                // for InFlight to settle — so this InFlight is ours.
                // What landed is the SNAPSHOT's slice: Done only while
                // that is still the current one (retarget waits out
                // in-flight fetches, so a drift here is belt to its
                // suspenders — but a fetch that raced a swap must
                // re-run, not masquerade as the successor's bytes).
                let current = inner
                    .target
                    .as_ref()
                    .map(|target| FamilySig::of(&target.manifest, stem));
                let state = if current.as_ref() == Some(&sig) {
                    StemState::Done(sig.clone())
                } else {
                    StemState::Pending
                };
                inner.states.insert(stem.to_string(), state);
                tracing::info!(stem, fetched, reused, "context family hydrated");
            }
            Err(error) => {
                // Back to Pending: the next touch (or fill pass)
                // retries. The caller surfaces the error through the
                // same quarantine a failed local load uses.
                inner.states.insert(stem.to_string(), StemState::Pending);
                tracing::warn!(stem, %error, "context family hydration failed");
            }
        }
        drop(inner);
        self.settled.notify_all();
        outcome.map(|_| ())
    }

    /// Runs on its own thread until every family settles: whatever
    /// first touch and the preload have not reached yet, this does —
    /// so the window in which the bucket's only complete generation is
    /// the predecessor closes on its own, not only under traffic.
    /// Failures retry forever (a bucket outage ends someday); the
    /// thread exits when nothing is left to do.
    pub(crate) fn spawn_background_fill(self: &Arc<Self>) {
        let hydrator = Arc::clone(self);
        std::thread::Builder::new()
            .name("taguru-hydrate".into())
            .spawn(move || {
                let started = Instant::now();
                loop {
                    let pending: Vec<String> = {
                        let inner = hydrator.inner.lock().unwrap();
                        inner
                            .states
                            .iter()
                            .filter(|(_, state)| matches!(state, StemState::Pending))
                            .map(|(stem, _)| stem.clone())
                            .collect()
                    };
                    let mut failed = false;
                    for stem in pending {
                        failed |= hydrator.ensure_context(&stem).is_err();
                    }
                    if hydrator.drained() {
                        tracing::info!(
                            ms = started.elapsed().as_millis() as u64,
                            "background hydration complete — every family is local",
                        );
                        return;
                    }
                    // Not drained: retries pending after a failure, or
                    // another caller is mid-family. Errors were logged
                    // per family by ensure_context.
                    std::thread::sleep(if failed {
                        Duration::from_secs(30)
                    } else {
                        Duration::from_millis(200)
                    });
                }
            })
            .expect("spawning the hydration fill thread");
    }

    /// The eager half, run once before boot and again by the replica
    /// tailer on every manifest change: land every non-family file
    /// (groups, the grant store, crash markers) plus every family's
    /// sidecar meta — everything enumeration and description need —
    /// and remove local files the manifest does not know, which in
    /// cache mode are relics of a lineage that moved on (a deleted
    /// context's family, a superseded group) and would resurrect on
    /// scan if left behind. A no-op until a target is provisioned:
    /// with no manifest there is nothing to verify against, and
    /// removing files against an EMPTY one would wipe the very cache
    /// a degraded boot is serving.
    pub(crate) async fn hydrate_shared(&self) -> io::Result<SharedReport> {
        let Some((root, manifest)) = ({
            let inner = self.inner.lock().unwrap();
            inner
                .target
                .as_ref()
                .map(|target| (target.root.clone(), target.manifest.clone()))
        }) else {
            return Ok(SharedReport::default());
        };
        let mut report = SharedReport::default();
        let family: BTreeSet<String> = manifest
            .files
            .keys()
            .filter_map(|name| name.strip_suffix(".ctx"))
            .flat_map(crate::registry::context_files)
            .collect();
        for entry in std::fs::read_dir(&self.data_dir)? {
            let entry = entry?;
            if !entry.metadata()?.is_file() {
                continue;
            }
            let Ok(name) = entry.file_name().into_string() else {
                continue;
            };
            if name == ".taguru.lock"
                || name == ship::REPLICATION_RECORD
                || manifest.files.contains_key(&name)
                || manifest.lanes.contains_key(&name)
            {
                continue;
            }
            tracing::info!(
                file = %name,
                "removing a local file the bucket lineage does not carry"
            );
            std::fs::remove_file(entry.path())?;
            report.removed += 1;
        }
        for (name, expect) in &manifest.files {
            if family.contains(name) && !name.ends_with(".meta.json") {
                continue;
            }
            match self.fetch_published_if_stale(&root, name, *expect).await? {
                true => report.fetched += 1,
                false => report.reused += 1,
            }
        }
        Ok(report)
    }

    /// One family, verified file by file against the snapshot's slice:
    /// bytes that already match are reused in place, the rest are
    /// fetched. Returns (fetched, reused).
    async fn hydrate_family(
        &self,
        stem: &str,
        root: &StorePath,
        sig: &FamilySig,
    ) -> io::Result<(usize, usize)> {
        let mut fetched = 0usize;
        let mut reused = 0usize;
        for name in crate::registry::context_files(stem) {
            let landed = if let Some(expect) = sig.files.get(&name) {
                self.fetch_published_if_stale(root, &name, *expect).await?
            } else if let Some(lane) = sig.lanes.get(&name) {
                self.fetch_lane_if_stale(root, &name, *lane).await?
            } else {
                // Not in the manifest: hydrate_shared already removed
                // any local relic under this name before boot, and
                // nothing local can have re-created it before this
                // family's first load.
                continue;
            };
            match landed {
                true => fetched += 1,
                false => reused += 1,
            }
        }
        Ok((fetched, reused))
    }

    async fn fetch_published_if_stale(
        &self,
        root: &StorePath,
        name: &str,
        expect: ManifestFile,
    ) -> io::Result<bool> {
        let path = self.data_dir.join(name);
        // Pre-read stat, same reasoning as fetch_lane_if_stale's: a
        // miss just falls through to the read-and-CRC check already
        // below, so it costs at most one redundant pass.
        let pre_stat = std::fs::metadata(&path).ok().map(|meta| FileSig::of(&meta));
        if let Some(sig) = pre_stat {
            let cached = self.file_cache.lock().unwrap().get(name).copied();
            if cached == Some((expect, sig)) {
                return Ok(false);
            }
        }
        let mut expect = expect;
        let mut round = 0usize;
        loop {
            if let Ok(local) = std::fs::read(&path)
                && local.len() as u64 == expect.len
                && crate::crc32c::crc32c(&local) == expect.crc
            {
                // This branch only ever reads, so the pre-read stat
                // still describes what is on disk: cache it against
                // this target so the next hydrate_shared pass (any
                // OTHER family's shipping activity re-triggers one,
                // since the whole bucket shares a manifest) can skip
                // straight past an unrelated, unchanged file.
                if let Some(sig) = pre_stat {
                    self.file_cache
                        .lock()
                        .unwrap()
                        .insert(name.to_string(), (expect, sig));
                }
                return Ok(false);
            }
            let key = root.clone().join("files").join(name);
            let bytes = ship::fetch(self.store.as_ref(), &key).await?;
            match ship::verify_file_bytes(name, &bytes, expect) {
                Ok(()) => {
                    ship::write_restored_file(&self.data_dir, name, &bytes)?;
                    return Ok(true);
                }
                Err(error) => {
                    if let Some(refreshed) = self
                        .refreshed_extent(root, name, round, error, |manifest| {
                            manifest.files.get(name).copied()
                        })
                        .await?
                    {
                        expect = refreshed;
                    }
                    round += 1;
                }
            }
        }
    }

    /// The mismatch-is-not-always-rot arbiter behind both fetchers: a
    /// LIVE lineage legitimately advances mid-hydration — the writer
    /// replaces an object, then refreshes the manifest — so a
    /// verification failure first re-reads the generation's manifest
    /// and retries against whatever it now says, and only a mismatch
    /// that survives [`FETCH_REFRESH_ROUNDS`] re-reads (a STABLE
    /// manifest disagreeing with the bytes) keeps the rot refusal.
    /// Bounded, briefly paced: the window is one shipping cycle — the
    /// object landed, its manifest refresh has not.
    async fn refreshed_extent<T>(
        &self,
        root: &StorePath,
        name: &str,
        round: usize,
        error: io::Error,
        extent_of: impl Fn(&ship::Manifest) -> Option<T>,
    ) -> io::Result<Option<T>> {
        if round >= FETCH_REFRESH_ROUNDS {
            return Err(error);
        }
        tokio::time::sleep(FETCH_REFRESH_PAUSE).await;
        match ship::read_manifest(self.store.as_ref(), root).await {
            Ok(Some(manifest)) => match extent_of(&manifest) {
                Some(extent) => {
                    tracing::info!(
                        file = %name,
                        round,
                        "bucket object moved past the manifest snapshot; retrying \
                         against the refreshed manifest — a live writer shipped \
                         mid-hydration",
                    );
                    Ok(Some(extent))
                }
                // Vanished from the lineage: nothing newer to verify
                // against — surface the original mismatch.
                None => Err(error),
            },
            // A pre-manifest (empty) marker cannot re-verify anything.
            Ok(None) => Err(error),
            // A transient manifest read failure spends the round; the
            // next attempt re-fetches under the extent it already had.
            Err(read_error) => {
                tracing::debug!(
                    file = %name,
                    error = %read_error,
                    "manifest re-read failed while arbitrating a mismatch; retrying",
                );
                Ok(None)
            }
        }
    }

    /// A log lane reuses the local file iff the manifest's shipped
    /// extent is literally its prefix — the same arithmetic the
    /// shipping cursor lives by. What happens to a LONGER matching
    /// file is the [`LanePolicy`]: a writer-side cache keeps the tail
    /// (acknowledged records this very directory appended beyond the
    /// shipped stream — a crash before they shipped — recovered by
    /// WAL replay as ever), a replica truncates it to the shipped
    /// extent. A diverged or shorter file is refetched whole, and any
    /// diverged local tail is discarded — that is the takeover's
    /// stated cost.
    async fn fetch_lane_if_stale(
        &self,
        root: &StorePath,
        name: &str,
        lane: ManifestLane,
    ) -> io::Result<bool> {
        let path = self.data_dir.join(name);
        // Taken before the read below, exactly like ship_lane's
        // pre-read stat: a miss (file touched since, or first sight of
        // this lane) just falls through to the read-and-CRC check that
        // already ran unconditionally, so it can only cost one
        // redundant pass, never a false "already matches".
        let pre_stat = std::fs::metadata(&path).ok().map(|meta| FileSig::of(&meta));
        if let Some(sig) = pre_stat {
            let cached = self.lane_cache.lock().unwrap().get(name).copied();
            if cached == Some((lane, sig)) {
                return Ok(false);
            }
        }
        let mut lane = lane;
        let mut round = 0usize;
        loop {
            match std::fs::read(&path) {
                Ok(local) => {
                    let shipped = lane.len as usize;
                    if local.len() >= shipped
                        && crate::crc32c::crc32c(&local[..shipped]) == lane.crc
                    {
                        let truncated = local.len() > shipped
                            && matches!(self.lane_policy, LanePolicy::ShippedExact);
                        if truncated {
                            tracing::info!(
                                lane = %name,
                                kept = shipped,
                                dropped = local.len() - shipped,
                                "truncating a local log tail the lineage does not carry — \
                                 a replica serves the shipped stream exactly",
                            );
                            let file = std::fs::OpenOptions::new().write(true).open(&path)?;
                            file.set_len(lane.len)?;
                            file.sync_all()?;
                        } else if let Some(sig) = pre_stat {
                            // Unmodified by this call, so the pre-read
                            // stat still describes it: cache it against
                            // this lane target so a sibling file's
                            // change elsewhere in the family doesn't
                            // force this unrelated, unchanged lane
                            // through another read-and-CRC pass.
                            self.lane_cache
                                .lock()
                                .unwrap()
                                .insert(name.to_string(), (lane, sig));
                        }
                        return Ok(false);
                    }
                    if local.len() < shipped {
                        // The replica's steady tailing beat, not an
                        // anomaly: the shipped stream grew past the local
                        // copy (or, rarely, the local bytes diverged
                        // within it — indistinguishable without the
                        // download, identical remedy). Nothing beyond the
                        // shipped extent exists locally, so nothing is
                        // discarded; the refetch is bounded by the lane's
                        // size, which the writer's flush cadence resets.
                        tracing::debug!(
                            lane = %name,
                            local = local.len(),
                            shipped,
                            "shipped stream is ahead of the local log; fetching the lane",
                        );
                    } else if !local.is_empty() {
                        tracing::warn!(
                            lane = %name,
                            "local log diverged from the bucket lineage; refetching — any \
                             un-shipped local tail is discarded",
                        );
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
            // Fetch + record verification as one attempt: a lane
            // replaced under the reader (the writer's flush resets the
            // series) can surface as a missing segment, an extent
            // mismatch, or torn record framing — all the same story,
            // arbitrated by [`Self::refreshed_extent`] exactly like a
            // published file's.
            let attempt = async {
                let assembled = ship::fetch_lane(self.store.as_ref(), root, name, lane).await?;
                // The same record-by-record verification restore runs:
                // rot must refuse loudly, not replay quietly.
                crate::wal::shippable_records(&assembled).map_err(|error| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("lane {name} series {}: {error}", lane.series),
                    )
                })?;
                Ok::<Vec<u8>, io::Error>(assembled)
            }
            .await;
            match attempt {
                Ok(assembled) => {
                    crate::storage::write_atomic(&path, &assembled)?;
                    return Ok(true);
                }
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::InvalidData | io::ErrorKind::NotFound
                    ) =>
                {
                    if let Some(refreshed) = self
                        .refreshed_extent(root, name, round, error, |manifest| {
                            manifest.lanes.get(name).copied()
                        })
                        .await?
                    {
                        lane = refreshed;
                    }
                    round += 1;
                }
                Err(error) => return Err(error),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::AppState;
    use crate::ship::{ShipProgress, Shipper};
    use crate::wal::{self, WalOp};

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("taguru-hydrate-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn local_store(bucket: &FsPath) -> Arc<dyn ObjectStore> {
        Arc::new(object_store::local::LocalFileSystem::new_with_prefix(bucket).unwrap())
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

    fn ino(path: &FsPath) -> u64 {
        std::os::unix::fs::MetadataExt::ino(&std::fs::metadata(path).unwrap())
    }

    /// Rewinds a bucket object's `last_modified` (LocalFileSystem
    /// reads the file's mtime), to age a claim past the guard's grace.
    fn age(path: &FsPath, secs: u64) {
        let file = std::fs::File::options().write(true).open(path).unwrap();
        file.set_times(
            std::fs::FileTimes::new()
                .set_modified(SystemTime::now() - Duration::from_secs(secs + 1)),
        )
        .unwrap();
    }

    /// Ships one hand-laid family into a fresh file:// bucket with the
    /// real shipper — generation 1, manifest written — optionally
    /// retiring the generation the way a graceful stop does.
    async fn shipped_bucket(tag: &str, retire: bool) -> (PathBuf, PathBuf) {
        let bucket = scratch(&format!("{tag}-bucket"));
        let writer = scratch(&format!("{tag}-writer"));
        std::fs::write(writer.join("ctx_a.ctx"), b"image-v1").unwrap();
        std::fs::write(
            writer.join("ctx_a.meta.json"),
            br#"{"description":"d","pinned":false}"#,
        )
        .unwrap();
        wal::append_batch(&writer.join("ctx_a.wal.jsonl"), 1, &[associate("a")]).unwrap();
        let state = AppState::boot(writer.clone(), 64 * 1024 * 1024, None).unwrap();
        let mut shipper = Shipper::claim(
            local_store(&bucket),
            StorePath::default(),
            url_of(tag),
            writer.clone(),
            Arc::new(ShipProgress::new()),
            state,
            None,
        )
        .await
        .unwrap();
        shipper.cycle().await.unwrap();
        if retire {
            shipper.retire_generation().await;
        }
        (bucket, writer)
    }

    /// A reader whose manifest snapshot has aged while the writer kept
    /// shipping — the boot-time race a replica (or stateless boot)
    /// falls into against a live lineage. Every fetched shape is
    /// covered: a published file replaced under the reader (the meta,
    /// via hydrate_shared; the image, via first touch) and a log lane
    /// whose series was reset and its old segments removed. Each
    /// mismatch must arbitrate through a manifest re-read and land the
    /// NEWER bytes, not refuse as rot — a replica booting against an
    /// actively-shipping writer would otherwise crash-loop on a window
    /// it can never win.
    #[tokio::test]
    async fn a_mismatch_against_a_moving_lineage_heals_through_a_manifest_reread() {
        let bucket = scratch("race-bucket");
        let writer = scratch("race-writer");
        std::fs::write(writer.join("ctx_a.ctx"), b"image-v1").unwrap();
        std::fs::write(
            writer.join("ctx_a.meta.json"),
            br#"{"description":"d","pinned":false}"#,
        )
        .unwrap();
        wal::append_batch(&writer.join("ctx_a.wal.jsonl"), 1, &[associate("a")]).unwrap();
        let state = AppState::boot(writer.clone(), 64 * 1024 * 1024, None).unwrap();
        let mut shipper = Shipper::claim(
            local_store(&bucket),
            StorePath::default(),
            url_of("race"),
            writer.clone(),
            Arc::new(ShipProgress::new()),
            state,
            None,
        )
        .await
        .unwrap();
        shipper.cycle().await.unwrap();

        // The reader's snapshot: cycle 1's manifest, held while the
        // writer moves on.
        let store = local_store(&bucket);
        let root = ship::gen_root(&StorePath::default(), 1);
        let stale = ship::read_manifest(store.as_ref(), &root)
            .await
            .unwrap()
            .expect("cycle 1 wrote a manifest");
        let stale_series = stale.lanes["ctx_a.wal.jsonl"].series;

        // Cycle 2: the image and meta are replaced, and the WAL's
        // series resets (the writer's flush truncated it) — every
        // extent the stale snapshot describes is now wrong.
        std::fs::write(writer.join("ctx_a.ctx"), b"image-v2-and-longer").unwrap();
        std::fs::write(
            writer.join("ctx_a.meta.json"),
            br#"{"description":"moved on","pinned":false}"#,
        )
        .unwrap();
        std::fs::write(writer.join("ctx_a.wal.jsonl"), b"").unwrap();
        wal::append_batch(&writer.join("ctx_a.wal.jsonl"), 9, &[associate("b")]).unwrap();
        shipper.cycle().await.unwrap();
        // The old series' segments age out of the bucket (a cleanup a
        // future shipper may run; simulated here) — the stale lane
        // fetch cannot even 404 its way to the old bytes.
        let old_segment_prefix = format!("{stale_series:010}-");
        for entry in walk_files(&bucket) {
            let is_old_segment = entry
                .parent()
                .and_then(|parent| parent.parent())
                .and_then(|lane_dir| lane_dir.file_name())
                .is_some_and(|dir| dir == "wal")
                && entry
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with(&old_segment_prefix));
            if is_old_segment {
                std::fs::remove_file(entry).unwrap();
            }
        }

        let target = scratch("race-target");
        let hydrator = Hydrator::new(
            Arc::clone(&store),
            1,
            root,
            target.clone(),
            stale,
            LanePolicy::ShippedExact,
        );
        // The shared pass fetches the meta: bytes v2 vs snapshot v1.
        hydrator
            .hydrate_shared()
            .await
            .expect("a moving lineage heals through the manifest re-read");
        assert_eq!(
            std::fs::read(target.join("ctx_a.meta.json")).unwrap(),
            br#"{"description":"moved on","pinned":false}"#
        );
        // First touch fetches the image and the lane: both stale.
        hydrator.ensure_context("ctx_a").unwrap();
        assert_eq!(
            std::fs::read(target.join("ctx_a.ctx")).unwrap(),
            b"image-v2-and-longer"
        );
        let lane = std::fs::read(target.join("ctx_a.wal.jsonl")).unwrap();
        assert!(
            String::from_utf8_lossy(&lane).contains("\"b\""),
            "the healed lane carries cycle 2's records"
        );
        let _ = std::fs::remove_dir_all(&bucket);
        let _ = std::fs::remove_dir_all(&writer);
        let _ = std::fs::remove_dir_all(&target);
    }

    /// The refusal the retry must NOT soften: bytes disagreeing with a
    /// manifest that is not moving is rot, and it still fails the
    /// fetch after the re-read rounds expire.
    #[tokio::test]
    async fn true_rot_still_refuses_after_the_reread_rounds() {
        let (bucket, writer) = shipped_bucket("rot-arbiter", true).await;
        let store = local_store(&bucket);
        let root = ship::gen_root(&StorePath::default(), 1);
        let manifest = ship::read_manifest(store.as_ref(), &root)
            .await
            .unwrap()
            .unwrap();

        // The object rots in place; the manifest never moves.
        let meta = walk_files(&bucket)
            .into_iter()
            .find(|path| path.ends_with("files/ctx_a.meta.json"))
            .expect("the shipped meta object exists");
        std::fs::write(&meta, b"rotten").unwrap();

        let target = scratch("rot-arbiter-target");
        let hydrator = Hydrator::new(
            Arc::clone(&store),
            1,
            root,
            target.clone(),
            manifest,
            LanePolicy::ShippedExact,
        );
        let error = hydrator
            .hydrate_shared()
            .await
            .expect_err("a stable manifest disagreeing with the bytes is rot");
        assert!(
            error.to_string().contains("do not match"),
            "unexpected error: {error}"
        );
        let _ = std::fs::remove_dir_all(&bucket);
        let _ = std::fs::remove_dir_all(&writer);
        let _ = std::fs::remove_dir_all(&target);
    }

    /// Every file under `dir`, recursively — the tests' tiny walker.
    fn walk_files(dir: &FsPath) -> Vec<PathBuf> {
        let mut files = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(current) = stack.pop() {
            for entry in std::fs::read_dir(current).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    files.push(path);
                }
            }
        }
        files
    }

    fn url_of(tag: &str) -> String {
        format!("file://taguru-test-{tag}")
    }

    #[tokio::test]
    async fn an_empty_directory_hydrates_shared_files_then_families_on_demand() {
        let (bucket, _writer) = shipped_bucket("lazy", true).await;
        let target = scratch("lazy-target");
        let store = local_store(&bucket);

        let hydrator = prepare(
            &store,
            &StorePath::default(),
            &url_of("lazy"),
            &target,
            false,
        )
        .await
        .unwrap()
        .expect("an empty directory against a complete generation hydrates");

        // Shared pass: the sidecar meta landed (enumeration needs it);
        // the image and the log did not (they are the lazy payload).
        assert!(target.join("ctx_a.meta.json").exists());
        assert!(!target.join("ctx_a.ctx").exists());
        assert!(!target.join("ctx_a.wal.jsonl").exists());
        assert_eq!(hydrator.context_stems(), ["ctx_a".to_string()]);
        assert!(!hydrator.drained());

        // The directory is durably marked as a cache of generation 1,
        // before any claim exists.
        let record = ship::read_replication_record(&target).expect("cache mode is recorded");
        assert_eq!(record.hydrated_from, Some(1));
        assert_eq!(record.claimed_generation, None);

        // First touch materializes the family, verified; a second
        // touch (and an unknown stem) is a no-op.
        hydrator.ensure_context("ctx_a").unwrap();
        assert_eq!(
            std::fs::read(target.join("ctx_a.ctx")).unwrap(),
            b"image-v1"
        );
        assert!(target.join("ctx_a.wal.jsonl").exists());
        assert!(hydrator.drained());
        hydrator.ensure_context("ctx_a").unwrap();
        hydrator.ensure_context("never-shipped").unwrap();

        // A vetoed family stays vetoed (drained, but never refetched).
        hydrator.veto("ctx_a");
        assert!(hydrator.drained());
        let _ = std::fs::remove_dir_all(&bucket);
        let _ = std::fs::remove_dir_all(&target);
    }

    #[tokio::test]
    async fn the_takeover_guard_demands_intent_only_while_the_writer_may_live() {
        let (bucket, _writer) = shipped_bucket("guard", false).await;
        let store = local_store(&bucket);
        let url = url_of("guard");

        // Un-retired and freshly alive: an empty-directory boot must
        // state its intent…
        let target = scratch("guard-target");
        let error = prepare(&store, &StorePath::default(), &url, &target, false)
            .await
            .expect_err("deposing a live writer needs intent");
        assert!(error.to_string().contains("take-over"), "{error}");

        // …and with it, the takeover proceeds.
        let taken = prepare(&store, &StorePath::default(), &url, &target, true)
            .await
            .unwrap();
        assert!(taken.is_some());

        // A crashed writer ages out of the guard: no retired marker,
        // but the fence and heartbeat are past the grace.
        let aged_target = scratch("guard-aged");
        age(
            &bucket.join("fence").join(format!("{:020}", 1)),
            ship::TAKEOVER_GRACE.as_secs(),
        );
        age(
            &bucket.join("gen-00000000000000000001").join("heartbeat"),
            ship::TAKEOVER_GRACE.as_secs(),
        );
        let aged = prepare(&store, &StorePath::default(), &url, &aged_target, false)
            .await
            .unwrap();
        assert!(aged.is_some(), "a stale generation is claimable freely");

        // A cleanly retired generation never guards at all.
        let (retired_bucket, _writer) = shipped_bucket("guard-retired", true).await;
        let retired_target = scratch("guard-retired-target");
        let retired = prepare(
            &local_store(&retired_bucket),
            &StorePath::default(),
            &url_of("guard-retired"),
            &retired_target,
            false,
        )
        .await
        .unwrap();
        assert!(retired.is_some());

        for dir in [bucket, target, aged_target, retired_bucket, retired_target] {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[tokio::test]
    async fn a_warm_restart_boots_local_and_a_moved_lineage_reverifies_the_cache() {
        let (bucket, _writer) = shipped_bucket("warm", true).await;
        let store = local_store(&bucket);
        let url = url_of("warm");
        let target = scratch("warm-target");

        // Hydrate once, then pretend our shipper claimed generation 2
        // (what a real serve does right after).
        let hydrator = prepare(&store, &StorePath::default(), &url, &target, false)
            .await
            .unwrap()
            .expect("hydrates");
        hydrator.ensure_context("ctx_a").unwrap();
        std::fs::write(bucket.join("fence").join(format!("{:020}", 2)), b"{}").unwrap();
        ship::write_replication_record(
            &target,
            &ship::ReplicationRecord {
                url: url.clone(),
                claimed_generation: Some(2),
                hydrated_from: Some(1),
            },
        )
        .unwrap();

        // The bucket's newest writer is us: no verification, no guard.
        let warm = prepare(&store, &StorePath::default(), &url, &target, false)
            .await
            .unwrap();
        assert!(warm.is_none(), "the lineage's own tip boots as before");

        // A third claim by someone else moves the lineage on; the
        // cache must re-verify (behind the guard, since that claim is
        // fresh and un-retired).
        std::fs::write(bucket.join("fence").join(format!("{:020}", 3)), b"{}").unwrap();
        let error = prepare(&store, &StorePath::default(), &url, &target, false)
            .await
            .expect_err("a fresh foreign claim guards even a cache-mode boot");
        assert!(error.to_string().contains("take-over"), "{error}");
        let reverify = prepare(&store, &StorePath::default(), &url, &target, true)
            .await
            .unwrap();
        assert!(
            reverify.is_some(),
            "a cache whose lineage moved on re-verifies instead of trusting local files"
        );

        let _ = std::fs::remove_dir_all(&bucket);
        let _ = std::fs::remove_dir_all(&target);
    }

    #[tokio::test]
    async fn hydration_reuses_matching_bytes_keeps_a_matching_tail_and_replaces_divergence() {
        let (bucket, writer) = shipped_bucket("reuse", true).await;
        let store = local_store(&bucket);
        let url = url_of("reuse");
        let target = scratch("reuse-target");

        // Seed the target as a cache-mode directory holding: the exact
        // shipped image, the shipped log plus a genuine appended tail,
        // and a relic file the lineage does not carry.
        let image = std::fs::read(writer.join("ctx_a.ctx")).unwrap();
        std::fs::write(target.join("ctx_a.ctx"), &image).unwrap();
        let shipped_wal = std::fs::read(writer.join("ctx_a.wal.jsonl")).unwrap();
        std::fs::write(target.join("ctx_a.wal.jsonl"), &shipped_wal).unwrap();
        wal::append_batch(&target.join("ctx_a.wal.jsonl"), 2, &[associate("tail")]).unwrap();
        let with_tail = std::fs::read(target.join("ctx_a.wal.jsonl")).unwrap();
        assert!(with_tail.len() > shipped_wal.len());
        std::fs::write(target.join("ctx_b.ctx"), b"relic of a deleted context").unwrap();
        ship::write_replication_record(
            &target,
            &ship::ReplicationRecord {
                url: url.clone(),
                claimed_generation: None,
                hydrated_from: Some(1),
            },
        )
        .unwrap();

        let image_ino = ino(&target.join("ctx_a.ctx"));
        let hydrator = prepare(&store, &StorePath::default(), &url, &target, false)
            .await
            .unwrap()
            .expect("a cache directory re-verifies");
        assert!(
            !target.join("ctx_b.ctx").exists(),
            "a relic the lineage does not carry is removed before boot could resurrect it"
        );
        hydrator.ensure_context("ctx_a").unwrap();
        assert_eq!(
            ino(&target.join("ctx_a.ctx")),
            image_ino,
            "matching bytes are reused in place, not re-downloaded"
        );
        assert_eq!(
            std::fs::read(target.join("ctx_a.wal.jsonl")).unwrap(),
            with_tail,
            "a log whose shipped prefix matches keeps its acknowledged tail"
        );

        // A diverged log (rewritten over the shipped offsets) is
        // replaced by the lineage's own bytes.
        let diverged_target = scratch("reuse-diverged");
        std::fs::write(
            diverged_target.join("ctx_a.wal.jsonl"),
            b"not the shipped bytes",
        )
        .unwrap();
        ship::write_replication_record(
            &diverged_target,
            &ship::ReplicationRecord {
                url: url.clone(),
                claimed_generation: None,
                hydrated_from: Some(1),
            },
        )
        .unwrap();
        let hydrator = prepare(&store, &StorePath::default(), &url, &diverged_target, false)
            .await
            .unwrap()
            .expect("hydrates");
        hydrator.ensure_context("ctx_a").unwrap();
        assert_eq!(
            std::fs::read(diverged_target.join("ctx_a.wal.jsonl")).unwrap(),
            shipped_wal,
            "a diverged log is replaced by the shipped stream"
        );

        for dir in [bucket, target, diverged_target] {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[tokio::test]
    async fn a_replica_boot_is_always_a_cache_never_a_claimant_and_never_foreign_truth() {
        let (bucket, writer) = shipped_bucket("replica-table", false).await;
        let store = local_store(&bucket);
        let url = url_of("replica-table");

        // Real data, no record: somebody's truth — refused, even
        // though the same directory would boot fine as a writer.
        let foreign = scratch("replica-foreign");
        std::fs::write(foreign.join("mine.ctx"), b"local truth").unwrap();
        let error = prepare_replica(&store, &StorePath::default(), &url, &foreign)
            .await
            .expect_err("a replica must not consume a directory that is not a cache");
        assert!(error.to_string().contains("not a cache"), "{error}");

        // The bucket's newest writer's own directory: demoting it
        // would discard its un-shipped tail — refused.
        let error = prepare_replica(&store, &StorePath::default(), &url, &writer)
            .await
            .expect_err("the current writer's directory must not demote");
        assert!(error.to_string().contains("newest writer"), "{error}");

        // No takeover guard applies: the writer above is un-retired
        // and freshly alive, yet an EMPTY directory replicates freely
        // (a replica deposes nobody).
        let target = scratch("replica-table-target");
        let hydrator = prepare_replica(&store, &StorePath::default(), &url, &target)
            .await
            .expect("an empty directory replicates a live lineage without ceremony");
        assert_eq!(hydrator.generation(), Some(1));
        let record = ship::read_replication_record(&target).unwrap();
        assert_eq!(record.hydrated_from, Some(1));
        assert_eq!(record.claimed_generation, None);

        // A FORMER writer (its claim deposed by a newer fence) demotes
        // freely, and the demotion clears the stale claim.
        std::fs::write(bucket.join("fence").join(format!("{:020}", 9)), b"{}").unwrap();
        let deposed = prepare_replica(&store, &StorePath::default(), &url, &writer)
            .await
            .expect("a deposed writer's directory demotes to a cache");
        assert_eq!(deposed.generation(), Some(1));
        let record = ship::read_replication_record(&writer).unwrap();
        assert_eq!(
            record.claimed_generation, None,
            "the stale claim is dropped"
        );

        for dir in [bucket, writer, foreign, target] {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[tokio::test]
    async fn a_replica_truncates_an_acked_tail_where_a_writer_cache_keeps_it() {
        let (bucket, writer) = shipped_bucket("replica-tail", true).await;
        let store = local_store(&bucket);
        let url = url_of("replica-tail");

        // Seed a cache directory holding the shipped log PLUS a
        // genuine acked tail — exactly the writer-cache reuse case.
        let target = scratch("replica-tail-target");
        let shipped_wal = std::fs::read(writer.join("ctx_a.wal.jsonl")).unwrap();
        std::fs::write(target.join("ctx_a.wal.jsonl"), &shipped_wal).unwrap();
        wal::append_batch(&target.join("ctx_a.wal.jsonl"), 2, &[associate("tail")]).unwrap();
        ship::write_replication_record(
            &target,
            &ship::ReplicationRecord {
                url: url.clone(),
                claimed_generation: None,
                hydrated_from: Some(1),
            },
        )
        .unwrap();

        let hydrator = prepare_replica(&store, &StorePath::default(), &url, &target)
            .await
            .expect("a cache re-verifies");
        hydrator.ensure_context("ctx_a").unwrap();
        assert_eq!(
            std::fs::read(target.join("ctx_a.wal.jsonl")).unwrap(),
            shipped_wal,
            "a replica serves the shipped stream exactly: the tail is truncated away"
        );

        for dir in [bucket, writer, target] {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[tokio::test]
    async fn a_replica_with_nothing_to_tail_serves_what_it_has_and_waits() {
        // A virgin bucket: an empty replica boots unprovisioned (and
        // the tailer will provision it when a lineage appears).
        let empty_bucket = scratch("replica-virgin-bucket");
        let store = local_store(&empty_bucket);
        let target = scratch("replica-virgin-target");
        let hydrator = prepare_replica(&store, &StorePath::default(), &url_of("virgin-r"), &target)
            .await
            .expect("a virgin bucket replicates as empty");
        assert_eq!(hydrator.generation(), None);
        assert!(hydrator.drained(), "nothing pending");
        hydrator.ensure_context("anything").unwrap();

        // An unreachable bucket: a cache serves its last watermark
        // degraded; an empty directory has nothing to serve — refused.
        // (A merely-ABSENT local path lists as empty — i.e. virgin —
        // so "unreachable" here is the path turning into a file, which
        // errors every listing the way a dead endpoint would.)
        let gone = scratch("replica-gone-bucket");
        let unreachable = local_store(&gone);
        std::fs::remove_dir_all(&gone).unwrap();
        std::fs::write(&gone, b"not a directory").unwrap();
        let cache = scratch("replica-degraded-cache");
        std::fs::write(cache.join("ctx_a.ctx"), b"cached image").unwrap();
        ship::write_replication_record(
            &cache,
            &ship::ReplicationRecord {
                url: url_of("gone"),
                claimed_generation: None,
                hydrated_from: Some(1),
            },
        )
        .unwrap();
        let degraded =
            prepare_replica(&unreachable, &StorePath::default(), &url_of("gone"), &cache)
                .await
                .expect("a cache boots degraded when the bucket is unreachable");
        assert_eq!(degraded.generation(), None);
        assert_eq!(
            std::fs::read(cache.join("ctx_a.ctx")).unwrap(),
            b"cached image",
            "nothing is wiped while unprovisioned"
        );
        let empty = scratch("replica-degraded-empty");
        let error = prepare_replica(&unreachable, &StorePath::default(), &url_of("gone"), &empty)
            .await
            .expect_err("an empty replica with no bucket has nothing to serve");
        assert!(error.to_string().contains("nothing to serve"), "{error}");

        // An UNVERIFIED conversion (a record without `hydrated_from` —
        // phase one of the two-phase write) is not a cache yet: with
        // the bucket unreachable there is nothing honest to serve.
        let half = scratch("replica-degraded-half");
        std::fs::write(half.join("ctx_a.ctx"), b"a demoted writer's bytes").unwrap();
        ship::write_replication_record(
            &half,
            &ship::ReplicationRecord {
                url: url_of("gone"),
                claimed_generation: None,
                hydrated_from: None,
            },
        )
        .unwrap();
        let error = prepare_replica(&unreachable, &StorePath::default(), &url_of("gone"), &half)
            .await
            .expect_err("an unverified conversion must not serve degraded");
        assert!(
            error.to_string().contains("not yet a verified cache"),
            "{error}"
        );

        let _ = std::fs::remove_file(&gone);
        for dir in [empty_bucket, target, cache, empty, half] {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[tokio::test]
    async fn the_verified_watermark_advances_only_after_hydration_lands() {
        let (bucket, writer) = shipped_bucket("two-phase", true).await;
        let store = local_store(&bucket);
        let url = url_of("two-phase");
        let target = scratch("two-phase-target");

        // Break the shared hydration: the manifest names a meta whose
        // object is gone (set aside, as a mid-upload crash or an
        // eventually-consistent read would present it).
        let meta_key = bucket
            .join("gen-00000000000000000001")
            .join("files")
            .join("ctx_a.meta.json");
        let aside = bucket.join("meta-aside");
        std::fs::rename(&meta_key, &aside).unwrap();

        let error = prepare_replica(&store, &StorePath::default(), &url, &target)
            .await
            .expect_err("a shared hydration that cannot land refuses the boot");
        assert!(error.to_string().contains("ctx_a.meta.json"), "{error}");
        let record = ship::read_replication_record(&target)
            .expect("phase one marked the directory as this URL's replica workspace");
        assert_eq!(
            record.hydrated_from, None,
            "the verified watermark must not advance past a failed hydration"
        );

        // The object returns; the same boot completes and phase two
        // records the verified generation.
        std::fs::rename(&aside, &meta_key).unwrap();
        let hydrator = prepare_replica(&store, &StorePath::default(), &url, &target)
            .await
            .expect("the retry hydrates");
        assert_eq!(hydrator.generation(), Some(1));
        let record = ship::read_replication_record(&target).unwrap();
        assert_eq!(record.hydrated_from, Some(1));

        for dir in [bucket, writer, target] {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[tokio::test]
    async fn retarget_diffs_families_vetoes_vanished_and_revives_readded_ones() {
        let (bucket, _writer) = shipped_bucket("retarget", true).await;
        let store = local_store(&bucket);
        let url = url_of("retarget");
        let target = scratch("retarget-target");
        let hydrator = prepare_replica(&store, &StorePath::default(), &url, &target)
            .await
            .expect("hydrates");
        hydrator.ensure_context("ctx_a").unwrap();
        assert!(hydrator.drained());

        let generation_root = ship::gen_root(&StorePath::default(), 1);
        let manifest = ship::read_manifest(store.as_ref(), &generation_root)
            .await
            .unwrap()
            .expect("the shipped generation carries a manifest");

        // The same manifest again: nothing is stale, nothing vanished.
        let report = hydrator.retarget(1, generation_root.clone(), manifest.clone());
        assert!(report.stale.is_empty(), "{report:?}");
        assert!(report.vanished.is_empty(), "{report:?}");

        // A moved lane slice marks the family stale.
        let mut moved = manifest.clone();
        moved
            .lanes
            .get_mut("ctx_a.wal.jsonl")
            .expect("the lane exists")
            .seq += 1;
        let report = hydrator.retarget(1, generation_root.clone(), moved);
        assert_eq!(report.stale, ["ctx_a"], "{report:?}");
        assert!(!hydrator.drained(), "a stale family is pending again");

        // A manifest without the family vetoes it, exactly once.
        let report = hydrator.retarget(2, generation_root.clone(), Manifest::default());
        assert_eq!(report.vanished, ["ctx_a"], "{report:?}");
        assert!(hydrator.drained(), "vetoed counts as settled");
        let report = hydrator.retarget(2, generation_root.clone(), Manifest::default());
        assert!(report.vanished.is_empty(), "vanished reports once");

        // A later manifest carrying it again revives it.
        let report = hydrator.retarget(3, generation_root.clone(), manifest);
        assert_eq!(report.stale, ["ctx_a"], "{report:?}");
        hydrator.ensure_context("ctx_a").unwrap();
        assert!(hydrator.drained());

        for dir in [bucket, target] {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[tokio::test]
    async fn a_virgin_bucket_and_a_conventional_directory_boot_as_before() {
        let empty_bucket = scratch("virgin-bucket");
        let store = local_store(&empty_bucket);
        let target = scratch("virgin-target");
        let plan = prepare(
            &store,
            &StorePath::default(),
            &url_of("virgin"),
            &target,
            false,
        )
        .await
        .unwrap();
        assert!(plan.is_none(), "nothing to hydrate from, nothing to guard");

        // A conventional directory (real data, no cache marker) never
        // hydrates — even when the bucket carries a foreign lineage —
        // though the guard still asks for intent while that lineage's
        // writer may be alive.
        let (bucket, _writer) = shipped_bucket("conventional", true).await;
        let local = scratch("conventional-local");
        std::fs::write(local.join("mine.ctx"), b"local truth").unwrap();
        let plan = prepare(
            &local_store(&bucket),
            &StorePath::default(),
            &url_of("conventional"),
            &local,
            false,
        )
        .await
        .unwrap();
        assert!(
            plan.is_none(),
            "pre-#128 directories keep booting on local truth"
        );
        assert_eq!(
            std::fs::read(local.join("mine.ctx")).unwrap(),
            b"local truth"
        );

        for dir in [empty_bucket, target, bucket, local] {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}
