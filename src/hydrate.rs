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

use std::collections::BTreeMap;
use std::io;
use std::path::{Path as FsPath, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime};

use object_store::ObjectStore;
use object_store::path::Path as StorePath;

use crate::ship::{self, Manifest, ManifestFile, ManifestLane};

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
        generation_root,
        data_dir.to_path_buf(),
        manifest,
    ));
    let report = hydrator.hydrate_shared().await?;
    tracing::info!(
        generation,
        contexts = hydrator.states.lock().unwrap().len(),
        shared_fetched = report.fetched,
        shared_reused = report.reused,
        removed = report.removed,
        "hydrating from the bucket — shared files landed; context families \
         hydrate on preload, first touch, or the background fill",
    );
    Ok(Some(hydrator))
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
async fn head_modified(
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
struct SharedReport {
    fetched: usize,
    reused: usize,
    removed: usize,
}

/// Where one context family stands in this boot's hydration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StemState {
    /// Not hydrated yet — first touch, the preload, or the background
    /// fill will take it.
    Pending,
    /// Someone is hydrating it right now; wait, don't duplicate.
    InFlight,
    /// Locally materialized and verified (or reused) — never looked at
    /// again: everything after this point is this server's own work.
    Done,
    /// Deleted or renamed away locally before hydration reached it —
    /// the bucket copy must NOT be re-materialized, or the vanished
    /// family would resurrect on the next boot.
    Vetoed,
}

/// The lazy half of a bucket boot, shared by the registry (first
/// touch, the pinned preload), the background fill, and the shipper
/// (whose manifest write gates on [`Hydrator::drained`]).
#[derive(Debug)]
pub(crate) struct Hydrator {
    store: Arc<dyn ObjectStore>,
    generation_root: StorePath,
    data_dir: PathBuf,
    manifest: Manifest,
    states: Mutex<BTreeMap<String, StemState>>,
    settled: Condvar,
}

impl Hydrator {
    fn new(
        store: Arc<dyn ObjectStore>,
        generation_root: StorePath,
        data_dir: PathBuf,
        manifest: Manifest,
    ) -> Self {
        let states = manifest
            .files
            .keys()
            .filter_map(|name| name.strip_suffix(".ctx"))
            .map(|stem| (stem.to_string(), StemState::Pending))
            .collect();
        Self {
            store,
            generation_root,
            data_dir,
            manifest,
            states: Mutex::new(states),
            settled: Condvar::new(),
        }
    }

    /// Every context stem the manifest carries, for boot's registry
    /// registration — the sidecar metas these describe are already
    /// local by the time boot runs ([`Self::hydrate_shared`]).
    pub(crate) fn context_stems(&self) -> Vec<String> {
        self.states.lock().unwrap().keys().cloned().collect()
    }

    /// Whether every family is settled (hydrated or vetoed): the
    /// shipper's gate for writing its own generation's manifest — a
    /// `complete` written earlier would crown a generation that lacks
    /// every family still waiting in the predecessor.
    pub(crate) fn drained(&self) -> bool {
        self.states
            .lock()
            .unwrap()
            .values()
            .all(|state| matches!(state, StemState::Done | StemState::Vetoed))
    }

    /// Marks a family off-limits before its local files are deleted or
    /// renamed away: hydration must never re-materialize what the
    /// server just removed. Waits out an in-flight hydration first so
    /// the caller's deletion cannot interleave with a half-landed
    /// download.
    pub(crate) fn veto(&self, stem: &str) {
        let mut states = self.states.lock().unwrap();
        while matches!(states.get(stem), Some(StemState::InFlight)) {
            states = self.settled.wait(states).unwrap();
        }
        if states.contains_key(stem) {
            states.insert(stem.to_string(), StemState::Vetoed);
        }
    }

    /// Materializes one context's family before its first load —
    /// synchronous, callable from any thread (each hydration runs on
    /// its own scoped worker thread with its own small runtime, so no
    /// caller ever block_on's inside an async worker). Families the
    /// manifest does not know (created after this boot) return
    /// immediately; concurrent callers of the same stem coalesce.
    pub(crate) fn ensure_context(&self, stem: &str) -> io::Result<()> {
        let mut states = self.states.lock().unwrap();
        loop {
            match states.get(stem) {
                None | Some(StemState::Done | StemState::Vetoed) => return Ok(()),
                Some(StemState::InFlight) => states = self.settled.wait(states).unwrap(),
                Some(StemState::Pending) => break,
            }
        }
        states.insert(stem.to_string(), StemState::InFlight);
        drop(states);

        let outcome = std::thread::scope(|scope| {
            scope
                .spawn(|| -> io::Result<(usize, usize)> {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()?;
                    runtime.block_on(self.hydrate_family(stem))
                })
                .join()
                .expect("the hydration worker must not panic")
        });

        let mut states = self.states.lock().unwrap();
        match &outcome {
            Ok((fetched, reused)) => {
                // A veto cannot have landed meanwhile — veto() waits
                // for InFlight to settle — so this InFlight is ours.
                states.insert(stem.to_string(), StemState::Done);
                tracing::info!(stem, fetched, reused, "context family hydrated");
            }
            Err(error) => {
                // Back to Pending: the next touch (or fill pass)
                // retries. The caller surfaces the error through the
                // same quarantine a failed local load uses.
                states.insert(stem.to_string(), StemState::Pending);
                tracing::warn!(stem, %error, "context family hydration failed");
            }
        }
        drop(states);
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
                        let states = hydrator.states.lock().unwrap();
                        states
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

    /// The eager half, run once before boot: land every non-family
    /// file (groups, the grant store, crash markers) plus every
    /// family's sidecar meta — everything enumeration and description
    /// need — and remove local files the manifest does not know, which
    /// in cache mode are relics of a lineage that moved on (a deleted
    /// context's family, a superseded group) and would resurrect on
    /// scan if left behind.
    async fn hydrate_shared(&self) -> io::Result<SharedReport> {
        let mut report = SharedReport::default();
        let family: std::collections::BTreeSet<String> = self
            .states
            .lock()
            .unwrap()
            .keys()
            .flat_map(|stem| crate::registry::context_files(stem))
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
                || self.manifest.files.contains_key(&name)
                || self.manifest.lanes.contains_key(&name)
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
        for (name, expect) in &self.manifest.files {
            if family.contains(name) && !name.ends_with(".meta.json") {
                continue;
            }
            match self.fetch_published_if_stale(name, *expect).await? {
                true => report.fetched += 1,
                false => report.reused += 1,
            }
        }
        Ok(report)
    }

    /// One family, verified file by file against the manifest: bytes
    /// that already match are reused in place, the rest are fetched.
    /// Returns (fetched, reused).
    async fn hydrate_family(&self, stem: &str) -> io::Result<(usize, usize)> {
        let mut fetched = 0usize;
        let mut reused = 0usize;
        for name in crate::registry::context_files(stem) {
            let landed = if let Some(expect) = self.manifest.files.get(&name) {
                self.fetch_published_if_stale(&name, *expect).await?
            } else if let Some(lane) = self.manifest.lanes.get(&name) {
                self.fetch_lane_if_stale(&name, *lane).await?
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

    async fn fetch_published_if_stale(&self, name: &str, expect: ManifestFile) -> io::Result<bool> {
        let path = self.data_dir.join(name);
        if let Ok(local) = std::fs::read(&path)
            && local.len() as u64 == expect.len
            && crate::crc32c::crc32c(&local) == expect.crc
        {
            return Ok(false);
        }
        let key = self.generation_root.clone().join("files").join(name);
        let bytes = ship::fetch(self.store.as_ref(), &key).await?;
        ship::verify_file_bytes(name, &bytes, expect)?;
        ship::write_restored_file(&self.data_dir, name, &bytes)?;
        Ok(true)
    }

    /// A log lane reuses the local file iff the manifest's shipped
    /// extent is literally its prefix — the same arithmetic the
    /// shipping cursor lives by. A LONGER matching file keeps its
    /// tail: those are acknowledged records this very directory
    /// appended beyond the shipped stream (a crash before they
    /// shipped), and WAL replay is their normal recovery. A diverged
    /// or shorter file is refetched whole, and any diverged local
    /// tail is discarded — that is the takeover's stated cost.
    async fn fetch_lane_if_stale(&self, name: &str, lane: ManifestLane) -> io::Result<bool> {
        let path = self.data_dir.join(name);
        match std::fs::read(&path) {
            Ok(local) => {
                let shipped = lane.len as usize;
                if local.len() >= shipped && crate::crc32c::crc32c(&local[..shipped]) == lane.crc {
                    return Ok(false);
                }
                if !local.is_empty() {
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
        let assembled =
            ship::fetch_lane(self.store.as_ref(), &self.generation_root, name, lane).await?;
        // The same record-by-record verification restore runs: rot
        // must refuse loudly, not replay quietly.
        crate::wal::shippable_records(&assembled).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("lane {name} series {}: {error}", lane.series),
            )
        })?;
        crate::storage::write_atomic(&path, &assembled)?;
        Ok(true)
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
