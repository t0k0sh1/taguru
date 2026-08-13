//! Read replicas tailing the shipped WAL (issue #129): `serve
//! --replica` is issue #128's hydration running CONTINUOUSLY — poll
//! the bucket's newest complete manifest, re-verify what moved, land
//! it locally, and drop the loaded copies so the next read serves the
//! new bytes through the ordinary load path (image plus watermark
//! replay). A replica never claims a generation, never heartbeats,
//! never ships; epoch fencing already guarantees it cannot disturb
//! the writer, and refusing every mutating verb (the replica gate in
//! `main::routes`) guarantees clients cannot disturb IT.
//!
//! # What a replica promises
//!
//! Consistency is per context, at that context's applied watermark:
//! one context's reads are some exact prefix of the writer's
//! acknowledged history (a shipped-cycle boundary), but two contexts
//! may sit at different cycles — cross-context skew is possible — and
//! staleness is bounded by the writer's shipping lag plus this
//! replica's poll interval. `/metrics` shows the per-context
//! arithmetic (`taguru_replica_*`): applied vs newest-shipped seq per
//! lane, and how long a lane has been behind. Reads never block on
//! the bucket: an unreachable bucket freezes the replica at its last
//! watermark (and the poll-age metric says so) instead of failing
//! reads.
//!
//! # What may write to the cache, and when
//!
//! Steady state, only the hydrator/tailer touch the data directory —
//! every locally-derived persistence path (usage counters, BM25
//! flushes, eviction's passage compaction) is suppressed under the
//! replica role. BOOT is the one deliberate exception: `boot_with`'s
//! scan may finish a shipped mid-rename and reconcile group records
//! exactly as any boot does. Those one-shot writes cannot loop — the
//! tailer's first diff detects any byte that diverged from the
//! manifest and refetches it, converging the cache — and keeping the
//! boot path identical to the writer's is what keeps a replica's view
//! of a directory from ever meaning something different than a
//! writer's would.
//!
//! # Promotion is manual, and a restart
//!
//! The series decision (#127) stands: no lease, no auto-failover. A
//! replica is promoted by stopping it and starting the same data
//! directory as the writer (`serve` without `--replica`): the
//! directory is already a verified cache at its watermark, so the
//! restart re-verifies cheaply (matching bytes are reused), and the
//! new writer's claim fences the old one wherever it is. The takeover
//! guard applies exactly as documented in `hydrate`: a cleanly
//! retired predecessor promotes without ceremony; a crashed one
//! demands `--take-over` until it ages out. What promotion cannot
//! recover is stated plainly: the old writer's acknowledged-but-
//! unshipped tail — the async-replication RPO, read straight off the
//! replica's lag metrics before flipping.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime};

use object_store::ObjectStore;
use object_store::path::Path as StorePath;
// parking_lot, not std::sync: a panic while the lock is held must not
// poison it and brick the write-refusal's fence-holder reporting for
// the rest of the process (the same reasoning as the registry — see
// Cargo.toml).
use parking_lot::Mutex;

use crate::hydrate::{self, Hydrator};
use crate::registry::AppState;
use crate::ship;

/// What the write-refusal needs to say: where writes actually go.
/// `writer_url` is the operator's own routing statement
/// (`TAGURU_WRITER_URL` — the name clients should call); the fence
/// holder is what the bucket says (`HOSTNAME#pid` at claim time),
/// kept fresh by the tailer as a best-effort supplement.
#[derive(Debug, Default)]
pub struct ReplicaInfo {
    writer_url: Option<String>,
    fence: Mutex<Option<(u64, Option<String>)>>,
}

impl ReplicaInfo {
    pub(crate) fn new(writer_url: Option<String>) -> Self {
        Self {
            writer_url,
            fence: Mutex::new(None),
        }
    }

    fn note_fence(&self, generation: u64, holder: Option<String>) {
        *self.fence.lock() = Some((generation, holder));
    }

    /// The refusal body every mutating verb answers with. Names the
    /// writer as precisely as this replica can: the operator-stated
    /// URL first, the bucket's fence holder as evidence, and an
    /// honest "no writer known" when the bucket has neither.
    pub fn refusal(&self) -> String {
        let mut message = String::from(
            "this instance is a read replica: it serves every retrieval \
                          verb, but writes go to the writer",
        );
        if let Some(url) = &self.writer_url {
            message.push_str(&format!(" at {url}"));
        }
        match &*self.fence.lock() {
            Some((generation, Some(holder))) => {
                message.push_str(&format!(
                    " (replication generation {generation}, claimed by {holder})"
                ));
            }
            Some((generation, None)) => {
                message.push_str(&format!(" (replication generation {generation})"));
            }
            None if self.writer_url.is_none() => {
                message.push_str(" (none known to the bucket yet)");
            }
            None => {}
        }
        message
    }
}

/// The serve-side handle: signals the tailer to stop and joins it.
pub(crate) struct TailerHandle {
    stop: Arc<AtomicBool>,
    wake: std::sync::mpsc::Sender<()>,
    thread: std::thread::JoinHandle<()>,
}

/// Bound on `TailerHandle::shutdown`'s wait for the tailer thread.
/// `poll_once` chains several sequential bucket calls (a fence lookup,
/// `newest_complete_generation`, `read_manifest`, `hydrate_shared`,
/// one `ensure_context` per stale stem) with no per-await cancellation
/// — `stop` is only checked between whole polls and once inside the
/// per-stem worklist loop, never at the earlier await points — so a
/// single hung bucket call can hold up the whole cycle. `object_store`
/// defaults each individual call's own timeout to 30s; this is a
/// multiple of that; margin for several such calls landing in one
/// cycle, not a promise that the tailer itself stops by then.
const TAILER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(90);

impl TailerHandle {
    /// Signals stop and waits for the tailer, bounded. `shutdown` runs
    /// inside `main`'s own shutdown sequence
    /// (`tokio::task::block_in_place(|| tailer.shutdown())`) — a plain
    /// `JoinHandle::join` here would block the WHOLE process's clean
    /// exit for as long as a hung poll takes, which `poll_once`'s lack
    /// of per-await cancellation (see [`TAILER_SHUTDOWN_TIMEOUT`])
    /// could make indefinite in the worst case.
    pub(crate) fn shutdown(self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.wake.send(());
        match join_bounded(self.thread, TAILER_SHUTDOWN_TIMEOUT) {
            JoinOutcome::Clean => {}
            JoinOutcome::Panicked => tracing::warn!("replica tailer did not shut down cleanly"),
            JoinOutcome::TimedOut => tracing::warn!(
                timeout_secs = TAILER_SHUTDOWN_TIMEOUT.as_secs(),
                "replica tailer did not stop within the shutdown timeout (a bucket call is \
                 likely hung); continuing shutdown without it",
            ),
            JoinOutcome::CouldNotSpawnWatchdog => tracing::warn!(
                "could not spawn a thread to bound the replica tailer's shutdown wait; \
                 the tailer thread is now unjoined and will finish on its own"
            ),
        }
    }
}

#[derive(Debug)]
enum JoinOutcome {
    Clean,
    Panicked,
    TimedOut,
    CouldNotSpawnWatchdog,
}

/// Joins `thread`, bounded by `timeout` — `JoinHandle::join` itself
/// has no timeout in std, so the join runs on a detached helper
/// thread instead, and the caller waits on THAT thread through a
/// channel it can time out on. Only one thread may ever call
/// `JoinHandle::join` on a given handle, which is why this spawns a
/// helper rather than racing the join against a timer directly.
///
/// Past the bound, this returns `TimedOut` and the original `thread`
/// stays unjoined — not aborted, not signaled, just no longer waited
/// on. It finishes on its own (or is torn down with the process) at
/// whatever pace the thing it is blocked on allows.
fn join_bounded(thread: std::thread::JoinHandle<()>, timeout: Duration) -> JoinOutcome {
    let (done, waited) = std::sync::mpsc::channel();
    let joiner = std::thread::Builder::new()
        .name("taguru-replica-shutdown-wait".into())
        .spawn(move || {
            let clean = thread.join().is_ok();
            let _ = done.send(clean);
        });
    let Ok(joiner) = joiner else {
        // `thread` moved into the closure above regardless of whether
        // the spawn itself succeeded — std hands back no way to
        // reclaim it from a failed `spawn`, so there is no "join it
        // directly" fallback left to fall back to.
        return JoinOutcome::CouldNotSpawnWatchdog;
    };
    let outcome = match waited.recv_timeout(timeout) {
        Ok(true) => JoinOutcome::Clean,
        Ok(false) => JoinOutcome::Panicked,
        Err(_) => JoinOutcome::TimedOut,
    };
    // Detached either way: joining the watchdog here would just
    // reintroduce the unbounded wait this function exists to avoid
    // (the watchdog itself only returns once ITS join finishes).
    drop(joiner);
    outcome
}

/// Boots the tailer on its own thread (with its own small runtime,
/// like every hydration worker): poll, apply, sleep, forever. The
/// FIRST pass doubles as the background fill — a fresh boot's
/// families are all stale relative to the manifest, so the tailer
/// materializes whatever the pinned preload and first touch have not,
/// and the lag metrics settle to zero as it goes.
pub(crate) fn spawn(
    store: Arc<dyn ObjectStore>,
    root: StorePath,
    replicate: ship::ReplicateConfig,
    data_dir: PathBuf,
    state: AppState,
    hydrator: Arc<Hydrator>,
    info: Arc<ReplicaInfo>,
) -> TailerHandle {
    let ship::ReplicateConfig { url, interval } = replicate;
    let stop = Arc::new(AtomicBool::new(false));
    let (wake, waker) = std::sync::mpsc::channel::<()>();
    let stopping = Arc::clone(&stop);
    let thread = std::thread::Builder::new()
        .name("taguru-replica".into())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("building the replica tailer runtime");
            let mut tailer = Tailer {
                store,
                root,
                url,
                data_dir,
                state,
                hydrator,
                info,
                stop: stopping,
                manifest_stamp: None,
                fence_seen: None,
                pending_refresh: Default::default(),
            };
            loop {
                match runtime.block_on(tailer.poll_once()) {
                    Ok(()) => tailer.state.metrics().record_replica_poll(true),
                    Err(error) => {
                        tailer.state.metrics().record_replica_poll(false);
                        tracing::warn!(%error, "replica poll failed; will retry");
                    }
                }
                if tailer.stop.load(Ordering::Relaxed) {
                    return;
                }
                match waker.recv_timeout(interval) {
                    // A wake or a closed channel both mean "stop now".
                    Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                }
            }
        })
        .expect("spawning the replica tailer thread");
    TailerHandle { stop, wake, thread }
}

struct Tailer {
    store: Arc<dyn ObjectStore>,
    root: StorePath,
    url: String,
    data_dir: PathBuf,
    state: AppState,
    hydrator: Arc<Hydrator>,
    info: Arc<ReplicaInfo>,
    stop: Arc<AtomicBool>,
    /// The followed manifest as last applied in full: its generation
    /// and the `complete` object's store-clock `last_modified`.
    /// Advanced only after a whole diff lands, so a partial failure
    /// re-reads and re-diffs — retarget is idempotent and settled
    /// families skip in O(1).
    manifest_stamp: Option<(u64, SystemTime)>,
    /// The newest fence generation whose body was fetched (one GET
    /// per new claimant, for the refusal's holder string).
    fence_seen: Option<u64>,
    /// Stems owed a [`AppState::replica_refresh`]: added when
    /// `retarget` reports them stale, removed only once a poll
    /// actually refreshes them. The debt outlives the staleness
    /// signal — when this tailer's own hydration attempt fails and a
    /// per-request loader (`ensure_hot`, the passage first touch)
    /// completes the same stem before the next poll, `retarget` sees
    /// the family signature already current and never reports the
    /// stem stale again, yet the entry's in-memory meta (pinned,
    /// description, revision/cache bookkeeping) was never re-read.
    pending_refresh: std::collections::BTreeSet<String>,
}

impl Tailer {
    /// One poll: cheap when nothing moved (a fence list plus a couple
    /// of HEADs), a verified diff-and-apply when something did.
    async fn poll_once(&mut self) -> std::io::Result<()> {
        let fence = ship::newest_fence(&self.store, &self.root)
            .await
            .map_err(std::io::Error::from)?;
        if let Some(fence) = fence
            && self.fence_seen != Some(fence.generation)
        {
            let holder =
                ship::fence_holder(self.store.as_ref(), &self.root, fence.generation).await;
            let resolved = holder.is_some();
            self.info.note_fence(fence.generation, holder);
            // A transient GET failure must not silence the refusal's
            // holder line for this generation forever: only a resolved
            // body settles the lookup; an unresolved one retries next
            // poll (one small GET — a fence body that genuinely never
            // parses costs that per poll, and nothing else).
            if resolved {
                self.fence_seen = Some(fence.generation);
            }
        }
        let generation =
            match ship::newest_complete_generation(self.store.as_ref(), &self.root).await {
                Ok(generation) => generation,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    // Nothing complete to tail (a virgin bucket, or a
                    // claimant mid-baseline): keep serving what we
                    // have; the next poll keeps looking.
                    return Ok(());
                }
                Err(error) => return Err(error),
            };
        let generation_root = ship::gen_root(&self.root, generation);
        let stamp = hydrate::head_modified(&self.store, &ship::complete_key(&generation_root))
            .await?
            .ok_or_else(|| {
                std::io::Error::other(format!(
                    "generation {generation}: complete marker vanished between listing and head"
                ))
            })?;
        self.state.metrics().note_replica_manifest(stamp);
        if self.manifest_stamp == Some((generation, stamp)) {
            return Ok(());
        }

        let Some(manifest) = ship::read_manifest(self.store.as_ref(), &generation_root).await?
        else {
            tracing::warn!(
                generation,
                "the newest complete generation predates the shipping manifest; \
                 staying at the last verified watermark until a writer ships once"
            );
            return Ok(());
        };

        let switched = self.hydrator.generation() != Some(generation);
        if switched {
            tracing::info!(
                generation,
                from = self.hydrator.generation(),
                "the bucket lineage moved to a new generation; re-verifying the cache \
                 against it"
            );
        }

        let report = self
            .hydrator
            .retarget(generation, generation_root, manifest.clone());
        self.state.metrics().note_replica_generation(generation);
        if switched {
            // Applied seqs are per-lineage: see `reset_replica_lanes`.
            self.state.metrics().reset_replica_lanes();
        }
        // Stale stems join the refresh debt BEFORE anything below can
        // fail the poll; the worklist is the whole debt, not this
        // retarget's report — a stem whose hydration a per-request
        // loader completed after this tailer's own failed attempt (a
        // family fetch below, or `hydrate_shared` erroring out of the
        // whole poll) never turns stale again, but its refresh is
        // still owed (`ensure_context` on a settled stem is O(1), so
        // paying the debt late costs one meta re-read, not a
        // re-hydration).
        self.pending_refresh.extend(report.stale.iter().cloned());
        for stem in &report.vanished {
            let Some(name) = crate::registry::name_from_stem(stem) else {
                continue;
            };
            tracing::info!(context = %name, "the lineage no longer carries this context; dropping it");
            self.state.replica_deregister(&name);
            self.state.metrics().forget_replica_context(&name);
            self.pending_refresh.remove(stem);
        }
        // Shared files (groups, the grant store, every sidecar meta)
        // next, so the per-family passes below see fresh metas and the
        // relics of vanished families leave the disk.
        self.hydrator.hydrate_shared().await?;
        self.state.replica_reload_groups();

        let worklist: Vec<String> = self.pending_refresh.iter().cloned().collect();
        let mut failed: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for stem in &worklist {
            if self.stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            let Some(name) = crate::registry::name_from_stem(stem) else {
                // Undecodable: never registered, so it was never
                // hydrated either — it must count as failed, or the
                // per-lane metrics below (keyed on `failed`) would
                // report this lane as fully caught up.
                failed.insert(stem.as_str());
                continue;
            };
            self.state.replica_register(stem);
            if let Err(error) = self.hydrator.ensure_context(stem) {
                tracing::warn!(
                    context = %name,
                    %error,
                    "tailing this context failed; it stays at its last watermark"
                );
                failed.insert(stem.as_str());
                continue;
            }
            self.state.replica_refresh(&name);
            self.pending_refresh.remove(stem);
        }
        // Lag rows for every lane the manifest carries. A family that
        // landed (this pass or any earlier one — retarget reported
        // only the moved ones) is applied AT the shipped seq; one that
        // failed keeps its old applied value (or none) and shows the
        // gap, with the age counting from the first poll that saw it
        // behind.
        for (lane_name, lane) in &manifest.lanes {
            let (context, lane_label) = ship::lane_metric_labels(lane_name);
            let stem = lane_name
                .strip_suffix(".passages.wal.jsonl")
                .or_else(|| lane_name.strip_suffix(".wal.jsonl"))
                .unwrap_or(lane_name);
            if failed.contains(stem) {
                self.state
                    .metrics()
                    .note_replica_shipped(&context, lane_label, lane.seq);
            } else {
                self.state
                    .metrics()
                    .note_replica_lane(&context, lane_label, lane.seq, lane.seq);
            }
        }
        if !failed.is_empty() {
            return Err(std::io::Error::other(format!(
                "{} context families could not be tailed; retrying next poll",
                failed.len()
            )));
        }
        if switched {
            // The record's `hydrated_from` is a VERIFIED watermark —
            // degraded boots trust it — so it advances only after the
            // whole diff landed. A crash mid-switch leaves the record
            // naming the last fully-applied generation: an honest
            // claim for an unreachable-bucket boot, and a reachable
            // one re-verifies against the newest lineage regardless.
            ship::write_replication_record(
                &self.data_dir,
                &ship::ReplicationRecord {
                    url: self.url.clone(),
                    claimed_generation: None,
                    hydrated_from: Some(generation),
                },
            )?;
        }
        self.manifest_stamp = Some((generation, stamp));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ship::{ShipProgress, Shipper};
    use crate::wal::{self, WalOp};
    use std::path::Path as FsPath;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("taguru-replica-{tag}-{}", std::process::id()));
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

    fn url_of(tag: &str) -> String {
        format!("file://taguru-replica-test-{tag}")
    }

    /// A bucket holding one context whose graph lane spans TWO shipped
    /// segments — the shape the torn-segment fault needs.
    async fn two_segment_bucket(tag: &str) -> (PathBuf, PathBuf) {
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
            Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES)),
            state,
            None,
        )
        .await
        .unwrap();
        shipper.cycle().await.unwrap();
        wal::append_batch(&writer.join("ctx_a.wal.jsonl"), 2, &[associate("b")]).unwrap();
        shipper.cycle().await.unwrap();
        shipper.retire_generation().await;
        (bucket, writer)
    }

    fn tailer_for(
        bucket: &FsPath,
        url: String,
        data_dir: PathBuf,
        state: AppState,
        hydrator: Arc<Hydrator>,
    ) -> Tailer {
        Tailer {
            store: local_store(bucket),
            root: StorePath::default(),
            url,
            data_dir,
            state,
            hydrator,
            info: Arc::new(ReplicaInfo::new(None)),
            stop: Arc::new(AtomicBool::new(false)),
            manifest_stamp: None,
            fence_seen: None,
            pending_refresh: Default::default(),
        }
    }

    fn scrape(state: &AppState) -> String {
        state.metrics().render_prometheus(&state.gauge_snapshot())
    }

    #[tokio::test]
    async fn a_torn_segment_fails_the_poll_cleanly_and_heals_on_recovery() {
        let (bucket, writer) = two_segment_bucket("torn").await;
        let url = url_of("torn");
        let store = local_store(&bucket);
        let target = scratch("torn-target");
        let hydrator =
            crate::hydrate::prepare_replica(&store, &StorePath::default(), &url, &target)
                .await
                .expect("hydrates");
        let state = AppState::boot(target.clone(), 64 * 1024 * 1024, None).unwrap();
        state.metrics().set_replica_mode();

        // Tear the lane: the manifest advertises two segments; the
        // second vanishes mid-poll (an eventual-consistency read, a
        // mid-upload crash — the shape is the same).
        let lane_dir = bucket
            .join("gen-00000000000000000001")
            .join("wal")
            .join("ctx_a.wal.jsonl");
        let segments: Vec<_> = std::fs::read_dir(&lane_dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(segments.len(), 2, "the fixture ships two segments");
        let torn = lane_dir.join("torn-aside");
        std::fs::rename(segments.iter().max().unwrap(), &torn).unwrap();

        let mut tailer = tailer_for(
            &bucket,
            url.clone(),
            target.clone(),
            state.clone(),
            hydrator,
        );
        let error = tailer
            .poll_once()
            .await
            .expect_err("a torn lane fails the poll");
        assert!(error.to_string().contains("could not be tailed"), "{error}");
        assert!(
            !target.join("ctx_a.wal.jsonl").exists(),
            "an un-assemblable lane lands nothing — no partial file"
        );
        let text = scrape(&state);
        assert!(
            text.contains("taguru_replica_shipped_seq{context=\"ctx_a\",lane=\"graph\"} 2"),
            "{text}"
        );
        assert!(
            text.contains("taguru_replica_applied_seq{context=\"ctx_a\",lane=\"graph\"} 0"),
            "{text}"
        );

        // The segment returns; the next poll heals the family whole.
        std::fs::rename(&torn, segments.iter().max().unwrap()).unwrap();
        tailer
            .poll_once()
            .await
            .expect("the retry lands the family");
        assert_eq!(
            std::fs::read(target.join("ctx_a.wal.jsonl")).unwrap(),
            std::fs::read(writer.join("ctx_a.wal.jsonl")).unwrap(),
            "the healed lane is byte-identical to the shipped stream"
        );
        let text = scrape(&state);
        assert!(
            text.contains("taguru_replica_applied_seq{context=\"ctx_a\",lane=\"graph\"} 2"),
            "{text}"
        );
        assert!(
            text.contains("taguru_replica_behind_seconds{context=\"ctx_a\",lane=\"graph\"} 0"),
            "{text}"
        );

        for dir in [bucket, writer, target] {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    /// The refresh debt outlives the staleness signal: when the
    /// tailer's own hydration attempt fails and a per-request loader
    /// completes the same stem before the next poll, `retarget` sees
    /// the family signature already current and never reports the stem
    /// stale again — without the remembered debt, the entry's
    /// in-memory meta (description, pinned) would stay frozen at the
    /// pre-failure state indefinitely while the served data is fresh.
    #[tokio::test]
    async fn a_refresh_owed_from_a_failed_poll_lands_even_when_a_reader_hydrates_first() {
        let bucket = scratch("owed-bucket");
        let writer = scratch("owed-writer");
        std::fs::write(writer.join("ctx_a.ctx"), b"image-v1").unwrap();
        std::fs::write(
            writer.join("ctx_a.meta.json"),
            br#"{"description":"old","pinned":false}"#,
        )
        .unwrap();
        wal::append_batch(&writer.join("ctx_a.wal.jsonl"), 1, &[associate("a")]).unwrap();
        let writer_state = AppState::boot(writer.clone(), 64 * 1024 * 1024, None).unwrap();
        let mut shipper = Shipper::claim(
            local_store(&bucket),
            StorePath::default(),
            url_of("owed"),
            writer.clone(),
            Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES)),
            writer_state,
            None,
        )
        .await
        .unwrap();
        shipper.cycle().await.unwrap();

        let url = url_of("owed");
        let store = local_store(&bucket);
        let target = scratch("owed-target");
        let hydrator =
            crate::hydrate::prepare_replica(&store, &StorePath::default(), &url, &target)
                .await
                .expect("hydrates");
        let state = AppState::boot(target.clone(), 64 * 1024 * 1024, None).unwrap();
        state.metrics().set_replica_mode();
        let mut tailer = tailer_for(
            &bucket,
            url.clone(),
            target.clone(),
            state.clone(),
            hydrator.clone(),
        );
        tailer.poll_once().await.expect("the first manifest lands");
        assert_eq!(state.directory_entry("ctx_a").unwrap().description, "old");

        // The writer moves the meta and ships a second segment.
        std::fs::write(
            writer.join("ctx_a.meta.json"),
            br#"{"description":"new","pinned":false}"#,
        )
        .unwrap();
        wal::append_batch(&writer.join("ctx_a.wal.jsonl"), 2, &[associate("b")]).unwrap();
        shipper.cycle().await.unwrap();
        shipper.retire_generation().await;

        // Tear the new segment: the tailer's own attempt fails, the
        // refresh is skipped, and the manifest stamp does not advance.
        let lane_dir = bucket
            .join("gen-00000000000000000001")
            .join("wal")
            .join("ctx_a.wal.jsonl");
        let segments: Vec<_> = std::fs::read_dir(&lane_dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        let torn = lane_dir.join("torn-aside");
        std::fs::rename(segments.iter().max().unwrap(), &torn).unwrap();
        tailer
            .poll_once()
            .await
            .expect_err("the torn lane fails this poll");

        // The transient clears and a per-request loader (ensure_hot's
        // shape) hydrates the stem first: the family signature is now
        // current, so the next retarget will not report it stale.
        std::fs::rename(&torn, segments.iter().max().unwrap()).unwrap();
        hydrator
            .ensure_context("ctx_a")
            .expect("the reader's own hydration lands");
        assert_eq!(
            state.directory_entry("ctx_a").unwrap().description,
            "old",
            "hydration alone re-reads no meta — the refresh is still owed"
        );

        tailer.poll_once().await.expect("the owed refresh lands");
        assert_eq!(
            state.directory_entry("ctx_a").unwrap().description,
            "new",
            "the remembered debt must pay out even though retarget \
             reported nothing stale"
        );

        for dir in [bucket, writer, target] {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    /// The debt must be recorded before `hydrate_shared` can fail the
    /// poll: a torn shared object (here the published meta.json)
    /// aborts the poll before any family applies — and if a
    /// per-request loader hydrates the stem before the next poll,
    /// retarget never reports it stale again, so a debt recorded only
    /// after the shared pass would have been lost with the error.
    #[tokio::test]
    async fn a_refresh_owed_survives_a_shared_hydration_failure() {
        let bucket = scratch("shared-owed-bucket");
        let writer = scratch("shared-owed-writer");
        std::fs::write(writer.join("ctx_a.ctx"), b"image-v1").unwrap();
        std::fs::write(
            writer.join("ctx_a.meta.json"),
            br#"{"description":"old","pinned":false}"#,
        )
        .unwrap();
        wal::append_batch(&writer.join("ctx_a.wal.jsonl"), 1, &[associate("a")]).unwrap();
        let writer_state = AppState::boot(writer.clone(), 64 * 1024 * 1024, None).unwrap();
        let mut shipper = Shipper::claim(
            local_store(&bucket),
            StorePath::default(),
            url_of("shared-owed"),
            writer.clone(),
            Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES)),
            writer_state,
            None,
        )
        .await
        .unwrap();
        shipper.cycle().await.unwrap();

        let url = url_of("shared-owed");
        let store = local_store(&bucket);
        let target = scratch("shared-owed-target");
        let hydrator =
            crate::hydrate::prepare_replica(&store, &StorePath::default(), &url, &target)
                .await
                .expect("hydrates");
        let state = AppState::boot(target.clone(), 64 * 1024 * 1024, None).unwrap();
        state.metrics().set_replica_mode();
        let mut tailer = tailer_for(
            &bucket,
            url.clone(),
            target.clone(),
            state.clone(),
            hydrator.clone(),
        );
        tailer.poll_once().await.expect("the first manifest lands");
        assert_eq!(state.directory_entry("ctx_a").unwrap().description, "old");

        std::fs::write(
            writer.join("ctx_a.meta.json"),
            br#"{"description":"new","pinned":false}"#,
        )
        .unwrap();
        wal::append_batch(&writer.join("ctx_a.wal.jsonl"), 2, &[associate("b")]).unwrap();
        shipper.cycle().await.unwrap();
        shipper.retire_generation().await;

        // Tear the published meta object: the shared pass fails the
        // whole poll before any family is even attempted.
        let meta_object = bucket
            .join("gen-00000000000000000001")
            .join("files")
            .join("ctx_a.meta.json");
        let torn = meta_object.with_extension("torn-aside");
        std::fs::rename(&meta_object, &torn).unwrap();
        tailer
            .poll_once()
            .await
            .expect_err("a torn shared object fails the poll");

        // The transient clears and a reader hydrates the family first:
        // the next retarget will not report the stem stale.
        std::fs::rename(&torn, &meta_object).unwrap();
        hydrator
            .ensure_context("ctx_a")
            .expect("the reader's own hydration lands");
        assert_eq!(
            state.directory_entry("ctx_a").unwrap().description,
            "old",
            "hydration alone re-reads no meta — the refresh is still owed"
        );

        tailer.poll_once().await.expect("the owed refresh lands");
        assert_eq!(
            state.directory_entry("ctx_a").unwrap().description,
            "new",
            "a debt recorded only after the shared pass would have been \
             lost with the error"
        );

        for dir in [bucket, writer, target] {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[tokio::test]
    async fn a_restart_mid_apply_re_verifies_the_cache_and_heals() {
        let (bucket, writer) = two_segment_bucket("midapply").await;
        let url = url_of("midapply");
        let store = local_store(&bucket);
        let target = scratch("midapply-target");
        let hydrator =
            crate::hydrate::prepare_replica(&store, &StorePath::default(), &url, &target)
                .await
                .expect("hydrates");
        let state = AppState::boot(target.clone(), 64 * 1024 * 1024, None).unwrap();
        let mut tailer = tailer_for(&bucket, url.clone(), target.clone(), state, hydrator);
        tailer.poll_once().await.expect("the first sync lands");
        drop(tailer);

        // A crash mid-apply, restaged: one family member torn to
        // garbage, another missing entirely.
        std::fs::write(target.join("ctx_a.wal.jsonl"), b"torn mid-write").unwrap();
        std::fs::remove_file(target.join("ctx_a.ctx")).unwrap();

        // The restart's boot decision re-verifies the whole cache —
        // the same prepare a real `serve --replica` runs.
        let hydrator =
            crate::hydrate::prepare_replica(&store, &StorePath::default(), &url, &target)
                .await
                .expect("a cache re-verifies after a crash");
        hydrator.ensure_context("ctx_a").unwrap();
        assert_eq!(
            std::fs::read(target.join("ctx_a.ctx")).unwrap(),
            b"image-v1"
        );
        assert_eq!(
            std::fs::read(target.join("ctx_a.wal.jsonl")).unwrap(),
            std::fs::read(writer.join("ctx_a.wal.jsonl")).unwrap(),
            "the torn lane is refetched whole"
        );

        for dir in [bucket, writer, target] {
            let _ = std::fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn the_refusal_names_what_the_replica_knows() {
        let bare = ReplicaInfo::new(None);
        assert!(bare.refusal().contains("none known"), "{}", bare.refusal());

        bare.note_fence(3, Some("db-1#42".into()));
        let named = bare.refusal();
        assert!(named.contains("generation 3"), "{named}");
        assert!(named.contains("db-1#42"), "{named}");

        let routed = ReplicaInfo::new(Some("http://writer.internal:8248".into()));
        assert!(
            routed.refusal().contains("http://writer.internal:8248"),
            "{}",
            routed.refusal()
        );
    }

    /// #616 item 4: `join_bounded` must return promptly on a thread
    /// that never finishes — standing in for `poll_once` stuck on a
    /// hung bucket call with no per-await cancellation — rather than
    /// blocking the caller (in production, the whole process's
    /// shutdown sequence) for as long as the thread takes.
    #[test]
    fn join_bounded_times_out_on_a_thread_that_never_finishes() {
        let (release, held) = std::sync::mpsc::channel::<()>();
        let thread = std::thread::spawn(move || {
            let _ = held.recv();
        });
        let started = std::time::Instant::now();
        let outcome = join_bounded(thread, Duration::from_millis(50));
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "must return promptly instead of waiting for the hung thread: {:?}",
            started.elapsed()
        );
        assert!(matches!(outcome, JoinOutcome::TimedOut));
        // Release the leaked thread so it does not outlive the test
        // binary for no reason.
        let _ = release.send(());
    }

    /// The companion case: a thread that finishes well within the
    /// bound reports `Clean`, not a spurious timeout.
    #[test]
    fn join_bounded_reports_clean_for_a_thread_that_finishes_in_time() {
        let thread = std::thread::spawn(|| {});
        let outcome = join_bounded(thread, Duration::from_secs(5));
        assert!(matches!(outcome, JoinOutcome::Clean), "{outcome:?}");
    }

    /// And a thread whose body panics reports `Panicked`, matching
    /// what `shutdown` warns on today (`join().is_err()`).
    #[test]
    fn join_bounded_reports_panicked_for_a_thread_that_panics() {
        let thread = std::thread::Builder::new()
            .spawn(|| panic!("intentional panic for join_bounded's Panicked arm"))
            .unwrap();
        let outcome = join_bounded(thread, Duration::from_secs(5));
        assert!(matches!(outcome, JoinOutcome::Panicked), "{outcome:?}");
    }
}
