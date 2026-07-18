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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use object_store::ObjectStore;
use object_store::path::Path as StorePath;

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
        *self.fence.lock().unwrap() = Some((generation, holder));
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
        match &*self.fence.lock().unwrap() {
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

impl TailerHandle {
    pub(crate) fn shutdown(self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.wake.send(());
        if self.thread.join().is_err() {
            tracing::warn!("replica tailer did not shut down cleanly");
        }
    }
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
            self.info.note_fence(fence.generation, holder);
            self.fence_seen = Some(fence.generation);
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

        if self.hydrator.generation() != Some(generation) {
            tracing::info!(
                generation,
                from = self.hydrator.generation(),
                "the bucket lineage moved to a new generation; re-verifying the cache \
                 against it"
            );
            // Durably restate what this directory is a cache of BEFORE
            // touching files, so a crash mid-switch re-verifies against
            // the new generation instead of trusting a half-moved cache.
            ship::write_replication_record(
                &self.data_dir,
                &ship::ReplicationRecord {
                    url: self.url.clone(),
                    claimed_generation: None,
                    hydrated_from: Some(generation),
                },
            )?;
        }

        let report = self
            .hydrator
            .retarget(generation, generation_root, manifest.clone());
        self.state.metrics().note_replica_generation(generation);
        for stem in &report.vanished {
            let Some(name) = crate::registry::name_from_stem(stem) else {
                continue;
            };
            tracing::info!(context = %name, "the lineage no longer carries this context; dropping it");
            self.state.replica_deregister(&name);
            self.state.metrics().forget_replica_context(&name);
        }
        // Shared files (groups, the grant store, every sidecar meta)
        // next, so the per-family passes below see fresh metas and the
        // relics of vanished families leave the disk.
        self.hydrator.hydrate_shared().await?;
        self.state.replica_reload_groups();

        let mut failed: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for stem in &report.stale {
            if self.stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            self.state.replica_register(stem);
            let Some(name) = crate::registry::name_from_stem(stem) else {
                continue;
            };
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
            Arc::new(ShipProgress::new()),
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
}
