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
    claimed_dyn(
        Arc::clone(store) as Arc<dyn ObjectStore>,
        dir,
        state,
        progress,
    )
    .await
}

async fn claimed_dyn(
    store: Arc<dyn ObjectStore>,
    dir: &FsPath,
    state: &AppState,
    progress: &Arc<ShipProgress>,
) -> Shipper {
    Shipper::claim(
        store,
        StorePath::default(),
        "mem://test".to_string(),
        dir.to_path_buf(),
        Arc::clone(progress),
        state.clone(),
        None,
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

/// A store that delegates every call to `inner` except two dials
/// this test module turns: `fail_puts` and `fail_deletes` each
/// count down by one on a matching call, answering a transient I/O
/// error while `> 0` and falling through to the real store once
/// they hit zero — "the bucket flaked N times, then came back",
/// the shape a retry-on-next-cycle fix needs proof against.
#[derive(Debug)]
struct FlakyStore {
    inner: Arc<dyn ObjectStore>,
    fail_puts: Arc<Mutex<usize>>,
    fail_deletes: Arc<Mutex<usize>>,
}

impl std::fmt::Display for FlakyStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FlakyStore({})", self.inner)
    }
}

fn injected(operation: &'static str) -> object_store::Error {
    object_store::Error::Generic {
        store: "flaky",
        source: format!("injected transient {operation} failure").into(),
    }
}

#[async_trait::async_trait]
impl ObjectStore for FlakyStore {
    async fn put_opts(
        &self,
        location: &StorePath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        {
            let mut remaining = self.fail_puts.lock();
            if *remaining > 0 {
                *remaining -= 1;
                return Err(injected("put"));
            }
        }
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &StorePath,
        opts: object_store::PutMultipartOptions,
    ) -> object_store::Result<Box<dyn object_store::MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(
        &self,
        location: &StorePath,
        options: object_store::GetOptions,
    ) -> object_store::Result<object_store::GetResult> {
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: futures_util::stream::BoxStream<'static, object_store::Result<StorePath>>,
    ) -> futures_util::stream::BoxStream<'static, object_store::Result<StorePath>> {
        let inner = Arc::clone(&self.inner);
        let fail_deletes = Arc::clone(&self.fail_deletes);
        locations
            .then(move |location| {
                let inner = Arc::clone(&inner);
                let fail_deletes = Arc::clone(&fail_deletes);
                async move {
                    let location = location?;
                    {
                        let mut remaining = fail_deletes.lock();
                        if *remaining > 0 {
                            *remaining -= 1;
                            return Err(injected("delete"));
                        }
                    }
                    inner.delete(&location).await?;
                    Ok(location)
                }
            })
            .boxed()
    }

    fn list(
        &self,
        prefix: Option<&StorePath>,
    ) -> futures_util::stream::BoxStream<'static, object_store::Result<object_store::ObjectMeta>>
    {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&StorePath>,
    ) -> object_store::Result<object_store::ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &StorePath,
        to: &StorePath,
        options: object_store::CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

#[tokio::test]
async fn claims_are_monotonic_and_a_race_converges_on_distinct_generations() {
    let dir = scratch_dir("claim");
    let state = state_for(&dir);
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
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
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
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
            &"gen-00000000000000000001/wal/ctx_a.wal.jsonl/0000000000-0000000000.jsonl".to_string()
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
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
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
            &"gen-00000000000000000001/wal/ctx_a.wal.jsonl/0000000001-0000000000.jsonl".to_string()
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
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
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
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
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
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
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

/// A retire whose remote delete fails transiently must leave the
/// name retryable — not silently orphan the object in the bucket.
/// Before the fix, `retire_file` dropped `self.lanes`/`self.files`
/// before the delete's result was known, so a failed delete still
/// erased the only record (`scan.vanished`) that would have asked
/// for a retry.
#[tokio::test]
async fn a_failed_retire_delete_is_retried_not_orphaned() {
    let dir = scratch_dir("retire-flaky");
    let state = state_for(&dir);
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
    let inner = Arc::new(InMemory::new());
    let fail_deletes = Arc::new(Mutex::new(0usize));
    let store: Arc<dyn ObjectStore> = Arc::new(FlakyStore {
        inner: Arc::clone(&inner) as Arc<dyn ObjectStore>,
        fail_puts: Arc::new(Mutex::new(0)),
        fail_deletes: Arc::clone(&fail_deletes),
    });

    std::fs::write(dir.join("ctx_a.ctx"), b"image-v1").unwrap();
    let mut shipper = claimed_dyn(Arc::clone(&store), &dir, &state, &progress).await;
    shipper.cycle().await.unwrap();
    assert!(
        object_names(&inner)
            .await
            .iter()
            .any(|name| name.contains("ctx_a")),
        "the file must have shipped before this test deletes it locally"
    );

    std::fs::remove_file(dir.join("ctx_a.ctx")).unwrap();
    *fail_deletes.lock() = 1;
    let error = shipper.cycle().await.unwrap_err();
    assert!(matches!(error, ShipError::Io(_)), "{error}");
    assert!(
        object_names(&inner)
            .await
            .iter()
            .any(|name| name.contains("ctx_a")),
        "the object must still be in the bucket after the failed delete"
    );

    // The next cycle retries — nothing local changed, but the
    // vanished name must still be there to retire.
    shipper.cycle().await.unwrap();
    let names = object_names(&inner).await;
    assert!(
        !names.iter().any(|name| name.contains("ctx_a")),
        "the retry must finish retiring the object: {names:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The same guarantee as the published-file case above, but for a
/// log lane: `retire_file`'s two branches (published file, lane)
/// must behave identically on a failed remote delete — retryable,
/// not orphaned. A lane's delete goes through `delete_prefix`
/// (segment-by-segment), a different code path from the file
/// branch's single `store.delete`, so this needs its own case.
#[tokio::test]
async fn a_failed_lane_retire_delete_is_retried_not_orphaned() {
    let dir = scratch_dir("retire-lane-flaky");
    let state = state_for(&dir);
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
    let inner = Arc::new(InMemory::new());
    let fail_deletes = Arc::new(Mutex::new(0usize));
    let store: Arc<dyn ObjectStore> = Arc::new(FlakyStore {
        inner: Arc::clone(&inner) as Arc<dyn ObjectStore>,
        fail_puts: Arc::new(Mutex::new(0)),
        fail_deletes: Arc::clone(&fail_deletes),
    });

    let wal_path = dir.join("ctx_a.wal.jsonl");
    wal::append_batch(&wal_path, 1, &[associate("a")]).unwrap();
    let mut shipper = claimed_dyn(Arc::clone(&store), &dir, &state, &progress).await;
    shipper.cycle().await.unwrap();
    assert!(
        object_names(&inner)
            .await
            .iter()
            .any(|name| name.contains("ctx_a")),
        "the lane must have shipped before this test deletes it locally"
    );

    std::fs::remove_file(&wal_path).unwrap();
    *fail_deletes.lock() = 1;
    let error = shipper.cycle().await.unwrap_err();
    assert!(matches!(error, ShipError::Io(_)), "{error}");
    assert!(
        object_names(&inner)
            .await
            .iter()
            .any(|name| name.contains("ctx_a")),
        "the lane's segments must still be in the bucket after the failed delete"
    );

    // The next cycle retries — nothing local changed, but the
    // vanished lane must still be there to retire.
    shipper.cycle().await.unwrap();
    let names = object_names(&inner).await;
    assert!(
        !names.iter().any(|name| name.contains("ctx_a")),
        "the retry must finish retiring the lane: {names:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A `put_manifest` that fails on its own — every file/lane upload
/// in the cycle already succeeded — must not go unretried. Before
/// the fix, the retry condition was `shipped` (this cycle's own
/// upload activity), which a context deletion clears from
/// `self.files`/`self.lanes` before the next cycle even looks —
/// so a manifest describing already-deleted objects would stand
/// forever once the server went idle.
#[tokio::test]
async fn a_failed_manifest_put_is_retried_on_the_next_idle_cycle() {
    let dir = scratch_dir("manifest-flaky");
    let state = state_for(&dir);
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
    let inner = Arc::new(InMemory::new());
    let fail_puts = Arc::new(Mutex::new(0usize));
    let store: Arc<dyn ObjectStore> = Arc::new(FlakyStore {
        inner: Arc::clone(&inner) as Arc<dyn ObjectStore>,
        fail_puts: Arc::clone(&fail_puts),
        fail_deletes: Arc::new(Mutex::new(0)),
    });

    std::fs::write(dir.join("ctx_a.ctx"), b"image-v1").unwrap();
    let mut shipper = claimed_dyn(Arc::clone(&store), &dir, &state, &progress).await;
    shipper.cycle().await.unwrap();
    assert!(
        !read_object(&inner, "gen-00000000000000000001/complete")
            .await
            .is_empty(),
        "the baseline manifest must land on the first cycle"
    );

    // Delete the context locally — the shipper retires the object
    // and, in the same cycle, tries to republish the manifest
    // without it. Fail only that manifest PUT.
    std::fs::remove_file(dir.join("ctx_a.ctx")).unwrap();
    *fail_puts.lock() = 1;
    let error = shipper.cycle().await.unwrap_err();
    assert!(matches!(error, ShipError::Io(_)), "{error}");
    assert!(
        !object_names(&inner)
            .await
            .iter()
            .any(|name| name.contains("ctx_a") && name.contains("files/")),
        "the retire itself must still have landed"
    );
    let stale = read_object(&inner, "gen-00000000000000000001/complete").await;
    let stale: Manifest = serde_json::from_slice(&stale).unwrap();
    assert!(
        stale.files.contains_key("ctx_a.ctx"),
        "the manifest is still the pre-retire one: {stale:?}"
    );

    // Nothing local changed, so an old (buggy) `shipped`-only gate
    // would see this cycle as having nothing to do and skip the
    // manifest PUT forever. The fix must retry it here.
    shipper.cycle().await.unwrap();
    let fresh = read_object(&inner, "gen-00000000000000000001/complete").await;
    let fresh: Manifest = serde_json::from_slice(&fresh).unwrap();
    assert!(
        !fresh.files.contains_key("ctx_a.ctx"),
        "the manifest must catch up to the retire on the very next cycle: {fresh:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn restore_refuses_a_gapped_segment_run() {
    let dir = scratch_dir("gap");
    let state = state_for(&dir);
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
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
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
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
    let progress = ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES);
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

#[tokio::test]
async fn the_manifest_records_every_shipped_extent_and_restore_verifies_it() {
    let dir = scratch_dir("manifest");
    let state = state_for(&dir);
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
    let store = Arc::new(InMemory::new());

    std::fs::write(dir.join("ctx_a.ctx"), b"image-v1").unwrap();
    let wal_path = dir.join("ctx_a.wal.jsonl");
    wal::append_batch(&wal_path, 1, &[associate("a")]).unwrap();
    let mut shipper = claimed(&store, &dir, &state, &progress).await;
    shipper.cycle().await.unwrap();

    let manifest: Manifest =
        serde_json::from_slice(&read_object(&store, "gen-00000000000000000001/complete").await)
            .expect("complete carries the manifest now");
    assert_eq!(manifest.generation, 1);
    let image = manifest
        .files
        .get("ctx_a.ctx")
        .expect("the image is listed");
    assert_eq!(image.len, 8);
    assert_eq!(image.crc, crate::crc32c::crc32c(b"image-v1"));
    let wal_bytes = std::fs::read(&wal_path).unwrap();
    let lane = manifest
        .lanes
        .get("ctx_a.wal.jsonl")
        .expect("the lane is listed");
    assert_eq!((lane.series, lane.segments, lane.seq), (0, 1, 1));
    assert_eq!(lane.len, wal_bytes.len() as u64);
    assert_eq!(lane.crc, crate::crc32c::crc32c(&wal_bytes));

    // Tampered bytes no longer restore quietly: the manifest CRC
    // refuses them by name.
    (store.as_ref() as &dyn ObjectStore)
        .put(
            &StorePath::parse("gen-00000000000000000001/files/ctx_a.ctx").unwrap(),
            PutPayload::from(b"tampered".to_vec()),
        )
        .await
        .unwrap();
    let restored_dir = scratch_dir("manifest-out");
    let error = restore_into(
        store.as_ref() as &dyn ObjectStore,
        &StorePath::default(),
        &restored_dir,
    )
    .await
    .expect_err("bytes that contradict the manifest must refuse");
    assert!(error.to_string().contains("manifest"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&restored_dir);
}

/// A manifest is bucket data, not the writer's own local state —
/// bucket-write access lets an attacker add a file entry under a
/// name like `"../escaped.txt"`, with the CRC computed over their
/// own uploaded bytes (so [`verify_file_bytes`] alone would pass
/// it). Restore must refuse the whole manifest before ever
/// `out.join`-ing that name, not just refuse the content mismatch.
#[tokio::test]
async fn restore_refuses_a_manifest_naming_a_path_outside_the_target_directory() {
    let dir = scratch_dir("manifest-traversal");
    let state = state_for(&dir);
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
    let store = Arc::new(InMemory::new());

    std::fs::write(dir.join("ctx_a.ctx"), b"image-v1").unwrap();
    let mut shipper = claimed(&store, &dir, &state, &progress).await;
    shipper.cycle().await.unwrap();

    // Tamper with the manifest itself, not just an object's bytes:
    // add an entry naming a path outside the restore target, with
    // its CRC computed over the attacker's own payload so content
    // verification alone would accept it.
    let generation_root = gen_root(&StorePath::default(), 1);
    let traversal_name = "../taguru-test-escaped-payload.txt";
    let payload = b"hijacked!".to_vec();
    let key = generation_root.clone().join("files").join(traversal_name);
    (store.as_ref() as &dyn ObjectStore)
        .put(&key, PutPayload::from(payload.clone()))
        .await
        .unwrap();
    let bytes = read_object(&store, "gen-00000000000000000001/complete").await;
    let mut manifest: Manifest = serde_json::from_slice(&bytes).unwrap();
    manifest.files.insert(
        traversal_name.to_string(),
        ManifestFile {
            len: payload.len() as u64,
            crc: crate::crc32c::crc32c(&payload),
        },
    );
    (store.as_ref() as &dyn ObjectStore)
        .put(
            &generation_root.clone().join(COMPLETE_MARKER),
            PutPayload::from(serde_json::to_vec(&manifest).unwrap()),
        )
        .await
        .unwrap();

    let restored_dir = scratch_dir("manifest-traversal-out");
    let escape_target = restored_dir
        .parent()
        .expect("scratch dirs always have a parent")
        .join("taguru-test-escaped-payload.txt");
    let _ = std::fs::remove_file(&escape_target);
    let error = restore_into(
        store.as_ref() as &dyn ObjectStore,
        &StorePath::default(),
        &restored_dir,
    )
    .await
    .expect_err("a manifest naming a path outside the target must refuse");
    assert!(
        error.to_string().contains("not a safe file name"),
        "{error}"
    );
    assert!(
        !escape_target.exists(),
        "the traversal name must never be written outside the restore target"
    );
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&restored_dir);
    let _ = std::fs::remove_file(&escape_target);
}

#[tokio::test]
async fn a_claim_is_recorded_locally_and_liveness_markers_land() {
    let dir = scratch_dir("liveness");
    let state = state_for(&dir);
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
    let store = Arc::new(InMemory::new());

    let mut shipper = claimed(&store, &dir, &state, &progress).await;
    let record = read_replication_record(&dir)
        .unwrap()
        .expect("the claim writes the record");
    assert_eq!(record.claimed_generation, Some(1));
    assert_eq!(record.url, "mem://test");
    assert!(record.hydrated_from.is_none());

    shipper.cycle().await.unwrap();
    let names = object_names(&store).await;
    assert!(
        names.contains(&"gen-00000000000000000001/heartbeat".to_string()),
        "the first cycle beats: {names:?}"
    );
    assert!(
        !names.iter().any(|name| name.contains(REPLICATION_RECORD)),
        "the local record must never ship: {names:?}"
    );

    shipper.retire_generation().await;
    assert!(
        object_names(&store)
            .await
            .contains(&"gen-00000000000000000001/retired".to_string())
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn classification_and_lane_parents_agree_with_the_family_layout() {
    assert_eq!(classify(".taguru.lock"), EntryKind::Skip);
    assert_eq!(classify(REPLICATION_RECORD), EntryKind::Skip);
    assert_eq!(classify("x.ctx.tmp42"), EntryKind::Skip);
    // The shape `storage::staging_path` actually builds — a flush
    // racing a ship cycle leaves exactly this beside x.ctx.
    assert_eq!(classify("x.tmp72943-9"), EntryKind::Skip);
    assert_eq!(classify("x.meta.tmp72943-10"), EntryKind::Skip);
    // A name that only WEARS a tmp-ish suffix still ships.
    assert_eq!(classify("x.tmp"), EntryKind::Published);
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
            let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
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
