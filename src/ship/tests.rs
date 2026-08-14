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

/// A store that delegates every call to `inner` except three dials
/// this test module turns: `fail_puts` and `fail_deletes` each
/// count down by one on a matching call, answering a transient I/O
/// error while `> 0` and falling through to the real store once
/// they hit zero — "the bucket flaked N times, then came back",
/// the shape a retry-on-next-cycle fix needs proof against.
/// `fail_puts_permanently` answers `PermissionDenied` instead — a
/// deployment-shaped failure `store_error` must classify as
/// `ShipError::Permanent`, never `Io` — and never counts down (a
/// revoked credential does not spontaneously heal).
#[derive(Debug)]
struct FlakyStore {
    inner: Arc<dyn ObjectStore>,
    fail_puts: Arc<Mutex<usize>>,
    fail_deletes: Arc<Mutex<usize>>,
    fail_puts_permanently: Arc<Mutex<bool>>,
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
        if *self.fail_puts_permanently.lock() {
            return Err(object_store::Error::PermissionDenied {
                path: location.to_string(),
                source: "injected revoked credential".into(),
            });
        }
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

/// Delegates everything to `inner` except reads of one specific key,
/// which always answer a non-`NotFound` error — the shape
/// `newest_complete_generation`'s `head()` check on the newest
/// candidate generation must fail loudly on rather than silently
/// falling back to an older, possibly-complete generation (#616 item
/// 5). `head()`'s default implementation routes through `get_opts`
/// (see `object_store::ObjectStore::head`), so overriding `get_opts`
/// alone is enough to reach it.
#[derive(Debug)]
struct GetFailsOnStore {
    inner: Arc<dyn ObjectStore>,
    fails_on: StorePath,
}

impl std::fmt::Display for GetFailsOnStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GetFailsOnStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for GetFailsOnStore {
    async fn put_opts(
        &self,
        location: &StorePath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
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
        if *location == self.fails_on {
            return Err(injected("get"));
        }
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: futures_util::stream::BoxStream<'static, object_store::Result<StorePath>>,
    ) -> futures_util::stream::BoxStream<'static, object_store::Result<StorePath>> {
        self.inner.delete_stream(locations)
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

/// Delegates everything to `inner`, except `put_opts` on one specific
/// key: that FIRST call signals `started`, then blocks on `release`
/// before proceeding — a deterministic window for a test to inject a
/// competing write through the unwrapped store before this call's own
/// `put_opts` actually lands, without relying on non-deterministic
/// task-scheduling luck to force a real race.
#[derive(Debug)]
struct PausingStore {
    inner: Arc<dyn ObjectStore>,
    pause_on: StorePath,
    started: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
}

impl std::fmt::Display for PausingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PausingStore({})", self.inner)
    }
}

#[async_trait::async_trait]
impl ObjectStore for PausingStore {
    async fn put_opts(
        &self,
        location: &StorePath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<object_store::PutResult> {
        if *location == self.pause_on {
            self.started.notify_one();
            self.release.notified().await;
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
        self.inner.delete_stream(locations)
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
async fn a_lane_deleted_between_the_scan_and_its_own_turn_ships_nothing_not_an_error() {
    let dir = scratch_dir("vanished-mid-cycle");
    let state = state_for(&dir);
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
    let inner: Arc<dyn ObjectStore> = Arc::new(InMemory::new());

    std::fs::write(dir.join("ctx_a.ctx"), b"image-v1").unwrap();
    std::fs::write(dir.join("ctx_b.ctx"), b"image-v1").unwrap();
    let ctx_b_wal = dir.join("ctx_b.wal.jsonl");
    wal::append_batch(&ctx_b_wal, 1, &[associate("b")]).unwrap();

    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    // ctx_a.ctx's publish PUT: a `scan.changed` entry, which the
    // whole `scan.changed` loop finishes BEFORE the `scan.lanes` loop
    // (that reaches ctx_b's wal file) ever begins — a real `.await`
    // boundary a test can pause on, deterministically, unlike a race
    // WITHIN one `ship_lane` call (no `.await` between its own
    // `fs::metadata`/`fs::read` pair for a test to land in).
    let files_key = gen_root(&StorePath::default(), 1)
        .join("files")
        .join("ctx_a.ctx");
    let store: Arc<dyn ObjectStore> = Arc::new(PausingStore {
        inner: Arc::clone(&inner),
        pause_on: files_key,
        started: Arc::clone(&started),
        release: Arc::clone(&release),
    });

    let dir2 = dir.clone();
    let state2 = state.clone();
    let progress2 = Arc::clone(&progress);
    let cycle_task = tokio::spawn(async move {
        let mut shipper = claimed_dyn(store, &dir2, &state2, &progress2).await;
        shipper.cycle().await
    });

    started.notified().await;
    std::fs::remove_file(&ctx_b_wal).unwrap();
    release.notify_one();

    let result = tokio::time::timeout(Duration::from_secs(5), cycle_task)
        .await
        .expect("must not hang")
        .unwrap();
    assert!(
        result.is_ok(),
        "a lane deleted between the directory scan and the lanes loop reaching its own \
         turn must ship as if nothing changed there, not fail the whole cycle: {result:?}"
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
        fail_puts_permanently: Arc::new(Mutex::new(false)),
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
        fail_puts_permanently: Arc::new(Mutex::new(false)),
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
        fail_puts_permanently: Arc::new(Mutex::new(false)),
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

/// A deployment-shaped store failure (revoked credentials) must reach
/// the caller as `ShipError::Permanent`, not the generic `Io` every
/// other injected failure in this module produces — that distinction
/// is the whole fix (#616 item 1): a `handle::spawn` that only ever
/// sees `Io` cannot tell "the network blipped" from "this will never
/// succeed again without operator action".
#[tokio::test]
async fn a_permission_denied_upload_classifies_as_a_permanent_ship_error() {
    let dir = scratch_dir("permission-denied");
    let state = state_for(&dir);
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
    let inner = Arc::new(InMemory::new());
    let fail_puts_permanently = Arc::new(Mutex::new(false));
    let store: Arc<dyn ObjectStore> = Arc::new(FlakyStore {
        inner: Arc::clone(&inner) as Arc<dyn ObjectStore>,
        fail_puts: Arc::new(Mutex::new(0)),
        fail_deletes: Arc::new(Mutex::new(0)),
        fail_puts_permanently: Arc::clone(&fail_puts_permanently),
    });

    std::fs::write(dir.join("ctx_a.ctx"), b"image-v1").unwrap();
    let mut shipper = claimed_dyn(Arc::clone(&store), &dir, &state, &progress).await;
    *fail_puts_permanently.lock() = true;
    let error = shipper.cycle().await.unwrap_err();
    assert!(
        matches!(error, ShipError::Permanent(_)),
        "a PermissionDenied upload must classify as Permanent, got: {error}"
    );

    // Unlike a transient failure, this does not spontaneously heal —
    // the very next cycle hits the same wall, still classified the
    // same way (no accidental one-shot "permanent" that quietly
    // reverts to Io on retry).
    let error = shipper.cycle().await.unwrap_err();
    assert!(matches!(error, ShipError::Permanent(_)), "{error}");
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

/// The fail-loud counterpart to the test above (#616 item 5): a
/// genuine store error while checking the NEWEST candidate generation
/// must propagate immediately, not be swallowed the way `NotFound`
/// is — falling back to an older, possibly-complete generation on ANY
/// store error would risk restoring stale data while believing the
/// newest generation was simply never shipped, when the bucket may in
/// fact be unreachable or the object corrupt in a way `NotFound`
/// never means.
#[tokio::test]
async fn newest_complete_generation_fails_loudly_rather_than_falling_back_past_a_store_error() {
    let dir = scratch_dir("newest-store-error");
    let state = state_for(&dir);
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
    let inner = Arc::new(InMemory::new());

    // Generation 1: a real, complete baseline.
    std::fs::write(dir.join("ctx_a.ctx"), b"gen1-image").unwrap();
    let mut first = claimed(&inner, &dir, &state, &progress).await;
    first.cycle().await.unwrap();
    assert_eq!(first.generation, 1);

    // Generation 2: claimed but never shipped — ordinarily its
    // missing complete marker would make `newest_complete_generation`
    // skip it and fall back to generation 1 (the test above pins
    // exactly that). Here the store is wrapped so reading THAT marker
    // answers a real error instead of the natural `NotFound`.
    let second = claimed(&inner, &dir, &state, &progress).await;
    assert_eq!(second.generation, 2);
    let marker = StorePath::parse("gen-00000000000000000002/complete").unwrap();
    let wrapped: Arc<dyn ObjectStore> = Arc::new(GetFailsOnStore {
        inner: Arc::clone(&inner) as Arc<dyn ObjectStore>,
        fails_on: marker,
    });

    let error = newest_complete_generation(wrapped.as_ref(), &StorePath::default())
        .await
        .unwrap_err();
    assert_ne!(
        error.kind(),
        io::ErrorKind::NotFound,
        "a genuine store error must not be reported as simply not-shipped-yet: {error}"
    );
    assert!(
        error.to_string().contains("checking generation 2"),
        "{error}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// `run`'s exit-code contract (#616 item 3): a usage mistake in the
/// URL is exit 2, a store that refuses to open is exit 1 — end to end
/// through the real CLI entry point, not just `open_store` in
/// isolation.
#[test]
fn run_maps_a_usage_mistake_and_a_rejected_store_to_different_exit_codes() {
    let out = scratch_dir("run-exit-codes-out");

    let usage_code = run(&[
        "--out".to_string(),
        out.display().to_string(),
        "not a url at all".to_string(),
    ]);
    assert_eq!(usage_code, 2, "a malformed URL must be a usage error");

    let _env = crate::ship::test_support::ScrubbedAzureEnv::new();
    let rejected_code = run(&[
        "--out".to_string(),
        out.display().to_string(),
        "az://some-bucket".to_string(),
    ]);
    assert_eq!(
        rejected_code, 1,
        "a well-formed URL the store refuses to open must be bucket-unusable, not a usage error"
    );

    let _ = std::fs::remove_dir_all(&out);
}

/// #618: an unrecognized flag must refuse as a usage error naming the
/// flag, DURING ARGUMENT PARSING — before `--out`'s directory is ever
/// touched. A dash-prefixed string can never parse as a URL either, so
/// a mutant that disables this guard and lets the flag fall through to
/// the positional arm still ultimately fails with the same usage exit
/// code (2) once `open_store` rejects it as malformed — the exit code
/// alone cannot distinguish the two. Whether `--out`'s directory got
/// CREATED can: `open_store` runs well after `create_dir_all`, so only
/// the buggy, later failure leaves it behind.
#[test]
fn run_refuses_an_unrecognized_flag_before_touching_out() {
    let out = std::env::temp_dir().join(format!(
        "taguru-run-unknown-flag-out-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&out);
    let code = run(&[
        "--out".to_string(),
        out.display().to_string(),
        "--bogus".to_string(),
    ]);
    assert_eq!(code, 2, "an unrecognized flag must be a usage error");
    assert!(
        !out.exists(),
        "an unrecognized flag must be refused during argument parsing, before --out's \
         directory is created"
    );
    let _ = std::fs::remove_dir_all(&out);
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
}

/// #618: a `.taguru.replication` record that exists but cannot be
/// parsed must surface as an error, never silently treated as absent
/// — `prepare`/`prepare_replica` both refuse to boot on it rather than
/// risk forking the lineage. A directory in place of the file is an
/// easy, portable way to make the read fail with something other than
/// `NotFound`.
#[test]
fn a_corrupt_replication_record_is_an_error_not_a_missing_one() {
    let dir = scratch_dir("corrupt-replication-record");
    std::fs::create_dir(dir.join(REPLICATION_RECORD)).unwrap();
    let error = read_replication_record(&dir)
        .expect_err("a directory in place of the record must not read as 'never written'");
    assert_ne!(error.kind(), io::ErrorKind::NotFound, "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// #618: `Shipper::claim`'s retry loop must land on a generation past
/// one taken between its own `newest_fence` read and its own
/// `put_opts` — not loop forever on the same number. Forced
/// deterministically (no reliance on real task-scheduling luck): a
/// wrapper store pauses THIS claim's very first `put_opts` attempt
/// until the test has stolen that exact generation out from under it
/// via a raw write through the unwrapped store.
#[tokio::test]
async fn a_claim_retries_past_a_generation_taken_between_its_check_and_its_write() {
    let dir = scratch_dir("claim-retry-race");
    let state = state_for(&dir);
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
    let inner: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let target = fence_key(&StorePath::default(), 1);
    let store: Arc<dyn ObjectStore> = Arc::new(PausingStore {
        inner: Arc::clone(&inner),
        pause_on: target.clone(),
        started: Arc::clone(&started),
        release: Arc::clone(&release),
    });

    let dir2 = dir.clone();
    let state2 = state.clone();
    let progress2 = Arc::clone(&progress);
    let claim_task =
        tokio::spawn(async move { claimed_dyn(store, &dir2, &state2, &progress2).await });

    started.notified().await;
    inner
        .put_opts(
            &target,
            PutPayload::from(Vec::new()),
            PutOptions::from(PutMode::Create),
        )
        .await
        .expect("stealing generation 1 out from under the paused claim");
    release.notify_one();

    // Bounded, not `claim_task.await` bare: a retry that never
    // advances past the stolen generation loops forever bidding the
    // same taken number — this must fail fast, not hang the suite.
    let shipper = tokio::time::timeout(Duration::from_secs(5), claim_task)
        .await
        .expect("a claim retrying past a taken generation must not loop forever")
        .unwrap();
    assert_eq!(
        shipper.generation, 2,
        "a generation taken between the check and this claim's own write must be \
         retried past, not looped on forever"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// #618: the replication record's `hydrated_from` carries forward only
/// when the EXISTING record names the SAME bucket url this claim is
/// against — a directory whose record predates a re-pointed bucket
/// must not inherit a generation number that means nothing there.
#[tokio::test]
async fn a_claim_carries_hydrated_from_forward_only_for_the_same_bucket_url() {
    let dir = scratch_dir("claim-hydrated-from");
    write_replication_record(
        &dir,
        &ReplicationRecord {
            url: "mem://test".to_string(),
            claimed_generation: None,
            hydrated_from: Some(5),
        },
    )
    .unwrap();
    let state = state_for(&dir);
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
    let store = Arc::new(InMemory::new());

    // `claimed()` always claims against "mem://test" — the same url
    // the pre-seeded record above names.
    claimed(&store, &dir, &state, &progress).await;
    let record = read_replication_record(&dir).unwrap().unwrap();
    assert_eq!(
        record.hydrated_from,
        Some(5),
        "a claim against the SAME bucket must carry the prior hydrated_from forward"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// #618: under a lazy (not-yet-drained) hydration, a cycle whose ONLY
/// activity is a lane shipping something for the first time must still
/// report `true` — the manifest-publish step that (when hydration IS
/// drained) would independently re-assert `shipped = true` is exactly
/// the step this scenario skips, so the lane loop's own return value
/// is what the caller actually sees.
#[tokio::test]
async fn a_cycle_with_undrained_hydration_still_reports_a_shipped_lane() {
    // A separate, already-shipped bucket to hydrate FROM: a real
    // claimed generation with one context family, so `prepare_replica`
    // returns a hydrator whose stems start `Pending` (undrained) and
    // stay that way — nothing here ever calls `ensure_context`.
    let source_dir = scratch_dir("undrained-source");
    let source_state = state_for(&source_dir);
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
    let store = Arc::new(InMemory::new());
    std::fs::write(source_dir.join("ctx_a.ctx"), b"image-v1").unwrap();
    let mut source_shipper = claimed(&store, &source_dir, &source_state, &progress).await;
    source_shipper.cycle().await.unwrap();

    let target_dir = scratch_dir("undrained-target");
    let hydrator = crate::hydrate::prepare_replica(
        &(store.clone() as Arc<dyn ObjectStore>),
        &StorePath::default(),
        "mem://test",
        &target_dir,
    )
    .await
    .expect("hydrates");
    assert!(
        !hydrator.drained(),
        "a freshly prepared hydrator with a real family must start undrained"
    );

    let target_state = state_for(&target_dir);
    let mut shipper = Shipper::claim(
        store.clone() as Arc<dyn ObjectStore>,
        StorePath::default(),
        "mem://test".to_string(),
        target_dir.clone(),
        Arc::clone(&progress),
        target_state,
        Some(hydrator),
    )
    .await
    .unwrap();

    // A genuinely new local write, unrelated to the hydration above —
    // the server keeps accepting writes while background-hydrating.
    wal::append_batch(&target_dir.join("ctx_a.wal.jsonl"), 1, &[associate("a")]).unwrap();

    assert!(
        shipper.cycle().await.unwrap(),
        "a lane that genuinely shipped something must report true even when hydration \
         has not drained and the manifest-publish step is skipped entirely"
    );
    let _ = std::fs::remove_dir_all(&source_dir);
    let _ = std::fs::remove_dir_all(&target_dir);
}

/// #618: `fence_holder` was never directly exercised — only ever
/// called from `main`'s replica-status wiring.
#[tokio::test]
async fn fence_holder_reads_back_the_claiming_holder() {
    let dir = scratch_dir("fence-holder");
    let state = state_for(&dir);
    let progress = Arc::new(ShipProgress::new(crate::registry::DEFAULT_WAL_MAX_BYTES));
    let store = Arc::new(InMemory::new());

    let shipper = claimed(&store, &dir, &state, &progress).await;
    let holder = fence_holder(
        store.as_ref() as &dyn ObjectStore,
        &StorePath::default(),
        shipper.generation,
    )
    .await
    .expect("the just-claimed generation's fence body reads back");
    assert!(holder.contains('#'), "the holder is HOSTNAME#pid: {holder}");

    // A generation nothing ever claimed: best-effort `None`, not an
    // error or a panic.
    assert!(
        fence_holder(
            store.as_ref() as &dyn ObjectStore,
            &StorePath::default(),
            9999
        )
        .await
        .is_none()
    );
    let _ = std::fs::remove_dir_all(&dir);
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
