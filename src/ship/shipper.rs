//! [`Shipper`]: the per-generation state machine that scans the data
//! directory each cycle and ships whatever changed, plus the fence
//! primitives ([`newest_fence`], [`fence_holder`]) it claims and
//! watches its generation against.

use super::*;

pub(crate) struct Shipper {
    store: Arc<dyn ObjectStore>,
    root: StorePath,
    pub(super) generation: u64,
    data_dir: PathBuf,
    files: BTreeMap<String, ShippedFile>,
    lanes: BTreeMap<String, LaneState>,
    progress: Arc<ShipProgress>,
    state: AppState,
    /// Present when this boot hydrated lazily from the bucket: the
    /// manifest (`complete`) must not be written while contexts are
    /// still only in the PREDECESSOR generation — a `complete` written
    /// early would crown a generation missing every un-hydrated
    /// family, and a restore would pick it. Until the hydrator drains,
    /// the newest complete generation stays the one this boot restored
    /// from, which really does hold everything.
    hydration: Option<Arc<crate::hydrate::Hydrator>>,
    baseline_complete: bool,
    last_heartbeat: Option<Instant>,
    /// Set the instant `self.files`/`self.lanes` changes underneath
    /// the manifest (`complete`) — a file/lane ships, or a name
    /// retires — and cleared only once `put_manifest` actually lands.
    /// `shipped` alone (whether THIS cycle uploaded anything) is not
    /// enough: a manifest PUT can fail after everything else in the
    /// cycle succeeded, and with no other signal that it is stale, a
    /// server that goes idle right after would never retry it — the
    /// bucket's manifest would describe a state that no longer exists
    /// (an object it still lists as retired, or a segment it never
    /// learned about) until an unrelated write came along to re-trip
    /// `shipped`.
    manifest_dirty: bool,
}

impl Shipper {
    /// Claims the next generation and returns a shipper whose first
    /// cycle will run the baseline sync. Everything before the claim
    /// succeeds is refusal-shaped: a bucket that cannot even take the
    /// fence write is a bucket that cannot take data.
    pub(crate) async fn claim(
        store: Arc<dyn ObjectStore>,
        root: StorePath,
        url: String,
        data_dir: PathBuf,
        progress: Arc<ShipProgress>,
        state: AppState,
        hydration: Option<Arc<crate::hydrate::Hydrator>>,
    ) -> Result<Self, ShipError> {
        let mut generation = newest_fence(&store, &root)
            .await?
            .map_or(0, |f| f.generation)
            + 1;
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
        // Remember the claim locally: the next boot of THIS directory
        // proves "the bucket's newest writer was me" by comparing this
        // number against the newest fence — the warm-restart shortcut,
        // and the reason a plain restart never trips the takeover
        // guard. Best-effort: without it the next boot just re-verifies
        // (cache mode) or asks for explicit intent (a recent foreign
        // generation).
        let record = ReplicationRecord {
            url: url.clone(),
            claimed_generation: Some(generation),
            // `.ok()`: boot already refused a corrupt record (see
            // `hydrate::prepare`), so an error here means it rotted
            // mid-run — carrying no `hydrated_from` forward is the
            // same best-effort degradation as the write failing below.
            hydrated_from: read_replication_record(&data_dir)
                .ok()
                .flatten()
                .filter(|record| record.url == url)
                .and_then(|record| record.hydrated_from),
        };
        if let Err(error) = write_replication_record(&data_dir, &record) {
            tracing::warn!(%error, "could not persist {REPLICATION_RECORD}");
        }
        Ok(Self {
            store,
            root,
            generation,
            data_dir,
            files: BTreeMap::new(),
            lanes: BTreeMap::new(),
            progress,
            state,
            hydration,
            baseline_complete: false,
            last_heartbeat: None,
            manifest_dirty: false,
        })
    }

    /// Whether a newer claimant has taken the bucket. One GET per
    /// dirty cycle: a successor claims exactly `our generation + 1`
    /// upward, and every claim is caused by a real claimant, so the
    /// first successor's fence is the one to watch for.
    async fn fenced_by(&self) -> Result<Option<u64>, ShipError> {
        match newest_fence(&self.store, &self.root).await? {
            Some(newest) if newest.generation > self.generation => Ok(Some(newest.generation)),
            _ => Ok(None),
        }
    }

    /// One poll cycle: scan, and if anything changed (or a heartbeat
    /// is due), re-check the fence and ship the difference. Returns
    /// whether anything shipped. `Fenced` is terminal — the caller
    /// stops the loop.
    pub(crate) async fn cycle(&mut self) -> Result<bool, ShipError> {
        let scan = self.scan()?;
        // Under a lazy hydration the baseline is not writable yet (see
        // the `hydration` field): until the hydrator drains, only real
        // local changes make a cycle dirty.
        let hydration_drained = self.hydration.as_ref().is_none_or(|h| h.drained());
        let dirty = (!self.baseline_complete && hydration_drained)
            || !scan.changed.is_empty()
            || !scan.vanished.is_empty()
            || !scan.lanes.is_empty()
            // A manifest PUT that failed on an earlier cycle needs its
            // own retry even when THIS scan finds nothing changed —
            // otherwise an idle server (no lanes, no vanished/changed
            // names) never re-enters the block below that retries it.
            || self.manifest_dirty;
        let heartbeat_due = self
            .last_heartbeat
            .is_none_or(|at| at.elapsed() >= HEARTBEAT_INTERVAL);
        if !dirty && !heartbeat_due {
            return Ok(false);
        }
        // The fence check gates UPLOADS, not local reads: an idle
        // deposed shipper discovers its deposition on its next real
        // work — a due heartbeat included, since beating past one's
        // successor would advertise a liveness the takeover already
        // ended — and never sooner, because a fenced check with
        // nothing to ship would fail-stop a server whose bucket
        // successor changes nothing about its local correctness.
        if let Some(newer_generation) = self.fenced_by().await? {
            return Err(ShipError::Fenced { newer_generation });
        }
        let mut shipped = false;
        if dirty {
            for name in &scan.vanished {
                self.retire_file(name).await?;
                shipped = true;
            }
            for name in &scan.changed {
                // The lane loop below owns log files; `scan` never lists
                // them here.
                let file = self.ship_published(name).await?;
                self.files.insert(name.clone(), file);
                self.manifest_dirty = true;
                shipped = true;
            }
            for name in &scan.lanes {
                shipped |= self.ship_lane(name).await?;
            }
            // The manifest (`complete`) follows every batch of uploads
            // so it always describes the bucket's current state — but
            // never before a lazy hydration has drained (the field doc
            // explains why an early `complete` would be a lie).
            // `manifest_dirty` (not just `shipped`) gates this: a PUT
            // that failed on an earlier cycle must retry here even if
            // THIS cycle uploaded nothing new — see the field's doc.
            if (!self.baseline_complete || shipped || self.manifest_dirty) && hydration_drained {
                self.put_manifest().await?;
                self.manifest_dirty = false;
                shipped = true;
                if !self.baseline_complete {
                    self.baseline_complete = true;
                    tracing::info!(
                        target: "taguru::audit",
                        generation = self.generation,
                        "replication baseline complete — the bucket can restore this directory",
                    );
                }
            }
        }
        if heartbeat_due {
            self.put(
                &gen_root(&self.root, self.generation).join(HEARTBEAT_MARKER),
                Vec::new(),
            )
            .await?;
            self.last_heartbeat = Some(Instant::now());
        }
        if shipped {
            self.state.metrics().record_replication_success();
        }
        Ok(shipped)
    }

    /// Uploads the manifest under the `complete` key: the generation's
    /// exact shipped state, and (on its first write) the marker that
    /// makes the generation restorable at all.
    async fn put_manifest(&self) -> Result<(), ShipError> {
        let manifest = Manifest {
            generation: self.generation,
            files: self
                .files
                .iter()
                .map(|(name, file)| (name.clone(), file.content))
                .collect(),
            lanes: self
                .lanes
                .iter()
                .map(|(name, lane)| {
                    (
                        name.clone(),
                        ManifestLane {
                            series: lane.series,
                            segments: lane.next_seg,
                            len: lane.shipped_offset,
                            crc: lane.shipped_crc,
                            seq: lane.shipped_seq,
                        },
                    )
                })
                .collect(),
        };
        self.put(
            &gen_root(&self.root, self.generation).join(COMPLETE_MARKER),
            serde_json::to_vec(&manifest).expect("no unserializable field"),
        )
        .await
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
                    if self.files.get(&name).map(|file| file.sig) != Some(sig) {
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
    async fn ship_published(&mut self, name: &str) -> Result<ShippedFile, ShipError> {
        let path = self.data_dir.join(name);
        let metadata = std::fs::metadata(&path).map_err(ShipError::Io)?;
        let sig = FileSig::of(&metadata);
        let bytes = std::fs::read(&path).map_err(ShipError::Io)?;
        let content = ManifestFile {
            len: bytes.len() as u64,
            crc: crate::crc32c::crc32c(&bytes),
        };
        let key = gen_root(&self.root, self.generation)
            .join("files")
            .join(name);
        self.put(&key, bytes).await?;
        Ok(ShippedFile { sig, content })
    }

    /// Removes a vanished file's remote counterpart — and, for a log
    /// lane, its whole segment prefix, so a re-created context of the
    /// same name can never interleave with the old incarnation's
    /// records on restore.
    async fn retire_file(&mut self, name: &str) -> Result<(), ShipError> {
        let generation_root = gen_root(&self.root, self.generation);
        // Bookkeeping is dropped only AFTER the remote delete succeeds:
        // `scan.vanished` (the next cycle's retry list) is computed
        // from `self.lanes`/`self.files`, so removing the entry first
        // and then failing the delete would make the object an orphan
        // — gone from every future scan, never retried, and left in
        // the bucket forever.
        if self.lanes.contains_key(name) {
            let prefix = generation_root.join("wal").join(name);
            if let Err(error) = delete_prefix(self.store.as_ref(), &prefix).await {
                self.state.metrics().record_replication_error();
                return Err(error);
            }
            self.lanes.remove(name);
            self.manifest_dirty = true;
            self.progress.forget(&self.data_dir.join(name));
            let (context, lane_kind) = lane_metric_labels(name);
            self.state
                .metrics()
                .forget_replication_lane(&context, lane_kind);
        } else {
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
            self.files.remove(name);
            self.manifest_dirty = true;
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
        let stat = match std::fs::metadata(&path) {
            Ok(metadata) => metadata,
            // Vanished mid-cycle: the next scan retires it.
            Err(error) if vanished_mid_cycle(&error) => return Ok(false),
            Err(error) => return Err(ShipError::Io(error)),
        };
        let sig = FileSig::of(&stat);
        let mut lane = self
            .lanes
            .get(name)
            .cloned()
            .unwrap_or_else(|| LaneState::fresh(0));

        // An unchanged signature means the last full read already saw
        // everything this file currently has — no bytes to append, no
        // divergence — so re-reading now and re-hashing the whole
        // shipped prefix could only reproduce that same answer. Only
        // the lag metric below still needs a beat every cycle
        // (`age_secs` is elapsed time, not file content), and the
        // cached `local_seq` answers that without touching the file.
        //
        // Known limitation: unlike a published file (a new version is
        // always a new inode via rename), a lane mutates IN PLACE —
        // same inode, same length possible after a rollback that
        // rewrites an identical byte count within one mtime granule.
        // On a filesystem with second-level mtime resolution (some
        // NFS/HFS+/ext3 configurations), that specific rewrite could
        // share this file's `FileSig` with the pre-rollback one and be
        // skipped for one cycle. Accepted: every target this project
        // ships to (APFS, ext4) carries sub-second mtimes, and the very
        // next append (any length change) forces a fresh signature
        // regardless.
        let shipped = if lane.last_seen_sig == Some(sig) {
            false
        } else {
            let bytes = match std::fs::read(&path) {
                Ok(bytes) => bytes,
                // Vanished between the stat above and this read: the
                // next scan retires it.
                Err(error) if vanished_mid_cycle(&error) => return Ok(false),
                Err(error) => return Err(ShipError::Io(error)),
            };

            // Is the shipped prefix still literally the file's prefix?
            let prefix_intact = bytes.len() as u64 >= lane.shipped_offset
                && crate::crc32c::crc32c(&bytes[..lane.shipped_offset as usize])
                    == lane.shipped_crc;

            if !prefix_intact {
                // Divergence: reset, rollback, or delete + re-create. The
                // parent snapshot covers everything the divergence
                // discarded, and it must be IN THE BUCKET before the new
                // series exists there, or a restore in the window between
                // the two could see a series whose first record is beyond
                // the restored snapshot's watermark.
                let parent =
                    parent_snapshot_of(name).expect("lane names always carry a wal suffix");
                let parent_path = self.data_dir.join(&parent);
                match std::fs::metadata(&parent_path) {
                    Ok(metadata) => {
                        // Skip the upload when this cycle (or an earlier
                        // one) already shipped exactly this version — the
                        // common case, since the reset that diverged the
                        // lane follows the very flush that published it.
                        if self.files.get(&parent).map(|file| file.sig)
                            != Some(FileSig::of(&metadata))
                        {
                            let file = self.ship_published(&parent).await?;
                            self.files.insert(parent, file);
                            self.manifest_dirty = true;
                        }
                    }
                    Err(error) if vanished_mid_cycle(&error) => {
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
                // The series bump alone makes the manifest stale even
                // if this cycle ends up shipping no new segment below
                // (an empty or torn-only file right after the reset).
                self.manifest_dirty = true;
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
                        self.manifest_dirty = true;
                        true
                    }
                }
            };

            lane.local_seq = newest_seq(&bytes).unwrap_or(lane.shipped_seq);
            lane.last_seen_sig = Some(sig);
            shipped
        };

        // Lag bookkeeping, shipped or not: how far the local log's
        // newest record is beyond the shipped one, and for how long.
        update_pending_since(&mut lane);
        let age_secs = lane
            .pending_since
            .map(|since| since.elapsed().as_secs())
            .unwrap_or(0);
        let (context, lane_kind) = lane_metric_labels(name);
        self.state.metrics().note_replication_lane(
            &context,
            lane_kind,
            lane.local_seq.saturating_sub(lane.shipped_seq),
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

    /// Marks this generation cleanly stopped (`gen-{N}/retired`): the
    /// takeover guard reads it as "no live writer remains", so the
    /// next boot against this bucket proceeds without explicit intent.
    /// Best-effort — a miss (crash, unreachable bucket) only means the
    /// next claimant waits out [`super::naming::TAKEOVER_GRACE`] or
    /// passes `--take-over`, exactly the posture a crash should leave.
    pub(crate) async fn retire_generation(&self) {
        let key = gen_root(&self.root, self.generation).join(RETIRED_MARKER);
        if let Err(error) = self.put(&key, Vec::new()).await {
            tracing::warn!(%error, "could not mark the replication generation retired");
        }
    }
}

/// Whether a local fs error on a lane file mid-cycle means "this name
/// vanished between the directory scan that found it and this call"
/// (ship as if nothing changed; the next scan retires it) vs. any
/// other local error (propagate). Pulled into its own function so its
/// mutation targets at `ship_lane`'s three call sites can each be
/// judged on their own reachability — a real filesystem race between
/// the scan and one of these calls cannot be pinned deterministically
/// in a test, but a genuine `NotFound` reaching this comparison IS
/// (see `a_lane_deleted_between_the_scan_and_its_own_turn_ships_nothing_not_an_error`,
/// which races an earlier `scan.changed` upload to delete a lane file
/// before the lanes loop reaches its turn) — see `.cargo/mutants.toml`
/// for which of the three call sites remain excluded, and why.
fn vanished_mid_cycle(error: &io::Error) -> bool {
    error.kind() == io::ErrorKind::NotFound
}

/// The newest (highest) seq among the file's complete lines, ignoring
/// integrity: this feeds the LAG metric only, where an honest "how far
/// behind" matters more than validity — corrupt bytes will surface as
/// a shipping error, not a hidden zero lag. `pub(super)` (not private)
/// so `ship::tests` can pin it directly with byte inputs instead of
/// only through a full `ship_lane` cycle.
pub(super) fn newest_seq(bytes: &[u8]) -> Option<u64> {
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

/// Whether a lane's local log has grown past what shipped, and for how
/// long. `pub(super)` (not private) so `ship::tests` can pin the
/// `local_seq == shipped_seq` boundary directly on a `LaneState`,
/// rather than only through a full `ship_lane` cycle where the two
/// are, by construction, never observed to diverge (see `newest_seq`'s
/// call site in `ship_lane`: both values come from the SAME read of
/// `bytes`, and every error path out of that read discards this
/// call's `lane` mutations before they would ever be compared) — this
/// function's own boundary still needs pinning on its own terms, since
/// `>` vs `>=` disagree exactly there.
pub(super) fn update_pending_since(lane: &mut LaneState) {
    if lane.local_seq > lane.shipped_seq {
        lane.pending_since.get_or_insert_with(Instant::now);
    } else {
        lane.pending_since = None;
    }
}

#[derive(Default)]
struct Scan {
    changed: Vec<String>,
    vanished: Vec<String>,
    lanes: Vec<String>,
}

/// The newest fence: its generation and when it was claimed, by the
/// store's own clock (`last_modified`) — the one clock every reader
/// shares, which is why the takeover guard reads it rather than the
/// fence body's writer-stamped time.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FenceInfo {
    pub(crate) generation: u64,
    pub(crate) claimed: Option<SystemTime>,
}

/// The highest generation with a fence object, scanned by listing the
/// fence prefix — names are fixed-width decimals, so the maximum is
/// the lexicographic maximum.
pub(crate) async fn newest_fence(
    store: &Arc<dyn ObjectStore>,
    root: &StorePath,
) -> Result<Option<FenceInfo>, ShipError> {
    let prefix = root.clone().join(FENCE_PREFIX);
    let mut newest: Option<FenceInfo> = None;
    let mut listing = store.list(Some(&prefix));
    while let Some(meta) = listing.next().await {
        let meta = meta.map_err(|error| store_error("listing the replication fence", error))?;
        if let Some(name) = meta.location.filename()
            && let Ok(generation) = name.parse::<u64>()
            // `<` vs `<=` is unobservable here (issue #618): every
            // fence object's name is that generation's decimal key,
            // written at most once each (`claim`'s `PutMode::Create`
            // refuses a second write to the same name), so one
            // listing pass can never present the same `generation`
            // twice — `fence.generation == generation` never holds
            // mid-loop, and a mutant swapping `<=` in computes the
            // identical `newest` for every possible listing.
            && newest.is_none_or(|fence| fence.generation < generation)
        {
            newest = Some(FenceInfo {
                generation,
                claimed: Some(SystemTime::from(meta.last_modified)),
            });
        }
    }
    Ok(newest)
}

/// The holder a fence body names (`HOSTNAME#pid`, stamped at claim
/// time) — for the replica's write-refusal message and its status
/// surface. Best-effort: a missing or unreadable body leaves the
/// writer unnamed, never fails a poll.
pub(crate) async fn fence_holder(
    store: &dyn ObjectStore,
    root: &StorePath,
    generation: u64,
) -> Option<String> {
    let bytes = fetch(store, &fence_key(root, generation)).await.ok()?;
    serde_json::from_slice::<FenceBody>(&bytes)
        .ok()
        .map(|body| body.holder)
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
