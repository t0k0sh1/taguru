//! Restore: materialize a data directory from the bucket, both as the
//! `taguru restore` CLI entry point and as the shared download/verify
//! primitives `crate::hydrate` reuses for its own boot-time reads.

use super::*;

/// What one restore did, for the CLI's report and the tests'
/// assertions.
#[derive(Debug, Default)]
pub(crate) struct RestoreReport {
    pub(crate) generation: u64,
    pub(crate) files: usize,
    pub(crate) lanes: usize,
    pub(crate) records: usize,
}

const RESTORE_USAGE: &str = "\
usage: taguru restore --out DIR [--config FILE] [URL]

Materialize a data directory from a replication bucket (the newest
complete generation): every published file, plus each context's two
log lanes reassembled from their shipped segments. The URL defaults
to TAGURU_REPLICATE_URL; credentials ride each cloud's default chain.
DIR must not already contain a data directory — restore refuses to
mix two histories. Verify the result with: taguru inspect DIR
";

/// `taguru restore --out DIR [URL]`: materializes a data directory
/// from the bucket's newest COMPLETE generation. Exit codes follow the
/// house rule: 0 restored · 1 bucket unusable (no fence, no complete
/// generation, corrupt segments, or the store itself refused to open
/// — bad/missing credentials, a rejected cloud config, an
/// inaccessible local path) · 2 usage error (a malformed URL, an
/// unrecognized scheme, or a bad flag).
pub(crate) fn run(args: &[String]) -> i32 {
    let usage = |message: &str| crate::config::subcommand_usage_error("restore", message);
    let mut out: Option<PathBuf> = None;
    let mut url: Option<String> = None;
    let mut config: Option<PathBuf> = None;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{RESTORE_USAGE}");
                return 0;
            }
            "--out" => match rest.next() {
                Some(path) if out.is_none() => out = Some(PathBuf::from(path)),
                Some(_) => return usage("--out given twice"),
                None => return usage("--out needs a directory path"),
            },
            "--config" => match rest.next() {
                Some(path) if config.is_none() => config = Some(PathBuf::from(path)),
                Some(_) => return usage("--config given twice"),
                None => return usage("--config needs a file path"),
            },
            flag if flag.starts_with('-') => {
                return usage(&format!("'restore' does not take '{flag}'"));
            }
            positional => {
                if url.replace(positional.to_string()).is_some() {
                    return usage(&format!(
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
        return usage("restore needs --out DIR (the directory to materialize)");
    };
    let Some(url) = url.or_else(|| std::env::var("TAGURU_REPLICATE_URL").ok()) else {
        return usage("restore needs a bucket URL — pass one, or set TAGURU_REPLICATE_URL");
    };

    // The directory must exist before the lock below can create its
    // file inside it — idempotent (a no-op if `out` already exists),
    // so no separate exists/missing branch is needed.
    if let Err(error) = std::fs::create_dir_all(&out) {
        eprintln!("taguru: restore: cannot create {}: {error}", out.display());
        return 1;
    }
    // Every file this restore writes below fsyncs its own directory
    // (`write_atomic`'s rename), but that directory's OWN entry in ITS
    // parent was never synced — without this, "restored generation N"
    // printed on success could still have power loss drop `out`'s
    // directory entry wholesale, even though every file inside it
    // landed durably. Harmless (and cheap) to redo when `out` already
    // existed.
    if let Err(error) = crate::storage::fsync_parent_dir(&out) {
        eprintln!(
            "taguru: restore: cannot durably create {}: {error}",
            out.display()
        );
        return 1;
    }

    // Hold the same advisory lock every writer takes, for the whole
    // materialization — acquired BEFORE the occupied-directory check
    // below, not after: `lock_data_dir` fails immediately (never
    // blocks) if another `taguru` process already holds it, so
    // acquiring it first turns a race between two concurrent restores
    // (or a restore and a live `serve`/`import`) against the same
    // directory into an immediate, clear refusal for the loser instead
    // of both passing the occupied check and one of them later wiping
    // out the other's in-progress files via the failure cleanup below.
    let _dir_lock = match crate::storage::lock_data_dir(&out) {
        Ok(lock) => lock,
        Err(error) => {
            eprintln!("taguru: restore: {error}");
            return 1;
        }
    };

    // Refuse a target that already holds data: a restore layered onto
    // an existing directory would interleave two histories — exactly
    // the corruption the fence exists to prevent bucket-side. A lone
    // `.taguru.lock` does not count as data — it is the empty leftover
    // of the lock just taken (an earlier restore that died before
    // writing anything), and the lock file itself never ships. Reading
    // this AFTER the lock, not before, is what makes it safe: nothing
    // else can be concurrently writing into `out` while this decides
    // whether it is empty.
    let occupied = match std::fs::read_dir(&out) {
        Ok(entries) => entries
            .filter_map(|entry| entry.ok())
            .any(|entry| entry.file_name() != ".taguru.lock"),
        Err(error) => {
            eprintln!("taguru: restore: cannot read {}: {error}", out.display());
            return 1;
        }
    };
    if occupied {
        eprintln!(
            "taguru: restore: {} is not empty — restore refuses to mix histories; \
             point --out at a new or empty directory",
            out.display()
        );
        return 1;
    }

    let (store, root) = match open_store(&url) {
        Ok(opened) => opened,
        Err(error) => {
            eprintln!("taguru: restore: {error}");
            // `open_store`'s two failure shapes carry different exit
            // codes: a malformed URL/scheme/local-path is a usage
            // mistake (`InvalidInput`/`NotFound`, exit 2 — fixing the
            // argument fixes it); a store that parsed fine but could
            // not be OPENED (bad/missing credentials, a cloud builder
            // rejecting the config, a local path that exists but is
            // not actually usable) is the store itself being
            // unusable, same bucket as "no fence, no complete
            // generation" (exit 1) — no amount of retrying THIS
            // invocation's arguments would fix it.
            return if error.kind() == io::ErrorKind::Other {
                1
            } else {
                2
            };
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
            // Clean up so a retry does not need the operator to
            // manually empty the directory: the occupied check above
            // already refused to start unless `out` held nothing but
            // our own `.taguru.lock`, so every OTHER entry here was
            // created by THIS attempt — never a second history to
            // preserve.
            if let Err(cleanup_error) = clean_partial_restore(&out) {
                eprintln!(
                    "taguru: restore: additionally failed to clean up the partial \
                     result ({cleanup_error}) — remove {} by hand before retrying",
                    out.display()
                );
            }
            1
        }
    }
}

/// Removes everything a just-failed restore left under `out`, so the
/// next `taguru restore --out DIR` against the same directory sees it
/// empty again instead of tripping the occupied-directory refusal on
/// its own half-written state. Only ever called after `run`'s own
/// occupied check has already confirmed `out` held nothing but
/// `.taguru.lock` when this attempt started — that is what makes
/// "remove everything except the lock" safe unconditionally here.
fn clean_partial_restore(out: &FsPath) -> io::Result<()> {
    for entry in std::fs::read_dir(out)? {
        let entry = entry?;
        if entry.file_name() == ".taguru.lock" {
            continue;
        }
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            std::fs::remove_dir_all(&path)?;
        } else {
            std::fs::remove_file(&path)?;
        }
    }
    Ok(())
}

/// The restore body: pick the newest complete generation and
/// materialize it exactly as its manifest (issue #128 onward)
/// describes — the object set is exactly what the writer said it
/// shipped, and every downloaded byte is checked against the writer's
/// own CRC before it lands, so a swapped or rotted object is a
/// refusal, not a quiet divergence. A generation whose `complete`
/// marker predates the manifest (empty body) refuses outright: every
/// generation any current writer ships carries a manifest, so this
/// only ever names a bucket from before #128, which restore no longer
/// supports materializing.
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

    let Some(manifest) = read_manifest(store, &generation_root).await? else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "generation {generation}: its complete marker predates the shipping \
                 manifest (issue #128) — restore cannot materialize a bucket this old"
            ),
        ));
    };
    for (name, expect) in &manifest.files {
        let key = generation_root.clone().join("files").join(name.as_str());
        let bytes = fetch(store, &key).await?;
        verify_file_bytes(name, &bytes, *expect)?;
        write_restored_file(out, name, &bytes)?;
        report.files += 1;
    }
    for (name, lane) in &manifest.lanes {
        let assembled = fetch_lane(store, &generation_root, name, *lane).await?;
        let records = crate::wal::shippable_records(&assembled).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("lane {name} series {}: {error}", lane.series),
            )
        })?;
        report.records += records.len();
        crate::storage::write_atomic(&out.join(name), &assembled)?;
        report.lanes += 1;
    }
    Ok(report)
}

/// A manifest-supplied name must land as exactly one ordinary path
/// component under a local directory — never an absolute path, `..`,
/// or a name carrying its own separator. `verify_file_bytes` catches
/// rotted or swapped CONTENT, but an attacker with bucket-write access
/// computes the CRC over their own bytes, so a hostile NAME needs its
/// own check: without one, `out.join(name)` (restore) or
/// `data_dir.join(name)` (hydrate) would happily write outside the
/// target directory for a name like `"/home/user/.ssh/authorized_keys"`
/// or `"../../etc/x"`.
fn safe_manifest_name(name: &str) -> bool {
    let mut components = FsPath::new(name).components();
    matches!(components.next(), Some(std::path::Component::Normal(_)))
        && components.next().is_none()
}

/// Reads the generation's manifest out of its `complete` marker.
/// `None` for the pre-manifest (empty) marker; an error for a
/// non-empty body that does not parse, OR that carries a file/lane
/// name [`safe_manifest_name`] refuses — that is rot (or tampering),
/// not age. Checked once, here, so every caller (restore, hydrate,
/// replica tailing) inherits the guarantee without its own check.
pub(crate) async fn read_manifest(
    store: &dyn ObjectStore,
    generation_root: &StorePath,
) -> io::Result<Option<Manifest>> {
    let bytes = fetch(store, &generation_root.clone().join(COMPLETE_MARKER)).await?;
    if bytes.is_empty() {
        return Ok(None);
    }
    let manifest: Manifest = serde_json::from_slice(&bytes).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("the generation's manifest (complete) does not parse: {error}"),
        )
    })?;
    for name in manifest.files.keys().chain(manifest.lanes.keys()) {
        if !safe_manifest_name(name) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{name}: not a safe file name — the manifest (complete) may be tampered with"
                ),
            ));
        }
    }
    Ok(Some(manifest))
}

/// Refuses downloaded bytes that do not match what the manifest says
/// was uploaded.
pub(crate) fn verify_file_bytes(name: &str, bytes: &[u8], expect: ManifestFile) -> io::Result<()> {
    if bytes.len() as u64 != expect.len || crate::crc32c::crc32c(bytes) != expect.crc {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{name}: downloaded bytes do not match the manifest — the bucket rotted, \
                 or an object was replaced under the reader"
            ),
        ));
    }
    Ok(())
}

/// Lands one restored file locally, atomically, preserving the grant
/// store's owner-only mode — the one secret-bearing family member.
pub(crate) fn write_restored_file(out: &FsPath, name: &str, bytes: &[u8]) -> io::Result<()> {
    let path = out.join(name);
    if name == "oauth.json" {
        crate::storage::write_atomic_private(&path, bytes)
    } else {
        crate::storage::write_atomic(&path, bytes)
    }
}

/// Downloads and reassembles one lane exactly as the manifest
/// describes it — the segment names are derivable (`series` +
/// `0..segments`), so no listing is involved — and verifies the
/// concatenation's extent against the manifest's prefix arithmetic.
pub(crate) async fn fetch_lane(
    store: &dyn ObjectStore,
    generation_root: &StorePath,
    name: &str,
    lane: ManifestLane,
) -> io::Result<Vec<u8>> {
    let lane_prefix = generation_root.clone().join("wal").join(name);
    // `lane.len` comes from the bucket-held manifest, unverified until
    // the CRC check below — a tampered manifest declaring a huge `len`
    // must not be able to abort the restore via allocation failure
    // before a single byte is fetched.
    let mut assembled = Vec::with_capacity((lane.len as usize).min(8 * 1024 * 1024));
    for seg in 0..lane.segments {
        let key = lane_prefix.clone().join(segment_name(lane.series, seg));
        // Segments hold acknowledged writes; one the bucket cannot
        // hand back — lost, dropped, or deleted — is a refusal with
        // its position named, never a quiet skip.
        let bytes = fetch(store, &key).await.map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "lane {name} series {}: segment {seg} of {}: {error}",
                    lane.series, lane.segments
                ),
            )
        })?;
        assembled.extend_from_slice(&bytes);
    }
    if assembled.len() as u64 != lane.len || crate::crc32c::crc32c(&assembled) != lane.crc {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "lane {name} series {}: the reassembled segments do not match the \
                 manifest — the bucket rotted, or an object was replaced",
                lane.series
            ),
        ));
    }
    Ok(assembled)
}

/// The newest generation whose baseline finished (`complete` exists).
/// A newer INCOMPLETE generation is normal — a writer that claimed
/// and is mid-baseline, or died there — and restoring it would hand
/// back a directory with holes; fall back to the newest complete one.
pub(crate) async fn newest_complete_generation(
    store: &dyn ObjectStore,
    root: &StorePath,
) -> io::Result<u64> {
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

pub(crate) async fn fetch(store: &dyn ObjectStore, key: &StorePath) -> io::Result<Vec<u8>> {
    // A missing object keeps its kind: hydration's mismatch arbiter
    // (`hydrate::Hydrator::refreshed_extent`) tells "the lineage moved
    // and took this object with it" (retryable against a re-read
    // manifest) from every other download failure by exactly this.
    let wrap = |error: object_store::Error| {
        let kind = if matches!(error, object_store::Error::NotFound { .. }) {
            io::ErrorKind::NotFound
        } else {
            io::ErrorKind::Other
        };
        io::Error::new(kind, format!("downloading {key}: {error}"))
    };
    let result = store.get(key).await.map_err(wrap)?;
    let bytes = result.bytes().await.map_err(wrap)?;
    Ok(bytes.to_vec())
}
