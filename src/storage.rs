//! Filesystem primitives shared by every on-disk format in the crate
//! (context images, meta sidecars, groups, the OAuth grant store, WAL
//! segments, passage snapshots): atomic stage-then-rename writes,
//! fsync choke points, the data-directory lock, and the blocking-work
//! offload used by every cold load. Callers pick a durability shape
//! ([`write_atomic`] vs [`write_atomic_private`]) and a file layout;
//! everything below that is format-agnostic.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Writes via a temporary file, fsync, and rename — a crash mid-write
/// leaves the previous version intact, and power loss after return
/// cannot tear or lose the new one. The rename itself is an entry in
/// the parent directory's own data, so the parent is fsynced too;
/// without that a crash can forget the rename even though the file
/// contents reached disk.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    write_atomic_with(path, bytes, false)
}

/// [`write_atomic`] for secret-bearing files (the OAuth grant store):
/// the staged file drops to owner-only permissions BEFORE any content
/// lands in it, and the rename carries the mode to the final name —
/// no moment exists where another local account could read the bytes.
/// Non-Unix platforms have no mode bits and get the plain behavior.
pub(crate) fn write_atomic_private(path: &Path, bytes: &[u8]) -> io::Result<()> {
    write_atomic_with(path, bytes, true)
}

/// Test-only deterministic fault injection for registry persistence.
///
/// The calling thread fails exactly one persistence operation after
/// `successes` stage, commit (a publish rename OR a same-directory
/// move — the registry's whole-family context rename included, since
/// it routes every one of its ten renames through
/// [`rename_persisted_file`] like every other rename in the crate),
/// unlink, WAL append, or WAL truncate operations have run normally.
/// Keeping the counter thread-local makes parallel tests independent,
/// and routing every MUTATING operation through shared choke points
/// avoids a flag for each call site. Two things are deliberately
/// outside this net: [`fsync_dir`] (durability, not mutation — a
/// dropped fsync loses nothing a crash-consistency test can observe
/// without also modeling real fsync failures, which this harness does
/// not attempt) and any plain read (`fs::read` — a read has no
/// write-ordering story to get wrong, so injecting into it would only
/// test the reader's own error handling, not persistence ordering).
#[cfg(test)]
pub(crate) fn fail_persistence_ops_after(successes: u32) {
    PERSISTENCE_FAULT.with(|cell| cell.set(Some(successes)));
}

#[cfg(test)]
thread_local! {
    static PERSISTENCE_FAULT: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

/// Clears an armed fault and reports whether it was still pending. A
/// sweep uses this after each attempt: `false` means the selected op
/// was reached and failed, while `true` means the operation had fewer
/// persistence steps and the sweep is complete.
#[cfg(test)]
pub(crate) fn clear_persistence_fault() -> bool {
    PERSISTENCE_FAULT.with(|cell| cell.take().is_some())
}

#[cfg(test)]
pub(crate) fn injected_persistence_failure(operation: &str) -> Option<io::Error> {
    PERSISTENCE_FAULT.with(|cell| match cell.get() {
        Some(0) => {
            cell.set(None);
            Some(io::Error::other(format!(
                "injected registry persistence failure during {operation}"
            )))
        }
        Some(remaining) => {
            cell.set(Some(remaining - 1));
            None
        }
        None => None,
    })
}

#[cfg(not(test))]
pub(crate) fn injected_persistence_failure(_operation: &str) -> Option<io::Error> {
    None
}

#[cfg(test)]
thread_local! {
    static FORCE_STAGING_COLLISIONS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Test-only: forces this thread's next `count` `stage_bytes` staging-
/// name attempts to report as already-taken, exercising the
/// create_new retry loop the way a genuine cross-process name
/// collision would — without needing two real processes racing each
/// other.
#[cfg(test)]
pub(crate) fn force_next_staging_collisions(count: u32) {
    FORCE_STAGING_COLLISIONS.with(|cell| cell.set(count));
}

#[cfg(test)]
fn take_forced_staging_collision() -> bool {
    FORCE_STAGING_COLLISIONS.with(|cell| {
        let remaining = cell.get();
        if remaining > 0 {
            cell.set(remaining - 1);
            true
        } else {
            false
        }
    })
}

#[cfg(not(test))]
fn take_forced_staging_collision() -> bool {
    false
}

/// The unlink choke point shared by registry and group persistence.
pub(crate) fn remove_persisted_file(path: impl AsRef<Path>) -> io::Result<()> {
    if let Some(error) = injected_persistence_failure("unlink") {
        return Err(error);
    }
    fs::remove_file(path)
}

/// The rename choke point shared by atomic publication and recovery
/// paths that move corrupt bytes aside.
pub(crate) fn rename_persisted_file(
    from: impl AsRef<Path>,
    to: impl AsRef<Path>,
) -> io::Result<()> {
    if let Some(error) = injected_persistence_failure("commit") {
        return Err(error);
    }
    fs::rename(from, to)
}

fn write_atomic_with(path: &Path, bytes: &[u8], private: bool) -> io::Result<()> {
    let staged = stage_bytes(path, bytes, private)?;
    let result = commit_staged(&staged, path);
    if result.is_err() {
        // A failed rename leaves `staged` sitting under its temporary
        // name — clean it up rather than leave it for unbounded disk
        // litter until the next boot's sweep. (A failed parent fsync
        // after a successful rename finds nothing here: the file is
        // already `path`, and removing the stale `staged` name is a
        // harmless no-op.)
        let _ = remove_persisted_file(&staged);
    }
    result
}

/// A collision-resistant staging name beside `path`. `std::process::id()`
/// makes same-name collisions between independent PROCESSES exceedingly
/// unlikely (unlike a bare, process-local counter starting at zero,
/// which two separate processes racing the same destination could both
/// pick identically) even before `stage_bytes`'s create_new retry loop
/// has to matter. Concurrent stagers within THIS process (the flusher
/// against an eviction, a shutdown flush against a tick) still never
/// share a name either, since `STAGING_NONCE` only grows. Leftovers
/// from a crash are swept at boot.
fn staging_path(path: &Path) -> PathBuf {
    static STAGING_NONCE: AtomicU64 = AtomicU64::new(0);
    let nonce = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
    // A `-`, not a `.`, between the pid and the nonce: `Path::extension()`
    // only ever returns the bytes after the LAST dot, and boot's sweep
    // (registry/boot.rs) matches leftover staging files by
    // `extension().starts_with("tmp")`. A dot here would leave the
    // `tmp{pid}` half outside the extension entirely — `extension()`
    // would answer just `{nonce}`, which never starts with "tmp" — and
    // a crash-orphaned staging file would silently stop being swept.
    path.with_extension(format!("tmp{}-{nonce}", std::process::id()))
}

/// True when `extension` has exactly the shape [`staging_path`] builds —
/// `tmp{pid}-{nonce}`, digits on both sides of the `-` (plus the
/// hyphen-less `tmp{digits}` form older names carried) — so scanners
/// that must not over-skip (the shipper's directory classifier,
/// `ship::classify`) can recognize staging litter without also
/// swallowing a genuine `*.tmp`-suffixed file. Boot's sweep
/// (registry/boot.rs) deliberately stays broader
/// (`starts_with("tmp")`): the data dir is server-owned, so
/// over-sweeping there is safe, while over-skipping in the shipper
/// would silently drop a real file from every manifest.
pub(crate) fn is_staging_extension(extension: &str) -> bool {
    let Some(rest) = extension.strip_prefix("tmp") else {
        return false;
    };
    let (pid, nonce) = match rest.split_once('-') {
        Some((pid, nonce)) => (pid, Some(nonce)),
        None => (rest, None),
    };
    !pid.is_empty()
        && pid.bytes().all(|byte| byte.is_ascii_digit())
        && nonce.is_none_or(|nonce| {
            !nonce.is_empty() && nonce.bytes().all(|byte| byte.is_ascii_digit())
        })
}

/// Bounded retries for [`stage_bytes`]'s `create_new` loop: enough to
/// absorb a genuine cross-process name collision (astronomically
/// unlikely already thanks to `staging_path`'s process-id'd nonce)
/// without trusting the name alone, while still failing fast — and
/// loudly — rather than spinning forever if something pathological is
/// going on (e.g. a directory full of pre-created `tmp*` names).
const MAX_STAGING_ATTEMPTS: u32 = 8;

/// The heavy half of [`write_atomic`]: writes and fsyncs `bytes` under
/// a staging name beside `path`. Safe to run without any lock — the
/// file is invisible until [`commit_staged`] publishes it. Exposed
/// beyond `write_atomic`/`write_atomic_private` for flush, which
/// stages outside the entry lock and only takes the lock to publish.
pub(crate) fn stage_bytes(path: &Path, bytes: &[u8], private: bool) -> io::Result<PathBuf> {
    use std::io::Write;

    if let Some(error) = injected_persistence_failure("stage") {
        return Err(error);
    }
    // create_new, private or not — never File::create, which silently
    // truncates whatever already sits under a colliding name. A
    // process-id'd nonce (see `staging_path`) makes a same-name
    // collision between independent processes astronomically unlikely,
    // but "unlikely" still must not mean "corrupts another writer's
    // bytes if it happens anyway" — a handful of retries absorbs a
    // genuine collision instead of trusting the name alone.
    let mut last_error = None;
    for _ in 0..MAX_STAGING_ATTEMPTS {
        let staged = staging_path(path);
        if take_forced_staging_collision() {
            last_error = Some(io::Error::from(io::ErrorKind::AlreadyExists));
            continue;
        }
        let open = {
            #[cfg(unix)]
            {
                // A private file must be BORN owner-only:
                // create-then-chmod leaves a window (the default-umask
                // create, ~0644, before the chmod) in which another
                // local account can open() the staging file and keep
                // reading it — the secret bytes land in that fd
                // afterwards. `mode` on the open() sets the creation
                // mode atomically, so no readable moment ever exists.
                use std::os::unix::fs::OpenOptionsExt;
                let mode = if private { 0o600 } else { 0o666 };
                fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(mode)
                    .open(&staged)
            }
            #[cfg(not(unix))]
            {
                let _ = private;
                fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&staged)
            }
        };
        let mut file = match open {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                last_error = Some(error);
                continue;
            }
            Err(error) => return Err(error),
        };
        let write = file.write_all(bytes).and_then(|()| file.sync_all());
        return match write {
            Ok(()) => Ok(staged),
            Err(error) => {
                // The file never held valid content and was never
                // handed to a caller — remove it (ours, and ours
                // alone: create_new just proved no one else could have
                // been writing under this exact name) rather than
                // leave a partial write behind under its temporary
                // name.
                let _ = remove_persisted_file(&staged);
                Err(error)
            }
        };
    }
    Err(last_error
        .unwrap_or_else(|| io::Error::other("could not allocate a unique staging file name")))
}

/// The cheap half of [`write_atomic`]: atomically publishes a staged
/// file at its final path — rename plus parent-directory fsync. Also
/// the whole of a same-directory file move (`staged` need not be a
/// `.tmp*` name): `rename_group` and `groups::scan_groups`'s
/// rename-marker resume both use it that way. [`move_context_files`]
/// moves ten files under one parent and fsyncs once itself instead of
/// calling this per file — see there.
pub(crate) fn commit_staged(staged: &Path, path: &Path) -> io::Result<()> {
    rename_persisted_file(staged, path)?;
    fsync_parent_dir(path)
}

/// Takes the advisory exclusive lock (`.taguru.lock`) that admits one
/// registry per data directory at a time: `taguru serve` and `taguru
/// import` both boot through here, so whichever is second gets a
/// refusal naming the conflict instead of a silent last-flush-wins
/// overwrite. The lock lives on the open descriptor, not the file —
/// a crash releases it with the process, and the empty lock file
/// left behind means nothing. Advisory: it binds taguru processes,
/// not arbitrary tools, and network filesystems honor it unreliably.
pub(crate) fn lock_data_dir(dir: &Path) -> io::Result<fs::File> {
    let file = fs::File::create(dir.join(".taguru.lock"))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(fs::TryLockError::WouldBlock) => Err(io::Error::other(format!(
            "data directory {} is held by another taguru process \
             (a running serve, or an import) — stop that one first",
            dir.display()
        ))),
        Err(fs::TryLockError::Error(error)) => Err(error),
    }
}

/// Persists a rename or file creation by syncing the directory that
/// holds it. Unix-only; elsewhere the rename stays atomic against a
/// crash mid-write, just not durable against power loss — unix is
/// what this server targets.
pub(crate) fn fsync_dir(dir: &Path) -> io::Result<()> {
    #[cfg(unix)]
    fs::File::open(dir)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// [`fsync_dir`] on `path`'s parent — the common case, a single file
/// rename or creation. A single-component relative `path` (`"mydata"`,
/// no leading `./`) makes `Path::parent()` answer `Some("")`, not
/// `None` — an empty path `fs::File::open` refuses with `ENOENT`, even
/// though the parent this call actually means is the current
/// directory. Treat that one shape as `"."` rather than skipping it:
/// skipping would silently drop the fsync for exactly the relative,
/// single-segment paths a caller (a bare `--data-dir mydata`, a bare
/// `--out mydata`) is likely to pass.
pub(crate) fn fsync_parent_dir(path: &Path) -> io::Result<()> {
    match path.parent() {
        Some(parent) if parent.as_os_str().is_empty() => fsync_dir(Path::new(".")),
        Some(parent) => fsync_dir(parent),
        None => Ok(()),
    }
}

/// Whether a sidecar read failure is worth a log line. `NotFound` is
/// the routine "never written yet" case every derived-cache `load`
/// treats as cold-start; anything else (`PermissionDenied`, `EIO`, a
/// full disk on some platforms) is a standing problem that silently
/// taxes every residency with a full rebuild and deserves a name.
/// Pulled out as a pure predicate — a branch that only gates a log
/// line is otherwise unkillable by a behavioral test (the same reason
/// `remove_persisted_file_quietly` in `registry/boot.rs` needs
/// `#[mutants::skip]`) — so the mutation gate has something to catch.
pub(crate) fn sidecar_read_worth_warning(error: &io::Error) -> bool {
    error.kind() != io::ErrorKind::NotFound
}

/// Reads a derived-cache sidecar (BM25 index, vector store): `Some` on
/// success, `None` on any read failure — a missing or unreadable
/// sidecar costs a rebuild, never an outage. Unlike a parse failure
/// (which the caller already warns on for its own bytes), a read that
/// never got bytes at all was silent everywhere before this: `what`
/// names the sidecar kind for [`sidecar_read_worth_warning`]'s warning.
///
/// A thin wrapper over [`read_sidecar_checked`] that discards the
/// missing-vs-failed distinction that function keeps — the right
/// choice for a caller that already treats "nothing to load" and "an
/// unreadable file" the same way (a derived cache rebuilds either way).
pub(crate) fn read_sidecar(path: &Path, what: &str) -> Option<Vec<u8>> {
    read_sidecar_checked(path, what).unwrap_or(None)
}

/// Like [`read_sidecar`], but keeps the one distinction that function
/// discards: `Ok(None)` is the benign case (nothing written yet —
/// nothing worth remembering), `Err(())` is a genuine read failure,
/// already warned about here, that a caller wanting to QUARANTINE
/// against (rather than silently re-treat as "empty" on every access)
/// needs to tell apart — see `AppState::entry_vectors`/
/// `entry_passage_vectors` (issue #677 item 3).
pub(crate) fn read_sidecar_checked(path: &Path, what: &str) -> Result<Option<Vec<u8>>, ()> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) => {
            if sidecar_read_worth_warning(&error) {
                // Never a bare `error` field — see the ADR 0008 note on
                // `rename_context`'s `Stuck` arm in `lifecycle.rs`.
                tracing::warn!(
                    path = %path.display(),
                    read_error = %error,
                    "{what} sidecar unreadable; costing a full rebuild every residency until fixed",
                );
                Err(())
            } else {
                Ok(None)
            }
        }
    }
}

/// Runs blocking work — a cold load's disk read plus full-image
/// validation — off the async runtime when called from one:
/// `block_in_place` tells the multi-thread runtime this worker will
/// stall, so queued tasks migrate to other workers instead of waiting
/// behind synchronous IO. The CLI entrances (import, export) and plain
/// `#[test]`s run with no runtime, and a current_thread test runtime
/// cannot block-in-place — both fall through to running the work
/// inline. Nested calls are safe: tokio treats an inner
/// `block_in_place` on an already-blocking thread as a no-op (the
/// api layer wraps writes and passage search in one already).
pub(crate) fn offload<T>(work: impl FnOnce() -> T) -> T {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(work)
        }
        _ => work(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single-component relative path (`"mydata"`, no leading `./`)
    /// makes `Path::parent()` answer `Some("")` — the exact shape
    /// `taguru serve --data-dir mydata` or `taguru restore --out
    /// mydata` produces. `fs::File::open("")` refuses with `ENOENT`,
    /// so this must not go through `fsync_dir` unchanged; it must
    /// resolve to the current directory instead. No file named
    /// "mydata" needs to exist — only its parent (".") does, and every
    /// process has one.
    #[test]
    fn fsync_parent_dir_resolves_a_single_component_relative_paths_empty_parent() {
        fsync_parent_dir(Path::new("taguru-test-fsync-parent-dir-need-not-exist"))
            .expect("an empty Path::parent() must resolve to the current directory, not ENOENT");
    }

    /// A genuine (non-empty) parent that does not exist must surface
    /// `fsync_dir`'s own open failure — the one observable way to tell
    /// this apart from a no-op: a mutated `fsync_parent_dir` that
    /// always returned `Ok(())`, or one whose `is_empty()` guard always
    /// took the `"."` branch, would both silently succeed here instead.
    #[cfg(unix)]
    #[test]
    fn fsync_parent_dir_propagates_a_real_parents_open_failure() {
        let path = std::env::temp_dir()
            .join(format!(
                "taguru-storage-fsync-missing-parent-{}",
                std::process::id()
            ))
            .join("nonexistent-dir")
            .join("file.txt");

        let error = fsync_parent_dir(&path)
            .expect_err("fsyncing a nonexistent parent directory must fail, not silently succeed");
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    /// `NotFound` (the routine "never written yet" cold-start case
    /// every sidecar `load` sees on a fresh context) must stay quiet;
    /// anything else — permission, I/O — is the standing-problem case
    /// #603 found silent: a permanently unreadable sidecar paid a full
    /// rebuild every residency with nothing in the logs.
    #[test]
    fn sidecar_read_worth_warning_is_false_only_for_not_found() {
        assert!(!sidecar_read_worth_warning(&io::Error::from(
            io::ErrorKind::NotFound
        )));
        assert!(sidecar_read_worth_warning(&io::Error::from(
            io::ErrorKind::PermissionDenied
        )));
        assert!(sidecar_read_worth_warning(&io::Error::other(
            "simulated I/O failure"
        )));
    }

    #[test]
    fn read_sidecar_round_trips_bytes_and_answers_none_when_missing() {
        let path = std::env::temp_dir().join(format!(
            "taguru-storage-read-sidecar-{}",
            std::process::id()
        ));
        assert_eq!(
            read_sidecar(&path, "test sidecar"),
            None,
            "a never-written sidecar must answer None, not panic or warn"
        );

        fs::write(&path, b"the sidecar's bytes").unwrap();
        assert_eq!(
            read_sidecar(&path, "test sidecar"),
            Some(b"the sidecar's bytes".to_vec())
        );

        let _ = fs::remove_file(&path);
    }

    /// `Path::extension()` only ever returns the bytes after the LAST
    /// dot — boot's resume-sweep (registry/boot.rs's `scan_data_dir`)
    /// matches leftover staging files by `extension().starts_with("tmp")`.
    /// A `.` between the pid and nonce (instead of `staging_path`'s
    /// actual `-`) would push the `tmp{pid}` half out of the extension
    /// entirely, leaving crash-orphaned staging files unswept at every
    /// later boot.
    #[test]
    fn staging_path_extension_still_starts_with_tmp_for_boots_sweep_to_find() {
        let staged = staging_path(Path::new("output.txt"));
        let extension = staged.extension().and_then(|e| e.to_str());
        assert!(
            extension.is_some_and(|e| e.starts_with("tmp")),
            "boot's sweep would silently stop cleaning up this name: {staged:?}"
        );
    }

    /// The shipper's directory classifier keeps staging litter out of
    /// every manifest through [`is_staging_extension`] — so the
    /// predicate must recognize what [`staging_path`] ACTUALLY builds
    /// (the hyphenated `tmp{pid}-{nonce}`; an earlier all-digits check
    /// missed it and shipped mid-flush stagers, which a replica then
    /// failed to boot on once the rename dropped the object), while
    /// still refusing near-misses a user file could plausibly wear.
    #[test]
    fn staging_extension_predicate_matches_staging_path_and_only_it() {
        let staged = staging_path(Path::new("sake.ctx"));
        let extension = staged.extension().and_then(|e| e.to_str()).unwrap();
        assert!(
            is_staging_extension(extension),
            "the shipper would ship this stager: {staged:?}"
        );
        // The hyphen-less digits form older staging names carried.
        assert!(is_staging_extension("tmp42"));
        // Not stagers: no digits, a bare "tmp", torn or non-digit halves.
        assert!(!is_staging_extension("tmp"));
        assert!(!is_staging_extension("tmp4a-9"));
        assert!(!is_staging_extension("tmp42-"));
        assert!(!is_staging_extension("tmp-9"));
        assert!(!is_staging_extension("tmp42-9-1"));
        assert!(!is_staging_extension("bak42"));
    }

    /// Simulates two independent allocators — two processes racing the
    /// same destination — picking the identical staging name.
    /// `STAGING_NONCE` is a process-global counter (`staging_path`'s
    /// own doc), so nothing within one process can collide with
    /// itself this way; this forces the cross-process shape directly
    /// rather than actually racing two processes. `stage_bytes` must
    /// not give up or, worse, reuse the colliding name — it must retry
    /// onto a fresh, unclaimed name and still succeed.
    #[test]
    fn stage_bytes_retries_past_a_forced_staging_collision() {
        let dir = std::env::temp_dir().join(format!(
            "taguru-storage-stage-collision-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("output.txt");

        // Force two AlreadyExists results before letting a real
        // attempt through.
        force_next_staging_collisions(2);

        let staged = stage_bytes(&path, b"my bytes", false)
            .expect("stage_bytes must retry past forced collisions, not fail the write");
        assert_eq!(fs::read(&staged).unwrap(), b"my bytes");

        let _ = fs::remove_dir_all(&dir);
    }

    /// The sibling above forces a collision through the test hook,
    /// which short-circuits before `stage_bytes` ever calls the real
    /// `OpenOptions::create_new` — the retry loop's actual match guard
    /// on a GENUINE `AlreadyExists` (a real cross-process race, in
    /// production) stays unexercised by every other test in this file.
    /// This one predicts `staging_path`'s next nonce and pre-creates a
    /// real file there, so the open call itself fails for real.
    /// Reliable only because nextest gives every test its own process
    /// — a fresh, zeroed `STAGING_NONCE` seen by nothing but this
    /// test's own calls; the project's other collision tests all use
    /// the forcing hook specifically to avoid depending on that.
    #[test]
    fn stage_bytes_retries_past_a_genuine_open_collision() {
        let dir = std::env::temp_dir().join(format!(
            "taguru-storage-real-collision-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("output.txt");

        // `staging_path`'s own nonce only ever grows, so calling it
        // ourselves to "peek" the next name would consume the very
        // nonce stage_bytes's first attempt is about to draw, leaving
        // our decoy at a name stage_bytes never touches — construct
        // the same shape directly instead. Reliable because nextest
        // gives every test its own process: this is the FIRST call
        // that would ever touch STAGING_NONCE here, so it is
        // guaranteed to be nonce 0.
        let predicted = path.with_extension(format!("tmp{}-0", std::process::id()));
        fs::write(&predicted, b"already here").unwrap();

        let staged = stage_bytes(&path, b"my bytes", false)
            .expect("a genuine AlreadyExists on one attempt must retry, not fail outright");
        assert_ne!(staged, predicted, "must land on a distinct, unclaimed name");
        assert_eq!(
            fs::read(&predicted).unwrap(),
            b"already here",
            "the collided name must be left untouched"
        );
        assert_eq!(fs::read(&staged).unwrap(), b"my bytes");

        let _ = fs::remove_dir_all(&dir);
    }

    /// A collision on every single attempt (pathological, but the
    /// retry loop must still be bounded) reports a clear error rather
    /// than looping forever.
    #[test]
    fn stage_bytes_exhausts_its_retries_and_reports_a_clear_error() {
        let dir = std::env::temp_dir().join(format!(
            "taguru-storage-stage-exhausted-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("output.txt");

        // More than MAX_STAGING_ATTEMPTS, so every real attempt this
        // call could make is forced to collide.
        force_next_staging_collisions(MAX_STAGING_ATTEMPTS * 2);

        let error = stage_bytes(&path, b"my bytes", false)
            .expect_err("exhausting every retry attempt must report an error, not hang or panic");
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);

        // The forcing hook still has collisions armed; clear it so it
        // cannot bleed into a later test sharing this thread.
        force_next_staging_collisions(0);

        let _ = fs::remove_dir_all(&dir);
    }

    /// A forced collision must never let this attempt clobber some
    /// unrelated file that already sits in the staging directory (the
    /// way a real cross-process collision must never clobber whatever
    /// the other process legitimately owns): `stage_bytes` must retry
    /// onto a NEW, distinct name and leave a pre-existing neighbor
    /// completely untouched. The retry loop's forced-collision branch
    /// never performs any filesystem operation on the name it was
    /// forced to treat as taken (`staging_path`'s monotonic nonce
    /// means it can never be asked to reuse it either), so a neighbor
    /// planted before the call is a stand-in for "whatever already
    /// occupied the colliding name" in the real, cross-process case.
    #[test]
    fn stage_bytes_leaves_an_unrelated_neighbor_untouched_across_a_forced_collision() {
        let dir = std::env::temp_dir().join(format!(
            "taguru-storage-stage-collision-owner-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("output.txt");

        // Stand-in for a file some other allocator already legitimately
        // owns under a staging-shaped name in the same directory.
        let neighbor = staging_path(&path);
        fs::write(&neighbor, b"other writer's bytes").unwrap();

        force_next_staging_collisions(1);
        let staged = stage_bytes(&path, b"my bytes", false)
            .expect("stage_bytes must retry onto a new name rather than fail");

        assert_ne!(staged, neighbor, "must land on a distinct, unclaimed name");
        assert_eq!(
            fs::read(&neighbor).unwrap(),
            b"other writer's bytes",
            "the neighbor's file must be untouched"
        );
        assert_eq!(fs::read(&staged).unwrap(), b"my bytes");

        let _ = fs::remove_dir_all(&dir);
    }

    /// The private path must also retry past a collision — and the
    /// file it eventually lands on must still be born owner-only
    /// (mode 0600), never a create-then-chmod window.
    #[cfg(unix)]
    #[test]
    fn stage_bytes_private_retries_past_a_collision_and_stays_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!(
            "taguru-storage-stage-private-collision-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("secret.txt");

        force_next_staging_collisions(2);
        let staged = stage_bytes(&path, b"secret bytes", true)
            .expect("stage_bytes must retry past forced collisions for private writes too");

        assert_eq!(fs::read(&staged).unwrap(), b"secret bytes");
        let mode = fs::metadata(&staged).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "staging file mode {mode:o}");

        let _ = fs::remove_dir_all(&dir);
    }

    /// `lock_data_dir`'s `WouldBlock` arm (issue #587) — untested
    /// despite being the exact branch that answers a concurrent
    /// `taguru serve`/`taguru import` with a named refusal instead of a
    /// silent last-flush-wins overwrite. `flock` locks belong to the
    /// open file description, not the process, so a second, independent
    /// `File::create` + `try_lock` on the same path — exactly what a
    /// second process racing this one would do — reliably contends with
    /// the first even from within one process; no subprocess needed.
    #[test]
    fn lock_data_dir_refuses_a_second_lock_naming_the_conflict() {
        let dir = std::env::temp_dir().join(format!(
            "taguru-storage-lock-data-dir-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();

        let held = lock_data_dir(&dir).expect("the first lock must succeed");

        let error = lock_data_dir(&dir)
            .expect_err("a second concurrent lock attempt must be refused, not silently granted");
        assert!(
            error
                .to_string()
                .contains("is held by another taguru process"),
            "{error}"
        );

        drop(held);
        lock_data_dir(&dir).expect("once the first lock drops, a fresh lock must succeed");

        let _ = fs::remove_dir_all(&dir);
    }

    /// `offload`'s three arms (issue #587): a multi-thread Tokio runtime
    /// routes through `block_in_place`, a current-thread runtime and no
    /// runtime at all (the CLI import/export path, and every plain
    /// `#[test]`) both fall through to running the work inline —
    /// `block_in_place` panics on a current-thread runtime, so picking
    /// the wrong arm there would be immediately visible, not silently
    /// wrong.
    #[test]
    fn offload_runs_the_work_and_returns_its_value_with_no_runtime() {
        assert_eq!(offload(|| 1 + 1), 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn offload_runs_the_work_and_returns_its_value_on_a_current_thread_runtime() {
        assert_eq!(
            tokio::runtime::Handle::current().runtime_flavor(),
            tokio::runtime::RuntimeFlavor::CurrentThread
        );
        assert_eq!(offload(|| 1 + 1), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn offload_runs_the_work_and_returns_its_value_on_a_multi_thread_runtime() {
        assert_eq!(
            tokio::runtime::Handle::current().runtime_flavor(),
            tokio::runtime::RuntimeFlavor::MultiThread
        );
        assert_eq!(offload(|| 1 + 1), 2);
    }

    /// The behavioral reason `offload` picks `block_in_place` on a
    /// multi-thread runtime instead of running inline (the return
    /// value alone, as in the sibling test above, cannot tell the two
    /// apart): a single worker thread must still be able to service
    /// other tasks while `offload`'s closure blocks it. Pinning
    /// `worker_threads = 1` makes this deterministic rather than
    /// dependent on the CI runner's core count — with only one worker,
    /// the spawned task below can be polled during the blocking window
    /// ONLY if `block_in_place` frees this worker for it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn offload_frees_the_lone_worker_so_other_tasks_still_run() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::Duration;

        let ran_while_blocked = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran_while_blocked);
        // Both tasks must be spawned onto the pool, not run inline in
        // this test function's own body: the outer test task is driven
        // by `block_on` on the harness's calling thread, separate from
        // the `worker_threads = 1` pool, so blocking IT proves nothing
        // about the pool's lone worker.
        let blocker = tokio::spawn(async {
            offload(|| std::thread::sleep(Duration::from_millis(150)));
        });
        let checker = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            flag.store(true, Ordering::SeqCst);
        });

        // Check partway through `blocker`'s 150ms window, not after —
        // awaiting both to completion would always see the flag set by
        // then regardless of scheduling order, since `checker` finishes
        // either way once given a turn. This runs on the harness's own
        // calling thread (`block_on`, separate from the `worker_threads
        // = 1` pool), so it keeps progressing on its own timer
        // regardless of what the pool's lone worker is doing.
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert!(
            ran_while_blocked.load(Ordering::SeqCst),
            "with a single worker, block_in_place must free it for the \
             spawned task to run during offload's blocking closure — an \
             offload that skipped block_in_place would starve it until \
             this call returns"
        );

        let _ = tokio::join!(blocker, checker);
    }
}
