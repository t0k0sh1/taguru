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
/// `successes` stage, commit, unlink, WAL append, or WAL truncate
/// operations have run normally.
/// Keeping the counter thread-local makes parallel tests independent,
/// and routing every operation through shared choke points avoids a
/// flag for each call site.
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

/// A unique per-process staging name beside `path`. Concurrent
/// stagers (the flusher against an eviction, a shutdown flush against
/// a tick) must never write the same temporary file — with a fixed
/// name, one truncates the other mid-write and a torn image gets
/// renamed into place. Leftovers from a crash are swept at boot.
fn staging_path(path: &Path) -> PathBuf {
    static STAGING_NONCE: AtomicU64 = AtomicU64::new(0);
    let nonce = STAGING_NONCE.fetch_add(1, Ordering::Relaxed);
    path.with_extension(format!("tmp{nonce}"))
}

/// The heavy half of [`write_atomic`]: writes and fsyncs `bytes` under
/// a staging name beside `path`. Safe to run without any lock — the
/// file is invisible until [`commit_staged`] publishes it. Exposed
/// beyond `write_atomic`/`write_atomic_private` for flush, which
/// stages outside the entry lock and only takes the lock to publish.
pub(crate) fn stage_bytes(path: &Path, bytes: &[u8], private: bool) -> io::Result<PathBuf> {
    use std::io::Write;

    #[cfg(not(unix))]
    let _ = private;
    let staged = staging_path(path);
    if let Some(error) = injected_persistence_failure("stage") {
        return Err(error);
    }
    // A private file must be BORN owner-only: create-then-chmod leaves a
    // window (the default-umask create, ~0644, before the chmod) in
    // which another local account can open() the staging file and keep
    // reading it — the secret bytes land in that fd afterwards. `mode`
    // on the open() sets the creation mode atomically, so no readable
    // moment ever exists. `create_new` also refuses to reuse a file an
    // attacker pre-created, closing the mirror-image swap. The staging
    // name is per-process unique (`staging_path`), so create_new never
    // collides with our own concurrent stagers.
    let open = |staged: &Path| -> io::Result<fs::File> {
        #[cfg(unix)]
        if private {
            use std::os::unix::fs::OpenOptionsExt;
            return fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(staged);
        }
        fs::File::create(staged)
    };
    let write = open(&staged).and_then(|mut file| {
        file.write_all(bytes)?;
        file.sync_all()
    });
    match write {
        Ok(()) => Ok(staged),
        Err(error) => {
            // The file (if it even got created) never held valid
            // content and was never handed to a caller — remove it
            // rather than leave a partial write behind under its
            // temporary name.
            let _ = remove_persisted_file(&staged);
            Err(error)
        }
    }
}

/// The cheap half of [`write_atomic`]: atomically publishes a staged
/// file at its final path — rename plus parent-directory fsync. Also
/// the whole of a same-directory file move (`staged` need not be a
/// `.tmp*` name): `rename_group` and `groups::scan_groups`'s
/// rename-marker resume both use it that way. [`move_context_files`]
/// moves nine files under one parent and fsyncs once itself instead of
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
/// rename or creation.
pub(crate) fn fsync_parent_dir(path: &Path) -> io::Result<()> {
    match path.parent() {
        Some(parent) => fsync_dir(parent),
        None => Ok(()),
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
