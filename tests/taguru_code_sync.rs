//! End-to-end freshness (evaluation axis 3): the taguru-code binary
//! against a real throwaway git repository — full sync, then an
//! edit+commit incremental sync must move the line locators, and a
//! rename+delete+commit sync must retire the old names. Everything
//! runs the built binary (`CARGO_BIN_EXE_taguru-code`), so this is
//! the exact surface an agent drives.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

struct Repo {
    dir: PathBuf,
}

impl Repo {
    fn new() -> Repo {
        let dir = std::env::temp_dir().join(format!(
            "taguru-code-e2e-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let repo = Repo { dir };
        repo.git(&["init", "-q", "-b", "main"]);
        repo.git(&["config", "user.email", "e2e@example.com"]);
        repo.git(&["config", "user.name", "e2e"]);
        repo
    }

    fn git(&self, args: &[&str]) {
        let output = Command::new("git")
            .arg("-C")
            .arg(&self.dir)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn write(&self, path: &str, contents: &str) {
        let full = self.dir.join(path);
        fs::create_dir_all(full.parent().unwrap()).unwrap();
        fs::write(full, contents).unwrap();
    }

    fn commit(&self, message: &str) {
        self.git(&["add", "-A"]);
        self.git(&["commit", "-q", "-m", message]);
    }

    /// Runs taguru-code in the repo; returns (exit code, stdout).
    fn run(&self, args: &[&str]) -> (i32, String) {
        let output = Command::new(env!("CARGO_BIN_EXE_taguru-code"))
            .current_dir(&self.dir)
            .args(args)
            .output()
            .unwrap();
        (
            output.status.code().unwrap_or(-1),
            format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ),
        )
    }
}

impl Drop for Repo {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn sync_find_edit_rename_delete_round_trip() {
    let repo = Repo::new();
    repo.write(
        "src/alpha.rs",
        "pub fn locate_me() {}\npub struct Anchor;\n",
    );
    repo.write("src/beta.rs", "pub fn doomed() {}\n");
    repo.write("README.md", "not code\n");
    repo.commit("base");

    // Full sync builds .taguru and finds the symbol with exact lines.
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0, "full sync: {out}");
    assert!(repo.dir.join(".taguru").is_dir());
    let (code, out) = repo.run(&["find", "locate_me"]);
    assert_eq!(code, 0, "find after full sync: {out}");
    assert!(out.contains("src/alpha.rs::locate_me"), "{out}");
    assert!(out.contains("src/alpha.rs:1-1"), "{out}");

    // A no-op re-sync is a fast, honest no-op.
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0);
    assert!(out.contains("up to date"), "{out}");

    // Edit + commit: the incremental sync must move the locator.
    repo.write(
        "src/alpha.rs",
        "// a comment pushes everything down\n\npub fn locate_me() {\n    // body\n}\npub struct Anchor;\n",
    );
    repo.commit("shift lines");
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0, "incremental sync: {out}");
    let (_, out) = repo.run(&["find", "locate_me"]);
    assert!(out.contains("src/alpha.rs:3-5"), "locator must move: {out}");

    // A clean tree at the anchor is a no-op...
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0);
    assert!(out.contains("up to date"), "{out}");

    // A tracked file edited but NOT committed must sync too, and
    // reverting it must heal the locator back — this is the
    // current-dirty ∪ previous-dirty bookkeeping's core path.
    repo.write(
        "src/alpha.rs",
        "// a comment pushes everything down\n\n\npub fn locate_me() {\n    // body\n}\npub struct Anchor;\n",
    );
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0, "dirty sync: {out}");
    let (_, out) = repo.run(&["find", "locate_me"]);
    assert!(
        out.contains("src/alpha.rs:4-6"),
        "dirty edit must sync: {out}"
    );
    // Re-syncing the same dirty-but-stable tree is fingerprint-
    // skipped: no re-import, no registry boot, just the anchor move.
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0, "{out}");
    assert!(
        out.contains("nothing to apply (1 unchanged"),
        "stable dirty file must be fingerprint-skipped: {out}"
    );

    repo.git(&["checkout", "--", "src/alpha.rs"]);
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0, "{out}");
    let (_, out) = repo.run(&["find", "locate_me"]);
    assert!(out.contains("src/alpha.rs:3-5"), "revert must heal: {out}");

    // ...but the working tree IS the universe: an untracked file's
    // symbol becomes findable on sync, no commit required.
    repo.write("src/delta.rs", "pub fn quokka_untracked() {}\n");
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0, "{out}");
    let (code, out) = repo.run(&["find", "quokka_untracked"]);
    assert_eq!(code, 0, "untracked symbol must be findable: {out}");
    assert!(out.contains("src/delta.rs:1-1"), "{out}");

    // Deleting that untracked file heals on the next sync — it is in
    // no git diff, only the recorded dirty set can catch it.
    fs::remove_file(repo.dir.join("src/delta.rs")).unwrap();
    let (code, _) = repo.run(&["sync", "."]);
    assert_eq!(code, 0);
    let (code, _) = repo.run(&["find", "quokka_untracked"]);
    assert_eq!(code, 1, "removed untracked symbol must be gone");

    // Rename + delete + commit: old names retire, new ones appear.
    repo.git(&["mv", "src/alpha.rs", "src/gamma.rs"]);
    fs::remove_file(repo.dir.join("src/beta.rs")).unwrap();
    repo.commit("rename and delete");
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0, "rename/delete sync: {out}");
    assert!(out.contains("retracted 2"), "beta + old alpha: {out}");
    let (_, out) = repo.run(&["find", "locate_me"]);
    assert!(out.contains("src/gamma.rs::locate_me"), "{out}");
    assert!(
        !out.contains("src/alpha.rs"),
        "old path must be gone: {out}"
    );
    let (code, _) = repo.run(&["find", "doomed"]);
    assert_eq!(code, 1, "deleted file's symbol must be gone");

    // tree stays coherent after the churn.
    let (code, out) = repo.run(&["tree", "src"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("src/gamma.rs"), "{out}");
    assert!(!out.contains("src/beta.rs"), "{out}");
}

/// A fact-model bump voids fingerprints but must NOT void deletion
/// tracking: files deleted since the last sync are absent from the
/// full listing, and only the old anchor's diff can retract them.
#[test]
fn fact_model_rebuild_still_retracts_deleted_sources() {
    let repo = Repo::new();
    repo.write("src/alpha.rs", "pub fn locate_me() {}\n");
    repo.write("src/beta.rs", "pub fn doomed() {}\n");
    repo.commit("base");
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0, "{out}");

    fs::remove_file(repo.dir.join("src/beta.rs")).unwrap();
    repo.commit("delete beta");

    // Simulate a binary upgrade: the recorded facts version is older.
    let state_path = repo.dir.join(".taguru/code-sync.json");
    let doctored = fs::read_to_string(&state_path)
        .unwrap()
        .replace("\"facts_version\":1", "\"facts_version\":0");
    fs::write(&state_path, doctored).unwrap();

    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("full re-sync"), "{out}");
    let (code, _) = repo.run(&["find", "doomed"]);
    assert_eq!(code, 1, "deleted source must retract across a rebuild");
    let (code, out) = repo.run(&["find", "locate_me"]);
    assert_eq!(code, 0, "survivors must re-import: {out}");
}

/// A repo with nothing to import must still complete its first sync
/// (the state file needs its directory created without the boot).
#[test]
fn first_sync_of_a_repo_with_no_supported_files_succeeds() {
    let repo = Repo::new();
    repo.write("README.md", "prose only\n");
    repo.commit("base");
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("nothing to apply"), "{out}");
    assert!(repo.dir.join(".taguru/code-sync.json").is_file());
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("up to date"), "{out}");
}

/// Kills the watch process even when an assertion panics first.
struct WatchGuard(std::process::Child);

impl Drop for WatchGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// An edit racing the INITIAL sync must not vanish: watch records its
/// baseline snapshot before that sync, so anything landing during it
/// still reads as a change on a later poll. The repo is padded so the
/// initial sync takes long enough for the immediate write to land
/// inside it (and the test stays valid — just race-free — when it
/// doesn't).
#[test]
fn watch_catches_an_edit_racing_the_initial_sync() {
    let repo = Repo::new();
    for index in 0..150 {
        repo.write(
            &format!("src/pad_{index}.rs"),
            &format!("pub fn pad_{index}() {{}}\n"),
        );
    }
    repo.commit("base");

    let child = Command::new(env!("CARGO_BIN_EXE_taguru-code"))
        .current_dir(&repo.dir)
        .args(["watch", ".", "--interval-ms", "100"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let _guard = WatchGuard(child);
    // No waiting: this write races the initial sync on purpose.
    repo.write("src/raced.rs", "pub fn wrote_during_initial_sync() {}\n");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut seen = false;
    while std::time::Instant::now() < deadline {
        let (code, out) = repo.run(&["find", "wrote_during_initial_sync"]);
        if code == 0 && out.contains("src/raced.rs:1-1") {
            seen = true;
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    assert!(seen, "an edit during the initial sync must not be lost");
}

#[test]
fn watch_follows_edits_without_manual_syncs() {
    let repo = Repo::new();
    repo.write("src/lib.rs", "pub fn already_here() {}\n");
    repo.commit("base");

    let child = Command::new(env!("CARGO_BIN_EXE_taguru-code"))
        .current_dir(&repo.dir)
        .args(["watch", ".", "--interval-ms", "100"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let _guard = WatchGuard(child);

    // The initial sync lands first; then an untracked file appears
    // and must become findable with no manual sync. Polling stays at
    // a HUMANE cadence: each find boots the registry and takes the
    // data-dir lock, and hammering it every 100 ms can starve watch's
    // own sync out of the lock for the whole deadline — the agent
    // asks occasionally; so does this test.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut seen = false;
    let mut wrote = false;
    while std::time::Instant::now() < deadline {
        if !wrote {
            let (code, _) = repo.run(&["find", "already_here"]);
            if code == 0 {
                // Initial sync done — now change the tree.
                repo.write("src/fresh.rs", "pub fn appeared_by_watch() {}\n");
                wrote = true;
            }
        } else {
            let (code, out) = repo.run(&["find", "appeared_by_watch"]);
            if code == 0 && out.contains("src/fresh.rs:1-1") {
                seen = true;
                break;
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
    assert!(
        seen,
        "watch must pick up the new file without a manual sync"
    );
}

#[test]
fn sync_refuses_outside_a_git_repository() {
    let dir = std::env::temp_dir().join(format!("taguru-code-nonrepo-e2e-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_taguru-code"))
        .current_dir(&dir)
        .args(["sync", "."])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let _ = fs::remove_dir_all(&dir);
}

/// Axis-1 machinery end to end: evalset derives cases from the AST,
/// eval replays them through find and gates on thresholds.
#[test]
fn evalset_and_eval_gate_round_trip() {
    let repo = Repo::new();
    repo.write(
        "src/lib.rs",
        "pub fn one() {}\npub fn two() {}\npub struct Three;\nimpl Three { pub fn four(&self) {} }\n",
    );
    repo.commit("base");
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0, "{out}");

    let eval_path = repo.dir.join("eval.jsonl");
    let (code, out) = repo.run(&[
        "evalset",
        "--out",
        eval_path.to_str().unwrap(),
        "--sample",
        "10",
    ]);
    assert_eq!(code, 0, "{out}");

    let (code, out) = repo.run(&["eval", "--eval", eval_path.to_str().unwrap()]);
    assert_eq!(code, 0, "report-only eval: {out}");
    assert!(
        out.contains("\"hit1_rate\":1.0"),
        "tiny repo must be perfect: {out}"
    );

    // A passing gate exits 0; an impossible gate exits 3.
    let pass = repo.dir.join("pass.json");
    fs::write(&pass, "{\"hit1_rate\": 0.9, \"line_drift\": 0}\n").unwrap();
    let (code, out) = repo.run(&[
        "eval",
        "--eval",
        eval_path.to_str().unwrap(),
        "--thresholds",
        pass.to_str().unwrap(),
    ]);
    assert_eq!(code, 0, "{out}");

    let fail = repo.dir.join("fail.json");
    fs::write(&fail, "{\"hit1_rate\": 1.1}\n").unwrap();
    let (code, _) = repo.run(&[
        "eval",
        "--eval",
        eval_path.to_str().unwrap(),
        "--thresholds",
        fail.to_str().unwrap(),
    ]);
    assert_eq!(code, 3, "violated gate must exit 3");
}

/// Agents pipe find/tree/status through `head`/`grep -m`, which close
/// the pipe as soon as they have enough — the binary must die quietly
/// on SIGPIPE like grep does, not panic in println! (issue #453).
#[cfg(unix)]
#[test]
fn closed_downstream_pipe_kills_quietly_instead_of_panicking() {
    use std::os::unix::process::ExitStatusExt;
    use std::process::Stdio;

    let repo = Repo::new();
    // Enough symbols that tree's output overflows a pipe buffer
    // (64 KiB on Linux/macOS): the child can never exit cleanly
    // without a reader, so the SIGPIPE path is exercised even if it
    // starts writing before our read end is closed below.
    let body: String = (0..4000)
        .map(|i| format!("pub fn pad_{i}() {{}}\n"))
        .collect();
    repo.write("src/lib.rs", &body);
    repo.commit("base");
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0, "{out}");

    let mut child = Command::new(env!("CARGO_BIN_EXE_taguru-code"))
        .current_dir(&repo.dir)
        .args(["tree", "src/lib.rs"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Drop the only read end: every stdout write now raises SIGPIPE.
    drop(child.stdout.take());
    let output = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panicked"),
        "broken pipe must not panic: {stderr}"
    );
    assert_eq!(
        output.status.signal(),
        Some(13), // SIGPIPE
        "expected death by SIGPIPE, got {:?}: {stderr}",
        output.status
    );
}
