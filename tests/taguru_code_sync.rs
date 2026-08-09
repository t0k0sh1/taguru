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
    /// Where this test's usage log lands. Every spawn of the binary
    /// must point `TAGURU_USAGE_LOG_DIR` here (outside the repo, so
    /// log writes never read as working-tree changes) — without it
    /// the binary would append to the developer's real
    /// `$HOME/.taguru/logs`.
    logs_dir: PathBuf,
}

impl Repo {
    fn new() -> Repo {
        // Not a timestamp: two tests constructing Repos concurrently
        // have collided inside the clock's real resolution ("could not
        // lock config file … File exists" from the raced `git init`),
        // and the process id deduplicates nothing within one test
        // binary. A counter is unique by construction.
        static NEXT_REPO: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let name = format!(
            "taguru-code-e2e-{}-{}",
            std::process::id(),
            NEXT_REPO.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        );
        let dir = std::env::temp_dir().join(&name);
        let logs_dir = std::env::temp_dir().join(format!("{name}-logs"));
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&logs_dir);
        fs::create_dir_all(&dir).unwrap();
        let repo = Repo { dir, logs_dir };
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
        self.run_with_env(args, &[])
    }

    fn run_with_env(&self, args: &[&str], envs: &[(&str, &str)]) -> (i32, String) {
        let mut command = Command::new(env!("CARGO_BIN_EXE_taguru-code"));
        command
            .current_dir(&self.dir)
            .env("TAGURU_USAGE_LOG_DIR", &self.logs_dir)
            .args(args);
        for (key, value) in envs {
            command.env(key, value);
        }
        let output = command.output().unwrap();
        (
            output.status.code().unwrap_or(-1),
            format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            ),
        )
    }

    /// Every usage record written so far, in append order per file.
    fn usage_records(&self) -> Vec<serde_json::Value> {
        let Ok(entries) = fs::read_dir(&self.logs_dir) else {
            return Vec::new();
        };
        let mut paths: Vec<PathBuf> = entries.filter_map(|e| Some(e.ok()?.path())).collect();
        paths.sort();
        paths
            .iter()
            .flat_map(|path| {
                fs::read_to_string(path)
                    .unwrap()
                    .lines()
                    .map(|line| serde_json::from_str(line).expect("valid JSONL"))
                    .collect::<Vec<serde_json::Value>>()
            })
            .collect()
    }
}

impl Drop for Repo {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
        let _ = fs::remove_dir_all(&self.logs_dir);
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
        .env("TAGURU_USAGE_LOG_DIR", &repo.logs_dir)
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
        .env("TAGURU_USAGE_LOG_DIR", &repo.logs_dir)
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

    // The watch's re-syncs land in the usage log as "watch-sync"
    // (the initial sync plus the one that picked up src/fresh.rs).
    // Each record is appended just AFTER its sync returns, so the
    // second one may trail the find that proved the sync happened —
    // poll briefly instead of asserting the instant `seen` flips.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let watch_syncs = repo
            .usage_records()
            .iter()
            .filter(|record| record["command"] == "watch-sync")
            .count();
        if watch_syncs >= 2 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "expected at least 2 watch-sync usage records, got {watch_syncs}"
        );
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

#[test]
fn sync_refuses_outside_a_git_repository() {
    let dir = std::env::temp_dir().join(format!("taguru-code-nonrepo-e2e-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let logs_dir = dir.join("logs"); // hermetic: never the real $HOME
    let output = Command::new(env!("CARGO_BIN_EXE_taguru-code"))
        .current_dir(&dir)
        .env("TAGURU_USAGE_LOG_DIR", &logs_dir)
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

/// Offline syncs exit before the passage store's own compaction
/// triggers (server write-path floor, eviction) can ever run, so the
/// append log used to ratchet toward 4 MiB across runs (issue #452).
/// The end of sync must fold a log that outgrew the snapshot into
/// `<context>.passages.bin` — and must NOT rewrite the snapshot for a
/// trickle under the floor.
#[test]
fn sync_compacts_the_passage_log_once_it_outgrows_the_snapshot() {
    let repo = Repo::new();
    // Enough symbols that the first sync's passage appends clear the
    // 64 KiB sync floor by a wide margin.
    let body: String = (0..2000)
        .map(|i| format!("pub fn pad_{i}() {{}}\n"))
        .collect();
    repo.write("src/lib.rs", &body);
    repo.commit("base");
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0, "{out}");

    let wal = repo.dir.join(".taguru/code.passages.wal.jsonl");
    let snapshot = repo.dir.join(".taguru/code.passages.bin");
    assert!(
        snapshot.exists(),
        "sync must fold the outgrown log into a snapshot"
    );
    assert_eq!(
        fs::metadata(&wal).unwrap().len(),
        0,
        "the covered log must be truncated"
    );

    // One small file is well under the floor: the log keeps it and
    // the snapshot is not rewritten over a trickle.
    repo.write("src/extra.rs", "pub fn one_more() {}\n");
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0, "{out}");
    assert!(
        fs::metadata(&wal).unwrap().len() > 0,
        "a trickle under the floor must stay in the log"
    );
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
        .env("TAGURU_USAGE_LOG_DIR", &repo.logs_dir)
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

/// The usage log (this feature's whole point): every verb appends one
/// JSONL record — command, args, timing, exit code, and for the read
/// verbs the hit count (0 included — the improvement signal) and the
/// bytes printed.
#[test]
fn usage_log_records_each_invocation_with_hits_and_bytes() {
    let repo = Repo::new();
    repo.write("src/lib.rs", "pub fn locate_me() {}\n");
    repo.commit("base");
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0, "{out}");
    let (code, _) = repo.run(&["find", "locate_me"]);
    assert_eq!(code, 0);
    let (code, _) = repo.run(&["find", "no_such_symbol_anywhere"]);
    assert_eq!(code, 1);
    let (code, _) = repo.run(&["tree", "src"]);
    assert_eq!(code, 0);

    let records = repo.usage_records();
    let commands: Vec<&str> = records
        .iter()
        .map(|record| record["command"].as_str().unwrap())
        .collect();
    assert_eq!(commands, ["sync", "find", "find", "tree"], "{records:?}");
    for record in &records {
        assert!(record["ts"].as_str().unwrap().ends_with('Z'), "{record}");
        assert!(record["duration_ms"].is_u64(), "{record}");
        // The recorded root is the discovered worktree (possibly a
        // canonicalized spelling of repo.dir, e.g. /private/tmp on
        // macOS) — pin the unique tail, not the prefix.
        let root = record["project_root"].as_str().unwrap();
        assert!(
            root.ends_with(repo.dir.file_name().unwrap().to_str().unwrap()),
            "{record}"
        );
    }
    assert_eq!(records[0]["args"], serde_json::json!(["."]));
    assert_eq!(records[0]["exit_code"], 0);
    // The hit: 1 result, some bytes printed.
    assert_eq!(records[1]["hits"], 1, "{}", records[1]);
    assert!(records[1]["output_bytes"].as_u64().unwrap() > 0);
    // The miss: hits: 0 recorded, exit 1, nothing printed to stdout.
    assert_eq!(
        records[2]["args"],
        serde_json::json!(["no_such_symbol_anywhere"])
    );
    assert_eq!(records[2]["hits"], 0, "{}", records[2]);
    assert_eq!(records[2]["exit_code"], 1);
    assert!(records[2]["output_bytes"].is_null());
    // tree src lists one entry (src/lib.rs).
    assert_eq!(records[3]["hits"], 1, "{}", records[3]);
    assert!(records[3]["output_bytes"].as_u64().unwrap() > 0);
}

#[test]
fn usage_log_off_writes_nothing() {
    let repo = Repo::new();
    repo.write("src/lib.rs", "pub fn here() {}\n");
    repo.commit("base");
    for value in ["off", "0", "false"] {
        let (code, out) = repo.run_with_env(&["sync", "."], &[("TAGURU_USAGE_LOG", value)]);
        assert_eq!(code, 0, "{out}");
    }
    assert!(
        repo.usage_records().is_empty(),
        "disabled log must write nothing"
    );
}

/// The cap: oldest days go first, today's file survives even when the
/// cap is tiny — the record just written must outlive its own
/// enforcement pass.
#[test]
fn usage_log_cap_deletes_oldest_days_but_keeps_today() {
    let repo = Repo::new();
    repo.write("src/lib.rs", "pub fn here() {}\n");
    repo.commit("base");
    fs::create_dir_all(&repo.logs_dir).unwrap();
    fs::write(repo.logs_dir.join("usage-20200101.jsonl"), vec![b'x'; 4096]).unwrap();
    fs::write(repo.logs_dir.join("usage-20200102.jsonl"), vec![b'x'; 64]).unwrap();

    let (code, out) = repo.run_with_env(&["sync", "."], &[("TAGURU_USAGE_LOG_MAX_BYTES", "600")]);
    assert_eq!(code, 0, "{out}");

    assert!(
        !repo.logs_dir.join("usage-20200101.jsonl").exists(),
        "the oldest day must be deleted to fit the cap"
    );
    assert!(
        repo.logs_dir.join("usage-20200102.jsonl").exists(),
        "a day that fits once the oldest is gone must survive"
    );
    // Today's file is the one that is neither planted dummy (the
    // dummies hold 'x' padding, not JSONL, so parse it alone).
    let todays: Vec<PathBuf> = fs::read_dir(&repo.logs_dir)
        .unwrap()
        .filter_map(|entry| Some(entry.ok()?.path()))
        .filter(|path| !path.ends_with("usage-20200102.jsonl"))
        .collect();
    assert_eq!(todays.len(), 1, "{todays:?}");
    let content = fs::read_to_string(&todays[0]).unwrap();
    let record: serde_json::Value = serde_json::from_str(content.lines().next().unwrap()).unwrap();
    assert_eq!(record["command"], "sync", "today's record survives");
}

/// `models` lists the local-embedding catalog in recommendation
/// order, feature or no feature — the default build must still teach
/// what a `--features local-embed` build could run.
#[test]
fn every_verb_answers_help_with_the_full_usage_and_exit_zero() {
    let repo = Repo::new();
    // `--help` used to fall into each verb's own parser as an unknown
    // flag (exit 2) — the most natural question an agent asks a CLI
    // was the one it refused. `watch --help` doubly so: without the
    // intercept it would loop forever instead of answering.
    for verb in [
        "sync", "watch", "find", "tree", "status", "evalset", "eval", "models",
    ] {
        let (code, out) = repo.run(&[verb, "--help"]);
        assert_eq!(code, 0, "{verb} --help: {out}");
        assert!(out.contains("USAGE:"), "{verb} --help: {out}");
    }
    // The bare spellings answer identically…
    for bare in [&["help"][..], &["--help"][..], &["-h"][..]] {
        let (code, out) = repo.run(bare);
        assert_eq!(code, 0, "{bare:?}: {out}");
        assert!(out.contains("USAGE:"), "{bare:?}: {out}");
    }
    // …while NO arguments at all is still the exit-2 nudge.
    let (code, out) = repo.run(&[]);
    assert_eq!(code, 2, "{out}");
    assert!(out.contains("USAGE:"), "{out}");

    let (code, out) = repo.run(&["find", "-h"]);
    assert_eq!(code, 0, "{out}");
    // The usage names every flag the verbs accept — the doc page and
    // --help must not disagree about what exists.
    for flag in [
        "--dry-run",
        "--interval-ms",
        "--json",
        "--limit",
        "--out",
        "--sample",
        "--eval",
        "--thresholds",
        "--context",
        "--data-dir",
    ] {
        assert!(out.contains(flag), "usage omits {flag}: {out}");
    }
}

#[test]
fn models_lists_the_catalog_best_first_and_speaks_json() {
    let repo = Repo::new();
    let (code, out) = repo.run(&["models"]);
    assert_eq!(code, 0, "{out}");
    let paraphrase = out
        .find("paraphrase-multilingual-minilm-l12-v2-q")
        .expect("the recommended default must be listed");
    let minilm = out.find("all-minilm-l6-v2-q").expect("catalog entry");
    assert!(
        paraphrase < minilm,
        "recommendation order: the multilingual default comes first\n{out}"
    );
    assert!(out.contains("Apache-2.0") && out.contains("MIT"), "{out}");

    let (code, out) = repo.run(&["models", "--json"]);
    assert_eq!(code, 0, "{out}");
    let parsed: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
    let models = parsed["models"].as_array().unwrap();
    assert_eq!(models[0]["name"], "paraphrase-multilingual-minilm-l12-v2-q");
    assert!(
        models
            .iter()
            .all(|m| m["license"] == "Apache-2.0" || m["license"] == "MIT")
    );
    assert!(parsed["runnable"].is_boolean());

    // fastembed lets HF_HOME override the cache directory it is
    // handed, so the listing must report the EFFECTIVE path, not a
    // hardcoded ~/.taguru/models promise.
    let hf_home = repo.dir.join("hf-cache");
    let (code, out) = repo.run_with_env(
        &["models", "--json"],
        &[("HF_HOME", hf_home.to_str().unwrap())],
    );
    assert_eq!(code, 0, "{out}");
    let parsed: serde_json::Value = serde_json::from_str(out.trim()).unwrap();
    assert_eq!(
        parsed["cache_dir"].as_str().unwrap(),
        hf_home.to_str().unwrap(),
        "HF_HOME must win over the default cache path"
    );

    let (code, _) = repo.run(&["models", "--nonsense"]);
    assert_eq!(code, 2, "unknown flags must refuse, not be ignored");
}
