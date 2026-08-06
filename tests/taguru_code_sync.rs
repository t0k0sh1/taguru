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

    // Uncommitted edits stay invisible — committed state only.
    repo.write("src/alpha.rs", "pub fn zebra_quokka() {}\n");
    let (code, out) = repo.run(&["sync", "."]);
    assert_eq!(code, 0);
    assert!(
        out.contains("up to date"),
        "dirty tree is not new work: {out}"
    );
    let (code, _) = repo.run(&["find", "zebra_quokka"]);
    assert_eq!(code, 1, "uncommitted symbol must not be findable");
    repo.git(&["checkout", "--", "."]);

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
