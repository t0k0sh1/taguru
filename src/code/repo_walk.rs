//! The git boundary: every fact about which files exist, changed, or
//! vanished comes from `git` subprocess calls, never from a
//! hand-rolled directory walk. The universe is exactly ripgrep's:
//! tracked plus untracked files, minus everything .gitignore excludes
//! — `git ls-files --cached --others --exclude-standard` IS that
//! authority. Incrementally, `git diff --name-status` between the
//! last synced commit and HEAD covers the committed churn and
//! `git status --porcelain` covers the working tree's (staged,
//! unstaged, untracked). A directory that is not a git work tree is
//! a refusal, not a fallback — a bespoke walker would re-implement
//! ignore semantics wrong.

use std::path::{Path, PathBuf};
use std::process::Command;

/// One repository, pinned to its top-level directory.
pub(crate) struct RepoWalk {
    root: PathBuf,
}

/// One path's fate between two commits, from `--name-status -z -M`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Change {
    /// Added, copied, or otherwise new-here: (re)import it.
    Added(String),
    /// Content or type changed: re-import (retract-then-apply makes
    /// this the same call as Added; the variants exist for reporting).
    Modified(String),
    /// Gone: retract its source.
    Deleted(String),
    /// Moved: retract `from`, import `to` — never assumed
    /// content-identical, a rename usually rides with an edit.
    Renamed { from: String, to: String },
}

impl RepoWalk {
    /// Resolves `start` to its repository root via `git rev-parse
    /// --show-toplevel`. A non-repository is an error carrying git's
    /// own words.
    pub(crate) fn discover(start: &Path) -> Result<RepoWalk, String> {
        let top = git_in(start, &["rev-parse", "--show-toplevel"])?;
        let root = String::from_utf8_lossy(&top).trim().to_string();
        if root.is_empty() {
            return Err(format!("{} is not inside a git work tree", start.display()));
        }
        Ok(RepoWalk {
            root: PathBuf::from(root),
        })
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    /// The commit anchor incremental syncs diff against: HEAD's sha.
    /// A repository with no commits yet refuses here — the anchor is
    /// what makes the next sync cheap.
    pub(crate) fn head(&self) -> Result<String, String> {
        let out = self
            .git(&["rev-parse", "HEAD"])
            .map_err(|error| format!("{error} — make at least one commit first"))?;
        Ok(String::from_utf8_lossy(&out).trim().to_string())
    }

    /// Every file ripgrep would search — tracked plus untracked,
    /// .gitignore excluded — repo-relative: the full first-run
    /// listing. `--cached` and `--others` are disjoint, so no path
    /// repeats.
    pub(crate) fn full_listing(&self) -> Result<Vec<String>, String> {
        let out = self.git(&[
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])?;
        split_nul(&out)
    }

    /// Every path the working tree currently differs on — staged,
    /// unstaged, and untracked, .gitignore excluded. `--no-renames`
    /// keeps every entry a plain `XY PATH` pair (a rename shows as
    /// its delete and add), `-uall` lists untracked files
    /// individually instead of collapsing new directories.
    pub(crate) fn dirty_files(&self) -> Result<Vec<String>, String> {
        let out = self.git(&["status", "--porcelain", "-z", "-uall", "--no-renames"])?;
        Ok(split_nul(&out)?
            .into_iter()
            .filter_map(|entry| entry.get(3..).map(str::to_string))
            .collect())
    }

    /// The work list between `commit` and HEAD. `-M` turns a
    /// delete+add pair back into the rename it was.
    pub(crate) fn changes_since(&self, commit: &str) -> Result<Vec<Change>, String> {
        let out = self.git(&["diff", "--name-status", "-z", "-M", commit, "HEAD"])?;
        let mut fields = split_nul(&out)?.into_iter();
        let mut changes = Vec::new();
        while let Some(status) = fields.next() {
            let path = fields
                .next()
                .ok_or_else(|| format!("git diff: status '{status}' with no path"))?
                .to_string();
            changes.push(match status.as_bytes().first() {
                Some(b'D') => Change::Deleted(path),
                Some(b'A') | Some(b'C') => Change::Added(path),
                Some(b'R') => {
                    let to = fields
                        .next()
                        .ok_or_else(|| format!("git diff: rename of '{path}' with no destination"))?
                        .to_string();
                    Change::Renamed { from: path, to }
                }
                // M, T (type change), and anything a future git adds:
                // re-importing is always the safe reading.
                _ => Change::Modified(path),
            });
        }
        Ok(changes)
    }

    /// Reads files as they are on disk — the same bytes ripgrep (and
    /// the agent's editor) sees, so staged, unstaged, and untracked
    /// content all land in the facts. A path that is missing (a
    /// deletion) or not UTF-8 (a binary) comes back `None`.
    pub(crate) fn read_worktree(&self, paths: &[String]) -> Vec<(String, Option<String>)> {
        paths
            .iter()
            .map(|path| {
                (
                    path.clone(),
                    std::fs::read_to_string(self.root.join(path)).ok(),
                )
            })
            .collect()
    }

    fn git(&self, args: &[&str]) -> Result<Vec<u8>, String> {
        git_in(&self.root, args)
    }
}

fn git_in(dir: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .map_err(|error| format!("running git: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "git {} failed: {}",
            args.first().unwrap_or(&""),
            stderr.trim()
        ));
    }
    Ok(output.stdout)
}

/// NUL-separated git output as owned strings. A non-UTF-8 field is a
/// hard refusal, never a silent drop: `changes_since` pairs status
/// and path fields by position, and a vanished field would shift
/// every later pair — a real file's facts could get retracted off a
/// misread status string.
fn split_nul(bytes: &[u8]) -> Result<Vec<String>, String> {
    bytes
        .split(|byte| *byte == 0)
        .filter(|field| !field.is_empty())
        .map(|field| {
            std::str::from_utf8(field).map(str::to_string).map_err(|_| {
                "git reported a non-UTF-8 path — taguru-code requires UTF-8 paths".to_string()
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A throwaway repo under the platform temp dir, cleaned on drop.
    struct TestRepo {
        dir: PathBuf,
    }

    impl TestRepo {
        fn new(tag: &str) -> TestRepo {
            let dir =
                std::env::temp_dir().join(format!("taguru-code-repo-{tag}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            let repo = TestRepo { dir };
            repo.git(&["init", "-q", "-b", "main"]);
            repo.git(&["config", "user.email", "test@example.com"]);
            repo.git(&["config", "user.name", "test"]);
            repo
        }

        fn git(&self, args: &[&str]) {
            let status = Command::new("git")
                .arg("-C")
                .arg(&self.dir)
                .args(args)
                .output()
                .unwrap();
            assert!(
                status.status.success(),
                "git {args:?}: {}",
                String::from_utf8_lossy(&status.stderr)
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
    }

    impl Drop for TestRepo {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn discover_finds_the_root_from_a_subdirectory_and_refuses_outside() {
        let repo = TestRepo::new("discover");
        repo.write("src/lib.rs", "fn a() {}\n");
        repo.commit("init");
        let from_sub = RepoWalk::discover(&repo.dir.join("src")).unwrap();
        assert_eq!(
            from_sub.root().canonicalize().unwrap(),
            repo.dir.canonicalize().unwrap()
        );

        let outside =
            std::env::temp_dir().join(format!("taguru-code-nonrepo-{}", std::process::id()));
        let _ = fs::remove_dir_all(&outside);
        fs::create_dir_all(&outside).unwrap();
        assert!(RepoWalk::discover(&outside).is_err());
        let _ = fs::remove_dir_all(&outside);
    }

    #[test]
    fn full_listing_is_the_ripgrep_universe_and_head_requires_a_commit() {
        let repo = TestRepo::new("listing");
        repo.write(".gitignore", "target/\n.taguru/\n");
        repo.write("src/lib.rs", "fn a() {}\n");
        repo.write("target/out.rs", "fn generated() {}\n");
        repo.write(".taguru/state", "x");
        let walk = RepoWalk::discover(&repo.dir).unwrap();
        assert!(walk.head().is_err(), "no commits yet must refuse");
        repo.commit("init");
        assert_eq!(walk.head().unwrap().len(), 40);
        // An untracked (but not ignored) file joins the listing; the
        // ignored ones never do — exactly what ripgrep would search.
        repo.write("src/untracked.rs", "fn scratch() {}\n");
        let mut listing = walk.full_listing().unwrap();
        listing.sort();
        assert_eq!(
            listing,
            vec![".gitignore", "src/lib.rs", "src/untracked.rs"]
        );
    }

    #[test]
    fn dirty_files_reports_staged_unstaged_and_untracked_never_ignored() {
        let repo = TestRepo::new("dirty");
        repo.write(".gitignore", "secret.env\n");
        repo.write("src/clean.rs", "fn clean() {}\n");
        repo.write("src/edited.rs", "fn old() {}\n");
        repo.commit("base");
        assert!(walk_of(&repo).dirty_files().unwrap().is_empty());

        repo.write("src/edited.rs", "fn new_version() {}\n"); // unstaged
        repo.write("src/staged.rs", "fn staged() {}\n");
        repo.git(&["add", "src/staged.rs"]); // staged
        repo.write("newdir/untracked.rs", "fn fresh() {}\n"); // untracked, new dir
        repo.write("secret.env", "TOKEN=x\n"); // ignored
        let mut dirty = walk_of(&repo).dirty_files().unwrap();
        dirty.sort();
        assert_eq!(
            dirty,
            vec!["newdir/untracked.rs", "src/edited.rs", "src/staged.rs"]
        );
    }

    fn walk_of(repo: &TestRepo) -> RepoWalk {
        RepoWalk::discover(&repo.dir).unwrap()
    }

    #[test]
    fn read_worktree_serves_disk_bytes_and_none_for_missing() {
        let repo = TestRepo::new("cat");
        repo.write("src/a.rs", "fn committed() {}\n");
        repo.commit("base");
        // The working tree wins over HEAD: the edit is what an agent
        // (and ripgrep) sees, so it is what the facts must carry.
        repo.write("src/a.rs", "fn edited() {}\n");
        let walk = RepoWalk::discover(&repo.dir).unwrap();
        let contents = walk.read_worktree(&["src/a.rs".to_string(), "src/nope.rs".to_string()]);
        assert_eq!(
            contents[0],
            ("src/a.rs".to_string(), Some("fn edited() {}\n".to_string()))
        );
        assert_eq!(contents[1], ("src/nope.rs".to_string(), None));
    }

    #[test]
    fn changes_since_reports_adds_edits_deletes_and_renames() {
        let repo = TestRepo::new("changes");
        repo.write("src/keep.rs", "fn keep() {}\n");
        repo.write("src/gone.rs", "fn gone() {}\n");
        repo.write(
            "src/moved.rs",
            "fn moved() {}\nfn stays_recognizable() {}\n",
        );
        repo.commit("base");
        let walk = RepoWalk::discover(&repo.dir).unwrap();
        let base = walk.head().unwrap();

        repo.write("src/keep.rs", "fn keep() {}\nfn extra() {}\n");
        repo.write("src/new.rs", "fn fresh() {}\n");
        fs::remove_file(repo.dir.join("src/gone.rs")).unwrap();
        fs::rename(
            repo.dir.join("src/moved.rs"),
            repo.dir.join("src/renamed.rs"),
        )
        .unwrap();
        repo.commit("churn");

        let mut changes = walk.changes_since(&base).unwrap();
        changes.sort_by_key(|change| format!("{change:?}"));
        assert_eq!(
            changes,
            vec![
                Change::Added("src/new.rs".into()),
                Change::Deleted("src/gone.rs".into()),
                Change::Modified("src/keep.rs".into()),
                Change::Renamed {
                    from: "src/moved.rs".into(),
                    to: "src/renamed.rs".into(),
                },
            ]
        );
        assert!(
            walk.changes_since(&walk.head().unwrap())
                .unwrap()
                .is_empty(),
            "HEAD to HEAD is an empty work list"
        );
    }

    /// The Err this produces is what sync's full-re-sync fallback
    /// keys on — the anchor commit can genuinely vanish (rebase then
    /// gc), so the refusal is a real code path, not a formality.
    #[test]
    fn changes_since_refuses_a_commit_the_repository_does_not_have() {
        let repo = TestRepo::new("unknown-anchor");
        repo.write("src/a.rs", "fn a() {}\n");
        repo.commit("base");
        let walk = RepoWalk::discover(&repo.dir).unwrap();
        assert!(
            walk.changes_since("0000000000000000000000000000000000000000")
                .is_err()
        );
    }
}
