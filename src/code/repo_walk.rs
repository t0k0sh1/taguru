//! The git boundary: every fact about which files exist, changed, or
//! vanished comes from `git` subprocess calls, never from a
//! hand-rolled directory walk. `git ls-files` IS the .gitignore
//! authority (tracked, non-ignored, committed-or-staged); `git diff
//! --name-status` between the last synced commit and HEAD IS the
//! incremental work list, deletions and renames included. A directory
//! that is not a git work tree is a refusal, not a fallback — a
//! bespoke walker would re-implement ignore semantics wrong.

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

    /// The commit the work list is read at: HEAD's sha. A repository
    /// with no commits yet refuses here — the sync contract is
    /// committed state only.
    pub(crate) fn head(&self) -> Result<String, String> {
        let out = self.git(&["rev-parse", "HEAD"]).map_err(|error| {
            format!("{error} — taguru-code syncs committed state; commit first")
        })?;
        Ok(String::from_utf8_lossy(&out).trim().to_string())
    }

    /// Every tracked file, repo-relative, NUL-separated — the full
    /// first-run listing.
    pub(crate) fn full_listing(&self) -> Result<Vec<String>, String> {
        let out = self.git(&["ls-files", "-z"])?;
        Ok(split_nul(&out).map(str::to_string).collect())
    }

    /// The work list between `commit` and HEAD. `-M` turns a
    /// delete+add pair back into the rename it was.
    pub(crate) fn changes_since(&self, commit: &str) -> Result<Vec<Change>, String> {
        let out = self.git(&["diff", "--name-status", "-z", "-M", commit, "HEAD"])?;
        let mut fields = split_nul(&out);
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

    /// Reads files as committed at HEAD — never the working tree, so
    /// a dirty checkout cannot leak uncommitted lines into the facts.
    /// One `git cat-file --batch` subprocess serves every path;
    /// requests stream from a writer thread so a large repo cannot
    /// deadlock both pipes. A path that is missing at HEAD or not
    /// UTF-8 (a binary) comes back `None`.
    pub(crate) fn read_at_head(
        &self,
        paths: &[String],
    ) -> Result<Vec<(String, Option<String>)>, String> {
        use std::io::{BufRead, BufReader, Read, Write};

        if paths.is_empty() {
            return Ok(Vec::new());
        }
        let mut child = Command::new("git")
            .arg("-C")
            .arg(&self.root)
            .args(["cat-file", "--batch"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|error| format!("running git cat-file: {error}"))?;

        let mut stdin = child.stdin.take().expect("piped stdin");
        let requests: Vec<String> = paths.iter().map(|path| format!("HEAD:{path}\n")).collect();
        let writer = std::thread::spawn(move || {
            for request in requests {
                if stdin.write_all(request.as_bytes()).is_err() {
                    break;
                }
            }
            // Dropping stdin closes the pipe; cat-file exits after
            // answering what it was asked.
        });

        let mut reader = BufReader::new(child.stdout.take().expect("piped stdout"));
        let mut contents = Vec::with_capacity(paths.len());
        for path in paths {
            let mut header = String::new();
            if reader
                .read_line(&mut header)
                .map_err(|error| format!("git cat-file: {error}"))?
                == 0
            {
                return Err("git cat-file: output ended early".to_string());
            }
            let fields: Vec<&str> = header.trim_end().split(' ').collect();
            match fields.as_slice() {
                [_, _, size] => {
                    let size: usize = size
                        .parse()
                        .map_err(|_| format!("git cat-file: bad size in '{header}'"))?;
                    let mut blob = vec![0u8; size + 1]; // content + trailing \n
                    reader
                        .read_exact(&mut blob)
                        .map_err(|error| format!("git cat-file: {error}"))?;
                    blob.pop();
                    contents.push((path.clone(), String::from_utf8(blob).ok()));
                }
                _ => contents.push((path.clone(), None)), // "<spec> missing"
            }
        }
        let _ = writer.join();
        let _ = child.wait();
        Ok(contents)
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

fn split_nul(bytes: &[u8]) -> impl Iterator<Item = &str> {
    bytes
        .split(|byte| *byte == 0)
        .filter(|field| !field.is_empty())
        .map(|field| std::str::from_utf8(field).unwrap_or(""))
        .filter(|field| !field.is_empty())
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
    fn full_listing_respects_gitignore_and_head_requires_a_commit() {
        let repo = TestRepo::new("listing");
        repo.write(".gitignore", "target/\n.taguru/\n");
        repo.write("src/lib.rs", "fn a() {}\n");
        repo.write("target/out.rs", "fn generated() {}\n");
        repo.write(".taguru/state", "x");
        let walk = RepoWalk::discover(&repo.dir).unwrap();
        assert!(walk.head().is_err(), "no commits yet must refuse");
        repo.commit("init");
        assert_eq!(walk.head().unwrap().len(), 40);
        let listing = walk.full_listing().unwrap();
        assert_eq!(listing, vec![".gitignore", "src/lib.rs"]);
    }

    #[test]
    fn read_at_head_serves_committed_bytes_not_the_working_tree() {
        let repo = TestRepo::new("cat");
        repo.write("src/a.rs", "fn committed() {}\n");
        repo.commit("base");
        // Dirty the working tree AFTER the commit: sync must not see it.
        repo.write("src/a.rs", "fn uncommitted_edit() {}\n");
        let walk = RepoWalk::discover(&repo.dir).unwrap();
        let contents = walk
            .read_at_head(&["src/a.rs".to_string(), "src/nope.rs".to_string()])
            .unwrap();
        assert_eq!(
            contents[0],
            (
                "src/a.rs".to_string(),
                Some("fn committed() {}\n".to_string())
            )
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
}
