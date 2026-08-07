//! Usage log for `taguru-code`: one JSONL record per invocation,
//! appended under `~/.taguru/logs/usage-YYYYMMDD.jsonl`, so the tool's
//! own efficiency can be studied offline (zero-hit cues, latency and
//! output volume per verb) — the data directory stays per-project,
//! this log is per-user across every repository the binary touches.
//!
//! Strictly best-effort: every failure path (no HOME, unwritable
//! directory, full disk) drops the record and returns — a usage log
//! must never change an answer, an exit code, or a byte of the verbs'
//! own output. For the same reason there is no lock file: a record is
//! serialized to one buffer and appended with a single `O_APPEND`
//! write, which POSIX keeps atomic between concurrent processes (a
//! `find` racing a `watch` sync), so lines never interleave.
//!
//! Knobs (documented in the CLI USAGE, deliberately NOT in
//! `config::KNOWN_KEYS` — that list feeds the SERVER's `--config`
//! typo detection and its help-text test, and these are taguru-code's
//! alone): `TAGURU_USAGE_LOG` (0/false/off disables),
//! `TAGURU_USAGE_LOG_DIR` (default `$HOME/.taguru/logs`),
//! `TAGURU_USAGE_LOG_MAX_BYTES` (total cap across the directory's
//! `usage-*.jsonl`, oldest days deleted first, 0 = uncapped — the
//! `TAGURU_WAL_MAX_BYTES` convention).

use std::cell::RefCell;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub(crate) const DEFAULT_MAX_TOTAL_BYTES: u64 = 50 * 1024 * 1024;

/// Per-invocation extras only the verb itself knows (hit counts,
/// printed bytes, the resolved project root). A STACK, not a slot:
/// `watch` records its re-syncs through [`record`] while a `record`
/// frame of its own may be open, and the inner sync's numbers must
/// not bleed into the outer frame.
#[derive(Default)]
struct Detail {
    hits: Option<u64>,
    output_bytes: Option<u64>,
    project_root: Option<String>,
}

thread_local! {
    static DETAIL_STACK: RefCell<Vec<Detail>> = const { RefCell::new(Vec::new()) };
}

fn note(update: impl FnOnce(&mut Detail)) {
    DETAIL_STACK.with(|stack| {
        if let Some(top) = stack.borrow_mut().last_mut() {
            update(top);
        }
    });
}

/// Search hits for `find` / entries listed for `tree` — 0 is worth
/// recording (a zero-hit cue is exactly the improvement signal this
/// log exists for).
pub(crate) fn note_hits(count: usize) {
    note(|detail| detail.hits = Some(count as u64));
}

pub(crate) fn note_output_bytes(bytes: usize) {
    note(|detail| detail.output_bytes = Some(bytes as u64));
}

pub(crate) fn note_project_root(root: &Path) {
    note(|detail| detail.project_root = Some(root.display().to_string()));
}

/// Runs one verb and appends its usage record. The verb's exit code
/// passes through untouched whether the append worked or not.
pub(crate) fn record(command: &str, args: &[String], verb: impl FnOnce() -> i32) -> i32 {
    DETAIL_STACK.with(|stack| stack.borrow_mut().push(Detail::default()));
    let started = Instant::now();
    let exit_code = verb();
    let duration_ms = started.elapsed().as_millis() as u64;
    let detail = DETAIL_STACK
        .with(|stack| stack.borrow_mut().pop())
        .unwrap_or_default();
    append_best_effort(command, args, duration_ms, exit_code, detail);
    exit_code
}

/// The disable dialect: `0`/`false` (the TAGURU_ boolean convention)
/// plus `off` (what a shell-history reader expects a log switch to
/// take), all case-insensitive. Anything else — including typos —
/// leaves logging ON: this knob guards an append to the user's own
/// disk, and `env_bool`'s warn-on-typo contract has nowhere to land
/// here (taguru-code installs no tracing subscriber).
fn enabled(value: Option<&str>) -> bool {
    match value {
        Some(value) => {
            !(value == "0"
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("off"))
        }
        None => true,
    }
}

#[cfg(unix)]
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|home| !home.is_empty())
        .map(PathBuf::from)
}

#[cfg(not(unix))]
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .filter(|home| !home.is_empty())
        .map(PathBuf::from)
}

fn log_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("TAGURU_USAGE_LOG_DIR").filter(|dir| !dir.is_empty()) {
        return Some(PathBuf::from(dir));
    }
    Some(home_dir()?.join(".taguru").join("logs"))
}

/// `"2026-08-07T09:14:22Z"` → `"20260807"`. Derived from the ISO
/// string (not a second clock read) so the stamp and the record's
/// `ts` can never name different days.
fn day_stamp(iso: &str) -> String {
    iso.get(0..10).unwrap_or("1970-01-01").replace('-', "")
}

fn append_best_effort(
    command: &str,
    args: &[String],
    duration_ms: u64,
    exit_code: i32,
    detail: Detail,
) {
    if !enabled(std::env::var("TAGURU_USAGE_LOG").ok().as_deref()) {
        return;
    }
    let Some(dir) = log_dir() else { return };
    let ts = crate::clock::iso8601_utc(crate::clock::now_unix_secs());
    let today = day_stamp(&ts);
    let record = serde_json::json!({
        "ts": ts,
        "command": command,
        "args": args,
        "duration_ms": duration_ms,
        "exit_code": exit_code,
        "hits": detail.hits,
        "output_bytes": detail.output_bytes,
        "project_root": detail.project_root.unwrap_or_else(|| {
            std::env::current_dir()
                .map(|cwd| cwd.display().to_string())
                .unwrap_or_default()
        }),
    });
    let Ok(mut line) = serde_json::to_vec(&record) else {
        return;
    };
    line.push(b'\n');
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(format!("usage-{today}.jsonl")))
    else {
        return;
    };
    if file.write_all(&line).is_err() {
        return;
    }
    let cap = crate::env::env_number(
        "TAGURU_USAGE_LOG_MAX_BYTES",
        DEFAULT_MAX_TOTAL_BYTES as usize,
    ) as u64;
    enforce_cap(&dir, &today, cap);
}

/// Deletes oldest-day files until the directory's `usage-*.jsonl`
/// total fits under `cap`. Today's file is never deleted, even when
/// it alone exceeds the cap — the record just written must survive
/// its own enforcement pass. Lockless on purpose: two processes
/// racing this can at worst both delete (the second `remove_file`
/// fails and is ignored) or delete one file more than strictly
/// needed — acceptable for telemetry, where blocking a verb is not.
fn enforce_cap(dir: &Path, today: &str, cap: u64) {
    if cap == 0 {
        return; // 0 = uncapped, the TAGURU_WAL_MAX_BYTES convention
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(PathBuf, u64, String)> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name().to_str()?.to_string();
            let day = name.strip_prefix("usage-")?.strip_suffix(".jsonl")?;
            Some((entry.path(), entry.metadata().ok()?.len(), day.to_string()))
        })
        .collect();
    // YYYYMMDD sorts chronologically as a string: oldest first.
    files.sort_by(|a, b| a.2.cmp(&b.2));
    let mut total: u64 = files.iter().map(|(_, size, _)| size).sum();
    for (path, size, day) in &files {
        if total <= cap {
            break;
        }
        if day == today {
            continue;
        }
        if std::fs::remove_file(path).is_ok() || !path.exists() {
            total = total.saturating_sub(*size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disable_dialect_accepts_off_and_the_boolean_convention() {
        assert!(enabled(None));
        assert!(enabled(Some("1")));
        assert!(enabled(Some("true")));
        assert!(enabled(Some("banana"))); // typos leave logging on
        assert!(!enabled(Some("0")));
        assert!(!enabled(Some("false")));
        assert!(!enabled(Some("FALSE")));
        assert!(!enabled(Some("off")));
        assert!(!enabled(Some("OFF")));
    }

    #[test]
    fn day_stamp_strips_the_date_from_an_iso_timestamp() {
        assert_eq!(day_stamp("2026-08-07T09:14:22Z"), "20260807");
        assert_eq!(day_stamp("1970-01-01T00:00:00Z"), "19700101");
        assert_eq!(day_stamp("bogus"), "19700101"); // malformed → epoch, never a panic
    }

    /// A throwaway directory per test, unique by counter like the e2e
    /// suite's `Repo` (timestamps collide within clock resolution).
    fn scratch_dir() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "taguru-usage-log-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn plant(dir: &Path, day: &str, bytes: usize) {
        std::fs::write(dir.join(format!("usage-{day}.jsonl")), vec![b'x'; bytes]).unwrap();
    }

    #[test]
    fn enforce_cap_deletes_oldest_days_first() {
        let dir = scratch_dir();
        plant(&dir, "20260101", 100);
        plant(&dir, "20260102", 100);
        plant(&dir, "20260103", 100);
        enforce_cap(&dir, "20260103", 250);
        assert!(!dir.join("usage-20260101.jsonl").exists(), "oldest goes");
        assert!(dir.join("usage-20260102.jsonl").exists());
        assert!(dir.join("usage-20260103.jsonl").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn enforce_cap_never_deletes_todays_file() {
        let dir = scratch_dir();
        plant(&dir, "20260103", 1000);
        enforce_cap(&dir, "20260103", 10);
        assert!(
            dir.join("usage-20260103.jsonl").exists(),
            "today survives even alone over the cap"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn enforce_cap_zero_is_uncapped_and_strangers_are_ignored() {
        let dir = scratch_dir();
        plant(&dir, "20260101", 1000);
        std::fs::write(dir.join("notes.txt"), vec![b'x'; 5000]).unwrap();
        enforce_cap(&dir, "20260102", 0);
        assert!(dir.join("usage-20260101.jsonl").exists(), "0 = uncapped");
        enforce_cap(&dir, "20260102", 100);
        assert!(!dir.join("usage-20260101.jsonl").exists());
        assert!(
            dir.join("notes.txt").exists(),
            "only usage-*.jsonl is counted or deleted"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
