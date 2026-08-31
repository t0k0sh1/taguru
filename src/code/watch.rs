//! `taguru-code watch`: a foreground loop that keeps the map fresh so
//! neither the agent nor the human ever thinks about syncing. Polling,
//! not filesystem events — the SDK's directory watcher took the same
//! posture: a poll is editor- and OS-agnostic, and the failure mode is
//! a bounded delay, not a missed event.
//!
//! Each tick takes a cheap snapshot (HEAD sha + the dirty set + each
//! dirty file's content hash, the data directory excluded) and syncs
//! only when the snapshot is STABLE — identical on two consecutive
//! polls — and differs from the last synced one. The two-poll rule is
//! the debounce: a save burst keeps moving the snapshot, so the sync
//! lands once the tree settles instead of racing every keystroke. Even
//! a redundant sync is nearly free (the fingerprint skip turns it into
//! hashes and a state-file write), so a wasted tick costs milliseconds,
//! never a re-import.
//!
//! The sync inside the loop is the ordinary one — same locking, same
//! refusals. A transient failure prints and retries next tick; only a
//! failed FIRST sync exits, because a watch that never worked is
//! misconfiguration, not turbulence.

use std::path::PathBuf;

use crate::code::repo_walk::RepoWalk;
use crate::code::sync;
use crate::code::usage_log;

/// Default poll interval.
const DEFAULT_INTERVAL_MS: u64 = 2000;

struct WatchArgs {
    start: PathBuf,
    context: Option<String>,
    data_dir: Option<PathBuf>,
    interval_ms: u64,
}

fn parse_args(args: &[String]) -> Result<WatchArgs, String> {
    let mut parsed = WatchArgs {
        start: PathBuf::from("."),
        context: None,
        data_dir: None,
        interval_ms: DEFAULT_INTERVAL_MS,
    };
    let mut rest = args.iter();
    let mut saw_path = false;
    while let Some(arg) = rest.next() {
        let mut value = |flag: &str| {
            rest.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match arg.as_str() {
            "--context" => parsed.context = Some(value("--context")?),
            "--data-dir" => parsed.data_dir = Some(PathBuf::from(value("--data-dir")?)),
            "--interval-ms" => {
                parsed.interval_ms = value("--interval-ms")?
                    .parse()
                    .ok()
                    .filter(|interval| *interval > 0)
                    .ok_or_else(|| "--interval-ms needs a positive number".to_string())?;
            }
            other if other.starts_with('-') => return Err(format!("unknown flag '{other}'")),
            path => {
                if saw_path {
                    return Err(format!("unexpected second path '{path}'"));
                }
                parsed.start = PathBuf::from(path);
                saw_path = true;
            }
        }
    }
    Ok(parsed)
}

/// The `watch` verb.
pub(crate) fn run(args: &[String]) -> i32 {
    let args = match parse_args(args) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("taguru-code: watch: {message}");
            return 2;
        }
    };
    let walk = match RepoWalk::discover(&args.start) {
        Ok(walk) => walk,
        Err(message) => {
            eprintln!("taguru-code: watch: {message}");
            return 1;
        }
    };
    let data_dir = sync::resolve_data_dir(args.data_dir.clone(), walk.root());

    // The loop re-enters the ordinary sync via its CLI surface — one
    // argument grammar, one behavior, whether a human or this loop
    // drives it.
    let mut sync_argv: Vec<String> = vec![args.start.display().to_string()];
    if let Some(context) = &args.context {
        sync_argv.extend(["--context".to_string(), context.clone()]);
    }
    if let Some(dir) = &args.data_dir {
        sync_argv.extend(["--data-dir".to_string(), dir.display().to_string()]);
    }

    // The snapshot is taken BEFORE the sync it stands for — here and
    // in the loop below alike. Taken after, an edit that lands while
    // the sync runs would be baked into `last_synced` without being
    // in the map, and no later poll would ever see a difference until
    // some unrelated edit came along.
    let pre_sync = snapshot(&walk, &data_dir).unwrap_or_default();

    // A watch that cannot complete its first sync is misconfigured
    // (no commits, locked dir, broken repo) — say so and exit rather
    // than retrying a hopeless loop forever. Both this sync and the
    // loop's re-syncs log as "watch-sync": the watch verb itself never
    // returns, so only its actual syncs carry usage-analysis signal —
    // and a name apart from "sync" keeps manual and watch-driven runs
    // separable in the log.
    let first = usage_log::record("watch-sync", &sync_argv, || sync::run(&sync_argv));
    if first != 0 {
        eprintln!("taguru-code: watch: initial sync failed — not watching");
        return first;
    }

    println!(
        "watching {} every {}ms (Ctrl-C to stop)",
        walk.root().display(),
        args.interval_ms
    );
    let mut last_synced = pre_sync;
    let mut previous_poll = last_synced.clone();
    loop {
        std::thread::sleep(std::time::Duration::from_millis(args.interval_ms));
        let current = match snapshot(&walk, &data_dir) {
            Ok(current) => current,
            Err(message) => {
                // A mid-operation git failure (rebase in progress, a
                // lock) is turbulence: report once, retry next tick.
                eprintln!("taguru-code: watch: {message}");
                continue;
            }
        };
        if current == last_synced {
            previous_poll = current;
            continue;
        }
        if current != previous_poll {
            // Still moving — let the tree settle for one more tick.
            previous_poll = current;
            continue;
        }
        if usage_log::record("watch-sync", &sync_argv, || sync::run(&sync_argv)) == 0 {
            last_synced = current.clone();
        }
        previous_poll = current;
    }
}

/// What "did anything change" means for one tick: HEAD plus every
/// dirty path with its content hash (deleted → a marker), the data
/// directory excluded the same way sync excludes it. Hashing the
/// dirty files (never the whole tree) keeps a quiet tick at a few
/// milliseconds while catching in-place edits that path lists alone
/// would miss.
fn snapshot(walk: &RepoWalk, data_dir: &std::path::Path) -> Result<String, String> {
    let mut parts = vec![walk.head()?];
    let mut dirty: Vec<String> = walk
        .dirty_files()?
        .into_iter()
        .filter(|path| !sync::under_data_dir(path, walk.root(), data_dir))
        .collect();
    dirty.sort();
    for path in dirty {
        // Through the same guard sync reads content with: a path that
        // leaves the repository is not ours to hash, and hashing what
        // it points at would make an outside edit tick the watcher.
        let digest = match walk.inside(&path).map(std::fs::read) {
            Some(Ok(bytes)) => crate::sha256::sha256_hex(&bytes),
            _ => "gone".to_string(),
        };
        parts.push(format!("{path}\u{0}{digest}"));
    }
    Ok(crate::sha256::sha256_hex(parts.join("\u{1}").as_bytes()))
}
