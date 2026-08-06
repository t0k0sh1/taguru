//! `taguru-code sync`: the write path. The ripgrep universe in, facts
//! out — tracked plus untracked files, .gitignore excluded, read as
//! they are on disk, so staged, unstaged, and untracked code is all
//! findable the moment it is synced. Incrementally a run touches only
//! `git diff <last synced commit>..HEAD` (the committed churn) plus
//! the working tree's dirty set — current AND the one recorded last
//! run, so a revert or a since-committed file heals back to its real
//! content. A candidate that is gone from disk retracts its source
//! (deletions and renames included). Batches go through the same
//! parse/apply the server's own import uses, into a data directory
//! that defaults to `$PROJECT_ROOT/.taguru` — a normal taguru data
//! dir, so `taguru serve` can serve it later unchanged. The one piece
//! of state this verb owns is `code-sync.json` in that directory
//! (last synced commit + last dirty set); a lost, corrupt, or
//! gc-orphaned anchor degrades to a full re-sync, which the
//! retract-then-apply batch contract makes idempotent.

use std::path::{Path, PathBuf};

use crate::code::grammars;
use crate::code::repo_walk::{Change, RepoWalk};
use crate::code::{facts, grammar::Grammar};

/// Context every sync writes into unless `--context` says otherwise.
pub(crate) const DEFAULT_CONTEXT: &str = "code";

/// Description stamped when the context is first created.
const CREATE_DESCRIPTION: &str =
    "Codebase map maintained by taguru-code: symbol locations and structure, synced from git";

/// Mirrors `taguru import`'s flush cadence (`src/ingest/local.rs`).
const FLUSH_EVERY_OPS: usize = 100_000;

/// The state file name inside the data directory.
const STATE_FILE: &str = "code-sync.json";

struct SyncArgs {
    start: PathBuf,
    context: String,
    data_dir: Option<PathBuf>,
    dry_run: bool,
}

fn parse_args(args: &[String]) -> Result<SyncArgs, String> {
    let mut parsed = SyncArgs {
        start: PathBuf::from("."),
        context: DEFAULT_CONTEXT.to_string(),
        data_dir: None,
        dry_run: false,
    };
    let mut rest = args.iter();
    let mut saw_path = false;
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--dry-run" => parsed.dry_run = true,
            "--context" => {
                parsed.context = rest
                    .next()
                    .ok_or_else(|| "--context needs a name".to_string())?
                    .clone();
            }
            "--data-dir" => {
                parsed.data_dir = Some(PathBuf::from(
                    rest.next()
                        .ok_or_else(|| "--data-dir needs a path".to_string())?,
                ));
            }
            other if other.starts_with('-') => {
                return Err(format!("unknown flag '{other}'"));
            }
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

/// The `sync` verb.
pub(crate) fn run(args: &[String]) -> i32 {
    let args = match parse_args(args) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("taguru-code: sync: {message}");
            return 2;
        }
    };
    match sync(&args) {
        Ok(code) => code,
        Err(message) => {
            eprintln!("taguru-code: sync: {message}");
            1
        }
    }
}

/// The `status` verb: what the state file knows, nothing booted.
pub(crate) fn status(args: &[String]) -> i32 {
    let args = match parse_args(args) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("taguru-code: status: {message}");
            return 2;
        }
    };
    let walk = match RepoWalk::discover(&args.start) {
        Ok(walk) => walk,
        Err(message) => {
            eprintln!("taguru-code: status: {message}");
            return 1;
        }
    };
    let data_dir = data_dir_of(&args, &walk);
    println!("repo:     {}", walk.root().display());
    println!("data dir: {}", data_dir.display());
    match load_state(&data_dir) {
        Some(state) => {
            println!("context:  {}", state.context);
            println!("synced:   {}", state.commit);
            let dirty_now: Vec<String> = walk
                .dirty_files()
                .unwrap_or_default()
                .into_iter()
                .filter(|path| !under_data_dir(path, walk.root(), &data_dir))
                .collect();
            let dirty_pending = dirty_now.len()
                + state
                    .dirty
                    .iter()
                    .filter(|path| !dirty_now.contains(path))
                    .count();
            match walk.head() {
                Ok(head) if head == state.commit && dirty_pending == 0 => {
                    println!("up to date with HEAD and a clean working tree")
                }
                Ok(head) => {
                    let committed = if head == state.commit {
                        Ok(Vec::new())
                    } else {
                        walk.changes_since(&state.commit)
                    };
                    match committed {
                        Ok(changes) => println!(
                            "pending: {} committed change(s), {} working-tree file(s) — \
                             run `taguru-code sync`",
                            changes.len(),
                            dirty_pending
                        ),
                        Err(message) => println!("pending changes unknown: {message}"),
                    }
                }
                Err(message) => println!("HEAD unknown: {message}"),
            }
        }
        None => println!("never synced — run `taguru-code sync`"),
    }
    0
}

fn data_dir_of(args: &SyncArgs, walk: &RepoWalk) -> PathBuf {
    resolve_data_dir(args.data_dir.clone(), walk.root())
}

/// The one data-dir resolution `sync`, `status`, and `watch` share.
/// `under_data_dir` compares against absolute worktree paths, so a
/// relative `--data-dir` must be absolutized first or the
/// self-storage exclusion could never match it.
pub(crate) fn resolve_data_dir(data_dir: Option<PathBuf>, root: &Path) -> PathBuf {
    match data_dir {
        None => root.join(".taguru"),
        Some(dir) if dir.is_absolute() => dir,
        Some(dir) => std::env::current_dir()
            .map(|cwd| cwd.join(&dir))
            .unwrap_or(dir),
    }
}

/// Whether a repo-relative path lives inside the data directory. The
/// map must never index its own storage: `.taguru/` is untracked, so
/// without this the dirty set would grow by its own files every run
/// and "up to date" could never hold (gitignoring it is advised, but
/// correctness cannot depend on advice). `watch` needs the same
/// exclusion or its own writes would look like fresh work forever.
pub(crate) fn under_data_dir(path: &str, root: &Path, data_dir: &Path) -> bool {
    root.join(path).starts_with(data_dir)
}

struct SyncState {
    commit: String,
    context: String,
    /// The working tree's dirty set as of that sync — re-examined on
    /// the next run, so a file that got reverted (or committed, or
    /// deleted) since heals to its current truth.
    dirty: Vec<String>,
    /// Content fingerprints (sha256 of the imported bytes) per synced
    /// file: a candidate whose bytes still match skips the
    /// retract-then-reimport entirely. This is what makes repeated
    /// syncs over a dirty-but-stable tree (an editor session, `watch`
    /// polling) nearly free.
    fingerprints: std::collections::BTreeMap<String, String>,
    /// [`facts::FACTS_VERSION`] the fingerprints were recorded under —
    /// a mismatch means the stored facts predate the current fact
    /// model, and every fingerprint is void.
    facts_version: u32,
}

fn load_state(data_dir: &Path) -> Option<SyncState> {
    let raw = std::fs::read_to_string(data_dir.join(STATE_FILE)).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    Some(SyncState {
        commit: value.get("commit")?.as_str()?.to_string(),
        context: value.get("context")?.as_str()?.to_string(),
        dirty: value
            .get("dirty")
            .and_then(|dirty| dirty.as_array())
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(|entry| entry.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default(),
        fingerprints: value
            .get("fingerprints")
            .and_then(|fingerprints| fingerprints.as_object())
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(|(path, fingerprint)| {
                        fingerprint
                            .as_str()
                            .map(|fingerprint| (path.clone(), fingerprint.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default(),
        // Absent in pre-fingerprint state files: 0 never matches, so
        // the one-time cost is a full re-sync, never a stale skip.
        facts_version: value
            .get("facts_version")
            .and_then(|version| version.as_u64())
            .unwrap_or(0) as u32,
    })
}

fn save_state(data_dir: &Path, state: &SyncState) -> Result<(), String> {
    let value = serde_json::json!({
        "commit": state.commit,
        "context": state.context,
        "dirty": state.dirty,
        "fingerprints": state.fingerprints,
        "facts_version": state.facts_version,
    });
    std::fs::write(data_dir.join(STATE_FILE), format!("{value}\n"))
        .map_err(|error| format!("writing {STATE_FILE}: {error}"))
}

fn sync(args: &SyncArgs) -> Result<i32, String> {
    let started = std::time::Instant::now();
    let walk = RepoWalk::discover(&args.start)?;
    let data_dir = data_dir_of(args, &walk);
    let head = walk.head()?;

    let previous = load_state(&data_dir);
    if let Some(previous) = &previous
        && previous.context != args.context
    {
        return Err(format!(
            "this data dir was synced into context '{}'; pass --context {} or a fresh --data-dir",
            previous.context, previous.context
        ));
    }
    // Facts recorded under an older fact model are stale whatever
    // their fingerprints say. The previous state is still KEPT — its
    // commit anchor and dirty set are how deletions since the last
    // sync get retracted, and a rebuild that dropped them would leave
    // the old model's facts of since-deleted files in the map forever.
    // Only the fingerprints are voided (below), and the candidate set
    // widens to the full listing so every survivor re-imports.
    let rebuild = previous
        .as_ref()
        .is_some_and(|state| state.facts_version != facts::FACTS_VERSION);
    if rebuild {
        let recorded = previous
            .as_ref()
            .map(|state| state.facts_version)
            .unwrap_or(0);
        eprintln!(
            "taguru-code: sync: fact model changed (v{recorded} → v{}) — full re-sync",
            facts::FACTS_VERSION
        );
    }
    let first_run = previous.is_none();
    let current_dirty: Vec<String> = walk
        .dirty_files()?
        .into_iter()
        .filter(|path| !under_data_dir(path, walk.root(), &data_dir))
        .collect();
    // Candidates: every path whose truth may have moved since the
    // last sync. What each one becomes — re-import or retraction —
    // is decided by the disk below, not by which list found it.
    let mut candidates: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    match &previous {
        None => candidates.extend(walk.full_listing()?),
        Some(state)
            if !rebuild
                && state.commit == head
                && current_dirty.is_empty()
                && state.dirty.is_empty() =>
        {
            println!("up to date with HEAD ({head}) and a clean working tree");
            return Ok(0);
        }
        Some(state) => {
            if rebuild {
                // Every survivor re-imports under the new model; the
                // diff and dirty sets below still add the deletions.
                candidates.extend(walk.full_listing()?);
            }
            let committed = if state.commit == head {
                Vec::new()
            } else {
                match walk.changes_since(&state.commit) {
                    Ok(changes) => changes,
                    Err(message) => {
                        // The anchor is gone (rebase + gc, a rewritten
                        // history) — degrade to a full re-sync, which
                        // idempotency makes safe, and say so.
                        eprintln!(
                            "taguru-code: sync: {message} — sync anchor unusable, \
                             falling back to a full re-sync"
                        );
                        candidates.extend(walk.full_listing()?);
                        Vec::new()
                    }
                }
            };
            for change in committed {
                match change {
                    Change::Added(path) | Change::Modified(path) | Change::Deleted(path) => {
                        candidates.insert(path);
                    }
                    Change::Renamed { from, to } => {
                        candidates.insert(from);
                        candidates.insert(to);
                    }
                }
            }
            candidates.extend(current_dirty.iter().cloned());
            candidates.extend(state.dirty.iter().cloned());
        }
    }

    // The disk decides each candidate's fate: present and claimed by
    // a grammar → re-import; present but unclaimed → skipped (counted,
    // never silently); gone → retract its source, whatever its
    // extension once was.
    let mut supported = Vec::new();
    let mut retractions = Vec::new();
    let mut skipped = 0usize;
    for path in candidates {
        if under_data_dir(&path, walk.root(), &data_dir) {
            continue; // never index the map's own storage
        }
        if !walk.root().join(&path).exists() {
            retractions.push(path);
            continue;
        }
        let claimed = Path::new(&path)
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .and_then(|extension| grammars::for_extension(&extension));
        match claimed {
            Some(grammar) => supported.push((path, grammar)),
            None => skipped += 1,
        }
    }

    if args.dry_run {
        for (path, _) in &supported {
            println!("would import  {path}");
        }
        for path in &retractions {
            println!("would retract {path}");
        }
        println!(
            "dry run: {} file(s) to import, {} to retract, {} unsupported skipped — nothing applied",
            supported.len(),
            retractions.len(),
            skipped
        );
        return Ok(0);
    }

    // Parse and build every batch before booting the registry: a file
    // that refuses must never cost a directory lock or a write. A
    // candidate whose bytes fingerprint the same as what the map
    // already holds is skipped outright — dirty files re-enter the
    // candidate set every run, and without this each editor-session
    // sync (and every `watch` poll) would re-import them unchanged.
    let mut fingerprints = if rebuild {
        Default::default() // an old model's fingerprints prove nothing
    } else {
        previous
            .as_ref()
            .map(|state| state.fingerprints.clone())
            .unwrap_or_default()
    };
    let paths: Vec<String> = supported.iter().map(|(path, _)| path.clone()).collect();
    let contents = walk.read_worktree(&paths);
    let mut batches = Vec::new();
    let mut warnings = Vec::new();
    let mut binaries = 0usize;
    let mut unchanged = 0usize;
    let mut symbol_count = 0usize;
    for ((path, grammar), (_, content)) in supported.iter().zip(contents) {
        let Some(content) = content else {
            binaries += 1;
            continue;
        };
        let fingerprint = crate::sha256::sha256_hex(content.as_bytes());
        if fingerprints.get(path) == Some(&fingerprint) {
            unchanged += 1;
            continue;
        }
        let symbols = grammar.symbols(&content, path);
        symbol_count += symbols.len();
        let facts = facts::build(path, &symbols);
        warnings.extend(facts.warnings.iter().cloned());
        let rendered = facts::render_batch(&args.context, &facts, Some(CREATE_DESCRIPTION));
        let batch = crate::ingest::parse_batch(rendered.as_bytes())
            .map_err(|message| format!("{path}: rendered batch refused: {message}"))?;
        batches.push((path.clone(), batch, fingerprint));
    }

    // Nothing survived the fingerprints and nothing is gone: no lock,
    // no boot, no write — just move the anchor. This is the branch a
    // `watch` poll over a stable-but-dirty tree lands in.
    if batches.is_empty() && retractions.is_empty() {
        // A first sync of a repo with nothing to import still lands
        // here — the boot below (which normally creates the data dir)
        // never runs, so the state file needs its directory made.
        std::fs::create_dir_all(&data_dir)
            .map_err(|error| format!("creating {}: {error}", data_dir.display()))?;
        save_state(
            &data_dir,
            &SyncState {
                commit: head.clone(),
                context: args.context.clone(),
                dirty: current_dirty,
                fingerprints,
                facts_version: facts::FACTS_VERSION,
            },
        )?;
        println!(
            "nothing to apply ({unchanged} unchanged, {skipped} unsupported) at {head} in {:.1}s",
            started.elapsed().as_secs_f64()
        );
        return Ok(0);
    }

    // Once per process, not per sync: `watch` re-enters this path
    // every time the tree changes, and the tracing subscriber can
    // only be installed once.
    static INIT_LOGGING: std::sync::Once = std::sync::Once::new();
    INIT_LOGGING.call_once(crate::ingest::init_logging);
    let embedder: Option<std::sync::Arc<dyn crate::embedding::EmbeddingProvider>> =
        crate::embedding::HttpEmbeddings::from_env(crate::embedding::ShutdownFlag::default())
            .map(|provider| std::sync::Arc::new(provider) as _);
    let mut config = crate::registry::BootConfig::from_env();
    config.data_dir = data_dir.clone();
    // Per-op WAL durability is for servers that must survive a crash
    // mid-write; this sync is idempotent (the sync point only advances
    // on success, re-importing is the recovery), so the flush-interval
    // window is the right trade — an order of magnitude less fsync.
    config.wal_enabled = false;
    let state = config
        .boot(embedder, None, None, None, None)
        .map_err(|error| error.to_string())?;

    let mut failures = 0usize;
    let mut retracted = 0usize;
    for source in &retractions {
        match state.retract_source(&args.context, source) {
            Ok(_) => {
                retracted += 1;
                fingerprints.remove(source);
            }
            Err(crate::registry::AccessError::NotFound) => {
                // Nothing imported yet under this context — the
                // deletion's truth (source absent) already holds.
                retracted += 1;
                fingerprints.remove(source);
            }
            Err(error) => {
                eprintln!("taguru-code: sync: retracting {source}: {error:?}");
                failures += 1;
            }
        }
    }
    let mut imported = 0usize;
    let mut ops_since_flush = 0usize;
    for (path, batch, fingerprint) in &batches {
        match crate::ingest::apply_batch(&state, batch) {
            Ok(_) => {
                imported += 1;
                // Recorded only on success: a refused batch must stay
                // re-importable, never fingerprint-skipped.
                fingerprints.insert(path.clone(), fingerprint.clone());
            }
            Err(refusal) => {
                eprintln!("taguru-code: sync: {path}: {}", refusal.text());
                failures += 1;
            }
        }
        ops_since_flush += batch.op_count();
        if ops_since_flush >= FLUSH_EVERY_OPS {
            state.flush_dirty();
            ops_since_flush = 0;
        }
    }
    state.flush_dirty();
    state.persist_usage();
    if state.embeddings_configured()
        && (imported > 0 || retracted > 0)
        && let Some(Err(error)) =
            state.refresh_embeddings(&args.context, taguru::deadline::Deadline::unbounded())
    {
        eprintln!("taguru-code: sync: embeddings refresh: {error}");
    }

    for warning in &warnings {
        eprintln!("taguru-code: sync: note: {warning}");
    }
    if failures == 0 {
        save_state(
            &data_dir,
            &SyncState {
                commit: head.clone(),
                context: args.context.clone(),
                dirty: current_dirty,
                fingerprints,
                facts_version: facts::FACTS_VERSION,
            },
        )?;
    } else {
        eprintln!(
            "taguru-code: sync: {failures} failure(s) — sync point not advanced; \
             fix and re-run (re-importing is idempotent)"
        );
    }
    println!(
        "synced {imported} file(s) ({symbol_count} symbols), retracted {retracted}, \
         {unchanged} unchanged, skipped {skipped} unsupported{} at {head} in {:.1}s",
        if binaries > 0 {
            format!(" and {binaries} non-text")
        } else {
            String::new()
        },
        started.elapsed().as_secs_f64()
    );
    if first_run {
        println!(
            "note: add `.taguru/` to your .gitignore — the data directory lives inside the repo"
        );
    }
    Ok(if failures == 0 { 0 } else { 1 })
}
