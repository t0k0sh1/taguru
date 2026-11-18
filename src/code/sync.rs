//! `taguru-code sync`: the write path. The ripgrep universe in, facts
//! out — tracked plus untracked files, .gitignore excluded, read as
//! they are on disk, so staged, unstaged, and untracked code is all
//! findable the moment it is synced. Incrementally a run touches only
//! `git diff <last synced commit>..HEAD` (the committed churn) plus
//! the working tree's dirty set — current AND the one recorded last
//! run, so a revert or a since-committed file heals back to its real
//! content. A candidate that is gone from disk retracts its source
//! (deletions and renames included), and so does a synced file whose
//! bytes can no longer be imported (turned binary, symlink, or
//! non-UTF-8, or past a batch cap — counted and named, never a
//! failure that would freeze the sync point). Batches go through the
//! same parse/apply the server's own import uses, into a data
//! directory that defaults to `$PROJECT_ROOT/.taguru` — a normal
//! taguru data dir, so `taguru serve` can serve it later unchanged.
//! The one piece of state this verb owns is `code-sync.json` in that
//! directory (last synced commit + last dirty set + per-file
//! fingerprints); a lost, corrupt, or gc-orphaned anchor degrades to
//! a full re-sync — the universe plus every path the map still holds
//! facts for, so files deleted meanwhile retract too — which the
//! retract-then-apply batch contract makes idempotent.

use std::path::{Path, PathBuf};

use crate::code::grammars;
use crate::code::repo_walk::{Change, RepoWalk};
use crate::code::{facts, grammar::Grammar};

/// `context` every sync writes into unless `--context` says otherwise.
pub(crate) const DEFAULT_CONTEXT: &str = "code";

/// Description stamped when the `context` is first created.
const CREATE_DESCRIPTION: &str =
    "Codebase map maintained by taguru-code: symbol locations and structure, synced from git";

/// Mirrors `taguru import`'s flush cadence (`src/ingest/local.rs`).
const FLUSH_EVERY_OPS: usize = 100_000;

/// Below this much pending passage log, sync leaves compaction alone
/// even when the log outgrew the snapshot: a small repo's trickle
/// should not rewrite the snapshot every few syncs for kilobytes of
/// reclaim. Deliberately far under the store's own 4 MiB write-path
/// floor — that one is tuned for a long-lived server that compacts on
/// eviction anyway, while this process exits before either trigger
/// can ever run (issue #452).
const PASSAGE_COMPACT_FLOOR_BYTES: u64 = 64 * 1024;

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
    crate::code::usage_log::note_project_root(walk.root());
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
/// `under_data_dir` compares against worktree paths under git's
/// toplevel — which git reports with every symlink resolved
/// (`/private/var/…` on macOS, never `/var/…`) — so the data dir must
/// be put in that same shape or the self-storage exclusion silently
/// misses and `watch` keeps seeing its own writes: a relative
/// `--data-dir` is absolutized, then [`settle`]d.
pub(crate) fn resolve_data_dir(data_dir: Option<PathBuf>, root: &Path) -> PathBuf {
    let absolute = match data_dir {
        None => root.join(".taguru"),
        Some(dir) if dir.is_absolute() => dir,
        Some(dir) => std::env::current_dir()
            .map(|cwd| cwd.join(&dir))
            .unwrap_or(dir),
    };
    settle(&absolute)
}

/// `path` with its ancestors canonicalized — symlinks and `..`/`.`
/// resolved — and its last component kept as named. The leaf is NOT
/// followed on purpose: a `.taguru` that is itself a symlink to
/// another disk shows up in git's listing as `.taguru`, never as the
/// link's target, so only the unresolved leaf can match it. Ancestors
/// that do not exist yet (the first sync creates the directory, and
/// `watch` resolves before that sync runs) are appended lexically
/// below the deepest one that does — with no symlink possible there,
/// lexical `..` is the truth. Unresolvable altogether (no ancestor
/// canonicalizes), the path stays as given.
fn settle(path: &Path) -> PathBuf {
    use std::path::Component;
    let (Some(parent), Some(leaf @ Component::Normal(_))) =
        (path.parent(), path.components().next_back())
    else {
        // A leaf of `..`/`.` or a bare root names no entry of its own:
        // resolving the whole path is the only reading.
        return path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    };
    let mut missing = Vec::new();
    let mut existing = parent;
    let mut settled = loop {
        if let Ok(canonical) = existing.canonicalize() {
            break canonical;
        }
        match (existing.parent(), existing.components().next_back()) {
            (Some(up), Some(component)) => {
                missing.push(component);
                existing = up;
            }
            _ => return path.to_path_buf(),
        }
    };
    for component in missing.into_iter().rev() {
        match component {
            Component::ParentDir => {
                settled.pop();
            }
            Component::CurDir => {}
            other => settled.push(other),
        }
    }
    settled.push(leaf);
    settled
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
    // Atomic: a crash mid-write must leave the previous state, never
    // a torn file — a torn one reads as "no state" and costs the next
    // run a full re-sync plus a whole-map read to recover what the
    // fingerprints would have said (see `previously_synced` in `sync`).
    crate::storage::write_atomic(&data_dir.join(STATE_FILE), format!("{value}\n").as_bytes())
        .map_err(|error| format!("writing {STATE_FILE}: {error}"))
}

fn sync(args: &SyncArgs) -> Result<i32, String> {
    let started = std::time::Instant::now();
    let walk = RepoWalk::discover(&args.start)?;
    crate::code::usage_log::note_project_root(walk.root());
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
    // The current universe — `git ls-files --cached --others
    // --exclude-standard` — read once: it seeds the full-listing
    // candidate paths below AND gates every candidate's fate (ADR
    // 0010 §3: a gitignored file, secrets included, is never in the
    // universe, however it became a candidate).
    let universe: std::collections::BTreeSet<String> = walk.full_listing()?.into_iter().collect();
    // What the map holds facts for right now: every path an earlier
    // sync imported. The state file's fingerprints are that record
    // (inserted on each successful import, dropped on each
    // retraction); without a state file — lost, corrupt — the stored
    // map itself is read back instead, or a full re-sync could never
    // retract the files deleted while the record was missing. Two
    // things need it: a full re-sync must make every such path a
    // candidate (the universe lists only survivors, so a file deleted
    // between the anchor and HEAD is in neither the universe nor any
    // dirty set — only this record still names it), and a candidate
    // that can no longer be imported must retract what an earlier
    // sync imported for it.
    let recovered = match &previous {
        None => crate::code::query::stored_sources(&data_dir, &args.context),
        Some(_) => std::collections::BTreeSet::new(),
    };
    let previously_synced = |path: &str| match &previous {
        Some(state) => state.fingerprints.contains_key(path),
        None => recovered.contains(path),
    };
    // Candidates: every path whose truth may have moved since the
    // last sync. What each one becomes — re-import or retraction —
    // is decided by the disk below, not by which list found it.
    let mut candidates: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    match &previous {
        None => {
            candidates.extend(universe.iter().cloned());
            candidates.extend(recovered.iter().cloned());
        }
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
            // A rebuild re-imports every survivor under the new model;
            // the diff and dirty sets below still add the deletions.
            let mut full = rebuild;
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
                        full = true;
                        Vec::new()
                    }
                }
            };
            if full {
                // Survivors from the universe, deletions from the
                // record: with no usable diff, the synced paths the
                // universe no longer lists are the only trace of what
                // was deleted since the anchor.
                candidates.extend(universe.iter().cloned());
                candidates.extend(state.fingerprints.keys().cloned());
            }
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
    // a grammar → re-import (or, when its bytes cannot be imported,
    // a retraction — below, once they are read); present but
    // unclaimed → skipped (counted, never silently); gone → retract
    // its source, whatever its extension once was.
    let mut supported = Vec::new();
    let mut retractions = Vec::new();
    let mut skipped = 0usize;
    for path in candidates {
        if under_data_dir(&path, walk.root(), &data_dir) {
            continue; // never index the map's own storage
        }
        // Membership first, disk second (issue #733, ADR 0010 §3):
        // a file newly gitignored — `git rm --cached` plus a staged
        // .gitignore, nothing committed — still exists on disk, but
        // it has LEFT the universe, so it must never re-import and
        // whatever an earlier sync imported for it retracts now.
        // "On disk" is the directory entry itself, never what a
        // symlink points at: a link is read (and refused) as what it
        // is below, not judged present or gone by its target.
        if !universe.contains(&path) || std::fs::symlink_metadata(walk.root().join(&path)).is_err()
        {
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
    // The headers name the context by id (#965), and this flow renders
    // them before any server state exists. The id is derived from the
    // `--context` name, so a re-run finds the context the last sync
    // made; a data directory that already holds the name under another
    // id is re-headed onto it right after boot (below).
    let header_context_id = crate::registry::derived_context_id(&args.context);
    let mut warnings = Vec::new();
    let mut binaries = 0usize;
    let mut refused = 0usize;
    let mut unchanged = 0usize;
    let mut symbol_count = 0usize;
    for ((path, grammar), (_, content)) in supported.iter().zip(contents) {
        let Some(content) = content else {
            // Present, but not a plain UTF-8 file inside the repo — a
            // binary, a symlink (never followed, see `open_inside`),
            // a Latin-1 re-save, an unreadable mode. Its current truth
            // is "no facts": a `.rs` that WAS synced and then got
            // swapped for any of these must not keep serving the old
            // version's symbols (and fingerprint) forever. One never
            // synced has nothing to withdraw, so it costs no boot.
            binaries += 1;
            if previously_synced(path.as_str()) {
                retractions.push(path.clone());
            }
            continue;
        };
        let fingerprint = crate::sha256::sha256_hex(content.as_bytes());
        if fingerprints.get(path) == Some(&fingerprint) {
            unchanged += 1;
            continue;
        }
        let symbols = grammar.symbols(&content, path);
        let facts = facts::build(path, &symbols);
        let rendered = facts::render_batch(
            &header_context_id,
            &facts,
            Some(crate::format::HeaderCreate {
                name: &args.context,
                description: CREATE_DESCRIPTION,
            }),
        );
        match crate::ingest::parse_batch(rendered.as_bytes()) {
            Ok(batch) => {
                symbol_count += symbols.len();
                warnings.extend(facts.warnings);
                batches.push((path.clone(), batch, fingerprint));
            }
            Err(message) => {
                // A cap the rendered batch exceeds (an over-long
                // path, a passage past 8 MiB) is a property of this
                // file's bytes, not turbulence: no re-run succeeds
                // until the file changes. Failing the sync on it
                // froze the sync point — every later sync, and
                // `watch`'s first one, refused on this one file while
                // the rest of the repo went stale. So it is counted,
                // named, and skipped like an unsupported file: never
                // fingerprinted (it re-tries whenever it is a
                // candidate again), and whatever an earlier sync
                // imported for it is retracted — the map must not
                // keep answering for a version that no longer exists.
                eprintln!(
                    "taguru-code: sync: {path}: rendered source file refused, skipped: {message}"
                );
                refused += 1;
                if previously_synced(path.as_str()) {
                    retractions.push(path.clone());
                }
            }
        }
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
            "nothing to apply ({unchanged} unchanged, {skipped} unsupported{}{}) at {head} in {:.1}s",
            if binaries > 0 {
                format!(", {binaries} non-text")
            } else {
                String::new()
            },
            if refused > 0 {
                format!(", {refused} refused")
            } else {
                String::new()
            },
            started.elapsed().as_secs_f64()
        );
        return Ok(0);
    }

    // Once per process, not per sync: `watch` re-enters this path
    // every time the tree changes, and the tracing subscriber can
    // only be installed once.
    static INIT_LOGGING: std::sync::Once = std::sync::Once::new();
    INIT_LOGGING.call_once(crate::ingest::init_logging);
    let embedder = crate::embedding::provider_from_env(crate::embedding::ShutdownFlag::default());
    let mut config = crate::registry::BootConfig::from_env();
    config.data_dir = data_dir.clone();
    // Per-op WAL durability is for servers that must survive a crash
    // mid-write; this sync is idempotent (the sync point only advances
    // on success, re-importing is the recovery), so the flush-interval
    // window is the right trade — an order of magnitude less fsync.
    config.wal_enabled = false;
    // Same idempotency argument for the batch-open tear markers
    // (issue #443 item 2): a crash mid-batch is repaired by the re-run
    // the unadvanced sync point already forces, so detection buys
    // nothing here — and the marker's durable open was 2 of the ~3
    // fsyncs each imported file cost, the bulk of a full sync's wall
    // clock.
    config.import_markers_enabled = false;
    let state = config
        .boot(embedder, None, None, None, None)
        .map_err(|error| error.to_string())?;

    let mut failures = 0usize;
    let mut retracted = 0usize;
    // The data paths key on the context's ID (#964); the sync's
    // `--context` stays a display name (this offline flow owns its
    // whole data directory, where that name is unambiguous — it only
    // ever creates one context). `None` simply means nothing imported
    // yet: the import below creates the context by name, and the id
    // resolves again right after.
    let resolve_id = |state: &crate::registry::AppState| state.context_id_of(&args.context);
    let context_id = resolve_id(&state);
    // A data directory that already holds this name under another id
    // (a map written before headers carried ids) keeps the context it
    // has: the batches are re-headed onto it rather than minting a twin
    // under the derived id.
    if let Some(existing) = &context_id {
        for (_, batch, _) in &mut batches {
            batch.context_id.clone_from(existing);
        }
    }
    for source in &retractions {
        let Some(context_id) = context_id.as_deref() else {
            // Nothing imported yet under this context — the
            // deletion's truth (source absent) already holds.
            retracted += 1;
            fingerprints.remove(source);
            continue;
        };
        match state.retract_source(context_id, source) {
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
    let mut partial_writes = false;
    let mut ops_since_flush = 0usize;
    for (path, batch, fingerprint) in &batches {
        match crate::ingest::apply_batch(&state, batch, taguru::deadline::Deadline::unbounded()) {
            Ok(_) => {
                imported += 1;
                // Recorded only on success: a refused batch must stay
                // re-importable, never fingerprint-skipped.
                fingerprints.insert(path.clone(), fingerprint.clone());
            }
            Err(refusal) => {
                eprintln!("taguru-code: sync: {path}: {}", refusal.text());
                // A refusal can land AFTER durable writes (the
                // retraction, the passage, a partial prefix) — the
                // same reason `taguru import` folds
                // `wrote_anything()` into its touched set: those
                // writes still need their vectors below.
                partial_writes |= refusal.wrote_anything();
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
    let wrote_something = imported > 0 || retracted > 0 || partial_writes;
    // Re-resolved: the first import of a fresh map just created the
    // context this run.
    let context_id = resolve_id(&state);
    if state.embeddings_configured()
        && wrote_something
        && let Some(context_id) = context_id.as_deref()
        && let Some(Err(error)) =
            state.refresh_embeddings(context_id, taguru::deadline::Deadline::unbounded())
    {
        eprintln!("taguru-code: sync: embeddings refresh: {error}");
    }
    // Same consent logic as `taguru import` (#479): with
    // TAGURU_EMBED_PASSAGES=1 the passage vectors must follow the
    // glosses, or the vector lane stays silently absent until a
    // server-side refresh this offline flow never performs.
    if state.embeddings_configured()
        && state.passage_embedding_enabled()
        && wrote_something
        && let Some(context_id) = context_id.as_deref()
        && let Some(Err(error)) =
            state.refresh_passage_embeddings(context_id, taguru::deadline::Deadline::unbounded())
    {
        eprintln!("taguru-code: sync: passage embeddings refresh: {error}");
    }
    // Retract-then-reimport appends to the passage log on every sync,
    // and this process exits before the store's own compaction
    // triggers (write-path floor, eviction) can ever run — left
    // alone, the log ratchets toward the server-tuned 4 MiB floor and
    // dwarfs the content it shadows (39% of the data directory on
    // taguru's own map, issue #452). Fold it into the snapshot once
    // it outgrows the snapshot itself; housekeeping, so the sync's
    // outcome above never depends on it.
    state.compact_passages_if_worthwhile(&args.context, PASSAGE_COMPACT_FLOOR_BYTES);

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
         {unchanged} unchanged, skipped {skipped} unsupported{}{} at {head} in {:.1}s",
        if binaries > 0 {
            format!(" and {binaries} non-text")
        } else {
            String::new()
        },
        if refused > 0 {
            format!(" and {refused} refused")
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch tree under the platform temp dir — deliberately the
    /// UNcanonical spelling (`/var/…` on macOS), the shape a user's
    /// `--data-dir` arrives in — removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Scratch {
            let dir =
                std::env::temp_dir().join(format!("taguru-code-sync-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(dir.join("repo/sub")).unwrap();
            Scratch(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The self-exclusion compares against git's toplevel, which git
    /// reports canonical. A data dir spelled through `..`, through a
    /// symlinked ancestor, or not created yet must still exclude the
    /// map's own files — or `watch` re-syncs on its own writes
    /// forever.
    #[test]
    fn the_data_dir_exclusion_survives_dotdot_symlinks_and_absence() {
        let scratch = Scratch::new("settle");
        let repo = scratch.0.join("repo");
        // What `RepoWalk::discover` would hand over: git's toplevel.
        let root = repo.canonicalize().unwrap();
        std::fs::create_dir_all(repo.join(".taguru")).unwrap();

        // Existing, spelled through `..` (and, on macOS, through the
        // `/var` → `/private/var` symlink).
        let dotdot = resolve_data_dir(Some(repo.join("sub/../.taguru")), &root);
        assert!(
            under_data_dir(".taguru/code-sync.json", &root, &dotdot),
            "{} must cover the repo's .taguru",
            dotdot.display()
        );
        // Not created yet — `watch` resolves before its first sync.
        let fresh = resolve_data_dir(Some(repo.join("sub/../later/.map")), &root);
        assert!(under_data_dir("later/.map/x.bin", &root, &fresh));
        assert!(!under_data_dir("later/other.rs", &root, &fresh));
        // The default is unchanged by settling.
        let default = resolve_data_dir(None, &root);
        assert_eq!(default, root.join(".taguru"));

        // A data dir that is itself a symlink shows up in git's
        // listing under the LINK's name; following it would point the
        // exclusion at the target and miss every listed path.
        #[cfg(unix)]
        {
            let target = scratch.0.join("elsewhere");
            std::fs::create_dir_all(&target).unwrap();
            std::os::unix::fs::symlink(&target, repo.join(".linked")).unwrap();
            let linked = resolve_data_dir(Some(repo.join(".linked")), &root);
            assert!(under_data_dir(".linked", &root, &linked));
        }
    }
}
