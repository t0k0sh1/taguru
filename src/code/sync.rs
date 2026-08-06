//! `taguru-code sync`: the write path. Committed state in, facts out —
//! first run imports every tracked file `git ls-files` names, later
//! runs replay only `git diff --name-status <last synced commit>..HEAD`
//! (deletions retract, renames retract-then-import). Batches go through
//! the same parse/apply the server's own import uses, into a data
//! directory that defaults to `$PROJECT_ROOT/.taguru` — a normal taguru
//! data dir, so `taguru serve` can serve it later unchanged. The one
//! piece of state this verb owns is `code-sync.json` in that directory
//! (last synced commit); the registry's boot scan ignores non-`.ctx`
//! files, and a lost or corrupt state file just degrades to a full
//! re-sync, which the retract-then-apply batch contract makes
//! idempotent.

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
            match walk.head() {
                Ok(head) if head == state.commit => println!("up to date with HEAD"),
                Ok(head) => match walk.changes_since(&state.commit) {
                    Ok(changes) => println!(
                        "behind HEAD {head}: {} change(s) pending — run `taguru-code sync`",
                        changes.len()
                    ),
                    Err(message) => println!("pending changes unknown: {message}"),
                },
                Err(message) => println!("HEAD unknown: {message}"),
            }
        }
        None => println!("never synced — run `taguru-code sync`"),
    }
    0
}

fn data_dir_of(args: &SyncArgs, walk: &RepoWalk) -> PathBuf {
    args.data_dir
        .clone()
        .unwrap_or_else(|| walk.root().join(".taguru"))
}

struct SyncState {
    commit: String,
    context: String,
}

fn load_state(data_dir: &Path) -> Option<SyncState> {
    let raw = std::fs::read_to_string(data_dir.join(STATE_FILE)).ok()?;
    let value: serde_json::Value = serde_json::from_str(&raw).ok()?;
    Some(SyncState {
        commit: value.get("commit")?.as_str()?.to_string(),
        context: value.get("context")?.as_str()?.to_string(),
    })
}

fn save_state(data_dir: &Path, state: &SyncState) -> Result<(), String> {
    let value = serde_json::json!({ "commit": state.commit, "context": state.context });
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
    let first_run = previous.is_none();
    let (mut imports, mut retractions) = (Vec::new(), Vec::new());
    match &previous {
        None => imports = walk.full_listing()?,
        Some(state) if state.commit == head => {
            println!("up to date with HEAD ({head})");
            return Ok(0);
        }
        Some(state) => {
            for change in walk.changes_since(&state.commit)? {
                match change {
                    Change::Added(path) | Change::Modified(path) => imports.push(path),
                    Change::Deleted(path) => retractions.push(path),
                    Change::Renamed { from, to } => {
                        retractions.push(from);
                        imports.push(to);
                    }
                }
            }
        }
    }

    // Only files a grammar claims are parsed; the rest are skipped and
    // counted (never silently — the summary names the count).
    let mut supported = Vec::new();
    let mut skipped = 0usize;
    for path in imports {
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
    // A deleted file's source must retract whether or not this build
    // still parses its extension — it was imported when it was
    // supported, or the retraction is a no-op.

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
    // that refuses must never cost a directory lock or a write.
    let paths: Vec<String> = supported.iter().map(|(path, _)| path.clone()).collect();
    let contents = walk.read_at_head(&paths)?;
    let mut batches = Vec::new();
    let mut warnings = Vec::new();
    let mut binaries = 0usize;
    let mut symbol_count = 0usize;
    for ((path, grammar), (_, content)) in supported.iter().zip(contents) {
        let Some(content) = content else {
            binaries += 1;
            continue;
        };
        let symbols = grammar.symbols(&content, path);
        symbol_count += symbols.len();
        let facts = facts::build(path, &symbols);
        warnings.extend(facts.warnings.iter().cloned());
        let rendered = facts::render_batch(&args.context, &facts, Some(CREATE_DESCRIPTION));
        let batch = crate::ingest::parse_batch(rendered.as_bytes())
            .map_err(|message| format!("{path}: rendered batch refused: {message}"))?;
        batches.push((path.clone(), batch));
    }

    crate::ingest::init_logging();
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
            Ok(_) => retracted += 1,
            Err(crate::registry::AccessError::NotFound) => {
                // Nothing imported yet under this context — the
                // deletion's truth (source absent) already holds.
                retracted += 1;
            }
            Err(error) => {
                eprintln!("taguru-code: sync: retracting {source}: {error:?}");
                failures += 1;
            }
        }
    }
    let mut imported = 0usize;
    let mut ops_since_flush = 0usize;
    for (path, batch) in &batches {
        match crate::ingest::apply_batch(&state, batch) {
            Ok(_) => imported += 1,
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
         skipped {skipped} unsupported{} at {head} in {:.1}s",
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
