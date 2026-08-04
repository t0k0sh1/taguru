//! `taguru import`: offline batch ingestion — the bulk/initial-load
//! path that the REST API is the wrong tool for. A batch file is JSON
//! Lines: one header naming the context and the source, then
//! association / alias / passage lines (the same shapes the HTTP
//! endpoints accept, minus per-line sources — the header's source is
//! stamped on every line, see below).
//!
//! One file states one source's COMPLETE truth: applying it first
//! retracts the source, then adds the file's facts, so re-importing a
//! file is idempotent and importing a revised one is the same
//! differential sync agents do live (`retract_source` → re-ingest).
//! That contract is why association lines may not carry their own
//! source: a source the header does not name would survive the
//! retraction and double on every re-import.
//!
//! Validation is a separate pass: every file parses completely (and
//! the set of files is checked for two files claiming one source)
//! before anything applies. Apply-stage failures cannot all be
//! pre-checked (capacity, disk); those are reported per file and the
//! remaining files still run — every file is one source, independent
//! by construction, and a partially applied one heals on re-import.
//! Until that re-import (or a retraction), the batch-open marker
//! written around every apply keeps the tear visible: boot and
//! `taguru inspect` name the source, however the batch stopped short
//! (see [`apply_batch`]).
//!
//! The writes go through the same registry every server write goes
//! through — WAL-staged, budget-enforced, flushed — and the data
//! directory lock makes the server/import conflict a refusal instead
//! of a corruption.
//!
//! The same contract is served live as `POST /import` (one request =
//! one batch file), so bulk loads reach a running server without a
//! downtime window; [`parse_batch`] and [`apply_batch`] are that
//! endpoint's core too, which is what keeps the two entrances from
//! drifting apart.
//!
//! Beside batches, a stream may carry GROUP records: one
//! `taguru_group` line states one group's complete truth (name,
//! description, member contexts, child groups) the way one batch
//! states one source's. Applying one is a create-or-replace of the
//! whole record — never a delta — so re-importing stays idempotent.
//! Groups apply AFTER every batch of the run (one CLI invocation, one
//! `POST /import` body), whatever file or position carried them, so a
//! group and the member contexts it names can travel together in any
//! order; a member that still does not exist at that point refuses
//! the whole group set, with every batch already durably landed.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::fs;
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;
use taguru::context::{AliasError, Context};
use taguru::deadline::Deadline;

use crate::api::{
    MAX_ASSOCIATION_WEIGHT, MAX_ASSOCIATIONS_PER_REQUEST, MAX_CONTEXT_NAME_BYTES,
    MAX_DESCRIPTION_BYTES, MAX_NAME_BYTES,
};
use crate::env::DEFAULT_MAX_BODY_BYTES;
use crate::groups::{GroupRecord, MAX_GROUP_MEMBERS};
use crate::registry::{AccessError, AppState, AssocOp, ContextMeta, CreateError};
use crate::remote::{Api, ImportFailure};
use crate::schema;

const USAGE: &str = "\
usage: taguru import [--dry-run] [--no-embed] [--json] [--config FILE]
                      [--url URL] FILE|DIR...

Applies JSONL batch files to TAGURU_DATA_DIR offline (the server must
not be running — the directory lock enforces it), or to a RUNNING
server with --url. One batch = one source's complete truth: import
retracts the source, then applies the batch, so re-importing is
idempotent. A file carries one batch or a whole stream of them (each
`taguru_batch` header line starts the next) — `taguru export` writes
such streams. A `taguru_schema` line states one context's whole
schema document (ADR 0009 §13); it installs AFTER every batch, BEFORE
any group, so a schema record can name a context a batch of the same
stream just created. A `taguru_group` line states one group's complete
truth the same way; groups restore AFTER every batch and schema of
the run (create-or-replace of the whole record), so group files
re-apply in any order. A directory expands to its *.jsonl files,
sorted by name. Format: docs/import.html.

  --dry-run    validate every file and report; touch nothing (with
               --url, POSTs every chunk as ?dry_run=true instead)
  --no-embed   skip the embedding refresh TAGURU_EMBED_URL would enable
               (offline only — combined with --url this is a usage
               error, since the server's own configuration decides
               once the request lands there)
  --config F   read KEY=VALUE environment from F (same dialect as serve)

  --url URL    import into a RUNNING server instead of TAGURU_DATA_DIR
               directly: POST /import, one request per chunk. The
               input is split on batch boundaries only — never
               mid-batch — into chunks under the server's body cap
               (TAGURU_MAX_BODY_BYTES, 8 MiB by default), starting at
               a 4 MiB budget and halved (never crossing a batch
               boundary) and resent on a 413. A single batch that
               alone exceeds the cap is a hard error naming the
               source: raise TAGURU_MAX_BODY_BYTES on the server, or
               split that source's content upstream of import. A lost
               connection names which chunk landed and points at
               --dry-run to confirm before resuming — nothing past
               that is retried automatically (import's
               retract-then-apply contract makes any resend exact).
               Auth rides TAGURU_API_TOKEN (or the first name:token
               entry of TAGURU_API_TOKENS), the admin role the server
               requires. In CI, name the target per invocation:
                 taguru import --url \"$TAGURU_URL\" backups/
  --json       one JSON document instead of per-file lines: {dry_run,
               batches: [...], schemas: [...], groups: [...]} — the
               same shape POST /import answers with (schemas/groups
               omitted when empty). With --url, every field is exact
               (the server previews with the same code path a real
               apply runs). Offline, a real (non-dry-run) run is exact
               the same way; offline --dry-run cannot open the data
               directory without the lock a running import would need,
               so its batch counts are read straight from each file
               (created/retracted and the *_dropped fields all report
               0/false) and its schemas/groups arrays are always empty
               — a preview, not the server's exact one. Every exit path
               prints exactly one document, failures included: validation
               refusing every file, the
               registry refusing to boot, and a remote transport error
               or a server-refused chunk each add a top-level 'error'
               string beside whatever batches/groups already landed; a
               batch refused mid-run (offline only) is named under a
               failed_batches array instead of batches, since there is
               no successful outcome to report for it.
";

/// The one format version this build reads and docs/import.html
/// describes. `pub(crate)` so `GET /version` (ADR 0005 §3, §6) can
/// report it under `batch_formats`.
pub(crate) const BATCH_VERSION: u64 = 1;

/// The `taguru_group` record's own version stamp — separate from
/// [`BATCH_VERSION`] so either shape can rev without dragging the
/// other along. Export serializes it; parse refuses any other value.
pub(crate) const GROUP_VERSION: u64 = 1;

/// Per-line byte cap. Lines are one fact or one passage; past this
/// something is wrong with the producer, and refusing early beats
/// buffering a runaway line.
const MAX_LINE_BYTES: usize = 16 * 1024 * 1024;

/// Passage cap, mirroring the HTTP default: over the API a passage
/// rides under `TAGURU_MAX_BODY_BYTES` (8 MiB), and a file must not
/// smuggle in what a request could not. Extract caps whole documents
/// here too — a document over it could not ride as a passage.
pub(crate) const MAX_PASSAGE_BYTES: usize = 8 * 1024 * 1024;

/// Ops applied between mid-run flushes. Import batches can dwarf any
/// live traffic; flushing every so often keeps each context's WAL far
/// from `TAGURU_WAL_MAX_BYTES` (past which writes are refused).
const FLUSH_EVERY_OPS: usize = 100_000;

/// The starting byte budget `import --url` packs each chunk up to
/// (ADR 0002 §9) — half the server's default body cap, leaving
/// headroom for a server configured lower. A 413 still hit at this
/// budget halves it further (never below one unit) rather than
/// retrying at the same size.
const REMOTE_IMPORT_BUDGET_BYTES: usize = DEFAULT_MAX_BODY_BYTES / 2;

pub fn run(args: &[String]) -> i32 {
    let mut dry_run = false;
    let mut no_embed = false;
    let mut as_json = false;
    let mut config: Option<PathBuf> = None;
    let mut url: Option<String> = None;
    let mut paths: Vec<String> = Vec::new();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{USAGE}");
                return 0;
            }
            "--dry-run" => dry_run = true,
            "--no-embed" => no_embed = true,
            "--json" => as_json = true,
            "--config" => match rest.next() {
                Some(path) => config = Some(PathBuf::from(path)),
                None => {
                    return crate::config::subcommand_usage_error(
                        "import",
                        "--config needs a file path",
                    );
                }
            },
            // Trailing '/' trimmed the same way export/compact/calibrate/
            // communities already do — a joined path segment must not
            // double up.
            "--url" => match rest.next() {
                Some(value) => url = Some(value.trim_end_matches('/').to_string()),
                None => {
                    return crate::config::subcommand_usage_error(
                        "import",
                        "--url needs a server URL",
                    );
                }
            },
            other if other.starts_with('-') => {
                return crate::config::subcommand_usage_error(
                    "import",
                    &format!("unknown flag '{other}'"),
                );
            }
            path => paths.push(path.to_string()),
        }
    }
    if paths.is_empty() {
        eprint!("{USAGE}");
        return 2;
    }
    // ADR 0002 §5: a flag that only makes sense offline, combined with
    // --url, is a usage error rather than a silent no-op — the
    // server's own embedding configuration decides once the request
    // lands there, so --no-embed has nothing left to control remotely.
    if no_embed && url.is_some() {
        return crate::config::subcommand_usage_error(
            "import",
            "--no-embed cannot be combined with --url — the server's own embedding \
             configuration decides",
        );
    }
    // TAGURU_CONFIG fallback (issue #248 item 2): --config wins, but a
    // deployment file baked in via the environment still applies when
    // it's absent — the same priority serve/health/calibrate/
    // communities/evaluate/restore already give it.
    let config = config.or_else(|| std::env::var("TAGURU_CONFIG").ok().map(PathBuf::from));
    // SAFETY (same contract as serve): applied while the process is
    // still single-threaded — import never starts a runtime at all.
    // Loaded before the --url dispatch below, too: a config file is
    // the usual place a deployment's TAGURU_API_TOKEN lives, and
    // Api::new reads the bearer from the environment at construction.
    if let Some(path) = &config {
        crate::config::load_config(path);
    }

    let files = match expand(&paths) {
        Ok(files) => files,
        Err(message) => return crate::config::subcommand_usage_error("import", &message),
    };

    match url {
        // ADR 0002 §5/§6: `--url` is the only way `import` goes
        // remote — no positional URL argument, no TAGURU_URL or
        // default_base_url() fallback the way `health`/`calibrate`/
        // `communities` have one. Absent, this is exactly the local
        // path that ran before this flag existed.
        Some(base) => run_remote(&base, &files, dry_run, as_json),
        None => run_local(&files, dry_run, no_embed, as_json),
    }
}

fn run_local(files: &[PathBuf], dry_run: bool, no_embed: bool, as_json: bool) -> i32 {
    // Pass 1 — every file parses, or nothing applies. Apply-stage
    // refusals can strand a half-written source; a malformed line is
    // knowable up front, so it must never cost a write. A file may
    // carry one batch or a whole stream (`taguru export` output);
    // either way each batch stands alone from here on.
    let mut batches = Vec::new();
    let mut schemas: Vec<(&PathBuf, String, schema::InstalledSchema)> = Vec::new();
    let mut groups: Vec<(&PathBuf, String, GroupRecord)> = Vec::new();
    let mut broken = 0;
    let mut owners: HashSet<(String, String)> = HashSet::new();
    let mut schema_owners: HashSet<String> = HashSet::new();
    let mut group_owners: HashSet<String> = HashSet::new();
    for path in files {
        let parsed = fs::File::open(path)
            .map_err(|error| error.to_string())
            .and_then(|file| parse_stream(std::io::BufReader::new(file)));
        // A stream file can carry several batches, schemas, or groups,
        // so it can trip several of the checks below; `broken` counts
        // files, not events, so one file's several conflicts must
        // still add only 1.
        let mut file_broken = false;
        match parsed {
            Ok(stream) => {
                for batch in stream.batches {
                    if !owners.insert((batch.context.clone(), batch.source.clone())) {
                        eprintln!(
                            "taguru: import: {}: {}",
                            path.display(),
                            duplicate_source_message(&batch.context, &batch.source)
                        );
                        file_broken = true;
                        continue;
                    }
                    batches.push((path, batch));
                }
                for (context, installed) in stream.schemas {
                    if !schema_owners.insert(context.clone()) {
                        eprintln!(
                            "taguru: import: {}: {}",
                            path.display(),
                            duplicate_schema_message(&context)
                        );
                        file_broken = true;
                        continue;
                    }
                    schemas.push((path, context, installed));
                }
                for (name, record) in stream.groups {
                    if !group_owners.insert(name.clone()) {
                        eprintln!(
                            "taguru: import: {}: {}",
                            path.display(),
                            duplicate_group_message(&name)
                        );
                        file_broken = true;
                        continue;
                    }
                    groups.push((path, name, record));
                }
            }
            Err(message) => {
                eprintln!("taguru: import: {}: {message}", path.display());
                file_broken = true;
            }
        }
        if file_broken {
            broken += 1;
        }
    }
    if broken > 0 {
        let message = format!(
            "{broken} of {} file(s) refused during validation; nothing was applied",
            files.len()
        );
        eprintln!("taguru: import: {message}");
        if as_json {
            print_import_json(
                dry_run,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Some(message),
            );
        }
        return 1;
    }

    if dry_run {
        if as_json {
            // Offline never boots the registry for a dry run (that's
            // the whole point — a validate-only pass needs no
            // directory lock), so nothing here can know created/
            // retracted or any of the *_dropped counts the way
            // preview_batch (the server's own dry-run path) can.
            // Reported as 0/false rather than guessed — see the
            // --help text. Groups are always empty here for the same
            // reason POST /import's own dry run omits them: restoring
            // a group has no read-only twin to preview through.
            let batches: Vec<crate::api::ImportOutcome> = batches
                .iter()
                .map(|(_, batch)| dry_run_outcome_of(batch))
                .collect();
            print_import_json(true, batches, Vec::new(), Vec::new(), Vec::new(), None);
            return 0;
        }
        for (path, batch) in &batches {
            println!("{}: {}", path.display(), batch.describe());
        }
        // Schema records, like group records, have no read-only twin
        // to preview through (installing one can depend on a batch of
        // this same run having just created its context) — described
        // structurally from the parse alone, same as `describe_group`.
        for (path, context, installed) in &schemas {
            println!(
                "{}: {}",
                path.display(),
                describe_schema(context, installed)
            );
        }
        for (path, name, record) in &groups {
            println!("{}: {}", path.display(), describe_group(name, record));
        }
        let mut summary = format!("dry run: {} batch(es)", batches.len());
        if !schemas.is_empty() {
            summary.push_str(&format!(", {} schema record(s)", schemas.len()));
        }
        if !groups.is_empty() {
            summary.push_str(&format!(" and {} group record(s)", groups.len()));
        }
        summary.push_str(" valid, nothing applied");
        println!("{summary}");
        return 0;
    }

    // Registry warnings (WAL replay notes, load errors) must reach the
    // operator; stdout stays reserved for the report lines.
    init_logging();
    let embedder: Option<std::sync::Arc<dyn crate::embedding::EmbeddingProvider>> = if no_embed {
        None
    } else {
        // A fresh, never-raised flag: the import runs one command to
        // completion, with no graceful drain to unblock.
        crate::embedding::HttpEmbeddings::from_env(crate::embedding::ShutdownFlag::default())
            .map(|provider| std::sync::Arc::new(provider) as _)
    };
    // The same knobs serve boots with — one reading for both entrances
    // (cli.rs documents them once).
    let state = match crate::registry::BootConfig::from_env().boot(embedder, None, None, None, None)
    {
        Ok(state) => state,
        Err(error) => {
            eprintln!("taguru: import: {error}");
            // ADR 0002 §5: the directory-lock refusal gains one added
            // sentence pointing at the way out — importing into a
            // RUNNING server is a different command, not a second
            // offline process racing the first for the same lock.
            if error.to_string().contains("held by another taguru process") {
                eprintln!(
                    "taguru: import: importing into a *running* server is `taguru import \
                     --url http://127.0.0.1:8248 FILE...`, not a second offline process \
                     racing the first"
                );
            }
            if as_json {
                print_import_json(
                    dry_run,
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Some(error.to_string()),
                );
            }
            return 1;
        }
    };

    // Pass 2 — apply, one file at a time, in the order given.
    let mut failures = 0;
    let mut touched: BTreeSet<String> = BTreeSet::new();
    let mut ops_since_flush = 0usize;
    let mut json_batches: Vec<crate::api::ImportOutcome> = Vec::new();
    let mut failed_batches: Vec<FailedBatch> = Vec::new();
    for (path, batch) in &batches {
        match apply_batch(&state, batch) {
            Ok(applied) => {
                if as_json {
                    json_batches.push(crate::api::import_outcome(batch, &applied));
                } else {
                    println!("{}: {}", path.display(), report(batch, &applied));
                }
                touched.insert(batch.context.clone());
                ops_since_flush += batch.op_count();
            }
            Err(refusal) => {
                eprintln!("taguru: import: {}: {}", path.display(), refusal.text());
                // A refused batch is not necessarily a no-op batch:
                // everything up to the refusal (the retraction, the
                // passage, a partial prefix) landed durably, and this
                // process exits before any server-side tick could pick
                // the context up — skipping it here would leave those
                // writes' glosses unembedded for good.
                if refusal.wrote_anything() {
                    touched.insert(batch.context.clone());
                }
                ops_since_flush += refusal.ops_written();
                failures += 1;
                if as_json {
                    failed_batches.push(FailedBatch {
                        context: batch.context.clone(),
                        source: batch.source.clone(),
                        error: refusal.text(),
                    });
                }
                // Represented in `failed_batches`, not `json_batches`:
                // there is no `Applied` to build an `ImportOutcome`
                // from. The failure still reaches stderr above and counts
                // against the exit code the same as the human path.
            }
        }
        if ops_since_flush >= FLUSH_EVERY_OPS {
            state.flush_dirty();
            ops_since_flush = 0;
        }
    }

    // Schemas install after every batch, before groups restore (ADR
    // 0009 §13): a schema record's context may exist only because a
    // batch of THIS SAME run just created it. Unlike group restore,
    // each record is independent — one context's schema installing
    // does not gate another's — so failures are tallied per record
    // rather than judged as one whole set.
    let mut schema_failures = 0usize;
    let mut json_schemas: Vec<crate::api::SchemaImportOutcome> = Vec::new();
    for (path, context, installed) in &schemas {
        match apply_schema_record(&state, context, installed.clone()) {
            Ok(document) => {
                if as_json {
                    json_schemas.push(crate::api::schema_import_outcome(context, &document));
                } else {
                    println!(
                        "{}: context '{context}' schema installed (mode: {})",
                        path.display(),
                        document.mode.as_str()
                    );
                }
            }
            Err(error) => {
                eprintln!(
                    "taguru: import: {}: context '{context}': {error}",
                    path.display()
                );
                schema_failures += 1;
            }
        }
    }

    // Groups restore LAST — after every batch, whatever file carried
    // them — so a record and the member contexts it names can travel
    // in one run in any order. One set, validated whole: a member that
    // still does not exist refuses every group record, batches
    // untouched (they landed above; re-importing is idempotent).
    let mut restored = 0usize;
    let mut group_failures = 0usize;
    let mut json_groups: Vec<crate::api::GroupImportOutcome> = Vec::new();
    // Set on a group-restore refusal, carried into the final --json
    // document's `error` field — the whole group set was judged and
    // refused as one (restore_groups's own contract), so this is a
    // whole-run-level error the same way a validation or boot failure
    // is, not a per-batch one `FailedBatch` would fit.
    let mut group_restore_error: Option<String> = None;
    if !groups.is_empty() {
        let records: Vec<(String, GroupRecord)> = groups
            .iter()
            .map(|(_, name, record)| (name.clone(), record.clone()))
            .collect();
        match state.restore_groups(&records, Deadline::unbounded()) {
            Ok(outcomes) => {
                restored = outcomes.len();
                for ((path, name, record), (_, outcome)) in groups.iter().zip(&outcomes) {
                    if as_json {
                        json_groups.push(crate::api::GroupImportOutcome {
                            name: name.clone(),
                            outcome: outcome.as_str(),
                            contexts: record.contexts.len(),
                            groups: record.groups.len(),
                        });
                    } else {
                        println!(
                            "{}: {} — {}",
                            path.display(),
                            describe_group(name, record),
                            outcome.as_str()
                        );
                    }
                }
            }
            Err(refusal) => {
                restored = refusal.applied();
                eprintln!("taguru: import: {}", refusal.text());
                group_failures = groups.len() - restored;
                if as_json {
                    group_restore_error = Some(refusal.text());
                }
            }
        }
    }

    state.flush_dirty();
    state.persist_usage();

    let mut embed_failures = 0;
    if state.embeddings_configured() {
        for name in &touched {
            match state.refresh_embeddings(name, Deadline::unbounded()) {
                None | Some(Ok((0, _))) => {}
                Some(Ok((embedded, _))) => {
                    if !as_json {
                        println!("{name}: embedded {embedded} glosses");
                    }
                }
                Some(Err(error)) => {
                    eprintln!(
                        "taguru: import: {name}: embedding refresh failed ({error}) — the \
                         graph is imported and durable; refresh later via POST \
                         /contexts/{name}/embeddings/refresh"
                    );
                    embed_failures += 1;
                }
            }
        }
    }

    if as_json {
        print_import_json(
            false,
            json_batches,
            json_schemas,
            json_groups,
            failed_batches,
            group_restore_error,
        );
    } else {
        println!(
            "import: {} of {} batch(es) applied across {} context(s)",
            batches.len() - failures,
            batches.len(),
            touched.len()
        );
        if !schemas.is_empty() {
            println!(
                "import: {} of {} schema record(s) installed",
                schemas.len() - schema_failures,
                schemas.len()
            );
        }
        if !groups.is_empty() {
            println!(
                "import: {restored} of {} group record(s) restored",
                groups.len()
            );
        }
    }
    if failures > 0 || embed_failures > 0 || schema_failures > 0 || group_failures > 0 {
        1
    } else {
        0
    }
}

/// The pre-application counts `--dry-run --json` reports offline: read
/// straight off the parsed [`Batch`] (the same numbers `Batch::describe`
/// already prints as text), never touching state. `created`/
/// `retracted` and every `*_dropped` field need a real apply (or the
/// server's `preview_batch`, which needs a boot this path deliberately
/// avoids) to know — reported as 0/false rather than guessed.
fn dry_run_outcome_of(batch: &Batch) -> crate::api::ImportOutcome {
    crate::api::ImportOutcome {
        context: batch.context.clone(),
        source: batch.source.clone(),
        created: false,
        retracted: 0,
        associations: batch.associations.len(),
        aliases: batch.concepts.len() + batch.labels.len(),
        passage_stored: batch.passage.is_some(),
        passage_dropped: false,
        questions_stored: batch.questions.len(),
        questions_dropped: 0,
        sections_stored: batch.sections.len(),
        sections_dropped: 0,
        locators_stored: batch.locators.len(),
        locators_dropped: 0,
        association_paragraphs_dropped: 0,
        // Same rationale as every other field above: a real schema
        // judgment needs a booted state (`predicted_schema_rejection`
        // reads the live graph), which this parse-only path deliberately
        // avoids — reported as 0 rather than guessed.
        schema_violations: 0,
    }
}

/// One local batch [`apply_batch`] refused — `--json`'s only view of a
/// per-batch failure, since there is no `Applied` to build an
/// [`crate::api::ImportOutcome`] from. Local-only: a remote refusal
/// fails its whole chunk (the server's own contract, `src/api/
/// import.rs`), not one batch at a time, so `run_remote` has no
/// per-batch failure to name this way — only the whole-chunk `error`
/// text `print_import_json_values` carries.
#[derive(serde::Serialize)]
struct FailedBatch {
    context: String,
    source: String,
    error: String,
}

/// `import --json`'s single shared print path for the local, typed
/// entrance: `{dry_run, error, failed_batches, batches, schemas,
/// groups}` — the last three the same shape [`crate::api::
/// ImportStreamOutcome`] answers with. `error` is a whole-run failure
/// (validation refused every file, the registry wouldn't boot, group
/// restoration refused); `failed_batches` is the per-batch failures
/// the run continued past. Every `--json` exit path calls this —
/// including failure ones — so stdout is always exactly one
/// parseable document, never silence.
fn print_import_json(
    dry_run: bool,
    batches: Vec<crate::api::ImportOutcome>,
    schemas: Vec<crate::api::SchemaImportOutcome>,
    groups: Vec<crate::api::GroupImportOutcome>,
    failed_batches: Vec<FailedBatch>,
    error: Option<String>,
) {
    #[derive(serde::Serialize)]
    struct Report {
        dry_run: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        failed_batches: Vec<FailedBatch>,
        #[serde(flatten)]
        stream: crate::api::ImportStreamOutcome,
    }
    let report = Report {
        dry_run,
        error,
        failed_batches,
        stream: crate::api::ImportStreamOutcome {
            batches,
            schemas,
            groups,
        },
    };
    match serde_json::to_string_pretty(&report) {
        Ok(text) => println!("{text}"),
        Err(error) => eprintln!("taguru: import: report did not serialize: {error}"),
    }
}

/// The dry-run report line for one schema record — what the batch's
/// `describe` is to a batch. No outcome verb, unlike the applied
/// (non-dry-run) report line: whether install/replace/no-op applies
/// needs live state a dry run never loads.
fn describe_schema(context: &str, installed: &schema::InstalledSchema) -> String {
    let document = installed.document();
    format!(
        "context '{context}' schema (mode: {}): {} type(s), {} relation(s)",
        document.mode.as_str(),
        document.types.len(),
        document.relations.len()
    )
}

/// The dry-run and report line for one group record — what the batch's
/// `describe` is to a batch.
fn describe_group(name: &str, record: &GroupRecord) -> String {
    format!(
        "group '{name}': {} member context(s), {} child group(s)",
        record.contexts.len(),
        record.groups.len()
    )
}

/// The cross-file "already claimed" refusal — shared by the local and
/// remote import passes' duplicate-source checks so the wording can
/// never drift between the two entrances.
fn duplicate_source_message(context: &str, source: &str) -> String {
    format!(
        "source '{source}' in context '{context}' is already stated by an earlier file \
         — one file owns one source's truth"
    )
}

/// [`duplicate_source_message`]'s schema-record twin.
fn duplicate_schema_message(context: &str) -> String {
    format!(
        "context '{context}' schema is already stated by an earlier file — one record \
         owns one context's schema"
    )
}

/// [`duplicate_source_message`]'s group-record twin.
fn duplicate_group_message(name: &str) -> String {
    format!(
        "group '{name}' is already stated by an earlier file — one record owns one \
         group's truth"
    )
}

/// Why a `taguru_schema` record's install failed after it already
/// parsed and validated — [`crate::registry::PutSchemaError`]'s
/// entrance-agnostic twin, so both the offline CLI (this module) and
/// `POST /import` (`src/api/import.rs`) can format or map each case
/// without matching on the registry error directly.
#[derive(Debug)]
pub(crate) enum SchemaApplyError {
    /// The record's context does not exist — not yet created by an
    /// earlier batch of the same stream, nor previously.
    NoContext,
    ReservedAlias(String),
    Load(String),
    Io(std::io::Error),
}

impl std::fmt::Display for SchemaApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoContext => write!(
                f,
                "does not exist — a schema record's context must already exist (created \
                 by an earlier batch of the same stream, or previously) before its schema \
                 can install"
            ),
            Self::ReservedAlias(alias) => write!(
                f,
                "label alias '{alias}' resolves to '{}', the relation label reserved for \
                 type assertions (ADR 0009 §6.3) — rename the alias before installing this \
                 schema",
                schema::SCHEMA_TYPE_LABEL
            ),
            Self::Load(message) => write!(f, "schema could not be loaded: {message}"),
            Self::Io(error) => write!(f, "schema not persisted: {error}"),
        }
    }
}

/// Installs one parsed schema record via [`AppState::put_schema`] —
/// shared by the offline CLI's pass 2 (here) and `POST /import`'s
/// apply stage (`src/api/import.rs`) so both entrances judge the same
/// failure the same way. `Ok` carries the installed document back so
/// each caller builds its own report/outcome shape from one source of
/// truth, the way [`apply_batch`]'s `Applied` feeds both `report()`
/// and [`crate::api::import_outcome`].
pub(crate) fn apply_schema_record(
    state: &AppState,
    context: &str,
    installed: schema::InstalledSchema,
) -> Result<schema::SchemaDocument, SchemaApplyError> {
    match state.put_schema(context, installed) {
        None => Err(SchemaApplyError::NoContext),
        Some(Ok(document)) => Ok(document),
        Some(Err(crate::registry::PutSchemaError::ReservedAlias(alias))) => {
            Err(SchemaApplyError::ReservedAlias(alias))
        }
        Some(Err(crate::registry::PutSchemaError::Load(message))) => {
            Err(SchemaApplyError::Load(message))
        }
        Some(Err(crate::registry::PutSchemaError::Io(error))) => Err(SchemaApplyError::Io(error)),
    }
}

/// Which record kind a [`Unit`] carries — [`run_remote`]'s "batches
/// not yet sent" tally (and its schema-record twin) reads this apart
/// from group records, which restore through a separate path and are
/// never counted as either.
#[derive(Clone, Copy, PartialEq, Eq)]
enum UnitKind {
    Batch,
    Schema,
    Group,
}

/// One batch's, schema's, or group's rendered bytes, packed into a
/// [`Chunk`] for `import --url`'s wire chunking (ADR 0002 §9) —
/// `label` names the source (or context, or group) for the hard-error
/// and progress messages.
struct Unit {
    text: String,
    label: String,
    kind: UnitKind,
}

impl Unit {
    fn len(&self) -> usize {
        self.text.len()
    }
}

/// One `POST /import` request's worth of units, in stream order — a
/// prefix of whole batch units followed (only in the last chunk that
/// carries any) by whole group units, since groups restore after
/// every batch of the run.
struct Chunk {
    units: Vec<Unit>,
}

impl Chunk {
    fn size(&self) -> usize {
        self.units.iter().map(Unit::len).sum()
    }

    /// The wire body: every unit's text concatenated, each guaranteed
    /// to end in its own newline — a unit sliced off the last batch of
    /// a file (or EOF) may not already carry one.
    fn body(&self) -> String {
        let mut body = String::with_capacity(self.size() + self.units.len());
        for unit in &self.units {
            body.push_str(&unit.text);
            if !unit.text.ends_with('\n') {
                body.push('\n');
            }
        }
        body
    }

    /// Splits this chunk in two at the unit boundary closest to half
    /// its accumulated bytes, both halves non-empty — the ADR 0002 §9
    /// 413 adaptation, and the same halving the pre-send budget check
    /// below uses. Only ever called on a chunk carrying more than one
    /// unit: a lone oversized unit is refused before it ever reaches a
    /// chunk (splitting a batch's own record set client-side would
    /// break the retract-then-apply contract's atomicity boundary).
    fn halve(mut self) -> (Chunk, Chunk) {
        debug_assert!(self.units.len() > 1, "a lone unit cannot be halved further");
        let total = self.size();
        let mut running = 0usize;
        let mut split_at = 1;
        for (index, unit) in self.units.iter().enumerate() {
            running += unit.len();
            if running * 2 >= total {
                split_at = (index + 1).clamp(1, self.units.len() - 1);
                break;
            }
        }
        let tail = self.units.split_off(split_at);
        (Chunk { units: self.units }, Chunk { units: tail })
    }
}

/// Greedily packs units, in order, into whole-unit chunks under
/// `budget` bytes — a chunk always carries at least one unit even if
/// that unit alone exceeds `budget` (the caller refuses that case
/// before this ever runs; see [`run_remote`]'s pre-send check). Pulled
/// out of [`run_remote`] so the packing rule has a unit test
/// independent of any network call.
fn pack_chunks(units: Vec<Unit>, budget: usize) -> VecDeque<Chunk> {
    let mut queue: VecDeque<Chunk> = VecDeque::new();
    let mut pending: Vec<Unit> = Vec::new();
    let mut pending_size = 0usize;
    for unit in units {
        if !pending.is_empty() && pending_size + unit.len() > budget {
            queue.push_back(Chunk {
                units: std::mem::take(&mut pending),
            });
            pending_size = 0;
        }
        pending_size += unit.len();
        pending.push(unit);
    }
    if !pending.is_empty() {
        queue.push_back(Chunk { units: pending });
    }
    queue
}

/// The hard failure for a single batch (or group record) that alone
/// exceeds the byte budget — reported and refused before the network
/// is ever touched, naming the two real fixes (ADR 0002 §9).
fn oversized_unit_error(label: &str, size: usize, budget: usize) -> i32 {
    eprintln!(
        "taguru: import: {label} alone is {size} byte(s), over the {budget}-byte chunk \
         budget — splitting a batch's own record set client-side would break the \
         retract-then-apply contract's atomicity boundary, so this cannot be packed \
         automatically; reduce what this source's batch carries (split the source \
         upstream of import) — raising the server's TAGURU_MAX_BODY_BYTES alone will \
         not help, since this budget is fixed client-side regardless of the server's cap"
    );
    1
}

/// The hard failure for a single batch (or group record) that already
/// fit under this client's own packing budget — it passed the pre-send
/// check [`oversized_unit_error`] guards — but the SERVER still
/// answered 413 for it. Unlike that client-side refusal, this one IS
/// the server's own body-size cap, so raising `TAGURU_MAX_BODY_BYTES`
/// server-side is a real fix here, not the dead end
/// [`oversized_unit_error`]'s wording correctly rules out for its own
/// (client-fixed-budget) case.
fn server_refused_single_unit_error(label: &str, size: usize) -> i32 {
    eprintln!(
        "taguru: import: {label} ({size} byte(s)) was refused by the server as too \
         large, and cannot be split further — splitting a batch's own record set \
         client-side would break the retract-then-apply contract's atomicity \
         boundary. Raise the server's TAGURU_MAX_BODY_BYTES, or reduce what this \
         source's batch carries (split the source upstream of import)."
    );
    1
}

/// One chunk's landed `ImportOutcome` array (`POST /import`'s response
/// shape, src/api/import.rs), summarized into the same vocabulary
/// [`report`] uses for the local path's per-batch line — one line per
/// chunk instead of one per batch, since a remote chunk can carry many.
fn summarize_chunk_outcomes(outcomes: &[Value]) -> String {
    let created = outcomes
        .iter()
        .filter(|outcome| outcome["created"].as_bool() == Some(true))
        .count();
    let retracted: u64 = outcomes
        .iter()
        .filter_map(|o| o["retracted"].as_u64())
        .sum();
    let associations: u64 = outcomes
        .iter()
        .filter_map(|o| o["associations"].as_u64())
        .sum();
    let aliases: u64 = outcomes.iter().filter_map(|o| o["aliases"].as_u64()).sum();
    let passages = outcomes
        .iter()
        .filter(|outcome| outcome["passage_stored"].as_bool() == Some(true))
        .count();
    // Absent on any server old enough to predate #382, so a missing
    // field (`as_u64()` on `None` → `None`) reads as 0 the same way a
    // real schema-free batch's own count would — a remote run against
    // such a server never prints a warning line it cannot back up.
    let schema_violations: u64 = outcomes
        .iter()
        .filter_map(|o| o["schema_violations"].as_u64())
        .sum();
    format!(
        "{} batch(es){} ({retracted} association(s) retracted): +{associations} \
         association(s), +{aliases} alias(es){}{}",
        outcomes.len(),
        match created {
            0 => String::new(),
            created => format!(", {created} created"),
        },
        match passages {
            0 => String::new(),
            passages => format!(", {passages} passage(s) stored"),
        },
        match schema_violations {
            0 => String::new(),
            violations => format!(", {violations} schema warning(s)"),
        },
    )
}

/// The remote twin of [`run_local`]: validates every file the same
/// way, then packs whole batches (and, after them, whole group
/// records) into chunks under a byte budget and POSTs each to a
/// running server's `/import`, adapting to a 413 by halving the chunk
/// and resending — never splitting a batch's own record set, never
/// crossing into the next batch (ADR 0002 §9). `--dry-run` sends every
/// chunk as `?dry_run=true` instead of touching anything.
fn run_remote(base: &str, files: &[PathBuf], dry_run: bool, as_json: bool) -> i32 {
    // ADR 0002 §7: caught before any request leaves the process.
    if let Err(message) = crate::remote::reject_userinfo(base) {
        return crate::config::subcommand_usage_error("import", &message);
    }
    // A malformed `--url` is a usage problem, not a partial apply — it
    // must exit 2 (`cli.rs`'s own documented usage-error code, and
    // `evaluate --url`'s precedent) rather than fall through to
    // `Api::url`'s per-request `InvalidUrl` failure, which used to
    // surface later as an exit-1 "nothing landed" message that reads
    // like a network problem. Checked up front rather than relying on
    // the chunk loop to hit it: an input with zero batches and zero
    // groups (every batch failed the earlier owner-uniqueness check,
    // say) never enters that loop at all, so a bad URL would otherwise
    // go entirely undetected and exit 0.
    // `http`/`https` only: `url::Url::parse` alone happily accepts
    // `file://`/`ftp://` and anything else with a well-formed
    // authority, but `ureq` (the transport underneath `Api`) speaks
    // only HTTP — a non-http(s) scheme would otherwise sail through
    // this check and only fail once the request reaches `ureq`, as
    // `ImportFailure::Transport` (exit 1, "connection lost"), exactly
    // the network-problem-shaped message this upfront check exists to
    // avoid for a usage mistake.
    match url::Url::parse(base) {
        Ok(parsed) if matches!(parsed.scheme(), "http" | "https") => {}
        Ok(parsed) => {
            return crate::config::subcommand_usage_error(
                "import",
                &format!(
                    "'{base}' uses '{}', but --url only supports http/https",
                    parsed.scheme()
                ),
            );
        }
        Err(_) => {
            return crate::config::subcommand_usage_error(
                "import",
                &format!("'{base}' is not a usable base URL"),
            );
        }
    }

    // Pass 1 — every file parses, or nothing applies (same contract as
    // run_local's own Pass 1). File bytes are held past this point
    // (unlike the local path's streaming parse) because split_batches
    // needs them: the chunk packer below slices each batch straight
    // out of its source file rather than re-serializing it.
    let mut units: Vec<Unit> = Vec::new();
    let mut schema_units: Vec<Unit> = Vec::new();
    let mut group_units: Vec<Unit> = Vec::new();
    let mut batch_count = 0usize;
    let mut schema_count = 0usize;
    let mut group_count = 0usize;
    let mut broken = 0usize;
    let mut owners: HashSet<(String, String)> = HashSet::new();
    let mut schema_owners: HashSet<String> = HashSet::new();
    let mut group_owners: HashSet<String> = HashSet::new();
    for path in files {
        let mut bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) => {
                eprintln!("taguru: import: {}: {error}", path.display());
                broken += 1;
                continue;
            }
        };
        // A UTF-8 BOM only ever means anything at byte 0 of the WHOLE
        // stream (parse_stream's own note): stripped here, before
        // split_batches runs, so a batch sliced mid-stream can never
        // carry one riding into a wire chunk that isn't the first.
        if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
            bytes.drain(0..3);
        }
        let mut file_broken = false;
        match parse_stream(&bytes[..]) {
            Ok(stream) => {
                let ranges = split_batches(&bytes);
                // `split_batches` and `parse_stream` are two
                // independent scanners over the same bytes; they agree
                // today, but a `zip` below silently truncates to the
                // shorter side if they ever diverge — dropping trailing
                // batches from this remote import with no error and no
                // non-zero exit (the `--dry-run` summary's batch count
                // is computed inside this same truncated loop, so it
                // would agree with the loss and hide it completely).
                // Checked here, always — not `debug_assert_eq!`, which
                // release builds (what an operator actually runs)
                // compile out — so a divergence refuses this file
                // loudly instead of restoring a silently incomplete
                // backup.
                if ranges.len() != stream.batches.len() {
                    eprintln!(
                        "taguru: import: {}: internal error: split_batches sliced {} \
                         range(s) for {} parsed batch(es) — refusing rather than risk \
                         silently dropping batches",
                        path.display(),
                        ranges.len(),
                        stream.batches.len()
                    );
                    broken += 1;
                    continue;
                }
                for (batch, range) in stream.batches.iter().zip(ranges) {
                    if !owners.insert((batch.context.clone(), batch.source.clone())) {
                        eprintln!(
                            "taguru: import: {}: {}",
                            path.display(),
                            duplicate_source_message(&batch.context, &batch.source)
                        );
                        file_broken = true;
                        continue;
                    }
                    let text = String::from_utf8(bytes[range].to_vec())
                        .expect("parse_stream already proved this range is UTF-8");
                    units.push(Unit {
                        text,
                        label: format!(
                            "{}: context '{}' source '{}'",
                            path.display(),
                            batch.context,
                            batch.source
                        ),
                        kind: UnitKind::Batch,
                    });
                    batch_count += 1;
                }
                for (context, installed) in &stream.schemas {
                    if !schema_owners.insert(context.clone()) {
                        eprintln!(
                            "taguru: import: {}: {}",
                            path.display(),
                            duplicate_schema_message(context)
                        );
                        file_broken = true;
                        continue;
                    }
                    schema_units.push(Unit {
                        text: crate::export::render_schema(context, installed.document()),
                        label: format!("{}: context '{context}' schema", path.display()),
                        kind: UnitKind::Schema,
                    });
                    schema_count += 1;
                }
                for (name, record) in &stream.groups {
                    if !group_owners.insert(name.clone()) {
                        eprintln!(
                            "taguru: import: {}: {}",
                            path.display(),
                            duplicate_group_message(name)
                        );
                        file_broken = true;
                        continue;
                    }
                    group_units.push(Unit {
                        text: crate::export::render_group(name, record),
                        label: format!("{}: group '{name}'", path.display()),
                        kind: UnitKind::Group,
                    });
                    group_count += 1;
                }
            }
            Err(message) => {
                eprintln!("taguru: import: {}: {message}", path.display());
                file_broken = true;
            }
        }
        if file_broken {
            broken += 1;
        }
    }
    if broken > 0 {
        let message = format!(
            "{broken} of {} file(s) refused during validation; nothing was applied",
            files.len()
        );
        eprintln!("taguru: import: {message}");
        if as_json {
            print_import_json_values(dry_run, Vec::new(), Vec::new(), Vec::new(), Some(message));
        }
        return 1;
    }

    // Schemas install after every batch, before groups restore — the
    // same order run_local's Pass 2 keeps (ADR 0009 §13) — so a
    // schema record naming a context an earlier chunk's batch creates
    // must always be sent after it, and a group naming a context an
    // earlier chunk's schema installs onto must always be sent after
    // that.
    units.append(&mut schema_units);
    units.append(&mut group_units);

    // A single unit that alone exceeds the byte budget cannot be
    // packed into any chunk, however the packer below runs — refused
    // before the network is ever touched (ADR 0002 §9).
    if let Some(oversized) = units
        .iter()
        .find(|unit| unit.len() > REMOTE_IMPORT_BUDGET_BYTES)
    {
        return oversized_unit_error(
            &oversized.label,
            oversized.len(),
            REMOTE_IMPORT_BUDGET_BYTES,
        );
    }

    let api = Api::new(base.to_string());
    // ADR 0002 §5: every remote, mutating invocation prints its target
    // before sending anything.
    eprintln!("import → {base}");
    api.warn_on_version_skew("import");
    // ADR 0009 §13's explicit compatibility refusal, checked only when
    // the stream actually carries a schema record — a schema-free
    // import must behave exactly as it did before this preflight
    // existed, skew warning included, whatever the peer reports.
    if schema_count > 0
        && let Some(message) = api.schema_import_refusal()
    {
        eprintln!("{message}");
        if as_json {
            print_import_json_values(dry_run, Vec::new(), Vec::new(), Vec::new(), Some(message));
        }
        return 1;
    }

    let mut queue = pack_chunks(units, REMOTE_IMPORT_BUDGET_BYTES);

    // A LIVE estimate, not a fixed denominator: every 413-adaptive
    // halve (below, and at the loop's own proactive check) replaces
    // one queued chunk with two, growing this by one. A chunk printed
    // "N/M" before a later halving can end up describing a since-grown
    // M — the display always reflects "M chunks planned as of THIS
    // line," matching every other adaptive-retry progress display's
    // convention, not a promise that stays fixed for the whole run.
    let mut total = queue.len();
    let mut landed_chunks = 0usize;
    let mut budget = REMOTE_IMPORT_BUDGET_BYTES;
    let mut batches_landed = 0usize;
    let mut schema_records_landed = 0usize;
    let mut group_records_landed = 0usize;
    let mut contexts: BTreeSet<String> = BTreeSet::new();
    let mut json_batches: Vec<Value> = Vec::new();
    let mut json_schemas: Vec<Value> = Vec::new();
    let mut json_groups: Vec<Value> = Vec::new();

    while let Some(chunk) = queue.pop_front() {
        if chunk.size() > budget && chunk.units.len() > 1 {
            let (first, second) = chunk.halve();
            queue.push_front(second);
            queue.push_front(first);
            total += 1;
            continue;
        }
        match api.import_chunk(&chunk.body(), dry_run) {
            Ok(result) => {
                landed_chunks += 1;
                let outcomes = result
                    .get("batches")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                for outcome in &outcomes {
                    if let Some(context) = outcome.get("context").and_then(Value::as_str) {
                        contexts.insert(context.to_string());
                    }
                }
                batches_landed += outcomes.len();
                let schemas = result
                    .get("schemas")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                schema_records_landed += schemas.len();
                let groups = result
                    .get("groups")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                group_records_landed += groups.len();
                if as_json {
                    // The server already answers each chunk in exactly
                    // the shape `--json` reports — accumulated across
                    // chunks and printed once at the end, never
                    // reparsed into a different type that could drift
                    // from what the server actually said.
                    json_batches.extend(outcomes);
                    json_schemas.extend(schemas);
                    json_groups.extend(groups);
                } else {
                    // One line per LANDED chunk, always — every group unit
                    // rides after every batch unit (the append above), so
                    // the tail chunk(s) of a run can carry only group
                    // records and `outcomes` is empty for those. Skipping
                    // the line there (as this used to do) made `chunk N/M`
                    // visibly jump a number, and an operator watching the
                    // log cannot tell that apart from a crash between
                    // prints.
                    println!(
                        "chunk {landed_chunks}/{total}: {}",
                        if outcomes.is_empty() {
                            "schema/group record(s) only".to_string()
                        } else {
                            summarize_chunk_outcomes(&outcomes)
                        }
                    );
                    for schema in &schemas {
                        let context = schema.get("context").and_then(Value::as_str).unwrap_or("?");
                        let mode = schema.get("mode").and_then(Value::as_str).unwrap_or("?");
                        let types = schema.get("types").and_then(Value::as_u64).unwrap_or(0);
                        let relations =
                            schema.get("relations").and_then(Value::as_u64).unwrap_or(0);
                        println!(
                            "context '{context}' schema (mode: {mode}): {types} type(s), \
                             {relations} relation(s) — installed"
                        );
                    }
                    for group in &groups {
                        let name = group.get("name").and_then(Value::as_str).unwrap_or("?");
                        let outcome = group.get("outcome").and_then(Value::as_str).unwrap_or("?");
                        let member_contexts =
                            group.get("contexts").and_then(Value::as_u64).unwrap_or(0);
                        let child_groups = group.get("groups").and_then(Value::as_u64).unwrap_or(0);
                        println!(
                            "group '{name}': {member_contexts} member context(s), \
                             {child_groups} child group(s) — {outcome}"
                        );
                    }
                }
            }
            Err(ImportFailure::TooLarge(message)) => {
                if chunk.units.len() == 1 {
                    eprintln!("taguru: import: {message}");
                    if as_json {
                        print_import_json_values(
                            dry_run,
                            json_batches,
                            json_schemas,
                            json_groups,
                            Some(message),
                        );
                    }
                    return server_refused_single_unit_error(
                        &chunk.units[0].label,
                        chunk.units[0].len(),
                    );
                }
                // Pre-application rejection (ADR 0002 §8): safe to
                // adapt to and resend automatically — repeatedly, if
                // a further 413 keeps arriving, until this chunk is
                // a single batch (server_refused_single_unit_error) or
                // lands.
                budget = (chunk.size() / 2).max(1);
                let (first, second) = chunk.halve();
                queue.push_front(second);
                queue.push_front(first);
                total += 1;
            }
            Err(ImportFailure::InvalidUrl(message)) => {
                // No request was ever sent — nothing landed, nothing
                // to resume; this is a usage problem, not a partial
                // apply, so it exits like one (2, not 1) — the same
                // fix as the upfront `--url` check above, for the
                // narrower "parses but cannot carry a path" shape that
                // check does not itself catch (e.g. a `data:`/`mailto:`
                // URL).
                return crate::config::subcommand_usage_error("import", &message);
            }
            Err(ImportFailure::Transport(message)) => {
                eprintln!("taguru: import: {message}");
                if dry_run {
                    eprintln!(
                        "taguru: import: connection lost after chunk \
                         {landed_chunks}/{total} of the dry run — nothing was applied"
                    );
                } else {
                    eprintln!(
                        "taguru: import: connection lost after chunk {landed_chunks}/{total} \
                         — re-run `--dry-run` to confirm what would change, then resume"
                    );
                }
                if as_json {
                    print_import_json_values(
                        dry_run,
                        json_batches,
                        json_schemas,
                        json_groups,
                        Some(message),
                    );
                }
                return 1;
            }
            Err(ImportFailure::Refused {
                status,
                message,
                body,
            }) => {
                eprintln!(
                    "taguru: import: chunk {}/{total} refused: {message}",
                    landed_chunks + 1
                );
                if matches!(status, 401 | 403) {
                    eprintln!(
                        "taguru: import: /import requires the admin role — check that \
                         TAGURU_API_TOKEN (or TAGURU_API_TOKENS) names a key with it"
                    );
                }
                if let Some(integrity) = body.get("integrity").and_then(Value::as_str) {
                    eprintln!("taguru: import: integrity: {integrity}");
                }
                let unsent_batches: usize = queue
                    .iter()
                    .flat_map(|chunk| &chunk.units)
                    .filter(|unit| unit.kind == UnitKind::Batch)
                    .count();
                if unsent_batches > 0 {
                    eprintln!(
                        "taguru: import: {unsent_batches} batch(es) after this chunk were \
                         never sent"
                    );
                }
                let unsent_schemas: usize = queue
                    .iter()
                    .flat_map(|chunk| &chunk.units)
                    .filter(|unit| unit.kind == UnitKind::Schema)
                    .count();
                if unsent_schemas > 0 {
                    eprintln!(
                        "taguru: import: {unsent_schemas} schema record(s) after this chunk \
                         were never sent"
                    );
                }
                if dry_run {
                    eprintln!(
                        "taguru: import: {landed_chunks} chunk(s) of the dry run \
                         previewed cleanly before this refusal; nothing was applied"
                    );
                } else {
                    eprintln!(
                        "taguru: import: {landed_chunks} chunk(s) already landed durably; \
                         re-running the corrected stream is exact (each batch replaces its \
                         own source)"
                    );
                }
                if as_json {
                    print_import_json_values(
                        dry_run,
                        json_batches,
                        json_schemas,
                        json_groups,
                        Some(message),
                    );
                }
                return 1;
            }
        }
    }

    if as_json {
        print_import_json_values(dry_run, json_batches, json_schemas, json_groups, None);
    } else if dry_run {
        let mut summary = format!("dry run: {batch_count} batch(es)");
        if schema_count > 0 {
            summary.push_str(&format!(", {schema_count} schema record(s)"));
        }
        if group_count > 0 {
            summary.push_str(&format!(" and {group_count} group record(s)"));
        }
        summary.push_str(" valid, nothing applied");
        println!("{summary}");
    } else {
        println!(
            "import: {batches_landed} batch(es) applied across {} context(s) in {total} \
             chunk(s)",
            contexts.len()
        );
        if schema_count > 0 {
            println!(
                "import: {schema_records_landed} of {schema_count} schema record(s) installed"
            );
        }
        if group_count > 0 {
            println!("import: {group_records_landed} of {group_count} group record(s) restored");
        }
    }
    0
}

/// [`print_import_json`]'s remote twin: the server already answers
/// each chunk in exactly the shape `ImportStreamOutcome` describes, so
/// this builds the same `{dry_run, batches, schemas, groups}` envelope
/// directly from the accumulated `Value`s instead of round-tripping
/// them through the typed structs (which would risk silently dropping
/// a field the server sent that this build's types don't know about).
/// Same `{dry_run, error, batches, schemas, groups}` envelope (no
/// `failed_batches`: a remote refusal fails its whole chunk, never one
/// batch within it, so there is nothing per-batch to name — the
/// chunk's `error` text already says what happened). Called on every
/// remote `--json` exit path, including failures, with whatever
/// `batches`/`schemas`/`groups` landed before the failure — never
/// silence.
fn print_import_json_values(
    dry_run: bool,
    batches: Vec<Value>,
    schemas: Vec<Value>,
    groups: Vec<Value>,
    error: Option<String>,
) {
    let mut report = serde_json::Map::new();
    report.insert("dry_run".to_string(), Value::Bool(dry_run));
    if let Some(message) = error {
        report.insert("error".to_string(), Value::String(message));
    }
    report.insert("batches".to_string(), Value::Array(batches));
    // Matches ImportStreamOutcome's own `skip_serializing_if` on
    // `schemas`/`groups` — omitted entirely when empty, not printed as
    // `[]`, so local and remote --json agree byte for byte on a
    // schema-/group-less run.
    if !schemas.is_empty() {
        report.insert("schemas".to_string(), Value::Array(schemas));
    }
    if !groups.is_empty() {
        report.insert("groups".to_string(), Value::Array(groups));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&Value::Object(report))
            .expect("a Map of already-valid JSON values always serializes")
    );
}

/// Explicit files are taken as given; a directory contributes its
/// `*.jsonl` files in name order. An empty directory is an error — a
/// place the operator pointed at with nothing to do is a mistake, not
/// a success.
fn expand(paths: &[String]) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    for raw in paths {
        let path = Path::new(raw);
        if path.is_file() {
            files.push(path.to_path_buf());
        } else if path.is_dir() {
            let mut found: Vec<PathBuf> = fs::read_dir(path)
                .map_err(|error| format!("cannot read {raw}: {error}"))?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|p| p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
                .collect();
            if found.is_empty() {
                return Err(format!("no .jsonl files under {raw}"));
            }
            found.sort();
            files.append(&mut found);
        } else {
            return Err(format!("{raw} is neither a file nor a directory"));
        }
    }
    Ok(files)
}

/// One parsed stream: the batches, then the schema records, then the
/// group records it carried, each in stream order within its own
/// vector. The split IS the apply order — batches first, all of them,
/// then schemas, then groups (ADR 0009 §13) — so a schema record can
/// name a context a batch of the SAME stream just created, and a
/// group record can name a context whose schema just landed.
#[cfg_attr(test, derive(Debug))]
pub(crate) struct Stream {
    pub(crate) batches: Vec<Batch>,
    pub(crate) schemas: Vec<(String, schema::InstalledSchema)>,
    pub(crate) groups: Vec<(String, GroupRecord)>,
}

/// One parsed batch file: the header's claims plus the accumulated op
/// lines, every association already stamped with the header's source.
#[cfg_attr(test, derive(Debug))]
pub(crate) struct Batch {
    pub(crate) context: String,
    pub(crate) source: String,
    create: Option<ContextMeta>,
    passage: Option<String>,
    /// doc2query questions, (paragraph index, question). Structure is
    /// validated here (caps, a passage to attach to); whether each
    /// index exists in the passage's split is settled at store time,
    /// one rule for every entrance.
    questions: Vec<(u32, String)>,
    /// Section start markers, (paragraph index, label) — same
    /// structure-here/range-at-store-time split as `questions`.
    sections: Vec<(u32, String)>,
    /// Typed citation locators (ADR 0007 §7), (paragraph index,
    /// locator) — same structure-here/range-at-store-time split as
    /// `questions`/`sections`, but independent of `sections`: a
    /// locator does not extend to the next paragraph.
    locators: Vec<(u32, crate::passages::Locator)>,
    /// Source metadata (#167), riding the passage line. `stored_at`
    /// present means an export being restored — the original stamp is
    /// preserved; absent means the store stamps the import time.
    stored_at: Option<u64>,
    date: Option<u64>,
    tags: Vec<String>,
    associations: Vec<AssocOp>,
    concepts: BTreeMap<String, String>,
    labels: BTreeMap<String, String>,
}

impl Batch {
    fn op_count(&self) -> usize {
        self.associations.len() + self.concepts.len() + self.labels.len()
    }

    /// Whether applying this batch can grow the context: any passage
    /// or graph payload counts (questions/sections/locators ride the
    /// passage).
    /// A header-only batch is a pure source retraction — plus, at
    /// most, a create — which is the import-shaped way DOWN in size,
    /// so the storage-quota pre-check must let it through exactly as
    /// the write path lets retract/unalias through.
    pub(crate) fn carries_growth(&self) -> bool {
        self.passage.is_some()
            || !self.associations.is_empty()
            || !self.concepts.is_empty()
            || !self.labels.is_empty()
    }

    /// The relation spellings this batch settles on — extract feeds
    /// them to later documents' prompts so one run reuses one
    /// vocabulary.
    pub(crate) fn label_vocabulary(&self) -> BTreeSet<String> {
        self.associations
            .iter()
            .map(|op| op.label.clone())
            .chain(self.labels.values().cloned())
            .collect()
    }

    fn describe(&self) -> String {
        format!(
            "context '{}' ← source '{}': {} association(s), {} alias(es){}{}{}{}",
            self.context,
            self.source,
            self.associations.len(),
            self.concepts.len() + self.labels.len(),
            if self.passage.is_some() {
                ", 1 passage"
            } else {
                ""
            },
            if self.questions.is_empty() {
                String::new()
            } else {
                format!(", {} question(s)", self.questions.len())
            },
            if self.sections.is_empty() {
                String::new()
            } else {
                format!(", {} section(s)", self.sections.len())
            },
            if self.locators.is_empty() {
                String::new()
            } else {
                format!(", {} locator(s)", self.locators.len())
            }
        )
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    taguru_batch: u64,
    context: String,
    source: String,
    #[serde(default)]
    create: Option<CreateBlock>,
}

/// The header's optional create block — the same fields as
/// PUT /contexts/{name}, applied only when the context does not exist.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct CreateBlock {
    description: String,
    pinned: bool,
    dice_floor: Option<f64>,
    semantic_floor: Option<f32>,
}

/// The `taguru_group` record line: one group's complete truth, the
/// same fields `GET /groups/{name}` serves. Absent fields read as
/// empty — matching what export omits — so the round trip is exact.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupLine {
    taguru_group: u64,
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    contexts: Vec<String>,
    #[serde(default)]
    groups: Vec<String>,
}

/// Validates one group record line into the shape the registry
/// restores. List duplicates fold into the set silently — membership
/// is a set, exactly as over the API — but structural trouble
/// (version, sizes, an over-cap SET) refuses with the line number.
fn parse_group(value: serde_json::Value, number: usize) -> Result<(String, GroupRecord), String> {
    let line: GroupLine = serde_json::from_value(value)
        .map_err(|error| format!("line {number}: not a group record: {error}"))?;
    if line.taguru_group != GROUP_VERSION {
        return Err(format!(
            "line {number}: taguru_group {} is not a version this taguru reads (it reads \
             {GROUP_VERSION})",
            line.taguru_group
        ));
    }
    check_size(number, "name", &line.name, MAX_CONTEXT_NAME_BYTES)?;
    check_nonempty(number, "name", &line.name)?;
    check_size(
        number,
        "description",
        &line.description,
        MAX_DESCRIPTION_BYTES,
    )?;
    let mut record = GroupRecord {
        description: line.description,
        contexts: BTreeSet::new(),
        groups: BTreeSet::new(),
    };
    for (field, names, set) in [
        ("contexts", line.contexts, &mut record.contexts),
        ("groups", line.groups, &mut record.groups),
    ] {
        for member in names {
            check_size(number, field, &member, MAX_CONTEXT_NAME_BYTES)?;
            check_nonempty(number, field, &member)?;
            set.insert(member);
        }
        if set.len() > MAX_GROUP_MEMBERS {
            return Err(format!(
                "line {number}: {} {field} where a group holds at most {MAX_GROUP_MEMBERS} \
                 — split into nested child groups",
                set.len()
            ));
        }
    }
    Ok((line.name, record))
}

/// The `taguru_schema` record line: one context's whole schema
/// document, plus the `context` it installs onto. Unlike
/// [`GroupLine`], NO field defaults — every field required mirrors
/// [`schema::SchemaDocument`]'s own at-rest posture (a missing field
/// is a parse refusal, never a silent default, per ADR 0009 §13).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SchemaLine {
    taguru_schema: u64,
    context: String,
    mode: schema::SchemaMode,
    closed_labels: bool,
    types: BTreeMap<String, schema::TypeDef>,
    relations: BTreeMap<String, schema::RelationDef>,
}

/// Validates one schema record line into the installed document its
/// context restores to. Follows [`parse_group`]'s exact wording shape
/// for the version refusal (ADR 0009 §13 bullet 4) — a
/// `taguru_schema` this build cannot read refuses by line number,
/// never a silent skip. Every other structural rule (type/relation
/// caps, name lengths, `is_a` cycles and depth, the reserved relation)
/// runs through [`schema::install`], the same gate a hand-edited
/// `{stem}.schema.json` passes through at boot.
fn parse_schema(
    value: serde_json::Value,
    number: usize,
) -> Result<(String, schema::InstalledSchema), String> {
    let line: SchemaLine = serde_json::from_value(value)
        .map_err(|error| format!("line {number}: not a schema record: {error}"))?;
    if line.taguru_schema != schema::SCHEMA_VERSION {
        return Err(format!(
            "line {number}: taguru_schema {} is not a version this taguru reads (it reads \
             {})",
            line.taguru_schema,
            schema::SCHEMA_VERSION
        ));
    }
    check_size(number, "context", &line.context, MAX_CONTEXT_NAME_BYTES)?;
    check_nonempty(number, "context", &line.context)?;
    let document = schema::SchemaDocument {
        schema: line.taguru_schema,
        mode: line.mode,
        closed_labels: line.closed_labels,
        types: line.types,
        relations: line.relations,
    };
    let installed =
        schema::install(document).map_err(|violation| format!("line {number}: {violation}"))?;
    Ok((line.context, installed))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AssociationLine {
    subject: String,
    label: String,
    object: String,
    weight: f64,
    #[serde(default)]
    paragraph: Option<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AliasLine {
    alias: String,
    canonical: String,
    kind: AliasKind,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum AliasKind {
    Concept,
    Label,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PassageLine {
    passage: String,
    /// Source metadata (#167). All three default to absent, so every
    /// pre-metadata export still parses; `deny_unknown_fields` above
    /// still refuses fields this taguru does not know.
    #[serde(default)]
    stored_at: Option<u64>,
    #[serde(default)]
    date: Option<u64>,
    #[serde(default)]
    tags: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct QuestionLine {
    paragraph: u32,
    question: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SectionLine {
    paragraph: u32,
    section: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LocatorLine {
    paragraph: u32,
    locator: crate::passages::Locator,
}

/// Parses one single-batch file completely, or says which line refused
/// and why — the shape `taguru extract` emits and re-validates. Streams
/// that may carry several batches, or group records, go through
/// [`parse_stream`].
pub(crate) fn parse_batch(reader: impl BufRead) -> Result<Batch, String> {
    let mut stream = parse_stream(reader)?;
    if let Some((context, _)) = stream.schemas.first() {
        return Err(format!(
            "schema record for context '{context}' in a file where exactly one batch was \
             expected"
        ));
    }
    if let Some((name, _)) = stream.groups.first() {
        return Err(format!(
            "group record '{name}' in a file where exactly one batch was expected"
        ));
    }
    if stream.batches.len() > 1 {
        return Err(format!(
            "{} batches in one file where exactly one was expected",
            stream.batches.len()
        ));
    }
    Ok(stream
        .batches
        .pop()
        .expect("parse_stream refuses empty streams"))
}

/// Parses a batch stream: one batch, or several concatenated — the
/// shape `taguru export` renders — with any `taguru_group` records
/// riding alongside. Every `taguru_batch` header line closes the batch
/// before it and opens the next; a `taguru_group` line closes it too
/// and stands alone, so an op line after one needs a fresh header.
/// Line numbers in errors count from the stream's first line. Two
/// batches claiming one (context, source) pair — or two records
/// claiming one group — refuse the whole stream, within a stream
/// exactly as across import's files: one batch owns one source's
/// truth, one record one group's.
pub(crate) fn parse_stream(mut reader: impl BufRead) -> Result<Stream, String> {
    let mut batches: Vec<Batch> = Vec::new();
    let mut schemas: Vec<(String, schema::InstalledSchema)> = Vec::new();
    let mut groups: Vec<(String, GroupRecord)> = Vec::new();
    let mut current: Option<Batch> = None;
    let mut owners: HashSet<(String, String)> = HashSet::new();
    let mut schema_owners: HashSet<String> = HashSet::new();
    let mut group_owners: HashSet<String> = HashSet::new();
    // Per-paragraph question tally, carried as we parse so the per-line
    // cap check is a map lookup instead of a rescan of every question
    // seen so far — a batch piling questions on one paragraph would
    // otherwise be quadratic. Reset at every batch boundary.
    let mut question_counts: BTreeMap<u32, usize> = BTreeMap::new();
    // (paragraph, question) pairs already accepted this batch, so an
    // exact repeat — a doc2query generator's own duplicate, or a batch
    // author pasting a line twice — folds into the one entry already
    // held instead of spending another of the paragraph's capped
    // slots on text that adds nothing. A set lookup, for the same
    // quadratic-blowup reason `question_counts` is a map instead of a
    // rescan. Reset at every batch boundary, same as `question_counts`.
    let mut seen_questions: HashSet<(u32, String)> = HashSet::new();
    let mut raw: Vec<u8> = Vec::new();
    let mut number = 0usize;
    loop {
        number += 1;
        raw.clear();
        // Read one line without ever buffering past the cap: a single
        // newline-free run cannot force an unbounded allocation before
        // the size check. `read_until` stops at the newline or at the
        // `take` ceiling, whichever comes first — reaching the ceiling
        // with no newline is a line past the cap.
        let read = (&mut reader)
            .take(MAX_LINE_BYTES as u64 + 1)
            .read_until(b'\n', &mut raw)
            .map_err(|error| format!("line {number}: {error}"))?;
        if read == 0 {
            break;
        }
        // A UTF-8 BOM only ever means anything at byte 0 of the whole
        // stream — many Windows editors stamp one onto every file they
        // save. Left in place it rides invisibly onto the first key of
        // the first JSON object, which then fails as "not JSON" (or, if
        // it parsed at all, as an unrecognized field) with no hint that
        // the file itself looks completely normal.
        if number == 1 && raw.starts_with(&[0xEF, 0xBB, 0xBF]) {
            raw.drain(0..3);
        }
        if raw.last() != Some(&b'\n') && raw.len() > MAX_LINE_BYTES {
            return Err(format!(
                "line {number}: exceeds the {MAX_LINE_BYTES}-byte line cap"
            ));
        }
        let line = std::str::from_utf8(&raw)
            .map_err(|error| format!("line {number}: not UTF-8: {error}"))?
            .trim();
        if line.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line)
            .map_err(|error| format!("line {number}: not JSON: {error}"))?;
        let has_key = |key: &str| {
            value
                .as_object()
                .is_some_and(|object| object.contains_key(key))
        };
        let is_header = has_key("taguru_batch");
        let is_schema = has_key("taguru_schema");
        let is_group = has_key("taguru_group");
        if is_header || is_schema || is_group {
            // Any stream-level record closes the batch before it — one
            // boundary step, however many marker kinds exist.
            if let Some(finished) = current.take() {
                batches.push(finish_batch(finished)?);
                question_counts.clear();
                seen_questions.clear();
            }
        }
        if is_header {
            let batch = parse_header(value, number)?;
            if !owners.insert((batch.context.clone(), batch.source.clone())) {
                return Err(format!(
                    "line {number}: source '{}' in context '{}' is already stated by \
                     an earlier batch of this stream — one batch owns one source's truth",
                    batch.source, batch.context
                ));
            }
            current = Some(batch);
        } else if is_schema {
            let (context, installed) = parse_schema(value, number)?;
            if !schema_owners.insert(context.clone()) {
                return Err(format!(
                    "line {number}: context '{context}' schema is already stated by an \
                     earlier record of this stream — one record owns one context's schema"
                ));
            }
            schemas.push((context, installed));
        } else if is_group {
            let (name, record) = parse_group(value, number)?;
            if !group_owners.insert(name.clone()) {
                return Err(format!(
                    "line {number}: group '{name}' is already stated by an earlier record \
                     of this stream — one record owns one group's truth"
                ));
            }
            groups.push((name, record));
        } else {
            match &mut current {
                None => {
                    return Err(format!(
                        "line {number}: not a batch header (no taguru_batch field) where \
                         one was expected"
                    ));
                }
                Some(batch) => parse_op(
                    batch,
                    &mut question_counts,
                    &mut seen_questions,
                    value,
                    number,
                )?,
            }
        }
    }
    match current.take() {
        Some(finished) => batches.push(finish_batch(finished)?),
        // A stream of schema or group records alone is a legitimate
        // restore; a stream of nothing is a mistake.
        None if batches.is_empty() && schemas.is_empty() && groups.is_empty() => {
            return Err(
                "empty file: expected a batch header, schema record, or group record line"
                    .to_string(),
            );
        }
        None => {}
    }
    Ok(Stream {
        batches,
        schemas,
        groups,
    })
}

/// Byte ranges of each batch in a stream [`parse_stream`] already
/// validated: a batch runs from its `taguru_batch` header line to the
/// next stream-level record (header, `taguru_schema`, or
/// `taguru_group` line) or EOF. Schema- and group-record bytes belong
/// to no batch — they are re-rendered from the parsed records instead
/// of sliced. Lives beside the parser because the boundary rule is a
/// property of the stream FORMAT, not of either caller: `router`'s
/// cross-shard import scatter-gather and `import --url`'s chunk packer
/// both need the same ranges and must never compute them two different
/// ways.
pub(crate) fn split_batches(body: &[u8]) -> Vec<std::ops::Range<usize>> {
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
    let mut current_start: Option<usize> = None;
    let mut offset = 0usize;
    for line in body.split_inclusive(|byte| *byte == b'\n') {
        let start = offset;
        offset += line.len();
        let mut text = line;
        if start == 0 && text.starts_with(&[0xEF, 0xBB, 0xBF]) {
            text = &text[3..];
        }
        let Ok(text) = std::str::from_utf8(text) else {
            continue;
        };
        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) else {
            continue;
        };
        let Some(object) = value.as_object() else {
            continue;
        };
        if object.contains_key("taguru_batch")
            || object.contains_key("taguru_schema")
            || object.contains_key("taguru_group")
        {
            if let Some(batch_start) = current_start.take() {
                ranges.push(batch_start..start);
            }
            if object.contains_key("taguru_batch") {
                current_start = Some(start);
            }
        }
    }
    if let Some(batch_start) = current_start {
        ranges.push(batch_start..body.len());
    }
    ranges
}

/// The end-of-batch validations that need the whole batch in hand.
fn finish_batch(batch: Batch) -> Result<Batch, String> {
    // Questions attach to paragraphs of THIS batch's passage; with no
    // passage line there is no text for them to name (apply retracts
    // the source first, so "the previously stored text" does not exist
    // either).
    if !batch.questions.is_empty() && batch.passage.is_none() {
        return Err(format!(
            "{} question line(s) but no passage line — questions attach to this \
             file's passage",
            batch.questions.len()
        ));
    }
    // Sections attach to paragraphs the same way questions do, and need
    // the same passage-to-attach-to guard.
    if !batch.sections.is_empty() && batch.passage.is_none() {
        return Err(format!(
            "{} section line(s) but no passage line — sections attach to this \
             file's passage",
            batch.sections.len()
        ));
    }
    // Locators attach to paragraphs the same way sections do, and need
    // the same passage-to-attach-to guard.
    if !batch.locators.is_empty() && batch.passage.is_none() {
        return Err(format!(
            "{} locator line(s) but no passage line — locators attach to this \
             file's passage",
            batch.locators.len()
        ));
    }
    // A paragraph locator on an association names a spot in THIS batch's
    // passage, exactly as a question or section does. With no passage
    // line there is nothing to name — and `apply_batch` retracts the
    // source first, so any previously stored passage is gone too. Refuse
    // rather than persist a locator pointing into a passage that will
    // not exist (the resident-store clamp cannot catch it: the source is
    // already retracted, so it has nothing to clamp against).
    if batch.passage.is_none()
        && let Some(paragraph) = batch.associations.iter().find_map(|op| op.paragraph)
    {
        return Err(format!(
            "an association names paragraph {paragraph} but the batch has no passage \
             line — a paragraph locator attaches to this file's passage"
        ));
    }
    Ok(batch)
}

fn parse_header(value: serde_json::Value, number: usize) -> Result<Batch, String> {
    let header: Header = serde_json::from_value(value)
        .map_err(|error| format!("line {number}: not a batch header: {error}"))?;
    if header.taguru_batch != BATCH_VERSION {
        return Err(format!(
            "line {number}: taguru_batch {} is not a version this taguru reads (it reads \
             {BATCH_VERSION})",
            header.taguru_batch
        ));
    }
    check_size(number, "context", &header.context, MAX_CONTEXT_NAME_BYTES)?;
    check_nonempty(number, "context", &header.context)?;
    check_size(number, "source", &header.source, MAX_NAME_BYTES)?;
    check_nonempty(number, "source", &header.source)?;
    if let Some(create) = &header.create {
        check_size(
            number,
            "create.description",
            &create.description,
            MAX_DESCRIPTION_BYTES,
        )?;
    }
    Ok(Batch {
        context: header.context,
        source: header.source,
        create: header.create.map(|block| ContextMeta {
            description: block.description,
            pinned: block.pinned,
            dice_floor: block.dice_floor.map(|floor| floor.clamp(0.0, 1.0)),
            semantic_floor: block.semantic_floor.map(|floor| floor.clamp(0.0, 1.0)),
        }),
        passage: None,
        questions: Vec::new(),
        sections: Vec::new(),
        locators: Vec::new(),
        stored_at: None,
        date: None,
        tags: Vec::new(),
        associations: Vec::new(),
        concepts: BTreeMap::new(),
        labels: BTreeMap::new(),
    })
}

/// Classifies an op line by its distinguishing key, then parses the
/// matching shape strictly — so the error for a stray field names the
/// field instead of shrugging at every shape at once.
fn parse_op(
    batch: &mut Batch,
    question_counts: &mut BTreeMap<u32, usize>,
    seen_questions: &mut HashSet<(u32, String)>,
    value: serde_json::Value,
    number: usize,
) -> Result<(), String> {
    let Some(object) = value.as_object() else {
        return Err(format!("line {number}: a batch line must be a JSON object"));
    };
    if object.contains_key("subject") {
        let op: AssociationLine = serde_json::from_value(value)
            .map_err(|error| format!("line {number}: association: {error}"))?;
        if !op.weight.is_finite() || op.weight.abs() > MAX_ASSOCIATION_WEIGHT {
            return Err(format!(
                "line {number}: weight {} is outside the accepted range (finite, \
                 |weight| <= {MAX_ASSOCIATION_WEIGHT})",
                op.weight
            ));
        }
        for (field, text) in [
            ("subject", &op.subject),
            ("label", &op.label),
            ("object", &op.object),
        ] {
            check_size(number, field, text, MAX_NAME_BYTES)?;
            check_nonempty(number, field, text)?;
        }
        batch.associations.push(AssocOp {
            subject: op.subject,
            label: op.label,
            object: op.object,
            weight: op.weight,
            source: Some(batch.source.clone()),
            paragraph: op.paragraph,
        });
    } else if object.contains_key("alias") {
        let op: AliasLine = serde_json::from_value(value)
            .map_err(|error| format!("line {number}: alias: {error}"))?;
        check_size(number, "alias", &op.alias, MAX_NAME_BYTES)?;
        check_nonempty(number, "alias", &op.alias)?;
        check_size(number, "canonical", &op.canonical, MAX_NAME_BYTES)?;
        check_nonempty(number, "canonical", &op.canonical)?;
        let namespace = match op.kind {
            AliasKind::Concept => &mut batch.concepts,
            AliasKind::Label => &mut batch.labels,
        };
        if namespace.insert(op.alias.clone(), op.canonical).is_some() {
            return Err(format!(
                "line {number}: alias '{}' appears twice in this file",
                op.alias
            ));
        }
    } else if object.contains_key("passage") {
        let op: PassageLine = serde_json::from_value(value)
            .map_err(|error| format!("line {number}: passage: {error}"))?;
        if op.passage.len() > MAX_PASSAGE_BYTES {
            return Err(format!(
                "line {number}: passage of {} bytes exceeds the {MAX_PASSAGE_BYTES}-byte cap",
                op.passage.len()
            ));
        }
        if op.tags.len() > crate::api::MAX_TAGS_PER_SOURCE {
            return Err(format!(
                "line {number}: {} tags where a source carries at most {}",
                op.tags.len(),
                crate::api::MAX_TAGS_PER_SOURCE
            ));
        }
        for tag in &op.tags {
            check_size(number, "tag", tag, crate::api::MAX_TAG_BYTES)?;
            check_nonempty(number, "tag", tag)?;
        }
        if batch.passage.replace(op.passage).is_some() {
            return Err(format!(
                "line {number}: a second passage line — one batch file carries at most \
                 one passage (the header source's original text)"
            ));
        }
        // Metadata rides the (single) passage line, so these can only
        // land once — behind the replace check above.
        batch.stored_at = op.stored_at;
        batch.date = op.date;
        batch.tags = op.tags;
    } else if object.contains_key("question") {
        let op: QuestionLine = serde_json::from_value(value)
            .map_err(|error| format!("line {number}: question: {error}"))?;
        check_size(
            number,
            "question",
            &op.question,
            crate::api::MAX_QUESTION_BYTES,
        )?;
        // An empty question would still be embedded on the next refresh,
        // and providers refuse zero-length input — failing the whole
        // refresh pass, every pass, at the same spot.
        check_nonempty(number, "question", &op.question)?;
        // Identical (paragraph, question) pairs fold into the one entry
        // already held silently — matching the group-list dedup elsewhere
        // in this file — rather than spending one of the paragraph's
        // capped slots on a duplicate doc2query line.
        if seen_questions.insert((op.paragraph, op.question.clone())) {
            let siblings = question_counts.entry(op.paragraph).or_insert(0);
            if *siblings >= crate::api::MAX_QUESTIONS_PER_PARAGRAPH {
                return Err(format!(
                    "line {number}: paragraph {} already carries {} questions (the cap)",
                    op.paragraph,
                    crate::api::MAX_QUESTIONS_PER_PARAGRAPH
                ));
            }
            *siblings += 1;
            batch.questions.push((op.paragraph, op.question));
        }
    } else if object.contains_key("section") {
        let op: SectionLine = serde_json::from_value(value)
            .map_err(|error| format!("line {number}: section: {error}"))?;
        check_size(
            number,
            "section",
            &op.section,
            crate::api::MAX_SECTION_BYTES,
        )?;
        check_nonempty(number, "section", &op.section)?;
        batch.sections.push((op.paragraph, op.section));
    } else if object.contains_key("locator") {
        let op: LocatorLine = serde_json::from_value(value)
            .map_err(|error| format!("line {number}: locator: {error}"))?;
        check_size(
            number,
            "locator.kind",
            &op.locator.kind,
            crate::api::MAX_LOCATOR_KIND_BYTES,
        )?;
        check_nonempty(number, "locator.kind", &op.locator.kind)?;
        check_size(
            number,
            "locator.value",
            &op.locator.value,
            crate::api::MAX_LOCATOR_VALUE_BYTES,
        )?;
        check_nonempty(number, "locator.value", &op.locator.value)?;
        batch.locators.push((op.paragraph, op.locator));
    } else {
        return Err(format!(
            "line {number}: not an association (subject/label/object/weight), an alias \
             (alias/canonical/kind), a passage line, a question (paragraph/question) line, \
             a section (paragraph/section) line, or a locator (paragraph/locator) line"
        ));
    }
    Ok(())
}

fn check_size(number: usize, field: &str, text: &str, cap: usize) -> Result<(), String> {
    if text.len() > cap {
        return Err(format!(
            "line {number}: {field} of {} bytes exceeds the {cap}-byte cap",
            text.len()
        ));
    }
    Ok(())
}

/// Companion to `check_size`, at the other end of the range: an empty
/// subject/label/object is not a degenerate name, it is no name — see
/// `api::empty`, which guards the same triple at the HTTP boundary.
fn check_nonempty(number: usize, field: &str, text: &str) -> Result<(), String> {
    if text.is_empty() {
        return Err(format!("line {number}: {field} must not be empty"));
    }
    Ok(())
}

/// What one batch accomplished — the CLI formats it into a report
/// line, `POST /import` serializes it into the response.
#[cfg_attr(test, derive(Debug))]
pub(crate) struct Applied {
    pub(crate) created: bool,
    pub(crate) retracted: usize,
    pub(crate) associations: usize,
    pub(crate) aliases: usize,
    pub(crate) passage_stored: bool,
    /// A previously stored passage for this source was retracted and the
    /// batch carried no replacement. With `passage_stored` this is a
    /// routine replace; without it, the batch just erased passage text —
    /// surfaced so that loss is never silent.
    pub(crate) passage_dropped: bool,
    pub(crate) questions_stored: usize,
    /// Questions naming a paragraph their passage's split does not
    /// have — most often a producer's index drifting from the server's
    /// canonical split.
    pub(crate) questions_dropped: usize,
    pub(crate) sections_stored: usize,
    /// Sections naming a paragraph their passage's split does not have
    /// (same convention and same likely cause as `questions_dropped`),
    /// plus any but the last of two or more sections claiming the same
    /// paragraph — a start marker governs until the next one, so only
    /// one can ever apply.
    pub(crate) sections_dropped: usize,
    pub(crate) locators_stored: usize,
    /// Locators naming a paragraph their passage's split does not have
    /// (same convention and same likely cause as `sections_dropped`),
    /// plus any but the last of two or more locators claiming the same
    /// paragraph — unlike a section's start marker, a locator names
    /// only its own paragraph, but the same one-per-paragraph
    /// last-write-wins rule applies.
    pub(crate) locators_dropped: usize,
    /// Association paragraph locators naming a spot this batch's own
    /// passage split does not have. Dropped exactly as `questions_dropped`
    /// and `sections_dropped` are — the association's fact still lands,
    /// only the paragraph pointer is cleared — and surfaced for the same
    /// reason: so the loss is a reported number, not a silent one.
    pub(crate) association_paragraphs_dropped: usize,
    /// `warn`-mode schema violations this batch's associations raised
    /// (ADR 0009 §8.3) — the true count, surviving truncation, mirrored
    /// into `ImportOutcome.schema_violations`. Always 0 for `off`, no
    /// schema, or `strict` (a `strict` violation refuses the batch
    /// instead — see [`ApplyRefusal::Schema`] — so it never reaches
    /// here).
    pub(crate) schema_violations: usize,
    /// The same violations, capped at `MAX_LISTED_ISSUES` and with
    /// batch-relative paths (`associations[{i}]...`, no `batches[{b}].`
    /// prefix — that is `src/api/import.rs`'s to add, once it knows this
    /// batch's stream position). Not part of `ImportOutcome` itself: the
    /// HTTP handler reads this to build the response envelope's
    /// `issues`; the CLI only reports the count.
    pub(crate) schema_issues: Vec<crate::api::Issue>,
}

/// Why a batch did not (fully) apply — one shape for both entrances:
/// the CLI prints [`ApplyRefusal::text`], the HTTP endpoint maps the
/// variant onto a status and sends the same words.
#[cfg_attr(test, derive(Debug))]
pub(crate) enum ApplyRefusal {
    /// The context does not exist and the batch brought no create
    /// block (404 over HTTP).
    NoContext(String),
    /// Filesystem trouble creating the context or persisting the
    /// passage (500).
    Io(String),
    /// The registry refused access (mapped like every other write).
    Access(AccessError),
    /// The library rejected an op partway; `applied` counts what
    /// landed first, `full` distinguishes capacity (507) from
    /// conflict (409). The retraction makes a corrected retry exact.
    Partial {
        applied: usize,
        message: String,
        full: bool,
    },
    /// Predicted before anything mutated: this batch's own alias
    /// operations would resolve to `AliasError::UnknownCanonical` or
    /// `Conflict` once actually applied, so the whole batch is
    /// refused up front (409) — no context created, no marker opened,
    /// no retraction, nothing. Distinct from `Partial { applied: 0,
    /// .. }`, which can only follow the retraction (itself a write)
    /// already landing. Structured (issue #182) rather than a bare
    /// message, so the HTTP endpoint can name the offending alias as a
    /// path-addressed `Issue` instead of prose alone.
    Rejected(AliasRejection),
    /// Predicted before anything mutated, same position as `Rejected`
    /// (checked right after it): this batch's own associations would
    /// violate a `strict` context's schema, or its own `labels`
    /// declares the reserved `schema:type` alias (ADR 0009 §6.3 guard
    /// 2, §7.2 step 7). Structured for the same reason `Rejected` is —
    /// path-addressed `Issue`s an MCP host corrects and resends.
    Schema(SchemaRejection),
}

/// Which alias namespace a predicted rejection concerns — concepts
/// intern subjects/objects, labels intern relation names.
#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
pub(crate) enum AliasNamespace {
    Concept,
    Label,
}

impl AliasNamespace {
    pub(crate) fn as_str(&self) -> &'static str {
        match self {
            Self::Concept => "concepts",
            Self::Label => "labels",
        }
    }
}

/// A predicted alias rejection (issue #182): this batch's own alias
/// operations would resolve to [`AliasError::UnknownCanonical`] or
/// [`AliasError::Conflict`] once actually applied — named precisely
/// enough to build a structured `Issue` from, not just the prose
/// [`AliasRejection::text`] already reported.
#[cfg_attr(test, derive(Debug))]
pub(crate) struct AliasRejection {
    pub(crate) namespace: AliasNamespace,
    pub(crate) alias: String,
    pub(crate) canonical: String,
    pub(crate) error: AliasError,
}

impl AliasRejection {
    pub(crate) fn text(&self) -> String {
        format!(
            "{} alias '{}' → '{}': {}; nothing was applied",
            match self.namespace {
                AliasNamespace::Concept => "concept",
                AliasNamespace::Label => "label",
            },
            self.alias,
            self.canonical,
            self.error,
        )
    }
}

impl ApplyRefusal {
    /// Whether the batch may have durably written anything before the
    /// refusal. Only [`ApplyRefusal::NoContext`], [`ApplyRefusal::Rejected`],
    /// and [`ApplyRefusal::Schema`] provably precede the first write —
    /// all three are predicted before the context is even created.
    /// Everything past that point starts with the source retraction,
    /// itself a durable write, so a later refusal (a passage that
    /// would not persist, a partial prefix of associations or aliases)
    /// leaves real changes behind. `Io` from a failed create or a
    /// failed batch-marker write is the over-approximation (both
    /// precede the first graph write); the refresh pass answers an
    /// absent context with its no-op `None` arm anyway.
    pub(crate) fn wrote_anything(&self) -> bool {
        !matches!(
            self,
            Self::NoContext(_) | Self::Rejected(_) | Self::Schema(_)
        )
    }

    /// How many ops this refusal's batch durably wrote before failing.
    /// Only [`ApplyRefusal::Partial`] carries a count — association or
    /// alias ops that landed in the WAL before the op that tripped the
    /// refusal. Feeds `ops_since_flush` in the import loop: a run
    /// dominated by partial failures (a capacity cap hit over and
    /// over) still needs its mid-run flushes on schedule, or the very
    /// WAL growth `FLUSH_EVERY_OPS` exists to bound goes unwatched.
    pub(crate) fn ops_written(&self) -> usize {
        match self {
            Self::Partial { applied, .. } => *applied,
            Self::NoContext(_) | Self::Io(_) | Self::Access(_) | Self::Rejected(_) => 0,
            Self::Schema(_) => 0,
        }
    }

    pub(crate) fn text(&self) -> String {
        match self {
            Self::NoContext(context) => {
                format!("context '{context}' does not exist and the batch brought no create block")
            }
            Self::Io(message) => message.clone(),
            Self::Access(AccessError::NotFound) => {
                "the context was deleted out from under the batch".to_string()
            }
            Self::Access(AccessError::Load(error)) => {
                format!("the context image would not load: {error}")
            }
            Self::Access(AccessError::Unpersisted(error)) => {
                format!("the WAL refused the write: {error}")
            }
            // `import_refusal` (api.rs) routes the Access variant to
            // `access_error_noted` directly and never calls `text()`
            // on it; the CLI import path runs with
            // Deadline::unbounded(). Unreachable either way, kept for
            // exhaustiveness.
            Self::Access(AccessError::DeadlineExceeded) => "deadline exceeded".to_string(),
            // Same unreachability, other leg: the CLI import boots with
            // no quota declaration (offline commands run as the
            // operator), and the HTTP path never calls `text()` on
            // Access.
            Self::Access(AccessError::QuotaExceeded(message)) => message.clone(),
            Self::Partial { message, .. } => message.clone(),
            Self::Rejected(rejection) => rejection.text(),
            Self::Schema(rejection) => rejection.text(),
        }
    }
}

/// Association paragraph locators corrected against this batch's own
/// passage split: a locator naming a spot the split does not have is
/// meaningless, so it is cleared (the association's fact still lands)
/// and counted as dropped. A batch with no passage has nothing to
/// check a locator against, so every op passes through unchanged.
/// Shared between the write path ([`apply_batch`]) and its read-only
/// preview ([`preview_batch`]) so the two can never disagree.
/// `paragraph_count`, when already known (`preview_batch` needs it for
/// its own question/section drop counts), is reused instead of
/// re-splitting the same passage text.
fn corrected_associations(batch: &Batch, paragraph_count: Option<usize>) -> (Vec<AssocOp>, usize) {
    let Some(text) = &batch.passage else {
        return (batch.associations.clone(), 0);
    };
    let paragraph_count = paragraph_count.unwrap_or_else(|| crate::paragraph::split(text).len());
    let mut dropped = 0;
    let corrected = batch
        .associations
        .iter()
        .cloned()
        .map(|mut op| {
            if op.paragraph.is_some_and(|p| p as usize >= paragraph_count) {
                op.paragraph = None;
                dropped += 1;
            }
            op
        })
        .collect();
    (corrected, dropped)
}

/// Predicts, without writing anything, whether this batch's own alias
/// operations would resolve to `AliasError::UnknownCanonical` or
/// `Conflict` once actually applied — the only purely content-driven
/// (non-capacity) rejections anywhere in the four-step apply pipeline.
/// Shared between [`apply_batch`] and [`preview_batch`] so a dry run
/// can never disagree with the real import about this call.
///
/// Checks concepts before labels, mirroring the WAL op order
/// `add_aliases` actually writes in, so a predicted message names the
/// same operation that would be the first to fail for real.
///
/// A context that does not exist yet has no aliases and no
/// associations to seed fresh names with, so a batch with a `create`
/// block is checked against an empty [`Context::default`] — exactly
/// the value `AppState::create` seeds a new context with. A context
/// that does not exist and brings no `create` block is left to the
/// ordinary `NoContext` refusal that follows this check.
fn predicted_alias_rejection(state: &AppState, batch: &Batch) -> Option<AliasRejection> {
    if batch.concepts.is_empty() && batch.labels.is_empty() {
        return None;
    }
    let concepts = batch
        .associations
        .iter()
        .flat_map(|op| [op.subject.as_str(), op.object.as_str()]);
    let labels = batch.associations.iter().map(|op| op.label.as_str());
    let check = move |context: &Context| -> Option<AliasRejection> {
        if let Err((alias, canonical, error)) =
            context.check_concept_aliases(&batch.concepts, concepts)
        {
            return Some(AliasRejection {
                namespace: AliasNamespace::Concept,
                alias: alias.to_string(),
                canonical: canonical.to_string(),
                error,
            });
        }
        if let Err((alias, canonical, error)) = context.check_label_aliases(&batch.labels, labels) {
            return Some(AliasRejection {
                namespace: AliasNamespace::Label,
                alias: alias.to_string(),
                canonical: canonical.to_string(),
                error,
            });
        }
        None
    };

    if state.directory_entry(&batch.context).is_none() {
        return if batch.create.is_some() {
            check(&Context::default())
        } else {
            None
        };
    }
    state.read_context(&batch.context, check).ok().flatten()
}

/// `warn`-mode schema violations this batch's own associations raised
/// (ADR 0009 §8.3), capped like every other collect-all pass — empty
/// whenever the batch is clean, the context has no schema, or the
/// schema's mode is `off`.
#[cfg_attr(test, derive(Debug))]
pub(crate) struct SchemaWarnings {
    pub(crate) issues: Vec<crate::api::Issue>,
    pub(crate) total: usize,
}

impl SchemaWarnings {
    fn none() -> Self {
        Self {
            issues: Vec::new(),
            total: 0,
        }
    }
}

/// A predicted schema rejection (ADR 0009 §7.2, §6.3): this batch's own
/// associations would violate a `strict` context's domain/range
/// constraints, or this batch's own `labels` declares the reserved
/// `schema:type` alias — named precisely enough to build path-addressed
/// `Issue`s from, exactly like [`AliasRejection`] beside it. `reserved`
/// tells [`ApplyRefusal::text`] and the HTTP status
/// (`src/api/import.rs`) which of the two this is: a reserved-label
/// collision is a namespace conflict (409, like an alias `Conflict`), a
/// domain/range violation is a refused value (400, ADR 0009 §8.1).
#[cfg_attr(test, derive(Debug))]
pub(crate) struct SchemaRejection {
    pub(crate) issues: Vec<crate::api::Issue>,
    pub(crate) total: usize,
    pub(crate) reserved: bool,
}

impl SchemaRejection {
    pub(crate) fn text(&self) -> String {
        let what = if self.reserved {
            "this batch's label aliases"
        } else {
            "this batch's associations"
        };
        crate::api::collected_validation_message(what, &self.issues, self.total)
    }
}

/// Which caller is running [`predicted_schema_rejection`] (#388, S10 of
/// #218's ADR 0009 split §15): only [`Apply`](CheckPurpose::Apply) is a
/// real write gate, so only it feeds `taguru_schema_checks_total` —
/// [`Preview`](CheckPurpose::Preview) runs the identical check for
/// `?dry_run=true`/`preview_batch`, and counting it too would let a
/// validate-then-apply workflow double-count the same refusal.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CheckPurpose {
    Apply,
    Preview,
}

/// Predicts, without writing anything, whether this batch's own
/// associations would violate a `strict` context's schema, or its own
/// `labels` declares the reserved `schema:type` alias (ADR 0009 §6.3
/// guard 2's batch-local bullet, checked regardless of mode) — the
/// schema twin of [`predicted_alias_rejection`], run right after it: an
/// alias conflict is caught first, so by the time this runs every
/// concept/label spelling this batch's own `associations` use is
/// already known not to collide with its own `concepts`/`labels`
/// declarations, which `schema::check::SchemaEnv` relies on (its own
/// doc, `src/schema/check.rs:88-94`). Shared between [`apply_batch`]
/// and [`preview_batch`] for the same reason `predicted_alias_rejection`
/// is: a dry run can never disagree about this call either. `purpose`
/// distinguishes them for metrics only — the judgment itself is
/// identical either way.
///
/// No schema installed for this context — including one that does not
/// exist yet — returns `Ok` before a single lock is taken
/// (`AppState::schema_of`'s own fast path for `schema_digest.is_none()`):
/// the zero-cost path every schema-free context takes, ADR 0009 §7.2
/// step 1. A schema recorded but currently unreadable is never treated
/// as schema-free — `src/schema.rs`'s own module doc fixes that as a
/// hard refusal, never a silent fallback — so this maps such a read
/// failure to [`ApplyRefusal::Io`] instead of proceeding.
fn predicted_schema_rejection(
    state: &AppState,
    batch: &Batch,
    purpose: CheckPurpose,
) -> Result<SchemaWarnings, ApplyRefusal> {
    let schema = match state.schema_of(&batch.context) {
        None | Some(Ok(None)) => return Ok(SchemaWarnings::none()),
        Some(Ok(Some(schema))) => schema,
        Some(Err(message)) => {
            return Err(ApplyRefusal::Io(format!(
                "schema for context '{}' could not be read: {message}",
                batch.context
            )));
        }
    };

    // The exact ops the write will apply, not the raw batch (ADR 0009
    // §7.2 step 2) — `None` here (not the paragraph count `apply_batch`
    // does not have yet either) matches `apply_batch`'s own call
    // (`corrected_associations(batch, None)`), so the two entrances
    // build this from the identical input.
    let (ops, _) = corrected_associations(batch, None);

    let check = state
        .read_context(&batch.context, |context| {
            let env = crate::schema::SchemaEnv::build(
                context,
                crate::schema::SchemaCheckInput {
                    schema: schema.clone(),
                    ops: &ops,
                    declared_labels: &batch.labels,
                    // `apply_batch` retracts `batch.source` before
                    // applying (`:2354-2357` at the time of writing) —
                    // the live-half exclusion this passes on to
                    // `SchemaEnv::build` judges against the graph state
                    // this write is about to leave behind, not its
                    // current one (ADR 0009 §7.2 step 4).
                    retracted_source: Some(&batch.source),
                },
            );
            crate::schema::schema_issues(
                &env,
                &ops,
                crate::schema::IssuePath::Request { prefix: "" },
            )
        })
        .map_err(ApplyRefusal::Access)?;

    let mode = schema.document().mode;
    if purpose == CheckPurpose::Apply {
        state.note_schema_check(&batch.context, check.outcome(mode), check.violations.len());
    }

    if !check.reserved.is_empty() {
        let (issues, total) = crate::api::truncate_issues(check.reserved);
        return Err(ApplyRefusal::Schema(SchemaRejection {
            issues,
            total,
            reserved: true,
        }));
    }

    let (issues, total) = crate::api::truncate_issues(check.violations);
    if mode == crate::schema::SchemaMode::Strict && total > 0 {
        return Err(ApplyRefusal::Schema(SchemaRejection {
            issues,
            total,
            reserved: false,
        }));
    }
    // `off` and a clean `strict` batch both fall through here with an
    // empty `issues`/`total` — constructing `SchemaWarnings` either way
    // rather than special-casing keeps this function's one dispatch
    // point exactly what ADR 0009 §7.2 step 7 describes.
    Ok(SchemaWarnings { issues, total })
}

/// Applies one validated batch: ensure the context, retract the
/// source, then land passage → associations → aliases. Aliases go
/// last on purpose — an alias needs its canonical interned, and the
/// associations just before are what intern it. Before any of that,
/// [`predicted_alias_rejection`] checks whether this batch's own alias
/// operations would resolve to a conflict, then [`predicted_schema_rejection`]
/// checks whether they would violate the context's schema; either
/// predicted rejection refuses the whole batch ([`ApplyRefusal::Rejected`]
/// / [`ApplyRefusal::Schema`]) up front, so a bad alias or a schema
/// violation no longer surfaces only after the associations (or the
/// retraction) have already landed.
///
/// Past that point, the four mutations are separately durable, so a
/// crash between them leaves the source half-applied with every store
/// individually consistent — undetectable after the fact. A
/// batch-open marker brackets them: written before the retraction,
/// removed only after the aliases, so boot and `taguru inspect` can
/// name any batch that never finished. A batch refused after the
/// marker opens (capacity, disk trouble) keeps its marker too — the
/// refusal is reported once, the marker keeps saying so until the
/// documented repair (re-import, or retract the source) actually
/// runs. A predicted rejection opens no marker at all: nothing ran
/// yet for it to bracket. Cross-store atomicity is deliberately not
/// attempted for what prediction cannot catch: per-source
/// retract-then-apply idempotency already makes the repair exact, so
/// detection is the remaining gap.
pub(crate) fn apply_batch(state: &AppState, batch: &Batch) -> Result<Applied, ApplyRefusal> {
    if let Some(rejection) = predicted_alias_rejection(state, batch) {
        return Err(ApplyRefusal::Rejected(rejection));
    }
    let schema_warnings = predicted_schema_rejection(state, batch, CheckPurpose::Apply)?;

    let mut created = false;
    if state.directory_entry(&batch.context).is_none() {
        let Some(meta) = &batch.create else {
            return Err(ApplyRefusal::NoContext(batch.context.clone()));
        };
        match state.create(&batch.context, meta.clone()) {
            Ok(()) => created = true,
            // Another writer got between the check and the create —
            // possible on the live server, harmless everywhere: the
            // context exists now, which is all the batch needed.
            Err(CreateError::AlreadyExists) => {}
            // Unreachable in practice — `parse_header` already refused an
            // empty context name — but the registry guards it too, so the
            // match must speak for it.
            Err(CreateError::InvalidName) => {
                return Err(ApplyRefusal::Io(format!(
                    "context name '{}' is not usable (empty)",
                    batch.context
                )));
            }
            Err(CreateError::Io(io_error)) => {
                return Err(ApplyRefusal::Io(format!(
                    "creating context '{}': {io_error}",
                    batch.context
                )));
            }
        }
    }

    // The marker precedes the first mutation or the batch does not
    // run: starting untracked would silently reopen the exact
    // undetectable-tear window it exists to close.
    if let Err(error) = state.open_import_marker(&batch.context, &batch.source) {
        return Err(ApplyRefusal::Io(format!(
            "import marker not persisted: {error} — nothing was applied"
        )));
    }

    // Not `retract_source`: this batch's own marker (opened above)
    // already brackets this call along with every step that follows —
    // clearing it here too would reopen the batch to the exact gap it
    // exists to close.
    let (retracted, passage_removed, passage_removal_errored) = state
        .retract_source_unmarked(&batch.context, &batch.source)
        .map_err(ApplyRefusal::Access)?;
    // `passage_removed` alone is unconditional — true whenever a prior
    // passage existed and was removed, with no notion of a forthcoming
    // replacement. `Applied::passage_dropped` promises the narrower
    // "and the batch carried no replacement," same as `preview_batch`.
    let passage_dropped = passage_removed && batch.passage.is_none();

    // A genuine passage-store failure here only self-heals when this
    // batch carries a replacement passage: `store_passages` below then
    // overwrites whatever stale copy the failed retraction left
    // behind. With no replacement coming, that stale passage would
    // survive under a marker this function is about to clear as if
    // the source's truth were fully applied — refuse instead, leaving
    // the marker (and the documented repair) in place.
    if passage_removal_errored && batch.passage.is_none() {
        return Err(ApplyRefusal::Io(format!(
            "old passage for source '{}' could not be retracted and this batch carries no \
             replacement passage to overwrite it with — its truth may be half-applied",
            batch.source
        )));
    }

    let mut questions_stored = 0;
    let mut questions_dropped = 0;
    let mut sections_stored = 0;
    let mut sections_dropped = 0;
    let mut locators_stored = 0;
    let mut locators_dropped = 0;
    if let Some(text) = &batch.passage {
        let outcome = state
            .store_passages(
                &batch.context,
                BTreeMap::from([(
                    batch.source.clone(),
                    crate::passages::PassageSubmission {
                        text: text.clone(),
                        questions: batch.questions.clone(),
                        sections: batch.sections.clone(),
                        locators: batch.locators.clone(),
                        meta: crate::passages::SourceMeta {
                            stored_at: batch.stored_at,
                            date: batch.date,
                            tags: batch.tags.clone(),
                        },
                    },
                )]),
            )
            .ok_or(ApplyRefusal::Access(AccessError::NotFound))?
            .map_err(|error| match error {
                // The policy refusal keeps its shape (507 over HTTP,
                // via the same Access mapping every graph gate uses);
                // only genuine disk trouble flattens to Io.
                crate::registry::PassagesWriteError::QuotaExceeded(message) => {
                    ApplyRefusal::Access(AccessError::QuotaExceeded(message))
                }
                crate::registry::PassagesWriteError::Io(io_error) => {
                    ApplyRefusal::Io(format!("passage not persisted: {io_error}"))
                }
            })?;
        questions_stored = outcome.questions_stored;
        questions_dropped = outcome.questions_dropped;
        sections_stored = outcome.sections_stored;
        sections_dropped = outcome.sections_dropped;
        locators_stored = outcome.locators_stored;
        locators_dropped = outcome.locators_dropped;
    }

    // Same rule as questions/sections above, applied silently: a
    // paragraph naming a spot this batch's own passage does not have
    // is meaningless, so it is dropped rather than persisted — the
    // association itself (subject/label/object/weight) still lands.
    // Only checked against a passage this same batch carries; an
    // associations-only batch has nothing to check against, exactly
    // like questions/sections above.
    let (corrected, association_paragraphs_dropped) = corrected_associations(batch, None);
    let associations_to_apply: &[AssocOp] = &corrected;

    let mut associations = 0;
    for chunk in associations_to_apply.chunks(MAX_ASSOCIATIONS_PER_REQUEST) {
        match state
            .add_associations(&batch.context, chunk.to_vec(), Deadline::unbounded())
            .map_err(ApplyRefusal::Access)?
        {
            Ok(applied) => associations += applied,
            Err(partial) => {
                let applied = associations + partial.applied;
                return Err(ApplyRefusal::Partial {
                    applied,
                    message: format!(
                        "applied {applied} association(s), then: {} — fix the batch and \
                         re-import; the retraction makes the retry exact",
                        partial.message
                    ),
                    full: partial.full,
                });
            }
        }
    }

    let mut aliases = 0;
    if !batch.concepts.is_empty() || !batch.labels.is_empty() {
        match state
            .add_aliases(&batch.context, &batch.concepts, &batch.labels)
            .map_err(ApplyRefusal::Access)?
        {
            Ok(applied) => aliases += applied,
            Err(partial) => {
                return Err(ApplyRefusal::Partial {
                    // Same running total the association arm above
                    // reports: `applied` is the batch's cumulative
                    // count, not just this call's — a batch whose
                    // associations landed but whose first alias
                    // didn't must not report 0 (`partial.applied`
                    // alone) when `associations` ops are already
                    // durable.
                    applied: associations + partial.applied,
                    message: format!(
                        "applied {} alias(es), then: {}",
                        partial.applied, partial.message
                    ),
                    full: partial.full,
                });
            }
        }
    }

    // Only now is the source's stated truth fully on disk.
    state.clear_import_marker(&batch.context, &batch.source);

    state.note_write(&batch.context);
    Ok(Applied {
        created,
        retracted,
        associations,
        aliases,
        passage_stored: batch.passage.is_some(),
        passage_dropped,
        questions_stored,
        questions_dropped,
        sections_stored,
        sections_dropped,
        locators_stored,
        locators_dropped,
        association_paragraphs_dropped,
        schema_violations: schema_warnings.total,
        schema_issues: schema_warnings.issues,
    })
}

/// The read-only twin of [`apply_batch`], for `POST
/// /import?dry_run=true`: reports what a batch WOULD do without
/// writing anything — no context created, no marker opened, no source
/// retracted. Runs the same [`predicted_alias_rejection`] and
/// [`predicted_schema_rejection`] checks first, in the same order
/// `apply_batch` does, so a batch whose aliases would conflict or whose
/// associations would violate the schema is refused here exactly as it
/// would be by `apply_batch` — the two entrances can never disagree on
/// either call. Every other write step in `apply_batch` has a cheap
/// read-only counterpart here, except the `associations` and `aliases`
/// counts, which stay OPTIMISTIC (every op this batch carries,
/// corrected the same way `apply_batch` corrects them): a capacity cap
/// (507) can only surface by actually applying the op, so those two
/// COUNTS remain advisory even though an alias CONFLICT or a schema
/// VIOLATION no longer is — the real import can still apply fewer
/// associations or aliases than previewed. Every other field
/// (`retracted`, the drop counts, `schema_violations`) reads through to
/// the same state a real batch would query, so it matches exactly.
pub(crate) fn preview_batch(state: &AppState, batch: &Batch) -> Result<Applied, ApplyRefusal> {
    if let Some(rejection) = predicted_alias_rejection(state, batch) {
        return Err(ApplyRefusal::Rejected(rejection));
    }
    let schema_warnings = predicted_schema_rejection(state, batch, CheckPurpose::Preview)?;

    let created = state.directory_entry(&batch.context).is_none();
    if created && batch.create.is_none() {
        return Err(ApplyRefusal::NoContext(batch.context.clone()));
    }

    // A context about to be created has nothing to retract from yet.
    let retracted = if created {
        0
    } else {
        state
            .count_source_edges(&batch.context, &batch.source)
            .map_err(ApplyRefusal::Access)?
    };
    // Mirrors apply_batch's tolerance for a passage-store read that
    // fails: retract_source warns and reports no removal rather than
    // failing the whole batch, so the preview falls back the same way.
    let had_passage = state
        .passage_sources(&batch.context)
        .and_then(Result::ok)
        .is_some_and(|sources| sources.contains(&batch.source));
    let passage_dropped = had_passage && batch.passage.is_none();

    let paragraph_count = batch
        .passage
        .as_deref()
        .map(|text| crate::paragraph::split(text).len());
    let (questions_dropped, sections_dropped, locators_dropped) = match paragraph_count {
        Some(paragraph_count) => crate::passages::preview_drops(
            paragraph_count,
            &batch.questions,
            &batch.sections,
            &batch.locators,
        ),
        None => (0, 0, 0),
    };

    let (corrected, association_paragraphs_dropped) =
        corrected_associations(batch, paragraph_count);

    Ok(Applied {
        created,
        retracted,
        associations: corrected.len(),
        aliases: batch.concepts.len() + batch.labels.len(),
        passage_stored: batch.passage.is_some(),
        passage_dropped,
        questions_stored: batch.questions.len() - questions_dropped,
        questions_dropped,
        sections_stored: batch.sections.len() - sections_dropped,
        sections_dropped,
        locators_stored: batch.locators.len() - locators_dropped,
        locators_dropped,
        association_paragraphs_dropped,
        schema_violations: schema_warnings.total,
        schema_issues: schema_warnings.issues,
    })
}

/// The CLI's per-file report line.
fn report(batch: &Batch, applied: &Applied) -> String {
    format!(
        "context '{}'{} ← source '{}' ({} association(s) retracted): +{} \
         association(s), +{} alias(es){}{}{}{}{}{}",
        batch.context,
        if applied.created { " (created)" } else { "" },
        batch.source,
        applied.retracted,
        applied.associations,
        applied.aliases,
        match (applied.passage_stored, applied.passage_dropped) {
            (true, _) => ", passage stored",
            (false, true) => ", previous passage dropped (batch carried none)",
            (false, false) => "",
        },
        match (applied.questions_stored, applied.questions_dropped) {
            (0, 0) => String::new(),
            (stored, 0) => format!(", +{stored} question(s)"),
            (stored, dropped) => {
                format!(", +{stored} question(s) ({dropped} dropped: no such paragraph)")
            }
        },
        match (applied.sections_stored, applied.sections_dropped) {
            (0, 0) => String::new(),
            (stored, 0) => format!(", +{stored} section(s)"),
            (stored, dropped) => {
                format!(", +{stored} section(s) ({dropped} dropped: no such paragraph)")
            }
        },
        match (applied.locators_stored, applied.locators_dropped) {
            (0, 0) => String::new(),
            (stored, 0) => format!(", +{stored} locator(s)"),
            (stored, dropped) => {
                format!(", +{stored} locator(s) ({dropped} dropped: no such paragraph)")
            }
        },
        match applied.association_paragraphs_dropped {
            0 => String::new(),
            dropped => {
                format!(", {dropped} association paragraph locator(s) dropped: no such paragraph")
            }
        },
        match applied.schema_violations {
            0 => String::new(),
            violations => format!(", schema warnings: {violations}"),
        }
    )
}

/// Import (and export, which shares the need) logs like the server
/// does (RUST_LOG, stderr) so registry warnings — WAL replay notes,
/// load failures — are not dropped on the floor, but stdout stays
/// pure report.
pub(crate) fn init_logging() {
    let filter = tracing_subscriber::EnvFilter::try_from_env("RUST_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_usage_variable_is_a_known_key() {
        // This command's own USAGE is invisible to cli.rs's
        // consistency tests: a variable documented here but missing
        // from KNOWN_KEYS would make --config warn "typo?" on a
        // perfectly valid setting.
        crate::config::assert_usage_vars_are_known_keys(USAGE);
    }

    fn parse(text: &str) -> Result<Batch, String> {
        parse_batch(std::io::Cursor::new(text))
    }

    const HEADER: &str = r#"{"taguru_batch": 1, "context": "sake", "source": "doc-1"}"#;

    #[test]
    fn split_batches_slices_exactly_the_bytes_between_stream_level_records() {
        let body = concat!(
            "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s1\"}\n",
            "{\"assoc\": [\"a\", \"likes\", \"b\"]}\n",
            "\n",
            "{\"taguru_group\": 1, \"name\": \"g\", \"contexts\": [\"sake\"]}\n",
            "{\"taguru_batch\": 1, \"context\": \"beer\", \"source\": \"s2\"}\n",
            "{\"assoc\": [\"c\", \"likes\", \"d\"]}",
        )
        .as_bytes();
        let ranges = split_batches(body);
        assert_eq!(ranges.len(), 2);
        let first = std::str::from_utf8(&body[ranges[0].clone()]).unwrap();
        assert!(first.starts_with("{\"taguru_batch\": 1, \"context\": \"sake\""));
        // The batch's ops (and the blank line) ride along; the group
        // record between the batches belongs to neither.
        assert!(first.contains("likes"));
        assert!(!first.contains("taguru_group"));
        let second = std::str::from_utf8(&body[ranges[1].clone()]).unwrap();
        assert!(second.starts_with("{\"taguru_batch\": 1, \"context\": \"beer\""));
        assert!(second.ends_with("\"d\"]}"), "EOF closes the last batch");
    }

    /// [`split_batches_slices_exactly_the_bytes_between_stream_level_records`]'s
    /// `taguru_schema` case: a schema record between two batches
    /// belongs to neither, the same as a group record.
    #[test]
    fn split_batches_excludes_a_schema_record_from_either_adjacent_batch() {
        let body = format!(
            "{{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"s1\"}}\n\
             {{\"assoc\": [\"a\", \"likes\", \"b\"]}}\n\
             {SCHEMA_LINE}\n\
             {{\"taguru_batch\": 1, \"context\": \"beer\", \"source\": \"s2\"}}\n\
             {{\"assoc\": [\"c\", \"likes\", \"d\"]}}"
        );
        let body = body.as_bytes();
        let ranges = split_batches(body);
        assert_eq!(ranges.len(), 2);
        let first = std::str::from_utf8(&body[ranges[0].clone()]).unwrap();
        assert!(first.contains("likes"));
        assert!(!first.contains("taguru_schema"));
        let second = std::str::from_utf8(&body[ranges[1].clone()]).unwrap();
        assert!(second.starts_with("{\"taguru_batch\": 1, \"context\": \"beer\""));
    }

    #[test]
    fn a_batch_parses_and_the_header_source_stamps_every_association() {
        let batch = parse(&format!(
            "{HEADER}\n\
             {{\"subject\": \"青嶺酒造\", \"label\": \"創業年\", \"object\": \"1907年\", \"weight\": 2.0}}\n\
             \n\
             {{\"alias\": \"Aomine\", \"canonical\": \"青嶺酒造\", \"kind\": \"concept\"}}\n\
             {{\"alias\": \"設立年\", \"canonical\": \"創業年\", \"kind\": \"label\"}}\n\
             {{\"passage\": \"青嶺酒造は1907年創業。\"}}\n"
        ))
        .unwrap();
        assert_eq!(batch.context, "sake");
        assert_eq!(batch.associations.len(), 1);
        assert_eq!(batch.associations[0].source.as_deref(), Some("doc-1"));
        assert_eq!(batch.concepts.len(), 1);
        assert_eq!(batch.labels.len(), 1);
        assert_eq!(batch.passage.as_deref(), Some("青嶺酒造は1907年創業。"));
        assert_eq!(batch.op_count(), 3);
    }

    #[test]
    fn an_association_carrying_its_own_source_is_refused_by_line_number() {
        let error = parse(&format!(
            "{HEADER}\n\
             {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0, \
              \"source\": \"rogue\"}}\n"
        ))
        .unwrap_err();
        assert!(error.contains("line 2"), "{error}");
        assert!(error.contains("source"), "{error}");
    }

    #[test]
    fn the_first_line_must_be_a_header_of_a_readable_version() {
        let error =
            parse("{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}\n")
                .unwrap_err();
        assert!(error.contains("not a batch header"), "{error}");

        let error =
            parse("{\"taguru_batch\": 2, \"context\": \"c\", \"source\": \"s\"}\n").unwrap_err();
        assert!(error.contains("taguru_batch 2"), "{error}");

        assert!(parse("\n\n").unwrap_err().contains("empty file"));
    }

    /// Notepad and other Windows editors stamp a UTF-8 BOM onto every
    /// file they save; left in place it rides onto '{' as the first
    /// byte of the header line and fails to parse as JSON at all, with
    /// nothing in the error pointing at what actually went wrong.
    #[test]
    fn a_leading_bom_does_not_break_the_first_line() {
        let batch = parse(&format!("\u{FEFF}{HEADER}\n")).unwrap();
        assert_eq!(batch.context, "sake");
        assert_eq!(batch.source, "doc-1");
    }

    #[test]
    fn a_stream_of_batches_parses_with_per_batch_state() {
        let batches = parse_stream(std::io::Cursor::new(format!(
            "{HEADER}\n\
             {{\"passage\": \"第1段落。\"}}\n\
             {{\"paragraph\": 0, \"question\": \"何?\"}}\n\
             {{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-2\"}}\n\
             {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n"
        )))
        .unwrap()
        .batches;
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].source, "doc-1");
        assert_eq!(batches[0].questions.len(), 1);
        assert_eq!(batches[1].source, "doc-2");
        // Per-batch validation still applies at each boundary: the
        // second batch carries no passage, so its questions would have
        // refused — and doc-1's question must not leak into doc-2.
        assert!(batches[1].questions.is_empty());
        assert_eq!(batches[1].associations[0].source.as_deref(), Some("doc-2"));
    }

    #[test]
    fn a_stream_restating_one_source_is_refused() {
        let error = parse_stream(std::io::Cursor::new(format!(
            "{HEADER}\n\
             {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n\
             {HEADER}\n"
        )))
        .unwrap_err();
        assert!(
            error.contains("line 3") && error.contains("one batch owns one source's truth"),
            "{error}"
        );
    }

    #[test]
    fn a_batch_boundary_runs_the_finish_validations() {
        // The FIRST batch is the broken one (a question with no
        // passage); the boundary — not the end of the stream — must
        // catch it.
        let error = parse_stream(std::io::Cursor::new(format!(
            "{HEADER}\n\
             {{\"paragraph\": 0, \"question\": \"何?\"}}\n\
             {{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-2\"}}\n\
             {{\"passage\": \"本文。\"}}\n"
        )))
        .unwrap_err();
        assert!(error.contains("question"), "{error}");
    }

    #[test]
    fn parse_batch_refuses_a_multi_batch_stream() {
        let error = parse(&format!(
            "{HEADER}\n\
             {{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-2\"}}\n"
        ))
        .unwrap_err();
        assert!(error.contains("exactly one"), "{error}");
    }

    #[test]
    fn duplicate_aliases_and_second_passages_are_refused() {
        let error = parse(&format!(
            "{HEADER}\n\
             {{\"alias\": \"a\", \"canonical\": \"x\", \"kind\": \"concept\"}}\n\
             {{\"alias\": \"a\", \"canonical\": \"y\", \"kind\": \"concept\"}}\n"
        ))
        .unwrap_err();
        assert!(
            error.contains("line 3") && error.contains("twice"),
            "{error}"
        );

        let error = parse(&format!(
            "{HEADER}\n{{\"passage\": \"one\"}}\n{{\"passage\": \"two\"}}\n"
        ))
        .unwrap_err();
        assert!(
            error.contains("line 3") && error.contains("passage"),
            "{error}"
        );
    }

    /// An empty alias spelling would containment-match every future
    /// cue (`str::contains("")` is always true) — the import surface
    /// must refuse it just like the HTTP one does.
    #[test]
    fn empty_alias_spellings_are_refused() {
        for line in [
            "{\"alias\": \"\", \"canonical\": \"x\", \"kind\": \"concept\"}",
            "{\"alias\": \"a\", \"canonical\": \"\", \"kind\": \"label\"}",
        ] {
            let error = parse(&format!("{HEADER}\n{line}\n")).unwrap_err();
            assert!(
                error.contains("line 2") && error.contains("must not be empty"),
                "{error}"
            );
        }
    }

    /// Empty question or section text is refused like empty names: a
    /// question row is embedded verbatim on the next refresh, and
    /// providers refuse zero-length input — one empty row would fail
    /// its whole chunk (and abandon the pass) on every refresh.
    #[test]
    fn empty_question_and_section_text_is_refused() {
        for line in [
            "{\"paragraph\": 0, \"question\": \"\"}",
            "{\"paragraph\": 0, \"section\": \"\"}",
        ] {
            let error =
                parse(&format!("{HEADER}\n{{\"passage\": \"本文。\"}}\n{line}\n")).unwrap_err();
            assert!(
                error.contains("line 3") && error.contains("must not be empty"),
                "{error}"
            );
        }
    }

    /// An empty context name would `file_stem` to a bare `.ctx` the
    /// server's directory scan never rediscovers; an empty source name
    /// has no identity to retract a re-import against. Both are refused
    /// at the header, each naming its own field.
    #[test]
    fn an_empty_context_or_source_name_in_the_header_is_refused() {
        for (field, header) in [
            (
                "context",
                r#"{"taguru_batch": 1, "context": "", "source": "s"}"#,
            ),
            (
                "source",
                r#"{"taguru_batch": 1, "context": "c", "source": ""}"#,
            ),
        ] {
            let error = parse(header).unwrap_err();
            assert!(
                error.contains(field) && error.contains("must not be empty"),
                "{field}: {error}"
            );
        }
    }

    #[test]
    fn group_records_ride_a_stream_and_stand_alone() {
        let stream = parse_stream(std::io::Cursor::new(format!(
            "{HEADER}\n\
             {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n\
             {{\"taguru_group\": 1, \"name\": \"kura\", \"description\": \"蔵\", \
               \"contexts\": [\"sake\", \"sake\"], \"groups\": [\"kid\"]}}\n\
             {{\"taguru_group\": 1, \"name\": \"kid\"}}\n"
        )))
        .unwrap();
        assert_eq!(stream.batches.len(), 1);
        assert_eq!(stream.groups.len(), 2);
        let (name, record) = &stream.groups[0];
        assert_eq!(name, "kura");
        assert_eq!(record.description, "蔵");
        // List duplicates fold into the set — membership IS a set,
        // exactly as over the API.
        assert_eq!(record.contexts.len(), 1);
        assert_eq!(record.groups.len(), 1);
        // Absent fields read as empty, the shape export omits.
        assert_eq!(stream.groups[1].1, GroupRecord::default());

        // A group record closes the batch before it: an op line after
        // one has no batch to join.
        let error = parse_stream(std::io::Cursor::new(format!(
            "{HEADER}\n\
             {{\"taguru_group\": 1, \"name\": \"kura\"}}\n\
             {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n"
        )))
        .unwrap_err();
        assert!(
            error.contains("line 3") && error.contains("not a batch header"),
            "{error}"
        );

        // A groups-only stream is a legitimate restore; an empty one is
        // still a mistake.
        let alone = parse_stream(std::io::Cursor::new(
            "{\"taguru_group\": 1, \"name\": \"kura\"}\n",
        ))
        .unwrap();
        assert!(alone.batches.is_empty());
        assert_eq!(alone.groups.len(), 1);
        assert!(
            parse_stream(std::io::Cursor::new("\n"))
                .unwrap_err()
                .contains("group record")
        );
    }

    #[test]
    fn group_records_validate_their_shape_with_line_numbers() {
        let case =
            |line: &str| parse_stream(std::io::Cursor::new(format!("{line}\n"))).unwrap_err();
        assert!(case("{\"taguru_group\": 2, \"name\": \"g\"}").contains("taguru_group 2"));
        assert!(case("{\"taguru_group\": 1, \"name\": \"\"}").contains("must not be empty"));
        assert!(
            case("{\"taguru_group\": 1, \"name\": \"g\", \"nope\": 1}").contains("unknown field")
        );
        let long = "x".repeat(65);
        assert!(
            case(&format!("{{\"taguru_group\": 1, \"name\": \"{long}\"}}")).contains("65 bytes")
        );
        assert!(
            case(&format!(
                "{{\"taguru_group\": 1, \"name\": \"g\", \"contexts\": [\"{long}\"]}}"
            ))
            .contains("65 bytes")
        );

        // Restating one group refuses the whole stream, by line.
        let error = parse_stream(std::io::Cursor::new(
            "{\"taguru_group\": 1, \"name\": \"g\"}\n{\"taguru_group\": 1, \"name\": \"g\"}\n",
        ))
        .unwrap_err();
        assert!(
            error.contains("line 2") && error.contains("one record owns one group's truth"),
            "{error}"
        );

        // The member cap judges the SET: one name past it refuses.
        let over_set: String = (0..=MAX_GROUP_MEMBERS)
            .map(|i| format!("\"c{i:04}\""))
            .collect::<Vec<_>>()
            .join(", ");
        let error = case(&format!(
            "{{\"taguru_group\": 1, \"name\": \"g\", \"contexts\": [{over_set}]}}"
        ));
        assert!(error.contains("split into nested child groups"), "{error}");
    }

    /// The single-batch entrance (`taguru extract` re-validating its
    /// own output) never carries group records.
    #[test]
    fn parse_batch_refuses_group_records() {
        let error = parse(
            "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\"}\n\
             {\"taguru_group\": 1, \"name\": \"kura\"}\n",
        )
        .unwrap_err();
        assert!(error.contains("exactly one batch was expected"), "{error}");
    }

    const SCHEMA_LINE: &str = r#"{"taguru_schema": 1, "context": "sake", "mode": "warn", "closed_labels": false, "types": {}, "relations": {}}"#;

    /// `taguru_schema` records ride a stream and stand alone — the
    /// schema twin of [`group_records_ride_a_stream_and_stand_alone`]
    /// (ADR 0009 §13). A schema record closes the batch before it, an
    /// op line after one has no batch to join, a schema-only stream is
    /// a legitimate restore, and the empty-stream message now names
    /// all three record kinds.
    #[test]
    fn schema_records_ride_a_stream_and_stand_alone() {
        let stream = parse_stream(std::io::Cursor::new(format!(
            "{HEADER}\n\
             {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n\
             {SCHEMA_LINE}\n\
             {{\"taguru_group\": 1, \"name\": \"kid\"}}\n"
        )))
        .unwrap();
        assert_eq!(stream.batches.len(), 1);
        assert_eq!(stream.schemas.len(), 1);
        assert_eq!(stream.groups.len(), 1);
        let (context, installed) = &stream.schemas[0];
        assert_eq!(context, "sake");
        assert_eq!(installed.document().mode, crate::schema::SchemaMode::Warn);

        // A schema record closes the batch before it: an op line after
        // one has no batch to join.
        let error = parse_stream(std::io::Cursor::new(format!(
            "{HEADER}\n\
             {SCHEMA_LINE}\n\
             {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n"
        )))
        .unwrap_err();
        assert!(
            error.contains("line 3") && error.contains("not a batch header"),
            "{error}"
        );

        // A schemas-only stream is a legitimate restore.
        let alone = parse_stream(std::io::Cursor::new(format!("{SCHEMA_LINE}\n"))).unwrap();
        assert!(alone.batches.is_empty());
        assert_eq!(alone.schemas.len(), 1);

        // The empty-stream message now names every record kind.
        let error = parse_stream(std::io::Cursor::new("\n")).unwrap_err();
        assert!(error.contains("schema record"), "{error}");
        assert!(error.contains("group record"), "{error}");
    }

    /// The version-refusal wording, `deny_unknown_fields`, a missing
    /// field, an empty context, a cross-record duplicate, and a
    /// `schema::install`-level violation — the schema twin of
    /// [`group_records_validate_their_shape_with_line_numbers`].
    #[test]
    fn schema_records_validate_their_shape_with_line_numbers() {
        let case =
            |line: &str| parse_stream(std::io::Cursor::new(format!("{line}\n"))).unwrap_err();

        // parse_group's exact wording shape (ADR 0009 §13 bullet 4).
        assert!(
            case(
                r#"{"taguru_schema": 2, "context": "sake", "mode": "off", "closed_labels": false, "types": {}, "relations": {}}"#
            )
            .contains("taguru_schema 2 is not a version this taguru reads (it reads 1)")
        );

        assert!(
            case(
                r#"{"taguru_schema": 1, "context": "", "mode": "off", "closed_labels": false, "types": {}, "relations": {}}"#
            )
            .contains("must not be empty")
        );

        // The context name's own byte cap — mirrors
        // `group_records_validate_their_shape_with_line_numbers`'s
        // `long` case for a group's `name`.
        let long = "x".repeat(65);
        assert!(
            case(&format!(
                r#"{{"taguru_schema": 1, "context": "{long}", "mode": "off", "closed_labels": false, "types": {{}}, "relations": {{}}}}"#
            ))
            .contains("65 bytes")
        );

        // Every field is required — no struct-level default, matching
        // SchemaDocument's own at-rest posture.
        assert!(
            case(r#"{"taguru_schema": 1, "context": "sake", "mode": "off"}"#)
                .contains("missing field")
        );

        assert!(
            case(
                r#"{"taguru_schema": 1, "context": "sake", "mode": "off", "closed_labels": false, "types": {}, "relations": {}, "nope": 1}"#
            )
            .contains("unknown field")
        );

        // A structural violation `schema::install` itself catches
        // (here: the relation named the reserved type label) surfaces
        // with the line number, not just the bare violation text.
        let error = case(
            r#"{"taguru_schema": 1, "context": "sake", "mode": "off", "closed_labels": false, "types": {}, "relations": {"schema:type": {}}}"#,
        );
        assert!(
            error.contains("line 1") && error.contains("reserved"),
            "{error}"
        );

        // Restating one context's schema refuses the whole stream, by
        // line — mirrors a group record's own duplicate refusal.
        let error = parse_stream(std::io::Cursor::new(format!(
            "{SCHEMA_LINE}\n{SCHEMA_LINE}\n"
        )))
        .unwrap_err();
        assert!(
            error.contains("line 2") && error.contains("one record owns one context's schema"),
            "{error}"
        );
    }

    /// The single-batch entrance (`taguru extract` re-validating its
    /// own output) never carries schema records either.
    #[test]
    fn parse_batch_refuses_schema_records() {
        let error = parse(&format!("{HEADER}\n{SCHEMA_LINE}\n")).unwrap_err();
        assert!(
            error.contains("schema record for context 'sake'")
                && error.contains("exactly one batch was expected"),
            "{error}"
        );
    }

    /// [`apply_schema_record`]'s own failure path, not just
    /// `parse_schema`'s validation: a schema record naming a context
    /// neither an earlier batch of the same stream nor a previous
    /// request ever created returns [`SchemaApplyError::NoContext`] —
    /// the CLI-specific arm `run_local`'s Pass 2 counts into
    /// `schema_failures` (exit 1), and the server twin
    /// `schema_import_refusal` maps to 404 `no_context`.
    #[test]
    fn apply_schema_record_refuses_a_context_that_does_not_exist() {
        let dir = std::env::temp_dir().join(format!(
            "taguru-ingest-schema-no-context-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

        let installed = schema::install(schema::SchemaDocument {
            schema: schema::SCHEMA_VERSION,
            mode: schema::SchemaMode::Off,
            closed_labels: false,
            types: BTreeMap::new(),
            relations: BTreeMap::new(),
        })
        .unwrap();
        let error = apply_schema_record(&state, "ghost", installed).unwrap_err();
        assert!(matches!(error, SchemaApplyError::NoContext), "{error:?}");

        let _ = fs::remove_dir_all(&dir);
    }

    /// A line longer than the cap is refused at the cap, not buffered
    /// whole first: the bounded reader stops one byte past the ceiling,
    /// so a malicious 100 MiB line cannot force a 100 MiB allocation
    /// before the length check runs.
    #[test]
    fn a_line_past_the_byte_cap_is_refused_without_buffering_it_whole() {
        let giant = "x".repeat(MAX_LINE_BYTES + 1);
        let error = parse(&format!("{HEADER}\n{giant}")).unwrap_err();
        assert!(error.contains("line cap"), "{error}");
    }

    /// Source metadata (#167) rides the passage line; a pre-metadata
    /// line still parses (all three fields default), and the tag
    /// vocabulary is the same one the HTTP store enforces.
    #[test]
    fn passage_line_metadata_parses_validates_and_defaults() {
        let batch = parse(&format!(
            "{HEADER}\n\
             {{\"passage\": \"本文。\", \"stored_at\": 1700000000, \"date\": 1000, \
              \"tags\": [\"酒\", \"蔵\"]}}\n"
        ))
        .unwrap();
        assert_eq!(batch.stored_at, Some(1_700_000_000));
        assert_eq!(batch.date, Some(1_000));
        assert_eq!(batch.tags, vec!["酒".to_string(), "蔵".to_string()]);

        let bare = parse(&format!("{HEADER}\n{{\"passage\": \"本文。\"}}\n")).unwrap();
        assert_eq!(
            (bare.stored_at, bare.date, bare.tags.len()),
            (None, None, 0)
        );

        let empty = parse(&format!(
            "{HEADER}\n{{\"passage\": \"本文。\", \"tags\": [\"\"]}}\n"
        ))
        .unwrap_err();
        assert!(empty.contains("tag"), "{empty}");
        let oversized = parse(&format!(
            "{HEADER}\n{{\"passage\": \"本文。\", \"tags\": [\"{}\"]}}\n",
            "t".repeat(crate::api::MAX_TAG_BYTES + 1)
        ))
        .unwrap_err();
        assert!(oversized.contains("exceeds"), "{oversized}");
        let too_many = parse(&format!(
            "{HEADER}\n{{\"passage\": \"本文。\", \"tags\": [{}]}}\n",
            (0..=crate::api::MAX_TAGS_PER_SOURCE)
                .map(|i| format!("\"t{i}\""))
                .collect::<Vec<_>>()
                .join(", ")
        ))
        .unwrap_err();
        assert!(too_many.contains("at most"), "{too_many}");
    }

    #[test]
    fn a_question_line_rides_the_batch_and_needs_a_passage_to_attach_to() {
        let batch = parse(&format!(
            "{HEADER}\n\
             {{\"passage\": \"一つ目。\\n\\n二つ目。\"}}\n\
             {{\"paragraph\": 1, \"question\": \"二つ目は何?\"}}\n"
        ))
        .unwrap();
        assert_eq!(batch.questions, vec![(1, "二つ目は何?".to_string())]);
        assert!(
            batch.describe().contains("1 question(s)"),
            "{}",
            batch.describe()
        );

        // The same question line without a passage has nothing to name.
        let error = parse(&format!(
            "{HEADER}\n{{\"paragraph\": 1, \"question\": \"二つ目は何?\"}}\n"
        ))
        .unwrap_err();
        assert!(error.contains("no passage line"), "{error}");
    }

    #[test]
    fn more_than_the_per_paragraph_question_cap_in_one_file_is_refused() {
        let questions: String = (0..=crate::api::MAX_QUESTIONS_PER_PARAGRAPH)
            .map(|i| format!("{{\"paragraph\": 0, \"question\": \"言い換え{i}?\"}}\n"))
            .collect();
        let error = parse(&format!(
            "{HEADER}\n{{\"passage\": \"本文。\"}}\n{questions}"
        ))
        .unwrap_err();
        assert!(
            error.contains("already carries") && error.contains("the cap"),
            "{error}"
        );

        let long = "q".repeat(crate::api::MAX_QUESTION_BYTES + 1);
        let error = parse(&format!(
            "{HEADER}\n{{\"passage\": \"本文。\"}}\n{{\"paragraph\": 0, \"question\": \"{long}\"}}\n"
        ))
        .unwrap_err();
        assert!(
            error.contains("question") && error.contains("cap"),
            "{error}"
        );
    }

    /// A doc2query generator repeating itself, or a batch author pasting
    /// the same line twice, must not burn two of the paragraph's capped
    /// slots on text that says nothing new — it folds into one entry,
    /// matching the group-list dedup elsewhere in this file.
    #[test]
    fn a_repeated_question_on_the_same_paragraph_folds_into_one_entry() {
        let batch = parse(&format!(
            "{HEADER}\n\
             {{\"passage\": \"本文。\"}}\n\
             {{\"paragraph\": 0, \"question\": \"何?\"}}\n\
             {{\"paragraph\": 0, \"question\": \"何?\"}}\n"
        ))
        .unwrap();
        assert_eq!(batch.questions, vec![(0, "何?".to_string())]);

        // The repeat must not spend one of the paragraph's capped slots
        // either: MAX_QUESTIONS_PER_PARAGRAPH distinct questions plus one
        // repeat of the first must still fit under the cap.
        let distinct: String = (0..crate::api::MAX_QUESTIONS_PER_PARAGRAPH)
            .map(|i| format!("{{\"paragraph\": 0, \"question\": \"言い換え{i}?\"}}\n"))
            .collect();
        let batch = parse(&format!(
            "{HEADER}\n{{\"passage\": \"本文。\"}}\n{distinct}\
             {{\"paragraph\": 0, \"question\": \"言い換え0?\"}}\n"
        ))
        .unwrap();
        assert_eq!(
            batch.questions.len(),
            crate::api::MAX_QUESTIONS_PER_PARAGRAPH
        );
    }

    #[test]
    fn a_section_line_rides_the_batch_and_needs_a_passage_to_attach_to() {
        let batch = parse(&format!(
            "{HEADER}\n\
             {{\"passage\": \"導入。\\n\\n本編。\"}}\n\
             {{\"paragraph\": 1, \"section\": \"本編\"}}\n"
        ))
        .unwrap();
        assert_eq!(batch.sections, vec![(1, "本編".to_string())]);
        assert!(
            batch.describe().contains("1 section(s)"),
            "{}",
            batch.describe()
        );

        // The same section line without a passage has nothing to name.
        let error = parse(&format!(
            "{HEADER}\n{{\"paragraph\": 1, \"section\": \"本編\"}}\n"
        ))
        .unwrap_err();
        assert!(error.contains("no passage line"), "{error}");
    }

    #[test]
    fn a_locator_line_rides_the_batch_and_needs_a_passage_to_attach_to() {
        let batch = parse(&format!(
            "{HEADER}\n\
             {{\"passage\": \"導入。\\n\\n本編。\"}}\n\
             {{\"paragraph\": 1, \"locator\": {{\"kind\": \"page\", \"value\": \"12\"}}}}\n"
        ))
        .unwrap();
        assert_eq!(
            batch.locators,
            vec![(
                1,
                crate::passages::Locator {
                    kind: "page".to_string(),
                    value: "12".to_string(),
                }
            )]
        );
        assert!(
            batch.describe().contains("1 locator(s)"),
            "{}",
            batch.describe()
        );

        // The same locator line without a passage has nothing to name.
        let error = parse(&format!(
            "{HEADER}\n{{\"paragraph\": 1, \"locator\": {{\"kind\": \"page\", \"value\": \"12\"}}}}\n"
        ))
        .unwrap_err();
        assert!(error.contains("no passage line"), "{error}");
    }

    #[test]
    fn an_association_with_a_paragraph_needs_a_passage_to_attach_to() {
        // A paragraph locator on an association resolves against THIS
        // batch's passage, so it parses fine when the passage is present.
        let batch = parse(&format!(
            "{HEADER}\n\
             {{\"passage\": \"導入。\\n\\n本編。\"}}\n\
             {{\"subject\": \"青嶺酒造\", \"label\": \"創業年\", \"object\": \"1907年\", \"weight\": 1.0, \"paragraph\": 1}}\n"
        ))
        .unwrap();
        assert_eq!(batch.associations[0].paragraph, Some(1));

        // The same locator with no passage line has nothing to name, and
        // apply retracts the source first — so it must be refused rather
        // than persisted into a passage that will not exist.
        let error = parse(&format!(
            "{HEADER}\n\
             {{\"subject\": \"青嶺酒造\", \"label\": \"創業年\", \"object\": \"1907年\", \"weight\": 1.0, \"paragraph\": 1}}\n"
        ))
        .unwrap_err();
        assert!(error.contains("no passage line"), "{error}");

        // A plain association (no locator) still stands on its own.
        parse(&format!(
            "{HEADER}\n\
             {{\"subject\": \"青嶺酒造\", \"label\": \"創業年\", \"object\": \"1907年\", \"weight\": 1.0}}\n"
        ))
        .unwrap();
    }

    #[test]
    fn report_surfaces_a_dropped_passage_that_was_not_replaced() {
        let batch = parse(HEADER).unwrap();

        // A passage was retracted and the batch brought no replacement:
        // the loss must show in the report, never vanish silently.
        let dropped = Applied {
            created: false,
            retracted: 3,
            associations: 0,
            aliases: 0,
            passage_stored: false,
            passage_dropped: true,
            questions_stored: 0,
            questions_dropped: 0,
            sections_stored: 0,
            sections_dropped: 0,
            locators_stored: 0,
            locators_dropped: 0,
            association_paragraphs_dropped: 0,
            schema_violations: 0,
            schema_issues: Vec::new(),
        };
        let line = report(&batch, &dropped);
        assert!(line.contains("previous passage dropped"), "{line}");

        // A batch that carries a replacement reads as a store, not a
        // drop, even though the prior passage was removed to make room.
        let replaced = Applied {
            passage_stored: true,
            ..dropped
        };
        let line = report(&batch, &replaced);
        assert!(line.contains("passage stored"), "{line}");
        assert!(!line.contains("dropped"), "{line}");
    }

    #[test]
    fn a_section_beyond_the_byte_cap_is_refused() {
        let long = "s".repeat(crate::api::MAX_SECTION_BYTES + 1);
        let error = parse(&format!(
            "{HEADER}\n{{\"passage\": \"本文。\"}}\n{{\"paragraph\": 0, \"section\": \"{long}\"}}\n"
        ))
        .unwrap_err();
        assert!(
            error.contains("section") && error.contains("cap"),
            "{error}"
        );
    }

    #[test]
    fn a_malformed_section_line_is_refused_by_line_number() {
        let error = parse(&format!(
            "{HEADER}\n{{\"passage\": \"本文。\"}}\n{{\"paragraph\": \"zero\", \"section\": \"見出し\"}}\n"
        ))
        .unwrap_err();
        assert!(
            error.contains("line 3") && error.contains("section"),
            "{error}"
        );
    }

    #[test]
    fn a_locator_value_beyond_the_byte_cap_is_refused() {
        let long = "s".repeat(crate::api::MAX_LOCATOR_VALUE_BYTES + 1);
        let error = parse(&format!(
            "{HEADER}\n{{\"passage\": \"本文。\"}}\n\
             {{\"paragraph\": 0, \"locator\": {{\"kind\": \"page\", \"value\": \"{long}\"}}}}\n"
        ))
        .unwrap_err();
        assert!(
            error.contains("locator.value") && error.contains("cap"),
            "{error}"
        );
    }

    #[test]
    fn a_locator_kind_beyond_the_byte_cap_is_refused() {
        let long = "k".repeat(crate::api::MAX_LOCATOR_KIND_BYTES + 1);
        let error = parse(&format!(
            "{HEADER}\n{{\"passage\": \"本文。\"}}\n\
             {{\"paragraph\": 0, \"locator\": {{\"kind\": \"{long}\", \"value\": \"1\"}}}}\n"
        ))
        .unwrap_err();
        assert!(
            error.contains("locator.kind") && error.contains("cap"),
            "{error}"
        );
    }

    #[test]
    fn an_empty_locator_kind_or_value_is_refused() {
        for locator in [
            r#"{"kind": "", "value": "1"}"#,
            r#"{"kind": "page", "value": ""}"#,
        ] {
            let error = parse(&format!(
                "{HEADER}\n{{\"passage\": \"本文。\"}}\n\
                 {{\"paragraph\": 0, \"locator\": {locator}}}\n"
            ))
            .unwrap_err();
            assert!(
                error.contains("line 3") && error.contains("must not be empty"),
                "{error}"
            );
        }
    }

    #[test]
    fn a_malformed_locator_line_is_refused_by_line_number() {
        let error = parse(&format!(
            "{HEADER}\n{{\"passage\": \"本文。\"}}\n\
             {{\"paragraph\": \"zero\", \"locator\": {{\"kind\": \"page\", \"value\": \"1\"}}}}\n"
        ))
        .unwrap_err();
        assert!(
            error.contains("line 3") && error.contains("locator"),
            "{error}"
        );
    }

    #[test]
    fn a_locator_line_with_an_unknown_field_is_refused() {
        let error = parse(&format!(
            "{HEADER}\n{{\"passage\": \"本文。\"}}\n\
             {{\"paragraph\": 0, \"locator\": {{\"kind\": \"page\", \"value\": \"1\", \"page\": 1}}}}\n"
        ))
        .unwrap_err();
        assert!(error.contains("line 3"), "{error}");
    }

    #[test]
    fn weights_and_name_sizes_are_capped_like_the_api() {
        let error = parse(&format!(
            "{HEADER}\n\
             {{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1e300}}\n"
        ))
        .unwrap_err();
        assert!(
            error.contains("line 2") && error.contains("weight"),
            "{error}"
        );

        let long = "x".repeat(MAX_NAME_BYTES + 1);
        let error = parse(&format!(
            "{HEADER}\n\
             {{\"subject\": \"{long}\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n"
        ))
        .unwrap_err();
        assert!(
            error.contains("line 2") && error.contains("subject"),
            "{error}"
        );

        let error = parse(&format!(
            "{{\"taguru_batch\": 1, \"context\": \"{}\", \"source\": \"s\"}}\n",
            "c".repeat(MAX_CONTEXT_NAME_BYTES + 1)
        ))
        .unwrap_err();
        assert!(error.contains("context"), "{error}");
    }

    #[test]
    fn empty_subject_label_or_object_is_refused() {
        for (field, line) in [
            (
                "subject",
                r#"{"subject": "", "label": "l", "object": "b", "weight": 1.0}"#,
            ),
            (
                "label",
                r#"{"subject": "a", "label": "", "object": "b", "weight": 1.0}"#,
            ),
            (
                "object",
                r#"{"subject": "a", "label": "l", "object": "", "weight": 1.0}"#,
            ),
        ] {
            let error = parse(&format!("{HEADER}\n{line}\n")).unwrap_err();
            assert!(
                error.contains("line 2") && error.contains(field) && error.contains("empty"),
                "{field}: {error}"
            );
        }

        // Every field non-empty still parses fine.
        let batch = parse(&format!(
            "{HEADER}\n{{\"subject\": \"a\", \"label\": \"l\", \"object\": \"b\", \"weight\": 1.0}}\n"
        ))
        .unwrap();
        assert_eq!(batch.associations.len(), 1);
    }

    #[test]
    fn a_line_that_is_no_known_shape_names_the_known_shapes() {
        let error = parse(&format!("{HEADER}\n{{\"foo\": 1}}\n")).unwrap_err();
        assert!(
            error.contains("line 2") && error.contains("association"),
            "{error}"
        );
    }

    /// The batch-open marker around `apply_batch`'s four mutations:
    /// absent after success, never opened at all for a batch whose
    /// alias step is predicted to fail before anything runs, gone
    /// again once the documented repair — re-importing the source —
    /// completes. A marker surviving a genuine mid-batch refusal (one
    /// prediction cannot catch, e.g. a disk fault) is covered
    /// separately by
    /// [`apply_batch_refuses_when_an_unreplaced_passage_cannot_be_retracted`].
    #[test]
    fn apply_batch_brackets_its_steps_with_the_import_marker() {
        let dir = std::env::temp_dir().join(format!("taguru-ingest-marker-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

        // A completed batch leaves no marker: its truth is fully on disk.
        let happy = parse(
            "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\", \"create\": {}}\n\
             {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n",
        )
        .unwrap();
        apply_batch(&state, &happy).unwrap();
        assert!(
            crate::registry::import_marker_paths(&dir, "sake").is_empty(),
            "a completed batch clears its marker"
        );

        // A batch whose alias step is predicted to fail is refused
        // before anything runs: no marker opens for it to keep. (An
        // alias to a canonical nothing interned — the same rejection
        // `add_alias` would raise for real, just caught here first.)
        let torn = parse(
            "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-2\"}\n\
             {\"alias\": \"Aomine\", \"canonical\": \"存在しない\", \"kind\": \"concept\"}\n",
        )
        .unwrap();
        let refusal = apply_batch(&state, &torn).unwrap_err();
        assert!(matches!(refusal, ApplyRefusal::Rejected(_)));
        assert_eq!(
            crate::registry::import_marker_paths(&dir, "sake").len(),
            0,
            "a predicted rejection opens no marker"
        );

        // A corrected batch for the same source applies cleanly —
        // there was never a tear to repair, just a rejected batch
        // that nothing depended on.
        let fixed = parse(
            "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-2\"}\n\
             {\"subject\": \"青嶺酒造\", \"label\": \"銘柄\", \"object\": \"青嶺\", \"weight\": 1.0}\n\
             {\"alias\": \"Aomine\", \"canonical\": \"青嶺酒造\", \"kind\": \"concept\"}\n",
        )
        .unwrap();
        apply_batch(&state, &fixed).unwrap();
        assert!(
            crate::registry::import_marker_paths(&dir, "sake").is_empty(),
            "a normal import leaves no marker"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// An associations-only re-import (no passage line in this batch)
    /// for a source that already has one on disk: the differential
    /// sync still retracts that old passage first, same as any other
    /// batch. If the retraction genuinely fails to remove it — not
    /// "there was nothing to remove" — nothing later in this batch
    /// will ever overwrite the stale copy, so the batch must refuse
    /// and keep its marker rather than clear it over a source whose
    /// truth is now half-applied.
    #[test]
    fn apply_batch_refuses_when_an_unreplaced_passage_cannot_be_retracted() {
        let mut exhausted = false;
        let mut saw_the_refusal = false;
        for failure in 0..24 {
            let dir = std::env::temp_dir().join(format!(
                "taguru-ingest-marker-passage-fault-{failure}-{}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&dir);
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

            let seeded = parse(
                "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\", \"create\": {}}\n\
                 {\"passage\": \"杜氏は高瀬。\"}\n\
                 {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n",
            )
            .unwrap();
            apply_batch(&state, &seeded).unwrap();

            let reimport = parse(
                "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\"}\n\
                 {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬2\", \"weight\": 1.0}\n",
            )
            .unwrap();

            crate::storage::fail_persistence_ops_after(failure);
            let result = apply_batch(&state, &reimport);
            let past_end = crate::storage::clear_persistence_fault();

            if let Err(ApplyRefusal::Io(message)) = &result
                && message.contains("could not be retracted")
            {
                saw_the_refusal = true;
                assert_eq!(
                    crate::registry::import_marker_paths(&dir, "sake").len(),
                    1,
                    "step {failure}: refusing to retract an unreplaced passage still \
                     cleared the marker"
                );
                // The documented repair still converges: retrying the
                // same associations-only batch re-attempts the
                // retraction (idempotent per-source) with the fault
                // now cleared.
                apply_batch(&state, &reimport).unwrap();
                assert!(
                    crate::registry::import_marker_paths(&dir, "sake").is_empty(),
                    "step {failure}: repair did not clear the marker"
                );
            }

            drop(state);
            let _ = fs::remove_dir_all(&dir);

            if past_end {
                exhausted = true;
                break;
            }
        }
        assert!(
            exhausted,
            "sweep bound too small to reach past every persistence step"
        );
        assert!(
            saw_the_refusal,
            "the sweep never reached the passage-retract fault point"
        );
    }

    /// `Applied::passage_dropped` is documented as "retracted AND no
    /// replacement carried" — `preview_batch` implements exactly that
    /// AND, so a routine re-import that supplies a replacement passage
    /// must report `passage_dropped: false` from both entrances alike.
    #[test]
    fn apply_and_preview_agree_that_a_replaced_passage_is_not_dropped() {
        let dir = std::env::temp_dir().join(format!(
            "taguru-ingest-passage-replace-parity-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

        let seeded = parse(
            "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\", \"create\": {}}\n\
             {\"passage\": \"杜氏は高瀬。\"}\n",
        )
        .unwrap();
        apply_batch(&state, &seeded).unwrap();

        let reimport = parse(
            "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\"}\n\
             {\"passage\": \"杜氏は高瀬二代目。\"}\n",
        )
        .unwrap();

        let previewed = preview_batch(&state, &reimport).unwrap();
        assert!(
            !previewed.passage_dropped,
            "preview: a replacement passage was carried, so nothing was dropped"
        );

        let applied = apply_batch(&state, &reimport).unwrap();
        assert!(
            !applied.passage_dropped,
            "apply: a replacement passage was carried, so nothing was dropped, \
             matching preview_batch's own report for the identical batch"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    /// A batch that pairs a valid association with a conflicting
    /// alias, aimed at a context that does not exist yet, is refused
    /// before the association ever lands: predicting the alias step's
    /// outcome up front means a batch that would otherwise write the
    /// association and only then fail on its alias no longer gets to
    /// write anything at all — not even the context it would have
    /// created.
    #[test]
    fn a_predicted_alias_rejection_creates_nothing_and_applies_nothing() {
        let dir = std::env::temp_dir().join(format!(
            "taguru-ingest-predicted-rejection-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();

        let torn = parse(
            "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\", \"create\": {}}\n\
             {\"subject\": \"蔵\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n\
             {\"alias\": \"Aomine\", \"canonical\": \"存在しない\", \"kind\": \"concept\"}\n",
        )
        .unwrap();
        let refusal = apply_batch(&state, &torn).unwrap_err();
        assert!(
            matches!(refusal, ApplyRefusal::Rejected(_)),
            "expected a predicted rejection, got {refusal:?}"
        );
        assert!(!refusal.wrote_anything());
        assert_eq!(refusal.ops_written(), 0);
        assert!(
            state.directory_entry("sake").is_none(),
            "a predicted rejection must not create the context the batch named"
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn ops_written_counts_only_the_partial_refusal() {
        assert_eq!(ApplyRefusal::NoContext("sake".to_string()).ops_written(), 0);
        assert_eq!(ApplyRefusal::Io("boom".to_string()).ops_written(), 0);
        assert_eq!(ApplyRefusal::Access(AccessError::NotFound).ops_written(), 0);
        assert_eq!(
            ApplyRefusal::Partial {
                applied: 5,
                message: "boom".to_string(),
                full: false,
            }
            .ops_written(),
            5
        );
        assert_eq!(
            ApplyRefusal::Rejected(AliasRejection {
                namespace: AliasNamespace::Concept,
                alias: "a".to_string(),
                canonical: "c".to_string(),
                error: AliasError::UnknownCanonical,
            })
            .ops_written(),
            0
        );
    }

    /// Move one deterministic filesystem failure through the complete
    /// import: marker, source retraction, passage store, associations,
    /// aliases, and marker unlink. A stopped batch keeps its marker;
    /// a failure before the marker applies nothing; and any swallowed
    /// best-effort failure must still leave a complete, retryable truth.
    #[test]
    fn every_import_persistence_failure_is_detected_or_fully_repaired() {
        let mut exhausted = false;
        for failure in 0..24 {
            let dir = std::env::temp_dir().join(format!(
                "taguru-ingest-fault-{failure}-{}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&dir);
            let state = AppState::boot(dir.clone(), usize::MAX, None).unwrap();
            state.create("sake", ContextMeta::default()).unwrap();
            let batch = parse(
                "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-1\"}\n\
                 {\"passage\": \"青嶺酒造の杜氏は高瀬。\"}\n\
                 {\"subject\": \"青嶺酒造\", \"label\": \"杜氏\", \"object\": \"高瀬\", \"weight\": 1.0}\n\
                 {\"alias\": \"青嶺\", \"canonical\": \"青嶺酒造\", \"kind\": \"concept\"}\n",
            )
            .unwrap();

            crate::storage::fail_persistence_ops_after(failure);
            let first = apply_batch(&state, &batch);
            let past_end = crate::storage::clear_persistence_fault();
            let marker = crate::registry::import_marker_path(&dir, "sake", "doc-1");

            if past_end {
                assert!(
                    first.is_ok(),
                    "the past-end attempt must complete: {first:?}"
                );
                assert!(!marker.exists());
            } else {
                if let Err(refusal) = &first {
                    let before_marker = refusal.text().contains("marker not persisted");
                    assert_eq!(
                        marker.exists(),
                        !before_marker,
                        "a stopped batch at step {failure} lost its tear witness: {refusal:?}"
                    );
                    if let ApplyRefusal::Partial { applied, .. } = refusal {
                        assert_eq!(
                            refusal.ops_written(),
                            *applied,
                            "step {failure}: ops_written must mirror the partial \
                             refusal's own running total"
                        );
                    }
                }
                // Re-import is the documented repair and is exact even
                // when the injected error was swallowed after a fully
                // superseding write or only prevented marker cleanup.
                apply_batch(&state, &batch).unwrap();
                assert!(
                    !marker.exists(),
                    "repair did not clear failure step {failure}"
                );
            }

            assert_eq!(
                state
                    .read_context("sake", |context| context.association_count())
                    .unwrap(),
                1,
                "retry at step {failure} was not idempotent"
            );
            assert_eq!(
                state
                    .read_context("sake", |context| context.resolve("青嶺")[0].name.clone())
                    .unwrap(),
                "青嶺酒造",
                "alias step {failure} did not land"
            );
            assert_eq!(
                state
                    .lookup_passages("sake", &["doc-1".to_string()])
                    .unwrap()
                    .unwrap()
                    .0["doc-1"],
                "青嶺酒造の杜氏は高瀬。"
            );
            drop(state);
            let _ = fs::remove_dir_all(&dir);
            if past_end {
                exhausted = true;
                break;
            }
        }
        assert!(exhausted, "import exceeded the persistence sweep bound");
    }

    #[test]
    fn directories_expand_to_their_sorted_jsonl_files() {
        let dir = std::env::temp_dir().join(format!("taguru-ingest-expand-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("b.jsonl"), "x").unwrap();
        fs::write(dir.join("a.jsonl"), "x").unwrap();
        fs::write(dir.join("ignored.txt"), "x").unwrap();
        // A subdirectory that happens to be named like a batch file
        // must never ride along: `fs::File::open` on it would fail
        // with a confusing "Is a directory" error far from here,
        // instead of `expand` just not collecting it in the first
        // place.
        fs::create_dir_all(dir.join("c.jsonl")).unwrap();
        let files = expand(&[dir.to_string_lossy().into_owned()]).unwrap();
        let names: Vec<_> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, ["a.jsonl", "b.jsonl"]);

        let empty = dir.join("empty");
        fs::create_dir_all(&empty).unwrap();
        let error = expand(&[empty.to_string_lossy().into_owned()]).unwrap_err();
        assert!(error.contains("no .jsonl files"), "{error}");
        let _ = fs::remove_dir_all(&dir);
    }

    fn unit(label: &str, bytes: usize) -> Unit {
        Unit {
            text: "x".repeat(bytes),
            label: label.to_string(),
            kind: UnitKind::Batch,
        }
    }

    #[test]
    fn pack_chunks_fills_each_chunk_up_to_the_budget_without_splitting_a_unit() {
        let units = vec![unit("a", 40), unit("b", 40), unit("c", 40), unit("d", 5)];
        let queue = pack_chunks(units, 100);
        // a+b = 80 (fits); +c = 120 (over 100) → c starts the next
        // chunk; c+d = 45 (fits, and nothing follows).
        let sizes: Vec<usize> = queue.iter().map(Chunk::size).collect();
        assert_eq!(sizes, vec![80, 45], "{sizes:?}");
        assert_eq!(queue[0].units.len(), 2);
        assert_eq!(queue[1].units.len(), 2);
    }

    #[test]
    fn pack_chunks_never_splits_a_single_oversized_unit_across_two_chunks() {
        // A unit alone over budget still rides whole in its own chunk
        // — pack_chunks never refuses; the caller checks this case
        // before packing (run_remote's pre-send hard-error).
        let units = vec![unit("small", 10), unit("huge", 500)];
        let queue = pack_chunks(units, 100);
        assert_eq!(queue.len(), 2);
        assert_eq!(queue[0].units.len(), 1);
        assert_eq!(queue[1].units.len(), 1);
        assert_eq!(queue[1].units[0].label, "huge");
    }

    #[test]
    fn pack_chunks_on_an_empty_input_is_an_empty_queue() {
        assert!(pack_chunks(Vec::new(), 100).is_empty());
    }

    #[test]
    fn chunk_halve_splits_at_the_unit_boundary_closest_to_half_the_bytes() {
        let chunk = Chunk {
            units: vec![unit("one", 10), unit("two", 10), unit("three", 80)],
        };
        let (first, second) = chunk.halve();
        assert_eq!(
            first.units.len(),
            2,
            "{:?}",
            first.units.iter().map(|u| &u.label).collect::<Vec<_>>()
        );
        assert_eq!(second.units.len(), 1);
        assert_eq!(second.units[0].label, "three");
    }

    #[test]
    fn chunk_halve_never_produces_an_empty_half_on_two_units() {
        // Even a lopsided two-unit chunk (one unit far bigger than the
        // other) must still split into one unit per half — the split
        // point is clamped to 1..len-1, never 0 or len.
        let chunk = Chunk {
            units: vec![unit("big", 90), unit("small", 10)],
        };
        let (first, second) = chunk.halve();
        assert_eq!(first.units.len(), 1);
        assert_eq!(second.units.len(), 1);
        assert_eq!(first.units[0].label, "big");
        assert_eq!(second.units[0].label, "small");
    }

    #[test]
    fn chunk_body_guarantees_a_trailing_newline_per_unit() {
        let chunk = Chunk {
            units: vec![
                Unit {
                    text: "{\"a\":1}".to_string(),
                    label: "no-newline".to_string(),
                    kind: UnitKind::Batch,
                },
                Unit {
                    text: "{\"b\":2}\n".to_string(),
                    label: "has-newline".to_string(),
                    kind: UnitKind::Batch,
                },
            ],
        };
        assert_eq!(chunk.body(), "{\"a\":1}\n{\"b\":2}\n");
    }

    /// [`run_remote`]'s Pass 1 strips a leading BOM before
    /// `split_batches` runs — proven here at the byte level, matching
    /// the local path's own `a_leading_bom_does_not_break_the_first_line`
    /// pin. Without the strip, the BOM's three bytes would ride inside
    /// `split_batches`' very first range and be sent to the server as
    /// part of the wire chunk.
    #[test]
    fn a_leading_bom_is_stripped_before_split_batches_runs() {
        let mut bytes = vec![0xEF, 0xBB, 0xBF];
        bytes.extend_from_slice(b"{\"taguru_batch\": 1, \"context\": \"c\", \"source\": \"s\"}\n");
        assert!(bytes.starts_with(&[0xEF, 0xBB, 0xBF]));
        bytes.drain(0..3);
        let ranges = split_batches(&bytes);
        assert_eq!(ranges.len(), 1);
        let first = std::str::from_utf8(&bytes[ranges[0].clone()]).unwrap();
        assert!(first.starts_with("{\"taguru_batch\""), "{first}");
        assert!(!first.as_bytes().starts_with(&[0xEF, 0xBB, 0xBF]));
    }
}
