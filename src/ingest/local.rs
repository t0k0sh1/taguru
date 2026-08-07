//! The local/offline apply path: `taguru import` (no `--url`) writes
//! straight into `TAGURU_DATA_DIR`, validating every file up front
//! (Pass 1) before applying any of them (Pass 2) — see the hub's own
//! module doc for the two-pass contract this mirrors from `run_remote`.

use super::*;

/// Ops applied between mid-run flushes. Import batches can dwarf any
/// live traffic; flushing every so often keeps each context's WAL far
/// from `TAGURU_WAL_MAX_BYTES` (past which writes are refused).
const FLUSH_EVERY_OPS: usize = 100_000;

pub(super) fn run_local(files: &[PathBuf], dry_run: bool, no_embed: bool, as_json: bool) -> i32 {
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
    let embedder = if no_embed {
        None
    } else {
        // A fresh, never-raised flag: the import runs one command to
        // completion, with no graceful drain to unblock.
        crate::embedding::provider_from_env(crate::embedding::ShutdownFlag::default())
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
            // TAGURU_EMBED_PASSAGES=1 is the operator's consent to
            // spend on passage vectors too — skipping them here while
            // the glosses above embed automatically left the vector
            // lane silently absent until a manual refresh (#479).
            if state.passage_embedding_enabled() {
                match state.refresh_passage_embeddings(name, Deadline::unbounded()) {
                    None | Some(Ok(crate::registry::PassageRefreshOutcome { embedded: 0, .. })) => {
                    }
                    Some(Ok(outcome)) => {
                        if !as_json {
                            println!("{name}: embedded {} passages", outcome.embedded);
                        }
                    }
                    Some(Err(error)) => {
                        eprintln!(
                            "taguru: import: {name}: passage embedding refresh failed \
                             ({error}) — the passages are imported and durable; refresh \
                             later via POST /contexts/{name}/embeddings/refresh"
                        );
                        embed_failures += 1;
                    }
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
pub(super) fn duplicate_source_message(context: &str, source: &str) -> String {
    format!(
        "source '{source}' in context '{context}' is already stated by an earlier file \
         — one file owns one source's truth"
    )
}

/// [`duplicate_source_message`]'s schema-record twin.
pub(super) fn duplicate_schema_message(context: &str) -> String {
    format!(
        "context '{context}' schema is already stated by an earlier file — one record \
         owns one context's schema"
    )
}

/// [`duplicate_source_message`]'s group-record twin.
pub(super) fn duplicate_group_message(name: &str) -> String {
    format!(
        "group '{name}' is already stated by an earlier file — one record owns one \
         group's truth"
    )
}
