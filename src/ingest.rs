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
//!
//! The writes go through the same registry every server write goes
//! through — WAL-staged, budget-enforced, flushed — and the data
//! directory lock makes the server/import conflict a refusal instead
//! of a corruption.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::io::BufRead;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::api::{
    MAX_ASSOCIATION_WEIGHT, MAX_ASSOCIATIONS_PER_REQUEST, MAX_CONTEXT_NAME_BYTES,
    MAX_DESCRIPTION_BYTES, MAX_NAME_BYTES,
};
use crate::registry::{AccessError, AppState, AssocOp, ContextMeta};

const USAGE: &str = "\
usage: taguru import [--dry-run] [--no-embed] [--config FILE] FILE|DIR...

Applies JSONL batch files to TAGURU_DATA_DIR offline (the server must
not be running — the directory lock enforces it). One file = one
source's complete truth: import retracts the source, then applies the
file, so re-importing is idempotent. A directory expands to its
*.jsonl files, sorted by name. Format: docs/import.md.

  --dry-run    validate every file and report; touch nothing
  --no-embed   skip the embedding refresh TAGURU_EMBED_URL would enable
  --config F   read KEY=VALUE environment from F (same dialect as serve)
";

/// The one format version this build reads and docs/import.md
/// describes.
const BATCH_VERSION: u64 = 1;

/// Per-line byte cap. Lines are one fact or one passage; past this
/// something is wrong with the producer, and refusing early beats
/// buffering a runaway line.
const MAX_LINE_BYTES: usize = 16 * 1024 * 1024;

/// Passage cap, mirroring the HTTP default: over the API a passage
/// rides under `TAGURU_MAX_BODY_BYTES` (8 MiB), and a file must not
/// smuggle in what a request could not.
const MAX_PASSAGE_BYTES: usize = 8 * 1024 * 1024;

/// Ops applied between mid-run flushes. Import batches can dwarf any
/// live traffic; flushing every so often keeps each context's WAL far
/// from `TAGURU_WAL_MAX_BYTES` (past which writes are refused).
const FLUSH_EVERY_OPS: usize = 100_000;

pub fn run(args: &[String]) -> i32 {
    let mut dry_run = false;
    let mut no_embed = false;
    let mut config: Option<PathBuf> = None;
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
            "--config" => match rest.next() {
                Some(path) => config = Some(PathBuf::from(path)),
                None => return usage_error("--config needs a file path"),
            },
            other if other.starts_with('-') => {
                return usage_error(&format!("unknown flag '{other}'"));
            }
            path => paths.push(path.to_string()),
        }
    }
    if paths.is_empty() {
        eprint!("{USAGE}");
        return 2;
    }
    // SAFETY (same contract as serve): applied while the process is
    // still single-threaded — import never starts a runtime at all.
    if let Some(path) = &config {
        crate::cli::load_config(path);
    }

    let files = match expand(&paths) {
        Ok(files) => files,
        Err(message) => return usage_error(&message),
    };

    // Pass 1 — every file parses, or nothing applies. Apply-stage
    // refusals can strand a half-written source; a malformed line is
    // knowable up front, so it must never cost a write.
    let mut batches = Vec::new();
    let mut broken = 0;
    let mut owners: HashSet<(String, String)> = HashSet::new();
    for path in &files {
        let batch = fs::File::open(path)
            .map_err(|error| error.to_string())
            .and_then(|file| parse_batch(std::io::BufReader::new(file)));
        match batch {
            Ok(batch) => {
                if !owners.insert((batch.context.clone(), batch.source.clone())) {
                    eprintln!(
                        "taguru: import: {}: source '{}' in context '{}' is already \
                         stated by an earlier file — one file owns one source's truth",
                        path.display(),
                        batch.source,
                        batch.context
                    );
                    broken += 1;
                    continue;
                }
                batches.push((path, batch));
            }
            Err(message) => {
                eprintln!("taguru: import: {}: {message}", path.display());
                broken += 1;
            }
        }
    }
    if broken > 0 {
        eprintln!(
            "taguru: import: {broken} of {} file(s) refused during validation; \
             nothing was applied",
            files.len()
        );
        return 1;
    }

    if dry_run {
        for (path, batch) in &batches {
            println!("{}: {}", path.display(), batch.describe());
        }
        println!("dry run: {} file(s) valid, nothing applied", batches.len());
        return 0;
    }

    // Registry warnings (WAL replay notes, load errors) must reach the
    // operator; stdout stays reserved for the report lines.
    init_logging();
    let data_dir =
        PathBuf::from(std::env::var("TAGURU_DATA_DIR").unwrap_or_else(|_| "data".into()));
    let embedder: Option<std::sync::Arc<dyn crate::embedding::EmbeddingProvider>> = if no_embed {
        None
    } else {
        crate::embedding::HttpEmbeddings::from_env()
            .map(|provider| std::sync::Arc::new(provider) as _)
    };
    // Mirrors serve's reading of the same knobs (cli.rs documents them
    // once for both).
    let wal_enabled = std::env::var("TAGURU_WAL")
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(true);
    let state = match AppState::boot_with(
        data_dir,
        crate::env_number("TAGURU_CACHE_BYTES", 512 * 1024 * 1024),
        embedder,
        wal_enabled,
        crate::env_number(
            "TAGURU_WAL_MAX_BYTES",
            crate::registry::DEFAULT_WAL_MAX_BYTES,
        ),
        crate::env_floor("TAGURU_SEMANTIC_FLOOR"),
    ) {
        Ok(state) => state,
        Err(error) => {
            eprintln!("taguru: import: {error}");
            return 1;
        }
    };

    // Pass 2 — apply, one file at a time, in the order given.
    let mut failures = 0;
    let mut touched: BTreeSet<String> = BTreeSet::new();
    let mut ops_since_flush = 0usize;
    for (path, batch) in &batches {
        match apply(&state, batch) {
            Ok(outcome) => {
                println!("{}: {outcome}", path.display());
                touched.insert(batch.context.clone());
                ops_since_flush += batch.op_count();
            }
            Err(message) => {
                eprintln!("taguru: import: {}: {message}", path.display());
                failures += 1;
            }
        }
        if ops_since_flush >= FLUSH_EVERY_OPS {
            state.flush_dirty();
            ops_since_flush = 0;
        }
    }
    state.flush_dirty();
    state.persist_usage();

    let mut embed_failures = 0;
    if state.embeddings_configured() {
        for name in &touched {
            match state.refresh_embeddings(name) {
                None | Some(Ok((0, _))) => {}
                Some(Ok((embedded, _))) => println!("{name}: embedded {embedded} glosses"),
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

    println!(
        "import: {} of {} file(s) applied across {} context(s)",
        batches.len() - failures,
        batches.len(),
        touched.len()
    );
    if failures > 0 || embed_failures > 0 {
        1
    } else {
        0
    }
}

fn usage_error(message: &str) -> i32 {
    eprintln!("taguru: import: {message} — try 'taguru import --help'");
    2
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
                .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("jsonl"))
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

/// One parsed batch file: the header's claims plus the accumulated op
/// lines, every association already stamped with the header's source.
#[cfg_attr(test, derive(Debug))]
struct Batch {
    context: String,
    source: String,
    create: Option<ContextMeta>,
    passage: Option<String>,
    associations: Vec<AssocOp>,
    concepts: BTreeMap<String, String>,
    labels: BTreeMap<String, String>,
}

impl Batch {
    fn op_count(&self) -> usize {
        self.associations.len() + self.concepts.len() + self.labels.len()
    }

    fn describe(&self) -> String {
        format!(
            "context '{}' ← source '{}': {} association(s), {} alias(es){}",
            self.context,
            self.source,
            self.associations.len(),
            self.concepts.len() + self.labels.len(),
            if self.passage.is_some() {
                ", 1 passage"
            } else {
                ""
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AssociationLine {
    subject: String,
    label: String,
    object: String,
    weight: f64,
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
}

/// Parses one batch file completely, or says which line refused and
/// why. Blank lines are skipped; the first non-blank line must be the
/// header.
fn parse_batch(reader: impl BufRead) -> Result<Batch, String> {
    let mut batch: Option<Batch> = None;
    for (index, line) in reader.lines().enumerate() {
        let number = index + 1;
        let line = line.map_err(|error| format!("line {number}: {error}"))?;
        if line.len() > MAX_LINE_BYTES {
            return Err(format!(
                "line {number}: {} bytes exceeds the {MAX_LINE_BYTES}-byte line cap",
                line.len()
            ));
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match &mut batch {
            None => batch = Some(parse_header(line, number)?),
            Some(batch) => parse_op(batch, line, number)?,
        }
    }
    batch.ok_or_else(|| "empty file: expected a batch header line".to_string())
}

fn parse_header(line: &str, number: usize) -> Result<Batch, String> {
    let header: Header = serde_json::from_str(line)
        .map_err(|error| format!("line {number}: not a batch header: {error}"))?;
    if header.taguru_batch != BATCH_VERSION {
        return Err(format!(
            "line {number}: taguru_batch {} is not a version this taguru reads (it reads \
             {BATCH_VERSION})",
            header.taguru_batch
        ));
    }
    check_size(number, "context", &header.context, MAX_CONTEXT_NAME_BYTES)?;
    check_size(number, "source", &header.source, MAX_NAME_BYTES)?;
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
        associations: Vec::new(),
        concepts: BTreeMap::new(),
        labels: BTreeMap::new(),
    })
}

/// Classifies an op line by its distinguishing key, then parses the
/// matching shape strictly — so the error for a stray field names the
/// field instead of shrugging at every shape at once.
fn parse_op(batch: &mut Batch, line: &str, number: usize) -> Result<(), String> {
    let value: serde_json::Value =
        serde_json::from_str(line).map_err(|error| format!("line {number}: not JSON: {error}"))?;
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
        }
        batch.associations.push(AssocOp {
            subject: op.subject,
            label: op.label,
            object: op.object,
            weight: op.weight,
            source: Some(batch.source.clone()),
        });
    } else if object.contains_key("alias") {
        let op: AliasLine = serde_json::from_value(value)
            .map_err(|error| format!("line {number}: alias: {error}"))?;
        check_size(number, "alias", &op.alias, MAX_NAME_BYTES)?;
        check_size(number, "canonical", &op.canonical, MAX_NAME_BYTES)?;
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
        if batch.passage.replace(op.passage).is_some() {
            return Err(format!(
                "line {number}: a second passage line — one batch file carries at most \
                 one passage (the header source's original text)"
            ));
        }
    } else {
        return Err(format!(
            "line {number}: not an association (subject/label/object/weight), an alias \
             (alias/canonical/kind), or a passage line"
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

/// Applies one validated batch: ensure the context, retract the
/// source, then land passage → associations → aliases. Aliases go
/// last on purpose — an alias needs its canonical interned, and the
/// associations just before are what intern it.
fn apply(state: &AppState, batch: &Batch) -> Result<String, String> {
    let mut created = false;
    if state.directory_entry(&batch.context).is_none() {
        let Some(meta) = &batch.create else {
            return Err(format!(
                "context '{}' does not exist and the file has no create block",
                batch.context
            ));
        };
        state
            .create(&batch.context, meta.clone())
            .map_err(|error| match error {
                crate::registry::CreateError::AlreadyExists => {
                    "context appeared mid-run (impossible under the directory lock)".to_string()
                }
                crate::registry::CreateError::Io(io_error) => {
                    format!("creating context '{}': {io_error}", batch.context)
                }
            })?;
        created = true;
    }

    let (retracted, _passage_dropped) = state
        .retract_source(&batch.context, &batch.source)
        .map_err(access_text)?;

    if let Some(text) = &batch.passage {
        state
            .store_passages(
                &batch.context,
                BTreeMap::from([(batch.source.clone(), text.clone())]),
            )
            .ok_or_else(|| "context vanished mid-import".to_string())?
            .map_err(|io_error| format!("passage not persisted: {io_error}"))?;
    }

    let mut associations = 0;
    for chunk in batch.associations.chunks(MAX_ASSOCIATIONS_PER_REQUEST) {
        match state
            .add_associations(&batch.context, chunk.to_vec())
            .map_err(access_text)?
        {
            Ok(applied) => associations += applied,
            Err(partial) => {
                return Err(format!(
                    "applied {} association(s), then: {} — fix the file and re-import; \
                     the retraction makes the retry exact",
                    associations + partial.applied,
                    partial.message
                ));
            }
        }
    }

    let mut aliases = 0;
    if !batch.concepts.is_empty() || !batch.labels.is_empty() {
        match state
            .add_aliases(&batch.context, &batch.concepts, &batch.labels)
            .map_err(access_text)?
        {
            Ok(applied) => aliases += applied,
            Err(partial) => {
                return Err(format!(
                    "applied {} alias(es), then: {}",
                    partial.applied, partial.message
                ));
            }
        }
    }

    state.note_write(&batch.context);
    Ok(format!(
        "context '{}'{} ← source '{}' ({} association(s) retracted): +{associations} \
         association(s), +{aliases} alias(es){}",
        batch.context,
        if created { " (created)" } else { "" },
        batch.source,
        retracted,
        if batch.passage.is_some() {
            ", passage stored"
        } else {
            ""
        }
    ))
}

fn access_text(failure: AccessError) -> String {
    match failure {
        AccessError::NotFound => {
            "context disappeared mid-import (impossible under the directory lock)".to_string()
        }
        AccessError::Load(error) => format!("the context image would not load: {error}"),
        AccessError::Unpersisted(error) => format!("the WAL refused the write: {error}"),
    }
}

/// Import logs like the server does (RUST_LOG, stderr) so registry
/// warnings — WAL replay notes, load failures — are not dropped on the
/// floor, but stdout stays pure report.
fn init_logging() {
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

    fn parse(text: &str) -> Result<Batch, String> {
        parse_batch(std::io::Cursor::new(text))
    }

    const HEADER: &str = r#"{"taguru_batch": 1, "context": "sake", "source": "doc-1"}"#;

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
    fn a_line_that_is_no_known_shape_names_the_three_shapes() {
        let error = parse(&format!("{HEADER}\n{{\"foo\": 1}}\n")).unwrap_err();
        assert!(
            error.contains("line 2") && error.contains("association"),
            "{error}"
        );
    }

    #[test]
    fn directories_expand_to_their_sorted_jsonl_files() {
        let dir = std::env::temp_dir().join(format!("taguru-ingest-expand-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("b.jsonl"), "x").unwrap();
        fs::write(dir.join("a.jsonl"), "x").unwrap();
        fs::write(dir.join("ignored.txt"), "x").unwrap();
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
}
