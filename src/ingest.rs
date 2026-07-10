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
//!
//! The same contract is served live as `POST /import` (one request =
//! one batch file), so bulk loads reach a running server without a
//! downtime window; [`parse_batch`] and [`apply_batch`] are that
//! endpoint's core too, which is what keeps the two entrances from
//! drifting apart.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::api::{
    MAX_ASSOCIATION_WEIGHT, MAX_ASSOCIATIONS_PER_REQUEST, MAX_CONTEXT_NAME_BYTES,
    MAX_DESCRIPTION_BYTES, MAX_NAME_BYTES,
};
use crate::registry::{AccessError, AppState, AssocOp, ContextMeta, CreateError};

const USAGE: &str = "\
usage: taguru import [--dry-run] [--no-embed] [--config FILE] FILE|DIR...

Applies JSONL batch files to TAGURU_DATA_DIR offline (the server must
not be running — the directory lock enforces it). One file = one
source's complete truth: import retracts the source, then applies the
file, so re-importing is idempotent. A directory expands to its
*.jsonl files, sorted by name. Format: docs/import.md. A running
server accepts the same body at POST /import (authenticated), one
file per request — live systems need no downtime window.

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
/// smuggle in what a request could not. Extract caps whole documents
/// here too — a document over it could not ride as a passage.
pub(crate) const MAX_PASSAGE_BYTES: usize = 8 * 1024 * 1024;

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
    let embedder: Option<std::sync::Arc<dyn crate::embedding::EmbeddingProvider>> = if no_embed {
        None
    } else {
        crate::embedding::HttpEmbeddings::from_env()
            .map(|provider| std::sync::Arc::new(provider) as _)
    };
    // The same knobs serve boots with — one reading for both entrances
    // (cli.rs documents them once).
    let state = match crate::registry::BootConfig::from_env().boot(embedder) {
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
        match apply_batch(&state, batch) {
            Ok(applied) => {
                println!("{}: {}", path.display(), report(batch, &applied));
                touched.insert(batch.context.clone());
                ops_since_flush += batch.op_count();
            }
            Err(refusal) => {
                eprintln!("taguru: import: {}: {}", path.display(), refusal.text());
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
    associations: Vec<AssocOp>,
    concepts: BTreeMap<String, String>,
    labels: BTreeMap<String, String>,
}

impl Batch {
    fn op_count(&self) -> usize {
        self.associations.len() + self.concepts.len() + self.labels.len()
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
            "context '{}' ← source '{}': {} association(s), {} alias(es){}{}{}",
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

/// Parses one batch file completely, or says which line refused and
/// why. Blank lines are skipped; the first non-blank line must be the
/// header.
pub(crate) fn parse_batch(mut reader: impl BufRead) -> Result<Batch, String> {
    let mut batch: Option<Batch> = None;
    // Per-paragraph question tally, carried as we parse so the per-line
    // cap check is a map lookup instead of a rescan of every question
    // seen so far — a batch piling questions on one paragraph would
    // otherwise be quadratic.
    let mut question_counts: BTreeMap<u32, usize> = BTreeMap::new();
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
        match &mut batch {
            None => batch = Some(parse_header(line, number)?),
            Some(batch) => parse_op(batch, &mut question_counts, line, number)?,
        }
    }
    let batch = batch.ok_or_else(|| "empty file: expected a batch header line".to_string())?;
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
    line: &str,
    number: usize,
) -> Result<(), String> {
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
        if batch.passage.replace(op.passage).is_some() {
            return Err(format!(
                "line {number}: a second passage line — one batch file carries at most \
                 one passage (the header source's original text)"
            ));
        }
    } else if object.contains_key("question") {
        let op: QuestionLine = serde_json::from_value(value)
            .map_err(|error| format!("line {number}: question: {error}"))?;
        check_size(
            number,
            "question",
            &op.question,
            crate::api::MAX_QUESTION_BYTES,
        )?;
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
    } else if object.contains_key("section") {
        let op: SectionLine = serde_json::from_value(value)
            .map_err(|error| format!("line {number}: section: {error}"))?;
        check_size(
            number,
            "section",
            &op.section,
            crate::api::MAX_SECTION_BYTES,
        )?;
        batch.sections.push((op.paragraph, op.section));
    } else {
        return Err(format!(
            "line {number}: not an association (subject/label/object/weight), an alias \
             (alias/canonical/kind), a passage line, a question (paragraph/question) line, \
             or a section (paragraph/section) line"
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
    /// Sections naming a paragraph their passage's split does not
    /// have — same convention and same likely cause as
    /// `questions_dropped`.
    pub(crate) sections_dropped: usize,
    /// Association paragraph locators naming a spot this batch's own
    /// passage split does not have. Dropped exactly as `questions_dropped`
    /// and `sections_dropped` are — the association's fact still lands,
    /// only the paragraph pointer is cleared — and surfaced for the same
    /// reason: so the loss is a reported number, not a silent one.
    pub(crate) association_paragraphs_dropped: usize,
}

/// Why a batch did not (fully) apply — one shape for both entrances:
/// the CLI prints [`ApplyRefusal::text`], the HTTP endpoint maps the
/// variant onto a status and sends the same words.
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
}

impl ApplyRefusal {
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
            Self::Partial { message, .. } => message.clone(),
        }
    }
}

/// Applies one validated batch: ensure the context, retract the
/// source, then land passage → associations → aliases. Aliases go
/// last on purpose — an alias needs its canonical interned, and the
/// associations just before are what intern it.
pub(crate) fn apply_batch(state: &AppState, batch: &Batch) -> Result<Applied, ApplyRefusal> {
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

    let (retracted, passage_dropped) = state
        .retract_source(&batch.context, &batch.source)
        .map_err(ApplyRefusal::Access)?;

    let mut questions_stored = 0;
    let mut questions_dropped = 0;
    let mut sections_stored = 0;
    let mut sections_dropped = 0;
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
                    },
                )]),
            )
            .ok_or(ApplyRefusal::Access(AccessError::NotFound))?
            .map_err(|io_error| ApplyRefusal::Io(format!("passage not persisted: {io_error}")))?;
        questions_stored = outcome.questions_stored;
        questions_dropped = outcome.questions_dropped;
        sections_stored = outcome.sections_stored;
        sections_dropped = outcome.sections_dropped;
    }

    // Same rule as questions/sections above, applied silently: a
    // paragraph naming a spot this batch's own passage does not have
    // is meaningless, so it is dropped rather than persisted — the
    // association itself (subject/label/object/weight) still lands.
    // Only checked against a passage this same batch carries; an
    // associations-only batch has nothing to check against, exactly
    // like questions/sections above.
    let mut association_paragraphs_dropped = 0;
    let corrected_associations: Vec<AssocOp>;
    let associations_to_apply: &[AssocOp] = match &batch.passage {
        Some(text) => {
            let paragraph_count = crate::paragraph::split(text).len();
            corrected_associations = batch
                .associations
                .iter()
                .cloned()
                .map(|mut op| {
                    if op.paragraph.is_some_and(|p| p as usize >= paragraph_count) {
                        op.paragraph = None;
                        association_paragraphs_dropped += 1;
                    }
                    op
                })
                .collect();
            &corrected_associations
        }
        None => &batch.associations,
    };

    let mut associations = 0;
    for chunk in associations_to_apply.chunks(MAX_ASSOCIATIONS_PER_REQUEST) {
        match state
            .add_associations(&batch.context, chunk.to_vec())
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
                    applied: partial.applied,
                    message: format!(
                        "applied {} alias(es), then: {}",
                        partial.applied, partial.message
                    ),
                    full: partial.full,
                });
            }
        }
    }

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
        association_paragraphs_dropped,
    })
}

/// The CLI's per-file report line.
fn report(batch: &Batch, applied: &Applied) -> String {
    format!(
        "context '{}'{} ← source '{}' ({} association(s) retracted): +{} \
         association(s), +{} alias(es){}{}{}{}",
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
        match applied.association_paragraphs_dropped {
            0 => String::new(),
            dropped => {
                format!(", {dropped} association paragraph locator(s) dropped: no such paragraph")
            }
        }
    )
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
            association_paragraphs_dropped: 0,
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
