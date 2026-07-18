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

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};

use serde::Deserialize;
use taguru::deadline::Deadline;

use crate::api::{
    MAX_ASSOCIATION_WEIGHT, MAX_ASSOCIATIONS_PER_REQUEST, MAX_CONTEXT_NAME_BYTES,
    MAX_DESCRIPTION_BYTES, MAX_NAME_BYTES,
};
use crate::groups::{GroupRecord, MAX_GROUP_MEMBERS};
use crate::registry::{AccessError, AppState, AssocOp, ContextMeta, CreateError};

const USAGE: &str = "\
usage: taguru import [--dry-run] [--no-embed] [--config FILE] FILE|DIR...

Applies JSONL batch files to TAGURU_DATA_DIR offline (the server must
not be running — the directory lock enforces it). One batch = one
source's complete truth: import retracts the source, then applies the
batch, so re-importing is idempotent. A file carries one batch or a
whole stream of them (each `taguru_batch` header line starts the
next) — `taguru export` writes such streams. A `taguru_group` line
states one group's complete truth the same way; groups restore AFTER
every batch of the run (create-or-replace of the whole record), so
group files re-apply in any order. A directory expands to its *.jsonl
files, sorted by name. Format: docs/import.html. A running server
accepts the same bodies at POST /import (authenticated), one file per
request — live systems need no downtime window.

  --dry-run    validate every file and report; touch nothing
  --no-embed   skip the embedding refresh TAGURU_EMBED_URL would enable
  --config F   read KEY=VALUE environment from F (same dialect as serve)
";

/// The one format version this build reads and docs/import.html
/// describes.
const BATCH_VERSION: u64 = 1;

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
    // knowable up front, so it must never cost a write. A file may
    // carry one batch or a whole stream (`taguru export` output);
    // either way each batch stands alone from here on.
    let mut batches = Vec::new();
    let mut groups: Vec<(&PathBuf, String, GroupRecord)> = Vec::new();
    let mut broken = 0;
    let mut owners: HashSet<(String, String)> = HashSet::new();
    let mut group_owners: HashSet<String> = HashSet::new();
    for path in &files {
        let parsed = fs::File::open(path)
            .map_err(|error| error.to_string())
            .and_then(|file| parse_stream(std::io::BufReader::new(file)));
        // A stream file can carry several batches or groups, so it can
        // trip several of the checks below; `broken` counts files, not
        // events, so one file's several conflicts must still add only 1.
        let mut file_broken = false;
        match parsed {
            Ok(stream) => {
                for batch in stream.batches {
                    if !owners.insert((batch.context.clone(), batch.source.clone())) {
                        eprintln!(
                            "taguru: import: {}: source '{}' in context '{}' is already \
                             stated by an earlier file — one file owns one source's truth",
                            path.display(),
                            batch.source,
                            batch.context
                        );
                        file_broken = true;
                        continue;
                    }
                    batches.push((path, batch));
                }
                for (name, record) in stream.groups {
                    if !group_owners.insert(name.clone()) {
                        eprintln!(
                            "taguru: import: {}: group '{name}' is already stated by an \
                             earlier file — one record owns one group's truth",
                            path.display()
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
        for (path, name, record) in &groups {
            println!("{}: {}", path.display(), describe_group(name, record));
        }
        println!(
            "dry run: {} batch(es){} valid, nothing applied",
            batches.len(),
            match groups.len() {
                0 => String::new(),
                count => format!(" and {count} group record(s)"),
            }
        );
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
            }
        }
        if ops_since_flush >= FLUSH_EVERY_OPS {
            state.flush_dirty();
            ops_since_flush = 0;
        }
    }

    // Groups restore LAST — after every batch, whatever file carried
    // them — so a record and the member contexts it names can travel
    // in one run in any order. One set, validated whole: a member that
    // still does not exist refuses every group record, batches
    // untouched (they landed above; re-importing is idempotent).
    let mut restored = 0usize;
    let mut group_failures = 0usize;
    if !groups.is_empty() {
        let records: Vec<(String, GroupRecord)> = groups
            .iter()
            .map(|(_, name, record)| (name.clone(), record.clone()))
            .collect();
        match state.restore_groups(&records, Deadline::unbounded()) {
            Ok(outcomes) => {
                restored = outcomes.len();
                for ((path, name, record), (_, outcome)) in groups.iter().zip(&outcomes) {
                    println!(
                        "{}: {} — {}",
                        path.display(),
                        describe_group(name, record),
                        outcome.as_str()
                    );
                }
            }
            Err(refusal) => {
                restored = refusal.applied();
                eprintln!("taguru: import: {}", refusal.text());
                group_failures = groups.len() - restored;
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
        "import: {} of {} batch(es) applied across {} context(s)",
        batches.len() - failures,
        batches.len(),
        touched.len()
    );
    if !groups.is_empty() {
        println!(
            "import: {restored} of {} group record(s) restored",
            groups.len()
        );
    }
    if failures > 0 || embed_failures > 0 || group_failures > 0 {
        1
    } else {
        0
    }
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

/// One parsed stream: the batches, then the group records it carried,
/// each in stream order. The split IS the apply order — batches
/// first, all of them, then the groups — so a group and the member
/// contexts it names can ride one stream in any arrangement.
#[cfg_attr(test, derive(Debug))]
pub(crate) struct Stream {
    pub(crate) batches: Vec<Batch>,
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

/// Parses one single-batch file completely, or says which line refused
/// and why — the shape `taguru extract` emits and re-validates. Streams
/// that may carry several batches, or group records, go through
/// [`parse_stream`].
pub(crate) fn parse_batch(reader: impl BufRead) -> Result<Batch, String> {
    let mut stream = parse_stream(reader)?;
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
    let mut groups: Vec<(String, GroupRecord)> = Vec::new();
    let mut current: Option<Batch> = None;
    let mut owners: HashSet<(String, String)> = HashSet::new();
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
        let is_group = has_key("taguru_group");
        if is_header || is_group {
            // Either stream-level record closes the batch before it —
            // one boundary step, however many marker kinds exist.
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
        // A stream of group records alone is a legitimate restore; a
        // stream of nothing is a mistake.
        None if batches.is_empty() && groups.is_empty() => {
            return Err("empty file: expected a batch header or group record line".to_string());
        }
        None => {}
    }
    Ok(Stream { batches, groups })
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
}

impl ApplyRefusal {
    /// Whether the batch may have durably written anything before the
    /// refusal. Only [`ApplyRefusal::NoContext`] provably precedes the
    /// first write — everything past that point starts with the source
    /// retraction, itself a durable write, so a later refusal (a
    /// passage that would not persist, a partial prefix of
    /// associations or aliases) leaves real changes behind. `Io` from
    /// a failed create or a failed batch-marker write is the
    /// over-approximation (both precede the first graph write); the
    /// refresh pass answers an absent context with its no-op `None`
    /// arm anyway.
    pub(crate) fn wrote_anything(&self) -> bool {
        !matches!(self, Self::NoContext(_))
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
            Self::NoContext(_) | Self::Io(_) | Self::Access(_) => 0,
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
            Self::Partial { message, .. } => message.clone(),
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

/// Applies one validated batch: ensure the context, retract the
/// source, then land passage → associations → aliases. Aliases go
/// last on purpose — an alias needs its canonical interned, and the
/// associations just before are what intern it.
///
/// The four mutations are separately durable, so a crash between them
/// leaves the source half-applied with every store individually
/// consistent — undetectable after the fact. A batch-open marker
/// brackets them: written before the retraction, removed only after
/// the aliases, so boot and `taguru inspect` can name any batch that
/// never finished. A REFUSED batch keeps its marker too — the refusal
/// is reported once, the marker keeps saying so until the documented
/// repair (re-import, or retract the source) actually runs.
/// Cross-store atomicity is deliberately not attempted: per-source
/// retract-then-apply idempotency already makes the repair exact, so
/// detection is the whole remaining gap.
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
        association_paragraphs_dropped,
    })
}

/// The read-only twin of [`apply_batch`], for `POST
/// /import?dry_run=true`: reports what a batch WOULD do without
/// writing anything — no context created, no marker opened, no source
/// retracted. Every write step in `apply_batch` has a cheap read-only
/// counterpart here, except two: `associations` and `aliases` are
/// OPTIMISTIC counts (every op this batch carries, corrected the same
/// way `apply_batch` corrects them). A capacity cap (507) or a
/// conflicting concurrent write (409) can only surface by actually
/// applying the op, so on those two counts a dry run is advisory —
/// the real import can still apply fewer than previewed. Every other
/// field (`retracted`, the drop counts) reads through to the same
/// state a real batch would query, so it matches exactly.
pub(crate) fn preview_batch(state: &AppState, batch: &Batch) -> Result<Applied, ApplyRefusal> {
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
    let (questions_dropped, sections_dropped) = match paragraph_count {
        Some(paragraph_count) => {
            crate::passages::preview_drops(paragraph_count, &batch.questions, &batch.sections)
        }
        None => (0, 0),
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

    /// The batch-open marker around `apply_batch`'s four mutations:
    /// absent after success, present after a mid-batch refusal (the
    /// crash-shaped tear shares the same signature), gone again once
    /// the documented repair — re-importing the source — completes.
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

        // A batch refused partway keeps it: the refusal is reported
        // once, the marker keeps saying so until the repair runs. (An
        // alias to a canonical nothing interned fails AFTER the
        // retraction — a genuinely half-applied source.)
        let torn = parse(
            "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-2\"}\n\
             {\"alias\": \"Aomine\", \"canonical\": \"存在しない\", \"kind\": \"concept\"}\n",
        )
        .unwrap();
        let refusal = apply_batch(&state, &torn).unwrap_err();
        assert!(matches!(refusal, ApplyRefusal::Partial { .. }));
        assert_eq!(
            crate::registry::import_marker_paths(&dir, "sake").len(),
            1,
            "a refused batch keeps its marker"
        );

        // The documented repair — re-import a corrected file for the
        // same source — completes and clears the tear.
        let fixed = parse(
            "{\"taguru_batch\": 1, \"context\": \"sake\", \"source\": \"doc-2\"}\n\
             {\"subject\": \"青嶺酒造\", \"label\": \"銘柄\", \"object\": \"青嶺\", \"weight\": 1.0}\n\
             {\"alias\": \"Aomine\", \"canonical\": \"青嶺酒造\", \"kind\": \"concept\"}\n",
        )
        .unwrap();
        apply_batch(&state, &fixed).unwrap();
        assert!(
            crate::registry::import_marker_paths(&dir, "sake").is_empty(),
            "re-import completes and clears the tear"
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

            crate::registry::fail_persistence_ops_after(failure);
            let result = apply_batch(&state, &reimport);
            let past_end = crate::registry::clear_persistence_fault();

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

    /// A batch whose associations land before its alias step trips a
    /// refusal must report the running total, not just the alias
    /// count — otherwise a batch that durably wrote several
    /// associations and zero aliases reports `applied: 0`, which both
    /// undercounts the CLI's `ops_since_flush` and (over HTTP) skips
    /// `note_write` for a context that did change.
    #[test]
    fn a_partial_alias_refusal_reports_associations_already_applied() {
        let dir = std::env::temp_dir().join(format!(
            "taguru-ingest-partial-alias-{}",
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
        let ApplyRefusal::Partial { applied, .. } = &refusal else {
            panic!("expected a partial refusal, got {refusal:?}");
        };
        assert_eq!(
            *applied, 1,
            "the association landed before the alias step failed"
        );
        assert_eq!(refusal.ops_written(), 1);

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

            crate::registry::fail_persistence_ops_after(failure);
            let first = apply_batch(&state, &batch);
            let past_end = crate::registry::clear_persistence_fault();
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
}
