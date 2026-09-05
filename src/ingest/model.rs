//! The batch data model and the JSONL stream parser: the `Batch`/
//! `Stream` shapes [`super::rejection::apply_batch`] and
//! [`super::rejection::preview_batch`] act on, plus every line shape
//! (`taguru_batch` header, association, alias, passage, question,
//! section, locator, `taguru_group`, `taguru_schema`) `parse_stream`
//! reads.

use super::*;

/// Per-line byte cap. Lines are one fact or one passage; past this
/// something is wrong with the producer, and refusing early beats
/// buffering a runaway line.
pub(super) const MAX_LINE_BYTES: usize = 16 * 1024 * 1024;

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
    pub(super) create: Option<ContextMeta>,
    pub(super) passage: Option<String>,
    /// doc2query questions, (paragraph index, question). Structure is
    /// validated here (caps, a passage to attach to); whether each
    /// index exists in the passage's split is settled at store time,
    /// one rule for every entrance.
    pub(super) questions: Vec<(u32, String)>,
    /// Section start markers, (paragraph index, label) — same
    /// structure-here/range-at-store-time split as `questions`.
    pub(super) sections: Vec<(u32, String)>,
    /// Typed citation locators (ADR 0007 §7), (paragraph index,
    /// locator) — same structure-here/range-at-store-time split as
    /// `questions`/`sections`, but independent of `sections`: a
    /// locator does not extend to the next paragraph.
    pub(super) locators: Vec<(u32, crate::passages::Locator)>,
    /// Source metadata (#167), riding the passage line. `stored_at`
    /// present means an export being restored — the original stamp is
    /// preserved; absent means the store stamps the import time.
    pub(super) stored_at: Option<u64>,
    pub(super) date: Option<u64>,
    pub(super) tags: Vec<String>,
    pub(super) associations: Vec<AssocOp>,
    pub(super) concepts: BTreeMap<String, String>,
    pub(super) labels: BTreeMap<String, String>,
    /// The header's line in its file, and the first line of each kind
    /// of item that needs a passage to attach to (#863): the
    /// end-of-batch refusals name a line the operator can open, not
    /// just a count.
    pub(super) header_line: usize,
    pub(super) first_question_line: Option<usize>,
    pub(super) first_section_line: Option<usize>,
    pub(super) first_locator_line: Option<usize>,
    pub(super) first_paragraph_line: Option<usize>,
}

impl Batch {
    pub(crate) fn op_count(&self) -> usize {
        self.associations.len() + self.concepts.len() + self.labels.len()
    }

    /// Drops the header's create block. The promote verb (ADR 0018)
    /// strips it from every batch of its re-headed stream so a
    /// destination deleted mid-request refuses (`NoContext`) instead
    /// of being resurrected under the scratch's meta — promotion lands
    /// in an established context, never mints one.
    pub(crate) fn strip_create(&mut self) {
        self.create = None;
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

    /// [`Self::label_vocabulary`], but counting how many associations
    /// (plus alias canonicals) used each spelling — extract's
    /// `absorb_vocabulary` (issue #759) folds these into the run's
    /// reuse-frequency signal when a skipped document's batch is
    /// reread, the same way live extraction output counts its own.
    /// `schema:type` is excluded, matching
    /// `Extraction::label_usage_counts`'s ADR 0009 §6.3 exclusion 2 —
    /// without it, a document skipped under `--schema` would leak the
    /// reserved label into later prompts that a freshly extracted one
    /// never would.
    pub(crate) fn label_usage_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for label in self
            .associations
            .iter()
            .map(|op| &op.label)
            .chain(self.labels.values())
            .filter(|label| label.as_str() != crate::schema::SCHEMA_TYPE_LABEL)
        {
            *counts.entry(label.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// The concept spellings this batch settles on: every association
    /// subject/object, plus alias CANONICALS — never alias spellings,
    /// which are exactly the variants a canonical exists to fold.
    /// Extract's `--vocabulary` (ADR 0015, #496 S3) harvests these
    /// from an exported context so a new document is steered toward
    /// the spellings the graph already uses.
    pub(crate) fn concept_vocabulary(&self) -> BTreeSet<String> {
        self.associations
            .iter()
            .flat_map(|op| [op.subject.clone(), op.object.clone()])
            .chain(self.concepts.values().cloned())
            .collect()
    }

    /// The alias mappings this batch records, spelling → canonical,
    /// per namespace — what `add_alias` will intern once applied.
    /// Extract's cross-document claim set (#758) reads these from a
    /// manifest-skipped document's batch so a LATER document's alias
    /// that would rewire one of them is refused before import has to.
    pub(crate) fn concept_aliases(&self) -> &BTreeMap<String, String> {
        &self.concepts
    }

    /// [`Batch::concept_aliases`] for the label namespace.
    pub(crate) fn label_aliases(&self) -> &BTreeMap<String, String> {
        &self.labels
    }

    /// Every association as `[subject, label, object]` — the coverage
    /// check's input (ADR 0016, #496 S4) when a manifest-skipped
    /// document is judged from its already-written batch instead of a
    /// fresh extraction.
    /// The batch's passage, if it carries one — `taguru anchoring`'s
    /// original text (#793).
    pub(crate) fn passage(&self) -> Option<&str> {
        self.passage.as_deref()
    }

    /// The association operations verbatim, paragraph citations
    /// included — `taguru anchoring` judges these (#793);
    /// [`Batch::association_triples`] stays the coverage check's
    /// paragraph-less view.
    pub(crate) fn associations(&self) -> &[crate::registry::AssocOp] {
        &self.associations
    }

    pub(crate) fn association_triples(&self) -> Vec<[&str; 3]> {
        self.associations
            .iter()
            .map(|op| [op.subject.as_str(), op.label.as_str(), op.object.as_str()])
            .collect()
    }

    pub(super) fn describe(&self) -> String {
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
    // Each claim → the line that made it (#863): a duplicate names the
    // earlier line, not just "an earlier batch".
    let mut owners: HashMap<(String, String), usize> = HashMap::new();
    let mut schema_owners: HashMap<String, usize> = HashMap::new();
    let mut group_owners: HashMap<String, usize> = HashMap::new();
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
            if let Some(earlier) = owners.get(&(batch.context.clone(), batch.source.clone())) {
                return Err(format!(
                    "line {number}: source '{}' in context '{}' is already stated by \
                     an earlier batch of this stream, at line {earlier} — one batch owns \
                     one source's truth",
                    batch.source, batch.context
                ));
            }
            owners.insert((batch.context.clone(), batch.source.clone()), number);
            current = Some(batch);
        } else if is_schema {
            let (context, installed) = parse_schema(value, number)?;
            if let Some(earlier) = schema_owners.get(&context) {
                return Err(format!(
                    "line {number}: context '{context}' schema is already stated by an \
                     earlier record of this stream, at line {earlier} — one record owns one \
                     context's schema"
                ));
            }
            schema_owners.insert(context.clone(), number);
            schemas.push((context, installed));
        } else if is_group {
            let (name, record) = parse_group(value, number)?;
            if let Some(earlier) = group_owners.get(&name) {
                return Err(format!(
                    "line {number}: group '{name}' is already stated by an earlier record \
                     of this stream, at line {earlier} — one record owns one group's truth"
                ));
            }
            group_owners.insert(name.clone(), number);
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
    let header = batch.header_line;
    if !batch.questions.is_empty() && batch.passage.is_none() {
        return Err(format!(
            "line {}: {} question line(s) but no passage line — questions attach to the \
             passage of the batch headed at line {header}",
            batch.first_question_line.unwrap_or(header),
            batch.questions.len()
        ));
    }
    // Sections attach to paragraphs the same way questions do, and need
    // the same passage-to-attach-to guard.
    if !batch.sections.is_empty() && batch.passage.is_none() {
        return Err(format!(
            "line {}: {} section line(s) but no passage line — sections attach to the \
             passage of the batch headed at line {header}",
            batch.first_section_line.unwrap_or(header),
            batch.sections.len()
        ));
    }
    // Locators attach to paragraphs the same way sections do, and need
    // the same passage-to-attach-to guard.
    if !batch.locators.is_empty() && batch.passage.is_none() {
        return Err(format!(
            "line {}: {} locator line(s) but no passage line — locators attach to the \
             passage of the batch headed at line {header}",
            batch.first_locator_line.unwrap_or(header),
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
            "line {}: an association names paragraph {paragraph} but the batch headed at \
             line {header} has no passage line — a paragraph locator attaches to that passage",
            batch.first_paragraph_line.unwrap_or(header)
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
        header_line: number,
        first_question_line: None,
        first_section_line: None,
        first_locator_line: None,
        first_paragraph_line: None,
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
        if op.paragraph.is_some() {
            batch.first_paragraph_line.get_or_insert(number);
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
            batch.first_question_line.get_or_insert(number);
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
        batch.first_section_line.get_or_insert(number);
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
        batch.first_locator_line.get_or_insert(number);
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
