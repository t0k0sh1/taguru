//! The v0.9.7 release's export format, read as migration input (ADR
//! 0057). That release addressed `contexts` and `groups` by display
//! name: a source header said `{"taguru_batch":1,"context":<name>,
//! "source":<id>,"create"?:{...}}`, a schema record
//! `{"taguru_schema":1,"context":<name>,...}`, and a group record
//! `{"taguru_group":1,"name":..,"contexts":[names],"groups":[names]}`.
//! The operation lines (associations, aliases, passages, questions,
//! sections, locators) are unchanged since, so [`super::model`]'s one
//! stream loop reads them for both formats. This module reads the
//! three record shapes and then, against the receiving instance's
//! registry, settles every name on an id ([`resolve_legacy`]): a name
//! one standing `context` (or `group`) carries reuses its id, an
//! absent one gets a freshly minted id, and a name several carry is
//! refused rather than guessed. The result is an ordinary [`Stream`],
//! so every later step of an import is the current format's.
//!
//! Read-only support: nothing writes this format any more, and only the
//! v0.9.7 release's shapes are read — not the unreleased revisions
//! between it and this build.

use super::model::{
    Batch, CreateSpec, RecordKind, Stream, check_nonempty, check_size, create_meta,
};
use super::*;

/// The version the v0.9.7 release stamped on all three record kinds
/// (its `BATCH_VERSION`, `SCHEMA_VERSION` and `GROUP_VERSION`).
const LEGACY_RECORD_VERSION: u64 = 1;

/// A parsed v0.9.7 stream, still addressed by names.
#[cfg_attr(test, derive(Debug))]
pub(crate) struct LegacyStream {
    /// Each batch beside the context NAME its header carried; the
    /// batch's own `context_id` is empty until [`resolve_legacy`] sets
    /// it. A header's create block is already a [`CreateSpec`] whose
    /// name is that context name.
    pub(super) batches: Vec<(String, Batch)>,
    /// (context name, installed schema).
    pub(super) schemas: Vec<(String, schema::InstalledSchema)>,
    pub(super) groups: Vec<LegacyGroup>,
}

/// One v0.9.7 group record: its members named, not identified.
#[cfg_attr(test, derive(Debug))]
pub(crate) struct LegacyGroup {
    pub(super) name: String,
    pub(super) description: String,
    pub(super) contexts: BTreeSet<String>,
    pub(super) groups: BTreeSet<String>,
}

/// Which v0.9.7 record a line is, by the stamp key only that record
/// carries; `None` for anything else.
pub(super) fn legacy_record_kind(value: &Value) -> Option<RecordKind> {
    let object = value.as_object()?;
    if object.contains_key("taguru_batch") {
        Some(RecordKind::Header)
    } else if object.contains_key("taguru_schema") {
        Some(RecordKind::Schema)
    } else if object.contains_key("taguru_group") {
        Some(RecordKind::Group)
    } else {
        None
    }
}

/// Whether `body` is a v0.9.7 stream, judged by its first non-blank
/// line — a cheap test for an entrance that cannot resolve names (the
/// router, whose shards would each mint a different id for one name)
/// and must refuse such a body before parsing it. A body whose first
/// line is not JSON is not one; the parser refuses it on its own terms.
pub(crate) fn is_legacy_stream(body: &[u8]) -> bool {
    let body = body.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(body);
    body.split(|byte| *byte == b'\n')
        .map(<[u8]>::trim_ascii)
        .find(|line| !line.is_empty())
        .and_then(|line| serde_json::from_slice::<Value>(line).ok())
        .is_some_and(|value| legacy_record_kind(&value).is_some())
}

/// The refusal an entrance that reads only the current format answers
/// a v0.9.7 stream with — naming the entrances that do read one.
pub(crate) fn legacy_elsewhere_message() -> String {
    "a v0.9.7 export stream (`taguru_batch`/`taguru_schema`/`taguru_group` records): this \
     entrance reads the current format only. A v0.9.7 export imports through `taguru import \
     FILE|DIR...` (offline, no --url) or `POST /import` on a single instance, which resolve \
     its context and group names and mint ids (ADR 0057; \
     docs/import.html#upgrading-from-0-9-7)"
        .to_string()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyHeader {
    taguru_batch: u64,
    context: String,
    source: String,
    #[serde(default)]
    create: Option<LegacyCreate>,
}

/// The v0.9.7 create block: the config fields alone — the name was the
/// header's `context`.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct LegacyCreate {
    description: String,
    pinned: bool,
    dice_floor: Option<f64>,
    semantic_floor: Option<f32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacySchemaLine {
    taguru_schema: u64,
    context: String,
    mode: schema::SchemaMode,
    closed_labels: bool,
    types: BTreeMap<String, schema::TypeDef>,
    relations: BTreeMap<String, schema::RelationDef>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyGroupLine {
    taguru_group: u64,
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    contexts: Vec<String>,
    #[serde(default)]
    groups: Vec<String>,
}

fn check_record_version(number: usize, stamp: &str, found: u64) -> Result<(), String> {
    if found == LEGACY_RECORD_VERSION {
        return Ok(());
    }
    Err(format!(
        "line {number}: {stamp} {found} is not a version this taguru reads (it reads \
         {LEGACY_RECORD_VERSION}, the v0.9.7 release's)"
    ))
}

/// One v0.9.7 source header: the context name it addressed, and the
/// batch it opens (its `context_id` still empty).
pub(super) fn parse_header(value: Value, number: usize) -> Result<(String, Batch), String> {
    let header: LegacyHeader = serde_json::from_value(value)
        .map_err(|error| format!("line {number}: not a v0.9.7 source file header: {error}"))?;
    check_record_version(number, "taguru_batch", header.taguru_batch)?;
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
    let create = header.create.map(|block| CreateSpec {
        name: header.context.clone(),
        meta: create_meta(
            block.description,
            block.pinned,
            block.dice_floor,
            block.semantic_floor,
        ),
    });
    let batch = Batch::opened(String::new(), header.source, create, number);
    Ok((header.context, batch))
}

/// One v0.9.7 schema record: the context name it installs onto, and
/// the document — the same shape this build reads, whose `version`
/// [`schema::install`] stamps as this build's own.
pub(super) fn parse_schema(
    value: Value,
    number: usize,
) -> Result<(String, schema::InstalledSchema), String> {
    let line: LegacySchemaLine = serde_json::from_value(value)
        .map_err(|error| format!("line {number}: not a v0.9.7 schema record: {error}"))?;
    check_record_version(number, "taguru_schema", line.taguru_schema)?;
    check_size(number, "context", &line.context, MAX_CONTEXT_NAME_BYTES)?;
    check_nonempty(number, "context", &line.context)?;
    let installed = schema::install(schema::SchemaDocument {
        record_type: schema::SchemaType::Schema,
        version: None,
        mode: line.mode,
        closed_labels: line.closed_labels,
        types: line.types,
        relations: line.relations,
    })
    .map_err(|violation| format!("line {number}: {violation}"))?;
    Ok((line.context, installed))
}

/// One v0.9.7 group record, its members still names. List duplicates
/// fold into the set; sizes and the member cap refuse by line, as for
/// a current record.
pub(super) fn parse_group(value: Value, number: usize) -> Result<LegacyGroup, String> {
    let line: LegacyGroupLine = serde_json::from_value(value)
        .map_err(|error| format!("line {number}: not a v0.9.7 group record: {error}"))?;
    check_record_version(number, "taguru_group", line.taguru_group)?;
    check_size(number, "name", &line.name, MAX_CONTEXT_NAME_BYTES)?;
    check_nonempty(number, "name", &line.name)?;
    check_size(
        number,
        "description",
        &line.description,
        MAX_DESCRIPTION_BYTES,
    )?;
    let mut record = LegacyGroup {
        name: line.name,
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
    Ok(record)
}

/// Why a v0.9.7 import's names could not all be settled on ids. Every
/// one is judged before anything applies.
#[cfg_attr(test, derive(Debug, PartialEq))]
pub(crate) enum LegacyRefusal {
    /// Several standing `contexts` (`kind` "context") or `groups`
    /// ("group") carry the name; import never picks one.
    Ambiguous {
        kind: &'static str,
        name: String,
        count: usize,
    },
    /// No standing `context` carries the name and no source of the
    /// import brings a create block for it — what v0.9.7 refused as
    /// "does not exist and the batch brought no create block".
    NoContext { name: String, needed_by: String },
    /// A child `group` that is neither a record of the import nor a
    /// standing `group`.
    NoGroup { name: String, needed_by: String },
    /// Two records of one import state the same `group`.
    DuplicateGroup(String),
}

impl LegacyRefusal {
    pub(crate) fn text(&self) -> String {
        match self {
            Self::Ambiguous { kind, name, count } => format!(
                "{count} {kind}s on this server are named '{name}': a v0.9.7 stream names its \
                 {kind}s by name, and import will not pick one — rename all but one, or import \
                 into a data directory where the name is unique \
                 (docs/import.html#upgrading-from-0-9-7); nothing was applied"
            ),
            Self::NoContext { name, needed_by } => format!(
                "context '{name}' ({needed_by}) is not on this server, and no v0.9.7 source of \
                 this import brings a create block for it; nothing was applied"
            ),
            Self::NoGroup { name, needed_by } => format!(
                "group '{name}' ({needed_by}) is neither a record of this import nor a group on \
                 this server; nothing was applied"
            ),
            Self::DuplicateGroup(name) => format!(
                "group '{name}' is stated by two v0.9.7 records of this import — one record \
                 owns one group's truth; nothing was applied"
            ),
        }
    }
}

/// The context-name half of [`resolve_legacy`]: each name settled once,
/// in the order the import first mentions it.
struct ContextNames<'a> {
    standing: HashMap<String, Vec<String>>,
    /// The first create block the import brings per name — a v0.9.7
    /// export puts it on a context's first source only.
    creates: HashMap<&'a str, &'a CreateSpec>,
    /// name → (id, the create block every batch of the name carries
    /// when the id is freshly minted).
    settled: HashMap<String, (String, Option<CreateSpec>)>,
}

impl ContextNames<'_> {
    fn settle(
        &mut self,
        name: &str,
        needed_by: impl FnOnce() -> String,
    ) -> Result<(), LegacyRefusal> {
        if self.settled.contains_key(name) {
            return Ok(());
        }
        let answer = match self.standing.get(name).map(Vec::as_slice) {
            Some([id]) => (id.clone(), None),
            // Two or more: an empty list cannot occur (names are indexed
            // from rows), and a slice pattern says so without a guard.
            Some(ids @ [_, _, ..]) => {
                return Err(LegacyRefusal::Ambiguous {
                    kind: "context",
                    name: name.to_string(),
                    count: ids.len(),
                });
            }
            _ => match self.creates.get(name) {
                Some(spec) => (crate::registry::mint_context_id(), Some((*spec).clone())),
                None => {
                    return Err(LegacyRefusal::NoContext {
                        name: name.to_string(),
                        needed_by: needed_by(),
                    });
                }
            },
        };
        self.settled.insert(name.to_string(), answer);
        Ok(())
    }
}

/// Settles every name a set of v0.9.7 streams — one import invocation's
/// worth — mentions on an id, against `state`'s standing `contexts` and
/// `groups`, and returns the streams rewritten in the current format,
/// in the order given.
///
/// - A context name exactly one standing `context` carries resolves to
///   its id: a re-import lands on what the first one created.
/// - An absent one gets a minted id, and EVERY batch of that name
///   carries the first create block the import brings for it, so the
///   create does not hinge on which batch applies first; an absent
///   name no source brings a create block for is refused, as v0.9.7
///   refused such a batch.
/// - A group record's name resolves the same way (absent → minted);
///   its member contexts resolve as above, and each child group to a
///   record of the same invocation or to the one standing `group` of
///   that name. The records then restore as one set, children first
///   ([`AppState::restore_groups`]), so the order the record files
///   sorted in never matters.
/// - A name several standing `contexts` or `groups` carry is refused.
///
/// Read-only: nothing is created here; a refusal leaves the instance
/// untouched.
pub(crate) fn resolve_legacy(
    state: &AppState,
    streams: Vec<LegacyStream>,
) -> Result<Vec<Stream>, LegacyRefusal> {
    let mut standing_contexts: HashMap<String, Vec<String>> = HashMap::new();
    for row in state.directory() {
        standing_contexts.entry(row.name).or_default().push(row.id);
    }
    let mut standing_groups: HashMap<String, Vec<String>> = HashMap::new();
    for view in state.group_page(None, None, usize::MAX).1 {
        standing_groups.entry(view.name).or_default().push(view.id);
    }

    let mut names = ContextNames {
        standing: standing_contexts,
        creates: HashMap::new(),
        settled: HashMap::new(),
    };
    for stream in &streams {
        for (name, batch) in &stream.batches {
            if let Some(spec) = &batch.create {
                names.creates.entry(name.as_str()).or_insert(spec);
            }
        }
    }
    for stream in &streams {
        for (name, batch) in &stream.batches {
            names.settle(name, || format!("source '{}'", batch.source))?;
        }
        for (name, _) in &stream.schemas {
            names.settle(name, || "its schema record".to_string())?;
        }
        for record in &stream.groups {
            for member in &record.contexts {
                names.settle(member, || format!("a member of group '{}'", record.name))?;
            }
        }
    }
    let settled = names.settled;

    let mut group_ids: HashMap<String, String> = HashMap::new();
    for record in streams.iter().flat_map(|stream| &stream.groups) {
        if group_ids.contains_key(&record.name) {
            return Err(LegacyRefusal::DuplicateGroup(record.name.clone()));
        }
        let id = match standing_groups.get(&record.name).map(Vec::as_slice) {
            Some([id]) => id.clone(),
            Some(ids) if ids.len() > 1 => {
                return Err(LegacyRefusal::Ambiguous {
                    kind: "group",
                    name: record.name.clone(),
                    count: ids.len(),
                });
            }
            _ => crate::groups::mint_group_id(),
        };
        group_ids.insert(record.name.clone(), id);
    }
    for record in streams.iter().flat_map(|stream| &stream.groups) {
        for child in &record.groups {
            if group_ids.contains_key(child) {
                continue;
            }
            let id = match standing_groups.get(child).map(Vec::as_slice) {
                Some([id]) => id.clone(),
                Some(ids @ [_, _, ..]) => {
                    return Err(LegacyRefusal::Ambiguous {
                        kind: "group",
                        name: child.clone(),
                        count: ids.len(),
                    });
                }
                _ => {
                    return Err(LegacyRefusal::NoGroup {
                        name: child.clone(),
                        needed_by: format!("a child of group '{}'", record.name),
                    });
                }
            };
            group_ids.insert(child.clone(), id);
        }
    }

    let context_id = |name: &str| settled[name].0.clone();
    Ok(streams
        .into_iter()
        .map(|stream| Stream {
            batches: stream
                .batches
                .into_iter()
                .map(|(name, mut batch)| {
                    let (id, minted) = &settled[&name];
                    batch.context_id = id.clone();
                    if let Some(spec) = minted {
                        batch.create = Some(spec.clone());
                    }
                    batch
                })
                .collect(),
            schemas: stream
                .schemas
                .into_iter()
                .map(|(name, installed)| (context_id(&name), installed))
                .collect(),
            groups: stream
                .groups
                .into_iter()
                .map(|record| StoredGroup {
                    id: group_ids[&record.name].clone(),
                    context_ids: record
                        .contexts
                        .iter()
                        .map(|name| context_id(name))
                        .collect(),
                    group_ids: record
                        .groups
                        .iter()
                        .map(|name| group_ids[name].clone())
                        .collect(),
                    name: record.name,
                    description: record.description,
                })
                .collect(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::super::model::{ParsedStream, parse_any_stream};
    use super::*;

    fn legacy(text: &str) -> LegacyStream {
        match parse_any_stream(std::io::Cursor::new(text)).unwrap() {
            ParsedStream::Legacy(stream) => stream,
            ParsedStream::Current(stream) => panic!("read as current: {stream:?}"),
        }
    }

    fn scratch_state(tag: &str) -> AppState {
        let dir =
            std::env::temp_dir().join(format!("taguru-ingest-legacy-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        AppState::boot(dir, usize::MAX, None).unwrap()
    }

    /// A v0.9.7 export of two contexts, a schema, and nested groups —
    /// the parent `all` first, as its file sorts first.
    const EXPORT: &str = concat!(
        "{\"taguru_schema\":1,\"context\":\"sake/kura\",\"mode\":\"warn\",\"closed_labels\":false,",
        "\"types\":{\"Brewery\":{}},\"relations\":{}}\n",
        "{\"taguru_batch\":1,\"context\":\"sake/kura\",\"source\":\"a.md\",",
        "\"create\":{\"description\":\"酒蔵\",\"pinned\":true,\"dice_floor\":0.5}}\n",
        "{\"subject\":\"青嶺酒造\",\"label\":\"杜氏\",\"object\":\"高瀬\",\"weight\":2.0,\"paragraph\":0}\n",
        "{\"alias\":\"Aomine\",\"canonical\":\"青嶺酒造\",\"kind\":\"concept\"}\n",
        "{\"passage\":\"青嶺酒造の杜氏は高瀬。\",\"stored_at\":1700000000,\"date\":1600000000,",
        "\"tags\":[\"kura\"]}\n",
        "{\"paragraph\":0,\"question\":\"杜氏は誰?\"}\n",
        "{\"paragraph\":0,\"section\":\"概要\"}\n",
        "{\"paragraph\":0,\"locator\":{\"kind\":\"page\",\"value\":\"3\"}}\n",
        "{\"taguru_batch\":1,\"context\":\"sake/kura\",\"source\":\"b.md\"}\n",
        "{\"subject\":\"高瀬\",\"label\":\"出身\",\"object\":\"新潟\",\"weight\":1.0}\n",
        "{\"taguru_batch\":1,\"context\":\"wine\",\"source\":\"export:empty\",",
        "\"create\":{\"description\":\"\"}}\n",
        "{\"taguru_group\":1,\"name\":\"all\",\"groups\":[\"drinks\"]}\n",
        "{\"taguru_group\":1,\"name\":\"drinks\",\"description\":\"d\",",
        "\"contexts\":[\"sake/kura\",\"wine\"]}\n",
    );

    #[test]
    fn a_v097_stream_parses_by_name_with_its_op_lines_whole() {
        let stream = legacy(EXPORT);
        let names: Vec<&str> = stream
            .batches
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        assert_eq!(names, ["sake/kura", "sake/kura", "wine"]);
        let (_, first) = &stream.batches[0];
        assert_eq!(first.context_id, "");
        assert_eq!(first.source, "a.md");
        let create = first.create.as_ref().expect("the first source creates");
        assert_eq!(create.name, "sake/kura");
        assert_eq!(create.meta.description, "酒蔵");
        assert!(create.meta.pinned);
        assert_eq!(create.meta.dice_floor, Some(0.5));
        assert_eq!(first.associations.len(), 1);
        assert_eq!(first.associations[0].paragraph, Some(0));
        assert_eq!(first.concepts.len(), 1);
        assert_eq!(first.stored_at, Some(1_700_000_000));
        assert_eq!(first.date, Some(1_600_000_000));
        assert_eq!(first.tags, ["kura"]);
        assert_eq!(first.questions.len(), 1);
        assert_eq!(first.sections.len(), 1);
        assert_eq!(first.locators.len(), 1);
        assert!(stream.batches[1].1.create.is_none());
        assert_eq!(stream.schemas.len(), 1);
        assert_eq!(stream.schemas[0].0, "sake/kura");
        assert_eq!(
            stream.schemas[0].1.document().version.as_deref(),
            Some(crate::format::FORMAT_VERSION)
        );
        assert_eq!(stream.groups.len(), 2);
        assert_eq!(stream.groups[0].name, "all");
        assert_eq!(
            stream.groups[1].contexts,
            BTreeSet::from(["sake/kura".to_string(), "wine".to_string()])
        );
    }

    /// The current-format entrances name the two that read v0.9.7.
    #[test]
    fn parse_stream_refuses_a_v097_stream_naming_the_upgrade_path() {
        let error = parse_stream(std::io::Cursor::new(EXPORT)).unwrap_err();
        assert!(error.contains("v0.9.7 export stream"), "{error}");
        assert!(error.contains("#upgrading-from-0-9-7"), "{error}");
    }

    #[test]
    fn a_file_mixing_the_two_formats_is_refused_by_line() {
        let current = "{\"type\":\"source\",\"context_id\":\"cef2e28b-43f0-4b6c-8201-abab0785399f\",\
                       \"id\":\"a.md\"}\n";
        let old = "{\"taguru_batch\":1,\"context\":\"sake\",\"source\":\"b.md\"}\n";
        for (text, line) in [
            (format!("{current}{old}"), 2),
            (format!("{old}{current}"), 2),
            (
                format!(
                    "{old}{{\"type\":\"group\",\"id\":\"00000000-0000-4000-9000-000000000001\",\"name\":\"g\"}}\n"
                ),
                2,
            ),
        ] {
            let error = parse_any_stream(std::io::Cursor::new(text.as_str())).unwrap_err();
            assert!(error.starts_with(&format!("line {line}: ")), "{error}");
            assert!(error.contains("may not be mixed in one file"), "{error}");
            // The message names both formats, so the caller knows which
            // lines to split into their own file.
            assert!(
                error.contains("v0.9.7 (`taguru_batch`/`taguru_schema`/`taguru_group`)")
                    && error.contains("current-format (`type`)"),
                "{error}"
            );
        }
    }

    #[test]
    fn a_v097_record_of_another_version_is_refused() {
        let error = parse_any_stream(std::io::Cursor::new(
            "{\"taguru_batch\":2,\"context\":\"sake\",\"source\":\"a.md\"}\n",
        ))
        .unwrap_err();
        assert!(error.contains("taguru_batch 2 is not a version"), "{error}");
        let error = parse_any_stream(std::io::Cursor::new(
            "{\"taguru_group\":1,\"name\":\"g\",\"context_ids\":[]}\n",
        ))
        .unwrap_err();
        assert!(error.contains("not a v0.9.7 group record"), "{error}");
    }

    #[test]
    fn the_detector_reads_only_the_first_non_blank_line() {
        assert!(is_legacy_stream(EXPORT.as_bytes()));
        assert!(is_legacy_stream(
            b"\xEF\xBB\xBF\n  \n{\"taguru_group\":1,\"name\":\"g\"}\n"
        ));
        assert!(!is_legacy_stream(
            b"{\"type\":\"source\",\"context_id\":\"x\",\"id\":\"a\"}\n{\"taguru_batch\":1}\n"
        ));
        assert!(!is_legacy_stream(b"not json\n"));
        assert!(!is_legacy_stream(b""));
    }

    #[test]
    fn absent_names_mint_ids_and_every_batch_carries_the_create() {
        let state = scratch_state("mint");
        let streams = resolve_legacy(&state, vec![legacy(EXPORT)]).unwrap();
        let stream = &streams[0];
        let kura = stream.batches[0].context_id.clone();
        assert!(crate::registry::is_context_id(&kura), "{kura}");
        assert_eq!(stream.batches[1].context_id, kura);
        let second = stream.batches[1].create.as_ref().expect("minted ⇒ create");
        assert_eq!(second.name, "sake/kura");
        assert_eq!(second.meta.description, "酒蔵");
        let wine = stream.batches[2].context_id.clone();
        assert_ne!(wine, kura);
        assert_eq!(stream.schemas[0].0, kura);
        let all = &stream.groups[0];
        let drinks = &stream.groups[1];
        assert!(crate::groups::is_group_id(&all.id));
        assert_eq!(all.group_ids, BTreeSet::from([drinks.id.clone()]));
        assert_eq!(drinks.context_ids, BTreeSet::from([kura, wine]));
        assert_eq!(drinks.description, "d");
    }

    #[test]
    fn standing_names_resolve_to_their_ids_and_keep_them() {
        let state = scratch_state("reuse");
        let kura = state.create("sake/kura", ContextMeta::default()).unwrap();
        let wine = state.create("wine", ContextMeta::default()).unwrap();
        let drinks = state
            .create_group(
                None,
                "drinks",
                String::new(),
                BTreeSet::from([wine.clone()]),
                BTreeSet::new(),
            )
            .unwrap()
            .id;
        let streams = resolve_legacy(&state, vec![legacy(EXPORT)]).unwrap();
        let stream = &streams[0];
        assert_eq!(stream.batches[0].context_id, kura);
        // A standing context keeps the header's own create (ignored on
        // apply), and gains none.
        assert!(stream.batches[0].create.is_some());
        assert!(stream.batches[1].create.is_none());
        assert_eq!(stream.batches[2].context_id, wine);
        assert_eq!(stream.groups[1].id, drinks);
        assert_eq!(stream.groups[0].group_ids, BTreeSet::from([drinks]));
    }

    #[test]
    fn a_name_several_contexts_carry_is_refused() {
        let state = scratch_state("ambiguous");
        state.create("wine", ContextMeta::default()).unwrap();
        state.create("wine", ContextMeta::default()).unwrap();
        let refusal = resolve_legacy(&state, vec![legacy(EXPORT)]).unwrap_err();
        assert_eq!(
            refusal,
            LegacyRefusal::Ambiguous {
                kind: "context",
                name: "wine".to_string(),
                count: 2
            }
        );
        assert!(
            refusal
                .text()
                .contains("2 contexts on this server are named 'wine'")
        );
    }

    #[test]
    fn a_name_several_groups_carry_is_refused() {
        let state = scratch_state("ambiguous-group");
        for _ in 0..2 {
            state
                .create_group(None, "all", String::new(), BTreeSet::new(), BTreeSet::new())
                .unwrap();
        }
        let refusal = resolve_legacy(&state, vec![legacy(EXPORT)]).unwrap_err();
        assert_eq!(
            refusal,
            LegacyRefusal::Ambiguous {
                kind: "group",
                name: "all".to_string(),
                count: 2
            }
        );
    }

    /// A child group the stream names but does not define resolves
    /// against the standing groups; several carrying that name is the
    /// same ambiguity refusal as for a group record's own name — never
    /// "no such group".
    #[test]
    fn a_child_name_several_standing_groups_carry_is_refused_as_ambiguous() {
        let state = scratch_state("ambiguous-child");
        for _ in 0..2 {
            state
                .create_group(
                    None,
                    "kids",
                    String::new(),
                    BTreeSet::new(),
                    BTreeSet::new(),
                )
                .unwrap();
        }
        let refusal = resolve_legacy(
            &state,
            vec![legacy(
                "{\"taguru_group\":1,\"name\":\"parent\",\"groups\":[\"kids\"]}\n",
            )],
        )
        .unwrap_err();
        assert_eq!(
            refusal,
            LegacyRefusal::Ambiguous {
                kind: "group",
                name: "kids".to_string(),
                count: 2
            }
        );
    }

    /// A v0.9.7 group record holds at most [`MAX_GROUP_MEMBERS`] members
    /// per list — exactly the cap parses, one past it is refused by line.
    #[test]
    fn a_v097_group_list_is_capped_at_the_member_limit() {
        let record = |count: usize| {
            let names: Vec<String> = (0..count).map(|index| format!("\"c{index}\"")).collect();
            format!(
                "{{\"taguru_group\":1,\"name\":\"g\",\"contexts\":[{}]}}\n",
                names.join(",")
            )
        };
        let at_cap = legacy(&record(MAX_GROUP_MEMBERS));
        assert_eq!(at_cap.groups[0].contexts.len(), MAX_GROUP_MEMBERS);
        let error = parse_any_stream(std::io::Cursor::new(record(MAX_GROUP_MEMBERS + 1).as_str()))
            .unwrap_err();
        assert!(
            error.starts_with("line 1: ") && error.contains("a group holds at most"),
            "{error}"
        );
    }

    #[test]
    fn an_absent_name_nothing_creates_is_refused_by_name() {
        let state = scratch_state("no-create");
        let refusal = resolve_legacy(
            &state,
            vec![legacy(
                "{\"taguru_batch\":1,\"context\":\"sake\",\"source\":\"a.md\"}\n",
            )],
        )
        .unwrap_err();
        assert_eq!(
            refusal,
            LegacyRefusal::NoContext {
                name: "sake".to_string(),
                needed_by: "source 'a.md'".to_string()
            }
        );
        let refusal = resolve_legacy(
            &state,
            vec![legacy(
                "{\"taguru_group\":1,\"name\":\"g\",\"groups\":[\"h\"]}\n",
            )],
        )
        .unwrap_err();
        assert_eq!(
            refusal,
            LegacyRefusal::NoGroup {
                name: "h".to_string(),
                needed_by: "a child of group 'g'".to_string()
            }
        );
        let refusal = resolve_legacy(
            &state,
            vec![legacy(
                "{\"taguru_group\":1,\"name\":\"g\",\"contexts\":[\"x\"]}\n",
            )],
        )
        .unwrap_err();
        assert!(matches!(refusal, LegacyRefusal::NoContext { ref name, .. } if name == "x"));
    }

    /// Two files of one invocation: the create block in one reaches the
    /// other's batches, and a group record in a later file resolves its
    /// child from an earlier one; a group stated twice is refused.
    #[test]
    fn names_settle_across_every_stream_of_one_invocation() {
        let state = scratch_state("across");
        let streams = resolve_legacy(
            &state,
            vec![
                legacy("{\"taguru_group\":1,\"name\":\"all\",\"groups\":[\"drinks\"]}\n"),
                legacy(
                    "{\"taguru_batch\":1,\"context\":\"sake\",\"source\":\"a.md\",\
                     \"create\":{\"description\":\"\"}}\n",
                ),
                legacy("{\"taguru_group\":1,\"name\":\"drinks\",\"contexts\":[\"sake\"]}\n"),
            ],
        )
        .unwrap();
        assert_eq!(
            streams[0].groups[0].group_ids,
            BTreeSet::from([streams[2].groups[0].id.clone()])
        );
        assert_eq!(
            streams[2].groups[0].context_ids,
            BTreeSet::from([streams[1].batches[0].context_id.clone()])
        );
        let refusal = resolve_legacy(
            &state,
            vec![
                legacy("{\"taguru_group\":1,\"name\":\"g\"}\n"),
                legacy("{\"taguru_group\":1,\"name\":\"g\"}\n"),
            ],
        )
        .unwrap_err();
        assert_eq!(refusal, LegacyRefusal::DuplicateGroup("g".to_string()));
    }

    /// The parent sorts (and restores) before its child: one set,
    /// children landing first, so it cannot fail on order.
    #[test]
    fn nested_groups_restore_whatever_order_their_records_came_in() {
        let state = scratch_state("nested-restore");
        let streams = resolve_legacy(&state, vec![legacy(EXPORT)]).unwrap();
        for batch in &streams[0].batches {
            apply_batch(&state, batch, Deadline::unbounded()).unwrap();
        }
        let outcomes = state
            .restore_groups(&streams[0].groups, Deadline::unbounded())
            .unwrap();
        assert_eq!(outcomes.len(), 2);
        let all = state.group(&streams[0].groups[0].id).unwrap();
        assert_eq!(all.name, "all");
        assert_eq!(
            all.record.group_ids,
            BTreeSet::from([streams[0].groups[1].id.clone()])
        );
    }
}
