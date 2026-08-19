//! `taguru export`: one context back out as the same JSONL batch
//! stream `taguru import` and `POST /import` apply — the portable
//! backup the raw file family cannot be (images are
//! version-specific and must be copied as a consistent set; a batch
//! stream is plain text with a documented contract, readable by any
//! future taguru and by anything else that speaks JSON Lines).
//!
//! One context renders as one stream: one batch per source (sorted by
//! source id), the first batch carrying the context's create block
//! (description, pinned, floors), the last batch carrying the alias
//! table. Re-importing the stream is idempotent — each batch is one
//! source's complete truth, applied retract-first, exactly like any
//! other import. Restoring into a live context is therefore a
//! per-source sync: sources present in the stream are replaced,
//! sources that exist only in the context survive. A restore that
//! must equal the snapshot exactly starts from a deleted (or fresh)
//! context.
//!
//! A FULL export (no CONTEXT arguments) also writes every group as
//! `{out}/{group}.group.jsonl` — one `taguru_group` record each, the
//! group's complete truth the way a batch is a source's. Import
//! applies group records after every batch of its run, so the files
//! restore in any order; re-importing one REPLACES the record, the
//! same idempotence one storey up. A subset export (explicit CONTEXT
//! arguments) writes no groups: a group's truth spans contexts the
//! subset may not carry, and a partial truth would shrink the group
//! on restore.
//!
//! Fidelity notes, all deliberate:
//! - An attribution asserted `count` times re-renders as `count`
//!   association lines of its average weight, so corroboration
//!   counts survive the round trip (weights re-accumulate to the
//!   original sum, within float re-addition error).
//! - Weight contributed by sourceless writes (`associate` without a
//!   source — possible over the API, never produced by extract or
//!   import) cannot ride in a per-source batch honestly, so it lands
//!   in a reserved batch whose source id is `export:unsourced`; the
//!   numbers survive, and the attribution says where they came from.
//!   Import stamps that reserved id onto the batch's lines, so the
//!   restored context carries a real `export:unsourced` attribution —
//!   which the next export folds straight back into the sourceless
//!   batch rather than refusing, making the stream an exact fixed
//!   point across repeated round trips.
//! - Fully retracted edges (count 0) render as nothing — an
//!   export/import round trip sheds the dead records the append-only
//!   image keeps, so it doubles as offline compaction.
//! - An alias whose canonical no longer carries any live association
//!   is dropped (and counted): the import contract interns canonicals
//!   through association lines, and an edgeless canonical has none to
//!   ride in on.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use serde::Serialize;
use serde_json::Value;

use taguru::context::{Association, UNSOURCED_SOURCE};
use taguru::deadline::{Deadline, DeadlineExceeded};

use crate::groups::GroupRecord;
use crate::passages::PassageRecord;
use crate::registry::{AccessError, AppState, ContextMeta};
use crate::remote::Api;

const USAGE: &str = "\
usage: taguru export [--config FILE] [--url URL] --out DIR [CONTEXT...]

Writes each context back out as a JSONL batch stream —
{out}/{context}.jsonl, the exact format `taguru import` and POST
/import apply. A context's own schema (ADR 0009 §13) rides inside
that same stream as one taguru_schema record — no separate file —
when the context has one installed and its mode is not \"off\". No
CONTEXT arguments means every context, plus every group as
{out}/{group}.group.jsonl (one taguru_group record each; import
restores groups after every batch and schema, so the files re-apply
in any order). Naming CONTEXTs exports just those, groups omitted.
Re-importing a stream is idempotent (per-source retract-then-apply;
per-group replace); `taguru import --dry-run` validates an exported
file without touching anything.

Without --url: reads TAGURU_DATA_DIR directly, offline (the server
must not be running; the directory lock enforces it).

  --out DIR    where the streams land (created if missing)
  --config F   read KEY=VALUE environment from F (same dialect as serve)

  --url URL    export from a RUNNING server instead of the local data
               directory: GET /contexts and GET /groups are enumerated
               (keyset-paged) and each item is fetched from
               GET /contexts/{name}/export / GET /groups/{name}/export
               — the same files land in --out. Auth rides
               TAGURU_API_TOKEN (or the first name:token entry of
               TAGURU_API_TOKENS), the same variables the server
               itself reads; a URL carrying user:password@ is refused.
               Each context's own stream is internally consistent, but
               the run is NOT a point-in-time snapshot across
               contexts — two contexts fetched seconds apart may not
               reflect the same instant. An operator needing a true
               point-in-time snapshot across the whole server already
               has one: the replication bucket (TAGURU_REPLICATE_URL)
               plus `taguru restore`, which this flag does not
               replace. In CI, name the target per invocation rather
               than reading it from the environment inside the CLI:
                 taguru export --url \"$TAGURU_URL\" --out backups/
";

/// Reserved source id for the header-only batch an otherwise-empty
/// context exports — a batch stream must carry at least one header
/// for the create block to ride on.
pub(crate) const EMPTY_SOURCE: &str = "export:empty";

/// Everything one context's stream is rendered from, materialized
/// under a single registry fence so the graph half cannot shear
/// against the passage half (see [`AppState::export_context`]).
pub(crate) struct ExportSnapshot {
    pub(crate) meta: ContextMeta,
    pub(crate) associations: Vec<Association>,
    /// (alias, canonical) pairs, concept namespace.
    pub(crate) concept_aliases: Vec<(String, String)>,
    /// (alias, canonical) pairs, label namespace.
    pub(crate) label_aliases: Vec<(String, String)>,
    pub(crate) passages: Vec<(String, Arc<PassageRecord>)>,
    /// The installed schema, if any — `render` emits a `taguru_schema`
    /// record for it only when `mode != off` (ADR 0009 §13): a schema
    /// left in `off` while its types are drafted is not carried across
    /// export/restore, the same way `mode: off` is indistinguishable
    /// from "no schema" everywhere else in this module.
    pub(crate) schema: Option<crate::schema::SchemaDocument>,
}

/// What [`render`] accomplished, for the CLI report and tests.
#[cfg_attr(test, derive(Debug))]
pub(crate) struct Rendered {
    pub(crate) stream: String,
    pub(crate) batches: usize,
    pub(crate) association_lines: usize,
    pub(crate) aliases: usize,
    /// Aliases whose canonical had no live association to intern it —
    /// see the module doc. Surfaced so the loss is a number, never
    /// silent.
    pub(crate) aliases_dropped: usize,
    pub(crate) passages: usize,
}

#[derive(Serialize)]
struct HeaderLine<'a> {
    taguru_batch: u64,
    context: &'a str,
    source: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    create: Option<CreateLine<'a>>,
}

#[derive(Clone, Copy, Serialize)]
struct CreateLine<'a> {
    description: &'a str,
    #[serde(skip_serializing_if = "is_false")]
    pinned: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    dice_floor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    semantic_floor: Option<f32>,
}

fn is_false(flag: &bool) -> bool {
    !flag
}

#[derive(Serialize)]
struct PassageLine<'a> {
    passage: &'a str,
    /// Source metadata (#167), omitted when absent so pre-metadata
    /// consumers keep parsing and the stream stays a fixed point.
    /// `stored_at` is exported so a restore preserves when the source
    /// was ORIGINALLY stored — import re-stamps only when this is
    /// absent (a pre-metadata export).
    #[serde(skip_serializing_if = "Option::is_none")]
    stored_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    date: Option<u64>,
    #[serde(skip_serializing_if = "<[String]>::is_empty")]
    tags: &'a [String],
}

#[derive(Serialize)]
struct QuestionLine<'a> {
    paragraph: u32,
    question: &'a str,
}

#[derive(Serialize)]
struct SectionLine<'a> {
    paragraph: u32,
    section: &'a str,
}

#[derive(Serialize)]
struct LocatorLine<'a> {
    paragraph: u32,
    locator: &'a crate::passages::Locator,
}

#[derive(Serialize)]
struct AssociationLine<'a> {
    subject: &'a str,
    label: &'a str,
    object: &'a str,
    weight: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    paragraph: Option<u32>,
}

#[derive(Serialize)]
struct AliasLine<'a> {
    alias: &'a str,
    canonical: &'a str,
    kind: &'a str,
}

/// The `taguru_schema` record — [`crate::ingest`] parses the same
/// shape. Unlike [`GroupLine`], every field is required on the wire
/// (no `skip_serializing_if`): this mirrors [`crate::schema::
/// SchemaDocument`]'s own at-rest posture, where a missing field is a
/// parse refusal rather than a silently-defaulted one (ADR 0009 §13's
/// `deny_unknown_fields` promise applies symmetrically to fields that
/// go missing, not only ones that show up uninvited).
#[derive(Serialize)]
struct SchemaLine<'a> {
    taguru_schema: u64,
    context: &'a str,
    mode: crate::schema::SchemaMode,
    closed_labels: bool,
    types: &'a BTreeMap<String, crate::schema::TypeDef>,
    relations: &'a BTreeMap<String, crate::schema::RelationDef>,
}

/// Renders one context's schema as its import-stream record: one
/// `taguru_schema` line, newline-terminated. [`render`] calls this
/// only for a schema whose `mode != off` (see [`ExportSnapshot::
/// schema`]'s doc); `import --url`'s remote chunk packer
/// (`src/ingest.rs`) calls it directly to re-render a parsed record
/// for the wire, the same way it re-renders group records via
/// [`render_group`].
pub(crate) fn render_schema(context: &str, document: &crate::schema::SchemaDocument) -> String {
    let mut line = String::new();
    push_line(
        &mut line,
        &SchemaLine {
            taguru_schema: crate::schema::SCHEMA_VERSION,
            context,
            mode: document.mode,
            closed_labels: document.closed_labels,
            types: &document.types,
            relations: &document.relations,
        },
    );
    line
}

/// The `taguru_group` record — [`crate::ingest`] parses the same
/// shape. Empty fields are omitted and read back as empty, so the
/// round trip is exact.
#[derive(Serialize)]
struct GroupLine<'a> {
    taguru_group: u64,
    name: &'a str,
    #[serde(skip_serializing_if = "str::is_empty")]
    description: &'a str,
    #[serde(skip_serializing_if = "BTreeSet::is_empty")]
    contexts: &'a BTreeSet<String>,
    #[serde(skip_serializing_if = "BTreeSet::is_empty")]
    groups: &'a BTreeSet<String>,
}

/// Renders one group as its import-stream record: one `taguru_group`
/// line, newline-terminated — the group's complete truth, so
/// re-importing REPLACES the record (a restore, never a merge).
pub(crate) fn render_group(name: &str, record: &GroupRecord) -> String {
    let mut line = String::new();
    push_line(
        &mut line,
        &GroupLine {
            taguru_group: crate::ingest::GROUP_VERSION,
            name,
            description: &record.description,
            contexts: &record.contexts,
            groups: &record.groups,
        },
    );
    line
}

/// Filters a snapshot down to the named sources — the promote verb's
/// half of "export → filter → re-head → import" (ADR 0018). Each
/// association keeps only the named sources' live attributions, its
/// `count`/`weight` recomputed from what is kept so [`render`]'s
/// residual arithmetic finds nothing unsourced (the reserved
/// `export:unsourced` batch cannot travel with a promotion); edges
/// left with no attribution drop. Passages filter by source id.
/// Aliases pass through untouched — [`render`]'s own live-canonical
/// rule already drops the ones the kept slice no longer interns, with
/// the dropped count reported, not silent. The schema is cleared: a
/// promotion must never install the scratch's schema into the
/// destination, whose own installed schema judges the incoming
/// batches instead.
pub(crate) fn filter_to_sources(
    snapshot: ExportSnapshot,
    sources: &BTreeSet<String>,
) -> ExportSnapshot {
    let associations = snapshot
        .associations
        .into_iter()
        .filter_map(|mut association| {
            association.attributions.retain(|attribution| {
                attribution.count > 0 && sources.contains(&attribution.source)
            });
            let count: u64 = association
                .attributions
                .iter()
                .map(|attribution| attribution.count)
                .sum();
            if count == 0 {
                return None;
            }
            let sum: f64 = association
                .attributions
                .iter()
                .map(|attribution| attribution.weight)
                .sum();
            association.count = count;
            association.weight = sum / count as f64;
            Some(association)
        })
        .collect();
    ExportSnapshot {
        meta: snapshot.meta,
        associations,
        concept_aliases: snapshot.concept_aliases,
        label_aliases: snapshot.label_aliases,
        passages: snapshot
            .passages
            .into_iter()
            .filter(|(source, _)| sources.contains(source))
            .collect(),
        schema: None,
    }
}

/// The source ids a snapshot actually holds — a passage, or a LIVE
/// attribution (count-0 rows are retraction residue, not presence).
/// The promote verb's existence check (ADR 0018): a requested id
/// absent from this set refuses the request rather than silently
/// no-opping under retract-then-apply.
pub(crate) fn available_sources(snapshot: &ExportSnapshot) -> BTreeSet<&str> {
    snapshot
        .passages
        .iter()
        .map(|(source, _)| source.as_str())
        .chain(snapshot.associations.iter().flat_map(|association| {
            association
                .attributions
                .iter()
                .filter(|attribution| attribution.count > 0)
                .map(|attribution| attribution.source.as_str())
        }))
        .collect()
}

/// One source's share of the stream, accumulated before rendering.
#[derive(Default)]
struct Bucket<'a> {
    lines: Vec<AssociationLine<'a>>,
    passage: Option<&'a PassageRecord>,
}

/// Renders one context's snapshot as the import batch stream. The
/// only refusal is a real source colliding with a reserved id, or
/// `deadline` running out partway through — checked once per
/// association, passage, and alias, once per output batch, and (via
/// [`DEADLINE_CHECK_STRIDE`]) periodically inside the two loops a
/// single association's uncapped `count` can otherwise blow past all
/// of those: [`push_assertions`]'s line expansion and this function's
/// per-bucket line serialization. So a context large enough to make
/// rendering itself slow cannot run past its budget, whether that size
/// comes from many associations or one association corroborated many
/// times over. `snapshot` is already fully materialized by the time
/// this runs (see `AppState::export_context`), so a deadline that is
/// already tight when this is called cannot shorten that collection.
pub(crate) fn render(
    context: &str,
    snapshot: &ExportSnapshot,
    deadline: Deadline,
) -> Result<Rendered, String> {
    let mut buckets: BTreeMap<&str, Bucket> = BTreeMap::new();
    // Names that will be interned by an exported association line —
    // what an alias's canonical must be among to survive the trip.
    let mut live_concepts: BTreeSet<&str> = BTreeSet::new();
    let mut live_labels: BTreeSet<&str> = BTreeSet::new();

    for association in &snapshot.associations {
        if deadline.expired() {
            return Err(DeadlineExceeded.to_string());
        }
        if association.count == 0 {
            // Every assertion was retracted; the edge is dead space the
            // append-only image keeps and the export sheds.
            continue;
        }
        live_concepts.insert(&association.subject);
        live_concepts.insert(&association.object);
        live_labels.insert(&association.label);
        let mut attributed_count = 0u64;
        let mut attributed_sum = 0.0f64;
        for attribution in &association.attributions {
            if attribution.count == 0 {
                continue;
            }
            // UNSOURCED_SOURCE is what THIS renderer names sourceless
            // weight, so an attribution already carrying it is the round
            // trip coming back around (export → import stamps the batch's
            // reserved id onto the line). Fold it back into the residual
            // by NOT counting it as attributed — the residual push below
            // re-emits it under the same reserved id, making export an
            // exact fixed point instead of a one-way door that refuses
            // its own output. (A genuine, manually-created source
            // colliding with the id lands here too and is likewise
            // treated as sourceless — the id means what it says.)
            if attribution.source == UNSOURCED_SOURCE {
                continue;
            }
            // EMPTY_SOURCE only ever labels a header-only batch, which
            // carries no associations, so it can never arrive here from a
            // round trip — only a manual write could, and merging real
            // weight into the empty-context sentinel would be a lie.
            if attribution.source == EMPTY_SOURCE {
                return Err(format!(
                    "source id '{EMPTY_SOURCE}' is reserved by export — rename the source \
                     and re-export"
                ));
            }
            attributed_count += attribution.count;
            attributed_sum += attribution.weight;
            push_assertions(
                buckets.entry(&attribution.source).or_default(),
                association,
                attribution.weight,
                attribution.count,
                attribution.paragraph,
                deadline,
            )?;
        }
        // Whatever the edge's total says beyond the attributed share
        // came in without a source — including any UNSOURCED_SOURCE
        // attribution folded back in just above.
        let residual_count = association.count.saturating_sub(attributed_count);
        if residual_count > 0 {
            let residual_sum = association.weight * association.count as f64 - attributed_sum;
            push_assertions(
                buckets.entry(UNSOURCED_SOURCE).or_default(),
                association,
                residual_sum,
                residual_count,
                None,
                deadline,
            )?;
        }
    }

    for (source, record) in &snapshot.passages {
        if deadline.expired() {
            return Err(DeadlineExceeded.to_string());
        }
        if source == UNSOURCED_SOURCE || source == EMPTY_SOURCE {
            return Err(format!(
                "source id '{source}' is reserved by export — rename the source and re-export"
            ));
        }
        buckets.entry(source).or_default().passage = Some(record);
    }

    let mut aliases: Vec<AliasLine> = Vec::new();
    let mut aliases_dropped = 0usize;
    for (alias, canonical) in &snapshot.concept_aliases {
        if deadline.expired() {
            return Err(DeadlineExceeded.to_string());
        }
        if live_concepts.contains(canonical.as_str()) {
            aliases.push(AliasLine {
                alias,
                canonical,
                kind: "concept",
            });
        } else {
            aliases_dropped += 1;
        }
    }
    for (alias, canonical) in &snapshot.label_aliases {
        if deadline.expired() {
            return Err(DeadlineExceeded.to_string());
        }
        if live_labels.contains(canonical.as_str()) {
            aliases.push(AliasLine {
                alias,
                canonical,
                kind: "label",
            });
        } else {
            aliases_dropped += 1;
        }
    }

    let create = CreateLine {
        description: &snapshot.meta.description,
        pinned: snapshot.meta.pinned,
        dice_floor: snapshot.meta.dice_floor,
        semantic_floor: snapshot.meta.semantic_floor,
    };

    let mut stream = String::new();
    // The schema, if any, rides first — a preamble ahead of every
    // batch, so a restore always knows the context's constraints
    // before any association line that might assert against them
    // (ADR 0009 §13). `mode: off` is not exported — see
    // [`ExportSnapshot::schema`]'s doc.
    if let Some(document) = &snapshot.schema
        && document.mode != crate::schema::SchemaMode::Off
    {
        stream.push_str(&render_schema(context, document));
    }
    let mut association_lines = 0usize;
    let mut passages = 0usize;
    let batch_count = buckets.len().max(1);
    if buckets.is_empty() {
        // Nothing to carry the create block: emit a header-only batch
        // under the reserved empty id (its retract-first is a no-op).
        push_line(
            &mut stream,
            &HeaderLine {
                taguru_batch: 1,
                context,
                source: EMPTY_SOURCE,
                create: Some(create),
            },
        );
    } else {
        for (index, (source, bucket)) in buckets.iter().enumerate() {
            if deadline.expired() {
                return Err(DeadlineExceeded.to_string());
            }
            push_line(
                &mut stream,
                &HeaderLine {
                    taguru_batch: 1,
                    context,
                    source,
                    create: (index == 0).then_some(create),
                },
            );
            if let Some(record) = bucket.passage {
                passages += 1;
                push_line(
                    &mut stream,
                    &PassageLine {
                        passage: &record.text,
                        stored_at: record.meta.stored_at,
                        date: record.meta.date,
                        tags: &record.meta.tags,
                    },
                );
                for (paragraph, question) in &record.questions {
                    push_line(
                        &mut stream,
                        &QuestionLine {
                            paragraph: *paragraph,
                            question,
                        },
                    );
                }
                for (paragraph, section) in &record.sections {
                    push_line(
                        &mut stream,
                        &SectionLine {
                            paragraph: *paragraph,
                            section,
                        },
                    );
                }
                for (paragraph, locator) in &record.locators {
                    push_line(
                        &mut stream,
                        &LocatorLine {
                            paragraph: *paragraph,
                            locator,
                        },
                    );
                }
            }
            association_lines += bucket.lines.len();
            for (line_index, line) in bucket.lines.iter().enumerate() {
                if (line_index as u64).is_multiple_of(DEADLINE_CHECK_STRIDE) && deadline.expired() {
                    return Err(DeadlineExceeded.to_string());
                }
                push_line(&mut stream, line);
            }
            if index + 1 == batch_count {
                for alias in &aliases {
                    push_line(&mut stream, alias);
                }
            }
        }
    }

    Ok(Rendered {
        stream,
        batches: batch_count,
        association_lines,
        aliases: aliases.len(),
        aliases_dropped,
        passages,
    })
}

/// How often a `count`-bounded loop rechecks `deadline` instead of on
/// every iteration — `Instant::now()` is cheap but not free, and a
/// single edge's `count` (or a bucket's accumulated line total) is
/// otherwise the only unit of work between the association-level
/// checks around these loops' callers.
const DEADLINE_CHECK_STRIDE: u64 = 4096;

/// Re-renders one attribution as `count` assertion lines of its
/// average weight; the paragraph locator rides the first line only
/// (attribution locators are first-write-wins, so that reproduces
/// the stored one). `count` is a stored `u64` with no upper bound, so
/// this checks `deadline` on its own instead of trusting the
/// once-per-association check in its caller to bound it.
fn push_assertions<'a>(
    bucket: &mut Bucket<'a>,
    association: &'a Association,
    sum: f64,
    count: u64,
    paragraph: Option<u32>,
    deadline: Deadline,
) -> Result<(), String> {
    let weight = sum / count as f64;
    for index in 0..count {
        if index.is_multiple_of(DEADLINE_CHECK_STRIDE) && deadline.expired() {
            return Err(DeadlineExceeded.to_string());
        }
        bucket.lines.push(AssociationLine {
            subject: &association.subject,
            label: &association.label,
            object: &association.object,
            weight,
            paragraph: (index == 0).then_some(paragraph).flatten(),
        });
    }
    Ok(())
}

fn push_line(stream: &mut String, line: &impl Serialize) {
    stream.push_str(&serde_json::to_string(line).expect("export lines serialize infallibly"));
    stream.push('\n');
}

pub(crate) fn run(args: &[String]) -> i32 {
    let mut out: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut url: Option<String> = None;
    let mut names: Vec<String> = Vec::new();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{USAGE}");
                return 0;
            }
            // A flag given twice is a usage error, never a silent
            // last-wins — matching `serve`/`router`'s own `--config`
            // guard and `benchmark`'s convention throughout: a
            // scripted invocation that accidentally concatenates two
            // flag sets (e.g. two `--out`s) must not silently export
            // to whichever one happened to land last. A value that
            // itself looks like a flag (`--out --out DIR`, the first
            // `--out` swallowing the second AS ITS PATH) is rejected
            // the same way a missing value is, rather than accepted
            // literally — otherwise the duplicate-flag guard above is
            // trivial to route around, and `DIR` silently becomes a
            // context name instead of the export directory.
            "--out" => match rest.next() {
                Some(path) if out.is_none() && !path.starts_with('-') => {
                    out = Some(PathBuf::from(path));
                }
                Some(_) if out.is_none() => {
                    return crate::config::subcommand_usage_error(
                        "export",
                        "--out needs a directory path",
                    );
                }
                Some(_) => {
                    return crate::config::subcommand_usage_error("export", "--out given twice");
                }
                None => {
                    return crate::config::subcommand_usage_error(
                        "export",
                        "--out needs a directory path",
                    );
                }
            },
            "--config" => match rest.next() {
                Some(path) if config.is_none() && !path.starts_with('-') => {
                    config = Some(PathBuf::from(path));
                }
                Some(_) if config.is_none() => {
                    return crate::config::subcommand_usage_error(
                        "export",
                        "--config needs a file path",
                    );
                }
                Some(_) => {
                    return crate::config::subcommand_usage_error("export", "--config given twice");
                }
                None => {
                    return crate::config::subcommand_usage_error(
                        "export",
                        "--config needs a file path",
                    );
                }
            },
            // Trailing '/' trimmed the same way calibrate/communities
            // already do — a joined path segment must not double up.
            "--url" => match rest.next() {
                Some(value) if url.is_none() && !value.starts_with('-') => {
                    url = Some(value.trim_end_matches('/').to_string());
                }
                Some(_) if url.is_none() => {
                    return crate::config::subcommand_usage_error(
                        "export",
                        "--url needs a server URL",
                    );
                }
                Some(_) => {
                    return crate::config::subcommand_usage_error("export", "--url given twice");
                }
                None => {
                    return crate::config::subcommand_usage_error(
                        "export",
                        "--url needs a server URL",
                    );
                }
            },
            other if other.starts_with('-') => {
                return crate::config::subcommand_usage_error(
                    "export",
                    &format!("unknown flag '{other}'"),
                );
            }
            name => names.push(name.to_string()),
        }
    }
    let Some(out) = out else {
        return crate::config::subcommand_usage_error(
            "export",
            "--out DIR is required (where the streams land)",
        );
    };
    // TAGURU_CONFIG fallback (issue #248 item 2): --config wins, but a
    // deployment file baked in via the environment still applies when
    // it's absent — the same priority serve/health/calibrate/
    // communities/evaluate/restore already give it.
    let config = config.or_else(|| std::env::var("TAGURU_CONFIG").ok().map(PathBuf::from));
    // SAFETY (same contract as serve/import): applied while the
    // process is still single-threaded — export never starts a
    // runtime. Loaded before the --url dispatch below, too: a config
    // file is the usual place a deployment's TAGURU_API_TOKEN lives,
    // and Api::new reads the bearer from the environment at
    // construction — the token must already be in place by then.
    if let Some(path) = &config {
        crate::config::load_config(path);
    }

    match url {
        // ADR 0002 §5/§6: `--url` is the only way `export` goes
        // remote — no positional URL argument, no TAGURU_URL or
        // default_base_url() fallback the way `health`/`calibrate`/
        // `communities` have one. Absent, this is exactly the local
        // path that ran before this flag existed (§12.2).
        Some(base) => run_remote(&base, &out, names),
        None => run_local(&out, names),
    }
}

fn run_local(out: &std::path::Path, names: Vec<String>) -> i32 {
    // Registry warnings (WAL replay notes, load errors) must reach the
    // operator; stdout stays reserved for the report lines.
    crate::ingest::init_logging();
    // Export reads; it never embeds, whatever the environment says.
    let state = match crate::registry::BootConfig::from_env().boot(None, None, None, None, None) {
        Ok(state) => state,
        Err(error) => {
            eprintln!("taguru: export: {error}");
            return 1;
        }
    };

    let explicit = !names.is_empty();
    let names = if explicit {
        names
    } else {
        let all: Vec<String> = state
            .directory()
            .into_iter()
            .map(|entry| entry.name)
            .collect();
        if all.is_empty() {
            eprintln!("taguru: export: the data directory holds no contexts");
            return 1;
        }
        all
    };

    if let Err(error) = fs::create_dir_all(out) {
        eprintln!("taguru: export: cannot create {}: {error}", out.display());
        return 1;
    }

    let mut failures = 0usize;
    for name in &names {
        match export_one(&state, name, out) {
            Ok(report) => println!("{report}"),
            Err(message) => {
                eprintln!("taguru: export: context '{name}': {message}");
                failures += 1;
            }
        }
    }

    // Groups ride the FULL export only: a group's truth spans contexts
    // a subset may not carry, and re-importing a partial truth would
    // shrink the group (a restore replaces the record).
    let mut group_count = 0usize;
    let mut group_failures = 0usize;
    if !explicit {
        let (_, all_groups) = state.group_page(None, usize::MAX);
        group_count = all_groups.len();
        for (name, record) in &all_groups {
            match export_group_file(name, record, out) {
                Ok(report) => println!("{report}"),
                Err(message) => {
                    eprintln!("taguru: export: group '{name}': {message}");
                    group_failures += 1;
                }
            }
        }
    }

    println!(
        "{}",
        summary_line(
            names.len() - failures,
            names.len(),
            group_count,
            group_failures,
            out,
        )
    );
    if failures + group_failures > 0 { 1 } else { 0 }
}

/// The remote twin of [`run_local`]: enumerates a running server's
/// contexts and groups instead of reading `TAGURU_DATA_DIR`, and
/// fetches each item's stream from its own export endpoint instead of
/// rendering it in-process. Writes land under `out` in the identical
/// layout `run_local` uses, so the two are interchangeable inputs to
/// `taguru import` (ADR 0002 §9).
fn run_remote(base: &str, out: &std::path::Path, names: Vec<String>) -> i32 {
    // ADR 0002 §7: caught before any request leaves the process.
    if let Err(message) = crate::remote::reject_userinfo(base) {
        return crate::config::subcommand_usage_error("export", &message);
    }
    let api = Api::new(base.to_string());
    api.warn_on_version_skew("export");
    // ADR 0009 §13's explicit compatibility refusal — run
    // unconditionally (a fetch cannot know in advance which context,
    // if any, carries a schema) but only fatal when the server's
    // `schema_formats` names a version this CLI cannot read; an
    // absent key (a pre-schema server) is safe, since such a server
    // cannot have emitted a `taguru_schema` line in the first place.
    if let Some(message) = api.schema_export_refusal() {
        eprintln!("{message}");
        return 1;
    }

    // ADR 0002 §9: printed once, before the loop, so it still reaches
    // the operator on a run that fails partway through. stdout stays
    // reserved for report lines, matching run_local's own convention.
    eprintln!(
        "taguru: export: note: each context's stream is internally consistent, but contexts \
         are fetched one request at a time — this is not a point-in-time snapshot across \
         contexts; for one, replicate with TAGURU_REPLICATE_URL and restore it with \
         `taguru restore`"
    );

    let explicit = !names.is_empty();
    let names = if explicit {
        names
    } else {
        match api.list_names("contexts") {
            Ok(names) if names.is_empty() => {
                eprintln!("taguru: export: the server at {base} holds no contexts");
                return 1;
            }
            Ok(names) => names,
            Err(error) => {
                eprintln!("taguru: export: {error}");
                return 1;
            }
        }
    };

    if let Err(error) = fs::create_dir_all(out) {
        eprintln!("taguru: export: cannot create {}: {error}", out.display());
        return 1;
    }

    let mut failures = 0usize;
    for name in &names {
        match remote_export_one(&api, name, out) {
            Ok(report) => println!("{report}"),
            Err(message) => {
                eprintln!("taguru: export: context '{name}': {message}");
                failures += 1;
            }
        }
    }

    // Same rule as run_local: groups ride the FULL export only.
    let mut group_count = 0usize;
    let mut group_failures = 0usize;
    if !explicit {
        match api.list_names("groups") {
            Ok(group_names) => {
                group_count = group_names.len();
                for name in &group_names {
                    match remote_export_group(&api, name, out) {
                        Ok(report) => println!("{report}"),
                        Err(message) => {
                            eprintln!("taguru: export: group '{name}': {message}");
                            group_failures += 1;
                        }
                    }
                }
            }
            Err(error) => {
                // A full export that cannot even learn its group list
                // is incomplete, whatever the context loop above did —
                // reported the same way any other failure is.
                eprintln!("taguru: export: groups: {error}");
                group_failures += 1;
            }
        }
    }

    println!(
        "{}",
        summary_line(
            names.len() - failures,
            names.len(),
            group_count,
            group_failures,
            out,
        )
    );
    if failures + group_failures > 0 { 1 } else { 0 }
}

/// One context's stream, fetched whole from `GET
/// /contexts/{name}/export` and written exactly like the local path
/// writes its own render. A request that outlives `Api`'s 35s budget,
/// or a context deleted between enumeration and this fetch, surfaces
/// as an `Err` here and is counted as a per-item failure by the
/// caller — never a reason to abort the rest of the run.
fn remote_export_one(api: &Api, name: &str, out: &std::path::Path) -> Result<String, String> {
    let stream = api.get_raw(&["contexts", name, "export"])?;
    // Validated before it touches disk: write_atomic's whole point is
    // that a crash mid-write must never shred a previously-good
    // backup, which a malformed response would defeat if it landed
    // first and only then failed to parse (see remote_export_group,
    // whose comment this one mirrors). `stream_counts` below is only a
    // best-effort tally for the report line, not a validator — it
    // never errors on malformed input, so real validation has to run
    // through the same parser `taguru import` trusts.
    crate::ingest::parse_stream(stream.as_bytes())
        .map_err(|error| format!("context '{name}': not a taguru export stream: {error}"))?;
    let path = out.join(format!("{}.jsonl", crate::registry::file_stem(name)));
    crate::storage::write_atomic(&path, stream.as_bytes())
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    let (batches, lines) = stream_counts(&stream);
    Ok(format!(
        "{}: context '{name}' → {batches} batch(es), {lines} line(s)",
        path.display()
    ))
}

/// One group's record, fetched from `GET /groups/{name}/export` — the
/// remote twin of [`export_group_file`].
fn remote_export_group(api: &Api, name: &str, out: &std::path::Path) -> Result<String, String> {
    let stream = api.get_raw(&["groups", name, "export"])?;
    // Validated before it touches disk: write_atomic's whole point is
    // that a crash mid-write must never shred a previously-good
    // backup, which a malformed response would defeat if it landed
    // first and only then failed to parse.
    let record: Value = serde_json::from_str(stream.trim_end())
        .map_err(|error| format!("group '{name}': not a taguru group record: {error}"))?;
    let path = out.join(format!("{}.group.jsonl", crate::registry::file_stem(name)));
    crate::storage::write_atomic(&path, stream.as_bytes())
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    let contexts = record["contexts"].as_array().map_or(0, Vec::len);
    let groups = record["groups"].as_array().map_or(0, Vec::len);
    Ok(format!(
        "{}: group '{name}' → {contexts} member context(s), {groups} child group(s)",
        path.display()
    ))
}

/// (batches, lines) counted back out of a fetched context stream — the
/// counts a remote fetch can honestly claim, since (unlike the local
/// render) there is no in-process [`Rendered`] to report from. A
/// batch is any line whose object carries `taguru_batch`; the key
/// check (rather than a text prefix match) survives field reordering.
fn stream_counts(stream: &str) -> (usize, usize) {
    let mut batches = 0usize;
    let mut lines = 0usize;
    for line in stream.lines() {
        if line.trim().is_empty() {
            continue;
        }
        lines += 1;
        if let Ok(Value::Object(object)) = serde_json::from_str::<Value>(line)
            && object.contains_key("taguru_batch")
        {
            batches += 1;
        }
    }
    (batches, lines)
}

/// The final report line, identical whether the streams came from the
/// local data directory ([`run_local`]) or a running server
/// ([`run_remote`]) — one summary shape for both.
fn summary_line(
    context_ok: usize,
    context_total: usize,
    group_count: usize,
    group_failures: usize,
    out: &std::path::Path,
) -> String {
    format!(
        "export: {context_ok} of {context_total} context(s){} written to {}",
        match (group_count, group_failures) {
            (0, 0) => String::new(),
            // `group_count` stays 0 when `list_names("groups")` itself
            // errored — the run still failed (the caller exits
            // non-zero on any `group_failures`), so the summary must
            // say so rather than silently dropping groups as if this
            // were a subset export that never touched them.
            (0, _) => " and groups: enumeration failed".to_string(),
            (count, failures) => format!(" and {} of {count} group(s)", count - failures),
        },
        out.display()
    )
}

/// Writes one group's `taguru_group` record beside the context
/// streams, `{out}/{stem}.group.jsonl` — a name a context stream can
/// never claim (stems percent-encode `.`), exactly the collision
/// argument the data directory's own `.group` extension makes.
fn export_group_file(
    name: &str,
    record: &GroupRecord,
    out: &std::path::Path,
) -> Result<String, String> {
    let path = out.join(format!("{}.group.jsonl", crate::registry::file_stem(name)));
    crate::storage::write_atomic(&path, render_group(name, record).as_bytes())
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    Ok(format!(
        "{}: group '{name}' → {} member context(s), {} child group(s)",
        path.display(),
        record.contexts.len(),
        record.groups.len()
    ))
}

fn export_one(state: &AppState, name: &str, out: &std::path::Path) -> Result<String, String> {
    let snapshot = state
        .export_context(name, Deadline::unbounded())
        .map_err(|failure| match failure {
            AccessError::NotFound => "no such context".to_string(),
            AccessError::Load(error) => error,
            AccessError::Unpersisted(error) => error,
            // The CLI runs with Deadline::unbounded(), which never
            // expires — unreachable in practice, kept for
            // exhaustiveness.
            AccessError::DeadlineExceeded => "deadline exceeded".to_string(),
            // Same unreachability: the offline CLI boots with no quota
            // declaration (and an export is a read besides).
            AccessError::QuotaExceeded(error) => error,
        })?;
    let rendered = render(name, &snapshot, Deadline::unbounded())?;
    let path = out.join(format!("{}.jsonl", crate::registry::file_stem(name)));
    // Stage + fsync + rename, never a truncating write in place: a
    // backup that "wrote" but never reached the platter is worse than a
    // refusal, and a crash while REFRESHING an existing backup must not
    // shred the previous good copy — the exact hazard the same helper
    // guards for the server's own images.
    crate::storage::write_atomic(&path, rendered.stream.as_bytes())
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    Ok(format!(
        "{}: context '{name}' → {} batch(es), {} association line(s), {} alias(es){}{}",
        path.display(),
        rendered.batches,
        rendered.association_lines,
        rendered.aliases,
        match rendered.passages {
            0 => String::new(),
            stored => format!(", {stored} passage(s)"),
        },
        match rendered.aliases_dropped {
            0 => String::new(),
            dropped => format!(", {dropped} alias(es) dropped (canonical has no live association)"),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use taguru::context::Attribution;

    use crate::ingest;
    use crate::passages::PassageSubmission;
    use crate::registry::AssocOp;

    #[test]
    fn every_usage_variable_is_a_known_key() {
        // This command's own USAGE is invisible to cli.rs's
        // consistency tests: a variable documented here but missing
        // from KNOWN_KEYS would make --config warn "typo?" on a
        // perfectly valid setting.
        crate::config::assert_usage_vars_are_known_keys(USAGE);
    }

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("taguru-export-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn op(
        subject: &str,
        label: &str,
        object: &str,
        weight: f64,
        source: Option<&str>,
        paragraph: Option<u32>,
    ) -> AssocOp {
        AssocOp {
            subject: subject.to_string(),
            label: label.to_string(),
            object: object.to_string(),
            weight,
            source: source.map(String::from),
            paragraph,
        }
    }

    /// One edge reduced to what must survive the round trip: the real
    /// (non-reserved) attributions, and the sourceless share — which A
    /// holds as arithmetic residual and B holds as an attribution to
    /// the reserved id. Weights are exact here because the fixture
    /// uses exactly representable values.
    #[derive(Debug, PartialEq, PartialOrd)]
    struct Reduced {
        subject: String,
        label: String,
        object: String,
        count: u64,
        real: Vec<(String, f64, u64, Option<u32>)>,
        unsourced_sum: f64,
        unsourced_count: u64,
    }

    fn reduce(associations: &[Association]) -> Vec<Reduced> {
        let mut reduced: Vec<Reduced> = associations
            .iter()
            .filter(|association| association.count > 0)
            .map(|association| {
                let mut real: Vec<(String, f64, u64, Option<u32>)> = Vec::new();
                let mut unsourced_sum = 0.0;
                let mut unsourced_count = 0;
                let mut attributed_sum = 0.0;
                let mut attributed_count = 0;
                for attribution in &association.attributions {
                    attributed_sum += attribution.weight;
                    attributed_count += attribution.count;
                    if attribution.source == UNSOURCED_SOURCE {
                        unsourced_sum += attribution.weight;
                        unsourced_count += attribution.count;
                    } else {
                        real.push((
                            attribution.source.clone(),
                            attribution.weight,
                            attribution.count,
                            attribution.paragraph,
                        ));
                    }
                }
                // Whatever the totals say beyond the attributions is
                // A-side sourceless weight; fold it in with B's
                // reserved attribution so both sides reduce alike.
                unsourced_sum += association.weight * association.count as f64 - attributed_sum;
                unsourced_count += association.count - attributed_count;
                real.sort_by(|a, b| a.partial_cmp(b).unwrap());
                Reduced {
                    subject: association.subject.clone(),
                    label: association.label.clone(),
                    object: association.object.clone(),
                    count: association.count,
                    real,
                    unsourced_sum,
                    unsourced_count,
                }
            })
            .collect();
        reduced.sort_by(|a, b| a.partial_cmp(b).unwrap());
        reduced
    }

    #[test]
    fn a_context_round_trips_through_export_and_import() {
        let state_a =
            crate::registry::AppState::boot(scratch_dir("roundtrip-a"), usize::MAX, None).unwrap();
        state_a
            .create(
                "sake",
                ContextMeta {
                    description: "酒蔵の知識".to_string(),
                    pinned: true,
                    dice_floor: Some(0.25),
                    semantic_floor: Some(0.5),
                },
            )
            .map_err(|_| "create")
            .unwrap();
        state_a
            .add_associations(
                "sake",
                vec![
                    // Corroborated twice by a.md — the locator rides the
                    // first assertion — and once more by b.md.
                    op("青嶺酒造", "代表銘柄", "青嶺", 1.0, Some("a.md"), Some(0)),
                    op("青嶺酒造", "代表銘柄", "青嶺", 1.0, Some("a.md"), None),
                    op("青嶺酒造", "代表銘柄", "青嶺", 2.5, Some("b.md"), Some(1)),
                    // Sourceless weight — the reserved-batch case.
                    op("青嶺酒造", "杜氏", "高瀬", 1.5, None, None),
                    // Will be fully retracted: must shed on export.
                    op("青嶺酒造", "廃止銘柄", "旧銘", 1.0, Some("gone.md"), None),
                ],
                Deadline::unbounded(),
            )
            .unwrap()
            .unwrap();
        state_a
            .store_passages(
                "sake",
                BTreeMap::from([
                    (
                        "a.md".to_string(),
                        PassageSubmission {
                            text: "青嶺酒造は1907年創業。\n\n代表銘柄は青嶺。".to_string(),
                            questions: vec![(0, "いつ創業した?".to_string())],
                            sections: vec![(0, "沿革".to_string())],
                            // A locator (ADR 0007 §7) rides beside the
                            // section: the round trip below must
                            // preserve both independently.
                            locators: vec![(
                                0,
                                crate::passages::Locator {
                                    kind: "page".to_string(),
                                    value: "12".to_string(),
                                },
                            )],
                            // Populated metadata (#167): the round trip
                            // below must preserve it — stored_at
                            // included, or a backup would silently
                            // re-date every source to its restore time.
                            meta: crate::passages::SourceMeta {
                                stored_at: Some(1_700_000_000),
                                date: Some(1_181_600_000),
                                tags: vec!["蔵".to_string(), "酒".to_string()],
                            },
                        },
                    ),
                    (
                        "b.md".to_string(),
                        PassageSubmission {
                            text: "杜氏の紹介。\n\n代表銘柄の解説。".to_string(),
                            questions: Vec::new(),
                            sections: Vec::new(),
                            locators: Vec::new(),
                            meta: crate::passages::SourceMeta::default(),
                        },
                    ),
                ]),
            )
            .unwrap()
            .unwrap();
        state_a
            .add_aliases(
                "sake",
                &BTreeMap::from([
                    ("Aomine".to_string(), "青嶺酒造".to_string()),
                    // Canonical loses its every edge below: must drop.
                    ("Kyumei".to_string(), "旧銘".to_string()),
                ]),
                &BTreeMap::from([("とじ".to_string(), "杜氏".to_string())]),
            )
            .unwrap()
            .unwrap();
        state_a.retract_source("sake", "gone.md").unwrap();

        let snapshot_a = state_a
            .export_context("sake", Deadline::unbounded())
            .unwrap();
        let rendered = render("sake", &snapshot_a, Deadline::unbounded()).unwrap();
        assert_eq!(rendered.aliases_dropped, 1, "the edgeless canonical");

        // Restore into a fresh directory, twice — the second pass
        // proves the stream is idempotent (retract-then-apply).
        let state_b =
            crate::registry::AppState::boot(scratch_dir("roundtrip-b"), usize::MAX, None).unwrap();
        for _ in 0..2 {
            let stream = ingest::parse_stream(rendered.stream.as_bytes()).unwrap();
            assert_eq!(
                stream.batches.len(),
                3,
                "a.md, b.md, and the reserved batch"
            );
            for batch in &stream.batches {
                ingest::apply_batch(&state_b, batch, Deadline::unbounded())
                    .map_err(|r| r.text())
                    .unwrap();
            }
        }
        let snapshot_b = state_b
            .export_context("sake", Deadline::unbounded())
            .unwrap();

        // The restored context has a REAL attribution to the reserved
        // "export:unsourced" id now (import stamped it from the batch
        // header). Re-rendering must NOT refuse — it folds back into a
        // sourceless batch — so the backup stream is a true fixed point.
        let rendered_b =
            render("sake", &snapshot_b, Deadline::unbounded()).expect("re-export must not refuse");
        let rendered_c = {
            let state_c =
                crate::registry::AppState::boot(scratch_dir("roundtrip-c"), usize::MAX, None)
                    .unwrap();
            for batch in ingest::parse_stream(rendered_b.stream.as_bytes())
                .unwrap()
                .batches
            {
                ingest::apply_batch(&state_c, &batch, Deadline::unbounded())
                    .map_err(|r| r.text())
                    .unwrap();
            }
            render(
                "sake",
                &state_c
                    .export_context("sake", Deadline::unbounded())
                    .unwrap(),
                Deadline::unbounded(),
            )
            .unwrap()
        };
        assert_eq!(
            rendered_b.stream, rendered_c.stream,
            "export must be a fixed point across a second round trip"
        );

        assert_eq!(snapshot_b.meta.description, "酒蔵の知識");
        assert!(snapshot_b.meta.pinned);
        assert_eq!(snapshot_b.meta.dice_floor, Some(0.25));
        assert_eq!(snapshot_b.meta.semantic_floor, Some(0.5));
        assert_eq!(
            reduce(&snapshot_a.associations),
            reduce(&snapshot_b.associations),
            "live graph content must survive the trip"
        );
        assert!(
            snapshot_b.associations.len() < snapshot_a.associations.len(),
            "the fully retracted edge must shed (export doubles as compaction)"
        );
        let aliases_b: Vec<_> = snapshot_b.concept_aliases.clone();
        assert_eq!(
            aliases_b,
            vec![("Aomine".to_string(), "青嶺酒造".to_string())]
        );
        assert_eq!(
            snapshot_b.label_aliases,
            vec![("とじ".to_string(), "杜氏".to_string())]
        );
        type StoredPassage = (
            String,
            Vec<(u32, String)>,
            Vec<(u32, String)>,
            Vec<(u32, crate::passages::Locator)>,
        );
        let passages_b: BTreeMap<String, StoredPassage> = snapshot_b
            .passages
            .iter()
            .map(|(source, record)| {
                (
                    source.clone(),
                    (
                        record.text.to_string(),
                        record.questions.clone(),
                        record.sections.clone(),
                        record.locators.clone(),
                    ),
                )
            })
            .collect();
        assert_eq!(passages_b.len(), 2);
        assert_eq!(
            passages_b["a.md"],
            (
                "青嶺酒造は1907年創業。\n\n代表銘柄は青嶺。".to_string(),
                vec![(0, "いつ創業した?".to_string())],
                vec![(0, "沿革".to_string())],
                vec![(
                    0,
                    crate::passages::Locator {
                        kind: "page".to_string(),
                        value: "12".to_string(),
                    },
                )],
            )
        );
        assert_eq!(passages_b["b.md"].1, Vec::new());
        // The metadata trio (#167) survives the trip intact —
        // stored_at especially: import must preserve the ORIGINAL
        // stamp, not re-date the source to its restore time.
        let meta = |snapshot: &ExportSnapshot, source: &str| {
            snapshot
                .passages
                .iter()
                .find(|(name, _)| name == source)
                .map(|(_, record)| record.meta.clone())
                .unwrap()
        };
        assert_eq!(
            meta(&snapshot_b, "a.md"),
            crate::passages::SourceMeta {
                stored_at: Some(1_700_000_000),
                date: Some(1_181_600_000),
                tags: vec!["蔵".to_string(), "酒".to_string()],
            }
        );
        assert_eq!(
            meta(&snapshot_b, "b.md").stored_at,
            meta(&snapshot_a, "b.md").stored_at,
            "a store-stamped source keeps its stamp through the trip"
        );
    }

    fn snapshot(associations: Vec<Association>) -> ExportSnapshot {
        ExportSnapshot {
            meta: ContextMeta {
                description: "テスト".to_string(),
                pinned: false,
                dice_floor: None,
                semantic_floor: None,
            },
            associations,
            concept_aliases: Vec::new(),
            label_aliases: Vec::new(),
            passages: Vec::new(),
            schema: None,
        }
    }

    fn association(count: u64, attributions: Vec<Attribution>) -> Association {
        // weight is the average; reconstruct it from the parts the way
        // the library exposes it.
        let sum: f64 = attributions.iter().map(|a| a.weight).sum();
        Association {
            subject: "青嶺酒造".to_string(),
            label: "杜氏".to_string(),
            object: "高瀬".to_string(),
            weight: if count == 0 { 0.0 } else { sum / count as f64 },
            count,
            attributions,
        }
    }

    #[test]
    fn corroboration_re_renders_as_count_lines() {
        let rendered = render(
            "sake",
            &snapshot(vec![association(
                3,
                vec![Attribution {
                    source: "a.md".to_string(),
                    weight: 6.0,
                    count: 3,
                    paragraph: Some(2),
                }],
            )]),
            Deadline::unbounded(),
        )
        .unwrap();
        assert_eq!(rendered.batches, 1);
        assert_eq!(rendered.association_lines, 3);
        let lines: Vec<&str> = rendered.stream.lines().collect();
        assert_eq!(lines.len(), 4, "header + three assertions");
        // The locator rides the first assertion only.
        assert!(lines[1].contains("\"paragraph\":2"));
        assert!(!lines[2].contains("paragraph"));
        assert!(lines[1].contains("\"weight\":2.0"));
    }

    #[test]
    fn unsourced_weight_lands_in_the_reserved_batch() {
        // count 3, but only 2 assertions attributed: one came sourceless.
        let mut edge = association(
            3,
            vec![Attribution {
                source: "a.md".to_string(),
                weight: 2.0,
                count: 2,
                paragraph: None,
            }],
        );
        edge.weight = (2.0 + 4.0) / 3.0; // attributed 2.0 + unsourced 4.0
        let rendered = render("sake", &snapshot(vec![edge]), Deadline::unbounded()).unwrap();
        assert_eq!(rendered.batches, 2);
        let unsourced: Vec<&str> = rendered
            .stream
            .lines()
            .filter(|line| line.contains(UNSOURCED_SOURCE) || line.contains("\"weight\":4.0"))
            .collect();
        assert_eq!(
            unsourced.len(),
            2,
            "one reserved header, one residual assertion: {}",
            rendered.stream
        );
    }

    /// The promote verb's existence check (ADR 0018): a passage or a
    /// LIVE attribution counts as presence; a count-0 attribution is
    /// retraction residue and must not — an id present only that way
    /// would promote as an empty no-op instead of refusing.
    #[test]
    fn available_sources_sees_passages_and_live_attributions_only() {
        let edge = association(
            1,
            vec![
                Attribution {
                    source: "live.md".to_string(),
                    weight: 1.0,
                    count: 1,
                    paragraph: None,
                },
                Attribution {
                    source: "dead.md".to_string(),
                    weight: 0.0,
                    count: 0,
                    paragraph: None,
                },
            ],
        );
        let mut snapshot = snapshot(vec![edge]);
        snapshot.passages = vec![("note.md".to_string(), PassageRecord::for_tests("原文"))];

        let available = available_sources(&snapshot);
        assert!(available.contains("note.md"), "a passage is presence");
        assert!(
            available.contains("live.md"),
            "a live attribution is presence"
        );
        assert!(
            !available.contains("dead.md"),
            "retraction residue is not presence: {available:?}"
        );
    }

    /// The promote verb's filter (ADR 0018): each edge keeps only the
    /// named sources' live shares with count/weight recomputed from
    /// what is kept, edges left with nothing drop, and — because
    /// attributed count now equals total by construction — the
    /// rendered stream carries no `export:unsourced` residual batch.
    #[test]
    fn filtering_to_sources_keeps_only_their_shares_and_no_residual() {
        let keep: BTreeSet<String> = ["a.md".to_string()].into();
        let shared = association(
            4,
            vec![
                // Two assertions summing to 3.0: the recomputed weight
                // must be the mean (1.5), never the product's shape.
                Attribution {
                    source: "a.md".to_string(),
                    weight: 3.0,
                    count: 2,
                    paragraph: Some(4),
                },
                Attribution {
                    source: "b.md".to_string(),
                    weight: 1.0,
                    count: 1,
                    paragraph: None,
                },
                // The retracted share of the kept source stays dead.
                Attribution {
                    source: "a.md".to_string(),
                    weight: 0.0,
                    count: 0,
                    paragraph: None,
                },
            ],
        );
        let mut theirs = association(
            1,
            vec![Attribution {
                source: "b.md".to_string(),
                weight: 1.0,
                count: 1,
                paragraph: None,
            }],
        );
        theirs.object = "別件".to_string();
        let mut snapshot = snapshot(vec![shared, theirs]);
        snapshot.passages = vec![
            ("a.md".to_string(), PassageRecord::for_tests("甲")),
            ("b.md".to_string(), PassageRecord::for_tests("乙")),
        ];
        snapshot.schema = Some(warn_schema_document());

        let filtered = filter_to_sources(snapshot, &keep);
        assert_eq!(filtered.associations.len(), 1, "b.md's own edge drops");
        let edge = &filtered.associations[0];
        assert_eq!(edge.count, 2, "only a.md's live share remains");
        assert_eq!(edge.weight, 1.5, "sum 3.0 over count 2 — the mean");
        assert_eq!(edge.attributions.len(), 1);
        assert_eq!(edge.attributions[0].source, "a.md");
        assert_eq!(
            filtered.passages.len(),
            1,
            "passages filter by source id too"
        );
        assert!(
            filtered.schema.is_none(),
            "a promotion never carries the scratch's schema"
        );

        let rendered = render("perm", &filtered, Deadline::unbounded()).unwrap();
        assert_eq!(rendered.batches, 1, "{}", rendered.stream);
        assert!(
            !rendered.stream.contains(UNSOURCED_SOURCE),
            "no residual may travel with a promotion: {}",
            rendered.stream
        );
        assert!(
            rendered.stream.contains("\"paragraph\":4"),
            "the kept share's locator survives: {}",
            rendered.stream
        );
    }

    #[test]
    fn fully_retracted_edges_render_as_nothing() {
        let rendered = render(
            "sake",
            &snapshot(vec![association(0, Vec::new())]),
            Deadline::unbounded(),
        )
        .unwrap();
        assert_eq!(rendered.association_lines, 0);
        assert_eq!(rendered.batches, 1, "the empty batch still carries create");
        assert!(rendered.stream.contains(EMPTY_SOURCE));
        assert!(rendered.stream.contains("\"description\":\"テスト\""));
    }

    /// An attribution already carrying UNSOURCED_SOURCE is the round
    /// trip coming back: it must fold into the sourceless residual (one
    /// reserved batch), NOT refuse — otherwise export cannot re-export
    /// its own imported output.
    #[test]
    fn an_unsourced_attribution_folds_back_instead_of_refusing() {
        let edge = association(
            2,
            vec![Attribution {
                source: UNSOURCED_SOURCE.to_string(),
                weight: 3.0,
                count: 2,
                paragraph: None,
            }],
        );
        let rendered = render("sake", &snapshot(vec![edge]), Deadline::unbounded()).unwrap();
        assert_eq!(rendered.batches, 1, "just the reserved batch");
        assert_eq!(rendered.association_lines, 2);
        assert!(
            rendered.stream.contains(UNSOURCED_SOURCE),
            "{}",
            rendered.stream
        );
        // Re-rendering the same reduced shape is a fixed point.
        assert!(!rendered.stream.contains("reserved"));
    }

    /// EMPTY_SOURCE cannot arise from a round trip (empty batches carry
    /// no associations), so a real attribution under it is manual
    /// misuse and still refuses.
    #[test]
    fn an_empty_source_attribution_still_refuses() {
        let edge = association(
            1,
            vec![Attribution {
                source: EMPTY_SOURCE.to_string(),
                weight: 1.0,
                count: 1,
                paragraph: None,
            }],
        );
        let refusal = render("sake", &snapshot(vec![edge]), Deadline::unbounded()).unwrap_err();
        assert!(refusal.contains("reserved"), "{refusal}");
    }

    #[test]
    fn aliases_ride_the_last_batch_and_edgeless_canonicals_drop() {
        let mut snapshot = snapshot(vec![association(
            1,
            vec![Attribution {
                source: "a.md".to_string(),
                weight: 1.0,
                count: 1,
                paragraph: None,
            }],
        )]);
        snapshot.concept_aliases = vec![
            ("Aomine".to_string(), "青嶺酒造".to_string()),
            ("orphan".to_string(), "退役した概念".to_string()),
        ];
        snapshot.label_aliases = vec![("toji".to_string(), "杜氏".to_string())];
        let rendered = render("sake", &snapshot, Deadline::unbounded()).unwrap();
        assert_eq!(rendered.aliases, 2);
        assert_eq!(rendered.aliases_dropped, 1);
        let last_lines: Vec<&str> = rendered.stream.lines().rev().take(2).collect();
        assert!(last_lines.iter().all(|line| line.contains("\"alias\"")));
        assert!(!rendered.stream.contains("orphan"));
    }

    /// The label-alias half of the drop rule: a label alias whose
    /// canonical LABEL carries no live association drops exactly like
    /// an edgeless concept canonical (the test above only exercises
    /// the concept loop). And the unpinned create line omits `pinned`
    /// entirely — `skip_serializing_if` — so old consumers keep
    /// reading a create object without the key.
    #[test]
    fn a_label_alias_to_a_dead_label_drops_and_unpinned_stays_unserialized() {
        let mut snapshot = snapshot(vec![association(
            1,
            vec![Attribution {
                source: "a.md".to_string(),
                weight: 1.0,
                count: 1,
                paragraph: None,
            }],
        )]);
        snapshot.label_aliases = vec![
            ("toji".to_string(), "杜氏".to_string()),
            ("orphan-label".to_string(), "廃止銘柄".to_string()),
        ];
        let rendered = render("sake", &snapshot, Deadline::unbounded()).unwrap();
        assert_eq!(rendered.aliases, 1, "the live 杜氏 alias");
        assert_eq!(rendered.aliases_dropped, 1, "the dead-label alias");
        assert!(!rendered.stream.contains("orphan-label"));
        assert!(
            !rendered.stream.contains("\"pinned\""),
            "an unpinned create line must omit the field: {}",
            rendered.stream
        );
    }

    /// One group renders as one `taguru_group` line, empties omitted,
    /// and the parser reads it back exactly — the group half of the
    /// stream's fixed point.
    #[test]
    fn a_group_renders_as_one_record_and_round_trips() {
        let record = GroupRecord {
            description: "蔵まとめ".to_string(),
            contexts: ["sake", "bunko"].iter().map(|c| c.to_string()).collect(),
            groups: ["kid"].iter().map(|g| g.to_string()).collect(),
        };
        let line = render_group("kura", &record);
        assert_eq!(
            line,
            "{\"taguru_group\":1,\"name\":\"kura\",\"description\":\"蔵まとめ\",\
             \"contexts\":[\"bunko\",\"sake\"],\"groups\":[\"kid\"]}\n"
        );
        let stream = ingest::parse_stream(line.as_bytes()).unwrap();
        assert_eq!(stream.groups, vec![("kura".to_string(), record)]);

        let bare = render_group("kid", &GroupRecord::default());
        assert_eq!(bare, "{\"taguru_group\":1,\"name\":\"kid\"}\n");
        let stream = ingest::parse_stream(bare.as_bytes()).unwrap();
        assert_eq!(stream.groups[0].1, GroupRecord::default());
    }

    fn warn_schema_document() -> crate::schema::SchemaDocument {
        crate::schema::SchemaDocument {
            schema: crate::schema::SCHEMA_VERSION,
            mode: crate::schema::SchemaMode::Warn,
            closed_labels: false,
            types: [("醸造所".to_string(), crate::schema::TypeDef::default())]
                .into_iter()
                .collect(),
            relations: [(
                "杜氏".to_string(),
                crate::schema::RelationDef {
                    domain: ["醸造所".to_string()].into_iter().collect(),
                    range: BTreeSet::new(),
                },
            )]
            .into_iter()
            .collect(),
        }
    }

    /// `render_schema`'s exact bytes, and the round trip back through
    /// `parse_stream` — [`a_group_renders_as_one_record_and_round_trips`]'s
    /// `taguru_schema` twin (ADR 0009 §13). Unlike a group record, no
    /// field is ever omitted: every one of `SchemaDocument`'s fields is
    /// required on the wire, matching its own at-rest posture.
    #[test]
    fn a_schema_renders_as_one_record_and_round_trips() {
        let document = warn_schema_document();
        let line = render_schema("sake", &document);
        assert_eq!(
            line,
            "{\"taguru_schema\":1,\"context\":\"sake\",\"mode\":\"warn\",\
             \"closed_labels\":false,\"types\":{\"醸造所\":{\"is_a\":[]}},\
             \"relations\":{\"杜氏\":{\"domain\":[\"醸造所\"],\"range\":[]}}}\n"
        );
        let stream = ingest::parse_stream(line.as_bytes()).unwrap();
        assert_eq!(stream.schemas.len(), 1);
        assert_eq!(stream.schemas[0].0, "sake");
        assert_eq!(stream.schemas[0].1.document(), &document);
    }

    /// `render`'s own gate (ADR 0009 §13): a `taguru_schema` record
    /// rides first, before any batch header, but ONLY when the
    /// context has a schema AND its mode is not `off` — the three
    /// cases side by side so a regression in the gate shows up as an
    /// exact diff, not a missing assertion.
    #[test]
    fn render_emits_a_schema_record_first_only_when_mode_is_not_off() {
        let edge = association(
            1,
            vec![Attribution {
                source: "a.md".to_string(),
                weight: 1.0,
                count: 1,
                paragraph: None,
            }],
        );

        let mut no_schema = snapshot(vec![edge.clone()]);
        no_schema.schema = None;
        let rendered = render("sake", &no_schema, Deadline::unbounded()).unwrap();
        assert!(
            !rendered.stream.contains("taguru_schema"),
            "{}",
            rendered.stream
        );

        let mut off_schema = snapshot(vec![edge.clone()]);
        off_schema.schema = Some(crate::schema::SchemaDocument {
            mode: crate::schema::SchemaMode::Off,
            ..warn_schema_document()
        });
        let rendered = render("sake", &off_schema, Deadline::unbounded()).unwrap();
        assert!(
            !rendered.stream.contains("taguru_schema"),
            "mode: off must not export — {}",
            rendered.stream
        );

        let mut warn_schema = snapshot(vec![edge]);
        warn_schema.schema = Some(warn_schema_document());
        let rendered = render("sake", &warn_schema, Deadline::unbounded()).unwrap();
        let first_line = rendered.stream.lines().next().unwrap();
        assert!(
            first_line.starts_with("{\"taguru_schema\":1,\"context\":\"sake\""),
            "the schema record must ride first — {}",
            rendered.stream
        );
    }

    /// `stream_counts` is what a remote fetch has to report from
    /// instead of an in-process [`Rendered`]: two headers plus their
    /// association/alias lines count as 2 batches, 5 lines total.
    #[test]
    fn stream_counts_reads_batches_and_lines_back_out_of_a_stream() {
        let stream = concat!(
            "{\"taguru_batch\":1,\"context\":\"sake\",\"source\":\"a.md\"}\n",
            "{\"subject\":\"s\",\"label\":\"l\",\"object\":\"o\",\"weight\":1.0}\n",
            "{\"taguru_batch\":1,\"context\":\"sake\",\"source\":\"b.md\"}\n",
            "{\"subject\":\"s2\",\"label\":\"l2\",\"object\":\"o2\",\"weight\":2.0}\n",
            "{\"alias\":\"a\",\"canonical\":\"b\",\"kind\":\"concept\"}\n",
        );
        assert_eq!(stream_counts(stream), (2, 5));
    }

    #[test]
    fn stream_counts_ignores_blank_lines_and_non_object_lines() {
        let stream = "\n{\"taguru_batch\":1}\n\n\"just a string\"\n";
        assert_eq!(stream_counts(stream), (1, 2));
    }
}
