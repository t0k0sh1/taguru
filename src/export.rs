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

use taguru::context::Association;

use crate::passages::PassageRecord;
use crate::registry::{AccessError, AppState, ContextMeta};

const USAGE: &str = "\
usage: taguru export [--config FILE] --out DIR [CONTEXT...]

Writes each context back out of TAGURU_DATA_DIR as a JSONL batch
stream — {out}/{context}.jsonl, the exact format `taguru import` and
POST /import apply — offline (the server must not be running; the
directory lock enforces it). No CONTEXT arguments means every context
in the directory. A running server serves the same stream at
GET /contexts/{name}/export. Re-importing a stream is idempotent
(per-source retract-then-apply); `taguru import --dry-run` validates
an exported file without touching anything.

  --out DIR    where the streams land (created if missing)
  --config F   read KEY=VALUE environment from F (same dialect as serve)
";

/// Reserved source id for weight that was asserted without a source.
/// A real source with this exact id has its attributions folded into
/// the same sourceless residual rather than merged in as attributed
/// weight (see the fixed-point note in [`render`]) — but if it also
/// has a passage attached, export refuses the context outright rather
/// than picking a winner silently.
pub(crate) const UNSOURCED_SOURCE: &str = "export:unsourced";

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

/// One source's share of the stream, accumulated before rendering.
#[derive(Default)]
struct Bucket<'a> {
    lines: Vec<AssociationLine<'a>>,
    passage: Option<&'a PassageRecord>,
}

/// Renders one context's snapshot as the import batch stream. The
/// only refusal is a real source colliding with a reserved id —
/// everything else a context can hold has a rendering.
pub(crate) fn render(context: &str, snapshot: &ExportSnapshot) -> Result<Rendered, String> {
    let mut buckets: BTreeMap<&str, Bucket> = BTreeMap::new();
    // Names that will be interned by an exported association line —
    // what an alias's canonical must be among to survive the trip.
    let mut live_concepts: BTreeSet<&str> = BTreeSet::new();
    let mut live_labels: BTreeSet<&str> = BTreeSet::new();

    for association in &snapshot.associations {
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
            );
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
            );
        }
    }

    for (source, record) in &snapshot.passages {
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
            }
            association_lines += bucket.lines.len();
            for line in &bucket.lines {
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

/// Re-renders one attribution as `count` assertion lines of its
/// average weight; the paragraph locator rides the first line only
/// (attribution locators are first-write-wins, so that reproduces
/// the stored one).
fn push_assertions<'a>(
    bucket: &mut Bucket<'a>,
    association: &'a Association,
    sum: f64,
    count: u64,
    paragraph: Option<u32>,
) {
    let weight = sum / count as f64;
    for index in 0..count {
        bucket.lines.push(AssociationLine {
            subject: &association.subject,
            label: &association.label,
            object: &association.object,
            weight,
            paragraph: (index == 0).then_some(paragraph).flatten(),
        });
    }
}

fn push_line(stream: &mut String, line: &impl Serialize) {
    stream.push_str(&serde_json::to_string(line).expect("export lines serialize infallibly"));
    stream.push('\n');
}

pub(crate) fn run(args: &[String]) -> i32 {
    let mut out: Option<PathBuf> = None;
    let mut config: Option<PathBuf> = None;
    let mut names: Vec<String> = Vec::new();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{USAGE}");
                return 0;
            }
            "--out" => match rest.next() {
                Some(path) => out = Some(PathBuf::from(path)),
                None => return usage_error("--out needs a directory path"),
            },
            "--config" => match rest.next() {
                Some(path) => config = Some(PathBuf::from(path)),
                None => return usage_error("--config needs a file path"),
            },
            other if other.starts_with('-') => {
                return usage_error(&format!("unknown flag '{other}'"));
            }
            name => names.push(name.to_string()),
        }
    }
    let Some(out) = out else {
        return usage_error("--out DIR is required (where the streams land)");
    };
    // SAFETY (same contract as serve/import): applied while the
    // process is still single-threaded — export never starts a runtime.
    if let Some(path) = &config {
        crate::cli::load_config(path);
    }

    // Registry warnings (WAL replay notes, load errors) must reach the
    // operator; stdout stays reserved for the report lines.
    crate::ingest::init_logging();
    // Export reads; it never embeds, whatever the environment says.
    let state = match crate::registry::BootConfig::from_env().boot(None) {
        Ok(state) => state,
        Err(error) => {
            eprintln!("taguru: export: {error}");
            return 1;
        }
    };

    let names = if names.is_empty() {
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
    } else {
        names
    };

    if let Err(error) = fs::create_dir_all(&out) {
        eprintln!("taguru: export: cannot create {}: {error}", out.display());
        return 1;
    }

    let mut failures = 0usize;
    for name in &names {
        match export_one(&state, name, &out) {
            Ok(report) => println!("{report}"),
            Err(message) => {
                eprintln!("taguru: export: context '{name}': {message}");
                failures += 1;
            }
        }
    }
    println!(
        "export: {} of {} context(s) written to {}",
        names.len() - failures,
        names.len(),
        out.display()
    );
    if failures > 0 { 1 } else { 0 }
}

fn export_one(state: &AppState, name: &str, out: &std::path::Path) -> Result<String, String> {
    let snapshot = state
        .export_context(name)
        .map_err(|failure| match failure {
            AccessError::NotFound => "no such context".to_string(),
            AccessError::Load(error) => error,
            AccessError::Unpersisted(error) => error,
        })?;
    let rendered = render(name, &snapshot)?;
    let path = out.join(format!("{}.jsonl", crate::registry::file_stem(name)));
    // Stage + fsync + rename, never a truncating write in place: a
    // backup that "wrote" but never reached the platter is worse than a
    // refusal, and a crash while REFRESHING an existing backup must not
    // shred the previous good copy — the exact hazard the same helper
    // guards for the server's own images.
    crate::registry::write_atomic(&path, rendered.stream.as_bytes())
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

fn usage_error(message: &str) -> i32 {
    eprintln!("taguru: export: {message} — try 'taguru export --help'");
    2
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use taguru::context::Attribution;

    use crate::ingest;
    use crate::passages::PassageSubmission;
    use crate::registry::AssocOp;

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
                        },
                    ),
                    (
                        "b.md".to_string(),
                        PassageSubmission {
                            text: "杜氏の紹介。\n\n代表銘柄の解説。".to_string(),
                            questions: Vec::new(),
                            sections: Vec::new(),
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

        let snapshot_a = state_a.export_context("sake").unwrap();
        let rendered = render("sake", &snapshot_a).unwrap();
        assert_eq!(rendered.aliases_dropped, 1, "the edgeless canonical");

        // Restore into a fresh directory, twice — the second pass
        // proves the stream is idempotent (retract-then-apply).
        let state_b =
            crate::registry::AppState::boot(scratch_dir("roundtrip-b"), usize::MAX, None).unwrap();
        for _ in 0..2 {
            let batches = ingest::parse_batches(rendered.stream.as_bytes()).unwrap();
            assert_eq!(batches.len(), 3, "a.md, b.md, and the reserved batch");
            for batch in &batches {
                ingest::apply_batch(&state_b, batch)
                    .map_err(|r| r.text())
                    .unwrap();
            }
        }
        let snapshot_b = state_b.export_context("sake").unwrap();

        // The restored context has a REAL attribution to the reserved
        // "export:unsourced" id now (import stamped it from the batch
        // header). Re-rendering must NOT refuse — it folds back into a
        // sourceless batch — so the backup stream is a true fixed point.
        let rendered_b = render("sake", &snapshot_b).expect("re-export must not refuse");
        let rendered_c = {
            let state_c =
                crate::registry::AppState::boot(scratch_dir("roundtrip-c"), usize::MAX, None)
                    .unwrap();
            for batch in ingest::parse_batches(rendered_b.stream.as_bytes()).unwrap() {
                ingest::apply_batch(&state_c, &batch)
                    .map_err(|r| r.text())
                    .unwrap();
            }
            render("sake", &state_c.export_context("sake").unwrap()).unwrap()
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
        type StoredPassage = (String, Vec<(u32, String)>, Vec<(u32, String)>);
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
            )
        );
        assert_eq!(passages_b["b.md"].1, Vec::new());
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
        let rendered = render("sake", &snapshot(vec![edge])).unwrap();
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

    #[test]
    fn fully_retracted_edges_render_as_nothing() {
        let rendered = render("sake", &snapshot(vec![association(0, Vec::new())])).unwrap();
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
        let rendered = render("sake", &snapshot(vec![edge])).unwrap();
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
        let refusal = render("sake", &snapshot(vec![edge])).unwrap_err();
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
        let rendered = render("sake", &snapshot).unwrap();
        assert_eq!(rendered.aliases, 2);
        assert_eq!(rendered.aliases_dropped, 1);
        let last_lines: Vec<&str> = rendered.stream.lines().rev().take(2).collect();
        assert!(last_lines.iter().all(|line| line.contains("\"alias\"")));
        assert!(!rendered.stream.contains("orphan"));
    }
}
