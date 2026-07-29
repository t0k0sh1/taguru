//! `differences.jsonl` (issue #259, ADR 0003 §9.4): the model-pair
//! paired diff `taguru benchmark compare` writes alongside
//! `measurements.json`/`.csv`. Reuses `benchmark::identity` (issue
//! #258) for same-ness matching, exactly as `compare`'s own
//! `stability_metrics` does — a same-ness judgment about "did two runs
//! of one model agree" and "did two models agree" is always traceable
//! to the same `Matching` parameter set, recorded verbatim in this
//! file's own header.
//!
//! **No verdict language.** #189 and #259 forbid it — no side is a
//! baseline or an "expected" answer, since there is no gold data to
//! judge against. Sides are `a`/`b`, bound to `model_id`, never
//! `baseline`/`candidate`; `present_in` is an array of `model_id`s;
//! counts are `n_present`, never `n_missing`; an absent side serializes
//! `null`, never a fault. `differences_lexicon_test` (in
//! `super::tests`) asserts no emitted key name or `kind` value matches
//! ADR 0003 §9.4's banned lexicon.
//!
//! **Fire condition: set disjointness, not inequality.** Two sides each
//! carry observations from potentially several runs. A
//! `polarity_difference`/`attribution_difference`/
//! `alias_resolution_difference` record fires only when the two sides'
//! *entire* observed sets share no element — the one claim that is
//! purely between models, since within-model run-to-run variation is
//! already `stability.*`'s concern (`measurements.json`, issue #258).
//! Both sides' full sets are still carried on the record, so a reader
//! wanting the weaker "disagreed at least once" relation can derive it
//! without re-reading `cells/**`.
//!
//! **No byte span is persisted.** `locator.paragraph` is a
//! `crate::paragraph::split` index; `--with-text` (default off)
//! recomputes the paragraph's byte range and text from the corpus file
//! on disk, after verifying it against `manifest.json`'s pinned
//! `document_sha256` — the same "recomputed, not persisted" posture
//! `src/paragraph.rs`'s own module doc states (ADR 0003 §7).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Serialize;

use super::identity;

// Bumped from 1: `pair_id` changed shape from `"{a}__{b}"` to a
// length-prefixed `"{len}:{a}__{b}"` to close a collision between
// differently-split underscored model ids — a reader keyed on the old
// format would otherwise silently fail to match a pair record it
// should have found.
const BENCHMARK_DIFFERENCES_VERSION: u64 = 2;

/// `locator.text`'s cap (ADR 0003 §9.4 names no number; this module
/// picks one so a single pathological paragraph cannot inflate the
/// artifact unboundedly). Floored to the nearest `char` boundary at or
/// below the cap, never split mid-codepoint.
const TEXT_MAX_BYTES: usize = 4096;

pub(super) struct DifferencesOptions {
    pub(super) with_text: bool,
}

// ============================== Header shapes ==============================

#[derive(Clone, Serialize)]
struct PairInfo {
    pair_id: String,
    a: String,
    b: String,
}

#[derive(Serialize)]
struct DifferencesHeader {
    kind: &'static str,
    taguru_benchmark_differences: u64,
    run_id: String,
    pairs: Vec<PairInfo>,
    matching: identity::Matching,
    text_included: bool,
}

// ============================== Record shapes ==============================

/// `sides.a`/`sides.b` — `null` when that side has no matching
/// observation, never a count of absence (ADR 0003 §9.4).
#[derive(Clone, Serialize)]
struct Sides<T> {
    a: Option<T>,
    b: Option<T>,
}

#[derive(Clone, Serialize)]
struct RunsBlock {
    runs: Vec<usize>,
    n_present: usize,
}

impl RunsBlock {
    fn from_presence(presence: &identity::KeyPresence) -> Self {
        Self {
            runs: presence.runs().collect(),
            n_present: presence.n_present(),
        }
    }

    fn from_runs(runs: &BTreeSet<usize>) -> Self {
        Self {
            runs: runs.iter().copied().collect(),
            n_present: runs.len(),
        }
    }
}

#[derive(Clone, Serialize)]
struct PolaritySide {
    runs: Vec<usize>,
    n_present: usize,
    weight_signs: Vec<i8>,
}

#[derive(Clone, Serialize)]
struct AttributionSide {
    runs: Vec<usize>,
    n_present: usize,
    paragraphs: Vec<Option<u64>>,
}

#[derive(Clone, Serialize)]
struct AliasSide {
    runs: Vec<usize>,
    n_present: usize,
    canonicals: Vec<String>,
}

#[derive(Clone, Serialize)]
struct SurfaceTriple {
    subject: String,
    label: String,
    object: String,
}

impl From<(String, String, String)> for SurfaceTriple {
    fn from((subject, label, object): (String, String, String)) -> Self {
        Self {
            subject,
            label,
            object,
        }
    }
}

#[derive(Clone, Serialize)]
struct SurfaceSide {
    runs: Vec<usize>,
    n_present: usize,
    surface_forms: Vec<SurfaceTriple>,
}

#[derive(Clone, Serialize)]
struct KeyBlock {
    subject: String,
    label: String,
    object: String,
}

/// Byte spans are never persisted (ADR 0003 §7): `paragraph` is a
/// `crate::paragraph::split` index, and `chunk_index`/`chunk_sha256`
/// are derived from it via `manifest.json`'s document dictionary at
/// diff time, not carried through from extraction. `text` stays `null`
/// unless `--with-text` was given.
#[derive(Clone, Serialize)]
struct Locator {
    document_id: String,
    source: String,
    document_sha256: String,
    paragraph: Option<u64>,
    chunk_index: Option<usize>,
    chunk_sha256: Option<String>,
    text: Option<String>,
    text_truncated: bool,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum DifferenceRecord {
    /// Whether a document counts toward this pair's association records
    /// at all — a document only one side ever completed a run for is
    /// excluded from every key-level record below and shows up only
    /// here, so "no differences on this document" and "this document
    /// was never comparable" stay distinguishable (a harness/model
    /// completion failure is `document.written_rate`'s fact, not an
    /// extraction difference — the same posture `stability_metrics`
    /// already takes, ADR 0003 §9.3).
    DocumentCoverage {
        pair_id: String,
        present_in: Vec<String>,
        sides: Sides<RunsBlock>,
    },
    AssociationShared {
        pair_id: String,
        present_in: Vec<String>,
        key: KeyBlock,
        sides: Sides<RunsBlock>,
        locator: Locator,
    },
    AssociationSingleSide {
        pair_id: String,
        present_in: Vec<String>,
        key: KeyBlock,
        sides: Sides<RunsBlock>,
        locator: Locator,
    },
    PolarityDifference {
        pair_id: String,
        present_in: Vec<String>,
        key: KeyBlock,
        sides: Sides<PolaritySide>,
        locator: Locator,
    },
    AttributionDifference {
        pair_id: String,
        present_in: Vec<String>,
        key: KeyBlock,
        sides: Sides<AttributionSide>,
        locator: Locator,
    },
    AliasResolutionDifference {
        pair_id: String,
        present_in: Vec<String>,
        alias_kind: identity::AliasKind,
        spelling: String,
        sides: Sides<AliasSide>,
        locator: Locator,
    },
    SurfaceFormVariation {
        pair_id: String,
        present_in: Vec<String>,
        key: KeyBlock,
        sides: Sides<SurfaceSide>,
        locator: Locator,
    },
}

// ============================== Per-model, per-document presence ==============================

/// One model's material for one document: which of its runs completed
/// that document (`outcome: "written"`, batch readable — the same gate
/// `stability_metrics` uses), its association keys across those runs,
/// and its alias declarations across those runs. Scoped to one document
/// (unlike `stability_metrics`' whole-model `PresenceTable`) because
/// every `differences.jsonl` record is per-document via `locator`.
#[derive(Debug, Default)]
struct SidePresence {
    completed_runs: BTreeSet<usize>,
    presence: identity::PresenceTable,
    alias_presence: BTreeMap<(identity::AliasKind, String), BTreeMap<usize, String>>,
}

fn presence_by_model_and_document(
    matching: &identity::Matching,
    doc_rows: &[super::DocRow],
) -> BTreeMap<String, BTreeMap<String, SidePresence>> {
    let mut out: BTreeMap<String, BTreeMap<String, SidePresence>> = BTreeMap::new();
    for doc in doc_rows {
        let Some(batch) = &doc.batch else { continue };
        let entry = out
            .entry(doc.model_id.clone())
            .or_default()
            .entry(doc.document_id.clone())
            .or_default();
        entry.completed_runs.insert(doc.run_index);

        let aliases = identity::AliasMap::from_batch(matching, &batch.rows.aliases);
        let keyed = identity::keyed_associations(
            matching,
            &doc.document_id,
            &batch.rows.associations,
            &aliases,
        );
        entry.presence.insert_run(doc.run_index, keyed);

        for alias_line in &batch.rows.aliases {
            let normalized = identity::normalize_term(matching, &alias_line.alias);
            let canonical = aliases.resolve(alias_line.kind, &normalized);
            entry
                .alias_presence
                .entry((alias_line.kind, normalized))
                .or_default()
                .insert(doc.run_index, canonical);
        }
    }
    out
}

fn collect_signs(presence: &identity::KeyPresence) -> BTreeSet<i8> {
    presence
        .observations()
        .flat_map(|o| o.weights.iter().copied().map(identity::weight_sign))
        .collect()
}

fn collect_paragraphs(presence: &identity::KeyPresence) -> BTreeSet<Option<u64>> {
    presence
        .observations()
        .flat_map(|o| o.paragraphs.iter().copied())
        .collect()
}

fn collect_surfaces(presence: &identity::KeyPresence) -> BTreeSet<(String, String, String)> {
    presence
        .observations()
        .flat_map(|o| o.surfaces.iter().cloned())
        .collect()
}

// ============================== Locator ==============================

/// `paragraph_first..=paragraph_last` containment against
/// `manifest.json`'s document dictionary — `None` when `paragraph` is
/// `None`, out of every chunk's range, or the document's `chunks` list
/// is empty (an older `manifest.json` predating issue #262's
/// provenance, or a document `--no-passage` stripped locators from).
fn derive_chunk(
    document: &super::super::DocumentInfo,
    paragraph: Option<u64>,
) -> (Option<usize>, Option<String>) {
    let Some(paragraph) = paragraph else {
        return (None, None);
    };
    let Ok(paragraph) = u32::try_from(paragraph) else {
        return (None, None);
    };
    match document
        .chunks
        .iter()
        .find(|chunk| chunk.paragraph_first <= paragraph && paragraph <= chunk.paragraph_last)
    {
        Some(chunk) => (Some(chunk.chunk_index), Some(chunk.chunk_sha256.clone())),
        None => (None, None),
    }
}

fn truncate_text(text: &str) -> (Option<String>, bool) {
    if text.len() <= TEXT_MAX_BYTES {
        return (Some(text.to_string()), false);
    }
    let mut cap = TEXT_MAX_BYTES;
    while !text.is_char_boundary(cap) {
        cap -= 1;
    }
    (Some(text[..cap].to_string()), true)
}

/// Reads and sha256-verifies each corpus file on disk at most once per
/// document, lazily — only when `--with-text` was given and a record
/// actually needs it. A sha256 mismatch or unreadable file is a loud
/// refusal (the `consistency_mismatch`, src/benchmark.rs, posture): text
/// the models never saw must never be embedded silently.
struct TextResolver {
    with_text: bool,
    cache: BTreeMap<String, Result<(String, Vec<crate::paragraph::ParagraphSpan>), String>>,
}

impl TextResolver {
    fn new(with_text: bool) -> Self {
        Self {
            with_text,
            cache: BTreeMap::new(),
        }
    }

    fn resolve(
        &mut self,
        document: &super::super::DocumentInfo,
        paragraph: Option<u64>,
    ) -> Result<(Option<String>, bool), String> {
        if !self.with_text {
            return Ok((None, false));
        }
        let Some(paragraph) = paragraph else {
            return Ok((None, false));
        };
        let loaded = self
            .cache
            .entry(document.document_id.clone())
            .or_insert_with(|| Self::load(document));
        let (text, spans) = match loaded.as_ref() {
            Ok(pair) => pair,
            Err(message) => return Err(message.clone()),
        };
        let Ok(paragraph) = u32::try_from(paragraph) else {
            return Ok((None, false));
        };
        // A model-cited paragraph past the corpus's own range (the
        // corpus itself is already sha256-verified above) is tolerated
        // — the same posture `analyze_batch`'s
        // `paragraph_out_of_range` counter takes, never a refusal.
        let Some(span) = spans.iter().find(|span| span.index == paragraph) else {
            return Ok((None, false));
        };
        Ok(truncate_text(&text[span.start as usize..span.end as usize]))
    }

    fn load(
        document: &super::super::DocumentInfo,
    ) -> Result<(String, Vec<crate::paragraph::ParagraphSpan>), String> {
        let text = crate::extract::read_document(Path::new(&document.path)).map_err(|error| {
            format!(
                "--with-text: cannot read {} for document {}: {error}",
                document.path, document.document_id
            )
        })?;
        let sha256 = crate::sha256::sha256_hex(text.as_bytes());
        if sha256 != document.sha256 {
            return Err(format!(
                "--with-text: {} has changed since this results directory was created \
                 (manifest.json pins document_sha256 {}, found {sha256})",
                document.path, document.sha256
            ));
        }
        let spans = crate::paragraph::split(&text);
        Ok((text, spans))
    }
}

fn build_locator(
    document: &super::super::DocumentInfo,
    document_id: &str,
    a_presence: Option<&identity::KeyPresence>,
    b_presence: Option<&identity::KeyPresence>,
    text_resolver: &mut TextResolver,
) -> Result<Locator, String> {
    // The minimum `Some(paragraph)` across the union of both sides'
    // observations — a deterministic, earliest-mention representative.
    // Each difference kind that needs the full per-side set (attribution
    // in particular) carries it directly on `sides`, so nothing is lost
    // by picking one pointer here.
    let mut paragraphs: BTreeSet<Option<u64>> = BTreeSet::new();
    if let Some(presence) = a_presence {
        paragraphs.extend(collect_paragraphs(presence));
    }
    if let Some(presence) = b_presence {
        paragraphs.extend(collect_paragraphs(presence));
    }
    let paragraph = paragraphs.into_iter().flatten().min();
    let (chunk_index, chunk_sha256) = derive_chunk(document, paragraph);
    let (text, text_truncated) = text_resolver.resolve(document, paragraph)?;
    Ok(Locator {
        document_id: document_id.to_string(),
        source: document.path.clone(),
        document_sha256: document.sha256.clone(),
        paragraph,
        chunk_index,
        chunk_sha256,
        text,
        text_truncated,
    })
}

/// Alias lines carry no paragraph locator of their own (`render_batch`
/// never attaches one) — `alias_resolution_difference` records point at
/// the document only.
fn alias_locator(document: &super::super::DocumentInfo, document_id: &str) -> Locator {
    Locator {
        document_id: document_id.to_string(),
        source: document.path.clone(),
        document_sha256: document.sha256.clone(),
        paragraph: None,
        chunk_index: None,
        chunk_sha256: None,
        text: None,
        text_truncated: false,
    }
}

// ============================== Assembly ==============================

pub(super) fn compute_differences(
    manifest: &super::super::BenchManifest,
    doc_rows: &[super::DocRow],
    options: &DifferencesOptions,
) -> Result<String, String> {
    let matching = identity::Matching::default();
    let documents_by_id: BTreeMap<&str, &super::super::DocumentInfo> = manifest
        .documents
        .iter()
        .map(|doc| (doc.document_id.as_str(), doc))
        .collect();

    let presence = presence_by_model_and_document(&matching, doc_rows);

    // Model order: manifest.json's own declaration order (§8), with any
    // model_id appearing only in `cells` (defensive — every cell should
    // name a declared model) appended lexicographically after.
    let mut model_order: Vec<String> = manifest.models.iter().map(|m| m.model_id.clone()).collect();
    {
        let known: BTreeSet<&str> = model_order.iter().map(String::as_str).collect();
        let mut extra: BTreeSet<String> = BTreeSet::new();
        for cell in &manifest.cells {
            if !known.contains(cell.model_id.as_str()) {
                extra.insert(cell.model_id.clone());
            }
        }
        drop(known);
        model_order.extend(extra);
    }

    let mut pairs: Vec<PairInfo> = Vec::new();
    for i in 0..model_order.len() {
        for j in (i + 1)..model_order.len() {
            // `a`/`b` must land in the same canonical order `pair_key`
            // itself now applies — otherwise a manifest that declares
            // its models out of lexicographic order would emit an
            // `a`/`b` pair that visibly disagrees with its own
            // `pair_id` (and with `search`'s output for the same
            // unordered pair).
            let (a, b) = super::super::canonical_pair(&model_order[i], &model_order[j]);
            pairs.push(PairInfo {
                pair_id: super::super::pair_key(a, b),
                a: a.to_string(),
                b: b.to_string(),
            });
        }
    }

    let header = DifferencesHeader {
        kind: "header",
        taguru_benchmark_differences: BENCHMARK_DIFFERENCES_VERSION,
        run_id: manifest.run_id.clone(),
        pairs: pairs.clone(),
        matching,
        text_included: options.with_text,
    };
    let mut lines: Vec<String> = vec![
        serde_json::to_string(&header)
            .map_err(|error| format!("serializing differences header: {error}"))?,
    ];

    let empty_docs: BTreeMap<String, SidePresence> = BTreeMap::new();
    let mut text_resolver = TextResolver::new(options.with_text);

    for pair in &pairs {
        let side_a_docs = presence.get(&pair.a).unwrap_or(&empty_docs);
        let side_b_docs = presence.get(&pair.b).unwrap_or(&empty_docs);

        let mut doc_ids: BTreeSet<&String> = BTreeSet::new();
        doc_ids.extend(side_a_docs.keys());
        doc_ids.extend(side_b_docs.keys());

        // Pass 1: document_coverage for every document either side
        // attempted, in document_id order.
        let mut eligible: Vec<&String> = Vec::new();
        for doc_id in &doc_ids {
            if !documents_by_id.contains_key(doc_id.as_str()) {
                eprintln!(
                    "taguru: benchmark: compare: differences: document {doc_id} has no \
                     manifest.json entry — excluded from {}",
                    pair.pair_id
                );
                continue;
            }
            let a_side = side_a_docs.get(doc_id.as_str());
            let b_side = side_b_docs.get(doc_id.as_str());
            let present_in: Vec<String> = match (a_side, b_side) {
                (Some(_), Some(_)) => vec![pair.a.clone(), pair.b.clone()],
                (Some(_), None) => vec![pair.a.clone()],
                (None, Some(_)) => vec![pair.b.clone()],
                (None, None) => continue,
            };
            let record = DifferenceRecord::DocumentCoverage {
                pair_id: pair.pair_id.clone(),
                present_in,
                sides: Sides {
                    a: a_side.map(|side| RunsBlock::from_runs(&side.completed_runs)),
                    b: b_side.map(|side| RunsBlock::from_runs(&side.completed_runs)),
                },
            };
            lines.push(
                serde_json::to_string(&record)
                    .map_err(|error| format!("serializing differences record: {error}"))?,
            );
            if a_side.is_some() && b_side.is_some() {
                eligible.push(*doc_id);
            }
        }

        // Pass 2: per eligible document, per association key in Ord
        // order — the shared/single-side record, then any
        // polarity/attribution/surface-form difference that fired.
        for doc_id in &eligible {
            let document = documents_by_id[doc_id.as_str()];
            let a_side = &side_a_docs[doc_id.as_str()];
            let b_side = &side_b_docs[doc_id.as_str()];

            let mut keys: BTreeSet<&identity::AssocKey> = BTreeSet::new();
            keys.extend(a_side.presence.iter().map(|(key, _)| key));
            keys.extend(b_side.presence.iter().map(|(key, _)| key));

            for key in keys {
                let a_presence = a_side.presence.get(key);
                let b_presence = b_side.presence.get(key);
                let present_in: Vec<String> = match (a_presence, b_presence) {
                    (Some(_), Some(_)) => vec![pair.a.clone(), pair.b.clone()],
                    (Some(_), None) => vec![pair.a.clone()],
                    (None, Some(_)) => vec![pair.b.clone()],
                    (None, None) => continue, // unreachable: key came from the union
                };
                let key_block = KeyBlock {
                    subject: key.subject.clone(),
                    label: key.label.clone(),
                    object: key.object.clone(),
                };
                let locator =
                    build_locator(document, doc_id, a_presence, b_presence, &mut text_resolver)?;

                let record = if a_presence.is_some() && b_presence.is_some() {
                    DifferenceRecord::AssociationShared {
                        pair_id: pair.pair_id.clone(),
                        present_in: present_in.clone(),
                        key: key_block.clone(),
                        sides: Sides {
                            a: a_presence.map(RunsBlock::from_presence),
                            b: b_presence.map(RunsBlock::from_presence),
                        },
                        locator: locator.clone(),
                    }
                } else {
                    DifferenceRecord::AssociationSingleSide {
                        pair_id: pair.pair_id.clone(),
                        present_in: present_in.clone(),
                        key: key_block.clone(),
                        sides: Sides {
                            a: a_presence.map(RunsBlock::from_presence),
                            b: b_presence.map(RunsBlock::from_presence),
                        },
                        locator: locator.clone(),
                    }
                };
                lines.push(
                    serde_json::to_string(&record)
                        .map_err(|error| format!("serializing differences record: {error}"))?,
                );

                if let (Some(a_presence), Some(b_presence)) = (a_presence, b_presence) {
                    let signs_a = collect_signs(a_presence);
                    let signs_b = collect_signs(b_presence);
                    if signs_a.is_disjoint(&signs_b) {
                        let record = DifferenceRecord::PolarityDifference {
                            pair_id: pair.pair_id.clone(),
                            present_in: present_in.clone(),
                            key: key_block.clone(),
                            sides: Sides {
                                a: Some(PolaritySide {
                                    runs: a_presence.runs().collect(),
                                    n_present: a_presence.n_present(),
                                    weight_signs: signs_a.into_iter().collect(),
                                }),
                                b: Some(PolaritySide {
                                    runs: b_presence.runs().collect(),
                                    n_present: b_presence.n_present(),
                                    weight_signs: signs_b.into_iter().collect(),
                                }),
                            },
                            locator: locator.clone(),
                        };
                        lines.push(
                            serde_json::to_string(&record).map_err(|error| {
                                format!("serializing differences record: {error}")
                            })?,
                        );
                    }

                    let paragraphs_a = collect_paragraphs(a_presence);
                    let paragraphs_b = collect_paragraphs(b_presence);
                    if paragraphs_a.is_disjoint(&paragraphs_b) {
                        let record = DifferenceRecord::AttributionDifference {
                            pair_id: pair.pair_id.clone(),
                            present_in: present_in.clone(),
                            key: key_block.clone(),
                            sides: Sides {
                                a: Some(AttributionSide {
                                    runs: a_presence.runs().collect(),
                                    n_present: a_presence.n_present(),
                                    paragraphs: paragraphs_a.into_iter().collect(),
                                }),
                                b: Some(AttributionSide {
                                    runs: b_presence.runs().collect(),
                                    n_present: b_presence.n_present(),
                                    paragraphs: paragraphs_b.into_iter().collect(),
                                }),
                            },
                            locator: locator.clone(),
                        };
                        lines.push(
                            serde_json::to_string(&record).map_err(|error| {
                                format!("serializing differences record: {error}")
                            })?,
                        );
                    }
                }

                // Surface-form variation: also fires on a single-side
                // key, when that one side alone used 2+ raw spellings
                // for it across its own runs.
                let surfaces_a = a_presence.map(collect_surfaces).unwrap_or_default();
                let surfaces_b = b_presence.map(collect_surfaces).unwrap_or_default();
                if surfaces_a.union(&surfaces_b).count() >= 2 {
                    let record = DifferenceRecord::SurfaceFormVariation {
                        pair_id: pair.pair_id.clone(),
                        present_in,
                        key: key_block,
                        sides: Sides {
                            a: a_presence.map(|presence| SurfaceSide {
                                runs: presence.runs().collect(),
                                n_present: presence.n_present(),
                                surface_forms: surfaces_a
                                    .iter()
                                    .cloned()
                                    .map(SurfaceTriple::from)
                                    .collect(),
                            }),
                            b: b_presence.map(|presence| SurfaceSide {
                                runs: presence.runs().collect(),
                                n_present: presence.n_present(),
                                surface_forms: surfaces_b
                                    .iter()
                                    .cloned()
                                    .map(SurfaceTriple::from)
                                    .collect(),
                            }),
                        },
                        locator,
                    };
                    lines.push(
                        serde_json::to_string(&record)
                            .map_err(|error| format!("serializing differences record: {error}"))?,
                    );
                }
            }
        }

        // Pass 3: alias_resolution_difference across every eligible
        // document, sorted by (document_id, alias_kind, spelling).
        for doc_id in &eligible {
            let document = documents_by_id[doc_id.as_str()];
            let a_side = &side_a_docs[doc_id.as_str()];
            let b_side = &side_b_docs[doc_id.as_str()];

            let mut spellings: BTreeSet<&(identity::AliasKind, String)> = BTreeSet::new();
            spellings.extend(a_side.alias_presence.keys());
            spellings.extend(b_side.alias_presence.keys());

            for spelling_key in spellings {
                let a_runs = a_side.alias_presence.get(spelling_key);
                let b_runs = b_side.alias_presence.get(spelling_key);
                let (Some(a_runs), Some(b_runs)) = (a_runs, b_runs) else {
                    continue; // only a comparison between two sides' own declarations is in scope
                };
                let canonicals_a: BTreeSet<String> = a_runs.values().cloned().collect();
                let canonicals_b: BTreeSet<String> = b_runs.values().cloned().collect();
                if !canonicals_a.is_disjoint(&canonicals_b) {
                    continue;
                }
                let (alias_kind, spelling) = spelling_key;
                let record = DifferenceRecord::AliasResolutionDifference {
                    pair_id: pair.pair_id.clone(),
                    present_in: vec![pair.a.clone(), pair.b.clone()],
                    alias_kind: *alias_kind,
                    spelling: spelling.clone(),
                    sides: Sides {
                        a: Some(AliasSide {
                            runs: a_runs.keys().copied().collect(),
                            n_present: a_runs.len(),
                            canonicals: canonicals_a.into_iter().collect(),
                        }),
                        b: Some(AliasSide {
                            runs: b_runs.keys().copied().collect(),
                            n_present: b_runs.len(),
                            canonicals: canonicals_b.into_iter().collect(),
                        }),
                    },
                    locator: alias_locator(document, doc_id),
                };
                lines.push(
                    serde_json::to_string(&record)
                        .map_err(|error| format!("serializing differences record: {error}"))?,
                );
            }
        }
    }

    Ok(lines.join("\n") + "\n")
}
