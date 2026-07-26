//! `taguru benchmark compare`: ADR 0003 §9.3's aggregation stage (issue
//! #257). Reads a finished results directory — `manifest.json`,
//! `runs/*.jsonl`, and the written `cells/**` batches — and derives
//! `measurements.json`/`measurements.csv`. Calls no model and touches
//! no network: a pure function of the results directory (R4, ADR 0003
//! §4). Every metric's unit, statistic, source, and known caveat ships
//! inside the artifact itself as a `definitions` block, not only in
//! this module's doc comments or in docs/benchmark.html.
//!
//! No single score or ranking is possible by construction: every
//! per-cell/per-model/per-document map is a `BTreeMap` (lexicographic
//! key order, never array position, so nothing here can be read as a
//! rank), and `no_banned_key_appears_anywhere_in_the_artifact` asserts
//! the emitted key set never contains `rank`/`score`/`winner`/`best`/
//! `recommended`/`overall`/`delta_vs_*` (ADR 0003 §9.3; #189's
//! acceptance criterion 10).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{Map, Value};

use crate::config::subcommand_usage_error;

const BENCHMARK_MEASUREMENTS_VERSION: u64 = 1;

const USAGE: &str = "\
usage: taguru benchmark compare RESULTS_DIR

Reads a finished results directory (manifest.json, runs/*.jsonl,
cells/**) and derives measurements.json/measurements.csv — the
distributions, rates, and extraction-shape counts ADR 0003 §9.3
defines. A pure function of RESULTS_DIR: no model is called, no
network is touched, and re-running always overwrites with a fresh
computation from what is on disk right now.

  RESULTS_DIR   a directory `taguru benchmark extract` wrote

Contract and discipline: docs/benchmark.html,
adr/0003-extraction-model-benchmark.md.
";

pub(super) fn run_compare(args: &[String]) -> i32 {
    if matches!(
        args.first().map(String::as_str),
        Some("--help") | Some("-h")
    ) {
        print!("{USAGE}");
        return 0;
    }
    let dir = match parse_args(args) {
        Ok(dir) => dir,
        Err(code) => return code,
    };
    match compute_measurements(&dir)
        .and_then(|measurements| write_measurements(&dir, &measurements))
    {
        Ok(()) => 0,
        Err(message) => {
            eprintln!("taguru: benchmark: compare: {message}");
            1
        }
    }
}

fn parse_args(args: &[String]) -> Result<PathBuf, i32> {
    let mut positional: Option<&str> = None;
    for arg in args {
        if arg.starts_with("--") {
            return Err(subcommand_usage_error(
                "benchmark",
                &format!("unknown flag '{arg}' for compare"),
            ));
        }
        if positional.is_some() {
            return Err(subcommand_usage_error(
                "benchmark",
                "compare takes exactly one RESULTS_DIR argument",
            ));
        }
        positional = Some(arg.as_str());
    }
    positional.map(PathBuf::from).ok_or_else(|| {
        subcommand_usage_error("benchmark", "compare requires a RESULTS_DIR argument")
    })
}

// ============================== Value shapes ==============================

/// `{n, min, p50, p90, p99, max, mean, sum}` (ADR 0003 §9.3). Every
/// field always serializes — `n: 0` pairs with every statistic `null`,
/// never an omitted key, so a reader tells "measured, zero samples"
/// from "this metric does not apply here" by key presence alone.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct Distribution {
    n: u64,
    min: Option<f64>,
    p50: Option<f64>,
    p90: Option<f64>,
    p99: Option<f64>,
    max: Option<f64>,
    mean: Option<f64>,
    sum: Option<f64>,
}

impl Distribution {
    const EMPTY: Self = Self {
        n: 0,
        min: None,
        p50: None,
        p90: None,
        p99: None,
        max: None,
        mean: None,
        sum: None,
    };

    /// Builds from a sample set: drops non-finite values defensively,
    /// sorts once, and applies nearest-rank (ADR 0003 §9.3) with no
    /// interpolation — always an observed value, exactly reproducible
    /// by an external re-aggregator reading the same `runs/*.jsonl`.
    fn from_samples(mut samples: Vec<f64>) -> Self {
        samples.retain(|v| v.is_finite());
        let n = samples.len();
        if n == 0 {
            return Self::EMPTY;
        }
        samples.sort_by(f64::total_cmp);
        let sum: f64 = samples.iter().sum();
        Self {
            n: n as u64,
            min: Some(samples[0]),
            p50: Some(nearest_rank(&samples, 50)),
            p90: Some(nearest_rank(&samples, 90)),
            p99: Some(nearest_rank(&samples, 99)),
            max: Some(samples[n - 1]),
            mean: Some(sum / n as f64),
            sum: Some(sum),
        }
    }
}

/// `x[ceil(p/100 * n) - 1]` — nearest-rank, no interpolation (ADR 0003
/// §9.3): always an observed value, never a synthesized one between two
/// samples. `sorted` must already be ascending and non-empty.
fn nearest_rank(sorted: &[f64], p: usize) -> f64 {
    let n = sorted.len();
    let rank = (p * n).div_ceil(100).clamp(1, n);
    sorted[rank - 1]
}

/// `{value, n, numerator}` (ADR 0003 §9.3): `numerator` attempts/
/// documents/lines matched some predicate, out of `n` total in scope.
/// `n: 0` pairs with `value: None` and `numerator: None` — never a
/// divide-by-zero `NaN`, never a silently misleading `0.0`.
#[derive(Debug, Clone, PartialEq, Serialize)]
struct Ratio {
    value: Option<f64>,
    n: u64,
    numerator: Option<u64>,
}

fn ratio_metric(numerator: u64, n: u64) -> Ratio {
    if n == 0 {
        Ratio {
            value: None,
            n: 0,
            numerator: None,
        }
    } else {
        Ratio {
            value: Some(numerator as f64 / n as f64),
            n,
            numerator: Some(numerator),
        }
    }
}

/// `{value, n}`: a pooled total, union size, or derived rate, alongside
/// the number of documents/attempts it was pooled from. Used where
/// `Ratio`'s numerator-over-n-of-the-same-population shape does not
/// fit (a cross-unit rate) or where the value is not itself a share
/// (a union size, a summed count).
#[derive(Debug, Clone, PartialEq, Serialize)]
struct Count {
    value: Option<f64>,
    n: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
enum MetricValue {
    Distribution(Distribution),
    Ratio(Ratio),
    Count(Count),
}

impl MetricValue {
    fn sample_size(&self) -> u64 {
        match self {
            MetricValue::Distribution(d) => d.n,
            MetricValue::Ratio(r) => r.n,
            MetricValue::Count(c) => c.n,
        }
    }
}

/// One `definitions` entry (ADR 0003 §9.3): unit, statistic shape,
/// applicable scopes, description, source pointer, and known caveat —
/// embedded in the artifact itself, not only documented in prose
/// elsewhere.
#[derive(Debug, Clone, Serialize)]
struct MetricDef {
    unit: &'static str,
    statistic: &'static str,
    scopes: &'static [&'static str],
    description: String,
    source: &'static str,
    caveat: Option<String>,
}

type MetricsMap = BTreeMap<String, MetricValue>;
type DocumentsSection = BTreeMap<String, BTreeMap<String, BTreeMap<String, MetricsMap>>>;

#[derive(Debug, Clone, Serialize)]
struct CellBlock {
    model_id: String,
    run_index: usize,
    #[serde(flatten)]
    metrics: MetricsMap,
}

#[derive(Debug, Clone, Serialize)]
struct InputsBlock {
    runs: Vec<String>,
    cells: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct MeasurementsFile {
    taguru_benchmark_measurements: u64,
    run_id: String,
    generated_at: String,
    percentile_method: &'static str,
    inputs: InputsBlock,
    definitions: BTreeMap<String, MetricDef>,
    cells: BTreeMap<String, CellBlock>,
    models: BTreeMap<String, MetricsMap>,
    documents: DocumentsSection,
}

// ============================== Intermediate rows ==============================

/// One `kind: "attempt"` line, denormalized with the harness identity
/// `runs/*.jsonl` already carries on every attempt record (ADR 0003
/// §9.2) so a metric can pool by cell, model, or (cell, document)
/// without re-reading the enclosing header.
#[derive(Debug)]
struct AttemptRow {
    cell_id: String,
    model_id: String,
    document_id: String,
    chunk_index: usize,
    stage: String,
    state: String,
    attempt_no: usize,
    length_limited: bool,
    elapsed_seconds: f64,
    finish_reason: Option<String>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
    provider_metadata_present: bool,
    parse_error_present: bool,
    validation_rejected: bool,
}

/// One `kind: "document"` `phase: "end"` line's payload, before it is
/// joined against its `phase: "start"` counterpart.
#[derive(Debug, Default)]
struct DocumentEndRaw {
    ts: f64,
    outcome: Option<String>,
    associations: Option<u64>,
    concepts: Option<u64>,
    labels: Option<u64>,
    questions: Option<u64>,
    duplicates: Option<u64>,
    dropped: Option<u64>,
    batch_path: Option<String>,
}

/// One (cell, document) row: the join of its `runs/*.jsonl` start/end
/// records with (when written) its `cells/**` batch's vocabulary
/// stats. This is both the raw material `documents`' own per-run
/// entries render from directly, and the sample cell/model scopes pool
/// over.
#[derive(Debug)]
struct DocRow {
    cell_id: String,
    model_id: String,
    run_index: usize,
    document_id: String,
    start_ts: Option<f64>,
    end_ts: Option<f64>,
    outcome: Option<String>,
    associations: Option<u64>,
    concepts: Option<u64>,
    labels: Option<u64>,
    questions: Option<u64>,
    duplicates: Option<u64>,
    dropped: Option<u64>,
    /// Sum of every attempt's `elapsed_seconds` for this document,
    /// precomputed once so `latency.seconds_per_association` never
    /// needs to re-filter the full attempt pool per scope.
    elapsed_seconds_sum: f64,
    /// `None` when no attempt for this document reported
    /// `input_tokens` at all (every attempt was timeout/transport, or
    /// there were none) — excluded from a rate's sample rather than
    /// treated as a zero-token document.
    input_tokens_sum: Option<u64>,
    /// `Some` only when `outcome == "written"` and the batch named by
    /// `batch_path` was readable — extraction-shape metrics (ADR 0003
    /// §9.3's "`cells/**` re-read") come from here.
    batch: Option<BatchStats>,
}

/// One written batch's vocabulary shape (ADR 0003 §9.3: association
/// weight split, distinct subjects/relations, alias orphans,
/// out-of-range paragraph locators, malformed lines) — computed by a
/// lenient line classifier, never `crate::ingest::parse_batch`'s
/// fail-fast reader, since counting malformed lines requires reading
/// past the first one.
#[derive(Debug, Default)]
struct BatchStats {
    assoc_lines: u64,
    weight_positive: u64,
    weight_negative: u64,
    paragraph_attributed: u64,
    subjects: BTreeSet<String>,
    relations: BTreeSet<String>,
    alias_orphans: u64,
    paragraph_out_of_range: u64,
    invalid_lines: u64,
}

/// Classifies every line of one `cells/**` batch file into the five
/// shapes `crate::extract::render_batch` writes (header, passage,
/// question, association, alias), tolerating and counting anything
/// else rather than failing on the first bad line — the opposite
/// trade-off from `crate::ingest::parse_batch`, which this module
/// deliberately does not reuse (ADR 0003 §9.3).
fn analyze_batch(text: &str, paragraph_count: usize) -> BatchStats {
    struct AssocRow {
        subject: String,
        object: String,
        label: String,
        weight: f64,
        paragraph: Option<u32>,
    }
    struct AliasRow {
        canonical: String,
        kind: String,
    }

    let mut assoc_rows: Vec<AssocRow> = Vec::new();
    let mut alias_rows: Vec<AliasRow> = Vec::new();
    let mut locator_paragraphs: Vec<u32> = Vec::new();
    let mut invalid_lines: u64 = 0;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(obj) = serde_json::from_str::<Map<String, Value>>(trimmed) else {
            invalid_lines += 1;
            continue;
        };
        if obj.contains_key("taguru_batch") {
            continue; // header
        }
        if obj.contains_key("passage") {
            continue;
        }
        let subject = obj.get("subject").and_then(Value::as_str);
        let label = obj.get("label").and_then(Value::as_str);
        let object = obj.get("object").and_then(Value::as_str);
        let weight = obj.get("weight").and_then(Value::as_f64);
        if let (Some(subject), Some(label), Some(object), Some(weight)) =
            (subject, label, object, weight)
        {
            let paragraph = obj
                .get("paragraph")
                .and_then(Value::as_u64)
                .map(|v| v as u32);
            if let Some(p) = paragraph {
                locator_paragraphs.push(p);
            }
            assoc_rows.push(AssocRow {
                subject: subject.to_string(),
                object: object.to_string(),
                label: label.to_string(),
                weight,
                paragraph,
            });
            continue;
        }
        let alias = obj.get("alias").and_then(Value::as_str);
        let canonical = obj.get("canonical").and_then(Value::as_str);
        let kind = obj.get("kind").and_then(Value::as_str);
        if let (Some(_alias), Some(canonical), Some(kind)) = (alias, canonical, kind) {
            alias_rows.push(AliasRow {
                canonical: canonical.to_string(),
                kind: kind.to_string(),
            });
            continue;
        }
        if obj.contains_key("question") {
            if let Some(p) = obj.get("paragraph").and_then(Value::as_u64) {
                locator_paragraphs.push(p as u32);
            }
            continue;
        }
        invalid_lines += 1;
    }

    let mut stats = BatchStats {
        invalid_lines,
        ..Default::default()
    };
    let mut concept_universe: BTreeSet<String> = BTreeSet::new();
    for row in &assoc_rows {
        stats.assoc_lines += 1;
        if row.weight > 0.0 {
            stats.weight_positive += 1;
        } else if row.weight < 0.0 {
            stats.weight_negative += 1;
        }
        if row.paragraph.is_some() {
            stats.paragraph_attributed += 1;
        }
        stats.subjects.insert(row.subject.clone());
        stats.relations.insert(row.label.clone());
        concept_universe.insert(row.subject.clone());
        concept_universe.insert(row.object.clone());
    }
    for paragraph in &locator_paragraphs {
        if *paragraph as usize >= paragraph_count {
            stats.paragraph_out_of_range += 1;
        }
    }
    for alias in &alias_rows {
        let known = match alias.kind.as_str() {
            "concept" => concept_universe.contains(&alias.canonical),
            "label" => stats.relations.contains(&alias.canonical),
            // An unrecognized alias kind is not this metric's call to
            // flag as orphaned — it is simply outside the two kinds
            // render_batch ever writes.
            _ => true,
        };
        if !known {
            stats.alias_orphans += 1;
        }
    }
    stats
}

// ============================== Aggregation ==============================

fn compute_measurements(dir: &Path) -> Result<MeasurementsFile, String> {
    let manifest = super::load_bench_manifest(&dir.join("manifest.json"))?;
    let documents_by_id: BTreeMap<&str, &super::DocumentInfo> = manifest
        .documents
        .iter()
        .map(|doc| (doc.document_id.as_str(), doc))
        .collect();

    let mut inputs_runs: Vec<String> = Vec::new();
    let mut all_attempts: Vec<AttemptRow> = Vec::new();
    let mut doc_rows: Vec<DocRow> = Vec::new();
    let mut observed_finish_reasons: BTreeSet<String> = BTreeSet::new();

    for cell in &manifest.cells {
        let runs_path = dir.join(&cell.runs_file);
        let text = fs::read_to_string(&runs_path).map_err(|error| {
            format!(
                "manifest.json names cell {} at {} but it could not be read: {error}",
                cell.cell_id,
                runs_path.display()
            )
        })?;
        inputs_runs.push(cell.runs_file.clone());

        let mut starts: BTreeMap<String, f64> = BTreeMap::new();
        let mut ends: BTreeMap<String, DocumentEndRaw> = BTreeMap::new();
        let mut cell_attempts: Vec<AttemptRow> = Vec::new();

        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
                continue;
            };
            match value.get("kind").and_then(Value::as_str) {
                Some("attempt") => {
                    let document_id = value
                        .get("document_id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let finish_reason = value
                        .pointer("/provider_metadata/finish_reason")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    if let Some(reason) = &finish_reason {
                        observed_finish_reasons.insert(reason.clone());
                    }
                    cell_attempts.push(AttemptRow {
                        cell_id: cell.cell_id.clone(),
                        model_id: cell.model_id.clone(),
                        document_id,
                        chunk_index: value
                            .get("chunk_index")
                            .and_then(Value::as_u64)
                            .unwrap_or(0) as usize,
                        stage: value
                            .get("stage")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        state: value
                            .get("state")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        attempt_no: value.get("attempt").and_then(Value::as_u64).unwrap_or(0)
                            as usize,
                        length_limited: value
                            .get("length_limited")
                            .and_then(Value::as_bool)
                            .unwrap_or(false),
                        elapsed_seconds: value
                            .get("elapsed_seconds")
                            .and_then(Value::as_f64)
                            .unwrap_or(0.0),
                        finish_reason,
                        input_tokens: value
                            .pointer("/provider_metadata/input_tokens")
                            .and_then(Value::as_u64),
                        output_tokens: value
                            .pointer("/provider_metadata/output_tokens")
                            .and_then(Value::as_u64),
                        total_tokens: value
                            .pointer("/provider_metadata/total_tokens")
                            .and_then(Value::as_u64),
                        provider_metadata_present: value
                            .get("provider_metadata")
                            .is_some_and(|v| !v.is_null()),
                        parse_error_present: value.get("parse_error").is_some_and(|v| !v.is_null()),
                        validation_rejected: value
                            .get("validation_issues")
                            .is_some_and(|v| !v.is_null()),
                    });
                }
                Some("document") => {
                    let document_id = value
                        .get("document_id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    let ts = value.get("ts").and_then(Value::as_f64).unwrap_or(0.0);
                    match value.get("phase").and_then(Value::as_str) {
                        // A resumed cell can log more than one `start`
                        // for the same document (ADR 0003 §6) — keep
                        // the earliest.
                        Some("start") => {
                            starts.entry(document_id).or_insert(ts);
                        }
                        Some("end") => {
                            ends.entry(document_id).or_insert_with(|| DocumentEndRaw {
                                ts,
                                outcome: value
                                    .get("outcome")
                                    .and_then(Value::as_str)
                                    .map(str::to_string),
                                associations: value.get("associations").and_then(Value::as_u64),
                                concepts: value.get("concepts").and_then(Value::as_u64),
                                labels: value.get("labels").and_then(Value::as_u64),
                                questions: value.get("questions").and_then(Value::as_u64),
                                duplicates: value.get("duplicates").and_then(Value::as_u64),
                                dropped: value.get("dropped").and_then(Value::as_u64),
                                batch_path: value
                                    .get("batch_path")
                                    .and_then(Value::as_str)
                                    .map(str::to_string),
                            });
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        let mut document_ids: BTreeSet<String> = BTreeSet::new();
        document_ids.extend(starts.keys().cloned());
        document_ids.extend(ends.keys().cloned());

        for document_id in document_ids {
            let start_ts = starts.get(&document_id).copied();
            let end = ends.get(&document_id);
            let attempts_for_doc: Vec<&AttemptRow> = cell_attempts
                .iter()
                .filter(|a| a.document_id == document_id)
                .collect();
            let elapsed_seconds_sum: f64 = attempts_for_doc.iter().map(|a| a.elapsed_seconds).sum();
            let input_tokens_values: Vec<u64> = attempts_for_doc
                .iter()
                .filter_map(|a| a.input_tokens)
                .collect();
            let input_tokens_sum = if input_tokens_values.is_empty() {
                None
            } else {
                Some(input_tokens_values.iter().sum())
            };

            let batch = if end.and_then(|e| e.outcome.as_deref()) == Some("written") {
                let paragraph_count = documents_by_id
                    .get(document_id.as_str())
                    .map(|d| d.paragraph_count)
                    .unwrap_or(usize::MAX);
                end.and_then(|e| e.batch_path.as_ref()).and_then(|rel| {
                    let batch_path = dir.join(rel);
                    match fs::read_to_string(&batch_path) {
                        Ok(text) => Some(analyze_batch(&text, paragraph_count)),
                        Err(error) => {
                            eprintln!(
                                "taguru: benchmark: compare: {}: {error} — excluded from \
                                 extraction-shape metrics",
                                batch_path.display()
                            );
                            None
                        }
                    }
                })
            } else {
                None
            };

            doc_rows.push(DocRow {
                cell_id: cell.cell_id.clone(),
                model_id: cell.model_id.clone(),
                run_index: cell.run_index,
                document_id: document_id.clone(),
                start_ts,
                end_ts: end.map(|e| e.ts),
                outcome: end.and_then(|e| e.outcome.clone()),
                associations: end.and_then(|e| e.associations),
                concepts: end.and_then(|e| e.concepts),
                labels: end.and_then(|e| e.labels),
                questions: end.and_then(|e| e.questions),
                duplicates: end.and_then(|e| e.duplicates),
                dropped: end.and_then(|e| e.dropped),
                elapsed_seconds_sum,
                input_tokens_sum,
                batch,
            });
        }

        all_attempts.extend(cell_attempts);
    }

    // ---- pools (borrow the now-fixed owned rows above) ----
    let mut attempts_by_cell: BTreeMap<String, Vec<&AttemptRow>> = BTreeMap::new();
    let mut attempts_by_model: BTreeMap<String, Vec<&AttemptRow>> = BTreeMap::new();
    let mut attempts_by_cell_doc: BTreeMap<(String, String), Vec<&AttemptRow>> = BTreeMap::new();
    for a in &all_attempts {
        attempts_by_cell
            .entry(a.cell_id.clone())
            .or_default()
            .push(a);
        attempts_by_model
            .entry(a.model_id.clone())
            .or_default()
            .push(a);
        attempts_by_cell_doc
            .entry((a.cell_id.clone(), a.document_id.clone()))
            .or_default()
            .push(a);
    }
    let mut docs_by_cell: BTreeMap<String, Vec<&DocRow>> = BTreeMap::new();
    let mut docs_by_model: BTreeMap<String, Vec<&DocRow>> = BTreeMap::new();
    for d in &doc_rows {
        docs_by_cell.entry(d.cell_id.clone()).or_default().push(d);
        docs_by_model.entry(d.model_id.clone()).or_default().push(d);
    }

    // ---- cells ----
    let empty_attempts: Vec<&AttemptRow> = Vec::new();
    let empty_docs: Vec<&DocRow> = Vec::new();
    let mut cells: BTreeMap<String, CellBlock> = BTreeMap::new();
    for cell in &manifest.cells {
        let attempts = attempts_by_cell
            .get(&cell.cell_id)
            .unwrap_or(&empty_attempts);
        let docs = docs_by_cell.get(&cell.cell_id).unwrap_or(&empty_docs);
        let mut metrics = attempt_distribution_metrics(attempts);
        metrics.extend(document_pooled_metrics(docs));
        metrics.extend(attempt_rate_metrics(attempts));
        metrics.extend(document_outcome_rates(docs));
        cells.insert(
            cell.cell_id.clone(),
            CellBlock {
                model_id: cell.model_id.clone(),
                run_index: cell.run_index,
                metrics,
            },
        );
    }

    // ---- models ----
    let mut model_ids: BTreeSet<String> =
        manifest.models.iter().map(|m| m.model_id.clone()).collect();
    for cell in &manifest.cells {
        model_ids.insert(cell.model_id.clone());
    }
    let mut models: BTreeMap<String, MetricsMap> = BTreeMap::new();
    for model_id in &model_ids {
        let attempts = attempts_by_model.get(model_id).unwrap_or(&empty_attempts);
        let docs = docs_by_model.get(model_id).unwrap_or(&empty_docs);
        let mut metrics = attempt_distribution_metrics(attempts);
        metrics.extend(document_pooled_metrics(docs));
        metrics.extend(attempt_rate_metrics(attempts));
        metrics.extend(document_outcome_rates(docs));
        let cell_total = manifest
            .cells
            .iter()
            .filter(|c| &c.model_id == model_id)
            .count() as u64;
        let cell_complete = manifest
            .cells
            .iter()
            .filter(|c| &c.model_id == model_id && c.outcome == "complete")
            .count() as u64;
        metrics.insert(
            "cell.complete_rate".to_string(),
            MetricValue::Ratio(ratio_metric(cell_complete, cell_total)),
        );
        models.insert(model_id.clone(), metrics);
    }

    // ---- documents (run-level granularity: model -> document -> run_label) ----
    let mut documents: DocumentsSection = BTreeMap::new();
    for doc in &doc_rows {
        let attempts_for_doc = attempts_by_cell_doc
            .get(&(doc.cell_id.clone(), doc.document_id.clone()))
            .cloned()
            .unwrap_or_default();
        let metrics = document_scope_metrics(doc, &attempts_for_doc);
        documents
            .entry(doc.model_id.clone())
            .or_default()
            .entry(doc.document_id.clone())
            .or_default()
            .insert(super::run_label(doc.run_index), metrics);
    }

    inputs_runs.sort();
    let definitions = build_definitions(&observed_finish_reasons);

    Ok(MeasurementsFile {
        taguru_benchmark_measurements: BENCHMARK_MEASUREMENTS_VERSION,
        run_id: manifest.run_id.clone(),
        generated_at: super::iso8601_utc(super::now_unix_secs()),
        percentile_method: "nearest-rank",
        inputs: InputsBlock {
            runs: inputs_runs,
            cells: "cells/",
        },
        definitions,
        cells,
        models,
        documents,
    })
}

fn distribution_metric(values: impl Iterator<Item = Option<f64>>) -> Distribution {
    Distribution::from_samples(values.flatten().collect())
}

/// Groups `stage != "cross_chunk"` attempts by `(cell_id, document_id,
/// chunk_index)`, summing `elapsed_seconds` across every retry of the
/// same chunk — cross-chunk alias correction is a separate pass over
/// an already-extracted chunk and does not belong to "how long did
/// this chunk take to extract".
fn chunk_seconds_distribution(attempts: &[&AttemptRow]) -> Distribution {
    let mut sums: BTreeMap<(String, String, usize), f64> = BTreeMap::new();
    for a in attempts {
        if a.stage == "cross_chunk" {
            continue;
        }
        *sums
            .entry((a.cell_id.clone(), a.document_id.clone(), a.chunk_index))
            .or_insert(0.0) += a.elapsed_seconds;
    }
    Distribution::from_samples(sums.into_values().collect())
}

fn output_tokens_per_second(a: &AttemptRow) -> Option<f64> {
    let tokens = a.output_tokens?;
    if a.elapsed_seconds <= 0.0 {
        return None;
    }
    Some(tokens as f64 / a.elapsed_seconds)
}

fn attempt_distribution_metrics(attempts: &[&AttemptRow]) -> MetricsMap {
    let mut m = MetricsMap::new();
    m.insert(
        "latency.attempt_seconds".to_string(),
        MetricValue::Distribution(distribution_metric(
            attempts.iter().map(|a| Some(a.elapsed_seconds)),
        )),
    );
    m.insert(
        "latency.chunk_seconds".to_string(),
        MetricValue::Distribution(chunk_seconds_distribution(attempts)),
    );
    m.insert(
        "throughput.output_tokens_per_second".to_string(),
        MetricValue::Distribution(distribution_metric(
            attempts.iter().map(|a| output_tokens_per_second(a)),
        )),
    );
    m.insert(
        "tokens.input_per_attempt".to_string(),
        MetricValue::Distribution(distribution_metric(
            attempts.iter().map(|a| a.input_tokens.map(|v| v as f64)),
        )),
    );
    m.insert(
        "tokens.output_per_attempt".to_string(),
        MetricValue::Distribution(distribution_metric(
            attempts.iter().map(|a| a.output_tokens.map(|v| v as f64)),
        )),
    );
    m.insert(
        "tokens.total_per_attempt".to_string(),
        MetricValue::Distribution(distribution_metric(
            attempts.iter().map(|a| a.total_tokens.map(|v| v as f64)),
        )),
    );
    m
}

fn wall_seconds(doc: &DocRow) -> Option<f64> {
    match (doc.start_ts, doc.end_ts) {
        (Some(start), Some(end)) => Some(end - start),
        _ => None,
    }
}

fn seconds_per_association(doc: &DocRow) -> Option<f64> {
    let associations = doc.associations?;
    if associations == 0 {
        return None;
    }
    Some(doc.elapsed_seconds_sum / associations as f64)
}

/// The pooled (cell/model-scope) view of every document-derived
/// metric: a `Distribution` over the scope's documents, one sample per
/// document. Document scope's own view of the *same* metric names is
/// [`document_scope_metrics`], which uses a `Count`/`Ratio` shape at
/// that finer grain instead (see each metric's `caveat` in
/// [`build_definitions`] for why the shape differs by scope).
fn document_pooled_metrics(docs: &[&DocRow]) -> MetricsMap {
    let mut m = MetricsMap::new();
    m.insert(
        "latency.document_wall_seconds".to_string(),
        MetricValue::Distribution(distribution_metric(docs.iter().map(|d| wall_seconds(d)))),
    );
    m.insert(
        "latency.seconds_per_association".to_string(),
        MetricValue::Distribution(distribution_metric(
            docs.iter().map(|d| seconds_per_association(d)),
        )),
    );
    m.extend(extraction_count_metrics_pooled(docs));
    m.extend(vocabulary_metrics_pooled(docs));
    m.extend(rate_and_ratio_metrics_over_docs(docs));
    m
}

macro_rules! extraction_field_pooled {
    ($m:expr, $name:expr, $docs:expr, $field:ident) => {
        $m.insert(
            $name.to_string(),
            MetricValue::Distribution(distribution_metric(
                $docs.iter().map(|d| d.$field.map(|v| v as f64)),
            )),
        );
    };
}

fn extraction_count_metrics_pooled(docs: &[&DocRow]) -> MetricsMap {
    let mut m = MetricsMap::new();
    extraction_field_pooled!(m, "extraction.associations", docs, associations);
    extraction_field_pooled!(m, "extraction.concepts", docs, concepts);
    extraction_field_pooled!(m, "extraction.labels", docs, labels);
    extraction_field_pooled!(m, "extraction.questions", docs, questions);
    extraction_field_pooled!(m, "extraction.duplicates", docs, duplicates);
    extraction_field_pooled!(m, "extraction.dropped", docs, dropped);
    m
}

fn extraction_count_metrics_document(doc: &DocRow) -> MetricsMap {
    let mut m = MetricsMap::new();
    macro_rules! c {
        ($name:expr, $field:ident) => {
            m.insert(
                $name.to_string(),
                MetricValue::Count(Count {
                    value: doc.$field.map(|v| v as f64),
                    n: u64::from(doc.$field.is_some()),
                }),
            );
        };
    }
    c!("extraction.associations", associations);
    c!("extraction.concepts", concepts);
    c!("extraction.labels", labels);
    c!("extraction.questions", questions);
    c!("extraction.duplicates", duplicates);
    c!("extraction.dropped", dropped);
    m
}

fn vocabulary_metrics_pooled(docs: &[&DocRow]) -> MetricsMap {
    let mut m = MetricsMap::new();
    m.insert(
        "extraction.weight_positive".to_string(),
        MetricValue::Distribution(distribution_metric(
            docs.iter()
                .map(|d| d.batch.as_ref().map(|b| b.weight_positive as f64)),
        )),
    );
    m.insert(
        "extraction.weight_negative".to_string(),
        MetricValue::Distribution(distribution_metric(
            docs.iter()
                .map(|d| d.batch.as_ref().map(|b| b.weight_negative as f64)),
        )),
    );
    m.insert(
        "extraction.subjects_distinct".to_string(),
        MetricValue::Distribution(distribution_metric(
            docs.iter()
                .map(|d| d.batch.as_ref().map(|b| b.subjects.len() as f64)),
        )),
    );
    m.insert(
        "extraction.relations_distinct".to_string(),
        MetricValue::Distribution(distribution_metric(
            docs.iter()
                .map(|d| d.batch.as_ref().map(|b| b.relations.len() as f64)),
        )),
    );
    m.insert(
        "extraction.alias_orphans".to_string(),
        MetricValue::Distribution(distribution_metric(
            docs.iter()
                .map(|d| d.batch.as_ref().map(|b| b.alias_orphans as f64)),
        )),
    );
    m.insert(
        "extraction.paragraph_out_of_range".to_string(),
        MetricValue::Distribution(distribution_metric(
            docs.iter()
                .map(|d| d.batch.as_ref().map(|b| b.paragraph_out_of_range as f64)),
        )),
    );
    m.insert(
        "extraction.batch_lines_invalid".to_string(),
        MetricValue::Distribution(distribution_metric(
            docs.iter()
                .map(|d| d.batch.as_ref().map(|b| b.invalid_lines as f64)),
        )),
    );
    m
}

fn vocabulary_metrics_document(doc: &DocRow) -> MetricsMap {
    let mut m = MetricsMap::new();
    macro_rules! c {
        ($name:expr, $field:expr) => {
            m.insert(
                $name.to_string(),
                MetricValue::Count(Count {
                    value: doc.batch.as_ref().map(|b| $field(b) as f64),
                    n: u64::from(doc.batch.is_some()),
                }),
            );
        };
    }
    c!("extraction.weight_positive", |b: &BatchStats| b
        .weight_positive);
    c!("extraction.weight_negative", |b: &BatchStats| b
        .weight_negative);
    c!(
        "extraction.subjects_distinct",
        |b: &BatchStats| b.subjects.len() as u64
    );
    c!(
        "extraction.relations_distinct",
        |b: &BatchStats| b.relations.len() as u64
    );
    c!("extraction.alias_orphans", |b: &BatchStats| b.alias_orphans);
    c!("extraction.paragraph_out_of_range", |b: &BatchStats| b
        .paragraph_out_of_range);
    c!("extraction.batch_lines_invalid", |b: &BatchStats| b
        .invalid_lines);
    m
}

/// `extraction.paragraph_attributed_rate`, `extraction.relation_reuse_ratio`,
/// and `extraction.associations_per_1k_input_tokens` — uniform in
/// shape at every scope (unlike the count-like metrics above), since
/// pooling by summing numerator/denominator across documents is
/// meaningful even for a single document (a one-element "pool").
fn rate_and_ratio_metrics_over_docs(docs: &[&DocRow]) -> MetricsMap {
    let mut m = MetricsMap::new();
    let mut attributed = 0u64;
    let mut assoc_lines_total = 0u64;
    let mut relations_distinct_sum = 0u64;
    let mut assoc_for_rate = 0u64;
    let mut tokens_for_rate = 0u64;
    let mut rate_docs = 0u64;
    for d in docs {
        if let Some(b) = &d.batch {
            attributed += b.paragraph_attributed;
            assoc_lines_total += b.assoc_lines;
            relations_distinct_sum += b.relations.len() as u64;
        }
        if let (Some(a), Some(t)) = (d.associations, d.input_tokens_sum) {
            assoc_for_rate += a;
            tokens_for_rate += t;
            rate_docs += 1;
        }
    }
    m.insert(
        "extraction.paragraph_attributed_rate".to_string(),
        MetricValue::Ratio(ratio_metric(attributed, assoc_lines_total)),
    );
    let reuse_numerator = assoc_lines_total.saturating_sub(relations_distinct_sum);
    m.insert(
        "extraction.relation_reuse_ratio".to_string(),
        MetricValue::Ratio(ratio_metric(reuse_numerator, assoc_lines_total)),
    );
    let rate = if rate_docs == 0 || tokens_for_rate == 0 {
        Count { value: None, n: 0 }
    } else {
        Count {
            value: Some(assoc_for_rate as f64 / (tokens_for_rate as f64 / 1000.0)),
            n: rate_docs,
        }
    };
    m.insert(
        "extraction.associations_per_1k_input_tokens".to_string(),
        MetricValue::Count(rate),
    );
    m
}

fn document_outcome_rates(docs: &[&DocRow]) -> MetricsMap {
    let total = docs.len() as u64;
    let written = docs
        .iter()
        .filter(|d| d.outcome.as_deref() == Some("written"))
        .count() as u64;
    let failed = docs
        .iter()
        .filter(|d| d.outcome.as_deref() == Some("failed"))
        .count() as u64;
    let mut m = MetricsMap::new();
    m.insert(
        "document.written_rate".to_string(),
        MetricValue::Ratio(ratio_metric(written, total)),
    );
    m.insert(
        "document.failed_rate".to_string(),
        MetricValue::Ratio(ratio_metric(failed, total)),
    );
    m
}

const ATTEMPT_STATES: [&str; 7] = [
    "stop_valid",
    "stop_malformed",
    "length_limited",
    "empty",
    "refusal",
    "timeout",
    "transport",
];

fn attempt_rate_metrics(attempts: &[&AttemptRow]) -> MetricsMap {
    let total = attempts.len() as u64;
    let mut m = MetricsMap::new();
    for state in ATTEMPT_STATES {
        let n = attempts.iter().filter(|a| a.state == state).count() as u64;
        m.insert(
            format!("attempt.state_rate.{state}"),
            MetricValue::Ratio(ratio_metric(n, total)),
        );
    }
    let length_limited = attempts.iter().filter(|a| a.length_limited).count() as u64;
    m.insert(
        "attempt.length_limited_rate".to_string(),
        MetricValue::Ratio(ratio_metric(length_limited, total)),
    );
    let retries = attempts.iter().filter(|a| a.attempt_no > 1).count() as u64;
    m.insert(
        "attempt.retry_rate".to_string(),
        MetricValue::Ratio(ratio_metric(retries, total)),
    );
    let parse_errors = attempts.iter().filter(|a| a.parse_error_present).count() as u64;
    m.insert(
        "attempt.parse_error_rate".to_string(),
        MetricValue::Ratio(ratio_metric(parse_errors, total)),
    );
    let validation_rejected = attempts.iter().filter(|a| a.validation_rejected).count() as u64;
    m.insert(
        "attempt.validation_rejected_rate".to_string(),
        MetricValue::Ratio(ratio_metric(validation_rejected, total)),
    );
    let metadata_missing = attempts
        .iter()
        .filter(|a| !a.provider_metadata_present || a.finish_reason.is_none())
        .count() as u64;
    m.insert(
        "attempt.provider_metadata_missing_rate".to_string(),
        MetricValue::Ratio(ratio_metric(metadata_missing, total)),
    );
    let mut reasons: BTreeSet<&str> = BTreeSet::new();
    for a in attempts {
        if let Some(reason) = &a.finish_reason {
            reasons.insert(reason.as_str());
        }
    }
    for reason in reasons {
        let n = attempts
            .iter()
            .filter(|a| a.finish_reason.as_deref() == Some(reason))
            .count() as u64;
        m.insert(
            format!("attempt.finish_reason_rate.{reason}"),
            MetricValue::Ratio(ratio_metric(n, total)),
        );
    }
    m
}

/// Document scope's own metric set: `Count`/`Ratio` shapes at the
/// (cell, document) grain, versus the `Distribution`-over-documents
/// shape [`document_pooled_metrics`] uses at cell/model scope for the
/// same metric names.
fn document_scope_metrics(doc: &DocRow, attempts_for_doc: &[&AttemptRow]) -> MetricsMap {
    let mut m = attempt_distribution_metrics(attempts_for_doc);
    let single = [doc];
    m.insert(
        "latency.document_wall_seconds".to_string(),
        MetricValue::Distribution(distribution_metric(std::iter::once(wall_seconds(doc)))),
    );
    m.insert(
        "latency.seconds_per_association".to_string(),
        MetricValue::Distribution(distribution_metric(std::iter::once(
            seconds_per_association(doc),
        ))),
    );
    m.extend(extraction_count_metrics_document(doc));
    m.extend(vocabulary_metrics_document(doc));
    m.extend(rate_and_ratio_metrics_over_docs(&single));
    m
}

// ============================== Definitions ==============================

fn def(
    unit: &'static str,
    statistic: &'static str,
    scopes: &'static [&'static str],
    description: impl Into<String>,
    source: &'static str,
    caveat: Option<&str>,
) -> MetricDef {
    MetricDef {
        unit,
        statistic,
        scopes,
        description: description.into(),
        source,
        caveat: caveat.map(str::to_string),
    }
}

const CMD_SCOPES: &[&str] = &["cell", "model", "document"];
const CM_SCOPES: &[&str] = &["cell", "model"];
const M_SCOPES: &[&str] = &["model"];

const COUNT_SCOPE_CAVEAT: &str = "At document scope this is a single count ({value, n: 0 or 1}), \
     not a distribution — a document that was not written (failed or interrupted), or whose \
     batch could not be read, has value: null, n: 0. At cell/model scope this is the \
     distribution of that count across the scope's documents.";

fn build_definitions(observed_finish_reasons: &BTreeSet<String>) -> BTreeMap<String, MetricDef> {
    let mut d = BTreeMap::new();

    d.insert(
        "latency.attempt_seconds".to_string(),
        def(
            "second",
            "distribution",
            CMD_SCOPES,
            "Wall time of one completion call, request start to parsed response.",
            "runs/*.jsonl kind=attempt .elapsed_seconds",
            Some(
                "Includes attempts that ended in timeout or transport failure, whose \
                 elapsed_seconds reflects when the harness gave up rather than useful model \
                 work.",
            ),
        ),
    );
    d.insert(
        "latency.chunk_seconds".to_string(),
        def(
            "second",
            "distribution",
            CMD_SCOPES,
            "Time to extract one chunk, summing every retry of that chunk's attempts.",
            "runs/*.jsonl kind=attempt .elapsed_seconds, summed per (cell_id, document_id, \
             chunk_index) for stage=item attempts",
            Some(
                "A chunk retried N times reports the sum of all N attempts, not wall time; \
                 under --parallel this sum can exceed the run's actual wall-clock span. \
                 Cross-chunk alias correction (stage=cross_chunk) is a separate pass over an \
                 already-extracted chunk and is excluded.",
            ),
        ),
    );
    d.insert(
        "latency.document_wall_seconds".to_string(),
        def(
            "second",
            "distribution",
            CMD_SCOPES,
            "Wall-clock span from a document's first phase=start record to its phase=end record.",
            "runs/*.jsonl kind=document .ts at phase=start and phase=end",
            Some(
                "A resumed cell can log more than one phase=start for the same document (ADR \
                 0003 §6); the earliest is used. A document with no phase=end (interrupted) is \
                 excluded from this metric's sample — see document.written_rate for its count.",
            ),
        ),
    );
    d.insert(
        "latency.seconds_per_association".to_string(),
        def(
            "second",
            "distribution",
            CMD_SCOPES,
            "A written document's total attempt time divided by the number of associations it \
             produced.",
            "sum of runs/*.jsonl kind=attempt .elapsed_seconds for a document, divided by that \
             document's kind=document .associations",
            Some(
                "Documents that were not written, or that produced zero associations, are \
                 excluded rather than reported as a divide-by-zero.",
            ),
        ),
    );
    d.insert(
        "throughput.output_tokens_per_second".to_string(),
        def(
            "token/second",
            "distribution",
            CMD_SCOPES,
            "One attempt's output token count divided by its elapsed time.",
            "runs/*.jsonl kind=attempt .provider_metadata.output_tokens / .elapsed_seconds",
            Some(
                "elapsed_seconds is end-to-end wall time including network and queueing, not a \
                 pure decode-time measurement. Attempts with no provider_metadata (timeout, \
                 transport) are excluded.",
            ),
        ),
    );
    for (metric, field) in [
        ("tokens.input_per_attempt", "input_tokens"),
        ("tokens.output_per_attempt", "output_tokens"),
        ("tokens.total_per_attempt", "total_tokens"),
    ] {
        d.insert(
            metric.to_string(),
            def(
                "token",
                "distribution",
                CMD_SCOPES,
                format!("{field} reported for one completion call."),
                match field {
                    "input_tokens" => "runs/*.jsonl kind=attempt .provider_metadata.input_tokens",
                    "output_tokens" => "runs/*.jsonl kind=attempt .provider_metadata.output_tokens",
                    _ => "runs/*.jsonl kind=attempt .provider_metadata.total_tokens",
                },
                Some(
                    "null for every timeout/transport attempt, since no response was ever \
                     received; those attempts are excluded from n rather than counted as 0.",
                ),
            ),
        );
    }
    d.insert(
        "extraction.associations_per_1k_input_tokens".to_string(),
        def(
            "association/kilotoken",
            "rate",
            CMD_SCOPES,
            "Associations produced per 1,000 input tokens spent, pooling every written \
             document's associations and every attempt's input tokens in scope.",
            "kind=document .associations summed, divided by kind=attempt \
             .provider_metadata.input_tokens summed / 1000",
            Some(
                "The token denominator includes retries, so a document that needed several \
                 attempts spends more tokens per association than one that succeeded on the \
                 first try — this reflects total cost, not per-call efficiency.",
            ),
        ),
    );

    let state_caveats: BTreeMap<&str, &str> = BTreeMap::from([
        (
            "stop_malformed",
            "Conflates a JSON syntax failure with a Stage 1 validation rejection; \
             .validation_issues != null (see attempt.validation_rejected_rate) separates them \
             (ADR 0001 §7; src/extract.rs:2269).",
        ),
        (
            "length_limited",
            "The legacy (non-ladder) extraction path never reports this state — it reports \
             stop_valid with length_limited: true instead. attempt.length_limited_rate reads \
             the flag directly and is not subject to this gap.",
        ),
        (
            "refusal",
            "The legacy (non-ladder) extraction path folds a refusal into stop_valid rather \
             than reporting this state; this rate only sees refusals from the ladder path.",
        ),
    ]);
    for state in ATTEMPT_STATES {
        d.insert(
            format!("attempt.state_rate.{state}"),
            def(
                "ratio",
                "ratio",
                CM_SCOPES,
                format!(
                    "Share of attempts in scope whose terminal state was {state} (ADR 0001 §7)."
                ),
                "runs/*.jsonl kind=attempt .state",
                state_caveats.get(state).copied(),
            ),
        );
    }
    d.insert(
        "attempt.length_limited_rate".to_string(),
        def(
            "ratio",
            "ratio",
            CM_SCOPES,
            "Share of attempts in scope whose length_limited flag was set, independent of which \
             state the attempt otherwise terminated in.",
            "runs/*.jsonl kind=attempt .length_limited",
            Some(
                "Catches truncations the legacy path reports as stop_valid + length_limited: \
                 true, which attempt.state_rate.length_limited alone would miss.",
            ),
        ),
    );
    d.insert(
        "attempt.retry_rate".to_string(),
        def(
            "ratio",
            "ratio",
            CM_SCOPES,
            "Share of attempts in scope that were not the first attempt at their chunk.",
            "runs/*.jsonl kind=attempt .attempt > 1",
            Some("Counts attempt occurrences, not chunks — a chunk retried twice contributes 2 to the numerator."),
        ),
    );
    d.insert(
        "attempt.parse_error_rate".to_string(),
        def(
            "ratio",
            "ratio",
            CM_SCOPES,
            "Share of attempts in scope whose response failed to parse as the expected \
             structure.",
            "runs/*.jsonl kind=attempt .parse_error != null",
            None,
        ),
    );
    d.insert(
        "attempt.validation_rejected_rate".to_string(),
        def(
            "ratio",
            "ratio",
            CM_SCOPES,
            "Share of attempts in scope whose parsed response was rejected by Stage 1 \
             validation.",
            "runs/*.jsonl kind=attempt .validation_issues != null",
            Some(
                "Together with attempt.state_rate.stop_malformed, this separates a JSON syntax \
                 failure (validation_issues null) from a validation rejection (validation_issues \
                 present) — ADR 0001 §7.",
            ),
        ),
    );
    for reason in observed_finish_reasons {
        d.insert(
            format!("attempt.finish_reason_rate.{reason}"),
            def(
                "ratio",
                "ratio",
                CM_SCOPES,
                format!(
                    "Share of attempts in scope whose provider-reported finish_reason was \
                     {reason}."
                ),
                "runs/*.jsonl kind=attempt .provider_metadata.finish_reason",
                Some(
                    "The denominator is every attempt in scope, including timeout/transport \
                     attempts that carry no provider_metadata at all.",
                ),
            ),
        );
    }
    d.insert(
        "attempt.provider_metadata_missing_rate".to_string(),
        def(
            "ratio",
            "ratio",
            CM_SCOPES,
            "Share of attempts in scope with no provider_metadata, or with provider_metadata \
             but no finish_reason.",
            "runs/*.jsonl kind=attempt .provider_metadata, .provider_metadata.finish_reason",
            Some(
                "Always 1.0 within any timeout/transport-only slice — no response was ever \
                 received to carry metadata.",
            ),
        ),
    );
    d.insert(
        "document.written_rate".to_string(),
        def(
            "ratio",
            "ratio",
            CM_SCOPES,
            "Share of documents in scope that reached outcome=written, over every document that \
             has at least a phase=start record.",
            "runs/*.jsonl kind=document phase=end .outcome==written, over phase=start records",
            Some("A document with a start but no end this run (interrupted) counts in the denominator only."),
        ),
    );
    d.insert(
        "document.failed_rate".to_string(),
        def(
            "ratio",
            "ratio",
            CM_SCOPES,
            "Share of documents in scope that reached outcome=failed, over every document that \
             has at least a phase=start record.",
            "runs/*.jsonl kind=document phase=end .outcome==failed, over phase=start records",
            Some("A document with a start but no end this run (interrupted) counts in the denominator only."),
        ),
    );
    d.insert(
        "cell.complete_rate".to_string(),
        def(
            "ratio",
            "ratio",
            M_SCOPES,
            "Share of a model's cells whose outcome is complete.",
            "manifest.json .cells[].outcome==complete, scoped to one model_id",
            Some(
                "Read from manifest.json, not from a runs/*.jsonl kind=cell line — the harness \
                 omits that line entirely for an interrupted cell by design (ADR 0003 §9.2), so \
                 manifest.json is the only reliable source for this rate.",
            ),
        ),
    );

    for (metric, unit, description) in [
        (
            "extraction.associations",
            "association",
            "Number of associations written for a document.",
        ),
        (
            "extraction.concepts",
            "alias",
            "Number of concept aliases (subject/object canonicalizations) written for a \
             document.",
        ),
        (
            "extraction.labels",
            "alias",
            "Number of label aliases (relation canonicalizations) written for a document.",
        ),
        (
            "extraction.questions",
            "question",
            "Number of doc2query search questions written for a document.",
        ),
        (
            "extraction.duplicates",
            "association",
            "Number of duplicate records merged while extracting a document.",
        ),
        (
            "extraction.dropped",
            "record",
            "Number of malformed records dropped while extracting a document.",
        ),
    ] {
        d.insert(
            metric.to_string(),
            def(
                unit,
                "distribution",
                CMD_SCOPES,
                description,
                "runs/*.jsonl kind=document phase=end",
                Some(COUNT_SCOPE_CAVEAT),
            ),
        );
    }

    d.insert(
        "extraction.weight_positive".to_string(),
        def(
            "association",
            "distribution",
            CMD_SCOPES,
            "Number of association lines in a written batch whose weight is greater than 0.",
            "cells/**/*.jsonl association lines, .weight > 0",
            Some(&format!(
                "A weight of exactly 0 counts toward neither this metric nor \
                 extraction.weight_negative. {COUNT_SCOPE_CAVEAT}"
            )),
        ),
    );
    d.insert(
        "extraction.weight_negative".to_string(),
        def(
            "association",
            "distribution",
            CMD_SCOPES,
            "Number of association lines in a written batch whose weight is less than 0.",
            "cells/**/*.jsonl association lines, .weight < 0",
            Some(&format!(
                "A weight of exactly 0 counts toward neither this metric nor \
                 extraction.weight_positive. {COUNT_SCOPE_CAVEAT}"
            )),
        ),
    );
    d.insert(
        "extraction.subjects_distinct".to_string(),
        def(
            "subject",
            "distribution",
            CMD_SCOPES,
            "Number of distinct .subject values among a written batch's association lines.",
            "cells/**/*.jsonl association lines, distinct .subject",
            Some(&format!(
                "This is the distinct-subject count within each document (or, at cell/model \
                 scope, the distribution of that per-document count) — not the union of \
                 subjects across an entire run or model. {COUNT_SCOPE_CAVEAT}"
            )),
        ),
    );
    d.insert(
        "extraction.relations_distinct".to_string(),
        def(
            "relation",
            "distribution",
            CMD_SCOPES,
            "Number of distinct .label values among a written batch's association lines.",
            "cells/**/*.jsonl association lines, distinct .label",
            Some(&format!(
                "taguru's own batch format calls this field label, not relation; this metric \
                 counts distinct label values. It is the distinct-relation count within each \
                 document, not the union across an entire run or model. {COUNT_SCOPE_CAVEAT}"
            )),
        ),
    );
    d.insert(
        "extraction.alias_orphans".to_string(),
        def(
            "alias",
            "distribution",
            CMD_SCOPES,
            "Number of alias lines in a written batch whose canonical does not match any \
             subject/object (kind=concept) or label (kind=label) used in the same batch.",
            "cells/**/*.jsonl alias lines vs. the same batch's association lines",
            Some(&format!(
                "extract already drops orphaned aliases before writing a batch (src/extract.rs), \
                 so a nonzero count here signals a harness or writer defect rather than normal \
                 model behavior. {COUNT_SCOPE_CAVEAT}"
            )),
        ),
    );
    d.insert(
        "extraction.paragraph_out_of_range".to_string(),
        def(
            "record",
            "distribution",
            CMD_SCOPES,
            "Number of association/question lines in a written batch whose paragraph locator is \
             at or past the document's paragraph_count.",
            "cells/**/*.jsonl association/question .paragraph vs. manifest.json \
             documents[].paragraph_count",
            Some(COUNT_SCOPE_CAVEAT),
        ),
    );
    d.insert(
        "extraction.batch_lines_invalid".to_string(),
        def(
            "line",
            "distribution",
            CMD_SCOPES,
            "Number of lines in a written batch that parse as JSON but match none of the batch \
             format's known shapes (header, passage, question, association, alias), or that fail \
             to parse as JSON at all.",
            "cells/**/*.jsonl, classified leniently rather than with crate::ingest's fail-fast \
             reader",
            Some(COUNT_SCOPE_CAVEAT),
        ),
    );
    d.insert(
        "extraction.paragraph_attributed_rate".to_string(),
        def(
            "ratio",
            "ratio",
            CMD_SCOPES,
            "Share of association lines in a written batch that carry a paragraph locator.",
            "cells/**/*.jsonl association lines with a .paragraph field, over all association \
             lines in the batch",
            Some(
                "With --no-passage, taguru's own batch writer (render_batch, src/extract.rs) \
                 strips every paragraph locator before writing, since there is no passage line \
                 left to locate into — this metric is structurally 0 for a --no-passage run \
                 regardless of what the model attributed, not a measurement of model behavior.",
            ),
        ),
    );
    d.insert(
        "extraction.relation_reuse_ratio".to_string(),
        def(
            "ratio",
            "ratio",
            CMD_SCOPES,
            "Share of a scope's association lines whose label was not the first occurrence of \
             that label within its own document.",
            "cells/**/*.jsonl: (association lines - distinct .label values) per document, \
             pooled by summing both across documents in scope",
            Some(
                "Computed per document, then pooled by summing the numerator and denominator \
                 across documents in scope — a label reused only across different documents, \
                 never within one, is not counted as reuse here.",
            ),
        ),
    );

    d
}

// ============================== CSV projection ==============================

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn metric_csv_rows(value: &MetricValue) -> Vec<(&'static str, Option<f64>)> {
    match value {
        MetricValue::Distribution(d) => vec![
            ("n", Some(d.n as f64)),
            ("min", d.min),
            ("p50", d.p50),
            ("p90", d.p90),
            ("p99", d.p99),
            ("max", d.max),
            ("mean", d.mean),
            ("sum", d.sum),
        ],
        MetricValue::Ratio(r) => vec![
            ("value", r.value),
            ("n", Some(r.n as f64)),
            ("numerator", r.numerator.map(|v| v as f64)),
        ],
        MetricValue::Count(c) => vec![("value", c.value), ("n", Some(c.n as f64))],
    }
}

fn run_index_from_label(label: &str) -> Option<usize> {
    label.strip_prefix("run")?.parse().ok()
}

#[allow(clippy::too_many_arguments)]
fn append_metric_rows(
    out: &mut String,
    scope: &str,
    model_id: &str,
    run_index: Option<usize>,
    document_id: Option<&str>,
    metric: &str,
    value: &MetricValue,
    definitions: &BTreeMap<String, MetricDef>,
) {
    let unit = definitions.get(metric).map_or("", |d| d.unit);
    let n = value.sample_size();
    for (stat, v) in metric_csv_rows(value) {
        out.push_str(&format!(
            "{scope},{},{},{},{},{stat},{},{unit},{n}\n",
            csv_field(model_id),
            run_index.map(|v| v.to_string()).unwrap_or_default(),
            document_id.map(csv_field).unwrap_or_default(),
            csv_field(metric),
            v.map(|x| format!("{x}")).unwrap_or_default(),
        ));
    }
}

/// `measurements.csv` is a value projection of `measurements.json`
/// (ADR 0003 §9.3): every numeric field of every metric becomes one
/// row, in scope order (cell, model, document) and then `BTreeMap`
/// key order within each scope — deterministic byte-for-byte across
/// re-runs of the same results directory. `definitions`, `inputs`,
/// `run_id`, and `generated_at` stay JSON-only.
fn render_csv(measurements: &MeasurementsFile) -> String {
    let mut out = String::from("scope,model_id,run_index,document_id,metric,stat,value,unit,n\n");
    for block in measurements.cells.values() {
        for (metric, value) in &block.metrics {
            append_metric_rows(
                &mut out,
                "cell",
                &block.model_id,
                Some(block.run_index),
                None,
                metric,
                value,
                &measurements.definitions,
            );
        }
    }
    for (model_id, metrics) in &measurements.models {
        for (metric, value) in metrics {
            append_metric_rows(
                &mut out,
                "model",
                model_id,
                None,
                None,
                metric,
                value,
                &measurements.definitions,
            );
        }
    }
    for (model_id, by_document) in &measurements.documents {
        for (document_id, by_run) in by_document {
            for (run_label, metrics) in by_run {
                let run_index = run_index_from_label(run_label);
                for (metric, value) in metrics {
                    append_metric_rows(
                        &mut out,
                        "document",
                        model_id,
                        run_index,
                        Some(document_id.as_str()),
                        metric,
                        value,
                        &measurements.definitions,
                    );
                }
            }
        }
    }
    out
}

// ============================== Writing ==============================

/// Writes `measurements.csv` then `measurements.json`, each via
/// stage-then-rename (`crate::storage`). `measurements.json` carries
/// the version stamp and `generated_at`, so it is the commit a
/// consumer should treat as authoritative; committing the CSV first
/// means a crash between the two renames leaves a stale-but-consistent
/// CSV that the next `compare` run overwrites, rather than a fresh
/// JSON paired with a stale CSV. ADR 0003 §9.3 calls the pair "written
/// atomically together" — two independent renames can only approximate
/// that on a POSIX filesystem, not guarantee it.
fn write_measurements(dir: &Path, measurements: &MeasurementsFile) -> Result<(), String> {
    let json_path = dir.join("measurements.json");
    let csv_path = dir.join("measurements.csv");
    let json_text = serde_json::to_string_pretty(measurements)
        .map_err(|error| format!("serializing measurements.json: {error}"))?;
    let csv_text = render_csv(measurements);

    let staged_csv = crate::storage::stage_bytes(&csv_path, csv_text.as_bytes(), false)
        .map_err(|error| format!("staging {}: {error}", csv_path.display()))?;
    let staged_json = crate::storage::stage_bytes(&json_path, json_text.as_bytes(), false)
        .map_err(|error| format!("staging {}: {error}", json_path.display()))?;
    crate::storage::commit_staged(&staged_csv, &csv_path)
        .map_err(|error| format!("publishing {}: {error}", csv_path.display()))?;
    crate::storage::commit_staged(&staged_json, &json_path)
        .map_err(|error| format!("publishing {}: {error}", json_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests;
