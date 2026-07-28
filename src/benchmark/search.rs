//! `taguru benchmark search` (issue #260, ADR 0003 §11): builds one
//! per-model corpus from a finished `taguru benchmark extract` results
//! directory, runs `eval.jsonl`'s shared question set against each
//! corpus over `POST /contexts/{name}/sources/search`, and writes
//! `RESULTS_DIR/retrieval.json` — the one search endpoint ADR 0003
//! §11 names for this verb; no other retrieval endpoint is called
//! (see the module-level note on `POST /contexts/{name}/recall` below
//! for why one might look tempting and isn't used).
//!
//! Gold-data-free and judgment-free by construction, the same posture
//! [`super::compare`] holds for `measurements.json`: absent
//! user-supplied `expected_sources`/`expected_concepts`, this reports
//! only differences between models' corpora — empty-result rate,
//! hit-set overlap, rank movement, source diversity, per-lane hit
//! counts — and asserts nothing about which model's results are
//! better. `HitLocator.rank` names a hit's own position in one
//! model's result list (mirroring `LaneEvidence::rank`,
//! src/api/sources.rs) — an ordinary IR term, not a cross-model
//! ranking, so it is not treated as a banned word here the way
//! [`super::compare`]'s `rank`/`score`/`winner` key-fragment ban
//! treats it; `winner`/`best`/`recommended`/`overall`/`delta_vs` still
//! are (`search::tests::no_verdict_word_appears_anywhere_in_the_artifact`).
//! When a case DOES carry `expected_sources`/`expected_concepts`,
//! recall@k and MRR are computed from them (ADR 0003 §11) — the one
//! thing this shares with #215's own quality gate, so `recall`/
//! `expected` are legitimate, ADR-mandated vocabulary here even though
//! `differences.jsonl` (#259, ADR §9.4) bans them for itself.
//!
//! Two decisions that are not obvious from the ADR's prose alone,
//! found by reading the server's own implementation while building
//! this:
//!
//! - `POST /contexts/{name}/embeddings/refresh` only re-embeds concept
//!   and label GLOSSES (`registry/embeddings.rs`'s `refresh_embeddings`,
//!   what `resolve`'s semantic fallback reads) — never passage
//!   vectors, which is what `sources/search`'s vector lane actually
//!   reads. Passage embeddings refresh only from the server's own
//!   background flusher tick (`main.rs`'s `run_flush_tick`, gated on
//!   `auto_embed` and `TAGURU_EMBED_PASSAGES`) and have no HTTP
//!   endpoint at all. So this module never calls embeddings/refresh —
//!   there is nothing useful it would do — and instead reads
//!   `GET /contexts/{name}/embeddings` once per corpus, purely as
//!   diagnostic evidence for [`CorpusBlock::passage_vectors`], for why
//!   the vector lane may answer `ran: false`.
//! - `Context::recall(cue)` (`src/context/query.rs`) requires `cue` to
//!   equal a stored concept or label id exactly — no fuzzy or semantic
//!   resolution happens inside it (`tests/common/recall.rs`'s own
//!   harness resolves a cue via `resolve_label` *before* calling
//!   `recall`, for exactly this reason). A hand-written `eval.jsonl`'s
//!   `cues[]` are natural-language strings, not stored ids, so driving
//!   `expected_concepts` through `POST /contexts/{name}/recall` would
//!   silently miss almost everything. `expected_concepts` is instead
//!   folded into the same recall@k/MRR computation `expected_sources`
//!   feeds, checked against the very `sources/search` hits already
//!   fetched for the case (a case/NFKC-folded substring test against
//!   each hit's own `text`) — one search call per case per model,
//!   never two.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::Value;

use crate::api::MAX_CONTEXT_NAME_BYTES;
use crate::api::sources::{PassageHit, PassageLanes, PassagePage, SearchContextPlan};
use crate::cli::default_base_url;
use crate::config::{load_config, subcommand_usage_error};
use crate::evalset::{self, EvalCase, ExpectedSource};
use crate::measure::{Distribution, MetricDef, MetricValue, MetricsMap, def, ratio_metric};
use crate::remote::{Api, ApiFailure};

use super::identity;
use super::{BenchManifest, DocumentInfo, ManifestModel, load_bench_manifest};

const BENCHMARK_RETRIEVAL_VERSION: u64 = 1;
const DEFAULT_LIMIT: usize = 10;
/// Mirrors the server's own `MAX_MATCH_LIMIT` (src/api.rs) — a
/// `--limit`/`options.limit` above this could never be honored anyway.
const MAX_SEARCH_LIMIT: usize = 1000;

const SEARCH_USAGE: &str = "\
usage: taguru benchmark search --eval FILE [--url URL] [--config FILE]
                      [--run N] [--context-prefix NAME] [--skip-import]
                      RESULTS_DIR

Builds one per-model corpus (a context named PREFIX::MODEL_ID) from a
finished `taguru benchmark extract` results directory's batch files,
runs eval.jsonl's shared question set against each corpus over
POST /contexts/{name}/sources/search (ADR 0003 §11), and writes
RESULTS_DIR/retrieval.json: per-case/per-model hit counts, lane
evidence, and model-pair hit-set overlap — gold-data-free and
judgment-free by construction. When a case carries expected_sources or
expected_concepts, recall@k and MRR are also computed against them,
the one thing this shares with #215's own quality gate.

  RESULTS_DIR            a directory `taguru benchmark extract` wrote
  --eval FILE            eval.jsonl (ADR 0003 §11's shared dataset)
  --url URL              the server to import into and search; default
                        resolves the same way `taguru health` does
                        (TAGURU_ADDR, or --config/TAGURU_CONFIG)
  --config FILE          load before resolving --url (same as --config
                        everywhere else)
  --run N                which extract run to import from, 1-based (1)
  --context-prefix NAME  corpus context names are NAME::MODEL_ID
                        (default: the extraction context
                        RESULTS_DIR's manifest.json recorded)
  --skip-import          search existing PREFIX::MODEL_ID contexts
                        instead of importing fresh ones

Contract and discipline: docs/benchmark.html,
adr/0003-extraction-model-benchmark.md.
";

pub(super) fn run_search(args: &[String]) -> i32 {
    if matches!(
        args.first().map(String::as_str),
        Some("--help") | Some("-h")
    ) {
        print!("{SEARCH_USAGE}");
        return 0;
    }
    let search_args = match parse_args(args) {
        Ok(search_args) => search_args,
        Err(code) => return code,
    };

    let loaded = match evalset::load_eval_file(
        &search_args.eval,
        evalset::Extensions::CarryThrough {
            consumer: "taguru benchmark search",
        },
    ) {
        Ok(loaded) => loaded,
        Err(message) => {
            eprintln!("taguru: benchmark: search: {message}");
            return 2;
        }
    };
    for warning in &loaded.warnings {
        eprintln!("taguru: benchmark: search: {warning}");
    }
    if let Err(message) = validate_limits(&loaded.cases) {
        eprintln!("taguru: benchmark: search: {message}");
        return 2;
    }

    let manifest = match load_bench_manifest(&search_args.dir.join("manifest.json")) {
        Ok(manifest) => manifest,
        Err(message) => {
            eprintln!("taguru: benchmark: search: {message}");
            return 2;
        }
    };
    if manifest.models.is_empty() {
        eprintln!("taguru: benchmark: search: manifest.json lists no models");
        return 2;
    }
    if !search_args.skip_import
        && !manifest
            .cells
            .iter()
            .any(|c| c.run_index == search_args.run)
    {
        let available: BTreeSet<usize> = manifest.cells.iter().map(|c| c.run_index).collect();
        eprintln!(
            "taguru: benchmark: search: no cell recorded for run {} — available runs: {:?}",
            search_args.run, available
        );
        return 2;
    }

    let prefix = search_args
        .context_prefix
        .clone()
        .unwrap_or_else(|| manifest.extraction_settings.context.clone());
    let mut context_names: BTreeMap<String, String> = BTreeMap::new();
    let mut oversized = Vec::new();
    for model in &manifest.models {
        let name = corpus_context_name(&prefix, &model.model_id);
        if name.len() > MAX_CONTEXT_NAME_BYTES {
            oversized.push(format!("'{name}' ({} bytes)", name.len()));
        }
        context_names.insert(model.model_id.clone(), name);
    }
    if !oversized.is_empty() {
        eprintln!(
            "taguru: benchmark: search: corpus context name(s) exceed {MAX_CONTEXT_NAME_BYTES} \
             bytes — pick a shorter --context-prefix: {}",
            oversized.join(", ")
        );
        return 2;
    }

    let config = search_args
        .config
        .clone()
        .or_else(|| std::env::var("TAGURU_CONFIG").ok().map(PathBuf::from));
    if let Some(path) = &config {
        load_config(path);
    }
    let base = match search_args.url.clone() {
        Some(url) => url,
        None => match default_base_url() {
            Ok(url) => url,
            Err(error) => {
                eprintln!("taguru: benchmark: search: {error}");
                return 2;
            }
        },
    };
    let api = Api::new(base.clone());

    // ADR 0003 §11's own degradation only covers a response that
    // omits `plan`/`lanes` (an older server) or a single case/model's
    // HTTP failure — never a server this run cannot reach at all, and
    // #260's only retrieval path is HTTP (see this module's own doc
    // comment), so nothing downstream could produce a meaningful
    // artifact without one. Fail loudly now instead of writing an
    // empty retrieval.json case by case.
    if let Err(error) = api.get_raw(&["health"]) {
        eprintln!("taguru: benchmark: search: server at {base} is not reachable: {error}");
        return 1;
    }

    let matching = identity::Matching::default();

    let mut corpus_blocks: BTreeMap<String, CorpusBlock> = BTreeMap::new();
    let mut availability: BTreeMap<String, (String, bool)> = BTreeMap::new();
    for model in &manifest.models {
        let context = context_names
            .get(&model.model_id)
            .expect("computed for every model above")
            .clone();
        let block = if search_args.skip_import {
            search_only_probe(&api, &context)
        } else {
            build_corpus(
                &api,
                &search_args.dir,
                &manifest,
                search_args.run,
                model,
                &context,
            )
        };
        let available = block.outcome == "built" || block.outcome == "search_only";
        availability.insert(model.model_id.clone(), (context, available));
        corpus_blocks.insert(model.model_id.clone(), block);
    }

    if availability.values().all(|(_, available)| !available) {
        eprintln!(
            "taguru: benchmark: search: no model's corpus is available to search — see \
             retrieval.json's corpus block would have explained why, but nothing is written \
             when every model fails"
        );
        return 1;
    }

    let context = SearchContext {
        api: &api,
        matching: &matching,
        documents: &manifest.documents,
        availability: &availability,
        default_limit: DEFAULT_LIMIT,
    };
    let mut cases = Vec::with_capacity(loaded.cases.len());
    let mut warnings = loaded.warnings.clone();
    for case in &loaded.cases {
        let (block, mut case_warnings) = build_case_block(&context, case);
        warnings.append(&mut case_warnings);
        cases.push(block);
    }

    let (models_agg, pairs_agg) = aggregate(&cases);

    let retrieval = RetrievalFile {
        taguru_benchmark_retrieval: BENCHMARK_RETRIEVAL_VERSION,
        run_id: manifest.run_id.clone(),
        generated_at: crate::clock::iso8601_utc(crate::clock::now_unix_secs()),
        matching,
        inputs: InputsBlock {
            results_dir: search_args.dir.display().to_string(),
            eval: EvalInputsBlock {
                path: search_args.eval.display().to_string(),
                name: loaded.name.clone(),
                cases: loaded.cases.len(),
            },
            url: base,
            run_index: search_args.run,
            default_limit: DEFAULT_LIMIT,
        },
        definitions: build_definitions(),
        warnings,
        corpus: corpus_blocks,
        cases,
        models: models_agg,
        pairs: pairs_agg,
    };

    if let Err(error) = write_retrieval(&search_args.dir, &retrieval) {
        eprintln!("taguru: benchmark: search: {error}");
        return 1;
    }
    0
}

// ============================== Arguments ==============================

#[derive(Debug)]
struct SearchArgs {
    eval: PathBuf,
    dir: PathBuf,
    url: Option<String>,
    config: Option<PathBuf>,
    run: usize,
    context_prefix: Option<String>,
    skip_import: bool,
}

fn parse_args(args: &[String]) -> Result<SearchArgs, i32> {
    let usage = |message: &str| subcommand_usage_error("benchmark", message);
    let mut eval: Option<PathBuf> = None;
    let mut url: Option<String> = None;
    let mut config: Option<PathBuf> = None;
    let mut run: usize = 1;
    let mut context_prefix: Option<String> = None;
    let mut skip_import = false;
    let mut positional: Option<&str> = None;

    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{SEARCH_USAGE}");
                return Err(0);
            }
            "--eval" => match rest.next() {
                Some(path) if eval.is_none() => eval = Some(PathBuf::from(path)),
                Some(_) => return Err(usage("--eval given twice")),
                None => return Err(usage("--eval needs a file path")),
            },
            "--url" => match rest.next() {
                Some(value) if url.is_none() => url = Some(value.trim_end_matches('/').to_string()),
                Some(_) => return Err(usage("--url given twice")),
                None => return Err(usage("--url needs a value")),
            },
            "--config" => match rest.next() {
                Some(path) if config.is_none() => config = Some(PathBuf::from(path)),
                Some(_) => return Err(usage("--config given twice")),
                None => return Err(usage("--config needs a file path")),
            },
            "--run" => match rest.next().and_then(|value| value.parse::<usize>().ok()) {
                Some(n) if n >= 1 => run = n,
                _ => return Err(usage("--run needs a positive integer")),
            },
            "--context-prefix" => match rest.next() {
                Some(name) if context_prefix.is_none() => context_prefix = Some(name.clone()),
                Some(_) => return Err(usage("--context-prefix given twice")),
                None => return Err(usage("--context-prefix needs a name")),
            },
            "--skip-import" => skip_import = true,
            flag if flag.starts_with("--") => {
                return Err(usage(&format!("unknown flag '{flag}' for search")));
            }
            value => {
                if positional.is_some() {
                    return Err(usage("search takes exactly one RESULTS_DIR argument"));
                }
                positional = Some(value);
            }
        }
    }

    let eval = eval.ok_or_else(|| usage("--eval FILE is required"))?;
    let dir = positional
        .map(PathBuf::from)
        .ok_or_else(|| usage("search requires a RESULTS_DIR argument"))?;

    Ok(SearchArgs {
        eval,
        dir,
        url,
        config,
        run,
        context_prefix,
        skip_import,
    })
}

fn validate_limits(cases: &[EvalCase]) -> Result<(), String> {
    for case in cases {
        if let Some(limit) = case.options.limit
            && !(1..=MAX_SEARCH_LIMIT).contains(&limit)
        {
            return Err(format!(
                "case '{}': options.limit must be 1..={MAX_SEARCH_LIMIT}, got {limit}",
                case.case_id
            ));
        }
    }
    Ok(())
}

fn corpus_context_name(prefix: &str, model_id: &str) -> String {
    format!("{prefix}::{model_id}")
}

// ============================== Corpus construction ==============================

const OWNER_DESCRIPTION_PREFIX: &str = "taguru benchmark search corpus";

/// `run_id` alone is not enough: it names the whole `taguru benchmark
/// extract` invocation, not one `--run N` within it, so a marker
/// without `run_index` would pass the ownership check unchanged
/// across `--run 1` then `--run 2` against the same results
/// directory — silently mixing two runs' documents into one corpus,
/// since import never removes a source absent from the current
/// import's batch set.
fn ownership_marker(run_id: &str, model_id: &str, run_index: usize) -> String {
    format!("{OWNER_DESCRIPTION_PREFIX}: run {run_id}, model {model_id}, run_index {run_index}")
}

fn corpus_block(context: &str, outcome: &str, reason: Option<String>) -> CorpusBlock {
    CorpusBlock {
        context: context.to_string(),
        outcome: outcome.to_string(),
        reason,
        documents_imported: 0,
        documents_failed: 0,
        passage_vectors: None,
    }
}

/// Builds one model's corpus: an ownership check against whatever
/// already lives at `context` (never overwrite a context this run
/// didn't create), then one `POST /import` per batch file in the
/// requested run's cell directory, each with its header's `context`
/// rewritten and its `create.description` stamped with this run's
/// ownership marker — every other line byte-for-byte unchanged.
fn build_corpus(
    api: &Api,
    dir: &Path,
    manifest: &BenchManifest,
    run_index: usize,
    model: &ManifestModel,
    context: &str,
) -> CorpusBlock {
    let marker = ownership_marker(&manifest.run_id, &model.model_id, run_index);
    match api.get_envelope(&["contexts", context]) {
        Ok(entry) => {
            let existing = entry
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("");
            if existing != marker {
                return corpus_block(
                    context,
                    "rejected",
                    Some(format!(
                        "context '{context}' already exists with a description that does not \
                         match this run's ownership marker — refusing to import into it (pass \
                         --context-prefix to pick a different name)"
                    )),
                );
            }
        }
        Err(ApiFailure::NotFound(_)) => {
            // Fine — the first batch's `create` block below makes it.
        }
        Err(ApiFailure::Other(message)) => {
            return corpus_block(context, "failed", Some(message));
        }
    }

    let Some(cell) = manifest
        .cells
        .iter()
        .find(|cell| cell.model_id == model.model_id && cell.run_index == run_index)
    else {
        return corpus_block(
            context,
            "skipped",
            Some(format!(
                "no cell recorded for model '{}' at run {run_index}",
                model.model_id
            )),
        );
    };
    if cell.outcome != "complete" {
        return corpus_block(
            context,
            "skipped",
            Some(format!(
                "cell outcome is '{}'; only complete cells are imported",
                cell.outcome
            )),
        );
    }

    let cell_dir = dir.join(&cell.cell_dir);
    let batch_files = match list_batch_files(&cell_dir) {
        Ok(files) => files,
        Err(message) => return corpus_block(context, "failed", Some(message)),
    };
    if batch_files.is_empty() {
        return corpus_block(
            context,
            "skipped",
            Some(format!("no batch files in {}", cell_dir.display())),
        );
    }

    let mut imported = 0usize;
    let mut failed = 0usize;
    let mut failures = Vec::new();
    for file in &batch_files {
        match rewrite_and_import(api, file, context, &marker) {
            Ok(()) => imported += 1,
            Err(message) => {
                failed += 1;
                failures.push(message);
            }
        }
    }

    CorpusBlock {
        context: context.to_string(),
        outcome: if imported == 0 { "failed" } else { "built" }.to_string(),
        reason: (!failures.is_empty()).then(|| failures.join("; ")),
        documents_imported: imported,
        documents_failed: failed,
        passage_vectors: fetch_passage_vectors(api, context),
    }
}

fn search_only_probe(api: &Api, context: &str) -> CorpusBlock {
    match api.get_envelope(&["contexts", context]) {
        Ok(_) => CorpusBlock {
            context: context.to_string(),
            outcome: "search_only".to_string(),
            reason: None,
            documents_imported: 0,
            documents_failed: 0,
            passage_vectors: fetch_passage_vectors(api, context),
        },
        Err(ApiFailure::NotFound(_)) => corpus_block(
            context,
            "failed",
            Some("--skip-import given but this context does not exist".to_string()),
        ),
        Err(ApiFailure::Other(message)) => corpus_block(context, "failed", Some(message)),
    }
}

/// Best-effort, read-only: reports whether passage vectors exist for
/// `context` today, purely as evidence for why the vector lane may or
/// may not run — never a trigger. See this module's own doc comment
/// for why nothing here calls an embeddings/refresh endpoint.
fn fetch_passage_vectors(api: &Api, context: &str) -> Option<PassageVectorInfo> {
    let value = api
        .get_envelope(&["contexts", context, "embeddings"])
        .ok()?;
    let passages = value.get("passages")?;
    if passages.is_null() {
        return None;
    }
    Some(PassageVectorInfo {
        model: passages
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string),
        rows: passages.get("rows").and_then(Value::as_u64).unwrap_or(0) as usize,
    })
}

/// Every `*.jsonl` batch file in a cell directory, `diagnostics.jsonl`
/// excluded — the same layout `taguru benchmark extract` documents
/// (`src/benchmark.rs`'s own USAGE text: batch files, diagnostics.jsonl,
/// stdout.log, stderr.log, exit_code, sit side by side).
fn list_batch_files(cell_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = fs::read_dir(cell_dir)
        .map_err(|error| format!("reading {}: {error}", cell_dir.display()))?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("reading {}: {error}", cell_dir.display()))?;
        let path = entry.path();
        let is_batch = path.extension().is_some_and(|ext| ext == "jsonl")
            && path
                .file_name()
                .is_some_and(|name| name != "diagnostics.jsonl");
        if is_batch {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

/// Rewrites one batch file's header line — `context` replaced,
/// `create.description` stamped with this run's ownership marker,
/// unconditionally (so a first import creates the context under it,
/// and a resumed run's re-import still carries it) — and sends the
/// result as one `POST /import` request. Every other line rides
/// through byte-for-byte.
fn rewrite_and_import(api: &Api, path: &Path, context: &str, marker: &str) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let rewritten = rewrite_batch_header(&text, context, marker)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    api.import_chunk(&rewritten, false)
        .map(|_| ())
        .map_err(|failure| format!("{}: {}", path.display(), failure.into_message()))
}

/// The pure half of [`rewrite_and_import`]: parses line 1 as the batch
/// header, replaces `context`, and stamps `create.description` with
/// `marker` — unconditionally, overwriting whatever `create` block (if
/// any) the extract cell wrote, since #260 always owns the corpus it
/// imports into. Every following line is copied verbatim (blank lines
/// dropped), never re-parsed, so a fact line's own formatting can never
/// drift from what `taguru extract` wrote.
fn rewrite_batch_header(text: &str, context: &str, marker: &str) -> Result<String, String> {
    let mut lines = text.lines();
    let Some(header_line) = lines.next() else {
        return Err("empty batch file".to_string());
    };
    let mut header: Value = serde_json::from_str(header_line)
        .map_err(|error| format!("header is not JSON: {error}"))?;
    header["context"] = Value::String(context.to_string());
    header["create"] = serde_json::json!({ "description": marker });

    let mut rewritten = header.to_string();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        rewritten.push('\n');
        rewritten.push_str(line);
    }
    rewritten.push('\n');
    Ok(rewritten)
}

// ============================== Search execution ==============================

/// Bundles the read-only handles [`build_case_block`] needs so its own
/// signature stays short — none of these change per case.
struct SearchContext<'a> {
    api: &'a Api,
    matching: &'a identity::Matching,
    documents: &'a [DocumentInfo],
    /// model_id -> (context, available). `available` is false when
    /// the model's corpus was never built or rejected — such a model
    /// is recorded as a per-case failure without an HTTP call.
    availability: &'a BTreeMap<String, (String, bool)>,
    default_limit: usize,
}

fn build_case_block(context: &SearchContext, case: &EvalCase) -> (CaseBlock, Vec<String>) {
    let limit = case.options.limit.unwrap_or(context.default_limit);
    let mut warnings = Vec::new();
    let mut models: BTreeMap<String, SearchOutcome> = BTreeMap::new();
    let mut hit_locators: BTreeMap<String, Vec<HitLocator>> = BTreeMap::new();

    // Resolved once per case — never once per model — so an
    // unresolvable or ambiguous expected_sources entry warns exactly
    // once no matter how many models this case searches.
    let expected_items = case.has_expectations().then(|| {
        let (items, mut resolve_warnings) = resolve_expected_items(
            context.matching,
            &case.case_id,
            &case.expected_sources,
            &case.expected_concepts,
            context.documents,
        );
        warnings.append(&mut resolve_warnings);
        items
    });

    for (model_id, (model_context, available)) in context.availability {
        if !available {
            models.insert(
                model_id.clone(),
                SearchOutcome::Failed {
                    message: "corpus not available for this model — see the corpus block"
                        .to_string(),
                },
            );
            continue;
        }
        match search_case_model(context.api, model_context, &case.query, limit) {
            Ok((mut hits, plan)) => {
                hits.truncate(limit);
                let locators: Vec<HitLocator> = hits
                    .iter()
                    .enumerate()
                    .map(|(index, hit)| HitLocator {
                        rank: index + 1,
                        source: hit.source.clone(),
                        paragraph: hit.paragraph,
                    })
                    .collect();
                let lanes = lane_hit_counts(&hits, plan.is_some());
                let recall = expected_items.as_ref().and_then(|items| {
                    let result = score_recall(context.matching, items, &hits);
                    (result.expected_total > 0).then_some(result)
                });
                let distinct_sources = hits
                    .iter()
                    .map(|hit| hit.source.as_str())
                    .collect::<BTreeSet<_>>()
                    .len();
                hit_locators.insert(model_id.clone(), locators.clone());
                models.insert(
                    model_id.clone(),
                    SearchOutcome::Searched {
                        hit_count: hits.len(),
                        empty: hits.is_empty(),
                        distinct_sources,
                        lanes,
                        plan: lanes_echo(plan.as_ref()),
                        hits: locators,
                        recall,
                    },
                );
            }
            Err(message) => {
                models.insert(model_id.clone(), SearchOutcome::Failed { message });
            }
        }
    }

    let mut pairs: BTreeMap<String, PairOutcome> = BTreeMap::new();
    let model_ids: Vec<&String> = context.availability.keys().collect();
    for i in 0..model_ids.len() {
        for j in (i + 1)..model_ids.len() {
            let (a, b) = (model_ids[i], model_ids[j]);
            let key = pair_key(a, b);
            let outcome = match (hit_locators.get(a), hit_locators.get(b)) {
                (Some(hits_a), Some(hits_b)) => {
                    let (jaccard, shared_hits, mean_rank_difference) = pair_overlap(hits_a, hits_b);
                    PairOutcome::Compared {
                        jaccard,
                        shared_hits,
                        mean_rank_difference,
                    }
                }
                _ => PairOutcome::Unavailable {
                    message: "one or both models were not successfully searched for this case"
                        .to_string(),
                },
            };
            pairs.insert(key, outcome);
        }
    }

    (
        CaseBlock {
            case_id: case.case_id.clone(),
            query: case.query.clone(),
            cues: case.cues.clone(),
            limit,
            has_expectations: case.has_expectations(),
            models,
            pairs,
        },
        warnings,
    )
}

fn pair_key(a: &str, b: &str) -> String {
    if a <= b {
        format!("{a}__{b}")
    } else {
        format!("{b}__{a}")
    }
}

/// One case's search against one model's corpus: the hits (already
/// deserialized into the real wire type when the server's response
/// shape allows it) and, when available, that context's own entry
/// from the response's `plan`.
fn search_case_model(
    api: &Api,
    context: &str,
    query: &str,
    limit: usize,
) -> Result<(Vec<PassageHit>, Option<SearchContextPlan>), String> {
    let body = serde_json::json!({ "query": query, "limit": limit });
    let value = api.post(&["contexts", context, "sources", "search"], &body)?;
    extract_hits(&value)
}

/// Prefers the real `{plan, hits}` shape; falls back to pulling
/// `source`/`paragraph`/`score`/`text` out of either a bare hits array
/// (the pre-#151 wire shape `PassagePage`'s own doc comment
/// describes) or an object whose `hits` array doesn't fit
/// `PassageHit`'s current (lane-carrying) shape — an older or
/// otherwise-nonconforming server, still enough evidence for every
/// metric except per-hit lane evidence, which is `None` on this path.
fn extract_hits(value: &Value) -> Result<(Vec<PassageHit>, Option<SearchContextPlan>), String> {
    if let Ok(page) = serde_json::from_value::<PassagePage>(value.clone()) {
        let plan = page.plan.contexts.into_iter().next();
        return Ok((page.hits, plan));
    }
    let raw_hits: &Vec<Value> = if let Some(array) = value.as_array() {
        array
    } else if let Some(array) = value.get("hits").and_then(Value::as_array) {
        array
    } else {
        return Err("response carries no recognizable hits (plan/hits shape mismatch)".to_string());
    };
    let mut hits = Vec::with_capacity(raw_hits.len());
    for raw in raw_hits {
        let source = raw
            .get("source")
            .and_then(Value::as_str)
            .ok_or("a hit is missing 'source'")?
            .to_string();
        let paragraph = raw
            .get("paragraph")
            .and_then(Value::as_u64)
            .ok_or("a hit is missing 'paragraph'")? as u32;
        let score = raw.get("score").and_then(Value::as_f64).unwrap_or(0.0) as f32;
        let text = raw
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        hits.push(PassageHit {
            source,
            paragraph,
            score,
            text,
            lanes: PassageLanes {
                bm25: None,
                vector: None,
            },
        });
    }
    Ok((hits, None))
}

fn lanes_echo(plan: Option<&SearchContextPlan>) -> Option<LanesEcho> {
    plan.map(|plan| LanesEcho {
        bm25: LaneEcho {
            ran: plan.lanes.bm25.ran,
            reason: plan.lanes.bm25.reason.clone(),
            floor: plan.lanes.bm25.floor,
        },
        vector: LaneEcho {
            ran: plan.lanes.vector.ran,
            reason: plan.lanes.vector.reason.clone(),
            floor: plan.lanes.vector.floor,
        },
    })
}

fn lane_hit_counts(hits: &[PassageHit], lanes_known: bool) -> LaneHitCounts {
    if !lanes_known {
        return LaneHitCounts {
            unknown: hits.len(),
            ..LaneHitCounts::default()
        };
    }
    let mut counts = LaneHitCounts::default();
    for hit in hits {
        match (hit.lanes.bm25.is_some(), hit.lanes.vector.is_some()) {
            (true, true) => counts.both += 1,
            (true, false) => counts.bm25_only += 1,
            (false, true) => counts.vector_only += 1,
            (false, false) => counts.neither += 1,
        }
    }
    counts
}

// ============================== Recall@k / MRR ==============================

/// One case's expected evidence, source and concept entries folded
/// into a single list — ADR 0003 §11 computes recall@k/MRR "from the
/// same fields" for both, not as two separate scores.
enum ExpectedItem {
    Source { path: String, paragraphs: Vec<u32> },
    Concept { normalized: String },
}

impl ExpectedItem {
    fn matches(&self, hit: &PassageHit, normalized_hit_text: &str) -> bool {
        match self {
            ExpectedItem::Source { path, paragraphs } => {
                &hit.source == path
                    && (paragraphs.is_empty() || paragraphs.contains(&hit.paragraph))
            }
            ExpectedItem::Concept { normalized } => {
                normalized_hit_text.contains(normalized.as_str())
            }
        }
    }
}

/// Resolves one `expected_sources[].source` string against the
/// manifest's document dictionary: (a) an exact `path` match, (b) a
/// `document_id` match, (c) a unique path-suffix match. A hit's own
/// `source` is always exactly a manifest `path` (the corpus rewrite
/// leaves `source` untouched — only `context` changes), so (a) is the
/// common case; (b)/(c) exist for a hand-written `eval.jsonl` that
/// spells a source more loosely. Unresolved or ambiguous entries are
/// compared against the literal string given — which will simply never
/// match a hit — rather than dropped, so they stay in recall@k's
/// denominator per ADR 0003 §11's "the eval.jsonl decides the
/// denominator" posture; the caller is warned once so the miss is not
/// silent.
fn resolve_expected_source_path(
    expected: &str,
    documents: &[DocumentInfo],
) -> (String, Option<String>) {
    if documents.iter().any(|doc| doc.path == expected) {
        return (expected.to_string(), None);
    }
    if let Some(doc) = documents.iter().find(|doc| doc.document_id == expected) {
        return (doc.path.clone(), None);
    }
    let matches: Vec<&str> = documents
        .iter()
        .filter(|doc| doc.path.ends_with(expected))
        .map(|doc| doc.path.as_str())
        .collect();
    match matches.as_slice() {
        [only] => (only.to_string(), None),
        [] => (
            expected.to_string(),
            Some(format!(
                "expected_sources source '{expected}' matches no document path, document_id, \
                 or unique path suffix in this results directory's manifest — it can never \
                 match a hit"
            )),
        ),
        several => (
            expected.to_string(),
            Some(format!(
                "expected_sources source '{expected}' matches {} document paths by suffix — \
                 ambiguous, so it is compared literally and will likely never match a hit",
                several.len()
            )),
        ),
    }
}

/// Resolves a case's `expected_sources`/`expected_concepts` into one
/// item list, with the resolution warnings — depends only on the
/// case and the results directory's document dictionary, never on any
/// model's hits, so a caller searching several models for one case
/// must call this exactly once and reuse the result (matching this
/// module's own "warn once per case, not once per model" posture) —
/// see [`score_recall`], the per-model half that does depend on hits.
fn resolve_expected_items(
    matching: &identity::Matching,
    case_id: &str,
    expected_sources: &[ExpectedSource],
    expected_concepts: &[String],
    documents: &[DocumentInfo],
) -> (Vec<ExpectedItem>, Vec<String>) {
    let mut warnings = Vec::new();
    let mut items = Vec::new();
    for expected in expected_sources {
        if expected.relevance == 0 {
            continue;
        }
        let (path, warning) = resolve_expected_source_path(&expected.source, documents);
        if let Some(message) = warning {
            warnings.push(format!("case '{case_id}': {message}"));
        }
        items.push(ExpectedItem::Source {
            path,
            paragraphs: expected.paragraphs.clone(),
        });
    }
    for concept in expected_concepts {
        items.push(ExpectedItem::Concept {
            normalized: identity::normalize_term(matching, concept),
        });
    }
    (items, warnings)
}

/// Scores one model's hits against a case's already-resolved expected
/// items (see [`resolve_expected_items`]) — generates no warnings of
/// its own, so it is safe to call once per model without duplicating
/// the resolution warnings that many models would otherwise repeat.
fn score_recall(
    matching: &identity::Matching,
    items: &[ExpectedItem],
    hits: &[PassageHit],
) -> RecallResult {
    let normalized_texts: Vec<String> = hits
        .iter()
        .map(|hit| identity::normalize_term(matching, &hit.text))
        .collect();

    let matched = items
        .iter()
        .filter(|item| {
            hits.iter()
                .zip(&normalized_texts)
                .any(|(hit, text)| item.matches(hit, text))
        })
        .count();

    let mut mrr = 0.0;
    for (index, (hit, text)) in hits.iter().zip(&normalized_texts).enumerate() {
        if items.iter().any(|item| item.matches(hit, text)) {
            mrr = 1.0 / (index as f64 + 1.0);
            break;
        }
    }

    let expected_total = items.len();
    let recall_at_k = if expected_total == 0 {
        0.0
    } else {
        matched as f64 / expected_total as f64
    };
    RecallResult {
        recall_at_k,
        mrr,
        expected_total,
        matched,
    }
}

/// [`resolve_expected_items`] then [`score_recall`] in one call — a
/// test convenience for exercising one model's recall against a case
/// in a single call. No production caller has this shape:
/// [`build_case_block`] always calls the two halves separately,
/// resolving once per case and scoring once per model, to avoid
/// exactly the duplicate-warning bug a combined call would invite for
/// more than one model.
#[cfg(test)]
fn compute_recall(
    matching: &identity::Matching,
    case_id: &str,
    expected_sources: &[ExpectedSource],
    expected_concepts: &[String],
    documents: &[DocumentInfo],
    hits: &[PassageHit],
) -> (RecallResult, Vec<String>) {
    let (items, warnings) = resolve_expected_items(
        matching,
        case_id,
        expected_sources,
        expected_concepts,
        documents,
    );
    let result = score_recall(matching, &items, hits);
    (result, warnings)
}

/// `(source, paragraph)` overlap between two models' hit lists for one
/// case: Jaccard (`None` when both sides are empty, ADR 0003 §9.3's
/// "measured, zero samples is not a value" posture — see
/// [`identity::jaccard`]), how many locators both sides found, and the
/// mean absolute rank difference among those shared locators.
fn pair_overlap(a: &[HitLocator], b: &[HitLocator]) -> (Option<f64>, usize, Option<f64>) {
    let set_a: BTreeSet<(String, u32)> = a
        .iter()
        .map(|hit| (hit.source.clone(), hit.paragraph))
        .collect();
    let set_b: BTreeSet<(String, u32)> = b
        .iter()
        .map(|hit| (hit.source.clone(), hit.paragraph))
        .collect();
    let jaccard = identity::jaccard(&set_a, &set_b);

    let rank_a: BTreeMap<(String, u32), usize> = a
        .iter()
        .map(|hit| ((hit.source.clone(), hit.paragraph), hit.rank))
        .collect();
    let rank_b: BTreeMap<(String, u32), usize> = b
        .iter()
        .map(|hit| ((hit.source.clone(), hit.paragraph), hit.rank))
        .collect();
    let diffs: Vec<f64> = set_a
        .intersection(&set_b)
        .filter_map(|key| match (rank_a.get(key), rank_b.get(key)) {
            (Some(&ra), Some(&rb)) => Some((ra as i64 - rb as i64).unsigned_abs() as f64),
            _ => None,
        })
        .collect();
    let shared_hits = diffs.len();
    let mean_rank_difference =
        (!diffs.is_empty()).then(|| diffs.iter().sum::<f64>() / diffs.len() as f64);
    (jaccard, shared_hits, mean_rank_difference)
}

// ============================== Aggregation ==============================

fn aggregate(cases: &[CaseBlock]) -> (BTreeMap<String, MetricsMap>, BTreeMap<String, MetricsMap>) {
    let mut model_ids: BTreeSet<String> = BTreeSet::new();
    let mut pair_keys: BTreeSet<String> = BTreeSet::new();
    for case in cases {
        model_ids.extend(case.models.keys().cloned());
        pair_keys.extend(case.pairs.keys().cloned());
    }

    let mut models = BTreeMap::new();
    for model_id in &model_ids {
        models.insert(model_id.clone(), aggregate_model(cases, model_id));
    }

    let mut pairs = BTreeMap::new();
    for key in &pair_keys {
        pairs.insert(key.clone(), aggregate_pair(cases, key));
    }

    (models, pairs)
}

fn aggregate_model(cases: &[CaseBlock], model_id: &str) -> MetricsMap {
    let mut hit_counts = Vec::new();
    let mut empties = 0u64;
    let mut searched = 0u64;
    let mut errors = 0u64;
    let mut total = 0u64;
    let mut distinct_sources = Vec::new();
    let mut bm25_only = Vec::new();
    let mut vector_only = Vec::new();
    let mut both = Vec::new();
    let mut neither = Vec::new();
    let mut unknown = Vec::new();
    let mut recall_values = Vec::new();
    let mut mrr_values = Vec::new();

    for case in cases {
        let Some(outcome) = case.models.get(model_id) else {
            continue;
        };
        total += 1;
        match outcome {
            SearchOutcome::Searched {
                hit_count,
                empty,
                distinct_sources: sources,
                lanes,
                recall,
                ..
            } => {
                searched += 1;
                hit_counts.push(*hit_count as f64);
                if *empty {
                    empties += 1;
                }
                distinct_sources.push(*sources as f64);
                // `neither`/`unknown` get their own samples too, not
                // just `bm25_only`/`vector_only`/`both` — a case whose
                // lanes could not be recovered (the legacy fallback in
                // `extract_hits`) pushes 0 into every one of the first
                // three, which would otherwise be indistinguishable
                // from a lane that genuinely evidenced nothing.
                bm25_only.push(lanes.bm25_only as f64);
                vector_only.push(lanes.vector_only as f64);
                both.push(lanes.both as f64);
                neither.push(lanes.neither as f64);
                unknown.push(lanes.unknown as f64);
                if let Some(result) = recall {
                    recall_values.push(result.recall_at_k);
                    mrr_values.push(result.mrr);
                }
            }
            SearchOutcome::Failed { .. } => errors += 1,
        }
    }

    let mut metrics: MetricsMap = BTreeMap::new();
    metrics.insert(
        "hits.count".to_string(),
        MetricValue::Distribution(Distribution::from_samples(hit_counts)),
    );
    metrics.insert(
        "hits.empty_rate".to_string(),
        MetricValue::Ratio(ratio_metric(empties, searched)),
    );
    metrics.insert(
        "hits.distinct_sources".to_string(),
        MetricValue::Distribution(Distribution::from_samples(distinct_sources)),
    );
    metrics.insert(
        "lanes.bm25_only".to_string(),
        MetricValue::Distribution(Distribution::from_samples(bm25_only)),
    );
    metrics.insert(
        "lanes.vector_only".to_string(),
        MetricValue::Distribution(Distribution::from_samples(vector_only)),
    );
    metrics.insert(
        "lanes.both".to_string(),
        MetricValue::Distribution(Distribution::from_samples(both)),
    );
    metrics.insert(
        "lanes.neither".to_string(),
        MetricValue::Distribution(Distribution::from_samples(neither)),
    );
    metrics.insert(
        "lanes.unknown".to_string(),
        MetricValue::Distribution(Distribution::from_samples(unknown)),
    );
    metrics.insert(
        "cases.error_rate".to_string(),
        MetricValue::Ratio(ratio_metric(errors, total)),
    );
    if !recall_values.is_empty() {
        metrics.insert(
            "recall.recall_at_k".to_string(),
            MetricValue::Distribution(Distribution::from_samples(recall_values)),
        );
        metrics.insert(
            "recall.mrr".to_string(),
            MetricValue::Distribution(Distribution::from_samples(mrr_values)),
        );
    }
    metrics
}

fn aggregate_pair(cases: &[CaseBlock], key: &str) -> MetricsMap {
    let mut jaccards = Vec::new();
    let mut shared = Vec::new();
    let mut rank_diffs = Vec::new();
    let mut unavailable = 0u64;
    let mut total = 0u64;

    for case in cases {
        let Some(outcome) = case.pairs.get(key) else {
            continue;
        };
        total += 1;
        match outcome {
            PairOutcome::Compared {
                jaccard,
                shared_hits,
                mean_rank_difference,
            } => {
                if let Some(value) = jaccard {
                    jaccards.push(*value);
                }
                shared.push(*shared_hits as f64);
                if let Some(value) = mean_rank_difference {
                    rank_diffs.push(*value);
                }
            }
            PairOutcome::Unavailable { .. } => unavailable += 1,
        }
    }

    let mut metrics: MetricsMap = BTreeMap::new();
    metrics.insert(
        "overlap.jaccard".to_string(),
        MetricValue::Distribution(Distribution::from_samples(jaccards)),
    );
    metrics.insert(
        "overlap.shared_hits".to_string(),
        MetricValue::Distribution(Distribution::from_samples(shared)),
    );
    metrics.insert(
        "overlap.mean_rank_difference".to_string(),
        MetricValue::Distribution(Distribution::from_samples(rank_diffs)),
    );
    metrics.insert(
        "pairs.unavailable_rate".to_string(),
        MetricValue::Ratio(ratio_metric(unavailable, total)),
    );
    metrics
}

fn build_definitions() -> BTreeMap<String, MetricDef> {
    let mut d = BTreeMap::new();
    d.insert(
        "hits.count".to_string(),
        def(
            "count",
            "distribution",
            &["model"],
            "Number of passage hits sources/search returned for a case, already truncated to \
             the case's own limit.",
            "POST /contexts/{name}/sources/search",
            None,
        ),
    );
    d.insert(
        "hits.empty_rate".to_string(),
        def(
            "ratio",
            "ratio",
            &["model"],
            "Share of successfully searched cases whose search returned zero hits.",
            "POST /contexts/{name}/sources/search",
            None,
        ),
    );
    d.insert(
        "hits.distinct_sources".to_string(),
        def(
            "count",
            "distribution",
            &["model"],
            "Number of distinct source documents among a case's hits — a coarse diversity \
             signal.",
            "POST /contexts/{name}/sources/search",
            None,
        ),
    );
    d.insert(
        "lanes.bm25_only".to_string(),
        def(
            "count",
            "distribution",
            &["model"],
            "Hits a case's search found via the lexical (BM25) lane only.",
            "POST /contexts/{name}/sources/search response plan.lanes",
            None,
        ),
    );
    d.insert(
        "lanes.vector_only".to_string(),
        def(
            "count",
            "distribution",
            &["model"],
            "Hits found via the semantic (vector) lane only.",
            "POST /contexts/{name}/sources/search response plan.lanes",
            Some(
                "0 for every case when the vector lane never ran — see each case's own \
                 models.*.plan.vector.reason.",
            ),
        ),
    );
    d.insert(
        "lanes.both".to_string(),
        def(
            "count",
            "distribution",
            &["model"],
            "Hits both lanes agreed on.",
            "POST /contexts/{name}/sources/search response plan.lanes",
            None,
        ),
    );
    d.insert(
        "lanes.neither".to_string(),
        def(
            "count",
            "distribution",
            &["model"],
            "Hits whose per-hit lane evidence was readable but named neither lane — expected to \
             be 0 in practice; tracked so a genuine occurrence is visible rather than silently \
             folded into bm25_only/vector_only.",
            "POST /contexts/{name}/sources/search response hits[].lanes",
            None,
        ),
    );
    d.insert(
        "lanes.unknown".to_string(),
        def(
            "count",
            "distribution",
            &["model"],
            "Hits whose per-hit lane evidence could not be read at all — an older or otherwise \
             non-conforming server's response. Kept apart from bm25_only/vector_only/both/neither \
             so a legacy response's hits are never misread as evidencing zero lane activity.",
            "taguru benchmark search's legacy response fallback",
            None,
        ),
    );
    d.insert(
        "cases.error_rate".to_string(),
        def(
            "ratio",
            "ratio",
            &["model"],
            "Share of cases whose search request failed outright — a network/HTTP failure, or \
             the model's corpus was never available to search.",
            "taguru benchmark search",
            None,
        ),
    );
    d.insert(
        "recall.recall_at_k".to_string(),
        def(
            "ratio",
            "distribution",
            &["model"],
            "Fraction of a case's expected_sources (relevance >= 1) and expected_concepts, \
             jointly, found within the top-k hits. Only computed for cases carrying at least \
             one expectation.",
            "eval.jsonl expected_sources/expected_concepts",
            Some(
                "expected_sources match by (source, paragraph) resolved against the results \
                 directory's document dictionary; expected_concepts match by a case/NFKC-folded \
                 substring test against a hit's own text.",
            ),
        ),
    );
    d.insert(
        "recall.mrr".to_string(),
        def(
            "ratio",
            "distribution",
            &["model"],
            "1 / rank of the first hit that satisfies any of a case's expected_sources or \
             expected_concepts entries, 0 if none does.",
            "eval.jsonl expected_sources/expected_concepts",
            None,
        ),
    );
    d.insert(
        "overlap.jaccard".to_string(),
        def(
            "ratio",
            "distribution",
            &["pair"],
            "Jaccard similarity of two models' hit (source, paragraph) sets for the same case. \
             Omitted from the sample when both sides returned zero hits.",
            "benchmark::identity::jaccard over sources/search hits",
            None,
        ),
    );
    d.insert(
        "overlap.shared_hits".to_string(),
        def(
            "count",
            "distribution",
            &["pair"],
            "Number of (source, paragraph) locators both models' hits agreed on, for one case.",
            "sources/search hits",
            None,
        ),
    );
    d.insert(
        "overlap.mean_rank_difference".to_string(),
        def(
            "count",
            "distribution",
            &["pair"],
            "Mean absolute difference in hit position (1-based) among locators both models \
             found, within one case.",
            "sources/search hits",
            None,
        ),
    );
    d.insert(
        "pairs.unavailable_rate".to_string(),
        def(
            "ratio",
            "ratio",
            &["pair"],
            "Share of cases where at least one side of the pair could not be searched (corpus \
             unavailable, or the request failed).",
            "taguru benchmark search",
            None,
        ),
    );
    d
}

fn write_retrieval(dir: &Path, retrieval: &RetrievalFile) -> Result<(), String> {
    let path = dir.join("retrieval.json");
    let text = serde_json::to_string_pretty(retrieval).expect("a retrieval file serializes");
    crate::storage::write_atomic(&path, text.as_bytes())
        .map_err(|error| format!("writing {}: {error}", path.display()))
}

// ============================== Value shapes ==============================

#[derive(Debug, Clone, Serialize)]
struct RetrievalFile {
    taguru_benchmark_retrieval: u64,
    run_id: String,
    generated_at: String,
    matching: identity::Matching,
    inputs: InputsBlock,
    definitions: BTreeMap<String, MetricDef>,
    warnings: Vec<String>,
    corpus: BTreeMap<String, CorpusBlock>,
    cases: Vec<CaseBlock>,
    models: BTreeMap<String, MetricsMap>,
    pairs: BTreeMap<String, MetricsMap>,
}

#[derive(Debug, Clone, Serialize)]
struct InputsBlock {
    results_dir: String,
    eval: EvalInputsBlock,
    url: String,
    run_index: usize,
    default_limit: usize,
}

#[derive(Debug, Clone, Serialize)]
struct EvalInputsBlock {
    path: String,
    name: Option<String>,
    cases: usize,
}

/// `outcome` is a plain string ("built" | "skipped" | "rejected" |
/// "failed" | "search_only"), not a closed enum — ADR 0003 §10's
/// range-acceptance posture for values, matching `ManifestCell.outcome`
/// (src/benchmark.rs).
#[derive(Debug, Clone, Serialize)]
struct CorpusBlock {
    context: String,
    outcome: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    documents_imported: usize,
    documents_failed: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    passage_vectors: Option<PassageVectorInfo>,
}

#[derive(Debug, Clone, Serialize)]
struct PassageVectorInfo {
    model: Option<String>,
    rows: usize,
}

#[derive(Debug, Clone, Serialize)]
struct CaseBlock {
    case_id: String,
    query: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    cues: Vec<String>,
    limit: usize,
    has_expectations: bool,
    models: BTreeMap<String, SearchOutcome>,
    pairs: BTreeMap<String, PairOutcome>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
enum SearchOutcome {
    Searched {
        hit_count: usize,
        empty: bool,
        distinct_sources: usize,
        lanes: LaneHitCounts,
        #[serde(skip_serializing_if = "Option::is_none")]
        plan: Option<LanesEcho>,
        hits: Vec<HitLocator>,
        #[serde(skip_serializing_if = "Option::is_none")]
        recall: Option<RecallResult>,
    },
    Failed {
        message: String,
    },
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
struct LaneHitCounts {
    bm25_only: usize,
    vector_only: usize,
    both: usize,
    neither: usize,
    /// Hits from a response whose per-hit lane evidence could not be
    /// read (the legacy fallback path in [`extract_hits`]) — kept
    /// apart from `neither` (a lane genuinely recording no evidence)
    /// so the two are never conflated.
    unknown: usize,
}

#[derive(Debug, Clone, Serialize)]
struct LanesEcho {
    bm25: LaneEcho,
    vector: LaneEcho,
}

#[derive(Debug, Clone, Serialize)]
struct LaneEcho {
    ran: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    floor: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
struct HitLocator {
    rank: usize,
    source: String,
    paragraph: u32,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct RecallResult {
    recall_at_k: f64,
    mrr: f64,
    expected_total: usize,
    matched: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
enum PairOutcome {
    Compared {
        jaccard: Option<f64>,
        shared_hits: usize,
        mean_rank_difference: Option<f64>,
    },
    Unavailable {
        message: String,
    },
}

#[cfg(test)]
mod tests;
