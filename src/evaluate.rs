//! `taguru evaluate` (issue #215, ADR 0004): a quality gate over one
//! already-populated context, driven entirely over HTTP like `taguru
//! benchmark search` (ADR 0003 §11) — no in-process retrieval, no
//! answer-generation LLM anywhere on this path. Per case, two
//! independent lanes run in a fixed order with no fusion between them
//! (ADR 0004 §7):
//!
//! - **Passage lane** (always): `POST /contexts/{name}/sources/search`.
//! - **Structural lane** (only when a case declares
//!   `expected_concepts`/`expected_labels`/`expected_associations`):
//!   `/resolve`/`/resolve_label` for coverage, then `/query` for each
//!   `expected_associations[]` entry whose three positions all resolve
//!   to exactly one name apiece.
//! - **Citation lane** (only when a case declares `expected_citations`):
//!   one `POST /contexts/{name}/citations` call per entry, always run —
//!   independent of whether the passage lane found anything at all (ADR
//!   0004 §8).
//!
//! `recall`, `activate`, `explore`, and `describe` are deliberately
//! never called. `recall`'s HTTP layer pages a hub concept's incident
//! edges at `clamp(limit, 100, 1000)` (`api/recall.rs`), which can push
//! an expected triple outside the page — a silent false miss that looks
//! like a retrieval failure but is a paging artifact; `query` pins the
//! exact triple instead and has no such risk. `activate`/`explore`
//! depend on `decay`/`max_depth`, which no eval field declares. `describe`
//! is `Role::Read` and equally tempting to reach for; it is named here
//! explicitly so a future reader does not assume it was overlooked.
//!
//! Recall@k/MRR/nDCG (graded relevance, ADR 0004 §274) and
//! concept/label/association retrieval coverage are scored here, from
//! rank and label alone — no lane's raw graph/BM25/vector score is
//! ever folded into another's (#215's own requirement). Citation recall
//! and locator validity (ADR 0004 §8) are two measurements that are
//! never merged into one score — see the `Citation lane` and
//! `citations.*`/`missed` scoring below. Configurable thresholds and
//! exit 3 (#276) land in the [`thresholds`] submodule; `taguru evaluate
//! compare` (#277, ADR 0004 §9.2), which classifies two evaluation.json
//! runs into improved/regressed/added/removed cases, lands in the
//! [`compare`] submodule.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;
use serde_json::Value;
use taguru::context::normalize_entry;

use crate::api::MatchPage;
use crate::api::evidence::budget::{BudgetLimits, BudgetUsage};
use crate::api::evidence::rerank::{REASON_EMPTY_POOL, RerankerPlan};
use crate::api::evidence::select::{CitationRef, EvidenceItem, SelectionPlan, budget_only_omitted};
use crate::api::resolve::TieredResolution;
use crate::api::sources::{Citation, PassageHit, PassageLanes, PassagePage, SearchContextPlan};
use crate::cli::default_base_url;
use crate::config::{load_config, subcommand_usage_error};
use crate::evalset::{self, EvalCase, ExpectedAssociation, ExpectedCitation, ExpectedSource};
use crate::measure::{Distribution, MetricDef, MetricValue, MetricsMap, def, ratio_metric};
use crate::registry::{ContextRevision, DirectoryEntry};
use crate::remote::{self, Api, ApiFailure};
use thresholds::{ThresholdReport, load_thresholds};

const EVALUATION_VERSION: u64 = 1;
/// `options.limit` unspecified — matches `benchmark search`'s own
/// `DEFAULT_LIMIT` (`search.rs:78`).
const DEFAULT_LIMIT: usize = 10;
/// ADR 0004 §7 step 1: omitting `limit` on `/resolve`/`/resolve_label`
/// means "the ceiling itself" (up to 1000 candidates,
/// `resolve.rs:26-31`) — no eval case needs that, so evaluate always
/// names this explicitly.
const RESOLVE_LIMIT: usize = 5;
/// Mirrors the server's own `MAX_MATCH_LIMIT` (`src/api.rs`) — an
/// `options.limit` above this could never be honored anyway. Doubles
/// as the source-preflight walk's own page size.
const MAX_SEARCH_LIMIT: usize = 1000;
/// A server-provided error message is truncated to this many bytes, on
/// a UTF-8 character boundary (ADR 0004 §11) — long enough to diagnose,
/// short enough that a run against a misbehaving server cannot blow up
/// the artifact.
const MAX_ERROR_BYTES: usize = 200;
const DEFAULT_OUT: &str = "evaluation.json";

const USAGE: &str = "\
usage: taguru evaluate --eval FILE --context NAME [--url URL]
                        [--config FILE] [--out FILE] [--thresholds FILE]
                        [--assembly] [--max-items N] [--max-bytes N]
                        [--max-tokens N] [--rerank MODEL]
       taguru evaluate compare BASE.json HEAD.json [--out FILE]

Runs eval.jsonl's cases (ADR 0003 §11's shared dataset, #215's own
extension fields) against one already-populated context's live
retrieval endpoints and writes evaluation.json: per-case passage-lane
hits and structural-lane resolve/query outcomes, recall@k/MRR/nDCG,
concept/label/association coverage, citation recall and locator
validity, corpus revision bracketing, and run metadata.

  --eval FILE            eval.jsonl (ADR 0003 §11's shared dataset)
  --context NAME          the already-populated context to evaluate
  --url URL              the server to query; default resolves the
                        same way `taguru health` does (TAGURU_ADDR, or
                        --config/TAGURU_CONFIG)
  --config FILE          load before resolving --url (same as --config
                        everywhere else)
  --out FILE             where to write the artifact (evaluation.json)
  --thresholds FILE      a checked-in JSON file of regression bounds
                        (ADR 0004 §9.3); a completed run that violates
                        one exits 3. Without --thresholds every
                        completed run exits 0 and is report-only (a
                        stderr line says so).
  --assembly             #308 (ADR 0006 §14): replaces the passage
                        lane's `sources/search` call with
                        `POST /contexts/{name}/evidence` — the same
                        structural lane runs either way, so a
                        baseline/assembly pair of runs stays
                        comparable. Without --assembly, evaluate's
                        behavior is unchanged from before this flag
                        existed.
  --max-items N          #308's equal-budget contract: the same three
  --max-bytes N          ceilings apply to both an --assembly run (via
  --max-tokens N          the request's own `budget`) and a baseline
                        run (truncated client-side with the identical
                        accounting `crate::api::evidence::budget` uses
                        server-side). Any one given truncates every
                        case; none given leaves both modes untruncated,
                        matching every run before this flag existed.
  --rerank MODEL         opts an --assembly run into a configured
                        reranker (ADR 0006 §12); usage error without
                        --assembly. No provider configured, or a model
                        mismatch, degrades to the deterministic order
                        and is recorded, never a failure.

`taguru evaluate compare BASE.json HEAD.json [--out FILE]` (ADR 0004
§9.2) reads two evaluation.json runs and writes changes.jsonl: one
record per case whose outcome moved, improved/regressed/added/removed
— unchanged cases are counted, never emitted per case. See `taguru
evaluate compare --help`.

EXIT CODES: 0 ok, or a completed run with no --thresholds or every
threshold satisfied, or a completed compare (regardless of regressions
found) · 1 the run could not complete, or changes.jsonl could not be
written · 2 usage or input error, including a malformed --thresholds
file or a malformed evaluation.json given to compare · 3 the run
completed and a threshold was violated.

Contract and discipline: docs/evaluate.html, docs/evidence.html,
adr/0004-retrieval-citation-quality-gate.md,
adr/0006-budgeted-evidence-assembly.md §14.
";

pub fn run(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("--help") | Some("-h") => {
            print!("{USAGE}");
            0
        }
        Some(flag) if flag.starts_with("--") => run_evaluate(args),
        Some("compare") => compare::run(&args[1..]),
        Some(other) => subcommand_usage_error(
            "evaluate",
            &format!(
                "unknown subcommand '{other}' — the default mode is selected by a leading \
                 flag, e.g. 'taguru evaluate --eval FILE --context NAME', or use the \
                 'compare' subcommand"
            ),
        ),
        None => subcommand_usage_error("evaluate", "expected --eval FILE --context NAME"),
    }
}

fn run_evaluate(args: &[String]) -> i32 {
    let eval_args = match parse_args(args) {
        Ok(eval_args) => eval_args,
        Err(code) => return code,
    };

    let loaded = match evalset::load_eval_file(&eval_args.eval, evalset::Extensions::Interpret) {
        Ok(loaded) => loaded,
        Err(message) => {
            eprintln!("taguru: evaluate: {message}");
            return 2;
        }
    };
    for warning in &loaded.warnings {
        eprintln!("taguru: evaluate: {warning}");
    }
    if let Err(message) = validate_limits(&loaded.cases) {
        eprintln!("taguru: evaluate: {message}");
        return 2;
    }

    // #308 (ADR 0006 §14): `POST /contexts/{name}/evidence` has no
    // `tags`/`since`/`until` request fields at all (§5.1) — a case
    // declaring any of them still runs under --assembly (never a
    // usage error; a dataset shared with a baseline run must still
    // load), but the filter it named is silently not applied unless
    // this is said out loud.
    let mut extra_warnings: Vec<String> = Vec::new();
    if eval_args.assembly {
        for case in &loaded.cases {
            let unsupported = !case.options.tags.is_empty()
                || case.options.since.is_some()
                || case.options.until.is_some();
            if unsupported {
                let warning = format!(
                    "case '{}': options.tags/since/until are not applied under --assembly \
                     (POST /contexts/{{name}}/evidence has no such request fields)",
                    case.case_id
                );
                eprintln!("taguru: evaluate: {warning}");
                extra_warnings.push(warning);
            }
        }
    }

    // Validated before any network call: a malformed --thresholds file
    // is a usage error (exit 2) that should never depend on a
    // reachable server to be reported (ADR 0004 §9.3).
    let definitions = build_definitions();
    let case_ids: BTreeSet<String> = loaded
        .cases
        .iter()
        .map(|case| case.case_id.clone())
        .collect();
    let loaded_thresholds = match &eval_args.thresholds {
        Some(path) => match load_thresholds(path, &definitions, &case_ids) {
            Ok(thresholds) => Some(thresholds),
            Err(message) => {
                eprintln!("taguru: evaluate: {message}");
                return 2;
            }
        },
        None => None,
    };

    let config = eval_args
        .config
        .clone()
        .or_else(|| std::env::var("TAGURU_CONFIG").ok().map(PathBuf::from));
    if let Some(path) = &config {
        load_config(path);
    }
    let base = match eval_args.url.clone() {
        Some(url) => url,
        None => match default_base_url() {
            Ok(url) => url,
            Err(error) => {
                eprintln!("taguru: evaluate: {error}");
                return 2;
            }
        },
    };
    if let Err(message) = remote::reject_userinfo(&base) {
        eprintln!("taguru: evaluate: {message}");
        return 2;
    }
    // `reject_userinfo` deliberately leaves an unparsable `base` alone
    // (`remote.rs`'s own doc comment) so `Api::url` can report it once,
    // on the first real request — safe for every other consumer, none
    // of which write `base` anywhere first. `evaluate` writes it into
    // `evaluation.json` (via `mask_url`) before any request is made,
    // and a string that fails to parse as a URL can still be, or
    // contain, `user:pass@host` text (e.g. a malformed host); refuse
    // it here instead of letting `mask_url` decide what to do with it
    // (issue #289, mirroring `benchmark search`'s own fix, issue #281
    // / PR #288).
    if url::Url::parse(&base).is_err() {
        eprintln!("taguru: evaluate: the --url value could not be parsed as a URL");
        return 2;
    }

    let api = Api::new(base.clone());
    let masked_url = mask_url(&base);
    // ADR 0002 §5: print the target before sending anything, even
    // though evaluate is read-only — the same reason it calls
    // warn_on_version_skew below despite ADR 0002 §5 technically
    // scoping that to mutating verbs (ADR 0004 §11).
    eprintln!("evaluate → {masked_url}");
    api.warn_on_version_skew("evaluate");

    // Matches `search.rs:217-225`'s hard-fail-on-unreachable posture:
    // nothing downstream can produce a meaningful artifact without a
    // reachable server, so this fails loudly now instead of writing an
    // empty evaluation.json case by case.
    if let Err(error) = api.get_raw(&["health"]) {
        eprintln!("taguru: evaluate: server at {masked_url} is not reachable: {error}");
        return 1;
    }

    let context = &eval_args.context;
    let entry_before_value = match api.get_envelope(&["contexts", context]) {
        Ok(value) => value,
        Err(ApiFailure::NotFound { message, .. }) => {
            eprintln!("taguru: evaluate: {message}");
            return 2;
        }
        Err(ApiFailure::Other(message)) => {
            eprintln!("taguru: evaluate: {message}");
            return 1;
        }
    };
    let entry_before: DirectoryEntry = match serde_json::from_value(entry_before_value) {
        Ok(entry) => entry,
        Err(error) => {
            eprintln!("taguru: evaluate: {context}: unreadable context entry: {error}");
            return 1;
        }
    };

    let embeddings = fetch_embeddings(&api, context);

    let sources = match list_all_sources(&api, context) {
        Ok(sources) => sources,
        Err(message) => {
            eprintln!("taguru: evaluate: {message}");
            return 1;
        }
    };
    // Preflight (ADR 0004 §6): an expected_sources entry naming a
    // source the corpus does not carry is a reported error at startup,
    // never a silent zero-recall case buried in the aggregate.
    let missing_sources: Vec<(&str, &str)> = loaded
        .cases
        .iter()
        .flat_map(|case| {
            case.expected_sources
                .iter()
                .map(move |expected| (case.case_id.as_str(), expected.source.as_str()))
        })
        .filter(|(_, source)| !sources.contains(*source))
        .collect();
    if !missing_sources.is_empty() {
        for (case_id, source) in &missing_sources {
            eprintln!(
                "taguru: evaluate: case '{case_id}': expected_sources names '{source}', which \
                 '{context}' does not carry"
            );
        }
        return 2;
    }

    // Always resolved (defaulted 40/65536/4000 when no --max-* flag
    // was given) — an --assembly run always sends this to
    // POST /contexts/{name}/evidence, since that endpoint has no
    // unbudgeted mode (ADR 0006 §8). `budget_given` below is what
    // decides whether `baseline` mode additionally truncates to the
    // same ceilings, and whether either mode's artifact carries a
    // `budget` block at all.
    let budget_given = eval_args.budget.is_some();
    let limits: BudgetLimits = match &eval_args.budget {
        Some(budget) => budget.resolve(),
        None => BudgetLimits::resolve(None),
    };
    let run_config = RunConfig {
        assembly: eval_args.assembly,
        limits,
        budget_given,
        rerank: eval_args.rerank.as_deref(),
    };

    let default_limit = DEFAULT_LIMIT;
    let cases: Vec<CaseBlock> = loaded
        .cases
        .iter()
        .map(|case| build_case_block(&api, context, case, default_limit, &run_config))
        .collect();

    let entry_after_value = match api.get_envelope(&["contexts", context]) {
        Ok(value) => value,
        Err(failure) => {
            eprintln!(
                "taguru: evaluate: {context}: could not re-read the context after the run: {}",
                failure.into_message()
            );
            return 1;
        }
    };
    let entry_after: DirectoryEntry = match serde_json::from_value(entry_after_value) {
        Ok(entry) => entry,
        Err(error) => {
            eprintln!(
                "taguru: evaluate: {context}: unreadable context entry after the run: {error}"
            );
            return 1;
        }
    };

    // ADR 0004 §12: equality across all three ContextRevision lanes,
    // never ordering — bracketing before-and-after is the only way to
    // detect a write that landed mid-run, since evaluate spans many
    // independent HTTP calls with no transactional boundary.
    let stable = entry_before.revision == entry_after.revision;

    let metrics = build_metrics(&cases, run_config.rerank.is_some());
    let evaluation = EvaluationFile {
        taguru_evaluation: EVALUATION_VERSION,
        generated_at: crate::clock::iso8601_utc(crate::clock::now_unix_secs()),
        matching: MatchingBlock::default(),
        inputs: InputsBlock {
            eval: EvalInputsBlock {
                path: eval_args.eval.display().to_string(),
                name: loaded.name.clone(),
                cases: loaded.cases.len(),
            },
            context: context.clone(),
            url: masked_url.clone(),
            out: eval_args.out.display().to_string(),
            default_limit,
            resolve_limit: RESOLVE_LIMIT,
            mode: if eval_args.assembly {
                "assembly"
            } else {
                "baseline"
            }
            .to_string(),
            budget: (eval_args.assembly || budget_given).then_some(limits),
            rerank: eval_args.rerank.clone(),
        },
        corpus: CorpusBlock {
            revision_before: entry_before.revision,
            revision_after: entry_after.revision,
            stable,
            last_write_epoch_before: entry_before.usage.last_write_epoch,
            last_write_epoch_after: entry_after.usage.last_write_epoch,
            embeddings,
            sources_count: sources.len(),
        },
        // `None` when --thresholds was not given (report-only); `Some`
        // once the loaded bounds have been checked against this run's
        // own cases/metrics/stable, just below. Borrows `cases`/
        // `metrics` here, before either moves into the fields below —
        // struct-literal field initializers evaluate in source order.
        thresholds: loaded_thresholds
            .as_ref()
            .map(|loaded| loaded.evaluate(&cases, &metrics, stable)),
        definitions,
        warnings: loaded
            .warnings
            .iter()
            .cloned()
            .chain(extra_warnings)
            .collect(),
        cases,
        metrics,
    };

    if let Err(message) = write_evaluation(&eval_args.out, &evaluation) {
        eprintln!("taguru: evaluate: {message}");
        return 1;
    }

    print_summary(&evaluation, &masked_url, context);

    match &evaluation.thresholds {
        Some(report) => {
            print_threshold_summary(report);
            if report.passed { 0 } else { 3 }
        }
        None => {
            eprintln!(
                "taguru: evaluate: no --thresholds given — this run is report-only and \
                 always exits 0"
            );
            0
        }
    }
}

// ============================== Arguments ==============================

#[derive(Debug)]
struct EvaluateArgs {
    eval: PathBuf,
    context: String,
    url: Option<String>,
    config: Option<PathBuf>,
    out: PathBuf,
    thresholds: Option<PathBuf>,
    /// #308: swaps the passage lane from `sources/search` to
    /// `POST /contexts/{name}/evidence` (ADR 0006). The structural
    /// lane never changes between the two modes — coverage/lane-cross
    /// stay comparable across a `baseline`/`assembly` pair.
    assembly: bool,
    /// #308's equal-budget contract: when any of the three is given,
    /// both modes are truncated to the identical ceiling — `assembly`
    /// via the request's own `budget` object, `baseline` by reusing
    /// the server's own accounting (`crate::api::evidence::budget`)
    /// client-side. Absent entirely, `baseline` behaves exactly as it
    /// did before this flag existed (no truncation, no `budget` block
    /// in the artifact) — this default is what keeps every archived
    /// `evaluation.json` and every existing caller unaffected.
    /// `assembly` has no such unbudgeted mode: `POST
    /// /contexts/{name}/evidence` always enforces *some* budget, so an
    /// `--assembly` run with none of these flags still runs — and
    /// still carries a `budget` block — under the server's own
    /// defaults (`max_items: 40, max_bytes: 65536, max_tokens: 4000`).
    budget: Option<evidence::EvidenceBudgetArgs>,
    /// `--rerank MODEL`: usage error (exit 2) without `--assembly`.
    rerank: Option<String>,
}

/// The per-run mode settings [`build_case_block`] threads through to
/// [`lanes::run_passage_lane`]/[`evidence::run_evidence_lane`] —
/// resolved once from [`EvaluateArgs`] rather than re-derived per case.
pub(crate) struct RunConfig<'a> {
    pub(crate) assembly: bool,
    /// Always a resolved value (defaulted when no `--max-*` flag was
    /// given) — `assembly` mode always sends it; `budget_given` below
    /// gates whether `baseline` mode uses it at all.
    pub(crate) limits: BudgetLimits,
    /// Whether `--max-items`/`--max-bytes`/`--max-tokens` was actually
    /// given. In `assembly` mode this only decides between the
    /// caller's own ceilings and the server's own defaults — the
    /// request always carries a `budget` object either way. In
    /// `baseline` mode it decides whether truncation (and a `budget`
    /// block in the artifact) happens at all.
    pub(crate) budget_given: bool,
    pub(crate) rerank: Option<&'a str>,
}

/// Shared by `--max-items`/`--max-bytes`/`--max-tokens` in
/// [`parse_args`]: the "given twice" check runs BEFORE the value is
/// parsed, so a second occurrence whose value fails to parse (e.g.
/// `--max-items 5 --max-items abc`) is still diagnosed as a duplicate
/// flag, not the generic "needs a positive integer" message a
/// `Some(_) if already_set` guard on the parsed `Option` would fall
/// through to instead (a failed parse collapses the whole `and_then`
/// chain to `None`, which never matches that guard at all).
fn parse_positive_usize_flag<'a>(
    rest: &mut impl Iterator<Item = &'a String>,
    flag: &str,
    slot: &mut Option<usize>,
) -> Result<(), i32> {
    if slot.is_some() {
        return Err(subcommand_usage_error(
            "evaluate",
            &format!("{flag} given twice"),
        ));
    }
    match rest.next().and_then(|value| value.parse::<usize>().ok()) {
        Some(n) if n >= 1 => *slot = Some(n),
        _ => {
            return Err(subcommand_usage_error(
                "evaluate",
                &format!("{flag} needs a positive integer"),
            ));
        }
    }
    Ok(())
}

fn parse_args(args: &[String]) -> Result<EvaluateArgs, i32> {
    let usage = |message: &str| subcommand_usage_error("evaluate", message);
    let mut eval: Option<PathBuf> = None;
    let mut context: Option<String> = None;
    let mut url: Option<String> = None;
    let mut config: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut thresholds: Option<PathBuf> = None;
    let mut assembly = false;
    let mut max_items: Option<usize> = None;
    let mut max_bytes: Option<usize> = None;
    let mut max_tokens: Option<usize> = None;
    let mut rerank: Option<String> = None;

    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--help" | "-h" => {
                print!("{USAGE}");
                return Err(0);
            }
            "--eval" => match rest.next() {
                Some(path) if eval.is_none() => eval = Some(PathBuf::from(path)),
                Some(_) => return Err(usage("--eval given twice")),
                None => return Err(usage("--eval needs a file path")),
            },
            "--context" => match rest.next() {
                Some(name) if context.is_none() => context = Some(name.clone()),
                Some(_) => return Err(usage("--context given twice")),
                None => return Err(usage("--context needs a name")),
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
            "--out" => match rest.next() {
                Some(path) if out.is_none() => out = Some(PathBuf::from(path)),
                Some(_) => return Err(usage("--out given twice")),
                None => return Err(usage("--out needs a file path")),
            },
            "--thresholds" => match rest.next() {
                Some(path) if thresholds.is_none() => thresholds = Some(PathBuf::from(path)),
                Some(_) => return Err(usage("--thresholds given twice")),
                None => return Err(usage("--thresholds needs a file path")),
            },
            "--assembly" => {
                if assembly {
                    return Err(usage("--assembly given twice"));
                }
                assembly = true;
            }
            "--max-items" => parse_positive_usize_flag(&mut rest, "--max-items", &mut max_items)?,
            "--max-bytes" => parse_positive_usize_flag(&mut rest, "--max-bytes", &mut max_bytes)?,
            "--max-tokens" => {
                parse_positive_usize_flag(&mut rest, "--max-tokens", &mut max_tokens)?
            }
            "--rerank" => match rest.next() {
                Some(model) if rerank.is_none() => rerank = Some(model.clone()),
                Some(_) => return Err(usage("--rerank given twice")),
                None => return Err(usage("--rerank needs a model name")),
            },
            flag if flag.starts_with("--") => {
                return Err(usage(&format!("unknown flag '{flag}' for evaluate")));
            }
            _ => return Err(usage("evaluate takes no positional arguments")),
        }
    }

    let eval = eval.ok_or_else(|| usage("--eval FILE is required"))?;
    let context = context.ok_or_else(|| usage("--context NAME is required"))?;
    if rerank.is_some() && !assembly {
        return Err(usage("--rerank requires --assembly"));
    }
    let budget = if max_items.is_some() || max_bytes.is_some() || max_tokens.is_some() {
        Some(evidence::EvidenceBudgetArgs {
            max_items,
            max_bytes,
            max_tokens,
        })
    } else {
        None
    };

    Ok(EvaluateArgs {
        eval,
        context,
        url,
        config,
        out: out.unwrap_or_else(|| PathBuf::from(DEFAULT_OUT)),
        thresholds,
        assembly,
        budget,
        rerank,
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

// ============================ URL and messages ============================

/// scheme + host + port only — no userinfo, path, or query (ADR 0004
/// §11). `reject_userinfo` and the unparsable-`base` check above
/// already refuse a `base` this function would otherwise have to
/// redact before this runs, but `mask_url` redacts an unparsable input
/// to a fixed placeholder rather than echoing it back regardless — a
/// string that fails to parse as a URL is not proven free of
/// credential-shaped text, and this artifact is shared and archived
/// (issue #289, mirroring `benchmark search`'s own fix, issue #281 /
/// PR #288).
fn mask_url(base: &str) -> String {
    match url::Url::parse(base) {
        Ok(url) => {
            let mut masked = format!("{}://{}", url.scheme(), url.host_str().unwrap_or(""));
            if let Some(port) = url.port() {
                masked.push_str(&format!(":{port}"));
            }
            masked
        }
        Err(_) => "<unparseable-url>".to_string(),
    }
}

/// Truncates a server-provided message to [`MAX_ERROR_BYTES`], cutting
/// only on a UTF-8 character boundary (ADR 0004 §11) — never inside a
/// multi-byte sequence, which would produce invalid UTF-8.
fn truncate_message(message: &str) -> String {
    if message.len() <= MAX_ERROR_BYTES {
        return message.to_string();
    }
    let mut end = MAX_ERROR_BYTES;
    while end > 0 && !message.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &message[..end])
}

// ============================ Context metadata ============================

fn fetch_embeddings(api: &Api, context: &str) -> Option<EmbeddingsBlock> {
    match api.get(&["contexts", context, "embeddings"]) {
        Ok(value) => Some(EmbeddingsBlock {
            provider_model: value
                .get("provider_model")
                .and_then(Value::as_str)
                .map(str::to_string),
        }),
        Err(message) => {
            eprintln!(
                "taguru: evaluate: {context}: could not read embedding status (recording none): \
                 {message}"
            );
            None
        }
    }
}

/// Keyset-walks `GET /contexts/{name}/sources` itself: its page items
/// are bare strings, not `{name: ...}` objects, so `Api::list_names`
/// (built for the latter) does not fit — see `remote.rs`'s own comment
/// on `get_with_query` for why this is not new HTTP client code.
fn list_all_sources(api: &Api, context: &str) -> Result<BTreeSet<String>, String> {
    let mut names = BTreeSet::new();
    let mut after: Option<String> = None;
    loop {
        let limit = MAX_SEARCH_LIMIT.to_string();
        let mut query: Vec<(&str, &str)> = vec![("limit", limit.as_str())];
        if let Some(cursor) = after.as_deref() {
            query.push(("after", cursor));
        }
        let body = api.get_with_query(&["contexts", context, "sources"], &query)?;
        let items = body["sources"]
            .as_array()
            .ok_or_else(|| format!("{context}: not a taguru sources page"))?;
        let mut page: Vec<String> = Vec::with_capacity(items.len());
        for item in items {
            let name = item
                .as_str()
                .ok_or_else(|| format!("{context}: a sources entry is not a string"))?;
            page.push(name.to_string());
        }
        let page_len = page.len();
        // Same "did the cursor actually advance" guard as
        // `remote.rs`'s own `list_names_paged`.
        if let (Some(cursor), Some(first)) = (after.as_deref(), page.first())
            && first.as_str() <= cursor
        {
            return Err(format!(
                "{context}: the server's sources page did not advance past '{cursor}'"
            ));
        }
        after = page.last().cloned();
        names.extend(page);
        if page_len == 0 {
            break;
        }
    }
    Ok(names)
}

/// Everything [`build_case_block`] needs beyond the two lanes' own raw
/// observations — computed by pure functions over what the lanes
/// already returned, so this never makes a network call of its own.
struct CaseScores {
    recall: Option<RecallBlock>,
    coverage: Option<CoverageBlock>,
    lane_cross: Option<LaneCrossBlock>,
    citations: Option<CitationsBlock>,
    missed: Vec<String>,
    missed_truncated: usize,
}

fn score_case(
    case: &EvalCase,
    passage: &PassageOutcome,
    structural: Option<&StructuralBlock>,
    citation_checks: &[CitationCheck],
) -> CaseScores {
    let hits: Option<&[HitLocator]> = match passage {
        PassageOutcome::Searched { hits, .. } => Some(hits),
        PassageOutcome::Failed { .. } => None,
    };

    let recall = hits.and_then(|hits| score_recall(&case.expected_sources, hits));

    let concept_resolutions: Vec<&CueResolution> = structural
        .map(|s| s.cues.iter().filter(|cue| cue.kind == "concept").collect())
        .unwrap_or_default();
    let label_resolutions: Vec<&CueResolution> = structural
        .map(|s| s.cues.iter().filter(|cue| cue.kind == "label").collect())
        .unwrap_or_default();

    let concepts_coverage = coverage_counts(&case.expected_concepts, &concept_resolutions);
    let labels_coverage = coverage_counts(&case.expected_labels, &label_resolutions);
    let associations_coverage = structural.and_then(|s| association_coverage(&s.associations));

    // "Structural hit" for the lane cross-tab: any structural
    // expectation this case declared was covered — concept, label, or
    // association alike, mirroring `runs_structural_lane`'s own
    // three-way OR.
    let structural_hit = concepts_coverage
        .iter()
        .chain(labels_coverage.iter())
        .chain(associations_coverage.iter())
        .any(|counts| counts.matched > 0);

    let coverage = (concepts_coverage.is_some()
        || labels_coverage.is_some()
        || associations_coverage.is_some())
    .then_some(CoverageBlock {
        concepts: concepts_coverage,
        labels: labels_coverage,
        associations: associations_coverage,
    });

    // Lane cross-tab (ADR 0004 §7): computed only over cases declaring
    // BOTH a structural expectation and a `relevance >= 1` source
    // expectation, and only when the passage lane actually completed —
    // a failed passage lane cannot honestly report `passage_hit`
    // either way, so it is excluded from the denominator just like it
    // is excluded from `recall`.
    let has_structural_expectation = runs_structural_lane(case);
    let has_source_expectation = case.expected_sources.iter().any(|e| e.relevance >= 1);
    let lane_cross = if has_structural_expectation && has_source_expectation && hits.is_some() {
        Some(LaneCrossBlock {
            structural_hit,
            passage_hit: recall.as_ref().is_some_and(|r| r.matched > 0),
        })
    } else {
        None
    };

    let (missed, missed_truncated) = build_missed(
        case,
        hits,
        &concept_resolutions,
        &label_resolutions,
        structural,
        citation_checks,
    );

    // `Some` only when the case actually declared expected_citations —
    // an empty `citation_checks` always means the lane never ran, never
    // "ran and found zero," so there is nothing to divide by (ADR 0004
    // §8's two measurements are undefined, not zero, on an empty set).
    let citations = (!citation_checks.is_empty()).then(|| CitationsBlock {
        recall: score_citation_recall(citation_checks),
        validity: score_citation_validity(citation_checks),
        checks: citation_checks.to_vec(),
    });

    CaseScores {
        recall,
        coverage,
        lane_cross,
        citations,
        missed,
        missed_truncated,
    }
}

fn elapsed_ms(started_at: Instant) -> u64 {
    started_at.elapsed().as_millis() as u64
}

// ================================ Metrics =================================

/// `rerank_requested_this_run` is `run_config.rerank.is_some()` — whether
/// THIS run's own `--rerank MODEL` flag was given, independent of
/// whether the queried server happens to have a reranker configured.
/// `RerankerPlan::not_requested` (`assemble.rs`) sets `configured` to
/// the *server's* own state (`state.reranker().is_some()`) even when a
/// case's request carried no `rerank` object at all — so a case's own
/// `reranker.configured` cannot answer "did this run ask for
/// reranking," only "could a reranker have run here if asked." Reading
/// it as request detection (as an earlier version of this function did)
/// let `rerank.ran`'s denominator count a `--rerank`-less run against a
/// rerank-configured server, contradicting `build_definitions`' own
/// "empty when `--rerank` was not given" contract for that metric.
fn build_metrics(cases: &[CaseBlock], rerank_requested_this_run: bool) -> MetricsMap {
    let mut passage_latencies = Vec::new();
    let mut resolve_latencies = Vec::new();
    let mut query_latencies = Vec::new();
    let mut passage_failures = 0u64;
    let mut structural_cases = 0u64;

    let mut recall_at_k = Vec::new();
    let mut mrr = Vec::new();
    let mut ndcg = Vec::new();
    let mut coverage_concepts = Vec::new();
    let mut coverage_labels = Vec::new();
    let mut coverage_associations = Vec::new();
    let mut lane_cross_n = 0u64;
    let mut structural_hit_n = 0u64;
    let mut passage_hit_n = 0u64;
    let mut both_n = 0u64;
    let mut neither_n = 0u64;

    let mut citation_latencies = Vec::new();
    let mut citation_recall = Vec::new();
    let mut citation_validity = Vec::new();
    // Denominator for resolved/no_source/no_paragraph: every citation
    // call actually made, across every case. section/quote each have
    // their own narrower denominator — only checks where that
    // particular field was declared at all.
    let mut citation_total = 0u64;
    let mut citation_resolved_n = 0u64;
    let mut citation_no_source_n = 0u64;
    let mut citation_no_paragraph_n = 0u64;
    let mut citation_other_n = 0u64;
    let mut citation_section_checked = 0u64;
    let mut citation_section_matched = 0u64;
    let mut citation_quote_checked = 0u64;
    let mut citation_quote_matched = 0u64;

    // #308 (ADR 0006 §14)
    let mut diversity_sources = Vec::new();
    let mut evidence_latencies = Vec::new();
    let mut budget_items_used = Vec::new();
    let mut budget_bytes_used = Vec::new();
    let mut budget_tokens_used = Vec::new();
    let mut budget_omitted_n = 0u64;
    let mut budget_considered_n = 0u64;
    let mut rerank_requested_n = 0u64;
    let mut rerank_ran_n = 0u64;

    for case in cases {
        match &case.passage {
            PassageOutcome::Searched { latency_ms, .. } => {
                passage_latencies.push(*latency_ms as f64)
            }
            PassageOutcome::Failed { latency_ms, .. } => {
                passage_latencies.push(*latency_ms as f64);
                passage_failures += 1;
            }
        }
        if let Some(structural) = &case.structural {
            structural_cases += 1;
            for cue in &structural.cues {
                resolve_latencies.push(cue.latency_ms as f64);
            }
            for assoc in &structural.associations {
                resolve_latencies.push(position_latency_ms(&assoc.subject) as f64);
                resolve_latencies.push(position_latency_ms(&assoc.label) as f64);
                resolve_latencies.push(position_latency_ms(&assoc.object) as f64);
                if let Some(query) = &assoc.query {
                    query_latencies.push(query_latency_ms(query) as f64);
                }
            }
        }

        if let Some(recall) = &case.recall {
            recall_at_k.push(recall.recall_at_k);
            mrr.push(recall.mrr);
            ndcg.push(recall.ndcg);
        }
        if let Some(coverage) = &case.coverage {
            if let Some(concepts) = &coverage.concepts {
                coverage_concepts.push(concepts.value);
            }
            if let Some(labels) = &coverage.labels {
                coverage_labels.push(labels.value);
            }
            if let Some(associations) = &coverage.associations {
                coverage_associations.push(associations.value);
            }
        }
        if let Some(cross) = &case.lane_cross {
            lane_cross_n += 1;
            structural_hit_n += u64::from(cross.structural_hit);
            passage_hit_n += u64::from(cross.passage_hit);
            both_n += u64::from(cross.structural_hit && cross.passage_hit);
            neither_n += u64::from(!cross.structural_hit && !cross.passage_hit);
        }
        if let Some(citations) = &case.citations {
            citation_recall.push(citations.recall.value);
            citation_validity.push(citations.validity.value);
            for check in &citations.checks {
                citation_total += 1;
                citation_latencies.push(check.latency_ms as f64);
                match &check.outcome {
                    CitationOutcome::Resolved { section, quote } => {
                        citation_resolved_n += 1;
                        match section {
                            SectionCheck::NotChecked => {}
                            SectionCheck::Matched { .. } => {
                                citation_section_checked += 1;
                                citation_section_matched += 1;
                            }
                            SectionCheck::Mismatched { .. } => {
                                citation_section_checked += 1;
                            }
                        }
                        if let Some(quote) = quote {
                            citation_quote_checked += 1;
                            citation_quote_matched += u64::from(quote.matched);
                        }
                    }
                    CitationOutcome::Unresolved { code, .. } => match code.as_deref() {
                        Some("no_source") => citation_no_source_n += 1,
                        Some("no_paragraph") => citation_no_paragraph_n += 1,
                        // An absent code (transport failure, unparsable
                        // response) or any other ErrorCode this loader
                        // does not special-case still counted toward
                        // citation_total above — bucketed here so
                        // resolved + no_source + no_paragraph + other
                        // always sums to citation_total instead of
                        // silently losing the record.
                        _ => citation_other_n += 1,
                    },
                }
            }
        }

        // #308 (ADR 0006 §14)
        if let Some(sources) = case.diversity_sources {
            diversity_sources.push(sources as f64);
        }
        // Assembly-mode `omitted_by_reason` can carry `duplicate_passage`
        // (near-duplicate suppression, which runs BEFORE budget
        // admission and would drop the same candidate at any budget)
        // alongside the budget-driven reasons — `budget_only_omitted`
        // (select.rs) isolates just the latter, so the loop below can
        // report `budget.omitted_rate`'s numerator as exactly what its
        // own documented definition promises ("share … a budget ceiling
        // dropped"), not `omitted_total`'s every-reason count. `None`
        // for a baseline case (no `EvidenceOutcome` at all), which falls
        // back to `BudgetAccounting::omitted_total` below — already
        // budget-only there, since baseline truncation has no other
        // omission reason.
        let mut budget_only_omitted_n: Option<u64> = None;
        if let Some(evidence) = &case.evidence {
            match evidence {
                EvidenceOutcome::Assembled {
                    latency_ms,
                    reranker,
                    omitted_by_reason,
                    ..
                } => {
                    evidence_latencies.push(*latency_ms as f64);
                    budget_only_omitted_n = Some(budget_only_omitted(omitted_by_reason) as u64);
                    // Gated on `rerank_requested_this_run` (this run's
                    // own `--rerank` flag), not `reranker.configured` —
                    // that field reflects the *server's* configuration,
                    // which stays `true` even on a case whose request
                    // carried no `rerank` object at all when the
                    // queried server happens to have one configured
                    // (see this function's own doc comment). Within a
                    // run that did pass `--rerank`, `empty_pool` (fewer
                    // than two survivors — nothing a reranker could
                    // have reordered either way) is still excluded: it
                    // is not a configured reranker's failure path
                    // firing, so it does not belong in "the
                    // configured-reranker success rate" this metric
                    // documents itself as.
                    let attempted = rerank_requested_this_run
                        && reranker.reason.as_deref() != Some(REASON_EMPTY_POOL);
                    if attempted {
                        rerank_requested_n += 1;
                        rerank_ran_n += u64::from(reranker.ran);
                    }
                }
                EvidenceOutcome::Failed { latency_ms, .. } => {
                    evidence_latencies.push(*latency_ms as f64);
                }
            }
        }
        if let Some(budget) = &case.budget {
            budget_items_used.push(budget.usage.items_used as f64);
            budget_bytes_used.push(budget.usage.bytes_used as f64);
            budget_tokens_used.push(budget.usage.tokens_used as f64);
            let omitted = budget_only_omitted_n.unwrap_or(budget.omitted_total as u64);
            budget_omitted_n += omitted;
            budget_considered_n += budget.omitted_total as u64 + budget.usage.items_used as u64;
        }
    }

    let total = cases.len() as u64;
    let mut metrics: MetricsMap = BTreeMap::new();
    metrics.insert(
        "latency.passage_ms".to_string(),
        MetricValue::Distribution(Distribution::from_samples(passage_latencies)),
    );
    metrics.insert(
        "latency.resolve_ms".to_string(),
        MetricValue::Distribution(Distribution::from_samples(resolve_latencies)),
    );
    metrics.insert(
        "latency.query_ms".to_string(),
        MetricValue::Distribution(Distribution::from_samples(query_latencies)),
    );
    metrics.insert(
        "passage.failure_rate".to_string(),
        MetricValue::Ratio(ratio_metric(passage_failures, total)),
    );
    metrics.insert(
        "structural.case_rate".to_string(),
        MetricValue::Ratio(ratio_metric(structural_cases, total)),
    );
    metrics.insert(
        "recall.recall_at_k".to_string(),
        MetricValue::Distribution(Distribution::from_samples(recall_at_k)),
    );
    metrics.insert(
        "recall.mrr".to_string(),
        MetricValue::Distribution(Distribution::from_samples(mrr)),
    );
    metrics.insert(
        "recall.ndcg".to_string(),
        MetricValue::Distribution(Distribution::from_samples(ndcg)),
    );
    metrics.insert(
        "coverage.concepts".to_string(),
        MetricValue::Distribution(Distribution::from_samples(coverage_concepts)),
    );
    metrics.insert(
        "coverage.labels".to_string(),
        MetricValue::Distribution(Distribution::from_samples(coverage_labels)),
    );
    metrics.insert(
        "coverage.associations".to_string(),
        MetricValue::Distribution(Distribution::from_samples(coverage_associations)),
    );
    metrics.insert(
        "citations.recall".to_string(),
        MetricValue::Distribution(Distribution::from_samples(citation_recall)),
    );
    metrics.insert(
        "citations.locator_validity".to_string(),
        MetricValue::Distribution(Distribution::from_samples(citation_validity)),
    );
    metrics.insert(
        "citations.resolved".to_string(),
        MetricValue::Ratio(ratio_metric(citation_resolved_n, citation_total)),
    );
    metrics.insert(
        "citations.no_source".to_string(),
        MetricValue::Ratio(ratio_metric(citation_no_source_n, citation_total)),
    );
    metrics.insert(
        "citations.no_paragraph".to_string(),
        MetricValue::Ratio(ratio_metric(citation_no_paragraph_n, citation_total)),
    );
    metrics.insert(
        "citations.other".to_string(),
        MetricValue::Ratio(ratio_metric(citation_other_n, citation_total)),
    );
    metrics.insert(
        "citations.section_match".to_string(),
        MetricValue::Ratio(ratio_metric(
            citation_section_matched,
            citation_section_checked,
        )),
    );
    metrics.insert(
        "citations.quote_match".to_string(),
        MetricValue::Ratio(ratio_metric(citation_quote_matched, citation_quote_checked)),
    );
    metrics.insert(
        "latency.citation_ms".to_string(),
        MetricValue::Distribution(Distribution::from_samples(citation_latencies)),
    );
    metrics.insert(
        "lanes.structural_hit".to_string(),
        MetricValue::Ratio(ratio_metric(structural_hit_n, lane_cross_n)),
    );
    metrics.insert(
        "lanes.passage_hit".to_string(),
        MetricValue::Ratio(ratio_metric(passage_hit_n, lane_cross_n)),
    );
    metrics.insert(
        "lanes.both".to_string(),
        MetricValue::Ratio(ratio_metric(both_n, lane_cross_n)),
    );
    metrics.insert(
        "lanes.neither".to_string(),
        MetricValue::Ratio(ratio_metric(neither_n, lane_cross_n)),
    );
    // #308 (ADR 0006 §14)
    metrics.insert(
        "diversity.sources".to_string(),
        MetricValue::Distribution(Distribution::from_samples(diversity_sources)),
    );
    metrics.insert(
        "latency.evidence_ms".to_string(),
        MetricValue::Distribution(Distribution::from_samples(evidence_latencies)),
    );
    metrics.insert(
        "budget.items_used".to_string(),
        MetricValue::Distribution(Distribution::from_samples(budget_items_used)),
    );
    metrics.insert(
        "budget.bytes_used".to_string(),
        MetricValue::Distribution(Distribution::from_samples(budget_bytes_used)),
    );
    metrics.insert(
        "budget.tokens_used".to_string(),
        MetricValue::Distribution(Distribution::from_samples(budget_tokens_used)),
    );
    metrics.insert(
        "budget.omitted_rate".to_string(),
        MetricValue::Ratio(ratio_metric(budget_omitted_n, budget_considered_n)),
    );
    metrics.insert(
        "rerank.ran".to_string(),
        MetricValue::Ratio(ratio_metric(rerank_ran_n, rerank_requested_n)),
    );
    metrics
}

// ============================= Output artifact =============================

fn write_evaluation(path: &Path, evaluation: &EvaluationFile) -> Result<(), String> {
    let text = serde_json::to_string_pretty(evaluation).expect("an evaluation file serializes");
    crate::storage::write_atomic(path, text.as_bytes())
        .map_err(|error| format!("writing {}: {error}", path.display()))
}

/// #308 (ADR 0006 §14): the assembly-lane summary lines — budget
/// usage, source diversity, reranker outcome — that [`print_summary`]
/// appends after the citation block. The budget and reranker lines key
/// off the run's own configuration (`inputs.budget` / `inputs.rerank`),
/// so a lane the operator asked for still reports its zero-sample
/// state rather than vanishing — the same "measured, zero samples" vs
/// "does not apply" distinction the artifact's metric shapes draw
/// (see [`Distribution`]'s doc). The diversity line has no configuring
/// flag, so it keys off its metric's sample count instead. A plain
/// baseline run prints nothing new. Returned as lines rather than
/// printed so tests can assert on them without capturing stdout.
fn assembly_summary_lines(inputs: &InputsBlock, metrics: &MetricsMap) -> Vec<String> {
    let mut lines = Vec::new();
    let scalar = |name: &str| metrics.get(name).and_then(MetricValue::scalar);
    if inputs.budget.is_some() {
        let n = metrics
            .get("budget.items_used")
            .map_or(0, MetricValue::sample_size);
        let mean = |name: &str| match scalar(name) {
            Some(value) => format!("{value:.1}"),
            None => "n/a".to_string(),
        };
        let omitted_rate = match scalar("budget.omitted_rate") {
            Some(value) => format!("{value:.3}"),
            None => "n/a".to_string(),
        };
        lines.push(format!(
            "  budget over {n} case(s): mean {} item(s) / {} byte(s) / {} token(s) used, \
             budget-omitted rate {omitted_rate}",
            mean("budget.items_used"),
            mean("budget.bytes_used"),
            mean("budget.tokens_used"),
        ));
    }
    if let Some(diversity) = metrics.get("diversity.sources")
        && diversity.sample_size() > 0
    {
        lines.push(format!(
            "  diversity over {} case(s): mean {:.1} distinct source(s) in admitted evidence",
            diversity.sample_size(),
            diversity.scalar().unwrap_or(0.0),
        ));
    }
    if let Some(model) = &inputs.rerank {
        let (ran, attempted) = match metrics.get("rerank.ran") {
            Some(value) => (value.numerator().unwrap_or(0), value.sample_size()),
            None => (0, 0),
        };
        lines.push(format!(
            "  rerank '{model}' over {attempted} attempted case(s): {ran} ran, {} degraded",
            attempted.saturating_sub(ran)
        ));
    }
    lines
}

fn print_summary(evaluation: &EvaluationFile, masked_url: &str, context: &str) {
    let total = evaluation.cases.len();
    let passage_failed = evaluation
        .cases
        .iter()
        .filter(|case| matches!(case.passage, PassageOutcome::Failed { .. }))
        .count();
    let structural_cases = evaluation
        .cases
        .iter()
        .filter(|case| case.structural.is_some())
        .count();
    let positions = || {
        evaluation
            .cases
            .iter()
            .filter_map(|case| case.structural.as_ref())
            .flat_map(|structural| &structural.associations)
            .flat_map(|assoc| [&assoc.subject, &assoc.label, &assoc.object])
    };
    let ambiguous = positions()
        .filter(|outcome| matches!(outcome, PositionOutcome::Ambiguous { .. }))
        .count();
    let not_found = positions()
        .filter(|outcome| matches!(outcome, PositionOutcome::NotFound { .. }))
        .count();

    let recalls: Vec<&RecallBlock> = evaluation
        .cases
        .iter()
        .filter_map(|c| c.recall.as_ref())
        .collect();
    let cross: Vec<&LaneCrossBlock> = evaluation
        .cases
        .iter()
        .filter_map(|c| c.lane_cross.as_ref())
        .collect();

    println!("taguru evaluate: {masked_url} / context '{context}'");
    println!(
        "  {total} case(s) — passage: {} ok, {passage_failed} failed; structural lane ran on \
         {structural_cases} case(s) ({ambiguous} ambiguous, {not_found} not-found position(s))",
        total - passage_failed
    );
    if !recalls.is_empty() {
        let mean = |pick: fn(&&RecallBlock) -> f64| {
            recalls.iter().map(pick).sum::<f64>() / recalls.len() as f64
        };
        println!(
            "  recall over {} case(s): mean recall@k {:.3}, mean MRR {:.3}, mean nDCG {:.3}",
            recalls.len(),
            mean(|r| r.recall_at_k),
            mean(|r| r.mrr),
            mean(|r| r.ndcg)
        );
    }
    if !cross.is_empty() {
        let both = cross
            .iter()
            .filter(|c| c.structural_hit && c.passage_hit)
            .count();
        let neither = cross
            .iter()
            .filter(|c| !c.structural_hit && !c.passage_hit)
            .count();
        println!(
            "  lane cross-tab over {} case(s) declaring both a structural and a source \
             expectation: {both} both, {neither} neither",
            cross.len()
        );
    }
    let citations: Vec<&CitationsBlock> = evaluation
        .cases
        .iter()
        .filter_map(|c| c.citations.as_ref())
        .collect();
    if !citations.is_empty() {
        let mean = |pick: fn(&&CitationsBlock) -> f64| {
            citations.iter().map(pick).sum::<f64>() / citations.len() as f64
        };
        println!(
            "  citations over {} case(s): mean recall {:.3}, mean locator validity {:.3}",
            citations.len(),
            mean(|c| c.recall.value),
            mean(|c| c.validity.value)
        );
        // ADR 0004 §11: the artifact never carries served paragraph
        // text, so a reader who wants to see why a locator is invalid
        // re-runs this exact call themselves.
        let invalid: Vec<&CitationCheck> = citations
            .iter()
            .flat_map(|block| block.checks.iter())
            .filter(|check| !citation_is_valid(check))
            .collect();
        for check in invalid.iter().take(3) {
            println!(
                "    invalid locator — reproduce with: POST /contexts/{context}/citations \
                 {{\"source\": \"{}\", \"paragraph\": {}}}",
                check.source, check.paragraph
            );
        }
        if invalid.len() > 3 {
            println!("    ... and {} more invalid locator(s)", invalid.len() - 3);
        }
    }
    for line in assembly_summary_lines(&evaluation.inputs, &evaluation.metrics) {
        println!("{line}");
    }
    if !evaluation.corpus.stable {
        println!(
            "  WARNING: the context's revision changed during this run — a write landed \
             mid-run (corpus.revision_before != corpus.revision_after)"
        );
    }
    println!("  wrote {}", evaluation.inputs.out);
}

/// The human-readable failure summary #276's completion condition
/// requires (in the spirit of `assert_golden_recall`'s "every case
/// stays satisfied" table). Every `aggregate`/`corpus` violation
/// prints in full — there are at most one per declared bound — but
/// per-case violations are capped at 10 lines with the dropped count
/// stated (ADR 0004 §11's "no silent truncation" discipline, the same
/// one `missed[]`/`missed_truncated` already follows).
fn print_threshold_summary(report: &ThresholdReport) {
    if report.passed {
        println!(
            "  thresholds: PASS (sha256 {}…)",
            &report.sha256[..12.min(report.sha256.len())]
        );
        return;
    }
    println!(
        "  thresholds: FAIL — {} violation(s) (sha256 {}…)",
        report.violations.len(),
        &report.sha256[..12.min(report.sha256.len())]
    );
    let (case_scoped, other): (Vec<_>, Vec<_>) = report
        .violations
        .iter()
        .partition(|violation| violation.scope == "case");
    for violation in &other {
        println!(
            "    [{}] {}: {}",
            violation.scope, violation.metric, violation.reason
        );
    }
    const MAX_CASE_LINES: usize = 10;
    for violation in case_scoped.iter().take(MAX_CASE_LINES) {
        println!(
            "    [case {}] {}: {}",
            violation.case_id.as_deref().unwrap_or("?"),
            violation.metric,
            violation.reason
        );
    }
    if case_scoped.len() > MAX_CASE_LINES {
        println!(
            "    ... and {} more case-scoped violation(s)",
            case_scoped.len() - MAX_CASE_LINES
        );
    }
    if !report.skipped.is_empty() {
        println!(
            "  {} case-scoped threshold(s) skipped as not applicable",
            report.skipped.len()
        );
    }
}

mod compare;
mod definitions;
mod evidence;
mod lanes;
mod model;
mod thresholds;

#[cfg(test)]
mod tests;

pub(crate) use definitions::build_definitions;
pub(crate) use lanes::{
    association_coverage, build_case_block, build_missed, citation_is_valid, coverage_counts,
    position_latency_ms, query_latency_ms, runs_structural_lane, score_citation_recall,
    score_citation_validity, score_recall,
};
/// Exercised directly by `tests.rs` (unit tests for the lane runners'
/// pure decision logic) but never called from this hub's own
/// production code, which only ever reaches them indirectly through
/// [`build_case_block`].
#[cfg(test)]
pub(crate) use lanes::{
    check_section, classify_position, extract_passages, quote_matches, served_locators, top_tier,
};
pub(crate) use model::{
    AssociationProbe, AttributionLocator, BudgetAccounting, CaseBlock, CitationCheck,
    CitationOutcome, CitationRecallBlock, CitationValidityBlock, CitationsBlock, CorpusBlock,
    CoverageBlock, CoverageCounts, CueResolution, EmbeddingsBlock, EvalInputsBlock, EvaluationFile,
    EvidenceOutcome, HitLocator, InputsBlock, LaneCrossBlock, MatchingBlock, PassageOutcome,
    PositionOutcome, QueryProbe, QuoteCheck, RecallBlock, SectionCheck, StructuralBlock,
    evidence_locators, hits_from_evidence_items,
};
