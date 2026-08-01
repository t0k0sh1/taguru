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
/// [`run_passage_lane`]/[`evidence::run_evidence_lane`] — resolved
/// once from [`EvaluateArgs`] rather than re-derived per case.
struct RunConfig<'a> {
    assembly: bool,
    /// Always a resolved value (defaulted when no `--max-*` flag was
    /// given) — `assembly` mode always sends it; `budget_given` below
    /// gates whether `baseline` mode uses it at all.
    limits: BudgetLimits,
    /// Whether `--max-items`/`--max-bytes`/`--max-tokens` was actually
    /// given. In `assembly` mode this only decides between the
    /// caller's own ceilings and the server's own defaults — the
    /// request always carries a `budget` object either way. In
    /// `baseline` mode it decides whether truncation (and a `budget`
    /// block in the artifact) happens at all.
    budget_given: bool,
    rerank: Option<&'a str>,
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

// ============================== Passage lane ==============================

fn run_passage_lane(api: &Api, context: &str, case: &EvalCase, limit: usize) -> PassageOutcome {
    match fetch_passage_hits(api, context, case, limit) {
        Ok((hits, plan, latency_ms)) => PassageOutcome::Searched {
            plan,
            hits: hits.into_iter().map(HitLocator::from).collect(),
            latency_ms,
        },
        Err((message, latency_ms)) => PassageOutcome::Failed {
            message,
            latency_ms,
        },
    }
}

/// The passage lane's raw fetch, kept separate from [`run_passage_lane`]
/// so #308's `baseline` truncation path
/// ([`evidence::truncate_to_budget`]) can measure each hit's real
/// `PassageHit.text` byte/token cost — the same content
/// `crate::api::evidence::budget` accounts for server-side — before
/// [`HitLocator::from`] strips it. Truncating already-stripped
/// locators would undercount every hit's true size and make the
/// "equal budget" comparison dishonest.
/// `Ok((hits, plan, latency_ms))` on success, `Err((message,
/// latency_ms))` on failure — named so [`fetch_passage_hits`]'s
/// signature reads clearly instead of tripping clippy's
/// `type_complexity` lint on the raw nested tuple type.
type PassageFetch = Result<(Vec<PassageHit>, Option<SearchContextPlan>, u64), (String, u64)>;

fn fetch_passage_hits(api: &Api, context: &str, case: &EvalCase, limit: usize) -> PassageFetch {
    let body = serde_json::json!({
        "query": case.query,
        "limit": limit,
        "semantic_floor": case.options.floor,
        "tags": case.options.tags,
        "since": case.options.since,
        "until": case.options.until,
    });
    let started_at = Instant::now();
    match api.post(&["contexts", context, "sources", "search"], &body) {
        Ok(value) => {
            let latency_ms = elapsed_ms(started_at);
            extract_passages(&value)
                .map(|(hits, plan)| (hits, plan, latency_ms))
                .map_err(|message| (message, latency_ms))
        }
        Err(message) => Err((truncate_message(&message), elapsed_ms(started_at))),
    }
}

/// Prefers the real `{plan, hits}` shape (`PassagePage`, made
/// `Deserialize` for exactly this purpose by #282); falls back to
/// pulling `source`/`paragraph`/`score` out of a bare hits array or an
/// object whose `hits` don't fit `PassageHit`'s lane-carrying shape —
/// an older or otherwise-nonconforming server, matching
/// `benchmark/search.rs`'s own `extract_hits`.
fn extract_passages(value: &Value) -> Result<(Vec<PassageHit>, Option<SearchContextPlan>), String> {
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
        hits.push(PassageHit {
            source,
            paragraph,
            score,
            // `HitLocator::from` still strips this before anything
            // reaches evaluation.json (ADR 0004 §11 — no corpus body
            // text written to disk), but #308's `--max-bytes`/
            // `--max-tokens` truncation (`evidence::truncate_to_budget`)
            // measures this same field's real byte/token cost BEFORE
            // that stripping happens. Recovering it here when the raw
            // hit still carries it (the common case even when the
            // whole response fails to match `PassagePage`'s stricter
            // lane shape) keeps that measurement honest; a server that
            // omits `text` entirely still degrades to an undercount,
            // which is unavoidable with no data to measure.
            text: raw
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            lanes: PassageLanes {
                bm25: None,
                vector: None,
            },
        });
    }
    Ok((hits, None))
}

// ============================= Structural lane =============================

fn runs_structural_lane(case: &EvalCase) -> bool {
    !case.expected_concepts.is_empty()
        || !case.expected_labels.is_empty()
        || !case.expected_associations.is_empty()
}

fn build_structural_block(api: &Api, context: &str, case: &EvalCase) -> StructuralBlock {
    let cues: Vec<String> = if case.cues.is_empty() {
        vec![case.query.clone()]
    } else {
        case.cues.clone()
    };

    let mut cue_resolutions = Vec::new();
    if !case.expected_concepts.is_empty() {
        for cue in &cues {
            cue_resolutions.push(resolve_cue(api, context, cue, "concept", false));
        }
    }
    if !case.expected_labels.is_empty() {
        for cue in &cues {
            cue_resolutions.push(resolve_cue(api, context, cue, "label", true));
        }
    }

    let associations = case
        .expected_associations
        .iter()
        .map(|assoc| build_association_probe(api, context, assoc))
        .collect();

    StructuralBlock {
        cues: cue_resolutions,
        associations,
    }
}

fn call_resolve(
    api: &Api,
    context: &str,
    cue: &str,
    labels: bool,
    limit: usize,
) -> Result<(Vec<TieredResolution>, u64), String> {
    let endpoint = if labels { "resolve_label" } else { "resolve" };
    let body = serde_json::json!({ "cue": cue, "limit": limit });
    let started_at = Instant::now();
    let value = api.post(&["contexts", context, endpoint], &body)?;
    let latency_ms = elapsed_ms(started_at);
    let resolved: Vec<TieredResolution> = serde_json::from_value(value)
        .map_err(|error| format!("{endpoint} response did not parse: {error}"))?;
    Ok((resolved, latency_ms))
}

/// ADR 0004 §7 step 1: resolve tiers are not comparable, so a caller
/// only ever reads ONE tier of a response — never lexical and semantic
/// candidates mixed. Lexical candidates always sort first when both
/// tiers are present (`resolve.rs`'s own `merge_tiers`), so the rule
/// is: the lexical group when it is non-empty, the semantic group
/// otherwise.
fn top_tier(candidates: &[TieredResolution]) -> Vec<&TieredResolution> {
    let lexical: Vec<&TieredResolution> =
        candidates.iter().filter(|c| c.tier == "lexical").collect();
    if !lexical.is_empty() {
        lexical
    } else {
        candidates.iter().filter(|c| c.tier == "semantic").collect()
    }
}

fn resolve_cue(
    api: &Api,
    context: &str,
    cue: &str,
    kind: &'static str,
    labels: bool,
) -> CueResolution {
    let started_at = Instant::now();
    match call_resolve(api, context, cue, labels, RESOLVE_LIMIT) {
        Ok((resolved, latency_ms)) => {
            let group = top_tier(&resolved);
            CueResolution {
                cue: cue.to_string(),
                kind,
                tier: group.first().map(|candidate| candidate.tier.clone()),
                resolved_names: group.into_iter().map(|c| c.name.clone()).collect(),
                limit: RESOLVE_LIMIT,
                latency_ms,
                error: None,
            }
        }
        Err(message) => CueResolution {
            cue: cue.to_string(),
            kind,
            resolved_names: Vec::new(),
            tier: None,
            limit: RESOLVE_LIMIT,
            latency_ms: elapsed_ms(started_at),
            error: Some(truncate_message(&message)),
        },
    }
}

/// ADR 0004 §7 step 2's stricter policy: unlike coverage's "expand the
/// whole top tier," pinning a `/query` triple needs exactly one stored
/// name per position. Zero candidates in the top tier is `not_found`;
/// two or more is `ambiguous` — either way `query` is never called for
/// that position, and no combination is guessed at.
fn resolve_position(api: &Api, context: &str, cue: &str, labels: bool) -> PositionOutcome {
    let started_at = Instant::now();
    match call_resolve(api, context, cue, labels, RESOLVE_LIMIT) {
        Ok((resolved, latency_ms)) => classify_position(top_tier(&resolved), latency_ms),
        Err(message) => PositionOutcome::Errored {
            message: truncate_message(&message),
            latency_ms: elapsed_ms(started_at),
        },
    }
}

/// The pure decision behind [`resolve_position`], split out so the
/// multi-candidate policy (ADR 0004 §7 step 2) is unit-testable
/// without a network round trip: exactly one top-tier candidate pins
/// the position, zero is `not_found`, several is `ambiguous`.
fn classify_position(group: Vec<&TieredResolution>, latency_ms: u64) -> PositionOutcome {
    match group.len() {
        0 => PositionOutcome::NotFound { latency_ms },
        1 => PositionOutcome::Resolved {
            name: group[0].name.clone(),
            tier: group[0].tier.clone(),
            latency_ms,
        },
        _ => PositionOutcome::Ambiguous {
            tier: group[0].tier.clone(),
            candidates: group.iter().map(|c| c.name.clone()).collect(),
            latency_ms,
        },
    }
}

fn position_name(outcome: &PositionOutcome) -> Option<&str> {
    match outcome {
        PositionOutcome::Resolved { name, .. } => Some(name.as_str()),
        _ => None,
    }
}

fn position_latency_ms(outcome: &PositionOutcome) -> u64 {
    match outcome {
        PositionOutcome::Resolved { latency_ms, .. }
        | PositionOutcome::NotFound { latency_ms }
        | PositionOutcome::Ambiguous { latency_ms, .. }
        | PositionOutcome::Errored { latency_ms, .. } => *latency_ms,
    }
}

/// `query` pins all three positions exactly (ADR 0004 §7 step 2) —
/// `limit: 1` is enough since a pinned triple's `total` is 0 or 1; a
/// single match can still carry several `attributions[]` (the same
/// triple asserted from more than one source), which citation recall
/// (ADR 0004 §8) reads as served `(source, paragraph)` locators
/// alongside the passage lane's own hits.
fn run_query(api: &Api, context: &str, subject: &str, label: &str, object: &str) -> QueryProbe {
    let body = serde_json::json!({
        "subject": subject,
        "label": label,
        "object": object,
        "limit": 1,
    });
    let started_at = Instant::now();
    match api.post(&["contexts", context, "query"], &body) {
        Ok(value) => {
            let latency_ms = elapsed_ms(started_at);
            match serde_json::from_value::<MatchPage>(value) {
                Ok(page) => {
                    let attributions = page
                        .matches
                        .iter()
                        .flat_map(|matched| matched.attributions.iter())
                        .map(|attribution| AttributionLocator {
                            source: attribution.source.clone(),
                            paragraph: attribution.paragraph,
                        })
                        .collect();
                    QueryProbe::Queried {
                        total: page.total,
                        matches: page.matches.len(),
                        attributions,
                        latency_ms,
                    }
                }
                Err(error) => QueryProbe::Errored {
                    message: format!("query response did not parse: {error}"),
                    latency_ms,
                },
            }
        }
        Err(message) => QueryProbe::Errored {
            message: truncate_message(&message),
            latency_ms: elapsed_ms(started_at),
        },
    }
}

fn query_latency_ms(probe: &QueryProbe) -> u64 {
    match probe {
        QueryProbe::Queried { latency_ms, .. } | QueryProbe::Errored { latency_ms, .. } => {
            *latency_ms
        }
    }
}

fn build_association_probe(
    api: &Api,
    context: &str,
    assoc: &ExpectedAssociation,
) -> AssociationProbe {
    let subject = resolve_position(api, context, &assoc.subject, false);
    let label = resolve_position(api, context, &assoc.label, true);
    let object = resolve_position(api, context, &assoc.object, false);

    let query = match (
        position_name(&subject),
        position_name(&label),
        position_name(&object),
    ) {
        (Some(s), Some(l), Some(o)) => Some(run_query(api, context, s, l, o)),
        _ => None,
    };

    AssociationProbe {
        subject_cue: assoc.subject.clone(),
        label_cue: assoc.label.clone(),
        object_cue: assoc.object.clone(),
        subject,
        label,
        object,
        query,
    }
}

// ============================ Citation lane ============================

fn runs_citation_lane(case: &EvalCase) -> bool {
    !case.expected_citations.is_empty()
}

/// One `POST /contexts/{name}/citations` call per `expected_citations[]`
/// entry, strictly sequential — the endpoint takes exactly one locator
/// per request (`src/api/sources.rs:67-74`), never a batch, so N
/// expectations cost N round trips. Deliberately NOT preflighted the
/// way `expected_sources` is (`run_evaluate`'s missing-source check
/// above): a citation naming a source or paragraph the corpus does not
/// carry is exactly the failure ADR 0004 §8's locator-validity
/// measurement exists to catch, so it must run to completion and be
/// recorded — never abort the whole run. This also means the lane runs
/// even on a case whose passage-lane search missed entirely: citation
/// recall and locator validity are measured independently.
fn run_citation_lane(
    api: &Api,
    context: &str,
    case: &EvalCase,
    served: &BTreeSet<(String, u32)>,
) -> Vec<CitationCheck> {
    case.expected_citations
        .iter()
        .map(|expected| check_citation(api, context, expected, served))
        .collect()
}

fn check_citation(
    api: &Api,
    context: &str,
    expected: &ExpectedCitation,
    served: &BTreeSet<(String, u32)>,
) -> CitationCheck {
    let body = serde_json::json!({
        "source": expected.source,
        "paragraph": expected.paragraph,
    });
    let started_at = Instant::now();
    let outcome = match api.post_envelope(&["contexts", context, "citations"], &body) {
        Ok(value) => match serde_json::from_value::<Citation>(value) {
            Ok(citation) => {
                let section = check_section(&expected.section, &citation.section);
                let quote = expected.quote.as_ref().map(|quote| QuoteCheck {
                    matched: quote_matches(quote, &citation.text),
                    declared: quote.clone(),
                });
                CitationOutcome::Resolved { section, quote }
            }
            Err(error) => CitationOutcome::Unresolved {
                code: None,
                message: truncate_message(&format!("citation response did not parse: {error}")),
            },
        },
        Err(ApiFailure::NotFound { code, message }) => CitationOutcome::Unresolved {
            code,
            message: truncate_message(&message),
        },
        Err(ApiFailure::Other(message)) => CitationOutcome::Unresolved {
            code: None,
            message: truncate_message(&message),
        },
    };
    CitationCheck {
        source: expected.source.clone(),
        paragraph: expected.paragraph,
        served: served.contains(&(expected.source.clone(), expected.paragraph)),
        outcome,
        latency_ms: elapsed_ms(started_at),
    }
}

// ================================ Scoring ==================================
//
// Everything below reads only rank (a hit's position in `hits[]`, a
// cue's own resolved-name list, a query's `total`) and label
// (`expected_sources`' `relevance`, `expected_concepts`/
// `expected_labels`/`expected_associations`) — never a `HitLocator`'s
// or `TieredResolution`'s own `score` field (#215's "do not collapse
// incomparable raw scores from graph, BM25, and vector lanes into one
// invented scale" requirement). Pure functions, no network access, so
// every rule here is unit-testable without a server.

/// One case's recall@k/MRR/nDCG against `expected_sources` (ADR 0004
/// §274). `None` when the case declares no `relevance >= 1` source —
/// `relevance == 0` means "not evidence for this case," dropped from
/// the denominator like `benchmark::search`'s own `resolve_expected_items`
/// (`search.rs:1014-1016`).
/// Whether `hit` satisfies `expected`: same `source`, and either
/// `expected.paragraphs` is empty (any paragraph of this source
/// answers the case) or it names `hit.paragraph` explicitly. The one
/// place this policy lives — [`score_recall`] and [`build_missed`]
/// both call it, so a future change to source matching cannot diverge
/// between "how a case scores" and "why a case says it missed."
fn source_matches(expected: &ExpectedSource, hit: &HitLocator) -> bool {
    hit.source == expected.source
        && (expected.paragraphs.is_empty() || expected.paragraphs.contains(&hit.paragraph))
}

fn score_recall(expected_sources: &[ExpectedSource], hits: &[HitLocator]) -> Option<RecallBlock> {
    let items: Vec<&ExpectedSource> = expected_sources
        .iter()
        .filter(|expected| expected.relevance >= 1)
        .collect();
    if items.is_empty() {
        return None;
    }

    let matched = items
        .iter()
        .filter(|item| hits.iter().any(|hit| source_matches(item, hit)))
        .count();

    let mut mrr = 0.0;
    for (rank, hit) in hits.iter().enumerate() {
        if items.iter().any(|item| source_matches(item, hit)) {
            mrr = 1.0 / (rank as f64 + 1.0);
            break;
        }
    }

    // One credit per expectation, at the rank of the first hit that
    // satisfies it — not one credit per matching hit. A wildcard entry
    // (`paragraphs` empty) that lands on several hits is not double
    // counted, so IDCG is well-defined from the label multiset alone
    // regardless of how many hits happen to carry one label.
    let dcg: f64 = items
        .iter()
        .filter_map(|item| {
            hits.iter()
                .position(|hit| source_matches(item, hit))
                .map(|rank| item.relevance as f64 / (rank as f64 + 2.0).log2())
        })
        .sum();

    let mut ideal: Vec<u8> = items.iter().map(|item| item.relevance).collect();
    ideal.sort_unstable_by(|a, b| b.cmp(a));
    let idcg: f64 = ideal
        .iter()
        .enumerate()
        .map(|(rank, relevance)| *relevance as f64 / (rank as f64 + 2.0).log2())
        .sum();
    // IDCG assigns each expectation its own distinct rank, but DCG does
    // not: two expectations satisfied by the same hit (e.g. a wildcard
    // `paragraphs: []` entry and a paragraph-restricted entry on the
    // same source) both credit at that hit's one rank, so DCG can
    // exceed IDCG. Clamped to keep this a well-formed 0..=1 ratio.
    let ndcg = if idcg > 0.0 {
        (dcg / idcg).min(1.0)
    } else {
        0.0
    };

    Some(RecallBlock {
        recall_at_k: matched as f64 / items.len() as f64,
        mrr,
        ndcg,
        expected_total: items.len(),
        matched,
    })
}

/// Every `(source, paragraph)` this case's served results carry (ADR
/// 0004 §8's citation recall): the passage lane's hits up to its own
/// `limit`, plus the structural lane's `AttributionOut` locators when
/// it ran — never one lane's own raw score, only rank-independent
/// membership. An attribution with no `paragraph` locator at all
/// contributes nothing, since `expected_citations` always names one.
fn served_locators(
    passage: &PassageOutcome,
    structural: Option<&StructuralBlock>,
) -> BTreeSet<(String, u32)> {
    let mut served = BTreeSet::new();
    if let PassageOutcome::Searched { hits, .. } = passage {
        served.extend(hits.iter().map(|hit| (hit.source.clone(), hit.paragraph)));
    }
    if let Some(structural) = structural {
        served.extend(
            structural
                .associations
                .iter()
                .filter_map(|assoc| assoc.query.as_ref())
                .flat_map(|query| {
                    let attributions: &[AttributionLocator] = match query {
                        QueryProbe::Queried { attributions, .. } => attributions,
                        QueryProbe::Errored { .. } => &[],
                    };
                    attributions.iter()
                })
                .filter_map(|locator| locator.paragraph.map(|p| (locator.source.clone(), p))),
        );
    }
    served
}

/// [`ExpectedCitation::section`]'s three-valued check (ADR 0004 §8): an
/// absent key (`None`) means "don't check section" and never runs a
/// comparison; a present key — including an explicit `null` — is
/// compared against the server's `Citation.section` as-is, so
/// `Some(None)` correctly asserts "outside every stored section."
fn check_section(expected: &Option<Option<String>>, actual: &Option<String>) -> SectionCheck {
    match expected {
        None => SectionCheck::NotChecked,
        Some(expected) if expected == actual => SectionCheck::Matched {
            expected: expected.clone(),
        },
        Some(expected) => SectionCheck::Mismatched {
            expected: expected.clone(),
        },
    }
}

/// Whether `quote` is a substring of `text` after both sides go through
/// `normalize_entry` (ADR 0004 §8 — the same folding the passage index
/// itself uses, trimmed first since `normalize_entry` does not trim,
/// matching [`resolved_contains`]'s own preprocessing). `text` is
/// always exactly one paragraph (`Citation.text`, `src/api/sources.rs:
/// 84-88`), so a `quote` spanning a paragraph boundary can never match
/// here — ADR 0004 §8's documented workaround is splitting it into two
/// `expected_citations` entries.
fn quote_matches(quote: &str, text: &str) -> bool {
    let target = normalize_entry(quote.trim());
    normalize_entry(text.trim()).contains(&target)
}

/// A [`CitationCheck`] is valid (ADR 0004 §8's locator-validity
/// measurement) when it resolved, its `section` check never mismatched
/// (an absent expectation or a match both count), and its `quote` check
/// — when the case declared one — matched.
fn citation_is_valid(check: &CitationCheck) -> bool {
    match &check.outcome {
        CitationOutcome::Resolved { section, quote } => {
            !matches!(section, SectionCheck::Mismatched { .. })
                && quote.as_ref().is_none_or(|quote| quote.matched)
        }
        CitationOutcome::Unresolved { .. } => false,
    }
}

/// Citation recall (ADR 0004 §8): the fraction of `checks` whose
/// `(source, paragraph)` appeared among this case's served results.
/// Callers only call this on a non-empty `checks` — an empty
/// `expected_citations` means the citation lane never ran at all, so
/// there is nothing here to divide by.
fn score_citation_recall(checks: &[CitationCheck]) -> CitationRecallBlock {
    let matched = checks.iter().filter(|check| check.served).count();
    CitationRecallBlock {
        expected_total: checks.len(),
        matched,
        value: matched as f64 / checks.len() as f64,
    }
}

/// Locator validity (ADR 0004 §8): the fraction of `checks` that
/// resolved with a matching `section` (when declared) and `quote` (when
/// declared) — never merged with [`score_citation_recall`]'s own ratio.
fn score_citation_validity(checks: &[CitationCheck]) -> CitationValidityBlock {
    let valid = checks
        .iter()
        .filter(|check| citation_is_valid(check))
        .count();
    CitationValidityBlock {
        expected_total: checks.len(),
        valid,
        value: valid as f64 / checks.len() as f64,
    }
}

/// Whether `expected` (case-declared) appears among `resolutions`'
/// `resolved_names[]`, matched with `normalize_entry` on both sides
/// (ADR 0004 §8) — `normalize_entry` does not trim, so both sides are
/// trimmed first.
fn resolved_contains(resolutions: &[&CueResolution], expected: &str) -> bool {
    let target = normalize_entry(expected.trim());
    resolutions
        .iter()
        .flat_map(|resolution| &resolution.resolved_names)
        .any(|name| normalize_entry(name.trim()) == target)
}

/// Coverage of one expectation list (`expected_concepts` or
/// `expected_labels`) against the matching-kind `CueResolution`s.
/// `None` when the case declares none — coverage does not apply, as
/// distinct from applying and finding zero.
fn coverage_counts(expected: &[String], resolutions: &[&CueResolution]) -> Option<CoverageCounts> {
    if expected.is_empty() {
        return None;
    }
    let matched = expected
        .iter()
        .filter(|item| resolved_contains(resolutions, item))
        .count();
    Some(CoverageCounts {
        expected: expected.len(),
        matched,
        value: matched as f64 / expected.len() as f64,
    })
}

/// Coverage of `expected_associations`: an entry counts as covered
/// only when its `/query` call ran and returned `total >= 1` — the
/// same "exactly one candidate pins a position, otherwise `query` is
/// never called" policy (ADR 0004 §7 step 2) means `not_found`/
/// `ambiguous`/`Errored` positions and a never-run `query` are all
/// uncovered, never guessed at.
fn association_coverage(associations: &[AssociationProbe]) -> Option<CoverageCounts> {
    if associations.is_empty() {
        return None;
    }
    let matched = associations
        .iter()
        .filter(
            |probe| matches!(&probe.query, Some(QueryProbe::Queried { total, .. }) if *total >= 1),
        )
        .count();
    Some(CoverageCounts {
        expected: associations.len(),
        matched,
        value: matched as f64 / associations.len() as f64,
    })
}

/// `missed[]`, capped at 3 entries with a count of the entries
/// *dropped* — not the total (ADR 0004 §11). Order: sources, concepts,
/// labels, associations, citations. Silent about sources when the
/// passage lane failed — its own `Failed` outcome and
/// `passage.failure_rate` already record that; guessing at which
/// sources it would have missed adds no information. Citation checks
/// run regardless of the passage lane's own outcome (ADR 0004 §8), so
/// they are never silenced that way.
fn build_missed(
    case: &EvalCase,
    hits: Option<&[HitLocator]>,
    concept_resolutions: &[&CueResolution],
    label_resolutions: &[&CueResolution],
    structural: Option<&StructuralBlock>,
    citation_checks: &[CitationCheck],
) -> (Vec<String>, usize) {
    let mut all = Vec::new();

    if let Some(hits) = hits {
        for expected in &case.expected_sources {
            if expected.relevance == 0 {
                continue;
            }
            let hit = hits.iter().any(|h| source_matches(expected, h));
            if !hit {
                all.push(format!(
                    "expected_sources: '{}' not found among passage hits",
                    expected.source
                ));
            }
        }
    }

    for concept in &case.expected_concepts {
        if !resolved_contains(concept_resolutions, concept) {
            all.push(format!("expected_concepts: '{concept}' not resolved"));
        }
    }
    for label in &case.expected_labels {
        if !resolved_contains(label_resolutions, label) {
            all.push(format!("expected_labels: '{label}' not resolved"));
        }
    }
    if let Some(structural) = structural {
        for assoc in &structural.associations {
            let why: Option<String> = match &assoc.query {
                Some(QueryProbe::Queried { total, .. }) if *total >= 1 => None,
                Some(QueryProbe::Queried { .. }) => {
                    Some("query returned no association".to_string())
                }
                Some(QueryProbe::Errored { .. }) => Some("query errored".to_string()),
                None => {
                    // A position's own `/resolve` call failing outright
                    // (`PositionOutcome::Errored`, a transport-level
                    // fault) is never the same fact as zero/multiple
                    // candidates coming back cleanly — the former is
                    // possibly-transient infra trouble, the latter a
                    // data-quality miss. Keep them distinguishable in
                    // `missed[]` rather than collapsing both into "a
                    // position did not pin."
                    let errored_positions: Vec<(&str, &str)> = [
                        ("subject", &assoc.subject),
                        ("label", &assoc.label),
                        ("object", &assoc.object),
                    ]
                    .into_iter()
                    .filter_map(|(name, outcome)| match outcome {
                        PositionOutcome::Errored { message, .. } => Some((name, message.as_str())),
                        _ => None,
                    })
                    .collect();
                    if errored_positions.is_empty() {
                        Some("query never ran (a position did not pin)".to_string())
                    } else {
                        Some(format!(
                            "a position could not be resolved due to a transport error: {}",
                            errored_positions
                                .iter()
                                .map(|(name, message)| format!("{name}: {message}"))
                                .collect::<Vec<_>>()
                                .join("; ")
                        ))
                    }
                }
            };
            if let Some(why) = why {
                all.push(format!(
                    "expected_associations: ({}, {}, {}) {why}",
                    assoc.subject_cue, assoc.label_cue, assoc.object_cue
                ));
            }
        }
    }

    // ADR 0004 §8's two citation measurements are never merged into one
    // score, so a bad check can add up to two distinct entries here —
    // one for recall (not served), one for validity (locator itself
    // wrong) — never collapsed into a single ambiguous message.
    for check in citation_checks {
        if !check.served {
            all.push(format!(
                "expected_citations: ({}, {}) not found among served results",
                check.source, check.paragraph
            ));
        }
        if !citation_is_valid(check) {
            all.push(format!(
                "expected_citations: ({}, {}) failed locator validity",
                check.source, check.paragraph
            ));
        }
    }

    let truncated = all.len().saturating_sub(3);
    all.truncate(3);
    (all, truncated)
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

// ================================ Per case ================================

/// #308 (ADR 0006 §14): builds the passage lane's `PassageOutcome`
/// (what every scoring function below reads) plus the two mode-
/// specific diagnostics — `EvidenceOutcome` (`assembly` only) and
/// `BudgetAccounting` (either mode, once a budget flag was given).
/// `assembly` mode always sends `run_config.limits` to
/// `POST /contexts/{name}/evidence` (that endpoint has no unbudgeted
/// mode, ADR 0006 §8); `baseline` mode only truncates — and only then
/// carries a `budget` block at all — when `run_config.budget_given`.
fn run_passage_or_evidence_lane(
    api: &Api,
    context: &str,
    case: &EvalCase,
    limit: usize,
    run_config: &RunConfig,
) -> (
    PassageOutcome,
    Option<EvidenceOutcome>,
    Option<BudgetAccounting>,
) {
    if run_config.assembly {
        match evidence::run_evidence_lane(
            api,
            context,
            case,
            limit,
            &run_config.limits,
            run_config.rerank,
        ) {
            evidence::LaneResult::Assembled {
                package,
                latency_ms,
            } => {
                let hits = hits_from_evidence_items(&package.items, limit);
                let locators = evidence_locators(&package.items);
                let passage = PassageOutcome::Searched {
                    plan: None,
                    hits,
                    latency_ms,
                };
                let budget = BudgetAccounting {
                    usage: package.budget,
                    omitted_total: package.omitted_total,
                };
                let outcome = EvidenceOutcome::Assembled {
                    latency_ms,
                    items: locators,
                    omitted_by_reason: package.omitted_by_reason,
                    selection: package.plan.selection,
                    reranker: package.plan.reranker,
                };
                (passage, Some(outcome), Some(budget))
            }
            evidence::LaneResult::Failed {
                message,
                latency_ms,
            } => {
                let passage = PassageOutcome::Failed {
                    message: message.clone(),
                    latency_ms,
                };
                let outcome = EvidenceOutcome::Failed {
                    message,
                    latency_ms,
                };
                (passage, Some(outcome), None)
            }
        }
    } else if run_config.budget_given {
        match fetch_passage_hits(api, context, case, limit) {
            Ok((hits, plan, latency_ms)) => {
                let truncated = evidence::truncate_to_budget(context, hits, &run_config.limits);
                let passage = PassageOutcome::Searched {
                    plan,
                    hits: truncated.hits.into_iter().map(HitLocator::from).collect(),
                    latency_ms,
                };
                let budget = BudgetAccounting {
                    usage: truncated.usage,
                    omitted_total: truncated.omitted_total,
                };
                (passage, None, Some(budget))
            }
            Err((message, latency_ms)) => (
                PassageOutcome::Failed {
                    message,
                    latency_ms,
                },
                None,
                None,
            ),
        }
    } else {
        (run_passage_lane(api, context, case, limit), None, None)
    }
}

fn build_case_block(
    api: &Api,
    context: &str,
    case: &EvalCase,
    default_limit: usize,
    run_config: &RunConfig,
) -> CaseBlock {
    let limit = case.options.limit.unwrap_or(default_limit);
    let (passage, evidence_outcome, budget) =
        run_passage_or_evidence_lane(api, context, case, limit, run_config);
    let diversity_sources = match &passage {
        PassageOutcome::Searched { hits, .. } => Some(
            hits.iter()
                .map(|hit| hit.source.as_str())
                .collect::<BTreeSet<_>>()
                .len(),
        ),
        PassageOutcome::Failed { .. } => None,
    };
    let structural = runs_structural_lane(case).then(|| build_structural_block(api, context, case));
    // Citation recall needs to know what the first two lanes already
    // served BEFORE the citation lane's own network calls run, so the
    // per-check `served` bit can be set at request time rather than
    // reconciled afterward.
    let served = served_locators(&passage, structural.as_ref());
    let citation_checks = if runs_citation_lane(case) {
        run_citation_lane(api, context, case, &served)
    } else {
        Vec::new()
    };
    let scores = score_case(case, &passage, structural.as_ref(), &citation_checks);

    CaseBlock {
        case_id: case.case_id.clone(),
        query: case.query.clone(),
        cues: case.cues.clone(),
        limit,
        passage,
        structural,
        recall: scores.recall,
        coverage: scores.coverage,
        lane_cross: scores.lane_cross,
        citations: scores.citations,
        missed: scores.missed,
        missed_truncated: scores.missed_truncated,
        evidence: evidence_outcome,
        budget,
        diversity_sources,
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

fn build_definitions() -> BTreeMap<String, MetricDef> {
    let mut d = BTreeMap::new();
    d.insert(
        "latency.passage_ms".to_string(),
        def(
            "ms",
            "distribution",
            &["case"],
            "Wall-clock round trip of the passage lane's own \
             POST /contexts/{name}/sources/search call.",
            "POST /contexts/{name}/sources/search",
            None,
        ),
    );
    d.insert(
        "latency.resolve_ms".to_string(),
        def(
            "ms",
            "distribution",
            &["case"],
            "Wall-clock round trip of every /resolve and /resolve_label call \
             the structural lane made — coverage cues and association \
             positions alike.",
            "POST /contexts/{name}/resolve, POST /contexts/{name}/resolve_label",
            None,
        ),
    );
    d.insert(
        "latency.query_ms".to_string(),
        def(
            "ms",
            "distribution",
            &["case"],
            "Wall-clock round trip of each expected_associations[] entry's \
             POST /contexts/{name}/query call.",
            "POST /contexts/{name}/query",
            Some(
                "empty when no case declares expected_associations, or none \
                 of them resolved every position to exactly one name",
            ),
        ),
    );
    d.insert(
        "passage.failure_rate".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of cases whose passage lane call did not complete \
             (transport error or an unparseable response).",
            "POST /contexts/{name}/sources/search",
            None,
        ),
    );
    d.insert(
        "structural.case_rate".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of cases that declared expected_concepts, \
             expected_labels, or expected_associations and so ran the \
             structural lane.",
            "eval.jsonl case fields",
            None,
        ),
    );
    d.insert(
        "recall.recall_at_k".to_string(),
        def(
            "ratio",
            "distribution",
            &["case"],
            "Fraction of a case's expected_sources (relevance >= 1) found \
             among the passage lane's hits, up to the case's own limit.",
            "eval.jsonl expected_sources, POST /contexts/{name}/sources/search",
            Some(
                "empty when no case declares a relevance >= 1 expected_sources \
                 entry, or the passage lane failed outright",
            ),
        ),
    );
    d.insert(
        "recall.mrr".to_string(),
        def(
            "ratio",
            "distribution",
            &["case"],
            "1 / rank of the first passage hit that satisfies any of a \
             case's expected_sources entries, 0 if none does.",
            "eval.jsonl expected_sources, POST /contexts/{name}/sources/search",
            Some(
                "empty when no case declares a relevance >= 1 expected_sources \
                 entry, or the passage lane failed outright",
            ),
        ),
    );
    d.insert(
        "recall.ndcg".to_string(),
        def(
            "ratio",
            "distribution",
            &["case"],
            "Graded-relevance nDCG over expected_sources: each expectation \
             contributes its own relevance (0..=3) once, at the rank of the \
             first hit that satisfies it; IDCG orders the case's own \
             expected relevances descending.",
            "eval.jsonl expected_sources.relevance, POST /contexts/{name}/sources/search",
            Some(
                "empty when no case declares a relevance >= 1 expected_sources \
                 entry, or the passage lane failed outright",
            ),
        ),
    );
    d.insert(
        "coverage.concepts".to_string(),
        def(
            "ratio",
            "distribution",
            &["case"],
            "Fraction of a case's expected_concepts found, after \
             normalize_entry folding, among the structural lane's \
             concept-cue resolved_names[].",
            "eval.jsonl expected_concepts, POST /contexts/{name}/resolve",
            Some("empty when no case declares expected_concepts"),
        ),
    );
    d.insert(
        "coverage.labels".to_string(),
        def(
            "ratio",
            "distribution",
            &["case"],
            "Fraction of a case's expected_labels found, after \
             normalize_entry folding, among the structural lane's \
             label-cue resolved_names[].",
            "eval.jsonl expected_labels, POST /contexts/{name}/resolve_label",
            Some("empty when no case declares expected_labels"),
        ),
    );
    d.insert(
        "coverage.associations".to_string(),
        def(
            "ratio",
            "distribution",
            &["case"],
            "Fraction of a case's expected_associations whose /query call \
             ran (all three positions pinned) and returned total >= 1.",
            "eval.jsonl expected_associations, POST /contexts/{name}/query",
            Some(
                "empty when no case declares expected_associations; a \
                 not_found/ambiguous position never runs query and so \
                 counts as uncovered, never guessed at",
            ),
        ),
    );
    d.insert(
        "citations.recall".to_string(),
        def(
            "ratio",
            "distribution",
            &["case"],
            "Fraction of a case's expected_citations whose (source, \
             paragraph) appeared among that case's served results — \
             passage hits up to limit, plus the structural lane's \
             AttributionOut locators when it ran (ADR 0004 §8). Never \
             merged with citations.locator_validity.",
            "eval.jsonl expected_citations, POST /contexts/{name}/sources/search, \
             POST /contexts/{name}/query",
            Some("empty when no case declares expected_citations"),
        ),
    );
    d.insert(
        "citations.locator_validity".to_string(),
        def(
            "ratio",
            "distribution",
            &["case"],
            "Fraction of a case's expected_citations whose \
             POST /contexts/{name}/citations call resolved with a \
             matching section (when declared) and quote (when declared) \
             — computed even for a case whose passage lane missed \
             outright (ADR 0004 §8). Never merged with citations.recall.",
            "POST /contexts/{name}/citations",
            Some("empty when no case declares expected_citations"),
        ),
    );
    d.insert(
        "citations.resolved".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of every POST /contexts/{name}/citations call made \
             across all cases that resolved (neither no_source nor \
             no_paragraph).",
            "POST /contexts/{name}/citations",
            None,
        ),
    );
    d.insert(
        "citations.no_source".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of every citation call that failed with ErrorCode \
             no_source — the expected_citations entry names a source \
             this context does not carry.",
            "POST /contexts/{name}/citations",
            None,
        ),
    );
    d.insert(
        "citations.no_paragraph".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of every citation call that failed with ErrorCode \
             no_paragraph — the expected_citations entry names a \
             paragraph index out of range for its source.",
            "POST /contexts/{name}/citations",
            None,
        ),
    );
    d.insert(
        "citations.other".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of every citation call that neither resolved nor \
             failed with ErrorCode no_source/no_paragraph — a transport \
             failure, an unparsable response, or any other ErrorCode. \
             resolved + no_source + no_paragraph + other always sums \
             to 1.0.",
            "POST /contexts/{name}/citations",
            None,
        ),
    );
    d.insert(
        "citations.section_match".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Of the resolved citation calls whose expected_citations \
             entry declared a section key (an explicit null included), \
             the share whose declared value matched the server's own \
             Citation.section.",
            "eval.jsonl expected_citations.section, POST /contexts/{name}/citations",
            Some("n is 0 when no expected_citations entry declares section"),
        ),
    );
    d.insert(
        "citations.quote_match".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Of the resolved citation calls whose expected_citations \
             entry declared a quote, the share whose declared quote was \
             a normalize_entry-folded substring of the returned text. A \
             quote spanning a paragraph boundary can never match here, \
             since Citation.text is exactly one paragraph.",
            "eval.jsonl expected_citations.quote, POST /contexts/{name}/citations",
            Some("n is 0 when no expected_citations entry declares quote"),
        ),
    );
    d.insert(
        "latency.citation_ms".to_string(),
        def(
            "ms",
            "distribution",
            &["case"],
            "Wall-clock round trip of each expected_citations[] entry's \
             own POST /contexts/{name}/citations call.",
            "POST /contexts/{name}/citations",
            Some("empty when no case declares expected_citations"),
        ),
    );
    d.insert(
        "lanes.structural_hit".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of cases declaring BOTH a structural expectation and a \
             relevance >= 1 source expectation whose structural coverage \
             matched at least one expectation (ADR 0004 §7).",
            "derived from coverage.* and recall.recall_at_k",
            Some(
                "denominator excludes cases with only a structural or only a \
                 source expectation, and any case whose passage lane failed",
            ),
        ),
    );
    d.insert(
        "lanes.passage_hit".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of cases declaring BOTH a structural expectation and a \
             relevance >= 1 source expectation whose passage lane matched \
             at least one expected_sources entry (ADR 0004 §7).",
            "derived from coverage.* and recall.recall_at_k",
            Some(
                "denominator excludes cases with only a structural or only a \
                 source expectation, and any case whose passage lane failed",
            ),
        ),
    );
    d.insert(
        "lanes.both".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of the same denominator as lanes.structural_hit/ \
             lanes.passage_hit where both lanes hit (ADR 0004 §7).",
            "derived from coverage.* and recall.recall_at_k",
            None,
        ),
    );
    d.insert(
        "lanes.neither".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of the same denominator as lanes.structural_hit/ \
             lanes.passage_hit where neither lane hit (ADR 0004 §7). A \
             case counts here purely by rank/label outcome, never by a \
             graph/BM25/vector score value.",
            "derived from coverage.* and recall.recall_at_k",
            None,
        ),
    );
    // #308 (ADR 0006 §14): equal-budget comparison metrics —
    // `diversity.sources` is the one metric #216/ADR 0006 name
    // explicitly ("source diversity at equal evidence budget") that
    // ADR 0004 does not already define; the rest let a `--thresholds`
    // file and `evaluate compare` see budget consumption and reranker
    // degrade rate the same way they already see recall/citations.
    d.insert(
        "diversity.sources".to_string(),
        def(
            "source",
            "distribution",
            &["case"],
            "Count of distinct source locators among a case's admitted \
             evidence — read from the same hits[] recall/citation \
             scoring itself uses: baseline's (possibly budget-truncated) \
             passage hits, or assembly's admitted items' own locators \
             union their citation_refs (an association's corroborating \
             attributions; a passage/community item's own locator, \
             since its citation_refs is always empty by design).",
            "POST /contexts/{name}/sources/search, POST /contexts/{name}/evidence",
            Some("empty when the passage/evidence lane failed outright"),
        ),
    );
    d.insert(
        "latency.evidence_ms".to_string(),
        def(
            "ms",
            "distribution",
            &["case"],
            "Wall-clock round trip of an --assembly run's own \
             POST /contexts/{name}/evidence call.",
            "POST /contexts/{name}/evidence",
            Some("empty in baseline mode"),
        ),
    );
    d.insert(
        "budget.items_used".to_string(),
        def(
            "item",
            "distribution",
            &["case"],
            "A case's own items_used against its resolved budget ceiling \
             — baseline's client-side truncation accounting or \
             assembly's server-returned BudgetUsage, computed with the \
             identical ADR 0006 §8 formula either way.",
            "crate::api::evidence::budget",
            Some("empty when no --max-items/--max-bytes/--max-tokens flag was given"),
        ),
    );
    d.insert(
        "budget.bytes_used".to_string(),
        def(
            "byte",
            "distribution",
            &["case"],
            "A case's own bytes_used against its resolved budget ceiling \
             (ADR 0006 §8: the items array's compact JSON length plus \
             the citations array's, excluding each item's own bytes/ \
             estimated_tokens fields).",
            "crate::api::evidence::budget",
            Some("empty when no --max-items/--max-bytes/--max-tokens flag was given"),
        ),
    );
    d.insert(
        "budget.tokens_used".to_string(),
        def(
            "token",
            "distribution",
            &["case"],
            "A case's own tokens_used against its resolved budget \
             ceiling — ADR 0006 §8's fixed estimator (0.25 tokens per \
             Basic Latin scalar, 1.0 otherwise), never a real tokenizer \
             count.",
            "crate::api::evidence::budget",
            Some("empty when no --max-items/--max-bytes/--max-tokens flag was given"),
        ),
    );
    d.insert(
        "budget.omitted_rate".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of every candidate a budget-truncated case considered \
             (admitted plus omitted, either mode) that a budget ceiling \
             dropped — the equal-budget \"how much got left out\" \
             counterpart to items_used/bytes_used/tokens_used.",
            "crate::api::evidence::budget, POST /contexts/{name}/evidence",
            Some("empty when no --max-items/--max-bytes/--max-tokens flag was given"),
        ),
    );
    d.insert(
        "rerank.ran".to_string(),
        def(
            "ratio",
            "ratio",
            &["run"],
            "Share of --rerank cases whose configured reranker actually \
             reordered the pool (plan.reranker.ran); the complement is \
             the degrade rate — no provider configured, a model \
             mismatch, or any other ADR 0006 §12 fallback reason.",
            "POST /contexts/{name}/evidence plan.reranker",
            Some("empty when --rerank was not given"),
        ),
    );
    d
}

// ============================= Output artifact =============================

fn write_evaluation(path: &Path, evaluation: &EvaluationFile) -> Result<(), String> {
    let text = serde_json::to_string_pretty(evaluation).expect("an evaluation file serializes");
    crate::storage::write_atomic(path, text.as_bytes())
        .map_err(|error| format!("writing {}: {error}", path.display()))
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

// ============================= Value shapes =============================

// `Debug`/`Clone` are dropped from this and every struct that
// transitively embeds `SearchContextPlan`/`PassageLanes` (the real
// server response types, reused verbatim per #282): neither derives
// them, matching this codebase's convention of keeping wire-response
// DTOs lean. Nothing here is ever cloned; `Serialize` is all a
// write-once artifact needs.
#[derive(Serialize)]
struct EvaluationFile {
    taguru_evaluation: u64,
    generated_at: String,
    matching: MatchingBlock,
    inputs: InputsBlock,
    corpus: CorpusBlock,
    /// `None` when `--thresholds` was not given (a report-only run);
    /// `Some` — the loaded bounds checked against this run — otherwise
    /// (ADR 0004 §9.1's threshold identity, extended by #276 with the
    /// pass/fail verdict; see [`thresholds::ThresholdReport`]).
    thresholds: Option<ThresholdReport>,
    definitions: BTreeMap<String, MetricDef>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<String>,
    cases: Vec<CaseBlock>,
    metrics: MetricsMap,
}

/// Records the identity-matching choice this run made, following ADR
/// 0003 §9.4's precedent for recording such choices in an artifact
/// header. ADR 0004 §8: `evaluate` matches with
/// `taguru::context::normalize_entry` — the same folding the passage
/// index itself uses — never `benchmark::identity::normalize_term`,
/// whose deliberate katakana exception exists for a cross-model
/// comparison this verb does not do.
#[derive(Debug, Clone, Serialize)]
struct MatchingBlock {
    normalization: &'static str,
    /// What `normalization` is applied to, both sides, before an exact
    /// match: `expected_concepts`/`expected_labels` against a cue's
    /// `resolved_names[]`. `expected_associations` is deliberately
    /// absent — its coverage never does a client-side string
    /// comparison at all; `/resolve`/`/resolve_label` pin each
    /// position server-side (ADR 0004 §7 step 2), and coverage is then
    /// just whether the pinned `/query` call returned `total >= 1`
    /// (see [`association_coverage`]).
    normalized: &'static [&'static str],
    /// `expected_sources` match by exact `(source, paragraph)` instead
    /// — the source preflight already requires an exact manifest path
    /// match, and a paragraph index is not text to fold.
    sources: &'static str,
}

impl Default for MatchingBlock {
    fn default() -> Self {
        Self {
            normalization: "taguru::context::normalize_entry",
            normalized: &["expected_concepts", "expected_labels"],
            sources: "exact (source, paragraph) match — no normalization",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct InputsBlock {
    eval: EvalInputsBlock,
    context: String,
    /// scheme + host + port only (ADR 0004 §11) — never the literal
    /// `--url` value.
    url: String,
    out: String,
    default_limit: usize,
    resolve_limit: usize,
    /// #308 (ADR 0006 §14): `"baseline"` (unchanged passage lane) or
    /// `"assembly"` (`--assembly`, ADR 0006's evidence-assembly
    /// passage lane) — an open string, not a closed Rust enum, per
    /// this codebase's convention for wire-visible mode tags.
    mode: String,
    /// The equal-budget ceilings this run enforced. Always present in
    /// `assembly` mode — `POST /contexts/{name}/evidence` has no
    /// unbudgeted mode, so this records the server's own defaults
    /// (`max_items: 40, max_bytes: 65536, max_tokens: 4000`) even when
    /// no `--max-*` flag was given. `None` only in `baseline` mode,
    /// when no `--max-items`/`--max-bytes`/`--max-tokens` flag was
    /// given — that mode alone has a genuinely unbudgeted state
    /// (unchanged from before this flag existed).
    #[serde(skip_serializing_if = "Option::is_none")]
    budget: Option<BudgetLimits>,
    /// The `--rerank MODEL` value, when given — `None` in `baseline`
    /// mode, and `None` in `assembly` mode when `--rerank` was not
    /// given (a fully deterministic assembly run, ADR 0006 §14
    /// configuration 2). Whether the requested model actually reranked
    /// is per-case, in `CaseBlock.evidence`'s `reranker` block — a
    /// server with no `TAGURU_RERANK_URL`/`_MODEL` configured, or a
    /// model mismatch, degrades every case the same way.
    #[serde(skip_serializing_if = "Option::is_none")]
    rerank: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct EvalInputsBlock {
    path: String,
    name: Option<String>,
    cases: usize,
}

#[derive(Debug, Clone, Serialize)]
struct CorpusBlock {
    revision_before: ContextRevision,
    revision_after: ContextRevision,
    /// Equality across all three `ContextRevision` lanes, never
    /// ordering (ADR 0004 §12) — a write landing mid-run flips this to
    /// `false` without aborting the run itself.
    stable: bool,
    last_write_epoch_before: u64,
    last_write_epoch_after: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    embeddings: Option<EmbeddingsBlock>,
    sources_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct EmbeddingsBlock {
    provider_model: Option<String>,
}

#[derive(Serialize)]
struct CaseBlock {
    case_id: String,
    query: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    cues: Vec<String>,
    limit: usize,
    passage: PassageOutcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    structural: Option<StructuralBlock>,
    /// `Some` only when the case declares at least one `relevance >= 1`
    /// `expected_sources` entry and the passage lane completed.
    #[serde(skip_serializing_if = "Option::is_none")]
    recall: Option<RecallBlock>,
    /// `Some` only when the case declares at least one of
    /// `expected_concepts`/`expected_labels`/`expected_associations`.
    #[serde(skip_serializing_if = "Option::is_none")]
    coverage: Option<CoverageBlock>,
    /// The ADR 0004 §7 lane cross-tab input for this one case — `Some`
    /// only when the case declares both a structural and a source
    /// expectation and the passage lane completed; aggregated into
    /// `metrics`' `lanes.*` ratios (§9.1).
    #[serde(skip_serializing_if = "Option::is_none")]
    lane_cross: Option<LaneCrossBlock>,
    /// ADR 0004 §8's two citation measurements — `Some` only when the
    /// case declares at least one `expected_citations` entry. Runs
    /// independent of every other lane, including a passage lane that
    /// missed outright.
    #[serde(skip_serializing_if = "Option::is_none")]
    citations: Option<CitationsBlock>,
    /// Unmet expectations (ADR 0004 §11): sources, then concepts, then
    /// labels, then associations, then citations, capped at 3 entries.
    /// Silent (empty) when the passage lane failed outright — see
    /// [`build_missed`].
    missed: Vec<String>,
    missed_truncated: usize,
    /// #308 (ADR 0006 §14): the assembly lane's own diagnostic detail
    /// — selection trace, reranker outcome, omission breakdown. `Some`
    /// only in `assembly` mode; `passage` above (built from the same
    /// admitted package, see [`hits_from_evidence_items`]) is what recall/
    /// citation/lane-cross scoring actually reads, so this block never
    /// duplicates scoring logic — only what `passage`'s `HitLocator`
    /// projection cannot carry.
    #[serde(skip_serializing_if = "Option::is_none")]
    evidence: Option<EvidenceOutcome>,
    /// #308's equal-budget accounting — `Some` in both modes when a
    /// budget flag was given, `None` when none was (both modes then
    /// run untruncated, matching every run before this flag existed).
    #[serde(skip_serializing_if = "Option::is_none")]
    budget: Option<BudgetAccounting>,
    /// #308's `diversity.sources` metric input (ADR 0006 §14's own new
    /// metric): the count of distinct `source` locators among this
    /// case's `passage.hits` — `baseline`'s (possibly truncated)
    /// passage hits, or `assembly`'s admitted items' own locators
    /// union their `citation_refs` (the identical set
    /// [`hits_from_evidence_items`] builds for recall/citation
    /// scoring, so this reads the same evidence those metrics do).
    /// `None` when the passage/evidence lane failed outright, matching
    /// `recall`/`coverage`'s own not-applicable convention.
    #[serde(skip_serializing_if = "Option::is_none")]
    diversity_sources: Option<usize>,
}

/// Recall@k/MRR/nDCG against `expected_sources`' graded relevance (ADR
/// 0004 §274) — see [`score_recall`].
#[derive(Debug, Clone, Serialize)]
struct RecallBlock {
    recall_at_k: f64,
    mrr: f64,
    ndcg: f64,
    expected_total: usize,
    matched: usize,
}

/// One expectation category's coverage count — see [`coverage_counts`]/
/// [`association_coverage`].
#[derive(Debug, Clone, Serialize)]
struct CoverageCounts {
    expected: usize,
    matched: usize,
    value: f64,
}

#[derive(Debug, Clone, Serialize)]
struct CoverageBlock {
    #[serde(skip_serializing_if = "Option::is_none")]
    concepts: Option<CoverageCounts>,
    #[serde(skip_serializing_if = "Option::is_none")]
    labels: Option<CoverageCounts>,
    #[serde(skip_serializing_if = "Option::is_none")]
    associations: Option<CoverageCounts>,
}

/// One case's contribution to the ADR 0004 §7 lane cross-tab —
/// aggregated across every case that has one into `metrics`'
/// `lanes.structural_hit`/`lanes.passage_hit`/`lanes.both`/
/// `lanes.neither`.
#[derive(Debug, Clone, Serialize)]
struct LaneCrossBlock {
    structural_hit: bool,
    passage_hit: bool,
}

/// ADR 0004 §8's two, deliberately un-merged citation measurements for
/// one case, plus the per-entry checks both are computed from.
#[derive(Debug, Clone, Serialize)]
struct CitationsBlock {
    /// Citation recall: the fraction of `checks` whose `(source,
    /// paragraph)` appeared among this case's served results (passage
    /// hits, plus the structural lane's `AttributionOut` locators when
    /// it ran) — see [`served_locators`]. Independent of `validity`.
    recall: CitationRecallBlock,
    /// Locator validity: the fraction of `checks` whose
    /// `POST /contexts/{name}/citations` call resolved with a matching
    /// `section` (when declared) and `quote` (when declared) — see
    /// [`citation_is_valid`]. Computed even for a case whose passage
    /// lane missed outright.
    validity: CitationValidityBlock,
    /// One entry per `expected_citations[]`, in request order (the
    /// endpoint does not echo `paragraph` back, so requests and
    /// responses correlate positionally).
    checks: Vec<CitationCheck>,
}

#[derive(Debug, Clone, Serialize)]
struct CitationRecallBlock {
    expected_total: usize,
    matched: usize,
    value: f64,
}

#[derive(Debug, Clone, Serialize)]
struct CitationValidityBlock {
    expected_total: usize,
    valid: usize,
    value: f64,
}

/// One `expected_citations[]` entry's outcome from both measurements:
/// `served` is citation recall's own per-entry bit, `outcome` is
/// locator validity's `POST /contexts/{name}/citations` result.
#[derive(Debug, Clone, Serialize)]
struct CitationCheck {
    source: String,
    paragraph: u32,
    served: bool,
    #[serde(flatten)]
    outcome: CitationOutcome,
    latency_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
enum CitationOutcome {
    Resolved {
        section: SectionCheck,
        #[serde(skip_serializing_if = "Option::is_none")]
        quote: Option<QuoteCheck>,
    },
    /// `code` is the server's stable `ErrorCode` string (`no_source`/
    /// `no_paragraph`/etc., `src/api.rs`) when the failure carried one
    /// — `None` for a transport failure or an unparseable response.
    /// `message` is truncated to [`MAX_ERROR_BYTES`] (ADR 0004 §11).
    Unresolved {
        code: Option<String>,
        message: String,
    },
}

/// [`ExpectedCitation::section`]'s three-valued check (ADR 0004 §8):
/// absent stays [`SectionCheck::NotChecked`]; present — including an
/// explicit `null` — is checked against the server's `Citation.section`
/// and comes back `Matched`/`Mismatched`. The server's own `section`
/// value is never recorded on a mismatch (ADR 0004 §11's no-corpus-text
/// posture) — only the user's own declared `expected` rides along, so a
/// reader diagnoses the miss by re-running the printed reproduction
/// command, never by reading served text out of the artifact.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "check", rename_all = "snake_case")]
enum SectionCheck {
    NotChecked,
    Matched { expected: Option<String> },
    Mismatched { expected: Option<String> },
}

/// [`ExpectedCitation::quote`]'s check result (ADR 0004 §8). ADR 0004
/// §11's one documented exception to "no corpus body text": even on a
/// mismatch this records only the user's own declared `quote` and a
/// boolean, never the served paragraph body.
#[derive(Debug, Clone, Serialize)]
struct QuoteCheck {
    declared: String,
    matched: bool,
}

#[derive(Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
enum PassageOutcome {
    Searched {
        #[serde(skip_serializing_if = "Option::is_none")]
        plan: Option<SearchContextPlan>,
        hits: Vec<HitLocator>,
        latency_ms: u64,
    },
    Failed {
        message: String,
        latency_ms: u64,
    },
}

/// A passage hit's locator, stripped of `text` — no corpus body text
/// is written into `evaluation.json` (ADR 0004 §11).
#[derive(Serialize)]
struct HitLocator {
    source: String,
    paragraph: u32,
    score: f32,
    lanes: PassageLanes,
}

impl From<PassageHit> for HitLocator {
    fn from(hit: PassageHit) -> Self {
        HitLocator {
            source: hit.source,
            paragraph: hit.paragraph,
            score: hit.score,
            lanes: hit.lanes,
        }
    }
}

/// #308: flattens an assembled package's admitted `items[]` into the
/// same `HitLocator` shape the baseline passage lane produces, in
/// `fused_rank` order — one `HitLocator` per `citation_refs` entry, so
/// an item with several independent attributions (ADR 0006 §9's
/// corroboration) contributes one locator per source, and an item with
/// none (a zero-attribution association — `EvidenceCandidate`'s own
/// documented case) contributes none. This is what lets
/// `score_recall`/`served_locators`/`build_missed` — every scoring
/// function the baseline passage lane already feeds — run unchanged
/// against an assembled package: they only ever compare
/// `(source, paragraph)` pairs and rank order, never a lane's own
/// score, and RRF's `fused_score` is deliberately never serialized at
/// all (ADR 0006 §7) — `score: 0.0` below is a placeholder, not a
/// discarded real value.
/// #308: the diagnostic-locator projection of an assembled package's
/// admitted `items[]` — [`EvidenceLocator`], stripped of
/// `association`/`passage`/`community` (which can carry passage body
/// text) the same way [`hits_from_evidence_items`] strips it from the
/// scoring-facing `HitLocator` projection.
/// One [`EvidenceItem`]'s own `(source, paragraph)` locator — `None`
/// for an association item, whose locator instead lives entirely in
/// `citation_refs` ([`EvidenceCandidate::citation_refs`],
/// `src/api/evidence.rs`: an association's own payload carries no
/// single source/paragraph of its own, only a set of attributions).
/// `Some` for a passage item (`PassageHit.source`/`.paragraph`
/// directly) or a community item (`community:{id}`, matching the
/// artifact-source-id convention `CommunityHit`'s own doc comment
/// names, and its own `.paragraph`).
fn evidence_item_locator(item: &EvidenceItem) -> Option<(String, u32)> {
    if let Some(passage) = &item.passage {
        return Some((passage.source.clone(), passage.paragraph));
    }
    if let Some(community) = &item.community {
        return Some((
            format!("community:{}", community.community),
            community.paragraph,
        ));
    }
    None
}

fn evidence_locators(items: &[EvidenceItem]) -> Vec<EvidenceLocator> {
    items
        .iter()
        .map(|item| {
            let (source, paragraph) = match evidence_item_locator(item) {
                Some((source, paragraph)) => (Some(source), Some(paragraph)),
                None => (None, None),
            };
            EvidenceLocator {
                candidate_id: item.candidate_id.clone(),
                kind: item.kind.clone(),
                fused_rank: item.fused_rank,
                source,
                paragraph,
                citation_refs: item.citation_refs.clone(),
            }
        })
        .collect()
}

/// [`HitLocator`]s for scoring: each item's own locator
/// ([`evidence_item_locator`]) union its `citation_refs` (an
/// association's corroborating attributions) — a passage/community
/// item without this union would never enter recall/citation/
/// diversity scoring at all, since `citation_refs` is empty for both
/// kinds by design (their locator lives on the payload itself, not as
/// a separate attribution list). `limit` is the case's own configured
/// limit (`options.limit`/`default_limit`, the same value the baseline
/// passage lane's own `sources/search` call already bounds its `hits`
/// response to server-side): `items[]` can carry up to `budget.max_items`
/// admitted candidates (default 40) plus one `citation_refs` entry per
/// association's attribution, structurally unrelated to `limit` — left
/// unbounded, that would size an assembly-mode case's `hits[]` (and
/// therefore its `k` for `recall_at_k`/`mrr`/`ndcg`, and its
/// `lane_cross.passage_hit` classification, ADR 0004 §8) differently
/// from the identical-`limit` baseline case it exists to compare
/// against.
fn hits_from_evidence_items(items: &[EvidenceItem], limit: usize) -> Vec<HitLocator> {
    items
        .iter()
        .flat_map(|item| {
            let own = evidence_item_locator(item).into_iter();
            let attributed = item
                .citation_refs
                .iter()
                .map(|reference| (reference.source.clone(), reference.paragraph));
            own.chain(attributed)
        })
        .take(limit)
        .map(|(source, paragraph)| HitLocator {
            source,
            paragraph,
            score: 0.0,
            lanes: PassageLanes {
                bm25: None,
                vector: None,
            },
        })
        .collect()
}

/// #308's diagnostic locator for one admitted assembly item — stripped
/// of `association`/`passage`/`community` (which can carry passage
/// body text) the same way [`HitLocator`] strips `PassageHit.text`
/// (ADR 0004 §11's no-corpus-text rule). `source`/`paragraph` is the
/// item's own locator ([`evidence_item_locator`]) — absent for an
/// association item, whose locators live in `citation_refs` instead.
#[derive(Serialize)]
struct EvidenceLocator {
    candidate_id: String,
    kind: String,
    fused_rank: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    paragraph: Option<u32>,
    citation_refs: Vec<CitationRef>,
}

/// #308's per-case assembly diagnostics (ADR 0006 §14) — everything
/// `CaseBlock.passage`'s `HitLocator` projection (built by
/// [`hits_from_evidence_items`]) cannot carry. `selection`/`reranker`
/// are the *existing* `EvidencePackage` wire blocks, embedded verbatim
/// — no parallel mirror — matching how `items[]` itself embeds
/// `AssociationOut`/`PassageHit`/`CommunityHit` verbatim upstream.
#[derive(Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
enum EvidenceOutcome {
    Assembled {
        latency_ms: u64,
        items: Vec<EvidenceLocator>,
        #[serde(skip_serializing_if = "BTreeMap::is_empty")]
        omitted_by_reason: BTreeMap<String, usize>,
        selection: SelectionPlan,
        reranker: RerankerPlan,
    },
    Failed {
        message: String,
        latency_ms: u64,
    },
}

/// #308's equal-budget accounting for one case, shared by both modes —
/// `crate::api::evidence::budget::BudgetUsage` (the same wire shape a
/// configured request already returns) flattened alongside how many
/// candidates this case's budget dropped, so the artifact reads as one
/// object rather than a nested `usage` key.
#[derive(Serialize)]
struct BudgetAccounting {
    #[serde(flatten)]
    usage: BudgetUsage,
    omitted_total: usize,
}

#[derive(Debug, Clone, Serialize)]
struct StructuralBlock {
    cues: Vec<CueResolution>,
    associations: Vec<AssociationProbe>,
}

#[derive(Debug, Clone, Serialize)]
struct CueResolution {
    cue: String,
    kind: &'static str,
    resolved_names: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tier: Option<String>,
    limit: usize,
    latency_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct AssociationProbe {
    subject_cue: String,
    label_cue: String,
    object_cue: String,
    subject: PositionOutcome,
    label: PositionOutcome,
    object: PositionOutcome,
    #[serde(skip_serializing_if = "Option::is_none")]
    query: Option<QueryProbe>,
}

/// One `expected_associations[]` position's resolution (ADR 0004 §7
/// step 2's stricter multi-candidate policy): exactly one top-tier
/// candidate pins the position; zero or several do not, and `query` is
/// never called in either of those cases.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
enum PositionOutcome {
    Resolved {
        name: String,
        tier: String,
        latency_ms: u64,
    },
    NotFound {
        latency_ms: u64,
    },
    Ambiguous {
        tier: String,
        candidates: Vec<String>,
        latency_ms: u64,
    },
    Errored {
        message: String,
        latency_ms: u64,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
enum QueryProbe {
    Queried {
        total: usize,
        matches: usize,
        /// The structural lane's own served citation locators (ADR 0004
        /// §8) — every `attributions[]` entry off every matched
        /// association, `source` + `paragraph` only. No corpus body
        /// text (ADR 0004 §11): `AttributionOut` never carries any.
        #[serde(skip_serializing_if = "Vec::is_empty")]
        attributions: Vec<AttributionLocator>,
        latency_ms: u64,
    },
    Errored {
        message: String,
        latency_ms: u64,
    },
}

/// One served `(source, paragraph)` locator read off an `AttributionOut`
/// (`src/api.rs:1453-1462`) — stripped of `weight`/`count`/`section`,
/// which citation recall does not need. `paragraph: None` (an
/// attribution with no paragraph locator at all) never satisfies an
/// `expected_citations` entry, which always names one.
#[derive(Debug, Clone, Serialize)]
struct AttributionLocator {
    source: String,
    paragraph: Option<u32>,
}

mod compare;
mod evidence;
mod thresholds;

#[cfg(test)]
mod tests;
