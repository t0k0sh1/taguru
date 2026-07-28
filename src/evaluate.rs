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
//! ever folded into another's (#215's own requirement). Citation
//! recall and locator validity (#275, the only reason
//! `POST /contexts/{name}/citations` is not called here), configurable
//! thresholds and exit 3 (#276), and `taguru evaluate compare` (#277)
//! land as separate, focused changes on top of this one.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;
use serde_json::Value;
use taguru::context::normalize_entry;

use crate::api::MatchPage;
use crate::api::resolve::TieredResolution;
use crate::api::sources::{PassageHit, PassageLanes, PassagePage, SearchContextPlan};
use crate::cli::default_base_url;
use crate::config::{load_config, subcommand_usage_error};
use crate::evalset::{self, EvalCase, ExpectedAssociation, ExpectedSource};
use crate::measure::{Distribution, MetricDef, MetricValue, MetricsMap, def, ratio_metric};
use crate::registry::{ContextRevision, DirectoryEntry};
use crate::remote::{self, Api, ApiFailure};

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
                        [--config FILE] [--out FILE]

Runs eval.jsonl's cases (ADR 0003 §11's shared dataset, #215's own
extension fields) against one already-populated context's live
retrieval endpoints and writes evaluation.json: per-case passage-lane
hits and structural-lane resolve/query outcomes, recall@k/MRR/nDCG,
concept/label/association coverage, corpus revision bracketing, and
run metadata — a report-only quality gate (this build has no
--thresholds; every completed run exits 0). Citation checks and
configurable thresholds land in follow-up issues on top of this build.

  --eval FILE            eval.jsonl (ADR 0003 §11's shared dataset)
  --context NAME          the already-populated context to evaluate
  --url URL              the server to query; default resolves the
                        same way `taguru health` does (TAGURU_ADDR, or
                        --config/TAGURU_CONFIG)
  --config FILE          load before resolving --url (same as --config
                        everywhere else)
  --out FILE             where to write the artifact (evaluation.json)

`taguru evaluate compare` (comparing two evaluation.json runs) is not
yet implemented.

Contract and discipline: docs/evaluate.html,
adr/0004-retrieval-citation-quality-gate.md.
";

pub fn run(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("--help") | Some("-h") => {
            print!("{USAGE}");
            0
        }
        Some(flag) if flag.starts_with("--") => run_evaluate(args),
        Some(other) => subcommand_usage_error(
            "evaluate",
            &format!(
                "unknown subcommand '{other}' — the default mode is selected by a leading \
                 flag, e.g. 'taguru evaluate --eval FILE --context NAME'"
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
        Err(ApiFailure::NotFound(message)) => {
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

    let default_limit = DEFAULT_LIMIT;
    let cases: Vec<CaseBlock> = loaded
        .cases
        .iter()
        .map(|case| build_case_block(&api, context, case, default_limit))
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

    let metrics = build_metrics(&cases);
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
        // #276 populates this from --thresholds; a report-only run
        // (this build has no other kind) always carries null here.
        thresholds: None,
        definitions: build_definitions(),
        warnings: loaded.warnings.clone(),
        cases,
        metrics,
    };

    if let Err(message) = write_evaluation(&eval_args.out, &evaluation) {
        eprintln!("taguru: evaluate: {message}");
        return 1;
    }

    print_summary(&evaluation, &masked_url, context);
    eprintln!(
        "taguru: evaluate: no --thresholds given — this run is report-only and always exits 0"
    );
    0
}

// ============================== Arguments ==============================

#[derive(Debug)]
struct EvaluateArgs {
    eval: PathBuf,
    context: String,
    url: Option<String>,
    config: Option<PathBuf>,
    out: PathBuf,
}

fn parse_args(args: &[String]) -> Result<EvaluateArgs, i32> {
    let usage = |message: &str| subcommand_usage_error("evaluate", message);
    let mut eval: Option<PathBuf> = None;
    let mut context: Option<String> = None;
    let mut url: Option<String> = None;
    let mut config: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;

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
            flag if flag.starts_with("--") => {
                return Err(usage(&format!("unknown flag '{flag}' for evaluate")));
            }
            _ => return Err(usage("evaluate takes no positional arguments")),
        }
    }

    let eval = eval.ok_or_else(|| usage("--eval FILE is required"))?;
    let context = context.ok_or_else(|| usage("--context NAME is required"))?;

    Ok(EvaluateArgs {
        eval,
        context,
        url,
        config,
        out: out.unwrap_or_else(|| PathBuf::from(DEFAULT_OUT)),
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
/// §11). `reject_userinfo` already refused a URL carrying credentials
/// before this runs; an unparsable `base` (a bare `host:port` with no
/// scheme, say) is echoed back verbatim rather than dropped, since it
/// carries no secret to begin with.
fn mask_url(base: &str) -> String {
    match url::Url::parse(base) {
        Ok(url) => {
            let mut masked = format!("{}://{}", url.scheme(), url.host_str().unwrap_or(""));
            if let Some(port) = url.port() {
                masked.push_str(&format!(":{port}"));
            }
            masked
        }
        Err(_) => base.to_string(),
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
            match extract_passages(&value) {
                Ok((hits, plan)) => PassageOutcome::Searched {
                    plan,
                    hits: hits.into_iter().map(HitLocator::from).collect(),
                    latency_ms,
                },
                Err(message) => PassageOutcome::Failed {
                    message,
                    latency_ms,
                },
            }
        }
        Err(message) => PassageOutcome::Failed {
            message: truncate_message(&message),
            latency_ms: elapsed_ms(started_at),
        },
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
            // Never read downstream (see HitLocator::from) — no corpus
            // body text is written into evaluation.json (ADR 0004 §11).
            text: String::new(),
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
/// `limit: 1` is enough since a pinned triple's `total` is 0 or 1.
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
                Ok(page) => QueryProbe::Queried {
                    total: page.total,
                    matches: page.matches.len(),
                    latency_ms,
                },
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
fn score_recall(expected_sources: &[ExpectedSource], hits: &[HitLocator]) -> Option<RecallBlock> {
    let items: Vec<&ExpectedSource> = expected_sources
        .iter()
        .filter(|expected| expected.relevance >= 1)
        .collect();
    if items.is_empty() {
        return None;
    }

    let matches = |item: &ExpectedSource, hit: &HitLocator| {
        hit.source == item.source
            && (item.paragraphs.is_empty() || item.paragraphs.contains(&hit.paragraph))
    };

    let matched = items
        .iter()
        .filter(|item| hits.iter().any(|hit| matches(item, hit)))
        .count();

    let mut mrr = 0.0;
    for (rank, hit) in hits.iter().enumerate() {
        if items.iter().any(|item| matches(item, hit)) {
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
                .position(|hit| matches(item, hit))
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
    let ndcg = if idcg > 0.0 { dcg / idcg } else { 0.0 };

    Some(RecallBlock {
        recall_at_k: matched as f64 / items.len() as f64,
        mrr,
        ndcg,
        expected_total: items.len(),
        matched,
    })
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
/// labels, associations. Silent when the passage lane failed — its own
/// `Failed` outcome and `passage.failure_rate` already record that;
/// guessing at which sources it would have missed adds no information.
fn build_missed(
    case: &EvalCase,
    hits: Option<&[HitLocator]>,
    concept_resolutions: &[&CueResolution],
    label_resolutions: &[&CueResolution],
    structural: Option<&StructuralBlock>,
) -> (Vec<String>, usize) {
    let mut all = Vec::new();

    if let Some(hits) = hits {
        for expected in &case.expected_sources {
            if expected.relevance == 0 {
                continue;
            }
            let hit = hits.iter().any(|h| {
                h.source == expected.source
                    && (expected.paragraphs.is_empty()
                        || expected.paragraphs.contains(&h.paragraph))
            });
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
            let hit = matches!(
                &assoc.query,
                Some(QueryProbe::Queried { total, .. }) if *total >= 1
            );
            if !hit {
                all.push(format!(
                    "expected_associations: ({}, {}, {}) not queried",
                    assoc.subject_cue, assoc.label_cue, assoc.object_cue
                ));
            }
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
    missed: Vec<String>,
    missed_truncated: usize,
}

fn score_case(
    case: &EvalCase,
    passage: &PassageOutcome,
    structural: Option<&StructuralBlock>,
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
    );

    CaseScores {
        recall,
        coverage,
        lane_cross,
        missed,
        missed_truncated,
    }
}

// ================================ Per case ================================

fn build_case_block(api: &Api, context: &str, case: &EvalCase, default_limit: usize) -> CaseBlock {
    let limit = case.options.limit.unwrap_or(default_limit);
    let passage = run_passage_lane(api, context, case, limit);
    let structural = runs_structural_lane(case).then(|| build_structural_block(api, context, case));
    let scores = score_case(case, &passage, structural.as_ref());

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
        missed: scores.missed,
        missed_truncated: scores.missed_truncated,
    }
}

fn elapsed_ms(started_at: Instant) -> u64 {
    started_at.elapsed().as_millis() as u64
}

// ================================ Metrics =================================

fn build_metrics(cases: &[CaseBlock]) -> MetricsMap {
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
    if !evaluation.corpus.stable {
        println!(
            "  WARNING: the context's revision changed during this run — a write landed \
             mid-run (corpus.revision_before != corpus.revision_after)"
        );
    }
    println!("  wrote {}", evaluation.inputs.out);
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
    thresholds: Option<ThresholdIdentity>,
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
    /// `resolved_names[]`, and `expected_associations`' subject/label/
    /// object cues against a resolved position's stored name.
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
            normalized: &[
                "expected_concepts",
                "expected_labels",
                "expected_associations",
            ],
            sources: "exact (source, paragraph) match — no normalization",
        }
    }
}

/// #276 fills this in with a hash of the threshold file's canonical
/// byte content — defined now only so `evaluation.json`'s `thresholds`
/// key has a concrete (if always-`null` in this build) shape to grow
/// into.
#[derive(Debug, Clone, Serialize)]
struct ThresholdIdentity {
    sha256: String,
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
    /// Unmet expectations (ADR 0004 §11): sources, then concepts, then
    /// labels, then associations, capped at 3 entries. Citation misses
    /// (#275) are not this issue's own scope and stay absent until
    /// then. Silent (empty) when the passage lane failed outright — see
    /// [`build_missed`].
    missed: Vec<String>,
    missed_truncated: usize,
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
        latency_ms: u64,
    },
    Errored {
        message: String,
        latency_ms: u64,
    },
}

#[cfg(test)]
mod tests;
