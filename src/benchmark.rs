//! `taguru benchmark`: ADR 0003's model-matrix harness (issue #256)
//! plus its aggregation stage (issue #257). `taguru benchmark extract`
//! spawns one `taguru extract --diagnostics-out` child process per
//! (model, run) cell against a model matrix (`models.json`), every
//! cell reading the same corpus under the same task settings, and
//! assembles `manifest.json` (the reproduction record) plus
//! `runs/<model_id>.run<NN>.jsonl` (the `AttemptRecord` superset) from
//! what each child reports. `taguru benchmark compare` (see
//! [`compare`]) then reads a finished results directory and derives
//! `measurements.json`/`measurements.csv`.
//!
//! R1-R4 (ADR 0003 §4): a cell runs the exact `extract` an operator
//! runs (R1); every setting reaching a child is set explicitly, never
//! inherited (R2); a cell writes only into its own directory (R3);
//! everything downstream of the models is a pure function of the
//! results directory (R4).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::api::{MAX_CONTEXT_NAME_BYTES, MAX_DESCRIPTION_BYTES, MAX_QUESTIONS_PER_PARAGRAPH};
use crate::clock::{iso8601_utc, now_unix_secs};
use crate::config::subcommand_usage_error;

#[path = "benchmark/compare.rs"]
mod compare;
#[path = "benchmark/identity.rs"]
mod identity;
#[path = "benchmark/search.rs"]
mod search;

const TOP_USAGE: &str = "\
usage: taguru benchmark <extract|compare|search> ...

  extract   run `taguru extract` across a model matrix, writing a
            results directory — see `taguru benchmark extract --help`
  compare   derive measurements.json/measurements.csv from a finished
            results directory — see `taguru benchmark compare --help`
  search    build per-model corpora from a finished results directory
            and compare their search results — see `taguru benchmark
            search --help`

Contract and discipline: docs/benchmark.html,
adr/0003-extraction-model-benchmark.md.
";

const USAGE: &str = "\
usage: taguru benchmark extract --models FILE --context NAME --out DIR
                      [--runs N] [--questions N] [--fact-budget N]
                      [--no-passage] [--lossy] [--candidates] [--vocabulary PATH]
                      [--description TEXT]
                      [--parallel N] [--max-output-tokens N]
                      [--max-attempts N] CORPUS_DIR

Runs one `taguru extract --diagnostics-out` subprocess per (model, run)
cell against the model matrix --models names, every cell over the same
corpus under the same task settings (ADR 0003). Writes, under --out:

  manifest.json                  reproduction record: resolved settings,
                                  the document/chunk dictionary, per-model
                                  provider facts, per-cell outcomes
  models.lock.json                --models with defaults folded in, no
                                  secrets
  runs/<model_id>.run<NN>.jsonl   one file per cell: header, document
                                  start/end, chunk, attempt (the
                                  diagnostics sidecar's own records,
                                  carried through unmodified, plus
                                  harness identity), cell totals
  cells/<model_id>/run<NN>/       the child's own --out: batch files,
                                  .extract-manifest.json,
                                  .extract-checkpoints/, diagnostics.jsonl,
                                  stdout.log, stderr.log, exit_code

  --models FILE        model matrix (see docs/benchmark.html for the
                      schema)
  --context NAME      the context every cell's batch files target
  --out DIR           results directory; re-running the same --out
                      resumes — a cell already recorded complete or
                      failed is never re-run
  --runs N            runs per model, 1-99 (1)
  --questions N       forwarded to every cell's --questions
  --fact-budget N     forwarded to every cell's --fact-budget
  --no-passage        forwarded to every cell's --no-passage
  --lossy             forwarded to every cell's --lossy
  --candidates        forwarded to every cell's --candidates
  --vocabulary PATH   forwarded to every cell's --vocabulary (ADR 0015)
  --description TEXT  forwarded to every cell's --description
  --parallel N        forwarded to every cell's --parallel (1)
  --max-output-tokens N  forwarded to every cell's --max-output-tokens
  --max-attempts N    forwarded to every cell's
                      TAGURU_EXTRACT_MAX_ATTEMPTS, 1-10 (2)
  CORPUS_DIR          exactly one directory (.md/.txt, sorted by name) —
                      every cell sees the same document order, since
                      vocabulary accumulates across a run (ADR 0003 §6)

Contract and discipline: docs/benchmark.html,
adr/0003-extraction-model-benchmark.md.
";

pub fn run(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("--help") | Some("-h") => {
            print!("{TOP_USAGE}");
            0
        }
        Some("extract") => run_extract(&args[1..]),
        Some("compare") => compare::run_compare(&args[1..]),
        Some("search") => search::run_search(&args[1..]),
        Some(other) => subcommand_usage_error(
            "benchmark",
            &format!("unknown subcommand '{other}' (expected 'extract', 'compare', or 'search')"),
        ),
        None => subcommand_usage_error(
            "benchmark",
            "expected a subcommand ('extract', 'compare', or 'search')",
        ),
    }
}

// ============================== Arguments ==============================

const MAX_RUNS: usize = 99;

#[derive(Debug)]
struct BenchArgs {
    models: PathBuf,
    out: PathBuf,
    runs: usize,
    context: String,
    questions: usize,
    fact_budget: Option<usize>,
    no_passage: bool,
    lossy: bool,
    candidates: bool,
    vocabulary: Option<PathBuf>,
    description: Option<String>,
    parallel: usize,
    max_output_tokens: Option<usize>,
    max_attempts: usize,
    corpus: String,
}

impl BenchArgs {
    fn parse(args: &[String]) -> Result<Self, i32> {
        let mut models: Option<PathBuf> = None;
        let mut out: Option<PathBuf> = None;
        let mut runs: Option<usize> = None;
        let mut context: Option<String> = None;
        let mut questions: usize = 0;
        let mut fact_budget: Option<usize> = None;
        let mut no_passage = false;
        let mut lossy = false;
        let mut candidates = false;
        let mut vocabulary: Option<PathBuf> = None;
        let mut description: Option<String> = None;
        let mut parallel: Option<usize> = None;
        let mut max_output_tokens: Option<usize> = None;
        let mut max_attempts: Option<usize> = None;
        let mut corpus: Vec<String> = Vec::new();

        let mut rest = args.iter();
        while let Some(arg) = rest.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print!("{USAGE}");
                    return Err(0);
                }
                "--models" => match rest.next() {
                    Some(path) if models.is_none() => models = Some(PathBuf::from(path)),
                    Some(_) => {
                        return Err(subcommand_usage_error("benchmark", "--models given twice"));
                    }
                    None => {
                        return Err(subcommand_usage_error(
                            "benchmark",
                            "--models needs a file path",
                        ));
                    }
                },
                "--out" => match rest.next() {
                    Some(dir) if out.is_none() => out = Some(PathBuf::from(dir)),
                    Some(_) => {
                        return Err(subcommand_usage_error("benchmark", "--out given twice"));
                    }
                    None => {
                        return Err(subcommand_usage_error(
                            "benchmark",
                            "--out needs a directory",
                        ));
                    }
                },
                "--runs" => match rest.next().map(|n| n.parse::<usize>()) {
                    Some(_) if runs.is_some() => {
                        return Err(subcommand_usage_error("benchmark", "--runs given twice"));
                    }
                    Some(Ok(n)) if (1..=MAX_RUNS).contains(&n) => runs = Some(n),
                    _ => {
                        return Err(subcommand_usage_error(
                            "benchmark",
                            &format!("--runs needs an integer between 1 and {MAX_RUNS}"),
                        ));
                    }
                },
                "--context" => match rest.next() {
                    Some(name) if context.is_none() => context = Some(name.clone()),
                    Some(_) => {
                        return Err(subcommand_usage_error("benchmark", "--context given twice"));
                    }
                    None => {
                        return Err(subcommand_usage_error(
                            "benchmark",
                            "--context needs a name",
                        ));
                    }
                },
                "--questions" => match rest.next().map(|n| n.parse::<usize>()) {
                    Some(_) if questions > 0 => {
                        return Err(subcommand_usage_error(
                            "benchmark",
                            "--questions given twice",
                        ));
                    }
                    Some(Ok(n)) if (1..=MAX_QUESTIONS_PER_PARAGRAPH).contains(&n) => questions = n,
                    _ => {
                        return Err(subcommand_usage_error(
                            "benchmark",
                            &format!(
                                "--questions takes 1..={MAX_QUESTIONS_PER_PARAGRAPH} (per paragraph)"
                            ),
                        ));
                    }
                },
                "--fact-budget" => match rest.next().map(|n| n.parse::<usize>()) {
                    Some(_) if fact_budget.is_some() => {
                        return Err(subcommand_usage_error(
                            "benchmark",
                            "--fact-budget given twice",
                        ));
                    }
                    Some(Ok(n)) if n >= 1 => fact_budget = Some(n),
                    _ => {
                        return Err(subcommand_usage_error(
                            "benchmark",
                            "--fact-budget needs an integer of at least 1",
                        ));
                    }
                },
                "--no-passage" => no_passage = true,
                "--lossy" => lossy = true,
                "--candidates" => candidates = true,
                "--vocabulary" => match rest.next() {
                    Some(path) if vocabulary.is_none() => vocabulary = Some(PathBuf::from(path)),
                    _ => {
                        return Err(subcommand_usage_error(
                            "benchmark",
                            "--vocabulary needs a path and may only be given once",
                        ));
                    }
                },
                "--description" => match rest.next() {
                    Some(text) if description.is_none() => description = Some(text.clone()),
                    Some(_) => {
                        return Err(subcommand_usage_error(
                            "benchmark",
                            "--description given twice",
                        ));
                    }
                    None => {
                        return Err(subcommand_usage_error(
                            "benchmark",
                            "--description needs a text",
                        ));
                    }
                },
                "--parallel" => match rest.next().map(|n| n.parse::<usize>()) {
                    Some(_) if parallel.is_some() => {
                        return Err(subcommand_usage_error(
                            "benchmark",
                            "--parallel given twice",
                        ));
                    }
                    Some(Ok(n)) if n >= 1 => parallel = Some(n),
                    _ => {
                        return Err(subcommand_usage_error(
                            "benchmark",
                            "--parallel needs an integer of at least 1",
                        ));
                    }
                },
                "--max-output-tokens" => match rest.next().map(|n| n.parse::<usize>()) {
                    Some(_) if max_output_tokens.is_some() => {
                        return Err(subcommand_usage_error(
                            "benchmark",
                            "--max-output-tokens given twice",
                        ));
                    }
                    Some(Ok(n)) if n >= 1 => max_output_tokens = Some(n),
                    _ => {
                        return Err(subcommand_usage_error(
                            "benchmark",
                            "--max-output-tokens needs an integer of at least 1",
                        ));
                    }
                },
                "--max-attempts" => match rest.next().map(|n| n.parse::<usize>()) {
                    Some(_) if max_attempts.is_some() => {
                        return Err(subcommand_usage_error(
                            "benchmark",
                            "--max-attempts given twice",
                        ));
                    }
                    Some(Ok(n)) if (1..=crate::extract::MAX_EXTRACT_ATTEMPTS).contains(&n) => {
                        max_attempts = Some(n);
                    }
                    _ => {
                        return Err(subcommand_usage_error(
                            "benchmark",
                            &format!(
                                "--max-attempts needs an integer between 1 and {}",
                                crate::extract::MAX_EXTRACT_ATTEMPTS
                            ),
                        ));
                    }
                },
                other if other.starts_with('-') => {
                    return Err(subcommand_usage_error(
                        "benchmark",
                        &format!("unknown flag '{other}'"),
                    ));
                }
                path => corpus.push(path.to_string()),
            }
        }
        let Some(models) = models else {
            return Err(subcommand_usage_error(
                "benchmark",
                "--models FILE is required",
            ));
        };
        let Some(out) = out else {
            return Err(subcommand_usage_error("benchmark", "--out DIR is required"));
        };
        let Some(context) = context else {
            return Err(subcommand_usage_error(
                "benchmark",
                "--context NAME is required",
            ));
        };
        if context.len() > MAX_CONTEXT_NAME_BYTES {
            return Err(subcommand_usage_error(
                "benchmark",
                &format!(
                    "context name of {} bytes exceeds the {MAX_CONTEXT_NAME_BYTES}-byte cap",
                    context.len()
                ),
            ));
        }
        if let Some(text) = &description
            && text.len() > MAX_DESCRIPTION_BYTES
        {
            return Err(subcommand_usage_error(
                "benchmark",
                &format!(
                    "description of {} bytes exceeds the {MAX_DESCRIPTION_BYTES}-byte cap",
                    text.len()
                ),
            ));
        }
        if questions > 0 && no_passage {
            return Err(subcommand_usage_error(
                "benchmark",
                "--questions needs the passage (--no-passage strips the text the questions \
                 would attach to)",
            ));
        }
        let corpus = match corpus.len() {
            0 => {
                return Err(subcommand_usage_error(
                    "benchmark",
                    "a corpus directory is required",
                ));
            }
            1 => corpus.into_iter().next().expect("len checked above"),
            _ => {
                return Err(subcommand_usage_error(
                    "benchmark",
                    "exactly one corpus directory is accepted — every cell must see the same \
                     document order (ADR 0003 §6)",
                ));
            }
        };
        if !Path::new(&corpus).is_dir() {
            return Err(subcommand_usage_error(
                "benchmark",
                &format!("'{corpus}' is not a directory"),
            ));
        }
        Ok(Self {
            models,
            out,
            runs: runs.unwrap_or(1),
            context,
            questions,
            fact_budget,
            no_passage,
            lossy,
            candidates,
            vocabulary,
            description,
            parallel: parallel.unwrap_or(1),
            max_output_tokens,
            max_attempts: max_attempts.unwrap_or(crate::extract::DEFAULT_MAX_ATTEMPTS),
            corpus,
        })
    }
}

#[cfg(test)]
mod args_tests {
    use super::*;

    fn args(words: &[&str]) -> Result<BenchArgs, i32> {
        BenchArgs::parse(&words.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn required_flags_are_enforced() {
        assert_eq!(args(&[]).unwrap_err(), 2);
        assert_eq!(args(&["--context", "c"]).unwrap_err(), 2);
    }

    #[test]
    fn runs_and_max_attempts_reject_out_of_range_values() {
        let dir = std::env::temp_dir();
        let corpus = dir.to_str().unwrap();
        assert_eq!(
            args(&[
                "--models",
                "m.json",
                "--context",
                "c",
                "--out",
                "o",
                "--runs",
                "0",
                corpus
            ])
            .unwrap_err(),
            2
        );
        assert_eq!(
            args(&[
                "--models",
                "m.json",
                "--context",
                "c",
                "--out",
                "o",
                "--max-attempts",
                "11",
                corpus
            ])
            .unwrap_err(),
            2
        );
    }

    /// A flag given twice is a usage error, never a silent last-wins —
    /// the same contract every other subcommand's parser enforces.
    #[test]
    fn value_flags_given_twice_are_usage_errors() {
        let dir = std::env::temp_dir();
        let corpus = dir.to_str().unwrap();
        for duplicated in [
            ["--models", "m2.json"],
            ["--out", "o2"],
            ["--context", "c2"],
            ["--runs", "2"],
            ["--questions", "1"],
            ["--fact-budget", "5"],
            ["--description", "d"],
            ["--parallel", "2"],
            ["--max-output-tokens", "100"],
            ["--max-attempts", "3"],
            ["--vocabulary", "v2"],
        ] {
            let mut words = vec!["--models", "m.json", "--context", "c", "--out", "o"];
            // A first occurrence for the flags the base line lacks, so
            // the duplicate below is always the second one seen.
            if !words.contains(&duplicated[0]) {
                words.extend([duplicated[0], duplicated[1]]);
            }
            words.extend(duplicated);
            words.push(corpus);
            assert_eq!(args(&words).unwrap_err(), 2, "{duplicated:?}");
        }
    }

    /// The happy path the duplicate test above cannot see: a single
    /// `--vocabulary` must parse and carry its path (a mutated guard
    /// that rejects every occurrence would still pass the
    /// duplicate-errors test).
    #[test]
    fn vocabulary_flag_parses_once() {
        let dir = std::env::temp_dir();
        let corpus = dir.to_str().unwrap();
        let parsed = args(&[
            "--models",
            "m.json",
            "--context",
            "c",
            "--out",
            "o",
            "--vocabulary",
            "vocab.jsonl",
            corpus,
        ])
        .unwrap();
        assert_eq!(parsed.vocabulary.as_deref(), Some(Path::new("vocab.jsonl")));
    }

    #[test]
    fn exactly_one_corpus_directory_is_required() {
        let dir = std::env::temp_dir();
        let corpus = dir.to_str().unwrap();
        assert_eq!(
            args(&["--models", "m.json", "--context", "c", "--out", "o"]).unwrap_err(),
            2,
            "no corpus is a usage error"
        );
        assert_eq!(
            args(&[
                "--models",
                "m.json",
                "--context",
                "c",
                "--out",
                "o",
                corpus,
                corpus
            ])
            .unwrap_err(),
            2,
            "two corpus arguments is a usage error"
        );
    }

    #[test]
    fn questions_and_no_passage_are_mutually_exclusive() {
        let dir = std::env::temp_dir();
        let corpus = dir.to_str().unwrap();
        assert_eq!(
            args(&[
                "--models",
                "m.json",
                "--context",
                "c",
                "--out",
                "o",
                "--questions",
                "1",
                "--no-passage",
                corpus
            ])
            .unwrap_err(),
            2
        );
    }

    #[test]
    fn a_valid_invocation_parses_every_field() {
        let dir = std::env::temp_dir();
        let corpus = dir.to_str().unwrap();
        let parsed = args(&[
            "--models",
            "m.json",
            "--context",
            "c",
            "--out",
            "o",
            "--runs",
            "3",
            corpus,
        ])
        .expect("valid arguments must parse");
        assert_eq!(parsed.runs, 3);
        assert_eq!(parsed.context, "c");
        assert_eq!(parsed.corpus, corpus);
        assert_eq!(parsed.max_attempts, crate::extract::DEFAULT_MAX_ATTEMPTS);
        assert_eq!(parsed.parallel, 1);
    }
}

// ============================== models.json ==============================
//
// ADR 0003 §8: a per-model record names provider identity and capability
// only, never a task setting — the fairness invariant is enforced by
// construction: no such field exists to parse. `taguru_benchmark_models`
// is equality-checked (ADR 0003 §10): this file is authored by hand, not
// written and re-read by taguru itself, so a shape it wasn't built for is
// refused rather than defaulted.

const MODELS_VERSION: u64 = 1;
const MAX_MODEL_ID_BYTES: usize = 64;

#[derive(Deserialize)]
struct ModelsFile {
    taguru_benchmark_models: u64,
    #[serde(default)]
    defaults: ModelDefaults,
    #[serde(default)]
    models: Vec<ModelEntry>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Deserialize, Default)]
struct ModelDefaults {
    timeout_secs: Option<usize>,
    structured_output: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
    #[serde(default)]
    label: Option<String>,
    model: String,
    url: String,
    #[serde(default)]
    api_key_env: Option<String>,
    #[serde(default)]
    structured_output: Option<String>,
    #[serde(default)]
    timeout_secs: Option<usize>,
    #[serde(default)]
    note: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

/// A `models.json` entry with `defaults` folded in — what
/// `models.lock.json` and `manifest.json`'s own `models[]` record, so
/// neither reader has to re-apply defaults. `structured_output` keeps
/// the validated REQUESTED spelling (`auto`/`json-schema`/`json-object`/
/// `off`), never a probe's resolution — that varies per cell, not per
/// model (ADR 0003 §6).
#[derive(Debug, Serialize, Clone)]
struct ResolvedModel {
    id: String,
    label: Option<String>,
    model: String,
    url: String,
    api_key_env: Option<String>,
    structured_output: String,
    timeout_secs: usize,
    note: Option<String>,
}

/// The file's raw bytes (hashed into `manifest.json`'s `config_sha256`
/// without a second read), its resolved models, and typo warnings to
/// print once.
type LoadedModelsFile = (Vec<u8>, Vec<ResolvedModel>, Vec<String>);

/// Parses and validates `path`, folding `defaults` into every entry.
/// Every check here runs before any model is called (ADR 0003 §8's
/// secrets rule and fairness invariant are both enforced at parse time,
/// never discovered later mid-matrix).
fn load_models_file(path: &Path) -> Result<LoadedModelsFile, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let parsed: ModelsFile =
        serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()))?;
    if parsed.taguru_benchmark_models != MODELS_VERSION {
        return Err(format!(
            "{}: taguru_benchmark_models must be {MODELS_VERSION}, got {}",
            path.display(),
            parsed.taguru_benchmark_models
        ));
    }
    if parsed.models.is_empty() {
        return Err(format!(
            "{}: 'models' must list at least one entry",
            path.display()
        ));
    }
    let label = path.display();
    let mut warnings = Vec::new();
    for key in parsed.extra.keys() {
        warnings.push(format!(
            "{label}: '{key}' is not a key taguru reads (typo?)"
        ));
    }
    for key in parsed.defaults.extra.keys() {
        warnings.push(format!(
            "{label}: defaults.'{key}' is not a key taguru reads (typo?)"
        ));
    }

    let mut seen_ids = BTreeSet::new();
    let mut resolved = Vec::with_capacity(parsed.models.len());
    for entry in &parsed.models {
        validate_model_id(path, &entry.id)?;
        if !seen_ids.insert(entry.id.clone()) {
            return Err(format!("{label}: duplicate model id '{}'", entry.id));
        }
        for key in entry.extra.keys() {
            warnings.push(format!(
                "{label}: model '{}': '{key}' is not a key taguru reads (typo?)",
                entry.id
            ));
        }
        if entry.model.is_empty() {
            return Err(format!(
                "{label}: model '{}': 'model' must not be empty",
                entry.id
            ));
        }
        validate_model_url(path, &entry.id, &entry.url)?;
        if let Some(name) = &entry.api_key_env {
            validate_api_key_env_name(path, &entry.id, name)?;
            if std::env::var_os(name).is_none() {
                return Err(format!(
                    "{label}: model '{}': api_key_env '{name}' is not set in the environment",
                    entry.id
                ));
            }
        }
        let structured_output = entry
            .structured_output
            .clone()
            .or_else(|| parsed.defaults.structured_output.clone())
            .unwrap_or_else(|| "off".to_string());
        if crate::extract::StructuredOutputMode::parse(&structured_output).is_none() {
            return Err(format!(
                "{label}: model '{}': structured_output must be auto, json-schema, \
                 json-object, or off, got '{structured_output}'",
                entry.id
            ));
        }
        let timeout_secs = entry
            .timeout_secs
            .or(parsed.defaults.timeout_secs)
            .unwrap_or(crate::extract::DEFAULT_TIMEOUT_SECS);
        resolved.push(ResolvedModel {
            id: entry.id.clone(),
            label: entry.label.clone(),
            model: entry.model.clone(),
            url: entry.url.clone(),
            api_key_env: entry.api_key_env.clone(),
            structured_output,
            timeout_secs,
            note: entry.note.clone(),
        });
    }
    Ok((bytes, resolved, warnings))
}

fn validate_model_id(path: &Path, id: &str) -> Result<(), String> {
    let label = path.display();
    if id.is_empty() || id.len() > MAX_MODEL_ID_BYTES {
        return Err(format!(
            "{label}: model id '{id}' must be 1..={MAX_MODEL_ID_BYTES} bytes"
        ));
    }
    let mut chars = id.chars();
    let first = chars.next().expect("non-empty checked above");
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return Err(format!(
            "{label}: model id '{id}' must start with a lowercase letter or digit"
        ));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '.' | '_' | '-'))
    {
        return Err(format!(
            "{label}: model id '{id}' may only contain lowercase letters, digits, '.', \
             '_', '-'"
        ));
    }
    Ok(())
}

/// The lexicographically smaller id first — the canonical order
/// [`pair_key`] and every caller that carries a parallel `a`/`b` pair
/// (`compare::differences`'s `PairInfo`) must agree on, so the same
/// unordered pair produces the same key and the same `(a, b)` display
/// regardless of manifest declaration order or which side a caller's
/// own iteration happened to visit first.
pub(crate) fn canonical_pair<'a>(a: &'a str, b: &'a str) -> (&'a str, &'a str) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Builds a collision-safe string key identifying an unordered pair of
/// model ids — shared by `search`'s `pairs` map (a `BTreeMap<String,
/// _>` serialized as a JSON object, so the key must stay a plain
/// string) and `compare::differences`'s `pair_id` output field. Always
/// canonicalizes via [`canonical_pair`] first, so `pair_key(a, b)` and
/// `pair_key(b, a)` produce the same key — callers that iterate model
/// ids in different orders (manifest declaration order vs. a sorted
/// `BTreeMap`'s keys) still land on one shared entry per pair.
/// `validate_model_id` permits consecutive underscores in a model id,
/// so a naive `format!("{a}__{b}")` is not injective: `(a, "b__c")`
/// and `("a__b", c)` both format to `"a__b__c"`, silently colliding.
/// Prefixing `a`'s byte length followed by `:` — a byte
/// `validate_model_id` never allows in a model id — makes the split
/// point between `a` and `b` unambiguous, so distinct pairs can never
/// produce the same key.
pub(crate) fn pair_key(a: &str, b: &str) -> String {
    let (a, b) = canonical_pair(a, b);
    format!("{}:{a}__{b}", a.len())
}

/// Rejects inline `user:password@` userinfo at parse time (ADR 0003
/// §8's Secrets rule): a credential belongs in `api_key_env`, never
/// embedded in the endpoint URL where it would ride into every artifact
/// that records `url` verbatim.
fn validate_model_url(path: &Path, model_id: &str, url: &str) -> Result<(), String> {
    let label = path.display();
    let parsed = url::Url::parse(url)
        .map_err(|error| format!("{label}: model '{model_id}': url '{url}' is invalid: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(format!(
            "{label}: model '{model_id}': url '{url}' must be http or https"
        ));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(format!(
            "{label}: model '{model_id}': url '{url}' carries inline credentials \
             (user:password@) — use api_key_env instead"
        ));
    }
    Ok(())
}

/// `api_key_env` names an environment variable; it is never the
/// credential's value. A value that cannot be a variable name (a `sk-…`
/// key, for instance) is refused here rather than silently forwarded as
/// a literal env-var name that will never resolve.
fn validate_api_key_env_name(path: &Path, model_id: &str, name: &str) -> Result<(), String> {
    let label = path.display();
    let mut chars = name.chars();
    let ok_first = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_');
    let ok_rest = !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !ok_first || !ok_rest {
        return Err(format!(
            "{label}: model '{model_id}': api_key_env '{name}' does not look like an \
             environment variable name (it names a variable, never the key's value)"
        ));
    }
    Ok(())
}

#[derive(Serialize)]
struct ModelsLock<'a> {
    taguru_benchmark_models: u64,
    models: &'a [ResolvedModel],
}

fn write_models_lock(path: &Path, models: &[ResolvedModel]) -> io::Result<()> {
    let lock = ModelsLock {
        taguru_benchmark_models: MODELS_VERSION,
        models,
    };
    let text = serde_json::to_string_pretty(&lock).expect("a models lock serializes");
    crate::storage::write_atomic(path, text.as_bytes())
}

#[cfg(test)]
mod models_json_tests {
    use super::*;

    fn write_temp(tag: &str, text: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "taguru-benchmark-test-{tag}-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn a_wrong_version_is_refused_by_equality_not_range() {
        let path = write_temp(
            "version",
            r#"{"taguru_benchmark_models":2,"models":[{"id":"m","model":"x","url":"http://h/v1/chat/completions"}]}"#,
        );
        let error = load_models_file(&path).unwrap_err();
        assert!(error.contains("taguru_benchmark_models"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn model_ids_are_validated_and_deduplicated() {
        let dummy = Path::new("models.json");
        assert!(validate_model_id(dummy, "qwen25-7b_q4.a").is_ok());
        assert!(validate_model_id(dummy, "-bad").is_err());
        assert!(validate_model_id(dummy, "Bad").is_err());
        assert!(validate_model_id(dummy, "").is_err());
        assert!(validate_model_id(dummy, &"a".repeat(65)).is_err());

        let path = write_temp(
            "dup",
            r#"{"taguru_benchmark_models":1,"models":[
                {"id":"m","model":"x","url":"http://h/v1/chat/completions"},
                {"id":"m","model":"y","url":"http://h/v1/chat/completions"}
            ]}"#,
        );
        let error = load_models_file(&path).unwrap_err();
        assert!(error.contains("duplicate"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_url_carrying_inline_userinfo_is_refused() {
        let path = write_temp(
            "userinfo",
            r#"{"taguru_benchmark_models":1,"models":[
                {"id":"m","model":"x","url":"http://user:pass@h/v1/chat/completions"}
            ]}"#,
        );
        let error = load_models_file(&path).unwrap_err();
        assert!(error.contains("inline credentials"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_api_key_env_that_looks_like_a_key_value_is_refused() {
        let path = write_temp(
            "keyshaped",
            r#"{"taguru_benchmark_models":1,"models":[
                {"id":"m","model":"x","url":"http://h/v1/chat/completions","api_key_env":"sk-abc123"}
            ]}"#,
        );
        let error = load_models_file(&path).unwrap_err();
        assert!(
            error.contains("does not look like an environment variable"),
            "{error}"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_unset_api_key_env_is_refused_before_any_model_is_called() {
        // SAFETY: single-threaded test process, this key is unique to it.
        unsafe { std::env::remove_var("TAGURU_BENCH_TEST_UNSET_KEY") };
        let path = write_temp(
            "unsetkey",
            r#"{"taguru_benchmark_models":1,"models":[
                {"id":"m","model":"x","url":"http://h/v1/chat/completions","api_key_env":"TAGURU_BENCH_TEST_UNSET_KEY"}
            ]}"#,
        );
        let error = load_models_file(&path).unwrap_err();
        assert!(error.contains("is not set"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_unrecognized_structured_output_value_is_refused() {
        let path = write_temp(
            "badstructured",
            r#"{"taguru_benchmark_models":1,"models":[
                {"id":"m","model":"x","url":"http://h/v1/chat/completions","structured_output":"json_schema"}
            ]}"#,
        );
        let error = load_models_file(&path).unwrap_err();
        assert!(error.contains("structured_output"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_empty_model_name_is_refused() {
        let path = write_temp(
            "emptymodel",
            r#"{"taguru_benchmark_models":1,"models":[
                {"id":"m","model":"","url":"http://h/v1/chat/completions"}
            ]}"#,
        );
        let error = load_models_file(&path).unwrap_err();
        assert!(error.contains("'model' must not be empty"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn validation_errors_and_warnings_name_the_actual_configured_path_not_a_fixed_literal() {
        let path = write_temp(
            "custompath",
            r#"{"taguru_benchmark_models":1,"models":[{"id":"Bad","model":"x","url":"http://h/v1/chat/completions"}]}"#,
        );
        let error = load_models_file(&path).unwrap_err();
        assert!(
            error.contains(path.to_str().unwrap()),
            "error must name the file actually given via --models, not a hardcoded \
             'models.json': {error}"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn defaults_fold_into_entries_that_omit_the_field() {
        let path = write_temp(
            "defaults",
            r#"{"taguru_benchmark_models":1,
                "defaults":{"timeout_secs":123,"structured_output":"auto"},
                "models":[
                    {"id":"a","model":"x","url":"http://h/v1/chat/completions"},
                    {"id":"b","model":"y","url":"http://h/v1/chat/completions","timeout_secs":9,"structured_output":"off"}
                ]}"#,
        );
        let (_, resolved, warnings) = load_models_file(&path).expect("valid file must parse");
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(resolved[0].timeout_secs, 123);
        assert_eq!(resolved[0].structured_output, "auto");
        assert_eq!(resolved[1].timeout_secs, 9);
        assert_eq!(resolved[1].structured_output, "off");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_unknown_key_earns_a_warning_not_a_hard_error() {
        let path = write_temp(
            "typo",
            r#"{"taguru_benchmark_models":1,"modles":"typo","models":[
                {"id":"m","model":"x","url":"http://h/v1/chat/completions","nte":"typo"}
            ]}"#,
        );
        let (_, resolved, warnings) = load_models_file(&path).expect("typos must not be fatal");
        assert_eq!(resolved.len(), 1);
        assert!(
            warnings.iter().any(|w| w.contains("modles")),
            "{warnings:?}"
        );
        assert!(warnings.iter().any(|w| w.contains("nte")), "{warnings:?}");
        let _ = fs::remove_file(&path);
    }
}

// ============================ Time and identity ============================

fn random_hex(bytes: usize) -> String {
    let mut buf = vec![0u8; bytes];
    getrandom::fill(&mut buf).expect("the OS random source must work");
    buf.iter()
        .fold(String::with_capacity(bytes * 2), |mut hex, byte| {
            use std::fmt::Write as _;
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

/// `"20260726T091422Z-3f7a1c"`: a compact timestamp (collision-unlikely
/// on its own for anything run by a human) plus 3 random bytes, so two
/// matrices started in the same second never share a `run_id`.
fn generate_run_id() -> String {
    let compact = iso8601_utc(now_unix_secs()).replace(['-', ':'], "");
    format!("{compact}-{}", random_hex(3))
}

#[cfg(test)]
mod time_tests {
    use super::*;

    #[test]
    fn run_id_carries_a_compact_timestamp_and_a_random_suffix() {
        let id = generate_run_id();
        let (stamp, suffix) = id.split_once('-').expect("run_id has a dash separator");
        assert_eq!(stamp.len(), "20260726T091422Z".len());
        assert_eq!(suffix.len(), 6, "3 random bytes as hex");
    }
}

// ============================ Document dictionary ============================

/// One chunk's provenance, mirroring `crate::extract::ChunkDescriptor`
/// (ADR 0003 §7): a `crate::paragraph::split` index range, never a byte
/// offset into the labeled rendering `chunk()` actually operates on.
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
struct ChunkInfo {
    chunk_index: usize,
    chunk_sha256: String,
    chunk_bytes: usize,
    paragraph_first: u32,
    paragraph_last: u32,
}

/// One corpus document's dictionary entry (ADR 0003 §9.1). `path` is
/// the exact source string `crate::extract::expand_documents` would
/// hand `taguru extract` for this file — computed once here and reused
/// verbatim by every cell, so a cell's own `source` field always joins
/// against this dictionary by exact string match.
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
struct DocumentInfo {
    document_id: String,
    path: String,
    bytes: usize,
    sha256: String,
    paragraph_count: usize,
    chunk_total: usize,
    chunks: Vec<ChunkInfo>,
}

/// The relative-path candidate for a document id, before collision
/// dedup: `source` relative to `corpus_root`, `/`/`\` flattened to
/// `__`, extension stripped. Collisions (distinct sources flattening to
/// the same candidate) are resolved by the caller with a hash suffix —
/// this function alone is not injective, by design (ADR 0003 §9.1).
fn candidate_document_id(source: &str, corpus_root: &str) -> String {
    let rel = Path::new(source)
        .strip_prefix(Path::new(corpus_root))
        .unwrap_or(Path::new(source));
    let rel_str = rel.to_string_lossy().replace(['/', '\\'], "__");
    match rel_str.rfind('.') {
        Some(pos) if pos > 0 => rel_str[..pos].to_string(),
        _ => rel_str,
    }
}

/// Enumerates the corpus exactly as `crate::extract::expand_documents`
/// will for every spawned cell (same function, same argument), then
/// builds each document's hash/paragraph/chunk provenance in-process
/// via `crate::extract::chunk_plan` — the seam #262 added precisely so
/// this dictionary never re-implements `paragraph::split`'s packing
/// rule a second time.
fn build_document_dictionary(corpus: &str) -> Result<Vec<DocumentInfo>, String> {
    let files = crate::extract::expand_documents(&[corpus.to_string()])?;
    let mut sources = Vec::with_capacity(files.len());
    let mut candidates: BTreeMap<String, usize> = BTreeMap::new();
    let mut rows = Vec::with_capacity(files.len());
    for path in &files {
        let source = path.to_string_lossy().into_owned();
        let text =
            crate::extract::read_document(path).map_err(|error| format!("{source}: {error}"))?;
        let sha256 = crate::sha256::sha256_hex(text.as_bytes());
        let paragraph_count = crate::paragraph::split(&text).len();
        let descriptors = crate::extract::chunk_plan(&text);
        let chunks: Vec<ChunkInfo> = descriptors
            .iter()
            .enumerate()
            .map(|(index, descriptor)| ChunkInfo {
                chunk_index: index,
                chunk_sha256: descriptor.sha256.clone(),
                chunk_bytes: descriptor.text.len(),
                paragraph_first: descriptor.paragraph_first,
                paragraph_last: descriptor.paragraph_last,
            })
            .collect();
        let candidate = candidate_document_id(&source, corpus);
        *candidates.entry(candidate.clone()).or_insert(0) += 1;
        rows.push((source.clone(), text.len(), sha256, paragraph_count, chunks));
        sources.push(candidate);
    }
    let mut documents = Vec::with_capacity(rows.len());
    for (candidate, (source, bytes, sha256, paragraph_count, chunks)) in
        sources.into_iter().zip(rows)
    {
        let document_id = if candidates[&candidate] > 1 {
            format!(
                "{candidate}-{}",
                &crate::sha256::sha256_hex(source.as_bytes())[..16]
            )
        } else {
            candidate
        };
        documents.push(DocumentInfo {
            document_id,
            path: source,
            bytes,
            sha256,
            paragraph_count,
            chunk_total: chunks.len(),
            chunks,
        });
    }
    Ok(documents)
}

#[cfg(test)]
mod document_dictionary_tests {
    use super::*;

    fn corpus_dir(tag: &str, files: &[(&str, &str)]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "taguru-benchmark-corpus-{tag}-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for (name, text) in files {
            fs::write(dir.join(name), text).unwrap();
        }
        dir
    }

    #[test]
    fn candidate_ids_flatten_and_strip_the_extension() {
        assert_eq!(
            candidate_document_id("corpus/brewery.md", "corpus"),
            "brewery"
        );
        assert_eq!(
            candidate_document_id("corpus/sub/dir/doc.md", "corpus"),
            "sub__dir__doc"
        );
    }

    #[test]
    fn colliding_candidates_get_a_hash_suffix_unique_ones_do_not() {
        let dir = corpus_dir(
            "collide",
            &[
                ("a.md", "first document text."),
                ("b.md", "second document text."),
            ],
        );
        let corpus = dir.to_str().unwrap();
        let documents = build_document_dictionary(corpus).expect("must build");
        assert_eq!(documents.len(), 2);
        // Distinct file stems never collide, so no suffix is added.
        assert!(!documents[0].document_id.contains('-') || documents[0].document_id == "a");
        assert_eq!(documents[0].document_id, "a");
        assert_eq!(documents[1].document_id, "b");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_dictionary_matches_chunk_plan_and_paragraph_split_directly() {
        let dir = corpus_dir(
            "match",
            &[("only.md", "第一段落。\n\n第二段落。\n\n第三段落。")],
        );
        let corpus = dir.to_str().unwrap();
        let documents = build_document_dictionary(corpus).expect("must build");
        assert_eq!(documents.len(), 1);
        let text = fs::read_to_string(dir.join("only.md")).unwrap();
        assert_eq!(
            documents[0].paragraph_count,
            crate::paragraph::split(&text).len()
        );
        assert_eq!(
            documents[0].chunk_total,
            crate::extract::chunk_plan(&text).len()
        );
        assert_eq!(
            documents[0].sha256,
            crate::sha256::sha256_hex(text.as_bytes())
        );
        let _ = fs::remove_dir_all(&dir);
    }
}

// ============================ Environment probe ============================

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
struct EnvironmentBlock {
    os: String,
    arch: String,
    cpu_model: Option<String>,
    cpu_cores: Option<usize>,
    memory_bytes: Option<u64>,
    gpu: Option<String>,
    hostname_hash: Option<String>,
}

fn environment_block() -> EnvironmentBlock {
    EnvironmentBlock {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_model: probe_cpu_model(),
        cpu_cores: std::thread::available_parallelism().ok().map(|n| n.get()),
        memory_bytes: probe_memory_bytes(),
        gpu: None,
        hostname_hash: probe_hostname_hash(),
    }
}

#[cfg(target_os = "macos")]
fn sysctl_string(name: &str) -> Option<String> {
    use std::ffi::CString;
    let cname = CString::new(name).ok()?;
    let mut size: libc::size_t = 0;
    unsafe {
        if libc::sysctlbyname(
            cname.as_ptr(),
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        ) != 0
            || size == 0
        {
            return None;
        }
        let mut buf = vec![0u8; size];
        if libc::sysctlbyname(
            cname.as_ptr(),
            buf.as_mut_ptr() as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Some(String::from_utf8_lossy(&buf[..end]).into_owned())
    }
}

#[cfg(target_os = "macos")]
fn sysctl_u64(name: &str) -> Option<u64> {
    use std::ffi::CString;
    let cname = CString::new(name).ok()?;
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>() as libc::size_t;
    unsafe {
        if libc::sysctlbyname(
            cname.as_ptr(),
            &mut value as *mut u64 as *mut libc::c_void,
            &mut size,
            std::ptr::null_mut(),
            0,
        ) != 0
        {
            return None;
        }
    }
    Some(value)
}

#[cfg(target_os = "macos")]
fn probe_cpu_model() -> Option<String> {
    sysctl_string("machdep.cpu.brand_string")
}

#[cfg(target_os = "linux")]
fn probe_cpu_model() -> Option<String> {
    let text = fs::read_to_string("/proc/cpuinfo").ok()?;
    text.lines()
        .find(|line| line.starts_with("model name"))
        .and_then(|line| line.split_once(':'))
        .map(|(_, value)| value.trim().to_string())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn probe_cpu_model() -> Option<String> {
    None
}

#[cfg(target_os = "macos")]
fn probe_memory_bytes() -> Option<u64> {
    sysctl_u64("hw.memsize")
}

#[cfg(target_os = "linux")]
fn probe_memory_bytes() -> Option<u64> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|line| line.starts_with("MemTotal:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kib * 1024)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn probe_memory_bytes() -> Option<u64> {
    None
}

#[cfg(unix)]
fn probe_hostname_hash() -> Option<String> {
    let mut buf = vec![0u8; 256];
    let ret = unsafe { libc::gethostname(buf.as_mut_ptr() as *mut libc::c_char, buf.len()) };
    if ret != 0 {
        return None;
    }
    let end = buf.iter().position(|&b| b == 0)?;
    Some(crate::sha256::sha256_hex(&buf[..end]))
}

#[cfg(not(unix))]
fn probe_hostname_hash() -> Option<String> {
    std::env::var("COMPUTERNAME")
        .ok()
        .map(|hostname| crate::sha256::sha256_hex(hostname.as_bytes()))
}

// ============================ Model provider probe ============================

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
struct ProviderProbe {
    attempted: Vec<String>,
    ok: bool,
    note: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
struct ManifestModel {
    model_id: String,
    model_name: String,
    endpoint: String,
    digest: Option<String>,
    quantization: Option<String>,
    context_window: Option<u64>,
    structured_output_requested: String,
    timeout_secs: usize,
    provider_probe: ProviderProbe,
}

const PROBE_TIMEOUT_SECS: u64 = 10;

/// Tries Ollama's `/api/show` (quantization, context window) and
/// `/api/tags` (digest) first, then falls back to an OpenAI-compatible
/// `/v1/models` (existence only — no digest/quantization/context-window
/// field exists there). Never fatal: a probe failure is recorded in
/// `provider_probe`, not raised, since a model's own reachability is
/// established for real only when its first cell actually calls it
/// (ADR 0003 §9.1's obtainability triage).
fn probe_model(model: &ResolvedModel) -> ManifestModel {
    let mut attempted = Vec::new();
    let base = ManifestModel {
        model_id: model.id.clone(),
        model_name: model.model.clone(),
        endpoint: model.url.clone(),
        digest: None,
        quantization: None,
        context_window: None,
        structured_output_requested: model.structured_output.clone(),
        timeout_secs: model.timeout_secs,
        provider_probe: ProviderProbe::default(),
    };
    let Ok(parsed_url) = url::Url::parse(&model.url) else {
        return ManifestModel {
            provider_probe: ProviderProbe {
                attempted,
                ok: false,
                note: Some("url is unparseable".to_string()),
            },
            ..base
        };
    };
    let Some(host) = parsed_url.host_str() else {
        return ManifestModel {
            provider_probe: ProviderProbe {
                attempted,
                ok: false,
                note: Some("url has no host".to_string()),
            },
            ..base
        };
    };
    let origin = match parsed_url.port() {
        Some(port) => format!("{}://{host}:{port}", parsed_url.scheme()),
        None => format!("{}://{host}", parsed_url.scheme()),
    };
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(PROBE_TIMEOUT_SECS)))
        .http_status_as_error(false)
        .build()
        .into();
    // Built once, applied to every probe request — a gated Ollama-
    // compatible proxy that requires auth on /api/show and /api/tags
    // too must not see those two answer 401 while only the /v1/models
    // fallback carries the credential.
    let auth: Option<String> = model
        .api_key_env
        .as_ref()
        .and_then(|name| std::env::var(name).ok())
        .map(|key| format!("Bearer {key}"));

    attempted.push("POST /api/show".to_string());
    let show_body = serde_json::json!({ "model": model.model }).to_string();
    let mut quantization = None;
    let mut context_window = None;
    let mut show_ok = false;
    let mut show_request = agent
        .post(&format!("{origin}/api/show"))
        .header("Content-Type", "application/json");
    if let Some(auth) = &auth {
        show_request = show_request.header("Authorization", auth);
    }
    if let Ok(mut response) = show_request.send(&show_body)
        && response.status().as_u16() == 200
        && let Ok(text) = response.body_mut().read_to_string()
        && let Ok(value) = serde_json::from_str::<Value>(&text)
    {
        show_ok = true;
        quantization = value["details"]["quantization_level"]
            .as_str()
            .map(str::to_string);
        context_window = value["model_info"].as_object().and_then(|fields| {
            fields
                .iter()
                .find(|(key, _)| key.ends_with("context_length"))
                .and_then(|(_, value)| value.as_u64())
        });
    }
    if !show_ok {
        attempted.push("GET /v1/models".to_string());
        let mut request = agent.get(&format!("{origin}/v1/models"));
        if let Some(auth) = &auth {
            request = request.header("Authorization", auth);
        }
        return match request.call() {
            Ok(response) if response.status().as_u16() == 200 => ManifestModel {
                provider_probe: ProviderProbe {
                    attempted,
                    ok: true,
                    note: Some(
                        "the OpenAI-compatible model list carries no digest, quantization, \
                         or context-window field"
                            .to_string(),
                    ),
                },
                ..base
            },
            Ok(response) => ManifestModel {
                provider_probe: ProviderProbe {
                    attempted,
                    ok: false,
                    note: Some(format!("{origin}/v1/models answered {}", response.status())),
                },
                ..base
            },
            Err(error) => ManifestModel {
                provider_probe: ProviderProbe {
                    attempted,
                    ok: false,
                    note: Some(error.to_string()),
                },
                ..base
            },
        };
    }
    attempted.push("GET /api/tags".to_string());
    let mut digest = None;
    let mut tags_request = agent.get(&format!("{origin}/api/tags"));
    if let Some(auth) = &auth {
        tags_request = tags_request.header("Authorization", auth);
    }
    if let Ok(mut response) = tags_request.call()
        && response.status().as_u16() == 200
        && let Ok(text) = response.body_mut().read_to_string()
        && let Ok(value) = serde_json::from_str::<Value>(&text)
    {
        digest = value["models"].as_array().and_then(|models| {
            models
                .iter()
                .find(|entry| entry["name"].as_str() == Some(model.model.as_str()))
                .and_then(|entry| entry["digest"].as_str())
                .map(str::to_string)
        });
    }
    ManifestModel {
        digest,
        quantization,
        context_window,
        provider_probe: ProviderProbe {
            attempted,
            ok: true,
            note: None,
        },
        ..base
    }
}

#[cfg(test)]
mod probe_model_tests {
    use super::*;

    /// A canned-response probe stub: each connection is answered by the
    /// first route whose prefix matches the request line (`"POST
    /// /api/show"`, `"GET /api/tags"`, `"GET /v1/models"`), with 404 for
    /// anything unrouted. `Connection: close` forces the client to
    /// reconnect per request, so routing stays per-request. The acceptor
    /// thread is spawn-and-forget, the same shape as the extract stubs:
    /// `probe_model` returns only after every response it will ever read
    /// has arrived, so nothing waits on the thread.
    fn stub_probe_server(routes: Vec<(&'static str, u16, String)>) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut buffer = Vec::new();
                let mut chunk = [0u8; 8192];
                let header_end = loop {
                    let Ok(read) = stream.read(&mut chunk) else {
                        break 0;
                    };
                    if read == 0 {
                        break 0;
                    }
                    buffer.extend_from_slice(&chunk[..read]);
                    if let Some(position) =
                        buffer.windows(4).position(|window| window == b"\r\n\r\n")
                    {
                        break position + 4;
                    }
                };
                if header_end == 0 {
                    continue;
                }
                let headers = String::from_utf8_lossy(&buffer[..header_end]).to_string();
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())?
                    })
                    .unwrap_or(0);
                while buffer.len() < header_end + content_length {
                    let Ok(read) = stream.read(&mut chunk) else {
                        break;
                    };
                    if read == 0 {
                        break;
                    }
                    buffer.extend_from_slice(&chunk[..read]);
                }
                let (status, body) = routes
                    .iter()
                    .find(|(prefix, ..)| headers.starts_with(prefix))
                    .map(|(_, status, body)| (*status, body.clone()))
                    .unwrap_or((404, String::new()));
                let response = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        url
    }

    fn model_at(url: &str) -> ResolvedModel {
        ResolvedModel {
            id: "qwen25-7b-q4".to_string(),
            label: None,
            model: "qwen2.5:7b-instruct-q4_K_M".to_string(),
            url: format!("{url}/v1/chat/completions"),
            api_key_env: None,
            structured_output: "auto".to_string(),
            timeout_secs: 120,
            note: None,
        }
    }

    #[test]
    fn ollama_native_facts_reach_the_manifest_model() {
        let url = stub_probe_server(vec![
            (
                "POST /api/show",
                200,
                serde_json::json!({
                    "details": {"quantization_level": "Q4_K_M"},
                    "model_info": {"qwen2.context_length": 32768},
                })
                .to_string(),
            ),
            (
                "GET /api/tags",
                200,
                serde_json::json!({
                    "models": [
                        {"name": "other:latest", "digest": "sha256:other"},
                        {"name": "qwen2.5:7b-instruct-q4_K_M", "digest": "sha256:abc123"},
                    ],
                })
                .to_string(),
            ),
        ]);
        let model = model_at(&url);
        let entry = probe_model(&model);
        assert_eq!(entry.digest.as_deref(), Some("sha256:abc123"));
        assert_eq!(entry.quantization.as_deref(), Some("Q4_K_M"));
        assert_eq!(entry.context_window, Some(32768));
        assert!(entry.provider_probe.ok);
        assert_eq!(entry.provider_probe.note, None);
        assert_eq!(
            entry.provider_probe.attempted,
            vec!["POST /api/show", "GET /api/tags"]
        );
        assert_eq!(entry.model_id, "qwen25-7b-q4");
        assert_eq!(entry.model_name, "qwen2.5:7b-instruct-q4_K_M");
        assert_eq!(entry.endpoint, model.url);
        assert_eq!(entry.structured_output_requested, "auto");
        assert_eq!(entry.timeout_secs, 120);
    }

    #[test]
    fn the_openai_compatible_fallback_records_the_attempt_trail() {
        let url = stub_probe_server(vec![
            ("POST /api/show", 404, "{}".to_string()),
            (
                "GET /v1/models",
                200,
                serde_json::json!({"data": []}).to_string(),
            ),
        ]);
        let entry = probe_model(&model_at(&url));
        assert!(entry.provider_probe.ok);
        assert_eq!(
            entry.provider_probe.attempted,
            vec!["POST /api/show", "GET /v1/models"]
        );
        let note = entry
            .provider_probe
            .note
            .expect("fallback must carry a note");
        assert!(note.contains("carries no digest"), "{note}");
        assert_eq!(entry.digest, None);
        assert_eq!(entry.quantization, None);
        assert_eq!(entry.context_window, None);
    }

    #[test]
    fn a_non_200_fallback_is_recorded_as_not_ok() {
        let url = stub_probe_server(vec![
            ("POST /api/show", 404, "{}".to_string()),
            ("GET /v1/models", 500, "{}".to_string()),
        ]);
        let entry = probe_model(&model_at(&url));
        assert!(!entry.provider_probe.ok);
        assert_eq!(
            entry.provider_probe.attempted,
            vec!["POST /api/show", "GET /v1/models"]
        );
        let note = entry
            .provider_probe
            .note
            .expect("a refusal must carry a note");
        assert!(note.contains("/v1/models answered"), "{note}");
        assert!(note.contains("500"), "{note}");
    }

    #[test]
    fn an_unreachable_endpoint_is_recorded_as_not_ok() {
        // Bind then drop, so the port is known-closed rather than
        // known-slow — both probe requests fail at connect time.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let entry = probe_model(&model_at(&url));
        assert!(!entry.provider_probe.ok);
        assert_eq!(
            entry.provider_probe.attempted,
            vec!["POST /api/show", "GET /v1/models"]
        );
        let note = entry
            .provider_probe
            .note
            .expect("an error must carry a note");
        assert!(!note.is_empty());
    }

    #[test]
    fn an_unparseable_url_is_recorded_without_a_probe() {
        let entry = probe_model(&ResolvedModel {
            url: "not a url".to_string(),
            ..model_at("http://127.0.0.1:1")
        });
        assert!(!entry.provider_probe.ok);
        assert_eq!(
            entry.provider_probe.note.as_deref(),
            Some("url is unparseable")
        );
        assert!(entry.provider_probe.attempted.is_empty());
        assert_eq!(entry.model_id, "qwen25-7b-q4");
    }

    #[test]
    fn a_url_without_a_host_is_recorded_without_a_probe() {
        let entry = probe_model(&ResolvedModel {
            url: "data:text/plain,hello".to_string(),
            ..model_at("http://127.0.0.1:1")
        });
        assert!(!entry.provider_probe.ok);
        assert_eq!(
            entry.provider_probe.note.as_deref(),
            Some("url has no host")
        );
        assert!(entry.provider_probe.attempted.is_empty());
    }
}

// ============================== manifest.json ==============================
//
// ADR 0003 §10: range-acceptance (IMAGE_VERSION posture) — taguru both
// writes and re-reads this file, so `#[serde(default)]` everywhere lets
// an older shape still load, and a revision may only add a field.

const BENCHMARK_MANIFEST_VERSION: u64 = 1;
const BENCHMARK_RUNS_VERSION: u64 = 1;

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
struct HarnessBlock {
    #[serde(default)]
    execution: String,
    #[serde(default)]
    runs_per_model: usize,
    #[serde(default)]
    documents_root: String,
    #[serde(default)]
    document_order: Vec<String>,
    #[serde(default)]
    config_path: String,
    #[serde(default)]
    config_sha256: String,
}

/// The task settings every cell ran under (ADR 0003 §8's fairness
/// invariant, made an artifact). `timeout_secs` here is the resolved
/// *default* the matrix was invoked with; a model whose own
/// `models.json` entry overrides it records that override in
/// `manifest.json`'s own `models[].timeout_secs` instead (ADR 0003 §8
/// names `timeout_secs` a per-model field, not a task setting).
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
struct ExtractionSettings {
    #[serde(default)]
    prompt_version: u32,
    #[serde(default)]
    chunk_bytes: usize,
    #[serde(default)]
    temperature: u32,
    #[serde(default)]
    context: String,
    #[serde(default)]
    questions: usize,
    #[serde(default)]
    fact_budget: usize,
    #[serde(default)]
    no_passage: bool,
    #[serde(default)]
    description: String,
    #[serde(default)]
    max_output_tokens: Option<usize>,
    #[serde(default)]
    max_attempts: usize,
    #[serde(default)]
    parallel: usize,
    #[serde(default)]
    timeout_secs: usize,
    #[serde(default)]
    schema_sha256: String,
    /// Pre-existing gap closed alongside `candidates`: --lossy shapes
    /// what every cell's batches even contain, so a resume under a
    /// flipped flag must invalidate like any other setting change.
    #[serde(default)]
    lossy: bool,
    /// ADR 0014 (#496 S2): the candidate-block control, same reasoning.
    #[serde(default)]
    candidates: bool,
    /// ADR 0015 (#496 S3): the vocabulary content digest ("" = off) —
    /// the same fingerprint extract folds into its own manifests.
    #[serde(default)]
    vocabulary_sha256: String,
    /// Issue #734 (ADR 0003 §5): the remaining TAGURU_EXTRACT_* knobs
    /// `run_cell` pins explicitly — today all at extract's own
    /// defaults, recorded so the manifest names every resolved value
    /// the cells ran under, defaults included. Keep these five in
    /// step with `run_cell`'s scrub-then-pin env block.
    #[serde(default)]
    corrective_context_bytes: Option<usize>,
    #[serde(default)]
    coverage: bool,
    #[serde(default)]
    diagnostics: String,
    #[serde(default)]
    diagnostics_raw_bytes: Option<usize>,
    #[serde(default)]
    context_schema: String,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
struct ManifestCell {
    #[serde(default)]
    cell_id: String,
    #[serde(default)]
    model_id: String,
    #[serde(default)]
    run_index: usize,
    #[serde(default)]
    runs_file: String,
    #[serde(default)]
    cell_dir: String,
    #[serde(default)]
    structured_output_resolved: Option<String>,
    #[serde(default)]
    started_at: String,
    #[serde(default)]
    finished_at: Option<String>,
    /// "complete" | "failed" | "interrupted" — a plain string, not a
    /// closed enum, so a future outcome needs no version bump to add
    /// (ADR 0003 §10's range-acceptance posture applies to values here
    /// too, not only to top-level shape).
    #[serde(default)]
    outcome: String,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
struct BenchManifest {
    #[serde(default)]
    taguru_benchmark_manifest: u64,
    #[serde(default)]
    run_id: String,
    #[serde(default)]
    started_at: String,
    #[serde(default)]
    finished_at: Option<String>,
    #[serde(default)]
    taguru_version: String,
    #[serde(default)]
    sdk_versions: Map<String, Value>,
    #[serde(default)]
    harness: HarnessBlock,
    #[serde(default)]
    extraction_settings: ExtractionSettings,
    #[serde(default)]
    documents: Vec<DocumentInfo>,
    #[serde(default)]
    models: Vec<ManifestModel>,
    #[serde(default)]
    cells: Vec<ManifestCell>,
    #[serde(default)]
    environment: EnvironmentBlock,
}

fn load_bench_manifest(path: &Path) -> Result<BenchManifest, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let manifest: BenchManifest =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    if !(1..=BENCHMARK_MANIFEST_VERSION).contains(&manifest.taguru_benchmark_manifest) {
        return Err(format!(
            "{}: taguru_benchmark_manifest {} is not supported by this build (accepts \
             1..={BENCHMARK_MANIFEST_VERSION})",
            path.display(),
            manifest.taguru_benchmark_manifest
        ));
    }
    Ok(manifest)
}

fn write_bench_manifest(path: &Path, manifest: &BenchManifest) -> io::Result<()> {
    let text = serde_json::to_string_pretty(manifest).expect("a manifest serializes");
    crate::storage::write_atomic(path, text.as_bytes())
}

/// Refuses to resume a results directory whose matrix definition has
/// drifted since it was created (ADR 0003 §6): two cells that ran under
/// different settings must never share a directory, and a changed
/// corpus or `models.json` would mean exactly that. `taguru_version`
/// differing is a warning, not a mismatch — a rebuild of the same
/// taguru is expected to resume its own prior matrix.
fn consistency_mismatch(
    existing: &BenchManifest,
    config_sha256: &str,
    extraction_settings: &ExtractionSettings,
    documents: &[DocumentInfo],
    models: &[ResolvedModel],
) -> Option<String> {
    if existing.harness.config_sha256 != config_sha256 {
        return Some(
            "models.json has changed since this results directory was created".to_string(),
        );
    }
    if &existing.extraction_settings != extraction_settings {
        return Some(
            "benchmark flags differ from this results directory's original run".to_string(),
        );
    }
    if existing.documents.len() != documents.len()
        || existing
            .documents
            .iter()
            .zip(documents)
            .any(|(a, b)| a.document_id != b.document_id || a.sha256 != b.sha256)
    {
        return Some("the corpus has changed since this results directory was created".to_string());
    }
    let existing_ids: Vec<&str> = existing
        .models
        .iter()
        .map(|m| m.model_id.as_str())
        .collect();
    let new_ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();
    if existing_ids != new_ids
        || existing
            .models
            .iter()
            .zip(models)
            .any(|(a, b)| a.model_name != b.model || a.endpoint != b.url)
    {
        return Some(
            "models.json's model set differs from this results directory's original run"
                .to_string(),
        );
    }
    None
}

fn run_label(run_index: usize) -> String {
    format!("run{run_index:02}")
}

fn cell_id_for(model_id: &str, run_index: usize) -> String {
    format!("{model_id}.{}", run_label(run_index))
}

#[cfg(test)]
mod manifest_tests {
    use super::*;

    #[test]
    fn a_version_outside_the_accepted_range_is_refused() {
        let path = std::env::temp_dir().join(format!(
            "taguru-benchmark-manifest-version-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::write(&path, r#"{"taguru_benchmark_manifest":2}"#).unwrap();
        let error = load_bench_manifest(&path).unwrap_err();
        assert!(error.contains("taguru_benchmark_manifest"), "{error}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn an_old_manifest_missing_new_fields_still_loads() {
        let path = std::env::temp_dir().join(format!(
            "taguru-benchmark-manifest-old-{}-{}",
            std::process::id(),
            line!()
        ));
        fs::write(&path, r#"{"taguru_benchmark_manifest":1,"run_id":"abc"}"#).unwrap();
        let manifest = load_bench_manifest(&path).expect("must load with defaults");
        assert_eq!(manifest.run_id, "abc");
        assert!(manifest.documents.is_empty());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_changed_config_sha256_is_a_mismatch() {
        let existing = BenchManifest {
            harness: HarnessBlock {
                config_sha256: "old".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let reason =
            consistency_mismatch(&existing, "new", &ExtractionSettings::default(), &[], &[]);
        assert!(reason.is_some());
    }

    #[test]
    fn matching_state_is_not_a_mismatch() {
        let settings = ExtractionSettings {
            context: "sake".to_string(),
            ..Default::default()
        };
        let existing = BenchManifest {
            harness: HarnessBlock {
                config_sha256: "same".to_string(),
                ..Default::default()
            },
            extraction_settings: settings.clone(),
            ..Default::default()
        };
        assert!(consistency_mismatch(&existing, "same", &settings, &[], &[]).is_none());
    }

    #[test]
    fn cell_id_is_the_model_id_and_a_zero_padded_run_label() {
        assert_eq!(cell_id_for("qwen25-7b-q4", 1), "qwen25-7b-q4.run01");
        assert_eq!(cell_id_for("qwen25-7b-q4", 12), "qwen25-7b-q4.run12");
    }
}

// ============================== Cell execution ==============================

/// Reads whatever new complete lines a still-growing file has, buffering
/// a trailing partial line across polls. A regular file's `read`
/// returns `0` at the current end without blocking (unlike a pipe or
/// TTY), so this is the same polling shape `tail -f` uses without
/// depending on the file existing yet — `--diagnostics-out`'s sink opens
/// slightly after the child spawns.
struct DiagnosticsTail {
    path: PathBuf,
    file: Option<fs::File>,
    buf: Vec<u8>,
}

impl DiagnosticsTail {
    fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            file: None,
            buf: Vec::new(),
        }
    }

    fn poll(&mut self) -> Vec<String> {
        if self.file.is_none() {
            self.file = fs::File::open(&self.path).ok();
        }
        let Some(file) = &mut self.file else {
            return Vec::new();
        };
        let mut chunk = [0u8; 8192];
        loop {
            match file.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => self.buf.extend_from_slice(&chunk[..n]),
                Err(_) => break,
            }
        }
        let mut lines = Vec::new();
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = self.buf.drain(..=pos).collect();
            if let Ok(text) = String::from_utf8(line_bytes) {
                let trimmed = text.trim_end_matches(['\n', '\r']);
                if !trimmed.is_empty() {
                    lines.push(trimmed.to_string());
                }
            }
        }
        lines
    }
}

/// Appends one JSONL line at a time, flushing every write (no fsync —
/// this sidecar is advisory the same way `--diagnostics-out` itself is;
/// `manifest.json`'s own atomic writes are the durable record). Opens
/// in append mode unconditionally: a fresh file gets its header written
/// immediately after creation, a resumed cell's existing lines are left
/// untouched and new ones land after them (ADR 0003 §6's resume design
/// — see `run_cell`'s doc comment for what that implies about a
/// resumed document's records).
struct RunsWriter {
    file: fs::File,
    path: PathBuf,
    /// Set once a record fails to serialize, so that failure mode is
    /// reported exactly once. Independent of `io_warned` below: a
    /// serialization failure must not suppress a LATER, unrelated
    /// disk-full or permission error from being reported too — both
    /// are real desyncs between `documents_written`/`attempts_total`
    /// (advisory counters that keep counting regardless) and the
    /// file's actual contents, and neither may silently mask the
    /// other.
    serialize_warned: bool,
    /// Set once a write or flush fails, so a disk-full or permission
    /// error is reported exactly once. See `serialize_warned` above.
    io_warned: bool,
}

impl RunsWriter {
    fn open(path: &Path, header: Option<&Value>) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        let mut writer = Self {
            file,
            path: path.to_path_buf(),
            serialize_warned: false,
            io_warned: false,
        };
        if let Some(header) = header {
            writer.write_value(header);
        }
        Ok(writer)
    }

    fn write_value(&mut self, value: &Value) {
        let mut line = match serde_json::to_string(value) {
            Ok(line) => line,
            Err(error) => {
                if !self.serialize_warned {
                    self.serialize_warned = true;
                    eprintln!(
                        "taguru: benchmark: warning: {}: failed to serialize a record ({error}) \
                         — it was dropped, so documents_written/attempts_total may now \
                         overcount what this file holds",
                        self.path.display()
                    );
                }
                return;
            }
        };
        line.push('\n');
        let result = self
            .file
            .write_all(line.as_bytes())
            .and_then(|()| self.file.flush());
        if let Err(error) = result
            && !self.io_warned
        {
            self.io_warned = true;
            eprintln!(
                "taguru: benchmark: warning: {}: write failed ({error}) — \
                 documents_written/attempts_total may now overcount what this file holds",
                self.path.display()
            );
        }
    }
}

/// Counts prior activity in an already-existing `runs/*.jsonl` (a
/// resumed cell) so the final `kind: "cell"` summary is a true
/// cumulative total across both the interrupted attempt and this retry
/// — cheaper than replaying the whole file's state, and all that a
/// summary statistic needs.
fn seed_counts_from_existing_runs_file(path: &Path) -> (usize, usize) {
    let Ok(text) = fs::read_to_string(path) else {
        return (0, 0);
    };
    let mut attempts = 0usize;
    let mut documents_written = 0usize;
    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        match value.get("kind").and_then(Value::as_str) {
            Some("attempt") => attempts += 1,
            Some("document") if value.get("phase").and_then(Value::as_str) == Some("end") => {
                documents_written += 1;
            }
            _ => {}
        }
    }
    (attempts, documents_written)
}

/// The token right after `"structured output: "` in one of the five
/// lines `resolve_response_format` prints (extract.rs), which names the
/// rung that ACTUALLY carried it — never a substring match, since the
/// json_object-fallback message itself mentions "json_schema" in its
/// own parenthetical explanation of why it fell back.
fn parse_structured_output_line(line: &str) -> Option<String> {
    let marker = "structured output: ";
    let rest = &line[line.find(marker)? + marker.len()..];
    match rest.split_whitespace().next()? {
        "json_schema" => Some("json_schema".to_string()),
        "json_object" => Some("json_object".to_string()),
        "prompted" => Some("prompted".to_string()),
        _ => None,
    }
}

/// Context threaded through every diagnostics line transcribed into
/// `runs/*.jsonl` for one cell — grouped so the transcription function
/// below does not carry eight separate parameters.
struct CellRunsContext<'a> {
    dictionary: &'a BTreeMap<String, DocumentInfo>,
    cell_id: String,
    model_id: String,
    run_index: usize,
    cell_dir: PathBuf,
    cell_dir_rel: PathBuf,
}

impl CellRunsContext<'_> {
    /// The child reports `batch_path` under whatever `--out` it was
    /// given (`cell_dir`, possibly prefixed by the results directory);
    /// re-anchored under `cell_dir_rel` so `runs/*.jsonl` stays
    /// portable, matching ADR 0003 §9.2's relative example paths.
    fn relativize_batch_path(&self, batch_path: &str) -> String {
        match Path::new(batch_path).strip_prefix(&self.cell_dir) {
            Ok(rel) => self
                .cell_dir_rel
                .join(rel)
                .to_string_lossy()
                .replace('\\', "/"),
            Err(_) => batch_path.to_string(),
        }
    }
}

/// Transcribes one child diagnostics line into `runs/*.jsonl` (ADR 0003
/// §9.2): synthesizes a `document`(`phase: "start"`) the first time a
/// source is seen this session, denormalizes `chunk`/`attempt` records
/// from the preflight-computed dictionary (never from a live `chunk`
/// line — the dictionary is fully deterministic from the corpus alone,
/// so this is robust to a resumed document's checkpointed chunks never
/// re-emitting their own `chunk` line), and closes a document on its
/// `document` line. `open` tracks document ids started but not yet
/// closed THIS invocation; whatever remains in it once the child exits
/// is either left alone (interrupted) or closed with `outcome: "failed"`
/// by the caller.
fn transcribe_diagnostics_line(
    raw_line: &str,
    ctx: &CellRunsContext,
    writer: &mut RunsWriter,
    open: &mut BTreeSet<String>,
    attempts_total: &mut usize,
    documents_written: &mut usize,
) {
    let Ok(mut map) = serde_json::from_str::<Map<String, Value>>(raw_line) else {
        return;
    };
    let Some(kind) = map.get("kind").and_then(Value::as_str).map(str::to_string) else {
        return;
    };
    let Some(source) = map
        .get("source")
        .and_then(Value::as_str)
        .map(str::to_string)
    else {
        return;
    };
    let Some(doc) = ctx.dictionary.get(&source) else {
        return;
    };
    let document_id = doc.document_id.clone();
    let now = now_unix_secs() as f64;

    if !open.contains(&document_id) {
        open.insert(document_id.clone());
        writer.write_value(&serde_json::json!({
            "kind": "document",
            "ts": now,
            "cell_id": ctx.cell_id,
            "document_id": document_id,
            "source": source,
            "document_sha256": doc.sha256,
            "chunk_total": doc.chunk_total,
            "phase": "start",
        }));
    }

    match kind.as_str() {
        "chunk" => {
            map.insert("ts".to_string(), serde_json::json!(now));
            map.insert("cell_id".to_string(), serde_json::json!(ctx.cell_id));
            map.insert("document_id".to_string(), serde_json::json!(document_id));
            writer.write_value(&Value::Object(map));
        }
        "attempt" => {
            let chunk_index = map.get("chunk_index").and_then(Value::as_u64).unwrap_or(0) as usize;
            map.insert("ts".to_string(), serde_json::json!(now));
            map.insert("cell_id".to_string(), serde_json::json!(ctx.cell_id));
            map.insert("model_id".to_string(), serde_json::json!(ctx.model_id));
            map.insert("run_index".to_string(), serde_json::json!(ctx.run_index));
            map.insert("document_id".to_string(), serde_json::json!(document_id));
            map.insert("document_sha256".to_string(), serde_json::json!(doc.sha256));
            if let Some(chunk) = doc.chunks.get(chunk_index) {
                map.insert(
                    "chunk_sha256".to_string(),
                    serde_json::json!(chunk.chunk_sha256),
                );
                map.insert(
                    "paragraph_first".to_string(),
                    serde_json::json!(chunk.paragraph_first),
                );
                map.insert(
                    "paragraph_last".to_string(),
                    serde_json::json!(chunk.paragraph_last),
                );
            }
            *attempts_total += 1;
            writer.write_value(&Value::Object(map));
        }
        "document" => {
            open.remove(&document_id);
            *documents_written += 1;
            let batch_path = map
                .get("batch_path")
                .and_then(Value::as_str)
                .map(|path| ctx.relativize_batch_path(path));
            writer.write_value(&serde_json::json!({
                "kind": "document",
                "ts": now,
                "cell_id": ctx.cell_id,
                "document_id": document_id,
                "source": source,
                "document_sha256": doc.sha256,
                "phase": "end",
                "outcome": "written",
                "associations": map.get("associations"),
                "concepts": map.get("concepts"),
                "labels": map.get("labels"),
                "questions": map.get("questions"),
                "duplicates": map.get("duplicates"),
                "dropped": map.get("dropped"),
                "batch_path": batch_path,
            }));
        }
        _ => {}
    }
}

/// Every document still open when the cell concludes cleanly (exit code
/// 0 or 1) never reached a legitimate `document` line — its extraction
/// did not complete, ordering-independent of which document that was or
/// how many others succeeded around it. Not called on an interrupted
/// cell: there, a bare `start` with no matching `end` IS the
/// interruption signal (ADR 0003 §9.2), not a failure.
fn synthesize_failed_ends(
    open: &BTreeSet<String>,
    dictionary: &BTreeMap<String, DocumentInfo>,
    writer: &mut RunsWriter,
    cell_id: &str,
) {
    for (source, doc) in dictionary {
        if open.contains(&doc.document_id) {
            writer.write_value(&serde_json::json!({
                "kind": "document",
                "ts": now_unix_secs() as f64,
                "cell_id": cell_id,
                "document_id": doc.document_id,
                "source": source,
                "document_sha256": doc.sha256,
                "phase": "end",
                "outcome": "failed",
                "associations": Value::Null,
                "concepts": Value::Null,
                "labels": Value::Null,
                "questions": Value::Null,
                "duplicates": Value::Null,
                "dropped": Value::Null,
                "batch_path": Value::Null,
            }));
        }
    }
}

/// Maps a concluded child's exit status to a cell outcome. `2` is
/// `extract`'s own usage-error code — reachable not only from a
/// benchmark-constructed command line (a real harness bug) but from a
/// `models.json` entry that parses here yet `extract` itself rejects at
/// startup (`ChatClient::from_env`'s own validation). Recording the
/// cell as `failed`, exit code included, keeps the rest of the matrix
/// running instead of one misconfigured model aborting every other
/// cell.
fn outcome_for_exit_code(exit_code: Option<i32>, cell_id: &str) -> Result<&'static str, String> {
    match exit_code {
        Some(0) => Ok("complete"),
        Some(1) | Some(2) => Ok("failed"),
        Some(130) | None => Ok("interrupted"),
        Some(other) => Err(format!(
            "extract exited with an unexpected code ({other}) for cell {cell_id} — this is a \
             benchmark harness bug, not a usage error"
        )),
    }
}

/// Builds and runs one (model, run) cell's `taguru extract` subprocess
/// (ADR 0003 §5/§6): the `TAGURU_EXTRACT_*` namespace is scrubbed from
/// the child's inherited environment, then set explicitly (every
/// resolved value, including ones left at defaults) — R2, the cell owns
/// its environment. `--out`/`--diagnostics-out` are always this cell's
/// own fresh-or-resumed directory — R3. Resuming an interrupted cell
/// re-invokes the exact same command into the exact same `cell_dir`,
/// never `--force`, so `extract`'s own `.extract-manifest.json`/
/// `.extract-checkpoints/` do the document/chunk-level resume (issue
/// #179) — the accepted cost is that a document open at the moment of
/// interruption may show a second `start` record and repeat some
/// `chunk` records on this retry (ADR 0003 §6), since every consumer
/// joins by key, never by line position.
#[allow(clippy::too_many_arguments)]
fn run_cell(
    bench_args: &BenchArgs,
    model: &ResolvedModel,
    run_index: usize,
    dictionary: &BTreeMap<String, DocumentInfo>,
    run_id: &str,
) -> Result<ManifestCell, String> {
    let cell_id = cell_id_for(&model.id, run_index);
    let cell_dir_rel = PathBuf::from("cells")
        .join(&model.id)
        .join(run_label(run_index));
    let cell_dir = bench_args.out.join(&cell_dir_rel);
    fs::create_dir_all(&cell_dir)
        .map_err(|error| format!("creating {}: {error}", cell_dir.display()))?;
    let diagnostics_path = cell_dir.join("diagnostics.jsonl");
    let runs_file_rel =
        PathBuf::from("runs").join(format!("{}.{}.jsonl", model.id, run_label(run_index)));
    let runs_file_path = bench_args.out.join(&runs_file_rel);

    let is_resume = runs_file_path.is_file();
    let (mut attempts_total, mut documents_written) = if is_resume {
        seed_counts_from_existing_runs_file(&runs_file_path)
    } else {
        (0, 0)
    };
    let header = if is_resume {
        None
    } else {
        Some(serde_json::json!({
            "kind": "header",
            "taguru_benchmark_runs": BENCHMARK_RUNS_VERSION,
            "run_id": run_id,
            "cell_id": cell_id,
            "model_id": model.id,
            "model_name": model.model,
            "run_index": run_index,
            "prompt_version": crate::extract::PROMPT_VERSION,
        }))
    };
    let mut writer = RunsWriter::open(&runs_file_path, header.as_ref())
        .map_err(|error| format!("opening {}: {error}", runs_file_path.display()))?;

    let exe =
        std::env::current_exe().map_err(|error| format!("locating the running binary: {error}"))?;
    let mut cmd = Command::new(exe);
    cmd.arg("extract");
    for key in crate::config::KNOWN_KEYS
        .iter()
        .copied()
        .filter(|key| key.starts_with("TAGURU_EXTRACT_"))
    {
        cmd.env_remove(key);
    }
    cmd.env("TAGURU_EXTRACT_URL", &model.url);
    cmd.env("TAGURU_EXTRACT_MODEL", &model.model);
    cmd.env(
        "TAGURU_EXTRACT_TIMEOUT_SECS",
        model.timeout_secs.to_string(),
    );
    if let Some(env_name) = &model.api_key_env
        && let Ok(key) = std::env::var(env_name)
    {
        cmd.env("TAGURU_EXTRACT_API_KEY", key);
    }
    cmd.env("TAGURU_EXTRACT_PARALLEL", bench_args.parallel.to_string());
    if let Some(budget) = bench_args.fact_budget {
        cmd.env("TAGURU_EXTRACT_FACT_BUDGET", budget.to_string());
    }
    cmd.env(
        "TAGURU_EXTRACT_MAX_ATTEMPTS",
        bench_args.max_attempts.to_string(),
    );
    cmd.env("TAGURU_EXTRACT_STRUCTURED_OUTPUT", &model.structured_output);
    if let Some(tokens) = bench_args.max_output_tokens {
        cmd.env("TAGURU_EXTRACT_MAX_OUTPUT_TOKENS", tokens.to_string());
    }
    cmd.env(
        "TAGURU_EXTRACT_LOSSY",
        if bench_args.lossy { "1" } else { "0" },
    );
    cmd.env(
        "TAGURU_EXTRACT_CANDIDATES",
        if bench_args.candidates { "1" } else { "0" },
    );
    match &bench_args.vocabulary {
        Some(path) => cmd.env("TAGURU_EXTRACT_VOCABULARY", path),
        None => cmd.env("TAGURU_EXTRACT_VOCABULARY", ""),
    };
    // ADR 0003 §5 (issue #734): the remaining TAGURU_EXTRACT_* knobs,
    // pinned EXPLICITLY at extract's own defaults ("" = unset for
    // every one of them) so a cell's whole extraction environment is
    // this block — never the launching shell — and the manifest's
    // `extraction_settings` mirror of these values stays honest.
    // (Keep this block and `ExtractionSettings`' matching fields in
    // step with config.rs's TAGURU_EXTRACT_* inventory.)
    cmd.env("TAGURU_EXTRACT_CORRECTIVE_CONTEXT_BYTES", "");
    cmd.env("TAGURU_EXTRACT_COVERAGE", "0");
    cmd.env("TAGURU_EXTRACT_DIAGNOSTICS", "");
    cmd.env("TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES", "");
    cmd.env("TAGURU_EXTRACT_SCHEMA", "");

    cmd.arg("--context").arg(&bench_args.context);
    if bench_args.questions > 0 {
        cmd.arg("--questions").arg(bench_args.questions.to_string());
    }
    if bench_args.no_passage {
        cmd.arg("--no-passage");
    }
    if let Some(description) = &bench_args.description {
        cmd.arg("--description").arg(description);
    }
    cmd.arg("--out").arg(&cell_dir);
    cmd.arg("--diagnostics-out").arg(&diagnostics_path);
    cmd.arg(&bench_args.corpus);

    cmd.stdin(Stdio::null());
    let stdout_log = fs::File::create(cell_dir.join("stdout.log"))
        .map_err(|error| format!("creating stdout.log: {error}"))?;
    cmd.stdout(Stdio::from(stdout_log));
    cmd.stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|error| format!("spawning extract: {error}"))?;
    let child_stderr = child.stderr.take().expect("stderr is piped");
    let stderr_log_path = cell_dir.join("stderr.log");
    let resolved_rung: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let resolved_rung_writer = Arc::clone(&resolved_rung);
    let stderr_thread = std::thread::spawn(move || {
        let Ok(mut log) = fs::File::create(&stderr_log_path) else {
            return;
        };
        for line in io::BufReader::new(child_stderr)
            .lines()
            .map_while(Result::ok)
        {
            if let Some(rung) = parse_structured_output_line(&line) {
                *resolved_rung_writer
                    .lock()
                    .unwrap_or_else(|p| p.into_inner()) = Some(rung);
            }
            let _ = writeln!(log, "{line}");
        }
    });

    let started_at = iso8601_utc(now_unix_secs());
    let mut tail = DiagnosticsTail::new(&diagnostics_path);
    let mut open: BTreeSet<String> = BTreeSet::new();
    let ctx = CellRunsContext {
        dictionary,
        cell_id: cell_id.clone(),
        model_id: model.id.clone(),
        run_index,
        cell_dir: cell_dir.clone(),
        cell_dir_rel: cell_dir_rel.clone(),
    };
    loop {
        for line in tail.poll() {
            transcribe_diagnostics_line(
                &line,
                &ctx,
                &mut writer,
                &mut open,
                &mut attempts_total,
                &mut documents_written,
            );
        }
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(_) => break,
        }
    }
    for line in tail.poll() {
        transcribe_diagnostics_line(
            &line,
            &ctx,
            &mut writer,
            &mut open,
            &mut attempts_total,
            &mut documents_written,
        );
    }
    let status = child
        .wait()
        .map_err(|error| format!("waiting on extract: {error}"))?;
    let _ = stderr_thread.join();

    let exit_code = status.code();
    let _ = fs::write(
        cell_dir.join("exit_code"),
        exit_code.map(|code| code.to_string()).unwrap_or_default(),
    );

    let outcome = outcome_for_exit_code(exit_code, &cell_id)?;
    if outcome != "interrupted" {
        synthesize_failed_ends(&open, dictionary, &mut writer, &cell_id);
        writer.write_value(&serde_json::json!({
            "kind": "cell",
            "ts": now_unix_secs() as f64,
            "cell_id": cell_id,
            "outcome": outcome,
            "documents_written": documents_written,
            "attempts_total": attempts_total,
            "exit_code": exit_code,
        }));
    }

    let structured_output_resolved = if model.structured_output == "off" {
        Some("off".to_string())
    } else {
        resolved_rung
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    };

    Ok(ManifestCell {
        cell_id,
        model_id: model.id.clone(),
        run_index,
        runs_file: runs_file_rel.to_string_lossy().replace('\\', "/"),
        cell_dir: cell_dir_rel.to_string_lossy().replace('\\', "/"),
        structured_output_resolved,
        started_at,
        finished_at: Some(iso8601_utc(now_unix_secs())),
        outcome: outcome.to_string(),
    })
}

#[cfg(test)]
mod cell_runs_tests {
    use super::*;

    #[test]
    fn exit_code_2_is_recorded_failed_not_a_harness_bug() {
        assert_eq!(outcome_for_exit_code(Some(0), "m.run01"), Ok("complete"));
        assert_eq!(outcome_for_exit_code(Some(1), "m.run01"), Ok("failed"));
        assert_eq!(
            outcome_for_exit_code(Some(2), "m.run01"),
            Ok("failed"),
            "extract's own usage-error code must not abort the whole matrix"
        );
        assert_eq!(
            outcome_for_exit_code(Some(130), "m.run01"),
            Ok("interrupted")
        );
        assert_eq!(outcome_for_exit_code(None, "m.run01"), Ok("interrupted"));
        assert!(outcome_for_exit_code(Some(42), "m.run01").is_err());
    }

    #[test]
    fn structured_output_rung_lines_parse_the_first_token_only() {
        assert_eq!(
            parse_structured_output_line(
                "taguru: extract: structured output: json_schema (pinned)"
            ),
            Some("json_schema".to_string())
        );
        assert_eq!(
            parse_structured_output_line(
                "taguru: extract: structured output: json_object (the json_schema probe failed)"
            ),
            Some("json_object".to_string()),
            "the fallback explanation mentions json_schema but the RESOLVED rung is json_object"
        );
        assert_eq!(
            parse_structured_output_line(
                "taguru: extract: structured output: prompted JSON only (both probes failed)"
            ),
            Some("prompted".to_string())
        );
        assert_eq!(parse_structured_output_line("unrelated line"), None);
    }

    #[test]
    fn batch_path_is_relativized_to_the_results_directory() {
        let ctx = CellRunsContext {
            dictionary: &BTreeMap::new(),
            cell_id: "m.run01".to_string(),
            model_id: "m".to_string(),
            run_index: 1,
            cell_dir: PathBuf::from("/tmp/results/cells/m/run01"),
            cell_dir_rel: PathBuf::from("cells/m/run01"),
        };
        assert_eq!(
            ctx.relativize_batch_path("/tmp/results/cells/m/run01/brewery.md.jsonl"),
            "cells/m/run01/brewery.md.jsonl"
        );
        // A path that doesn't share the prefix is passed through as-is
        // rather than panicking.
        assert_eq!(
            ctx.relativize_batch_path("elsewhere.jsonl"),
            "elsewhere.jsonl"
        );
    }

    #[test]
    fn transcription_opens_a_document_once_and_denormalizes_from_the_dictionary() {
        let mut dictionary = BTreeMap::new();
        dictionary.insert(
            "corpus/a.md".to_string(),
            DocumentInfo {
                document_id: "a".to_string(),
                path: "corpus/a.md".to_string(),
                bytes: 10,
                sha256: "docsha".to_string(),
                paragraph_count: 1,
                chunk_total: 1,
                chunks: vec![ChunkInfo {
                    chunk_index: 0,
                    chunk_sha256: "chunksha".to_string(),
                    chunk_bytes: 10,
                    paragraph_first: 0,
                    paragraph_last: 0,
                }],
            },
        );
        let path = std::env::temp_dir().join(format!(
            "taguru-benchmark-transcribe-{}-{}",
            std::process::id(),
            line!()
        ));
        let mut writer = RunsWriter::open(&path, None).unwrap();
        let mut open = BTreeSet::new();
        let mut attempts = 0;
        let mut docs_written = 0;
        let ctx = CellRunsContext {
            dictionary: &dictionary,
            cell_id: "m.run01".to_string(),
            model_id: "m".to_string(),
            run_index: 1,
            cell_dir: PathBuf::from("/out/cells/m/run01"),
            cell_dir_rel: PathBuf::from("cells/m/run01"),
        };
        let attempt_line = r#"{"kind":"attempt","source":"corpus/a.md","stage":"item","chunk_index":0,"attempt":1,"max_attempts":2,"state":"stop_valid","length_limited":false,"elapsed_seconds":1.0,"provider_metadata":null,"parse_error":null,"validation_issues":null}"#;
        transcribe_diagnostics_line(
            attempt_line,
            &ctx,
            &mut writer,
            &mut open,
            &mut attempts,
            &mut docs_written,
        );
        assert_eq!(attempts, 1);
        assert_eq!(open.len(), 1, "the document opened but has not closed yet");
        drop(writer);

        let text = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2, "one synthesized start, one attempt");
        let start: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(start["kind"], "document");
        assert_eq!(start["phase"], "start");
        assert_eq!(start["document_id"], "a");
        let attempt: Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(attempt["kind"], "attempt");
        assert_eq!(attempt["chunk_sha256"], "chunksha");
        assert_eq!(attempt["paragraph_first"], 0);
        assert_eq!(attempt["document_id"], "a");
        assert_eq!(
            attempt["state"], "stop_valid",
            "Layer 1 carries the original field verbatim"
        );
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn a_document_still_open_at_cleanup_is_closed_as_failed() {
        let mut dictionary = BTreeMap::new();
        dictionary.insert(
            "corpus/a.md".to_string(),
            DocumentInfo {
                document_id: "a".to_string(),
                path: "corpus/a.md".to_string(),
                sha256: "docsha".to_string(),
                ..Default::default()
            },
        );
        let mut open = BTreeSet::new();
        open.insert("a".to_string());
        let path = std::env::temp_dir().join(format!(
            "taguru-benchmark-failed-end-{}-{}",
            std::process::id(),
            line!()
        ));
        let mut writer = RunsWriter::open(&path, None).unwrap();
        synthesize_failed_ends(&open, &dictionary, &mut writer, "m.run01");
        drop(writer);
        let text = fs::read_to_string(&path).unwrap();
        let value: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(value["phase"], "end");
        assert_eq!(value["outcome"], "failed");
        assert!(value["associations"].is_null());
        let _ = fs::remove_file(&path);
    }
}

// ============================== Orchestration ==============================

fn run_extract(args: &[String]) -> i32 {
    // Issue #213's reasoning applies here too: benchmark spawns its own
    // StopSignal listener thread below, and that must happen before any
    // other thread (the stderr tee threads run_cell spawns per cell)
    // exists, so the delivery-eligible signal mask stays clear
    // everywhere except that one dedicated listener.
    crate::extract::block_stop_signals_on_this_thread();
    let args = match BenchArgs::parse(args) {
        Ok(args) => args,
        Err(code) => return code,
    };

    let (models_bytes, models, warnings) = match load_models_file(&args.models) {
        Ok(v) => v,
        Err(message) => {
            eprintln!("taguru: benchmark: {message}");
            return 2;
        }
    };
    for warning in &warnings {
        eprintln!("taguru: benchmark: {warning}");
    }

    let documents = match build_document_dictionary(&args.corpus) {
        Ok(documents) => documents,
        Err(message) => return subcommand_usage_error("benchmark", &message),
    };
    let dictionary: BTreeMap<String, DocumentInfo> = documents
        .iter()
        .map(|doc| (doc.path.clone(), doc.clone()))
        .collect();

    let config_sha256 = crate::sha256::sha256_hex(&models_bytes);
    let schema_sha256 = crate::sha256::sha256_hex(
        crate::extract::json_schema_response_format()
            .to_string()
            .as_bytes(),
    );
    let vocabulary_sha256 = match &args.vocabulary {
        Some(path) => match crate::extract::vocabulary_digest(path) {
            Ok(digest) => digest,
            Err(message) => {
                eprintln!("taguru: benchmark: --vocabulary: {message}");
                return 1;
            }
        },
        None => String::new(),
    };
    let extraction_settings = ExtractionSettings {
        prompt_version: crate::extract::PROMPT_VERSION,
        chunk_bytes: crate::extract::CHUNK_BYTES,
        temperature: 0,
        context: args.context.clone(),
        questions: args.questions,
        fact_budget: args.fact_budget.unwrap_or(0),
        no_passage: args.no_passage,
        description: args.description.clone().unwrap_or_default(),
        max_output_tokens: args.max_output_tokens,
        max_attempts: args.max_attempts,
        parallel: args.parallel,
        timeout_secs: crate::extract::DEFAULT_TIMEOUT_SECS,
        schema_sha256,
        lossy: args.lossy,
        candidates: args.candidates,
        vocabulary_sha256,
        // The five knobs run_cell pins at extract's defaults (issue
        // #734) — mirrored here verbatim; "" / None / false all mean
        // "explicitly the default".
        corrective_context_bytes: None,
        coverage: false,
        diagnostics: String::new(),
        diagnostics_raw_bytes: None,
        context_schema: String::new(),
    };

    if let Err(error) = fs::create_dir_all(&args.out) {
        eprintln!(
            "taguru: benchmark: creating {}: {error}",
            args.out.display()
        );
        return 1;
    }
    let manifest_path = args.out.join("manifest.json");
    let lock_path = args.out.join("models.lock.json");

    let mut manifest = if manifest_path.is_file() {
        let mut existing = match load_bench_manifest(&manifest_path) {
            Ok(manifest) => manifest,
            Err(message) => {
                eprintln!("taguru: benchmark: {message}");
                return 2;
            }
        };
        if let Some(reason) = consistency_mismatch(
            &existing,
            &config_sha256,
            &extraction_settings,
            &documents,
            &models,
        ) {
            eprintln!(
                "taguru: benchmark: {reason} — use a new --out directory (ADR 0003 §6: a cell \
                 writes only into its own directory)"
            );
            return 2;
        }
        if existing.taguru_version != env!("CARGO_PKG_VERSION") {
            eprintln!(
                "taguru: benchmark: resuming a results directory written by taguru {} (this \
                 build is {}) — continuing",
                existing.taguru_version,
                env!("CARGO_PKG_VERSION")
            );
        }
        // Resuming with a wider --runs than the directory was created
        // with is allowed (it only adds cells); the manifest must keep
        // describing the widest matrix it actually holds cells for, or
        // a reader computing per-model coverage from this field alone
        // would undercount.
        existing.harness.runs_per_model = existing.harness.runs_per_model.max(args.runs);
        existing
    } else {
        let manifest_models: Vec<ManifestModel> = models.iter().map(probe_model).collect();
        if let Err(error) = write_models_lock(&lock_path, &models) {
            eprintln!(
                "taguru: benchmark: writing {}: {error}",
                lock_path.display()
            );
            return 1;
        }
        let manifest = BenchManifest {
            taguru_benchmark_manifest: BENCHMARK_MANIFEST_VERSION,
            run_id: generate_run_id(),
            started_at: iso8601_utc(now_unix_secs()),
            finished_at: None,
            taguru_version: env!("CARGO_PKG_VERSION").to_string(),
            sdk_versions: Map::new(),
            harness: HarnessBlock {
                execution: "subprocess".to_string(),
                runs_per_model: args.runs,
                documents_root: args.corpus.clone(),
                document_order: documents.iter().map(|doc| doc.path.clone()).collect(),
                config_path: args.models.display().to_string(),
                config_sha256: config_sha256.clone(),
            },
            extraction_settings: extraction_settings.clone(),
            documents: documents.clone(),
            models: manifest_models,
            cells: Vec::new(),
            environment: environment_block(),
        };
        if let Err(error) = write_bench_manifest(&manifest_path, &manifest) {
            eprintln!(
                "taguru: benchmark: writing {}: {error}",
                manifest_path.display()
            );
            return 1;
        }
        manifest
    };

    let stop = crate::extract::StopSignal::install("benchmark");
    let mut interrupted = false;
    'cells: for model in &models {
        for run_index in 1..=args.runs {
            if stop.check() {
                interrupted = true;
                break 'cells;
            }
            let id = cell_id_for(&model.id, run_index);
            let already_done = manifest.cells.iter().any(|cell| {
                cell.cell_id == id && matches!(cell.outcome.as_str(), "complete" | "failed")
            });
            if already_done {
                continue;
            }
            match run_cell(&args, model, run_index, &dictionary, &manifest.run_id) {
                Ok(cell) => {
                    let cell_interrupted = cell.outcome == "interrupted";
                    manifest.cells.retain(|existing| existing.cell_id != id);
                    manifest.cells.push(cell);
                    if let Err(error) = write_bench_manifest(&manifest_path, &manifest) {
                        eprintln!(
                            "taguru: benchmark: writing {}: {error}",
                            manifest_path.display()
                        );
                        return 1;
                    }
                    if cell_interrupted {
                        interrupted = true;
                        break 'cells;
                    }
                }
                Err(message) => {
                    eprintln!("taguru: benchmark: {message}");
                    return 1;
                }
            }
        }
    }

    if !interrupted {
        manifest.finished_at = Some(iso8601_utc(now_unix_secs()));
        if let Err(error) = write_bench_manifest(&manifest_path, &manifest) {
            eprintln!(
                "taguru: benchmark: writing {}: {error}",
                manifest_path.display()
            );
            return 1;
        }
    }
    if interrupted { 130 } else { 0 }
}
