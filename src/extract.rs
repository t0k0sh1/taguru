//! `taguru extract`: documents → batch files, through an
//! OpenAI-compatible chat model — the producer half of `taguru
//! import`. It reads .md/.txt documents, has the model decompose each
//! into associations under the /protocol ingest discipline, and
//! writes one JSONL batch file per document, the document's path as
//! the source id. Extraction quality is the model's; the contract is
//! enforced here — caps, in-document dedup, alias sanity — and every
//! emitted file is re-parsed with the import parser before it is
//! written, so extract cannot produce a file import refuses.
//!
//! The server never holds model credentials; extract keeps that
//! boundary. It is an offline producer carrying TAGURU_EXTRACT_* in
//! its own environment, exactly like the agent-side pipelines
//! docs/import.html describes — packaged as a subcommand. Vendor APIs
//! (Bedrock, native Anthropic) bridge the same way embeddings do:
//! LiteLLM or any proxy speaking /chat/completions.
//!
//! Extraction is the expensive step (model calls per document), so a
//! manifest in the output directory records what each batch file was
//! computed from — document hash × model × prompt version × target
//! context — and unchanged documents are skipped (`--force`
//! overrides). Import is idempotent, so re-running the whole pipeline
//! is always safe.
//!
//! Split into submodules by concern: `args` parses the CLI into
//! `Args`/`Outcome`/`CorrectionPolicy`/`StructuredOutputMode`/
//! `LadderConfig`; `candidates` segments a document's own names for
//! the prompt's candidate block (ADR 0014); `mechanical` is ADR 0013's
//! deterministic validation pass; `signals` is the cooperative
//! stop-request listener;
//! `run` is `Run` and the per-document/per-chunk pipeline that drives
//! it; `documents` discovers and chunks input files; `chat_client` is
//! the `/chat/completions` client; `diagnostics` is the JSONL sidecar;
//! `structured_output` resolves and probes ADR 0001 §6's response
//! format ladder; `chunking` is the chunk/piece/round extraction loop
//! with its corrective-turn machinery; `prompt` builds the system/user
//! messages; `parse` turns one model answer into a `ModelOutput`;
//! `aggregate` validates and merges chunk outputs into an
//! `Extraction`; `render` writes the JSONL batch text; `manifest`
//! tracks what each batch file was computed from; `checkpoint` resumes
//! an interrupted document without re-asking the model. This hub keeps
//! the `run` dispatcher, `USAGE`, and the crate-consumed constants
//! (`PROMPT_VERSION`, `CHUNK_BYTES`, `DEFAULT_TIMEOUT_SECS`,
//! `DEFAULT_MAX_ATTEMPTS`, `MAX_EXTRACT_ATTEMPTS`) — `benchmark.rs` and
//! `communities.rs` consume those plus `StructuredOutputMode`,
//! `StopSignal`, `block_stop_signals_on_this_thread`, `chunk_plan`,
//! `read_document`, `expand_documents`, `json_schema_response_format`,
//! `ChatClient`, and `RequestOptions` via `crate::extract::`,
//! re-exported here unchanged.

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Deserialize;

use crate::api::{
    MAX_ASSOCIATION_WEIGHT, MAX_CONTEXT_NAME_BYTES, MAX_DESCRIPTION_BYTES, MAX_NAME_BYTES,
};
use crate::ingest::MAX_PASSAGE_BYTES;
use crate::sha256::sha256_hex;

#[path = "extract/aggregate.rs"]
mod aggregate;
#[path = "extract/args.rs"]
mod args;
#[path = "extract/attempts.rs"]
mod attempts;
#[path = "extract/candidates.rs"]
mod candidates;
#[path = "extract/chat_client.rs"]
mod chat_client;
#[path = "extract/checkpoint.rs"]
mod checkpoint;
#[path = "extract/chunking.rs"]
mod chunking;
#[path = "extract/completions.rs"]
mod completions;
#[path = "extract/coverage.rs"]
mod coverage;
#[path = "extract/diagnostics.rs"]
mod diagnostics;
#[path = "extract/documents.rs"]
mod documents;
#[path = "extract/manifest.rs"]
mod manifest;
#[path = "extract/mechanical.rs"]
mod mechanical;
#[path = "extract/parse.rs"]
mod parse;
#[path = "extract/prompt.rs"]
mod prompt;
#[path = "extract/render.rs"]
mod render;
#[path = "extract/replay.rs"]
mod replay;
#[path = "extract/run.rs"]
mod run;
#[path = "extract/signals.rs"]
mod signals;
#[path = "extract/structured_output.rs"]
mod structured_output;
#[path = "extract/tests.rs"]
#[cfg(test)]
mod tests;
#[path = "extract/trace.rs"]
mod trace;
#[path = "extract/vocabulary.rs"]
mod vocabulary;

use args::{
    Args, CorrectionPolicy, DEFAULT_ESCALATION_FACTOR, LadderConfig, Outcome, ReplayMode, Rung,
    escalation_manifest_value,
};
use diagnostics::DiagnosticsSink;
use manifest::{ComputationInputs, Manifest};
use run::Run;
use structured_output::resolve_rung;

pub(crate) use args::StructuredOutputMode;
pub(crate) use chat_client::{AttemptRef, ChatClient, RequestOptions};
pub(crate) use documents::{ChunkDescriptor, chunk_plan, expand_documents, read_document};
use documents::{chunk_bytes_manifest_value, chunk_plan_with_cap};
pub(crate) use signals::{StopSignal, block_stop_signals_on_this_thread};
use structured_output::json_object_response_format;
pub(crate) use structured_output::json_schema_response_format;
pub(crate) use vocabulary::vocabulary_digest;

// Cross-submodule wiring: each of these is private to the one
// submodule that defines it, but at least one *other* submodule
// names it directly (a sibling's `use super::*;` only sees what the
// hub itself brings into scope) — the same reason ingest.rs's hub
// centralizes this instead of having every submodule import from
// every sibling it needs.
use aggregate::{Extraction, ItemKey, association_name_sets, combined_cross_output_issues, merge};
use attempts::{
    AttemptLog, MoveRecord, Observers, ReplayRecord, ReplaySummaryRecord, SettingsRecord,
    attempts_file_name, attempts_log_enabled,
};
use candidates::{candidate_terms, candidates_block, candidates_manifest_value};
#[cfg(test)]
use chat_client::RETRY_ATTEMPTS;
use chat_client::{
    ChatCompletion, ChatError, ChatFailure, TokenUsage, classify_io_error, mint_run_id,
};
use checkpoint::{CheckpointFingerprint, CheckpointStore, CheckpointUnit};
use chunking::{
    AnswerFault, ChunkOutput, MIN_SPLIT_CAP, corrective_assistant_turn,
    corrective_validation_message, evaluate_answer, extract_chunk_or_ladder,
    indicates_length_limit, indicates_refusal,
};
use completions::Completions;
use coverage::{CoverageGap, coverage_gaps};
use diagnostics::{DiagnosticsAttempt, removed_item_texts};
use manifest::{CHECKPOINT_DIR_NAME, batch_file_name, checkpoint_file_name};
use mechanical::{
    ClaimedNames, Removal, mechanical_interpret, name_occurs, normalize_for_occurrence,
    prune_claimed_aliases, prune_uncorrected_aliases, prune_unresolvable_aliases,
};
use parse::{
    ItemRules, ModelAlias, ModelAssociation, ModelOutput, candidate_json, describe_value,
    empty_answer_diagnosis, get_present, interpret_alias_item, interpret_association_item,
    interpret_model_output, interpret_questions, is_empty_answer, model_output_json_schema,
    quote_for_issue, strip_fences,
};
#[cfg(test)]
use prompt::VOCABULARY_CAP;
use prompt::{
    ranked_vocabulary, schema_constrained_relations, schema_type_names, system_prompt,
    user_message, user_message_document,
};
use render::{chunk, floor_char_boundary, render_batch, split_labeled_piece, split_oversized};
use replay::{
    MissDiagnostic, RecordedSettings, ReplayIndex, ReplayLookup, SystemPinDecision,
    settings_differences,
};
use run::labeled_document;
use structured_output::{jittered_backoff, parse_retry_after, read_capped_chat_body, snippet};
use trace::{
    PieceOrigin, SteeringSchema, TRACE_DIR_NAME, TraceSteering, VocabularyEntry, render_trace,
    write_trace,
};
use vocabulary::{ContextVocabulary, context_names_block, load_vocabulary};

// Test-only cross-submodule access: production code never names these
// at the hub level (each is private to the one submodule that both
// defines and calls it), but the unified test module (`tests.rs`,
// `use super::*;`) exercises them across the split the same way the
// single pre-split file's inline tests did.
#[cfg(test)]
use aggregate::{cross_output_issues, schema_output_issues};
#[cfg(test)]
use args::parse_date;
#[cfg(test)]
use attempts::{attempts_log_enabled_from, first_failure};
#[cfg(test)]
use candidates::{CANDIDATE_CAP, CANDIDATE_MAX_BYTES};
#[cfg(test)]
use chat_client::build_chat_body;
#[cfg(test)]
use chunking::non_object_elements;
#[cfg(test)]
use chunking::{
    AttemptOutcome, MAX_LISTED_ISSUES, PieceContext, RoundOutcome, classify_attempt,
    corrective_message, demotion_reason, extract_piece,
};
#[cfg(test)]
use coverage::GAP_QUOTE_MAX_BYTES;
#[cfg(test)]
use diagnostics::{AttemptRecord, ChunkRecord, DocumentRecord, ProviderMetadataRecord};
#[cfg(test)]
use mechanical::alias_issue_index;
#[cfg(test)]
use parse::{ModelQuestion, parse_model_output};
#[cfg(test)]
use prompt::schema_block;
#[cfg(test)]
use run::with_resume_hint;
#[cfg(test)]
use structured_output::{
    ProbeVerdict, RETRY_MAX_BACKOFF, conforms_to_model_output_shape, probe_structured_output,
    random_duration_up_to,
};
#[cfg(test)]
use trace::paragraph_range;

const USAGE: &str = "\
usage: taguru extract [--dry-run] [--force] [--no-passage] [--questions N]
                      [--fact-budget N] [--config FILE] [--parallel N]
                      [--structured-output MODE] [--max-output-tokens N]
                      [--lossy] [--candidates] [--vocabulary PATH] [--coverage]
                      [--diagnostics-out FILE] [--schema FILE]
                      [--replay MODE] [--replay-from DIR]
                      [--source-id ID] [--date WHEN] [--tag TAG]...
                      --context NAME [--description TEXT] --out DIR FILE|DIR...

Reads documents (.md/.txt; a directory expands to its files, sorted by
name) and writes one batch file per document into --out, ready for
`taguru import` or POST /import. The model is any OpenAI-compatible
chat endpoint:

  TAGURU_EXTRACT_URL      /chat/completions endpoint (required)
  TAGURU_EXTRACT_MODEL    model name (required)
  TAGURU_EXTRACT_API_KEY  bearer credential (optional)
  TAGURU_EXTRACT_TIMEOUT_SECS  per-completion budget; 0 = none (300)
  TAGURU_EXTRACT_PARALLEL  concurrent chunk completions per document (1)
  TAGURU_EXTRACT_FACT_BUDGET  default for --fact-budget (0, off)
  TAGURU_EXTRACT_MAX_ATTEMPTS  total attempts at valid JSON per chunk, 1-10 (2)
  TAGURU_EXTRACT_CORRECTIVE_CONTEXT_BYTES  cap a corrective turn's replay of
                      the model's own prior bad answer to this many bytes;
                      0 omits it entirely (unset: replay it in full)
  TAGURU_EXTRACT_STRUCTURED_OUTPUT  default for --structured-output (off)
  TAGURU_EXTRACT_MAX_OUTPUT_TOKENS  default for --max-output-tokens (unset)
  TAGURU_EXTRACT_ESCALATION_FACTOR  cap of the one escalated resend after an
                      answer ends at --max-output-tokens, as a multiple of
                      that budget; 0 = uncapped (2)
  TAGURU_EXTRACT_CHUNK_BYTES  default for --chunk-bytes (24576)
  TAGURU_EXTRACT_LOSSY  default for --lossy (0/false)
  TAGURU_EXTRACT_CANDIDATES  default for --candidates (0/false)
  TAGURU_EXTRACT_VOCABULARY  default for --vocabulary (unset, off)
  TAGURU_EXTRACT_COVERAGE  default for --coverage (0/false)
  TAGURU_EXTRACT_DIAGNOSTICS  default for --diagnostics-out (unset, off)
  TAGURU_EXTRACT_TRACE_ATTEMPTS  `off` disables the per-document attempts log
                      under OUT/.extract-trace/ (every completion's full
                      prompt and answer; on by default — ADR 0025)
  TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES  attach the model's raw answer text to
                      each diagnostics record, capped to this many bytes;
                      unset or 0 = never attach it (metadata only)
  TAGURU_EXTRACT_SCHEMA  default for --schema (unset, off)
  TAGURU_EXTRACT_REPLAY  default for --replay (off)
  TAGURU_EXTRACT_REPLAY_FROM  default for --replay-from (OUT/.extract-trace)

  --dry-run           list what would extract or skip; call nothing
  --force             re-extract documents the manifest says are unchanged
  --no-passage        omit the document text from the batch (facts only)
  --questions N       doc2query: also propose up to N search questions per
                      paragraph (embedded beside it by servers running
                      TAGURU_EMBED_PASSAGES); rides the same model calls
  --fact-budget N     ask the model to keep each chunk's answer to at most N
                      associations (0, off); a soft instruction, never
                      enforced after the fact
  --structured-output MODE  constrain the answer's shape on the wire:
                      'auto' probes the endpoint once at startup and keeps
                      the strongest rung it verifies (json_schema
                      constrained decoding, then json_object mode, then
                      prompted JSON); 'json-schema'/'json-object' pin a
                      rung without probing; 'off' (default) sends today's
                      plain request
  --max-output-tokens N  explicit output budget per completion, sent as
                      max_tokens (default: none sent). An answer cut off
                      at the budget escalates once without it, then splits
                      the chunk — never re-asked under the limit it just
                      hit
  --chunk-bytes N     document bytes per model call (default 24576, at least
                      512); chunks split at paragraph boundaries — lower it
                      for a slow provider or output-dense documents (statutes,
                      minutes); overrides TAGURU_EXTRACT_CHUNK_BYTES
  --config F          read KEY=VALUE environment from F (same dialect as serve)
  --parallel N        chunk completions to run concurrently within one
                      document (1, sequential); documents themselves stay
                      sequential — vocabulary accumulates as they land
  --lossy             restore the pre-#199 behavior: a business-rule-invalid
                      item (bad weight, dangling alias, out-of-range
                      question, …) is dropped and counted instead of
                      triggering a corrective turn or failing the source;
                      the report always marks a lossy run's drops as such.
                      Default (off): an invalid item earns one targeted
                      corrective turn; if it is still invalid afterward,
                      the source fails and nothing is written.
  --candidates        offer the document's own names (kanji/katakana
                      compounds, ASCII identifiers — segmented
                      deterministically, no dictionary) to the model as
                      preferred subject/object spellings. Non-restrictive:
                      names outside the list stay allowed. Off by default;
                      toggling it re-extracts (a computation input)
  --vocabulary PATH   steer spellings toward a target context's existing
                      vocabulary: PATH is an exported batch stream (or a
                      directory of them, e.g. taguru export --out DIR);
                      its concept names and relation labels are offered
                      to the model as preferred spellings, and a
                      context spelling never fails the occurrence check.
                      Off by default; changing the names re-extracts
  --coverage          report every sentence that holds two or more of the
                      document's own names (the --candidates segmentation)
                      yet is covered by no extracted association — one
                      stderr line per sentence, a count on the report
                      line. Report-only: the batch is never changed, no
                      extra model call is made, and a manifest-skipped
                      document is judged from its already-written batch.
                      Off by default
  --diagnostics-out FILE  write a JSONL sidecar of tagged records (`kind`):
                      one \"chunk\" record per chunk with its provenance
                      (source, chunk_index/total, hash, paragraph range);
                      one \"attempt\" record per LLM attempt (source, chunk,
                      attempt, ADR 0001 §7 state, finish_reason, token
                      usage, latency, parse/validation issues); one
                      \"document\" record per document written (association/
                      alias/duplicate/dropped/uncovered counts) — metadata only;
                      TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES opts an \"attempt\"
                      record into a byte-capped raw answer. The first record
                      is \"run\" (run_id), and every \"attempt\" carries run_id/
                      attempt_seq/piece_id — the keys the per-document trace
                      under OUT/.extract-trace/ joins on (ADR 0023). Truncated
                      fresh at open: FILE describes this run, not a log
                      appended across runs. Default (unset): no sidecar,
                      stdout/stderr unchanged. Ignored under --dry-run, which
                      calls nothing to record.
  --source-id ID      write ID as the batch header's source instead of the
                      document path — the promotion runbook's
                      session:{agent}:{id} convention (docs/promotion.html).
                      With several documents, each gets ID/{file stem}; two
                      documents landing on one source id is an error (import
                      retracts-then-applies per source id). Changing it
                      rewrites the batch but reuses cached chunk answers
  --date WHEN         the session's own date, written on the batch's passage
                      line (the assertion time windowed reads and the
                      staleness audit run on): YYYY-MM-DD (UTC midnight) or
                      positive epoch seconds. Needs the passage
  --tag TAG           tag the batch's source (repeatable, deduplicated) —
                      written on the passage line; how a later session finds
                      its trail via passage search's tags filter. Needs the
                      passage
  --context NAME      the context every batch file targets
  --description TEXT  add a create block (used only if the context is absent)
  --schema FILE       the target context's schema document (same shape as
                      {stem}.schema.json / GET /contexts/{name}/schema):
                      folds allowed entity types and constrained relations
                      into the system prompt and self-validates each answer
                      against it, same as the server would (ADR 0009 §11).
                      Off by default — extract has no server to fetch this
                      from, so it must be handed the document explicitly. A
                      file that fails to parse or validate is a startup
                      error, not a silent skip.
  --replay MODE       satisfy completions from a prior run's attempts log
                      instead of a live call (ADR 0031): 'auto' falls
                      through to a live call when nothing matches; 'strict'
                      fails the document instead, for a run with no model
                      endpoint at all (TAGURU_EXTRACT_URL becomes optional,
                      TAGURU_EXTRACT_MODEL stays required). 'off' (default)
                      never consults the log. A request is matched by its
                      exact conversation content, never by piece or attempt
                      number, so a genuinely changed request always falls
                      through on its own. Bypasses the manifest skip and the
                      checkpoint store; never itself a computation input
  --replay-from DIR   where --replay reads attempts logs from (default:
                      --out/.extract-trace, the directory a run already
                      writes its own to)

Contract and discipline: docs/extract.html.
";

/// Stamped into every manifest entry; bump when the system prompt
/// changes so already-extracted documents re-extract under the new
/// discipline.
///
/// 3 (ADR 0009 §11.1): `system_prompt` gained the schema block — a
/// document extracted under 2's schema-free wording must never be
/// silently reused now that a schema can shape the prompt. A schema's
/// own *content* changing without a version bump is instead covered by
/// `schema_digest` (`CheckpointFingerprint`/`ManifestEntry`), the same
/// division `--fact-budget` (a computation input) and `PROMPT_VERSION`
/// (the prompt's wording) already draw.
///
/// `pub(crate)` so `benchmark`'s manifest can record the same prompt
/// version a cell actually ran under (ADR 0003 §9.1) without
/// re-declaring it.
pub(crate) const PROMPT_VERSION: u32 = 3;

/// Document bytes per model call. Chunks split at paragraph
/// boundaries; facts spanning a boundary can be missed, so the cap
/// leans large.
///
/// `pub(crate)` so `benchmark`'s manifest can record the cap a cell
/// actually ran under (ADR 0003 §9.1) without re-declaring it.
pub(crate) const CHUNK_BYTES: usize = 24 * 1024;

/// One chat completion's default budget. Local models can be slower
/// than any cloud default assumes — thinking-mode models
/// pathologically so — hence the knob (TAGURU_EXTRACT_TIMEOUT_SECS,
/// 0 = no limit).
///
/// `pub(crate)` so `benchmark`'s `models.json` can fold this in as a
/// per-model default (ADR 0003 §8) without re-declaring it.
pub(crate) const DEFAULT_TIMEOUT_SECS: usize = 300;

/// Total attempts (1 initial + corrections) at getting the model to
/// answer with the JSON object [`extract_chunk`] asked for — NOT
/// [`RETRY_ATTEMPTS`], which is the transport layer below it (429/5xx/
/// transport error on one HTTP call). This is "the model answered with
/// something other than the JSON object," resolved from
/// TAGURU_EXTRACT_MAX_ATTEMPTS. Today's fixed 0..2 loop is this value
/// at its default.
///
/// `pub(crate)` so `benchmark`'s `--max-attempts` flag shares
/// `extract`'s own default instead of a second constant that could
/// drift from it.
pub(crate) const DEFAULT_MAX_ATTEMPTS: usize = 2;

/// Hard ceiling on TAGURU_EXTRACT_MAX_ATTEMPTS: a misconfigured value
/// must not be able to turn one stubborn chunk into an unbounded
/// number of model calls.
///
/// `pub(crate)` so `benchmark`'s `--max-attempts` flag enforces the
/// same ceiling `extract` itself does.
pub(crate) const MAX_EXTRACT_ATTEMPTS: usize = 10;

const MANIFEST_NAME: &str = ".extract-manifest.json";

pub fn run(args: &[String]) -> i32 {
    // Issue #213: must happen before anything that could block on I/O
    // (including argument parsing's own error paths, which are cheap
    // but not worth special-casing) — see block_stop_signals_on_this_
    // thread's doc comment for why.
    block_stop_signals_on_this_thread();
    let args = match Args::parse(args) {
        Ok(args) => args,
        Err(code) => return code,
    };
    // SAFETY (same contract as serve and import): applied while the
    // process is still single-threaded — extract never starts a
    // runtime at all.
    if let Some(path) = &args.config {
        crate::config::load_config(path);
    }

    let files = match expand_documents(&args.paths) {
        Ok(files) => files,
        Err(message) => return crate::config::subcommand_usage_error("extract", &message),
    };

    // Flag-over-env, same validation strength as --structured-output
    // below (ADR 0031 §3.3's vocabulary is closed too — an unknown
    // value is a hard usage error, never a silent "off").
    let replay_mode = match args.replay {
        Some(mode) => mode,
        None => match std::env::var("TAGURU_EXTRACT_REPLAY") {
            Ok(value) => match ReplayMode::parse(&value) {
                Some(mode) => mode,
                None => {
                    return crate::config::subcommand_usage_error(
                        "extract",
                        "TAGURU_EXTRACT_REPLAY takes auto, strict, or off",
                    );
                }
            },
            Err(_) => ReplayMode::Off,
        },
    };
    let replaying = !matches!(replay_mode, ReplayMode::Off);

    // The provider is demanded up front even when every document ends
    // up skipped: a run whose environment cannot extract should say so
    // before it reports success. --dry-run alone calls nothing and
    // needs nothing. `--replay strict` (ADR 0031 §3.7/§3.8) is the one
    // other case that can run with no endpoint at all — the model name
    // is still required (it is a manifest computation input, ADR 0031
    // §3.7), only the URL is optional.
    let client = if args.dry_run {
        None
    } else {
        if std::env::var("TAGURU_EXTRACT_MODEL").is_err() {
            eprintln!("taguru: extract: TAGURU_EXTRACT_MODEL is not set");
            return 2;
        }
        match ChatClient::from_env() {
            Ok(client) => Some(client),
            Err(message) => {
                // TAGURU_EXTRACT_MODEL is already confirmed present
                // above, so a failure here can only be a missing
                // TAGURU_EXTRACT_URL — the one gap --replay strict
                // tolerates, running on replay alone.
                if replay_mode == ReplayMode::Strict {
                    None
                } else {
                    eprintln!("taguru: extract: {message}");
                    return 2;
                }
            }
        }
    };
    let model_name = match &client {
        Some(client) => client.model.clone(),
        None => std::env::var("TAGURU_EXTRACT_MODEL").unwrap_or_default(),
    };

    if !args.dry_run
        && let Err(error) = fs::create_dir_all(&args.out)
    {
        eprintln!("taguru: extract: creating {}: {error}", args.out.display());
        return 1;
    }
    let manifest_path = args.out.join(MANIFEST_NAME);
    // Validated with the same strength as --parallel itself: extract
    // never initializes a tracing subscriber (it exits before serve()'s
    // init_telemetry(), and unlike compact it has no init_logging()
    // fallback either), so a silently-ignored bad value would have no
    // way to reach the user.
    let parallel = match args.parallel {
        Some(n) => n,
        None => match std::env::var("TAGURU_EXTRACT_PARALLEL") {
            Ok(value) => match value.parse::<usize>() {
                Ok(n) if n >= 1 => n,
                _ => {
                    return crate::config::subcommand_usage_error(
                        "extract",
                        "TAGURU_EXTRACT_PARALLEL needs an integer of at least 1",
                    );
                }
            },
            Err(_) => 1,
        },
    };
    // Same validation strength and the same reasoning as --parallel
    // above: a silently-ignored bad value would have no way to reach
    // the user.
    let fact_budget = match args.fact_budget {
        Some(n) => n,
        None => match std::env::var("TAGURU_EXTRACT_FACT_BUDGET") {
            Ok(value) => match value.parse::<usize>() {
                Ok(n) if n >= 1 => n,
                _ => {
                    return crate::config::subcommand_usage_error(
                        "extract",
                        "TAGURU_EXTRACT_FACT_BUDGET needs an integer of at least 1",
                    );
                }
            },
            Err(_) => 0,
        },
    };
    // Same "hard usage error, not a silent warning" reasoning as
    // --parallel/--fact-budget above.
    let max_attempts = match std::env::var("TAGURU_EXTRACT_MAX_ATTEMPTS") {
        Ok(value) => match value.parse::<usize>() {
            Ok(n) if (1..=MAX_EXTRACT_ATTEMPTS).contains(&n) => n,
            _ => {
                return crate::config::subcommand_usage_error(
                    "extract",
                    &format!(
                        "TAGURU_EXTRACT_MAX_ATTEMPTS needs an integer between 1 and \
                         {MAX_EXTRACT_ATTEMPTS}"
                    ),
                );
            }
        },
        Err(_) => DEFAULT_MAX_ATTEMPTS,
    };
    // Unlike the others, 0 is a meaningful value here (omit the prior
    // bad answer entirely) rather than the sentinel for "unset" — so
    // this resolves to an Option directly instead of routing through a
    // sentinel-then-default step.
    let corrective_context_cap = match std::env::var("TAGURU_EXTRACT_CORRECTIVE_CONTEXT_BYTES") {
        // "" reads as unset, matching the path-valued keys
        // (VOCABULARY/DIAGNOSTICS/SCHEMA) — it is how `benchmark`'s
        // scrub-then-pin block spells "explicitly the default"
        // (issue #734, ADR 0003 §5).
        Ok(value) if value.is_empty() => None,
        Ok(value) => match value.parse::<usize>() {
            Ok(n) => Some(n),
            Err(_) => {
                return crate::config::subcommand_usage_error(
                    "extract",
                    "TAGURU_EXTRACT_CORRECTIVE_CONTEXT_BYTES needs an integer",
                );
            }
        },
        Err(_) => None,
    };
    // Same flag-over-env resolution as --fact-budget above. The mode
    // vocabulary is closed, so an unknown value is a hard usage error,
    // never a silent "off".
    let structured_output = match args.structured_output {
        Some(mode) => mode,
        None => match std::env::var("TAGURU_EXTRACT_STRUCTURED_OUTPUT") {
            Ok(value) => match StructuredOutputMode::parse(&value) {
                Some(mode) => mode,
                None => {
                    return crate::config::subcommand_usage_error(
                        "extract",
                        "TAGURU_EXTRACT_STRUCTURED_OUTPUT takes auto, json-schema, \
                         json-object, or off",
                    );
                }
            },
            Err(_) => StructuredOutputMode::Off,
        },
    };
    // ADR 0031 §3.7: the ADR 0021 probe requires a live call by
    // construction, and replay is never something that can stand in
    // for it — the one usage error `--replay strict` with no
    // TAGURU_EXTRACT_URL can reach that `--dry-run` (client is also
    // `None` there) does not, since a dry run probes nothing at all.
    if client.is_none() && !args.dry_run && matches!(structured_output, StructuredOutputMode::Auto)
    {
        return crate::config::subcommand_usage_error(
            "extract",
            "--structured-output auto needs a live model endpoint to probe \
             (TAGURU_EXTRACT_URL) — --replay strict alone cannot resolve it; pin a rung \
             explicitly instead (--structured-output json-schema/json-object), reading the \
             recorded attempt's own `rung` field to see which one the original run settled on",
        );
    }
    let max_output_tokens = match args.max_output_tokens {
        Some(n) => Some(n),
        None => match std::env::var("TAGURU_EXTRACT_MAX_OUTPUT_TOKENS") {
            Ok(value) => match value.parse::<usize>() {
                Ok(n) if n >= 1 => Some(n),
                _ => {
                    return crate::config::subcommand_usage_error(
                        "extract",
                        "TAGURU_EXTRACT_MAX_OUTPUT_TOKENS needs an integer of at least 1",
                    );
                }
            },
            Err(_) => None,
        },
    };
    // ADR 0020: the chunk cap. Same floor as the split rung's — below
    // it a chunk could not split at all.
    let chunk_bytes = match args.chunk_bytes {
        Some(n) => n,
        None => match std::env::var("TAGURU_EXTRACT_CHUNK_BYTES") {
            Ok(value) => match value.parse::<usize>() {
                Ok(n) if n >= MIN_SPLIT_CAP => n,
                _ => {
                    return crate::config::subcommand_usage_error(
                        "extract",
                        "TAGURU_EXTRACT_CHUNK_BYTES needs an integer of at least 512",
                    );
                }
            },
            Err(_) => CHUNK_BYTES,
        },
    };
    // ADR 0019: the escalation rung's cap as a multiple of the budget.
    // Read unconditionally (a bad value is a usage error whether or not
    // a budget is set, like every other TAGURU_EXTRACT_* knob), applied
    // only when a budget engages the ladder.
    let escalation_factor = match std::env::var("TAGURU_EXTRACT_ESCALATION_FACTOR") {
        Ok(value) => match value.parse::<usize>() {
            Ok(n) => n,
            Err(_) => {
                return crate::config::subcommand_usage_error(
                    "extract",
                    "TAGURU_EXTRACT_ESCALATION_FACTOR needs an integer of at least 0 \
                     (0 = uncapped escalation)",
                );
            }
        },
        Err(_) => DEFAULT_ESCALATION_FACTOR,
    };
    // ADR 0001 §4/§6: the structured-output rung is resolved once per
    // run — probed when asked to, never assumed, never re-derived per
    // chunk. Any engaged control (a mechanism, or an output budget)
    // switches the run from the legacy corrective loop onto the §7
    // ladder; with neither, requests and retries stay byte-for-byte
    // today's. --dry-run calls nothing, so it also probes nothing.
    let ladder = match (&client, structured_output, max_output_tokens) {
        (_, StructuredOutputMode::Off, None) => None,
        (None, _, _) => None,
        // ADR 0021: only an `auto` resolution can be demoted later —
        // a pinned rung is the operator's choice.
        (Some(client), mode, budget) => Some(LadderConfig::new(
            resolve_rung(client, mode),
            matches!(mode, StructuredOutputMode::Auto),
            budget,
            escalation_factor,
        )),
    };
    // Unlike the old `client.map(...)`, dry-run is the only reason
    // `completions` is ever `None` — under `--replay strict` with no
    // `TAGURU_EXTRACT_URL` (ADR 0031 §3.7/§3.8), `client` itself is
    // `None` but a non-dry run still needs a `Completions` to replay
    // through.
    let completions = (!args.dry_run).then(|| Completions::new(client));
    // Same "hard usage error, not a silent warning" reasoning as
    // --parallel/--fact-budget above (extract never initializes a
    // tracing subscriber, so env::env_bool's warn! would go nowhere).
    let lossy = match args.lossy {
        Some(value) => value,
        None => match std::env::var("TAGURU_EXTRACT_LOSSY") {
            Ok(value) if value == "1" || value.eq_ignore_ascii_case("true") => true,
            Ok(value) if value == "0" || value.eq_ignore_ascii_case("false") => false,
            Ok(_) => {
                return crate::config::subcommand_usage_error(
                    "extract",
                    "TAGURU_EXTRACT_LOSSY takes 1/true or 0/false",
                );
            }
            Err(_) => false,
        },
    };
    // Same resolution and validation strength as --lossy above.
    let candidates_on = match args.candidates {
        Some(value) => value,
        None => match std::env::var("TAGURU_EXTRACT_CANDIDATES") {
            Ok(value) if value == "1" || value.eq_ignore_ascii_case("true") => true,
            Ok(value) if value == "0" || value.eq_ignore_ascii_case("false") => false,
            Ok(_) => {
                return crate::config::subcommand_usage_error(
                    "extract",
                    "TAGURU_EXTRACT_CANDIDATES takes 1/true or 0/false",
                );
            }
            Err(_) => false,
        },
    };
    // Same resolution and validation strength as --candidates above.
    // ADR 0016 (#496 S4): report-only, so unlike --candidates this is
    // NOT a computation input — no fingerprint carries it.
    let coverage_on = match args.coverage {
        Some(value) => value,
        None => match std::env::var("TAGURU_EXTRACT_COVERAGE") {
            Ok(value) if value == "1" || value.eq_ignore_ascii_case("true") => true,
            Ok(value) if value == "0" || value.eq_ignore_ascii_case("false") => false,
            Ok(_) => {
                return crate::config::subcommand_usage_error(
                    "extract",
                    "TAGURU_EXTRACT_COVERAGE takes 1/true or 0/false",
                );
            }
            Err(_) => false,
        },
    };
    // Flag-over-env, same pattern as --schema below. ADR 0015: the
    // named file/directory must load and yield names, or the run stops
    // — silently extracting without the vocabulary the operator asked
    // for would let every new document drift.
    let vocabulary_path = args.vocabulary.or_else(|| {
        std::env::var("TAGURU_EXTRACT_VOCABULARY")
            .ok()
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    });
    let context_vocabulary = match &vocabulary_path {
        Some(path) => match load_vocabulary(path) {
            Ok(vocabulary) => Some(vocabulary),
            Err(message) => {
                eprintln!("taguru: extract: --vocabulary: {message}");
                return 1;
            }
        },
        None => None,
    };
    // Flag-over-env, same pattern as --parallel above. Unlike a parsed
    // knob, any nonempty path is a valid value, so there is no "bad env
    // value" usage error here.
    let diagnostics_path = args.diagnostics_out.or_else(|| {
        std::env::var("TAGURU_EXTRACT_DIAGNOSTICS")
            .ok()
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    });
    // Same "hard usage error, not a silent warning" reasoning as
    // --parallel/--fact-budget above. Validated even when
    // `diagnostics_path` ends up `None`, so a typo'd cap is never a
    // silent no-op just because --diagnostics-out was left off too.
    let diagnostics_raw_bytes = match std::env::var("TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES") {
        // "" reads as unset — same spelling as
        // TAGURU_EXTRACT_CORRECTIVE_CONTEXT_BYTES above (issue #734).
        Ok(value) if value.is_empty() => None,
        Ok(value) => match value.parse::<usize>() {
            Ok(n) => Some(n),
            Err(_) => {
                return crate::config::subcommand_usage_error(
                    "extract",
                    "TAGURU_EXTRACT_DIAGNOSTICS_RAW_BYTES needs an integer",
                );
            }
        },
        Err(_) => None,
    };
    // --dry-run calls nothing, so it opens no sidecar either (the
    // --diagnostics-out usage text says so) — same reasoning as the
    // client-construction skip above.
    let diagnostics = match &diagnostics_path {
        Some(path) if !args.dry_run => {
            // ADR 0023: the sidecar's first record names the run —
            // `Completions` minted the id (one per invocation), and
            // under `!dry_run` it exists.
            let run_id = completions
                .as_ref()
                .map(Completions::run_id)
                .unwrap_or_default();
            match DiagnosticsSink::open(path.clone(), diagnostics_raw_bytes, run_id) {
                Ok(sink) => Some(sink),
                Err(error) => {
                    eprintln!(
                        "taguru: extract: opening diagnostics file {}: {error}",
                        path.display()
                    );
                    return 1;
                }
            }
        }
        _ => None,
    };
    // Flag-over-env, same pattern as --config above. Unlike a parsed
    // knob, any nonempty path is a valid value, so there is no "bad env
    // value" usage error here — the FILE's contents are still validated
    // hard, below.
    let schema_path = args.schema.or_else(|| {
        std::env::var("TAGURU_EXTRACT_SCHEMA")
            .ok()
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    });
    // ADR 0009 §11.1/§13: extract has no server to fetch a schema from
    // (no new credential surface — the same posture as its LLM-only
    // TAGURU_EXTRACT_* environment), so --schema/TAGURU_EXTRACT_SCHEMA is
    // the only way one reaches the prompt. Unlike the "best effort,
    // degrade quietly" postures elsewhere in this file (an unreadable
    // manifest, a skipped document's unreadable batch), a document the
    // operator explicitly named that fails to parse or fails
    // `schema::install`'s own checks is a hard startup error: silently
    // extracting under no schema when one was asked for would let a
    // corpus drift out from under a `strict` context unnoticed.
    let (schema, schema_digest) = match &schema_path {
        Some(path) => {
            let bytes = match fs::read(path) {
                Ok(bytes) => bytes,
                Err(error) => {
                    eprintln!("taguru: extract: reading {}: {error}", path.display());
                    return 1;
                }
            };
            let document: crate::schema::SchemaDocument = match serde_json::from_slice(&bytes) {
                Ok(document) => document,
                Err(error) => {
                    eprintln!("taguru: extract: parsing {}: {error}", path.display());
                    return 1;
                }
            };
            let installed = match crate::schema::install(document) {
                Ok(installed) => installed,
                Err(violation) => {
                    eprintln!("taguru: extract: {}: {violation}", path.display());
                    return 1;
                }
            };
            // Canonical bytes, not the file's own — two files naming the
            // identical document (different key order, whitespace) must
            // fingerprint identically, the same reasoning
            // `document_bytes`'s own doc gives for hashing what will be
            // persisted rather than what was read.
            let canonical = match crate::schema::document_bytes(installed.document()) {
                Ok(bytes) => bytes,
                Err(error) => {
                    eprintln!("taguru: extract: {}: {error}", path.display());
                    return 1;
                }
            };
            (Some(Arc::new(installed)), sha256_hex(&canonical))
        }
        None => (None, String::new()),
    };
    // Flag-over-env, same pattern as --vocabulary above; unlike a
    // parsed knob, any nonempty path is a valid value. Computed
    // unconditionally (Run reads it only under `replaying`, but a
    // typo'd env value should not silently do nothing just because
    // --replay was left off too).
    let replay_from = args
        .replay_from
        .or_else(|| {
            std::env::var("TAGURU_EXTRACT_REPLAY_FROM")
                .ok()
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
        })
        .unwrap_or_else(|| args.out.join(TRACE_DIR_NAME));
    let mut run = Run {
        context: args.context,
        description: args.description,
        force: args.force,
        dry_run: args.dry_run,
        no_passage: args.no_passage,
        questions: args.questions,
        fact_budget,
        correction: CorrectionPolicy {
            max_attempts,
            corrective_context_cap,
        },
        structured_output,
        max_output_tokens,
        escalation_factor,
        chunk_bytes,
        ladder,
        out: args.out,
        completions,
        model_name,
        manifest: Manifest::load(&manifest_path),
        // ADR 0015: the exported context's label spellings seed the
        // run vocabulary, so the existing "relation labels already in
        // use" block carries them from the first document — no new
        // prompt machinery for labels. The export carries no per-label
        // occurrence count (#759), so each seeded label starts at 1 —
        // "established", not "unknown" — and grows from there as this
        // run's own documents land.
        vocabulary: context_vocabulary
            .as_ref()
            .map(|vocabulary| {
                vocabulary
                    .labels
                    .iter()
                    .map(|label| (label.clone(), 1))
                    .collect()
            })
            .unwrap_or_default(),
        // #758: the context's settled spellings are claimed from the
        // first document, the way its labels seed the prompt above.
        claimed_names: context_vocabulary
            .as_ref()
            .map(|vocabulary| ClaimedNames::seeded(&vocabulary.concepts, &vocabulary.labels))
            .unwrap_or_default(),
        source_id: args.source_id,
        date: args.date,
        tags: args.tags,
        multi_document: files.len() > 1,
        claimed_source_ids: BTreeMap::new(),
        claimed: BTreeMap::new(),
        parallel,
        lossy,
        candidates: candidates_on,
        coverage: coverage_on,
        vocabulary_names: context_vocabulary
            .as_ref()
            .map(ContextVocabulary::prompt_names)
            .unwrap_or_default(),
        vocabulary_allowlist: context_vocabulary
            .as_ref()
            .map(|vocabulary| vocabulary.allowlist.clone())
            .unwrap_or_default(),
        vocabulary_digest: context_vocabulary
            .as_ref()
            .map(|vocabulary| vocabulary.digest.clone())
            .unwrap_or_default(),
        diagnostics,
        schema,
        schema_digest,
        stop: StopSignal::install("extract"),
        attempts_log: attempts_log_enabled(),
        replay_mode,
        replaying,
        replay_from,
    };

    let mut written = 0usize;
    let mut planned = 0usize;
    let mut skipped = 0usize;
    let mut failures = 0usize;
    // Issue #179: a stop request is checked between documents (and,
    // inside extract_document, between top-level chunks) — never mid
    // model-call. Whichever document was in flight when it landed keeps
    // every unit already checkpointed; nothing after it is attempted.
    let mut interrupted = false;
    for path in &files {
        if run.stop.check() {
            interrupted = true;
            break;
        }
        let source = path.to_string_lossy().into_owned();
        match run.extract_document(path, &source) {
            Ok(Outcome::Written) => {
                written += 1;
                // Persisted per document, not just once after the loop: a
                // run this size is LLM-bound (seconds per document), so an
                // interruption (Ctrl+C, a CI timeout's SIGKILL, a panic on
                // a later document) would otherwise strand the manifest
                // behind every batch file it should already credit,
                // making the next run re-extract documents that already
                // succeeded.
                if let Err(error) = run.manifest.save(&manifest_path) {
                    eprintln!(
                        "taguru: extract: {source}: saving the manifest: {error} — \
                         the batch is written; the next run re-extracts it"
                    );
                }
            }
            Ok(Outcome::Unchanged) => skipped += 1,
            Ok(Outcome::Planned) => planned += 1,
            Ok(Outcome::Interrupted) => {
                interrupted = true;
                break;
            }
            Err(message) => {
                eprintln!("taguru: extract: {source}: {message}");
                failures += 1;
            }
        }
    }

    if !run.dry_run
        && let Err(error) = run.manifest.save(&manifest_path)
    {
        eprintln!(
            "taguru: extract: saving the manifest: {error} — the batches are written; \
             the next run re-extracts"
        );
    }
    // `written` and `planned` are mutually exclusive across a whole run
    // (dry_run is one flag for every document), so the line reports
    // whichever one actually applies instead of always printing a
    // count that is guaranteed zero.
    if run.dry_run {
        println!(
            "extract: {planned} planned, {skipped} unchanged, {failures} failed of {} document(s)",
            files.len()
        );
    } else if interrupted {
        println!(
            "extract: {written} written, {skipped} unchanged, {failures} failed of {} \
             document(s) — stopped early, chunk checkpoints saved; rerun to resume",
            files.len()
        );
    } else {
        println!(
            "extract: {written} written, {skipped} unchanged, {failures} failed of {} document(s)",
            files.len()
        );
    }
    if failures > 0 {
        1
    } else if interrupted {
        // 128 + SIGINT(2), the same convention StopSignal's forced
        // second-signal exit uses — a script can tell "stopped safely,
        // rerun to resume" apart from a hard failure (exit 1).
        130
    } else {
        0
    }
}
