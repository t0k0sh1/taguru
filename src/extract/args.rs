//! CLI argument parsing (`Args`) and the run-scoped enums/configs it
//! resolves into: `Outcome`, `CorrectionPolicy`, `StructuredOutputMode`,
//! `LadderConfig`.

use super::*;

/// The flags and paths one invocation settled on. `Err` from
/// [`Args::parse`] is the process exit code — 0 after `--help`, 2 for
/// a usage error (already reported on stderr).
pub(super) struct Args {
    pub(super) dry_run: bool,
    pub(super) force: bool,
    pub(super) no_passage: bool,
    /// doc2query: search questions per paragraph the model is asked
    /// for (0 = off, the default — question generation rides the same
    /// extraction calls but still spends output tokens).
    pub(super) questions: usize,
    /// `None` defers to TAGURU_EXTRACT_FACT_BUDGET, and then to 0 (off,
    /// today's unbounded behavior) — resolved in [`run`], same pattern
    /// as `parallel`. The resolved value is folded into the system
    /// prompt as a soft instruction, never enforced post-hoc by
    /// `merge`: a provider that ignores it just gets everything it
    /// returned.
    pub(super) fact_budget: Option<usize>,
    pub(super) config: Option<PathBuf>,
    /// `None` defers to TAGURU_EXTRACT_PARALLEL, and then to 1 (today's
    /// sequential behavior) — resolved in [`run`], not here, since the
    /// flag must win over the environment variable.
    pub(super) parallel: Option<usize>,
    /// `None` defers to TAGURU_EXTRACT_STRUCTURED_OUTPUT, and then to
    /// `Off` (today's plain request) — resolved in [`run`], same
    /// pattern as `fact_budget`.
    pub(super) structured_output: Option<StructuredOutputMode>,
    /// `None` defers to TAGURU_EXTRACT_MAX_OUTPUT_TOKENS, and then to
    /// sending no output-token parameter at all (today's request) —
    /// resolved in [`run`].
    pub(super) max_output_tokens: Option<usize>,
    /// `None` defers to TAGURU_EXTRACT_LOSSY, and then to `false`
    /// (issue #199's default: an invalid item earns a corrective turn,
    /// never a silent drop) — resolved in [`run`], same pattern as
    /// `structured_output`.
    pub(super) lossy: Option<bool>,
    /// `None` defers to TAGURU_EXTRACT_CANDIDATES, and then to `false`
    /// (ADR 0014's default-off: the prompt stays byte-for-byte pre-S2)
    /// — resolved in [`run`], same pattern as `lossy`.
    pub(super) candidates: Option<bool>,
    /// `None` defers to TAGURU_EXTRACT_VOCABULARY, and then to no
    /// vocabulary at all — resolved in [`run`], same pattern as
    /// `schema`. ADR 0015 (#496 S3).
    pub(super) vocabulary: Option<PathBuf>,
    /// `None` defers to TAGURU_EXTRACT_COVERAGE, and then to `false`
    /// (no coverage report — stdout/stderr byte-for-byte pre-S4) —
    /// resolved in [`run`], same pattern as `candidates`. ADR 0016
    /// (#496 S4).
    pub(super) coverage: Option<bool>,
    /// `None` defers to TAGURU_EXTRACT_DIAGNOSTICS, and then to no
    /// sidecar at all (today's behavior: one stderr line per failed
    /// document, nothing else) — resolved in [`run`], same pattern as
    /// `parallel`. Issue #200.
    pub(super) diagnostics_out: Option<PathBuf>,
    /// `None` defers to TAGURU_EXTRACT_SCHEMA, and then to no schema at
    /// all (today's behavior) — resolved in [`run`], same pattern as
    /// `config`.
    pub(super) schema: Option<PathBuf>,
    pub(super) context: String,
    pub(super) description: Option<String>,
    /// #466 S1 (ADR 0017): the promotion runbook's `session:{agent}:{id}`
    /// source id, replacing the document path in the written batch
    /// header. `None` keeps the path (today's batch, byte for byte).
    pub(super) source_id: Option<String>,
    /// #466 S1: the session's own date (epoch seconds), emitted on the
    /// written batch's passage line — the assertion time every windowed
    /// read and the staleness audit run on. `None` emits no field.
    pub(super) date: Option<u64>,
    /// #466 S1: the session's topic tags, emitted on the written
    /// batch's passage line. Empty emits no field.
    pub(super) tags: Vec<String>,
    pub(super) out: PathBuf,
    pub(super) paths: Vec<String>,
}

impl Args {
    pub(super) fn parse(args: &[String]) -> Result<Self, i32> {
        let mut dry_run = false;
        let mut force = false;
        let mut no_passage = false;
        let mut questions = 0usize;
        let mut fact_budget: Option<usize> = None;
        let mut config: Option<PathBuf> = None;
        let mut parallel: Option<usize> = None;
        let mut structured_output: Option<StructuredOutputMode> = None;
        let mut max_output_tokens: Option<usize> = None;
        let mut lossy: Option<bool> = None;
        let mut candidates: Option<bool> = None;
        let mut vocabulary: Option<PathBuf> = None;
        let mut coverage: Option<bool> = None;
        let mut diagnostics_out: Option<PathBuf> = None;
        let mut schema: Option<PathBuf> = None;
        let mut context: Option<String> = None;
        let mut description: Option<String> = None;
        let mut source_id: Option<String> = None;
        let mut date: Option<u64> = None;
        let mut tags: Vec<String> = Vec::new();
        let mut out: Option<PathBuf> = None;
        let mut paths: Vec<String> = Vec::new();
        let mut rest = args.iter();
        while let Some(arg) = rest.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print!("{USAGE}");
                    return Err(0);
                }
                "--dry-run" => dry_run = true,
                "--force" => force = true,
                "--no-passage" => no_passage = true,
                "--lossy" => lossy = Some(true),
                "--candidates" => candidates = Some(true),
                "--coverage" => coverage = Some(true),
                "--vocabulary" => match rest.next() {
                    Some(path) if vocabulary.is_none() => vocabulary = Some(PathBuf::from(path)),
                    Some(_) => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--vocabulary given twice",
                        ));
                    }
                    None => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--vocabulary needs a file or directory path",
                        ));
                    }
                },
                "--questions" => match rest.next().map(|n| n.parse::<usize>()) {
                    Some(_) if questions > 0 => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--questions given twice",
                        ));
                    }
                    Some(Ok(n)) if (1..=crate::api::MAX_QUESTIONS_PER_PARAGRAPH).contains(&n) => {
                        questions = n;
                    }
                    Some(_) => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            &format!(
                                "--questions takes 1..={} (per paragraph)",
                                crate::api::MAX_QUESTIONS_PER_PARAGRAPH
                            ),
                        ));
                    }
                    None => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--questions needs a count",
                        ));
                    }
                },
                "--fact-budget" => match rest.next().map(|n| n.parse::<usize>()) {
                    Some(_) if fact_budget.is_some() => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--fact-budget given twice",
                        ));
                    }
                    Some(Ok(n)) if n >= 1 => fact_budget = Some(n),
                    _ => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--fact-budget needs an integer of at least 1",
                        ));
                    }
                },
                "--config" => match rest.next() {
                    Some(path) if config.is_none() => config = Some(PathBuf::from(path)),
                    Some(_) => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--config given twice",
                        ));
                    }
                    None => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--config needs a file path",
                        ));
                    }
                },
                "--parallel" => match rest.next().map(|value| value.parse::<usize>()) {
                    Some(_) if parallel.is_some() => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--parallel given twice",
                        ));
                    }
                    Some(Ok(n)) if n >= 1 => parallel = Some(n),
                    _ => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--parallel needs an integer of at least 1",
                        ));
                    }
                },
                "--structured-output" => {
                    if structured_output.is_some() {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--structured-output given twice",
                        ));
                    }
                    match rest
                        .next()
                        .and_then(|mode| StructuredOutputMode::parse(mode))
                    {
                        Some(mode) => structured_output = Some(mode),
                        None => {
                            return Err(crate::config::subcommand_usage_error(
                                "extract",
                                "--structured-output takes auto, json-schema, json-object, or off",
                            ));
                        }
                    }
                }
                "--max-output-tokens" => match rest.next().map(|value| value.parse::<usize>()) {
                    Some(_) if max_output_tokens.is_some() => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--max-output-tokens given twice",
                        ));
                    }
                    Some(Ok(n)) if n >= 1 => max_output_tokens = Some(n),
                    _ => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--max-output-tokens needs an integer of at least 1",
                        ));
                    }
                },
                "--diagnostics-out" => match rest.next() {
                    Some(path) if diagnostics_out.is_none() => {
                        diagnostics_out = Some(PathBuf::from(path));
                    }
                    Some(_) => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--diagnostics-out given twice",
                        ));
                    }
                    None => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--diagnostics-out needs a file path",
                        ));
                    }
                },
                "--schema" => match rest.next() {
                    Some(path) if schema.is_none() => schema = Some(PathBuf::from(path)),
                    Some(_) => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--schema given twice",
                        ));
                    }
                    None => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--schema needs a file path",
                        ));
                    }
                },
                "--context" => match rest.next() {
                    Some(name) if context.is_none() => context = Some(name.clone()),
                    Some(_) => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--context given twice",
                        ));
                    }
                    None => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--context needs a name",
                        ));
                    }
                },
                "--description" => match rest.next() {
                    Some(text) if description.is_none() => description = Some(text.clone()),
                    Some(_) => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--description given twice",
                        ));
                    }
                    None => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--description needs a text",
                        ));
                    }
                },
                "--source-id" => match rest.next() {
                    Some(id) if source_id.is_none() && !id.is_empty() => {
                        source_id = Some(id.clone());
                    }
                    Some(id) if id.is_empty() => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--source-id must not be empty",
                        ));
                    }
                    Some(_) => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--source-id given twice",
                        ));
                    }
                    None => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--source-id needs a source id (session:{agent}:{id})",
                        ));
                    }
                },
                "--date" => match rest.next() {
                    Some(_) if date.is_some() => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--date given twice",
                        ));
                    }
                    Some(when) => match parse_date(when) {
                        Some(seconds) => date = Some(seconds),
                        None => {
                            return Err(crate::config::subcommand_usage_error(
                                "extract",
                                "--date takes YYYY-MM-DD (UTC midnight) or positive epoch seconds",
                            ));
                        }
                    },
                    None => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--date needs a date (YYYY-MM-DD or epoch seconds)",
                        ));
                    }
                },
                "--tag" => match rest.next() {
                    Some(tag) if tag.trim().is_empty() => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--tag must not be empty",
                        ));
                    }
                    Some(tag) if tag.len() > crate::api::MAX_TAG_BYTES => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            &format!(
                                "tag of {} bytes exceeds the {}-byte cap",
                                tag.len(),
                                crate::api::MAX_TAG_BYTES
                            ),
                        ));
                    }
                    Some(tag) => {
                        if !tags.contains(tag) {
                            tags.push(tag.clone());
                        }
                        if tags.len() > crate::api::MAX_TAGS_PER_SOURCE {
                            return Err(crate::config::subcommand_usage_error(
                                "extract",
                                &format!(
                                    "more than {} tags where a source carries at most that many",
                                    crate::api::MAX_TAGS_PER_SOURCE
                                ),
                            ));
                        }
                    }
                    None => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--tag needs a tag",
                        ));
                    }
                },
                "--out" => match rest.next() {
                    Some(dir) if out.is_none() => out = Some(PathBuf::from(dir)),
                    Some(_) => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--out given twice",
                        ));
                    }
                    None => {
                        return Err(crate::config::subcommand_usage_error(
                            "extract",
                            "--out needs a directory",
                        ));
                    }
                },
                other if other.starts_with('-') => {
                    return Err(crate::config::subcommand_usage_error(
                        "extract",
                        &format!("unknown flag '{other}'"),
                    ));
                }
                path => paths.push(path.to_string()),
            }
        }
        let Some(context) = context else {
            return Err(crate::config::subcommand_usage_error(
                "extract",
                "--context NAME is required",
            ));
        };
        let Some(out) = out else {
            return Err(crate::config::subcommand_usage_error(
                "extract",
                "--out DIR is required",
            ));
        };
        if context.len() > MAX_CONTEXT_NAME_BYTES {
            return Err(crate::config::subcommand_usage_error(
                "extract",
                &format!(
                    "context name of {} bytes exceeds the {MAX_CONTEXT_NAME_BYTES}-byte cap",
                    context.len()
                ),
            ));
        }
        if let Some(text) = &description
            && text.len() > MAX_DESCRIPTION_BYTES
        {
            return Err(crate::config::subcommand_usage_error(
                "extract",
                &format!(
                    "description of {} bytes exceeds the {MAX_DESCRIPTION_BYTES}-byte cap",
                    text.len()
                ),
            ));
        }
        if paths.is_empty() {
            eprint!("{USAGE}");
            return Err(2);
        }
        if questions > 0 && no_passage {
            return Err(crate::config::subcommand_usage_error(
                "extract",
                "--questions needs the passage (--no-passage strips the text the \
                 questions would attach to)",
            ));
        }
        // Source metadata rides the batch's passage line (the import
        // wire format has nowhere else to put it — an associations-only
        // source stores no metadata and is invisible to every windowed
        // read, docs/promotion.html's own warning), so stripping the
        // passage while asking for metadata is a contradiction, not a
        // silent drop.
        if (date.is_some() || !tags.is_empty()) && no_passage {
            return Err(crate::config::subcommand_usage_error(
                "extract",
                "--date/--tag ride the passage line (--no-passage strips it, and an \
                 associations-only source stores no metadata)",
            ));
        }
        if let Some(id) = &source_id
            && id.len() > MAX_NAME_BYTES
        {
            return Err(crate::config::subcommand_usage_error(
                "extract",
                &format!(
                    "source id of {} bytes exceeds the {MAX_NAME_BYTES}-byte cap",
                    id.len()
                ),
            ));
        }
        // TAGURU_CONFIG fallback (issue #248 item 2): --config wins,
        // but a deployment file baked in via the environment still
        // applies when it's absent — the same priority serve/health/
        // calibrate/communities/evaluate/restore already give it.
        let config = config.or_else(|| std::env::var("TAGURU_CONFIG").ok().map(PathBuf::from));
        Ok(Self {
            dry_run,
            force,
            no_passage,
            questions,
            fact_budget,
            config,
            parallel,
            structured_output,
            max_output_tokens,
            lossy,
            candidates,
            vocabulary,
            coverage,
            diagnostics_out,
            schema,
            context,
            description,
            source_id,
            date,
            tags,
            out,
            paths,
        })
    }
}

/// `--date`'s two spellings: bare digits are epoch seconds (the wire
/// format's own unit), `YYYY-MM-DD` is that day's UTC midnight —
/// what a session note actually records. Zero is rejected in both
/// spellings so the manifest can keep 0 as its off sentinel, the
/// `max_output_tokens` precedent.
pub(super) fn parse_date(text: &str) -> Option<u64> {
    if !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit()) {
        return text.parse::<u64>().ok().filter(|&seconds| seconds > 0);
    }
    let mut parts = text.split('-');
    let year = parts.next()?.parse::<i64>().ok()?;
    let month = parts.next()?.parse::<u32>().ok()?;
    let day = parts.next()?.parse::<u32>().ok()?;
    // Separate checks rather than one `||` chain: the round-trip below
    // already refuses any out-of-range month/day (it renders as a
    // different date), so a chained condition's operator is
    // mutation-equivalent noise — and the range checks' real job is
    // keeping the civil arithmetic's inputs in-domain (day 0 would
    // underflow `d - 1` there).
    if parts.next().is_some() {
        return None;
    }
    if !(1..=12).contains(&month) {
        return None;
    }
    if !(1..=31).contains(&day) {
        return None;
    }
    let days = crate::clock::days_from_civil(year, month, day);
    // Round-trip through the rendering direction: a lexically plausible
    // but non-existent date (2026-02-30) normalizes to a different day,
    // so refusing the mismatch refuses the typo.
    let seconds = u64::try_from(days.checked_mul(86400)?).ok()?;
    if crate::clock::iso8601_utc(seconds).starts_with(&format!("{year:04}-{month:02}-{day:02}T")) {
        Some(seconds).filter(|&seconds| seconds > 0)
    } else {
        None
    }
}

/// What one document's pipeline concluded; [`run`] only counts these
/// into the summary line.
pub(super) enum Outcome {
    /// A fresh batch file is on disk and recorded in the manifest.
    Written,
    /// The manifest proved the computation inputs unchanged; nothing
    /// was called.
    Unchanged,
    /// `--dry-run` reported what would happen without calling anything.
    Planned,
    /// Issue #179: a cooperative stop request was observed between
    /// chunks or between documents. Whatever units already landed stay
    /// checkpointed on disk; nothing was merged, imported, or recorded
    /// in the manifest — a rerun resumes exactly where this stopped.
    Interrupted,
}

/// How [`extract_chunk`] handles a model answer that isn't the JSON
/// object it asked for. Resolved once per run from
/// TAGURU_EXTRACT_MAX_ATTEMPTS/TAGURU_EXTRACT_CORRECTIVE_CONTEXT_BYTES
/// (docs/extract.html). The all-defaults value (`DEFAULT_MAX_ATTEMPTS`,
/// `None`) reproduces today's fixed "one corrective turn, full replay"
/// behavior byte for byte.
pub(super) struct CorrectionPolicy {
    /// Total attempts (1 initial + corrections), always in
    /// `1..=MAX_EXTRACT_ATTEMPTS`.
    pub(super) max_attempts: usize,
    /// How much of the model's own prior bad answer gets replayed back
    /// to it in the next attempt's corrective turn: `None` replays it
    /// in full (today's behavior), `Some(0)` omits it behind a
    /// placeholder, `Some(n)` truncates it to `n` bytes.
    pub(super) corrective_context_cap: Option<usize>,
}

/// `--structured-output`'s closed vocabulary: which rung of ADR 0001
/// §6's fallback ladder the run may put on the wire. `Off` — the
/// default — sends today's plain request and keeps the legacy
/// corrective loop.
///
/// `pub(crate)` so `benchmark` validates a `models.json` entry's
/// `structured_output` string against the exact same closed vocabulary
/// `extract` itself enforces (ADR 0003 §8), rather than a second,
/// possibly-drifting copy of the match arms.
#[derive(Clone, Copy)]
pub(crate) enum StructuredOutputMode {
    /// Probe the endpoint once at startup and keep the strongest rung
    /// it verifies: json_schema, then json_object, then bare prompted
    /// JSON.
    Auto,
    /// Pin schema-constrained decoding without probing; a backend that
    /// rejects the parameter surfaces its 400 on the first document
    /// rather than being silently downgraded.
    JsonSchema,
    /// Pin JSON mode (syntax forced, shape not) without probing.
    JsonObject,
    Off,
}

impl StructuredOutputMode {
    pub(crate) fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Self::Auto),
            "json-schema" => Some(Self::JsonSchema),
            "json-object" => Some(Self::JsonObject),
            "off" => Some(Self::Off),
            _ => None,
        }
    }

    /// The manifest spelling: the REQUESTED mode, never the probe's
    /// resolution — which rung carried a run depends on the backend,
    /// but the computation input is what the operator asked for. `Off`
    /// is the empty string so entries written before this field
    /// existed keep matching all-defaults runs instead of forcing a
    /// spurious re-extraction of everything.
    pub(super) fn manifest_value(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::JsonSchema => "json-schema",
            Self::JsonObject => "json-object",
            Self::Off => "",
        }
    }
}

/// The §7 ladder's per-run inputs, settled once at startup: the
/// verified (or pinned) `response_format` for every extraction
/// request, and the operator's output budget. Present exactly when
/// some new control is engaged; `None` keeps the legacy loop
/// byte-for-byte.
pub(super) struct LadderConfig {
    pub(super) response_format: Option<serde_json::Value>,
    pub(super) max_output_tokens: Option<usize>,
}
