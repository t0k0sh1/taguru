//! The per-document attempts log (ADR 0025, #788): `--out/.extract-trace/
//! <batch stem>.attempts.jsonl`, every completion's full prompt and full
//! answer, on by default — and `Observers`, the one seam every
//! extraction call site reports an attempt through, fanning out to
//! this log and to the opt-in diagnostics sidecar.

use super::*;

/// `TAGURU_EXTRACT_TRACE_ATTEMPTS`: `off`/`0`/`false`/`no` disables the
/// attempts log; anything else (and unset) keeps the default on.
pub(super) const ATTEMPTS_ENV: &str = "TAGURU_EXTRACT_TRACE_ATTEMPTS";

pub(super) fn attempts_log_enabled() -> bool {
    attempts_log_enabled_from(std::env::var(ATTEMPTS_ENV).ok().as_deref())
}

pub(super) fn attempts_log_enabled_from(value: Option<&str>) -> bool {
    !matches!(
        value
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "off" | "0" | "false" | "no"
    )
}

/// The attempts log's file name for a source: the batch file's own
/// name with `.attempts.jsonl` in place of `.jsonl`, so a `--out`
/// listing pairs them by eye.
pub(super) fn attempts_file_name(batch_file_name: &str) -> String {
    format!(
        "{}.attempts.jsonl",
        batch_file_name
            .strip_suffix(".jsonl")
            .unwrap_or(batch_file_name)
    )
}

/// One document's attempts log. Same mechanics as `DiagnosticsSink`:
/// one `write_all` + `flush` per record so a killed run keeps every
/// completed line, `Mutex`-guarded for `--parallel`, one warning on
/// the first write failure and silence after (a log never fails the
/// document — ADR 0023 §3.6).
pub(super) struct AttemptLog {
    writer: Mutex<std::io::BufWriter<fs::File>>,
    path: PathBuf,
    warned: AtomicBool,
    /// System prompts already written as `kind: "system"` records, by
    /// sha256 — the prompt is fixed per document, so it is written
    /// once and every attempt names it by hash (ADR 0025 §3.3).
    systems: Mutex<HashSet<String>>,
}

impl AttemptLog {
    /// Opens the log — truncating for a document starting fresh,
    /// appending for one resuming from a checkpoint (`resuming`), so
    /// the file spans exactly the runs that built the batch, as the
    /// checkpoint does — and writes the `document` record.
    pub(super) fn open(
        path: PathBuf,
        resuming: bool,
        run_id: &str,
        source: &str,
        document_sha256: &str,
    ) -> std::io::Result<Self> {
        let file = if resuming {
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)?
        } else {
            fs::File::create(&path)?
        };
        let log = Self {
            writer: Mutex::new(std::io::BufWriter::new(file)),
            path,
            warned: AtomicBool::new(false),
            systems: Mutex::new(HashSet::new()),
        };
        log.write_record(&AttemptsDocumentRecord {
            kind: "document",
            run_id,
            source,
            document_sha256,
            resumed: resuming,
        });
        Ok(log)
    }

    /// Records one completion: the conversation as sent (the system
    /// turn by hash, written in full once per distinct prompt), the
    /// answer in full, and the attempt's identity and classification
    /// — the same facts the sidecar's `attempt` record carries, minus
    /// the byte cap.
    pub(super) fn record(&self, attempt: &DiagnosticsAttempt, messages: &[serde_json::Value]) {
        let mut turns = Vec::with_capacity(messages.len());
        for message in messages {
            let role = message["role"].as_str().unwrap_or("");
            let content = message["content"].as_str().unwrap_or("");
            if role == "system" {
                let sha256 = sha256_hex(content.as_bytes());
                let first = {
                    let mut seen = match self.systems.lock() {
                        Ok(seen) => seen,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    seen.insert(sha256.clone())
                };
                if first {
                    self.write_record(&SystemRecord {
                        kind: "system",
                        sha256: &sha256,
                        bytes: content.len(),
                        content,
                    });
                }
                turns.push(Turn {
                    role,
                    content: None,
                    system_sha256: Some(sha256),
                });
            } else {
                turns.push(Turn {
                    role,
                    content: Some(content),
                    system_sha256: None,
                });
            }
        }
        let response = attempt.response;
        self.write_record(&AttemptFullRecord {
            kind: "attempt",
            run_id: &attempt.attempt_ref.run_id,
            attempt_seq: attempt.attempt_ref.attempt_seq,
            corrects: attempt.corrects,
            replayed_from: attempt.replayed_from,
            piece_id: attempt.piece_id,
            source: attempt.source,
            chunk_index: attempt.chunk_index,
            stage: attempt.stage,
            attempt: attempt.attempt,
            max_attempts: attempt.max_attempts,
            state: attempt.state,
            length_limited: attempt.length_limited,
            transport_retries: attempt.transport_retries,
            elapsed_seconds: attempt.elapsed.as_secs_f64(),
            requested_max_tokens: attempt.requested_max_tokens,
            rung: attempt.rung,
            finish_reason: response.and_then(|response| response.finish_reason.as_deref()),
            input_tokens: response.and_then(|r| r.usage.and_then(|u| u.input_tokens)),
            output_tokens: response.and_then(|r| r.usage.and_then(|u| u.output_tokens)),
            messages: turns,
            answer: response.map(|response| response.content.as_str()),
            parse_error: attempt.parse_error,
            validation_issues: attempt.validation_issues,
            removed_items: attempt.removed_items.as_deref(),
        });
    }

    /// Where this document's log lives — the failure line's pointer
    /// (ADR 0037 §3.2).
    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn write_record(&self, record: &impl serde::Serialize) {
        let mut line = match serde_json::to_string(record) {
            Ok(line) => line,
            Err(_) => return,
        };
        line.push('\n');
        let mut writer = match self.writer.lock() {
            Ok(writer) => writer,
            Err(poisoned) => poisoned.into_inner(),
        };
        let wrote = writer
            .write_all(line.as_bytes())
            .and_then(|()| writer.flush());
        if let Err(error) = wrote
            && first_failure(&self.warned)
        {
            eprintln!(
                "taguru: extract: attempts log: writing {}: {error} — further records are \
                 dropped",
                self.path.display()
            );
        }
    }
}

/// The warn-once gate: `true` exactly for the first call on a given
/// flag, so one write failure earns one stderr line and every later
/// dropped record is silent.
pub(super) fn first_failure(warned: &AtomicBool) -> bool {
    !warned.swap(true, Ordering::Relaxed)
}

/// The log's first line per run over this document.
#[derive(serde::Serialize)]
struct AttemptsDocumentRecord<'a> {
    kind: &'static str,
    run_id: &'a str,
    source: &'a str,
    document_sha256: &'a str,
    /// `true` when this run appended to a log an earlier, incomplete
    /// run started (checkpoint resume).
    resumed: bool,
}

/// ADR 0031 §3.2/§3.9: the run's compute-input settings, as a
/// diagnostic — never a gate. Written once per document, right after
/// the `document` record. The same field set `CheckpointFingerprint`
/// checks (minus `sha256`/`context`/`no_passage`/`description`/
/// `escalation_factor`, which name the document or a value that never
/// reaches the model), plus `rung`. A later replay run compares its
/// own settings against this record and names any field that differs
/// — matching itself is still decided by the conversation (ADR 0031
/// §3.2), never by this record; a mismatch here is a hint, not a
/// verdict.
#[derive(serde::Serialize)]
pub(super) struct SettingsRecord<'a> {
    pub(super) kind: &'static str,
    pub(super) prompt_version: u32,
    pub(super) model: &'a str,
    pub(super) questions_n: usize,
    pub(super) fact_budget: usize,
    pub(super) structured_output: &'a str,
    pub(super) max_output_tokens: usize,
    pub(super) chunk_bytes: &'a str,
    /// ADR 0033: `""` = off, the mode name otherwise.
    pub(super) chunk_context: &'a str,
    pub(super) lossy: bool,
    pub(super) schema_digest: &'a str,
    pub(super) candidates: &'a str,
    /// `--redact`'s version (`""` = off) — ADR 0038 §3.5.
    pub(super) redaction: &'a str,
    pub(super) vocabulary_digest: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) rung: Option<&'static str>,
}

/// ADR 0031 §3.4: this document's replay mode and source directory,
/// once per document — right after `settings`, and written only when
/// `--replay` is `auto` or `strict`.
#[derive(serde::Serialize)]
pub(super) struct ReplayRecord<'a> {
    pub(super) kind: &'static str,
    pub(super) mode: &'static str,
    pub(super) replay_from: &'a str,
}

/// ADR 0031 §3.4: this document's replayed/live completion counts —
/// the last record of a replaying document's own attempts, echoing
/// the `replayed N/M completions (K live)` stderr line. `replayed +
/// live` is always the document's total completion count: a
/// completion is exactly one or the other, never both, so a separate
/// "missed" count would only restate `live` (`--replay auto`'s every
/// miss becomes a live call) or never be written at all (`--replay
/// strict`'s first miss fails the document before this record is
/// reached).
#[derive(serde::Serialize)]
pub(super) struct ReplaySummaryRecord {
    pub(super) kind: &'static str,
    pub(super) replayed: u64,
    pub(super) live: u64,
}

/// One distinct system prompt, in full, written the first time an
/// attempt of this document sends it.
#[derive(serde::Serialize)]
struct SystemRecord<'a> {
    kind: &'static str,
    sha256: &'a str,
    bytes: usize,
    content: &'a str,
}

/// One turn of the conversation as sent: the system turn by hash
/// (see `SystemRecord`), every other turn in full.
#[derive(serde::Serialize)]
struct Turn<'a> {
    role: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_sha256: Option<String>,
}

/// One completion, in full. Field names match the sidecar's
/// `AttemptRecord` wherever the concept matches, so the two join on
/// `(run_id, attempt_seq)` and read alike.
#[derive(serde::Serialize)]
struct AttemptFullRecord<'a> {
    kind: &'static str,
    run_id: &'a str,
    attempt_seq: u64,
    /// ADR 0028: the attempt this corrective attempt replays and asks
    /// to fix; absent on base attempts — and on a Stage 2 correction
    /// of a unit reused from a pre-0.9.5 (pre-ADR 0023) checkpoint,
    /// which recorded no attempt to name.
    #[serde(skip_serializing_if = "Option::is_none")]
    corrects: Option<&'a AttemptRef>,
    /// ADR 0031 §3.2 (#823): the original attempt whose recorded
    /// answer this one reused, when it came from `--replay` instead of
    /// a live call — absent on a live completion. `elapsed_seconds`/
    /// `input_tokens`/`output_tokens` here describe the replay itself
    /// (near-zero, no model call made — the original's real numbers
    /// sit on the named attempt), so a cost/time rollup must skip any
    /// record carrying this field to avoid double-counting.
    #[serde(skip_serializing_if = "Option::is_none")]
    replayed_from: Option<&'a AttemptRef>,
    piece_id: &'a str,
    source: &'a str,
    chunk_index: usize,
    stage: &'static str,
    attempt: usize,
    max_attempts: usize,
    state: &'static str,
    length_limited: bool,
    transport_retries: usize,
    elapsed_seconds: f64,
    requested_max_tokens: Option<usize>,
    /// ADR 0031 §3.2: the structured-output rung this completion was
    /// asked under; `null` off the ladder (the legacy path).
    #[serde(skip_serializing_if = "Option::is_none")]
    rung: Option<&'static str>,
    finish_reason: Option<&'a str>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    messages: Vec<Turn<'a>>,
    /// The assistant's final text, in full; `null` for
    /// `timeout`/`transport`, where none arrived.
    answer: Option<&'a str>,
    parse_error: Option<&'a str>,
    validation_issues: Option<&'a [String]>,
    removed_items: Option<&'a [String]>,
}

/// ADR 0029 (#791): one ladder move (ADR 0001 §7 / 0019 / 0020 /
/// 0021), as a record instead of a stderr sentence — `escalate` (the
/// budget raised for one neutral resend), `demote` (the run's
/// structured-output rung lowered; the piece restarts), `split` (the
/// piece divided and each part rerun from the ladder's top). The
/// per-attempt facts (states, timeouts, token costs) stay on the
/// `attempt` records; a move record says what the ladder DID next and
/// why, joinable by `piece_id` and, run-wide, by `run_id`.
#[derive(serde::Serialize)]
pub(super) struct MoveRecord<'a> {
    pub(super) kind: &'static str,
    #[serde(rename = "move")]
    pub(super) action: &'static str,
    pub(super) run_id: &'a str,
    pub(super) piece_id: &'a str,
    pub(super) chunk_index: usize,
    /// Why the ladder moved, in the stderr line's own words.
    pub(super) reason: &'a str,
    /// `escalate` only: the budget of the round that ended `length`,
    /// and the escalated resend's.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) from_max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) to_max_tokens: Option<usize>,
    /// `demote` only: the rung observed failing, and the one the run
    /// continues on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) from_rung: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) to_rung: Option<&'static str>,
    /// `split`: the piece's size, the sub-piece cap, and how many
    /// sub-pieces it divided into. `runaway` (ADR 0035): the piece's
    /// size again, beside `answer_bytes`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) piece_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) split_cap: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) sub_pieces: Option<usize>,
    /// `runaway` only (ADR 0035): the length-limited answer's bytes
    /// that exceeded the ratio against `piece_bytes`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) answer_bytes: Option<usize>,
}

impl<'a> MoveRecord<'a> {
    pub(super) fn blank(
        action: &'static str,
        run_id: &'a str,
        piece_id: &'a str,
        chunk_index: usize,
        reason: &'a str,
    ) -> Self {
        Self {
            kind: "move",
            action,
            run_id,
            piece_id,
            chunk_index,
            reason,
            from_max_tokens: None,
            to_max_tokens: None,
            from_rung: None,
            to_rung: None,
            piece_bytes: None,
            split_cap: None,
            sub_pieces: None,
            answer_bytes: None,
        }
    }
}

/// Everything an attempt is reported to, bundled so the call sites
/// that classify attempts (ADR 0001 §7) report each one exactly once
/// and never know which observers are on: the opt-in diagnostics
/// sidecar (run-scoped, metadata) and the per-document attempts log
/// (default-on, full text).
#[derive(Clone, Copy)]
pub(super) struct Observers<'a> {
    pub(super) sink: Option<&'a DiagnosticsSink>,
    pub(super) log: Option<&'a AttemptLog>,
}

impl Observers<'_> {
    pub(super) fn emit(&self, attempt: &DiagnosticsAttempt, messages: &[serde_json::Value]) {
        if let Some(sink) = self.sink {
            sink.emit(attempt);
        }
        if let Some(log) = self.log {
            log.record(attempt, messages);
        }
    }

    /// One ladder move (ADR 0029) into the attempts log; the sidecar
    /// is unchanged — moves are per-document facts and live with the
    /// document's own record.
    pub(super) fn move_event(&self, record: &MoveRecord) {
        if let Some(log) = self.log {
            log.write_record(record);
        }
    }
}
