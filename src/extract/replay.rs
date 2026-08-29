//! `ReplayIndex` (ADR 0031 §3.2/§3.4): an in-memory index over one
//! document's attempts log, built once at load time so a later run can
//! satisfy a completion from a recorded conversation instead of a live
//! model call — read by `Completions::complete` and diffed against a
//! document's own settings in `Run::extract_document` (#819).

use super::*;
use std::collections::{HashMap, VecDeque};

/// Turns a live request's `messages` (the same shape
/// [`AttemptLog::record`] receives) into the `(role, field)` pairs
/// [`conversation_key`] and the miss diagnostic both compare on.
fn normalize_request(messages: &[serde_json::Value]) -> Vec<(String, String)> {
    messages
        .iter()
        .map(|message| {
            let role = message["role"].as_str().unwrap_or("").to_string();
            let content = message["content"].as_str().unwrap_or("");
            let field = if role == "system" {
                sha256_hex(content.as_bytes())
            } else {
                content.to_string()
            };
            (role, field)
        })
        .collect()
}

/// ADR 0031 §3.2's matching key: a normalized conversation digest plus
/// `requested_max_tokens`, so an escalated resend (same messages, a
/// larger cap) never collides with the attempt it escalated from.
type ReplayKey = (String, Option<usize>);

/// `sha256( for each turn: len(role) || role || len(field) || field )`
/// — length-prefixed so no content, including a stray NUL byte, can
/// ever be mistaken for a boundary between fields or between
/// messages (ADR 0031 §3.2).
fn conversation_key(turns: &[(String, String)], requested_max_tokens: Option<usize>) -> ReplayKey {
    let mut buf = Vec::new();
    for (role, field) in turns {
        buf.extend_from_slice(&(role.len() as u64).to_le_bytes());
        buf.extend_from_slice(role.as_bytes());
        buf.extend_from_slice(&(field.len() as u64).to_le_bytes());
        buf.extend_from_slice(field.as_bytes());
    }
    (sha256_hex(&buf), requested_max_tokens)
}

/// One `attempt` record, as much of it as replay needs to reconstruct
/// either a [`ChatCompletion`] or the [`ChatError`] the original call
/// ended in — parsing, validation, and mechanical review are not
/// stored here because a replay run redoes them itself against the
/// replayed answer (that redo is the entire point of #781).
struct StoredAttempt {
    turns: Vec<(String, String)>,
    requested_max_tokens: Option<usize>,
    attempt: usize,
    state: String,
    answer: Option<String>,
    finish_reason: Option<String>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    transport_retries: usize,
    /// This record's own `(run_id, attempt_seq)` (ADR 0023) — the
    /// origin [`ChatCompletion::replayed_from`] names on a hit (#823),
    /// so a reader can join a replay run's re-emitted `attempt` record
    /// back to the one carrying its *real* `elapsed_seconds`/tokens.
    origin: AttemptRef,
}

impl StoredAttempt {
    /// Reconstructs the outcome [`ChatClient::complete`] would have
    /// returned. `answer: null` on the record means `timeout` or
    /// `transport` (ADR 0025 §3.3) — every other state carries an
    /// answer, so any other `None` here is treated as `transport`
    /// rather than panicking on a record this build doesn't
    /// recognize.
    fn to_outcome(&self, piece_id: &str) -> Result<ChatCompletion, ChatError> {
        match &self.answer {
            Some(answer) => Ok(ChatCompletion {
                content: answer.clone(),
                finish_reason: self.finish_reason.clone(),
                usage: (self.input_tokens.is_some() || self.output_tokens.is_some()).then_some(
                    TokenUsage {
                        input_tokens: self.input_tokens,
                        output_tokens: self.output_tokens,
                        total_tokens: None,
                    },
                ),
                transport_retries: self.transport_retries,
                replayed_from: Some(self.origin.clone()),
            }),
            None => {
                let kind = if self.state == "timeout" {
                    ChatFailure::Timeout
                } else {
                    ChatFailure::Transport
                };
                let mut error = ChatError::new(
                    kind,
                    format!(
                        "replayed a recorded {} completion for piece {piece_id} attempt {}",
                        self.state, self.attempt
                    ),
                );
                error.transport_retries = self.transport_retries;
                error.replayed_from = Some(self.origin.clone());
                Err(error)
            }
        }
    }
}

/// The first turn a recorded conversation and a requested one
/// disagree on — the diagnostic [`ReplayIndex::lookup`] returns on a
/// miss, so an operator sees *why* nothing matched instead of a bare
/// "not found" (ADR 0031 §3.2). Carries a `sha256` of each side's
/// turn, never the text itself: this diagnostic reaches stderr and a
/// failed document's `ChatError` message (and, from there, the
/// diagnostics sidecar's `parse_error`), which are metadata by design
/// (ADR 0001 §10) — the document text belongs to the attempts log's
/// own `messages` alone. A digest still lets an operator confirm
/// *which* recorded turn a candidate string matches without this
/// diagnostic ever carrying document content itself.
pub(super) struct TurnDifference {
    pub(super) turn_index: usize,
    pub(super) recorded_role: Option<String>,
    pub(super) recorded_digest: Option<String>,
    pub(super) requested_role: Option<String>,
    pub(super) requested_digest: Option<String>,
}

/// [`ReplayIndex::lookup`]'s miss report: how many attempts this
/// `piece_id` has on record (0 when the piece itself is unknown) and,
/// when at least one exists, where the closest of them first diverges
/// from what was actually asked.
pub(super) struct MissDiagnostic {
    pub(super) piece_id: String,
    pub(super) recorded: usize,
    pub(super) first_difference: Option<TurnDifference>,
}

/// [`ReplayIndex::lookup`]'s result: either a reconstructed outcome
/// (a hit, consumed FIFO — ADR 0031 §3.2) or a diagnosable miss.
pub(super) enum ReplayLookup {
    Hit(Result<ChatCompletion, ChatError>),
    Miss(MissDiagnostic),
}

/// One `piece_id`'s recorded attempts: kept in file order (`records`)
/// for the miss diagnostic, and indexed by matching key (`by_key`, a
/// FIFO queue of indices into `records`) for a hit. ADR 0031 §3.2:
/// grouping by `piece_id` first is what makes `--parallel` replay
/// determinism structural — two different pieces never share a
/// `piece_id`, so no worker's FIFO consumption can race another's.
#[derive(Default)]
struct PieceRecords {
    records: Vec<StoredAttempt>,
    by_key: HashMap<ReplayKey, VecDeque<usize>>,
}

/// One `system` record: its text, and the run_id in effect (the
/// nearest preceding `document` record) when it was first seen —
/// [`ReplayIndex::pinned_system`]'s `pinned_from`.
struct RecordedSystem {
    content: String,
    run_id: String,
}

/// [`ReplayIndex::pinned_system`]'s decision (ADR 0031 §3.6).
pub(super) enum SystemPinDecision<'a> {
    /// Exactly one distinct `system` record — pin it verbatim, and the
    /// run_id it was originally recorded under.
    Pin { content: &'a str, run_id: &'a str },
    /// More than one distinct text recorded (a checkpoint-resumed
    /// document whose log spans a run where the vocabulary differed)
    /// — ambiguity is never resolved by guessing.
    Ambiguous { distinct: usize },
    /// No `system` record at all — nothing to pin.
    NoRecord,
}

/// The recorded conversations of one document's attempts log, indexed
/// for replay (ADR 0031 §3.1–§3.4). Built once, read-only except for
/// the FIFO queues a hit consumes from — those are the only mutation,
/// guarded by [`Mutex`] so `--parallel` workers (each pinned to a
/// distinct `piece_id`, per ADR 0031 §3.2) can share one index safely.
pub(super) struct ReplayIndex {
    pieces: Mutex<HashMap<String, PieceRecords>>,
    /// `system` records by `sha256` — [`ReplayIndex::pinned_system`]
    /// reads this for the pin decision (ADR 0031 §3.6).
    systems: HashMap<String, RecordedSystem>,
    /// The last `kind: "settings"` record seen in file order (ADR
    /// 0031 §3.9): a checkpoint-resumed document's log can hold one
    /// per run it spans, and the most recent is the closest baseline
    /// to diff a replay run's own settings against. Diagnostic only —
    /// never consulted by matching itself.
    settings: Option<RecordedSettings>,
}

impl ReplayIndex {
    /// Reads the whole attempts log into memory before this run opens
    /// its own log for writing (ADR 0031 §3.4), so a later truncate or
    /// append never disturbs what was already loaded. A missing or
    /// unreadable file, and any individual malformed line, degrade to
    /// "no record" — the same posture [`Manifest::load`] takes —
    /// never an error.
    pub(super) fn load(path: &Path) -> Self {
        let mut pieces: HashMap<String, PieceRecords> = HashMap::new();
        let mut systems: HashMap<String, RecordedSystem> = HashMap::new();
        let mut settings: Option<RecordedSettings> = None;
        // The `document` record most recently seen in file order —
        // `system`'s own record carries no run_id (ADR 0025 §3.3: it
        // is written once per document, not once per run), so this is
        // how a system record's ORIGINATING run is recovered for
        // `pinned_from` (ADR 0031 §3.6). `None` until a valid
        // `document` record is seen — a `system` record with no
        // preceding one (a truncated or malformed log) names no real
        // origin and must never be registered as pinnable.
        let mut current_run_id: Option<String> = None;
        if let Ok(text) = fs::read_to_string(path) {
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                    continue;
                };
                match value["kind"].as_str() {
                    Some("document") => {
                        current_run_id = value["run_id"].as_str().map(str::to_string);
                    }
                    Some("system") => {
                        if let (Some(sha256), Some(content), Some(run_id)) = (
                            value["sha256"].as_str(),
                            value["content"].as_str(),
                            current_run_id.as_deref(),
                        ) && sha256_hex(content.as_bytes()) == sha256
                        {
                            // The first run to record this hash owns
                            // it — a later run's identical `system`
                            // line (the same prompt sent again) is not
                            // a different origin.
                            systems
                                .entry(sha256.to_string())
                                .or_insert_with(|| RecordedSystem {
                                    content: content.to_string(),
                                    run_id: run_id.to_string(),
                                });
                        }
                    }
                    Some("attempt") => {
                        if let Some(stored) = parse_attempt_record(&value) {
                            let piece_id = value["piece_id"].as_str().unwrap_or("").to_string();
                            let key = conversation_key(&stored.turns, stored.requested_max_tokens);
                            let piece = pieces.entry(piece_id).or_default();
                            let index = piece.records.len();
                            piece.by_key.entry(key).or_default().push_back(index);
                            piece.records.push(stored);
                        }
                    }
                    Some("settings") => {
                        if let Some(parsed) = parse_settings_record(&value) {
                            settings = Some(parsed);
                        }
                    }
                    _ => {}
                }
            }
        }
        Self {
            pieces: Mutex::new(pieces),
            systems,
            settings,
        }
    }

    /// Looks up a match for `piece_id`'s next request. A hit pops the
    /// oldest still-unconsumed record sharing this exact key (ADR
    /// 0031 §3.2's FIFO rule); anything else is a miss, diagnosed
    /// against whichever recorded attempt of this piece shares the
    /// longest matching prefix with what was actually asked.
    pub(super) fn lookup(
        &self,
        piece_id: &str,
        messages: &[serde_json::Value],
        requested_max_tokens: Option<usize>,
    ) -> ReplayLookup {
        let turns = normalize_request(messages);
        let key = conversation_key(&turns, requested_max_tokens);
        let mut pieces = match self.pieces.lock() {
            Ok(pieces) => pieces,
            Err(poisoned) => poisoned.into_inner(),
        };
        let Some(piece) = pieces.get_mut(piece_id) else {
            return ReplayLookup::Miss(MissDiagnostic {
                piece_id: piece_id.to_string(),
                recorded: 0,
                first_difference: None,
            });
        };
        if let Some(index) = piece.by_key.get_mut(&key).and_then(VecDeque::pop_front) {
            return ReplayLookup::Hit(piece.records[index].to_outcome(piece_id));
        }
        ReplayLookup::Miss(MissDiagnostic {
            piece_id: piece_id.to_string(),
            recorded: piece.records.len(),
            first_difference: closest_difference(&turns, &piece.records),
        })
    }

    /// Whether to pin this document's system prompt verbatim from the
    /// log, and why not when it declines (ADR 0031 §3.6).
    pub(super) fn pinned_system(&self) -> SystemPinDecision<'_> {
        let mut systems = self.systems.values();
        let Some(only) = systems.next() else {
            return SystemPinDecision::NoRecord;
        };
        let remaining = systems.count();
        if remaining > 0 {
            return SystemPinDecision::Ambiguous {
                distinct: remaining + 1,
            };
        }
        SystemPinDecision::Pin {
            content: &only.content,
            run_id: &only.run_id,
        }
    }

    /// The log's most recent `settings` record, for a replay run to
    /// diff its own settings against (ADR 0031 §3.2/§3.9) — a hint,
    /// never a gate: matching itself is decided by the conversation.
    pub(super) fn settings(&self) -> Option<&RecordedSettings> {
        self.settings.as_ref()
    }
}

/// [`SettingsRecord`]'s eleven fields, parsed back from JSON, minus
/// `rung` (never part of a settings comparison; ADR 0031 §3.2 keeps
/// it off the matching key for the same reason). `SettingsRecord`
/// already excludes `CheckpointFingerprint`'s `sha256`/`context`/
/// `no_passage`/`description`/`escalation_factor` — values that name
/// the document or never reach the model — so `settings_differences`
/// never reports on those either; that is inherited from
/// `SettingsRecord`'s own contract, not a gap here.
pub(super) struct RecordedSettings {
    pub(super) prompt_version: u64,
    pub(super) model: String,
    pub(super) questions_n: u64,
    pub(super) fact_budget: u64,
    pub(super) structured_output: String,
    pub(super) max_output_tokens: u64,
    pub(super) chunk_bytes: String,
    pub(super) lossy: bool,
    pub(super) schema_digest: String,
    pub(super) candidates: String,
    pub(super) vocabulary_digest: String,
}

fn parse_settings_record(value: &serde_json::Value) -> Option<RecordedSettings> {
    Some(RecordedSettings {
        prompt_version: value["prompt_version"].as_u64()?,
        model: value["model"].as_str()?.to_string(),
        questions_n: value["questions_n"].as_u64()?,
        fact_budget: value["fact_budget"].as_u64()?,
        structured_output: value["structured_output"].as_str()?.to_string(),
        max_output_tokens: value["max_output_tokens"].as_u64()?,
        chunk_bytes: value["chunk_bytes"].as_str()?.to_string(),
        lossy: value["lossy"].as_bool()?,
        schema_digest: value["schema_digest"].as_str()?.to_string(),
        candidates: value["candidates"].as_str()?.to_string(),
        vocabulary_digest: value["vocabulary_digest"].as_str()?.to_string(),
    })
}

/// Names every field where `recorded` (the log's last `settings`
/// record) and `current` (this run's own settings) disagree, each as
/// `"field old → new"` — the stderr line's own words (ADR 0031 §3.2's
/// diagnostic point). Empty when nothing differs; `prompt_version`'s
/// difference is worded as itself, not decoded, since it names a
/// build, not a setting an operator chose.
pub(super) fn settings_differences(
    recorded: &RecordedSettings,
    current: &RecordedSettings,
) -> Vec<String> {
    let mut differences = Vec::new();
    macro_rules! diff {
        ($field:ident, $label:literal) => {
            if recorded.$field != current.$field {
                differences.push(format!(
                    "{} {:?} \u{2192} {:?}",
                    $label, recorded.$field, current.$field
                ));
            }
        };
    }
    diff!(prompt_version, "prompt_version");
    diff!(model, "model");
    diff!(questions_n, "questions");
    diff!(fact_budget, "fact_budget");
    diff!(structured_output, "structured_output");
    diff!(max_output_tokens, "max_output_tokens");
    diff!(chunk_bytes, "chunk_bytes");
    diff!(lossy, "lossy");
    diff!(schema_digest, "schema_digest");
    diff!(candidates, "candidates");
    diff!(vocabulary_digest, "vocabulary_digest");
    differences
}

/// ADR 0001 §7's attempt-state vocabulary (also documented on
/// [`DiagnosticsAttempt::state`]) — a `state` outside this set is not
/// one this build recognizes, so the whole record is rejected rather
/// than silently misclassified.
const KNOWN_STATES: &[&str] = &[
    "stop_valid",
    "stop_malformed",
    "length_limited",
    "empty",
    "refusal",
    "timeout",
    "transport",
];

/// Reads an optional numeric field: absent or `null` is a real
/// "no value" (`Some(None)`), a JSON number is that value
/// (`Some(Some(n))`), and anything else — a string, a bool — is a
/// malformed record (`None`), never silently coerced to "no value"
/// the way [`serde_json::Value::as_u64`] alone would (a `"512"` string
/// would otherwise read as an absent cap instead of a corrupt one).
fn optional_u64(value: &serde_json::Value, key: &str) -> Option<Option<u64>> {
    match value.get(key) {
        None | Some(serde_json::Value::Null) => Some(None),
        Some(field) => field.as_u64().map(Some),
    }
}

/// Extracts one `attempt` record's replay-relevant fields from its
/// parsed JSON. `None` for any record missing what replay needs to
/// index it (a truncated last line, a future field this build does
/// not know), carrying a `state` outside [`KNOWN_STATES`], a
/// wrong-typed optional numeric field, or an `answer`/`state`
/// combination ADR 0025 §3.3 never produces — that record is simply
/// not offered for replay, never a load failure.
fn parse_attempt_record(value: &serde_json::Value) -> Option<StoredAttempt> {
    let run_id = value["run_id"].as_str()?.to_string();
    let attempt_seq = value["attempt_seq"].as_u64()?;
    let attempt = value["attempt"].as_u64()? as usize;
    let state = value["state"].as_str()?.to_string();
    if !KNOWN_STATES.contains(&state.as_str()) {
        return None;
    }
    let requested_max_tokens = optional_u64(value, "requested_max_tokens")?.map(|n| n as usize);
    let messages = value["messages"].as_array()?;
    let mut turns = Vec::with_capacity(messages.len());
    for message in messages {
        let role = message["role"].as_str()?.to_string();
        let field = if role == "system" {
            message["system_sha256"].as_str()?.to_string()
        } else {
            message["content"].as_str()?.to_string()
        };
        turns.push((role, field));
    }
    let answer = value["answer"].as_str().map(str::to_string);
    // ADR 0025 §3.3: `answer` is `null` exactly for `timeout`/
    // `transport` — every other state carries one. A record that
    // violates this is internally inconsistent.
    let expects_answer = state != "timeout" && state != "transport";
    if expects_answer != answer.is_some() {
        return None;
    }
    let input_tokens = optional_u64(value, "input_tokens")?;
    let output_tokens = optional_u64(value, "output_tokens")?;
    let transport_retries = optional_u64(value, "transport_retries")?.unwrap_or(0) as usize;
    Some(StoredAttempt {
        turns,
        requested_max_tokens,
        attempt,
        state,
        answer,
        finish_reason: value["finish_reason"].as_str().map(str::to_string),
        input_tokens,
        output_tokens,
        transport_retries,
        origin: AttemptRef {
            run_id,
            attempt_seq,
        },
    })
}

/// Among `records`, finds the one sharing the longest leading run of
/// identical `(role, field)` turns with `turns`, and reports where
/// the two first disagree — the "closest" recorded attempt, not just
/// the first one in file order, so a piece with several stale
/// recordings still points at the most informative comparison.
fn closest_difference(
    turns: &[(String, String)],
    records: &[StoredAttempt],
) -> Option<TurnDifference> {
    let (closest, prefix_len) = records
        .iter()
        .map(|record| (record, common_prefix_len(turns, &record.turns)))
        .max_by_key(|(_, len)| *len)?;
    let recorded_turn = closest.turns.get(prefix_len);
    let requested_turn = turns.get(prefix_len);
    Some(TurnDifference {
        turn_index: prefix_len,
        recorded_role: recorded_turn.map(|(role, _)| role.clone()),
        recorded_digest: recorded_turn.map(|(_, field)| sha256_hex(field.as_bytes())),
        requested_role: requested_turn.map(|(role, _)| role.clone()),
        requested_digest: requested_turn.map(|(_, field)| sha256_hex(field.as_bytes())),
    })
}

fn common_prefix_len(a: &[(String, String)], b: &[(String, String)]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}
