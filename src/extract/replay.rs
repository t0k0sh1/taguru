//! `ReplayIndex` (ADR 0031 §3.2/§3.4): an in-memory index over one
//! document's attempts log, built once at load time so a later run can
//! satisfy a completion from a recorded conversation instead of a live
//! model call. Internal only — no `--replay` flag reads this yet
//! (that's #819); this module exists to be unit-testable on its own.

// Nothing in production calls this module yet by design (#818's own
// scope: an internal API, verified by its unit tests) — the call
// sites that wire it into `Completions` land in #819.
#![allow(dead_code)]

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
                Err(error)
            }
        }
    }
}

/// The first turn a recorded conversation and a requested one
/// disagree on — the diagnostic [`ReplayIndex::lookup`] returns on a
/// miss, so an operator sees *why* nothing matched instead of a bare
/// "not found" (ADR 0031 §3.2).
pub(super) struct TurnDifference {
    pub(super) turn_index: usize,
    pub(super) recorded_role: Option<String>,
    pub(super) recorded_prefix: Option<String>,
    pub(super) requested_role: Option<String>,
    pub(super) requested_prefix: Option<String>,
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

/// The recorded conversations of one document's attempts log, indexed
/// for replay (ADR 0031 §3.1–§3.4). Built once, read-only except for
/// the FIFO queues a hit consumes from — those are the only mutation,
/// guarded by [`Mutex`] so `--parallel` workers (each pinned to a
/// distinct `piece_id`, per ADR 0031 §3.2) can share one index safely.
pub(super) struct ReplayIndex {
    pieces: Mutex<HashMap<String, PieceRecords>>,
    /// `system` records by `sha256`, for the pin decision (#821) —
    /// stored here because both need the same read of the log.
    systems: HashMap<String, String>,
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
        let mut systems: HashMap<String, String> = HashMap::new();
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
                    Some("system") => {
                        if let (Some(sha256), Some(content)) =
                            (value["sha256"].as_str(), value["content"].as_str())
                            && sha256_hex(content.as_bytes()) == sha256
                        {
                            systems.insert(sha256.to_string(), content.to_string());
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
                    _ => {}
                }
            }
        }
        Self {
            pieces: Mutex::new(pieces),
            systems,
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

    /// A `system` record's full text by `sha256` — the pin decision
    /// (#821) reads this once it has decided, from the set of
    /// distinct hashes seen, that exactly one applies.
    pub(super) fn system(&self, sha256: &str) -> Option<&str> {
        self.systems.get(sha256).map(String::as_str)
    }
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
        recorded_prefix: recorded_turn.map(|(_, field)| snippet(field)),
        requested_role: requested_turn.map(|(role, _)| role.clone()),
        requested_prefix: requested_turn.map(|(_, field)| snippet(field)),
    })
}

fn common_prefix_len(a: &[(String, String)], b: &[(String, String)]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}
