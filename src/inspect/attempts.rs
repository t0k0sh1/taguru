//! `taguru inspect <batch stem>.attempts.jsonl` (ADR 0037, #850): the
//! human's way into an extract attempts log (ADR 0025). The log holds
//! every completion a document had with the model — the piece as sent,
//! the answer, the ladder's moves — but as one JSON line per record,
//! `messages` escaped, nothing addressed by paragraph. A failed
//! document's stderr line names its piece (ADR 0037 §3.1); this view
//! is where that name resolves to text a person can read: one row per
//! attempt in issue order, and under `--piece`/`--paragraph`, the piece
//! text and the answer verbatim.
//!
//! Same posture as the rest of `inspect`: `--json` is a second
//! rendering of the one [`AttemptsReport`] the text view prints, never
//! a second reading of the file. A torn trailing line (a killed run)
//! is counted and said, never a failure — the log is written
//! incrementally on purpose.

use std::path::Path;

use serde::Serialize;
use serde_json::Value;

use crate::extract::{
    paragraph_range, shell_quote, short_piece_id, user_message_document, user_message_part,
};

/// What `--piece`/`--paragraph` narrowed the view to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Filter {
    /// Every attempt, one summary row each — no texts.
    All,
    /// The attempts of pieces whose id starts with this prefix, texts
    /// included.
    Piece(String),
    /// The attempts of pieces whose `[N]` label range covers this
    /// paragraph, texts included.
    Paragraph(u32),
}

/// One attempt of the log, as the report states it. `piece_text`,
/// `corrective_ask`, and `answer` are read for every row (the
/// paragraph range comes from the text) but rendered — and serialized
/// — only under a narrowing filter, so the unfiltered view stays one
/// line per attempt.
#[derive(Serialize)]
pub(super) struct AttemptRow {
    seq: u64,
    chunk_index: usize,
    chunk_total: Option<usize>,
    piece_id: String,
    paragraph_first: Option<u32>,
    paragraph_last: Option<u32>,
    piece_bytes: Option<usize>,
    stage: String,
    attempt: u64,
    max_attempts: u64,
    state: String,
    finish_reason: Option<String>,
    length_limited: bool,
    rung: Option<String>,
    requested_max_tokens: Option<u64>,
    elapsed_seconds: Option<f64>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    corrects: Option<Value>,
    replayed_from: Option<Value>,
    parse_error: Option<String>,
    validation_issues: Vec<String>,
    removed_items: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    piece_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    corrective_ask: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    answer: Option<String>,
}

/// One ladder move (ADR 0029), placed in the report after the attempt
/// it followed.
#[derive(Serialize)]
pub(super) struct MoveRow {
    after_seq: Option<u64>,
    action: String,
    piece_id: String,
    chunk_index: usize,
    reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    from_max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    to_max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    from_rung: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    to_rung: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    piece_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    split_cap: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sub_pieces: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    answer_bytes: Option<u64>,
}

#[derive(Serialize)]
pub(super) struct DocumentRow {
    source: String,
    run_id: String,
    resumed: bool,
}

/// The whole view, built once from the file and rendered twice.
#[derive(Serialize)]
pub(super) struct AttemptsReport {
    target: String,
    kind: &'static str,
    /// The last `document` record — a resumed document appends a new
    /// one per run, and the latest run is the one whose attempts the
    /// tail of the file holds.
    document: Option<DocumentRow>,
    /// Every `document` record's `run_id`, oldest first.
    runs: Vec<String>,
    settings: Option<Value>,
    filter: String,
    attempts: Vec<AttemptRow>,
    moves: Vec<MoveRow>,
    /// Lines that did not parse as JSON objects — a torn tail from a
    /// killed run, most often.
    unreadable_lines: usize,
    /// Attempts matched by the filter (every attempt under `All`).
    matched: usize,
}

/// Reads the log and renders it; the exit code is 0 for a readable
/// log whether or not the filter matched anything (an empty match is
/// said, not failed), 1 when the file cannot be read or holds no
/// attempts-log record at all.
pub(super) fn inspect_attempts_log(path: &Path, filter: &Filter, as_json: bool) -> i32 {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("{}: UNREADABLE — {error}", path.display());
            return 1;
        }
    };
    let report = build_report(&path.display().to_string(), &text, filter);
    if report.document.is_none() && report.attempts.is_empty() && report.moves.is_empty() {
        eprintln!(
            "{}: not an extract attempts log — no document or attempt record in it",
            path.display()
        );
        return 1;
    }
    let rendered = if as_json {
        match serde_json::to_string_pretty(&report) {
            Ok(json) => json + "\n",
            Err(error) => {
                eprintln!("taguru: inspect: rendering JSON: {error}");
                return 1;
            }
        }
    } else {
        render_text(&report, filter)
    };
    // A piece is pages of text, so this view is what gets piped into
    // `head` or `less`; a reader that closes early is not an error.
    use std::io::Write;
    let mut stdout = std::io::stdout().lock();
    match stdout
        .write_all(rendered.as_bytes())
        .and_then(|()| stdout.flush())
    {
        Ok(()) => 0,
        Err(error) => write_failure_exit(&error),
    }
}

/// The exit for a failed stdout write: `| head` closing the pipe after
/// its first lines is the reader's choice, not a failure (0, silent);
/// anything else is said and exits 1. Only reproducible by a reader
/// that exits mid-stream, which no in-process test can stage, so the
/// judgment is not mutation-tested.
#[mutants::skip]
fn write_failure_exit(error: &std::io::Error) -> i32 {
    if error.kind() == std::io::ErrorKind::BrokenPipe {
        0
    } else {
        eprintln!("taguru: inspect: writing stdout: {error}");
        1
    }
}

pub(super) fn build_report(target: &str, text: &str, filter: &Filter) -> AttemptsReport {
    let mut report = AttemptsReport {
        target: target.to_string(),
        kind: "attempts",
        document: None,
        runs: Vec::new(),
        settings: None,
        filter: describe_filter(filter),
        attempts: Vec::new(),
        moves: Vec::new(),
        unreadable_lines: 0,
        matched: 0,
    };
    let mut last_seq: Option<u64> = None;
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(Value::Object(record)) = serde_json::from_str::<Value>(line) else {
            report.unreadable_lines += 1;
            continue;
        };
        match record.get("kind").and_then(Value::as_str) {
            Some("document") => {
                let run_id = str_field(&record, "run_id");
                report.runs.push(run_id.clone());
                report.document = Some(DocumentRow {
                    source: str_field(&record, "source"),
                    run_id,
                    resumed: record
                        .get("resumed")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                });
            }
            Some("settings") => {
                let mut settings = Value::Object(record);
                if let Value::Object(map) = &mut settings {
                    map.remove("kind");
                }
                report.settings = Some(settings);
            }
            Some("attempt") => {
                let row = attempt_row(&record);
                last_seq = Some(row.seq);
                report.attempts.push(row);
            }
            Some("move") => report.moves.push(MoveRow {
                after_seq: last_seq,
                action: str_field(&record, "move"),
                piece_id: str_field(&record, "piece_id"),
                chunk_index: u64_field(&record, "chunk_index").unwrap_or(0) as usize,
                reason: str_field(&record, "reason"),
                from_max_tokens: u64_field(&record, "from_max_tokens"),
                to_max_tokens: u64_field(&record, "to_max_tokens"),
                from_rung: opt_str_field(&record, "from_rung"),
                to_rung: opt_str_field(&record, "to_rung"),
                piece_bytes: u64_field(&record, "piece_bytes"),
                split_cap: u64_field(&record, "split_cap"),
                sub_pieces: u64_field(&record, "sub_pieces"),
                answer_bytes: u64_field(&record, "answer_bytes"),
            }),
            // `system`, `replay`, `replay_summary`, and anything a
            // later version adds: not this view's rows.
            _ => {}
        }
    }
    // The filter decides which rows keep their texts; every row keeps
    // its summary so a narrowed view still reads in the log's order.
    let hit = |row: &AttemptRow| match filter {
        Filter::All => true,
        Filter::Piece(prefix) => row.piece_id.starts_with(prefix.as_str()),
        Filter::Paragraph(n) => matches!(
            (row.paragraph_first, row.paragraph_last),
            (Some(first), Some(last)) if first <= *n && *n <= last
        ),
    };
    let matched = report.attempts.iter().filter(|row| hit(row)).count();
    if *filter == Filter::All {
        for row in &mut report.attempts {
            row.piece_text = None;
            row.corrective_ask = None;
            row.answer = None;
        }
    } else {
        // The matched rows stay whether or not they carry a text — an
        // attempt that sent nothing (transport failure) or got nothing
        // back (timeout) is still the piece's row.
        report.attempts.retain(|row| hit(row));
        let kept: std::collections::BTreeSet<String> = report
            .attempts
            .iter()
            .map(|row| row.piece_id.clone())
            .collect();
        report.moves.retain(|m| kept.contains(&m.piece_id));
    }
    report.matched = matched;
    report
}

fn attempt_row(record: &serde_json::Map<String, Value>) -> AttemptRow {
    let turns: Vec<&Value> = record
        .get("messages")
        .and_then(Value::as_array)
        .map(|turns| turns.iter().collect())
        .unwrap_or_default();
    let user_turns: Vec<&str> = turns
        .iter()
        .filter(|turn| turn.get("role").and_then(Value::as_str) == Some("user"))
        .filter_map(|turn| turn.get("content").and_then(Value::as_str))
        .collect();
    // The first user turn carries the piece behind its preamble; a
    // corrective round adds the replayed answer and the ask after it.
    let first_user = user_turns.first().copied();
    let piece_text = first_user.map(|user| user_message_document(user).to_string());
    // A single-chunk document's turn carries no `part K of N` clause:
    // that is chunk 1 of 1. A Stage 2 correction's turn carries none
    // either, and is no chunk at all.
    let stage = str_field(record, "stage");
    let chunk_total = first_user
        .and_then(user_message_part)
        .map(|(_, total)| total)
        .or_else(|| (stage == "item" && first_user.is_some()).then_some(1));
    let corrective_ask = (user_turns.len() > 1)
        .then(|| user_turns.last().copied())
        .flatten()
        .map(str::to_string);
    let range = piece_text.as_deref().and_then(paragraph_range);
    AttemptRow {
        seq: u64_field(record, "attempt_seq").unwrap_or(0),
        chunk_index: u64_field(record, "chunk_index").unwrap_or(0) as usize,
        chunk_total,
        piece_id: str_field(record, "piece_id"),
        paragraph_first: range.map(|(first, _)| first),
        paragraph_last: range.map(|(_, last)| last),
        piece_bytes: piece_text.as_ref().map(String::len),
        stage,
        attempt: u64_field(record, "attempt").unwrap_or(0),
        max_attempts: u64_field(record, "max_attempts").unwrap_or(0),
        state: str_field(record, "state"),
        finish_reason: opt_str_field(record, "finish_reason"),
        length_limited: record
            .get("length_limited")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        rung: opt_str_field(record, "rung"),
        requested_max_tokens: u64_field(record, "requested_max_tokens"),
        elapsed_seconds: record.get("elapsed_seconds").and_then(Value::as_f64),
        input_tokens: u64_field(record, "input_tokens"),
        output_tokens: u64_field(record, "output_tokens"),
        corrects: record.get("corrects").filter(|v| !v.is_null()).cloned(),
        replayed_from: record
            .get("replayed_from")
            .filter(|v| !v.is_null())
            .cloned(),
        parse_error: opt_str_field(record, "parse_error"),
        validation_issues: str_list(record, "validation_issues"),
        removed_items: str_list(record, "removed_items"),
        piece_text,
        corrective_ask,
        answer: opt_str_field(record, "answer"),
    }
}

fn str_field(record: &serde_json::Map<String, Value>, key: &str) -> String {
    opt_str_field(record, key).unwrap_or_default()
}

fn opt_str_field(record: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    record.get(key).and_then(Value::as_str).map(str::to_string)
}

fn u64_field(record: &serde_json::Map<String, Value>, key: &str) -> Option<u64> {
    record.get(key).and_then(Value::as_u64)
}

fn str_list(record: &serde_json::Map<String, Value>, key: &str) -> Vec<String> {
    record
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn describe_filter(filter: &Filter) -> String {
    match filter {
        Filter::All => "all".to_string(),
        Filter::Piece(prefix) => format!("piece {prefix}"),
        Filter::Paragraph(n) => format!("paragraph {n}"),
    }
}

/// The rows print piece ids as the failure line does (ADR 0037 §3.1),
/// so the one pastes into `--piece` as the other.
pub(super) fn short_id(piece_id: &str) -> &str {
    short_piece_id(piece_id)
}

/// The row's "where": `chunk 1/3`, then the piece and its paragraphs.
fn locate(row: &AttemptRow) -> String {
    let chunk = match row.chunk_total {
        Some(total) => format!("chunk {}/{total}", row.chunk_index + 1),
        None if row.stage == "cross_chunk" => {
            format!("cross-chunk (chunk {})", row.chunk_index + 1)
        }
        None => format!("chunk {}", row.chunk_index + 1),
    };
    let mut text = format!("{chunk}  piece {}", short_id(&row.piece_id));
    match (row.paragraph_first, row.paragraph_last, row.piece_bytes) {
        (Some(first), Some(last), Some(bytes)) if first == last => {
            text.push_str(&format!("  paragraph {first} ({bytes} B)"));
        }
        (Some(first), Some(last), Some(bytes)) => {
            text.push_str(&format!("  paragraphs {first}–{last} ({bytes} B)"));
        }
        (_, _, Some(bytes)) if bytes > 0 => text.push_str(&format!("  ({bytes} B, unlabeled)")),
        _ => {}
    }
    text
}

/// One attempt's summary line and its detail line(s), as the text
/// view prints them under any filter.
fn summary_lines(row: &AttemptRow) -> Vec<String> {
    let stage = match row.stage.as_str() {
        "item" => format!("attempt {}/{}", row.attempt, row.max_attempts),
        "cross_chunk" => format!("correction {}/{}", row.attempt, row.max_attempts),
        other => format!("{other} {}/{}", row.attempt, row.max_attempts),
    };
    let mut line = format!("  #{}  {}  {stage}  {}", row.seq, locate(row), row.state);
    if let Some(reason) = &row.finish_reason
        && reason != "stop"
    {
        line.push_str(&format!(" (finish {reason})"));
    }
    if let Some(cap) = row.requested_max_tokens {
        line.push_str(&format!("  cap {cap} tok"));
    }
    if let Some(rung) = &row.rung {
        line.push_str(&format!("  {rung}"));
    }
    if let Some(seconds) = row.elapsed_seconds {
        line.push_str(&format!("  {seconds:.1} s"));
    }
    if let (Some(input), Some(output)) = (row.input_tokens, row.output_tokens) {
        line.push_str(&format!("  {input}→{output} tok"));
    }
    if let Some(corrects) = &row.corrects
        && let Some(seq) = corrects.get("attempt_seq").and_then(Value::as_u64)
    {
        line.push_str(&format!("  corrects #{seq}"));
    }
    if let Some(from) = &row.replayed_from
        && let Some(seq) = from.get("attempt_seq").and_then(Value::as_u64)
    {
        line.push_str(&format!("  replayed from #{seq}"));
    }
    let mut lines = vec![line];
    if let Some(error) = &row.parse_error {
        lines.push(format!("        {error}"));
    }
    let listed = row.validation_issues.len().min(3);
    for issue in &row.validation_issues[..listed] {
        lines.push(format!("        {issue}"));
    }
    if row.validation_issues.len() > listed {
        lines.push(format!(
            "        … and {} more issue(s)",
            row.validation_issues.len() - listed
        ));
    }
    if !row.removed_items.is_empty() {
        lines.push(format!(
            "        {} item(s) removed (mechanical validation)",
            row.removed_items.len()
        ));
    }
    lines
}

fn move_line(m: &MoveRow) -> String {
    let what = match m.action.as_str() {
        "escalate" => format!(
            "max_tokens {} → {}",
            m.from_max_tokens.unwrap_or(0),
            m.to_max_tokens.unwrap_or(0)
        ),
        "split" => format!(
            "{} sub-piece(s), cap {} B",
            m.sub_pieces.unwrap_or(0),
            m.split_cap.unwrap_or(0)
        ),
        "demote" => format!(
            "{} → {}",
            m.from_rung.as_deref().unwrap_or("?"),
            m.to_rung.as_deref().unwrap_or("?")
        ),
        "runaway" => format!(
            "{} B answered for a {} B piece",
            m.answer_bytes.unwrap_or(0),
            m.piece_bytes.unwrap_or(0)
        ),
        _ => String::new(),
    };
    let mut line = format!("      ↳ {}", m.action);
    if !what.is_empty() {
        line.push_str(&format!("  {what}"));
    }
    if !m.reason.is_empty() {
        line.push_str(&format!(" — {}", m.reason));
    }
    line
}

pub(super) fn render_text(report: &AttemptsReport, filter: &Filter) -> String {
    let mut out = String::new();
    match &report.document {
        Some(document) => {
            let runs = if report.runs.len() > 1 {
                format!(
                    ", {} run(s) — latest run {}",
                    report.runs.len(),
                    document.run_id
                )
            } else {
                format!(", run {}", document.run_id)
            };
            out.push_str(&format!(
                "{}: attempts log for '{}'{runs}{}\n",
                report.target,
                document.source,
                if document.resumed { " (resumed)" } else { "" }
            ));
        }
        None => out.push_str(&format!(
            "{}: attempts log (no document record)\n",
            report.target
        )),
    }
    if let Some(Value::Object(settings)) = &report.settings {
        let shown: Vec<String> = [
            "model",
            "prompt_version",
            "structured_output",
            "max_output_tokens",
            "chunk_bytes",
            "chunk_context",
            "rung",
        ]
        .iter()
        .filter_map(|key| {
            settings.get(*key).and_then(|value| match value {
                Value::String(text) if text.is_empty() => None,
                Value::String(text) => Some(format!("{key} {text}")),
                Value::Number(number) if number.as_u64() == Some(0) => None,
                Value::Number(number) => Some(format!("{key} {number}")),
                _ => None,
            })
        })
        .collect();
        if !shown.is_empty() {
            out.push_str(&format!("  settings: {}\n", shown.join(", ")));
        }
    }
    let mut moves = report.moves.iter().peekable();
    // Moves recorded before any attempt (none today) print first.
    while let Some(m) = moves.peek()
        && m.after_seq.is_none()
    {
        out.push_str(&move_line(m));
        out.push('\n');
        moves.next();
    }
    let mut shown_pieces: std::collections::BTreeMap<String, u64> =
        std::collections::BTreeMap::new();
    for row in &report.attempts {
        for line in summary_lines(row) {
            out.push_str(&line);
            out.push('\n');
        }
        if *filter != Filter::All {
            if let Some(text) = &row.piece_text {
                match shown_pieces.get(&row.piece_id) {
                    Some(seq) => {
                        out.push_str(&format!("        (piece text as sent: same as #{seq})\n"))
                    }
                    None => {
                        shown_pieces.insert(row.piece_id.clone(), row.seq);
                        out.push_str(&format!(
                            "--- piece text as sent (#{}, {} B) ---\n{}\n",
                            row.seq,
                            text.len(),
                            text.trim_end_matches('\n')
                        ));
                    }
                }
            }
            if let Some(ask) = &row.corrective_ask {
                out.push_str(&format!(
                    "--- corrective ask (#{}, {} B) ---\n{}\n",
                    row.seq,
                    ask.len(),
                    ask.trim_end_matches('\n')
                ));
            }
            match &row.answer {
                Some(answer) => out.push_str(&format!(
                    "--- answer (#{}, {} B) ---\n{}\n",
                    row.seq,
                    answer.len(),
                    answer.trim_end_matches('\n')
                )),
                None => out.push_str(&format!("--- answer (#{}) --- none arrived\n", row.seq)),
            }
            out.push_str("--- end ---\n");
        }
        while let Some(m) = moves.peek()
            && m.after_seq == Some(row.seq)
        {
            out.push_str(&move_line(m));
            out.push('\n');
            moves.next();
        }
    }
    for m in moves {
        out.push_str(&move_line(m));
        out.push('\n');
    }
    // Footer: the tally, the torn tail, and — unfiltered — the piece
    // worth opening next, so the path from "1 failed" to the text is
    // one more command.
    let mut states: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for row in &report.attempts {
        *states.entry(row.state.as_str()).or_default() += 1;
    }
    let tally: Vec<String> = states
        .iter()
        .map(|(state, count)| format!("{count} {state}"))
        .collect();
    match filter {
        Filter::All => out.push_str(&format!(
            "  {} attempt(s) ({}), {} move(s)\n",
            report.attempts.len(),
            if tally.is_empty() {
                "none".to_string()
            } else {
                tally.join(", ")
            },
            report.moves.len()
        )),
        _ => out.push_str(&format!(
            "  {} attempt(s) matched {}\n",
            report.matched, report.filter
        )),
    }
    if report.unreadable_lines > 0 {
        out.push_str(&format!(
            "  {} unreadable line(s) skipped — a torn tail from a killed run, most likely\n",
            report.unreadable_lines
        ));
    }
    if *filter == Filter::All
        && let Some(row) = piece_to_open(report)
    {
        out.push_str(&format!(
            "  to read a piece as the model saw it: taguru inspect {} --piece {}\n",
            shell_quote(&report.target),
            short_id(&row.piece_id)
        ));
    }
    out
}

/// The piece the footer suggests opening: the last attempt of a piece
/// that never reached `stop_valid` — a failed document's culprit, or
/// the piece still in flight when a run was killed — and, when every
/// piece got there, simply the last attempt.
fn piece_to_open(report: &AttemptsReport) -> Option<&AttemptRow> {
    let settled: std::collections::BTreeSet<&str> = report
        .attempts
        .iter()
        .filter(|row| row.state == "stop_valid")
        .map(|row| row.piece_id.as_str())
        .collect();
    report
        .attempts
        .iter()
        .rev()
        .find(|row| !settled.contains(row.piece_id.as_str()))
        .or(report.attempts.last())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn attempt(
        seq: u64,
        piece: &str,
        chunk: usize,
        state: &str,
        doc: &str,
        answer: Option<&str>,
    ) -> String {
        let user = format!("Document 'a.md', part {} of 2:\n\n{doc}", chunk + 1);
        json!({
            "kind": "attempt", "run_id": "r1", "attempt_seq": seq, "piece_id": piece,
            "source": "a.md", "chunk_index": chunk, "stage": "item", "attempt": 1,
            "max_attempts": 2, "state": state, "length_limited": state == "length_limited",
            "transport_retries": 0, "elapsed_seconds": 1.5, "requested_max_tokens": 4000,
            "finish_reason": if state == "length_limited" { "length" } else { "stop" },
            "input_tokens": 10, "output_tokens": 20,
            "messages": [{"role": "system", "system_sha256": "s"}, {"role": "user", "content": user}],
            "answer": answer,
            "parse_error": if state == "stop_malformed" { Some("not a JSON object: invalid escape at line 1 column 7") } else { None },
            "validation_issues": null, "removed_items": null
        })
        .to_string()
    }

    fn sample_log() -> String {
        [
            json!({"kind": "document", "run_id": "r1", "source": "a.md", "document_sha256": "d", "resumed": false}).to_string(),
            json!({"kind": "settings", "prompt_version": 4, "model": "stub", "questions_n": 0, "fact_budget": 0, "structured_output": "off", "max_output_tokens": 4000, "chunk_bytes": "", "chunk_context": "off", "lossy": false, "schema_digest": "", "candidates": "", "vocabulary_digest": ""}).to_string(),
            json!({"kind": "system", "sha256": "s", "bytes": 3, "content": "sys"}).to_string(),
            attempt(1, "aaaa1111bbbb2222cccc", 0, "length_limited", "[0] alpha\n\n[1] beta\n\n[2] gamma", Some("{\"associations\": [")),
            json!({"kind": "move", "move": "split", "run_id": "r1", "piece_id": "aaaa1111bbbb2222cccc", "chunk_index": 0, "reason": "the answer still ended at the output cap", "piece_bytes": 27, "split_cap": 14, "sub_pieces": 2}).to_string(),
            attempt(2, "dddd3333eeee4444ffff", 0, "stop_valid", "[0] alpha\n\n[1] beta", Some("{\"associations\": []}")),
            attempt(3, "9999888877776666", 0, "stop_malformed", "[2] gamma", Some("{\"associations\": [{\"subject\": \"g\\x\"}]}")),
            attempt(4, "5555444433332222", 1, "stop_valid", "[3] delta", Some("{\"associations\": []}")),
            "{not json".to_string(),
        ]
        .join("\n")
    }

    #[test]
    fn unfiltered_view_lists_every_attempt_with_its_paragraphs_and_the_moves() {
        let report = build_report("log", &sample_log(), &Filter::All);
        let text = render_text(&report, &Filter::All);
        assert!(
            text.starts_with("log: attempts log for 'a.md', run r1\n"),
            "{text}"
        );
        assert!(text.contains("  settings: model stub, prompt_version 4, structured_output off, max_output_tokens 4000, chunk_context off\n"), "{text}");
        assert!(text.contains("  #1  chunk 1/2  piece aaaa1111bbbb  paragraphs 0–2 (30 B)  attempt 1/2  length_limited (finish length)  cap 4000 tok  1.5 s  10→20 tok\n"), "{text}");
        assert!(text.contains("      ↳ split  2 sub-piece(s), cap 14 B — the answer still ended at the output cap\n"), "{text}");
        assert!(text.contains("  #3  chunk 1/2  piece 999988887777  paragraph 2 (9 B)  attempt 1/2  stop_malformed  cap 4000 tok  1.5 s  10→20 tok\n        not a JSON object: invalid escape at line 1 column 7\n"), "{text}");
        assert!(
            text.contains("  #4  chunk 2/2  piece 555544443333  paragraph 3 (9 B)"),
            "{text}"
        );
        assert!(
            text.contains(
                "  4 attempt(s) (1 length_limited, 1 stop_malformed, 2 stop_valid), 1 move(s)\n"
            ),
            "{text}"
        );
        assert!(text.contains("  1 unreadable line(s) skipped"), "{text}");
        // The pointer names the last non-valid piece — the one to open.
        assert!(
            text.ends_with(
                "  to read a piece as the model saw it: taguru inspect log --piece 999988887777\n"
            ),
            "{text}"
        );
        // No texts in the unfiltered view.
        assert!(!text.contains("--- piece text"), "{text}");
        assert_eq!(report.matched, 4);
        assert_eq!(report.unreadable_lines, 1);
    }

    #[test]
    fn piece_filter_prints_the_piece_text_and_answer_verbatim() {
        let filter = Filter::Piece("9999".to_string());
        let report = build_report("log", &sample_log(), &filter);
        assert_eq!(report.attempts.len(), 1);
        assert_eq!(report.matched, 1);
        assert!(
            report.moves.is_empty(),
            "the split belongs to another piece"
        );
        let text = render_text(&report, &filter);
        assert!(
            text.contains("--- piece text as sent (#3, 9 B) ---\n[2] gamma\n"),
            "{text}"
        );
        // One user turn is the piece alone — no corrective ask to show.
        assert!(!text.contains("--- corrective ask"), "{text}");
        assert!(text.contains("--- answer (#3, 38 B) ---\n{\"associations\": [{\"subject\": \"g\\x\"}]}\n--- end ---\n"), "{text}");
        assert!(text.ends_with("  1 attempt(s) matched piece 9999\n  1 unreadable line(s) skipped — a torn tail from a killed run, most likely\n"), "{text}");
        // The JSON rendering carries the same texts, only for the match.
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["attempts"][0]["piece_text"], "[2] gamma");
        assert_eq!(json["attempts"][0]["paragraph_first"], 2);
        assert_eq!(json["filter"], "piece 9999");
    }

    #[test]
    fn paragraph_filter_selects_every_piece_whose_range_covers_it() {
        let filter = Filter::Paragraph(1);
        let report = build_report("log", &sample_log(), &filter);
        let ids: Vec<&str> = report
            .attempts
            .iter()
            .map(|row| short_id(&row.piece_id))
            .collect();
        assert_eq!(ids, ["aaaa1111bbbb", "dddd3333eeee"]);
        assert_eq!(report.matched, 2);
        // The split of the first piece is kept — it is that piece's move.
        assert_eq!(report.moves.len(), 1);
        let text = render_text(&report, &filter);
        assert!(
            text.contains(
                "--- piece text as sent (#1, 30 B) ---\n[0] alpha\n\n[1] beta\n\n[2] gamma\n"
            ),
            "{text}"
        );
        assert!(
            text.contains(
                "--- answer (#1, 18 B) ---\n{\"associations\": [\n--- end ---\n      ↳ split"
            ),
            "{text}"
        );
        assert!(text.ends_with("  2 attempt(s) matched paragraph 1\n  1 unreadable line(s) skipped — a torn tail from a killed run, most likely\n"), "{text}");
        // Paragraph 5 is nowhere: an empty match is said, not failed.
        let none = Filter::Paragraph(5);
        let report = build_report("log", &sample_log(), &none);
        assert_eq!(report.matched, 0);
        assert!(render_text(&report, &none).contains("  0 attempt(s) matched paragraph 5\n"));
    }

    /// The rows the sample log does not reach: every move kind, a
    /// resumed document's second run, a settings line with a zero
    /// budget (hidden) and a rung (shown), an overflowing issue list,
    /// and a log with no torn tail.
    #[test]
    fn every_move_kind_a_second_run_and_the_overflow_lines_render() {
        let issues: Vec<String> = (0..4)
            .map(|i| format!("associations[{i}].weight: bad"))
            .collect();
        let many = json!({
            "kind": "attempt", "run_id": "r2", "attempt_seq": 2, "piece_id": "cafe000000000000",
            "source": "a.md", "chunk_index": 0, "stage": "item", "attempt": 1, "max_attempts": 2,
            "state": "stop_malformed", "length_limited": false, "transport_retries": 0,
            "elapsed_seconds": 1.0, "requested_max_tokens": null, "finish_reason": "stop",
            "input_tokens": null, "output_tokens": null,
            "messages": [{"role": "user", "content": "Document 'a.md':\n\n[0] a"}],
            "answer": "{}", "parse_error": null, "validation_issues": issues, "removed_items": null
        })
        .to_string();
        let log = [
            json!({"kind": "document", "run_id": "r1", "source": "a.md", "document_sha256": "d", "resumed": false}).to_string(),
            json!({"kind": "document", "run_id": "r2", "source": "a.md", "document_sha256": "d", "resumed": true}).to_string(),
            json!({"kind": "settings", "prompt_version": 4, "model": "stub", "max_output_tokens": 0, "rung": "json_schema", "lossy": true, "chunk_bytes": ""}).to_string(),
            attempt(1, "cafe000000000000", 0, "length_limited", "[0] a", Some("{")),
            json!({"kind": "move", "move": "escalate", "run_id": "r2", "piece_id": "cafe000000000000", "chunk_index": 0, "reason": "the answer ended at the output cap", "from_max_tokens": 4000, "to_max_tokens": 8000}).to_string(),
            json!({"kind": "move", "move": "demote", "run_id": "r2", "piece_id": "cafe000000000000", "chunk_index": 0, "reason": "the rung looped", "from_rung": "json_schema", "to_rung": "json_object"}).to_string(),
            json!({"kind": "move", "move": "runaway", "run_id": "r2", "piece_id": "cafe000000000000", "chunk_index": 0, "reason": "outgrew the piece", "piece_bytes": 50, "answer_bytes": 21745}).to_string(),
            many,
        ]
        .join("\n");
        let report = build_report("log", &log, &Filter::All);
        assert_eq!(report.runs, ["r1", "r2"]);
        let text = render_text(&report, &Filter::All);
        assert!(
            text.starts_with("log: attempts log for 'a.md', 2 run(s) — latest run r2 (resumed)\n"),
            "{text}"
        );
        assert!(
            text.contains("  settings: model stub, prompt_version 4, rung json_schema\n"),
            "{text}"
        );
        assert!(text.contains("      ↳ escalate  max_tokens 4000 → 8000 — the answer ended at the output cap\n      ↳ demote  json_schema → json_object — the rung looped\n      ↳ runaway  21745 B answered for a 50 B piece — outgrew the piece\n  #2  "), "{text}");
        assert!(
            text.contains("        associations[2].weight: bad\n        … and 1 more issue(s)\n"),
            "{text}"
        );
        assert!(!text.contains("associations[3].weight"), "{text}");
        assert!(!text.contains("unreadable line"), "{text}");
        assert!(
            text.contains("  2 attempt(s) (1 length_limited, 1 stop_malformed), 3 move(s)\n"),
            "{text}"
        );
    }

    #[test]
    fn a_corrective_round_shows_the_ask_and_a_repeated_piece_text_is_named_not_reprinted() {
        let base = attempt(
            1,
            "abcdef0123456789",
            0,
            "stop_malformed",
            "[0] alpha",
            Some("nope"),
        );
        let retry = json!({
            "kind": "attempt", "run_id": "r1", "attempt_seq": 2, "piece_id": "abcdef0123456789",
            "corrects": {"run_id": "r1", "attempt_seq": 1},
            "replayed_from": {"run_id": "r0", "attempt_seq": 3},
            "source": "a.md", "chunk_index": 0, "stage": "item", "attempt": 2, "max_attempts": 2,
            "state": "stop_valid", "length_limited": false, "transport_retries": 0,
            "elapsed_seconds": 0.5, "requested_max_tokens": null, "finish_reason": "stop",
            "input_tokens": null, "output_tokens": null,
            "messages": [
                {"role": "system", "system_sha256": "s"},
                {"role": "user", "content": "Document 'a.md':\n\n[0] alpha"},
                {"role": "assistant", "content": "nope"},
                {"role": "user", "content": "Your previous answer was not JSON. Answer again."}
            ],
            "answer": "{\"associations\": []}", "parse_error": null,
            "validation_issues": ["associations[0].weight: expected a number"], "removed_items": ["associations[1]: object empty"]
        })
        .to_string();
        let log = [base, retry].join("\n");
        let filter = Filter::Piece("abcdef".to_string());
        let report = build_report("log", &log, &filter);
        let text = render_text(&report, &filter);
        assert!(
            text.starts_with("log: attempts log (no document record)\n"),
            "{text}"
        );
        assert!(text.contains("  #2  chunk 1/1  piece abcdef012345  paragraph 0 (9 B)  attempt 2/2  stop_valid  0.5 s  corrects #1  replayed from #3\n        associations[0].weight: expected a number\n        1 item(s) removed (mechanical validation)\n        (piece text as sent: same as #1)\n--- corrective ask (#2, 48 B) ---\nYour previous answer was not JSON. Answer again.\n--- answer (#2, 20 B) ---"), "{text}");
    }

    #[test]
    fn cross_chunk_attempts_and_unlabeled_pieces_still_locate() {
        let cross = json!({
            "kind": "attempt", "run_id": "r1", "attempt_seq": 7, "piece_id": "feedfacefeedface",
            "source": "a.md", "chunk_index": 1, "stage": "cross_chunk", "attempt": 1, "max_attempts": 1,
            "state": "stop_valid", "length_limited": false, "transport_retries": 0,
            "elapsed_seconds": 2.0, "requested_max_tokens": 8000, "finish_reason": "stop",
            "input_tokens": 5, "output_tokens": 6,
            "messages": [{"role": "system", "system_sha256": "s"}, {"role": "user", "content": "Fix these aliases: …"}],
            "answer": null, "parse_error": null, "validation_issues": null, "removed_items": null
        })
        .to_string();
        let unlabeled = attempt(
            8,
            "0123456789abcdef",
            0,
            "timeout",
            "plain text without labels",
            None,
        );
        let log = [cross, unlabeled].join("\n");
        let text = render_text(&build_report("log", &log, &Filter::All), &Filter::All);
        assert!(text.contains("  #7  cross-chunk (chunk 2)  piece feedfacefeed  (22 B, unlabeled)  correction 1/1  stop_valid  cap 8000 tok  2.0 s  5→6 tok\n"), "{text}");
        assert!(
            text.contains(
                "  #8  chunk 1/2  piece 0123456789ab  (25 B, unlabeled)  attempt 1/2  timeout"
            ),
            "{text}"
        );
        assert!(text.ends_with("--piece 0123456789ab\n"), "{text}");
        // A piece that was length-limited and then settled is not the
        // one to open; with every piece settled, the last attempt is.
        let settled = [
            attempt(
                1,
                "aaaa000000000000",
                0,
                "length_limited",
                "[0] a",
                Some("{"),
            ),
            attempt(2, "aaaa000000000000", 0, "stop_valid", "[0] a", Some("{}")),
            attempt(3, "bbbb000000000000", 1, "stop_valid", "[1] b", Some("{}")),
        ]
        .join("\n");
        let text = render_text(&build_report("log", &settled, &Filter::All), &Filter::All);
        assert!(text.ends_with("--piece bbbb00000000\n"), "{text}");
        // An item attempt with no user turn at all (nothing was sent)
        // is still a chunk, not a cross-chunk row, and an empty
        // document part earns no byte suffix.
        let bare = json!({
            "kind": "attempt", "run_id": "r1", "attempt_seq": 9, "piece_id": "bare000000000000",
            "source": "a.md", "chunk_index": 2, "stage": "item", "attempt": 1, "max_attempts": 2,
            "state": "transport", "length_limited": false, "transport_retries": 4,
            "elapsed_seconds": 0.1, "requested_max_tokens": null, "finish_reason": null,
            "input_tokens": null, "output_tokens": null, "messages": [],
            "answer": null, "parse_error": null, "validation_issues": null, "removed_items": null
        })
        .to_string();
        let empty = attempt(10, "e0e0000000000000", 1, "stop_valid", "", Some("{}"));
        let text = render_text(
            &build_report(
                "log",
                &[bare.clone(), empty.clone()].join("\n"),
                &Filter::All,
            ),
            &Filter::All,
        );
        assert!(
            text.contains("  #9  chunk 3  piece bare00000000  attempt 1/2  transport  0.1 s\n"),
            "{text}"
        );
        assert!(
            text.contains(
                "  #10  chunk 2/2  piece e0e000000000  attempt 1/2  stop_valid  cap 4000 tok"
            ),
            "{text}"
        );
        assert!(!text.contains("0 B"), "{text}");
        // Filtered to that textless piece, its row is still the row
        // (nothing sent, nothing back — said as such).
        let filter = Filter::Piece("bare".to_string());
        let report = build_report("log", &[bare.clone(), empty.clone()].join("\n"), &filter);
        assert_eq!(report.matched, 1);
        assert_eq!(report.attempts.len(), 1);
        let text = render_text(&report, &filter);
        assert!(text.contains("  #9  chunk 3  piece bare00000000  attempt 1/2  transport  0.1 s\n--- answer (#9) --- none arrived\n--- end ---\n"), "{text}");
        assert!(
            text.contains("  1 attempt(s) matched piece bare\n"),
            "{text}"
        );
        // An odd piece id — multi-byte, or shorter than the prefix —
        // prints, never panics.
        assert_eq!(short_id("ééééééééééééééé"), "éééééééééééé");
        assert_eq!(short_id("short"), "short");
        assert_eq!(short_id(""), "");
        let filter = Filter::Piece("0123".to_string());
        let text = render_text(&build_report("log", &log, &filter), &filter);
        assert!(
            text.contains("--- answer (#8) --- none arrived\n"),
            "{text}"
        );
    }
}
